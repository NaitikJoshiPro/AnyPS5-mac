// Presentation members of the Metal device (work package D6, docs/mac/M2_DESIGN.md "DRAW
// results"): frames go to the CAMetalLayer of the window (libSceVideoOut's SDL Metal view) or, when
// the window has no Metal layer (SDL's offscreen driver, tests), to two offscreen textures of the
// window's drawable size: the headless mode, which runs every presentation step but the display.
// A device without a window is not presentable, as upstream.
//
// Bring-up path: the source image is the display buffer decoded on the CPU (ReadDisplayBuffer: the
// flush hook first waits for recorded GPU writes over it), or the pixels PresentPixels gets; it is
// uploaded into a BGRA8 texture and drawn aspect-fit (AspectFit.hpp) over an opaque black clear by
// a small scaler pipeline, recorded into the recorder's open batch, which the present submits.
// Slots: each present remembers its batch serial (and drawable); RetirePresents(keep) waits for the
// oldest beyond `keep` (FlipInFlight). Frame dumps (APS5_DUMP_FRAMES, _EVERY, APS5_DUMP_SCALE) are
// written from the decoded source pixels, as upstream's APS5_NO_GPU_DUMP path.
// Test hook: APS5_METAL_PRESENT_READBACK=1 copies every presented frame back (Metal::LastPresentedFrame).

#include "MetalDeviceState.h"
#include "MetalCommandStream.h"
#include "prx/libSceAgcDriver/Execution/include/AspectFit.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/PresentMode.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CATransaction.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace AgcDriver {

namespace {

[[noreturn]] void Fail(const std::string& what) {
    throw std::runtime_error("Metal presentation (work package D6): " + what);
}

constexpr const char* ScalerSource = R"(#include <metal_stdlib>
using namespace metal;
struct ScalerOut { float4 position [[position]]; float2 uv; };
// rect: left, top, right, bottom in NDC.
vertex ScalerOut aps5_scaler_vertex(uint id [[vertex_id]], constant float4& rect [[buffer(0)]]) {
    const float2 corner = float2(float(id & 1u), float(id >> 1u));
    ScalerOut out;
    out.position = float4(mix(rect.x, rect.z, corner.x), mix(rect.y, rect.w, corner.y), 0.0f, 1.0f);
    out.uv = corner;
    return out;
}
fragment float4 aps5_scaler_fragment(ScalerOut in [[stage_in]], texture2d<float> source [[texture(0)]], sampler filter [[sampler(0)]]) {
    return float4(source.sample(filter, in.uv).rgb, 1.0f);
}
)";

std::uint32_t DumpScale() {
    static const std::uint32_t scale = [] {
        const char* value = std::getenv("APS5_DUMP_SCALE");
        if (value == nullptr) return std::uint32_t{4};
        std::uint32_t parsed = 0;
        const auto end = value + std::strlen(value);
        const auto result = std::from_chars(value, end, parsed);
        if (result.ec != std::errc{} || result.ptr != end || parsed == 0 || parsed > 16384) Fail("APS5_DUMP_SCALE must be between 1 and 16384");
        return parsed;
    }();
    return scale;
}

struct FrameDumps {
    int limit;
    int every;
    int dumped = 0;
    std::uint64_t presents = 0;
};

FrameDumps& Dumps() {
    static FrameDumps dumps {[] {
        const char* value = std::getenv("APS5_DUMP_FRAMES");
        return value ? std::atoi(value) : 0;
    }(), [] {
        const char* value = std::getenv("APS5_DUMP_FRAMES_EVERY");
        return value ? std::max(std::atoi(value), 1) : 1;
    }()};
    return dumps;
}

// frame_<index>.bmp: every scale-th pixel of a BGRA8 image, 32-bit top-down (upstream's format).
void WriteFrameBmp(int index, std::uint32_t fullWidth, std::uint32_t fullHeight, std::span<const std::byte> full, std::uint32_t scale) {
    if (scale == 0u || full.size() != static_cast<std::size_t>(fullWidth) * fullHeight * 4u) Fail("invalid frame dump pixels");
    const std::uint32_t width = (fullWidth + scale - 1u) / scale;
    const std::uint32_t height = (fullHeight + scale - 1u) / scale;
    std::vector<std::byte> pixels(static_cast<std::size_t>(width) * height * 4u);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) std::memcpy(pixels.data() + (static_cast<std::size_t>(y) * width + x) * 4u, full.data() + (static_cast<std::size_t>(y) * scale * fullWidth + static_cast<std::size_t>(x) * scale) * 4u, 4u);
    }
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%03d.bmp", index);
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(name, std::ios::binary);
    const std::uint32_t imageBytes = width * height * 4u;
    const std::uint32_t header[13] = {0, 0, 54, 40, width, static_cast<std::uint32_t>(-static_cast<std::int32_t>(height)), 1u | (32u << 16u), 0, imageBytes, 2835, 2835, 0, 0};
    const std::uint16_t magic = 0x4d42;
    const std::uint32_t fileBytes = 54u + imageBytes;
    file.write(reinterpret_cast<const char*>(&magic), 2);
    file.write(reinterpret_cast<const char*>(&fileBytes), 4);
    file.write(reinterpret_cast<const char*>(header + 1), 48);
    file.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

// Every scale-th pixel of a BGRA8 image (the dump's pixels, upstream's downscale).
std::vector<std::byte> DownscaleFrame(std::uint32_t fullWidth, std::uint32_t fullHeight, std::span<const std::byte> full, std::uint32_t scale, std::uint32_t& width, std::uint32_t& height) {
    if (scale == 0u || full.size() != static_cast<std::size_t>(fullWidth) * fullHeight * 4u) Fail("invalid frame dump pixels");
    width = (fullWidth + scale - 1u) / scale;
    height = (fullHeight + scale - 1u) / scale;
    std::vector<std::byte> pixels(static_cast<std::size_t>(width) * height * 4u);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) std::memcpy(pixels.data() + (static_cast<std::size_t>(y) * width + x) * 4u, full.data() + (static_cast<std::size_t>(y) * scale * fullWidth + static_cast<std::size_t>(x) * scale) * 4u, 4u);
    }
    return pixels;
}

// Upstream's FrameDumpWriter (VulkanDevice.cpp): frame dumps are written by a worker, so a slow
// file (or a pipe nobody reads yet: video_out_frame_dump_queue) never stalls a flip. Owned by the
// presenter; device teardown joins it after the queued dumps are written.
class FrameDumpWriter {
public:
    ~FrameDumpWriter() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        changed.notify_one();
        if (worker.joinable()) worker.join();
    }

    void Enqueue(int index, std::uint32_t width, std::uint32_t height, std::vector<std::byte> pixels) {
        std::lock_guard lock(mutex);
        if (stopping) Fail("frame dump writer has stopped");
        if (pixels.size() > MaxBytes - queuedBytes) Fail("frame dump queue exceeded 64 MiB");
        if (!worker.joinable()) worker = std::thread([this] { run(); });
        if (std::getenv("APS5_METAL_TRACE") != nullptr) std::fprintf(stderr, "[metal] frame_%03d.bmp queued (%ux%u)\n", index, width, height);
        const auto bytes = pixels.size();
        pending.push_back({index, width, height, std::move(pixels)});
        queuedBytes += bytes;
        changed.notify_one();
    }

private:
    struct Frame {
        int index;
        std::uint32_t width;
        std::uint32_t height;
        std::vector<std::byte> pixels;
    };

    void run() {
        try {
            for (;;) {
                std::deque<Frame> batch;
                {
                    std::unique_lock lock(mutex);
                    changed.wait(lock, [this] { return stopping || !pending.empty(); });
                    if (pending.empty() && stopping) return;
                    batch.swap(pending);
                }
                for (const auto& frame : batch) {
                    WriteFrameBmp(frame.index, frame.width, frame.height, frame.pixels, 1u);
                    if (std::getenv("APS5_METAL_TRACE") != nullptr) std::fprintf(stderr, "[metal] frame_%03d.bmp written (%ux%u)\n", frame.index, frame.width, frame.height);
                    std::lock_guard lock(mutex);
                    queuedBytes -= frame.pixels.size();
                }
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[gpu] frame dump writer failed: %s\n", error.what());
            std::terminate();
        }
    }

    static constexpr std::size_t MaxBytes = 64u * 1024u * 1024u;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Frame> pending;
    std::thread worker;
    std::size_t queuedBytes = 0;
    bool stopping = false;
};

bool Readback() {
    static const bool readback = std::getenv("APS5_METAL_PRESENT_READBACK") != nullptr;
    return readback;
}

// APS5_METAL_TRACE: presenter attach, every present and every drawable handed to the display.
bool PresentTrace() {
    static const bool trace = std::getenv("APS5_METAL_TRACE") != nullptr;
    return trace;
}

std::mutex& FrameMutex() {
    static std::mutex mutex;
    return mutex;
}

Metal::PresentedFrame& LastFrame() {
    static Metal::PresentedFrame frame;
    return frame;
}

VulkanDevice::PresentStatistics& Counters() {
    static VulkanDevice::PresentStatistics counters {};
    return counters;
}

}

namespace Metal {

PresentedFrame LastPresentedFrame() {
    std::lock_guard lock(FrameMutex());
    return LastFrame();
}

// The device's presentation state (made by AttachPresentation, destroyed by Teardown once the
// recorder is idle; it removes its residency sets from the queue).
struct Presenter {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    void* window = nullptr;
    void (*getDrawableSize)(void*, std::uint32_t*, std::uint32_t*) = nullptr;
    CAMetalLayer* layer = nil;
    id<MTLResidencySet> residency = nil;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // Headless targets (no layer), used in turn.
    std::array<id<MTLTexture>, 2> offscreen {};
    std::uint32_t nextOffscreen = 0;
    // The acquired target.
    id<CAMetalDrawable> drawable = nil;
    id<MTLTexture> target = nil;
    bool queuePending = false;
    // The scaler.
    id<MTLRenderPipelineState> scaler = nil;
    id<MTLSamplerState> filter = nil;
    id<MTL4ArgumentTable> arguments = nil;
    id<MTLTexture> source = nil;
    struct Slot {
        std::uint64_t serial;
        id<CAMetalDrawable> drawable;
        id<MTLBuffer> readback;
        std::uint32_t width;
        std::uint32_t height;
    };
    std::deque<Slot> slots;
    // Frame dumps (APS5_DUMP_FRAMES), written off the presenting thread.
    FrameDumpWriter dumpWriter;

    ~Presenter() {
        if (queue != nil && residency != nil) [queue removeResidencySet:residency];
        if (queue != nil && layer != nil) [queue removeResidencySet:layer.residencySet];
    }

    void Resident(id<MTLAllocation> allocation) {
        [residency addAllocation:allocation];
        [residency commit];
    }

    void MakeOffscreen() {
        for (auto& texture : offscreen) {
            if (texture != nil) [residency removeAllocation:texture];
            texture = nil;
        }
        [residency commit];
        if (layer != nil || width == 0u || height == 0u) return;
        for (auto& texture : offscreen) {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:width height:height mipmapped:NO];
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            texture = [device newTextureWithDescriptor:descriptor];
            if (texture == nil) Fail("cannot create the headless presentation targets");
            [residency addAllocation:texture];
        }
        [residency commit];
    }
};

}

void VulkanDevice::State::AttachPresentation(const PresentationWindow& window) {
    auto presenter = std::make_shared<Metal::Presenter>();
    presenter->device = device;
    presenter->queue = queue;
    presenter->window = window.context;
    presenter->getDrawableSize = window.getDrawableSize;
    MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
    descriptor.initialCapacity = 8;
    NSError* error = nil;
    presenter->residency = [device newResidencySetWithDescriptor:descriptor error:&error];
    if (presenter->residency == nil) Fail("cannot create the presentation residency set: " + Metal::Describe(error));
    [queue addResidencySet:presenter->residency];
    // The Metal view first: the drawable size is the view's (pixels, HiDPI) once it exists.
#ifdef __APPLE__
    if (window.metalLayer != nullptr) presenter->layer = (__bridge CAMetalLayer*)window.metalLayer(window.context);
#endif
    std::uint32_t width = window.width;
    std::uint32_t height = window.height;
    if (window.getDrawableSize != nullptr) window.getDrawableSize(window.context, &width, &height);
    presenter->width = width;
    presenter->height = height;
    // ANYPS5_PRESENT_MODE (upstream #4106): a CAMetalLayer presents FIFO (display sync) or
    // immediately (no display sync); Core Animation has no mailbox or latest-ready queue.
    static constexpr VkPresentModeKHR layerModes[] = {VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR};
    const auto presentMode = presenter->layer != nil ? SelectPresentMode(std::getenv("ANYPS5_PRESENT_MODE"), layerModes) : VK_PRESENT_MODE_FIFO_KHR;
    if (presenter->layer != nil) {
        // Layer properties change off the main thread inside an explicit transaction (an implicit
        // one on a thread without a run loop is never committed).
        [CATransaction begin];
        [CATransaction setDisableActions:YES];
        presenter->layer.device = device;
        presenter->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        presenter->layer.framebufferOnly = YES;
        presenter->layer.maximumDrawableCount = 3;
        presenter->layer.displaySyncEnabled = presentMode == VK_PRESENT_MODE_FIFO_KHR ? YES : NO;
        [CATransaction commit];
        // Metal 4: drawables are resident through the layer's own residency set on the queue.
        [queue addResidencySet:presenter->layer.residencySet];
    }
    presenter->MakeOffscreen();
    if (PresentTrace()) std::fprintf(stderr, "[metal] present attach: %s, drawable %ux%u%s\n", presenter->layer != nil ? "CAMetalLayer" : "headless (offscreen target)", width, height, presenter->layer != nil ? (presenter->layer.displaySyncEnabled ? ", display sync" : ", no display sync") : "");

    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_2;
    id<MTLLibrary> library = [device newLibraryWithSource:@(ScalerSource) options:options error:&error];
    if (library == nil) Fail("the scaler does not compile: " + Metal::Describe(error));
    MTLRenderPipelineDescriptor* pipeline = [MTLRenderPipelineDescriptor new];
    pipeline.vertexFunction = [library newFunctionWithName:@"aps5_scaler_vertex"];
    pipeline.fragmentFunction = [library newFunctionWithName:@"aps5_scaler_fragment"];
    pipeline.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    presenter->scaler = [device newRenderPipelineStateWithDescriptor:pipeline error:&error];
    if (presenter->scaler == nil) Fail("the scaler has no pipeline: " + Metal::Describe(error));
    MTLSamplerDescriptor* sampler = [MTLSamplerDescriptor new];
    sampler.minFilter = MTLSamplerMinMagFilterLinear;
    sampler.magFilter = MTLSamplerMinMagFilterLinear;
    sampler.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler.supportArgumentBuffers = YES;
    presenter->filter = [device newSamplerStateWithDescriptor:sampler];
    MTL4ArgumentTableDescriptor* table = [MTL4ArgumentTableDescriptor new];
    table.maxBufferBindCount = 1;
    table.maxTextureBindCount = 1;
    table.maxSamplerStateBindCount = 1;
    presenter->arguments = [device newArgumentTableWithDescriptor:table error:&error];
    if (presenter->filter == nil || presenter->arguments == nil) Fail("cannot create the scaler's sampler or argument table: " + Metal::Describe(error));
    this->presenter = std::move(presenter);
}

void* VulkanDevice::Window() const {
    return state->presenter != nullptr ? state->presenter->window : nullptr;
}

void VulkanDevice::Resize(std::uint32_t width, std::uint32_t height) {
    auto* presenter = state->presenter.get();
    if (presenter == nullptr) return;
    if (width == 0u || height == 0u) {
        presenter->width = 0;
        presenter->height = 0;
        return;
    }
    // The arguments are the window's drawable size (Driver::Present asks for it every frame).
    if (width == presenter->width && height == presenter->height && (presenter->layer != nil || presenter->offscreen[0] != nil)) return;
    RetirePresents(0);
    presenter->drawable = nil;
    presenter->target = nil;
    presenter->width = width;
    presenter->height = height;
    // SDL's Metal view owns the layer's drawableSize and contentsScale (it updates them on resizes
    // and backing-scale changes); the presenter only follows the size.
    presenter->MakeOffscreen();
}

bool VulkanDevice::Presentable() const {
    return state->presenter != nullptr && state->presenter->width != 0u && state->presenter->height != 0u;
}

std::size_t VulkanDevice::FlipInFlight() {
    static const std::size_t inFlight = [] {
        if (std::getenv("APS5_SYNC_FLIP") != nullptr) return std::size_t{0};
        const char* value = std::getenv("APS5_FLIP_INFLIGHT");
        if (value == nullptr) return std::size_t{1};
        return static_cast<std::size_t>(std::clamp(std::atoi(value), 0, 2));
    }();
    return inFlight;
}

double VulkanDevice::RetirePresents(std::size_t keepInFlight) {
    auto* presenter = state->presenter.get();
    if (presenter == nullptr || state->recorder == nullptr) return 0.0;
    const auto start = std::chrono::steady_clock::now();
    while (presenter->slots.size() > keepInFlight) {
        auto slot = std::move(presenter->slots.front());
        presenter->slots.pop_front();
        state->recorder->WaitSerial(slot.serial);
        if (slot.readback != nil) {
            std::lock_guard lock(FrameMutex());
            auto& frame = LastFrame();
            frame.width = slot.width;
            frame.height = slot.height;
            frame.pixels.resize(static_cast<std::size_t>(slot.width) * slot.height * 4u);
            std::memcpy(frame.pixels.data(), slot.readback.contents, frame.pixels.size());
            ++frame.frames;
        }
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

bool VulkanDevice::PresentWaitsForSlots(const DisplayBuffer*) const {
    return false;
}

bool VulkanDevice::AcquireImage() {
    auto* presenter = state->presenter.get();
    if (presenter == nullptr || !Presentable()) return false;
    if (presenter->target != nil) return true;
    @autoreleasepool {
        if (presenter->layer != nil) {
            id<CAMetalDrawable> drawable = [presenter->layer nextDrawable];
            if (drawable == nil) return false;
            presenter->drawable = drawable;
            presenter->target = drawable.texture;
        } else {
            presenter->target = presenter->offscreen[presenter->nextOffscreen++ % presenter->offscreen.size()];
        }
    }
    return presenter->target != nil;
}

bool VulkanDevice::present(std::uint32_t width, std::uint32_t height, bool opaque, std::span<const std::byte> pixels) {
    auto* presenter = state->presenter.get();
    if (presenter == nullptr || state->recorder == nullptr || !Presentable()) return false;
    // A slot in flight beyond the allowed count is retired first (its drawable and readback).
    RetirePresents(FlipInFlight());
    if (!AcquireImage()) return false;
    auto& recorder = *state->recorder;
    @autoreleasepool {
        const bool hasSource = !pixels.empty();
        if (hasSource) {
            if (width == 0u || height == 0u || width > 16384u || height > 16384u || pixels.size() != static_cast<std::size_t>(width) * height * 4u) Fail("invalid source image for the present");
            if (presenter->source == nil || presenter->source.width != width || presenter->source.height != height) {
                if (presenter->source != nil) {
                    // Earlier presents read the old source: they finish first.
                    RetirePresents(0);
                    [presenter->residency removeAllocation:presenter->source];
                }
                MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:width height:height mipmapped:NO];
                descriptor.storageMode = MTLStorageModePrivate;
                descriptor.usage = MTLTextureUsageShaderRead;
                presenter->source = [state->device newTextureWithDescriptor:descriptor];
                if (presenter->source == nil) Fail("cannot create the present source texture");
                presenter->Resident(presenter->source);
            }
        }
        auto& stream = recorder.Stream();
        if (hasSource) {
            auto upload = stream.Write(pixels, 256u);
            id<MTL4ComputeCommandEncoder> copy = stream.Compute();
            [copy copyFromBuffer:upload.buffer sourceOffset:upload.offset sourceBytesPerRow:static_cast<NSUInteger>(width) * 4u sourceBytesPerImage:pixels.size() sourceSize:MTLSizeMake(width, height, 1) toTexture:presenter->source destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            stream.CountCommand();
        }
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = presenter->target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, opaque ? 1.0 : 0.0);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTL4RenderCommandEncoder> encoder = stream.BeginRender(pass);
        if (hasSource) {
            const auto targetWidth = static_cast<std::uint32_t>(presenter->target.width);
            const auto targetHeight = static_cast<std::uint32_t>(presenter->target.height);
            const auto rect = ComputeContainRect_nid_postfix(width, height, targetWidth, targetHeight);
            const std::array<float, 4> ndc {
                2.0f * static_cast<float>(rect.x) / static_cast<float>(targetWidth) - 1.0f,
                1.0f - 2.0f * static_cast<float>(rect.y) / static_cast<float>(targetHeight),
                2.0f * static_cast<float>(rect.x + static_cast<std::int32_t>(rect.width)) / static_cast<float>(targetWidth) - 1.0f,
                1.0f - 2.0f * static_cast<float>(rect.y + static_cast<std::int32_t>(rect.height)) / static_cast<float>(targetHeight)};
            const auto rectUpload = stream.Write(std::as_bytes(std::span(ndc)), 16u);
            [presenter->arguments setAddress:rectUpload.address atIndex:0];
            [presenter->arguments setTexture:presenter->source.gpuResourceID atIndex:0];
            [presenter->arguments setSamplerState:presenter->filter.gpuResourceID atIndex:0];
            [encoder setRenderPipelineState:presenter->scaler];
            [encoder setArgumentTable:presenter->arguments atStages:MTLRenderStageVertex | MTLRenderStageFragment];
            [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<double>(targetWidth), static_cast<double>(targetHeight), 0.0, 1.0}];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
            stream.CountCommand();
        }
        stream.EndEncoder();
        id<MTLBuffer> readback = nil;
        if (Readback() && presenter->layer == nil) {
            const auto bytes = static_cast<std::size_t>(presenter->target.width) * presenter->target.height * 4u;
            readback = [state->device newBufferWithLength:bytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            if (readback == nil) Fail("cannot allocate the present readback");
            presenter->Resident(readback);
            stream.Barrier();
            id<MTL4ComputeCommandEncoder> copy = stream.Compute();
            [copy copyFromTexture:presenter->target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(presenter->target.width, presenter->target.height, 1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:presenter->target.width * 4u destinationBytesPerImage:bytes];
            stream.CountCommand();
            auto* residency = presenter;
            recorder.OnComplete([residency, readback] {
                // The slot keeps the buffer; it only leaves the residency set here.
                [residency->residency removeAllocation:readback];
                [residency->residency commit];
            });
        }
        // Metal 4 drawables: the queue waits for the drawable before the batch and signals it after.
        if (presenter->drawable != nil) [state->queue waitForDrawable:presenter->drawable];
        const auto serial = recorder.SubmitAndEpoch();
        if (presenter->drawable != nil) [state->queue signalDrawable:presenter->drawable];
        if (PresentTrace()) std::fprintf(stderr, "[metal] present %llu: source %ux%u%s into %lux%lu, %s, batch %llu\n", static_cast<unsigned long long>(Counters().presents + 1u), width, height, hasSource ? "" : " (clear)", static_cast<unsigned long>(presenter->target.width), static_cast<unsigned long>(presenter->target.height), presenter->drawable != nil ? "drawable" : "offscreen", static_cast<unsigned long long>(serial));
        presenter->slots.push_back({serial, presenter->drawable, readback, static_cast<std::uint32_t>(presenter->target.width), static_cast<std::uint32_t>(presenter->target.height)});
        presenter->queuePending = presenter->drawable != nil;
        presenter->target = nil;
        ++Counters().presents;
    }
    return true;
}

bool VulkanDevice::PresentClear(std::uint32_t width, std::uint32_t height, bool opaque) {
    return present(width, height, opaque, {});
}

void VulkanDevice::PresentPixels(std::uint32_t width, std::uint32_t height, std::span<const std::byte> pixels) {
    if (width == 0u || height == 0u || width > 16384u || height > 16384u) Fail("invalid display image extent");
    if (pixels.size() != static_cast<std::uint64_t>(width) * height * 4u) Fail("invalid display pixel buffer size");
    auto& dumps = Dumps();
    if (dumps.dumped < dumps.limit) {
        if (auto* presenter = state->presenter.get(); presenter != nullptr) presenter->dumpWriter.Enqueue(dumps.dumped++, width, height, {pixels.begin(), pixels.end()});
        else WriteFrameBmp(dumps.dumped++, width, height, pixels, 1u);
    }
    // Tests and tools present pixels synchronously; the game path splits the steps (Driver::Present).
    if (!present(width, height, true, pixels)) return;
    FinishPresent();
    QueuePresent();
}

bool VulkanDevice::PresentDisplayBuffer(const DisplayBuffer& buffer) {
    if (!Presentable()) return false;
    // Upstream: APS5_STORE_AT_FLIP stores every pending resident image at each flip; otherwise the
    // images an APS5 storage dump selects are stored (a70ec3946).
    static const bool storeAtFlip = Graphics::StorageTexture::StoreAtFlipRequested(std::getenv("APS5_STORE_AT_FLIP"));
    if (storeAtFlip) Graphics::StorageTexture::FlushAllPending("flip");
    else Graphics::StorageTexture::FlushDumpSelected("flip");
    // The bring-up source: the display buffer decoded on the CPU (linear or tiled, the formats
    // DisplayBuffer.cpp decodes); the read waits for recorded GPU writes over it.
    const auto pixels = ReadDisplayBuffer(buffer);
    auto& dumps = Dumps();
    int dumpIndex = -1;
    if (dumps.dumped < dumps.limit && ++dumps.presents % static_cast<std::uint64_t>(dumps.every) == 0u) dumpIndex = dumps.dumped++;
    if (!present(buffer.width, buffer.height, true, pixels)) {
        // A dropped frame keeps the dump numbering contiguous.
        if (dumpIndex >= 0) --dumps.dumped;
        return false;
    }
    if (dumpIndex >= 0) {
        std::uint32_t dumpWidth = 0;
        std::uint32_t dumpHeight = 0;
        auto scaled = DownscaleFrame(buffer.width, buffer.height, pixels, DumpScale(), dumpWidth, dumpHeight);
        state->presenter->dumpWriter.Enqueue(dumpIndex, dumpWidth, dumpHeight, std::move(scaled));
    }
    return true;
}

double VulkanDevice::FinishPresent() {
    return RetirePresents(0);
}

void VulkanDevice::QueuePresent() {
    auto* presenter = state->presenter.get();
    if (presenter == nullptr || !presenter->queuePending) return;
    presenter->queuePending = false;
    if (presenter->drawable != nil) {
        [presenter->drawable present];
        if (PresentTrace()) std::fprintf(stderr, "[metal] present queued to the display (drawable %lux%lu)\n", static_cast<unsigned long>(presenter->drawable.texture.width), static_cast<unsigned long>(presenter->drawable.texture.height));
        presenter->drawable = nil;
    }
}

void VulkanDevice::FlipBatches(std::uint64_t& submissions, std::uint64_t& unsignaled) const {
    submissions = state->recorder != nullptr ? state->recorder->Submissions() : 0u;
    unsignaled = state->recorder != nullptr ? state->recorder->InFlightBatches() : 0u;
}

VulkanDevice::PresentStatistics VulkanDevice::PresentCounts() {
    return Counters();
}

}
