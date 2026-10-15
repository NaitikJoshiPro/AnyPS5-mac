// Presentation members of the Metal device (work package D6, docs/mac/M2_DESIGN.md "DRAW
// results"): frames go to the CAMetalLayer of the window (libSceVideoOut's SDL Metal view) or, when
// the window has no Metal layer (SDL's offscreen driver, tests), to two offscreen textures of the
// window's drawable size: the headless mode, which runs every presentation step but the display.
// A device without a window is not presentable, as upstream.
//
// The source image is the display buffer decoded on the GPU (D66, below) or on the CPU
// (ReadDisplayBuffer: the flush hook first waits for recorded GPU writes over it), or the pixels
// PresentPixels gets; it is copied into a BGRA8 texture and drawn aspect-fit (AspectFit.hpp) over an opaque black clear by
// a small scaler pipeline, recorded into the recorder's open batch, which the present submits.
// Slots: each present remembers its batch serial (and drawable); RetirePresents(keep) waits for the
// oldest beyond `keep` (FlipInFlight). Frame dumps (APS5_DUMP_FRAMES, _EVERY, APS5_DUMP_SCALE) are
// written from the decoded source pixels, as upstream's APS5_NO_GPU_DUMP path.
// Test hook: APS5_METAL_PRESENT_READBACK=1 copies every presented frame back (Metal::LastPresentedFrame).
// Upscaling (D61): when the aspect-fit rectangle is larger than the source in both dimensions, a
// MetalFX spatial scaler (MTL4FXSpatialScaler, perceptual color) upscales the source to the
// rectangle's size and the scaler pipeline copies that 1:1 into the target; downscales and equal
// sizes keep the bilinear scaler. APS5_METALFX=0 always uses the bilinear scaler. Headless targets
// take the window's drawable size, or APS5_PRESENT_SIZE=<w>x<h> (measurements of a 16:9 output).
// GPU decode (D66): a display buffer inside a host import is detiled and converted on the GPU by
// aps5_display_decode, reading the guest memory in place, in the batch the present submits. The
// flush hook records what must land first (queued labels, storage image stores) and waits for
// nothing: the recorded GPU writes precede the decode in queue order, and the flip is reported
// done only once the present's batch finished (Driver::Present), so the guest cannot overwrite the
// buffer under the decode. Frame dumps, memory outside every import, and a pending CPU write-back
// keep the CPU decode. APS5_PRESENT_GPU_DECODE=0 always decodes on the CPU.

#include "MetalDeviceState.h"
#include "MetalCommandStream.h"
#include "prx/libSceAgcDriver/Execution/include/AspectFit.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HalfFloatScanout.hpp"
#include "prx/libSceAgcDriver/Execution/include/PresentMode.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "MetalResidency.h"
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CATransaction.h>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
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
// The display buffer decode (D66), DisplayBuffer.cpp's DecodeDisplayBuffer on the GPU: one thread
// per pixel, BGRA8 out. mode: 1 tiled (64 KiB R_X), 2 R8G8B8A8, 4 10-bit, 8 16-bit float, 16 red low.
struct DecodeParameters {
    uint width;
    uint height;
    uint pitch;
    uint mode;
    uint blocksPerRow;
    uint unused;
    device const uint* source;
    device uint* destination;
    device const uint* thresholds;
};
static uint scanoutCode(device const uint* thresholds, uint bits, uint base) {
    if (bits > 0x7c00u) return 0u;
    uint code = 0u;
    for (uint step = 128u; step != 0u; step >>= 1u) {
        if (code + step <= 255u && thresholds[base + code + step - 1u] <= bits) code += step;
    }
    return code;
}
kernel void aps5_display_decode(constant DecodeParameters& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
    const uint x = id.x;
    const uint y = id.y;
    if (x >= p.width || y >= p.height) return;
    const bool tiled = (p.mode & 1u) != 0u;
    uint pixel;
    if ((p.mode & 8u) != 0u) {
        uint offset;
        if (tiled) {
            offset = ((y / 64u) * p.blocksPerRow + x / 128u) * 65536u + (((x << 3u) & 0x0008u) ^ ((x << 4u) & 0x0460u) ^ ((x << 5u) & 0x0300u) ^ ((x << 6u) & 0x0800u) ^ ((x << 9u) & 0x8000u) ^ ((x << 10u) & 0x2000u)
                ^ ((y << 4u) & 0x0010u) ^ ((y << 6u) & 0x0080u) ^ ((y << 5u) & 0x0f00u) ^ ((y << 10u) & 0x5000u));
        } else {
            offset = (y * p.pitch + x) * 8u;
        }
        const uint first = p.source[offset / 4u];
        const uint second = p.source[offset / 4u + 1u];
        const bool redLow = (p.mode & 16u) != 0u;
        const uint red = redLow ? (second & 0xffffu) : (first & 0xffffu);
        const uint blue = redLow ? (first & 0xffffu) : (second & 0xffffu);
        pixel = scanoutCode(p.thresholds, red, 0u) | (scanoutCode(p.thresholds, first >> 16u, 0u) << 8u) | (scanoutCode(p.thresholds, blue, 0u) << 16u) | (scanoutCode(p.thresholds, second >> 16u, 255u) << 24u);
    } else {
        uint offset;
        if (tiled) {
            offset = ((y / 128u) * p.blocksPerRow + x / 128u) * 65536u + (((y << 4u) & 0x0070u) ^ ((y << 5u) & 0x0f00u) ^ ((y << 9u) & 0x1000u) ^ ((y << 8u) & 0x4000u)
                ^ ((x << 2u) & 0x000cu) ^ ((x << 5u) & 0x0380u) ^ ((x << 4u) & 0x0400u) ^ ((x << 6u) & 0x0800u) ^ ((x << 9u) & 0xa000u));
        } else {
            offset = (y * p.pitch + x) * 4u;
        }
        const uint word = p.source[offset / 4u];
        const bool rgba = (p.mode & 2u) != 0u;
        if ((p.mode & 4u) != 0u) {
            const uint r = (word >> 2u) & 0xffu;
            const uint g = (word >> 12u) & 0xffu;
            const uint b = (word >> 22u) & 0xffu;
            pixel = (rgba ? b : r) | (g << 8u) | ((rgba ? r : b) << 16u) | 0xff000000u;
        } else {
            pixel = rgba ? ((word & 0xff00ff00u) | ((word & 0xffu) << 16u) | ((word >> 16u) & 0xffu)) : word;
        }
    }
    p.destination[y * p.width + x] = pixel;
}
)";

struct DecodeParameters {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pitch;
    std::uint32_t mode;
    std::uint32_t blocksPerRow;
    std::uint32_t unused;
    std::uint64_t source;
    std::uint64_t destination;
    std::uint64_t thresholds;
};

// APS5_PRESENT_GPU_DECODE=0: every display buffer is decoded on the CPU (ReadDisplayBuffer).
bool GpuDecodeRequested() {
    static const bool requested = [] {
        const char* value = std::getenv("APS5_PRESENT_GPU_DECODE");
        if (value == nullptr || *value == '\0' || std::strcmp(value, "1") == 0) return true;
        if (std::strcmp(value, "0") == 0) return false;
        Fail(std::string("APS5_PRESENT_GPU_DECODE must be 0 or 1, not '") + value + "'");
    }();
    return requested;
}

// The display buffer the next present() decodes on the GPU (set by PresentDisplayBuffer around the call).
struct GpuSource {
    const DisplayBuffer* buffer;
    id<MTLBuffer> import;
    std::uint64_t offset;
};
thread_local const GpuSource* pendingGpuSource = nullptr;

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

// APS5_METALFX: 0 turns the MetalFX upscale off (the bilinear scaler presents every frame).
bool MetalFxRequested() {
    static const bool requested = [] {
        const char* value = std::getenv("APS5_METALFX");
        if (value == nullptr || *value == '\0') return true;
        if (std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        Fail(std::string("APS5_METALFX must be 0 or 1, not '") + value + "'");
    }();
    return requested;
}

// APS5_PRESENT_SIZE=<width>x<height>: the size of the headless targets (default: the window's drawable size).
bool HeadlessSizeOverride(std::uint32_t& width, std::uint32_t& height) {
    static const std::pair<std::uint32_t, std::uint32_t> size = [] {
        const char* value = std::getenv("APS5_PRESENT_SIZE");
        if (value == nullptr || *value == '\0') return std::pair<std::uint32_t, std::uint32_t>{0u, 0u};
        const auto end = value + std::strlen(value);
        std::uint32_t parsedWidth = 0;
        std::uint32_t parsedHeight = 0;
        const auto first = std::from_chars(value, end, parsedWidth);
        const auto second = first.ec == std::errc{} && first.ptr != end && *first.ptr == 'x' ? std::from_chars(first.ptr + 1, end, parsedHeight) : std::from_chars_result{first.ptr, std::errc::invalid_argument};
        if (second.ec != std::errc{} || second.ptr != end || parsedWidth == 0u || parsedHeight == 0u || parsedWidth > 16384u || parsedHeight > 16384u) Fail(std::string("APS5_PRESENT_SIZE must be <width>x<height> (1-16384), not '") + value + "'");
        return std::pair{parsedWidth, parsedHeight};
    }();
    if (size.first == 0u) return false;
    width = size.first;
    height = size.second;
    return true;
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

std::atomic<std::uint64_t>& UpscaledPresents() {
    static std::atomic<std::uint64_t> count {0};
    return count;
}

std::atomic<std::uint64_t>& GpuDecodes() {
    static std::atomic<std::uint64_t> count {0};
    return count;
}

std::atomic<std::uint64_t>& CpuDecodes() {
    static std::atomic<std::uint64_t> count {0};
    return count;
}

}

namespace Metal {

PresentedFrame LastPresentedFrame() {
    std::lock_guard lock(FrameMutex());
    return LastFrame();
}

std::uint64_t MetalFxPresents() {
    return UpscaledPresents().load();
}

std::uint64_t GpuDecodedPresents() {
    return GpuDecodes().load();
}

std::uint64_t CpuDecodedPresents() {
    return CpuDecodes().load();
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
    // The GPU display decode (D66): its pipeline, the BGRA8 rows it writes (copied into `source`)
    // and the 16-bit float scanout thresholds.
    id<MTLComputePipelineState> decode = nil;
    id<MTLBuffer> decoded = nil;
    id<MTLBuffer> thresholds = nil;
    // The MetalFX upscale (D61): the scaler of one input and output size, its output (the size of the
    // aspect-fit rectangle) and the fence that orders it after the source upload. `fxUnavailable`:
    // the device or the scaler refused, the bilinear scaler presents from then on.
    id<MTL4FXSpatialScaler> fx = nil;
    id<MTLTexture> fxOutput = nil;
    id<MTLFence> fxFence = nil;
    bool fxUnavailable = false;
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
        if (queue != nil && residency != nil) ::AgcDriver::Metal::Residency::Detach(queue, residency);
        if (queue != nil && layer != nil) ::AgcDriver::Metal::Residency::Detach(queue, layer.residencySet);
    }

    void Resident(id<MTLAllocation> allocation) {
        ::AgcDriver::Metal::Residency::Add(residency, allocation);
        ::AgcDriver::Metal::Residency::Commit(residency);
    }

    // The MetalFX scaler and output for an upscale of `width` x `height` to `outputWidth` x `outputHeight`
    // (made again when a size changes); false when MetalFX is not available here (reported once).
    bool PrepareUpscale(id<MTL4Compiler> compiler, std::uint32_t width, std::uint32_t height, std::uint32_t outputWidth, std::uint32_t outputHeight, const std::function<void()>& retire) {
        if (fxUnavailable) return false;
        if (fx != nil && fx.inputWidth == width && fx.inputHeight == height && fx.outputWidth == outputWidth && fx.outputHeight == outputHeight) return true;
        const auto unavailable = [&](const std::string& why) {
            fxUnavailable = true;
            std::fprintf(stderr, "[metal] MetalFX upscale unavailable (%s): the bilinear scaler presents\n", why.c_str());
            return false;
        };
        if (![MTLFXSpatialScalerDescriptor supportsMetal4FX:device]) return unavailable("the device has no Metal 4 spatial scaler");
        if (compiler == nil) return unavailable("no Metal 4 compiler");
        // Earlier presents read the old output: they finish first.
        if (fx != nil) retire();
        if (fxOutput != nil) {
            ::AgcDriver::Metal::Residency::Remove(residency, fxOutput);
            fxOutput = nil;
        }
        fx = nil;
        MTLFXSpatialScalerDescriptor* descriptor = [MTLFXSpatialScalerDescriptor new];
        descriptor.colorTextureFormat = MTLPixelFormatBGRA8Unorm;
        descriptor.outputTextureFormat = MTLPixelFormatBGRA8Unorm;
        descriptor.inputWidth = width;
        descriptor.inputHeight = height;
        descriptor.outputWidth = outputWidth;
        descriptor.outputHeight = outputHeight;
        // The display buffer holds display-referred (gamma-encoded) color.
        descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
        fx = [descriptor newSpatialScalerWithDevice:device compiler:compiler];
        if (fx == nil) return unavailable("the scaler for " + std::to_string(width) + "x" + std::to_string(height) + " to " + std::to_string(outputWidth) + "x" + std::to_string(outputHeight) + " was refused");
        MTLTextureDescriptor* output = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:outputWidth height:outputHeight mipmapped:NO];
        output.storageMode = MTLStorageModePrivate;
        output.usage = fx.outputTextureUsage | MTLTextureUsageShaderRead;
        fxOutput = [device newTextureWithDescriptor:output];
        if (fxOutput == nil) Fail("cannot create the MetalFX output texture");
        Resident(fxOutput);
        if (fxFence == nil) fxFence = [device newFence];
        if (fxFence == nil) Fail("cannot create the MetalFX fence");
        if (PresentTrace()) std::fprintf(stderr, "[metal] MetalFX spatial scaler %ux%u to %ux%u\n", width, height, outputWidth, outputHeight);
        return true;
    }

    // The decode's output rows for a `width` x `height` display buffer and the thresholds table.
    void PrepareDecode(std::uint32_t width, std::uint32_t height, const std::function<void()>& retire) {
        if (thresholds == nil) {
            const auto& table = Graphics::HalfFloatScanoutThresholds();
            thresholds = [device newBufferWithBytes:table.data() length:sizeof(table) options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            if (thresholds == nil) Fail("cannot create the display decode thresholds");
            Resident(thresholds);
        }
        const auto bytes = static_cast<NSUInteger>(width) * height * 4u;
        if (decoded != nil && decoded.length == bytes) return;
        if (decoded != nil) {
            // Earlier presents decode into the old rows: they finish first.
            retire();
            ::AgcDriver::Metal::Residency::Remove(residency, decoded);
        }
        decoded = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
        if (decoded == nil) Fail("cannot create the display decode rows");
        Resident(decoded);
    }

    void MakeOffscreen() {
        for (auto& texture : offscreen) {
            if (texture != nil) ::AgcDriver::Metal::Residency::Remove(residency, texture);
            texture = nil;
        }
        ::AgcDriver::Metal::Residency::Commit(residency);
        if (layer != nil || width == 0u || height == 0u) return;
        for (auto& texture : offscreen) {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:width height:height mipmapped:NO];
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            texture = [device newTextureWithDescriptor:descriptor];
            if (texture == nil) Fail("cannot create the headless presentation targets");
            ::AgcDriver::Metal::Residency::Add(residency, texture);
        }
        ::AgcDriver::Metal::Residency::Commit(residency);
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
    ::AgcDriver::Metal::Residency::Attach(queue, presenter->residency);
    // The Metal view first: the drawable size is the view's (pixels, HiDPI) once it exists.
#ifdef __APPLE__
    if (window.metalLayer != nullptr) presenter->layer = (__bridge CAMetalLayer*)window.metalLayer(window.context);
#endif
    std::uint32_t width = window.width;
    std::uint32_t height = window.height;
    if (window.getDrawableSize != nullptr) window.getDrawableSize(window.context, &width, &height);
    if (presenter->layer == nil) HeadlessSizeOverride(width, height);
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
        ::AgcDriver::Metal::Residency::Attach(queue, presenter->layer.residencySet);
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
    presenter->decode = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"aps5_display_decode"] error:&error];
    if (presenter->decode == nil) Fail("the display decode has no pipeline: " + Metal::Describe(error));
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
    if (presenter->layer == nil && width != 0u && height != 0u) HeadlessSizeOverride(width, height);
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
            // Upstream acquires with a zero timeout and drops the frame when no swapchain image is
            // free. CAMetalLayer has no non-blocking acquire: nextDrawable waits up to its 1 s
            // timeout and returns nil, which is counted as the same drop.
            id<CAMetalDrawable> drawable = [presenter->layer nextDrawable];
            if (drawable == nil) {
                ++Counters().acquireDrops;
                return false;
            }
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
        const auto* gpuSource = std::exchange(pendingGpuSource, nullptr);
        const bool hasSource = !pixels.empty() || gpuSource != nullptr;
        if (!pixels.empty() && (width == 0u || height == 0u || width > 16384u || height > 16384u || pixels.size() != static_cast<std::size_t>(width) * height * 4u)) Fail("invalid source image for the present");
        const auto targetWidth = static_cast<std::uint32_t>(presenter->target.width);
        const auto targetHeight = static_cast<std::uint32_t>(presenter->target.height);
        const AspectFitRect rect = hasSource ? ComputeContainRect_nid_postfix(width, height, targetWidth, targetHeight) : AspectFitRect{};
        const bool upscale = hasSource && MetalFxRequested() && rect.width > width && rect.height > height && presenter->PrepareUpscale(state->compiler, width, height, rect.width, rect.height, [this] { RetirePresents(0); });
        MTLTextureUsage sourceUsage = MTLTextureUsageShaderRead;
        if (upscale) sourceUsage |= presenter->fx.colorTextureUsage;
        if (hasSource) {
            if (presenter->source == nil || presenter->source.width != width || presenter->source.height != height || (presenter->source.usage & sourceUsage) != sourceUsage) {
                if (presenter->source != nil) {
                    // Earlier presents read the old source: they finish first.
                    RetirePresents(0);
                    ::AgcDriver::Metal::Residency::Remove(presenter->residency, presenter->source);
                }
                MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:width height:height mipmapped:NO];
                descriptor.storageMode = MTLStorageModePrivate;
                descriptor.usage = sourceUsage;
                presenter->source = [state->device newTextureWithDescriptor:descriptor];
                if (presenter->source == nil) Fail("cannot create the present source texture");
                presenter->Resident(presenter->source);
            }
        }
        if (gpuSource != nullptr) presenter->PrepareDecode(width, height, [this] { RetirePresents(0); });
        auto& stream = recorder.Stream();
        if (hasSource) {
            const auto rowBytes = static_cast<NSUInteger>(width) * 4u;
            const auto imageBytes = rowBytes * height;
            id<MTL4ComputeCommandEncoder> copy = nil;
            if (gpuSource != nullptr) {
                const auto& buffer = *gpuSource->buffer;
                const bool halfFloat = DisplayHalfFloat(buffer.pixelFormat);
                const bool rgba = (buffer.pixelFormat & ~0x0100000000000000ull) == 0x8000000022000000ull;
                const std::uint32_t mode = (buffer.tilingMode == 0 ? 1u : 0u) | (!halfFloat && rgba ? 2u : 0u) | (!halfFloat && (buffer.pixelFormat & 0x0100000000000000ull) != 0 ? 4u : 0u)
                    | (halfFloat ? 8u : 0u) | (halfFloat && DisplayRedLow(buffer.pixelFormat) ? 16u : 0u);
                const DecodeParameters parameters{width, height, buffer.pitchInPixel == 0 ? width : buffer.pitchInPixel, mode, (width + 127u) / 128u, 0u,
                                                  gpuSource->import.gpuAddress + gpuSource->offset, presenter->decoded.gpuAddress, presenter->thresholds.gpuAddress};
                // A new encoder: its queue barrier orders the decode after every write recorded or
                // committed before it (the scene, the stores the flush hook just recorded).
                stream.EndEncoder();
                stream.DispatchHelper(presenter->decode, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake((width + 15u) / 16u, (height + 15u) / 16u, 1), MTLSizeMake(16, 16, 1));
                stream.Barrier(MTLStageDispatch, MTLStageBlit);
                copy = stream.Compute();
                [copy copyFromBuffer:presenter->decoded sourceOffset:0 sourceBytesPerRow:rowBytes sourceBytesPerImage:imageBytes sourceSize:MTLSizeMake(width, height, 1) toTexture:presenter->source destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            } else {
                auto upload = stream.Write(pixels, 256u);
                copy = stream.Compute();
                [copy copyFromBuffer:upload.buffer sourceOffset:upload.offset sourceBytesPerRow:rowBytes sourceBytesPerImage:imageBytes sourceSize:MTLSizeMake(width, height, 1) toTexture:presenter->source destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            }
            stream.CountCommand();
            if (upscale) {
                // MetalFX encodes its own passes into the batch: it waits for the fence the upload
                // updates and updates it when its output is written; the scaler pass waits for it.
                [copy updateFence:presenter->fxFence afterEncoderStages:MTLStageBlit | MTLStageDispatch];
                stream.EndEncoder();
                presenter->fx.colorTexture = presenter->source;
                presenter->fx.outputTexture = presenter->fxOutput;
                presenter->fx.inputContentWidth = width;
                presenter->fx.inputContentHeight = height;
                presenter->fx.fence = presenter->fxFence;
                [presenter->fx encodeToCommandBuffer:stream.CommandBuffer()];
                stream.CountCommand();
                UpscaledPresents().fetch_add(1u);
            }
        }
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = presenter->target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, opaque ? 1.0 : 0.0);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTL4RenderCommandEncoder> encoder = stream.BeginRender(pass);
        if (upscale) [encoder waitForFence:presenter->fxFence beforeEncoderStages:MTLRenderStageFragment];
        if (hasSource) {
            const std::array<float, 4> ndc {
                2.0f * static_cast<float>(rect.x) / static_cast<float>(targetWidth) - 1.0f,
                1.0f - 2.0f * static_cast<float>(rect.y) / static_cast<float>(targetHeight),
                2.0f * static_cast<float>(rect.x + static_cast<std::int32_t>(rect.width)) / static_cast<float>(targetWidth) - 1.0f,
                1.0f - 2.0f * static_cast<float>(rect.y + static_cast<std::int32_t>(rect.height)) / static_cast<float>(targetHeight)};
            const auto rectUpload = stream.Write(std::as_bytes(std::span(ndc)), 16u);
            [presenter->arguments setAddress:rectUpload.address atIndex:0];
            [presenter->arguments setTexture:(upscale ? presenter->fxOutput : presenter->source).gpuResourceID atIndex:0];
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
                ::AgcDriver::Metal::Residency::Remove(residency->residency, readback);
                ::AgcDriver::Metal::Residency::Commit(residency->residency);
            });
        }
        // Metal 4 drawables: the queue waits for the drawable before the batch and signals it after.
        if (presenter->drawable != nil) [state->queue waitForDrawable:presenter->drawable];
        const auto serial = recorder.SubmitAndEpoch();
        if (presenter->drawable != nil) [state->queue signalDrawable:presenter->drawable];
        if (PresentTrace()) std::fprintf(stderr, "[metal] present %llu: source %ux%u%s into %lux%lu%s, %s, batch %llu\n", static_cast<unsigned long long>(Counters().presents + 1u), width, height, hasSource ? "" : " (clear)", static_cast<unsigned long>(presenter->target.width), static_cast<unsigned long>(presenter->target.height), upscale ? (", MetalFX to " + std::to_string(rect.width) + "x" + std::to_string(rect.height)).c_str() : "", presenter->drawable != nil ? "drawable" : "offscreen", static_cast<unsigned long long>(serial));
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
    auto& dumps = Dumps();
    int dumpIndex = -1;
    if (dumps.dumped < dumps.limit && ++dumps.presents % static_cast<std::uint64_t>(dumps.every) == 0u) dumpIndex = dumps.dumped++;
    // D66: decoded on the GPU from the host import; a dump frame needs the pixels on the CPU.
    if (dumpIndex < 0 && GpuDecodeRequested()) {
        const auto size = DisplayBufferSize(buffer);
        std::uint64_t offset = 0;
        if (id<MTLBuffer> import = state->ImportOf(buffer.address, size, &offset); import != nil) {
            PerformanceTimer timing("DisplayBuffer.GpuDecode");
            GuestMemory::CheckRange(reinterpret_cast<const void*>(buffer.address), size, 65536);
            const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::Scanout);
            Graphics::Metal::FlushForGpuRead(buffer.address, size);
            timing.Mark("flush");
            const GpuSource source{&buffer, import, offset};
            struct Pending {
                explicit Pending(const GpuSource* source) { pendingGpuSource = source; }
                ~Pending() { pendingGpuSource = nullptr; }
            } pending(&source);
            const bool presented = present(buffer.width, buffer.height, true, {});
            timing.Mark("encode");
            if (presented) GpuDecodes().fetch_add(1u);
            return presented;
        }
    }
    // The CPU decode (linear or tiled, the formats DisplayBuffer.cpp decodes); the read waits for
    // recorded GPU writes over it.
    const auto pixels = ReadDisplayBuffer(buffer);
    CpuDecodes().fetch_add(1u);
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
