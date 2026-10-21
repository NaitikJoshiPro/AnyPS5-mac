// Metal twins of the tests/Recorder.cpp modes upstream added between 5d9dcc42d2 and 0d078f75d3 (UPSYNC3 round 3):
// - --device-selection-only (agc_recorder_device_selection, deviceSelectionTests, upstream 82b4a665fa): upstream checks
//   the rule its recorder fixtures pick a Vulkan device by (ANYPS5_GPU names a device by a case-insensitive part of its
//   name; without it the best-ranked device type; devices below Vulkan 1.1 never). The Metal device picks by the rule
//   of MetalDevice.mm SelectDevice (ANYPS5_GPU: the first device whose name contains it, case-insensitive; without it
//   the system default; a device without Metal 4 is never usable). The twin checks that rule over upstream's table
//   (the system default standing in for the discrete GPU upstream ranks first, Metal 4 support for Vulkan 1.1) and
//   over the machine's own devices: each is selected by its full name and by a case-changed part of it.
// - --sampled-storage-base-level-usage-only (agc_sampled_storage_base_level_usage, sampledStorageBaseLevelUsageTests,
//   upstream 724b16624f): a sampled Texture over a mipmapped storage image (sRGB RGBA8, format 130) makes its
//   base-level view lazily, once, and reads it decoded. Upstream also checks that the view restricts its usage to
//   SAMPLED (VkImageViewUsageCreateInfo, through an intercepted vkCreateImageView): Metal texture views carry no
//   usage subset (MetalVulkanImages.mm CreateImageView makes a newTextureViewWithPixelFormat view), so the twin checks
//   instead that the view is a one-level 2D sRGB view of the storage texture and that it is reused. The decoded red is
//   read by an MSL kernel. Expected values are upstream's.
// - --deferred-failure-report-only (agc_recorder_deferred_failure_report, deferredFailureReportTests): copied
//   unchanged with its StderrCapture and DeferredFailureLines helpers; Recorder.mm reports deferred write-back
//   failures as Recorder.cpp does (every plain failure, the first 8 unaligned BDA reports per instruction, then
//   every 1024th).
// - --gpu-timing-complete-only / --gpu-timing-incomplete-only (agc_gpu_timing_complete, agc_gpu_timing_incomplete,
//   gpuTimingCompletionTests): copied unchanged but for upstream's "queue timestamp queries unavailable" exit: the
//   Metal recorder always has its MTL4 timestamp heap (Recorder.mm readGpuTiming), so there is nothing to skip.

#include "RecorderTwinHarness.h"
#include <functional>
#include <unistd.h>
#include "RecorderTwins.h"
#include "../src/MetalCommandStream.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Candidate {
    std::string name;
    bool metal4 = true;
    bool systemDefault = false;
};

// MetalDevice.mm SelectDevice's rule over a list of devices.
std::optional<std::size_t> selectDevice(std::span<const Candidate> devices, std::string_view request) {
    const auto lower = [](std::string_view text) {
        std::string result(text);
        for (auto& character : result) character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
        return result;
    };
    const auto wanted = lower(request);
    for (std::size_t index = 0; index < devices.size(); ++index) {
        const auto& candidate = devices[index];
        if (!candidate.metal4) continue;
        if (request.empty() ? candidate.systemDefault : lower(candidate.name).find(wanted) != std::string::npos) return index;
    }
    return std::nullopt;
}

void deviceSelectionTests() {
    std::array<Candidate, 4> devices{
        Candidate{"AMD Radeon Graphics", true, false},
        Candidate{"NVIDIA GeForce RTX 5080", true, true},
        Candidate{"Virtual GPU", true, false},
        Candidate{"CPU", true, false},
    };
    Require(!selectDevice({}, {}).has_value(), "an empty device list selected a device");
    Require(selectDevice(devices, {}) == 1u, "default selection did not take the system default device");
    Require(selectDevice(devices, "aMd rAdEoN") == 0u, "a requested GPU lost to the system default device");
    Require(selectDevice(devices, "rtx 5080") == 1u, "a partial GPU name did not match without regard to case");
    Require(selectDevice(devices, "Virtual") == 2u, "a requested virtual GPU was not selected");
    Require(selectDevice(devices, "cpu") == 3u, "a requested CPU device was not selected");
    Require(!selectDevice(devices, "missing").has_value(), "an unmatched request selected another GPU");
    Require(!selectDevice(devices, "NVIDIA GeForce RTX 5080 with a longer name").has_value(), "a request longer than the device name matched");
    devices[0].metal4 = false;
    Require(!selectDevice(devices, "AMD").has_value(), "a requested device without Metal 4 was selected");
    Require(selectDevice(devices, {}) == 1u, "an unsupported device changed default selection");
    devices[0] = devices[1];
    Require(selectDevice(devices, {}) == 0u && selectDevice(devices, "NVIDIA") == 0u, "equal matches did not retain enumeration order");
    for (auto& device : devices) device.metal4 = false;
    Require(!selectDevice(devices, {}).has_value(), "a list without Metal 4 selected a device");

    // The machine's devices, as MTLCopyAllDevices enumerates them for SelectDevice.
    @autoreleasepool {
        id<MTLDevice> systemDefault = MTLCreateSystemDefaultDevice();
        std::vector<Candidate> machine;
        for (id<MTLDevice> device in MTLCopyAllDevices()) {
            machine.push_back({device.name.UTF8String, static_cast<bool>([device supportsFamily:MTLGPUFamilyMetal4]), systemDefault != nil && device.registryID == systemDefault.registryID});
        }
        Require(!machine.empty(), "Metal enumerated no device");
        const auto selected = selectDevice(machine, {});
        Require(systemDefault == nil || ![systemDefault supportsFamily:MTLGPUFamilyMetal4] || (selected.has_value() && machine[*selected].systemDefault), "default selection did not take the machine's system default device");
        for (std::size_t index = 0; index < machine.size(); ++index) {
            const auto& candidate = machine[index];
            if (!candidate.metal4) continue;
            const auto byName = selectDevice(machine, candidate.name);
            Require(byName.has_value() && machine[*byName].name == candidate.name, "the machine's device " + candidate.name + " was not selected by its name");
            std::string upper = candidate.name.substr(0, std::min<std::size_t>(candidate.name.size(), 5));
            for (auto& character : upper) character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
            const auto byPart = selectDevice(machine, upper);
            Require(byPart.has_value() && byPart <= index, "the machine's device " + candidate.name + " was not matched by a case-changed part of its name");
            std::cout << "Metal device " << index << ": " << candidate.name << (candidate.systemDefault ? " (system default)" : "") << '\n';
        }
    }
}

// The red channel of texel (0, 0) of a one-level 2D view (Texture::prepareBaseLevelView makes the base-level view of a
// 2D array texture a 2D view of layer 0), read by an MSL kernel (reads of an sRGB view return linear values).
float readRed(const Context& context, VkImageView view) {
    id<MTLDevice> device = AgcDriver::Graphics::Metal::DeviceOf(context);
    id<MTLTexture> texture = AgcDriver::Graphics::Metal::TextureOfView(view);
    Require(device != nil && texture != nil, "the base-level view has no Metal texture");
    static NSString* const source = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void red2D(texture2d<float, access::read> image [[texture(0)]], device float* out [[buffer(0)]]) {\n"
        "    out[0] = image.read(uint2(0, 0), 0u).x;\n"
        "}\n";
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    Require(library != nil, std::string("the red probe kernel did not compile: ") + (error != nil ? error.localizedDescription.UTF8String : "unknown error"));
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"red2D"] error:&error];
    Require(pipeline != nil, "the red probe pipeline was not created");
    id<MTLBuffer> output = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    *static_cast<float*>(output.contents) = -1.0f;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:texture atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    Require(commands.status == MTLCommandBufferStatusCompleted, "the red probe command buffer failed");
    return *static_cast<const float*>(output.contents);
}

void expectRed(float got, float want, const char* what) {
    if (std::abs(got - want) > 1.5f / 255.0f) throw std::runtime_error(std::string(what) + ": read " + std::to_string(got) + ", expected " + std::to_string(want));
}

void sampledStorageBaseLevelUsageTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.width = resource.height = 8;
    resource.depthOrLastArray = 1;
    resource.mipCount = 2;
    resource.lastLevel = 1;
    resource.minLod = context.imageViewMinLod ? 0x100 : 0;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2DArray;
    resource.format = 130;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    const auto geometry = DescribeSurface(resource);
    std::vector<std::byte> memory(static_cast<std::size_t>(geometry.guestBytes) + 256, std::byte{0x80});
    resource.baseAddress = (reinterpret_cast<std::uintptr_t>(memory.data()) + 255) & ~std::uintptr_t{255};
    auto storage = StorageTexture::Create(context, detiler, resource, 0);
    recorder.Keep(storage);
    recorder.Sync();
    Texture sampled(context, storage, resource, VkComponentMapping{});
    const auto base = sampled.BaseLevelView();
    Require(base != VK_NULL_HANDLE, "the sampled storage image has no base-level view");
    id<MTLTexture> view = AgcDriver::Graphics::Metal::TextureOfView(base);
    Require(view != nil && view.mipmapLevelCount == 1u && view.textureType == MTLTextureType2D && view.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB, "the lazy base-level view of a sampled storage image is not a one-level 2D sRGB view");
    recorder.Sync();
    expectRed(readRed(context, base), 0.21586f, "sampled sRGB base level");
    Require(sampled.BaseLevelView() == base, "the lazy base-level view of a sampled storage image must restrict usage to SAMPLED and be reused");
}

const bool deviceSelection = AgcDriver::MetalTwins::RegisterRecorderMode("--device-selection-only", [] {
    try {
        deviceSelectionTests();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "recorder device selection tests passed\n";
    return 0;
});

const bool sampledStorageBaseLevelUsage = AgcDriver::MetalTwins::RegisterRecorderMode("--sampled-storage-base-level-usage-only", [] {
    return RunMode([](const Device& device, Recorder& recorder) { sampledStorageBaseLevelUsageTests(device, recorder); });
});


class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        Require(file != nullptr, "cannot open a temporary file for stderr");
        saved = dup(fileno(stderr));
        dup2(fileno(file), fileno(stderr));
    }

    std::string Finish() {
        std::fflush(stderr);
        dup2(saved, fileno(stderr));
        close(saved);
        std::rewind(file);
        std::string text;
        char buffer[4096];
        for (std::size_t read = 0; (read = std::fread(buffer, 1, sizeof(buffer), file)) != 0;) text.append(buffer, read);
        std::fclose(file);
        std::fwrite(text.data(), 1, text.size(), stderr);
        return text;
    }

private:
    std::FILE* file = nullptr;
    int saved = -1;
};

std::size_t DeferredFailureLines(Recorder& recorder, std::size_t count, const std::function<void()>& failure, std::string* log = nullptr) {
    StderrCapture capture;
    for (std::size_t index = 0; index < count; ++index) recorder.OnComplete(failure);
    recorder.Sync();
    const auto text = capture.Finish();
    if (log != nullptr) *log = text;
    std::size_t lines = 0;
    for (auto at = text.find("deferred write-back failed"); at != std::string::npos; at = text.find("deferred write-back failed", at + 1)) ++lines;
    return lines;
}

void deferredFailureReportTests(Recorder& recorder) {
    using AgcDriver::Graphics::BdaUnalignedAccess;
    std::uint32_t serial = 0;
    const auto plain = [&serial] { throw std::runtime_error("image table: poisoned entry " + std::to_string(serial++)); };
    const auto unaligned = [](std::uint32_t instruction) { return [instruction] { throw BdaUnalignedAccess("BDA access failed: unaligned", instruction); }; };
    Require(DeferredFailureLines(recorder, 12, plain) == 12, "a deferred failure other than an unaligned BDA access was not printed every time");
    Require(DeferredFailureLines(recorder, 12, unaligned(0x5c8)) == 8, "the first 8 deferred reports of an unaligned BDA access were not the only ones printed");
    std::string log;
    Require(DeferredFailureLines(recorder, 1012, unaligned(0x5c8), &log) == 1 && log.find("pc 0x5c8 1024 times so far") != std::string::npos, "the 1024th deferred report of an unaligned BDA access was not the only one printed");
    Require(DeferredFailureLines(recorder, 12, unaligned(0x7a0)) == 8, "unaligned BDA reports at one instruction limited the printing of the reports at another");
    Require(DeferredFailureLines(recorder, 12, plain) == 12, "unaligned BDA reports limited the printing of other deferred failures");
}

const bool deferredFailureReport = AgcDriver::MetalTwins::RegisterRecorderMode("--deferred-failure-report-only", [] {
    return RunMode([](const Device&, Recorder& recorder) {
        deferredFailureReportTests(recorder);
        std::cout << "Deferred failure report tests passed\n";
    });
});

int gpuTimingCompletionTests(const Device& device, bool incomplete) {
    const auto& context = device.GetContext();
    Require(Recorder::GpuTimingEnabled(), "GPU timing completion test requires APS5_PROFILE_GPU");
    std::unique_lock gpu(GpuMutex());
    int completed = 0;
    Recorder recorder(context);
    recorder.Activate();
    for (int round = 0; round < 3; ++round) {
        const bool missingEnd = incomplete && round == 1;
        const auto first = recorder.BeginGpuTiming(0x12340);
        Require(first != Recorder::NoTiming, "first GPU timing range was not allocated");
        if (!missingEnd) recorder.EndGpuTiming(first);
        const auto second = recorder.BeginGpuTiming(0x12341);
        Require(second != Recorder::NoTiming && second != first, "second GPU timing range was not allocated");
        recorder.EndGpuTiming(second);
        recorder.OnComplete([&] { ++completed; });
        const auto serial = recorder.Submissions() + 1;
        recorder.Submit();
        recorder.Sync();
        Require(completed == round + 1 && recorder.Idle(), "GPU timing prevented batch completion");
        std::size_t missing = 0;
        const auto batches = recorder.CompletedBatches(serial - 1, serial, missing);
        Require(missing == 0 && batches.size() == 1 && batches.front().serial == serial, "GPU timing batch completion record is missing");
        const auto& batch = batches.front();
        if (missingEnd) {
            Require(batch.gpuStartNs == 0 && batch.gpuEndNs == 0, "incomplete query batch published GPU timings");
        } else {
            Require(batch.gpuStartNs > 0 && batch.gpuEndNs >= batch.gpuStartNs, "complete query batch lost GPU timings");
        }
    }
    std::cout << "GPU timing batch completion checked, incomplete=" << incomplete << '\n';
    return 0;
}

int RunGpuTiming(bool incomplete) {
    @autoreleasepool {
        try {
            Device device;
            return gpuTimingCompletionTests(device, incomplete);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}

const bool gpuTimingComplete = AgcDriver::MetalTwins::RegisterRecorderMode("--gpu-timing-complete-only", [] { return RunGpuTiming(false); });
const bool gpuTimingIncomplete = AgcDriver::MetalTwins::RegisterRecorderMode("--gpu-timing-incomplete-only", [] { return RunGpuTiming(true); });

}
