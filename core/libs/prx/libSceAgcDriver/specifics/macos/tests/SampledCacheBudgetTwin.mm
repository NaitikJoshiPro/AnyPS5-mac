// Metal twin of tests/execution/SampledCacheBudget.cpp (agc_driver_sampled_cache_budget, its `memory` mode, the
// agc_driver_resource_cache_* modes and the device-selection modes), swapped in by twins/D5-cache-twins.cmake. The upstream file is compiled
// here unchanged with its main renamed, and every one of its checks runs as written: the sampled texture cache and
// the storage image cache, the resource cache and its admission/clear rules are backend-neutral code over the Metal
// image path (ShaderResources.mm, Texture.mm, MetalTextures.mm). Only the device differs: instead of a Vulkan
// instance, the Context is the Metal image cache's (Metal::ImageCache::Attach, as MetalDevice.mm builds it) with
// - host imports as the device enables them (no-copy Metal buffers over whole 16 KiB pages, D3), so the storage-image
//   view and resource-cache checks run (upstream needs VK_EXT_external_memory_host for them);
// - the device-local heap and memory budget report the Metal device's own Context has (Metal::DescribeMemory,
//   MetalMemoryBudget.h, as MetalDevice.mm calls it): one heap of the recommended working set, and upstream's
//   VK_EXT_memory_budget query answered with usage = the device's currently allocated size and a budget bounded by
//   the working set, the system's free memory and OS memory pressure. The `memory` mode replaces the report with
//   upstream's fixed one, as upstream does.
// Upstream's recorderEvictionTests (resource-admission mode) reads the evicted storage image back with
// vkCmdCopyImageToBuffer through Context::Function; here that call is D4's Metal::Vk::CmdCopyImageToBuffer (the
// entry point the texture twins use for it), recorded into the active Metal recorder's command stream. The type
// name is swapped for a twin-local callable only while the upstream file compiles, so no shared header changes.

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"

// Declared as in ../src/MetalVulkanImages.h, which is included after the upstream file (it imports Foundation).
namespace AgcDriver::Graphics::Metal::Vk {
void CmdCopyImageToBuffer(VkCommandBuffer commands, VkImage source, VkImageLayout layout, VkBuffer destination, std::uint32_t count, const VkBufferImageCopy* regions);
}

namespace MetalSampledCacheTwin {

struct CopyImageToBuffer {
    void operator()(VkCommandBuffer commands, VkImage image, VkImageLayout layout, VkBuffer buffer, std::uint32_t count, const VkBufferImageCopy* regions) const {
        AgcDriver::Graphics::Metal::Vk::CmdCopyImageToBuffer(commands, image, layout, buffer, count, regions);
    }
};

}

namespace AgcDriver::Graphics {

template<>
inline MetalSampledCacheTwin::CopyImageToBuffer Context::Function<MetalSampledCacheTwin::CopyImageToBuffer>(const char*) const {
    return {};
}

}

#define PFN_vkCmdCopyImageToBuffer MetalSampledCacheTwin::CopyImageToBuffer
#define main AgcDriverSampledCacheBudgetVulkanMain
#include "../../../tests/execution/SampledCacheBudget.cpp"
#undef main
#undef PFN_vkCmdCopyImageToBuffer

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalImages.h"
#include "../src/MetalMemoryBudget.h"
#include "MetalTestDevice.h"

namespace MetalSampledCacheTwin {

using namespace AgcDriver::Graphics;

constexpr int Skipped = 77;

id<MTLDevice> reportingDevice = nil;

// The Metal test device as upstream's Device: GetContext() is the image cache's Context (detiler, placeholders,
// sampler and texture caches). No recorder is active outside upstream's recorderEvictionTests, as upstream's test has
// none: with one, the batch it records keeps evicted storage images and rejected resources until it completes
// (MetalTextures.mm, ShaderResources.mm), as on Vulkan, which is what recorderEvictionTests checks; the other image
// work runs in one-shot CommandBatches.
class Device {
public:
    Device() : cache(test.device, test.queue) {
        auto& context = test.context;
        // The limits the device's setup gives its Context (MetalDevice.mm), which the test device leaves at zero.
        const auto bufferLength = static_cast<std::uint64_t>(test.device.maxBufferLength);
        context.limits.maxStorageBufferRange = static_cast<std::uint32_t>(std::min<std::uint64_t>(bufferLength, 0xffffffffu));
        context.limits.maxUniformBufferRange = context.limits.maxStorageBufferRange;
        context.limits.maxComputeWorkGroupCount[0] = context.limits.maxComputeWorkGroupCount[1] = context.limits.maxComputeWorkGroupCount[2] = 65535u;
        context.hostImportAlignment = 16384u;
        Metal::DescribeMemory(context, test.device);
        reportingDevice = test.device;
        cache.Attach(context);
    }
    ~Device() {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        cache.Teardown();
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    const Context& GetContext() const { return *cache.ImageContext(); }

private:
    MetalTest::Device test;
    mutable Metal::ImageCache cache;
};

}

// upstream main with the Metal device; the modes, environment and checks are upstream's.
int main(int argc, char** argv) {
    try {
        const bool memory = argc > 1 && std::string(argv[1]) == "memory";
        const std::string mode = argc > 1 ? argv[1] : "";
        // Upstream's Vulkan physical-device ranking (8a47d68ba1, b71f1b8000) is pure CPU logic: it runs as written.
        if (mode == "device-selection-cpu") {
            deviceSelectionTests();
            return 0;
        }
        const bool resources = mode == "resource-admission" || mode == "resource-oversized" || mode == "resource-clear";
        if (!memory) setenv("APS5_TEXTURE_CACHE_MIB", "1", 1);
        // b4cd843324: the template audit interval lookupChangedTests waits past (upstream sets it the same way).
        setenv("APS5_TEMPLATE_AUDIT_MS", "250", 0);
        // The Metal device always imports host memory: a refused import fails the resource modes instead of skipping.
        setenv("ANYPS5_REQUIRE_VULKAN", "1", 1);
        MetalSampledCacheTwin::Device device;
        auto context = device.GetContext();
        std::printf("device %s: recommended working set %llu MiB\n", MetalSampledCacheTwin::reportingDevice.name.UTF8String, static_cast<unsigned long long>(context.memory.memoryHeaps[0].size >> 20u));
        if (mode == "device-selection") {
            // The Metal fixture's device is the system default MTLDevice (MetalTest::Device), as the Metal driver's.
            // Upstream's check that the fixture's physical device matches ANYPS5_GPU, through upstream's selectDevice
            // over the Metal device described as a Vulkan 1.1 integrated GPU (Apple GPUs are unified-memory).
            VkPhysicalDeviceProperties properties{};
            properties.apiVersion = VK_API_VERSION_1_1;
            properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
            std::snprintf(properties.deviceName, sizeof(properties.deviceName), "%s", MetalSampledCacheTwin::reportingDevice.name.UTF8String);
            const char* request = std::getenv("ANYPS5_GPU");
            Require(selectDevice(std::span(&properties, 1), request != nullptr ? request : "").has_value(), "the sampled-cache fixture's Metal device does not match ANYPS5_GPU");
            std::puts("sampled cache device selection passed on Metal");
            return 0;
        }
        if (resources) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            if (!resourceCacheTests(context, mode)) return MetalSampledCacheTwin::Skipped;
            std::puts("resource cache tests passed on Metal");
            return 0;
        }
        // 3c43ce33a0 / 56ddbbbd0d: templates over a released or changed sampled lookup must rebuild. The Metal device
        // always imports host memory (ANYPS5_REQUIRE_VULKAN above), so lookup-changed fails instead of skipping.
        if (mode == "resource-replaced") {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            replacedEntryTests(context);
            std::puts("resource cache replaced entry tests passed on Metal");
            return 0;
        }
        if (mode == "resource-lookup-changed") {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            if (!lookupChangedTests(context)) return MetalSampledCacheTwin::Skipped;
            std::puts("resource cache lookup-changed tests passed on Metal");
            return 0;
        }
        if (memory) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            memoryTests(context);
            storageMemoryTests(context);
            ClearCachedTextures(context.device);
            std::puts("sampled texture memory budget tests passed on Metal");
            return 0;
        }
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            {
                // viewTests and viewEvictionTests return early when the block is not imported: on Metal it must be.
                Block probe(context);
                Require(HostImportFor(context, probe.Surface(0), BlockBytes) != nullptr, "the Metal device refused the host import of the texture block");
            }
            reportedBudgetTests(context);
            snapshotTests(context);
            heldTests(context);
            rescanTests(context);
            viewEvictionTests(context);
            viewTests(context);
            frameTests(context);
            earlyMissTests(context);
            collidingKeyTests(context);
            ClearCachedTextures(context.device);
        }
        std::puts("sampled texture cache budget tests passed on Metal");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
