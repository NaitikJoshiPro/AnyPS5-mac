// Metal twin of tests/execution/SampledCacheBudget.cpp (agc_driver_sampled_cache_budget, its `memory` mode and the
// three agc_driver_resource_cache_* modes), swapped in by twins/D5-cache-twins.cmake. The upstream file is compiled
// here unchanged with its main renamed, and every one of its checks runs as written: the sampled texture cache and
// the storage image cache, the resource cache and its admission/clear rules are backend-neutral code over the Metal
// image path (ShaderResources.mm, Texture.mm, MetalTextures.mm). Only the device differs: instead of a Vulkan
// instance, the Context is the Metal image cache's (Metal::ImageCache::Attach, as MetalDevice.mm builds it) with
// - host imports as the device enables them (no-copy Metal buffers over whole 16 KiB pages, D3), so the storage-image
//   view and resource-cache checks run (upstream needs VK_EXT_external_memory_host for them);
// - the device-local heap of a unified-memory Mac, its recommended working set (as BufferPool.mm describes it), and
//   the MTLDevice standing in for the physical device;
// - a memory budget report (upstream's VK_EXT_memory_budget query) from the same device: budget = the recommended
//   working set, usage = the device's currently allocated size. The `memory` mode replaces it with upstream's fixed
//   report, as upstream does.
// Gap (TechnicalDebt metal-vulkan-internal-tests): the Metal device's own Context does not describe its heap or
// report a budget yet, so outside these tests the texture caches keep the 2 GiB floor.

#define main AgcDriverSampledCacheBudgetVulkanMain
#include "../../../tests/execution/SampledCacheBudget.cpp"
#undef main

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalImages.h"
#include "MetalTestDevice.h"

namespace MetalSampledCacheTwin {

using namespace AgcDriver::Graphics;

constexpr int Skipped = 77;

id<MTLDevice> reportingDevice = nil;

// VK_EXT_memory_budget's report for the Metal device: one device-local heap, the recommended working set.
void VKAPI_PTR ReportMetalBudget(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2* properties) {
    auto* reported = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(properties->pNext);
    reported->heapBudget[0] = static_cast<VkDeviceSize>(reportingDevice.recommendedMaxWorkingSetSize);
    reported->heapUsage[0] = static_cast<VkDeviceSize>(reportingDevice.currentAllocatedSize);
}

// The Metal test device as upstream's Device: GetContext() is the image cache's Context (detiler, placeholders,
// sampler and texture caches) with the device's recorder active.
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
        context.physical = reinterpret_cast<VkPhysicalDevice>(context.device);
        context.memory.memoryHeapCount = 1;
        context.memory.memoryHeaps[0] = {static_cast<VkDeviceSize>(test.device.recommendedMaxWorkingSetSize), VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
        reportingDevice = test.device;
        context.memoryProperties2 = &ReportMetalBudget;
        recorder = std::make_unique<Recorder>(context, true);
        context.recorder = recorder.get();
        recorder->Activate();
        cache.Attach(context);
    }
    ~Device() {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder->Sync();
        cache.Teardown();
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    const Context& GetContext() const { return *cache.ImageContext(); }

private:
    MetalTest::Device test;
    mutable Metal::ImageCache cache;
    std::unique_ptr<Recorder> recorder;
};

}

// upstream main with the Metal device; the modes, environment and checks are upstream's.
int main(int argc, char** argv) {
    try {
        const bool memory = argc > 1 && std::string(argv[1]) == "memory";
        const std::string mode = argc > 1 ? argv[1] : "";
        const bool resources = mode == "resource-admission" || mode == "resource-oversized" || mode == "resource-clear";
        if (!memory) setenv("APS5_TEXTURE_CACHE_MIB", "1", 1);
        // The Metal device always imports host memory: a refused import fails the resource modes instead of skipping.
        setenv("ANYPS5_REQUIRE_VULKAN", "1", 1);
        MetalSampledCacheTwin::Device device;
        auto context = device.GetContext();
        std::printf("device %s: recommended working set %llu MiB\n", MetalSampledCacheTwin::reportingDevice.name.UTF8String, static_cast<unsigned long long>(context.memory.memoryHeaps[0].size >> 20u));
        if (resources) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            if (!resourceCacheTests(context, mode)) return MetalSampledCacheTwin::Skipped;
            std::puts("resource cache tests passed on Metal");
            return 0;
        }
        if (memory) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            memoryTests(context);
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
