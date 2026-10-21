// Metal twin of tests/execution/PartiallyResidentSnapshot.cpp (agc_driver_partially_resident_snapshot with no
// arguments, agc_driver_partially_resident_pending_stores with --pending-stores, agc_driver_large_tracked_snapshot
// with --large-tracked), swapped in by a twins/*.cmake line.
// The upstream file is compiled here unchanged with its main renamed, and its Run, PendingStores and LargeTracked
// (6849eeaec0: a large write-tracked surface keeps no guest byte snapshot, so a driver store re-uploads it) run as written:
// the snapshot cache of a sampled texture over a partially committed surface (CollectCommittedWrites, write stamps
// over the committed runs, Recorder::SnapshotWriteOverlaps) is backend-neutral code over the Metal image path
// (ShaderResources.mm, Texture.mm, MetalTextures.mm), and the reserved surface is committed, decommitted and
// write-watched by upstream's own POSIX ReservedSurface (mmap + the mprotect write watch of macOS). Only the device
// differs: instead of a Vulkan instance (upstream's Device stays compiled but unused), the Context is the Metal image
// cache's (Metal::ImageCache::Attach, as MetalDevice.mm builds it), with
// - the limits, the device-local heap and memory budget report the Metal device's own Context has
//   (Metal::DescribeMemory, MetalMemoryBudget.h), which size the sampled texture cache;
// - host imports as the device enables them (no-copy Metal buffers over whole 16 KiB pages, D3) and its import watch
//   decision (PrepareImportWatch: on Metal imports leave the write watch and are compared, DECISIONS D19). A surface
//   with reserved pages is not importable (not Accessible), so it is sampled through a snapshot as on Vulkan;
// - the image cache's detiler (Metal TextureDetiler) instead of upstream's TextureDetiler over the Vulkan Context.
// --untracked-hint keeps upstream's non-Windows branch (a skip): its test is compiled only under _WIN32. Metal host
// imports leave the write watch too, so the mode could run here once ported; it is not registered on macOS
// (upstream registers agc_driver_untracked_snapshot_hint only on WIN32).

#define main AgcDriverPartiallyResidentSnapshotVulkanMain
#include "../../../tests/execution/PartiallyResidentSnapshot.cpp"
#undef main

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalImages.h"
#include "../src/MetalMemoryBudget.h"
#include "MetalTestDevice.h"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include <algorithm>

namespace MetalPartiallyResidentTwin {

using namespace AgcDriver::Graphics;

constexpr int Skipped = 77;

// The Metal test device as upstream's Device: GetContext() is the image cache's Context (detiler, placeholders,
// sampler and texture caches), set up as MetalDevice.mm sets up the device's Context.
class Device {
public:
    Device() : cache(test.device, test.queue) {
        auto& context = test.context;
        auto& limits = context.limits;
        const auto bufferLength = static_cast<std::uint64_t>(test.device.maxBufferLength);
        limits.maxStorageBufferRange = static_cast<std::uint32_t>(std::min<std::uint64_t>(bufferLength, 0xffffffffu));
        limits.maxUniformBufferRange = limits.maxStorageBufferRange;
        limits.maxBoundDescriptorSets = 4u;
        limits.minStorageBufferOffsetAlignment = 4u;
        limits.nonCoherentAtomSize = 1u;
        limits.maxComputeWorkGroupCount[0] = limits.maxComputeWorkGroupCount[1] = limits.maxComputeWorkGroupCount[2] = 65535u;
        const auto threads = test.device.maxThreadsPerThreadgroup;
        limits.maxComputeWorkGroupSize[0] = static_cast<std::uint32_t>(threads.width);
        limits.maxComputeWorkGroupSize[1] = static_cast<std::uint32_t>(threads.height);
        limits.maxComputeWorkGroupSize[2] = static_cast<std::uint32_t>(threads.depth);
        limits.maxComputeWorkGroupInvocations = 1024u;
        limits.maxComputeSharedMemorySize = static_cast<std::uint32_t>(test.device.maxThreadgroupMemoryLength);
        context.subgroup.subgroupSize = 32u;
        // Host imports as the device's setup enables them: no-copy buffers over whole 16 KiB pages.
        context.hostImportAlignment = 16384u;
        Metal::DescribeMemory(context, test.device);
        cache.Attach(context);
        PrepareImportWatch(*cache.ImageContext());
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

// upstream main with the Metal device; the modes and checks are upstream's.
int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (!AgcDriver::GuestMemory::WriteWatched()) {
                std::puts("skipped, guest memory has no write watch");
                return MetalPartiallyResidentTwin::Skipped;
            }
            std::optional<MetalPartiallyResidentTwin::Device> device;
            device.emplace();
            {
                auto context = device->GetContext();
                const bool large = argc == 2 && std::string_view(argv[1]) == "--large-tracked";
                ReservedSurface surface(large ? LargeSurfaceBytes : SurfaceBytes);
                std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
                if (large) {
                    LargeTracked(context, surface, LargeSurfaceBytes);
                    LargeTracked(context, surface, BlockBytes);
                } else if (argc == 2 && std::string_view(argv[1]) == "--pending-stores") PendingStores(context, surface);
                else if (argc == 2 && std::string_view(argv[1]) == "--untracked-hint") {
#ifdef _WIN32
                    UntrackedHint(context, surface);
#else
                    std::puts("skipped, only Windows host imports leave the write watch with a hint");
                    return MetalPartiallyResidentTwin::Skipped;
#endif
                } else Run(context, surface);
                ClearCachedTextures(context.device);
            }
            device.reset();
            std::puts("partially resident snapshot tests passed on Metal");
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
