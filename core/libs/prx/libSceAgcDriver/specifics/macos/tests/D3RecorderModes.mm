// D3 test twins (work package D3, taken over by TEX): tests/Recorder.cpp's --unit-shadow-only,
// --host-import-unmap-only, --unchanged-cpu-stamps-only and --settled-baseline-only modes on Metal.
// The test functions (Recorder.cpp:70-97, 2138-2321, 2493-2576, 3389-3585) are copied unchanged but
// for their Vulkan calls, which go through Metal::Vk (MetalVulkanImages.h); `Device` is the Metal
// test device with the image cache attached as device creation attaches it.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/PipelineLinkOptimization.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineLibrary.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/BdaFeatures.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include <sys/mman.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "RecorderTwins.h"
#include "MetalTestDevice.h"
#include "../src/MetalImages.h"
#include "../src/MetalVulkanImages.h"
#include <sys/mman.h>

namespace {
extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
}
using namespace AgcDriver::Graphics;
using AgcDriver::GuestMemory::GpuMutex;

// The Metal test device as the tests' Device: its Context is the image cache's (detiler,
// placeholders, sampler and texture caches), WaitQueue waits for the active recorder's work.
class Device {
public:
    Device() : cache(test.device, test.queue) {
    // The limits the device's setup gives its Context (MetalDevice.mm), which the test device leaves at zero.
    const auto bufferLength = static_cast<std::uint64_t>(test.device.maxBufferLength);
    test.context.limits.maxStorageBufferRange = static_cast<std::uint32_t>(std::min<std::uint64_t>(bufferLength, 0xffffffffu));
    test.context.limits.maxUniformBufferRange = test.context.limits.maxStorageBufferRange;
    test.context.limits.maxComputeWorkGroupCount[0] = test.context.limits.maxComputeWorkGroupCount[1] = test.context.limits.maxComputeWorkGroupCount[2] = 65535u;
        // Host imports as the device's setup enables them: no-copy buffers over whole 16 KiB pages.
        test.context.hostImportAlignment = 16384u;
        cache.Attach(test.context);
    }
    const Context& GetContext() const { return *cache.ImageContext(); }
    void WaitQueue() const {
        if (auto* recorder = Recorder::Active()) recorder->Sync();
    }

private:
    MetalTest::Device test;
    mutable Metal::ImageCache cache;
};

void* AllocateWatched(std::size_t bytes, std::size_t alignment) {
    if (!AgcDriver::GuestMemory::WriteWatched()) return nullptr;
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, alignment);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    if (!AgcDriver::GuestMemory::Watched(reinterpret_cast<std::uint64_t>(block), bytes)) throw std::runtime_error("the watched block is not watched");
    return block;
}

void ReleaseWatched(void* block, std::size_t bytes) {
#ifdef _WIN32
    GuestArena::GuestArenaReset_nid_postfix(block, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(block, bytes);
#else
    munmap(block, bytes);
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
#endif
}

void unitShadowTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    if (!UnitShadowEnabled()) {
        Require(!AnyShadowedOverlaps(0x10000, 16) && PublishShadow(0x10000, 16, PublishScope::Whole, PublishReason::Hook) == 0, "unit shadows are off but not inert");
        std::cout << "unit shadows off (APS5_NO_UNIT_SHADOW, or no write watching): primitives inert\n";
        return;
    }
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: unit shadows not tested\n";
        return;
    }
    constexpr std::uint64_t unit = 65536;
    // Three slabs at the default 8 MiB: units 0..127, 128..255, 256..271.
    constexpr std::size_t bytes = (17u << 20u);
    void* block = AllocateWatched(bytes, 65536);
    Require(block != nullptr, "unit shadows are on without write watching");
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        bool armed = true;
        ~Unregister() {
            if (!armed) return;
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr) {
        std::cout << "host import of the shadow test block refused: unit shadows not tested\n";
        return;
    }
    if (!Watched(address, bytes)) {
        std::cout << "host imports are compared, not watched: unit shadows not tested\n";
        return;
    }
    Require(import->base == address && import->bytes == bytes, "the import does not cover the block");
    CollectWritesUncached(address, bytes);
    const auto* words = static_cast<const std::uint8_t*>(block);
    const auto unit3 = address + 3 * unit;
    const auto unit4 = unit3 + unit;
    const auto unit5 = unit4 + unit;
    // A half-unit piece seeds its unit from the import; a whole-unit piece does not; both land in
    // one slab at one offset.
    auto half = ShadowDestinationFor(context, *import, unit3, unit3 + unit / 2);
    Require(half.has_value() && half->seedUnits.size() == 1 && half->seedUnits[0].first == unit3 && half->seedUnits[0].second == unit4, "a half-unit retile piece does not seed its unit");
    auto whole = ShadowDestinationFor(context, *import, unit3, unit4);
    Require(whole.has_value() && whole->seedUnits.empty() && whole->slab == half->slab && whole->offset == half->offset && whole->buffer == half->buffer, "a whole-unit retile piece seeds, or lands elsewhere");
    Require(!AnyShadowedOverlaps(unit3, unit), "a destination alone counts as shadowed");
    const auto firstSlab = half->slab;
    Require(firstSlab->pins.load() == 2 && half->pin != nullptr, "destinations do not pin their slab");
    // The retile as writeBackWindows records it: the seed, then the piece's bytes over its half.
    const auto copyBuffer = Metal::Vk::CmdCopyBuffer;
    const auto retile = [&](const ShadowDestination& destination, std::uint64_t begin, std::uint64_t length, std::uint8_t value, bool seed) {
        auto pattern = std::make_shared<Buffer>(context, static_cast<std::size_t>(length), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memset(pattern->Bytes().data(), value, pattern->Bytes().size());
        const auto commands = recorder.Commands();
        recorder.Keep(pattern);
        recorder.Keep(destination.slab);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        if (seed) {
            const VkBufferCopy seedCopy{begin - import->base, SlabOffset(*import, *destination.slab, begin), unit};
            copyBuffer(commands, import->buffer, destination.buffer, 1, &seedCopy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        const VkBufferCopy piece{0, SlabOffset(*import, *destination.slab, begin), length};
        copyBuffer(commands, pattern->Handle(), destination.buffer, 1, &piece);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        recorder.MarkCovered(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const ShadowedRange range{begin & ~(unit - 1), (begin & ~(unit - 1)) + unit, destination.slab};
        MarkShadowed(*import, std::span(&range, 1), TrackerGeneration());
    };
    retile(*half, unit3, unit / 2, 0x22, true);
    Require(AnyShadowedOverlaps(unit3, 1) && AnyShadowedOverlaps(unit4 - 1, 1) && AnyShadowedOverlaps(unit3 + 1000, 64), "a fresh unit is not seen as shadowed");
    Require(!AnyShadowedOverlaps(unit3 - 1, 1) && !AnyShadowedOverlaps(unit4, unit) && !AnyShadowedOverlaps(address, 3 * unit), "a range outside the unit is seen as shadowed");
    {
        // An upload's runs over units 2..4 split at unit 3's freshness; the fresh piece reads the slab.
        const std::pair<std::uint64_t, std::uint64_t> run{2 * unit, 5 * unit};
        const auto sources = ShadowSources(context, *import, address, std::span(&run, 1), {});
        Require(sources.size() == 3, "an upload run is not split at the fresh unit");
        Require(!sources[0].shadow && sources[0].begin == 2 * unit && sources[0].end == 3 * unit && sources[0].buffer == import->buffer && sources[0].offset == 2 * unit, "the import piece before the fresh unit is wrong");
        Require(sources[1].shadow && sources[1].begin == 3 * unit && sources[1].end == 4 * unit && sources[1].buffer == half->buffer && sources[1].offset == half->offset && sources[1].slab == half->slab, "the fresh unit's piece does not read the slab");
        Require(!sources[2].shadow && sources[2].begin == 4 * unit && sources[2].end == 5 * unit, "the import piece after the fresh unit is wrong");
    }
    // A whole publish: the untouched half keeps the seed (the import's bytes), the other the
    // retile's; the copy stamps nothing, and a second publish copies nothing.
    const auto pre = CollectWritesUncached(address, bytes);
    Require(PublishShadow(unit3, unit, PublishScope::Whole, PublishReason::Hook) == 1, "the whole publish did not copy the unit");
    Require(recorder.PendingWriteOverlaps(unit3, unit), "the publish noted no pending write");
    recorder.Sync();
    Require(words[3 * unit] == 0x22 && words[3 * unit + unit / 2 - 1] == 0x22 && words[3 * unit + unit / 2] == 0x11 && words[4 * unit - 1] == 0x11 && words[3 * unit - 1] == 0x11 && words[4 * unit] == 0x11, "the published bytes are wrong");
    Require(!AnyShadowedOverlaps(unit3, unit), "a published unit still counts as shadowed");
    Require(PublishShadow(unit3, unit, PublishScope::Whole, PublishReason::Hook) == 0, "a second publish copied the unit again");
    Require(UnchangedSince(unit3, unit, pre), "the publish stamped the unit");
    {
        // The published unit still reads from the slab (both hold the bytes).
        const std::pair<std::uint64_t, std::uint64_t> run{3 * unit, 4 * unit};
        const auto sources = ShadowSources(context, *import, address, std::span(&run, 1), {});
        Require(sources.size() == 1 && sources[0].shadow, "a published unit does not read from the slab");
    }
    // Partial-unit scope over unit 3 (partly, fresh again) and unit 4 (wholly, fresh): unit 3 only.
    retile(*whole, unit3, unit, 0x33, false);
    auto four = ShadowDestinationFor(context, *import, unit4, unit5);
    Require(four.has_value() && four->seedUnits.empty() && four->slab == half->slab, "unit 4's destination is not in the same slab");
    retile(*four, unit4, unit, 0x44, false);
    Require(AnyShadowedOverlaps(unit3, unit) && AnyShadowedOverlaps(unit4, unit), "re-marked units are not shadowed");
    Require(PublishShadow(unit3 + 100, static_cast<std::size_t>(2 * unit - 100), PublishScope::PartialUnits, PublishReason::Fill) == 1, "the partial publish did not copy exactly the partly covered unit");
    Require(!AnyShadowedOverlaps(unit3, unit) && AnyShadowedOverlaps(unit4, unit), "the partial publish copied the wrong unit");
    recorder.Sync();
    Require(words[3 * unit] == 0x33 && words[4 * unit - 1] == 0x33 && words[4 * unit] == 0x11, "the partial publish's bytes are wrong");
    // A CPU write into a page of unit 4 makes it stale: the publish collects the unit itself (no
    // walk saw the store before it), copies nothing and drops it.
    static_cast<void>(CollectWritesUncached(unit4, unit));
    *static_cast<volatile std::uint8_t*>(static_cast<void*>(static_cast<std::uint8_t*>(block) + 4 * unit + 4096)) = 0x55;
    Require(PublishShadow(unit4, unit, PublishScope::Whole, PublishReason::Hook) == 0, "a stale unit was published");
    Require(!AnyShadowedOverlaps(unit4, unit), "a stale unit still counts as shadowed after its publish");
    recorder.Sync();
    Require(words[4 * unit + 4096] == 0x55 && words[4 * unit] == 0x11, "a stale unit's publish overwrote the CPU's bytes");
    // The slab boundary: a piece over units 127 and 128 is refused, each alone lands in its slab.
    const auto unit127 = address + 127 * unit;
    const auto unit128 = unit127 + unit;
    Require(SlabBoundary(*import, unit127) == unit128 && SlabBoundary(*import, address) == unit128 && SlabBoundary(*import, unit128) == unit128 + 128 * unit, "the slab boundary is wrong");
    Require(!ShadowDestinationFor(context, *import, unit127, unit128 + unit).has_value(), "a piece across a slab boundary was accepted");
    auto low = ShadowDestinationFor(context, *import, unit127, unit128);
    Require(low.has_value() && low->slab == firstSlab && low->offset == 127 * unit, "unit 127 does not land in the first slab");
    retile(*low, unit127, unit, 0x66, false);
    auto high = ShadowDestinationFor(context, *import, unit128, unit128 + unit);
    const char* budgetText = std::getenv("APS5_UNIT_SHADOW_MIB");
    const auto budgetMiB = budgetText != nullptr ? std::strtoull(budgetText, nullptr, 10) : 1024ull;
    if (budgetMiB < 16) {
        // One slab fits: while destinations pin the first slab the second is refused (a
        // write-back in progress must keep its slab); with the pins released, the second slab's
        // allocation evicts the first after publishing its fresh unit 127 (the eviction publish is
        // recorded, so the bytes land at the sync).
        Require(!high.has_value(), "a pinned slab was evicted for a second slab");
        half.reset();
        whole.reset();
        four.reset();
        low.reset();
        Require(firstSlab->pins.load() == 0, "a released destination left its pin");
        high = ShadowDestinationFor(context, *import, unit128, unit128 + unit);
        Require(high.has_value() && high->slab != firstSlab && high->offset == 0, "unit 128 does not land in a second slab under the one-slab budget");
        recorder.Sync();
        Require(words[127 * unit] == 0x66 && !AnyShadowedOverlaps(unit127, unit), "the evicted slab's fresh unit was not published");
        std::cout << "unit shadow eviction under APS5_UNIT_SHADOW_MIB=" << budgetMiB << " verified\n";
    } else {
        Require(high.has_value() && high->slab != firstSlab && high->offset == 0 && AnyShadowedOverlaps(unit127, unit), "unit 128 does not land in a second slab beside the first");
        half.reset();
        whole.reset();
        four.reset();
        low.reset();
    }
    retile(*high, unit128, unit, 0x77, false);
    Require(AnyShadowedOverlaps(unit128, unit), "the second slab's unit is not shadowed");
    high.reset();
    retile(*ShadowDestinationFor(context, *import, unit5, unit5 + unit), unit5, unit, 0x88, false);
    retile(*ShadowDestinationFor(context, *import, address + 2 * unit, unit3), address + 2 * unit, unit, 0x99, false);
    Require(AnyShadowedOverlaps(unit5, unit) && AnyShadowedOverlaps(address + 2 * unit, unit), "units 2 and 5 are not shadowed before the retire");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        mutation.Add(block, static_cast<std::size_t>(5 * unit), true, true, true);
    }
    Require(HostImportFor(context, address, bytes) == nullptr, "a shrunk registration still imports the old range");
    Require(!AnyShadowedOverlaps(address, bytes), "the retired import's shadow survived");
    recorder.Sync();
    Require(words[2 * unit] == 0x99 && words[3 * unit - 1] == 0x99, "the retire did not publish the unit still registered");
    Require(words[5 * unit] == 0x88 && words[5 * unit + unit - 1] == 0x88, "the retire did not publish a unit covered by the old registration");
    Require(words[127 * unit] == 0x66 && words[128 * unit] == 0x77, "the retire did not publish the second slab's units covered by the old registration");
}

void hostImportUnmapTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: import unmap lifetime not tested\n";
        return;
    }
    const auto alignment = static_cast<std::size_t>(std::max<VkDeviceSize>(65536u, context.hostImportAlignment));
    constexpr std::size_t RequestedBytes = 17u << 20u;
    const auto bytes = ((RequestedBytes + alignment - 1u) / alignment) * alignment;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(alignment, bytes);
#endif
    Require(block != nullptr, "cannot allocate the host import unmap block");
    struct Block {
        void* address;
        std::size_t bytes;
        bool registered = true;
        bool released = false;
        void Release() {
#ifdef _WIN32
            Require(VirtualFree(address, 0, MEM_RELEASE) != 0, "cannot release the host import unmap block");
#else
            std::free(address);
#endif
            released = true;
        }
        ~Block() {
            if (registered) {
                try {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(address);
                } catch (...) {
                    return;
                }
            }
            if (!released) {
#ifdef _WIN32
                VirtualFree(address, 0, MEM_RELEASE);
#else
                std::free(address);
#endif
            }
        }
    } allocation{block, bytes};
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    const HostImport* imported = nullptr;
    {
        std::lock_guard gpu(GpuMutex());
        imported = HostImportFor(context, address, bytes);
    }
    if (imported == nullptr) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        allocation.registered = false;
        allocation.Release();
        std::cout << "host import refused: import unmap lifetime not tested\n";
        return;
    }
    bool retiredBeforeRelease = false;
    {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(block, bytes, [&](const void*, std::size_t, const void*, bool) {
            retiredBeforeRelease = !HostImportCovers(context, address, bytes);
            if (retiredBeforeRelease) allocation.Release();
        });
        allocation.registered = false;
    }
    if (!retiredBeforeRelease) {
        {
            std::lock_guard gpu(GpuMutex());
            static_cast<void>(HostImportFor(context, address, bytes));
            recorder.Sync();
        }
        allocation.Release();
    }
    Require(retiredBeforeRelease, "guest unmap released pages before retiring their host import");
    std::cout << "host import unmap lifetime: ok\n";
}

void unchangedCpuStampTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::uint64_t offset = 0x2000;
    constexpr std::size_t bytes = 6 * 65536;
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    for (const bool imported : {false, true}) {
#ifdef _WIN32
        if (imported) {
            std::cout << "unchanged CPU stamps over a watched host import: Linux write watch only\n";
            continue;
        }
#endif
        if (imported && base.hostImportAlignment == 0) {
            std::cout << "host imports unavailable: unchanged CPU stamps over a watched host import not tested\n";
            continue;
        }
        const auto decided = PrepareImportWatch(base);
        if (imported) SetImportWatch(base, ImportWatch::Watch);
        struct Restore {
            const Context& context;
            ImportWatch decided;
            ~Restore() { SetImportWatch(context, decided); }
        } restore{base, decided};
        for (const bool changed : {false, true}) {
            const std::string what = std::string(changed ? "a CPU store that changed a byte" : "a CPU store that left the bytes as they were") + (imported ? " over a watched host import" : " over watched memory");
            void* block = AllocateWatched(bytes, 65536);
            if (block == nullptr) {
                std::cout << "no write watching: unchanged CPU stamps not tested\n";
                return;
            }
            auto* memory = static_cast<std::uint8_t*>(block);
            for (std::size_t i = 0; i < bytes; ++i) memory[i] = pattern(i);
            const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
            const auto address = blockAddress + offset;
            {
                GuestAllocations::Mutation mutation;
                mutation.Add(block, bytes, true, true, true);
            }
            struct Unregister {
                const Context& context;
                void* block;
                std::uint64_t address;
                std::size_t bytes;
                ~Unregister() {
                    {
                        GuestAllocations::Mutation mutation;
                        mutation.Remove(block);
                    }
                    HostImportFor(context, address, bytes);
                    ReleaseWatched(block, bytes);
                }
            } unregister{base, block, blockAddress, bytes};
            AgcDriver::GuestMemory::CollectWritesUncached(blockAddress, bytes);
            TextureDetiler detiler(base);
            auto context = base;
            if (!imported) context.hostImportAlignment = 0;
            context.detiler = &detiler;
            GuestTextureResource resource{};
            resource.baseAddress = address;
            resource.width = side;
            resource.height = side;
            resource.mipCount = 1;
            resource.tileMode = TextureTileMode::kLinear;
            resource.dimension = TextureDimension::k2D;
            resource.format = 56;
            resource.dstSelX = 4;
            resource.dstSelY = 5;
            resource.dstSelZ = 6;
            resource.dstSelW = 7;
            Require(DescribeSurface(resource).guestBytes == surfaceBytes, "(c) unexpected surface size");
            {
                auto image = StorageTexture::Create(context, detiler, resource, 0);
                const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                const auto commands = recorder.Commands();
                recorder.Keep(image);
                const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                for (std::size_t i = 0; i < 4096; ++i) memory[offset + i] = pattern(offset + i);
                if (changed) memory[offset + 16] = static_cast<std::uint8_t>(~pattern(offset + 16));
                AgcDriver::GuestMemory::CollectWritesUncached(blockAddress, bytes);
                Require(AgcDriver::GuestMemory::Watched(address, surfaceBytes) && (HostImportFor(context, address, surfaceBytes) != nullptr) == imported, "(c) " + what + ": the image is not on the path under test");
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                std::vector<std::byte> read(bytes);
                AgcDriver::GuestMemory::Read(blockAddress, read);
                const auto texel = [&](std::size_t at) {
                    for (std::size_t c = 0; c < 4; ++c) {
                        if (std::to_integer<std::uint8_t>(read[at + c]) != red[c]) return false;
                    }
                    return true;
                };
                const auto headEnd = static_cast<std::size_t>(65536);
                std::size_t stored = 0, total = 0;
                for (std::size_t at = offset; at < offset + surfaceBytes; at += 4) {
                    ++total;
                    if (texel(at)) ++stored;
                }
                if (!changed) {
                    if (stored != total) throw std::runtime_error(std::string("unchanged CPU stamps: ") + what + " dropped " + std::to_string(total - stored) + " of " + std::to_string(total) + " texels the GPU wrote");
                } else {
                    Require(std::to_integer<std::uint8_t>(read[offset + 16]) == static_cast<std::uint8_t>(~pattern(offset + 16)), "(c) " + what + ": the changed byte was overwritten by the GPU results");
                    for (std::size_t at = headEnd; at < offset + surfaceBytes; at += 4) Require(texel(at), "(c) " + what + ": GPU results outside the block the CPU changed were dropped");
                }
            }
            recorder.Sync();
        }
    }
    std::cout << "unchanged CPU stamps: ok\n";
}

void settledBaselineTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
#ifdef _WIN32
    std::cout << "generation baseline over a pending store: Linux write watch only\n";
    return;
#endif
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: generation baseline over a pending store not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::uint64_t offset = 0x2000;
    constexpr std::size_t bytes = 6 * 65536;
    const auto decided = PrepareImportWatch(base);
    SetImportWatch(base, ImportWatch::Watch);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{base, decided};
    for (const bool pending : {false, true}) {
        void* block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: generation baseline over a pending store not tested\n";
            return;
        }
        std::memset(block, 0x5a, bytes);
        const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
        const auto address = blockAddress + offset;
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(block, bytes, true, true, true);
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                HostImportFor(context, address, bytes);
                ReleaseWatched(block, bytes);
            }
        } unregister{base, block, blockAddress, bytes};
        AgcDriver::GuestMemory::CollectWritesUncached(blockAddress, bytes);
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = side;
        resource.height = side;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kLinear;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto surfaceBytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
        Require(HostImportFor(context, address, surfaceBytes) != nullptr && AgcDriver::GuestMemory::Watched(address, surfaceBytes), "(b) the surface is not a watched host import");
        static_cast<void>(recorder.Commands());
        if (pending) recorder.NotePendingWrite(address + 4096, 4096);
        const auto before = recorder.Submissions();
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            recorder.Keep(image);
            if (pending) Require(recorder.Submissions() == before, "(b) an upload over a pending recorded store submitted the open batch to read its generation baseline");
        }
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::cout << "generation baseline over a pending store: ok\n";
}

// db3af40970 (--idle-publish-wait-only): the idle flush waits for its unit-shadow publish before returning, so a CPU
// store made after it survives; a label flush's publish stays recorded. Upstream's check unchanged over Metal buffers.
void idlePublishWaitTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    constexpr std::uint64_t unit = 65536;
    constexpr std::size_t bytes = 8 * unit;
    void* block = AllocateWatched(bytes, 65536);
    Require(block != nullptr, "unit shadows are on without write watching");
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr || !Watched(address, bytes)) {
        std::cout << "host import of the idle publish test block refused or not watched: idle publish wait not tested\n";
        return;
    }
    recorder.Sync();
    CollectWritesUncached(address, bytes);
    auto* words = static_cast<std::uint8_t*>(block);
    const auto copyBuffer = Metal::Vk::CmdCopyBuffer;
    const auto retile = [&](std::uint64_t begin, std::uint8_t value) {
        auto destination = ShadowDestinationFor(context, *import, begin, begin + unit);
        Require(destination.has_value() && destination->seedUnits.empty(), "a whole-unit retile piece has no destination");
        auto pattern = std::make_shared<Buffer>(context, static_cast<std::size_t>(unit), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memset(pattern->Bytes().data(), value, pattern->Bytes().size());
        const auto commands = recorder.Commands();
        recorder.Keep(pattern);
        recorder.Keep(destination->slab);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkBufferCopy piece{0, destination->offset, unit};
        copyBuffer(commands, pattern->Handle(), destination->buffer, 1, &piece);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        recorder.MarkCovered(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const ShadowedRange range{begin, begin + unit, destination->slab};
        MarkShadowed(*import, std::span(&range, 1), TrackerGeneration());
        Require(AnyShadowedOverlaps(begin, unit), "the retiled unit is not shadowed");
    };
    retile(address + 2 * unit, 0x22);
    retile(address + 3 * unit, 0x33);
    StorageTexture::FlushAllPending("idle");
    Require(!AnyShadowedOverlaps(address + 2 * unit, static_cast<std::size_t>(2 * unit)), "the idle flush did not publish the shadowed units");
    Require(!recorder.PendingWriteOverlaps(address + 2 * unit, static_cast<std::size_t>(2 * unit)), "the idle flush returned before its publish's copies completed");
    words[2 * unit] = 0x44;
    words[4 * unit - 1] = 0x45;
    recorder.Sync();
    Require(words[2 * unit] == 0x44 && words[4 * unit - 1] == 0x45, "a CPU store made after the idle flush returned was overwritten by its publish");
    Require(words[2 * unit + 1] == 0x22 && words[3 * unit - 1] == 0x22 && words[3 * unit] == 0x33 && words[1 * unit] == 0x11 && words[4 * unit] == 0x11, "the idle publish's bytes are wrong");
    retile(address + 5 * unit, 0x55);
    StorageTexture::FlushAllPending("label");
    Require(recorder.PendingWriteOverlaps(address + 5 * unit, unit), "a publish of a flush other than the idle one was waited");
    recorder.Sync();
    Require(words[5 * unit] == 0x55 && words[6 * unit - 1] == 0x55, "the label flush's publish did not land");
    std::cout << "idle publish wait verified\n";
}

template <typename Test>
int Run(Test test, bool unlocked = false, bool shadows = false) {
    @autoreleasepool {
        try {
            Device device;
            if (shadows && (!AgcDriver::Graphics::UnitShadowEnabled() || device.GetContext().hostImportAlignment == 0)) {
                std::cout << "skipped, unit shadows or host imports are unavailable\n";
                return 77;
            }
            std::unique_lock gpu(GpuMutex());
            PrepareImportWatch(device.GetContext());
            Recorder recorder(device.GetContext());
            recorder.Activate();
            if (unlocked) gpu.unlock();
            test(device, recorder);
            recorder.Sync();
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}

const bool registered = [] {
    AgcDriver::MetalTwins::RegisterRecorderMode("--unchanged-cpu-stamps-only", [] { return Run(unchangedCpuStampTests); });
    AgcDriver::MetalTwins::RegisterRecorderMode("--settled-baseline-only", [] { return Run(settledBaselineTests); });
    AgcDriver::MetalTwins::RegisterRecorderMode("--host-import-unmap-only", [] { return Run(hostImportUnmapTests, true); });
    AgcDriver::MetalTwins::RegisterRecorderMode("--unit-shadow-only", [] { return Run(unitShadowTests, false, true); });
    AgcDriver::MetalTwins::RegisterRecorderMode("--idle-publish-wait-only", [] { return Run(idlePublishWaitTests, false, true); });
    return true;
}();

}
