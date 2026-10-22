// Metal twins of the tests/Recorder.cpp modes upstream added between 721db55dc and 5d9dcc42d2 (UPSYNC3, second
// round):
//   --import-watch-only   agc_import_watch    importWatchTests (the non-Windows half)
//   --sample-dumps-only   agc_sample_dumps    sampleDumpTests + queuedSampleDumpTests
// (--import-watch-other-family-only, agc_import_watch_other_family, is registered on Windows only.)
// The test functions are upstream's, expected values included, but for:
// - importWatchTests: upstream first runs ProbeImportWriteProtection, the GPU-store-into-protected-pages probe,
//   and requires it to succeed. The Metal device never probes (DECISIONS D19, TechnicalDebt
//   metal-d3-import-watch: the macOS write watch is mprotect-based and the probe would store into protected pages
//   on purpose), so the twin checks that the probe refuses with its documented reason instead, then runs
//   upstream's decision checks unchanged: an import under the Unwatch decision leaves the watch, a remap and a
//   re-import follow the decision, and an import under the Watch decision stays watched. The dma-buf probe is
//   Linux only (context.dmaBufImport is false here).
// - sampleDumpTests: upstream ends with inPassSampleDumpTests over CoverPass, a Vulkan render pass and pipeline
//   built from SPIR-V; the Metal recorder's in-pass occlusion counting and its dumps are what agc_metal_recorder's
//   OcclusionSamples and OcclusionPasses check, so the twin stops after queuedSampleDumpTests as upstream does
//   for a device without graphics.
// - `Device`, AllocateWatched and ReleaseWatched come from RecorderTwinHarness.h (host imports at 16 KiB, so
//   upstream's "host imports unavailable" exits never trigger and are kept for parity).
// The mode wrappers do what upstream main does around each mode (RunMode: GPU mutex, import watch, an active
// recorder).

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include <sys/mman.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void importWatchTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0 || !WriteWatched()) {
        std::cout << "host imports or write watching unavailable: import watch decisions not tested\n";
        return;
    }
    // Metal: the probe refuses (D19) instead of measuring; upstream requires probe.failure == nullptr here.
    const auto probe = ProbeImportWriteProtection(context);
    Require(probe.failure != nullptr && std::string(probe.failure).find("not implemented") != std::string::npos, "(u) the Metal import probe did not refuse as D19 documents");
    Require(PrepareImportWatch(context) == ImportWatch::Unwatch, "(u) Metal host imports stayed watched by default");
    std::cout << "import probe: not run on Metal (DECISIONS D19: imported ranges are compared)\n";
    std::cout << "dma-buf imports unavailable: the dma-buf import probe not tested\n";
    const auto decided = PrepareImportWatch(context);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{context, decided};
    constexpr std::size_t unit = 65536;
    const auto remap = [](void* block, std::size_t bytes) {
        munmap(block, bytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
        Require(mmap(block, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == block, "(u) cannot map the test block again");
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
    };
    const auto registerRange = [](void* block, std::size_t bytes) {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    };
    const auto unregisterRange = [](void* block) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    };
    SetImportWatch(context, ImportWatch::Unwatch);
    {
        constexpr std::size_t bytes = 3 * unit;
        void* block = AllocateWatched(bytes, unit);
        const auto address = reinterpret_cast<std::uint64_t>(block);
        std::memset(block, 0x11, bytes);
        const auto before = CollectWrites(address, bytes);
        Require(before != 0, "(u) a watched block is not collected");
        BumpCollectEpoch();
        Require(CollectWrites(address, 2 * unit) != 0, "(u) the memoized collect of a watched block failed");
        registerRange(block, 2 * unit);
        const auto* import = HostImportFor(context, address, 2 * unit);
        Require(import != nullptr && import->unwatched, "(u) an import made under the unwatch decision is not marked unwatched");
        Require(!Watched(address, 2 * unit) && !Watched(address + unit, 4096), "(u) an imported range stays watched");
        Require(CollectWrites(address, 2 * unit) == 0 && CollectWrites(address + 4096, 4096) == 0, "(u) a collect memoized before the import answers for the unwatched range");
        Require(CollectWrites(address + 2 * unit - 4096, 8192) == 0, "(u) a range partly imported is collected");
        Require(Watched(address + 2 * unit, unit), "(u) the unimported neighbour left the watch");
        const auto neighbour = CollectWrites(address + 2 * unit, unit);
        Require(neighbour != 0, "(u) the unimported neighbour is not collected");
        static_cast<volatile std::uint8_t*>(block)[4096] = 0x22;
        Require(!UnchangedSince(address, 2 * unit, before) && !UnchangedSince(address, 4096, TrackerGeneration()) && !UnchangedSinceCollected(address, 4096, TrackerGeneration()), "(u) an unwatched range reports unchanged");
        const std::array<std::uint64_t, 2> generations{before, TrackerGeneration()};
        std::array<std::uint8_t, 2> changed{};
        std::array<std::uint8_t, 2> cpu{};
        Require(!ChangedBlocks(address, 2 * unit, generations, changed, cpu) && changed[0] != 0 && changed[1] != 0, "(u) the stamps of an unwatched range are offered as tracked");
        Require(MarkWritten(address, unit) == 0, "(u) a GPU write into an unwatched range claims a generation");
        Require(MarkWritten(address + 2 * unit, 4096) != 0 && !UnchangedSince(address + 2 * unit, unit, neighbour), "(u) a GPU write into the watched neighbour is not stamped");
        unregisterRange(block);
        remap(block, 2 * unit);
        Require(Watched(address, 2 * unit), "(u) a remapped range is not watched");
        registerRange(block, 2 * unit);
        const auto* kept = HostImportFor(context, address, 2 * unit);
        Require(kept != nullptr && kept->unwatched && !Watched(address, 2 * unit) && CollectWrites(address, 2 * unit) == 0, "(u) an import kept over a remap left the range watched");
        unregisterRange(block);
        remap(block, 2 * unit);
        registerRange(block, unit);
        const auto* again = HostImportFor(context, address, unit);
        Require(again != nullptr && again->unwatched && again->bytes == unit && !Watched(address, unit), "(u) a re-import after an unmap did not follow the decision");
        Require(Watched(address + unit, unit) && CollectWrites(address + unit, unit) != 0, "(u) the unregistered rest of a remapped range is not watched");
        unregisterRange(block);
        Require(HostImportFor(context, address, unit) == nullptr, "(u) an unregistered range still imports");
        ReleaseWatched(block, bytes);
    }
    SetImportWatch(context, ImportWatch::Watch);
    {
        constexpr std::size_t bytes = unit;
        void* block = AllocateWatched(bytes, unit);
        const auto address = reinterpret_cast<std::uint64_t>(block);
        registerRange(block, bytes);
        const auto* import = HostImportFor(context, address, bytes);
        Require(import != nullptr && !import->unwatched && Watched(address, bytes), "(u) an import made under the watch decision left the watch");
        const auto generation = CollectWrites(address, bytes);
        Require(generation != 0 && UnchangedSince(address, bytes, generation), "(u) a watched import is not collected");
        Require(MarkWritten(address, 4096) != 0 && !UnchangedSince(address, bytes, generation), "(u) a GPU write into a watched import is not stamped");
        unregisterRange(block);
        HostImportFor(context, address, bytes);
        ReleaseWatched(block, bytes);
    }
}

constexpr std::uint64_t SampleReady = 1ull << 63u;

void queuedSampleDumpTests(Recorder& recorder, const HostImport& import, std::uint64_t* words, std::uint64_t address, VkDeviceAddress target) {
    constexpr std::uint64_t untouched = 0xaaaaaaaaaaaaaaaaull;
    const auto storeZero = [&](std::uint64_t at, std::size_t bytes) {
        const std::vector<std::byte> zero(bytes);
        recorder.RecordStore(import.buffer, at - import.base, zero, at);
    };
    recorder.Sync();
    const auto base = recorder.SamplesTotal();
    std::fill(words, words + 256, untouched);
    Require(recorder.DumpSamples(target, address), "a dump was not queued");
    Require(recorder.HasQueuedStores() && recorder.QueuedStoreOverlaps(address + 240, 8) && !recorder.QueuedStoreOverlaps(address + 248, 8), "a queued dump does not cover its 16 counters");
    recorder.FlushStoresOverlapping(address + 256, 64);
    Require(recorder.HasQueuedStores(), "a reader of other bytes recorded the queued dump");
    Require(words[0] == untouched, "a queued dump landed before anything recorded it");
    recorder.FlushStoresOverlapping(address + 16, 8);
    Require(!recorder.HasQueuedStores(), "a reader of the dumped bytes left the dump queued");
    recorder.Sync();
    Require(words[0] == (base | SampleReady) && words[2] == SampleReady && words[1] == untouched, "the dump a reader recorded stored the wrong counters");
    std::fill(words, words + 256, untouched);
    storeZero(address, 256);
    Require(recorder.DumpSamples(target, address), "a dump after a clear was not queued");
    storeZero(address + 512, 16);
    Require(recorder.DumpSamples(target + 8, address + 8), "the end dump was not queued");
    Require(recorder.DumpSamples(target + 256, address + 256), "a dump of another slot was not queued");
    storeZero(address + 256, 16);
    Require(recorder.DumpSamples(target + 256 + 8, address + 256 + 8), "a dump after a store over an earlier dump was not queued");
    recorder.Sync();
    Require(words[0] == (base | SampleReady) && words[1] == (base | SampleReady) && words[2] == SampleReady && words[3] == SampleReady, "a clear and its dumps landed out of order");
    Require(words[32] == 0 && words[34] == SampleReady, "a store after a dump of its bytes did not land last");
    Require(words[33] == (base | SampleReady) && words[35] == SampleReady, "a dump after a store over an earlier dump landed before it");
    Require(words[64] == 0 && words[65] == 0 && words[66] == untouched, "a store between dumps was lost");
}

void sampleDumpTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0 || !context.bufferDeviceAddress) {
        std::cout << "host imports or buffer device addresses unavailable: occlusion counter dumps on the GPU not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
    void* block = std::aligned_alloc(65536, bytes);
    Require(block != nullptr, "cannot allocate the occlusion counter block");
    auto* words = static_cast<std::uint64_t*>(block);
    constexpr std::uint64_t untouched = 0xaaaaaaaaaaaaaaaaull;
    std::fill(words, words + 64, untouched);
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
            std::free(block);
        }
    } unregister{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr || import->address == 0) {
        std::cout << "host import of the occlusion counter block refused: occlusion counter dumps on the GPU not tested\n";
        return;
    }
    const auto target = import->address + (address - import->base);
    recorder.Sync();
    constexpr std::uint64_t ready = 1ull << 63u;
    Require(recorder.DumpSamples(target, address), "the occlusion counters were not dumped on the GPU");
    Require(words[0] == untouched && !recorder.Idle(), "the occlusion counter dump waited for the GPU or landed before its batch ran");
    recorder.Sync();
    const auto begin = recorder.SamplesTotal();
    for (std::size_t db = 0; db < 16; ++db) {
        Require(words[db * 2] == ((db == 0 ? begin : 0) | ready), "an occlusion counter dump stored the wrong value");
        Require(words[db * 2 + 1] == untouched, "an occlusion counter dump stored over the next counter");
    }
    Require(recorder.DumpSamples(target + 8, address + 8), "the second occlusion counter dump was not made on the GPU");
    recorder.Submit();
    recorder.Sync();
    for (std::size_t db = 0; db < 16; ++db) {
        Require(words[db * 2 + 1] == ((db == 0 ? begin : 0) | ready), "the counters changed with nothing drawn between two dumps");
        Require(words[db * 2] == ((db == 0 ? begin : 0) | ready), "the second dump stored over the first");
    }
    Require(recorder.SamplesTotal() == begin, "the sample total moved with nothing drawn");
    for (int i = 0; i < 40; ++i) Require(recorder.DumpSamples(target, address), "a dump past one batch's query slots was not made on the GPU");
    recorder.Sync();
    Require(words[0] == (begin | ready) && recorder.SamplesTotal() == begin, "dumps past one batch's query slots moved the total");
    Require(recorder.QueuesSampleDumps(), "occlusion dumps are recorded at once, not queued with the batch's stores");
    queuedSampleDumpTests(recorder, *import, words, address, target);
    std::cout << "occlusion dumps inside a render pass: agc_metal_recorder (OcclusionSamples, OcclusionPasses) checks them on Metal\n";
}

const bool registered = [] {
    using AgcDriver::MetalTwins::RegisterRecorderMode;
    RegisterRecorderMode("--import-watch-only", [] { return RunMode([](const Device& device, Recorder&) { importWatchTests(device); }); });
    RegisterRecorderMode("--sample-dumps-only", [] {
        return RunMode([](const Device& device, Recorder& recorder) {
            sampleDumpTests(device, recorder);
            std::cout << "Occlusion counter dump tests passed\n";
        });
    });
    return true;
}();

}
