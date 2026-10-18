// Metal twin of tests/Recorder.cpp's --draw-inputs-in-place-only mode (agc_draw_inputs_in_place). The test
// function (drawInputInPlaceTests, Recorder.cpp:2037-2131) is copied unchanged. It checks the draw-input
// primitives of Graphics/src/Draw.cpp (InPlaceDrawInput, InPlaceIndexRange, compiled unchanged on macOS) over
// Metal host imports (GuestBufferMemory.mm: no-copy buffers over 16 KiB pages of write-watched memory) and the
// Metal recorder (Recorder.mm): an input is bound in place only when imported, aligned, recorded and clear of
// pending GPU writes (a vertex input asking for it is bound behind them instead, upstream 13a82bdccf) and queued
// label stores; its read is noted on the open batch and retired with it; the highest
// index of a kept index range is reused until a guest write changes it. Expected values are upstream's.
// Upstream's "not tested" exits (no host imports, no write watching) are requirements here: the Metal device
// has both. Its imports are compared, not watched (DECISIONS D19), so instead of upstream's kept-highest-index
// checks (watched memory only) the twin checks that unwatched index ranges are rescanned at every use.
// The Metal draw path (MetalDraw.mm) reads vertex and index memory through the device's guest page table rather
// than through these primitives (TechnicalDebt metal-vulkan-internal-tests).

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

namespace {

void drawInputInPlaceTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    constexpr std::size_t bytes = 65536;
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    Require(block != nullptr, "host imports or write watching unavailable: in-place draw inputs not tested");
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    Require(HostImportFor(context, address, bytes) != nullptr, "host import of the watched block refused: in-place draw inputs not tested");
    context.recorder = &recorder;
    recorder.Sync();
    const auto vertices = address + 4096;
    const auto* import = InPlaceDrawInput(context, vertices, 256, 4);
    Require(import != nullptr && import->base <= vertices && vertices + 256 <= import->base + import->bytes, "an imported draw input was not bound in place");
    const auto read = recorder.DescribePendingRead(vertices, 256);
    Require(read.has_value() && read->kind == Recorder::ReadKind::DrawInput && read->open, "an in-place draw input's read was not noted on the open batch");
    Require(InPlaceDrawInput(context, vertices + 2, 256, 4) == nullptr, "a draw input misaligned for its format was bound in place");
    Require(InPlaceDrawInput(context, vertices + 2, 256, 2) != nullptr, "an aligned 16-bit draw input was not bound in place");
    recorder.NotePendingWrite(address + 8192, 16);
    Require(InPlaceDrawInput(context, address + 8192, 256, 4) != nullptr && !recorder.PendingWriteOverlaps(address + 8192, 256), "a draw input under a pending GPU write was bound before the write completed");
    bool behind = false;
    Require(InPlaceDrawInput(context, address + 20480, 256, 4, &behind) != nullptr && !behind, "a vertex input free of pending GPU writes was marked as ordered behind one");
    recorder.NotePendingWrite(address + 20480, 16);
    Require(InPlaceDrawInput(context, address + 20480, 256, 4, &behind) != nullptr && behind && recorder.PendingWriteOverlaps(address + 20480, 256), "a vertex input under a pending GPU write was not bound in place behind it");
    const std::array<std::byte, 4> label{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    recorder.RecordStore(import->buffer, address + 12288 - import->base, label, address + 12288);
    Require(recorder.QueuedStoreOverlaps(address + 12288, 4), "the label store was not queued");
    Require(InPlaceDrawInput(context, address + 12288, 256, 4) == nullptr, "a draw input under a queued label store was bound in place");
    behind = false;
    Require(InPlaceDrawInput(context, address + 12288, 256, 4, &behind) == nullptr, "a vertex input under a queued label store was bound in place");
    Require(InPlaceDrawInput(context, address + 16384, 256, 4) != nullptr, "a queued label store refused a disjoint draw input");
    alignas(64) static std::array<std::byte, 256> unregistered{};
    Require(InPlaceDrawInput(context, reinterpret_cast<std::uint64_t>(unregistered.data()), unregistered.size(), 4) == nullptr, "memory without a host import was bound in place");
    auto unrecorded = context;
    unrecorded.recorder = nullptr;
    Require(InPlaceDrawInput(unrecorded, vertices, 256, 4) == nullptr, "a draw without a recorder bound its input in place");
    recorder.Sync();
    Require(!recorder.PendingReadOverlaps(vertices, 256), "an in-place draw input's read outlived its batch");
    if (!AgcDriver::GuestMemory::Watched(address, bytes)) {
        // Metal host imports are compared, not watched (DECISIONS D19): the import unwatched the block, so no write
        // generation proves a kept highest index current. Every use rescans, and sees guest writes.
        const auto indices = address + 20480;
        constexpr std::size_t kept = Recorder::KeptIndexRangeIndices * 2;
        std::vector<std::uint16_t> first(Recorder::KeptIndexRangeIndices * 2, 1);
        first[0] = 3;
        first[1] = 9;
        first[2] = 0xffff;
        std::memcpy(reinterpret_cast<void*>(indices), first.data(), kept * 2);
        auto range = InPlaceIndexRange(recorder, indices, kept, 2, false);
        Require(!range.reused && range.highest == 0xffffu, "an in-place index range was not scanned at its first use");
        range = InPlaceIndexRange(recorder, indices, kept, 2, false);
        Require(!range.reused && range.highest == 0xffffu, "an in-place index range of unwatched memory was kept");
        range = InPlaceIndexRange(recorder, indices, kept, 2, true);
        Require(!range.reused && range.highest == 9u, "an in-place index range under the restart rule has the wrong highest index");
        const std::uint16_t raised = 12;
        std::memcpy(reinterpret_cast<void*>(indices + 2), &raised, sizeof(raised));
        range = InPlaceIndexRange(recorder, indices, kept, 2, true);
        Require(!range.reused && range.highest == 12u, "an in-place index range of unwatched memory missed a guest write");
        range = InPlaceIndexRange(recorder, indices, kept * 2, 4, false);
        Require(!range.reused && range.highest == 0x000c0003u, "a 32-bit in-place index range has the wrong highest index");
        std::cout << "host imports are compared, not watched: kept highest indices are rescanned at every use\n";
        return;
    }
    const auto indices = address + 20480;
    constexpr std::size_t kept = Recorder::KeptIndexRangeIndices * 2;
    std::vector<std::uint16_t> first(Recorder::KeptIndexRangeIndices * 2, 1);
    first[0] = 3;
    first[1] = 9;
    first[2] = 0xffff;
    std::memcpy(reinterpret_cast<void*>(indices), first.data(), kept * 2);
    auto range = InPlaceIndexRange(recorder, indices, kept, 2, false);
    Require(!range.reused && range.highest == 0xffffu, "an in-place index range was not scanned at its first use");
    range = InPlaceIndexRange(recorder, indices, kept, 2, false);
    Require(range.reused && range.highest == 0xffffu, "an unchanged in-place index range was scanned again");
    range = InPlaceIndexRange(recorder, indices, kept, 2, true);
    Require(!range.reused && range.highest == 9u, "an in-place index range served a scan made under the other restart rule");
    Require(!InPlaceIndexRange(recorder, indices, kept + 2, 2, true).reused, "an in-place index range served a scan of another size");
    const std::uint16_t raised = 12;
    std::memcpy(reinterpret_cast<void*>(indices + 2), &raised, sizeof(raised));
    range = InPlaceIndexRange(recorder, indices, kept, 2, true);
    Require(!range.reused && range.highest == 12u, "an in-place index range kept its highest index over a guest write");
    range = InPlaceIndexRange(recorder, indices, kept, 2, true);
    Require(range.reused && range.highest == 12u, "a rescanned in-place index range was not kept");
    range = InPlaceIndexRange(recorder, indices, kept - 2, 2, true);
    Require(!range.reused && range.highest == 12u, "an in-place index range below the kept size was not scanned");
    range = InPlaceIndexRange(recorder, indices, kept - 2, 2, true);
    Require(!range.reused && range.highest == 12u, "an in-place index range below the kept size was kept");
    range = InPlaceIndexRange(recorder, indices, kept * 2 - 4, 4, false);
    Require(!range.reused && range.highest == 0x000c0003u, "a 32-bit in-place index range below the kept size was not scanned");
    Require(!InPlaceIndexRange(recorder, indices, kept * 2 - 4, 4, false).reused, "a 32-bit in-place index range below the kept size was kept");
    Require(!InPlaceIndexRange(recorder, indices, kept * 2, 4, false).reused, "a 32-bit in-place index range was not scanned at its first use");
    range = InPlaceIndexRange(recorder, indices, kept * 2, 4, false);
    Require(range.reused && range.highest == 0x000c0003u, "an unchanged 32-bit in-place index range was scanned again");
    const std::vector<std::uint16_t> restarts(Recorder::KeptIndexRangeIndices, 0xffff);
    std::memcpy(reinterpret_cast<void*>(indices + 20480), restarts.data(), kept);
    Require(!InPlaceIndexRange(recorder, indices + 20480, kept, 2, true).highest.has_value(), "an index range of restart indices has a highest index");
    range = InPlaceIndexRange(recorder, indices + 20480, kept, 2, true);
    Require(range.reused && !range.highest.has_value(), "an empty in-place index range was scanned again");
}

const bool registered = AgcDriver::MetalTwins::RegisterRecorderMode("--draw-inputs-in-place-only", [] {
    return RunMode([](const Device& device, Recorder& recorder) {
        drawInputInPlaceTests(device, recorder);
        std::cout << "In-place draw input tests passed\n";
    });
});

}
