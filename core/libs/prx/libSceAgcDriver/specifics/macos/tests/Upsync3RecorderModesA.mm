// Metal twins of tests/Recorder.cpp modes upstream added after 0214069e3 (UPSYNC3):
//   --read-list-only              agc_recorder_read_list        readListTests
//   --space-pending-writes-only   agc_space_pending_writes      overlappedRangeTests + spaceWriteSyncTests
//   --pending-write-snapshot-only agc_pending_write_snapshot    pendingWriteSnapshotTests
//   --queued-write-groups-only    agc_queued_write_groups       queuedWriteGroupTests (no device, as upstream)
//   --pending-key-stores-only     agc_pending_key_stores        pendingKeyStoreTests
//   --shadowed-key-stores-only    agc_shadowed_key_stores       shadowedKeyStoreTests (false: skip, 77)
//   --kept-unit-settle-only       agc_storage_kept_unit_settle  keptUnitSettleTests
// None of the ctest registrations sets an environment variable (only Windows' PATH) or needs a shader cache.
// The test functions are copied from upstream unchanged, expected values included, but for:
// - their Vulkan calls: vkCmdClearColorImage -> Metal::Vk::CmdClearColorImage (keptUnitSettleTests),
//   vkCmdCopyBuffer -> Metal::Vk::CmdCopyBuffer (shadowedKeyStoreTests' slab copy), as in D3RecorderModes.mm;
// - `Device`, AllocateWatched and ReleaseWatched, which come from RecorderTwinHarness.h (the Metal test device
//   with the image cache attached; host imports enabled at 16 KiB, so upstream main's "host imports are
//   unavailable" skips never trigger here and are kept only for parity).
// No page-size constant needed adapting: every block, unit and range in these tests is a multiple of 64 KiB
// (the unit-shadow and 64 KiB tile unit, Metal's UnitShadow UnitBytes too), which whole 16 KiB pages divide;
// upstream's 128-byte key edge sits inside one page either way. The _WIN32 branches stay as upstream wrote them
// (not compiled here). The mode wrappers do what upstream main does around each mode (RunMode: GPU mutex,
// import watch, an active recorder); --queued-write-groups-only runs before any device, as upstream's does.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include <sys/mman.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
}

namespace {

using Kind = Recorder::ReadKind;

Recorder::WriteRanges expectedWriteUnion(Recorder::WriteRanges ranges) {
    std::sort(ranges.begin(), ranges.end());
    Recorder::WriteRanges result;
    for (const auto& [begin, end] : ranges) {
        if (!result.empty() && begin <= result.back().second) result.back().second = std::max(result.back().second, end);
        else result.emplace_back(begin, end);
    }
    return result;
}

void pendingWriteSnapshotTests(const Device& device, Recorder& recorder) {
    recorder.Sync();
    Recorder::WriteRanges expected;
    const auto check = [&] {
        const auto snapshot = Recorder::PendingWriteSnapshot();
        Require(snapshot != nullptr && *snapshot == expectedWriteUnion(expected), "pending write snapshot differs from the full union");
    };
    const auto note = [&](std::uint64_t address, std::size_t bytes) {
        const auto previous = Recorder::PendingWriteSnapshot();
        const auto saved = previous == nullptr ? Recorder::WriteRanges{} : *previous;
        recorder.NotePendingWrite(address, bytes);
        if (bytes != 0) expected.emplace_back(address, address + bytes);
        check();
        Require(previous == nullptr || *previous == saved, "publishing a new write modified an older reader's snapshot");
    };
    note(0x10000, 16);
    note(0x10020, 16);
    note(0x10010, 16);
    note(0x10004, 4);
    note(0x10000, 0);
    note(0, 1);
    note(UINT64_MAX - 16, 16);
    const Recorder::WriteRanges additions{{0x20020, 0x20040}, {0x20000, 0x20020}, {0x20010, 0x20030}, {0x20040, 0x20040}, {0x20050, 0x20040}};
    recorder.NotePendingWrites(additions);
    for (const auto& range : additions) if (range.second > range.first) expected.push_back(range);
    check();
    recorder.NotePendingFill(0x30000, 16, 0xA5);
    expected.emplace_back(0x30000, 0x30010);
    check();
    alignas(64) static std::array<std::byte, 64> labels{};
    const auto labelAddress = reinterpret_cast<std::uint64_t>(labels.data());
    const std::array value{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    recorder.NoteLabel(labelAddress, value, 1, 0);
    expected.emplace_back(labelAddress, labelAddress + value.size());
    check();
    recorder.Submit();
    device.WaitQueue();
    recorder.AfterCompletions(labelAddress + 8, value, 2, 0, true);
    expected.emplace_back(labelAddress + 8, labelAddress + 8 + value.size());
    check();
    note(0x40000, 16);
    bool completing = false;
    recorder.OnComplete([&] {
        const auto snapshot = Recorder::PendingWriteSnapshot();
        Require(Recorder::SnapshotOverlaps(snapshot.get(), 0x40000, 16), "a finishing batch left the snapshot before its completion");
        completing = true;
    });
    recorder.Sync();
    Require(completing && Recorder::PendingWriteSnapshot()->empty(), "completed writes remained published");
    std::mt19937_64 random(0x41505335u);
    for (int trial = 0; trial < 16; ++trial) {
        expected.clear();
        for (int step = 0; step < 64; ++step) note(0x80000 + random() % 4096, 1 + random() % 128);
        recorder.Sync();
        Require(Recorder::PendingWriteSnapshot()->empty(), "random writes survived their GPU batch");
    }
    expected.clear();
    note(0x90000, 16);
    {
        std::atomic<bool> invalid{false};
        std::atomic<unsigned> ready{0};
        std::atomic<std::uint64_t> reads{0};
        std::array<std::jthread, 2> readers;
        for (auto& reader : readers) reader = std::jthread([&](std::stop_token stop) {
            ready.fetch_add(1, std::memory_order_release);
            while (!stop.stop_requested()) {
                const auto snapshot = Recorder::PendingWriteSnapshot();
                if (snapshot == nullptr || !Recorder::SnapshotOverlaps(snapshot.get(), 0x90000, 16)) invalid.store(true, std::memory_order_relaxed);
                if (snapshot != nullptr) {
                    for (std::size_t i = 0; i < snapshot->size(); ++i) {
                        const auto [begin, end] = (*snapshot)[i];
                        if (end <= begin || (i != 0 && (*snapshot)[i - 1].second >= begin)) invalid.store(true, std::memory_order_relaxed);
                    }
                }
                reads.fetch_add(1, std::memory_order_relaxed);
            }
        });
        while (ready.load(std::memory_order_acquire) != readers.size()) std::this_thread::yield();
        for (std::uint64_t i = 0; i < 256; ++i) note(0x91000 + i * 128, 32);
        for (auto& reader : readers) reader.request_stop();
        for (auto& reader : readers) reader.join();
        Require(reads.load() != 0 && !invalid.load(), "concurrent snapshot readers observed an invalid or incomplete union");
    }
    recorder.Sync();
    expected.clear();
    note(UINT64_MAX - 3, 8);
    note(0x50000, 16);
    recorder.Sync();
    recorder.NotePendingWrite(0x58000, 16);
    {
        Recorder other(device.GetContext());
        other.Activate();
        recorder.NotePendingWrite(0x59000, 16);
        recorder.Activate();
        recorder.NotePendingWrite(0x5A000, 16);
        Require(*Recorder::PendingWriteSnapshot() == Recorder::WriteRanges{{0x58000, 0x58010}, {0x59000, 0x59010}, {0x5A000, 0x5A010}}, "reactivation missed writes recorded while inactive");
        other.Activate();
        recorder.Sync();
        recorder.Activate();
        recorder.NotePendingWrite(0x5B000, 16);
        Require(*Recorder::PendingWriteSnapshot() == Recorder::WriteRanges{{0x5B000, 0x5B010}}, "reactivation retained writes completed while inactive");
        recorder.Sync();
        other.Activate();
        other.NotePendingWrite(0x60000, 16);
        const auto otherSnapshot = Recorder::PendingWriteSnapshot();
        recorder.Activate();
        recorder.NotePendingWrite(0x70000, 16);
        const auto ownSnapshot = Recorder::PendingWriteSnapshot();
        Require(ownSnapshot != nullptr && *ownSnapshot == Recorder::WriteRanges{{0x70000, 0x70010}}, "a recorder extended another recorder's snapshot");
        Require(*otherSnapshot == Recorder::WriteRanges{{0x60000, 0x60010}}, "a replaced owner's retained snapshot changed");
        other.Sync();
        Require(Recorder::PendingWriteSnapshot() == ownSnapshot, "an inactive recorder replaced the active snapshot");
        recorder.Sync();
    }
    Require(Recorder::PendingWriteSnapshot()->empty(), "snapshot ownership tests left pending writes");
    std::cout << "Pending write snapshot differential, lifetime, completion and ownership tests passed\n";
}

void readListTests(const Device& device, Recorder& recorder) {
    using Ranges = Recorder::ReadList;
    Require(Recorder::ValidReadList({{0x1000, 0x2000}, {0x2000, 0x2100}}) && Recorder::ValidReadList({}), "a sorted disjoint list is refused");
    Require(!Recorder::ValidReadList({{0x2000, 0x2100}, {0x1000, 0x2000}}) && !Recorder::ValidReadList({{0x1000, 0x2001}, {0x2000, 0x2100}}) && !Recorder::ValidReadList({{0x1000, 0x1000}}), "an unsorted, overlapping or empty list is accepted");
    std::uint64_t seed = 0x9e3779b97f4a7c15ull;
    const auto next = [&](std::uint64_t bound) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return (seed >> 33u) % bound;
    };
    const auto makeList = [&](std::uint64_t base, std::size_t count) {
        auto list = std::make_shared<Ranges>();
        auto at = base;
        for (std::size_t index = 0; index < count; ++index) {
            at += next(3) * 0x100;
            const auto bytes = (next(16) + 1) * 0x10;
            list->emplace_back(at, at + bytes);
            at += bytes;
        }
        return list;
    };
    const auto first = makeList(0x100000, 300);
    const auto second = makeList(0x100000, 120);
    const Ranges before{{0x100400, 0x100410}, {0x200000, 0x200100}};
    const Ranges between{{0x100800, 0x100900}, {0x300000, 0x300040}};
    const Ranges after{{0x100000, 0x180000}};
    const auto note = [&](bool listed) {
        recorder.NotePendingReads(before, Kind::DispatchElement);
        if (listed) recorder.NotePendingReadList(first, Kind::AddressBased);
        else recorder.NotePendingReads(*first, Kind::AddressBased);
        recorder.NotePendingReads(between, Kind::GpuCopy);
        if (listed) {
            recorder.NotePendingReadList(second, Kind::Indirect);
            recorder.NotePendingReadList(first, Kind::AddressBased);
        } else {
            recorder.NotePendingReads(*second, Kind::Indirect);
        }
        recorder.NotePendingReads(after, Kind::CopySource);
    };
    std::vector<std::pair<std::uint64_t, std::uint64_t>> queries;
    for (const auto* list : {first.get(), second.get()}) {
        for (const auto& [begin, end] : *list) {
            for (const auto at : {begin - 1, begin, end - 1, end}) queries.emplace_back(at, 1 + next(0x80));
        }
    }
    for (int query = 0; query < 2000; ++query) queries.emplace_back(0xff000 + next(0x300000), 1 + next(0x2000));
    const auto answers = [&] {
        std::vector<int> result;
        for (const auto& [address, bytes] : queries) {
            const auto described = recorder.DescribePendingRead(address, bytes);
            result.push_back(recorder.PendingReadOverlaps(address, bytes) ? 1 : 0);
            result.push_back(described.has_value() ? 10 + static_cast<int>(described->kind) : -1);
        }
        recorder.Submit();
        for (const auto& [address, bytes] : queries) result.push_back(recorder.PendingReadOverlaps(address, bytes, false) ? 1 : 0);
        recorder.Sync();
        Require(recorder.Idle() && !recorder.PendingReadOverlaps(0x100000, 0x200000, false), "listed reads outlived their batch");
        return result;
    };
    const auto notedBefore = Recorder::ReadCounts().noted;
    note(false);
    const auto plain = answers();
    const auto notedPlain = Recorder::ReadCounts().noted - notedBefore;
    note(true);
    const auto listed = answers();
    const auto notedListed = Recorder::ReadCounts().noted - notedBefore - notedPlain;
    Require(plain == listed, "reads noted by reference answer a query differently from the same ranges noted one by one");
    Require(notedListed == notedPlain + first->size(), "a list's ranges are not counted as noted");
    for (const auto kind : {Kind::DispatchElement, Kind::AddressBased, Kind::CopySource}) Require(std::count(plain.begin(), plain.end(), 10 + static_cast<int>(kind)) != 0, "the queries miss a reader kind: the comparison proves too little");
    recorder.NotePendingReadList(std::make_shared<Ranges>(), Kind::AddressBased);
    recorder.NotePendingReadList(nullptr, Kind::AddressBased);
    Require(!recorder.Recording(), "an empty list opened a batch");
    static_cast<void>(device);
}

struct TestRange {
    std::uint64_t begin;
    std::uint64_t end;
};

void overlappedRangeTests(Recorder& recorder) {
    Require(recorder.Idle(), "the recorder is busy before the overlap tests");
    recorder.NotePendingWrite(0x210000, 0x100);
    recorder.NotePendingWrite(0x210400, 0x10);
    recorder.Submit();
    recorder.NotePendingWrite(0x210800, 0x200);
    recorder.NotePendingWrite(0x210900, 0x200);
    const std::array<TestRange, 10> ranges{{{0x20f000, 0x210000}, {0x210000, 0x210080}, {0x210080, 0x210400}, {0x210400, 0x210410}, {0x210410, 0x210800}, {0x210800, 0x210801}, {0x210801, 0x210a00}, {0x210a00, 0x210b00}, {0x210b00, 0x211000}, {0x300000, 0x300010}}};
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    recorder.PendingWriteRanges(writes);
    Require(writes.size() == 4, "the pending write ranges miss a batch's writes");
    std::vector<std::size_t> indices;
    std::size_t matched = 0;
    for (const std::size_t first : {0u, 3u, 6u, 10u}) {
        OverlappedRanges(std::span<const TestRange>(ranges), first, writes, indices);
        Require(std::is_sorted(indices.begin(), indices.end()) && std::adjacent_find(indices.begin(), indices.end()) == indices.end(), "the overlapped ranges are not sorted and unique");
        for (std::size_t index = 0; index < ranges.size(); ++index) {
            const bool listed = std::find(indices.begin(), indices.end(), index) != indices.end();
            const bool pending = index >= first && recorder.PendingWriteOverlaps(ranges[index].begin, static_cast<std::size_t>(ranges[index].end - ranges[index].begin));
            Require(listed == pending, "a range is listed as overlapped differently from PendingWriteOverlaps");
            matched += listed ? 1 : 0;
        }
    }
    Require(matched == 12, "the overlap comparison covered too few overlapped ranges");
    recorder.Sync();
    recorder.PendingWriteRanges(writes);
    Require(writes.empty(), "pending write ranges outlived their batches");
}

void spaceWriteSyncTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: address space pending writes not tested\n";
        return;
    }
    constexpr std::size_t bytes = 0x10000;
    constexpr std::size_t count = 4;
    void* block = ::operator new(2 * count * bytes, std::align_val_t{65536});
    std::memset(block, 0, 2 * count * bytes);
    auto* memory = static_cast<std::byte*>(block);
    const auto registry = [&](bool add) {
        GuestAllocations::Mutation mutation;
        for (std::size_t index = 0; index < count; ++index) {
            if (add) mutation.Add(memory + 2 * index * bytes, bytes, true, false, true);
            else mutation.Remove(memory + 2 * index * bytes);
        }
    };
    const auto rangeAt = [&](std::size_t index) { return reinterpret_cast<std::uint64_t>(memory + 2 * index * bytes); };
    const auto build = [&] {
        GuestBufferMemory leased(context);
        leased.AcquireRegistered();
        leased.AddReadable(rangeAt(0) + 64, 16);
        leased.Upload(true);
        const auto reads = leased.DeviceReads();
        leased.WriteBack();
        return reads.size();
    };
    registry(true);
    Require(recorder.Idle(), "the recorder is busy before the address space tests");
    const auto regions = build();
    if (regions < count) {
        registry(false);
        ::operator delete(block, std::align_val_t{65536});
        std::cout << "the registered ranges are not served in place: address space pending writes not tested\n";
        return;
    }
    const std::array<std::byte, 4> value{std::byte{0x5a}, std::byte{0xa5}, std::byte{0x5a}, std::byte{0xa5}};
    const auto landed = [&](std::uint64_t address) { return std::memcmp(reinterpret_cast<const void*>(address), value.data(), value.size()) == 0; };
    const auto inside = rangeAt(2) + 0x100;
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(inside, value, 7000, 0, false);
    Require(Recorder::PendingCompletionLabels() == 1 && !landed(inside), "the label inside a registered range is not pending");
    build();
    Require(landed(inside) && Recorder::PendingCompletionLabels() == 0, "a build over the address space did not complete the label pending in one of its ranges");
    recorder.Sync();
    const auto gap = rangeAt(2) + bytes + 0x100;
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(gap, value, 7001, 0, false);
    build();
    Require(!landed(gap) && Recorder::PendingCompletionLabels() == 1, "a build over the address space completed a label outside its ranges");
    recorder.Sync();
    Require(landed(gap) && Recorder::PendingCompletionLabels() == 0, "the label outside the ranges did not land");
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(rangeAt(1) + 0x40, value, 7002, 0, false);
    recorder.Submit();
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(rangeAt(3) + 0x40, value, 7003, 0, false);
    build();
    Require(landed(rangeAt(1) + 0x40) && landed(rangeAt(3) + 0x40) && Recorder::PendingCompletionLabels() == 0, "a build over the address space missed a label behind an earlier sync");
    recorder.Sync();
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(rangeAt(1) + 0x80, value, 7004, 0, false);
    bool noted = false;
    recorder.OnComplete([&] {
        static_cast<void>(recorder.Commands());
        recorder.AfterCompletions(rangeAt(3) + 0x80, value, 7005, 0, false);
        noted = true;
    });
    build();
    Require(noted && landed(rangeAt(1) + 0x80) && landed(rangeAt(3) + 0x80) && Recorder::PendingCompletionLabels() == 0, "a build over the address space missed a label a completion noted during its sync");
    recorder.Sync();
    registry(false);
    ::operator delete(block, std::align_val_t{65536});
}

void queuedWriteGroupTests() {
    using Write = Recorder::QueuedWrite;
    using Groups = std::vector<std::size_t>;
    const auto paint = [](const std::vector<Write>& writes, const Groups& groups, std::size_t bytes) {
        std::vector<std::uint8_t> memory(bytes, 0);
        std::size_t first = 0;
        for (const auto end : groups) {
            for (const bool computed : {false, true}) {
                for (auto i = first; i < end; ++i) {
                    if (writes[i].computed != computed) continue;
                    for (auto at = writes[i].begin; at < writes[i].end; ++at) memory[at] = static_cast<std::uint8_t>(i + 1);
                }
            }
            first = end;
        }
        return memory;
    };
    const auto programOrder = [](const std::vector<Write>& writes, std::size_t bytes) {
        std::vector<std::uint8_t> memory(bytes, 0);
        for (std::size_t i = 0; i < writes.size(); ++i) {
            for (auto at = writes[i].begin; at < writes[i].end; ++at) memory[at] = static_cast<std::uint8_t>(i + 1);
        }
        return memory;
    };
    Require(Recorder::QueuedWriteGroups({}).empty(), "no queued writes made a group");
    const std::vector<Write> query{{0x100, 0x200, false}, {0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x200, 0x300, false}, {0x200, 0x2f8, true}, {0x208, 0x300, true}};
    Require(Recorder::QueuedWriteGroups(query) == Groups{3, 6}, "the clear of the next query slot did not wait for the dumps before it");
    const std::vector<Write> reused{{0x100, 0x1f8, true}, {0x100, 0x200, false}, {0x100, 0x1f8, true}};
    Require(Recorder::QueuedWriteGroups(reused) == Groups{1, 3}, "a store over a computed write of its group did not start a new group");
    const std::vector<Write> apart{{0x100, 0x1f8, true}, {0x1f8, 0x200, false}, {0x300, 0x310, false}};
    Require(Recorder::QueuedWriteGroups(apart) == Groups{1, 3}, "stores after the computed writes joined their group");
    const std::vector<Write> label{{0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x400, 0x408, false}};
    Require(Recorder::QueuedWriteGroups(label) == Groups{2, 3}, "a label after the dumps landed before them");
    const std::vector<Write> stores{{0x100, 0x110, false}, {0x100, 0x110, false}};
    Require(Recorder::QueuedWriteGroups(stores) == Groups{2}, "stores alone were split into groups");
    std::uint64_t seed = 0x9e3779b97f4a7c15ull;
    const auto next = [&](std::uint64_t bound) {
        seed ^= seed << 13u;
        seed ^= seed >> 7u;
        seed ^= seed << 17u;
        return seed % bound;
    };
    constexpr std::size_t bytes = 64;
    for (int round = 0; round < 20000; ++round) {
        std::vector<Write> writes;
        const auto count = 1 + next(12);
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto begin = next(bytes - 1);
            const auto end = begin + 1 + next(std::min<std::uint64_t>(16, bytes - begin));
            writes.push_back({begin, std::min<std::uint64_t>(end, bytes), next(2) == 0});
        }
        const auto groups = Recorder::QueuedWriteGroups(writes);
        Require(!groups.empty() && groups.back() == writes.size() && std::is_sorted(groups.begin(), groups.end()), "queued write groups do not end at the last write in order");
        Require(paint(writes, groups, bytes) == programOrder(writes, bytes), "recording each group's stores before its computed writes changed the final bytes of the program order");
    }
}

void pendingKeyStoreTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: pending key stores not tested\n";
        return;
    }
    constexpr std::size_t surfaceBytes = 65536;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the pending key store block");
    auto* keys = static_cast<std::uint8_t*>(block);
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
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the pending key store block refused: pending key stores not tested\n";
        return;
    }
    recorder.Sync();
    std::memset(keys, 0x00, bytes);
    MarkDccUncompressed(context, address, surfaceBytes);
    Require(recorder.PendingWriteOverlaps(address, keyCount) && keys[0] == 0x00, "the uncompressed key store did not stay pending");
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed, "the keys after a pending uncompressed store do not read as uncompressed");
    Require(recorder.PendingWriteOverlaps(address, keyCount), "reading keys the driver's own pending store wrote waited for the GPU");
    Require(CurrentDccKeys(address + 16, surfaceBytes / 2) == DccKeys::Uncompressed, "a key range inside the pending store does not read as uncompressed");
    recorder.NotePendingWrite(address + 16, 16);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address, keyCount), "a key read with a later writer over the pending store did not wait for it");
    Require(std::all_of(keys, keys + keyCount, [](std::uint8_t key) { return key == 0xff; }), "the uncompressed key store did not land");
    recorder.NotePendingWrite(address, keyCount);
    NoteKeysFillOnGpu(address, keyCount, DccKeys::Clear0001);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of a pending fill did not read as its keys without a wait");
    Require(KnownPendingDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of a pending fill were not known without a wait");
    Require(KnownPendingDccKeys(address + 16, surfaceBytes / 2) == DccKeys::Clear0001, "a key range inside a pending fill was not known");
    Require(!KnownPendingDccKeys(address, 2 * surfaceBytes).has_value() && recorder.PendingWriteOverlaps(address, keyCount), "a range wider than the pending fill was known, or waited for");
    recorder.NotePendingWrite(address, 2 * keyCount);
    Require(!KnownPendingDccKeys(address, surfaceBytes).has_value() && recorder.PendingWriteOverlaps(address, keyCount), "the keys under a later writer over a pending fill were known");
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address, keyCount), "a later wider writer over a pending fill was not waited for");
    Require(!KnownPendingDccKeys(address, surfaceBytes).has_value(), "keys nothing writes were known as pending keys");
    recorder.NotePendingFill(address, keyCount / 2, 0x40);
    recorder.NotePendingFill(address + keyCount / 2, keyCount / 2, 0x40);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of pending 0001 fills over both halves did not read as 0001 without a wait");
    recorder.NotePendingWrite(address + keyCount / 2, 16);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address, keyCount), "a later writer of unknown bytes over pending fills was not waited for");
    recorder.NotePendingFill(address, keyCount / 2, 0x00);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Mixed && recorder.PendingWriteOverlaps(address, keyCount), "a pending 0000 fill over half of uncompressed keys did not read as mixed without a wait");
    recorder.Sync();
#ifdef _WIN32
    auto* edgeBlock = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2 * bytes, MEM_RESERVE, PAGE_NOACCESS));
    Require(edgeBlock != nullptr && VirtualAlloc(edgeBlock, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr, "cannot reserve the key block past committed memory");
#else
    void* edgeMapping = mmap(nullptr, 2 * bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(edgeMapping != MAP_FAILED && mprotect(edgeMapping, bytes, PROT_READ | PROT_WRITE) == 0, "cannot reserve the key block past committed memory");
    auto* edgeBlock = static_cast<std::uint8_t*>(edgeMapping);
#endif
    const auto edge = reinterpret_cast<std::uint64_t>(edgeBlock) + bytes - keyCount / 2;
    Require(!AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(edge), keyCount), "keys past the end of committed memory read as accessible");
    recorder.NotePendingFill(edge, keyCount / 2, 0x40);
    Require(CurrentDccKeys(edge, surfaceBytes) == DccKeys::Unreadable && !recorder.PendingWriteOverlaps(edge, keyCount), "keys running past the end of committed memory, with a pending fill over their committed half, were not read as unreadable after a wait");
#ifdef _WIN32
    VirtualFree(edgeBlock, 0, MEM_RELEASE);
#else
    munmap(edgeBlock, 2 * bytes);
#endif
    constexpr std::uint64_t displayBytes = 15u * 9u * 65536u;
    constexpr std::size_t consoleKeys = 49152;
    const auto displayKeys = DccKeyCount(TextureTileMode::kR64KBX, 4, 1920, 1080, displayBytes);
    std::memset(keys, 0x00, consoleKeys);
    MarkDccUncompressed(context, address, displayBytes, displayKeys);
    Require(recorder.PendingWriteOverlaps(address, consoleKeys) && keys[consoleKeys - 1] == 0x00, "the uncompressed key store over a 1920x1080 surface did not stay pending");
    Require(CurrentDccKeys(address + displayBytes / 256, (consoleKeys - displayBytes / 256) * 256) == DccKeys::Uncompressed && recorder.PendingWriteOverlaps(address, consoleKeys), "the keys past one per 256 bytes of a pending 1920x1080 store did not read as uncompressed without a wait");
    Require(CurrentDccKeys(address, displayBytes, displayKeys) == DccKeys::Uncompressed && recorder.PendingWriteOverlaps(address, consoleKeys), "the keys of a pending 1920x1080 store did not read as uncompressed over the console's extent without a wait");
    recorder.NotePendingWrite(address + 40000, 1);
    Require(CurrentDccKeys(address, displayBytes, displayKeys) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address + 40000, 1), "a later writer of key 40000 over a pending 1920x1080 store was not waited for");
    Require(std::all_of(keys, keys + consoleKeys, [](std::uint8_t key) { return key == 0xff; }), "the uncompressed key store over a 1920x1080 surface did not land on all 49152 keys");
    recorder.Sync();
}

bool shadowedKeyStoreTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    if (!UnitShadowEnabled() || context.hostImportAlignment == 0) {
        std::cout << "unit shadows or host imports unavailable: keys under a unit shadow not tested\n";
        return false;
    }
    constexpr std::uint64_t unit = 65536;
    constexpr std::size_t bytes = 2 * unit;
    constexpr std::size_t surfaceBytes = 65536;
    constexpr std::size_t keyCount = surfaceBytes / 256;
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, unit);
    Require(block != nullptr, "cannot reserve the shadowed key block");
    const auto section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(bytes), nullptr);
    Require(section != nullptr, "cannot create the shadowed key block");
    GuestArena::GuestArenaMap_nid_postfix(block, bytes, section, 0, PAGE_READWRITE);
    CloseHandle(section);
#else
    void* block = AllocateWatched(bytes, unit);
    Require(block != nullptr, "cannot allocate the shadowed key block");
#endif
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    auto* keys = static_cast<std::uint8_t*>(block);
    std::memset(block, 0xff, bytes);
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
        std::cout << "the shadowed key block is not imported and watched: keys under a unit shadow not tested\n";
        return false;
    }
    recorder.Sync();
    CollectWritesUncached(address, bytes);
    const auto copyBuffer = Metal::Vk::CmdCopyBuffer;
    const auto shadow = [&](std::uint8_t value) {
        auto destination = ShadowDestinationFor(context, *import, address, address + unit);
        Require(destination.has_value() && destination->seedUnits.empty(), "the key unit has no unit shadow destination");
        auto pattern = std::make_shared<Buffer>(context, static_cast<std::size_t>(unit), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memset(pattern->Bytes().data(), value, pattern->Bytes().size());
        const auto commands = recorder.Commands();
        recorder.Keep(pattern);
        recorder.Keep(destination->slab);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkBufferCopy piece{0, destination->offset, unit};
        copyBuffer(commands, pattern->Handle(), destination->buffer, 1, &piece);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        recorder.MarkCovered(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const ShadowedRange range{address, address + unit, destination->slab};
        MarkShadowed(*import, std::span(&range, 1), TrackerGeneration());
        Require(AnyShadowedOverlaps(address, keyCount), "the key unit is not shadowed");
    };
    recorder.NotePendingFill(address, unit, 0x00);
    MarkWritten(address, unit);
    shadow(0x40);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && !recorder.PendingWriteOverlaps(address, keyCount), "the keys under a fresh unit shadow were not read from its published bytes after a wait");
    Require(keys[0] == 0x40 && keys[unit - 1] == 0x40 && keys[unit] == 0xff && !AnyShadowedOverlaps(address, unit), "the fresh unit's publish did not land");
    recorder.Sync();
    shadow(0x22);
    recorder.NotePendingFill(address, unit, 0xc0);
    MarkWritten(address, unit);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear1111 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of a pending fill over a stale unit shadow were not read as the fill's keys without a wait");
    Require(!AnyShadowedOverlaps(address, unit), "the stale unit still counts as shadowed after the key read");
    recorder.Sync();
    GuestTextureResource resource{};
    resource.baseAddress = address + unit;
    resource.format = 56;
    resource.dccAddress = address;
    shadow(0x22);
    recorder.NotePendingFill(address, unit, 0xc0);
    MarkWritten(address, unit);
    Require(!WaitForKeyWriters(resource, surfaceBytes).has_value() && recorder.PendingWriteOverlaps(address, keyCount), "the key writers over a stale unit shadow were waited for");
    recorder.Sync();
    Require(keys[0] == 0x40 && keys[unit - 1] == 0x40, "a stale unit's slab bytes were published");
    return true;
}

void keptUnitSettleTests(const Device& device, Recorder& recorder) {
    Require(AgcDriver::GuestMemory::WriteWatched(), "(k) kept-unit settle tests require host write watching");
    const auto& base = device.GetContext();
    constexpr std::uint32_t width = 1024;
    constexpr std::uint32_t height = 512;
    constexpr std::size_t surfaceBytes = std::size_t{width} * height * 4;
    constexpr std::size_t unitBytes = 65536;
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    constexpr std::array<std::uint8_t, 4> blue{0, 0, 255, 255};
    for (const bool direct : {false, true}) {
        const std::string what = direct ? "GPU-mapped direct memory" : "watched memory";
        if (direct && base.hostImportAlignment == 0) {
            std::cout << "host imports unavailable: units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        struct Backing {
            Recorder& recorder;
            bool direct;
            void* block = nullptr;
            std::int64_t physical = -1;
            ~Backing() {
                try {
                    recorder.Sync();
                } catch (const std::exception& error) {
                    std::cerr << "(k) the recorded work did not finish, the test memory stays mapped: " << error.what() << '\n';
                    return;
                }
                if (direct) {
                    if (block != nullptr) sceKernelMunmap(block, surfaceBytes);
                    if (physical >= 0) sceKernelReleaseDirectMemory(physical, surfaceBytes);
                    return;
                }
                if (block == nullptr) return;
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                ReleaseWatched(block, surfaceBytes);
            }
        } backing{recorder, direct};
        if (direct) {
            Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, surfaceBytes, unitBytes, 0, &backing.physical) == 0, "(k) allocate the direct memory");
            Require(sceKernelMapDirectMemory(&backing.block, surfaceBytes, 0x33, 0, backing.physical, unitBytes) == 0, "(k) map the direct memory for the GPU");
        } else {
            backing.block = AllocateWatched(surfaceBytes, unitBytes);
            Require(backing.block != nullptr, "(k) no watched block although write watching is on");
            GuestAllocations::Mutation mutation;
            mutation.Add(backing.block, surfaceBytes, true, true, true);
        }
        auto* memory = static_cast<std::uint8_t*>(backing.block);
        std::memset(memory, 0x5a, surfaceBytes);
        const auto address = reinterpret_cast<std::uint64_t>(backing.block);
        AgcDriver::GuestMemory::CollectWritesUncached(address, surfaceBytes);
        TextureDetiler detiler(base);
        auto context = base;
        if (!direct) context.hostImportAlignment = 0;
        context.detiler = &detiler;
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = width;
        resource.height = height;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kR64KBX;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        Require(DescribeSurface(resource).guestBytes == surfaceBytes, "(k) unexpected surface size");
        if (direct && !AgcDriver::GuestMemory::Watched(address, surfaceBytes)) {
            std::cout << "GPU-mapped direct memory is not write-watched here: units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        Require(AgcDriver::GuestMemory::Watched(address, surfaceBytes), "(k) " + what + " is not watched");
        if (direct && HostImportFor(context, address, surfaceBytes) == nullptr) {
            std::cout << "host import of the direct memory refused: units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        if (direct && !AgcDriver::GuestMemory::Watched(address, surfaceBytes)) {
            std::cout << "the host import of the direct memory took it out of the write watch (its imports are compared): units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        Require(direct || HostImportFor(context, address, surfaceBytes) == nullptr, "(k) watched memory was imported although host imports are off for it");
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const auto clear = [&](VkClearColorValue value) {
                const auto commands = recorder.Commands();
                recorder.Keep(image);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
            };
            const auto store = [&] {
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
            };
            const auto unitHolds = [&](std::size_t unit, const std::array<std::uint8_t, 4>& texel) {
                std::vector<std::byte> read(unitBytes);
                AgcDriver::GuestMemory::Read(address + unit * unitBytes, read);
                for (std::size_t i = 0; i < unitBytes; ++i) {
                    if (std::to_integer<std::uint8_t>(read[i]) != texel[i % 4]) return false;
                }
                return true;
            };
            clear({{1.0f, 0.0f, 0.0f, 1.0f}});
            memory[16] = 0x77;
            AgcDriver::GuestMemory::CollectWritesUncached(address, surfaceBytes);
            store();
            std::vector<std::byte> head(unitBytes);
            AgcDriver::GuestMemory::Read(address, head);
            Require(std::to_integer<std::uint8_t>(head[16]) == 0x77, "(k) " + what + ": the CPU store into unit 0 was overwritten by results it was newer than");
            for (std::size_t unit = 1; unit < surfaceBytes / unitBytes; ++unit) Require(unitHolds(unit, red), "(k) " + what + ": the first write-back did not store unit " + std::to_string(unit));
            clear({{0.0f, 0.0f, 1.0f, 1.0f}});
            store();
            for (std::size_t unit = 1; unit < surfaceBytes / unitBytes; ++unit) {
                Require(unitHolds(unit, blue), "(k) " + what + ": unit " + std::to_string(unit) + " kept the results of the earlier write-back; that write-back's own store was taken for a CPU write because a neighbouring unit was kept for the CPU");
            }
            AgcDriver::GuestMemory::Read(address, head);
            Require(std::to_integer<std::uint8_t>(head[16]) == 0x77, "(k) " + what + ": the second write-back stored over unit 0, which the CPU wrote after that unit's generation");
        }
        std::cout << "units stored next to a unit kept for the CPU in " << what << ": ok\n";
    }
}

// RunMode (RecorderTwinHarness.h) for a test that reports its own skip: the test's exit code.
template <typename Test>
int RunCoded(Test test) {
    @autoreleasepool {
        try {
            Device device;
            std::unique_lock gpu(GpuMutex());
            std::cout << "host imports " << (PrepareImportWatch(device.GetContext()) == ImportWatch::Unwatch ? "are compared" : "stay watched") << '\n';
            Recorder recorder(device.GetContext());
            recorder.Activate();
            const int code = test(device, recorder);
            recorder.Sync();
            return code;
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}

const bool registered = [] {
    using AgcDriver::MetalTwins::RegisterRecorderMode;
    RegisterRecorderMode("--queued-write-groups-only", [] {
        try {
            queuedWriteGroupTests();
            std::cout << "queued write group tests passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    });
    RegisterRecorderMode("--pending-write-snapshot-only", [] { return RunMode([](const Device& device, Recorder& recorder) { pendingWriteSnapshotTests(device, recorder); }); });
    RegisterRecorderMode("--kept-unit-settle-only", [] { return RunMode([](const Device& device, Recorder& recorder) { keptUnitSettleTests(device, recorder); }); });
    RegisterRecorderMode("--read-list-only", [] {
        return RunMode([](const Device& device, Recorder& recorder) {
            readListTests(device, recorder);
            std::cout << "Read list tests passed\n";
        });
    });
    RegisterRecorderMode("--pending-key-stores-only", [] {
        return RunCoded([](const Device& device, Recorder& recorder) {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, host imports are unavailable\n";
                return 77;
            }
            pendingKeyStoreTests(device, recorder);
            std::cout << "Pending DCC key store tests passed\n";
            return 0;
        });
    });
    RegisterRecorderMode("--shadowed-key-stores-only", [] {
        return RunCoded([](const Device& device, Recorder& recorder) {
            if (!shadowedKeyStoreTests(device, recorder)) {
                std::cout << "skipped, the device imports no watched host memory under a unit shadow\n";
                return 77;
            }
            std::cout << "DCC keys under unit shadows tests passed\n";
            return 0;
        });
    });
    RegisterRecorderMode("--space-pending-writes-only", [] {
        return RunMode([](const Device& device, Recorder& recorder) {
            overlappedRangeTests(recorder);
            spaceWriteSyncTests(device, recorder);
            std::cout << "Address space pending write tests passed\n";
        });
    });
    return true;
}();

}
