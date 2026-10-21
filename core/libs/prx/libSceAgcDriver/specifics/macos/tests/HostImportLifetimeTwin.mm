// Metal twin of tests/HostImportLifetime.cpp (agc_driver_host_import_lifetime, agc_driver_host_import_mutation_race;
// upstream 304798015b, part of agc_driver_recorder_tests), swapped in by twins/UPSYNC3-tests.cmake. It registers the
// --host-import-lifetime-only and --host-import-mutation-race-only modes with RecorderTwin.mm (RecorderTwins.h) and
// runs them as upstream main does: on the test device, without the GPU lock or an active recorder.
// Upstream observes a host import's Vulkan objects by intercepting vkAllocateMemory, vkBindBufferMemory,
// vkDestroyBuffer and vkFreeMemory through the context's device resolver. On Metal an import is a no-copy MTLBuffer
// (GuestBufferMemory.mm) and its destruction reaches the test through Metal::SetImportReleaseProbe (MetalNoCopy.h),
// which GuestBufferMemory.mm's import destroy calls with the import's buffer handle before it releases the buffer:
// - "freed" (upstream: vkFreeMemory returned) is the probe's call for the import's buffer, made once; the guest pages
//   must still be registered and accessible then; a probe call that blocks (upstream: a blocked vkFreeMemory) holds
//   the destroy on its thread until the test releases it;
// - the import an upload made (upstream: the vkAllocateMemory it saw) is the live import HostImportFor finds over the
//   uploaded range, which must not have existed before the upload;
// - upstream's vkQueueWaitIdle is Recorder::WaitSerial on the newest batch (it does not reap, as vkQueueWaitIdle).
// Not ported (Metal has no failure seam for them; TechnicalDebt metal-host-import-failure-seams): failedConstruction
// (an injected vkGetBufferDeviceAddressKHR failure after the import's memory was made) and productionWaiterFailure
// (an injected vkQueueSubmit failure of the recorder submit the production pin waiter makes; an MTL4 commit returns
// no error). The other checks and their messages are upstream's.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "../src/MetalNoCopy.h"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct ImportState {
    VkBuffer buffer;
    std::uint64_t address;
    std::size_t bytes;
    std::atomic<bool> destroyed{false};
    std::atomic<bool> freed{false};
    std::atomic<unsigned> frees{0};
    std::atomic<bool> outOfOrder{false};
    std::atomic<bool> inaccessible{false};
    std::mutex mutex;
    std::condition_variable wake;
    bool blockFree = false;
    bool freeing = false;
    bool allowFree = false;
    std::thread::id freeingThread;
};

struct Tracking {
    std::mutex mutex;
    std::map<const void*, std::shared_ptr<ImportState>> imports;

    std::shared_ptr<ImportState> Add(const HostImport& import) {
        auto state = std::make_shared<ImportState>();
        state->buffer = import.buffer;
        state->address = import.base;
        state->bytes = import.bytes;
        std::lock_guard lock(mutex);
        imports[reinterpret_cast<const void*>(import.buffer)] = state;
        return state;
    }
};

Tracking* tracking = nullptr;

// GuestBufferMemory.mm's import destroy, before it releases the no-copy buffer (upstream: vkDestroyBuffer and
// vkFreeMemory of the import).
void releaseProbe(const void* buffer) {
    if (tracking == nullptr) return;
    std::shared_ptr<ImportState> state;
    {
        std::lock_guard lock(tracking->mutex);
        const auto found = tracking->imports.find(buffer);
        if (found != tracking->imports.end() && !found->second->freed.load()) state = found->second;
    }
    if (state == nullptr) return;
    state->destroyed.store(true);
    if (!RegisteredReadableCovers(state->address, state->bytes) || !AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(state->address), state->bytes, false)) state->inaccessible.store(true);
    {
        std::unique_lock lock(state->mutex);
        if (state->blockFree) {
            state->freeing = true;
            state->freeingThread = std::this_thread::get_id();
            state->wake.notify_all();
            if (!state->wake.wait_for(lock, std::chrono::seconds(2), [&] { return state->allowFree; })) state->outOfOrder.store(true);
        }
    }
    state->frees.fetch_add(1);
    state->freed.store(true);
}

void requireFreed(const std::shared_ptr<ImportState>& state) {
    Require(state->freed.load(), "guest mutation reached host apply before vkFreeMemory completed");
    Require(state->destroyed.load() && !state->outOfOrder.load() && state->frees.load() == 1, "host import Vulkan objects were destroyed out of order or more than once");
    Require(!state->inaccessible.load(), "guest pages became inaccessible before vkFreeMemory returned");
}

class Fixture {
public:
    explicit Fixture(const Context& original) : context(original) {
        Require(tracking == nullptr, "host import tracking is already active");
        tracking = &calls;
        AgcDriver::Metal::SetImportReleaseProbe(&releaseProbe);
        std::lock_guard gpu(GpuMutex());
        recorder = std::make_unique<Recorder>(context);
        recorder->Activate();
    }

    ~Fixture() {
        std::lock_guard gpu(GpuMutex());
        recorder.reset();
        ClearHostImports(context.device);
        context.bufferPool.reset();
        AgcDriver::Metal::SetImportReleaseProbe(nullptr);
        tracking = nullptr;
    }

    std::shared_ptr<ImportState> Import(std::uint64_t address, std::size_t bytes) {
        std::lock_guard gpu(GpuMutex());
        const auto* imported = HostImportFor(context, address, bytes);
        Require(imported != nullptr, "host pointer import was refused for the lifetime test");
        return calls.Add(*imported);
    }

    void Cleanup() {
        std::lock_guard gpu(GpuMutex());
        recorder->Sync();
        ClearHostImports(context.device);
    }

    // Upstream's vkQueueWaitIdle: the newest submitted batch has completed; nothing is reaped.
    void WaitQueue() {
        recorder->WaitSerial(recorder->NewestSubmitted());
    }

    Context context;
    Tracking calls;
    std::unique_ptr<Recorder> recorder;
};

class Allocation {
public:
    Allocation(Fixture& fixture, std::size_t units = 1) : fixture(fixture) {
        alignment = static_cast<std::size_t>(std::max<VkDeviceSize>(65536, fixture.context.hostImportAlignment));
        bytes = alignment * units;
        data = ::operator new(bytes, std::align_val_t{alignment});
        std::memset(data, 0x37, bytes);
        GuestAllocations::Mutation mutation;
        mutation.Add(data, bytes, true, true, true);
    }

    ~Allocation() {
        try {
            if (registered) {
                fixture.Cleanup();
                GuestAllocations::Mutation mutation;
                mutation.Remove(data);
            }
            Release();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "host import test cleanup failed: %s\n", error.what());
            std::terminate();
        }
    }

    std::uint64_t Address() const { return reinterpret_cast<std::uint64_t>(data); }

    void Release() {
        if (released) return;
        ::operator delete(data, std::align_val_t{alignment});
        released = true;
    }

    Fixture& fixture;
    void* data = nullptr;
    std::size_t alignment = 0;
    std::size_t bytes = 0;
    bool registered = true;
    bool released = false;
};

void descriptorFirstImport(Fixture& fixture) {
    Allocation allocation(fixture);
    Require(!HostImportCovers(fixture.context, allocation.Address(), allocation.bytes), "descriptor-only lifetime test did not run before the first host import");
    std::shared_ptr<ImportState> imported;
    {
        std::lock_guard gpu(GpuMutex());
        GuestBufferMemory memory(fixture.context);
        memory.AddReadable(allocation.Address(), allocation.bytes);
        memory.Upload(false);
        Require(HostImportCovers(fixture.context, allocation.Address(), allocation.bytes), "descriptor-only upload did not import its allocation");
        const auto* found = HostImportFor(fixture.context, allocation.Address(), allocation.bytes);
        Require(found != nullptr && found->base == allocation.Address(), "descriptor-only upload did not import its allocation");
        imported = fixture.calls.Add(*found);
        std::uint32_t adjustment = 0;
        const auto descriptor = memory.Descriptor(allocation.Address(), allocation.bytes, adjustment);
        Require(descriptor.buffer == imported->buffer && descriptor.offset == 0 && adjustment == 0, "descriptor-only upload did not bind the tracked import");
    }
    Require(HostImportCovers(fixture.context, allocation.Address(), allocation.bytes), "descriptor-only import did not persist after its build ended");
    {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(allocation.data, allocation.bytes, [&](const void*, std::size_t, const void*, bool) {
            requireFreed(imported);
            allocation.Release();
        });
        allocation.registered = false;
    }
    Require(!HostImportCovers(fixture.context, allocation.Address(), allocation.bytes), "descriptor-only import survived allocation removal");
}

class PendingRetirement {
public:
    explicit PendingRetirement(Fixture& fixture) : fixture(fixture), first(fixture) {
        std::lock_guard gpu(GpuMutex());
        build = std::make_unique<GuestBufferMemory>(fixture.context);
        build->AcquireRegistered();
        build->UploadPrepare(true);
        retry = std::make_unique<GuestBufferMemory>(fixture.context);
        retry->AcquireRegistered();
        retry->UploadPrepare(true);
        allocation = std::make_unique<Allocation>(fixture);
        imported = fixture.Import(allocation->Address(), allocation->bytes);
        result = std::make_unique<Buffer>(fixture.context, allocation->bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = fixture.recorder->Commands();
        RecordMemoryBarrier(fixture.context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        CopyBuffer(fixture.context, commands, imported->buffer, 0, result->Handle(), 0, allocation->bytes);
        RecordMemoryBarrier(fixture.context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        fixture.recorder->NotePendingRead(allocation->Address(), allocation->bytes, Recorder::ReadKind::AddressBased);
        fixture.recorder->Submit();
        fixture.WaitQueue();
        generation = std::make_unique<Allocation>(fixture);
        Require(!fixture.recorder->Idle(), "retirement fixture has no submitted batch");
    }

    ~PendingRetirement() {
        if (imported != nullptr) {
            {
                std::lock_guard lock(imported->mutex);
                imported->allowFree = true;
            }
            imported->wake.notify_all();
        }
        build.reset();
        retry.reset();
        fixture.Cleanup();
    }

    void VerifyCopy() {
        result->Invalidate();
        const auto bytes = result->Bytes();
        Require(std::all_of(bytes.begin(), bytes.end(), [](std::byte value) { return value == std::byte{0x37}; }), "retirement lost the submitted import read");
    }

    Fixture& fixture;
    Allocation first;
    std::unique_ptr<Buffer> result;
    std::unique_ptr<Allocation> allocation;
    std::unique_ptr<Allocation> generation;
    std::unique_ptr<GuestBufferMemory> build;
    std::unique_ptr<GuestBufferMemory> retry;
    std::shared_ptr<ImportState> imported;
};

void deferredRetirement(Fixture& fixture) {
    PendingRetirement pending(fixture);
    pending.imported->blockFree = true;
    {
        std::lock_guard gpu(GpuMutex());
        pending.build->UploadFinish(true);
        Require(!HostImportCovers(fixture.context, pending.allocation->Address(), pending.allocation->bytes), "deferred retirement left its import cached");
        Require(!pending.imported->freed.load(), "retirement did not keep its import through the pending batch");
        fixture.recorder->Sync();
        fixture.recorder->FlushDeferredReleases();
    }
    bool entered = false;
    bool onReleaseThread = false;
    {
        std::unique_lock lock(pending.imported->mutex);
        entered = pending.imported->wake.wait_for(lock, std::chrono::seconds(2), [&] { return pending.imported->freeing; });
        onReleaseThread = pending.imported->freeingThread != std::this_thread::get_id();
    }
    std::atomic<bool> applied{false};
    const auto waitsBefore = LeaseCounters().contentionWaits;
    std::promise<void> starting;
    auto started = starting.get_future();
    auto mutation = std::async(std::launch::async, [&] {
        starting.set_value();
        GuestAllocations::Mutation mutation;
        mutation.Unmap(pending.allocation->data, pending.allocation->bytes, [&](const void*, std::size_t, const void*, bool) {
            requireFreed(pending.imported);
            applied.store(true);
            pending.allocation->Release();
        });
        pending.allocation->registered = false;
    });
    started.wait();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (LeaseCounters().contentionWaits == waitsBefore && mutation.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool reachedWaiter = LeaseCounters().contentionWaits > waitsBefore;
    const bool blocked = mutation.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout;
    const bool premature = applied.load();
    const bool registered = RegisteredReadableCovers(pending.allocation->Address(), pending.allocation->bytes);
    {
        std::lock_guard lock(pending.imported->mutex);
        pending.imported->allowFree = true;
    }
    pending.imported->wake.notify_all();
    mutation.get();
    Require(entered && onReleaseThread, "retired import destruction did not run on the recorder release thread");
    Require(reachedWaiter && blocked && !premature && registered, "guest mutation passed a retired import whose Vulkan free was still running");
    Require(applied.load(), "guest mutation did not resume after deferred Vulkan free");
    requireFreed(pending.imported);
    pending.VerifyCopy();
}

void idleUnmap(Fixture& fixture) {
    Allocation allocation(fixture);
    const auto imported = fixture.Import(allocation.Address(), allocation.bytes);
    {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(allocation.data, allocation.bytes, [&](const void*, std::size_t, const void*, bool) {
            requireFreed(imported);
            allocation.Release();
        });
        allocation.registered = false;
    }
    Require(!HostImportCovers(fixture.context, allocation.Address(), allocation.bytes), "an unmapped range still has a live host import");
}

void pendingRead(Fixture& fixture, bool submitted, bool locked) {
    Allocation allocation(fixture);
    const auto imported = fixture.Import(allocation.Address(), allocation.bytes);
    Buffer result(fixture.context, allocation.bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::atomic<bool> completed{false};
    struct Finish {
        Fixture& fixture;
        ~Finish() { fixture.Cleanup(); }
    } finish{fixture};
    std::unique_lock gpu(GpuMutex());
    const auto commands = fixture.recorder->Commands();
    RecordMemoryBarrier(fixture.context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    CopyBuffer(fixture.context, commands, imported->buffer, 0, result.Handle(), 0, allocation.bytes);
    RecordMemoryBarrier(fixture.context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    fixture.recorder->NotePendingRead(allocation.Address(), allocation.bytes, Recorder::ReadKind::AddressBased);
    fixture.recorder->Keep(std::make_shared<GuestAllocations::Lease>(GuestAllocations::GuestAllocationsAcquire_nid_postfix()));
    fixture.recorder->OnComplete([&] { completed.store(true); });
    if (submitted) fixture.recorder->Submit();
    if (!locked) gpu.unlock();
    {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(allocation.data, allocation.bytes, [&](const void*, std::size_t, const void*, bool) {
            requireFreed(imported);
            Require(completed.load(), "guest unmap passed a pending GPU read");
            result.Invalidate();
            const auto copied = result.Bytes();
            Require(std::all_of(copied.begin(), copied.end(), [](std::byte value) { return value == std::byte{0x37}; }), "GPU read lost the source contents before guest unmap");
            allocation.Release();
        });
        allocation.registered = false;
    }
}

void partialUnmap(Fixture& fixture) {
    Allocation allocation(fixture, 3);
    Allocation unrelated(fixture);
    const auto imported = fixture.Import(allocation.Address(), allocation.bytes);
    const auto other = fixture.Import(unrelated.Address(), unrelated.bytes);
    auto* middle = static_cast<std::byte*>(allocation.data) + allocation.alignment;
    {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(middle, allocation.alignment, [&](const void* piece, std::size_t bytes, const void* original, bool last) {
            requireFreed(imported);
            Require(!other->freed.load(), "partial unmap retired a disjoint host import");
            Require(piece == middle && bytes == allocation.alignment && original == allocation.data && !last, "partial unmap changed the wrong guest extent");
        });
    }
    Require(!RegisteredReadableCovers(reinterpret_cast<std::uint64_t>(middle), allocation.alignment), "the middle of a partial unmap remains registered");
    const auto low = fixture.Import(allocation.Address(), allocation.alignment);
    const auto high = fixture.Import(allocation.Address() + 2 * allocation.alignment, allocation.alignment);
    // Upstream a9b14023dd: the whole import retires before the cut; nothing imported across the unmapped range is reused.
    Require(imported->freed.load(), "the side kept from an unmap inside the host import reused memory imported across the unmapped range");
    Require(high->address == allocation.Address() + 2 * allocation.alignment && high->bytes == allocation.alignment, "the kept side of the host import does not cover its remaining range");
    Require(!low->freed.load() && !high->freed.load() && !other->freed.load(), "remaining partial mappings could not be imported");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(allocation.data);
        allocation.registered = false;
    }
    requireFreed(low);
    requireFreed(high);
    Require(!other->freed.load(), "removing a split allocation retired a disjoint import");
}

void protectRollback(Fixture& fixture) {
    Allocation allocation(fixture);
    auto imported = fixture.Import(allocation.Address(), allocation.bytes);
    bool called = false;
    try {
        GuestAllocations::Mutation mutation;
        mutation.Protect(allocation.data, allocation.bytes, true, false, true, [&] {
            requireFreed(imported);
            called = true;
            throw std::runtime_error("expected host protection failure");
        });
        throw std::runtime_error("host protection failure was ignored");
    } catch (const std::runtime_error& error) {
        Require(called && std::string(error.what()) == "expected host protection failure", "unexpected protection failure");
    }
    {
        GuestAllocations::Mutation mutation;
        const auto range = mutation.Find(allocation.data);
        Require(range.readable && range.writable && range.gpu, "failed host protection changed the guest registration");
    }
    imported = fixture.Import(allocation.Address(), allocation.bytes);
    {
        GuestAllocations::Mutation mutation;
        mutation.Protect(allocation.data, allocation.bytes, true, false, true, [&] { requireFreed(imported); });
    }
    {
        GuestAllocations::Mutation mutation;
        const auto range = mutation.Find(allocation.data);
        Require(range.readable && !range.writable && range.gpu, "successful protection did not update the guest registration");
        mutation.Protect(allocation.data, allocation.bytes, true, true, true, [] {});
    }
    imported = fixture.Import(allocation.Address(), allocation.bytes);
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(allocation.data);
        allocation.registered = false;
    }
    requireFreed(imported);
}

void blockedFree(Fixture& fixture) {
    Allocation allocation(fixture);
    const auto imported = fixture.Import(allocation.Address(), allocation.bytes);
    imported->blockFree = true;
    std::atomic<bool> applied{false};
    auto mutation = std::async(std::launch::async, [&] {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(allocation.data, allocation.bytes, [&](const void*, std::size_t, const void*, bool) {
            requireFreed(imported);
            applied.store(true);
            allocation.Release();
        });
        allocation.registered = false;
    });
    bool entered = false;
    {
        std::unique_lock lock(imported->mutex);
        entered = imported->wake.wait_for(lock, std::chrono::seconds(2), [&] { return imported->freeing; });
    }
    const bool premature = applied.load();
    const bool stillRegistered = RegisteredReadableCovers(allocation.Address(), allocation.bytes);
    {
        std::lock_guard lock(imported->mutex);
        imported->allowFree = true;
    }
    imported->wake.notify_all();
    mutation.get();
    Require(entered, "guest unmap never reached Vulkan import destruction");
    Require(!premature && stillRegistered, "guest unmap released its registration while vkFreeMemory was blocked");
    Require(applied.load(), "guest unmap did not resume after vkFreeMemory returned");
}
void mutationDuringCompletion(Fixture& fixture) {
    Allocation allocation(fixture);
    const auto imported = fixture.Import(allocation.Address(), allocation.bytes);
    Buffer result(fixture.context, allocation.bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    bool rejected = false;
    bool applied = false;
    bool laterCompletion = false;
    bool copied = false;
    std::lock_guard gpu(GpuMutex());
    const auto commands = fixture.recorder->Commands();
    RecordMemoryBarrier(fixture.context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    CopyBuffer(fixture.context, commands, imported->buffer, 0, result.Handle(), 0, allocation.bytes);
    RecordMemoryBarrier(fixture.context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    fixture.recorder->OnComplete([&] {
        try {
            GuestAllocations::Mutation mutation;
            mutation.Protect(allocation.data, allocation.bytes, false, false, false, [&] { applied = true; });
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("active GPU completion") != std::string::npos;
        }
    });
    fixture.recorder->OnComplete([&] {
        laterCompletion = true;
        result.Invalidate();
        const auto bytes = result.Bytes();
        copied = std::all_of(bytes.begin(), bytes.end(), [](std::byte value) { return value == std::byte{0x37}; });
    });
    fixture.recorder->Sync();
    Require(rejected && !applied, "an active completion allowed guest memory mutation before later completions");
    Require(laterCompletion && copied, "rejecting a completion mutation lost later GPU results");
    Require(!imported->freed.load(), "rejecting a completion mutation retired its host import");
    {
        GuestAllocations::Mutation mutation;
        const auto range = mutation.Find(allocation.data);
        Require(range.readable && range.writable && range.gpu, "rejecting a completion mutation changed its guest range");
        mutation.Remove(allocation.data);
        allocation.registered = false;
    }
    requireFreed(imported);
}

void completedLeaseUnderLock(Fixture& fixture) {
    Allocation allocation(fixture);
    const auto imported = fixture.Import(allocation.Address(), allocation.bytes);
    std::lock_guard gpu(GpuMutex());
    fixture.recorder->Keep(std::make_shared<GuestAllocations::Lease>(GuestAllocations::GuestAllocationsAcquire_nid_postfix()));
    fixture.recorder->Sync();
    Require(fixture.recorder->Idle(), "completed lease test recorder is not idle");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(allocation.data);
        allocation.registered = false;
    }
    requireFreed(imported);
}


struct MutationRace {
    Fixture* fixture;
    Allocation* allocation;
    std::shared_ptr<ImportState> replacement;
    unsigned calls = 0;
};

MutationRace* mutationRace = nullptr;

bool raceWaiter(std::uintptr_t address, std::size_t bytes) {
    auto& race = *mutationRace;
    Require(address == race.allocation->Address() && bytes == race.allocation->bytes, "pin waiter received the wrong mutation extent");
    std::lock_guard gpu(GpuMutex());
    race.fixture->Cleanup();
    if (++race.calls == 1) race.replacement = race.fixture->Import(address, bytes);
    return true;
}

bool failingWaiter(std::uintptr_t, std::size_t) {
    throw std::runtime_error("expected pin waiter failure");
}

void waiterFailure(Fixture& fixture) {
    Allocation allocation(fixture);
    const auto imported = fixture.Import(allocation.Address(), allocation.bytes);
    GuestAllocations::GuestAllocationsSetPinWaiter_nid_postfix(&failingWaiter);
    bool failed = false;
    bool excluded = false;
    std::future<void> reader;
    {
        GuestAllocations::Mutation mutation;
        try {
            mutation.Protect(allocation.data, allocation.bytes, false, false, false, [] { throw std::runtime_error("host apply ran after a pin waiter failure"); });
        } catch (const std::runtime_error& error) {
            failed = std::string(error.what()) == "expected pin waiter failure";
        }
        std::promise<void> starting;
        auto ready = starting.get_future();
        reader = std::async(std::launch::async, [&] {
            starting.set_value();
            static_cast<void>(GuestAllocations::GuestAllocationsAcquire_nid_postfix());
        });
        ready.wait();
        excluded = reader.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
        const auto range = mutation.Find(allocation.data);
        Require(range.readable && range.writable && range.gpu, "a pin waiter failure changed the registered permissions");
    }
    reader.get();
    Require(failed, "a pin waiter error did not propagate");
    Require(excluded, "a failed pin waiter did not restore the registry lock");
    Require(!imported->freed.load(), "a failed pin waiter retired an import");
}

void reimportBeforeRelock(Fixture& fixture) {
    Allocation allocation(fixture);
    const auto imported = fixture.Import(allocation.Address(), allocation.bytes);
    MutationRace race{&fixture, &allocation};
    mutationRace = &race;
    GuestAllocations::GuestAllocationsSetPinWaiter_nid_postfix(&raceWaiter);
    {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(allocation.data, allocation.bytes, [&](const void*, std::size_t, const void*, bool) {
            Require(race.calls == 2 && race.replacement != nullptr, "guest mutation did not recheck an import acquired before registry relock");
            requireFreed(imported);
            requireFreed(race.replacement);
            allocation.Release();
        });
        allocation.registered = false;
    }
    mutationRace = nullptr;
}

// Runs a mode as upstream main runs --host-import-lifetime-only and --host-import-mutation-race-only: the test device
// alone, then the mode's own fixture (its recorder, the release probe).
template <typename Test>
int RunLifetime(Test test) {
    @autoreleasepool {
        try {
            Device device;
            const auto& context = device.GetContext();
            if (context.hostImportAlignment == 0) {
                std::puts("skipped, the device has no host imports (no-copy MTLBuffers)");
                return 77;
            }
            Fixture fixture(context);
            test(fixture);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}

const bool hostImportLifetime = AgcDriver::MetalTwins::RegisterRecorderMode("--host-import-lifetime-only", [] {
    return RunLifetime([](Fixture& fixture) {
        descriptorFirstImport(fixture);
        idleUnmap(fixture);
        pendingRead(fixture, false, false);
        pendingRead(fixture, true, false);
        pendingRead(fixture, false, true);
        pendingRead(fixture, true, true);
        partialUnmap(fixture);
        protectRollback(fixture);
        completedLeaseUnderLock(fixture);
        mutationDuringCompletion(fixture);
        blockedFree(fixture);
        std::puts("failed import construction and a failed production pin waiter submit not tested: Metal has no failure seam for them (TechnicalDebt metal-host-import-failure-seams)");
        deferredRetirement(fixture);
        std::puts("host import lifetime tests passed");
    });
});

const bool hostImportMutationRace = AgcDriver::MetalTwins::RegisterRecorderMode("--host-import-mutation-race-only", [] {
    return RunLifetime([](Fixture& fixture) {
        waiterFailure(fixture);
        reimportBeforeRelock(fixture);
        std::puts("host import mutation race tests passed");
    });
});

}
