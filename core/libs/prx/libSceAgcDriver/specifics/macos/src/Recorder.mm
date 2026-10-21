// The AGC driver's command recorder on Metal 4 (work package D2, docs/mac/M2_DESIGN.md §3, §6).
// Twin of Graphics/src/Recorder.cpp: the CPU-side logic (pending writes and reads, the label
// table, completion actions, kept objects, deferred releases, statistics and reports) is the
// upstream code unchanged; what talked to Vulkan now records into Metal::CommandStream and submits
// to the device's MTL4 command queue. A batch is one MTL4 command buffer with its own allocator,
// committed with a feedback handler and followed by a signal of the recorder's MTLSharedEvent with
// the batch serial (the timeline: Vulkan fences and the timeline semaphore are both that value).
// Store runs (labels) are uploads copied into the host imports; [gputime] ranges are MTL4 counter
// heap timestamps; occlusion samples are visibility-result counters of the render passes, folded
// by the MSL twins of the SampleCounter/SampleDumps kernels (RecorderKernels.h).
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/Progress.hpp"
#include "MetalCommandStream.h"
#include "MetalVmTrace.h"
#include "prx/libSceAgcDriver/Execution/include/AsyncLog.hpp"
#include "MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libkernel/Time/include/SleepWake.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelCompare.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceOverlay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4Opcodes.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include "prx/libc/include/general/AtomicSharedPtr.hpp"
#include "MetalResidency.h"

namespace AgcDriver::Graphics {

namespace {

// The Recorder has a member named `id`, which hides Objective-C's `id` inside its member
// functions: they name Metal object types through these aliases.
using MetalBuffer = id<MTLBuffer>;
using MetalRenderEncoder = id<MTL4RenderCommandEncoder>;
using MetalCommandBuffer = id<MTL4CommandBuffer>;
using MetalPipeline = id<MTLComputePipelineState>;
using MetalEvent = id<MTLSharedEvent>;
using MetalFeedback = id<MTL4CommitFeedback>;

Recorder* activeRecorder = nullptr;

bool DrawProfiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profiled;
}

std::uint64_t syncCounts[5] = {};
// Fence wait time by sync source (APS5_PROFILE_DRAW), and the source CountSync announced for the
// Sync/SyncThrough that follows it on this thread.
double syncWaitedMs[5] = {};
thread_local int announcedSource = 4;
// The call site CountSync named for that sync (nullptr: none; the sync's own return address is
// taken then), and the fence-wait table per (source, site) it feeds (APS5_PROFILE_DRAW, under the
// GpuMutex like finish()): which caller's syncs wait, when the source alone does not say (source
// 0 is every WaitIdle and drain). `activeSyncSite` is the entry of the sync in progress on this
// thread (an index: a nested sync from a completion replaces and restores it), so finish() charges
// each batch's wait to it.
thread_local const void* announcedSite = nullptr;
struct SyncSiteWaits {
    int source;
    const void* site;
    std::uint64_t syncs;
    std::uint64_t batches;
    double waitedMs;
};
std::vector<SyncSiteWaits> syncSites;
constexpr std::size_t NoSyncSite = static_cast<std::size_t>(-1);
constexpr std::size_t SyncSiteLimit = 48;
thread_local std::size_t activeSyncSite = NoSyncSite;

bool SyncSitesProfiled() {
    // APS5_NO_SYNC_SITES=1 leaves only the per-source and per-thread counts.
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr && std::getenv("APS5_NO_SYNC_SITES") == nullptr;
    return profiled;
}

// Begins a sync's attribution: returns the previous active entry for the caller to restore, after
// counting the sync under (source, site). Beyond the table's limit every further site shares the
// last entry (a nullptr site), so the table stays small.
std::size_t BeginSyncSite(int source, const void* site) {
    const auto previous = activeSyncSite;
    if (!SyncSitesProfiled()) return previous;
    auto it = std::find_if(syncSites.begin(), syncSites.end(), [&](const SyncSiteWaits& entry) { return entry.source == source && entry.site == site; });
    if (it == syncSites.end()) {
        if (syncSites.size() >= SyncSiteLimit) {
            it = std::find_if(syncSites.begin(), syncSites.end(), [&](const SyncSiteWaits& entry) { return entry.source == source && entry.site == nullptr; });
            if (it == syncSites.end()) it = syncSites.insert(syncSites.end(), SyncSiteWaits{source, nullptr, 0, 0, 0});
        } else {
            it = syncSites.insert(syncSites.end(), SyncSiteWaits{source, site, 0, 0, 0});
        }
    }
    ++it->syncs;
    activeSyncSite = static_cast<std::size_t>(it - syncSites.begin());
    return previous;
}

void CountSiteWait(double ms) {
    if (activeSyncSite == NoSyncSite || activeSyncSite >= syncSites.size()) return;
    auto& entry = syncSites[activeSyncSite];
    ++entry.batches;
    entry.waitedMs += ms;
}

// The sites by wait, longest first, as "source@+offset syncs/batches/wait" (offsets symbolize with
// nm against the driver's image like the [guestmem] callers; +0x0 is the overflow entry).
std::string SyncSiteReport() {
    static const char* const names[5] = {"idle", "pending-write", "recorded-store", "address-based", "other"};
    std::vector<const SyncSiteWaits*> order;
    for (const auto& entry : syncSites) order.push_back(&entry);
    std::sort(order.begin(), order.end(), [](const SyncSiteWaits* a, const SyncSiteWaits* b) { return a->waitedMs > b->waitedMs; });
    std::string report;
    for (std::size_t i = 0; i < order.size() && i < 8; ++i) {
        char text[96];
        std::snprintf(text, sizeof(text), " %s@+0x%llx %llu/%llu/%.1fs", names[order[i]->source], order[i]->site != nullptr ? GuestMemory::CodeOffset(order[i]->site) : 0ull, static_cast<unsigned long long>(order[i]->syncs), static_cast<unsigned long long>(order[i]->batches), order[i]->waitedMs / 1000);
        report += text;
    }
    return report;
}
// Fence waits by thread and source (APS5_PROFILE_DRAW), keyed by the thread's GpuMutex queue tag:
// a sync waits for the GPU under the mutex, so which worker's syncs (and which kind) hold it against
// queue 0 has to be visible. Its own mutex: the table is read while the report line is printed.
struct ThreadSyncs {
    std::uint32_t tag;
    std::array<std::uint64_t, 5> counts;
    std::array<double, 5> waitedMs;
};
std::vector<ThreadSyncs> threadSyncs;
std::mutex threadSyncsMutex;

// This thread's fence and timeline waits in total (Recorder::ThreadWaitedMs): a caller times a span
// of its own work and reads the difference to learn how much of it was waiting for the GPU.
thread_local double threadWaitedMs = 0;

void CountThreadSync(int source, double ms) {
    threadWaitedMs += ms;
    const auto tag = GuestMemory::GpuLockThreadTag();
    std::lock_guard lock(threadSyncsMutex);
    auto it = std::find_if(threadSyncs.begin(), threadSyncs.end(), [&](const ThreadSyncs& thread) { return thread.tag == tag; });
    if (it == threadSyncs.end()) it = threadSyncs.insert(threadSyncs.end(), ThreadSyncs{tag, {}, {}});
    ++it->counts[source];
    it->waitedMs[source] += ms;
}

std::string ThreadSyncReport() {
    static const char* const names[5] = {"idle", "pending-write", "recorded-store", "address-based", "other"};
    std::string report;
    std::lock_guard lock(threadSyncsMutex);
    for (const auto& thread : threadSyncs) {
        char text[64];
        if (thread.tag == 0xffffffffu) std::snprintf(text, sizeof(text), " untagged:");
        else std::snprintf(text, sizeof(text), " queue 0x%x:", thread.tag);
        report += text;
        for (int source = 0; source < 5; ++source) {
            if (thread.counts[source] == 0) continue;
            std::snprintf(text, sizeof(text), " %s %llu/%.1fs", names[source], static_cast<unsigned long long>(thread.counts[source]), thread.waitedMs[source] / 1000);
            report += text;
        }
    }
    return report;
}
// Unlocked timeline waits (WaitSerial): count and time, for the [recorder] line.
std::atomic<std::uint64_t> unlockedWaits{0}, unlockedWaitedUs{0};
// vkQueueSubmit calls and their time (APS5_PROFILE_DRAW; Submit runs under the GpuMutex).
std::uint64_t submitCount = 0;
double submitUs = 0, submitMaxUs = 0;
// Threads inside a timeline wait with the GpuMutex released (WaitSerial, syncThroughUnlocked): a
// recorder's teardown, after its Sync() made every wait satisfiable, waits until they left the
// semaphore before destroying it. Counted per recorder id (a small fixed array; the waiter cannot
// touch a recorder that may be dying, so it is keyed by the id it copied out), so a replaced
// device's old recorder does not spin on the new device's in-flight waits.
constexpr std::size_t WaiterSlots = 16;
std::atomic<int> unlockedWaiters[WaiterSlots]{};
std::atomic<int>& WaitersOf(std::uint64_t id) { return unlockedWaiters[id % WaiterSlots]; }
// Recorders alive, by id (own mutex, taken under the GpuMutex or with nothing held): a thread that
// released the GpuMutex around a wait learns whether its recorder still exists before touching it.
std::mutex liveRecordersMutex;
std::vector<std::uint64_t> liveRecorders;
std::atomic<std::uint64_t> nextRecorderId{1};
std::atomic<std::uint64_t> samplesPassed{0};
constexpr std::uint32_t SampleFoldCapacity = 64;
constexpr std::uint32_t SampleCounterPushBytes = 24;
constexpr std::uint32_t SegmentSlots = 128;
constexpr std::uint32_t SegmentFoldCapacity = 1024;
constexpr std::uint32_t DumpFoldCapacity = 1024;
constexpr std::uint32_t SampleDumpBytes = 15 * 16 + 8;
constexpr VkDeviceSize DumpListOffset = 16 + 8 * SegmentFoldCapacity;
constexpr std::uint32_t SampleDumpPushBytes = 40;
std::atomic<std::uint64_t> dumpsQueued{0}, dumpsQueuedInPass{0}, dumpsAtOnce{0}, segmentsClosed{0}, dumpFolds{0};

// macOS: an occlusion segment pool is a visibility result buffer of SegmentSlots 8-byte counters
// (a render pass accumulates its sampled draws into one slot, see Recorder::SampleVisibilityBuffer),
// resident through the recorder's residency set and recycled once its batch completed (zeroed by
// takeSegmentPool before reuse).
struct SamplePoolCache {
    // The recorder's residency set every pool joins when made and leaves when dropped.
    id<MTLResidencySet> residency = nil;
    std::mutex mutex;
    std::vector<id<MTLBuffer>> free;
    bool closed = false;
    void put(id<MTLBuffer> pool) {
        std::lock_guard lock(mutex);
        if (!closed && free.size() < 256) {
            free.push_back(pool);
            return;
        }
        ::AgcDriver::Metal::Residency::Remove(residency, pool);
        ::AgcDriver::Metal::Residency::Commit(residency);
    }
    id<MTLBuffer> take() {
        std::lock_guard lock(mutex);
        if (free.empty()) return nil;
        id<MTLBuffer> pool = free.back();
        free.pop_back();
        return pool;
    }
    void close() {
        std::lock_guard lock(mutex);
        closed = true;
        for (id<MTLBuffer> pool : free) ::AgcDriver::Metal::Residency::Remove(residency, pool);
        if (!free.empty()) ::AgcDriver::Metal::Residency::Commit(residency);
        free.clear();
    }
};

struct SamplePool {
    SamplePool(std::shared_ptr<SamplePoolCache> cache, id<MTLBuffer> pool) : cache(std::move(cache)), pool(pool) {}
    SamplePool(const SamplePool&) = delete;
    SamplePool& operator=(const SamplePool&) = delete;
    ~SamplePool() { cache->put(pool); }
    std::shared_ptr<SamplePoolCache> cache;
    id<MTLBuffer> pool;
};

bool RecorderAlive(std::uint64_t id) {
    std::lock_guard lock(liveRecordersMutex);
    return std::find(liveRecorders.begin(), liveRecorders.end(), id) != liveRecorders.end();
}

// What commit feedback reports about one batch. The handler runs on a Metal thread and may run
// after the batch's event value was signaled (measured on 27.0.1: it lags), so finish() reads it
// when present and a failure reported late is raised by a later finish() through the sink.
struct BatchFeedback {
    std::atomic<bool> delivered{false};
    double gpuStartSeconds = 0;
    double gpuEndSeconds = 0;
};

struct FeedbackSink {
    std::mutex mutex;
    std::vector<std::pair<std::uint64_t, std::string>> errors;
};

// The longest a recorder waits for one batch before it reports the GPU hung and throws (the
// upstream fence wait loops forever, reporting every 5 s; a wedged Apple GPU can take the
// WindowServer down, so the Metal driver gives up). APS5_METAL_BATCH_TIMEOUT_MS, default 20 s.
std::uint64_t BatchTimeoutMs() {
    static const std::uint64_t limit = [] {
        const char* value = std::getenv("APS5_METAL_BATCH_TIMEOUT_MS");
        const auto parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{20000};
    }();
    return limit;
}

// Waits until `event` reaches `value`, reporting every 5 s like the upstream fence waits; false
// once BatchTimeoutMs passed.
bool WaitEvent(id<MTLSharedEvent> event, std::uint64_t value, const char* what) {
    const auto limit = BatchTimeoutMs();
    for (std::uint64_t waited = 0; waited < limit;) {
        const auto slice = std::min<std::uint64_t>(5000, limit - waited);
        if ([event waitUntilSignaledValue:value timeoutMS:slice]) return true;
        waited += slice;
        if (waited < limit) std::fprintf(stderr, "[gpu] recorded batch %llu still running on the GPU after %llu s (%s)\n", static_cast<unsigned long long>(value), static_cast<unsigned long long>(waited / 1000), what);
    }
    return event.signaledValue >= value;
}

}

namespace Metal {

struct BatchObjects {
    id<MTL4CommandBuffer> commandBuffer = nil;
    id<MTL4CommandAllocator> allocator = nil;
    std::unique_ptr<CommandStream> stream;
    // Timestamps of the batch's [gputime] ranges (BeginGpuTiming), resolved on the CPU at finish.
    id<MTL4CounterHeap> timestamps = nil;
    std::shared_ptr<BatchFeedback> feedback;
    VkCommandBuffer Handle() { return Metal::Handle(*stream); }
};

struct RecorderObjects {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    // The timeline: every Submit signals the batch serial after its commit.
    id<MTLSharedEvent> timeline = nil;
    // The recorder's own long-lived buffers (occlusion visibility pools), attached to the queue.
    id<MTLResidencySet> residency = nil;
    std::shared_ptr<UploadArena> arena;
    std::shared_ptr<FeedbackSink> sink = std::make_shared<FeedbackSink>();
    double nanosecondsPerTick = 1.0;
    // The objects of completed batches, reused by later ones (hundreds of batches per frame would
    // otherwise make their command buffers, allocators and streams each time).
    std::vector<std::unique_ptr<BatchObjects, ObjectsDeleter>> spare;
    // Two-entry timestamp heaps of the batch stamps, reused when only those are taken.
    std::vector<id<MTL4CounterHeap>> spareHeaps;
    ~RecorderObjects() {
        if (queue != nil && residency != nil) ::AgcDriver::Metal::Residency::Detach(queue, residency);
    }
};

void ObjectsDeleter::operator()(RecorderObjects* objects) const noexcept {
    delete objects;
}

void ObjectsDeleter::operator()(BatchObjects* objects) const noexcept {
    delete objects;
}

namespace {
thread_local const char* failNextSubmit = nullptr;
thread_local bool failSubmitBeforeCommit = false;
}

void FailNextRecorderSubmit(const char* what, bool beforeCommit) {
    failNextSubmit = what;
    failSubmitBeforeCommit = beforeCommit;
}

}

namespace {

// One listener for every recorder's timeline notifications (D80), on a serial user-interactive queue: its blocks only
// raise a progress event and end the guest's short (<= 1 ms) sleeps (SleepWake.hpp; upstream's ProgressWatcher thread,
// 82dd286192: here the listener the timeline already notifies is that thread). Both are lock-free or take only their
// own small mutex, never GuestMemory::GpuMutex.
MTLSharedEventListener* ProgressListener() {
    static MTLSharedEventListener* listener = [[MTLSharedEventListener alloc]
        initWithDispatchQueue:dispatch_queue_create("aps5.gpu-progress", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0))];
    return listener;
}

MTLSharedEventNotificationBlock ProgressBlock() {
    static MTLSharedEventNotificationBlock block = ^(MetalEvent, uint64_t) {
        ::AgcDriver::Progress::Notify();
        KernelWakeSleepers_nid_no_patch();
    };
    return block;
}

// The handle the shared code passes around for a batch's commands (a CommandStream's address).
VkCommandBuffer CommandsOf(const std::unique_ptr<Metal::BatchObjects, Metal::ObjectsDeleter>& objects) {
    return objects->Handle();
}

// Completion actions in progress on this thread (finish()): a pending-write sync the flush hook
// makes from inside one (a copied buffer's write-back store) is skipped, see SyncThrough.
thread_local int completionDepth = 0;
// This thread's hook syncs that waited for an unsignaled target (Recorder::ThreadHookWaits).
thread_local std::uint64_t hookRealWaits = 0;

}

bool Recorder::InCompletion() {
    return completionDepth != 0;
}

namespace {

// APS5_NO_HOOK_COMPLETION_GUARD=1: the flush hook records the accessing worker's queued labels
// even from inside a completion action, as before (see FlushForAccess).
bool HookCompletionGuard() {
    static const bool guard = std::getenv("APS5_NO_HOOK_COMPLETION_GUARD") == nullptr;
    return guard;
}

// APS5_COMPLETION_STORE_SYNC=1: a store made by a completion action waits for later batches that
// note its range, as before (the wait runs under whichever hold reaped the batch).
bool CompletionStoreSyncs() {
    static const bool enabled = std::getenv("APS5_COMPLETION_STORE_SYNC") != nullptr;
    return enabled;
}

// APS5_HOOK_LOCKED_WAIT=1: the flush hook's pending-write wait stays under the GpuMutex as before.
bool HookLockedWait() {
    static const bool locked = std::getenv("APS5_HOOK_LOCKED_WAIT") != nullptr;
    return locked;
}

// APS5_HOOK_FULL_SYNC=1: the flush hook's recorded-store wait is a full Sync() under the GpuMutex
// (every batch in flight) as before, instead of SyncThrough's targeted, unlocked wait (H1).
bool HookFullSync() {
    static const bool hookFullSync = std::getenv("APS5_HOOK_FULL_SYNC") != nullptr;
    return hookFullSync;
}

// APS5_HOOK_FLUSH_CPU_BLOCKS=0: an access whose every 64 KiB block the CPU wrote since the
// overlapping pending images' generations flushes them as before, instead of the C4 skip (H2:
// StorageTexture::AccessKeptByCpu).
bool HookFlushCpuBlocks() {
    static const bool hookFlushCpuBlocks = [] {
        const char* text = std::getenv("APS5_HOOK_FLUSH_CPU_BLOCKS");
        return text != nullptr && text[0] == '0';
    }();
    return hookFlushCpuBlocks;
}

// What the holds spend on the recorder (APS5_PROFILE_DRAW, all under the GpuMutex, printed at the
// end of the [recorder] line): completions and their time (of which releasing the kept objects),
// pending-write syncs from inside completions (skipped, or waited with the kill switch), reaps and
// the batches they retired, and the flush hook's unlocked waits (with those that found the
// recorder torn down when they retook the mutex, and those that had to wait locked).
struct HoldCounters {
    std::uint64_t completions = 0;
    double completionMs = 0;
    double keptReleaseMs = 0;
    std::uint64_t completionSyncsSkipped = 0;
    std::uint64_t completionSyncsWaited = 0;
    double completionSyncWaitMs = 0;
    std::uint64_t reaps = 0;
    std::uint64_t reapsWithWork = 0;
    std::uint64_t reapBatches = 0;
    double reapMs = 0;
    std::uint64_t hookUnlockedWaits = 0;
    // The GPU wait alone (timeline wait with the mutex released) and, separately, the relock's
    // own wait for the GpuMutex (also counted by the [lock] line under 'hook'), so the two
    // remaining costs, GPU latency and hold contention, stay apart.
    double hookUnlockedWaitMs = 0;
    double hookRelockMs = 0;
    // The same by sync source (1 pending write, 2 recorded store).
    std::array<std::uint64_t, 5> hookUnlockedWaitsBySource{};
    std::array<double, 5> hookUnlockedWaitMsBySource{};
    std::uint64_t hookUnlockedTornDown = 0;
    // Genuine nested hooks (depth >= 2 outside a completion action) that had to wait locked.
    std::uint64_t hookLockedWaits = 0;
};
HoldCounters holdCounters;

// Deferred release of a finished batch's kept objects (and its completion actions, whose captures
// hold some of the same objects): finish() moves them to this list of the finishing thread instead
// of destroying them under the mutex, and the GpuMutex unlock hook (ReleaseDeferredKeeps, set by
// the constructor) hands the list to the release thread once the thread gave up its outermost
// hold. Destroying a ShaderResources with its buffers, descriptor set, textures and pipelines cost
// ~1.5 ms per reap with work, inside whichever hold reaped ('kept objects released', 15.9 s per
// 200 s run), and most reaps are queue 0's own (ReapRecorded before its dispatches), so a release
// on the unlocking thread would still be the frame's time. The destructors touch only their own
// caches' mutexes and the device, never the recorder or GpuMutex, so they need no hold and no
// particular thread (every Vulkan object is externally synchronized per object, and its cache's
// mutex covers its pool); they need the device alive, which ~Recorder guarantees by draining its
// own thread's list, stopping and joining the release thread (it drains the queue before it
// leaves) and waiting for every release still in progress on another thread (`deferredPending`:
// counted up under the mutex, down after each batch's objects are gone) before the device goes.
// The queue is bounded (APS5_RELEASE_QUEUE_MAX batches, default 256): a thread whose hand-off
// would exceed it destroys its batches itself, as it did before the release thread, so memory
// cannot grow behind a release thread that falls behind.
// APS5_RELEASE_UNDER_LOCK=1 destroys them in finish() as before; APS5_RELEASE_ON_UNLOCK=1 destroys
// them on the unlocking thread (no release thread).
struct DeferredBatch {
    std::vector<std::shared_ptr<void>> kept;
    std::vector<std::function<void()>> completions;
};
struct DeferredBatchesTag {};
auto& DeferredBatches() { return HostThreadLocal<std::vector<DeferredBatch>, DeferredBatchesTag>(); }
struct DeferredBytesTag {};
auto& DeferredBytes() { return HostThreadLocal<std::size_t, DeferredBytesTag>(); }
std::vector<DeferredBatch> TakeDeferred() {
    auto taken = std::move(DeferredBatches());
    DeferredBatches().clear();
    DeferredBytes() = 0;
    return taken;
}
std::atomic<std::uint64_t> deferredPending{0};
std::atomic<std::uint64_t> threadReleases{0}, threadObjects{0}, threadReleaseUs{0};
std::atomic<std::uint64_t> inlineReleases{0}, inlineObjects{0}, inlineReleaseUs{0}, inlineOverBound{0};
std::atomic<std::uint64_t> releaseQueueMax{0};

bool ReleaseUnderLock() {
    static const bool locked = std::getenv("APS5_RELEASE_UNDER_LOCK") != nullptr;
    return locked;
}

bool ReleaseOnUnlock() {
    static const bool onUnlock = std::getenv("APS5_RELEASE_ON_UNLOCK") != nullptr;
    return onUnlock;
}

std::size_t ReleaseQueueBound() {
    static const std::size_t bound = [] {
        const char* value = std::getenv("APS5_RELEASE_QUEUE_MAX");
        const auto parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::size_t>(parsed) : std::size_t{256};
    }();
    return bound;
}

// Destroys deferred batches, on the release thread (`onThread`) or on the thread that deferred
// them, and counts each batch off `deferredPending` only once its objects are gone: ~Recorder
// waits for that count.
void DestroyDeferred(std::vector<DeferredBatch> releasing, bool onThread) {
    if (releasing.empty()) return;
    const bool profile = DrawProfiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto count = releasing.size();
    std::uint64_t objects = 0;
    for (auto& batch : releasing) {
        objects += batch.kept.size();
        // The completions first (their captures hold some of the objects), then the objects.
        batch.completions.clear();
        batch.kept.clear();
        deferredPending.fetch_sub(1, std::memory_order_acq_rel);
    }
    releasing.clear();
    if (profile) {
        const auto us = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        (onThread ? threadReleases : inlineReleases).fetch_add(count, std::memory_order_relaxed);
        (onThread ? threadObjects : inlineObjects).fetch_add(objects, std::memory_order_relaxed);
        (onThread ? threadReleaseUs : inlineReleaseUs).fetch_add(us, std::memory_order_relaxed);
    }
}

// The release thread's queue: one for the process. The thread is started at the first hand-off and
// stopped and joined by ~Recorder (JoinReleaseThread), after which the next hand-off starts it
// again (a device replacement makes a new recorder). The queue object is leaked so that a static
// destructor never races the thread: at process exit the thread is idle on its condition variable
// (every ~Recorder joined it) or, when no ~Recorder ran, mid-release like any worker thread. Nothing
// is done under `mutex` but the hand-off and the take; `joinMutex` serializes joiners and orders
// before `mutex`; neither is ever taken by a kept object's destructor.
struct ReleaseQueue {
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<DeferredBatch> items;
    std::thread thread;
    bool started = false;
    // Set by a joiner under `mutex` until the thread was joined: the thread leaves once the queue
    // is empty, and hand-offs meanwhile destroy inline (nothing may be queued without a taker).
    bool stop = false;
    std::mutex joinMutex;
};

ReleaseQueue& ReleaseThreadQueue() {
    static ReleaseQueue* const queue = new ReleaseQueue;
    return *queue;
}

void ReleaseThreadMain() {
    CpuTopology::PinHelperThread("release thread");
    auto& queue = ReleaseThreadQueue();
    std::unique_lock lock(queue.mutex);
    for (;;) {
        queue.wake.wait(lock, [&] { return !queue.items.empty() || queue.stop; });
        if (queue.items.empty()) return;
        auto items = std::move(queue.items);
        queue.items.clear();
        lock.unlock();
        DestroyDeferred(std::move(items), true);
        lock.lock();
    }
}

// Stops the release thread once it emptied the queue and joins it (~Recorder, before its device
// goes). Hand-offs made while the thread stops destroy inline, so nothing can sit in the queue
// without a taker; a later hand-off starts the thread again.
void JoinReleaseThread() {
    auto& queue = ReleaseThreadQueue();
    std::lock_guard joining(queue.joinMutex);
    std::thread worker;
    {
        std::lock_guard lock(queue.mutex);
        if (!queue.started) return;
        queue.stop = true;
        worker = std::move(queue.thread);
    }
    queue.wake.notify_all();
    worker.join();
    std::lock_guard lock(queue.mutex);
    queue.started = false;
    queue.stop = false;
}

// The unlock hook: hands what this thread deferred to the release thread, or destroys it here
// (APS5_RELEASE_ON_UNLOCK=1, the queue is full or stopping, or the hand-off cannot be made).
void ReleaseDeferredKeeps() {
    if (DeferredBatches().empty()) return;
    // Taken off the thread's list first: a destructor that took and released the mutex (none is
    // known to) would re-enter here and must find nothing.
    auto releasing = TakeDeferred();
    if (ReleaseOnUnlock()) {
        DestroyDeferred(std::move(releasing), false);
        return;
    }
    auto& queue = ReleaseThreadQueue();
    const bool profile = DrawProfiled();
    bool handedOff = false;
    bool overBound = false;
    try {
        std::lock_guard lock(queue.mutex);
        if (queue.items.size() + releasing.size() > ReleaseQueueBound()) {
            overBound = true;
        } else if (!queue.stop) {
            // The thread before the items: items queued with no thread to take them would keep
            // `deferredPending` up and ~Recorder waiting forever.
            if (!queue.started) {
                queue.thread = std::thread(&ReleaseThreadMain);
                queue.started = true;
            }
            // The allocation before any move: a failure here leaves `releasing` intact for the
            // fallback below, and the moves cannot throw.
            queue.items.reserve(queue.items.size() + releasing.size());
            for (auto& batch : releasing) queue.items.push_back(std::move(batch));
            handedOff = true;
            if (profile) {
                const auto queued = static_cast<std::uint64_t>(queue.items.size());
                auto seen = releaseQueueMax.load(std::memory_order_relaxed);
                while (queued > seen && !releaseQueueMax.compare_exchange_weak(seen, queued, std::memory_order_relaxed)) {
                }
            }
        }
    } catch (...) {
    }
    if (!handedOff) {
        if (profile && overBound) inlineOverBound.fetch_add(releasing.size(), std::memory_order_relaxed);
        DestroyDeferred(std::move(releasing), false);
        return;
    }
    queue.wake.notify_one();
}
// Lock-free state of the active recorder's open batch, read by the queue workers between packets
// and inside WAIT_REG_MEM polls (see the static readers in Recorder.hpp). Written under the mutex.
constexpr std::int64_t NoPendingLabel = std::numeric_limits<std::int64_t>::min();
std::atomic<std::int64_t> pendingLabelSince{NoPendingLabel};
std::atomic<std::uint64_t> writeGeneration{0};
std::atomic<std::uint64_t> publishGeneration{0};
std::atomic<std::uint64_t> completionLabels{0};
std::atomic<std::uint64_t> writeBackCompletions{0};
std::atomic<std::uint64_t> completionStoresSkipped{0};
std::atomic<std::uint64_t> completionStoresRun{0};
// Write-backs noted over a tracked label dword (NoteWrittenBack): a CPU store over a GPU label.
std::atomic<std::uint64_t> writeBacksOverLabels{0};
// Completion labels that entered the pending count at a write-back (see NoteWrittenBack).
std::atomic<std::uint64_t> completionLabelsCountedLate{0};
std::atomic<std::uint64_t> workSinceSubmit{0};

// APS5_COUNT_ALL_COMPLETION_LABELS=1: a completion label also stored on the GPU counts as pending
// from its registration, as before (see Recorder::AfterCompletions).
bool CountAllCompletionLabels() {
    static const bool all = std::getenv("APS5_COUNT_ALL_COMPLETION_LABELS") != nullptr;
    return all;
}

// APS5_NO_PROC_TABLE=1: the recorder resolves its Vulkan entry points per call, as before.
bool ProcTable() {
    static const bool table = std::getenv("APS5_NO_PROC_TABLE") == nullptr;
    return table;
}
// The pending-label table's own mutex: every mutation of a recorder's `labels` holds it (under the
// GPU mutex), so a WAIT_REG_MEM looks a label up with this small lock alone (Recorder::LookupLabel)
// instead of queueing behind a dispatch. `labelTableOwner` is the active recorder while it lives,
// set and cleared under this mutex, so a lookup never touches a table being destroyed. Nothing
// under this mutex takes the GPU mutex (no lock-order cycle).
std::mutex labelTableMutex;
Recorder* labelTableOwner = nullptr;
// The ranges of the calling worker's queued labels (NoteQueuedLabel, cleared by
// ForgetQueuedLabels): the flush hook, on the same thread, has them recorded before an access that
// overlaps one, through the function the driver installed.
struct QueuedLabelRangesTag {};
auto& QueuedLabelRanges() { return HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, QueuedLabelRangesTag>(); }
std::atomic<void (*)()> queuedLabelRecorder{nullptr};
// The dwords (with their stamps) this thread noted since its last CloseLabelGroup (see Recorder::CloseLabelGroup).
struct LabelGroupDwordsTag {};
auto& LabelGroupDwords() { return HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, LabelGroupDwordsTag>(); }
// Late-rule counters (Recorder::LateCounts), per thread: lookups run on the waiting worker.
thread_local Recorder::LateStatistics lateCounts{};

// APS5_LABEL_TRUST_LATE=1 trusts entries with stamp <= afterStamp under the late rule; off by
// default until the frame-rate collapse it produced at the video stage (lt_on: 119 -> 25 presents
// per 10 s with 0 late-trusted waits) is understood.
bool LateTrustEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_LABEL_TRUST_LATE");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}
// Store-run and queued-label counters (Recorder::StoreCounts), relaxed: they are only reported.
std::atomic<std::uint64_t> storeCount{0}, storeRuns{0}, storesJoined{0}, storesReplaced{0}, storeWawBarriers{0}, storeJoinsRefused{0};
std::atomic<std::uint64_t> keyStoreCount{0}, keyStoreRuns{0}, keyStoreRunsForWriter{0}, keyStoresJoined{0};

// Debug aid: APS5_DCC_KEYS_EACH=1 records every DCC key store at once (see Recorder::QueueKeyStore).
bool KeyStoresEach() {
    static const bool each = std::getenv("APS5_DCC_KEYS_EACH") != nullptr;
    return each;
}
std::atomic<std::uint64_t> queuedLabelsNoted{0}, queuedLabelsOverRecorded{0}, queuedLabelHits{0}, queuedLabelHookRecords{0}, queuedLabelHookInCompletion{0};
// Read-tracking counters (Recorder::ReadCounts), relaxed: they are only reported.
constexpr std::size_t ReadKinds = static_cast<std::size_t>(Recorder::ReadKind::Count);
std::atomic<std::uint64_t> readsNoted{0}, readQueries{0}, readStaleIgnored{0};
std::atomic<std::uint64_t> readHits[ReadKinds]{};

bool ReadTrackingEnabled() {
    // APS5_COPY_READ_TRACKING=0: no in-place reads are noted; the copy HLE's CPU path then requires
    // an idle recorder (VulkanDevice::CopyBuffer), never less.
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_COPY_READ_TRACKING");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool LabelRunsEnabled() {
    // Debug aid: APS5_NO_LABEL_RUNS=1 records every store between a barrier pair of its own
    // (APS5_NO_LABEL_BATCHING=1 turns off every label-batching change of Driver.cpp too).
    static const bool enabled = std::getenv("APS5_NO_LABEL_RUNS") == nullptr && std::getenv("APS5_NO_LABEL_BATCHING") == nullptr;
    return enabled;
}

// One store run per batch, recorded at Submit (see Recorder::RecordStore). Debug aid:
// APS5_LABEL_RUNS_INLINE=1 records the stores at once and closes the run at the next command.
bool LabelRunsPerBatch() {
    static const bool perBatch = LabelRunsEnabled() && std::getenv("APS5_LABEL_RUNS_INLINE") == nullptr;
    return perBatch;
}
std::atomic<std::uint64_t> storeRunsAtSubmit{0}, storeRunsForced{0};

// APS5_NO_JOIN_WAW_CHECK=1: a contiguous store joins the pending one without the check against
// the run's recorded stores, as before (see RecordStore).
bool JoinWawCheck() {
    static const bool check = std::getenv("APS5_NO_JOIN_WAW_CHECK") == nullptr;
    return check;
}

// APS5_NO_SEPARATE_QUEUED_LABELS=1: a queued label enters the recorded table and replaces a
// recorded entry of its dword, as before (see NoteQueuedLabel).
bool SeparateQueuedLabels() {
    static const bool separate = std::getenv("APS5_NO_SEPARATE_QUEUED_LABELS") == nullptr;
    return separate;
}

bool QueuedLabelOverlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    for (const auto& [begin, finish] : QueuedLabelRanges()) {
        if (address < finish && begin < end) return true;
    }
    return false;
}
// Flush hook statistics: calls, calls whose snapshot check overlapped (the GpuMutex was taken),
// targeted syncs and the batches they left in flight.
std::atomic<std::uint64_t> hookCalls{0}, hookLocks{0}, targetedSyncs{0}, batchesLeftInFlight{0};
// Snapshot maintenance (under the GpuMutex): notes whose range the snapshot already covered (no
// rebuild), rebuilds, and the time the rebuilds took (APS5_PROFILE_DRAW), so their cost is visible.
std::uint64_t snapshotCovered = 0, snapshotRebuilds = 0;
double snapshotRebuildMs = 0;

using WriteRanges = Recorder::WriteRanges;
// The sorted, merged union of the guest ranges every unfinished batch will write. Rebuilt under
// GuestMemory::GpuMutex (the only writer) and read by the flush hook without it: the lock is the
// graphics worker's main stall, so a no-overlap access must not wait for another queue's device
// work. A range leaves the snapshot only after its batch's completions (CPU write-backs) ran, so a
// reader that sees no overlap either precedes the note (the queues are unordered then, as on the
// GPU) or follows the write-back.
AtomicSharedPtr<const WriteRanges> pendingWrites;

bool HookSnapshotEnabled() {
    // Debug aid: APS5_NO_HOOK_SNAPSHOT=1 takes the GpuMutex on every access as before.
    static const bool enabled = std::getenv("APS5_NO_HOOK_SNAPSHOT") == nullptr;
    return enabled;
}

bool SnapshotOverlaps(const WriteRanges* snapshot, std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || snapshot == nullptr || snapshot->empty()) return false;
    // Merged ranges are ordered by both bounds: the first one ending past the access decides.
    const auto it = std::partition_point(snapshot->begin(), snapshot->end(), [&](const auto& range) { return range.second <= address; });
    return it != snapshot->end() && it->first < address + bytes;
}

bool SnapshotOverlaps(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0) return false;
    const auto snapshot = pendingWrites.load(std::memory_order_acquire);
    return SnapshotOverlaps(snapshot.get(), address, bytes);
}

// Whether one merged range of the snapshot contains [address, end) entirely.
bool SnapshotCovers(const WriteRanges* snapshot, std::uint64_t address, std::uint64_t end) {
    if (snapshot == nullptr || snapshot->empty()) return false;
    const auto it = std::partition_point(snapshot->begin(), snapshot->end(), [&](const auto& range) { return range.second <= address; });
    return it != snapshot->end() && it->first <= address && end <= it->second;
}

bool SnapshotCovers(std::uint64_t address, std::uint64_t end) {
    const auto snapshot = pendingWrites.load(std::memory_order_acquire);
    return SnapshotCovers(snapshot.get(), address, end);
}

// Attribution of the pending-write syncs the hook makes (the [hooksync] line every 10 s, under
// APS5_PROFILE_DRAW; APS5_NO_HOOKSYNC_PROFILE=1 leaves only the plain counts). Every sync is charged
// to the packet the accessing thread executes and the read site it named (or, when it named none,
// its return addresses), together with the noted range it hit, the batch that noted it and the
// time waited. A small read (up to 64 KiB) has its bytes compared before and after the wait: a
// sync whose bytes come out unchanged although its target batch had not run yet was not needed
// for that read (the noted range was larger than what the GPU wrote, or the GPU wrote the same
// values), which separates false overlaps from real producer/consumer dependencies; see
// HookSyncOutcomes for the cases that prove nothing. All state lives under the GpuMutex, which
// the hook holds.
// Debug aid: APS5_NO_SYNC_THROUGH=1 makes every pending-write sync a full Sync (SyncThrough and
// the attribution's DescribePendingWrite read the same switch).
bool SyncThroughEnabled() {
    static const bool enabled = std::getenv("APS5_NO_SYNC_THROUGH") == nullptr;
    return enabled;
}

// Six frames: whether the hook's constructor and FlushGpuWrites are inlined into the GuestMemory
// entry decides at which frame the driver caller sits, so enough are kept to see its own caller.
constexpr std::size_t HookSyncFrames = 6;

struct HookSyncKey {
    std::uint32_t queue;
    std::uint32_t opcode;
    GuestMemory::ReadSite site;
    std::array<unsigned long long, HookSyncFrames> frames;
    bool operator<(const HookSyncKey& other) const {
        return std::tie(queue, opcode, site, frames) < std::tie(other.queue, other.opcode, other.site, other.frames);
    }
};

// Whether the bytes of a small access changed over the wait, split by the state of the target
// batch: only an unchanged access whose target had not run yet (open, or in flight with its fence
// unsignaled) shows the sync was not needed; against a signaled fence the copy made before the
// wait already held the GPU's values, so unchanged proves nothing. A store (ReadSite::Store) is
// counted apart: its sync orders a CPU write after the GPU's, whatever the bytes do.
struct HookSyncOutcomes {
    std::uint64_t unchangedOpen = 0;
    std::uint64_t unchangedPending = 0;
    std::uint64_t unchangedSignaled = 0;
    std::uint64_t changed = 0;
    std::uint64_t stores = 0;
    std::uint64_t unchecked = 0;
};

struct HookSyncTotals {
    std::uint64_t count = 0;
    HookSyncOutcomes outcomes;
    std::uint64_t rangeBytes = 0;
    std::uint64_t accessBytes = 0;
    double waitedMs = 0;
};

// Size buckets for the noted ranges and the accesses: <=4K, <=64K, <=1M, <=16M, <=256M, larger.
constexpr std::size_t SizeBuckets = 6;
constexpr const char* SizeBucketNames[SizeBuckets] = {"<=4K", "<=64K", "<=1M", "<=16M", "<=256M", ">256M"};

std::size_t SizeBucket(std::uint64_t bytes) {
    if (bytes <= 4096) return 0;
    if (bytes <= 65536) return 1;
    if (bytes <= (1u << 20u)) return 2;
    if (bytes <= (16u << 20u)) return 3;
    if (bytes <= (256u << 20u)) return 4;
    return 5;
}

struct HookSyncStats {
    std::map<HookSyncKey, HookSyncTotals> byKey;
    std::uint64_t count = 0, openTargets = 0, signaledTargets = 0, batchesFinished = 0;
    HookSyncOutcomes outcomes;
    double waitedMs = 0;
    std::array<std::uint64_t, SizeBuckets> rangeBuckets{};
    std::array<std::uint64_t, SizeBuckets> accessBuckets{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

HookSyncStats& HookSyncs() {
    static HookSyncStats stats;
    return stats;
}

bool HookSyncProfiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr && std::getenv("APS5_NO_HOOKSYNC_PROFILE") == nullptr;
    return profiled;
}

std::string PacketName(std::uint32_t opcode) {
    if (opcode == GuestMemory::NoPacket) return "no-packet";
    if (opcode == 0xffffu) return "flip";
    // 0xfffd is reserved for the deferred-label stores (recordDeferredLabels), once its owner
    // brackets them with SetCurrentPacket; until then they are charged to the following packet.
    if (opcode == 0xfffdu) return "deferred-labels";
    for (const auto& entry : Pm4::Opcodes) {
        if (entry.value == opcode) return std::string(entry.name);
    }
    char text[24];
    std::snprintf(text, sizeof(text), "op 0x%x", opcode);
    return text;
}

void ReportHookSyncs(HookSyncStats& stats) {
    std::vector<std::pair<const HookSyncKey*, const HookSyncTotals*>> hot;
    hot.reserve(stats.byKey.size());
    for (const auto& [key, totals] : stats.byKey) hot.emplace_back(&key, &totals);
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.second->waitedMs > b.second->waitedMs; });
    std::string report;
    char text[512];
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto& o = stats.outcomes;
    std::snprintf(text, sizeof(text), "[hooksync] %llu pending-write syncs waited %.0f ms (10 s); read bytes after the wait: unchanged %llu (target open %llu, in flight unsignaled %llu, signaled %llu), changed %llu; stores %llu, unchecked %llu; targets: %llu open, %llu already signaled, %llu batches finished; noted range:", count(stats.count), stats.waitedMs, count(o.unchangedOpen + o.unchangedPending + o.unchangedSignaled), count(o.unchangedOpen), count(o.unchangedPending), count(o.unchangedSignaled), count(o.changed), count(o.stores), count(o.unchecked), count(stats.openTargets), count(stats.signaledTargets), count(stats.batchesFinished));
    report += text;
    for (std::size_t i = 0; i < SizeBuckets; ++i) {
        if (stats.rangeBuckets[i] == 0) continue;
        std::snprintf(text, sizeof(text), " %s %llu", SizeBucketNames[i], static_cast<unsigned long long>(stats.rangeBuckets[i]));
        report += text;
    }
    report += "; access:";
    for (std::size_t i = 0; i < SizeBuckets; ++i) {
        if (stats.accessBuckets[i] == 0) continue;
        std::snprintf(text, sizeof(text), " %s %llu", SizeBucketNames[i], static_cast<unsigned long long>(stats.accessBuckets[i]));
        report += text;
    }
    report += "; top by wait (queue packet site frames: count/ms, unchanged open/unsignaled/signaled, changed, stores, avg range/access):";
    for (std::size_t i = 0; i < hot.size() && i < 10; ++i) {
        const auto& key = *hot[i].first;
        const auto& totals = *hot[i].second;
        const auto& k = totals.outcomes;
        std::snprintf(text, sizeof(text), " [0x%x %s %s +0x%llx/+0x%llx/+0x%llx/+0x%llx/+0x%llx/+0x%llx: %llu/%.0fms u%llu/%llu/%llu c%llu s%llu %.0fK/%.0fK]", key.queue, PacketName(key.opcode).c_str(), GuestMemory::ReadSiteName(key.site), key.frames[0], key.frames[1], key.frames[2], key.frames[3], key.frames[4], key.frames[5], count(totals.count), totals.waitedMs, count(k.unchangedOpen), count(k.unchangedPending), count(k.unchangedSignaled), count(k.changed), count(k.stores), totals.rangeBytes / 1024.0 / totals.count, totals.accessBytes / 1024.0 / totals.count);
        report += text;
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", report.c_str());
    stats = HookSyncStats{};
}

// Wraps one pending-write sync (constructed before it, under the GpuMutex): gathers what the sync
// waits for and, in the destructor, the time it took and whether the bytes changed.
class HookSyncScope {
public:
    HookSyncScope(const Recorder& recorder, std::uint64_t address, std::size_t bytes) : enabled(HookSyncProfiled()), address(address), bytes(bytes) {
        if (!enabled) return;
        info = recorder.DescribePendingWrite(address, bytes);
        const auto packet = GuestMemory::CurrentPacket();
        key = HookSyncKey{packet.queue, packet.opcode, GuestMemory::CurrentReadSite(), {}};
        // Frame 0 is the hook's caller (the GuestMemory entry, or the driver code when the entry
        // was inlined); the driver code that made the access is one of the next frames.
        GuestMemory::CaptureCallerOffsets(key.frames, 1);
        // The access's bytes are copied raw: a Read here would re-enter the hook. The same access
        // check the accessing code makes guards the copy; a range that fails it is left unchecked.
        // A store's bytes are not compared (counted as a store instead).
        if (key.site != GuestMemory::ReadSite::Store && bytes <= CompareLimit && GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) {
            before.resize(bytes);
            std::memcpy(before.data(), reinterpret_cast<const void*>(address), bytes);
        }
        start = std::chrono::steady_clock::now();
    }
    ~HookSyncScope() {
        if (!enabled) return;
        const auto now = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration<double, std::milli>(now - start).count();
        auto& stats = HookSyncs();
        auto& totals = stats.byKey[key];
        ++totals.count;
        totals.waitedMs += ms;
        totals.accessBytes += bytes;
        ++stats.count;
        stats.waitedMs += ms;
        stats.accessBuckets[SizeBucket(bytes)] += 1;
        if (info.has_value()) {
            const auto rangeBytes = info->rangeEnd - info->rangeBegin;
            totals.rangeBytes += rangeBytes;
            stats.rangeBuckets[SizeBucket(rangeBytes)] += 1;
            if (info->open) ++stats.openTargets;
            if (info->signaled) ++stats.signaledTargets;
            stats.batchesFinished += info->batchesToFinish;
        }
        // One outcome per sync, on both the key's and the interval's counters.
        const auto outcome = [&]() -> std::uint64_t HookSyncOutcomes::* {
            if (key.site == GuestMemory::ReadSite::Store) return &HookSyncOutcomes::stores;
            if (before.empty() || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) return &HookSyncOutcomes::unchecked;
            if (std::memcmp(before.data(), reinterpret_cast<const void*>(address), bytes) != 0) return &HookSyncOutcomes::changed;
            // A target already signaled (or no target found: the batch is finishing) had run
            // before the copy, so its unchanged bytes are not evidence of a false overlap.
            if (!info.has_value() || info->signaled) return &HookSyncOutcomes::unchangedSignaled;
            return info->open ? &HookSyncOutcomes::unchangedOpen : &HookSyncOutcomes::unchangedPending;
        }();
        ++(totals.outcomes.*outcome);
        ++(stats.outcomes.*outcome);
        if (now - stats.lastReport > std::chrono::seconds(10)) ReportHookSyncs(stats);
    }
    HookSyncScope(const HookSyncScope&) = delete;
    HookSyncScope& operator=(const HookSyncScope&) = delete;

private:
    static constexpr std::size_t CompareLimit = 65536;
    bool enabled;
    std::uint64_t address;
    std::size_t bytes;
    std::optional<Recorder::PendingWriteInfo> info;
    HookSyncKey key{};
    std::vector<std::byte> before;
    std::chrono::steady_clock::time_point start;
};

// Attribution of the hook's recorded-store syncs (source 2: a pending image's write-back or a unit
// publish was recorded for the access and SyncThrough waits for it), the [hooksync] recorded-store
// line every 10 s under the same switch as HookSyncScope: per (queue, packet, read site), the syncs,
// their wait, the batches in flight at the sync, the C4 classification of the access (every 64 KiB
// block CPU-written since the overlapping images' generations: the store would not change a byte
// the CPU reads), whether the bytes changed over the wait, whether the images overlapped had a
// consumer proof within StorageTexture::DeadImagePresents presents, and the accesses the C4 skip
// took instead of a flush (with the images that left the registry there as an empty write-back
// would have left them). The skips are counted without the GpuMutex (their own mutex); the syncs
// under it.
struct RecordedStoreTotals {
    std::uint64_t count = 0;
    double waitedMs = 0;
    std::uint64_t batchesFinished = 0;
    std::uint64_t cpuWrittenYes = 0, cpuWrittenNo = 0, unclassified = 0, publishOnly = 0;
    std::uint64_t changed = 0, unchanged = 0, unchecked = 0;
    std::uint64_t imagesDead = 0, imagesLive = 0;
    std::uint64_t skipped = 0, flushed = 0, evicted = 0;
};

struct RecordedStoreStats {
    std::mutex mutex;
    std::map<HookSyncKey, RecordedStoreTotals> byKey;
    RecordedStoreTotals totals;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

RecordedStoreStats& RecordedStoreSyncs() {
    static RecordedStoreStats stats;
    return stats;
}

void ReportRecordedStoreSyncs(RecordedStoreStats& stats) {
    std::vector<std::pair<const HookSyncKey*, const RecordedStoreTotals*>> hot;
    for (const auto& [key, totals] : stats.byKey) hot.emplace_back(&key, &totals);
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return std::tie(a.second->waitedMs, a.second->skipped) > std::tie(b.second->waitedMs, b.second->skipped); });
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto& t = stats.totals;
    const auto skips = StorageTexture::TakeHookSkipCounts();
    char text[640];
    std::snprintf(text, sizeof(text), "[hooksync] recorded-store syncs (10 s): %llu waited %.0f ms, %llu batches in flight at them; skipped (cpu-written blocks) %llu / flushed %llu / evicted stale %llu; flushed later after a skip %llu (by the same site %llu); C4 every block CPU-written: yes %llu / no %llu / unclassified %llu (publish only %llu); bytes after the wait: changed %llu, unchanged %llu, unchecked %llu; images overlapped: dead %llu / live %llu (no consumer proof for %llu presents); by queue packet site (count/ms, skipped, cpu-written yes/no, changed/unchanged, dead/live):", count(t.count), t.waitedMs, count(t.batchesFinished), count(t.skipped), count(t.flushed), count(t.evicted), count(skips.flushedAfterSkip), count(skips.flushedAfterSkipSameSite), count(t.cpuWrittenYes), count(t.cpuWrittenNo), count(t.unclassified), count(t.publishOnly), count(t.changed), count(t.unchanged), count(t.unchecked), count(t.imagesDead), count(t.imagesLive), count(StorageTexture::DeadImagePresents));
    std::string report = text;
    for (std::size_t i = 0; i < hot.size() && i < 10; ++i) {
        const auto& key = *hot[i].first;
        const auto& k = *hot[i].second;
        std::snprintf(text, sizeof(text), " [0x%x %s %s +0x%llx/+0x%llx/+0x%llx: %llu/%.0fms k%llu y%llu/n%llu c%llu/u%llu d%llu/l%llu]", key.queue, PacketName(key.opcode).c_str(), GuestMemory::ReadSiteName(key.site), key.frames[0], key.frames[1], key.frames[2], count(k.count), k.waitedMs, count(k.skipped), count(k.cpuWrittenYes), count(k.cpuWrittenNo), count(k.changed), count(k.unchanged), count(k.imagesDead), count(k.imagesLive));
        report += text;
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", report.c_str());
    stats.byKey.clear();
    stats.totals = {};
}

// The key of the access the hook handles now: always inlined into FlushForAccess so that frame 0
// is the hook's caller for the skip counters and the sync scope alike, whatever the inliner does
// with the scope's constructor.
[[gnu::always_inline]] inline HookSyncKey RecordedStoreKey() {
    const auto packet = GuestMemory::CurrentPacket();
    HookSyncKey key{packet.queue, packet.opcode, GuestMemory::CurrentReadSite(), {}};
    GuestMemory::CaptureCallerOffsets(key.frames, 1);
    return key;
}

void CountRecordedStoreSkip(const HookSyncKey& key, std::size_t evicted) {
    auto& stats = RecordedStoreSyncs();
    std::lock_guard lock(stats.mutex);
    for (auto* totals : {&stats.byKey[key], &stats.totals}) {
        ++totals->skipped;
        totals->evicted += evicted;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.lastReport > std::chrono::seconds(10)) {
        stats.lastReport = now;
        ReportRecordedStoreSyncs(stats);
    }
}

class RecordedStoreSyncScope {
public:
    RecordedStoreSyncScope(const Recorder& recorder, std::uint64_t address, std::size_t bytes, const StorageTexture::AccessClassification& classification, bool stored, const HookSyncKey& key) : enabled(HookSyncProfiled()), address(address), bytes(bytes), classification(classification), stored(stored), key(key) {
        if (!enabled) return;
        batchesInFlight = recorder.InFlightBatches();
        if (key.site != GuestMemory::ReadSite::Store && bytes <= CompareLimit && GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) {
            before.resize(bytes);
            std::memcpy(before.data(), reinterpret_cast<const void*>(address), bytes);
        }
        start = std::chrono::steady_clock::now();
    }
    ~RecordedStoreSyncScope() {
        if (!enabled) return;
        const auto now = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration<double, std::milli>(now - start).count();
        auto& stats = RecordedStoreSyncs();
        std::lock_guard lock(stats.mutex);
        const auto add = [&](RecordedStoreTotals& totals) {
            ++totals.count;
            totals.waitedMs += ms;
            totals.batchesFinished += batchesInFlight;
            if (stored) ++totals.flushed;
            if (classification.images == 0) ++totals.publishOnly;
            else if (!classification.checked) ++totals.unclassified;
            else ++(classification.allCpuWritten ? totals.cpuWrittenYes : totals.cpuWrittenNo);
            totals.imagesDead += classification.dead;
            totals.imagesLive += classification.live;
            if (before.empty() || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) ++totals.unchecked;
            else if (std::memcmp(before.data(), reinterpret_cast<const void*>(address), bytes) != 0) ++totals.changed;
            else ++totals.unchanged;
        };
        add(stats.byKey[key]);
        add(stats.totals);
        if (now - stats.lastReport > std::chrono::seconds(10)) {
            stats.lastReport = now;
            ReportRecordedStoreSyncs(stats);
        }
    }
    RecordedStoreSyncScope(const RecordedStoreSyncScope&) = delete;
    RecordedStoreSyncScope& operator=(const RecordedStoreSyncScope&) = delete;

private:
    static constexpr std::size_t CompareLimit = 65536;
    bool enabled;
    std::uint64_t address;
    std::size_t bytes;
    StorageTexture::AccessClassification classification;
    bool stored;
    HookSyncKey key;
    std::size_t batchesInFlight = 0;
    std::vector<std::byte> before;
    std::chrono::steady_clock::time_point start;
};

// GuestMemory flush hook: a CPU access to memory that recorded GPU work will write waits for that
// work first; then storage image results pending for the range are stored. The recorder is only
// dereferenced under the GpuMutex (it is destroyed with its device under that lock).
// `gpuReader`: the access is GPU work this thread records into the open batch next (the Metal
// present's display decode, D66). Every recorded GPU write over the range precedes it in queue
// order, so the hook records what must land first (queued labels, storage image stores) and skips
// both CPU waits; FlushForGpuRead only asks for that while no batch holds a CPU write-back.
void FlushForAccessImpl(std::uint64_t address, std::size_t bytes, bool gpuReader) {
    hookCalls.fetch_add(1, std::memory_order_relaxed);
    // A label this worker queued but has not recorded writes the range: in queue order the label
    // precedes this access, so the group is recorded first (it then notes the range as a pending
    // write of the open batch, and the check below syncs on it like any recorded store). Not from
    // inside a completion action (a write-back store made while a batch is reaped): the record
    // would reap nested in the reap in progress, landing later batches' write-backs before the
    // rest of this one's, and this queue's queued label follows every batch of its own in flight
    // anyway, so the write-back landing first is the order the hardware gives.
    if (!QueuedLabelRanges().empty() && QueuedLabelOverlaps(address, bytes)) {
        if (completionDepth != 0 && HookCompletionGuard()) {
            queuedLabelHookInCompletion.fetch_add(1, std::memory_order_relaxed);
        } else if (auto* record = queuedLabelRecorder.load(std::memory_order_acquire); record != nullptr) {
            queuedLabelHookRecords.fetch_add(1, std::memory_order_relaxed);
            record();
        }
    }
    if (!gpuReader && (!HookSnapshotEnabled() || SnapshotOverlaps(address, bytes))) {
        hookLocks.fetch_add(1, std::memory_order_relaxed);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
        std::lock_guard gpu(GuestMemory::GpuMutex());
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->PendingWriteOverlaps(address, bytes)) {
            const HookSyncScope attribution(*recorder, address, bytes);
            Recorder::CountSync(1);
            // The wait runs without the mutex when this acquisition is the outermost one (a stage-A
            // read, a game thread); the recorder is not dereferenced after the call.
            recorder->SyncThrough(address, bytes, true);
        }
    }
    // A reader takes every shadowed unit of the range; a store only the units it covers partly
    // (its own stamp makes the others stale). Inside a completion action a store has no publish
    // recorded: the batch is not waited for, and the copy would land over the store later.
    const auto site = GuestMemory::CurrentReadSite();
    const auto scope = site != GuestMemory::ReadSite::Store ? PublishScope::Whole : completionDepth != 0 ? PublishScope::None : PublishScope::PartialUnits;
    bool published = false;
    // The classification is made before the store (the store changes what it measures); profile only.
    StorageTexture::AccessClassification classification{};
    if (HookSyncProfiled()) StorageTexture::ClassifyAccess(address, bytes, classification);
    // C4: an access whose every 64 KiB block the CPU wrote since the overlapping images' generations
    // reads the same bytes with or without their store (the store keeps those blocks), so the
    // images stay pending (short of the layers the store would have left empty-handed anyway) and
    // only the unit shadows over the range are published; with no image over the range that
    // publish is all FlushPending would do (one registry scan, not two).
    bool stored = false;
    std::size_t images = 0, evicted = 0;
    const bool kept = !HookFlushCpuBlocks() && StorageTexture::AccessKeptByCpu(address, bytes, &images, &evicted);
    if (kept || (images == 0 && !HookFlushCpuBlocks())) {
        // 5937ef03ef: an image another thread is storing over the range lands before the host read.
        stored = StorageTexture::WaitForFlushing(address, bytes);
        published = StorageTexture::PublishShadowsOnly(address, bytes, scope);
        if (kept && HookSyncProfiled()) CountRecordedStoreSkip(RecordedStoreKey(), evicted);
    } else {
        stored = StorageTexture::FlushPending(address, bytes, nullptr, "memory access", scope, &published);
    }
    if ((stored || published) && !gpuReader) {
        // Stores into imported memory were only recorded; the CPU is about to read them. The wait
        // targets the batch holding them (the open one: SyncThrough submits it) and runs with the
        // mutex released when this acquisition is the outermost; a nested hook waits locked through
        // that batch ([lock] 'locked GPU waits'). Inside a completion action the store must land
        // before the completing batch's CPU write-back (an older result under a newer one), which
        // only the full Sync() gives there.
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
        std::lock_guard gpu(GuestMemory::GpuMutex());
        if (auto* recorder = Recorder::Active(); recorder != nullptr) {
            const RecordedStoreSyncScope attribution(*recorder, address, bytes, classification, stored, HookSyncProfiled() ? RecordedStoreKey() : HookSyncKey{});
            Recorder::CountSync(2);
            if (HookFullSync() || completionDepth != 0) recorder->Sync();
            else recorder->SyncThrough(address, bytes, true);
        }
    }
}

void FlushForAccess(std::uint64_t address, std::size_t bytes) {
    FlushForAccessImpl(address, bytes, false);
}

std::atomic<std::uint64_t> gpuReadsOrdered{0};
std::atomic<std::uint64_t> gpuReadsWaited{0};

}

Recorder::Recorder(const Context& context, bool) : context(context), id(nextRecorderId.fetch_add(1)) {
    // macOS: the Context carries the device's MTLDevice and MTL4CommandQueue (Metal::DeviceOf,
    // Metal::QueueOf); the timeline is this recorder's MTLSharedEvent, so the unlocked waits are
    // always available (the Vulkan `timelineSemaphores` switch has no Metal counterpart).
    auto objects = std::unique_ptr<Metal::RecorderObjects, Metal::ObjectsDeleter>(new Metal::RecorderObjects);
    objects->device = Metal::DeviceOf(context);
    objects->queue = Metal::QueueOf(context);
    if (objects->device == nil || objects->queue == nil) Metal::Fail("the recorder needs a Context carrying a Metal device and command queue (a Vulkan context reached the Metal recorder)");
    objects->timeline = [objects->device newSharedEvent];
    if (objects->timeline == nil) Metal::Fail("cannot create the recorder's timeline event");
    objects->timeline.label = @"AGC recorder timeline";
    MTLResidencySetDescriptor* residency = [MTLResidencySetDescriptor new];
    residency.label = @"AGC recorder";
    residency.initialCapacity = 8;
    NSError* error = nil;
    objects->residency = [objects->device newResidencySetWithDescriptor:residency error:&error];
    if (objects->residency == nil) Metal::Fail("the recorder's residency set: " + Metal::Describe(error));
    ::AgcDriver::Metal::Residency::Attach(objects->queue, objects->residency);
    objects->arena = std::make_shared<Metal::UploadArena>(objects->device, objects->queue);
    const auto frequency = [objects->device queryTimestampFrequency];
    objects->nanosecondsPerTick = frequency != 0 ? 1e9 / static_cast<double>(frequency) : 1.0;
    metal = std::move(objects);
    {
        std::lock_guard lock(liveRecordersMutex);
        liveRecorders.push_back(id);
    }
    // Idempotent: the same hook for every recorder (the deferred lists are per thread, not per recorder).
    GuestMemory::SetGpuUnlockHook(&ReleaseDeferredKeeps);
}

Recorder::~Recorder() {
    try {
        Sync();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[gpu] recorder teardown: %s\n", error.what());
    }
    holdKept.clear();
    if (activeRecorder == this) {
        activeRecorder = nullptr;
        pendingWrites.store(nullptr, std::memory_order_release);
        publishGeneration.fetch_add(1, std::memory_order_release);
        pendingLabelSince.store(NoPendingLabel, std::memory_order_release);
        completionLabels.store(0, std::memory_order_release);
        writeBackCompletions.store(0, std::memory_order_release);
        workSinceSubmit.store(0, std::memory_order_release);
    }
    {
        // A thread that released the mutex around a wait on this recorder re-checks the id when it
        // retakes the mutex and touches nothing then. That check is only sound while the device
        // (and so this destructor) is torn down under the GpuMutex, as the driver's drain and
        // failure paths do; a worker dropping the last device reference after releasing its lock
        // would run this without the mutex, concurrently with a relocked waiter.
        std::lock_guard lock(liveRecordersMutex);
        liveRecorders.erase(std::remove(liveRecorders.begin(), liveRecorders.end(), id), liveRecorders.end());
    }
    // Sync() above completed every batch, so every timeline wait in progress on this recorder
    // returns now; the semaphore is destroyed only once they all left it.
    while (WaitersOf(id).load(std::memory_order_acquire) != 0) std::this_thread::yield();
    // The kept objects of every batch this thread finished (Sync() above included) belong to the
    // device that is going away: destroyed now, here (no hand-off: nothing should stay in flight
    // behind this thread); the release thread is then stopped and joined (it drains its queue
    // first, holding no mutex this thread holds: the destructors take only their caches' own
    // mutexes, which order after the GpuMutex), and the releases still in progress inline on
    // other threads (an unlock hook that found the queue stopping or full) are waited for before
    // the caller destroys the device. The count is global: a newer recorder's batches (device
    // replacement) are waited for too, which only prolongs the spin.
    if (!DeferredBatches().empty()) DestroyDeferred(TakeDeferred(), false);
    JoinReleaseThread();
    while (deferredPending.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    {
        // Unlocked lookups stop before `labels` is destroyed (after this body).
        std::lock_guard tableLock(labelTableMutex);
        if (labelTableOwner == this) labelTableOwner = nullptr;
    }
    if (sampleCounterState > 0) {
        try {
            samplesPassed.store(SamplesTotal(), std::memory_order_release);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[gpu] recorder teardown: occlusion samples: %s\n", error.what());
        }
        if (samplePools != nullptr) std::static_pointer_cast<SamplePoolCache>(samplePools)->close();
        if (segmentPools != nullptr) std::static_pointer_cast<SamplePoolCache>(segmentPools)->close();
        sampleCounter.reset();
    }
    // Sync() above finished every batch: the spare command buffers, allocators and streams, the
    // timeline event and the residency set go with `metal` (the helper pipelines are per device,
    // Metal::HelperPipeline).
    pendingSamples.clear();
}

std::optional<std::chrono::steady_clock::time_point> Recorder::PendingLabelSince() {
    const auto since = pendingLabelSince.load(std::memory_order_acquire);
    if (since == NoPendingLabel) return std::nullopt;
    return std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(since));
}

std::uint64_t Recorder::WriteGeneration() {
    return writeGeneration.load(std::memory_order_acquire);
}

std::uint64_t Recorder::PublishGeneration() {
    return publishGeneration.load(std::memory_order_acquire);
}

std::uint64_t Recorder::PendingCompletionLabels() {
    return completionLabels.load(std::memory_order_acquire);
}

std::uint64_t Recorder::PendingWriteBackCompletions() {
    return writeBackCompletions.load(std::memory_order_acquire);
}

std::uint64_t Recorder::RecordedWorkSinceSubmit() {
    return workSinceSubmit.load(std::memory_order_relaxed);
}

void Recorder::CountRecordedWork() {
    workSinceSubmit.fetch_add(1, std::memory_order_relaxed);
}

Recorder* Recorder::Active() {
    return activeRecorder;
}

void Recorder::Activate() {
    activeRecorder = this;
    // macOS: what the one-shot CommandBatch (Resources.mm) calls on the active recorder.
    Metal::SetActiveRecorderCalls({+[] {
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) recorder->Submit();
    }, +[] {
        if (auto* recorder = Recorder::Active()) recorder->Reap();
    }});
    {
        std::lock_guard tableLock(labelTableMutex);
        labelTableOwner = this;
    }
    GuestMemory::SetFlushHook(&FlushForAccess);
}

std::optional<Recorder::LabelHit> Recorder::LookupLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, LabelRefusal* refusal) {
    std::lock_guard tableLock(labelTableMutex);
    if (labelTableOwner == nullptr) return std::nullopt;
    return labelTableOwner->lookupLabel(address, bytes, afterStamp, refusal);
}

bool Recorder::WideLabelIn(std::uint64_t address, std::size_t bytes) {
    std::lock_guard tableLock(labelTableMutex);
    return labelTableOwner != nullptr && labelTableOwner->wideLabelInLocked(address, bytes);
}

bool Recorder::wideLabelInLocked(std::uint64_t address, std::size_t bytes) const {
    if (wideLabels.empty() || bytes == 0) return false;
    const auto end = address + bytes;
    for (auto it = wideLabels.lower_bound(address >= wideLabelBytes ? address - wideLabelBytes + 1 : 0); it != wideLabels.end() && it->first < end; ++it) {
        if (it->second > address) return true;
    }
    return false;
}

std::optional<std::uint64_t> Recorder::LookupLabelValue(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp) {
    const auto hit = LookupLabel(address, bytes, afterStamp);
    if (!hit.has_value()) return std::nullopt;
    return hit->value;
}

bool Recorder::LateTrust() {
    return LateTrustEnabled();
}

Recorder::LateStatistics Recorder::LateCounts() {
    return lateCounts;
}

void Recorder::CloseLabelGroup(std::uint64_t trackerGeneration) {
    if (LabelGroupDwords().empty()) return;
    auto group = std::move(LabelGroupDwords());
    LabelGroupDwords().clear();
    if (trackerGeneration == 0) return;
    std::lock_guard tableLock(labelTableMutex);
    if (labelTableOwner == nullptr) return;
    auto& recorded = labelTableOwner->labels;
    for (const auto& [dword, stamp] : group) {
        // Only an entry still ours (stamps are unique): a later note of the dword has its own group.
        // And only while its batch is unsubmitted: a submitted one (a drain or APS5_LABEL_SUBMIT_NOW
        // inside the group) may have landed the value already, and a CPU store between that and
        // this close is not newer than the generation.
        const auto found = recorded.find(dword);
        if (found != recorded.end() && found->second.stamp == stamp && found->second.batch != nullptr && !found->second.batch->submitted) found->second.generation = trackerGeneration;
    }
    // A poller that refused the group as unclosed re-consults the table on the next generation.
    writeGeneration.fetch_add(1, std::memory_order_release);
    ::AgcDriver::Progress::Notify();
}

void Recorder::NoteQueuedLabel(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue) {
    if (bytes.size() < 4 || address % 4 != 0) return;
    // The range before the table: the hook covers the group whether or not a table exists to
    // enter it in (no owner: the callback records it outright, by a CPU store when need be).
    QueuedLabelRanges().emplace_back(address, address + bytes.size());
    queuedLabelsNoted.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard tableLock(labelTableMutex);
    if (labelTableOwner == nullptr) return;
    auto& recorded = labelTableOwner->labels;
    auto& table = SeparateQueuedLabels() ? labelTableOwner->queuedLabels : recorded;
    for (std::size_t offset = 0; offset + 4 <= bytes.size(); offset += 4) {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, 4);
        if (const auto found = recorded.find(address + offset); found != recorded.end() && found->second.batch != nullptr) queuedLabelsOverRecorded.fetch_add(1, std::memory_order_relaxed);
        table.insert_or_assign(address + offset, LabelEntry{value, queue, stamp, nullptr});
    }
    labelTableOwner->recordedLabels.store(recorded.size(), std::memory_order_relaxed);
}

void Recorder::ForgetQueuedLabels() {
    if (QueuedLabelRanges().empty()) return;
    auto ranges = std::move(QueuedLabelRanges());
    QueuedLabelRanges().clear();
    std::lock_guard tableLock(labelTableMutex);
    if (labelTableOwner == nullptr) return;
    auto& recorded = labelTableOwner->labels;
    auto& queued = labelTableOwner->queuedLabels;
    const auto tag = GuestMemory::GpuLockThreadTag();
    for (const auto& [begin, end] : ranges) {
        for (auto dword = begin; dword + 4 <= end; dword += 4) {
            if (const auto own = queued.find(dword); own != queued.end() && own->second.queue == tag) queued.erase(own);
            const auto found = recorded.find(dword);
            if (found != recorded.end() && found->second.batch == nullptr) recorded.erase(found);
        }
    }
    labelTableOwner->recordedLabels.store(recorded.size(), std::memory_order_relaxed);
}

void Recorder::SetQueuedLabelRecorder(void (*recorder)()) {
    queuedLabelRecorder.store(recorder, std::memory_order_release);
}

Recorder::StoreStatistics Recorder::StoreCounts() {
    return StoreStatistics{storeCount.load(std::memory_order_relaxed), storeRuns.load(std::memory_order_relaxed), storesJoined.load(std::memory_order_relaxed), storesReplaced.load(std::memory_order_relaxed), storeWawBarriers.load(std::memory_order_relaxed), storeJoinsRefused.load(std::memory_order_relaxed), queuedLabelsNoted.load(std::memory_order_relaxed), queuedLabelsOverRecorded.load(std::memory_order_relaxed), queuedLabelHits.load(std::memory_order_relaxed), queuedLabelHookRecords.load(std::memory_order_relaxed), queuedLabelHookInCompletion.load(std::memory_order_relaxed), keyStoreCount.load(std::memory_order_relaxed), keyStoreRuns.load(std::memory_order_relaxed), keyStoreRunsForWriter.load(std::memory_order_relaxed), keyStoresJoined.load(std::memory_order_relaxed), storeRunsAtSubmit.load(std::memory_order_relaxed), storeRunsForced.load(std::memory_order_relaxed)};
}

std::uint64_t Recorder::ThreadHookWaits() {
    return hookRealWaits;
}

bool Recorder::SnapshotWriteOverlaps(std::uint64_t address, std::size_t bytes) {
    return AgcDriver::Graphics::SnapshotOverlaps(address, bytes);
}

std::shared_ptr<const Recorder::WriteRanges> Recorder::PendingWriteSnapshot() {
    return pendingWrites.load(std::memory_order_acquire);
}

bool Recorder::SnapshotOverlaps(const WriteRanges* snapshot, std::uint64_t address, std::size_t bytes) {
    return AgcDriver::Graphics::SnapshotOverlaps(snapshot, address, bytes);
}

void Recorder::CountSync(int source, const void* site) {
    if (source < 0 || source >= 5) return;
    ++syncCounts[source];
    announcedSource = source;
    announcedSite = site;
}

void Recorder::AnnounceSyncSite(const void* site) {
    announcedSite = site;
}

double Recorder::ThreadWaitedMs() {
    return threadWaitedMs;
}

std::uint64_t Recorder::ReapsWithWork() {
    return holdCounters.reapsWithWork;
}

VkCommandBuffer Recorder::Commands(VkAccessFlags* coveredAccess) {
    ensureOpen();
    // Work recorded after a draw's render pass or an inline store run must see their writes: the
    // pass's end and the run's trailing barrier go in first (a per-batch run waits for Submit, or
    // for a caller whose ranges overlap a queued store: FlushStores).
    if (open->renderPass.open) endOpenRenderPass();
    if (open->run.open && !LabelRunsPerBatch()) closeStoreRun();
    if (coveredAccess != nullptr) *coveredAccess = open->coveredAccess;
    open->coveredAccess = 0;
    return CommandsOf(open->metal);
}

Metal::CommandStream& Recorder::Stream(VkAccessFlags* coveredAccess) {
    return Metal::Stream(Commands(coveredAccess));
}

void Recorder::MarkCovered(VkAccessFlags access) {
    if (open != nullptr) open->coveredAccess = access;
}

Recorder::DrawPassStart Recorder::StartDrawPass(std::uint64_t key, bool forced, bool fenced, const PassAccess& access) {
    ensureOpen();
    auto& pass = open->renderPass;
    if (!pass.open || forced || (countingSamples && segmentMode() && !segmentSlotReady(*open))) {
        const auto commands = Commands();
        open->passHazards.Clear();
        open->passHazards.BeginPass();
        return {commands, false, true};
    }
    const auto hazard = open->passHazards.Check(access, pass.key == key);
    if (hazard == PassHazard::None && pass.key == key) return {CommandsOf(open->metal), true, false};
    const bool merge = MergeBarriers() && !fenced && !pass.fenced && !pass.afterPass;
    const bool elide = merge && hazard == PassHazard::None;
    endOpenRenderPass(!merge);
    const auto commands = Commands();
    if (!elide) open->passHazards.Clear();
    open->passHazards.BeginPass();
    return {commands, false, !elide};
}

void Recorder::LeaveRenderPassOpen(std::uint64_t key, std::uint32_t timing, bool fenced, const PassAccess& access, std::function<void(VkCommandBuffer)> afterPass) {
    Require(open != nullptr, "no batch is open for the render pass");
    auto& pass = open->renderPass;
    if (!pass.open) {
        pass.timing = timing;
        pass.afterPass = std::move(afterPass);
    }
    pass.open = true;
    pass.key = key;
    pass.fenced = pass.fenced || fenced;
    open->passHazards.Add(access);
}

// 12d720f743: a guest flush seals the open pass's write-only stores (PacketExecution -> device -> here).
void Recorder::SealPassWriteOnly() {
    if (open != nullptr) open->passHazards.SealWriteOnly();
}

void Recorder::endOpenRenderPass(bool barrier) {
    auto& pass = open->renderPass;
    if (open->segmentActive) endSegment(*open);
    // macOS: the pass is the stream's render encoder (the draw that began it opened it).
    open->metal->stream->EndEncoder();
    // The pass's attachment and shader writes are visible to everything recorded after it (the
    // host sees them at the batch's fence).
    if (barrier) {
        recordBarrier(CommandsOf(open->metal), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        CountBarriers(CommandClass::Draw);
    }
    if (pass.afterPass) pass.afterPass(CommandsOf(open->metal));
    EndGpuTiming(pass.timing);
    open->coveredAccess = barrier ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT : 0;
    open->hostReadOwed = true;
    pass = {};
}

void Recorder::QueueKeyStore(VkBuffer buffer, VkDeviceSize first, VkDeviceSize last, std::shared_ptr<void> seed, std::uint64_t begin, std::uint64_t end) {
    ensureOpen();
    // A queued label store over the keys lands first (program order: the key store is later).
    FlushStoresOverlapping(begin, static_cast<std::size_t>(end - begin));
    keyStoreCount.fetch_add(1, std::memory_order_relaxed);
    if (seed != nullptr) open->kept.push_back(seed);
    // The same range queued twice lands once: nothing wrote over it in between, or the writer
    // would have recorded the first store first.
    for (const auto& store : open->keyStores) {
        if (store.buffer == buffer && store.first == first && store.last == last) {
            keyStoresJoined.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    open->keyStores.push_back({buffer, first, last, std::move(seed), begin, end});
    if (KeyStoresEach()) recordKeyStores(false);
}

bool Recorder::QueuedKeyStoreOverlaps(std::uint64_t address, std::size_t bytes) const {
    if (open == nullptr || bytes == 0) return false;
    const auto end = address + bytes;
    return std::any_of(open->keyStores.begin(), open->keyStores.end(), [&](const Batch::KeyStore& store) { return address < store.end && store.begin < end; });
}

bool Recorder::AnyQueuedKeyStore(const std::function<bool(std::uint64_t, std::uint64_t)>& overlaps) const {
    if (open == nullptr) return false;
    return std::any_of(open->keyStores.begin(), open->keyStores.end(), [&](const Batch::KeyStore& store) { return overlaps(store.begin, store.end); });
}

void Recorder::FlushKeyStores() {
    if (open == nullptr || open->keyStores.empty()) return;
    recordKeyStores(true);
}

void Recorder::recordKeyStores(bool forWriter) {
    auto stores = std::move(open->keyStores);
    open->keyStores.clear();
    if (stores.empty()) return;
    const auto commands = Commands();
    const auto timing = beginTiming(ClassKey(CommandClass::DccKeyStore));
    // Ordered behind every earlier recorded read or write of the ranges (the title's DCC clear or
    // decompress kernel storing keys through a V#), and visible to the work after and to the host.
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    auto& stream = Metal::Stream(commands);
    std::uint64_t bytes = 0;
    for (const auto& store : stores) {
        const VkDeviceSize fillBegin = std::min((store.first + 3) & ~VkDeviceSize{3}, store.last);
        const VkDeviceSize fillEnd = std::max(store.last & ~VkDeviceSize{3}, fillBegin);
        MetalBuffer target = Metal::BufferOf(store.buffer);
        if (fillEnd > fillBegin) stream.FillBuffer(target, fillBegin, fillEnd - fillBegin, 0xffffffffu);
        if (store.seed != nullptr) {
            // The seed's first bytes supply the unaligned head and tail (both from its offset 0).
            MetalBuffer seed = Metal::BufferOf(static_cast<Buffer*>(store.seed.get())->Handle());
            if (fillBegin > store.first) stream.CopyBuffer(seed, 0, target, store.first, fillBegin - store.first);
            if (store.last > fillEnd) stream.CopyBuffer(seed, 0, target, fillEnd, store.last - fillEnd);
        }
        bytes += store.last - store.first;
    }
    recordBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
    CountBarriers(CommandClass::DccKeyStore, 2);
    EndGpuTiming(timing, bytes);
    open->coveredAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    keyStoreRuns.fetch_add(1, std::memory_order_relaxed);
    if (forWriter) keyStoreRunsForWriter.fetch_add(1, std::memory_order_relaxed);
}

void Recorder::ensureOpen() {
    GuestMemory::AssertGpuLockHeld("Recorder::Commands");
    if (open == nullptr) {
        auto batch = std::make_unique<Batch>();
        try {
            if (!metal->spare.empty()) {
                // A completed batch's objects: its allocator was reset and its stream recycled.
                batch->metal = std::move(metal->spare.back());
                metal->spare.pop_back();
            } else {
                batch->metal.reset(new Metal::BatchObjects);
                batch->metal->allocator = [metal->device newCommandAllocator];
                batch->metal->commandBuffer = [metal->device newCommandBuffer];
                if (batch->metal->allocator == nil || batch->metal->commandBuffer == nil) Metal::Fail("cannot create a command buffer and allocator");
                batch->metal->stream = std::make_unique<Metal::CommandStream>(metal->arena);
            }
            batch->metal->feedback = std::make_shared<BatchFeedback>();
            batch->metal->stream->Begin(batch->metal->commandBuffer, batch->metal->allocator);
        } catch (...) {
            release(*batch);
            throw;
        }
        batch->queue = GuestMemory::GpuLockThreadTag();
        batch->openedAt = std::chrono::steady_clock::now();
        if (countingSamples && !segmentMode()) beginSamples(*batch);
        dropStaleHoldKept();
        batch->kept.insert(batch->kept.end(), holdKept.begin(), holdKept.end());
        open = std::move(batch);
        if (PerformanceTrace::Recording()) {
            static std::atomic<std::uint64_t> traceIds{0};
            open->traceId = traceIds.fetch_add(1, std::memory_order_relaxed) + 1;
        }
        TraceBatchTag = open->traceId;
        // The whole-batch range: its start stamp waits for the previous batches like any
        // bottom-of-pipe stamp, so it marks when this batch's execution began.
        if (BatchStampsEnabled()) open->batchTiming = beginTiming(BatchTimingKey);
    }
}

void Recorder::RecordStore(VkBuffer buffer, VkDeviceSize offset, std::span<const std::byte> bytes, std::uint64_t address) {
    if (bytes.empty()) return;
    ensureOpen();
    auto& run = open->run;
    const VkDeviceSize end = offset + bytes.size();
    storeCount.fetch_add(1, std::memory_order_relaxed);
    if (LabelRunsPerBatch()) {
        // Queued for the batch's run: a store inside the last queued one replaces its bytes, a
        // contiguous one joins it (a joined range overlapping an earlier queued store takes the
        // WAW barrier when the run is recorded, so no join is refused).
        if (run.queued.empty()) storeRuns.fetch_add(1, std::memory_order_relaxed);
        else if (auto& last = run.queued.back(); last.buffer == buffer) {
            const VkDeviceSize lastEnd = last.offset + last.bytes.size();
            if (offset >= last.offset && end <= lastEnd) {
                std::memcpy(last.bytes.data() + (offset - last.offset), bytes.data(), bytes.size());
                storesReplaced.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (offset == lastEnd && last.bytes.size() + bytes.size() <= 65536) {
                last.bytes.insert(last.bytes.end(), bytes.begin(), bytes.end());
                storesJoined.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        run.queued.push_back({buffer, offset, std::vector<std::byte>(bytes.begin(), bytes.end()), address});
        return;
    }
    // A label says the work before it is done: the queued key stores (results of that work) land
    // ahead of every store of the run.
    if (!open->keyStores.empty()) recordKeyStores(true);
    if (open->renderPass.open) endOpenRenderPass();
    open->coveredAccess = 0;
    // Two transfers to the same bytes have no order of their own within the run.
    const auto overlapsRecorded = [&] { return std::any_of(run.recorded.begin(), run.recorded.end(), [&](const auto& store) { return std::get<0>(store) == buffer && offset < std::get<2>(store) && std::get<1>(store) < end; }); };
    if (!run.open) {
        run.timing = beginTiming(ClassKey(CommandClass::LabelRun));
        run.storedBytes = 0;
        // The run's leading barrier: every earlier write of the range (a shader's, a transfer's, a
        // color attachment's) lands before the stores of the run.
        recordBarrier(CommandsOf(open->metal), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        CountBarriers(CommandClass::LabelRun);
        run.open = true;
        storeRuns.fetch_add(1, std::memory_order_relaxed);
    } else if (run.buffer == buffer && !run.bytes.empty()) {
        const VkDeviceSize pendingEnd = run.offset + run.bytes.size();
        if (offset >= run.offset && end <= pendingEnd) {
            std::memcpy(run.bytes.data() + (offset - run.offset), bytes.data(), bytes.size());
            storesReplaced.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (offset == pendingEnd && run.bytes.size() + bytes.size() <= 65536) {
            // A joined store shares the pending vkCmdUpdateBuffer, which no barrier precedes: one
            // that overlaps a store the run already recorded takes the WAW barrier below instead.
            if (!JoinWawCheck() || !overlapsRecorded()) {
                run.bytes.insert(run.bytes.end(), bytes.begin(), bytes.end());
                storesJoined.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            storeJoinsRefused.fetch_add(1, std::memory_order_relaxed);
        }
    }
    flushPendingStore();
    const bool waw = overlapsRecorded();
    if (waw) {
        recordBarrier(CommandsOf(open->metal), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        CountBarriers(CommandClass::LabelRun);
        storeWawBarriers.fetch_add(1, std::memory_order_relaxed);
    }
    run.buffer = buffer;
    run.offset = offset;
    run.bytes.assign(bytes.begin(), bytes.end());
    if (!LabelRunsEnabled()) closeStoreRun();
}

void Recorder::recordBarrier(VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags, VkAccessFlags) const {
    // macOS: the access masks have no Metal counterpart (every barrier makes its writes visible to
    // the device); host stages vanish (a completed command buffer's writes reach the CPU).
    Metal::Stream(commands).PipelineBarrier(sourceStage, destinationStage);
}

void Recorder::flushPendingStore() {
    auto& run = open->run;
    if (run.bytes.empty()) return;
    open->metal->stream->UpdateBuffer(Metal::BufferOf(run.buffer), run.offset, run.bytes);
    run.recorded.emplace_back(run.buffer, run.offset, run.offset + run.bytes.size());
    run.storedBytes += run.bytes.size();
    run.bytes.clear();
}

std::string DebugImportResidency(void* buffer);

namespace {

// APS5_TRACE_LABEL_LANDING: the label copies a batch recorded, reported when the GPU finishes it.
struct LandingCopy {
    id<MTLBuffer> source;
    NSUInteger sourceOffset;
    id<MTLBuffer> destination;
    NSUInteger destinationOffset;
    std::size_t bytes;
    std::uint64_t address;
    id<MTLBuffer> readback;
    NSUInteger readbackOffset;
};
std::mutex landingMutex;
std::multimap<std::uint64_t, LandingCopy> landingCopies;

void NoteLandingCopy(std::uint64_t serial, LandingCopy copy) {
    std::lock_guard lock(landingMutex);
    landingCopies.emplace(serial, std::move(copy));
}

void ReportLandingCopies(std::uint64_t serial) {
    std::vector<LandingCopy> copies;
    {
        std::lock_guard lock(landingMutex);
        const auto [first, last] = landingCopies.equal_range(serial);
        for (auto it = first; it != last; ++it) copies.push_back(it->second);
        landingCopies.erase(first, last);
    }
    for (const auto& copy : copies) {
        std::uint64_t source = 0, destination = 0, guest = 0, gpuSees = 0;
        std::memcpy(&gpuSees, static_cast<const std::byte*>(copy.readback.contents) + copy.readbackOffset, copy.bytes);
        std::uint64_t gpuPage = 0, cpuPage = 0;
        std::memcpy(&gpuPage, static_cast<const std::byte*>(copy.readback.contents) + copy.readbackOffset + 8, 8);
        std::memcpy(&cpuPage, reinterpret_cast<const void*>(::GuestArena::HostAddress(copy.address & ~std::uint64_t{0x3fff})), 8);
        std::memcpy(&source, static_cast<const std::byte*>(copy.source.contents) + copy.sourceOffset, copy.bytes);
        std::memcpy(&destination, static_cast<const std::byte*>(copy.destination.contents) + copy.destinationOffset, copy.bytes);
        std::memcpy(&guest, reinterpret_cast<const void*>(::GuestArena::HostAddress(copy.address)), copy.bytes);
        std::fprintf(stderr, "[landing] batch %llu done: copy of 0x%llx: upload holds 0x%llx, destination buffer %p holds 0x%llx, guest memory holds 0x%llx, the GPU read back 0x%llx through the destination; page start: GPU 0x%llx CPU 0x%llx\n", static_cast<unsigned long long>(serial), static_cast<unsigned long long>(copy.address), static_cast<unsigned long long>(source), (__bridge void*)copy.destination, static_cast<unsigned long long>(destination), static_cast<unsigned long long>(guest), static_cast<unsigned long long>(gpuSees), static_cast<unsigned long long>(gpuPage), static_cast<unsigned long long>(cpuPage));
    }
}

}

bool Recorder::closeStoreRun(bool atSubmit) {
    auto& run = open->run;
    if (LabelRunsPerBatch()) {
        auto stores = std::move(run.queued);
        run.queued.clear();
        open->queuedDumps = 0;
        if (stores.empty()) return false;
        // A label says the work before it is done: the queued key stores (results of that work)
        // land ahead of every store of the run, and the pass a draw left open ends.
        if (!open->keyStores.empty()) recordKeyStores(true);
        if (open->renderPass.open) endOpenRenderPass();
        const auto commands = CommandsOf(open->metal);
        run.timing = beginTiming(ClassKey(CommandClass::LabelRun));
        if (BarrierValidate()) {
            std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
            for (const auto& store : stores) ranges.emplace_back(store.address, store.End());
            NoteAccess(CommandClass::LabelRun, Access{{}, ranges, {}, VK_PIPELINE_STAGE_TRANSFER_BIT});
        }
        // At Submit every result of the batch reaches the host before the stores (the title's
        // pollers read results after the label); in place, later work sees the stores as well.
        if (atSubmit) recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
        else recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        std::uint32_t barriers = 2;
        std::uint64_t storedBytes = 0;
        auto& stream = Metal::Stream(commands);
        std::vector<QueuedWrite> writes;
        writes.reserve(stores.size());
        for (const auto& store : stores) writes.push_back({store.address, store.End(), store.dumpTarget != 0});
        // eef987ae7b: the run's occlusion dumps move together (QueuedWriteOrder) so they fold once.
        if (const auto order = QueuedWriteOrder(writes); !std::is_sorted(order.begin(), order.end())) {
            std::vector<Batch::StoreRun::Queued> ordered;
            std::vector<QueuedWrite> orderedWrites;
            ordered.reserve(stores.size());
            orderedWrites.reserve(writes.size());
            for (const auto index : order) {
                ordered.push_back(std::move(stores[index]));
                orderedWrites.push_back(writes[index]);
            }
            stores = std::move(ordered);
            writes = std::move(orderedWrites);
        }
        const bool dumps = std::any_of(writes.begin(), writes.end(), [](const QueuedWrite& write) { return write.computed; });
        if (dumps) recordQueuedDumps(commands, stores);
        std::size_t first = 0;
        std::uint32_t dumpIndex = 0, consumed = 0;
        for (const auto end : QueuedWriteGroups(writes)) {
            if (first != 0) {
                recordBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                ++barriers;
                run.recorded.clear();
            }
            for (std::size_t i = first; i < end; ++i) {
                const auto& store = stores[i];
                if (store.dumpTarget != 0) continue;
                const VkDeviceSize storeEnd = store.offset + store.bytes.size();
                if (std::any_of(run.recorded.begin(), run.recorded.end(), [&](const auto& earlier) { return std::get<0>(earlier) == store.buffer && store.offset < std::get<2>(earlier) && std::get<1>(earlier) < storeEnd; })) {
                    recordBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                    ++barriers;
                    storeWawBarriers.fetch_add(1, std::memory_order_relaxed);
                }
                static const bool traceLanding = std::getenv("APS5_TRACE_LABEL_LANDING") != nullptr;
                if (traceLanding && store.address != 0) {
                    auto target = Metal::BufferOf(store.buffer);
                    const auto host = reinterpret_cast<std::uint64_t>(target.contents) + store.offset;
                    std::fprintf(stderr, "[landing] store run: guest 0x%llx+0x%zx into buffer %p (contents %p, length 0x%llx) offset 0x%llx = host 0x%llx%s\n", static_cast<unsigned long long>(store.address), store.bytes.size(), (__bridge void*)target, target.contents, static_cast<unsigned long long>(target.length), static_cast<unsigned long long>(store.offset), static_cast<unsigned long long>(host), host == ::GuestArena::HostAddress(store.address) ? "" : " (NOT the guest's host address)");
                    std::fprintf(stderr, "[landing]   page now: %s\n", ::AgcDriver::Metal::VmWiredInfo(host).c_str());
                    // APS5_TRACE_LABEL_LANDING_WATCH=1: backtrace of the first CPU store to the page after it.
                    static const bool watchPage = std::getenv("APS5_TRACE_LABEL_LANDING_WATCH") != nullptr;
                    static bool watched = false;
                    if (watchPage && !watched) {
                        watched = true;
                        ::AgcDriver::Metal::WatchWrites(host);
                    }
                }
                if (traceLanding && store.address != 0 && store.bytes.size() <= 8) {
                    // The same copy UpdateBuffer records, with its source kept so the batch's completion
                    // can report what the GPU copied from and what the destination holds afterwards.
                    const auto upload = stream.Write(store.bytes, 4);
                    [stream.Compute() copyFromBuffer:upload.buffer sourceOffset:upload.offset toBuffer:Metal::BufferOf(store.buffer) destinationOffset:store.offset size:store.bytes.size()];
                    // Read the stored bytes back through the same buffer on the GPU, into upload memory.
                    const auto readback = stream.Allocate(16, 8);
                    std::memset(readback.host, 0xee, 16);
                    [stream.Compute() barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
                    [stream.Compute() copyFromBuffer:Metal::BufferOf(store.buffer) sourceOffset:store.offset toBuffer:readback.buffer destinationOffset:readback.offset size:store.bytes.size()];
                    // Also the first 8 bytes of the store's 16 KiB page, which the CPU wrote (a command buffer header).
                    [stream.Compute() copyFromBuffer:Metal::BufferOf(store.buffer) sourceOffset:(store.offset & ~VkDeviceSize{0x3fff}) toBuffer:readback.buffer destinationOffset:readback.offset + 8 size:8];
                    std::fprintf(stderr, "[landing]   residency: %s; recorder queue %p\n", DebugImportResidency((__bridge void*)Metal::BufferOf(store.buffer)).c_str(), (__bridge void*)metal->queue);
                    NoteLandingCopy(submissions + 1, LandingCopy{upload.buffer, upload.offset, Metal::BufferOf(store.buffer), store.offset, store.bytes.size(), store.address, readback.buffer, readback.offset});
                } else {
                    stream.UpdateBuffer(Metal::BufferOf(store.buffer), store.offset, store.bytes);
                }
                run.recorded.emplace_back(store.buffer, store.offset, storeEnd);
                storedBytes += store.bytes.size();
            }
            const auto before = dumpIndex;
            foldQueuedDumps(commands, stores, first, end, dumpIndex, consumed);
            if (dumpIndex != before) barriers += 3;
            first = end;
        }
        if (dumps) {
            retireSegments(consumed);
            dumpFolds.fetch_add(1, std::memory_order_relaxed);
        }
        const VkPipelineStageFlags written = VK_PIPELINE_STAGE_TRANSFER_BIT | (dumps ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0u);
        const VkAccessFlags writtenAccess = VK_ACCESS_TRANSFER_WRITE_BIT | (dumps ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
        if (atSubmit) {
            recordBarrier(commands, written, VK_PIPELINE_STAGE_HOST_BIT, writtenAccess, VK_ACCESS_HOST_READ_BIT);
            open->coveredAccess = 0;
            storeRunsAtSubmit.fetch_add(1, std::memory_order_relaxed);
        } else {
            recordBarrier(commands, written, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, writtenAccess, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            open->coveredAccess = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            storeRunsForced.fetch_add(1, std::memory_order_relaxed);
        }
        CountBarriers(CommandClass::LabelRun, barriers);
        EndGpuTiming(run.timing, storedBytes);
        run.timing = NoTiming;
        run.recorded.clear();
        return atSubmit;
    }
    flushPendingStore();
    // The run's stores are visible to the host and to everything recorded after them.
    recordBarrier(CommandsOf(open->metal), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    CountBarriers(CommandClass::LabelRun);
    EndGpuTiming(run.timing, run.storedBytes);
    run.timing = NoTiming;
    open->coveredAccess = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    run.open = false;
    run.buffer = VK_NULL_HANDLE;
    run.recorded.clear();
    return false;
}

bool Recorder::QueuedStoreOverlaps(std::uint64_t address, std::size_t bytes) const {
    if (open == nullptr || bytes == 0) return false;
    const auto end = address + bytes;
    return std::any_of(open->run.queued.begin(), open->run.queued.end(), [&](const Batch::StoreRun::Queued& store) { return address < store.End() && store.address < end; });
}

bool Recorder::AnyQueuedStore(const std::function<bool(std::uint64_t, std::uint64_t)>& overlaps) const {
    if (open == nullptr) return false;
    return std::any_of(open->run.queued.begin(), open->run.queued.end(), [&](const Batch::StoreRun::Queued& store) { return overlaps(store.address, store.End()); });
}

void Recorder::FlushStores() {
    if (open == nullptr || open->run.queued.empty()) return;
    closeStoreRun(false);
}

namespace {

constexpr std::uint32_t MaxTimedRanges = 512;
constexpr std::size_t CommandClasses = static_cast<std::size_t>(Recorder::CommandClass::Count);
constexpr const char* CommandClassNames[CommandClasses] = {"dispatch-lead", "dispatch-trail", "indirect-args", "label-run", "fill", "fill-clear", "copy", "staging-in", "staging-out", "draw", "storage-upload", "storage-writeback", "dcc-clear", "dcc-keys", "present-blit", "shadow-publish", "template-refresh"};
// [barriers] counters by class (relaxed: only reported) and the [gputime] totals, which the
// presenter's thread adds to without the GPU mutex (AddGpuTiming).
std::atomic<std::uint64_t> classBarriers[CommandClasses]{};
std::atomic<std::uint64_t> classMerged[CommandClasses]{};
// The hazard tracker's counts (APS5_BARRIER_VALIDATE=1): simulated barriers by class and by
// hazard kind, leading barriers it would have skipped by class, batch-end barriers, and the
// APS5_TRACE_BARRIERS budget left.
constexpr std::size_t HazardKinds = 5;
constexpr const char* HazardNames[HazardKinds] = {"raw", "waw", "war", "image", "conservative"};
std::atomic<std::uint64_t> validateEmitted[CommandClasses]{}, validateSkipped[CommandClasses]{}, validateKinds[HazardKinds]{};
std::atomic<std::uint64_t> validateBatchEnds{0};
std::atomic<std::int64_t> traceBarriersLeft{[] {
    const char* text = std::getenv("APS5_TRACE_BARRIERS");
    return text != nullptr ? static_cast<std::int64_t>(std::strtoll(text, nullptr, 10)) : std::int64_t{0};
}()};
std::atomic<std::uint64_t> timingPresents{0};
std::atomic<std::uint64_t> presentSerial{0};
std::atomic<std::uint64_t> timingDropped{0};
std::atomic<std::uint64_t> timingInvalid{0};
struct TimingTotals { std::uint64_t count = 0; double ms = 0; std::uint64_t bytes = 0; };
std::mutex timingMutex;
std::map<std::uint64_t, TimingTotals> timingByKey;
double timingProgramMs = 0, timingClassMs = 0, timingUnionMs = 0, timingBatchMs = 0;
std::uint64_t timingBatches = 0;

bool DrawOrGpuProfiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr || Recorder::GpuTimingEnabled();
    return profiled;
}

void reportBarriers() {
    if (!DrawOrGpuProfiled()) return;
    static auto lastReport = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::uint64_t total = 0, merged = 0;
    std::string line, mergedLine;
    for (std::size_t i = 0; i < CommandClasses; ++i) {
        const auto count = classBarriers[i].exchange(0, std::memory_order_relaxed);
        total += count;
        char text[64];
        std::snprintf(text, sizeof(text), " %s %llu", CommandClassNames[i], static_cast<unsigned long long>(count));
        line += text;
        const auto mergedCount = classMerged[i].exchange(0, std::memory_order_relaxed);
        if (mergedCount == 0) continue;
        merged += mergedCount;
        std::snprintf(text, sizeof(text), " %s %llu", CommandClassNames[i], static_cast<unsigned long long>(mergedCount));
        mergedLine += text;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[barriers] %llu recorded (10 s) by class:%s; merged %llu:%s", static_cast<unsigned long long>(total), line.c_str(), static_cast<unsigned long long>(merged), mergedLine.c_str());
    if (Recorder::BarrierValidate()) {
        std::uint64_t emitted = 0, skipped = 0;
        std::string classes, kinds;
        for (std::size_t i = 0; i < CommandClasses; ++i) {
            const auto emittedCount = validateEmitted[i].exchange(0, std::memory_order_relaxed);
            const auto skippedCount = validateSkipped[i].exchange(0, std::memory_order_relaxed);
            emitted += emittedCount;
            skipped += skippedCount;
            if (emittedCount == 0 && skippedCount == 0) continue;
            char text[96];
            std::snprintf(text, sizeof(text), " %s %llu/%llu", CommandClassNames[i], static_cast<unsigned long long>(emittedCount), static_cast<unsigned long long>(skippedCount));
            classes += text;
        }
        for (std::size_t i = 0; i < HazardKinds; ++i) {
            char text[64];
            std::snprintf(text, sizeof(text), " %s %llu", HazardNames[i], static_cast<unsigned long long>(validateKinds[i].exchange(0, std::memory_order_relaxed)));
            kinds += text;
        }
        AgcDriver::ProfilePrint_nid_no_patch("; validate: would emit %llu (%s) + %llu batch ends, would skip %llu; emitted/skipped by class:%s", static_cast<unsigned long long>(emitted), kinds.c_str() + 1, static_cast<unsigned long long>(validateBatchEnds.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(skipped), classes.c_str());
    }
    AgcDriver::ProfilePrint_nid_no_patch("\n");
    AgcDriver::ProfilePrint_nid_no_patch("[samples] occlusion dumps (cumulative): %llu queued (%llu inside an open pass), %llu recorded at once; %llu segments closed; %llu folds of queued dumps\n", static_cast<unsigned long long>(dumpsQueued.load(std::memory_order_relaxed)), static_cast<unsigned long long>(dumpsQueuedInPass.load(std::memory_order_relaxed)), static_cast<unsigned long long>(dumpsAtOnce.load(std::memory_order_relaxed)), static_cast<unsigned long long>(segmentsClosed.load(std::memory_order_relaxed)), static_cast<unsigned long long>(dumpFolds.load(std::memory_order_relaxed)));
}

}

bool Recorder::GpuTimingEnabled() {
    static const bool enabled = std::getenv("APS5_PROFILE_GPU") != nullptr;
    return enabled;
}

bool Recorder::BatchStampsEnabled() {
    return GpuTimingEnabled() || DrawProfiled() || PerformanceTrace::Configured();
}

void Recorder::CountBarriers(CommandClass which, std::uint32_t count) {
    if (!DrawOrGpuProfiled()) return;
    classBarriers[static_cast<std::size_t>(which)].fetch_add(count, std::memory_order_relaxed);
}

bool Recorder::MergeBarriers() {
    static const bool merge = std::getenv("APS5_FULL_BARRIERS") == nullptr && std::getenv("APS5_NO_BARRIER_ELISION") == nullptr;
    return merge;
}

void Recorder::CountMerged(CommandClass which) {
    if (!DrawOrGpuProfiled()) return;
    classMerged[static_cast<std::size_t>(which)].fetch_add(1, std::memory_order_relaxed);
}

bool Recorder::BarrierValidate() {
    static const bool validate = std::getenv("APS5_BARRIER_VALIDATE") != nullptr;
    return validate;
}

void Recorder::NoteAccess(CommandClass which, const Access& access) {
    if (!BarrierValidate()) return;
    ensureOpen();
    auto& tracker = open->tracker;
    const auto overlapsAny = [](const auto& ranges, const std::pair<std::uint64_t, std::uint64_t>& range) {
        return std::any_of(ranges.begin(), ranges.end(), [&](const auto& other) { return range.first < other.end && other.begin < range.second; });
    };
    std::size_t kind = HazardKinds;
    if (access.conservative) kind = 4;
    for (const auto& range : access.reads) {
        if (kind != HazardKinds) break;
        if (overlapsAny(tracker.writes, range)) kind = 0;
    }
    for (const auto& range : access.writes) {
        if (kind != HazardKinds) break;
        if (overlapsAny(tracker.writes, range)) kind = 1;
        else if (overlapsAny(tracker.reads, range)) kind = 2;
    }
    for (const auto& [image, written] : access.images) {
        if (kind != HazardKinds) break;
        if (std::any_of(tracker.images.begin(), tracker.images.end(), [&](const auto& seen) { return seen.image == image && (seen.written || written); })) kind = 3;
    }
    const auto index = static_cast<std::size_t>(which);
    if (kind != HazardKinds) {
        validateEmitted[index].fetch_add(1, std::memory_order_relaxed);
        validateKinds[kind].fetch_add(1, std::memory_order_relaxed);
        tracker.reads.clear();
        tracker.writes.clear();
        tracker.images.clear();
    } else {
        validateSkipped[index].fetch_add(1, std::memory_order_relaxed);
        if (traceBarriersLeft.load(std::memory_order_relaxed) > 0 && traceBarriersLeft.fetch_sub(1, std::memory_order_relaxed) > 0) {
            const auto first = [](const auto& ranges) { return ranges.empty() ? std::pair<std::uint64_t, std::uint64_t>{0, 0} : std::pair<std::uint64_t, std::uint64_t>{ranges.front().first, ranges.front().second}; };
            const auto reads = first(access.reads);
            const auto writes = first(access.writes);
            std::fprintf(stderr, "[barriers] would skip %s: reads %zu (first 0x%llx+0x%llx), writes %zu (first 0x%llx+0x%llx), images %zu; since the last barrier: %zu reads, %zu writes, %zu images\n", CommandClassNames[index], access.reads.size(), static_cast<unsigned long long>(reads.first), static_cast<unsigned long long>(reads.second - reads.first), access.writes.size(), static_cast<unsigned long long>(writes.first), static_cast<unsigned long long>(writes.second - writes.first), access.images.size(), tracker.reads.size(), tracker.writes.size(), tracker.images.size());
        }
    }
    for (const auto& [begin, end] : access.reads) tracker.reads.push_back({begin, end, access.stages});
    for (const auto& [begin, end] : access.writes) tracker.writes.push_back({begin, end, access.stages});
    for (const auto& [image, written] : access.images) tracker.images.push_back({image, written, access.stages});
}

void Recorder::CountPresent() {
    timingPresents.fetch_add(1, std::memory_order_relaxed);
    presentSerial.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t Recorder::Presents() {
    return presentSerial.load(std::memory_order_relaxed);
}

void Recorder::AddGpuTiming(CommandClass which, double nanoseconds, std::uint64_t bytes) {
    std::lock_guard lock(timingMutex);
    auto& totals = timingByKey[ClassKey(which)];
    ++totals.count;
    totals.ms += nanoseconds / 1e6;
    totals.bytes += bytes;
    timingClassMs += nanoseconds / 1e6;
}

std::uint32_t Recorder::BeginGpuTiming(std::uint64_t key) {
    if (!GpuTimingEnabled()) return NoTiming;
    Commands();
    return beginTiming(key);
}

std::uint32_t Recorder::beginTiming(std::uint64_t key) {
    const bool full = GpuTimingEnabled();
    if (!full && !(key == BatchTimingKey && BatchStampsEnabled())) return NoTiming;
    const std::uint32_t queryCount = full ? MaxTimedRanges * 2 : 2;
    auto& objects = *open->metal;
    if (objects.timestamps == nil) {
        // macOS: an MTL4 counter heap of timestamps (a two-entry one, for the batch stamps alone,
        // is reused from the batches that finished).
        if (!full && !metal->spareHeaps.empty()) {
            objects.timestamps = metal->spareHeaps.back();
            metal->spareHeaps.pop_back();
        } else {
            MTL4CounterHeapDescriptor* descriptor = [MTL4CounterHeapDescriptor new];
            descriptor.type = MTL4CounterHeapTypeTimestamp;
            descriptor.count = queryCount;
            objects.timestamps = [metal->device newCounterHeapWithDescriptor:descriptor error:nil];
            if (objects.timestamps == nil) return NoTiming;
        }
        [objects.timestamps invalidateCounterRange:NSMakeRange(0, objects.timestamps.count)];
    }
    if (open->timedKeys.size() >= objects.timestamps.count / 2) {
        timingDropped.fetch_add(1, std::memory_order_relaxed);
        return NoTiming;
    }
    const auto index = static_cast<std::uint32_t>(open->timedKeys.size());
    open->timedKeys.push_back(key);
    open->timedBytes.push_back(0);
    // Both stamps follow the work recorded before them, so the range is the timed work alone.
    objects.stream->WriteTimestamp(objects.timestamps, index * 2);
    return index;
}

void Recorder::EndGpuTiming(std::uint32_t index, std::uint64_t bytes) {
    if (index == NoTiming || open == nullptr || open->metal->timestamps == nil) return;
    if (index < open->timedBytes.size()) open->timedBytes[index] += bytes;
    open->metal->stream->WriteTimestamp(open->metal->timestamps, index * 2 + 1);
}

void Recorder::CountSamples() {
    GuestMemory::AssertGpuLockHeld("Recorder::CountSamples");
    gpuSampleCounter();
    countingSamples = true;
    if (segmentMode() || open == nullptr || open->sampleActive) return;
    if (open->renderPass.open) endOpenRenderPass();
    beginSamples(*open);
}

std::uint64_t Recorder::SamplesTotal() {
    if (sampleCounterState <= 0) return samplesPassed.load(std::memory_order_acquire);
    if (segmentMode() && !Idle()) Sync();
    std::uint64_t total = 0;
    std::memcpy(&total, sampleCounter->Bytes().data(), sizeof(total));
    // macOS: the segments not folded yet are read from their visibility result buffers (every batch
    // that counted into them finished in the Sync above).
    for (const auto& segment : pendingSamples) {
        std::uint64_t samples = 0;
        const auto* pool = static_cast<const SamplePool*>(segment.pool.get());
        Require(pool != nullptr, "an occlusion segment lost its visibility buffer");
        std::memcpy(&samples, static_cast<const std::byte*>(pool->pool.contents) + 8u * segment.slot, sizeof(samples));
        total += samples;
    }
    pendingSamples.clear();
    std::memcpy(sampleCounter->Bytes().data(), &total, sizeof(total));
    return total;
}

void Recorder::NoteSampledDraw(VkCommandBuffer commands) {
    if (open == nullptr) return;
    if (segmentMode()) {
        if (countingSamples && commands == CommandsOf(open->metal) && !open->segmentActive) beginSegment(*open);
        return;
    }
    if (open->sampleActive) open->samplesDrawn = true;
}

void* Recorder::SampleVisibilityBuffer() const {
    if (open == nullptr || !countingSamples || !segmentMode() || open->segmentPool == nullptr) return nullptr;
    return (__bridge void*)static_cast<const SamplePool*>(open->segmentPool.get())->pool;
}

bool Recorder::gpuSampleCounter() {
    if (sampleCounterState != 0) return sampleCounterState > 0;
    sampleCounterState = -1;
    // macOS: the counter and dump list in one shared buffer, the two fold kernels in MSL.
    try {
        sampleCounter = std::make_unique<Buffer>(context, DumpListOffset + 16 * DumpFoldCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        std::memset(sampleCounter->Bytes().data(), 0, sampleCounter->Bytes().size());
        const auto initial = samplesPassed.load(std::memory_order_acquire);
        std::memcpy(sampleCounter->Bytes().data(), &initial, sizeof(initial));
        static_cast<void>(Metal::HelperPipeline(metal->device, Metal::HelperKernel::SampleCounter));
        static_cast<void>(Metal::HelperPipeline(metal->device, Metal::HelperKernel::SampleDumps));
    } catch (const std::exception& error) {
        sampleCounter.reset();
        std::fprintf(stderr, "[gpu] occlusion counter dumps drain the device: %s\n", error.what());
        return false;
    }
    sampleCounterState = 1;
    return true;
}

std::uint64_t Recorder::Batch::StoreRun::Queued::End() const {
    return address + (dumpTarget != 0 ? SampleDumpBytes : bytes.size());
}

std::vector<std::size_t> Recorder::QueuedWriteOrder(std::span<const QueuedWrite> writes) {
    std::vector<std::size_t> order(writes.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::vector<bool> coveredLater(writes.size()), overLater(writes.size());
    std::map<std::uint64_t, std::uint64_t> later;
    for (auto i = writes.size(); i-- > 0;) {
        const auto& write = writes[i];
        if (!write.computed) {
            const auto after = later.upper_bound(write.begin);
            coveredLater[i] = after != later.begin() && std::prev(after)->second >= write.end;
            const auto last = later.lower_bound(write.end);
            overLater[i] = last != later.begin() && std::prev(last)->second > write.begin;
            continue;
        }
        auto begin = write.begin, end = write.end;
        auto at = later.lower_bound(begin);
        if (at != later.begin() && std::prev(at)->second >= begin) --at;
        while (at != later.end() && at->first <= end) {
            begin = std::min(begin, at->first);
            end = std::max(end, at->second);
            at = later.erase(at);
        }
        later.emplace(begin, end);
    }
    std::vector<std::size_t> before, dumps, after;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> earlier, delayed;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        const auto& write = writes[i];
        const auto over = [&](const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
            return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return write.begin < range.second && range.first < write.end; });
        };
        if (write.computed) {
            dumps.push_back(i);
            earlier.emplace_back(write.begin, write.end);
        } else if (dumps.empty()) {
            before.push_back(i);
        } else if (overLater[i]) {
            if (over(delayed) || (!coveredLater[i] && over(earlier))) return order;
            before.push_back(i);
        } else {
            after.push_back(i);
            delayed.emplace_back(write.begin, write.end);
        }
    }
    order = std::move(before);
    order.insert(order.end(), dumps.begin(), dumps.end());
    order.insert(order.end(), after.begin(), after.end());
    return order;
}

std::vector<std::size_t> Recorder::QueuedWriteGroups(std::span<const QueuedWrite> writes) {
    std::vector<std::size_t> ends;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> computed;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        const auto& write = writes[i];
        if (!write.computed && !computed.empty()) {
            ends.push_back(i);
            computed.clear();
        }
        if (write.computed) computed.emplace_back(write.begin, write.end);
    }
    if (!writes.empty()) ends.push_back(writes.size());
    return ends;
}

bool Recorder::segmentMode() const {
    return sampleCounterState > 0;
}

bool Recorder::QueuesSampleDumps() const {
    return segmentMode() && LabelRunsPerBatch();
}

Recorder::SampleDumpStatistics Recorder::SampleDumpCounts() {
    return SampleDumpStatistics{dumpsQueued.load(std::memory_order_relaxed), dumpsQueuedInPass.load(std::memory_order_relaxed), dumpsAtOnce.load(std::memory_order_relaxed), segmentsClosed.load(std::memory_order_relaxed), dumpFolds.load(std::memory_order_relaxed)};
}

bool Recorder::segmentSlotReady(const Batch& batch) const {
    return batch.segmentActive || (batch.segments != nullptr && batch.segmentNext < SegmentSlots);
}

void Recorder::PrepareSampleSlot() {
    GuestMemory::AssertGpuLockHeld("Recorder::PrepareSampleSlot");
    if (!countingSamples || !segmentMode()) return;
    ensureOpen();
    Require(!open->renderPass.open, "a sample slot is prepared inside a render pass");
    if (!segmentSlotReady(*open)) takeSegmentPool(*open);
}

void Recorder::PrepareSamplePass() {
    GuestMemory::AssertGpuLockHeld("Recorder::PrepareSamplePass");
    if (!countingSamples || !segmentMode()) return;
    ensureOpen();
    if (open->segmentActive) endSegment(*open);
    if (!segmentSlotReady(*open)) takeSegmentPool(*open);
}

void Recorder::EndPassSamples() {
    if (open != nullptr && open->segmentActive) endSegment(*open);
}

void Recorder::takeSegmentPool(Batch& batch) {
    auto cache = std::static_pointer_cast<SamplePoolCache>(segmentPools);
    if (cache == nullptr) {
        cache = std::make_shared<SamplePoolCache>();
        cache->residency = metal->residency;
        segmentPools = cache;
    }
    if (batch.segmentPool != nullptr) batch.kept.push_back(std::move(batch.segmentPool));
    batch.segmentPool.reset();
    batch.segments = nullptr;
    // macOS: a visibility result buffer of SegmentSlots counters, zeroed here (the pass accumulates
    // into its slot; a recycled buffer's batches all finished) in place of vkCmdResetQueryPool.
    MetalBuffer pool = cache->take();
    if (pool == nil) {
        pool = [metal->device newBufferWithLength:8u * SegmentSlots options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        if (pool == nil) Metal::Fail("cannot allocate an occlusion visibility buffer");
        // Kept resident for the recorder's life: the cache bounds how many there are.
        ::AgcDriver::Metal::Residency::Add(metal->residency, pool);
        ::AgcDriver::Metal::Residency::Commit(metal->residency);
    }
    std::memset(pool.contents, 0, pool.length);
    try {
        batch.segmentPool = std::make_shared<SamplePool>(cache, pool);
    } catch (...) {
        cache->put(pool);
        throw;
    }
    batch.segments = (__bridge const void*)pool;
    batch.segmentNext = 0;
}

void Recorder::beginSegment(Batch& batch) {
    Require(batch.segments != nullptr && batch.segmentNext < SegmentSlots, "an occlusion segment begins without a reset query slot");
    auto& stream = *batch.metal->stream;
    MetalRenderEncoder render = stream.Render();
    Require(render != nil, "an occlusion segment begins outside a render pass");
    Require((__bridge const void*)stream.RenderVisibilityBuffer() == batch.segments, "the sampled draw's render pass was begun without the recorder's visibility buffer (Recorder::SampleVisibilityBuffer)");
    batch.segmentSlot = batch.segmentNext++;
    [render setVisibilityResultMode:MTLVisibilityResultModeCounting offset:8u * batch.segmentSlot];
    batch.segmentActive = true;
}

void Recorder::endSegment(Batch& batch) {
    if (MetalRenderEncoder render = batch.metal->stream->Render(); render != nil) [render setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    batch.segmentActive = false;
    pendingSamples.push_back({batch.segmentPool, batch.segments, batch.segmentSlot});
    segmentsClosed.fetch_add(1, std::memory_order_relaxed);
}

void Recorder::copySegments(VkCommandBuffer commands, std::uint32_t count) {
    Require(count <= pendingSamples.size() && count <= SegmentFoldCapacity, "occlusion segments exceed the fold capacity");
    auto& stream = Metal::Stream(commands);
    MetalBuffer counter = Metal::BufferOf(sampleCounter->Handle());
    for (std::uint32_t i = 0; i < count;) {
        const auto& first = pendingSamples[i];
        std::uint32_t run = 1;
        while (i + run < count && pendingSamples[i + run].handle == first.handle && pendingSamples[i + run].slot == first.slot + run) ++run;
        stream.CopyBuffer(static_cast<const SamplePool*>(first.pool.get())->pool, 8u * first.slot, counter, 16u + 8u * i, 8u * run);
        i += run;
    }
}

void Recorder::retireSegments(std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; ++i) {
        if (pendingSamples[i].pool != nullptr) open->kept.push_back(std::move(pendingSamples[i].pool));
    }
    pendingSamples.erase(pendingSamples.begin(), pendingSamples.begin() + count);
}

void Recorder::recordQueuedDumps(VkCommandBuffer commands, std::span<const Batch::StoreRun::Queued> writes) {
    std::uint32_t segments = 0;
    std::vector<std::uint32_t> list;
    for (const auto& write : writes) {
        if (write.dumpTarget == 0) continue;
        segments = write.dumpSegments;
        list.insert(list.end(), {static_cast<std::uint32_t>(write.dumpTarget), static_cast<std::uint32_t>(write.dumpTarget >> 32u), write.dumpSegments, 0u});
    }
    Require(list.size() <= 4u * DumpFoldCapacity, "queued occlusion dumps exceed the fold capacity");
    copySegments(commands, segments);
    Metal::Stream(commands).UpdateBuffer(Metal::BufferOf(sampleCounter->Handle()), DumpListOffset, std::as_bytes(std::span(list)));
}

void Recorder::dispatchDumps(VkCommandBuffer commands, std::uint32_t firstDump, std::uint32_t dumps, std::uint32_t firstSegment, std::uint32_t lastSegment) {
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    struct {
        VkDeviceAddress counter;
        VkDeviceAddress list;
        std::uint32_t firstDump;
        std::uint32_t dumps;
        std::uint32_t firstSegment;
        std::uint32_t lastSegment;
        std::uint32_t phase;
    } parameters{sampleCounter->DeviceAddress(), sampleCounter->DeviceAddress() + DumpListOffset, firstDump, dumps, firstSegment, lastSegment, 0u};
    static_assert(sizeof(parameters) == SampleDumpPushBytes);
    auto& stream = Metal::Stream(commands);
    MetalPipeline pipeline = Metal::HelperPipeline(metal->device, Metal::HelperKernel::SampleDumps);
    stream.DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    if (dumps == 0) return;
    // The counts land before the ready bit (phases 1 then 2, each behind a barrier).
    const auto publish = [&](std::uint32_t phase) {
        recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
        parameters.phase = phase;
        stream.DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    };
    publish(1u);
    publish(2u);
}

void Recorder::foldQueuedDumps(VkCommandBuffer commands, std::span<const Batch::StoreRun::Queued> writes, std::size_t first, std::size_t end, std::uint32_t& dumpIndex, std::uint32_t& consumed) {
    std::uint32_t count = 0, segments = consumed;
    for (std::size_t i = first; i < end; ++i) {
        if (writes[i].dumpTarget == 0) continue;
        ++count;
        segments = writes[i].dumpSegments;
    }
    if (count == 0) return;
    dispatchDumps(commands, dumpIndex, count, consumed, segments);
    dumpIndex += count;
    consumed = segments;
}

void Recorder::foldSegments() {
    const auto commands = CommandsOf(open->metal);
    const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(pendingSamples.size(), SegmentFoldCapacity));
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    copySegments(commands, count);
    dispatchDumps(commands, 0, 0, 0, count);
    recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
    CountBarriers(CommandClass::Copy, 3);
    retireSegments(count);
}

void Recorder::endSamples(Batch& batch) {
    // macOS: whole-batch queries (the mode without the fold kernels) do not exist; beginSamples
    // refuses them, so nothing is active here.
    if (!batch.sampleActive) return;
    batch.sampleActive = false;
    batch.samplesDrawn = false;
}

void Recorder::foldSamples(Batch& batch, VkDeviceAddress target) {
    const auto commands = CommandsOf(batch.metal);
    auto& stream = Metal::Stream(commands);
    MetalBuffer counter = Metal::BufferOf(sampleCounter->Handle());
    MetalPipeline pipeline = Metal::HelperPipeline(metal->device, Metal::HelperKernel::SampleCounter);
    std::size_t done = 0;
    do {
        const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(SampleFoldCapacity, pendingSamples.size() - done));
        if (done != 0) recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& segment = pendingSamples[done + i];
            stream.CopyBuffer(static_cast<const SamplePool*>(segment.pool.get())->pool, 8u * segment.slot, counter, 16u + 8u * i, 8u);
            if (segment.pool != nullptr) batch.kept.push_back(segment.pool);
        }
        done += count;
        const bool last = done == pendingSamples.size();
        recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        struct {
            VkDeviceAddress counter;
            VkDeviceAddress target;
            std::uint32_t count;
            std::uint32_t store;
        } parameters{sampleCounter->DeviceAddress(), target, count, last && target != 0 ? 1u : 0u};
        static_assert(sizeof(parameters) == SampleCounterPushBytes);
        stream.DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
        if (parameters.store == 1u) {
            // The counts land before the ready bit (stores 3 then 2, each behind a barrier).
            const auto publish = [&](std::uint32_t store) {
                recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
                parameters.store = store;
                stream.DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
            };
            publish(3u);
            publish(2u);
        }
    } while (done < pendingSamples.size());
    recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_HOST_READ_BIT);
    CountBarriers(CommandClass::Copy, target != 0 ? 4u : 2u);
    pendingSamples.clear();
}

bool Recorder::DumpSamples(VkDeviceAddress target, std::uint64_t address) {
    GuestMemory::AssertGpuLockHeld("Recorder::DumpSamples");
    if (target == 0 || !gpuSampleCounter()) return false;
    countingSamples = true;
    // macOS: gpuSampleCounter() succeeding is segment mode, the only one there is.
    ensureOpen();
    if (address != 0 && LabelRunsPerBatch() && pendingSamples.size() < SegmentFoldCapacity && open->queuedDumps < DumpFoldCapacity) {
        if (open->segmentActive) endSegment(*open);
        if (open->renderPass.open) dumpsQueuedInPass.fetch_add(1, std::memory_order_relaxed);
        if (open->run.queued.empty()) storeRuns.fetch_add(1, std::memory_order_relaxed);
        auto& queued = open->run.queued.emplace_back();
        queued.address = address;
        queued.dumpTarget = target;
        queued.dumpSegments = static_cast<std::uint32_t>(pendingSamples.size());
        ++open->queuedDumps;
        dumpsQueued.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    FlushStores();
    Commands();
    foldSamples(*open, target);
    dumpsAtOnce.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool Recorder::RecordMeshArguments(VkCommandBuffer commands, VkDeviceAddress record, VkDeviceAddress arguments, std::span<const std::uint32_t, 7> rules) {
    constexpr std::uint32_t PushBytes = 48;
    MetalPipeline pipeline = nil;
    if (meshArgumentState >= 0) {
        try {
            pipeline = Metal::HelperPipeline(metal->device, Metal::HelperKernel::MeshArguments);
            meshArgumentState = 1;
        } catch (const std::exception& error) {
            meshArgumentState = -1;
            std::fprintf(stderr, "[gpu] mesh indirect draws read their records on the CPU: %s\n", error.what());
        }
    }
    if (meshArgumentState < 0) return false;
    // macOS: a dispatch ends a render pass, so the draw records this before it begins its pass.
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    struct {
        VkDeviceAddress record;
        VkDeviceAddress arguments;
        std::array<std::uint32_t, 8> rules;
    } parameters{record, arguments, {rules[0], rules[1], rules[2], rules[3], rules[4], rules[5], rules[6], 0u}};
    static_assert(sizeof(parameters) == PushBytes);
    Metal::Stream(commands).DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    CountBarriers(CommandClass::Draw, 2);
    return true;
}

// e54d2f35fe: an indirect dispatch's group counts checked against the device limit on the GPU
// (DispatchArguments in RecorderKernels.h); the caller dispatches from `arguments` (12 bytes). False:
// the kernel did not build, and the caller reads the counts from the import unchecked, as before.
// 612b41e4f3: with `threadGroup` the source holds thread counts; the kernel turns them into group
// counts and keeps the counts at `arguments` + 12 (24 bytes), which the caller then copies (a blit,
// hence the transfer reader below) into the ShaderData thread limit.
bool Recorder::RecordDispatchArguments(VkCommandBuffer commands, VkDeviceAddress source, VkDeviceAddress arguments, const std::array<std::uint32_t, 3>* threadGroup) {
    constexpr std::uint32_t PushBytes = 48;
    MetalPipeline pipeline = nil;
    if (dispatchArgumentState >= 0) {
        try {
            pipeline = Metal::HelperPipeline(metal->device, Metal::HelperKernel::DispatchArguments);
            dispatchArgumentState = 1;
        } catch (const std::exception& error) {
            dispatchArgumentState = -1;
            std::fprintf(stderr, "[gpu] indirect dispatches read their group counts unchecked: %s\n", error.what());
        }
    }
    if (dispatchArgumentState < 0) return false;
    recordBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    const auto* limit = context.limits.maxComputeWorkGroupCount;
    struct {
        VkDeviceAddress source;
        VkDeviceAddress arguments;
        std::array<std::uint32_t, 4> limits;
        std::array<std::uint32_t, 4> group;
    } parameters{source, arguments, {limit[0], limit[1], limit[2], threadGroup != nullptr ? 1u : 0u}, {1u, 1u, 1u, 0u}};
    if (threadGroup != nullptr) {
        Require(std::none_of(threadGroup->begin(), threadGroup->end(), [](std::uint32_t size) { return size == 0; }), "Metal dispatch: a thread-dimension indirect dispatch has an empty thread group");
        std::copy(threadGroup->begin(), threadGroup->end(), parameters.group.begin());
    }
    static_assert(sizeof(parameters) == PushBytes);
    Metal::Stream(commands).DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    const VkPipelineStageFlags readers = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | (threadGroup != nullptr ? VK_PIPELINE_STAGE_TRANSFER_BIT : 0u);
    const VkAccessFlags reads = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | (threadGroup != nullptr ? VK_ACCESS_TRANSFER_READ_BIT : 0u);
    recordBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, readers, VK_ACCESS_SHADER_WRITE_BIT, reads);
    CountBarriers(CommandClass::DispatchLeading, 2);
    return true;
}

void Recorder::beginSamples(Batch&) {
    // macOS: occlusion counting runs only through the fold kernels (segment mode); without them
    // there is no whole-batch query to fall back to.
    Metal::Fail("occlusion counting needs the sample counter kernels, which did not build (see the [gpu] occlusion counter line)");
}

void Recorder::readSamples(Batch&) {
}

void Recorder::readGpuTiming(Batch& batch) {
    if (batch.metal == nullptr || batch.metal->timestamps == nil || batch.timedKeys.empty() || !batch.submitted) return;
    std::vector<std::uint64_t> stamps(batch.timedKeys.size() * 2);
    // macOS: the counter heap resolved on the CPU (the batch finished, so every stamp landed).
    NSData* resolved = [batch.metal->timestamps resolveCounterRange:NSMakeRange(0, stamps.size())];
    if (resolved == nil || resolved.length < stamps.size() * sizeof(MTL4TimestampHeapEntry)) return;
    for (std::size_t i = 0; i < stamps.size(); ++i) stamps[i] = static_cast<const MTL4TimestampHeapEntry*>(resolved.bytes)[i].timestamp;
    const auto period = metal->nanosecondsPerTick;
    if (!GpuTimingEnabled()) {
        if (batch.batchTiming == 0 && Metal::TimestampPairValid(stamps[0], stamps[1])) {
            batch.gpuStartNs = static_cast<double>(stamps[0]) * period;
            batch.gpuEndNs = static_cast<double>(stamps[1]) * period;
            PerformanceTrace::GpuRange("batch", batch.gpuStartNs, batch.gpuEndNs, batch.traceId);
        }
        return;
    }
    static auto lastReport = std::chrono::steady_clock::now();
    // The union of the timed ranges (class ranges nest program ranges: a copy's transfer inside
    // its class range) against the batch span gives what no range covers.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
    intervals.reserve(batch.timedKeys.size());
    std::unique_lock lock(timingMutex);
    for (std::size_t i = 0; i < batch.timedKeys.size(); ++i) {
        // A stamp the GPU never wrote (MTLCounterErrorValue) or a pair out of order is left out and
        // counted, instead of an unsigned difference of ~1.8e19 ticks.
        if (!Metal::TimestampPairValid(stamps[i * 2], stamps[i * 2 + 1])) {
            timingInvalid.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const auto ns = static_cast<double>(stamps[i * 2 + 1] - stamps[i * 2]) * period;
        if (i == batch.batchTiming) {
            // The whole batch: apart from the per-program totals it contains.
            batch.gpuStartNs = static_cast<double>(stamps[i * 2]) * period;
            batch.gpuEndNs = static_cast<double>(stamps[i * 2 + 1]) * period;
            PerformanceTrace::GpuRange("batch", batch.gpuStartNs, batch.gpuEndNs, batch.traceId);
            timingBatchMs += ns / 1e6;
            AgcDriver::PerformanceCounters::AddGpuNanoseconds(static_cast<std::uint64_t>(ns));
            continue;
        }
        const auto key = batch.timedKeys[i];
        if (key >= ClassKey(CommandClass::DispatchLeading) && key < ClassKey(CommandClass::Count)) PerformanceTrace::GpuRange(CommandClassNames[key - ClassKey(CommandClass::DispatchLeading)], static_cast<double>(stamps[i * 2]) * period, static_cast<double>(stamps[i * 2 + 1]) * period, batch.traceId);
        auto& totals = timingByKey[key];
        ++totals.count;
        totals.ms += ns / 1e6;
        totals.bytes += batch.timedBytes[i];
        (key >= ClassKey(CommandClass::DispatchLeading) && key < ClassKey(CommandClass::Count) ? timingClassMs : timingProgramMs) += ns / 1e6;
        intervals.emplace_back(stamps[i * 2], stamps[i * 2 + 1]);
    }
    std::sort(intervals.begin(), intervals.end());
    std::uint64_t coveredTicks = 0, unionEnd = 0;
    for (const auto& [begin, end] : intervals) {
        const auto from = std::max(begin, unionEnd);
        if (end > from) coveredTicks += end - from;
        unionEnd = std::max(unionEnd, end);
    }
    timingUnionMs += static_cast<double>(coveredTicks) * period / 1e6;
    ++timingBatches;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    const auto presents = timingPresents.exchange(0, std::memory_order_relaxed);
    const double perPresent = presents != 0 ? 1.0 / static_cast<double>(presents) : 0.0;
    std::vector<std::pair<std::uint64_t, TimingTotals>> hot;
    std::string classes;
    for (const auto& [key, totals] : timingByKey) {
        if (key >= ClassKey(CommandClass::DispatchLeading) && key < ClassKey(CommandClass::Count)) {
            char text[160];
            std::snprintf(text, sizeof(text), " %s x%llu %.1fms %.1fMiB (per present x%.1f %.2fms %.2fMiB)", CommandClassNames[key - ClassKey(CommandClass::DispatchLeading)], static_cast<unsigned long long>(totals.count), totals.ms, totals.bytes / 1048576.0, static_cast<double>(totals.count) * perPresent, totals.ms * perPresent, totals.bytes / 1048576.0 * perPresent);
            classes += text;
        } else {
            hot.emplace_back(key, totals);
        }
    }
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    // The first field stays the program sum: the measure scripts match on it.
    AgcDriver::ProfilePrint_nid_no_patch("[gputime] %.0f ms of GPU time in %llu batches over 10 s (batch %.0f ms first to last command; classes %.0f ms, all ranges %.0f ms, untimed %.0f ms outside every range; %llu presents; %llu ranges dropped at the %u cap, %llu with an unwritten or reversed stamp); by program:", timingProgramMs, static_cast<unsigned long long>(timingBatches), timingBatchMs, timingClassMs, timingUnionMs, timingBatchMs - timingUnionMs, static_cast<unsigned long long>(presents), static_cast<unsigned long long>(timingDropped.exchange(0, std::memory_order_relaxed)), MaxTimedRanges, static_cast<unsigned long long>(timingInvalid.exchange(0, std::memory_order_relaxed)));
    for (std::size_t i = 0; i < hot.size() && i < 12; ++i) AgcDriver::ProfilePrint_nid_no_patch(" 0x%llx x%llu %.0fms", static_cast<unsigned long long>(hot[i].first), static_cast<unsigned long long>(hot[i].second.count), hot[i].second.ms);
    AgcDriver::ProfilePrint_nid_no_patch("; by class:%s\n", classes.c_str());
    timingByKey.clear();
    timingProgramMs = timingClassMs = timingUnionMs = timingBatchMs = 0;
    timingBatches = 0;
    lock.unlock();
    reportBarriers();
    // D80: progress events and the waits they served, per 10 s.
    static ::AgcDriver::Progress::Counters lastProgress{};
    const auto progress = ::AgcDriver::Progress::Read();
    AgcDriver::ProfilePrint_nid_no_patch("[progress] %llu events, %llu waits over 10 s: %llu woken by an event, %llu timed out\n",
        static_cast<unsigned long long>(progress.notifies - lastProgress.notifies), static_cast<unsigned long long>(progress.waits - lastProgress.waits),
        static_cast<unsigned long long>(progress.wokenByEvent - lastProgress.wokenByEvent), static_cast<unsigned long long>(progress.timedOut - lastProgress.timedOut));
    lastProgress = progress;
}

bool Recorder::FlipReadCheck() {
    static const bool enabled = std::getenv("APS5_FLIP_READ_CHECK") != nullptr;
    return enabled;
}

std::vector<Recorder::Completed> Recorder::CompletedBatches(std::uint64_t afterSerial, std::uint64_t throughSerial, std::size_t& missing) const {
    std::vector<Completed> found;
    missing = 0;
    if (throughSerial <= afterSerial) return found;
    std::lock_guard lock(completedMutex);
    for (auto serial = afterSerial + 1; serial <= throughSerial; ++serial) {
        const auto& entry = completed[serial % completed.size()];
        if (entry.serial == serial) found.push_back(entry);
        else ++missing;
    }
    return found;
}

std::uint64_t Recorder::NewestSubmitted(std::chrono::steady_clock::time_point* submittedAt) const {
    std::lock_guard lock(completedMutex);
    if (submittedAt != nullptr) *submittedAt = newestSubmittedAt;
    return newestSubmitted;
}

std::size_t Recorder::UnsignaledBatches() const {
    return static_cast<std::size_t>(std::count_if(inFlight.begin(), inFlight.end(), [&](const auto& batch) { return !signaled(*batch); }));
}

void Recorder::Keep(std::shared_ptr<void> object, std::size_t bytes) {
    ensureOpen();
    open->kept.push_back(std::move(object));
    open->keptBytes += bytes;
}

void Recorder::KeepThroughHold(std::shared_ptr<void> object, std::size_t bytes) {
    GuestMemory::AssertGpuLockHeld("Recorder::KeepThroughHold");
    if (const auto hold = GuestMemory::GpuHoldSerial(); hold != holdKeptSerial) {
        holdKept.clear();
        holdKeptSerial = hold;
    }
    Keep(object, bytes);
    holdKept.push_back(std::move(object));
}

void Recorder::dropStaleHoldKept() {
    if (!holdKept.empty() && holdKeptSerial != GuestMemory::GpuHoldSerial() && GuestMemory::GpuMutex().HeldByThisThread()) holdKept.clear();
}

void Recorder::KeepBytes(const void* owner, std::size_t bytes) {
    if (owner == nullptr || bytes == 0) return;
    ensureOpen();
    if (open->keptOwners.insert(owner).second) open->keptBytes += bytes;
}

void Recorder::BoundKeptBytes() {
    if (!GuestMemory::GpuMutex().HeldByThisThread()) return;
    if (open != nullptr && open->keptBytes >= KeptBytesBudget) Submit();
    if (DeferredBytes() != 0 && inFlightKeptBytes + DeferredBytes() > 2 * KeptBytesBudget) DestroyDeferred(TakeDeferred(), false);
    while (!inFlight.empty() && inFlightKeptBytes > 2 * KeptBytesBudget) {
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), true, 4, true);
    }
}

namespace {
std::size_t SnapshotPool(Recorder::SnapshotUse use) {
    return use == Recorder::SnapshotUse::Storage ? 0 : 1;
}
}

void Recorder::eraseDrawSnapshot(std::map<DrawSnapshotKey, DrawSnapshot>::iterator entry) {
    auto& pool = drawSnapshotPools[SnapshotPool(std::get<1>(entry->first))];
    pool.bytes -= std::get<2>(entry->first);
    pool.recency.erase(entry->second.recent);
    drawSnapshots.erase(entry);
}

// 3518eb8ded: small draw snapshots are copied into slices of one shared-storage chunk per recorder
// instead of a buffer each; a full chunk is replaced and lives on through the slices that hold it.
Recorder::UploadSlice Recorder::DrawUpload(std::size_t bytes) {
    const VkDeviceSize alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 16);
    auto offset = (drawUploadUsed + alignment - 1) & ~(alignment - 1);
    if (drawUpload == nullptr || offset + bytes > drawUpload->Bytes().size()) {
        drawUpload = std::make_shared<Buffer>(context, std::max(DrawUploadChunk, bytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        offset = 0;
    }
    drawUploadUsed = offset + bytes;
    return {drawUpload, offset, drawUpload->Bytes().data() + offset};
}

void Recorder::retireUnwrittenSnapshots(const std::array<std::uint64_t, 3>& stamp) {
    if (stamp == unwrittenStamp) return;
    unwrittenSnapshots.clear();
    unwrittenBytes = 0;
    unwrittenStamp = stamp;
}

std::shared_ptr<Buffer> Recorder::UnwrittenDrawSnapshot(std::uint64_t address, std::size_t bytes, const std::array<std::uint64_t, 3>& stamp, VkDeviceSize* offset) {
    retireUnwrittenSnapshots(stamp);
    const auto found = unwrittenSnapshots.find(address);
    if (found == unwrittenSnapshots.end() || found->second.bytes != bytes) return {};
    if (offset != nullptr) *offset = found->second.offset;
    return found->second.buffer;
}

void Recorder::NoteUnwrittenDrawSnapshot(std::uint64_t address, std::size_t bytes, const std::array<std::uint64_t, 3>& stamp, std::shared_ptr<Buffer> buffer, VkDeviceSize offset) {
    retireUnwrittenSnapshots(stamp);
    if (const auto found = unwrittenSnapshots.find(address); found != unwrittenSnapshots.end()) {
        unwrittenBytes -= found->second.bytes;
        unwrittenSnapshots.erase(found);
    }
    if (unwrittenSnapshots.size() >= UnwrittenSnapshotEntries || bytes > DrawSnapshotBudget - unwrittenBytes) return;
    unwrittenSnapshots.emplace(address, UnwrittenSnapshot{bytes, std::move(buffer), offset});
    unwrittenBytes += bytes;
}

std::shared_ptr<Buffer> Recorder::ReusableDrawSnapshot(std::uint64_t address, std::size_t bytes, SnapshotUse use, std::uint32_t* derived, std::uint64_t generation) {
    auto found = use == SnapshotUse::Vertex ? drawSnapshots.lower_bound({address, use, bytes}) : drawSnapshots.find({address, use, bytes});
    if (found == drawSnapshots.end() || std::get<0>(found->first) != address || std::get<1>(found->first) != use) return {};
    const bool sameRegistry = found->second.registryGeneration == GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (!sameRegistry || !GuestMemory::UnchangedSince(address, bytes, found->second.generation)) {
        if (!sameRegistry || generation == 0 || use != SnapshotUse::Storage || !refreshDrawSnapshot(found->second, address, bytes)) {
            eraseDrawSnapshot(found);
            return {};
        }
        found->second.generation = generation;
    }
    auto& recency = drawSnapshotPools[SnapshotPool(use)].recency;
    recency.splice(recency.end(), recency, found->second.recent);
    if (derived != nullptr) *derived = found->second.derived;
    return found->second.buffer;
}

bool Recorder::refreshDrawSnapshot(DrawSnapshot& entry, std::uint64_t address, std::size_t bytes) {
    const auto held = entry.buffer->Bytes();
    if (held.size() != bytes || bytes == 0) return false;
    constexpr std::uint64_t block = 65536;
    const auto aligned = address / block * block;
    const auto end = address + bytes;
    const auto count = static_cast<std::size_t>((end - 1) / block - address / block + 1);
    struct Scratch {
        std::vector<std::uint64_t> generations;
        std::vector<std::uint8_t> changed;
        std::vector<std::pair<std::size_t, std::size_t>> differing;
        std::vector<CompareSpan> spans;
        std::vector<std::uint8_t> equal;
    };
    auto& scratch = HostThreadLocal<Scratch, Scratch>();
    auto& generations = scratch.generations;
    auto& changed = scratch.changed;
    auto& differing = scratch.differing;
    auto& spans = scratch.spans;
    auto& equal = scratch.equal;
    generations.assign(count, entry.generation);
    changed.assign(count, GuestMemory::BlockWritten);
    differing.clear();
    spans.clear();
    const bool sole = entry.buffer.use_count() == 1;
    GuestMemory::ChangedBlocks(address, bytes, generations, changed);
    for (std::size_t k = 0; k < count; ++k) {
        if (changed[k] == GuestMemory::BlockUnchanged) continue;
        const auto from = std::max(address, aligned + k * block);
        const auto to = std::min(end, aligned + (k + 1) * block);
        const auto offset = static_cast<std::size_t>(from - address);
        spans.push_back({held.data() + offset, reinterpret_cast<const void*>(GuestArena::HostRange(from, static_cast<std::size_t>(to - from))), static_cast<std::size_t>(to - from)}); // alias if displaced (D31, D33)
    }
    equal.assign(spans.size(), 0);
    CompareSpans(spans, equal);
    for (std::size_t k = 0; k < spans.size(); ++k) {
        if (equal[k] != 0) continue;
        const auto offset = static_cast<std::size_t>(static_cast<const std::byte*>(spans[k].first) - held.data());
        const auto length = spans[k].bytes;
        if (!sole) return false;
        if (!differing.empty() && differing.back().first + differing.back().second == offset) differing.back().second += length;
        else differing.emplace_back(offset, length);
    }
    for (const auto& [offset, length] : differing) std::memcpy(held.data() + offset, reinterpret_cast<const void*>(GuestArena::HostRange(address + offset, length)), length);
    return true;
}

void Recorder::KeepDrawSnapshot(std::uint64_t address, std::size_t bytes, std::uint64_t generation, std::uint64_t registryGeneration, std::shared_ptr<Buffer> buffer, SnapshotUse use, std::uint32_t derived) {
    const bool storage = use == SnapshotUse::Storage;
    const auto budget = storage ? DrawSnapshotBudget : DrawInputBudget;
    const auto maxEntries = storage ? DrawSnapshotEntries : DrawInputEntries;
    auto& pool = drawSnapshotPools[SnapshotPool(use)];
    if (generation == 0 || bytes > budget) return;
    if (use == SnapshotUse::Vertex) {
        for (auto it = drawSnapshots.lower_bound({address, use, 0}); it != drawSnapshots.end() && std::get<0>(it->first) == address && std::get<1>(it->first) == use && std::get<2>(it->first) <= bytes;) eraseDrawSnapshot(it++);
    } else if (const auto found = drawSnapshots.find({address, use, bytes}); found != drawSnapshots.end()) {
        eraseDrawSnapshot(found);
    }
    while (!pool.recency.empty() && (pool.bytes + bytes > budget || pool.recency.size() >= maxEntries)) eraseDrawSnapshot(drawSnapshots.find(pool.recency.front()));
    const DrawSnapshotKey key{address, use, bytes};
    pool.recency.push_back(key);
    try {
        drawSnapshots.emplace(key, DrawSnapshot{generation, registryGeneration, std::prev(pool.recency.end()), std::move(buffer), derived});
    } catch (...) {
        pool.recency.pop_back();
        throw;
    }
    pool.bytes += bytes;
}

bool Recorder::KnownHighestIndex(std::uint64_t address, std::size_t bytes, SnapshotUse use, std::optional<std::uint32_t>& highest) {
    const auto found = indexRanges.find({address, use, bytes});
    if (found == indexRanges.end()) return false;
    if (found->second.registryGeneration != GuestAllocations::GuestAllocationsGeneration_nid_postfix() || !GuestMemory::UnchangedSince(address, bytes, found->second.generation)) {
        indexRangeRecency.erase(found->second.recent);
        indexRanges.erase(found);
        return false;
    }
    indexRangeRecency.splice(indexRangeRecency.end(), indexRangeRecency, found->second.recent);
    highest = found->second.highest;
    return true;
}

void Recorder::KeepHighestIndex(std::uint64_t address, std::size_t bytes, SnapshotUse use, std::uint64_t generation, std::uint64_t registryGeneration, std::optional<std::uint32_t> highest) {
    Require(use != SnapshotUse::Storage && use != SnapshotUse::Vertex, "a highest index belongs to an index buffer");
    if (generation == 0) return;
    const DrawSnapshotKey key{address, use, bytes};
    if (const auto found = indexRanges.find(key); found != indexRanges.end()) {
        indexRangeRecency.erase(found->second.recent);
        indexRanges.erase(found);
    }
    while (indexRanges.size() >= IndexRangeEntries) {
        indexRanges.erase(indexRangeRecency.front());
        indexRangeRecency.pop_front();
    }
    indexRangeRecency.push_back(key);
    try {
        indexRanges.emplace(key, IndexRange{generation, registryGeneration, std::prev(indexRangeRecency.end()), highest});
    } catch (...) {
        indexRangeRecency.pop_back();
        throw;
    }
}

void Recorder::OnComplete(std::function<void()> action) {
    ensureOpen();
    open->completions.push_back(std::move(action));
    ++open->writeBackCompletionCount;
    writeBackCompletions.fetch_add(1, std::memory_order_acq_rel);
}

bool Recorder::noteWrite(std::uint64_t address, std::size_t bytes, bool ownLabel, int value, std::int64_t word) {
    CaptureTrace::Log("buffer-write batch=%llu address=%llx bytes=%zu label=%d", static_cast<unsigned long long>(submissions + 1), static_cast<unsigned long long>(address), bytes, ownLabel);
    if (bytes == 0) return false;
    ensureOpen();
    const auto end = address + bytes;
    open->writes.emplace_back(address, end);
    open->writeNotes.push_back(++writeNoteCount);
    // 5234ab2f4a: the recorder's own last published union stays a superset of its pending writes
    // while every new range falls inside it; PendingWriteOverlaps rules out misses against it.
    if (published != nullptr && !SnapshotCovers(published.get(), address, end)) published = nullptr;
    open->writeValues.push_back(static_cast<std::int16_t>(value));
    open->writeWords.push_back(word);
    if (!ownLabel) markOverwritten(address, end);
    // A poller waiting on this range learns that the open batch may now hold its producer.
    if (activeRecorder == this) {
        writeGeneration.fetch_add(1, std::memory_order_release);
        ::AgcDriver::Progress::Notify();
    }
    // Consecutive dispatches write the same output buffers: a range the snapshot already contains
    // leaves the published union unchanged, so the O(N log N) rebuild is skipped.
    if (SnapshotCovers(address, end)) {
        ++snapshotCovered;
        return false;
    }
    return true;
}

void Recorder::noteWriteOn(Batch& batch, std::uint64_t address, std::size_t bytes, bool ownLabel) {
    if (bytes == 0) return;
    if (&batch == open.get()) {
        if (!noteWrite(address, bytes, ownLabel)) return;
        const std::array ranges{std::pair{address, address + bytes}};
        publishPendingWrites(ranges);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return;
    }
    // An in-flight batch: its range joins the snapshot at once (the completion that stores it runs
    // when the batch finishes, and the hook must sync for a CPU read until then). The generation
    // moves as well, so a poller re-consults the label table for a completion label.
    batch.writes.emplace_back(address, address + bytes);
    batch.writeNotes.push_back(++writeNoteCount);
    if (published != nullptr && !SnapshotCovers(published.get(), address, address + bytes)) published = nullptr;
    batch.writeValues.push_back(-1);
    batch.writeWords.push_back(-1);
    if (!ownLabel) markOverwritten(address, address + bytes);
    if (activeRecorder == this) {
        writeGeneration.fetch_add(1, std::memory_order_release);
        ::AgcDriver::Progress::Notify();
    }
    if (!SnapshotCovers(address, address + bytes)) {
        const std::array ranges{std::pair{address, address + bytes}};
        publishPendingWrites(ranges);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Recorder::markOverwritten(std::uint64_t address, std::uint64_t end) {
    if (recordedLabels.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard tableLock(labelTableMutex);
    // A dword overlaps the range when it starts before the end and ends after the start.
    for (auto it = labels.lower_bound(address >= 3 ? address - 3 : 0); it != labels.end() && it->first < end; ++it) it->second.overwritten = true;
}

void Recorder::NotePendingWrite(std::uint64_t address, std::size_t bytes) {
    if (!noteWrite(address, bytes)) return;
    const std::array ranges{std::pair{address, address + bytes}};
    publishPendingWrites(ranges);
    // The note precedes this thread's vkQueueSubmit and the label another queue polls for; the
    // fence makes that order hold without relying on x86 store ordering.
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Recorder::NotePendingFill(std::uint64_t address, std::size_t bytes, std::uint8_t value) {
    if (!noteWrite(address, bytes, false, value, value * std::int64_t{0x01010101})) return;
    const std::array ranges{std::pair{address, address + bytes}};
    publishPendingWrites(ranges);
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Recorder::NotePendingWordFill(std::uint64_t address, std::size_t bytes, std::uint32_t word) {
    const int value = word == (word & 0xffu) * 0x01010101u ? static_cast<int>(word & 0xffu) : -1;
    if (!noteWrite(address, bytes, false, value, word)) return;
    const std::array ranges{std::pair{address, address + bytes}};
    publishPendingWrites(ranges);
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Recorder::NotePendingWrites(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    bool publish = false;
    for (const auto& [begin, end] : ranges) {
        if (end > begin && noteWrite(begin, static_cast<std::size_t>(end - begin))) publish = true;
    }
    if (!publish) return;
    publishPendingWrites(ranges);
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void Recorder::publishPendingWrites(std::span<const std::pair<std::uint64_t, std::uint64_t>> additions) const {
    if (activeRecorder != this) {
        publishedWriteRangesValid = false;
        return;
    }
    const auto started = DrawProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto previous = pendingWrites.load(std::memory_order_acquire);
    const auto owned = publishedWriteRanges.lock();
    const bool extend = !additions.empty() && previous != nullptr && previous == owned && publishedWriteRangesValid &&
        std::all_of(additions.begin(), additions.end(), [](const auto& range) { return range.second > range.first; });
    publishedWriteRangesValid = false;
    auto merged = std::make_shared<WriteRanges>();
    if (extend && additions.size() == 1) {
        auto range = additions.front();
        const auto first = std::partition_point(previous->begin(), previous->end(), [&](const auto& entry) { return entry.second < range.first; });
        auto last = first;
        while (last != previous->end() && last->first <= range.second) {
            range.first = std::min(range.first, last->first);
            range.second = std::max(range.second, last->second);
            ++last;
        }
        merged->reserve(previous->size() - static_cast<std::size_t>(last - first) + 1);
        merged->insert(merged->end(), previous->begin(), first);
        merged->push_back(range);
        merged->insert(merged->end(), last, previous->end());
    } else {
        if (extend) {
            merged->reserve(previous->size() + additions.size());
            merged->insert(merged->end(), previous->begin(), previous->end());
            merged->insert(merged->end(), additions.begin(), additions.end());
        } else {
            if (open != nullptr) merged->insert(merged->end(), open->writes.begin(), open->writes.end());
            for (const auto& batch : inFlight) merged->insert(merged->end(), batch->writes.begin(), batch->writes.end());
            for (const auto* batch : finishing) merged->insert(merged->end(), batch->writes.begin(), batch->writes.end());
        }
        std::sort(merged->begin(), merged->end());
        std::size_t out = 0;
        for (const auto& [begin, end] : *merged) {
            if (out != 0 && begin <= (*merged)[out - 1].second) (*merged)[out - 1].second = std::max((*merged)[out - 1].second, end);
            else (*merged)[out++] = {begin, end};
        }
        merged->resize(out);
    }
    publishedWriteRangesValid = std::all_of(merged->begin(), merged->end(), [](const auto& range) { return range.second > range.first; });
    publishedWriteRanges = merged;
    published = merged;
    pendingWrites.store(std::move(merged), std::memory_order_release);
    publishGeneration.fetch_add(1, std::memory_order_release);
    ++snapshotRebuilds;
    if (DrawProfiled()) snapshotRebuildMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

bool Recorder::overlaps(const Batch& batch, std::uint64_t address, std::uint64_t end) {
    for (const auto& [begin, finish] : batch.writes) {
        if (address < finish && begin < end) return true;
    }
    return false;
}

bool Recorder::PendingWriteOverlaps(std::uint64_t address, std::size_t bytes) const {
    if (bytes == 0) return false;
    if (published != nullptr && !SnapshotOverlaps(published.get(), address, bytes)) return false;
    const auto end = address + bytes;
    if (open != nullptr && overlaps(*open, address, end)) return true;
    for (const auto& batch : inFlight) {
        if (overlaps(*batch, address, end)) return true;
    }
    return false;
}

bool Recorder::CompletionWriteOverlaps(std::uint64_t address, std::size_t bytes) const {
    if (bytes == 0) return false;
    const auto end = address + bytes;
    if (open != nullptr && open->writeBackCompletionCount != 0 && overlaps(*open, address, end)) return true;
    for (const auto& batch : inFlight) {
        if (batch->writeBackCompletionCount != 0 && overlaps(*batch, address, end)) return true;
    }
    for (const auto* batch : finishing) {
        if (batch->writeBackCompletionCount != 0 && overlaps(*batch, address, end)) return true;
    }
    return false;
}

bool Recorder::OpenWriteOverlaps(std::uint64_t address, std::size_t bytes) const {
    return bytes != 0 && open != nullptr && overlaps(*open, address, address + bytes);
}

// 9fffceaa9a: the newest unfinished batch that writes the range or holds it in an address-based
// lease (its stores are marked only when its completion runs); 0 when none does.
std::uint64_t Recorder::PendingWriterSerial(std::uint64_t address, std::size_t bytes) const {
    if (bytes == 0) return 0;
    const auto end = address + bytes;
    const auto writes = [&](const Batch& batch) { return overlaps(batch, address, end) || readOverlap(batch, address, end, ReadKind::AddressBased).has_value(); };
    if (open != nullptr && writes(*open)) return OpenSerial();
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it) {
        if (writes(**it)) return (*it)->serial;
    }
    return 0;
}

bool Recorder::signaled(const Batch& batch) const {
    // macOS: the batch's fence is its serial on the recorder's timeline event.
    return batch.submitted && metal->timeline.signaledValue >= batch.serial;
}

bool Recorder::PendingWriteSettled(std::uint64_t address, std::size_t bytes) const {
    if (bytes == 0) return true;
    const auto end = address + bytes;
    if (open != nullptr && overlaps(*open, address, end)) return false;
    for (const auto& batch : inFlight) {
        if (overlaps(*batch, address, end) && !signaled(*batch)) return false;
    }
    return true;
}

std::uint64_t Recorder::LastWriteNote(std::uint64_t address, std::size_t bytes) const {
    if (open == nullptr || open->writes.empty() || open->writeNotes.size() != open->writes.size()) return 0;
    if (open->writes.back() != std::pair<std::uint64_t, std::uint64_t>{address, address + bytes}) return 0;
    return open->writeNotes.back() == writeNoteCount ? writeNoteCount : 0;
}

std::uint64_t Recorder::NewestWriteNote(std::uint64_t address, std::size_t bytes) const {
    if (bytes == 0) return 0;
    const auto end = address + bytes;
    std::uint64_t newest = 0;
    const auto scan = [&](const Batch& batch) {
        for (std::size_t i = 0; i < batch.writes.size(); ++i) {
            if (address < batch.writes[i].second && batch.writes[i].first < end) newest = std::max(newest, batch.writeNotes[i]);
        }
    };
    if (open != nullptr) scan(*open);
    for (const auto& batch : inFlight) scan(*batch);
    return newest;
}

std::vector<Recorder::PendingWrite> Recorder::PendingWritesOver(std::uint64_t address, std::size_t bytes) const {
    std::vector<PendingWrite> found;
    if (bytes == 0) return found;
    const auto end = address + bytes;
    const auto scan = [&](const Batch& batch) {
        for (std::size_t i = 0; i < batch.writes.size(); ++i) {
            if (address < batch.writes[i].second && batch.writes[i].first < end) found.push_back({batch.writes[i].first, batch.writes[i].second, batch.writeNotes[i], batch.writeValues[i], batch.writeWords[i]});
        }
    };
    if (open != nullptr) scan(*open);
    for (const auto& batch : inFlight) scan(*batch);
    return found;
}

void Recorder::PendingWriteRanges(std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) const {
    ranges.clear();
    if (open != nullptr) ranges.insert(ranges.end(), open->writes.begin(), open->writes.end());
    for (const auto& batch : inFlight) ranges.insert(ranges.end(), batch->writes.begin(), batch->writes.end());
}

bool Recorder::ReadTracking() {
    return ReadTrackingEnabled();
}

void Recorder::NotePendingRead(std::uint64_t address, std::size_t bytes, ReadKind kind) {
    CaptureTrace::Log("buffer-read batch=%llu address=%llx bytes=%zu kind=%d", static_cast<unsigned long long>(submissions + 1), static_cast<unsigned long long>(address), bytes, static_cast<int>(kind));
    if (bytes == 0 || !ReadTrackingEnabled()) return;
    ensureOpen();
    open->reads.push_back({address, address + bytes, kind});
    readsNoted.fetch_add(1, std::memory_order_relaxed);
}

void Recorder::NotePendingReads(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, ReadKind kind) {
    if (ranges.empty() || !ReadTrackingEnabled()) return;
    ensureOpen();
    for (const auto& [begin, end] : ranges) {
        CaptureTrace::Log("buffer-read batch=%llu address=%llx bytes=%llu kind=%d", static_cast<unsigned long long>(submissions + 1), static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin), static_cast<int>(kind));
        if (end > begin) open->reads.push_back({begin, end, kind});
    }
    readsNoted.fetch_add(ranges.size(), std::memory_order_relaxed);
}

bool Recorder::ValidReadList(const ReadList& ranges) {
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        if (ranges[index].second <= ranges[index].first) return false;
        if (index != 0 && ranges[index].first < ranges[index - 1].second) return false;
    }
    return true;
}

void Recorder::NotePendingReadList(std::shared_ptr<const ReadList> ranges, ReadKind kind) {
    if (ranges == nullptr || ranges->empty() || !ReadTrackingEnabled()) return;
    if (CaptureTrace::Enabled() || FlipReadCheck()) {
        NotePendingReads(*ranges, kind);
        return;
    }
    ensureOpen();
    readsNoted.fetch_add(ranges->size(), std::memory_order_relaxed);
    for (const auto& note : open->readLists) {
        if (note.ranges == ranges && note.kind == kind) return;
    }
    open->readLists.push_back({std::move(ranges), open->reads.size(), kind});
}

std::optional<Recorder::ReadKind> Recorder::readOverlap(const Batch& batch, std::uint64_t address, std::uint64_t end, std::optional<ReadKind> only) {
    const auto listOverlaps = [&](const Batch::ReadListNote& note) {
        if (only.has_value() && note.kind != *only) return false;
        const auto& list = *note.ranges;
        const auto found = std::partition_point(list.begin(), list.end(), [&](const auto& range) { return range.second <= address; });
        return found != list.end() && found->first < end;
    };
    std::size_t next = 0;
    for (std::size_t index = 0; index < batch.reads.size(); ++index) {
        for (; next < batch.readLists.size() && batch.readLists[next].position == index; ++next) {
            if (listOverlaps(batch.readLists[next])) return batch.readLists[next].kind;
        }
        const auto& read = batch.reads[index];
        if ((!only.has_value() || read.kind == *only) && address < read.end && read.begin < end) return read.kind;
    }
    for (; next < batch.readLists.size(); ++next) {
        if (listOverlaps(batch.readLists[next])) return batch.readLists[next].kind;
    }
    return std::nullopt;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> Recorder::notedReads(const Batch& batch) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
    std::size_t next = 0;
    const auto expand = [&](std::size_t position) {
        for (; next < batch.readLists.size() && batch.readLists[next].position == position; ++next) result.insert(result.end(), batch.readLists[next].ranges->begin(), batch.readLists[next].ranges->end());
    };
    for (std::size_t index = 0; index < batch.reads.size(); ++index) {
        expand(index);
        result.emplace_back(batch.reads[index].begin, batch.reads[index].end);
    }
    expand(batch.reads.size());
    return result;
}

bool Recorder::PendingReadOverlaps(std::uint64_t address, std::size_t bytes, bool ignoreSignaled) const {
    if (bytes == 0) return false;
    readQueries.fetch_add(1, std::memory_order_relaxed);
    const auto end = address + bytes;
    if (open != nullptr) {
        if (const auto kind = readOverlap(*open, address, end)) {
            readHits[static_cast<std::size_t>(*kind)].fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    // Newest first: the batch most likely still running decides without a status query per batch.
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it) {
        const auto kind = readOverlap(**it, address, end);
        if (!kind.has_value()) continue;
        if (ignoreSignaled && signaled(**it)) {
            readStaleIgnored.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        readHits[static_cast<std::size_t>(*kind)].fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

std::optional<Recorder::PendingReadInfo> Recorder::DescribePendingRead(std::uint64_t address, std::size_t bytes) const {
    if (bytes == 0) return std::nullopt;
    const auto end = address + bytes;
    if (open != nullptr) {
        if (const auto kind = readOverlap(*open, address, end)) return PendingReadInfo{submissions + 1, open->queue, *kind, true, false};
    }
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it) {
        if (const auto kind = readOverlap(**it, address, end)) return PendingReadInfo{(*it)->serial, (*it)->queue, *kind, false, signaled(**it)};
    }
    return std::nullopt;
}

Recorder::ReadStatistics Recorder::ReadCounts() {
    ReadStatistics counts{readsNoted.load(std::memory_order_relaxed), readQueries.load(std::memory_order_relaxed), readStaleIgnored.load(std::memory_order_relaxed), {}};
    for (std::size_t kind = 0; kind < ReadKinds; ++kind) counts.hits[kind] = readHits[kind].load(std::memory_order_relaxed);
    return counts;
}

std::optional<Recorder::PendingWriteInfo> Recorder::DescribePendingWrite(std::uint64_t address, std::size_t bytes) const {
    if (bytes == 0) return std::nullopt;
    const auto end = address + bytes;
    // The same choice SyncThrough makes: the open batch forces a full Sync, otherwise the newest
    // overlapping in-flight batch is the target. Of that batch's ranges the first overlapping one
    // is reported (a dispatch notes each written V# once, so it is the range the access hit).
    const auto firstOverlap = [&](const Batch& batch) -> const std::pair<std::uint64_t, std::uint64_t>* {
        for (const auto& range : batch.writes) {
            if (address < range.second && range.first < end) return &range;
        }
        return nullptr;
    };
    // With SyncThrough disabled every hit is a full Sync: the open batch (if any) is submitted and
    // everything in flight finishes, whichever batch noted the range.
    const bool syncAll = !SyncThroughEnabled();
    const auto allBatches = inFlight.size() + (open != nullptr ? 1 : 0);
    if (open != nullptr) {
        if (const auto* range = firstOverlap(*open)) return PendingWriteInfo{submissions + 1, true, false, range->first, range->second, inFlight.size() + 1};
    }
    std::size_t finished = inFlight.size();
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it, --finished) {
        const auto* range = firstOverlap(**it);
        if (range == nullptr) continue;
        // In flight means submitted, so the fence is live; signaled = the GPU already ran it.
        const bool signaled = this->signaled(**it);
        if (syncAll) return PendingWriteInfo{submissions + (open != nullptr ? 1 : 0), open != nullptr, signaled, range->first, range->second, allBatches};
        return PendingWriteInfo{(*it)->serial, false, signaled, range->first, range->second, finished};
    }
    return std::nullopt;
}

bool Recorder::HasCompletions() const {
    if (open != nullptr && !open->completions.empty()) return true;
    for (const auto& batch : inFlight) {
        if (!batch->completions.empty()) return true;
    }
    return false;
}

void Recorder::noteLabelOn(Batch& batch, std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue, bool behindCompletion) {
    // One entry per dword; a label overlapping older entries (a 4-byte store inside an 8-byte one or
    // the reverse) replaces exactly the dwords it stores, so a lookup composes what memory will hold.
    std::lock_guard tableLock(labelTableMutex);
    for (std::size_t offset = 0; offset + 4 <= bytes.size(); offset += 4) {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, 4);
        const auto dword = address + offset;
        labels.insert_or_assign(dword, LabelEntry{value, queue, stamp, &batch, 0, false, behindCompletion});
        batch.labelDwords.push_back(dword);
        LabelGroupDwords().emplace_back(dword, stamp);
    }
    recordedLabels.store(labels.size(), std::memory_order_relaxed);
}

void Recorder::markBehindCompletion(const Batch& batch, std::uint64_t begin, std::uint64_t end) {
    std::lock_guard tableLock(labelTableMutex);
    for (auto it = labels.lower_bound(begin); it != labels.end() && it->first < end; ++it) {
        if (it->second.batch == &batch) it->second.behindCompletion = true;
    }
}

void Recorder::NoteLabel(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue) {
    ensureOpen();
    // The table entry before the write note: the note bumps the write generation a poller
    // watches, and a poller that sees the bump then finds the label without the GPU mutex.
    const bool tabled = bytes.size() <= LabelTableBytes;
    if (tabled) {
        noteLabelOn(*open, address, bytes, stamp, queue);
    } else {
        std::lock_guard tableLock(labelTableMutex);
        labels.erase(labels.lower_bound(address), labels.lower_bound(address + bytes.size()));
        recordedLabels.store(labels.size(), std::memory_order_relaxed);
        open->wideLabels.push_back(wideLabels.emplace(address, address + bytes.size()));
        wideLabelBytes = std::max<std::uint64_t>(wideLabelBytes, bytes.size());
    }
    if (activeRecorder == this && pendingLabelSince.load(std::memory_order_relaxed) == NoPendingLabel) {
        pendingLabelSince.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
    }
    noteWriteOn(*open, address, bytes.size(), true);
}

std::optional<Recorder::LabelHit> Recorder::PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, LabelRefusal* refusal) const {
    // Under the GPU mutex; the table mutex is taken too so this one path serves both lock regimes.
    std::lock_guard tableLock(labelTableMutex);
    return lookupLabel(address, bytes, afterStamp, refusal);
}

std::size_t Recorder::PendingLabels() const {
    std::lock_guard tableLock(labelTableMutex);
    return labels.size();
}

bool Recorder::PendingLabelIn(std::uint64_t address, std::size_t bytes) const {
    std::lock_guard tableLock(labelTableMutex);
    if (bytes == 0) return false;
    const auto end = address + bytes;
    if (const auto first = labels.lower_bound(address); first != labels.end() && first->first < end) return true;
    if (wideLabelInLocked(address, bytes)) return true;
    // A queued label of another queue is unordered against the caller on hardware (nothing that
    // queue recorded could have satisfied a wait yet); it is still reported, since the table mutex
    // is held anyway and a caller deciding a CPU store wants the conservative answer.
    if (const auto first = queuedLabels.lower_bound(address); first != queuedLabels.end() && first->first < end) return true;
    return false;
}

bool Recorder::CompletionLabelIn(std::uint64_t address, std::size_t bytes) const {
    std::lock_guard tableLock(labelTableMutex);
    if (bytes == 0) return false;
    const auto end = address + bytes;
    for (auto it = labels.lower_bound(address); it != labels.end() && it->first < end; ++it) {
        if (it->second.behindCompletion) return true;
    }
    if (const auto first = queuedLabels.lower_bound(address); first != queuedLabels.end() && first->first < end) return true;
    return false;
}

std::optional<Recorder::LabelHit> Recorder::lookupLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, LabelRefusal* refusal) const {
    if (refusal != nullptr) *refusal = LabelRefusal::None;
    if ((labels.empty() && queuedLabels.empty()) || (bytes != 4 && bytes != 8) || address % 4 != 0) return std::nullopt;
    LabelHit hit{0, 0, 0, false, std::numeric_limits<std::uint64_t>::max()};
    bool queued = false;
    bool candidate = false;
    const auto tag = GuestMemory::GpuLockThreadTag();
    const auto refuse = [&](LabelRefusal reason, std::uint64_t LateStatistics::*counter, const LabelEntry& entry) {
        if (refusal != nullptr) *refusal = entry.behindCompletion ? LabelRefusal::BehindCompletion : reason;
        if (counter != nullptr) ++(lateCounts.*counter);
        return std::nullopt;
    };
    for (std::size_t offset = 0; offset < bytes; offset += 4) {
        const auto dword = address + offset;
        const LabelEntry* entry = nullptr;
        // A queued label is only good for its own queue's wait (see NoteQueuedLabel), where it is
        // that queue's newest store of the dword; everyone else sees the recorded entry.
        if (const auto own = queuedLabels.find(dword); own != queuedLabels.end() && own->second.queue == tag) {
            entry = &own->second;
            queued = true;
        } else if (const auto found = labels.find(dword); found != labels.end()) {
            entry = &found->second;
            if (entry->batch == nullptr) {
                if (entry->queue != tag) return std::nullopt;
                queued = true;
            }
        }
        if (entry == nullptr) return std::nullopt;
        if (entry->stamp <= afterStamp) {
            // The late rule (see NoteLabel): counted in both modes, so a control run shows the population.
            if (!candidate) ++lateCounts.candidates;
            candidate = true;
            if (!LateTrustEnabled()) return refuse(LabelRefusal::TrustOff, nullptr, *entry);
            if (entry->batch == nullptr) return refuse(LabelRefusal::Queued, &LateStatistics::queued, *entry);
            if (entry->overwritten) return refuse(LabelRefusal::Overwritten, &LateStatistics::overwritten, *entry);
            if (entry->generation == 0) return refuse(LabelRefusal::Unclosed, &LateStatistics::unclosed, *entry);
            hit.late = true;
            hit.generation = std::min(hit.generation, entry->generation);
        }
        hit.value |= static_cast<std::uint64_t>(entry->value) << (offset * 8u);
        if (offset == 0) {
            hit.queue = entry->queue;
            hit.stamp = entry->stamp;
        }
    }
    if (!hit.late) hit.generation = 0;
    if (queued) queuedLabelHits.fetch_add(1, std::memory_order_relaxed);
    return hit;
}

bool Recorder::unsignaled(std::uint64_t serial) const {
    for (const auto& batch : inFlight) {
        if (batch->serial == serial) return !signaled(*batch);
    }
    return false;
}

void Recorder::AfterCompletions(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue, bool storedOnGpu) {
    Require(open != nullptr || !inFlight.empty(), "no batch to append a completion label to");
    Batch& batch = open != nullptr ? *open : *inFlight.back();
    std::vector<std::byte> copy(bytes.begin(), bytes.end());
    // The store bypasses the flush hook: it runs inside finish() under the mutex, and the hook would
    // find this very range noted (or a later label to it) and sync re-entrantly. Stamped like every
    // driver store so a collect memoized for the packet sees it.
    static const bool always = std::getenv("APS5_LABEL_STORE_ALWAYS") != nullptr;
    const auto sequence = [&] {
        std::lock_guard ringLock(writtenBackMutex);
        return writtenBackSequence;
    }();
    batch.completions.push_back([this, address, copy = std::move(copy), sequence, storedOnGpu] {
        if (storedOnGpu && !always && !writtenBackSince(sequence, address, address + copy.size())) {
            completionStoresSkipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        completionStoresRun.fetch_add(1, std::memory_order_relaxed);
        GuestMemory::CheckRange(reinterpret_cast<const void*>(address), copy.size(), 4, true);
        GuestMemory::StoreOwnBytes(address, copy);
    });
    // Counted per batch and counted down when the batch finishes, on every exit path of finish()
    // (a label whose range became unmapped is reported there, and a lost device throws before the
    // completions run): a label that will never land must not leave the workers reaping forever.
    // One stored on the GPU already needs no reap unless a write-back overlaps it (the store above
    // is skipped otherwise), so it enters the count at that write-back (NoteWrittenBack).
    const bool behindCompletion = !(storedOnGpu && !always && !CountAllCompletionLabels());
    if (!behindCompletion) {
        batch.completionLabelRanges.push_back({address, address + bytes.size(), false});
    } else {
        ++batch.completionLabelCount;
        completionLabels.fetch_add(1, std::memory_order_acq_rel);
    }
    // The table entry before the write note: the note bumps the generation a poller watches, and a
    // poller that sees the bump then finds the label without the GPU mutex.
    noteLabelOn(batch, address, bytes, stamp, queue, behindCompletion);
    noteWriteOn(batch, address, bytes.size(), true);
    if (&batch == open.get() && activeRecorder == this && pendingLabelSince.load(std::memory_order_relaxed) == NoPendingLabel) {
        pendingLabelSince.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
    }
}

bool Recorder::AfterRecordedWork(std::function<void()> action) {
    if (Idle()) return false;
    Batch& batch = open != nullptr ? *open : *inFlight.back();
    batch.completions.push_back(std::move(action));
    ++batch.completionLabelCount;
    completionLabels.fetch_add(1, std::memory_order_acq_rel);
    if (&batch == open.get() && activeRecorder == this && pendingLabelSince.load(std::memory_order_relaxed) == NoPendingLabel) {
        pendingLabelSince.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
    }
    return true;
}

void Recorder::NoteWrittenBack(std::uint64_t address, std::size_t bytes) {
    Recorder* recorder = activeRecorder;
    if (recorder == nullptr || bytes == 0) return;
    GuestMemory::AssertGpuLockHeld("Recorder::NoteWrittenBack");
    const auto end = address + bytes;
    {
        std::lock_guard ringLock(recorder->writtenBackMutex);
        recorder->writtenBack.push_back({++recorder->writtenBackSequence, address, end});
        while (recorder->writtenBack.size() > 16384) recorder->writtenBack.pop_front();
    }
    if (recorder->recordedLabels.load(std::memory_order_relaxed) != 0 && recorder->PendingLabelIn(address, bytes)) writeBacksOverLabels.fetch_add(1, std::memory_order_relaxed);
    // The GPU-stored completion labels this write-back overlaps now need their completion store,
    // so they enter the pending count (once each; finish() subtracts the batch's count). Batches
    // in `finishing` run their own completions in the same finish(), so only open and in-flight
    // ones are scanned.
    const auto count = [&](Batch& batch) {
        for (auto& label : batch.completionLabelRanges) {
            if (label.counted || label.begin >= end || address >= label.end) continue;
            label.counted = true;
            ++batch.completionLabelCount;
            completionLabels.fetch_add(1, std::memory_order_acq_rel);
            completionLabelsCountedLate.fetch_add(1, std::memory_order_relaxed);
            recorder->markBehindCompletion(batch, label.begin, label.end);
        }
    };
    if (recorder->open != nullptr) count(*recorder->open);
    for (auto& batch : recorder->inFlight) count(*batch);
}

bool Recorder::writtenBackSince(std::uint64_t sequence, std::uint64_t begin, std::uint64_t end) {
    std::lock_guard ringLock(writtenBackMutex);
    if (writtenBack.empty() || writtenBack.back()[0] <= sequence) return false;
    if (writtenBack.front()[0] > sequence + 1) return true;
    for (auto it = writtenBack.rbegin(); it != writtenBack.rend() && (*it)[0] > sequence; ++it) {
        if ((*it)[1] < end && begin < (*it)[2]) return true;
    }
    return false;
}

void Recorder::Submit() {
    PerformanceTimer frameTiming("Recorder.Submit");
    // The work count is cleared even when nothing is open: the driver counts a dispatch after its
    // call returns (outside the mutex), so a submit by another thread in between leaves a stale
    // count behind, and the callers that act on it would otherwise take the mutex for nothing at
    // every packet or submission end until a real batch is next submitted.
    if (activeRecorder == this) workSinceSubmit.store(0, std::memory_order_relaxed);
    dropStaleHoldKept();
    if (open == nullptr) return;
    GuestMemory::AssertGpuLockHeld("Recorder::Submit");
    if (!open->keyStores.empty()) recordKeyStores(false);
    if (open->renderPass.open) endOpenRenderPass();
    const bool segments = segmentMode();
    if (!segments) {
        endSamples(*open);
        if (pendingSamples.size() >= SampleFoldCapacity) foldSamples(*open, 0);
    }
    const bool hostReadCovered = (open->run.open || !open->run.queued.empty()) && closeStoreRun(true);
    if (segments && pendingSamples.size() >= SampleFoldCapacity) foldSegments();
    if (BarrierValidate()) validateBatchEnds.fetch_add(1, std::memory_order_relaxed);
    if (open->hostReadOwed && !hostReadCovered) {
        // A fence makes nothing visible to the host: a lean draw's writes (no download barrier of
        // its own) get the batch's one here.
        recordBarrier(CommandsOf(open->metal), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        CountBarriers(CommandClass::Draw);
    }
    EndGpuTiming(open->batchTiming);
    if (!GpuTimingEnabled()) reportBarriers();
    if (FlipReadCheck()) {
        // A CPU write into a noted read after this collect (and so possibly before the GPU read
        // it) fails UnchangedSince at the presenter's check (VulkanDevice RetirePresents).
        for (const auto& [begin, end] : notedReads(*open)) open->readGeneration = std::max(open->readGeneration, GuestMemory::CollectWrites(begin, end - begin));
    }
    auto batch = std::move(open);
    // A batch that fails before its commit is discarded (its labels and completion counts leave
    // the tables) instead of leaking (upstream 72d6c9e16b). Metal's commit reports no device loss
    // (that arrives in the commit feedback), so there is no vkDeviceWaitIdle step here.
    struct DiscardUnsubmitted {
        Recorder& recorder;
        std::unique_ptr<Batch>& batch;
        ~DiscardUnsubmitted() {
            if (batch != nullptr && !batch->submitted) recorder.discard(std::move(batch));
        }
    } discardUnsubmitted{*this, batch};
    MetalCommandBuffer commandBuffer = batch->metal->stream->End();
    if (Metal::failNextSubmit != nullptr && Metal::failSubmitBeforeCommit) throw std::runtime_error(std::exchange(Metal::failNextSubmit, nullptr));
    // The timeline reaches this batch's serial when it completes (see WaitSerial).
    const std::uint64_t serial = submissions + 1;
    CaptureTrace::Log("submit batch=%llu reads=%zu writes=%zu", static_cast<unsigned long long>(serial), batch->reads.size(), batch->writes.size());
    // macOS: the commit's feedback reports a GPU failure (and the batch's GPU times), possibly
    // after the event value landed; the event is signaled right behind the commit, so it reaches
    // the serial once this command buffer (and everything committed before it) completed.
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    std::shared_ptr<BatchFeedback> feedback = batch->metal->feedback;
    std::shared_ptr<FeedbackSink> sink = metal->sink;
    static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
    const auto committedAt = std::chrono::steady_clock::now();
    if (traceGpu) std::fprintf(stderr, "[gpu] recorder commits batch %llu (open %.1f ms)\n", static_cast<unsigned long long>(serial), std::chrono::duration<double, std::milli>(committedAt - batch->openedAt).count());
    [options addFeedbackHandler:^(MetalFeedback commitFeedback) {
        static const bool traceLandingCopies = std::getenv("APS5_TRACE_LABEL_LANDING") != nullptr;
        if (traceLandingCopies) ReportLandingCopies(serial);
        if (traceGpu) std::fprintf(stderr, "[gpu] recorder batch %llu completed on the GPU (%.2f ms; feedback %.1f ms after the commit)%s%s\n", static_cast<unsigned long long>(serial), (commitFeedback.GPUEndTime - commitFeedback.GPUStartTime) * 1e3,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - committedAt).count(), commitFeedback.error != nil ? ": " : "", commitFeedback.error != nil ? Metal::Describe(commitFeedback.error).c_str() : "");
        if (traceGpu && commitFeedback.error != nil) std::fprintf(stderr, "[gpu] recorder batch %llu error domain %s code %ld user info %s\n", static_cast<unsigned long long>(serial), commitFeedback.error.domain.UTF8String, static_cast<long>(commitFeedback.error.code), commitFeedback.error.userInfo.description.UTF8String);
        feedback->gpuStartSeconds = commitFeedback.GPUStartTime;
        feedback->gpuEndSeconds = commitFeedback.GPUEndTime;
        if (commitFeedback.error != nil) {
            std::lock_guard lock(sink->mutex);
            sink->errors.emplace_back(serial, Metal::Describe(commitFeedback.error));
        }
        feedback->delivered.store(true, std::memory_order_release);
    }];
    const auto submitStart = std::chrono::steady_clock::now();
    if (Metal::failNextSubmit != nullptr) throw std::runtime_error(std::exchange(Metal::failNextSubmit, nullptr));
    MetalCommandBuffer commandBuffers[1] = {commandBuffer};
    // Under the residency lock: the commit walks the queue's residency sets, which the release
    // thread changes (MetalResidency.h, D44).
    ::AgcDriver::Metal::Residency::CommitQueue([&] {
        [metal->queue commit:commandBuffers count:1 options:options];
        [metal->queue signalEvent:metal->timeline value:serial];
    });
    // D80: when the timeline reaches this batch, wake the threads waiting for completions or labels instead of
    // letting them find it on their next poll.
    [metal->timeline notifyListener:ProgressListener() atValue:serial block:ProgressBlock()];
    batch->submitted = true;
    batch->serial = ++submissions;
    batch->submittedAt = std::chrono::steady_clock::now();
    if (DrawProfiled()) {
        const auto us = std::chrono::duration<double, std::micro>(batch->submittedAt - submitStart).count();
        ++submitCount;
        submitUs += us;
        submitMaxUs = std::max(submitMaxUs, us);
        if (auto* frame = PerformanceContext::Current()) frame->Add(frame->Get("Recorder", "submit"), batch->submittedAt - submitStart);
    }
    {
        std::lock_guard lock(completedMutex);
        newestSubmitted = batch->serial;
        newestSubmittedAt = batch->submittedAt;
    }
    inFlight.push_back(std::move(batch));
    inFlightKeptBytes += inFlight.back()->keptBytes;
    if (activeRecorder == this) pendingLabelSince.store(NoPendingLabel, std::memory_order_release);
}

std::uint64_t Recorder::SubmitAndEpoch() {
    Submit();
    return inFlight.empty() ? 0 : submissions;
}

namespace {

// One thread's registration as a timeline waiter (see `unlockedWaiters`): constructed while the
// caller still holds the mutex (or the device), so a teardown that starts after the mutex is
// released sees it and keeps the semaphore until the wait returned.
struct UnlockedWaiter {
    std::atomic<int>& count;
    explicit UnlockedWaiter(std::uint64_t recorderId) : count(WaitersOf(recorderId)) { count.fetch_add(1, std::memory_order_acq_rel); }
    ~UnlockedWaiter() { count.fetch_sub(1, std::memory_order_acq_rel); }
    UnlockedWaiter(const UnlockedWaiter&) = delete;
    UnlockedWaiter& operator=(const UnlockedWaiter&) = delete;
};

// Restores the thread's active sync-site entry on every exit path (a Check throw on a lost device
// included), so the next sync on the thread is not attributed to this one's site.
struct RestoreSyncSite {
    std::size_t site;
    ~RestoreSyncSite() { activeSyncSite = site; }
};

// The timeline wait itself, on the event copied out of the recorder (retained by the caller): the
// caller holds no mutex, and the recorder may be torn down meanwhile (its destructor waits for
// the registered waiters to leave after completing every batch).
bool WaitTimeline(id<MTLSharedEvent> timeline, std::uint64_t serial) {
    return WaitEvent(timeline, serial, "timeline wait");
}

void CheckTimeline(bool reached, std::uint64_t serial) {
    if (!reached) Metal::Fail("recorded batch " + std::to_string(serial) + " did not finish on the GPU within " + std::to_string(BatchTimeoutMs()) + " ms (timeline wait)");
}

}

void Recorder::WaitSerial(std::uint64_t serial) {
    PerformanceTimer frameTiming("Recorder.WaitSerial");
    if (metal == nullptr || serial == 0) return;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const UnlockedWaiter waiter{id};
    MetalEvent timeline = metal->timeline;
    const auto reached = WaitTimeline(timeline, serial);
    if (profile) {
        unlockedWaits.fetch_add(1, std::memory_order_relaxed);
        unlockedWaitedUs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
    }
    CheckTimeline(reached, serial);
}

void Recorder::FinishUpTo(std::uint64_t serial) {
    // A drain: the wait already happened (unlocked), the fences are signaled, so the source-0 wait
    // recorded here is the bookkeeping only (a caller that did not wait first, WaitForLeases, shows
    // up in the site table by its own address).
    ++syncCounts[0];
    announcedSource = 4;
    const void* site = std::exchange(announcedSite, nullptr);
    const auto previousSite = inFlight.empty() || inFlight.front()->serial > serial ? activeSyncSite : BeginSyncSite(0, site != nullptr ? site : __builtin_return_address(0));
    while (!inFlight.empty() && inFlight.front()->serial <= serial) {
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), true, 0);
    }
    activeSyncSite = previousSite;
}

void Recorder::Sync() {
    PerformanceTimer frameTiming("Recorder.Sync");
    const auto source = std::exchange(announcedSource, 4);
    const void* site = std::exchange(announcedSite, nullptr);
    Submit();
    // Counted only when something is waited for: an idle recorder's Sync is free and would only
    // swamp the site table with the callers that check nothing first.
    const auto previousSite = inFlight.empty() ? activeSyncSite : BeginSyncSite(source, site != nullptr ? site : __builtin_return_address(0));
    while (!inFlight.empty()) {
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), true, source);
    }
    activeSyncSite = previousSite;
}

void Recorder::FlushDeferredReleases() {
    holdKept.clear();
    ReleaseDeferredKeeps();
}

void Recorder::SyncThrough(std::uint64_t address, std::size_t bytes, bool waitUnlocked) {
    if (bytes == 0) return;
    const auto end = address + bytes;
    if (completionDepth != 0 && !CompletionStoreSyncs()) {
        // A store made by a completion action (a copied buffer's write-back, GuestBufferMemory::
        // WriteBack) reached the flush hook, and a batch still in flight notes the range. Every
        // batch in flight now was recorded after the completing one (finish runs front to back),
        // so in program order this store precedes their writes: landing it before them is the
        // order the hardware gives; waiting for them first (what the hook would do) lands the
        // completing batch's bytes over the later batch's, which is wrong, and does so under the
        // hold of whoever reaped (the [hooksync] 'store' entries, up to tens of ms per reap). The
        // later batch's own ordering against this write-back is its recorder's business at record
        // time (FillBuffer and DispatchIndirect consult copiedWriters; a later copied writer's
        // write-back runs after this one in finish order; a label behind completions lands after).
        announcedSource = 4;
        announcedSite = nullptr;
        ++holdCounters.completionSyncsSkipped;
        return;
    }
    if (completionDepth != 0) ++holdCounters.completionSyncsWaited;
    const auto completionWaitStart = completionDepth != 0 && DrawProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    struct CompletionWait {
        std::chrono::steady_clock::time_point start;
        ~CompletionWait() {
            if (start != std::chrono::steady_clock::time_point{}) holdCounters.completionSyncWaitMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
    } completionWait{completionWaitStart};
    if (waitUnlocked) {
        // The hook's wait without the mutex: its own acquisition is the outermost on this thread.
        const auto source = announcedSource;
        const void* site = announcedSite != nullptr ? announcedSite : __builtin_return_address(0);
        if (syncThroughUnlocked(address, end, source, site)) return;
        // A completion-store wait (kill switch above) is already counted as 'waited'.
        if (completionDepth == 0) ++holdCounters.hookLockedWaits;
    }
    if (!SyncThroughEnabled() || (open != nullptr && overlaps(*open, address, end))) {
        // The announced site (or this call's caller) names the sync, not this function.
        if (announcedSite == nullptr) announcedSite = __builtin_return_address(0);
        // The open batch's work has not run; without one the newest in flight is the target.
        if (open != nullptr || (!inFlight.empty() && unsignaled(inFlight.back()->serial))) ++hookRealWaits;
        Sync();
        return;
    }
    const auto source = std::exchange(announcedSource, 4);
    const void* site = std::exchange(announcedSite, nullptr);
    std::uint64_t targetSerial = 0;
    for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it) {
        if (overlaps(**it, address, end)) {
            targetSerial = (*it)->serial;
            break;
        }
    }
    if (targetSerial == 0) return;
    if (unsignaled(targetSerial)) ++hookRealWaits;
    targetedSyncs.fetch_add(1, std::memory_order_relaxed);
    const RestoreSyncSite restoreSite{BeginSyncSite(source, site != nullptr ? site : __builtin_return_address(0))};
    // Batches are in flight in serial order. A completion's own guest access can sync re-entrantly
    // and finish the target first (and a new batch may reuse its allocation, so the serial, not the
    // pointer, identifies it); the loop stops then, and later batches stay in flight either way.
    while (!inFlight.empty() && inFlight.front()->serial <= targetSerial) {
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), true, source);
    }
    batchesLeftInFlight.fetch_add(inFlight.size(), std::memory_order_relaxed);
}

namespace {

// What the unlocked hook waits waited for, per read site (the [syncwait] line every 10 s under
// APS5_PROFILE_DRAW): the batches up to the target, how many of them the GPU had not finished when
// the wait began, and whether the target was the newest submission. APS5_TRACE_CAPSYNC=1 prints
// every such wait ([capsync] sync lines).
struct SyncWaitSite {
    std::uint64_t count = 0;
    std::uint64_t batchesToTarget = 0;
    std::uint64_t unsignaledAtStart = 0;
    std::uint64_t newest = 0;
    double waitedMs = 0;
    // The target batch's submit -> wait-end latency: what the wait's length is made of when the
    // batches ahead of it carry little GPU work (submission latency, not throughput).
    double targetAgeMs = 0;
};

struct SyncWaitStats {
    std::array<SyncWaitSite, static_cast<std::size_t>(GuestMemory::ReadSite::Count)> bySite{};
    // Batches up to the targets, by the queue that opened them: whether a wait for one queue's
    // producer is really a wait for the other queues' batches ahead of it on the one timeline.
    std::map<std::uint32_t, std::uint64_t> aheadByQueue;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

SyncWaitStats& SyncWaits() {
    static SyncWaitStats stats;
    return stats;
}

bool TraceCapSync() {
    static const bool trace = std::getenv("APS5_TRACE_CAPSYNC") != nullptr;
    return trace;
}

void ReportSyncWaits(SyncWaitStats& stats) {
    std::string report = "[syncwait] unlocked hook waits by read site (10 s; count/GPU wait, avg batches up to the target, avg of them unsignaled at the start, target was the newest submission, avg target submit-to-signal ms):";
    for (std::size_t site = 0; site < stats.bySite.size(); ++site) {
        const auto& entry = stats.bySite[site];
        if (entry.count == 0) continue;
        char text[160];
        std::snprintf(text, sizeof(text), " %s %llu/%.0fms %.1f %.1f %.0f%% %.1f", GuestMemory::ReadSiteName(static_cast<GuestMemory::ReadSite>(site)), static_cast<unsigned long long>(entry.count), entry.waitedMs, static_cast<double>(entry.batchesToTarget) / entry.count, static_cast<double>(entry.unsignaledAtStart) / entry.count, entry.newest * 100.0 / entry.count, entry.targetAgeMs / entry.count);
        report += text;
    }
    report += "; batches ahead by queue:";
    for (const auto& [queue, count] : stats.aheadByQueue) {
        char text[48];
        if (queue == 0xffffffffu) std::snprintf(text, sizeof(text), " untagged %llu", static_cast<unsigned long long>(count));
        else std::snprintf(text, sizeof(text), " 0x%x %llu", queue, static_cast<unsigned long long>(count));
        report += text;
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", report.c_str());
    stats = SyncWaitStats{};
}

}

bool Recorder::syncThroughUnlocked(std::uint64_t address, std::uint64_t end, int source, const void* site) {
    // Only for the outermost acquisition: a nested hook (inside a dispatch's hold) must not give
    // up the caller's mutex, and a thread that waits here holds nothing the relock could invert
    // (the hook took the mutex from a depth of 0 with the same locks held).
    if (HookLockedWait() || metal == nullptr || GuestMemory::GpuMutex().DepthOnThisThread() != 1) return false;
    // The target as SyncThrough chooses it: the open batch (submitted here, so the GPU can reach
    // it) makes everything the target, else the newest overlapping in-flight batch; with
    // SyncThrough disabled everything is the target as well.
    std::uint64_t targetSerial = 0;
    if (!SyncThroughEnabled() || (open != nullptr && overlaps(*open, address, end))) {
        Submit();
        targetSerial = submissions;
    } else {
        for (auto it = inFlight.rbegin(); it != inFlight.rend(); ++it) {
            if (overlaps(**it, address, end)) {
                targetSerial = (*it)->serial;
                break;
            }
        }
    }
    announcedSource = 4;
    announcedSite = nullptr;
    if (targetSerial == 0) return true;
    if (unsignaled(targetSerial)) ++hookRealWaits;
    // Everything the wait needs is copied out: the recorder may be torn down while the mutex is
    // released (its destructor completes every batch and waits for the semaphore to be left, see
    // WaitTimeline), after which `this` must not be touched. The id re-check below relies on the
    // teardown running under the GpuMutex (the driver's drain and failure paths reset the device
    // under it; a worker dropping its last device reference outside its lock would not).
    const auto myId = id;
    MetalEvent semaphore = metal->timeline;
    // The site entry is taken under the mutex (the table is only touched under it; entries are
    // appended, never removed, so the index stays valid across the release).
    const RestoreSyncSite restoreSite{BeginSyncSite(source, site)};
    const bool profile = DrawProfiled();
    const bool trace = TraceCapSync();
    const auto readSite = GuestMemory::CurrentReadSite();
    const auto newestSerial = submissions;
    std::size_t batchesToTarget = 0, unsignaledAtStart = 0, targetRanges = 0;
    std::pair<std::uint64_t, std::uint64_t> hit{0, 0};
    std::chrono::steady_clock::time_point targetSubmittedAt{};
    std::map<std::uint32_t, std::uint64_t> aheadByQueue;
    if (profile || trace) {
        for (const auto& batch : inFlight) {
            if (batch->serial > targetSerial) break;
            ++batchesToTarget;
            ++aheadByQueue[batch->queue];
            if (!signaled(*batch)) ++unsignaledAtStart;
            if (batch->serial != targetSerial) continue;
            targetSubmittedAt = batch->submittedAt;
            targetRanges = batch->writes.size();
            for (const auto& range : batch->writes) {
                if (address < range.second && range.first < end) {
                    hit = range;
                    break;
                }
            }
        }
    }
    const auto start = profile || trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto waited = start;
    auto& mutex = GuestMemory::GpuMutex();
    bool result = true;
    {
        // Registered before the release: a teardown can only start once the mutex is free.
        const UnlockedWaiter waiter{myId};
        mutex.unlock();
        result = WaitTimeline(semaphore, targetSerial);
        // Read before the relock: the GPU wait, not the wait for the mutex behind it.
        if (profile || trace) waited = std::chrono::steady_clock::now();
    }
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
    mutex.lock();
    const auto ms = std::chrono::duration<double, std::milli>(waited - start).count();
    if (profile) {
        // The GPU wait is charged like a locked wait, so the per-source, per-thread and per-site
        // totals of the [recorder] line keep their meaning; the unlocked count says how many were
        // not held. The relock's wait for the mutex is kept apart (the [lock] line also counts it
        // under 'hook' waits, for a cross-check).
        syncWaitedMs[source] += ms;
        CountThreadSync(source, ms);
        CountSiteWait(ms);
        ++holdCounters.hookUnlockedWaits;
        holdCounters.hookUnlockedWaitMs += ms;
        if (source >= 0 && source < 5) {
            ++holdCounters.hookUnlockedWaitsBySource[source];
            holdCounters.hookUnlockedWaitMsBySource[source] += ms;
        }
        holdCounters.hookRelockMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waited).count();
        auto& stats = SyncWaits();
        auto& siteStats = stats.bySite[static_cast<std::size_t>(readSite)];
        ++siteStats.count;
        siteStats.waitedMs += ms;
        siteStats.batchesToTarget += batchesToTarget;
        siteStats.unsignaledAtStart += unsignaledAtStart;
        if (targetSerial == newestSerial) ++siteStats.newest;
        if (targetSubmittedAt != std::chrono::steady_clock::time_point{}) siteStats.targetAgeMs += std::chrono::duration<double, std::milli>(waited - targetSubmittedAt).count();
        for (const auto& [queue, count] : aheadByQueue) stats.aheadByQueue[queue] += count;
        if (std::chrono::steady_clock::now() - stats.lastReport > std::chrono::seconds(10)) ReportSyncWaits(stats);
    }
    if (trace) {
        static std::atomic<int> lines{0};
        if (lines.fetch_add(1) < 600) {
            const auto packet = GuestMemory::CurrentPacket();
            std::fprintf(stderr, "[capsync] sync q0x%x %s %s 0x%llx+0x%llx: target %llu (newest %llu), %zu batches up to it (%zu unsignaled at the start), target noted %zu ranges, hit 0x%llx+0x%llx, GPU wait %.1f ms\n", packet.queue, PacketName(packet.opcode).c_str(), GuestMemory::ReadSiteName(readSite), static_cast<unsigned long long>(address), static_cast<unsigned long long>(end - address), static_cast<unsigned long long>(targetSerial), static_cast<unsigned long long>(newestSerial), batchesToTarget, unsignaledAtStart, targetRanges, static_cast<unsigned long long>(hit.first), static_cast<unsigned long long>(hit.second - hit.first), ms);
        }
    }
    if (!RecorderAlive(myId)) {
        // Torn down meanwhile: its destructor's Sync() ran every completion, nothing is pending.
        ++holdCounters.hookUnlockedTornDown;
        CheckTimeline(result, targetSerial);
        return true;
    }
    CheckTimeline(result, targetSerial);
    // The completions up to the target, under the mutex again; the fences are signaled, so the
    // waits inside finish() return at once (charged to the same site entry). Another thread may
    // have finished some meanwhile.
    targetedSyncs.fetch_add(1, std::memory_order_relaxed);
    while (!inFlight.empty() && inFlight.front()->serial <= targetSerial) {
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), true, source);
    }
    batchesLeftInFlight.fetch_add(inFlight.size(), std::memory_order_relaxed);
    return true;
}

bool Recorder::Reap() {
    PerformanceTimer frameTiming("Recorder.Reap");
    dropStaleHoldKept();
    const bool profile = DrawProfiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::uint64_t retired = 0;
    while (!inFlight.empty()) {
        if (!signaled(*inFlight.front())) break;
        auto batch = std::move(inFlight.front());
        inFlight.pop_front();
        finish(std::move(batch), false, 4);
        ++retired;
    }
    if (profile) {
        ++holdCounters.reaps;
        if (retired != 0) ++holdCounters.reapsWithWork;
        holdCounters.reapBatches += retired;
        holdCounters.reapMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    return inFlight.empty();
}

void Recorder::finish(std::unique_ptr<Batch> batch, bool wait, int source, bool releaseNow) {
    inFlightKeptBytes -= batch->keptBytes;
    // The batch's writes stay published until its completions ran, on every exit path.
    struct Finishing {
        Recorder& recorder;
        const Batch* batch;
        ~Finishing() {
            auto& list = recorder.finishing;
            list.erase(std::remove(list.begin(), list.end(), batch), list.end());
            // The batch's completion labels have landed, or never will (fence failure below).
            if (batch->completionLabelCount != 0) completionLabels.fetch_sub(batch->completionLabelCount, std::memory_order_acq_rel);
            if (batch->writeBackCompletionCount != 0) writeBackCompletions.fetch_sub(batch->writeBackCompletionCount, std::memory_order_acq_rel);
            // This also runs while the fence-failure throw unwinds: an allocation failure in the
            // rebuild keeps the previous snapshot, which is a superset (a stale entry only causes a
            // false hit, never a miss) rather than terminating.
            try {
                recorder.publishPendingWrites();
            } catch (...) {
            }
        }
    } finishingScope{*this, batch.get()};
    finishing.push_back(batch.get());
    if (wait) {
        // APS5_PROFILE_DRAW: how long the CPU waits for recorded GPU work, reported every 10 s.
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        static double waitedMs = 0;
        static std::uint64_t waits = 0;
        static auto lastReport = std::chrono::steady_clock::now();
        const auto waitStart = std::chrono::steady_clock::now();
        // A fence still unsignaled here makes this a real GPU wait under the mutex (finish runs
        // under it): the [lock] line's 'locked GPU waits'.
        const bool signaledAtStart = !profile || signaled(*batch);
        struct Report {
            bool enabled;
            int source;
            std::chrono::steady_clock::time_point start;
            const Recorder& recorder;
            bool signaledAtStart;
            ~Report() {
                if (!enabled) return;
                const auto now = std::chrono::steady_clock::now();
                const auto ms = std::chrono::duration<double, std::milli>(now - start).count();
                waitedMs += ms;
                syncWaitedMs[source] += ms;
                ++waits;
                CountThreadSync(source, ms);
                CountSiteWait(ms);
                if (!signaledAtStart) GuestMemory::NoteLockedGpuWait(ms);
                if (now - lastReport > std::chrono::seconds(10)) {
                    lastReport = now;
                    const auto& h = holdCounters;
                    AgcDriver::ProfilePrint_nid_no_patch("[recorder] %llu syncs waited %.1f s for the GPU in total (sources, count/wait: idle %llu/%.1fs, pending write %llu/%.1fs, recorded store %llu/%.1fs, address-based %llu/%.1fs, other %llu/%.1fs); hook %llu calls, %llu locked; %llu targeted syncs left %llu batches in flight; snapshot %llu rebuilds %.0f ms, %llu notes covered; %llu unlocked timeline waits %.1f s; %llu submissions, %zu label entries, %llu completion labels pending; fence waits by thread (count/wait):%s; top sync sites (source@caller syncs/batches/wait):%s; under holds (cumulative): %llu completions ran %.0f ms (kept objects released %.0f ms), pending-write syncs from completions: %llu skipped, %llu waited %.0f ms; %llu reaps (%llu with work) retired %llu batches in %.0f ms; hook waits unlocked %llu / %.0f ms GPU (pending write %llu / %.0f ms, recorded store %llu / %.0f ms) + %.0f ms relock (%llu found the recorder torn down), locked %llu; deferred releases: on the release thread %llu batches (%llu objects) in %.0f ms, inline %llu batches (%llu objects) in %.0f ms (%llu batches over the queue bound of %zu), queue max %llu batches, %llu pending\n",static_cast<unsigned long long>(waits), waitedMs / 1000, static_cast<unsigned long long>(syncCounts[0]), syncWaitedMs[0] / 1000, static_cast<unsigned long long>(syncCounts[1]), syncWaitedMs[1] / 1000, static_cast<unsigned long long>(syncCounts[2]), syncWaitedMs[2] / 1000, static_cast<unsigned long long>(syncCounts[3]), syncWaitedMs[3] / 1000, static_cast<unsigned long long>(syncCounts[4]), syncWaitedMs[4] / 1000, static_cast<unsigned long long>(hookCalls.load()), static_cast<unsigned long long>(hookLocks.load()), static_cast<unsigned long long>(targetedSyncs.load()), static_cast<unsigned long long>(batchesLeftInFlight.load()), static_cast<unsigned long long>(snapshotRebuilds), snapshotRebuildMs, static_cast<unsigned long long>(snapshotCovered), static_cast<unsigned long long>(unlockedWaits.load()), unlockedWaitedUs.load() / 1e6, static_cast<unsigned long long>(recorder.submissions), recorder.PendingLabels(), static_cast<unsigned long long>(completionLabels.load()), ThreadSyncReport().c_str(), SyncSiteReport().c_str(), static_cast<unsigned long long>(h.completions), h.completionMs, h.keptReleaseMs, static_cast<unsigned long long>(h.completionSyncsSkipped), static_cast<unsigned long long>(h.completionSyncsWaited), h.completionSyncWaitMs, static_cast<unsigned long long>(h.reaps), static_cast<unsigned long long>(h.reapsWithWork), static_cast<unsigned long long>(h.reapBatches), h.reapMs, static_cast<unsigned long long>(h.hookUnlockedWaits), h.hookUnlockedWaitMs, static_cast<unsigned long long>(h.hookUnlockedWaitsBySource[1]), h.hookUnlockedWaitMsBySource[1], static_cast<unsigned long long>(h.hookUnlockedWaitsBySource[2]), h.hookUnlockedWaitMsBySource[2], h.hookRelockMs, static_cast<unsigned long long>(h.hookUnlockedTornDown), static_cast<unsigned long long>(h.hookLockedWaits), static_cast<unsigned long long>(threadReleases.load()), static_cast<unsigned long long>(threadObjects.load()), threadReleaseUs.load() / 1000.0, static_cast<unsigned long long>(inlineReleases.load()), static_cast<unsigned long long>(inlineObjects.load()), inlineReleaseUs.load() / 1000.0, static_cast<unsigned long long>(inlineOverBound.load()), ReleaseQueueBound(), static_cast<unsigned long long>(releaseQueueMax.load()), static_cast<unsigned long long>(deferredPending.load()));
                    AgcDriver::ProfilePrint_nid_no_patch("[recorder] completion label stores: %llu run, %llu skipped (no CPU write-back overlapped them); %llu counted pending at a write-back; %llu write-backs over a tracked label; %llu write-back completions pending\n", static_cast<unsigned long long>(completionStoresRun.load()), static_cast<unsigned long long>(completionStoresSkipped.load()), static_cast<unsigned long long>(completionLabelsCountedLate.load()), static_cast<unsigned long long>(writeBacksOverLabels.load()), static_cast<unsigned long long>(writeBackCompletions.load()));
                    AgcDriver::ProfilePrint_nid_no_patch("[recorder] submits %llu, vkQueueSubmit mean %.1f us, max %.1f us\n", static_cast<unsigned long long>(submitCount), submitCount != 0 ? submitUs / static_cast<double>(submitCount) : 0.0, submitMaxUs);
                    const auto expansions = Metal::IndexExpansionCounts();
                    const auto snapshots = StorageTexture::TakeSnapshotCounts();
                    AgcDriver::ProfilePrint_nid_no_patch("[cpuwaits] index expansions: %llu gathered on the GPU, %llu read on the CPU after a sync (D81); storage snapshots: %llu copied on the GPU, %llu waited for by a reader, uploads kept on the CPU: shadowed %llu, CPU write-back %llu, no import %llu (D82); cumulative\n", static_cast<unsigned long long>(expansions.gathered), static_cast<unsigned long long>(expansions.settled), static_cast<unsigned long long>(snapshots.gpu), static_cast<unsigned long long>(snapshots.settled), static_cast<unsigned long long>(snapshots.refused[0]), static_cast<unsigned long long>(snapshots.refused[1]), static_cast<unsigned long long>(snapshots.refused[2]));
                    const auto reads = Recorder::ReadCounts();
                    AgcDriver::ProfilePrint_nid_no_patch("[recorder] in-place reads: %llu noted, %llu queries, hits by reader: dispatch element %llu, gpu copy %llu, address-based %llu, indirect %llu, storage upload %llu, copy source %llu, draw input %llu; %llu hits on signaled batches ignored\n", static_cast<unsigned long long>(reads.noted), static_cast<unsigned long long>(reads.queries), static_cast<unsigned long long>(reads.hits[0]), static_cast<unsigned long long>(reads.hits[1]), static_cast<unsigned long long>(reads.hits[2]), static_cast<unsigned long long>(reads.hits[3]), static_cast<unsigned long long>(reads.hits[4]), static_cast<unsigned long long>(reads.hits[5]), static_cast<unsigned long long>(reads.hits[6]), static_cast<unsigned long long>(reads.staleIgnored));
                }
            }
        } report{profile, source, waitStart, *this, signaledAtStart};
        // macOS: the batch's serial on the timeline event; a batch still running after 5 s is
        // reported (every 5 s) so a GPU-side hang is visible, and one past BatchTimeoutMs fails.
        if (batch->submitted && !WaitEvent(metal->timeline, batch->serial, "fence wait")) {
            const auto serial = batch->serial;
            // Its command buffer may still be executing on a hung GPU: its objects (and the upload
            // memory it reads) are deliberately leaked rather than reused or freed under it.
            static_cast<void>(batch->metal.release());
            release(*batch);
            Metal::Fail("recorded batch " + std::to_string(serial) + " did not finish on the GPU within " + std::to_string(BatchTimeoutMs()) + " ms");
        }
    }
    // A failure the commit feedback reported for this batch or an earlier one (feedback can land
    // after the event value, so a late report surfaces at a later finish).
    {
        std::string failure;
        std::uint64_t failed = 0;
        {
            std::lock_guard lock(metal->sink->mutex);
            if (!metal->sink->errors.empty()) {
                failed = metal->sink->errors.front().first;
                failure = std::move(metal->sink->errors.front().second);
                metal->sink->errors.erase(metal->sink->errors.begin());
            }
        }
        if (failed != 0) {
            release(*batch);
            Metal::Fail("recorded batch " + std::to_string(failed) + " failed on the GPU: " + failure);
        }
    }
    if (BatchStampsEnabled() && batch->metal != nullptr && batch->metal->feedback != nullptr && batch->metal->feedback->delivered.load(std::memory_order_acquire) && batch->metal->timestamps == nil) {
        // Stamps asked for but no counter heap (its creation failed): the commit feedback's GPU
        // times, when it is in. Never with profiling off (the stamps stay 0 then, as upstream).
        batch->gpuStartNs = batch->metal->feedback->gpuStartSeconds * 1e9;
        batch->gpuEndNs = batch->metal->feedback->gpuEndSeconds * 1e9;
    }
    readGpuTiming(*batch);
    readSamples(*batch);
    if (batch->serial != 0) {
        std::lock_guard ringLock(completedMutex);
        auto& entry = completed[batch->serial % completed.size()];
        entry.serial = batch->serial;
        entry.gpuStartNs = batch->gpuStartNs;
        entry.gpuEndNs = batch->gpuEndNs;
        entry.submittedAt = batch->submittedAt;
        entry.fenceSeenAt = std::chrono::steady_clock::now();
        entry.readGeneration = batch->readGeneration;
        entry.reads.clear();
        if (FlipReadCheck()) {
            entry.reads = notedReads(*batch);
        }
    }
    // Completions store GPU results to guest memory; a failing one is reported, the rest still run.
    // Timed (APS5_PROFILE_DRAW) with the release of the kept objects: both run under whichever hold
    // reaped or synced the batch. Inside them the flush hook makes no pending-write wait (SyncThrough).
    const bool profile = DrawProfiled();
    const auto completionStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    {
        struct InCompletion {
            InCompletion() { ++completionDepth; }
            ~InCompletion() { --completionDepth; }
        } inCompletion;
        for (auto& action : batch->completions) {
            try {
                action();
            } catch (const BdaUnalignedAccess& error) {
                // 0f0197c16d, 69b683a901: the unaligned BDA reports are rate-limited per instruction.
                static std::mutex failuresMutex;
                static std::unordered_map<std::uint32_t, std::uint64_t> failures;
                constexpr std::uint64_t kDeferredFailureFirst = 8, kDeferredFailureEvery = 1024;
                const auto count = [&] {
                    std::lock_guard lock(failuresMutex);
                    return ++failures[error.instruction];
                }();
                if (count <= kDeferredFailureFirst) std::fprintf(stderr, "[gpu] deferred write-back failed: %s\n", error.what());
                else if (count % kDeferredFailureEvery == 0) std::fprintf(stderr, "[gpu] deferred write-back failed with an unaligned BDA access at pc 0x%x %llu times so far; latest: %s\n", error.instruction, static_cast<unsigned long long>(count), error.what());
            } catch (const std::exception& error) {
                AgcDriver::LogCoalescedStderr_nid_no_patch("[gpu] deferred write-back failed: %s\n", error.what());
            }
        }
    }
    if (!batch->completions.empty()) KernelWakeSleepers_nid_no_patch();
    if (profile) holdCounters.completions += batch->completions.size();
    const auto keptStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Deferred only from inside a hold: the unlock that releases the list is this thread's own, and
    // a caller finishing batches without the mutex (a test) might never make one.
    if (releaseNow || ReleaseUnderLock() || !GuestMemory::GpuMutex().HeldByThisThread()) {
        batch->completions.clear();
        batch->kept.clear();
    } else {
        // To this thread's unlock (see ReleaseDeferredKeeps); the completions go with the objects,
        // their captures hold the same ones. The push comes first: a failed one (the list's growth)
        // destroys the batch's objects here, under the mutex, counts nothing the destructor would
        // wait for and must not escape (the rest of finish() releases the fence and command buffer).
        try {
            DeferredBatches().push_back({std::move(batch->kept), std::move(batch->completions)});
            deferredPending.fetch_add(1, std::memory_order_acq_rel);
            DeferredBytes() += batch->keptBytes;
        } catch (...) {
        }
        batch->completions.clear();
        batch->kept.clear();
    }
    if (profile) {
        const auto now = std::chrono::steady_clock::now();
        holdCounters.keptReleaseMs += std::chrono::duration<double, std::milli>(now - keptStart).count();
        holdCounters.completionMs += std::chrono::duration<double, std::milli>(now - completionStart).count();
    }
    // The batch's label entries leave the table (a later label to the same dword already replaced
    // its entry and belongs to another batch). Correctness never depended on this removal.
    removeLabels(*batch);
    release(*batch);
}

void Recorder::discard(std::unique_ptr<Batch> batch) noexcept {
    removeLabels(*batch);
    if (batch->completionLabelCount != 0) completionLabels.fetch_sub(batch->completionLabelCount, std::memory_order_acq_rel);
    if (batch->writeBackCompletionCount != 0) writeBackCompletions.fetch_sub(batch->writeBackCompletionCount, std::memory_order_acq_rel);
    if (activeRecorder == this) pendingLabelSince.store(NoPendingLabel, std::memory_order_release);
    release(*batch);
    try {
        publishPendingWrites();
    } catch (...) {
    }
}

void Recorder::removeLabels(Batch& batch) {
    if (!batch.labelDwords.empty() || !batch.wideLabels.empty()) {
        std::lock_guard tableLock(labelTableMutex);
        for (const auto dword : batch.labelDwords) {
            const auto found = labels.find(dword);
            if (found != labels.end() && found->second.batch == &batch) labels.erase(found);
        }
        for (const auto entry : batch.wideLabels) wideLabels.erase(entry);
        recordedLabels.store(labels.size(), std::memory_order_relaxed);
    }
    batch.labelDwords.clear();
    batch.wideLabels.clear();
}

void Recorder::release(Batch& batch) noexcept {
    batch.samplePool.reset();
    batch.segments = nullptr;
    batch.segmentPool.reset();
    if (batch.metal == nullptr) return;
    auto& objects = *batch.metal;
    if (objects.timestamps != nil && !GpuTimingEnabled() && objects.timestamps.count == 2 && metal->spareHeaps.size() < 64) {
        try {
            metal->spareHeaps.push_back(objects.timestamps);
        } catch (...) {
        }
    }
    objects.timestamps = nil;
    objects.feedback.reset();
    // A completed (or never submitted) batch's objects are kept for reuse: the command buffer is
    // not executing, so its allocator can be reset and its upload memory recycled.
    try {
        if (objects.stream != nullptr && objects.stream->Recording()) static_cast<void>(objects.stream->End());
        if (objects.stream != nullptr) objects.stream->Recycle();
        if (objects.allocator != nil) [objects.allocator reset];
        if (metal->spare.size() < 64 && objects.stream != nullptr) {
            metal->spare.push_back(std::move(batch.metal));
            return;
        }
    } catch (...) {
    }
    batch.metal.reset();
}

}

namespace AgcDriver::Graphics::Metal {

bool FlushForGpuRead(std::uint64_t address, std::size_t bytes) {
    // A completion action may write guest memory on the CPU when its batch finishes (a write-back):
    // queue order does not cover that, so the access waits as a CPU read would.
    bool writeBack = false;
    if (Recorder::PendingWriteBackCompletions() != 0) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
        std::lock_guard gpu(GuestMemory::GpuMutex());
        auto* recorder = Recorder::Active();
        writeBack = recorder != nullptr && recorder->CompletionWriteOverlaps(address, bytes);
    }
    if (writeBack) {
        FlushForAccessImpl(address, bytes, false);
        gpuReadsWaited.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    FlushForAccessImpl(address, bytes, true);
    gpuReadsOrdered.fetch_add(1, std::memory_order_relaxed);
    return true;
}

std::pair<std::uint64_t, std::uint64_t> GpuReadCounts() {
    return {gpuReadsOrdered.load(std::memory_order_relaxed), gpuReadsWaited.load(std::memory_order_relaxed)};
}

}
