#include "GuestStats.hpp"

#include <FEXCore/Debug/InternalThreadState.h>
// FEXCore's targets define ARCHITECTURE_arm64; aps5-fex's do not, and SHMStats.h picks its cycle counter by it.
#if defined(__aarch64__) && !defined(ARCHITECTURE_arm64)
#define ARCHITECTURE_arm64 1
#endif
#include <FEXCore/Utils/SHMStats.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace Aps5Fex {

namespace {

using FEXCore::SHMStats::ThreadStats;

// Sums of the fields a report prints. Reads race the owning thread's plain increments; on arm64 an aligned 64-bit
// load is single-copy atomic, so a report is at most one event behind.
struct Totals {
    std::uint64_t jitTime = 0, jitCount = 0, cacheMiss = 0, readLock = 0, writeLock = 0, diskHit = 0, diskMiss = 0,
                  diskLookup = 0, smc = 0;

    void Add(const ThreadStats& stats) {
        jitTime += std::atomic_ref(const_cast<std::uint64_t&>(stats.AccumulatedJITTime)).load(std::memory_order_relaxed);
        jitCount += stats.AccumulatedJITCount;
        cacheMiss += stats.AccumulatedCacheMissCount;
        readLock += std::atomic_ref(const_cast<std::uint64_t&>(stats.AccumulatedCacheReadLockTime)).load(std::memory_order_relaxed);
        writeLock += std::atomic_ref(const_cast<std::uint64_t&>(stats.AccumulatedCacheWriteLockTime)).load(std::memory_order_relaxed);
        diskHit += stats.AccumulatedDiskCacheHitCount;
        diskMiss += stats.AccumulatedDiskCacheMissCount;
        diskLookup += stats.AccumulatedDiskCacheLookupTime;
        smc += stats.AccumulatedSMCCount;
    }
};

struct State {
    std::mutex mutex;
    std::vector<std::pair<FEXCore::Core::InternalThreadState*, std::unique_ptr<ThreadStats>>> live;
    Totals retired;  // threads that ended
    Totals reported; // everything up to the previous report
    std::uint64_t windows = 0;
    std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
    unsigned reportSeconds = 0;
    bool started = false;
};

State& Instance() {
    static auto* state = new State; // never destroyed: the report thread and atexit outlive statics
    return *state;
}

unsigned ReportSeconds() {
    static const unsigned seconds = [] {
        const char* value = std::getenv("APS5_FEX_STATS");
        if (value == nullptr || *value == 0) return 0u;
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || *end != 0) {
            std::fprintf(stderr, "[fex-stats] APS5_FEX_STATS=%s is not a number of seconds; statistics off\n", value);
            return 0u;
        }
        return static_cast<unsigned>(parsed);
    }();
    return seconds;
}

double TicksPerMs() {
    std::uint64_t frequency = 0;
    __asm volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
    return frequency == 0 ? 24000.0 : static_cast<double>(frequency) / 1000.0;
}

void ReportAtExit() { GuestStats::Report(stderr); }

}

bool GuestStats::Enabled() { return ReportSeconds() != 0; }

void GuestStats::Start() {
    if (!Enabled()) return;
    auto& state = Instance();
    {
        std::lock_guard lock(state.mutex);
        if (state.started) return;
        state.started = true;
        state.reportSeconds = ReportSeconds();
        state.windowStart = std::chrono::steady_clock::now();
    }
    std::fprintf(stderr, "[fex-stats] translation statistics on, report every %u s\n", state.reportSeconds);
    std::atexit(ReportAtExit);
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(Instance().reportSeconds));
            Report(stderr);
        }
    }).detach();
}

void GuestStats::Register(FEXCore::Core::InternalThreadState* thread) {
    if (!Enabled() || thread == nullptr) return;
    auto stats = std::make_unique<ThreadStats>();
    thread->ThreadStats = stats.get();
    auto& state = Instance();
    std::lock_guard lock(state.mutex);
    state.live.emplace_back(thread, std::move(stats));
}

void GuestStats::Unregister(FEXCore::Core::InternalThreadState* thread) {
    if (!Enabled() || thread == nullptr) return;
    auto& state = Instance();
    std::lock_guard lock(state.mutex);
    const auto found = std::find_if(state.live.begin(), state.live.end(), [thread](const auto& entry) { return entry.first == thread; });
    if (found == state.live.end()) return;
    state.retired.Add(*found->second);
    thread->ThreadStats = nullptr;
    state.live.erase(found);
}

void GuestStats::Report(std::FILE* out) {
    auto& state = Instance();
    std::lock_guard lock(state.mutex);
    Totals now = state.retired;
    for (const auto& [thread, stats] : state.live) now.Add(*stats);
    const auto& before = state.reported;
    const auto clock = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(clock - state.windowStart).count();
    static const double ticksPerMs = TicksPerMs();
    const std::uint64_t compiled = now.jitCount - before.jitCount;
    const double jitMs = static_cast<double>(now.jitTime - before.jitTime) / ticksPerMs;
    std::fprintf(out,
                 "[fex-stats] window %" PRIu64 ": %.1f s, %zu guest threads; JIT %" PRIu64 " blocks in %.1f ms (%.1f us each), "
                 "lookup-cache misses %" PRIu64 ", cache lock read %.1f ms write %.1f ms, disk cache %" PRIu64 " hits %" PRIu64
                 " misses (%.1f ms lookup), SMC %" PRIu64 "; total JIT %" PRIu64 " blocks in %.1f ms\n",
                 state.windows++, seconds, state.live.size(), compiled, jitMs, compiled == 0 ? 0.0 : jitMs * 1000.0 / static_cast<double>(compiled),
                 now.cacheMiss - before.cacheMiss, static_cast<double>(now.readLock - before.readLock) / ticksPerMs,
                 static_cast<double>(now.writeLock - before.writeLock) / ticksPerMs, now.diskHit - before.diskHit,
                 now.diskMiss - before.diskMiss, static_cast<double>(now.diskLookup - before.diskLookup) / ticksPerMs,
                 now.smc - before.smc, now.jitCount, static_cast<double>(now.jitTime) / ticksPerMs);
    std::fflush(out);
    state.reported = now;
    state.windowStart = clock;
}

}
