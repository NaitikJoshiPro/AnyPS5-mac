#pragma once

#include <algorithm>
#include <cstdint>

// Guest yield loops (scePthreadYield, sched_yield, pthread_yield). On macOS sched_yield returns at once when no other
// thread wants the core, so a guest that waits by yielding in a loop (a task scheduler's idle spin, a yield-based
// lock) keeps a core at 100% doing nothing: ASTRO BOT's three TBB workers each spent 8% of a core in yields (D73).
// The first FreeYields of a streak are plain yields; after that the thread sleeps, starting at MinSleepNs and
// doubling up to MaxSleepNs. A gap of more than StreakGapNs between two yields ends the streak.
struct YieldBackoff {
    static constexpr std::uint32_t FreeYields = 64;
    static constexpr std::uint64_t StreakGapNs = 1'000'000;
    static constexpr std::uint64_t MinSleepNs = 50'000;
    static constexpr std::uint64_t MaxSleepNs = 400'000;

    std::uint64_t lastNs = 0;
    std::uint32_t streak = 0;

    // The nanoseconds to sleep instead of yielding at time nowNs, or 0 to yield.
    std::uint64_t Next(std::uint64_t nowNs) {
        if (streak != 0 && (nowNs < lastNs || nowNs - lastNs > StreakGapNs)) streak = 0;
        lastNs = nowNs;
        if (++streak <= FreeYields) return 0;
        const std::uint32_t doublings = std::min<std::uint32_t>((streak - FreeYields - 1) / FreeYields, 3);
        return std::min(MinSleepNs << doublings, MaxSleepNs);
    }

    // Called after a sleep: the streak continues from the time the thread woke.
    void Woke(std::uint64_t nowNs) { lastNs = nowNs; }
};
