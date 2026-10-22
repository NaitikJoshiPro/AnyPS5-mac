// YieldBackoff (libkernel Pthread/include/YieldBackoff.hpp, D73): guest yield loops back off to short sleeps.
#include "../prx/libkernel/Pthread/include/YieldBackoff.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

}

int main() {
    YieldBackoff backoff;
    std::uint64_t now = 1'000'000'000;
    for (std::uint32_t i = 0; i < YieldBackoff::FreeYields; ++i, now += 1'000)
        Require(backoff.Next(now) == 0, "the first yields of a streak are plain yields");
    // Past the free yields the thread sleeps, doubling every FreeYields yields up to the cap.
    std::uint64_t first = backoff.Next(now);
    Require(first == YieldBackoff::MinSleepNs, "the first backed-off yield sleeps MinSleepNs, got " + std::to_string(first));
    std::uint64_t last = first;
    for (std::uint32_t i = 0; i < 8 * YieldBackoff::FreeYields; ++i) {
        now += last + 1'000;
        backoff.Woke(now - 1'000);
        last = backoff.Next(now);
        Require(last >= YieldBackoff::MinSleepNs && last <= YieldBackoff::MaxSleepNs, "sleeps stay within the bounds");
    }
    Require(last == YieldBackoff::MaxSleepNs, "a long streak sleeps MaxSleepNs, got " + std::to_string(last));
    // A pause longer than the gap starts a new streak of plain yields.
    now += YieldBackoff::StreakGapNs + 1;
    Require(backoff.Next(now) == 0, "a yield after a gap is a plain yield again");
    // A clock that goes backwards does not leave the thread sleeping.
    YieldBackoff backwards;
    for (std::uint32_t i = 0; i <= YieldBackoff::FreeYields; ++i) backwards.Next(5'000'000 + i);
    Require(backwards.Next(1'000) == 0, "a time before the last yield starts a new streak");
    std::puts("yield backoff: ok");
    return 0;
}
