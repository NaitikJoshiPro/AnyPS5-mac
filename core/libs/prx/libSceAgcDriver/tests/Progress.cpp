// Driver progress events (Execution/src/Driver/Synchronization/Progress.cpp, D80): waits wake on an event, time out
// without one, never miss an event raised between reading the epoch and sleeping, and call the driver's hook.
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/Progress.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

using Clock = std::chrono::steady_clock;
namespace Progress = AgcDriver::Progress;

}

int main() {
    // No event: the wait runs to its deadline.
    auto seen = Progress::Epoch();
    auto start = Clock::now();
    Require(!Progress::Wait(seen, start + std::chrono::milliseconds(20)), "a wait without an event times out");
    Require(Clock::now() - start >= std::chrono::milliseconds(19), "the timed-out wait slept to its deadline");

    // An event from another thread wakes the wait long before its deadline.
    seen = Progress::Epoch();
    std::thread notifier([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        Progress::Notify();
    });
    start = Clock::now();
    Require(Progress::Wait(seen, start + std::chrono::seconds(5)), "a notified wait reports the event");
    const auto waited = Clock::now() - start;
    notifier.join();
    Require(waited < std::chrono::seconds(1), "the event woke the wait (" + std::to_string(std::chrono::duration<double, std::milli>(waited).count()) + " ms)");

    // An event between reading the epoch and waiting is not lost: the wait returns at once.
    seen = Progress::Epoch();
    Progress::Notify();
    start = Clock::now();
    Require(Progress::Wait(seen, start + std::chrono::seconds(5)), "an event before the wait counts");
    Require(Clock::now() - start < std::chrono::milliseconds(100), "the wait did not sleep");

    // The hook runs on every event.
    static std::atomic<int> calls{0};
    int context = 7;
    Progress::SetHook([](void* value) { if (*static_cast<int*>(value) == 7) ++calls; }, &context);
    Progress::Notify();
    Progress::Notify();
    Progress::SetHook(nullptr, nullptr);
    Progress::Notify();
    Require(calls.load() == 2, "the hook ran once per event while set (" + std::to_string(calls.load()) + ")");

    const auto counters = Progress::Read();
    Require(counters.waits == 3 && counters.wokenByEvent == 2 && counters.timedOut == 1, "counters: waits, wake-ups and timeouts");
    std::puts("progress events: ok");
    return 0;
}
