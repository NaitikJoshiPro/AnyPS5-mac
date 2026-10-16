#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/Progress.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace AgcDriver::Progress {

namespace {

struct State {
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<std::uint64_t> epoch{0};
    std::atomic<std::uint32_t> waiting{0};
    std::atomic<void (*)(void*)> hook{nullptr};
    std::atomic<void*> hookContext{nullptr};
    std::atomic<std::uint64_t> notifies{0}, waits{0}, woken{0}, timedOut{0};
};

State& Get() {
    static auto* state = new State; // never destroyed: Metal's listener thread may notify during exit
    return *state;
}

}

std::uint64_t Epoch() {
    return Get().epoch.load(std::memory_order_acquire);
}

void Notify() {
    auto& state = Get();
    state.epoch.fetch_add(1, std::memory_order_acq_rel);
    state.notifies.fetch_add(1, std::memory_order_relaxed);
    if (state.waiting.load(std::memory_order_acquire) != 0) {
        // Taking the mutex orders this with a waiter between its epoch check and its sleep: no lost wake-up.
        { std::lock_guard lock(state.mutex); }
        state.changed.notify_all();
    }
    if (const auto hook = state.hook.load(std::memory_order_acquire)) hook(state.hookContext.load(std::memory_order_acquire));
}

bool Wait(std::uint64_t seen, std::chrono::steady_clock::time_point deadline) {
    auto& state = Get();
    state.waits.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock lock(state.mutex);
    state.waiting.fetch_add(1, std::memory_order_acq_rel);
    const bool woken = state.changed.wait_until(lock, deadline, [&] { return state.epoch.load(std::memory_order_acquire) != seen; });
    state.waiting.fetch_sub(1, std::memory_order_acq_rel);
    (woken ? state.woken : state.timedOut).fetch_add(1, std::memory_order_relaxed);
    return woken;
}

void SetHook(void (*hook)(void*), void* context) {
    auto& state = Get();
    state.hookContext.store(context, std::memory_order_release);
    state.hook.store(hook, std::memory_order_release);
}

Counters Read() {
    auto& state = Get();
    return {state.notifies.load(std::memory_order_relaxed), state.waits.load(std::memory_order_relaxed),
            state.woken.load(std::memory_order_relaxed), state.timedOut.load(std::memory_order_relaxed)};
}

}
