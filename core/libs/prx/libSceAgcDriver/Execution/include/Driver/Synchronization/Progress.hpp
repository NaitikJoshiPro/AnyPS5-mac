#pragma once

#include <chrono>
#include <cstdint>

// Driver progress events (D80): a GPU batch finished (the Metal timeline's listener), completion labels were reaped,
// a packet finished, a recorded write landed. Threads that wait for guest memory (WAIT_REG_MEM) or for completions
// sleep on it instead of polling every 200 us / 1 ms. A guest CPU write to a label raises no event, so every wait
// keeps a bounded fallback timeout.
namespace AgcDriver::Progress {

// The current epoch. Read it before checking a condition, then Wait(epoch, ...) sleeps only if nothing happened since.
std::uint64_t Epoch();
// One progress event: wakes every Wait. Cheap when nobody waits (an atomic increment and a load).
void Notify();
// Sleeps until the epoch differs from `seen` or until `deadline`. True when woken by an event.
bool Wait(std::uint64_t seen, std::chrono::steady_clock::time_point deadline);
// Called on every Notify, outside the progress lock (the driver wakes its queue workers' condition variable with it).
void SetHook(void (*hook)(void*), void* context);

struct Counters {
    std::uint64_t notifies = 0;
    std::uint64_t waits = 0;
    std::uint64_t wokenByEvent = 0;
    std::uint64_t timedOut = 0;
};
Counters Read();

}
