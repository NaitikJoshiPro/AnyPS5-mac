#include "bench.h"

extern "C" {

void* memcpy(void* destination, const void* source, unsigned long size);
void* memset(void* destination, int value, unsigned long size);
void qsort(void* base, unsigned long count, unsigned long size, int (*compare)(const void*, const void*));
int puts(const char* text);
[[noreturn]] void exit(int status);
u64 sceKernelGetProcessTimeCounter();
int clock_gettime(int clock, i64* time);
u64 sceKernelGetProcessTimeCounterFrequency();
int scePthreadCreate(void** thread, const void* attributes, void* (*entry)(void*), void* argument, const char* name);
int scePthreadJoin(void* thread, void** value);

// The process time counter ticks at the TSC frequency (measured at startup, near but not exactly 1 GHz):
// nanoseconds for the timings.
// A plain global, not a function static: the guest links no C++ runtime (no __cxa_guard_*).
u64 counterFrequency = 0;

u64 BenchNow() {
    if (counterFrequency == 0) counterFrequency = sceKernelGetProcessTimeCounterFrequency();
    const u64 frequency = counterFrequency;
    const u64 ticks = sceKernelGetProcessTimeCounter();
    return ticks / frequency * 1000000000ULL + ticks % frequency * 1000000000ULL / frequency;
}

u64 BenchCpuNow() {
    constexpr int ThreadCpuTime = 14;
    i64 time[2] = {0, 0};
    clock_gettime(ThreadCpuTime, time);
    return static_cast<u64>(time[0]) * 1000000000ULL + static_cast<u64>(time[1]);
}

void BenchWrite(const char* line) {
    puts(line);
}

void* BenchCopy(void* destination, const void* source, unsigned long size) {
    return memcpy(destination, source, size);
}

void* BenchFill(void* destination, int value, unsigned long size) {
    return memset(destination, value, size);
}

void BenchSort(void* base, unsigned long count, unsigned long size, int (*compare)(const void*, const void*)) {
    qsort(base, count, size, compare);
}

// One bridge call per use; the result is the native program's constant whatever the measured frequency is.
u64 BenchTrivial() {
    return sceKernelGetProcessTimeCounterFrequency() != 0 ? 1000000000ULL : 0;
}

bool BenchSpawn(void** thread, void* (*entry)(void*), void* argument) {
    return scePthreadCreate(thread, nullptr, entry, argument, "bench") == 0;
}

void BenchJoin(void* thread) {
    scePthreadJoin(thread, nullptr);
}

void BenchExit(int status) {
    exit(status);
}

[[noreturn]] void _start(void* block) {
    const auto* words = static_cast<const u64*>(block);
    BenchExit(BenchMain(static_cast<int>(words[0]), reinterpret_cast<const char* const*>(words + 1)));
}

}
