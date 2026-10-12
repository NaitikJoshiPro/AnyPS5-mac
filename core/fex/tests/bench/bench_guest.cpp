#include "bench.h"

extern "C" {

void* memcpy(void* destination, const void* source, unsigned long size);
void* memset(void* destination, int value, unsigned long size);
void qsort(void* base, unsigned long count, unsigned long size, int (*compare)(const void*, const void*));
int puts(const char* text);
[[noreturn]] void exit(int status);
u64 sceKernelGetProcessTimeCounter();
u64 sceKernelGetProcessTimeCounterFrequency();
int scePthreadCreate(void** thread, const void* attributes, void* (*entry)(void*), void* argument, const char* name);
int scePthreadJoin(void* thread, void** value);

u64 BenchNow() {
    return sceKernelGetProcessTimeCounter();
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

u64 BenchTrivial() {
    return sceKernelGetProcessTimeCounterFrequency();
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
