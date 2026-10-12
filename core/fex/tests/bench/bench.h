#ifndef CORE_FEX_TESTS_BENCH_BENCH_H
#define CORE_FEX_TESTS_BENCH_BENCH_H

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
using i16 = short;
using i32 = int;
using i64 = long long;

extern "C" {

u64 BenchNow();
void BenchWrite(const char* line);
void* BenchCopy(void* destination, const void* source, unsigned long size);
void* BenchFill(void* destination, int value, unsigned long size);
void BenchSort(void* base, unsigned long count, unsigned long size, int (*compare)(const void*, const void*));
u64 BenchTrivial();
bool BenchSpawn(void** thread, void* (*entry)(void*), void* argument);
void BenchJoin(void* thread);
[[noreturn]] void BenchExit(int status);

}

struct BenchKernel {
    const char* name;
    void (*prepare)();
    u64 (*run)(u64 work);
    u64 work;
    u32 threads;
};

extern const BenchKernel BenchKernels[];
extern const u32 BenchKernelCount;
extern u64 benchExtra;
extern u64 benchElapsed;

int BenchMain(int count, const char* const* arguments);

#endif // CORE_FEX_TESTS_BENCH_BENCH_H
