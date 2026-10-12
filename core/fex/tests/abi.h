// Shared by the abi_*.cpp fixtures: the exports of the test library libAps5BridgeAbi.prx
// (tests/prx/libAps5BridgeAbi/Export.cpp) and a failure bitmask reported through the exit status.
#ifndef CORE_FEX_TESTS_ABI_H
#define CORE_FEX_TESTS_ABI_H

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

struct Position {
    float x, y, z;
};

struct Mixed {
    double d;
    long long i;
};

struct Pair {
    long long a, b;
};

struct Large {
    long long a, b, c;
};

extern "C" {
int Aps5AbiHfa(void* handle, Position position, float spread, float* out, u32 count, u8 flag, float radius);
u32 Aps5AbiStack(u64* out, u8 a1, u16 a2, u8 a3, u8 a4, const void* a5, u8 a6, u64 a7, u16 a8, u16 a9, u8 a10, u32 a11, signed char a12,
                 short a13);
Mixed Aps5AbiMixed(Mixed mixed, float scale, Pair pair);
Large Aps5AbiLarge(int add, Large large);
long double Aps5AbiLongDouble(long double a, int b, long double c);
__int128 Aps5AbiInt128(int a, __int128 b, int c);
Position Aps5AbiPosition(float scale);
double Aps5AbiManyDoubles(double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9,
                          int last);
int Aps5AbiCallInts(int (*function)(int, int, int, int, int, int, int, int), int base);
double Aps5AbiCallShaped(double (*function)(Position, long double, int, float, double, double, double, double, double, double, double),
                         float scale);
Large Aps5AbiCallLarge(Large (*function)(Large, int), int add);
long double Aps5AbiCallLongDouble(long double (*function)(long double, int), int count);
void Aps5AbiStore(int (*function)(int));
int (*Aps5AbiLoad())(int);
int Aps5AbiCallStored(int value);
[[noreturn]] void exit(int);
}

static void (*volatile exitPointer)(int) = exit;
static int failures = 0;
static int firstFailure = -1;

// The exit status is 43, or 100 plus the number of the first check that failed.
static void Check(bool right, int bit) {
    if (right) return;
    failures |= 1 << bit;
    if (firstFailure < 0) firstFailure = bit;
}

// Keeps the compiler from folding a value the test passes.
template <typename T>
static T Opaque(T value) {
    asm volatile("" : "+m"(value));
    return value;
}

#define ABI_MAIN(body)                                                  \
    extern "C" [[noreturn]] void _start(void*) {                        \
        body;                                                           \
        exitPointer(failures == 0 ? 43 : 100 + firstFailure);            \
        __builtin_unreachable();                                        \
    }

#endif
