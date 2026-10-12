// A test-only PS5 library for the bridge: one export per argument and result class whose arm64 (Apple)
// and x86-64 SysV placements differ, plus callbacks of the same classes. The guest fixture abi.cpp calls
// them and checks what arrived. Built only with BUILD_TESTING; never staged into the build's libs.
#include <cstdint>

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

int Aps5AbiHfa(void* handle, Position position, float spread, float* out, std::uint32_t count, std::uint8_t flag, float radius) {
    out[0] = position.x;
    out[1] = position.y;
    out[2] = position.z;
    out[3] = spread;
    out[4] = static_cast<float>(count);
    out[5] = static_cast<float>(flag);
    out[6] = radius;
    return static_cast<int>(reinterpret_cast<std::uintptr_t>(handle) & 0x7fff);
}

std::uint32_t Aps5AbiStack(std::uint64_t* out, std::uint8_t a1, std::uint16_t a2, std::uint8_t a3, std::uint8_t a4, const void* a5,
                           std::uint8_t a6, std::uint64_t a7, std::uint16_t a8, std::uint16_t a9, std::uint8_t a10, std::uint32_t a11,
                           std::int8_t a12, std::int16_t a13) {
    const std::uint64_t values[] = {a1, a2, a3, a4, reinterpret_cast<std::uint64_t>(a5), a6, a7, a8, a9, a10, a11,
                                    static_cast<std::uint64_t>(static_cast<std::int64_t>(a12)),
                                    static_cast<std::uint64_t>(static_cast<std::int64_t>(a13))};
    for (std::uint32_t index = 0; index < 13; ++index) out[index] = values[index];
    return 13;
}

Mixed Aps5AbiMixed(Mixed mixed, float scale, Pair pair) {
    return {mixed.d * 2 + scale, mixed.i + pair.a + pair.b};
}

Large Aps5AbiLarge(int add, Large large) {
    return {large.a + add, large.b + add, large.c + add};
}

long double Aps5AbiLongDouble(long double a, int b, long double c) {
    return a * b + c;
}

__int128 Aps5AbiInt128(int a, __int128 b, int c) {
    return b + a + c;
}

Position Aps5AbiPosition(float scale) {
    return {scale, 2 * scale, 3 * scale};
}

double Aps5AbiManyDoubles(double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9,
                          int last) {
    return d0 + 2 * d1 + 3 * d2 + 4 * d3 + 5 * d4 + 6 * d5 + 7 * d6 + 8 * d7 + 9 * d8 + 10 * d9 + 11 * last;
}

int Aps5AbiCallInts(int (*function)(int, int, int, int, int, int, int, int), int base) {
    return function(base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6, base + 7);
}

double Aps5AbiCallShaped(double (*function)(Position, long double, int, float, double, double, double, double, double, double, double),
                         float scale) {
    return function({scale, 2 * scale, 3 * scale}, 0.5L, 7, 0.25f, 1, 2, 3, 4, 5, 6, 7);
}

Large Aps5AbiCallLarge(Large (*function)(Large, int), int add) {
    const Large large = function({1, 2, 3}, add);
    return {large.a * 10, large.b * 10, large.c * 10};
}

long double Aps5AbiCallLongDouble(long double (*function)(long double, int), int count) {
    return function(1.25L, count) * 2;
}

namespace {
int (*stored)(int) = nullptr;
}

void Aps5AbiStore(int (*function)(int)) {
    stored = function;
}

int (*Aps5AbiLoad())(int) {
    return stored;
}

int Aps5AbiCallStored(int value) {
    return stored(value);
}

}
