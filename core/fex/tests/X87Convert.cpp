// The bridge's x87 80-bit <-> double conversions (src/X87Convert.hpp): exact widening, rounding to
// nearest even when narrowing, overflow, the subnormal range and NaN payloads.
#include "X87Convert.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <source_location>

namespace {

using namespace Aps5Fex;

void Require(bool value, std::source_location where = std::source_location::current()) {
    if (!value) {
        std::fprintf(stderr, "x87 conversion check failed at line %u\n", static_cast<unsigned>(where.line()));
        std::abort();
    }
}

struct Extended {
    std::uint64_t mantissa;
    std::uint16_t signExponent;
};

double Narrow(Extended value) {
    std::uint8_t bytes[16] {};
    std::memcpy(bytes, &value.mantissa, 8);
    std::memcpy(bytes + 8, &value.signExponent, 2);
    return X87ToDouble(bytes);
}

Extended Widen(double value) {
    std::uint8_t bytes[16];
    DoubleToX87(value, bytes);
    Extended result {};
    std::memcpy(&result.mantissa, bytes, 8);
    std::memcpy(&result.signExponent, bytes + 8, 2);
    for (int index = 10; index < 16; ++index) Require(bytes[index] == 0);
    return result;
}

bool Same(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

}

int main() {
    constexpr std::uint64_t One = std::uint64_t {1} << 63;
    // Every double survives a round trip, across the normal and subnormal ranges and the specials.
    const double samples[] = {0.0, -0.0, 1.0, -2.5, 0.1, 1e300, -1e-300, std::numeric_limits<double>::denorm_min(),
                              -std::numeric_limits<double>::denorm_min() * 12345, std::numeric_limits<double>::min(),
                              std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity()};
    for (const double sample : samples) Require(Same(Narrow(Widen(sample)), sample));
    const Extended one = Widen(1.0);
    Require(one.mantissa == One && one.signExponent == 16383);
    const Extended tiny = Widen(std::numeric_limits<double>::denorm_min());
    Require(tiny.mantissa == One && tiny.signExponent == 16383 - 1074);
    // NaN keeps its quiet bit and the top of its payload.
    const double nan = std::bit_cast<double>(0x7ff8000000000123ull);
    Require(Same(Narrow(Widen(nan)), nan));
    Require(std::isnan(Narrow({One | (One >> 1), 0x7fff})));

    // Rounding to nearest, ties to even, at the 53rd bit.
    Require(Same(Narrow({One | 0x3ff, 16383}), 1.0));                      // below half an ulp
    Require(Same(Narrow({One | 0x401, 16383}), std::nextafter(1.0, 2.0)));  // above half an ulp
    Require(Same(Narrow({One | 0x400, 16383}), 1.0));                      // tie, even stays
    Require(Same(Narrow({One | 0xc00, 16383}), 1.0 + 2 * std::numeric_limits<double>::epsilon()));  // tie, odd rounds up
    Require(Same(Narrow({~std::uint64_t {0}, 16383}), 2.0));               // carries into the exponent
    // Overflow, the largest double, and underflow to the subnormal range and to zero.
    Require(Same(Narrow({One, 16383 + 1024}), std::numeric_limits<double>::infinity()));
    Require(Same(Narrow({~std::uint64_t {0}, 16383 + 1023}), std::numeric_limits<double>::infinity()));
    Require(Same(Narrow({0xfffffffffffff800ull, 16383 + 1023}), std::numeric_limits<double>::max()));
    Require(Same(Narrow({One, 16383 - 1075}), 0.0));                        // exactly half the smallest: tie to 0
    Require(Same(Narrow({One | 1, 16383 - 1075}), std::numeric_limits<double>::denorm_min()));
    Require(Same(Narrow({One, 0x8000 | (16383 - 1080)}), -0.0));
    Require(Same(Narrow({One | (One >> 1), 16383 - 1023}), std::bit_cast<double>(0x000c000000000000ull)));
    // x87 denormals and unnormals (no integer bit) are normalised first.
    Require(Same(Narrow({One >> 1, 16384}), 1.0));
    Require(Same(Narrow({0, 0x8000}), -0.0));
    std::printf("x87 conversions checked\n");
    return 0;
}
