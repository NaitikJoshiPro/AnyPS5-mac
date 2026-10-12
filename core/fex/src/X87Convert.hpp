#ifndef CORE_FEX_SRC_X87CONVERT_HPP
#define CORE_FEX_SRC_X87CONVERT_HPP

#include <bit>
#include <cstdint>
#include <cstring>

namespace Aps5Fex {

// The guest's long double is the x87 80-bit format (64-bit significand with an explicit integer bit, 15-bit
// exponent); the host's is a double. The bridge converts at the call boundary: to a double rounding to nearest
// even, from a double exactly.

inline double X87ToDouble(const void* bytes) {
    std::uint64_t mantissa = 0;
    std::uint16_t signExponent = 0;
    std::memcpy(&mantissa, bytes, sizeof(mantissa));
    std::memcpy(&signExponent, static_cast<const std::uint8_t*>(bytes) + 8, sizeof(signExponent));
    const std::uint64_t sign = static_cast<std::uint64_t>(signExponent >> 15) << 63;
    const int exponent = signExponent & 0x7fff;
    if (exponent == 0x7fff) {
        if ((mantissa << 1) == 0) return std::bit_cast<double>(sign | 0x7ff0000000000000ull);
        return std::bit_cast<double>(sign | 0x7ff8000000000000ull | ((mantissa << 1) >> 12));
    }
    if (mantissa == 0) return std::bit_cast<double>(sign);
    int unbiased = (exponent == 0 ? 1 : exponent) - 16383;
    while ((mantissa >> 63) == 0) {
        mantissa <<= 1;
        --unbiased;
    }
    if (unbiased > 1023) return std::bit_cast<double>(sign | 0x7ff0000000000000ull);
    const int shift = unbiased >= -1022 ? 11 : 11 + (-1022 - unbiased);
    std::uint64_t kept = 0;
    bool up = false;
    if (shift < 64) {
        kept = mantissa >> shift;
        const std::uint64_t rest = mantissa & ((std::uint64_t {1} << shift) - 1);
        const std::uint64_t half = std::uint64_t {1} << (shift - 1);
        up = rest > half || (rest == half && (kept & 1) != 0);
    } else if (shift == 64) {
        up = mantissa > (std::uint64_t {1} << 63);
    }
    kept += up ? 1 : 0;
    if (unbiased < -1022) return std::bit_cast<double>(sign | kept);
    if (kept == (std::uint64_t {1} << 53)) {
        kept >>= 1;
        ++unbiased;
        if (unbiased > 1023) return std::bit_cast<double>(sign | 0x7ff0000000000000ull);
    }
    return std::bit_cast<double>(sign | (static_cast<std::uint64_t>(unbiased + 1023) << 52) | (kept & ((std::uint64_t {1} << 52) - 1)));
}

// Writes 16 bytes: the 10-byte value and six zero bytes.
inline void DoubleToX87(double value, void* bytes) {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    std::uint16_t signExponent = static_cast<std::uint16_t>((bits >> 63) << 15);
    const int exponent = static_cast<int>((bits >> 52) & 0x7ff);
    const std::uint64_t fraction = bits & ((std::uint64_t {1} << 52) - 1);
    std::uint64_t mantissa = 0;
    if (exponent == 0x7ff) {
        signExponent |= 0x7fff;
        mantissa = (std::uint64_t {1} << 63) | (fraction << 11);
    } else if (exponent == 0) {
        if (fraction != 0) {
            mantissa = fraction << 11;
            int unbiased = -1022;
            while ((mantissa >> 63) == 0) {
                mantissa <<= 1;
                --unbiased;
            }
            signExponent |= static_cast<std::uint16_t>(unbiased + 16383);
        }
    } else {
        mantissa = (std::uint64_t {1} << 63) | (fraction << 11);
        signExponent |= static_cast<std::uint16_t>(exponent - 1023 + 16383);
    }
    auto* out = static_cast<std::uint8_t*>(bytes);
    std::memset(out, 0, 16);
    std::memcpy(out, &mantissa, sizeof(mantissa));
    std::memcpy(out + 8, &signExponent, sizeof(signExponent));
}

}

#endif
