#include "MetalBackend/MslEmitterState.hpp"
#include <array>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

// f64 operations (work package B2). MSL has no double, so binary64 is computed in software on the
// ulong bit pattern (own implementation, no third-party code): round to nearest even, denormals kept,
// correctly rounded add, mul, fma, div and sqrt. The IR semantics are the SpirvBackend's
// (SpirvAluEmitterFloat64.cpp): an operation returns its first NaN operand quieted and an invalid
// operation the RDNA default NaN 0xfff8000000000000; the front end adds RDNA's own NaN, clamp and
// output-modifier rules around them. v_dot2_f32_f16 and the interpolation/mix operations, which the
// SPIR-V path computes in f64, use the same routines.

namespace {

constexpr std::string_view Float64Text = R"(#define APS5_F64_SIGN 0x8000000000000000ul
#define APS5_F64_ABS 0x7ffffffffffffffful
#define APS5_F64_INF 0x7ff0000000000000ul
#define APS5_F64_QUIET 0x0008000000000000ul
#define APS5_F64_DEFAULT_NAN 0xfff8000000000000ul
#define APS5_F64_ONE 0x3ff0000000000000ul
#define APS5_F64_FRAC 0x000ffffffffffffful
#define APS5_F64_HIDDEN 0x0010000000000000ul

static inline bool aps5_f64_nan(ulong a) {
    return (a & APS5_F64_ABS) > APS5_F64_INF;
}

static inline bool aps5_f64_snan(ulong a) {
    return aps5_f64_nan(a) && (a & APS5_F64_QUIET) == 0ul;
}

static inline bool aps5_f64_zero(ulong a) {
    return (a & APS5_F64_ABS) == 0ul;
}

static inline uint aps5_f64_exponent(ulong a) {
    return uint(a >> 52) & 0x7ffu;
}

// Orders f64 bit patterns like their values for non-NaN operands; -0 == +0.
static inline long aps5_f64_key(ulong a) {
    const long magnitude = long(a & APS5_F64_ABS);
    return (a & APS5_F64_SIGN) != 0ul ? -magnitude : magnitude;
}

// A finite nonzero operand as sign, exponent and a significand in [2^52, 2^53):
// value = sig * 2^(exp - 1075) (denormals normalized, so exp can be below 1).
struct Aps5F64Parts {
    ulong sign;
    int exp;
    ulong sig;
};

static inline Aps5F64Parts aps5_f64_unpack(ulong a) {
    Aps5F64Parts parts;
    parts.sign = a & APS5_F64_SIGN;
    const uint exponent = aps5_f64_exponent(a);
    const ulong fraction = a & APS5_F64_FRAC;
    if (exponent != 0u) {
        parts.exp = int(exponent);
        parts.sig = fraction | APS5_F64_HIDDEN;
    } else {
        const int shift = int(clz(fraction)) - 11;
        parts.exp = 1 - shift;
        parts.sig = fraction << uint(shift);
    }
    return parts;
}

// Rounds sig * 2^(exp - 1084) to binary64. sig has its leading one at bit 62 for a normal result and
// bit 0 sticky; exp is the biased exponent minus one. Round to nearest even, or to odd when `odd`
// (truncate, set the lowest bit when inexact: an intermediate that a later rounding to <= 51 bits
// rounds correctly).
static ulong aps5_f64_round_pack(ulong sign, int exp, ulong sig, bool odd) {
    if (exp < 0) {
        sig = aps5_jam_right(sig, uint(min(-exp, 64)));
        exp = 0;
    } else if (exp >= 0x7fd) {
        if (odd) {
            if (exp > 0x7fd) return sign | 0x7feffffffffffffful;
        } else if (exp > 0x7fd || sig + 0x200ul >= 0x8000000000000000ul) {
            return sign | APS5_F64_INF;
        }
    }
    const uint roundBits = uint(sig) & 0x3ffu;
    if (odd) {
        sig = (sig >> 10) | (roundBits != 0u ? 1ul : 0ul);
    } else {
        sig = (sig + 0x200ul) >> 10;
        if (roundBits == 0x200u) sig &= ~1ul;
    }
    if (sig == 0ul) exp = 0;
    return sign + (ulong(exp) << 52) + sig;
}

// As aps5_f64_round_pack for a significand with its leading one anywhere (0 gives a signed zero).
static ulong aps5_f64_norm_round_pack(ulong sign, int exp, ulong sig, bool odd) {
    if (sig == 0ul) return sign;
    const int shift = int(clz(sig)) - 1;
    if (shift < 0) {
        sig = (sig >> 1) | (sig & 1ul);
        exp += 1;
    } else {
        sig <<= uint(shift);
        exp -= shift;
    }
    return aps5_f64_round_pack(sign, exp, sig, odd);
}

static ulong aps5_f64_add_rounded(ulong a, ulong b, bool odd) {
    const uint expA = aps5_f64_exponent(a);
    const uint expB = aps5_f64_exponent(b);
    if (expA == 0x7ffu || expB == 0x7ffu) {
        if (aps5_f64_nan(a)) return a | APS5_F64_QUIET;
        if (aps5_f64_nan(b)) return b | APS5_F64_QUIET;
        if (expA == 0x7ffu && expB == 0x7ffu && ((a ^ b) & APS5_F64_SIGN) != 0ul) return APS5_F64_DEFAULT_NAN;
        return expA == 0x7ffu ? a : b;
    }
    ulong signA = a & APS5_F64_SIGN;
    ulong signB = b & APS5_F64_SIGN;
    ulong sigA = (a & APS5_F64_FRAC) << 9;
    ulong sigB = (b & APS5_F64_FRAC) << 9;
    int eA = int(expA);
    int eB = int(expB);
    if (eA != 0) sigA |= APS5_F64_HIDDEN << 9; else eA = 1;
    if (eB != 0) sigB |= APS5_F64_HIDDEN << 9; else eB = 1;
    if (eA < eB || (eA == eB && sigA < sigB)) {
        const ulong sign = signA; signA = signB; signB = sign;
        const ulong sig = sigA; sigA = sigB; sigB = sig;
        const int exponent = eA; eA = eB; eB = exponent;
    }
    sigB = aps5_jam_right(sigB, uint(min(eA - eB, 64)));
    if (signA == signB) return aps5_f64_norm_round_pack(signA, eA, sigA + sigB, odd);
    const ulong difference = sigA - sigB;
    return difference == 0ul ? 0ul : aps5_f64_norm_round_pack(signA, eA, difference, odd);
}

static inline ulong aps5_f64_add(ulong a, ulong b) {
    return aps5_f64_add_rounded(a, b, false);
}

static ulong aps5_f64_mul(ulong a, ulong b) {
    const ulong sign = (a ^ b) & APS5_F64_SIGN;
    if (aps5_f64_exponent(a) == 0x7ffu || aps5_f64_exponent(b) == 0x7ffu) {
        if (aps5_f64_nan(a)) return a | APS5_F64_QUIET;
        if (aps5_f64_nan(b)) return b | APS5_F64_QUIET;
        if (aps5_f64_zero(a) || aps5_f64_zero(b)) return APS5_F64_DEFAULT_NAN;
        return sign | APS5_F64_INF;
    }
    if (aps5_f64_zero(a) || aps5_f64_zero(b)) return sign;
    const Aps5F64Parts x = aps5_f64_unpack(a);
    const Aps5F64Parts y = aps5_f64_unpack(b);
    const ulong high = mulhi(x.sig, y.sig);
    const ulong low = x.sig * y.sig;
    // The product lies in [2^104, 2^106): keep bits 105..43, the rest is sticky.
    const ulong sig = (high << 21) | (low >> 43) | ((low & 0x7fffffffffful) != 0ul ? 1ul : 0ul);
    return aps5_f64_norm_round_pack(sign, x.exp + y.exp - 1023, sig, false);
}

// Correctly rounded division by restoring long division (63 quotient bits and a sticky remainder).
static ulong aps5_f64_div(ulong a, ulong b) {
    const ulong sign = (a ^ b) & APS5_F64_SIGN;
    if (aps5_f64_nan(a)) return a | APS5_F64_QUIET;
    if (aps5_f64_nan(b)) return b | APS5_F64_QUIET;
    const bool infiniteA = aps5_f64_exponent(a) == 0x7ffu;
    const bool infiniteB = aps5_f64_exponent(b) == 0x7ffu;
    if (infiniteA) return infiniteB ? APS5_F64_DEFAULT_NAN : (sign | APS5_F64_INF);
    if (infiniteB) return sign;
    if (aps5_f64_zero(b)) return aps5_f64_zero(a) ? APS5_F64_DEFAULT_NAN : (sign | APS5_F64_INF);
    if (aps5_f64_zero(a)) return sign;
    const Aps5F64Parts x = aps5_f64_unpack(a);
    const Aps5F64Parts y = aps5_f64_unpack(b);
    ulong remainder = x.sig;
    int exp = x.exp - y.exp + 1022;
    if (remainder < y.sig) {
        remainder <<= 1;
        exp -= 1;
    }
    ulong quotient = 0ul;
    for (uint bit = 0u; bit < 63u; ++bit) {
        quotient <<= 1;
        if (remainder >= y.sig) {
            remainder -= y.sig;
            quotient |= 1ul;
        }
        remainder <<= 1;
    }
    return aps5_f64_norm_round_pack(sign, exp, quotient | (remainder != 0ul ? 1ul : 0ul), false);
}

// Correctly rounded square root, digit by digit (56 root bits and a sticky remainder).
static ulong aps5_f64_sqrt(ulong a) {
    if (aps5_f64_nan(a)) return a | APS5_F64_QUIET;
    if (aps5_f64_zero(a)) return a;
    if ((a & APS5_F64_SIGN) != 0ul) return APS5_F64_DEFAULT_NAN;
    if (aps5_f64_exponent(a) == 0x7ffu) return a;
    const Aps5F64Parts x = aps5_f64_unpack(a);
    int exponent = x.exp - 1075;
    ulong radicand = x.sig;
    if ((exponent & 1) != 0) {
        radicand <<= 1;
        exponent -= 1;
    }
    // root = floor(sqrt(radicand * 2^58)), two radicand bits per step from bit 111 down.
    ulong remainder = 0ul;
    ulong root = 0ul;
    for (int step = 0; step < 56; ++step) {
        const int shift = 52 - 2 * step;
        const ulong pair = shift >= 0 ? (radicand >> uint(shift)) & 3ul : 0ul;
        remainder = (remainder << 2) | pair;
        const ulong trial = (root << 2) | 1ul;
        if (remainder >= trial) {
            remainder -= trial;
            root = (root << 1) | 1ul;
        } else {
            root <<= 1;
        }
    }
    return aps5_f64_norm_round_pack(0ul, (exponent - 58) / 2 + 1084, root | (remainder != 0ul ? 1ul : 0ul), false);
}

// 128-bit helpers for the fused multiply-add.
struct Aps5U128 {
    ulong hi;
    ulong lo;
};

static inline Aps5U128 aps5_u128_shift_left(Aps5U128 value, uint distance) {
    if (distance == 0u) return value;
    if (distance >= 64u) return Aps5U128{value.lo << (distance - 64u), 0ul};
    return Aps5U128{(value.hi << distance) | (value.lo >> (64u - distance)), value.lo << distance};
}

static inline Aps5U128 aps5_u128_jam_right(Aps5U128 value, uint distance) {
    if (distance == 0u) return value;
    if (distance >= 128u) return Aps5U128{0ul, (value.hi | value.lo) != 0ul ? 1ul : 0ul};
    if (distance >= 64u) {
        const ulong lost = value.lo | (distance > 64u ? value.hi << (128u - distance) : 0ul);
        return Aps5U128{0ul, (distance > 64u ? value.hi >> (distance - 64u) : value.hi) | (lost != 0ul ? 1ul : 0ul)};
    }
    const ulong lost = value.lo << (64u - distance);
    return Aps5U128{value.hi >> distance, (value.lo >> distance) | (value.hi << (64u - distance)) | (lost != 0ul ? 1ul : 0ul)};
}

static inline Aps5U128 aps5_u128_add(Aps5U128 a, Aps5U128 b) {
    const ulong lo = a.lo + b.lo;
    return Aps5U128{a.hi + b.hi + (lo < a.lo ? 1ul : 0ul), lo};
}

static inline Aps5U128 aps5_u128_sub(Aps5U128 a, Aps5U128 b) {
    return Aps5U128{a.hi - b.hi - (a.lo < b.lo ? 1ul : 0ul), a.lo - b.lo};
}

static inline bool aps5_u128_less(Aps5U128 a, Aps5U128 b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

// (a * b + c) * 2^scale with one rounding (v_fma_f64, and v_div_fmas_f64's scaled form).
static ulong aps5_f64_fma(ulong a, ulong b, ulong c, int scale) {
    if (aps5_f64_nan(a)) return a | APS5_F64_QUIET;
    if (aps5_f64_nan(b)) return b | APS5_F64_QUIET;
    if (aps5_f64_nan(c)) return c | APS5_F64_QUIET;
    const ulong signProduct = (a ^ b) & APS5_F64_SIGN;
    const ulong signC = c & APS5_F64_SIGN;
    const bool infiniteC = aps5_f64_exponent(c) == 0x7ffu;
    if (aps5_f64_exponent(a) == 0x7ffu || aps5_f64_exponent(b) == 0x7ffu) {
        if (aps5_f64_zero(a) || aps5_f64_zero(b)) return APS5_F64_DEFAULT_NAN;
        if (infiniteC && signC != signProduct) return APS5_F64_DEFAULT_NAN;
        return signProduct | APS5_F64_INF;
    }
    if (infiniteC) return c;
    scale = clamp(scale, -4096, 4096);
    if (aps5_f64_zero(a) || aps5_f64_zero(b)) {
        if (aps5_f64_zero(c)) return signProduct & signC;
        const Aps5F64Parts z = aps5_f64_unpack(c);
        return aps5_f64_round_pack(z.sign, z.exp + scale - 1, z.sig << 10, false);
    }
    const Aps5F64Parts x = aps5_f64_unpack(a);
    const Aps5F64Parts y = aps5_f64_unpack(b);
    // Both terms with their leading one at bit 125: the larger exponent is the larger magnitude, and the
    // lowest 20 (product) or 73 (addend) bits are zero, so a shift by one stays exact.
    Aps5U128 product = aps5_u128_shift_left(Aps5U128{mulhi(x.sig, y.sig), x.sig * y.sig}, 20u);
    int productExp = x.exp + y.exp - 2170;
    if ((product.hi >> 61) == 0ul) {
        product = aps5_u128_shift_left(product, 1u);
        productExp -= 1;
    }
    Aps5U128 addend = Aps5U128{0ul, 0ul};
    int addendExp = productExp - 200;
    if (!aps5_f64_zero(c)) {
        const Aps5F64Parts z = aps5_f64_unpack(c);
        addend = Aps5U128{z.sig << 9, 0ul};
        addendExp = z.exp - 1148;
    }
    Aps5U128 large = product;
    Aps5U128 small = addend;
    ulong signLarge = signProduct;
    ulong signSmall = signC;
    int exp = productExp;
    int smallExp = addendExp;
    if (addendExp > productExp || (addendExp == productExp && aps5_u128_less(product, addend))) {
        large = addend;
        small = product;
        signLarge = signC;
        signSmall = signProduct;
        exp = addendExp;
        smallExp = productExp;
    }
    small = aps5_u128_jam_right(small, uint(min(exp - smallExp, 128)));
    const Aps5U128 sum = signLarge == signSmall ? aps5_u128_add(large, small) : aps5_u128_sub(large, small);
    if (sum.hi == 0ul && sum.lo == 0ul) return 0ul;
    const uint leading = sum.hi != 0ul ? uint(clz(sum.hi)) : 64u + uint(clz(sum.lo));
    const Aps5U128 normalized = aps5_u128_shift_left(sum, leading - 1u);
    const ulong sig = normalized.hi | (normalized.lo != 0ul ? 1ul : 0ul);
    return aps5_f64_round_pack(signLarge, exp + 1149 - int(leading) + scale, sig, false);
}

// x * 2^n with one rounding; zeros, infinities and NaNs (quieted) pass through.
static ulong aps5_f64_ldexp(ulong a, int n) {
    if (aps5_f64_nan(a)) return a | APS5_F64_QUIET;
    if (aps5_f64_zero(a) || aps5_f64_exponent(a) == 0x7ffu) return a;
    const Aps5F64Parts x = aps5_f64_unpack(a);
    return aps5_f64_round_pack(x.sign, x.exp + clamp(n, -4096, 4096) - 1, x.sig << 10, false);
}

// v_min_f64 / v_max_f64 (SpirvAluEmitterFloat64 MinMax).
static ulong aps5_f64_min_max(ulong a, ulong b, bool maxValue) {
    const bool nanA = aps5_f64_nan(a);
    const bool nanB = aps5_f64_nan(b);
    if (nanA || nanB) {
        ulong result = nanB ? a : b;
        if (aps5_f64_snan(b)) result = b | APS5_F64_QUIET;
        if (aps5_f64_snan(a)) result = a | APS5_F64_QUIET;
        return result;
    }
    if (aps5_f64_zero(a) && aps5_f64_zero(b)) {
        const bool negative = (a & APS5_F64_SIGN) != 0ul;
        return maxValue ? (negative ? b : a) : (negative ? a : b);
    }
    const bool ordered = maxValue ? aps5_f64_key(a) > aps5_f64_key(b) : aps5_f64_key(a) < aps5_f64_key(b);
    return ordered ? a : b;
}

// Clamp to [0, 1]: NaN, negatives and -0 give +0, denormals are kept.
static inline ulong aps5_f64_saturate(ulong a) {
    return a - 1ul < APS5_F64_INF ? min(a, APS5_F64_ONE) : 0ul;
}

static ulong aps5_f64_trunc(ulong a) {
    const uint exponent = aps5_f64_exponent(a);
    if (exponent >= 1075u) return aps5_f64_nan(a) ? a | APS5_F64_QUIET : a;
    if (exponent < 1023u) return a & APS5_F64_SIGN;
    return a & ~((1ul << (1075u - exponent)) - 1ul);
}

static ulong aps5_f64_floor(ulong a) {
    const ulong truncated = aps5_f64_trunc(a);
    return (a & APS5_F64_SIGN) != 0ul && truncated != a && !aps5_f64_nan(a) ? aps5_f64_add(truncated, APS5_F64_SIGN | APS5_F64_ONE) : truncated;
}

static ulong aps5_f64_ceil(ulong a) {
    const ulong truncated = aps5_f64_trunc(a);
    return (a & APS5_F64_SIGN) == 0ul && truncated != a && !aps5_f64_nan(a) ? aps5_f64_add(truncated, APS5_F64_ONE) : truncated;
}

static ulong aps5_f64_round_even(ulong a) {
    const uint exponent = aps5_f64_exponent(a);
    if (exponent >= 1075u) return aps5_f64_nan(a) ? a | APS5_F64_QUIET : a;
    const ulong sign = a & APS5_F64_SIGN;
    if (exponent < 1022u) return sign;
    if (exponent == 1022u) return (a & APS5_F64_FRAC) != 0ul ? (sign | APS5_F64_ONE) : sign;
    const uint fractionBits = 1075u - exponent;
    const ulong unit = 1ul << fractionBits;
    const ulong low = a & (unit - 1ul);
    const ulong halfway = unit >> 1;
    ulong result = a & ~(unit - 1ul);
    if (low > halfway || (low == halfway && (result & unit) != 0ul)) result += unit;
    return result;
}

static ulong aps5_f64_fract(ulong a) {
    const ulong fraction = aps5_f64_add(a, aps5_f64_floor(a) ^ APS5_F64_SIGN);
    if (aps5_f64_nan(a)) return a | APS5_F64_QUIET;
    if (aps5_f64_nan(fraction)) return APS5_F64_DEFAULT_NAN;
    return aps5_f64_key(fraction) > aps5_f64_key(0x3feffffffffffffful) ? 0x3feffffffffffffful : fraction;
}

static ulong aps5_f64_frexp_mantissa(ulong a) {
    if (aps5_f64_exponent(a) == 0x7ffu) return aps5_f64_nan(a) ? a | APS5_F64_QUIET : a;
    if (aps5_f64_zero(a)) return a;
    const Aps5F64Parts x = aps5_f64_unpack(a);
    return x.sign | (1022ul << 52) | (x.sig & APS5_F64_FRAC);
}

static uint aps5_f64_frexp_exponent(ulong a) {
    if (aps5_f64_exponent(a) == 0x7ffu || aps5_f64_zero(a)) return 0u;
    return uint(aps5_f64_unpack(a).exp - 1022);
}

constant uint aps5_two_over_pi[40] = {
    0xa2f9836eu, 0x4e441529u, 0xfc2757d1u, 0xf534ddc0u, 0xdb629599u, 0x3c439041u, 0xfe5163abu, 0xdebbc561u,
    0xb7246e3au, 0x424dd2e0u, 0x06492eeau, 0x09d1921cu, 0xfe1deb1cu, 0xb129a73eu, 0xe88235f5u, 0x2ebb4484u,
    0xe99c7026u, 0xb45f7e41u, 0x3991d639u, 0x835339f4u, 0x9c845f8bu, 0xbdf9283bu, 0x1ff897ffu, 0xde05980fu,
    0xef2f118bu, 0x5a0a6d1fu, 0x6d367ecfu, 0x27cb09b7u, 0x4f463f66u, 0x9e5fea2du, 0x7527bac7u, 0xebe5f17bu,
    0x3d0739f7u, 0x8a5292eau, 0x6bfb5fb1u, 0x1f8d5d08u, 0x56033046u, 0x00000000u, 0x00000000u, 0x00000000u,
};

// v_trig_preop_f64: 53 bits of 2/pi starting at bit segment*53 (+ the input's excess exponent), scaled
// (SpirvAluEmitterFloat64 EmitFPTrigPreop64).
static ulong aps5_f64_trig_preop(ulong a, uint segment) {
    const uint exponent = aps5_f64_exponent(a);
    const uint shift = min((segment & 31u) * 53u + (max(exponent, 1077u) - 1077u), 1201u);
    const uint word = shift >> 5;
    const uint offset = shift & 31u;
    const uint first = aps5_two_over_pi[word];
    const uint second = aps5_two_over_pi[word + 1u];
    const uint third = aps5_two_over_pi[word + 2u];
    const uint upper = offset == 0u ? first : (first << offset) | (second >> (32u - offset));
    const uint lower = offset == 0u ? second : (second << offset) | (third >> (32u - offset));
    const ulong mantissa = (ulong(upper) << 21) | ulong(lower >> 11);
    const int scaleExponent = int((exponent >= 1968u ? 1023u + 600u - 53u + 128u : 1023u + 600u - 53u) - shift);
    return aps5_f64_norm_round_pack(0ul, scaleExponent - 539, mantissa, false);
}

// f32 -> f64 exactly; a NaN keeps its payload and becomes quiet.
static ulong aps5_f32_to_f64(float value) {
    const uint bits = as_type<uint>(value);
    const ulong sign = ulong(bits & 0x80000000u) << 32;
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return sign | 0x7ff8000000000000ul | (ulong(magnitude & 0x007fffffu) << 29);
    if (magnitude == 0x7f800000u) return sign | APS5_F64_INF;
    if (magnitude == 0u) return sign;
    if (magnitude < 0x00800000u) {
        const int shift = int(clz(magnitude)) - 8;
        return sign | (ulong(897 - shift) << 52) | (ulong((magnitude << uint(shift)) & 0x007fffffu) << 29);
    }
    return sign | (ulong((magnitude >> 23) + 896u) << 52) | (ulong(magnitude & 0x007fffffu) << 29);
}

// f64 -> f32 bits, round to nearest even with f32 denormals; a NaN becomes quiet with the top of its
// payload.
static uint aps5_f64_to_f32_bits(ulong a) {
    const uint sign = uint(a >> 32) & 0x80000000u;
    const uint exponent = aps5_f64_exponent(a);
    const ulong fraction = a & APS5_F64_FRAC;
    if (exponent == 0x7ffu) return fraction != 0ul ? (sign | 0x7fc00000u | uint(fraction >> 29)) : (sign | 0x7f800000u);
    if (aps5_f64_zero(a)) return sign;
    const Aps5F64Parts x = aps5_f64_unpack(a);
    int exp32 = x.exp - 896;
    if (exp32 >= 255) return sign | 0x7f800000u;
    uint shift = 29u;
    if (exp32 < 1) {
        shift = uint(min(29 + 1 - exp32, 60));
        exp32 = 0;
    }
    const ulong kept = x.sig >> shift;
    const ulong rest = x.sig & ((1ul << shift) - 1ul);
    const ulong halfway = 1ul << (shift - 1u);
    const ulong rounded = kept + (rest > halfway || (rest == halfway && (kept & 1ul) != 0ul) ? 1ul : 0ul);
    return sign | (exp32 == 0 ? uint(rounded) : (uint(exp32 - 1) << 23) + uint(rounded));
}

static ulong aps5_u32_to_f64(uint value) {
    if (value == 0u) return 0ul;
    const int shift = int(clz(value)) + 21;
    return (ulong(1075 - shift) << 52) | ((ulong(value) << uint(shift)) & APS5_F64_FRAC);
}

static ulong aps5_i32_to_f64(uint value) {
    const ulong sign = (value & 0x80000000u) != 0u ? APS5_F64_SIGN : 0ul;
    return sign | aps5_u32_to_f64(sign != 0ul ? 0u - value : value);
}

// v_cvt_i32_f64: toward zero, saturating, NaN -> 0.
static uint aps5_f64_to_i32(ulong a) {
    if (aps5_f64_nan(a)) return 0u;
    const bool negative = (a & APS5_F64_SIGN) != 0ul;
    if ((a & APS5_F64_ABS) >= 0x41e0000000000000ul) return negative ? 0x80000000u : 0x7fffffffu;
    const uint exponent = aps5_f64_exponent(a);
    if (exponent < 1023u) return 0u;
    const uint integer = uint(((a & APS5_F64_FRAC) | APS5_F64_HIDDEN) >> (1075u - exponent));
    return negative ? 0u - integer : integer;
}

// v_cvt_u32_f64: toward zero, saturating, NaN and negatives -> 0.
static uint aps5_f64_to_u32(ulong a) {
    if (aps5_f64_nan(a) || (a & APS5_F64_SIGN) != 0ul) return 0u;
    if (a >= 0x41f0000000000000ul) return 0xffffffffu;
    const uint exponent = aps5_f64_exponent(a);
    if (exponent < 1023u) return 0u;
    return uint(((a & APS5_F64_FRAC) | APS5_F64_HIDDEN) >> (1075u - exponent));
}

// Rounds a finite value to `fractionBits` fraction bits, or to a multiple of 2^(smallestStep - 1023)
// below that format's normal range, sign kept: rounding 0 to nearest even, 1 toward +inf, 2 toward
// -inf, 3 toward zero (SpirvAluEmitterFloat64 RoundToPrecision and RoundToPrecisionDirected).
static ulong aps5_f64_round_to_precision(ulong a, int fractionBits, int smallestStep, uint rounding) {
    if (aps5_f64_zero(a) || aps5_f64_exponent(a) == 0x7ffu) return a;
    const int step = max(int(aps5_f64_exponent(a)) - fractionBits, smallestStep);
    const Aps5F64Parts x = aps5_f64_unpack(a);
    const int dropped = step - x.exp + 52;
    if (dropped <= 0) return a;
    const uint shift = uint(min(dropped, 63));
    const ulong kept = x.sig >> shift;
    const ulong rest = x.sig & ((1ul << shift) - 1ul);
    const ulong halfway = 1ul << (shift - 1u);
    const bool negative = x.sign != 0ul;
    bool up;
    if (rounding == 0u) up = rest > halfway || (rest == halfway && (kept & 1ul) != 0ul);
    else if (rounding == 1u) up = !negative && rest != 0ul;
    else if (rounding == 2u) up = negative && rest != 0ul;
    else up = false;
    const ulong multiple = kept + (up ? 1ul : 0ul);
    // multiple * 2^(step - 1023) is exact.
    return aps5_f64_norm_round_pack(x.sign, step + 61, multiple, false);
}

static inline uint aps5_flush_denormal32(uint bits, bool flush, uint smallestNormal) {
    const uint magnitude = bits & 0x7fffffffu;
    return flush && magnitude != 0u && magnitude < smallestNormal ? bits & 0x80000000u : bits;
}

// v_interp_*/v_fma_mix with an f32 or f16 result (SpirvAluEmitterFloat64 Interpolate): the product and
// sum in f64 (round to odd), then one rounding to the result precision. `mode` holds InterpolationQuiet
// (1), InterpolationFlush32 (2), InterpolationFlush16 (4) and the f16 rounding at bit 3.
static float aps5_interpolate(float deltaValue, float coordinateValue, float baseValue, bool halfResult, uint mode) {
    const bool quietNans = halfResult || (mode & 1u) != 0u;
    const bool flush32 = (mode & 2u) != 0u;
    const bool flush16 = (mode & 4u) != 0u;
    const uint delta = aps5_flush_denormal32(as_type<uint>(deltaValue), flush16, 0x38800000u);
    const uint coordinate = aps5_flush_denormal32(as_type<uint>(coordinateValue), flush32, 0x00800000u);
    const uint base = halfResult ? aps5_flush_denormal32(as_type<uint>(baseValue), flush32, 0x00800000u) : aps5_flush_denormal32(as_type<uint>(baseValue), flush16, 0x38800000u);
    const ulong product = aps5_f64_mul(aps5_f32_to_f64(as_type<float>(delta)), aps5_f32_to_f64(as_type<float>(coordinate)));
    const ulong base64 = aps5_f32_to_f64(as_type<float>(base));
    const ulong sum = aps5_f64_add_rounded(product, base64, true);
    const uint rounding16 = halfResult ? (mode >> 3) & 3u : 0u;
    ulong rounded = halfResult ? aps5_f64_round_to_precision(sum, 10, 999, rounding16) : aps5_f64_round_to_precision(sum, 23, 874, 0u);
    if (halfResult) {
        const uint high = uint(rounded >> 32);
        const uint sign = high & 0x80000000u;
        const uint magnitude = high & 0x7fffffffu;
        if (rounding16 == 0u) {
            if (magnitude > 0x40f00000u) rounded = ulong(sign | 0x40f00000u) << 32;
        } else {
            uint limit = 0x40effc00u;
            if (rounding16 == 1u) limit = sign != 0u ? 0x40effc00u : 0x40f00000u;
            if (rounding16 == 2u) limit = sign != 0u ? 0x40f00000u : 0x40effc00u;
            if (magnitude > 0x40effc00u) rounded = ulong(sign | limit) << 32;
            if (rounding16 == 2u) {
                const bool positiveZeros = product == 0ul && base64 == 0ul;
                if ((sum & APS5_F64_ABS) == 0ul && !positiveZeros) rounded = APS5_F64_SIGN;
            }
        }
    }
    const uint finite = aps5_f64_to_f32_bits(rounded);
    const ulong unbounded = aps5_f64_add(product, base64);
    uint special = aps5_f64_nan(unbounded) ? 0xffc00000u : (uint(unbounded >> 32) & 0x80000000u) | 0x7f800000u;
    const bool nonFinite = (delta & 0x7f800000u) == 0x7f800000u || (coordinate & 0x7f800000u) == 0x7f800000u || (base & 0x7f800000u) == 0x7f800000u;
    const bool zeroDelta = (delta & 0x7fffffffu) == 0u;
    const bool zeroCoordinate = (coordinate & 0x7fffffffu) == 0u;
    const bool infiniteDelta = (delta & 0x7fffffffu) == 0x7f800000u;
    const bool infiniteCoordinate = (coordinate & 0x7fffffffu) == 0x7f800000u;
    const bool invalidProduct = (infiniteDelta && zeroCoordinate) || (zeroDelta && infiniteCoordinate);
    const uint quietBit = quietNans ? 0x00400000u : 0u;
    if (aps5_f32_nan(base) && !invalidProduct) special = base | quietBit;
    if (aps5_f32_nan(coordinate)) special = coordinate | quietBit;
    if (aps5_f32_nan(delta)) special = delta | quietBit;
    const uint result = nonFinite ? special : finite;
    return as_type<float>(halfResult ? aps5_flush_denormal32(result, flush16, 0x38800000u) : aps5_flush_denormal32(result, flush32, 0x00800000u));
}

// f16 lane `shift` of a packed pair as an exact f64.
static inline ulong aps5_f16_lane_to_f64(uint packed, uint shift) {
    const uint bits = (packed >> shift) & 0xffffu;
    const uint magnitude = bits & 0x7fffu;
    const ulong sign = ulong(bits & 0x8000u) << 48;
    if (magnitude < 0x400u) return sign | aps5_f64_ldexp(aps5_u32_to_f64(magnitude), -24);
    return sign | (ulong((magnitude << 10) + (1008u << 20)) << 32);
}

static inline uint aps5_quiet_half_as_f32(uint half16) {
    return ((half16 & 0x8000u) << 16) | 0x7fc00000u | ((half16 & 0x3ffu) << 13);
}

static inline void aps5_two_sum64(ulong a, ulong b, thread ulong& sum, thread ulong& error) {
    sum = aps5_f64_add(a, b);
    const ulong bPart = aps5_f64_add(sum, a ^ APS5_F64_SIGN);
    const ulong aPart = aps5_f64_add(sum, bPart ^ APS5_F64_SIGN);
    error = aps5_f64_add(aps5_f64_add(a, aPart ^ APS5_F64_SIGN), aps5_f64_add(b, bPart ^ APS5_F64_SIGN));
}

// v_dot2_f32_f16 with one rounding (SpirvAluEmitterFloat64 Dot2F32F16Exact): exact f16 products, an
// f32 accumulator whose denormals are flushed, and the RDNA special-value rules.
static uint aps5_dot2_f32_f16_exact(uint a, uint b, uint c) {
    const ulong aLow = aps5_f16_lane_to_f64(a, 0u);
    const ulong bLow = aps5_f16_lane_to_f64(b, 0u);
    const ulong aHigh = aps5_f16_lane_to_f64(a, 16u);
    const ulong bHigh = aps5_f16_lane_to_f64(b, 16u);
    const uint magnitudeC = c & 0x7fffffffu;
    const uint signC = c & 0x80000000u;
    const bool nanC = magnitudeC > 0x7f800000u;
    const bool infiniteC = magnitudeC == 0x7f800000u;
    const ulong valueC = magnitudeC < 0x00800000u ? ulong(signC) << 32 : aps5_f32_to_f64(as_type<float>(c));
    const ulong productLow = aps5_f64_mul(aLow, bLow);
    const ulong productHigh = aps5_f64_mul(aHigh, bHigh);
    ulong upperSum, upperError, totalSum, totalError;
    aps5_two_sum64(productHigh, valueC, upperSum, upperError);
    aps5_two_sum64(productLow, upperSum, totalSum, totalError);
    const ulong sum = aps5_f64_add_rounded(totalSum, aps5_f64_add_rounded(totalError, upperError, true), true);
    uint result = aps5_f64_to_f32_bits(sum);
    const uint lowA = a & 0xffffu, lowB = b & 0xffffu, highA = a >> 16, highB = b >> 16;
    const uint signLow = ((lowA ^ lowB) & 0x8000u) << 16;
    const uint signHigh = ((highA ^ highB) & 0x8000u) << 16;
    if ((sum & APS5_F64_ABS) == 0ul) result = signLow & signHigh & signC;
    const bool zeroLowA = (lowA & 0x7fffu) == 0u, zeroLowB = (lowB & 0x7fffu) == 0u;
    const bool zeroHighA = (highA & 0x7fffu) == 0u, zeroHighB = (highB & 0x7fffu) == 0u;
    const bool infLowA = (lowA & 0x7fffu) == 0x7c00u, infLowB = (lowB & 0x7fffu) == 0x7c00u;
    const bool infHighA = (highA & 0x7fffu) == 0x7c00u, infHighB = (highB & 0x7fffu) == 0x7c00u;
    const bool infiniteLow = (infLowA && !zeroLowB) || (infLowB && !zeroLowA);
    const bool infiniteHigh = (infHighA && !zeroHighB) || (infHighB && !zeroHighA);
    bool invalid = (infLowA && zeroLowB) || (infLowB && zeroLowA) || (infHighA && zeroHighB) || (infHighB && zeroHighA);
    invalid = invalid || (infiniteLow && infiniteHigh && signLow != signHigh);
    invalid = invalid || (infiniteLow && infiniteC && signLow != signC);
    invalid = invalid || (infiniteHigh && infiniteC && signHigh != signC);
    if (infiniteLow || infiniteHigh || infiniteC) result = (infiniteLow ? signLow : (infiniteHigh ? signHigh : signC)) | 0x7f800000u;
    if (invalid) result = 0xffc00000u;
    if (nanC) result = c | 0x00400000u;
    if ((highB & 0x7fffu) > 0x7c00u) result = aps5_quiet_half_as_f32(highB);
    if ((highA & 0x7fffu) > 0x7c00u) result = aps5_quiet_half_as_f32(highA);
    if ((lowB & 0x7fffu) > 0x7c00u) result = aps5_quiet_half_as_f32(lowB);
    if ((lowA & 0x7fffu) > 0x7c00u) result = aps5_quiet_half_as_f32(lowA);
    return result;
}

static inline float aps5_dot2_half(uint packed, uint shift, thread bool& finite) {
    const uint bits = (packed >> shift) & 0xffffu;
    const uint magnitude = bits & 0x7fffu;
    const uint sign = (bits & 0x8000u) << 16;
    const uint normal = (magnitude << 13) + (112u << 23);
    const uint subnormal = as_type<uint>(float(magnitude) * as_type<float>(0x33800000u));
    finite = finite && magnitude < 0x7c00u;
    return as_type<float>((magnitude < 0x400u ? subnormal : normal) | sign);
}

static inline void aps5_two_sum32(float a, float b, thread float& sum, thread float& error) {
    sum = a + b;
    const float bPart = sum - a;
    const float aPart = sum - bPart;
    error = (a - aPart) + (b - bPart);
}

// v_dot2_f32_f16 (SpirvAluEmitterFloat64 EmitFPDot2F32F16): an f32 evaluation with a round-to-odd
// correction while every term stays well inside the normal range, else the exact routine.
static uint aps5_dot2_f32_f16(uint a, uint b, uint c) {
    const uint magnitudeC = c & 0x7fffffffu;
    const bool flushedC = magnitudeC < 0x00800000u;
    bool fast = magnitudeC < 0x7f800000u && (flushedC || magnitudeC >= 0x0d800000u);
    const float valueC = as_type<float>(flushedC ? (c & 0x80000000u) : c);
    const float productLow = aps5_dot2_half(a, 0u, fast) * aps5_dot2_half(b, 0u, fast);
    const float productHigh = aps5_dot2_half(a, 16u, fast) * aps5_dot2_half(b, 16u, fast);
    float upperSum, upperError, totalSum, totalError, nearest, error;
    aps5_two_sum32(productHigh, valueC, upperSum, upperError);
    aps5_two_sum32(productLow, upperSum, totalSum, totalError);
    aps5_two_sum32(totalError, upperError, nearest, error);
    const uint nearestBits = as_type<uint>(nearest);
    const uint errorBits = as_type<uint>(error);
    const bool inexact = (errorBits & 0x7fffffffu) != 0u;
    const bool even = (nearestBits & 1u) == 0u;
    const bool sameSign = ((nearestBits ^ errorBits) & 0x80000000u) == 0u;
    const uint odd = nearestBits + (sameSign ? 1u : 0xffffffffu);
    const float roundedToOdd = as_type<float>(inexact && even ? odd : nearestBits);
    const uint result = as_type<uint>(totalSum + roundedToOdd);
    const uint magnitude = result & 0x7fffffffu;
    fast = fast && magnitude >= 0x0d800000u && magnitude < 0x7f000000u;
    return fast ? result : aps5_dot2_f32_f16_exact(a, b, c);
}

)";

constexpr std::array<MslPreludeChunk, 2> OnAluFloat16{MslPreludeChunk::Alu, MslPreludeChunk::Float16};

void Call(MslEmitterState& state, const IrValue& instruction, std::string_view function, std::string_view extra = {}) {
    state.RequirePrelude(MslPreludeChunk::Float64);
    std::string expression = std::string(function) + "(";
    for (std::size_t index = 0; index < instruction.ArgumentCount(); ++index) expression += (index == 0u ? "" : ", ") + state.Arg(instruction, index);
    state.Define(instruction, expression + std::string(extra) + ")");
}

void Interpolate(MslEmitterState& state, const IrValue& instruction, bool halfResult) {
    const auto* mode = instruction.Argument(3u)->Resolve();
    if (!mode->HasImmediate()) state.Fail(instruction, "needs an immediate interpolation mode (package B2)");
    state.RequirePrelude(MslPreludeChunk::Float64);
    state.Define(instruction, "aps5_interpolate(" + state.Arg(instruction, 0u) + ", " + state.Arg(instruction, 1u) + ", " + state.Arg(instruction, 2u) + ", " + (halfResult ? "true" : "false") + ", " + MslHex(mode->ImmediateU32()) + "u)");
}

}

bool EmitMslFloat64Instruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::FPAdd64:
        Call(state, instruction, "aps5_f64_add");
        return true;
    case IrOpcode::FPMul64:
        Call(state, instruction, "aps5_f64_mul");
        return true;
    case IrOpcode::FPFma64:
        Call(state, instruction, "aps5_f64_fma", ", 0");
        return true;
    case IrOpcode::FPFmaScale64:
        state.RequirePrelude(MslPreludeChunk::Float64);
        state.Define(instruction, "aps5_f64_fma(" + state.Arg(instruction, 0u) + ", " + state.Arg(instruction, 1u) + ", " + state.Arg(instruction, 2u) + ", as_type<int>(" + state.Arg(instruction, 3u) + "))");
        return true;
    case IrOpcode::FPMin64:
        Call(state, instruction, "aps5_f64_min_max", ", false");
        return true;
    case IrOpcode::FPMax64:
        Call(state, instruction, "aps5_f64_min_max", ", true");
        return true;
    case IrOpcode::FPSaturate64:
        Call(state, instruction, "aps5_f64_saturate");
        return true;
    case IrOpcode::FPLdexp64:
        state.RequirePrelude(MslPreludeChunk::Float64);
        state.Define(instruction, "aps5_f64_ldexp(" + state.Arg(instruction, 0u) + ", as_type<int>(" + state.Arg(instruction, 1u) + "))");
        return true;
    case IrOpcode::FPRoundEven64:
        Call(state, instruction, "aps5_f64_round_even");
        return true;
    case IrOpcode::FPFloor64:
        Call(state, instruction, "aps5_f64_floor");
        return true;
    case IrOpcode::FPCeil64:
        Call(state, instruction, "aps5_f64_ceil");
        return true;
    case IrOpcode::FPTrunc64:
        Call(state, instruction, "aps5_f64_trunc");
        return true;
    case IrOpcode::FPFract64:
        Call(state, instruction, "aps5_f64_fract");
        return true;
    case IrOpcode::FPFrexpMant64:
        Call(state, instruction, "aps5_f64_frexp_mantissa");
        return true;
    case IrOpcode::FPFrexpExp64:
        Call(state, instruction, "aps5_f64_frexp_exponent");
        return true;
    case IrOpcode::FPRcp64:
        state.RequirePrelude(MslPreludeChunk::Float64);
        state.Define(instruction, "aps5_f64_div(APS5_F64_ONE, " + state.Arg(instruction, 0u) + ")");
        return true;
    case IrOpcode::FPRsq64:
        state.RequirePrelude(MslPreludeChunk::Float64);
        state.Define(instruction, "aps5_f64_div(APS5_F64_ONE, aps5_f64_sqrt(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::FPSqrt64:
        Call(state, instruction, "aps5_f64_sqrt");
        return true;
    case IrOpcode::FPTrigPreop64:
        Call(state, instruction, "aps5_f64_trig_preop");
        return true;
    case IrOpcode::FPDot2F32F16:
        Call(state, instruction, "aps5_dot2_f32_f16");
        return true;
    case IrOpcode::FPInterpolateF32:
        Interpolate(state, instruction, false);
        return true;
    case IrOpcode::FPInterpolateF16:
        Interpolate(state, instruction, true);
        return true;
    case IrOpcode::ConvertF32F64:
        state.RequirePrelude(MslPreludeChunk::Float64);
        state.Define(instruction, "as_type<float>(aps5_f64_to_f32_bits(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::ConvertF64F32:
        Call(state, instruction, "aps5_f32_to_f64");
        return true;
    case IrOpcode::ConvertF64S32:
        Call(state, instruction, "aps5_i32_to_f64");
        return true;
    case IrOpcode::ConvertF64U32:
        Call(state, instruction, "aps5_u32_to_f64");
        return true;
    case IrOpcode::ConvertS32F64:
        Call(state, instruction, "aps5_f64_to_i32");
        return true;
    case IrOpcode::ConvertU32F64:
        Call(state, instruction, "aps5_f64_to_u32");
        return true;
    default:
        return false;
    }
}

MslPreludeEntry MslFloat64Prelude(MslPreludeChunk chunk) {
    if (chunk != MslPreludeChunk::Float64) throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not the Float64 chunk (package B2)");
    return {Float64Text, OnAluFloat16};
}

}
