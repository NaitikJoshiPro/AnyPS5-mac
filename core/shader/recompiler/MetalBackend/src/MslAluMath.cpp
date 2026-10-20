#include "MetalBackend/MslEmitterState.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler {

// Integer, logic, compare, select, composite, conversion and f32/f16 operations (work package B2).
// The reference semantics are the SpirvBackend's (SpirvAluEmitterMath.cpp, SpirvModuleEmitter.cpp):
// the front end already models RDNA2's NaN payloads, output modifiers, clamps and FLOAT_MODE flushes
// with explicit IR, so each IR operation here is plain IEEE arithmetic plus the RDNA-specific helpers
// the SPIR-V path implements in its emitter (min/max/med3 zero and NaN rules, class masks, the
// transcendental series, f16 conversions on bits).
//
// Apple GPU facts this file is built on (probe .mac-work/research/metal-probes/probe_float_b2.swift,
// M3, macOS 27.0.1, MTLMathModeSafe + precise functions):
// - f32 denormals are flushed to signed zero on input and output of every float operation, compares
//   and min/max included; f16 denormals are kept; float->half conversion rounds to nearest even with
//   half denormals; half arithmetic rounds to nearest even.
// - NaN results are the canonical 0x7fc00000 (payloads are lost), so NaN payloads stay on bits.
// - float->int conversions saturate and map NaN to 0 (as v_cvt_i32_f32 / v_cvt_u32_f32).
// - `#pragma METAL fp contract(off)` keeps a*b+c two roundings; fma() is fused.
// Shift counts are masked to the operand width, as RDNA2 hardware does. U64 is MSL ulong.

namespace {

constexpr std::string_view AluText = R"(#pragma METAL fp contract(off)

static inline bool aps5_f32_nan(uint bits) {
    return (bits & 0x7fffffffu) > 0x7f800000u;
}

// Orders f32 bit patterns like their values (exact for denormals, which float compares flush); -0 == +0.
static inline int aps5_f32_key(uint bits) {
    const int magnitude = int(bits & 0x7fffffffu);
    return (bits & 0x80000000u) != 0u ? -magnitude : magnitude;
}

// v_min_f32 / v_max_f32 (SpirvModuleEmitter EmitMinMaxF32Value): a NaN rhs gives lhs, a NaN lhs gives
// rhs, min(-0, +0) = -0 and max(-0, +0) = +0.
static inline float aps5_fmin(float lhs, float rhs) {
    const uint l = as_type<uint>(lhs);
    const uint r = as_type<uint>(rhs);
    if (aps5_f32_nan(r)) return lhs;
    if (aps5_f32_nan(l)) return rhs;
    if (((l | r) & 0x7fffffffu) == 0u) return as_type<float>(l | r);
    return aps5_f32_key(l) < aps5_f32_key(r) ? lhs : rhs;
}

static inline float aps5_fmax(float lhs, float rhs) {
    const uint l = as_type<uint>(lhs);
    const uint r = as_type<uint>(rhs);
    if (aps5_f32_nan(r)) return lhs;
    if (aps5_f32_nan(l)) return rhs;
    if (((l | r) & 0x7fffffffu) == 0u) return as_type<float>(l & r);
    return aps5_f32_key(l) >= aps5_f32_key(r) ? lhs : rhs;
}

// One step of v_med3_f32 (SpirvAluEmitterMath EmitFPMedTri32 `numeric`): ordered compares, so a NaN
// operand selects rhs.
static inline uint aps5_med3_pick(uint lhs, uint rhs, bool maxValue) {
    const uint either = lhs | rhs;
    if ((either & 0x7fffffffu) == 0u) return maxValue ? (lhs & rhs) : either;
    if (aps5_f32_nan(lhs) || aps5_f32_nan(rhs)) return rhs;
    const bool ordered = maxValue ? aps5_f32_key(lhs) >= aps5_f32_key(rhs) : aps5_f32_key(lhs) < aps5_f32_key(rhs);
    return ordered ? lhs : rhs;
}

static inline float aps5_fmed3(float a, float b, float c) {
    const uint x = as_type<uint>(a);
    const uint y = as_type<uint>(b);
    const uint z = as_type<uint>(c);
    const bool nanY = aps5_f32_nan(y);
    const bool nanZ = aps5_f32_nan(z);
    const uint minXy = nanY ? x : aps5_med3_pick(x, y, false);
    if (aps5_f32_nan(x) || nanY || nanZ) return as_type<float>(nanZ ? minXy : aps5_med3_pick(minXy, z, false));
    return as_type<float>(aps5_med3_pick(minXy, aps5_med3_pick(aps5_med3_pick(x, y, true), z, false), true));
}

// v_cmp_class_f32 mask bits: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal,
// 8 +normal, 9 +inf.
static inline bool aps5_f32_class(float value, uint mask) {
    const uint bits = as_type<uint>(value);
    const uint magnitude = bits & 0x7fffffffu;
    const bool negative = (bits & 0x80000000u) != 0u;
    const uint exponent = magnitude & 0x7f800000u;
    const uint mantissa = magnitude & 0x007fffffu;
    uint classes;
    if (exponent == 0x7f800000u) classes = mantissa != 0u ? ((mantissa & 0x00400000u) != 0u ? 0x2u : 0x1u) : (negative ? 0x4u : 0x200u);
    else if (exponent != 0u) classes = negative ? 0x8u : 0x100u;
    else if (mantissa != 0u) classes = negative ? 0x10u : 0x80u;
    else classes = negative ? 0x20u : 0x40u;
    return (classes & mask) != 0u;
}

// Clamp to [0, 1] on bits: NaN, negatives and -0 give +0, denormals are kept.
static inline float aps5_fsat(float value) {
    const uint bits = as_type<uint>(value);
    return as_type<float>(bits - 1u < 0x7f800000u ? min(bits, 0x3f800000u) : 0u);
}

static inline float aps5_fabs(float value) {
    return as_type<float>(as_type<uint>(value) & 0x7fffffffu);
}

static inline bool aps5_f32_denormal(uint bits) {
    return (bits & 0x7fffffffu) - 1u < 0x007fffffu;
}

// Shift right, ORing every bit shifted out into bit 0 (sticky).
static inline ulong aps5_jam_right(ulong a, uint distance) {
    if (distance == 0u) return a;
    if (distance >= 64u) return a != 0ul ? 1ul : 0ul;
    return (a >> distance) | ((a << (64u - distance)) != 0ul ? 1ul : 0ul);
}

// A finite nonzero f32 as m * 2^(exponent - 150), m in [2^23, 2^24) (denormals normalized).
static inline uint aps5_f32_unpack(uint bits, thread int& exponent) {
    const uint field = (bits >> 23) & 0xffu;
    const uint fraction = bits & 0x007fffffu;
    if (field != 0u) {
        exponent = int(field);
        return fraction | 0x00800000u;
    }
    const int shift = int(clz(fraction)) - 8;
    exponent = 1 - shift;
    return fraction << uint(shift);
}

// sig * 2^exp (sig != 0) rounded to f32: nearest even, denormals kept, overflow to infinity.
static float aps5_f32_round_pack(uint sign, int exp, ulong sig) {
    const int leading = 63 - int(clz(sig));
    const int exponent = exp + leading;
    if (exponent > 127) return as_type<float>(sign | 0x7f800000u);
    const int shift = exponent >= -126 ? leading - 23 : -(exp + 149);
    ulong mantissa;
    if (shift <= 0) {
        mantissa = sig << uint(-shift);
    } else {
        if (shift >= 64) return as_type<float>(sign);
        const ulong kept = sig >> uint(shift);
        const ulong rest = sig & ((1ul << uint(shift)) - 1ul);
        const ulong halfway = 1ul << uint(shift - 1);
        mantissa = kept + (rest > halfway || (rest == halfway && (kept & 1ul) != 0ul) ? 1ul : 0ul);
    }
    if (exponent < -126) return as_type<float>(sign | uint(mantissa));
    return as_type<float>(sign | ((uint(exponent + 126) << 23) + uint(mantissa)));
}

// IEEE f32 fused multiply-add on bits (nearest even, denormals kept), for what the GPU's f32 flushing
// gets wrong: denormal operands and results below 2^-126. A NaN result comes from the GPU (the front
// end chooses NaN payloads itself).
static float aps5_f32_fma_exact(float a, float b, float c) {
    const uint x = as_type<uint>(a);
    const uint y = as_type<uint>(b);
    const uint z = as_type<uint>(c);
    if (aps5_f32_nan(x) || aps5_f32_nan(y) || aps5_f32_nan(z)) return fma(a, b, c);
    const uint signProduct = (x ^ y) & 0x80000000u;
    const uint signC = z & 0x80000000u;
    const uint magnitudeX = x & 0x7fffffffu;
    const uint magnitudeY = y & 0x7fffffffu;
    const uint magnitudeZ = z & 0x7fffffffu;
    if (magnitudeX == 0x7f800000u || magnitudeY == 0x7f800000u) {
        if (magnitudeX == 0u || magnitudeY == 0u || (magnitudeZ == 0x7f800000u && signC != signProduct)) return as_type<float>(0xffc00000u);
        return as_type<float>(signProduct | 0x7f800000u);
    }
    if (magnitudeZ == 0x7f800000u) return c;
    if (magnitudeX == 0u || magnitudeY == 0u) return magnitudeZ == 0u ? as_type<float>(signProduct & signC) : c;
    int exponentX;
    int exponentY;
    const uint mantissaX = aps5_f32_unpack(x, exponentX);
    const uint mantissaY = aps5_f32_unpack(y, exponentY);
    // value = sig * 2^exp with the leading one of both terms at bit 61: the lowest 13 (product) or 38
    // (addend) bits are zero, so aligning by one bit stays exact where the terms can cancel.
    ulong product = (ulong(mantissaX) * ulong(mantissaY)) << 14;
    int productExp = exponentX + exponentY - 314;
    if ((product >> 61) == 0ul) {
        product <<= 1;
        productExp -= 1;
    }
    ulong addend = 0ul;
    int addendExp = productExp - 100;
    if (magnitudeZ != 0u) {
        int exponentZ;
        addend = ulong(aps5_f32_unpack(z, exponentZ)) << 38;
        addendExp = exponentZ - 188;
    }
    ulong large = product;
    ulong small = addend;
    uint signLarge = signProduct;
    uint signSmall = signC;
    int exp = productExp;
    int smallExp = addendExp;
    if (addendExp > productExp || (addendExp == productExp && addend > product)) {
        large = addend;
        small = product;
        signLarge = signC;
        signSmall = signProduct;
        exp = addendExp;
        smallExp = productExp;
    }
    small = aps5_jam_right(small, uint(min(exp - smallExp, 64)));
    const ulong sum = signLarge == signSmall ? large + small : large - small;
    return sum == 0ul ? 0.0f : aps5_f32_round_pack(signLarge, exp, sum);
}

// v_mul_f32 / v_fma_f32: RDNA detects tininess after rounding (a product just below 2^-126 that
// rounds to it is kept), the GPU before rounding (it flushes it), and the GPU flushes denormal
// operands: a zero result from nonzero operands is recomputed exactly.
static inline float aps5_fmul(float a, float b) {
    const float r = a * b;
    return r == 0.0f ? aps5_f32_fma_exact(a, b, -0.0f) : r;
}

static inline float aps5_fma(float a, float b, float c) {
    const float r = fma(a, b, c);
    return r == 0.0f ? aps5_f32_fma_exact(a, b, c) : r;
}

// v_mul_f32 in FLOAT_MODE f32 flush, round to nearest even (FPMulNativeFlushFlag, DECISIONS D54):
// the GPU flushes denormal operands and results itself and detects tininess before rounding, RDNA
// after rounding, so of the products the GPU flushes (|exact| < 2^-126) RDNA keeps those whose
// exact value rounds to +-2^-126 at an unbounded exponent. Here |a| < 1 (both operands are normal:
// |a * b| < 2^-126 with |b| >= 2^-126), so a * 2^64 is exact and the scaled product rounds as RDNA.
static inline float aps5_fmul_ftz(float a, float b) {
    const float r = a * b;
    if (r != 0.0f) return r;
    const uint scaled = as_type<uint>((a * as_type<float>(0x5f800000u)) * b);
    return (scaled & 0x7fffffffu) == 0x20800000u ? as_type<float>((scaled & 0x80000000u) | 0x00800000u) : r;
}

// The fused multiply-add of aps5_fmul_ftz (FPMulNativeFlushFlag on FPFma32): the GPU flushes
// operands and results itself; a zero result is recomputed as the front end's emulation does
// (TranslationContext::flushTinyProduct and the result flush around aps5_fma): with a zero addend a
// product tiny after rounding at an unbounded exponent is a signed zero, otherwise the exact result
// rounded at the denormal precision is kept when it is normal (+-2^-126) and flushed when it is not.
static inline float aps5_fma_ftz(float a, float b, float c) {
    const float r = fma(a, b, c);
    if (r != 0.0f) return r;
    const uint ua = as_type<uint>(a);
    const uint ub = as_type<uint>(b);
    const uint uc = as_type<uint>(c);
    const float fa = (ua & 0x7f800000u) == 0u ? as_type<float>(ua & 0x80000000u) : a;
    const float fb = (ub & 0x7f800000u) == 0u ? as_type<float>(ub & 0x80000000u) : b;
    const float fc = (uc & 0x7f800000u) == 0u ? as_type<float>(uc & 0x80000000u) : c;
    const uint scaled = as_type<uint>(aps5_fmul(aps5_fmul(fa, as_type<float>(0x5f800000u)), fb)) & 0x7fffffffu;
    if ((as_type<uint>(fc) & 0x7fffffffu) == 0u && scaled - 1u < 0x207fffffu) return as_type<float>((as_type<uint>(fa) ^ as_type<uint>(fb)) & 0x80000000u);
    const uint exact = as_type<uint>(aps5_f32_fma_exact(fa, fb, fc));
    return as_type<float>((exact & 0x7f800000u) == 0u ? exact & 0x80000000u : exact);
}

// FLOAT_MODE keeps f32 denormals: exact unless every operand and the result are normal.
static inline float aps5_fmul_keep(float a, float b) {
    const float r = a * b;
    return aps5_f32_denormal(as_type<uint>(a)) || aps5_f32_denormal(as_type<uint>(b)) || r == 0.0f ? aps5_f32_fma_exact(a, b, -0.0f) : r;
}

static inline float aps5_fadd_keep(float a, float b) {
    const float r = a + b;
    return aps5_f32_denormal(as_type<uint>(a)) || aps5_f32_denormal(as_type<uint>(b)) || (as_type<uint>(r) & 0x7f800000u) == 0u ? aps5_f32_fma_exact(a, 1.0f, b) : r;
}

static inline float aps5_fma_keep(float a, float b, float c) {
    const float r = fma(a, b, c);
    return aps5_f32_denormal(as_type<uint>(a)) || aps5_f32_denormal(as_type<uint>(b)) || aps5_f32_denormal(as_type<uint>(c)) || r == 0.0f ? aps5_f32_fma_exact(a, b, c) : r;
}

// Compares on bits (exact for denormals, which float compares flush), for FLOAT_MODE keeping them.
static inline bool aps5_f32_ordered(float a, float b) {
    return !aps5_f32_nan(as_type<uint>(a)) && !aps5_f32_nan(as_type<uint>(b));
}

static inline bool aps5_feq(float a, float b) {
    return aps5_f32_ordered(a, b) && aps5_f32_key(as_type<uint>(a)) == aps5_f32_key(as_type<uint>(b));
}

static inline bool aps5_fne(float a, float b) {
    return aps5_f32_ordered(a, b) && aps5_f32_key(as_type<uint>(a)) != aps5_f32_key(as_type<uint>(b));
}

static inline bool aps5_flt(float a, float b) {
    return aps5_f32_ordered(a, b) && aps5_f32_key(as_type<uint>(a)) < aps5_f32_key(as_type<uint>(b));
}

static inline bool aps5_fle(float a, float b) {
    return aps5_f32_ordered(a, b) && aps5_f32_key(as_type<uint>(a)) <= aps5_f32_key(as_type<uint>(b));
}

static inline bool aps5_fgt(float a, float b) {
    return aps5_f32_ordered(a, b) && aps5_f32_key(as_type<uint>(a)) > aps5_f32_key(as_type<uint>(b));
}

static inline bool aps5_fge(float a, float b) {
    return aps5_f32_ordered(a, b) && aps5_f32_key(as_type<uint>(a)) >= aps5_f32_key(as_type<uint>(b));
}

// floor/ceil of a denormal (the GPU flushes it to a signed zero first), FLOAT_MODE keeping them.
static inline float aps5_floor_keep(float value) {
    const uint bits = as_type<uint>(value);
    return aps5_f32_denormal(bits) ? ((bits & 0x80000000u) != 0u ? -1.0f : 0.0f) : floor(value);
}

static inline float aps5_ceil_keep(float value) {
    const uint bits = as_type<uint>(value);
    return aps5_f32_denormal(bits) ? ((bits & 0x80000000u) != 0u ? -0.0f : 1.0f) : ceil(value);
}

// GLSL fract on |x| (x - floor(x), no clamp below 1) and the RDNA sign rule (SpirvAluEmitterMath
// EmitFPFract32).
static inline float aps5_fract(float value) {
    const float magnitude = aps5_fabs(value);
    const float fraction = magnitude - floor(magnitude);
    return value < 0.0f && fraction > 0.0f ? 1.0f - fraction : fraction;
}

// FLOAT_MODE keeping denormals: the fraction of a denormal is itself (1 - |x| rounds to 1 below zero).
static inline float aps5_fract_keep(float value) {
    const uint bits = as_type<uint>(value);
    return aps5_f32_denormal(bits) ? ((bits & 0x80000000u) != 0u ? 1.0f : value) : aps5_fract(value);
}

// v_log_f32 near 1 (SpirvAluEmitterMath EmitFPLog2): the host log2 loses accuracy there, a series in
// (x - 1) replaces it when the two differ by more than 4 ulp. No denormal flush of the input (upstream
// #4705): the front end's unaryFloatSpecials replaces the result of every input below 2^-126.
static inline float aps5_log2(float value) {
    const float source = value;
    const float result = log2(source);
    const float difference = source - 1.0f;
    float polynomial = as_type<float>(0xbe000000u);
    polynomial = fma(polynomial, difference, as_type<float>(0x3e124925u));
    polynomial = fma(polynomial, difference, as_type<float>(0xbe2aaaabu));
    polynomial = fma(polynomial, difference, as_type<float>(0x3e4ccccdu));
    polynomial = fma(polynomial, difference, as_type<float>(0xbe800000u));
    polynomial = fma(polynomial, difference, as_type<float>(0x3eaaaaabu));
    polynomial = fma(polynomial, difference, as_type<float>(0xbf000000u));
    polynomial = fma(polynomial, difference, as_type<float>(0x3f800000u));
    const float product = difference * polynomial;
    const float series = product * as_type<float>(0x3fb8aa3bu);
    const uint valueBits = as_type<uint>(result);
    const uint seriesBits = as_type<uint>(series);
    const bool near = aps5_fabs(difference) <= 0.125f;
    return near && min(valueBits - seriesBits, seriesBits - valueBits) > 4u ? series : result;
}

// v_sin_f32 / v_cos_f32 take turns (SpirvAluEmitterMath EmitFPSin / EmitFPCos): the host function
// of the reduced cycle, replaced by turn series where they differ by more than 4 ulp and by the
// linear term next to zero.
static inline float aps5_trig_cycle(float value, bool preserveSignedZero) {
    const float reduced = value - rint(value);
    return preserveSignedZero && (as_type<uint>(value) & 0x7fffffffu) == 0u ? value : reduced;
}

static inline float aps5_sin_turns(float turns) {
    const float square = turns * turns;
    float polynomial = as_type<float>(0xc17183a8u);
    polynomial = fma(polynomial, square, as_type<float>(0x42283c1au));
    polynomial = fma(polynomial, square, as_type<float>(0xc2996966u));
    polynomial = fma(polynomial, square, as_type<float>(0x42a335e3u));
    polynomial = fma(polynomial, square, as_type<float>(0xc2255de7u));
    polynomial = fma(polynomial, square, as_type<float>(0x40c90fdbu));
    return turns * polynomial;
}

static inline float aps5_cos_turns(float turns) {
    const float square = turns * turns;
    float polynomial = as_type<float>(0x40fce9c5u);
    polynomial = fma(polynomial, square, as_type<float>(0xc1d368f9u));
    polynomial = fma(polynomial, square, as_type<float>(0x4270fa83u));
    polynomial = fma(polynomial, square, as_type<float>(0xc2aae9e4u));
    polynomial = fma(polynomial, square, as_type<float>(0x4281e0f8u));
    polynomial = fma(polynomial, square, as_type<float>(0xc19de9e6u));
    polynomial = fma(polynomial, square, as_type<float>(0x3f800000u));
    return polynomial;
}

static inline float aps5_series_when_far(float value, float series) {
    const uint valueBits = as_type<uint>(value);
    const uint seriesBits = as_type<uint>(series);
    return min(valueBits - seriesBits, seriesBits - valueBits) > 4u ? series : value;
}

static inline float aps5_trig_linear_near_zero(float value, float distance, float linear) {
    return (as_type<uint>(distance) & 0x7fffffffu) < 0x39800000u ? distance * linear : value;
}

static inline float aps5_sin(float value) {
    const float cycle = aps5_trig_cycle(value, true);
    const float absolute = aps5_fabs(cycle);
    const uint sign = as_type<uint>(cycle) & 0x80000000u;
    const bool nearHalf = absolute > 0.25f;
    const float distance = nearHalf ? 0.5f - absolute : absolute;
    const float series = distance > 0.125f ? aps5_cos_turns(0.25f - distance) : aps5_sin_turns(distance);
    const float host = aps5_series_when_far(sin(absolute * as_type<float>(0x40c90fdbu)), series);
    const float result = aps5_trig_linear_near_zero(host, distance, as_type<float>(nearHalf ? 0x40c90fdbu : 0x40c90fd5u));
    return as_type<float>(as_type<uint>(result) | sign);
}

static inline float aps5_cos(float value) {
    const float cycle = aps5_trig_cycle(value, false);
    const float absolute = aps5_fabs(cycle);
    const float distance = 0.25f - absolute;
    const float low = aps5_cos_turns(absolute);
    const float middle = aps5_sin_turns(distance);
    const float high = -aps5_cos_turns(0.5f - absolute);
    const float series = absolute > 0.375f ? high : (absolute > 0.125f ? middle : low);
    const float host = aps5_series_when_far(cos(absolute * as_type<float>(0x40c90fdbu)), series);
    return aps5_trig_linear_near_zero(host, distance, as_type<float>(0x40c90fdbu));
}

)";

constexpr std::string_view Float16Text = R"(// f16 <-> f32 on bits: the GPU conversions drop NaN payloads (and flush nothing that matters here).
static inline float aps5_f16_to_f32(ushort halfBits) {
    const uint bits = uint(halfBits);
    const uint sign = (bits & 0x8000u) << 16;
    const uint exponent = (bits >> 10) & 0x1fu;
    const uint mantissa = bits & 0x3ffu;
    uint magnitude;
    if (exponent == 0u) magnitude = as_type<uint>(float(mantissa) * 0x1p-24f);
    else if (exponent == 0x1fu) magnitude = 0x7f800000u | (mantissa << 13);
    else magnitude = ((exponent + 112u) << 23) | (mantissa << 13);
    return as_type<float>(sign | magnitude);
}

// The hardware conversion (MslTargetOptions::float16DenormPreserve, upstream #3988 EmitF16BitsToF32):
// exact where the GPU keeps half denormals; a NaN's bits are rebuilt from the half (sign at bit 31,
// payload at bits 22:13) whatever NaN the GPU returns.
static inline float aps5_f16_to_f32_native(ushort halfBits) {
    const float value = float(as_type<half>(halfBits));
    const uint nan = uint(as_type<int>(uint(halfBits) << 16) >> 3) | 0x7f800000u;
    return isnan(value) ? as_type<float>(nan) : value;
}

// The hardware conversion for operands that are only compared (OnlyCompared, upstream #4011): any
// NaN will do, so the payload is not rebuilt.
static inline float aps5_f16_to_f32_any_nan(ushort halfBits) {
    return float(as_type<half>(halfBits));
}

// Round to nearest even (half denormals kept, overflow to infinity); a NaN keeps its sign and the top
// of its payload and becomes quiet.
static inline uint aps5_f32_to_f16(float value) {
    const uint bits = as_type<uint>(value);
    if (aps5_f32_nan(bits)) return ((bits >> 16) & 0x8000u) | 0x7e00u | ((bits & 0x007fffffu) >> 13);
    return uint(as_type<ushort>(half(value)));
}

// Round toward zero (SpirvAluEmitterMath EmitF32ToF16BitsTowardZero): one step back from a nearest
// result that lies further from zero than the source.
static inline uint aps5_f32_to_f16_rtz(float value) {
    const uint nearest = aps5_f32_to_f16(value);
    const float widened = aps5_f16_to_f32(ushort(nearest));
    return aps5_fabs(widened) > aps5_fabs(value) ? nearest - 1u : nearest;
}

)";

constexpr std::array<MslPreludeChunk, 1> OnAlu{MslPreludeChunk::Alu};

// SpirvAluEmitterMath NarrowF16Field (upstream #3988): a half taken from a bit field narrower than 16
// bits (SDWA byte selects) keeps the integer decode on every target.
bool NarrowF16Field(const IrValue& half) {
    const IrValue* value = half.Resolve();
    while (!value->HasImmediate() && (value->Opcode() == IrOpcode::BitCastF16U16 || value->Opcode() == IrOpcode::ConvertU16U32)) value = value->Argument(0)->Resolve();
    if (value->HasImmediate() || (value->Opcode() != IrOpcode::BitFieldUExtract && value->Opcode() != IrOpcode::BitFieldSExtract)) return false;
    const IrValue* count = value->Argument(2)->Resolve();
    return !count->HasImmediate() || count->ImmediateU32() < 16u;
}

// SpirvAluEmitterMath OnlyCompared (upstream #4011): a converted half whose every
// use is an f32 compare or NaN test (directly or through abs, neg and Identity) only needs to be NaN
// or not, so the hardware conversion skips the NaN payload rebuild.
bool OnlyCompared(const IrValue& value) {
    return std::ranges::all_of(value.Uses(), [](const IrValue* user) {
        switch (user->Opcode()) {
        case IrOpcode::FPOrdEqual32:
        case IrOpcode::FPUnordEqual32:
        case IrOpcode::FPOrdNotEqual32:
        case IrOpcode::FPUnordNotEqual32:
        case IrOpcode::FPOrdLessThan32:
        case IrOpcode::FPUnordLessThan32:
        case IrOpcode::FPOrdGreaterThan32:
        case IrOpcode::FPUnordGreaterThan32:
        case IrOpcode::FPOrdLessThanEqual32:
        case IrOpcode::FPUnordLessThanEqual32:
        case IrOpcode::FPOrdGreaterThanEqual32:
        case IrOpcode::FPUnordGreaterThanEqual32:
        case IrOpcode::FPIsNan32:
            return true;
        case IrOpcode::FPAbs32:
        case IrOpcode::FPNeg32:
        case IrOpcode::Identity:
            return OnlyCompared(*user);
        default:
            return false;
        }
    });
}

void Binary(MslEmitterState& state, const IrValue& instruction, std::string_view operation) {
    state.Define(instruction, "(" + state.Arg(instruction, 0u) + " " + std::string(operation) + " " + state.Arg(instruction, 1u) + ")");
}

void Signed32(MslEmitterState& state, const IrValue& instruction, std::string_view operation) {
    state.Define(instruction, "(as_type<int>(" + state.Arg(instruction, 0u) + ") " + std::string(operation) + " as_type<int>(" + state.Arg(instruction, 1u) + "))");
}

void Signed64(MslEmitterState& state, const IrValue& instruction, std::string_view operation) {
    state.Define(instruction, "(as_type<long>(" + state.Arg(instruction, 0u) + ") " + std::string(operation) + " as_type<long>(" + state.Arg(instruction, 1u) + "))");
}

void Call(MslEmitterState& state, const IrValue& instruction, std::string_view function) {
    std::string expression = std::string(function) + "(";
    for (std::size_t index = 0; index < instruction.ArgumentCount(); ++index) expression += (index == 0u ? "" : ", ") + state.Arg(instruction, index);
    state.Define(instruction, expression + ")");
}

// A float operation: the Alu chunk carries `#pragma METAL fp contract(off)` and the helpers.
void FloatCall(MslEmitterState& state, const IrValue& instruction, std::string_view function) {
    state.RequirePrelude(MslPreludeChunk::Alu);
    Call(state, instruction, function);
}

void FloatBinary(MslEmitterState& state, const IrValue& instruction, std::string_view operation) {
    state.RequirePrelude(MslPreludeChunk::Alu);
    Binary(state, instruction, operation);
}

void FloatExpression(MslEmitterState& state, const IrValue& instruction, const std::string& expression) {
    state.RequirePrelude(MslPreludeChunk::Alu);
    state.Define(instruction, expression);
}

void Construct(MslEmitterState& state, const IrValue& instruction) {
    Call(state, instruction, MslType(instruction.Type()));
}

void Extract(MslEmitterState& state, const IrValue& instruction) {
    const auto* index = instruction.Argument(1u)->Resolve();
    if (!index->HasImmediate()) state.Fail(instruction, "needs an immediate component index");
    state.Define(instruction, state.Arg(instruction, 0u) + "[" + std::to_string(index->ImmediateU32()) + "]");
}

std::string SignedTri(MslEmitterState& state, const IrValue& instruction, std::string_view function) {
    const auto a = "as_type<int>(" + state.Arg(instruction, 0u) + ")";
    const auto b = "as_type<int>(" + state.Arg(instruction, 1u) + ")";
    const auto c = "as_type<int>(" + state.Arg(instruction, 2u) + ")";
    return "as_type<uint>(" + std::string(function) + "(" + std::string(function) + "(" + a + ", " + b + "), " + c + "))";
}

// An f32 compare. Each unordered form is the negation of the complementary ordered one. Native MSL
// compares (IEEE in safe math mode, ordered except !=) when f32 denormals are flushed (the front end
// flushes the operands), compares on bits when FLOAT_MODE keeps them.
std::string FloatCompare(MslEmitterState& state, const IrValue& instruction) {
    const auto a = state.Arg(instruction, 0u);
    const auto b = state.Arg(instruction, 1u);
    const bool exact = state.target.keepF32Denormals;
    const auto call = [&](std::string_view function) { return std::string(function) + "(" + a + ", " + b + ")"; };
    const auto native = [&](std::string_view operation) { return "(" + a + " " + std::string(operation) + " " + b + ")"; };
    // Ordered not-equal on NaN bits: the Metal compiler folds `a < b || a > b` into a compare that
    // is true for NaN operands (v_cmp_lg_f32 of 0 and NaN measured 1 on the M3).
    state.RequirePrelude(MslPreludeChunk::Alu);
    const auto orderedNotEqual = "(aps5_f32_ordered(" + a + ", " + b + ") && " + a + " != " + b + ")";
    switch (instruction.Opcode()) {
    case IrOpcode::FPOrdEqual32: return exact ? call("aps5_feq") : native("==");
    case IrOpcode::FPOrdNotEqual32: return exact ? call("aps5_fne") : orderedNotEqual;
    case IrOpcode::FPOrdLessThan32: return exact ? call("aps5_flt") : native("<");
    case IrOpcode::FPOrdGreaterThan32: return exact ? call("aps5_fgt") : native(">");
    case IrOpcode::FPOrdLessThanEqual32: return exact ? call("aps5_fle") : native("<=");
    case IrOpcode::FPOrdGreaterThanEqual32: return exact ? call("aps5_fge") : native(">=");
    case IrOpcode::FPUnordEqual32: return "!" + (exact ? call("aps5_fne") : orderedNotEqual);
    case IrOpcode::FPUnordNotEqual32: return exact ? "!" + call("aps5_feq") : native("!=");
    case IrOpcode::FPUnordLessThan32: return "!" + (exact ? call("aps5_fge") : native(">="));
    case IrOpcode::FPUnordGreaterThan32: return "!" + (exact ? call("aps5_fle") : native("<="));
    case IrOpcode::FPUnordLessThanEqual32: return "!" + (exact ? call("aps5_fgt") : native(">"));
    case IrOpcode::FPUnordGreaterThanEqual32: return "!" + (exact ? call("aps5_flt") : native("<"));
    default: state.Fail(instruction, "is not an f32 compare");
    }
}

std::string Median(const std::string& a, const std::string& b, const std::string& c) {
    return "max(min(" + a + ", " + b + "), min(max(" + a + ", " + b + "), " + c + "))";
}

// Window narrowing (upstream 30fab0777e, SpirvModuleSetup.cpp WindowIndexBounds/IndexedWindowSlots and
// SpirvAluEmitterRegisters.cpp AccessNarrowedWindow): the window array holds only the slots the relative
// accesses likely reach, from a bound analysis of their indices; every other slot is a local of its own the
// Metal compiler can keep in a register (a function-scope array that is dynamically indexed lives in stack
// memory on Apple GPUs). A relative access whose index lands past the array at run time switches over those
// locals, so the guess only affects speed. APS5_NO_WINDOW_NARROWING=1 turns it off. The analysis is
// upstream's, kept in step with it.
class WindowIndexBounds {
public:
    std::optional<std::uint64_t> Of(const IrValue& input) {
        const IrValue* value = input.Resolve();
        if (value->HasImmediate()) return value->ImmediateU32();
        if (const auto found = memo.find(value); found != memo.end()) return found->second;
        if (!active.insert(value).second || active.size() > 256u) return std::nullopt;
        const auto bound = compute(*value);
        active.erase(value);
        memo.emplace(value, bound);
        return bound;
    }

private:
    using Bound = std::optional<std::uint64_t>;

    Bound maxOf(std::initializer_list<Bound> bounds) {
        std::uint64_t result = 0;
        for (const auto& bound : bounds) {
            if (!bound) return std::nullopt;
            result = std::max(result, *bound);
        }
        return result;
    }

    static std::vector<const IrValue*> reachingStores(const IrValue& load, std::uint32_t slot) {
        std::vector<const IrValue*> values;
        std::unordered_set<const IrBlock*> visited;
        std::vector<std::pair<IrBlock*, const IrValue*>> pending{{load.Parent(), &load}};
        while (!pending.empty()) {
            const auto [block, before] = pending.back();
            pending.pop_back();
            const auto& instructions = block->Instructions();
            auto it = instructions.rbegin();
            if (before != nullptr) it = std::next(std::find(instructions.rbegin(), instructions.rend(), before));
            bool stored = false;
            for (; it != instructions.rend() && !stored; ++it) {
                if ((*it)->Opcode() != IrOpcode::StoreVectorWindow) continue;
                const IrValue* index = (*it)->Argument(0)->Resolve();
                if (!index->HasImmediate() || index->ImmediateU32() != slot) continue;
                values.push_back((*it)->Argument(1));
                stored = true;
            }
            if (stored) continue;
            for (IrBlock* predecessor : block->Predecessors()) {
                if (visited.insert(predecessor).second) pending.emplace_back(predecessor, nullptr);
            }
        }
        return values;
    }

    Bound compute(const IrValue& value) {
        const auto arg = [&](std::size_t index) { return Of(*value.Argument(index)); };
        const auto immediate = [&](std::size_t index) -> std::optional<std::uint32_t> {
            const IrValue* operand = value.Argument(index)->Resolve();
            if (!operand->HasImmediate()) return std::nullopt;
            return operand->ImmediateU32();
        };
        const auto fits = [](std::uint64_t bound) -> Bound { return bound <= 0xffffffffull ? Bound(bound) : std::nullopt; };
        switch (value.Opcode()) {
        case IrOpcode::BitFieldUExtract: {
            const auto count = immediate(2);
            if (count && *count < 32u) return (std::uint64_t{1} << *count) - 1u;
            return arg(0);
        }
        case IrOpcode::BitwiseAnd32: {
            const auto lhs = arg(0);
            const auto rhs = arg(1);
            if (lhs && rhs) return std::min(*lhs, *rhs);
            return lhs ? lhs : rhs;
        }
        case IrOpcode::UMin32: {
            const auto lhs = arg(0);
            const auto rhs = arg(1);
            if (!lhs || !rhs) return std::nullopt;
            return std::min(*lhs, *rhs);
        }
        case IrOpcode::UMax32:
            return maxOf({arg(0), arg(1)});
        case IrOpcode::BitwiseOr32: {
            const auto both = maxOf({arg(0), arg(1)});
            if (!both) return std::nullopt;
            std::uint64_t mask = 0;
            while (mask < *both) mask = mask << 1u | 1u;
            return mask;
        }
        case IrOpcode::IAdd32: {
            const auto lhs = arg(0);
            const auto rhs = arg(1);
            if (!lhs || !rhs) return std::nullopt;
            return fits(*lhs + *rhs);
        }
        case IrOpcode::ShiftRightLogical32: {
            const auto shift = immediate(1);
            const auto bound = arg(0);
            if (!bound) return std::nullopt;
            return shift ? *bound >> (*shift & 31u) : *bound;
        }
        case IrOpcode::ShiftLeftLogical32: {
            const auto shift = immediate(1);
            const auto bound = arg(0);
            if (!bound || !shift) return std::nullopt;
            return fits(*bound << (*shift & 31u));
        }
        case IrOpcode::ConvertU32U16:
            return 0xffffu;
        case IrOpcode::Select:
        case IrOpcode::SelectU32: {
            if (value.Argument(0)->Resolve()->Opcode() == IrOpcode::GetExec || value.Argument(2)->Resolve()->Opcode() == IrOpcode::LoadVectorWindow) return arg(1);
            const auto lhs = arg(1);
            const auto rhs = arg(2);
            if (lhs && rhs) return std::max(*lhs, *rhs);
            return lhs ? lhs : rhs;
        }
        case IrOpcode::ReadFirstLane:
            return arg(0);
        case IrOpcode::Identity:
        case IrOpcode::Phi: {
            Bound bound;
            for (std::size_t index = 0; index < value.ArgumentCount(); ++index) {
                if (const auto incoming = arg(index)) bound = std::max(bound.value_or(0u), *incoming);
            }
            return bound;
        }
        case IrOpcode::LoadVectorWindow: {
            const IrValue* index = value.Argument(0)->Resolve();
            if (!index->HasImmediate()) return std::nullopt;
            Bound bound = 0u;
            for (const IrValue* stored : reachingStores(value, index->ImmediateU32())) {
                bound = maxOf({bound, Of(*stored)});
                if (!bound) return std::nullopt;
            }
            return bound;
        }
        default:
            return std::nullopt;
        }
    }

    std::unordered_map<const IrValue*, Bound> memo;
    std::unordered_set<const IrValue*> active;
};

std::uint32_t IndexedWindowSlots(const IrProgram& program) {
    const auto window = program.Metadata().vectorWindowSize;
    WindowIndexBounds bounds;
    std::uint64_t reached = 0;
    for (const auto& block : program.Blocks()) {
        for (const IrValue* inst : block->Instructions()) {
            if (inst->Opcode() != IrOpcode::LoadVectorWindow && inst->Opcode() != IrOpcode::StoreVectorWindow) continue;
            const IrValue* index = inst->Argument(0)->Resolve();
            if (index->HasImmediate()) continue;
            const auto bound = bounds.Of(*index);
            if (bound && *bound < window / 2u) reached = std::max(reached, *bound + 1u);
        }
    }
    return static_cast<std::uint32_t>(reached);
}

// The relative-addressing window of vector registers (v_movrels/v_movreld): a function-scope array per
// guest lane of the invocation, as the SPIR-V path's Function-storage `vector_window` per lane half, plus
// one local per narrowed-out slot (`<window>_s<slot>`). `access(element)` emits the load or store of one
// element expression; a dynamic index past the array switches over the slot locals (a load of a slot past
// the window reads 0, as upstream's OpSwitch default).
void AccessVectorWindow(MslEmitterState& state, const IrValue& instruction, const std::function<void(const std::string&)>& access, const std::string* loadTarget) {
    const auto size = state.program.Metadata().vectorWindowSize;
    if (size == 0u) state.Fail(instruction, "has no relative-addressing window in the program metadata");
    if (!state.vectorWindowIndexed) {
        static const bool narrowWindow = std::getenv("APS5_NO_WINDOW_NARROWING") == nullptr;
        state.vectorWindowIndexed = narrowWindow ? IndexedWindowSlots(state.program) : 0u;
    }
    const std::uint32_t indexed = *state.vectorWindowIndexed;
    const std::uint32_t arraySize = indexed != 0u ? indexed : size;
    const std::string name = state.half == 0u ? "aps5_vector_window" : "aps5_vector_window_h";
    if (state.locals.find(" " + name + "[") == std::string::npos) {
        state.DeclareLocal("uint", name + "[" + std::to_string(arraySize) + "]", "{}");
        for (std::uint32_t slot = arraySize; slot < size; ++slot) state.DeclareLocal("uint", name + "_s" + std::to_string(slot), "0u");
    }
    const auto slotName = [&](std::uint32_t slot) { return name + "_s" + std::to_string(slot); };
    const IrValue* index = instruction.Argument(0)->Resolve();
    const std::string at = state.Arg(instruction, 0u);
    if (indexed == 0u) return access(name + "[" + at + "]");
    if (index->HasImmediate()) {
        const auto slot = index->ImmediateU32();
        if (slot < indexed) return access(name + "[" + at + "]");
        if (slot < size) return access(slotName(slot));
        if (loadTarget != nullptr) state.Line(*loadTarget + " = 0u;");
        return;
    }
    state.Line("if (" + at + " < " + std::to_string(indexed) + "u) {");
    state.Indent();
    access(name + "[" + at + "]");
    state.Outdent();
    state.Line("} else {");
    state.Indent();
    state.Line("switch (" + at + ") {");
    for (std::uint32_t slot = indexed; slot < size; ++slot) {
        state.Line("case " + std::to_string(slot) + "u:");
        state.Indent();
        access(slotName(slot));
        state.Line("break;");
        state.Outdent();
    }
    state.Line("default:");
    state.Indent();
    if (loadTarget != nullptr) state.Line(*loadTarget + " = 0u;");
    state.Line("break;");
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
}

}

bool EmitMslAluInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    // Integer arithmetic.
    case IrOpcode::IAdd32:
    case IrOpcode::IAdd64:
        Binary(state, instruction, "+");
        return true;
    case IrOpcode::ISub32:
    case IrOpcode::ISub64:
        Binary(state, instruction, "-");
        return true;
    case IrOpcode::IMul32:
    case IrOpcode::IMul64:
        Binary(state, instruction, "*");
        return true;
    case IrOpcode::UDiv32:
        state.Define(instruction, "(" + state.Arg(instruction, 1u) + " == 0u ? 0xffffffffu : " + state.Arg(instruction, 0u) + " / " + state.Arg(instruction, 1u) + ")");
        return true;
    case IrOpcode::UMulHi:
        Call(state, instruction, "mulhi");
        return true;
    case IrOpcode::SMulHi:
        state.Define(instruction, "as_type<uint>(mulhi(as_type<int>(" + state.Arg(instruction, 0u) + "), as_type<int>(" + state.Arg(instruction, 1u) + ")))");
        return true;
    case IrOpcode::IAddCarry32:
        state.Define(instruction, "uint2(" + state.Arg(instruction, 0u) + " + " + state.Arg(instruction, 1u) + ", (" + state.Arg(instruction, 0u) + " + " + state.Arg(instruction, 1u) + ") < " + state.Arg(instruction, 0u) + " ? 1u : 0u)");
        return true;
    case IrOpcode::IAbs32:
        state.Define(instruction, "as_type<uint>(abs(as_type<int>(" + state.Arg(instruction, 0u) + ")))");
        return true;
    case IrOpcode::ShiftLeftLogical32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " << (" + state.Arg(instruction, 1u) + " & 31u))");
        return true;
    case IrOpcode::ShiftRightLogical32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " >> (" + state.Arg(instruction, 1u) + " & 31u))");
        return true;
    case IrOpcode::ShiftRightArithmetic32:
        state.Define(instruction, "as_type<uint>(as_type<int>(" + state.Arg(instruction, 0u) + ") >> (" + state.Arg(instruction, 1u) + " & 31u))");
        return true;
    case IrOpcode::ShiftLeftLogical64:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " << (" + state.Arg(instruction, 1u) + " & 63u))");
        return true;
    case IrOpcode::ShiftRightLogical64:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " >> (" + state.Arg(instruction, 1u) + " & 63u))");
        return true;
    case IrOpcode::ShiftRightArithmetic64:
        state.Define(instruction, "as_type<ulong>(as_type<long>(" + state.Arg(instruction, 0u) + ") >> (" + state.Arg(instruction, 1u) + " & 63u))");
        return true;
    case IrOpcode::BitwiseAnd32:
    case IrOpcode::BitwiseAnd64:
        Binary(state, instruction, "&");
        return true;
    case IrOpcode::BitwiseOr32:
        Binary(state, instruction, "|");
        return true;
    case IrOpcode::BitwiseXor32:
        Binary(state, instruction, "^");
        return true;
    case IrOpcode::BitwiseNot32:
        state.Define(instruction, "~" + state.Arg(instruction, 0u));
        return true;
    case IrOpcode::BitReverse32:
        Call(state, instruction, "reverse_bits");
        return true;
    case IrOpcode::BitCount32:
    case IrOpcode::BitCount64:
        state.Define(instruction, "uint(popcount(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::FindILsb32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " == 0u ? 0xffffffffu : ctz(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::FindUMsb32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " == 0u ? 0xffffffffu : 31u - clz(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::FindUMsb64:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " == 0ul ? 0xffffffffu : 63u - uint(clz(" + state.Arg(instruction, 0u) + ")))");
        return true;
    case IrOpcode::BitFieldUExtract:
        Call(state, instruction, "extract_bits");
        return true;
    case IrOpcode::BitFieldSExtract:
        state.Define(instruction, "as_type<uint>(extract_bits(as_type<int>(" + state.Arg(instruction, 0u) + "), " + state.Arg(instruction, 1u) + ", " + state.Arg(instruction, 2u) + "))");
        return true;
    case IrOpcode::BitFieldInsert:
        Call(state, instruction, "insert_bits");
        return true;
    case IrOpcode::UMin32:
        Call(state, instruction, "min");
        return true;
    case IrOpcode::UMax32:
        Call(state, instruction, "max");
        return true;
    case IrOpcode::SMin32:
        state.Define(instruction, "as_type<uint>(min(as_type<int>(" + state.Arg(instruction, 0u) + "), as_type<int>(" + state.Arg(instruction, 1u) + ")))");
        return true;
    case IrOpcode::SMax32:
        state.Define(instruction, "as_type<uint>(max(as_type<int>(" + state.Arg(instruction, 0u) + "), as_type<int>(" + state.Arg(instruction, 1u) + ")))");
        return true;
    case IrOpcode::UMinTri32:
        state.Define(instruction, "min(min(" + state.Arg(instruction, 0u) + ", " + state.Arg(instruction, 1u) + "), " + state.Arg(instruction, 2u) + ")");
        return true;
    case IrOpcode::UMaxTri32:
        state.Define(instruction, "max(max(" + state.Arg(instruction, 0u) + ", " + state.Arg(instruction, 1u) + "), " + state.Arg(instruction, 2u) + ")");
        return true;
    case IrOpcode::SMinTri32:
        state.Define(instruction, SignedTri(state, instruction, "min"));
        return true;
    case IrOpcode::SMaxTri32:
        state.Define(instruction, SignedTri(state, instruction, "max"));
        return true;
    case IrOpcode::UMedTri32:
        state.Define(instruction, Median(state.Arg(instruction, 0u), state.Arg(instruction, 1u), state.Arg(instruction, 2u)));
        return true;
    case IrOpcode::SMedTri32: {
        const auto a = "as_type<int>(" + state.Arg(instruction, 0u) + ")";
        const auto b = "as_type<int>(" + state.Arg(instruction, 1u) + ")";
        const auto c = "as_type<int>(" + state.Arg(instruction, 2u) + ")";
        state.Define(instruction, "as_type<uint>(" + Median(a, b, c) + ")");
        return true;
    }

    // Integer and logical compares.
    case IrOpcode::IEqual32:
    case IrOpcode::IEqual64:
        Binary(state, instruction, "==");
        return true;
    case IrOpcode::INotEqual32:
    case IrOpcode::INotEqual64:
        Binary(state, instruction, "!=");
        return true;
    case IrOpcode::ULessThan32:
    case IrOpcode::ULessThan64:
        Binary(state, instruction, "<");
        return true;
    case IrOpcode::ULessThanEqual32:
        Binary(state, instruction, "<=");
        return true;
    case IrOpcode::UGreaterThan32:
    case IrOpcode::UGreaterThan64:
        Binary(state, instruction, ">");
        return true;
    case IrOpcode::UGreaterThanEqual32:
        Binary(state, instruction, ">=");
        return true;
    case IrOpcode::SLessThan32:
        Signed32(state, instruction, "<");
        return true;
    case IrOpcode::SLessThan64:
        Signed64(state, instruction, "<");
        return true;
    case IrOpcode::SLessThanEqual32:
        Signed32(state, instruction, "<=");
        return true;
    case IrOpcode::SGreaterThan32:
        Signed32(state, instruction, ">");
        return true;
    case IrOpcode::SGreaterThanEqual32:
        Signed32(state, instruction, ">=");
        return true;
    case IrOpcode::LogicalAnd:
        Binary(state, instruction, "&&");
        return true;
    case IrOpcode::LogicalOr:
        Binary(state, instruction, "||");
        return true;
    case IrOpcode::LogicalXor:
        Binary(state, instruction, "!=");
        return true;
    case IrOpcode::LogicalNot:
        state.Define(instruction, "!" + state.Arg(instruction, 0u));
        return true;
    case IrOpcode::Select:
    case IrOpcode::SelectU1:
    case IrOpcode::SelectU32:
    case IrOpcode::SelectF32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " ? " + state.Arg(instruction, 1u) + " : " + state.Arg(instruction, 2u) + ")");
        return true;

    // Composites, casts and integer conversions.
    case IrOpcode::CompositeConstructU32x2:
    case IrOpcode::CompositeConstructU32x3:
    case IrOpcode::CompositeConstructU32x4:
    case IrOpcode::CompositeConstructF32x2:
        Construct(state, instruction);
        return true;
    case IrOpcode::CompositeConstructU64:
        state.Define(instruction, "(ulong(" + state.Arg(instruction, 0u) + ") | (ulong(" + state.Arg(instruction, 1u) + ") << 32))");
        return true;
    case IrOpcode::CompositeExtractU32x2:
    case IrOpcode::CompositeExtractU32x3:
    case IrOpcode::CompositeExtractU32x4:
        Extract(state, instruction);
        return true;
    case IrOpcode::CompositeExtractU64: {
        const auto* index = instruction.Argument(1u)->Resolve();
        if (!index->HasImmediate() || index->ImmediateU32() > 1u) state.Fail(instruction, "needs an immediate half index");
        state.Define(instruction, "uint(" + state.Arg(instruction, 0u) + " >> " + std::to_string(index->ImmediateU32() * 32u) + ")");
        return true;
    }
    case IrOpcode::ConvertU16U32:
        state.Define(instruction, "ushort(" + state.Arg(instruction, 0u) + " & 0xffffu)");
        return true;
    case IrOpcode::ConvertU8U32:
        state.Define(instruction, "uchar(" + state.Arg(instruction, 0u) + " & 0xffu)");
        return true;
    case IrOpcode::ConvertU32U16:
    case IrOpcode::ConvertU32U8:
        state.Define(instruction, "uint(" + state.Arg(instruction, 0u) + ")");
        return true;
    case IrOpcode::BitCastF32U32:
    case IrOpcode::BitCastU32F32:
    case IrOpcode::BitCastF16U16:
    case IrOpcode::BitCastU16F16:
        state.Define(instruction, "as_type<" + MslType(instruction.Type()) + ">(" + state.Arg(instruction, 0u) + ")");
        return true;
    case IrOpcode::UndefU1:
    case IrOpcode::UndefU8:
    case IrOpcode::UndefU16:
    case IrOpcode::UndefU32:
    case IrOpcode::UndefU64:
        state.Define(instruction, MslType(instruction.Type()) + "(0)");
        return true;

    // Float conversions. float->int saturates and maps NaN to 0 on Apple GPUs, as v_cvt_*_f32.
    case IrOpcode::ConvertS32F32:
        FloatExpression(state, instruction, "as_type<uint>(int(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::ConvertU32F32:
        FloatExpression(state, instruction, "uint(" + state.Arg(instruction, 0u) + ")");
        return true;
    case IrOpcode::ConvertF32S32:
        FloatExpression(state, instruction, "float(as_type<int>(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::ConvertF32U32:
        FloatExpression(state, instruction, "float(" + state.Arg(instruction, 0u) + ")");
        return true;
    case IrOpcode::ConvertF32F16:
        state.RequirePrelude(MslPreludeChunk::Float16);
        {
            const bool native = state.target.float16DenormPreserve && !NarrowF16Field(*instruction.Argument(0u));
            const char* function = !native ? "aps5_f16_to_f32" : OnlyCompared(instruction) ? "aps5_f16_to_f32_any_nan" : "aps5_f16_to_f32_native";
            state.Define(instruction, std::string(function) + "(as_type<ushort>(" + state.Arg(instruction, 0u) + "))");
        }
        return true;
    case IrOpcode::ConvertF16F32:
        state.RequirePrelude(MslPreludeChunk::Float16);
        state.Define(instruction, "as_type<half>(ushort(aps5_f32_to_f16(" + state.Arg(instruction, 0u) + ")))");
        return true;
    case IrOpcode::PackHalf2x16: {
        state.RequirePrelude(MslPreludeChunk::Float16);
        const auto pair = state.Arg(instruction, 0u);
        state.Define(instruction, "(aps5_f32_to_f16(" + pair + ".x) | (aps5_f32_to_f16(" + pair + ".y) << 16))");
        return true;
    }
    case IrOpcode::PackFloat2x16Rtz:
        state.RequirePrelude(MslPreludeChunk::Float16);
        state.Define(instruction, "(aps5_f32_to_f16_rtz(" + state.Arg(instruction, 0u) + ") | (aps5_f32_to_f16_rtz(" + state.Arg(instruction, 1u) + ") << 16))");
        return true;
    // Packed f16 (only with the native-f16 capabilities): half arithmetic rounds to nearest even and
    // keeps denormals; the front end fixes up NaN lanes.
    case IrOpcode::FPAdd16x2:
        FloatExpression(state, instruction, "as_type<uint>(as_type<half2>(" + state.Arg(instruction, 0u) + ") + as_type<half2>(" + state.Arg(instruction, 1u) + "))");
        return true;
    case IrOpcode::FPMul16x2:
        FloatExpression(state, instruction, "as_type<uint>(as_type<half2>(" + state.Arg(instruction, 0u) + ") * as_type<half2>(" + state.Arg(instruction, 1u) + "))");
        return true;

    // f32 sign, clamp and classification on bits.
    case IrOpcode::FPAbs32:
        FloatCall(state, instruction, "aps5_fabs");
        return true;
    case IrOpcode::FPNeg32:
        FloatExpression(state, instruction, "as_type<float>(as_type<uint>(" + state.Arg(instruction, 0u) + ") ^ 0x80000000u)");
        return true;
    case IrOpcode::FPSaturate32:
        FloatCall(state, instruction, "aps5_fsat");
        return true;
    case IrOpcode::FPIsNan32:
        FloatExpression(state, instruction, "aps5_f32_nan(as_type<uint>(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::FPCmpClass32:
        FloatCall(state, instruction, "aps5_f32_class");
        return true;

    // f32 compares. With f32 denormals flushed (the front end flushes the operands), MSL compares are
    // IEEE in safe math mode (ordered except !=); with them kept, compares run on bits.
    case IrOpcode::FPOrdEqual32:
    case IrOpcode::FPUnordNotEqual32:
    case IrOpcode::FPOrdLessThan32:
    case IrOpcode::FPOrdGreaterThan32:
    case IrOpcode::FPOrdLessThanEqual32:
    case IrOpcode::FPOrdGreaterThanEqual32:
    case IrOpcode::FPUnordEqual32:
    case IrOpcode::FPOrdNotEqual32:
    case IrOpcode::FPUnordLessThan32:
    case IrOpcode::FPUnordGreaterThan32:
    case IrOpcode::FPUnordLessThanEqual32:
    case IrOpcode::FPUnordGreaterThanEqual32:
        FloatExpression(state, instruction, FloatCompare(state, instruction));
        return true;

    // f32 arithmetic: IEEE with the contraction pragma; fma() is the fused form. Products take RDNA's
    // tininess after rounding; with FLOAT_MODE keeping f32 denormals every operation is exact.
    case IrOpcode::FPAdd32:
        if (state.target.keepF32Denormals) FloatCall(state, instruction, "aps5_fadd_keep");
        else FloatBinary(state, instruction, "+");
        return true;
    case IrOpcode::FPSub32:
        if (state.target.keepF32Denormals) FloatExpression(state, instruction, "aps5_fadd_keep(" + state.Arg(instruction, 0u) + ", as_type<float>(as_type<uint>(" + state.Arg(instruction, 1u) + ") ^ 0x80000000u))");
        else FloatBinary(state, instruction, "-");
        return true;
    case IrOpcode::FPMul32:
        if ((instruction.Flags<std::uint32_t>() & FPMulNativeFlushFlag) != 0u && !state.target.keepF32Denormals) FloatCall(state, instruction, "aps5_fmul_ftz");
        else FloatCall(state, instruction, state.target.keepF32Denormals ? "aps5_fmul_keep" : "aps5_fmul");
        return true;
    case IrOpcode::FPFma32:
        if ((instruction.Flags<std::uint32_t>() & FPMulNativeFlushFlag) != 0u && !state.target.keepF32Denormals) FloatCall(state, instruction, "aps5_fma_ftz");
        else FloatCall(state, instruction, state.target.keepF32Denormals ? "aps5_fma_keep" : "aps5_fma");
        return true;
    case IrOpcode::FPMad32: {
        const auto product = std::string(state.target.keepF32Denormals ? "aps5_fmul_keep(" : "aps5_fmul(") + state.Arg(instruction, 0u) + ", " + state.Arg(instruction, 1u) + ")";
        if (state.target.keepF32Denormals) FloatExpression(state, instruction, "aps5_fadd_keep(" + product + ", " + state.Arg(instruction, 2u) + ")");
        else FloatExpression(state, instruction, "(" + product + " + " + state.Arg(instruction, 2u) + ")");
        return true;
    }
    case IrOpcode::FPMin32:
        FloatCall(state, instruction, "aps5_fmin");
        return true;
    case IrOpcode::FPMax32:
        FloatCall(state, instruction, "aps5_fmax");
        return true;
    case IrOpcode::FPMinTri32:
        FloatExpression(state, instruction, "aps5_fmin(aps5_fmin(" + state.Arg(instruction, 0u) + ", " + state.Arg(instruction, 1u) + "), " + state.Arg(instruction, 2u) + ")");
        return true;
    case IrOpcode::FPMaxTri32:
        FloatExpression(state, instruction, "aps5_fmax(aps5_fmax(" + state.Arg(instruction, 0u) + ", " + state.Arg(instruction, 1u) + "), " + state.Arg(instruction, 2u) + ")");
        return true;
    case IrOpcode::FPMedTri32:
        FloatCall(state, instruction, "aps5_fmed3");
        return true;

    // f32 transcendentals. No denormal flush of the source (upstream #4705, as the SPIR-V path): the
    // front end's unaryFloatSpecials replaces the result of every input below 2^-126.
    case IrOpcode::FPRecip32:
    case IrOpcode::FPRecipIFlag32:
        FloatExpression(state, instruction, "(1.0f / " + state.Arg(instruction, 0u) + ")");
        return true;
    case IrOpcode::FPRecipSqrt32:
        FloatCall(state, instruction, "rsqrt");
        return true;
    case IrOpcode::FPSqrt:
        FloatCall(state, instruction, "sqrt");
        return true;
    case IrOpcode::FPExp2:
        FloatCall(state, instruction, "exp2");
        return true;
    case IrOpcode::FPLog2:
        FloatCall(state, instruction, "aps5_log2");
        return true;
    case IrOpcode::FPSin:
        FloatCall(state, instruction, "aps5_sin");
        return true;
    case IrOpcode::FPCos:
        FloatCall(state, instruction, "aps5_cos");
        return true;
    case IrOpcode::FPRoundEven32:
        FloatCall(state, instruction, "rint");
        return true;
    case IrOpcode::FPFloor32:
        FloatCall(state, instruction, state.target.keepF32Denormals ? "aps5_floor_keep" : "floor");
        return true;
    case IrOpcode::FPCeil32:
        FloatCall(state, instruction, state.target.keepF32Denormals ? "aps5_ceil_keep" : "ceil");
        return true;
    case IrOpcode::FPTrunc32:
        FloatCall(state, instruction, "trunc");
        return true;
    case IrOpcode::FPFract32:
        FloatCall(state, instruction, state.target.keepF32Denormals ? "aps5_fract_keep" : "aps5_fract");
        return true;

    // Relative addressing of vector registers.
    case IrOpcode::LoadVectorWindow: {
        const auto target = state.Name(instruction);
        AccessVectorWindow(state, instruction, [&](const std::string& element) { state.Line(target + " = " + element + ";"); }, &target);
        return true;
    }
    case IrOpcode::StoreVectorWindow: {
        const auto value = state.Arg(instruction, 1u);
        AccessVectorWindow(state, instruction, [&](const std::string& element) { state.Line(element + " = " + value + ";"); }, nullptr);
        return true;
    }
    default:
        return false;
    }
}

MslPreludeEntry MslAluPrelude(MslPreludeChunk chunk) {
    switch (chunk) {
    case MslPreludeChunk::Alu:
        return {AluText, {}};
    case MslPreludeChunk::Float16:
        return {Float16Text, OnAlu};
    default:
        throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not an ALU chunk (package B2)");
    }
}

}
