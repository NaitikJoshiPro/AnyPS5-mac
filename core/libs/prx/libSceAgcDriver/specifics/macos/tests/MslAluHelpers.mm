// agc_metal_alu_helpers (work package B2): the MetalBackend ALU prelude (Alu, Float16, Float64 chunks,
// exactly the text MslPrelude() hands the emitter) run on the GPU against CPU references over
// deterministic pseudo-random operands: special values, denormals, ties, near-cancellations.
// - soft-float binary64: add (nearest even and to odd), mul, div, sqrt, fma (plain and scaled),
//   f32/i32/u32 conversions both ways, trunc/floor/ceil/round-even/fract, ldexp, frexp, min/max;
// - f32: the exact fma behind the denormal/tininess paths, the flush- and keep-mode mul/add/sub/fma
//   helpers, min/max/med3, class masks, saturate, bit compares, floor/ceil with denormals;
// - f16 <-> f32 on bits, round toward zero; the hardware half to float conversion (denormals kept?).
// The CPU (Apple arm64) computes IEEE binary32/64 with denormals; NaN results follow the SpirvBackend
// rules (first NaN operand quieted, else the RDNA default NaN). One dispatch per operation, each with a
// 10 s watchdog. Usage: agc_metal_alu_helpers_tests [cases-per-operation] [seed].
#pragma clang fp contract(off)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalBackend/MslEmitterState.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace {

using ShaderRecompiler::MslPrelude;
using ShaderRecompiler::MslPreludeChunk;

constexpr const char* KernelText = R"(
kernel void aps5_alu_check(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]], device const ulong* c [[buffer(2)]],
                           device ulong* result [[buffer(3)]], constant uint& op [[buffer(4)]], constant uint& count [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const ulong x = a[i], y = b[i], z = c[i];
    const float fx = as_type<float>(uint(x)), fy = as_type<float>(uint(y)), fz = as_type<float>(uint(z));
    ulong r = 0ul;
    switch (op) {
    case 0: r = aps5_f64_add(x, y); break;
    case 1: r = aps5_f64_add_rounded(x, y, true); break;
    case 2: r = aps5_f64_mul(x, y); break;
    case 3: r = aps5_f64_div(x, y); break;
    case 4: r = aps5_f64_sqrt(x); break;
    case 5: r = aps5_f64_fma(x, y, z, 0); break;
    case 6: r = aps5_f64_fma(x, y, z, (z >> 52) & 1ul ? 128 : -128); break;
    case 7: r = aps5_f32_to_f64(fx); break;
    case 8: r = aps5_f64_to_f32_bits(x); break;
    case 9: r = aps5_f64_to_i32(x); break;
    case 10: r = aps5_f64_to_u32(x); break;
    case 11: r = aps5_i32_to_f64(uint(x)); break;
    case 12: r = aps5_u32_to_f64(uint(x)); break;
    case 13: r = aps5_f64_trunc(x); break;
    case 14: r = aps5_f64_floor(x); break;
    case 15: r = aps5_f64_ceil(x); break;
    case 16: r = aps5_f64_round_even(x); break;
    case 17: r = aps5_f64_fract(x); break;
    case 18: r = aps5_f64_ldexp(x, as_type<int>(uint(y))); break;
    case 19: r = aps5_f64_frexp_mantissa(x); break;
    case 20: r = aps5_f64_frexp_exponent(x); break;
    case 21: r = aps5_f64_min_max(x, y, false); break;
    case 22: r = aps5_f64_min_max(x, y, true); break;
    case 23: r = as_type<uint>(aps5_f32_fma_exact(fx, fy, fz)); break;
    case 24: r = as_type<uint>(aps5_fmul(fx, fy)); break;
    case 25: r = as_type<uint>(aps5_fmul_keep(fx, fy)); break;
    case 26: r = as_type<uint>(aps5_fadd_keep(fx, fy)); break;
    case 27: r = as_type<uint>(aps5_fadd_keep(fx, as_type<float>(uint(y) ^ 0x80000000u))); break;
    case 28: r = as_type<uint>(aps5_fma(fx, fy, fz)); break;
    case 29: r = as_type<uint>(aps5_fma_keep(fx, fy, fz)); break;
    case 30: r = as_type<uint>(aps5_fmin(fx, fy)); break;
    case 31: r = as_type<uint>(aps5_fmax(fx, fy)); break;
    case 32: r = as_type<uint>(aps5_fmed3(fx, fy, fz)); break;
    case 33: r = aps5_f32_class(fx, uint(y) & 0x3ffu) ? 1ul : 0ul; break;
    case 34: r = as_type<uint>(aps5_fsat(fx)); break;
    case 35: r = (aps5_flt(fx, fy) ? 1ul : 0ul) | (aps5_fle(fx, fy) ? 2ul : 0ul) | (aps5_feq(fx, fy) ? 4ul : 0ul) | (aps5_fne(fx, fy) ? 8ul : 0ul) | (aps5_fgt(fx, fy) ? 16ul : 0ul) | (aps5_fge(fx, fy) ? 32ul : 0ul); break;
    case 36: r = as_type<uint>(aps5_floor_keep(fx)); break;
    case 37: r = as_type<uint>(aps5_ceil_keep(fx)); break;
    case 38: r = as_type<uint>(aps5_f16_to_f32(ushort(x))); break;
    case 39: r = aps5_f32_to_f16(fx); break;
    case 40: r = aps5_f32_to_f16_rtz(fx); break;
    case 41: r = as_type<uint>(aps5_f16_to_f32_native(ushort(x))); break;
    }
    result[i] = r;
}
)";

constexpr std::uint64_t Sign64 = 0x8000000000000000ull;
constexpr std::uint64_t Quiet64 = 0x0008000000000000ull;
constexpr std::uint64_t DefaultNan64 = 0xfff8000000000000ull;

double D(std::uint64_t bits) { return std::bit_cast<double>(bits); }
std::uint64_t B(double value) { return std::bit_cast<std::uint64_t>(value); }
float F(std::uint64_t bits) { return std::bit_cast<float>(static_cast<std::uint32_t>(bits)); }
std::uint64_t B32(float value) { return std::bit_cast<std::uint32_t>(value); }
bool Nan64(std::uint64_t x) { return (x & ~Sign64) > 0x7ff0000000000000ull; }
bool Nan32(std::uint64_t x) { return (x & 0x7fffffffu) > 0x7f800000u; }
bool Denormal32(std::uint64_t x) { return (x & 0x7fffffffu) != 0u && (x & 0x7f800000u) == 0u; }
std::int64_t Key32(std::uint64_t x) { const auto m = static_cast<std::int64_t>(x & 0x7fffffffu); return (x & 0x80000000u) != 0u ? -m : m; }

std::uint64_t NanRule(std::initializer_list<std::uint64_t> inputs, double value) {
    for (const auto input : inputs) {
        if (Nan64(input)) return input | Quiet64;
    }
    return std::isnan(value) ? DefaultNan64 : B(value);
}

// Expected result: exact bits, any NaN (f32 helpers leave NaN payloads to the front end), or no check.
struct Expect {
    enum class Kind { Bits, AnyNan32, Skip } kind = Kind::Skip;
    std::uint64_t bits = 0;
    static Expect Of(std::uint64_t value) { return {Kind::Bits, value}; }
    static Expect F32(float value) { return std::isnan(value) ? Expect{Kind::AnyNan32, 0} : Of(B32(value)); }
    static Expect None() { return {}; }
};

// splitmix64: reproducible operands for a given seed.
struct Random {
    std::uint64_t state;
    std::uint64_t Next() {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    std::uint64_t Below(std::uint64_t bound) { return Next() % bound; }
};

std::uint64_t Operand64(Random& random) {
    static constexpr std::uint64_t Specials[] = {0, Sign64, 0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8000000000000ull, 0xfff8000000000001ull,
        0x7ff0000000000001ull, 0x7ff4000000001234ull, 1, Sign64 | 1, 0x000fffffffffffffull, 0x0010000000000000ull, 0x7fefffffffffffffull,
        0x3ff0000000000000ull, 0xbff0000000000000ull, 0x3fe0000000000000ull, 0x41e0000000000000ull, 0xc1e0000000000000ull, 0x41f0000000000000ull,
        0x3ff8000000000000ull, 0x4330000000000000ull, 0x4330000000000001ull, 0x3fefffffffffffffull, 0x0008000000000000ull};
    const auto kind = random.Below(100);
    const std::uint64_t sign = random.Below(2) != 0 ? Sign64 : 0;
    std::uint64_t mantissa = random.Next() & 0x000fffffffffffffull;
    if (random.Below(4) == 0) mantissa &= ~((1ull << random.Below(53)) - 1ull);
    if (kind < 8) return Specials[random.Below(std::size(Specials))];
    if (kind < 40) return sign | (random.Below(0x7ff) << 52) | mantissa;
    if (kind < 55) return sign | mantissa;
    if (kind < 70) return sign | ((1 + random.Below(60)) << 52) | mantissa;
    return sign | ((973 + random.Below(101)) << 52) | mantissa;
}

std::uint64_t Operand32(Random& random) {
    static constexpr std::uint32_t Specials[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc12345u, 0x7f812345u, 1, 0x80000001u,
        0x007fffffu, 0x00800000u, 0x80800000u, 0x00800001u, 0x3f7ffffeu, 0x3f800000u, 0xbf800000u, 0x3f000000u, 0x7f7fffffu, 0x33800000u,
        0x33000000u, 0x33000001u, 0x477fe000u, 0x477ff000u, 0x38800000u, 0x387fc000u};
    const auto kind = random.Below(100);
    const std::uint32_t sign = random.Below(2) != 0 ? 0x80000000u : 0u;
    std::uint32_t mantissa = static_cast<std::uint32_t>(random.Next()) & 0x007fffffu;
    if (random.Below(4) == 0) mantissa &= ~((1u << random.Below(24)) - 1u);
    if (kind < 8) return Specials[random.Below(std::size(Specials))];
    if (kind < 35) return sign | (static_cast<std::uint32_t>(random.Below(0xff)) << 23) | mantissa;
    if (kind < 50) return sign | mantissa;
    if (kind < 70) return sign | (static_cast<std::uint32_t>(1 + random.Below(30)) << 23) | mantissa;
    return sign | (static_cast<std::uint32_t>(90 + random.Below(61)) << 23) | mantissa;
}

// A value a few units (in some low bit position) away from x, with a random sign.
std::uint64_t Near64(Random& random, std::uint64_t x) {
    const auto delta = (static_cast<std::int64_t>(random.Below(7)) - 3) << random.Below(41);
    const auto moved = std::clamp<std::int64_t>(static_cast<std::int64_t>(x & ~Sign64) + delta, 0, 0x7fefffffffffffffll);
    return static_cast<std::uint64_t>(moved) | (random.Below(2) != 0 ? Sign64 : 0);
}

std::uint64_t Near32(Random& random, std::uint64_t x) {
    const auto delta = (static_cast<std::int64_t>(random.Below(7)) - 3) << random.Below(21);
    const auto moved = std::clamp<std::int64_t>(static_cast<std::int64_t>(x & 0x7fffffffu) + delta, 0, 0x7f7fffffll);
    return static_cast<std::uint64_t>(moved) | (random.Below(2) != 0 ? 0x80000000u : 0u);
}

std::uint64_t Median3(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    const auto pick = [](std::uint64_t l, std::uint64_t r, bool maxValue) {
        const auto either = l | r;
        if ((either & 0x7fffffffu) == 0u) return maxValue ? (l & r) : either;
        if (Nan32(l) || Nan32(r)) return r;
        return (maxValue ? Key32(l) >= Key32(r) : Key32(l) < Key32(r)) ? l : r;
    };
    const auto minXy = Nan32(b) ? a : pick(a, b, false);
    if (Nan32(a) || Nan32(b) || Nan32(c)) return Nan32(c) ? minXy : pick(minXy, c, false);
    return pick(minXy, pick(pick(a, b, true), c, false), true);
}

struct Operation {
    const char* name;
    // Fills the operands of one case (x, y, z), returns the expected result.
    std::function<Expect(Random&, std::uint64_t&, std::uint64_t&, std::uint64_t&)> make;
};

std::vector<Operation> Operations() {
    const auto three64 = [](Random& r, std::uint64_t& x, std::uint64_t& y, std::uint64_t& z) { x = Operand64(r); y = Operand64(r); z = Operand64(r); };
    const auto three32 = [](Random& r, std::uint64_t& x, std::uint64_t& y, std::uint64_t& z) { x = Operand32(r); y = Operand32(r); z = Operand32(r); };
    std::vector<Operation> ops;
    ops.push_back({"f64 add", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (r.Below(3) == 0) y = Near64(r, x);
        return Expect::Of(NanRule({x, y}, D(x) + D(y)));
    }});
    ops.push_back({"f64 add to odd", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (r.Below(3) == 0) y = Near64(r, x);
        if (Nan64(x) || Nan64(y)) return Expect::Of(NanRule({x, y}, 0.0));
        const double s = D(x) + D(y);
        if (!std::isfinite(s)) return Expect::None();
        const double bp = s - D(x);
        const double ap = s - bp;
        const double e = (D(x) - ap) + (D(y) - bp);
        if (e == 0.0) return Expect::Of(B(s));
        std::uint64_t bits = B(s);
        if ((s > 0.0) != (e > 0.0)) bits -= 1u;
        return Expect::Of(bits | 1u);
    }});
    ops.push_back({"f64 mul", [=](Random& r, auto& x, auto& y, auto& z) { three64(r, x, y, z); return Expect::Of(NanRule({x, y}, D(x) * D(y))); }});
    ops.push_back({"f64 div", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (r.Below(4) == 0) x = B(D(y) * static_cast<double>(1 + r.Below(4096)));
        return Expect::Of(NanRule({x, y}, D(x) / D(y)));
    }});
    ops.push_back({"f64 sqrt", [=](Random& r, auto& x, auto& y, auto& z) { three64(r, x, y, z); return Expect::Of(NanRule({x}, std::sqrt(D(x)))); }});
    ops.push_back({"f64 fma", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (r.Below(3) == 0) z = Near64(r, B(D(x) * D(y)) ^ Sign64);
        return Expect::Of(NanRule({x, y, z}, std::fma(D(x), D(y), D(z))));
    }});
    ops.push_back({"f64 fma scaled", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        x = (x & ~(0x7ffull << 52)) | ((900 + r.Below(201)) << 52);
        z = (z & ~(0x7ffull << 52)) | ((900 + r.Below(201)) << 52);
        if (Nan64(x) || Nan64(y) || Nan64(z)) return Expect::Of(NanRule({x, y, z}, 0.0));
        const int scale = ((z >> 52) & 1u) != 0u ? 128 : -128;
        const double xs = std::ldexp(D(x), scale);
        const double zs = std::ldexp(D(z), scale);
        if (std::ldexp(xs, -scale) != D(x) || std::ldexp(zs, -scale) != D(z)) return Expect::None();
        return Expect::Of(NanRule({x, y, z}, std::fma(xs, D(y), zs)));
    }});
    ops.push_back({"f32->f64", [=](Random& r, auto& x, auto& y, auto& z) {
        x = Operand32(r); y = z = 0;
        if (Nan32(x)) return Expect::Of(((x & 0x80000000u) << 32) | 0x7ff8000000000000ull | ((x & 0x007fffffu) << 29));
        return Expect::Of(B(static_cast<double>(F(x))));
    }});
    ops.push_back({"f64->f32", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (r.Below(2) == 0) x = (x & ~(0x7ffull << 52)) | ((850 + r.Below(330)) << 52);
        if (Nan64(x)) return Expect::Of(((x >> 32) & 0x80000000u) | 0x7fc00000u | ((x & 0x000fffffffffffffull) >> 29));
        return Expect::Of(B32(static_cast<float>(D(x))));
    }});
    ops.push_back({"f64->i32", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (Nan64(x)) return Expect::Of(0);
        const double v = D(x);
        if (v <= -2147483648.0) return Expect::Of(0x80000000u);
        if (v >= 2147483648.0) return Expect::Of(0x7fffffffu);
        return Expect::Of(static_cast<std::uint32_t>(static_cast<std::int32_t>(std::trunc(v))));
    }});
    ops.push_back({"f64->u32", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (Nan64(x) || D(x) <= 0.0) return Expect::Of(0);
        if (D(x) >= 4294967296.0) return Expect::Of(0xffffffffu);
        return Expect::Of(static_cast<std::uint32_t>(std::trunc(D(x))));
    }});
    ops.push_back({"i32->f64", [=](Random& r, auto& x, auto& y, auto& z) { x = r.Next() & 0xffffffffu; y = z = 0; return Expect::Of(B(static_cast<double>(static_cast<std::int32_t>(x)))); }});
    ops.push_back({"u32->f64", [=](Random& r, auto& x, auto& y, auto& z) { x = r.Next() & 0xffffffffu; y = z = 0; return Expect::Of(B(static_cast<double>(static_cast<std::uint32_t>(x)))); }});
    ops.push_back({"f64 trunc", [=](Random& r, auto& x, auto& y, auto& z) { three64(r, x, y, z); return Expect::Of(Nan64(x) ? x | Quiet64 : B(std::trunc(D(x)))); }});
    ops.push_back({"f64 floor", [=](Random& r, auto& x, auto& y, auto& z) { three64(r, x, y, z); return Expect::Of(Nan64(x) ? x | Quiet64 : B(std::floor(D(x)))); }});
    ops.push_back({"f64 ceil", [=](Random& r, auto& x, auto& y, auto& z) { three64(r, x, y, z); return Expect::Of(Nan64(x) ? x | Quiet64 : B(std::ceil(D(x)))); }});
    ops.push_back({"f64 round even", [=](Random& r, auto& x, auto& y, auto& z) { three64(r, x, y, z); return Expect::Of(Nan64(x) ? x | Quiet64 : B(std::nearbyint(D(x)))); }});
    ops.push_back({"f64 fract", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (Nan64(x)) return Expect::Of(x | Quiet64);
        const double fraction = D(x) - std::floor(D(x));
        if (std::isnan(fraction)) return Expect::Of(DefaultNan64);
        return Expect::Of(B(std::min(fraction, D(0x3fefffffffffffffull))));
    }});
    ops.push_back({"f64 ldexp", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        y = static_cast<std::uint32_t>(static_cast<std::int32_t>(r.Below(4401)) - 2200);
        if (Nan64(x)) return Expect::Of(x | Quiet64);
        if (D(x) == 0.0 || std::isinf(D(x))) return Expect::Of(x);
        return Expect::Of(B(std::ldexp(D(x), static_cast<std::int32_t>(y))));
    }});
    ops.push_back({"f64 frexp mantissa", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (((x >> 52) & 0x7ffu) == 0x7ffu) return Expect::Of(Nan64(x) ? x | Quiet64 : x);
        int exponent = 0;
        return Expect::Of(B(std::frexp(D(x), &exponent)));
    }});
    ops.push_back({"f64 frexp exponent", [=](Random& r, auto& x, auto& y, auto& z) {
        three64(r, x, y, z);
        if (((x >> 52) & 0x7ffu) == 0x7ffu || (x & ~Sign64) == 0) return Expect::Of(0);
        int exponent = 0;
        std::frexp(D(x), &exponent);
        return Expect::Of(static_cast<std::uint32_t>(exponent));
    }});
    for (const bool maxValue : {false, true}) {
        ops.push_back({maxValue ? "f64 max" : "f64 min", [=](Random& r, auto& x, auto& y, auto& z) {
            three64(r, x, y, z);
            const auto snan = [](std::uint64_t v) { return Nan64(v) && (v & Quiet64) == 0; };
            if (Nan64(x) || Nan64(y)) {
                std::uint64_t result = Nan64(y) ? x : y;
                if (snan(y)) result = y | Quiet64;
                if (snan(x)) result = x | Quiet64;
                return Expect::Of(result);
            }
            if ((x & ~Sign64) == 0 && (y & ~Sign64) == 0) {
                const bool negative = (x & Sign64) != 0;
                return Expect::Of(maxValue ? (negative ? y : x) : (negative ? x : y));
            }
            return Expect::Of((maxValue ? D(x) > D(y) : D(x) < D(y)) ? x : y);
        }});
    }
    ops.push_back({"f32 fma exact", [=](Random& r, auto& x, auto& y, auto& z) {
        three32(r, x, y, z);
        if (r.Below(3) == 0) z = Near32(r, B32(F(x) * F(y)) ^ 0x80000000u);
        return Expect::F32(std::fma(F(x), F(y), F(z)));
    }});
    ops.push_back({"f32 mul (flush mode)", [=](Random& r, auto& x, auto& y, auto& z) {
        three32(r, x, y, z);
        if (r.Below(3) == 0) { x = 0x00400000u + r.Below(0x00c00000u); y = 0x3f000000u + r.Below(0x00800001u); }
        if (Denormal32(x) || Denormal32(y)) return Expect::None();
        return Expect::F32(F(x) * F(y));
    }});
    ops.push_back({"f32 mul (keep mode)", [=](Random& r, auto& x, auto& y, auto& z) {
        three32(r, x, y, z);
        if (r.Below(3) == 0) { x = 0x00400000u + r.Below(0x00c00000u); y = 0x3f000000u + r.Below(0x00800001u); }
        return Expect::F32(F(x) * F(y));
    }});
    ops.push_back({"f32 add (keep mode)", [=](Random& r, auto& x, auto& y, auto& z) { three32(r, x, y, z); if (r.Below(3) == 0) y = Near32(r, x); return Expect::F32(F(x) + F(y)); }});
    ops.push_back({"f32 sub (keep mode)", [=](Random& r, auto& x, auto& y, auto& z) { three32(r, x, y, z); if (r.Below(3) == 0) y = Near32(r, x); return Expect::F32(F(x) - F(y)); }});
    ops.push_back({"f32 fma (flush mode)", [=](Random& r, auto& x, auto& y, auto& z) {
        three32(r, x, y, z);
        if (r.Below(3) == 0) z = Near32(r, B32(F(x) * F(y)) ^ 0x80000000u);
        if (r.Below(3) == 0) {
            // A product below 2^-126 next to a small normal addend: the fused result must keep it.
            x = (x & 0x807fffffu) | ((60 + r.Below(31)) << 23);
            y = (y & 0x807fffffu) | ((60 + r.Below(31)) << 23);
            z = (z & 0x807fffffu) | ((1 + r.Below(40)) << 23);
        }
        if (Denormal32(x) || Denormal32(y) || Denormal32(z)) return Expect::None();
        return Expect::F32(std::fma(F(x), F(y), F(z)));
    }});
    ops.push_back({"f32 fma (keep mode)", [=](Random& r, auto& x, auto& y, auto& z) {
        three32(r, x, y, z);
        if (r.Below(3) == 0) z = Near32(r, B32(F(x) * F(y)) ^ 0x80000000u);
        return Expect::F32(std::fma(F(x), F(y), F(z)));
    }});
    for (const bool maxValue : {false, true}) {
        ops.push_back({maxValue ? "f32 max" : "f32 min", [=](Random& r, auto& x, auto& y, auto& z) {
            three32(r, x, y, z);
            if (Nan32(y)) return Expect::Of(x);
            if (Nan32(x)) return Expect::Of(y);
            if (((x | y) & 0x7fffffffu) == 0u) return Expect::Of(maxValue ? (x & y) : (x | y));
            return Expect::Of((maxValue ? Key32(x) >= Key32(y) : Key32(x) < Key32(y)) ? x : y);
        }});
    }
    ops.push_back({"f32 med3", [=](Random& r, auto& x, auto& y, auto& z) { three32(r, x, y, z); return Expect::Of(Median3(x, y, z)); }});
    ops.push_back({"f32 class", [=](Random& r, auto& x, auto& y, auto& z) {
        three32(r, x, y, z);
        const auto m = x & 0x7fffffffu;
        const bool negative = (x & 0x80000000u) != 0u;
        std::uint64_t cls = 0;
        if ((m & 0x7f800000u) == 0x7f800000u) cls = (m & 0x007fffffu) != 0u ? ((m & 0x00400000u) != 0u ? 2u : 1u) : (negative ? 4u : 0x200u);
        else if ((m & 0x7f800000u) != 0u) cls = negative ? 8u : 0x100u;
        else if (m != 0u) cls = negative ? 0x10u : 0x80u;
        else cls = negative ? 0x20u : 0x40u;
        return Expect::Of((cls & y & 0x3ffu) != 0u ? 1u : 0u);
    }});
    ops.push_back({"f32 saturate", [=](Random& r, auto& x, auto& y, auto& z) {
        three32(r, x, y, z);
        if (Nan32(x) || (x & 0x80000000u) != 0u || x == 0u) return Expect::Of(0);
        return Expect::Of(std::min<std::uint64_t>(x, 0x3f800000u));
    }});
    ops.push_back({"f32 compares on bits", [=](Random& r, auto& x, auto& y, auto& z) {
        three32(r, x, y, z);
        if (r.Below(3) == 0) y = Near32(r, x);
        const float a = F(x);
        const float b = F(y);
        return Expect::Of((a < b ? 1u : 0u) | (a <= b ? 2u : 0u) | (a == b ? 4u : 0u) | (a < b || a > b ? 8u : 0u) | (a > b ? 16u : 0u) | (a >= b ? 32u : 0u));
    }});
    ops.push_back({"f32 floor (keep mode)", [=](Random& r, auto& x, auto& y, auto& z) { three32(r, x, y, z); return Expect::F32(std::floor(F(x))); }});
    ops.push_back({"f32 ceil (keep mode)", [=](Random& r, auto& x, auto& y, auto& z) { three32(r, x, y, z); return Expect::F32(std::ceil(F(x))); }});
    ops.push_back({"f16->f32", [=](Random& r, auto& x, auto& y, auto& z) {
        x = r.Next() & 0xffffu; y = z = 0;
        if ((x & 0x7fffu) > 0x7c00u) return Expect::Of(((x & 0x8000u) << 16) | 0x7f800000u | ((x & 0x3ffu) << 13));
        return Expect::Of(B32(static_cast<float>(std::bit_cast<_Float16>(static_cast<std::uint16_t>(x)))));
    }});
    for (const bool towardZero : {false, true}) {
        ops.push_back({towardZero ? "f32->f16 toward zero" : "f32->f16", [=](Random& r, auto& x, auto& y, auto& z) {
            three32(r, x, y, z);
            if (r.Below(2) == 0) x = (x & 0x807fffffu) | ((98 + r.Below(48)) << 23);
            if (Nan32(x)) return Expect::Of(((x >> 16) & 0x8000u) | 0x7e00u | ((x & 0x007fffffu) >> 13));
            std::uint16_t half = std::bit_cast<std::uint16_t>(static_cast<_Float16>(F(x)));
            if (towardZero && std::fabs(static_cast<float>(std::bit_cast<_Float16>(half))) > std::fabs(F(x))) half -= 1u;
            return Expect::Of(half);
        }});
    }
    // Upstream #3988: the hardware half to float conversion with the NaN bits rebuilt
    // (MslTargetOptions::float16DenormPreserve). Every half, denormals included, must convert exactly:
    // this is the measurement behind the Metal target's float16DenormPreserve.
    ops.push_back({"f16->f32 (native)", [=](Random& r, auto& x, auto& y, auto& z) {
        x = r.Below(4) == 0 ? (r.Next() & 0x83ffu) : (r.Next() & 0xffffu); y = z = 0;
        if ((x & 0x7fffu) > 0x7c00u) return Expect::Of(((x & 0x8000u) << 16) | 0x7f800000u | ((x & 0x3ffu) << 13));
        return Expect::Of(B32(static_cast<float>(std::bit_cast<_Float16>(static_cast<std::uint16_t>(x)))));
    }});
    return ops;
}

bool Wait(id<MTLCommandBuffer> buffer) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [buffer addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(done); }];
    [buffer commit];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10ll * NSEC_PER_SEC)) != 0) {
        std::fprintf(stderr, "FAIL: GPU watchdog (10 s)\n");
        std::exit(1);
    }
    return buffer.status == MTLCommandBufferStatusCompleted;
}

}

int main(int argc, char** argv) {
    @autoreleasepool {
        const std::uint32_t cases = argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 0)) : 20000u;
        const std::uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 0) : 0x0b2a1u;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::puts("skipped, the machine has no Metal device");
            return 77;
        }
        std::string source = "#include <metal_stdlib>\nusing namespace metal;\n";
        for (const auto chunk : {MslPreludeChunk::Alu, MslPreludeChunk::Float16, MslPreludeChunk::Float64}) source += MslPrelude(chunk).text;
        source += KernelText;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
        if (library == nil) {
            std::fprintf(stderr, "FAIL: the ALU prelude does not compile: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"aps5_alu_check"] error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (pipeline == nil || queue == nil) {
            std::fprintf(stderr, "FAIL: no pipeline or queue: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        const NSUInteger bytes = static_cast<NSUInteger>(cases) * 8u;
        id<MTLBuffer> buffers[4];
        for (auto& buffer : buffers) buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Random random{seed};
        const auto operations = Operations();
        std::uint64_t failures = 0;
        std::vector<Expect> expected(cases);
        for (std::uint32_t op = 0; op < operations.size(); ++op) {
            auto* a = static_cast<std::uint64_t*>(buffers[0].contents);
            auto* b = static_cast<std::uint64_t*>(buffers[1].contents);
            auto* c = static_cast<std::uint64_t*>(buffers[2].contents);
            for (std::uint32_t i = 0; i < cases; ++i) expected[i] = operations[op].make(random, a[i], b[i], c[i]);
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            for (NSUInteger index = 0; index < 4u; ++index) [encoder setBuffer:buffers[index] offset:0 atIndex:index];
            [encoder setBytes:&op length:4 atIndex:4];
            [encoder setBytes:&cases length:4 atIndex:5];
            [encoder dispatchThreads:MTLSizeMake(cases, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding];
            if (!Wait(commands)) {
                std::fprintf(stderr, "FAIL: %s: command buffer error %s\n", operations[op].name, commands.error.localizedDescription.UTF8String);
                return 1;
            }
            const auto* result = static_cast<const std::uint64_t*>(buffers[3].contents);
            std::uint32_t opFailures = 0;
            std::uint32_t checked = 0;
            for (std::uint32_t i = 0; i < cases; ++i) {
                const Expect& want = expected[i];
                if (want.kind == Expect::Kind::Skip) continue;
                ++checked;
                const bool ok = want.kind == Expect::Kind::AnyNan32 ? Nan32(result[i]) : result[i] == want.bits;
                if (ok) continue;
                if (++opFailures <= 4u) std::fprintf(stderr, "FAIL: %s a=0x%016llx b=0x%016llx c=0x%016llx gpu=0x%016llx expected=0x%016llx%s\n", operations[op].name, static_cast<unsigned long long>(a[i]), static_cast<unsigned long long>(b[i]), static_cast<unsigned long long>(c[i]), static_cast<unsigned long long>(result[i]), static_cast<unsigned long long>(want.bits), want.kind == Expect::Kind::AnyNan32 ? " (a NaN)" : "");
            }
            std::printf("%-24s %6u checked, %u failures\n", operations[op].name, checked, opFailures);
            failures += opFailures;
        }
        if (failures != 0u) {
            std::fprintf(stderr, "FAIL: %llu mismatches (seed 0x%llx)\n", static_cast<unsigned long long>(failures), static_cast<unsigned long long>(seed));
            return 1;
        }
        std::puts("MSL ALU helpers match the CPU references");
        return 0;
    }
}
