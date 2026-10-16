// agc_metal_native_f32_flush (DECISIONS D54): with SpirvTarget::float32DenormFlushNative the front
// end leaves the f32 flushes of v_add/v_sub/v_subrev/v_mul_f32 and the f32 compares (FLOAT_MODE f32
// flush, round to nearest even) to the device, and v_mul_f32 computes RDNA's tininess after rounding
// in the MSL helper aps5_fmul_ftz. The program below runs over every pair of special values, random
// values and products around 2^-126 twice per float mode: once translated with the flag and once
// without it (the front end's emulation, which agc_driver_f32_denormal_flush checks against RDNA
// results). Every result must be bit-identical, NaNs included. The program also has operations the
// flag does not change (fma, min/max/med3, fract/floor/ceil, conversions, transcendentals) and
// output modifiers, so a change in their surroundings shows up too.
//
// Guest program (llvm-mc -arch=amdgcn -mcpu=gfx1030; user data s[0:3] input, s[4:7] output, s8
// workgroup id; 4 input dwords v4-v7 and 40 results v10-v49 per invocation).

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;


constexpr std::uint32_t GroupThreads = 256;
constexpr std::uint32_t Inputs = 4;
constexpr std::uint32_t Results = 40;

alignas(256) constexpr std::uint32_t Code[] = {
    0x8f098808u, 0x4a040009u, 0x34020484u, 0x160604ffu, 0x000000a0u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u,
    0x06140b04u, 0x08160905u, 0x0a180d04u, 0x101a0b04u, 0x101c0d05u, 0xd503020fu, 0x22020b04u, 0xd5080110u,
    0x42020d04u, 0xd5088011u, 0x02020b04u, 0xd5080012u, 0x0a020b04u, 0xd5080013u, 0x1a020d04u, 0xd5030014u,
    0x12020d05u, 0xd54b0015u, 0x041a0b04u, 0x7e2c0306u, 0x562c0b04u, 0x582e0d04u, 0x3f000000u, 0x1e300b04u,
    0x20320d05u, 0xd557001au, 0x041a0b04u, 0x7e364104u, 0x7e384905u, 0x7e3a4506u, 0x7c020b04u, 0xd501001eu,
    0x01a90280u, 0x7c040d05u, 0xd501001fu, 0x01a90280u, 0x7c0c0d04u, 0xd5010020u, 0x01a90280u, 0x7c1a0b04u,
    0xd5010021u, 0x01a90280u, 0x7c1c0906u, 0xd5010022u, 0x01a90280u, 0xd404006au, 0x02010104u, 0xd5010023u,
    0x01a90280u, 0x7e481104u, 0x7e4a0f05u, 0x7e4c6b04u, 0x5e4e0d05u, 0x105008f8u, 0x06520affu, 0x80000001u,
    0x0e540b04u, 0x7e566705u, 0x7e585506u, 0x7e5a4b04u, 0x7e5c4f05u, 0xd762002fu, 0x02018704u, 0x10601d0du,
    0x06621b0au, 0xe0781000u, 0x80010a03u, 0xe0781010u, 0x80010e03u, 0xe0781020u, 0x80011203u, 0xe0781030u,
    0x80011603u, 0xe0781040u, 0x80011a03u, 0xe0781050u, 0x80011e03u, 0xe0781060u, 0x80012203u, 0xe0781070u,
    0x80012603u, 0xe0781080u, 0x80012a03u, 0xe0781090u, 0x80012e03u, 0xbf810000u
};

constexpr const char* Names[Results] = {
    "v_add_f32 v10, v4, v5",
    "v_sub_f32 v11, v5, v4",
    "v_subrev_f32 v12, v4, v6",
    "v_mul_f32 v13, v4, v5",
    "v_mul_f32 v14, v5, v6",
    "v_add_f32_e64 v15, -v4, |v5|",
    "v_mul_f32_e64 v16, |v4|, -v6",
    "v_mul_f32_e64 v17, v4, v5 clamp",
    "v_mul_f32_e64 v18, v4, v5 mul:2",
    "v_mul_f32_e64 v19, v4, v6 div:2",
    "v_add_f32_e64 v20, v5, v6 mul:4",
    "v_fma_f32 v21, v4, v5, v6",
    "v_fmac_f32 v22, v4, v5",
    "v_fmamk_f32 v23, v4, 0x3f000000, v6",
    "v_min_f32 v24, v4, v5",
    "v_max_f32 v25, v5, v6",
    "v_med3_f32 v26, v4, v5, v6",
    "v_fract_f32 v27, v4",
    "v_floor_f32 v28, v5",
    "v_ceil_f32 v29, v6",
    "v_cmp_lt_f32 vcc_lo, v4, v5",
    "v_cmp_eq_f32 vcc_lo, v5, v6",
    "v_cmp_ge_f32 vcc_lo, v4, v6",
    "v_cmp_neq_f32 vcc_lo, v4, v5",
    "v_cmp_nlt_f32 vcc_lo, v6, v4",
    "v_cmp_gt_f32 vcc_lo, v4, 0",
    "v_cvt_i32_f32 v36, v4",
    "v_cvt_u32_f32 v37, v5",
    "v_sin_f32 v38, v4",
    "v_cvt_pkrtz_f16_f32 v39, v5, v6",
    "v_mul_f32 v40, 0x3e22f983, v4",
    "v_add_f32 v41, 0x80000001, v5",
    "v_mul_legacy_f32 v42, v4, v5",
    "v_sqrt_f32 v43, v5",
    "v_rcp_f32 v44, v6",
    "v_exp_f32 v45, v4",
    "v_log_f32 v46, v5",
    "v_ldexp_f32 v47, v4, -3",
    "v_mul_f32 v48, v13, v14",
    "v_add_f32 v49, v10, v13"
};

std::uint32_t Bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

// Special f32 values: signed zeros, denormals, the smallest normals and their neighbours, values
// whose products land near 2^-126, ordinary values, infinities and NaNs with payloads.
std::vector<std::uint32_t> Specials() {
    std::vector<std::uint32_t> values{
        0x00000000u, 0x00000001u, 0x00000002u, 0x00012345u, 0x00400000u, 0x007fffffu, 0x00800000u, 0x00800001u,
        0x00ffffffu, 0x01000000u, 0x0c000000u, 0x1f000000u, 0x1f7fffffu, 0x1f800000u, 0x1fffffffu, 0x20000000u,
        0x20000001u, 0x207fffffu, 0x20800000u, 0x2f800000u, 0x34000000u, 0x3dcccccdu, 0x3e22f983u, 0x3f000000u,
        0x3f000001u, 0x3f7ffffeu, 0x3f7fffffu, 0x3f800000u, 0x3f800001u, 0x3fc00000u, 0x40000000u, 0x40490fdbu,
        0x41200000u, 0x4b000000u, 0x4b800000u, 0x5f000000u, 0x5f800000u, 0x7effffffu, 0x7f7fffffu, 0x7f800000u,
        0x7f800001u, 0x7fa00000u, 0x7fc00000u, 0x7fc12345u, 0x7fffffffu, 0x5e800000u, 0x60000000u, 0x00000003u,
    };
    const auto count = values.size();
    for (std::size_t index = 0; index < count; ++index) values.push_back(values[index] ^ 0x80000000u);
    return values;
}

struct Random {
    std::uint64_t state = 0x9e3779b97f4a7c15ull;
    std::uint32_t Next() {
        state ^= state << 13u;
        state ^= state >> 7u;
        state ^= state << 17u;
        return static_cast<std::uint32_t>(state >> 16u);
    }
};

std::vector<std::array<std::uint32_t, Inputs>> Rows() {
    const auto specials = Specials();
    std::vector<std::array<std::uint32_t, Inputs>> rows;
    const std::array<std::uint32_t, 4> addends{0x00000000u, 0x80000001u, 0x3f800000u, 0x007fffffu};
    std::size_t addend = 0;
    for (const auto a : specials) {
        for (const auto b : specials) rows.push_back({a, b, specials[(addend * 7u + 3u) % specials.size()] ^ (addends[addend++ % addends.size()] & 0x80000000u), 0u});
    }
    Random random;
    // Random bit patterns, and random values of moderate and tiny magnitude.
    for (std::uint32_t index = 0; index < 4096u; ++index) rows.push_back({random.Next(), random.Next(), random.Next(), 0u});
    for (std::uint32_t index = 0; index < 4096u; ++index) {
        const auto mantissa = [&] { return random.Next() & 0x807fffffu; };
        rows.push_back({mantissa() | ((100u + random.Next() % 60u) << 23u), mantissa() | ((random.Next() % 40u) << 23u), mantissa() | ((random.Next() % 4u) << 23u), 0u});
    }
    // Products within a few ulps of 2^-126 (both sides): b = 2^-126 / a, nudged.
    for (std::uint32_t index = 0; index < 4096u; ++index) {
        const float a = std::ldexp(1.0f + static_cast<float>(random.Next() & 0xffffffu) / 16777216.0f, -static_cast<int>(random.Next() % 100u) - 1);
        const float b = std::ldexp(1.0f, -126) / a;
        const std::int32_t nudge = static_cast<std::int32_t>(random.Next() % 9u) - 4;
        const std::uint32_t sign = random.Next() & 0x80000000u;
        rows.push_back({Bits(a) ^ sign, static_cast<std::uint32_t>(static_cast<std::int32_t>(Bits(b)) + nudge), random.Next() & 0x80ffffffu, 0u});
    }
    while (rows.size() % GroupThreads != 0u) rows.push_back({0u, 0u, 0u, 0u});
    return rows;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint64_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(bytes), 0x31016facu};
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

struct Buffers {
    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> output;
};

std::vector<std::uint32_t> Run(AgcDriver::VulkanDevice& device, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode, bool native, Buffers& buffers, std::uint32_t groups) {
    std::fill(buffers.output.begin(), buffers.output.end(), 0xdeadbeefu);
    std::vector<std::uint32_t> userData(8, 0u);
    const auto input = BufferDescriptor(buffers.input.data(), buffers.input.size() * 4u);
    const auto output = BufferDescriptor(buffers.output.data(), buffers.output.size() * 4u);
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{GroupThreads, 1, 1}, 0u, {true, false, false}, false, 1};
    auto target = device.Target();
    target.float32DenormFlushNative = native;
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.context.floatMode = floatMode;
    const auto result = ShaderRecompiler::Recompile(request);
    if (native) {
        // The flag must reach the MSL: the flagged products use the helper.
        const bool flushMode = !floatMode.has_value() || (((floatMode->floatMode >> 4u) & 3u) == 0u && (floatMode->floatMode & 3u) == 0u);
        const auto& source = result.msl.Shader().source;
        Require(!flushMode || (source.find("aps5_fmul_ftz(v") != std::string::npos && source.find("aps5_fma_ftz(v") != std::string::npos),
            "native f32 flush: the MSL of a FLOAT_MODE flush program has no aps5_fmul_ftz product or aps5_fma_ftz");
        Require(flushMode || (source.find("aps5_fmul_ftz(v") == std::string::npos && source.find("aps5_fma_ftz(v") == std::string::npos),
            "native f32 flush: a program outside FLOAT_MODE flush with round to nearest even uses the native flush");
    }
    device.Dispatch(result, groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    return buffers.output;
}

bool IsNan(std::uint32_t value) {
    return (value & 0x7fffffffu) > 0x7f800000u;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (!device->Target().float32DenormFlushNative) {
            std::puts("skipped: the device target does not flush f32 denormals natively");
            return VulkanTestSkipped;
        }
        const auto rows = Rows();
        const auto groups = static_cast<std::uint32_t>(rows.size() / GroupThreads);
        Buffers buffers;
        buffers.input.resize(rows.size() * Inputs);
        for (std::size_t row = 0; row < rows.size(); ++row) std::copy(rows[row].begin(), rows[row].end(), buffers.input.begin() + static_cast<std::ptrdiff_t>(row * Inputs));
        buffers.output.resize(rows.size() * Results);
        const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 6> modes{
            std::nullopt,
            ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false},
            ShaderRecompiler::ShaderFloatMode{0x00u, false, true, false},
            ShaderRecompiler::ShaderFloatMode{0xc0u, false, true, false},
            ShaderRecompiler::ShaderFloatMode{0xc1u, true, false, false},
            ShaderRecompiler::ShaderFloatMode{0xf0u, true, false, false},
        };
        std::size_t compared = 0;
        for (const auto& mode : modes) {
            const std::string name = mode.has_value() ? "FLOAT_MODE " + Hex(mode->floatMode) + (mode->ieeeMode ? " IEEE" : "") : "no float mode";
            const auto emulated = Run(*device, mode, false, buffers, groups);
            const auto native = Run(*device, mode, true, buffers, groups);
            std::size_t differences = 0;
            for (std::size_t row = 0; row < rows.size(); ++row) {
                for (std::uint32_t result = 0; result < Results; ++result) {
                    const auto index = row * Results + result;
                    Require(emulated[index] != 0xdeadbeefu || native[index] == 0xdeadbeefu, "native f32 flush: " + name + " " + Names[result] + " was not written");
                    if (emulated[index] == native[index]) continue;
                    if (differences++ < 16u) {
                        std::printf("%s row %zu (%s %s %s) %s: emulated %s native %s%s\n", name.c_str(), row, Hex(rows[row][0]).c_str(), Hex(rows[row][1]).c_str(), Hex(rows[row][2]).c_str(), Names[result], Hex(emulated[index]).c_str(), Hex(native[index]).c_str(),
                            IsNan(emulated[index]) && IsNan(native[index]) ? " (both NaN)" : "");
                    }
                }
            }
            compared += rows.size() * Results;
            Require(differences == 0u, "native f32 flush: " + name + ": " + std::to_string(differences) + " result(s) differ from the emulated flush");
        }
        std::printf("native f32 flush: %zu rows x %u results x %zu float modes bit-identical (%zu results)\n", rows.size(), Results, modes.size(), compared);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
