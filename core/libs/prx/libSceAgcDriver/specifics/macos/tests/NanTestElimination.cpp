// agc_metal_nan_test_elimination (DECISIONS D63): NanTestEliminator drops the NaN tests of values
// that can never be a NaN, so RDNA's NaN emulation (quieted NaN sources, else 0xffc00000) stays only
// where a NaN can reach. The program below runs over every pair of special values (NaNs with payloads
// and signs, infinities, zeros, denormals, integer extremes) and random bit patterns, translated with
// and without SpirvTarget::mslNanTestElimination, in four float modes (IEEE on and off; FLOAT_MODE
// flush and keep): every result must be bit-identical, NaNs included, and the eliminating translation
// must have fewer NaN tests. It mixes results that cannot be NaNs (conversions of integers and
// arithmetic over them, clamps, fract/sqrt/floor of finite values, max with a non-NaN operand) with
// results that can (raw input bits, inf - inf and 0 * inf of exp2 of a converted integer, fract of an
// infinity, rcp(0) * 0, min of two raw inputs).
//
// Guest program (llvm-mc -arch=amdgcn -mcpu=gfx1030, wave32; user data s[0:3] input V#, s[4:7]
// output V#, s8 workgroup id; 2 input dwords v3-v4 and 26 results per invocation):
//   s_lshl_b32 s9, s8, 6
//   v_add_nc_u32 v1, s9, v0
//   v_lshlrev_b32 v2, 3, v1
//   buffer_load_dwordx2 v[3:4], v2, s[0:3], 0 offen
//   v_mul_u32_u24 v40, 0x70, v1
//   s_waitcnt vmcnt(0)
//   // never NaN: integer conversions and arithmetic over them
//   v_cvt_f32_u32 v5, v3
//   v_cvt_f32_i32 v6, v4
//   v_add_f32 v7, v5, v6
//   v_mul_f32 v8, v5, v6
//   v_fma_f32 v9, v5, v6, v7
//   // exp2 of a converted integer may be an infinity: inf - inf and 0 * inf are NaNs
//   v_exp_f32 v10, v6
//   v_sub_f32 v11, v10, v10
//   v_mul_f32 v12, 0, v10
//   // the raw input bits may be NaNs
//   v_add_f32 v13, v3, v5
//   v_max_f32 v14, v3, v5
//   v_add_f32 v15, 1.0, v14
//   v_mul_f32 v16, v14, v14
//   v_add_f32_e64 v17, v3, v4 clamp
//   v_mul_f32 v18, v17, v5
//   v_sqrt_f32 v19, v17
//   v_fract_f32 v20, v5
//   v_fract_f32 v21, v10
//   v_rcp_f32 v22, v6
//   v_mul_f32 v23, v22, v6
//   v_floor_f32 v24, v5
//   v_min_f32 v25, v3, v4
//   v_cvt_f32_ubyte0 v26, v3
//   v_med3_f32 v27, v5, v6, v7
//   v_sub_f32 v28, v5, v6
//   v_mul_f32 v28, v28, v28
//   v_sqrt_f32 v29, v28
//   v_add_f32 v30, v24, v20
//   buffer_store_dwordx4 v[5:8], v40, s[4:7], 0 offen
//   buffer_store_dwordx4 v[9:12], v40, s[4:7], 0 offen offset:16
//   buffer_store_dwordx4 v[13:16], v40, s[4:7], 0 offen offset:32
//   buffer_store_dwordx4 v[17:20], v40, s[4:7], 0 offen offset:48
//   buffer_store_dwordx4 v[21:24], v40, s[4:7], 0 offen offset:64
//   buffer_store_dwordx4 v[25:28], v40, s[4:7], 0 offen offset:80
//   buffer_store_dwordx2 v[29:30], v40, s[4:7], 0 offen offset:96
//   s_endpgm

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint32_t GroupThreads = 64;
constexpr std::uint32_t Inputs = 2;
constexpr std::uint32_t Stride = 28;
constexpr std::uint32_t Results = 26;

alignas(256) constexpr std::array<std::uint32_t, 53> Code{
    0x8f098608u, 0x4a020009u, 0x34040283u, 0xe0341000u, 0x80000302u, 0x165002ffu, 0x00000070u, 0xbf8c3f70u,
    0x7e0a0d03u, 0x7e0c0b04u, 0x060e0d05u, 0x10100d05u, 0xd54b0009u, 0x041e0d05u, 0x7e144b06u, 0x0816150au,
    0x10181480u, 0x061a0b03u, 0x201c0b03u, 0x061e1cf2u, 0x10201d0eu, 0xd5038011u, 0x02020903u, 0x10240b11u,
    0x7e266711u, 0x7e284105u, 0x7e2a410au, 0x7e2c5506u, 0x102e0d16u, 0x7e304905u, 0x1e320903u, 0x7e342303u,
    0xd557001bu, 0x041e0d05u, 0x08380d05u, 0x1038391cu, 0x7e3a671cu, 0x063c2918u, 0xe0781000u, 0x80010528u,
    0xe0781010u, 0x80010928u, 0xe0781020u, 0x80010d28u, 0xe0781030u, 0x80011128u, 0xe0781040u, 0x80011528u,
    0xe0781050u, 0x80011928u, 0xe0741060u, 0x80011d28u, 0xbf810000u
};

std::vector<std::uint32_t> Specials() {
    std::vector<std::uint32_t> values{
        0x00000000u, 0x00000001u, 0x00000002u, 0x00012345u, 0x007fffffu, 0x00800000u, 0x3f000000u, 0x3f800000u,
        0x3fc00000u, 0x40490fdbu, 0x4b000000u, 0x4f000000u, 0x5f800000u, 0x7effffffu, 0x7f7fffffu, 0x7f800000u,
        0x7f800001u, 0x7fa00000u, 0x7fc00000u, 0x7fc12345u, 0x7fffffffu, 0x0000001fu, 0x00000080u, 0x000000ffu,
        0x7ffffffeu, 0x42fe0000u, 0x43000000u, 0x00000100u,
    };
    const auto count = values.size();
    for (std::size_t index = 0; index < count; ++index) values.push_back(values[index] ^ 0x80000000u);
    return values;
}

std::vector<std::array<std::uint32_t, Inputs>> Rows() {
    const auto specials = Specials();
    std::vector<std::array<std::uint32_t, Inputs>> rows;
    for (const auto a : specials) {
        for (const auto b : specials) rows.push_back({a, b});
    }
    std::uint64_t state = 0x9e3779b97f4a7c15ull;
    const auto next = [&] {
        state ^= state << 13u;
        state ^= state >> 7u;
        state ^= state << 17u;
        return static_cast<std::uint32_t>(state >> 16u);
    };
    for (std::uint32_t index = 0; index < 8192u; ++index) rows.push_back({next(), next()});
    for (std::uint32_t index = 0; index < 4096u; ++index) rows.push_back({next() % 256u, next() % 128u});
    while (rows.size() % GroupThreads != 0u) rows.push_back({0u, 0u});
    return rows;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint64_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(bytes), 0x31016facu};
}

std::size_t NanTests(const std::string& source) {
    std::size_t count = 0;
    for (std::size_t at = source.find("0x7f800000u"); at != std::string::npos; at = source.find("0x7f800000u", at + 1u)) ++count;
    return count;
}

struct Translation {
    std::vector<std::uint32_t> output;
    std::size_t nanTests = 0;
    std::size_t bytes = 0;
};

Translation Run(AgcDriver::VulkanDevice& device, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode, bool eliminate, const std::vector<std::uint32_t>& input, std::uint32_t groups) {
    Translation translation;
    translation.output.assign(static_cast<std::size_t>(groups) * GroupThreads * Stride, 0xdeadbeefu);
    std::vector<std::uint32_t> userData(8, 0u);
    const auto in = BufferDescriptor(input.data(), input.size() * 4u);
    const auto out = BufferDescriptor(translation.output.data(), translation.output.size() * 4u);
    std::copy(in.begin(), in.end(), userData.begin());
    std::copy(out.begin(), out.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{GroupThreads, 1, 1}, 0u, {true, false, false}, false, 1};
    auto target = device.Target();
    target.mslNanTestElimination = eliminate;
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.context.floatMode = floatMode;
    const auto result = ShaderRecompiler::Recompile(request);
    translation.nanTests = NanTests(result.msl.Shader().source);
    translation.bytes = result.msl.Shader().source.size();
    device.Dispatch(result, groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    return translation;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (!device->Target().mslNanTestElimination) {
            std::puts("skipped: the device target keeps every NaN test");
            return VulkanTestSkipped;
        }
        const auto rows = Rows();
        const auto groups = static_cast<std::uint32_t>(rows.size() / GroupThreads);
        std::vector<std::uint32_t> input(rows.size() * Inputs);
        for (std::size_t row = 0; row < rows.size(); ++row) std::copy(rows[row].begin(), rows[row].end(), input.begin() + static_cast<std::ptrdiff_t>(row * Inputs));
        const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 4> modes{
            std::nullopt,
            ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false},
            ShaderRecompiler::ShaderFloatMode{0x00u, false, true, false},
            ShaderRecompiler::ShaderFloatMode{0xf0u, true, false, false},
        };
        std::size_t compared = 0;
        for (const auto& mode : modes) {
            const std::string name = mode.has_value() ? "FLOAT_MODE " + std::to_string(mode->floatMode) + (mode->ieeeMode ? " IEEE" : "") : "no float mode";
            const auto kept = Run(*device, mode, false, input, groups);
            const auto eliminated = Run(*device, mode, true, input, groups);
            std::size_t differences = 0;
            for (std::size_t row = 0; row < rows.size(); ++row) {
                for (std::uint32_t result = 0; result < Results; ++result) {
                    const auto index = row * Stride + result;
                    Require(kept.output[index] != 0xdeadbeefu || eliminated.output[index] == 0xdeadbeefu, "NaN test elimination: " + name + " result " + std::to_string(result) + " was not written");
                    if (kept.output[index] == eliminated.output[index]) continue;
                    if (differences++ < 16u) std::printf("%s row %zu (0x%08x 0x%08x) result %u: kept 0x%08x, eliminated 0x%08x\n", name.c_str(), row, rows[row][0], rows[row][1], result, kept.output[index], eliminated.output[index]);
                }
            }
            Require(differences == 0u, "NaN test elimination: " + name + ": " + std::to_string(differences) + " result(s) differ");
            Require(eliminated.nanTests < kept.nanTests, "NaN test elimination: " + name + ": the eliminating translation has " + std::to_string(eliminated.nanTests) + " NaN tests, the other " + std::to_string(kept.nanTests));
            compared += rows.size() * Results;
            std::printf("%s: NaN tests %zu -> %zu, MSL %zu -> %zu bytes, %zu results bit-identical\n", name.c_str(), kept.nanTests, eliminated.nanTests, kept.bytes, eliminated.bytes, rows.size() * Results);
        }
        std::printf("NaN test elimination: %zu results bit-identical\n", compared);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
