// agc_metal_msl_upstream_parity (MslAbi Versions 10 and 11): the SPIR-V backend behaviour changes of
// the upstream syncs to 0f2683cb7 and fdefd7660 as the MSL backend mirrors them, compile and pipeline
// link only (no GPU work is submitted).
// The rendering itself is checked by the upstream execution tests on the Metal device
// (agc_driver_vertex_clip_cull_distances, agc_driver_f16_to_f32_unpack, ...).
// - #4030: v_interp_mov P10 of a flat input reads the flat (provoking vertex) value: no per-vertex
//   emulation, FragmentParameter::flatP10 recorded;
// - #423: guest cull distances are emulated in one more clip distance, per triangle (triangle-
//   vertices draws) or per point (APS5_CULL_POINTS); the library refuses any other draw;
// - #4470: vertex programs of a target without vertexSubgroups are one-lane waves;
// - #3988: f16 operands convert in hardware when the target keeps f16 denormals.
// MslAbi Version 11 (upstream sync to fdefd7660):
// - #4011: an f16 operand that is only compared converts without the NaN payload rebuild;
// - #4210: earlyZ pixel functions test early unless they use alpha-to-coverage (Metal: or export
//   depth or stencil, which [[early_fragment_tests]] forbids);
// - #5097: the ancillary word's render target index is [[render_target_array_index]] times the
//   LayerExported function constant, and a pipeline whose vertex function writes no layer links;
// - #4279: GDS plain reads and writes are relaxed atomics (the coherent gds buffer).
// Upstream sync to 721db55dc:
// - #5492 (731575c80): ds_add_f32 flushes operands and result per FLOAT_MODE and stores NaNs as the
//   hardware does (aps5_lds_fadd32);
// - #5176 (46a1cc0fc, 3a7e82a17): ds_ordered_count adds the first active lane's operand to the GDS
//   counter at offset0 (no M0 base).
// Upstream sync UPSYNC4 (MslAbi Version 19):
// - b0558ce95: ds_ordered_count with offset1 bit 4 exchanges the first active lane's operand into
//   the counter.
// Upstream sync UPSYNC5 (MslAbi Version 21):
// - 57ceb9e370: a request with a sample order maps the coverage the pixel function reads and writes
//   between the guest's and the host's sample order of its quad pixel;
// - e5b4bd72dd: a GLC buffer_store_dwordx2 is one 64-bit store under the lock table (a plain
//   dwordx2 store stays two dword stores).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/tests/execution/VulkanTestDevice.hpp"
#include "MslAbi.hpp"
#include "PipelineSpecialization.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdio>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ShaderRecompiler::PixelInput;
using ShaderRecompiler::PixelInputBit;
using ShaderRecompiler::ShaderStage;
namespace Abi = ShaderRecompiler::MslAbi;

constexpr int Skipped = 77;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Describe(NSError* error) {
    return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

bool Has(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
}

// tests/execution/VertexClipCullDistances.cpp: position and the distance vector from a buffer,
// exported as pos0 and pos1.
alignas(256) constexpr std::array<std::uint32_t, 10> DistanceVertexCode{
    0xe0382000, 0x80000005, 0xe0382010, 0x80000405, 0xbf8c3f70,
    0xf80000cf, 0x03020100, 0xf80008df, 0x07060504, 0xbf810000,
};
constexpr std::uint32_t DistanceVector = 0x00400000u;

// tests/Graphics.cpp: v_interp_mov P10 of input 0 into v4-v7 (flatP10Color), and the P0 form.
alignas(256) constexpr std::array<std::uint32_t, 7> FlatP10PixelCode{0xc8120000u, 0xc8160100u, 0xc81a0200u, 0xc81e0300u, 0xf800180fu, 0x07060504u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 7> FlatP0PixelCode{0xc8120002u, 0xc8160102u, 0xc81a0202u, 0xc81e0302u, 0xf800180fu, 0x07060504u, 0xbf810000u};

// v_cvt_f32_f16 v0, v2; v_mov_b32 v1-v3 = 1.0; export mrt0 v0-v3.
alignas(256) constexpr std::array<std::uint32_t, 7> F16PixelCode{0x7e001702u, 0x7e0202f2u, 0x7e0402f2u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u};

// Wave32 vertex: v_mbcnt_lo (the lane) exported as parameter 0 next to the position (a cross-lane
// value, which a one-lane wave reads as 0).
alignas(256) constexpr std::array<std::uint32_t, 10> LaneVertexCode{
    0xe0382000, 0x80000005, 0xd7650004, 0x000100c1, 0xbf8c3f70,
    0xf800020f, 0x04040404, 0xf80008cf, 0x03020100, 0xbf810000,
};

// tests/execution/LayeredVolumeDraw.cpp LayerFieldPixelCode: the render target index field of the
// ancillary word (v_bfe of v0 bits 16-26) as a colour.
alignas(256) constexpr std::array<std::uint32_t, 6> LayerFieldPixelCode{0x2c020090u, 0x7e020d01u, 0x100202f0u, 0xf800180fu, 0x01010101u, 0xbf810000u};

// tests/execution/F16ToF32Unpack.cpp: v_cvt_f32_f16 of a loaded half (ConvertCode), and v_cmp_u_f16 /
// v_cmp_lt_f16 of one (CompareCode).
alignas(256) constexpr std::array<std::uint32_t, 10> F16ConvertCode{
    0x34020082, 0x8f098708, 0x4a020209, 0xe0301000, 0x80000401, 0xbf8c3f70, 0x7e0a1704, 0xe0701000,
    0x80010501, 0xbf810000,
};
alignas(256) constexpr std::array<std::uint32_t, 22> F16CompareCode{
    0x34020082, 0x34040083, 0x8f098708, 0x8f0a8808, 0x4a020209, 0x4a04040a, 0xe0301000, 0x80000401,
    0x7e0c02ff, 0x00003c00, 0xbf8c3f70, 0x7dd00904, 0xd5010005, 0x01a90280, 0x7d920d04, 0xd5010007,
    0x01a90280, 0xe0701000, 0x80010502, 0xe0701004, 0x80010702, 0xbf810000,
};

// tests/ShaderMemory.cpp verifyCoherentGds: ds_read_b32 and ds_write_b32 on GDS.
alignas(256) constexpr std::array<std::uint32_t, 9> GdsCode{0x34000082u, 0xbefc03ffu, 0x0000ffffu, 0xd8da0000u, 0x01000000u, 0xbf8cc07fu, 0xd8360080u, 0x00000100u, 0xbf810000u};

// ds_add_f32 v0, v1 gds (v1 = 1.0) after M0 = 0xffff.
alignas(256) constexpr std::array<std::uint32_t, 8> GdsAddF32Code{0x34000082u, 0xbefc03ffu, 0x0000ffffu, 0x7e0202f2u, 0xd8560000u, 0x00000100u, 0xbf8cc07fu, 0xbf810000u};
// tests/execution/GdsAppendConsume.cpp OrderedCode: ds_ordered_count v5, v1 offset0:16 offset1:3 gds.
alignas(256) constexpr std::array<std::uint32_t, 10> OrderedCountCode{0x34060082u, 0x4a020083u, 0xbefc03ffu, 0x0000ffffu, 0xd8fe0310u, 0x05000001u, 0xbf8cc07fu, 0xe0702000u, 0x80000503u, 0xbf810000u};
// The same with offset1 bit 4 set: ds_ordered_count swap (upstream b0558ce95, offset1:0x13).
alignas(256) constexpr std::array<std::uint32_t, 10> OrderedSwapCode{0x34060082u, 0x4a020083u, 0xbefc03ffu, 0x0000ffffu, 0xd8fe1310u, 0x05000001u, 0xbf8cc07fu, 0xe0702000u, 0x80000503u, 0xbf810000u};

// v_mov_b32 v5, v0; v_mov_b32 v6, v0; buffer_store_dwordx2 v[5:6], off, s[0:3], 0 offset:8 (glc in
// the coherent form).
alignas(256) constexpr std::array<std::uint32_t, 5> CoherentPairCode{0x7e0a0300u, 0x7e0c0300u, 0xe0744008u, 0x80000500u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 5> PlainPairCode{0x7e0a0300u, 0x7e0c0300u, 0xe0740008u, 0x80000500u, 0xbf810000u};

alignas(256) std::array<std::uint32_t, 64> Backing{};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, 0x01016facu};
}

ShaderRecompiler::RecompileResult RecompileVertex(const ShaderRecompiler::SpirvTarget& target, std::span<const std::uint32_t> code, std::uint32_t waveSize, std::uint32_t paClVsOutCntl) {
    std::vector<std::uint32_t> userData(4, 0u);
    const auto buffer = BufferDescriptor(Backing.data(), 32u, 3u);
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    ShaderRecompiler::ShaderVertexStageInfo vertex{};
    vertex.paClVsOutCntl = paClVsOutCntl;
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {waveSize, 0, userData, std::nullopt, std::nullopt, vertex, memory},
        target,
        {0, 0, 0, 64}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

ShaderRecompiler::RecompileResult RecompilePixel(const ShaderRecompiler::SpirvTarget& target, std::span<const std::uint32_t> code, std::uint32_t interpolator, std::uint32_t inputs, const std::function<void(ShaderRecompiler::ShaderPixelStageInfo&)>& configure = {}) {
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = interpolator != 0u ? 1u : 0u;
    pixel.interpolatorSettings[0] = interpolator;
    pixel.inputAddr = inputs;
    pixel.hasPerspectiveCenterVgpr = (inputs & PixelInputBit(PixelInput::PerspectiveCenter)) != 0u;
    pixel.noPerspective = (inputs & PixelInputBit(PixelInput::LinearCenter)) != 0u;
    pixel.executeOnNoop = true;
    pixel.targetOutputMode[0] = 9u;
    pixel.targetExportMapping.fill(0xe4u);
    if (configure) configure(pixel);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const std::vector<std::uint32_t> userData(4, 0u);
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64u, 0, userData, std::nullopt, pixel, std::nullopt, memory},
        target,
        PixelPushLayout(0u, target)
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

ShaderRecompiler::RecompileResult RecompileCompute(const ShaderRecompiler::SpirvTarget& target, std::span<const std::uint32_t> code) {
    std::vector<std::uint32_t> userData(8, 0u);
    const auto address = reinterpret_cast<std::uintptr_t>(Backing.data());
    const std::array<std::uint32_t, 4> buffer{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(sizeof(Backing)), 0x31016facu};
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(buffer.begin(), buffer.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{32u, 1u, 1u}, 0u, {true, false, false}, false, 1u};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32u, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.context.floatMode = ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false};
    return ShaderRecompiler::Recompile(request);
}

std::uint32_t Constant(const ShaderRecompiler::RecompileResult& result, std::uint32_t id) {
    for (const auto& constant : result.specialization) {
        if (constant.id == id) return constant.value;
    }
    throw std::runtime_error("function constant " + std::to_string(id) + " is not supplied");
}

// The library's entry point, or nil with the compiler's message in `failure`.
id<MTLFunction> Compile(id<MTLDevice> device, const ShaderRecompiler::RecompileResult& result, NSDictionary<NSString*, NSObject*>* macros, std::string& failure) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = static_cast<MTLLanguageVersion>((Abi::LanguageMajor << 16u) | Abi::LanguageMinor);
    options.mathMode = MTLMathModeSafe;
    options.preprocessorMacros = macros;
    NSError* error = nil;
    const std::string source(result.msl.Source());
    id<MTLLibrary> library = [device newLibraryWithSource:@(source.c_str()) options:options error:&error];
    if (library == nil) {
        failure = Describe(error);
        return nil;
    }
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    for (const auto& constant : result.specialization) {
        const std::uint32_t value = constant.value;
        [constants setConstantValue:&value type:MTLDataTypeUInt atIndex:constant.id];
    }
    id<MTLFunction> function = [library newFunctionWithName:@(Abi::EntryPoint) constantValues:constants error:&error];
    if (function == nil) failure = Describe(error);
    return function;
}

id<MTLFunction> MustCompile(id<MTLDevice> device, const std::string& name, const ShaderRecompiler::RecompileResult& result, NSDictionary<NSString*, NSObject*>* macros) {
    std::string failure;
    id<MTLFunction> function = Compile(device, result, macros, failure);
    if (function == nil) {
        std::cerr << result.msl.Source() << '\n';
        throw std::runtime_error(name + ": MSL does not compile: " + failure);
    }
    return function;
}

void FlatP10(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    const auto inputs = PixelInputBit(PixelInput::PerspectiveCenter);
    const auto p10 = RecompilePixel(target, FlatP10PixelCode, 0x400u, inputs);
    const auto p0 = RecompilePixel(target, FlatP0PixelCode, 0x400u, inputs);
    for (const auto* result : {&p10, &p0}) {
        Require(result->msl.Reflection().perVertexLocations == 0u && result->msl.Reflection().emulatedBarycentrics == 0u && !result->barycentricEmulation.active, "flat v_interp_mov: a flat input read through P10 or P0 needs no per-vertex emulation");
        Require(Has(result->msl.Source(), "[[user(locn0), flat]]"), "flat v_interp_mov: input 0 is not a flat fragment input");
        Require(result->fragmentParameters.size() == 1u && result->fragmentParameters[0].flat && !result->fragmentParameters[0].perVertex, "flat v_interp_mov: unexpected fragment parameter metadata");
        static_cast<void>(MustCompile(device, "flat v_interp_mov", *result, @{}));
    }
    Require(p10.fragmentParameters[0].flatP10 && !p0.fragmentParameters[0].flatP10, "flat v_interp_mov: FragmentParameter::flatP10 is not recorded for P10 alone");
    std::puts("flat P10: provoking-vertex value, flatP10 recorded");
}

void CullDistances(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    // Clip plane 0 from component 0, cull plane 0 from component 2 (VertexClipCullDistances rows).
    const auto clipped = RecompileVertex(target, DistanceVertexCode, 64u, DistanceVector | 0x1u);
    Require(clipped.msl.Reflection().cullDistances == 0u && Has(clipped.msl.Source(), "float aps5_clip [[clip_distance]] [2]") && !Has(clipped.msl.Source(), "aps5_cull"), "clip distances: a clip-only program has the wrong clip array or emulates culling");
    static_cast<void>(MustCompile(device, "clip distance", clipped, @{}));
    for (const std::uint32_t control : {DistanceVector | 0x400u, DistanceVector | 0x100u, DistanceVector | 0x101u | 0x200u}) {
        const auto culled = RecompileVertex(target, DistanceVertexCode, 64u, control);
        const auto& reflection = culled.msl.Reflection();
        const std::string source(culled.msl.Source());
        const auto name = "cull distances (PA_CL_VS_OUT_CNTL " + std::to_string(control) + ")";
        Require(reflection.abiVersion == Abi::Version && reflection.cullDistances != 0u, name + ": Reflection::cullDistances is not reported");
        Require(Has(source, "aps5_cull[0] = as_type<float>(") && Has(source, "thread float* aps5_cull"), name + ": the cull distance is not recorded per guest vertex");
        Require(Has(source, "aps5_cull[0][i] < 0.0f && aps5_cull[1][i] < 0.0f && aps5_cull[2][i] < 0.0f"), name + ": no per-triangle cull verdict");
        Require(Has(source, "aps5_culled ? -1.0f : 0.0f;"), name + ": the cull verdict is not a clip distance");
        static_cast<void>(MustCompile(device, name + ", triangle vertices", culled, @{@(Abi::TriangleVerticesMacro): @"1"}));
        static_cast<void>(MustCompile(device, name + ", points", culled, @{@(Abi::CullPointsMacro): @"1"}));
        static_cast<void>(MustCompile(device, name + ", rect list", culled, @{@(Abi::RectListMacro): @"1"}));
        std::string failure;
        Require(Compile(device, culled, @{}, failure) == nil && Has(failure, "cull distances need"), name + ": the library compiles for a draw that cannot emulate cull distances");
    }
    std::puts("cull distances: one emulated clip distance per triangle or point, other draws refused");
}

void SingleLane(id<MTLDevice> device, ShaderRecompiler::SpirvTarget target) {
    target.vertexSubgroups = false;
    for (const std::uint32_t waveSize : {32u, 64u}) {
        const auto result = RecompileVertex(target, LaneVertexCode, waveSize, 0u);
        const std::string source(result.msl.Source());
        const auto name = "one-lane vertex wave" + std::to_string(waveSize);
        Require(result.hostSubgroupSize == 1u, name + ": the program is not laid out for one lane");
        Require(Has(source, "#define simd_ballot(p)") && Has(source, "#undef simd_prefix_exclusive_sum"), name + ": the SIMD-group functions are not mapped to one lane");
        static_cast<void>(MustCompile(device, name, result, @{}));
        static_cast<void>(MustCompile(device, name + ", triangle vertices", result, @{@(Abi::TriangleVerticesMacro): @"1"}));
    }
    std::puts("one-lane vertex waves: compiled");
}

void NativeF16(id<MTLDevice> device, ShaderRecompiler::SpirvTarget target) {
    const auto inputs = PixelInputBit(PixelInput::PerspectiveCenter) | PixelInputBit(PixelInput::LinearCenter);
    for (const bool preserve : {false, true}) {
        target.float16DenormPreserve = preserve;
        const auto result = RecompilePixel(target, F16PixelCode, 0u, inputs);
        const auto name = std::string("v_cvt_f32_f16, float16DenormPreserve ") + (preserve ? "on" : "off");
        Require(Has(result.msl.Source(), "aps5_f16_to_f32_native(as_type<ushort>(") == preserve, name + ": the conversion does not follow the target flag");
        static_cast<void>(MustCompile(device, name, result, @{}));
    }
    std::puts("f16 operands: hardware conversion with float16DenormPreserve, decode without");
}

void ComparedF16(id<MTLDevice> device, ShaderRecompiler::SpirvTarget target) {
    target.float16DenormPreserve = true;
    const auto converted = RecompileCompute(target, F16ConvertCode);
    const auto compared = RecompileCompute(target, F16CompareCode);
    Require(Has(converted.msl.Source(), "aps5_f16_to_f32_native(as_type<ushort>(") && !Has(converted.msl.Source(), "aps5_f16_to_f32_any_nan(as_type<ushort>("), "v_cvt_f32_f16: the converted half must keep its NaN payload");
    Require(Has(compared.msl.Source(), "aps5_f16_to_f32_any_nan(as_type<ushort>(") && !Has(compared.msl.Source(), "aps5_f16_to_f32_native(as_type<ushort>("), "v_cmp_u_f16 / v_cmp_lt_f16: a half that is only compared must skip the NaN payload rebuild");
    static_cast<void>(MustCompile(device, "v_cvt_f32_f16", converted, @{}));
    static_cast<void>(MustCompile(device, "v_cmp_u_f16 / v_cmp_lt_f16", compared, @{}));
    std::puts("f16 compares: no NaN payload rebuild for compared operands");
}

void EarlyTests(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    const auto inputs = PixelInputBit(PixelInput::PerspectiveCenter) | PixelInputBit(PixelInput::LinearCenter);
    struct Row {
        const char* name;
        std::function<void(ShaderRecompiler::ShaderPixelStageInfo&)> configure;
        bool early;
    };
    const std::array<Row, 5> rows{{
        {"earlyZ", [](auto& pixel) { pixel.earlyZ = true; }, true},
        {"earlyZ with kill (DEPTH_BEFORE_SHADER)", [](auto& pixel) { pixel.earlyZ = true; pixel.pixelKillEnable = true; }, true},
        {"earlyZ with alpha-to-coverage", [](auto& pixel) { pixel.earlyZ = true; pixel.alphaToCoverage = true; }, false},
        {"earlyZ with a depth export", [](auto& pixel) { pixel.earlyZ = true; pixel.depthExportEnable = true; }, false},
        {"late Z", [](auto&) {}, false},
    }};
    for (const auto& row : rows) {
        const auto result = RecompilePixel(target, F16PixelCode, 0u, inputs, row.configure);
        Require(Has(result.msl.Source(), "[[early_fragment_tests]]") == row.early && result.msl.Reflection().earlyFragmentTests == row.early, std::string(row.name) + ": early fragment tests are " + (row.early ? "missing" : "declared"));
        if (row.early) static_cast<void>(MustCompile(device, row.name, result, @{}));
    }
    std::puts("early fragment tests: earlyZ without alpha-to-coverage, depth or stencil export");
}

// Upstream 16ea32c97 / c434467f6: a reversed-order blend code routes copies of the MRT0 export into
// the second blend source (color(0) index(1), under APS5_REVERSED_BLEND) and, for non-Own colour
// routes, into MRT0 itself.
void ReversedBlend(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    namespace Code = ShaderRecompiler::ReversedBlendCode;
    const auto inputs = PixelInputBit(PixelInput::PerspectiveCenter) | PixelInputBit(PixelInput::LinearCenter);
    // Second source: every component reads component 0; colour: component 3 reads 1.
    const std::uint32_t code = Code::Enable | (0x55u << Code::SecondShift) | (Code::One << (Code::ColorShift + 3u * Code::RouteBits));
    const auto result = RecompilePixel(target, F16PixelCode, 0u, inputs, [&](auto& pixel) { pixel.reversedBlend = code; });
    const auto& source = result.msl.Source();
    Require(Has(source, "aps5_rb [[color(0), index(1)]]") && Has(source, "aps5_reversed_route("), "reversed blend: no second blend source from the MRT0 export");
    Require(Constant(result, ShaderRecompiler::PipelineSpecialization::ReversedBlend) == code, "reversed blend: ReversedBlend is not specialised from the request");
    static_cast<void>(MustCompile(device, "reversed blend", result, @{@(Abi::ReversedBlendMacro): @"1"}));
    static_cast<void>(MustCompile(device, "reversed blend, macro unset", result, @{}));
    std::string failure;
    Require(Compile(device, result, @{@(Abi::ReversedBlendMacro): @"1", @(Abi::DualSourceMacro): @"1"}, failure) == nil && Has(failure, "dual-source"), "reversed blend: the library compiles with dual-source blending");
    std::puts("reversed blend: MRT0 routes into color(0) index(1) under APS5_REVERSED_BLEND");
}

void LayerField(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    for (const bool exported : {false, true}) {
        const auto result = RecompilePixel(target, LayerFieldPixelCode, 0u, PixelInputBit(PixelInput::Ancillary), [&](auto& pixel) {
            pixel.ancillary = true;
            pixel.layerExported = exported;
        });
        const auto name = std::string("ancillary render target index, layer ") + (exported ? "exported" : "not exported");
        Require(Has(result.msl.Source(), "[[render_target_array_index]]") && Has(result.msl.Source(), "aps5_fc" + std::to_string(ShaderRecompiler::PipelineSpecialization::LayerExported)), name + ": the field does not read the layer times LayerExported");
        Require(Constant(result, ShaderRecompiler::PipelineSpecialization::LayerExported) == (exported ? 1u : 0u), name + ": LayerExported is not specialised from the request");
        id<MTLFunction> fragment = MustCompile(device, name, result, @{});
        // A vertex function that writes no render target index: the pipeline must still link (the
        // field is then multiplied by 0).
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nstruct V { float4 position [[position]]; };\nvertex V aps5_vs(uint id [[vertex_id]]) { V v; v.position = float4(float(id & 1u), float(id >> 1u), 0.0f, 1.0f); return v; }\n" options:nil error:&error];
        Require(library != nil, name + ": the plain vertex function does not compile: " + Describe(error));
        MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = [library newFunctionWithName:@"aps5_vs"];
        descriptor.fragmentFunction = fragment;
        descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
        Require(pipeline != nil, name + ": a pipeline whose vertex function writes no layer does not link: " + Describe(error));
    }
    std::puts("ancillary render target index: [[render_target_array_index]] times LayerExported");
}

void CoherentGds(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    const auto result = RecompileCompute(target, GdsCode);
    const std::string source(result.msl.Source());
    Require(Has(source, "atomic_load_explicit((reinterpret_cast<device atomic_uint*>(aps5_gds0)") && Has(source, "atomic_store_explicit((reinterpret_cast<device atomic_uint*>(aps5_gds0)"), "GDS: ds_read_b32 / ds_write_b32 are not coherent (relaxed atomic) accesses");
    static_cast<void>(MustCompile(device, "GDS", result, @{}));
    std::puts("GDS: coherent reads and writes");
}

void GdsAddF32(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    const auto result = RecompileCompute(target, GdsAddF32Code);
    const std::string source(result.msl.Source());
    // FLOAT_MODE 0xc0 keeps no f32 denormals: operands and result flush.
    Require(Has(source, "aps5_lds_fadd32(") && Has(source, ", true, true)"), "ds_add_f32: the add does not flush operands and result as FLOAT_MODE says");
    static_cast<void>(MustCompile(device, "ds_add_f32", result, @{}));
    std::puts("ds_add_f32: hardware flush and NaN rules");
}

void OrderedCount(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    const auto result = RecompileCompute(target, OrderedCountCode);
    const std::string source(result.msl.Source());
    Require(Has(source, "aps5_amt") && Has(source, "atomic_fetch_add_explicit(") && Has(source, " = 4u;"), "ds_ordered_count: the first lane's operand is not added at offset0 / 4");
    static_cast<void>(MustCompile(device, "ds_ordered_count", result, @{}));
    std::puts("ds_ordered_count: first active lane's operand added to the GDS counter");
}

void OrderedSwap(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    const auto result = RecompileCompute(target, OrderedSwapCode);
    const std::string source(result.msl.Source());
    Require(Has(source, "aps5_amt") && Has(source, "atomic_exchange_explicit(") && !Has(source, "atomic_fetch_add_explicit(") && Has(source, " = 4u;"), "ds_ordered_count swap: the first lane's operand is not exchanged into the counter at offset0 / 4");
    static_cast<void>(MustCompile(device, "ds_ordered_count swap", result, @{}));
    std::puts("ds_ordered_count swap: first active lane's operand exchanged into the GDS counter");
}

// Upstream 57ceb9e370: Village's 2x checkerboard swaps its two samples on odd rows; with pixel
// (0,0)'s locations standing in, the function reads and writes coverage in each pixel's guest order.
void SampleOrder(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    const auto inputs = PixelInputBit(PixelInput::PerspectiveCenter) | PixelInputBit(PixelInput::LinearCenter);
    constexpr std::array<std::uint32_t, 4> rowSwap{0x76543210u, 0x76543210u, 0x76543201u, 0x76543201u};
    const auto ordered = RecompilePixel(target, F16PixelCode, 0u, inputs, [&](auto& pixel) {
        pixel.alphaToCoverage = true;
        pixel.sampleOrder = rowSwap;
        pixel.quadSampleMask = {1u, 1u, 1u, 1u};
    });
    const std::string source(ordered.msl.Source());
    const std::string toHost = "aps5_out.aps5_sample_mask = aps5_samples_to_host(";
    Require(Has(source, toHost) && Has(source, std::to_string(0x76543201u) + "u"), "sample order: the written coverage is not mapped to the host's order");
    Require(Has(source, "if (((aps5_samples_to_guest(") && source.find(toHost) > source.find("aps5_out.aps5_sample_mask &= aps5_quad_coverage"), "sample order: the quad coverage is not taken in the guest's order before the coverage goes to the host's");
    static_cast<void>(MustCompile(device, "sample order", ordered, @{}));
    static_cast<void>(MustCompile(device, "sample order, two samples", ordered, @{@"APS5_SAMPLE_COUNT": @"2u", @"APS5_SAMPLE_MASK": @"3u"}));
    const auto kept = RecompilePixel(target, F16PixelCode, 0u, inputs, [&](auto& pixel) {
        pixel.alphaToCoverage = true;
        pixel.sampleOrder = {0x76543210u, 0u, 0x76543210u, 0u};
    });
    Require(!Has(kept.msl.Source(), "aps5_samples_to_host") && !Has(kept.msl.Source(), "aps5_guest_sample"), "sample order: a kept order maps samples");
    std::puts("sample order: coverage in the guest's order of each quad pixel");
}

void CoherentPairStore(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target) {
    const auto coherent = RecompileCompute(target, CoherentPairCode);
    const std::string source(coherent.msl.Source());
    Require(Has(source, "aps5_atomic64_lock(") && Has(source, "atomic_max_explicit(reinterpret_cast<device atomic_ulong*>") && Has(source, "aps5_driver.atomicLocks != 0ul"), "coherent dwordx2 store: the pair is not one 64-bit store under the lock table");
    static_cast<void>(MustCompile(device, "coherent dwordx2 store", coherent, @{}));
    const auto plain = RecompileCompute(target, PlainPairCode);
    Require(!Has(plain.msl.Source(), "aps5_atomic64_lock("), "plain dwordx2 store: takes the 64-bit lock");
    static_cast<void>(MustCompile(device, "plain dwordx2 store", plain, @{}));
    std::puts("coherent dwordx2 store: one 64-bit store under the lock table");
}

}

int main() {
    try {
        @autoreleasepool {
            id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
            if (metal == nil || ![metal supportsFamily:MTLGPUFamilyMetal4]) {
                std::puts("skipped, the device has no Metal 4 support");
                return Skipped;
            }
            const AgcDriver::VulkanDevice device{AgcDriver::VulkanDevice::CompileOnly{}};
            const auto target = device.Target();
            Require(target.backend == ShaderRecompiler::ShaderBackend::Msl, "the Metal device's target is not the MSL backend");
            FlatP10(metal, target);
            CullDistances(metal, target);
            SingleLane(metal, target);
            NativeF16(metal, target);
            ComparedF16(metal, target);
            EarlyTests(metal, target);
            LayerField(metal, target);
            ReversedBlend(metal, target);
            CoherentGds(metal, target);
            GdsAddF32(metal, target);
            OrderedCount(metal, target);
            OrderedSwap(metal, target);
            SampleOrder(metal, target);
            CoherentPairStore(metal, target);
        }
        std::puts("msl upstream parity check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
