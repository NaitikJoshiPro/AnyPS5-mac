// agc_metal_msl_upstream_parity (MslAbi Version 10): the SPIR-V backend behaviour changes of the
// upstream sync to 0f2683cb7 as the MSL backend mirrors them, compile only (no GPU work is submitted).
// The rendering itself is checked by the upstream execution tests on the Metal device
// (agc_driver_vertex_clip_cull_distances, agc_driver_f16_to_f32_unpack, ...).
// - #4030: v_interp_mov P10 of a flat input reads the flat (provoking vertex) value: no per-vertex
//   emulation, FragmentParameter::flatP10 recorded;
// - #423: guest cull distances are emulated in one more clip distance, per triangle (triangle-
//   vertices draws) or per point (APS5_CULL_POINTS); the library refuses any other draw;
// - #4470: vertex programs of a target without vertexSubgroups are one-lane waves;
// - #3988: f16 operands convert in hardware when the target keeps f16 denormals.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/tests/execution/VulkanTestDevice.hpp"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdio>
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

ShaderRecompiler::RecompileResult RecompilePixel(const ShaderRecompiler::SpirvTarget& target, std::span<const std::uint32_t> code, std::uint32_t interpolator, std::uint32_t inputs) {
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = interpolator != 0u ? 1u : 0u;
    pixel.interpolatorSettings[0] = interpolator;
    pixel.inputAddr = inputs;
    pixel.hasPerspectiveCenterVgpr = (inputs & PixelInputBit(PixelInput::PerspectiveCenter)) != 0u;
    pixel.noPerspective = (inputs & PixelInputBit(PixelInput::LinearCenter)) != 0u;
    pixel.executeOnNoop = true;
    pixel.targetOutputMode[0] = 9u;
    pixel.targetExportMapping.fill(0xe4u);
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
        Require(reflection.abiVersion == 10u && reflection.cullDistances != 0u, name + ": Reflection::cullDistances is not reported");
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
        }
        std::puts("msl upstream parity check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
