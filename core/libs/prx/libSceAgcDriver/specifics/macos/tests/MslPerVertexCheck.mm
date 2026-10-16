// agc_metal_msl_per_vertex (MslAbi Version 5): emulated per-vertex fragment values (and the alpha-to-
// coverage sample mask of float and uint exports), compile only (no
// GPU work is submitted). Pixel programs that need per-vertex values (exact I/J reads, v_interp_mov
// of P10/P20, 16-bit interpolation) are recompiled for the Metal device's target and must reflect
// Reflection::perVertexLocations / emulatedBarycentrics and read them through the triangle-vertices
// fragment inputs; plain v_interp_p1/p2 must stay host-interpolated. Each fragment function is then
// linked with the vertex function compiled in the triangle-vertices mode (APS5_TRIANGLE_VERTICES,
// APS5_PER_VERTEX_LOCATIONS, APS5_BARYCENTRICS from the fragment reflection, APS5_CONSERVATIVE off
// and on) into a render pipeline, which checks that the user() outputs and inputs match. The
// rendering itself is checked by agc_driver_pixel_barycentric_emulation and agc_driver_interpolation_f16.
// MslAbi Version 14 (DECISIONS D69): the vertex function also links for points and lines
// (APS5_PRIMITIVE_VERTICES 1 and 2).

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

bool Has(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

// tests/execution/PixelBarycentricEmulation.cpp: position from a buffer, exported as position and
// parameter 0.
alignas(256) constexpr std::array<std::uint32_t, 8> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf800020f, 0x03020100, 0xf80008cf, 0x03020100, 0xbf810000,
};
// Exports the I/J VGPRs (exact barycentrics).
alignas(256) constexpr std::array<std::uint32_t, 3> BarycentricPixelCode{
    0xf800180f, 0x01000100, 0xbf810000,
};
// v_interp_mov of P10, P20 and P0.
alignas(256) constexpr std::array<std::uint32_t, 6> VertexValuePixelCode{
    0xc80a0000, 0xc80e0001, 0xc8120002, 0xf800180f, 0x04040302, 0xbf810000,
};
// Plain v_interp_p1/p2 (host interpolation, no per-vertex values).
alignas(256) constexpr std::array<std::uint32_t, 5> InterpolatedPixelCode{
    0xc8080000, 0xc8090001, 0xf800180f, 0x02020202, 0xbf810000,
};
// DECISIONS D38, one input at several locations (llvm-mc -arch=amdgcn -mcpu=gfx1030): v_interp_p1/p2
// of input 0 over the sample, center and centroid pairs (v[0:1], v[2:3], v[4:5]), exported.
alignas(256) constexpr std::array<std::uint32_t, 9> ThreeLocationsPixelCode{
    0xc8180000u, 0xc8190001u, 0xc81c0002u, 0xc81d0003u, 0xc8200004u, 0xc8210005u, 0xf800180fu, 0x07080706u, 0xbf810000u,
};
// Exports two I/J pairs (exact barycentrics at the first two enabled locations): exp mrt0 v0, v1, v2, v3.
alignas(256) constexpr std::array<std::uint32_t, 3> TwoPairsPixelCode{
    0xf800180fu, 0x03020100u, 0xbf810000u,
};
// tests/execution/InterpolationF16.cpp: 16-bit interpolation (v_interp_p1ll/p1lv/p2_f16) of an
// FP16_INTERP_MODE input, results stored to a buffer.
alignas(256) constexpr std::array<std::uint32_t, 47> F16PixelCode{
    0x7e080f02u, 0x7e0a0f03u, 0xd5430006u, 0x0411ff05u, 0x000000c0u, 0x7e300300u, 0x7e320301u, 0xd742001au,
    0x00020000u, 0x7e3602ffu, 0xabcd0000u, 0xd75a001bu, 0x046a0200u, 0xd7420008u, 0x00020100u, 0x7e3802ffu,
    0x12340000u, 0xd75a001cu, 0x04220300u, 0x7e1e02ffu, 0x4500bc00u, 0xd743001du, 0x043e0140u, 0x7e3c0280u,
    0xd75a001eu, 0x04760340u, 0x7e3e0280u, 0xd75a801fu, 0x046a0200u, 0xd7420020u, 0x00020181u, 0xd7420021u,
    0x000201c1u, 0xd7420022u, 0x40020000u, 0xd7430023u, 0x043e0000u, 0xe0782000u, 0x80011806u, 0xe0782010u,
    0x80011c06u, 0xe0782020u, 0x80012006u, 0x7e0e02f2u, 0xf800180fu, 0x07070707u, 0xbf810000u,
};
constexpr std::uint32_t Fp16Input = 0x03080000u;
constexpr std::uint32_t Fp16DefaultHighInput = 0x01580000u;

alignas(256) std::array<std::uint32_t, 64> Backing{};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, 0x01016facu};
}

ShaderRecompiler::RecompileResult RecompileVertex(const ShaderRecompiler::SpirvTarget& target) {
    std::vector<std::uint32_t> userData(4, 0u);
    const auto buffer = BufferDescriptor(Backing.data(), 16u, 3u);
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {64u, 0, userData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, memory},
        target,
        {0, 0, 0, 64}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

struct PixelCase {
    const char* name;
    std::span<const std::uint32_t> code;
    std::uint32_t inputs;
    bool f16;
    std::uint32_t waveSize;
    std::uint64_t perVertexLocations;
    std::uint32_t barycentrics;
    bool alphaToCoverage = false;
    std::uint32_t outputMode = 9u;
    // MslAbi Version 9 (DECISIONS D28): the qualifier of host-interpolated input 0, by the I/J pair
    // the program's v_interp_p2 reads (empty: not checked).
    const char* qualifier = "";
    // DECISIONS D38: the input (host) or the weights (emulated) are read at several locations,
    // through interpolant<> (pull-model interpolation) instead of a qualifier.
    bool pullModel = false;
};

ShaderRecompiler::RecompileResult RecompilePixel(const ShaderRecompiler::SpirvTarget& target, const PixelCase& test, std::uint32_t vertexPush) {
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.wave32 = test.waveSize == 32u;
    std::vector<std::uint32_t> userData;
    if (test.f16) {
        pixel.interpolatorCount = 2;
        pixel.interpolatorSettings[0] = Fp16Input;
        pixel.interpolatorSettings[1] = Fp16DefaultHighInput;
        pixel.posX = true;
        pixel.posY = true;
        userData.assign(8, 0u);
        const auto results = BufferDescriptor(Backing.data(), 48u, 1u);
        std::copy(results.begin(), results.end(), userData.begin() + 4);
    } else {
        pixel.interpolatorCount = 1;
        pixel.executeOnNoop = true;
    }
    pixel.inputAddr = test.inputs;
    pixel.hasPerspectiveCenterVgpr = (test.inputs & PixelInputBit(PixelInput::PerspectiveCenter)) != 0u;
    pixel.noPerspective = (test.inputs & PixelInputBit(PixelInput::LinearCenter)) != 0u;
    pixel.perspectiveCentroid = (test.inputs & PixelInputBit(PixelInput::PerspectiveCentroid)) != 0u;
    pixel.linearCentroid = (test.inputs & PixelInputBit(PixelInput::LinearCentroid)) != 0u;
    pixel.targetOutputMode[0] = test.outputMode;
    pixel.alphaToCoverage = test.alphaToCoverage;
    pixel.targetExportMapping.fill(0xe4u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(test.code.data()), std::as_bytes(test.code)}}};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(test.code.data()), test.code, 0, {}},
        {test.waveSize, 0, userData, std::nullopt, pixel, std::nullopt, memory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    request.useCache = false;
    if (test.f16) request.context.floatMode = ShaderRecompiler::ShaderFloatMode{0xf0u, true, true, false};
    return ShaderRecompiler::Recompile(request);
}

id<MTLFunction> Compile(id<MTLDevice> device, const std::string& name, const ShaderRecompiler::RecompileResult& result, NSDictionary<NSString*, NSObject*>* macros) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = static_cast<MTLLanguageVersion>((Abi::LanguageMajor << 16u) | Abi::LanguageMinor);
    options.mathMode = MTLMathModeSafe;
    options.preprocessorMacros = macros;
    NSError* error = nil;
    const std::string source(result.msl.Source());
    id<MTLLibrary> library = [device newLibraryWithSource:@(source.c_str()) options:options error:&error];
    if (library == nil) {
        std::cerr << source << '\n';
        throw std::runtime_error(name + ": MSL does not compile: " + Describe(error));
    }
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    for (const auto& constant : result.specialization) {
        const std::uint32_t value = constant.value;
        [constants setConstantValue:&value type:MTLDataTypeUInt atIndex:constant.id];
    }
    id<MTLFunction> function = [library newFunctionWithName:@(Abi::EntryPoint) constantValues:constants error:&error];
    Require(function != nil, name + ": no specializable entry point: " + Describe(error));
    return function;
}

NSString* Decimal(std::uint64_t value) {
    return [NSString stringWithFormat:@"%llu", static_cast<unsigned long long>(value)];
}

void Link(id<MTLDevice> device, const std::string& name, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& pixel, std::uint32_t conservative, MTLPixelFormat format, std::uint32_t primitiveVertices = 3u) {
    const auto& fragment = pixel.msl.Reflection();
    const auto& vertexReflection = vertex.msl.Reflection();
    const auto what = name + (conservative != 0u ? " (conservative)" : "") + (primitiveVertices == 1u ? " (points)" : primitiveVertices == 2u ? " (lines)" : "");
    NSMutableDictionary<NSString*, NSObject*>* vertexMacros = [@{
        @(Abi::TriangleVerticesMacro): @"1",
        @(Abi::PerVertexLocationsMacro): Decimal(fragment.perVertexLocations & 0xffffffffu),
        @(Abi::PerVertexLocationsHighMacro): Decimal(fragment.perVertexLocations >> 32u),
        @(Abi::BarycentricsMacro): Decimal(fragment.emulatedBarycentrics),
        @(Abi::ConservativeMacro): Decimal(conservative),
    } mutableCopy];
    // MslAbi Version 14 (DECISIONS D69): points and lines take one or two guest vertices per primitive.
    if (primitiveVertices != 3u) vertexMacros[@(Abi::PrimitiveVerticesMacro)] = [NSString stringWithFormat:@"%uu", primitiveVertices];
    id<MTLFunction> vertexFunction = Compile(device, what + " vertex", vertex, vertexMacros);
    NSMutableDictionary<NSString*, NSObject*>* fragmentMacros = [@{
        @(Abi::VertexOutputsMacro): Decimal(vertexReflection.outputLocations & 0xffffffffu),
        @(Abi::VertexOutputsHighMacro): Decimal(vertexReflection.outputLocations >> 32u),
        @(Abi::ConservativeMacro): Decimal(conservative),
    } mutableCopy];
    // DECISIONS D78: points and lines output and read only their own corners.
    if (primitiveVertices != 3u) fragmentMacros[@(Abi::PrimitiveVerticesMacro)] = [NSString stringWithFormat:@"%uu", primitiveVertices];
    id<MTLFunction> fragmentFunction = Compile(device, what + " fragment", pixel, fragmentMacros);
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = vertexFunction;
    descriptor.fragmentFunction = fragmentFunction;
    descriptor.colorAttachments[0].pixelFormat = format;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    Require(pipeline != nil, what + ": the triangle-vertices vertex function does not link with the fragment function: " + Describe(error));
}

void Check(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target, const ShaderRecompiler::RecompileResult& vertex, const PixelCase& test) {
    const auto pixel = RecompilePixel(target, test, static_cast<std::uint32_t>(vertex.pushConstants.size()));
    const std::string name(test.name);
    Require(!pixel.msl.empty() && pixel.msl.Reflection().abiVersion == Abi::Version, name + ": no MslAbi v5 fragment function");
    const auto& reflection = pixel.msl.Reflection();
    const std::string source(pixel.msl.Source());
    Require(reflection.perVertexLocations == test.perVertexLocations, name + ": Reflection::perVertexLocations is " + std::to_string(reflection.perVertexLocations) + ", expected " + std::to_string(test.perVertexLocations));
    Require(reflection.emulatedBarycentrics == test.barycentrics, name + ": Reflection::emulatedBarycentrics is " + std::to_string(reflection.emulatedBarycentrics) + ", expected " + std::to_string(test.barycentrics));
    Require(!Has(source, "[[barycentric_coord"), name + ": the fragment function uses Metal's barycentric_coord instead of the emulated weights");
    Require(pixel.barycentricEmulation.active == (test.perVertexLocations != 0u || test.barycentrics != 0u), name + ": RecompileResult::barycentricEmulation does not match the per-vertex use");
    for (std::uint32_t location = 0; location < 64u; ++location) {
        if (((test.perVertexLocations >> location) & 1u) == 0u) continue;
        for (std::uint32_t corner = 0; corner < 3u; ++corner) {
            const auto field = "pv" + std::to_string(location) + "_" + std::to_string(corner);
            Require(Has(source, "float4 " + field + " [[user(aps5_" + field + "), flat]]"), name + ": no flat fragment input " + field);
        }
    }
    if (test.pullModel && test.barycentrics != 0u) {
        Require(Has(source, "interpolant<float3, interpolation::perspective> aps5_bw [[user(aps5_bw)]]") == ((test.barycentrics & 1u) != 0u), name + ": perspective weights are not an interpolant");
        Require(Has(source, "aps5_in.aps5_bw.interpolate_at_sample(aps5_sample_id)") && Has(source, "aps5_in.aps5_bw.interpolate_at_center()"), name + ": the weights are not read at the sample and the center");
    } else {
        Require(Has(source, "aps5_bw [[user(aps5_bw), center_perspective]]") == ((test.barycentrics & 1u) != 0u), name + ": perspective weights input does not match the reflection");
        Require(Has(source, "aps5_bwl [[user(aps5_bwl), center_no_perspective]]") == ((test.barycentrics & 2u) != 0u), name + ": linear weights input does not match the reflection");
    }
    if (test.pullModel && test.barycentrics == 0u) {
        Require(Has(source, "interpolant<float4, interpolation::perspective> p0 [[user(locn0)]]"), name + ": input 0 is not an interpolant");
        for (const char* read : {".interpolate_at_sample(aps5_sample_id)", ".interpolate_at_center()", ".interpolate_at_centroid()"}) Require(Has(source, std::string("aps5_in.p0") + read), name + ": input 0 has no " + read + " read");
    }
    if (test.pullModel) Require(reflection.perSample && Has(source, "[[sample_id]]"), name + ": a read at the sample does not run per sample");
    if (test.alphaToCoverage) {
        // Alpha to coverage starts from full coverage (SPIR-V: StoreCoverage(-1)); only a float MRT0
        // export lowers it, so a uint export must keep every sample.
        Require(Has(source, "aps5_out.aps5_sample_mask = 0xffffffffu;") && Has(source, "[[sample_mask]]"), name + ": alpha to coverage does not start from full coverage");
        // MslAbi Version 9: the measured proportional mask (DECISIONS D28), over the guest sample count.
        Require(Has(source, "aps5_out.aps5_sample_mask = aps5_alpha_to_mask(as_type<float>(") == (test.outputMode != 7u), name + ": the alpha-to-coverage mask does not match the export type");
        Require(Has(source, "#define APS5_SAMPLE_COUNT get_num_samples()") == (test.outputMode != 7u), name + ": the alpha-to-coverage mask has no guest sample count default");
        Require(Has(source, "#if APS5_SAMPLE_BASE"), name + ": the exported mask is not shifted to the pass's samples");
    }
    if (*test.qualifier != '\0') Require(Has(source, std::string(" [[user(locn0), ") + test.qualifier + "]]"), name + ": input 0 is not interpolated with " + test.qualifier);
    const auto format = test.outputMode == 7u ? MTLPixelFormatRGBA32Uint : MTLPixelFormatRGBA32Float;
    Link(device, name, vertex, pixel, 0u, format);
    Link(device, name, vertex, pixel, 1u, format);
    Link(device, name, vertex, pixel, 0u, format, 1u);
    Link(device, name, vertex, pixel, 0u, format, 2u);
    std::printf("%s: per-vertex locations 0x%llx, barycentrics %u, linked in the triangle-vertices mode\n", test.name, static_cast<unsigned long long>(reflection.perVertexLocations), reflection.emulatedBarycentrics);
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
            Require(target.backend == ShaderRecompiler::ShaderBackend::Msl && target.fragmentShaderBarycentricEnabled, "the Metal target does not advertise emulated per-vertex values");
            const auto vertex = RecompileVertex(target);
            const auto& vertexSource = vertex.msl.Source();
            Require(Has(std::string(vertexSource), "#elif defined(APS5_TRIANGLE_VERTICES) && APS5_TRIANGLE_VERTICES") && Has(std::string(vertexSource), "[[buffer(" + std::to_string(Abi::Index(Abi::BufferIndex::TriangleVertices)) + ")]]"), "the vertex function has no triangle-vertices mode");
            Require(Has(std::string(vertexSource), std::string("#ifndef ") + Abi::PrimitiveVerticesMacro) && Has(std::string(vertexSource), std::string("aps5_vertex_id / ") + Abi::PrimitiveVerticesMacro), "the triangle-vertices mode does not take the primitive's vertex count (points and lines)");
            const auto perspective = PixelInputBit(PixelInput::PerspectiveCenter);
            const auto linear = PixelInputBit(PixelInput::LinearCenter);
            const auto position = PixelInputBit(PixelInput::PositionX) | PixelInputBit(PixelInput::PositionY);
            const auto threeLocations = PixelInputBit(PixelInput::PerspectiveSample) | perspective | PixelInputBit(PixelInput::PerspectiveCentroid);
            const std::array<PixelCase, 15> cases{{
                {"perspective I/J", BarycentricPixelCode, perspective, false, 64u, 0u, 1u},
                {"linear I/J", BarycentricPixelCode, linear, false, 64u, 0u, 2u},
                {"v_interp_mov P10/P20/P0", VertexValuePixelCode, perspective, false, 64u, 1u, 0u},
                {"host v_interp_p1/p2", InterpolatedPixelCode, perspective, false, 64u, 0u, 0u},
                {"16-bit interpolation wave64", F16PixelCode, perspective | position, true, 64u, 1u, 1u},
                {"16-bit interpolation wave32", F16PixelCode, perspective | position, true, 32u, 1u, 1u},
                {"alpha to coverage, float export", InterpolatedPixelCode, perspective, false, 64u, 0u, 0u, true, 9u},
                {"alpha to coverage, uint export", InterpolatedPixelCode, perspective, false, 64u, 0u, 0u, true, 7u},
                // The same v_interp_p1/p2 over the I/J pair of each location (the only enabled pair
                // is v0/v1): the input's MSL interpolation qualifier follows it.
                {"host v_interp at the center", InterpolatedPixelCode, perspective, false, 64u, 0u, 0u, false, 9u, "center_perspective"},
                {"host v_interp at the sample (PERSP_SAMPLE)", InterpolatedPixelCode, PixelInputBit(PixelInput::PerspectiveSample), false, 64u, 0u, 0u, false, 9u, "sample_perspective"},
                {"host v_interp at the sample (LINEAR_SAMPLE)", InterpolatedPixelCode, PixelInputBit(PixelInput::LinearSample), false, 64u, 0u, 0u, false, 9u, "sample_no_perspective"},
                {"host v_interp at the centroid (PERSP_CENTROID)", InterpolatedPixelCode, PixelInputBit(PixelInput::PerspectiveCentroid), false, 64u, 0u, 0u, false, 9u, "centroid_perspective"},
                {"host v_interp at the centroid (LINEAR_CENTROID)", InterpolatedPixelCode, PixelInputBit(PixelInput::LinearCentroid), false, 64u, 0u, 0u, false, 9u, "centroid_no_perspective"},
                // DECISIONS D38: input 0 at the sample, center and centroid (interpolant<>), and the
                // exact sample and center I/J (emulated weights as an interpolant<>).
                {"host v_interp at sample, center and centroid", ThreeLocationsPixelCode, threeLocations, false, 64u, 0u, 0u, false, 9u, "", true},
                {"exact I/J at the sample and the center", TwoPairsPixelCode, PixelInputBit(PixelInput::PerspectiveSample) | perspective, false, 64u, 0u, 1u, false, 9u, "", true},
            }};
            for (const auto& test : cases) Check(metal, target, vertex, test);
        }
        std::puts("msl per-vertex check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
