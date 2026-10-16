// agc_metal_msl_mesh: MSL mesh functions (NGG primitive shaders, B6 stage 5), compile only (no GPU
// work is submitted). tests/Mesh.cpp's geometry program (wave64 one- and two-wave subgroups, wave32,
// strips, fans and rect lists) is recompiled for the Metal device's target; each mesh function must reflect
// MslAbi::Stage::Mesh with the host threadgroup size the driver dispatches (wave64 at two lanes per
// invocation: 32 host threads per guest wave), and links with the Mesh.cpp pixel program into a
// mesh render pipeline (MTLMeshRenderPipelineDescriptor), which checks that the mesh vertex
// user() outputs match the fragment inputs. Each non-rect mesh function is also compiled in its
// triangle-vertices mode (MslAbi Version 13) and linked with a per-vertex/barycentric fragment
// function. The rendering itself is checked by agc_driver_mesh and agc_driver_mesh_per_vertex_inputs.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/tests/execution/VulkanTestDevice.hpp"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

// tests/Mesh.cpp: the NGG geometry program (index fetch, LDS vertex exchange, GS_ALLOC_REQ,
// primitive, position and parameter exports) and its pixel program.
alignas(256) constexpr std::array<std::uint32_t, 104> GeometryCode{
    0x8f6a9003, 0x94fe6ac1, 0xbf88000b, 0xd7650006, 0x000100c1, 0xd7660006, 0x00020cc1, 0x93ebff03,
    0x00040018, 0xd7460006, 0x04190c6b, 0x340c0c82, 0xd8340000, 0x00000506, 0xbf8cc07f, 0xbefe04c1,
    0xbf8a0000, 0x938dff02, 0x00090016, 0x9382ff03, 0x00040018, 0x938cff03, 0x00080008, 0xd7650009,
    0x000100c1, 0xd7660009, 0x000212c1, 0xd746000a, 0x04250c02, 0x7da8120c, 0xbf88002e, 0x361600ff,
    0x0000ffff, 0x2c180090, 0x361a02ff, 0x0000ffff, 0xd8d80000, 0x0b00000b, 0xd8d80000, 0x0c00000c,
    0xd8d80000, 0x0d00000d, 0xbf8cc07f, 0xe0382000, 0x8002100b, 0xe0382010, 0x8002140b, 0xe0382000,
    0x8002180c, 0xe0382010, 0x80021c0c, 0xe0382000, 0x8002200d, 0xe0382010, 0x8002240d, 0x161c1483,
    0x161e14ff, 0x00000060, 0xbf8c3f70, 0xdb7c0400, 0x0000100f, 0xdb7c0410, 0x0000140f, 0xdb7c0420,
    0x0000180f, 0xdb7c0430, 0x00001c0f, 0xdb7c0440, 0x0000200f, 0xdb7c0450, 0x0000240f, 0x4a501c81,
    0x4a521c82, 0x3450508a, 0x34525294, 0xd772002a, 0x04a6510e, 0xbf8cc07f, 0xbefe04c1, 0xbf8a0000,
    0x930e830d, 0xbf078002, 0xbf850003, 0x8f0f8c0d, 0x887c0f0e, 0xbf900009, 0x7da8140d, 0xbf880002,
    0xf8000941, 0x0000002a, 0xbefe04c1, 0x7da8140e, 0xbf88000a, 0x34561485, 0xdbfc0400, 0x2c00002b,
    0xdbfc0410, 0x3000002b, 0xbf8cc07f, 0xf80008cf, 0x2f2e2d2c, 0xf800020f, 0x33323130, 0xbf810000,
};

alignas(256) constexpr auto Wave32GeometryCode = [] {
    auto code = GeometryCode;
    code[10] = 0x04190a6bu;
    code[28] = 0x04250a02u;
    return code;
}();

alignas(256) constexpr std::array<std::uint32_t, 7> PixelCode{
    0xc8020002, 0xc8060102, 0xc80a0202, 0xc80e0302, 0xf800180f, 0x03020100, 0xbf810000,
};

// Per-vertex inputs (MslAbi Version 13, the mesh triangle-vertices mode; DECISIONS
// D46): v_interp_mov P10, P20 and P0 of attr0.x and the raw
// PERSP_CENTER I exported to MRT0, so the fragment function reads per-vertex location 0 and the
// emulated perspective barycentrics, as SuperTuxKart's radeonsi programs do.
alignas(256) constexpr std::array<std::uint32_t, 6> PerVertexPixelCode{
    0xc80a0000, 0xc80e0001, 0xc8120002, 0xf800180f, 0x00040302, 0xbf810000,
};

alignas(256) std::array<std::uint16_t, 64> Indices{};
alignas(256) std::array<std::uint32_t, 64> Backing{};

struct MeshCase {
    const char* name;
    ShaderRecompiler::MeshConfiguration mesh;
    std::uint32_t waveSize;
    bool indexed;
};

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

void Check(id<MTLDevice> device, const ShaderRecompiler::SpirvTarget& target, const MeshCase& test) {
    const std::string name(test.name);
    const AgcDriver::Pm4::DrawParameters draw{test.indexed ? reinterpret_cast<std::uintptr_t>(Indices.data()) : 0u, 12u, test.indexed ? 2u : 0u, 1u, 0u, test.indexed};
    std::vector<std::uint32_t> userData(12, 0u);
    const auto index = AgcDriver::Graphics::MeshIndexBufferDescriptor(draw);
    std::copy(index.begin(), index.end(), userData.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    const auto address = reinterpret_cast<std::uintptr_t>(Backing.data());
    const std::array<std::uint32_t, 4> vertexBuffer{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (32u << 16u), 8u, 0x01016facu};
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), userData.begin() + 8);
    const auto& code = test.waveSize == 32u ? Wave32GeometryCode : GeometryCode;
    const std::array<ShaderRecompiler::MemoryRegion, 1> geometryMemory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(std::span(code))}}};
    ShaderRecompiler::RecompileRequest geometry{
        {ShaderStage::Mesh, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {test.waveSize, 0, userData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, geometryMemory},
        target,
        {0, 0, 0, ShaderRecompiler::MeshDrawPushOffsetBytes},
        ShaderRecompiler::GraphicsCompileContext{0, {}, test.mesh, std::nullopt, {draw.indexAddress, draw.indexCount, draw.indexSize, draw.instanceCount}}
    };
    geometry.useCache = false;
    const auto mesh = ShaderRecompiler::Recompile(geometry);
    Require(!mesh.msl.empty(), name + ": no MSL mesh function");
    const auto& reflection = mesh.msl.Reflection();
    const std::string source(mesh.msl.Source());
    Require(reflection.stage == Abi::Stage::Mesh, name + ": the reflection's stage is not Mesh");
    const auto groupThreads = (test.mesh.threadsPerGroup + test.waveSize - 1u) / test.waveSize * test.waveSize;
    const auto hostThreads = test.waveSize == 64u ? groupThreads / 2u : groupThreads;
    Require(reflection.threadsPerThreadgroup[0] == hostThreads && reflection.threadsPerThreadgroup[1] == 1u && reflection.threadsPerThreadgroup[2] == 1u,
            name + ": the mesh threadgroup is " + std::to_string(reflection.threadsPerThreadgroup[0]) + " threads, expected " + std::to_string(hostThreads));
    // Rect lists (DECISIONS D41): the group's vertices plus one per rect, two triangles per rect.
    const bool rectList = test.mesh.inputPrimitive == 17u;
    const auto rects = std::min(test.mesh.maxPrimitives, test.mesh.primitivesPerGroup);
    const auto outputVertices = rectList ? std::min(test.mesh.maxVertices, test.mesh.verticesPerGroup) + rects : test.mesh.maxVertices;
    const auto outputPrimitives = rectList ? rects * 2u : test.mesh.maxPrimitives;
    Require(Has(source, "[[mesh]] void " + std::string(Abi::EntryPoint)) && Has(source, ", " + std::to_string(outputVertices) + ", " + std::to_string(outputPrimitives) + (rectList ? ", metal::topology::triangle>" : ", APS5_MESH_TOPOLOGY>")), name + ": no mesh<> output of the subgroup's limits");
    // MslAbi Version 16 (DECISIONS D75): triangles unless the driver draws points or lines (rect lists
    // are always triangles).
    if (!rectList) Require(Has(source, "#define APS5_MESH_TOPOLOGY metal::topology::triangle") && Has(source, "#define APS5_MESH_TOPOLOGY metal::topology::line") && Has(source, "#define APS5_MESH_TOPOLOGY metal::topology::point"), name + ": the mesh output topology does not follow APS5_PRIMITIVE_VERTICES");
    Require(rectList == Has(source, "threadgroup Aps5MeshVertex aps5_rv[") && rectList == Has(source, "set_primitive_count(aps5_rects * 2u)"), name + ": rect-list expansion " + (rectList ? "missing" : "in a mesh function that is not a rect list"));
    Require(Has(source, "set_primitive_count(") && Has(source, "[[primitive_culled]]"), name + ": the mesh function does not set its primitive count and cull flags");
    Require(reflection.usesPushData, name + ": the mesh function does not read the draw words from the push data");

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = 1;
    pixel.interpolatorSettings[0] = 0x400u;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const auto meshPush = static_cast<std::uint32_t>(mesh.pushConstants.size());
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
    ShaderRecompiler::RecompileRequest fragmentRequest{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
        {64, 0, {}, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(meshPush, target),
        std::nullopt
    };
    fragmentRequest.useCache = false;
    const auto fragment = ShaderRecompiler::Recompile(fragmentRequest);

    id<MTLFunction> meshFunction = Compile(device, name + " mesh", mesh, @{});
    id<MTLFunction> fragmentFunction = Compile(device, name + " fragment", fragment, @{
        @(Abi::VertexOutputsMacro): Decimal(reflection.outputLocations & 0xffffffffu),
        @(Abi::VertexOutputsHighMacro): Decimal(reflection.outputLocations >> 32u),
    });
    MTLMeshRenderPipelineDescriptor* descriptor = [MTLMeshRenderPipelineDescriptor new];
    descriptor.meshFunction = meshFunction;
    descriptor.fragmentFunction = fragmentFunction;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithMeshDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
    Require(pipeline != nil, name + ": the mesh function does not link with the fragment function: " + Describe(error));
    Require(pipeline.maxTotalThreadsPerMeshThreadgroup >= hostThreads, name + ": the mesh pipeline allows " + std::to_string(pipeline.maxTotalThreadsPerMeshThreadgroup) + " threads per threadgroup, the program needs " + std::to_string(hostThreads));
    std::printf("%s: mesh<%u, %u> with %u host threads, linked with the fragment function\n", test.name, outputVertices, outputPrimitives, hostThreads);
    if (rectList) {
        Require(!Has(source, "aps5_tv_v[") && !Has(source, "aps5_lds_words[aps5_i * "), name + ": a rect-list mesh function has the triangle-vertices mode");
        return;
    }

    // The triangle-vertices mode: unshared vertices, MeshTriangleVerticesPrimitives triangles per
    // split threadgroup, compiled with the driver's macros and linked with a fragment function that
    // reads per-vertex values and emulated barycentrics (flat user(aps5_pv0_<j>) from the primitive
    // record, user(aps5_bw) from the vertices).
    const auto split = Abi::MeshTriangleVerticesPrimitives(test.mesh.maxPrimitives);
    Require(split * 3u <= Abi::MeshMaxVertices && split * Abi::MeshTriangleVerticesSplits(test.mesh.maxPrimitives) >= test.mesh.maxPrimitives, name + ": the triangle-vertices split does not cover the subgroup");
    Require(Has(source, ", " + std::to_string(3u * split) + ", " + std::to_string(split) + ", APS5_MESH_TOPOLOGY>"), name + ": no triangle-vertices mesh<> of " + std::to_string(3u * split) + " vertices and " + std::to_string(split) + " primitives");
    // The staging is in the guest's LDS words when the program has LDS (DECISIONS D77).
    Require((Has(source, "aps5_tv_v[") || Has(source, "aps5_lds_words[aps5_i * ")) && Has(source, "float4 pv0_0 [[user(aps5_pv0_0)]]") && Has(source, "float3 aps5_bw [[user(aps5_bw)]]"), name + ": the triangle-vertices epilogue or its outputs are missing");
    ShaderRecompiler::ShaderPixelStageInfo perVertexPixel{};
    perVertexPixel.interpolatorCount = 1;
    perVertexPixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PerspectiveCenter);
    perVertexPixel.hasPerspectiveCenterVgpr = true;
    perVertexPixel.targetOutputMode[0] = 9;
    perVertexPixel.targetExportMapping.fill(0xe4u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> perVertexMemory{{{reinterpret_cast<std::uintptr_t>(PerVertexPixelCode.data()), std::as_bytes(std::span(PerVertexPixelCode))}}};
    ShaderRecompiler::RecompileRequest perVertexRequest{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PerVertexPixelCode.data()), PerVertexPixelCode, 0, {}},
        {64, 0, {}, std::nullopt, perVertexPixel, std::nullopt, perVertexMemory},
        target,
        PixelPushLayout(meshPush, target),
        std::nullopt
    };
    perVertexRequest.useCache = false;
    const auto perVertexFragment = ShaderRecompiler::Recompile(perVertexRequest);
    const auto& fragmentReflection = perVertexFragment.msl.Reflection();
    Require(fragmentReflection.perVertexLocations == 1u && (fragmentReflection.emulatedBarycentrics & 1u) != 0u, name + ": the per-vertex pixel program reads per-vertex locations 0x" + std::to_string(fragmentReflection.perVertexLocations) + " and emulated barycentrics " + std::to_string(fragmentReflection.emulatedBarycentrics) + ", expected location 0 and perspective weights");
    for (const bool last : {false, true}) {
        NSMutableDictionary<NSString*, NSObject*>* macros = [@{
            @(Abi::TriangleVerticesMacro): @"1",
            @(Abi::PerVertexLocationsMacro): [Decimal(fragmentReflection.perVertexLocations & 0xffffffffu) stringByAppendingString:@"u"],
            @(Abi::PerVertexLocationsHighMacro): [Decimal(fragmentReflection.perVertexLocations >> 32u) stringByAppendingString:@"u"],
            @(Abi::BarycentricsMacro): [Decimal(fragmentReflection.emulatedBarycentrics) stringByAppendingString:@"u"],
        } mutableCopy];
        if (last) macros[@(Abi::ProvokingLastMacro)] = @"1";
        id<MTLFunction> splitMesh = Compile(device, name + " triangle-vertices mesh", mesh, macros);
        id<MTLFunction> perVertexFunction = Compile(device, name + " per-vertex fragment", perVertexFragment, @{
            @(Abi::VertexOutputsMacro): Decimal(reflection.outputLocations & 0xffffffffu),
            @(Abi::VertexOutputsHighMacro): Decimal(reflection.outputLocations >> 32u),
        });
        MTLMeshRenderPipelineDescriptor* splitDescriptor = [MTLMeshRenderPipelineDescriptor new];
        splitDescriptor.meshFunction = splitMesh;
        splitDescriptor.fragmentFunction = perVertexFunction;
        splitDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
        id<MTLRenderPipelineState> splitPipeline = [device newRenderPipelineStateWithMeshDescriptor:splitDescriptor options:MTLPipelineOptionNone reflection:nil error:&error];
        Require(splitPipeline != nil, name + ": the triangle-vertices mesh function does not link with the per-vertex fragment function: " + Describe(error));
    }
    std::printf("%s: triangle-vertices mesh<%u, %u> x %u splits linked with per-vertex values and barycentrics (first and last provoking)\n", test.name, 3u * split, split, Abi::MeshTriangleVerticesSplits(test.mesh.maxPrimitives));
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
            Require(target.mesh.has_value(), "the Metal target advertises no mesh shader limits");
            // tests/Mesh.cpp's subgroups: {input primitive, primitives and vertices per group, max
            // vertices, max primitives, threads, LDS bytes, provoking vertex, ESGS item size}.
            const std::array<MeshCase, 11> cases{{
                {"triangle list, one-wave subgroups, wave64", {4u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u}, 64u, true},
                {"triangle list, two-wave subgroups, wave64", {4u, 32u, 96u, 96u, 32u, 128u, 2048u, 0u, 4u}, 64u, true},
                {"triangle list, wave32", {4u, 4u, 12u, 12u, 4u, 32u, 1024u, 0u, 4u}, 32u, false},
                {"triangle list, not indexed", {4u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u}, 64u, false},
                {"triangle strip", {6u, 3u, 5u, 9u, 3u, 64u, 1024u, 0u, 4u}, 64u, true},
                {"triangle fan", {5u, 3u, 5u, 9u, 3u, 64u, 1024u, 0u, 4u}, 64u, true},
                // 128 primitives: the triangle-vertices mode splits the subgroup in two.
                {"triangle strip, 128 primitives", {6u, 128u, 130u, 130u, 128u, 192u, 4096u, 0u, 4u}, 64u, false},
                {"rect list, one-wave subgroups, wave64", {17u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u}, 64u, false},
                {"rect list, two-wave subgroups, wave64", {17u, 32u, 96u, 96u, 32u, 128u, 2048u, 0u, 4u}, 64u, true},
                {"rect list, wave32", {17u, 4u, 12u, 12u, 4u, 32u, 1024u, 0u, 4u}, 32u, false},
                            // DECISIONS D77: 20 KiB of LDS beside the triangle-vertices staging of 192 vertices
                // (12.3 KiB): together past the 32 KiB threadgroup limit, so the staging reuses the
                // LDS once the guest program returned (ASTRO BOT's worldmap meshes).
                {"triangle list, LDS beside the triangle-vertices staging", {4u, 64u, 192u, 192u, 64u, 192u, 20480u, 0u, 4u}, 64u, true},
            }};
            for (const auto& test : cases) Check(metal, target, test);
        }
        std::puts("msl mesh check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
