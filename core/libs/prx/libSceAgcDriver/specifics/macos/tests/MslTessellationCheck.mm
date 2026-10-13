// agc_metal_msl_tessellation (DECISIONS D24, MslAbi Version 7): tessellation on Metal mesh pipelines,
// compile only (no GPU work is submitted). tests/execution/Tessellation.cpp's local, hull, domain and
// pixel programs are recompiled for the Metal device's target; the three tessellation programs must
// reflect Stage::Local / Hull / Domain and define aps5_stage, and LinkMslTessellation must turn them
// into an object function and a mesh function that compile and link with the pixel program into a
// mesh render pipeline (MTLMeshRenderPipelineDescriptor with an object function), for every domain,
// partitioning and output topology the tessellator supports. The rendering is checked by
// agc_driver_tessellation, the tessellator itself by agc_metal_tessellator.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/tests/execution/VulkanTestDevice.hpp"
#include "MetalBackend/MslEmitter.hpp"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdio>
#include <iostream>
#include <optional>
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

// tests/execution/Tessellation.cpp's programs: a local program writing 3 dwords of LDS per vertex, a
// hull program (3 control points, factors and a control point copy by control point 0's guard), a
// domain program interpolating the control points by the domain location, and a pixel program.
alignas(256) constexpr std::array<std::uint32_t, 29> LocalCode{
    0xd5480004, 0x02090502, 0xd5480005, 0x02090102, 0xd5480006, 0x02110902, 0xd5480007, 0x02111102,
    0x7e080d04, 0x7e0a0d05, 0x7e0c0d06, 0x7e0e0d07, 0x10100cff, 0x3f19999a, 0x061010ff, 0xbf666666,
    0x3e1008ff, 0x3e800000, 0xd54b0009, 0x03c5e105, 0x161406a0, 0xd9340000, 0x0000080a, 0xd8340008,
    0x0000070a, 0xd8340010, 0x0000070a, 0xbf8cc07f, 0xbefd2106,
};

alignas(256) constexpr std::array<std::uint32_t, 49> HullCode{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000060, 0xdbfc0000,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000030, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0x7da82881, 0xbe94047e, 0xb06a0060, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013, 0xd8dc1008,
    0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x34142484, 0xbefe0414,
    0xe0781000, 0x0403180a, 0x4a162481, 0x16161690, 0x4c1616ff, 0x00008000, 0xe0741000, 0x0202070b,
    0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 30> DomainCode{
    0xf8000941, 0x00000000, 0x34020e84, 0x16020283, 0xe0341000, 0x04030a01, 0xe0341010, 0x04030c01,
    0xe0341020, 0x04030e01, 0xbf8c3f70, 0x08040af2, 0x08040d02, 0x10201505, 0x3e201906, 0x3e201d02,
    0x10221705, 0x3e221b06, 0x3e221f02, 0xd5510012, 0x040a0d05, 0x102424ff, 0x40400000, 0x7e260280,
    0x7e2802f2, 0xf80008cf, 0x14131110, 0xf800020f, 0x14131214, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 7> PixelCode{
    0xc8020002, 0xc8060102, 0xc80a0202, 0xc80e0302, 0xf800180f, 0x03020100, 0xbf810000,
};

constexpr ShaderRecompiler::TessellationConfiguration Patches{3u, 3u, 1u, 2u, 2u};
alignas(256) constexpr std::array<std::uint16_t, 9> Indices{0, 1, 2, 3, 4, 5, 6, 7, 8};

ShaderRecompiler::RecompileResult Compile(const ShaderRecompiler::SpirvTarget& target, ShaderStage stage, std::span<const std::uint32_t> code, std::uint32_t& pushCursor) {
    static const std::array<std::uint32_t, 16> hullUserData{0x1000u};
    static const std::array<std::uint32_t, 8> domainUserData{};
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const bool hull = stage == ShaderStage::TessellationControl;
    const auto userData = hull ? std::span<const std::uint32_t>(hullUserData) : stage == ShaderStage::TessellationEvaluation ? std::span<const std::uint32_t>(domainUserData) : std::span<const std::uint32_t>();
    std::optional<ShaderRecompiler::ShaderPixelStageInfo> pixel;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex;
    std::optional<ShaderRecompiler::GraphicsCompileContext> graphics;
    if (stage == ShaderStage::Fragment) {
        ShaderRecompiler::ShaderPixelStageInfo info{};
        info.interpolatorCount = 1;
        info.interpolatorSettings[0] = 0x400u;
        info.targetOutputMode[0] = 9;
        info.targetExportMapping.fill(0xe4u);
        pixel = info;
    } else {
        vertex = ShaderRecompiler::ShaderVertexStageInfo{};
        graphics = ShaderRecompiler::GraphicsCompileContext{hull ? 0u : 8u, {}, std::nullopt, Patches, {reinterpret_cast<std::uintptr_t>(Indices.data()), static_cast<std::uint32_t>(Indices.size()), 2, 1}};
    }
    ShaderRecompiler::RecompileRequest request{
        {stage, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, hull ? 0u : 8u, userData, std::nullopt, pixel, vertex, memory},
        target,
        stage == ShaderStage::Fragment ? PixelPushLayout(pushCursor, target) : ShaderRecompiler::BindingLayout{0, 0, pushCursor, 128u - pushCursor},
        graphics
    };
    request.useCache = false;
    auto result = ShaderRecompiler::Recompile(request);
    pushCursor += static_cast<std::uint32_t>(result.pushConstants.size());
    return result;
}

ShaderRecompiler::MslTessellationProgram Program(const ShaderRecompiler::RecompileResult& result) {
    ShaderRecompiler::MslTessellationProgram program;
    program.shader = &result.msl.Shader();
    for (const auto& constant : result.specialization) program.constants.emplace_back(constant.id, constant.value);
    return program;
}

id<MTLFunction> Function(id<MTLDevice> device, const std::string& name, const std::string& source, NSDictionary<NSString*, NSObject*>* macros, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& constantValues) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = static_cast<MTLLanguageVersion>((Abi::LanguageMajor << 16u) | Abi::LanguageMinor);
    options.preprocessorMacros = macros;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(source.c_str()) options:options error:&error];
    if (library == nil) {
        std::cerr << source << '\n';
        throw std::runtime_error(name + ": MSL does not compile: " + Describe(error));
    }
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    for (const auto& [id, value] : constantValues) {
        const std::uint32_t copy = value;
        [constants setConstantValue:&copy type:MTLDataTypeUInt atIndex:id];
    }
    id<MTLFunction> function = [library newFunctionWithName:@(Abi::EntryPoint) constantValues:constants error:&error];
    Require(function != nil, name + ": no entry point: " + Describe(error));
    return function;
}

NSString* Decimal(std::uint64_t value) {
    return [NSString stringWithFormat:@"%llu", static_cast<unsigned long long>(value)];
}

}

int main() {
    try {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        @autoreleasepool {
            id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
            if (metal == nil || ![metal supportsFamily:MTLGPUFamilyMetal4]) {
                std::puts("skipped, the device has no Metal 4 support");
                return Skipped;
            }
            const AgcDriver::VulkanDevice device{AgcDriver::VulkanDevice::CompileOnly{}};
            const auto target = device.Target();
            Require(target.backend == ShaderRecompiler::ShaderBackend::Msl, "the Metal device's target is not the MSL backend");
            Require(target.tessellation.has_value(), "the Metal target advertises no tessellation");
            std::uint32_t cursor = 0;
            const auto local = Compile(target, ShaderStage::Local, LocalCode, cursor);
            const auto hull = Compile(target, ShaderStage::TessellationControl, HullCode, cursor);
            const auto domain = Compile(target, ShaderStage::TessellationEvaluation, DomainCode, cursor);
            const auto fragment = Compile(target, ShaderStage::Fragment, PixelCode, cursor);
            std::puts("recompiled the local, hull, domain and pixel programs");
            const std::array<std::pair<const ShaderRecompiler::RecompileResult*, Abi::Stage>, 3> stages{{{&local, Abi::Stage::Local}, {&hull, Abi::Stage::Hull}, {&domain, Abi::Stage::Domain}}};
            for (const auto& [result, stage] : stages) {
                Require(!result->msl.empty() && result->msl.Reflection().stage == stage, "a tessellation program does not reflect its tessellation stage");
                Require(Has(result->msl.Source(), "static void aps5_stage(") && !Has(result->msl.Source(), "[[vertex]]") && !Has(result->msl.Source(), "vertex Aps5VertexOut"), "a tessellation library has no aps5_stage or has a stage function");
                // Each library compiles on its own (prepared tessellation stages are compiled alone).
                MTLCompileOptions* options = [MTLCompileOptions new];
                options.languageVersion = static_cast<MTLLanguageVersion>((Abi::LanguageMajor << 16u) | Abi::LanguageMinor);
                NSError* error = nil;
                const std::string source(result->msl.Source());
                if ([metal newLibraryWithSource:@(source.c_str()) options:options error:&error] == nil) {
                    std::cerr << source << '\n';
                    throw std::runtime_error("a tessellation library does not compile alone: " + Describe(error));
                }
            }
            Require(Has(hull.msl.Source(), "aps5_tess_payload[4u] =") || Has(hull.msl.Source(), "aps5_tess_payload[4u]"), "the hull program does not store the triangle's inner factor in payload word 4");
            Require(domain.msl.Reflection().outputLocations != 0u, "the domain program writes no parameter");
            // Upstream #4727: hull input reads clamp the control point to the patch's last one.
            Require(!Has(hull.msl.Source(), "aps5_tess_local[") || Has(hull.msl.Source(), "min(aps5_ta"), "the hull program reads control inputs without clamping the control point to the patch");

            struct Case {
                std::uint32_t domain;
                std::uint32_t partitioning;
                std::uint32_t topology;
            };
            // One case per partitioning, output topology and domain (each links two libraries).
            const std::vector<Case> cases{{1u, 2u, 2u}, {1u, 0u, 3u}, {1u, 3u, 0u}, {1u, 1u, 1u}, {2u, 3u, 2u}, {0u, 2u, 1u}};
            for (const auto& test : cases) {
                auto config = Patches;
                config.domain = test.domain;
                config.partitioning = test.partitioning;
                config.outputTopology = test.topology;
                const auto name = "domain " + std::to_string(test.domain) + " partitioning " + std::to_string(test.partitioning) + " topology " + std::to_string(test.topology);
                const auto linked = ShaderRecompiler::LinkMslTessellation(Program(local), Program(hull), Program(domain), config);
                Require(linked.object.reflection.stage == Abi::Stage::Object && linked.mesh.reflection.stage == Abi::Stage::Mesh, name + ": wrong linked stages");
                Require(Has(linked.object.source, "[[object]] void aps5_main(") && Has(linked.mesh.source, "[[mesh]] void aps5_main("), name + ": no object or mesh function");
                Require(!Has(linked.object.source, "[[function_constant(") && !Has(linked.mesh.source, "[[function_constant("), name + ": a linked library still declares function constants");
                Require(linked.objectThreads == 3u && linked.meshThreads == 256u && linked.payloadBytes <= Abi::TessellationPayloadBytes, name + ": unexpected launch shape");
                // MslAbi Version 10 (upstream #4082): inner factors at or below 1 become max(minLevel, 1),
                // before the tessellator processes the clamped payload once.
                Require(Has(linked.object.source, "fmax(aps5_draw.minLevel, 1.0f)") && !Has(linked.object.source, "aps5_patch.culled == 0"), name + ": the object stage does not clamp the factors as upstream's hull programs do");
                id<MTLFunction> objectFunction = Function(metal, name + " object", linked.object.source, @{}, {});
                id<MTLFunction> meshFunction = Function(metal, name + " mesh", linked.mesh.source, @{@(Abi::DepthNegativeOneToOneMacro): @"1"}, {});
                std::vector<std::pair<std::uint32_t, std::uint32_t>> fragmentConstants;
                for (const auto& constant : fragment.specialization) fragmentConstants.emplace_back(constant.id, constant.value);
                id<MTLFunction> fragmentFunction = Function(metal, name + " fragment", fragment.msl.Source(), @{
                    @(Abi::VertexOutputsMacro): Decimal(linked.mesh.reflection.outputLocations & 0xffffffffu),
                    @(Abi::VertexOutputsHighMacro): Decimal(linked.mesh.reflection.outputLocations >> 32u),
                }, fragmentConstants);
                MTLMeshRenderPipelineDescriptor* descriptor = [MTLMeshRenderPipelineDescriptor new];
                descriptor.objectFunction = objectFunction;
                descriptor.meshFunction = meshFunction;
                descriptor.fragmentFunction = fragmentFunction;
                descriptor.maxTotalThreadsPerObjectThreadgroup = linked.objectThreads;
                descriptor.maxTotalThreadsPerMeshThreadgroup = linked.meshThreads;
                descriptor.maxTotalThreadgroupsPerMeshGrid = linked.maxMeshThreadgroups;
                descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
                NSError* error = nil;
                id<MTLRenderPipelineState> pipeline = [metal newRenderPipelineStateWithMeshDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
                Require(pipeline != nil, name + ": the object, mesh and fragment functions do not link: " + Describe(error));
                Require(pipeline.maxTotalThreadsPerObjectThreadgroup >= linked.objectThreads && pipeline.maxTotalThreadsPerMeshThreadgroup >= linked.meshThreads, name + ": the pipeline allows fewer threads than the draw launches");
                std::printf("%s: object %u threads, mesh %u threads, payload %u bytes, linked\n", name.c_str(), linked.objectThreads, linked.meshThreads, linked.payloadBytes);
            }
        }
        std::puts("msl tessellation check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
