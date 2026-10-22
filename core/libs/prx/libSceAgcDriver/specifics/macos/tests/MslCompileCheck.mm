// Compile-only check of MetalBackend output (no GPU work is submitted): every program below is
// recompiled for the Metal device's target, its MSL is compiled with newLibraryWithSource and a
// compute pipeline is built with the result's function constants. Also checks that the checker
// itself rejects invalid MSL, that the reflection matches the MslAbi table layout and the lane
// mapping (wave64 programs run two guest lanes per host invocation, also when the request claims a
// 64-lane host), that s_memtime/s_memrealtime read the emulated clock, and that every descriptor
// table member type compiles with the layout MslAbi gives it.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Recompiler.hpp"
#include "MetalBackend/MslEmitterState.hpp"
#include <array>
#include <cstdio>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int Skipped = 77;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Describe(NSError* error) {
    return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

// tests/execution/Vop1Control.cpp: buffer_load_dword, VOP1 control ops, v_add_nc_u32, buffer_store_dword.
alignas(256) constexpr std::array<std::uint32_t, 17> Vop1ControlCode{
    0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0x7e000000, 0x7e003600, 0x7e008200, 0x4a080881,
    0xd5800000, 0x00000000, 0xd59b0000, 0x00000000, 0xd5c10000, 0x00000000, 0xe0702000, 0x80010401, 0xbf810000,
};

// tests/execution/BranchPastEndpgm.cpp: scalar branches into blocks after s_endpgm, and a loop whose
// body sits after the test that jumps back to it (selector in s8).
alignas(256) constexpr std::array<std::uint32_t, 20> TailBlockCode{
    0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0xbf068008, 0xbf840008, 0x4a080881, 0xe0702000, 0x80010401, 0xbf810000,
    0x4a080882, 0xe0702000, 0x80010401, 0xbf810000, 0xbf068108, 0xbf85fffa, 0x4a080883, 0xe0702000, 0x80010401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 18> LoopBodyCode{
    0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0xbe890380, 0xbf068008, 0xbf840006, 0x4a0808c0, 0xe0702000,
    0x80010401, 0xbf810000, 0x4a080881, 0x80098109, 0xbf0a0809, 0xbf85fffc, 0xe0702000, 0x80010401, 0xbf810000,
};

// tests/execution/ShaderClock.cpp: s_memtime and s_memrealtime around a loop (MSL reads the emulated
// clock, MslFlowEmitter.cpp EmitClock).
alignas(256) constexpr std::array<std::uint32_t, 49> ClockCode{
    0x34020084, 0x34060086, 0xe0301000, 0x80000401, 0xf4900200, 0x00000000, 0xf4940280, 0x00000000,
    0xbf8c0070, 0x7e0a0280, 0xbe9003ff, 0x000007d0, 0xd5430005, 0x04110705, 0x80908110, 0xbf078010,
    0xbf85fffb, 0xf4900300, 0x00000000, 0xf4940380, 0x00000000, 0xbf8cc07f, 0x7e140208, 0x7e160209,
    0x7e18020a, 0x7e1a020b, 0x7e1c020c, 0x7e1e020d, 0x7e20020e, 0x7e22020f, 0xe0701000, 0x80010a03,
    0xe0701004, 0x80010b03, 0xe0701008, 0x80010c03, 0xe070100c, 0x80010d03, 0xe0701010, 0x80010e03,
    0xe0701014, 0x80010f03, 0xe0701018, 0x80011003, 0xe070101c, 0x80011103, 0xe0701020, 0x80010503,
    0xbf810000,
};

// tests/execution/ShaderMemory.cpp computeLane63: v_readlane_b32 of lane 63 in a wave64 program.
alignas(256) constexpr std::array<std::uint32_t, 6> Lane63Code{0xd7600006u, 0x00017f00u, 0x7e020206u, 0xe0700000u, 0x80000100u, 0xbf810000u};

struct Program {
    const char* name;
    std::span<const std::uint32_t> code;
    std::uint32_t threads;
    std::uint32_t buffers;
    std::uint32_t waveSize;
    std::uint32_t userDataCount;
    // Host subgroup width the request claims instead of the device's (0: the device's). MSL programs
    // are laid out for 32 lanes whatever the claim (Recompiler.cpp HostSubgroupSize).
    std::uint32_t claimedSubgroupSize = 0u;
    // Text the MSL must contain (or `alternative`).
    const char* expect = nullptr;
    const char* alternative = nullptr;
};

constexpr std::array<Program, 10> Programs{{
    {"vop1_control", Vop1ControlCode, 64u, 2u, 32u, 8u},
    {"vop1_control_wave64", Vop1ControlCode, 64u, 2u, 64u, 8u},
    {"vop1_control_wave64_partial", Vop1ControlCode, 96u, 2u, 64u, 8u},
    {"tail_blocks", TailBlockCode, 64u, 2u, 32u, 9u},
    {"tail_blocks_wave64", TailBlockCode, 64u, 2u, 64u, 9u},
    {"loop_body", LoopBodyCode, 64u, 2u, 32u, 9u},
    {"loop_body_wave64", LoopBodyCode, 128u, 2u, 64u, 9u},
    {"shader_clock", ClockCode, 32u, 2u, 32u, 8u, 0u, "aps5_clock_reads"},
    {"shader_clock_wave64", ClockCode, 64u, 2u, 64u, 8u, 0u, "aps5_clock_reads"},
    // A lane read: a SIMD-group shuffle at two lanes per invocation, a threadgroup exchange at one
    // lane per invocation (DECISIONS D62).
    {"readlane63_wave64_claimed_64_lanes", Lane63Code, 64u, 1u, 64u, 4u, 64u, "simd_shuffle", "aps5_wx["},
}};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), count, 0x01016facu};
}

ShaderRecompiler::RecompileResult Recompile(const AgcDriver::VulkanDevice& device, const Program& program, std::span<std::uint32_t> backing) {
    std::vector<std::uint32_t> userData(program.userDataCount, 0u);
    for (std::uint32_t buffer = 0; buffer < program.buffers; ++buffer) {
        const auto descriptor = BufferDescriptor(backing.data(), static_cast<std::uint32_t>(backing.size()));
        std::copy(descriptor.begin(), descriptor.end(), userData.begin() + buffer * 4u);
    }
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(program.code.data()), std::as_bytes(program.code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{program.threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(program.code.data()), program.code, 0, {}},
        {program.waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    if (program.claimedSubgroupSize != 0u) request.target.subgroupSize = program.claimedSubgroupSize;
    return ShaderRecompiler::Recompile(request);
}

id<MTLLibrary> CompileLibrary(id<MTLDevice> device, const std::string& name, const std::string& source) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = static_cast<MTLLanguageVersion>((ShaderRecompiler::MslAbi::LanguageMajor << 16u) | ShaderRecompiler::MslAbi::LanguageMinor);
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
    if (library == nil) {
        std::cerr << source << '\n';
        throw std::runtime_error(name + ": MSL does not compile: " + Describe(error));
    }
    return library;
}

void CompileCheck(id<MTLDevice> device, const Program& program, const ShaderRecompiler::RecompileResult& result) {
    Require(!result.msl.empty(), std::string(program.name) + ": the recompiler produced no MSL for a Metal target");
    Require(result.spirv.empty(), std::string(program.name) + ": the recompiler also produced SPIR-V for a Metal target");
    const auto& reflection = result.msl.Reflection();
    Require(reflection.abiVersion == ShaderRecompiler::MslAbi::Version, std::string(program.name) + ": unexpected MslAbi version");
    Require(reflection.stage == ShaderRecompiler::MslAbi::Stage::Kernel, std::string(program.name) + ": a compute program must reflect a kernel");
    // A wave64 program runs two guest lanes per host invocation: 32 host invocations per guest wave,
    // in a one-dimensional threadgroup; or, without cross-lane dependences (DECISIONS D62, wave64
    // lane-local), one lane per invocation like wave32.
    const std::array<std::uint32_t, 3> perLane{program.threads, 1u, 1u};
    const std::array<std::uint32_t, 3> threads = program.waveSize == 64u ? std::array<std::uint32_t, 3>{(program.threads + 63u) / 64u * 32u, 1u, 1u} : perLane;
    const bool laneLocal = program.waveSize == 64u && reflection.threadsPerThreadgroup == perLane;
    Require(reflection.threadsPerThreadgroup == threads || laneLocal, std::string(program.name) + ": threadgroup size does not match the workgroup and its lane mapping");
    const auto* buffers = reflection.table.Find(ShaderRecompiler::RuntimeAbi::Binding::Buffers);
    Require(buffers != nullptr && buffers->count == program.buffers && buffers->offset == 0u, std::string(program.name) + ": the descriptor table lacks the guest buffers");
    Require(reflection.table.bytes == program.buffers * sizeof(ShaderRecompiler::MslAbi::BufferEntry), std::string(program.name) + ": unexpected descriptor table size");
    if (program.expect != nullptr) Require(result.msl.Source().find(program.expect) != std::string::npos || (program.alternative != nullptr && result.msl.Source().find(program.alternative) != std::string::npos), std::string(program.name) + ": the MSL lacks " + program.expect);
    id<MTLLibrary> library = CompileLibrary(device, program.name, result.msl.Source());
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    for (const auto& constant : result.specialization) {
        const std::uint32_t value = constant.value;
        [constants setConstantValue:&value type:MTLDataTypeUInt atIndex:constant.id];
    }
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@(ShaderRecompiler::MslAbi::EntryPoint) constantValues:constants error:&error];
    Require(function != nil, std::string(program.name) + ": the library has no specializable entry point: " + Describe(error));
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    Require(pipeline != nil, std::string(program.name) + ": compute pipeline creation failed: " + Describe(error));
    Require(pipeline.maxTotalThreadsPerThreadgroup >= threads[0] * threads[1] * threads[2], std::string(program.name) + ": the pipeline cannot run the workgroup");
}

// Every RuntimeAbi binding kind in one table (two entries each): the member types MetalBackend
// declares must compile, and the static assertions MslTableDeclaration adds check that Metal lays
// the struct out as MslAbi::LayoutTable does. Each texture is touched so its type is instantiated.
void TableTypesCheck(id<MTLDevice> device) {
    using ShaderRecompiler::RuntimeAbi::Binding;
    std::vector<ShaderRecompiler::MslAbi::TableBinding> bindings;
    for (std::uint32_t kind = 0; kind < static_cast<std::uint32_t>(Binding::Count); ++kind) bindings.push_back({static_cast<Binding>(kind), 2u});
    // MslAbi Version 15: the image table's ImageTableSlots typed array entries.
    bindings.push_back({Binding::ImageTable, ShaderRecompiler::MslAbi::ImageTableSlots});
    const auto layout = ShaderRecompiler::MslAbi::LayoutTable(bindings);
    std::string source = "#include <metal_stdlib>\nusing namespace metal;\n\n" + std::string(ShaderRecompiler::MslCorePrelude(ShaderRecompiler::MslPreludeChunk::Common).text) + ShaderRecompiler::MslTableDeclaration(layout);
    source += "kernel void aps5_main(device const Aps5Table& aps5_table [[buffer(0)]], device uint* out [[buffer(1)]]) {\n    uint width = 0u;\n";
    for (const auto& range : layout.ranges) {
        if (range.kind != ShaderRecompiler::MslAbi::EntryKind::Texture) continue;
        const auto field = "aps5_table." + ShaderRecompiler::MslTableField(range.binding) + "[1]";
        source += "    width += is_null_texture(" + field + ") ? 0u : " + field + ".get_width();\n";
    }
    // A table access as MslImage.cpp emits it: a texture and a sampler read through the arrays.
    source += "    {\n        const uint e = out[1];\n        texture2d<float> t = aps5_table.image_table.k1.address[e].value;\n";
    source += "        width += is_null_texture(t) ? 0u : uint(t.sample(aps5_table.sampler_table[0].address[e + 1u].value, float2(0.5f), level(0.0f)).x);\n";
    source += "        depth2d<float> d = aps5_table.image_table.k22.address[e].value;\n        width += is_null_texture(d) ? 0u : d.get_width();\n    }\n";
    source += "    out[0] = width + aps5_table.buffers[1].bytes;\n}\n";
    id<MTLLibrary> library = CompileLibrary(device, "descriptor table types", source);
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@(ShaderRecompiler::MslAbi::EntryPoint)];
    Require(function != nil, "descriptor table types: the library has no entry point");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    Require(pipeline != nil, "descriptor table types: compute pipeline creation failed: " + Describe(error));
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
            Require(device.Target().backend == ShaderRecompiler::ShaderBackend::Msl, "the Metal device target does not select the MSL backend");
            NSError* error = nil;
            Require([metal newLibraryWithSource:@"kernel void aps5_main(device uint* x [[buffer(0)]]) { x[0] = undefined_name; }" options:nil error:&error] == nil && error != nil, "the checker accepted invalid MSL");
            std::vector<std::uint32_t> backing(512u, 0u);
            for (const auto& program : Programs) {
                CompileCheck(metal, program, Recompile(device, program, backing));
                std::printf("%s: MSL compiled\n", program.name);
            }
            TableTypesCheck(metal);
            std::puts("descriptor table types: MSL compiled");
        }
        std::puts("msl compile check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
