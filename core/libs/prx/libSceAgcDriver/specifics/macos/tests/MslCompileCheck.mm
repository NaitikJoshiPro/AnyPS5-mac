// Compile-only check of MetalBackend output (no GPU work is submitted): every program below is
// recompiled for the Metal device's target, its MSL is compiled with newLibraryWithSource and a
// compute pipeline is built with the result's function constants. Also checks that the checker
// itself rejects invalid MSL and that the reflection matches the MslAbi table layout.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Recompiler.hpp"
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

struct Program {
    const char* name;
    std::span<const std::uint32_t> code;
    std::uint32_t threads;
    std::uint32_t buffers;
};

constexpr std::array<Program, 1> Programs{{
    {"vop1_control", Vop1ControlCode, 64u, 2u},
}};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), count, 0x01016facu};
}

ShaderRecompiler::RecompileResult Recompile(const AgcDriver::VulkanDevice& device, const Program& program, std::span<std::uint32_t> backing) {
    std::vector<std::uint32_t> userData(program.buffers * 4u, 0u);
    for (std::uint32_t buffer = 0; buffer < program.buffers; ++buffer) {
        const auto descriptor = BufferDescriptor(backing.data(), static_cast<std::uint32_t>(backing.size()));
        std::copy(descriptor.begin(), descriptor.end(), userData.begin() + buffer * 4u);
    }
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(program.code.data()), std::as_bytes(program.code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{program.threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(program.code.data()), program.code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

void CompileCheck(id<MTLDevice> device, const Program& program, const ShaderRecompiler::RecompileResult& result) {
    Require(!result.msl.empty(), std::string(program.name) + ": the recompiler produced no MSL for a Metal target");
    Require(result.spirv.empty(), std::string(program.name) + ": the recompiler also produced SPIR-V for a Metal target");
    const auto& reflection = result.msl.Reflection();
    Require(reflection.abiVersion == ShaderRecompiler::MslAbi::Version, std::string(program.name) + ": unexpected MslAbi version");
    Require(reflection.stage == ShaderRecompiler::MslAbi::Stage::Kernel, std::string(program.name) + ": a compute program must reflect a kernel");
    Require(reflection.threadsPerThreadgroup[0] * reflection.threadsPerThreadgroup[1] * reflection.threadsPerThreadgroup[2] == program.threads, std::string(program.name) + ": threadgroup size does not match the workgroup");
    const auto* buffers = reflection.table.Find(ShaderRecompiler::RuntimeAbi::Binding::Buffers);
    Require(buffers != nullptr && buffers->count == program.buffers && buffers->offset == 0u, std::string(program.name) + ": the descriptor table lacks the guest buffers");
    Require(reflection.table.bytes == program.buffers * sizeof(ShaderRecompiler::MslAbi::BufferEntry), std::string(program.name) + ": unexpected descriptor table size");
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:result.msl.Source().c_str()] options:options error:&error];
    if (library == nil) {
        std::cerr << result.msl.Source() << '\n';
        throw std::runtime_error(std::string(program.name) + ": MSL does not compile: " + Describe(error));
    }
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    for (const auto& constant : result.specialization) {
        const std::uint32_t value = constant.value;
        [constants setConstantValue:&value type:MTLDataTypeUInt atIndex:constant.id];
    }
    id<MTLFunction> function = [library newFunctionWithName:@(ShaderRecompiler::MslAbi::EntryPoint) constantValues:constants error:&error];
    Require(function != nil, std::string(program.name) + ": the library has no specializable entry point: " + Describe(error));
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    Require(pipeline != nil, std::string(program.name) + ": compute pipeline creation failed: " + Describe(error));
    Require(pipeline.maxTotalThreadsPerThreadgroup >= program.threads, std::string(program.name) + ": the pipeline cannot run the workgroup");
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
            std::vector<std::uint32_t> backing(256u, 0u);
            for (const auto& program : Programs) {
                CompileCheck(metal, program, Recompile(device, program, backing));
                std::printf("%s: MSL compiled\n", program.name);
            }
        }
        std::puts("msl compile check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
