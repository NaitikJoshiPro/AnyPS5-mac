// Metal twin of tests/execution/InterpolationModes.cpp (agc_driver_interpolation_modes), swapped in by
// twins/B2-alu.cmake (work package B2). The upstream test assembles a SPIR-V module from SpirvBackend
// internals around EmitFPInterpolateF32/F16; this twin runs the same 32 hardware rows, 3 operations and
// 8 float modes through the MetalBackend's FPInterpolateF32/F16 routine (aps5_interpolate in the Float64
// prelude chunk, exactly the text MslPrelude() hands the emitter) on the GPU, the f16 result packed as
// ConvertF16F32 packs it. The upstream file is compiled here unchanged with its main renamed, so its
// table keeps tracking upstream (tools/mac/port_ledger.py).

#define main AgcDriverInterpolationModesVulkanMain
#include "../../../tests/execution/InterpolationModes.cpp"
#undef main

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalBackend/MslEmitterState.hpp"
#include <cstring>

namespace {

constexpr const char* TwinKernel = R"(
kernel void aps5_interpolation_modes(device uint* data [[buffer(0)]], constant uint& mode [[buffer(1)]], constant uint& rows [[buffer(2)]],
                                     uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    device uint* words = data + row * 12u;
    for (uint op = 0u; op < 3u; ++op) {
        const bool halfResult = op == 1u;
        const float value = aps5_interpolate(as_type<float>(words[op * 3u]), as_type<float>(words[op * 3u + 1u]), as_type<float>(words[op * 3u + 2u]), halfResult, mode);
        words[9u + op] = halfResult ? aps5_f32_to_f16(value) : as_type<uint>(value);
    }
}
)";

void RunOnMetal(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, std::uint32_t mode) {
    Data.fill(Sentinel);
    for (std::uint32_t row = 0; row < Rows.size(); ++row) std::copy(Rows[row].input.begin(), Rows[row].input.end(), Data.begin() + row * Words);
    id<MTLBuffer> buffer = [device newBufferWithBytes:Data.data() length:sizeof(Data) options:MTLResourceStorageModeShared];
    std::uint32_t flags = ModeFlags[mode];
    std::uint32_t rows = static_cast<std::uint32_t>(Rows.size());
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&flags length:4 atIndex:1];
    [encoder setBytes:&rows length:4 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(rows, 1, 1)];
    [encoder endEncoding];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [commands addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(done); }];
    [commands commit];
    Require(dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10ll * NSEC_PER_SEC)) == 0, "interpolation modes: the GPU did not finish within 10 s");
    Require(commands.status == MTLCommandBufferStatusCompleted, std::string("interpolation modes: command buffer failed: ") + (commands.error ? commands.error.localizedDescription.UTF8String : "unknown"));
    std::memcpy(Data.data(), buffer.contents, sizeof(Data));
    for (std::uint32_t row = 0; row < Rows.size(); ++row) {
        for (std::uint32_t input = 0; input < 9u; ++input) Require(Data[row * Words + input] == Rows[row].input[input], "interpolation modes changed an input");
        for (std::uint32_t op = 0; op < 3u; ++op) {
            const auto actual = Data[row * Words + 9u + op];
            const auto expected = Rows[row].expected[mode][op];
            char message[200];
            std::snprintf(message, sizeof(message), "%s, %s, row %u: got 0x%08x, expected 0x%08x", OpNames[op], ModeNames[mode], row, actual, expected);
            Require(actual == expected, message);
        }
    }
    for (std::size_t index = Rows.size() * Words; index < Data.size(); ++index) Require(Data[index] == Sentinel, "interpolation modes wrote beyond the rows");
}

}

int main() {
    try {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                std::puts("skipped, the machine has no Metal device");
                return VulkanTestSkipped;
            }
            std::string source = "#include <metal_stdlib>\nusing namespace metal;\n";
            for (const auto chunk : {MslPreludeChunk::Alu, MslPreludeChunk::Float16, MslPreludeChunk::Float64}) source += MslPrelude(chunk).text;
            source += TwinKernel;
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion4_0;
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
            Require(library != nil, std::string("interpolation modes: the MSL does not compile: ") + (error ? error.localizedDescription.UTF8String : "unknown"));
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"aps5_interpolation_modes"] error:&error];
            id<MTLCommandQueue> queue = [device newCommandQueue];
            Require(pipeline != nil && queue != nil, "interpolation modes: no pipeline or queue");
            for (std::uint32_t mode = 0; mode < ModeFlags.size(); ++mode) RunOnMetal(device, queue, pipeline, mode);
        }
        std::puts("interpolation modes passed on Metal: 32 hardware rows, 3 ops, 8 float modes");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
