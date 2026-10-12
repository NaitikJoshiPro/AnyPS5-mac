#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_TESTS_METALPIPELINECACHECHECKS_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_TESTS_METALPIPELINECACHECHECKS_H

// Work package D5: the fixed pipeline workload of the pipeline cache tests (two compute pipelines of
// one kernel with different function constants, one render pipeline) and a tiny GPU run of the
// compute pipelines. Objective-C++ only; header-only so each test executable has its own copy.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalPipelineCache.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace AgcDriver::Graphics::MetalPipelineCacheChecks {

inline int Fail(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    return 1;
}

inline constexpr std::uint64_t WorkloadPipelines = 3;

// Metal shader validation disables the archive (MetalPipelineCache.h): the tests then check that
// instead of the reuse.
inline bool ShaderValidation() {
    const char* value = std::getenv("MTL_SHADER_VALIDATION");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

inline Metal::PipelineFunction ComputeFunction(std::uint32_t multiplier) {
    static const auto shader = std::make_shared<const ShaderRecompiler::MslShader>(ShaderRecompiler::MslShader{
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "constant uint aps5_fc7 [[function_constant(7)]];\n"
        "constant uint aps5_multiplier = is_function_constant_defined(aps5_fc7) ? aps5_fc7 : 1u;\n"
        "kernel void aps5_main(device uint* data [[buffer(0)]], uint index [[thread_position_in_grid]]) {\n"
        "    data[index] = data[index] * aps5_multiplier + index;\n"
        "}\n",
        {}});
    Metal::PipelineFunction function;
    function.shader = shader;
    function.constants = {{7u, multiplier}};
    return function;
}

inline Metal::RenderPipelineRequest RenderRequest() {
    static const auto vertex = std::make_shared<const ShaderRecompiler::MslShader>(ShaderRecompiler::MslShader{
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "vertex float4 aps5_main(uint id [[vertex_id]]) { return float4(float(id & 1u), float(id >> 1u), 0.0, 1.0); }\n",
        {}});
    static const auto fragment = std::make_shared<const ShaderRecompiler::MslShader>(ShaderRecompiler::MslShader{
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "fragment float4 aps5_main(float4 position [[position]]) { return float4(position.x * 0.001, 0.25, 0.5, 1.0); }\n",
        {}});
    Metal::RenderPipelineRequest request;
    request.vertex.shader = vertex;
    Metal::PipelineFunction pixel;
    pixel.shader = fragment;
    request.fragment = pixel;
    request.colors[0].format = MTLPixelFormatRGBA8Unorm;
    request.topology = MTLPrimitiveTopologyClassTriangle;
    return request;
}

// Builds the workload and runs both compute pipelines on 64 dwords each (results: the 128 dwords).
// Returns an error message, empty on success.
inline std::string BuildWorkload(id<MTLDevice> device, Metal::PipelineCache& cache, std::vector<std::uint32_t>* results) {
    id<MTLComputePipelineState> pipelines[2] = {cache.Compute(ComputeFunction(3u)), cache.Compute(ComputeFunction(5u))};
    if (cache.Render(RenderRequest()) == nil) return "no render pipeline";
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* tableDescriptor = [MTL4ArgumentTableDescriptor new];
    tableDescriptor.maxBufferBindCount = 1;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:tableDescriptor error:&error];
    id<MTLBuffer> buffer = [device newBufferWithLength:128 * 4 options:MTLResourceStorageModeShared];
    if (queue == nil || allocator == nil || commands == nil || table == nil || buffer == nil) return "Metal 4 setup failed";
    auto* words = static_cast<std::uint32_t*>(buffer.contents);
    for (std::uint32_t index = 0; index < 128; ++index) words[index] = 0x1000u + index * 7u;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    [residency addAllocation:buffer];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setArgumentTable:table];
    for (std::uint32_t pass = 0; pass < 2; ++pass) {
        [encoder setComputePipelineState:pipelines[pass]];
        [table setAddress:buffer.gpuAddress + pass * 256u atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [encoder endEncoding];
    [commands endCommandBuffer];
    id<MTLSharedEvent> event = [device newSharedEvent];
    const id<MTL4CommandBuffer> list[] = {commands};
    [queue commit:list count:1];
    [queue signalEvent:event value:1];
    if (![event waitUntilSignaledValue:1 timeoutMS:10000]) return "the GPU did not finish within 10 s";
    results->assign(words, words + 128);
    for (std::uint32_t index = 0; index < 128; ++index) {
        const std::uint32_t original = 0x1000u + index * 7u;
        const std::uint32_t expected = original * (index < 64 ? 3u : 5u) + (index & 63u);
        if (words[index] != expected) return "dword " + std::to_string(index) + " is " + std::to_string(words[index]) + ", expected " + std::to_string(expected);
    }
    return {};
}

}

#endif
