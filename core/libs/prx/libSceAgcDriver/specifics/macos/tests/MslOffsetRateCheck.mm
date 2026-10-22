// agc_metal_msl_offset_rate: does a fragment function that reads a pull-model interpolant<> with
// interpolate_at_offset run once per pixel, as one with a plain center_perspective input does? The MSL
// backend reads PERSP/LINEAR_SAMPLE barycentrics of a once-per-pixel program at sample 0's location that
// way (MslStageIo.cpp OffsetBarycentric); agc_driver_pixel_sample_iteration saw such a program run once
// per sample on Metal. Each variant draws one full-target triangle into a 2x2, 4x attachment-less pass
// and counts its fragment invocations with an atomic counter:
// (a) interpolate_at_offset on an interpolant<>, (b) a plain center_perspective input,
// (c) interpolate_at_sample([[sample_id]]) (per sample, the control), (d) interpolate_at_center() on an
// interpolant<> (pull model without an offset).
// The test prints the four counts and fails if (a) differs from (b). Tiny GPU work: 4 draws of 4 pixels.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int Skipped = 77;
constexpr std::uint32_t Size = 2;
constexpr std::uint32_t Samples = 4;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Describe(NSError* error) {
    return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

constexpr const char* Source = R"(
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 position [[position]];
    float2 value [[user(value)]];
};
vertex VOut offset_rate_vertex(uint vid [[vertex_id]]) {
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    VOut out;
    out.position = float4(corners[vid], 0.0f, 1.0f);
    out.value = corners[vid];
    return out;
}
struct PullIn {
    float4 position [[position]];
    interpolant<float2, interpolation::perspective> value [[user(value)]];
};
struct CenterIn {
    float4 position [[position]];
    float2 value [[user(value), center_perspective]];
};
static void record(device atomic_uint* counter, device float* sink, float2 value) {
    const uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < 64u) sink[slot] = value.x + value.y;
}
fragment void offset_rate_offset(PullIn in [[stage_in]], device atomic_uint* counter [[buffer(0)]], device float* sink [[buffer(1)]]) {
    record(counter, sink, in.value.interpolate_at_offset(float2(0.375f, 0.125f)));
}
fragment void offset_rate_center(CenterIn in [[stage_in]], device atomic_uint* counter [[buffer(0)]], device float* sink [[buffer(1)]]) {
    record(counter, sink, in.value);
}
fragment void offset_rate_sample(PullIn in [[stage_in]], uint sample_index [[sample_id]], device atomic_uint* counter [[buffer(0)]], device float* sink [[buffer(1)]]) {
    record(counter, sink, in.value.interpolate_at_sample(sample_index));
}
fragment void offset_rate_pull_center(PullIn in [[stage_in]], device atomic_uint* counter [[buffer(0)]], device float* sink [[buffer(1)]]) {
    record(counter, sink, in.value.interpolate_at_center());
}
)";

std::uint32_t Count(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library, const char* fragment) {
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"offset_rate_vertex"];
    descriptor.fragmentFunction = [library newFunctionWithName:@(fragment)];
    Require(descriptor.fragmentFunction != nil, std::string(fragment) + ": no function");
    descriptor.rasterSampleCount = Samples;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    Require(pipeline != nil, std::string(fragment) + ": no pipeline: " + Describe(error));
    id<MTLBuffer> counter = [device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> sink = [device newBufferWithLength:64 * sizeof(float) options:MTLResourceStorageModeShared];
    *static_cast<std::uint32_t*>(counter.contents) = 0u;
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.renderTargetWidth = Size;
    pass.renderTargetHeight = Size;
    pass.defaultRasterSampleCount = Samples;
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setFragmentBuffer:counter offset:0 atIndex:0];
    [encoder setFragmentBuffer:sink offset:0 atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    Require(commands.status == MTLCommandBufferStatusCompleted, std::string(fragment) + ": the command buffer failed: " + Describe(commands.error));
    return *static_cast<const std::uint32_t*>(counter.contents);
}

}

int main() {
    try {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil || ![device supportsTextureSampleCount:Samples]) {
                std::puts("skipped, no Metal device with 4x multisampling");
                return Skipped;
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:@(Source) options:[MTLCompileOptions new] error:&error];
            Require(library != nil, "the probe MSL does not compile: " + Describe(error));
            id<MTLCommandQueue> queue = [device newCommandQueue];
            const auto offset = Count(device, queue, library, "offset_rate_offset");
            const auto center = Count(device, queue, library, "offset_rate_center");
            const auto sample = Count(device, queue, library, "offset_rate_sample");
            const auto pullCenter = Count(device, queue, library, "offset_rate_pull_center");
            std::printf("fragment invocations over %u pixels at %ux: interpolate_at_offset %u, center_perspective %u, interpolate_at_sample %u, interpolate_at_center %u\n", Size * Size, Samples, offset, center, sample, pullCenter);
            Require(center == Size * Size, "a plain center_perspective function did not run once per pixel");
            Require(sample == Size * Size * Samples, "an interpolate_at_sample([[sample_id]]) function did not run once per sample");
            Require(offset == center, "an interpolate_at_offset function runs at a different rate than a center_perspective one");
        }
        std::puts("msl offset rate check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
