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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int Skipped = 77;
constexpr std::uint32_t TargetSize = 2;
constexpr std::uint32_t Samples = 4;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

float lastSink = 0.0f;
std::string sinkValues;

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
struct BareIn {
    float2 value [[user(value), center_perspective]];
};
fragment void offset_rate_bare(BareIn in [[stage_in]], device atomic_uint* counter [[buffer(0)]], device float* sink [[buffer(1)]]) {
    record(counter, sink, in.value);
}
fragment void offset_rate_none(device atomic_uint* counter [[buffer(0)]], device float* sink [[buffer(1)]]) {
    record(counter, sink, float2(0.0f));
}
fragment float4 offset_rate_colored(CenterIn in [[stage_in]], device atomic_uint* counter [[buffer(0)]], device float* sink [[buffer(1)]]) {
    record(counter, sink, in.value);
    return float4(in.value, 0.0f, 1.0f);
}
fragment void offset_rate_mask(uint mask [[sample_mask]], device atomic_uint* counter [[buffer(0)]], device float* sink [[buffer(1)]]) {
    record(counter, sink, float2(float(popcount(mask)), 0.0f));
}
fragment void offset_rate_pull_center(PullIn in [[stage_in]], device atomic_uint* counter [[buffer(0)]], device float* sink [[buffer(1)]]) {
    record(counter, sink, in.value.interpolate_at_center());
}
struct SurveyOut {
    float4 position [[position]];
};
vertex SurveyOut survey_vertex(uint vid [[vertex_id]], constant float2* corners [[buffer(0)]]) {
    SurveyOut out;
    out.position = float4(corners[vid], 0.5f, 1.0f);
    return out;
}
// Per-pixel slots of 4 words: invocations, OR of coverage masks, sum of coverage popcounts, invocations whose
// position is not the pixel center.
static void survey(device atomic_uint* slots, float4 position, uint mask, uint width) {
    const uint pixel = uint(position.y) * width + uint(position.x);
    device atomic_uint* slot = slots + pixel * 4u;
    atomic_fetch_add_explicit(slot, 1u, memory_order_relaxed);
    atomic_fetch_or_explicit(slot + 1, mask, memory_order_relaxed);
    atomic_fetch_add_explicit(slot + 2, popcount(mask), memory_order_relaxed);
    if (fract(position.x) != 0.5f || fract(position.y) != 0.5f) atomic_fetch_add_explicit(slot + 3, 1u, memory_order_relaxed);
}
fragment void survey_mask(float4 position [[position]], uint mask [[sample_mask]], device atomic_uint* slots [[buffer(0)]], constant uint& width [[buffer(1)]]) {
    survey(slots, position, mask, width);
}
fragment void survey_position(float4 position [[position]], device atomic_uint* slots [[buffer(0)]], constant uint& width [[buffer(1)]]) {
    survey(slots, position, 0u, width);
}
fragment void survey_sample(float4 position [[position]], uint mask [[sample_mask]], uint sample [[sample_id]], device atomic_uint* slots [[buffer(0)]], constant uint& width [[buffer(1)]]) {
    survey(slots, position, mask, width);
}
)";

// `attached`: the pass has a memoryless 4x color attachment the function never writes (write mask none), versus a
// pass without attachments (defaultRasterSampleCount).
std::uint32_t Count(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library, const char* fragment, bool attached) {
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"offset_rate_vertex"];
    descriptor.fragmentFunction = [library newFunctionWithName:@(fragment)];
    Require(descriptor.fragmentFunction != nil, std::string(fragment) + ": no function");
    descriptor.rasterSampleCount = Samples;
    if (attached) {
        descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatR8Unorm;
        descriptor.colorAttachments[0].writeMask = std::string(fragment) == "offset_rate_colored" ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    }
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    Require(pipeline != nil, std::string(fragment) + ": no pipeline: " + Describe(error));
    id<MTLBuffer> counter = [device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> sink = [device newBufferWithLength:64 * sizeof(float) options:MTLResourceStorageModeShared];
    *static_cast<std::uint32_t*>(counter.contents) = 0u;
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.renderTargetWidth = TargetSize;
    pass.renderTargetHeight = TargetSize;
    pass.defaultRasterSampleCount = Samples;
    if (attached) {
        MTLTextureDescriptor* target = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:TargetSize height:TargetSize mipmapped:NO];
        target.textureType = MTLTextureType2DMultisample;
        target.sampleCount = Samples;
        target.storageMode = MTLStorageModeMemoryless;
        target.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> texture = [device newTextureWithDescriptor:target];
        Require(texture != nil, std::string(fragment) + ": no memoryless 4x target");
        pass.colorAttachments[0].texture = texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    }
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:(MTLViewport){0.0, 0.0, double(TargetSize), double(TargetSize), 0.0, 1.0}];
    [encoder setScissorRect:(MTLScissorRect){0, 0, TargetSize, TargetSize}];
    [encoder setFragmentBuffer:counter offset:0 atIndex:0];
    [encoder setFragmentBuffer:sink offset:0 atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    Require(commands.status == MTLCommandBufferStatusCompleted, std::string(fragment) + ": the command buffer failed: " + Describe(commands.error));
    lastSink = static_cast<const float*>(sink.contents)[0];
    sinkValues.clear();
    const auto total = std::min<std::uint32_t>(*static_cast<const std::uint32_t*>(counter.contents), 64u);
    for (std::uint32_t i = 0; i < total; ++i) sinkValues += std::to_string(static_cast<int>(static_cast<const float*>(sink.contents)[i])) + " ";
    return *static_cast<const std::uint32_t*>(counter.contents);
}

struct SurveyResult {
    std::uint32_t invocations = 0;
    std::uint32_t pixels = 0;
    std::uint32_t maxPerPixel = 0;
    std::uint32_t popcounts = 0;
    std::uint32_t offCenter = 0;
};

// One triangle (pixel-space corners) into a width x height target at `samples`, counting each pixel's invocations.
SurveyResult Survey(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library, const char* fragment, std::uint32_t samples, std::uint32_t width, std::uint32_t height, const float (&corners)[6]) {
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"survey_vertex"];
    descriptor.fragmentFunction = [library newFunctionWithName:@(fragment)];
    descriptor.rasterSampleCount = samples;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    Require(pipeline != nil, std::string(fragment) + ": no pipeline: " + Describe(error));
    const std::size_t slotBytes = std::size_t{width} * height * 4u * sizeof(std::uint32_t);
    id<MTLBuffer> slots = [device newBufferWithLength:slotBytes options:MTLResourceStorageModeShared];
    std::memset(slots.contents, 0, slotBytes);
    float ndc[6];
    for (int i = 0; i < 3; ++i) {
        ndc[2 * i] = corners[2 * i] / float(width) * 2.0f - 1.0f;
        ndc[2 * i + 1] = 1.0f - corners[2 * i + 1] / float(height) * 2.0f;
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.renderTargetWidth = width;
    pass.renderTargetHeight = height;
    pass.defaultRasterSampleCount = samples;
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:(MTLViewport){0.0, 0.0, double(width), double(height), 0.0, 1.0}];
    [encoder setScissorRect:(MTLScissorRect){0, 0, width, height}];
    [encoder setVertexBytes:ndc length:sizeof(ndc) atIndex:0];
    [encoder setFragmentBuffer:slots offset:0 atIndex:0];
    [encoder setFragmentBytes:&width length:sizeof(width) atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    Require(commands.status == MTLCommandBufferStatusCompleted, std::string(fragment) + ": the survey command buffer failed: " + Describe(commands.error));
    SurveyResult result;
    const auto* words = static_cast<const std::uint32_t*>(slots.contents);
    for (std::uint32_t pixel = 0; pixel < width * height; ++pixel) {
        const auto count = words[pixel * 4u];
        result.invocations += count;
        result.pixels += count != 0u ? 1u : 0u;
        result.maxPerPixel = std::max(result.maxPerPixel, count);
        result.popcounts += words[pixel * 4u + 2u];
        result.offCenter += words[pixel * 4u + 3u];
    }
    return result;
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
            const char* names[8] = {"offset_rate_offset", "offset_rate_center", "offset_rate_sample", "offset_rate_pull_center", "offset_rate_bare", "offset_rate_none", "offset_rate_colored", "offset_rate_mask"};
            (void)0;
            std::uint32_t counts[2][8] = {};
            for (int attached = 0; attached < 2; ++attached) {
                for (int i = 0; i < 8; ++i) counts[attached][i] = Count(device, queue, library, names[i], attached != 0);
                std::printf("fragment invocations over %u pixels at %ux %s: interpolate_at_offset %u, center_perspective %u, interpolate_at_sample %u, interpolate_at_center %u, center input without [[position]] %u, no inputs %u, color output %u, coverage mask read %u (first popcount %.0f)\n", TargetSize * TargetSize, Samples, attached != 0 ? "with an unwritten color attachment" : "without attachments", counts[attached][0], counts[attached][1], counts[attached][2], counts[attached][3], counts[attached][4], counts[attached][5], counts[attached][6], counts[attached][7], lastSink);
            }
            std::printf("coverage popcounts of the last pass's invocations: %s\n", sinkValues.c_str());
            // Survey: per-pixel invocation counts of a triangle inside an 8x8 target and of one past its edges.
            const float inside[6] = {1.0f, 1.0f, 7.0f, 1.0f, 1.0f, 7.0f};
            const float full[6] = {-8.0f, -8.0f, 24.0f, -8.0f, -8.0f, 24.0f};
            const float tiny[6] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
            struct Shape { const char* name; const float (*corners)[6]; std::uint32_t size; };
            const Shape shapes[3] = {{"inside", &inside, 8u}, {"oversized", &full, 8u}, {"2x2 probe triangle", &tiny, 2u}};
            // WIP probe: ctest truncates passed output, so the survey goes to a file under the main checkout's logs.
            std::FILE* out = std::fopen("/Users/naitikjoshi/PS5/.mac-work/logs/sample8/offset-rate-survey.txt", "w");
            if (out == nullptr) out = stdout;
            for (const char* function : {"survey_position", "survey_mask", "survey_sample"}) {
                for (const std::uint32_t samples : {1u, 2u, 4u}) {
                    for (const auto& shape : shapes) {
                        const auto r = Survey(device, queue, library, function, samples, shape.size, shape.size, *shape.corners);
                        std::fprintf(out, "survey %s %ux %s %ux%u: %u invocations over %u pixels, at most %u per pixel, coverage popcount sum %u, %u off-center positions\n", function, samples, shape.name, shape.size, shape.size, r.invocations, r.pixels, r.maxPerPixel, r.popcounts, r.offCenter);
                    }
                }
            }
            if (out != stdout) std::fclose(out);
            const auto* rates = counts[1];
            // Measured on an M3 (2026-10-22, with and without validation): every variant, a function without inputs
            // included, ran 16 times over the 4 pixels, with overlapping input coverage masks (TechnicalDebt
            // metal-fragment-invocation-rate). What the MSL backend relies on is checked: an offset or pull-model
            // center read does not change the rate a plain center read runs at, and per-sample reads run at least
            // once per sample.
            Require(rates[0] == rates[1] && rates[3] == rates[1], "a pull-model interpolate_at_offset or interpolate_at_center function runs at a different rate than a center_perspective one");
            Require(rates[2] >= TargetSize * TargetSize * Samples, "an interpolate_at_sample([[sample_id]]) function ran fewer times than once per sample");
        }
        std::puts("msl offset rate check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
