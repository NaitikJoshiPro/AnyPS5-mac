// Startup self-tests of the Metal device (work package D1, docs/mac/M2_DESIGN.md §5): GPU facts
// that Apple does not document or that differ from the Vulkan model the front end assumes, measured
// once per process with tiny offscreen draws (no window). Ported from the research probes
// (.mac-work/research/metal-probes/probe_runtime.swift, metal-limits.md §4).
// - Vertex-stage SIMD-group functions: VulkanDevice::SubgroupStages() reports the vertex stage when
//   simd_sum/simd_ballot/simd_prefix_exclusive_sum/simd_shuffle/simd_max in a vertex function give
//   consistent 32-wide groups of consecutive vertices.
// - Barycentric order: [[barycentric_coord]] for a triangle list and for the odd triangle of a
//   strip, compared with per-vertex one-hot attributes (DeviceProfile::oddStripOrder).
// The draws run on a Metal 3 queue of their own (not the device's MTL4 queue), each waited for with
// a timeout.

#include "MetalDeviceState.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver::Metal {

namespace {

constexpr std::int64_t SelfTestTimeoutNs = 5'000'000'000;

constexpr const char* SelfTestSource = R"(#include <metal_stdlib>
using namespace metal;

struct Aps5SimdResult {
    uint sum;
    uint ballotLow;
    uint ballotHigh;
    uint prefix;
    uint laneZeroVertex;
    uint maxVertex;
};

vertex void aps5_vertex_simd(uint vid [[vertex_id]], device Aps5SimdResult* results [[buffer(0)]]) {
    Aps5SimdResult result;
    result.sum = simd_sum(1u);
    const ulong ballot = static_cast<ulong>(simd_ballot(true));
    result.ballotLow = uint(ballot);
    result.ballotHigh = uint(ballot >> 32);
    result.prefix = simd_prefix_exclusive_sum(1u);
    result.laneZeroVertex = simd_shuffle(vid, 0);
    result.maxVertex = simd_max(vid);
    results[vid] = result;
}

struct Aps5BaryVertex {
    float4 position [[position]];
    float4 identity;
};

vertex Aps5BaryVertex aps5_bary_vertex(uint vid [[vertex_id]], constant float2* positions [[buffer(0)]]) {
    Aps5BaryVertex out;
    out.position = float4(positions[vid], 0.5, 1.0);
    out.identity = float4(vid == 0 ? 1.0 : 0.0, vid == 1 ? 1.0 : 0.0, vid == 2 ? 1.0 : 0.0, vid == 3 ? 1.0 : 0.0);
    return out;
}

struct Aps5BaryTargets {
    float4 barycentric [[color(0)]];
    float4 identity [[color(1)]];
};

fragment Aps5BaryTargets aps5_bary_fragment(Aps5BaryVertex in [[stage_in]], float3 barycentric [[barycentric_coord, center_no_perspective]], uint primitive [[primitive_id]]) {
    Aps5BaryTargets out;
    out.barycentric = float4(barycentric, float(primitive));
    out.identity = in.identity;
    return out;
}
)";

struct SimdResult {
    std::uint32_t sum;
    std::uint32_t ballotLow;
    std::uint32_t ballotHigh;
    std::uint32_t prefix;
    std::uint32_t laneZeroVertex;
    std::uint32_t maxVertex;
};

bool Wait(id<MTLCommandBuffer> commands, std::string& failure, const char* what) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [commands addCompletedHandler:^(id<MTLCommandBuffer>) {
        dispatch_semaphore_signal(done);
    }];
    [commands commit];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, SelfTestTimeoutNs)) != 0) {
        failure = std::string(what) + ": the GPU did not finish within 5 s";
        return false;
    }
    if (commands.status != MTLCommandBufferStatusCompleted) {
        failure = std::string(what) + ": command buffer failed: " + Describe(commands.error);
        return false;
    }
    return true;
}

// 100 points through a vertex function with rasterization off; each vertex stores what its
// SIMD group computed. Consistent: every group is the 32 (or, last, fewer) consecutive vertices
// starting at a multiple of 32, and ballot, prefix and max agree with the group.
void MeasureVertexSimd(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library, DeviceProfile& profile) {
    constexpr std::uint32_t Vertices = 100;
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"aps5_vertex_simd"];
    descriptor.rasterizationEnabled = NO;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (pipeline == nil) {
        profile.summary += "vertex-simd: pipeline failed: " + Describe(error) + "\n";
        return;
    }
    id<MTLBuffer> results = [device newBufferWithLength:Vertices * sizeof(SimdResult) options:MTLResourceStorageModeShared];
    std::memset(results.contents, 0xff, results.length);
    MTLTextureDescriptor* targetDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
    targetDescriptor.usage = MTLTextureUsageRenderTarget;
    targetDescriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> target = [device newTextureWithDescriptor:targetDescriptor];
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setVertexBuffer:results offset:0 atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:Vertices];
    [encoder endEncoding];
    std::string failure;
    if (!Wait(commands, failure, "vertex-simd")) {
        profile.summary += failure + "\n";
        return;
    }
    const auto* values = static_cast<const SimdResult*>(results.contents);
    std::uint32_t width = 0;
    bool consistent = true;
    for (std::uint32_t vertex = 0; vertex < Vertices; ++vertex) {
        const auto& value = values[vertex];
        width = std::max(width, value.sum);
        const auto popcount = static_cast<std::uint32_t>(__builtin_popcount(value.ballotLow) + __builtin_popcount(value.ballotHigh));
        const auto first = vertex - vertex % 32u;
        const auto last = std::min(first + 32u, Vertices) - 1u;
        consistent = consistent && popcount == value.sum && value.prefix < value.sum && value.laneZeroVertex == first && value.maxVertex == last && value.sum == last - first + 1u && value.prefix == vertex - first;
    }
    profile.vertexSimdWidth = width;
    profile.vertexSimdGroups = consistent && width == 32u;
    profile.summary += std::string("vertex-simd: ") + (profile.vertexSimdGroups ? "32-wide groups of consecutive vertices" : "inconsistent") + " (width " + std::to_string(width) + ", vertex 37: sum " + std::to_string(values[37].sum) + " prefix " + std::to_string(values[37].prefix) + " lane0 " + std::to_string(values[37].laneZeroVertex) + ")\n";
}

struct Pixels {
    std::vector<float> barycentric;
    std::vector<float> identity;
};

constexpr std::uint32_t BaryTarget = 8;

bool DrawBarycentric(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLRenderPipelineState> pipeline, MTLPrimitiveType primitive, const std::vector<float>& positions, Pixels& pixels, std::string& failure) {
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:BaryTarget height:BaryTarget mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget;
    descriptor.storageMode = MTLStorageModeShared;
    std::array<id<MTLTexture>, 2> targets {[device newTextureWithDescriptor:descriptor], [device newTextureWithDescriptor:descriptor]};
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    for (std::size_t index = 0; index < targets.size(); ++index) {
        pass.colorAttachments[index].texture = targets[index];
        pass.colorAttachments[index].loadAction = MTLLoadActionClear;
        pass.colorAttachments[index].clearColor = MTLClearColorMake(-1, -1, -1, -1);
        pass.colorAttachments[index].storeAction = MTLStoreActionStore;
    }
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setVertexBytes:positions.data() length:positions.size() * sizeof(float) atIndex:0];
    [encoder drawPrimitives:primitive vertexStart:0 vertexCount:positions.size() / 2u];
    [encoder endEncoding];
    if (!Wait(commands, failure, "barycentric")) return false;
    pixels.barycentric.assign(BaryTarget * BaryTarget * 4u, 0.0f);
    pixels.identity.assign(BaryTarget * BaryTarget * 4u, 0.0f);
    const auto region = MTLRegionMake2D(0, 0, BaryTarget, BaryTarget);
    [targets[0] getBytes:pixels.barycentric.data() bytesPerRow:BaryTarget * 16u fromRegion:region mipmapLevel:0];
    [targets[1] getBytes:pixels.identity.data() bytesPerRow:BaryTarget * 16u fromRegion:region mipmapLevel:0];
    return true;
}

// The vertices (0-3) the three barycentric weights of a covered pixel belong to, from a pixel of
// primitive `primitive` whose three per-vertex weights are distinct. Empty when no pixel decides.
std::vector<std::uint32_t> BarycentricOrder(const Pixels& pixels, std::uint32_t primitive) {
    for (std::uint32_t pixel = 0; pixel < BaryTarget * BaryTarget; ++pixel) {
        const float* bary = &pixels.barycentric[pixel * 4u];
        const float* identity = &pixels.identity[pixel * 4u];
        if (bary[0] < 0.0f || static_cast<std::uint32_t>(bary[3]) != primitive) continue;
        std::vector<std::uint32_t> present;
        for (std::uint32_t vertex = 0; vertex < 4u; ++vertex) {
            if (identity[vertex] > 0.01f) present.push_back(vertex);
        }
        if (present.size() != 3u) continue;
        const auto apart = [&](std::uint32_t a, std::uint32_t b) { return std::fabs(identity[present[a]] - identity[present[b]]) > 0.05f; };
        if (!apart(0, 1) || !apart(0, 2) || !apart(1, 2)) continue;
        std::vector<std::uint32_t> order;
        for (std::uint32_t weight = 0; weight < 3u; ++weight) {
            for (const auto vertex : present) {
                if (std::fabs(identity[vertex] - bary[weight]) < 0.02f) order.push_back(vertex);
            }
        }
        if (order.size() == 3u) return order;
    }
    return {};
}

std::string OrderText(const std::vector<std::uint32_t>& order) {
    if (order.empty()) return "undecided";
    std::string text = "(";
    for (std::size_t index = 0; index < order.size(); ++index) text += (index == 0u ? "v" : ", v") + std::to_string(order[index]);
    return text + ")";
}

void MeasureBarycentrics(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library, DeviceProfile& profile) {
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"aps5_bary_vertex"];
    descriptor.fragmentFunction = [library newFunctionWithName:@"aps5_bary_fragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    descriptor.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Float;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (pipeline == nil) {
        profile.summary += "barycentric: pipeline failed: " + Describe(error) + "\n";
        return;
    }
    std::string failure;
    Pixels list;
    if (!DrawBarycentric(device, queue, pipeline, MTLPrimitiveTypeTriangle, {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f}, list, failure)) {
        profile.summary += failure + "\n";
        return;
    }
    const auto listOrder = BarycentricOrder(list, 0u);
    profile.barycentrics = listOrder == std::vector<std::uint32_t>{0u, 1u, 2u};
    Pixels strip;
    if (!DrawBarycentric(device, queue, pipeline, MTLPrimitiveTypeTriangleStrip, {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f}, strip, failure)) {
        profile.summary += failure + "\n";
        return;
    }
    const auto evenOrder = BarycentricOrder(strip, 0u);
    const auto oddOrder = BarycentricOrder(strip, 1u);
    using StripOrder = DeviceProfile::StripOrder;
    if (oddOrder == std::vector<std::uint32_t>{1u, 3u, 2u}) {
        profile.oddStripOrder = StripOrder::I_I2_I1;
    } else if (oddOrder == std::vector<std::uint32_t>{2u, 1u, 3u}) {
        profile.oddStripOrder = StripOrder::I1_I_I2;
    } else {
        profile.oddStripOrder = oddOrder.empty() ? StripOrder::Unknown : StripOrder::Other;
    }
    profile.barycentrics = profile.barycentrics && evenOrder == std::vector<std::uint32_t>{0u, 1u, 2u};
    profile.summary += "barycentric: list " + OrderText(listOrder) + ", strip even " + OrderText(evenOrder) + ", strip odd " + OrderText(oddOrder) + (profile.oddStripOrder == StripOrder::I_I2_I1 ? " = (v_i, v_i+2, v_i+1), the Vulkan strip order" : "") + "\n";
}

}

const DeviceProfile& Profile(id<MTLDevice> device) {
    static std::once_flag once;
    static DeviceProfile profile;
    std::call_once(once, [&] {
        @autoreleasepool {
            try {
                NSError* error = nil;
                id<MTLLibrary> library = [device newLibraryWithSource:@(SelfTestSource) options:[MTLCompileOptions new] error:&error];
                if (library == nil) {
                    profile.failure = "self-test library failed: " + Describe(error);
                    return;
                }
                id<MTLCommandQueue> queue = [device newCommandQueue];
                if (queue == nil) {
                    profile.failure = "no Metal 3 queue for the self-tests";
                    return;
                }
                MeasureVertexSimd(device, queue, library, profile);
                MeasureBarycentrics(device, queue, library, profile);
                profile.measured = true;
            } catch (const std::exception& exception) {
                profile.failure = exception.what();
            }
            if (!profile.failure.empty()) profile.summary += "self-test failed: " + profile.failure + "\n";
            if (std::getenv("APS5_METAL_PROFILE") != nullptr) std::fprintf(stderr, "[metal] device profile:\n%s", profile.summary.c_str());
        }
    });
    return profile;
}

}
