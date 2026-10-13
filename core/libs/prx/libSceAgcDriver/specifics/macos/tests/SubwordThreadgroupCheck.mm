// agc_metal_subword_threadgroup: a minimal repro of the TESS2 finding (83ed2d518, TechnicalDebt
// metal-tessellation gap 11): in the tessellation mesh function, uchar/ushort threadgroup arrays
// declared after an int corner array appeared to have their stores land in the corner array on the
// M3. This test runs the pre-fix vertex sharing shape (thread 0 fills `corners`, a barrier, every
// corner thread finds its owner corner, stores it in a ushort array, owners take a uchar slot from an
// atomic counter, a barrier, read back) in a compute kernel and in a mesh function, once with the
// sub-word arrays and once with 32-bit arrays, for 256, 255 (85 x 3) and 128 x 2 corners, the arrays sized as the mesh function sized them
// (slice x corners). It checks
// that the corner array is unchanged, owners match the CPU and slots are a permutation.
// The emitters no longer declare sub-word threadgroup arrays (all are uint/int words); this test
// only records whether the Metal compiler/GPU reproduces the corruption in isolation.
// Tiny: one threadgroup per dispatch, 256 threads.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int Skipped = 77;
constexpr int Threads = 256;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// FIRST_T / SLOT_T: ushort/uchar (pre-fix) or uint/uint (control).
constexpr const char* Source = R"(
#include <metal_stdlib>
using namespace metal;

#define APS5_BODY                                                                                   \
    threadgroup int corners[N];                                                                   \
    threadgroup uint count[1];                                                                      \
    threadgroup SLOT_T slot[N];                                                                   \
    threadgroup FIRST_T first[N];                                                                 \
    threadgroup atomic_uint vertices[1];                                                            \
    if (tid == 0u) {                                                                                \
        for (uint i = 0u; i < used; ++i) corners[i] = input[i];                                     \
        count[0] = used;                                                                            \
        atomic_store_explicit(&vertices[0], 0u, memory_order_relaxed);                              \
    }                                                                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                \
    const uint n = count[0];                                                                        \
    if (tid < n) {                                                                                  \
        uint owner = tid;                                                                           \
        for (uint s = 0u; s < tid; ++s) {                                                           \
            if (corners[s] == corners[tid]) { owner = s; break; }                                   \
        }                                                                                           \
        first[tid] = FIRST_T(owner);                                                                \
        if (owner == tid) slot[tid] = SLOT_T(atomic_fetch_add_explicit(&vertices[0], 1u, memory_order_relaxed)); \
    }                                                                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                \
    if (tid < n) {                                                                                  \
        output[tid * 3u] = uint(corners[tid]);                                                      \
        output[tid * 3u + 1u] = uint(first[tid]);                                                   \
        output[tid * 3u + 2u] = uint(slot[first[tid]]);                                             \
    }

kernel void aps5_compute(device const int* input [[buffer(0)]], device uint* output [[buffer(1)]], constant uint& used [[buffer(2)]], uint tid [[thread_index_in_threadgroup]])
{
    APS5_BODY
}

struct Aps5Vertex { float4 position [[position]]; };
struct Aps5Primitive { bool culled [[primitive_culled]]; };
using Aps5Mesh = metal::mesh<Aps5Vertex, Aps5Primitive, N, N, metal::topology::point>;

[[mesh]] void aps5_mesh(Aps5Mesh mesh, device const int* input [[buffer(0)]], device uint* output [[buffer(1)]], constant uint& used [[buffer(2)]], uint tid [[thread_index_in_threadgroup]])
{
    if (tid == 0u) mesh.set_primitive_count(0u);
    APS5_BODY
}

fragment float4 aps5_fragment() { return float4(0.0f); }
)";

struct Variant {
    const char* name;
    const char* firstType;
    const char* slotType;
};

}

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                std::puts("subword threadgroup: skipped, no Metal device");
                return Skipped;
            }
            id<MTLCommandQueue> queue = [device newCommandQueue];
            MTLTextureDescriptor* targetDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:1 height:1 mipmapped:NO];
            targetDescriptor.usage = MTLTextureUsageRenderTarget;
            targetDescriptor.storageMode = MTLStorageModePrivate;
            id<MTLTexture> target = [device newTextureWithDescriptor:targetDescriptor];
            const Variant variants[] = {{"sub-word (ushort first, uchar slot)", "ushort", "uchar"}, {"32-bit words", "uint", "uint"}};
            const std::uint32_t sizes[] = {256u, 255u, 256u};
            const int distinct[] = {97, 85, 129};
            int failures = 0;
            for (const auto& variant : variants) {
                for (std::size_t c = 0; c < std::size(sizes); ++c) {
                    const std::uint32_t used = sizes[c];
                    MTLCompileOptions* options = [MTLCompileOptions new];
                    options.languageVersion = MTLLanguageVersion4_0;
                    options.preprocessorMacros = @{@"N": @(used), @"FIRST_T": [NSString stringWithUTF8String:variant.firstType], @"SLOT_T": [NSString stringWithUTF8String:variant.slotType]};
                    NSError* error = nil;
                    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:Source] options:options error:&error];
                    Require(library != nil, std::string("the repro does not compile: ") + (error != nil ? error.localizedDescription.UTF8String : "no error"));
                    id<MTLComputePipelineState> compute = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"aps5_compute"] error:&error];
                    Require(compute != nil, "no compute pipeline");
                    MTLMeshRenderPipelineDescriptor* meshDescriptor = [MTLMeshRenderPipelineDescriptor new];
                    meshDescriptor.meshFunction = [library newFunctionWithName:@"aps5_mesh"];
                    meshDescriptor.fragmentFunction = [library newFunctionWithName:@"aps5_fragment"];
                    meshDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
                    id<MTLRenderPipelineState> mesh = [device newRenderPipelineStateWithMeshDescriptor:meshDescriptor options:MTLPipelineOptionNone reflection:nil error:&error];
                    Require(mesh != nil, std::string("no mesh pipeline: ") + (error != nil ? error.localizedDescription.UTF8String : "no error"));
                    std::vector<std::int32_t> input(Threads);
                    for (std::uint32_t i = 0; i < used; ++i) input[i] = static_cast<std::int32_t>((i * 7919u + c) % static_cast<std::uint32_t>(distinct[c]));
                    std::vector<std::uint32_t> owners(used);
                    std::uint32_t ownerCount = 0;
                    for (std::uint32_t t = 0; t < used; ++t) {
                        owners[t] = t;
                        for (std::uint32_t s = 0; s < t; ++s) {
                            if (input[s] == input[t]) {
                                owners[t] = s;
                                break;
                            }
                        }
                        if (owners[t] == t) ++ownerCount;
                    }
                    for (int stage = 0; stage < 2; ++stage) {
                        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data() length:input.size() * 4u options:MTLResourceStorageModeShared];
                        id<MTLBuffer> outputBuffer = [device newBufferWithLength:Threads * 12u options:MTLResourceStorageModeShared];
                        std::memset(outputBuffer.contents, 0xff, outputBuffer.length);
                        id<MTLCommandBuffer> commands = [queue commandBuffer];
                        if (stage == 0) {
                            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
                            [encoder setComputePipelineState:compute];
                            [encoder setBuffer:inputBuffer offset:0 atIndex:0];
                            [encoder setBuffer:outputBuffer offset:0 atIndex:1];
                            [encoder setBytes:&used length:4 atIndex:2];
                            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(Threads, 1, 1)];
                            [encoder endEncoding];
                        } else {
                            MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                            pass.colorAttachments[0].texture = target;
                            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                            id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
                            [encoder setRenderPipelineState:mesh];
                            [encoder setMeshBuffer:inputBuffer offset:0 atIndex:0];
                            [encoder setMeshBuffer:outputBuffer offset:0 atIndex:1];
                            [encoder setMeshBytes:&used length:4 atIndex:2];
                            [encoder drawMeshThreadgroups:MTLSizeMake(1, 1, 1) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(Threads, 1, 1)];
                            [encoder endEncoding];
                        }
                        [commands commit];
                        [commands waitUntilCompleted];
                        Require(commands.status == MTLCommandBufferStatusCompleted, "the repro dispatch failed");
                        const auto* out = static_cast<const std::uint32_t*>(outputBuffer.contents);
                        int cornerErrors = 0;
                        int ownerErrors = 0;
                        std::vector<int> seen(Threads, 0);
                        bool slotsOk = true;
                        for (std::uint32_t t = 0; t < used; ++t) {
                            if (out[t * 3] != static_cast<std::uint32_t>(input[t])) ++cornerErrors;
                            if (out[t * 3 + 1] != owners[t]) ++ownerErrors;
                            const auto slot = out[t * 3 + 2];
                            if (slot >= ownerCount) slotsOk = false;
                            else if (owners[t] == t && seen[slot]++ != 0) slotsOk = false;
                        }
                        const bool ok = cornerErrors == 0 && ownerErrors == 0 && slotsOk;
                        std::printf("subword threadgroup: %-36s %-7s %3u corners (%u owners): corners %s (%d wrong), owners %d wrong, slots %s\n", variant.name, stage == 0 ? "compute" : "mesh", used, ownerCount, cornerErrors == 0 ? "intact" : "CORRUPTED", cornerErrors, ownerErrors, slotsOk ? "ok" : "WRONG");
                        if (!ok) ++failures;
                    }
                }
            }
            if (failures != 0) {
                std::printf("subword threadgroup: %d of 12 runs differ from the CPU\n", failures);
                return 1;
            }
            std::puts("Subword threadgroup tests passed");
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
