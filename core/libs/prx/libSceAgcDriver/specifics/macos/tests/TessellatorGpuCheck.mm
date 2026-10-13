// agc_metal_tessellator_gpu (DECISIONS D26): the software tessellator core (MslTessellatorCore.inc)
// compiled as MSL with the device's math mode and run on the GPU must give exactly what the same
// core gives compiled as C++ on the CPU (which agc_metal_tessellator checks against Microsoft's D3D11
// reference): per factor set, the primitive and point counts, every slice window's corner indices
// (windows as the tessellation mesh function takes them) and every point's (u, v) bits. One thread
// per factor set, one dispatch; window output goes to device memory instead of threadgroup memory.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalBackend/MslTessellator.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace T = ShaderRecompiler::MslTessellator;

constexpr int Skipped = 77;
constexpr int MaxCorners = 3 * 1400;
constexpr int MaxPoints = 640;
constexpr int Slices = 24;

struct Case {
    std::uint32_t domain;
    std::uint32_t partitioning;
    std::uint32_t topology;
    std::uint32_t pad;
    float outer[4];
    float inner[2];
    float pad2[2];
};
static_assert(sizeof(Case) == 48);

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr const char* Kernel = R"(
struct Aps5TessCase {
    uint domain;
    uint partitioning;
    uint topology;
    uint pad;
    float outer[4];
    float inner[2];
    float pad2[2];
};

kernel void aps5_tess_gpu(device const Aps5TessCase* cases [[buffer(0)]], device int* corners [[buffer(1)]], device uint* points [[buffer(2)]], device int* counts [[buffer(3)]], constant uint& total [[buffer(4)]], uint id [[thread_position_in_grid]])
{
    if (id >= total) return;
    const Aps5TessCase c = cases[id];
    Aps5TessFactors f;
    for (uint i = 0u; i < 4u; ++i) f.outer[i] = c.outer[i];
    for (uint i = 0u; i < 2u; ++i) f.inner[i] = c.inner[i];
    const Aps5TessPatch p = aps5_tess_process(int(c.domain), int(c.partitioning), int(c.topology), f);
    device int* out = corners + id * MAX_CORNERS;
    const int n = aps5_tess_primitive_count(p, out);
    const int pointCount = p.culled != 0 ? 0 : p.points;
    counts[id * 2u] = n;
    counts[id * 2u + 1u] = pointCount;
    const int slice = c.topology == 0u ? 256 : c.topology == 1u ? 128 : 85;
    const int k = c.topology == 0u ? 1 : c.topology == 1u ? 2 : 3;
    for (int first = 0; first < n; first += slice) {
        if ((first + min(slice, n - first)) * k > MAX_CORNERS) break;
        aps5_tess_window(p, first, min(slice, n - first), out + first * k);
    }
    for (int i = 0; i < pointCount && i < MAX_POINTS; ++i) {
        const Aps5TessPoint point = aps5_tess_point(p, i);
        points[(id * MAX_POINTS + uint(i)) * 2u] = as_type<uint>(point.u);
        points[(id * MAX_POINTS + uint(i)) * 2u + 1u] = as_type<uint>(point.v);
    }
}

)";

constexpr const char* SlicedKernel = R"(
// As the tessellation mesh function: one 256-thread threadgroup per (factor set, slice); thread 0
// windows the slice into threadgroup memory, then every corner thread places its point.
kernel void aps5_tess_gpu_sliced(device const Aps5TessCase* cases [[buffer(0)]], device uint* corners [[buffer(1)]], constant uint& total [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup int window[255];
    threadgroup uint used[1];
    threadgroup int none[1];
    const uint id = group.x;
    if (id >= total) return;
    const Aps5TessCase c = cases[id];
    Aps5TessFactors f;
    for (uint i = 0u; i < 4u; ++i) f.outer[i] = c.outer[i];
    for (uint i = 0u; i < 2u; ++i) f.inner[i] = c.inner[i];
    const Aps5TessPatch p = aps5_tess_process(int(c.domain), int(c.partitioning), int(c.topology), f);
    const int slice = c.topology == 0u ? 256 : c.topology == 1u ? 128 : 85;
    const int k = c.topology == 0u ? 1 : c.topology == 1u ? 2 : 3;
    if (tid == 0u) {
        const int n = aps5_tess_primitive_count(p, none);
        const int first = int(group.y) * slice;
        used[0] = first < n ? uint(aps5_tess_window(p, first, min(slice, n - first), window) * k) : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint at = (id * SLICES + group.y) * 256u + tid;
    if (tid < used[0]) {
        const Aps5TessPoint point = aps5_tess_point(p, window[tid]);
        corners[at * 3u] = uint(window[tid]);
        corners[at * 3u + 1u] = as_type<uint>(point.u);
        corners[at * 3u + 2u] = as_type<uint>(point.v);
    }
}
)";

std::vector<Case> Cases() {
    std::vector<Case> cases;
    const auto add = [&](std::uint32_t domain, std::uint32_t partitioning, std::uint32_t topology, std::array<float, 4> outer, std::array<float, 2> inner) {
        Case c{};
        c.domain = domain;
        c.partitioning = partitioning;
        c.topology = topology;
        std::copy(outer.begin(), outer.end(), c.outer);
        std::copy(inner.begin(), inner.end(), c.inner);
        cases.push_back(c);
    };
    // The agc_metal_tessellation_domains multi-slice quad first.
    add(T::Quad, T::FractionalOdd, T::TriangleCw, {9.5f, 8.2f, 7.7f, 10.1f}, {9.9f, 8.8f});
    std::mt19937 random(0x7e552u);
    std::uniform_real_distribution<float> level(0.5f, 16.0f);
    std::uniform_int_distribution<int> special(0, 15);
    const auto factor = [&] {
        switch (special(random)) {
        case 0: return 0.0f;
        case 1: return std::numeric_limits<float>::quiet_NaN();
        case 2: return std::floor(level(random));
        default: return level(random);
        }
    };
    for (int round = 0; round < 640; ++round) {
        const auto domain = static_cast<std::uint32_t>(round % 3);
        const auto partitioning = static_cast<std::uint32_t>((round / 3) % 4);
        const auto topology = domain == T::Isoline ? static_cast<std::uint32_t>((round / 12) % 2) : static_cast<std::uint32_t>((round / 12) % 4);
        add(domain, partitioning, topology, {factor(), factor(), factor(), factor()}, {factor(), factor()});
    }
    return cases;
}

}

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                std::puts("tessellator gpu: skipped, no Metal device");
                return Skipped;
            }
            std::string source = T::Source();
            const std::string threadgroupOut = "#define APS5_TESS_OUT threadgroup\n";
            const auto at = source.find(threadgroupOut);
            Require(at != std::string::npos, "the tessellator source lost its APS5_TESS_OUT definition");
            source.replace(at, threadgroupOut.size(), "#define APS5_TESS_OUT device\n");
            source = "#include <metal_stdlib>\nusing namespace metal;\n#define MAX_CORNERS " + std::to_string(MaxCorners) + "\n#define MAX_POINTS " + std::to_string(MaxPoints) + "\n#define SLICES " + std::to_string(Slices) + "\n" + source + Kernel;
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion4_0;
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
            Require(library != nil, std::string("the tessellator kernel does not compile: ") + (error != nil ? error.localizedDescription.UTF8String : "no error"));
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"aps5_tess_gpu"] error:&error];
            Require(pipeline != nil, "no tessellator kernel pipeline");
            const auto cases = Cases();
            const auto total = static_cast<std::uint32_t>(cases.size());
            id<MTLBuffer> caseBuffer = [device newBufferWithBytes:cases.data() length:cases.size() * sizeof(Case) options:MTLResourceStorageModeShared];
            id<MTLBuffer> cornerBuffer = [device newBufferWithLength:std::size_t(total) * MaxCorners * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> pointBuffer = [device newBufferWithLength:std::size_t(total) * MaxPoints * 8u options:MTLResourceStorageModeShared];
            id<MTLBuffer> countBuffer = [device newBufferWithLength:std::size_t(total) * 8u options:MTLResourceStorageModeShared];
            std::memset(cornerBuffer.contents, 0xff, cornerBuffer.length);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:caseBuffer offset:0 atIndex:0];
            [encoder setBuffer:cornerBuffer offset:0 atIndex:1];
            [encoder setBuffer:pointBuffer offset:0 atIndex:2];
            [encoder setBuffer:countBuffer offset:0 atIndex:3];
            [encoder setBytes:&total length:sizeof(total) atIndex:4];
            [encoder dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            Require(commands.status == MTLCommandBufferStatusCompleted, "the tessellator kernel failed");
            const auto* gpuCorners = static_cast<const std::int32_t*>(cornerBuffer.contents);
            const auto* gpuPoints = static_cast<const std::uint32_t*>(pointBuffer.contents);
            const auto* gpuCounts = static_cast<const std::int32_t*>(countBuffer.contents);
            std::uint64_t checkedCorners = 0;
            std::uint64_t checkedPoints = 0;
            int mismatches = 0;
            std::vector<int> window(MaxCorners);
            for (std::uint32_t id = 0; id < total; ++id) {
                const auto& c = cases[id];
                const T::Aps5TessFactors f{{c.outer[0], c.outer[1], c.outer[2], c.outer[3]}, {c.inner[0], c.inner[1]}};
                const auto patch = T::aps5_tess_process(static_cast<int>(c.domain), static_cast<int>(c.partitioning), static_cast<int>(c.topology), f);
                const int n = T::aps5_tess_primitive_count(patch, nullptr);
                const int pointCount = patch.culled != 0 ? 0 : patch.points;
                char what[160];
                std::snprintf(what, sizeof(what), "case %u (domain %u partitioning %u topology %u, outer %g %g %g %g inner %g %g)", id, c.domain, c.partitioning, c.topology, c.outer[0], c.outer[1], c.outer[2], c.outer[3], c.inner[0], c.inner[1]);
                const auto report = [&](const std::string& message) {
                    if (++mismatches <= 12) std::cerr << what << ": " << message << '\n';
                };
                if (gpuCounts[id * 2] != n || gpuCounts[id * 2 + 1] != pointCount) {
                    report("counts GPU " + std::to_string(gpuCounts[id * 2]) + "/" + std::to_string(gpuCounts[id * 2 + 1]) + ", CPU " + std::to_string(n) + "/" + std::to_string(pointCount));
                    continue;
                }
                const int k = c.topology == 0 ? 1 : c.topology == 1 ? 2 : 3;
                const int slice = T::SlicePrimitives(static_cast<int>(c.topology));
                for (int first = 0; first < n; first += slice) {
                    const int count = std::min(slice, n - first);
                    if ((first + count) * k > MaxCorners) break;
                    T::aps5_tess_window(patch, first, count, window.data());
                    for (int i = 0; i < count * k; ++i) {
                        ++checkedCorners;
                        const int gpu = gpuCorners[std::size_t(id) * MaxCorners + std::size_t(first * k + i)];
                        if (gpu != window[std::size_t(i)]) {
                            report("slice " + std::to_string(first / slice) + " corner " + std::to_string(i) + ": GPU " + std::to_string(gpu) + ", CPU " + std::to_string(window[std::size_t(i)]));
                            break;
                        }
                    }
                }
                for (int i = 0; i < pointCount && i < MaxPoints; ++i) {
                    ++checkedPoints;
                    const auto point = T::aps5_tess_point(patch, i);
                    const auto u = gpuPoints[(std::size_t(id) * MaxPoints + std::size_t(i)) * 2];
                    const auto v = gpuPoints[(std::size_t(id) * MaxPoints + std::size_t(i)) * 2 + 1];
                    if (u != std::bit_cast<std::uint32_t>(point.u) || v != std::bit_cast<std::uint32_t>(point.v)) {
                        report("point " + std::to_string(i) + ": GPU (" + std::to_string(std::bit_cast<float>(u)) + ", " + std::to_string(std::bit_cast<float>(v)) + "), CPU (" + std::to_string(point.u) + ", " + std::to_string(point.v) + ")");
                        break;
                    }
                }
            }
            // The mesh-function structure: threadgroup windows, one threadgroup per slice.
            const std::string slicedSource = "#include <metal_stdlib>\nusing namespace metal;\n#define SLICES " + std::to_string(Slices) + "\n" + T::Source() + SlicedKernel;
            id<MTLLibrary> slicedLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:slicedSource.c_str()] options:options error:&error];
            Require(slicedLibrary != nil, std::string("the sliced tessellator kernel does not compile: ") + (error != nil ? error.localizedDescription.UTF8String : "no error"));
            id<MTLComputePipelineState> sliced = [device newComputePipelineStateWithFunction:[slicedLibrary newFunctionWithName:@"aps5_tess_gpu_sliced"] error:&error];
            Require(sliced != nil, "no sliced tessellator kernel pipeline");
            id<MTLBuffer> slicedBuffer = [device newBufferWithLength:std::size_t(total) * Slices * 256u * 12u options:MTLResourceStorageModeShared];
            std::memset(slicedBuffer.contents, 0xff, slicedBuffer.length);
            commands = [queue commandBuffer];
            encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:sliced];
            [encoder setBuffer:caseBuffer offset:0 atIndex:0];
            [encoder setBuffer:slicedBuffer offset:0 atIndex:1];
            [encoder setBytes:&total length:sizeof(total) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(total, Slices, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            Require(commands.status == MTLCommandBufferStatusCompleted, "the sliced tessellator kernel failed");
            const auto* slicedWords = static_cast<const std::uint32_t*>(slicedBuffer.contents);
            std::uint64_t slicedCorners = 0;
            for (std::uint32_t id = 0; id < total; ++id) {
                const auto& c = cases[id];
                const T::Aps5TessFactors f{{c.outer[0], c.outer[1], c.outer[2], c.outer[3]}, {c.inner[0], c.inner[1]}};
                const auto patch = T::aps5_tess_process(static_cast<int>(c.domain), static_cast<int>(c.partitioning), static_cast<int>(c.topology), f);
                const int n = T::aps5_tess_primitive_count(patch, nullptr);
                const int k = c.topology == 0 ? 1 : c.topology == 1 ? 2 : 3;
                const int slice = T::SlicePrimitives(static_cast<int>(c.topology));
                for (int s = 0; s < Slices && s * slice < n; ++s) {
                    const int count = T::aps5_tess_window(patch, s * slice, std::min(slice, n - s * slice), window.data());
                    for (int t = 0; t < count * k; ++t) {
                        ++slicedCorners;
                        const auto* word = &slicedWords[((std::size_t(id) * Slices + std::size_t(s)) * 256u + std::size_t(t)) * 3u];
                        const auto point = T::aps5_tess_point(patch, window[std::size_t(t)]);
                        if (static_cast<int>(word[0]) != window[std::size_t(t)] || word[1] != std::bit_cast<std::uint32_t>(point.u) || word[2] != std::bit_cast<std::uint32_t>(point.v)) {
                            if (++mismatches <= 12) std::cerr << "sliced case " << id << " slice " << s << " corner " << t << ": GPU index " << static_cast<int>(word[0]) << " (" << std::bit_cast<float>(word[1]) << ", " << std::bit_cast<float>(word[2]) << "), CPU index " << window[std::size_t(t)] << " (" << point.u << ", " << point.v << ")\n";
                            break;
                        }
                    }
                }
            }
            std::printf("tessellator gpu: %llu sliced corners compared\n", static_cast<unsigned long long>(slicedCorners));
            std::printf("tessellator gpu: %u factor sets, %llu corners, %llu points compared, %d mismatching sets\n", total, static_cast<unsigned long long>(checkedCorners), static_cast<unsigned long long>(checkedPoints), mismatches);
            Require(mismatches == 0, "the GPU tessellator differs from the CPU tessellator");
            std::puts("Tessellator GPU tests passed");
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
