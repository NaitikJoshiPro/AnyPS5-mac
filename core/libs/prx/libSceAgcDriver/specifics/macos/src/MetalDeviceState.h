#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALDEVICESTATE_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALDEVICESTATE_H

// The Metal device's state, shared by the Objective-C++ files that implement the VulkanDevice API
// on macOS (MetalDevice.mm: device, submission, compute, labels, fills, copies;
// MetalDeviceProfile.mm: startup self-tests; MetalPresentation.mm; MetalDraw.mm).
// Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace AgcDriver {

namespace Metal {

[[noreturn]] inline void Unimplemented(const char* what, const char* package) {
    throw std::runtime_error(std::string("Metal device: ") + what + " is not implemented yet (" + package + ", docs/mac/M2_DESIGN.md)");
}

inline std::string Describe(NSError* error) {
    return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

// What the startup self-tests measured on this GPU (MetalDeviceProfile.mm), once per process. The
// facts are undocumented by Apple or differ from the Vulkan model the front end assumes, so they
// are measured rather than derived from the GPU family (docs/mac/M2_DESIGN.md §5, metal-limits.md).
struct DeviceProfile {
    // The self-tests ran to completion (false: a test failed to build or run, see `failure`).
    bool measured = false;
    std::string failure;
    // SIMD-group functions (simd_sum, simd_ballot, simd_prefix_exclusive_sum, simd_shuffle,
    // simd_max) in a vertex function: they compile and gave consistent 32-wide groups of
    // consecutive vertex ids. VulkanDevice::SubgroupStages() reports the vertex stage from this.
    bool vertexSimdGroups = false;
    std::uint32_t vertexSimdWidth = 0;
    // [[barycentric_coord]] runs and weights the vertices of a triangle list in order (v0, v1, v2).
    bool barycentrics = false;
    // The vertex order the barycentric weights of an odd triangle i of a triangle strip follow.
    // I_I2_I1: (v_i, v_i+2, v_i+1), the Vulkan strip order (triangle i = (i, i+1+i%2, i+2-i%2)).
    // I1_I_I2: (v_i+1, v_i, v_i+2). Other: neither (the backend must not rely on strip order).
    enum class StripOrder : std::uint32_t { Unknown, I_I2_I1, I1_I_I2, Other };
    StripOrder oddStripOrder = StripOrder::Unknown;
    // One line per self-test, for logs and the [metal] profile line.
    std::string summary;
};

// Runs the self-tests on `device` the first time it is called in the process (a few tiny GPU
// draws, each waited for with a timeout) and returns the cached result afterwards. Never throws:
// a failure is recorded in the profile and every fact stays at its conservative default.
const DeviceProfile& Profile(id<MTLDevice> device);

}

struct VulkanDevice::State {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTLSharedEvent> event = nil;
    id<MTLResidencySet> residency = nil;
    id<MTL4ArgumentTable> arguments = nil;
    // Transient upload ring (descriptor tables, push data, DriverData, BDA tables, label bytes),
    // linear, reset whenever the device is idle.
    id<MTLBuffer> ring = nil;
    // The BdaAbi fault buffer: fault record + written-page record, shared by every dispatch, checked
    // and reset whenever the device goes idle (CheckFaults).
    id<MTLBuffer> faults = nil;
    // Device-memory LDS arena (MslAbi::BufferIndex::WorkgroupMemory), grown in 1 MiB granules
    // while idle; its lock dwords start at zero.
    id<MTLBuffer> workgroupArena = nil;
    std::size_t ringUsed = 0;
    // Upload buffers of the command being built that did not fit the ring (Commit hands them to
    // its batch).
    std::vector<id<MTLBuffer>> uploads;
    std::uint64_t submitted = 0;
    std::vector<std::uint32_t> capabilities;
    std::vector<std::string_view> extensions;

    // A committed command buffer: its event value, its allocator, what it keeps alive (command
    // buffers retain nothing in Metal 4: upload buffers beyond the ring, the registry lease of its
    // BDA page table) and the completion actions run when it retires.
    struct Batch {
        std::uint64_t value;
        id<MTL4CommandAllocator> allocator;
        std::vector<id<MTLBuffer>> uploads;
        GuestAllocations::Lease lease;
        std::vector<std::function<void()>> completions;
    };
    std::deque<Batch> inflight;
    std::vector<id<MTL4CommandAllocator>> freeAllocators;
    // Command buffer errors reported by commit feedback, which may arrive after the device is gone.
    struct Errors {
        std::mutex mutex;
        std::vector<std::string> messages;
    };
    std::shared_ptr<Errors> errors = std::make_shared<Errors>();

    // Guest memory wrapped in place (newBufferWithBytesNoCopy over 16 KiB pages), dropped whenever
    // the device is idle; `wrapperEpoch` counts the drops (cached GPU addresses expire with it).
    struct Wrapper {
        std::uint64_t begin;
        std::uint64_t end;
        id<MTLBuffer> buffer;
    };
    std::vector<Wrapper> wrappers;
    std::uint64_t wrapperEpoch = 0;

    // The interim BDA page table (BdaAbi::Header + Range[]) built from the GuestAllocations
    // registry, cached while neither the registry generation nor the wrapper epoch changed.
    struct AddressTable {
        std::uint64_t generation = ~std::uint64_t{0};
        std::uint64_t wrapperEpoch = ~std::uint64_t{0};
        std::vector<std::byte> bytes;
        GuestAllocations::Lease lease;
        std::size_t skipped = 0;
    } addressTable;

    std::mutex pipelinesMutex;
    std::unordered_map<std::string, id<MTLLibrary>> libraries;
    std::unordered_map<std::string, id<MTLComputePipelineState>> computePipelines;
    // The driver's own kernels (fills, copies), compiled on first use.
    id<MTLLibrary> helpers = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> helperPipelines;

    // An open command buffer with its compute encoder, ordered after all earlier work on the queue.
    struct Encoding {
        id<MTL4CommandAllocator> allocator = nil;
        id<MTL4CommandBuffer> commands = nil;
        id<MTL4ComputeCommandEncoder> encoder = nil;
    };

    // Upload memory for the command being built: its GPU address, CPU pointer, buffer and offset.
    struct Upload {
        std::uint64_t address;
        void* host;
        id<MTLBuffer> buffer;
        std::uint64_t offset;
    };
    Upload Stage(std::size_t bytes, std::size_t alignment);
    std::uint64_t Allocate(std::size_t bytes, std::size_t alignment, void** host);
    // Makes room for `bytes` of ring allocations before anything of a command is wrapped or
    // allocated (a full ring waits for idle, which drops wrappers).
    void Reserve(std::size_t bytes);
    std::uint64_t Wrap(std::uint64_t address, std::uint64_t bytes);
    // Wrap without throwing: 0 when the range cannot be wrapped (unmapped, executable, unreadable).
    std::uint64_t TryWrap(std::uint64_t address, std::uint64_t bytes);
    // The wrapper holding [address, address + bytes) and the offset of `address` in it (after Wrap).
    id<MTLBuffer> WrapperOf(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset) const;
    id<MTLLibrary> Library(const ShaderRecompiler::MslShader& shader);
    id<MTLComputePipelineState> ComputePipeline(const ShaderRecompiler::RecompileResult& shader);
    id<MTLComputePipelineState> HelperPipeline(const char* name);
    // The device-memory LDS arena for `groups` workgroups of `strideDwords` each.
    id<MTLBuffer> WorkgroupArena(std::uint64_t groups, std::uint32_t strideDwords);
    // The BDA page table for a dispatch: its GPU address and size; `lease` receives the registry
    // lease the table's wrappers rely on (kept by the batch).
    std::uint64_t BuildAddressTable(std::uint32_t* bytes, GuestAllocations::Lease* lease);
    // Compute dispatches (MetalDevice.mm): the checks every dispatch makes, and the recording of
    // one grid (split over the group limit) or, with `indirect` (a guest address), of a dispatch
    // whose counts the GPU reads.
    void RequireDispatchable(const ShaderRecompiler::RecompileResult& shader) const;
    void RecordCompute(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint64_t indirect);
    Encoding Begin();
    // Ends the encoding and commits it; returns the event value that signals its completion.
    std::uint64_t Commit(Encoding& encoding, Batch batch);
    void Retire(bool wait);
    void CheckFaults();
};

}

#endif
