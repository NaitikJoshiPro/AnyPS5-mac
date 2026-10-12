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

// What the last device teardown did (DECISIONS D19.5), for the teardown tests and logs: the device
// waited for every recorded batch, joined the release thread (through ~Recorder), removed its own
// residency set from the queue and dropped its guest-memory wrappers (static __DATA pages included).
struct TeardownReport {
    std::uint64_t teardowns = 0;
    // The recorder was idle (every submitted batch completed) before anything was released.
    bool idle = false;
    std::uint32_t syncFailures = 0;
    std::uint64_t wrappersDropped = 0;
    bool residencyRemoved = false;
    // The teardown ran from the exit handler (a device still alive at exit()).
    bool atExit = false;
};
TeardownReport LastTeardown();

}

struct VulkanDevice::State {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    id<MTL4Compiler> compiler = nil;
    // Long-lived resources the device's own commands use (guest-memory wrappers, the LDS arena, the
    // fault buffer), attached to the queue.
    id<MTLResidencySet> residency = nil;
    // The argument table of the device's dispatches (bindings are snapshotted per dispatch).
    id<MTL4ArgumentTable> arguments = nil;
    // The BdaAbi fault buffer: fault record + written-page record, shared by every dispatch, checked
    // and reset when a batch with a dispatch completes (CheckFaults).
    id<MTLBuffer> faults = nil;
    // MslAbi v2: the 64-bit atomic lock table (AtomicLockTableBytes, zero-filled, GPU-only after setup).
    id<MTLBuffer> atomicLocks = nil;
    // Device-memory LDS arena (MslAbi::BufferIndex::WorkgroupMemory), grown in 1 MiB granules
    // while idle; its lock dwords start at zero.
    id<MTLBuffer> workgroupArena = nil;
    std::vector<std::uint32_t> capabilities;
    std::vector<std::string_view> extensions;

    // D2's recorder (specifics/macos/src/Recorder.mm): every command of the device records into its
    // open batch (a MTL4 command buffer), which it submits, waits for and completes; its pending
    // writes back the GuestMemory flush hook, its label table WAIT_REG_MEM. The Context carries the
    // Metal device and queue in its Vulkan-typed fields (D-M2-3) for the twins of D2-D6.
    Graphics::Context context {};
    std::unique_ptr<Graphics::Recorder> recorder;
    // The device is being destroyed: completion actions log faults instead of throwing.
    bool tearingDown = false;
    // Teardown() ran (from ~VulkanDevice or from the exit handler, whichever came first).
    bool tornDown = false;
    // Releases everything in the order the GPU driver needs (D19.5): waits for every submitted
    // batch, destroys the recorder (joins the release thread, removes its residency set), drops the
    // wrappers, removes the device's residency set from the queue, then the buffer pool's.
    // Idempotent, never throws.
    void Teardown(bool atExit) noexcept;

    // Guest memory wrapped in place (newBufferWithBytesNoCopy over 16 KiB pages), dropped only when
    // the recorder is idle (batches read them, and Metal 4 command buffers retain nothing);
    // `wrapperEpoch` counts the drops (cached GPU addresses expire with it).
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
        std::shared_ptr<const GuestAllocations::Lease> lease;
        // The writable ranges (a dispatch binding the table may store anywhere in them).
        std::vector<std::pair<std::uint64_t, std::uint64_t>> writable;
        std::size_t skipped = 0;
    } addressTable;

    std::mutex pipelinesMutex;
    std::unordered_map<std::string, id<MTLLibrary>> libraries;
    std::unordered_map<std::string, id<MTLComputePipelineState>> computePipelines;

    std::uint64_t Wrap(std::uint64_t address, std::uint64_t bytes);
    // Wrap without throwing: 0 when the range cannot be wrapped (unmapped, executable, unreadable).
    std::uint64_t TryWrap(std::uint64_t address, std::uint64_t bytes);
    // TryWrap limited to registered guest allocations (upstream HostImportFor): labels, fills and
    // copies into other memory take the PM4 core's drain-and-CPU path.
    std::uint64_t TryWrapRegistered(std::uint64_t address, std::uint64_t bytes);
    // The wrapper holding [address, address + bytes) and the offset of `address` in it (after Wrap).
    id<MTLBuffer> WrapperOf(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset) const;
    // Drops the wrappers (and the cached BDA table) once the recorder is idle.
    void DropWrappersIfIdle();
    id<MTLLibrary> Library(const ShaderRecompiler::MslShader& shader);
    id<MTLComputePipelineState> ComputePipeline(const ShaderRecompiler::RecompileResult& shader);
    // The device-memory LDS arena for `groups` workgroups of `strideDwords` each.
    id<MTLBuffer> WorkgroupArena(std::uint64_t groups, std::uint32_t strideDwords);
    // The BDA page table bytes for a dispatch; `lease` receives the registry lease its wrappers
    // rely on (kept by the batch).
    const std::vector<std::byte>& AddressTableBytes(std::shared_ptr<const GuestAllocations::Lease>* lease);
    // Compute dispatches (MetalDevice.mm): the checks every dispatch makes, and the recording of
    // one grid (split over the group limit) or, with `indirect` (a guest address), of a dispatch
    // whose counts the GPU reads.
    void RequireDispatchable(const ShaderRecompiler::RecompileResult& shader) const;
    void RecordCompute(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint64_t indirect);
    // Waits for every recorded batch and runs their completions (VulkanDevice::WaitIdle).
    void Sync();
    void CheckFaults();
};

}

#endif
