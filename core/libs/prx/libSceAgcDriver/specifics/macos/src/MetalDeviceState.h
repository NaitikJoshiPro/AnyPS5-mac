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
#include "BdaAbi.hpp"
#include <array>
#include <span>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <utility>

namespace AgcDriver::Graphics {
class StorageTexture;
}

namespace AgcDriver::Graphics::Metal {
struct TessellationStages;
class CommandStream;
class PipelineCache;
class ImageCache;
class ImageBindings;
// Readies guest memory [address, address + bytes) for GPU work this thread records into the open
// batch next (Recorder.mm, D66): queued labels and storage image stores over it are recorded, and
// no CPU wait is made, since recorded GPU writes precede the reader in queue order. Returns false
// when a pending CPU write-back made it wait as a CPU read does (the read is still ordered).
bool FlushForGpuRead(std::uint64_t address, std::size_t bytes);
// {reads ordered by the queue alone, reads that waited} since start (tests, [metal] trace).
std::pair<std::uint64_t, std::uint64_t> GpuReadCounts();
}

namespace AgcDriver {

namespace Metal {

// D6's per-device draw objects (MetalDraw.mm) and presentation state (MetalPresentation.mm).
struct DrawCache;
struct Presenter;

// The last frame a headless present copied back (APS5_METAL_PRESENT_READBACK=1, for tests):
// BGRA8 rows of the presentation target, `frames` counts the copies so far.
struct PresentedFrame {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::byte> pixels;
    std::uint64_t frames = 0;
};
PresentedFrame LastPresentedFrame();
// Presents the MetalFX spatial scaler upscaled (D61).
std::uint64_t MetalFxPresents();
// Display buffers decoded on the GPU from their host import (D66), and those decoded on the CPU.
std::uint64_t GpuDecodedPresents();
std::uint64_t CpuDecodedPresents();

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

// The BDA page table's ranges made disjoint and sorted by begin, as aps5_bda_find's binary search
// needs: every address covered by any input range is covered by exactly one output range, whose
// device address is that of the first input range (in input order) covering it and whose
// permissions are the OR of every input range covering it; adjacent pieces of one input range
// with equal permissions are joined. Empty input ranges are dropped.
std::vector<ShaderRecompiler::BdaAbi::Range> DisjointRanges(std::span<const ShaderRecompiler::BdaAbi::Range> ranges);

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
    // The first BDA fault of a draw's graphics stage since the last WaitIdle (upstream throws it out
    // of its synchronous Draw; recorded Metal draws surface it at WaitIdle). Dispatch faults stay
    // logged only, as upstream's recorded dispatches.
    std::mutex drawFaultMutex;
    std::exception_ptr drawFault;
    std::uint32_t lastFaultStage = 0;
    // Upstream's BdaResources::markWrittenPages check on the device's one written-page record: the
    // image table ranges of the commands of each batch (keyed by the recorder's serial of the batch
    // they were recorded into), consumed by the batch's first completion. A store of the batch that
    // overlaps one is "written by its own dispatch". The record is batch-wide, so a later command
    // of the same batch storing over a table an earlier one read is reported too (upstream checks
    // per build): docs/dev/TechnicalDebt.md#metal-image-table-self-write-batch.
    std::mutex tableRangeMutex;
    std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> batchTableRanges;
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
        // The registered ranges of `bytes` (sorted, disjoint), the base of per-dispatch tables.
        std::vector<ShaderRecompiler::BdaAbi::Range> ranges;
        // Weak: in-flight batches hold the lease (recorder Keep); a cached one would pin the registry
        // and block guest mutations (GuestBufferMemory's pin waiter) while the device is idle.
        std::weak_ptr<const GuestAllocations::Lease> lease;
        // The writable ranges (a dispatch binding the table may store anywhere in them).
        std::vector<std::pair<std::uint64_t, std::uint64_t>> writable;
        std::size_t skipped = 0;
        // Bytes of long allocations the host-import budget left out (D52).
        std::uint64_t refusedBytes = 0;
    } addressTable;

    // D5's shader library and pipeline cache (MTL4Compiler + MTL4Archive, background builds); it
    // owns `compiler`. Destroyed in Teardown after the recorder is idle (joins its workers, saves).
    std::unique_ptr<Graphics::Metal::PipelineCache> pipelines;
    // The process-wide guest compute pipeline count when `pipelines` was created: the base of
    // VulkanDevice::ComputePipelinesCreated (upstream 4aafda8e3 counts per device).
    std::uint64_t computePipelinesAtStart = 0;
    // D4/D5's image cache: textures and samplers of dispatch bindings, its residency set on the queue.
    std::unique_ptr<Graphics::Metal::ImageCache> images;
    // D6's draw objects (MetalDraw.mm): argument tables, depth surfaces, tiled target textures.
    // Destroyed in Teardown after the recorder is idle, before `images` (whose residency set holds
    // their textures).
    std::shared_ptr<Metal::DrawCache> draws;
    // D6's presentation (null without a window): the window's CAMetalLayer or headless targets,
    // the scaler, the present slots. Destroyed in Teardown with `draws`.
    std::shared_ptr<Metal::Presenter> presenter;
    void AttachPresentation(const PresentationWindow& window);

    std::uint64_t Wrap(std::uint64_t address, std::uint64_t bytes);
    // Wrap, or for a range the command only reads that cannot be wrapped (executable pages), a copy in
    // the stream's upload memory made now (D3's copy fallback).
    std::uint64_t WrapOrCopy(std::uint64_t address, std::uint64_t bytes, Graphics::Metal::CommandStream& stream, bool written);
    // Wrap without throwing: 0 when the range cannot be wrapped (unmapped, executable, unreadable).
    std::uint64_t TryWrap(std::uint64_t address, std::uint64_t bytes);
    // The host import holding [address, address + bytes) (D3, GuestBufferMemory.mm), nil if none.
    id<MTLBuffer> ImportOf(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset) const;
    // The wrapper holding [address, address + bytes) and the offset of `address` in it (after Wrap).
    id<MTLBuffer> WrapperOf(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset) const;
    // Charges the page wrapper holding a range (unregistered memory) to the open batch's kept bytes.
    void KeepWrapped(std::uint64_t address, std::uint64_t bytes);
    // Drops the wrappers (and the cached BDA table) once the recorder is idle.
    void DropWrappersIfIdle();
    // The device-memory LDS arena for `groups` workgroups of `strideDwords` each.
    id<MTLBuffer> WorkgroupArena(std::uint64_t groups, std::uint32_t strideDwords);
    // The BDA page table bytes for a dispatch; `lease` receives the registry lease its wrappers
    // rely on (kept by the batch).
    const std::vector<std::byte>& AddressTableBytes(std::shared_ptr<const GuestAllocations::Lease>* lease);
    // Compute dispatches (MetalDevice.mm): the checks every dispatch makes, and the recording of
    // one grid (split over the group limit) or, with `indirect` (a guest address), of a dispatch
    // whose counts the GPU reads.
    void RequireDispatchable(const ShaderRecompiler::RecompileResult& shader) const;
    // The bound V# ranges, GDS and snapshots a dispatch adds to the BDA page table.
    std::vector<ShaderRecompiler::BdaAbi::Range> AddressExtras(const ShaderRecompiler::RecompileResult& shader, std::span<const Graphics::GuestMemorySnapshot> snapshots, Graphics::Metal::CommandStream& stream);
    void RecordCompute(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint64_t indirect, std::span<const Graphics::GuestMemorySnapshot> snapshots = {});

    // The MslAbi bindings of one shader function (a kernel, or one stage of a draw), built around
    // taking the open batch's stream: before it PlanBindings (guest ranges), SettleBindings, PlanImages
    // (image and sampler table entries, which may upload surfaces and sync) and WrapUnions; after it
    // WriteTable (the descriptor table in the stream's upload memory).
    struct StageBindings {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
        bool addressBased = false;
        std::vector<std::pair<std::uint32_t, std::vector<std::uint64_t>>> imageEntries;
        // MslAbi Version 15: the image table, sampler table and table map arrays (WriteTable uploads
        // each and points its table entries at it).
        std::vector<std::pair<ShaderRecompiler::RuntimeAbi::Binding, std::vector<std::uint64_t>>> tableArrays;
    };
    void PlanBindings(const ShaderRecompiler::RecompileResult& shader, StageBindings& plan);
    void PlanImages(const ShaderRecompiler::RecompileResult& shader, Graphics::Metal::ImageBindings& images, StageBindings& plan);
    // Flushes the recorder's queued label and key stores the planned ranges touch (they must precede
    // the command).
    void SettleBindings(std::span<const StageBindings* const> plans);
    // Wraps overlapping guest ranges as their union (one mapping per array), after PlanImages (whose
    // syncs may drop the wrappers).
    void WrapUnions(std::span<const StageBindings* const> plans);
    // Writes the shader's descriptor table into the stream (0: the shader has none). `push` is the
    // shader's own push constants inside the 256-byte block (the memory-offset bytes of bound V#s
    // are zeroed there, as the guest-buffer contract says).
    std::uint64_t WriteTable(const ShaderRecompiler::RecompileResult& shader, std::span<const Graphics::GuestMemorySnapshot> snapshots, Graphics::Metal::CommandStream& stream, StageBindings& plan, std::span<std::byte> push);
    // The DriverData record of a function of the batch (workgroupBase as given).
    std::uint64_t WriteDriverData(Graphics::Metal::CommandStream& stream, std::array<std::uint32_t, 3> workgroupBase);
    // After the command was recorded: pending reads and writes of the planned ranges, and the
    // fault check of the batch.
    void NoteBindings(const StageBindings& plan);
    void CheckFaultsOnComplete(bool draw = false, std::vector<std::pair<std::uint64_t, std::uint64_t>> tableRanges = {});

    // Draws (MetalDraw.mm, D6): the device's draw objects (made at the first draw), a sync when
    // recorded GPU work writes guest memory the CPU is about to read, the GPU buffer holding a
    // wrapped guest range, and the recording of one draw into the open batch.
    Metal::DrawCache& Draws();
    void SettleRange(std::uint64_t address, std::uint64_t bytes);
    // CMASK fast clears expanded into guest memory (MetalDraw.mm).
    // `resident`: the target's resident image, refreshed from the patched memory.
    void MaterializeCmask(const Graphics::ColorTarget& color, Graphics::StorageTexture* resident);
    void MaterializeDcc(const Graphics::ColorTarget& color, Graphics::StorageTexture* resident);
    id<MTLBuffer> BufferAt(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset);
    // `tessellation`: a tessellated draw (MetalTessellation.h); `vertex` is then its domain stage.
    void RecordDraw(const Graphics::State& graphics, const Pm4::DrawParameters& draw, const Graphics::CompiledShader& vertex, const Graphics::CompiledShader* fragment, std::span<const Graphics::GuestMemorySnapshot> snapshots, const Graphics::Metal::TessellationStages* tessellation = nullptr);
    // Waits for every recorded batch and runs their completions (VulkanDevice::WaitIdle).
    void Sync();
    // `serial`: the batch whose completion checks (0: no image table ranges are consumed).
    void CheckFaults(std::uint64_t serial = 0);
};

}

#endif
