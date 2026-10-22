#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALCOMMANDSTREAM_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALCOMMANDSTREAM_H

// The Metal command stream of the AGC driver (work package D2, docs/mac/M2_DESIGN.md §3, §6):
// what a recorded batch (Recorder) or a one-shot CommandBatch records into, and what the Vulkan
// handle types of the shared headers carry during the transition (D-M2-3). Objective-C++ only.
//
// Recording model (Metal 4, bring-up barrier policy of D-M2-1):
// - One MTL4CommandBuffer per batch, one encoder open at a time: Compute() for dispatches, copies
//   and fills (Metal 4 unifies compute and blit work in the compute encoder), BeginRender() for a
//   render pass. Opening an encoder ends the open one.
// - Every encoder starts with a consumer queue barrier (all earlier stages of the queue, device
//   visibility), so it runs after everything committed or encoded before it, in this command
//   buffer or an earlier one. Commands inside one compute encoder run concurrently unless
//   Barrier() separates them: a caller orders dependent commands exactly where Vulkan records a
//   pipeline barrier (RecordMemoryBarrier and the recorder's barriers map to Barrier()).
// - Metal 4 command buffers retain nothing and have no setBytes: a resource stays alive through
//   Recorder::Keep (or the CommandBatch) until the batch completed, and must be resident (in a
//   residency set attached to the queue); inline data (vkCmdUpdateBuffer, parameters, descriptor
//   tables) goes through Upload()/Write(), memory the stream keeps until it is recycled.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver::Graphics::Metal {

// Whether a pair of resolved MTL4 timestamps measures a range: both written (an entry the GPU never
// wrote reads MTLCounterErrorValue, all ones; a zero is no time either) and in order (D59).
inline bool TimestampPairValid(std::uint64_t begin, std::uint64_t end) {
    constexpr std::uint64_t unwritten = ~std::uint64_t{0};
    return begin != 0u && begin != unwritten && end != unwritten && end >= begin;
}

[[noreturn]] inline void Fail(const std::string& what) {
    throw std::runtime_error("Metal recorder (work package D2): " + what);
}

inline std::string Describe(NSError* error) {
    return error != nil && error.localizedDescription != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

// The Metal objects the Vulkan-typed fields of a Context carry (D-M2-3): `device` a bridged
// id<MTLDevice>, `queue` a bridged id<MTL4CommandQueue>, both owned by the device that filled the
// Context. A Context that resolves Vulkan entry points (`deviceProc`, a Vulkan or mock-Vulkan
// device) carries no Metal objects: nil.
inline bool IsMetal(const Context& context) {
    return context.deviceProc == nullptr && context.device != VK_NULL_HANDLE;
}

inline id<MTLDevice> DeviceOf(const Context& context) {
    return IsMetal(context) ? (__bridge id<MTLDevice>)reinterpret_cast<void*>(context.device) : nil;
}

inline id<MTL4CommandQueue> QueueOf(const Context& context) {
    return IsMetal(context) && context.queue != VK_NULL_HANDLE ? (__bridge id<MTL4CommandQueue>)reinterpret_cast<void*>(context.queue) : nil;
}

inline VkDevice DeviceHandle(id<MTLDevice> device) {
    return reinterpret_cast<VkDevice>((__bridge void*)device);
}

inline VkQueue QueueHandle(id<MTL4CommandQueue> queue) {
    return reinterpret_cast<VkQueue>((__bridge void*)queue);
}

// A VkBuffer carries an id<MTLBuffer> (unretained: the Buffer, DeviceBuffer or host import that
// made it owns the buffer).
inline id<MTLBuffer> BufferOf(VkBuffer buffer) {
    return (__bridge id<MTLBuffer>)reinterpret_cast<void*>(buffer);
}

inline VkBuffer BufferHandle(id<MTLBuffer> buffer) {
    return reinterpret_cast<VkBuffer>((__bridge void*)buffer);
}

// The Metal stages a Vulkan pipeline stage mask covers on the GPU. Host access has no GPU stage
// (a completed command buffer's writes are visible to the CPU).
MTLStages StagesOf(VkPipelineStageFlags stages);

// Shared, untracked buffers for inline data, resident through one residency set attached to the
// queue. Chunks of ChunkBytes are recycled; a larger request gets a buffer of its own that leaves
// the residency set when its stream is recycled. Thread-safe.
class UploadArena {
public:
    static constexpr std::size_t ChunkBytes = std::size_t{256} << 10u;
    // `queue` may be nil (no residency set is attached then: the arena serves CPU-side tests only).
    UploadArena(id<MTLDevice> device, id<MTL4CommandQueue> queue);
    ~UploadArena();
    UploadArena(const UploadArena&) = delete;
    UploadArena& operator=(const UploadArena&) = delete;
    id<MTLDevice> Device() const { return device; }
    id<MTLBuffer> Take(std::size_t bytes);
    void Give(std::vector<id<MTLBuffer>>& buffers);
    // Chunks and dedicated buffers made so far (tests).
    std::size_t Made() const;

private:
    id<MTLDevice> device;
    id<MTL4CommandQueue> queue;
    id<MTLResidencySet> residency;
    mutable std::mutex mutex;
    std::vector<id<MTLBuffer>> free;
    std::size_t made = 0;
};

// Bytes of an upload: valid for the GPU until the stream is recycled.
struct Upload {
    id<MTLBuffer> buffer;
    NSUInteger offset;
    std::uint64_t address;
    std::byte* host;
};

class CommandStream {
public:
    explicit CommandStream(std::shared_ptr<UploadArena> arena);
    ~CommandStream();
    CommandStream(const CommandStream&) = delete;
    CommandStream& operator=(const CommandStream&) = delete;

    // Starts recording into `commandBuffer` (begun with `allocator` here); both stay the caller's.
    void Begin(id<MTL4CommandBuffer> commandBuffer, id<MTL4CommandAllocator> allocator);
    // Ends the open encoder and the command buffer, which the caller then commits.
    id<MTL4CommandBuffer> End();
    bool Recording() const { return commandBuffer != nil && recording; }
    id<MTLDevice> Device() const;
    id<MTL4CommandBuffer> CommandBuffer() const { return commandBuffer; }

    // The open compute encoder, opening one (after its queue barrier) when none is. Debug aid:
    // APS5_METAL_SERIAL_COMMANDS=1 records a full encoder barrier at every call that returns an
    // already open encoder (every command then runs after the one before).
    id<MTL4ComputeCommandEncoder> Compute();
    // Ends the open encoder and begins a render pass (after its queue barrier).
    id<MTL4RenderCommandEncoder> BeginRender(MTL4RenderPassDescriptor* descriptor);
    id<MTL4RenderCommandEncoder> Render() const { return render; }
    // The open compute encoder, or nil (unlike Compute(), never opens one).
    id<MTL4ComputeCommandEncoder> OpenCompute() const { return compute; }
    // The visibility result buffer the open render pass was begun with (nil: none, or no pass).
    id<MTLBuffer> RenderVisibilityBuffer() const { return renderVisibility; }
    void EndEncoder();

    // A pipeline barrier between the commands recorded before and after it (Metal stage masks):
    // an encoder barrier inside the open compute encoder (device visibility); inside a render pass
    // the pass ends, since Apple GPUs have no intra-pass barrier that waits for fragment work, and
    // the next encoder's queue barrier orders the work; with no encoder open the next encoder's
    // queue barrier already does.
    void Barrier(MTLStages after = MTLStageAll, MTLStages before = MTLStageAll);
    // The same for a Vulkan pipeline barrier's stage masks (a distinct name: Vulkan stage constants
    // convert to both mask types).
    void PipelineBarrier(VkPipelineStageFlags source, VkPipelineStageFlags destination) { Barrier(StagesOf(source), StagesOf(destination)); }

    // Inline data: `bytes` of upload memory (aligned) the CPU fills now and the GPU reads when the
    // stream executes.
    Upload Allocate(std::size_t bytes, std::size_t alignment = 16);
    Upload Write(std::span<const std::byte> bytes, std::size_t alignment = 16);
    // vkCmdUpdateBuffer: `bytes` stored into `destination` at `offset` when the stream executes.
    void UpdateBuffer(id<MTLBuffer> destination, NSUInteger offset, std::span<const std::byte> bytes);
    void CopyBuffer(id<MTLBuffer> source, NSUInteger sourceOffset, id<MTLBuffer> destination, NSUInteger destinationOffset, NSUInteger bytes);
    // vkCmdFillBuffer: `value` repeated over [offset, offset + bytes) (4-byte aligned). A value of
    // four equal bytes is a Metal fill (MTLStageBlit); any other dword, and FillPattern's 16-byte
    // patterns, run the stream's fill kernel (MTLStageDispatch). Vulkan TRANSFER barriers cover
    // both stages (StagesOf).
    void FillBuffer(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger bytes, std::uint32_t value);
    void FillPattern(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger bytes, std::span<const std::uint32_t, 4> pattern);
    // An argument table for the stream's helper dispatches (and callers that bind no table of
    // their own); bindings are snapshotted per dispatch, so one table serves the whole stream.
    id<MTL4ArgumentTable> Arguments();
    // Dispatches `pipeline` with `parameters` uploaded and bound at buffer index 0.
    void DispatchHelper(id<MTLComputePipelineState> pipeline, std::span<const std::byte> parameters, MTLSize threadgroups, MTLSize threadsPerThreadgroup);

    // A GPU timestamp of the work recorded so far, written into `heap` at `index` (by the open
    // encoder, or by the command buffer between encoders).
    void WriteTimestamp(id<MTL4CounterHeap> heap, NSUInteger index);

    // Returns the stream's upload memory once the GPU finished the command buffer (or it was never
    // committed). The owner destroys a stream only in that state too.
    void Recycle();
    // An object the recorded commands use that nothing else holds until they complete (a texture view
    // a helper dispatch binds by resource ID): a one-shot batch has no recorder to keep it, so the
    // stream does, until Recycle or its destruction (both only once the GPU is done with it).
    void Keep(std::shared_ptr<void> object) { kept.push_back(std::move(object)); }

    // Counters for tests and the [barriers] line: encoders opened, barriers recorded, commands.
    std::uint64_t Encoders() const { return encoders; }
    std::uint64_t Barriers() const { return barriers; }
    std::uint64_t CommandCount() const { return commands; }
    void CountCommand() { ++commands; }

private:
    std::shared_ptr<UploadArena> arena;
    id<MTL4CommandBuffer> commandBuffer = nil;
    id<MTL4ComputeCommandEncoder> compute = nil;
    id<MTL4RenderCommandEncoder> render = nil;
    id<MTLBuffer> renderVisibility = nil;
    bool recording = false;
    std::vector<id<MTLBuffer>> uploads;
    std::vector<std::shared_ptr<void>> kept;
    id<MTLBuffer> current = nil;
    std::size_t currentUsed = 0;
    id<MTL4ArgumentTable> arguments = nil;
    std::uint64_t encoders = 0;
    std::uint64_t barriers = 0;
    std::uint64_t commands = 0;
};

// What a one-shot CommandBatch needs from the active recorder: submit its open batch first (queue
// order) and retire its finished batches after a wait. Recorder::Activate registers them, so the
// buffer primitives link in targets that compile Resources.cpp without Recorder.cpp
// (agc_driver_bda_device_tests); with no recorder registered they do nothing, as with no active
// recorder.
struct ActiveRecorderCalls {
    void (*submitOpen)() = nullptr;
    void (*reap)() = nullptr;
};
void SetActiveRecorderCalls(const ActiveRecorderCalls& calls);
ActiveRecorderCalls ActiveRecorder();

// The driver's helper kernels (MSL in RecorderKernels.h), compiled once per device and kept for
// the process: FillPattern, MeshArguments, SampleCounter, SampleDumps, DispatchArguments, IndexGather.
enum class HelperKernel { FillPattern, MeshArguments, SampleCounter, SampleDumps, DispatchArguments, IndexGather, Count };
id<MTLComputePipelineState> HelperPipeline(id<MTLDevice> device, HelperKernel kernel);

// A VkCommandBuffer handle carries the address of a CommandStream.
inline CommandStream& Stream(VkCommandBuffer handle) {
    if (handle == VK_NULL_HANDLE) Fail("a null command buffer handle carries no command stream");
    return *reinterpret_cast<CommandStream*>(handle);
}

inline VkCommandBuffer Handle(CommandStream& stream) {
    return reinterpret_cast<VkCommandBuffer>(&stream);
}

// Test hook (RecorderKeptBytesMetal.mm, the twin of upstream's mocked vkEndCommandBuffer and
// vkQueueSubmit failures): the next Recorder::Submit on this thread throws `what` after ending its
// command buffer (`beforeCommit`) or instead of committing it, before the batch counts as submitted.
void FailNextRecorderSubmit(const char* what, bool beforeCommit);

}

#endif
