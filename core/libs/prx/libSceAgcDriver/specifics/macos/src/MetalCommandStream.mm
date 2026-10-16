// The Metal command stream (work package D2): encoders, the bring-up barrier policy, inline data
// and the helper kernels every recorded or one-shot batch uses. See MetalCommandStream.h.

#include "MetalCommandStream.h"
#include "RecorderKernels.h"
#include "MetalResidency.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

namespace AgcDriver::Graphics::Metal {

MTLStages StagesOf(VkPipelineStageFlags stages) {
    MTLStages result = 0;
    if ((stages & (VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)) != 0) return MTLStageAll;
    if ((stages & VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT) != 0) result |= MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh;
    // Indirect arguments are read by the draw (vertex, object, mesh) or the dispatch that uses them.
    if ((stages & VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT) != 0) result |= MTLStageVertex | MTLStageObject | MTLStageMesh | MTLStageDispatch;
    if ((stages & (VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT)) != 0) result |= MTLStageVertex;
    if ((stages & (VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT)) != 0) result |= MTLStageObject;
    if ((stages & (VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT)) != 0) result |= MTLStageMesh;
    if ((stages & (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)) != 0) result |= MTLStageFragment;
    if ((stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0) result |= MTLStageDispatch;
    // Transfers are blits, or dispatches of the fill kernel (CommandStream::FillBuffer).
    if ((stages & VK_PIPELINE_STAGE_TRANSFER_BIT) != 0) result |= MTLStageBlit | MTLStageDispatch;
    return result;
}

UploadArena::UploadArena(id<MTLDevice> device, id<MTL4CommandQueue> queue) : device(device), queue(queue) {
    if (device == nil) Fail("an upload arena needs a Metal device");
    if (queue == nil) return;
    MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
    descriptor.label = @"AGC upload arena";
    descriptor.initialCapacity = 16;
    NSError* error = nil;
    residency = [device newResidencySetWithDescriptor:descriptor error:&error];
    if (residency == nil) Fail("the upload arena's residency set: " + Describe(error));
    ::AgcDriver::Metal::Residency::Attach(queue, residency);
}

UploadArena::~UploadArena() {
    if (queue != nil && residency != nil) ::AgcDriver::Metal::Residency::Detach(queue, residency);
}

id<MTLBuffer> UploadArena::Take(std::size_t bytes) {
    std::lock_guard lock(mutex);
    if (bytes <= ChunkBytes && !free.empty()) {
        id<MTLBuffer> buffer = free.back();
        free.pop_back();
        return buffer;
    }
    const auto length = std::max(bytes, ChunkBytes);
    id<MTLBuffer> buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    if (buffer == nil) Fail("cannot allocate " + std::to_string(length) + " bytes of upload memory");
    if (residency != nil) {
        ::AgcDriver::Metal::Residency::Add(residency, buffer);
        ::AgcDriver::Metal::Residency::Commit(residency);
    }
    ++made;
    return buffer;
}

void UploadArena::Give(std::vector<id<MTLBuffer>>& buffers) {
    std::lock_guard lock(mutex);
    bool removed = false;
    for (id<MTLBuffer> buffer : buffers) {
        // Standard chunks are kept (a bounded number); dedicated ones leave the residency set.
        if (buffer.length == ChunkBytes && free.size() < 64) {
            free.push_back(buffer);
            continue;
        }
        if (residency != nil) {
            ::AgcDriver::Metal::Residency::Remove(residency, buffer);
            removed = true;
        }
    }
    if (removed) ::AgcDriver::Metal::Residency::Commit(residency);
    buffers.clear();
}

std::size_t UploadArena::Made() const {
    std::lock_guard lock(mutex);
    return made;
}

CommandStream::CommandStream(std::shared_ptr<UploadArena> arena) : arena(std::move(arena)) {
    if (this->arena == nullptr) Fail("a command stream needs an upload arena");
}

CommandStream::~CommandStream() {
    // The owner destroys a stream whose command buffer is not in flight (never committed, or
    // completed), so its upload memory can go back to the arena; a stream destroyed while
    // recording (a failed record) ends its encoder and command buffer first.
    if (recording) {
        EndEncoder();
        [commandBuffer endCommandBuffer];
        recording = false;
    }
    if (!uploads.empty()) arena->Give(uploads);
}

void CommandStream::Begin(id<MTL4CommandBuffer> commandBuffer, id<MTL4CommandAllocator> allocator) {
    if (recording) Fail("a command stream began while it was still recording");
    if (commandBuffer == nil || allocator == nil) Fail("a command stream began without a command buffer or allocator");
    this->commandBuffer = commandBuffer;
    [commandBuffer beginCommandBufferWithAllocator:allocator];
    recording = true;
    compute = nil;
    render = nil;
    renderVisibility = nil;
}

id<MTL4CommandBuffer> CommandStream::End() {
    if (!recording) Fail("a command stream ended without recording");
    EndEncoder();
    [commandBuffer endCommandBuffer];
    recording = false;
    return commandBuffer;
}

id<MTLDevice> CommandStream::Device() const {
    return arena->Device();
}

namespace {

// Debug aid: APS5_METAL_SERIAL_COMMANDS=1 puts a full encoder barrier before every command recorded
// into an open compute encoder (each Compute() call), so a missing barrier can be bisected.
bool SerialCommands() {
    static const bool serial = std::getenv("APS5_METAL_SERIAL_COMMANDS") != nullptr;
    return serial;
}

}

id<MTL4ComputeCommandEncoder> CommandStream::Compute() {
    if (compute != nil) {
        if (SerialCommands()) Barrier(MTLStageAll, MTLStageAll);
        return compute;
    }
    if (!recording) Fail("a compute encoder was requested from a stream that is not recording");
    EndEncoder();
    compute = [commandBuffer computeCommandEncoder];
    if (compute == nil) Fail("the command buffer refused a compute encoder");
    [compute barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch | MTLStageBlit | MTLStageAccelerationStructure visibilityOptions:MTL4VisibilityOptionDevice];
    ++encoders;
    return compute;
}

id<MTL4RenderCommandEncoder> CommandStream::BeginRender(MTL4RenderPassDescriptor* descriptor) {
    if (!recording) Fail("a render pass was begun on a stream that is not recording");
    EndEncoder();
    render = [commandBuffer renderCommandEncoderWithDescriptor:descriptor];
    if (render == nil) Fail("the command buffer refused a render pass");
    [render barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh visibilityOptions:MTL4VisibilityOptionDevice];
    renderVisibility = descriptor.visibilityResultBuffer;
    ++encoders;
    return render;
}

void CommandStream::EndEncoder() {
    if (compute != nil) [compute endEncoding];
    if (render != nil) [render endEncoding];
    compute = nil;
    render = nil;
    renderVisibility = nil;
}

void CommandStream::Barrier(MTLStages after, MTLStages before) {
    if (compute != nil) {
        constexpr MTLStages computeStages = MTLStageDispatch | MTLStageBlit | MTLStageAccelerationStructure;
        after &= computeStages;
        before &= computeStages;
        // Work of another encoder kind (a render pass's) is ordered by this encoder's queue barrier.
        if (after == 0 || before == 0) return;
        [compute barrierAfterEncoderStages:after beforeEncoderStages:before visibilityOptions:MTL4VisibilityOptionDevice];
        ++barriers;
        return;
    }
    if (render != nil && after != 0 && before != 0) {
        EndEncoder();
        ++barriers;
    }
}

Upload CommandStream::Allocate(std::size_t bytes, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1u)) != 0) Fail("an upload alignment must be a power of two");
    if (bytes == 0) bytes = alignment;
    auto offset = (currentUsed + alignment - 1u) & ~(alignment - 1u);
    if (current == nil || offset + bytes > current.length) {
        current = arena->Take(bytes);
        uploads.push_back(current);
        offset = 0;
    }
    currentUsed = offset + bytes;
    auto* host = static_cast<std::byte*>(current.contents) + offset;
    return {current, offset, current.gpuAddress + offset, host};
}

Upload CommandStream::Write(std::span<const std::byte> bytes, std::size_t alignment) {
    auto upload = Allocate(bytes.size(), alignment);
    if (!bytes.empty()) std::memcpy(upload.host, bytes.data(), bytes.size());
    return upload;
}

void CommandStream::UpdateBuffer(id<MTLBuffer> destination, NSUInteger offset, std::span<const std::byte> bytes) {
    if (bytes.empty()) return;
    if (destination == nil) Fail("a buffer update has no destination buffer");
    if (offset + bytes.size() > destination.length) Fail("a buffer update of " + std::to_string(bytes.size()) + " bytes at " + std::to_string(offset) + " overruns its " + std::to_string(destination.length) + "-byte buffer");
    const auto upload = Write(bytes, 4);
    [Compute() copyFromBuffer:upload.buffer sourceOffset:upload.offset toBuffer:destination destinationOffset:offset size:bytes.size()];
    ++commands;
}

void CommandStream::CopyBuffer(id<MTLBuffer> source, NSUInteger sourceOffset, id<MTLBuffer> destination, NSUInteger destinationOffset, NSUInteger bytes) {
    if (bytes == 0) return;
    if (source == nil || destination == nil) Fail("a buffer copy has no source or destination buffer");
    if (sourceOffset + bytes > source.length || destinationOffset + bytes > destination.length) Fail("a buffer copy of " + std::to_string(bytes) + " bytes overruns its source or destination");
    [Compute() copyFromBuffer:source sourceOffset:sourceOffset toBuffer:destination destinationOffset:destinationOffset size:bytes];
    ++commands;
}

void CommandStream::FillBuffer(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger bytes, std::uint32_t value) {
    if (bytes == 0) return;
    if (buffer == nil) Fail("a buffer fill has no buffer");
    if (offset + bytes > buffer.length) Fail("a buffer fill of " + std::to_string(bytes) + " bytes at " + std::to_string(offset) + " overruns its " + std::to_string(buffer.length) + "-byte buffer");
    const auto byte = value & 0xffu;
    if (value == byte * 0x01010101u) {
        [Compute() fillBuffer:buffer range:NSMakeRange(offset, bytes) value:static_cast<std::uint8_t>(byte)];
        ++commands;
        return;
    }
    const std::array<std::uint32_t, 4> pattern{value, value, value, value};
    FillPattern(buffer, offset, bytes, pattern);
}

void CommandStream::FillPattern(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger bytes, std::span<const std::uint32_t, 4> pattern) {
    if (bytes == 0) return;
    if (buffer == nil) Fail("a pattern fill has no buffer");
    if (offset % 4 != 0 || bytes % 4 != 0) Fail("a pattern fill must be dword aligned");
    if (offset + bytes > buffer.length) Fail("a pattern fill overruns its buffer");
    if (bytes / 4 > 0xffffffffull) Fail("a pattern fill exceeds 2^32 dwords");
    struct {
        std::uint64_t target;
        std::uint32_t pattern[4];
        std::uint32_t dwords;
        std::uint32_t first;
    } parameters{buffer.gpuAddress + offset, {pattern[0], pattern[1], pattern[2], pattern[3]}, static_cast<std::uint32_t>(bytes / 4), 0u};
    static_assert(sizeof(parameters) == 32);
    id<MTLComputePipelineState> pipeline = HelperPipeline(Device(), HelperKernel::FillPattern);
    const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    const NSUInteger groups = (bytes / 4 + width - 1) / width;
    DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake(groups, 1, 1), MTLSizeMake(width, 1, 1));
}

id<MTL4ArgumentTable> CommandStream::Arguments() {
    if (arguments != nil) return arguments;
    MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = 8;
    descriptor.label = @"AGC command stream";
    NSError* error = nil;
    arguments = [Device() newArgumentTableWithDescriptor:descriptor error:&error];
    if (arguments == nil) Fail("the command stream's argument table: " + Describe(error));
    return arguments;
}

void CommandStream::DispatchHelper(id<MTLComputePipelineState> pipeline, std::span<const std::byte> parameters, MTLSize threadgroups, MTLSize threadsPerThreadgroup) {
    const auto upload = Write(parameters, 16);
    auto* encoder = Compute();
    id<MTL4ArgumentTable> table = Arguments();
    [table setAddress:upload.address atIndex:0];
    [encoder setArgumentTable:table];
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    ++commands;
}

void CommandStream::WriteTimestamp(id<MTL4CounterHeap> heap, NSUInteger index) {
    if (!recording) Fail("a timestamp was written into a stream that is not recording");
    if (compute != nil) [compute writeTimestampWithGranularity:MTL4TimestampGranularityPrecise intoHeap:heap atIndex:index];
    else if (render != nil) [render writeTimestampWithGranularity:MTL4TimestampGranularityPrecise afterStage:MTLRenderStageFragment intoHeap:heap atIndex:index];
    else [commandBuffer writeTimestampIntoHeap:heap atIndex:index];
}

void CommandStream::Recycle() {
    if (recording) Fail("a command stream was recycled while recording");
    current = nil;
    currentUsed = 0;
    arena->Give(uploads);
    kept.clear();
}

namespace {

std::atomic<void (*)()> activeSubmitOpen{nullptr};
std::atomic<void (*)()> activeReap{nullptr};

}

void SetActiveRecorderCalls(const ActiveRecorderCalls& calls) {
    activeSubmitOpen.store(calls.submitOpen, std::memory_order_release);
    activeReap.store(calls.reap, std::memory_order_release);
}

ActiveRecorderCalls ActiveRecorder() {
    return {activeSubmitOpen.load(std::memory_order_acquire), activeReap.load(std::memory_order_acquire)};
}

id<MTLComputePipelineState> HelperPipeline(id<MTLDevice> device, HelperKernel kernel) {
    static std::mutex mutex;
    static std::map<std::pair<const void*, HelperKernel>, id<MTLComputePipelineState>> pipelines;
    static std::map<const void*, id<MTLLibrary>> libraries;
    static constexpr std::array<const char*, static_cast<std::size_t>(HelperKernel::Count)> names{"FillPattern", "MeshArguments", "SampleCounter", "SampleDumps", "IndexGather"};
    if (device == nil) Fail("a helper kernel needs a Metal device");
    const auto key = std::pair{(__bridge const void*)device, kernel};
    std::lock_guard lock(mutex);
    if (const auto found = pipelines.find(key); found != pipelines.end()) return found->second;
    auto& library = libraries[key.first];
    NSError* error = nil;
    if (library == nil) {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_2;
        options.mathMode = MTLMathModeSafe;
        library = [device newLibraryWithSource:@(RecorderKernelSource) options:options error:&error];
        if (library == nil) Fail(std::string("the helper kernels do not compile: ") + Describe(error));
    }
    const char* name = names[static_cast<std::size_t>(kernel)];
    id<MTLFunction> function = [library newFunctionWithName:@(name)];
    if (function == nil) Fail(std::string("the helper kernel library has no function ") + name);
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) Fail(std::string("the helper kernel ") + name + " has no pipeline: " + Describe(error));
    pipelines.emplace(key, pipeline);
    return pipeline;
}

}
