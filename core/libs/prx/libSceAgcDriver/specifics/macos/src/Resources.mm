// The buffer primitives on Metal (work package D2): twin of Graphics/src/Resources.cpp. Buffer and
// DeviceBuffer hold pool allocations (MTLBuffers, resident on the device's queue); CommandBatch is a
// one-shot MTL4 command buffer with its own event; CopyBuffer and RecordMemoryBarrier record into a
// Metal::CommandStream. The active recorder is reached through Metal::ActiveRecorder(), not its
// symbols, so targets without the recorder link.

#include "MetalCommandStream.h"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "MetalResidency.h"
#include <algorithm>
#include <cstdio>
#include <exception>

namespace AgcDriver::Graphics {

namespace Metal {

struct CommandBatchObjects {
    id<MTL4CommandQueue> queue = nil;
    id<MTL4CommandAllocator> allocator = nil;
    id<MTL4CommandBuffer> commandBuffer = nil;
    id<MTLSharedEvent> event = nil;
    std::unique_ptr<CommandStream> stream;
    std::shared_ptr<std::string> failure = std::make_shared<std::string>();
    std::shared_ptr<std::mutex> failureMutex = std::make_shared<std::mutex>();
};

void ObjectsDeleter::operator()(CommandBatchObjects* objects) const noexcept {
    delete objects;
}

namespace {

// The upload arena of the one-shot batches of a device (one per queue, kept for the process: a
// CommandBatch is short-lived and should not make a residency set each time).
std::shared_ptr<UploadArena> BatchArena(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    static std::mutex mutex;
    static std::vector<std::pair<const void*, std::shared_ptr<UploadArena>>> arenas;
    std::lock_guard lock(mutex);
    for (const auto& [key, arena] : arenas) {
        if (key == (__bridge const void*)queue) return arena;
    }
    auto arena = std::make_shared<UploadArena>(device, queue);
    arenas.emplace_back((__bridge const void*)queue, arena);
    return arena;
}

}

}

Buffer::Buffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) : context(context), size(size), capacity(BufferPool::Capacity(size, properties)), usage(BufferPool::Usage(usage, properties)), properties(properties) {
    Require(size != 0, "zero-sized GPU buffer");
    // macOS: every Metal buffer has a GPU address, so SHADER_DEVICE_ADDRESS needs no device feature.
    cache = GetBufferPool(context);
    if (const auto taken = cache->Take(size, this->usage, properties)) allocation = *taken;
    else allocation = cache->Allocate(capacity, this->usage, properties);
    capacity = allocation.bytes;
    mapping = allocation.mapping;
    deviceAddress = allocation.address;
    ready = true;
}

Buffer::~Buffer() {
    release();
}

void Buffer::release() noexcept {
    if (ready && cache) cache->Put(allocation);
    ready = false;
}

VkBuffer Buffer::Handle() const {
    return allocation.buffer;
}

std::span<std::byte> Buffer::Bytes() {
    Require(mapping != nullptr, "device-local buffer has no host mapping");
    return {static_cast<std::byte*>(mapping), size};
}

void Buffer::Invalidate() {
    Require(mapping != nullptr, "cannot invalidate an unmapped GPU buffer");
}

DeviceBuffer::DeviceBuffer(const Context& context, std::size_t size, VkBufferUsageFlags usage) : context(context), size(size), capacity(BufferPool::Capacity(size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)), usage(BufferPool::Usage(usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    Require(size != 0, "zero-sized device buffer");
    cache = GetBufferPool(context);
    if (const auto taken = cache->Take(size, this->usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) allocation = *taken;
    else allocation = cache->Allocate(capacity, this->usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    capacity = allocation.bytes;
}

DeviceBuffer::~DeviceBuffer() {
    release();
}

void DeviceBuffer::release() noexcept {
    if (allocation.buffer != VK_NULL_HANDLE && cache) cache->Put(allocation);
    allocation.buffer = VK_NULL_HANDLE;
}

VkBuffer DeviceBuffer::Handle() const {
    return allocation.buffer;
}

std::size_t DeviceBuffer::Size() const {
    return size;
}

void CopyBuffer(const Context&, VkCommandBuffer commands, VkBuffer source, VkDeviceSize sourceOffset, VkBuffer destination, VkDeviceSize destinationOffset, VkDeviceSize bytes) {
    Metal::Stream(commands).CopyBuffer(Metal::BufferOf(source), sourceOffset, Metal::BufferOf(destination), destinationOffset, bytes);
}

void RecordMemoryBarrier(const Context&, VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags, VkAccessFlags) {
    Metal::Stream(commands).PipelineBarrier(sourceStage, destinationStage);
}

void FillDeviceFunctions(const Context&, DeviceFunctions& functions) {
    functions = {};
}

RenderTarget::RenderTarget(const Context& context, const ColorTarget&, bool) : context(context) {
    Metal::Fail("RenderTarget (a Vulkan colour target image) is not implemented on Metal yet: it needs the format table of work package D4 and the render passes of D6");
}

RenderTarget::~RenderTarget() = default;

VkImage RenderTarget::Image() const {
    return image;
}

VkImageView RenderTarget::View() const {
    return view;
}

CommandBatch::CommandBatch(const Context& context) : context(context) {
    // Records into the device's one queue: device-lock work only.
    GuestMemory::AssertGpuLockHeld("CommandBatch");
    auto objects = std::unique_ptr<Metal::CommandBatchObjects, Metal::ObjectsDeleter>(new Metal::CommandBatchObjects);
    id<MTLDevice> device = Metal::DeviceOf(context);
    objects->queue = Metal::QueueOf(context);
    if (device == nil || objects->queue == nil) Metal::Fail("a command batch needs a Context carrying a Metal device and command queue");
    objects->allocator = [device newCommandAllocator];
    objects->commandBuffer = [device newCommandBuffer];
    objects->event = [device newSharedEvent];
    if (objects->allocator == nil || objects->commandBuffer == nil || objects->event == nil) Metal::Fail("cannot create a one-shot command buffer");
    objects->stream = std::make_unique<Metal::CommandStream>(Metal::BatchArena(device, objects->queue));
    objects->stream->Begin(objects->commandBuffer, objects->allocator);
    metal = std::move(objects);
    try {
        // Work recorded so far goes first in queue order, so this batch sees its results.
        if (const auto recorder = Metal::ActiveRecorder(); recorder.submitOpen != nullptr) recorder.submitOpen();
    } catch (...) {
        release();
        throw;
    }
}

CommandBatch::~CommandBatch() {
    release();
}

void CommandBatch::release() noexcept {
    if (metal == nullptr) return;
    if (pending) {
        // Destroyed before Wait: the GPU must be done with the stream's upload memory first.
        if (![metal->event waitUntilSignaledValue:1 timeoutMS:20000]) {
            std::fprintf(stderr, "[gpu] a one-shot command batch did not finish within 20 s; its objects are leaked\n");
            static_cast<void>(metal.release());
            return;
        }
        pending = false;
    }
    metal.reset();
}

VkCommandBuffer CommandBatch::Handle() const {
    return Metal::Handle(*metal->stream);
}

void CommandBatch::SubmitAndWait() {
    Submit();
    Wait();
}

void CommandBatch::Submit() {
    PerformanceTimer timing("Graphics.Submit");
    Require(!submitted, "command batch has already been submitted");
    id<MTL4CommandBuffer> commandBuffer = metal->stream->End();
    timing.Mark("command_end");
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    std::shared_ptr<std::string> failure = metal->failure;
    std::shared_ptr<std::mutex> failureMutex = metal->failureMutex;
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        if (feedback.error == nil) return;
        std::lock_guard lock(*failureMutex);
        *failure = Metal::Describe(feedback.error);
    }];
    id<MTL4CommandBuffer> commandBuffers[1] = {commandBuffer};
    ::AgcDriver::Metal::Residency::CommitQueue([&] {
        [metal->queue commit:commandBuffers count:1 options:options];
        [metal->queue signalEvent:metal->event value:1];
    });
    pending = true;
    submitted = true;
    timing.Mark("queue_submit");
}

void CommandBatch::Wait() {
    Require(submitted, "command batch has not been submitted");
    if (!pending) return;
    PerformanceTimer timing("Graphics.Wait");
    const bool done = [metal->event waitUntilSignaledValue:1 timeoutMS:20000];
    timing.Mark("fence_wait");
    if (!done) Metal::Fail("a one-shot command batch did not finish on the GPU within 20 s");
    pending = false;
    metal->stream->Recycle();
    {
        std::lock_guard lock(*metal->failureMutex);
        if (!metal->failure->empty()) Metal::Fail("a one-shot command batch failed on the GPU: " + *metal->failure);
    }
    // Recorded batches preceded this one, so their completions (write-backs) can run now.
    if (const auto recorder = Metal::ActiveRecorder(); recorder.reap != nullptr) recorder.reap();
}

}
