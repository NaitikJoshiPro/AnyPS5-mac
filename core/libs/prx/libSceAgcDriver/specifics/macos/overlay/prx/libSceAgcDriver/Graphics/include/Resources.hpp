#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESOURCES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESOURCES_HPP

#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MetalObjects.hpp"
#include <span>

// macOS (work package D2): the buffer primitives of Graphics/src/Resources.cpp on Metal
// (specifics/macos/src/Resources.mm, BufferAddress.mm). The signatures are the upstream ones;
// their Vulkan handle types carry Metal objects (D-M2-3): a VkBuffer is a bridged id<MTLBuffer>
// (Metal::BufferOf), a VkCommandBuffer the address of a Metal::CommandStream (Metal::Stream).

namespace AgcDriver::Graphics {

class Buffer {
public:
    Buffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    VkBuffer Handle() const;
    VkDeviceAddress DeviceAddress() const;
    // Host-visible buffers only: a device-local one (properties without HOST_VISIBLE, the staging
    // shadows of GuestBufferMemory) is private storage and its bytes move by GPU copies alone.
    std::span<std::byte> Bytes();
    bool Mapped() const { return mapping != nullptr; }
    // Unified memory: a completed command buffer's writes are visible to the CPU, so this only
    // checks the buffer is mapped.
    void Invalidate();

private:
    void release() noexcept;
    Context context;
    // The pool allocation this buffer holds (its retained MTLBuffer), returned at release.
    BufferAllocation allocation{};
    VkDeviceAddress deviceAddress = 0;
    void* mapping = nullptr;
    std::size_t size;
    std::size_t capacity;
    // Fully made (or taken from the pool), so release returns it to the pool.
    bool ready = false;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags properties;
    std::shared_ptr<BufferPool> cache;
};

// Device-local scratch memory for GPU-side layout conversion (private storage on Metal).
class DeviceBuffer {
public:
    DeviceBuffer(const Context& context, std::size_t size, VkBufferUsageFlags usage);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    VkBuffer Handle() const;
    std::size_t Size() const;

private:
    void release() noexcept;
    Context context;
    BufferAllocation allocation{};
    std::size_t size;
    std::size_t capacity;
    VkBufferUsageFlags usage;
    std::shared_ptr<BufferPool> cache;
};

// Records a buffer copy (a blit in the stream's compute encoder).
void CopyBuffer(const Context& context, VkCommandBuffer commands, VkBuffer source, VkDeviceSize sourceOffset, VkBuffer destination, VkDeviceSize destinationOffset, VkDeviceSize bytes);

// Records a global memory barrier (Metal::CommandStream::Barrier with the stages mapped; the
// access masks have no Metal counterpart).
void RecordMemoryBarrier(const Context& context, VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess);

// Vulkan only: Metal has no entry points to resolve; leaves `functions` empty.
void FillDeviceFunctions(const Context& context, DeviceFunctions& functions);

// A colour render target image. macOS: not implemented yet (needs the Metal format table of work
// package D4 and the render passes of D6); the constructor throws.
class RenderTarget {
public:
    RenderTarget(const Context& context, const ColorTarget& target, bool blending);
    ~RenderTarget();
    RenderTarget(const RenderTarget&) = delete;
    RenderTarget& operator=(const RenderTarget&) = delete;
    VkImage Image() const;
    VkImageView View() const;

private:
    Context context;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

// A one-shot command buffer outside the recorder: its own MTL4 command buffer, allocator and
// command stream, committed to the device's queue and waited for through its own MTLSharedEvent.
// Recorded batches are submitted first, so it sees their results in queue order.
class CommandBatch {
public:
    explicit CommandBatch(const Context& context);
    ~CommandBatch();
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;
    VkCommandBuffer Handle() const;
    void SubmitAndWait();
    void Submit();
    void Wait();

private:
    void release() noexcept;
    Context context;
    std::unique_ptr<Metal::CommandBatchObjects, Metal::ObjectsDeleter> metal;
    bool pending = false;
    bool submitted = false;
};

}

#endif
