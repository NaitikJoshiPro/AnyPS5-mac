// Metal twin of tests/execution/PooledBufferRange.cpp (agc_driver_pooled_buffer_range), swapped in by
// twins/TWINB-tests.cmake. Upstream checks that a pooled buffer exposes exactly the bytes it was asked for,
// although its memory is a whole size class that an earlier user may have left bytes in: (1) a robust
// shader read past the end of a 100-byte pooled buffer bound with VK_WHOLE_SIZE returns zero or its own
// bytes, never the earlier user's; (2) each request makes a VkBuffer of exactly the requested size, over
// fresh or pooled class memory. On Metal (specifics/macos/src/BufferPool.mm, Resources.mm) an allocation
// is one MTLBuffer of the class length that the pool hands out again as it is, and the requested range is
// the Buffer's own size (Bytes(), Size()). So the twin checks, with upstream's sizes and classes:
// (2') a request takes a fresh MTLBuffer of its class only when the class has none retained, a smaller
// request of the same class reuses the retained one, and every buffer reports the bytes requested, not its
// class; (1') a GPU copy of a reused 100-byte buffer's range moves only its 100 fresh bytes, none of the
// earlier user's bytes that remain in the class tail. The VK_WHOLE_SIZE descriptor read of (1) has no Metal
// counterpart: pooled buffers never reach a guest shader through a sized descriptor (guest buffers are
// bound by guest address, bounded by their V# and the device page table; MetalDevice.mm AddressExtras), so
// it is excluded with that reason (TechnicalDebt metal-vulkan-internal-tests).
// Upstream's later cases (5d9dcc42d2..0d078f75d3) intercept Vulkan entry points through Context.deviceProc and have
// no Metal counterpart either: SlabMetadataFailures (a4485bd82: slab metadata allocation failures after
// vkAllocateMemory/vkMapMemory; the Metal pool has no slabs), DeviceBindingFailures (432fe5cf9: a failed
// vkBindBufferMemory; a Metal buffer is created with its memory in one call) and RunCommandBatchFailureTests
// (tests/CommandBatchFailure.cpp: vkEndCommandBuffer/vkQueueSubmit faults and fence waits counted through
// deviceProc). The Metal pool's allocation-failure path (trim, then Context::cacheReclaim) is checked by
// agc_buffer_pool (BufferPoolMetal.mm CacheReclaimRecovery).

#include "MetalTestDevice.h"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>

namespace {

using namespace AgcDriver::Graphics;

constexpr std::uint32_t Stale = 0xababababu;
constexpr std::uint32_t Fresh = 0x11111111u;
constexpr std::uint32_t Untouched = 0x5a5a5a5au;
constexpr VkMemoryPropertyFlags HostMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr std::size_t RequestedBytes = 100;
constexpr std::uint32_t PastEnd = 28;

// A Context with a pool of its own, as upstream makes a fresh device for the test.
Context OwnPool(const Context& device) {
    Context context = device;
    context.bufferPool = std::make_shared<BufferPool>(context);
    return context;
}

std::size_t ClassOf(VkBuffer buffer) {
    return static_cast<std::size_t>(Metal::BufferOf(buffer).length);
}

std::uint64_t Made(const Context& context) {
    return GetBufferPool(context)->Counts().made;
}

void Fill(std::span<std::byte> bytes, std::uint32_t word) {
    for (std::size_t offset = 0; offset + sizeof(word) <= bytes.size(); offset += sizeof(word)) std::memcpy(bytes.data() + offset, &word, sizeof(word));
}

void RangePastEnd(const Context& device) {
    const auto context = OwnPool(device);
    constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const auto classBytes = BufferPool::Capacity(RequestedBytes, HostMemory);
    Require(classBytes > (PastEnd + 1) * sizeof(std::uint32_t), "the requested buffer's size class does not reach the word read past its end");
    VkBuffer earlierHandle = VK_NULL_HANDLE;
    {
        Buffer earlier(context, classBytes, usage);
        earlierHandle = earlier.Handle();
        Fill(earlier.Bytes(), Stale);
    }
    Buffer source(context, RequestedBytes, usage);
    Require(source.Handle() == earlierHandle, "the 100-byte buffer did not reuse the retained buffer of its class, so nothing stale is behind its end");
    const auto bytes = source.Bytes();
    Require(bytes.size() == RequestedBytes, "the pooled buffer does not span the requested bytes");
    Fill(bytes, Fresh);
    std::uint32_t tail = 0;
    std::memcpy(&tail, static_cast<const std::byte*>(Metal::BufferOf(source.Handle()).contents) + PastEnd * sizeof(tail), sizeof(tail));
    Require(tail == Stale, "the class tail of the reused buffer does not hold the earlier user's bytes (the check below would prove nothing)");
    Buffer destination(context, classBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Fill(destination.Bytes(), Untouched);
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        CommandBatch batch(context);
        CopyBuffer(context, batch.Handle(), source.Handle(), 0, destination.Handle(), 0, source.Bytes().size());
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
    }
    destination.Invalidate();
    const auto copied = destination.Bytes();
    for (std::size_t offset = 0; offset + sizeof(std::uint32_t) <= copied.size(); offset += sizeof(std::uint32_t)) {
        std::uint32_t value = 0;
        std::memcpy(&value, copied.data() + offset, sizeof(value));
        const auto want = offset + sizeof(value) <= RequestedBytes ? Fresh : Untouched;
        char text[200];
        std::snprintf(text, sizeof(text), "word %zu of a GPU copy of a %zu-byte pooled buffer's range is 0x%08x, not 0x%08x (0x%08x: the bytes a %zu-byte earlier user left in its memory)", offset / 4u, RequestedBytes, value, want, Stale, classBytes);
        Require(value == want, text);
    }
    std::puts("range past end: a reused pooled buffer's range holds only its own bytes");
}

void ExactSizes(const Context& device) {
    const auto context = OwnPool(device);
    constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto hostClass = BufferPool::Capacity(100, HostMemory);
    const auto deviceClass = BufferPool::Capacity(3000, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Require(BufferPool::Capacity(80, HostMemory) == hostClass && hostClass != 80, "80 and 100 bytes are not one host size class");
    Require(BufferPool::Capacity(2500, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == deviceClass && deviceClass != 2500, "2500 and 3000 bytes are not one device size class");
    const auto expect = [&](std::uint64_t before, std::uint64_t made, VkBuffer handle, std::size_t length, std::size_t range, std::size_t requested, const std::string& what) {
        const auto now = Made(context);
        Require(now - before == made, what + ": made " + std::to_string(now - before) + " MTLBuffers, not " + std::to_string(made));
        Require(ClassOf(handle) == length, what + ": its MTLBuffer has " + std::to_string(ClassOf(handle)) + " bytes, not " + std::to_string(length));
        Require(range == requested, what + ": it spans " + std::to_string(range) + " bytes, not the " + std::to_string(requested) + " requested");
    };
    VkBuffer hostHandle = VK_NULL_HANDLE;
    {
        const auto before = Made(context);
        Buffer fresh(context, 100, usage);
        hostHandle = fresh.Handle();
        expect(before, 1, fresh.Handle(), hostClass, fresh.Bytes().size(), 100, "a fresh 100-byte buffer");
    }
    {
        const auto before = Made(context);
        Buffer pooled(context, 80, usage);
        expect(before, 0, pooled.Handle(), hostClass, pooled.Bytes().size(), 80, "an 80-byte buffer taken from the " + std::to_string(hostClass) + "-byte class");
        Require(pooled.Handle() == hostHandle, "an 80-byte buffer did not reuse the retained buffer of its class");
    }
    {
        const auto before = Made(context);
        Buffer again(context, 80, usage);
        expect(before, 0, again.Handle(), hostClass, again.Bytes().size(), 80, "an 80-byte buffer reusing an 80-byte one");
    }
    VkBuffer deviceHandle = VK_NULL_HANDLE;
    {
        const auto before = Made(context);
        DeviceBuffer fresh(context, 3000, usage);
        deviceHandle = fresh.Handle();
        expect(before, 1, fresh.Handle(), deviceClass, fresh.Size(), 3000, "a fresh 3000-byte device buffer");
        Require(Metal::BufferOf(fresh.Handle()).storageMode == MTLStorageModePrivate, "a device buffer is not private storage");
    }
    {
        const auto before = Made(context);
        DeviceBuffer pooled(context, 2500, usage);
        expect(before, 0, pooled.Handle(), deviceClass, pooled.Size(), 2500, "a 2500-byte device buffer taken from the " + std::to_string(deviceClass) + "-byte class");
        Require(pooled.Handle() == deviceHandle, "a 2500-byte device buffer did not reuse the retained buffer of its class");
    }
    std::puts("exact sizes: requests take their class's buffer and span the requested bytes");
}

}

int main() {
    @autoreleasepool {
        try {
            MetalTest::Device device;
            std::printf("device %s\n", device.device.name.UTF8String);
            RangePastEnd(device.context);
            ExactSizes(device.context);
            std::puts("pooled buffer range tests passed");
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            return 1;
        }
    }
}
