// Metal twin of tests/BufferPool.cpp (work package D2): the size classes, tiers, LRU budget and
// reuse rules of the buffer pool on Metal buffers, plus what is new on Metal: every allocation is
// resident on the queue (a GPU fill through a CommandBatch lands in a pooled buffer), device-local
// buffers are private storage, every buffer has a GPU address. The upstream test's slab cases
// have no Metal counterpart (the pool has no slabs, see BufferPool.hpp).

#include "MetalTestDevice.h"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace AgcDriver::Graphics;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

constexpr std::size_t MiB = std::size_t{1} << 20u;
constexpr VkMemoryPropertyFlags HostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

// A Context with a pool of its own, as each upstream case makes a fresh mock device.
Context Fresh(const Context& device) {
    Context context = device;
    context.bufferPool = std::make_shared<BufferPool>(context);
    return context;
}

std::size_t ClassOf(VkBuffer buffer) {
    return static_cast<std::size_t>(Metal::BufferOf(buffer).length);
}

void SizeClassesAndDirections(const Context& device) {
    const auto context = Fresh(device);
    VkBuffer first = VK_NULL_HANDLE;
    {
        DeviceBuffer upload(context, 10 * MiB + 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        first = upload.Handle();
        Expect(ClassOf(upload.Handle()) == 11 * MiB, "a 10 MiB + 4 KiB device buffer has " + std::to_string(ClassOf(upload.Handle())) + " bytes, not its 11 MiB class");
        Expect(Metal::BufferOf(upload.Handle()).storageMode == MTLStorageModePrivate, "a device buffer is not private storage");
        Expect(upload.Size() == 10 * MiB + 4096, "a device buffer reports its class instead of the requested size");
    }
    const auto made = GetBufferPool(context)->Counts().made;
    DeviceBuffer writeBack(context, 10 * MiB + 512 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Expect(writeBack.Handle() == first && GetBufferPool(context)->Counts().made == made, "a retained 11 MiB device buffer did not serve a write-back of another size and direction in its class");
    Expect(writeBack.Size() == 10 * MiB + 512 * 1024, "a reused device buffer reports the retained size");
}

void LargerClass(const Context& device) {
    const auto context = Fresh(device);
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer twelve = VK_NULL_HANDLE;
    {
        DeviceBuffer large(context, 12 * MiB, usage);
        twelve = large.Handle();
    }
    {
        DeviceBuffer smaller(context, 7 * MiB, usage);
        Expect(smaller.Handle() == twelve, "a 7 MiB request did not take the retained 12 MiB buffer");
    }
    const auto made = GetBufferPool(context)->Counts().made;
    {
        DeviceBuffer tooSmall(context, 5 * MiB, usage);
        Expect(tooSmall.Handle() != twelve && GetBufferPool(context)->Counts().made == made + 1, "a 5 MiB request took a retained buffer of more than twice its size");
        Expect(ClassOf(tooSmall.Handle()) == 5 * MiB, "a 5 MiB device buffer was not created at its own class");
    }
    DeviceBuffer again(context, 12 * MiB, usage);
    Expect(again.Handle() == twelve, "a buffer served to a smaller request did not return to its own 12 MiB class");
    DeviceBuffer best(context, 4 * MiB + 1, usage);
    Expect(best.Handle() != twelve && ClassOf(best.Handle()) == 5 * MiB, "a 4 MiB + 1 request did not take the smallest retained class that fits (5 MiB)");
}

void KeptUntilRelease(const Context& device) {
    const auto context = Fresh(device);
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    std::vector<std::shared_ptr<DeviceBuffer>> batch;
    for (int i = 0; i < 3; ++i) batch.push_back(std::make_shared<DeviceBuffer>(context, 6 * MiB + static_cast<std::size_t>(i + 1) * 4096, usage));
    for (std::size_t i = 0; i < batch.size(); ++i) {
        for (std::size_t j = i + 1; j < batch.size(); ++j) Expect(batch[i]->Handle() != batch[j]->Handle(), "two buffers of one unfinished batch share a Metal buffer");
    }
    const auto kept = batch.front()->Handle();
    const auto made = GetBufferPool(context)->Counts().made;
    {
        DeviceBuffer during(context, 6 * MiB, usage);
        Expect(during.Handle() != kept && GetBufferPool(context)->Counts().made == made + 1, "a buffer kept by an unfinished batch was handed out again");
    }
    batch.clear();
    std::vector<std::unique_ptr<DeviceBuffer>> after;
    for (int i = 0; i < 3; ++i) after.push_back(std::make_unique<DeviceBuffer>(context, 6 * MiB + 100, usage));
    after.push_back(std::make_unique<DeviceBuffer>(context, 6 * MiB, usage));
    Expect(GetBufferPool(context)->Counts().made == made + 1, "the buffers released after their batch were not reused (" + std::to_string(GetBufferPool(context)->Counts().made - made - 1) + " new allocations)");
}

// The device tier's budget is 64 MiB here (APS5_STAGING_POOL_MIB, set by main before the pool).
void Budget(const Context& device) {
    const auto context = Fresh(device);
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    std::vector<std::unique_ptr<DeviceBuffer>> live;
    for (int i = 0; i < 12; ++i) live.push_back(std::make_unique<DeviceBuffer>(context, (6 + static_cast<std::size_t>(i)) * MiB + 4096, usage));
    const auto first = live.front()->Handle();
    std::size_t madeBytes = 0;
    for (const auto& buffer : live) madeBytes += ClassOf(buffer->Handle());
    Expect(madeBytes > 64 * MiB, "the budget case does not exceed the budget");
    const auto before = GetBufferPool(context)->Counts();
    for (auto& buffer : live) buffer.reset();
    const auto after = GetBufferPool(context)->Counts();
    Expect(after.device.budget == 64 * MiB, "the device tier's budget is " + std::to_string(after.device.budget / MiB) + " MiB, not the 64 MiB APS5_STAGING_POOL_MIB asked for");
    Expect(after.destroyed > before.destroyed && after.device.retainedBytes <= 64 * MiB, "the device tier retains " + std::to_string(after.device.retainedBytes / MiB) + " MiB after " + std::to_string(after.destroyed - before.destroyed) + " evictions, over its 64 MiB budget");
    DeviceBuffer probe(context, 6 * MiB + 4096, usage);
    Expect(probe.Handle() != first, "the least recently used device buffer survived the eviction");
}

void AddressAndHostUnchanged(const Context& device) {
    const auto context = Fresh(device);
    VkBuffer scratch = VK_NULL_HANDLE;
    {
        DeviceBuffer buffer(context, 2 * MiB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        scratch = buffer.Handle();
    }
    // Every Metal buffer has a GPU address: an addressable device-local buffer may take the scratch.
    Buffer addressed(context, 2 * MiB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Expect(addressed.Handle() == scratch && addressed.DeviceAddress() == Metal::BufferOf(scratch).gpuAddress && !addressed.Mapped(), "an addressable device-local buffer did not take the retained scratch buffer of its class, or reports another address or a mapping");
    VkBuffer host = VK_NULL_HANDLE;
    {
        Buffer copy(context, 10 * MiB + 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HostVisible);
        host = copy.Handle();
        Expect(ClassOf(host) == 10 * MiB + 4096, "a large host buffer was rounded to a class");
        Expect(copy.Mapped() && copy.Bytes().size() == 10 * MiB + 4096 && Metal::BufferOf(host).storageMode == MTLStorageModeShared, "a host buffer is not mapped shared storage of its size");
    }
    Buffer other(context, 10 * MiB + 8192, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HostVisible);
    Expect(other.Handle() != host, "a large host buffer of another exact size was reused");
    Buffer same(context, 10 * MiB + 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HostVisible);
    Expect(same.Handle() == host, "a large host buffer of the same exact size was not reused");
}

void SmallDeviceClasses(const Context& device) {
    const auto context = Fresh(device);
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer mebibyte = VK_NULL_HANDLE;
    VkBuffer half = VK_NULL_HANDLE;
    {
        DeviceBuffer one(context, MiB, usage);
        mebibyte = one.Handle();
    }
    {
        DeviceBuffer smaller(context, 300 * 1024, usage);
        Expect(smaller.Handle() != mebibyte && ClassOf(smaller.Handle()) == 512 * 1024, "a 300 KiB request took a 1 MiB buffer or skipped its power-of-two class");
        half = smaller.Handle();
    }
    DeviceBuffer again(context, 260 * 1024, usage);
    Expect(again.Handle() == half, "a 260 KiB request did not take the retained 512 KiB buffer");
    constexpr auto indirect = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer tiny = VK_NULL_HANDLE;
    {
        DeviceBuffer args(context, 20, indirect);
        tiny = args.Handle();
    }
    const auto made = GetBufferPool(context)->Counts().made;
    DeviceBuffer nextArgs(context, 20, indirect);
    Expect(nextArgs.Handle() == tiny && GetBufferPool(context)->Counts().made == made, "a 20-byte device request did not reuse the retained 256-byte buffer of its class");
}

// Metal: pooled buffers are resident on the queue (a GPU copy into them lands), and every
// allocation alive or retained is in the pool's residency set.
void ResidentForTheGpu(const Context& device) {
    const auto context = Fresh(device);
    Buffer source(context, 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    Buffer target(context, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    DeviceBuffer scratch(context, 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto words = std::span(reinterpret_cast<std::uint32_t*>(source.Bytes().data()), 1024);
    for (std::uint32_t i = 0; i < words.size(); ++i) words[i] = 0xc0de0000u + i;
    std::memset(target.Bytes().data(), 0, 4096);
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        CommandBatch batch(context);
        // Host -> private -> host: both pooled storage modes take part.
        CopyBuffer(context, batch.Handle(), source.Handle(), 0, scratch.Handle(), 0, 4096);
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        CopyBuffer(context, batch.Handle(), scratch.Handle(), 16, target.Handle(), 0, 4096 - 16);
        batch.SubmitAndWait();
    }
    const auto* result = reinterpret_cast<const std::uint32_t*>(target.Bytes().data());
    bool same = true;
    for (std::uint32_t i = 0; i < 1024 - 4; ++i) same = same && result[i] == 0xc0de0000u + i + 4;
    Expect(same && result[1020] == 0, "a copy through pooled shared and private buffers did not land (first word " + std::to_string(result[0]) + ")");
    const auto counts = GetBufferPool(context)->Counts();
    Expect(counts.resident == counts.made - counts.destroyed, "the pool's residency set holds " + std::to_string(counts.resident) + " allocations, not the " + std::to_string(counts.made - counts.destroyed) + " alive or retained");
}

}

int main() {
    // The device tier's budget for Budget(); read once, before the pool exists.
    setenv("APS5_STAGING_POOL_MIB", "64", 1);
    @autoreleasepool {
        MetalTest::Device device;
        try {
            SizeClassesAndDirections(device.context);
            LargerClass(device.context);
            KeptUntilRelease(device.context);
            Budget(device.context);
            AddressAndHostUnchanged(device.context);
            SmallDeviceClasses(device.context);
            ResidentForTheGpu(device.context);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 1;
        }
        if (failures != 0) {
            std::fprintf(stderr, "%d buffer pool checks failed\n", failures);
            return 1;
        }
        std::printf("buffer pool tests passed\n");
    }
    return 0;
}
