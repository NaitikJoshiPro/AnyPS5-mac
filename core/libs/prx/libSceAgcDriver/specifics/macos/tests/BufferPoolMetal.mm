// Metal twin of tests/BufferPool.cpp (work package D2): the size classes, tiers, LRU budget and
// reuse rules of the buffer pool on Metal buffers, plus what is new on Metal: every allocation is
// resident on the queue (a GPU fill through a CommandBatch lands in a pooled buffer), device-local
// buffers are private storage, every buffer has a GPU address. The upstream test's slab cases
// have no Metal counterpart (the pool has no slabs, see BufferPool.hpp). Neither has upstream's HostPlacement
// (d04e98320): a device buffer that does not fit the VK_EXT_memory_budget budget of a discrete device-local heap goes
// to a memory type on the host heap and skips the device tier on release. Apple GPUs have unified memory: one heap
// (recommendedMaxWorkingSetSize, BufferPool.mm), private and shared storage both live in it, so there is no other
// heap to place into and nothing for the case to observe.
// Upstream 5d9dcc42d2..0d078f75d3 cases: SmallDeviceRetention (27ded2c4d: small device buffers stay in the small
// tier unless APS5_BUFFER_POOL_SHARED, also run alone as --small-device-retention under the environments of
// agc_buffer_pool_small_device_shared / _no_staging) and CacheReclaimRecovery (8cda58273: after the trim, the
// texture caches are asked once for the request plus its reserve, and only a release that freed bytes is retried)
// are ported; Metal cannot be made to refuse an allocation on demand, so the refusal is a buffer over the device's
// maxBufferLength. SlabMetadataFailure, SlabMapFailure (a4485bd82: slabs) and DeviceBindingFailures (432fe5cf9:
// vkBindBufferMemory) have no Metal counterpart: the pool has no slabs and a Metal buffer is created with its memory
// in one call, so there is no separate mapping or binding to fail.
// Upstream 0d078f75d3..661f8b7171: AllocationFailureReportsHeapBudgets (e6a78f41bf, 415dcab140, c5f8a3ff22) is
// ported for the one Metal heap. The slab return case (5001f87bc9) has no counterpart (no slabs), nor have the host
// placements of refused device-local memory (ad11cb1b3e, 9b2339e735, cfde713e10): unified memory, one heap.
// Upstream 661f8b7171..6b6c701636: ImageMemoryReuse (4513d47104, bafc97bb1e) is ported over the pool's image heaps
// (placement MTLHeaps, BufferPool.hpp) at a quarter of the sizes (1 MiB images under the 16 MiB image budget that
// APS5_STAGING_POOL_MIB=64 gives), plus a texture placed in a pooled heap; its last step (an allocation failure frees
// the retained image memory) is not, nor are ImageMemoryReclaim (a refused allocation recovered by the image a cache
// release returns) and ImageMemoryOverBudget (the heap's reported usage over its budget): Metal cannot be made to
// refuse a heap on demand, nor its allocated bytes to exceed the recommended working set
// (TechnicalDebt metal-image-memory-pool-tests).

#include "MetalTestDevice.h"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
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

// Upstream 27ded2c4d: churning large device buffers through the device tier's budget keeps a retained small device
// buffer (it lives in the small tier), unless APS5_BUFFER_POOL_SHARED puts every device buffer in one tier. The
// budget is the pool's own (APS5_STAGING_POOL_MIB; 0 disables the device tier, and the large tier's 512 MiB evicts).
void SmallDeviceRetention(const Context& device) {
    const auto context = Fresh(device);
    const auto pool = GetBufferPool(context);
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto deviceBudget = pool->Counts().device.budget;
    const auto bytes = static_cast<std::size_t>((deviceBudget != 0 ? deviceBudget : 512 * MiB) / 2) + MiB;
    VkBuffer small = VK_NULL_HANDLE;
    {
        DeviceBuffer buffer(context, 256, usage);
        small = buffer.Handle();
    }
    const auto destroyed = pool->Counts().destroyed;
    {
        std::vector<std::unique_ptr<DeviceBuffer>> large;
        for (int i = 0; i < 2; ++i) large.push_back(std::make_unique<DeviceBuffer>(context, bytes, usage));
    }
    Expect(pool->Counts().destroyed != destroyed, "large device buffers did not exercise pool budget eviction");
    const auto made = pool->Counts().made;
    const bool isolated = std::getenv("APS5_BUFFER_POOL_SHARED") == nullptr;
    DeviceBuffer again(context, 256, usage);
    // A destroyed buffer's handle may be reused by the next allocation, so the shared case is judged by the count.
    if (isolated) Expect(again.Handle() == small, "small device buffer retention did not follow the separate pool budget");
    Expect(pool->Counts().made == made + (isolated ? 0 : 1), "large device buffer churn changed small buffer allocation reuse");
    const auto counts = pool->Counts();
    Expect(counts.resident == counts.made - counts.destroyed, "device pool tier retention left " + std::to_string(counts.resident) + " allocations resident, not the " + std::to_string(counts.made - counts.destroyed) + " alive or retained");
}

struct ReclaimProbe {
    std::uint64_t requests = 0;
    VkDeviceSize asked = 0;
    VkDeviceSize released = 0;
};

ReclaimProbe reclaimProbe;

VkDeviceSize ProbeCacheReclaim(VkDeviceSize bytes) {
    ++reclaimProbe.requests;
    reclaimProbe.asked = bytes;
    return reclaimProbe.released;
}

// Upstream 8cda58273 (Context::cacheReclaim, 0f1564cc2): an allocation the trim did not satisfy asks the texture
// caches once for ReclaimTarget of its size and is retried only when they released something; the failure reports
// both the trimmed and the released bytes. Metal refuses a buffer longer than maxBufferLength, so that request fails
// every attempt and the attempt count shows the retries.
void CacheReclaimRecovery(const Context& device, std::uint64_t maxBufferLength) {
    Expect(BufferPool::ReclaimTarget(4 * MiB) == 260 * MiB, "a 4 MiB request does not ask the caches for 256 MiB on top of itself");
    Expect(BufferPool::ReclaimTarget(512 * MiB) == 1024 * MiB, "a 512 MiB request does not ask the caches for twice its size");
    constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const auto refused = static_cast<std::size_t>(maxBufferLength) + MiB;
    const auto target = BufferPool::ReclaimTarget(BufferPool::Capacity(refused, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    const auto withReclaim = [&] {
        Context context = device;
        context.cacheReclaim = ProbeCacheReclaim;
        context.bufferPool = std::make_shared<BufferPool>(context);
        return context;
    };
    const auto attempt = [&](const Context& context) {
        try {
            DeviceBuffer buffer(context, refused, usage);
        } catch (const std::exception& error) {
            return std::string(error.what());
        }
        return std::string();
    };
    const auto contains = [](const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; };
    {
        reclaimProbe = {};
        reclaimProbe.released = 10 * MiB;
        const auto context = withReclaim();
        { DeviceBuffer idle(context, 2 * MiB, usage); }
        const auto before = GetBufferPool(context)->Counts();
        const auto failure = attempt(context);
        const auto after = GetBufferPool(context)->Counts();
        Expect(!failure.empty(), "Metal allocated a buffer over maxBufferLength");
        Expect(after.destroyed == before.destroyed + 1 && after.device.slots == 0, "the trim before the cache release did not free the retained buffer");
        Expect(reclaimProbe.requests == 1 && reclaimProbe.asked == target, "the caches were not asked once for the request plus its reserve (" + std::to_string(reclaimProbe.requests) + " requests, last for " + std::to_string(reclaimProbe.asked) + " bytes)");
        Expect(contains(failure, "released " + std::to_string(10 * MiB) + " cache bytes") && contains(failure, "attempts 3"), "an allocation the pool's own slots could not satisfy did not retry once after the trim and once after the cache release: " + failure);
    }
    {
        reclaimProbe = {};
        reclaimProbe.released = 10 * MiB;
        const auto context = withReclaim();
        const auto failure = attempt(context);
        Expect(reclaimProbe.requests == 1 && contains(failure, "reclaimed 0 bytes") && contains(failure, "attempts 2"), "an empty pool did not go straight to the cache release and retry once: " + failure);
    }
    {
        reclaimProbe = {};
        const auto context = withReclaim();
        const auto failure = attempt(context);
        Expect(reclaimProbe.requests == 1 && contains(failure, "released 0 cache bytes") && contains(failure, "attempts 1"), "a cache release that freed nothing was retried after, or the failure lost the released bytes: " + failure);
        DeviceBuffer later(context, 2 * MiB, usage);
        Expect(later.Handle() != VK_NULL_HANDLE && reclaimProbe.requests == 1, "an allocation after the refused one asked the caches for memory");
    }
}

// Upstream e6a78f41bf (AllocationFailureReportsHeapBudgets): a refused allocation reports each heap's usage against
// its budget. Metal has one heap (the device's allocated bytes against recommendedMaxWorkingSetSize) and always
// knows both, so upstream's second case (no VK_EXT_memory_budget, no heap data) has no counterpart.
void AllocationFailureReportsHeapBudgets(const Context& device, std::uint64_t maxBufferLength, std::uint64_t workingSet) {
    const auto context = Fresh(device);
    std::string failure;
    try {
        DeviceBuffer buffer(context, static_cast<std::size_t>(maxBufferLength) + MiB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    Expect(!failure.empty(), "Metal allocated a buffer over maxBufferLength");
    Expect(failure.find("; heap 0 ") != std::string::npos && failure.find("/" + std::to_string(workingSet >> 20u) + " MiB") != std::string::npos, "allocation failure omitted the heap's usage against its budget: " + failure);
    Expect(failure.find("heap 1") == std::string::npos, "allocation failure reported a second heap on unified memory: " + failure);
}

// Upstream ImageMemoryReuse (4513d47104, bafc97bb1e): released private image memory is retained by type and size and
// handed to the next image of that type and size, shared image memory is freed, and retained images are evicted least
// recently used under the image budget (16 MiB here: a quarter of the 64 MiB device tier).
void ImageMemoryReuse(const Context& device) {
    const auto context = Fresh(device);
    const auto pool = GetBufferPool(context);
    const auto base = pool->Counts();
    const auto made = [&] { return pool->Counts().made - base.made; };
    const auto freed = [&] { return pool->Counts().destroyed - base.destroyed; };
    Expect(base.images.budget == 16 * MiB, "the image budget is " + std::to_string(base.images.budget / MiB) + " MiB, not a quarter of the 64 MiB device tier");
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = MiB;
    allocation.memoryTypeIndex = BufferPool::PrivateImageMemory;
    std::uint32_t type = 99;
    const auto first = AllocateImageMemory(context, allocation, 1u, "test image", &type);
    Expect(first != VK_NULL_HANDLE && type == BufferPool::PrivateImageMemory && made() == 1, "the first image allocates device memory of its type");
    ReleaseImageMemory(context, first, MiB, BufferPool::PrivateImageMemory);
    Expect(freed() == 0 && pool->RetainedImages() == 1, "a released device-local image allocation is retained");
    const auto again = AllocateImageMemory(context, allocation, 1u, "test image", &type);
    Expect(again == first && type == BufferPool::PrivateImageMemory && made() == 1 && pool->RetainedImages() == 0, "the next image of the same type and size takes the retained allocation");
    auto smaller = allocation;
    smaller.allocationSize = MiB / 2;
    const auto other = AllocateImageMemory(context, smaller, 1u, "test image", &type);
    Expect(other != again && made() == 2, "an image of another size allocates");
    ReleaseImageMemory(context, other, MiB / 2, BufferPool::PrivateImageMemory);
    ReleaseImageMemory(context, again, MiB, BufferPool::PrivateImageMemory);
    auto host = allocation;
    host.memoryTypeIndex = BufferPool::SharedImageMemory;
    const auto placed = AllocateImageMemory(context, host, 2u, "test image", &type);
    Expect(type == BufferPool::SharedImageMemory && made() == 3 && ((__bridge id<MTLHeap>)reinterpret_cast<void*>(placed)).storageMode == MTLStorageModeShared, "a host-visible image allocation is made");
    ReleaseImageMemory(context, placed, MiB, BufferPool::SharedImageMemory);
    Expect(freed() == 1 && pool->RetainedImages() == 2, "a host-visible image allocation is freed when released");
    const auto unsized = AllocateImageMemory(context, smaller, 1u, "test image", &type);
    Expect(unsized == other && made() == 3 && pool->RetainedImages() == 1, "an image of a retained size takes that allocation");
    ReleaseImageMemory(context, unsized, 0, BufferPool::PrivateImageMemory);
    Expect(freed() == 2 && pool->RetainedImages() == 1, "an allocation of unknown size is freed");
    std::vector<VkDeviceMemory> many;
    for (int i = 0; i < 17; ++i) many.push_back(AllocateImageMemory(context, allocation, 1u, "test image", &type));
    Expect(many.front() == again && made() == 19, "a burst takes the retained allocation first and allocates the rest");
    for (const auto memory : many) ReleaseImageMemory(context, memory, MiB, BufferPool::PrivateImageMemory);
    Expect(pool->RetainedImages() == 16 && freed() == 3, "retained image memory is evicted under its budget (" + std::to_string(pool->RetainedImages()) + " retained, " + std::to_string(freed()) + " freed)");
    std::vector<VkDeviceMemory> retaken;
    for (int i = 0; i < 16; ++i) retaken.push_back(AllocateImageMemory(context, allocation, 1u, "test image", &type));
    Expect(made() == 19 && std::find(retaken.begin(), retaken.end(), many.front()) == retaken.end() && std::find(retaken.begin(), retaken.end(), many.back()) != retaken.end(), "retained image memory is not evicted least recently used");
    for (const auto memory : retaken) ReleaseImageMemory(context, memory, MiB, BufferPool::PrivateImageMemory);
    const auto counts = pool->Counts();
    Expect(counts.images.slots == 16 && counts.images.retainedBytes == 16 * MiB && counts.images.evictions == 1, "the image counters disagree with the retained images");
    Expect(counts.resident == counts.made - counts.destroyed, "the pool's residency set holds " + std::to_string(counts.resident) + " allocations, not the " + std::to_string(counts.made - counts.destroyed) + " alive or retained");
    // A retained image heap never serves a device buffer of its size.
    DeviceBuffer buffer(context, MiB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Expect(pool->RetainedImages() == 16, "a device buffer took a retained image heap");
}

// The way an image is placed in its memory on Metal: a texture of the size heapTextureSizeAndAlignWithDescriptor
// gives fits at offset 0 of the heap AllocateImageMemory makes, and again in the same heap once it is retained.
void ImageInPooledHeap(const Context& device, id<MTLDevice> metal) {
    const auto context = Fresh(device);
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:256 height:256 mipmapped:YES];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    const auto sized = [metal heapTextureSizeAndAlignWithDescriptor:descriptor];
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = sized.size;
    allocation.memoryTypeIndex = BufferPool::PrivateImageMemory;
    const auto memory = AllocateImageMemory(context, allocation, 1u, "test image");
    id<MTLHeap> heap = (__bridge id<MTLHeap>)reinterpret_cast<void*>(memory);
    @autoreleasepool {
        id<MTLTexture> texture = [heap newTextureWithDescriptor:descriptor offset:0];
        Expect(texture != nil && texture.heap == heap, "a texture does not fit in image memory of its heap size");
    }
    ReleaseImageMemory(context, memory, sized.size, BufferPool::PrivateImageMemory);
    const auto again = AllocateImageMemory(context, allocation, 1u, "test image");
    Expect(again == memory, "the image heap was not retained for an image of its size");
    @autoreleasepool {
        id<MTLTexture> texture = [(__bridge id<MTLHeap>)reinterpret_cast<void*>(again) newTextureWithDescriptor:descriptor offset:0];
        Expect(texture != nil, "a retained image heap does not take the next image of its size");
    }
    ReleaseImageMemory(context, again, sized.size, BufferPool::PrivateImageMemory);
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

int main(int argc, char** argv) {
    // --small-device-retention runs that case alone under the environment ctest gives it (upstream's
    // agc_buffer_pool_small_device_shared and _no_staging).
    const bool retentionOnly = argc == 2 && std::string(argv[1]) == "--small-device-retention";
    // The device tier's budget for Budget(); read once, before the pool exists.
    if (!retentionOnly) setenv("APS5_STAGING_POOL_MIB", "64", 1);
    @autoreleasepool {
        MetalTest::Device device;
        try {
            if (retentionOnly) {
                SmallDeviceRetention(device.context);
                if (failures != 0) {
                    std::fprintf(stderr, "%d small device retention checks failed\n", failures);
                    return 1;
                }
                std::printf("small device retention tests passed\n");
                return 0;
            }
            SizeClassesAndDirections(device.context);
            LargerClass(device.context);
            KeptUntilRelease(device.context);
            SmallDeviceRetention(device.context);
            Budget(device.context);
            AddressAndHostUnchanged(device.context);
            SmallDeviceClasses(device.context);
            CacheReclaimRecovery(device.context, static_cast<std::uint64_t>(device.device.maxBufferLength));
            AllocationFailureReportsHeapBudgets(device.context, static_cast<std::uint64_t>(device.device.maxBufferLength), static_cast<std::uint64_t>(device.device.recommendedMaxWorkingSetSize));
            ResidentForTheGpu(device.context);
            ImageMemoryReuse(device.context);
            ImageInPooledHeap(device.context, device.device);
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
