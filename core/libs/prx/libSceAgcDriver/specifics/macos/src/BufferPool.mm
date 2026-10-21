// The buffer pool on Metal (work package D2): twin of Graphics/src/BufferPool.cpp. The size
// classes, tiers, budgets and least-recently-used eviction are the upstream ones; an allocation is
// a MTLBuffer (shared storage when host-visible, private when device-local) kept resident through
// the pool's residency set, which is attached to the device's queue. Image memory (upstream
// 4513d47104, bafc97bb1e) is a placement MTLHeap in the same residency set, retained in the device
// tier by type and size once released (BufferPool.hpp).

#include "MetalCommandStream.h"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "MetalResidency.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace AgcDriver::Graphics {

namespace Metal {

struct PoolObjects {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    id<MTLResidencySet> residency = nil;
    // Residency-set changes (allocation and destruction happen outside the pool's mutex).
    std::mutex residencyMutex;
    std::uint64_t made = 0;
    std::uint64_t destroyed = 0;
    ~PoolObjects() {
        if (queue != nil && residency != nil) ::AgcDriver::Metal::Residency::Detach(queue, residency);
    }
};

void ObjectsDeleter::operator()(PoolObjects* objects) const noexcept {
    delete objects;
}

}

namespace {

// APS5_BUFFER_POOL_SHARED=1: one tier for every size, as before the split.
bool sharedTiers() {
    static const bool shared = std::getenv("APS5_BUFFER_POOL_SHARED") != nullptr;
    return shared;
}

bool exactSizes() {
    static const bool exact = std::getenv("APS5_NO_BUFFER_CLASSES") != nullptr;
    return exact;
}

bool systemMemoryHeap(const VkPhysicalDeviceMemoryProperties& memory, std::uint32_t heap) {
    bool typed = false;
    for (std::uint32_t type = 0; type < std::min<std::uint32_t>(memory.memoryTypeCount, VK_MAX_MEMORY_TYPES); ++type) {
        if (memory.memoryTypes[type].heapIndex != heap) continue;
        if ((memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) return false;
        typed = true;
    }
    return typed;
}

constexpr VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

}

BufferPool::BufferPool(const Context& context) : maxSlots(std::max<std::size_t>(1, MaxSlots())), cacheReclaim(context.cacheReclaim), metal(new Metal::PoolObjects) {
    metal->device = Metal::DeviceOf(context);
    if (metal->device == nil) Metal::Fail("the buffer pool needs a Context carrying a Metal device (a Vulkan context reached the Metal buffer pool)");
    metal->queue = Metal::QueueOf(context);
    MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
    descriptor.label = @"AGC buffer pool";
    descriptor.initialCapacity = 256;
    NSError* error = nil;
    metal->residency = [metal->device newResidencySetWithDescriptor:descriptor error:&error];
    if (metal->residency == nil) Metal::Fail("the buffer pool's residency set: " + Metal::Describe(error));
    // A compile-only device has no queue: its buffers are never used by the GPU.
    if (metal->queue != nil) ::AgcDriver::Metal::Residency::Attach(metal->queue, metal->residency);
    smallTier.budget = smallBudget;
    largeTier.budget = budget;
    // The device-local heap of a unified-memory Mac is its recommended working set, and its memory
    // is host-visible system memory (upstream 878c30378: a smaller device tier on such heaps).
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 1;
    memory.memoryHeaps[0] = {static_cast<VkDeviceSize>(metal->device.recommendedMaxWorkingSetSize), VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    memory.memoryTypeCount = 1;
    memory.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0};
    deviceTier.budget = DeviceTierEnabled() ? DeviceBudget(memory) : 0;
    imageBudget = ImageBudget(deviceTier.budget);
}

BufferPool::~BufferPool() {
    for (auto* tier : {&smallTier, &largeTier, &deviceTier}) {
        for (const auto& [key, slots] : tier->free) {
            for (const auto& slot : slots) destroy(slot.allocation);
        }
    }
}

BufferAllocation BufferPool::Allocate(std::size_t capacity, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    const bool host = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    const MTLResourceOptions options = MTLResourceHazardTrackingModeUntracked | (host ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate);
    const auto length = std::max<std::size_t>(capacity, 1);
    id<MTLBuffer> buffer = [metal->device newBufferWithLength:length options:options];
    VkDeviceSize reclaimed = 0;
    VkDeviceSize released = 0;
    unsigned attempts = 1;
    if (buffer == nil) {
        reclaimed = trim();
        if (reclaimed != 0) {
            buffer = [metal->device newBufferWithLength:length options:options];
            ++attempts;
        }
    }
    // Upstream 8cda58273: release the texture caches when trimming the pool was not enough. The
    // image memory the release hands back to the pool is freed before the retry (upstream 4513d47104).
    if (buffer == nil) {
        if (cacheReclaim != nullptr) released = cacheReclaim(ReclaimTarget(length));
        if (released != 0) {
            reclaimed += trim();
            buffer = [metal->device newBufferWithLength:length options:options];
            ++attempts;
        }
    }
    if (buffer != nil && attempts > 1) std::fprintf(stderr, "[bufferpool] buffer allocation recovered: allocation=%zu reclaimed=%llu released=%llu attempts=%u\n", length, static_cast<unsigned long long>(reclaimed), static_cast<unsigned long long>(released), attempts);
    if (buffer == nil) {
        // Upstream e6a78f41bf: the failure reports the heap budgets. Metal has one heap (unified memory):
        // the device's allocated bytes against its recommended working set, the budget DeviceBudget uses.
        const auto used = static_cast<unsigned long long>(metal->device.currentAllocatedSize) >> 20u;
        const auto heapBudget = static_cast<unsigned long long>(metal->device.recommendedMaxWorkingSetSize) >> 20u;
        Metal::Fail("cannot allocate a " + std::to_string(length) + "-byte " + (host ? "shared" : "private") + " buffer (reclaimed " + std::to_string(reclaimed) + " bytes and released " + std::to_string(released) + " cache bytes first, attempts " + std::to_string(attempts) + "); heap 0 " + std::to_string(used) + "/" + std::to_string(heapBudget) + " MiB");
    }
    {
        std::lock_guard lock(metal->residencyMutex);
        ::AgcDriver::Metal::Residency::Add(metal->residency, buffer);
        ::AgcDriver::Metal::Residency::Commit(metal->residency);
        ++metal->made;
    }
    BufferAllocation allocation{};
    allocation.buffer = reinterpret_cast<VkBuffer>(const_cast<void*>(CFBridgingRetain(buffer)));
    allocation.mapping = host ? buffer.contents : nullptr;
    allocation.address = buffer.gpuAddress;
    allocation.allocationBytes = buffer.length;
    allocation.bytes = capacity;
    allocation.usage = usage;
    allocation.properties = properties;
    return allocation;
}

VkDeviceSize BufferPool::DeviceBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported) {
    static const char* value = std::getenv("APS5_STAGING_POOL_MIB");
    if (value != nullptr) return static_cast<VkDeviceSize>(std::strtoull(value, nullptr, 10)) << 20u;
    std::optional<std::uint32_t> largest;
    for (std::uint32_t heap = 0; heap < memory.memoryHeapCount && heap < VK_MAX_MEMORY_HEAPS; ++heap) {
        if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 && (!largest || memory.memoryHeaps[heap].size > memory.memoryHeaps[*largest].size)) largest = heap;
    }
    VkDeviceSize size = largest ? memory.memoryHeaps[*largest].size : 0;
    if (largest && reported != nullptr && reported->heapBudget[*largest] != 0) size = std::min(size, reported->heapBudget[*largest]);
    if (largest && systemMemoryHeap(memory, *largest)) return std::max<VkDeviceSize>(VkDeviceSize{128} << 20u, size / 64u);
    return std::max<VkDeviceSize>(VkDeviceSize{512} << 20u, size / 8u);
}

VkDeviceSize BufferPool::ImageBudget(VkDeviceSize deviceBudget) {
    static const char* value = std::getenv("APS5_IMAGE_POOL_MIB");
    if (value != nullptr) return static_cast<VkDeviceSize>(std::strtoull(value, nullptr, 10)) << 20u;
    return deviceBudget / 4u;
}

bool BufferPool::imagePooled(VkDeviceSize bytes, std::uint32_t memoryType) const {
    return imageBudget != 0 && bytes != 0 && bytes <= imageBudget && bytes <= deviceTier.budget && memoryType == PrivateImageMemory && DeviceTierEnabled();
}

// Metal reports one heap (unified memory): over budget while the device's allocated bytes exceed its
// recommended working set (the budget BufferPool::DeviceBudget reads). Read at most every 100 ms.
bool BufferPool::heapOverBudget() {
    const auto now = std::chrono::steady_clock::now();
    if (heapRead != std::chrono::steady_clock::time_point{} && now - heapRead < std::chrono::milliseconds(100)) return heapOver;
    heapRead = now;
    const auto budget = static_cast<std::uint64_t>(metal->device.recommendedMaxWorkingSetSize);
    heapOver = budget != 0 && static_cast<std::uint64_t>(metal->device.currentAllocatedSize) > budget;
    return heapOver;
}

VkDeviceMemory BufferPool::allocateImageHeap(VkDeviceSize bytes, std::uint32_t memoryType, const char* operation) {
    if (memoryType != PrivateImageMemory && memoryType != SharedImageMemory) Metal::Fail(std::string(operation) + ": image memory type " + std::to_string(memoryType) + " is not a Metal image memory type (BufferPool::PrivateImageMemory or SharedImageMemory)");
    if (bytes == 0) Metal::Fail(std::string(operation) + ": image memory of 0 bytes");
    MTLHeapDescriptor* descriptor = [MTLHeapDescriptor new];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.size = static_cast<NSUInteger>(bytes);
    descriptor.storageMode = memoryType == SharedImageMemory ? MTLStorageModeShared : MTLStorageModePrivate;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    const auto make = [&] { return [metal->device newHeapWithDescriptor:descriptor]; };
    id<MTLHeap> heap = make();
    VkDeviceSize reclaimed = 0;
    VkDeviceSize released = 0;
    unsigned attempts = 1;
    if (heap == nil) {
        reclaimed = trim();
        if (reclaimed != 0) {
            heap = make();
            ++attempts;
        }
    }
    if (heap == nil) {
        if (cacheReclaim != nullptr) released = cacheReclaim(ReclaimTarget(bytes));
        if (released != 0) {
            reclaimed += trim();
            heap = make();
            ++attempts;
        }
    }
    if (heap != nil && attempts > 1) std::fprintf(stderr, "[bufferpool] %s recovered: allocation=%llu reclaimed=%llu released=%llu attempts=%u\n", operation, static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(reclaimed), static_cast<unsigned long long>(released), attempts);
    if (heap == nil) {
        const auto used = static_cast<unsigned long long>(metal->device.currentAllocatedSize) >> 20u;
        const auto heapBudget = static_cast<unsigned long long>(metal->device.recommendedMaxWorkingSetSize) >> 20u;
        Metal::Fail(std::string(operation) + ": cannot allocate a " + std::to_string(bytes) + "-byte " + (memoryType == SharedImageMemory ? "shared" : "private") + " image heap (reclaimed " + std::to_string(reclaimed) + " bytes and released " + std::to_string(released) + " cache bytes first, attempts " + std::to_string(attempts) + "); heap 0 " + std::to_string(used) + "/" + std::to_string(heapBudget) + " MiB");
    }
    {
        std::lock_guard lock(metal->residencyMutex);
        ::AgcDriver::Metal::Residency::Add(metal->residency, heap);
        ::AgcDriver::Metal::Residency::Commit(metal->residency);
        ++metal->made;
    }
    return reinterpret_cast<VkDeviceMemory>(const_cast<void*>(CFBridgingRetain(heap)));
}

VkDeviceMemory BufferPool::TakeImageMemory(const VkMemoryAllocateInfo& allocation, const char* operation, std::uint32_t* allocatedType) {
    const auto type = allocation.memoryTypeIndex;
    if (imagePooled(allocation.allocationSize, type)) {
        std::lock_guard lock(mutex);
        ++clock;
        const auto found = deviceTier.free.find({static_cast<std::size_t>(allocation.allocationSize), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, type, true});
        if (found != deviceTier.free.end() && !found->second.empty()) {
            auto& slots = found->second;
            const auto memory = slots.back().allocation.memory;
            imageOrder.erase(slots.back().order);
            slots.pop_back();
            if (slots.empty()) deviceTier.free.erase(found);
            --deviceTier.slots;
            deviceTier.retainedBytes -= allocation.allocationSize;
            --imageSlots;
            imageBytes -= allocation.allocationSize;
            ++imageHits;
            if (allocatedType != nullptr) *allocatedType = type;
            return memory;
        }
        ++imageMisses;
    }
    const auto memory = allocateImageHeap(allocation.allocationSize, type, operation);
    if (allocatedType != nullptr) *allocatedType = type;
    return memory;
}

void BufferPool::PutImageMemory(VkDeviceMemory memory, VkDeviceSize bytes, std::uint32_t memoryType) noexcept {
    if (memory == VK_NULL_HANDLE) return;
    BufferAllocation allocation{};
    allocation.buffer = VK_NULL_HANDLE;
    allocation.allocationBytes = bytes;
    allocation.bytes = static_cast<std::size_t>(bytes);
    allocation.properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    allocation.memory = memory;
    allocation.memoryType = memoryType;
    if (!imagePooled(bytes, memoryType)) {
        destroy(allocation);
        return;
    }
    std::vector<BufferAllocation> evicted;
    try {
        std::lock_guard lock(mutex);
        if (heapOverBudget()) {
            while (imageSlots != 0 && evictOldestImage(evicted)) {
            }
            evicted.push_back(allocation);
        } else {
            while (imageSlots != 0 && imageBytes + bytes > imageBudget && evictOldestImage(evicted)) {
            }
            while (deviceTier.slots != 0 && (deviceTier.retainedBytes + bytes > deviceTier.budget || deviceTier.slots >= maxSlots)) evictOldest(deviceTier, evicted);
            const SlotKey imageKey{allocation.bytes, allocation.properties, memoryType, true};
            auto& slots = deviceTier.free[imageKey];
            slots.push_back({allocation, ++clock, {}});
            try {
                slots.back().order = imageOrder.insert(imageOrder.end(), &slots.back());
            } catch (...) {
                slots.pop_back();
                if (slots.empty()) deviceTier.free.erase(imageKey);
                throw;
            }
            ++deviceTier.slots;
            deviceTier.retainedBytes += bytes;
            ++imageSlots;
            imageBytes += bytes;
        }
    } catch (...) {
        destroy(allocation);
    }
    for (const auto& gone : evicted) destroy(gone);
}

std::size_t BufferPool::RetainedImages() {
    std::lock_guard lock(mutex);
    return imageSlots;
}

VkDeviceSize BufferPool::ReclaimTarget(VkDeviceSize bytes) {
    return bytes + std::max<VkDeviceSize>(bytes, VkDeviceSize{256} << 20u);
}

bool BufferPool::DeviceTierEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_STAGING_POOL_MIB");
        return value == nullptr || std::strtoull(value, nullptr, 10) != 0;
    }();
    return enabled;
}

VkDeviceSize BufferPool::destroy(const BufferAllocation& allocation) noexcept {
    if (allocation.buffer == VK_NULL_HANDLE && allocation.memory != VK_NULL_HANDLE) {
        // Image memory: the retained heap leaves the residency set (kept alive until the next queue
        // commit returned, MetalResidency.h).
        id<MTLHeap> heap = CFBridgingRelease(reinterpret_cast<CFTypeRef>(allocation.memory));
        std::lock_guard lock(metal->residencyMutex);
        ::AgcDriver::Metal::Residency::Remove(metal->residency, heap);
        ::AgcDriver::Metal::Residency::Commit(metal->residency);
        ++metal->destroyed;
        return allocation.allocationBytes;
    }
    if (allocation.buffer == VK_NULL_HANDLE) return 0;
    id<MTLBuffer> buffer = CFBridgingRelease(reinterpret_cast<CFTypeRef>(allocation.buffer));
    {
        std::lock_guard lock(metal->residencyMutex);
        ::AgcDriver::Metal::Residency::Remove(metal->residency, buffer);
        ::AgcDriver::Metal::Residency::Commit(metal->residency);
        ++metal->destroyed;
    }
    return allocation.allocationBytes;
}

bool BufferPool::DeviceTiered(VkMemoryPropertyFlags properties) {
    return (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 && (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0 && DeviceTierEnabled();
}

VkDeviceSize BufferPool::SlabBlockBytes(std::size_t capacity) {
    return std::max<VkDeviceSize>(VkDeviceSize{2} << 20u, static_cast<VkDeviceSize>(capacity) * 8u);
}

std::size_t BufferPool::Capacity(std::size_t bytes, VkMemoryPropertyFlags properties) {
    constexpr std::size_t smallest = 256;
    if (exactSizes()) return bytes;
    if (bytes < classLimit) return std::max(smallest, std::bit_ceil(bytes));
    if (!DeviceTiered(properties)) return bytes;
    const auto step = std::bit_floor(bytes) >> deviceClassBits;
    return (bytes + step - 1) / step * step;
}

std::size_t BufferPool::NextCapacity(std::size_t capacity) {
    return capacity < classLimit ? capacity * 2 : capacity + (std::bit_floor(capacity) >> deviceClassBits);
}

VkBufferUsageFlags BufferPool::Usage(VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    return DeviceTiered(properties) ? usage | deviceUsage : usage;
}

BufferPool::Tier& BufferPool::tierFor(std::size_t capacity, VkMemoryPropertyFlags properties) {
    // Upstream 27ded2c4d: small device buffers stay in the small tier (unless the tiers are shared).
    if (DeviceTiered(properties) && (sharedTiers() || capacity >= classLimit)) return deviceTier;
    return !sharedTiers() && capacity < classLimit ? smallTier : largeTier;
}

std::optional<BufferAllocation> BufferPool::Take(std::size_t bytes, VkBufferUsageFlags, VkMemoryPropertyFlags properties) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto capacity = Capacity(bytes, properties);
    std::lock_guard lock(mutex);
    ++clock;
    if (profile) {
        static auto lastReport = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now - lastReport > std::chrono::seconds(10)) {
            lastReport = now;
            AgcDriver::ProfilePrint_nid_no_patch("[bufferpool] small: %llu hits, %llu misses, %llu evictions, %zu retained (%.0f MiB); large: %llu hits, %llu misses, %llu evictions, %zu retained (%.0f MiB)%s; device: %llu hits, %llu misses, %llu evictions, %zu retained (%.0f MiB of %llu); images: %llu hits, %llu misses, %llu evictions, %zu retained (%.0f MiB of %llu)\n", static_cast<unsigned long long>(smallTier.hits), static_cast<unsigned long long>(smallTier.misses), static_cast<unsigned long long>(smallTier.evictions), smallTier.slots, smallTier.retainedBytes / 1048576.0, static_cast<unsigned long long>(largeTier.hits), static_cast<unsigned long long>(largeTier.misses), static_cast<unsigned long long>(largeTier.evictions), largeTier.slots, largeTier.retainedBytes / 1048576.0, sharedTiers() ? " (shared)" : "", static_cast<unsigned long long>(deviceTier.hits), static_cast<unsigned long long>(deviceTier.misses), static_cast<unsigned long long>(deviceTier.evictions), deviceTier.slots, deviceTier.retainedBytes / 1048576.0, static_cast<unsigned long long>(deviceTier.budget >> 20u), static_cast<unsigned long long>(imageHits), static_cast<unsigned long long>(imageMisses), static_cast<unsigned long long>(imageEvictions), imageSlots, imageBytes / 1048576.0, static_cast<unsigned long long>(imageBudget >> 20u));
        }
    }
    auto& tier = tierFor(capacity, properties);
    const auto largest = &tier == &deviceTier && !exactSizes() && bytes <= std::numeric_limits<std::size_t>::max() / 2 ? std::max(capacity, bytes * 2) : capacity;
    auto found = tier.free.end();
    for (auto size = capacity; size <= largest && found == tier.free.end(); size = NextCapacity(size)) found = tier.free.find({size, properties});
    if (found == tier.free.end() || found->second.empty()) {
        ++tier.misses;
        return std::nullopt;
    }
    auto result = found->second.back().allocation;
    found->second.pop_back();
    if (found->second.empty()) tier.free.erase(found);
    --tier.slots;
    tier.retainedBytes -= result.allocationBytes;
    ++tier.hits;
    return result;
}

void BufferPool::evictOldest(Tier& tier, std::vector<BufferAllocation>& evicted) {
    const auto oldest = std::min_element(tier.free.begin(), tier.free.end(), [](const auto& left, const auto& right) { return left.second.front().lastUse < right.second.front().lastUse; });
    // First: a throw here leaves the slot retained and counted.
    evicted.push_back(oldest->second.front().allocation);
    tier.retainedBytes -= oldest->second.front().allocation.allocationBytes;
    if (oldest->first.image) {
        imageOrder.erase(oldest->second.front().order);
        imageBytes -= oldest->second.front().allocation.allocationBytes;
        --imageSlots;
        ++imageEvictions;
    }
    oldest->second.pop_front();
    if (oldest->second.empty()) tier.free.erase(oldest);
    --tier.slots;
    ++tier.evictions;
}

// Upstream bafc97bb1e: the oldest retained image is the front of imageOrder (constant time); its
// slot is the front of its key's list, as both are kept oldest first.
bool BufferPool::evictOldestImage(std::vector<BufferAllocation>& evicted) {
    if (imageOrder.empty()) return false;
    const Slot* oldest = imageOrder.front();
    const SlotKey key{oldest->allocation.bytes, oldest->allocation.properties, oldest->allocation.memoryType, true};
    const auto found = deviceTier.free.find(key);
    evicted.push_back(oldest->allocation);
    imageOrder.pop_front();
    const auto bytes = oldest->allocation.allocationBytes;
    deviceTier.retainedBytes -= bytes;
    imageBytes -= bytes;
    found->second.pop_front();
    if (found->second.empty()) deviceTier.free.erase(found);
    --deviceTier.slots;
    --imageSlots;
    ++deviceTier.evictions;
    ++imageEvictions;
    return true;
}

std::size_t BufferPool::MaxSlots() {
    // APS5_BUFFER_POOL_SLOTS=<n> bounds the retained allocations of each tier.
    static const std::size_t slots = [] {
        const char* value = std::getenv("APS5_BUFFER_POOL_SLOTS");
        const auto parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
        return parsed != 0 ? static_cast<std::size_t>(parsed) : defaultSlots;
    }();
    return slots;
}

VkDeviceSize BufferPool::trim() {
    Tier retiredSmall;
    Tier retiredLarge;
    Tier retiredDevice;
    {
        std::lock_guard lock(mutex);
        for (const auto& [tier, retired] : {std::pair{&smallTier, &retiredSmall}, std::pair{&largeTier, &retiredLarge}, std::pair{&deviceTier, &retiredDevice}}) {
            tier->free.swap(retired->free);
            tier->evictions += tier->slots;
            tier->retainedBytes = 0;
            tier->slots = 0;
        }
        imageEvictions += imageSlots;
        imageBytes = 0;
        imageSlots = 0;
        imageOrder.clear();
    }
    VkDeviceSize bytes = 0;
    for (const auto* tier : {&retiredSmall, &retiredLarge, &retiredDevice}) {
        for (const auto& [key, slots] : tier->free) {
            for (const auto& slot : slots) bytes += destroy(slot.allocation);
        }
    }
    return bytes;
}

void BufferPool::Put(const BufferAllocation& allocation) noexcept {
    // Evicted allocations are destroyed after the mutex is released (see evictOldest). The vector
    // may throw on growth; a Put that cannot retain simply destroys, as noexcept requires.
    std::vector<BufferAllocation> evicted;
    try {
        std::lock_guard lock(mutex);
        auto& tier = tierFor(allocation.bytes, allocation.properties);
        if (allocation.allocationBytes > tier.budget) {
            evicted.push_back(allocation);
        } else {
            while (tier.slots != 0 && (tier.retainedBytes + allocation.allocationBytes > tier.budget || tier.slots >= maxSlots)) evictOldest(tier, evicted);
            tier.free[{allocation.bytes, allocation.properties}].push_back({allocation, ++clock, {}});
            ++tier.slots;
            tier.retainedBytes += allocation.allocationBytes;
        }
    } catch (...) {
        destroy(allocation);
    }
    for (const auto& gone : evicted) destroy(gone);
}

BufferPool::Statistics BufferPool::Counts() {
    Statistics statistics{};
    {
        std::lock_guard lock(mutex);
        const auto counts = [](const Tier& tier) { return TierCounts{tier.hits, tier.misses, tier.evictions, tier.slots, tier.retainedBytes, tier.budget}; };
        statistics.small = counts(smallTier);
        statistics.large = counts(largeTier);
        statistics.device = counts(deviceTier);
        statistics.images = TierCounts{imageHits, imageMisses, imageEvictions, imageSlots, imageBytes, imageBudget};
    }
    std::lock_guard lock(metal->residencyMutex);
    statistics.made = metal->made;
    statistics.destroyed = metal->destroyed;
    statistics.resident = static_cast<std::size_t>(metal->residency.allocationCount);
    return statistics;
}

std::shared_ptr<BufferPool> GetBufferPool(const Context& context) {
    if (!context.bufferPool) context.bufferPool = std::make_shared<BufferPool>(context);
    return context.bufferPool;
}

VkDeviceMemory AllocateImageMemory(const Context& context, const VkMemoryAllocateInfo& allocation, std::uint32_t, const char* operation, std::uint32_t* allocatedType) {
    return GetBufferPool(context)->TakeImageMemory(allocation, operation, allocatedType);
}

void ReleaseImageMemory(const Context& context, VkDeviceMemory memory, VkDeviceSize bytes, std::uint32_t memoryType) noexcept {
    if (memory == VK_NULL_HANDLE) return;
    if (context.bufferPool) {
        context.bufferPool->PutImageMemory(memory, bytes, memoryType);
        return;
    }
    // No pool (it made every heap, so it outlives none of them): release the heap. It may still be
    // in a destroyed pool's residency set's graveyard, which holds its own reference.
    CFRelease(reinterpret_cast<CFTypeRef>(memory));
}

}
