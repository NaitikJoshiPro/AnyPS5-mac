#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BUFFERPOOL_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BUFFERPOOL_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MetalObjects.hpp"
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace AgcDriver::Graphics {

// macOS (work package D2): one Metal buffer of the pool. `buffer` carries a retained
// id<MTLBuffer> (Metal::BufferOf reads it): whoever holds the allocation (a Buffer, a DeviceBuffer
// or the pool's free list) owns that reference, and the pool's destroy releases it. Shared storage
// for host-visible properties (`mapping` is its contents), private storage for device-local ones
// (no mapping). Every allocation is resident on the context's queue through the pool's residency
// set from its creation to its destruction.
struct BufferAllocation {
    VkBuffer buffer;
    void* mapping;
    VkDeviceAddress address;
    VkDeviceSize allocationBytes;
    std::size_t bytes;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
};

// Released buffer allocations kept for reuse, since creating a buffer and adding it to a residency
// set costs microseconds and a build makes several. Small requests are served by size class (the
// next power of two), so the many differently sized data and copied-region buffers of consecutive
// builds hit; requests of a MiB and more keep their exact size (the multi-MiB registered-range
// snapshots repeat exactly, and rounding them would waste memory). Retained allocations are evicted
// least recently used under a byte budget and a slot count, in three tiers with a budget each
// (small classes, large exact sizes, device-local staging), so a burst of small buffers between two
// large builds does not sweep the large ones out. Reuse is safe because a Buffer is only released
// once the GPU work using it completed (kept until the batch's event value, or after a
// CommandBatch wait). APS5_NO_BUFFER_CLASSES=1 matches exact sizes only; APS5_BUFFER_POOL_SLOTS=<n>
// sets the slot count of each tier; APS5_BUFFER_POOL_SHARED=1 keeps one tier for host memory;
// APS5_STAGING_POOL_MIB=<n> sets the device tier's budget (0 disables it).
// Metal differences: usage flags do not split the free lists (a Metal buffer serves every use),
// and there are no slabs (a shared buffer costs ~3 us to create and ~0.4 us to make resident on
// this M3, so sub-allocating small ones would not pay for itself).
class BufferPool {
public:
    explicit BufferPool(const Context& context);
    // The device tier's budget: an eighth of the largest device-local heap, at least 512 MiB
    // (APS5_STAGING_POOL_MIB overrides), limited by the heap's `reported` budget when it has one. On
    // Metal the heap is the device's recommended working set.
    static VkDeviceSize DeviceBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported = nullptr);
    // The bytes the texture caches are asked to release (Context::cacheReclaim) when an allocation of
    // `bytes` fails after trimming the pool.
    static VkDeviceSize ReclaimTarget(VkDeviceSize bytes);
    ~BufferPool();
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    // The size a buffer for `bytes` is created with: its size class, or `bytes` itself when large.
    static std::size_t Capacity(std::size_t bytes, VkMemoryPropertyFlags properties);
    static VkBufferUsageFlags Usage(VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    // Upstream's slab block size for a size class (Vulkan slabs; Metal pools whole buffers). Kept for the shared
    // tests' expectations.
    static VkDeviceSize SlabBlockBytes(std::size_t capacity);
    // A new allocation of `capacity` bytes (resident). When Metal refuses it, retained allocations are
    // trimmed and the allocation retried, then the texture caches are released (Context::cacheReclaim)
    // and it is retried again.
    BufferAllocation Allocate(std::size_t capacity, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    std::optional<BufferAllocation> Take(std::size_t bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void Put(const BufferAllocation& allocation) noexcept;
    // Counters (cumulative) for tests and the [bufferpool] line.
    struct TierCounts {
        std::uint64_t hits, misses, evictions;
        std::size_t slots;
        VkDeviceSize retainedBytes, budget;
    };
    struct Statistics {
        TierCounts small, large, device;
        // Allocations made and destroyed, and the allocations the residency set holds now.
        std::uint64_t made, destroyed;
        std::size_t resident;
    };
    Statistics Counts();

private:
    struct Slot {
        BufferAllocation allocation;
        std::uint64_t lastUse;
    };
    struct SlotKey {
        std::size_t bytes;
        VkMemoryPropertyFlags properties;
        bool operator==(const SlotKey&) const = default;
    };
    struct SlotKeyHash {
        std::size_t operator()(const SlotKey& key) const noexcept {
            return std::hash<std::size_t>{}(key.bytes) ^ (static_cast<std::size_t>(key.properties) << 48u);
        }
    };
    struct Tier {
        std::unordered_map<SlotKey, std::deque<Slot>, SlotKeyHash> free;
        std::size_t slots = 0;
        VkDeviceSize retainedBytes = 0;
        VkDeviceSize budget = 0;
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t evictions = 0;
    };
    Tier& tierFor(std::size_t capacity, VkMemoryPropertyFlags properties);
    static bool DeviceTierEnabled();
    static bool DeviceTiered(VkMemoryPropertyFlags properties);
    static std::size_t NextCapacity(std::size_t capacity);
    VkDeviceSize destroy(const BufferAllocation& allocation) noexcept;
    VkDeviceSize trim();
    void evictOldest(Tier& tier, std::vector<BufferAllocation>& evicted);
    static std::size_t MaxSlots();
    std::size_t maxSlots;
    VkDeviceSize (*cacheReclaim)(VkDeviceSize bytes);
    std::mutex mutex;
    // Not `small`/`large`: <rpcndr.h> (via <windows.h>) defines `small` as a macro.
    Tier smallTier;
    Tier largeTier;
    Tier deviceTier;
    std::uint64_t clock = 0;
    // The Metal device, the queue the residency set is attached to, and the set (BufferPool.mm).
    std::unique_ptr<Metal::PoolObjects, Metal::ObjectsDeleter> metal;
    static constexpr VkDeviceSize budget = 512ull * 1024 * 1024;
    static constexpr VkDeviceSize smallBudget = 64ull * 1024 * 1024;
    static constexpr std::size_t classLimit = std::size_t{1} << 20u;
    static constexpr unsigned deviceClassBits = 3;
    static constexpr std::size_t defaultSlots = 4096;
};

std::shared_ptr<BufferPool> GetBufferPool(const Context& context);

}

#endif
