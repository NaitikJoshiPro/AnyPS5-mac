#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libc/include/WindowsMappings.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#ifndef _WIN32
#include <cerrno>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif
#include <cstdio>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

namespace GuestArena {
namespace {

std::atomic<void (*)(std::uintptr_t, std::size_t)> hostReadHook{nullptr};

#ifdef _WIN32
std::atomic<std::uint64_t> commitGeneration{1};
std::atomic<void (*)(std::uintptr_t, std::size_t, std::uint64_t)> privateMappingObserver{nullptr};
#endif
std::atomic<void (*)(std::uintptr_t, std::size_t)> accessObserver{nullptr};

void noteAccess(const void* pointer, std::size_t bytes) {
    if (const auto callback = accessObserver.load(std::memory_order_acquire)) callback(reinterpret_cast<std::uintptr_t>(pointer), bytes);
}

constexpr std::uintptr_t GuestArenaOffset = GuestArena::GuestAddressOffset;
constexpr std::uintptr_t ArenaStart = 0x0000000200000000ull + GuestArenaOffset;
constexpr std::uintptr_t SystemReservedStart = 0x00000007FFFFC000ull + GuestArenaOffset;
constexpr std::uintptr_t SystemReservedEnd = 0x0000001000000000ull + GuestArenaOffset;
constexpr std::uintptr_t ApplicationAreaEnd = 0x000000FC00000000ull + GuestArenaOffset;

std::uintptr_t alignUp(std::uintptr_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
}

class Arena {
public:
    static Arena& Get() {
        static auto* arena = new Arena;
        return *arena;
    }

    bool Available() const {
        return _base != 0;
    }

    bool Contains(const void* pointer, std::size_t bytes) const {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        return _base != 0 && address >= _base && bytes <= _end - _base && address - _base <= (_end - _base) - bytes;
    }

    void* Allocate(std::size_t bytes, std::size_t alignment) {
        return AllocateAtOrAbove(0, bytes, alignment);
    }

    void* AllocateAtOrAbove(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) {
        RequireAllocatable(alignment);
        std::lock_guard lock(_lock);
        if (hint >= _end) throw std::runtime_error("mapping address hint is above the guest address space arena");
        return Claim(FindAtOrAbove(hint, bytes, alignment), bytes);
    }

    void* AllocateFromHint(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) {
        RequireAllocatable(alignment);
        std::lock_guard lock(_lock);
        auto candidate = hint < _end ? FindAtOrAbove(hint, bytes, alignment) : 0;
        if (candidate == 0 && hint > _base) candidate = FindAtOrAbove(_base, bytes, alignment);
        return Claim(candidate, bytes);
    }

    std::uintptr_t HighestFree(std::size_t bytes, std::size_t alignment) {
        if (_base == 0) throw std::runtime_error("guest address space arena is unavailable");
        if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) throw std::invalid_argument("invalid guest arena window");
        std::lock_guard lock(_lock);
        auto limit = _end;
        while (limit >= _base && bytes <= limit - _base) {
            const auto candidate = (limit - bytes) & ~(alignment - 1);
            if (candidate < _base) break;
            const auto next = _used.lower_bound(candidate + bytes);
            if (next == _used.begin() || std::prev(next)->second <= candidate) return candidate;
            limit = std::prev(next)->first;
        }
        return 0;
    }

    bool NextUsed(std::uintptr_t address, std::uintptr_t* start, std::uintptr_t* end) {
        std::lock_guard lock(_lock);
        auto it = _used.upper_bound(address);
        if (it != _used.begin() && std::prev(it)->second > address) --it;
        if (it == _used.end()) return false;
        *start = it->first;
        *end = it->second;
        return true;
    }

    bool MarkUsedForCommit(const void* pointer, std::size_t bytes) {
        std::lock_guard lock(_lock);
        const auto start = reinterpret_cast<std::uintptr_t>(pointer);
        const auto end = start + bytes;
        auto& used = UsedFor(start, bytes);
        const auto exact = used.find(start);
        if (exact != used.end() && exact->second >= end) return false;
        const auto next = used.lower_bound(start);
        if ((next != used.end() && next->first < end) || (next != used.begin() && std::prev(next)->second > start)) {
            char message[160];
            std::snprintf(message, sizeof(message), "guest arena commit 0x%llx+0x%zx overlaps a used range", static_cast<unsigned long long>(start), bytes);
            throw std::runtime_error(message);
        }
        used.emplace(start, end);
        return true;
    }

    void MarkUsed(const void* pointer, std::size_t bytes) {
        std::lock_guard lock(_lock);
        const auto start = reinterpret_cast<std::uintptr_t>(pointer);
        if (bytes != 0 && start < SystemReservedEnd && start + bytes > SystemReservedStart) {
            char message[192];
            std::snprintf(message, sizeof(message), "fixed guest mapping 0x%llx+0x%zx overlaps the PS5 system-reserved range 0x%llx-0x%llx",
                static_cast<unsigned long long>(start), bytes, static_cast<unsigned long long>(SystemReservedStart), static_cast<unsigned long long>(SystemReservedEnd));
            throw std::runtime_error(message);
        }
        auto& used = UsedFor(start, bytes);
        const auto next = used.lower_bound(start);
        if ((next != used.end() && next->first < start + bytes) || (next != used.begin() && std::prev(next)->second > start)) {
            char message[160];
            std::snprintf(message, sizeof(message), "fixed guest mapping 0x%llx+0x%zx overlaps %s", static_cast<unsigned long long>(start), bytes, OverlapsHostRegion(start, bytes) ? "a host region in the guest arena" : "a guest arena range");
            throw std::runtime_error(message);
        }
        used.emplace(start, start + bytes);
    }

    void Release(const void* pointer, std::size_t bytes) {
        std::lock_guard lock(_lock);
        const auto start = reinterpret_cast<std::uintptr_t>(pointer);
        const auto end = start + bytes;
        auto& used = UsedFor(start, bytes);
        auto it = used.upper_bound(start);
        if (it != used.begin()) --it;
        while (it != used.end() && it->first < end) {
            const auto rangeStart = it->first;
            const auto rangeEnd = it->second;
            if (rangeEnd <= start) {
                ++it;
                continue;
            }
            it = used.erase(it);
            if (rangeStart < start) used.emplace(rangeStart, start);
            if (rangeEnd > end) used.emplace(end, rangeEnd);
        }
        if (&used == &_displacedUsed) return;
        for (const auto& [holeStart, holeEnd] : _holes) {
            if (holeStart < end && start < holeEnd) _used.emplace(holeStart, holeEnd);
        }
    }

private:
    Arena() {
#if defined(__linux__)
        SeedHostUsedRanges();
#elif defined(__APPLE__)
        // Not upstream's SeedHostUsedRanges: the arm64 port displaces the host's reserved ranges (D31, D33).
        SeedAppleUsedRanges();
#elif defined(_WIN32)
        _writeWatched = std::getenv("APS5_NO_WRITE_WATCH") == nullptr;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        const std::uintptr_t granularity = system.dwAllocationGranularity;
        const std::uintptr_t end = ApplicationAreaEnd;
        std::uint32_t retries = 0;
        for (std::uintptr_t cursor = ArenaStart; cursor < end;) {
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) == 0) throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "query the guest arena range");
            const auto regionEnd = std::min(end, reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize);
            const auto first = info.State == MEM_FREE ? std::min(regionEnd, alignUp(cursor, granularity)) : regionEnd;
            const auto last = std::max(first, regionEnd & ~(granularity - 1));
            if (first < last && WindowsMappings::Get().Reserve(reinterpret_cast<void*>(first), last - first) == nullptr) {
                const auto error = GetLastError();
                if (error == ERROR_INVALID_ADDRESS && ++retries < 64u) continue;
                throw std::system_error(static_cast<int>(error), std::system_category(), "reserve the guest arena range");
            }
            retries = 0;
            if (cursor < first) _holes.emplace_back(cursor, first);
            if (last < regionEnd) _holes.emplace_back(last, regionEnd);
            cursor = regionEnd;
        }
        _hostRegions = _holes;
        _holes.emplace_back(SystemReservedStart, SystemReservedEnd);
        for (const auto& [holeStart, holeEnd] : _holes) _used.emplace(holeStart, holeEnd);
        _base = ArenaStart;
        _end = end;
#endif
    }

#if defined(__linux__)
    static std::vector<std::pair<std::uintptr_t, std::uintptr_t>> HostMappings() {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> mappings;
        std::ifstream maps("/proc/self/maps");
        if (!maps.is_open()) throw std::runtime_error("guest arena: cannot open /proc/self/maps");
        std::string line;
        while (std::getline(maps, line)) {
            std::istringstream fields(line);
            std::uintptr_t begin = 0, end = 0;
            char dash = 0;
            if (!(fields >> std::hex >> begin >> dash >> end) || dash != '-' || end <= begin) throw std::runtime_error("guest arena: malformed /proc/self/maps line: " + line);
            mappings.emplace_back(begin, end);
        }
        return mappings;
    }

    static int ReserveNoReplace(std::uintptr_t address, std::size_t bytes) {
        void* const wanted = reinterpret_cast<void*>(address);
        void* const placed = mmap(wanted, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
        if (placed == wanted) return 0;
        if (placed == MAP_FAILED) return errno;
        munmap(placed, bytes);
        return EINVAL;
    }
#elif defined(__APPLE__)
    static std::vector<std::pair<std::uintptr_t, std::uintptr_t>> HostMappings() {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> mappings;
        mach_vm_address_t address = ArenaStart;
        while (address < ApplicationAreaEnd) {
            mach_vm_size_t size = 0;
            vm_region_basic_info_data_64_t info{};
            mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t object = MACH_PORT_NULL;
            const kern_return_t result = mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object);
            if (result == KERN_INVALID_ADDRESS) break;
            if (result != KERN_SUCCESS) throw std::runtime_error("guest arena: mach_vm_region failed");
            mappings.emplace_back(static_cast<std::uintptr_t>(address), static_cast<std::uintptr_t>(address + size));
            address += size;
        }
        return mappings;
    }

    static int ReserveNoReplace(std::uintptr_t address, std::size_t bytes) {
        mach_vm_address_t wanted = address;
        const kern_return_t result = mach_vm_map(mach_task_self(), &wanted, bytes, 0, VM_FLAGS_FIXED, MEMORY_OBJECT_NULL, 0, FALSE, VM_PROT_NONE, VM_PROT_ALL, VM_INHERIT_DEFAULT);
        if (result == KERN_SUCCESS) return 0;
        return result == KERN_NO_SPACE ? EEXIST : ENOMEM;
    }
#endif

#if defined(__linux__)
    void SeedHostUsedRanges() {
        for (const auto& [begin, end] : HostMappings()) {
            if (end <= ArenaStart || begin >= ApplicationAreaEnd) continue;
            const auto first = std::max(begin, ArenaStart);
            const auto last = std::min(end, ApplicationAreaEnd);
            _used.emplace(first, last);
            _hostRegions.emplace_back(first, last);
        }
        _used.emplace(SystemReservedStart, SystemReservedEnd);
        _hostRegions.emplace_back(SystemReservedStart, SystemReservedEnd);
        ReserveFreeRanges();
        _base = ArenaStart;
        _end = ApplicationAreaEnd;
    }
#elif defined(__APPLE__)
    // The PS5 layout from 8 GiB is reserved as on Linux, around what the host already mapped there: on
    // arm64 the dyld shared region spans 6 to 63 GiB, system guard ranges follow up to 448 GiB, and malloc
    // places its regions between 448 and 512 GiB (macOS 27). Those stay host regions; what the host maps
    // later lands below 8 GiB or above the arena. The shared region's unused tail, the commpage guard and the GPU
    // carveout are displaced instead (D31, D33).
    void SeedAppleUsedRanges() {
        _displacedActive = GuestDisplacedEnable_nid_no_patch();
        for (mach_vm_address_t address = ArenaStart; address < ApplicationAreaEnd;) {
            mach_vm_size_t size = 0;
            vm_region_basic_info_data_64_t info{};
            mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t object = MACH_PORT_NULL;
            if (mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || address >= ApplicationAreaEnd) break;
            const auto first = std::max<std::uintptr_t>(address, ArenaStart);
            const auto last = std::min<std::uintptr_t>(address + size, ApplicationAreaEnd);
            address += size;
            if (_displacedActive && OverlapsDisplaced(first, last - first)) {
                // The host's reserved ranges there are backed by the displaced alias (GuestDisplaced.hpp): they stay
                // out of first-fit placement (AddUsed below) but take fixed guest mappings.
                if (first < DisplacedStart()) {
                    AddUsed(first, DisplacedStart());
                    _hostRegions.emplace_back(first, DisplacedStart());
                }
                if (last > DisplacedEnd) {
                    AddUsed(DisplacedEnd, last);
                    _hostRegions.emplace_back(DisplacedEnd, last);
                }
                continue;
            }
            AddUsed(first, last);
            _hostRegions.emplace_back(first, last);
        }
        if (_displacedActive) AddUsed(DisplacedStart(), DisplacedEnd);
        AddUsed(SystemReservedStart, SystemReservedEnd);
        _hostRegions.emplace_back(SystemReservedStart, SystemReservedEnd);
        ReserveFreeRanges();
        _base = ArenaStart;
        _end = ApplicationAreaEnd;
    }

    void AddUsed(std::uintptr_t first, std::uintptr_t last) {
        const auto [entry, added] = _used.emplace(first, last);
        if (!added) entry->second = std::max(entry->second, last);
    }
#endif

#if defined(__linux__) || defined(__APPLE__)
    static void TryReserve(std::vector<std::pair<std::uintptr_t, std::uintptr_t>>& occupied, std::uintptr_t from, std::uintptr_t to) {
        constexpr std::uintptr_t kChunk = 1ULL << 30;
        for (std::uintptr_t cursor = from; cursor < to; cursor += kChunk) {
            const auto end = std::min(cursor + kChunk, to);
            const int failed = ReserveNoReplace(cursor, end - cursor);
            if (failed == 0) continue;
            if (failed != EEXIST) {
                char message[128];
                std::snprintf(message, sizeof(message), "guest arena: cannot reserve 0x%llx+0x%llx",
                    static_cast<unsigned long long>(cursor), static_cast<unsigned long long>(end - cursor));
                throw std::system_error(failed, std::system_category(), message);
            }
            occupied.emplace_back(cursor, end);
        }
    }

    void ReserveFreeRanges() {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> occupied;
        std::uintptr_t cursor = ArenaStart;
        for (const auto& [usedStart, usedEnd] : _used) {
            if (usedStart > ApplicationAreaEnd) break;
            if (usedStart > cursor) TryReserve(occupied, cursor, std::min(usedStart, ApplicationAreaEnd));
            if (usedEnd > cursor) cursor = usedEnd;
            if (cursor >= ApplicationAreaEnd) break;
        }
        if (cursor < ApplicationAreaEnd) TryReserve(occupied, cursor, ApplicationAreaEnd);
        for (const auto& [from, to] : occupied) {
            _used.emplace(from, to);
            _hostRegions.emplace_back(from, to);
        }
    }
#endif

    // Fixed guest mappings inside the displaced window are tracked apart from _used, where the whole window stays
    // one used range so first-fit placement never lands there.
    std::map<std::uintptr_t, std::uintptr_t>& UsedFor(std::uintptr_t start, std::size_t bytes) {
        if (!_displacedActive || !OverlapsDisplaced(start, bytes)) return _used;
        if (start < DisplacedStart() || start + bytes > DisplacedEnd) {
            // The ranges inside the window share one contiguous alias, so a range may cross from one to the next;
            // across the window's outer edges half of it would be backed by the host and half by the alias.
            char message[224];
            std::snprintf(message, sizeof(message), "guest range 0x%llx+0x%zx straddles the edge of the displaced window 0x%llx-0x%llx (host-reserved memory on one side)",
                static_cast<unsigned long long>(start), bytes, static_cast<unsigned long long>(DisplacedStart()), static_cast<unsigned long long>(DisplacedEnd));
            throw std::runtime_error(message);
        }
        // The PS5 system-reserved range stays a used range, as on every host: fixed guest mappings there are refused.
        if (start < SystemReservedEnd && start + bytes > SystemReservedStart) return _used;
        return _displacedUsed;
    }

    std::mutex _lock;
    bool _displacedActive = false;
    std::map<std::uintptr_t, std::uintptr_t> _displacedUsed;
    std::map<std::uintptr_t, std::uintptr_t> _used;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> _holes;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> _hostRegions;
    void RequireAllocatable(std::size_t alignment) const {
        if (_base == 0) throw std::runtime_error("guest address space arena is unavailable");
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) throw std::invalid_argument("invalid guest arena alignment");
    }

    std::uintptr_t FindAtOrAbove(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) const {
        std::uintptr_t candidate = alignUp(std::max(_base, hint), alignment);
        auto it = _used.upper_bound(candidate);
        if (it != _used.begin() && std::prev(it)->second > candidate) --it;
        for (; it != _used.end(); ++it) {
            const auto& [start, end] = *it;
            if (candidate <= _end && bytes <= _end - candidate && candidate + bytes <= start) break;
            candidate = std::max(candidate, alignUp(end, alignment));
        }
        if (candidate > _end || bytes > _end - candidate) return 0;
        return candidate;
    }

    void* Claim(std::uintptr_t candidate, std::size_t bytes) {
        if (candidate == 0) throw std::runtime_error("guest address space arena exhausted");
        _used.emplace(candidate, candidate + bytes);
        return reinterpret_cast<void*>(candidate);
    }

    std::uintptr_t _base = 0;
    std::uintptr_t _end = 0;
    bool _writeWatched = false;

public:
    std::uintptr_t Base() const { return _base; }
    std::size_t Size() const { return _end - _base; }
    bool WriteWatched() const { return _writeWatched; }
    bool OverlapsHostRegion(std::uintptr_t start, std::size_t bytes) const {
        for (const auto& [regionStart, regionEnd] : _hostRegions) {
            if (regionStart < start + bytes && start < regionEnd) return true;
        }
        return false;
    }
};

const bool g_reserved = (Arena::Get(), true);

}

bool GuestArenaAvailable_nid_postfix() {
    return Arena::Get().Available();
}

bool GuestArenaContains_nid_postfix(const void* pointer, std::size_t bytes) {
    return Arena::Get().Contains(pointer, bytes);
}

void* GuestArenaAllocate_nid_postfix(std::size_t bytes, std::size_t alignment) {
    return Arena::Get().Allocate(bytes, alignment);
}

void* GuestArenaAllocateAtOrAbove_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) {
    return Arena::Get().AllocateAtOrAbove(hint, bytes, alignment);
}

void* GuestArenaAllocateFromHint_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment) {
    return Arena::Get().AllocateFromHint(hint, bytes, alignment);
}

std::uintptr_t GuestArenaHighestFree_nid_no_patch(std::size_t bytes, std::size_t alignment) {
    return Arena::Get().HighestFree(bytes, alignment);
}

bool GuestArenaNextUsed_nid_no_patch(std::uintptr_t address, std::uintptr_t* start, std::uintptr_t* end) {
    return Arena::Get().NextUsed(address, start, end);
}

void GuestArenaMarkUsed_nid_postfix(const void* pointer, std::size_t bytes) {
    Arena::Get().MarkUsed(pointer, bytes);
}

void GuestArenaRelease_nid_postfix(const void* pointer, std::size_t bytes) {
    Arena::Get().Release(pointer, bytes);
}

void GuestArenaRange_nid_postfix(std::uintptr_t* base, std::size_t* bytes) {
    *base = Arena::Get().Base();
    *bytes = Arena::Get().Size();
}

namespace {

std::invalid_argument OutsideArena(const char* operation, const void* pointer, std::size_t bytes) {
    char message[128];
    std::snprintf(message, sizeof(message), "%s 0x%llx+0x%zx outside the guest arena", operation, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(pointer)), bytes);
    return std::invalid_argument(message);
}

}

#ifdef _WIN32
void GuestArenaSetProtection_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint32_t protection) {
    WindowsMappings::Get().SetProtection(address, bytes, protection);
    noteAccess(reinterpret_cast<const void*>(address), bytes);
}

bool GuestArenaHandleWrite_nid_postfix(std::uintptr_t address) {
    return WindowsMappings::Get().HandleWrite(address);
}

bool GuestArenaHandleRead_nid_postfix(std::uintptr_t address) {
    return WindowsMappings::Get().HandleRead(address);
}

void GuestArenaPinWritable_nid_postfix(const void* pointer, std::size_t bytes) {
    if (!Arena::Get().Contains(pointer, bytes)) return;
    WindowsMappings::Get().Pin(reinterpret_cast<std::uintptr_t>(pointer), bytes);
}

void GuestArenaUnpinWritable_nid_postfix(const void* pointer, std::size_t bytes) {
    if (!Arena::Get().Contains(pointer, bytes)) return;
    WindowsMappings::Get().Unpin(reinterpret_cast<std::uintptr_t>(pointer), bytes);
}

bool GuestArenaProtection_nid_postfix(std::uintptr_t address, std::uint32_t* protection) {
    return WindowsMappings::Get().Protection(address, protection);
}

bool GuestArenaCollectWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
    return WindowsMappings::Get().Collect(address, bytes, pages, count, clear);
}

bool GuestArenaCollectArmedWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
    return WindowsMappings::Get().Collect(address, bytes, pages, count, clear, WindowsMappings::Collection::Armed);
}

bool GuestArenaCollectFreshWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
    return WindowsMappings::Get().Collect(address, bytes, pages, count, clear, WindowsMappings::Collection::Fresh);
}

bool GuestArenaHostRegionOverlaps_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    return Arena::Get().OverlapsHostRegion(address, bytes);
}

std::uint64_t GuestArenaCommitGeneration_nid_postfix() {
    return commitGeneration.load(std::memory_order_acquire);
}

void GuestArenaSetPrivateMappingObserver_nid_postfix(void (*callback)(std::uintptr_t, std::size_t, std::uint64_t)) {
    privateMappingObserver.store(callback, std::memory_order_release);
}

void GuestArenaCommit_nid_postfix(void* pointer, std::size_t bytes, std::uint32_t protection, std::size_t granule) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("commit", pointer, bytes);
    const auto generation = commitGeneration.fetch_add(1, std::memory_order_acq_rel) + 1;
    const auto created = WindowsMappings::Get().Commit(pointer, bytes, protection, granule, Arena::Get().WriteWatched());
    if (const auto callback = privateMappingObserver.load(std::memory_order_acquire)) {
        for (const auto& [address, size] : created) callback(address, size, generation);
    }
    noteAccess(pointer, bytes);
}

void GuestArenaReset_nid_postfix(void* pointer, std::size_t bytes) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("reset", pointer, bytes);
    WindowsMappings::Get().Reset(pointer, bytes);
}

void GuestArenaMap_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("shared mapping", pointer, bytes);
    WindowsMappings::Get().Map(pointer, bytes, section, offset, protection, Arena::Get().WriteWatched(), false);
    noteAccess(pointer, bytes);
}

void GuestArenaMapDirect_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("direct memory mapping", pointer, bytes);
    const auto generation = commitGeneration.fetch_add(1, std::memory_order_acq_rel) + 1;
    std::vector<std::pair<std::uintptr_t, std::size_t>> created;
    std::exception_ptr failure;
    try {
        WindowsMappings::Get().Map(pointer, bytes, section, offset, protection, Arena::Get().WriteWatched(), true, PAGE_EXECUTE_READWRITE, SIZE_MAX, &created);
    } catch (...) {
        failure = std::current_exception();
    }
    try {
        if (const auto callback = privateMappingObserver.load(std::memory_order_acquire)) {
            for (const auto& [address, size] : created) callback(address, size, generation);
        }
    } catch (...) {
        if (failure != nullptr) std::rethrow_exception(failure);
        throw;
    }
    if (failure != nullptr) std::rethrow_exception(failure);
    noteAccess(pointer, bytes);
}

void GuestArenaMapFile_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection, std::uint32_t viewProtection, std::size_t sectionBytes) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("file mapping", pointer, bytes);
    WindowsMappings::Get().Map(pointer, bytes, section, offset, protection, Arena::Get().WriteWatched(), false, viewProtection, sectionBytes);
}

void* GuestArenaMapAlias_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    return WindowsMappings::Get().MapAlias(address, bytes);
}

void GuestArenaUnmapAlias_nid_postfix(void* alias) {
    WindowsMappings::Get().UnmapAlias(alias);
}

bool GuestArenaResidentSpan_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uintptr_t* start, std::uintptr_t* end) {
    return WindowsMappings::Get().ResidentSpan(address, bytes, start, end);
}

void GuestArenaForgetSection_nid_postfix(void* section) {
    WindowsMappings::Get().ForgetSection(section);
}

bool GuestArenaForgetReusableSection_nid_no_patch(void* section) {
    return WindowsMappings::Get().ForgetSection(section);
}
#endif

#if defined(__linux__) || defined(__APPLE__)
namespace {
int PosixProtection(std::uint32_t windowsProtection) {
    switch (windowsProtection) {
    case 0x01: return PROT_NONE;
    case 0x02: return PROT_READ;
    case 0x04: return PROT_READ | PROT_WRITE;
    case 0x08: return PROT_READ | PROT_WRITE;
    case 0x10: return PROT_EXEC;
    case 0x20: return PROT_READ | PROT_EXEC;
    case 0x40: return PROT_READ | PROT_WRITE | PROT_EXEC;
    default: throw std::runtime_error("guest arena: unmapped protection value");
    }
}

// Returns the guest address on success.
void* PlaceAt(void* pointer, std::size_t bytes, std::uint32_t protection) {
    void* const placed = mmap(HostPointer(pointer), bytes, PosixProtection(protection), MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    return placed == MAP_FAILED ? MAP_FAILED : pointer;
}
}
#endif

#if defined(__linux__) || defined(__APPLE__)
void GuestArenaSetProtection_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint32_t protection) {
    if (!Arena::Get().Contains(reinterpret_cast<void*>(address), bytes)) throw OutsideArena("set protection", reinterpret_cast<void*>(address), bytes);
    if (mprotect(reinterpret_cast<void*>(HostAddress(address)), bytes, PosixProtection(protection)) != 0)
        throw std::system_error(errno, std::generic_category(), "guest arena protection failed");
    noteAccess(reinterpret_cast<const void*>(address), bytes);
}

void GuestArenaCommit_nid_postfix(void* pointer, std::size_t bytes, std::uint32_t protection, std::size_t granule) {
    (void)granule;
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("commit", pointer, bytes);
    const bool marked = Arena::Get().MarkUsedForCommit(pointer, bytes);
    try {
        if (PlaceAt(pointer, bytes, protection) == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "guest arena commit failed");
    } catch (...) {
        if (marked) Arena::Get().Release(pointer, bytes);
        throw;
    }
    noteAccess(pointer, bytes);
}

void GuestArenaReset_nid_postfix(void* pointer, std::size_t bytes) {
    if (!Arena::Get().Contains(pointer, bytes)) throw OutsideArena("reset", pointer, bytes);
    if (PlaceAt(pointer, bytes, 0x01) == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "guest arena reset failed");
}
#endif

bool GuestArenaWriteWatched_nid_postfix() {
    return Arena::Get().WriteWatched();
}

void GuestArenaSetAccessObserver_nid_postfix(void (*callback)(std::uintptr_t address, std::size_t bytes)) {
    accessObserver.store(callback, std::memory_order_release);
}

void GuestArenaSetHostReadHook_nid_no_patch(void (*hook)(std::uintptr_t, std::size_t)) {
    hostReadHook.store(hook, std::memory_order_release);
}

void GuestArenaPrepareHostRead_nid_no_patch(const void* pointer, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (address == 0 || bytes == 0 || bytes > std::numeric_limits<std::uintptr_t>::max() - address) return;
    if (const auto hook = hostReadHook.load(std::memory_order_acquire)) hook(address, bytes);
}

bool GuestArenaBeginHostWrite_nid_postfix(void* pointer, std::size_t bytes) {
#ifdef _WIN32
    return WindowsMappings::Get().BeginHostWrite(reinterpret_cast<std::uintptr_t>(pointer), bytes);
#else
    GuestWriteWatch::GuestWriteWatchBeginHostWrite_nid_postfix(pointer, bytes);
    return true;
#endif
}

#ifdef _WIN32
bool GuestArenaBeginHostImport_nid_no_patch(void* pointer, std::size_t bytes) {
    return WindowsMappings::Get().BeginHostWrite(reinterpret_cast<std::uintptr_t>(pointer), bytes, false);
}

void GuestArenaEndHostImport_nid_no_patch(void* pointer, std::size_t bytes) {
    WindowsMappings::Get().EndHostWrite(reinterpret_cast<std::uintptr_t>(pointer), bytes, false);
}
#endif

void GuestArenaEndHostWrite_nid_postfix(void* pointer, std::size_t bytes) {
#ifdef _WIN32
    WindowsMappings::Get().EndHostWrite(reinterpret_cast<std::uintptr_t>(pointer), bytes);
#else
    GuestWriteWatch::GuestWriteWatchEndHostWrite_nid_postfix(pointer, bytes);
#endif
}

#ifndef _WIN32
namespace {

std::atomic<SharedBackingResolver> sharedBackingResolver{nullptr};
std::atomic<SharedBackingWriter> sharedBackingWriter{nullptr};

}

void GuestArenaSetSharedBacking_nid_postfix(SharedBackingResolver resolver) {
    sharedBackingResolver.store(resolver, std::memory_order_release);
}

void GuestArenaSetSharedBackingWriter_nid_no_patch(SharedBackingWriter writer) {
    sharedBackingWriter.store(writer, std::memory_order_release);
}

bool GuestArenaWriteSharedBacking_nid_no_patch(std::uintptr_t address, const void* source, std::size_t bytes) {
    const auto writer = sharedBackingWriter.load(std::memory_order_acquire);
    return writer != nullptr && writer(address, source, bytes);
}

bool GuestArenaSharedBacking_nid_postfix(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset) {
    const auto resolver = sharedBackingResolver.load(std::memory_order_acquire);
    return resolver != nullptr && resolver(address, bytes, file, offset);
}
#endif

}
