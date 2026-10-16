#include "CodePool.hpp"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <libkern/OSCacheControl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>

namespace Aps5Fex {

namespace {

std::size_t RoundUp(std::size_t value) {
    return (value + CodePool::Granule - 1) & ~(CodePool::Granule - 1);
}

// Live pools, read from fault handlers (no locks).
constexpr std::size_t MaxPools = 8;
std::atomic<CodePool*> pools[MaxPools];
std::atomic<CodePool::FreeListener> freeListener{nullptr};

}

void CodePool::SetFreeListener(FreeListener listener) {
    freeListener.store(listener, std::memory_order_release);
}

CodePool::CodePool(std::size_t size) : size(RoundUp(size)) {
    if (this->size == 0) throw std::runtime_error("the JIT pool cannot be empty");
    mach_vm_address_t rw = 0;
    mach_vm_address_t rx = 0;
    if (mach_vm_allocate(mach_task_self(), &rw, this->size, VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
        throw std::runtime_error("cannot allocate the JIT pool");
    vm_prot_t current {};
    vm_prot_t maximum {};
    if (mach_vm_remap(mach_task_self(), &rx, this->size, 0, VM_FLAGS_ANYWHERE, mach_task_self(), rw, FALSE, &current, &maximum,
                      VM_INHERIT_NONE) != KERN_SUCCESS) {
        mach_vm_deallocate(mach_task_self(), rw, this->size);
        throw std::runtime_error("cannot map the JIT pool twice");
    }
    if (mprotect(reinterpret_cast<void*>(rx), this->size, PROT_READ | PROT_EXEC) != 0) {
        mach_vm_deallocate(mach_task_self(), rx, this->size);
        mach_vm_deallocate(mach_task_self(), rw, this->size);
        throw std::runtime_error("cannot map the JIT pool executable");
    }
    code = reinterpret_cast<std::uint8_t*>(rx);
    writable = reinterpret_cast<std::uint8_t*>(rw);
    writeOffset = static_cast<std::int64_t>(rw) - static_cast<std::int64_t>(rx);
    free.emplace(0, this->size);
    for (auto& slot : pools) {
        CodePool* empty = nullptr;
        if (slot.compare_exchange_strong(empty, this, std::memory_order_acq_rel)) break;
    }
}

CodePool::~CodePool() {
    for (auto& slot : pools) {
        CodePool* self = this;
        slot.compare_exchange_strong(self, nullptr, std::memory_order_acq_rel);
    }
    mach_vm_deallocate(mach_task_self(), reinterpret_cast<mach_vm_address_t>(code), size);
    mach_vm_deallocate(mach_task_self(), reinterpret_cast<mach_vm_address_t>(writable), size);
}

void* CodePool::Allocate(std::size_t length) {
    const std::size_t aligned = RoundUp(length);
    if (aligned == 0) throw std::runtime_error("JIT pool: empty allocation");
    std::lock_guard lock(mutex);
    for (auto range = free.begin(); range != free.end(); ++range) {
        if (range->second < aligned) continue;
        const std::size_t offset = range->first;
        const std::size_t rest = range->second - aligned;
        free.erase(range);
        if (rest != 0) free.emplace(offset + aligned, rest);
        used += aligned;
        // Pages freed earlier were marked reusable; take them back into this task's footprint.
        madvise(writable + offset, aligned, MADV_FREE_REUSE);
        return code + offset;
    }
    throw std::runtime_error("JIT pool exhausted: no free range of " + std::to_string(aligned >> 10) + " KiB in " +
                             std::to_string(size >> 10) + " KiB (" + std::to_string(used >> 10) + " KiB in use, " +
                             std::to_string(free.size()) + " free ranges)");
}

void* CodePool::AllocateNear(std::uintptr_t pc, std::size_t length, std::size_t reach) {
    const std::size_t aligned = RoundUp(length);
    if (aligned == 0 || !Contains(reinterpret_cast<const void*>(pc))) return nullptr;
    const std::size_t target = (pc - reinterpret_cast<std::uintptr_t>(code)) & ~(Granule - 1);
    std::lock_guard lock(mutex);
    auto best = free.end();
    std::size_t bestStart = 0, bestDistance = ~std::size_t{0};
    for (auto range = free.begin(); range != free.end(); ++range) {
        if (range->second < aligned) continue;
        const std::size_t start = std::min(std::max(target, range->first), range->first + range->second - aligned);
        const std::size_t distance = start >= target ? start + aligned - target : target - start;
        if (distance < bestDistance) {
            best = range;
            bestStart = start;
            bestDistance = distance;
        }
    }
    if (best == free.end() || bestDistance > reach) return nullptr;
    const std::size_t offset = best->first, rangeLength = best->second;
    free.erase(best);
    if (bestStart > offset) free.emplace(offset, bestStart - offset);
    if (bestStart + aligned < offset + rangeLength) free.emplace(bestStart + aligned, offset + rangeLength - bestStart - aligned);
    used += aligned;
    madvise(writable + bestStart, aligned, MADV_FREE_REUSE);
    return code + bestStart;
}

bool CodePool::Free(void* address, std::size_t length) {
    auto* byte = static_cast<std::uint8_t*>(address);
    if (!Contains(byte)) return false;
    const std::size_t offset = static_cast<std::size_t>(byte - code);
    const std::size_t aligned = RoundUp(length);
    if (offset % Granule != 0 || aligned == 0 || offset + aligned > size)
        throw std::runtime_error("JIT pool: bad free of " + std::to_string(length) + " bytes at offset " + std::to_string(offset));
    // Before the pool lock: the listener takes libc's displaced-site lock, which a fault handler holds while it
    // allocates stub memory from this pool (AllocateBeside), so the order is always site lock -> pool lock.
    if (const auto listener = freeListener.load(std::memory_order_acquire))
        listener(reinterpret_cast<std::uintptr_t>(byte), reinterpret_cast<std::uintptr_t>(byte) + aligned);
    std::lock_guard lock(mutex);
    auto next = free.lower_bound(offset);
    if (next != free.end() && next->first < offset + aligned) throw std::runtime_error("JIT pool: range freed twice");
    auto previous = next == free.begin() ? free.end() : std::prev(next);
    if (previous != free.end() && previous->first + previous->second > offset) throw std::runtime_error("JIT pool: range freed twice");
    // Drop the pages without unmapping either view; both aliases share them.
    madvise(writable + offset, aligned, MADV_FREE_REUSABLE);
    used -= aligned;
    std::size_t start = offset;
    std::size_t end = offset + aligned;
    if (previous != free.end() && previous->first + previous->second == start) {
        start = previous->first;
        free.erase(previous);
    }
    if (next != free.end() && next->first == end) {
        end = next->first + next->second;
        free.erase(next);
    }
    free.emplace(start, end - start);
    return true;
}

bool CodePool::Contains(const void* address) const {
    const auto* byte = static_cast<const std::uint8_t*>(address);
    return code != nullptr && byte >= code && byte < code + size;
}

std::size_t CodePool::Used() const {
    std::lock_guard lock(mutex);
    return used;
}

std::size_t CodePool::FreeRanges() const {
    std::lock_guard lock(mutex);
    return free.size();
}

bool CodePool::WriteInstruction(std::uintptr_t pc, std::uint32_t instruction) {
    if ((pc & 3u) != 0) return false;
    for (auto& slot : pools) {
        auto* pool = slot.load(std::memory_order_acquire);
        if (pool == nullptr || !pool->Contains(reinterpret_cast<const void*>(pc))) continue;
        auto* target = reinterpret_cast<std::atomic<std::uint32_t>*>(pc + pool->writeOffset);
        target->store(instruction, std::memory_order_release);
        sys_icache_invalidate(reinterpret_cast<void*>(pc), 4);
        return true;
    }
    return false;
}

std::uintptr_t CodePool::AllocateBeside(std::uintptr_t pc, std::size_t bytes, std::uintptr_t* writable) {
    for (auto& slot : pools) {
        auto* pool = slot.load(std::memory_order_acquire);
        if (pool == nullptr || !pool->Contains(reinterpret_cast<const void*>(pc))) continue;
        // Nearest free range: the stubs must be within branch range (+-128 MiB) of pc, and first fit in a 1 GiB pool
        // often is not (ASTRO BOT: JIT sites ~240 MiB into the pool got far stubs, one SIGTRAP per access).
        const auto rx = reinterpret_cast<std::uintptr_t>(pool->AllocateNear(pc, bytes, NearReach));
        if (rx == 0) return 0;
        *writable = rx + pool->writeOffset;
        return rx;
    }
    return 0;
}

}
