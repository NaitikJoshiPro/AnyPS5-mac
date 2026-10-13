#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

namespace Aps5Fex {

// The JIT's executable memory: one region mapped twice, read-execute where code runs and read-write at a fixed
// offset where code is written (macOS allows no writable-executable pages without MAP_JIT toggling per thread).
// FEXCore allocates its code buffers through the allocator hooks; the pool hands out 16 KiB-granular ranges
// first fit and takes freed ranges back, coalescing neighbours, so buffers dropped by ClearCodeCache are reused.
class CodePool {
public:
    static constexpr std::size_t Granule = 0x4000;

    explicit CodePool(std::size_t size);
    ~CodePool();

    CodePool(const CodePool&) = delete;
    CodePool& operator=(const CodePool&) = delete;

    // Returns the executable address of a new range of at least length bytes; throws std::runtime_error when the
    // pool has no free range that large.
    void* Allocate(std::size_t length);

    // Returns a range (or part of one) to the pool. False when address is outside the pool; throws on a range
    // that is not allocated.
    bool Free(void* address, std::size_t length);

    bool Contains(const void* address) const;
    // Replaces one instruction word of JIT code at its executable address in any live pool (async-signal-safe;
    // used to backpatch displaced guest accesses, GuestDisplaced.hpp). False outside every pool.
    static bool WriteInstruction(std::uintptr_t pc, std::uint32_t instruction);
    // Allocates bytes from the pool that contains pc (never freed); returns the executable address and stores the
    // writable one, or 0 when pc is in no pool or the pool is full.
    static std::uintptr_t AllocateBeside(std::uintptr_t pc, std::size_t bytes, std::uintptr_t* writable);
    std::uint8_t* Code() const { return code; }
    std::int64_t WriteOffset() const { return writeOffset; }
    std::size_t Size() const { return size; }
    std::size_t Used() const;
    std::size_t FreeRanges() const;

private:
    std::size_t size;
    std::uint8_t* code = nullptr;
    std::uint8_t* writable = nullptr;
    std::int64_t writeOffset = 0;
    mutable std::mutex mutex;
    std::map<std::size_t, std::size_t> free; // offset -> length, disjoint and never adjacent
    std::size_t used = 0;
};

}
