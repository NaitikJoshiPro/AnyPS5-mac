// The runner's dual-mapped JIT pool: first-fit allocation at 16 KiB granularity, reuse of freed ranges with
// coalescing, both views sharing memory, exhaustion as an exception; and FEXCore's VirtualDontNeed zeroing memory on
// macOS as FEX's lookup-cache and call-return-stack resets assume.
#include "CodePool.hpp"

#include <FEXCore/Utils/AllocatorHooks.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <source_location>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>
#include <vector>

namespace {

using Aps5Fex::CodePool;

void Require(bool value, std::source_location where = std::source_location::current()) {
    if (!value) {
        std::fprintf(stderr, "code pool check failed at line %u\n", static_cast<unsigned>(where.line()));
        std::abort();
    }
}

template <typename Function>
bool Throws(Function function) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

std::uint8_t* At(void* address) {
    return static_cast<std::uint8_t*>(address);
}

void Granularity() {
    CodePool pool(16 * CodePool::Granule);
    Require(pool.Size() == 16 * CodePool::Granule);
    void* first = pool.Allocate(1);
    void* second = pool.Allocate(CodePool::Granule + 1);
    Require(first == pool.Code());
    Require(At(second) == pool.Code() + CodePool::Granule);
    Require(pool.Used() == 3 * CodePool::Granule);
}

void DualMapping() {
    CodePool pool(4 * CodePool::Granule);
    auto* code = At(pool.Allocate(CodePool::Granule));
    // ret (0xd65f03c0) written through the writable view, read and run through the executable one.
    const std::uint32_t ret = 0xd65f03c0;
    std::memcpy(code + pool.WriteOffset(), &ret, sizeof(ret));
    __builtin___clear_cache(reinterpret_cast<char*>(code), reinterpret_cast<char*>(code + 4));
    std::uint32_t read = 0;
    std::memcpy(&read, code, sizeof(read));
    Require(read == ret);
    reinterpret_cast<void (*)()>(code)();
}

void ReuseAndCoalesce() {
    CodePool pool(8 * CodePool::Granule);
    void* a = pool.Allocate(2 * CodePool::Granule);
    void* b = pool.Allocate(2 * CodePool::Granule);
    void* c = pool.Allocate(2 * CodePool::Granule);
    Require(pool.FreeRanges() == 1);
    Require(Throws([&] { pool.Allocate(4 * CodePool::Granule); }));
    Require(pool.Free(a, 2 * CodePool::Granule));
    Require(pool.Free(c, 2 * CodePool::Granule));
    Require(pool.FreeRanges() == 2); // [a] and [c..end]: c merged with the tail
    Require(pool.Free(b, 2 * CodePool::Granule));
    Require(pool.FreeRanges() == 1);
    Require(pool.Used() == 0);
    // Everything merged: the whole pool fits again, at the start.
    Require(pool.Allocate(8 * CodePool::Granule) == pool.Code());
}

void FirstFitReusesFreedBuffer() {
    // FEX's sequence: buffers of 1, 2, 4 units, then each ClearCodeCache drops the oldest and asks for 4 more. Three
    // live buffers of 4 need 12 units; first fit settles into a cycle with 16 (a bump allocator would need 4000).
    CodePool pool(16 * CodePool::Granule);
    std::vector<std::pair<void*, std::size_t>> live;
    for (std::size_t units : {1, 2, 4}) live.emplace_back(pool.Allocate(units * CodePool::Granule), units * CodePool::Granule);
    for (int round = 0; round < 1000; ++round) {
        Require(pool.Free(live.front().first, live.front().second));
        live.erase(live.begin());
        live.emplace_back(pool.Allocate(4 * CodePool::Granule), 4 * CodePool::Granule);
    }
    Require(pool.Used() == 12 * CodePool::Granule);
}

void BadFrees() {
    CodePool pool(4 * CodePool::Granule);
    void* a = pool.Allocate(CodePool::Granule);
    int outside = 0;
    Require(!pool.Free(&outside, sizeof(outside)));
    Require(Throws([&] { pool.Free(At(a) + 16, 16); }));
    Require(pool.Free(a, CodePool::Granule));
    Require(Throws([&] { pool.Free(a, CodePool::Granule); }));
    Require(Throws([&] { pool.Free(pool.Code() + 3 * CodePool::Granule, 2 * CodePool::Granule); }));
}

void Exhaustion() {
    CodePool pool(2 * CodePool::Granule);
    pool.Allocate(2 * CodePool::Granule);
    bool named = false;
    try {
        pool.Allocate(1);
    } catch (const std::runtime_error& error) {
        named = std::strstr(error.what(), "JIT pool exhausted") != nullptr;
    }
    Require(named);
}

void Threads() {
    CodePool pool(64 * CodePool::Granule);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&pool] {
            for (int i = 0; i < 2000; ++i) {
                const std::size_t length = (1 + i % 3) * CodePool::Granule;
                void* range = pool.Allocate(length);
                Require(pool.Free(range, length));
            }
        });
    for (auto& thread : threads) thread.join();
    Require(pool.Used() == 0 && pool.FreeRanges() == 1);
}

void DontNeedZeroes() {
    const std::size_t page = static_cast<std::size_t>(getpagesize());
    const std::size_t size = 8 * page;
    auto* memory = static_cast<std::uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
    Require(memory != MAP_FAILED);
    std::memset(memory, 0x5a, size);
    // Whole pages, then a range with partial pages at both ends; the bytes outside must stay.
    FEXCore::Allocator::VirtualDontNeed(memory, 2 * page);
    for (std::size_t i = 0; i < 2 * page; ++i) Require(memory[i] == 0);
    Require(memory[2 * page] == 0x5a);
    FEXCore::Allocator::VirtualDontNeed(memory + 3 * page - 24, 2 * page + 48, false);
    Require(memory[3 * page - 25] == 0x5a && memory[5 * page + 24] == 0x5a);
    for (std::size_t i = 3 * page - 24; i < 5 * page + 24; ++i) Require(memory[i] == 0);
    // Smaller than a page.
    FEXCore::Allocator::VirtualDontNeed(memory + 6 * page + 8, 64);
    Require(memory[6 * page + 7] == 0x5a && memory[6 * page + 8] == 0 && memory[6 * page + 71] == 0 && memory[6 * page + 72] == 0x5a);
    munmap(memory, size);
}

}

int main() {
    Granularity();
    DualMapping();
    ReuseAndCoalesce();
    FirstFitReusesFreedBuffer();
    BadFrees();
    Exhaustion();
    Threads();
    DontNeedZeroes();
    std::puts("code pool: all checks passed");
    return 0;
}
