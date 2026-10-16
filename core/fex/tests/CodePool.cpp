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

// Displaced-access stub chunks must be within branch range of the JIT site: AllocateNear takes the free range nearest
// the pc (not the first one), splits it around the allocation and refuses beyond the reach.
void Nearest() {
    CodePool pool(64 * CodePool::Granule);
    const auto granule = CodePool::Granule;
    void* low = pool.Allocate(8 * granule); // [0, 8): the first free range is now [8, 64)
    const auto pc = reinterpret_cast<std::uintptr_t>(pool.Code()) + 40 * granule + 12;
    void* near = pool.AllocateNear(pc, granule, 64 * granule);
    Require(At(near) == pool.Code() + 40 * granule); // inside the free range, at pc's granule
    Require(pool.FreeRanges() == 2);                  // [8, 40) and [41, 64)
    Require(pool.Used() == 9 * granule);
    void* after = pool.AllocateNear(pc, granule, 64 * granule);
    Require(At(after) == pool.Code() + 41 * granule || At(after) == pool.Code() + 39 * granule);
    Require(pool.AllocateNear(pc, 60 * granule, 64 * granule) == nullptr);             // no free range that large
    Require(pool.AllocateNear(reinterpret_cast<std::uintptr_t>(pool.Code()), granule, granule) == nullptr); // free space beyond the reach
    Require(pool.AllocateNear(0x1000, granule, 64 * granule) == nullptr);               // pc outside the pool
    Require(pool.Free(near, granule) && pool.Free(after, granule) && pool.Free(low, 8 * granule));
    Require(pool.FreeRanges() == 1 && pool.Used() == 0);
}

// Displaced-access sites (libc GuestDisplaced) are released with the code that holds them: Free reports every range
// to the listener, with the rounded length, before the range can be allocated again.
std::uintptr_t listenedStart = 0, listenedEnd = 0;
std::size_t listened = 0;
std::size_t Listener(std::uintptr_t start, std::uintptr_t end) {
    listenedStart = start;
    listenedEnd = end;
    ++listened;
    return 0;
}

void FreeListener() {
    CodePool pool(16 * CodePool::Granule);
    CodePool::SetFreeListener(&Listener);
    void* buffer = pool.Allocate(3 * CodePool::Granule - 100);
    Require(listened == 0);
    Require(pool.Free(buffer, 3 * CodePool::Granule - 100));
    Require(listened == 1 && listenedStart == reinterpret_cast<std::uintptr_t>(buffer) &&
            listenedEnd == reinterpret_cast<std::uintptr_t>(buffer) + 3 * CodePool::Granule);
    int outside = 0;
    Require(!pool.Free(&outside, 4)); // not pool memory: not reported
    Require(listened == 1);
    CodePool::SetFreeListener(nullptr);
    void* again = pool.Allocate(CodePool::Granule);
    Require(pool.Free(again, CodePool::Granule) && listened == 1);
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
    Nearest();
    FreeListener();
    std::puts("code pool: all checks passed");
    return 0;
}
