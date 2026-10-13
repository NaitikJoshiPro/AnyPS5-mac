#include "prx/libc/include/GuestDisplaced.hpp"

#if defined(__APPLE__) && defined(__aarch64__)
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <libkern/OSCacheControl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>

extern "C" int __shared_region_check_np(std::uint64_t* startAddress); // libsystem_kernel; the shared cache's address

// D31, D33. Every 64-bit macOS arm64 process starts with permanent, inaccessible ranges the guest may ask for at
// fixed addresses: the VM_MEMORY_GUARD entry over [MACH_VM_MIN_GPU_CARVEOUT_ADDRESS, MACH_VM_MAX_GPU_CARVEOUT_ADDRESS)
// = [64 GiB, 448 GiB) and the commpage nesting guard [0xFC0000000, 64 GiB) (xnu vm_map_exec, vm_reserved_regions),
// which cannot be deallocated, overwritten or reprotected; and the dyld shared region, whose submap runs to
// 0xFC0000000 with nothing but inaccessible entries after the shared cache. The tail could be overwritten, but that is
// undocumented and depends on dyld; it is displaced like the guards. Guest memory in these ranges is backed at
// guest | DisplacedFlip, reserved here up front. The ranges are found at startup (the cache moves with its slide).
//
// A guest (or native) load/store to a displaced address faults with si_addr in the window. The handler builds a
// stub for that instruction that runs it with its base register flipped into the alias:
//
//   checked: sub sp, sp, #144          ; keep clear of the 128-byte red zone
//            stp xA, xB, [sp]
//            ldr xB, start             ; d = xN - start
//            sub xA, xN, xB
//            ldr xB, size              ; displaced iff d < size (unsigned): d - size borrows and d >= 0
//            sub xB, xA, xB
//            bic xB, xB, xA
//            tbz xB, #63, plain
//   flip:    eor xA, xN, #(1 << 40)
//            <instruction with Rn = xA>
//            [eor xN, xA, #(1 << 40)]  ; only for writeback forms
//            ldp xA, xB, [sp]
//            add sp, sp, #144
//            b next
//   plain:   ldp xA, xB, [sp]
//            add sp, sp, #144
//            <instruction>
//            b next
//   forced:  sub sp, sp, #144
//            stp xA, xB, [sp]
//            b flip
//   start:   .quad window start
//   size:    .quad window size
//
// The signal path resumes at `forced` (the base need not be the displaced operand of a register-offset form); a JIT
// site is then patched to `b checked`, so later executions cost ~12 instructions and no signal. Nothing in the stub
// touches NZCV. xA/xB are chosen outside every register field of the instruction. Stubs live in an RX view of
// memory written through an RW view (no MAP_JIT toggling), within branch range of the faulting instruction.
namespace GuestArena {
namespace {

std::uintptr_t g_start = CarveoutStart;

constexpr std::uint32_t kSlotWords = 32; // longest stub: 8 check + 8 flip + 6 plain + 3 forced + 1 pad + 4 literal words
constexpr std::size_t kChunkBytes = 1u << 20;
constexpr std::size_t kMaxChunks = 64;
constexpr std::size_t kTableSize = 1u << 15;
constexpr std::int64_t kBranchRange = (1ll << 27) - (1ll << 20);

struct Chunk {
    std::uintptr_t rx = 0;
    std::uintptr_t rw = 0;
    std::size_t used = 0;
    bool far = false; // out of branch range: stubs return through BRK and the SIGTRAP handler
};

constexpr std::uint32_t kReturnBrk = 0xd4200000u | (0xd15u << 5);

struct Site {
    std::uintptr_t pc = 0;
    std::uintptr_t stub = 0; // checked entry; forced entry at stub + forcedOffset
    std::uint32_t forcedOffset = 0;
    std::uint32_t instruction = 0; // JIT code at pc can be freed and reused for another instruction
    bool far = false;
};

constexpr std::size_t kMaxRanges = 4;
DisplacedRange g_ranges[kMaxRanges];
std::size_t g_rangeCount = 0;

std::atomic<bool> g_enabled{false};
std::atomic<bool> g_attempted{false};
std::atomic<std::uint64_t> g_faults{0};
std::atomic<std::uint64_t> g_patches{0};
std::atomic<bool (*)(std::uintptr_t, std::uint32_t)> g_writer{nullptr};
std::atomic<std::uintptr_t (*)(std::uintptr_t, std::size_t, std::uintptr_t*)> g_allocator{nullptr};
std::atomic_flag g_lock = ATOMIC_FLAG_INIT;
Chunk g_chunks[kMaxChunks];
std::size_t g_chunkCount = 0;
Site g_sites[kTableSize];
struct sigaction g_previousBus {}, g_previousSegv {};
std::once_flag g_once;
bool g_trace = false;

struct SpinLock {
    SpinLock() {
        while (g_lock.test_and_set(std::memory_order_acquire)) {
        }
    }
    ~SpinLock() { g_lock.clear(std::memory_order_release); }
};

bool InBranchRange(std::uintptr_t from, std::uintptr_t to) {
    const auto distance = static_cast<std::int64_t>(to) - static_cast<std::int64_t>(from);
    return distance > -kBranchRange && distance < kBranchRange;
}

std::uint32_t Branch(std::uintptr_t from, std::uintptr_t to) {
    const auto distance = static_cast<std::int64_t>(to) - static_cast<std::int64_t>(from);
    return 0x14000000u | (static_cast<std::uint32_t>(distance >> 2) & 0x03ffffffu);
}

// A chunk with an RW view and an RX view of the same pages; the RX view is placed near `pc`.
Chunk* NewChunkNear(std::uintptr_t pc) {
    if (g_chunkCount == kMaxChunks) return nullptr;
    // JIT code: the code pool hands out memory near its own code (the arena and the pool leave nothing free nearby).
    if (auto* allocator = g_allocator.load(std::memory_order_acquire)) {
        std::uintptr_t writable = 0;
        const auto rx = allocator(pc, kChunkBytes, &writable);
        if (rx != 0 && InBranchRange(pc, rx) && InBranchRange(pc, rx + kChunkBytes)) {
            auto& chunk = g_chunks[g_chunkCount++];
            chunk = Chunk{rx, writable, 0};
            return &chunk;
        }
    }
    mach_vm_address_t rw = 0;
    if (mach_vm_allocate(mach_task_self(), &rw, kChunkBytes, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) return nullptr;
    // Nearest free chunk-aligned spot first, alternating below and above; a fixed remap fails fast where it is taken.
    const std::uintptr_t origin = pc & ~static_cast<std::uintptr_t>(kChunkBytes - 1);
    for (std::uintptr_t step = 1; step < static_cast<std::uintptr_t>(kBranchRange) / kChunkBytes - 1; ++step) {
        for (const bool below : {true, false}) {
            const std::uintptr_t delta = step * kChunkBytes;
            if (below && origin < delta) continue;
            mach_vm_address_t rx = below ? origin - delta : origin + delta;
            vm_prot_t current = VM_PROT_NONE, maximum = VM_PROT_NONE;
            if (mach_vm_remap(mach_task_self(), &rx, kChunkBytes, 0, VM_FLAGS_FIXED, mach_task_self(), rw, FALSE, &current, &maximum, VM_INHERIT_NONE) != KERN_SUCCESS) continue;
            if (mach_vm_protect(mach_task_self(), rx, kChunkBytes, FALSE, VM_PROT_READ | VM_PROT_EXECUTE) != KERN_SUCCESS) {
                mach_vm_deallocate(mach_task_self(), rx, kChunkBytes);
                continue;
            }
            auto& chunk = g_chunks[g_chunkCount++];
            chunk = Chunk{rx, rw, 0};
            return &chunk;
        }
    }
    mach_vm_deallocate(mach_task_self(), rw, kChunkBytes);
    return nullptr;
}

// Native code in the dyld shared cache sits far from any free memory; its stubs go anywhere.
Chunk* NewFarChunk() {
    if (g_chunkCount == kMaxChunks) return nullptr;
    mach_vm_address_t rw = 0;
    if (mach_vm_allocate(mach_task_self(), &rw, kChunkBytes, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) return nullptr;
    mach_vm_address_t rx = 0;
    vm_prot_t current = VM_PROT_NONE, maximum = VM_PROT_NONE;
    if (mach_vm_remap(mach_task_self(), &rx, kChunkBytes, 0, VM_FLAGS_ANYWHERE, mach_task_self(), rw, FALSE, &current, &maximum, VM_INHERIT_NONE) != KERN_SUCCESS) {
        mach_vm_deallocate(mach_task_self(), rw, kChunkBytes);
        return nullptr;
    }
    if (mach_vm_protect(mach_task_self(), rx, kChunkBytes, FALSE, VM_PROT_READ | VM_PROT_EXECUTE) != KERN_SUCCESS) {
        mach_vm_deallocate(mach_task_self(), rx, kChunkBytes);
        mach_vm_deallocate(mach_task_self(), rw, kChunkBytes);
        return nullptr;
    }
    auto& chunk = g_chunks[g_chunkCount++];
    chunk = Chunk{rx, rw, 0, true};
    return &chunk;
}

std::uintptr_t AllocateSlot(std::uintptr_t pc, std::uint32_t*& writable, bool& far) {
    Chunk* chunk = nullptr;
    for (std::size_t i = 0; i < g_chunkCount; ++i) {
        auto& candidate = g_chunks[i];
        if (!candidate.far && candidate.used + kSlotWords * 4 <= kChunkBytes && InBranchRange(pc, candidate.rx) && InBranchRange(pc, candidate.rx + kChunkBytes)) {
            chunk = &candidate;
            break;
        }
    }
    if (chunk == nullptr) chunk = NewChunkNear(pc);
    if (chunk == nullptr) {
        for (std::size_t i = 0; i < g_chunkCount && chunk == nullptr; ++i) {
            if (g_chunks[i].far && g_chunks[i].used + kSlotWords * 4 <= kChunkBytes) chunk = &g_chunks[i];
        }
        if (chunk == nullptr) chunk = NewFarChunk();
    }
    if (chunk == nullptr) return 0;
    far = chunk->far;
    const auto offset = chunk->used;
    chunk->used += kSlotWords * 4;
    writable = reinterpret_cast<std::uint32_t*>(chunk->rw + offset);
    return chunk->rx + offset;
}

// A64 loads and stores: op0 = x1x0. Literal loads (PC-relative) cannot move into a stub.
bool IsLoadStore(std::uint32_t insn) {
    if ((insn & 0x0a000000u) != 0x08000000u) return false;
    if ((insn & 0x3b000000u) == 0x18000000u) return false;
    return true;
}

bool HasWriteback(std::uint32_t insn) {
    if ((insn & 0x3b200000u) == 0x38000000u) return (insn & 0x400u) != 0;  // imm9 post (01) / pre (11)
    if ((insn & 0x3a000000u) == 0x28000000u) return (insn & 0x00800000u) != 0; // pair post / pre
    if ((insn & 0xbe000000u) == 0x0c000000u) return (insn & 0x00800000u) != 0; // AdvSIMD structures, post-index
    return false;
}

Site* FindSite(std::uintptr_t pc, bool insert) {
    auto index = (pc >> 2) & (kTableSize - 1);
    for (std::size_t probe = 0; probe < kTableSize; ++probe, index = (index + 1) & (kTableSize - 1)) {
        auto& site = g_sites[index];
        if (site.pc == pc) return &site;
        if (site.pc == 0) {
            if (!insert) return nullptr;
            site.pc = pc;
            return &site;
        }
    }
    return nullptr;
}

Site* BuildSite(std::uintptr_t pc, std::uint32_t insn) {
    const unsigned rn = (insn >> 5) & 31u;
    bool busy[32] = {};
    for (const unsigned shift : {0u, 5u, 10u, 16u}) busy[(insn >> shift) & 31u] = true;
    unsigned scratch[2] = {32, 32};
    for (unsigned reg = 9, found = 0; reg <= 15 && found < 2; ++reg) {
        if (!busy[reg]) scratch[found++] = reg;
    }
    const unsigned a = scratch[0], b = scratch[1];
    std::uint32_t* code = nullptr;
    bool far = false;
    const auto stub = AllocateSlot(pc, code, far);
    if (stub == 0) {
        if (g_trace) (void)!write(2, "[displaced] no stub memory\n", 27);
        return nullptr;
    }
    std::uint32_t words[kSlotWords];
    std::uint32_t n = 0;
    const auto subSp = 0xd10243ffu;
    const auto addSp = 0x910243ffu;
    const auto stp = 0xa9000000u | (b << 10) | (31u << 5) | a;
    const auto ldp = 0xa9400000u | (b << 10) | (31u << 5) | a;
    const auto back = [&](std::uint32_t at) {
        if (!far) {
            words[at] = Branch(stub + at * 4, pc + 4);
            return at + 1;
        }
        const std::uint64_t target = pc + 4;
        words[at] = kReturnBrk;
        words[at + 1] = static_cast<std::uint32_t>(target);
        words[at + 2] = static_cast<std::uint32_t>(target >> 32);
        return at + 3;
    };
    const auto eorFlip = [](unsigned dst, unsigned src) { return 0xd2400000u | (24u << 16) | (src << 5) | dst; };
    const auto sub = [](unsigned dst, unsigned left, unsigned right) { return 0xcb000000u | (right << 16) | (left << 5) | dst; };
    words[n++] = subSp;
    words[n++] = stp;
    const auto ldrStartAt = n++;
    words[n++] = sub(a, rn, b);
    const auto ldrSizeAt = n++;
    words[n++] = sub(b, a, b);
    words[n++] = 0x8a200000u | (a << 16) | (b << 5) | b; // bic xB, xB, xA
    const auto tbzAt = n++;
    const auto flipAt = n;
    words[n++] = eorFlip(a, rn);
    words[n++] = (insn & ~(31u << 5)) | (a << 5);
    if (HasWriteback(insn)) words[n++] = eorFlip(rn, a);
    words[n++] = ldp;
    words[n++] = addSp;
    n = back(n);
    const auto plainAt = n;
    words[n++] = ldp;
    words[n++] = addSp;
    words[n++] = insn;
    n = back(n);
    const auto forcedAt = n;
    words[n++] = subSp;
    words[n++] = stp;
    words[n] = 0x14000000u | ((flipAt - n) & 0x03ffffffu);
    ++n;
    n = (n + 1) & ~1u; // 8-byte literals (slots are 128-byte aligned)
    const auto literalAt = n;
    const std::uint64_t window[2] = {DisplacedStart(), DisplacedEnd - DisplacedStart()};
    std::memcpy(&words[n], window, sizeof(window));
    n += 4;
    words[ldrStartAt] = 0x58000000u | ((literalAt - ldrStartAt) << 5) | b;
    words[ldrSizeAt] = 0x58000000u | ((literalAt + 2 - ldrSizeAt) << 5) | b;
    words[tbzAt] = 0xb6f80000u | (((plainAt - tbzAt) & 0x3fffu) << 5) | b; // tbz xB, #63
    std::memcpy(code, words, n * 4);
    sys_icache_invalidate(reinterpret_cast<void*>(stub), n * 4);
    auto* site = FindSite(pc, true);
    if (site == nullptr) {
        if (g_trace) (void)!write(2, "[displaced] site table full\n", 28);
        return nullptr;
    }
    site->stub = stub;
    site->forcedOffset = forcedAt * 4;
    site->instruction = insn;
    site->far = far;
    return site;
}

void Chain(int signal, siginfo_t* info, void* context) {
    const auto& previous = signal == SIGBUS ? g_previousBus : g_previousSegv;
    if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
        previous.sa_sigaction(signal, info, context);
        return;
    }
    if ((previous.sa_flags & SA_SIGINFO) == 0 && previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN && previous.sa_handler != nullptr) {
        previous.sa_handler(signal);
        return;
    }
    std::signal(signal, SIG_DFL);
}

bool Redirect(siginfo_t* info, void* context) {
    if (!IsDisplaced(reinterpret_cast<std::uintptr_t>(info->si_addr))) return false;
    auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
    const std::uintptr_t pc = arm_thread_state64_get_pc(machine->__ss);
    if (pc == 0 || (pc & 3u) != 0) return false;
    const auto insn = *reinterpret_cast<const std::uint32_t*>(pc);
    if (!IsLoadStore(insn) || ((insn >> 5) & 31u) == 31u) return false;
    std::uintptr_t forced = 0;
    std::uintptr_t checked = 0;
    bool far = false;
    {
        SpinLock lock;
        auto* site = FindSite(pc, false);
        if (site == nullptr || site->stub == 0 || site->instruction != insn) site = BuildSite(pc, insn);
        if (site == nullptr) return false;
        checked = site->stub;
        forced = site->stub + site->forcedOffset;
        far = site->far;
    }
    const auto fault = g_faults.fetch_add(1, std::memory_order_relaxed);
    if (g_trace && fault < 64) {
        char line[160];
        const int length = std::snprintf(line, sizeof(line), "[displaced] fault %llu pc 0x%llx insn 0x%08x addr 0x%llx\n", static_cast<unsigned long long>(fault),
            static_cast<unsigned long long>(pc), insn, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)));
        if (length > 0) (void)!write(2, line, static_cast<std::size_t>(length));
    }
    if (auto* writer = g_writer.load(std::memory_order_acquire); !far && writer != nullptr && writer(pc, Branch(pc, checked)))
        g_patches.fetch_add(1, std::memory_order_relaxed);
    arm_thread_state64_set_pc_fptr(machine->__ss, reinterpret_cast<void*>(forced));
    return true;
}

struct sigaction g_previousTrap {};

// A far stub ends in BRK #0xd15 followed by its 64-bit return address.
bool Return(void* context) {
    auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
    const std::uintptr_t pc = arm_thread_state64_get_pc(machine->__ss);
    bool inside = false;
    for (std::size_t i = 0; i < g_chunkCount && !inside; ++i)
        inside = g_chunks[i].far && pc >= g_chunks[i].rx && pc < g_chunks[i].rx + kChunkBytes;
    if (!inside || *reinterpret_cast<const std::uint32_t*>(pc) != kReturnBrk) return false;
    std::uint64_t target = 0;
    std::memcpy(&target, reinterpret_cast<const void*>(pc + 4), sizeof(target));
    arm_thread_state64_set_pc_fptr(machine->__ss, reinterpret_cast<void*>(target));
    return true;
}

void OnTrap(int signal, siginfo_t* info, void* context) {
    if (Return(context)) return;
    if ((g_previousTrap.sa_flags & SA_SIGINFO) != 0 && g_previousTrap.sa_sigaction != nullptr) {
        g_previousTrap.sa_sigaction(signal, info, context);
        return;
    }
    if ((g_previousTrap.sa_flags & SA_SIGINFO) == 0 && g_previousTrap.sa_handler != SIG_DFL && g_previousTrap.sa_handler != SIG_IGN && g_previousTrap.sa_handler != nullptr) {
        g_previousTrap.sa_handler(signal);
        return;
    }
    std::signal(signal, SIG_DFL);
}

void OnFault(int signal, siginfo_t* info, void* context) {
    const int saved = errno;
    const bool handled = Redirect(info, context);
    static std::atomic<int> unhandled{0};
    if (!handled && g_trace && unhandled.fetch_add(1) < 16) {
        auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
        char line[160];
        const int length = std::snprintf(line, sizeof(line), "[displaced] not ours: signal %d pc 0x%llx addr 0x%llx\n", signal,
            static_cast<unsigned long long>(arm_thread_state64_get_pc(machine->__ss)), static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)));
        if (length > 0) (void)!write(2, line, static_cast<std::size_t>(length));
    }
    errno = saved;
    if (!handled) Chain(signal, info, context);
}

struct Entry {
    mach_vm_address_t start = 0;
    mach_vm_size_t size = 0;
    natural_t depth = 0;
    vm_region_submap_info_data_64_t info{};
};

// The leaf entry at or above `address`, descending into submaps (the shared region is one).
bool LeafAt(mach_vm_address_t address, Entry& entry) {
    for (natural_t depth = 0; depth < 4;) {
        entry.start = address;
        entry.depth = depth;
        mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
        if (mach_vm_region_recurse(mach_task_self(), &entry.start, &entry.size, &entry.depth, reinterpret_cast<vm_region_recurse_info_t>(&entry.info), &count) != KERN_SUCCESS)
            return false;
        if (!entry.info.is_submap || entry.depth != depth) return true;
        ++depth;
    }
    return true;
}

bool Inaccessible(const Entry& entry) {
    return entry.info.protection == VM_PROT_NONE && entry.info.max_protection == VM_PROT_NONE;
}

// [start, end) is covered, without holes, by inaccessible entries only (submap contents included).
bool AllInaccessible(mach_vm_address_t start, mach_vm_address_t end) {
    for (auto cursor = start; cursor < end;) {
        Entry entry;
        if (!LeafAt(cursor, entry) || entry.start > cursor || !Inaccessible(entry)) return false;
        cursor = entry.start + entry.size;
    }
    return true;
}

// The shared region's tail: from the end of the shared cache's last accessible mapping to `limit`, when everything in
// between is an inaccessible entry of the shared region submap (no hole the host could map into later).
mach_vm_address_t SharedRegionTail(mach_vm_address_t limit) {
    std::uint64_t cache = 0;
    const bool cacheKnown = __shared_region_check_np(&cache) == 0 && cache >= DisplacedLowest && cache < limit;
    mach_vm_address_t tail = 0;
    mach_vm_address_t cursor = cacheKnown ? cache : DisplacedLowest;
    while (cursor < limit) {
        Entry leaf;
        if (!LeafAt(cursor, leaf) || leaf.start >= limit) return 0; // unmapped up to the limit
        if (leaf.start > cursor) tail = 0;                           // a hole the host can map into
        if (leaf.depth > 0 && Inaccessible(leaf)) {
            if (tail == 0) tail = leaf.start;
        } else {
            tail = 0;
        }
        cursor = leaf.start + leaf.size;
    }
    if (tail == 0 || (cacheKnown && cache >= tail)) return 0;
    // Cross-check with the cache header (dyld_cache_header: sharedRegionStart at 224, sharedRegionSize at 232; the
    // first mapping's unslid address gives the slide): the cache's own extent must lie below the tail.
    if (cacheKnown && std::memcmp(reinterpret_cast<const void*>(cache), "dyld_v1", 7) == 0) {
        const auto* header = reinterpret_cast<const std::uint8_t*>(cache);
        std::uint32_t mappingOffset = 0;
        std::uint64_t regionStart = 0, regionSize = 0, unslid = 0;
        std::memcpy(&mappingOffset, header + 16, sizeof(mappingOffset));
        std::memcpy(&regionStart, header + 224, sizeof(regionStart));
        std::memcpy(&regionSize, header + 232, sizeof(regionSize));
        if (mappingOffset >= 248 && mappingOffset < 0x4000) std::memcpy(&unslid, header + mappingOffset, sizeof(unslid));
        if (unslid != 0 && cache >= unslid && regionStart + regionSize + (cache - unslid) > tail) return 0;
    }
    return tail;
}

bool Enable() {
    g_trace = std::getenv("APS5_DISPLACED_TRACE") != nullptr;
    // Only needed where the ranges really are host guards: probe them. The carveout is the anchor; the commpage guard
    // and the shared region's tail join only when each is contiguous with the next range up.
    if (!AllInaccessible(CarveoutStart, DisplacedEnd)) return false;
    std::size_t count = 0;
    DisplacedRange found[kMaxRanges];
    found[count++] = {CarveoutStart, DisplacedEnd, CarveoutStart | DisplacedFlip, "GPU carveout"};
    constexpr std::uintptr_t CommpageStart = 0x0000000FC0000000ull; // _COMM_PAGE64_NESTING_START
    if (std::getenv("APS5_DISPLACED_CARVEOUT_ONLY") == nullptr && AllInaccessible(CommpageStart, CarveoutStart)) {
        found[count++] = {CommpageStart, CarveoutStart, CommpageStart | DisplacedFlip, "commpage guard"};
        if (const auto tail = SharedRegionTail(CommpageStart); tail != 0)
            found[count++] = {static_cast<std::uintptr_t>(tail), CommpageStart, static_cast<std::uintptr_t>(tail) | DisplacedFlip, "shared region tail"};
    }
    // The alias of the whole window, lowest range first; drop the lower ranges if their alias is taken.
    while (count > 0) {
        const auto start = found[count - 1].guestStart;
        mach_vm_address_t alias = start | DisplacedFlip;
        if (mach_vm_map(mach_task_self(), &alias, DisplacedEnd - start, 0, VM_FLAGS_FIXED, MEMORY_OBJECT_NULL, 0, FALSE, VM_PROT_NONE, VM_PROT_ALL, VM_INHERIT_DEFAULT) == KERN_SUCCESS)
            break;
        std::fprintf(stderr, "[memory] the displaced alias 0x%llx-0x%llx is occupied; fixed guest mappings in the %s will fail\n",
            static_cast<unsigned long long>(start | DisplacedFlip), static_cast<unsigned long long>(DisplacedEnd | DisplacedFlip), found[count - 1].name);
        --count;
    }
    if (count == 0) return false;
    for (std::size_t i = 0; i < count; ++i) g_ranges[i] = found[count - 1 - i];
    g_rangeCount = count;
    g_start = g_ranges[0].guestStart;
    struct sigaction action {};
    action.sa_sigaction = OnFault;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    sigaction(SIGBUS, &action, &g_previousBus);
    sigaction(SIGSEGV, &action, &g_previousSegv);
    action.sa_sigaction = OnTrap;
    sigaction(SIGTRAP, &action, &g_previousTrap);
    if (g_trace) {
        for (std::size_t i = 0; i < g_rangeCount; ++i)
            std::fprintf(stderr, "[displaced] %s 0x%llx-0x%llx at 0x%llx\n", g_ranges[i].name, static_cast<unsigned long long>(g_ranges[i].guestStart),
                static_cast<unsigned long long>(g_ranges[i].guestEnd), static_cast<unsigned long long>(g_ranges[i].hostBase));
        std::fprintf(stderr, "[displaced] fault handler installed (previous SIGSEGV %p)\n", reinterpret_cast<void*>(g_previousSegv.sa_sigaction));
    }
    return true;
}

}

bool GuestDisplacedEnable_nid_no_patch() {
    std::call_once(g_once, [] {
        g_enabled.store(Enable(), std::memory_order_release);
        g_attempted.store(true, std::memory_order_release);
    });
    return g_enabled.load(std::memory_order_acquire);
}

bool GuestDisplacedActive_nid_no_patch() {
    return g_enabled.load(std::memory_order_acquire);
}

std::uintptr_t GuestDisplacedStart_nid_no_patch() {
    return g_start; // written once, before any guest code, by the arena's static initialisation
}

void GuestDisplacedSetCodeWriter_nid_no_patch(bool (*writer)(std::uintptr_t, std::uint32_t), std::uintptr_t (*allocator)(std::uintptr_t, std::size_t, std::uintptr_t*)) {
    g_allocator.store(allocator, std::memory_order_release);
    g_writer.store(writer, std::memory_order_release);
}

void GuestDisplacedStats_nid_no_patch(std::uint64_t* faults, std::uint64_t* patches) {
    if (faults != nullptr) *faults = g_faults.load(std::memory_order_relaxed);
    if (patches != nullptr) *patches = g_patches.load(std::memory_order_relaxed);
}

void GuestDisplacedRanges_nid_no_patch(const DisplacedRange** ranges, std::size_t* count) {
    const bool active = g_enabled.load(std::memory_order_acquire);
    if (ranges != nullptr) *ranges = g_ranges;
    if (count != nullptr) *count = active ? g_rangeCount : 0;
}

}
#else
namespace GuestArena {
std::uintptr_t GuestDisplacedStart_nid_no_patch() { return 0; }
bool GuestDisplacedEnable_nid_no_patch() { return false; }
bool GuestDisplacedActive_nid_no_patch() { return false; }
void GuestDisplacedSetCodeWriter_nid_no_patch(bool (*)(std::uintptr_t, std::uint32_t), std::uintptr_t (*)(std::uintptr_t, std::size_t, std::uintptr_t*)) {}
void GuestDisplacedStats_nid_no_patch(std::uint64_t* faults, std::uint64_t* patches) {
    if (faults != nullptr) *faults = 0;
    if (patches != nullptr) *patches = 0;
}
void GuestDisplacedRanges_nid_no_patch(const DisplacedRange** ranges, std::size_t* count) {
    if (ranges != nullptr) *ranges = nullptr;
    if (count != nullptr) *count = 0;
}
}
#endif
