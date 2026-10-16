#include "prx/libc/include/GuestDisplaced.hpp"

#if defined(__APPLE__) && defined(__aarch64__)
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <libkern/OSCacheControl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <dlfcn.h>
#include <mutex>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <time.h>
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
// stub for that instruction that runs it with its base register flipped into the alias (one scratch register):
//
//   checked: str xA, [sp, #-144]!      ; keep clear of the 128-byte red zone
//            lsr xA, xN, #36           ; xN >= DisplacedEnd (7 << 36): ordinary memory (the arena), leave
//            sub xA, xA, #7
//            tbz xA, #63, plain
//            lsr xA, xN, #14           ; the window's start is 16 KiB aligned: compare in pages
//            sub xA, xA, #hi, lsl #12  ; (start >> 14) as two 12-bit immediates
//            sub xA, xA, #lo
//            tbnz xA, #63, plain       ; xN < start: ordinary memory
//   flip:    eor xA, xN, #(1 << 40)
//            [nop]                     ; barrier slot before an ordered store
//            <instruction with Rn = xA>
//            [nop]                     ; barrier slot after an ordered load
//            [eor xN, xA, #(1 << 40)]  ; only for writeback forms
//            ldr xA, [sp], #144
//            b next                    ; (far stubs: b tail)
//   plain:   ldr xA, [sp], #144
//            [nop] <instruction> [nop]
//            b next                    ; (far stubs: tail)
//   forced:  str xA, [sp, #-144]!
//            b flip
//
// A register-offset access ([xN, xM, ext #s]) checks the address it really accesses: each `lsr` above reads
// xA = xN + ext(xM) << s (an `add` of the same extend first), so a small base plus a displaced index takes the flip
// path (eor of xN then adds 1 << 40 to the sum) and a displaced base plus an index that leaves the window does not.
// Before, the check looked at xN alone, so `[small + pointer]` sites went down the plain path, faulted inside the
// stub, and every execution built one more stub for that copy (a chain that grew by one site per execution).
//
// Sites are forgotten and their stubs freed when the code they patch is freed (GuestDisplacedCodeFreed: FEX's
// code buffers, through the code pool), so the table and the stub chunks hold only live code's sites.
//
// The signal path resumes at `forced` (the base need not be the displaced operand of a register-offset form); a JIT
// site is then patched to `b checked`, so later executions cost a few instructions and no signal (ordinary pointers
// above the window: 6 plus the access). Nothing in the stub touches NZCV. xA is chosen outside every register field
// of the instruction. Stubs live in an RX view of memory written through an RW view (no MAP_JIT toggling), within
// branch range of the faulting instruction.
//
// Native code in the dyld shared cache has no free memory within branch range, so its stubs are "far". A far stub
// returns by branch when the native code ahead overwrites a register without reading it within a few relocatable
// instructions (FarReturn): the tail runs those instructions, loads the return address into that dead register and
// `br`s to it. Otherwise the tail is BRK #0xd15 + the return address, and the SIGTRAP handler sets the pc.
//
// FEX's TSO emulation emits ordered accesses (LDAPUR/STLUR, LDAR/LDAPR/STLR), which raise SIGBUS (BUS_ADRALN) on
// the M3 when they cross 16 bytes. FEX's own handler only rewrites code in its code buffers, so a copy in a stub
// carries a NOP slot next to it, and FixUnaligned rewrites the copy the way FEX rewrites its sites (HalfBarrier):
// LDUR + DMB ISHLD for a load, DMB ISH + STUR for a store, for good.
namespace GuestArena {
namespace {

std::uintptr_t g_start = CarveoutStart;

constexpr std::uint32_t kSlotWords = 36; // longest stub: 10 check + 7 flip + 4 plain + 8 tail + 2 forced + 1 pad + 2 literal words (+ spare); 16-byte multiple
constexpr std::size_t kChunkBytes = 1u << 20;
constexpr std::size_t kMaxChunks = 512; // 64K stubs per chunk: chunks are reserved lazily, 1 MiB each
// Patched sites, open addressing. ASTRO BOT keeps most of its heap in displaced memory and patched 32k JIT sites
// in its first minute; a full table left new sites unhandled (a SIGBUS/SIGSEGV that looked like an unaligned
// access). 1M entries of 32 bytes are 32 MiB of zero-fill memory, touched only where sites land.
constexpr std::size_t kTableSize = 1u << 20;
constexpr std::int64_t kBranchRange = (1ll << 27) - (1ll << 20);

struct Chunk {
    std::uintptr_t rx = 0;
    std::uintptr_t rw = 0;
    std::size_t used = 0;
    bool far = false; // out of branch range: stubs return through BRK and the SIGTRAP handler
    std::size_t bytes = kChunkBytes;
    std::uintptr_t freeHead = 0; // freed slots (rx addresses), linked through their writable view
    std::size_t freeSlots = 0;
};

constexpr std::uint32_t kFreedSlotBrk = 0xd4200000u | (0xd16u << 5); // first word of a freed slot: a stray entry traps

constexpr std::uint32_t kReturnBrk = 0xd4200000u | (0xd15u << 5);

struct Site {
    std::uintptr_t pc = 0;
    std::uintptr_t stub = 0; // checked entry; forced entry at stub + forcedOffset
    std::uint32_t forcedOffset = 0;
    std::uint32_t instruction = 0; // JIT code at pc can be freed and reused for another instruction
    bool far = false;
    bool inStub = false; // pc is inside another site's stub
};

constexpr std::size_t kMaxRanges = 4;
DisplacedRange g_ranges[kMaxRanges];
std::size_t g_rangeCount = 0;

std::atomic<bool> g_enabled{false};
std::atomic<bool> g_attempted{false};
std::atomic<std::uint64_t> g_faults{0};
std::atomic<std::uint64_t> g_patches{0};
std::atomic<std::uint64_t> g_unaligned{0}; // ordered stub accesses rewritten to FEX's half-barrier form
std::atomic<bool (*)(std::uintptr_t, std::uint32_t)> g_writer{nullptr};
std::atomic<std::uintptr_t (*)(std::uintptr_t, std::size_t, std::uintptr_t*)> g_allocator{nullptr};
std::atomic_flag g_lock = ATOMIC_FLAG_INIT;
Chunk g_chunks[kMaxChunks];
std::size_t g_chunkCount = 0;
Site g_sites[kTableSize];
// Live sites are capped below the table size: linear probing degrades sharply near full, and every probe runs in a
// signal handler under the lock. Tests lower both caps (GuestDisplacedTestLimits) to show reclamation.
constexpr std::size_t kDefaultSiteLimit = kTableSize - kTableSize / 8;
std::size_t g_siteLimit = kDefaultSiteLimit;
std::size_t g_chunkLimit = kMaxChunks;
std::size_t g_liveSites = 0, g_liveStubSites = 0; // under the lock
std::atomic<std::uint64_t> g_forgotten{0};
struct sigaction g_previousBus {}, g_previousSegv {};
std::once_flag g_once;
bool g_trace = false;
std::uint64_t g_stackEvery = 0; // APS5_DISPLACED_STACKS=N: a frame-pointer backtrace of every Nth fault (32 at most)
std::uint64_t g_nativeStackEvery = 0; // APS5_DISPLACED_STACKS_NATIVE=N: the same, counting only unpatched faults
std::uintptr_t g_cacheStart = 0, g_cacheEnd = 0; // the dyld shared cache's immutable code: [cache, DisplacedStart())
std::uint64_t g_enabledAt = 0;  // CLOCK_MONOTONIC_RAW ns when the handler was installed

// Fault-rate milestones, always on: a line at 1024 faults and every power of two after it, and every 8192. Native
// code that reads displaced memory without a host view shows up in any title log as a fast-growing count (each fault
// is two signals, ~2-4 us). clock_gettime_nsec_np and snprintf+write are safe here (no locks, no allocation).
std::atomic<std::uint64_t> g_farReturns{0}; // far stubs return through BRK #0xd15: one SIGTRAP per execution

std::atomic<std::uint64_t> g_unpatched{0};      // faults at sites that stay unpatched (native code): one signal each, every time
std::atomic<std::uint64_t> g_farBranchSites{0}; // far stubs that return by branch through a dead register

// One line with every counter, so the last line of a log brackets the totals.
void ReportMilestone(std::uint64_t count, const char* what = "faults", std::uint64_t first = 1024) {
    if (count < first || ((count & (count - 1)) != 0 && (count & 8191) != 0)) return;
    const auto seconds = static_cast<double>(clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) - g_enabledAt) / 1e9;
    char line[384];
    const int length = std::snprintf(line, sizeof(line),
        "[displaced] %llu %s %.1f s after enable (faults %llu, sites patched %llu, unpatched faults %llu, far-stub SIGTRAP returns %llu, branch-return far sites %llu, "
        "live sites %llu, sites in stubs %llu, forgotten %llu)\n",
        static_cast<unsigned long long>(count), what, seconds, static_cast<unsigned long long>(g_faults.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_patches.load(std::memory_order_relaxed)), static_cast<unsigned long long>(g_unpatched.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_farReturns.load(std::memory_order_relaxed)), static_cast<unsigned long long>(g_farBranchSites.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(__atomic_load_n(&g_liveSites, __ATOMIC_RELAXED)), static_cast<unsigned long long>(__atomic_load_n(&g_liveStubSites, __ATOMIC_RELAXED)),
        static_cast<unsigned long long>(g_forgotten.load(std::memory_order_relaxed)));
    if (length > 0) (void)!write(2, line, std::min(static_cast<std::size_t>(length), sizeof(line) - 1));
}

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
    if (g_chunkCount >= g_chunkLimit) return nullptr;
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
    if (g_chunkCount >= g_chunkLimit) return nullptr;
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

std::uintptr_t g_testFarStart = 0, g_testFarEnd = 0; // tests: code treated like the shared cache (far, immutable)

bool InTestFarCode(std::uintptr_t pc) {
    return pc >= g_testFarStart && pc < g_testFarEnd;
}

bool HasRoom(const Chunk& chunk) {
    return chunk.freeHead != 0 || chunk.used + kSlotWords * 4 <= chunk.bytes;
}

Chunk* ChunkOf(std::uintptr_t pc) {
    for (std::size_t i = 0; i < g_chunkCount; ++i)
        if (pc >= g_chunks[i].rx && pc < g_chunks[i].rx + g_chunks[i].bytes) return &g_chunks[i];
    return nullptr;
}

// Freed slots first (a stub whose code was freed), then fresh space. Caller holds the lock.
std::uintptr_t AllocateSlot(std::uintptr_t pc, std::uint32_t*& writable, bool& far) {
    Chunk* chunk = nullptr;
    const bool forceFar = InTestFarCode(pc);
    for (std::size_t i = 0; i < g_chunkCount && !forceFar; ++i) {
        auto& candidate = g_chunks[i];
        if (!candidate.far && HasRoom(candidate) && InBranchRange(pc, candidate.rx) && InBranchRange(pc, candidate.rx + candidate.bytes)) {
            chunk = &candidate;
            break;
        }
    }
    if (chunk == nullptr && !forceFar) chunk = NewChunkNear(pc);
    if (chunk == nullptr) {
        for (std::size_t i = 0; i < g_chunkCount && chunk == nullptr; ++i) {
            if (g_chunks[i].far && HasRoom(g_chunks[i])) chunk = &g_chunks[i];
        }
        if (chunk == nullptr) chunk = NewFarChunk();
    }
    if (chunk == nullptr) return 0;
    far = chunk->far;
    if (chunk->freeHead != 0) {
        const auto slot = chunk->freeHead;
        writable = reinterpret_cast<std::uint32_t*>(chunk->rw + (slot - chunk->rx));
        std::memcpy(&chunk->freeHead, writable + 2, sizeof(chunk->freeHead));
        --chunk->freeSlots;
        return slot;
    }
    const auto offset = chunk->used;
    chunk->used += kSlotWords * 4;
    writable = reinterpret_cast<std::uint32_t*>(chunk->rw + offset);
    return chunk->rx + offset;
}

// Puts a stub's slot on its chunk's free list; its first word becomes a BRK so a stray branch into it traps instead
// of running a stale access. Caller holds the lock.
void FreeSlot(std::uintptr_t stub) {
    auto* chunk = ChunkOf(stub);
    if (chunk == nullptr) return;
    auto* writable = reinterpret_cast<std::uint32_t*>(chunk->rw + (stub - chunk->rx));
    __atomic_store_n(writable, kFreedSlotBrk, __ATOMIC_RELEASE);
    std::memcpy(writable + 2, &chunk->freeHead, sizeof(chunk->freeHead));
    sys_icache_invalidate(reinterpret_cast<void*>(stub), 4);
    chunk->freeHead = stub;
    ++chunk->freeSlots;
}

// A64 loads and stores: op0 = x1x0. Literal loads (PC-relative) cannot move into a stub.
bool IsLoadStore(std::uint32_t insn) {
    if ((insn & 0x0a000000u) != 0x08000000u) return false;
    if ((insn & 0x3b000000u) == 0x18000000u) return false;
    return true;
}

constexpr std::uint32_t kNop = 0xd503201fu;
constexpr std::uint32_t kDmbIsh = 0xd5033bbfu;   // FEX's DMB (inner shareable, all)
constexpr std::uint32_t kDmbIshLd = 0xd50339bfu; // FEX's DMB_LD (inner shareable, loads)

enum class Ordered { None, Load, Store };

// The ordered single-register accesses FEX's unaligned handler rewrites (FEX ArchHelpers/Arm64.cpp: LDAR, LDAPR,
// STLR, LDAPUR, STLUR; all sizes). Writeback forms do not exist among them.
Ordered OrderedAccess(std::uint32_t insn) {
    if ((insn & 0x3ffffc00u) == 0x08dffc00u || (insn & 0x3ffffc00u) == 0x38bfc000u) return Ordered::Load; // LDAR, LDAPR
    if ((insn & 0x3ffffc00u) == 0x089ffc00u) return Ordered::Store;                                      // STLR
    if ((insn & 0x3fe00c00u) == 0x19400000u) return Ordered::Load;                                       // LDAPUR
    if ((insn & 0x3fe00c00u) == 0x19000000u) return Ordered::Store;                                      // STLUR
    return Ordered::None;
}

// The plain form: LDUR/STUR with the same size, registers and (for LDAPUR/STLUR) offset.
std::uint32_t Unordered(std::uint32_t insn, Ordered kind) {
    const bool rcpc2 = (insn & 0x3f000000u) == 0x19000000u;
    const auto offset = rcpc2 ? insn & (0x1ffu << 12) : 0u;
    return (insn & 0xc0000000u) | (kind == Ordered::Load ? 0x38400000u : 0x38000000u) | offset | (insn & 0x3ffu);
}

bool IsUnorderedOf(std::uint32_t insn, Ordered kind) {
    return (insn & 0x3fe00c00u) == (kind == Ordered::Load ? 0x38400000u : 0x38000000u);
}

bool HasWriteback(std::uint32_t insn) {
    if ((insn & 0x3b200000u) == 0x38000000u) return (insn & 0x400u) != 0;  // imm9 post (01) / pre (11)
    if ((insn & 0x3a000000u) == 0x28000000u) return (insn & 0x00800000u) != 0; // pair post / pre
    if ((insn & 0xbe000000u) == 0x0c000000u) return (insn & 0x00800000u) != 0; // AdvSIMD structures, post-index
    return false;
}

std::size_t Home(std::uintptr_t pc) {
    return (pc >> 2) & (kTableSize - 1);
}

// Open addressing with linear probing; entries are removed by backward shift (no tombstones), so lookups stay short
// however many sites come and go. Caller holds the lock.
Site* FindSite(std::uintptr_t pc, bool insert) {
    auto index = Home(pc);
    for (std::size_t probe = 0; probe < kTableSize; ++probe, index = (index + 1) & (kTableSize - 1)) {
        auto& site = g_sites[index];
        if (site.pc == pc) return &site;
        if (site.pc == 0) {
            if (!insert || g_liveSites >= g_siteLimit) return nullptr;
            site = Site{};
            site.pc = pc;
            ++g_liveSites;
            return &site;
        }
    }
    return nullptr;
}

// Empties slot `hole` and moves later entries of its probe run back so that every entry stays reachable from its home.
void Erase(std::size_t hole) {
    if (g_sites[hole].inStub) --g_liveStubSites;
    --g_liveSites;
    for (auto next = (hole + 1) & (kTableSize - 1);; next = (next + 1) & (kTableSize - 1)) {
        const auto& candidate = g_sites[next];
        if (candidate.pc == 0) break;
        const auto home = Home(candidate.pc);
        // The candidate may fill the hole unless its home lies cyclically in (hole, next].
        const bool homeAfterHole = hole <= next ? (home > hole && home <= next) : (home > hole || home <= next);
        if (homeAfterHole) continue;
        g_sites[hole] = candidate;
        hole = next;
    }
    g_sites[hole] = Site{};
}

// Forgets the sites with pc in [start, end) and frees their stubs; then the sites inside those stubs (a stub's own
// copied access that faulted), down to a small depth. Caller holds the lock with signals blocked.
std::size_t ForgetLocked(std::uintptr_t start, std::uintptr_t end, unsigned depth) {
    if (end <= start) return 0;
    constexpr std::size_t kNested = 64;
    std::uintptr_t freed[kNested];
    std::size_t freedCount = 0, forgotten = 0;
    bool nestedOverflow = false;
    // Every site in range has its home in the `span` slots from Home(start) and lies in the probe run that starts at
    // its home, so the scan covers those slots and then runs on to the first empty one. Erase only moves entries
    // into the hole being examined or into later holes, so rechecking the hole finds everything.
    const std::size_t span = std::min<std::size_t>(kTableSize, (end - start + 3) / 4);
    auto index = Home(start);
    for (std::size_t visited = 0; visited < kTableSize + span;) {
        auto& site = g_sites[index];
        if (site.pc == 0 && visited >= span) break;
        if (site.pc != 0 && site.pc >= start && site.pc < end) {
            if (site.stub != 0) {
                FreeSlot(site.stub);
                if (freedCount < kNested) freed[freedCount++] = site.stub;
                else nestedOverflow = true;
            }
            Erase(index);
            ++forgotten;
            continue;
        }
        index = (index + 1) & (kTableSize - 1);
        ++visited;
    }
    if (depth < 4) {
        for (std::size_t i = 0; i < freedCount; ++i)
            if (g_liveStubSites != 0) forgotten += ForgetLocked(freed[i], freed[i] + kSlotWords * 4, depth + 1);
        if (nestedOverflow && g_liveStubSites != 0) {
            // More freed stubs than recorded: forget every site inside a free slot (chunks are few, slots small).
            for (std::size_t c = 0; c < g_chunkCount; ++c)
                for (auto slot = g_chunks[c].freeHead; slot != 0;) {
                    forgotten += ForgetLocked(slot, slot + kSlotWords * 4, depth + 1);
                    std::memcpy(&slot, reinterpret_cast<const void*>(g_chunks[c].rw + (slot - g_chunks[c].rx) + 8), sizeof(slot));
                }
        }
    }
    return forgotten;
}

// The lock from ordinary (non-signal) code: block every signal first, or a fault handled on this thread while the
// lock is held would spin on it forever.
struct BlockedSpinLock {
    sigset_t previous{};
    BlockedSpinLock() {
        sigset_t all;
        sigfillset(&all);
        pthread_sigmask(SIG_BLOCK, &all, &previous);
        while (g_lock.test_and_set(std::memory_order_acquire)) {
        }
    }
    ~BlockedSpinLock() {
        g_lock.clear(std::memory_order_release);
        pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    }
};

// Load/store register (register offset): [xN, xM, <extend> #s]. The address it uses is xN + (ext(xM) << s).
bool RegisterOffset(std::uint32_t insn) {
    return (insn & 0x3b200c00u) == 0x38200800u;
}

// add xD, xN, xM, <extend> #s: the register-offset form's address (the load's option field is ADD's extend).
std::uint32_t AddressOf(std::uint32_t insn, unsigned rd) {
    const unsigned size = insn >> 30;
    const bool simd = (insn & 0x04000000u) != 0;
    const unsigned scale = simd && size == 0 && (insn & 0x00800000u) != 0 ? 4u : size; // 128-bit SIMD: Q
    const unsigned amount = (insn & 0x1000u) != 0 ? scale : 0u;
    return 0x8b200000u | (insn & (31u << 16)) | (insn & (7u << 13)) | (amount << 10) | (insn & (31u << 5)) | rd;
}

// Registers an instruction reads (bit per register; 31 is the zero register or SP, never a dead candidate) and the one
// general register it overwrites without reading, for the instruction classes FarReturn looks through. Anything not
// decoded here is not used as a return point.
int OverwrittenWithoutRead(std::uint32_t insn) {
    const unsigned rd = insn & 31u, rn = (insn >> 5) & 31u, rm = (insn >> 16) & 31u, ra = (insn >> 10) & 31u;
    const auto unless = [&](std::initializer_list<unsigned> reads) {
        if (rd == 31) return -1;
        for (const auto read : reads)
            if (read == rd) return -1;
        return static_cast<int>(rd);
    };
    if ((insn & 0x1f000000u) == 0x10000000u) return unless({});                           // ADR, ADRP
    if ((insn & 0x7f800000u) == 0x12800000u || (insn & 0x7f800000u) == 0x52800000u) return unless({}); // MOVN, MOVZ (not MOVK)
    if ((insn & 0x1f800000u) == 0x11000000u) return unless({rn});                         // ADD/SUB immediate
    if ((insn & 0x1f800000u) == 0x12000000u) return unless({rn});                         // logical immediate
    if ((insn & 0x1f800000u) == 0x13000000u && ((insn >> 29) & 3u) != 1u) return unless({rn}); // SBFM, UBFM (not BFM)
    if ((insn & 0x1f000000u) == 0x0a000000u) return unless({rn, rm});                     // logical shifted register
    if ((insn & 0x1f200000u) == 0x0b000000u) return unless({rn, rm});                     // ADD/SUB shifted register
    if ((insn & 0x1fe00000u) == 0x1a800000u) return unless({rn, rm});                     // CSEL, CSINC, CSINV, CSNEG
    if ((insn & 0x1f000000u) == 0x1b000000u) return unless({rn, rm, ra});                 // MADD, MSUB, SMADDL...
    // LDR/LDRB/LDRH/LDRSx into a general register: unsigned offset, unscaled and register offset (no writeback).
    const unsigned size = insn >> 30, opc = (insn >> 22) & 3u;
    const bool load = opc != 0 && !(size == 3 && opc >= 2) && !(size == 2 && opc == 3); // not a store, PRFM or unallocated
    if ((insn & 0x3f000000u) == 0x39000000u && load) return unless({rn});
    if ((insn & 0x3f200c00u) == 0x38000000u && load) return unless({rn});
    if ((insn & 0x3f200c00u) == 0x38200800u && load) return unless({rn, rm});
    if ((insn & 0x3fc00000u) == 0x29400000u || (insn & 0x3fc00000u) == 0x29c00000u) { // LDP/LDPSW signed offset / pre-index
        const unsigned rt2 = (insn >> 10) & 31u;
        if ((insn & 0x00800000u) != 0) return -1; // pre-index writes back Rn
        return rt2 != rd ? unless({rn}) : -1;
    }
    return -1;
}

// Position-independent instructions with no control flow that may run from a stub instead of in place: data
// processing (immediate, except ADR/ADRP; register; SIMD and FP) and loads/stores other than literal loads and
// exclusives. A copied access to displaced memory faults in the stub and is patched there like a JIT site.
bool Relocatable(std::uint32_t insn) {
    if ((insn & 0x1c000000u) == 0x10000000u) return (insn & 0x1f000000u) != 0x10000000u;
    if ((insn & 0x0e000000u) == 0x0a000000u || (insn & 0x0e000000u) == 0x0e000000u) return true;
    if (!IsLoadStore(insn)) return false;
    return (insn & 0x3f000000u) != 0x08000000u; // exclusives and LDAR/STLR family
}

// A trap-free return from a far stub (D-VA7): no free memory lies within branch range of the dyld shared cache, so the
// stub cannot `b` back, and every general register may be live. But a register the native code overwrites without
// reading is dead just before that instruction. Scan forward over up to kFarLookahead instructions the stub can run
// itself; the stub runs them, loads the address of the overwriting instruction into its dead register and branches
// there. `copies` receives the instructions to run; the result is that register, or -1 (BRK return).
constexpr unsigned kFarLookahead = 6;
int FarReturn(std::uintptr_t pc, std::uint32_t* copies, unsigned& count) {
    count = 0;
    const bool immutable = (pc >= g_cacheStart && pc + 4 * (kFarLookahead + 2) <= g_cacheEnd) ||
                           (InTestFarCode(pc) && pc + 4 * (kFarLookahead + 2) <= g_testFarEnd);
    if (!immutable) return -1;
    const auto* code = reinterpret_cast<const std::uint32_t*>(pc + 4);
    for (unsigned i = 0; i <= kFarLookahead; ++i) {
        const auto next = code[i];
        if (const int dead = OverwrittenWithoutRead(next); dead >= 0) return dead;
        if (i == kFarLookahead || !Relocatable(next)) return -1;
        copies[count++] = next;
    }
    return -1;
}

Site* BuildSite(std::uintptr_t pc, std::uint32_t insn) {
    const unsigned rn = (insn >> 5) & 31u;
    bool busy[32] = {};
    for (const unsigned shift : {0u, 5u, 10u, 16u}) busy[(insn >> shift) & 31u] = true;
    unsigned a = 32;
    for (unsigned reg = 9; reg <= 15 && a == 32; ++reg)
        if (!busy[reg]) a = reg;
    std::uint32_t* code = nullptr;
    bool far = false;
    const auto stub = AllocateSlot(pc, code, far);
    if (stub == 0) {
        (void)!write(2, "[displaced] no stub memory\n", 27); // fatal for this access: always said
        return nullptr;
    }
    std::uint32_t copies[kFarLookahead];
    unsigned copyCount = 0;
    const int dead = far ? FarReturn(pc, copies, copyCount) : -1;
    std::uint32_t words[kSlotWords];
    std::uint32_t n = 0;
    const auto push = 0xf8170fe0u | a; // str xA, [sp, #-144]!   (clear of the 128-byte red zone)
    const auto pop = 0xf84907e0u | a;  // ldr xA, [sp], #144
    const auto eorFlip = [](unsigned dst, unsigned src) { return 0xd2400000u | (24u << 16) | (src << 5) | dst; };
    // checked: ordinary memory above the window (the arena, first fit from 512 GiB) leaves after three instructions.
    // A register-offset access checks its full address (xA = xN + ext(xM) << s first).
    const bool indexed = RegisterOffset(insn);
    const unsigned source = indexed ? a : rn;
    words[n++] = push;
    if (indexed) words[n++] = AddressOf(insn, a);
    words[n++] = 0xd364fc00u | (source << 5) | a; // lsr xA, src, #36      (DisplacedEnd = 7 << 36)
    words[n++] = 0xd1001c00u | (a << 5) | a;      // sub xA, xA, #7
    const auto aboveAt = n++;                     // tbz xA, #63, plain   (src >= DisplacedEnd)
    const std::uint64_t page = DisplacedStart() >> 14; // 16 KiB aligned (checked at Enable)
    if (indexed) words[n++] = AddressOf(insn, a);
    words[n++] = 0xd34efc00u | (source << 5) | a;                                          // lsr xA, src, #14
    words[n++] = 0xd1400000u | (static_cast<std::uint32_t>((page >> 12) & 0xfffu) << 10) | (a << 5) | a; // sub xA, xA, #hi, lsl #12
    words[n++] = 0xd1000000u | (static_cast<std::uint32_t>(page & 0xfffu) << 10) | (a << 5) | a;         // sub xA, xA, #lo
    const auto belowAt = n++;                                                              // tbnz xA, #63, plain (src < start)
    const auto flipAt = n;
    const auto ordered = OrderedAccess(insn);
    const auto access = [&](std::uint32_t copy) {
        if (ordered == Ordered::Store) words[n++] = kNop;
        words[n++] = copy;
        if (ordered == Ordered::Load) words[n++] = kNop;
    };
    words[n++] = eorFlip(a, rn);
    access((insn & ~(31u << 5)) | (a << 5));
    if (HasWriteback(insn)) words[n++] = eorFlip(rn, a);
    words[n++] = pop;
    std::uint32_t toTailAt = 0;
    if (far) {
        toTailAt = n++;
    } else {
        words[n] = Branch(stub + n * 4, pc + 4);
        ++n;
    }
    const auto plainAt = n;
    words[n++] = pop;
    access(insn);
    std::uint32_t targetLiteralAt = 0;
    if (far) {
        words[toTailAt] = 0x14000000u | ((n - toTailAt) & 0x03ffffffu);
        if (dead >= 0) {
            for (unsigned i = 0; i < copyCount; ++i) words[n++] = copies[i];
            targetLiteralAt = n++;                                // ldr xR, target
            words[n++] = 0xd61f0000u | (static_cast<unsigned>(dead) << 5); // br xR
        } else {
            words[n++] = kReturnBrk;
            const std::uint64_t target = pc + 4;
            std::memcpy(&words[n], &target, sizeof(target));
            n += 2;
        }
    } else {
        words[n] = Branch(stub + n * 4, pc + 4);
        ++n;
    }
    const auto forcedAt = n;
    words[n++] = push;
    words[n] = 0x14000000u | ((flipAt - n) & 0x03ffffffu);
    ++n;
    if (targetLiteralAt != 0) {
        n = (n + 1) & ~1u; // 8-byte literal (slots are 16-byte aligned)
        const std::uint64_t target = pc + 4 * (copyCount + 1);
        std::memcpy(&words[n], &target, sizeof(target));
        words[targetLiteralAt] = 0x58000000u | ((n - targetLiteralAt) << 5) | static_cast<unsigned>(dead);
        n += 2;
    }
    words[aboveAt] = 0xb6f80000u | (((plainAt - aboveAt) & 0x3fffu) << 5) | a; // tbz xA, #63
    words[belowAt] = 0xb7f80000u | (((plainAt - belowAt) & 0x3fffu) << 5) | a; // tbnz xA, #63
    std::memcpy(code, words, n * 4);
    sys_icache_invalidate(reinterpret_cast<void*>(stub), n * 4);
    auto* site = FindSite(pc, true);
    if (site == nullptr) {
        (void)!write(2, "[displaced] site table full\n", 28); // fatal for this access: always said
        FreeSlot(stub);
        return nullptr;
    }
    if (!site->inStub && ChunkOf(pc) != nullptr) {
        site->inStub = true;
        ++g_liveStubSites;
    }
    site->stub = stub;
    site->forcedOffset = forcedAt * 4;
    site->instruction = insn;
    site->far = far && dead < 0; // a branch-returning far stub needs no SIGTRAP
    if (far && dead >= 0) g_farBranchSites.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<int> farReported{0}; // APS5_DISPLACED_TRACE: the first far sites (no free memory in branch range)
    if (far && g_trace && farReported.fetch_add(1) < 16) {
        char line[160];
        const int length = std::snprintf(line, sizeof(line), "[displaced] far stub for pc 0x%llx insn 0x%08x (%s)\n", static_cast<unsigned long long>(pc), insn,
            dead >= 0 ? "branch return" : "BRK return");
        if (length > 0) (void)!write(2, line, std::min(static_cast<std::size_t>(length), sizeof(line) - 1));
    }
    return site;
}

// A site inside a stub (an instruction a far stub runs for the native code it returns into) is patched through the
// chunk's writable view, like a JIT site. Caller holds the lock.
bool PatchChunk(std::uintptr_t pc, std::uint32_t word) {
    for (std::size_t i = 0; i < g_chunkCount; ++i) {
        const auto& chunk = g_chunks[i];
        if (pc < chunk.rx || pc >= chunk.rx + chunk.bytes) continue;
        __atomic_store_n(reinterpret_cast<std::uint32_t*>(chunk.rw + (pc - chunk.rx)), word, __ATOMIC_RELEASE);
        sys_icache_invalidate(reinterpret_cast<void*>(pc), 4);
        return true;
    }
    return false;
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

// One frame as image+offset (dladdr: not async-signal-safe, acceptable for this opt-in debug aid) or a bare address
// (JIT code). Symbolize the offsets offline with atos -l 0 against the image.
int Frame(char* out, std::size_t size, std::uintptr_t address) {
    Dl_info image{};
    if (dladdr(reinterpret_cast<const void*>(address), &image) != 0 && image.dli_fname != nullptr && image.dli_fbase != nullptr) {
        const char* name = std::strrchr(image.dli_fname, '/');
        if (image.dli_sname != nullptr && image.dli_saddr != nullptr)
            return std::snprintf(out, size, " %s+0x%llx(%s+0x%llx)", name != nullptr ? name + 1 : image.dli_fname,
                static_cast<unsigned long long>(address - reinterpret_cast<std::uintptr_t>(image.dli_fbase)), image.dli_sname,
                static_cast<unsigned long long>(address - reinterpret_cast<std::uintptr_t>(image.dli_saddr)));
        return std::snprintf(out, size, " %s+0x%llx", name != nullptr ? name + 1 : image.dli_fname,
            static_cast<unsigned long long>(address - reinterpret_cast<std::uintptr_t>(image.dli_fbase)));
    }
    return std::snprintf(out, size, " 0x%llx", static_cast<unsigned long long>(address));
}

// Who drives a fault storm: pc, lr and the frame-pointer chain of the faulting thread, walked only inside its own
// stack (JIT code may use x29 for anything).
void Backtrace(std::uint64_t fault, std::uintptr_t pc, std::uint32_t insn, const siginfo_t* info, const __darwin_mcontext64* machine) {
    char line[1536];
    int length = std::snprintf(line, sizeof(line), "[displaced] stack of fault %llu: insn 0x%08x addr 0x%llx pc", static_cast<unsigned long long>(fault), insn,
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)));
    length += Frame(line + length, sizeof(line) - static_cast<std::size_t>(length), pc);
    length += std::snprintf(line + length, sizeof(line) - static_cast<std::size_t>(length), " lr");
    length += Frame(line + length, sizeof(line) - static_cast<std::size_t>(length), static_cast<std::uintptr_t>(arm_thread_state64_get_lr(machine->__ss)));
    const auto self = pthread_self();
    const auto top = reinterpret_cast<std::uintptr_t>(pthread_get_stackaddr_np(self));
    const auto bottom = top - pthread_get_stacksize_np(self);
    auto fp = static_cast<std::uintptr_t>(arm_thread_state64_get_fp(machine->__ss));
    for (int depth = 0; depth < 16 && length > 0 && length < static_cast<int>(sizeof(line)) - 96; ++depth) {
        if (fp < bottom || fp + 16 > top || (fp & 15u) != 0) break;
        const auto* frame = reinterpret_cast<const std::uintptr_t*>(fp);
        length += Frame(line + length, sizeof(line) - static_cast<std::size_t>(length), frame[1]);
        if (frame[0] <= fp) break;
        fp = frame[0];
    }
    if (length > 0) {
        length = std::min(length, static_cast<int>(sizeof(line)) - 2);
        line[length++] = '\n';
        (void)!write(2, line, static_cast<std::size_t>(length));
    }
}

bool Redirect(siginfo_t* info, void* context) {
    if (!IsDisplaced(reinterpret_cast<std::uintptr_t>(info->si_addr))) return false;
    auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
    const std::uintptr_t pc = arm_thread_state64_get_pc(machine->__ss);
    if (pc == 0 || (pc & 3u) != 0) return false;
    auto insn = *reinterpret_cast<const std::uint32_t*>(pc);
    if (!IsLoadStore(insn)) {
        // Two threads faulted on the same access; the first one's handler patched the site to a branch to its stub
        // before the second one's handler read the instruction. Run the recorded access (ASTRO BOT died of this
        // with SIGSEGV on a patched B, run astro3-s7).
        SpinLock lock;
        const auto* site = FindSite(pc, false);
        if (site == nullptr || site->stub == 0 || site->far || insn != Branch(pc, site->stub)) return false;
        insn = site->instruction;
    }
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
    ReportMilestone(fault + 1);
    if (g_trace && fault < 64) {
        char line[160];
        const int length = std::snprintf(line, sizeof(line), "[displaced] fault %llu pc 0x%llx insn 0x%08x addr 0x%llx\n", static_cast<unsigned long long>(fault),
            static_cast<unsigned long long>(pc), insn, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)));
        if (length > 0) (void)!write(2, line, static_cast<std::size_t>(length));
    }
    if (g_stackEvery != 0 && fault % g_stackEvery == 0 && fault / g_stackEvery < 32) Backtrace(fault, pc, insn, info, machine);
    bool patched = false;
    if (!far) {
        if (auto* writer = g_writer.load(std::memory_order_acquire); writer != nullptr && writer(pc, Branch(pc, checked))) {
            patched = true;
        } else {
            SpinLock lock;
            patched = PatchChunk(pc, Branch(pc, checked));
        }
    }
    if (patched) {
        g_patches.fetch_add(1, std::memory_order_relaxed);
    } else {
        const auto unpatched = g_unpatched.fetch_add(1, std::memory_order_relaxed);
        ReportMilestone(unpatched + 1, "unpatched (native) faults", 1); // rare now: every power of two from the first
        if (g_nativeStackEvery != 0 && unpatched % g_nativeStackEvery == 0 && unpatched / g_nativeStackEvery < 32) Backtrace(unpatched, pc, insn, info, machine);
    }
    arm_thread_state64_set_pc_fptr(machine->__ss, reinterpret_cast<void*>(forced));
    return true;
}

// True when [address, address + bytes) is mapped with `protection` (a real fault, not a rewritten site, otherwise).
bool Accessible(std::uintptr_t address, std::size_t bytes, vm_prot_t protection) {
    for (std::uintptr_t at = address; at < address + bytes;) {
        mach_vm_address_t start = at;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info{};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &start, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS) return false;
        if (start > at || (info.protection & protection) != protection) return false;
        at = start + size;
    }
    return true;
}

// SIGBUS (BUS_ADRALN) on an ordered access in a stub that crosses 16 bytes: rewrite it to FEX's half-barrier form
// (see the comment at the top) and resume at the barrier (store) or the access (load). The stub's registers and
// stack are untouched, so resuming mid-stub is safe. A thread that faulted on the old instruction while another one
// rewrote it finds the plain form and runs it again; one whose plain access faults (a real protection fault) is
// chained. Anything else in a stub that faults for alignment (an LSE atomic crossing 16 bytes) is named and chained.
bool FixUnaligned(int signal, siginfo_t* info, void* context) {
    if (signal != SIGBUS || info->si_code != BUS_ADRALN) return false;
    auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
    const std::uintptr_t pc = arm_thread_state64_get_pc(machine->__ss);
    SpinLock lock;
    const Chunk* chunk = nullptr;
    for (std::size_t i = 0; i < g_chunkCount && chunk == nullptr; ++i)
        if (pc >= g_chunks[i].rx && pc < g_chunks[i].rx + g_chunks[i].bytes) chunk = &g_chunks[i];
    if (chunk == nullptr) return false;
    const auto report = [&](const char* what) {
        char line[200];
        const int length = std::snprintf(line, sizeof(line), "[displaced] unaligned access in a displaced-access stub %s: pc 0x%llx insn 0x%08x addr 0x%llx\n", what,
            static_cast<unsigned long long>(pc), *reinterpret_cast<const std::uint32_t*>(pc), static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)));
        if (length > 0) (void)!write(2, line, std::min(static_cast<std::size_t>(length), sizeof(line) - 1));
        return false;
    };
    if ((pc & 3u) != 0 || pc < chunk->rx + 4 || pc + 8 > chunk->rx + chunk->bytes) return report("left unhandled");
    const auto* code = reinterpret_cast<const std::uint32_t*>(pc);
    auto* writable = reinterpret_cast<std::uint32_t*>(chunk->rw + (pc - chunk->rx));
    const auto insn = __atomic_load_n(&code[0], __ATOMIC_ACQUIRE);
    const unsigned rn = (insn >> 5) & 31u;
    if (rn == 31) return report("left unhandled");
    const std::uint64_t base = rn <= 28 ? machine->__ss.__x[rn] : rn == 29 ? machine->__ss.__fp : machine->__ss.__lr;
    const bool rcpc2 = (insn & 0x3f000000u) == 0x19000000u || (insn & 0x3f000000u) == 0x38000000u;
    const auto address = base + static_cast<std::uint64_t>(rcpc2 ? (static_cast<std::int32_t>(insn << 11) >> 23) : 0);
    const std::size_t bytes = std::size_t{1} << (insn >> 30);
    if ((address & 15u) + bytes <= 16) return report("that does not cross 16 bytes");
    const auto kind = OrderedAccess(insn);
    if (kind == Ordered::None) {
        for (const auto rewritten : {Ordered::Load, Ordered::Store}) {
            const auto barrier = rewritten == Ordered::Load ? code[1] : code[-1];
            if (!IsUnorderedOf(insn, rewritten) || barrier != (rewritten == Ordered::Load ? kDmbIshLd : kDmbIsh)) continue;
            if (!Accessible(address, bytes, rewritten == Ordered::Load ? VM_PROT_READ : VM_PROT_READ | VM_PROT_WRITE)) return false;
            if (rewritten == Ordered::Store) arm_thread_state64_set_pc_fptr(machine->__ss, reinterpret_cast<void*>(pc - 4));
            return true;
        }
        return report("left unhandled");
    }
    const int slot = kind == Ordered::Load ? 1 : -1;
    if (code[slot] != kNop) return report("without a barrier slot");
    __atomic_store_n(&writable[slot], kind == Ordered::Load ? kDmbIshLd : kDmbIsh, __ATOMIC_RELEASE);
    __atomic_store_n(&writable[0], Unordered(insn, kind), __ATOMIC_RELEASE);
    sys_icache_invalidate(reinterpret_cast<void*>(pc - 4), 12);
    if (g_unaligned.fetch_add(1, std::memory_order_relaxed) < 64 && g_trace) report("rewritten to a plain access and a barrier");
    if (kind == Ordered::Store) arm_thread_state64_set_pc_fptr(machine->__ss, reinterpret_cast<void*>(pc - 4));
    return true;
}

struct sigaction g_previousTrap {};

// A far stub ends in BRK #0xd15 followed by its 64-bit return address.
bool Return(void* context) {
    auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
    const std::uintptr_t pc = arm_thread_state64_get_pc(machine->__ss);
    bool inside = false;
    for (std::size_t i = 0; i < g_chunkCount && !inside; ++i)
        inside = g_chunks[i].far && pc >= g_chunks[i].rx && pc < g_chunks[i].rx + g_chunks[i].bytes;
    if (!inside || *reinterpret_cast<const std::uint32_t*>(pc) != kReturnBrk) return false;
    std::uint64_t target = 0;
    std::memcpy(&target, reinterpret_cast<const void*>(pc + 4), sizeof(target));
    arm_thread_state64_set_pc_fptr(machine->__ss, reinterpret_cast<void*>(target));
    ReportMilestone(g_farReturns.fetch_add(1, std::memory_order_relaxed) + 1, "far-stub returns (SIGTRAP)");
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
    const bool redirected = Redirect(info, context);
    const bool handled = redirected || FixUnaligned(signal, info, context);
    {
        // Debug aid: APS5_TRACE_FAULT_LOOP=1 counts the faults this handler sees and how it took them.
        static const bool traceLoop = std::getenv("APS5_TRACE_FAULT_LOOP") != nullptr;
        if (traceLoop) {
            static std::atomic<std::uint64_t> total{0};
            const auto count = total.fetch_add(1, std::memory_order_relaxed) + 1;
            if (count % 20000 == 0) {
                auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
                char line[200];
                const int length = std::snprintf(line, sizeof(line), "[displaced] %llu faults; this one: signal %d code %d pc 0x%llx addr 0x%llx esr 0x%llx redirected %d fixed %d\n", static_cast<unsigned long long>(count), signal, info->si_code,
                    static_cast<unsigned long long>(arm_thread_state64_get_pc(machine->__ss)), static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)), static_cast<unsigned long long>(machine->__es.__esr), redirected ? 1 : 0, handled && !redirected ? 1 : 0);
                if (length > 0) (void)!write(2, line, static_cast<std::size_t>(length));
            }
        }
    }
    static std::atomic<int> unhandled{0};
    if (!handled && g_trace && unhandled.fetch_add(1) < 16) {
        auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
        char line[160];
        const std::uintptr_t pc = arm_thread_state64_get_pc(machine->__ss);
        std::uint32_t insn = 0;
        if (pc != 0 && (pc & 3u) == 0 && Accessible(pc, 4, VM_PROT_READ)) insn = *reinterpret_cast<const std::uint32_t*>(pc);
        const int length = std::snprintf(line, sizeof(line), "[displaced] not ours: signal %d pc 0x%llx insn 0x%08x esr 0x%llx addr 0x%llx\n", signal,
            static_cast<unsigned long long>(pc), insn, static_cast<unsigned long long>(machine->__es.__esr),
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)));
        if (length > 0) (void)!write(2, line, static_cast<std::size_t>(length));
        Backtrace(0, pc, insn, info, machine); // who made the access (image+offset(symbol+offset) frames)
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
    g_enabledAt = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
    if (const char* every = std::getenv("APS5_DISPLACED_STACKS")) g_stackEvery = std::strtoull(every, nullptr, 0);
    if (const char* every = std::getenv("APS5_DISPLACED_STACKS_NATIVE")) g_nativeStackEvery = std::strtoull(every, nullptr, 0);
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
    if ((g_start & 0x3fffu) != 0 || (g_start >> 38) != 0) { // the stub compares 16 KiB pages as two 12-bit immediates
        std::fprintf(stderr, "[memory] the displaced window start 0x%llx is not a 16 KiB page below 256 GiB; displacement disabled\n", static_cast<unsigned long long>(g_start));
        return false;
    }
    if (std::uint64_t cache = 0; __shared_region_check_np(&cache) == 0 && cache >= DisplacedLowest && cache < g_start &&
                                 std::strcmp(g_ranges[0].name, "shared region tail") == 0) {
        g_cacheStart = static_cast<std::uintptr_t>(cache);
        g_cacheEnd = g_start;
    }
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

void GuestDisplacedTestFarCode_nid_no_patch(std::uintptr_t start, std::uintptr_t end) {
    BlockedSpinLock lock;
    // Sites recorded for the old or new range describe code a test may since have replaced: forget them.
    g_forgotten.fetch_add(ForgetLocked(g_testFarStart, g_testFarEnd, 0) + ForgetLocked(start, end, 0), std::memory_order_relaxed);
    g_testFarStart = start;
    g_testFarEnd = end;
}

std::size_t GuestDisplacedCodeFreed_nid_no_patch(std::uintptr_t start, std::uintptr_t end) {
    if (!g_enabled.load(std::memory_order_acquire)) return 0;
    BlockedSpinLock lock;
    const auto forgotten = ForgetLocked(start, end, 0);
    g_forgotten.fetch_add(forgotten, std::memory_order_relaxed);
    static std::atomic<int> reported{0}; // APS5_DISPLACED_TRACE: the first frees that released sites
    if (g_trace && forgotten != 0 && reported.fetch_add(1) < 16) {
        char line[160];
        const int length = std::snprintf(line, sizeof(line), "[displaced] code freed 0x%llx-0x%llx: %zu sites forgotten\n", static_cast<unsigned long long>(start),
            static_cast<unsigned long long>(end), forgotten);
        if (length > 0) (void)!write(2, line, std::min(static_cast<std::size_t>(length), sizeof(line) - 1));
    }
    return forgotten;
}

void GuestDisplacedSiteStats_nid_no_patch(DisplacedSiteStats* stats) {
    if (stats == nullptr) return;
    BlockedSpinLock lock;
    *stats = DisplacedSiteStats{};
    stats->liveSites = g_liveSites;
    stats->forgottenSites = g_forgotten.load(std::memory_order_relaxed);
    stats->stubSites = g_liveStubSites;
    stats->chunks = g_chunkCount;
    for (std::size_t i = 0; i < g_chunkCount; ++i) {
        stats->slotsInUse += g_chunks[i].used / (kSlotWords * 4) - g_chunks[i].freeSlots;
        stats->slotsFree += g_chunks[i].freeSlots;
    }
}

void GuestDisplacedTestLimits_nid_no_patch(std::size_t maxSites, std::size_t maxChunks) {
    BlockedSpinLock lock;
    g_siteLimit = maxSites == 0 ? kDefaultSiteLimit : std::min(maxSites, kDefaultSiteLimit);
    g_chunkLimit = maxChunks == 0 ? kMaxChunks : std::min(maxChunks, kMaxChunks);
}

void GuestDisplacedNativeStats_nid_no_patch(std::uint64_t* unpatched, std::uint64_t* trapReturns, std::uint64_t* branchSites) {
    if (unpatched != nullptr) *unpatched = g_unpatched.load(std::memory_order_relaxed);
    if (branchSites != nullptr) *branchSites = g_farBranchSites.load(std::memory_order_relaxed);
    if (trapReturns != nullptr) *trapReturns = g_farReturns.load(std::memory_order_relaxed);
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
void GuestDisplacedTestFarCode_nid_no_patch(std::uintptr_t, std::uintptr_t) {}
std::size_t GuestDisplacedCodeFreed_nid_no_patch(std::uintptr_t, std::uintptr_t) { return 0; }
void GuestDisplacedSiteStats_nid_no_patch(DisplacedSiteStats* stats) {
    if (stats != nullptr) *stats = DisplacedSiteStats{};
}
void GuestDisplacedTestLimits_nid_no_patch(std::size_t, std::size_t) {}
void GuestDisplacedNativeStats_nid_no_patch(std::uint64_t* unpatched, std::uint64_t* trapReturns, std::uint64_t* branchSites) {
    if (unpatched != nullptr) *unpatched = 0;
    if (trapReturns != nullptr) *trapReturns = 0;
    if (branchSites != nullptr) *branchSites = 0;
}
void GuestDisplacedRanges_nid_no_patch(const DisplacedRange** ranges, std::size_t* count) {
    if (ranges != nullptr) *ranges = nullptr;
    if (count != nullptr) *count = 0;
}
}
#endif
