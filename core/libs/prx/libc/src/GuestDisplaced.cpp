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

// D31. Every 64-bit macOS arm64 process starts with a permanent, inaccessible VM_MEMORY_GUARD entry over
// [MACH_VM_MIN_GPU_CARVEOUT_ADDRESS, MACH_VM_MAX_GPU_CARVEOUT_ADDRESS) = [64 GiB, 448 GiB) (xnu vm_map_exec,
// vm_reserved_regions "GPU Carveout"): it cannot be deallocated, overwritten or reprotected, and no entitlement
// removes it. Guest memory there is backed at guest | DisplacedFlip, reserved here up front.
//
// A guest (or native) load/store to a displaced address faults with si_addr in the window. The handler builds a
// stub for that instruction that runs it with its base register flipped into the alias:
//
//   checked: sub sp, sp, #144          ; keep clear of the 128-byte red zone
//            stp xA, xB, [sp]
//            lsr xA, xN, #39           ; >= 512 GiB: not displaced
//            cbnz xA, plain
//            lsr xA, xN, #36           ; 64 GiB units: 1..6 are displaced
//            movz xB, #0x7e
//            lsrv xB, xB, xA
//            tbz xB, #0, plain
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
//
// The signal path resumes at `forced` (the base need not be the displaced operand of a register-offset form); a JIT
// site is then patched to `b checked`, so later executions cost ~12 instructions and no signal. Nothing in the stub
// touches NZCV. xA/xB are chosen outside every register field of the instruction. Stubs live in an RX view of
// memory written through an RW view (no MAP_JIT toggling), within branch range of the faulting instruction.
namespace GuestArena {
namespace {

constexpr std::uint32_t kSlotWords = 32;
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
    const auto lsr = [](unsigned dst, unsigned src, unsigned amount) { return 0xd340fc00u | (amount << 16) | (src << 5) | dst; };
    words[n++] = subSp;
    words[n++] = stp;
    words[n++] = lsr(a, rn, 39);
    const auto cbnzAt = n++;
    words[n++] = lsr(a, rn, 36);
    words[n++] = 0xd2800000u | (0x7eu << 5) | b;
    words[n++] = 0x9ac02400u | (a << 16) | (b << 5) | b;
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
    words[cbnzAt] = 0xb5000000u | (((plainAt - cbnzAt) & 0x7ffffu) << 5) | a;
    words[tbzAt] = 0x36000000u | (((plainAt - tbzAt) & 0x3fffu) << 5) | b;
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

bool Enable() {
    g_trace = std::getenv("APS5_DISPLACED_TRACE") != nullptr;
    // Only needed where the window is really a host guard: probe it.
    mach_vm_address_t probe = DisplacedStart;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &probe, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || probe > DisplacedStart)
        return false;
    mach_vm_address_t alias = DisplacedStart | DisplacedFlip;
    if (mach_vm_map(mach_task_self(), &alias, DisplacedEnd - DisplacedStart, 0, VM_FLAGS_FIXED, MEMORY_OBJECT_NULL, 0, FALSE, VM_PROT_NONE, VM_PROT_ALL, VM_INHERIT_DEFAULT) != KERN_SUCCESS) {
        std::fprintf(stderr, "[memory] the displaced guest window's alias 0x%llx+0x%llx is occupied; fixed guest mappings in 0x%llx-0x%llx will fail\n",
            static_cast<unsigned long long>(DisplacedStart | DisplacedFlip), static_cast<unsigned long long>(DisplacedEnd - DisplacedStart),
            static_cast<unsigned long long>(DisplacedStart), static_cast<unsigned long long>(DisplacedEnd));
        return false;
    }
    struct sigaction action {};
    action.sa_sigaction = OnFault;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    sigaction(SIGBUS, &action, &g_previousBus);
    sigaction(SIGSEGV, &action, &g_previousSegv);
    action.sa_sigaction = OnTrap;
    sigaction(SIGTRAP, &action, &g_previousTrap);
    if (g_trace) std::fprintf(stderr, "[displaced] alias 0x%llx reserved, fault handler installed (previous SIGSEGV %p)\n",
        static_cast<unsigned long long>(DisplacedStart | DisplacedFlip), reinterpret_cast<void*>(g_previousSegv.sa_sigaction));
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

void GuestDisplacedSetCodeWriter_nid_no_patch(bool (*writer)(std::uintptr_t, std::uint32_t), std::uintptr_t (*allocator)(std::uintptr_t, std::size_t, std::uintptr_t*)) {
    g_allocator.store(allocator, std::memory_order_release);
    g_writer.store(writer, std::memory_order_release);
}

void GuestDisplacedStats_nid_no_patch(std::uint64_t* faults, std::uint64_t* patches) {
    if (faults != nullptr) *faults = g_faults.load(std::memory_order_relaxed);
    if (patches != nullptr) *patches = g_patches.load(std::memory_order_relaxed);
}

}
#else
namespace GuestArena {
bool GuestDisplacedEnable_nid_no_patch() { return false; }
bool GuestDisplacedActive_nid_no_patch() { return false; }
void GuestDisplacedSetCodeWriter_nid_no_patch(bool (*)(std::uintptr_t, std::uint32_t), std::uintptr_t (*)(std::uintptr_t, std::size_t, std::uintptr_t*)) {}
void GuestDisplacedStats_nid_no_patch(std::uint64_t* faults, std::uint64_t* patches) {
    if (faults != nullptr) *faults = 0;
    if (patches != nullptr) *patches = 0;
}
}
#endif
