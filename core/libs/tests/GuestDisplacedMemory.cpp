// Fixed guest mappings in host-reserved ranges on macOS arm64 (D31, D33): the dyld shared region's unused tail, the
// commpage guard and the GPU carveout (64-448 GiB, a permanent guard in every process) are backed at
// guest | DisplacedFlip, and native accesses to the guest address are redirected by the fault handler. Translated
// (FEX) accesses are covered by the `displaced` fixture.
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__APPLE__)
#include <libkern/OSCacheControl.h>
#include <mach/mach.h>
#include <pthread.h>
#include <csignal>
#include <sys/ucontext.h>
#include <sys/mman.h>
#include <mach/mach_vm.h>
#endif

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
int APS5_VABI sceKernelMprotect(const void*, std::size_t, int);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
// Guest pthread objects are pointer-sized slots in guest memory (void** here: the slot's type does not matter).
int APS5_VABI scePthreadMutexInit(void** mutex, const void* attr, const char* name);
int APS5_VABI scePthreadMutexDestroy(void** mutex);
int APS5_VABI scePthreadMutexLock(void** mutex);
int APS5_VABI scePthreadMutexUnlock(void** mutex);
int APS5_VABI scePthreadMutexTrylock(void** mutex);
int APS5_VABI scePthreadCondInit(void** cond, const void* attr, const char* name);
int APS5_VABI scePthreadCondSignal(void** cond);
int APS5_VABI scePthreadCondDestroy(void** cond);
}

namespace {

constexpr std::uintptr_t Base = 0x4000000000ull;     // GPU carveout (oops-sdk AGC_VM_BASE)
constexpr std::uintptr_t TailBase = 0x400000000ull;  // shared region tail (oops-mesa's GPU VA high range)
constexpr std::size_t Bytes = 0x400000;
constexpr int CpuGpuReadWrite = 0x33;
constexpr int CpuRead = 0x01;
constexpr int MapFixed = 0x10;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
T Load(std::uintptr_t address) {
    return *reinterpret_cast<const volatile T*>(address);
}

template <typename T>
void Store(std::uintptr_t address, T value) {
    *reinterpret_cast<volatile T*>(address) = value;
}

std::uint64_t Faults() {
    std::uint64_t faults = 0;
    GuestArena::GuestDisplacedStats_nid_no_patch(&faults, nullptr);
    return faults;
}

#if defined(__APPLE__)
int HostProtection(std::uintptr_t address) {
    mach_vm_address_t region = address;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || region > address) return -1;
    return info.protection;
}
#endif

}

#if defined(__APPLE__) && defined(__aarch64__)
// More distinct faulting sites than the handler's first site table held (32768): ASTRO BOT patched 32k JIT sites on
// displaced memory in its first minute, and once the table filled every new site was left unhandled (a crash).
// Generated code with one load per site, run once on a displaced address.
void ManySites(std::uintptr_t displaced) {
    constexpr std::size_t Sites = 40000;
    const std::size_t bytes = ((Sites + 2) * 4 + 0x3fff) & ~std::size_t{0x3fff};
    void* code = mmap(nullptr, bytes, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    Require(code != MAP_FAILED, "MAP_JIT for the many-sites code");
    pthread_jit_write_protect_np(0);
    auto* words = static_cast<std::uint32_t*>(code);
    for (std::size_t i = 0; i < Sites; ++i) words[i] = 0xb9400009u; // ldr w9, [x0]
    words[Sites] = 0x2a0903e0u;                                      // mov w0, w9
    words[Sites + 1] = 0xd65f03c0u;                                  // ret
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(code, bytes);
    const auto before = Faults();
    const auto value = reinterpret_cast<std::uint32_t (*)(std::uintptr_t)>(code)(displaced);
    Require(value == Load<std::uint32_t>(GuestArena::HostAddress(displaced)), "the many-sites loads read the wrong value");
    Require(Faults() - before >= Sites, "the many-sites loads did not each go through the fault handler");
    Require(GuestArena::GuestDisplacedCodeFreed_nid_no_patch(reinterpret_cast<std::uintptr_t>(code), reinterpret_cast<std::uintptr_t>(code) + bytes) == Sites,
        "freeing the many-sites code did not forget its sites");
    munmap(code, bytes);
}

// A second thread that faulted on a site before the first thread's handler patched it to a branch to its stub
// reaches the handler with the branch at its pc. The handler must run the recorded access, not chain the fault
// (ASTRO BOT died of SIGSEGV on such a branch). The second thread is simulated by calling the installed handler
// with a context whose pc is the patched site.
bool WriteCode(std::uintptr_t at, std::uint32_t word) {
    pthread_jit_write_protect_np(0);
    __atomic_store_n(reinterpret_cast<std::uint32_t*>(at), word, __ATOMIC_RELEASE);
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(reinterpret_cast<void*>(at), 4);
    return true;
}

void PatchedSiteRace(std::uintptr_t displaced) {
    void* code = mmap(nullptr, 0x4000, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    Require(code != MAP_FAILED, "MAP_JIT for the patched-site code");
    pthread_jit_write_protect_np(0);
    auto* words = static_cast<std::uint32_t*>(code);
    words[0] = 0xb9400009u; // ldr w9, [x0]
    words[1] = 0x2a0903e0u; // mov w0, w9
    words[2] = 0xd65f03c0u; // ret
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(code, 12);
    GuestArena::GuestDisplacedSetCodeWriter_nid_no_patch(&WriteCode, nullptr);
    const auto value = reinterpret_cast<std::uint32_t (*)(std::uintptr_t)>(code)(displaced);
    GuestArena::GuestDisplacedSetCodeWriter_nid_no_patch(nullptr, nullptr);
    Require(value == Load<std::uint32_t>(GuestArena::HostAddress(displaced)), "the patched-site load read the wrong value");
    const std::uint32_t patched = __atomic_load_n(&words[0], __ATOMIC_ACQUIRE);
    Require((patched & 0xfc000000u) == 0x14000000u, "the faulting site was not patched to a branch");

    struct sigaction installed {};
    Require(sigaction(SIGSEGV, nullptr, &installed) == 0 && (installed.sa_flags & SA_SIGINFO) != 0 && installed.sa_sigaction != nullptr,
        "no SIGSEGV handler is installed");
    __darwin_mcontext64 machine{};
    for (unsigned i = 0; i < 29; ++i) machine.__ss.__x[i] = 0;
    machine.__ss.__x[0] = displaced;
    arm_thread_state64_set_pc_fptr(machine.__ss, code);
    ucontext_t context{};
    context.uc_mcontext = &machine;
    siginfo_t info{};
    info.si_signo = SIGSEGV;
    info.si_addr = reinterpret_cast<void*>(displaced);
    installed.sa_sigaction(SIGSEGV, &info, &context);
    struct sigaction after {};
    sigaction(SIGSEGV, nullptr, &after);
    const bool chained = after.sa_sigaction != installed.sa_sigaction;
    if (chained) sigaction(SIGSEGV, &installed, nullptr); // the chain ended at SIG_DFL
    const auto resumed = static_cast<std::uintptr_t>(arm_thread_state64_get_pc(machine.__ss));
    Require(!chained && resumed != reinterpret_cast<std::uintptr_t>(code),
        "a fault at a site another thread already patched was not redirected to its stub");
    GuestArena::GuestDisplacedCodeFreed_nid_no_patch(reinterpret_cast<std::uintptr_t>(code), reinterpret_cast<std::uintptr_t>(code) + 0x4000);
    munmap(code, 0x4000);
}

// Generated code in a MAP_JIT page (not hooked to a code writer, so never patched in place).
struct GeneratedCode {
    void* code = nullptr;
    explicit GeneratedCode(std::initializer_list<std::uint32_t> words) {
        code = mmap(nullptr, 0x4000, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
        Require(code != MAP_FAILED, "MAP_JIT for generated code");
        pthread_jit_write_protect_np(0);
        std::uint32_t* out = static_cast<std::uint32_t*>(code);
        for (const auto word : words) *out++ = word;
        for (int i = 0; i < 16; ++i) *out++ = 0xd4200000u; // brk #0 after the end: the look-ahead stops in mapped code
        pthread_jit_write_protect_np(1);
        sys_icache_invalidate(code, 0x4000);
    }
    ~GeneratedCode() {
        GuestArena::GuestDisplacedCodeFreed_nid_no_patch(Begin(), Begin() + 0x4000);
        munmap(code, 0x4000);
    }
    std::uintptr_t Begin() const { return reinterpret_cast<std::uintptr_t>(code); }
    template <typename F>
    F As() const { return reinterpret_cast<F>(code); }
};

struct NativeCounts {
    std::uint64_t faults = 0, patches = 0, unpatched = 0, traps = 0, branchSites = 0;
    static NativeCounts Now() {
        NativeCounts counts;
        GuestArena::GuestDisplacedStats_nid_no_patch(&counts.faults, &counts.patches);
        GuestArena::GuestDisplacedNativeStats_nid_no_patch(&counts.unpatched, &counts.traps, &counts.branchSites);
        return counts;
    }
};

// Far stubs (native code out of branch range of free memory, the dyld shared cache) return without SIGTRAP when the
// code ahead overwrites a register without reading it; otherwise through BRK. The generated code is declared far
// and immutable through the test hook, like the shared cache.
void FarStubReturns(std::uintptr_t source, std::uintptr_t destination) {
    Store<std::uint64_t>(GuestArena::HostAddress(source), 0x1000);
    {
        // ldr x9, [x0]; add x10, x9, #1 (x10 dead: the return point); mov x0, x10; ret
        GeneratedCode code({0xf9400009u, 0x9100052au, 0xaa0a03e0u, 0xd65f03c0u});
        GuestArena::GuestDisplacedTestFarCode_nid_no_patch(code.Begin(), code.Begin() + 0x4000);
        const auto before = NativeCounts::Now();
        for (int i = 0; i < 100; ++i) Require(code.As<std::uint64_t (*)(std::uintptr_t)>()(source) == 0x1001, "far stub, branch return: wrong value");
        const auto after = NativeCounts::Now();
        Require(after.branchSites == before.branchSites + 1, "a far stub with a dead register ahead did not get a branch return");
        Require(after.traps == before.traps, "a branch-returning far stub took SIGTRAP");
        Require(after.unpatched - before.unpatched == 100, "every execution of an unpatched native site is one fault");
    }
    {
        // ldr x9, [x0]; str x9, [x1] (displaced too: runs in the stub, faults there once and is patched there);
        // add x9, x9, #3; movz x11, #7 (x11 dead); add x0, x9, x11; ret
        GeneratedCode code({0xf9400009u, 0xf9000029u, 0x91000d29u, 0xd28000ebu, 0x8b0b0120u, 0xd65f03c0u});
        GuestArena::GuestDisplacedTestFarCode_nid_no_patch(code.Begin(), code.Begin() + 0x4000);
        const auto before = NativeCounts::Now();
        for (int i = 0; i < 100; ++i) {
            Store<std::uint64_t>(GuestArena::HostAddress(destination), 0);
            Require(code.As<std::uint64_t (*)(std::uintptr_t, std::uintptr_t)>()(source, destination) == 0x1000 + 3 + 7, "far stub with copied instructions: wrong value");
            Require(Load<std::uint64_t>(GuestArena::HostAddress(destination)) == 0x1000, "far stub with copied instructions: the copied store was lost");
        }
        const auto after = NativeCounts::Now();
        Require(after.branchSites == before.branchSites + 1 && after.traps == before.traps, "far stub with copied instructions did not return by branch");
        std::printf("far stub with copies: faults %llu patches %llu unpatched %llu traps %llu branch sites %llu\n",
            static_cast<unsigned long long>(after.faults - before.faults), static_cast<unsigned long long>(after.patches - before.patches),
            static_cast<unsigned long long>(after.unpatched - before.unpatched), static_cast<unsigned long long>(after.traps - before.traps),
            static_cast<unsigned long long>(after.branchSites - before.branchSites));
        Require(after.patches == before.patches + 1, "the copied displaced store was not patched inside the stub");
        Require(after.faults - before.faults == 101, "expected one fault per call plus one for the copied store");
    }
    {
        // ldr x0, [x0]; ret: nothing ahead to return through, BRK return.
        GeneratedCode code({0xf9400000u, 0xd65f03c0u});
        GuestArena::GuestDisplacedTestFarCode_nid_no_patch(code.Begin(), code.Begin() + 0x4000);
        const auto before = NativeCounts::Now();
        for (int i = 0; i < 10; ++i) Require(code.As<std::uint64_t (*)(std::uintptr_t)>()(source) == 0x1000, "far stub, BRK return: wrong value");
        const auto after = NativeCounts::Now();
        Require(after.branchSites == before.branchSites && after.traps - before.traps == 10, "a far stub with no return register did not return through BRK");
    }
    GuestArena::GuestDisplacedTestFarCode_nid_no_patch(0, 0);
}

// A patched site's stub on every kind of pointer: above the window (malloc's regions, the arena), below it (the
// stack, below the shared cache) and displaced; none of them faults once the site is patched.
void PatchedSitePointers(std::uintptr_t displaced) {
    // ldr x9, [x0]; add x9, x9, #1; str x9, [x0]; mov x0, x9; ret
    GeneratedCode code({0xf9400009u, 0x91000529u, 0xf9000009u, 0xaa0903e0u, 0xd65f03c0u});
    GuestArena::GuestDisplacedSetCodeWriter_nid_no_patch(&WriteCode, nullptr);
    const auto run = code.As<std::uint64_t (*)(std::uintptr_t)>();
    Store<std::uint64_t>(GuestArena::HostAddress(displaced), 41);
    Require(run(displaced) == 42, "patched sites: the first (faulting) call");
    GuestArena::GuestDisplacedSetCodeWriter_nid_no_patch(nullptr, nullptr);
    Require((*static_cast<const std::uint32_t*>(code.code) & 0xfc000000u) == 0x14000000u && (static_cast<const std::uint32_t*>(code.code)[2] & 0xfc000000u) == 0x14000000u,
        "the load and store sites were not both patched");
    auto* heap = new std::uint64_t(100);
    std::uint64_t stack = 200;
    const auto before = NativeCounts::Now();
    for (int i = 0; i < 1000; ++i) {
        run(reinterpret_cast<std::uintptr_t>(heap));
        run(reinterpret_cast<std::uintptr_t>(&stack));
        run(displaced);
    }
    const auto after = NativeCounts::Now();
    Require(*heap == 1100 && stack == 1200 && Load<std::uint64_t>(GuestArena::HostAddress(displaced)) == 1042, "patched sites: wrong values");
    Require(after.faults == before.faults, "a patched site faulted again");
    Require(reinterpret_cast<std::uintptr_t>(heap) >= GuestArena::DisplacedEnd || reinterpret_cast<std::uintptr_t>(heap) < GuestArena::DisplacedStart(), "the heap is displaced");
    delete heap;
}

GuestArena::DisplacedSiteStats SiteStats() {
    GuestArena::DisplacedSiteStats stats;
    GuestArena::GuestDisplacedSiteStats_nid_no_patch(&stats);
    return stats;
}

// Register-offset accesses ([xN, xM, ext #s]) check the address they really use. Before, the stub checked xN alone:
// `[small + displaced]` took the plain path, faulted inside the stub, and every execution built one more stub (a
// chain one site longer per call); a displaced base with an index leaving the window was flipped wrongly.
void RegisterOffsetSites(std::uintptr_t displaced) {
    Store<std::uint64_t>(GuestArena::HostAddress(displaced), 0x1234);
    Store<std::uint32_t>(GuestArena::HostAddress(displaced) + 12, 0x5678);
    auto* heap = new std::uint64_t[4]{7, 8, 9, 10};
    GuestArena::GuestDisplacedSetCodeWriter_nid_no_patch(&WriteCode, nullptr);
    {
        // ldr x9, [x1, x0]; mov x0, x9; ret   (x1 small, x0 the pointer: the base is not the displaced operand)
        GeneratedCode code({0xf8606829u, 0xaa0903e0u, 0xd65f03c0u});
        const auto run = code.As<std::uint64_t (*)(std::uintptr_t, std::uintptr_t)>();
        const auto before = SiteStats();
        const auto faults = Faults();
        for (int i = 0; i < 200; ++i) {
            Require(run(displaced - 8, 8) == 0x1234, "[small + displaced]: wrong value");
            Require(run(reinterpret_cast<std::uintptr_t>(heap) - 8, 8) == 7, "[small + heap] through the patched site: wrong value");
        }
        const auto after = SiteStats();
        std::printf("register offset [small + displaced]: faults %llu, live sites +%lld, sites in stubs %llu\n", static_cast<unsigned long long>(Faults() - faults),
            static_cast<long long>(after.liveSites - before.liveSites), static_cast<unsigned long long>(after.stubSites));
        Require(Faults() - faults == 1, "[small + displaced] faulted more than once: the stub does not check the full address");
        Require(after.liveSites == before.liveSites + 1 && after.stubSites == 0, "[small + displaced] built sites inside its stub");
    }
    {
        // ldr w9, [x0, w1, sxtw #2]; mov x0, x9; ret   (displaced base, scaled signed index)
        GeneratedCode code({0xb861d809u, 0xaa0903e0u, 0xd65f03c0u});
        const auto run = code.As<std::uint64_t (*)(std::uintptr_t, std::int64_t)>();
        const auto faults = Faults();
        for (int i = 0; i < 200; ++i) {
            Require(run(displaced, 3) == 0x5678, "[displaced + w1 sxtw #2]: wrong value");
            Require(run(reinterpret_cast<std::uintptr_t>(heap) + 16, -2) == 8, "[heap + negative index] through the patched site: wrong value");
        }
        Require(Faults() - faults == 1, "[displaced + scaled index] faulted more than once");
    }
    GuestArena::GuestDisplacedSetCodeWriter_nid_no_patch(nullptr, nullptr);
    delete[] heap;
}

// One churner: a JIT-like code region rewritten every round with `sites` loads, run on a displaced pointer (one fault
// and one patch per site), run again (patched: no fault) and on an ordinary pointer, then freed the way the CPU backend
// frees a code buffer. Without reclamation the live sites and the stub slots grow by `sites` every round.
void Churn(std::uintptr_t displaced, std::size_t sites, std::size_t rounds, const char* name, std::size_t* patched) {
    const std::size_t bytes = ((sites + 2) * 4 + 0x3fff) & ~std::size_t{0x3fff};
    void* code = mmap(nullptr, bytes, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    Require(code != MAP_FAILED, "MAP_JIT for the churn code");
    const auto begin = reinterpret_cast<std::uintptr_t>(code);
    auto* words = static_cast<std::uint32_t*>(code);
    std::uint32_t ordinary = 0;
    for (std::size_t round = 0; round < rounds; ++round) {
        const unsigned reg = 9 + static_cast<unsigned>(round % 3); // a different instruction at every pc each round
        pthread_jit_write_protect_np(0);
        for (std::size_t i = 0; i < sites; ++i) words[i] = 0xb9400000u | reg;   // ldr wR, [x0]
        words[sites] = 0x2a0003e0u | (reg << 16);                               // mov w0, wR
        words[sites + 1] = 0xd65f03c0u;                                         // ret
        pthread_jit_write_protect_np(1);
        sys_icache_invalidate(code, bytes);
        const auto run = reinterpret_cast<std::uint32_t (*)(std::uintptr_t)>(code);
        const auto expected = Load<std::uint32_t>(GuestArena::HostAddress(displaced));
        if (run(displaced) != expected) throw std::runtime_error(std::string(name) + ": the faulting run read the wrong value");
        if (run(displaced) != expected) throw std::runtime_error(std::string(name) + ": the patched run read the wrong value");
        ordinary = static_cast<std::uint32_t>(round);
        if (run(reinterpret_cast<std::uintptr_t>(&ordinary)) != ordinary) throw std::runtime_error(std::string(name) + ": the patched run on ordinary memory read the wrong value");
        if ((__atomic_load_n(&words[0], __ATOMIC_ACQUIRE) & 0xfc000000u) != 0x14000000u) throw std::runtime_error(std::string(name) + ": the first site was not patched");
        const auto forgotten = GuestArena::GuestDisplacedCodeFreed_nid_no_patch(begin, begin + bytes);
        if (forgotten != sites) throw std::runtime_error(std::string(name) + ": freeing the code forgot " + std::to_string(forgotten) + " sites, expected " + std::to_string(sites));
        *patched += sites;
    }
    munmap(code, bytes);
}

// Sites and stubs are reclaimed when code is freed (cache flush, SMC, unload): churn far past the caps, on two
// threads at once (faults and patches on one thread while the other frees its code), first under small test caps
// (the sites and the stub chunks alive before plus 2048 sites and 2 chunks, 7281 stubs each), then past the real
// table (1M entries, 7/8 of them usable) at the built-in caps.
void SiteReclamation(std::uintptr_t displaced) {
    Store<std::uint32_t>(GuestArena::HostAddress(displaced), 0xc0ffee);
    GuestArena::GuestDisplacedSetCodeWriter_nid_no_patch(&WriteCode, nullptr);
    const auto before = SiteStats();
    const auto faults = Faults();
    std::printf("site reclamation: before live %llu, slots in use %llu, free %llu, chunks %llu\n", static_cast<unsigned long long>(before.liveSites),
        static_cast<unsigned long long>(before.slotsInUse), static_cast<unsigned long long>(before.slotsFree), static_cast<unsigned long long>(before.chunks));
    const auto churnPair = [&](std::size_t sites, std::size_t rounds) {
        std::size_t first = 0, second = 0;
        std::string error;
        std::thread other([&] {
            try {
                Churn(displaced, sites, rounds, "churn thread 2", &second);
            } catch (const std::exception& e) {
                error = e.what();
            }
        });
        Churn(displaced, sites, rounds, "churn thread 1", &first);
        other.join();
        if (!error.empty()) throw std::runtime_error(error);
        return first + second;
    };
    GuestArena::GuestDisplacedTestLimits_nid_no_patch(before.liveSites + 2048, before.chunks + 2);
    const auto small = churnPair(1000, 100); // 200k sites patched through a 2048-site table and 14.5k stub slots
    GuestArena::GuestDisplacedTestLimits_nid_no_patch(0, 0);
    const auto mid = SiteStats();
    Require(mid.liveSites == before.liveSites && mid.slotsInUse == before.slotsInUse && mid.stubSites == before.stubSites, "sites or stubs leaked under the small caps");
    Require(mid.chunks <= before.chunks + 2, "the small-cap churn reserved more stub chunks than its cap");
    const auto large = churnPair(4000, 160); // 1.28M sites: more than the whole table
    const auto after = SiteStats();
    std::printf("site reclamation: %zu + %zu sites patched and forgotten, %llu faults; after live %llu, slots in use %llu, free %llu, chunks %llu, forgotten %llu\n", small, large,
        static_cast<unsigned long long>(Faults() - faults), static_cast<unsigned long long>(after.liveSites), static_cast<unsigned long long>(after.slotsInUse),
        static_cast<unsigned long long>(after.slotsFree), static_cast<unsigned long long>(after.chunks), static_cast<unsigned long long>(after.forgottenSites));
    Require(small + large > (std::size_t{1} << 20), "the churn did not exceed the table size");
    Require(after.liveSites == before.liveSites && after.slotsInUse == before.slotsInUse, "sites or stubs leaked under the built-in caps");
    Require(after.chunks <= mid.chunks + 2, "the churn kept reserving stub chunks");
    GuestArena::GuestDisplacedSetCodeWriter_nid_no_patch(nullptr, nullptr);
}
#endif

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("guest displaced memory: no displaced window on this host, skipped");
    return 0;
#else
    try {
        Require(GuestArena::GuestArenaAvailable_nid_postfix(), "the guest arena is unavailable");
        Require(GuestArena::GuestDisplacedActive_nid_no_patch(), "the displaced window is not active on macOS arm64");
        Require(GuestArena::HostAddress(Base) == (Base | GuestArena::DisplacedFlip) && GuestArena::GuestAddress(Base | GuestArena::DisplacedFlip) == Base,
            "guest and host addresses do not round-trip");
        Require(GuestArena::HostAddress(0x100000000ull) == 0x100000000ull, "an address below the window was translated");
        Require(GuestArena::HostAddress(0x8000000000ull) == 0x8000000000ull, "an address above the window was translated");

        // The range table: shared region tail, commpage guard, GPU carveout, contiguous, one flip.
        const GuestArena::DisplacedRange* ranges = nullptr;
        std::size_t rangeCount = 0;
        GuestArena::GuestDisplacedRanges_nid_no_patch(&ranges, &rangeCount);
        for (std::size_t i = 0; i < rangeCount; ++i)
            std::printf("displaced range %s 0x%llx-0x%llx at 0x%llx\n", ranges[i].name, static_cast<unsigned long long>(ranges[i].guestStart),
                static_cast<unsigned long long>(ranges[i].guestEnd), static_cast<unsigned long long>(ranges[i].hostBase));
        Require(rangeCount == 3, "expected three displaced ranges (shared region tail, commpage guard, GPU carveout)");
        Require(ranges[0].guestStart == GuestArena::DisplacedStart() && ranges[2].guestEnd == GuestArena::DisplacedEnd, "the table does not span the window");
        for (std::size_t i = 0; i < rangeCount; ++i) {
            Require(ranges[i].hostBase == (ranges[i].guestStart | GuestArena::DisplacedFlip), "a range's alias is not its guest address with the flip");
            Require(i == 0 || ranges[i].guestStart == ranges[i - 1].guestEnd, "the displaced ranges are not contiguous");
        }
        Require(ranges[1].guestStart == 0xFC0000000ull && ranges[2].guestStart == 0x1000000000ull, "unexpected commpage guard or carveout bounds");
        Require(GuestArena::DisplacedStart() <= TailBase && GuestArena::DisplacedStart() > 0x180000000ull, "0x400000000 is not in the shared region tail");
        Require(GuestArena::HostAddress(TailBase) == (TailBase | GuestArena::DisplacedFlip), "the shared region tail is not translated");
        Require(GuestArena::HostAddress(GuestArena::DisplacedStart() - 0x4000) == GuestArena::DisplacedStart() - 0x4000, "the shared cache was translated");

        // First-fit placement never lands in a displaced range.
        for (int i = 0; i < 8; ++i) {
            void* placed = GuestArena::GuestArenaAllocate_nid_postfix(0x10000000, 0x200000);
            const auto at = reinterpret_cast<std::uintptr_t>(placed);
            Require(at >= GuestArena::DisplacedEnd || at + 0x10000000 <= GuestArena::DisplacedLowest, "first fit placed an allocation in a displaced range");
            GuestArena::GuestArenaRelease_nid_postfix(placed, 0x10000000);
        }

        std::int64_t physical = 0;
        Require(sceKernelAllocateDirectMemory(0, 0x100000000ll, Bytes, 0x200000, 3, &physical) == 0, "allocate direct memory");
        void* fixed = reinterpret_cast<void*>(Base);
        Require(sceKernelMapDirectMemory(&fixed, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && fixed == reinterpret_cast<void*>(Base),
            "the fixed mapping at 0x4000000000 failed");
        void* view = nullptr;
        Require(sceKernelMapDirectMemory(&view, Bytes, CpuGpuReadWrite, 0, physical, 0x200000) == 0, "the second view failed");
        const auto viewAddress = reinterpret_cast<std::uintptr_t>(view);
        Require(!GuestArena::OverlapsDisplaced(viewAddress, Bytes), "first fit placed a mapping in the displaced window");

        // The alias holds the pages; native code reaches them through the guest address via the fault handler.
        const auto alias = GuestArena::HostAddress(Base);
        Store<std::uint64_t>(alias + 8, 0x1122334455667788ull);
        const auto before = Faults();
        Require(Load<std::uint64_t>(Base + 8) == 0x1122334455667788ull, "a native load from the guest address missed the alias");
        Store<std::uint32_t>(Base + 0x104, 0xa5a5a5a5u);
        Require(Load<std::uint32_t>(alias + 0x104) == 0xa5a5a5a5u, "a native store to the guest address missed the alias");
        Require(Load<std::uint32_t>(viewAddress + 0x104) == 0xa5a5a5a5u, "the second view does not share the pages");
        Require(Faults() >= before + 2, "the accesses did not go through the fault handler");

        // Library code (SIMD pairs, post-index addressing) on the guest address.
        char pattern[4096];
        for (std::size_t i = 0; i < sizeof(pattern); ++i) pattern[i] = static_cast<char>(i * 7 + 1);
        std::memcpy(reinterpret_cast<void*>(Base + 0x10000), pattern, sizeof(pattern));
        Require(std::memcmp(reinterpret_cast<const void*>(viewAddress + 0x10000), pattern, sizeof(pattern)) == 0, "memcpy to the guest address");
        char back[4096];
        std::memcpy(back, reinterpret_cast<const void*>(Base + 0x10000), sizeof(back));
        Require(std::memcmp(back, pattern, sizeof(back)) == 0, "memcpy from the guest address");

        // Protection is applied to the alias.
        Require(sceKernelMprotect(fixed, 0x4000, CpuRead) == 0, "mprotect in the window");
        Require(HostProtection(alias) == VM_PROT_READ, "mprotect did not reach the alias");
        Require(sceKernelMprotect(fixed, 0x4000, CpuGpuReadWrite) == 0, "mprotect back to read/write");

        // Unmapping leaves the alias reserved and inaccessible; mapping again gives the same pages.
        Require(sceKernelMunmap(fixed, Bytes) == 0, "munmap in the window");
        Require(HostProtection(alias) == VM_PROT_NONE, "the unmapped alias is still accessible");
        void* again = reinterpret_cast<void*>(Base);
        Require(sceKernelMapDirectMemory(&again, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && again == fixed, "mapping again in the window");
        Require(Load<std::uint32_t>(Base + 0x104) == 0xa5a5a5a5u, "the mapping again lost the direct memory contents");

        // A fixed mapping in the shared region tail at 0x400000000 (SuperTuxKart's first GEM_VA).
        void* tail = reinterpret_cast<void*>(TailBase);
        Require(sceKernelMapDirectMemory(&tail, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && tail == reinterpret_cast<void*>(TailBase),
            "the fixed mapping at 0x400000000 failed");
        const auto tailFaults = Faults();
        Require(Load<std::uint32_t>(TailBase + 0x104) == 0xa5a5a5a5u, "a native load in the shared region tail missed the alias");
        Store<std::uint64_t>(TailBase + 0x2000, 0x0102030405060708ull);
        Require(Load<std::uint64_t>(GuestArena::HostAddress(TailBase) + 0x2000) == 0x0102030405060708ull, "a native store in the shared region tail missed the alias");
        Require(Load<std::uint64_t>(Base + 0x2000) == 0x0102030405060708ull, "the tail and carveout views do not share the pages");
        Require(Faults() >= tailFaults + 3, "the tail accesses did not go through the fault handler");
        ManySites(TailBase + 0x104);
        PatchedSiteRace(TailBase + 0x104);
        FarStubReturns(TailBase + 0x200, Base + 0x300);
        PatchedSitePointers(TailBase + 0x400);
        PatchedSitePointers(Base + 0x408);
        RegisterOffsetSites(TailBase + 0x600);
        SiteReclamation(Base + 0x700);
        {
            // Mutex and condition slots in displaced memory (ASTRO BOT's are at 0x10xxxxxxxx): libkernel resolves them
            // through the slot's host view, so locking takes no fault (VA6's second storm: scePthreadMutexTrylock).
            auto** mutex = reinterpret_cast<void**>(TailBase + 0x3000);
            auto** cond = reinterpret_cast<void**>(TailBase + 0x3008);
            const auto before = Faults();
            Require(scePthreadMutexInit(mutex, nullptr, "displaced") == 0 && scePthreadCondInit(cond, nullptr, "displaced") == 0, "pthread init on displaced slots");
            for (int i = 0; i < 1000; ++i) {
                Require(scePthreadMutexLock(mutex) == 0 && scePthreadCondSignal(cond) == 0 && scePthreadMutexUnlock(mutex) == 0, "pthread lock on a displaced slot");
                Require(scePthreadMutexTrylock(mutex) == 0 && scePthreadMutexUnlock(mutex) == 0, "pthread trylock on a displaced slot");
            }
            Require(scePthreadCondDestroy(cond) == 0 && scePthreadMutexDestroy(mutex) == 0, "pthread destroy on displaced slots");
            const auto taken = Faults() - before;
            if (taken != 0) throw std::runtime_error(std::to_string(taken) + " displaced faults resolving pthread slots");
        }
        Require(sceKernelMprotect(tail, 0x4000, CpuRead) == 0 && HostProtection(GuestArena::HostAddress(TailBase)) == VM_PROT_READ, "mprotect in the tail");
        Require(sceKernelMunmap(tail, Bytes) == 0 && HostProtection(GuestArena::HostAddress(TailBase)) == VM_PROT_NONE, "munmap in the tail");

        const auto refusedWith = [](std::uintptr_t at, std::size_t bytes, const char* expected) {
            try {
                GuestArena::GuestArenaMarkUsed_nid_postfix(reinterpret_cast<void*>(at), bytes);
            } catch (const std::runtime_error& error) {
                std::printf("refused 0x%llx+0x%zx: %s\n", static_cast<unsigned long long>(at), bytes, error.what());
                return std::string(error.what()).find(expected) != std::string::npos;
            }
            GuestArena::GuestArenaRelease_nid_postfix(reinterpret_cast<void*>(at), bytes);
            return false;
        };
        // Across the shared cache's end: half host memory, half alias.
        Require(refusedWith(GuestArena::DisplacedStart() - 0x4000, 0x8000, "straddles"), "a range straddling the shared cache's end was accepted");
        // The PS5 system-reserved range (0x7FFFFC000-0x1000000000) covers the tail/commpage and commpage/carveout
        // boundaries: refused there as on every host, with its own message.
        Require(refusedWith(0xFC0000000ull - 0x4000, 0x8000, "system-reserved"), "a range across the tail/commpage boundary was accepted");
        Require(refusedWith(0x1000000000ull - 0x4000, 0x8000, "system-reserved"), "a range across the commpage/carveout boundary was accepted");

        // A range across the window's edge has no single backing.
        bool refused = false;
        try {
            GuestArena::GuestArenaMarkUsed_nid_postfix(reinterpret_cast<void*>(GuestArena::DisplacedEnd - 0x4000), 0x8000);
        } catch (const std::runtime_error& error) {
            refused = std::string(error.what()).find("straddles") != std::string::npos;
        }
        Require(refused, "a range straddling the window edge was accepted");

        // D37: a fixed request inside the dyld shared cache (readable host memory, neither mappable nor displaceable)
        // is served at request + 0xD000000000 and the address comes back through the in/out argument.
        constexpr std::uintptr_t CacheRequest = 0x300000000ull; // ASTRO BOT's first mapping
        constexpr std::uintptr_t Relocated = CacheRequest + 0xD000000000ull;
        Require(GuestArena::DisplacedStart() > CacheRequest + Bytes, "0x300000000 is not inside the shared cache on this host");
        void* cached = reinterpret_cast<void*>(CacheRequest);
        Require(sceKernelMapDirectMemory(&cached, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0, "a fixed mapping in the shared cache failed");
        Require(cached == reinterpret_cast<void*>(Relocated), "a fixed mapping in the shared cache was not relocated by the constant offset");
        Require(Load<std::uint32_t>(Relocated + 0x104) == 0xa5a5a5a5u, "the relocated mapping does not show the direct memory");
        Store<std::uint64_t>(Relocated + 0x3000, 0x5566778899aabbccull);
        Require(Load<std::uint64_t>(viewAddress + 0x3000) == 0x5566778899aabbccull, "the relocated mapping does not share the pages");
        // Deterministic: the same request again meets the first, as on the console (no-overwrite fails, replace works).
        void* second = reinterpret_cast<void*>(CacheRequest);
        Require(sceKernelMapDirectMemory(&second, Bytes, CpuGpuReadWrite, MapFixed | 0x80, physical, 0x200000) != 0, "a no-overwrite request over a relocated mapping succeeded");
        second = reinterpret_cast<void*>(CacheRequest + 0x200000);
        Require(sceKernelMapDirectMemory(&second, 0x200000, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && second == reinterpret_cast<void*>(Relocated + 0x200000),
            "a fixed request overlapping a relocated mapping did not land at the same offset");
        Require(Load<std::uint32_t>(Relocated + 0x200000 + 0x104) == 0xa5a5a5a5u, "the overlapping relocated mapping does not show the direct memory");
        Require(sceKernelMunmap(cached, Bytes) == 0, "munmap of the relocated mapping");
        Require(HostProtection(Relocated) == VM_PROT_NONE, "the unmapped relocated range is still accessible");
        // A request that runs from the cache into the tail (ASTRO's 0x300000000+0x79800000 does) is relocated whole.
        void* across = reinterpret_cast<void*>((GuestArena::DisplacedStart() - 1) & ~std::uintptr_t{0x1fffff});
        Require(sceKernelMapDirectMemory(&across, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 &&
                    across == reinterpret_cast<void*>(((GuestArena::DisplacedStart() - 1) & ~std::uintptr_t{0x1fffff}) + 0xD000000000ull),
            "a fixed request across the shared cache's end was not relocated");
        Require(sceKernelMunmap(across, Bytes) == 0, "munmap of the relocated straddling mapping");

        Require(sceKernelMunmap(again, Bytes) == 0 && sceKernelMunmap(view, Bytes) == 0, "final munmap");
        Require(sceKernelReleaseDirectMemory(physical, Bytes) == 0, "release direct memory");
        std::puts("guest displaced memory tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "guest displaced memory: %s\n", error.what());
        return 1;
    }
#endif
}
