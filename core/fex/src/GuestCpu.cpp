#include "GuestCpu.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"

#include "BareGuest.hpp"
#include "Bridge.hpp"
#include "CodeCache.hpp"
#include "CodePool.hpp"
#include "ConsoleSyscalls.hpp"
#include "GuestGuards.hpp"
#include "GuestProgram.hpp"
#include "GuestSampler.hpp"
#include "GuestStats.hpp"
#include "Options.hpp"

#include <FEXCore/Config/Config.h>
#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/HostFeatures.h>
#include <FEXCore/Core/SignalDelegator.h>
#include <FEXCore/Debug/InternalThreadState.h>
#include <FEXCore/HLE/SyscallHandler.h>
#include <FEXCore/Utils/AllocatorHooks.h>
#include <FEXCore/Utils/ArchHelpers/Arm64.h>
#include <FEXCore/Utils/DualMap.h>
#include <FEXCore/Utils/LogManager.h>
#include <FEXCore/Utils/Threads.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <csignal>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mach/mach.h>
#include <unistd.h>
#include <mach/mach_vm.h>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <utility>
#include <vector>
#include <sys/sysctl.h>

namespace Aps5Fex {

namespace {

constexpr std::size_t PoolSize = std::size_t {1} << 30;
CodePool* pool = nullptr;

void CreateCodePool() {
    if (pool != nullptr) return;
    pool = new CodePool(PoolSize); // lives as long as the process: FEX code may run until exit
    FEXCore::DualMap::WriteOffset = pool->WriteOffset();
}

// FEXCore maps code buffers read-write-execute through these hooks; they come from the dual-mapped pool instead.
void* PoolMmap(void* address, std::size_t length, int protection, int flags, int file, off_t offset) {
    if ((protection & PROT_EXEC) == 0 || address != nullptr) return ::mmap(address, length, protection, flags, file, offset);
    return pool->Allocate(length);
}

int PoolMunmap(void* address, std::size_t length) {
    if (pool != nullptr && pool->Free(address, length)) return 0;
    return ::munmap(address, length);
}

bool HasFeature(const char* name) {
    int value = 0;
    std::size_t size = sizeof(value);
    return sysctlbyname((std::string("hw.optional.arm.") + name).c_str(), &value, &size, nullptr, 0) == 0 && value != 0;
}

// MIDR_EL1 with Apple's implementer code (0x61) and part number 0.
constexpr std::uint32_t AppleSiliconMidr = 0x61000000;

FEXCore::HostFeatures HostFeatures() {
    FEXCore::HostFeatures features {};
    features.DCacheLineLog2 = 4; // 64-byte lines: DCacheSize() is 4 << DCacheLineLog2
    features.SupportsAES = HasFeature("FEAT_AES");
    features.SupportsCRC = HasFeature("FEAT_CRC32");
    features.SupportsSHA = HasFeature("FEAT_SHA1") && HasFeature("FEAT_SHA256");
    features.SupportsAtomics = HasFeature("FEAT_LSE");
    features.SupportsRAND = HasFeature("FEAT_RNG");
    features.SupportsAFP = HasFeature("FEAT_AFP");
    features.SupportsRCPC = HasFeature("FEAT_LRCPC");
    features.SupportsTSOImm9 = HasFeature("FEAT_LRCPC2");
    features.SupportsPMULL_128Bit = HasFeature("FEAT_PMULL");
    features.SupportsCSSC = HasFeature("FEAT_CSSC");
    features.SupportsFCMA = HasFeature("FEAT_FCMA");
    features.SupportsFlagM = HasFeature("FEAT_FlagM");
    features.SupportsFlagM2 = HasFeature("FEAT_FlagM2");
    features.SupportsFRINTTS = HasFeature("FEAT_FRINTTS");
    features.SupportsRPRES = HasFeature("FEAT_RPRES");
    features.SupportsECV = HasFeature("FEAT_ECV");
    features.SupportsWFXT = HasFeature("FEAT_WFxT");
    features.SupportsMOPS = HasFeature("FEAT_MOPS");
    features.SupportsI8MM = HasFeature("FEAT_I8MM");
    features.SupportsDotProd = HasFeature("FEAT_DotProd");
    features.SupportsAVX = true;
    features.SupportsAES256 = features.SupportsAES;
    // FEX sizes its per-CPU CPUID data (the brand string of leaves 0x80000002-0x80000004, big/little flags) from
    // CPUMIDRs, which only Linux fills (sysfs midr_el1); left empty, those leaves read past an empty vector and the
    // guest crashes on a null product name. macOS does not expose MIDR_EL1 to user code, so every core gets Apple's
    // implementer code with part 0, which FEX's table names "Apple Silicon" (one kind of core: not hybrid).
    int cores = 0;
    std::size_t coresSize = sizeof(cores);
    if (sysctlbyname("hw.logicalcpu", &cores, &coresSize, nullptr, 0) != 0 || cores < 1) cores = 1;
    features.CPUMIDRs.assign(static_cast<std::size_t>(cores), AppleSiliconMidr);
    return features;
}

void LogMessage(LogMan::DebugLevels level, const char* message) {
    if (level <= LogMan::ERROR) std::fprintf(stderr, "[FEXCore] %s\n", message);
}

void LogAssertion(const char* message) {
    std::fprintf(stderr, "[FEXCore] %s\n", message);
    std::abort();
}

struct GuestCode {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    const Bridge* bridge;
    const GuestTlsLayout* tls;
    FEXCore::Context::Context* context;
};

GuestCode guestCode {};

std::optional<std::pair<std::uint64_t, std::uint64_t>> GuestCodeRange(std::uint64_t address) {
    if (guestCode.bridge == nullptr) return std::nullopt;
    if (guestCode.bridge->OwnsStub(address)) return std::pair {guestCode.bridge->StubBase(), guestCode.bridge->StubBase() + guestCode.bridge->StubSize()};
    for (const auto& range : guestCode.ranges)
        if (address >= range.first && address < range.second) return range;
    return std::nullopt;
}

bool IsGuestCode(std::uint64_t address) {
    return GuestCodeRange(address).has_value();
}

class BridgeSyscalls final : public FEXCore::HLE::SyscallHandler {
public:
    explicit BridgeSyscalls(Bridge& bridge) : bridge(bridge) {}

    // FEX-2610 ends the block at `syscall` and runs this from the dispatcher: the guest registers are in
    // frame->State, and execution continues at State.rip, which Bridge::Call advances past the instruction
    // or redirects to a resumed guest frame.
    void HandleSyscall(FEXCore::Core::CpuStateFrame* frame) override {
        auto& state = frame->State;
        const std::uint64_t result =
            bridge.Call(state, state.gregs[FEXCore::X86State::REG_RAX], state.gregs[FEXCore::X86State::REG_RDI]);
        state.gregs[FEXCore::X86State::REG_RAX] = result;
    }

    FEXCore::HLE::ExecutableRangeInfo QueryGuestExecutableRange(FEXCore::Core::InternalThreadState*, std::uint64_t address) override {
        if (modules != nullptr)
            if (const auto range = modules->ExecutableRange(address)) return *range;
        if (const auto range = GuestCodeRange(address)) return {range->first, range->second - range->first, false};
        return {address & ~std::uint64_t {0x3fff}, 0x4000, false};
    }

    std::optional<FEXCore::ExecutableFileSectionInfo> LookupExecutableFileSection(FEXCore::Core::InternalThreadState*, std::uint64_t address) override {
        return modules == nullptr ? std::nullopt : modules->Section(address);
    }

    const GuestModules* modules = nullptr;

private:
    Bridge& bridge;
};

class CallbackReturns final : public FEXCore::SignalDelegator {
public:
    explicit CallbackReturns(const Bridge& bridge) : bridge(bridge) {}

    std::uintptr_t GetThunkCallbackRET() const override { return bridge.CallbackReturn(); }

private:
    const Bridge& bridge;
};

}

struct GuestCpu::State {
    BridgeSyscalls syscalls;
    CallbackReturns returns;
    fextl::unique_ptr<FEXCore::Context::Context> context;
    std::unique_ptr<GuestModules> modules;
    std::filesystem::path coverage; // APS5_FEX_COVERAGE: written at exit
};

namespace {

using ThreadState = FEXCore::Core::InternalThreadState;
using CPUState = FEXCore::Core::CPUState;

constexpr std::size_t Page = 0x4000;
constexpr std::size_t ThreadStackSize = std::size_t {8} << 20;

std::mutex threadsMutex;

std::uint8_t* CreateThreadPointer(const GuestTlsLayout& tls, void** allocation) {
    constexpr std::size_t ControlBlock = 0x100;
    const std::size_t alignment = std::max<std::size_t>(tls.alignment, 64);
    const std::size_t offset = (tls.size + alignment - 1) / alignment * alignment;
    if (posix_memalign(allocation, alignment, offset + ControlBlock) != 0) throw std::bad_alloc();
    std::memset(*allocation, 0, offset + ControlBlock);
    auto* pointer = static_cast<std::uint8_t*>(*allocation) + offset;
    for (const auto& block : tls.blocks)
        std::memcpy(pointer - block.offset, reinterpret_cast<const void*>(block.templateAddress), block.fileSize);
    const auto self = reinterpret_cast<std::uint64_t>(pointer);
    std::memcpy(pointer, &self, sizeof(self));
    std::uint64_t guard = 0;
    arc4random_buf(&guard, sizeof(guard));
    std::memcpy(pointer + 0x28, &guard, sizeof(guard));
    return pointer;
}

class GuestThread {
public:
    GuestThread(std::uint64_t rip, std::uint64_t rsp, std::size_t stackSize) {
        if (stackSize != 0) {
            stack = mmap(nullptr, stackSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
            if (stack == MAP_FAILED) throw std::runtime_error("cannot allocate a guest thread stack");
            this->stackSize = stackSize;
            rsp = reinterpret_cast<std::uint64_t>(stack) + stackSize - Page;
        }
        const std::uint64_t fs = guestCode.tls == nullptr ? 0 : reinterpret_cast<std::uint64_t>(CreateThreadPointer(*guestCode.tls, &tls));
        {
            std::lock_guard lock(threadsMutex);
            thread = guestCode.context->CreateThread();
        }
        GuestSampler::Register(thread);
        GuestStats::Register(thread);
        thread->CurrentFrame->State.rip = rip;
        thread->CurrentFrame->State.gregs[FEXCore::X86State::REG_RSP] = rsp;
        callRet = mmap(nullptr, ThreadState::CALLRET_STACK_SIZE + 2 * Page, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (callRet == MAP_FAILED) throw std::runtime_error("cannot allocate the call-return stack");
        thread->CallRetStackBase = static_cast<std::uint8_t*>(callRet) + Page;
        mprotect(thread->CallRetStackBase, ThreadState::CALLRET_STACK_SIZE, PROT_READ | PROT_WRITE);
        auto& guest = thread->CurrentFrame->State;
        guest.callret_sp = reinterpret_cast<std::uint64_t>(thread->CallRetStackBase) + ThreadState::CALLRET_DEFAULT_OFFSET;
        guest.callret_sp_base = reinterpret_cast<std::uint64_t>(thread->CallRetStackBase);
        thread->CurrentFrame->Pointers.ThunkCallbackRet = guestCode.bridge == nullptr ? 0 : guestCode.bridge->CallbackReturn();

        guest.segment_arrays[CPUState::SEGMENT_ARRAY_INDEX_GDT] = gdt;
        guest.segment_arrays[CPUState::SEGMENT_ARRAY_INDEX_LDT] = gdt;
        guest.cs_idx = CPUState::DEFAULT_USER_CS << 3;
        auto* code = CPUState::GetSegmentFromIndex(guest, guest.cs_idx);
        CPUState::SetGDTBase(code, 0);
        CPUState::SetGDTLimit(code, 0xfffffU);
        code->L = 1;
        code->D = 0;
        guest.cs_cached = CPUState::CalculateGDTBase(*code);
        guest.fs_cached = fs;
    }

    GuestThread(const GuestThread&) = delete;
    GuestThread& operator=(const GuestThread&) = delete;

    ~GuestThread() {
        GuestSampler::Unregister(thread);
        GuestStats::Unregister(thread);
        {
            std::lock_guard lock(threadsMutex);
            guestCode.context->DestroyThread(thread);
        }
        munmap(callRet, ThreadState::CALLRET_STACK_SIZE + 2 * Page);
        std::free(tls);
        if (stack != nullptr) munmap(stack, stackSize);
    }

    ThreadState* Thread() const { return thread; }

private:
    ThreadState* thread = nullptr;
    void* callRet = nullptr;
    void* tls = nullptr;
    void* stack = nullptr;
    std::size_t stackSize = 0;
    CPUState::gdt_segment gdt[32] {};
};

thread_local ThreadState* currentThread = nullptr;
thread_local unsigned guestDepth = 0;

struct AdoptedThread {
    std::unique_ptr<GuestThread> thread;

    ~AdoptedThread() {
        if (guestDepth != 0) (void)thread.release();
    }
};

thread_local AdoptedThread adoptedThread;

ThreadState* CurrentThread() {
    if (currentThread == nullptr) {
        ApplyThreadQos();
        adoptedThread.thread = std::make_unique<GuestThread>(0, 0, ThreadStackSize);
        currentThread = adoptedThread.thread->Thread();
    }
    return currentThread;
}

}

}

extern "C" void Aps5GuestCallEntry();

struct Aps5GuestCallFrame {
    std::uint64_t gpr[8];
    std::uint8_t vector[8][16];
};

namespace Aps5Fex {

namespace {

void CallGuest(ThreadState* thread, std::uint64_t target, Aps5GuestCallFrame* frame) {
    using namespace FEXCore::X86State;
    auto& state = thread->CurrentFrame->State;
    const std::uint64_t rip = state.rip;
    std::uint64_t gregs[16];
    std::memcpy(gregs, state.gregs, sizeof(gregs));
    std::uint64_t xmm[16][2];
    std::memcpy(xmm, state.xmm.sse.data, sizeof(xmm));

    const std::uint64_t top = ((state.gregs[REG_RSP] - 32) & ~std::uint64_t {15}) + 8;
    std::memcpy(reinterpret_cast<void*>(top - 8), &frame->gpr[6], 16);
    state.gregs[REG_RSP] = top;
    state.gregs[REG_RDI] = frame->gpr[0];
    state.gregs[REG_RSI] = frame->gpr[1];
    state.gregs[REG_RDX] = frame->gpr[2];
    state.gregs[REG_RCX] = frame->gpr[3];
    state.gregs[REG_R8] = frame->gpr[4];
    state.gregs[REG_R9] = frame->gpr[5];
    std::memcpy(state.xmm.sse.data, frame->vector, sizeof(frame->vector));
    if (guestDepth == 0 && ClearCodeOnGuestEntry()) thread->CTX->ClearCodeCache(thread);
    ++guestDepth;
    thread->CTX->HandleCallback(thread, target);
    --guestDepth;
    frame->gpr[0] = state.gregs[REG_RAX];
    frame->gpr[1] = state.gregs[REG_RDX];
    std::memcpy(frame->vector, state.xmm.sse.data, 2 * sizeof(frame->vector[0]));

    std::memcpy(state.gregs, gregs, sizeof(gregs));
    std::memcpy(state.xmm.sse.data, xmm, sizeof(xmm));
    state.rip = rip;
}

}

}

extern "C" void Aps5RunGuestCall(std::uint64_t target, Aps5GuestCallFrame* frame) {
    Aps5Fex::CallGuest(Aps5Fex::CurrentThread(), target, frame);
}

std::uint64_t Aps5Fex::CallGuestFunction(std::uint64_t target, const std::uint64_t* arguments) {
    if (!IsGuestCode(target)) {
        using Host = std::uint64_t (*)(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);
        return reinterpret_cast<Host>(target)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
    }
    Aps5GuestCallFrame frame {};
    std::memcpy(frame.gpr, arguments, 6 * sizeof(arguments[0]));
    Aps5RunGuestCall(target, &frame);
    return frame.gpr[0];
}

bool Aps5Fex::IsGuestCodeAddress(std::uint64_t address) {
    return IsGuestCode(address);
}

// Like CallGuest, with the registers and stack set by the caller (setup) and the results read by it
// (finish), for calls whose arguments or result positional mapping places wrongly (GuestCall.cpp).
void Aps5Fex::CallGuestWith(std::uint64_t target, GuestStateHook setup, GuestStateHook finish, void* context) {
    ThreadState* thread = CurrentThread();
    auto& state = thread->CurrentFrame->State;
    const std::uint64_t rip = state.rip;
    std::uint64_t gregs[16];
    std::memcpy(gregs, state.gregs, sizeof(gregs));
    std::uint64_t xmm[16][2];
    std::memcpy(xmm, state.xmm.sse.data, sizeof(xmm));
    setup(state, context);
    ++guestDepth;
    thread->CTX->HandleCallback(thread, target);
    --guestDepth;
    finish(state, context);
    std::memcpy(state.gregs, gregs, sizeof(gregs));
    std::memcpy(state.xmm.sse.data, xmm, sizeof(xmm));
    state.rip = rip;
}

namespace Aps5Fex {

namespace {

struct sigaction previousBus {}, previousSegv {};
FEXCore::ArchHelpers::Arm64::UnalignedHandlerType unalignedHandler = FEXCore::ArchHelpers::Arm64::UnalignedHandlerType::HalfBarrier;

// The M3 raises SIGBUS (BUS_ADRALN) when an ordered access (LDAPR/LDAPUR/STLR/STLUR) or an atomic crosses a
// 16-byte boundary. FEX emulates the access or backpatches it to a plain access plus barrier; the backpatch writes
// through the code pool's writable alias (FEX Arm64.cpp).
void ReportUnaligned(const char* what, std::uint64_t pc, const siginfo_t* info) {
    char message[200];
    const int length = std::snprintf(message, sizeof(message), "[aps5-fex] unaligned access %s: pc 0x%llx instruction 0x%08x address 0x%llx\n", what,
        static_cast<unsigned long long>(pc), *reinterpret_cast<const std::uint32_t*>(pc), static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)));
    if (length > 0) (void)!write(STDERR_FILENO, message, std::min(static_cast<std::size_t>(length), sizeof(message) - 1));
}

bool HandleUnaligned(int signal, siginfo_t* info, __darwin_mcontext64* machine) {
    if (signal != SIGBUS || info->si_code != BUS_ADRALN || currentThread == nullptr) return false;
    const std::uint64_t pc = arm_thread_state64_get_pc(machine->__ss);
    if (!currentThread->CTX->IsAddressInCodeBuffer(currentThread, pc)) {
        // Pool memory outside FEX's code buffers holds the displaced-access stubs (GuestDisplaced.cpp). FEX's
        // backpatch would rewrite the stub's neighbouring words, so the fault goes down the chain to libc's
        // displaced handler, which rewrites the stub's own copy the same way (and names what it cannot handle).
        return false;
    }
    // __x[0..28], __fp, __lr, __sp and __pc are contiguous: FEX indexes the GPRs as x0..x31.
    static_assert(offsetof(__darwin_arm_thread_state64, __x) + 29 * 8 == offsetof(__darwin_arm_thread_state64, __fp));
    static_assert(offsetof(__darwin_arm_thread_state64, __fp) + 8 == offsetof(__darwin_arm_thread_state64, __lr));
    static_assert(offsetof(__darwin_arm_thread_state64, __lr) + 8 == offsetof(__darwin_arm_thread_state64, __sp));
    const auto result = FEXCore::ArchHelpers::Arm64::HandleUnalignedAccess(currentThread, unalignedHandler, pc, machine->__ss.__x);
    if (!result) {
        ReportUnaligned("that FEX cannot emulate", pc, info);
        return false;
    }
    arm_thread_state64_set_pc_fptr(machine->__ss, reinterpret_cast<void*>(pc + *result));
    return true;
}

void OnFault(int signal, siginfo_t* info, void* context) {
    auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
    const std::uint64_t pc = arm_thread_state64_get_pc(machine->__ss);
    if (IsGuestCode(pc) && reinterpret_cast<std::uint64_t>(info->si_addr) == pc) {
        machine->__ss.__x[16] = pc;
        arm_thread_state64_set_pc_fptr(machine->__ss, &Aps5GuestCallEntry);
        return;
    }
    // macOS raises SIGBUS/BUS_ADRALN for protection faults as well. One on a displaced guest address is a host guard
    // (D31, D33); libc's GuestDisplaced handler, further down the chain, runs the access at the alias. fex does not
    // link libc, so it skips the coarse bounds (the dyld shared cache below the window is never guest memory).
    if (!GuestArena::MayBeDisplaced(reinterpret_cast<std::uintptr_t>(info->si_addr)) && HandleUnaligned(signal, info, machine)) return;
    const struct sigaction& previous = signal == SIGBUS ? previousBus : previousSegv;
    if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
        previous.sa_sigaction(signal, info, context);
        return;
    }
    if ((previous.sa_flags & SA_SIGINFO) == 0 && previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN) {
        previous.sa_handler(signal);
        return;
    }
    std::signal(signal, SIG_DFL);
}

void InstallFaultHandler() {
    struct sigaction action {};
    action.sa_sigaction = OnFault;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    sigaction(SIGBUS, &action, &previousBus);
    sigaction(SIGSEGV, &action, &previousSegv);
    for (auto* previous : {&previousBus, &previousSegv})
        if ((previous->sa_flags & SA_SIGINFO) != 0 && previous->sa_sigaction == OnFault) *previous = {};
}

// FEXCore's own helper threads (the disk cache writer) are created through the frontend.
class HostThread final : public FEXCore::Threads::Thread {
public:
    HostThread(FEXCore::Threads::ThreadFunc function, void* argument, FEXCore::Threads::Flags flags, const char* name)
        : start {function, argument, flags.LowPriority, name != nullptr ? name : ""} {
        if (pthread_create(&thread, nullptr, Entry, &start) != 0) throw std::runtime_error("cannot create a FEXCore thread");
        running = true;
    }
    ~HostThread() override {
        if (running) pthread_detach(thread);
    }
    bool joinable() override { return running; }
    bool join(void** result) override {
        if (!running) return false;
        running = false;
        return pthread_join(thread, result) == 0;
    }
    bool detach() override {
        if (!running) return false;
        running = false;
        return pthread_detach(thread) == 0;
    }
    bool IsSelf() override { return running && pthread_equal(thread, pthread_self()); }

private:
    struct Start {
        FEXCore::Threads::ThreadFunc function;
        void* argument;
        bool lowPriority;
        std::string name;
    };
    static void* Entry(void* data) {
        const auto* start = static_cast<const Start*>(data);
        if (!start->name.empty()) pthread_setname_np(start->name.c_str());
        if (start->lowPriority) pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        return start->function(start->argument);
    }

    Start start;
    pthread_t thread {};
    bool running = false;
};

fextl::unique_ptr<FEXCore::Threads::Thread> CreateHostThread(FEXCore::Threads::ThreadFunc function, void* argument,
                                                            FEXCore::Threads::Flags flags, const char* name) {
    return fextl::make_unique<HostThread>(function, argument, flags, name);
}

void CleanupAfterFork() {}

void PrepareFex(const std::vector<std::pair<FEXCore::Config::ConfigOption, std::string>>& config) {
    FEXCore::Threads::Thread::SetInternalPointers({CreateHostThread, CleanupAfterFork});
    LogMan::Msg::InstallHandler(LogMessage);
    LogMan::Throw::InstallHandler(LogAssertion);
    CreateCodePool();
    FEXCore::Allocator::mmap = PoolMmap;
    FEXCore::Allocator::munmap = PoolMunmap;
    FEXCore::Config::Initialize();
    FEXCore::Config::Load();
    FEXCore::Config::ReloadMetaLayer();
    FEXCore::Config::Set(FEXCore::Config::CONFIG_IS64BIT_MODE, "1");
    // A locked access that crosses 16 bytes is emulated in the SIGBUS handler with two CASes, which tear under
    // contention unless such accesses share a lock (x86 makes them atomic). Only that slow path takes the lock.
    FEXCore::Config::Set(FEXCore::Config::CONFIG_STRICTINPROCESSSPLITLOCKS, "1");
    // FEX's memory-saving lookup defaults (no L2 table, an L1 that shrinks) sent ASTRO BOT's indirect branches through
    // CompileCode's map lookup 700-900 times a second per process (lookup-cache misses, APS5_FEX_STATS). The L2 table is
    // a lazily touched virtual reservation; with it and a full-size L1 the misses drop to a few per 30 s (D71).
    // APS5_FEX_CONFIG=DisableL2Cache=1,DynamicL1Cache=1 restores FEX's defaults.
    FEXCore::Config::Set(FEXCore::Config::CONFIG_DISABLEL2CACHE, "0");
    FEXCore::Config::Set(FEXCore::Config::CONFIG_DYNAMICL1CACHE, "0");
    for (const auto& [option, value] : config)
        FEXCore::Config::Set(option, value);
    LoadOptions();
    FEX_CONFIG_OPT(halfBarrierTso, HALFBARRIERTSOENABLED);
    unalignedHandler = halfBarrierTso() ? FEXCore::ArchHelpers::Arm64::UnalignedHandlerType::HalfBarrier
                                        : FEXCore::ArchHelpers::Arm64::UnalignedHandlerType::NonAtomic;
}

}

namespace {

struct ExitFlush {
    FEXCore::Context::Context* context;
    const GuestModules* modules;
    std::filesystem::path coverage;
};
std::optional<ExitFlush> exitFlush;

// The guest ends the process with exit(): queued disk-cache writes must reach the file first.
void FlushAtExit() {
    if (!exitFlush) return;
    exitFlush->context->FlushDiskCache();
    if (!exitFlush->coverage.empty()) Coverage::Write(exitFlush->coverage, "run", *exitFlush->modules, nullptr);
}

bool atExitRegistered = false;

}

GuestCpu::GuestCpu(Bridge& bridge, const GuestProgram& program) {
    std::vector<std::pair<FEXCore::Config::ConfigOption, std::string>> config;
    if (const auto cache = CodeCacheDirectory(program.Executable().Path())) {
        std::filesystem::create_directories(*cache);
        config.emplace_back(FEXCore::Config::CONFIG_DISKCACHE, "1");
        // FEX appends the file names to this path as is.
        config.emplace_back(FEXCore::Config::CONFIG_DISKCACHEPATH, cache->string() + "/");
    }
    PrepareFex(config);

    // The disk cache is set up when the context is created, from the configuration above.
    state.reset(new State {BridgeSyscalls(bridge), CallbackReturns(bridge), FEXCore::Context::Context::CreateNewContext(HostFeatures()),
                           std::make_unique<GuestModules>(program), {}});
    state->syscalls.modules = state->modules.get();
    if (const char* coverage = std::getenv("APS5_FEX_COVERAGE"); coverage != nullptr && *coverage != 0) {
        state->coverage = coverage;
        Coverage::Enable();
    }
    if (Coverage::Enabled()) state->context->SetCompiledBlockObserver(Coverage::Observe);
    state->context->SetSyscallHandler(&state->syscalls);
    state->context->SetSignalDelegator(&state->returns);
    state->context->SetThunkHandler(Bridge::Thunks());
    state->context->EnableExitOnHLT();
    if (!state->context->InitCore()) throw std::runtime_error("FEXCore failed to initialise");
    if (const std::uint64_t gadget = CreateConsoleSyscallGadget(bridge)) state->context->AddThunkTrampolineIRHandler(ConsoleSyscallGadget, gadget);
    RedirectGuestGuards(bridge, program, *state->context);
    guestCode = {program.CodeRanges(), &bridge, &program.Tls(), state->context.get()};
    InstallFaultHandler();
    GuestStats::Start();
    GuestSampler::Start(state->context.get(), state->modules.get(), state->returns.GetConfig().SRAGPRMapping,
                        [&bridge](std::uint64_t rip) { return bridge.StubName(rip); });
    if (!atExitRegistered) atExitRegistered = std::atexit(FlushAtExit) == 0;
    exitFlush = ExitFlush {state->context.get(), state->modules.get(), state->coverage};
}

GuestCpu::~GuestCpu() {
    GuestSampler::Stop(state->context.get());
    if (exitFlush && exitFlush->context == state->context.get()) exitFlush.reset();
    state->context->FlushDiskCache();
    guestCode = {};
}

std::size_t GuestCpu::Precompile(unsigned threads, const std::filesystem::path& coverage) {
    if (!Coverage::Enabled()) throw std::runtime_error("Precompile needs the compiled-block observer (Coverage::Enable before GuestCpu)");
    std::vector<AotSeeds> seeds;
    std::vector<std::uint64_t> addresses;
    for (const auto& module : state->modules->Modules()) {
        seeds.push_back(FindAotSeeds(module.seedImage));
        addresses.insert(addresses.end(), seeds.back().addresses.begin(), seeds.back().addresses.end());
    }
    const auto compile = [&](const std::vector<std::uint64_t>& batch) {
        std::atomic<std::size_t> next {0};
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < std::max(1u, threads); ++t) {
            workers.emplace_back([&] {
                ApplyThreadQos();
                GuestThread thread(0, 0, 0);
                currentThread = thread.Thread();
                for (std::size_t i = next++; i < batch.size(); i = next++) state->context->CompileRIP(thread.Thread(), batch[i]);
                currentThread = nullptr;
            });
        }
        for (auto& worker : workers) worker.join();
    };
    compile(addresses);
    // A block ends at a call: the code after it, the return site, starts where the compiled bytes stop. Inside a
    // function's FDE range, seed the first byte of every gap that follows compiled code, and repeat.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> functions;
    for (const auto& found : seeds) functions.insert(functions.end(), found.fdeRanges.begin(), found.fdeRanges.end());
    std::sort(functions.begin(), functions.end());
    std::vector<std::uint64_t> tried(addresses.begin(), addresses.end());
    std::sort(tried.begin(), tried.end());
    constexpr int MaxPasses = 16;
    for (int pass = 0; pass < MaxPasses && !functions.empty(); ++pass) {
        const auto compiled = Coverage::CompiledRanges();
        std::vector<std::uint64_t> batch;
        for (const auto& [begin, end] : compiled) {
            const auto function = std::upper_bound(functions.begin(), functions.end(), std::pair {end, ~std::uint64_t {0}});
            if (function == functions.begin()) continue;
            const auto& [low, high] = *std::prev(function);
            if (end <= low || end >= high || std::binary_search(tried.begin(), tried.end(), end)) continue;
            batch.push_back(end);
        }
        if (batch.empty()) break;
        std::sort(batch.begin(), batch.end());
        batch.erase(std::unique(batch.begin(), batch.end()), batch.end());
        compile(batch);
        tried.insert(tried.end(), batch.begin(), batch.end());
        std::sort(tried.begin(), tried.end());
        addresses.insert(addresses.end(), batch.begin(), batch.end());
    }
    state->context->FlushDiskCache();
    if (!coverage.empty()) Coverage::Write(coverage, "precompile", *state->modules, &seeds);
    return addresses.size();
}

namespace {

struct ProgramStart {
    std::uint64_t start;
    std::vector<std::uint64_t> initializers;
};

ProgramStart programStart;

void StartProgram(void* block, void*) {
    ThreadState* thread = CurrentThread();
    for (const std::uint64_t initializer : programStart.initializers) {
        Aps5GuestCallFrame frame {};
        CallGuest(thread, initializer, &frame);
    }
    Aps5GuestCallFrame frame {};
    frame.gpr[0] = reinterpret_cast<std::uint64_t>(block);
    CallGuest(thread, programStart.start, &frame);
}

}

void GuestCpu::Start(Bridge& bridge, std::uint64_t start, std::uint64_t block, std::vector<std::uint64_t> initializers) {
    using StartGuest = void (*)(void (*)(void*, void*), void*);
    auto* startGuest = reinterpret_cast<StartGuest>(bridge.HostSymbol("Aps5StartGuest_nid_no_patch"));
    if (startGuest == nullptr) throw std::runtime_error("libkernel has no Aps5StartGuest");
    programStart = {start, std::move(initializers)};
    startGuest(StartProgram, reinterpret_cast<void*>(block));
    std::abort();
}

namespace {

std::uint64_t RunHere(FEXCore::Context::Context& context, std::uint64_t rip, std::uint64_t rsp, const std::vector<std::uint64_t>& initializers);

#ifdef __APPLE__
// The entry-point form of Run on macOS (no StartFunction): AppKit only creates windows on the process main
// thread, and the libraries reach it through Aps5RunOnMainThread, which needs the main thread in its run
// loop. So, as for Start, the guest's main thread runs on another thread (Aps5StartGuest, main-thread stack
// size) and the process main thread serves the main queue. Before this, the guest kept the process main thread,
// Aps5RunOnMainThread ran its function in place on the calling thread, and the first windowed sceVideoOutOpen
// failed with "NSWindow should only be instantiated on the main thread!" (DOOM PPSA99666, 2026-10-13).
struct ProgramEntry {
    FEXCore::Context::Context* context;
    std::uint64_t rip;
    std::uint64_t rsp;
    std::vector<std::uint64_t> initializers;
};

ProgramEntry programEntry;

void RunProgramEntry(void*, void*) {
    try {
        const std::uint64_t rax = RunHere(*programEntry.context, programEntry.rip, programEntry.rsp, programEntry.initializers);
        std::fprintf(stderr, "[aps5-fex] the guest halted with RAX %#llx\n", static_cast<unsigned long long>(rax));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[aps5-fex] %s\n", error.what());
    }
    std::fflush(nullptr);
    std::exit(1);
}
#endif

}

std::uint64_t GuestCpu::Run(std::uint64_t rip, std::uint64_t rsp, const std::vector<std::uint64_t>& initializers) {
#ifdef __APPLE__
    using StartGuest = void (*)(void (*)(void*, void*), void*);
    if (guestCode.bridge != nullptr) {
        if (auto* startGuest = reinterpret_cast<StartGuest>(guestCode.bridge->HostSymbol("Aps5StartGuest_nid_no_patch"))) {
            programEntry = {state->context.get(), rip, rsp, initializers};
            startGuest(RunProgramEntry, nullptr);
            std::abort();
        }
    }
#endif
    return RunHere(*state->context, rip, rsp, initializers);
}

namespace {

std::uint64_t RunHere(FEXCore::Context::Context& context, std::uint64_t rip, std::uint64_t rsp, const std::vector<std::uint64_t>& initializers) {
    ApplyThreadQos();
    GuestThread main(rip, rsp, 0);
    currentThread = main.Thread();
    for (const std::uint64_t initializer : initializers) {
        Aps5GuestCallFrame frame {};
        CallGuest(currentThread, initializer, &frame);
    }
    context.ExecuteThread(currentThread);
    currentThread = nullptr;
    return main.Thread()->CurrentFrame->State.gregs[FEXCore::X86State::REG_RAX];
}

}

namespace {

class BareSyscalls final : public FEXCore::HLE::SyscallHandler {
public:
    BareSyscalls(FEXCore::Context::Context& context, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& executable)
        : context(context), executable(executable) {}

    void HandleSyscall(FEXCore::Core::CpuStateFrame* frame) override {
        frame->State.rip += 2;
        if (count++ == 0) first = frame->State.gregs[FEXCore::X86State::REG_RAX];
        constexpr std::uint64_t LinuxNoSyscall = 38;
        frame->State.gregs[FEXCore::X86State::REG_RAX] = -LinuxNoSyscall;
    }

    void InvalidateGuestCodeRange(FEXCore::Core::InternalThreadState*, std::uint64_t start, std::uint64_t length) override {
        std::lock_guard lock(context.GetCodeInvalidationMutex());
        context.InvalidateCodeBuffersCodeRange(start, length);
        if (currentThread != nullptr) context.InvalidateThreadCachedCodeRange(currentThread, start, length);
    }

    FEXCore::HLE::ExecutableRangeInfo QueryGuestExecutableRange(FEXCore::Core::InternalThreadState*, std::uint64_t address) override {
        for (const auto& [begin, end] : executable)
            if (address >= begin && address < end) return {begin, end - begin, true};
        return {address, 0, false};
    }

    std::optional<FEXCore::ExecutableFileSectionInfo> LookupExecutableFileSection(FEXCore::Core::InternalThreadState*, std::uint64_t) override {
        return std::nullopt;
    }

    std::uint32_t count = 0;
    std::uint64_t first = 0;

private:
    FEXCore::Context::Context& context;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> executable;
};

FEXCore::Context::Context* bareContext = nullptr;

}

FEXCore::HostFeatures RunnerHostFeatures() {
    return HostFeatures();
}

void RunBareGuest(const BareGuest& guest, BareGuestResult& result) {
    PrepareFex(guest.config);
    auto context = FEXCore::Context::Context::CreateNewContext(HostFeatures());
    BareSyscalls syscalls(*context, guest.executable);
    FEXCore::SignalDelegator signals;
    context->SetSyscallHandler(&syscalls);
    context->SetSignalDelegator(&signals);
    context->EnableExitOnHLT();
    if (!context->InitCore()) throw std::runtime_error("FEXCore failed to initialise");
    guestCode = {{}, nullptr, nullptr, context.get()};
    bareContext = context.get();
    InstallFaultHandler();
    {
        GuestThread thread(guest.rip, guest.rsp, 0);
        currentThread = thread.Thread();
        context->ExecuteThread(currentThread);
        std::memcpy(&result.state, &currentThread->CurrentFrame->State, sizeof(result.state));
        __uint128_t low[CPUState::NUM_XMMS];
        __uint128_t high[CPUState::NUM_XMMS];
        context->ReconstructXMMRegisters(currentThread, low, high);
        for (std::size_t index = 0; index < CPUState::NUM_XMMS; ++index) {
            std::memcpy(&result.state.xmm.avx.data[index][0], &low[index], sizeof(low[index]));
            std::memcpy(&result.state.xmm.avx.data[index][2], &high[index], sizeof(high[index]));
        }
        currentThread = nullptr;
    }
    bareContext = nullptr;
    guestCode = {};
    result.syscalls = syscalls.count;
    result.firstSyscall = syscalls.first;
}

std::uint64_t BareGuestRip(std::uint64_t hostPc) {
    if (bareContext == nullptr || currentThread == nullptr) return 0;
    return bareContext->RestoreRIPFromHostPC(currentThread, hostPc);
}

}
