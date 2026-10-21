#include "Bridge.hpp"
#include "CodePool.hpp"

#include "BridgeSignatures.hpp"
#include "GuestCall.hpp"
#include "GuestCpu.hpp"
#include "GuestImage.hpp"
#include "X87Convert.hpp"

#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Debug/InternalThreadState.h>
#include <FEXCore/Core/Thunks.h>
#include <FEXCore/IR/IR.h>
#include <FEXCore/Core/X86Enums.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/sysctl.h>

struct Aps5NativeArguments {
    std::uint64_t gpr[8];
    std::uint8_t vector[8][16];
    const std::uint64_t* stack;
    std::uint64_t stackSlots;
    std::uint64_t x8;
};
static_assert(offsetof(Aps5NativeArguments, vector) == 64 && offsetof(Aps5NativeArguments, stack) == 192 &&
              offsetof(Aps5NativeArguments, stackSlots) == 200 && offsetof(Aps5NativeArguments, x8) == 208);

struct Aps5NativeResult {
    std::uint64_t gpr[2];
    std::uint8_t vector[4][16];
};
static_assert(offsetof(Aps5NativeResult, vector) == 16);

extern "C" void Aps5NativeCall(const void* target, const Aps5NativeArguments* arguments, Aps5NativeResult* result);
extern "C" void Aps5BridgeThunkEntry(void* block);

namespace Aps5Fex {

namespace {

constexpr std::size_t BlockSize = 0xb8;
constexpr std::size_t BlockVectors = 0x30;

constexpr std::uint64_t StackArgumentSlots = 8;

using namespace FEXCore::X86State;

struct GuestCall {
    FEXCore::Core::CPUState* state;
    std::uint64_t block;
    const GuestCall* outer;
};

thread_local const GuestCall* currentCall = nullptr;

std::size_t ReadableBytes(const void* address, std::size_t size) {
    constexpr std::uintptr_t Page = 0x4000;
    const auto first = reinterpret_cast<std::uintptr_t>(address);
    const auto next = (first & ~(Page - 1)) + Page;
    if (first + size <= next) return size;
    char resident = 0;
    if (mincore(reinterpret_cast<void*>(next), 1, &resident) == 0) return size;
    return next - first;
}

constexpr std::uint32_t DwarfRegisters[16] = {REG_RAX, REG_RDX, REG_RCX, REG_RBX, REG_RSI, REG_RDI, REG_RBP, REG_RSP,
                                              REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};

void CaptureGuestFrame(std::uintptr_t* registers) {
    if (currentCall == nullptr) {
        std::fprintf(stderr, "[aps5-fex] a library unwound outside a guest call\n");
        std::abort();
    }
    const auto* saved = reinterpret_cast<const std::uint64_t*>(currentCall->block);
    for (std::size_t i = 0; i < 16; ++i) registers[i] = currentCall->state->gregs[DwarfRegisters[i]];
    registers[5] = saved[0];
    registers[4] = saved[1];
    registers[1] = saved[2];
    registers[2] = saved[3];
    registers[8] = saved[4];
    registers[9] = saved[5];
    registers[7] = currentCall->block + BlockSize + 8;
    registers[16] = *reinterpret_cast<const std::uint64_t*>(currentCall->block + BlockSize);
}

struct GuestFrame {
    std::uintptr_t registers[17];
};

[[noreturn]] void ResumeGuestFrame(const std::uintptr_t* registers) {
    GuestFrame frame;
    std::memcpy(frame.registers, registers, sizeof(frame.registers));
    throw frame;
}

// libc's GuestMachineState (core/libs/prx/libc/include/GuestMachineState.hpp).
struct MachineState {
    std::uint64_t gregs[16];
    std::uint64_t rip;
    std::uint64_t rflags;
    alignas(16) std::uint8_t fxsave[512];
    std::uint64_t fsBase;
};
static_assert(offsetof(MachineState, fxsave) == 144 && offsetof(MachineState, fsBase) == 656 && sizeof(MachineState) == 672);

struct MachineFrame {
    MachineState state;
};

constexpr std::size_t FxsaveStatus = 2;
constexpr std::size_t FxsaveTags = 4;
constexpr std::size_t FxsaveMxcsr = 24;
constexpr std::size_t FxsaveMxcsrMask = 28;
constexpr std::size_t FxsaveX87 = 32;
constexpr std::size_t FxsaveXmm = 160;
constexpr std::uint32_t X87ConditionBits[4][2] = {{X87FLAG_C0_LOC, 8}, {X87FLAG_C1_LOC, 9}, {X87FLAG_C2_LOC, 10}, {X87FLAG_C3_LOC, 14}};

FEXCore::Core::InternalThreadState* ThreadOf(FEXCore::Core::CPUState& state) {
    static_assert(offsetof(FEXCore::Core::CpuStateFrame, State) == 0);
    return reinterpret_cast<FEXCore::Core::CpuStateFrame*>(&state)->Thread;
}

// The guest's state at the innermost guest call, as the guest sees it after the call returns.
void GetMachineState(MachineState* machine) {
    std::uintptr_t registers[17];
    CaptureGuestFrame(registers);
    auto& state = *currentCall->state;
    std::memset(machine, 0, sizeof(*machine));
    std::memcpy(machine->gregs, registers, sizeof(machine->gregs));
    machine->rip = registers[16];
    auto* thread = ThreadOf(state);
    machine->rflags = thread != nullptr ? thread->CTX->ReconstructCompactedEFLAGS(thread, false, nullptr, 0) : 0x202;
    std::uint8_t* image = machine->fxsave;
    std::memcpy(image, &state.FCW, sizeof(state.FCW));
    const unsigned top = state.flags[X87FLAG_TOP_LOC] & 7u;
    std::uint16_t status = static_cast<std::uint16_t>(top << 11);
    for (const auto& [flag, bit] : X87ConditionBits) status |= static_cast<std::uint16_t>((state.flags[flag] & 1u) << bit);
    std::memcpy(image + FxsaveStatus, &status, sizeof(status));
    image[FxsaveTags] = state.AbridgedFTW;
    std::memcpy(image + FxsaveMxcsr, &state.mxcsr, sizeof(state.mxcsr));
    const std::uint32_t mask = 0xffbf;
    std::memcpy(image + FxsaveMxcsrMask, &mask, sizeof(mask));
    for (unsigned index = 0; index < 8; ++index) std::memcpy(image + FxsaveX87 + 16 * index, state.mm[(top + index) & 7u], 10);
    // xmm0-7 as the guest passed them to the call (the import trampoline saved them), xmm8-15 untouched.
    std::memcpy(image + FxsaveXmm, reinterpret_cast<const void*>(currentCall->block + BlockVectors), 8 * 16);
    std::memcpy(image + FxsaveXmm + 8 * 16, state.xmm.sse.data[8], 8 * 16);
    machine->fsBase = state.fs_cached;
}

[[noreturn]] void SetMachineState(const MachineState* machine) {
    if (currentCall == nullptr) {
        std::fprintf(stderr, "[aps5-fex] a library resumed the guest outside a guest call\n");
        std::abort();
    }
    throw MachineFrame {*machine};
}

// FEXCore keeps the SSE rounding mode, flush-to-zero and denormals-are-zero in the host FPCR, which only its
// LDMXCSR/FXRSTOR code writes (OpDispatchBuilder::RestoreMXCSRState, SetRoundingMode in JIT/MiscOps.cpp).
// A resumed state with another MXCSR sets them here the same way: RC (bits 13-14) bit-swapped into RMode
// (22-23), FTZ (15) into FZ (24), DAZ (6) into FIZ (0) where the host has FEAT_AFP.
bool HostHasAfp() {
    static const bool afp = [] {
        int value = 0;
        std::size_t size = sizeof(value);
        return sysctlbyname("hw.optional.arm.FEAT_AFP", &value, &size, nullptr, 0) == 0 && value != 0;
    }();
    return afp;
}

void ApplyMxcsrToHost(std::uint32_t mxcsr) {
    const std::uint64_t rc = (mxcsr >> 13) & 3u;
    const std::uint64_t mode = ((rc & 1u) << 1) | (rc >> 1);
    std::uint64_t fpcr = __builtin_arm_rsr64("fpcr");
    fpcr = (fpcr & ~(std::uint64_t {3} << 22)) | (mode << 22);
    fpcr = (fpcr & ~(std::uint64_t {1} << 24)) | (std::uint64_t {(mxcsr >> 15) & 1u} << 24);
    if (HostHasAfp()) fpcr = (fpcr & ~std::uint64_t {1}) | ((mxcsr >> 6) & 1u);
    __builtin_arm_wsr64("fpcr", fpcr);
}

void ApplyMachineState(FEXCore::Core::CPUState& state, const MachineState& machine) {
    for (std::size_t i = 0; i < 16; ++i) state.gregs[DwarfRegisters[i]] = machine.gregs[i];
    state.rip = machine.rip;
    const std::uint8_t* image = machine.fxsave;
    std::memcpy(&state.FCW, image, sizeof(state.FCW));
    std::uint16_t status = 0;
    std::memcpy(&status, image + FxsaveStatus, sizeof(status));
    const unsigned top = (status >> 11) & 7u;
    state.flags[X87FLAG_TOP_LOC] = static_cast<std::uint8_t>(top);
    for (const auto& [flag, bit] : X87ConditionBits) state.flags[flag] = static_cast<std::uint8_t>((status >> bit) & 1u);
    state.AbridgedFTW = image[FxsaveTags];
    std::memcpy(&state.mxcsr, image + FxsaveMxcsr, sizeof(state.mxcsr));
    state.mxcsr &= 0xffc0u;
    ApplyMxcsrToHost(state.mxcsr);
    for (unsigned index = 0; index < 8; ++index) {
        std::memset(state.mm[(top + index) & 7u], 0, sizeof(state.mm[0]));
        std::memcpy(state.mm[(top + index) & 7u], image + FxsaveX87 + 16 * index, 10);
    }
    std::memcpy(state.xmm.sse.data, image + FxsaveXmm, 16 * 16);
    if (machine.fsBase != 0) state.fs_cached = machine.fsBase;
    if (auto* thread = ThreadOf(state)) thread->CTX->SetFlagsFromCompactedEFLAGS(thread, static_cast<std::uint32_t>(machine.rflags));
}

// makecontext's guest entry (rdi = the context, rsi = its function, rdx = six arguments; rsp = the context's
// stack, a null return address on top): call the function, then Aps5MakecontextFinish_nid_no_patch(context).
constexpr std::uint8_t MakecontextEntryCode[] = {
    0x48, 0x89, 0xfb,       // mov rbx, rdi
    0x48, 0x89, 0xf0,       // mov rax, rsi
    0x49, 0x89, 0xd2,       // mov r10, rdx
    0x49, 0x8b, 0x3a,       // mov rdi, [r10]
    0x49, 0x8b, 0x72, 0x08, // mov rsi, [r10 + 8]
    0x49, 0x8b, 0x52, 0x10, // mov rdx, [r10 + 16]
    0x49, 0x8b, 0x4a, 0x18, // mov rcx, [r10 + 24]
    0x4d, 0x8b, 0x42, 0x20, // mov r8, [r10 + 32]
    0x4d, 0x8b, 0x4a, 0x28, // mov r9, [r10 + 40]
    0x48, 0x83, 0xec, 0x08, // sub rsp, 8
    0xff, 0xd0,             // call rax
    0x48, 0x89, 0xdf,       // mov rdi, rbx
    0xe8, 0, 0, 0, 0,       // call Aps5MakecontextFinish (rel32 patched)
    0x0f, 0x0b,             // ud2
};
constexpr std::size_t MakecontextFinishCall = 41;
static_assert(MakecontextEntryCode[MakecontextFinishCall] == 0xe8);

std::vector<std::uint64_t> tlsOffsets;

void* TlsGetAddr(const std::uint64_t* index) {
    // Module ids start at 1 (GuestProgram numbers image i as i + 1); 0 is no module.
    if (currentCall == nullptr || index[0] == 0 || index[0] >= tlsOffsets.size()) {
        std::fprintf(stderr, "[aps5-fex] __tls_get_addr for TLS module %llu\n", static_cast<unsigned long long>(index[0]));
        std::abort();
    }
    return reinterpret_cast<void*>(currentCall->state->fs_cached - tlsOffsets[index[0]] + index[1]);
}

void GuestControl(std::uint32_t* mxcsr, std::uint16_t* fcw) {
    if (currentCall == nullptr) {
        std::fprintf(stderr, "[aps5-fex] a library read the guest's control words outside a guest call\n");
        std::abort();
    }
    *mxcsr = currentCall->state->mxcsr;
    *fcw = currentCall->state->FCW;
}

void FiberEntryReturned() {
    std::fprintf(stderr, "[aps5-fex] the entry function of a fiber returned\n");
    std::abort();
}

struct ImageInfo {
    std::uintptr_t address;
    const char* name;
    const void* programHeaders;
    std::uint16_t programHeaderCount;
};

struct VaList {
    std::uint32_t gpOffset;
    std::uint32_t fpOffset;
    void* overflowArea;
    void* registerSaveArea;
};
static_assert(sizeof(VaList) == 24);

struct VariadicExport {
    std::string_view name;
    std::uint8_t kind;
    std::uint8_t fixed;
};

constexpr std::uint8_t GuestList = 1;
constexpr std::uint8_t Integers = 2;
constexpr std::uint8_t Unsupported = 3;

constexpr VariadicExport VariadicExports[] = {
    {"asprintf_nid_postfix", GuestList, 2},
    {"err_nid_postfix", GuestList, 2},
    {"errc_nid_postfix", GuestList, 3},
    {"errx_nid_postfix", GuestList, 2},
    {"fprintf_nid_postfix", GuestList, 2},
    {"fscanf_nid_postfix", GuestList, 2},
    {"libc_printf_nid_postfix", GuestList, 1},
    {"makecontext_nid_postfix", GuestList, 3},
    {"openat_nid_postfix", GuestList, 3},
    {"printf_nid_postfix", GuestList, 1},
    {"printf_s_nid_postfix", GuestList, 1},
    {"sceKernelFcntl", GuestList, 2},
    {"snprintf_nid_postfix", GuestList, 3},
    {"snprintf_s_nid_postfix", GuestList, 3},
    {"snwprintf_s_nid_postfix", GuestList, 3},
    {"sprintf_nid_postfix", GuestList, 2},
    {"sprintf_s_nid_postfix", GuestList, 3},
    {"sscanf_nid_postfix", GuestList, 2},
    {"sscanf_s_nid_postfix", GuestList, 2},
    {"swprintf_nid_postfix", GuestList, 3},
    {"swprintf_s_nid_postfix", GuestList, 3},
    {"swscanf_nid_postfix", GuestList, 2},
    {"swscanf_s_nid_postfix", GuestList, 2},
    {"syslog_nid_postfix", GuestList, 2},
    {"warn_nid_postfix", GuestList, 1},
    {"warnc_nid_postfix", GuestList, 2},
    {"warnx_nid_postfix", GuestList, 1},
    {"wprintf_nid_postfix", GuestList, 1},
    {"_open_nid_postfix", Integers, 2},
    {"fcntl_nid_postfix", Integers, 2},
    {"_fcntl_nid_postfix", Integers, 2},
};

constexpr std::string_view X87Results[] = {"strtold_nid_postfix", "wcstold_nid_postfix", "__floatdixf_nid_postfix",
                                           "__floatundixf_nid_postfix", "__floattixf_nid_postfix", "__floatuntixf_nid_postfix"};

template <std::size_t... Sizes>
constexpr std::array<std::uint8_t, (Sizes + ...)> Join(const std::array<std::uint8_t, Sizes>&... parts) {
    std::array<std::uint8_t, (Sizes + ...)> joined {};
    std::size_t at = 0;
    ((std::copy(parts.begin(), parts.end(), joined.begin() + at), at += Sizes), ...);
    return joined;
}

using Bytes4 = std::array<std::uint8_t, 4>;
using Bytes5 = std::array<std::uint8_t, 5>;
using Bytes6 = std::array<std::uint8_t, 6>;
using Bytes7 = std::array<std::uint8_t, 7>;
using Bytes9 = std::array<std::uint8_t, 9>;

constexpr Bytes7 SubRspBlock {0x48, 0x81, 0xec, 0xb8, 0x00, 0x00, 0x00};
constexpr Bytes7 AddRspBlock {0x48, 0x81, 0xc4, 0xb8, 0x00, 0x00, 0x00};
constexpr Bytes4 StoreRdi {0x48, 0x89, 0x3c, 0x24};
constexpr Bytes5 StoreRsi {0x48, 0x89, 0x74, 0x24, 0x08};
constexpr Bytes5 StoreRdx {0x48, 0x89, 0x54, 0x24, 0x10};
constexpr Bytes5 StoreRcx {0x48, 0x89, 0x4c, 0x24, 0x18};
constexpr Bytes5 StoreR8 {0x4c, 0x89, 0x44, 0x24, 0x20};
constexpr Bytes5 StoreR9 {0x4c, 0x89, 0x4c, 0x24, 0x28};
constexpr Bytes6 StoreXmm0 {0xf3, 0x0f, 0x7f, 0x44, 0x24, 0x30};
constexpr Bytes6 StoreXmm1 {0xf3, 0x0f, 0x7f, 0x4c, 0x24, 0x40};
constexpr Bytes6 StoreXmm2 {0xf3, 0x0f, 0x7f, 0x54, 0x24, 0x50};
constexpr Bytes6 StoreXmm3 {0xf3, 0x0f, 0x7f, 0x5c, 0x24, 0x60};
constexpr Bytes6 StoreXmm4 {0xf3, 0x0f, 0x7f, 0x64, 0x24, 0x70};
constexpr Bytes9 StoreXmm5 {0xf3, 0x0f, 0x7f, 0xac, 0x24, 0x80, 0x00, 0x00, 0x00};
constexpr Bytes9 StoreXmm6 {0xf3, 0x0f, 0x7f, 0xb4, 0x24, 0x90, 0x00, 0x00, 0x00};
constexpr Bytes9 StoreXmm7 {0xf3, 0x0f, 0x7f, 0xbc, 0x24, 0xa0, 0x00, 0x00, 0x00};
constexpr std::array<std::uint8_t, 3> MoveRspToRdi {0x48, 0x89, 0xe7};
constexpr std::array<std::uint8_t, 2> Syscall {0x0f, 0x05};
constexpr Bytes4 LoadRax {0x48, 0x8b, 0x04, 0x24};
constexpr Bytes5 LoadRdx {0x48, 0x8b, 0x54, 0x24, 0x08};
constexpr Bytes6 LoadXmm0 {0xf3, 0x0f, 0x6f, 0x44, 0x24, 0x30};
constexpr Bytes6 LoadXmm1 {0xf3, 0x0f, 0x6f, 0x4c, 0x24, 0x40};
constexpr std::array<std::uint8_t, 3> LoadSt0 {0xdb, 0x2c, 0x24};
constexpr std::array<std::uint8_t, 1> Return {0xc3};

constexpr auto TrampolineCall = Join(SubRspBlock, StoreRdi, StoreRsi, StoreRdx, StoreRcx, StoreR8, StoreR9, StoreXmm0, StoreXmm1,
                                     StoreXmm2, StoreXmm3, StoreXmm4, StoreXmm5, StoreXmm6, StoreXmm7, MoveRspToRdi, Syscall);
constexpr auto TrampolineReturn = Join(LoadRax, LoadRdx, LoadXmm0, LoadXmm1, AddRspBlock, Return);
constexpr auto X87TrampolineReturn = Join(LoadSt0, AddRspBlock, Return);

// The fast trampolines call the bridge from JIT code through FEX's thunk instruction (0f 3f and a 32-byte
// name, OpDispatchBuilder::ThunkOp): FEXCore spills its static registers, calls the host function the
// ThunkHandler names with rdi (the block), refills them and returns to the address on the guest stack.
// No dispatcher exit and no block lookups, unlike `syscall`. The function number goes to the block's
// spare slot.
constexpr Bytes7 StoreEaxNumber {0x89, 0x84, 0x24, 0xb0, 0x00, 0x00, 0x00};
constexpr std::size_t NumberSlot = 0xb0;
constexpr std::uint8_t CallRelative = 0xe8;
constexpr std::array<std::uint8_t, 2> ThunkInstruction {0x0f, 0x3f};
constexpr std::array<std::uint8_t, 32> ThunkName {'a', 'p', 's', '5', '-', 'f', 'e', 'x', ' ', 'b', 'r', 'i', 'd', 'g', 'e', ' ',
                                                  'c', 'a', 'l', 'l', ' ', 't', 'h', 'u', 'n', 'k', ' ', 'v', '1', '.', '0', 0};
constexpr auto FastTrampolineCall = Join(SubRspBlock, StoreRdi, StoreRsi, StoreRdx, StoreRcx, StoreR8, StoreR9, StoreXmm0, StoreXmm1,
                                         StoreXmm2, StoreXmm3, StoreXmm4, StoreXmm5, StoreXmm6, StoreXmm7, StoreEaxNumber, MoveRspToRdi);
constexpr std::size_t FastCallBytes = 5;
constexpr auto FastThunk = Join(ThunkInstruction, ThunkName);

// Writes the fast trampoline at code: the saves, `call` to the thunk instruction after the return
// sequence, the return sequence, the thunk instruction. Returns its size.
template <std::size_t ReturnSize>
std::size_t WriteFastTrampoline(std::uint8_t* code, const std::array<std::uint8_t, ReturnSize>& ret) {
    std::size_t at = 0;
    std::memcpy(code, FastTrampolineCall.data(), FastTrampolineCall.size());
    at += FastTrampolineCall.size();
    const auto relative = static_cast<std::int32_t>(ReturnSize);
    code[at] = CallRelative;
    std::memcpy(code + at + 1, &relative, sizeof(relative));
    at += FastCallBytes;
    std::memcpy(code + at, ret.data(), ReturnSize);
    at += ReturnSize;
    std::memcpy(code + at, FastThunk.data(), FastThunk.size());
    return at + FastThunk.size();
}

Bridge* fastBridge = nullptr;

class FastBridgeThunks final : public FEXCore::ThunkHandler {
public:
    FEXCore::ThunkedFunction* LookupThunk(const FEXCore::IR::SHA256Sum& name) override {
        static_assert(sizeof(name) == ThunkName.size());
        if (std::memcmp(&name, ThunkName.data(), ThunkName.size()) != 0) return nullptr;
        return reinterpret_cast<FEXCore::ThunkedFunction*>(&Aps5BridgeThunkEntry);
    }
};

FastBridgeThunks fastThunks;

constexpr std::uint8_t MoveToEax = 0xb8;
constexpr std::uint8_t JumpRelative = 0xe9;

}

Bridge::Bridge(std::filesystem::path libraries) : directory(std::move(libraries)), trace(std::getenv("APS5_FEX_TRACE") != nullptr) {
    void* memory = mmap(nullptr, StubSize(), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (memory == MAP_FAILED) throw std::runtime_error("cannot allocate the import stubs");
    stubs = static_cast<std::uint8_t*>(memory);
    // Stubs are added while guest threads run (StubFor): Call reads entries without the lock, so the
    // table never moves.
    functions.reserve(StubCapacity);
    static_assert(TrampolineOffset + TrampolineCall.size() + TrampolineReturn.size() <= X87TrampolineOffset);
    static_assert(X87TrampolineOffset + TrampolineCall.size() + X87TrampolineReturn.size() <= FirstStubOffset);
    std::memset(stubs, 0xcc, FirstStubOffset);
    stubs[0] = 0x0f;
    stubs[1] = 0x3e;
    // APS5_FEX_BRIDGE_SYSCALL=1 keeps the older path through `syscall` and the dispatcher (for comparison).
    const char* viaSyscall = std::getenv("APS5_FEX_BRIDGE_SYSCALL");
    if (viaSyscall != nullptr && std::strcmp(viaSyscall, "0") != 0) {
        std::memcpy(stubs + TrampolineOffset, TrampolineCall.data(), TrampolineCall.size());
        std::memcpy(stubs + TrampolineOffset + TrampolineCall.size(), TrampolineReturn.data(), TrampolineReturn.size());
        std::memcpy(stubs + X87TrampolineOffset, TrampolineCall.data(), TrampolineCall.size());
        std::memcpy(stubs + X87TrampolineOffset + TrampolineCall.size(), X87TrampolineReturn.data(), X87TrampolineReturn.size());
    } else {
        static_assert(TrampolineOffset + FastTrampolineCall.size() + FastCallBytes + TrampolineReturn.size() + FastThunk.size() <=
                      X87TrampolineOffset);
        static_assert(X87TrampolineOffset + FastTrampolineCall.size() + FastCallBytes + X87TrampolineReturn.size() + FastThunk.size() <=
                      FirstStubOffset);
        WriteFastTrampoline(stubs + TrampolineOffset, TrampolineReturn);
        WriteFastTrampoline(stubs + X87TrampolineOffset, X87TrampolineReturn);
    }
    fastBridge = this;
}

FEXCore::ThunkHandler* Bridge::Thunks() {
    return &fastThunks;
}

// The fast trampolines' host side: block holds the guest's argument registers and the function number,
// frame is FEXCore's state (x28 in JIT code, Aps5BridgeThunkEntry passes it on).
void Bridge::ThunkCall(FEXCore::Core::CpuStateFrame* frame, std::uint64_t block) {
    auto& state = frame->State;
    std::uint32_t number = 0;
    std::memcpy(&number, reinterpret_cast<const void*>(block + NumberSlot), sizeof(number));
    if (number < FirstStub || number - FirstStub >= stubCount.load(std::memory_order_acquire)) {
        std::fprintf(stderr, "[aps5-fex] the bridge got function number %#x\n", number);
        std::abort();
    }
    if (!Invoke(state, functions[number - FirstStub], block)) return;
    // A native function resumed a guest frame. FEXCore continues at the address it pops from the guest
    // stack after the call: put the frame's rip there, below its stack pointer.
    const std::uint64_t rsp = state.gregs[REG_RSP] - 8;
    std::memcpy(reinterpret_cast<void*>(rsp), &state.rip, sizeof(state.rip));
    state.gregs[REG_RSP] = rsp;
}

Bridge::~Bridge() {
    munmap(stubs, StubSize());
}

void Bridge::Open(const std::string& name) {
    const auto path = directory / name;
    std::error_code error;
    const auto canonical = std::filesystem::canonical(path, error);
    if (error) throw std::runtime_error("missing library " + path.string());
    void* handle = dlopen(canonical.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) throw std::runtime_error(std::string("cannot load ") + canonical.string() + ": " + dlerror());
    if (std::find(handles.begin(), handles.end(), handle) != handles.end()) return;
    handles.push_back(handle);
    for (std::uint32_t i = 0; i < _dyld_image_count(); ++i) {
        const char* image = _dyld_get_image_name(i);
        std::error_code ignored;
        if (image != nullptr && std::filesystem::equivalent(image, canonical, ignored))
            images.push_back(reinterpret_cast<const mach_header_64*>(_dyld_get_image_header(i)));
    }
}

void Bridge::OpenIfPresent(const std::string& name) {
    std::error_code error;
    if (std::filesystem::exists(directory / name, error)) Open(name);
}

void* Bridge::HostSymbol(const char* name) const {
    for (void* library : handles)
        if (void* symbol = dlsym(library, name)) return symbol;
    return nullptr;
}

void Bridge::Connect() {
    void* setUnwind = nullptr;
    for (void* library : handles)
        if (setUnwind == nullptr) setUnwind = dlsym(library, "Aps5SetBridgeUnwind_nid_no_patch");
    if (setUnwind == nullptr) throw std::runtime_error("libc has no unwind hooks");
    reinterpret_cast<void (*)(void (*)(std::uintptr_t*), void (*)(const std::uintptr_t*))>(setUnwind)(CaptureGuestFrame, ResumeGuestFrame);
    if (void* setControl = HostSymbol("Aps5SetBridgeControl_nid_no_patch"))
        reinterpret_cast<void (*)(void (*)(std::uint32_t*, std::uint16_t*))>(setControl)(GuestControl);
    if (void* setCall = HostSymbol("Aps5SetBridgeGuestCall_nid_no_patch"))
        reinterpret_cast<void (*)(std::uint64_t (*)(std::uint64_t, const std::uint64_t*))>(setCall)(CallGuestFunction);
    // libkernel's __tls_get_addr export (NID vNe1w4diLCs) resolves through the same module numbering.
    if (void* setTls = HostSymbol("Aps5SetGuestTlsResolver_nid_no_patch"))
        reinterpret_cast<void (*)(void* (*)(const std::uint64_t*))>(setTls)(TlsGetAddr);
    if (void* setFiber = HostSymbol("Aps5SetFiberBridge_nid_no_patch")) {
        using SetFiberBridge = void (*)(void (*)(std::uintptr_t*), void (*)(const std::uintptr_t*), void (*)(std::uint32_t*, std::uint16_t*),
                                        std::uint64_t);
        reinterpret_cast<SetFiberBridge>(setFiber)(CaptureGuestFrame, ResumeGuestFrame, GuestControl,
                                                   AddStub(reinterpret_cast<void*>(&FiberEntryReturned), "fiber entry return"));
    }
    // Displaced guest accesses in JIT code are backpatched through the code pool's writable view (D31).
    if (void* setWriter = HostSymbol("GuestDisplacedSetCodeWriter_nid_no_patch"))
        reinterpret_cast<void (*)(bool (*)(std::uintptr_t, std::uint32_t), std::uintptr_t (*)(std::uintptr_t, std::size_t, std::uintptr_t*))>(setWriter)(
            &CodePool::WriteInstruction, &CodePool::AllocateBeside);
    // ... and forgotten, with their stubs, when FEX frees the code buffer that holds them.
    if (void* codeFreed = HostSymbol("GuestDisplacedCodeFreed_nid_no_patch"))
        CodePool::SetFreeListener(reinterpret_cast<CodePool::FreeListener>(codeFreed));
    if (void* setMachine = HostSymbol("Aps5SetBridgeMachineState_nid_no_patch")) {
        std::uint64_t entry = 0;
        if (void* finish = HostSymbol("Aps5MakecontextFinish_nid_no_patch")) {
            const std::uint64_t finishStub = AddStub(finish, "Aps5MakecontextFinish");
            std::uint8_t* code = stubs + MakecontextOffset;
            std::memcpy(code, MakecontextEntryCode, sizeof(MakecontextEntryCode));
            const auto relative = static_cast<std::int32_t>(static_cast<std::int64_t>(finishStub) -
                                                            reinterpret_cast<std::int64_t>(code + MakecontextFinishCall + 5));
            std::memcpy(code + MakecontextFinishCall + 1, &relative, sizeof(relative));
            entry = reinterpret_cast<std::uint64_t>(code);
        }
        using SetMachine = void (*)(void (*)(MachineState*), void (*)(const MachineState*), std::uint64_t);
        reinterpret_cast<SetMachine>(setMachine)(GetMachineState, SetMachineState, entry);
    }
}

void Bridge::Register(const GuestImage& image, const std::string& name) {
    void* registerImage = nullptr;
    for (void* library : handles)
        if (registerImage == nullptr) registerImage = dlsym(library, "Aps5RegisterGuestImage_nid_no_patch");
    if (registerImage == nullptr) throw std::runtime_error("libc has no guest image hook");
    const ImageInfo info {image.Base(), imageNames.emplace_back(name).c_str(), image.ProgramHeaders(),
                          static_cast<std::uint16_t>(image.ProgramHeaderCount())};
    reinterpret_cast<void (*)(const ImageInfo*)>(registerImage)(&info);
}

void Bridge::SetTlsOffsets(std::vector<std::uint64_t> offsets) {
    tlsOffsets = std::move(offsets);
}

std::uint64_t Bridge::TlsGetAddrStub() {
    if (tlsGetAddrStub == 0) tlsGetAddrStub = AddStub(reinterpret_cast<void*>(&TlsGetAddr), "__tls_get_addr");
    return tlsGetAddrStub;
}

std::string Bridge::StubName(std::uint64_t address) {
    if (!OwnsStub(address) || address < StubBase() + FirstStubOffset) return {};
    std::lock_guard lock(stubMutex);
    const std::size_t index = (address - StubBase() - FirstStubOffset) / StubBytes;
    return index < functions.size() ? functions[index].name : std::string {};
}

std::uint64_t Bridge::AddStub(void* address, const std::string& name, Variadic variadic, std::uint8_t fixed, bool x87Result,
                              const BridgeSignature* signature) {
    std::lock_guard lock(stubMutex);
    if (FirstStubOffset + (functions.size() + 1) * StubBytes > StubSize()) throw std::runtime_error("too many imported functions");
    const auto number = static_cast<std::uint32_t>(FirstStub + functions.size());
    std::uint8_t* stub = stubs + FirstStubOffset + functions.size() * StubBytes;
    const auto target = static_cast<std::int32_t>(x87Result ? X87TrampolineOffset : TrampolineOffset);
    const auto jump = target - static_cast<std::int32_t>(stub + 10 - stubs);
    const std::uint8_t code[] = {MoveToEax, static_cast<std::uint8_t>(number), static_cast<std::uint8_t>(number >> 8), static_cast<std::uint8_t>(number >> 16),
        static_cast<std::uint8_t>(number >> 24), JumpRelative, static_cast<std::uint8_t>(jump), static_cast<std::uint8_t>(jump >> 8),
        static_cast<std::uint8_t>(jump >> 16), static_cast<std::uint8_t>(jump >> 24)};
    std::memset(stub, 0xcc, StubBytes);
    std::memcpy(stub, code, sizeof(code));
    Function function {address, name, variadic, fixed, x87Result, signature, IsShaped(signature)};
    if (signature != nullptr && variadic == Variadic::None) {
        std::size_t stackBytes = 0;
        bool vectors = false;
        for (std::size_t value = 1; value <= signature->parameterCount; ++value)
            for (const BridgePiece& piece : BridgeGuestPieces(*signature, value)) {
                if (piece.place == BridgePlace::Stack) stackBytes = std::max<std::size_t>(stackBytes, piece.at + piece.size);
                vectors |= piece.place == BridgePlace::Vector;
            }
        function.stackSlots = static_cast<std::uint8_t>(std::min<std::size_t>((stackBytes + 7) / 8, 2 + StackArgumentSlots));
        function.vectors = vectors;
    }
    if (signature != nullptr) {
        for (const BridgeValue& parameter : BridgeParameters(*signature))
            function.callbacks |= parameter.kind == BridgeKind::Function && IsShaped(BridgeCallback(parameter));
        function.callbackResult = BridgeResult(*signature).kind == BridgeKind::Function;
    }
    functions.push_back(std::move(function));
    stubCount.store(functions.size(), std::memory_order_release);
    return reinterpret_cast<std::uint64_t>(stub);
}

std::uint64_t Bridge::StubFor(void* address, const std::string& name) {
    Dl_info info;
    if (address == nullptr || dladdr(address, &info) == 0) return 0;
    const auto* image = static_cast<const mach_header_64*>(info.dli_fbase);
    // Only the PS5 libraries: a symbol the system libraries define (dlsym searches a library's
    // dependencies too) has macOS semantics, which the guest never gets.
    if (std::find(images.begin(), images.end(), image) == images.end() &&
        (info.dli_fname == nullptr || !std::string_view(info.dli_fname).ends_with(".prx")))
        return 0;
    std::lock_guard lock(stubMutex);
    if (const auto found = guestAddresses.find(address); found != guestAddresses.end()) return found->second;
    unsigned long textSize = 0;
    const auto* text = getsectiondata(image, "__TEXT", "__text", &textSize);
    const auto* byte = static_cast<const std::uint8_t*>(address);
    const bool function = text != nullptr && byte >= text && byte < text + textSize;
    std::uint64_t guest = reinterpret_cast<std::uint64_t>(address);
    if (function) {
        Variadic variadic = Variadic::None;
        std::uint8_t fixed = 0;
        for (const auto& entry : VariadicExports) {
            if (info.dli_sname != nullptr && entry.name == info.dli_sname && info.dli_saddr == address) {
                variadic = static_cast<Variadic>(entry.kind);
                fixed = entry.fixed;
            }
        }
        if (variadic == Variadic::GuestList && setBridgeVaList == nullptr) {
            for (void* library : handles)
                if (auto* setter = dlsym(library, "Aps5SetBridgeVaList_nid_no_patch")) setBridgeVaList = reinterpret_cast<void (*)(void*)>(setter);
            if (setBridgeVaList == nullptr) throw std::runtime_error("libc has no Aps5SetBridgeVaList_nid_no_patch");
        }
        bool x87Result = info.dli_sname != nullptr && info.dli_saddr == address &&
            std::find(std::begin(X87Results), std::end(X87Results), info.dli_sname) != std::end(X87Results);
        const BridgeSignature* signature = nullptr;
        if (info.dli_sname != nullptr && info.dli_saddr == address) signature = FindBridgeSignature(info.dli_sname);
        // With the generated signatures, every export whose arguments or result positional mapping gets wrong is
        // marshalled from its layouts, the x87 results included (X87Results then only names the same exports).
        if (signature != nullptr) x87Result = GuestX87Result(signature);
        guest = AddStub(address, name, variadic, fixed, x87Result, signature);
    }
    guestAddresses.emplace(address, guest);
    return guest;
}

std::uint64_t Bridge::Resolve(const std::string& name, bool weak) {
    if (const auto found = resolved.find(name); found != resolved.end()) return found->second;
    for (void* handle : handles) {
        const std::uint64_t guest = StubFor(dlsym(handle, name.c_str()), name);
        if (guest == 0) continue;
        resolved.emplace(name, guest);
        return guest;
    }
    if (weak) return 0;
    missing.push_back(name);
    const std::uint64_t guest = AddStub(nullptr, name);
    resolved.emplace(name, guest);
    return guest;
}

namespace {

// Marshalling of the exports whose arguments or result the positional mapping places wrongly, from the
// layouts the signature generator computed for both conventions (BridgeSignatures.hpp).
struct ShapedFrame {
    Aps5NativeArguments arguments {};
    Aps5NativeResult result {};
    alignas(16) std::uint8_t hostStack[256] {};
    alignas(16) std::uint8_t scratch[512] {};
    std::size_t used = 0;

    std::uint8_t* Allocate(std::size_t size) {
        const std::size_t at = (used + 15) & ~std::size_t {15};
        if (at + size > sizeof(scratch)) throw std::runtime_error("a bridged value is too large");
        used = at + size;
        return scratch + at;
    }
};

std::uint8_t* GuestPlace(const BridgePiece& piece, std::uint64_t block) {
    switch (piece.place) {
    case BridgePlace::Gpr: return reinterpret_cast<std::uint8_t*>(block + 8 * piece.reg);
    case BridgePlace::Vector: return reinterpret_cast<std::uint8_t*>(block + BlockVectors + 16 * piece.reg + piece.at);
    case BridgePlace::Stack: return reinterpret_cast<std::uint8_t*>(block + BlockSize + 8 + piece.at);
    case BridgePlace::X87: return reinterpret_cast<std::uint8_t*>(block);
    case BridgePlace::Sret: break;
    }
    throw std::runtime_error("a guest value in no place");
}

std::uint8_t* HostArgumentPlace(const BridgePiece& piece, ShapedFrame& shaped) {
    switch (piece.place) {
    case BridgePlace::Gpr: return reinterpret_cast<std::uint8_t*>(&shaped.arguments.gpr[piece.reg]);
    case BridgePlace::Vector: return shaped.arguments.vector[piece.reg] + piece.at;
    case BridgePlace::Stack:
        if (piece.at + piece.size > sizeof(shaped.hostStack)) throw std::runtime_error("too many stack arguments");
        return shaped.hostStack + piece.at;
    case BridgePlace::X87:
    case BridgePlace::Sret: break;
    }
    throw std::runtime_error("a host argument in no place");
}

std::uint8_t* HostResultPlace(const BridgePiece& piece, ShapedFrame& shaped) {
    switch (piece.place) {
    case BridgePlace::Gpr: return reinterpret_cast<std::uint8_t*>(&shaped.result.gpr[piece.reg]);
    case BridgePlace::Vector: return shaped.result.vector[piece.reg] + piece.at;
    default: break;
    }
    throw std::runtime_error("a host result in no place");
}

void ExtendInteger(const BridgeValue& value, const BridgePiece& piece, std::uint8_t* place) {
    if (value.kind != BridgeKind::Signed || piece.size >= 8 || piece.place != BridgePlace::Gpr) return;
    std::uint64_t word = 0;
    std::memcpy(&word, place, 8);
    const unsigned shift = 64 - 8 * piece.size;
    word = static_cast<std::uint64_t>(static_cast<std::int64_t>(word << shift) >> shift);
    std::memcpy(place, &word, 8);
}

// The arguments in arm64 places; returns the guest's hidden result pointer, or 0.
std::uint64_t MarshalArguments(const BridgeSignature& signature, std::uint64_t block, ShapedFrame& shaped) {
    const auto parameters = BridgeParameters(signature);
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const BridgeValue& value = parameters[index];
        const auto guest = BridgeGuestPieces(signature, index + 1);
        const auto host = BridgeHostPieces(signature, index + 1);
        if (guest.empty()) continue;
        if (guest[0].reference) {
            std::memcpy(HostArgumentPlace(host[0], shaped), GuestPlace(guest[0], block), 8);
            continue;
        }
        const std::uint8_t* bytes = nullptr;
        if (guest.size() == 1 && guest[0].place == BridgePlace::Stack) {
            bytes = GuestPlace(guest[0], block);
        } else {
            std::uint8_t* gathered = shaped.Allocate(std::max<std::size_t>(value.size, 16));
            for (const BridgePiece& piece : guest)
                std::memcpy(gathered + piece.offset, GuestPlace(piece, block), piece.size);
            bytes = gathered;
        }
        if (value.kind == BridgeKind::LongDouble) {
            const double converted = X87ToDouble(bytes);
            std::uint8_t* narrowed = shaped.Allocate(sizeof(converted));
            std::memcpy(narrowed, &converted, sizeof(converted));
            bytes = narrowed;
        }
        for (const BridgePiece& piece : host) {
            std::uint8_t* place = HostArgumentPlace(piece, shaped);
            if (piece.reference) {
                const auto pointer = reinterpret_cast<std::uint64_t>(bytes + piece.offset);
                std::memcpy(place, &pointer, 8);
                continue;
            }
            std::memcpy(place, bytes + piece.offset, piece.size);
            ExtendInteger(value, piece, place);
        }
    }
    std::uint64_t hidden = 0;
    const auto guestResult = BridgeGuestPieces(signature, 0);
    const auto hostResult = BridgeHostPieces(signature, 0);
    if (!guestResult.empty() && guestResult[0].place == BridgePlace::Sret) hidden = reinterpret_cast<const std::uint64_t*>(block)[0];
    if (!hostResult.empty() && hostResult[0].place == BridgePlace::Sret)
        shaped.arguments.x8 = hidden != 0 ? hidden : reinterpret_cast<std::uint64_t>(shaped.Allocate(BridgeResult(signature).size));
    std::uint64_t slots = 0;
    for (std::size_t index = 0; index < parameters.size(); ++index)
        for (const BridgePiece& piece : BridgeHostPieces(signature, index + 1))
            if (piece.place == BridgePlace::Stack) slots = std::max<std::uint64_t>(slots, (piece.at + piece.size + 7) / 8);
    shaped.arguments.stack = reinterpret_cast<const std::uint64_t*>(shaped.hostStack);
    shaped.arguments.stackSlots = slots;
    return hidden;
}

void UnmarshalResult(const BridgeSignature& signature, std::uint64_t block, std::uint64_t hidden, ShapedFrame& shaped) {
    const auto guest = BridgeGuestPieces(signature, 0);
    const auto host = BridgeHostPieces(signature, 0);
    auto* registers = reinterpret_cast<std::uint64_t*>(block);
    if (guest.empty()) return;
    const bool hostHidden = !host.empty() && host[0].place == BridgePlace::Sret;
    if (hidden != 0) {
        if (!hostHidden)
            for (const BridgePiece& piece : host)
                std::memcpy(reinterpret_cast<std::uint8_t*>(hidden) + piece.offset, HostResultPlace(piece, shaped), piece.size);
        registers[0] = hidden;
        return;
    }
    const BridgeValue& value = BridgeResult(signature);
    std::uint8_t* bytes = hostHidden ? reinterpret_cast<std::uint8_t*>(shaped.arguments.x8) : shaped.Allocate(std::max<std::size_t>(value.size, 16));
    if (!hostHidden)
        for (const BridgePiece& piece : host)
            std::memcpy(bytes + piece.offset, HostResultPlace(piece, shaped), piece.size);
    if (guest[0].place == BridgePlace::X87) {
        if (value.kind == BridgeKind::LongDouble) {
            double result = 0;
            std::memcpy(&result, bytes, sizeof(result));
            DoubleToX87(result, GuestPlace(guest[0], block));
        } else {
            std::memcpy(GuestPlace(guest[0], block), bytes, 16);
        }
        return;
    }
    for (const BridgePiece& piece : guest) {
        std::uint8_t* place = GuestPlace(piece, block);
        if (piece.place == BridgePlace::Gpr) std::memset(place, 0, 8);
        std::memcpy(place, bytes + piece.offset, piece.size);
        ExtendInteger(value, piece, place);
    }
}

}

bool Bridge::IsShaped(const BridgeSignature* signature) {
    return signature != nullptr && !signature->variadic && (signature->reasons & ~BridgeReason::Variadic) != 0;
}

bool Bridge::GuestX87Result(const BridgeSignature* signature) {
    if (signature == nullptr) return false;
    const auto guest = BridgeGuestPieces(*signature, 0);
    return !guest.empty() && guest[0].place == BridgePlace::X87;
}

// Runs a native export for a guest call whose registers the import trampoline saved at block. Returns
// true when the export resumed a guest frame instead of returning: state then holds that frame.
bool Bridge::Invoke(FEXCore::Core::CPUState& state, const Function& function, std::uint64_t block) {
    if (function.address == nullptr) {
        std::fprintf(stderr, "[aps5-fex] the guest called %s, which no library exports\n", function.name.c_str());
        std::abort();
    }
    auto* registers = reinterpret_cast<std::uint64_t*>(block);
    if (trace) {
        std::fprintf(stderr, "[aps5-fex] %s(%#llx, %#llx, %#llx, %#llx, %#llx, %#llx)\n", function.name.c_str(),
                     static_cast<unsigned long long>(registers[0]), static_cast<unsigned long long>(registers[1]),
                     static_cast<unsigned long long>(registers[2]), static_cast<unsigned long long>(registers[3]),
                     static_cast<unsigned long long>(registers[4]), static_cast<unsigned long long>(registers[5]));
    }
    auto* guestStack = reinterpret_cast<std::uint64_t*>(block + BlockSize + 8);
    ShapedFrame* shaped = nullptr;
    std::optional<ShapedFrame> shapedStorage;
    Aps5NativeArguments positional;
    Aps5NativeResult positionalResult {};
    std::uint64_t slots[2 + StackArgumentSlots] {};
    VaList list {static_cast<std::uint32_t>(8 * function.fixed), 48, guestStack, registers};
    std::uint64_t variadicSlots[6 + StackArgumentSlots];
    std::uint64_t hidden = 0;
    const Aps5NativeArguments* arguments = &positional;
    Aps5NativeResult* result = &positionalResult;
    if (function.shaped) {
        shaped = &shapedStorage.emplace();
        hidden = MarshalArguments(*function.signature, block, *shaped);
        arguments = &shaped->arguments;
        result = &shaped->result;
    } else {
        // With a signature, only the guest stack slots and vector registers the export reads.
        if (function.stackSlots != 0) std::memcpy(slots, guestStack, ReadableBytes(guestStack, 8 * function.stackSlots));
        const std::uint64_t* stack = slots;
        positional = {
            {registers[0], registers[1], registers[2], registers[3], registers[4], registers[5], stack[0], stack[1]},
            {},
            stack + 2,
            function.stackSlots > 2 ? function.stackSlots - 2u : 0u,
            0,
        };
        if (function.vectors) std::memcpy(positional.vector, reinterpret_cast<const void*>(block + BlockVectors), sizeof(positional.vector));
        switch (function.variadic) {
        case Variadic::None: break;
        case Variadic::GuestList: setBridgeVaList(&list); break;
        case Variadic::Integers:
            std::memcpy(variadicSlots, registers + function.fixed, (6 - function.fixed) * sizeof(std::uint64_t));
            std::memcpy(variadicSlots + 6 - function.fixed, stack, StackArgumentSlots * sizeof(std::uint64_t));
            positional.stack = variadicSlots;
            positional.stackSlots = 6 - function.fixed + StackArgumentSlots;
            break;
        case Variadic::Unsupported:
            std::fprintf(stderr, "[aps5-fex] %s: its variadic arguments cannot be bridged\n", function.name.c_str());
            std::abort();
        }
    }
    if (function.callbacks) {
        // Function pointers whose calls positional mapping gets wrong (floats past the registers, hidden
        // results, long double...): guest functions become thunks that marshal from the callback's shape,
        // import stubs the native functions behind them. Other function pointers stay as they are: a
        // library may hand them back to the guest to run (a fiber's entry becomes its rip), and native
        // calls to them work positionally through the fault path or CallProgram.
        const auto parameters = BridgeParameters(*function.signature);
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            if (parameters[index].kind != BridgeKind::Function || !IsShaped(BridgeCallback(parameters[index]))) continue;
            const auto host = BridgeHostPieces(*function.signature, index + 1);
            if (host.size() != 1 || host[0].size != 8) continue;
            std::uint8_t* place = nullptr;
            if (host[0].place == BridgePlace::Gpr)
                place = reinterpret_cast<std::uint8_t*>(shaped != nullptr ? &shaped->arguments.gpr[host[0].reg] : &positional.gpr[host[0].reg]);
            else if (host[0].place == BridgePlace::Stack && shaped != nullptr)
                place = shaped->hostStack + host[0].at;
            if (place == nullptr) continue;
            std::uint64_t pointer = 0;
            std::memcpy(&pointer, place, 8);
            pointer = NativeCallable(pointer, BridgeCallback(parameters[index]));
            std::memcpy(place, &pointer, 8);
        }
    }
    const GuestCall call {&state, block, currentCall};
    currentCall = &call;
    try {
        Aps5NativeCall(function.address, arguments, result);
    } catch (const GuestFrame& frame) {
        currentCall = call.outer;
        if (function.variadic == Variadic::GuestList) setBridgeVaList(nullptr);
        for (std::size_t i = 0; i < 16; ++i) state.gregs[DwarfRegisters[i]] = frame.registers[i];
        state.rip = frame.registers[16];
        if (const std::uint64_t guest = GuestBehindThunk(state.rip)) state.rip = guest;
        return true;
    } catch (const MachineFrame& frame) {
        currentCall = call.outer;
        if (function.variadic == Variadic::GuestList) setBridgeVaList(nullptr);
        ApplyMachineState(state, frame.state);
        if (const std::uint64_t guest = GuestBehindThunk(state.rip)) state.rip = guest;
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[aps5-fex] %s: %s\n", function.name.c_str(), error.what());
        std::abort();
    }
    currentCall = call.outer;
    if (function.variadic == Variadic::GuestList) setBridgeVaList(nullptr);
    if (shaped != nullptr) {
        try {
            UnmarshalResult(*function.signature, block, hidden, *shaped);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[aps5-fex] %s: %s\n", function.name.c_str(), error.what());
            std::abort();
        }
        if (function.callbackResult) registers[0] = GuestCallable(registers[0]);
        if (trace)
            std::fprintf(stderr, "[aps5-fex] %s returned %#llx:%#llx\n", function.name.c_str(), static_cast<unsigned long long>(registers[0]),
                         static_cast<unsigned long long>(registers[1]));
        return false;
    }
    registers[0] = function.callbackResult ? GuestCallable(result->gpr[0]) : result->gpr[0];
    registers[1] = result->gpr[1];
    std::memcpy(reinterpret_cast<void*>(block + BlockVectors), result->vector, 2 * sizeof(result->vector[0]));
    return false;
}

std::uint64_t Bridge::NativeCallable(std::uint64_t value, const BridgeSignature* callback) {
    if (value == 0) return 0;
    if (OwnsStub(value)) {
        const std::uint64_t offset = value - StubBase();
        if (offset >= FirstStubOffset && (offset - FirstStubOffset) % StubBytes == 0) {
            const std::size_t index = (offset - FirstStubOffset) / StubBytes;
            if (index < stubCount.load(std::memory_order_acquire) && functions[index].address != nullptr)
                return reinterpret_cast<std::uint64_t>(functions[index].address);
        }
        return value;
    }
    if (IsGuestCodeAddress(value)) return GuestThunkFor(value, callback);
    return value;
}

std::uint64_t Bridge::GuestCallable(std::uint64_t value) {
    if (value == 0 || OwnsStub(value) || IsGuestCodeAddress(value)) return value;
    if (const std::uint64_t guest = GuestBehindThunk(value)) return guest;
    if (const std::uint64_t stub = StubFor(reinterpret_cast<void*>(value), "function pointer")) return stub;
    return value;
}

std::uint64_t Bridge::Call(FEXCore::Core::CPUState& state, std::uint64_t number, std::uint64_t block) {
    const std::uint64_t rip = state.rip;
    state.rip = rip + 2;
    if (number < FirstStub || number - FirstStub >= stubCount.load(std::memory_order_acquire)) {
        std::fprintf(stderr, "[aps5-fex] guest syscall %llu at %#llx is not supported\n", static_cast<unsigned long long>(number),
                     static_cast<unsigned long long>(rip));
        return static_cast<std::uint64_t>(-ENOSYS);
    }
    const Function& function = functions[number - FirstStub];
    if (Invoke(state, function, block)) return state.gregs[REG_RAX];
    return reinterpret_cast<const std::uint64_t*>(block)[0];
}

}

extern "C" void Aps5BridgeThunkCall(std::uint64_t block, FEXCore::Core::CpuStateFrame* frame) {
    Aps5Fex::fastBridge->ThunkCall(frame, block);
}
