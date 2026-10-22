#include "prx/libc/include/General.hpp"
#include "prx/libc/include/GuestMachineState.hpp"
#include "prx/libc/include/general/AsmMacros.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" int* APS5_VABI __error_nid_postfix();

struct GuestSignalSet {
    std::uint32_t bits[4];
};
static_assert(sizeof(GuestSignalSet) == 16);

namespace {
constexpr int MaximumSignal = 128;
constexpr int InvalidArgument = 22;

bool ValidSignal(int signal) { return signal > 0 && signal <= MaximumSignal; }
std::size_t WordIndex(int signal) { return static_cast<std::size_t>(signal - 1) >> 5; }
std::uint32_t Bit(int signal) { return 1u << ((signal - 1) & 31); }
int RejectSignal() {
    *__error_nid_postfix() = InvalidArgument;
    return -1;
}
void RequireSet(const GuestSignalSet* set, const char* function) {
    if (set == nullptr) throw std::invalid_argument(std::string(function) + ": set is null");
}
}

extern "C" {
int APS5_VABI sigemptyset_nid_postfix(GuestSignalSet* set) {
    RequireSet(set, "sigemptyset");
    for (auto& word : set->bits) word = 0;
    return 0;
}

int APS5_VABI sigfillset_nid_postfix(GuestSignalSet* set) {
    RequireSet(set, "sigfillset");
    for (auto& word : set->bits) word = ~0u;
    return 0;
}

int APS5_VABI sigaddset_nid_postfix(GuestSignalSet* set, int signal) {
    RequireSet(set, "sigaddset");
    if (!ValidSignal(signal)) return RejectSignal();
    set->bits[WordIndex(signal)] |= Bit(signal);
    return 0;
}

int APS5_VABI sigdelset_nid_postfix(GuestSignalSet* set, int signal) {
    RequireSet(set, "sigdelset");
    if (!ValidSignal(signal)) return RejectSignal();
    set->bits[WordIndex(signal)] &= ~Bit(signal);
    return 0;
}

int APS5_VABI sigismember_nid_postfix(const GuestSignalSet* set, int signal) {
    RequireSet(set, "sigismember");
    if (!ValidSignal(signal)) return RejectSignal();
    return (set->bits[WordIndex(signal)] & Bit(signal)) != 0 ? 1 : 0;
}
}
#if defined(__x86_64__)
#define APS5_SIGNAL_ASM_FUNCTION(name) APS5_ASM_FUNCTION(name)
#if defined(_WIN32) || defined(__APPLE__)
#define APS5_SIGNAL_ASM_CALL(name) "    call " APS5_ASM_SYMBOL(name) "\n"
#else
#define APS5_SIGNAL_ASM_CALL(name) "    call " name "@PLT\n"
#endif

asm(".text\n"
    APS5_SIGNAL_ASM_FUNCTION("sigsetjmp_nid_postfix")
    "    movl %esi, 88(%rdi)\n"
    "    testl %esi, %esi\n"
    "    jz 2f\n"
    "    pushq %rdi\n"
    "    movq %rdi, %rcx\n"
    "    movl $1, %edi\n"
    "    xorl %esi, %esi\n"
    "    leaq 72(%rcx), %rdx\n"
    APS5_SIGNAL_ASM_CALL("_sigprocmask_nid_postfix")
    "    popq %rdi\n"
    "2:\n"
    "    movq %rdi, %rcx\n"
    "    movq 0(%rsp), %rdx\n"
    "    movq %rdx, 0(%rcx)\n"
    "    movq %rbx, 8(%rcx)\n"
    "    movq %rsp, 16(%rcx)\n"
    "    movq %rbp, 24(%rcx)\n"
    "    movq %r12, 32(%rcx)\n"
    "    movq %r13, 40(%rcx)\n"
    "    movq %r14, 48(%rcx)\n"
    "    movq %r15, 56(%rcx)\n"
    "    fnstcw 64(%rcx)\n"
    "    xorl %eax, %eax\n"
    "    ret\n"
    APS5_SIGNAL_ASM_FUNCTION("siglongjmp_nid_postfix")
    "    cmpl $0, 88(%rdi)\n"
    "    jz 2f\n"
    "    movq %rdi, %rdx\n"
    "    pushq %rdi\n"
    "    pushq %rsi\n"
    "    movl $3, %edi\n"
    "    leaq 72(%rdx), %rsi\n"
    "    xorl %edx, %edx\n"
    "    subq $8, %rsp\n"
    APS5_SIGNAL_ASM_CALL("_sigprocmask_nid_postfix")
    "    addq $8, %rsp\n"
    "    popq %rsi\n"
    "    popq %rdi\n"
    "2:\n"
#ifdef _WIN32
    "    pushq %rdi\n"
    "    pushq %rsi\n"
    "    movq 16(%rdi), %rdi\n"
    "    subq $8, %rsp\n"
    "    call GuestStackSwitch_nid_no_patch\n"
    "    addq $8, %rsp\n"
    "    popq %rsi\n"
    "    popq %rdi\n"
#endif
    "    movq %rdi, %rdx\n"
    "    movl %esi, %eax\n"
    "    movq 0(%rdx), %rcx\n"
    "    movq 8(%rdx), %rbx\n"
    "    movq 16(%rdx), %rsp\n"
    "    movq 24(%rdx), %rbp\n"
    "    movq 32(%rdx), %r12\n"
    "    movq 40(%rdx), %r13\n"
    "    movq 48(%rdx), %r14\n"
    "    movq 56(%rdx), %r15\n"
    "    fldcw 64(%rdx)\n"
    "    testl %eax, %eax\n"
    "    jnz 1f\n"
    "    incl %eax\n"
    "1:\n"
    "    movq %rcx, 0(%rsp)\n"
    "    ret\n");
#elif defined(__APPLE__) && defined(__aarch64__)
// The guest's registers live in the arm64 runner: sigsetjmp and siglongjmp reach them through its bridge hooks, which
// libc exports for them (JumpBuffer.cpp), and fill the guest's sigjmp_buf as the x86-64 code above does.
extern "C" void LibcCaptureRegisters_nid_no_patch(std::uintptr_t* registers);
extern "C" [[noreturn]] void LibcRestoreRegisters_nid_no_patch(const std::uintptr_t* registers);
extern "C" void LibcGuestControl_nid_no_patch(std::uint32_t* mxcsr, std::uint16_t* fcw);
extern "C" int APS5_VABI _sigprocmask_nid_postfix(int how, const GuestSignalSet* set, GuestSignalSet* previousSet);

namespace {
// The return address, rbx, the stack pointer at the call (it points at the return address), rbp, r12-r15, the x87
// control word, the signal mask and whether sigsetjmp saved it.
constexpr std::size_t SavedRip = 0;
constexpr std::size_t SavedRbx = 8;
constexpr std::size_t SavedRsp = 16;
constexpr std::size_t SavedRbp = 24;
constexpr std::size_t SavedR12 = 32;
constexpr std::size_t SavedControlWord = 64;
constexpr std::size_t SavedMask = 72;
constexpr std::size_t SavedMaskFlag = 88;
constexpr int GuestSigBlock = 1;
constexpr int GuestSigSetmask = 3;

enum Register { Rax = 0, Rbx = 3, Rbp = 6, Rsp = 7, R12 = 12, Rip = 16, Count = 17 };

template<class T> void Put(void* buffer, std::size_t offset, T value) {
    std::memcpy(static_cast<unsigned char*>(buffer) + offset, &value, sizeof(value));
}

template<class T> T Get(const void* buffer, std::size_t offset) {
    T value;
    std::memcpy(&value, static_cast<const unsigned char*>(buffer) + offset, sizeof(value));
    return value;
}

std::uint16_t GuestControlWord() {
    std::uint32_t mxcsr = 0;
    std::uint16_t fcw = 0;
    LibcGuestControl_nid_no_patch(&mxcsr, &fcw);
    return fcw;
}
}

extern "C" int APS5_VABI sigsetjmp_nid_postfix(void* buffer, int saveMask) {
    Put<std::int32_t>(buffer, SavedMaskFlag, saveMask);
    if (saveMask != 0) {
        GuestSignalSet mask{};
        _sigprocmask_nid_postfix(GuestSigBlock, nullptr, &mask);
        Put(buffer, SavedMask, mask);
    }
    std::uintptr_t registers[Count];
    LibcCaptureRegisters_nid_no_patch(registers);
    // The hooks describe the caller after the return, with the return address popped.
    Put<std::uint64_t>(buffer, SavedRip, registers[Rip]);
    Put<std::uint64_t>(buffer, SavedRbx, registers[Rbx]);
    Put<std::uint64_t>(buffer, SavedRsp, registers[Rsp] - 8);
    Put<std::uint64_t>(buffer, SavedRbp, registers[Rbp]);
    for (std::size_t index = 0; index < 4; ++index) Put<std::uint64_t>(buffer, SavedR12 + 8 * index, registers[R12 + index]);
    Put(buffer, SavedControlWord, GuestControlWord());
    return 0;
}

extern "C" [[noreturn]] void APS5_VABI siglongjmp_nid_postfix(void* buffer, int value) {
    const bool controlChanges = GuestControlWord() != Get<std::uint16_t>(buffer, SavedControlWord);
    if (controlChanges && !LibcHasMachineState_nid_no_patch()) NotImplemented_nid_no_patch("siglongjmp to a different x87 control word");
    if (Get<std::int32_t>(buffer, SavedMaskFlag) != 0) {
        const auto mask = Get<GuestSignalSet>(buffer, SavedMask);
        _sigprocmask_nid_postfix(GuestSigSetmask, &mask, nullptr);
    }
    if (controlChanges) {
        // The x87 control word is not among the registers the unwind hooks restore: resume the whole state.
        GuestMachineState state{};
        LibcGetMachineState_nid_no_patch(&state);
        state.gregs[GuestMachine::Rax] = value != 0 ? static_cast<std::uint32_t>(value) : 1u;
        state.gregs[GuestMachine::Rbx] = Get<std::uint64_t>(buffer, SavedRbx);
        state.gregs[GuestMachine::Rsp] = Get<std::uint64_t>(buffer, SavedRsp) + 8;
        state.gregs[GuestMachine::Rbp] = Get<std::uint64_t>(buffer, SavedRbp);
        for (std::size_t index = 0; index < 4; ++index) state.gregs[GuestMachine::R12 + index] = Get<std::uint64_t>(buffer, SavedR12 + 8 * index);
        state.rip = Get<std::uint64_t>(buffer, SavedRip);
        const auto control = Get<std::uint16_t>(buffer, SavedControlWord);
        std::memcpy(state.fxsave, &control, sizeof(control));
        LibcSetMachineState_nid_no_patch(&state);
    }
    std::uintptr_t registers[Count];
    LibcCaptureRegisters_nid_no_patch(registers);
    registers[Rax] = value != 0 ? static_cast<std::uint32_t>(value) : 1u;
    registers[Rbx] = Get<std::uint64_t>(buffer, SavedRbx);
    registers[Rsp] = Get<std::uint64_t>(buffer, SavedRsp) + 8;
    registers[Rbp] = Get<std::uint64_t>(buffer, SavedRbp);
    for (std::size_t index = 0; index < 4; ++index) registers[R12 + index] = Get<std::uint64_t>(buffer, SavedR12 + 8 * index);
    registers[Rip] = Get<std::uint64_t>(buffer, SavedRip);
    LibcRestoreRegisters_nid_no_patch(registers);
}
#else
extern "C" int APS5_VABI sigsetjmp_nid_postfix(void*, int) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

extern "C" [[noreturn]] void APS5_VABI siglongjmp_nid_postfix(void*, int) {
    NotImplemented_nid_no_patch(__func__);
    __builtin_unreachable();
}
#endif
