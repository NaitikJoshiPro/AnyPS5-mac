#include "prx/libc/include/General.hpp"
#include "prx/libc/include/general/AsmMacros.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__)
// setjmp/longjmp must capture the guest's own frame, so they are written directly in assembly with
// the guest (System V) calling convention. The saved state fits the guest's 96-byte jmp_buf:
// return address, rbx, rsp, rbp, r12-r15, MXCSR and the x87 control word.

asm(".text\n"
    APS5_ASM_FUNCTION("setjmp_nid_postfix")
    "    mov (%rsp), %rax\n"
    "    mov %rax, 0(%rdi)\n"
    "    mov %rbx, 8(%rdi)\n"
    "    lea 8(%rsp), %rax\n"
    "    mov %rax, 16(%rdi)\n"
    "    mov %rbp, 24(%rdi)\n"
    "    mov %r12, 32(%rdi)\n"
    "    mov %r13, 40(%rdi)\n"
    "    mov %r14, 48(%rdi)\n"
    "    mov %r15, 56(%rdi)\n"
    "    stmxcsr 64(%rdi)\n"
    "    fnstcw 68(%rdi)\n"
    "    xor %eax, %eax\n"
    "    ret\n"
    APS5_ASM_FUNCTION("longjmp_nid_postfix")
#ifdef _WIN32
    "    mov %rdi, %rbx\n"
    "    mov %esi, %r12d\n"
    "    mov 16(%rdi), %rdi\n"
    "    and $-16, %rsp\n"
    "    call GuestStackSwitch_nid_no_patch\n"
    "    mov %rbx, %rdi\n"
    "    mov %r12d, %esi\n"
#endif
    "    mov %esi, %eax\n"
    "    test %eax, %eax\n"
    "    jnz 1f\n"
    "    inc %eax\n"
    "1:\n"
    "    mov 8(%rdi), %rbx\n"
    "    mov 16(%rdi), %rsp\n"
    "    mov 24(%rdi), %rbp\n"
    "    mov 32(%rdi), %r12\n"
    "    mov 40(%rdi), %r13\n"
    "    mov 48(%rdi), %r14\n"
    "    mov 56(%rdi), %r15\n"
    "    ldmxcsr 64(%rdi)\n"
    "    fldcw 68(%rdi)\n"
    "    jmp *0(%rdi)\n");
#elif defined(__APPLE__) && defined(__aarch64__)
extern "C" void LibcCaptureRegisters(std::uintptr_t* registers);
extern "C" [[noreturn]] void LibcRestoreRegisters(const std::uintptr_t* registers);
extern "C" void LibcGuestControl(std::uint32_t* mxcsr, std::uint16_t* fcw);

namespace {

struct JumpBuffer {
    std::uint64_t rip, rbx, rsp, rbp, r12, r13, r14, r15;
    std::uint32_t mxcsr;
    std::uint16_t fcw;
};
static_assert(offsetof(JumpBuffer, mxcsr) == 64 && offsetof(JumpBuffer, fcw) == 68);
constexpr std::size_t JumpBufferBytes = 70;

enum Register { Rax = 0, Rbx = 3, Rbp = 6, Rsp = 7, R12 = 12, R13 = 13, R14 = 14, R15 = 15, Rip = 16, Count = 17 };

}

extern "C" int APS5_VABI setjmp_nid_postfix(void* buffer) {
    std::uintptr_t registers[Count];
    LibcCaptureRegisters(registers);
    JumpBuffer saved {registers[Rip], registers[Rbx], registers[Rsp], registers[Rbp], registers[R12], registers[R13], registers[R14],
                      registers[R15], 0, 0};
    LibcGuestControl(&saved.mxcsr, &saved.fcw);
    std::memcpy(buffer, &saved, JumpBufferBytes);
    return 0;
}

extern "C" void APS5_VABI longjmp_nid_postfix(void* buffer, int value) {
    JumpBuffer saved {};
    std::memcpy(&saved, buffer, JumpBufferBytes);
    std::uint32_t mxcsr = 0;
    std::uint16_t fcw = 0;
    LibcGuestControl(&mxcsr, &fcw);
    if (mxcsr != saved.mxcsr || fcw != saved.fcw) NotImplemented_nid_no_patch("longjmp to a different MXCSR or x87 control word");
    std::uintptr_t registers[Count];
    LibcCaptureRegisters(registers);
    registers[Rax] = value != 0 ? static_cast<std::uint32_t>(value) : 1u;
    registers[Rbx] = saved.rbx;
    registers[Rsp] = saved.rsp;
    registers[Rbp] = saved.rbp;
    registers[R12] = saved.r12;
    registers[R13] = saved.r13;
    registers[R14] = saved.r14;
    registers[R15] = saved.r15;
    registers[Rip] = saved.rip;
    LibcRestoreRegisters(registers);
}

// The same hooks for the other libraries (libkernel's sigsetjmp and siglongjmp): a patched library exports only NIDs
// and _nid_no_patch names, so they cannot import the hooks under their own names.
extern "C" void LibcCaptureRegisters_nid_no_patch(std::uintptr_t* registers) {
    LibcCaptureRegisters(registers);
}

extern "C" [[noreturn]] void LibcRestoreRegisters_nid_no_patch(const std::uintptr_t* registers) {
    LibcRestoreRegisters(registers);
}

extern "C" void LibcGuestControl_nid_no_patch(std::uint32_t* mxcsr, std::uint16_t* fcw) {
    LibcGuestControl(mxcsr, fcw);
}
#else
extern "C" int APS5_VABI setjmp_nid_postfix(void* buffer) {
    (void)buffer;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

extern "C" void APS5_VABI longjmp_nid_postfix(void* buffer, int value) {
    (void)buffer;
    (void)value;
    NotImplemented_nid_no_patch(__func__);
    __builtin_unreachable();
}
#endif
