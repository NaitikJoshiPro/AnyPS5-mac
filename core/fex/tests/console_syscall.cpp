extern "C" [[noreturn]] void exit(int);

void (*volatile exitPointer)(int) = exit;

struct Registers {
    unsigned long rax, rdi, rsi, rdx, r8, r9, r10, rbx, r12, carry, xmm0, xmm1;
};

extern "C" void ConsoleCall(Registers* out, unsigned long number, unsigned long first, unsigned long second);

asm(R"(
    .text
    .globl ConsoleCall
ConsoleCall:
    pushq %rbx
    pushq %r12
    pushq %r13
    movq %rdi, %r13
    movq %rsi, %rax
    movq %rdx, %rdi
    movq %rcx, %rsi
    movq $0x5555, %rdx
    movq $0x1111, %r8
    movq $0x2222, %r9
    movq $0x3333, %r10
    movq $0x4444, %rbx
    movq $0x6666, %r12
    movabsq $0x3ff0000000000000, %r11
    movq %r11, %xmm0
    movabsq $0x4000000000000000, %r11
    movq %r11, %xmm1
    movabsq $0x8000004ea, %r11
    callq *%r11
    setc %cl
    movzbq %cl, %rcx
    movq %rcx, 72(%r13)
    movq %rax, 0(%r13)
    movq %rdi, 8(%r13)
    movq %rsi, 16(%r13)
    movq %rdx, 24(%r13)
    movq %r8, 32(%r13)
    movq %r9, 40(%r13)
    movq %r10, 48(%r13)
    movq %rbx, 56(%r13)
    movq %r12, 64(%r13)
    movq %xmm0, 80(%r13)
    movq %xmm1, 88(%r13)
    popq %r13
    popq %r12
    popq %rbx
    retq
)");

static bool Preserved(const Registers& r, unsigned long first, unsigned long second) {
    return r.rdi == first && r.rsi == second && r.rdx == 0x5555 && r.r8 == 0x1111 && r.r9 == 0x2222 && r.r10 == 0x3333 &&
           r.rbx == 0x4444 && r.r12 == 0x6666 && r.xmm0 == 0x3ff0000000000000ul && r.xmm1 == 0x4000000000000000ul;
}

extern "C" [[noreturn]] void _start(void*) {
    Registers pid {};
    ConsoleCall(&pid, 20, 7, 8);
    Registers bad {};
    ConsoleCall(&bad, 6, 0x7fffff, 9);
    int status = 43;
    if (!(pid.carry == 0 && static_cast<long>(pid.rax) > 0 && Preserved(pid, 7, 8))) status = 1;
    if (!(bad.carry == 1 && bad.rax == 9 && Preserved(bad, 0x7fffff, 9))) status = status == 43 ? 2 : 3;
    exitPointer(status);
    __builtin_unreachable();
}
