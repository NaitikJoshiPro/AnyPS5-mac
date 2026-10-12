// sceKernelRaiseException on the raising thread: the handler gets the guest's registers at the call
// (callee-saved values the caller set, rip after the call, rax 0) and a context it changes resumes the
// guest. Raising without a handler is an error the call reports.
#include "context.h"

// The exception handler's ucontext (libkernel Exception.cpp's GuestUcontext): unlike getcontext's, the
// machine context starts at 0x40 and carries four more words before its 832-byte FPU area.
struct ExceptionMcontext {
    u64 onstack, rdi, rsi, rdx, rcx, r8, r9, rax, rbx, rbp, r10, r11, r12, r13, r14, r15;
    u32 trapno;
    u16 fs, gs;
    u64 addr;
    u32 flags;
    u16 es, ds;
    u64 err, rip, cs, rflags, rsp, ss, len, fpformat, ownedfp, lbrfrom, lbrto, aux1, aux2;
    u64 fpstate[104];
    u64 fsbase, gsbase, spare[6];
};

struct ExceptionUcontext {
    u32 sigmask[4];
    int reserved[12];
    ExceptionMcontext mcontext;
};

static_assert(__builtin_offsetof(ExceptionUcontext, mcontext) + __builtin_offsetof(ExceptionMcontext, rsp) == 0xf8, "the handler's layout");

extern "C" {
int sceKernelInstallExceptionHandler(int signum, void (*handler)(int, void*));
int sceKernelRemoveExceptionHandler(int signum);
int sceKernelRaiseException(void* thread, int signum);
void* scePthreadSelf();
}

constexpr int RaisedSignal = 30;
static volatile int handled = 0;
static volatile u64 seenRip = 0;

static void Handler(int signum, void* context) {
    auto* ucontext = static_cast<ExceptionUcontext*>(context);
    auto& machine = ucontext->mcontext;
    ++handled;
    Check(signum == RaisedSignal, 0);
    Check(machine.rbx == 0x1111111111111111ull && machine.r12 == 0x2222222222222222ull && machine.r15 == 0x5555555555555555ull, 1);
    Check(machine.rax == 0 && machine.rip != 0 && machine.rsp != 0 && machine.len == sizeof(ExceptionMcontext), 2);
    seenRip = machine.rip;
    if (handled == 2) machine.r12 = 0x3333333333333333ull;
}

// Calls sceKernelRaiseException with rbx, r12 and r15 set; returns r12 after the call and the result.
static void* self;
static int (*raiseFunction)(void*, int);
static u64 r12After, returnAddress;
static int returned;

static u64 RaiseWithRegisters(int* result) {
    self = scePthreadSelf();
    raiseFunction = sceKernelRaiseException;
    asm volatile(
        "mov %[self], %%rdi\n"
        "mov %[raise], %%rax\n"
        "mov %%rsp, %%r13\n"
        "sub $128, %%rsp\n"
        "and $-16, %%rsp\n"
        "mov $30, %%esi\n"
        "movabs $0x1111111111111111, %%rbx\n"
        "movabs $0x2222222222222222, %%r12\n"
        "movabs $0x5555555555555555, %%r15\n"
        "call *%%rax\n"
        "1:\n"
        "mov %%r13, %%rsp\n"
        "lea 1b(%%rip), %%rcx\n"
        "mov %%rcx, %[after]\n"
        "mov %%r12, %[r12]\n"
        "mov %%eax, %[returned]\n"
        : [r12] "=m"(r12After), [returned] "=m"(returned), [after] "=m"(returnAddress)
        : [self] "m"(self), [raise] "m"(raiseFunction)
        : "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r15", "memory", "cc", "xmm0", "xmm1",
          "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
    *result = returned;
    Check(seenRip == returnAddress, 3);
    return r12After;
}

extern "C" [[noreturn]] void _start(void*) {
    Check(sceKernelInstallExceptionHandler(RaisedSignal, Handler) == 0, 4);
    int result = -1;
    // An unchanged context: the call returns 0 and the registers stay.
    Check(RaiseWithRegisters(&result) == 0x2222222222222222ull && result == 0 && handled == 1, 5);
    // The handler changes r12: the guest resumes with it.
    Check(RaiseWithRegisters(&result) == 0x3333333333333333ull && result == 0 && handled == 2, 6);
    Check(sceKernelRemoveExceptionHandler(RaisedSignal) == 0, 7);
    Finish();
}
