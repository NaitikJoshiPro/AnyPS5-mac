// Shared by context.cpp and raise.cpp: FreeBSD's ucontext_t and mcontext_t as the PS5 has them
// (core/libs/prx/libkernel/System/include/UserContext.hpp).
#ifndef CORE_FEX_TESTS_CONTEXT_H
#define CORE_FEX_TESTS_CONTEXT_H

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

struct Mcontext {
    u64 onstack, rdi, rsi, rdx, rcx, r8, r9, rax, rbx, rbp, r10, r11, r12, r13, r14, r15;
    u32 trapno;
    u16 fs, gs;
    u64 addr;
    u32 flags;
    u16 es, ds;
    u64 err, rip, cs, rflags, rsp, ss, len, fpformat, ownedfp;
    alignas(16) u64 fpstate[64];
    u64 fsbase, gsbase, xfpustate, xfpustateLength, spare[4];
};

struct Ucontext {
    u32 sigmask[4];
    Mcontext mcontext;
    Ucontext* link;
    void* stackPointer;
    u64 stackSize;
    int stackFlags, stackAlign, flags, spare[4];
};

static_assert(__builtin_offsetof(Mcontext, rsp) == 0xb8 && __builtin_offsetof(Mcontext, len) == 0xc8, "the FreeBSD mcontext layout");

extern "C" [[noreturn]] void exit(int);

static void (*volatile exitPointer)(int) = exit;
static int failures = 0;
static int firstFailure = -1;

// The exit status is 43, or 100 plus the number of the first check that failed.
static void Check(bool right, int bit) {
    if (right) return;
    failures |= 1 << bit;
    if (firstFailure < 0) firstFailure = bit;
}

[[noreturn]] static void Finish() {
    exitPointer(failures == 0 ? 43 : 100 + firstFailure);
    __builtin_unreachable();
}

#endif
