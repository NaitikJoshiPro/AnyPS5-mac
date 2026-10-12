// getcontext/setcontext/makecontext through the arm64 runner's machine state hooks: round trips with
// the general registers, xmm registers and MXCSR, a swapcontext built from getcontext and setcontext
// between the main stack and a made context, and siglongjmp to another x87 control word.
// "context-exit" runs the made context without a link: returning from it exits with status 0.
#include "context.h"

extern "C" {
int getcontext(Ucontext*);
int setcontext(const Ucontext*);
void makecontext(Ucontext*, void (*)(), int, ...);
int sigsetjmp(void*, int);
[[noreturn]] void siglongjmp(void*, int);
}

static int Swap(Ucontext* from, const Ucontext* to) {
    volatile int swapped = 0;
    if (getcontext(from) != 0) return -1;
    if (!swapped) {
        swapped = 1;
        setcontext(to);
        return -2;
    }
    return 0;
}

static Ucontext mainContext, coroutine, entryContext;
alignas(16) static unsigned char coroutineStack[64 * 1024];
static volatile u64 sum = 0;
static volatile int finished = 0;

static void Coroutine(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    sum = a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f;
    Swap(&coroutine, &mainContext);
    sum = sum + 1000;
    finished = 1;
}

static void RoundTrip() {
    static Ucontext context;
    volatile int passes = 0;
    alignas(16) static const u8 pattern[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    asm volatile("movdqa %0, %%xmm8" : : "m"(pattern) : "xmm8");
    Check(getcontext(&context) == 0, 0);
    u8 xmm15[16];
    asm volatile("movdqu %%xmm15, %0" : "=m"(xmm15));
    u32 mxcsr = 0;
    asm volatile("stmxcsr %0" : "=m"(mxcsr));
    ++passes;
    if (passes == 1) {
        Check(context.mcontext.len == sizeof(Mcontext) && context.mcontext.rip != 0, 1);
        Check(context.mcontext.rsp != 0 && (context.mcontext.rsp & 15) == 0, 1);
        const u8* saved = reinterpret_cast<const u8*>(context.mcontext.fpstate);
        bool same = true;
        for (int index = 0; index < 16; ++index) same = same && saved[160 + 8 * 16 + index] == pattern[index];
        Check(same, 2);
        // Resume with xmm15 = 0x5a... and round-down MXCSR.
        u8* image = reinterpret_cast<u8*>(context.mcontext.fpstate);
        for (int index = 0; index < 16; ++index) image[160 + 15 * 16 + index] = 0x5a;
        u32 changed = 0x1f80 | 0x2000;
        __builtin_memcpy(image + 24, &changed, 4);
        setcontext(&context);
        Check(false, 3);
    }
    Check(passes == 2, 3);
    bool restored = true;
    for (u8 byte : xmm15) restored = restored && byte == 0x5a;
    Check(restored, 4);
    Check((mxcsr & 0x6000) == 0x2000, 5);
    u32 standard = 0x1f80;
    asm volatile("ldmxcsr %0" : : "m"(standard));
}

static void Coroutines() {
    Check(getcontext(&coroutine) == 0, 6);
    coroutine.stackPointer = coroutineStack;
    coroutine.stackSize = sizeof(coroutineStack);
    coroutine.link = &entryContext;
    makecontext(&coroutine, reinterpret_cast<void (*)()>(Coroutine), 6, 1ull, 2ull, 3ull, 4ull, 5ull, 6ull);
    Check(coroutine.mcontext.len == sizeof(Mcontext), 6);
    Check(Swap(&mainContext, &coroutine) == 0 && sum == 1 + 4 + 9 + 16 + 25 + 36 && !finished, 7);
    // Back in the coroutine; when it returns, its link resumes entryContext, saved here.
    Check(Swap(&entryContext, &coroutine) == 0 && finished && sum == 1091, 8);
}

static void ExitWithoutLink() {
    Check(getcontext(&coroutine) == 0, 9);
    coroutine.stackPointer = coroutineStack;
    coroutine.stackSize = sizeof(coroutineStack);
    coroutine.link = nullptr;
    makecontext(&coroutine, reinterpret_cast<void (*)()>(Coroutine), 0);
    // Coroutine swaps back once, then returns: without a link the process exits with status 0.
    Swap(&mainContext, &coroutine);
    setcontext(&coroutine);
}

static u16 ControlWord() {
    u16 word = 0;
    asm volatile("fnstcw %0" : "=m"(word));
    return word;
}

static void JumpWithControlWord() {
    alignas(16) static u64 buffer[32];
    const u16 original = ControlWord();
    volatile int jumped = 0;
    const int value = sigsetjmp(buffer, 1);
    if (value == 0) {
        const u16 single = static_cast<u16>((original & ~0x300u) | 0x000u);
        asm volatile("fldcw %0" : : "m"(single));
        jumped = 1;
        siglongjmp(buffer, 5);
    }
    Check(value == 5 && jumped == 1 && ControlWord() == original, 10);
}

extern "C" [[noreturn]] void _start(void*) {
#ifdef CONTEXT_EXIT
    ExitWithoutLink();
#endif
    RoundTrip();
    Coroutines();
    JumpWithControlWord();
    Finish();
}
