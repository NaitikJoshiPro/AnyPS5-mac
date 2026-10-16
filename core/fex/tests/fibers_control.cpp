// Each fiber and the thread keep their own MXCSR and x87 control word across sceFiberRun, sceFiberSwitch and
// sceFiberReturnToThread (both are callee-saved in the SysV ABI; the x86-64 stack switch reloads them). A new
// fiber starts with the control words of whoever first runs it. The rounding direction is checked through a
// division as well as by reading the registers back. Exit 43 when every check holds; otherwise the bits of
// the failing checks.
extern "C" int _sceFiberInitializeImpl(void* fiber, const char* name, void (*entry)(unsigned long, unsigned long), unsigned long argument,
                                       void* context, unsigned long size, const void* option, unsigned version);
extern "C" int sceFiberRun(void* fiber, unsigned long argument, unsigned long* returned);
extern "C" int sceFiberSwitch(void* fiber, unsigned long argument, unsigned long* resumed);
extern "C" int sceFiberReturnToThread(unsigned long argument, unsigned long* resumed);
extern "C" [[noreturn]] void exit(int);

alignas(16) static unsigned char first[256];
alignas(16) static unsigned char second[256];
alignas(16) static unsigned char firstStack[16384];
alignas(16) static unsigned char secondStack[16384];
static int failures = 0;

constexpr unsigned Nearest = 0x1F80;
constexpr unsigned Down = 0x3F80;
constexpr unsigned Up = 0x5F80;
constexpr unsigned TowardZero = 0x7F80;
constexpr unsigned short X87Default = 0x037F;
constexpr unsigned short X87Truncate = 0x0F7F;
constexpr unsigned short X87Single = 0x007F;

static unsigned GetMxcsr() {
    unsigned value = 0;
    asm volatile("stmxcsr %0" : "=m"(value));
    return value;
}

static void SetMxcsr(unsigned value) {
    asm volatile("ldmxcsr %0" : : "m"(value));
}

static unsigned short GetFcw() {
    unsigned short value = 0;
    asm volatile("fnstcw %0" : "=m"(value));
    return value;
}

static void SetFcw(unsigned short value) {
    asm volatile("fldcw %0" : : "m"(value));
}

static volatile float numerator = 1.0f;
static volatile float denominator = 3.0f;

// The bits of 1/3 in single precision: ...AB rounded to nearest or up, ...AA rounded down or toward zero.
static unsigned Third() {
    const float value = numerator / denominator;
    unsigned bits = 0;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void runFirst(unsigned long, unsigned long) {
    if (GetMxcsr() != Nearest || GetFcw() != X87Default) failures |= 1;
    SetMxcsr(Down);
    SetFcw(X87Truncate);
    unsigned long resumed = 0;
    if (sceFiberReturnToThread(1, &resumed) != 0 || resumed != 2) failures |= 2;
    if (GetMxcsr() != Down) failures |= 4;
    if (GetFcw() != X87Truncate) failures |= 4096;
    if (Third() != 0x3EAAAAAA) failures |= 8192;
    if (sceFiberSwitch(second, 3, &resumed) != 0 || resumed != 4) failures |= 8;
    if (GetMxcsr() != Down || GetFcw() != X87Truncate || Third() != 0x3EAAAAAA) failures |= 16;
    sceFiberReturnToThread(5, &resumed);
    failures |= 32;
}

static void runSecond(unsigned long, unsigned long) {
    if (GetMxcsr() != Down || GetFcw() != X87Truncate) failures |= 64;
    SetMxcsr(TowardZero);
    SetFcw(X87Single);
    unsigned long resumed = 0;
    sceFiberSwitch(first, 4, &resumed);
    failures |= 128;
}

int (*volatile runPointer)(void*, unsigned long, unsigned long*) = sceFiberRun;

extern "C" [[noreturn]] void _start(void*) {
    SetMxcsr(Nearest);
    SetFcw(X87Default);
    if (_sceFiberInitializeImpl(first, "first", runFirst, 0, firstStack, sizeof(firstStack), nullptr, 0) != 0 ||
        _sceFiberInitializeImpl(second, "second", runSecond, 0, secondStack, sizeof(secondStack), nullptr, 0) != 0) exit(100);
    unsigned long returned = 0;
    if (runPointer(first, 0, &returned) != 0 || returned != 1) failures |= 256;
    if (GetMxcsr() != Nearest || GetFcw() != X87Default || Third() != 0x3EAAAAAB) failures |= 512;
    SetMxcsr(Up);
    if (runPointer(first, 2, &returned) != 0 || returned != 5) failures |= 1024;
    if (GetMxcsr() != Up || GetFcw() != X87Default || Third() != 0x3EAAAAAB) failures |= 2048;
    exit(failures == 0 ? 43 : failures);
}
