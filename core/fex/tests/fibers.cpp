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
static unsigned long order = 0;
static int failures = 0;

static void runFirst(unsigned long initial, unsigned long run) {
    if (initial != 1 || run != 5) failures |= 1;
    order = order * 10 + 1;
    unsigned long back = 0;
    if (sceFiberSwitch(second, 7, &back) != 0 || back != 9) failures |= 2;
    order = order * 10 + 3;
    unsigned long again = 0;
    if (sceFiberReturnToThread(back + 30, &again) != 0 || again != 11) failures |= 4;
    order = order * 10 + 4;
    sceFiberSwitch(second, 13, &back);
    failures |= 8;
}

static void runSecond(unsigned long initial, unsigned long run) {
    if (initial != 2 || run != 7) failures |= 16;
    order = order * 10 + 2;
    unsigned long back = 0;
    if (sceFiberSwitch(first, 9, &back) != 0 || back != 13) failures |= 32;
    order = order * 10 + 5;
    sceFiberReturnToThread(back, &back);
    failures |= 64;
}

int (*volatile runPointer)(void*, unsigned long, unsigned long*) = sceFiberRun;

extern "C" [[noreturn]] void _start(void*) {
    if (_sceFiberInitializeImpl(first, "first", runFirst, 1, firstStack, sizeof(firstStack), nullptr, 0) != 0 ||
        _sceFiberInitializeImpl(second, "second", runSecond, 2, secondStack, sizeof(secondStack), nullptr, 0) != 0) exit(100);
    unsigned long returned = 0;
    if (runPointer(first, 5, &returned) != 0 || returned != 39) failures |= 128;
    if (runPointer(first, 11, &returned) != 0 || returned != 13) failures |= 256;
    exit(order == 12345 && failures == 0 ? 43 : (failures != 0 ? failures : 200));
}
