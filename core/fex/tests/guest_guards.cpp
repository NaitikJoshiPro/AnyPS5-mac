// An executable that defines its own static-initialisation guards, as homebrew linking libc++abi built
// without threads does (oops-apps' libc++abi: _LIBCXXABI_HAS_NO_THREADS, cxa_guard.cpp's InitByteNoThreads):
// acquire aborts as "recursive initialization" whenever the guard is pending, so a second thread that arrives
// while the first runs the initialiser kills the program. aps5-fex redirects an image's own __cxa_guard_* to
// libc's thread-safe guards (GuestGuards.cpp): the second thread must wait and then see the initialised value.
// Linked as a shared object (fex_toolchain.link_tpoff_executable) so the guards are exported, as SuperTuxKart's
// are. Exit 43 when every check passes; 5 when the no-threads guard saw a pending guard; otherwise the failing
// checks' bits.
extern "C" int scePthreadCreate(void** thread, const void* attr, void* (*entry)(void*), void* arg, const char* name);
extern "C" int scePthreadJoin(void* thread, void** value);
extern "C" [[noreturn]] void exit(int);

// Called through data pointers, so the executable has a RELA section (the relinker wants one). Hidden, as the
// shared-object link would otherwise leave the executable importing its own variables.
__attribute__((visibility("hidden"))) int (*volatile create)(void**, const void*, void* (*)(void*), void*, const char*) = scePthreadCreate;
__attribute__((visibility("hidden"))) void (*volatile quit)(int) = exit;

// libc++abi's no-threads guards: byte 0 is "initialised", byte 1 is "pending". Hidden definitions with exported
// aliases: the program calls the definitions at the exported addresses (as a statically linked libc++abi's callers
// do), never through a symbol the loader could bind to libc.
extern "C" __attribute__((noinline, visibility("hidden"))) int GuardAcquire(unsigned long long* guard) {
    auto* bytes = reinterpret_cast<volatile unsigned char*>(guard);
    if (bytes[0] != 0) return 0;
    if (bytes[1] != 0) quit(5);
    bytes[1] = 1;
    return 1;
}

extern "C" __attribute__((noinline, visibility("hidden"))) void GuardRelease(unsigned long long* guard) {
    auto* bytes = reinterpret_cast<volatile unsigned char*>(guard);
    bytes[0] = 1;
    bytes[1] = 0;
}

extern "C" __attribute__((noinline, visibility("hidden"))) void GuardAbort(unsigned long long* guard) {
    reinterpret_cast<volatile unsigned char*>(guard)[1] = 0;
}

extern "C" int __cxa_guard_acquire(unsigned long long*) __attribute__((alias("GuardAcquire")));
extern "C" void __cxa_guard_release(unsigned long long*) __attribute__((alias("GuardRelease")));
extern "C" void __cxa_guard_abort(unsigned long long*) __attribute__((alias("GuardAbort")));

// Called through pointers so the compiler cannot inline the guards.
__attribute__((visibility("hidden"))) int (*volatile acquire)(unsigned long long*) = GuardAcquire;
__attribute__((visibility("hidden"))) void (*volatile release)(unsigned long long*) = GuardRelease;
__attribute__((visibility("hidden"))) void (*volatile abandon)(unsigned long long*) = GuardAbort;

namespace {

unsigned long long guard;
unsigned long long abortedGuard;
volatile int value;
volatile int holding;
volatile int calling;
volatile int secondResult = -1;
volatile int secondValue = -1;

void Spin(long iterations) {
    for (volatile long i = 0; i < iterations; ++i) {
    }
}

void* First(void*) {
    if (acquire(&guard) != 1) return reinterpret_cast<void*>(1);
    holding = 1;
    while (calling == 0) {
    }
    // The second thread is about to call acquire; run the "initialiser" long enough for it to arrive.
    Spin(20000000);
    value = 42;
    release(&guard);
    return nullptr;
}

void* Second(void*) {
    while (holding == 0) {
    }
    calling = 1;
    secondResult = acquire(&guard);
    secondValue = value;
    return nullptr;
}

}

extern "C" [[noreturn]] void _start(void*) {
    int failures = 0;
    // A guard abandoned by an initialiser that threw is acquired again.
    if (acquire(&abortedGuard) != 1) failures |= 1;
    abandon(&abortedGuard);
    if (acquire(&abortedGuard) != 1) failures |= 2;
    release(&abortedGuard);
    if (acquire(&abortedGuard) != 0) failures |= 4;

    void* first = nullptr;
    void* second = nullptr;
    void* firstResult = nullptr;
    if (create(&first, nullptr, First, nullptr, "guard-first") != 0) failures |= 8;
    if (create(&second, nullptr, Second, nullptr, "guard-second") != 0) failures |= 8;
    scePthreadJoin(first, &firstResult);
    scePthreadJoin(second, nullptr);
    if (firstResult != nullptr) failures |= 16;
    if (secondResult != 0 || secondValue != 42) failures |= 32;
    quit(failures == 0 ? 43 : failures);
    __builtin_unreachable();
}
