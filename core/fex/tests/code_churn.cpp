// Creates and joins 64 guest threads one after another. Run with APS5_FEX_CLEAR_CODE=1 (ctest fex_code_churn), the
// runner drops the code cache every time a thread enters guest code, so FEXCore allocates a new code buffer each
// time (growing to 128 MiB) and frees the old ones: the JIT pool must take them back. Exit 43 when every thread ran.
extern "C" int scePthreadCreate(void** thread, const void* attr, void* (*entry)(void*), void* arg, const char* name);
extern "C" int scePthreadJoin(void* thread, void** value);
extern "C" [[noreturn]] void exit(int);

// Called through a data pointer, so the executable has a RELA section (the relinker wants one).
int (*volatile create)(void**, const void*, void* (*)(void*), void*, const char*) = scePthreadCreate;

namespace {

__attribute__((noinline)) unsigned long Mix(unsigned long value, unsigned long round) {
    value ^= value >> 29;
    value *= 0xbf58476d1ce4e5b9UL + round;
    return value ^ (value >> 32);
}

void* Worker(void* argument) {
    unsigned long value = reinterpret_cast<unsigned long>(argument);
    for (unsigned long round = 0; round < 1000; ++round) value = Mix(value, round);
    return reinterpret_cast<void*>(value | 1);
}

}

extern "C" [[noreturn]] void _start(void*) {
    int ran = 0;
    for (long i = 0; i < 64; ++i) {
        void* thread = nullptr;
        if (create(&thread, nullptr, Worker, reinterpret_cast<void*>(i + 1), "churn") != 0) exit(1);
        void* result = nullptr;
        if (scePthreadJoin(thread, &result) != 0) exit(2);
        if ((reinterpret_cast<unsigned long>(result) & 1) != 0) ++ran;
    }
    exit(ran == 64 ? 43 : 3);
}
