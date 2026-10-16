// Displaced-access sites in code FEX frees (D31, D33): 64 guest threads one after another, each loading and storing
// displaced memory through its own translated sites. Run with APS5_FEX_CLEAR_CODE=1 (as fex_code_churn), the runner
// drops the code cache every time a thread enters guest code, so each thread's sites are patched in a new code buffer
// and the old buffers are freed; libc forgets their sites and frees their stubs (GuestDisplacedCodeFreed), and the
// slots are reused for the next buffer's sites. A stale site or a stub reused while still reachable shows up as a
// wrong value or a crash. Exit 43 when every thread computed the right sums.
extern "C" int sceKernelAllocateDirectMemory(long searchStart, long searchEnd, unsigned long length, unsigned long alignment, int type, long* physical);
extern "C" int sceKernelMapDirectMemory(void** address, unsigned long length, int protection, int flags, long physical, unsigned long alignment);
extern "C" int scePthreadCreate(void** thread, const void* attr, void* (*entry)(void*), void* arg, const char* name);
extern "C" int scePthreadJoin(void* thread, void** value);
extern "C" [[noreturn]] void exit(int);

int (*volatile create)(void**, const void*, void* (*)(void*), void*, const char*) = scePthreadCreate;

namespace {

constexpr unsigned long Base = 0x4000000000ul;
constexpr unsigned long Bytes = 0x200000ul;
constexpr unsigned long Words = 4096;

void* Worker(void* argument) {
    const unsigned long seed = reinterpret_cast<unsigned long>(argument);
    auto* words = reinterpret_cast<volatile unsigned long*>(Base) + (seed % 8) * Words;
    unsigned long ordinary[64];
    for (unsigned long i = 0; i < Words; ++i) words[i] = i * seed;
    for (unsigned long i = 0; i < 64; ++i) ordinary[i] = i * seed;
    unsigned long sum = 0, expected = 0;
    for (unsigned long i = 0; i < Words; ++i) sum += words[i];
    for (unsigned long i = 0; i < Words; ++i) expected += i * seed;
    // The same sites on ordinary memory once patched: a helper taking either pointer.
    volatile unsigned long* pointers[2] = {words, ordinary};
    unsigned long mixed = 0;
    for (int round = 0; round < 64; ++round) mixed += pointers[round & 1][round];
    unsigned long mixedExpected = 0;
    for (unsigned long round = 0; round < 64; ++round) mixedExpected += round * seed;
    return reinterpret_cast<void*>(sum == expected && mixed == mixedExpected ? 1ul : 2ul);
}

}

extern "C" [[noreturn]] void _start(void*) {
    long physical = 0;
    if (sceKernelAllocateDirectMemory(0, 0x100000000l, Bytes, 0x200000, 3, &physical) != 0) exit(1);
    void* fixed = reinterpret_cast<void*>(Base);
    if (sceKernelMapDirectMemory(&fixed, Bytes, 0x33, 0x10, physical, 0x200000) != 0 || fixed != reinterpret_cast<void*>(Base)) exit(2);
    int good = 0;
    for (long i = 0; i < 64; ++i) {
        void* thread = nullptr;
        if (create(&thread, nullptr, Worker, reinterpret_cast<void*>(i + 3), "displaced churn") != 0) exit(3);
        void* result = nullptr;
        if (scePthreadJoin(thread, &result) != 0) exit(4);
        if (reinterpret_cast<unsigned long>(result) == 1) ++good;
    }
    exit(good == 64 ? 43 : 5);
}
