// Unaligned guest accesses that cross a 16-byte boundary. Under FEX's TSO emulation a plain load or store becomes
// LDAPUR/STLUR and a locked one an LSE atomic; the M3 raises SIGBUS (BUS_ADRALN) for those when they cross 16 bytes,
// and the runner hands the fault to FEX, which emulates or backpatches the access. Every access runs in two passes
// (the second runs the backpatched code) on two threads at once. Exit 43 when every value is right; otherwise the
// failing checks' bits.
extern "C" int scePthreadCreate(void** thread, const void* attr, void* (*entry)(void*), void* arg, const char* name);
extern "C" int scePthreadJoin(void* thread, void** value);
extern "C" [[noreturn]] void exit(int);

// Called through a data pointer, so the executable has a RELA section (the relinker wants one).
int (*volatile create)(void**, const void*, void* (*)(void*), void*, const char*) = scePthreadCreate;

namespace {

constexpr int Iterations = 2000;
constexpr int Offsets[] = {12, 14, 15};

// One 64-byte line per offset and width, so no access crosses a cache line (the locked ones stay single-copy atomic).
alignas(64) unsigned char shared[3][3][64];

template <typename T>
T Load(const unsigned char* address) {
    T value;
    asm volatile("mov %1, %0" : "=r"(value) : "m"(*reinterpret_cast<const T*>(address)));
    return value;
}

template <typename T>
void Store(unsigned char* address, T value) {
    asm volatile("mov %1, %0" : "=m"(*reinterpret_cast<T*>(address)) : "r"(value));
}

template <typename T>
void LockAdd(unsigned char* address, T value) {
    asm volatile("lock xadd %0, %1" : "+r"(value), "+m"(*reinterpret_cast<T*>(address)) : : "memory");
}

template <typename T>
void LockIncrement(unsigned char* address) {
    T* target = reinterpret_cast<T*>(address);
    T expected = Load<T>(address);
    for (;;) {
        T desired = static_cast<T>(expected + 1);
        bool swapped;
        asm volatile("lock cmpxchg %3, %1" : "+a"(expected), "+m"(*target), "=@ccz"(swapped) : "r"(desired) : "memory");
        if (swapped) return;
    }
}

// Plain accesses on this thread's own lines: a value written must read back unchanged, bytes around it untouched.
template <typename T>
bool Plain(unsigned char (&lines)[3][64], int width, unsigned seed) {
    bool right = true;
    for (int o = 0; o < 3; ++o) {
        unsigned char* line = lines[o];
        for (int i = 0; i < 64; ++i) line[i] = 0xa5;
        for (int i = 0; i < 64; ++i) {
            const T value = static_cast<T>(0x0123456789abcdefULL * (seed + static_cast<unsigned>(i)) + static_cast<unsigned>(width));
            Store<T>(line + Offsets[o], value);
            right &= Load<T>(line + Offsets[o]) == value;
        }
        right &= line[Offsets[o] - 1] == 0xa5 && line[Offsets[o] + static_cast<int>(sizeof(T))] == 0xa5;
    }
    return right;
}

alignas(64) unsigned char ownLines[2][3][3][64];

struct Work {
    int index;
    bool plain;
};

template <typename T>
void Locked(int width) {
    for (int o = 0; o < 3; ++o) {
        unsigned char* address = shared[width][o] + Offsets[o];
        for (int i = 0; i < Iterations; ++i) {
            LockAdd<T>(address, 1);
            LockIncrement<T>(address + 0);
        }
    }
}

void* Worker(void* argument) {
    auto* work = static_cast<Work*>(argument);
    for (int pass = 0; pass < 2; ++pass) {
        work->plain &= Plain<unsigned short>(ownLines[work->index][0], 0, 1u + static_cast<unsigned>(pass));
        work->plain &= Plain<unsigned int>(ownLines[work->index][1], 1, 7u + static_cast<unsigned>(pass));
        work->plain &= Plain<unsigned long>(ownLines[work->index][2], 2, 13u + static_cast<unsigned>(pass));
        Locked<unsigned short>(0);
        Locked<unsigned int>(1);
        Locked<unsigned long>(2);
    }
    return nullptr;
}

template <typename T>
bool Counted(int width) {
    // Two threads, two passes, two increments per iteration.
    constexpr unsigned long Expected = 2UL * 2UL * 2UL * Iterations;
    bool right = true;
    for (int o = 0; o < 3; ++o) right &= static_cast<unsigned long>(Load<T>(shared[width][o] + Offsets[o])) == static_cast<T>(Expected);
    return right;
}

}

extern "C" [[noreturn]] void _start(void*) {
    Work works[2] = {{0, true}, {1, true}};
    void* threads[2] = {};
    for (int i = 0; i < 2; ++i)
        if (create(&threads[i], nullptr, Worker, &works[i], "unaligned") != 0) exit(1);
    for (void* thread : threads) scePthreadJoin(thread, nullptr);
    int failures = 0;
    if (!works[0].plain || !works[1].plain) failures |= 2;
    if (!Counted<unsigned short>(0)) failures |= 4;
    if (!Counted<unsigned int>(1)) failures |= 8;
    if (!Counted<unsigned long>(2)) failures |= 16;
    exit(failures == 0 ? 43 : failures);
}
