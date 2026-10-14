// A translated load/store site patched for a displaced guest address (D31, D33) runs a stub that copies the
// instruction. Under FEX's TSO emulation that copy is an ordered access (LDAPUR/STLUR, LDAR/STLR), which the M3
// faults (SIGBUS BUS_ADRALN) when it crosses 16 bytes; FEX's own handler only rewrites its code buffers, so libc's
// displaced handler rewrites the stub's copy (ASTRO BOT died on a STLURH there). Each site here is first patched
// with a displaced pointer, then fed ordinary and displaced pointers that cross 16 bytes, on a second thread so the
// code is translated with TSO on, in two passes (the second runs the rewritten stubs). Exit 43 when every value is
// right; otherwise the failing check's number.
extern "C" int sceKernelAllocateDirectMemory(long searchStart, long searchEnd, unsigned long length, unsigned long alignment, int type, long* physical);
extern "C" int sceKernelMapDirectMemory(void** address, unsigned long length, int protection, int flags, long physical, unsigned long alignment);
extern "C" int scePthreadCreate(void** thread, const void* attr, void* (*entry)(void*), void* arg, const char* name);
extern "C" int scePthreadJoin(void* thread, void** value);
extern "C" [[noreturn]] void exit(int);

// Called through a data pointer, so the executable has a RELA section (the relinker wants one).
int (*volatile create)(void**, const void*, void* (*)(void*), void*, const char*) = scePthreadCreate;

namespace {

constexpr unsigned long TailBase = 0x400000000ul; // the dyld shared region's tail (D33)
constexpr unsigned long Base = 0x4000000000ul;    // the GPU carveout (D31)
constexpr unsigned long Bytes = 0x200000ul;
constexpr int ReadWrite = 0x33;
constexpr int MapFixed = 0x10;

alignas(64) unsigned char heap[256];

// One store and one load site per width, with an immediate offset (STLUR/LDAPUR under TSO).
template <typename T>
[[gnu::noinline]] void Store(unsigned char* address, T value) {
    asm volatile("mov %1, 1(%0)" : : "r"(address), "r"(value) : "memory");
}

template <typename T>
[[gnu::noinline]] T Load(const unsigned char* address) {
    T value;
    asm volatile("mov 1(%1), %0" : "=r"(value) : "r"(address) : "memory");
    return value;
}

template <typename T>
int Width(unsigned char* view, int check) {
    auto* tail = reinterpret_cast<unsigned char*>(TailBase);
    auto* carveout = reinterpret_cast<unsigned char*>(Base);
    for (int pass = 0; pass < 2; ++pass) {
        const T seed = static_cast<T>(0x0123456789abcdefull + static_cast<unsigned long>(pass) * 0x1111 + sizeof(T));
        // Aligned displaced access first: the sites are patched to their stubs.
        Store<T>(tail + 63, seed);
        if (Load<T>(tail + 63) != seed) return check + 1;
        // Ordinary pointers crossing 16 bytes (the stub's plain path), bytes around them untouched.
        for (int o = 13; o <= 15; ++o) {
            unsigned char* at = heap + 64 + o - 1; // the access is at at + 1
            for (int i = 0; i < 64; ++i) heap[64 + i] = 0xa5;
            const T value = static_cast<T>(seed * 3 + static_cast<T>(o));
            Store<T>(at, value);
            if (Load<T>(at) != value) return check + 2;
            if (heap[64 + o - 1] != 0xa5 || heap[64 + o + static_cast<int>(sizeof(T))] != 0xa5) return check + 3;
        }
        // Displaced pointers crossing 16 bytes (the stub's flip path), checked through the ordinary view.
        const unsigned long offsets[] = {0x1000ul + 15, 0x2000ul + 13};
        for (const unsigned long offset : offsets) {
            unsigned char* windows[] = {tail, carveout};
            for (unsigned char* window : windows) {
                const T value = static_cast<T>(seed * 7 + static_cast<T>(offset));
                Store<T>(window + offset - 1, value);
                if (Load<T>(window + offset - 1) != value) return check + 4;
                T seen = 0;
                for (unsigned i = 0; i < sizeof(T); ++i) seen |= static_cast<T>(static_cast<T>(view[offset + i]) << (8 * i));
                if (seen != value) return check + 5;
            }
        }
    }
    return 0;
}

struct Work {
    unsigned char* view;
    int result;
};

void* Worker(void* argument) {
    auto* work = static_cast<Work*>(argument);
    work->result = Width<unsigned short>(work->view, 10);
    if (work->result == 0) work->result = Width<unsigned int>(work->view, 20);
    if (work->result == 0) work->result = Width<unsigned long>(work->view, 30);
    return nullptr;
}

}

extern "C" [[noreturn]] void _start(void*) {
    long physical = 0;
    if (sceKernelAllocateDirectMemory(0, 0x100000000l, Bytes, 0x200000, 3, &physical) != 0) exit(1);
    void* tail = reinterpret_cast<void*>(TailBase);
    if (sceKernelMapDirectMemory(&tail, Bytes, ReadWrite, MapFixed, physical, 0x200000) != 0 || tail != reinterpret_cast<void*>(TailBase)) exit(2);
    void* carveout = reinterpret_cast<void*>(Base);
    if (sceKernelMapDirectMemory(&carveout, Bytes, ReadWrite, MapFixed, physical, 0x200000) != 0 || carveout != reinterpret_cast<void*>(Base)) exit(3);
    void* view = nullptr;
    if (sceKernelMapDirectMemory(&view, Bytes, ReadWrite, 0, physical, 0x200000) != 0 || view == nullptr) exit(4);
    Work work{static_cast<unsigned char*>(view), -1};
    void* thread = nullptr;
    if (create(&thread, nullptr, Worker, &work, "displaced-unaligned") != 0) exit(5);
    scePthreadJoin(thread, nullptr);
    exit(work.result == 0 ? 43 : work.result);
}
