// Fixed guest mappings in the macOS arm64 GPU carveout (64-448 GiB), as the oops-sdk display path makes at
// 0x4000000000 (D31), and in the dyld shared region's unused tail, as oops-mesa's GPU VA high range does at
// 0x400000000 (D33). The memory is backed at a host alias; translated code reaches it through backpatched stubs
// and native library code through the fault path. A second, ordinary view of the same direct memory checks that
// both name the same pages.
extern "C" int sceKernelAllocateDirectMemory(long searchStart, long searchEnd, unsigned long length, unsigned long alignment, int type, long* physical);
extern "C" int sceKernelMapDirectMemory(void** address, unsigned long length, int protection, int flags, long physical, unsigned long alignment);
extern "C" int sceKernelMunmap(void* address, unsigned long length);
extern "C" void* memset(void* destination, int value, unsigned long length);
extern "C" void* memcpy(void* destination, const void* source, unsigned long length);
extern "C" [[noreturn]] void exit(int);

constexpr unsigned long Base = 0x4000000000ul;
constexpr unsigned long TailBase = 0x400000000ul;
constexpr unsigned long Bytes = 0x400000ul; // 4 MiB, two 2 MiB pages
constexpr int ReadWrite = 0x33;          // CPU and GPU read/write
constexpr int MapFixed = 0x10;

// A relocated pointer: the relinker wants a DT_RELA table.
void (*volatile exitPointer)(int) = exit;

template <typename T>
static T Load(const void* address) {
    return *static_cast<const volatile T*>(address);
}

// One load site for every pointer: once it is patched for a displaced address, the stub's window check must send
// ordinary addresses down the plain path and both displaced ranges through the alias.
[[gnu::noinline]] static unsigned long SumWords(const volatile unsigned long* words, unsigned long count) {
    unsigned long sum = 0;
    for (unsigned long i = 0; i < count; ++i) sum += words[i];
    return sum;
}

extern "C" [[noreturn]] void _start(void*) {
    long physical = 0;
    if (sceKernelAllocateDirectMemory(0, 0x100000000l, Bytes, 0x200000, 3, &physical) != 0) exit(1);
    void* fixed = reinterpret_cast<void*>(Base);
    if (sceKernelMapDirectMemory(&fixed, Bytes, ReadWrite, MapFixed, physical, 0x200000) != 0 || fixed != reinterpret_cast<void*>(Base)) exit(2);
    void* view = nullptr;
    if (sceKernelMapDirectMemory(&view, Bytes, ReadWrite, 0, physical, 0x200000) != 0 || view == nullptr) exit(3);
    if (reinterpret_cast<unsigned long>(view) - 0x1000000000ul < 0x6000000000ul) exit(4); // first fit stays out of the window

    // Translated stores and loads: a hot loop, so the sites are patched and run without faults.
    auto* words = static_cast<volatile unsigned long*>(fixed);
    const unsigned long count = Bytes / sizeof(unsigned long);
    for (unsigned long i = 0; i < count; ++i) words[i] = i * 0x9e3779b97f4a7c15ul;
    unsigned long sum = 0;
    for (unsigned long i = 0; i < count; ++i) sum += words[i];
    unsigned long expected = 0;
    for (unsigned long i = 0; i < count; ++i) expected += i * 0x9e3779b97f4a7c15ul;
    if (sum != expected) exit(5);
    // The same pages through the ordinary view.
    if (Load<unsigned long>(static_cast<char*>(view) + 8 * 12345) != 12345ul * 0x9e3779b97f4a7c15ul) exit(6);
    static_cast<volatile unsigned long*>(view)[7] = 0x1234;
    if (words[7] != 0x1234) exit(7);

    // Base register small, index register displaced: x86 [rax + rcx] where rcx holds the pointer.
    unsigned long index = Base + 64;
    unsigned long small = 8;
    unsigned long loaded = 0;
    __asm__ volatile("movq $0x55aa, (%1,%2)\n\tmovq (%2,%1), %0" : "=r"(loaded) : "r"(small), "r"(index) : "memory");
    if (loaded != 0x55aa || words[9] != 0x55aa) exit(8);

    // Atomics.
    auto* counter = reinterpret_cast<unsigned long*>(Base + 0x1000);
    *counter = 5;
    __atomic_fetch_add(counter, 3, __ATOMIC_SEQ_CST);
    unsigned long want = 8;
    if (!__atomic_compare_exchange_n(counter, &want, 42, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) || *counter != 42) exit(9);

    // Native library code on displaced memory (the fault path): a small memset and memcpy.
    memset(reinterpret_cast<void*>(Base + 0x2000), 0x5a, 256);
    if (Load<unsigned char>(static_cast<char*>(view) + 0x2000 + 255) != 0x5a) exit(10);
    memcpy(reinterpret_cast<void*>(Base + 0x3000), reinterpret_cast<void*>(Base + 0x2000), 128);
    if (Load<unsigned char>(static_cast<char*>(view) + 0x3000 + 127) != 0x5a) exit(11);

    // The shared region tail: a third view of the same pages at 0x400000000.
    void* tail = reinterpret_cast<void*>(TailBase);
    if (sceKernelMapDirectMemory(&tail, Bytes, ReadWrite, MapFixed, physical, 0x200000) != 0 || tail != reinterpret_cast<void*>(TailBase)) exit(15);
    auto* tailWords = static_cast<volatile unsigned long*>(tail);
    if (tailWords[7] != 0x1234) exit(16);
    for (unsigned long i = 0; i < 4096; ++i) tailWords[1024 + i] = i ^ 0xabcdef;
    if (Load<unsigned long>(static_cast<char*>(view) + 8 * 2000) != (976ul ^ 0xabcdef)) exit(17);
    const unsigned long tailSum = SumWords(tailWords + 1024, 4096);
    if (SumWords(words + 1024, 4096) != tailSum || SumWords(static_cast<volatile unsigned long*>(view) + 1024, 4096) != tailSum) exit(18);
    if (SumWords(tailWords + 1024, 4096) != tailSum) exit(19);
    if (sceKernelMunmap(tail, Bytes) != 0) exit(20);

    // Unmap and map again at the same place.
    if (sceKernelMunmap(fixed, Bytes) != 0) exit(12);
    void* again = reinterpret_cast<void*>(Base);
    if (sceKernelMapDirectMemory(&again, Bytes, ReadWrite, MapFixed, physical, 0x200000) != 0 || again != fixed) exit(13);
    if (words[7] != 0x1234) exit(14);
    exit(43);
}
