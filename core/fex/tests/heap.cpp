extern "C" void _init_env();
extern "C" void* malloc(unsigned long size);
extern "C" void free(void* pointer);
extern "C" void* calloc(unsigned long count, unsigned long size);
extern "C" void* realloc(void* pointer, unsigned long size);
extern "C" void* memalign(unsigned long alignment, unsigned long size);
extern "C" int posix_memalign(void** pointer, unsigned long alignment, unsigned long size);
extern "C" [[noreturn]] void exit(int);

void (*volatile exitPointer)(int) = exit;

static int failures = 0;

static void Check(bool right, int bit) {
    if (!right) failures |= 1 << bit;
}

alignas(4096) static unsigned char arena[32 << 20];
static unsigned long used = 0;
static unsigned long calls[7] = {};
static bool initialized = false;

static void* Take(unsigned long alignment, unsigned long size) {
    const unsigned long start = (used + 16 + alignment - 1) & ~(alignment - 1);
    if (start + size > sizeof(arena)) return nullptr;
    *reinterpret_cast<unsigned long*>(arena + start - 16) = size;
    used = start + size;
    return arena + start;
}

static void* Copy(void* from, void* to, unsigned long size) {
    const unsigned long old = *reinterpret_cast<unsigned long*>(static_cast<unsigned char*>(from) - 16);
    for (unsigned long i = 0; i < old && i < size; ++i) static_cast<unsigned char*>(to)[i] = static_cast<unsigned char*>(from)[i];
    return to;
}

static void* Allocate(unsigned long size) { ++calls[0]; return Take(16, size); }
static void Release(void*) { ++calls[1]; }
static void* Clear(unsigned long count, unsigned long size) {
    ++calls[2];
    auto* block = static_cast<unsigned char*>(Take(16, count * size));
    for (unsigned long i = 0; i < count * size; ++i) block[i] = 0;
    return block;
}
static void* Resize(void* pointer, unsigned long size) { ++calls[3]; return Copy(pointer, Take(16, size), size); }
static void* Aligned(unsigned long alignment, unsigned long size) { ++calls[4]; return Take(alignment, size); }
static void* Realigned(void* pointer, unsigned long size, unsigned long alignment) { ++calls[5]; return Copy(pointer, Take(alignment, size), size); }
static int PosixAligned(void** pointer, unsigned long alignment, unsigned long size) {
    ++calls[6];
    *pointer = Take(alignment, size);
    return *pointer != nullptr ? 0 : 12;
}
static void Initialize() { initialized = true; }
static void Finalize() {}

struct Replacement {
    unsigned long size, version;
    void (*initialize)();
    void (*finalize)();
    void* (*allocate)(unsigned long);
    void (*release)(void*);
    void* (*clear)(unsigned long, unsigned long);
    void* (*resize)(void*, unsigned long);
    void* (*aligned)(unsigned long, unsigned long);
    void* (*realigned)(void*, unsigned long, unsigned long);
    int (*posixAligned)(void**, unsigned long, unsigned long);
    const void* reserved[4];
};
static_assert(sizeof(Replacement) == 0x78, "the replacement table is 0x78 bytes");

struct LibcParameters {
    unsigned long size;
    unsigned long reserved[5];
    const Replacement* replacement;
};
static_assert(sizeof(LibcParameters) == 0x38, "libc reads the replacement at 0x30");

struct ProcessParameters {
    unsigned long size;
    unsigned magic, version;
    unsigned long reserved[5];
    const LibcParameters* libc;
};
static_assert(sizeof(ProcessParameters) == 0x40, "libc reads its parameters at 0x38");

static const Replacement replacement {0x78, 2, Initialize, Finalize, Allocate, Release, Clear, Resize, Aligned, Realigned, PosixAligned, {}};
static const LibcParameters libcParameters {0x38, {}, &replacement};
[[gnu::used]] static const ProcessParameters processParameters {0x40, 0x4942524f, 0, {}, &libcParameters};

static bool InArena(const void* pointer) {
    return pointer >= arena && pointer < arena + sizeof(arena);
}

extern "C" [[noreturn]] void _start(void*) {
    _init_env();
    Check(initialized, 0);

    bool owned = true;
    for (int i = 0; i < 100000; ++i) {
        auto* block = static_cast<unsigned char*>(malloc(48));
        owned = owned && InArena(block);
        block[47] = static_cast<unsigned char>(i);
        free(block);
    }
    Check(owned && calls[0] >= 100000 && calls[1] >= 100000, 1);

    auto* zeros = static_cast<unsigned char*>(calloc(100, 8));
    bool zeroed = InArena(zeros) && calls[2] == 1;
    for (int i = 0; i < 800; ++i) zeroed = zeroed && zeros[i] == 0;
    auto* text = static_cast<char*>(malloc(6));
    for (int i = 0; i < 6; ++i) text[i] = "heaps"[i];
    text = static_cast<char*>(realloc(text, 4096));
    Check(zeroed && InArena(text) && calls[3] == 1 && text[0] == 'h' && text[4] == 's', 2);

    void* page = memalign(4096, 100);
    void* line = nullptr;
    const int error = posix_memalign(&line, 256, 10);
    Check(InArena(page) && (reinterpret_cast<unsigned long>(page) & 4095) == 0 && calls[4] == 1 && error == 0 && InArena(line) &&
              (reinterpret_cast<unsigned long>(line) & 255) == 0 && calls[6] == 1,
          3);
    exitPointer(failures == 0 ? 43 : 100 + failures);
    __builtin_unreachable();
}
