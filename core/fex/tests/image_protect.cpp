// The runner maps the executable itself, so the guest memory registry must know its pages: the guest
// queries and protects its own code and data.
extern "C" int sceKernelMprotect(const void* address, unsigned long length, int protection);
extern "C" int sceKernelQueryMemoryProtection(void* address, void** start, void** end, int* protection);
extern "C" [[noreturn]] void exit(int);

constexpr int CpuRead = 1;
constexpr int CpuWrite = 2;
constexpr int CpuExecute = 4;
constexpr unsigned long GuestPage = 0x4000;

// One guest page of .data of its own.
alignas(GuestPage) static volatile unsigned char page[GuestPage] = {11};

// A relocated pointer: the relinker wants a DT_RELA table.
void (*volatile exitPointer)(int) = exit;

static bool Query(void* address, int expected, bool exact) {
    void* start = nullptr;
    void* end = nullptr;
    int protection = -1;
    if (sceKernelQueryMemoryProtection(address, &start, &end, &protection) != 0) return false;
    const auto at = reinterpret_cast<unsigned long>(address);
    if (reinterpret_cast<unsigned long>(start) > at || reinterpret_cast<unsigned long>(end) <= at) return false;
    if (exact && (start != address || reinterpret_cast<unsigned long>(end) != at + GuestPage)) return false;
    return protection == expected || (!exact && (protection & (CpuRead | CpuWrite)) == expected);
}

extern "C" [[noreturn]] void _start(void*) {
    auto* data = const_cast<unsigned char*>(page);
    // Code is image memory, which only the registry knows: the runner maps it as plain host data.
    void* start = nullptr;
    void* end = nullptr;
    int protection = -1;
    if (sceKernelQueryMemoryProtection(reinterpret_cast<void*>(&_start), &start, &end, &protection) != 0) exit(1);
    if ((protection & (CpuRead | CpuExecute)) != (CpuRead | CpuExecute)) exit(2);
    if (!Query(data, CpuRead | CpuWrite, false)) exit(3);
    if (sceKernelMprotect(data, GuestPage, CpuRead) != 0) exit(4);
    if (!Query(data, CpuRead, true)) exit(5);
    if (page[0] != 11) exit(6);
    if (sceKernelMprotect(data + 0x100, 0x10, CpuRead | CpuWrite) != 0) exit(7);
    if (!Query(data, CpuRead | CpuWrite, true)) exit(8);
    page[GuestPage - 1] = 31;
    if (page[GuestPage - 1] != 31) exit(9);
    // Executable data stays data on the host: the translator reads guest code, the CPU never runs it.
    if (sceKernelMprotect(data, GuestPage, CpuRead | CpuWrite | CpuExecute) != 0) exit(10);
    page[0] = 12;
    if (page[0] != 12 || !Query(data, CpuRead | CpuWrite | CpuExecute, true)) exit(11);
    exitPointer(43);
    __builtin_unreachable();
}
