// CPUID's processor brand string (leaves 0x80000002-0x80000004), which SuperTuxKart reads while it collects
// hardware information. FEX keeps it per host core in data sized from HostFeatures::CPUMIDRs, which only Linux
// filled; on macOS the leaves read past an empty vector and crashed on a null name. Also the highest extended
// leaf must cover them. Exit 43 when the string is non-empty, NUL-terminated within 48 bytes and printable;
// otherwise the failing checks' bits.
extern "C" [[noreturn]] void exit(int);

// Called through a data pointer, so the executable has a RELA section (the relinker wants one).
void (*volatile quit)(int) = exit;

namespace {

void Cpuid(unsigned leaf, unsigned out[4]) {
    asm volatile("cpuid" : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3]) : "a"(leaf), "c"(0));
}

}

extern "C" [[noreturn]] void _start(void*) {
    int failures = 0;
    unsigned registers[4];
    Cpuid(0x80000000u, registers);
    if (registers[0] < 0x80000004u) failures |= 1;
    char brand[49] = {};
    for (unsigned leaf = 0; leaf < 3; ++leaf) {
        Cpuid(0x80000002u + leaf, registers);
        __builtin_memcpy(brand + leaf * 16, registers, 16);
    }
    if (brand[0] == 0) failures |= 2;
    bool terminated = false;
    for (int i = 0; i < 48 && !terminated; ++i) {
        if (brand[i] == 0) terminated = true;
        else if (brand[i] < 0x20 || brand[i] > 0x7e) failures |= 4;
    }
    if (!terminated) failures |= 8;
    quit(failures == 0 ? 43 : failures);
    __builtin_unreachable();
}
