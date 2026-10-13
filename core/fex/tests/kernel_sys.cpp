// A title that imports POSIX file calls from libkernel (as SuperTuxKart does) reaches the ones AnyPS5
// implements in libkernel_sys: the runner opens libkernel_sys.prx after libkernel.prx.
extern "C" long pathconf(const char* path, int name);
extern "C" [[noreturn]] void exit(int);

// Taking the import's address gives the program a data relocation as well as its jump slots.
long (*volatile pathconfImport)(const char*, int) = pathconf;

extern "C" [[noreturn]] void _start(void*) {
    constexpr int PcNameMax = 4;
    if (pathconf(".", PcNameMax) != 255) exit(1);
    if (pathconfImport(".", PcNameMax) != 255) exit(2);
    exit(43);
}
