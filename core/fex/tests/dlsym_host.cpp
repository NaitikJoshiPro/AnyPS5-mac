// sceKernelDlsym on the native libraries: a function comes back as the bridge's x86 stub (the same one
// an import gets), data as its own address.
extern "C" int sceKernelDlsym(int module, const char* name, void** address);
extern "C" unsigned long strlen(const char* text);
extern "C" [[noreturn]] void exit(int);
namespace std {
using terminate_handler = void (*)();
terminate_handler set_terminate(terminate_handler handler) noexcept;
terminate_handler get_terminate() noexcept;
}

unsigned long (*volatile strlenImport)(const char*) = strlen;

extern "C" [[noreturn]] void _start(void*) {
    void* found = nullptr;
    if (sceKernelDlsym(0, "strlen", &found) != 0 || found == nullptr) exit(1);
    if (found != reinterpret_cast<void*>(strlenImport)) exit(2);
    if (reinterpret_cast<unsigned long (*)(const char*)>(found)("guest") != 5) exit(3);

    // A variadic function nothing imports: its stub is made on the spot.
    void* print = nullptr;
    if (sceKernelDlsym(0, "snprintf", &print) != 0 || print == nullptr) exit(4);
    char text[16] = {};
    const int written = reinterpret_cast<int (*)(char*, unsigned long, const char*, ...)>(print)(text, sizeof(text), "%d-%s", 42, "x");
    if (written != 4 || text[0] != '4' || text[1] != '2' || text[2] != '-' || text[3] != 'x' || text[4] != 0) exit(5);

    void* infinity = nullptr;
    if (sceKernelDlsym(0, "_FInf", &infinity) != 0 || infinity == nullptr) exit(6);
    if (*static_cast<const volatile unsigned*>(infinity) != 0x7f800000u) exit(7);

    void* missing = nullptr;
    if (sceKernelDlsym(0, "anyps5_no_such_symbol", &missing) == 0) exit(8);

    // The handler libc installs for a null one is native code too.
    std::set_terminate(nullptr);
    void* abortExport = nullptr;
    if (sceKernelDlsym(0, "abort", &abortExport) != 0 || reinterpret_cast<void*>(std::get_terminate()) != abortExport) exit(9);
    exit(43);
}
