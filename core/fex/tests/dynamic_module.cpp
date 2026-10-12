extern "C" int sceKernelLoadStartModule(const char* path, unsigned long args, const void* argp, unsigned flags, const void* option, int* result);
extern "C" int sceKernelDlsym(int module, const char* name, void** address);
extern "C" [[noreturn]] void exit(int);

int (*volatile loadStartModule)(const char*, unsigned long, const void*, unsigned, const void*, int*) = sceKernelLoadStartModule;

extern "C" [[noreturn]] void _start(void*) {
    int result = -1;
    const int module = loadStartModule("/app0/sce_module/libgreet.prx", 0, nullptr, 0, nullptr, &result);
    if (module < 0 || result != 0) exit(1);
    void* greet = nullptr;
    if (sceKernelDlsym(module, "greet", &greet) != 0 || greet == nullptr) exit(2);
    exit(reinterpret_cast<int (*)(int)>(greet)(21));
}
