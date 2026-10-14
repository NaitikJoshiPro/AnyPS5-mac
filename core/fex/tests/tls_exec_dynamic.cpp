// Dynamic TLS in the executable itself, resolved by libkernel's __tls_get_addr export (NID vNe1w4diLCs):
// global-dynamic and local-dynamic accesses call __tls_get_addr with a tls_index whose module comes from
// R_X86_64_DTPMOD64 (symbolless for the local-dynamic one), the offset from R_X86_64_DTPOFF64. The addresses
// must lie in the calling thread's block just below its thread pointer. A second thread must see the initial
// values, not the main thread's. Exit 43 when every value is right; otherwise the failing checks' bits.
extern "C" int scePthreadCreate(void** thread, const void* attr, void* (*entry)(void*), void* arg, const char* name);
extern "C" int scePthreadJoin(void* thread, void** value);
extern "C" [[noreturn]] void exit(int);

// Called through a data pointer, so the executable has a RELA section (the relinker wants one). Hidden, as
// the shared-object link would otherwise leave the executable importing its own variable.
__attribute__((visibility("hidden"))) int (*volatile create)(void**, const void*, void* (*)(void*), void*, const char*) = scePthreadCreate;

__attribute__((tls_model("global-dynamic"), visibility("default"))) thread_local long shared = 5;

namespace {

__attribute__((tls_model("local-dynamic"))) thread_local int counter = 40;
__attribute__((tls_model("local-dynamic"))) thread_local char bytes[3] = {1, 2, 3};
__attribute__((tls_model("local-dynamic"))) thread_local long zeroed[4];

// The executable's block sits just below the thread pointer (variant II), whose first word points at itself.
bool BelowThreadPointer(const void* address) {
    unsigned long pointer = 0;
    asm volatile("mov %%fs:0, %0" : "=r"(pointer));
    const auto value = reinterpret_cast<unsigned long>(address);
    return value < pointer && pointer - value <= 4096;
}

int Check(int bit) {
    int failures = 0;
    if (counter != 40) failures |= bit;
    if (bytes[0] != 1 || bytes[1] != 2 || bytes[2] != 3) failures |= bit << 1;
    for (long value : zeroed)
        if (value != 0) failures |= bit << 2;
    if (shared != 5) failures |= bit << 3;
    return failures;
}

void Change() {
    counter = 7;
    bytes[1] = 9;
    zeroed[3] = -1;
    shared += 3;
}

void* Worker(void* result) {
    int failures = Check(1) != 0 ? 64 : 0;
    if (!BelowThreadPointer(&shared) || !BelowThreadPointer(&counter)) failures |= 128;
    Change();
    *static_cast<int*>(result) = failures;
    return nullptr;
}

}

extern "C" [[noreturn]] void _start(void*) {
    int failures = Check(1);
    Change();
    if (counter != 7 || bytes[1] != 9 || zeroed[3] != -1 || shared != 8) failures |= 16;
    if (!BelowThreadPointer(&shared) || !BelowThreadPointer(&counter)) failures |= 256;
    int worker = -1;
    void* thread = nullptr;
    if (create(&thread, nullptr, Worker, &worker, "tls") != 0) exit(1);
    scePthreadJoin(thread, nullptr);
    failures |= worker;
    if (counter != 7 || bytes[1] != 9 || shared != 8) failures |= 32;
    if (reinterpret_cast<unsigned long>(&zeroed[0]) % alignof(long) != 0) failures |= 512;
    exit(failures == 0 ? 43 : failures);
}
