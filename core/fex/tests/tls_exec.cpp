// Initial-exec TLS in the executable itself, as a PS5 eboot has it: the GOT slots carry R_X86_64_TPOFF64
// relocations (symbolless for a file-local variable, and named for an exported one), which the relinker
// passes to the Linux ELF and aps5-fex resolves against the executable's block below the thread pointer.
// Linked as a shared object (fex_toolchain.link_tpoff_executable) because an LLVM executable link would
// relax every access to local exec. A second thread must see the initial values, not the main thread's.
// Exit 43 when every value is right; otherwise the failing checks' bits.
extern "C" int scePthreadCreate(void** thread, const void* attr, void* (*entry)(void*), void* arg, const char* name);
extern "C" int scePthreadJoin(void* thread, void** value);
extern "C" [[noreturn]] void exit(int);

// Called through a data pointer, so the executable has a RELA section (the relinker wants one). Hidden, as
// the shared-object link would otherwise leave the executable importing its own variable.
__attribute__((visibility("hidden"))) int (*volatile create)(void**, const void*, void* (*)(void*), void*, const char*) = scePthreadCreate;

namespace {

__attribute__((tls_model("initial-exec"))) thread_local int counter = 40;
__attribute__((tls_model("initial-exec"))) thread_local char bytes[3] = {1, 2, 3};
__attribute__((tls_model("initial-exec"))) thread_local long zeroed[4];

int Check(int bit) {
    int failures = 0;
    if (counter != 40) failures |= bit;
    if (bytes[0] != 1 || bytes[1] != 2 || bytes[2] != 3) failures |= bit << 1;
    for (long value : zeroed)
        if (value != 0) failures |= bit << 2;
    return failures;
}

void Change() {
    counter = 7;
    bytes[1] = 9;
    zeroed[3] = -1;
}

void* Worker(void* result) {
    *static_cast<int*>(result) = Check(1) != 0 ? 64 : 0;
    Change();
    return nullptr;
}

}

__attribute__((tls_model("initial-exec"), visibility("default"))) thread_local long exported = 5;

extern "C" [[noreturn]] void _start(void*) {
    int failures = Check(1);
    Change();
    exported += 3;
    if (counter != 7 || bytes[1] != 9 || zeroed[3] != -1) failures |= 8;
    int worker = -1;
    void* thread = nullptr;
    if (create(&thread, nullptr, Worker, &worker, "tls") != 0) exit(1);
    scePthreadJoin(thread, nullptr);
    failures |= worker;
    if (counter != 7 || bytes[1] != 9 || exported != 8) failures |= 16;
    if (reinterpret_cast<unsigned long>(&zeroed[0]) % alignof(long) != 0) failures |= 32;
    exit(failures == 0 ? 43 : failures);
}
