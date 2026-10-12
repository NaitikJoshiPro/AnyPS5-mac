// A guest program in the shape of a PS5 executable: _start gets the argument block, everything else
// comes from libc.prx by NID.
extern "C" int puts(const char*);
extern "C" [[noreturn]] void exit(int);

struct Error { int code; };

static int cleanups = 0;
struct Guard { ~Guard() { ++cleanups; } };

int thrower(int value) {
    Guard guard;
    if (value > 0) throw Error{value * 2};
    return 0;
}

int (*volatile throwerPointer)(int) = thrower;

int run(int value) {
    try {
        throwerPointer(value);
    } catch (const Error& error) {
        return error.code + cleanups;
    }
    return 1;
}

int (*volatile runPointer)(int) = run;

extern "C" [[noreturn]] void _start(void*) {
    puts("guest C++ exception test");
    exit(runPointer(21));
}
