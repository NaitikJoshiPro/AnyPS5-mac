// Variadic exports: the bridge hands libc the guest's own va_list (register save area and overflow
// area). Integers and doubles past the registers, and swprintf, which positional mapping could not pass.
extern "C" {
int snprintf(char*, unsigned long, const char*, ...);
int swprintf(char16_t*, unsigned long, const char16_t*, ...);
[[noreturn]] void exit(int);
}

static void (*volatile exitPointer)(int) = exit;
static int failures = 0;

static bool Same(const char* a, const char* b) {
    while (*a != 0 && *a == *b) ++a, ++b;
    return *a == *b;
}

static bool SameWide(const char16_t* a, const char16_t* b) {
    while (*a != 0 && *a == *b) ++a, ++b;
    return *a == *b;
}

extern "C" [[noreturn]] void _start(void*) {
    char text[160];
    snprintf(text, sizeof(text), "%d %d %d %d %d %d %d %d|%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f|%s", 1, 2, 3, 4, 5, 6, 7, 8,
             0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, "end");
    if (!Same(text, "1 2 3 4 5 6 7 8|0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5|end")) failures |= 1;
    char16_t wide[64];
    const int written = swprintf(wide, 64, u"%d-%d %ls", 12, -7, u"wide");
    if (written != 10 || !SameWide(wide, u"12--7 wide")) failures |= 2;
    exitPointer(failures == 0 ? 43 : 100 + failures);
    __builtin_unreachable();
}
