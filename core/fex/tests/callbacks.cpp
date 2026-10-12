extern "C" void qsort(void* base, unsigned long count, unsigned long size, int (*compare)(const void*, const void*));
extern "C" void* bsearch(const void* key, const void* base, unsigned long count, unsigned long size,
                         int (*compare)(const void*, const void*));
extern "C" int strcmp(const char* left, const char* right);
extern "C" [[noreturn]] void exit(int);

void (*volatile exitPointer)(int) = exit;

static int failures = 0;

static void Check(bool right, int bit) {
    if (!right) failures |= 1 << bit;
}

static int CompareIntegers(const void* left, const void* right) {
    const unsigned a = *static_cast<const unsigned*>(left), b = *static_cast<const unsigned*>(right);
    return a < b ? -1 : a > b;
}

static int CompareNames(const void* left, const void* right) {
    return strcmp(*static_cast<const char* const*>(left), *static_cast<const char* const*>(right));
}

static int CompareRoots(const void* left, const void* right) {
    const double a = __builtin_sqrt(*static_cast<const double*>(left)), b = __builtin_sqrt(*static_cast<const double*>(right));
    return a > b ? -1 : a < b;
}

static unsigned values[2000];

static void Integers() {
    unsigned x = 2463534242u;
    for (unsigned& value : values) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        value = x & ~1u;
    }
    const unsigned first = values[0], last = values[1999];
    qsort(values, 2000, sizeof(values[0]), CompareIntegers);
    bool sorted = true;
    for (int i = 1; i < 2000; ++i) sorted = sorted && values[i - 1] <= values[i];
    Check(sorted, 0);
    const auto* found = static_cast<const unsigned*>(bsearch(&first, values, 2000, sizeof(values[0]), CompareIntegers));
    Check(found != nullptr && *found == first, 1);
    found = static_cast<const unsigned*>(bsearch(&last, values, 2000, sizeof(values[0]), CompareIntegers));
    Check(found != nullptr && *found == last, 1);
    const unsigned missing = first | 1;
    Check(bsearch(&missing, values, 2000, sizeof(values[0]), CompareIntegers) == nullptr, 1);
}

static void Names() {
    const char* names[] = {"viola", "cello", "oboe", "bassoon", "flute", "harp", "clarinet", "horn", "tuba", "celesta"};
    qsort(names, 10, sizeof(names[0]), CompareNames);
    const char* expected[] = {"bassoon", "celesta", "cello", "clarinet", "flute", "harp", "horn", "oboe", "tuba", "viola"};
    bool right = true;
    for (int i = 0; i < 10; ++i) right = right && strcmp(names[i], expected[i]) == 0;
    Check(right, 2);
    const char* key = "horn";
    const auto* found = static_cast<const char* const*>(bsearch(&key, names, 10, sizeof(names[0]), CompareNames));
    Check(found == &names[6], 2);
}

static void Roots() {
    double squares[] = {4.0, 81.0, 0.25, 16.0, 2.0, 100.0, 9.0};
    qsort(squares, 7, sizeof(squares[0]), CompareRoots);
    const double expected[] = {100.0, 81.0, 16.0, 9.0, 4.0, 2.0, 0.25};
    bool right = true;
    for (int i = 0; i < 7; ++i) right = right && squares[i] == expected[i];
    Check(right, 3);
}

extern "C" [[noreturn]] void _start(void*) {
    Integers();
    Names();
    Roots();
    exitPointer(failures == 0 ? 43 : 100 + failures);
    __builtin_unreachable();
}
