// long double: the guest's x87 80-bit value on the stack and in st(0), a double on arm64. Covers libc's
// 16 long double math exports, strtold/wcstold (GuestLongDouble results), __signbitl (GuestLongDouble
// argument) and the test library's mix of long double and int arguments.
#include "abi.h"

extern "C" {
long double acosl(long double);
long double atanl(long double);
long double coshl(long double);
long double cosl(long double);
long double exp2l(long double);
long double expl(long double);
long double frexpl(long double, int*);
long double ldexpl(long double, int);
long double log10l(long double);
long double logl(long double);
long double powl(long double, long double);
long double sinhl(long double);
long double sinl(long double);
long double tanhl(long double);
long double tanl(long double);
long double truncl(long double);
long double strtold(const char*, char**);
long double wcstold(const char16_t*, char16_t**);
int __signbitl(long double);
}

// |a - b| within a few double ulps: the library computes in double precision.
static bool Near(long double a, long double b) {
    const long double difference = a > b ? a - b : b - a;
    const long double scale = (b < 0 ? -b : b) + 1e-300L;
    return difference <= scale * 1e-15L;
}

static void Run() {
    Check(Near(acosl(Opaque(0.5L)), 1.04719755119659774615L), 0);
    Check(Near(atanl(Opaque(1.0L)), 0.78539816339744830962L), 0);
    Check(Near(coshl(Opaque(1.0L)), 1.54308063481524377848L), 1);
    Check(Near(cosl(Opaque(0.0L)), 1.0L) && Near(sinl(Opaque(0.5L)), 0.47942553860420300027L), 1);
    Check(exp2l(Opaque(10.0L)) == 1024.0L && Near(expl(Opaque(1.0L)), 2.71828182845904523536L), 2);
    int exponent = 0;
    Check(frexpl(Opaque(48.0L), &exponent) == 0.75L && exponent == 6, 3);
    Check(ldexpl(Opaque(0.75L), Opaque(6)) == 48.0L, 3);
    Check(log10l(Opaque(1000.0L)) == 3.0L && Near(logl(Opaque(10.0L)), 2.30258509299404568402L), 4);
    Check(powl(Opaque(2.0L), Opaque(-3.0L)) == 0.125L, 4);
    Check(Near(sinhl(Opaque(1.0L)), 1.17520119364380145688L) && Near(tanhl(Opaque(0.5L)), 0.46211715726000975850L), 5);
    Check(Near(tanl(Opaque(1.0L)), 1.55740772465490223050L), 5);
    Check(truncl(Opaque(-2.75L)) == -2.0L && truncl(Opaque(0x1p1000L)) == 0x1p1000L, 6);
    // Special values and the subnormal range survive the x87 <-> double conversion.
    Check(__builtin_isinf(expl(Opaque(1e6L))) && __builtin_isnan(logl(Opaque(-1.0L))), 7);
    Check(truncl(Opaque(-0.0L)) == 0.0L && __signbitl(truncl(Opaque(-0.0L))) != 0, 7);
    Check(ldexpl(Opaque(1.0L), Opaque(-1074)) == 0x1p-1074L && ldexpl(Opaque(0x1.8p-1070L), Opaque(-1)) == 0x1.8p-1071L, 8);

    char* end = nullptr;
    const char text[] = "1.5e3xyz";
    Check(strtold(text, &end) == 1500.0L && end == text + 5, 9);
    const char16_t wide[] = u"-0.25";
    Check(wcstold(wide, nullptr) == -0.25L, 10);
    Check(__signbitl(Opaque(-3.0L)) != 0 && __signbitl(Opaque(3.0L)) == 0, 11);

    Check(Aps5AbiLongDouble(Opaque(1.5L), Opaque(-4), Opaque(0.25L)) == -5.75L, 12);
}

ABI_MAIN(Run())
