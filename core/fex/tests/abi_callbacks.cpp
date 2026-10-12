// Native code calling guest functions it got as arguments: integers past the sixth (stack on the guest,
// x6/x7 on arm64), a float HFA, long double and doubles past the eighth, a hidden-pointer struct result,
// and a long double result. A stored callback the library hands back is the guest's own address.
#include "abi.h"

static int SumInts(int a, int b, int c, int d, int e, int f, int g, int h) {
    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h;
}

static double Shaped(Position position, long double half, int seven, float quarter, double d1, double d2, double d3, double d4,
                     double d5, double d6, double d7) {
    return position.x + 10 * position.y + 100 * position.z + 1000 * static_cast<double>(half) + 10000 * seven + 100000 * quarter +
           d1 + 2 * d2 + 3 * d3 + 4 * d4 + 5 * d5 + 6 * d6 + 7 * d7;
}

static Large MakeLarge(Large large, int add) {
    return {large.a + add, large.b + add, large.c + add};
}

static long double Scale(long double value, int count) {
    return value * count;
}

static int Twice(int value) {
    return 2 * value;
}

static void Run() {
    Check(Aps5AbiCallInts(SumInts, 1) == 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64, 0);
    const double shaped = Aps5AbiCallShaped(Shaped, 1.0f);
    Check(shaped == 1 + 20 + 300 + 500 + 70000 + 25000 + 1 + 4 + 9 + 16 + 25 + 36 + 49, 1);
    const Large large = Aps5AbiCallLarge(MakeLarge, 4);
    Check(large.a == 50 && large.b == 60 && large.c == 70, 2);
    Check(Aps5AbiCallLongDouble(Scale, 3) == 7.5L, 3);
    Aps5AbiStore(Twice);
    Check(Aps5AbiLoad() == Twice, 4);
    Check(Aps5AbiCallStored(21) == 42, 5);
    int (*loaded)(int) = Aps5AbiLoad();
    Check(loaded(5) == 10, 6);
}

ABI_MAIN(Run())
