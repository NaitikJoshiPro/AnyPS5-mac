// Arguments and results in registers that the positional bridge places wrongly: a float HFA by value
// (xmm0/xmm1 against s0-s2) and every float after it, a mixed double/integer struct, __int128 after an int
// (arm64 starts it at an even register), an HFA result, and double arguments past the eighth.
#include "abi.h"

static void Run() {
    float out[7] = {};
    const int handle = Aps5AbiHfa(reinterpret_cast<void*>(0x1234), Position {1.5f, -2.25f, 3.0f}, 0.75f, out, 9, 200, -4.5f);
    Check(handle == 0x1234, 0);
    Check(out[0] == 1.5f && out[1] == -2.25f && out[2] == 3.0f, 1);
    Check(out[3] == 0.75f && out[6] == -4.5f, 2);
    Check(out[4] == 9.0f && out[5] == 200.0f, 3);

    const Mixed mixed = Aps5AbiMixed(Mixed {Opaque(1.25), Opaque(40LL)}, Opaque(0.5f), Pair {Opaque(1LL), Opaque(-3LL)});
    Check(mixed.d == 3.0 && mixed.i == 38, 4);

    const __int128 big = (static_cast<__int128>(0x1122334455667788LL) << 64) | 0x99aabbccddeeff00ULL;
    Check(Aps5AbiInt128(Opaque(5), Opaque(big), Opaque(-2)) == big + 3, 5);

    const Position position = Aps5AbiPosition(Opaque(1.5f));
    Check(position.x == 1.5f && position.y == 3.0f && position.z == 4.5f, 6);

    const double many = Aps5AbiManyDoubles(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, Opaque(2));
    Check(many == 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 + 100 + 22, 7);
}

ABI_MAIN(Run())
