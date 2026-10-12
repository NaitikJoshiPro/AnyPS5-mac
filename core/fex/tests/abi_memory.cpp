// Structs in memory: a 24-byte struct by value (copied to the guest stack, passed by reference on arm64)
// and results through a hidden pointer (rdi in and rax out on the guest, x8 on arm64), including libc's
// std::current_exception, whose exception_ptr is not trivially copyable.
#include "abi.h"

struct ExceptionPointer {
    void* object;
    ~ExceptionPointer() {}
};

#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern "C" ExceptionPointer _ZSt17current_exceptionv();

static void Run() {
    const Large large = Aps5AbiLarge(Opaque(5), Large {Opaque(10LL), Opaque(-20LL), Opaque(1LL << 40)});
    Check(large.a == 15 && large.b == -15 && large.c == (1LL << 40) + 5, 0);

    // The guest passes rdi for the result and expects it back in rax: the compiler reads the result
    // through rax here.
    Large (*volatile function)(int, Large) = Aps5AbiLarge;
    const Large again = function(1, Large {2, 3, 4});
    Check(again.a == 3 && again.b == 4 && again.c == 5, 1);

    ExceptionPointer none = _ZSt17current_exceptionv();
    Check(none.object == nullptr, 2);
}

ABI_MAIN(Run())
