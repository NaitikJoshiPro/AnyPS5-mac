#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTMACHINESTATE_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTMACHINESTATE_HPP

#include <cstddef>
#include <cstdint>

// The guest's whole x86-64 user state at the innermost guest call into a library, as the guest sees it just
// after the call returns: general registers in DWARF order (rax, rdx, rcx, rbx, rsi, rdi, rbp, rsp, r8-r15),
// rip (the return address), rflags, an FXSAVE image (x87 control/status/tags and st0-7, MXCSR, xmm0-15) and the
// fs base. On arm64 the registers live in the translator: the FEXCore runner hands libc hooks that read this
// state and resume the guest with a new one (Aps5SetBridgeMachineState_nid_no_patch).
struct GuestMachineState {
    std::uint64_t gregs[16];
    std::uint64_t rip;
    std::uint64_t rflags;
    alignas(16) std::uint8_t fxsave[512];
    std::uint64_t fsBase;
};

namespace GuestMachine {

enum Register : std::size_t { Rax = 0, Rdx = 1, Rcx = 2, Rbx = 3, Rsi = 4, Rdi = 5, Rbp = 6, Rsp = 7, R8 = 8, R9 = 9, R10 = 10,
                              R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15 };

inline constexpr std::size_t FxsaveMxcsr = 24;
inline constexpr std::size_t FxsaveMxcsrMask = 28;
inline constexpr std::size_t FxsaveX87 = 32;
inline constexpr std::size_t FxsaveXmm = 160;
inline constexpr std::uint32_t MxcsrMask = 0xffbf;

}

#if defined(__APPLE__) && defined(__aarch64__)
// libc's hooks for the other libraries (a patched library exports only NIDs and _nid_no_patch names).
// Whether the runner provides machine state.
extern "C" bool LibcHasMachineState_nid_no_patch();
// The state of the guest call the calling library function serves.
extern "C" void LibcGetMachineState_nid_no_patch(GuestMachineState* state);
// Resumes the guest with state instead of returning to it.
extern "C" [[noreturn]] void LibcSetMachineState_nid_no_patch(const GuestMachineState* state);
// The guest address makecontext points a context at: guest code that calls the context's function with its
// arguments (rdi = context, rsi = function, rdx = six arguments), then Aps5MakecontextFinish_nid_no_patch.
extern "C" std::uint64_t LibcMakecontextEntry_nid_no_patch();
#endif

#endif
