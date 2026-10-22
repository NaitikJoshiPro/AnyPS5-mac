#include "prx/libc/include/general/VabiMacros.hpp"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#if defined(__x86_64__)
#include <xmmintrin.h>

namespace {

constexpr unsigned Exceptions = 0x3f;
constexpr unsigned Rounding = 0xc00;

void ReplaceFlags(unsigned mask, unsigned flags) {
    std::array<std::uint32_t, 7> environment;
    asm volatile("fnstenv %0" : "=m"(environment));
    environment[1] = (environment[1] & ~mask) | (flags & mask);
    asm volatile("fldenv %0" : : "m"(environment));
    _mm_setcsr((_mm_getcsr() & ~mask) | (flags & mask));
}

int ChangeTraps(int exceptions, bool enable) {
    const auto mask = static_cast<unsigned>(exceptions) & Exceptions;
    std::uint16_t control;
    asm volatile("fnstcw %0" : "=m"(control));
    const unsigned csr = _mm_getcsr();
    const auto changed = static_cast<std::uint16_t>(enable ? (control & ~mask) : (control | mask));
    asm volatile("fldcw %0" : : "m"(changed));
    _mm_setcsr(enable ? (csr & ~(mask << 7)) : (csr | (mask << 7)));
    return static_cast<int>(~((csr >> 7) | control) | ~Exceptions);
}

}

extern "C" {

int APS5_VABI fegetround_nid_postfix() {
    std::uint16_t control;
    asm volatile("fnstcw %0" : "=m"(control));
    return control & Rounding;
}

int APS5_VABI fesetround_nid_postfix(int rounding) {
    const auto mode = static_cast<unsigned>(rounding);
    if ((mode & ~Rounding) != 0) return -1;
    std::uint16_t control;
    asm volatile("fnstcw %0" : "=m"(control));
    control = static_cast<std::uint16_t>((control & ~Rounding) | mode);
    asm volatile("fldcw %0" : : "m"(control));
    _mm_setcsr((_mm_getcsr() & ~(Rounding << 3)) | (mode << 3));
    return 0;
}

int APS5_VABI fetestexcept_nid_postfix(int exceptions) {
    std::uint16_t status;
    asm volatile("fnstsw %0" : "=am"(status));
    return (status | _mm_getcsr()) & static_cast<unsigned>(exceptions) & Exceptions;
}

int APS5_VABI fegetexceptflag_nid_postfix(std::uint16_t* flags, int exceptions) {
    if (flags == nullptr) throw std::invalid_argument("fegetexceptflag: null flags");
    *flags = static_cast<std::uint16_t>(fetestexcept_nid_postfix(exceptions));
    return 0;
}

int* APS5_VABI __error_nid_postfix();

void APS5_VABI _Feraise_nid_postfix(int exceptions) {
    constexpr unsigned Invalid = 0x01, Ranges = 0x1c, Overflows = 0x18, Inexact = 0x20;
    constexpr int GuestEdom = 33, GuestErange = 34;
    auto raised = static_cast<unsigned>(exceptions);
    if ((raised & Overflows) != 0) raised |= Inexact;
    *__error_nid_postfix() = GuestEdom;
    std::array<std::uint32_t, 7> environment;
    asm volatile("fnstenv %0" : "=m"(environment));
    environment[1] = (environment[1] & ~raised) | (raised & 0xffff);
    asm volatile("fldenv %0" : : "m"(environment));
    _mm_setcsr((_mm_getcsr() & ~raised) | (raised & 0xffff));
    asm volatile("fwait");
    if ((raised & Invalid) != 0) *__error_nid_postfix() = GuestEdom;
    else if ((raised & Ranges) != 0) *__error_nid_postfix() = GuestErange;
}

int APS5_VABI fesetexceptflag_nid_postfix(const std::uint16_t* flags, int exceptions) {
    if (flags == nullptr) throw std::invalid_argument("fesetexceptflag: null flags");
    ReplaceFlags(static_cast<unsigned>(exceptions) & Exceptions, *flags);
    return 0;
}

int APS5_VABI feclearexcept_nid_postfix(int exceptions) {
    const auto mask = static_cast<unsigned>(exceptions) & Exceptions;
    if (mask == Exceptions) {
        asm volatile("fnclex");
        _mm_setcsr(_mm_getcsr() & ~Exceptions);
    } else if (mask != 0) {
        ReplaceFlags(mask, 0);
    }
    return 0;
}


int APS5_VABI feenableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, true);
}

int APS5_VABI fedisableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, false);
}

int APS5_VABI __feenableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, true);
}

int APS5_VABI __fedisableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, false);
}

int APS5_VABI fegettrapenable_nid_postfix() {
    std::uint16_t control;
    asm volatile("fnstcw %0" : "=m"(control));
    return ~control & Exceptions;
}

int APS5_VABI fesettrapenable_nid_postfix(int exceptions) {
    const auto mask = static_cast<std::uint16_t>(exceptions);
    if (mask > Exceptions) return 1;
    std::uint16_t control;
    asm volatile("fnstcw %0" : "=m"(control));
    control = static_cast<std::uint16_t>(control & ~mask);
    asm volatile("fldcw %0" : : "m"(control));
    _mm_setcsr(_mm_getcsr() & ~(static_cast<unsigned>(mask) << 7));
    return 0;
}

}

#else

// arm64 host: the guest's x87/SSE state lives in the translator, whose bridge reports the guest's control
// words read-only (LibcGuestControl, as fegetenv in Math.cpp). Reads come from it; changing the guest's
// rounding mode or exception flags has no bridge yet and throws (docs/dev/TechnicalDebt.md
// arm64-guest-fenv-writes).
extern "C" void LibcGuestControl(std::uint32_t* mxcsr, std::uint16_t* fcw);

namespace {

constexpr unsigned Exceptions = 0x3f;
constexpr unsigned Rounding = 0xc00;

[[noreturn]] void NoGuestFenvWrite(const char* function) {
    throw std::runtime_error(std::string(function) + ": changing the guest's floating-point environment is not implemented on the arm64 host");
}

}

extern "C" {

int APS5_VABI fegetround_nid_postfix() {
    std::uint32_t mxcsr = 0;
    std::uint16_t fcw = 0;
    LibcGuestControl(&mxcsr, &fcw);
    return fcw & Rounding;
}

int APS5_VABI fesetround_nid_postfix(int rounding) {
    const auto mode = static_cast<unsigned>(rounding);
    if ((mode & ~Rounding) != 0) return -1;
    if (static_cast<unsigned>(fegetround_nid_postfix()) == mode) return 0;
    NoGuestFenvWrite("fesetround");
}

int APS5_VABI fetestexcept_nid_postfix(int exceptions) {
    std::uint32_t mxcsr = 0;
    std::uint16_t fcw = 0;
    LibcGuestControl(&mxcsr, &fcw);
    return mxcsr & static_cast<unsigned>(exceptions) & Exceptions;
}

int APS5_VABI fegetexceptflag_nid_postfix(std::uint16_t* flags, int exceptions) {
    if (flags == nullptr) throw std::invalid_argument("fegetexceptflag: null flags");
    *flags = static_cast<std::uint16_t>(fetestexcept_nid_postfix(exceptions));
    return 0;
}

int APS5_VABI fesetexceptflag_nid_postfix(const std::uint16_t* flags, int exceptions) {
    if (flags == nullptr) throw std::invalid_argument("fesetexceptflag: null flags");
    if ((static_cast<unsigned>(exceptions) & Exceptions) == 0) return 0;
    NoGuestFenvWrite("fesetexceptflag");
}

int APS5_VABI feclearexcept_nid_postfix(int exceptions) {
    const auto mask = static_cast<unsigned>(exceptions) & Exceptions;
    if (mask == 0 || fetestexcept_nid_postfix(static_cast<int>(mask)) == 0) return 0;
    NoGuestFenvWrite("feclearexcept");
}

int* APS5_VABI __error_nid_postfix();

// As on x86-64: errno follows the raised class; the flags themselves can only be raised when already set.
void APS5_VABI _Feraise_nid_postfix(int exceptions) {
    constexpr unsigned Invalid = 0x01, Ranges = 0x1c, Overflows = 0x18, Inexact = 0x20;
    constexpr int GuestEdom = 33, GuestErange = 34;
    auto raised = static_cast<unsigned>(exceptions);
    if ((raised & Overflows) != 0) raised |= Inexact;
    const auto flags = raised & Exceptions;
    if (flags != 0 && static_cast<unsigned>(fetestexcept_nid_postfix(static_cast<int>(flags))) != flags) NoGuestFenvWrite("_Feraise");
    if ((raised & Invalid) != 0) *__error_nid_postfix() = GuestEdom;
    else if ((raised & Ranges) != 0) *__error_nid_postfix() = GuestErange;
    else *__error_nid_postfix() = GuestEdom;
}

}

namespace {

// The traps enabled on both units before the change, as the x86-64 ChangeTraps reports them; a change throws.
int ChangeTraps(int exceptions, bool enable, const char* function) {
    const auto mask = static_cast<unsigned>(exceptions) & Exceptions;
    std::uint32_t mxcsr = 0;
    std::uint16_t fcw = 0;
    LibcGuestControl(&mxcsr, &fcw);
    const unsigned x87Masked = fcw & mask;
    const unsigned sseMasked = (mxcsr >> 7) & mask;
    if (enable ? (x87Masked != 0 || sseMasked != 0) : (x87Masked != mask || sseMasked != mask)) NoGuestFenvWrite(function);
    return static_cast<int>(~((mxcsr >> 7) | fcw) | ~Exceptions);
}

}

extern "C" {

int APS5_VABI feenableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, true, "feenableexcept");
}

int APS5_VABI fedisableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, false, "fedisableexcept");
}

int APS5_VABI __feenableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, true, "__feenableexcept");
}

int APS5_VABI __fedisableexcept_nid_postfix(int exceptions) {
    return ChangeTraps(exceptions, false, "__fedisableexcept");
}

int APS5_VABI fegettrapenable_nid_postfix() {
    std::uint32_t mxcsr = 0;
    std::uint16_t fcw = 0;
    LibcGuestControl(&mxcsr, &fcw);
    return ~fcw & Exceptions;
}

int APS5_VABI fesettrapenable_nid_postfix(int exceptions) {
    const auto mask = static_cast<std::uint16_t>(exceptions);
    if (mask > Exceptions) return 1;
    std::uint32_t mxcsr = 0;
    std::uint16_t fcw = 0;
    LibcGuestControl(&mxcsr, &fcw);
    if ((fcw & mask) != 0 || ((mxcsr >> 7) & mask) != 0) NoGuestFenvWrite("fesettrapenable");
    return 0;
}

}

#endif
