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

}

#endif
