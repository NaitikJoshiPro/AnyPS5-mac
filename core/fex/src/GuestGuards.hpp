#ifndef CORE_FEX_SRC_GUESTGUARDS_HPP
#define CORE_FEX_SRC_GUESTGUARDS_HPP

#include <cstddef>

namespace FEXCore::Context {
class Context;
}

namespace Aps5Fex {

class Bridge;
class GuestProgram;

// Static-initialisation guards (Itanium C++ ABI __cxa_guard_acquire/release/abort) that a guest
// image defines itself are redirected to libc's thread-safe ones. Homebrew links libc++abi built
// without threads (oops-apps: _LIBCXXABI_HAS_NO_THREADS), whose acquire aborts with "detected
// recursive initialization" when a second thread arrives while another runs the initialiser.
// An image is redirected only when it exports all three (by name, or by NID as a PS5 executable does). Returns the number of images redirected.
std::size_t RedirectGuestGuards(Bridge& bridge, const GuestProgram& program, FEXCore::Context::Context& context);

}

#endif
