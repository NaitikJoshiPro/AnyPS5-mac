#ifndef PRX_LIBC_INCLUDE_SPECIFICS_MACOS_GUESTIMAGES_HPP
#define PRX_LIBC_INCLUDE_SPECIFICS_MACOS_GUESTIMAGES_HPP

#ifdef __APPLE__
#include "prx/libc/include/specifics/linux/ElfTypes.hpp"

namespace GuestAllocations {
// Registers the PT_LOAD pages of an image the arm64 runner mapped itself (which dyld does not know) as
// guest image memory: not releasable, readable and writable as its program headers say.
void RegisterRunnerImage(const dl_phdr_info& image);
}

#ifdef __aarch64__
extern "C" {
// What the guest may call for a native export: on the arm64 runner the bridge's x86 stub for a function,
// the address itself for data, and null when no PS5 library holds the address. Without a runner the
// address is returned as it is.
void* LibcGuestCallable_nid_no_patch(const void* address, const char* name);
}
#endif
#endif

#endif
