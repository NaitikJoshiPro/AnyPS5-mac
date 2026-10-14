#include <cstdint>
#include <stdexcept>
#include "SceTypes.hpp"

// __tls_get_addr for guest code bound to libkernel's export (NID vNe1w4diLCs): a tls_index {module, offset}
// to the address of that offset in the calling thread's block of that module. Guest modules normally import
// it by name and the runner binds that name to its own resolver; this export serves every other import of
// the NID (an executable importing it from libkernel, a module the relinker leaves bound to the NID).
//
// arm64 macOS: the FEXCore runner lays every guest image's block out below the guest thread pointer and
// numbers the modules (DTPMOD64) itself, so the resolver is the runner's, registered at start-up.
// x86-64 macOS: the macOS relinker writes the image's TLS descriptor as the module id, which
// Aps5GuestTlsGetAddr (GuestTlsMacOs.cpp) resolves.

namespace {
#if defined(__APPLE__) && defined(__aarch64__)
void* (*guestResolver)(const std::uint64_t*) = nullptr;
#endif
}

extern "C" {

#if defined(__APPLE__) && defined(__aarch64__)
void Aps5SetGuestTlsResolver_nid_no_patch(void* (*resolver)(const std::uint64_t*)) {
    guestResolver = resolver;
}
#elif defined(__APPLE__) && defined(__x86_64__)
void* Aps5GuestTlsGetAddr_nid_no_patch(const void* index);
#endif

void* APS5_VABI __tls_get_addr_nid_postfix(const std::uint64_t* index) {
    if (index == nullptr) throw std::invalid_argument("__tls_get_addr: null tls_index");
#if defined(__APPLE__) && defined(__aarch64__)
    if (guestResolver == nullptr) throw std::runtime_error("__tls_get_addr: no runner registered a guest TLS resolver");
    return guestResolver(index);
#elif defined(__APPLE__) && defined(__x86_64__)
    return Aps5GuestTlsGetAddr_nid_no_patch(index);
#else
    throw std::runtime_error("__tls_get_addr: libkernel resolves guest dynamic TLS on macOS only");
#endif
}

}
