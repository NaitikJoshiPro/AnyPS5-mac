// fcntl F_DUPFD / F_DUPFD_CLOEXEC / F_GETFD / F_SETFD on a host-backed descriptor (INTERIM forwarding, File/src/FileDuplicate.cpp): the
// duplicate is at or above the requested minimum, shares the file offset, and F_DUPFD_CLOEXEC sets FD_CLOEXEC.
#include "prx/libc/include/general/VabiMacros.hpp"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

extern "C" {
int APS5_VABI fcntl_nid_postfix(int, int, ...);
int* APS5_VABI __error_nid_postfix();
}

namespace {
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
#ifdef _WIN32
    std::puts("guest fcntl duplicate: host descriptors are not forwarded on Windows, skipped");
    return 0;
#else
    try {
        char path[] = "/tmp/aps5-fcntl-dupXXXXXX";
        const int file = mkstemp(path);
        Require(file >= 0, "mkstemp");
        unlink(path);
        Require(write(file, "abcdef", 6) == 6, "write");
        const int duplicate = fcntl_nid_postfix(file, 17, std::int64_t{40}); // F_DUPFD_CLOEXEC, FreeBSD numbering
        Require(duplicate >= 40, "F_DUPFD_CLOEXEC returned a descriptor below the minimum");
        Require((::fcntl(duplicate, F_GETFD) & FD_CLOEXEC) != 0, "F_DUPFD_CLOEXEC did not set FD_CLOEXEC");
        Require(lseek(duplicate, 0, SEEK_CUR) == 6, "the duplicate does not share the file offset");
        Require(fcntl_nid_postfix(duplicate, 1) == 1, "F_GETFD did not report FD_CLOEXEC");
        Require(fcntl_nid_postfix(duplicate, 2, std::int64_t{0}) == 0 && fcntl_nid_postfix(duplicate, 1) == 0, "F_SETFD did not clear FD_CLOEXEC");
        const int plain = fcntl_nid_postfix(file, 0, std::int64_t{0}); // F_DUPFD
        Require(plain >= 0 && (::fcntl(plain, F_GETFD) & FD_CLOEXEC) == 0, "F_DUPFD failed or set FD_CLOEXEC");
        close(plain);
        close(duplicate);
        close(file);
        Require(fcntl_nid_postfix(file, 17, std::int64_t{0}) == -1 && *__error_nid_postfix() == 9, "F_DUPFD_CLOEXEC on a closed descriptor did not fail with EBADF");
        Require(fcntl_nid_postfix(file, 1) == -1 && *__error_nid_postfix() == 9, "F_GETFD on a closed descriptor did not fail with EBADF");
        std::puts("guest fcntl duplicate tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "guest fcntl duplicate: %s\n", error.what());
        return 1;
    }
#endif
}
