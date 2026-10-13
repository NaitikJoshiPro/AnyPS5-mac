// INTERIM (M3, until the upstream sync brings File/src/DescriptorControl.cpp): fcntl F_DUPFD / F_DUPFD_CLOEXEC on a
// host-backed descriptor (a regular file or device node). This branch's fcntl handles guest sockets only and failed
// these with EBADF, which stops oops-mesa's DRI screen (it duplicates its device fd with F_DUPFD_CLOEXEC). Delete this
// file and its call in Socket/src/Udp.cpp when the upstream fcntl lands.
#include "prx/libkernel/File/include/RandomDevice.hpp"
#include "prx/libkernel/Socket/include/SocketRuntime.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include <cerrno>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#ifndef _WIN32
#include <fcntl.h>
#endif

extern "C" int* APS5_VABI __error_nid_postfix();

extern "C" int FileDuplicateFcntl_nid_no_patch(int descriptor, int command, std::int64_t minimum) {
    constexpr int GuestDuplicate = 0, GuestDuplicateCloseOnExec = 17, GuestEbadf = 9, GuestEinval = 22, GuestEio = 5;
    const auto fail = [](int error) {
        *__error_nid_postfix() = error;
        return -1;
    };
    if (descriptor < 0 || descriptor >= GuestSockets::FirstDescriptor) return fail(GuestEbadf);
    if (minimum < 0 || minimum >= GuestSockets::FirstDescriptor) return fail(GuestEinval);
#ifdef _WIN32
    (void)command;
    throw std::runtime_error("fcntl: F_DUPFD on a host descriptor is not implemented on Windows");
#else
    if (command != GuestDuplicate && command != GuestDuplicateCloseOnExec) throw std::runtime_error("fcntl: unexpected command " + std::to_string(command));
    const int duplicate = ::fcntl(descriptor, command == GuestDuplicate ? F_DUPFD : F_DUPFD_CLOEXEC, static_cast<int>(minimum));
    if (duplicate < 0) return fail(errno > 0 && errno <= 34 ? errno : GuestEio);
    if (duplicate >= GuestSockets::FirstDescriptor) throw std::runtime_error("fcntl: host descriptor reached the guest socket range");
    if (File::IsRandomDevice(descriptor)) File::RememberRandomDevice(duplicate);
    return duplicate;
#endif
}
