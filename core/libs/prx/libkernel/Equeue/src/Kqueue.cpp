#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/event.h>
#include <time.h>

#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libkernel/Socket/include/SocketRuntime.hpp"

static constexpr std::int16_t GuestFilterVnode = -4;
static constexpr std::uint16_t GuestChangeFlags = 0x00ff;
static constexpr std::uint32_t GuestVnodeNotes = 0x7f;
static constexpr int GuestEfault = 14;
static constexpr int LastSharedError = 34;

static_assert(EVFILT_VNODE == GuestFilterVnode);
static_assert((EV_ADD | EV_DELETE | EV_ENABLE | EV_DISABLE | EV_ONESHOT | EV_CLEAR | EV_RECEIPT | EV_DISPATCH) == GuestChangeFlags);
static_assert((NOTE_DELETE | NOTE_WRITE | NOTE_EXTEND | NOTE_ATTRIB | NOTE_LINK | NOTE_RENAME | NOTE_REVOKE) == GuestVnodeNotes);

extern "C" int* APS5_VABI __error_nid_postfix();

static int Failure(const char* function, int error) {
    if (error <= 0 || error > LastSharedError) {
        throw std::runtime_error(std::string(function) + ": unexpected host error " + std::to_string(error));
    }
    *__error_nid_postfix() = error;
    return -1;
}

static struct kevent HostChange(const KernelEvent& change) {
    if (change.filter != GuestFilterVnode) {
        throw std::runtime_error("kevent: filter " + std::to_string(change.filter) + " is not supported");
    }
    if ((change.flags & ~GuestChangeFlags) != 0) {
        throw std::runtime_error("kevent: flags " + std::to_string(change.flags) + " are not supported");
    }
    if ((change.fflags & ~GuestVnodeNotes) != 0) {
        throw std::runtime_error("kevent: vnode notes " + std::to_string(change.fflags) + " are not supported");
    }
    if (change.ident >= static_cast<std::uintptr_t>(GuestSockets::FirstDescriptor)) {
        throw std::runtime_error("kevent: guest sockets are not supported");
    }
    struct kevent host;
    EV_SET(&host, change.ident, EVFILT_VNODE, change.flags, change.fflags, change.data, change.udata);
    return host;
}

static KernelEvent GuestEvent(const struct kevent& host) {
    if ((host.flags & EV_ERROR) != 0 && (host.data < 0 || host.data > LastSharedError)) {
        throw std::runtime_error("kevent: unexpected host error " + std::to_string(host.data));
    }
    KernelEvent guest;
    guest.ident = host.ident;
    guest.filter = host.filter;
    guest.flags = host.flags;
    guest.fflags = host.fflags;
    guest.data = host.data;
    guest.udata = host.udata;
    return guest;
}

extern "C" {

int APS5_VABI kqueue_nid_postfix() {
    const int queue = ::kqueue();
    return queue < 0 ? Failure("kqueue", errno) : queue;
}

int APS5_VABI kevent_nid_postfix(int queue, const KernelEvent* changes, int changeCount, KernelEvent* events, int eventCount, const KernelTimespec* timeout) {
    if ((changeCount > 0 && changes == nullptr) || (eventCount > 0 && events == nullptr)) {
        return Failure("kevent", GuestEfault);
    }
    std::vector<struct kevent> hostChanges;
    for (int index = 0; index < changeCount; ++index) {
        hostChanges.push_back(HostChange(changes[index]));
    }
    std::vector<struct kevent> hostEvents(eventCount > 0 ? static_cast<std::size_t>(eventCount) : 0);
    struct timespec hostTimeout{};
    if (timeout != nullptr) {
        hostTimeout.tv_sec = timeout->tv_sec;
        hostTimeout.tv_nsec = timeout->tv_nsec;
    }
    const int count = ::kevent(queue, hostChanges.data(), changeCount, hostEvents.data(), eventCount, timeout != nullptr ? &hostTimeout : nullptr);
    if (count < 0) {
        return Failure("kevent", errno);
    }
    for (int index = 0; index < count; ++index) {
        events[index] = GuestEvent(hostEvents[index]);
    }
    return count;
}

}
