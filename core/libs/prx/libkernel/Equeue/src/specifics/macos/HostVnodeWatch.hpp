#ifndef CORE_LIBS_PRX_LIBKERNEL_EQUEUE_SRC_SPECIFICS_MACOS_HOSTVNODEWATCH_HPP
#define CORE_LIBS_PRX_LIBKERNEL_EQUEUE_SRC_SPECIFICS_MACOS_HOSTVNODEWATCH_HPP

// macOS host backend for guest EVFILT_VNODE watches (FileEvent.cpp).
// It lives in its own translation unit because <sys/event.h> defines EVFILT_*, EV_* and NOTE_* as
// macros, which collide with the guest constants in Equeue.hpp. Keep this header free of both.

#include <cstdint>
#include <functional>
#include <memory>

namespace HostVnode {

class Watch;

// Open() result for a descriptor that is neither a regular file nor a directory.
constexpr int UNSUPPORTED_DESCRIPTOR = -1;

// Starts watching the host file or directory behind a guest descriptor. Returns 0, a guest errno
// (EBADF, EINVAL for pipes and sockets) or UNSUPPORTED_DESCRIPTOR. Host failures throw std::runtime_error.
int Open(int descriptor, std::shared_ptr<Watch>& watch);

// Blocks until stopDescriptor becomes readable or the watched vnode is revoked, passing guest
// NOTE_* hints to deliver with the same meaning the Linux inotify backend gives them.
void Run(Watch& watch, int stopDescriptor, const std::function<void(std::uint32_t)>& deliver);

}

#endif
