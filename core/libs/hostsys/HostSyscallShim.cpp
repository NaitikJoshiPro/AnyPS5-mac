// Host system call shim (D35). Linked as a static library into a prx (APS5_HOST_SYSCALL_SHIM), its hidden definitions
// of read, write, recv, ... take the place of libSystem's for every object in that library, the third-party code
// linked into it included. Each translates guest buffers in the displaced window (D31, D33) to their alias and calls
// libSystem's function: the kernel returns EFAULT for the guest address and never raises the signal the fault handler
// redirects. Host addresses, aliases included, pass through after one compare. A buffer across an edge of the window
// is passed unchanged, and the kernel still returns EFAULT for it.
#include "prx/libc/include/GuestDisplaced.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#if defined(__APPLE__) && defined(__aarch64__)

namespace {

template <typename T>
T* Host(T* guest, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(guest);
    if (!GuestArena::MayBeDisplaced(address) || bytes == 0) return guest;
    const auto start = GuestArena::DisplacedStart();
    if (address < start) return guest;
    if (bytes > GuestArena::DisplacedEnd - address) return guest;
    return reinterpret_cast<T*>(address | GuestArena::DisplacedFlip);
}

void* LibSystem(const char* name) {
    static void* const library = dlopen("/usr/lib/libSystem.B.dylib", RTLD_LAZY | RTLD_NOLOAD);
    void* symbol = library != nullptr ? dlsym(library, name) : nullptr;
    if (symbol == nullptr) throw std::runtime_error("host system call shim: libSystem has no " + std::string(name));
    return symbol;
}

struct Real {
    decltype(&::read) read = reinterpret_cast<decltype(&::read)>(LibSystem("read"));
    decltype(&::write) write = reinterpret_cast<decltype(&::write)>(LibSystem("write"));
    decltype(&::pread) pread = reinterpret_cast<decltype(&::pread)>(LibSystem("pread"));
    decltype(&::pwrite) pwrite = reinterpret_cast<decltype(&::pwrite)>(LibSystem("pwrite"));
    decltype(&::readv) readv = reinterpret_cast<decltype(&::readv)>(LibSystem("readv"));
    decltype(&::writev) writev = reinterpret_cast<decltype(&::writev)>(LibSystem("writev"));
    decltype(&::preadv) preadv = reinterpret_cast<decltype(&::preadv)>(LibSystem("preadv"));
    decltype(&::pwritev) pwritev = reinterpret_cast<decltype(&::pwritev)>(LibSystem("pwritev"));
    decltype(&::recv) recv = reinterpret_cast<decltype(&::recv)>(LibSystem("recv"));
    decltype(&::recvfrom) recvfrom = reinterpret_cast<decltype(&::recvfrom)>(LibSystem("recvfrom"));
    decltype(&::send) send = reinterpret_cast<decltype(&::send)>(LibSystem("send"));
    decltype(&::sendto) sendto = reinterpret_cast<decltype(&::sendto)>(LibSystem("sendto"));
    decltype(&::getentropy) getentropy = reinterpret_cast<decltype(&::getentropy)>(LibSystem("getentropy"));
    decltype(&::fread) fread = reinterpret_cast<decltype(&::fread)>(LibSystem("fread"));
    decltype(&::fwrite) fwrite = reinterpret_cast<decltype(&::fwrite)>(LibSystem("fwrite"));
};

const Real& Functions() {
    static const Real real;
    return real;
}

template <typename TCall>
ssize_t WithVectors(const iovec* vectors, int count, TCall&& call) {
    const iovec* given = Host(vectors, count > 0 ? sizeof(iovec) * static_cast<std::size_t>(count) : 0);
    if (count <= 0 || count > IOV_MAX) return call(given);
    bool translate = false;
    for (int i = 0; i < count && !translate; ++i)
        translate = Host(given[i].iov_base, given[i].iov_len) != given[i].iov_base;
    if (!translate) return call(given);
    std::array<iovec, IOV_MAX> copied;
    for (int i = 0; i < count; ++i) copied[i] = {Host(given[i].iov_base, given[i].iov_len), given[i].iov_len};
    return call(copied.data());
}

}

extern "C" {

#define APS5_HOST_SHIM __attribute__((visibility("hidden")))

APS5_HOST_SHIM ssize_t read(int fd, void* buffer, std::size_t bytes) {
    return Functions().read(fd, Host(buffer, bytes), bytes);
}

APS5_HOST_SHIM ssize_t write(int fd, const void* buffer, std::size_t bytes) {
    return Functions().write(fd, Host(buffer, bytes), bytes);
}

APS5_HOST_SHIM ssize_t pread(int fd, void* buffer, std::size_t bytes, off_t offset) {
    return Functions().pread(fd, Host(buffer, bytes), bytes, offset);
}

APS5_HOST_SHIM ssize_t pwrite(int fd, const void* buffer, std::size_t bytes, off_t offset) {
    return Functions().pwrite(fd, Host(buffer, bytes), bytes, offset);
}

APS5_HOST_SHIM ssize_t readv(int fd, const iovec* vectors, int count) {
    return WithVectors(vectors, count, [&](const iovec* host) { return Functions().readv(fd, host, count); });
}

APS5_HOST_SHIM ssize_t writev(int fd, const iovec* vectors, int count) {
    return WithVectors(vectors, count, [&](const iovec* host) { return Functions().writev(fd, host, count); });
}

APS5_HOST_SHIM ssize_t preadv(int fd, const iovec* vectors, int count, off_t offset) {
    return WithVectors(vectors, count, [&](const iovec* host) { return Functions().preadv(fd, host, count, offset); });
}

APS5_HOST_SHIM ssize_t pwritev(int fd, const iovec* vectors, int count, off_t offset) {
    return WithVectors(vectors, count, [&](const iovec* host) { return Functions().pwritev(fd, host, count, offset); });
}

APS5_HOST_SHIM ssize_t recv(int socket, void* buffer, std::size_t bytes, int flags) {
    return Functions().recv(socket, Host(buffer, bytes), bytes, flags);
}

APS5_HOST_SHIM ssize_t recvfrom(int socket, void* buffer, std::size_t bytes, int flags, sockaddr* address, socklen_t* length) {
    socklen_t* hostLength = Host(length, sizeof(socklen_t));
    sockaddr* hostAddress = address != nullptr && hostLength != nullptr ? Host(address, *hostLength) : address;
    return Functions().recvfrom(socket, Host(buffer, bytes), bytes, flags, hostAddress, hostLength);
}

APS5_HOST_SHIM ssize_t send(int socket, const void* buffer, std::size_t bytes, int flags) {
    return Functions().send(socket, Host(buffer, bytes), bytes, flags);
}

APS5_HOST_SHIM ssize_t sendto(int socket, const void* buffer, std::size_t bytes, int flags, const sockaddr* address, socklen_t length) {
    return Functions().sendto(socket, Host(buffer, bytes), bytes, flags, Host(address, length), length);
}

APS5_HOST_SHIM int getentropy(void* buffer, std::size_t bytes) {
    return Functions().getentropy(Host(buffer, bytes), bytes);
}

APS5_HOST_SHIM std::size_t fread(void* buffer, std::size_t size, std::size_t count, FILE* stream) {
    return Functions().fread(Host(buffer, size * count), size, count, stream);
}

APS5_HOST_SHIM std::size_t fwrite(const void* buffer, std::size_t size, std::size_t count, FILE* stream) {
    return Functions().fwrite(Host(buffer, size * count), size, count, stream);
}

#undef APS5_HOST_SHIM

}

#endif // defined(__APPLE__) && defined(__aarch64__)
