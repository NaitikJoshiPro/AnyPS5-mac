// Host system call shim (D35). Linked as a static library into a prx (APS5_HOST_SYSCALL_SHIM), its hidden definitions
// of read, write, open, stat, ... take the place of libSystem's for every object in that library, the third-party code
// linked into it included. Each translates guest pointers in the displaced window (D31, D33) to their alias and calls
// libSystem's function: the kernel returns EFAULT for the guest address and never raises the signal the fault handler
// redirects. Host addresses, aliases included, pass through after one compare. A buffer across an edge of the window
// is passed unchanged, and the kernel still returns EFAULT for it.
//
// Which libSystem imports are defined here, and why the others need nothing, is listed in host_imports.tsv next to
// this file; the test host_import_inventory fails when a prx imports a libSystem symbol that file does not classify.
// Pointers the kernel retains or hands back are never translated (kevent udata), or are mapped back (fgets).
// A path string starting in the window lies wholly inside it (the arena refuses mappings across its edges), so paths
// are translated without scanning them.
#include "prx/libc/include/GuestDisplaced.hpp"

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/mach_time.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/event.h>
#include <sys/fsgetpath.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/utsname.h>
#include <time.h>
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

// A NUL-terminated string: one byte decides (see the header comment).
template <typename T>
T* HostString(T* guest) {
    return Host(guest, 1);
}

void* LibSystem(const char* name) {
    // dlopen itself is shimmed here: libdyld's is the only exported one, so RTLD_DEFAULT finds it.
    static void* const library = reinterpret_cast<decltype(&::dlopen)>(dlsym(RTLD_DEFAULT, "dlopen"))(
        "/usr/lib/libSystem.B.dylib", RTLD_LAZY | RTLD_NOLOAD);
    void* symbol = library != nullptr ? dlsym(library, name) : nullptr;
    if (symbol == nullptr) throw std::runtime_error("host system call shim: libSystem has no " + std::string(name));
    return symbol;
}

// Every libSystem function a definition below forwards to. Each is resolved once, at load (ResolveShimmedCalls), never
// on first use: a definition can first run inside a signal handler (the displaced fault handler's trace and crash
// paths write; guest signal handlers reach read/write/nanosleep...), and dlsym there could deadlock on a dyld or
// malloc lock the interrupted thread holds. A call that runs before the constructor (another initializer of the
// same image) resolves its own slot, outside any signal handler.
#define APS5_SHIMMED_CALLS(X) \
    X(accept) X(access) X(bind) X(chmod) X(clock_getres) X(clock_gettime) X(connect) X(dlopen) X(execv) X(faccessat) \
    X(fchmodat) X(fcntl) X(fgets) X(fopen) X(fputs) X(fread) X(freopen) X(fsgetpath) X(fstat) X(fstatat) X(fstatfs) X(fstatvfs) \
    X(futimes) X(fwrite) X(getentropy) X(getpeername) X(getrlimit) X(getrusage) X(getsockname) X(getsockopt) \
    X(gettimeofday) X(ioctl) X(kevent) X(lstat) X(lutimes) X(mach_msg) X(mach_timebase_info) X(mach_vm_read_overwrite) \
    X(mach_vm_write) X(mkdir) X(mkstemp) \
    X(nanosleep) X(open) X(openat) X(opendir) X(pipe) X(pipe2) X(poll) X(pread) X(preadv) X(pthread_setname_np) \
    X(pthread_sigmask) X(puts) X(pwrite) X(pwritev) X(read) X(readv) X(recv) X(recvfrom) X(remove) X(rename) X(rmdir) \
    X(select) X(send) X(sendto) X(setrlimit) X(setsockopt) X(setvbuf) X(shm_open) X(shm_unlink) X(sigaction) \
    X(socketpair) X(stat) X(statvfs) X(sysctl) X(sysctlbyname) X(truncate) X(uname) X(unlink) X(utimes) X(write) X(writev)

enum class Shimmed : std::size_t {
#define APS5_SHIMMED_SLOT(name) name,
    APS5_SHIMMED_CALLS(APS5_SHIMMED_SLOT)
#undef APS5_SHIMMED_SLOT
    Count
};

constexpr const char* ShimmedNames[] = {
#define APS5_SHIMMED_NAME(name) #name,
    APS5_SHIMMED_CALLS(APS5_SHIMMED_NAME)
#undef APS5_SHIMMED_NAME
};

std::atomic<void*> g_real[static_cast<std::size_t>(Shimmed::Count)];

void* Real(Shimmed slot) {
    const auto index = static_cast<std::size_t>(slot);
    void* symbol = g_real[index].load(std::memory_order_acquire);
    if (symbol != nullptr) return symbol;
    symbol = LibSystem(ShimmedNames[index]); // only before the constructor ran, or for a symbol this host lacks (throws)
    g_real[index].store(symbol, std::memory_order_release);
    return symbol;
}

// libSystem's definition of `name` (inside this library the name binds to the hidden definition below).
#define APS5_REAL(name) (reinterpret_cast<decltype(&::name)>(Real(Shimmed::name)))

template <typename TCall>
ssize_t WithVectors(const iovec* vectors, int count, TCall&& call) {
    const iovec* given = Host(vectors, count > 0 ? sizeof(iovec) * static_cast<std::size_t>(count) : 0);
    if (count <= 0 || count > IOV_MAX) return call(given);
    bool translate = false;
    for (int i = 0; i < count && !translate; ++i)
        translate = Host(given[i].iov_base, given[i].iov_len) != given[i].iov_base;
    if (!translate) return call(given);
    iovec copied[IOV_MAX];
    for (int i = 0; i < count; ++i) copied[i] = {Host(given[i].iov_base, given[i].iov_len), given[i].iov_len};
    return call(copied);
}

// socklen_t* in/out length and the address it sizes.
template <typename TAddress>
TAddress* HostAddressOut(TAddress* address, socklen_t*& length) {
    length = Host(length, sizeof(socklen_t));
    return address != nullptr && length != nullptr ? Host(address, *length) : address;
}

std::size_t FcntlBytes(int command) {
    switch (command) {
    case F_GETPATH:
    case F_GETPATH_NOFIRMLINK: return MAXPATHLEN;
    case F_GETLK:
    case F_SETLK:
    case F_SETLKW: return sizeof(struct flock);
    case F_PREALLOCATE: return sizeof(fstore_t);
    case F_PUNCHHOLE: return sizeof(fpunchhole_t);
    case F_LOG2PHYS:
    case F_LOG2PHYS_EXT: return sizeof(struct log2phys);
    case F_RDADVISE: return sizeof(struct radvisory);
    case F_SETSIZE: return sizeof(off_t);
    default: return 0; // an integer argument, or a command whose argument size is not known here: passed through
    }
}

std::size_t FdSetBytes(int count) {
    return count > 0 ? ((static_cast<std::size_t>(count) + NFDBITS - 1) / NFDBITS) * sizeof(fd_mask) : 0;
}

int Format(FILE* stream, const char* format, va_list arguments) {
    char local[1024];
    va_list copy;
    va_copy(copy, arguments);
    const int length = std::vsnprintf(local, sizeof(local), format, copy);
    va_end(copy);
    if (length < 0) return length;
    char* text = local;
    if (static_cast<std::size_t>(length) >= sizeof(local)) {
        text = static_cast<char*>(std::malloc(static_cast<std::size_t>(length) + 1));
        if (text == nullptr) return -1;
        std::vsnprintf(text, static_cast<std::size_t>(length) + 1, format, arguments);
    }
    const std::size_t written = APS5_REAL(fwrite)(text, 1, static_cast<std::size_t>(length), stream);
    if (text != local) std::free(text);
    return written == static_cast<std::size_t>(length) ? length : -1;
}

}

extern "C" {

#define APS5_HOST_SHIM __attribute__((visibility("hidden")))

// Buffers.

APS5_HOST_SHIM ssize_t read(int fd, void* buffer, std::size_t bytes) {
    return APS5_REAL(read)(fd, Host(buffer, bytes), bytes);
}

APS5_HOST_SHIM ssize_t write(int fd, const void* buffer, std::size_t bytes) {
    return APS5_REAL(write)(fd, Host(buffer, bytes), bytes);
}

APS5_HOST_SHIM ssize_t pread(int fd, void* buffer, std::size_t bytes, off_t offset) {
    return APS5_REAL(pread)(fd, Host(buffer, bytes), bytes, offset);
}

APS5_HOST_SHIM ssize_t pwrite(int fd, const void* buffer, std::size_t bytes, off_t offset) {
    return APS5_REAL(pwrite)(fd, Host(buffer, bytes), bytes, offset);
}

APS5_HOST_SHIM ssize_t readv(int fd, const iovec* vectors, int count) {
    return WithVectors(vectors, count, [&](const iovec* host) { return APS5_REAL(readv)(fd, host, count); });
}

APS5_HOST_SHIM ssize_t writev(int fd, const iovec* vectors, int count) {
    return WithVectors(vectors, count, [&](const iovec* host) { return APS5_REAL(writev)(fd, host, count); });
}

APS5_HOST_SHIM ssize_t preadv(int fd, const iovec* vectors, int count, off_t offset) {
    return WithVectors(vectors, count, [&](const iovec* host) { return APS5_REAL(preadv)(fd, host, count, offset); });
}

APS5_HOST_SHIM ssize_t pwritev(int fd, const iovec* vectors, int count, off_t offset) {
    return WithVectors(vectors, count, [&](const iovec* host) { return APS5_REAL(pwritev)(fd, host, count, offset); });
}

APS5_HOST_SHIM ssize_t recv(int socket, void* buffer, std::size_t bytes, int flags) {
    return APS5_REAL(recv)(socket, Host(buffer, bytes), bytes, flags);
}

APS5_HOST_SHIM ssize_t recvfrom(int socket, void* buffer, std::size_t bytes, int flags, sockaddr* address, socklen_t* length) {
    sockaddr* hostAddress = HostAddressOut(address, length);
    return APS5_REAL(recvfrom)(socket, Host(buffer, bytes), bytes, flags, hostAddress, length);
}

APS5_HOST_SHIM ssize_t send(int socket, const void* buffer, std::size_t bytes, int flags) {
    return APS5_REAL(send)(socket, Host(buffer, bytes), bytes, flags);
}

APS5_HOST_SHIM ssize_t sendto(int socket, const void* buffer, std::size_t bytes, int flags, const sockaddr* address, socklen_t length) {
    return APS5_REAL(sendto)(socket, Host(buffer, bytes), bytes, flags, Host(address, length), length);
}

APS5_HOST_SHIM int accept(int socket, sockaddr* address, socklen_t* length) {
    sockaddr* hostAddress = HostAddressOut(address, length);
    return APS5_REAL(accept)(socket, hostAddress, length);
}

APS5_HOST_SHIM int getpeername(int socket, sockaddr* address, socklen_t* length) {
    sockaddr* hostAddress = HostAddressOut(address, length);
    return APS5_REAL(getpeername)(socket, hostAddress, length);
}

APS5_HOST_SHIM int getsockname(int socket, sockaddr* address, socklen_t* length) {
    sockaddr* hostAddress = HostAddressOut(address, length);
    return APS5_REAL(getsockname)(socket, hostAddress, length);
}

APS5_HOST_SHIM int bind(int socket, const sockaddr* address, socklen_t length) {
    return APS5_REAL(bind)(socket, Host(address, length), length);
}

APS5_HOST_SHIM int connect(int socket, const sockaddr* address, socklen_t length) {
    return APS5_REAL(connect)(socket, Host(address, length), length);
}

APS5_HOST_SHIM int getsockopt(int socket, int level, int name, void* value, socklen_t* length) {
    void* hostValue = HostAddressOut(value, length);
    return APS5_REAL(getsockopt)(socket, level, name, hostValue, length);
}

APS5_HOST_SHIM int setsockopt(int socket, int level, int name, const void* value, socklen_t length) {
    return APS5_REAL(setsockopt)(socket, level, name, Host(value, length), length);
}

APS5_HOST_SHIM int poll(pollfd* fds, nfds_t count, int timeout) {
    return APS5_REAL(poll)(Host(fds, sizeof(pollfd) * count), count, timeout);
}

// The SDK declares select outside extern "C" (sys/_select.h), so it is defined under another name with select's symbol.
APS5_HOST_SHIM int HostShimSelect(int count, fd_set* __restrict readable, fd_set* __restrict writable, fd_set* __restrict failed,
    timeval* __restrict timeout) __asm("_select");
APS5_HOST_SHIM int HostShimSelect(int count, fd_set* __restrict readable, fd_set* __restrict writable, fd_set* __restrict failed,
    timeval* __restrict timeout) {
    const std::size_t bytes = FdSetBytes(count);
    return APS5_REAL(select)(count, Host(readable, bytes), Host(writable, bytes), Host(failed, bytes), Host(timeout, sizeof(timeval)));
}

// udata is opaque to the kernel and comes back in the events unchanged: never translated.
APS5_HOST_SHIM int kevent(int queue, const struct kevent* changes, int changeCount, struct kevent* events, int eventCount,
    const timespec* timeout) {
    const auto bytes = [](int count) { return count > 0 ? sizeof(struct kevent) * static_cast<std::size_t>(count) : 0; };
    return APS5_REAL(kevent)(queue, Host(changes, bytes(changeCount)), changeCount, Host(events, bytes(eventCount)), eventCount,
        Host(timeout, sizeof(timespec)));
}

// The request encodes the argument's size and direction; IOC_VOID requests and size 0 pass the argument through
// (a request that takes a pointer without encoding its size is not translated: TechnicalDebt).
APS5_HOST_SHIM int ioctl(int fd, unsigned long request, ...) {
    va_list arguments;
    va_start(arguments, request);
    void* argument = va_arg(arguments, void*);
    va_end(arguments);
    if ((request & IOC_INOUT) != 0) argument = Host(argument, IOCPARM_LEN(request));
    return APS5_REAL(ioctl)(fd, request, argument);
}

APS5_HOST_SHIM int fcntl(int fd, int command, ...) {
    va_list arguments;
    va_start(arguments, command);
    void* argument = va_arg(arguments, void*);
    va_end(arguments);
    if (const std::size_t bytes = FcntlBytes(command); bytes != 0) argument = Host(argument, bytes);
    return APS5_REAL(fcntl)(fd, command, argument);
}

APS5_HOST_SHIM int fstat(int fd, struct stat* status) {
    return APS5_REAL(fstat)(fd, Host(status, sizeof(struct stat)));
}

APS5_HOST_SHIM int fstatfs(int fd, struct statfs* status) {
    return APS5_REAL(fstatfs)(fd, Host(status, sizeof(struct statfs)));
}

APS5_HOST_SHIM int fstatvfs(int fd, struct statvfs* status) {
    return APS5_REAL(fstatvfs)(fd, Host(status, sizeof(struct statvfs)));
}

APS5_HOST_SHIM int statvfs(const char* path, struct statvfs* status) {
    return APS5_REAL(statvfs)(HostString(path), Host(status, sizeof(struct statvfs)));
}

APS5_HOST_SHIM int sysctl(int* name, u_int nameLength, void* old, std::size_t* oldLength, void* replacement, std::size_t replacementLength) {
    std::size_t* hostOldLength = Host(oldLength, sizeof(std::size_t));
    void* hostOld = old != nullptr && hostOldLength != nullptr ? Host(old, *hostOldLength) : old;
    return APS5_REAL(sysctl)(Host(name, sizeof(int) * nameLength), nameLength, hostOld, hostOldLength, Host(replacement, replacementLength),
        replacementLength);
}

APS5_HOST_SHIM int sysctlbyname(const char* name, void* old, std::size_t* oldLength, void* replacement, std::size_t replacementLength) {
    std::size_t* hostOldLength = Host(oldLength, sizeof(std::size_t));
    void* hostOld = old != nullptr && hostOldLength != nullptr ? Host(old, *hostOldLength) : old;
    return APS5_REAL(sysctlbyname)(HostString(name), hostOld, hostOldLength, Host(replacement, replacementLength), replacementLength);
}

APS5_HOST_SHIM ssize_t fsgetpath(char* buffer, std::size_t bytes, fsid_t* fsid, std::uint64_t object) {
    return APS5_REAL(fsgetpath)(Host(buffer, bytes), bytes, Host(fsid, sizeof(fsid_t)), object);
}

APS5_HOST_SHIM int socketpair(int domain, int type, int protocol, int pair[2]) {
    return APS5_REAL(socketpair)(domain, type, protocol, Host(pair, 2 * sizeof(int)));
}

APS5_HOST_SHIM int pipe(int pair[2]) {
    return APS5_REAL(pipe)(Host(pair, 2 * sizeof(int)));
}

APS5_HOST_SHIM int pipe2(int pair[2], int flags) {
    return APS5_REAL(pipe2)(Host(pair, 2 * sizeof(int)), flags);
}

APS5_HOST_SHIM int futimes(int fd, const timeval times[2]) {
    return APS5_REAL(futimes)(fd, Host(times, 2 * sizeof(timeval)));
}

APS5_HOST_SHIM int sigaction(int signal, const struct sigaction* __restrict action, struct sigaction* __restrict previous) {
    return APS5_REAL(sigaction)(signal, Host(action, sizeof(struct sigaction)), Host(previous, sizeof(struct sigaction)));
}

APS5_HOST_SHIM int pthread_sigmask(int how, const sigset_t* set, sigset_t* previous) {
    return APS5_REAL(pthread_sigmask)(how, Host(set, sizeof(sigset_t)), Host(previous, sizeof(sigset_t)));
}

APS5_HOST_SHIM int getrlimit(int resource, rlimit* limit) {
    return APS5_REAL(getrlimit)(resource, Host(limit, sizeof(rlimit)));
}

APS5_HOST_SHIM int setrlimit(int resource, const rlimit* limit) {
    return APS5_REAL(setrlimit)(resource, Host(limit, sizeof(rlimit)));
}

APS5_HOST_SHIM int getrusage(int who, rusage* usage) {
    return APS5_REAL(getrusage)(who, Host(usage, sizeof(rusage)));
}

APS5_HOST_SHIM int uname(utsname* name) {
    return APS5_REAL(uname)(Host(name, sizeof(utsname)));
}

APS5_HOST_SHIM int nanosleep(const timespec* request, timespec* remaining) {
    return APS5_REAL(nanosleep)(Host(request, sizeof(timespec)), Host(remaining, sizeof(timespec)));
}

APS5_HOST_SHIM int gettimeofday(timeval* __restrict time, void* __restrict zone) {
    return APS5_REAL(gettimeofday)(Host(time, sizeof(timeval)), Host(zone, sizeof(struct timezone)));
}

APS5_HOST_SHIM int clock_gettime(clockid_t clock, timespec* time) {
    return APS5_REAL(clock_gettime)(clock, Host(time, sizeof(timespec)));
}

APS5_HOST_SHIM int clock_getres(clockid_t clock, timespec* resolution) {
    return APS5_REAL(clock_getres)(clock, Host(resolution, sizeof(timespec)));
}

APS5_HOST_SHIM kern_return_t mach_timebase_info(mach_timebase_info_t info) {
    return APS5_REAL(mach_timebase_info)(Host(info, sizeof(*info)));
}

// The kernel reads and writes these ranges through the task's VM map, where a displaced guest address has no mapping
// (KERN_INVALID_ADDRESS). Only this task's ranges are translated: another task's addresses are not guest addresses.
APS5_HOST_SHIM kern_return_t mach_vm_read_overwrite(vm_map_read_t task, mach_vm_address_t address, mach_vm_size_t size,
    mach_vm_address_t data, mach_vm_size_t* copied) {
    if (task == mach_task_self()) {
        address = reinterpret_cast<mach_vm_address_t>(Host(reinterpret_cast<void*>(address), size));
        data = reinterpret_cast<mach_vm_address_t>(Host(reinterpret_cast<void*>(data), size));
    }
    return APS5_REAL(mach_vm_read_overwrite)(task, address, size, data, copied);
}

APS5_HOST_SHIM kern_return_t mach_vm_write(vm_map_t task, mach_vm_address_t address, vm_offset_t data, mach_msg_type_number_t bytes) {
    if (task == mach_task_self()) {
        address = reinterpret_cast<mach_vm_address_t>(Host(reinterpret_cast<void*>(address), bytes));
        data = reinterpret_cast<vm_offset_t>(Host(reinterpret_cast<void*>(data), bytes));
    }
    return APS5_REAL(mach_vm_write)(task, address, data, bytes);
}

// Out-of-line descriptors inside the message are not translated.
APS5_HOST_SHIM mach_msg_return_t mach_msg(mach_msg_header_t* message, mach_msg_option_t option, mach_msg_size_t sendSize,
    mach_msg_size_t receiveSize, mach_port_name_t receiveName, mach_msg_timeout_t timeout, mach_port_name_t notify) {
    std::size_t bytes = 0;
    if ((option & MACH_SEND_MSG) != 0) bytes = sendSize;
    if ((option & MACH_RCV_MSG) != 0 && receiveSize > bytes) bytes = receiveSize;
    return APS5_REAL(mach_msg)(Host(message, bytes), option, sendSize, receiveSize, receiveName, timeout, notify);
}

APS5_HOST_SHIM int pthread_setname_np(const char* name) {
    return APS5_REAL(pthread_setname_np)(HostString(name));
}

APS5_HOST_SHIM int getentropy(void* buffer, std::size_t bytes) {
    return APS5_REAL(getentropy)(Host(buffer, bytes), bytes);
}

// Paths.

APS5_HOST_SHIM int access(const char* path, int mode) {
    return APS5_REAL(access)(HostString(path), mode);
}

APS5_HOST_SHIM int faccessat(int directory, const char* path, int mode, int flags) {
    return APS5_REAL(faccessat)(directory, HostString(path), mode, flags);
}

APS5_HOST_SHIM int fchmodat(int directory, const char* path, mode_t mode, int flags) {
    return APS5_REAL(fchmodat)(directory, HostString(path), mode, flags);
}

APS5_HOST_SHIM int chmod(const char* path, mode_t mode) {
    return APS5_REAL(chmod)(HostString(path), mode);
}

APS5_HOST_SHIM int mkdir(const char* path, mode_t mode) {
    return APS5_REAL(mkdir)(HostString(path), mode);
}

APS5_HOST_SHIM int rmdir(const char* path) {
    return APS5_REAL(rmdir)(HostString(path));
}

APS5_HOST_SHIM int unlink(const char* path) {
    return APS5_REAL(unlink)(HostString(path));
}

APS5_HOST_SHIM int remove(const char* path) {
    return APS5_REAL(remove)(HostString(path));
}

APS5_HOST_SHIM int rename(const char* from, const char* to) {
    return APS5_REAL(rename)(HostString(from), HostString(to));
}

APS5_HOST_SHIM int truncate(const char* path, off_t length) {
    return APS5_REAL(truncate)(HostString(path), length);
}

APS5_HOST_SHIM int open(const char* path, int flags, ...) {
    va_list arguments;
    va_start(arguments, flags);
    const int mode = va_arg(arguments, int);
    va_end(arguments);
    return APS5_REAL(open)(HostString(path), flags, mode);
}

APS5_HOST_SHIM int openat(int directory, const char* path, int flags, ...) {
    va_list arguments;
    va_start(arguments, flags);
    const int mode = va_arg(arguments, int);
    va_end(arguments);
    return APS5_REAL(openat)(directory, HostString(path), flags, mode);
}

APS5_HOST_SHIM int shm_open(const char* name, int flags, ...) {
    va_list arguments;
    va_start(arguments, flags);
    const int mode = va_arg(arguments, int);
    va_end(arguments);
    return APS5_REAL(shm_open)(HostString(name), flags, mode);
}

APS5_HOST_SHIM int shm_unlink(const char* name) {
    return APS5_REAL(shm_unlink)(HostString(name));
}

APS5_HOST_SHIM int stat(const char* path, struct stat* status) {
    return APS5_REAL(stat)(HostString(path), Host(status, sizeof(struct stat)));
}

APS5_HOST_SHIM int lstat(const char* path, struct stat* status) {
    return APS5_REAL(lstat)(HostString(path), Host(status, sizeof(struct stat)));
}

APS5_HOST_SHIM int fstatat(int directory, const char* path, struct stat* status, int flags) {
    return APS5_REAL(fstatat)(directory, HostString(path), Host(status, sizeof(struct stat)), flags);
}

APS5_HOST_SHIM int utimes(const char* path, const timeval times[2]) {
    return APS5_REAL(utimes)(HostString(path), Host(times, 2 * sizeof(timeval)));
}

APS5_HOST_SHIM int lutimes(const char* path, const timeval times[2]) {
    return APS5_REAL(lutimes)(HostString(path), Host(times, 2 * sizeof(timeval)));
}

// The template is rewritten in place; through the alias that is the same memory.
APS5_HOST_SHIM int mkstemp(char* path) {
    return APS5_REAL(mkstemp)(HostString(path));
}

APS5_HOST_SHIM DIR* opendir(const char* path) {
    return APS5_REAL(opendir)(HostString(path));
}

APS5_HOST_SHIM void* dlopen(const char* path, int mode) {
    return APS5_REAL(dlopen)(HostString(path), mode);
}

APS5_HOST_SHIM int execv(const char* path, char* const* arguments) {
    char* const* given = HostString(arguments);
    std::size_t count = 0;
    bool translate = given != arguments;
    if (given != nullptr)
        for (; given[count] != nullptr; ++count) translate |= HostString(given[count]) != given[count];
    if (!translate) return APS5_REAL(execv)(HostString(path), arguments);
    char** copied = static_cast<char**>(std::calloc(count + 1, sizeof(char*)));
    if (copied == nullptr) return -1;
    for (std::size_t i = 0; i < count; ++i) copied[i] = HostString(given[i]);
    const int result = APS5_REAL(execv)(HostString(path), copied); // returns only on failure
    std::free(copied);
    return result;
}

// stdio.

APS5_HOST_SHIM FILE* fopen(const char* __restrict path, const char* __restrict mode) {
    return APS5_REAL(fopen)(HostString(path), HostString(mode));
}

APS5_HOST_SHIM FILE* freopen(const char* __restrict path, const char* __restrict mode, FILE* __restrict stream) {
    return APS5_REAL(freopen)(HostString(path), HostString(mode), stream);
}

APS5_HOST_SHIM std::size_t fread(void* __restrict buffer, std::size_t size, std::size_t count, FILE* __restrict stream) {
    return APS5_REAL(fread)(Host(buffer, size * count), size, count, stream);
}

APS5_HOST_SHIM std::size_t fwrite(const void* __restrict buffer, std::size_t size, std::size_t count, FILE* __restrict stream) {
    return APS5_REAL(fwrite)(Host(buffer, size * count), size, count, stream);
}

APS5_HOST_SHIM int fputs(const char* __restrict text, FILE* __restrict stream) {
    return APS5_REAL(fputs)(HostString(text), stream);
}

APS5_HOST_SHIM int puts(const char* text) {
    return APS5_REAL(puts)(HostString(text));
}

// The returned pointer is the guest's own, never the alias.
APS5_HOST_SHIM char* fgets(char* __restrict buffer, int bytes, FILE* __restrict stream) {
    char* host = Host(buffer, bytes > 0 ? static_cast<std::size_t>(bytes) : 0);
    char* result = APS5_REAL(fgets)(host, bytes, stream);
    return result == host ? buffer : result;
}

// The FILE keeps the buffer (as its alias) and never hands it back.
APS5_HOST_SHIM int setvbuf(FILE* __restrict stream, char* __restrict buffer, int mode, std::size_t bytes) {
    return APS5_REAL(setvbuf)(stream, Host(buffer, bytes), mode, bytes);
}

// setbuf is setvbuf with BUFSIZ (or unbuffered for a null buffer), so the retained buffer gets the same alias.
APS5_HOST_SHIM void setbuf(FILE* __restrict stream, char* __restrict buffer) {
    APS5_REAL(setvbuf)(stream, Host(buffer, BUFSIZ), buffer != nullptr ? _IOFBF : _IONBF, BUFSIZ);
}

// Formatted into a host buffer, then written: a %s argument longer than the stream's buffer would otherwise go to
// write(2) straight from guest memory.
APS5_HOST_SHIM int vfprintf(FILE* __restrict stream, const char* __restrict format, va_list arguments) {
    return Format(stream, format, arguments);
}

APS5_HOST_SHIM int fprintf(FILE* __restrict stream, const char* __restrict format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const int result = Format(stream, format, arguments);
    va_end(arguments);
    return result;
}

APS5_HOST_SHIM int printf(const char* __restrict format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const int result = Format(stdout, format, arguments);
    va_end(arguments);
    return result;
}

#undef APS5_HOST_SHIM

}

namespace {

// Resolves every forwarded libSystem function at load (see APS5_SHIMMED_CALLS). A symbol this host's libSystem lacks
// stays unresolved and throws on use, as before.
__attribute__((constructor)) void ResolveShimmedCalls() {
    const int saved = errno;
    for (std::size_t index = 0; index < static_cast<std::size_t>(Shimmed::Count); ++index) {
        if (g_real[index].load(std::memory_order_relaxed) != nullptr) continue;
        try { g_real[index].store(LibSystem(ShimmedNames[index]), std::memory_order_release); }
        catch (const std::exception&) {}
    }
    errno = saved;
}

}

extern "C" {

// For tests: how many forwarded functions the shim has, and how many are resolved (all of them, after load).
__attribute__((visibility("hidden"))) void Aps5HostSyscallShimResolution(std::size_t* resolved, std::size_t* total) {
    std::size_t count = 0;
    for (const auto& slot : g_real) count += slot.load(std::memory_order_acquire) != nullptr ? 1 : 0;
    *resolved = count;
    *total = static_cast<std::size_t>(Shimmed::Count);
}

}

#endif // defined(__APPLE__) && defined(__aarch64__)
