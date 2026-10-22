// The host system call shim (D35) at displaced guest addresses (D31, D33): libSystem's own read returns EFAULT for a
// buffer at the guest address, the shim's definitions (linked into this executable) move the alias's bytes, and no
// signal is redirected on the way. Run at the shared region tail (0x400000000) and the GPU carveout (0x4000000000).
#include "SceTypes.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <poll.h>
#include <sys/event.h>
#include <sys/ioctl.h>
#include <sys/param.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/sysctl.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/utsname.h>
#include <unistd.h>

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
}

namespace {

constexpr std::uintptr_t CarveoutBase = 0x4000000000ull;
constexpr std::uintptr_t TailBase = 0x400000000ull;
constexpr std::size_t Bytes = 0x200000;
constexpr std::size_t Size = 0x30000;
constexpr int CpuGpuReadWrite = 0x33;
constexpr int MapFixed = 0x10;

void Require(bool condition, std::uintptr_t base, const std::string& message) {
    if (!condition) {
        char text[64];
        std::snprintf(text, sizeof(text), "0x%llx: ", static_cast<unsigned long long>(base));
        throw std::runtime_error(text + message);
    }
}

std::uint64_t Faults() {
    std::uint64_t faults = 0;
    GuestArena::GuestDisplacedStats_nid_no_patch(&faults, nullptr);
    return faults;
}

std::uint8_t Pattern(std::size_t index) {
    return static_cast<std::uint8_t>(index * 29u + 7u);
}


// Copies a NUL-terminated string into the window and returns its guest address.
char* GuestString(std::uint8_t* guest, std::uint8_t* alias, std::size_t offset, const std::string& text) {
    std::memcpy(alias + offset, text.c_str(), text.size() + 1);
    return reinterpret_cast<char*>(guest + offset);
}

// Calls beyond read/write: paths, out-structs, socket options, arrays, ioctl/fcntl, sysctl, stdio.
// Returns the faults fprintf and printf took: their %s argument is read in userspace (correct through the fault path, not EFAULT).
std::uint64_t RunKernelStructs(std::uintptr_t base, std::uint8_t* guest, std::uint8_t* alias, const std::filesystem::path& file) {
    constexpr std::size_t Scratch = 3 * Size; // [3 * Size, Bytes) is free for this function
    std::memset(alias + Scratch, 0, Bytes - Scratch);
    const auto at = [&](std::size_t offset) { return guest + Scratch + offset; };
    const auto host = [&](std::size_t offset) { return alias + Scratch + offset; };

    // Paths and stat out-structs, both in the window.
    const std::string directory = file.string() + ".d";
    char* guestDirectory = GuestString(guest, alias, Scratch + 0x000, directory);
    char* guestFile = GuestString(guest, alias, Scratch + 0x400, directory + "/a");
    char* guestRenamed = GuestString(guest, alias, Scratch + 0x800, directory + "/b");
    Require(::mkdir(guestDirectory, 0700) == 0, base, "mkdir with a displaced path");
    const int fd = ::open(guestFile, O_CREAT | O_RDWR | O_TRUNC, 0600);
    Require(fd >= 0, base, "open with a displaced path");
    Require(::write(fd, alias, 0x100) == 0x100, base, "write");
    auto* status = reinterpret_cast<struct stat*>(at(0x1000));
    auto* hostStatus = reinterpret_cast<struct stat*>(host(0x1000));
    Require(::fstat(fd, status) == 0 && hostStatus->st_size == 0x100, base, "fstat into a displaced struct");
    std::memset(hostStatus, 0, sizeof(struct stat));
    Require(::stat(guestFile, status) == 0 && hostStatus->st_size == 0x100, base, "stat with a displaced path and struct");
    Require(::access(guestFile, R_OK) == 0, base, "access with a displaced path");
    using Access = int (*)(const char*, int);
    const auto libSystemAccess = reinterpret_cast<Access>(dlsym(dlopen("/usr/lib/libSystem.B.dylib", RTLD_LAZY | RTLD_NOLOAD), "access"));
    errno = 0;
    Require(libSystemAccess != nullptr && libSystemAccess(guestFile, R_OK) == -1 && errno == EFAULT, base,
        "libSystem's access with the guest path did not fail with EFAULT");
    char* path = reinterpret_cast<char*>(at(0x1400));
    Require(::fcntl(fd, F_GETPATH, path) == 0 && std::strstr(reinterpret_cast<char*>(host(0x1400)), "/a") != nullptr, base,
        "fcntl F_GETPATH into a displaced buffer");
    auto* pending = reinterpret_cast<int*>(at(0x2000));
    Require(::lseek(fd, 0, SEEK_SET) == 0, base, "lseek");
    Require(::ioctl(fd, FIONREAD, pending) == 0 && *reinterpret_cast<int*>(host(0x2000)) == 0x100, base, "ioctl FIONREAD into a displaced int");
    ::close(fd);
    Require(::rename(guestFile, guestRenamed) == 0 && ::lstat(guestRenamed, status) == 0, base, "rename and lstat with displaced paths");
    FILE* stream = std::fopen(guestRenamed, GuestString(guest, alias, Scratch + 0xc00, "r+"));
    Require(stream != nullptr, base, "fopen with a displaced path and mode");

    // stdio: a stream buffer in the window, fgets mapping its result back, fputs and a long %s from the window.
    Require(::setvbuf(stream, reinterpret_cast<char*>(at(0x3000)), _IOFBF, 0x1000) == 0, base, "setvbuf with a displaced buffer");
    std::memset(host(0x5000), 'x', 0x3000);
    host(0x5000)[0x2fff] = 0;
    const auto formatBefore = Faults();
    Require(std::fprintf(stream, "%s\n", reinterpret_cast<char*>(at(0x5000))) == 0x3000, base, "fprintf of a long displaced string");
    const auto formatFaults = Faults() - formatBefore;
    Require(std::fputs(GuestString(guest, alias, Scratch + 0x8000, "tail line\n"), stream) >= 0 && std::fflush(stream) == 0, base,
        "fputs of a displaced string");
    Require(std::fseek(stream, 0x3000, SEEK_SET) == 0, base, "fseek");
    char* line = reinterpret_cast<char*>(at(0x9000));
    Require(std::fgets(line, 64, stream) == line && std::strcmp(reinterpret_cast<char*>(host(0x9000)), "tail line\n") == 0, base,
        "fgets into a displaced buffer returns the guest pointer");
    std::fclose(stream);
    // setbuf retains a BUFSIZ buffer like setvbuf (mbedtls in libSceSsl); printf formats a displaced %s on the host.
    FILE* buffered = std::fopen(guestRenamed, "r");
    Require(buffered != nullptr, base, "fopen for setbuf");
    std::setbuf(buffered, reinterpret_cast<char*>(at(0x9800)));
    Require(std::fgetc(buffered) == 'x' && host(0x9800)[0] == 'x', base, "setbuf with a displaced buffer reads through its alias");
    std::fclose(buffered);
    char* printed = GuestString(guest, alias, Scratch + 0x9c00, "host system call shim: displaced printf\n");
    const auto printBefore = Faults();
    Require(std::printf("%s", printed) == 40, base, "printf of a displaced string");
    const auto printFaults = Faults() - printBefore;
    Require(::unlink(guestRenamed) == 0 && ::rmdir(guestDirectory) == 0, base, "unlink and rmdir with displaced paths");

    // Sockets: name, options, poll/select/kevent arrays.
    int pair[2] = {-1, -1};
    auto* guestPair = reinterpret_cast<int*>(at(0xa000));
    Require(::socketpair(AF_UNIX, SOCK_STREAM, 0, guestPair) == 0, base, "socketpair into a displaced array");
    std::memcpy(pair, host(0xa000), sizeof(pair));
    auto* length = reinterpret_cast<socklen_t*>(at(0xa100));
    *reinterpret_cast<socklen_t*>(host(0xa100)) = sizeof(int);
    Require(::getsockopt(pair[0], SOL_SOCKET, SO_TYPE, at(0xa200), length) == 0 && *reinterpret_cast<int*>(host(0xa200)) == SOCK_STREAM, base,
        "getsockopt into displaced value and length");
    *reinterpret_cast<int*>(host(0xa200)) = 0x10000;
    Require(::setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, at(0xa200), sizeof(int)) == 0, base, "setsockopt from a displaced value");
    *reinterpret_cast<socklen_t*>(host(0xa100)) = sizeof(sockaddr_un);
    Require(::getsockname(pair[0], reinterpret_cast<sockaddr*>(at(0xa300)), length) == 0 &&
            reinterpret_cast<sockaddr*>(host(0xa300))->sa_family == AF_UNIX,
        base, "getsockname into a displaced address");
    Require(::write(pair[1], "z", 1) == 1, base, "write to the pair");
    auto* fds = reinterpret_cast<pollfd*>(at(0xb000));
    *reinterpret_cast<pollfd*>(host(0xb000)) = {pair[0], POLLIN, 0};
    Require(::poll(fds, 1, 1000) == 1 && (reinterpret_cast<pollfd*>(host(0xb000))->revents & POLLIN) != 0, base, "poll on a displaced array");
    auto* readable = reinterpret_cast<fd_set*>(at(0xb100));
    FD_ZERO(reinterpret_cast<fd_set*>(host(0xb100)));
    FD_SET(pair[0], reinterpret_cast<fd_set*>(host(0xb100)));
    *reinterpret_cast<timeval*>(host(0xb200)) = {1, 0};
    Require(::select(pair[0] + 1, readable, nullptr, nullptr, reinterpret_cast<timeval*>(at(0xb200))) == 1 &&
            FD_ISSET(pair[0], reinterpret_cast<fd_set*>(host(0xb100))),
        base, "select on displaced fd_set and timeout");
    const int queue = kqueue();
    void* const udata = guest + 0x1234; // a guest pointer: must come back unchanged
    EV_SET(reinterpret_cast<struct kevent*>(host(0xc000)), pair[0], EVFILT_READ, EV_ADD, 0, 0, udata);
    *reinterpret_cast<timespec*>(host(0xc200)) = {1, 0};
    Require(::kevent(queue, reinterpret_cast<struct kevent*>(at(0xc000)), 1, reinterpret_cast<struct kevent*>(at(0xc100)), 1,
                reinterpret_cast<timespec*>(at(0xc200))) == 1 &&
            reinterpret_cast<struct kevent*>(host(0xc100))->udata == udata,
        base, "kevent with displaced arrays keeps the guest udata");
    ::close(queue);
    ::close(pair[0]);
    ::close(pair[1]);
    Require(::pipe(reinterpret_cast<int*>(at(0xd000))) == 0, base, "pipe into a displaced array");
    ::close(reinterpret_cast<int*>(host(0xd000))[0]);
    ::close(reinterpret_cast<int*>(host(0xd000))[1]);

    // sysctl and friends.
    auto* mib = reinterpret_cast<int*>(at(0xe000));
    reinterpret_cast<int*>(host(0xe000))[0] = CTL_KERN;
    reinterpret_cast<int*>(host(0xe000))[1] = KERN_OSTYPE;
    auto* valueLength = reinterpret_cast<std::size_t*>(at(0xe100));
    *reinterpret_cast<std::size_t*>(host(0xe100)) = 64;
    Require(::sysctl(mib, 2, at(0xe200), valueLength, nullptr, 0) == 0 && std::strcmp(reinterpret_cast<char*>(host(0xe200)), "Darwin") == 0, base,
        "sysctl with displaced MIB, value and length");
    *reinterpret_cast<std::size_t*>(host(0xe100)) = 64;
    std::memset(host(0xe200), 0, 64);
    Require(::sysctlbyname(GuestString(guest, alias, Scratch + 0xe300, "kern.ostype"), at(0xe200), valueLength, nullptr, 0) == 0 &&
            std::strcmp(reinterpret_cast<char*>(host(0xe200)), "Darwin") == 0,
        base, "sysctlbyname with displaced name, value and length");
    Require(::uname(reinterpret_cast<utsname*>(at(0xf000))) == 0 && std::strcmp(reinterpret_cast<utsname*>(host(0xf000))->sysname, "Darwin") == 0,
        base, "uname into a displaced struct");
    Require(::getrusage(RUSAGE_SELF, reinterpret_cast<rusage*>(at(0x10000))) == 0 &&
            reinterpret_cast<rusage*>(host(0x10000))->ru_maxrss > 0,
        base, "getrusage into a displaced struct");
    Require(::getrlimit(RLIMIT_NOFILE, reinterpret_cast<rlimit*>(at(0x10200))) == 0 &&
            reinterpret_cast<rlimit*>(host(0x10200))->rlim_cur > 0,
        base, "getrlimit into a displaced struct");
    Require(::statvfs(GuestString(guest, alias, Scratch + 0x10400, file.string()), reinterpret_cast<struct statvfs*>(at(0x10800))) == 0 &&
            reinterpret_cast<struct statvfs*>(host(0x10800))->f_bsize > 0,
        base, "statvfs with a displaced path and struct");

    // libkernel's fault-safe guest copies (Socket/Udp.cpp ReadGuest/WriteGuest) go through the task's VM map.
    for (std::size_t index = 0; index < 0x40; ++index) host(0x11100)[index] = Pattern(index);
    Require(::mach_vm_write(mach_task_self(), reinterpret_cast<mach_vm_address_t>(at(0x11000)),
                reinterpret_cast<vm_offset_t>(at(0x11100)), 0x40) == KERN_SUCCESS &&
            std::memcmp(host(0x11000), host(0x11100), 0x40) == 0,
        base, "mach_vm_write between displaced ranges");
    std::uint8_t copy[0x40]{};
    mach_vm_size_t copied = 0;
    Require(::mach_vm_read_overwrite(mach_task_self(), reinterpret_cast<mach_vm_address_t>(at(0x11000)), sizeof(copy),
                reinterpret_cast<mach_vm_address_t>(copy), &copied) == KERN_SUCCESS &&
            copied == sizeof(copy) && std::memcmp(copy, host(0x11100), sizeof(copy)) == 0,
        base, "mach_vm_read_overwrite from a displaced range");
    return formatFaults + printFaults;
}

void Run(std::uintptr_t base, const std::filesystem::path& file) {
    auto* guest = reinterpret_cast<std::uint8_t*>(base);
    auto* alias = reinterpret_cast<std::uint8_t*>(GuestArena::HostAddress(base));
    for (std::size_t i = 0; i < Size; ++i) alias[i] = Pattern(i);
    std::memset(alias + Size, 0, Bytes - Size);
    const auto before = Faults();

    FILE* stream = std::fopen(file.c_str(), "w+b");
    Require(stream != nullptr, base, "open the scratch file");
    const int fd = fileno(stream);

    using Read = ssize_t (*)(int, void*, std::size_t);
    const auto libSystemRead = reinterpret_cast<Read>(dlsym(dlopen("/usr/lib/libSystem.B.dylib", RTLD_LAZY | RTLD_NOLOAD), "read"));
    Require(libSystemRead != nullptr, base, "libSystem read");
    Require(::write(fd, guest, Size) == static_cast<ssize_t>(Size), base, "write from a displaced buffer");
    Require(lseek(fd, 0, SEEK_SET) == 0, base, "lseek");
    errno = 0;
    Require(libSystemRead(fd, guest + Size, 16) == -1 && errno == EFAULT, base, "libSystem's read at the guest address did not fail with EFAULT");
    Require(::read(fd, guest + Size, Size) == static_cast<ssize_t>(Size) && std::memcmp(alias + Size, alias, Size) == 0, base,
        "read into a displaced buffer");
    std::memset(alias + Size, 0, Size);
    Require(::pread(fd, guest + Size, 0x1000, 0x2000) == 0x1000 && std::memcmp(alias + Size, alias + 0x2000, 0x1000) == 0, base,
        "pread into a displaced buffer");
    Require(::pwrite(fd, guest + 0x5000, 0x1000, 0) == 0x1000, base, "pwrite from a displaced buffer");

    // The iovec array lives in the window too, and so do its bases.
    auto* vectors = reinterpret_cast<iovec*>(alias + 2 * Size);
    vectors[0] = {guest + Size, 0x800};
    vectors[1] = {guest + Size + 0x3000, 0x800};
    auto* guestVectors = reinterpret_cast<iovec*>(guest + 2 * Size);
    std::memset(alias + Size, 0, 0x4000);
    Require(::preadv(fd, guestVectors, 2, 0) == 0x1000 && std::memcmp(alias + Size, alias + 0x5000, 0x800) == 0 &&
            std::memcmp(alias + Size + 0x3000, alias + 0x5800, 0x800) == 0,
        base, "preadv into displaced buffers");
    Require(lseek(fd, 0, SEEK_SET) == 0, base, "lseek");
    std::memset(alias + Size, 0, 0x4000);
    Require(::readv(fd, guestVectors, 2) == 0x1000 && alias[Size + 0x3000 + 0x7ff] == Pattern(0x5fff), base, "readv into displaced buffers");
    Require(::pwritev(fd, guestVectors, 2, 0) == 0x1000, base, "pwritev from displaced buffers");
    Require(lseek(fd, 0, SEEK_SET) == 0 && ::writev(fd, guestVectors, 2) == 0x1000, base, "writev from displaced buffers");

    // stdio past its buffer size reads and writes the caller's buffer directly.
    Require(std::fseek(stream, 0, SEEK_SET) == 0 && std::fwrite(guest, 1, Size, stream) == Size && std::fflush(stream) == 0, base,
        "fwrite from a displaced buffer");
    std::memset(alias + Size, 0, Size);
    Require(std::fseek(stream, 0, SEEK_SET) == 0 && std::fread(guest + Size, 1, Size, stream) == Size && std::memcmp(alias + Size, alias, Size) == 0,
        base, "fread into a displaced buffer");
    std::fclose(stream);

    int pair[2] = {-1, -1};
    Require(socketpair(AF_UNIX, SOCK_DGRAM, 0, pair) == 0, base, "socketpair");
    std::memset(alias + Size, 0, 0x1000);
    Require(::send(pair[0], guest, 0x400, 0) == 0x400 && ::recv(pair[1], guest + Size, 0x1000, 0) == 0x400 &&
            std::memcmp(alias + Size, alias, 0x400) == 0,
        base, "send/recv with displaced buffers");
    auto* length = reinterpret_cast<socklen_t*>(guest + 2 * Size + 0x100);
    *reinterpret_cast<socklen_t*>(alias + 2 * Size + 0x100) = sizeof(sockaddr_storage);
    Require(::sendto(pair[0], guest + 0x400, 0x200, 0, nullptr, 0) == 0x200 &&
            ::recvfrom(pair[1], guest + Size, 0x1000, 0, reinterpret_cast<sockaddr*>(guest + 2 * Size + 0x200), length) == 0x200 &&
            std::memcmp(alias + Size, alias + 0x400, 0x200) == 0,
        base, "sendto/recvfrom with displaced buffers and address length");
    close(pair[0]);
    close(pair[1]);

    std::memset(alias + Size, 0, 64);
    Require(::getentropy(guest + Size, 64) == 0, base, "getentropy into a displaced buffer");
    bool nonzero = false;
    for (std::size_t i = 0; i < 64; ++i) nonzero |= alias[Size + i] != 0;
    Require(nonzero, base, "getentropy wrote nothing");

    std::uint8_t host[16] {};
    int hostPair[2] = {-1, -1};
    Require(pipe(hostPair) == 0 && ::write(hostPair[1], alias, 16) == 16 && ::read(hostPair[0], host, 16) == 16 && std::memcmp(host, alias, 16) == 0, base,
        "host and alias buffers pass through");
    close(hostPair[0]);
    close(hostPair[1]);

    const auto formatFaults = RunKernelStructs(base, guest, alias, file);
    Require(Faults() == before + formatFaults, base, "the shim's calls redirected faults");
    std::filesystem::remove(file);
}

}

#if defined(__APPLE__) && defined(__aarch64__)
extern "C" void Aps5HostSyscallShimResolution(std::size_t* resolved, std::size_t* total); // the shim archive's

namespace {

// Every libSystem function the shim forwards to is resolved at load, before main runs any of them: a first call can
// come from a signal handler, where a lazy dlsym could deadlock on a lock the interrupted thread holds (D35).
void ResolvedAtLoad() {
    std::size_t resolved = 0, total = 0;
    Aps5HostSyscallShimResolution(&resolved, &total);
    if (total < 70) throw std::runtime_error("the shim forwards to only " + std::to_string(total) + " libSystem functions");
    if (resolved != total) throw std::runtime_error(std::to_string(total - resolved) + " of " + std::to_string(total) + " forwarded libSystem functions were not resolved at load");
    std::printf("host system call shim: %zu of %zu forwarded functions resolved at load\n", resolved, total);
}

}
#endif

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("host system call shim: no displaced window on this host, skipped");
    return 0;
#else
    try {
        ResolvedAtLoad(); // first: nothing in main has called a shimmed function yet
        if (!GuestArena::GuestDisplacedActive_nid_no_patch()) throw std::runtime_error("the displaced window is not active on macOS arm64");
        const auto file = std::filesystem::path("anyps5-host-syscall-shim-" + std::to_string(::getpid()));
        std::int64_t physical = 0;
        if (sceKernelAllocateDirectMemory(0, 0x100000000ll, Bytes, 0x200000, 3, &physical) != 0) throw std::runtime_error("allocate direct memory");
        for (const std::uintptr_t base : {TailBase, CarveoutBase}) {
            Require(GuestArena::IsDisplaced(base), base, "not displaced");
            void* fixed = reinterpret_cast<void*>(base);
            Require(sceKernelMapDirectMemory(&fixed, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && fixed == reinterpret_cast<void*>(base), base,
                "the fixed mapping failed");
            Run(base, file);
            Require(sceKernelMunmap(fixed, Bytes) == 0, base, "munmap");
            std::printf("host system call shim at 0x%llx passed\n", static_cast<unsigned long long>(base));
        }
        if (sceKernelReleaseDirectMemory(physical, Bytes) != 0) throw std::runtime_error("release direct memory");
        std::puts("host system call shim tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "host system call shim: %s\n", error.what());
        return 1;
    }
#endif
}
