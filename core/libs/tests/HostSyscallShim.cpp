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
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/uio.h>
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

    Require(Faults() == before, base, "the shim's calls redirected faults");
    std::filesystem::remove(file);
}

}

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("host system call shim: no displaced window on this host, skipped");
    return 0;
#else
    try {
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
