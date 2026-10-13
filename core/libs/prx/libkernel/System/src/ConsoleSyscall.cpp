#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" {
int* APS5_VABI __error_nid_postfix();
std::int64_t APS5_VABI read_nid_postfix(int d, void* buf, std::uint64_t nbytes);
std::int64_t APS5_VABI write_nid_postfix(int d, const char* str, std::int64_t size);
int APS5_VABI open_nid_postfix(const char* path, int flags, int mode);
int APS5_VABI close_nid_postfix(int d);
int APS5_VABI unlink_nid_postfix(const char* path);
int APS5_VABI chmod_nid_postfix(const char* path, int mode);
int APS5_VABI getpid_nid_postfix();
int APS5_VABI ioctl_nid_postfix(int descriptor, std::uint64_t request, void* argument);
int APS5_VABI munmap_nid_postfix(void* address, std::size_t length) noexcept;
int APS5_VABI socket_nid_postfix(int family, int type, int protocol);
int APS5_VABI connect_nid_postfix(int descriptor, const void* address, std::uint32_t length);
int APS5_VABI rename_nid_postfix(const char* from, const char* to);
int APS5_VABI mkdir_nid_postfix(const char* path, std::uint16_t mode);
int APS5_VABI rmdir_nid_postfix(const char* path);
int APS5_VABI clock_gettime_nid_postfix(int clockId, KernelTimespec* tp);
int APS5_VABI getdents_nid_postfix(int fd, char* buf, int nbytes);
void* APS5_VABI mmap_nid_postfix(void* address, std::size_t length, int protection, int flags, int descriptor, std::int64_t offset) noexcept;
std::int64_t APS5_VABI lseek_nid_postfix(int d, std::int64_t offset, int whence);
}

namespace {

constexpr std::uint64_t SysRead = 3;
constexpr std::uint64_t SysWrite = 4;
constexpr std::uint64_t SysOpen = 5;
constexpr std::uint64_t SysClose = 6;
constexpr std::uint64_t SysUnlink = 10;
constexpr std::uint64_t SysChmod = 15;
constexpr std::uint64_t SysGetpid = 20;
constexpr std::uint64_t SysIoctl = 54;
constexpr std::uint64_t SysMunmap = 73;
constexpr std::uint64_t SysSocket = 97;
constexpr std::uint64_t SysConnect = 98;
constexpr std::uint64_t SysRename = 128;
constexpr std::uint64_t SysMkdir = 136;
constexpr std::uint64_t SysRmdir = 137;
constexpr std::uint64_t SysClockGettime = 232;
constexpr std::uint64_t SysGetdents = 272;
constexpr std::uint64_t SysMmap = 477;
constexpr std::uint64_t SysLseek = 478;
constexpr std::uint64_t SysKlog = 601;

std::int64_t SyscallResult(std::int64_t value) {
    if (value != -1) return value;
    const int error = *__error_nid_postfix();
    if (error <= 0) throw std::runtime_error("a console system call failed without an errno");
    return -error;
}

const char* Text(std::uint64_t value) { return reinterpret_cast<const char*>(value); }
void* Pointer(std::uint64_t value) { return reinterpret_cast<void*>(value); }
int Int(std::uint64_t value) { return static_cast<int>(value); }

}

extern "C" std::int64_t APS5_VABI Aps5ConsoleSyscall_nid_no_patch(const std::uint64_t* frame) {
    const std::uint64_t number = frame[0];
    const std::uint64_t* a = frame + 1;
    switch (number) {
    case SysRead: return SyscallResult(read_nid_postfix(Int(a[0]), Pointer(a[1]), a[2]));
    case SysWrite: return SyscallResult(write_nid_postfix(Int(a[0]), Text(a[1]), static_cast<std::int64_t>(a[2])));
    case SysOpen: return SyscallResult(open_nid_postfix(Text(a[0]), Int(a[1]), Int(a[2])));
    case SysClose: return SyscallResult(close_nid_postfix(Int(a[0])));
    case SysUnlink: return SyscallResult(unlink_nid_postfix(Text(a[0])));
    case SysChmod: return SyscallResult(chmod_nid_postfix(Text(a[0]), Int(a[1])));
    case SysGetpid: return SyscallResult(getpid_nid_postfix());
    case SysIoctl: return SyscallResult(ioctl_nid_postfix(Int(a[0]), a[1], Pointer(a[2])));
    case SysMunmap: return SyscallResult(munmap_nid_postfix(Pointer(a[0]), a[1]));
    case SysSocket: return SyscallResult(socket_nid_postfix(Int(a[0]), Int(a[1]), Int(a[2])));
    case SysConnect: return SyscallResult(connect_nid_postfix(Int(a[0]), Pointer(a[1]), static_cast<std::uint32_t>(a[2])));
    case SysRename: return SyscallResult(rename_nid_postfix(Text(a[0]), Text(a[1])));
    case SysMkdir: return SyscallResult(mkdir_nid_postfix(Text(a[0]), static_cast<std::uint16_t>(a[1])));
    case SysRmdir: return SyscallResult(rmdir_nid_postfix(Text(a[0])));
    case SysClockGettime: return SyscallResult(clock_gettime_nid_postfix(Int(a[0]), static_cast<KernelTimespec*>(Pointer(a[1]))));
    case SysGetdents: return SyscallResult(getdents_nid_postfix(Int(a[0]), static_cast<char*>(Pointer(a[1])), Int(a[2])));
    case SysMmap: {
        void* mapped = mmap_nid_postfix(Pointer(a[0]), a[1], Int(a[2]), Int(a[3]), Int(a[4]), static_cast<std::int64_t>(a[5]));
        return SyscallResult(reinterpret_cast<std::int64_t>(mapped));
    }
    case SysLseek: return SyscallResult(lseek_nid_postfix(Int(a[0]), static_cast<std::int64_t>(a[1]), Int(a[2])));
    case SysKlog: {
        if (a[1] == 0) throw std::invalid_argument("console system call klog: null text");
        const char* text = Text(a[1]);
        const std::size_t length = std::strlen(text);
        std::fwrite(text, 1, length, stderr);
        if (length == 0 || text[length - 1] != '\n') std::fputc('\n', stderr);
        std::fflush(stderr);
        return 0;
    }
    default: throw std::runtime_error("console system call " + std::to_string(number) + " is not supported");
    }
}
