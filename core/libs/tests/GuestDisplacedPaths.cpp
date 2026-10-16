// Native paths that hand a guest buffer to the host kernel or move guest memory in bulk, at displaced guest addresses
// (D31, D33). A displaced range's guest address is a guard page: a host system call given it fails (EFAULT, ENOMEM, a
// guard region) without raising the signal the fault handler redirects, and bulk native copies through it take two
// signals per load/store. Each path must translate to the alias (guest | DisplacedFlip) up front, so the checks
// require both the right result and no redirected faults. Run at the shared region tail (0x400000000) and the GPU
// carveout (0x4000000000). Covered: mlock/munlock, VirtualQuery of unregistered memory, sceKernelRead/Write/
// Pread/Pwrite/Readv/Writev/Preadv, AIO reads and writes, libc fread/fwrite, the libc memory and string exports,
// host entropy (/dev/urandom reads, sceRandom) and mspace calloc/realloc over a displaced mspace.
#include "SceTypes.hpp"
#include "prx/libc/include/FileStream.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libkernel/File/include/FileFlags.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
int APS5_VABI sceKernelVirtualQuery(const void*, int, VirtualQueryInfo*, std::uint64_t);
int APS5_VABI sceKernelMlock_nid_postfix(void*, std::uint64_t);
int APS5_VABI sceKernelMunlock_nid_postfix(void*, std::uint64_t);
int APS5_VABI sceKernelOpen(const char*, int, std::uint16_t);
int APS5_VABI sceKernelClose(int);
std::int64_t APS5_VABI sceKernelRead(int, void*, std::size_t);
std::int64_t APS5_VABI sceKernelWrite(int, const void*, std::size_t);
std::int64_t APS5_VABI sceKernelPread(int, void*, std::size_t, std::int64_t);
std::int64_t APS5_VABI sceKernelPwrite(int, const void*, std::size_t, std::int64_t);
std::int64_t APS5_VABI sceKernelLseek(int, std::int64_t, int);
std::int64_t APS5_VABI sceKernelReadv(int, const void*, int);
std::int64_t APS5_VABI sceKernelWritev(int, const void*, int);
std::int64_t APS5_VABI sceKernelPreadv(int, const void*, int, std::int64_t);
int APS5_VABI sceKernelAioSubmitReadCommands(KernelAioRwRequest*, std::int32_t, std::int32_t, std::int32_t*);
int APS5_VABI sceKernelAioSubmitWriteCommands(KernelAioRwRequest*, std::int32_t, std::int32_t, std::int32_t*);
int APS5_VABI sceKernelAioWaitRequest(std::int32_t, std::int32_t*, std::uint32_t*);
int APS5_VABI sceKernelAioDeleteRequest(std::int32_t, std::int32_t*);
FileStream* APS5_VABI fopen_nid_postfix(const char*, const char*);
int APS5_VABI fclose_nid_postfix(FileStream*);
std::size_t APS5_VABI fread_nid_postfix(void*, std::size_t, std::size_t, FileStream*);
std::size_t APS5_VABI fwrite_nid_postfix(const void*, std::size_t, std::size_t, FileStream*);
int APS5_VABI fseek_nid_postfix(FileStream*, std::int64_t, int);
void* APS5_VABI memset_nid_postfix(void*, int, std::size_t);
void* APS5_VABI memcpy_nid_postfix(void*, const void*, std::size_t);
void* APS5_VABI memmove_nid_postfix(void*, const void*, std::size_t);
int APS5_VABI memcmp_nid_postfix(const void*, const void*, std::size_t);
const void* APS5_VABI memchr_nid_postfix(const void*, int, std::size_t);
std::size_t APS5_VABI strlen_nid_postfix(const char*);
char* APS5_VABI strcpy_nid_postfix(char*, const char*);
void HostEntropy_nid_no_patch(void*, std::size_t);
void* APS5_VABI sceLibcMspaceCreate_nid_postfix(const char*, void*, std::size_t, unsigned);
int APS5_VABI sceLibcMspaceDestroy_nid_postfix(void*);
void* APS5_VABI sceLibcMspaceMalloc_nid_postfix(void*, std::size_t);
void* APS5_VABI sceLibcMspaceCalloc_nid_postfix(void*, std::size_t, std::size_t);
void* APS5_VABI sceLibcMspaceRealloc_nid_postfix(void*, void*, std::size_t);
void APS5_VABI sceLibcMspaceFree_nid_postfix(void*, void*);
}

namespace {

constexpr std::uintptr_t CarveoutBase = 0x4000000000ull;
constexpr std::uintptr_t TailBase = 0x400000000ull;
constexpr std::size_t Bytes = 0x200000;
constexpr int CpuGpuReadWrite = 0x33;
constexpr int MapFixed = 0x10;
constexpr std::uint32_t ReadWrite = 0x04; // PAGE_READWRITE, as GuestArenaCommit takes it

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string At(std::uintptr_t base, const char* what) {
    char text[160];
    std::snprintf(text, sizeof(text), "0x%llx: %s", static_cast<unsigned long long>(base), what);
    return text;
}

std::uint64_t Faults() {
    std::uint64_t faults = 0;
    GuestArena::GuestDisplacedStats_nid_no_patch(&faults, nullptr);
    return faults;
}

struct Iovec {
    void* base;
    std::size_t length;
};

std::uint8_t Pattern(std::size_t index) {
    return static_cast<std::uint8_t>(index * 37u + 11u);
}

// Host system calls with a displaced guest buffer: each must succeed (at the guest address the kernel returns EFAULT)
// and move the bytes the alias holds.
void FilePaths(std::uintptr_t base, const std::filesystem::path& file) {
    constexpr std::size_t Size = 0x30000; // over Darwin stdio's buffer, so fread/fwrite hit read(2)/write(2) directly
    auto* guest = reinterpret_cast<std::uint8_t*>(base);
    auto* alias = reinterpret_cast<std::uint8_t*>(GuestArena::HostAddress(base));
    for (std::size_t i = 0; i < Size; ++i) alias[i] = Pattern(i);
    const auto before = Faults();

    { std::ofstream create(file, std::ios::binary | std::ios::trunc); }
    const int fd = sceKernelOpen(file.string().c_str(), SCE_KERNEL_O_RDWR, 0);
    Require(fd >= 0, At(base, "open the scratch file"));
    Require(sceKernelWrite(fd, guest, Size) == static_cast<std::int64_t>(Size), At(base, "sceKernelWrite from a displaced buffer"));
    Require(sceKernelLseek(fd, 0, 0) == 0, At(base, "lseek"));
    std::memset(alias + Size, 0, Size);
    Require(sceKernelRead(fd, guest + Size, Size) == static_cast<std::int64_t>(Size), At(base, "sceKernelRead into a displaced buffer"));
    Require(std::memcmp(alias + Size, alias, Size) == 0, At(base, "sceKernelRead data"));
    std::memset(alias + Size, 0, 0x1000);
    Require(sceKernelPread(fd, guest + Size, 0x1000, 0x2000) == 0x1000 && std::memcmp(alias + Size, alias + 0x2000, 0x1000) == 0,
        At(base, "sceKernelPread into a displaced buffer"));
    Require(sceKernelPwrite(fd, guest + 0x5000, 0x1000, 0) == 0x1000, At(base, "sceKernelPwrite from a displaced buffer"));
    // Vectors: the iovec array itself lives in ordinary memory, its bases in the window.
    Iovec vectors[2] = {{guest + Size, 0x800}, {guest + Size + 0x3000, 0x800}};
    std::memset(alias + Size, 0, 0x4000);
    Require(sceKernelPreadv(fd, vectors, 2, 0) == 0x1000, At(base, "sceKernelPreadv into displaced buffers"));
    Require(std::memcmp(alias + Size, alias + 0x5000, 0x800) == 0 && std::memcmp(alias + Size + 0x3000, alias + 0x5800, 0x800) == 0,
        At(base, "sceKernelPreadv data"));
    Require(sceKernelLseek(fd, 0, 0) == 0, At(base, "lseek"));
    std::memset(alias + Size, 0, 0x4000);
    Require(sceKernelReadv(fd, vectors, 2) == 0x1000 && alias[Size + 0x3000 + 0x7ff] == Pattern(0x5fff), At(base, "sceKernelReadv into displaced buffers"));
    Iovec sources[2] = {{guest, 0x400}, {guest + 0x400, 0x400}};
    Require(sceKernelLseek(fd, 0, 0) == 0 && sceKernelWritev(fd, sources, 2) == 0x800, At(base, "sceKernelWritev from displaced buffers"));

    // AIO reads and writes run pread/pwrite on a worker.
    KernelAioResult result{-1, 0};
    KernelAioRwRequest read{0x100, 0x1000, guest + Size, &result, fd};
    std::memset(alias + Size, 0, 0x1000);
    std::int32_t id = 0, state = 0, deleted = 0;
    Require(sceKernelAioSubmitReadCommands(&read, 1, 0, &id) == 0 && sceKernelAioWaitRequest(id, &state, nullptr) == 0, At(base, "AIO read"));
    Require(result.state == 3 && result.return_value == 0x1000 && std::memcmp(alias + Size, alias + 0x100, 0x1000) == 0, At(base, "AIO read into a displaced buffer"));
    sceKernelAioDeleteRequest(id, &deleted);
    KernelAioRwRequest write{0x8000, 0x1000, guest + 0x9000, &result, fd};
    Require(sceKernelAioSubmitWriteCommands(&write, 1, 0, &id) == 0 && sceKernelAioWaitRequest(id, &state, nullptr) == 0, At(base, "AIO write"));
    Require(result.state == 3 && result.return_value == 0x1000, At(base, "AIO write from a displaced buffer"));
    sceKernelAioDeleteRequest(id, &deleted);
    std::memset(alias + Size, 0, 0x1000);
    Require(sceKernelPread(fd, guest + Size, 0x1000, 0x8000) == 0x1000 && std::memcmp(alias + Size, alias + 0x9000, 0x1000) == 0, At(base, "AIO write data"));
    Require(sceKernelClose(fd) == 0, At(base, "close"));

    // libc stdio.
    auto* stream = fopen_nid_postfix(file.string().c_str(), "w+b");
    Require(stream != nullptr, At(base, "fopen"));
    Require(fwrite_nid_postfix(guest, 1, Size, stream) == Size, At(base, "fwrite from a displaced buffer"));
    Require(fseek_nid_postfix(stream, 0, SEEK_SET) == 0, At(base, "fseek"));
    std::memset(alias + Size, 0, Size);
    Require(fread_nid_postfix(guest + Size, 1, Size, stream) == Size && std::memcmp(alias + Size, alias, Size) == 0, At(base, "fread into a displaced buffer"));
    Require(fclose_nid_postfix(stream) == 0, At(base, "fclose"));

    // getentropy (the /dev/urandom device, sceRandomGetRandomNumber, arc4random_buf).
    std::memset(alias + Size, 0, 256);
    HostEntropy_nid_no_patch(guest + Size, 256);
    bool any = false;
    for (std::size_t i = 0; i < 256; ++i) any = any || alias[Size + i] != 0;
    Require(any, At(base, "host entropy into a displaced buffer"));
    Require(Faults() == before, At(base, "a file path touched the guest address instead of the alias"));
}

// The libc memory and string exports run host code over guest pointers and return the guest's pointers.
void LibcPaths(std::uintptr_t base) {
    auto* guest = reinterpret_cast<char*>(base);
    auto* alias = reinterpret_cast<char*>(GuestArena::HostAddress(base));
    const auto before = Faults();
    Require(memset_nid_postfix(guest, 0x5a, 0x10000) == guest && alias[0] == 0x5a && alias[0xffff] == 0x5a, At(base, "memset"));
    for (std::size_t i = 0; i < 0x10000; ++i) alias[0x10000 + i] = static_cast<char>(Pattern(i));
    Require(memcpy_nid_postfix(guest, guest + 0x10000, 0x10000) == guest && std::memcmp(alias, alias + 0x10000, 0x10000) == 0, At(base, "memcpy"));
    Require(memmove_nid_postfix(guest + 1, guest, 0x1000) == guest + 1 && alias[1] == static_cast<char>(Pattern(0)), At(base, "memmove"));
    Require(memcmp_nid_postfix(guest + 0x10000 + 0x2000, guest + 0x2000, 0x8000) == 0, At(base, "memcmp"));
    std::memset(alias + 0x20000, 0, 0x1000);
    alias[0x20100] = 'x';
    Require(memchr_nid_postfix(guest + 0x20000, 'x', 0x1000) == guest + 0x20100, At(base, "memchr did not return the guest pointer"));
    std::strcpy(alias + 0x30000, "displaced");
    Require(strlen_nid_postfix(guest + 0x30000) == 9, At(base, "strlen"));
    Require(strcpy_nid_postfix(guest + 0x30100, guest + 0x30000) == guest + 0x30100 && std::strcmp(alias + 0x30100, "displaced") == 0, At(base, "strcpy"));
    Require(Faults() == before, At(base, "a libc export touched the guest address instead of the alias"));
}

// A guest mspace laid over displaced memory: calloc clears and realloc copies through the alias.
void MspacePaths(std::uintptr_t base) {
    auto* alias = reinterpret_cast<std::uint8_t*>(GuestArena::HostAddress(base));
    std::memset(alias, 0xee, 0x100000);
    void* space = sceLibcMspaceCreate_nid_postfix("displaced", reinterpret_cast<void*>(base), 0x100000, 0);
    Require(space == reinterpret_cast<void*>(base), At(base, "mspace create"));
    const auto before = Faults();
    auto* cleared = static_cast<std::uint8_t*>(sceLibcMspaceCalloc_nid_postfix(space, 0x100, 0x100));
    Require(cleared != nullptr && GuestArena::IsDisplaced(reinterpret_cast<std::uintptr_t>(cleared)), At(base, "mspace calloc"));
    auto* clearedAlias = GuestArena::HostPointer(cleared);
    bool zero = true;
    for (std::size_t i = 0; i < 0x10000; ++i) zero = zero && clearedAlias[i] == 0;
    Require(zero, At(base, "mspace calloc did not clear"));
    for (std::size_t i = 0; i < 0x10000; ++i) clearedAlias[i] = Pattern(i);
    auto* moved = static_cast<std::uint8_t*>(sceLibcMspaceRealloc_nid_postfix(space, cleared, 0x40000));
    Require(moved != nullptr, At(base, "mspace realloc"));
    const auto* movedAlias = GuestArena::HostPointer(moved);
    bool same = true;
    for (std::size_t i = 0; i < 0x10000; ++i) same = same && movedAlias[i] == Pattern(i);
    Require(same, At(base, "mspace realloc lost the contents"));
    Require(Faults() == before, At(base, "mspace touched the guest address instead of the alias"));
    sceLibcMspaceFree_nid_postfix(space, moved);
    Require(sceLibcMspaceDestroy_nid_postfix(space) == 0, At(base, "mspace destroy"));
}

void LockPaths(std::uintptr_t base, std::int64_t physical) {
    void* fixed = reinterpret_cast<void*>(base);
    Require(sceKernelMapDirectMemory(&fixed, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && fixed == reinterpret_cast<void*>(base),
        At(base, "the fixed mapping failed"));
    // mlock and munlock reach the kernel with the alias: at the guest address the kernel sees the guard and refuses.
    Require(sceKernelMlock_nid_postfix(fixed, 0x8000) == 0, At(base, "sceKernelMlock of a displaced mapping failed"));
    Require(sceKernelMunlock_nid_postfix(fixed, 0x8000) == 0, At(base, "sceKernelMunlock of a displaced mapping failed"));
    Require(sceKernelMunmap(fixed, Bytes) == 0, At(base, "munmap"));
    // Unmapped again: the alias is inaccessible, so mlock reports ENOMEM as for any unmapped range.
    Require(sceKernelMlock_nid_postfix(fixed, 0x4000) != 0, At(base, "sceKernelMlock of an unmapped displaced range succeeded"));
}

// Memory the allocation registry does not know is described from the host's region around it, queried at the alias.
void QueryPath(std::uintptr_t base) {
    auto* guest = reinterpret_cast<void*>(base);
    GuestArena::GuestArenaCommit_nid_postfix(guest, Bytes, ReadWrite, 0x4000);
    const auto before = Faults();
    VirtualQueryInfo info{};
    Require(sceKernelVirtualQuery(reinterpret_cast<const void*>(base + 0x4000), 0, &info, sizeof(info)) == 0,
        At(base, "sceKernelVirtualQuery of unregistered displaced memory failed"));
    Require(info.start <= base + 0x4000 && info.end > base + 0x4000, At(base, "the query does not cover the address"));
    Require(GuestArena::IsDisplaced(info.start) && info.end <= GuestArena::DisplacedEnd, At(base, "the query reported the alias instead of guest addresses"));
    Require((info.protection & 3) == 3, At(base, "the query did not report read/write"));
    Require(Faults() == before, At(base, "the query touched the guest address"));
    GuestArena::GuestArenaReset_nid_postfix(guest, Bytes);
    GuestArena::GuestArenaRelease_nid_postfix(guest, Bytes);
}

}

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("guest displaced paths: no displaced window on this host, skipped");
    return 0;
#else
    try {
        Require(GuestArena::GuestDisplacedActive_nid_no_patch(), "the displaced window is not active on macOS arm64");
        const auto scratch = std::filesystem::path("anyps5-displaced-paths-" + std::to_string(::getpid())); // relative: guest paths resolve absolute ones
        std::int64_t physical = 0;
        Require(sceKernelAllocateDirectMemory(0, 0x100000000ll, Bytes, 0x200000, 3, &physical) == 0, "allocate direct memory");
        for (const std::uintptr_t base : {TailBase, CarveoutBase}) {
            Require(GuestArena::IsDisplaced(base), At(base, "not displaced"));
            LockPaths(base, physical);
            QueryPath(base);
            void* fixed = reinterpret_cast<void*>(base);
            Require(sceKernelMapDirectMemory(&fixed, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && fixed == reinterpret_cast<void*>(base),
                At(base, "the fixed mapping failed"));
            FilePaths(base, scratch);
            LibcPaths(base);
            MspacePaths(base);
            Require(sceKernelMunmap(fixed, Bytes) == 0, At(base, "munmap"));
            std::printf("guest displaced paths at 0x%llx passed\n", static_cast<unsigned long long>(base));
        }
        std::filesystem::remove(scratch);
        Require(sceKernelReleaseDirectMemory(physical, Bytes) == 0, "release direct memory");
        std::puts("guest displaced paths tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "guest displaced paths: %s\n", error.what());
        return 1;
    }
#endif
}
