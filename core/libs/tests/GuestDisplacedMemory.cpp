// Fixed guest mappings inside the macOS arm64 GPU carveout (D31): the kernel keeps 64-448 GiB as a permanent
// guard in every process, so guest memory there is backed at guest | DisplacedFlip and native accesses to the guest
// address are redirected by the fault handler. Translated (FEX) accesses are covered by the `displaced` fixture.
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
int APS5_VABI sceKernelMprotect(const void*, std::size_t, int);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
}

namespace {

constexpr std::uintptr_t Base = 0x4000000000ull;
constexpr std::size_t Bytes = 0x400000;
constexpr int CpuGpuReadWrite = 0x33;
constexpr int CpuRead = 0x01;
constexpr int MapFixed = 0x10;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
T Load(std::uintptr_t address) {
    return *reinterpret_cast<const volatile T*>(address);
}

template <typename T>
void Store(std::uintptr_t address, T value) {
    *reinterpret_cast<volatile T*>(address) = value;
}

std::uint64_t Faults() {
    std::uint64_t faults = 0;
    GuestArena::GuestDisplacedStats_nid_no_patch(&faults, nullptr);
    return faults;
}

#if defined(__APPLE__)
int HostProtection(std::uintptr_t address) {
    mach_vm_address_t region = address;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || region > address) return -1;
    return info.protection;
}
#endif

}

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("guest displaced memory: no displaced window on this host, skipped");
    return 0;
#else
    try {
        Require(GuestArena::GuestArenaAvailable_nid_postfix(), "the guest arena is unavailable");
        Require(GuestArena::GuestDisplacedActive_nid_no_patch(), "the displaced window is not active on macOS arm64");
        Require(GuestArena::HostAddress(Base) == (Base | GuestArena::DisplacedFlip) && GuestArena::GuestAddress(Base | GuestArena::DisplacedFlip) == Base,
            "guest and host addresses do not round-trip");
        Require(GuestArena::HostAddress(0x900000000ull) == 0x900000000ull, "an address outside the window was translated");

        std::int64_t physical = 0;
        Require(sceKernelAllocateDirectMemory(0, 0x100000000ll, Bytes, 0x200000, 3, &physical) == 0, "allocate direct memory");
        void* fixed = reinterpret_cast<void*>(Base);
        Require(sceKernelMapDirectMemory(&fixed, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && fixed == reinterpret_cast<void*>(Base),
            "the fixed mapping at 0x4000000000 failed");
        void* view = nullptr;
        Require(sceKernelMapDirectMemory(&view, Bytes, CpuGpuReadWrite, 0, physical, 0x200000) == 0, "the second view failed");
        const auto viewAddress = reinterpret_cast<std::uintptr_t>(view);
        Require(!GuestArena::OverlapsDisplaced(viewAddress, Bytes), "first fit placed a mapping in the displaced window");

        // The alias holds the pages; native code reaches them through the guest address via the fault handler.
        const auto alias = GuestArena::HostAddress(Base);
        Store<std::uint64_t>(alias + 8, 0x1122334455667788ull);
        const auto before = Faults();
        Require(Load<std::uint64_t>(Base + 8) == 0x1122334455667788ull, "a native load from the guest address missed the alias");
        Store<std::uint32_t>(Base + 0x104, 0xa5a5a5a5u);
        Require(Load<std::uint32_t>(alias + 0x104) == 0xa5a5a5a5u, "a native store to the guest address missed the alias");
        Require(Load<std::uint32_t>(viewAddress + 0x104) == 0xa5a5a5a5u, "the second view does not share the pages");
        Require(Faults() >= before + 2, "the accesses did not go through the fault handler");

        // Library code (SIMD pairs, post-index addressing) on the guest address.
        char pattern[4096];
        for (std::size_t i = 0; i < sizeof(pattern); ++i) pattern[i] = static_cast<char>(i * 7 + 1);
        std::memcpy(reinterpret_cast<void*>(Base + 0x10000), pattern, sizeof(pattern));
        Require(std::memcmp(reinterpret_cast<const void*>(viewAddress + 0x10000), pattern, sizeof(pattern)) == 0, "memcpy to the guest address");
        char back[4096];
        std::memcpy(back, reinterpret_cast<const void*>(Base + 0x10000), sizeof(back));
        Require(std::memcmp(back, pattern, sizeof(back)) == 0, "memcpy from the guest address");

        // Protection is applied to the alias.
        Require(sceKernelMprotect(fixed, 0x4000, CpuRead) == 0, "mprotect in the window");
        Require(HostProtection(alias) == VM_PROT_READ, "mprotect did not reach the alias");
        Require(sceKernelMprotect(fixed, 0x4000, CpuGpuReadWrite) == 0, "mprotect back to read/write");

        // Unmapping leaves the alias reserved and inaccessible; mapping again gives the same pages.
        Require(sceKernelMunmap(fixed, Bytes) == 0, "munmap in the window");
        Require(HostProtection(alias) == VM_PROT_NONE, "the unmapped alias is still accessible");
        void* again = reinterpret_cast<void*>(Base);
        Require(sceKernelMapDirectMemory(&again, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && again == fixed, "mapping again in the window");
        Require(Load<std::uint32_t>(Base + 0x104) == 0xa5a5a5a5u, "the mapping again lost the direct memory contents");

        // A range across the window's edge has no single backing.
        bool refused = false;
        try {
            GuestArena::GuestArenaMarkUsed_nid_postfix(reinterpret_cast<void*>(GuestArena::DisplacedEnd - 0x4000), 0x8000);
        } catch (const std::runtime_error& error) {
            refused = std::string(error.what()).find("straddles") != std::string::npos;
        }
        Require(refused, "a range straddling the window edge was accepted");

        Require(sceKernelMunmap(again, Bytes) == 0 && sceKernelMunmap(view, Bytes) == 0, "final munmap");
        Require(sceKernelReleaseDirectMemory(physical, Bytes) == 0, "release direct memory");
        std::puts("guest displaced memory tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "guest displaced memory: %s\n", error.what());
        return 1;
    }
#endif
}
