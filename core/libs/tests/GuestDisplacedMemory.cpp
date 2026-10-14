// Fixed guest mappings in host-reserved ranges on macOS arm64 (D31, D33): the dyld shared region's unused tail, the
// commpage guard and the GPU carveout (64-448 GiB, a permanent guard in every process) are backed at
// guest | DisplacedFlip, and native accesses to the guest address are redirected by the fault handler. Translated
// (FEX) accesses are covered by the `displaced` fixture.
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

constexpr std::uintptr_t Base = 0x4000000000ull;     // GPU carveout (oops-sdk AGC_VM_BASE)
constexpr std::uintptr_t TailBase = 0x400000000ull;  // shared region tail (oops-mesa's GPU VA high range)
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
        Require(GuestArena::HostAddress(0x100000000ull) == 0x100000000ull, "an address below the window was translated");
        Require(GuestArena::HostAddress(0x8000000000ull) == 0x8000000000ull, "an address above the window was translated");

        // The range table: shared region tail, commpage guard, GPU carveout, contiguous, one flip.
        const GuestArena::DisplacedRange* ranges = nullptr;
        std::size_t rangeCount = 0;
        GuestArena::GuestDisplacedRanges_nid_no_patch(&ranges, &rangeCount);
        for (std::size_t i = 0; i < rangeCount; ++i)
            std::printf("displaced range %s 0x%llx-0x%llx at 0x%llx\n", ranges[i].name, static_cast<unsigned long long>(ranges[i].guestStart),
                static_cast<unsigned long long>(ranges[i].guestEnd), static_cast<unsigned long long>(ranges[i].hostBase));
        Require(rangeCount == 3, "expected three displaced ranges (shared region tail, commpage guard, GPU carveout)");
        Require(ranges[0].guestStart == GuestArena::DisplacedStart() && ranges[2].guestEnd == GuestArena::DisplacedEnd, "the table does not span the window");
        for (std::size_t i = 0; i < rangeCount; ++i) {
            Require(ranges[i].hostBase == (ranges[i].guestStart | GuestArena::DisplacedFlip), "a range's alias is not its guest address with the flip");
            Require(i == 0 || ranges[i].guestStart == ranges[i - 1].guestEnd, "the displaced ranges are not contiguous");
        }
        Require(ranges[1].guestStart == 0xFC0000000ull && ranges[2].guestStart == 0x1000000000ull, "unexpected commpage guard or carveout bounds");
        Require(GuestArena::DisplacedStart() <= TailBase && GuestArena::DisplacedStart() > 0x180000000ull, "0x400000000 is not in the shared region tail");
        Require(GuestArena::HostAddress(TailBase) == (TailBase | GuestArena::DisplacedFlip), "the shared region tail is not translated");
        Require(GuestArena::HostAddress(GuestArena::DisplacedStart() - 0x4000) == GuestArena::DisplacedStart() - 0x4000, "the shared cache was translated");

        // First-fit placement never lands in a displaced range.
        for (int i = 0; i < 8; ++i) {
            void* placed = GuestArena::GuestArenaAllocate_nid_postfix(0x10000000, 0x200000);
            const auto at = reinterpret_cast<std::uintptr_t>(placed);
            Require(at >= GuestArena::DisplacedEnd || at + 0x10000000 <= GuestArena::DisplacedLowest, "first fit placed an allocation in a displaced range");
            GuestArena::GuestArenaRelease_nid_postfix(placed, 0x10000000);
        }

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

        // A fixed mapping in the shared region tail at 0x400000000 (SuperTuxKart's first GEM_VA).
        void* tail = reinterpret_cast<void*>(TailBase);
        Require(sceKernelMapDirectMemory(&tail, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && tail == reinterpret_cast<void*>(TailBase),
            "the fixed mapping at 0x400000000 failed");
        const auto tailFaults = Faults();
        Require(Load<std::uint32_t>(TailBase + 0x104) == 0xa5a5a5a5u, "a native load in the shared region tail missed the alias");
        Store<std::uint64_t>(TailBase + 0x2000, 0x0102030405060708ull);
        Require(Load<std::uint64_t>(GuestArena::HostAddress(TailBase) + 0x2000) == 0x0102030405060708ull, "a native store in the shared region tail missed the alias");
        Require(Load<std::uint64_t>(Base + 0x2000) == 0x0102030405060708ull, "the tail and carveout views do not share the pages");
        Require(Faults() >= tailFaults + 3, "the tail accesses did not go through the fault handler");
        Require(sceKernelMprotect(tail, 0x4000, CpuRead) == 0 && HostProtection(GuestArena::HostAddress(TailBase)) == VM_PROT_READ, "mprotect in the tail");
        Require(sceKernelMunmap(tail, Bytes) == 0 && HostProtection(GuestArena::HostAddress(TailBase)) == VM_PROT_NONE, "munmap in the tail");

        const auto refusedWith = [](std::uintptr_t at, std::size_t bytes, const char* expected) {
            try {
                GuestArena::GuestArenaMarkUsed_nid_postfix(reinterpret_cast<void*>(at), bytes);
            } catch (const std::runtime_error& error) {
                std::printf("refused 0x%llx+0x%zx: %s\n", static_cast<unsigned long long>(at), bytes, error.what());
                return std::string(error.what()).find(expected) != std::string::npos;
            }
            GuestArena::GuestArenaRelease_nid_postfix(reinterpret_cast<void*>(at), bytes);
            return false;
        };
        // Across the shared cache's end: half host memory, half alias.
        Require(refusedWith(GuestArena::DisplacedStart() - 0x4000, 0x8000, "straddles"), "a range straddling the shared cache's end was accepted");
        // The PS5 system-reserved range (0x7FFFFC000-0x1000000000) covers the tail/commpage and commpage/carveout
        // boundaries: refused there as on every host, with its own message.
        Require(refusedWith(0xFC0000000ull - 0x4000, 0x8000, "system-reserved"), "a range across the tail/commpage boundary was accepted");
        Require(refusedWith(0x1000000000ull - 0x4000, 0x8000, "system-reserved"), "a range across the commpage/carveout boundary was accepted");

        // A range across the window's edge has no single backing.
        bool refused = false;
        try {
            GuestArena::GuestArenaMarkUsed_nid_postfix(reinterpret_cast<void*>(GuestArena::DisplacedEnd - 0x4000), 0x8000);
        } catch (const std::runtime_error& error) {
            refused = std::string(error.what()).find("straddles") != std::string::npos;
        }
        Require(refused, "a range straddling the window edge was accepted");

        // D37: a fixed request inside the dyld shared cache (readable host memory, neither mappable nor displaceable)
        // is served at request + 0xD000000000 and the address comes back through the in/out argument.
        constexpr std::uintptr_t CacheRequest = 0x300000000ull; // ASTRO BOT's first mapping
        constexpr std::uintptr_t Relocated = CacheRequest + 0xD000000000ull;
        Require(GuestArena::DisplacedStart() > CacheRequest + Bytes, "0x300000000 is not inside the shared cache on this host");
        void* cached = reinterpret_cast<void*>(CacheRequest);
        Require(sceKernelMapDirectMemory(&cached, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0, "a fixed mapping in the shared cache failed");
        Require(cached == reinterpret_cast<void*>(Relocated), "a fixed mapping in the shared cache was not relocated by the constant offset");
        Require(Load<std::uint32_t>(Relocated + 0x104) == 0xa5a5a5a5u, "the relocated mapping does not show the direct memory");
        Store<std::uint64_t>(Relocated + 0x3000, 0x5566778899aabbccull);
        Require(Load<std::uint64_t>(viewAddress + 0x3000) == 0x5566778899aabbccull, "the relocated mapping does not share the pages");
        // Deterministic: the same request again meets the first, as on the console (no-overwrite fails, replace works).
        void* second = reinterpret_cast<void*>(CacheRequest);
        Require(sceKernelMapDirectMemory(&second, Bytes, CpuGpuReadWrite, MapFixed | 0x80, physical, 0x200000) != 0, "a no-overwrite request over a relocated mapping succeeded");
        second = reinterpret_cast<void*>(CacheRequest + 0x200000);
        Require(sceKernelMapDirectMemory(&second, 0x200000, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && second == reinterpret_cast<void*>(Relocated + 0x200000),
            "a fixed request overlapping a relocated mapping did not land at the same offset");
        Require(Load<std::uint32_t>(Relocated + 0x200000 + 0x104) == 0xa5a5a5a5u, "the overlapping relocated mapping does not show the direct memory");
        Require(sceKernelMunmap(cached, Bytes) == 0, "munmap of the relocated mapping");
        Require(HostProtection(Relocated) == VM_PROT_NONE, "the unmapped relocated range is still accessible");
        // A request that runs from the cache into the tail (ASTRO's 0x300000000+0x79800000 does) is relocated whole.
        void* across = reinterpret_cast<void*>((GuestArena::DisplacedStart() - 1) & ~std::uintptr_t{0x1fffff});
        Require(sceKernelMapDirectMemory(&across, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 &&
                    across == reinterpret_cast<void*>(((GuestArena::DisplacedStart() - 1) & ~std::uintptr_t{0x1fffff}) + 0xD000000000ull),
            "a fixed request across the shared cache's end was not relocated");
        Require(sceKernelMunmap(across, Bytes) == 0, "munmap of the relocated straddling mapping");

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
