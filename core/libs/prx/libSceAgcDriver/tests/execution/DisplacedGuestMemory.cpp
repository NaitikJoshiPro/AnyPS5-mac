// Guest memory in the displaced window (D31, D33: the macOS arm64 GPU carveout at 64-448 GiB and the dyld shared
// region's unused tail, backed at guest | DisplacedFlip) reaches the GPU, at 0x4000000000 and at 0x400000000: the driver wraps the alias, a GPU fill lands in the pages the CPU sees at both the
// guest address (through the fault handler) and the alias, and the driver's own guest reads and writes translate.
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint64_t CarveoutBase = 0x4000000000ull;
constexpr std::uint64_t TailBase = 0x400000000ull;
constexpr std::size_t Bytes = 0x40000;
constexpr std::uint32_t ReadWrite = 0x04; // PAGE_READWRITE, as GuestArenaCommit takes it

std::uint32_t HostWord(std::uint64_t guest, std::size_t index) {
    return reinterpret_cast<const volatile std::uint32_t*>(GuestArena::HostAddress(guest))[index];
}

std::uint64_t Faults() {
    std::uint64_t faults = 0, patches = 0;
    GuestArena::GuestDisplacedStats_nid_no_patch(&faults, &patches);
    return faults;
}

#if defined(__APPLE__)
int HostProtection(std::uint64_t address) {
    mach_vm_address_t region = address;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || region > address) return -1;
    return info.protection;
}
#endif

std::vector<std::pair<std::uintptr_t, std::uintptr_t>> CollectWritten(std::uint64_t guest, std::size_t bytes) {
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
    GuestWriteWatch::GuestWriteWatchCollect_nid_postfix(guest, bytes, [](void* context, std::uintptr_t begin, std::uintptr_t end) {
        static_cast<std::vector<std::pair<std::uintptr_t, std::uintptr_t>>*>(context)->emplace_back(begin, end);
    }, &runs);
    return runs;
}

std::uint32_t GuestWord(std::uint64_t guest, std::size_t index) {
    return reinterpret_cast<const volatile std::uint32_t*>(guest)[index];
}

}

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("skipped, no displaced window on this host");
    return VulkanTestSkipped;
#else
    try {
        Require(GuestArena::GuestDisplacedActive_nid_no_patch(), "displaced guest memory: the window is not active");
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Require(GuestArena::IsDisplaced(TailBase) && GuestArena::IsDisplaced(CarveoutBase), "displaced guest memory: 0x400000000 or 0x4000000000 is not displaced");
        for (const std::uint64_t Base : {CarveoutBase, TailBase}) {
            auto* guest = reinterpret_cast<void*>(Base);
            GuestArena::GuestArenaCommit_nid_postfix(guest, Bytes, ReadWrite, 0x4000);
            GuestAllocations::Mutation().Add(guest, Bytes, true, true, true);

            // The write watch keys pages by guest address and arms them at the alias. Wrapping the range for the GPU
            // (MetalNoCopy.h) brackets the guest range as a host write, so the armed alias pages are left writable
            // under the GPU mapping (the driver consumes the written marks itself). Looked up by the alias, the watch
            // would miss them and the wrap would re-protect them read-only behind the GPU mapping.
            const bool watched = GuestWriteWatch::GuestWriteWatchAvailable_nid_postfix();
            if (watched) {
                GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(guest, Bytes);
                static_cast<void>(CollectWritten(Base, Bytes)); // arms every page
                Require(HostProtection(GuestArena::HostAddress(Base)) == VM_PROT_READ, "displaced guest memory: the write watch did not arm the alias");
            }

            // A GPU fill through the import of the alias.
            {
                std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
                const std::array<std::uint32_t, 4> pattern{0x11u, 0x22u, 0x33u, 0x44u};
                Require(device->FillBuffer(Base, Bytes / 2, pattern), "displaced guest memory: the fill was not imported");
                device->WaitIdle();
            }
            for (std::size_t index : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, Bytes / 8 - 1}) {
                const std::uint32_t expected = 0x11u * static_cast<std::uint32_t>(index % 4 + 1);
                Require(HostWord(Base, index) == expected, "displaced guest memory: the GPU fill is not in the alias");
                Require(GuestWord(Base, index) == expected, "displaced guest memory: the GPU fill is not visible at the guest address");
            }
            Require(HostWord(Base, Bytes / 8) == 0, "displaced guest memory: the fill ran past its range");
            if (watched) {
                Require((HostProtection(GuestArena::HostAddress(Base)) & VM_PROT_WRITE) != 0, "displaced guest memory: the wrapped alias pages were left armed read-only");
                GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(guest, Bytes);
            }

            // DCC key scans read guest metadata in bulk at the alias, not through the fault handler.
            {
                constexpr std::uint64_t Surface = 0x100000;
                const auto Keys = AgcDriver::Graphics::DccKeyBytes(Surface);
                Require(Keys != 0 && Keys <= Bytes / 4, "displaced guest memory: unexpected DCC key size");
                std::memset(reinterpret_cast<void*>(GuestArena::HostAddress(Base + Bytes - Keys)), 0xff, Keys);
                const auto faults = Faults();
                Require(AgcDriver::Graphics::ReadDccKeys(Base + Bytes - Keys, Surface) == AgcDriver::Graphics::DccKeys::Uncompressed, "displaced guest memory: DCC keys misread");
                Require(Faults() == faults, "displaced guest memory: the DCC key scan faulted on the guest address");
                std::memset(reinterpret_cast<void*>(GuestArena::HostAddress(Base + Bytes - Keys)), 0, Keys);
            }

            // The CPU writes through the guest address; the driver reads it back and writes it.
            reinterpret_cast<volatile std::uint32_t*>(Base + Bytes / 2)[0] = 0xcafef00du;
            std::array<std::byte, 4> word{};
            AgcDriver::GuestMemory::Read(Base + Bytes / 2, word);
            std::uint32_t read = 0;
            std::memcpy(&read, word.data(), sizeof(read));
            Require(read == 0xcafef00du, "displaced guest memory: the driver read missed the alias");
            const std::array<std::byte, 4> written{std::byte{0x78}, std::byte{0x56}, std::byte{0x34}, std::byte{0x12}};
            AgcDriver::GuestMemory::Write(Base + Bytes / 2 + 4, written);
            Require(HostWord(Base + Bytes / 2, 1) == 0x12345678u, "displaced guest memory: the driver write missed the alias");

            // A second fill over what the CPU wrote: the GPU sees the same pages again.
            {
                std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
                const std::array<std::uint32_t, 4> pattern{7u, 7u, 7u, 7u};
                Require(device->FillBuffer(Base + Bytes / 2, Bytes / 2, pattern), "displaced guest memory: the second fill was not imported");
                device->WaitIdle();
            }
            Require(GuestWord(Base + Bytes / 2, 0) == 7u && HostWord(Base, Bytes / 4 - 1) == 7u, "displaced guest memory: the second fill is not visible");

            // Scan-out decode reads the alias, not the guest address: no fault per load (STK presents from Base).
            {
                AgcDriver::DisplayBuffer display{Base, 0x8000000000000000ull, 64, 16, 1};
                const auto size = AgcDriver::DisplayBufferSize(display);
                Require(size <= Bytes, "displaced guest memory: the display buffer does not fit");
                for (std::size_t index = 0; index < size / 4; ++index)
                    reinterpret_cast<volatile std::uint32_t*>(GuestArena::HostAddress(Base))[index] = 0xff000000u | static_cast<std::uint32_t>(index * 2654435761u >> 8);
                std::uint64_t faultsBefore = 0, patches = 0, faultsAfter = 0;
                GuestArena::GuestDisplacedStats_nid_no_patch(&faultsBefore, &patches);
                const auto pixels = AgcDriver::ReadDisplayBuffer(display);
                GuestArena::GuestDisplacedStats_nid_no_patch(&faultsAfter, &patches);
                const auto expected = AgcDriver::DecodeDisplayBuffer(display, {reinterpret_cast<const std::byte*>(GuestArena::HostAddress(Base)), size});
                Require(pixels == expected, "displaced guest memory: the display buffer decode differs from the alias");
                Require(faultsAfter == faultsBefore, "displaced guest memory: the display buffer decode faulted on the guest address");
            }

            // WAIT_REG_MEM label polls read the alias: a label in the window (STK's fences at 0x400000000) does not
            // fault on every poll, 32- and 64-bit alike.
            {
                const std::uint64_t label = Base + 0x40;
                reinterpret_cast<volatile std::uint64_t*>(GuestArena::HostAddress(label))[0] = 0xbeefcafe00000007ull;
                const auto wait32 = std::array<std::uint32_t, 7>{0xc0053c00u, 0x13u, static_cast<std::uint32_t>(label), static_cast<std::uint32_t>(label >> 32u), 7u, 0xffffffffu, 4u};
                const auto miss32 = std::array<std::uint32_t, 7>{0xc0053c00u, 0x13u, static_cast<std::uint32_t>(label), static_cast<std::uint32_t>(label >> 32u), 8u, 0xffffffffu, 4u};
                const auto wait64 = std::array<std::uint32_t, 9>{0xc0079300u, 0x13u, static_cast<std::uint32_t>(label), static_cast<std::uint32_t>(label >> 32u), 7u, 0xbeefcafeu, 0xffffffffu, 0xffffffffu, 4u};
                const auto faults = Faults();
                for (int poll = 0; poll < 1000; ++poll) {
                    Require(AgcDriver::Pm4::WaitSatisfiedUnchecked(wait32) && !AgcDriver::Pm4::WaitSatisfiedUnchecked(miss32), "displaced guest memory: a 32-bit label poll misread the label");
                    Require(AgcDriver::Pm4::WaitSatisfiedUnchecked(wait64), "displaced guest memory: a 64-bit label poll misread the label");
                }
                Require(Faults() == faults, "displaced guest memory: label polls faulted on the guest address (" + std::to_string(Faults() - faults) + " faults)");
            }

            GuestAllocations::Mutation().Remove(guest);
            GuestArena::GuestArenaReset_nid_postfix(guest, Bytes);
            GuestArena::GuestArenaRelease_nid_postfix(guest, Bytes);
            std::printf("displaced guest memory at 0x%llx passed\n", static_cast<unsigned long long>(Base));
        }

        // A commit across the shared cache's end would be half host memory, half alias: refused with a clear message.
        bool refused = false;
        try {
            GuestArena::GuestArenaCommit_nid_postfix(reinterpret_cast<void*>(GuestArena::DisplacedStart() - 0x4000), 0x8000, ReadWrite, 0x4000);
        } catch (const std::runtime_error& error) {
            refused = std::string(error.what()).find("straddles") != std::string::npos;
        }
        Require(refused, "displaced guest memory: a commit straddling the window's lower edge was accepted");
        std::puts("displaced guest memory tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
