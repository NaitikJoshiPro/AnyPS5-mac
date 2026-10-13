// Guest memory in the displaced window (D31: the macOS arm64 GPU carveout, 64-448 GiB, backed at guest |
// DisplacedFlip) reaches the GPU: the driver wraps the alias, a GPU fill lands in the pages the CPU sees at both the
// guest address (through the fault handler) and the alias, and the driver's own guest reads and writes translate.
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint64_t Base = 0x4000000000ull;
constexpr std::size_t Bytes = 0x40000;
constexpr std::uint32_t ReadWrite = 0x04; // PAGE_READWRITE, as GuestArenaCommit takes it

std::uint32_t HostWord(std::uint64_t guest, std::size_t index) {
    return reinterpret_cast<const volatile std::uint32_t*>(GuestArena::HostAddress(guest))[index];
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
        auto* guest = reinterpret_cast<void*>(Base);
        GuestArena::GuestArenaCommit_nid_postfix(guest, Bytes, ReadWrite, 0x4000);
        GuestAllocations::Mutation().Add(guest, Bytes, true, true, true);

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

        GuestAllocations::Mutation().Remove(guest);
        GuestArena::GuestArenaReset_nid_postfix(guest, Bytes);
        GuestArena::GuestArenaRelease_nid_postfix(guest, Bytes);
        std::puts("displaced guest memory tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
