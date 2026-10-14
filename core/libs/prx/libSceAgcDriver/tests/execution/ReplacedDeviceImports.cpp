// A device that replaces another (Driver::AttachWindow keeps the windowless device alive and makes a
// windowed one) writes through host imports the first device made, and through new imports that
// join the residency set the first device's queue carries. On Metal both devices share the
// MTLDevice, so the imports and their residency set outlive the first device's work; with a command
// queue per device the second device's command buffers ran without that set, and every GPU store
// through such an import was lost (PPSA99005's completion markers never reached the CPU). Upstream's
// second VkDevice re-imports instead; either way the CPU must see the second device's stores.
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t AllocationBytes = BlockBytes * 4;
constexpr std::size_t ReservationBytes = AllocationBytes + BlockBytes * 2;

class GuestAllocation {
public:
    GuestAllocation() {
#ifdef _WIN32
        allocation = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, ReservationBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        allocation = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, ReservationBytes));
#endif
        Require(allocation != nullptr, "replaced device imports: cannot allocate the guest memory");
        memory = allocation + BlockBytes;
        std::memset(memory, 0, AllocationBytes);
        GuestAllocations::Mutation().Add(memory, AllocationBytes, true, true, true);
    }

    ~GuestAllocation() {
        GuestAllocations::Mutation().Remove(memory);
#ifdef _WIN32
        VirtualFree(allocation, 0, MEM_RELEASE);
#else
        std::free(allocation);
#endif
    }

    GuestAllocation(const GuestAllocation&) = delete;
    GuestAllocation& operator=(const GuestAllocation&) = delete;

    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(memory)); }

    std::uint32_t Word(std::size_t offset) const {
        std::uint32_t value = 0;
        std::memcpy(&value, memory + offset, sizeof(value));
        return value;
    }

private:
    std::uint8_t* allocation = nullptr;
    std::uint8_t* memory = nullptr;
};

// Fills [offset, offset + BlockBytes) with `value` on `device` and waits; false when the device
// does not import guest memory.
bool Fill(AgcDriver::VulkanDevice& device, const GuestAllocation& guest, std::size_t offset, std::uint32_t value) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    const std::array<std::uint32_t, 4> pattern{value, value, value, value};
    if (!device.FillBuffer(guest.Address() + offset, BlockBytes, pattern)) return false;
    device.WaitIdle();
    return true;
}

void Expect(const GuestAllocation& guest, std::size_t offset, std::uint32_t value, const char* what) {
    for (const auto at : {offset, offset + BlockBytes / 2, offset + BlockBytes - 4}) {
        if (guest.Word(at) == value) continue;
        char text[192];
        std::snprintf(text, sizeof(text), "replaced device imports: %s: the CPU sees 0x%08x at +0x%zx, not 0x%08x", what, guest.Word(at), at, value);
        throw std::runtime_error(text);
    }
}

}

int main() {
    try {
        auto first = OpenVulkanTestDevice();
        if (!first) return VulkanTestSkipped;
        GuestAllocation guest;
        if (!Fill(*first, guest, 0, 0x11111111u)) {
            std::printf("skipped, the device does not import guest memory\n");
            return VulkanTestSkipped;
        }
        Expect(guest, 0, 0x11111111u, "the first device's fill");

        // The replacement: the first device stays alive (Driver::replacedDevices) and idle.
        auto second = OpenVulkanTestDevice();
        Require(second != nullptr, "replaced device imports: the second device did not open");
        Require(Fill(*second, guest, 0, 0x22222222u), "replaced device imports: the second device did not fill through the import");
        Expect(guest, 0, 0x22222222u, "the second device's fill over the first device's import");
        Require(Fill(*second, guest, BlockBytes * 2, 0x33333333u), "replaced device imports: the second device's fill of the other blocks");
        Expect(guest, BlockBytes * 2, 0x33333333u, "the second device's fill of blocks the first device never wrote");
        // Memory the guest maps after the replacement: its import joins the residency set made for the
        // first device, whose queue commits nothing more (a set's new allocations only take effect for
        // a queue that commits after they were added).
        GuestAllocation later;
        Require(Fill(*second, later, 0, 0x66666666u), "replaced device imports: the second device did not fill memory mapped after it opened");
        Expect(later, 0, 0x66666666u, "the second device's fill of memory mapped after the replacement");
        // The first device's stores still land too.
        Require(Fill(*first, guest, BlockBytes, 0x44444444u), "replaced device imports: the first device's later fill");
        Expect(guest, BlockBytes, 0x44444444u, "the first device's fill after the second device's");

        // The second device outlives the first (the replaced device goes at shutdown in the driver;
        // here before, which must not take the imports' residency from the second device).
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            first->WaitIdle();
        }
        first.reset();
        Require(Fill(*second, guest, BlockBytes * 3, 0x55555555u), "replaced device imports: the second device's fill after the first device went");
        Expect(guest, BlockBytes * 3, 0x55555555u, "the second device's fill after the first device went");
        std::puts("replaced device imports tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
