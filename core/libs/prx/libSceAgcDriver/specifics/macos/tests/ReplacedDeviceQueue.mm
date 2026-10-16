// Driver::AttachWindow replaces the windowless device with a windowed one and keeps the first alive.
// Both run on one MTLDevice, so the host imports and their residency set (GuestBufferMemory.mm, keyed by
// the MTLDevice) outlive the replacement. That set must be attached to the queue the replacement commits
// on: with a queue per device it stayed on the first device's queue, and the replacement's stores through
// imports were lost under load (PPSA99005: every completion marker; MetalDevice.mm
// SharedQueue). Checked directly (whether a non-resident allocation is still mapped depends on memory
// pressure), then by stores through imports made before and after the replacement.
#import <Metal/Metal.h>
#include "../src/MetalDeviceState.h"
#include "../src/MetalNoCopy.h"
#include "prx/libSceAgcDriver/tests/execution/VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static const void* QueueOf(const VulkanDevice& device) { return (__bridge const void*)device.state->queue; }
};

}

namespace {

constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t AllocationBytes = BlockBytes * 2;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("replaced device queue: " + message);
}

class GuestAllocation {
public:
    GuestAllocation() {
        allocation = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, AllocationBytes + 2 * BlockBytes));
        Require(allocation != nullptr, "cannot allocate the guest memory");
        memory = allocation + BlockBytes;
        std::memset(memory, 0, AllocationBytes);
        GuestAllocations::Mutation().Add(memory, AllocationBytes, true, true, true);
    }
    ~GuestAllocation() {
        GuestAllocations::Mutation().Remove(memory);
        std::free(allocation);
    }
    GuestAllocation(const GuestAllocation&) = delete;
    GuestAllocation& operator=(const GuestAllocation&) = delete;
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(memory); }
    std::uint32_t Word(std::size_t offset) const {
        std::uint32_t value = 0;
        std::memcpy(&value, memory + offset, sizeof(value));
        return value;
    }

private:
    std::uint8_t* allocation = nullptr;
    std::uint8_t* memory = nullptr;
};

bool Fill(AgcDriver::VulkanDevice& device, const GuestAllocation& guest, std::uint32_t value) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    const std::array<std::uint32_t, 4> pattern{value, value, value, value};
    if (!device.FillBuffer(guest.Address(), AllocationBytes, pattern)) return false;
    device.WaitIdle();
    return true;
}

void Expect(const GuestAllocation& guest, std::uint32_t value, const char* what) {
    for (const std::size_t at : {std::size_t{0}, AllocationBytes / 2, AllocationBytes - 4}) {
        char text[160];
        std::snprintf(text, sizeof(text), "%s: the CPU sees 0x%08x at +0x%zx, not 0x%08x", what, guest.Word(at), at, value);
        Require(guest.Word(at) == value, text);
    }
}

}

int main() {
    @autoreleasepool {
        try {
            auto first = OpenVulkanTestDevice();
            if (!first) return VulkanTestSkipped;
            GuestAllocation before;
            if (!Fill(*first, before, 0x11111111u)) {
                std::puts("skipped, the device does not import guest memory");
                return VulkanTestSkipped;
            }
            Expect(before, 0x11111111u, "the first device's fill");
            const void* metalDevice = reinterpret_cast<const void*>(first->Device());
            Require(AgcDriver::Metal::HostImportResidencyQueue(metalDevice) == AgcDriver::VulkanDeviceTestAccess::QueueOf(*first), "the host-import residency set is not on the first device's queue");

            auto second = OpenVulkanTestDevice();
            Require(second != nullptr, "the second device did not open");
            Require(reinterpret_cast<const void*>(second->Device()) == metalDevice, "the two devices run on different MTLDevices");
            // The replacement commits on the queue the imports' residency set is attached to.
            Require(AgcDriver::VulkanDeviceTestAccess::QueueOf(*second) == AgcDriver::Metal::HostImportResidencyQueue(metalDevice), "the replacing device commits on a queue the host-import residency set is not attached to");

            Require(Fill(*second, before, 0x22222222u), "the second device did not fill through the first device's import");
            Expect(before, 0x22222222u, "the second device's fill through the first device's import");
            GuestAllocation after;
            Require(Fill(*second, after, 0x33333333u), "the second device did not fill memory mapped after it opened");
            Expect(after, 0x33333333u, "the second device's fill of memory mapped after the replacement");
            Require(AgcDriver::VulkanDeviceTestAccess::QueueOf(*second) == AgcDriver::Metal::HostImportResidencyQueue(metalDevice), "the host-import residency set left the replacing device's queue");
            std::puts("replaced device queue tests passed");
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
