// A registered GPU range longer than 1 GiB is imported in aligned 1 GiB pieces (D42). Metal wires a no-copy buffer in
// full at its first GPU use, so one import of ASTRO BOT's 7.8 GiB direct-memory run would wire 7.8 GiB on a 16 GB Mac
// (or be refused whole by the import budget). The range here is 4.5 GiB of untouched anonymous memory: GPU fills on
// both sides of a 1 GiB boundary land in place (zero-copy, seen by the CPU), no import covers the range or crosses
// the boundary, and wired memory grows by the two pieces used, not the whole range. (A regression wires 4.5 GiB for
// the test's lifetime.)
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/mman.h>
#endif

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint64_t Piece = std::uint64_t{1} << 30u;
constexpr std::size_t RangeBytes = (std::size_t{9} << 30u) / 2; // 4.5 GiB
constexpr std::size_t FillBytes = 65536;

bool Imported(VkDevice device, std::uint64_t address, std::size_t bytes) {
    AgcDriver::Graphics::Context probe{};
    probe.device = device;
    probe.hostImportAlignment = 1;
    return AgcDriver::Graphics::HostImportCovers(probe, address, bytes);
}

#if defined(__APPLE__)
double WiredGiB() {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count);
    return static_cast<double>(vm.wire_count) * static_cast<double>(vm_page_size) / static_cast<double>(1ull << 30u);
}
#endif

}

int main() {
#if !defined(__APPLE__)
    std::puts("skipped, the 1 GiB import pieces are the Metal driver's");
    return VulkanTestSkipped;
#else
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        auto* memory = static_cast<std::uint8_t*>(mmap(nullptr, RangeBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
        Require(memory != MAP_FAILED, "host import pieces: cannot reserve 4.5 GiB");
        const auto base = reinterpret_cast<std::uint64_t>(memory);
        GuestAllocations::Mutation().Add(memory, RangeBytes, true, true, true);
        const auto wiredBefore = WiredGiB();
        // One address before the first 1 GiB boundary inside the range and one after it.
        const auto boundary = (base / Piece + 1) * Piece;
        Require(boundary > base && boundary < base + RangeBytes, "host import pieces: the range has no 1 GiB boundary");
        const std::array<std::uint64_t, 2> targets{(boundary - FillBytes * 4) / FillBytes * FillBytes, boundary + FillBytes * 2};
        std::uint32_t seed = 0x51u;
        for (const auto target : targets) {
            const std::array<std::uint32_t, 4> pattern{seed, seed + 1, seed + 2, seed + 3};
            bool filled = false;
            {
                std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
                filled = device->FillBuffer(target, FillBytes, pattern);
                device->WaitIdle();
            }
            if (!filled) {
                std::printf("skipped, the device does not import guest memory\n");
                return VulkanTestSkipped;
            }
            const auto* words = reinterpret_cast<const std::uint32_t*>(target);
            Require(words[0] == seed && words[3] == seed + 3 && words[FillBytes / 4 - 1] == seed + 3, "host import pieces: a GPU fill is not visible to the CPU");
            Require(Imported(device->Device(), target, FillBytes), "host import pieces: the filled range is not imported in place");
            seed += 0x10u;
        }
        Require(!Imported(device->Device(), base, RangeBytes), "host import pieces: one import covers the whole 4.5 GiB range");
        Require(!Imported(device->Device(), boundary - FillBytes, FillBytes * 2), "host import pieces: an import crosses a 1 GiB boundary");
        const auto wiredGrowth = WiredGiB() - wiredBefore;
        std::printf("host import pieces: wired memory grew %.2f GiB\n", wiredGrowth);
        Require(wiredGrowth < 3.0, "host import pieces: the fills wired more than the two 1 GiB pieces they used (one import of the whole range?)");
        GuestAllocations::Mutation().Remove(memory);
        {
            // Retire the imports while the memory is still mapped.
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            device->WaitIdle();
        }
        std::puts("host import pieces passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
#endif
}
