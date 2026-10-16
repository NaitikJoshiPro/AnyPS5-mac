// The BDA page table serves a guest allocation longer than 1 GiB piece by piece through host imports (D52). It used to
// wrap such an allocation whole in one no-copy page wrapper beside the imports, outside the import budget; Metal wires
// a no-copy buffer in full when a command buffer uses it, so ASTRO BOT's first address-based dispatch over its 3.1 +
// 4.6 GiB of direct memory failed on the GPU with kIOGPUCommandBufferCallbackErrorOutOfMemory (reported as
// MTL4CommandQueueErrorTimeout). Here a global-store kernel (GlobalStoreAddtid's program) writes 64 KiB blocks on both
// sides of a 1 GiB boundary inside a 4.5 GiB untouched allocation: the stores land in place and wired memory grows by
// the pieces the table imports within the import budget (2 GiB here), not by the whole allocation. (A regression
// wires 4.5 GiB for the test's lifetime.)
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/mman.h>
#endif

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

#if defined(__APPLE__)
constexpr std::uint64_t Piece = std::uint64_t{1} << 30u;
constexpr std::size_t RangeBytes = (std::size_t{9} << 30u) / 2; // 4.5 GiB
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Base = 2048;

// GlobalStoreAddtid.cpp's program: global_store_dword_addtid at s[8:9] + 2048 + {0, -256, -1024, 512 (lanes with
// (lane & 3) != 2), -2048, 2044} of (0x5100 + lane * 0x101) ^ {0, 0x11110000, ..., 0x55550000}.
alignas(256) constexpr std::array<std::uint32_t, 41> AddtidCode{
    0x800aff08, 0x00000800, 0x820b8009, 0x160800ff, 0x00000101, 0x4a0808ff, 0x00005100, 0x36500083,
    0x3a520880, 0xbefc0380, 0xdc5c8000, 0x000a2900, 0x3a5208ff, 0x11110000, 0xbefc03c0, 0xdc5c8f00,
    0x000a2900, 0x3a5208ff, 0x22220000, 0xbefc03ff, 0x0000ffff, 0xdc5c8c00, 0x000a2900, 0x3a5208ff,
    0x33330000, 0xbefc0380, 0x7daa5082, 0xdc5c8200, 0x000a2900, 0xbefe04c1, 0x3a5208ff, 0x44440000,
    0xbefc0380, 0xdc5c8800, 0x000a2900, 0x3a5208ff, 0x55550000, 0xbefc0380, 0xdc5c87fc, 0x000a2900,
    0xbf810000,
};

struct Store {
    std::int32_t offset;
    std::uint32_t pattern;
    bool masked;
};

constexpr std::array<Store, 6> Stores{{
    {0, 0x00000000u, false},
    {-256, 0x11110000u, false},
    {-1024, 0x22220000u, false},
    {512, 0x33330000u, true},
    {-2048, 0x44440000u, false},
    {2044, 0x55550000u, false},
}};

double WiredGiB() {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count);
    return static_cast<double>(vm.wire_count) * static_cast<double>(vm_page_size) / static_cast<double>(1ull << 30u);
}

void Run(AgcDriver::VulkanDevice& device, std::uint8_t* block, const std::string& name) {
    constexpr std::uint32_t Lanes = 32;
    std::memset(block, 0x5a, BlockBytes);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block));
    std::vector<std::uint32_t> userData(10, 0u);
    userData[8] = static_cast<std::uint32_t>(address);
    userData[9] = static_cast<std::uint32_t>(address >> 32u);
    const std::span<const std::uint32_t> code(AddtidCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> regions{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {Lanes, 0, userData, compute, std::nullopt, std::nullopt, regions},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    const auto* words = reinterpret_cast<const std::uint32_t*>(block);
    std::vector<std::uint32_t> expected(BlockBytes / 4u, 0x5a5a5a5au);
    for (const auto& store : Stores) {
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            if (store.masked && (lane & 3u) == 2u) continue;
            expected[(Base + store.offset + lane * 4u) / 4u] = (0x5100u + lane * 0x101u) ^ store.pattern;
        }
    }
    for (std::size_t dword = 0; dword < expected.size(); ++dword) {
        Require(words[dword] == expected[dword], "BDA long allocation: " + name + " byte " + std::to_string(dword * 4u) + " is " + std::to_string(words[dword]) + ", expected " + std::to_string(expected[dword]));
    }
}
#endif

}

int main() {
#if !defined(__APPLE__)
    std::puts("skipped, the 1 GiB import pieces of the BDA page table are the Metal driver's");
    return VulkanTestSkipped;
#else
    // A 2 GiB live-import budget: the table imports the allocation's first pieces (the two around the boundary) and
    // leaves the rest out, so the wired growth bounds what the GPU was given.
    setenv("APS5_HOST_IMPORT_MIB", "2048", 1);
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (device->Target().subgroupSize < 32u) {
            std::printf("skipped, the device's subgroups are narrower than a wave (%u lanes)\n", device->Target().subgroupSize);
            return VulkanTestSkipped;
        }
        auto* memory = static_cast<std::uint8_t*>(mmap(nullptr, RangeBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
        Require(memory != MAP_FAILED, "BDA long allocation: cannot reserve 4.5 GiB");
        const auto base = reinterpret_cast<std::uint64_t>(memory);
        GuestAllocations::Mutation().Add(memory, RangeBytes, true, true, true);
        const auto wiredBefore = WiredGiB();
        const auto boundary = (base / Piece + 1) * Piece;
        Require(boundary > base + BlockBytes * 4 && boundary + BlockBytes * 4 < base + RangeBytes, "BDA long allocation: the range has no 1 GiB boundary inside");
        Run(*device, reinterpret_cast<std::uint8_t*>(boundary - BlockBytes * 2), "below the boundary");
        Run(*device, reinterpret_cast<std::uint8_t*>(boundary + BlockBytes), "above the boundary");
        const auto wiredGrowth = WiredGiB() - wiredBefore;
        std::printf("BDA long allocation: wired memory grew %.2f GiB over a %.1f GiB allocation\n", wiredGrowth, RangeBytes / static_cast<double>(1ull << 30u));
        Require(wiredGrowth < 3.0, "BDA long allocation: the dispatches wired the whole 4.5 GiB allocation (one page wrapper instead of import pieces?)");
        GuestAllocations::Mutation().Remove(memory);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            device->WaitIdle();
        }
        std::puts("BDA long allocation passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
