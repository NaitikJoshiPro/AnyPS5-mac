// agc_metal_wave64_lane_local (DECISIONS D62): a wave64 compute program without cross-lane
// dependences runs at one guest lane per host invocation (MslLaneLocal.cpp) instead of two lanes per
// invocation. Each program below runs twice per configuration, translated with and without
// SpirvTarget::mslWave64LaneLocal, over inputs that make the lanes of a wave diverge (also across
// the two halves) and whole waves take one side; every output dword must be bit-identical, and the
// layout must be the expected one (reflection: threads per threadgroup).
//
// Programs (llvm-mc -arch=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize64; user data s[0:3] input V#,
// s[4:7] output V#, s8 threads per group, s9 workgroup id, v0 thread id; 8 output dwords per thread):
// - Branches: if/else over execz skips with a store in the then arm, a loop with a per-lane trip
//   count (execz exit, store inside), an own-bit mask (v_cmp to an SGPR pair, v_cndmask), a nested
//   if, a do-while with an execnz back edge: lane-local.
// - WaveLds: a single-wave LDS exchange without s_barrier (lanes ^ 33, then ^ 1 after a rewrite):
//   lane-local with threadgroup wave LDS barriers.
// - TwoWaveLds: an exchange with the other wave through s_barrier, then an if: lane-local.
// - ExecCount: s_bcnt1 of EXEC inside an if; Vccz: s_cbranch_vccz over code every EXEC lane runs;
//   CrossLane: DPP row shifts, v_permlanex16, ds_swizzle, ds_bpermute, v_readlane 31/63 and
//   v_readfirstlane: lane-local with threadgroup exchanges in one-wave threadgroups, two lanes per
//   invocation otherwise.

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include "Wave64LaneLocalPrograms.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint32_t Results = 8;
constexpr std::uint32_t Groups = 16;

struct Random {
    std::uint64_t state = 0x9e3779b97f4a7c15ull;
    std::uint32_t Next() {
        state ^= state << 13u;
        state ^= state >> 7u;
        state ^= state << 17u;
        return static_cast<std::uint32_t>(state >> 16u);
    }
};

// Group patterns: random values; all even (no lane in the then arm); all odd; odd in guest lanes
// 0-31 and even in 32-63 (the halves diverge); small values with every loop count; large values.
std::vector<std::uint32_t> Inputs(std::uint32_t threads) {
    Random random;
    std::vector<std::uint32_t> values(static_cast<std::size_t>(threads) * Groups);
    for (std::uint32_t group = 0; group < Groups; ++group) {
        for (std::uint32_t thread = 0; thread < threads; ++thread) {
            const std::uint32_t lane = thread % 64u;
            std::uint32_t value = random.Next();
            switch (group % 6u) {
            case 1: value &= ~1u; break;
            case 2: value |= 1u; break;
            case 3: value = lane < 32u ? (value | 1u) : (value & ~1u); break;
            case 4: value = (thread * 7u + group) % 64u; break;
            case 5: value = 1000u + (value % 64u); break;
            default: break;
            }
            values[static_cast<std::size_t>(group) * threads + thread] = value;
        }
    }
    return values;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint64_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(bytes), 0x31016facu};
}

std::vector<std::uint32_t> Run(AgcDriver::VulkanDevice& device, const Wave64LaneLocal::Case& test, bool laneLocal, const std::vector<std::uint32_t>& input) {
    std::vector<std::uint32_t> output(input.size() * Results, 0xdeadbeefu);
    std::vector<std::uint32_t> userData(9, 0u);
    const auto in = BufferDescriptor(input.data(), input.size() * 4u);
    const auto out = BufferDescriptor(output.data(), output.size() * 4u);
    std::copy(in.begin(), in.end(), userData.begin());
    std::copy(out.begin(), out.end(), userData.begin() + 4);
    userData[8] = test.threads;
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(test.code.data()), std::as_bytes(test.code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{test.threads, 1, 1}, test.ldsDwords, {true, false, false}, false, 1};
    auto target = device.Target();
    target.mslWave64LaneLocal = laneLocal;
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(test.code.data()), test.code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    const auto& threads = result.msl.Shader().reflection.threadsPerThreadgroup;
    const auto hostThreads = threads[0] * threads[1] * threads[2];
    const bool expectLocal = laneLocal && test.laneLocal;
    const auto expected = expectLocal ? test.threads : (test.threads + 63u) / 64u * 32u;
    Require(hostThreads == expected, std::string("wave64 lane-local: ") + test.name + (laneLocal ? " (lane-local allowed)" : " (two halves)") + " has " + std::to_string(hostThreads) + " host threads per group, expected " + std::to_string(expected));
    device.Dispatch(result, Groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(test.code.data()));
    device.WaitIdle();
    return output;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (!device->Target().mslWave64LaneLocal) {
            std::puts("skipped: the device target does not lay wave64 compute out at one lane per invocation");
            return VulkanTestSkipped;
        }
        const auto& cases = Wave64LaneLocal::Cases;
        std::size_t compared = 0;
        for (const auto& test : cases) {
            const auto input = Inputs(test.threads);
            const auto halves = Run(*device, test, false, input);
            const auto local = Run(*device, test, true, input);
            std::size_t differences = 0;
            std::size_t written = 0;
            for (std::size_t index = 0; index < halves.size(); ++index) {
                written += halves[index] != 0xdeadbeefu;
                if (halves[index] == local[index]) continue;
                if (differences++ < 8u) std::printf("%s: thread %zu dword %zu: two halves 0x%08x, lane-local 0x%08x (input 0x%08x)\n", test.name, index / Results, index % Results, halves[index], local[index], input[index / Results]);
            }
            Require(written != 0u, std::string("wave64 lane-local: ") + test.name + " wrote nothing");
            Require(differences == 0u, std::string("wave64 lane-local: ") + test.name + ": " + std::to_string(differences) + " output dword(s) differ between the layouts");
            compared += halves.size();
            std::printf("%s: %zu dwords identical (%s)\n", test.name, halves.size(), test.laneLocal ? "lane-local" : "two halves kept");
        }
        std::printf("wave64 lane-local: %zu cases, %zu output dwords bit-identical\n", cases.size(), compared);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
