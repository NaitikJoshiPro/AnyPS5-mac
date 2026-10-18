// agc_metal_wave64_lane_local_plan (DECISIONS D62): which wave64 compute programs MetalBackend lays
// out at one guest lane per host invocation (MslLaneLocal.cpp), checked on the CPU without a Metal
// device: the programs of agc_metal_wave64_lane_local get the expected layout (reflection: threads
// per threadgroup), the lane-local MSL has no second-half names, and the flag off keeps two lanes
// per invocation. A rejected program prints the planner's reason (APS5_MSL_LANE_SURVEY).

#include "Recompiler.hpp"
#include "BdaAbi.hpp"
#include "Wave64LaneLocalPrograms.h"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

ShaderRecompiler::SpirvTarget MetalTarget(bool laneLocal) {
    std::vector<std::uint32_t> capabilities;
    for (const auto capability : {spv::CapabilityShader, spv::CapabilityInt64, spv::CapabilityInt64Atomics, spv::CapabilityInt16, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformArithmetic, spv::CapabilitySignedZeroInfNanPreserve, spv::CapabilityImageQuery, spv::CapabilityFloat64}) {
        capabilities.push_back(static_cast<std::uint32_t>(capability));
    }
    std::ranges::sort(capabilities);
    ShaderRecompiler::SpirvTarget target {0u, 0u, 32u, ShaderRecompiler::BdaAbi::Version, capabilities, {}, false, {1024u, 1024u, 1024u}, 1024u, 32768u, {}, {}};
    target.backend = ShaderRecompiler::ShaderBackend::Msl;
    target.mslWave64LaneLocal = laneLocal;
    return target;
}

std::vector<std::uint32_t> Backing(1024u, 0u);

ShaderRecompiler::RecompileResult Recompile(const Wave64LaneLocal::Case& test, bool laneLocal) {
    std::vector<std::uint32_t> userData(9, 0u);
    const auto address = reinterpret_cast<std::uintptr_t>(Backing.data());
    for (std::uint32_t buffer = 0; buffer < 8u; buffer += 4u) {
        const std::array<std::uint32_t, 4> descriptor {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(Backing.size() * 4u), 0x31016facu};
        std::copy(descriptor.begin(), descriptor.end(), userData.begin() + buffer);
    }
    userData[8] = test.threads;
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(test.code.data()), std::as_bytes(test.code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{test.threads, 1, 1}, test.ldsDwords, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(test.code.data()), test.code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        MetalTarget(laneLocal),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

std::string Survey(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

}

int main() {
    try {
        // The planner's reasons go to APS5_MSL_LANE_SURVEY (set by the test registration).
        const char* surveyPath = std::getenv("APS5_MSL_LANE_SURVEY");
        const std::filesystem::path survey = surveyPath != nullptr ? surveyPath : "";
        for (const auto& test : Wave64LaneLocal::Cases) {
            if (!survey.empty()) std::filesystem::remove(survey);
            const auto local = Recompile(test, true);
            const auto reason = survey.empty() ? std::string("(APS5_MSL_LANE_SURVEY unset)") : Survey(survey);
            const auto& threads = local.msl.Reflection().threadsPerThreadgroup;
            const auto host = threads[0] * threads[1] * threads[2];
            const auto expected = test.laneLocal ? test.threads : (test.threads + 63u) / 64u * 32u;
            Require(host == expected, std::string(test.name) + ": " + std::to_string(host) + " host threads per group, expected " + std::to_string(expected) + "; planner: " + reason);
            const bool halves = std::regex_search(local.msl.Shader().source, std::regex("\\bv[0-9]+_h\\b"));
            Require(halves != test.laneLocal, std::string(test.name) + (test.laneLocal ? ": lane-local MSL still has second-half values" : ": two-half MSL has no second-half values"));
            const auto off = Recompile(test, false);
            const auto& offThreads = off.msl.Reflection().threadsPerThreadgroup;
            Require(offThreads[0] * offThreads[1] * offThreads[2] == (test.threads + 63u) / 64u * 32u, std::string(test.name) + ": without the flag the layout is not two lanes per invocation");
            std::printf("%s: %s, MSL %zu bytes (two halves %zu)\n", test.name, test.laneLocal ? "lane-local" : "two halves kept", local.msl.Shader().source.size(), off.msl.Shader().source.size());
        }
        if (!survey.empty()) std::filesystem::remove(survey);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
