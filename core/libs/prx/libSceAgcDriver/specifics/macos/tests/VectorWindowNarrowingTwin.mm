// Metal twin of tests/execution/VectorWindowNarrowing.cpp (agc_driver_vector_window_narrowing; upstream 30fab0777e,
// the vector window keeps its unindexed slots out of the array), swapped in by twins/UPSYNC3-tests.cmake. The upstream
// file is compiled here with its Run and main renamed, so its program, inputs and expectations (Check) keep tracking
// upstream; main below repeats upstream's (three data indices, wave32 and wave64) over the Run of this file. Upstream's
// Run checks the split by counting Function-storage OpVariables in result.spirv, which is empty for an MSL target; this
// Run checks the MSL source instead: the window narrowing of MslAluMath.cpp (AccessVectorWindow) declares one
// `uint aps5_vector_window[_h]_s<slot>` local per slot outside the indexed array, and the program's window has more
// than 8 of them. The dispatch and the GPU check of every lane are upstream's.

// The upstream file's headers first, so the renames below reach only the upstream file's own definitions.
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "../../../tests/execution/VulkanTestDevice.hpp"
#include <spirv/unified1/spirv.hpp>
#include <string_view>

#define Run AgcDriverVectorWindowNarrowingVulkanRun
#define main AgcDriverVectorWindowNarrowingVulkanMain
#include "../../../tests/execution/VectorWindowNarrowing.cpp"
#undef main
#undef Run

namespace {

std::uint32_t MslWindowSlotLocals(const ShaderRecompiler::RecompileResult& result) {
    Require(!result.msl.empty(), "vector window narrowing: the compile has no MSL source");
    const std::string& source = result.msl.Source();
    std::uint32_t count = 0;
    for (std::size_t at = source.find("uint aps5_vector_window"); at != std::string::npos; at = source.find("uint aps5_vector_window", at + 1u)) {
        const auto end = source.find_first_of(" =;[", at + 5u);
        const std::string_view name(source.data() + at + 5u, end - at - 5u);
        if (name.find("_s") != std::string_view::npos) ++count;
    }
    return count;
}

void Run(AgcDriver::VulkanDevice& device, std::uint32_t waveSize, std::uint32_t threads, std::uint32_t bounded, std::uint32_t data) {
    Output.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(10, 0u);
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    userData[8] = bounded;
    userData[9] = data;
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    Require(MslWindowSlotLocals(result) > 8u, "vector window narrowing: wave" + std::to_string(waveSize) + " window was not split into separate slot variables");
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

}

int main() {
    try {
        // Upstream's SPIR-V Run and main have no input on Metal; they stay compiled.
        (void)&AgcDriverVectorWindowNarrowingVulkanRun;
        (void)&AgcDriverVectorWindowNarrowingVulkanMain;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (device->Target().subgroupSize < 32u) {
            std::printf("skipped, subgroup size %u cannot hold a wave32\n", device->Target().subgroupSize);
            return VulkanTestSkipped;
        }
        for (const std::uint32_t data : {20u, 2u, 31u}) {
            Run(*device, 32, 32, 7u, data);
            Check(32, 32, 7u, data);
            Run(*device, 64, 64, 7u, data);
            Check(64, 64, 7u, data);
        }
        std::puts("vector window narrowing tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
