// Metal twin of tests/execution/LdsMixedWidthAtomics.cpp (agc_driver_lds_mixed_width_atomics and, with --lock-budget,
// agc_driver_lds_lock_budget; upstream ae56b9558c/0a9ee3698a), swapped in by twins/UPSYNC3-tests.cmake. The upstream
// file is compiled here with Run, LockBudget and main renamed; its programs, inputs and GPU checks (Check) keep
// tracking upstream, and every mode without --lock-budget runs upstream's main unchanged. Upstream's lock budget
// counts the Workgroup-storage bytes of result.spirv, which is empty for an MSL target; this file's Run counts the
// MSL source's `threadgroup uint name[N]` declarations instead (MslMemory.cpp MslLdsStorage: the guest LDS plus the
// lock dword, or device-memory LDS when they do not fit, MslAbi workgroupMemoryStrideDwords), with upstream's checks
// on them: within the device limit, the device-backed allocation upstream expects, and the exact-fit case keeping
// its lock in threadgroup memory. The dispatches and readbacks are upstream's Run.

// The upstream file's headers first, so the renames below reach only the upstream file's own definitions.
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "../../../tests/execution/VulkanTestDevice.hpp"
#include <optional>
#include <string>
#include <string_view>

#define Run AgcDriverLdsMixedWidthAtomicsVulkanRun
#define LockBudget AgcDriverLdsMixedWidthAtomicsVulkanLockBudget
#define main AgcDriverLdsMixedWidthAtomicsVulkanMain
#include "../../../tests/execution/LdsMixedWidthAtomics.cpp"
#undef main
#undef LockBudget
#undef Run

namespace {

std::uint64_t MslThreadgroupBytes(const ShaderRecompiler::RecompileResult& result) {
    Require(!result.msl.empty(), "lds lock budget: the compile has no MSL source");
    const std::string& source = result.msl.Source();
    constexpr std::string_view declaration = "threadgroup uint ";
    std::uint64_t total = 0;
    for (std::size_t at = source.find(declaration); at != std::string::npos; at = source.find(declaration, at + 1u)) {
        const auto open = source.find_first_of("[;(", at + declaration.size());
        if (open == std::string::npos || source[open] != '[') continue;
        total += std::stoull(source.substr(open + 1u)) * sizeof(std::uint32_t);
    }
    return total;
}

void Run(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, const char* name, std::uint64_t operations, std::uint32_t repeats, std::uint32_t ldsDwords, std::uint32_t expectedDeviceDwords) {
    std::vector<std::uint32_t> userData(8, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, ldsDwords, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    const auto bytes = MslThreadgroupBytes(result);
    const auto limit = request.target.maxWorkgroupSharedMemoryBytes;
    Require(bytes <= limit, "lds lock budget: guest LDS and lock exceed the device limit before dispatch");
    Require(result.workgroupMemoryDwords == expectedDeviceDwords, "lds lock budget: unexpected device-backed LDS allocation before dispatch");
    if (expectedDeviceDwords == 0u) Require(bytes == limit, "lds lock budget: exact-fit case did not retain its lock");
    std::printf("lds lock budget: %s declarations verified; guest=%u; threadgroup=%llu; device=%u\n", name, ldsDwords * 4u, static_cast<unsigned long long>(bytes), result.workgroupMemoryDwords);
    // Metal shader validation (MTL_SHADER_VALIDATION=1) instruments threadgroup memory and doubles a pipeline's
    // staticThreadgroupMemoryLength (measured: 32768 -> 65536 for the exact-fit program), so the validation layer
    // rejects the dispatch of a program that fills the device limit; without the instrumentation it runs.
    const char* validation = std::getenv("MTL_SHADER_VALIDATION");
    if (bytes * 2u > limit && validation != nullptr && std::string_view(validation) != "0") {
        std::printf("lds lock budget: %s dispatch not run under MTL_SHADER_VALIDATION (its instrumentation doubles the %llu threadgroup bytes past the %u-byte limit)\n", name, static_cast<unsigned long long>(bytes), limit);
        return;
    }
    // Upstream's Run without its SPIR-V checks: the same compile, dispatched and read back on every lane.
    AgcDriverLdsMixedWidthAtomicsVulkanRun(device, code, name, operations, repeats, ldsDwords, std::nullopt);
    std::printf("lds lock budget: %s GPU readback verified across 256 lanes\n", name);
}

void LockBudget(AgcDriver::VulkanDevice& device) {
    const auto target = device.Target();
    const auto limit = target.maxWorkgroupSharedMemoryBytes;
    std::printf("lds lock budget: robustBufferAccess2=%u; maxWorkgroupSharedMemoryBytes=%u\n", static_cast<unsigned>(target.robustBufferAccess2), limit);
    if (std::getenv("ANYPS5_REQUIRE_ROBUST2") != nullptr) Require(target.robustBufferAccess2, "lds lock budget: robustBufferAccess2 was not enabled");
    Require(limit >= 32u && limit % 4u == 0u, "lds lock budget: unexpected device shared-memory limit");
    const auto operations = static_cast<std::uint64_t>(Threads) * Iterations;
    Run(device, UniformCode, "limit", operations, 1, limit / 4u, limit / 4u + 1u);
    Run(device, UniformCode, "limit-4", operations, 1, (limit - 4u) / 4u, 0u);
    std::puts("lds lock budget tests passed");
}

}

int main(int argc, char** argv) {
    if (argc != 2 || std::string_view(argv[1]) != "--lock-budget") return AgcDriverLdsMixedWidthAtomicsVulkanMain(argc, argv);
    (void)&AgcDriverLdsMixedWidthAtomicsVulkanLockBudget;
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        LockBudget(*device);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
