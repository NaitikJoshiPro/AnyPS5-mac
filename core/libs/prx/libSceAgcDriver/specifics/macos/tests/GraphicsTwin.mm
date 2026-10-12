// Metal twin of tests/Graphics.cpp (agc_driver_graphics and its flags), swapped in by twins/T1-tests.cmake.
// The upstream file is compiled here unchanged with its main renamed, so its checks keep tracking upstream (see
// tools/mac/port_ledger.py). Its CPU-side checks of the graphics state, layouts, tiling and validation run on Metal
// as they are; the checks that drive Vulkan objects through MockVulkan test code the Metal twins replace, so they are
// listed in VulkanOnly with the work package that owns the Metal counterpart, and are not run. `--probe` runs every
// check, listed or not, in its own process and prints which pass: the owner of a twin that breaks a check moves it
// to VulkanOnly with a reason, or ports it.

#define main AgcDriverGraphicsVulkanMain
#include "../../../tests/Graphics.cpp"
#undef main

#include <sys/wait.h>
#include <unistd.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Check {
    const char* name;
    std::function<void()> run;
};

void DrawValidationChecks() {
    const AgcDriver::Graphics::Context context{};
    const AgcDriver::Graphics::State state{};
    AgcDriver::Pm4::DrawParameters draw{0, 0, 0, 1, 0, false};
    AgcDriver::Graphics::Draw(context, state, draw, {});
    draw.indexCount = 3;
    draw.instanceCount = 0;
    AgcDriver::Graphics::Draw(context, state, draw, {});
    draw.instanceCount = 2;
    draw.firstInstance = 0xffffffffu;
    expectFailure([&] { AgcDriver::Graphics::Draw(context, state, draw, {}); }, "instance range overflow");
    draw.firstInstance = 0;
    draw.firstVertex = 0xffffffffu;
    expectFailure([&] { AgcDriver::Graphics::Draw(context, state, draw, {}); }, "vertex range overflow");
    draw.firstVertex = 0;
    draw.indexAddress = 1;
    expectFailure([&] { AgcDriver::Graphics::Draw(context, state, draw, {}); }, "must not reference an index buffer");
    draw.indexAddress = 0;
    draw.flags = 1;
    expectFailure([&] { AgcDriver::Graphics::Draw(context, state, draw, {}); }, "draw modifiers");
    auto bounded = state;
    bounded.depthBoundsTest = true;
    bounded.minDepthBounds = 0.25f;
    AgcDriver::Graphics::ValidateDepthBounds(context, bounded);
    bounded.minDepthBounds = 1.5f;
    expectFailure([&] { AgcDriver::Graphics::ValidateDepthBounds(context, bounded); }, "depth bounds outside [0, 1]");
    auto unrestricted = context;
    unrestricted.depthRangeUnrestricted = true;
    AgcDriver::Graphics::ValidateDepthBounds(unrestricted, bounded);
}

std::vector<Check> Checks() {
    return {
        {"drawValidation", DrawValidationChecks},
        {"RunGuestLeaseWaitTests", RunGuestLeaseWaitTests},
        {"stateTests", stateTests},
        {"hdrComponentSwapTests", hdrComponentSwapTests},
        {"depthMaintenanceTests", depthMaintenanceTests},
        {"hardwareScreenOffsetTests", hardwareScreenOffsetTests},
        {"srgb8TargetTests", srgb8TargetTests},
        {"DepthClipTests", DepthClipTests},
        {"DepthStencilTests", DepthStencilTests},
        {"OneDimensionalColorTests", OneDimensionalColorTests},
        {"htileDepthClearTests", htileDepthClearTests},
        {"ZExportTests", ZExportTests},
        {"unorm10_11_11TargetTests", unorm10_11_11TargetTests},
        {"DepthBoundsBiasTests", DepthBoundsBiasTests},
        {"conservativeZExportTests", conservativeZExportTests},
        {"orderedPixelShaderTests", orderedPixelShaderTests},
        {"stencilRefExportTests", stencilRefExportTests},
        {"ProvokingVertexTests", ProvokingVertexTests},
        {"ProvokingVertexRenderPassTests", ProvokingVertexRenderPassTests},
        {"ConservativeRasterizationTests", ConservativeRasterizationTests},
        {"DisabledColorTests", DisabledColorTests},
        {"OpaqueDestinationAlphaTests", OpaqueDestinationAlphaTests},
        {"CompactedExportTests", CompactedExportTests},
        {"ReversedComponentOrderTests", ReversedComponentOrderTests},
        {"ShadersWriteMemoryTests", ShadersWriteMemoryTests},
        {"DualSourceBlendTests", DualSourceBlendTests},
        {"metadataPassTests", metadataPassTests},
        {"cmaskTests", cmaskTests},
        {"uint16ExportTests", uint16ExportTests},
        {"uint8x4TargetTests", uint8x4TargetTests},
        {"multisampleTests", multisampleTests},
        {"MsaaEnableSingleSampleTests", MsaaEnableSingleSampleTests},
        {"ShaderStageTests", ShaderStageTests},
        {"NggVertexWorkgroupTests", NggVertexWorkgroupTests},
        {"TuningFieldTests", TuningFieldTests},
        {"PixelInputLayoutTests", PixelInputLayoutTests},
        {"UnexportedPixelInputTests", UnexportedPixelInputTests},
        {"ColorPipeBankXorTests", ColorPipeBankXorTests},
        {"HeaderlessVertexStageTests", HeaderlessVertexStageTests},
        {"HiddenUserWordVertexTablesTests", HiddenUserWordVertexTablesTests},
        {"ComputeScratchTests", ComputeScratchTests},
        {"shaderUserDataTailPaddingTests", shaderUserDataTailPaddingTests},
        {"InitialContextTests", InitialContextTests},
        {"AlphaToCoverageTests", AlphaToCoverageTests},
        {"descriptorPoolTests", descriptorPoolTests},
        {"pushConstantTests", pushConstantTests},
        {"resourceTests", resourceTests},
        {"descriptorSnapshotTests", descriptorSnapshotTests},
        {"descriptorCacheTests", descriptorCacheTests},
        {"misalignedShaderDataTests", misalignedShaderDataTests},
        {"debugBranchTests", debugBranchTests},
        {"textureCacheBudgetTests", textureCacheBudgetTests},
        {"sampledTextureBudgetTests", sampledTextureBudgetTests},
        {"sampledBudgetReportTests", sampledBudgetReportTests},
        {"meshArgumentTests", meshArgumentTests},
        {"meshIndexBufferTests", meshIndexBufferTests},
        {"stagingPoolBudgetTests", stagingPoolBudgetTests},
        {"validationTests", validationTests},
        {"vertexCopyTests", vertexCopyTests},
        {"highestDrawIndexTests", highestDrawIndexTests},
        {"storeAtFlipTests", storeAtFlipTests},
        {"nullVertexDescriptorTests", nullVertexDescriptorTests},
        {"vertexZeroFillTests", vertexZeroFillTests},
        {"vertexZeroFillMergeTests", vertexZeroFillMergeTests},
        {"parallelCompareTests", parallelCompareTests},
        {"pixelParameterSlotTests", pixelParameterSlotTests},
        {"rectListTests", rectListTests},
        {"floatControlsModeTests", floatControlsModeTests},
        {"barycentricEmulationTests", barycentricEmulationTests},
        {"RunGuestAllocationTests", RunGuestAllocationTests},
        {"RunUnmappedGapTests", RunUnmappedGapTests},
        {"RunColorTargetLayoutTests", RunColorTargetLayoutTests},
        {"RunLiveStackAccessTests", RunLiveStackAccessTests},
        {"RunTextureFormatTests", RunTextureFormatTests},
        {"RunTextureTilingTests", RunTextureTilingTests},
        {"RunGuestTextureResourceTests", RunGuestTextureResourceTests},
        {"RunDepthSurfaceReuseTests", RunDepthSurfaceReuseTests},
        {"RunGuestSamplerResourceTests", RunGuestSamplerResourceTests},
    };
}

// Checks of Vulkan objects through MockVulkan, with the work package whose Metal twin replaces what they test.
const std::vector<std::pair<std::string, std::string>> VulkanOnly{
    {"resourceTests", "Vulkan buffers, images and pool slabs on MockVulkan: D2 (BufferPool, Resources twins)"},
    {"descriptorSnapshotTests", "descriptor snapshots recorded into a Vulkan recorder: D5 (ShaderResources twin), D2"},
    {"misalignedShaderDataTests", "shader data uploaded through the Vulkan buffer pool: D5 (ShaderResources twin), D2"},
    {"vertexZeroFillTests", "zero-filled vertex buffers from the Vulkan buffer pool: D6 (Draw twin), D2"},
    {"RunBdaResourceTests", "BDA buffers and descriptors on MockVulkan: D3 (BdaResources twin)"},
    {"RunDepthSurfaceReuseTests", "depth surface copies recorded into Vulkan command batches: D6 (DepthSurface twin), D2"},
    {"RunTextureDetilerTests", "detile dispatch descriptors on MockVulkan: D4 (TextureDetiler twin)"},
};

bool Listed(const std::string& name) {
    for (const auto& [listed, reason] : VulkanOnly) {
        if (listed == name) return true;
    }
    return false;
}

std::string Run(const Check& check) {
    try {
        check.run();
        return {};
    } catch (const std::exception& error) {
        return error.what();
    }
}

int Probe() {
    std::size_t failed = 0;
    for (const auto& check : Checks()) {
        std::fflush(stdout);
        const pid_t child = fork();
        if (child == 0) {
            const auto error = Run(check);
            if (!error.empty()) std::fprintf(stderr, "%s\n", error.c_str());
            std::fflush(stderr);
            std::_Exit(error.empty() ? 0 : 1);
        }
        int status = 0;
        waitpid(child, &status, 0);
        const bool passed = WIFEXITED(status) && WEXITSTATUS(status) == 0;
        failed += passed ? 0u : 1u;
        std::printf("%s %s%s\n", passed ? "PASS" : "FAIL", check.name, Listed(check.name) ? " (listed as Vulkan-only)" : "");
    }
    std::printf("%zu check(s) failed in isolation\n", failed);
    return 0;
}

}

int main(int argc, char** argv) {
    if (argc == 2 && (std::string_view(argv[1]) == "--descriptor-layout-flags" || std::string_view(argv[1]) == "--descriptor-pool-reset-failure")) {
        return AgcDriverGraphicsVulkanMain(argc, argv);
    }
    setenv("APS5_PIN_WAIT_MS", "200", 1);
    setenv("APS5_HEAP_MIRROR_MIB", "4", 1);
    if (argc == 2 && std::string_view(argv[1]) == "--probe") return Probe();
    if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [--probe | --descriptor-layout-flags | --descriptor-pool-reset-failure]\n";
        return 2;
    }
    std::size_t ran = 0;
    for (const auto& check : Checks()) {
        if (Listed(check.name)) continue;
        const auto error = Run(check);
        if (!error.empty()) {
            std::cerr << check.name << ": " << error << '\n';
            return 1;
        }
        ++ran;
    }
    std::cout << "Graphics validation tests passed on Metal: " << ran << " checks; not run, Vulkan-only:";
    for (const auto& [name, reason] : VulkanOnly) std::cout << ' ' << name << " (" << reason << ')';
    std::cout << '\n';
    return 0;
}
