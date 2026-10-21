// Metal twin of tests/execution/DepthRangeRemapDraw.cpp (agc_driver_depth_range_remap_draw, upstream 53212b18a),
// swapped in by twins/SKIPFIX5-tests.cmake. The upstream file is compiled here with its main renamed, so its draws,
// transforms and cases keep tracking upstream.
//
// Upstream checks two paths for a guest viewport depth range outside [0, 1]: the device's own unrestricted range
// (VK_EXT_depth_range_unrestricted), and the fallback for devices without it, which remaps clip z in the vertex
// shader (Graphics::ApplyDepthRemap, SpirvTarget::depthRangeRemap) and clamps with depthClamp. The Metal device is
// an unrestricted one: MTLViewport takes znear/zfar outside [0, 1] and MetalDraw.mm passes the guest range to it,
// clamping with MTLDepthClipModeClamp when the guest asks (Context::depthClamp), and the vertex function maps
// negative-one-to-one clip z itself (DepthNegativeOneToOneMacro, Context::depthClipControl). So Metal never takes
// the remap fallback (Target().depthRangeRemap stays false; the MSL backend has no remap, MslEmitter.cpp), and this
// twin runs every case of upstream's unrestricted path, with upstream's expected coverage, requiring that path
// instead of skipping without it. The remap-only cases (clamp bounds inside the viewport range, the shared-pipeline,
// fetchless and mode-0 remap checks) test the Vulkan fallback and do not run (TechnicalDebt metal-depth-range-remap).

#define main AgcDriverDepthRangeRemapDrawVulkanMain
#include "../../../tests/execution/DepthRangeRemapDraw.cpp"
#undef main

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = device->graphicsContext();
        Require(context.depthClamp, "the Metal device reports no depth clamping (MTLDepthClipModeClamp)");
        Require(context.depthRangeUnrestricted, "the Metal device reports a restricted viewport depth range");
        Require(context.depthClipControl, "the Metal device reports no negative-one-to-one depth");
        Require(!device->Target().depthRangeRemap, "the Metal target asks for the vertex-shader depth remap");
        constexpr float Step = 1.0f / 64.0f;
        constexpr Transform minecraft{true, 1.0f, 0.0f, -1.0f, 1.0f};
        constexpr Transform widened{false, 2.0f, -0.5f, 0.0f, 1.0f};
        constexpr Transform bounded{false, 2.0f, -0.5f, 0.25f, 0.75f};
        // Upstream's cases, unchanged (DepthRangeRemapDraw.cpp main).
        constexpr std::array<Case, 13> cases{{
            {minecraft, 0.5f - Step, true, 0.5f, false, false, "negative-one-to-one depth just below 0.5"},
            {minecraft, 0.5f + Step, true, 0.5f, true, false, "negative-one-to-one depth just above 0.5"},
            {minecraft, -1.5f, false, 0.5f, true, false, "negative-one-to-one clip z in front of the near plane"},
            {minecraft, 1.5f, false, 0.5f, true, false, "negative-one-to-one clip z behind the far plane"},
            {widened, 0.5f - Step / 2.0f, true, 0.5f, false, false, "scaled zero-to-one depth just below 0.5"},
            {widened, 0.5f + Step / 2.0f, true, 0.5f, true, false, "scaled zero-to-one depth just above 0.5"},
            {widened, -0.125f, false, 0.5f, true, false, "zero-to-one clip z in front of the near plane"},
            {widened, 1.125f, false, 0.5f, true, false, "zero-to-one clip z behind the far plane"},
            {bounded, 0.4f, true, 0.35f, false, false, "depth 0.3 inside clamp bounds [0.25, 0.75], probed at 0.35"},
            {bounded, 0.4f, true, 0.25f, true, false, "depth 0.3 inside clamp bounds [0.25, 0.75], probed at 0.25"},
            {bounded, 0.6f, true, 0.65f, true, false, "depth 0.7 inside clamp bounds [0.25, 0.75], probed at 0.65"},
            {bounded, 0.2f, true, 0.2f, true, true, "depth -0.1 clamped to 0.25, probed at 0.2"},
            {bounded, 0.8f, true, 0.8f, false, true, "depth 1.1 clamped to 0.75, probed at 0.8"},
        }};
        std::size_t surfaces = 0;
        std::size_t ran = 0;
        for (const auto& test : cases) {
            if (test.remapOnly) continue;
            Require(surfaces < Depths.size(), "the test ran out of depth surfaces");
            const auto depth = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Depths[surfaces++].data()));
            const auto drawn = Draw(*device, depth, test.transform, false, test.z, true);
            const auto passed = Draw(*device, depth, Plain, false, test.probe, false);
            if (drawn != (test.drawn ? Harness::Covered : 0u) || passed != (test.covered ? Harness::Covered : 0u)) {
                char detail[256];
                std::snprintf(detail, sizeof(detail), "%s (unrestricted path): the draw covered %u of %u pixels (expected %s) and the probe passed %u (expected %s)", test.what, drawn, Harness::Covered, test.drawn ? "all" : "none", passed, test.covered ? "all" : "none");
                throw std::runtime_error(detail);
            }
            ++ran;
        }
        Require(ran == 11u, "the unrestricted path ran " + std::to_string(ran) + " of upstream's 11 cases");
        std::printf("depth range remap draw tests passed on Metal (%zu cases of the unrestricted path; the vertex-shader remap fallback is Vulkan-only)\n", ran);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
