// Metal twin of tests/execution/SampleOrder.cpp (agc_driver_sample_order, agc_driver_sample_order_grid_reference,
// agc_driver_sample_order_recipe_depth_locations; upstream 57ceb9e370, adc46f3ea0, f8e8dc6f67, d9bd0f9b4d, b6cde5698),
// swapped in by twins/SKIPFIX5-msaa.cmake. Upstream builds its reference images with its own Vulkan images, storage
// views, compute and graphics pipelines (context.Function<vkCreate...>) and checks the driver's guest sample order:
// the reorder passes at render pass bounds (RecordSampleOrder), the pixel shader's SampleID/mask permutation, 2x2
// sample location grids (VK_EXT_sample_locations) and a depth-only draw replayed from its draw recipe. None of that
// machinery is on the Metal driver yet: HostSampleOrder names no order, the reorder has no Metal form (Metal compute
// cannot write multisampled textures), setSamplePositions takes one pattern for every pixel, and the Metal device
// writes no draw recipes. Each mode skips naming its gap; the skips are not allowlisted (Metal driver features not
// written yet, tools/check_skips.py --job macos reports them). The device's capability answers the upstream modes
// gate on are checked first, so a skip never stands in for a wrong capability report.

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "../../../tests/execution/VulkanTestDevice.hpp"
#include <cstdio>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? std::string_view(argv[1]) : std::string_view();
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = device->graphicsContext();
        AgcDriver::Graphics::Require(context.sampleRateShading, "the Metal device does not report sampleRateShading ([[sample_id]] fragment functions run per sample)");
        AgcDriver::Graphics::Require((context.limits.framebufferColorSampleCounts & 2u) != 0u && (context.limits.framebufferDepthSampleCounts & 2u) != 0u, "the Metal device does not report 2x color and depth targets");
        AgcDriver::Graphics::Require(context.storageImageMultisample, "the Metal device does not report multisampled storage (D22/D28 storage shadows)");
        if (mode.empty() || mode == "--bench") {
            std::puts("skipped, the guest sample order is not on Metal: HostSampleOrder names no order and the reorder passes at render pass bounds have no Metal form (per-sample render passes like CopyStorageToMultisample): TechnicalDebt metal-sample-order");
        } else if (mode == "--grid-reference") {
            std::puts("skipped, 2x2 sample location grids are not on Metal: setSamplePositions takes one pattern for every pixel, so the grid draw has no reference to match: TechnicalDebt metal-sample-location-grid");
        } else if (mode == "--recipe-depth-locations") {
            std::puts("skipped, the Metal device reports no custom sample locations to the shared sample-order code and writes no draw recipes to replay: TechnicalDebt metal-sample-order");
        } else {
            std::cerr << "unknown mode " << mode << '\n';
            return 1;
        }
        return VulkanTestSkipped;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
