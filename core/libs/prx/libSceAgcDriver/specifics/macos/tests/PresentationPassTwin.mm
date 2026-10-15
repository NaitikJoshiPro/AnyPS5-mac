// Metal twin of tests/PresentationPass.cpp (agc_driver_presentation_pass, upstream UPSYNC3), swapped in by
// twins/UPSYNC3-tests.cmake. Upstream builds its own Vulkan instance (libvulkan through SDL_LoadObject) and checks
// the GPU presentation pass (Execution/src/PresentationPass.cpp: resident display images drawn, scaled and
// PQ-converted on the GPU, slices of pending array images presented, the blit letterbox clear). That pass is
// Vulkan-only and unreachable on macOS: the Metal presenter decodes every display buffer on the CPU
// (MetalPresentation.mm) and scales it with MetalFX or bilinearly (docs/dev/TechnicalDebt.md#metal-present-pass,
// #metal-d6-present), so the upstream file would only skip "no usable Vulkan device" here (and its Vulkan image
// helpers do not compile against the Metal Resources.hpp overlay).
// What runs: upstream's letterboxBandTests, copied unchanged (the aspect-fit rectangle and the letterbox bands of
// PresentationScaler, which the library still compiles, must tile the destination). Then the GPU pass cases skip
// naming the missing Metal feature; the skip is not allowlisted (a feature gap, not an Apple limit), so
// tools/check_skips.py --job macos reports it until the Metal presenter has a GPU pass.

#include "prx/libSceAgcDriver/Execution/include/AspectFit.hpp"
#include "prx/libSceAgcDriver/Execution/include/PresentationScaler.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int Skipped = 77;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void letterboxBandTests() {
    const std::array<std::array<std::uint32_t, 4>, 7> cases{{{3840, 2160, 2261, 1272}, {3840, 2160, 3840, 2160}, {3840, 2160, 1000, 1000}, {3840, 2160, 2000, 500}, {200, 130, 260, 130}, {200, 130, 117, 76}, {7, 3, 5, 11}}};
    for (const auto& [width, height, destinationWidth, destinationHeight] : cases) {
        const auto rect = AgcDriver::ComputeContainRect_nid_postfix(width, height, destinationWidth, destinationHeight);
        const auto bands = AgcDriver::PresentationScaler::LetterboxBands(width, height, destinationWidth, destinationHeight);
        std::vector<std::uint8_t> covered(static_cast<std::size_t>(destinationWidth) * destinationHeight, 0);
        auto cover = [&](const AgcDriver::AspectFitRect& area) {
            Require(area.width != 0 && area.height != 0, "an empty letterbox band");
            for (std::uint32_t y = 0; y < area.height; ++y) {
                for (std::uint32_t x = 0; x < area.width; ++x) ++covered[static_cast<std::size_t>(area.y + y) * destinationWidth + area.x + x];
            }
        };
        cover(rect);
        for (const auto& band : bands) cover(band);
        Require(std::all_of(covered.begin(), covered.end(), [](std::uint8_t count) { return count == 1; }), "the letterbox bands and the image do not tile the destination: " + std::to_string(width) + "x" + std::to_string(height) + " into " + std::to_string(destinationWidth) + "x" + std::to_string(destinationHeight));
    }
    Require(AgcDriver::PresentationScaler::LetterboxBands(3840, 2160, 1920, 1080).empty(), "an exact fit has letterbox bands");
}

}

int main() {
    try {
        letterboxBandTests();
        std::puts("letterbox band tests passed");
        std::puts("skipped, the GPU presentation pass is not on Metal yet: the Metal presenter decodes display buffers on the CPU "
                  "(docs/dev/TechnicalDebt.md#metal-present-pass)");
        return Skipped;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
