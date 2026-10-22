// Metal twin of tests/execution/NativeMultisampleDepth.cpp (agc_driver_native_multisample_depth and its modes
// --retirement, --clears, --clear-refusals: agc_driver_native_depth_retirement, agc_driver_native_depth_clears,
// agc_driver_native_depth_clear_refusals; upstream ee1ba73a2f, f56e3bc467, da9fe3c708, 7d0dd69f0a, 9e27d0b623),
// swapped in by twins/UPSYNC3-tests.cmake. Upstream builds its own Vulkan instance and samples a multisampled depth
// attachment natively (a sampled view of the attachment's own image, the resident-depth guest guards, the HTILE
// clear transitions of a native owner and the attachment feedback refusal). The Metal depth surfaces keep a
// multisampled depth target as a multisample array (DECISIONS D27/D28) and have no sampled view of it:
// DepthSurfaceTexture refuses a multisampled base with a runtime_error (DepthSurface.mm), so no mode can run here.
// Every mode skips naming the missing Metal feature; the skip is not allowlisted (a feature gap, not an Apple
// limit), so tools/check_skips.py --job macos reports it until the Metal path samples multisampled depth.

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    const std::string mode = argc == 2 ? argv[1] : "";
    if (!mode.empty() && mode != "--retirement" && mode != "--clears" && mode != "--clear-refusals") {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    std::printf("skipped, native multisampled depth sampling is not on Metal yet (%s): DepthSurfaceTexture refuses a multisampled depth surface (docs/dev/TechnicalDebt.md#metal-native-msaa-depth)\n", mode.empty() ? "sampling" : mode.c_str() + 2);
    return 77;
}
