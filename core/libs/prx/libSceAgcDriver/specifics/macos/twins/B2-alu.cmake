# Work package B2 (ALU and f64): upstream tests of SpirvBackend internals = Metal twin (docs/mac/M2_DESIGN.md §8).
# InterpolationModesTwin.mm runs tests/execution/InterpolationModes.cpp's hardware table (compiled unchanged with its
# main renamed) through the MSL FPInterpolateF32/F16 routine instead of a hand-built SPIR-V module.
list(APPEND agcMetalTwins
        "tests/execution/InterpolationModes.cpp=specifics/macos/tests/InterpolationModesTwin.mm"
)
