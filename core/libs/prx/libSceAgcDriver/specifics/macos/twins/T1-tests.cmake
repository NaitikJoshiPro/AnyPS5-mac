# Work package T1 (test infrastructure): upstream tests of Vulkan internals = Metal twin (docs/mac/M2_DESIGN.md §8).
# GraphicsTwin.mm compiles tests/Graphics.cpp with its main renamed and runs its checks that do not drive Vulkan.
list(APPEND agcMetalTwins
        "tests/Graphics.cpp=specifics/macos/tests/GraphicsTwin.mm"
)
