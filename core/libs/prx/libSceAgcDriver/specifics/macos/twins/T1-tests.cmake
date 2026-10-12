# Work package T1 (test infrastructure): upstream tests of Vulkan internals = Metal twin (docs/mac/M2_DESIGN.md §8).
# GraphicsTwin.mm compiles tests/Graphics.cpp with its main renamed and runs its checks that do not drive Vulkan.
# RecorderTwin.mm runs the Metal port each package registers for a tests/Recorder.cpp mode (RecorderTwins.h);
# tests/ResidentPresent.cpp, which only the full Recorder.cpp run calls, goes with it.
list(APPEND agcMetalTwins
        "tests/Graphics.cpp=specifics/macos/tests/GraphicsTwin.mm"
        "tests/Recorder.cpp=specifics/macos/tests/RecorderTwin.mm"
        "tests/ResidentPresent.cpp=specifics/macos/tests/RecorderTwin.mm"
)
