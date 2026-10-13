# Work package T1 (test infrastructure): upstream tests of Vulkan internals = Metal twin (docs/mac/M2_DESIGN.md §8).
# GraphicsTwin.mm compiles tests/Graphics.cpp with its main renamed and runs its checks that do not drive Vulkan.
# RecorderTwin.mm runs the Metal port each package registers for a tests/Recorder.cpp mode (RecorderTwins.h);
# tests/ResidentPresent.cpp, which only the full Recorder.cpp run calls, goes with it.
# BdaDeviceTwin.mm runs the BDA contract checks and tests/BdaExecution.cpp's cases through the MSL Bda prelude and
# the Metal device's page table (it includes BdaExecution.cpp for its CPU decode checks); PendingTwins.mm reports the owner of each port still missing, or
# the allowlisted reason of a Vulkan-only behaviour. A package that ports one moves its pair to its own list.
list(APPEND agcMetalTwins
        "tests/Graphics.cpp=specifics/macos/tests/GraphicsTwin.mm"
        "tests/Recorder.cpp=specifics/macos/tests/RecorderTwin.mm"
        "tests/ResidentPresent.cpp=specifics/macos/tests/RecorderTwin.mm"
        "tests/BdaDevice.cpp=specifics/macos/tests/BdaDeviceTwin.mm"
        "tests/BdaExecution.cpp=specifics/macos/tests/BdaDeviceTwin.mm"
        # agc_driver_graphics_tests compiles the texture twins without
        # Execution/src/VulkanDevice.cpp, which carries D4's Metal::Vk image calls in the library
        # (each target gets a twin source once).
        "Graphics/src/Texture.cpp=specifics/macos/src/MetalVulkanImages.mm"
        "Graphics/src/Texture.cpp=specifics/macos/src/MetalTextures.mm"
        "Graphics/src/Texture.cpp=specifics/macos/src/MetalTextureFormat.mm"
        "tests/execution/PipelineOptimizedReadback.cpp=specifics/macos/tests/PendingTwins.mm"
        "tests/execution/OversizedDescriptorSets.cpp=specifics/macos/tests/PendingTwins.mm"
        "tests/execution/PooledBufferRange.cpp=specifics/macos/tests/PendingTwins.mm"
        "tests/execution/SampledCacheBudget.cpp=specifics/macos/tests/PendingTwins.mm"
)
