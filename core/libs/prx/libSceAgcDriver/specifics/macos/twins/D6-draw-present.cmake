# Work package D6 (draws, attachments, presentation): upstream Vulkan source = Metal twin.
list(APPEND agcMetalTwins
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalPresentation.mm"
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalDraw.mm"
        "Graphics/src/DepthSurface.cpp=specifics/macos/src/DepthSurface.mm"
)
# The CMASK fast-clear test's Metal twin: hand-written MSL cover shaders instead of the SPIR-V ones.
list(APPEND agcMetalTwins
        "tests/CmaskFastClear.cpp=specifics/macos/tests/CmaskFastClearTwin.mm"
        "tests/execution/ProvokingVertex.cpp=specifics/macos/tests/ProvokingVertexTwin.mm"
)
