# Work package D6 (draws, attachments, presentation): upstream Vulkan source = Metal twin.
list(APPEND agcMetalTwins
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalPresentation.mm"
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalDraw.mm"
)
