# Work package D1 (device, submission, compute dispatch): upstream Vulkan source = Metal twin.
list(APPEND agcMetalTwins
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalDevice.mm"
)
