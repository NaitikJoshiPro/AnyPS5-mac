# Tessellation on Metal (DECISIONS D24): the linked object/mesh libraries of tessellated draws ride with the draw path
# (MetalDraw.mm, the VulkanDevice.cpp twin).
list(APPEND agcMetalTwins
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalTessellation.mm"
)
