# Work packages D4 (textures, detiling, formats) and D5's image/sampler descriptors (src/MetalImages.h).
# The Metal image path is new code beside the device rather than twins of Graphics/src/Texture.cpp and
# ShaderResources.cpp (those stay compiled but unreachable on Metal: their Vulkan entry points have no
# resolver), so its sources ride with the device's original, Execution/src/VulkanDevice.cpp.
list(APPEND agcMetalTwins
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalTextureFormat.mm"
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalTextureDetiler.mm"
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalTextures.mm"
        "Execution/src/VulkanDevice.cpp=specifics/macos/src/MetalVulkanImages.mm"
        "Graphics/src/TextureDetiler.cpp=specifics/macos/src/TextureDetiler.mm"
        "Graphics/src/TextureDetilerDescriptors.cpp=specifics/macos/src/TextureDetiler.mm"
        "Graphics/src/Texture.cpp=specifics/macos/src/Texture.mm"
        # UPSYNC4: the Vulkan multisample compute readback (ab52b02381) refuses on Metal.
        "Graphics/src/MultisampleReadback.cpp=specifics/macos/src/MultisampleReadbackMetal.mm"
        "Graphics/src/UnitShadow.cpp=specifics/macos/src/UnitShadow.mm"
        "Graphics/src/GpuColorTransfer.cpp=specifics/macos/src/GpuColorTransfer.mm"
        "tests/Recorder.cpp=specifics/macos/tests/D3RecorderModes.mm"
)
