// Metal twin of Graphics/src/MultisampleReadback.cpp (upstream ab52b02381): the Vulkan compute readback of
// multisampled images into a buffer. Nothing on the Metal path calls it (multisampled targets read back through
// MultisampleTarget.mm's Metal copies, D27/D28), so both entry points refuse.
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include <stdexcept>

namespace AgcDriver::Graphics {

VkPipeline TextureDetiler::multisampleReadbackPipeline(std::uint32_t, bool) {
    throw std::runtime_error("TextureDetiler: the Vulkan multisample readback pipeline does not exist on Metal");
}

void TextureDetiler::DispatchMultisampleReadback(VkCommandBuffer, VkFormat, VkBuffer, std::uint64_t, std::uint64_t, VkImageView, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    throw std::runtime_error("TextureDetiler: the Vulkan multisample readback is not implemented on Metal (MultisampleTarget.mm reads samples back)");
}

}
