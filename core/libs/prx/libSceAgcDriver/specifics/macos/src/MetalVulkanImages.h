#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALVULKANIMAGES_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALVULKANIMAGES_H

// Work package D4: the Vulkan image calls of the texture twins (Texture.mm, ShaderResources.mm)
// carried out on Metal (D-M2-3: the Vulkan handle types carry Metal objects while the shared
// headers keep them). A VkImage holds a CF-retained id<MTLTexture> (made by CreateImage, freed by
// DestroyImage), a VkImageView a CF-retained texture view; VkDeviceMemory is a token (Metal
// textures own their memory). Commands record into the Metal::CommandStream a VkCommandBuffer
// carries (Metal 4 compute encoders do the copies); image layouts do not exist on Metal, so
// barriers are execution and memory barriers only. Each function has the signature of the Vulkan
// entry point it replaces, so the twins call `Metal::Vk::CreateImage` where upstream calls
// `context.Function<PFN_vkCreateImage>("vkCreateImage")`. Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <cstdint>

namespace AgcDriver::Graphics::Metal {

// The texture a VkImage or VkImageView carries (unretained), nil for a null handle.
inline id<MTLTexture> TextureOfImage(VkImage image) {
    return (__bridge id<MTLTexture>)reinterpret_cast<void*>(image);
}
inline id<MTLTexture> TextureOfView(VkImageView view) {
    return (__bridge id<MTLTexture>)reinterpret_cast<void*>(view);
}
// Handles for textures made elsewhere: a retained reference released by DestroyImage/DestroyImageView.
VkImage RetainImage(id<MTLTexture> texture);
VkImageView RetainView(id<MTLTexture> texture);

struct PixelFormatInfo {
    MTLPixelFormat format = MTLPixelFormatInvalid;
    std::uint32_t blockBytes = 0;
    std::uint32_t blockWidth = 1;
    std::uint32_t blockHeight = 1;
};
// The Metal pixel format of a Vulkan format (invalid: none; three-channel 32-bit formats have none).
PixelFormatInfo PixelFormatOf(VkFormat format);

namespace Vk {

VkResult CreateImage(VkDevice device, const VkImageCreateInfo* info, const VkAllocationCallbacks* allocator, VkImage* image);
void DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* allocator);
void GetImageMemoryRequirements(VkDevice device, VkImage image, VkMemoryRequirements* requirements);
VkResult AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* allocator, VkDeviceMemory* memory);
void FreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* allocator);
VkResult BindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset);
VkResult CreateImageView(VkDevice device, const VkImageViewCreateInfo* info, const VkAllocationCallbacks* allocator, VkImageView* view);
void DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* allocator);
void GetPhysicalDeviceFormatProperties(VkPhysicalDevice physical, VkFormat format, VkFormatProperties* properties);

void CmdPipelineBarrier(VkCommandBuffer commands, VkPipelineStageFlags source, VkPipelineStageFlags destination, VkDependencyFlags dependency, std::uint32_t memoryCount, const VkMemoryBarrier* memory, std::uint32_t bufferCount, const VkBufferMemoryBarrier* buffers, std::uint32_t imageCount, const VkImageMemoryBarrier* images);
void CmdCopyBufferToImage(VkCommandBuffer commands, VkBuffer source, VkImage destination, VkImageLayout layout, std::uint32_t count, const VkBufferImageCopy* regions);
void CmdCopyImageToBuffer(VkCommandBuffer commands, VkImage source, VkImageLayout layout, VkBuffer destination, std::uint32_t count, const VkBufferImageCopy* regions);
void CmdCopyImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, std::uint32_t count, const VkImageCopy* regions);
// Same-extent blits only (the attachment proxies): a copy.
void CmdBlitImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, std::uint32_t count, const VkImageBlit* regions, VkFilter filter);
// A compute kernel writes the clear value through a view of the image's format class.
void CmdClearColorImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearColorValue* color, std::uint32_t rangeCount, const VkImageSubresourceRange* ranges);
void CmdCopyBuffer(VkCommandBuffer commands, VkBuffer source, VkBuffer destination, std::uint32_t count, const VkBufferCopy* regions);

}

}

#endif
