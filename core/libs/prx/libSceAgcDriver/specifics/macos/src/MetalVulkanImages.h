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
#include <memory>
#include <span>

namespace AgcDriver::Graphics {
struct Context;
class Texture;
}

namespace AgcDriver::Graphics::Metal {

// The texture a VkImage or VkImageView carries (unretained), nil for a null handle.
inline id<MTLTexture> TextureOfImage(VkImage image) {
    return (__bridge id<MTLTexture>)reinterpret_cast<void*>(image);
}
inline id<MTLTexture> TextureOfView(VkImageView view) {
    return (__bridge id<MTLTexture>)reinterpret_cast<void*>(view);
}
inline id<MTLSamplerState> SamplerOf(VkSampler sampler) {
    return (__bridge id<MTLSamplerState>)reinterpret_cast<void*>(sampler);
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
// The memory handle AllocateMemory returns: a token (a Metal texture made by CreateImage owns its
// memory). Image memory from AllocateImageMemory (BufferPool.mm, upstream 4513d47104) is instead a
// retained placement id<MTLHeap>.
inline bool IsMemoryToken(VkDeviceMemory memory) {
    return reinterpret_cast<std::uintptr_t>(memory) == 1u;
}
// The placement-heap bytes and alignment an image of `info` takes (heapTextureSizeAndAlign), for
// AllocateImageMemory before CreateImageInMemory.
void GetImageCreateRequirements(VkDevice device, const VkImageCreateInfo* info, VkMemoryRequirements* requirements);
// CreateImage with the texture placed at offset 0 of `memory` (a heap from AllocateImageMemory of
// GetImageCreateRequirements' size). The texture is not added to the image residency set: the heap
// is resident while it lives. DestroyImage releases the texture; the memory goes back separately.
VkResult CreateImageInMemory(VkDevice device, const VkImageCreateInfo* info, VkDeviceMemory memory, VkImage* image);
VkResult CreateImageView(VkDevice device, const VkImageViewCreateInfo* info, const VkAllocationCallbacks* allocator, VkImageView* view);
void DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* allocator);
// A VkBuffer made here carries a CF-retained private id<MTLBuffer> (unit shadow slabs), resident
// through the image cache until DestroyBuffer; BufferOf() reads it like D2's buffers.
VkResult CreateBuffer(VkDevice device, const VkBufferCreateInfo* info, const VkAllocationCallbacks* allocator, VkBuffer* buffer);
void DestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator);
void GetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements);
VkResult BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset);
void GetPhysicalDeviceFormatProperties(VkPhysicalDevice physical, VkFormat format, VkFormatProperties* properties);
VkResult GetPhysicalDeviceImageFormatProperties(VkPhysicalDevice physical, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties* properties);
// A VkSampler carries a CF-retained id<MTLSamplerState> (argument-buffer capable).
VkResult CreateSampler(VkDevice device, const VkSamplerCreateInfo* info, const VkAllocationCallbacks* allocator, VkSampler* sampler);
void DestroySampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* allocator);

void CmdPipelineBarrier(VkCommandBuffer commands, VkPipelineStageFlags source, VkPipelineStageFlags destination, VkDependencyFlags dependency, std::uint32_t memoryCount, const VkMemoryBarrier* memory, std::uint32_t bufferCount, const VkBufferMemoryBarrier* buffers, std::uint32_t imageCount, const VkImageMemoryBarrier* images);
void CmdCopyBufferToImage(VkCommandBuffer commands, VkBuffer source, VkImage destination, VkImageLayout layout, std::uint32_t count, const VkBufferImageCopy* regions);
void CmdCopyImageToBuffer(VkCommandBuffer commands, VkImage source, VkImageLayout layout, VkBuffer destination, std::uint32_t count, const VkBufferImageCopy* regions);
void CmdCopyImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, std::uint32_t count, const VkImageCopy* regions);
// Same-extent blits only (the attachment proxies): a copy.
void CmdBlitImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, std::uint32_t count, const VkImageBlit* regions, VkFilter filter);
// A compute kernel writes the clear value through a view of the image's format class.
void CmdClearColorImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearColorValue* color, std::uint32_t rangeCount, const VkImageSubresourceRange* ranges);
void CmdCopyBuffer(VkCommandBuffer commands, VkBuffer source, VkBuffer destination, std::uint32_t count, const VkBufferCopy* regions);
void CmdUpdateBuffer(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes, const void* data);

// Descriptor sets do not exist on Metal: the twins keep their bookkeeping, the handles are tokens,
// and the writes a ShaderResources collects become MslAbi descriptor-table entries instead
// (ShaderResources::WriteTableEntries). Binding a set records nothing.
VkResult CreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* info, const VkAllocationCallbacks* allocator, VkDescriptorSetLayout* layout);
void DestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout layout, const VkAllocationCallbacks* allocator);
VkResult CreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* info, const VkAllocationCallbacks* allocator, VkDescriptorPool* pool);
void DestroyDescriptorPool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks* allocator);
VkResult ResetDescriptorPool(VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags flags);
VkResult AllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* sets);
void UpdateDescriptorSets(VkDevice device, std::uint32_t writeCount, const VkWriteDescriptorSet* writes, std::uint32_t copyCount, const VkCopyDescriptorSet* copies);
void CmdBindDescriptorSets(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout, std::uint32_t firstSet, std::uint32_t setCount, const VkDescriptorSet* sets, std::uint32_t dynamicOffsetCount, const std::uint32_t* dynamicOffsets);
// A render pass that loads the clear value (depth images, multisampled images).
void CmdClearDepthStencilImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearDepthStencilValue* value, std::uint32_t rangeCount, const VkImageSubresourceRange* ranges);

}

// Test seam, the twin of upstream's device-function interception (tests/Recorder.cpp
// StorageUploadProbe, which swaps Context.deviceProc): while a probe is set, CreateImage,
// CreateImageView, DestroyImage and CmdClearColorImage report to it. Only tests set one
// (specifics/macos/tests/StorageUploadTwin.mm); one probe at a time.
class ImageProbe {
public:
    virtual ~ImageProbe() = default;
    virtual void Created(VkImage) {}
    // True: CreateImageView fails with VK_ERROR_OUT_OF_DEVICE_MEMORY (failure injection).
    virtual bool FailView(const VkImageViewCreateInfo&) { return false; }
    // True: the probe keeps the image's reference and destroys it later itself (with no probe set).
    virtual bool Destroying(VkImage) { return false; }
    virtual void Cleared(VkImage) {}
    // Image-to-buffer and buffer-to-image copies (CmdCopyImageToBuffer, CmdCopyBufferToImage and the depth
    // plane copies of DepthSurface.mm, NoteImageCopy): the twin of counting vkCmdCopy* through deviceProc.
    virtual void CopiedToBuffer(VkImage) {}
    virtual void CopiedFromBuffer(VkImage) {}
};
// Reports an image copy recorded outside CmdCopyImageToBuffer/CmdCopyBufferToImage to the probe.
void NoteImageCopy(VkImage image, bool toBuffer);
// Sets (or, with nullptr, clears) the probe.
void SetImageProbe(ImageProbe* probe);

// Guest MSAA T#s (types 14 and 15, DECISIONS D27): whether `words` (a T#) is one.
inline bool MultisampledDescriptor(std::span<const std::uint32_t> words) {
    if (words.size() != 8u) return false;
    const auto type = (words[3] >> 28u) & 0xfu;
    return type == 14u || type == 15u;
}

// How a guest sample count renders on `device` (DECISIONS D28): natively (slices 1), or, when the
// device cannot render that many samples (8x on Apple9, which renders 1, 2 and 4), as `slices`
// slices of `samples` samples each of a 2D multisample array texture, slice k holding guest samples
// k * samples .. k * samples + samples - 1 (each slice rendered by its own pass with those samples'
// positions). APS5_MSAA_HOST_SAMPLES (tests) lowers the most samples a slice may hold, so a 4x
// target can be checked against its own 2 x 2x emulation. Throws when no split exists.
struct HostSamples {
    std::uint32_t samples = 1;
    std::uint32_t slices = 1;
};
HostSamples HostSamplesOf(id<MTLDevice> device, std::uint32_t guestSamples);
// The cost of the emulation, cumulative (MetalDraw.mm): draws recorded over several slices, the
// slice passes they recorded and how many of those began a render pass (the others continued the
// open one).
struct EmulatedSampleStatistics {
    std::uint64_t draws = 0;
    std::uint64_t slicePasses = 0;
    std::uint64_t begun = 0;
};
EmulatedSampleStatistics EmulatedSampleCounts();
// Index expansions (MetalDraw.mm, D81), cumulative: gathered on the GPU from indices recorded
// work writes, and read on the CPU after a recorder sync because recorded work wrote them.
struct IndexExpansionStatistics {
    std::uint64_t gathered = 0;
    std::uint64_t settled = 0;
};
IndexExpansionStatistics IndexExpansionCounts();

// A CB resolve (DECISIONS D27): every pixel of `destination` (single-sample, the same pixel format
// and extent) becomes the box-filtered average of `source`'s samples, through a render pass whose
// store action is MTLStoreActionMultisampleResolve; integer formats, which Metal does not resolve,
// take sample 0 (VK_RESOLVE_MODE_SAMPLE_ZERO) through a fragment pass that reads the
// texture2d_ms. Recorded into `commands`' stream; returns true when the sample-0 pass was used.
// A 2D multisample array source is an emulated sample count (HostSamplesOf, DECISIONS D28): the
// average of every sample of every slice (integer formats: sample 0 of slice 0), by a fragment pass.
bool ResolveMultisample(VkCommandBuffer commands, id<MTLTexture> source, id<MTLTexture> destination);

// The storage shadow of a multisampled image (DECISIONS D22, D28): Metal shaders cannot write
// multisampled textures, so guest storage access goes to a single-sample 2D array of layers * n
// slices (guest layer l, sample s at slice l * n + s, MslImage.cpp EmulatedMultisampleCoord). The
// multisampled side is a 2D multisample array holding guest layer l, sample s at slice l * p + s / h,
// sample s % h (h its sample count, p = n / h slices per layer, HostSamplesOf). These copy every
// sample one way: a compute kernel to the shadow; render passes (one per slice, per-sample fragment
// functions) back, since compute cannot write multisampled textures. Recorded into `commands`.
void CopyMultisampleToStorage(VkCommandBuffer commands, id<MTLTexture> samples, id<MTLTexture> storage, std::uint32_t guestSamples);
void CopyStorageToMultisample(VkCommandBuffer commands, id<MTLTexture> storage, id<MTLTexture> samples, std::uint32_t guestSamples);

}

namespace AgcDriver::Graphics::Metal {

// The sampled view of a guest MSAA T# (DECISIONS D27): the host image of the multisampled color
// target the T# addresses (MultisampleTarget.mm; samples exist only there), with the T#'s swizzle,
// after its pending CMASK fast clear. Throws when no multisampled target of the T#'s address, extent,
// sample count and format exists (guest-written MSAA memory has no modeled sample layout).
std::shared_ptr<Texture> MultisampleSampledTexture(const Context& context, std::span<const std::uint32_t> words);
// The storage view of a guest MSAA T# (DECISIONS D22, D28): the target's storage shadow (a
// single-sample 2D array of n slices, made on first use), brought up to date from the host image;
// the host image takes the shadow's samples back before its next use (draw, sampled view, resolve).
std::shared_ptr<Texture> MultisampleStorageTexture(const Context& context, std::span<const std::uint32_t> words);
// The copies the target at `address` (on any device) made so far (tests).
struct MultisampleShadowStatistics {
    std::uint64_t hostToShadow = 0;
    std::uint64_t shadowToHost = 0;
};
MultisampleShadowStatistics MultisampleShadowCounts(std::uint64_t address);

}

#endif
