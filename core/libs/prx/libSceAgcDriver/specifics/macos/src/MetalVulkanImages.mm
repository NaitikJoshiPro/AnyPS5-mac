// Work package D4: see MetalVulkanImages.h.
#include "MetalVulkanImages.h"
#include "MetalCommandStream.h"
#include "MetalImages.h"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"

#include <cstring>
#include <memory>
#include <vector>

#include <algorithm>
#include <atomic>
#include <span>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace AgcDriver::Graphics::Metal {

namespace {

[[noreturn]] void FailImages(const std::string& message) {
    throw std::runtime_error("Metal images (work package D4): " + message);
}

id<MTLDevice> DeviceOf(VkDevice device) {
    return (__bridge id<MTLDevice>)reinterpret_cast<void*>(device);
}

struct FormatEntry {
    VkFormat vulkan;
    MTLPixelFormat metal;
    std::uint32_t bytes;
    std::uint32_t blockWidth;
    std::uint32_t blockHeight;
};

constexpr FormatEntry Formats[] = {
    {VK_FORMAT_R8_UNORM, MTLPixelFormatR8Unorm, 1, 1, 1},
    {VK_FORMAT_R8_SNORM, MTLPixelFormatR8Snorm, 1, 1, 1},
    {VK_FORMAT_R8_UINT, MTLPixelFormatR8Uint, 1, 1, 1},
    {VK_FORMAT_R8_SINT, MTLPixelFormatR8Sint, 1, 1, 1},
    {VK_FORMAT_R8_SRGB, MTLPixelFormatR8Unorm_sRGB, 1, 1, 1},
    {VK_FORMAT_R16_UNORM, MTLPixelFormatR16Unorm, 2, 1, 1},
    {VK_FORMAT_R16_SNORM, MTLPixelFormatR16Snorm, 2, 1, 1},
    {VK_FORMAT_R16_UINT, MTLPixelFormatR16Uint, 2, 1, 1},
    {VK_FORMAT_R16_SINT, MTLPixelFormatR16Sint, 2, 1, 1},
    {VK_FORMAT_R16_SFLOAT, MTLPixelFormatR16Float, 2, 1, 1},
    {VK_FORMAT_R8G8_UNORM, MTLPixelFormatRG8Unorm, 2, 1, 1},
    {VK_FORMAT_R8G8_SNORM, MTLPixelFormatRG8Snorm, 2, 1, 1},
    {VK_FORMAT_R8G8_UINT, MTLPixelFormatRG8Uint, 2, 1, 1},
    {VK_FORMAT_R8G8_SINT, MTLPixelFormatRG8Sint, 2, 1, 1},
    {VK_FORMAT_R8G8_SRGB, MTLPixelFormatRG8Unorm_sRGB, 2, 1, 1},
    {VK_FORMAT_R5G6B5_UNORM_PACK16, MTLPixelFormatB5G6R5Unorm, 2, 1, 1},
    {VK_FORMAT_A1R5G5B5_UNORM_PACK16, MTLPixelFormatBGR5A1Unorm, 2, 1, 1},
    {VK_FORMAT_B4G4R4A4_UNORM_PACK16, MTLPixelFormatABGR4Unorm, 2, 1, 1},
    {VK_FORMAT_R32_UINT, MTLPixelFormatR32Uint, 4, 1, 1},
    {VK_FORMAT_R32_SINT, MTLPixelFormatR32Sint, 4, 1, 1},
    {VK_FORMAT_R32_SFLOAT, MTLPixelFormatR32Float, 4, 1, 1},
    {VK_FORMAT_R16G16_UNORM, MTLPixelFormatRG16Unorm, 4, 1, 1},
    {VK_FORMAT_R16G16_SNORM, MTLPixelFormatRG16Snorm, 4, 1, 1},
    {VK_FORMAT_R16G16_UINT, MTLPixelFormatRG16Uint, 4, 1, 1},
    {VK_FORMAT_R16G16_SINT, MTLPixelFormatRG16Sint, 4, 1, 1},
    {VK_FORMAT_R16G16_SFLOAT, MTLPixelFormatRG16Float, 4, 1, 1},
    {VK_FORMAT_B10G11R11_UFLOAT_PACK32, MTLPixelFormatRG11B10Float, 4, 1, 1},
    {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, MTLPixelFormatRGB9E5Float, 4, 1, 1},
    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, MTLPixelFormatRGB10A2Unorm, 4, 1, 1},
    {VK_FORMAT_A2B10G10R10_UINT_PACK32, MTLPixelFormatRGB10A2Uint, 4, 1, 1},
    {VK_FORMAT_A2R10G10B10_UNORM_PACK32, MTLPixelFormatBGR10A2Unorm, 4, 1, 1},
    {VK_FORMAT_R8G8B8A8_UNORM, MTLPixelFormatRGBA8Unorm, 4, 1, 1},
    {VK_FORMAT_R8G8B8A8_SNORM, MTLPixelFormatRGBA8Snorm, 4, 1, 1},
    {VK_FORMAT_R8G8B8A8_UINT, MTLPixelFormatRGBA8Uint, 4, 1, 1},
    {VK_FORMAT_R8G8B8A8_SINT, MTLPixelFormatRGBA8Sint, 4, 1, 1},
    {VK_FORMAT_R8G8B8A8_SRGB, MTLPixelFormatRGBA8Unorm_sRGB, 4, 1, 1},
    {VK_FORMAT_A8B8G8R8_UNORM_PACK32, MTLPixelFormatRGBA8Unorm, 4, 1, 1},
    {VK_FORMAT_A8B8G8R8_SNORM_PACK32, MTLPixelFormatRGBA8Snorm, 4, 1, 1},
    {VK_FORMAT_A8B8G8R8_UINT_PACK32, MTLPixelFormatRGBA8Uint, 4, 1, 1},
    {VK_FORMAT_A8B8G8R8_SINT_PACK32, MTLPixelFormatRGBA8Sint, 4, 1, 1},
    {VK_FORMAT_A8B8G8R8_SRGB_PACK32, MTLPixelFormatRGBA8Unorm_sRGB, 4, 1, 1},
    {VK_FORMAT_B8G8R8A8_UNORM, MTLPixelFormatBGRA8Unorm, 4, 1, 1},
    {VK_FORMAT_B8G8R8A8_SRGB, MTLPixelFormatBGRA8Unorm_sRGB, 4, 1, 1},
    {VK_FORMAT_R32G32_UINT, MTLPixelFormatRG32Uint, 8, 1, 1},
    {VK_FORMAT_R32G32_SINT, MTLPixelFormatRG32Sint, 8, 1, 1},
    {VK_FORMAT_R32G32_SFLOAT, MTLPixelFormatRG32Float, 8, 1, 1},
    {VK_FORMAT_R16G16B16A16_UNORM, MTLPixelFormatRGBA16Unorm, 8, 1, 1},
    {VK_FORMAT_R16G16B16A16_SNORM, MTLPixelFormatRGBA16Snorm, 8, 1, 1},
    {VK_FORMAT_R16G16B16A16_UINT, MTLPixelFormatRGBA16Uint, 8, 1, 1},
    {VK_FORMAT_R16G16B16A16_SINT, MTLPixelFormatRGBA16Sint, 8, 1, 1},
    {VK_FORMAT_R16G16B16A16_SFLOAT, MTLPixelFormatRGBA16Float, 8, 1, 1},
    {VK_FORMAT_R64_UINT, MTLPixelFormatRG32Uint, 8, 1, 1},
    {VK_FORMAT_R64_SINT, MTLPixelFormatRG32Uint, 8, 1, 1},
    {VK_FORMAT_R32G32B32A32_UINT, MTLPixelFormatRGBA32Uint, 16, 1, 1},
    {VK_FORMAT_R32G32B32A32_SINT, MTLPixelFormatRGBA32Sint, 16, 1, 1},
    {VK_FORMAT_R32G32B32A32_SFLOAT, MTLPixelFormatRGBA32Float, 16, 1, 1},
    {VK_FORMAT_D32_SFLOAT, MTLPixelFormatDepth32Float, 4, 1, 1},
    {VK_FORMAT_D16_UNORM, MTLPixelFormatDepth16Unorm, 2, 1, 1},
    {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, MTLPixelFormatBC1_RGBA, 8, 4, 4},
    {VK_FORMAT_BC1_RGBA_SRGB_BLOCK, MTLPixelFormatBC1_RGBA_sRGB, 8, 4, 4},
    {VK_FORMAT_BC2_UNORM_BLOCK, MTLPixelFormatBC2_RGBA, 16, 4, 4},
    {VK_FORMAT_BC2_SRGB_BLOCK, MTLPixelFormatBC2_RGBA_sRGB, 16, 4, 4},
    {VK_FORMAT_BC3_UNORM_BLOCK, MTLPixelFormatBC3_RGBA, 16, 4, 4},
    {VK_FORMAT_BC3_SRGB_BLOCK, MTLPixelFormatBC3_RGBA_sRGB, 16, 4, 4},
    {VK_FORMAT_BC4_UNORM_BLOCK, MTLPixelFormatBC4_RUnorm, 8, 4, 4},
    {VK_FORMAT_BC4_SNORM_BLOCK, MTLPixelFormatBC4_RSnorm, 8, 4, 4},
    {VK_FORMAT_BC5_UNORM_BLOCK, MTLPixelFormatBC5_RGUnorm, 16, 4, 4},
    {VK_FORMAT_BC5_SNORM_BLOCK, MTLPixelFormatBC5_RGSnorm, 16, 4, 4},
    {VK_FORMAT_BC6H_UFLOAT_BLOCK, MTLPixelFormatBC6H_RGBUfloat, 16, 4, 4},
    {VK_FORMAT_BC6H_SFLOAT_BLOCK, MTLPixelFormatBC6H_RGBFloat, 16, 4, 4},
    {VK_FORMAT_BC7_UNORM_BLOCK, MTLPixelFormatBC7_RGBAUnorm, 16, 4, 4},
    {VK_FORMAT_BC7_SRGB_BLOCK, MTLPixelFormatBC7_RGBAUnorm_sRGB, 16, 4, 4},
};

const FormatEntry* Find(VkFormat format) {
    for (const auto& entry : Formats) {
        if (entry.vulkan == format) return &entry;
    }
    return nullptr;
}

const FormatEntry* FindMetal(MTLPixelFormat format) {
    for (const auto& entry : Formats) {
        if (entry.metal == format) return &entry;
    }
    return nullptr;
}

MTLPixelFormat Required(VkFormat format) {
    const auto* entry = Find(format);
    if (entry == nullptr) FailImages("Vulkan format " + std::to_string(static_cast<int>(format)) + " has no Metal pixel format");
    return entry->metal;
}

MTLTextureSwizzle SwizzleOf(VkComponentSwizzle swizzle, MTLTextureSwizzle identity) {
    switch (swizzle) {
    case VK_COMPONENT_SWIZZLE_IDENTITY: return identity;
    case VK_COMPONENT_SWIZZLE_ZERO: return MTLTextureSwizzleZero;
    case VK_COMPONENT_SWIZZLE_ONE: return MTLTextureSwizzleOne;
    case VK_COMPONENT_SWIZZLE_R: return MTLTextureSwizzleRed;
    case VK_COMPONENT_SWIZZLE_G: return MTLTextureSwizzleGreen;
    case VK_COMPONENT_SWIZZLE_B: return MTLTextureSwizzleBlue;
    case VK_COMPONENT_SWIZZLE_A: return MTLTextureSwizzleAlpha;
    default: return identity;
    }
}

// Copy extents of a region of `texture` at `level` in texels (clipped to the level).
MTLSize RegionSize(id<MTLTexture> texture, std::uint32_t level, VkExtent3D extent) {
    const auto width = std::max<NSUInteger>(texture.width >> level, 1u);
    const auto height = std::max<NSUInteger>(texture.height >> level, 1u);
    const auto depth = texture.textureType == MTLTextureType3D ? std::max<NSUInteger>(texture.depth >> level, 1u) : 1u;
    return MTLSizeMake(std::min<NSUInteger>(extent.width, width), std::min<NSUInteger>(extent.height, height), std::min<NSUInteger>(std::max(extent.depth, 1u), depth));
}

struct ClearKernels {
    std::mutex mutex;
    std::unordered_map<void*, id<MTLLibrary>> libraries;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
};

ClearKernels& Clears() {
    static ClearKernels kernels;
    return kernels;
}

constexpr const char* ClearSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct ClearParams {
    uint4 bits;
    texture2d_array<float, access::write> f;
    texture2d_array<uint, access::write> u;
    texture2d_array<int, access::write> i;
    texture3d<float, access::write> f3;
    texture3d<uint, access::write> u3;
    texture3d<int, access::write> i3;
    uint kind;
};
kernel void clear_image(constant ClearParams& p [[buffer(0)]], uint3 id [[thread_position_in_grid]]) {
    switch (p.kind) {
    case 0u: if (id.x < p.f.get_width() && id.y < p.f.get_height()) p.f.write(as_type<float4>(p.bits), id.xy, id.z); break;
    case 1u: if (id.x < p.u.get_width() && id.y < p.u.get_height()) p.u.write(p.bits, id.xy, id.z); break;
    case 2u: if (id.x < p.i.get_width() && id.y < p.i.get_height()) p.i.write(as_type<int4>(p.bits), id.xy, id.z); break;
    case 3u: if (id.x < p.f3.get_width() && id.y < p.f3.get_height() && id.z < p.f3.get_depth()) p.f3.write(as_type<float4>(p.bits), id); break;
    case 4u: if (id.x < p.u3.get_width() && id.y < p.u3.get_height() && id.z < p.u3.get_depth()) p.u3.write(p.bits, id); break;
    case 5u: if (id.x < p.i3.get_width() && id.y < p.i3.get_height() && id.z < p.i3.get_depth()) p.i3.write(as_type<int4>(p.bits), id); break;
    }
}
)MSL";

id<MTLComputePipelineState> ClearPipeline(id<MTLDevice> device) {
    auto& kernels = Clears();
    std::lock_guard lock(kernels.mutex);
    const auto key = std::to_string(reinterpret_cast<std::uintptr_t>((__bridge void*)device));
    if (const auto found = kernels.pipelines.find(key); found != kernels.pipelines.end()) return found->second;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_1;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(ClearSource) options:options error:&error];
    if (library == nil) FailImages("the image clear kernel does not compile: " + Describe(error));
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"clear_image"] error:&error];
    if (pipeline == nil) FailImages("cannot build the image clear pipeline: " + Describe(error));
    kernels.libraries[(__bridge void*)device] = library;
    return kernels.pipelines.emplace(key, pipeline).first->second;
}

// The numeric class of a color format for the clear kernel: 0 float, 1 uint, 2 sint.
std::uint32_t NumericClass(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Uint: case MTLPixelFormatRG8Uint: case MTLPixelFormatRGBA8Uint: case MTLPixelFormatR16Uint: case MTLPixelFormatRG16Uint: case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatR32Uint: case MTLPixelFormatRG32Uint: case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGB10A2Uint:
        return 1u;
    case MTLPixelFormatR8Sint: case MTLPixelFormatRG8Sint: case MTLPixelFormatRGBA8Sint: case MTLPixelFormatR16Sint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatR32Sint: case MTLPixelFormatRG32Sint: case MTLPixelFormatRGBA32Sint:
        return 2u;
    default:
        return 0u;
    }
}

}

VkImage RetainImage(id<MTLTexture> texture) {
    return texture == nil ? VK_NULL_HANDLE : reinterpret_cast<VkImage>(const_cast<void*>(CFBridgingRetain(texture)));
}

VkImageView RetainView(id<MTLTexture> texture) {
    return texture == nil ? VK_NULL_HANDLE : reinterpret_cast<VkImageView>(const_cast<void*>(CFBridgingRetain(texture)));
}

PixelFormatInfo PixelFormatOf(VkFormat format) {
    const auto* entry = Find(format);
    if (entry == nullptr) return {};
    return {entry->metal, entry->bytes, entry->blockWidth, entry->blockHeight};
}

namespace Vk {

VkResult CreateImage(VkDevice deviceHandle, const VkImageCreateInfo* info, const VkAllocationCallbacks*, VkImage* image) {
    id<MTLDevice> device = DeviceOf(deviceHandle);
    if (device == nil) FailImages("CreateImage needs a Metal device");
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    const bool layered = info->arrayLayers > 1u || (info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0u;
    switch (info->imageType) {
    case VK_IMAGE_TYPE_1D:
    case VK_IMAGE_TYPE_2D:
        // 1D images are 2D textures one texel high (MslEntryPoint.cpp).
        if (info->samples > VK_SAMPLE_COUNT_1_BIT) descriptor.textureType = layered ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
        else descriptor.textureType = layered ? MTLTextureType2DArray : MTLTextureType2D;
        break;
    case VK_IMAGE_TYPE_3D:
        descriptor.textureType = MTLTextureType3D;
        break;
    default:
        FailImages("unknown image type");
    }
    descriptor.pixelFormat = Required(info->format);
    // Metal depth textures are 2D or cube only: a 3D "depth" image (the comparison heap's 3D
    // placeholder, which MSL declares texture3d<float>) is its color twin.
    if (info->imageType == VK_IMAGE_TYPE_3D) {
        if (descriptor.pixelFormat == MTLPixelFormatDepth16Unorm) descriptor.pixelFormat = MTLPixelFormatR16Unorm;
        if (descriptor.pixelFormat == MTLPixelFormatDepth32Float) descriptor.pixelFormat = MTLPixelFormatR32Float;
    }
    descriptor.width = info->extent.width;
    descriptor.height = std::max(info->extent.height, 1u);
    descriptor.depth = std::max(info->extent.depth, 1u);
    descriptor.arrayLength = std::max(info->arrayLayers, 1u);
    descriptor.mipmapLevelCount = std::max(info->mipLevels, 1u);
    descriptor.sampleCount = static_cast<NSUInteger>(info->samples);
    descriptor.storageMode = MTLStorageModePrivate;
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    const bool multisample = info->samples > VK_SAMPLE_COUNT_1_BIT;
    if ((info->usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0u && !multisample) usage |= MTLTextureUsageShaderWrite;
    // Clears of multisampled and depth images are render passes (CmdClear*): they need the usage.
    const bool depthFormat = info->format == VK_FORMAT_D32_SFLOAT || info->format == VK_FORMAT_D16_UNORM;
    // Images a clear may target that compute cannot write (no storage usage, multisampled, depth)
    // are cleared by a render pass: every renderable (non-compressed) image with transfer-destination use.
    if ((info->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0u && PixelFormatOf(info->format).blockWidth == 1u) usage |= MTLTextureUsageRenderTarget;
    static_cast<void>(depthFormat);
    if ((info->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) != 0u) usage |= MTLTextureUsageRenderTarget;
    // Texture atomics need the usage on a 32-bit integer image (storage images of R32 formats).
    // (Metal allows the usage only on R32Uint, R32Sint and RG32Uint textures.)
    if ((info->usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0u && !multisample && (descriptor.pixelFormat == MTLPixelFormatR32Uint || descriptor.pixelFormat == MTLPixelFormatR32Sint || descriptor.pixelFormat == MTLPixelFormatRG32Uint)) usage |= MTLTextureUsageShaderAtomic;
    descriptor.usage = usage;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (texture == nil && (usage & MTLTextureUsageShaderAtomic) != 0u) {
        descriptor.usage = usage & ~MTLTextureUsageShaderAtomic;
        texture = [device newTextureWithDescriptor:descriptor];
    }
    if (texture == nil) {
        *image = VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    if (auto* cache = ImageCache::For(device)) cache->Track(texture);
    *image = RetainImage(texture);
    return VK_SUCCESS;
}

void DestroyImage(VkDevice deviceHandle, VkImage image, const VkAllocationCallbacks*) {
    if (image == VK_NULL_HANDLE) return;
    id<MTLTexture> texture = CFBridgingRelease(reinterpret_cast<CFTypeRef>(image));
    if (auto* cache = ImageCache::For(DeviceOf(deviceHandle))) cache->Untrack(texture);
}

void GetImageMemoryRequirements(VkDevice, VkImage image, VkMemoryRequirements* requirements) {
    id<MTLTexture> texture = TextureOfImage(image);
    requirements->size = texture != nil ? texture.allocatedSize : 0u;
    requirements->alignment = 256u;
    requirements->memoryTypeBits = ~0u;
}

VkResult AllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory* memory) {
    // A token: Metal textures own their memory.
    *memory = reinterpret_cast<VkDeviceMemory>(std::uintptr_t{1});
    return VK_SUCCESS;
}

void FreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}

VkResult BindImageMemory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}

VkResult CreateImageView(VkDevice, const VkImageViewCreateInfo* info, const VkAllocationCallbacks*, VkImageView* view) {
    id<MTLTexture> texture = TextureOfImage(info->image);
    if (texture == nil) FailImages("CreateImageView of a null image");
    MTLTextureType type;
    const bool multisampled = texture.textureType == MTLTextureType2DMultisample || texture.textureType == MTLTextureType2DMultisampleArray;
    switch (info->viewType) {
    case VK_IMAGE_VIEW_TYPE_1D:
    case VK_IMAGE_VIEW_TYPE_2D: type = multisampled ? MTLTextureType2DMultisample : MTLTextureType2D; break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: type = multisampled ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray; break;
    case VK_IMAGE_VIEW_TYPE_CUBE: type = MTLTextureTypeCube; break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: type = MTLTextureTypeCubeArray; break;
    case VK_IMAGE_VIEW_TYPE_3D: type = MTLTextureType3D; break;
    default: FailImages("unknown image view type");
    }
    const auto& range = info->subresourceRange;
    const auto levels = range.levelCount == VK_REMAINING_MIP_LEVELS ? texture.mipmapLevelCount - range.baseMipLevel : range.levelCount;
    const auto layerTotal = texture.textureType == MTLTextureType3D ? 1u : texture.arrayLength;
    const auto layers = range.layerCount == VK_REMAINING_ARRAY_LAYERS ? layerTotal - range.baseArrayLayer : range.layerCount;
    const auto swizzle = MTLTextureSwizzleChannelsMake(SwizzleOf(info->components.r, MTLTextureSwizzleRed), SwizzleOf(info->components.g, MTLTextureSwizzleGreen), SwizzleOf(info->components.b, MTLTextureSwizzleBlue), SwizzleOf(info->components.a, MTLTextureSwizzleAlpha));
    auto viewFormat = Required(info->format);
    if (type == MTLTextureType3D && viewFormat == MTLPixelFormatDepth16Unorm) viewFormat = MTLPixelFormatR16Unorm;
    if (type == MTLTextureType3D && viewFormat == MTLPixelFormatDepth32Float) viewFormat = MTLPixelFormatR32Float;
    id<MTLTexture> result = [texture newTextureViewWithPixelFormat:viewFormat textureType:type levels:NSMakeRange(range.baseMipLevel, levels) slices:NSMakeRange(type == MTLTextureType3D ? 0u : range.baseArrayLayer, type == MTLTextureType3D ? 1u : layers) swizzle:swizzle];
    if (result == nil) {
        *view = VK_NULL_HANDLE;
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    *view = RetainView(result);
    return VK_SUCCESS;
}

void DestroyImageView(VkDevice, VkImageView view, const VkAllocationCallbacks*) {
    if (view != VK_NULL_HANDLE) CFRelease(reinterpret_cast<CFTypeRef>(view));
}

VkResult CreateBuffer(VkDevice deviceHandle, const VkBufferCreateInfo* info, const VkAllocationCallbacks*, VkBuffer* buffer) {
    id<MTLDevice> device = DeviceOf(deviceHandle);
    id<MTLBuffer> made = [device newBufferWithLength:std::max<VkDeviceSize>(info->size, 4u) options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
    if (made == nil) {
        *buffer = VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    if (auto* cache = ImageCache::For(device)) cache->TrackBuffer(made);
    *buffer = reinterpret_cast<VkBuffer>(const_cast<void*>(CFBridgingRetain(made)));
    return VK_SUCCESS;
}

void DestroyBuffer(VkDevice deviceHandle, VkBuffer buffer, const VkAllocationCallbacks*) {
    if (buffer == VK_NULL_HANDLE) return;
    id<MTLBuffer> released = CFBridgingRelease(reinterpret_cast<CFTypeRef>(buffer));
    if (auto* cache = ImageCache::For(DeviceOf(deviceHandle))) cache->UntrackBuffer(released);
}

void GetBufferMemoryRequirements(VkDevice, VkBuffer buffer, VkMemoryRequirements* requirements) {
    id<MTLBuffer> metal = BufferOf(buffer);
    requirements->size = metal != nil ? metal.allocatedSize : 0u;
    requirements->alignment = 256u;
    requirements->memoryTypeBits = ~0u;
}

VkResult BindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}

void GetPhysicalDeviceFormatProperties(VkPhysicalDevice, VkFormat format, VkFormatProperties* properties) {
    // Apple GPUs (Apple7+): every ordinary color format is sampled and filtered; shader writes and
    // reads (read_write, tier 2) work for the non-packed, non-sRGB, non-compressed ones; render
    // targets for the non-compressed ones (Metal Feature Set Tables).
    *properties = {};
    const auto* entry = Find(format);
    if (entry == nullptr) return;
    VkFormatFeatureFlags features = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    const bool compressed = entry->blockWidth > 1u;
    const bool depth = entry->metal == MTLPixelFormatDepth32Float || entry->metal == MTLPixelFormatDepth16Unorm;
    const bool srgb = entry->metal == MTLPixelFormatR8Unorm_sRGB || entry->metal == MTLPixelFormatRG8Unorm_sRGB || entry->metal == MTLPixelFormatRGBA8Unorm_sRGB || entry->metal == MTLPixelFormatBGRA8Unorm_sRGB;
    const bool packed16 = entry->metal == MTLPixelFormatB5G6R5Unorm || entry->metal == MTLPixelFormatBGR5A1Unorm || entry->metal == MTLPixelFormatABGR4Unorm;
    const bool sharedExponent = entry->metal == MTLPixelFormatRGB9E5Float;
    if (!compressed && !depth) features |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    if (!compressed && !depth && !srgb && !packed16 && !sharedExponent) features |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    if (depth) features |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    // Texture atomics: R32Uint/R32Sint (32-bit), RG32Uint (64-bit min/max, the R64 formats).
    if (entry->metal == MTLPixelFormatR32Uint || entry->metal == MTLPixelFormatR32Sint || entry->metal == MTLPixelFormatRG32Uint) features |= VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT;
    properties->optimalTilingFeatures = features;
}

VkResult GetPhysicalDeviceImageFormatProperties(VkPhysicalDevice, VkFormat format, VkImageType, VkImageTiling, VkImageUsageFlags, VkImageCreateFlags, VkImageFormatProperties* properties) {
    *properties = {};
    if (Find(format) == nullptr) return VK_ERROR_FORMAT_NOT_SUPPORTED;
    properties->maxExtent = {16384u, 16384u, 2048u};
    properties->maxMipLevels = 15u;
    properties->maxArrayLayers = 2048u;
    properties->sampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
    properties->maxResourceSize = VkDeviceSize{1} << 34u;
    return VK_SUCCESS;
}

namespace {

MTLSamplerMinMagFilter FilterOf(VkFilter filter) {
    return filter == VK_FILTER_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
}

MTLSamplerAddressMode AddressOf(VkSamplerAddressMode mode, VkBorderColor border, bool unnormalized) {
    switch (mode) {
    case VK_SAMPLER_ADDRESS_MODE_REPEAT: return MTLSamplerAddressModeRepeat;
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return MTLSamplerAddressModeMirrorRepeat;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: return MTLSamplerAddressModeClampToEdge;
    case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: return MTLSamplerAddressModeMirrorClampToEdge;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
        if (unnormalized) {
            if (border == VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK || border == VK_BORDER_COLOR_INT_TRANSPARENT_BLACK) return MTLSamplerAddressModeClampToZero;
            FailImages("an unnormalized sampler clamps to an opaque border color, which Metal samplers cannot do");
        }
        return MTLSamplerAddressModeClampToBorderColor;
    default:
        FailImages("unknown sampler address mode");
    }
}

MTLSamplerBorderColor BorderOf(VkBorderColor border) {
    switch (border) {
    case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
    case VK_BORDER_COLOR_INT_OPAQUE_BLACK: return MTLSamplerBorderColorOpaqueBlack;
    case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
    case VK_BORDER_COLOR_INT_OPAQUE_WHITE: return MTLSamplerBorderColorOpaqueWhite;
    default: return MTLSamplerBorderColorTransparentBlack;
    }
}

MTLCompareFunction CompareOf(VkCompareOp op) {
    switch (op) {
    case VK_COMPARE_OP_NEVER: return MTLCompareFunctionNever;
    case VK_COMPARE_OP_LESS: return MTLCompareFunctionLess;
    case VK_COMPARE_OP_EQUAL: return MTLCompareFunctionEqual;
    case VK_COMPARE_OP_LESS_OR_EQUAL: return MTLCompareFunctionLessEqual;
    case VK_COMPARE_OP_GREATER: return MTLCompareFunctionGreater;
    case VK_COMPARE_OP_NOT_EQUAL: return MTLCompareFunctionNotEqual;
    case VK_COMPARE_OP_GREATER_OR_EQUAL: return MTLCompareFunctionGreaterEqual;
    default: return MTLCompareFunctionAlways;
    }
}

}

VkResult CreateSampler(VkDevice deviceHandle, const VkSamplerCreateInfo* info, const VkAllocationCallbacks*, VkSampler* sampler) {
    id<MTLDevice> device = DeviceOf(deviceHandle);
    MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
    descriptor.supportArgumentBuffers = YES;
    const bool unnormalized = info->unnormalizedCoordinates == VK_TRUE;
    descriptor.minFilter = FilterOf(info->minFilter);
    descriptor.magFilter = FilterOf(info->magFilter);
    descriptor.mipFilter = unnormalized ? MTLSamplerMipFilterNotMipmapped : info->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    descriptor.sAddressMode = AddressOf(info->addressModeU, info->borderColor, unnormalized);
    descriptor.tAddressMode = AddressOf(info->addressModeV, info->borderColor, unnormalized);
    descriptor.rAddressMode = AddressOf(info->addressModeW, info->borderColor, unnormalized);
    descriptor.borderColor = BorderOf(info->borderColor);
    descriptor.maxAnisotropy = info->anisotropyEnable == VK_TRUE ? static_cast<NSUInteger>(std::clamp(info->maxAnisotropy, 1.0f, 16.0f)) : 1u;
    descriptor.lodMinClamp = unnormalized ? 0.0f : info->minLod;
    descriptor.lodMaxClamp = unnormalized ? 0.0f : info->maxLod;
    descriptor.lodBias = unnormalized ? 0.0f : info->mipLodBias;
    descriptor.normalizedCoordinates = unnormalized ? NO : YES;
    if (info->compareEnable == VK_TRUE) descriptor.compareFunction = CompareOf(info->compareOp);
    // A VkSamplerReductionModeCreateInfo is not applied: Metal ignores reductionMode on this GPU
    // (agc_metal_sampler_reduction); the shader reduces (PipelineSpecialization::ReductionBase).
    id<MTLSamplerState> state = [device newSamplerStateWithDescriptor:descriptor];
    if (state == nil) {
        *sampler = VK_NULL_HANDLE;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *sampler = reinterpret_cast<VkSampler>(const_cast<void*>(CFBridgingRetain(state)));
    return VK_SUCCESS;
}

void DestroySampler(VkDevice, VkSampler sampler, const VkAllocationCallbacks*) {
    if (sampler != VK_NULL_HANDLE) CFRelease(reinterpret_cast<CFTypeRef>(sampler));
}

void CmdPipelineBarrier(VkCommandBuffer commands, VkPipelineStageFlags source, VkPipelineStageFlags destination, VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {
    Stream(commands).PipelineBarrier(source, destination);
}

namespace {

std::uint32_t BytesPerRow(id<MTLTexture> texture, std::uint32_t texels, std::uint32_t& blockHeight) {
    const auto* entry = FindMetal(texture.pixelFormat);
    if (entry == nullptr) FailImages("image copy of an unknown pixel format " + std::to_string(static_cast<unsigned>(texture.pixelFormat)));
    blockHeight = entry->blockHeight;
    return (texels + entry->blockWidth - 1u) / entry->blockWidth * entry->bytes;
}

}

void CmdCopyBufferToImage(VkCommandBuffer commands, VkBuffer source, VkImage destination, VkImageLayout, std::uint32_t count, const VkBufferImageCopy* regions) {
    auto& stream = Stream(commands);
    id<MTLBuffer> buffer = BufferOf(source);
    id<MTLTexture> texture = TextureOfImage(destination);
    auto encoder = stream.Compute();
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& region = regions[index];
        const auto level = region.imageSubresource.mipLevel;
        const auto size = RegionSize(texture, level, region.imageExtent);
        std::uint32_t blockHeight = 1;
        const auto rowTexels = region.bufferRowLength != 0u ? region.bufferRowLength : static_cast<std::uint32_t>(size.width);
        const auto bytesPerRow = BytesPerRow(texture, rowTexels, blockHeight);
        const auto rows = ((region.bufferImageHeight != 0u ? region.bufferImageHeight : static_cast<std::uint32_t>(size.height)) + blockHeight - 1u) / blockHeight;
        const auto bytesPerImage = static_cast<NSUInteger>(bytesPerRow) * rows;
        for (std::uint32_t layer = 0; layer < std::max(region.imageSubresource.layerCount, 1u); ++layer) {
            [encoder copyFromBuffer:buffer sourceOffset:region.bufferOffset + layer * bytesPerImage * size.depth sourceBytesPerRow:bytesPerRow sourceBytesPerImage:bytesPerImage sourceSize:size toTexture:texture destinationSlice:region.imageSubresource.baseArrayLayer + layer destinationLevel:level destinationOrigin:MTLOriginMake(region.imageOffset.x, region.imageOffset.y, region.imageOffset.z)];
        }
    }
    stream.CountCommand();
}

void CmdCopyImageToBuffer(VkCommandBuffer commands, VkImage source, VkImageLayout, VkBuffer destination, std::uint32_t count, const VkBufferImageCopy* regions) {
    auto& stream = Stream(commands);
    id<MTLBuffer> buffer = BufferOf(destination);
    id<MTLTexture> texture = TextureOfImage(source);
    auto encoder = stream.Compute();
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& region = regions[index];
        const auto level = region.imageSubresource.mipLevel;
        const auto size = RegionSize(texture, level, region.imageExtent);
        std::uint32_t blockHeight = 1;
        const auto rowTexels = region.bufferRowLength != 0u ? region.bufferRowLength : static_cast<std::uint32_t>(size.width);
        const auto bytesPerRow = BytesPerRow(texture, rowTexels, blockHeight);
        const auto rows = ((region.bufferImageHeight != 0u ? region.bufferImageHeight : static_cast<std::uint32_t>(size.height)) + blockHeight - 1u) / blockHeight;
        const auto bytesPerImage = static_cast<NSUInteger>(bytesPerRow) * rows;
        for (std::uint32_t layer = 0; layer < std::max(region.imageSubresource.layerCount, 1u); ++layer) {
            [encoder copyFromTexture:texture sourceSlice:region.imageSubresource.baseArrayLayer + layer sourceLevel:level sourceOrigin:MTLOriginMake(region.imageOffset.x, region.imageOffset.y, region.imageOffset.z) sourceSize:size toBuffer:buffer destinationOffset:region.bufferOffset + layer * bytesPerImage * size.depth destinationBytesPerRow:bytesPerRow destinationBytesPerImage:bytesPerImage];
        }
    }
    stream.CountCommand();
}

void CmdCopyImage(VkCommandBuffer commands, VkImage source, VkImageLayout, VkImage destination, VkImageLayout, std::uint32_t count, const VkImageCopy* regions) {
    auto& stream = Stream(commands);
    id<MTLTexture> from = TextureOfImage(source);
    id<MTLTexture> to = TextureOfImage(destination);
    auto encoder = stream.Compute();
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& region = regions[index];
        const auto size = RegionSize(from, region.srcSubresource.mipLevel, region.extent);
        for (std::uint32_t layer = 0; layer < std::max(region.srcSubresource.layerCount, 1u); ++layer) {
            [encoder copyFromTexture:from sourceSlice:region.srcSubresource.baseArrayLayer + layer sourceLevel:region.srcSubresource.mipLevel sourceOrigin:MTLOriginMake(region.srcOffset.x, region.srcOffset.y, region.srcOffset.z) sourceSize:size toTexture:to destinationSlice:region.dstSubresource.baseArrayLayer + layer destinationLevel:region.dstSubresource.mipLevel destinationOrigin:MTLOriginMake(region.dstOffset.x, region.dstOffset.y, region.dstOffset.z)];
        }
    }
    stream.CountCommand();
}

void CmdBlitImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, std::uint32_t count, const VkImageBlit* regions, VkFilter) {
    std::vector<VkImageCopy> copies;
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& blit = regions[index];
        const auto width = static_cast<std::uint32_t>(blit.srcOffsets[1].x - blit.srcOffsets[0].x);
        const auto height = static_cast<std::uint32_t>(blit.srcOffsets[1].y - blit.srcOffsets[0].y);
        if (width != static_cast<std::uint32_t>(blit.dstOffsets[1].x - blit.dstOffsets[0].x) || height != static_cast<std::uint32_t>(blit.dstOffsets[1].y - blit.dstOffsets[0].y)) FailImages("scaling image blits are not implemented");
        copies.push_back({blit.srcSubresource, blit.srcOffsets[0], blit.dstSubresource, blit.dstOffsets[0], {width, height, 1u}});
    }
    CmdCopyImage(commands, source, sourceLayout, destination, destinationLayout, static_cast<std::uint32_t>(copies.size()), copies.data());
}

namespace {
void ClearByRenderPass(VkCommandBuffer commands, id<MTLTexture> texture, const VkImageSubresourceRange* ranges, std::uint32_t rangeCount, const VkClearColorValue* color, const VkClearDepthStencilValue* depth);
}

void CmdClearColorImage(VkCommandBuffer commands, VkImage image, VkImageLayout, const VkClearColorValue* color, std::uint32_t rangeCount, const VkImageSubresourceRange* ranges) {
    auto& stream = Stream(commands);
    id<MTLTexture> texture = TextureOfImage(image);
    if ((texture.usage & MTLTextureUsageShaderWrite) == 0u || texture.sampleCount > 1u) {
        ClearByRenderPass(commands, texture, ranges, rangeCount, color, nullptr);
        return;
    }
    id<MTLDevice> device = texture.device;
    const auto pipeline = ClearPipeline(device);
    const bool volume = texture.textureType == MTLTextureType3D;
    const auto numeric = NumericClass(texture.pixelFormat);
    struct Params {
        std::uint32_t bits[4];
        std::uint64_t textures[6];
        std::uint32_t kind;
        std::uint32_t pad[3];
    };
    for (std::uint32_t index = 0; index < rangeCount; ++index) {
        const auto& range = ranges[index];
        const auto levels = range.levelCount == VK_REMAINING_MIP_LEVELS ? texture.mipmapLevelCount - range.baseMipLevel : range.levelCount;
        const auto layers = volume ? 1u : (range.layerCount == VK_REMAINING_ARRAY_LAYERS ? texture.arrayLength - range.baseArrayLayer : range.layerCount);
        for (std::uint32_t level = range.baseMipLevel; level < range.baseMipLevel + levels; ++level) {
            id<MTLTexture> view = [texture newTextureViewWithPixelFormat:texture.pixelFormat textureType:(volume ? MTLTextureType3D : MTLTextureType2DArray) levels:NSMakeRange(level, 1) slices:NSMakeRange(volume ? 0u : range.baseArrayLayer, volume ? 1u : layers)];
            if (view == nil) FailImages("cannot view an image to clear it");
            Params params{};
            std::memcpy(params.bits, color->uint32, sizeof(params.bits));
            params.kind = numeric + (volume ? 3u : 0u);
            params.textures[params.kind] = view.gpuResourceID._impl;
            // The view lives until the batch that clears through it completes.
            if (auto* recorder = Recorder::Active()) {
                struct Held {
                    id<MTLTexture> view;
                };
                recorder->Keep(std::make_shared<Held>(Held{view}));
            }
            const auto width = std::max<NSUInteger>(texture.width >> level, 1u);
            const auto height = std::max<NSUInteger>(texture.height >> level, 1u);
            const auto depth = volume ? std::max<NSUInteger>(texture.depth >> level, 1u) : layers;
            stream.DispatchHelper(pipeline, std::as_bytes(std::span(&params, 1)), MTLSizeMake((width + 7u) / 8u, (height + 7u) / 8u, depth), MTLSizeMake(8, 8, 1));
        }
    }
}

namespace {

// Clears every (level, layer) of the ranges with a render pass that loads the value.
void ClearByRenderPass(VkCommandBuffer commands, id<MTLTexture> texture, const VkImageSubresourceRange* ranges, std::uint32_t rangeCount, const VkClearColorValue* color, const VkClearDepthStencilValue* depth) {
    auto& stream = Stream(commands);
    for (std::uint32_t index = 0; index < rangeCount; ++index) {
        const auto& range = ranges[index];
        const auto levels = range.levelCount == VK_REMAINING_MIP_LEVELS ? texture.mipmapLevelCount - range.baseMipLevel : range.levelCount;
        const bool volume = texture.textureType == MTLTextureType3D;
        const auto layers = volume ? 1u : (range.layerCount == VK_REMAINING_ARRAY_LAYERS ? texture.arrayLength - range.baseArrayLayer : range.layerCount);
        for (std::uint32_t level = range.baseMipLevel; level < range.baseMipLevel + levels; ++level) {
            const auto slices = volume ? std::max<NSUInteger>(texture.depth >> level, 1u) : layers;
            for (NSUInteger slice = 0; slice < slices; ++slice) {
                MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
                const bool depthTexture = texture.pixelFormat == MTLPixelFormatDepth32Float || texture.pixelFormat == MTLPixelFormatDepth16Unorm;
                VkClearColorValue depthAsColor{};
                if (depth != nullptr && !depthTexture) {
                    // A depth clear of a depth image's color twin (3D, CreateImage).
                    depthAsColor.float32[0] = depth->depth;
                    color = &depthAsColor;
                }
                if (depth != nullptr && depthTexture) {
                    pass.depthAttachment.texture = texture;
                    pass.depthAttachment.level = level;
                    pass.depthAttachment.slice = volume ? 0u : range.baseArrayLayer + slice;
                    pass.depthAttachment.loadAction = MTLLoadActionClear;
                    pass.depthAttachment.storeAction = MTLStoreActionStore;
                    pass.depthAttachment.clearDepth = depth->depth;
                } else {
                    pass.colorAttachments[0].texture = texture;
                    pass.colorAttachments[0].level = level;
                    if (volume) pass.colorAttachments[0].depthPlane = slice;
                    else pass.colorAttachments[0].slice = range.baseArrayLayer + slice;
                    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                    const auto numeric = NumericClass(texture.pixelFormat);
                    if (numeric == 1u) pass.colorAttachments[0].clearColor = MTLClearColorMake(color->uint32[0], color->uint32[1], color->uint32[2], color->uint32[3]);
                    else if (numeric == 2u) pass.colorAttachments[0].clearColor = MTLClearColorMake(color->int32[0], color->int32[1], color->int32[2], color->int32[3]);
                    else pass.colorAttachments[0].clearColor = MTLClearColorMake(color->float32[0], color->float32[1], color->float32[2], color->float32[3]);
                }
                pass.renderTargetWidth = std::max<NSUInteger>(texture.width >> level, 1u);
                pass.renderTargetHeight = std::max<NSUInteger>(texture.height >> level, 1u);
                stream.BeginRender(pass);
                stream.EndEncoder();
            }
        }
    }
}

}

void CmdClearDepthStencilImage(VkCommandBuffer commands, VkImage image, VkImageLayout, const VkClearDepthStencilValue* value, std::uint32_t rangeCount, const VkImageSubresourceRange* ranges) {
    ClearByRenderPass(commands, TextureOfImage(image), ranges, rangeCount, nullptr, value);
}

void CmdUpdateBuffer(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes, const void* data) {
    Stream(commands).UpdateBuffer(BufferOf(buffer), offset, std::span<const std::byte>(static_cast<const std::byte*>(data), bytes));
}

namespace {

// Distinct non-null tokens for the descriptor objects (never dereferenced).
template <typename Handle>
Handle Token() {
    static std::atomic<std::uintptr_t> next{0x1000u};
    return reinterpret_cast<Handle>(next.fetch_add(16u, std::memory_order_relaxed));
}

}

VkResult CreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout* layout) {
    *layout = Token<VkDescriptorSetLayout>();
    return VK_SUCCESS;
}

void DestroyDescriptorSetLayout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {}

VkResult CreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool* pool) {
    *pool = Token<VkDescriptorPool>();
    return VK_SUCCESS;
}

void DestroyDescriptorPool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) {}

VkResult ResetDescriptorPool(VkDevice, VkDescriptorPool, VkDescriptorPoolResetFlags) {
    return VK_SUCCESS;
}

VkResult AllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* sets) {
    for (std::uint32_t index = 0; index < info->descriptorSetCount; ++index) sets[index] = Token<VkDescriptorSet>();
    return VK_SUCCESS;
}

void UpdateDescriptorSets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t, const VkCopyDescriptorSet*) {}

void CmdBindDescriptorSets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t, std::uint32_t, const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {}

void CmdCopyBuffer(VkCommandBuffer commands, VkBuffer source, VkBuffer destination, std::uint32_t count, const VkBufferCopy* regions) {
    auto& stream = Stream(commands);
    for (std::uint32_t index = 0; index < count; ++index) stream.CopyBuffer(BufferOf(source), regions[index].srcOffset, BufferOf(destination), regions[index].dstOffset, regions[index].size);
}

}

}
