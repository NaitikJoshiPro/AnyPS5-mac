// Work packages D4 (textures) and D5 (image/sampler descriptors): see MetalImages.h.
#include "MetalImages.h"
#include "prx/libSceAgcDriver/Graphics/include/GuestSamplerResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "RuntimeAbi.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics::Metal {

namespace {

[[noreturn]] void Fail(const std::string& message) {
    throw std::runtime_error("Metal textures (work package D4): " + message);
}

[[noreturn]] void FailD5(const std::string& message) {
    throw std::runtime_error("Metal descriptors (work package D5): " + message);
}

bool Trace() {
    static const bool enabled = std::getenv("APS5_METAL_TRACE") != nullptr;
    return enabled;
}

constexpr std::uint32_t HeapDimensions = 7u;

// RuntimeAbi image heap (binding 1-56): class (0-2 sampled float/uint/sint, 3 comparison, 4 storage
// float, 5 storage uint, 6 atomic uint, 7 atomic64) and dimension index (1D, 1DArray, 2D, 2DArray,
// 2DMS, 2DMSArray, 3D).
struct Heap {
    std::uint32_t cls;
    std::uint32_t dimension;
};

Heap HeapOf(std::uint32_t heap) {
    if (heap < ShaderRecompiler::RuntimeAbi::FirstImageBinding || heap >= static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers)) FailD5("binding " + std::to_string(heap) + " is not an image heap");
    const auto index = heap - ShaderRecompiler::RuntimeAbi::FirstImageBinding;
    return {index / HeapDimensions, index % HeapDimensions};
}

MTLTextureType HeapTextureType(std::uint32_t dimension) {
    static constexpr MTLTextureType Types[HeapDimensions] = {MTLTextureType2D, MTLTextureType2DArray, MTLTextureType2D, MTLTextureType2DArray, MTLTextureType2DMultisample, MTLTextureType2DMultisampleArray, MTLTextureType3D};
    return Types[dimension];
}

MTLTextureSwizzle SwizzleOf(std::uint32_t dstSel, bool swapRedBlue) {
    switch (dstSel) {
    case 0: return MTLTextureSwizzleZero;
    case 1: return MTLTextureSwizzleOne;
    case 4: return swapRedBlue ? MTLTextureSwizzleBlue : MTLTextureSwizzleRed;
    case 5: return MTLTextureSwizzleGreen;
    case 6: return swapRedBlue ? MTLTextureSwizzleRed : MTLTextureSwizzleBlue;
    case 7: return MTLTextureSwizzleAlpha;
    default: Fail("guest texture descriptor has an invalid destination channel selector " + std::to_string(dstSel));
    }
}

// ViewComponents (Graphics/src/ShaderResources.cpp): converted formats are read raw; otherwise the
// descriptor's dst_sel through the format's channel order (TextureComponentChannel).
MTLTextureSwizzleChannels ViewSwizzle(const GuestTextureResource& resource, const TextureFormatInfo& format) {
    if (IsConvertedTextureFormat(resource.format)) return MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
    const auto channel = [&](std::uint8_t dstSel) {
        if (dstSel < 4u) return SwizzleOf(dstSel, false);
        // TextureComponentChannel maps R/G/B/A of formats whose components the Vulkan format orders
        // differently (BGR packed formats); apply it on the Vulkan enums, then convert.
        const auto vulkan = TextureComponentChannel(resource.format, static_cast<VkComponentSwizzle>(VK_COMPONENT_SWIZZLE_R + (dstSel - 4u)));
        return SwizzleOf(4u + static_cast<std::uint32_t>(vulkan - VK_COMPONENT_SWIZZLE_R), format.swapRedBlue);
    };
    return MTLTextureSwizzleChannelsMake(channel(resource.dstSelX), channel(resource.dstSelY), channel(resource.dstSelZ), channel(resource.dstSelW));
}

std::uint64_t ResourceId(id<MTLTexture> texture) {
    return texture.gpuResourceID._impl;
}

std::uint64_t ResourceId(id<MTLSamplerState> sampler) {
    return sampler.gpuResourceID._impl;
}

bool AllZero(std::span<const std::uint32_t> words) {
    return words[0] == 0u && (words[1] & 0xffu) == 0u;
}

MTLTextureUsage UsageFor(bool storage, bool atomic) {
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    if (storage) usage |= MTLTextureUsageShaderWrite;
    if (atomic) usage |= MTLTextureUsageShaderAtomic;
    return usage;
}

// Each (layer, level) of the surface, as the upload and write-back loops walk it.
template <typename Visit>
void ForEachLevel(const GuestSurface& surface, Visit&& visit) {
    const auto& geometry = surface.geometry;
    for (std::uint32_t layer = 0; layer < geometry.layers; ++layer) {
        for (std::uint32_t level = 0; level < geometry.mips.size() && level < surface.resource.mipCount; ++level) {
            if (!geometry.HasLayer(level, layer)) continue;
            visit(layer, level);
        }
    }
}

MTLRegion LevelRegion(const GuestSurface& surface, std::uint32_t layer, std::uint32_t level) {
    const auto width = std::max(surface.resource.width >> level, 1u);
    const auto height = std::max(surface.resource.height >> level, 1u);
    return MTLRegionMake3D(0, 0, static_cast<NSUInteger>(surface.geometry.CopyDepth(layer)), width, height, 1);
}

DetileLevel LevelOf(const GuestSurface& surface, std::uint32_t layer, std::uint32_t level) {
    return {surface.resource.tileMode, surface.format.elementBytes, &surface.geometry.mips[level], layer, surface.geometry.thick, surface.resource.pipeBankXor};
}

std::uint64_t LinearPitch(const GuestSurface& surface, std::uint32_t level) {
    return static_cast<std::uint64_t>(surface.geometry.mips[level].width) * surface.format.textureElementBytes;
}

void Upload(GuestSurface& surface) {
    const auto& geometry = surface.geometry;
    const auto* guest = reinterpret_cast<const std::byte*>(surface.resource.baseAddress);
    std::vector<std::byte> linear;
    ForEachLevel(surface, [&](std::uint32_t layer, std::uint32_t level) {
        const auto& mip = geometry.mips[level];
        const auto pitch = LinearPitch(surface, level);
        linear.assign(pitch * mip.height, std::byte{0});
        const auto offset = geometry.GuestLayerOffset(layer) + mip.tiledOffset;
        if (offset >= geometry.guestBytes) return;
        DetileCpu(LevelOf(surface, layer, level), guest + offset, geometry.guestBytes - offset, linear.data(), pitch, surface.format.textureElementBytes);
        const auto region = LevelRegion(surface, layer, level);
        [surface.texture replaceRegion:region mipmapLevel:level slice:geometry.CopyLayer(layer) withBytes:linear.data() bytesPerRow:pitch bytesPerImage:0];
    });
}

void WriteBack(const GuestSurface& surface, std::uint32_t firstLevel, std::uint32_t levels) {
    const auto& geometry = surface.geometry;
    auto* guest = reinterpret_cast<std::byte*>(surface.resource.baseAddress);
    std::vector<std::byte> linear;
    ForEachLevel(surface, [&](std::uint32_t layer, std::uint32_t level) {
        if (level < firstLevel || level >= firstLevel + levels) return;
        const auto& mip = geometry.mips[level];
        const auto pitch = LinearPitch(surface, level);
        linear.assign(pitch * mip.height, std::byte{0});
        const auto region = LevelRegion(surface, layer, level);
        [surface.texture getBytes:linear.data() bytesPerRow:pitch bytesPerImage:0 fromRegion:region mipmapLevel:level slice:geometry.CopyLayer(layer)];
        const auto offset = geometry.GuestLayerOffset(layer) + mip.tiledOffset;
        if (offset >= geometry.guestBytes) return;
        RetileCpu(LevelOf(surface, layer, level), guest + offset, geometry.guestBytes - offset, linear.data(), pitch, surface.format.textureElementBytes);
    });
}

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
        // Unnormalized samplers take only clamp-to-edge or clamp-to-zero.
        if (unnormalized) {
            if (border == VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK || border == VK_BORDER_COLOR_INT_TRANSPARENT_BLACK) return MTLSamplerAddressModeClampToZero;
            FailD5("an unnormalized sampler clamps to an opaque border color, which Metal samplers cannot do");
        }
        return MTLSamplerAddressModeClampToBorderColor;
    default: FailD5("unknown sampler address mode");
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

// --- ImageCache ----------------------------------------------------------------------------------

namespace {

struct CacheRegistry {
    std::mutex mutex;
    std::vector<std::pair<void*, ImageCache*>> caches;
};

CacheRegistry& Caches() {
    static CacheRegistry registry;
    return registry;
}

}

ImageCache* ImageCache::For(id<MTLDevice> device) {
    if (device == nil) return nullptr;
    auto& registry = Caches();
    std::lock_guard lock(registry.mutex);
    for (const auto& [key, cache] : registry.caches) {
        if (key == (__bridge void*)device) return cache;
    }
    return nullptr;
}

void ImageCache::Track(id<MTLTexture> texture) {
    const std::array<id<MTLTexture>, 1> one{texture};
    MakeResident(one);
}

void ImageCache::Untrack(id<MTLTexture> texture) {
    const std::array<id<MTLTexture>, 1> one{texture};
    Release(one);
}

ImageCache::ImageCache(id<MTLDevice> device, id<MTL4CommandQueue> queue) : device(device), queue(queue) {
    {
        auto& registry = Caches();
        std::lock_guard lock(registry.mutex);
        registry.caches.emplace_back((__bridge void*)device, this);
    }
    MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
    descriptor.label = @"aps5 images";
    NSError* error = nil;
    residency = [device newResidencySetWithDescriptor:descriptor error:&error];
    if (residency == nil) Fail("cannot create the image residency set");
    if (queue != nil) [queue addResidencySet:residency];
}

ImageCache::~ImageCache() {
    Teardown();
}

void ImageCache::Attach(const Context& deviceContext) {
    if (imageContext != nullptr) return;
    imageContext = std::make_unique<Context>(deviceContext);
    auto& context = *imageContext;
    detiler = std::make_unique<TextureDetiler>(context);
    context.detiler = detiler.get();
    context.textureCompressionBC = [device supportsBCTextureCompression];
    context.srgbDecodeFormats = SrgbDecodeFormats();
    context.nullDescriptors = false;
    context.descriptorCache = nullptr;
    context.descriptorIndexing = false;
    context.imageViewMinLod = false;
    // The limits a ShaderResources build checks (Metal argument tables hold the heaps; the MslAbi
    // capacities bound them).
    auto& limits = context.limits;
    limits.maxBoundDescriptorSets = std::max(limits.maxBoundDescriptorSets, 4u);
    limits.maxPerStageResources = std::max(limits.maxPerStageResources, 1u << 20u);
    limits.maxPerStageDescriptorStorageBuffers = std::max(limits.maxPerStageDescriptorStorageBuffers, 1u << 20u);
    limits.maxPerStageDescriptorSampledImages = std::max(limits.maxPerStageDescriptorSampledImages, 1u << 20u);
    limits.maxPerStageDescriptorStorageImages = std::max(limits.maxPerStageDescriptorStorageImages, 1u << 20u);
    limits.maxPerStageDescriptorSamplers = std::max(limits.maxPerStageDescriptorSamplers, 1u << 20u);
    limits.maxDescriptorSetStorageBuffers = std::max(limits.maxDescriptorSetStorageBuffers, 1u << 20u);
    limits.maxDescriptorSetSampledImages = std::max(limits.maxDescriptorSetSampledImages, 1u << 20u);
    limits.maxDescriptorSetStorageImages = std::max(limits.maxDescriptorSetStorageImages, 1u << 20u);
    limits.maxDescriptorSetSamplers = std::max(limits.maxDescriptorSetSamplers, 1u << 20u);
    limits.maxSamplerAnisotropy = std::max(limits.maxSamplerAnisotropy, 16.0f);
    limits.maxSamplerLodBias = std::max(limits.maxSamplerLodBias, 16.0f);
    limits.maxImageDimension2D = std::max(limits.maxImageDimension2D, 16384u);
    limits.maxImageDimension3D = std::max(limits.maxImageDimension3D, 2048u);
    limits.maxImageArrayLayers = std::max(limits.maxImageArrayLayers, 2048u);
    context.samplerAnisotropy = true;
    context.samplerFilterMinmax = true;
    placeholderImages = std::make_unique<PlaceholderImages>(context);
    context.placeholderImages = placeholderImages.get();
    upstreamSamplers = std::make_unique<SamplerCache>();
    context.samplerCache = upstreamSamplers.get();
}

void ImageCache::Teardown() noexcept {
    if (imageContext != nullptr) {
        // Every ShaderResources and cached texture of the device goes while the image cache's
        // residency set is still on the queue (the recorder is idle: the device tore it down).
        try {
            SharedResourceCache().Clear();
            ClearCachedTextures(imageContext->device);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[metal-textures] teardown: %s\n", error.what());
        }
        upstreamSamplers.reset();
        placeholderImages.reset();
        detiler.reset();
        imageContext.reset();
    }
    {
        auto& registry = Caches();
        std::lock_guard lock(registry.mutex);
        std::erase_if(registry.caches, [&](const auto& entry) { return entry.second == this; });
    }
    std::lock_guard lock(mutex);
    if (tornDown) return;
    tornDown = true;
    if (queue != nil && residency != nil) [queue removeResidencySet:residency];
    [residency removeAllAllocations];
    [residency commit];
    residentCounts.clear();
    placeholders.clear();
    samplerCache.clear();
}

void ImageCache::MakeResident(std::span<const id<MTLTexture>> textures) {
    std::lock_guard lock(mutex);
    bool changed = false;
    for (id<MTLTexture> texture : textures) {
        if (texture == nil) continue;
        auto& count = residentCounts[(__bridge void*)texture];
        if (count++ == 0u) {
            [residency addAllocation:texture];
            changed = true;
        }
    }
    if (changed) [residency commit];
}

void ImageCache::Release(std::span<const id<MTLTexture>> textures) {
    std::lock_guard lock(mutex);
    if (tornDown) return;
    bool changed = false;
    for (id<MTLTexture> texture : textures) {
        if (texture == nil) continue;
        const auto found = residentCounts.find((__bridge void*)texture);
        if (found == residentCounts.end()) continue;
        if (--found->second == 0u) {
            [residency removeAllocation:texture];
            residentCounts.erase(found);
            changed = true;
        }
    }
    if (changed) [residency commit];
}

ImageCache::Statistics ImageCache::Stats() const {
    std::lock_guard lock(mutex);
    return statistics;
}

id<MTLSamplerState> ImageCache::Sampler(std::span<const std::uint32_t> words, bool compare, bool unnormalized, bool integerBorder) {
    if (words.size() != 4u) FailD5("a guest sampler descriptor must contain 4 dwords");
    const std::array<std::uint32_t, 5> key{words[0], words[1], words[2], words[3], (compare ? 1u : 0u) | (unnormalized ? 2u : 0u) | (integerBorder ? 4u : 0u)};
    {
        std::lock_guard lock(mutex);
        if (const auto found = samplerCache.find(key); found != samplerCache.end()) {
            ++statistics.samplerHits;
            return found->second;
        }
    }
    // FORCE_DEGAMMA pairs only with sRGB textures (ImageBindings::Prepare checks), whose views decode.
    const auto decoded = DecodeSamplerResource(words, unnormalized, true, integerBorder);
    if (decoded.unnormalizedCoordinates && compare) FailD5("a sampler with unnormalized coordinates enables depth comparison, which is not implemented");
    MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
    descriptor.supportArgumentBuffers = YES;
    descriptor.minFilter = FilterOf(decoded.minFilter);
    descriptor.magFilter = FilterOf(decoded.magFilter);
    descriptor.mipFilter = decoded.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    descriptor.sAddressMode = AddressOf(decoded.addressModeU, decoded.borderColor, decoded.unnormalizedCoordinates);
    descriptor.tAddressMode = AddressOf(decoded.addressModeV, decoded.borderColor, decoded.unnormalizedCoordinates);
    descriptor.rAddressMode = AddressOf(decoded.addressModeW, decoded.borderColor, decoded.unnormalizedCoordinates);
    descriptor.borderColor = BorderOf(decoded.borderColor);
    descriptor.maxAnisotropy = decoded.anisotropyEnable ? static_cast<NSUInteger>(std::clamp(decoded.maxAnisotropy, 1.0f, 16.0f)) : 1u;
    descriptor.lodMinClamp = decoded.minLod;
    descriptor.lodMaxClamp = decoded.maxLod;
    descriptor.lodBias = decoded.lodBias;
    // Min/max reduction: MTLSamplerDescriptor.reductionMode is ignored on this GPU
    // (agc_metal_sampler_reduction) and some combinations fail Metal's descriptor validation, so the
    // sampler filters normally and the shader reduces the footprint it reads through it
    // (PipelineSpecialization::ReductionBase, MslImage.cpp aps5_reduce_sample).
    if (compare) descriptor.compareFunction = CompareOf(decoded.compareOp);
    if (decoded.unnormalizedCoordinates) {
        descriptor.normalizedCoordinates = NO;
        descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
        descriptor.lodMinClamp = 0.0f;
        descriptor.lodMaxClamp = 0.0f;
        descriptor.lodBias = 0.0f;
    }
    id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:descriptor];
    if (sampler == nil) FailD5("Metal refused a sampler state");
    std::lock_guard lock(mutex);
    ++statistics.samplers;
    return samplerCache.emplace(key, sampler).first->second;
}

id<MTLTexture> ImageCache::Placeholder(std::uint32_t heapNumber) {
    const auto heap = HeapOf(heapNumber);
    {
        std::lock_guard lock(mutex);
        if (const auto found = placeholders.find(heapNumber); found != placeholders.end()) return found->second;
    }
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = HeapTextureType(heap.dimension);
    descriptor.width = 1;
    descriptor.height = 1;
    descriptor.depth = 1;
    descriptor.arrayLength = 1;
    descriptor.mipmapLevelCount = 1;
    descriptor.sampleCount = heap.dimension == 4u || heap.dimension == 5u ? 4u : 1u;
    descriptor.storageMode = MTLStorageModePrivate;
    static constexpr MTLPixelFormat Formats[8] = {MTLPixelFormatRGBA32Float, MTLPixelFormatRGBA32Uint, MTLPixelFormatRGBA32Sint, MTLPixelFormatDepth32Float, MTLPixelFormatRGBA32Float, MTLPixelFormatR32Uint, MTLPixelFormatR32Uint, MTLPixelFormatRG32Uint};
    descriptor.pixelFormat = Formats[heap.cls];
    descriptor.usage = MTLTextureUsageShaderRead | (heap.cls >= 4u ? MTLTextureUsageShaderWrite : 0u) | (heap.cls >= 6u ? MTLTextureUsageShaderAtomic : 0u);
    if (heap.cls == 3u && heap.dimension == 6u) descriptor.pixelFormat = MTLPixelFormatR32Float;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (texture == nil) FailD5("cannot create the placeholder texture of image heap " + std::to_string(heapNumber));
    std::lock_guard lock(mutex);
    if (const auto found = placeholders.find(heapNumber); found != placeholders.end()) return found->second;
    if (!tornDown) {
        [residency addAllocation:texture];
        [residency commit];
        residentCounts[(__bridge void*)texture] += 1u;
    }
    return placeholders.emplace(heapNumber, texture).first->second;
}

// --- ImageBindings -------------------------------------------------------------------------------

ImageBindings::ImageBindings(ImageCache& cache, std::function<void(std::uint64_t, std::uint64_t)> settle) : cache(cache), settle(std::move(settle)) {}

ImageBindings::~ImageBindings() = default;

std::shared_ptr<GuestSurface> ImageBindings::surface(std::span<const std::uint32_t> words, bool storage, bool atomic, std::uint32_t mip) {
    static_cast<void>(mip);
    const auto resource = DecodeTextureResource(words);
    for (const auto& existing : surfaces) {
        const auto& other = existing->resource;
        if (existing->storage == storage && other.baseAddress == resource.baseAddress && other.format == resource.format && other.width == resource.width && other.height == resource.height && other.mipCount == resource.mipCount && other.tileMode == resource.tileMode
            && other.dimension == resource.dimension && other.depthOrLastArray == resource.depthOrLastArray && other.pipeBankXor == resource.pipeBankXor) {
            if (atomic && !existing->atomic) Fail("one storage surface is bound both with and without atomics in a dispatch, which is not implemented");
            return existing;
        }
    }
    auto result = std::make_shared<GuestSurface>();
    result->resource = resource;
    result->geometry = DescribeSurface(resource);
    result->format = GuestTextureFormat(resource.format);
    result->storage = storage;
    result->atomic = atomic;
    if (result->format.blockCompressed && storage) Fail("a block-compressed texture is bound as a storage image");
    if (atomic && result->format.elementBytes != 4u) Fail("an image atomic on a texture whose elements are not 32-bit");
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    switch (resource.dimension) {
    case TextureDimension::k1D:
    case TextureDimension::k2D:
        descriptor.textureType = MTLTextureType2D;
        break;
    case TextureDimension::k1DArray:
    case TextureDimension::k2DArray:
    case TextureDimension::kCube:
        descriptor.textureType = MTLTextureType2DArray;
        descriptor.arrayLength = result->geometry.imageLayers;
        break;
    case TextureDimension::k3D:
        descriptor.textureType = MTLTextureType3D;
        descriptor.depth = result->geometry.imageDepth;
        break;
    }
    descriptor.pixelFormat = atomic ? MTLPixelFormatR32Uint : result->format.format;
    descriptor.width = resource.width;
    descriptor.height = resource.height;
    descriptor.mipmapLevelCount = resource.mipCount;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = UsageFor(storage, atomic);
    result->texture = [cache.Device() newTextureWithDescriptor:descriptor];
    if (result->texture == nil) Fail("cannot create a " + std::to_string(resource.width) + "x" + std::to_string(resource.height) + " texture of guest format " + std::to_string(resource.format));
    if (settle) settle(resource.baseAddress, result->geometry.guestBytes);
    Upload(*result);
    if (Trace()) std::fprintf(stderr, "[metal-textures] %s 0x%llx %ux%u fmt %u tile %d dim %d mips %u layers %u\n", storage ? "storage" : "sampled", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode),
        static_cast<int>(resource.dimension), resource.mipCount, result->geometry.layers);
    {
        std::lock_guard lock(cache.mutex);
        ++cache.statistics.textures;
        cache.statistics.uploadedBytes += result->geometry.guestBytes;
    }
    surfaces.push_back(result);
    return result;
}

std::uint64_t ImageBindings::view(GuestSurface& surface, std::uint32_t heapNumber, bool firstLayer, std::uint32_t mip, bool baseLevelOnly) {
    const auto heap = HeapOf(heapNumber);
    const auto& resource = surface.resource;
    const auto type = HeapTextureType(heap.dimension);
    const bool storage = heap.cls >= 4u;
    if (heap.cls == 7u) Fail("64-bit image atomics are not implemented (work packages B3/B4/D4, group G6)");
    if (type == MTLTextureType2DMultisample || type == MTLTextureType2DMultisampleArray) Fail("multisampled guest images are not implemented");
    const auto textureType = surface.texture.textureType;
    const bool arrayView = type == MTLTextureType2DArray;
    if ((type == MTLTextureType3D) != (textureType == MTLTextureType3D)) Fail("a " + std::string(type == MTLTextureType3D ? "3D" : "2D") + " image heap binds a guest texture of dimension " + std::to_string(static_cast<int>(resource.dimension)));
    const auto lastLevel = std::min(resource.lastLevel, resource.mipCount - 1u);
    const NSRange levels = storage ? NSMakeRange(mip, 1) : NSMakeRange(resource.baseLevel, baseLevelOnly || lastLevel < resource.baseLevel ? 1u : lastLevel - resource.baseLevel + 1u);
    NSRange slices = NSMakeRange(0, 1);
    if (textureType == MTLTextureType2DArray) {
        const auto layers = surface.geometry.imageLayers;
        const auto base = std::min(resource.baseArray, layers - 1u);
        slices = (arrayView && !firstLayer) ? NSMakeRange(base, layers - base) : NSMakeRange(base, 1);
    }
    id<MTLTexture> base = surface.texture;
    auto format = HeapViewFormat(surface.texture.pixelFormat, heap.cls == 1u || heap.cls == 5u || heap.cls == 6u ? 1u : heap.cls == 2u ? 2u : 0u);
    if (heap.cls == 3u) {
        // Comparison heap: a depth-format twin of the surface (Metal views cannot change a color
        // format into a depth format).
        if (surface.depth == nil) {
            const auto depthFormat = DepthFormatFor(surface.texture.pixelFormat);
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
            descriptor.textureType = textureType;
            descriptor.pixelFormat = depthFormat;
            descriptor.width = surface.texture.width;
            descriptor.height = surface.texture.height;
            descriptor.arrayLength = surface.texture.arrayLength;
            descriptor.mipmapLevelCount = surface.texture.mipmapLevelCount;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
            surface.depth = [cache.Device() newTextureWithDescriptor:descriptor];
            if (surface.depth == nil) Fail("cannot create the depth twin of a comparison-sampled texture (depth textures may need private storage and a blit)");
            const auto& geometry = surface.geometry;
            const auto* guest = reinterpret_cast<const std::byte*>(resource.baseAddress);
            std::vector<std::byte> linear;
            ForEachLevel(surface, [&](std::uint32_t layer, std::uint32_t level) {
                const auto& mipLayout = geometry.mips[level];
                const auto pitch = LinearPitch(surface, level);
                linear.assign(pitch * mipLayout.height, std::byte{0});
                const auto offset = geometry.GuestLayerOffset(layer) + mipLayout.tiledOffset;
                if (offset >= geometry.guestBytes) return;
                DetileCpu(LevelOf(surface, layer, level), guest + offset, geometry.guestBytes - offset, linear.data(), pitch, surface.format.textureElementBytes);
                [surface.depth replaceRegion:LevelRegion(surface, layer, level) mipmapLevel:level slice:geometry.CopyLayer(layer) withBytes:linear.data() bytesPerRow:pitch bytesPerImage:0];
            });
        }
        base = surface.depth;
        format = surface.depth.pixelFormat;
    }
    const auto swizzle = storage || heap.cls == 3u ? MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha) : ViewSwizzle(resource, surface.format);
    id<MTLTexture> view = [base newTextureViewWithPixelFormat:format textureType:type levels:levels slices:slices swizzle:swizzle];
    if (view == nil) Fail("Metal refused a view of a guest texture (format " + std::to_string(static_cast<unsigned>(format)) + ", type " + std::to_string(static_cast<unsigned>(type)) + ")");
    surface.views.push_back(view);
    return ResourceId(view);
}

namespace {

bool ResourceCacheEnabled() {
    static const bool noResourceCache = std::getenv("APS5_NO_RESOURCE_CACHE") != nullptr;
    static const bool noTextureCache = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    return !noResourceCache && !noTextureCache;
}

bool TemplateDataRefresh() {
    static const bool enabled = std::getenv("APS5_NO_TEMPLATE_DATA_REFRESH") == nullptr;
    return enabled;
}

// VulkanDevice.cpp's DispatchContentKey: the build's content plus the device.
ResourceCache::Key DispatchContentKey(const CompiledShader& shader, VkDevice device) {
    auto key = ShaderResources::ContentKey(shader, !TemplateDataRefresh());
    const auto deviceHandle = reinterpret_cast<std::uint64_t>(device);
    key.push_back(static_cast<std::uint32_t>(deviceHandle));
    key.push_back(static_cast<std::uint32_t>(deviceHandle >> 32u));
    return key;
}

bool HasImages(const ShaderRecompiler::RecompileResult& shader) {
    return std::any_of(shader.bindings.begin(), shader.bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers; });
}

}

void ImageBindings::Prepare(const ShaderRecompiler::RecompileResult& shader) {
    if (const auto* context = cache.ImageContext(); context != nullptr && HasImages(shader)) {
        // Upstream's dispatch build (VulkanDevice::dispatch): the resource cache entry of the
        // content when it revalidates, else a new build, cached when reusable.
        const CompiledShader compiled{ShaderRecompiler::ShaderStage::Compute, &shader, 0u};
        const bool cacheable = ResourceCacheEnabled() && shader.PipelineVariantId() != 0;
        ResourceCache::Key key;
        if (cacheable) {
            key = DispatchContentKey(compiled, context->device);
            if (auto cached = SharedResourceCache().Find(key)) {
                if (cached->Revalidate(compiled)) {
                    resources = std::move(cached);
                } else {
                    SharedResourceCache().Remove(key, cached.get());
                    if (auto* recorder = Recorder::Active()) recorder->Keep(std::move(cached));
                }
            }
        }
        if (resources == nullptr) {
            resources = std::make_shared<ShaderResources>(*context, compiled, ShaderResources::MetalImagesOnly{});
            if (cacheable && resources->Reusable()) {
                std::vector<std::shared_ptr<ShaderResources>> evicted;
                SharedResourceCache().Insert(key, resources, &evicted);
                if (auto* recorder = Recorder::Active()) {
                    for (auto& object : evicted) recorder->Keep(std::move(object));
                }
            }
        }
        return;
    }
    const std::span<const ShaderRecompiler::DescriptorBinding> bindings(shader.bindings);
    if (shader.runtimeImageCount != 0u) {
        constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
        std::array<const ShaderRecompiler::DescriptorBinding*, bindingCount> heaps{};
        const ShaderRecompiler::DescriptorBinding* shaderData = nullptr;
        for (const auto& binding : bindings) {
            if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) shaderData = &binding;
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
            auto& heap = heaps[binding.binding % bindingCount];
            if (heap != nullptr) FailD5("duplicate runtime image binding");
            heap = &binding;
        }
        std::vector<std::uint32_t> pushWords;
        std::span<const std::uint32_t> words;
        if (shaderData != nullptr) {
            words = shaderData->guestDescriptor;
        } else {
            pushWords.resize(shader.pushConstants.size() / 4u);
            std::memcpy(pushWords.data(), shader.pushConstants.data(), pushWords.size() * 4u);
            words = pushWords;
        }
        constexpr auto metadataWords = sizeof(ShaderRecompiler::RuntimeAbi::ResourceMetadata) / sizeof(std::uint32_t);
        if (shader.imageMetadataDword > words.size() || shader.runtimeImageCount > (words.size() - shader.imageMetadataDword) / metadataWords) FailD5("runtime image metadata exceeds shader data");
        for (const auto index : shader.runtimeImageResources) {
            if (index >= shader.runtimeImageCount) FailD5("runtime image resource exceeds its compact layout");
            const auto offset = shader.imageMetadataDword + index * metadataWords;
            const auto kind = words[offset];
            const auto first = words[offset + 1u];
            const auto count = words[offset + 2u];
            if (count == 0u) FailD5("runtime image metadata has no descriptor elements");
            if (kind >= heaps.size() || heaps[kind] == nullptr) FailD5("runtime metadata references an unbound image heap");
            if (!(first < heaps[kind]->count && count <= heaps[kind]->count - first)) FailD5("runtime metadata exceeds its bound heap");
        }
    }
    const ShaderRecompiler::DescriptorBinding* samplerBinding = nullptr;
    for (const auto& binding : bindings) {
        if (binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers) samplerBinding = &binding;
    }
    if (samplerBinding == nullptr) return;
    const auto count = samplerBinding->guestDescriptor.size() / 4u;
    samplerWords.resize(count);
    std::vector<bool> single(count, false);
    std::vector<bool> mipmapped(count, false);
    for (std::size_t element = 0; element < count; ++element) std::copy_n(samplerBinding->guestDescriptor.begin() + static_cast<std::ptrdiff_t>(element * 4u), 4, samplerWords[element].begin());
    for (const auto& binding : bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.kind != ShaderRecompiler::DescriptorKind::SampledImage) continue;
        const auto elementWords = binding.guestDescriptor.size() / std::max<std::size_t>(binding.count, 1u);
        for (std::size_t element = 0; element < binding.count && element < binding.imageSamplers.size(); ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * elementWords, 8u);
            if (AllZero(words)) continue;
            const auto mask = binding.imageSamplers[element];
            const bool singleLevel = DescriptorSingleLevel(words);
            const auto format = DecodeTextureResource(words).format;
            for (std::uint32_t sampler = 0; sampler < 32u && sampler < count; ++sampler) {
                if (((mask >> sampler) & 1u) == 0u) continue;
                (singleLevel ? single : mipmapped)[sampler] = true;
                const bool forcesDegamma = ((samplerWords[sampler][0] >> 20u) & 1u) != 0u;
                if (forcesDegamma && !IsSrgbTextureFormat(format)) FailD5("a texture of guest format " + std::to_string(format) + ", which is not sRGB, is sampled through a sampler that forces sRGB decoding, which is not implemented");
            }
        }
    }
    for (std::size_t element = 0; element < count; ++element) {
        if (const auto variant = SingleLevelSamplerWords(std::span<const std::uint32_t, 4>(samplerWords[element]), single[element], mipmapped[element])) samplerWords[element] = *variant;
    }
}

void ImageBindings::Write(const ShaderRecompiler::DescriptorBinding& binding, std::span<std::uint64_t> entries) {
    if (committed) FailD5("image bindings written after Commit");
    if (resources != nullptr) {
        resources->WriteTableEntries(binding, entries);
        return;
    }
    const auto heap = binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    if (binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers) {
        if (binding.guestDescriptor.size() < entries.size() * 4u) FailD5("a guest sampler binding has fewer than four words per element");
        for (std::size_t element = 0; element < entries.size(); ++element) {
            auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 4u, 4u);
            if (element < samplerWords.size()) words = samplerWords[element];
            const bool compare = element < binding.samplerDepthCompare.size() && binding.samplerDepthCompare[element];
            const bool unnormalized = element < binding.samplerUnnormalized.size() && binding.samplerUnnormalized[element];
            const bool integerBorder = element < binding.samplerIntegerBorder.size() && binding.samplerIntegerBorder[element];
            id<MTLSamplerState> sampler = cache.Sampler(words, compare, unnormalized, integerBorder);
            samplers.push_back(sampler);
            entries[element] = ResourceId(sampler);
        }
        return;
    }
    if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) FailD5("ImageBindings::Write takes guest image and sampler bindings only");
    if (entries.empty()) return;
    const auto elementWords = binding.guestDescriptor.size() / std::max<std::size_t>(binding.count, 1u);
    if (elementWords < 8u) FailD5("a guest image binding has fewer than eight words per element");
    const auto info = HeapOf(heap);
    const bool storage = info.cls >= 4u;
    const bool atomic = info.cls >= 6u;
    std::uint32_t mipOffset = 0;
    for (std::size_t element = 0; element < entries.size(); ++element) {
        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * elementWords, 8u);
        // Consecutive identical storage descriptors address successive mips of one texture
        // (dynamic-mip storage writes), as ShaderResources::resolveImageBinding.
        const bool sameAsPrevious = storage && element != 0u && std::equal(words.begin(), words.end(), binding.guestDescriptor.begin() + static_cast<std::ptrdiff_t>((element - 1u) * elementWords));
        mipOffset = sameAsPrevious ? mipOffset + 1u : 0u;
        if (AllZero(words)) {
            entries[element] = ResourceId(cache.Placeholder(heap));
            continue;
        }
        const auto resource = DecodeTextureResource(words);
        const bool firstLayer = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
        if (binding.imageShape.has_value() && !firstLayer && !MatchesGuestDimension(*binding.imageShape, resource.dimension)) {
            FailD5("guest texture dimension disagrees with the shader's declared image shape (shape " + std::to_string(static_cast<int>(*binding.imageShape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
        }
        const auto mip = storage ? std::min(resource.baseLevel + mipOffset, resource.mipCount - 1u) : 0u;
        const bool elementAtomic = atomic || (element < binding.imageAtomic.size() && binding.imageAtomic[element]);
        bool baseLevelOnly = false;
        if (!storage && element < binding.imageUnnormalized.size() && binding.imageUnnormalized[element]) {
            // Unnormalized sampling (ShaderResources::resolveImageBinding): a single-layer 1D or 2D
            // view without a minimum LOD clamp that is single-level at mip 0, or read only by
            // image_sample_lz (then viewed at its base level alone).
            const auto lastLevel = std::min(resource.lastLevel, resource.mipCount - 1u);
            const auto levelCount = lastLevel >= resource.baseLevel ? lastLevel - resource.baseLevel + 1u : 1u;
            const auto layerCount = firstLayer || resource.dimension == TextureDimension::k1D || resource.dimension == TextureDimension::k2D ? 1u : DescribeSurface(resource).imageLayers - resource.baseArray;
            const bool exactAtBase = element < binding.imageUnnormalizedBaseLevel.size() && binding.imageUnnormalizedBaseLevel[element];
            const bool singleLevel = levelCount == 1u && resource.baseLevel == 0u;
            const bool planar = resource.dimension == TextureDimension::k1D || resource.dimension == TextureDimension::k2D || firstLayer;
            const bool supported = layerCount == 1u && EffectiveMinLod(resource) == 0.0f && (singleLevel || exactAtBase);
            if (!supported || !planar) {
                throw std::runtime_error("AGC graphics: guest texture sampled with unnormalized coordinates is not a single-layer 1D or 2D view without a minimum LOD clamp, and is neither a single-level view at mip 0 nor read only by image_sample_lz through an S# without minimum LOD or bias, which is not implemented (base level " + std::to_string(resource.baseLevel) + ", levels " + std::to_string(levelCount) + ", layers " + std::to_string(layerCount) + ")");
            }
            baseLevelOnly = levelCount > 1u;
        }
        auto target = surface(words, storage, elementAtomic && storage, mip);
        entries[element] = view(*target, heap, firstLayer, mip, baseLevelOnly);
        if (storage && (element >= binding.imageWritten.size() || binding.imageWritten[element] || elementAtomic)) {
            // Write back this level when the batch completes (MetalImages.h).
            auto& written = writtenLevels[target.get()];
            written.insert(mip);
        }
    }
}

void ImageBindings::Commit(Recorder& recorder) {
    if (committed) return;
    committed = true;
    if (resources != nullptr) {
        // ShaderResources::MarkGpuWrites marks the written storage images dirty (pending: stored
        // when a reader needs their memory) and notes their ranges; the batch keeps the build.
        resources->MarkGpuWrites(recorder);
        recorder.Keep(resources);
        return;
    }
    if (surfaces.empty() && samplers.empty()) return;
    std::vector<id<MTLTexture>> textures;
    for (const auto& surface : surfaces) {
        textures.push_back(surface->texture);
        if (surface->depth != nil) textures.push_back(surface->depth);
    }
    cache.MakeResident(textures);
    struct Held {
        std::vector<std::shared_ptr<GuestSurface>> surfaces;
        std::vector<id<MTLSamplerState>> samplers;
    };
    auto held = std::make_shared<Held>(Held{surfaces, samplers});
    std::size_t bytes = 0;
    for (const auto& surface : surfaces) bytes += surface->geometry.guestBytes;
    recorder.Keep(held, bytes);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pending;
    std::vector<std::pair<std::shared_ptr<GuestSurface>, std::set<std::uint32_t>>> writeBacks;
    for (const auto& surface : surfaces) {
        const auto found = writtenLevels.find(surface.get());
        if (found == writtenLevels.end()) continue;
        pending.emplace_back(surface->resource.baseAddress, surface->geometry.guestBytes);
        writeBacks.emplace_back(surface, found->second);
    }
    if (!pending.empty()) recorder.NotePendingWrites(pending);
    auto* owner = &cache;
    recorder.OnComplete([owner, textures, writeBacks = std::move(writeBacks)] {
        std::uint64_t written = 0;
        for (const auto& [surface, levels] : writeBacks) {
            for (const auto level : levels) WriteBack(*surface, level, 1u);
            written += surface->geometry.guestBytes;
        }
        if (written != 0u) {
            std::lock_guard lock(owner->mutex);
            owner->statistics.writtenBackBytes += written;
        }
        owner->Release(textures);
    });
}

}
