// Metal twin of Graphics/src/DepthSurface.cpp (work package D6): GPU-only depth/stencil surfaces
// keyed by their guest target, as upstream, cleared when made and again after an HTILE fast clear
// (NoteHtileDepthClear, NoteDepthMetadataFill), never written back to guest memory. The Metal draw
// path (MetalDraw.mm) acquires them with Metal::AcquireDepthSurface, which hands out the clears the
// next render pass performs (its depth/stencil load action Clear) instead of upstream's
// vkCmdClearDepthStencilImage. The HTILE helpers are upstream's. Sampling a depth surface as a
// texture (DepthSurfaceTexture) is not implemented yet: such reads see guest memory
// (docs/dev/TechnicalDebt.md metal-d6-depth-sampling), and storage image access to a depth surface
// (upstream 27e112277, DepthSurfaceStorage) throws (metal-d6-depth-storage). HostDepthFormat is the
// Metal storage of each guest depth format (MetalDraw.mm DepthFormat): Apple GPUs have neither
// D16_UNORM_S8_UINT nor D24_UNORM_S8_UINT, so a 16-bit depth with stencil is Depth32Float_Stencil8.
// The memory-value HTILE fill match is upstream's. Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "MetalDepthSurfaces.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

constexpr VkImageAspectFlags DepthAspect = VK_IMAGE_ASPECT_DEPTH_BIT;
constexpr VkImageAspectFlags StencilAspect = VK_IMAGE_ASPECT_STENCIL_BIT;

struct Surface {
    const void* device;
    DepthTarget target;
    MTLPixelFormat format;
    id<MTLTexture> texture;
    VkImageAspectFlags pendingClear;
};

bool sameSurface(const DepthTarget& a, const DepthTarget& b) {
    return a.address == b.address && a.stencilAddress == b.stencilAddress && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && a.samples == b.samples;
}

std::mutex& surfacesMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<Surface>& surfaces() {
    static auto* list = new std::vector<Surface>();
    return *list;
}

std::set<std::uint64_t>& clearedHtiles() {
    static auto* cleared = new std::set<std::uint64_t>();
    return *cleared;
}

[[noreturn]] void Fail(const std::string& what) {
    throw std::runtime_error("Metal depth surface (work package D6): " + what);
}

}

std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel) {
    const std::uint32_t blockWidth = bytesPerTexel == 4 ? 128u : 256u;
    const std::uint32_t blockHeight = bytesPerTexel == 1 ? 256u : 128u;
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1) / blockHeight * blockHeight);
    return width * height * bytesPerTexel;
}

bool SampleLocationsCompatibleDepth(const Context& context, VkFormat format, std::uint32_t samples) {
    // As upstream: sample counts with custom locations, then the depth format's multisample support
    // through the context's query (a test's mock device) or the Metal device's. Metal sets sample
    // positions per render pass (programmable sample positions), so any depth texture of a supported
    // sample count is compatible.
    if (samples <= 1 || (context.sampleLocationSampleCounts & samples) == 0) return false;
    VkImageFormatProperties properties{};
    constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const auto result = context.imageFormatProperties != nullptr
        ? context.imageFormatProperties(context.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage, VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT, &properties)
        : Metal::Vk::GetPhysicalDeviceImageFormatProperties(context.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage, VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT, &properties);
    if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) return false;
    Check(result, "vkGetPhysicalDeviceImageFormatProperties sample locations depth");
    return (properties.sampleCounts & samples) != 0;
}

// Upstream asks the device for the first of D16_UNORM_S8_UINT, D24_UNORM_S8_UINT, D32_SFLOAT_S8_UINT
// with attachment and sampled support; that query runs when the context carries one (a test's mock
// device). The Metal device has none (Apple GPUs support neither 16- nor 24-bit depth with stencil,
// Metal Feature Set Tables), so the answer is the format MetalDraw.mm's DepthFormat stores it in.
VkFormat HostDepthFormat(const Context& context, VkFormat format) {
    if (format != VK_FORMAT_D16_UNORM_S8_UINT) return format;
    if (context.formatProperties == nullptr) return VK_FORMAT_D32_SFLOAT_S8_UINT;
    static std::mutex mutex;
    static std::map<VkPhysicalDevice, VkFormat> chosen;
    std::lock_guard lock(mutex);
    if (const auto found = chosen.find(context.physical); found != chosen.end()) return found->second;
    constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    auto host = format;
    for (const auto candidate : {VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
        VkFormatProperties properties{};
        context.formatProperties(context.physical, candidate, &properties);
        if ((properties.optimalTilingFeatures & required) != required) continue;
        host = candidate;
        break;
    }
    chosen.emplace(context.physical, host);
    return host;
}

// As upstream: a 16-bit depth stored as 24-bit unorm scales the constant bias by 256; stored as
// 32-bit float (always on Metal) the bias unit depends on each primitive's depth exponent, which no
// constant reproduces, so a nonzero constant is refused.
float HostDepthBiasConstant(const Context& context, VkFormat format, float constant) {
    const auto host = HostDepthFormat(context, format);
    if (constant == 0.0f || host == format) return constant;
    Require(host == VK_FORMAT_D24_UNORM_S8_UINT, "a constant depth bias on a 16-bit depth target with stencil that the device stores as 32-bit float depth is not implemented");
    return constant * 256.0f;
}

VkImageView DepthSurfaceView(const Context&, const DepthTarget&) {
    Fail("DepthSurfaceView is the Vulkan draw path's; the Metal draw path acquires surfaces with Metal::AcquireDepthSurface");
}

std::uint64_t HtileDepthClearAddress(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& numThreads) {
    static constexpr std::array<std::uint32_t, 17> htileMaskKernel{0xd7460000u, 0x04010c06u, 0x34000084u, 0xdc388000u, 0x04020000u, 0xbf8c3f70u, 0xd7710008u, 0x00120805u, 0xd7710009u, 0x00120a05u, 0xd771000au, 0x00120c05u, 0xd771000bu, 0x00120e05u, 0xdc788000u, 0x00020800u, 0xbf810000u};
    if (userData.size() < 6 || numThreads[0] != 64 || numThreads[1] != 1 || numThreads[2] != 1) return 0;
    if (code.size() < htileMaskKernel.size() || !std::equal(htileMaskKernel.begin(), htileMaskKernel.end(), code.begin())) return 0;
    if ((userData[5] & 0xfu) != 0 || (userData[4] & 0xfu) != 0) return 0;
    return userData[2] | (static_cast<std::uint64_t>(userData[3]) << 32u);
}

std::optional<MemoryValueFill> MatchMemoryValueFill(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& numThreads, const std::array<std::uint32_t, 3>& groups) {
    static constexpr std::array<std::uint32_t, 9> kernel{0xd7460000u, 0x04010c08u, 0xf4201a82u, 0xfa000000u, 0xbf8cc07fu, 0x7e02026au, 0xe0102000u, 0x80000100u, 0xbf810000u};
    if (userData.size() < 8 || numThreads != std::array<std::uint32_t, 3>{64u, 1u, 1u} || groups[1] != 1u || groups[2] != 1u) return std::nullopt;
    if (code.size() < kernel.size() || !std::equal(kernel.begin(), kernel.end(), code.begin())) return std::nullopt;
    const auto stride = (userData[1] >> 16u) & 0x3fffu;
    const bool swizzled = ((userData[1] >> 31u) & 1u) != 0;
    const bool addTid = ((userData[3] >> 23u) & 1u) != 0;
    const auto type = userData[3] >> 30u;
    const auto format = (userData[3] >> 12u) & 0x7fu;
    const auto valueStride = (userData[5] >> 16u) & 0x3fffu;
    const auto valueBytes = valueStride == 0 ? static_cast<std::uint64_t>(userData[6]) : static_cast<std::uint64_t>(userData[6]) * valueStride;
    if (type != 0 || stride != 4 || swizzled || addTid || format < 20u || format > 22u || valueBytes < 4u) return std::nullopt;
    const auto dwords = std::min<std::uint64_t>(static_cast<std::uint64_t>(groups[0]) * 64u, userData[2]);
    const auto address = userData[0] | (static_cast<std::uint64_t>(userData[1] & 0xffffu) << 32u);
    const auto valueAddress = userData[4] | (static_cast<std::uint64_t>(userData[5] & 0xffffu) << 32u);
    if (dwords == 0 || valueAddress == 0 || (valueAddress & 3u) != 0) return std::nullopt;
    return MemoryValueFill{address, static_cast<std::size_t>(dwords * 4u), valueAddress};
}

void NoteMemoryValueFill(const MemoryValueFill& fill) {
    std::uint32_t value = 0;
    GuestMemory::Read(fill.valueAddress, std::as_writable_bytes(std::span(&value, 1)), 4);
    NoteDepthMetadataFill(fill.address, fill.bytes, value);
}

void NoteHtileDepthClear(std::uint64_t htileAddress) {
    std::lock_guard lock(surfacesMutex());
    bool existing = false;
    for (auto& surface : surfaces()) {
        if (surface.target.htileAddress != htileAddress) continue;
        surface.pendingClear |= DepthAspect;
        existing = true;
    }
    if (!existing) clearedHtiles().insert(htileAddress);
}

void ClearDepthSurfaces(VkDevice device) {
    std::lock_guard lock(surfacesMutex());
    std::erase_if(surfaces(), [&](const Surface& surface) { return surface.device == reinterpret_cast<const void*>(device); });
}

std::shared_ptr<Texture> DepthSurfaceTexture(const Context&, std::span<const std::uint32_t>, const GuestTextureResource&, VkComponentMapping) {
    return nullptr;
}

bool DepthSurfaceHolds(const Context&, const GuestTextureResource&, const Texture*) {
    return false;
}

VkImageAspectFlags HtileFillClears(std::uint32_t pattern, bool stencilInHtile) {
    VkImageAspectFlags cleared = (pattern & 0xfu) == 0 ? DepthAspect : 0u;
    if (stencilInHtile && ((pattern >> 8u) & 0x3u) == 0) cleared |= StencilAspect;
    return cleared;
}

std::uint64_t HtileMetadataBytes(VkExtent2D extent) {
    const auto blocks = static_cast<std::uint64_t>((extent.width + 1023u) / 1024u) * ((extent.height + 511u) / 512u);
    return blocks * 32768u;
}

bool HtileFillCovers(std::uint64_t htile, VkExtent2D extent, std::uint64_t address, std::size_t bytes) {
    if (htile == 0 || htile < address || htile >= address + bytes) return false;
    return address + bytes - htile >= HtileMetadataBytes(extent);
}

void NoteDepthMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern) {
    std::lock_guard lock(surfacesMutex());
    for (auto& surface : surfaces()) {
        const auto& target = surface.target;
        if (!HtileFillCovers(target.htileAddress, target.extent, address, bytes)) continue;
        const VkImageAspectFlags written = DepthAspect | (target.htileStencil ? StencilAspect : 0u);
        surface.pendingClear = (surface.pendingClear & ~written) | HtileFillClears(pattern, target.htileStencil);
    }
}

bool DepthSurfaceAt(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource) {
    std::lock_guard lock(surfacesMutex());
    const auto& list = surfaces();
    const auto key = reinterpret_cast<const void*>(context.device);
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const Surface& surface) {
        return surface.device == key && (surface.target.address == resource.baseAddress || (surface.target.stencilAddress != 0 && surface.target.stencilAddress == resource.baseAddress));
    });
    if (found == list.rend()) return false;
    const auto& target = found->target;
    if (resource.width != target.extent.width || resource.height != target.extent.height) return false;
    const bool stencil = target.stencilAddress != 0 && target.stencilAddress == resource.baseAddress;
    const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    return ResolveTextureFormat(resource.format) == expected || depthBits;
}

// Upstream (27e112277) copies the depth or stencil plane into a color storage image and back after
// the writes. The Metal surfaces are not sampled either (metal-d6-depth-sampling), so a storage
// access to one throws, as the twin's storage path did before (TechnicalDebt metal-d6-depth-storage).
std::shared_ptr<StorageTexture> DepthSurfaceStorage(const Context& context, std::span<const std::uint32_t>, const GuestTextureResource& resource, std::uint32_t mip) {
    std::lock_guard lock(surfacesMutex());
    const auto key = reinterpret_cast<const void*>(context.device);
    const auto found = std::find_if(surfaces().rbegin(), surfaces().rend(), [&](const Surface& surface) {
        return surface.device == key && (surface.target.address == resource.baseAddress || (surface.target.stencilAddress != 0 && surface.target.stencilAddress == resource.baseAddress));
    });
    if (found == surfaces().rend()) return nullptr;
    char text[288];
    std::snprintf(text, sizeof(text), "AGC graphics: storage image access to depth/stencil surface 0x%llx (storage %ux%u guest format %u mip %u) is not implemented on Metal (TechnicalDebt metal-d6-depth-storage)", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, mip);
    throw std::runtime_error(text);
}

// No storage image of a depth surface is handed out (DepthSurfaceStorage), so none is stored back.
bool DepthSurfaceStoreBack(const Context&, const StorageTexture&) {
    return false;
}

bool DepthSurfaceAt(std::uint64_t address) {
    std::lock_guard lock(surfacesMutex());
    return std::any_of(surfaces().begin(), surfaces().end(), [&](const Surface& surface) { return surface.target.address == address || surface.target.stencilAddress == address; });
}

namespace Metal {

DepthSurfaceUse AcquireDepthSurface(id<MTLDevice> device, const DepthTarget& target, MTLPixelFormat format, bool* created) {
    std::lock_guard lock(surfacesMutex());
    *created = false;
    const auto key = (__bridge const void*)device;
    const bool cleared = target.htileAddress != 0 && clearedHtiles().erase(target.htileAddress) != 0;
    const VkImageAspectFlags aspects = DepthAspect | (target.stencilAddress != 0 ? StencilAspect : 0u);
    for (auto& surface : surfaces()) {
        if (surface.device != key || !sameSurface(surface.target, target) || surface.format != format) continue;
        surface.target.clearDepth = target.clearDepth;
        surface.target.clearStencil = target.clearStencil;
        if (cleared) surface.pendingClear |= DepthAspect;
        const auto clears = surface.pendingClear & aspects;
        surface.pendingClear = 0;
        return {surface.texture, (clears & DepthAspect) != 0, (clears & StencilAspect) != 0, target.clearDepth, target.clearStencil};
    }
    if (target.extent.width == 0 || target.extent.height == 0 || target.extent.width > 16384u || target.extent.height > 16384u) Fail("depth target extent " + std::to_string(target.extent.width) + "x" + std::to_string(target.extent.height));
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:target.extent.width height:target.extent.height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    // Multisampled depth (DECISIONS D27): a 2D multisample texture, GPU-only like every surface
    // (upstream's sampling of a multisampled depth surface is not implemented either).
    // A 2D multisample array of HostSamplesOf's slices (DECISIONS D28: 8x on Apple9 is two 4x
    // halves, each rendered by its own pass); the draw attaches one slice per pass.
    if (target.samples > 1u) {
        const auto host = HostSamplesOf(device, target.samples);
        descriptor.textureType = MTLTextureType2DMultisampleArray;
        descriptor.sampleCount = host.samples;
        descriptor.arrayLength = host.slices;
    }
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (texture == nil) Fail("cannot create a depth surface");
    // Made cleared to the target's clear values, as upstream clears a new surface.
    surfaces().push_back({key, target, format, texture, 0});
    *created = true;
    return {texture, true, (aspects & StencilAspect) != 0, target.clearDepth, target.clearStencil};
}

std::vector<id<MTLTexture>> DepthSurfaceTextures(id<MTLDevice> device) {
    std::lock_guard lock(surfacesMutex());
    std::vector<id<MTLTexture>> textures;
    for (const auto& surface : surfaces()) {
        if (surface.device == (__bridge const void*)device) textures.push_back(surface.texture);
    }
    return textures;
}

}

}
