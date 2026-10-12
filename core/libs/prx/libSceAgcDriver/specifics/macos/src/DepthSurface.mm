// Metal twin of Graphics/src/DepthSurface.cpp (work package D6): GPU-only depth/stencil surfaces
// keyed by their guest target, as upstream, cleared when made and again after an HTILE fast clear
// (NoteHtileDepthClear, NoteDepthMetadataFill), never written back to guest memory. The Metal draw
// path (MetalDraw.mm) acquires them with Metal::AcquireDepthSurface, which hands out the clears the
// next render pass performs (its depth/stencil load action Clear) instead of upstream's
// vkCmdClearDepthStencilImage. The HTILE helpers are upstream's. Sampling a depth surface as a
// texture (DepthSurfaceTexture) is not implemented yet: such reads see guest memory
// (docs/dev/TechnicalDebt.md metal-d6-depth-sampling). Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "MetalDepthSurfaces.h"
#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
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

bool SampleLocationsCompatibleDepth(const Context&, VkFormat, std::uint32_t) {
    // Metal has no programmable sample locations for depth attachments.
    return false;
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
