// Metal twin of Graphics/src/DepthSurface.cpp (work package D6, upstream sync 0f2683cb7): GPU-only
// depth/stencil surfaces keyed by their guest target, as upstream, cleared when made and again after an
// HTILE fast clear (NoteHtileDepthClear, NoteDepthMetadataFill), never written back to guest memory. The
// Metal draw path (MetalDraw.mm) acquires them with Metal::AcquireDepthSurface, which hands out the
// clears the next render pass performs (its depth/stencil load action Clear) instead of upstream's
// vkCmdClearDepthStencilImage; DepthSurfaceView (upstream's draw-path entry, used by twins of upstream
// tests) makes and clears them as upstream does. Work outside a draw that reads a surface first applies
// its pending clear as a clear-only render pass (ApplyFastClear).
//
// Sampling (DepthSurfaceTexture, upstream #425 086f7d509 90d899d98 af7160475 50e5f088c e4a7cbfc8 cc3b47dda)
// and storage access (DepthSurfaceStorage / DepthSurfaceStoreBack, upstream 27e112277 7d06be06f 889e5d436
// f15671ecf) copy planes, as upstream's DepthPlaneCopy and copyPlane do, with Metal blits: a plane of a
// combined Depth32Float_Stencil8 texture moves with MTLBlitOptionDepthFromDepthStencil or
// StencilFromDepthStencil into a staging buffer and from there into a colour storage image or a sampled
// plane copy (and back for a store). Unlike upstream, a single plane is sampled through a plane copy too
// (one layer) rather than a view of the depth texture: a view of a combined depth/stencil texture names
// one aspect by pixel format (X32_Stencil8) only for stencil, and the copy keeps one path for every
// shape. Plane copies are refreshed per use (Texture::MarkRefreshedPerUse) from the surfaces written
// since their last copy and the guest layers the CPU or the GPU wrote since (upstream's rules).
//
// HostDepthFormat is the Metal storage of each guest depth format (MetalDraw.mm DepthFormat): Apple GPUs
// have neither D16_UNORM_S8_UINT nor D24_UNORM_S8_UINT, so a 16-bit depth with stencil is
// Depth32Float_Stencil8 (widened: its depth plane is refused for storage, integer and layered reads, as
// upstream refuses planes stored in another depth format). Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalCommandStream.h"
#include "MetalImages.h"
#include "MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
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
#include <tuple>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

constexpr VkImageAspectFlags DepthAspect = VK_IMAGE_ASPECT_DEPTH_BIT;
constexpr VkImageAspectFlags StencilAspect = VK_IMAGE_ASPECT_STENCIL_BIT;

[[noreturn]] void Fail(const std::string& what) {
    throw std::runtime_error("Metal depth surface (work package D6): " + what);
}

// Serial of depth surface writes (draws, clears, store-backs), under surfacesMutex.
std::uint64_t depthWrites = 0;

struct PlaneStorage {
    std::shared_ptr<StorageTexture> storage;
    std::vector<std::weak_ptr<StorageTexture>> replaced;
    std::shared_ptr<DeviceBuffer> staging;
};

struct Surface {
    Surface(const void* device, const DepthTarget& target, MTLPixelFormat format, id<MTLTexture> texture) : device(device), target(target), format(format), texture(texture) {}
    ~Surface() {
        if (view != VK_NULL_HANDLE) Metal::Vk::DestroyImageView(VK_NULL_HANDLE, view, nullptr);
    }
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    const void* device;
    DepthTarget target;
    MTLPixelFormat format;
    id<MTLTexture> texture;
    VkImageAspectFlags pendingClear = 0;
    std::uint64_t written = ++depthWrites;
    // DepthSurfaceView's handle of the texture (made on its first call).
    VkImageView view = VK_NULL_HANDLE;
    PlaneStorage depthPlane;
    PlaneStorage stencilPlane;

    bool HasDepth() const { return format != MTLPixelFormatStencil8; }
    bool HasStencil() const { return format == MTLPixelFormatDepth32Float_Stencil8 || format == MTLPixelFormatStencil8; }
    bool D16() const { return target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT; }
    // A 16-bit guest depth stored as 32-bit float (with stencil, Apple GPUs have no D16S8).
    bool Widened() const { return D16() && format != MTLPixelFormatDepth16Unorm; }
    std::uint32_t PlaneBytes(bool stencil) const { return stencil ? 1u : format == MTLPixelFormatDepth16Unorm ? 2u : 4u; }
    MTLBlitOption PlaneOption(bool stencil) const {
        if (format != MTLPixelFormatDepth32Float_Stencil8) return MTLBlitOptionNone;
        return stencil ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil;
    }
    VkImage Image() const { return reinterpret_cast<VkImage>((__bridge void*)texture); }
};

// The surface the last draw attached (upstream's drawnDepth): its plane copies are always refreshed, the
// draw being recorded may still write it.
Surface* drawnDepth = nullptr;

bool sameSurface(const DepthTarget& a, const DepthTarget& b) {
    return a.address == b.address && a.stencilAddress == b.stencilAddress && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && a.samples == b.samples;
}

std::mutex& surfacesMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::unique_ptr<Surface>>& surfaces() {
    static auto* list = new std::vector<std::unique_ptr<Surface>>();
    return *list;
}

std::set<std::uint64_t>& clearedHtiles() {
    static auto* cleared = new std::set<std::uint64_t>();
    return *cleared;
}

const void* DeviceKey(const Context& context) {
    return reinterpret_cast<const void*>(context.device);
}

Recorder* RecorderFor(const Context& context) {
    auto* recorder = Recorder::Active();
    return recorder != nullptr && recorder->Device() == context.device ? recorder : nullptr;
}

// The commands of the active recorder of the device, or of a one-shot batch submitted and waited for at
// Finish (upstream's RecorderFor / CommandBatch pattern).
class SurfaceCommands {
public:
    explicit SurfaceCommands(const Context& context) : recorder(RecorderFor(context)) {
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        handle = recorder != nullptr ? recorder->Commands() : batch->Handle();
    }
    VkCommandBuffer Handle() const { return handle; }
    Recorder* Recording() const { return recorder; }
    void Keep(std::shared_ptr<void> object) {
        if (recorder != nullptr) recorder->Keep(std::move(object));
        else kept.push_back(std::move(object));
    }
    void Finish() {
        if (batch) batch->SubmitAndWait();
        kept.clear();
    }

private:
    Recorder* recorder;
    std::unique_ptr<CommandBatch> batch;
    VkCommandBuffer handle = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<void>> kept;
};

// Copies one plane of the surface (all of it, slice 0) to or from `buffer` at `offset`, tightly packed.
void CopySurfacePlane(VkCommandBuffer commands, const Surface& surface, bool stencil, VkBuffer buffer, std::uint64_t offset, bool toBuffer) {
    auto& stream = Metal::Stream(commands);
    auto encoder = stream.Compute();
    const auto bytes = surface.PlaneBytes(stencil);
    const NSUInteger width = surface.target.extent.width;
    const NSUInteger height = surface.target.extent.height;
    const NSUInteger bytesPerRow = width * bytes;
    const NSUInteger bytesPerImage = bytesPerRow * height;
    const auto options = surface.PlaneOption(stencil);
    id<MTLBuffer> metalBuffer = Metal::BufferOf(buffer);
    if (toBuffer) {
        [encoder copyFromTexture:surface.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1) toBuffer:metalBuffer destinationOffset:offset destinationBytesPerRow:bytesPerRow destinationBytesPerImage:bytesPerImage options:options];
    } else {
        [encoder copyFromBuffer:metalBuffer sourceOffset:offset sourceBytesPerRow:bytesPerRow sourceBytesPerImage:bytesPerImage sourceSize:MTLSizeMake(width, height, 1) toTexture:surface.texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0) options:options];
    }
    stream.CountCommand();
    Metal::NoteImageCopy(surface.Image(), toBuffer);
}

// Upstream's ApplyFastClear: the pending HTILE/creation clears, as a clear-only render pass (one per
// slice of an emulated multisample array).
void ApplyFastClear(const Context& context, Surface& surface) {
    const auto aspects = surface.pendingClear & ((surface.HasDepth() ? DepthAspect : 0u) | (surface.HasStencil() ? StencilAspect : 0u));
    surface.pendingClear = 0;
    if (aspects == 0) return;
    surface.written = ++depthWrites;
    SurfaceCommands commands(context);
    constexpr VkAccessFlags access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, access, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    auto& stream = Metal::Stream(commands.Handle());
    const auto slices = surface.texture.textureType == MTLTextureType2DMultisampleArray ? surface.texture.arrayLength : NSUInteger{1};
    for (NSUInteger slice = 0; slice < slices; ++slice) {
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        if (surface.HasDepth()) {
            pass.depthAttachment.texture = surface.texture;
            pass.depthAttachment.slice = slice;
            pass.depthAttachment.loadAction = (aspects & DepthAspect) != 0 ? MTLLoadActionClear : MTLLoadActionLoad;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            pass.depthAttachment.clearDepth = surface.target.clearDepth;
        }
        if (surface.HasStencil()) {
            pass.stencilAttachment.texture = surface.texture;
            pass.stencilAttachment.slice = slice;
            pass.stencilAttachment.loadAction = (aspects & StencilAspect) != 0 ? MTLLoadActionClear : MTLLoadActionLoad;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
            pass.stencilAttachment.clearStencil = surface.target.clearStencil;
        }
        pass.renderTargetWidth = surface.texture.width;
        pass.renderTargetHeight = surface.texture.height;
        stream.BeginRender(pass);
        stream.EndEncoder();
        stream.CountCommand();
    }
    RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, access);
    if (commands.Recording() != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
    commands.Finish();
}

// Upstream's DepthSurface::Storage / copyPlane / StoreBack over a Metal surface.
void CopyPlane(const Context& context, Surface& surface, PlaneStorage& plane, bool stencil, bool toStorage) {
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "depth surface storage copies need an active recorder");
    const auto& target = surface.target;
    const auto bytes = static_cast<std::size_t>(target.extent.width) * target.extent.height * surface.PlaneBytes(stencil);
    if (plane.staging == nullptr) plane.staging = std::make_shared<DeviceBuffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const auto commands = recorder->Commands();
    recorder->Keep(plane.storage);
    recorder->Keep(plane.staging);
    const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
    VkBufferImageCopy colorRegion{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    if (toStorage) CopySurfacePlane(commands, surface, stencil, plane.staging->Handle(), 0, true);
    else Metal::Vk::CmdCopyImageToBuffer(commands, plane.storage->Image(), VK_IMAGE_LAYOUT_GENERAL, plane.staging->Handle(), 1, &colorRegion);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    if (toStorage) Metal::Vk::CmdCopyBufferToImage(commands, plane.staging->Handle(), plane.storage->Image(), VK_IMAGE_LAYOUT_GENERAL, 1, &colorRegion);
    else CopySurfacePlane(commands, surface, stencil, plane.staging->Handle(), 0, false);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
    recorder->EndGpuTiming(timing, bytes);
}

std::shared_ptr<StorageTexture> SurfaceStorage(const Context& context, Surface& surface, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip) {
    const auto& target = surface.target;
    if (resource.width != target.extent.width || resource.height != target.extent.height) return nullptr;
    const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
    const bool d16 = surface.D16();
    const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    if (ResolveTextureFormat(resource.format) != expected && !depthBits) return nullptr;
    const bool oneLayer = resource.dimension == TextureDimension::k2DArray && resource.baseArray == 0 && resource.depthOrLastArray == 0;
    if (resource.dimension != TextureDimension::k2D && !oneLayer) {
        char text[256];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image access to depth/stencil surface 0x%llx is not implemented (storage=%ux%u guest format=%u mip=%u dimension=%d tile=%d)", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, mip, static_cast<int>(resource.dimension), static_cast<int>(resource.tileMode));
        throw std::runtime_error(text);
    }
    Require(target.samples == 1, "storage image access to a multisampled depth surface is not implemented");
    if (!stencil && surface.Widened()) {
        char text[224];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the depth plane of depth surface 0x%llx (vk format %d) stored as vk format %d is not implemented", static_cast<unsigned long long>(target.address), static_cast<int>(target.format), static_cast<int>(VK_FORMAT_D32_SFLOAT_S8_UINT));
        throw std::runtime_error(text);
    }
    if (mip != 0 || resource.baseLevel != 0) {
        char text[224];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image access to level %u of the %s plane of depth surface 0x%llx is not implemented", mip, stencil ? "stencil" : "depth", static_cast<unsigned long long>(target.address));
        throw std::runtime_error(text);
    }
    Require(stencil ? surface.HasStencil() : surface.HasDepth(), "storage image access to a plane the depth surface does not have");
    Require(context.detiler != nullptr, "depth surface storage requires a texture detiler");
    ApplyFastClear(context, surface);
    auto& plane = stencil ? surface.stencilPlane : surface.depthPlane;
    if (plane.storage == nullptr || plane.storage->Descriptor().format != resource.format || plane.storage->Descriptor().tileMode != resource.tileMode || plane.storage->Descriptor().baseAddress != resource.baseAddress || plane.storage->Descriptor().dimension != resource.dimension) {
        std::erase_if(plane.replaced, [](const auto& old) { return old.expired(); });
        if (plane.storage != nullptr) plane.replaced.push_back(plane.storage);
        plane.storage = StorageTexture::Create(context, *context.detiler, resource, 0);
    }
    CopyPlane(context, surface, plane, stencil, true);
    return plane.storage;
}

bool SurfaceStoreBack(const Context& context, Surface& surface, const StorageTexture& candidate) {
    for (const bool stencil : {false, true}) {
        auto& plane = stencil ? surface.stencilPlane : surface.depthPlane;
        if (plane.storage.get() == &candidate) {
            CopyPlane(context, surface, plane, stencil, false);
            surface.written = ++depthWrites;
            return true;
        }
        if (std::none_of(plane.replaced.begin(), plane.replaced.end(), [&](const auto& old) { return old.lock().get() == &candidate; })) continue;
        char text[288];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image writes to the %s plane of depth surface 0x%llx through a view that another view of the plane (guest format %u, tile mode %d) replaced are not implemented", stencil ? "stencil" : "depth", static_cast<unsigned long long>(surface.target.address), plane.storage->Descriptor().format, static_cast<int>(plane.storage->Descriptor().tileMode));
        throw std::runtime_error(text);
    }
    return false;
}

// One layer of a plane copy: the surface holding it (null: guest memory) and its write serial at the
// lookup (0: always copied).
struct PlaneSlice {
    Surface* surface = nullptr;
    bool stencil = false;
    std::uint64_t written = 0;
};

// Upstream's DepthPlaneCopy: a sampled image of `layers` depth (or stencil, or 16-bit integer) planes,
// filled from the surfaces holding them and from guest memory for the others, refreshed per use.
class PlaneCopy {
public:
    PlaneCopy(const Context& context, VkExtent2D extent, VkFormat format, std::uint32_t layers) : context(context), extent(extent), format(format), layers(layers), guestLayers(layers), surfaceLayers(layers) {
        this->context.bufferPool.reset();
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {extent.width, extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = layers;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(Metal::Vk::CreateImage(context.device, &info, nullptr, &image), "vkCreateImage depth plane copy");
        staging = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(sliceBytes()) * layers, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    ~PlaneCopy() {
        textures.clear();
        staging.reset();
        if (image != VK_NULL_HANDLE) Metal::Vk::DestroyImage(context.device, image, nullptr);
    }
    PlaneCopy(const PlaneCopy&) = delete;
    PlaneCopy& operator=(const PlaneCopy&) = delete;

    std::shared_ptr<Texture> Refresh(std::span<const PlaneSlice> slices, const GuestTextureResource& resource, VkComponentMapping components, VkImageViewType viewType) {
        Require(slices.size() == layers, "depth plane copy slices do not match its layers");
        const bool guest = std::any_of(slices.begin(), slices.end(), [](const PlaneSlice& slice) { return slice.surface == nullptr; });
        SurfaceGeometry geometry{};
        if (guest) {
            geometry = DescribeSurface(resource);
            Require(geometry.layers == layers && geometry.sliceLinearBytes == sliceBytes() && !geometry.mips.empty(), "depth plane copy geometry does not match its layers");
        }
        std::vector<std::shared_ptr<Buffer>> uploads;
        std::vector<std::uint32_t> uploadLayers;
        std::vector<std::uint32_t> copiedLayers;
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            auto& held = guestLayers[layer];
            auto& surface = surfaceLayers[layer];
            if (slices[layer].surface != nullptr) {
                held = {};
                if (slices[layer].written != 0 && surface.texture == (__bridge void*)slices[layer].surface->texture && surface.written == slices[layer].written) continue;
                surface = {(__bridge void*)slices[layer].surface->texture, slices[layer].written};
                copiedLayers.push_back(layer);
                continue;
            }
            surface = {};
            const GuestLayer current{resource.baseAddress + geometry.GuestLayerOffset(layer), geometry.layerBytes, resource.tileMode, geometry.thick, GuestMemory::CollectWrites(resource.baseAddress + geometry.GuestLayerOffset(layer), static_cast<std::size_t>(geometry.layerBytes))};
            if (held.generation != 0 && held.address == current.address && held.bytes == current.bytes && held.tileMode == current.tileMode && held.thick == current.thick && !Recorder::SnapshotWriteOverlaps(current.address, static_cast<std::size_t>(current.bytes)) && GuestMemory::UnchangedSince(current.address, static_cast<std::size_t>(current.bytes), held.generation)) {
                held.generation = current.generation;
                continue;
            }
            auto upload = std::make_shared<Buffer>(context, static_cast<std::size_t>(geometry.layerBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            GuestMemory::ReadCommitted(current.address, upload->Bytes().first(static_cast<std::size_t>(geometry.layerBytes)));
            uploads.push_back(std::move(upload));
            uploadLayers.push_back(layer);
            held = current;
        }
        if (!copiedLayers.empty() || !uploads.empty()) record(slices, resource, geometry, copiedLayers, uploads, uploadLayers);
        const std::array<std::uint32_t, 5> key{static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a), static_cast<std::uint32_t>(viewType)};
        auto& texture = textures[key];
        if (texture == nullptr) {
            texture = std::make_shared<Texture>(context, image, format, aspect(), components, viewType);
            texture->MarkRefreshedPerUse();
        }
        return texture;
    }

    const Context context;
    const VkExtent2D extent;
    const VkFormat format;
    const std::uint32_t layers;

private:
    struct GuestLayer {
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        TextureTileMode tileMode = TextureTileMode::kLinear;
        bool thick = false;
        std::uint64_t generation = 0;
    };
    struct SurfaceLayer {
        const void* texture = nullptr;
        std::uint64_t written = 0;
    };

    VkImage image = VK_NULL_HANDLE;
    std::shared_ptr<DeviceBuffer> staging;
    std::map<std::array<std::uint32_t, 5>, std::shared_ptr<Texture>> textures;
    std::vector<GuestLayer> guestLayers;
    std::vector<SurfaceLayer> surfaceLayers;

    void record(std::span<const PlaneSlice> slices, const GuestTextureResource& resource, const SurfaceGeometry& geometry, std::span<const std::uint32_t> copiedLayers, std::span<const std::shared_ptr<Buffer>> uploads, std::span<const std::uint32_t> uploadLayers) {
        if (!uploads.empty()) {
            Require(context.detiler != nullptr, "depth plane copy requires a texture detiler");
            context.detiler->BeginBatch();
        }
        SurfaceCommands commands(context);
        commands.Keep(staging);
        RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        std::vector<VkBufferImageCopy> regions;
        const auto region = [&](std::uint32_t layer) {
            VkBufferImageCopy copy{};
            copy.bufferOffset = sliceBytes() * layer;
            copy.imageSubresource = {aspect(), 0, layer, 1};
            copy.imageExtent = {extent.width, extent.height, 1};
            return copy;
        };
        for (const auto layer : copiedLayers) {
            CopySurfacePlane(commands.Handle(), *slices[layer].surface, slices[layer].stencil, staging->Handle(), sliceBytes() * layer, true);
            regions.push_back(region(layer));
        }
        if (!uploads.empty()) {
            RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            for (std::size_t upload = 0; upload < uploads.size(); ++upload) {
                const auto layer = uploadLayers[upload];
                context.detiler->Dispatch(commands.Handle(), resource.tileMode, elementBytes(), uploads[upload]->Handle(), 0, staging->Handle(), geometry.LinearLayerOffset(layer), geometry.mips.front(), false, layer, geometry.thick);
                regions.push_back(region(layer));
                commands.Keep(uploads[upload]);
            }
        }
        RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        Metal::Vk::CmdCopyBufferToImage(commands.Handle(), staging->Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()), regions.data());
        RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        if (commands.Recording() != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, 3);
        commands.Finish();
    }

    std::uint32_t elementBytes() const {
        return format == VK_FORMAT_D32_SFLOAT ? 4u : format == VK_FORMAT_R8_UINT ? 1u : 2u;
    }

    VkDeviceSize sliceBytes() const {
        return static_cast<VkDeviceSize>(extent.width) * extent.height * elementBytes();
    }

    VkImageAspectFlags aspect() const {
        return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D16_UNORM ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    }
};

std::map<std::tuple<const void*, std::uint64_t, std::uint32_t, std::uint32_t, VkFormat, std::uint32_t>, std::unique_ptr<PlaneCopy>>& planeCopies() {
    static auto* copies = new std::map<std::tuple<const void*, std::uint64_t, std::uint32_t, std::uint32_t, VkFormat, std::uint32_t>, std::unique_ptr<PlaneCopy>>();
    return *copies;
}

Surface* FindSurface(const Context& context, std::uint64_t address) {
    const auto& list = surfaces();
    const auto key = DeviceKey(context);
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) {
        return surface->device == key && (surface->target.address == address || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == address));
    });
    return found == list.rend() ? nullptr : found->get();
}

// The Metal pixel format of a guest depth target (MetalDraw.mm DepthFormat).
MTLPixelFormat SurfaceFormat(const DepthTarget& target) {
    const bool stencil = target.stencilAddress != 0u;
    switch (target.format) {
    case VK_FORMAT_D32_SFLOAT: return stencil ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth32Float;
    case VK_FORMAT_D16_UNORM: return stencil ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth16Unorm;
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D16_UNORM_S8_UINT: return MTLPixelFormatDepth32Float_Stencil8;
    case VK_FORMAT_S8_UINT: return MTLPixelFormatStencil8;
    default: Fail("depth format " + std::to_string(target.format));
    }
}

// A new surface's texture (a 2D multisample array of HostSamplesOf's slices when multisampled, D27/D28).
id<MTLTexture> MakeSurfaceTexture(id<MTLDevice> device, const DepthTarget& target, MTLPixelFormat format) {
    if (target.extent.width == 0 || target.extent.height == 0 || target.extent.width > 16384u || target.extent.height > 16384u) Fail("depth target extent " + std::to_string(target.extent.width) + "x" + std::to_string(target.extent.height));
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:target.extent.width height:target.extent.height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    // Multisampled depth (DECISIONS D27): a 2D multisample texture, GPU-only like every surface
    // (upstream's sampling of a multisampled depth surface is not implemented either).
    // A 2D multisample array of HostSamplesOf's slices (DECISIONS D28: 8x on Apple9 is two 4x
    // halves, each rendered by its own pass); the draw attaches one slice per pass.
    if (target.samples > 1u) {
        const auto host = Metal::HostSamplesOf(device, target.samples);
        descriptor.textureType = MTLTextureType2DMultisampleArray;
        descriptor.sampleCount = host.samples;
        descriptor.arrayLength = host.slices;
    }
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (texture == nil) Fail("cannot create a depth surface");
    return texture;
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

// Upstream's draw-path entry (the Vulkan draw path attaches the returned view). The Metal draw path uses
// Metal::AcquireDepthSurface; twins of upstream tests and any caller of the upstream API get the same
// surface registry: a new surface is made and cleared at once, a known one gets its pending clears, and
// either becomes the drawn depth surface.
VkImageView DepthSurfaceView(const Context& context, const DepthTarget& target) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)reinterpret_cast<void*>(context.device);
    if (device == nil) Fail("DepthSurfaceView needs the Metal device's context");
    std::lock_guard lock(surfacesMutex());
    const auto format = SurfaceFormat(target);
    const bool cleared = target.htileAddress != 0 && clearedHtiles().erase(target.htileAddress) != 0;
    const auto key = DeviceKey(context);
    Surface* surface = nullptr;
    for (const auto& candidate : surfaces()) {
        if (candidate->device != key || !sameSurface(candidate->target, target) || candidate->format != format) continue;
        surface = candidate.get();
        surface->target.clearDepth = target.clearDepth;
        surface->target.clearStencil = target.clearStencil;
        if (cleared) surface->pendingClear |= DepthAspect;
        break;
    }
    if (surface == nullptr) {
        id<MTLTexture> texture = MakeSurfaceTexture(device, target, format);
        if (auto* images = Metal::ImageCache::For(device)) {
            const std::array<id<MTLTexture>, 1> resident{texture};
            images->MakeResident(resident);
        }
        surfaces().push_back(std::make_unique<Surface>(key, target, format, texture));
        surface = surfaces().back().get();
        surface->pendingClear = DepthAspect | StencilAspect;
    }
    ApplyFastClear(context, *surface);
    surface->written = ++depthWrites;
    drawnDepth = surface;
    if (surface->view == VK_NULL_HANDLE) surface->view = Metal::RetainView(surface->texture);
    return surface->view;
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
    for (const auto& surface : surfaces()) {
        if (surface->target.htileAddress != htileAddress) continue;
        surface->pendingClear |= DepthAspect;
        existing = true;
    }
    if (!existing) clearedHtiles().insert(htileAddress);
}

void ClearDepthSurfaces(VkDevice device) {
    std::lock_guard lock(surfacesMutex());
    const auto key = reinterpret_cast<const void*>(device);
    std::erase_if(planeCopies(), [&](const auto& entry) { return std::get<0>(entry.first) == key; });
    if (drawnDepth != nullptr && drawnDepth->device == key) drawnDepth = nullptr;
    std::erase_if(surfaces(), [&](const auto& surface) { return surface->device == key; });
}

// Upstream's DepthSurfaceTexture over plane copies (see the file comment): a single plane is a
// one-layer copy of the surface; several slices and integer views are upstream's DepthPlaneCopy.
std::shared_ptr<Texture> DepthSurfaceTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components) {
    {
        std::lock_guard lock(surfacesMutex());
        if (FindSurface(context, resource.baseAddress) == nullptr) return nullptr;
    }
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    auto* found = FindSurface(context, resource.baseAddress);
    if (found == nullptr) return nullptr;
    const auto& base = found->target;
    if (resource.width != base.extent.width || resource.height != base.extent.height) return nullptr;
    const bool stencil = base.stencilAddress != 0 && resource.baseAddress == base.stencilAddress;
    const bool d16 = found->D16();
    const auto format = ResolveTextureFormat(resource.format);
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    const bool integer = d16 && format == VK_FORMAT_R16_UINT && !depthBits;
    if (!stencil && format != (d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT) && !depthBits && !integer) return nullptr;
    if (stencil && BytesPerElement(resource.format) != 1u) return nullptr;
    ApplyFastClear(context, *found);
    const bool cube = resource.dimension == TextureDimension::kCube;
    const bool layered = cube || (resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray != 0);
    if (stencil || (!layered && !integer)) {
        // Upstream's DepthSurface::Sampled: one plane of the surface.
        Require(base.samples == 1, "sampling a multisampled depth surface is not implemented");
        const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
        const bool oneSlice = resource.dimension == TextureDimension::k2D || (resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray == 0);
        if ((format != expected && !depthBits) || !oneSlice || resource.baseLevel != 0 || resource.lastLevel != 0 || resource.baseArray != 0) {
            std::array<std::uint32_t, 8> key{};
            std::copy_n(words.begin(), std::min<std::size_t>(words.size(), 8), key.begin());
            char text[448];
            std::snprintf(text, sizeof(text), "AGC graphics: sampling the %s plane of depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u texture of guest format %u (vk %d), tile mode %u, dimension %d, levels %u-%u, slice %u is not implemented (T# %08x %08x %08x %08x %08x %08x %08x %08x)",
                          stencil ? "stencil" : "depth", static_cast<unsigned long long>(base.address), base.extent.width, base.extent.height, static_cast<int>(base.format), resource.width, resource.height, resource.format, static_cast<int>(format),
                          static_cast<unsigned>(resource.tileMode), static_cast<int>(resource.dimension), resource.baseLevel, resource.lastLevel, resource.baseArray, key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7]);
            throw std::runtime_error(text);
        }
        Require(stencil ? found->HasStencil() : found->HasDepth(), "sampling a plane the depth surface does not have");
        // The copy holds the plane as stored: a widened 16-bit depth is read as its 32-bit float values.
        const auto copyFormat = stencil ? VK_FORMAT_R8_UINT : found->PlaneBytes(false) == 2u ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
        const std::array<PlaneSlice, 1> slices{{{found, stencil, found == drawnDepth ? 0u : found->written}}};
        auto& copy = planeCopies()[{DeviceKey(context), resource.baseAddress, base.extent.width, base.extent.height, copyFormat, 1u}];
        if (copy == nullptr) copy = std::make_unique<PlaneCopy>(context, base.extent, copyFormat, 1u);
        return copy->Refresh(slices, resource, components, resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
    }
    Require(!found->Widened(), "sampling several slices or an integer view of a depth plane that the device stores in another depth format is not implemented");
    const auto copyFormat = integer ? VK_FORMAT_R16_UINT : d16 ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
    const auto layers = cube ? 6u : layered ? resource.depthOrLastArray + 1u : 1u;
    if ((format != (integer ? VK_FORMAT_R16_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT) && !depthBits) || resource.baseLevel != 0 || resource.lastLevel != 0 || resource.baseArray != 0 || (cube && resource.depthOrLastArray != 0 && resource.depthOrLastArray != 5) || (resource.dimension != TextureDimension::k2D && resource.dimension != TextureDimension::k2DArray && !cube)) {
        char text[320];
        std::snprintf(text, sizeof(text), "AGC graphics: sampling the depth planes of depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u texture of guest format %u (vk %d), dimension %d, %u slices, levels %u-%u, base slice %u is not implemented", static_cast<unsigned long long>(base.address), base.extent.width, base.extent.height, static_cast<int>(base.format), resource.width, resource.height, resource.format, static_cast<int>(format), static_cast<int>(resource.dimension), layers, resource.baseLevel, resource.lastLevel, resource.baseArray);
        throw std::runtime_error(text);
    }
    Require(base.samples == 1, "sampling a multisampled depth surface is not implemented");
    const auto key = DeviceKey(context);
    std::vector<PlaneSlice> slices;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const auto address = base.address + static_cast<std::uint64_t>(layer) * DepthSliceBytes(base.extent, d16 ? 2u : 4u);
        const auto& list = surfaces();
        const auto slice = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->device == key && surface->target.address == address && surface->target.extent.width == base.extent.width && surface->target.extent.height == base.extent.height && surface->target.format == base.format && surface->target.samples == 1; });
        if (slice == list.rend()) {
            slices.push_back({});
            continue;
        }
        ApplyFastClear(context, **slice);
        slices.push_back({slice->get(), false, slice->get() == drawnDepth ? 0u : (*slice)->written});
    }
    auto& copy = planeCopies()[{key, base.address, base.extent.width, base.extent.height, copyFormat, layers}];
    if (copy == nullptr) copy = std::make_unique<PlaneCopy>(context, base.extent, copyFormat, layers);
    return copy->Refresh(slices, resource, components, cube || resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
}

// Upstream answers for the views of the depth image it hands out; every texture this twin hands out is
// a plane copy refreshed per use (Texture::RefreshedPerUse), which a cached binding must not keep.
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
    for (const auto& surface : surfaces()) {
        const auto& target = surface->target;
        if (!HtileFillCovers(target.htileAddress, target.extent, address, bytes)) continue;
        const VkImageAspectFlags written = DepthAspect | (target.htileStencil ? StencilAspect : 0u);
        surface->pendingClear = (surface->pendingClear & ~written) | HtileFillClears(pattern, target.htileStencil);
    }
}

bool DepthSurfaceAt(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource) {
    std::lock_guard lock(surfacesMutex());
    const auto* found = FindSurface(context, resource.baseAddress);
    if (found == nullptr) return false;
    const auto& target = found->target;
    if (resource.width != target.extent.width || resource.height != target.extent.height) return false;
    const bool stencil = target.stencilAddress != 0 && target.stencilAddress == resource.baseAddress;
    const bool d16 = found->D16();
    const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    return ResolveTextureFormat(resource.format) == expected || depthBits;
}

std::shared_ptr<StorageTexture> DepthSurfaceStorage(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip) {
    std::lock_guard lock(surfacesMutex());
    auto* found = FindSurface(context, resource.baseAddress);
    return found == nullptr ? nullptr : SurfaceStorage(context, *found, words, resource, mip);
}

bool DepthSurfaceStoreBack(const Context& context, const StorageTexture& storage) {
    std::lock_guard lock(surfacesMutex());
    const auto key = DeviceKey(context);
    for (const auto& surface : surfaces()) {
        if (surface->device == key && SurfaceStoreBack(context, *surface, storage)) return true;
    }
    return false;
}

bool DepthSurfaceAt(std::uint64_t address) {
    std::lock_guard lock(surfacesMutex());
    return std::any_of(surfaces().begin(), surfaces().end(), [&](const auto& surface) { return surface->target.address == address || surface->target.stencilAddress == address; });
}

namespace Metal {

DepthSurfaceUse AcquireDepthSurface(id<MTLDevice> device, const DepthTarget& target, MTLPixelFormat format, bool* created) {
    std::lock_guard lock(surfacesMutex());
    *created = false;
    const auto key = (__bridge const void*)device;
    const bool cleared = target.htileAddress != 0 && clearedHtiles().erase(target.htileAddress) != 0;
    const VkImageAspectFlags aspects = DepthAspect | (target.stencilAddress != 0 ? StencilAspect : 0u);
    for (const auto& surface : surfaces()) {
        if (surface->device != key || !sameSurface(surface->target, target) || surface->format != format) continue;
        surface->target.clearDepth = target.clearDepth;
        surface->target.clearStencil = target.clearStencil;
        if (cleared) surface->pendingClear |= DepthAspect;
        const auto clears = surface->pendingClear & aspects;
        surface->pendingClear = 0;
        // The draw writes it (upstream's DepthSurfaceView bumps the serial and names the drawn surface).
        surface->written = ++depthWrites;
        drawnDepth = surface.get();
        return {surface->texture, (clears & DepthAspect) != 0, (clears & StencilAspect) != 0, target.clearDepth, target.clearStencil};
    }
    id<MTLTexture> texture = MakeSurfaceTexture(device, target, format);
    // Made cleared to the target's clear values (by the pass's load actions), as upstream clears a new surface.
    surfaces().push_back(std::make_unique<Surface>(key, target, format, texture));
    drawnDepth = surfaces().back().get();
    *created = true;
    return {texture, true, (aspects & StencilAspect) != 0, target.clearDepth, target.clearStencil};
}

std::vector<id<MTLTexture>> DepthSurfaceTextures(id<MTLDevice> device) {
    std::lock_guard lock(surfacesMutex());
    std::vector<id<MTLTexture>> textures;
    for (const auto& surface : surfaces()) {
        if (surface->device == (__bridge const void*)device) textures.push_back(surface->texture);
    }
    return textures;
}

}

}
