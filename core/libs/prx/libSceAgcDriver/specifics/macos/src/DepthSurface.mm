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
//
// Upstream sync 3 (merge of upstream/main 721db55dc): a depth plane is shared across its stencil pairings
// (80d8bd06b: the plane owners; a surface that takes over a plane copies it from its last owner with
// a Metal blit, CopyAspectFrom), a new surface without HTILE starts from the storage image written over
// its plane (7d898ae1a, TakeStorage), each slice of a depth array has its own HTILE (8fec47a4b,
// NoteHtileDepthClear with the fill's size), a storage image over a 2D array of depth slices copies every
// slice (696d1660c dd7d649c3, LayeredDepthStorage), a covering storage image is stored back into the
// newest pairing of its plane (a84afe931 b3e5acc8c), depth plane arrays are sampled from a later base
// slice (62eff52c4) and at the descriptor's layer offsets (03b7cb5a1), plane copies go through the
// texture's padded linear slices when guest layers fill them (6b480cc2b), and uploads count toward the
// recorder's kept bytes (875ecc139). DepthSurfacesSerial (d9e47d1e9) counts surface creations.
//
// Upstream sync 4 (merge of upstream/main 5d9dcc42d2): a surface whose memory another writer covers is
// retired (626cff7d26: RetireDepthSurfaces, and a sampled read the surface cannot serve over memory
// the guest rewrote since its last bind), a fast clear is taken from an HTILE any kernel wrote
// (4ddff7c6e5 3e61fcf1f9 cc022df348 139e3b86ed: storeBackClearedHtile, read and expanded outside the
// surfaces lock before the bind), a stencil plane is sampled as R8_UNORM (47f7118241: an R8_UNORM plane
// copy) and HtileDepthClearBytes / the resident-depth guard registry are upstream's (7d0dd69f0a
// 9e27d0b623 da9fe3c708). Upstream's native multisampled depth sampling (ee1ba73a2f f56e3bc467
// 6972b5ae07) is not ported: sampling a multisampled depth surface still throws here (the Metal
// surface is a D28 multisample array that no sampled view describes yet), so no surface registers a
// resident-depth guard and the native-owner checks have nothing to refuse.
//
// Upstream sync 5 (merge of upstream/main 0d078f75d3): a fill of a whole depth or stencil plane clears
// the surface to the filled value (3e1b7be111: NotePlaneFill, the pending clear's depthFill /
// stencilFill), a depth plane's UAV clear kernel (MatchStorageImageClear, DepthPlaneClear) skips the
// plane copy into the storage image and stores back as a clear pass of the surface (c05b2567be:
// StoreOverwrite), and a view of a later level of a depth surface is memory (92f170c673: no surface
// serves it). Native multisampled depth sampling is still not ported: none of these build on it.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalCommandStream.h"
#include "MetalImages.h"
#include "MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
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
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <span>
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

// Upstream c05b2567be: a depth plane's UAV clear kernel (MatchStorageImageClear) installs the value it
// writes for the dispatch (DepthPlaneClear); the plane's storage view then skips the copy in and stores
// back as a clear of the surface.
std::atomic<bool> depthPlaneClearEnabled{true};

enum DepthClearDecline : std::size_t { DeclineLayers, DeclineFormat, DeclineValue, DeclineColor, DeclinePending, DeclineCoverage, DeclineSize, DeclineCount };
static_assert(DeclineCount <= 8, "DepthClearStatistics::declined holds eight reasons");

struct DepthClearCounters {
    std::atomic<std::uint64_t> matched{0};
    std::atomic<std::uint64_t> taken{0};
    std::atomic<std::uint64_t> copiesSkipped{0};
    std::atomic<std::uint64_t> clears{0};
    std::atomic<std::uint64_t> bytesSkipped{0};
    std::array<std::atomic<std::uint64_t>, 8> declined{};
};

DepthClearCounters& depthClearCounters() {
    static DepthClearCounters counters;
    return counters;
}

thread_local DepthPlaneOverwrite* currentOverwrite = nullptr;

std::optional<DepthPlaneOverwrite> decodeOverwrite(const StorageImageClear& clear) {
    auto& counters = depthClearCounters();
    ++counters.matched;
    const auto decline = [&](DepthClearDecline why) -> std::optional<DepthPlaneOverwrite> {
        ++counters.declined[why];
        return std::nullopt;
    };
    DepthPlaneOverwrite overwrite;
    overwrite.image = clear.image;
    try {
        overwrite.view = DecodeTextureResource(clear.image);
    } catch (...) {
        return decline(DeclineFormat);
    }
    const auto& view = overwrite.view;
    const bool oneLayer = view.dimension == TextureDimension::k2D || (view.dimension == TextureDimension::k2DArray && view.baseArray == 0 && view.depthOrLastArray == 0);
    if (!oneLayer || view.baseLevel != 0) return decline(DeclineLayers);
    VkFormat format = VK_FORMAT_UNDEFINED;
    try {
        format = ResolveTextureFormat(view.format);
    } catch (...) {
        return decline(DeclineFormat);
    }
    enum class Kind { Unorm16, Uint16, Float32, Uint32 };
    Kind kind;
    switch (format) {
        case VK_FORMAT_R16_UNORM: kind = Kind::Unorm16; break;
        case VK_FORMAT_R16_UINT: kind = Kind::Uint16; break;
        case VK_FORMAT_R32_SFLOAT: kind = Kind::Float32; break;
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32_SINT: kind = Kind::Uint32; break;
        default: return decline(DeclineFormat);
    }
    const std::uint64_t base = clear.color[0] | (static_cast<std::uint64_t>(clear.color[1] & 0xffffu) << 32u);
    const std::uint64_t stride = (clear.color[1] >> 16u) & 0x3fffu;
    const std::uint64_t extent = stride == 0 ? clear.color[2] : static_cast<std::uint64_t>(clear.color[2]) * stride;
    if (base == 0 || (base & 3u) != 0u || extent < 16u || !GuestMemory::Accessible(reinterpret_cast<const void*>(base), 16)) return decline(DeclineColor);
    if (const auto* recorder = Recorder::Active(); recorder != nullptr && recorder->PendingWriteOverlaps(base, 16)) return decline(DeclinePending);
    const std::pair<std::uint64_t, std::uint64_t> colorRange{base, base + 16u};
    if (StorageTexture::AnyPendingOverlaps(std::span(&colorRange, 1))) return decline(DeclinePending);
    std::uint32_t c0 = 0;
    try {
        GuestMemory::Read(base, std::as_writable_bytes(std::span(&c0, 1)), 4);
    } catch (...) {
        return decline(DeclineColor);
    }
    switch (kind) {
        case Kind::Unorm16:
            if (c0 == 0u || c0 == 0x80000000u) overwrite.raw = 0u;
            else if (c0 == 0x3f800000u) overwrite.raw = 0xffffu;
            else return decline(DeclineValue);
            overwrite.bits = 16u;
            overwrite.depth = static_cast<float>(overwrite.raw) / 65535.0f;
            break;
        case Kind::Uint16:
            if (c0 != 0u && c0 != 0xffffu) return decline(DeclineValue);
            overwrite.raw = c0;
            overwrite.bits = 16u;
            overwrite.depth = static_cast<float>(overwrite.raw) / 65535.0f;
            break;
        case Kind::Float32:
        case Kind::Uint32: {
            const auto depth = std::bit_cast<float>(c0);
            if (c0 != 0u && !(std::isnormal(depth) && depth > 0.0f && depth <= 1.0f)) return decline(DeclineValue);
            overwrite.raw = c0;
            overwrite.bits = 32u;
            overwrite.depth = depth;
            break;
        }
    }
    overwrite.colorAddress = base;
    overwrite.colorBits = c0;
    overwrite.width = clear.width;
    overwrite.height = clear.height;
    return overwrite;
}

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
    // Upstream 3e1b7be111: the value a fill of the whole plane left (the pending clear takes it
    // instead of the target's clear value).
    std::optional<float> depthFill;
    std::optional<std::uint8_t> stencilFill;
    std::uint64_t written = ++depthWrites;
    // DepthSurfaceView's handle of the texture (made on its first call).
    VkImageView view = VK_NULL_HANDLE;
    PlaneStorage depthPlane;
    PlaneStorage stencilPlane;
    // Upstream 626cff7d26: retired once another writer covers its memory (RetireDepthSurfaces) or a read
    // it cannot serve finds its memory rewritten since its last bind; lookups skip it until it is bound
    // again. depthMemory / stencilMemory: the planes' guest write generations at the last bind.
    bool retired = false;
    std::uint64_t depthMemory = 0;
    std::uint64_t stencilMemory = 0;

    std::uint64_t DepthBytes() const { return DepthSliceBytes(target.extent, D16() ? 2u : 4u); }
    std::uint64_t StencilBytes() const { return DepthSliceBytes(target.extent, 1u); }
    bool Overlaps(std::uint64_t address, std::uint64_t bytes) const {
        const auto end = address + bytes;
        if (address < target.address + DepthBytes() && target.address < end) return true;
        return target.stencilAddress != 0 && address < target.stencilAddress + StencilBytes() && target.stencilAddress < end;
    }
    void NoteMemory() {
        depthMemory = GuestMemory::CollectWrites(target.address, static_cast<std::size_t>(DepthBytes()));
        stencilMemory = target.stencilAddress != 0 ? GuestMemory::CollectWrites(target.stencilAddress, static_cast<std::size_t>(StencilBytes())) : 0;
    }
    bool OverwrittenInMemory() const {
        GuestMemory::CollectWrites(target.address, static_cast<std::size_t>(DepthBytes()));
        if (GuestMemory::WrittenSince(target.address, static_cast<std::size_t>(DepthBytes()), depthMemory)) return true;
        if (target.stencilAddress == 0) return false;
        GuestMemory::CollectWrites(target.stencilAddress, static_cast<std::size_t>(StencilBytes()));
        return GuestMemory::WrittenSince(target.stencilAddress, static_cast<std::size_t>(StencilBytes()), stencilMemory);
    }

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

    void ForgetClear(VkImageAspectFlags aspects) {
        pendingClear &= ~aspects;
        if ((aspects & DepthAspect) != 0) depthFill.reset();
        if ((aspects & StencilAspect) != 0) stencilFill.reset();
    }
    void NoteHtileClear(VkImageAspectFlags aspects) {
        ForgetClear(aspects);
        pendingClear |= aspects;
    }
    void NotePlaneFill(VkImageAspectFlags aspect, float depth, std::uint8_t stencil) {
        pendingClear |= aspect;
        if (aspect == DepthAspect) depthFill = depth;
        else stencilFill = stencil;
    }
    // The pending clear's values (a plane fill's, else the target's); taking them forgets the fills.
    std::pair<float, std::uint8_t> TakeClearValues() {
        const std::pair<float, std::uint8_t> values{depthFill.value_or(target.clearDepth), stencilFill.value_or(target.clearStencil)};
        depthFill.reset();
        stencilFill.reset();
        return values;
    }
};

// The surface the last draw attached (upstream's drawnDepth): its plane copies are always refreshed, the
// draw being recorded may still write it.
Surface* drawnDepth = nullptr;

bool sameSurface(const DepthTarget& a, const DepthTarget& b) {
    return a.address == b.address && a.stencilAddress == b.stencilAddress && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && a.samples == b.samples && (a.samples == 1u || (a.htileAddress == b.htileAddress && a.htileStencil == b.htileStencil && a.depthSwizzle == b.depthSwizzle && a.pipeBankXor == b.pipeBankXor && a.firstSlice == b.firstSlice && a.lastSlice == b.lastSlice && a.iterate256 == b.iterate256 && a.htilePipeAligned == b.htilePipeAligned && a.htileVrsEncoding == b.htileVrsEncoding && a.htileDepthOnly == b.htileDepthOnly));
}

// Upstream's resident-depth guards (ee1ba73a2f da9fe3c708): the guest ranges of native multisampled
// depth owners, which guest reads must not publish. No Metal surface registers one yet (sampling a
// multisampled depth surface throws, see the file comment), so the registry stays empty and the
// public guard functions below answer for an empty set exactly as upstream does.
struct DepthGuard {
    VkDevice device;
    std::uint64_t address;
    std::size_t bytes;
    std::shared_ptr<std::atomic<bool>> active;
    std::vector<GuestMemory::UnchangedQuery> live;
};

std::shared_mutex& depthGuardMutex() {
    static std::shared_mutex mutex;
    return mutex;
}

std::vector<DepthGuard>& depthGuards() {
    static auto* ranges = new std::vector<DepthGuard>();
    return *ranges;
}

std::atomic<bool>& anyDepthGuard() {
    static std::atomic<bool> any{false};
    return any;
}

std::mutex& surfacesMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::unique_ptr<Surface>>& surfaces() {
    static auto* list = new std::vector<std::unique_ptr<Surface>>();
    return *list;
}

// HTILE clears noted before their surface exists: the fill's address and size (upstream 8fec47a4b,
// each slice of a depth array has its own HTILE inside one fill).
std::map<std::uint64_t, std::size_t>& clearedHtiles() {
    static auto* cleared = new std::map<std::uint64_t, std::size_t>();
    return *cleared;
}

bool takeClearedHtile(const DepthTarget& target) {
    if (target.htileAddress == 0) return false;
    auto& cleared = clearedHtiles();
    const auto found = std::find_if(cleared.begin(), cleared.end(), [&](const auto& entry) { return HtileFillCovers(target.htileAddress, target.extent, entry.first, entry.second); });
    if (found == cleared.end()) return false;
    cleared.erase(found);
    return true;
}

// Upstream 80d8bd06b: the surface that last bound each depth and stencil plane (keyed by device and
// plane address) and the order of the binds. Surfaces live until ClearDepthSurfaces, which drops these.
struct PlaneOwner {
    Surface* surface = nullptr;
    std::uint64_t serial = 0;
};

using PlaneOwners = std::map<std::pair<const void*, std::uint64_t>, PlaneOwner>;

std::uint64_t planeBinds = 0;

PlaneOwners& depthOwners() {
    static auto* owners = new PlaneOwners();
    return *owners;
}

PlaneOwners& stencilOwners() {
    static auto* owners = new PlaneOwners();
    return *owners;
}

// Upstream d9e47d1e9: changes when a surface is made or the device's surfaces are dropped.
std::atomic<std::uint64_t> surfacesSerial{0};

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
    void Keep(std::shared_ptr<void> object, std::size_t bytes = 0) {
        if (recorder != nullptr) recorder->Keep(std::move(object), bytes);
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

// Copies one plane of the surface (all of it, slice 0) to or from `buffer` at `offset`, rows `rowBytes`
// apart (0: tightly packed; a guest texture's padded linear pitch, upstream 6b480cc2b).
void CopySurfacePlane(VkCommandBuffer commands, const Surface& surface, bool stencil, VkBuffer buffer, std::uint64_t offset, bool toBuffer, std::uint32_t rowBytes = 0) {
    auto& stream = Metal::Stream(commands);
    auto encoder = stream.Compute();
    const auto bytes = surface.PlaneBytes(stencil);
    const NSUInteger width = surface.target.extent.width;
    const NSUInteger height = surface.target.extent.height;
    const NSUInteger bytesPerRow = rowBytes != 0 ? NSUInteger{rowBytes} : width * bytes;
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
void ClearSurfaceValue(const Context& context, Surface& surface, VkImageAspectFlags aspects, float depth, std::uint8_t stencil);

void ApplyFastClear(const Context& context, Surface& surface) {
    const auto aspects = surface.pendingClear & ((surface.HasDepth() ? DepthAspect : 0u) | (surface.HasStencil() ? StencilAspect : 0u));
    surface.pendingClear = 0;
    const auto [depth, stencil] = surface.TakeClearValues();
    if (aspects == 0) return;
    ClearSurfaceValue(context, surface, aspects, depth, stencil);
}

// Upstream's RecordClearValue: `aspects` of the surface cleared to `depth` / `stencil` (a clear-only
// render pass per slice).
void ClearSurfaceValue(const Context& context, Surface& surface, VkImageAspectFlags aspects, float depth, std::uint8_t stencil) {
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
            pass.depthAttachment.clearDepth = depth;
        }
        if (surface.HasStencil()) {
            pass.stencilAttachment.texture = surface.texture;
            pass.stencilAttachment.slice = slice;
            pass.stencilAttachment.loadAction = (aspects & StencilAspect) != 0 ? MTLLoadActionClear : MTLLoadActionLoad;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
            pass.stencilAttachment.clearStencil = stencil;
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

// Upstream's DepthSurface::Storage / copyPlane / StoreBack over a Metal surface. `storage` is the plane's
// storage image or (StoreBackCovering, upstream a84afe931) another storage image covering the plane.
void CopyPlane(const Context& context, Surface& surface, PlaneStorage& plane, const std::shared_ptr<StorageTexture>& storage, bool stencil, bool toStorage) {
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "depth surface storage copies need an active recorder");
    const auto& target = surface.target;
    const auto bytes = static_cast<std::size_t>(target.extent.width) * target.extent.height * surface.PlaneBytes(stencil);
    if (plane.staging == nullptr) plane.staging = std::make_shared<DeviceBuffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const auto commands = recorder->Commands();
    recorder->Keep(storage);
    recorder->Keep(plane.staging);
    const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
    VkBufferImageCopy colorRegion{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    if (toStorage) CopySurfacePlane(commands, surface, stencil, plane.staging->Handle(), 0, true);
    else Metal::Vk::CmdCopyImageToBuffer(commands, storage->Image(), VK_IMAGE_LAYOUT_GENERAL, plane.staging->Handle(), 1, &colorRegion);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    if (toStorage) Metal::Vk::CmdCopyBufferToImage(commands, plane.staging->Handle(), storage->Image(), VK_IMAGE_LAYOUT_GENERAL, 1, &colorRegion);
    else CopySurfacePlane(commands, surface, stencil, plane.staging->Handle(), 0, false);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
    recorder->EndGpuTiming(timing, bytes);
}

// Upstream's DepthSurface::StorageViewMatches: whether a storage view of the plane at the resource's
// address reads the plane's values (its extent, and the plane's format or a depth-bits view).
bool StorageViewMatches(const Surface& surface, std::span<const std::uint32_t> words, const GuestTextureResource& resource) {
    const auto& target = surface.target;
    if (resource.width != target.extent.width || resource.height != target.extent.height) return false;
    const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
    const bool d16 = surface.D16();
    const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    return ResolveTextureFormat(resource.format) == expected || depthBits;
}

// Upstream's DepthSurface::RequireStorable.
// Upstream 92f170c673: a later level is no surface's (DepthSurfaceStorage returns none for it).
void RequireStorable(const Context& context, const Surface& surface, const GuestTextureResource& resource) {
    const auto& target = surface.target;
    const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
    Require(target.samples == 1, "storage image access to a multisampled depth surface is not implemented");
    if (!stencil && surface.Widened()) {
        char text[224];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the depth plane of depth surface 0x%llx (vk format %d) stored as vk format %d is not implemented", static_cast<unsigned long long>(target.address), static_cast<int>(target.format), static_cast<int>(VK_FORMAT_D32_SFLOAT_S8_UINT));
        throw std::runtime_error(text);
    }
    Require(stencil ? surface.HasStencil() : surface.HasDepth(), "storage image access to a plane the depth surface does not have");
    Require(context.detiler != nullptr, "depth surface storage requires a texture detiler");
}

// Upstream's DepthSurface::OverwritesWholePlane / PlaneOverwritten / StoreOverwrite (c05b2567be).
bool OverwritesWholePlane(const Surface& surface, const DepthPlaneOverwrite& overwrite) {
    return overwrite.bits == (surface.D16() ? 16u : 32u) && overwrite.width >= surface.target.extent.width && overwrite.height >= surface.target.extent.height;
}

DepthPlaneOverwrite* PlaneOverwritten(const Surface& surface, std::span<const std::uint32_t> words) {
    auto* overwrite = currentOverwrite;
    if (overwrite == nullptr || words.size() < overwrite->image.size() || !std::equal(overwrite->image.begin(), overwrite->image.end(), words.begin())) return nullptr;
    auto& declined = depthClearCounters().declined;
    if (overwrite->bits != (surface.D16() ? 16u : 32u)) {
        ++declined[DeclineSize];
        return nullptr;
    }
    if (overwrite->width < surface.target.extent.width || overwrite->height < surface.target.extent.height) {
        ++declined[DeclineCoverage];
        return nullptr;
    }
    return overwrite;
}

// The storage view's store-back as a clear pass of the surface's depth plane, when the dispatch's clear
// kernel overwrote the whole plane with one value that the guest still holds.
bool StoreOverwrite(const Context& context, Surface& surface, const StorageTexture& mirror) {
    auto* overwrite = currentOverwrite;
    if (overwrite == nullptr || !OverwritesWholePlane(surface, *overwrite) || !surface.HasDepth() || surface.Widened()) return false;
    const auto& described = mirror.Descriptor();
    const auto& view = overwrite->view;
    if (described.baseAddress != view.baseAddress || described.format != view.format || described.tileMode != view.tileMode || described.dimension != view.dimension) return false;
    std::uint32_t current = 0;
    try {
        GuestMemory::Read(overwrite->colorAddress, std::as_writable_bytes(std::span(&current, 1)), 4);
    } catch (...) {
        return false;
    }
    if (current != overwrite->colorBits) return false;
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "depth surface storage clears need an active recorder");
    const auto bytes = static_cast<std::size_t>(surface.target.extent.width) * surface.target.extent.height * (surface.D16() ? 2u : 4u);
    recorder->Commands();
    const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
    // The plane's pending clear is forgotten: the clear pass below supersedes it.
    surface.ForgetClear(DepthAspect);
    ClearSurfaceValue(context, surface, DepthAspect, overwrite->depth, surface.target.clearStencil);
    recorder->EndGpuTiming(timing, bytes);
    auto& counters = depthClearCounters();
    ++counters.clears;
    counters.bytesSkipped += 2u * bytes;
    return true;
}

std::shared_ptr<StorageTexture> SurfaceStorage(const Context& context, Surface& surface, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip) {
    if (!StorageViewMatches(surface, words, resource)) return nullptr;
    const auto& target = surface.target;
    const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
    const bool oneLayer = resource.dimension == TextureDimension::k2DArray && resource.baseArray == 0 && resource.depthOrLastArray == 0;
    if (resource.dimension != TextureDimension::k2D && !oneLayer) {
        char text[256];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image access to depth/stencil surface 0x%llx is not implemented (storage=%ux%u guest format=%u mip=%u dimension=%d tile=%d)", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, mip, static_cast<int>(resource.dimension), static_cast<int>(resource.tileMode));
        throw std::runtime_error(text);
    }
    RequireStorable(context, surface, resource);
    auto* overwrite = stencil ? nullptr : PlaneOverwritten(surface, words);
    ApplyFastClear(context, surface);
    auto& plane = stencil ? surface.stencilPlane : surface.depthPlane;
    if (plane.storage == nullptr || plane.storage->Descriptor().format != resource.format || plane.storage->Descriptor().tileMode != resource.tileMode || plane.storage->Descriptor().baseAddress != resource.baseAddress || plane.storage->Descriptor().dimension != resource.dimension) {
        std::erase_if(plane.replaced, [](const auto& old) { return old.expired(); });
        if (plane.storage != nullptr) plane.replaced.push_back(plane.storage);
        plane.storage = StorageTexture::Create(context, *context.detiler, resource, 0);
    }
    if (overwrite != nullptr) ++depthClearCounters().copiesSkipped;
    else CopyPlane(context, surface, plane, plane.storage, stencil, true);
    return plane.storage;
}

bool SurfaceStoreBack(const Context& context, Surface& surface, const StorageTexture& candidate) {
    for (const bool stencil : {false, true}) {
        auto& plane = stencil ? surface.stencilPlane : surface.depthPlane;
        if (plane.storage.get() == &candidate) {
            if (stencil || !StoreOverwrite(context, surface, candidate)) CopyPlane(context, surface, plane, plane.storage, stencil, false);
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

// Upstream's DepthSurface::CoversPlane (a84afe931): a storage image of the whole plane, one level and one
// layer, in the plane's format, that the surface stores in the guest's format.
bool CoversPlane(const Surface& surface, const StorageTexture& storage, bool stencil) {
    const auto& target = surface.target;
    const auto& resource = storage.Descriptor();
    const auto address = stencil ? target.stencilAddress : target.address;
    if (address == 0 || resource.baseAddress != address || resource.width != target.extent.width || resource.height != target.extent.height || resource.baseLevel != 0 || resource.mipCount != 1) return false;
    if (resource.dimension != TextureDimension::k2D && (resource.dimension != TextureDimension::k2DArray || resource.baseArray != 0 || resource.depthOrLastArray != 0)) return false;
    if (target.samples != 1 || (!stencil && surface.Widened()) || !(stencil ? surface.HasStencil() : surface.HasDepth())) return false;
    return ResolveTextureFormat(resource.format) == (stencil ? VK_FORMAT_R8_UINT : surface.D16() ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT);
}

// Upstream's DepthSurface::StoreBackCovering (a84afe931): a storage image written over the plane that
// is not the plane's own storage view is stored into the plane.
bool SurfaceStoreBackCovering(const Context& context, Surface& surface, const StorageTexture& candidate) {
    for (const bool stencil : {false, true}) {
        if (!CoversPlane(surface, candidate, stencil)) continue;
        auto& plane = stencil ? surface.stencilPlane : surface.depthPlane;
        const auto source = std::const_pointer_cast<StorageTexture>(candidate.shared_from_this());
        CopyPlane(context, surface, plane, source, stencil, false);
        surface.written = ++depthWrites;
        return true;
    }
    return false;
}

// Upstream's storageFits (7d898ae1a): a storage image a new surface's depth plane can start from.
bool storageFits(const DepthTarget& target, const GuestTextureResource& descriptor) {
    const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto format = ResolveTextureFormat(descriptor.format);
    const bool sized = d16 ? (format == VK_FORMAT_R16_UINT || format == VK_FORMAT_R16_UNORM || format == VK_FORMAT_R16_SINT || format == VK_FORMAT_R16_SNORM || format == VK_FORMAT_R16_SFLOAT) : (format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT);
    return sized && descriptor.width == target.extent.width && descriptor.height == target.extent.height && descriptor.mipCount == 1 && (descriptor.dimension == TextureDimension::k2D || descriptor.dimension == TextureDimension::k2DArray);
}

// Upstream's DepthSurface::TakeStorage (7d898ae1a): a new single-sampled surface stored in the guest's
// format starts its depth plane from the storage image (a pending one first) written over the plane's
// memory, layer 0, instead of DB_DEPTH_CLEAR (its depth load action becomes Load).
void TakeStorage(const Context& context, Surface& surface) {
    const auto& target = surface.target;
    if (target.samples != 1 || surface.Widened() || !surface.HasDepth()) return;
    const bool d16 = surface.D16();
    const auto bytes = DepthSliceBytes(target.extent, d16 ? 2u : 4u);
    auto storage = StorageTexture::FindPending(target.address, bytes);
    if (storage == nullptr) storage = StorageTexture::FindLive(target.address, bytes);
    if (storage == nullptr || !storageFits(target, storage->Descriptor())) return;
    static_cast<void>(storage->Refresh());
    surface.ForgetClear(DepthAspect);
    auto& plane = surface.depthPlane;
    const auto planeBytes = static_cast<std::size_t>(target.extent.width) * target.extent.height * surface.PlaneBytes(false);
    if (plane.staging == nullptr) plane.staging = std::make_shared<DeviceBuffer>(context, planeBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    SurfaceCommands commands(context);
    commands.Keep(storage);
    commands.Keep(plane.staging);
    VkBufferImageCopy colorRegion{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
    RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    Metal::Vk::CmdCopyImageToBuffer(commands.Handle(), storage->Image(), VK_IMAGE_LAYOUT_GENERAL, plane.staging->Handle(), 1, &colorRegion);
    RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    CopySurfacePlane(commands.Handle(), surface, false, plane.staging->Handle(), 0, false);
    RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    if (commands.Recording() != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
    commands.Finish();
}

// Upstream's DepthSurface::CopyAspectFrom (80d8bd06b) with Metal blits: one plane of `source` into the
// same plane of `surface`. A texture-to-texture copy when both store the plane alone in the same pixel
// format and sample count (multisampled surfaces too); otherwise, single-sampled, through the
// destination plane's staging buffer with the depth/stencil blit options (a plane of a combined
// Depth32Float_Stencil8 texture). False (nothing recorded, the caller keeps the plane as it is: a new
// surface starts from its clear, as before this port) when Metal cannot copy the plane: a multisampled
// combined depth/stencil pairing, or a 16-bit depth plane paired once as Depth16Unorm and once widened
// to Depth32Float_Stencil8 (that needs a converting kernel).
bool CopyAspectFrom(const Context& context, Surface& surface, const Surface& source, bool stencil) {
    if (stencil ? !(surface.HasStencil() && source.HasStencil()) : !(surface.HasDepth() && source.HasDepth())) return false;
    const auto single = [](MTLPixelFormat format) { return format != MTLPixelFormatDepth32Float_Stencil8; };
    const bool whole = surface.format == source.format && single(surface.format) && surface.texture.sampleCount == source.texture.sampleCount && surface.texture.arrayLength == source.texture.arrayLength && surface.texture.textureType == source.texture.textureType;
    const bool planar = surface.target.samples == 1 && source.target.samples == 1 && surface.PlaneBytes(stencil) == source.PlaneBytes(stencil);
    if (!whole && !planar) return false;
    surface.ForgetClear(stencil ? StencilAspect : DepthAspect);
    SurfaceCommands commands(context);
    constexpr VkAccessFlags all = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, all, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    std::uint32_t barriers = 2;
    if (whole) {
        auto& stream = Metal::Stream(commands.Handle());
        [stream.Compute() copyFromTexture:source.texture toTexture:surface.texture];
        stream.CountCommand();
        Metal::NoteImageCopy(surface.Image(), false);
    } else {
        auto& plane = stencil ? surface.stencilPlane : surface.depthPlane;
        const auto bytes = static_cast<std::size_t>(surface.target.extent.width) * surface.target.extent.height * surface.PlaneBytes(stencil);
        if (plane.staging == nullptr) plane.staging = std::make_shared<DeviceBuffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        commands.Keep(plane.staging);
        CopySurfacePlane(commands.Handle(), source, stencil, plane.staging->Handle(), 0, true);
        RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        CopySurfacePlane(commands.Handle(), surface, stencil, plane.staging->Handle(), 0, false);
        barriers = 3;
    }
    RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, all);
    if (commands.Recording() != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, barriers);
    commands.Finish();
    return true;
}

// Upstream's takePlane / takePlanes (80d8bd06b): a surface binding a plane that another pairing of the
// same format, sample count and extent bound last takes its contents (its pending clear, or a copy).
// Without the device's context (no copy can be recorded) the plane is kept as it is.
void takePlane(const Context* context, Surface& surface, PlaneOwners& owners, std::uint64_t address, bool stencil) {
    auto& entry = owners[{surface.device, address}];
    auto* owner = entry.surface;
    if (owner != nullptr && owner != &surface && owner->target.format == surface.target.format && owner->target.samples == surface.target.samples && owner->target.extent.width == surface.target.extent.width && owner->target.extent.height == surface.target.extent.height) {
        const VkImageAspectFlags aspect = stencil ? StencilAspect : DepthAspect;
        if ((owner->pendingClear & aspect) != 0) {
            surface.pendingClear |= aspect;
            if (stencil) surface.stencilFill = owner->stencilFill;
            else surface.depthFill = owner->depthFill;
        } else if (context != nullptr) static_cast<void>(CopyAspectFrom(*context, surface, *owner, stencil));
    }
    entry = {&surface, ++planeBinds};
}

void takePlanes(const Context* context, Surface& surface) {
    takePlane(context, surface, depthOwners(), surface.target.address, false);
    if (surface.target.stencilAddress != 0) takePlane(context, surface, stencilOwners(), surface.target.stencilAddress, true);
}

// Upstream's planeSurface (80d8bd06b): the surface that bound the plane at `address` last (the newer of
// its depth and stencil binds), else the newest surface with a plane there.
Surface* planeSurface(const void* key, std::uint64_t address) {
    const auto depth = depthOwners().find({key, address});
    const auto stencil = stencilOwners().find({key, address});
    const bool hasDepth = depth != depthOwners().end();
    const bool hasStencil = stencil != stencilOwners().end();
    if (hasDepth && (!hasStencil || depth->second.serial > stencil->second.serial)) return depth->second.surface;
    if (hasStencil) return stencil->second.surface;
    const auto& list = surfaces();
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) {
        return surface->device == key && !surface->retired && (surface->target.address == address || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == address));
    });
    return found == list.rend() ? nullptr : found->get();
}

// Upstream's depthSlice (80d8bd06b): the surface holding the depth plane of one slice of a layered read,
// its plane's last owner first. Single-sampled only (a plane copy reads it through a buffer).
Surface* depthSlice(const void* key, std::uint64_t address, const DepthTarget& base) {
    const auto matches = [&](const Surface& surface) { return surface.target.extent.width == base.extent.width && surface.target.extent.height == base.extent.height && surface.target.format == base.format && surface.target.samples == 1; };
    if (const auto found = depthOwners().find({key, address}); found != depthOwners().end() && matches(*found->second.surface)) return found->second.surface;
    const auto& list = surfaces();
    const auto slice = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->device == key && !surface->retired && surface->target.address == address && matches(*surface); });
    return slice == list.rend() ? nullptr : slice->get();
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
        stagingSize = static_cast<std::size_t>(sliceBytes()) * layers;
        staging = std::make_shared<DeviceBuffer>(context, stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    ~PlaneCopy() {
        textures.clear();
        staging.reset();
        if (image != VK_NULL_HANDLE) Metal::Vk::DestroyImage(context.device, image, nullptr);
    }
    PlaneCopy(const PlaneCopy&) = delete;
    PlaneCopy& operator=(const PlaneCopy&) = delete;

    // `baseLayer`: the first slice the view reads (upstream 62eff52c4, a 2D array from a later base slice).
    std::shared_ptr<Texture> Refresh(std::span<const PlaneSlice> slices, const GuestTextureResource& resource, VkComponentMapping components, VkImageViewType viewType, std::uint32_t baseLayer = 0) {
        Require(slices.size() == layers, "depth plane copy slices do not match its layers");
        const bool guest = std::any_of(slices.begin(), slices.end(), [](const PlaneSlice& slice) { return slice.surface == nullptr; });
        // Guest layers are detiled into the texture's padded linear slices (upstream 6b480cc2b), and every
        // layer of the staging buffer then follows that layout; copies of surfaces alone stay tightly packed
        // (a single widened plane is read at its stored 32-bit size, which the guest pitch does not describe).
        SurfaceGeometry geometry{};
        if (guest) {
            geometry = DescribeSurface(resource);
            if (geometry.layers != layers || geometry.sliceLinearBytes < sliceBytes() || geometry.mips.empty() || geometry.mips.front().pitchBytes < extent.width * elementBytes()) {
                char text[320];
                std::snprintf(text, sizeof(text), "AGC graphics: depth plane copy geometry does not match its layers (copy %ux%u, %u layers, %llu bytes a slice; texture %ux%u guest format %u, dimension %d, tile mode %u, %u layers, %llu bytes a slice, %zu mips)", extent.width, extent.height, layers, static_cast<unsigned long long>(sliceBytes()), resource.width, resource.height, resource.format, static_cast<int>(resource.dimension), static_cast<unsigned>(resource.tileMode), geometry.layers, static_cast<unsigned long long>(geometry.sliceLinearBytes), geometry.mips.size());
                throw std::runtime_error(text);
            }
        }
        const auto stagingBytes = static_cast<std::size_t>(guest ? geometry.sliceLinearBytes : sliceBytes()) * layers;
        if (stagingBytes > stagingSize) {
            // The old buffer stays alive while recorded work holds it (SurfaceCommands::Keep).
            staging = std::make_shared<DeviceBuffer>(context, stagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            stagingSize = stagingBytes;
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
        if (!copiedLayers.empty() || !uploads.empty()) record(slices, resource, guest, geometry, copiedLayers, uploads, uploadLayers);
        const std::array<std::uint32_t, 6> key{static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a), static_cast<std::uint32_t>(viewType), baseLayer};
        auto& texture = textures[key];
        if (texture == nullptr) {
            texture = std::make_shared<Texture>(context, image, format, aspect(), components, viewType, baseLayer);
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
    std::size_t stagingSize = 0;
    std::map<std::array<std::uint32_t, 6>, std::shared_ptr<Texture>> textures;
    std::vector<GuestLayer> guestLayers;
    std::vector<SurfaceLayer> surfaceLayers;

    void record(std::span<const PlaneSlice> slices, const GuestTextureResource& resource, bool padded, const SurfaceGeometry& geometry, std::span<const std::uint32_t> copiedLayers, std::span<const std::shared_ptr<Buffer>> uploads, std::span<const std::uint32_t> uploadLayers) {
        if (!uploads.empty()) {
            Require(context.detiler != nullptr, "depth plane copy requires a texture detiler");
            context.detiler->BeginBatch();
        }
        SurfaceCommands commands(context);
        commands.Keep(staging);
        RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        std::vector<VkBufferImageCopy> regions;
        // The staging layout: the guest texture's padded linear slices when guest layers are detiled into
        // it (upstream 6b480cc2b), tightly packed slices otherwise.
        const auto offset = [&](std::uint32_t layer) { return padded ? geometry.LinearLayerOffset(layer) + geometry.mips.front().linearOffset : sliceBytes() * layer; };
        const std::uint32_t pitch = padded ? geometry.mips.front().pitchBytes : 0u;
        const auto region = [&](std::uint32_t layer) {
            VkBufferImageCopy copy{};
            copy.bufferOffset = offset(layer);
            copy.bufferRowLength = pitch / elementBytes();
            copy.imageSubresource = {aspect(), 0, layer, 1};
            copy.imageExtent = {extent.width, extent.height, 1};
            return copy;
        };
        for (const auto layer : copiedLayers) {
            CopySurfacePlane(commands.Handle(), *slices[layer].surface, slices[layer].stencil, staging->Handle(), offset(layer), true, pitch);
            regions.push_back(region(layer));
        }
        if (!uploads.empty()) {
            RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            for (std::size_t upload = 0; upload < uploads.size(); ++upload) {
                const auto layer = uploadLayers[upload];
                context.detiler->Dispatch(commands.Handle(), resource.tileMode, elementBytes(), uploads[upload]->Handle(), 0, staging->Handle(), geometry.LinearLayerOffset(layer), geometry.mips.front(), false, layer, geometry.thick);
                regions.push_back(region(layer));
                commands.Keep(uploads[upload], uploads[upload]->Bytes().size());
            }
        }
        RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        Metal::Vk::CmdCopyBufferToImage(commands.Handle(), staging->Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()), regions.data());
        RecordMemoryBarrier(context, commands.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        if (commands.Recording() != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, 3);
        commands.Finish();
    }

    std::uint32_t elementBytes() const {
        return format == VK_FORMAT_D32_SFLOAT ? 4u : format == VK_FORMAT_R8_UINT || format == VK_FORMAT_R8_UNORM ? 1u : 2u;
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

// One slice of a layered depth storage view: the surface holding it and its layer in the view.
struct LayeredSlice {
    Surface* surface;
    std::uint32_t layer;
};

// Upstream's LayeredDepthStorage (696d1660c dd7d649c3): a storage image viewing a 2D array whose layers
// are the depth planes of the surfaces at consecutive depth slices, copied to it on access and back
// after its writes through a staging buffer (Metal blits; a plane of each surface with CopySurfacePlane).
class LayeredDepthStorage {
public:
    LayeredDepthStorage(const Context& context, const GuestTextureResource& resource, const DepthTarget& base, std::uint32_t bytesPerTexel, std::uint32_t firstLayer, std::uint32_t layers) : context(context), base(base), extent(base.extent), bytesPerTexel(bytesPerTexel), firstLayer(firstLayer), layers(layers) {
        const auto bytes = DescribeSurface(resource).guestBytes;
        StorageTexture::FlushPending(resource.baseAddress, static_cast<std::size_t>(bytes), nullptr, "storage image creation", PublishScope::None);
        storage = StorageTexture::Create(context, *context.detiler, resource, 0);
        Require(storage->ImageLayers() == layers, "a layered depth storage view's image does not hold its layers");
        staging = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(sliceBytes()) * layers, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    void Copy(std::span<const LayeredSlice> slices, bool toStorage) {
        if (slices.empty()) return;
        auto* recorder = Recorder::Active();
        Require(recorder != nullptr, "depth surface storage copies need an active recorder");
        const auto commands = recorder->Commands();
        recorder->Keep(storage);
        recorder->Keep(staging);
        const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
        std::vector<VkBufferImageCopy> storageRegions;
        for (const auto& slice : slices) {
            VkBufferImageCopy copy{};
            copy.bufferOffset = sliceBytes() * slice.layer;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, slice.layer, 1};
            copy.imageExtent = {extent.width, extent.height, 1};
            storageRegions.push_back(copy);
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        if (toStorage) {
            for (const auto& slice : slices) CopySurfacePlane(commands, *slice.surface, false, staging->Handle(), sliceBytes() * slice.layer, true);
        } else {
            Metal::Vk::CmdCopyImageToBuffer(commands, storage->Image(), VK_IMAGE_LAYOUT_GENERAL, staging->Handle(), static_cast<std::uint32_t>(storageRegions.size()), storageRegions.data());
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        if (toStorage) {
            Metal::Vk::CmdCopyBufferToImage(commands, staging->Handle(), storage->Image(), VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(storageRegions.size()), storageRegions.data());
        } else {
            for (const auto& slice : slices) CopySurfacePlane(commands, *slice.surface, false, staging->Handle(), sliceBytes() * slice.layer, false);
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
        recorder->EndGpuTiming(timing, sliceBytes() * slices.size());
    }

    const Context context;
    const DepthTarget base;
    const VkExtent2D extent;
    const std::uint32_t bytesPerTexel;
    const std::uint32_t firstLayer;
    const std::uint32_t layers;
    std::shared_ptr<StorageTexture> storage;

private:
    std::shared_ptr<DeviceBuffer> staging;

    VkDeviceSize sliceBytes() const {
        return static_cast<VkDeviceSize>(extent.width) * extent.height * bytesPerTexel;
    }
};

using LayeredStorageKey = std::tuple<const void*, std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, int, int, std::uint32_t>;

std::map<LayeredStorageKey, std::unique_ptr<LayeredDepthStorage>>& layeredStorages() {
    static auto* storages = new std::map<LayeredStorageKey, std::unique_ptr<LayeredDepthStorage>>();
    return *storages;
}

// Upstream's layeredSlices: the single-sampled surfaces at the view's depth slices. A slice stored at
// another plane size than the view's texels (a 16-bit depth widened by its stencil pairing) is refused,
// as RequireStorable refuses the base surface's.
std::vector<LayeredSlice> layeredSlices(const void* key, const LayeredDepthStorage& storage) {
    const auto& base = storage.base;
    std::vector<LayeredSlice> slices;
    const auto& list = surfaces();
    for (std::uint32_t layer = storage.firstLayer; layer < storage.layers; ++layer) {
        const auto address = base.address + static_cast<std::uint64_t>(layer) * DepthSliceBytes(base.extent, storage.bytesPerTexel);
        const auto slice = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->device == key && !surface->retired && surface->target.address == address && surface->target.extent.width == base.extent.width && surface->target.extent.height == base.extent.height && surface->target.format == base.format && surface->target.samples == 1; });
        if (slice == list.rend()) continue;
        if (!(*slice)->HasDepth() || (*slice)->PlaneBytes(false) != storage.bytesPerTexel) {
            char text[256];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the depth planes of depth surface 0x%llx through a 2D array whose slice 0x%llx is stored in another depth format is not implemented", static_cast<unsigned long long>(base.address), static_cast<unsigned long long>(address));
            throw std::runtime_error(text);
        }
        slices.push_back({slice->get(), layer});
    }
    return slices;
}

// Upstream's layeredDepthStorage (696d1660c dd7d649c3).
std::shared_ptr<StorageTexture> layeredDepthStorage(const Context& context, Surface& surface, const GuestTextureResource& resource, std::uint32_t mip) {
    const auto& target = surface.target;
    const auto layers = resource.depthOrLastArray + 1u;
    const auto bytesPerTexel = surface.D16() ? 2u : 4u;
    RequireStorable(context, surface, resource);
    const auto geometry = DescribeSurface(resource);
    if (layers > 1 && geometry.GuestLayerOffset(1) != DepthSliceBytes(target.extent, bytesPerTexel)) {
        char text[256];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the depth planes of depth surface 0x%llx with a layer pitch of 0x%llx bytes, not the depth slice size 0x%llx, is not implemented", static_cast<unsigned long long>(target.address), static_cast<unsigned long long>(geometry.GuestLayerOffset(1)), static_cast<unsigned long long>(DepthSliceBytes(target.extent, bytesPerTexel)));
        throw std::runtime_error(text);
    }
    const auto key = DeviceKey(context);
    auto& entry = layeredStorages()[{key, resource.baseAddress, resource.width, resource.height, layers, resource.format, static_cast<int>(resource.tileMode), static_cast<int>(resource.dimension), resource.baseArray}];
    if (entry == nullptr) entry = std::make_unique<LayeredDepthStorage>(context, resource, target, bytesPerTexel, resource.baseArray, layers);
    else static_cast<void>(entry->storage->Refresh());
    const auto slices = layeredSlices(key, *entry);
    for (const auto& slice : slices) ApplyFastClear(context, *slice.surface);
    entry->Copy(slices, true);
    return entry->storage;
}

Surface* FindSurface(const Context& context, std::uint64_t address) {
    const auto& list = surfaces();
    const auto key = DeviceKey(context);
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) {
        return surface->device == key && !surface->retired && (surface->target.address == address || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == address));
    });
    return found == list.rend() ? nullptr : found->get();
}

// Upstream's retire (626cff7d26): lookups skip the surface and its planes have no last owner.
void retire(Surface& surface) {
    surface.retired = true;
    for (auto* owners : {&depthOwners(), &stencilOwners()}) {
        std::erase_if(*owners, [&](const auto& entry) { return entry.second.surface == &surface; });
    }
}

// Upstream's expandedHtileWord (4ddff7c6e5): the HTILE word after the bind takes its fast clear
// (depth expanded, ZMask 0xf; the stencil field marked uncompressed).
std::uint32_t expandedHtileWord(std::uint32_t word, bool stencilInHtile, VkImageAspectFlags cleared) {
    if (!stencilInHtile) return (cleared & DepthAspect) != 0 ? 0xfffc000fu : word;
    if ((cleared & DepthAspect) != 0) word = (word & 0x3f0u) | 0xfffff00fu;
    if ((cleared & StencilAspect) != 0) word |= 0x3f0u;
    return word;
}

// Upstream's htileReads (3e61fcf1f9): each HTILE's guest write generation when a bind last read it.
std::mutex& htileReadsMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::uint64_t, std::uint64_t>& htileReads() {
    static auto* reads = new std::map<std::uint64_t, std::uint64_t>();
    return *reads;
}

void forgetHtileReads(std::uint64_t address, std::size_t bytes) {
    std::lock_guard lock(htileReadsMutex());
    auto& reads = htileReads();
    reads.erase(reads.lower_bound(address), reads.lower_bound(address + bytes));
}

// Upstream's RoutesToNativeDepthOwner (139e3b86ed): multisampled targets and targets whose HTILE a
// resident native owner guards skip the HTILE read and store-back.
bool RoutesToNativeDepthOwner(const DepthTarget& target) {
    if (target.samples != 1u) return true;
    if (target.htileAddress == 0 || !anyDepthGuard().load(std::memory_order_acquire)) return false;
    const auto bytes = static_cast<std::uint64_t>(HtileMetadataBytes(target.extent));
    std::shared_lock lock(depthGuardMutex());
    return std::any_of(depthGuards().begin(), depthGuards().end(), [&](const auto& range) { return range.active->load(std::memory_order_acquire) && target.htileAddress < range.address + range.bytes && range.address < target.htileAddress + bytes; });
}

// Upstream's storeBackClearedHtile (4ddff7c6e5 3e61fcf1f9 cc022df348), run before the bind outside the
// surfaces lock: when every word of the target's HTILE (written by any kernel) proves a fast clear, the
// bind takes that clear and the words are stored back expanded. A slice of a depth array (firstSlice
// != 0) takes it only when its words are uniform. `recorder`: the device's active recorder, if any.
VkImageAspectFlags storeBackClearedHtile(Recorder* recorder, const DepthTarget& target) {
    if (target.htileAddress == 0) return 0;
    const auto bytes = static_cast<std::size_t>(HtileMetadataBytes(target.extent));
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(target.htileAddress), bytes, true)) return 0;
    const auto collected = GuestMemory::CollectWrites(target.htileAddress, bytes);
    if (recorder == nullptr || !recorder->PendingWriteOverlaps(target.htileAddress, bytes)) {
        std::lock_guard lock(htileReadsMutex());
        const auto found = htileReads().find(target.htileAddress);
        if (found != htileReads().end() && found->second != 0 && GuestMemory::UnchangedSince(target.htileAddress, bytes, found->second)) return 0;
    }
    std::vector<std::uint32_t> words(bytes / 4u);
    GuestMemory::Read(target.htileAddress, std::as_writable_bytes(std::span(words)), 4);
    const bool uniform = std::all_of(words.begin(), words.end(), [&](std::uint32_t word) { return word == words.front(); });
    VkImageAspectFlags cleared = target.firstSlice == 0 || uniform ? DepthAspect | (target.htileStencil ? StencilAspect : 0u) : 0u;
    for (const auto word : words) cleared &= HtileFillClears(word, target.htileStencil);
    auto generation = collected;
    if (cleared != 0) {
        for (auto& word : words) word = expandedHtileWord(word, target.htileStencil, cleared);
        GuestMemory::Write(target.htileAddress, std::as_bytes(std::span(words)), 4);
        generation = GuestMemory::CollectWrites(target.htileAddress, bytes);
    }
    std::lock_guard lock(htileReadsMutex());
    htileReads()[target.htileAddress] = generation;
    return cleared;
}

// The active recorder when it records for the device `key` (RecorderFor without a context).
Recorder* RecorderForDevice(const void* key) {
    auto* recorder = Recorder::Active();
    return recorder != nullptr && reinterpret_cast<const void*>(recorder->Device()) == key ? recorder : nullptr;
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

// Upstream's DepthSurfaceView binding (80d8bd06b 7d898ae1a 8fec47a4b), shared by DepthSurfaceView and the
// Metal draw path (AcquireDepthSurface), under GpuMutex and surfacesMutex: the surface of the target
// (made with both planes pending their clears), which takes its planes from their last pairings, starts
// a new depth plane without HTILE from the storage writes over it, and takes a noted HTILE clear of its
// slice. `context` is the device's (null: no copies can be recorded, planes are kept as they are).
// `resident`: make a new surface's texture resident here (the draw path does it itself). `htileCleared`:
// the clears storeBackClearedHtile took from the target's HTILE before the locks (upstream 4ddff7c6e5).
Surface* BindSurface(const Context* context, id<MTLDevice> device, const DepthTarget& target, MTLPixelFormat format, bool resident, VkImageAspectFlags htileCleared, bool* created) {
    const bool cleared = takeClearedHtile(target);
    const auto key = (__bridge const void*)device;
    Surface* surface = nullptr;
    for (const auto& candidate : surfaces()) {
        if (candidate->device != key || !sameSurface(candidate->target, target) || candidate->format != format) continue;
        surface = candidate.get();
        break;
    }
    *created = surface == nullptr;
    if (*created) {
        id<MTLTexture> texture = MakeSurfaceTexture(device, target, format);
        if (resident) {
            if (auto* images = Metal::ImageCache::For(device)) {
                const std::array<id<MTLTexture>, 1> textures{texture};
                images->MakeResident(textures);
            }
        }
        surfacesSerial.fetch_add(1, std::memory_order_release);
        surfaces().push_back(std::make_unique<Surface>(key, target, format, texture));
        surface = surfaces().back().get();
        // Upstream clears a new surface when it makes it; here the next pass's load actions (or
        // ApplyFastClear) do, unless a plane is taken from a pairing or from storage writes below.
        surface->pendingClear = DepthAspect | StencilAspect;
    }
    // A retired surface bound again serves its memory again (upstream 626cff7d26); retire drops its
    // plane owners, so a retired owner is never inherited from.
    surface->retired = false;
    const auto owner = depthOwners().find({key, target.address});
    const bool inherits = owner != depthOwners().end() && !owner->second.surface->retired;
    surface->target.clearDepth = target.clearDepth;
    surface->target.clearStencil = target.clearStencil;
    takePlanes(context, *surface);
    if (*created && !inherits && target.htileAddress == 0 && context != nullptr) TakeStorage(*context, *surface);
    if (cleared && (!*created || inherits)) surface->NoteHtileClear(DepthAspect);
    surface->NoteHtileClear(htileCleared);
    // The draw writes it (upstream bumps the serial of a bound surface and names the drawn surface).
    if (!*created || inherits) surface->written = ++depthWrites;
    surface->NoteMemory();
    drawnDepth = surface;
    return surface;
}

}

std::uint64_t DepthSurfacesSerial() {
    return surfacesSerial.load(std::memory_order_acquire);
}

std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel) {
    const std::uint32_t blockWidth = bytesPerTexel == 4 ? 128u : 256u;
    const std::uint32_t blockHeight = bytesPerTexel == 1 ? 256u : 128u;
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1) / blockHeight * blockHeight);
    return width * height * bytesPerTexel;
}

// Upstream ee1ba73a2f: the guest footprint of a multisampled depth plane (Z_X 64KB swizzle blocks).
std::uint64_t MultisampledDepthBytes_nid_no_patch(VkExtent2D extent, std::uint32_t bytesPerTexel, std::uint32_t samples) {
    Require(extent.width != 0u && extent.width <= 16384u && extent.height != 0u && extent.height <= 16384u, "invalid multisample depth extent");
    Require((bytesPerTexel == 1u || bytesPerTexel == 2u || bytesPerTexel == 4u) && (samples == 2u || samples == 4u || samples == 8u), "unsupported multisample depth footprint");
    const auto sampleBits = static_cast<std::uint32_t>(std::countr_zero(samples));
    const auto pixelBits = 16u - static_cast<std::uint32_t>(std::countr_zero(bytesPerTexel)) - sampleBits;
    const auto widthBits = (pixelBits + ((sampleBits & 1u) == 0u ? 1u : 0u)) / 2u;
    const auto blockWidth = 1u << widthBits;
    const auto blockHeight = 1u << (pixelBits - widthBits);
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1u) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1u) / blockHeight * blockHeight);
    return width * height * bytesPerTexel * samples;
}

// Upstream's resident-depth guard API (ee1ba73a2f da9fe3c708) over the registry above (empty on Metal).
void RequireNoResidentDepthRead_nid_no_patch(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0u || !anyDepthGuard().load(std::memory_order_acquire)) return;
    Require(address <= std::numeric_limits<std::uint64_t>::max() - bytes, "native depth read range overflows");
    std::shared_lock lock(depthGuardMutex());
    for (const auto& range : depthGuards()) Require(address >= range.address + range.bytes || range.address >= address + bytes, "sample-preserving guest publication of native multisample depth is not implemented");
}

bool HasResidentDepthGuards_nid_no_patch() {
    return anyDepthGuard().load(std::memory_order_acquire);
}

void RetireUnmappedDepth_nid_no_patch(std::uintptr_t address, std::size_t bytes) {
    if (bytes == 0u || !HasResidentDepthGuards_nid_no_patch()) return;
    Require(address <= std::numeric_limits<std::uint64_t>::max() - bytes, "native depth release range overflows");
    const auto end = static_cast<std::uint64_t>(address) + bytes;
    std::unique_lock lock(depthGuardMutex());
    for (auto& range : depthGuards()) {
        std::vector<GuestMemory::UnchangedQuery> remaining;
        remaining.reserve(range.live.size() + 1u);
        for (const auto& part : range.live) {
            const auto finish = part.address + part.bytes;
            const auto first = std::max<std::uint64_t>(address, part.address);
            const auto last = std::min<std::uint64_t>(end, finish);
            if (first >= last) remaining.push_back(part);
            else {
                if (part.address < first) remaining.push_back({part.address, static_cast<std::size_t>(first - part.address), 0u});
                if (last < finish) remaining.push_back({last, static_cast<std::size_t>(finish - last), 0u});
            }
        }
        range.live = std::move(remaining);
    }
    for (const auto& range : depthGuards()) {
        if (std::all_of(depthGuards().begin(), depthGuards().end(), [&](const auto& part) { return part.active != range.active || part.live.empty(); })) range.active->store(false, std::memory_order_release);
    }
    std::erase_if(depthGuards(), [](const auto& range) { return !range.active->load(std::memory_order_acquire); });
    anyDepthGuard().store(!depthGuards().empty(), std::memory_order_release);
}

bool SampleLocationsCompatibleDepth(const Context& context, VkFormat format, std::uint32_t samples) {
    // As upstream: sample counts with custom locations, then the depth format's multisample support
    // through the context's query (a test's mock device) or the Metal device's. Metal sets sample
    // positions per render pass (programmable sample positions), so any depth texture of a supported
    // sample count is compatible.
    if (samples == 0 || (context.sampleLocationSampleCounts & samples) == 0) return false;
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
    const auto htileCleared = RoutesToNativeDepthOwner(target) ? VkImageAspectFlags{0} : storeBackClearedHtile(RecorderFor(context), target);
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    bool created = false;
    auto* surface = BindSurface(&context, device, target, SurfaceFormat(target), true, htileCleared, &created);
    ApplyFastClear(context, *surface);
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

// Upstream 7d0dd69f0a 9e27d0b623: the HTILE bytes a completion-proven native HTILE mask dispatch clears
// (its group count times 1 KiB), 0 when the dispatch does not prove it.
std::optional<StorageImageClear> MatchStorageImageClear(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, const std::array<std::uint32_t, 3>& groups) {
    static constexpr std::array<std::uint32_t, 15> kernel{0xd7460004u, 0x0401060cu, 0xf4280204u, 0xfa000000u, 0xd7460005u, 0x0405060du, 0x7e0c020eu, 0xbf8cc07fu, 0x7e000208u, 0x7e020209u, 0x7e04020au, 0x7e06020bu, 0xf0200f28u, 0x00000004u, 0xbf810000u};
    if (code.size() < kernel.size() || code[0] != kernel[0] || !std::equal(kernel.begin(), kernel.end(), code.begin())) return std::nullopt;
    if (userData.size() != 12u) return std::nullopt;
    if (compute.numThreads != std::array<std::uint32_t, 3>{8u, 8u, 1u}) return std::nullopt;
    if (!compute.groupIdEnable[0] || !compute.groupIdEnable[1] || !compute.groupIdEnable[2]) return std::nullopt;
    if (compute.threadIdComponentCount < 2u) return std::nullopt;
    std::array<std::uint64_t, 3> covered{};
    for (std::size_t axis = 0; axis < 3u; ++axis) covered[axis] = compute.partialThreads[axis] != 0u ? compute.partialThreads[axis] : static_cast<std::uint64_t>(groups[axis]) * compute.numThreads[axis];
    if (covered[0] == 0u || covered[1] == 0u || covered[2] != 1u) return std::nullopt;
    StorageImageClear clear;
    std::copy_n(userData.begin(), clear.image.size(), clear.image.begin());
    std::copy_n(userData.begin() + static_cast<std::ptrdiff_t>(clear.image.size()), clear.color.size(), clear.color.begin());
    clear.width = static_cast<std::uint32_t>(std::min<std::uint64_t>(covered[0], std::numeric_limits<std::uint32_t>::max()));
    clear.height = static_cast<std::uint32_t>(std::min<std::uint64_t>(covered[1], std::numeric_limits<std::uint32_t>::max()));
    return clear;
}

std::uint64_t StorageImageClearAddress(const StorageImageClear& clear) {
    try {
        return DecodeTextureResource(clear.image).baseAddress;
    } catch (...) {
        return 0u;
    }
}

DepthPlaneClear::DepthPlaneClear(const StorageImageClear& clear) {
    if (!depthPlaneClearEnabled.load(std::memory_order_relaxed)) return;
    const auto decoded = decodeOverwrite(clear);
    if (decoded) {
        value = *decoded;
        previous = currentOverwrite;
        currentOverwrite = &value;
        installed = true;
        ++depthClearCounters().taken;
    }
}

DepthPlaneClear::~DepthPlaneClear() {
    if (installed) currentOverwrite = previous;
}

DepthClearStatistics DepthClearCounts() {
    const auto& counters = depthClearCounters();
    DepthClearStatistics statistics;
    statistics.matched = counters.matched.load();
    statistics.taken = counters.taken.load();
    statistics.copiesSkipped = counters.copiesSkipped.load();
    statistics.clears = counters.clears.load();
    statistics.bytesSkipped = counters.bytesSkipped.load();
    for (std::size_t reason = 0; reason < statistics.declined.size(); ++reason) statistics.declined[reason] = counters.declined[reason].load();
    return statistics;
}

void SetDepthPlaneClearForTest(bool enabled) {
    depthPlaneClearEnabled.store(enabled, std::memory_order_relaxed);
}

std::size_t HtileDepthClearBytes_nid_no_patch(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, std::span<const std::uint32_t> packet, std::uint64_t indirectArguments) {
    if (userData.size() != 6u || packet.size() < 5u || indirectArguments != 0u || compute.PartialGroups() || !compute.groupIdEnable[0] || compute.threadIdComponentCount == 0u || compute.threadIdComponentCount > 3u) return 0u;
    if (packet[1] == 0u || packet[1] > 0x400000u || packet[2] != 1u || packet[3] != 1u || (packet[4] & 0x23u) != 1u) return 0u;
    const auto address = HtileDepthClearAddress(code, userData, compute.numThreads);
    if (address == 0u || (address & 15u) != 0u) return 0u;
    const auto bytes = static_cast<std::uint64_t>(packet[1]) * 1024u;
    return address <= std::numeric_limits<std::uint64_t>::max() - bytes ? static_cast<std::size_t>(bytes) : 0u;
}

// Upstream 8fec47a4b: the fill clears the HTILE of every surface it covers (each slice of a depth array).
// `nativeBytes` (upstream 7d0dd69f0a) only feeds a native multisampled depth owner's clear transition,
// which Metal has none of (see the file comment); every surface takes the clear as before.
void NoteHtileDepthClear(std::uint64_t htileAddress, std::size_t bytes, std::size_t nativeBytes) {
    static_cast<void>(nativeBytes);
    std::lock_guard lock(surfacesMutex());
    forgetHtileReads(htileAddress, bytes);
    bool existing = false;
    for (const auto& surface : surfaces()) {
        if (!HtileFillCovers(surface->target.htileAddress, surface->target.extent, htileAddress, bytes)) continue;
        surface->NoteHtileClear(DepthAspect);
        existing = true;
    }
    if (!existing) clearedHtiles()[htileAddress] = bytes;
}

void ClearDepthSurfaces(VkDevice device) {
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    const auto key = reinterpret_cast<const void*>(device);
    surfacesSerial.fetch_add(1, std::memory_order_release);
    std::erase_if(depthOwners(), [&](const auto& entry) { return entry.first.first == key; });
    std::erase_if(stencilOwners(), [&](const auto& entry) { return entry.first.first == key; });
    std::erase_if(planeCopies(), [&](const auto& entry) { return std::get<0>(entry.first) == key; });
    std::erase_if(layeredStorages(), [&](const auto& entry) { return std::get<0>(entry.first) == key; });
    if (drawnDepth != nullptr && drawnDepth->device == key) drawnDepth = nullptr;
    std::erase_if(surfaces(), [&](const auto& surface) { return surface->device == key; });
    std::unique_lock guardLock(depthGuardMutex());
    std::erase_if(depthGuards(), [&](const auto& range) { return range.device == device; });
    anyDepthGuard().store(!depthGuards().empty(), std::memory_order_release);
    UpdateResidentImageUnmapObserver_nid_no_patch();
}

// Upstream's DepthSurfaceTexture over plane copies (see the file comment): a single plane is a
// one-layer copy of the surface; several slices and integer views are upstream's DepthPlaneCopy.
std::shared_ptr<Texture> DepthSurfaceTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components) {
    {
        std::lock_guard lock(surfacesMutex());
        if (planeSurface(DeviceKey(context), resource.baseAddress) == nullptr) return nullptr;
    }
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    // The plane's last binding (upstream 80d8bd06b), not merely the newest surface over it.
    auto* found = planeSurface(DeviceKey(context), resource.baseAddress);
    if (found == nullptr) return nullptr;
    const auto& base = found->target;
    const bool stencil = base.stencilAddress != 0 && resource.baseAddress == base.stencilAddress;
    // A stencil alias of another element size is not the stencil plane (upstream 6972b5ae07: filtered
    // before the multisampled path).
    if (stencil && BytesPerElement(resource.format) != 1u) return nullptr;
    // Upstream ee1ba73a2f samples a native multisampled depth attachment through a view of its image.
    // The Metal surface is a multisample array (D27/D28: 8x on Apple9 is two 4x slices) that no sampled
    // view or MSL texture2d_ms read describes yet, so it is refused, as upstream refused it before.
    if (base.samples != 1u || resource.dimension == TextureDimension::k2DMsaa) {
        Require(base.samples > 1u, "multisample reader overlaps a single-sample depth attachment");
        char text[256];
        std::snprintf(text, sizeof(text), "AGC graphics: sampling the %s plane of multisampled depth surface 0x%llx (%ux%u, %u samples, vk format %d) on Metal is not implemented", stencil ? "stencil" : "depth", static_cast<unsigned long long>(base.address), base.extent.width, base.extent.height, base.samples, static_cast<int>(base.format));
        throw std::runtime_error(text);
    }
    if (resource.width != base.extent.width || resource.height != base.extent.height || resource.baseLevel != 0) return nullptr;
    const bool d16 = found->D16();
    const auto format = ResolveTextureFormat(resource.format);
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    const bool integer = d16 && format == VK_FORMAT_R16_UINT && !depthBits;
    if (!stencil && format != (d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT) && !depthBits && !integer) return nullptr;
    const bool cube = resource.dimension == TextureDimension::kCube;
    const bool layered = cube || (resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray != 0);
    const bool sampled = stencil || (!layered && !integer);
    // Upstream 47f7118241: a stencil plane read through an R8_UNORM descriptor (a plane copy in R8_UNORM).
    const bool unormStencil = stencil && format == VK_FORMAT_R8_UNORM;
    // Upstream's SampledAccepts (one plane of the surface) and the plane-copy shapes (several slices or
    // an integer view).
    const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool oneSlice = resource.dimension == TextureDimension::k2D || (resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray == 0);
    const bool accepts = (format == expected || depthBits || unormStencil) && oneSlice && resource.baseLevel == 0 && resource.lastLevel == 0 && resource.baseArray == 0;
    const auto planeFormat = integer ? VK_FORMAT_R16_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool planes = (format == planeFormat || depthBits) && resource.baseLevel == 0 && resource.lastLevel == 0 && !(resource.baseArray != 0 && (cube || !layered || resource.baseArray > resource.depthOrLastArray)) && !(cube && resource.depthOrLastArray != 0 && resource.depthOrLastArray != 5) && (resource.dimension == TextureDimension::k2D || resource.dimension == TextureDimension::k2DArray || cube);
    // Upstream 626cff7d26: a read the surface cannot serve over memory the guest rewrote since the last
    // bind is a read of that memory, not of the surface: retire it and let the caller read guest memory.
    if (!(sampled ? accepts : planes) && found->OverwrittenInMemory()) {
        retire(*found);
        return nullptr;
    }
    ApplyFastClear(context, *found);
    if (sampled) {
        // Upstream's DepthSurface::Sampled: one plane of the surface.
        if (!accepts) {
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
        const auto copyFormat = stencil ? (unormStencil ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8_UINT) : found->PlaneBytes(false) == 2u ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
        const std::array<PlaneSlice, 1> slices{{{found, stencil, found == drawnDepth ? 0u : found->written}}};
        auto& copy = planeCopies()[{DeviceKey(context), resource.baseAddress, base.extent.width, base.extent.height, copyFormat, 1u}];
        if (copy == nullptr) copy = std::make_unique<PlaneCopy>(context, base.extent, copyFormat, 1u);
        return copy->Refresh(slices, resource, components, resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
    }
    Require(!found->Widened(), "sampling several slices or an integer view of a depth plane that the device stores in another depth format is not implemented");
    const auto copyFormat = integer ? VK_FORMAT_R16_UINT : d16 ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
    const auto layers = cube ? 6u : layered ? resource.depthOrLastArray + 1u : 1u;
    if (!planes) {
        char text[320];
        std::snprintf(text, sizeof(text), "AGC graphics: sampling the depth planes of depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u texture of guest format %u (vk %d), dimension %d, %u slices, levels %u-%u, base slice %u is not implemented", static_cast<unsigned long long>(base.address), base.extent.width, base.extent.height, static_cast<int>(base.format), resource.width, resource.height, resource.format, static_cast<int>(format), static_cast<int>(resource.dimension), layers, resource.baseLevel, resource.lastLevel, resource.baseArray);
        throw std::runtime_error(text);
    }
    const auto key = DeviceKey(context);
    std::vector<PlaneSlice> slices;
    // The slices at the descriptor's layer offsets (upstream 03b7cb5a1: a cube's faces), each from its
    // plane's last binding (80d8bd06b).
    const auto geometry = DescribeSurface(resource);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const auto address = base.address + geometry.GuestLayerOffset(layer);
        auto* slice = depthSlice(key, address, base);
        if (slice == nullptr) {
            slices.push_back({});
            continue;
        }
        Require(!slice->Widened(), "sampling several slices of a depth plane that the device stores in another depth format is not implemented");
        ApplyFastClear(context, *slice);
        slices.push_back({slice, false, slice == drawnDepth ? 0u : slice->written});
    }
    auto& copy = planeCopies()[{key, base.address, base.extent.width, base.extent.height, copyFormat, layers}];
    if (copy == nullptr) copy = std::make_unique<PlaneCopy>(context, base.extent, copyFormat, layers);
    // A 2D array read from a later base slice (upstream 62eff52c4) views the copy from that slice.
    return copy->Refresh(slices, resource, components, cube || resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, layered && !cube ? resource.baseArray : 0u);
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

bool PlaneFillCovers(std::uint64_t plane, std::uint64_t planeBytes, std::uint64_t address, std::size_t bytes) {
    return plane != 0 && plane >= address && plane + planeBytes <= address + bytes;
}

std::optional<std::uint8_t> StencilFillValue(std::uint32_t pattern) {
    const auto byte = static_cast<std::uint8_t>(pattern & 0xffu);
    if (pattern != byte * 0x01010101u) return std::nullopt;
    return byte;
}

std::optional<float> DepthFillValue(std::uint32_t pattern, bool d16) {
    if (d16) {
        if ((pattern >> 16u) != (pattern & 0xffffu)) return std::nullopt;
        return static_cast<float>(pattern & 0xffffu) / 65535.0f;
    }
    const auto value = std::bit_cast<float>(pattern);
    if (!(value >= 0.0f && value <= 1.0f)) return std::nullopt;
    return value;
}

void NoteDepthMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern) {
    std::lock_guard lock(surfacesMutex());
    forgetHtileReads(address, bytes);
    for (const auto& surface : surfaces()) {
        const auto& target = surface->target;
        if (HtileFillCovers(target.htileAddress, target.extent, address, bytes)) {
            const VkImageAspectFlags written = DepthAspect | (target.htileStencil ? StencilAspect : 0u);
            surface->ForgetClear(written);
            surface->NoteHtileClear(HtileFillClears(pattern, target.htileStencil));
            continue;
        }
        // Upstream 3e1b7be111: a fill of a whole single-sample plane clears it to the filled value.
        if (target.samples != 1) continue;
        const bool d16 = surface->D16();
        if (PlaneFillCovers(target.address, DepthSliceBytes(target.extent, d16 ? 2u : 4u), address, bytes)) {
            if (const auto depth = DepthFillValue(pattern, d16)) surface->NotePlaneFill(DepthAspect, *depth, 0);
        }
        if (PlaneFillCovers(target.stencilAddress, DepthSliceBytes(target.extent, 1u), address, bytes)) {
            if (const auto stencil = StencilFillValue(pattern)) surface->NotePlaneFill(StencilAspect, 0.0f, *stencil);
        }
    }
}

bool DepthSurfaceAt(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource) {
    std::lock_guard lock(surfacesMutex());
    const auto* found = planeSurface(DeviceKey(context), resource.baseAddress);
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
    if (mip != 0 || resource.baseLevel != 0) return nullptr;
    std::lock_guard lock(surfacesMutex());
    auto* found = FindSurface(context, resource.baseAddress);
    if (found == nullptr) return nullptr;
    auto& surface = *found;
    const bool stencil = surface.target.stencilAddress != 0 && surface.target.stencilAddress == resource.baseAddress;
    if (stencil || resource.dimension != TextureDimension::k2DArray || (resource.baseArray == 0 && resource.depthOrLastArray == 0)) return SurfaceStorage(context, surface, words, resource, mip);
    // A 2D array over consecutive depth slices (upstream 696d1660c dd7d649c3).
    if (!StorageViewMatches(surface, words, resource)) return nullptr;
    return layeredDepthStorage(context, surface, resource, mip);
}

bool DepthSurfaceStoreBack(const Context& context, const StorageTexture& storage) {
    std::lock_guard lock(surfacesMutex());
    const auto key = DeviceKey(context);
    for (auto& [entryKey, entry] : layeredStorages()) {
        if (std::get<0>(entryKey) != key || entry->storage.get() != &storage) continue;
        const auto slices = layeredSlices(key, *entry);
        entry->Copy(slices, false);
        for (const auto& slice : slices) slice.surface->written = ++depthWrites;
        return true;
    }
    for (const auto& surface : surfaces()) {
        if (surface->device == key && SurfaceStoreBack(context, *surface, storage)) return true;
    }
    // A storage image covering a plane that is not the plane's own view: into the newest pairing
    // (upstream a84afe931 b3e5acc8c).
    for (auto surface = surfaces().rbegin(); surface != surfaces().rend(); ++surface) {
        if ((*surface)->device == key && SurfaceStoreBackCovering(context, **surface, storage)) return true;
    }
    return false;
}

bool DepthSurfaceAt(std::uint64_t address) {
    std::lock_guard lock(surfacesMutex());
    return std::any_of(surfaces().begin(), surfaces().end(), [&](const auto& surface) { return !surface->retired && (surface->target.address == address || surface->target.stencilAddress == address); });
}

bool DepthSurfaceOverlaps(VkDevice device, std::uint64_t address, std::uint64_t bytes) {
    std::lock_guard lock(surfacesMutex());
    const auto key = reinterpret_cast<const void*>(device);
    return std::any_of(surfaces().begin(), surfaces().end(), [&](const auto& surface) { return !surface->retired && surface->device == key && surface->Overlaps(address, bytes); });
}

// Upstream 626cff7d26: another writer (a color target, a storage image) covers the memory of these
// surfaces, so their contents no longer stand for it.
void RetireDepthSurfaces(VkDevice device, std::uint64_t address, std::uint64_t bytes) {
    std::lock_guard lock(surfacesMutex());
    const auto key = reinterpret_cast<const void*>(device);
    for (const auto& surface : surfaces()) {
        if (surface->device == key && !surface->retired && surface->Overlaps(address, bytes)) retire(*surface);
    }
}

// Upstream f56e3bc467: a draw must not sample the native multisampled depth target it attaches. Metal
// hands out no view of a multisampled depth surface (DepthSurfaceTexture refuses one; single planes
// are separate plane copies), so a sampled texture can never be the attachment: nothing to refuse.
void RequireNoNativeDepthFeedback_nid_no_patch(const Context&, const DepthTarget&, std::span<const std::shared_ptr<Texture>>) {}

namespace Metal {

DepthSurfaceUse AcquireDepthSurface(id<MTLDevice> device, const DepthTarget& target, MTLPixelFormat format, bool* created) {
    // The HTILE fast clear any kernel wrote, read and stored back before the locks (upstream 3e61fcf1f9).
    const auto htileCleared = RoutesToNativeDepthOwner(target) ? VkImageAspectFlags{0} : storeBackClearedHtile(RecorderForDevice((__bridge const void*)device), target);
    // GpuMutex first, as upstream's DepthSurfaceView: taking a plane from storage refreshes a storage image.
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    // The device's context records the plane copies (pairings, storage writes) before the draw's pass.
    auto* images = ImageCache::For(device);
    const Context* context = images != nullptr ? images->ImageContext() : nullptr;
    auto* surface = BindSurface(context, device, target, format, false, htileCleared, created);
    const VkImageAspectFlags aspects = DepthAspect | (target.stencilAddress != 0 ? StencilAspect : 0u);
    const auto clears = surface->pendingClear & aspects;
    surface->pendingClear = 0;
    const auto [depth, stencil] = surface->TakeClearValues();
    return {surface->texture, (clears & DepthAspect) != 0, (clears & StencilAspect) != 0, depth, stencil};
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
