#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALIMAGES_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALIMAGES_H

// Work packages D4 (textures, detiling, formats) and D5 (descriptors, samplers): guest images and
// samplers of a dispatch's MslAbi descriptor table (docs/mac/M2_DESIGN.md §4, §6).
//
// Guest T#s and S#s go through the Metal twins of upstream's image path (Texture.mm =
// StorageTexture/TextureCache, ShaderResources.mm, PlaceholderImages, SamplerCache) over the
// device's Context, attached to the per-device ImageCache: sampled textures are cached across
// dispatches, storage results stay pending in their images until a reader needs guest memory. The
// bring-up path (per-dispatch CPU upload, write-back at batch completion) was removed after the
// upstream path matched or beat it on G1-G4 (M2_DESIGN "SHD3 results").
// Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {
class Recorder;
class ShaderResources;
class TextureDetiler;
class PlaceholderImages;
class SamplerCache;
class TextureCache;
struct Context;
}

namespace AgcDriver::Graphics::Metal {

// --- formats (MetalTextureFormat.mm) -------------------------------------------------------------

struct TextureFormatInfo {
    MTLPixelFormat format = MTLPixelFormatInvalid;
    // Bytes of one guest element (texel, or block of a compressed format).
    std::uint32_t elementBytes = 0;
    // Bytes of one element in the Metal texture (16 for the three-channel 32-bit formats, which
    // Metal holds as four channels).
    std::uint32_t textureElementBytes = 0;
    bool blockCompressed = false;
    // Guest format whose red and blue channels Metal stores swapped (B4G4R4A4 as ABGR4).
    bool swapRedBlue = false;
};

// The Metal format of a guest texture format (Graphics/src/TextureFormat.cpp's table); throws
// naming D4 for formats Metal has no counterpart for.
TextureFormatInfo GuestTextureFormat(std::uint32_t guestFormat);
// The format a view of `format` uses in a heap whose shader type has `numeric` scalars (0 float,
// 1 uint, 2 sint): the integer twin of a normalized format of the same size (R8Unorm → R8Uint),
// `format` itself when it already matches or has no twin.
MTLPixelFormat HeapViewFormat(MTLPixelFormat format, std::uint32_t numeric);
// RecompileRequest target bits (ShaderRecompiler::SrgbDecodeBit) of the sRGB formats the shader
// decodes itself: R8 and RG8 sRGB under APS5_SRGB_SHADER_DECODE=1, else none. GuestTextureFormat
// then gives their UNORM format.
std::uint32_t SrgbDecodeFormats();
// Depth format for comparison sampling of an R32 float or R16 unorm texture.
MTLPixelFormat DepthFormatFor(MTLPixelFormat format);

// --- detiling (MetalTextureDetiler.mm) -----------------------------------------------------------

// Moves the elements of one mip level of one layer between the guest's tiled bytes and a linear
// image of `pitchBytes` rows (the TextureDetile.comp addressing, on the CPU). `tiled` starts at the
// mip's first byte (guest layer offset + TileMipLayout::tiledOffset).
struct DetileLevel {
    TextureTileMode tileMode;
    std::uint32_t elementBytes;
    const TileMipLayout* mip;
    std::uint32_t slice;
    bool thick;
    std::uint32_t pipeBankXor;
};
// The byte offset of element (x, y) of the level from its first tiled byte.
std::uint64_t TiledElementOffset(const DetileLevel& level, std::uint32_t x, std::uint32_t y);
// Detile (tiled → linear) or retile (linear → tiled) every element of the level. `linear` rows are
// `linearPitch` bytes apart, elements `linearElementBytes` apart (≥ elementBytes: the first
// elementBytes of each are moved). `tiledBytes` bounds the tiled span (elements past it are skipped).
void DetileCpu(const DetileLevel& level, const std::byte* tiled, std::uint64_t tiledBytes, std::byte* linear, std::uint64_t linearPitch, std::uint32_t linearElementBytes);
void RetileCpu(const DetileLevel& level, std::byte* tiled, std::uint64_t tiledBytes, const std::byte* linear, std::uint64_t linearPitch, std::uint32_t linearElementBytes);

// --- textures and samplers -----------------------------------------------------------------------

class ImageCache;

// The images and samplers of one dispatch: Write() fills the table entries of each GuestImages and
// GuestSamplers binding (MTLResourceID values), Commit() hands the textures to the batch.
class ImageBindings {
public:
    // The second argument (a settle callback of the removed bring-up uploads) is ignored.
    ImageBindings(ImageCache& cache, std::function<void(std::uint64_t, std::uint64_t)> settle);
    ~ImageBindings();
    ImageBindings(const ImageBindings&) = delete;
    ImageBindings& operator=(const ImageBindings&) = delete;

    // Before the first Write: the dispatch's ShaderResources build (a revalidated resource-cache
    // entry or a new build) when the shader binds guest images or samplers.
    void Prepare(const ShaderRecompiler::RecompileResult& shader);
    // `entries`: the binding's first min(binding.count, range.count) table entries.
    void Write(const ShaderRecompiler::DescriptorBinding& binding, std::span<std::uint64_t> entries);
    // After the dispatch is recorded into the recorder's open batch: marks the written storage
    // images pending (ShaderResources::MarkGpuWrites) and keeps the build with the batch.
    void Commit(Recorder& recorder);

    bool Empty() const { return builds.empty(); }

private:
    ImageCache& cache;
    // The upstream builds (ShaderResources twin, MetalImagesOnly) over the cache's attached Context,
    // one per stage that binds images (a draw's vertex and fragment functions each have their own):
    // the image and sampler entries come from them, and the pending/live storage registries, the
    // texture caches and the resource cache are upstream's (Texture.mm, ShaderResources.mm).
    // `resources` is the build of the shader Prepare saw last, which its Writes read.
    std::vector<std::shared_ptr<ShaderResources>> builds;
    std::shared_ptr<ShaderResources> resources;
    bool committed = false;
};

// Per device: the residency set of image resources (attached to the device's queue) and the
// attached image Context with its detiler, placeholder images, sampler and texture caches.
class ImageCache {
public:
    ImageCache(id<MTLDevice> device, id<MTL4CommandQueue> queue);
    ~ImageCache();
    ImageCache(const ImageCache&) = delete;
    ImageCache& operator=(const ImageCache&) = delete;

    // Removes the residency set from the queue (DECISIONS D19.5); call when the recorder is idle.
    // Idempotent; the destructor calls it.
    void Teardown() noexcept;

    id<MTLDevice> Device() const { return device; }
    // Attaches the device's Context (once it is complete): image bindings then build upstream's
    // ShaderResources over a copy of it that carries the cache's detiler, placeholder images and
    // sampler cache (D4/D5 twins). Teardown clears the texture and resource caches of the device.
    void Attach(const Context& deviceContext);
    // The Context image builds use (null before Attach).
    const Context* ImageContext() const { return imageContext.get(); }

    // The image cache of a device (the one constructed with it), or null: the textures the
    // Vulkan-shaped twins make (MetalVulkanImages.mm) are tracked by it.
    static ImageCache* For(id<MTLDevice> device);
    // A texture the twins made: resident until Untrack (DestroyImage, after the batches using it).
    void Track(id<MTLTexture> texture);
    void Untrack(id<MTLTexture> texture);
    // Buffers the twins make (unit shadow slabs, MetalVulkanImages CreateBuffer), likewise.
    void TrackBuffer(id<MTLBuffer> buffer);
    void UntrackBuffer(id<MTLBuffer> buffer);
    // Residency of textures in flight: added before the batch is submitted, removed when it completes.
    void MakeResident(std::span<const id<MTLTexture>> textures);
    void Release(std::span<const id<MTLTexture>> textures);

private:
    friend class ImageBindings;
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    id<MTLResidencySet> residency = nil;
    mutable std::mutex mutex;
    std::map<void*, std::uint32_t> residentCounts;
    std::unique_ptr<Context> imageContext;
    std::unique_ptr<TextureDetiler> detiler;
    std::unique_ptr<PlaceholderImages> placeholderImages;
    std::unique_ptr<SamplerCache> upstreamSamplers;
    std::unique_ptr<TextureCache> textureCache;
    bool tornDown = false;
};

}

#endif
