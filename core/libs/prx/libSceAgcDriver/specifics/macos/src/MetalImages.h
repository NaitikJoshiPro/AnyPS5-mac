#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALIMAGES_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALIMAGES_H

// Work packages D4 (textures, detiling, formats) and D5 (descriptors, samplers): guest images and
// samplers of a dispatch's MslAbi descriptor table (docs/mac/M2_DESIGN.md §4, §6).
//
// A guest T# becomes an MTLTexture holding its surface detiled (MetalTextureDetiler.mm), viewed per
// image heap (type, numeric class, dst_sel swizzle, levels and layers of the descriptor); a guest S#
// becomes an MTLSamplerState (argument-buffer capable, cached by its words). Storage images are
// written back (re-tiled into guest memory) when the batch that wrote them completes. Absent
// elements (an all-zero T#) get a 1x1 placeholder of the heap's type.
//
// Bring-up limits (TechnicalDebt `metal-d4-texture-uploads`): every dispatch uploads its sampled
// and storage textures again (no cross-dispatch texture cache), detiling runs on the CPU, and
// GPU writes to a storage image reach guest memory at batch completion, not before a later
// dispatch of the same batch reads that memory as a buffer.
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

// One guest texture surface uploaded into a Metal texture (all its levels and layers).
struct GuestSurface {
    GuestTextureResource resource;
    SurfaceGeometry geometry;
    TextureFormatInfo format;
    id<MTLTexture> texture = nil;
    // Depth twin for comparison heaps (Depth32Float / Depth16Unorm), made on first use.
    id<MTLTexture> depth = nil;
    bool storage = false;
    bool atomic = false;
    // Views made for heaps, kept with the surface.
    std::vector<id<MTLTexture>> views;
};

// The images and samplers of one dispatch: Write() fills the table entries of each GuestImages and
// GuestSamplers binding (MTLResourceID values), Commit() hands the textures to the batch.
class ImageBindings {
public:
    // `settle(address, bytes)` makes recorded GPU writes over the range visible to the CPU before a
    // texture upload reads it (the device syncs when they overlap).
    ImageBindings(ImageCache& cache, std::function<void(std::uint64_t, std::uint64_t)> settle);
    ~ImageBindings();
    ImageBindings(const ImageBindings&) = delete;
    ImageBindings& operator=(const ImageBindings&) = delete;

    // Optional, before the first Write: scans all of the dispatch's bindings for what the samplers
    // depend on (the images paired with each sampler): ANISO_OVERRIDE samplers of single-level
    // images lose their anisotropy, FORCE_DEGAMMA samplers require sRGB images (as
    // ShaderResources::applyAnisoOverride / RequireDegammaFormat).
    // Also validates the bindless table metadata in the shader data against the bound heaps
    // (ShaderResources.cpp ValidateRuntimeResources).
    void Prepare(const ShaderRecompiler::RecompileResult& shader);
    // `entries`: the binding's first min(binding.count, range.count) table entries.
    void Write(const ShaderRecompiler::DescriptorBinding& binding, std::span<std::uint64_t> entries);
    // After the dispatch is recorded into the recorder's open batch: keeps the textures alive with
    // the batch, makes them resident, notes the storage images' guest ranges as pending GPU writes
    // and writes them back when the batch completes.
    void Commit(Recorder& recorder);

    bool Empty() const { return resources == nullptr && surfaces.empty() && samplers.empty(); }

private:
    std::shared_ptr<GuestSurface> surface(std::span<const std::uint32_t> words, bool storage, bool atomic, std::uint32_t mip);
    std::uint64_t view(GuestSurface& surface, std::uint32_t heap, bool firstLayer, std::uint32_t mip, bool baseLevelOnly = false);

    ImageCache& cache;
    std::function<void(std::uint64_t, std::uint64_t)> settle;
    std::vector<std::shared_ptr<GuestSurface>> surfaces;
    std::vector<id<MTLSamplerState>> samplers;
    // The upstream build (ShaderResources twin, MetalImagesOnly) once the cache is attached to the
    // device's Context: the image and sampler entries come from it, and the pending/live storage
    // registries, the texture caches and the resource cache are upstream's (Texture.mm,
    // ShaderResources.mm). Null: the bring-up path above (no cache attached).
    std::shared_ptr<ShaderResources> resources;
    // Storage surfaces and the levels the dispatch may write (written back at completion).
    std::map<GuestSurface*, std::set<std::uint32_t>> writtenLevels;
    // Prepare's adjusted sampler words per sampler element (empty: not prepared).
    std::vector<std::array<std::uint32_t, 4>> samplerWords;
    bool committed = false;
};

// Per device: the residency set of image resources (attached to the device's queue), the sampler
// cache and the placeholder textures.
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
    // The sampler of an S# (4 words) with the shader's use of it.
    id<MTLSamplerState> Sampler(std::span<const std::uint32_t> words, bool compare, bool unnormalized, bool integerBorder);
    // A 1x1 (one layer) zero texture of an image heap's type (RuntimeAbi binding 1-56).
    id<MTLTexture> Placeholder(std::uint32_t heap);

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

    struct Statistics {
        std::uint64_t textures = 0;
        std::uint64_t uploadedBytes = 0;
        std::uint64_t writtenBackBytes = 0;
        std::uint64_t samplers = 0;
        std::uint64_t samplerHits = 0;
    };
    Statistics Stats() const;

private:
    friend class ImageBindings;
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    id<MTLResidencySet> residency = nil;
    mutable std::mutex mutex;
    std::map<std::array<std::uint32_t, 5>, id<MTLSamplerState>> samplerCache;
    std::map<std::uint32_t, id<MTLTexture>> placeholders;
    std::map<void*, std::uint32_t> residentCounts;
    std::unique_ptr<Context> imageContext;
    std::unique_ptr<TextureDetiler> detiler;
    std::unique_ptr<PlaceholderImages> placeholderImages;
    std::unique_ptr<SamplerCache> upstreamSamplers;
    std::unique_ptr<TextureCache> textureCache;
    Statistics statistics;
    bool tornDown = false;
};

}

#endif
