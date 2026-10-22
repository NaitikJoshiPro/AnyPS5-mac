#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_METALSAMPLEORDER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_METALSAMPLEORDER_HPP

// macOS only (specifics/macos/overlay): the Metal form of the guest sample order's reorder passes
// (upstream 57ceb9e370 adc46f3ea0 f8e8dc6f67, SampleOrder.cpp RecordSampleOrder; TechnicalDebt
// metal-sample-order). Metal's programmable sample positions are one pattern for every pixel, so a
// 2x2 location grid whose pixels hold one set of positions in different orders draws with pixel
// (0,0)'s pattern and puts each pixel's samples in the guest's order at the render pass bounds.
// Upstream moves the samples with storage-image compute; Metal compute cannot write multisampled
// textures, so the samples move through per-sample render passes (a fragment function reading
// [[sample_id]] runs per sample and writes only its sample): the target is copied into a scratch
// multisampled texture by one identity pass per slice, then every destination sample takes its
// source sample from the scratch (color through an integer view of the same texel size, so every
// bit moves; depth through [[depth(any)]], stencil through [[stencil]]). Implemented in
// specifics/macos/src/MetalSampleOrder.mm; this header stays includable from C++
// (Graphics/src/SampleOrder.cpp branches to it under __APPLE__).

#include "prx/libSceAgcDriver/Graphics/include/SampleOrder.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace AgcDriver::Graphics::Metal {

// A Context the Metal device filled (MetalCommandStream.h IsMetal): it resolves no Vulkan entry
// points and carries an id<MTLDevice>. Mock and Vulkan contexts keep upstream's paths.
inline bool SampleOrderContext(const Context& context) {
    return context.deviceProc == nullptr && context.device != VK_NULL_HANDLE;
}

// Upstream SampleOrderStorageFormat on the Metal device: the integer element format of the
// target's texel size (R8/R16/R32/RG32/RGBA32 UINT, TextureDetiler::ImageElementFormat) when the
// guest format has a Metal pixel format of that size and the sample count is a guest MSAA count
// (2, 4 or 8; 8 is emulated on Apple9, DECISIONS D28); VK_FORMAT_UNDEFINED otherwise.
VkFormat SampleOrderStorageFormat(const Context& context, VkFormat format, std::uint32_t elementBytes, std::uint32_t samples);
// Whether a multisampled depth target of `format` (guest) can be put in the guest's order exactly:
// every Metal depth surface (Depth16Unorm, Depth32Float, Depth32Float_Stencil8, Stencil8,
// DepthSurface.mm SurfaceFormat) is shader-readable and written back exactly by [[depth(any)]] and
// [[stencil]].
bool SampleOrderDepthExact(VkFormat format);

// The texel mapping of one reorder: `map[quad]` (quad = (x & 1) | (y & 1) << 1 in window
// coordinates) gives each destination guest sample s its source guest sample in nibble s
// (SampleOrder.cpp mapFor: the order word toward the host, its inverse toward the guest).
using SampleOrderMap = std::array<std::uint32_t, 4>;
SampleOrderMap SampleOrderMapFor(const SampleOrder& order, bool toHost);

// One multisampled texture to reorder: `image` a VkImage handle carrying its id<MTLTexture> (a 2D
// multisample array of HostSamplesOf's slices: guest sample s of layer l at slice l * p + s / h,
// sample s % h), `elementBytes` its texel size (color; 0 for a depth/stencil texture).
struct SampleOrderTexture {
    VkImage image = VK_NULL_HANDLE;
    std::uint32_t elementBytes = 0;
};

// Records the reorder of every texture into `commands`' stream: pixels inside `region` (from the
// origin) take map's samples, the others keep theirs. `guestSamples` is the guest count n (the
// texture holds n / h slices per layer). The scratch textures and views stay alive through the
// stream (CommandStream::Keep) until the GPU is done with them. Throws std::runtime_error for a
// texture that is not a 2D multisample array of a reorderable format.
void ReorderSamples(VkCommandBuffer commands, std::span<const SampleOrderTexture> textures, const SampleOrderMap& map, VkExtent2D region, std::uint32_t guestSamples);

// Upstream RecordSampleOrder on the Metal device (SampleOrder.cpp branches here): the targets'
// images reordered by ReorderSamples, their keep references added to `kept`.
void RecordSampleOrder(const Context& context, VkCommandBuffer commands, const SampleOrderPass& pass, std::span<const SampleOrderTarget> targets, bool toHost, std::vector<std::shared_ptr<void>>& kept);

// Reorder passes recorded so far (tests): one per ReorderSamples texture.
std::uint64_t SampleOrderReorders();

// Test seam (SampleOrderTwin.mm --grid-reference, the twin of upstream's fallback context without
// shaderStorageImageMultisample): while suppressed, the Metal device names no sample order
// (VulkanDevice::HostSampleOrder, MetalDraw.mm), so a 2x2 location grid draws with pixel (0,0)'s
// positions alone. Process-wide; only tests set it.
void SetSampleOrderSuppressed(bool suppressed);
bool SampleOrderSuppressed();

}

#endif
