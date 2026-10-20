// Metal twin of Graphics/src/MultisampleTarget.cpp (TWIN-B, for agc_multisample_guest_access): the host
// image of a multisampled color target is a Metal 2D multisample texture made through D4's Metal::Vk image
// calls (MetalVulkanImages.h) and cleared by a render pass. The guest-access guards (CPU reads, DMA
// copies, texture and storage image reads of the target's memory throw), the CMASK fast-clear model and
// the per-device target list are upstream's, unchanged. CB resolves throw: the Metal draw path does not
// render multisampled color targets yet (MetalDraw.mm, TechnicalDebt metal-d6-targets).
// Upstream's target retirement on unmap (f7fd7569f), absorbed compute fills (5acd68f94), sampling a
// target or a never-rendered uniform MSAA surface (4b71846fa, 6e6334e7e, 84f636494, d0cd50c82) and
// FMASK decompress (9394a6f51) are ported; sampling goes through the D27 view of the host image.
// Upstream sync 4 (5d9dcc42d2, ee1ba73a2f da9fe3c708): the read guard and unmap observer are shared with
// DepthSurface.mm's resident-depth guards (InstallResidentImageReadGuard / UpdateResidentImageUnmapObserver).
// Upstream sync 5 (0d078f75d3): uniform multisampled DCC fast clears are imported as the target's initial
// clear and the DCC and CMASK ranges guarded (a0c51d5778 6eebcc2ce1 e040cb7f75), a target made from a
// filled texture takes the CMASK of its first draw (fromTexture), and a target is sampled through a
// format differing only in red/blue order with swapped components (90c3b61c6c). Upstream's storage
// image writes to a multisampled target (5c42e97f47, MultisampleTargetStorageView) are the D22/D28
// storage shadow here (MultisampleStorageTexture): Metal cannot write a multisampled texture from a
// shader, so MultisampleTargetStorageView refuses and no target is ever storable.
// Upstream sync 6 (661f8b7171): commands recorded on the target's own device (c83a7531e2 65f6549e0f),
// retired targets reclaimed once the recorder no longer keeps them (785f8f7bcf), CMASK codes read as
// MSAA codes (12b20e17f6), a full DCC fill of a resident target taken as its fast clear (74a6bdaa01),
// watched-range checks before writes and named DCC ranges (3738088b1a 4b674a72ba 58f0e89ca3), and
// sampling undrawn Z/D 64 KiB swizzle textures and Z ones whose HTILE is cleared (69afb0569f 24bc68375d).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalCommandStream.h"
#include "MetalVulkanImages.h"
#include <shared_mutex>
#include <stdexcept>
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

// Upstream 69afb0569f: the swizzle block of a multisampled color footprint (0: not modeled).
std::uint32_t MultisampledBlockLog2(ColorTileMode mode) {
    switch (mode) {
        case ColorTileMode::Standard4KB: return 12u;
        case ColorTileMode::Standard64KB: case ColorTileMode::Z64KBX: case ColorTileMode::D64KBX: case ColorTileMode::RenderTarget: return 16u;
        default: return 0u;
    }
}

// Upstream 58f0e89ca3: the write generation of the aligned 64 KiB blocks around a range (an aligned
// block import watches whole blocks), else of the range itself.
std::uint64_t collectBlocks(std::uint64_t address, std::size_t bytes) {
    constexpr std::uint64_t block = 65536;
    const auto first = address & ~(block - 1);
    const auto stop = (address + bytes + block - 1) & ~(block - 1);
    const auto generation = GuestMemory::CollectWritesUncached(first, static_cast<std::size_t>(stop - first));
    return generation != 0 ? generation : GuestMemory::CollectWritesUncached(address, bytes);
}

// Upstream 90c3b61c6c: a target read through a format differing only in red/blue order.
bool redBlueSwapped(VkFormat rendered, VkFormat sampled) {
    const auto pair = [&](VkFormat a, VkFormat b) { return (rendered == a && sampled == b) || (rendered == b && sampled == a); };
    return pair(VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM) || pair(VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB) || pair(VK_FORMAT_B8G8R8A8_SNORM, VK_FORMAT_R8G8B8A8_SNORM) || pair(VK_FORMAT_B8G8R8A8_UINT, VK_FORMAT_R8G8B8A8_UINT) || pair(VK_FORMAT_B8G8R8A8_SINT, VK_FORMAT_R8G8B8A8_SINT);
}

VkComponentMapping swapRedBlue(VkComponentMapping components) {
    const auto swapped = [](VkComponentSwizzle swizzle, VkComponentSwizzle identity) {
        if (swizzle == VK_COMPONENT_SWIZZLE_IDENTITY) swizzle = identity;
        if (swizzle == VK_COMPONENT_SWIZZLE_R) return VK_COMPONENT_SWIZZLE_B;
        if (swizzle == VK_COMPONENT_SWIZZLE_B) return VK_COMPONENT_SWIZZLE_R;
        return swizzle;
    };
    return {swapped(components.r, VK_COMPONENT_SWIZZLE_R), swapped(components.g, VK_COMPONENT_SWIZZLE_G), swapped(components.b, VK_COMPONENT_SWIZZLE_B), swapped(components.a, VK_COMPONENT_SWIZZLE_A)};
}

bool sameTarget(const ColorTarget& a, const ColorTarget& b) {
    return a.surfaceAddress == b.surfaceAddress && a.format == b.format && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.samples == b.samples;
}

VkClearColorValue fastClearColor(const ColorTarget& target) {
    Require(target.elementBytes != 0 && target.elementBytes <= sizeof(target.clearWords) && 16u % target.elementBytes == 0, "the fast clear of a multisampled target over 64-bit texels is not modeled");
    std::array<std::byte, 16> texels{};
    for (std::size_t offset = 0; offset < texels.size(); offset += target.elementBytes) std::memcpy(texels.data() + offset, target.clearWords.data(), target.elementBytes);
    std::array<std::uint32_t, 4> pattern{};
    std::memcpy(pattern.data(), texels.data(), texels.size());
    VkClearColorValue clear{};
    Require(ClearColorForTexel(target.format, target.elementBytes, pattern, clear), "the fast-clear color of a multisampled target has no exact clear value in its format");
    return clear;
}

template<typename Record>
void recordCommands(const Context& context, Record&& record) {
    auto* recorder = Recorder::Active();
    // Upstream c83a7531e2: only the target's own device's recorder records its commands.
    if (recorder != nullptr && recorder->Device() != context.device) recorder = nullptr;
    std::unique_ptr<CommandBatch> batch;
    if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
    record(recorder != nullptr ? recorder->Commands() : batch->Handle(), recorder);
    if (batch) batch->SubmitAndWait();
}

class MultisampleTarget : public std::enable_shared_from_this<MultisampleTarget> {
public:
    MultisampleTarget(const Context& context, const ColorTarget& target) : context(context), target(target), cmask(target.cmaskAddress, target.cmaskBytes), colorFill(target) {
        this->context.bufferPool.reset();
        if (target.dccAddress != 0) metadataValid = std::make_shared<std::atomic<bool>>(true);
        const auto initial = initialDccClear(target);
        id<MTLDevice> device = Metal::DeviceOf(context);
        Require(device != nil, "a multisampled color target needs a Context carrying a Metal device");
        // Sample counts the device cannot render are slices of a 2D multisample array (DECISIONS D28:
        // 8x on Apple9 is two 4x halves).
        host = Metal::HostSamplesOf(device, target.samples);
        // The host image's size for the recorder's kept-byte budget (upstream allocationBytes).
        allocationBytes = static_cast<std::uint64_t>(target.extent.width) * target.extent.height * target.elementBytes * host.samples * host.slices;
        const auto format = Metal::PixelFormatOf(target.format);
        Require(format.format != MTLPixelFormatInvalid && format.blockWidth == 1u, "multisampled render-target format cannot be a color attachment");
        Require(target.extent.width <= 16384u && target.extent.height <= 16384u, "multisampled render target exceeds device image limits");
        try {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            // A 2D multisample array (MetalVulkanImages.mm CreateImage): sampled through
            // texture2d_ms_array (MslAbi Version 9), rendered slice by slice (DECISIONS D28).
            info.flags = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
            info.format = target.format;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = host.slices;
            info.samples = static_cast<VkSampleCountFlagBits>(host.samples);
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(Metal::Vk::CreateImage(context.device, &info, nullptr, &image), "vkCreateImage multisampled target");
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            viewInfo.format = target.format;
            viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, host.slices};
            Check(Metal::Vk::CreateImageView(context.device, &viewInfo, nullptr, &view), "vkCreateImageView multisampled target");
            // Metal cannot write a multisampled texture from compute: the initial clear (zero, or the
            // imported uniform DCC fast clear) is a render pass that loads the clear value
            // (MetalVulkanImages.mm ClearByRenderPass).
            recordCommands(context, [&](VkCommandBuffer commands, Recorder* recorder) {
                Metal::Vk::CmdClearColorImage(commands, image, VK_IMAGE_LAYOUT_GENERAL, &initial, 1, &viewInfo.subresourceRange);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
                if (recorder != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
            });
        } catch (...) {
            release();
            throw;
        }
    }
    ~MultisampleTarget() { release(); }
    MultisampleTarget(const MultisampleTarget&) = delete;
    MultisampleTarget& operator=(const MultisampleTarget&) = delete;

    // Upstream 785f8f7bcf: the device's recorder keeps the target (and its bytes) until the batch using
    // it completes, so a retired target is destroyed only after its last recorded use.
    void Keep() {
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Device() == context.device) {
            recorder->Keep(shared_from_this());
            recorder->KeepBytes(this, static_cast<std::size_t>(allocationBytes));
        }
    }

    void ApplyFastClear(const ColorTarget& use) {
        // A CMASK fast clear, a uniform fill of the target's color memory (upstream 1baf077a7), or a
        // full fill of its DCC keys (upstream 74a6bdaa01): each clears every sample of every host
        // slice (DECISIONS D28).
        VkClearColorValue clear{};
        const bool refilled = absorbDccFill(use, clear);
        checkDccCurrent(use, refilled);
        if (!refilled && !TakeMultisampledClear(cmask, colorFill, use, clear)) return;
        Keep();
        recordCommands(context, [&](VkCommandBuffer commands, Recorder* recorder) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, host.slices};
            Metal::Vk::CmdClearColorImage(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
            if (recorder != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        });
        // The clear covers every sample: the shadow's writes are gone, the host image is newer.
        shadowNewer = false;
        hostNewer = true;
    }

    // Storage shadow coherence (DECISIONS D22, D28), in recording order: guest storage writes and
    // atomics go to the shadow (a single-sample 2D array of n slices), draws, fast clears and
    // resolves use the host image. Each side is brought up to date before the other is used.
    void SyncHost() {
        if (!shadowNewer) return;
        recordCommands(context, [&](VkCommandBuffer commands, Recorder* recorder) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
            Metal::CopyStorageToMultisample(commands, Metal::TextureOfImage(shadow), Metal::TextureOfImage(image), target.samples);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            if (recorder != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        });
        shadowNewer = false;
        ++shadowToHost;
    }

    // The sampled view of the host image (upstream 4b71846fa; on Metal the D27 view the MSL reads as
    // texture2d_ms_array), after a pending fast clear or color memory fill (upstream 6e6334e7e) and the
    // storage shadow's writes. Called with targetsMutex held.
    // The texture shares the target's ownership (upstream 785f8f7bcf), so a reclaimed target outlives
    // its users. Upstream 729558cc66's view type needs no key: the D27 view is a multisample array view
    // (texture2d_ms_array) for 2D MSAA and 2D MSAA array T#s alike.
    std::shared_ptr<Texture> Sampled(VkComponentMapping components) {
        Require(htileRefusal.empty(), htileRefusal);
        Keep();
        ApplyFastClear(target);
        SyncHost();
        const std::array<std::uint32_t, 4> key{static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a)};
        auto& texture = sampled[key];
        if (texture == nullptr) texture = std::make_shared<Texture>(context, image, target.format, VK_IMAGE_ASPECT_COLOR_BIT, components);
        return {shared_from_this(), texture.get()};
    }

    // Upstream 74a6bdaa01: the last fill over the whole DCC or CMASK range (nullopt after a partial one).
    void NoteMetadataFill(std::uint64_t fillAddress, std::size_t fillBytes, std::uint32_t pattern) {
        const auto note = [&](std::uint64_t address, std::uint64_t bytes, std::optional<std::uint32_t>& filled) {
            if (address == 0 || fillAddress >= address + bytes || address >= fillAddress + fillBytes) return;
            filled = fillAddress <= address && fillAddress + fillBytes >= address + bytes ? std::optional<std::uint32_t>(pattern) : std::nullopt;
        };
        note(target.dccAddress, MultisampledDccBytes(target), dccFill);
        note(target.cmaskAddress, target.cmaskBytes, cmaskFill);
    }

    void SyncShadow() {
        if (shadow == VK_NULL_HANDLE) {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = target.format;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = target.samples;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(Metal::Vk::CreateImage(context.device, &info, nullptr, &shadow), "vkCreateImage multisample storage shadow");
            allocationBytes += static_cast<std::uint64_t>(target.extent.width) * target.extent.height * target.elementBytes * target.samples;
            shadowView = std::make_shared<Texture>(context, shadow, target.format, VK_IMAGE_ASPECT_COLOR_BIT, VkComponentMapping{});
            hostNewer = true;
        }
        if (!hostNewer) return;
        recordCommands(context, [&](VkCommandBuffer commands, Recorder* recorder) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            Metal::CopyMultisampleToStorage(commands, Metal::TextureOfImage(image), Metal::TextureOfImage(shadow), target.samples);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            if (recorder != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        });
        hostNewer = false;
        ++hostToShadow;
    }

    Context context;
    ColorTarget target;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    // The host layout of the guest sample count (DECISIONS D28).
    Metal::HostSamples host;
    MultisampledCmask cmask;
    MultisampledColorFill colorFill;
    // Cleared when the target's memory is unmapped (upstream f7fd7569f): a retired target is no longer
    // found, guarded, filled or sampled; it is destroyed with the device's targets.
    std::shared_ptr<std::atomic<bool>> active = std::make_shared<std::atomic<bool>>(true);
    // Upstream a0c51d5778: cleared when the DCC or CMASK memory of an imported fast clear is unmapped.
    std::shared_ptr<std::atomic<bool>> metadataValid;
    // Made over a filled texture (createFilled): its first draw's CMASK replaces the empty one.
    bool fromTexture = false;
    // Upstream 69afb0569f 24bc68375d: the HTILE of a Z 64 KiB swizzle texture made by createFilled, and
    // why it is no longer modeled (a fill over it other than a whole clear to 0.0).
    std::uint64_t htileAddress = 0;
    std::uint64_t htileBytes = 0;
    std::string htileRefusal;
    // The host image's (and shadow's) bytes, kept with the target in the recorder's budget.
    std::uint64_t allocationBytes = 0;
    // Sampled views for guest MSAA T#s (DECISIONS D27), by component mapping.
    std::map<std::array<std::uint32_t, 4>, std::shared_ptr<Texture>> sampled;
    // The storage shadow (made at the first guest storage use) and which side holds the newest
    // samples; copies made each way (for tests and the cost report).
    VkImage shadow = VK_NULL_HANDLE;
    std::shared_ptr<Texture> shadowView;
    bool shadowNewer = false;
    bool hostNewer = true;
    std::uint64_t hostToShadow = 0;
    std::uint64_t shadowToHost = 0;

private:
    std::vector<GuestMemory::UnchangedQuery> dccRanges;
    std::optional<std::uint32_t> dccFill;
    std::optional<std::uint32_t> cmaskFill;

    // Upstream 74a6bdaa01: a fill of the whole DCC range with the uniform key byte its memory now holds
    // (and, with a CMASK, a CMASK that holds one uniform byte, filled when it was written) is the target's
    // fast clear; the metadata ranges then take their new write generations.
    bool absorbDccFill(const ColorTarget& use, VkClearColorValue& clear) {
        const auto dcc = std::exchange(dccFill, std::nullopt);
        const auto cmaskPattern = std::exchange(cmaskFill, std::nullopt);
        if (dccRanges.size() < 2 || !dcc || use.dccAddress != target.dccAddress || use.cmaskAddress != target.cmaskAddress || use.cmaskBytes != target.cmaskBytes) return false;
        const auto keys = LandedUniformMetadataByte(dccRanges[1].address, dccRanges[1].bytes);
        if (!keys || *dcc != *keys * 0x01010101u) return false;
        std::uint8_t cmaskKeys = 0;
        if (dccRanges.size() > 2) {
            const auto& range = dccRanges[2];
            auto* recorder = Recorder::Active();
            const bool written = !GuestMemory::UnchangedSince(range.address, range.bytes, range.generation) || (recorder != nullptr && recorder->PendingWriteOverlaps(range.address, range.bytes));
            const auto landed = LandedUniformMetadataByte(range.address, range.bytes);
            if (!landed || (written && (!cmaskPattern || *cmaskPattern != *landed * 0x01010101u))) return false;
            cmaskKeys = *landed;
        }
        std::vector<GuestMemory::UnchangedQuery> metadata(dccRanges.begin() + 1, dccRanges.end());
        for (auto& range : metadata) {
            if (!GuestMemory::Watched(range.address, range.bytes)) return false;
            range.generation = collectBlocks(range.address, range.bytes);
            if (range.generation == 0) return false;
        }
        clear = MultisampledDccClear(use, std::vector<std::byte>(dccRanges[1].bytes, std::byte{*keys}), std::vector<std::byte>(use.cmaskBytes, std::byte{cmaskKeys}));
        std::copy(metadata.begin(), metadata.end(), dccRanges.begin() + 1);
        return true;
    }

    // Upstream 4b674a72ba: a refusal naming the color, DCC or CMASK range.
    void requireDccRange(bool valid, const GuestMemory::UnchangedQuery& range, const char* reason) const {
        if (valid) return;
        const auto* plane = range.address == target.address ? "color" : range.address == target.dccAddress ? "DCC" : "CMASK";
        char text[384];
        std::snprintf(text, sizeof(text), "%s (%s range 0x%llx+0x%zx, %u samples)", reason, plane, static_cast<unsigned long long>(range.address), range.bytes, target.samples);
        Require(false, text);
    }

    VkClearColorValue initialDccClear(const ColorTarget& use) {
        if (use.dccAddress == 0) return {};
        const auto dccBytes = MultisampledDccBytes(use);
        Require((use.dccAddress & 4095u) == 0, "multisampled DCC metadata base alignment is not implemented");
        Require(use.cmaskAddress == 0 ? use.cmaskBytes == 0 : use.cmaskBytes == CmaskBytes(use.extent.width, use.extent.height), "multisampled DCC CMASK footprint is invalid");
        const auto colorBytes = MultisampledColorBytes(use);
        Require(colorBytes != 0, "multisampled DCC color footprint is not implemented");
        dccRanges.push_back({use.address, static_cast<std::size_t>(colorBytes), 0});
        dccRanges.push_back({use.dccAddress, dccBytes, 0});
        if (use.cmaskAddress != 0) dccRanges.push_back({use.cmaskAddress, use.cmaskBytes, 0});
        for (auto& range : dccRanges) {
            GuestMemory::CheckRange(reinterpret_cast<const void*>(range.address), range.bytes, 256);
            Require(GuestMemory::Watched(range.address, range.bytes), "initial multisampled DCC contents require watched guest memory");
            range.generation = collectBlocks(range.address, range.bytes);
            Require(range.generation != 0, "initial multisampled DCC contents have no write generation");
        }
        for (std::size_t first = 0; first < dccRanges.size(); ++first) {
            for (std::size_t second = first + 1; second < dccRanges.size(); ++second) {
                const auto& a = dccRanges[first];
                const auto& b = dccRanges[second];
                Require(a.address >= b.address + b.bytes || b.address >= a.address + a.bytes, "overlapping multisampled DCC color or metadata ranges are not implemented");
            }
        }
        std::vector<std::byte> dcc(dccRanges[1].bytes);
        GuestMemory::Read(use.dccAddress, dcc);
        std::vector<std::byte> cmaskBytes(use.cmaskBytes);
        if (!cmaskBytes.empty()) GuestMemory::Read(use.cmaskAddress, cmaskBytes);
        const auto clear = MultisampledDccClear(use, dcc, cmaskBytes);
        Require(GuestMemory::UnchangedSinceAll(dccRanges), "initial multisampled DCC contents changed during import");
        return clear;
    }

    // `metadataRefilled`: absorbDccFill took a fill of the metadata, so only the color range is checked.
    void checkDccCurrent(const ColorTarget& use, bool metadataRefilled) const {
        if (dccRanges.empty()) {
            Require(use.dccAddress == 0, "adding DCC to a resident multisampled color target is not implemented");
            return;
        }
        Require(metadataValid->load(std::memory_order_acquire), "multisampled DCC metadata was released while its color target remains resident");
        static_cast<void>(MultisampledDccBytes(use));
        Require(use.address == target.address && use.dccAddress == target.dccAddress && use.dccControl == target.dccControl && use.dccAlphaOnMsb == target.dccAlphaOnMsb && use.dccPipeAligned == target.dccPipeAligned && use.cmaskAddress == target.cmaskAddress && use.cmaskBytes == target.cmaskBytes && use.tileMode == target.tileMode && use.elementBytes == target.elementBytes && use.pipeBankXor == target.pipeBankXor, "multisampled DCC target layout or metadata changed");
        // Upstream 3738088b1a: every range is checked watched before any is checked for writes.
        const auto ranges = std::span(dccRanges).first(metadataRefilled ? 1u : dccRanges.size());
        for (const auto& range : ranges) requireDccRange(GuestMemory::Watched(range.address, range.bytes), range, "multisampled DCC target reuse requires watched guest memory");
        for (const auto& range : ranges) {
            GuestMemory::CollectWritesUncached(range.address, range.bytes);
            requireDccRange(GuestMemory::UnchangedSince(range.address, range.bytes, range.generation) || !GuestMemory::StoredOver(range.address, range.bytes, range.generation), range, "writes to a resident multisampled DCC target or its metadata are not implemented");
            if (auto* recorder = Recorder::Active(); recorder != nullptr) requireDccRange(!recorder->PendingWriteOverlaps(range.address, range.bytes), range, "pending writes to a resident multisampled DCC target or its metadata are not implemented");
        }
    }

    // A retired target is destroyed once no recorder keeps it (Keep) and no sampled texture shares it
    // (upstream 785f8f7bcf); the device's targets go at teardown (ClearMultisampleTargets), after the
    // device has drained its batches.
    void release() noexcept {
        sampled.clear();
        shadowView.reset();
        if (shadow) Metal::Vk::DestroyImage(context.device, shadow, nullptr);
        if (view) Metal::Vk::DestroyImageView(context.device, view, nullptr);
        if (image) Metal::Vk::DestroyImage(context.device, image, nullptr);
    }
};

struct GuardedRange {
    VkDevice device;
    std::uint64_t address;
    std::uint64_t bytes;
    std::uint32_t samples;
    std::shared_ptr<std::atomic<bool>> active;
    std::shared_ptr<std::atomic<bool>> metadataValid;
};

// Upstream 4d37c9336: winpthreads initializes a static rwlock on its first lock and fails a
// concurrent first lock, so the mutex is locked once at construction. Harmless on macOS; kept in
// step with the original.
class PrimedSharedMutex : public std::shared_mutex {
public:
    PrimedSharedMutex() {
        lock();
        unlock();
    }
};

std::shared_mutex& guardedMutex() {
    static PrimedSharedMutex mutex;
    return mutex;
}

std::vector<GuardedRange>& guarded() {
    static std::vector<GuardedRange> ranges;
    return ranges;
}

std::atomic<bool>& anyGuarded() {
    static std::atomic<bool> any{false};
    return any;
}

void guardReads(std::uint64_t address, std::size_t bytes) {
    RequireNoMultisampleTarget(address, bytes, "a CPU read");
}

// GuestAllocations' unmap observer (upstream f7fd7569f): a target whose whole guarded range is unmapped
// is retired, so memory mapped there again is not taken for it.
void retireUnmapped(std::uintptr_t address, std::size_t bytes) {
    RetireUnmappedDepth_nid_no_patch(address, bytes);
    std::unique_lock lock(guardedMutex());
    std::erase_if(guarded(), [&](const auto& range) {
        if (range.metadataValid && address < range.address + range.bytes && range.address < address + bytes) range.metadataValid->store(false, std::memory_order_release);
        if (range.address < address || range.address - address > bytes || range.bytes > bytes - (range.address - address)) return false;
        if (!range.metadataValid) range.active->store(false, std::memory_order_release);
        return true;
    });
    std::erase_if(guarded(), [](const auto& range) { return range.metadataValid && !range.active->load(std::memory_order_acquire); });
    anyGuarded().store(!guarded().empty(), std::memory_order_release);
}

void guard(const Context& context, const ColorTarget& target, const std::shared_ptr<std::atomic<bool>>& active, const std::shared_ptr<std::atomic<bool>>& metadataValid) {
    auto bytes = std::max<std::uint64_t>(target.bytes * target.samples, static_cast<std::uint64_t>(target.extent.width) * target.extent.height * target.elementBytes * target.samples);
    if (MultisampledBlockLog2(target.tileMode) == 16u && target.tileMode != ColorTileMode::Standard64KB && std::has_single_bit(target.elementBytes) && std::has_single_bit(target.samples)) {
        const auto log2Samples = std::countr_zero(target.samples);
        const auto log2Pixels = 16u - std::countr_zero(target.elementBytes) - log2Samples;
        const auto log2Width = (log2Pixels + ((log2Samples & 1u) == 0u ? 1u : 0u)) / 2u;
        const auto blockWidth = 1u << log2Width;
        const auto blockHeight = 1u << (log2Pixels - log2Width);
        const auto align = [](std::uint32_t extent, std::uint32_t block) { return (static_cast<std::uint64_t>(extent) + block - 1u) / block * block; };
        bytes = align(target.extent.width, blockWidth) * align(target.extent.height, blockHeight) * target.elementBytes * target.samples;
    }
    std::unique_lock lock(guardedMutex());
    guarded().push_back({context.device, target.address, bytes, target.samples, active});
    if (target.dccAddress != 0) {
        guarded().push_back({context.device, target.dccAddress, MultisampledDccBytes(target), target.samples, active, metadataValid});
        if (target.cmaskAddress != 0) guarded().push_back({context.device, target.cmaskAddress, target.cmaskBytes, target.samples, active, metadataValid});
    }
    anyGuarded().store(true, std::memory_order_release);
    InstallResidentImageReadGuard_nid_no_patch();
}

std::mutex& targetsMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::shared_ptr<MultisampleTarget>>& targets() {
    static std::vector<std::shared_ptr<MultisampleTarget>> list;
    return list;
}

// Upstream 785f8f7bcf: retired targets leave the list (a recorder or a sampled texture may still keep
// one alive until its last use).
void discardRetiredTargets() {
    std::erase_if(targets(), [](const auto& target) { return !target->active->load(std::memory_order_acquire); });
}

MultisampleTarget& acquire(const Context& context, const ColorTarget& use) {
    discardRetiredTargets();
    for (const auto& target : targets()) {
        if (!target->active->load(std::memory_order_acquire)) continue;
        if (target->context.device != context.device || !sameTarget(target->target, use)) continue;
        if (target->fromTexture && use.cmaskAddress != 0) {
            target->fromTexture = false;
            target->cmask = MultisampledCmask(use.cmaskAddress, use.cmaskBytes);
        }
        target->cmask.CheckAddress(use.cmaskAddress);
        target->Keep();
        target->ApplyFastClear(use);
        target->SyncHost();
        return *target;
    }
    auto target = std::make_shared<MultisampleTarget>(context, use);
    target->Keep();
    targets().push_back(std::move(target));
    auto& created = *targets().back();
    guard(context, use, created.active, created.metadataValid);
    // Upstream 12b20e17f6: the CMASK's uniform byte read as an MSAA code.
    if (use.cmaskFastClear && use.dccAddress == 0) created.cmask.SeedCode(CurrentUniformMetadataByte(use.cmaskAddress, static_cast<std::size_t>(use.cmaskBytes)));
    created.ApplyFastClear(use);
    return created;
}

// A multisampled texture no draw rendered to whose color memory holds one repeated 32-bit value
// (upstream d0cd50c82): a target is made over it with that value as its pending clear, which the
// first sample applies. Anything else refuses. Called with targetsMutex held.
MultisampleTarget& createFilled(const Context& context, const GuestTextureResource& resource, VkFormat format, const char* access = "sampling") {
    const auto refuse = [&](const std::string& reason) {
        char text[192];
        std::snprintf(text, sizeof(text), "AGC graphics: %s the %ux%u multisampled texture 0x%llx of %u samples, which no draw rendered to and ", access, resource.width, resource.height, static_cast<unsigned long long>(resource.baseAddress), resource.samples);
        throw std::runtime_error(text + reason + ", is not implemented");
    };
    ColorTarget use{};
    switch (resource.tileMode) {
        case TextureTileMode::kStandard4KB: use.tileMode = ColorTileMode::Standard4KB; break;
        case TextureTileMode::kStandard64KB: use.tileMode = ColorTileMode::Standard64KB; break;
        case TextureTileMode::kZ64KBX: use.tileMode = ColorTileMode::Z64KBX; break;
        case TextureTileMode::kD64KBX: use.tileMode = ColorTileMode::D64KBX; break;
        case TextureTileMode::kR64KBX: use.tileMode = ColorTileMode::RenderTarget; break;
        default: refuse("whose tile mode " + std::to_string(static_cast<int>(resource.tileMode)) + " has no modeled multisampled color footprint");
    }
    use.address = resource.baseAddress;
    use.surfaceAddress = resource.baseAddress;
    use.extent = {resource.width, resource.height};
    use.surfaceExtent = use.extent;
    use.format = format;
    use.elementBytes = BytesPerElement(resource.format);
    use.pipeBankXor = resource.pipeBankXor;
    use.samples = resource.samples;
    const auto bytes = MultisampledColorBytes(use);
    if (bytes == 0) refuse("whose " + std::to_string(use.elementBytes) + "-byte elements have no modeled multisampled color footprint");
    use.bytes = ColorTargetLayout(use.extent.width, use.extent.height, use.tileMode, use.elementBytes).Bytes();
    char range[64];
    std::snprintf(range, sizeof(range), "0x%llx+0x%llx", static_cast<unsigned long long>(use.address), static_cast<unsigned long long>(bytes));
    RequireNoMultisampleTarget(use.address, static_cast<std::size_t>(bytes), "sampling a multisampled texture no draw rendered to");
    GuestMemory::FlushGpuWrites(use.address, static_cast<std::size_t>(bytes));
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(use.address), static_cast<std::size_t>(bytes))) refuse(std::string("whose color memory ") + range + " is not mapped");
    // Upstream 69afb0569f: a Z 64 KiB swizzle texture's metadata (its T#'s compression address) is an
    // HTILE; all of it cleared (0: depth 0.0) makes the texture read 0, all of it expanded (ZMask 0xf)
    // leaves the color memory as the texture's contents. Anything else refuses.
    std::uint64_t htileBytes = 0;
    bool htileCleared = false;
    if (resource.dccAddress != 0) {
        if (use.tileMode != ColorTileMode::Z64KBX) refuse("whose compression metadata over a tile mode " + std::to_string(static_cast<int>(resource.tileMode)) + " surface is not modeled");
        htileBytes = HtileMetadataBytes(use.extent);
        char htileRange[64];
        std::snprintf(htileRange, sizeof(htileRange), "0x%llx+0x%llx", static_cast<unsigned long long>(resource.dccAddress), static_cast<unsigned long long>(htileBytes));
        GuestMemory::FlushGpuWrites(resource.dccAddress, static_cast<std::size_t>(htileBytes));
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(resource.dccAddress), static_cast<std::size_t>(htileBytes))) refuse(std::string("whose HTILE ") + htileRange + " is not mapped");
        std::vector<std::uint32_t> words(static_cast<std::size_t>(htileBytes / sizeof(std::uint32_t)));
        std::memcpy(words.data(), reinterpret_cast<const void*>(resource.dccAddress), words.size() * sizeof(std::uint32_t));
        htileCleared = words.front() == 0u;
        const auto mixed = std::find_if(words.begin(), words.end(), [&](std::uint32_t word) { return htileCleared ? word != 0u : (word & 15u) != 15u; });
        if (mixed != words.end()) {
            char text[96];
            std::snprintf(text, sizeof(text), " holds 0x%08x at +0x%llx after 0x%08x", *mixed, static_cast<unsigned long long>((mixed - words.begin()) * sizeof(std::uint32_t)), words.front());
            refuse(std::string("whose HTILE ") + htileRange + text + ", which is neither all cleared to 0.0 nor all expanded");
        }
    }
    const auto* memory = reinterpret_cast<const std::byte*>(use.address);
    std::uint32_t value = 0;
    if (!htileCleared) std::memcpy(&value, memory, sizeof(value));
    char held[16];
    std::snprintf(held, sizeof(held), "0x%08x", value);
    const auto repeated = htileCleared ? memory + bytes : std::mismatch(memory + sizeof(value), memory + bytes, memory).first;
    if (repeated != memory + bytes) {
        char offset[32];
        std::snprintf(offset, sizeof(offset), "+0x%llx", static_cast<unsigned long long>(repeated - memory));
        refuse(std::string("whose color memory ") + range + " does not hold one repeated 32-bit value (" + held + " until " + offset + ")");
    }
    VkClearColorValue clear{};
    if (!MultisampledFillClearColor(use, value, clear)) refuse(std::string("whose color memory ") + range + " holds " + held + ", which has no exact clear value in its format");
    auto target = std::make_shared<MultisampleTarget>(context, use);
    target->Keep();
    targets().push_back(std::move(target));
    auto& created = *targets().back();
    guard(context, use, created.active, created.metadataValid);
    created.colorFill.NoteFill(use.address, static_cast<std::size_t>(bytes), value);
    created.fromTexture = true;
    created.htileAddress = resource.dccAddress;
    created.htileBytes = htileBytes;
    return created;
}

}

void InstallResidentImageReadGuard_nid_no_patch() {
    static std::once_flag installed;
    std::call_once(installed, [] { GuestMemory::SetReadGuard(&guardReads); });
    UpdateResidentImageUnmapObserver_nid_no_patch();
}

void UpdateResidentImageUnmapObserver_nid_no_patch() {
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    GuestAllocations::GuestAllocationsSetUnmapObserver_nid_postfix(anyGuarded().load(std::memory_order_acquire) || HasResidentDepthGuards_nid_no_patch() ? &retireUnmapped : nullptr);
}

// Upstream 12b20e17f6: a CMASK byte holds two 4-bit MSAA tile codes; both 0 is a register color clear,
// both 0b11xx (0xcc .. 0xff) color-expanded tiles. Other codes are not modeled.
std::optional<DccKeys> MultisampledCmaskKeys(std::uint8_t code) {
    const auto tile = code & 0xfu;
    if ((code >> 4u) != tile) return std::nullopt;
    if (tile == 0u) return DccKeys::Clear0000;
    if ((tile >> 2u) == 3u) return DccKeys::Uncompressed;
    return std::nullopt;
}

void MultisampledCmask::NoteFill(std::uint64_t fillAddress, std::size_t fillBytes, std::uint32_t pattern) {
    if (address == 0 || fillAddress >= address + bytes || address >= fillAddress + fillBytes) return;
    if (fillAddress > address || fillAddress + fillBytes < address + bytes) {
        char text[192];
        std::snprintf(text, sizeof(text), "a fill of 0x%llx+0x%zx partly covers the CMASK 0x%llx+0x%llx of a multisampled color target, leaving its tiles mixed", static_cast<unsigned long long>(fillAddress), fillBytes, static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
        refusal = text;
        return;
    }
    refusal.clear();
    const auto code = static_cast<std::uint8_t>(pattern);
    if (const auto filled = MultisampledCmaskKeys(code); filled && pattern == code * 0x01010101u) {
        keys = *filled;
    } else {
        char text[224];
        std::snprintf(text, sizeof(text), "a fill of the CMASK 0x%llx+0x%llx of a multisampled color target with 0x%08x leaves its tiles mixed (neither a register color clear 0 nor a color-expanded code such as 0xcc or 0xff)", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), pattern);
        refusal = text;
    }
}

void MultisampledCmask::SeedCode(std::optional<std::uint8_t> code) {
    if (const auto seeded = code ? MultisampledCmaskKeys(*code) : std::nullopt) {
        keys = *seeded;
        return;
    }
    char text[192];
    if (code) std::snprintf(text, sizeof(text), "the CMASK 0x%llx of a multisampled color target reads as code 0x%02x, which is not modeled", static_cast<unsigned long long>(address), *code);
    else std::snprintf(text, sizeof(text), "the CMASK 0x%llx of a multisampled color target reads as mixed, which is not modeled", static_cast<unsigned long long>(address));
    refusal = text;
}

void MultisampledCmask::CheckAddress(std::uint64_t cmaskAddress) const {
    if (cmaskAddress == address) return;
    char text[192];
    std::snprintf(text, sizeof(text), "a multisampled color target used CMASK 0x%llx after its first use with CMASK 0x%llx, which is not modeled", static_cast<unsigned long long>(cmaskAddress), static_cast<unsigned long long>(address));
    Require(false, std::string(text));
}

bool MultisampledCmask::TakeClear() {
    Require(refusal.empty(), refusal);
    if (keys != DccKeys::Clear0000) return false;
    keys = DccKeys::Uncompressed;
    return true;
}

std::uint64_t MultisampledColorBytes(const ColorTarget& target) {
    const auto blockLog2 = MultisampledBlockLog2(target.tileMode);
    if (blockLog2 == 0 || !std::has_single_bit(target.elementBytes) || !std::has_single_bit(target.samples)) return 0;
    const auto samplesLog2 = static_cast<std::uint32_t>(std::countr_zero(target.samples));
    const auto elementsLog2 = blockLog2 - static_cast<std::uint32_t>(std::countr_zero(target.elementBytes)) - samplesLog2;
    const auto widthLog2 = (elementsLog2 + ((samplesLog2 & 1u) == 0 ? 1u : 0u)) / 2u;
    const std::uint64_t blockWidth = std::uint64_t{1} << widthLog2;
    const std::uint64_t blockHeight = std::uint64_t{1} << (elementsLog2 - widthLog2);
    return ((target.extent.width + blockWidth - 1) / blockWidth) * ((target.extent.height + blockHeight - 1) / blockHeight) << blockLog2;
}

std::size_t MultisampledDccBytes(const ColorTarget& target) {
    Require(target.samples == 2 || target.samples == 4 || target.samples == 8, "unsupported multisampled DCC fragment count");
    Require(target.tileMode == ColorTileMode::RenderTarget && target.dccPipeAligned, "multisampled DCC requires pipe-aligned SW_64KB_R_X");
    Require(std::has_single_bit(target.elementBytes) && target.elementBytes <= 16, "unsupported multisampled DCC element size");
    Require(target.extent.width != 0 && target.extent.width <= 16384 && target.extent.height != 0 && target.extent.height <= 16384, "invalid multisampled DCC extent");
    Require(target.mipCount == 1 && target.mip == 0 && !target.mipTail && target.depth == 1 && target.depthSlice == 0 && target.arrayLastSlice == 0 && target.arraySlice == 0 && target.layers == 1, "multisampled DCC array, volume or mip contents are not implemented");
    const auto bits = 19u - static_cast<std::uint32_t>(std::countr_zero(target.elementBytes));
    const std::uint64_t width = std::uint64_t{1} << ((bits + 1u) / 2u);
    const std::uint64_t height = std::uint64_t{1} << (bits / 2u);
    return static_cast<std::size_t>(((target.extent.width + width - 1) / width) * ((target.extent.height + height - 1) / height) * 4096u);
}

VkClearColorValue MultisampledDccClear(const ColorTarget& target, std::span<const std::byte> dcc, std::span<const std::byte> cmask) {
    Require(target.dccAddress != 0 && (target.dccControl & 0x3ff80u) == 0 && (target.pipeBankXor & ~0xcf00u) == 0, "multisampled DCC address, color transform, lossy mode or XOR is not implemented");
    Require(dcc.size() == MultisampledDccBytes(target), "multisampled DCC metadata snapshot size mismatch");
    Require(std::ranges::all_of(dcc, [&](std::byte key) { return key == dcc.front(); }), "mixed multisampled DCC metadata is not implemented");
    Require(target.cmaskAddress == 0 ? cmask.empty() && target.cmaskBytes == 0 : cmask.size() == target.cmaskBytes && target.cmaskBytes == CmaskBytes(target.extent.width, target.extent.height), "multisampled DCC CMASK snapshot size mismatch");
    if (!cmask.empty()) {
        const auto code = std::to_integer<std::uint32_t>(cmask.front());
        const auto expanded = target.samples == 2 ? 0xddu : target.samples == 4 ? 0xeeu : 0xffu;
        Require(std::ranges::all_of(cmask, [&](std::byte key) { return key == cmask.front(); }) && (code == 0xccu || code == expanded), "multisampled DCC CMASK compression or register clear is not implemented");
    }
    DccKeys keys;
    switch (std::to_integer<std::uint32_t>(dcc.front())) {
        case 0x00u: keys = DccKeys::Clear0000; break;
        case 0x40u: keys = DccKeys::Clear0001; break;
        case 0x80u: keys = DccKeys::Clear1110; break;
        case 0xc0u: keys = DccKeys::Clear1111; break;
        default: throw std::runtime_error("initial multisampled DCC contents are not a supported uniform fast clear");
    }
    std::array<std::byte, 16> texels{};
    Require(FillDccClear(target.format, keys, target.dccAlphaOnMsb, texels), "multisampled DCC clear format is not implemented");
    std::array<std::uint32_t, 4> pattern{};
    std::memcpy(pattern.data(), texels.data(), texels.size());
    VkClearColorValue clear{};
    Require(ClearColorForTexel(target.format, target.elementBytes, pattern, clear), "multisampled DCC fast clear has no exact host value");
    return clear;
}

bool MultisampledFillClearColor(const ColorTarget& target, std::uint32_t fill, VkClearColorValue& clear) {
    const std::array<std::uint32_t, 4> pattern{fill, fill, fill, fill};
    VkFormat unorm = VK_FORMAT_UNDEFINED;
    switch (target.format) {
        case VK_FORMAT_R8G8B8A8_SRGB: unorm = VK_FORMAT_R8G8B8A8_UNORM; break;
        case VK_FORMAT_B8G8R8A8_SRGB: unorm = VK_FORMAT_B8G8R8A8_UNORM; break;
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32: unorm = VK_FORMAT_A8B8G8R8_UNORM_PACK32; break;
        default: return ClearColorForTexel(target.format, target.elementBytes, pattern, clear);
    }
    if (!ClearColorForTexel(unorm, target.elementBytes, pattern, clear)) return false;
    return std::all_of(clear.float32, clear.float32 + 3, [](float value) { return value == 0.0f || value == 1.0f; });
}

MultisampledColorFill::MultisampledColorFill(const ColorTarget& target) : address(target.address), bytes(MultisampledColorBytes(target)) {}

void MultisampledColorFill::NoteFill(std::uint64_t fillAddress, std::size_t fillBytes, std::uint32_t pattern) {
    if (bytes == 0 ? fillAddress > address || address >= fillAddress + fillBytes : fillAddress >= address + bytes || address >= fillAddress + fillBytes) return;
    char text[192];
    if (bytes == 0) {
        std::snprintf(text, sizeof(text), "a fill of 0x%llx+0x%zx covers the color memory at 0x%llx of a multisampled target whose footprint is not modeled", static_cast<unsigned long long>(fillAddress), fillBytes, static_cast<unsigned long long>(address));
        refusal = text;
        return;
    }
    if (fillAddress > address || fillAddress + fillBytes < address + bytes) {
        std::snprintf(text, sizeof(text), "a fill of 0x%llx+0x%zx partly covers the color memory 0x%llx+0x%llx of a multisampled target, leaving its samples mixed", static_cast<unsigned long long>(fillAddress), fillBytes, static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
        refusal = text;
        return;
    }
    refusal.clear();
    filled = pattern;
}

std::optional<std::uint32_t> MultisampledColorFill::TakeFill() {
    Require(refusal.empty(), refusal);
    return std::exchange(filled, std::nullopt);
}

void MultisampledColorFill::Discard() {
    refusal.clear();
    filled.reset();
}

bool TakeMultisampledClear(MultisampledCmask& cmask, MultisampledColorFill& colorFill, const ColorTarget& use, VkClearColorValue& clear) {
    if (cmask.TakeClear()) {
        colorFill.Discard();
        clear = fastClearColor(use);
        return true;
    }
    const auto fill = colorFill.TakeFill();
    if (!fill) return false;
    Require(MultisampledFillClearColor(use, *fill, clear), "a fill over the color memory of a multisampled target has no exact clear value in its format");
    return true;
}

VkImageView MultisampleTargetView(const Context& context, const ColorTarget& target, VkImage* image) {
    std::lock_guard lock(targetsMutex());
    // The draw that asks for the view renders into the host image.
    auto& acquired = acquire(context, target);
    acquired.hostNewer = true;
    if (image != nullptr) *image = acquired.image;
    return acquired.view;
}

// Upstream 785f8f7bcf: the view with a share of the target's ownership, for a caller that uses it
// beyond the recorder's keep (KeepMultisampleTarget keeps it in a later batch).
VkImageView MultisampleTargetView(const Context& context, const ColorTarget& target, std::shared_ptr<void>& owner, VkImage* image) {
    std::lock_guard lock(targetsMutex());
    auto& acquired = acquire(context, target);
    acquired.hostNewer = true;
    owner = acquired.shared_from_this();
    if (image != nullptr) *image = acquired.image;
    return acquired.view;
}

void KeepMultisampleTarget_nid_no_patch(const std::shared_ptr<void>& owner) {
    std::static_pointer_cast<MultisampleTarget>(owner)->Keep();
}

// Upstream 4b71846fa's entry for Graphics/src/ShaderResources.cpp, whose Metal twin reaches the same
// target through Metal::MultisampleSampledTexture (the T# words); kept for shared callers.
std::shared_ptr<Texture> MultisampleTargetTexture(const Context& context, const GuestTextureResource& resource, VkComponentMapping components, VkImageViewType) {
    std::lock_guard lock(targetsMutex());
    discardRetiredTargets();
    const auto format = ResolveTextureFormat(resource.format);
    const MultisampleTarget* other = nullptr;
    for (const auto& target : targets()) {
        if (!target->active->load(std::memory_order_acquire)) continue;
        if (target->context.device != context.device || target->target.surfaceAddress != resource.baseAddress) continue;
        const auto& rendered = target->target;
        const bool sameShape = rendered.extent.width == resource.width && rendered.extent.height == resource.height && rendered.samples == resource.samples && rendered.mip == 0;
        if (sameShape && rendered.format == format) return target->Sampled(components);
        if (sameShape && redBlueSwapped(rendered.format, format)) return target->Sampled(swapRedBlue(components));
        other = target.get();
    }
    if (other == nullptr) {
        Require(Recorder::Active() != nullptr, "a multisampled texture is only sampled by recorded draws and dispatches");
        return createFilled(context, resource, format, "sampling").Sampled(components);
    }
    char text[320];
    std::snprintf(text, sizeof(text), "AGC graphics: sampling multisampled color target 0x%llx (%ux%u, %u samples, vk format %d) as a %ux%u texture of %u samples and guest format %u (vk %d) is not implemented", static_cast<unsigned long long>(resource.baseAddress), other->target.extent.width, other->target.extent.height, other->target.samples, static_cast<int>(other->target.format), resource.width, resource.height, resource.samples, resource.format, static_cast<int>(format));
    throw std::runtime_error(text);
}

// Upstream 5c42e97f47 writes a multisampled color target through a storage view of its image. Metal
// shaders cannot write multisampled textures; guest MSAA storage T#s use the target's storage shadow
// (MultisampleStorageTexture, DECISIONS D22/D28), so this entry is never reached on Metal.
VkImageView MultisampleTargetStorageView(const Context&, const GuestTextureResource& resource) {
    char text[200];
    std::snprintf(text, sizeof(text), "AGC graphics: a storage view of the image of multisampled color target 0x%llx is not implemented on Metal (guest MSAA storage writes go through the storage shadow)", static_cast<unsigned long long>(resource.baseAddress));
    throw std::runtime_error(text);
}

bool MultisampleTargetStorable(const Context&, const ColorTarget&) {
    return false;
}

// FMASK decompress (upstream 9394a6f51): the samples live only in the host image, so the pass
// applies the target's pending fast clear or fill there.
void MaterializeMultisampleTargetClear(const Context& context, const ColorTarget& target) {
    std::lock_guard lock(targetsMutex());
    acquire(context, target);
}

// Upstream records a render pass whose resolve attachment is the destination; on Metal the pass's
// store action is MTLStoreActionMultisampleResolve, or, for the integer formats Metal does not
// resolve, a fragment pass that takes sample 0 (Metal::ResolveMultisample, DECISIONS D27).
void ResolveMultisampleTarget(const Context& context, const ColorTarget& source, StorageTexture& destination, VkImageView destinationView) {
    std::lock_guard lock(targetsMutex());
    auto& target = acquire(context, source);
    target.Keep();
    id<MTLTexture> samples = Metal::TextureOfView(target.view);
    id<MTLTexture> resolved = Metal::TextureOfView(destinationView);
    Require(resolved != nil, "a CB resolve destination without an attachment view");
    recordCommands(context, [&](VkCommandBuffer commands, Recorder* recorder) {
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        Metal::ResolveMultisample(commands, samples, resolved);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
        if (recorder != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
    });
    destination.MarkDirty();
}

namespace Metal {

namespace {

// The multisampled color target a guest MSAA T# addresses (DECISIONS D27), after its pending CMASK
// fast clear or color memory fill; `what` names the use in refusals. A sampled T# over memory no
// draw rendered to (`fillUnrendered`) gets a target holding its uniform color memory (upstream
// d0cd50c82). Called with targetsMutex held.
MultisampleTarget& TargetOfDescriptor(const Context& context, std::span<const std::uint32_t> words, const char* what, bool fillUnrendered, bool* swapped = nullptr) {
    Require(MultisampledDescriptor(words), std::string("a ") + what + " of a T# that is not 2D MSAA");
    const auto address = ((static_cast<std::uint64_t>(words[0]) | (static_cast<std::uint64_t>(words[1] & 0xffu) << 32u)) & 0xffffffffffull) << 8u;
    const auto guestFormat = (words[1] >> 20u) & 0x1ffu;
    const auto width = (((words[1] >> 30u) & 0x3u) | ((words[2] & 0xfffu) << 2u)) + 1u;
    const auto height = ((words[2] >> 14u) & 0x3fffu) + 1u;
    const auto baseLevel = (words[3] >> 12u) & 0xfu;
    const auto samples = 1u << ((words[3] >> 16u) & 0xfu);
    const auto type = (words[3] >> 28u) & 0xfu;
    const auto layers = ((words[4] >> 0u) & 0x1fffu) + 1u;
    const auto baseArray = (words[4] >> 16u) & 0x1fffu;
    char described[224];
    std::snprintf(described, sizeof(described), " (%s, T# 0x%llx, %ux%u, %u samples, format %u, type %u)", what, static_cast<unsigned long long>(address), width, height, samples, guestFormat, type);
    Require(baseLevel == 0u, std::string("AGC graphics: a 2D MSAA T# with a nonzero base level (fragment) is not modeled") + described);
    // Type 15 (2D MSAA array) views of a target's single layer read the same texture2d_ms_array view
    // (MslAbi Version 9); targets have one layer, as upstream's.
    Require(type == 14u || (layers == 1u && baseArray == 0u), std::string("AGC graphics: a 2D MSAA array T# over more than its first layer is not implemented (multisampled color targets have one layer)") + described);
    const auto format = ResolveTextureFormat(guestFormat);
    discardRetiredTargets();
    for (const auto& target : targets()) {
        if (!target->active->load(std::memory_order_acquire)) continue;
        const auto& color = target->target;
        if (target->context.device != context.device || color.address != address) continue;
        if (color.extent.width != width || color.extent.height != height || color.samples != samples) {
            // Upstream's MultisampleTargetTexture text (agc_driver_sample_multisample_target matches it).
            char text[320];
            std::snprintf(text, sizeof(text), "AGC graphics: sampling multisampled color target 0x%llx (%ux%u, %u samples, vk format %d) as a %ux%u texture of %u samples and guest format %u (vk %d) is not implemented", static_cast<unsigned long long>(address), color.extent.width, color.extent.height, color.samples, static_cast<int>(color.format), width, height, samples, guestFormat, static_cast<int>(format));
            throw std::runtime_error(std::string(text) + described);
        }
        // Upstream 90c3b61c6c: a sampled view in the red/blue-swapped format reads with swapped components.
        const bool swap = swapped != nullptr && redBlueSwapped(color.format, format);
        Require(color.format == format || swap, std::string("AGC graphics: a 2D MSAA T# reads a multisampled color target in another format, which is not implemented") + described);
        if (swapped != nullptr) *swapped = swap;
        target->ApplyFastClear(color);
        return *target;
    }
    if (fillUnrendered) {
        // Upstream MultisampleTargetTexture / MultisampleTargetStorageView: memory no draw rendered
        // to becomes a target holding its uniform color memory, for sampling and for writing alike.
        const bool storage = std::string_view(what) == "storage view";
        Require(Recorder::Active() != nullptr, storage ? "a multisampled color target is only written by recorded dispatches" : "a multisampled texture is only sampled by recorded draws and dispatches");
        auto& created = createFilled(context, DecodeTextureResource(words), format, storage ? "writing" : "sampling");
        created.ApplyFastClear(created.target);
        return created;
    }
    throw std::runtime_error(std::string("AGC graphics: a 2D MSAA T# addresses memory that is no multisampled color target; guest-written MSAA memory (sample layout, FMASK) is not modeled") + described);
}

}

std::shared_ptr<Texture> MultisampleSampledTexture(const Context& context, std::span<const std::uint32_t> words) {
    std::lock_guard lock(targetsMutex());
    bool swapped = false;
    auto& target = TargetOfDescriptor(context, words, "sampled view", true, &swapped);
    const auto swizzle = [](std::uint32_t select) {
        switch (select) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 4: return VK_COMPONENT_SWIZZLE_R;
        case 5: return VK_COMPONENT_SWIZZLE_G;
        case 6: return VK_COMPONENT_SWIZZLE_B;
        case 7: return VK_COMPONENT_SWIZZLE_A;
        default: throw std::runtime_error("AGC graphics: a 2D MSAA T# with a reserved destination select");
        }
    };
    const auto selects = words[3] & 0xfffu;
    const VkComponentMapping components{swizzle(selects & 7u), swizzle((selects >> 3u) & 7u), swizzle((selects >> 6u) & 7u), swizzle((selects >> 9u) & 7u)};
    return target.Sampled(swapped ? swapRedBlue(components) : components);
}

std::shared_ptr<Texture> MultisampleStorageTexture(const Context& context, std::span<const std::uint32_t> words) {
    std::lock_guard lock(targetsMutex());
    auto& target = TargetOfDescriptor(context, words, "storage view", true);
    // Upstream 24bc68375d: a Z texture whose HTILE a fill left unmodeled refuses storage access too.
    Require(target.htileRefusal.empty(), target.htileRefusal);
    target.Keep();
    // Brought up to date before the access; the access (recorded after this) writes the shadow.
    target.SyncShadow();
    target.shadowNewer = true;
    // The shadow's image lives as long as the target (upstream 785f8f7bcf's shared ownership).
    return {target.shared_from_this(), target.shadowView.get()};
}

MultisampleShadowStatistics MultisampleShadowCounts(std::uint64_t address) {
    std::lock_guard lock(targetsMutex());
    for (const auto& target : targets()) {
        if (target->active->load(std::memory_order_acquire) && target->target.address == address) return {target->hostToShadow, target->shadowToHost};
    }
    return {};
}

}

// A compute fill a live target takes whole as its clear (upstream 5acd68f94): BufferFill skips
// storing it to memory; NoteColorMetadataFill hands the target the clear.
bool MultisampledColorFillAbsorbs(std::uint64_t address, std::size_t bytes, std::uint32_t pattern) {
    if (bytes == 0) return false;
    std::lock_guard lock(targetsMutex());
    return std::any_of(targets().begin(), targets().end(), [&](const auto& target) {
        VkClearColorValue clear{};
        return target->active->load(std::memory_order_acquire) && target->target.address == address && MultisampledColorBytes(target->target) == bytes && MultisampledFillClearColor(target->target, pattern, clear);
    });
}

void NoteColorMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern) {
    std::lock_guard lock(targetsMutex());
    for (const auto& target : targets()) {
        if (!target->active->load(std::memory_order_acquire)) continue;
        target->colorFill.NoteFill(address, bytes, pattern);
        // Upstream 69afb0569f 24bc68375d: a whole clear of a filled Z texture's HTILE to 0 makes it read
        // 0.0; any other fill over it is not modeled.
        if (target->htileBytes != 0 && address < target->htileAddress + target->htileBytes && target->htileAddress < address + bytes) {
            if (pattern == 0u && address <= target->htileAddress && address + bytes >= target->htileAddress + target->htileBytes) {
                target->htileRefusal.clear();
                target->colorFill.NoteFill(target->target.address, static_cast<std::size_t>(MultisampledColorBytes(target->target)), 0u);
            } else {
                char text[224];
                std::snprintf(text, sizeof(text), "a fill of 0x%llx+0x%zx with 0x%08x over the HTILE 0x%llx+0x%llx of a sampled multisampled depth-layout texture is not modeled", static_cast<unsigned long long>(address), bytes, pattern, static_cast<unsigned long long>(target->htileAddress), static_cast<unsigned long long>(target->htileBytes));
                target->htileRefusal = text;
            }
        }
        // Upstream 74a6bdaa01: a DCC target notes its metadata fills (ApplyFastClear takes a full DCC fill).
        if (target->target.dccAddress != 0) {
            target->NoteMetadataFill(address, bytes, pattern);
            continue;
        }
        const auto before = target->cmask.Keys();
        target->cmask.NoteFill(address, bytes, pattern);
        if (before == DccKeys::Clear0000 || target->cmask.Keys() != DccKeys::Clear0000) continue;
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1) < 8) std::fprintf(stderr, "[gpu] multisampled color target 0x%llx fast-cleared through its CMASK 0x%llx (fill 0x%llx+0x%zx)\n", static_cast<unsigned long long>(target->target.surfaceAddress), static_cast<unsigned long long>(target->target.cmaskAddress), static_cast<unsigned long long>(address), bytes);
    }
}

void ClearMultisampleTargets(VkDevice device) {
    std::lock_guard lock(targetsMutex());
    std::erase_if(targets(), [&](const auto& target) { return target->context.device == device; });
    std::unique_lock guardLock(guardedMutex());
    std::erase_if(guarded(), [&](const auto& range) { return range.device == device; });
    anyGuarded().store(!guarded().empty(), std::memory_order_release);
    UpdateResidentImageUnmapObserver_nid_no_patch();
}

void RequireNoMultisampleTarget(std::uint64_t address, std::size_t bytes, const char* access) {
    RequireNoResidentDepthRead_nid_no_patch(address, bytes);
    if (bytes == 0 || !anyGuarded().load(std::memory_order_acquire)) return;
    std::shared_lock lock(guardedMutex());
    for (const auto& range : guarded()) {
        if (address >= range.address + range.bytes || range.address >= address + bytes) continue;
        char text[256];
        std::snprintf(text, sizeof(text), "AGC graphics: %s of 0x%llx+0x%zx overlaps multisampled color target 0x%llx+0x%llx (%u samples), whose samples are only on the host: guest access to it is not implemented", access, static_cast<unsigned long long>(address), bytes, static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(range.bytes), range.samples);
        throw std::runtime_error(text);
    }
}

}
