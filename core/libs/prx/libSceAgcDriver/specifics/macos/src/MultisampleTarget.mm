// Metal twin of Graphics/src/MultisampleTarget.cpp (TWIN-B, for agc_multisample_guest_access): the host
// image of a multisampled color target is a Metal 2D multisample texture made through D4's Metal::Vk image
// calls (MetalVulkanImages.h) and cleared by a render pass. The guest-access guards (CPU reads, DMA
// copies, texture and storage image reads of the target's memory throw), the CMASK fast-clear model and
// the per-device target list are upstream's, unchanged. CB resolves throw: the Metal draw path does not
// render multisampled color targets yet (MetalDraw.mm, TechnicalDebt metal-d6-targets).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalCommandStream.h"
#include "MetalVulkanImages.h"
#include <shared_mutex>
#include <stdexcept>
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

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
    std::unique_ptr<CommandBatch> batch;
    if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
    record(recorder != nullptr ? recorder->Commands() : batch->Handle(), recorder);
    if (batch) batch->SubmitAndWait();
}

class MultisampleTarget {
public:
    MultisampleTarget(const Context& context, const ColorTarget& target) : context(context), target(target), cmask(target.cmaskAddress, target.cmaskBytes) {
        this->context.bufferPool.reset();
        id<MTLDevice> device = Metal::DeviceOf(context);
        Require(device != nil, "a multisampled color target needs a Context carrying a Metal device");
        // Sample counts the device cannot render are slices of a 2D multisample array (DECISIONS D28:
        // 8x on Apple9 is two 4x halves).
        host = Metal::HostSamplesOf(device, target.samples);
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
            // Metal cannot write a multisampled texture from compute: the zero clear is a render pass
            // that loads the clear value (MetalVulkanImages.mm ClearByRenderPass).
            recordCommands(context, [&](VkCommandBuffer commands, Recorder* recorder) {
                const VkClearColorValue zero{};
                Metal::Vk::CmdClearColorImage(commands, image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &viewInfo.subresourceRange);
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

    void ApplyFastClear(const ColorTarget& use) {
        if (!cmask.TakeClear()) return;
        const auto clear = fastClearColor(use);
        recordCommands(context, [&](VkCommandBuffer commands, Recorder* recorder) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, host.slices};
            Metal::Vk::CmdClearColorImage(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
            if (recorder != nullptr) Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        });
    }

    Context context;
    ColorTarget target;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    // The host layout of the guest sample count (DECISIONS D28).
    Metal::HostSamples host;
    MultisampledCmask cmask;
    // Sampled views for guest MSAA T#s (DECISIONS D27), by component mapping.
    std::map<std::uint32_t, std::shared_ptr<Texture>> sampled;

private:
    // The image may still be in use by recorded work only at device teardown (targets live for the
    // device's lifetime, ClearMultisampleTargets), after the device has drained its batches.
    void release() noexcept {
        sampled.clear();
        if (view) Metal::Vk::DestroyImageView(context.device, view, nullptr);
        if (image) Metal::Vk::DestroyImage(context.device, image, nullptr);
    }
};

struct GuardedRange {
    VkDevice device;
    std::uint64_t address;
    std::uint64_t bytes;
    std::uint32_t samples;
};

std::shared_mutex& guardedMutex() {
    static std::shared_mutex mutex;
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

void guard(const Context& context, const ColorTarget& target) {
    const auto bytes = std::max<std::uint64_t>(target.bytes * target.samples, static_cast<std::uint64_t>(target.extent.width) * target.extent.height * target.elementBytes * target.samples);
    std::unique_lock lock(guardedMutex());
    guarded().push_back({context.device, target.address, bytes, target.samples});
    anyGuarded().store(true, std::memory_order_release);
    static std::once_flag installed;
    std::call_once(installed, [] { GuestMemory::SetReadGuard(&guardReads); });
}

std::mutex& targetsMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::unique_ptr<MultisampleTarget>>& targets() {
    static std::vector<std::unique_ptr<MultisampleTarget>> list;
    return list;
}

MultisampleTarget& acquire(const Context& context, const ColorTarget& use) {
    for (const auto& target : targets()) {
        if (target->context.device != context.device || !sameTarget(target->target, use)) continue;
        target->cmask.CheckAddress(use.cmaskAddress);
        target->ApplyFastClear(use);
        return *target;
    }
    targets().push_back(std::make_unique<MultisampleTarget>(context, use));
    guard(context, use);
    auto& created = *targets().back();
    if (use.cmaskFastClear) created.cmask.SeedKeys(CurrentDccKeys(use.cmaskAddress, static_cast<std::uint64_t>(use.cmaskBytes) * 256u));
    created.ApplyFastClear(use);
    return created;
}

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
    if (pattern == 0) {
        keys = DccKeys::Clear0000;
    } else if (pattern == 0xffffffffu) {
        keys = DccKeys::Uncompressed;
    } else {
        char text[192];
        std::snprintf(text, sizeof(text), "a fill of the CMASK 0x%llx+0x%llx of a multisampled color target with 0x%08x leaves its tiles mixed (neither fast-cleared 0 nor expanded 0xffffffff)", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), pattern);
        refusal = text;
    }
}

void MultisampledCmask::SeedKeys(DccKeys seeded) {
    if (seeded == DccKeys::Clear0000 || seeded == DccKeys::Uncompressed) {
        keys = seeded;
        return;
    }
    char text[192];
    std::snprintf(text, sizeof(text), "the CMASK 0x%llx of a multisampled color target reads as %s, which is not modeled", static_cast<unsigned long long>(address), DccKeysName(seeded));
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

VkImageView MultisampleTargetView(const Context& context, const ColorTarget& target) {
    std::lock_guard lock(targetsMutex());
    return acquire(context, target).view;
}

// Upstream records a render pass whose resolve attachment is the destination; on Metal the pass's
// store action is MTLStoreActionMultisampleResolve, or, for the integer formats Metal does not
// resolve, a fragment pass that takes sample 0 (Metal::ResolveMultisample, DECISIONS D27).
void ResolveMultisampleTarget(const Context& context, const ColorTarget& source, StorageTexture& destination, VkImageView destinationView) {
    std::lock_guard lock(targetsMutex());
    auto& target = acquire(context, source);
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

std::shared_ptr<Texture> MultisampleSampledTexture(const Context& context, std::span<const std::uint32_t> words) {
    Require(MultisampledDescriptor(words), "a sampled multisampled view of a T# that is not 2D MSAA");
    const auto address = ((static_cast<std::uint64_t>(words[0]) | (static_cast<std::uint64_t>(words[1] & 0xffu) << 32u)) & 0xffffffffffull) << 8u;
    const auto guestFormat = (words[1] >> 20u) & 0x1ffu;
    const auto width = (((words[1] >> 30u) & 0x3u) | ((words[2] & 0xfffu) << 2u)) + 1u;
    const auto height = ((words[2] >> 14u) & 0x3fffu) + 1u;
    const auto baseLevel = (words[3] >> 12u) & 0xfu;
    const auto samples = 1u << ((words[3] >> 16u) & 0xfu);
    const auto type = (words[3] >> 28u) & 0xfu;
    const auto layers = ((words[4] >> 0u) & 0x1fffu) + 1u;
    const auto baseArray = (words[4] >> 16u) & 0x1fffu;
    char described[192];
    std::snprintf(described, sizeof(described), " (T# 0x%llx, %ux%u, %u samples, format %u, type %u)", static_cast<unsigned long long>(address), width, height, samples, guestFormat, type);
    Require(baseLevel == 0u, std::string("AGC graphics: a 2D MSAA T# with a nonzero base level (fragment) is not modeled") + described);
    // Type 15 (2D MSAA array) views of a target's single layer read the same texture2d_ms_array view
    // (MslAbi Version 9); targets have one layer, as upstream's.
    Require(type == 14u || (layers == 1u && baseArray == 0u), std::string("AGC graphics: a 2D MSAA array T# over more than its first layer is not implemented (multisampled color targets have one layer)") + described);
    const auto format = ResolveTextureFormat(guestFormat);
    std::lock_guard lock(targetsMutex());
    for (const auto& target : targets()) {
        const auto& color = target->target;
        if (target->context.device != context.device || color.address != address) continue;
        Require(color.extent.width == width && color.extent.height == height && color.samples == samples, std::string("AGC graphics: a 2D MSAA T# disagrees with the multisampled color target at its address in extent or sample count") + described);
        Require(color.format == format, std::string("AGC graphics: a 2D MSAA T# reads a multisampled color target in another format, which is not implemented") + described);
        target->ApplyFastClear(color);
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
        auto& texture = target->sampled[selects];
        if (texture == nullptr) {
            const VkComponentMapping components{swizzle(selects & 7u), swizzle((selects >> 3u) & 7u), swizzle((selects >> 6u) & 7u), swizzle((selects >> 9u) & 7u)};
            texture = std::make_shared<Texture>(target->context, target->image, color.format, VK_IMAGE_ASPECT_COLOR_BIT, components);
        }
        return texture;
    }
    throw std::runtime_error(std::string("AGC graphics: a 2D MSAA T# addresses memory that is no multisampled color target; guest-written MSAA memory (sample layout, FMASK) is not modeled") + described);
}

}

void NoteColorMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern) {
    std::lock_guard lock(targetsMutex());
    for (const auto& target : targets()) {
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
}

void RequireNoMultisampleTarget(std::uint64_t address, std::size_t bytes, const char* access) {
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
