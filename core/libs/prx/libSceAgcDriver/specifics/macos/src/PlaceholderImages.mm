// Work package D5: Metal twin of Graphics/src/PlaceholderImages.cpp; the Vulkan calls run on Metal through
// MetalVulkanImages.h (D-M2-3).
#include "MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <string>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

// MslAbi Version 6: storage heaps of the 2D MSAA (array) dimensions are texture2d_array textures of
// layers * samples slices (Metal shaders cannot write multisampled textures), so their placeholder
// is a single-sample 2D array (one layer of one sample: the default sample count constant is 1).
PlaceholderImageShape MetalShapeOf(std::uint32_t heap) {
    auto shape = PlaceholderImageShapeOf(heap);
    if (heap >= ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding && shape.samples != VK_SAMPLE_COUNT_1_BIT) {
        shape.samples = VK_SAMPLE_COUNT_1_BIT;
        shape.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    } else if (shape.samples != VK_SAMPLE_COUNT_1_BIT && heap < ShaderRecompiler::RuntimeAbi::FirstComparisonImageBinding) {
        // MslAbi Version 9: sampled heaps of both MSAA dimensions are texture2d_ms_array.
        shape.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    }
    return shape;
}

bool Supported(const Context& context, const PlaceholderImageShape& shape) {
    VkFormatProperties properties{};
    Metal::Vk::GetPhysicalDeviceFormatProperties(context.physical, shape.format, &properties);
    if ((properties.optimalTilingFeatures & shape.features) != shape.features) return false;
    VkImageFormatProperties limits{};
    if (Metal::Vk::GetPhysicalDeviceImageFormatProperties(context.physical, shape.format, shape.type, VK_IMAGE_TILING_OPTIMAL, shape.usage, 0, &limits) != VK_SUCCESS) return false;
    return (limits.sampleCounts & shape.samples) != 0;
}

}

PlaceholderImages::PlaceholderImages(const Context& context) : context(context) {
    try {
        std::vector<VkImageMemoryBarrier> toClear;
        for (std::uint32_t index = 0; index < images.size(); ++index) {
            const auto heap = ShaderRecompiler::RuntimeAbi::FirstImageBinding + index;
            // Upstream fd872264e2: no placeholder for a storage MSAA heap the device cannot back. Metal
            // emulates multisampled storage (D22/D28: storageImageMultisample is true), so none is omitted.
            if (!PlaceholderImageShapeForDevice(heap, context.storageImageMultisample)) continue;
            const auto shape = MetalShapeOf(heap);
            if (!Supported(context, shape)) continue;
            auto& image = images[index];
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = shape.type;
            if (shape.samples != VK_SAMPLE_COUNT_1_BIT && shape.viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY) info.flags = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
            info.format = shape.format;
            info.extent = {1, 1, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = shape.samples;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = shape.usage;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(Metal::Vk::CreateImage(context.device, &info, nullptr, &image.image), "vkCreateImage placeholder");
            VkMemoryRequirements requirements{};
            Metal::Vk::GetImageMemoryRequirements(context.device, image.image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = 0u;
            Check(Metal::Vk::AllocateMemory(context.device, &allocation, nullptr, &image.memory), "vkAllocateMemory placeholder");
            Check(Metal::Vk::BindImageMemory(context.device, image.image, image.memory, 0), "vkBindImageMemory placeholder");
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image.image;
            viewInfo.viewType = shape.viewType;
            viewInfo.format = shape.format;
            viewInfo.subresourceRange = {shape.aspect, 0, 1, 0, 1};
            Check(Metal::Vk::CreateImageView(context.device, &viewInfo, nullptr, &image.view), "vkCreateImageView placeholder");
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image.image;
            barrier.subresourceRange = viewInfo.subresourceRange;
            toClear.push_back(barrier);
        }
        if (toClear.empty()) return;
        CommandBatch batch(context);
        const auto commands = batch.Handle();
        const auto pipelineBarrier = Metal::Vk::CmdPipelineBarrier;
        pipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, static_cast<std::uint32_t>(toClear.size()), toClear.data());
        std::vector<VkImageMemoryBarrier> toGeneral;
        for (const auto& barrier : toClear) {
            if (barrier.subresourceRange.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT) {
                const VkClearDepthStencilValue zero{0.0f, 0};
                Metal::Vk::CmdClearDepthStencilImage(commands, barrier.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &barrier.subresourceRange);
            } else {
                const VkClearColorValue zero{};
                Metal::Vk::CmdClearColorImage(commands, barrier.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &barrier.subresourceRange);
            }
            auto general = barrier;
            general.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            toGeneral.push_back(general);
        }
        pipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, static_cast<std::uint32_t>(toGeneral.size()), toGeneral.data());
        batch.SubmitAndWait();
    } catch (...) {
        release();
        throw;
    }
}

PlaceholderImages::~PlaceholderImages() {
    release();
}

void PlaceholderImages::release() noexcept {
    for (auto& image : images) {
        if (image.view) Metal::Vk::DestroyImageView(context.device, image.view, nullptr);
        if (image.image) Metal::Vk::DestroyImage(context.device, image.image, nullptr);
        if (image.memory) Metal::Vk::FreeMemory(context.device, image.memory, nullptr);
        image = {};
    }
}

VkImageView PlaceholderImages::View(std::uint32_t heap) const {
    const auto index = heap - ShaderRecompiler::RuntimeAbi::FirstImageBinding;
    Require(heap >= ShaderRecompiler::RuntimeAbi::FirstImageBinding && index < images.size() && images[index].view != VK_NULL_HANDLE,
            "the device has no placeholder image for typed image heap " + std::to_string(heap));
    return images[index].view;
}

}
