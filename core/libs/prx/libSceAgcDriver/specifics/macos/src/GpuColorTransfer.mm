// Work package D4: Metal twin of Graphics/src/GpuColorTransfer.cpp. Graphics/shaders/ColorTransfer.comp
// in MSL, recorded through the command stream's helper dispatch; `pipeline` carries the
// CF-retained compute pipeline state, the descriptor members stay unused (D-M2-3).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Graphics/include/GpuColorTransfer.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HalfFloatScanout.hpp"
#include "MetalCommandStream.h"
#include "MetalVulkanImages.h"
#include <array>
#include <cstring>

namespace AgcDriver::Graphics {

namespace {

constexpr const char* ColorTransferSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    uint width;
    uint height;
    uint blocksPerRow;
    uint mode;
    device uint* tiled;
    device uint* linear;
    device const uint* thresholds;
};

static uint scanoutCode(device const uint* thresholds, uint bits, uint base) {
    if (bits > 0x7c00u) return 0u;
    uint code = 0u;
    for (uint step = 128u; step != 0u; step >>= 1u) {
        if (code + step <= 255u && thresholds[base + code + step - 1u] <= bits) code += step;
    }
    return code;
}

kernel void color_transfer(constant Parameters& parameters [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
    const uint x = id.x;
    const uint y = id.y;
    if (x >= parameters.width || y >= parameters.height) return;
    const uint linearIndex = y * parameters.width + x;
    uint tiledIndex = linearIndex;
    if ((parameters.mode & 4u) != 0u) {
        const uint offset = ((y << 4u) & 0x0070u) ^ ((y << 5u) & 0x0f00u) ^ ((y << 9u) & 0x1000u) ^ ((y << 8u) & 0x4000u)
            ^ ((x << 2u) & 0x000cu) ^ ((x << 5u) & 0x0380u) ^ ((x << 4u) & 0x0400u) ^ ((x << 6u) & 0x0800u) ^ ((x << 9u) & 0xa000u);
        tiledIndex = ((y / 128u) * parameters.blocksPerRow + x / 128u) * 16384u + offset / 4u;
    }
    device uint* tiled = parameters.tiled;
    uint pixel;
    if ((parameters.mode & 16u) != 0u) {
        uint texel = linearIndex * 2u;
        if ((parameters.mode & 4u) != 0u) {
            const uint offset = ((x << 3u) & 0x0008u) ^ ((x << 4u) & 0x0460u) ^ ((x << 5u) & 0x0300u) ^ ((x << 6u) & 0x0800u) ^ ((x << 9u) & 0x8000u) ^ ((x << 10u) & 0x2000u)
                ^ ((y << 4u) & 0x0010u) ^ ((y << 6u) & 0x0080u) ^ ((y << 5u) & 0x0f00u) ^ ((y << 10u) & 0x5000u);
            texel = (((y / 64u) * parameters.blocksPerRow + x / 128u) * 65536u + offset) / 4u;
        }
        const uint first = tiled[texel];
        const uint second = tiled[texel + 1u];
        device const uint* thresholds = parameters.thresholds;
        pixel = scanoutCode(thresholds, first & 0xffffu, 0u) | (scanoutCode(thresholds, first >> 16u, 0u) << 8u) | (scanoutCode(thresholds, second & 0xffffu, 0u) << 16u) | (scanoutCode(thresholds, second >> 16u, 255u) << 24u);
    } else {
        pixel = (parameters.mode & 1u) != 0u ? parameters.linear[linearIndex] : tiled[tiledIndex];
    }
    if ((parameters.mode & 8u) != 0u) pixel = ((pixel >> 2u) & 0xffu) | (((pixel >> 12u) & 0xffu) << 8u) | (((pixel >> 22u) & 0xffu) << 16u) | 0xff000000u;
    if ((parameters.mode & 2u) != 0u) pixel = (pixel & 0xff00ff00u) | ((pixel & 0xffu) << 16u) | ((pixel >> 16u) & 0xffu);
    if ((parameters.mode & 1u) != 0u) tiled[tiledIndex] = pixel;
    else parameters.linear[linearIndex] = pixel;
}
)MSL";

struct Parameters {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t blocksPerRow;
    std::uint32_t mode;
    std::uint64_t tiled;
    std::uint64_t linear;
    std::uint64_t thresholds;
};

id<MTLComputePipelineState> PipelineOf(VkPipeline pipeline) {
    return (__bridge id<MTLComputePipelineState>)reinterpret_cast<void*>(pipeline);
}

}

GpuColorTransfer::GpuColorTransfer(const Context& context) : context(context) {
    id<MTLDevice> device = Metal::DeviceOf(context);
    Require(device != nil, "the Metal color transfer needs a Metal device context");
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_1;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(ColorTransferSource) options:options error:&error];
    Require(library != nil, "the color transfer kernel does not compile: " + Metal::Describe(error));
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"color_transfer"] error:&error];
    Require(state != nil, "cannot build the color transfer pipeline: " + Metal::Describe(error));
    pipeline = reinterpret_cast<VkPipeline>(const_cast<void*>(CFBridgingRetain(state)));
    const auto& table = HalfFloatScanoutThresholds();
    thresholds = std::make_unique<Buffer>(context, sizeof(table), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memcpy(thresholds->Bytes().data(), table.data(), sizeof(table));
}

GpuColorTransfer::~GpuColorTransfer() {
    release();
}

void GpuColorTransfer::release() noexcept {
    if (pipeline != VK_NULL_HANDLE) CFRelease(reinterpret_cast<CFTypeRef>(pipeline));
    pipeline = VK_NULL_HANDLE;
}

void GpuColorTransfer::prepare(std::uint32_t newWidth, std::uint32_t newHeight, ColorTileMode newMode, std::uint32_t newElementBytes) {
    Require(newElementBytes == 4u || (newElementBytes == 8u && newMode == ColorTileMode::RenderTarget), "color transfer supports 8-byte texels only in the render target layout");
    const ColorTargetLayout layout(newWidth, newHeight, newMode, newElementBytes);
    const auto linearBytes = static_cast<std::size_t>(newWidth) * newHeight * 4u;
    Require(layout.Bytes() <= context.limits.maxStorageBufferRange && linearBytes <= context.limits.maxStorageBufferRange, "color transfer exceeds storage buffer limits");
    Require((newWidth + 7u) / 8u <= context.limits.maxComputeWorkGroupCount[0] && (newHeight + 7u) / 8u <= context.limits.maxComputeWorkGroupCount[1], "color transfer exceeds workgroup limits");
    if (tiled && width == newWidth && height == newHeight && mode == newMode && elementBytes == newElementBytes) return;
    constexpr VkBufferUsageFlags copies = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto newTiled = std::make_unique<Buffer>(context, layout.Bytes(), copies);
    auto newTiledDevice = std::make_unique<DeviceBuffer>(context, layout.Bytes(), copies | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto newLinear = std::make_unique<DeviceBuffer>(context, linearBytes, copies | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    tiled = std::move(newTiled);
    tiledDevice = std::move(newTiledDevice);
    linear = std::move(newLinear);
    width = newWidth;
    height = newHeight;
    mode = newMode;
    elementBytes = newElementBytes;
}

void GpuColorTransfer::Upload(std::uint64_t address, std::uint32_t newWidth, std::uint32_t newHeight, ColorTileMode newMode, std::uint32_t newElementBytes) {
    prepare(newWidth, newHeight, newMode, newElementBytes);
    const ColorTargetLayout layout(width, height, mode, elementBytes);
    GuestMemory::Read(address, tiled->Bytes(), layout.Alignment());
}

void GpuColorTransfer::convert(VkCommandBuffer commands, bool toTiled, bool swapRedBlue, bool tenBit, bool halfFloat) {
    Require(commands != VK_NULL_HANDLE && tiled && tiledDevice && linear, "color transfer is not prepared");
    Require(halfFloat == (elementBytes == 8u), "color transfer source does not match the prepared texel size");
    Require(!toTiled || !halfFloat, "color transfer cannot write 16-bit float texels back to the guest");
    // The guest bytes move to video memory with one DMA copy; the shader's scattered accesses stay local.
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    CopyBuffer(context, commands, tiled->Handle(), 0, tiledDevice->Handle(), 0, tiledDevice->Size());
    dispatch(commands, toTiled, swapRedBlue, tenBit, mode == ColorTileMode::RenderTarget, halfFloat);
    if (toTiled) {
        CopyBuffer(context, commands, tiledDevice->Handle(), 0, tiled->Handle(), 0, tiledDevice->Size());
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    }
}

void GpuColorTransfer::dispatch(VkCommandBuffer commands, bool toTiled, bool swapRedBlue, bool tenBit, bool tiledSource, bool halfFloat) {
    auto& stream = Metal::Stream(commands);
    stream.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    Parameters parameters{width, height, (width + 127u) / 128u, (toTiled ? 1u : 0u) | (swapRedBlue ? 2u : 0u) | (tiledSource ? 4u : 0u) | (tenBit ? 8u : 0u) | (halfFloat ? 16u : 0u),
                          Metal::BufferOf(tiledDevice->Handle()).gpuAddress, Metal::BufferOf(linear->Handle()).gpuAddress, Metal::BufferOf(thresholds->Handle()).gpuAddress};
    stream.DispatchHelper(PipelineOf(pipeline), std::as_bytes(std::span(&parameters, 1)), MTLSizeMake((width + 7u) / 8u, (height + 7u) / 8u, 1), MTLSizeMake(8, 8, 1));
    stream.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

void GpuColorTransfer::DetileImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, std::uint32_t newWidth, std::uint32_t newHeight, ColorTileMode newMode, bool swapRedBlue, bool tenBit, bool halfFloat) {
    Require(commands != VK_NULL_HANDLE && image != VK_NULL_HANDLE, "color transfer image source is unavailable");
    prepare(newWidth, newHeight, newMode, halfFloat ? 8u : 4u);
    Require(tiledDevice->Size() >= static_cast<VkDeviceSize>(width) * height * elementBytes, "color transfer buffer is smaller than the image");
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    Metal::Vk::CmdCopyImageToBuffer(commands, image, layout, tiledDevice->Handle(), 1, &copy);
    dispatch(commands, false, swapRedBlue, tenBit, false, halfFloat);
}

void GpuColorTransfer::Detile(VkCommandBuffer commands, bool swapRedBlue, bool tenBit, bool halfFloat) {
    convert(commands, false, swapRedBlue, tenBit, halfFloat);
}

void GpuColorTransfer::Tile(VkCommandBuffer commands) {
    convert(commands, true, false);
}

void GpuColorTransfer::WriteBack(std::uint64_t address) {
    Require(tiled != nullptr, "color transfer is not prepared for writeback");
    Require(elementBytes == 4u, "color transfer cannot write 16-bit float texels back to the guest");
    const ColorTargetLayout layout(width, height, mode, elementBytes);
    GuestMemory::Write(address, tiled->Bytes(), layout.Alignment());
}

VkBuffer GpuColorTransfer::LinearBuffer() const {
    Require(linear != nullptr, "color transfer is not prepared");
    return linear->Handle();
}

RenderTarget& GpuColorTransfer::Target(const ColorTarget& color, bool blending) {
    if (!target || targetExtent.width != color.extent.width || targetExtent.height != color.extent.height || targetFormat != color.format || targetBlending != blending) {
        auto replacement = std::make_unique<RenderTarget>(context, color, blending);
        target = std::move(replacement);
        targetExtent = color.extent;
        targetFormat = color.format;
        targetBlending = blending;
    }
    return *target;
}

}
