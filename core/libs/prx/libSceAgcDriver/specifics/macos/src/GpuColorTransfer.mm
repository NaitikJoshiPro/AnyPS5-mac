// Work package D4: Metal twin of Graphics/src/GpuColorTransfer.cpp. Graphics/shaders/ColorTransfer.comp
// in MSL, recorded through the command stream's helper dispatch; `pipeline` carries the
// CF-retained compute pipeline state, the descriptor members stay unused (D-M2-3). Mode 32
// (upstream 76c0ba53e) converts BT.2100 PQ A2B10G10R10 texels to 8-bit sRGB for an SDR output with
// Graphics/shaders/PqToSdr.glsl's tone map, ported to MSL below.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Graphics/include/GpuColorTransfer.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PqToSdr.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HalfFloatScanout.hpp"
#include "MetalCommandStream.h"
#include "MetalVulkanImages.h"
#include <array>
#include <cmath>
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
    device const float* pqNits;
    float whiteNits;
};

// Graphics/shaders/PqToSdr.glsl (Execution/include/PqToSdr.hpp on the CPU).
static float pqToneMap(float luminance, float relativeWhite) {
    const float knee = 0.75f;
    if (luminance <= knee) return luminance;
    const float t = (luminance - knee) / (1.0f - knee);
    const float w = (relativeWhite - knee) / (1.0f - knee);
    return knee + (1.0f - knee) * t * (1.0f + t / (w * w)) / (1.0f + t);
}

static float3 pqToSdrLinear(float3 nits, float whiteNits) {
    const float referenceWhite = min(203.0f, whiteNits);
    const float3 bt2020 = nits / referenceWhite;
    float3 rgb = max(float3(
        1.6604910f * bt2020.r - 0.5876411f * bt2020.g - 0.0728499f * bt2020.b,
        -0.1245505f * bt2020.r + 1.1328999f * bt2020.g - 0.0083494f * bt2020.b,
        -0.0181508f * bt2020.r - 0.1005789f * bt2020.g + 1.1187297f * bt2020.b), float3(0.0f));
    const float luminance = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
    if (luminance > 0.0f) rgb = min(rgb * (pqToneMap(luminance, whiteNits / referenceWhite) / luminance), float3(1.0f));
    return rgb;
}

static float srgbEncode(float linear) {
    const float v = clamp(linear, 0.0f, 1.0f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * precise::pow(v, 1.0f / 2.4f) - 0.055f;
}

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
    if ((parameters.mode & 32u) != 0u) {
        device const float* pqNits = parameters.pqNits;
        const float3 nits = float3(pqNits[pixel & 0x3ffu], pqNits[(pixel >> 10u) & 0x3ffu], pqNits[(pixel >> 20u) & 0x3ffu]);
        const float3 rgb = pqToSdrLinear((parameters.mode & 2u) != 0u ? nits : nits.zyx, parameters.whiteNits);
        const uint3 codes = uint3(floor(float3(srgbEncode(rgb.r), srgbEncode(rgb.g), srgbEncode(rgb.b)) * 255.0f + 0.5f));
        parameters.linear[linearIndex] = codes.b | (codes.g << 8u) | (codes.r << 16u) | 0xff000000u;
        return;
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
    std::uint64_t pqNits;
    float whiteNits;
    std::uint32_t padding;
};
static_assert(sizeof(Parameters) == 56, "color transfer parameters must match the MSL struct");

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
    const auto& luminance = AgcDriver::PqEotfNits();
    pqNits = std::make_unique<Buffer>(context, sizeof(luminance), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memcpy(pqNits->Bytes().data(), luminance.data(), sizeof(luminance));
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

void GpuColorTransfer::convert(VkCommandBuffer commands, bool toTiled, bool swapRedBlue, bool tenBit, bool halfFloat, std::optional<float> pqWhiteNits) {
    Require(commands != VK_NULL_HANDLE && tiled && tiledDevice && linear, "color transfer is not prepared");
    Require(halfFloat == (elementBytes == 8u), "color transfer source does not match the prepared texel size");
    Require(!toTiled || !halfFloat, "color transfer cannot write 16-bit float texels back to the guest");
    // The guest bytes move to video memory with one DMA copy; the shader's scattered accesses stay local.
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    CopyBuffer(context, commands, tiled->Handle(), 0, tiledDevice->Handle(), 0, tiledDevice->Size());
    dispatch(commands, toTiled, swapRedBlue, tenBit, mode == ColorTileMode::RenderTarget, halfFloat, pqWhiteNits);
    if (toTiled) {
        CopyBuffer(context, commands, tiledDevice->Handle(), 0, tiled->Handle(), 0, tiledDevice->Size());
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    }
}

void GpuColorTransfer::dispatch(VkCommandBuffer commands, bool toTiled, bool swapRedBlue, bool tenBit, bool tiledSource, bool halfFloat, std::optional<float> pqWhiteNits) {
    Require(!pqWhiteNits || (!toTiled && !tenBit && !halfFloat && elementBytes == 4u && std::isfinite(*pqWhiteNits) && *pqWhiteNits >= 1.0f), "color transfer converts PQ texels of 4 bytes to an 8-bit linear output with a white level of at least 1 nit");
    auto& stream = Metal::Stream(commands);
    stream.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    Parameters parameters{width, height, (width + 127u) / 128u, (toTiled ? 1u : 0u) | (swapRedBlue ? 2u : 0u) | (tiledSource ? 4u : 0u) | (tenBit ? 8u : 0u) | (halfFloat ? 16u : 0u) | (pqWhiteNits ? 32u : 0u),
                          Metal::BufferOf(tiledDevice->Handle()).gpuAddress, Metal::BufferOf(linear->Handle()).gpuAddress, Metal::BufferOf(thresholds->Handle()).gpuAddress,
                          Metal::BufferOf(pqNits->Handle()).gpuAddress, pqWhiteNits.value_or(0.0f), 0u};
    stream.DispatchHelper(PipelineOf(pipeline), std::as_bytes(std::span(&parameters, 1)), MTLSizeMake((width + 7u) / 8u, (height + 7u) / 8u, 1), MTLSizeMake(8, 8, 1));
    stream.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

void GpuColorTransfer::DetileImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, std::uint32_t newWidth, std::uint32_t newHeight, ColorTileMode newMode, bool swapRedBlue, bool tenBit, bool halfFloat, std::uint32_t layer) {
    Require(commands != VK_NULL_HANDLE && image != VK_NULL_HANDLE, "color transfer image source is unavailable");
    prepare(newWidth, newHeight, newMode, halfFloat ? 8u : 4u);
    Require(tiledDevice->Size() >= static_cast<VkDeviceSize>(width) * height * elementBytes, "color transfer buffer is smaller than the image");
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
    copy.imageExtent = {width, height, 1};
    Metal::Vk::CmdCopyImageToBuffer(commands, image, layout, tiledDevice->Handle(), 1, &copy);
    dispatch(commands, false, swapRedBlue, tenBit, false, halfFloat, std::nullopt);
}

void GpuColorTransfer::Detile(VkCommandBuffer commands, bool swapRedBlue, bool tenBit, bool halfFloat, std::optional<float> pqWhiteNits) {
    convert(commands, false, swapRedBlue, tenBit, halfFloat, pqWhiteNits);
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
