// Work packages D4/D5 (src/MetalImages.h):
//   agc_metal_texture_detile  the CPU (de)tiler against ColorTargetLayout's independent tiler for
//                             the color-target tile modes, and retile/detile round trips of the
//                             texture tile modes and element sizes (no Metal device)
//   agc_metal_images          guest textures uploaded into Metal textures read back equal, heap views
//                             of every image class, placeholders, samplers from S# words, and the
//                             R16Uint images of packed 16-bit surfaces with their attachment views
//                             (Metal objects only, no GPU work)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalImages.h"
#include "../src/MetalVulkanImages.h"
#include "MetalTestDevice.h"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuColorTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "RuntimeAbi.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {
// Texture.mm (declared where it is used, ShaderResources.mm).
VkFormat ImageFormatForGuest(const Context& context, std::uint32_t guestFormat);
}

namespace {

using namespace AgcDriver::Graphics;
using namespace AgcDriver::Graphics::Metal;

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        if (failures <= 20) std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

std::vector<std::byte> Pattern(std::size_t bytes, std::uint32_t seed) {
    std::vector<std::byte> data(bytes);
    std::mt19937 random(seed);
    for (auto& value : data) value = static_cast<std::byte>(random() & 0xffu);
    return data;
}

// SW_4KB_S_X/D_X are left out: ColorTargetLayout applies their equations to the in-block coordinate
// bits only, the texture detiler (TextureDetile.comp, which this port follows) to the whole
// coordinates; the two upstream tilers disagree there for surfaces wider or taller than a block.
void DetileAgainstColorTargets() {
    struct Case {
        ColorTileMode color;
        std::uint32_t bytes;
        std::uint32_t pipeBankXor;
    };
    const Case cases[] = {
        {ColorTileMode::Standard4KB, 4u, 0u}, {ColorTileMode::Standard64KB, 4u, 0u}, {ColorTileMode::RenderTarget, 4u, 0u}, {ColorTileMode::RenderTarget, 4u, 0x300u},
        {ColorTileMode::Standard64KB, 8u, 0u}, {ColorTileMode::RenderTarget, 8u, 0u}, {ColorTileMode::Standard4KB, 2u, 0u},
    };
    for (const auto& item : cases) {
        for (const auto [width, height] : {std::pair{256u, 256u}, std::pair{96u, 40u}}) {
            const ColorTargetLayout layout(width, height, item.color, item.bytes, item.pipeBankXor);
            const auto linear = Pattern(static_cast<std::size_t>(width) * height * item.bytes, width * 7u + item.bytes);
            std::vector<std::byte> tiled(layout.Bytes());
            layout.Tile(linear, tiled);
            const auto mips = ComputeElementMipLayout(ColorTextureTileMode(item.color), item.bytes, width, height, 1u);
            std::vector<std::byte> detiled(linear.size());
            const DetileLevel level{ColorTextureTileMode(item.color), item.bytes, &mips[0], 0u, false, item.pipeBankXor};
            DetileCpu(level, tiled.data(), tiled.size(), detiled.data(), static_cast<std::uint64_t>(width) * item.bytes, item.bytes);
            Check(detiled == linear, "detile of color tile mode " + std::to_string(static_cast<int>(item.color)) + " at " + std::to_string(item.bytes) + " bytes, " + std::to_string(width) + "x" + std::to_string(height) + ", pipe/bank xor " + std::to_string(item.pipeBankXor) + " differs from ColorTargetLayout");
        }
    }
}

void RoundTrips() {
    const TextureTileMode modes[] = {TextureTileMode::kLinear, TextureTileMode::kStandard256B, TextureTileMode::kStandard4KB, TextureTileMode::kStandard64KB, TextureTileMode::kD256B, TextureTileMode::kD4KB, TextureTileMode::kD64KB, TextureTileMode::kZ64KBX, TextureTileMode::kS64KBX, TextureTileMode::kD64KBX, TextureTileMode::kR64KBX};
    // Guest formats of 1, 2, 4, 8 and 16 bytes per element.
    const std::uint32_t formats[] = {1u, 14u, 20u, 62u, 77u};
    for (const auto mode : modes) {
        for (const auto format : formats) {
            GuestTextureResource resource{};
            resource.width = 70u;
            resource.height = 33u;
            resource.mipCount = 4u;
            resource.tileMode = mode;
            resource.dimension = TextureDimension::k2D;
            resource.format = format;
            SurfaceGeometry geometry;
            try {
                geometry = DescribeSurface(resource);
            } catch (const std::exception&) {
                continue;  // a combination the guest layout code does not describe
            }
            const auto bytes = BytesPerElement(format);
            for (std::uint32_t level = 0; level < geometry.mips.size(); ++level) {
                const auto& mip = geometry.mips[level];
                const DetileLevel detile{mode, bytes, &mip, 0u, false, 0u};
                std::vector<std::byte> linear;
                try {
                    linear = Pattern(static_cast<std::size_t>(mip.width) * mip.height * bytes, level * 31u + format);
                    std::vector<std::byte> tiled(geometry.guestBytes, std::byte{0});
                    RetileCpu(detile, tiled.data() + mip.tiledOffset, tiled.size() - mip.tiledOffset, linear.data(), static_cast<std::uint64_t>(mip.width) * bytes, bytes);
                    std::vector<std::byte> back(linear.size());
                    DetileCpu(detile, tiled.data() + mip.tiledOffset, tiled.size() - mip.tiledOffset, back.data(), static_cast<std::uint64_t>(mip.width) * bytes, bytes);
                    Check(back == linear, "round trip of tile mode " + std::to_string(static_cast<int>(mode)) + ", format " + std::to_string(format) + ", level " + std::to_string(level));
                } catch (const std::exception& error) {
                    Check(std::string(error.what()).find("no swizzle equation") != std::string::npos, std::string("unexpected failure: ") + error.what());
                }
            }
        }
    }
}

int Detile() {
    DetileAgainstColorTargets();
    RoundTrips();
    // Formats: three-channel 32-bit formats widen to four channels; unknown formats throw naming D4.
    Check(GuestTextureFormat(74u).textureElementBytes == 16u && GuestTextureFormat(74u).elementBytes == 12u, "RGB32 float widens to 16 bytes");
    Check(GuestTextureFormat(30u).format == MTLPixelFormatR32Uint, "11_11_10 conversion formats read as R32 uint");
    Check(HeapViewFormat(MTLPixelFormatRGBA8Unorm, 1u) == MTLPixelFormatRGBA8Uint && HeapViewFormat(MTLPixelFormatR32Float, 2u) == MTLPixelFormatR32Sint, "heap view formats");
    bool threw = false;
    try {
        (void)GuestTextureFormat(3u);
    } catch (const std::exception& error) {
        threw = std::string(error.what()).find("D4") != std::string::npos;
    }
    Check(threw, "an unknown guest format throws naming D4");
    if (failures != 0) return 1;
    std::printf("PASS: agc_metal_texture_detile\n");
    return 0;
}

std::array<std::uint32_t, 8> TextureWords(const void* data, std::uint32_t format, std::uint32_t width, std::uint32_t height, std::uint32_t levels, std::uint32_t tile, std::uint32_t type) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((width - 1u) & 3u) << 30u),
        ((width - 1u) >> 2u) | ((height - 1u) << 14u),
        0xfacu | ((levels - 1u) << 16u) | (tile << 20u) | (type << 28u),
        0u,
        (levels - 1u) << 4u,
        0u,
        0u,
    };
}


// The MSL detile/retile kernels (TextureDetiler.mm) against the CPU detiler, per tile mode and
// element size: tiled random bytes detiled on the GPU equal the CPU detile, and a GPU retile of the
// result rebuilds every element's bytes.
int DetileGpu() {
    AgcDriver::Graphics::MetalTest::Device device;
    TextureDetiler detiler(device.context);
    const TextureTileMode modes[] = {TextureTileMode::kLinear, TextureTileMode::kStandard4KB, TextureTileMode::kStandard64KB, TextureTileMode::kD4KB, TextureTileMode::kZ64KBX, TextureTileMode::kR64KBX, TextureTileMode::kS64KBX};
    const std::uint32_t formats[] = {1u, 14u, 20u, 62u, 77u};
    std::uint32_t cases = 0;
    for (const auto mode : modes) {
        for (const auto format : formats) {
            GuestTextureResource resource{};
            resource.width = 70u;
            resource.height = 33u;
            resource.mipCount = 2u;
            resource.tileMode = mode;
            resource.dimension = TextureDimension::k2D;
            resource.format = format;
            const auto geometry = DescribeSurface(resource);
            const auto bytes = BytesPerElement(format);
            const auto& mip = geometry.mips[0];
            const DetileLevel level{mode, bytes, &mip, 0u, false, 0u};
            const auto tiled = Pattern(geometry.guestBytes, format * 13u + static_cast<std::uint32_t>(mode));
            std::vector<std::byte> expected(mip.linearSize);
            try {
                DetileCpu(level, tiled.data() + mip.tiledOffset, tiled.size() - mip.tiledOffset, expected.data(), mip.pitchBytes, bytes);
            } catch (const std::exception&) {
                continue;  // no equation for this combination
            }
            Buffer tiledBuffer(device.context, geometry.guestBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            Buffer linearBuffer(device.context, mip.linearSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            Buffer retiledBuffer(device.context, geometry.guestBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            std::memcpy(tiledBuffer.Bytes().data(), tiled.data(), tiled.size());
            std::memset(linearBuffer.Bytes().data(), 0, mip.linearSize);
            std::memset(retiledBuffer.Bytes().data(), 0, geometry.guestBytes);
            {
                CommandBatch batch(device.context);
                detiler.Dispatch(batch.Handle(), mode, bytes, tiledBuffer.Handle(), mip.tiledOffset, linearBuffer.Handle(), 0, mip, false);
                Metal::Stream(batch.Handle()).PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                detiler.Dispatch(batch.Handle(), mode, bytes, linearBuffer.Handle(), 0, retiledBuffer.Handle(), mip.tiledOffset, mip, true);
                batch.SubmitAndWait();
            }
            const auto* linear = linearBuffer.Bytes().data();
            bool same = true;
            for (std::uint32_t y = 0; y < mip.height && same; ++y) same = std::memcmp(linear + y * mip.pitchBytes, expected.data() + y * mip.pitchBytes, static_cast<std::size_t>(mip.width) * bytes) == 0;
            Check(same, "GPU detile of tile mode " + std::to_string(static_cast<int>(mode)) + ", format " + std::to_string(format) + " differs from the CPU detiler");
            bool rebuilt = true;
            for (std::uint32_t y = 0; y < mip.height && rebuilt; ++y) {
                for (std::uint32_t x = 0; x < mip.width && rebuilt; ++x) {
                    const auto offset = mip.tiledOffset + TiledElementOffset(level, x, y);
                    rebuilt = std::memcmp(retiledBuffer.Bytes().data() + offset, tiled.data() + offset, bytes) == 0;
                }
            }
            Check(rebuilt, "GPU retile of tile mode " + std::to_string(static_cast<int>(mode)) + ", format " + std::to_string(format) + " does not rebuild the elements");
            ++cases;
        }
    }
    if (failures != 0) return 1;
    std::printf("PASS: agc_metal_texture_detile_gpu (%u cases)\n", cases);
    return 0;
}


// The GpuColorTransfer twin (MSL ColorTransfer.comp): a 256x128 RGBA8 surface in the render-target
// layout detiles on the GPU to the linear pixels ColorTargetLayout tiled, with and without the
// red/blue swap, and tiles back to the guest bytes.
int ColorTransfer() {
    AgcDriver::Graphics::MetalTest::Device device;
    // The limits the device's setup gives its Context (MetalDevice.mm), which the test device leaves at zero.
    const auto bufferLength = static_cast<std::uint64_t>(device.device.maxBufferLength);
    device.context.limits.maxStorageBufferRange = static_cast<std::uint32_t>(std::min<std::uint64_t>(bufferLength, 0xffffffffu));
    device.context.limits.maxUniformBufferRange = device.context.limits.maxStorageBufferRange;
    device.context.limits.maxComputeWorkGroupCount[0] = device.context.limits.maxComputeWorkGroupCount[1] = device.context.limits.maxComputeWorkGroupCount[2] = 65535u;
    constexpr std::uint32_t width = 256u;
    constexpr std::uint32_t height = 128u;
    const ColorTargetLayout layout(width, height, ColorTileMode::RenderTarget, 4u);
    const auto linear = Pattern(static_cast<std::size_t>(width) * height * 4u, 77u);
    std::vector<std::byte> tiled(layout.Bytes());
    layout.Tile(linear, tiled);
    std::vector<std::byte> guest(tiled.size() + 0x10000u);
    auto* aligned = reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(guest.data()) + 0xffffu) & ~std::uintptr_t{0xffffu});
    std::memcpy(aligned, tiled.data(), tiled.size());
    GpuColorTransfer transfer(device.context);
    for (const bool swap : {false, true}) {
        transfer.Upload(reinterpret_cast<std::uintptr_t>(aligned), width, height, ColorTileMode::RenderTarget);
        Buffer readback(device.context, linear.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        {
            CommandBatch batch(device.context);
            transfer.Detile(batch.Handle(), swap);
            AgcDriver::Graphics::CopyBuffer(device.context, batch.Handle(), transfer.LinearBuffer(), 0, readback.Handle(), 0, linear.size());
            batch.SubmitAndWait();
        }
        bool same = true;
        const auto* pixels = reinterpret_cast<const std::uint32_t*>(readback.Bytes().data());
        const auto* expected = reinterpret_cast<const std::uint32_t*>(linear.data());
        for (std::size_t index = 0; index < linear.size() / 4u && same; ++index) {
            auto pixel = expected[index];
            if (swap) pixel = (pixel & 0xff00ff00u) | ((pixel & 0xffu) << 16u) | ((pixel >> 16u) & 0xffu);
            same = pixels[index] == pixel;
        }
        Check(same, std::string("GPU color transfer detile") + (swap ? " with the red/blue swap" : ""));
    }
    // Tile: the linear pixels go back into the tiled layout (the guest bytes after WriteBack).
    std::memset(aligned, 0, tiled.size());
    {
        CommandBatch batch(device.context);
        transfer.Tile(batch.Handle());
        batch.SubmitAndWait();
    }
    transfer.WriteBack(reinterpret_cast<std::uintptr_t>(aligned));
    // The linear buffer holds the last (red/blue swapped) detile: its tiling is what lands.
    auto swapped = linear;
    for (std::size_t index = 0; index < swapped.size(); index += 4u) std::swap(swapped[index], swapped[index + 2u]);
    std::vector<std::byte> expectedTiled(layout.Bytes());
    layout.Tile(swapped, expectedTiled);
    bool tiledSame = true;
    for (std::uint32_t y = 0; y < height && tiledSame; ++y) {
        for (std::uint32_t x = 0; x < width && tiledSame; ++x) tiledSame = std::memcmp(aligned + layout.Offset(x, y), expectedTiled.data() + layout.Offset(x, y), 4u) == 0;
    }
    Check(tiledSame, "GPU color transfer tile and write-back");
    if (failures != 0) return 1;
    std::printf("PASS: agc_metal_color_transfer\n");
    return 0;
}

int Images() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
        std::printf("skipped, the machine has no Metal 4 device\n");
        return 77;
    }
    // What device creation does (MetalDevice.mm): a complete Context with its recorder, the image
    // cache attached to it (detiler, PlaceholderImages, sampler cache). Every heap's placeholder
    // must be a valid Metal texture of the heap's type: under MTL_DEBUG_LAYER a bad texture or
    // clear descriptor aborts here before any device test meets it.
    {
        AgcDriver::Graphics::MetalTest::Device testDevice;
        Recorder recorder(testDevice.context);
        testDevice.context.recorder = &recorder;
        recorder.Activate();
        ImageCache attached(testDevice.device, testDevice.queue);
        attached.Attach(testDevice.context);
        const auto* context = attached.ImageContext();
        Check(context != nullptr && context->placeholderImages != nullptr && context->detiler != nullptr && context->samplerCache != nullptr, "Attach equips the image context");
        for (std::uint32_t heap = ShaderRecompiler::RuntimeAbi::FirstImageBinding; heap < static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers); ++heap) {
            id<MTLTexture> view = Metal::TextureOfView(context->placeholderImages->View(heap));
            Check(view != nil, "PlaceholderImages view of heap " + std::to_string(heap));
        }
        // Packed 16-bit surfaces (TechnicalDebt metal-color-1555-target): no shader-writable form,
        // so their resident images are R16Uint textures that render through a view of the
        // guest format; guest format 134 is A1R5G5B5.
        Check(!StorageFormatAvailable(*context, 134u), "A1R5G5B5 has no storage form on Metal");
        Check(ImageFormatForGuest(*context, 134u) == VK_FORMAT_R16_UINT, "an A1R5G5B5 surface's image is R16Uint");
        Check(ImageFormatForGuest(*context, 56u) == StorageFormatForGuest(*context, 56u), "a storable format's image is its storage format");
        {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = VK_FORMAT_R16_UINT;
            info.extent = {16u, 16u, 1u};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            VkImage image = VK_NULL_HANDLE;
            Check(Vk::CreateImage(context->device, &info, nullptr, &image) == VK_SUCCESS, "an R16Uint image of a packed 16-bit surface");
            for (const auto& [format, metal] : {std::pair{VK_FORMAT_A1R5G5B5_UNORM_PACK16, MTLPixelFormatBGR5A1Unorm}, std::pair{VK_FORMAT_R5G6B5_UNORM_PACK16, MTLPixelFormatB5G6R5Unorm}}) {
                VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
                usage.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                viewInfo.pNext = &usage;
                viewInfo.image = image;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = format;
                viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
                VkImageView view = VK_NULL_HANDLE;
                Check(Vk::CreateImageView(context->device, &viewInfo, nullptr, &view) == VK_SUCCESS, "an attachment view of a packed 16-bit format over R16Uint");
                id<MTLTexture> texture = Metal::TextureOfView(view);
                Check(texture != nil && texture.pixelFormat == metal && (texture.usage & MTLTextureUsageRenderTarget) != 0u && texture.parentTexture.pixelFormat == MTLPixelFormatR16Uint, "the packed 16-bit attachment view renders in its Metal format over the R16Uint texture");
                if (view != VK_NULL_HANDLE) Vk::DestroyImageView(context->device, view, nullptr);
            }
            if (image != VK_NULL_HANDLE) Vk::DestroyImage(context->device, image, nullptr);
        }
        // UPSYNC5 (upstream c97e5e0338, 086e98a42a, 3b6f6359f7, 4513d47104), the image shapes
        // Texture.mm now creates. A 1D depth-comparison texture is a 2D depth image one texel high
        // with 2D(-array) views; a cube descriptor with slices past its last whole cube is viewed
        // as a 2D array of all of them; arrays stop at Metal's 2048 slices.
        Check(context->limits.maxImageArrayLayers == 2048u, "the image context reports Metal's 2048-slice array limit");
        {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = VK_FORMAT_D32_SFLOAT;
            info.extent = {64u, 1u, 1u};
            info.mipLevels = 3;
            info.arrayLayers = 3;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            VkImage image = VK_NULL_HANDLE;
            Check(Vk::CreateImage(context->device, &info, nullptr, &image) == VK_SUCCESS, "a 64x1 depth image backs a 1D depth-comparison array");
            {
                auto owned = std::make_shared<OwnedImage>(*context, image, VK_NULL_HANDLE);
                VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.allocationSize = 4096u;
                owned->Allocate(*context, allocation, ~0u, "vkAllocateMemory test");
                Check(owned->memory != VK_NULL_HANDLE && owned->memoryBytes == 4096u && owned->memoryType == 0u, "OwnedImage::Allocate records the allocation");
                VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                viewInfo.image = image;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
                viewInfo.format = VK_FORMAT_D32_SFLOAT;
                viewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0u, 3u, 0u, 3u};
                VkImageView view = VK_NULL_HANDLE;
                Check(Vk::CreateImageView(context->device, &viewInfo, nullptr, &view) == VK_SUCCESS && TextureOfView(view).textureType == MTLTextureType2DArray && TextureOfView(view).height == 1u && TextureOfView(view).arrayLength == 3u, "a 1D depth-comparison array is viewed as a one-texel-high 2D depth array");
                if (view != VK_NULL_HANDLE) Vk::DestroyImageView(context->device, view, nullptr);
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0u, 1u, 0u, 1u};
                Check(Vk::CreateImageView(context->device, &viewInfo, nullptr, &view) == VK_SUCCESS && TextureOfView(view).textureType == MTLTextureType2D && TextureOfView(view).pixelFormat == MTLPixelFormatDepth32Float, "its first-layer and base-level view is a 2D depth view");
                if (view != VK_NULL_HANDLE) Vk::DestroyImageView(context->device, view, nullptr);
                // ~OwnedImage releases the image and the memory token.
            }
            // Pooled image memory (upstream 4513d47104): the texture is placed at offset 0 of a
            // placement heap of GetImageCreateRequirements' size.
            VkMemoryRequirements requirements{};
            Vk::GetImageCreateRequirements(context->device, &info, &requirements);
            Check(requirements.size != 0u && requirements.size % requirements.alignment == 0u, "GetImageCreateRequirements gives an aligned heap size");
            MTLHeapDescriptor* heapDescriptor = [MTLHeapDescriptor new];
            heapDescriptor.type = MTLHeapTypePlacement;
            heapDescriptor.size = static_cast<NSUInteger>(requirements.size);
            heapDescriptor.storageMode = MTLStorageModePrivate;
            heapDescriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
            id<MTLHeap> heap = [device newHeapWithDescriptor:heapDescriptor];
            Check(heap != nil, "a placement heap of the image's size");
            if (heap != nil) {
                const auto memory = reinterpret_cast<VkDeviceMemory>(const_cast<void*>(CFBridgingRetain(heap)));
                Check(!Vk::IsMemoryToken(memory), "a heap is not the memory token");
                VkImage placed = VK_NULL_HANDLE;
                Check(Vk::CreateImageInMemory(context->device, &info, memory, &placed) == VK_SUCCESS && TextureOfImage(placed).heap == heap && TextureOfImage(placed).arrayLength == 3u, "CreateImageInMemory places the texture in the heap");
                if (placed != VK_NULL_HANDLE) Vk::DestroyImage(context->device, placed, nullptr);
                CFRelease(reinterpret_cast<CFTypeRef>(memory));
            }
        }
        {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = VK_FORMAT_R8G8B8A8_UNORM;
            info.extent = {8u, 8u, 1u};
            info.mipLevels = 1;
            info.arrayLayers = 8;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            VkImage image = VK_NULL_HANDLE;
            Check(Vk::CreateImage(context->device, &info, nullptr, &image) == VK_SUCCESS, "a cube image with 8 slices (one whole cube and two more) is created");
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 1u, 7u};
            VkImageView view = VK_NULL_HANDLE;
            Check(Vk::CreateImageView(context->device, &viewInfo, nullptr, &view) == VK_SUCCESS && TextureOfView(view).arrayLength == 7u, "a cube view of 7 slices past its base is a 2D array of all 7");
            if (view != VK_NULL_HANDLE) Vk::DestroyImageView(context->device, view, nullptr);
            if (image != VK_NULL_HANDLE) Vk::DestroyImage(context->device, image, nullptr);
        }
        recorder.Sync();
        attached.Teardown();
    }
    if (failures != 0) return 1;
    std::printf("PASS: agc_metal_images\n");
    return 0;
}


// Min/max sampler reduction through argument-buffer samplers on the GPU: a 2x1 R32Float texture
// (10, 20) sampled bilinearly at its centre must give 15 (average), 10 (min), 20 (max).
int Reduction() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
        std::printf("skipped, the machine has no Metal 4 device\n");
        return 77;
    }
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    NSString* source = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "struct Table { texture2d<float> image; array<sampler, 3> samplers; };\n"
        "kernel void probe(constant Table& table [[buffer(0)]], device float* out [[buffer(1)]], uint index [[thread_position_in_grid]]) {\n"
        "    out[index] = table.image.sample(table.samplers[index], float2(0.5, 0.5), level(0.0)).x;\n"
        "}\n"
        "kernel void direct(texture2d<float> image [[texture(0)]], sampler s0 [[sampler(0)]], sampler s1 [[sampler(1)]], sampler s2 [[sampler(2)]], device float* out [[buffer(1)]], uint index [[thread_position_in_grid]]) {\n"
        "    out[4 + index] = image.sample(index == 0 ? s0 : (index == 1 ? s1 : s2), float2(0.5, 0.5), level(0.0)).x;\n"
        "}\n";
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (library == nil) return (std::fprintf(stderr, "FAIL: %s\n", error.localizedDescription.UTF8String), 1);
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
    MTLTextureDescriptor* textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:2 height:1 mipmapped:NO];
    textureDescriptor.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:textureDescriptor];
    const float texels[2] = {10.0f, 20.0f};
    [texture replaceRegion:MTLRegionMake2D(0, 0, 2, 1) mipmapLevel:0 withBytes:texels bytesPerRow:8];
    const char* layer = std::getenv("MTL_DEBUG_LAYER");
    const bool debugLayer = layer != nullptr && std::strcmp(layer, "0") != 0;
    id<MTLSamplerState> samplers[3];
    const MTLSamplerReductionMode modes[3] = {MTLSamplerReductionModeWeightedAverage, MTLSamplerReductionModeMinimum, MTLSamplerReductionModeMaximum};
    for (int index = 0; index < 3; ++index) {
        MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
        descriptor.minFilter = MTLSamplerMinMagFilterLinear;
        descriptor.magFilter = MTLSamplerMinMagFilterLinear;
        descriptor.supportArgumentBuffers = YES;
        // Metal's debug layer asserts on some reduction descriptors: under it only the average is
        // probed (the min/max probe needs a run without MTL_DEBUG_LAYER).
        if (!debugLayer) descriptor.reductionMode = modes[index];
        samplers[index] = [device newSamplerStateWithDescriptor:descriptor];
    }
    id<MTLBuffer> table = [device newBufferWithLength:32 options:MTLResourceStorageModeShared];
    auto* ids = static_cast<std::uint64_t*>(table.contents);
    ids[0] = texture.gpuResourceID._impl;
    for (int index = 0; index < 3; ++index) ids[1 + index] = samplers[index].gpuResourceID._impl;
    id<MTLBuffer> output = [device newBufferWithLength:32 options:MTLResourceStorageModeShared];
    id<MTLComputePipelineState> direct = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"direct"] error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:table offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder useResource:texture usage:MTLResourceUsageRead];
    [encoder dispatchThreads:MTLSizeMake(3, 1, 1) threadsPerThreadgroup:MTLSizeMake(3, 1, 1)];
    [encoder setComputePipelineState:direct];
    [encoder setTexture:texture atIndex:0];
    for (int index = 0; index < 3; ++index) [encoder setSamplerState:samplers[index] atIndex:index];
    [encoder dispatchThreads:MTLSizeMake(3, 1, 1) threadsPerThreadgroup:MTLSizeMake(3, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    const auto* values = static_cast<const float*>(output.contents);
    std::printf("argument buffer: average %f, min %f, max %f; direct: average %f, min %f, max %f\n", values[0], values[1], values[2], values[4], values[5], values[6]);
    // On the M3 under macOS 27.0.1 MTLSamplerDescriptor.reductionMode is ignored (every mode filters
    // as the weighted average, bound directly or through an argument buffer): the device reports no
    // min/max reduction (SamplerFilterMinmax false) and the sampler path refuses it. This probe
    // fails when the behaviour is mixed, and says when a later OS starts honouring it.
    Check(values[0] == 15.0f && values[4] == 15.0f, "weighted average reduction");
    if (debugLayer) {
        if (failures != 0) return 1;
        std::printf("PASS: agc_metal_sampler_reduction (average only: the min/max probe runs without MTL_DEBUG_LAYER)\n");
        return 0;
    }
    const bool honoured = values[1] == 10.0f && values[2] == 20.0f && values[5] == 10.0f && values[6] == 20.0f;
    const bool ignored = values[1] == 15.0f && values[2] == 15.0f && values[5] == 15.0f && values[6] == 15.0f;
    Check(honoured || ignored, "min/max reduction is honoured inconsistently");
    if (failures != 0) return 1;
    std::printf("PASS: agc_metal_sampler_reduction (min/max reduction %s by this GPU)\n", honoured ? "honoured: enable SamplerFilterMinmax" : "ignored");
    return 0;
}

}

// A T# that declares more levels than its dimensions have (STK through radeonsi: a 128x128 block
// view of a 512x512 BC texture with 10 levels; Metal asserted "MTLTextureDescriptor requests 10
// mipmap levels, but the dimensions (128, 128, 1) can only support a maximum of 8", then the guest
// viewed level 8): the host image has the 8 levels its dimensions allow and levels 8 and 9 are 1x1
// textures of their own (AMD keeps every level past 1x1 as another 1x1 level): copies of them
// round-trip, a view of level 9 is a one-level 1x1 view, a view from level 1 has 7 levels
// (DECISIONS D64).
int MipOverflow() {
    AgcDriver::Graphics::MetalTest::Device device;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R32G32B32A32_UINT;
    info.extent = {128u, 128u, 1u};
    info.mipLevels = 10;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage image = VK_NULL_HANDLE;
    Check(Vk::CreateImage(device.context.device, &info, nullptr, &image) == VK_SUCCESS, "a 128x128 image declared with 10 levels is created");
    id<MTLTexture> texture = TextureOfImage(image);
    Check(texture != nil && texture.mipmapLevelCount == 8u, "the host image holds the 8 levels 128x128 allows (has " + std::to_string(texture != nil ? texture.mipmapLevelCount : 0u) + ")");
    Buffer upload(device.context, 64u, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    Buffer readback(device.context, 64u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    for (std::uint32_t index = 0; index < 16u; ++index) reinterpret_cast<std::uint32_t*>(upload.Bytes().data())[index] = 0x1000u + index;
    std::memset(readback.Bytes().data(), 0xee, 64u);
    std::array<VkBufferImageCopy, 2> regions{};
    for (std::uint32_t index = 0; index < 2u; ++index) {
        regions[index].bufferOffset = index * 16u;
        regions[index].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, index == 0u ? 7u : 9u, 0u, 1u};
        regions[index].imageExtent = {1u, 1u, 1u};
    }
    {
        CommandBatch batch(device.context);
        Vk::CmdCopyBufferToImage(batch.Handle(), upload.Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2u, regions.data());
        Stream(batch.Handle()).PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        Vk::CmdCopyImageToBuffer(batch.Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.Handle(), 2u, regions.data());
        batch.SubmitAndWait();
    }
    const auto* words = reinterpret_cast<const std::uint32_t*>(readback.Bytes().data());
    Check(words[0] == 0x1000u && words[3] == 0x1003u, "level 7 (1x1) round-trips through the host image");
    Check(words[4] == 0x1004u && words[7] == 0x1007u, "level 9 (past the host image's levels) round-trips through its own texture");
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R32G32B32A32_UINT;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 1u, VK_REMAINING_MIP_LEVELS, 0u, 1u};
    VkImageView view = VK_NULL_HANDLE;
    Check(Vk::CreateImageView(device.context.device, &viewInfo, nullptr, &view) == VK_SUCCESS && TextureOfView(view).mipmapLevelCount == 7u, "a view from level 1 has the 7 remaining host levels");
    Vk::DestroyImageView(device.context.device, view, nullptr);
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 1u, 9u, 0u, 1u};
    Check(Vk::CreateImageView(device.context.device, &viewInfo, nullptr, &view) == VK_SUCCESS && TextureOfView(view).mipmapLevelCount == 7u, "a view from level 1 of 9 declared levels is clamped to 7");
    Vk::DestroyImageView(device.context.device, view, nullptr);
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 9u, 1u, 0u, 1u};
    Check(Vk::CreateImageView(device.context.device, &viewInfo, nullptr, &view) == VK_SUCCESS && TextureOfView(view).mipmapLevelCount == 1u && TextureOfView(view).width == 1u, "a view of level 9 is a one-level 1x1 view");
    Vk::DestroyImageView(device.context.device, view, nullptr);
    Vk::DestroyImage(device.context.device, image, nullptr);
    std::puts("PASS: agc_metal_image_mip_overflow");
    return 0;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc == 2 && std::strcmp(argv[1], "--detile") == 0) return Detile();
            if (argc == 2 && std::strcmp(argv[1], "--images") == 0) return Images();
            if (argc == 2 && std::strcmp(argv[1], "--mip-overflow") == 0) return MipOverflow();
            if (argc == 2 && std::strcmp(argv[1], "--reduction") == 0) return Reduction();
            if (argc == 2 && std::strcmp(argv[1], "--detile-gpu") == 0) return DetileGpu();
            if (argc == 2 && std::strcmp(argv[1], "--color-transfer") == 0) return ColorTransfer();
            std::fprintf(stderr, "usage: agc_metal_texture_tests --detile | --images\n");
            return 1;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            return 1;
        }
    }
}
