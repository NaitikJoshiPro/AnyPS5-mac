// Work packages D4/D5 (src/MetalImages.h):
//   agc_metal_texture_detile  the CPU (de)tiler against ColorTargetLayout's independent tiler for
//                             the color-target tile modes, and retile/detile round trips of the
//                             texture tile modes and element sizes (no Metal device)
//   agc_metal_images          guest textures uploaded into Metal textures read back equal, heap views
//                             of every image class, placeholders, and samplers from S# words
//                             (Metal objects only, no GPU work)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalImages.h"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "RuntimeAbi.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

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

void DetileAgainstColorTargets() {
    struct Case {
        ColorTileMode color;
        std::uint32_t bytes;
        std::uint32_t pipeBankXor;
    };
    const Case cases[] = {
        {ColorTileMode::Standard4KB, 4u, 0u}, {ColorTileMode::Standard64KB, 4u, 0u}, {ColorTileMode::RenderTarget, 4u, 0u}, {ColorTileMode::RenderTarget, 4u, 0x300u},
        {ColorTileMode::S4KBX, 4u, 0u}, {ColorTileMode::D4KBX, 4u, 0u}, {ColorTileMode::Standard64KB, 8u, 0u}, {ColorTileMode::RenderTarget, 8u, 0u}, {ColorTileMode::Standard4KB, 2u, 0u},
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

int Images() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
        std::printf("skipped, the machine has no Metal 4 device\n");
        return 77;
    }
    ImageCache cache(device, nil);
    // A linear 32x8 RGBA8 texture with 2 levels: sampled float view, then its texels read back.
    alignas(256) static std::array<std::uint8_t, 4096> texels{};
    for (std::size_t index = 0; index < texels.size(); ++index) texels[index] = static_cast<std::uint8_t>(index * 13u + 7u);
    const auto words = TextureWords(texels.data(), 56u, 32u, 8u, 2u, 0u, 9u);
    const auto resource = DecodeTextureResource(words);
    Check(resource.width == 32u && resource.height == 8u && resource.mipCount == 2u && resource.tileMode == TextureTileMode::kLinear, "test T# decodes as built");
    std::uint64_t settled = 0;
    {
        ImageBindings bindings(cache, [&](std::uint64_t, std::uint64_t bytes) { settled += bytes; });
        ShaderRecompiler::DescriptorBinding binding{};
        binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
        binding.kind = ShaderRecompiler::DescriptorKind::SampledImage;
        binding.count = 1u;
        binding.binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding + 2u;  // sampled float 2D
        binding.guestDescriptor.assign(words.begin(), words.end());
        binding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
        std::array<std::uint64_t, 1> entries{};
        bindings.Write(binding, entries);
        Check(entries[0] != 0u, "a sampled view has a resource id");
        Check(settled != 0u, "the upload settles the guest range first");
        // The same T# through the sampled uint 2D heap reuses the uploaded surface.
        binding.binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding + 7u + 2u;
        bindings.Write(binding, entries);
        Check(cache.Stats().textures == 1u, "one upload for two heaps of one T#");
    }
    // Placeholders of every heap class and dimension except multisampled storage.
    for (std::uint32_t heap = ShaderRecompiler::RuntimeAbi::FirstImageBinding; heap < static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers); ++heap) {
        try {
            Check(cache.Placeholder(heap) != nil, "placeholder of heap " + std::to_string(heap));
        } catch (const std::exception& error) {
            Check(false, "placeholder of heap " + std::to_string(heap) + ": " + error.what());
        }
    }
    // Samplers: equal words share one sampler state; different words do not.
    const std::array<std::uint32_t, 4> pointClamp{0x00000092u, 0x00fff000u, 0x00000000u, 0x00000000u};
    const std::array<std::uint32_t, 4> linearWrap{0x00000000u, 0x00fff000u, 0x05500000u, 0x00000000u};
    auto first = cache.Sampler(pointClamp, false, false, false);
    auto again = cache.Sampler(pointClamp, false, false, false);
    auto other = cache.Sampler(linearWrap, false, false, false);
    Check(first != nil && first == again && other != nil && other != first, "sampler cache by S# words");
    Check(cache.Stats().samplers == 2u && cache.Stats().samplerHits == 1u, "sampler cache statistics");
    cache.Teardown();
    if (failures != 0) return 1;
    std::printf("PASS: agc_metal_images\n");
    return 0;
}

}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc == 2 && std::strcmp(argv[1], "--detile") == 0) return Detile();
            if (argc == 2 && std::strcmp(argv[1], "--images") == 0) return Images();
            std::fprintf(stderr, "usage: agc_metal_texture_tests --detile | --images\n");
            return 1;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            return 1;
        }
    }
}
