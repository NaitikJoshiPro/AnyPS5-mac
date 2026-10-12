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
    id<MTLSamplerState> samplers[3];
    const MTLSamplerReductionMode modes[3] = {MTLSamplerReductionModeWeightedAverage, MTLSamplerReductionModeMinimum, MTLSamplerReductionModeMaximum};
    for (int index = 0; index < 3; ++index) {
        MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
        descriptor.minFilter = MTLSamplerMinMagFilterLinear;
        descriptor.magFilter = MTLSamplerMinMagFilterLinear;
        descriptor.supportArgumentBuffers = YES;
        descriptor.reductionMode = modes[index];
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
    const bool honoured = values[1] == 10.0f && values[2] == 20.0f && values[5] == 10.0f && values[6] == 20.0f;
    const bool ignored = values[1] == 15.0f && values[2] == 15.0f && values[5] == 15.0f && values[6] == 15.0f;
    Check(honoured || ignored, "min/max reduction is honoured inconsistently");
    if (failures != 0) return 1;
    std::printf("PASS: agc_metal_sampler_reduction (min/max reduction %s by this GPU)\n", honoured ? "honoured: enable SamplerFilterMinmax" : "ignored");
    return 0;
}

}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc == 2 && std::strcmp(argv[1], "--detile") == 0) return Detile();
            if (argc == 2 && std::strcmp(argv[1], "--images") == 0) return Images();
            if (argc == 2 && std::strcmp(argv[1], "--reduction") == 0) return Reduction();
            std::fprintf(stderr, "usage: agc_metal_texture_tests --detile | --images\n");
            return 1;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            return 1;
        }
    }
}
