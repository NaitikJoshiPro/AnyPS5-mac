// Work package D4: guest texture formats as Metal pixel formats (the Metal twin of the table in
// Graphics/src/TextureFormat.cpp, whose CPU helpers the rest of the driver keeps using).
#include "MetalImages.h"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"

#include <cstdlib>
#include <cstring>

#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics::Metal {

namespace {

struct Entry {
    std::uint32_t guest;
    MTLPixelFormat format;
    std::uint32_t bytes;
    std::uint32_t textureBytes;
    bool blockCompressed;
    bool swapRedBlue;
};

// Packed 16-bit formats: Metal names them from the least significant bits (B5G6R5Unorm holds blue
// in bits 0-4, as Vulkan's R5G6B5_UNORM_PACK16), so 133 and 134 match directly; Vulkan's
// B4G4R4A4_UNORM_PACK16 (136) is Metal's ABGR4Unorm with red and blue exchanged.
constexpr Entry Formats[] = {
    {1, MTLPixelFormatR8Unorm, 1, 1, false, false},
    {2, MTLPixelFormatR8Snorm, 1, 1, false, false},
    {5, MTLPixelFormatR8Uint, 1, 1, false, false},
    {6, MTLPixelFormatR8Sint, 1, 1, false, false},
    {7, MTLPixelFormatR16Unorm, 2, 2, false, false},
    {8, MTLPixelFormatR16Snorm, 2, 2, false, false},
    {11, MTLPixelFormatR16Uint, 2, 2, false, false},
    {12, MTLPixelFormatR16Sint, 2, 2, false, false},
    {13, MTLPixelFormatR16Float, 2, 2, false, false},
    {14, MTLPixelFormatRG8Unorm, 2, 2, false, false},
    {15, MTLPixelFormatRG8Snorm, 2, 2, false, false},
    {18, MTLPixelFormatRG8Uint, 2, 2, false, false},
    {19, MTLPixelFormatRG8Sint, 2, 2, false, false},
    {20, MTLPixelFormatR32Uint, 4, 4, false, false},
    {21, MTLPixelFormatR32Sint, 4, 4, false, false},
    {22, MTLPixelFormatR32Float, 4, 4, false, false},
    {23, MTLPixelFormatRG16Unorm, 4, 4, false, false},
    {24, MTLPixelFormatRG16Snorm, 4, 4, false, false},
    {27, MTLPixelFormatRG16Uint, 4, 4, false, false},
    {28, MTLPixelFormatRG16Sint, 4, 4, false, false},
    {29, MTLPixelFormatRG16Float, 4, 4, false, false},
    {36, MTLPixelFormatRG11B10Float, 4, 4, false, false},
    {50, MTLPixelFormatRGB10A2Unorm, 4, 4, false, false},
    {54, MTLPixelFormatRGB10A2Uint, 4, 4, false, false},
    {56, MTLPixelFormatRGBA8Unorm, 4, 4, false, false},
    {57, MTLPixelFormatRGBA8Snorm, 4, 4, false, false},
    {60, MTLPixelFormatRGBA8Uint, 4, 4, false, false},
    {61, MTLPixelFormatRGBA8Sint, 4, 4, false, false},
    {62, MTLPixelFormatRG32Uint, 8, 8, false, false},
    {63, MTLPixelFormatRG32Sint, 8, 8, false, false},
    {64, MTLPixelFormatRG32Float, 8, 8, false, false},
    {65, MTLPixelFormatRGBA16Unorm, 8, 8, false, false},
    {66, MTLPixelFormatRGBA16Snorm, 8, 8, false, false},
    {69, MTLPixelFormatRGBA16Uint, 8, 8, false, false},
    {70, MTLPixelFormatRGBA16Sint, 8, 8, false, false},
    {71, MTLPixelFormatRGBA16Float, 8, 8, false, false},
    // Metal has no three-channel 32-bit formats: four channels, the fourth zero on upload.
    {72, MTLPixelFormatRGBA32Uint, 12, 16, false, false},
    {73, MTLPixelFormatRGBA32Sint, 12, 16, false, false},
    {74, MTLPixelFormatRGBA32Float, 12, 16, false, false},
    {75, MTLPixelFormatRGBA32Uint, 16, 16, false, false},
    {76, MTLPixelFormatRGBA32Sint, 16, 16, false, false},
    {77, MTLPixelFormatRGBA32Float, 16, 16, false, false},
    {128, MTLPixelFormatR8Unorm_sRGB, 1, 1, false, false},
    {129, MTLPixelFormatRG8Unorm_sRGB, 2, 2, false, false},
    {130, MTLPixelFormatRGBA8Unorm_sRGB, 4, 4, false, false},
    {132, MTLPixelFormatRGB9E5Float, 4, 4, false, false},
    {133, MTLPixelFormatB5G6R5Unorm, 2, 2, false, false},
    {134, MTLPixelFormatBGR5A1Unorm, 2, 2, false, false},
    {136, MTLPixelFormatABGR4Unorm, 2, 2, false, true},
    {169, MTLPixelFormatBC1_RGBA, 8, 8, true, false},
    {170, MTLPixelFormatBC1_RGBA_sRGB, 8, 8, true, false},
    {171, MTLPixelFormatBC2_RGBA, 16, 16, true, false},
    {172, MTLPixelFormatBC2_RGBA_sRGB, 16, 16, true, false},
    {173, MTLPixelFormatBC3_RGBA, 16, 16, true, false},
    {174, MTLPixelFormatBC3_RGBA_sRGB, 16, 16, true, false},
    {175, MTLPixelFormatBC4_RUnorm, 8, 8, true, false},
    {176, MTLPixelFormatBC4_RSnorm, 8, 8, true, false},
    {177, MTLPixelFormatBC5_RGUnorm, 16, 16, true, false},
    {178, MTLPixelFormatBC5_RGSnorm, 16, 16, true, false},
    {179, MTLPixelFormatBC6H_RGBUfloat, 16, 16, true, false},
    {180, MTLPixelFormatBC6H_RGBFloat, 16, 16, true, false},
    {181, MTLPixelFormatBC7_RGBAUnorm, 16, 16, true, false},
    {182, MTLPixelFormatBC7_RGBAUnorm_sRGB, 16, 16, true, false},
};

}

std::uint32_t SrgbDecodeFormats() {
    // Metal samples R8 and RG8 sRGB natively, but its decode differs from the hardware-measured
    // table in the low codes; APS5_SRGB_SHADER_DECODE=1 moves the decode into the shader (the
    // device reports these bits in Target().srgbDecodeFormats and the views use the UNORM format).
    const char* forced = std::getenv("APS5_SRGB_SHADER_DECODE");
    if (forced == nullptr || std::strcmp(forced, "1") != 0) return 0u;
    using ShaderRecompiler::IrBufferFormat;
    return ShaderRecompiler::SrgbDecodeBit(IrBufferFormat::Format8Srgb) | ShaderRecompiler::SrgbDecodeBit(IrBufferFormat::Format8_8Srgb);
}

TextureFormatInfo GuestTextureFormat(std::uint32_t guestFormat) {
    const auto decode = SrgbDecodeFormats();
    if (decode != 0u && (decode & ShaderRecompiler::SrgbDecodeBit(static_cast<ShaderRecompiler::IrBufferFormat>(guestFormat))) != 0u) {
        guestFormat = static_cast<std::uint32_t>(ShaderRecompiler::SrgbUnormFormat(static_cast<ShaderRecompiler::IrBufferFormat>(guestFormat)));
    }
    // Formats 30, 34 and 43 (11_11_10 and 10_11_11 variants) are read as R32 uint and unpacked in the
    // shader (ImageResource::conversionFormat), as on the Vulkan path. A sampled 43 is instead a
    // B10G11R11 (RG11B10Float) copy repacked by TextureDetiler::DispatchRepack (Texture.mm, upstream
    // 571fecadd); storage and packed 43 still take this R32 path.
    const auto remapped = guestFormat == 30u || guestFormat == 34u || guestFormat == 43u ? 20u : guestFormat;
    for (const auto& entry : Formats) {
        if (entry.guest == remapped) return {entry.format, entry.bytes, entry.textureBytes, entry.blockCompressed, entry.swapRedBlue};
    }
    throw std::runtime_error("Metal textures (work package D4): guest texture format " + std::to_string(guestFormat) + " has no Metal pixel format");
}

MTLPixelFormat HeapViewFormat(MTLPixelFormat format, std::uint32_t numeric) {
    if (numeric == 0u) return format;
    const bool uint = numeric == 1u;
    switch (format) {
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatR8Snorm:
    case MTLPixelFormatR8Unorm_sRGB:
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR8Sint:
        return uint ? MTLPixelFormatR8Uint : MTLPixelFormatR8Sint;
    case MTLPixelFormatR16Unorm:
    case MTLPixelFormatR16Snorm:
    case MTLPixelFormatR16Float:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR16Sint:
        return uint ? MTLPixelFormatR16Uint : MTLPixelFormatR16Sint;
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatRG8Unorm_sRGB:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG8Sint:
        return uint ? MTLPixelFormatRG8Uint : MTLPixelFormatRG8Sint;
    case MTLPixelFormatR32Float:
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatRGB9E5Float:
        return uint ? MTLPixelFormatR32Uint : MTLPixelFormatR32Sint;
    case MTLPixelFormatRG16Unorm:
    case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG16Sint:
        return uint ? MTLPixelFormatRG16Uint : MTLPixelFormatRG16Sint;
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA8Sint:
        return uint ? MTLPixelFormatRGBA8Uint : MTLPixelFormatRGBA8Sint;
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRGB10A2Uint:
        return uint ? MTLPixelFormatRGB10A2Uint : MTLPixelFormatR32Sint;
    case MTLPixelFormatRG32Float:
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRG32Sint:
        return uint ? MTLPixelFormatRG32Uint : MTLPixelFormatRG32Sint;
    case MTLPixelFormatRGBA16Unorm:
    case MTLPixelFormatRGBA16Snorm:
    case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Sint:
        return uint ? MTLPixelFormatRGBA16Uint : MTLPixelFormatRGBA16Sint;
    case MTLPixelFormatRGBA32Float:
    case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGBA32Sint:
        return uint ? MTLPixelFormatRGBA32Uint : MTLPixelFormatRGBA32Sint;
    default:
        return format;
    }
}

MTLPixelFormat DepthFormatFor(MTLPixelFormat format) {
    if (format == MTLPixelFormatR32Float) return MTLPixelFormatDepth32Float;
    if (format == MTLPixelFormatR16Unorm) return MTLPixelFormatDepth16Unorm;
    throw std::runtime_error("Metal textures (work package D4): comparison sampling requires an R32 float or R16 unorm depth texture");
}

}
