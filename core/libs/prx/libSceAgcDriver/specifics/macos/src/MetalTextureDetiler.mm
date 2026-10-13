// Work package D4: guest texture (de)tiling on the CPU, the addressing of
// Graphics/shaders/TextureDetile.comp (Graphics/src/TextureDetiler.cpp sets its constants): linear
// pitched rows, the standard 256 B / 4 KiB / 64 KiB swizzles, and the XOR equation modes (thin and
// thick 3D blocks). The detile kernel in MSL replaces this on the GPU later (TechnicalDebt
// `metal-d4-texture-uploads`).
#include "MetalImages.h"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"

#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics::Metal {

namespace {

[[noreturn]] void Fail(const std::string& message) {
    throw std::runtime_error("Metal textures (work package D4): " + message);
}

std::uint32_t BlockBytesFor(TextureTileMode tileMode) {
    switch (tileMode) {
    case TextureTileMode::kLinear: return 0u;
    case TextureTileMode::kStandard256B:
    case TextureTileMode::kD256B: return 256u;
    case TextureTileMode::kStandard4KB:
    case TextureTileMode::kD4KB:
    case TextureTileMode::kS4KBX:
    case TextureTileMode::kD4KBX: return 4096u;
    case TextureTileMode::kStandard64KB:
    case TextureTileMode::kD64KB:
    case TextureTileMode::kS64KBT:
    case TextureTileMode::kD64KBT:
    case TextureTileMode::kZ64KBX:
    case TextureTileMode::kS64KBX:
    case TextureTileMode::kD64KBX:
    case TextureTileMode::kR64KBX: return 65536u;
    }
    Fail("unknown tile mode");
}

std::uint32_t StandardOffset(std::uint32_t elementBytes, std::uint32_t x, std::uint32_t y) {
    switch (elementBytes) {
    case 1u:
        return ((y << 4) & 0x1f0u) ^ ((y << 5) & 0x400u) ^ (x & 0x00fu) ^ ((x << 5) & 0x200u) ^ ((x << 6) & 0x800u);
    case 2u:
        return ((y << 4) & 0x070u) ^ ((y << 5) & 0x100u) ^ ((y << 6) & 0x400u) ^ ((x << 1) & 0x00eu) ^ ((x << 4) & 0x080u) ^ ((x << 5) & 0x200u) ^ ((x << 6) & 0x800u);
    case 4u:
        return ((y << 4) & 0x070u) ^ ((y << 5) & 0x100u) ^ ((y << 6) & 0x400u) ^ ((x << 2) & 0x00cu) ^ ((x << 5) & 0x080u) ^ ((x << 6) & 0x200u) ^ ((x << 7) & 0x800u);
    case 8u:
        return ((y << 4) & 0x030u) ^ ((y << 6) & 0x100u) ^ ((y << 7) & 0x400u) ^ ((x << 3) & 0x008u) ^ ((x << 5) & 0x0c0u) ^ ((x << 6) & 0x200u) ^ ((x << 7) & 0x800u);
    default:
        return ((y << 4) & 0x030u) ^ ((y << 6) & 0x100u) ^ ((y << 7) & 0x400u) ^ ((x << 6) & 0x0c0u) ^ ((x << 7) & 0x200u) ^ ((x << 8) & 0x800u);
    }
}

std::uint32_t Standard64Extra(std::uint32_t elementBytes, std::uint32_t x, std::uint32_t y) {
    switch (elementBytes) {
    case 1u: return ((x << 7) & 0x2000u) ^ ((x << 8) & 0x8000u) ^ ((y << 6) & 0x1000u) ^ ((y << 7) & 0x4000u);
    case 2u: return ((x << 7) & 0x2000u) ^ ((x << 8) & 0x8000u) ^ ((y << 7) & 0x1000u) ^ ((y << 8) & 0x4000u);
    case 4u: return ((x << 8) & 0x2000u) ^ ((x << 9) & 0x8000u) ^ ((y << 7) & 0x1000u) ^ ((y << 8) & 0x4000u);
    case 8u: return ((x << 8) & 0x2000u) ^ ((x << 9) & 0x8000u) ^ ((y << 8) & 0x1000u) ^ ((y << 9) & 0x4000u);
    default: return ((x << 9) & 0x2000u) ^ ((x << 10) & 0x8000u) ^ ((y << 8) & 0x1000u) ^ ((y << 9) & 0x4000u);
    }
}

struct Addressing {
    std::uint32_t family = 0;  // 0 linear, 1 standard, 2 equation
    std::uint32_t blockBytes = 0;
    std::array<std::uint32_t, 16> equation{};
    std::uint32_t blockWidth = 0;
    std::uint32_t blockHeight = 0;
};

Addressing AddressingFor(const DetileLevel& level) {
    Addressing addressing;
    const auto elementBytes = level.elementBytes;
    if (level.tileMode != TextureTileMode::kLinear && (!std::has_single_bit(elementBytes) || elementBytes > 16u)) Fail("unsupported element size " + std::to_string(elementBytes) + " for texture detiling");
    addressing.blockBytes = BlockBytesFor(level.tileMode);
    addressing.family = level.tileMode == TextureTileMode::kLinear ? 0u : 1u;
    if (level.thick) {
        const auto tileMode = level.tileMode;
        const auto thickMode = tileMode == TextureTileMode::kStandard4KB ? 0x105u : tileMode == TextureTileMode::kStandard64KB ? 0x109u : tileMode == TextureTileMode::kS64KBX ? 0x119u : tileMode == TextureTileMode::kD64KBX ? 0x11au : 0u;
        if (thickMode == 0u) Fail("3D textures are only detiled thick from SW_4KB_S, SW_64KB_S, SW_64KB_S_X or SW_64KB_D_X");
        const auto* equation = FindTextureSwizzleEquation(thickMode, elementBytes);
        if (equation == nullptr) Fail("no thick swizzle equation for the element size");
        addressing.family = 2u;
        addressing.equation = equation->bits;
        const auto extent = ThickBlockExtent(tileMode, elementBytes);
        addressing.blockWidth = extent[0];
        addressing.blockHeight = extent[1];
    } else if (const auto mode = EquationSwizzleMode(level.tileMode); mode != 0u) {
        const auto* equation = FindTextureSwizzleEquation(mode, elementBytes);
        if (equation == nullptr) Fail("no swizzle equation for tile mode " + std::to_string(mode) + " at " + std::to_string(elementBytes) + " bytes per element");
        addressing.family = 2u;
        addressing.equation = equation->bits;
    }
    if (addressing.blockWidth == 0u && addressing.family != 0u) {
        const auto blockBytes = addressing.blockBytes;
        std::uint32_t width;
        if (blockBytes <= 256u) width = elementBytes <= 2u ? 16u : (elementBytes <= 8u ? 8u : 4u);
        else if (blockBytes <= 4096u) width = elementBytes <= 2u ? 64u : (elementBytes <= 8u ? 32u : 16u);
        else width = elementBytes <= 2u ? 256u : (elementBytes <= 8u ? 128u : 64u);
        addressing.blockWidth = width;
        addressing.blockHeight = blockBytes / (width * elementBytes);
    }
    return addressing;
}

std::uint32_t EquationOffset(const Addressing& addressing, std::uint32_t x, std::uint32_t y, std::uint32_t slice) {
    std::uint32_t offset = 0;
    for (std::uint32_t bit = 0; bit < 16u; ++bit) {
        const auto mask = addressing.equation[bit];
        const auto selected = (x & (mask & 0xfffu)) ^ ((y << 12) & (mask & 0xfff000u)) ^ ((slice << 24) & (mask & 0xff000000u));
        offset |= (static_cast<std::uint32_t>(std::popcount(selected)) & 1u) << bit;
    }
    return offset;
}

std::uint64_t Offset(const DetileLevel& level, const Addressing& addressing, std::uint32_t x, std::uint32_t y) {
    const auto& mip = *level.mip;
    if (addressing.family == 0u) return static_cast<std::uint64_t>(y) * mip.pitchBytes + static_cast<std::uint64_t>(x) * level.elementBytes;
    std::uint32_t sx = x;
    std::uint32_t sy = y;
    std::uint32_t bx = 0;
    std::uint32_t by = 0;
    if (mip.tail) {
        sx += mip.tailX;
        sy += mip.tailY;
    } else {
        bx = x / addressing.blockWidth;
        by = y / addressing.blockHeight;
    }
    std::uint32_t inBlock;
    if (addressing.family == 2u) {
        inBlock = EquationOffset(addressing, sx, sy, level.slice) ^ level.pipeBankXor;
    } else {
        inBlock = StandardOffset(level.elementBytes, sx, sy);
        if (addressing.blockBytes > 4096u) inBlock ^= Standard64Extra(level.elementBytes, sx, sy);
        inBlock &= addressing.blockBytes - 1u;
    }
    const auto blockIndex = static_cast<std::uint64_t>(by) * mip.blocksPerRow + bx;
    return blockIndex * addressing.blockBytes + inBlock;
}

template <bool Retile>
void Move(const DetileLevel& level, std::byte* tiled, std::uint64_t tiledBytes, std::byte* linear, std::uint64_t linearPitch, std::uint32_t linearElementBytes) {
    const auto addressing = AddressingFor(level);
    const auto& mip = *level.mip;
    const auto bytes = level.elementBytes;
    for (std::uint32_t y = 0; y < mip.height; ++y) {
        auto* row = linear + static_cast<std::uint64_t>(y) * linearPitch;
        for (std::uint32_t x = 0; x < mip.width; ++x) {
            const auto offset = Offset(level, addressing, x, y);
            if (offset + bytes > tiledBytes) continue;
            auto* element = row + static_cast<std::uint64_t>(x) * linearElementBytes;
            if constexpr (Retile) {
                std::memcpy(tiled + offset, element, bytes);
            } else {
                std::memcpy(element, tiled + offset, bytes);
            }
        }
    }
}

}

std::uint64_t TiledElementOffset(const DetileLevel& level, std::uint32_t x, std::uint32_t y) {
    return Offset(level, AddressingFor(level), x, y);
}

void DetileCpu(const DetileLevel& level, const std::byte* tiled, std::uint64_t tiledBytes, std::byte* linear, std::uint64_t linearPitch, std::uint32_t linearElementBytes) {
    Move<false>(level, const_cast<std::byte*>(tiled), tiledBytes, linear, linearPitch, linearElementBytes);
}

void RetileCpu(const DetileLevel& level, std::byte* tiled, std::uint64_t tiledBytes, const std::byte* linear, std::uint64_t linearPitch, std::uint32_t linearElementBytes) {
    Move<true>(level, tiled, tiledBytes, const_cast<std::byte*>(linear), linearPitch, linearElementBytes);
}

}
