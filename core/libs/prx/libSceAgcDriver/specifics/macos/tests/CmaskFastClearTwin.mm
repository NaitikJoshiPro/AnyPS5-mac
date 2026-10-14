// Metal twin of tests/CmaskFastClear.cpp (work package D6, specifics/macos/twins/D6-draw-present.cmake):
// the same CMASK fast-clear draw and eliminate cases, with the cover shaders as hand-written MSL
// functions (the original's are SPIR-V). Keep it in step with the original (tools/mac/port_ledger.py).
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

constexpr int VulkanTestSkipped = 77;
constexpr std::uint32_t Width = 61;
constexpr std::uint32_t Height = 13;
constexpr std::uint32_t Stale = 0x5a5a5a5au;
constexpr std::uint32_t Clear = 0x11223344u;
constexpr std::uint32_t Drawn = 0xffffffffu;

alignas(4096) std::array<std::byte, 2 * 65536> SurfaceStorage{};
const std::span<std::byte, 65536> Surface{reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(SurfaceStorage.data()) + 65535u) & ~std::uintptr_t{65535u}), 65536};
alignas(4096) std::array<std::uint8_t, 4096> Cmask{};
alignas(4096) std::array<std::uint8_t, 4096> Dcc{};

ColorTarget target(ColorTileMode mode) {
    const ColorTargetLayout layout(Width, Height, mode, 4);
    ColorTarget color{};
    color.address = reinterpret_cast<std::uintptr_t>(Surface.data());
    color.extent = {Width, Height};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.bytes = layout.Bytes();
    color.componentMapping = 0xe4u;
    color.tileMode = mode;
    color.elementBytes = 4;
    color.clearWords = {Clear, 0};
    color.surfaceAddress = color.address;
    color.surfaceExtent = color.extent;
    color.cmaskAddress = reinterpret_cast<std::uintptr_t>(Cmask.data());
    color.cmaskBytes = CmaskLayout(Width, Height).Bytes();
    return color;
}

void prepare(const ColorTarget& color, const std::function<bool(std::uint32_t, std::uint32_t)>& cleared, std::uint8_t code = 0) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    StorageTexture::FlushPending(color.address, Surface.size(), nullptr, "test");
    std::vector<std::byte> stale(Surface.size());
    for (std::size_t offset = 0; offset < stale.size(); offset += 4) std::memcpy(stale.data() + offset, &Stale, 4);
    AgcDriver::GuestMemory::Write(color.address, stale);
    const CmaskLayout layout(Width, Height);
    std::vector<std::byte> mask(Cmask.size(), std::byte{0xff});
    for (std::uint32_t tileY = 0; tileY < layout.TilesY(); ++tileY) {
        for (std::uint32_t tileX = 0; tileX < layout.TilesX(); ++tileX) {
            if (!cleared(tileX, tileY)) continue;
            const auto nibble = layout.Nibble(tileX, tileY);
            mask[nibble / 2u] &= std::byte{static_cast<std::uint8_t>((nibble % 2u) == 0 ? 0xf0u : 0x0fu)};
            mask[nibble / 2u] |= std::byte{static_cast<std::uint8_t>((nibble % 2u) == 0 ? code : code << 4u)};
        }
    }
    AgcDriver::GuestMemory::Write(color.cmaskAddress, mask);
}

// The cover shaders (tests/shaders/Cover.vert/.frag: a full-screen triangle in opaque white) as
// MetalBackend functions under the MslAbi Version 3 contract.
ShaderRecompiler::RecompileResult coverShader(bool vertexStage) {
    namespace Abi = ShaderRecompiler::MslAbi;
    static const char* vertexSource = R"(#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
vertex VertexOut aps5_main(uint id [[vertex_id]]) {
    VertexOut out;
    out.position = float4(float((id & 1u) << 2u) - 1.0f, float((id & 2u) << 1u) - 1.0f, 0.0f, 1.0f);
    return out;
}
)";
    static const char* fragmentSource = R"(#include <metal_stdlib>
using namespace metal;
struct FragmentOut { float4 color [[color(0)]]; };
fragment FragmentOut aps5_main() {
    FragmentOut out;
    out.color = float4(1.0f);
    return out;
}
)";
    Abi::Reflection reflection;
    reflection.stage = vertexStage ? Abi::Stage::Vertex : Abi::Stage::Fragment;
    reflection.group = vertexStage ? ShaderRecompiler::RuntimeAbi::Stage::Main : ShaderRecompiler::RuntimeAbi::Stage::Fragment;
    reflection.colorOutputs = vertexStage ? 0u : 1u;
    ShaderRecompiler::RecompileResult result;
    result.msl = ShaderRecompiler::SharedMsl(ShaderRecompiler::MslShader{vertexStage ? vertexSource : fragmentSource, reflection});
    return result;
}

void draw(AgcDriver::VulkanDevice& device, const ColorTarget& color, VkRect2D scissor) {
    const auto vertex = coverShader(true);
    const auto fragment = coverShader(false);
    const std::array<CompiledShader, 2> shaders{{
        {ShaderRecompiler::ShaderStage::Vertex, &vertex, 0},
        {ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}
    }};
    State state{};
    state.stages = {ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.color = color;
    state.colors = {color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, 0, static_cast<float>(Width), static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = scissor;
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    device.Draw(state, {0, 3, 0, 1, 0, false}, shaders);
    device.WaitIdle();
}

void expect(AgcDriver::VulkanDevice& device, const ColorTarget& color, const std::function<std::uint32_t(std::uint32_t, std::uint32_t)>& texel, const std::string& what) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    StorageTexture::FlushPending(color.address, color.bytes, nullptr, "test");
    device.WaitIdle();
    const ColorTargetLayout layout(Width, Height, color.tileMode, 4);
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            std::uint32_t stored = 0;
            std::memcpy(&stored, Surface.data() + layout.Offset(x, y), 4);
            Require(stored == texel(x, y), what + ": texel (" + std::to_string(x) + ", " + std::to_string(y) + ") holds " + std::to_string(stored));
        }
    }
    Require(std::all_of(Cmask.begin(), Cmask.end(), [](std::uint8_t value) { return value == 0xffu; }), what + ": the CMASK was not left expanded");
}

void eliminate(AgcDriver::VulkanDevice& device, const ColorTarget& color) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    device.ColorMetadataPass({ColorMetadataPass::Mode::EliminateFastClear, {color}});
}

void run(AgcDriver::VulkanDevice& device, ColorTileMode mode, const std::string& name) {
    const auto color = target(mode);
    prepare(color, [](std::uint32_t, std::uint32_t) { return true; });
    draw(device, color, {{0, 0}, {20, Height}});
    expect(device, color, [](std::uint32_t x, std::uint32_t) { return x < 20 ? Drawn : Clear; }, name + " fast-cleared target drawn over its left part");
    draw(device, color, {{40, 0}, {Width - 40, Height}});
    expect(device, color, [](std::uint32_t x, std::uint32_t) { return x < 20 || x >= 40 ? Drawn : Clear; }, name + " fast-cleared target drawn again over its right part");
    prepare(color, [](std::uint32_t tileX, std::uint32_t tileY) { return (tileX == 0 && tileY == 0) || (tileX == 7 && tileY == 1); });
    draw(device, color, {{0, 0}, {20, Height}});
    expect(device, color, [](std::uint32_t x, std::uint32_t y) { return x < 20 ? Drawn : x >= 56 && y >= 8 ? Clear : Stale; }, name + " partly fast-cleared target drawn over its left part");
    prepare(color, [](std::uint32_t, std::uint32_t) { return true; });
    eliminate(device, color);
    expect(device, color, [](std::uint32_t, std::uint32_t) { return Clear; }, name + " fast-cleared target eliminated without a draw");
    prepare(color, [](std::uint32_t tileX, std::uint32_t tileY) { return tileX == 7 && tileY == 1; });
    eliminate(device, color);
    expect(device, color, [](std::uint32_t x, std::uint32_t y) { return x >= 56 && y >= 8 ? Clear : Stale; }, name + " partly fast-cleared edge tile eliminated");
    prepare(color, [](std::uint32_t tileX, std::uint32_t tileY) { return (tileX + tileY) % 2u == 0; });
    eliminate(device, color);
    expect(device, color, [](std::uint32_t x, std::uint32_t y) { return (x / 8u + y / 8u) % 2u == 0 ? Clear : Stale; }, name + " checkerboard of fast-cleared tiles eliminated");
    prepare(color, [](std::uint32_t tileX, std::uint32_t tileY) { return tileX >= 3 && tileY == 1; });
    draw(device, color, {{0, 0}, {Width, 4}});
    expect(device, color, [](std::uint32_t x, std::uint32_t y) { return y < 4 ? Drawn : y >= 8 && x >= 24 ? Clear : Stale; }, name + " fast-cleared bottom edge tiles under a draw of the top rows");
}

// Upstream 0f2683cb7 (CmaskClear.comp, Draw.cpp): a CMASK code other than 0 (cleared) or 15 (expanded) keeps its
// texels like an expanded tile instead of failing the pass; the Metal eliminate kernel (TextureDetiler.mm) and the
// draw-time materialization (MetalDraw.mm MaterializeCmask) already skip such tiles.
void unknownCode(AgcDriver::VulkanDevice& device) {
    const auto color = target(ColorTileMode::RenderTarget);
    prepare(color, [](std::uint32_t tileX, std::uint32_t tileY) { return tileX == 2 && tileY == 0; }, 3);
    eliminate(device, color);
    expect(device, color, [](std::uint32_t, std::uint32_t) { return Stale; }, "a tile with CMASK code 3 eliminated");
    prepare(color, [](std::uint32_t, std::uint32_t) { return true; });
    eliminate(device, color);
    expect(device, color, [](std::uint32_t, std::uint32_t) { return Clear; }, "fast-cleared target eliminated after a CMASK code 3");
}

// Metal-only: CB metadata passes over DCC color targets (ASTRO BOT's first frame stopped at "CB metadata passes
// over DCC-compressed targets"). As upstream RunColorMetadataPass: uncompressed keys leave the texels, clear keys
// store the clear texel, comp-to-single keys (a DCC decompress) each 256-byte block's first texel, and the keys are
// left uncompressed; a fast-clear eliminate over comp-to-single keys is refused.
ColorTarget dccTarget(ColorTileMode mode) {
    auto color = target(mode);
    color.cmaskAddress = 0;
    color.cmaskBytes = 0;
    color.dccAddress = reinterpret_cast<std::uintptr_t>(Dcc.data());
    return color;
}

void prepareDcc(const ColorTarget& color, std::uint8_t key, std::uint32_t blockFirst = Stale) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    StorageTexture::FlushPending(color.address, Surface.size(), nullptr, "test");
    std::vector<std::byte> texels(Surface.size());
    for (std::size_t offset = 0; offset < texels.size(); offset += 4) std::memcpy(texels.data() + offset, offset % 256u == 0 ? &blockFirst : &Stale, 4);
    AgcDriver::GuestMemory::Write(color.address, texels);
    std::vector<std::byte> keys(Dcc.size(), std::byte{0xff});
    std::fill_n(keys.begin(), DccKeyBytes(color.bytes), std::byte{key});
    AgcDriver::GuestMemory::Write(color.dccAddress, keys);
}

void passDcc(AgcDriver::VulkanDevice& device, const ColorTarget& color, ColorMetadataPass::Mode mode) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    device.ColorMetadataPass({mode, {color}});
}

void expectDcc(AgcDriver::VulkanDevice& device, const ColorTarget& color, std::uint32_t texel, const std::string& what) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    StorageTexture::FlushPending(color.address, color.bytes, nullptr, "test");
    device.WaitIdle();
    const ColorTargetLayout layout(Width, Height, color.tileMode, 4);
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            std::uint32_t stored = 0;
            std::memcpy(&stored, Surface.data() + layout.Offset(x, y), 4);
            Require(stored == texel, what + ": texel (" + std::to_string(x) + ", " + std::to_string(y) + ") holds " + std::to_string(stored));
        }
    }
    Require(CurrentDccKeys(color.dccAddress, color.bytes) == DccKeys::Uncompressed, what + ": the DCC keys were not left uncompressed");
}

void dccPasses(AgcDriver::VulkanDevice& device) {
    for (const auto mode : {ColorTileMode::Linear, ColorTileMode::RenderTarget}) {
        const auto color = dccTarget(mode);
        const std::string name = mode == ColorTileMode::Linear ? "linear" : "SW_64KB_R_X";
        prepareDcc(color, 0xff);
        passDcc(device, color, ColorMetadataPass::Mode::EliminateFastClear);
        expectDcc(device, color, Stale, name + " DCC target with uncompressed keys after a fast-clear eliminate");
        prepareDcc(color, 0x20);
        passDcc(device, color, ColorMetadataPass::Mode::EliminateFastClear);
        expectDcc(device, color, Clear, name + " DCC register clear after a fast-clear eliminate");
        prepareDcc(color, 0x00);
        passDcc(device, color, ColorMetadataPass::Mode::DccDecompress);
        expectDcc(device, color, 0u, name + " DCC 0000 clear after a DCC decompress");
    }
    const auto color = dccTarget(ColorTileMode::RenderTarget);
    prepareDcc(color, 0x10, Clear);
    passDcc(device, color, ColorMetadataPass::Mode::DccDecompress);
    expectDcc(device, color, Clear, "SW_64KB_R_X comp-to-single DCC keys after a DCC decompress");
    prepareDcc(color, 0x10, Clear);
    bool refused = false;
    try {
        passDcc(device, color, ColorMetadataPass::Mode::EliminateFastClear);
    } catch (const std::exception&) {
        refused = true;
    }
    Require(refused, "a fast-clear eliminate over comp-to-single DCC keys was not refused");
}

}

int main() {
    try {
        std::unique_ptr<AgcDriver::VulkanDevice> device;
        try {
            device = std::make_unique<AgcDriver::VulkanDevice>();
        } catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::printf("skipped, no usable Metal device: %s\n", error.what());
            return VulkanTestSkipped;
        }
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(SurfaceStorage.data(), SurfaceStorage.size(), true, true, true);
            mutation.Add(Cmask.data(), Cmask.size(), true, true, true);
            mutation.Add(Dcc.data(), Dcc.size(), true, true, true);
        }
        run(*device, ColorTileMode::Linear, "linear");
        run(*device, ColorTileMode::RenderTarget, "SW_64KB_R_X");
        unknownCode(*device);
        dccPasses(*device);
        device->WaitIdle();
        device.reset();
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(SurfaceStorage.data());
            mutation.Remove(Cmask.data());
            mutation.Remove(Dcc.data());
        }
        std::cout << "CMASK fast clear draw and eliminate tests, DCC metadata passes passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
