// Multisampled Metal draws and CB resolves (DECISIONS D27, M2_DESIGN "MSAA results"): MetalDraw.mm
// renders 4x color targets into their MultisampleTarget host image and 4x depth into a multisampled
// surface, at the guest's sample positions; VulkanDevice::ColorMetadataPass resolves a 4x target into
// a tiled single-sample one (the image cache's resident image), which is read back through guest
// memory. Hand-written vertex and fragment functions against the MslAbi contract, as D6DrawTests.mm.
//   agc_metal_msaa_tests edges     (agc_metal_msaa_edges: per-pixel coverage of a diagonal triangle at
//                                   the standard 4x positions, a vertical edge at custom positions)
//   agc_metal_msaa_tests masks     (agc_metal_msaa_masks: an exported sample mask, the B6-emulated
//                                   alpha-to-coverage mask with Metal's own A2C kept off, per-sample
//                                   shading through [[sample_id]])
//   agc_metal_msaa_tests depth     (agc_metal_msaa_depth: a 4x depth surface, depth test per sample)
//   agc_metal_msaa_tests integer   (agc_metal_msaa_integer: an R32_UINT resolve takes sample 0)
//   agc_metal_msaa_tests sampled   (agc_metal_msaa_sampled: a guest image_load through a 2D MSAA T#
//                                   reads each sample of a rendered 4x target)
#include "../src/MetalDeviceState.h"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int Skipped = 77;
constexpr std::uint32_t Samples = 4u;
namespace Abi = ShaderRecompiler::MslAbi;
namespace RuntimeAbi = ShaderRecompiler::RuntimeAbi;
using AgcDriver::Graphics::ColorTileMode;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class GuestBlock {
public:
    explicit GuestBlock(std::size_t bytes) : size(bytes) {
        block = static_cast<std::byte*>(std::aligned_alloc(16384, (bytes + 16383u) & ~std::size_t{16383u}));
        Require(block != nullptr, "cannot allocate a guest block");
        std::memset(block, 0, bytes);
        GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    }
    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
        std::free(block);
    }
    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;
    std::byte* Data() const { return block; }
    std::uint64_t Address() const { return reinterpret_cast<std::uint64_t>(block); }
    std::size_t Size() const { return size; }

private:
    std::size_t size;
    std::byte* block = nullptr;
};

constexpr std::uint32_t Extent = 32u;

std::size_t DestinationBytes() {
    return AgcDriver::Graphics::ColorTargetLayout(Extent, Extent, ColorTileMode::RenderTarget, 4u).Bytes();
}

// Location 0 (float4 x, y, z, w) pulled through the VertexFetch slot.
const char* VertexSource = R"(#include <metal_stdlib>
using namespace metal;
struct Fetch { ulong address; uint stride; uint bytes; uint word3; uint rate; uint components; uint reserved; };
struct VertexOut { float4 position [[position]]; float4 p0 [[user(locn0)]]; };
vertex VertexOut aps5_main(uint vertexId [[vertex_id]], device const Fetch* fetch [[buffer(4)]]) {
    device const float4* data = reinterpret_cast<device const float4*>(fetch[0].address + ulong(vertexId) * fetch[0].stride);
    VertexOut out;
    out.position = float4(data->x, data->y, data->z, 1.0f);
    out.p0 = float4(1.0f);
    return out;
}
)";

// Push dword 32 is the RGBA8 color; dword 33 the sample mask (MASK), alpha >= 0.5 decides an
// all-or-nothing mask (A2C, B6's emulation), sample 0 alone is lit (PER_SAMPLE).
std::string FragmentSource(const std::string& defines) {
    return "#include <metal_stdlib>\nusing namespace metal;\n" + defines + R"(
struct FragmentIn { float4 p0 [[user(locn0)]]; };
struct FragmentOut {
#ifdef UINT_TARGET
    uint4 color [[color(0)]];
#else
    float4 color [[color(0)]];
#endif
#if defined(MASK) || defined(A2C)
    uint mask [[sample_mask]];
#endif
};
fragment FragmentOut aps5_main(FragmentIn in [[stage_in]], constant uint* push [[buffer(1)]]
#if defined(PER_SAMPLE) || defined(SAMPLE_COLOR)
    , uint sampleId [[sample_id]]
#endif
) {
    FragmentOut out;
#ifdef UINT_TARGET
    out.color = uint4(push[32], 0u, 0u, 0u);
#else
    out.color = unpack_unorm4x8_to_float(push[32]) * in.p0;
#endif
#ifdef MASK
    out.mask = push[33];
#endif
#ifdef A2C
    out.mask = !(out.color.w >= 0.5f) ? 0u : 0xffffffffu;
#endif
#ifdef PER_SAMPLE
    if (sampleId != 0u) out.color = float4(0.0f, 0.0f, 0.0f, out.color.w);
#endif
#ifdef SAMPLE_COLOR
    out.color = float4((float(sampleId) * 64.0f + 32.0f) / 255.0f, 0.0f, 0.0f, 1.0f);
#endif
    return out;
}
)";
}

struct Shaders {
    ShaderRecompiler::RecompileResult vertex;
    ShaderRecompiler::RecompileResult fragment;
};

Shaders MakeShaders(std::uint64_t vertices, std::uint32_t records, std::uint32_t color, const std::string& defines = "", std::uint32_t mask = ~0u) {
    Shaders shaders;
    Abi::Reflection vertex;
    vertex.stage = Abi::Stage::Vertex;
    vertex.group = RuntimeAbi::Stage::Main;
    vertex.outputLocations = 1u;
    vertex.vertexFetchLocations = 1u;
    shaders.vertex.msl = ShaderRecompiler::SharedMsl(ShaderRecompiler::MslShader{VertexSource, vertex});
    ShaderRecompiler::VertexAttribute attribute {};
    attribute.location = 0u;
    attribute.components = 4u;
    attribute.resource.fields = {static_cast<std::uint32_t>(vertices), static_cast<std::uint32_t>(vertices >> 32u) | (16u << 16u), records, 0x00000facu | (77u << 12u)};
    attribute.formatComponents = 4u;
    shaders.vertex.vertexAttributes.push_back(attribute);
    Abi::Reflection fragment;
    fragment.stage = Abi::Stage::Fragment;
    fragment.group = RuntimeAbi::Stage::Fragment;
    fragment.colorOutputs = 1u;
    fragment.usesPushData = true;
    fragment.exportsSampleMask = defines.find("MASK") != std::string::npos || defines.find("A2C") != std::string::npos;
    fragment.perSample = defines.find("PER_SAMPLE") != std::string::npos || defines.find("SAMPLE_COLOR") != std::string::npos;
    shaders.fragment.msl = ShaderRecompiler::SharedMsl(ShaderRecompiler::MslShader{FragmentSource(defines), fragment});
    shaders.fragment.pushConstants.resize(8u);
    std::memcpy(shaders.fragment.pushConstants.data(), &color, 4u);
    std::memcpy(shaders.fragment.pushConstants.data() + 4u, &mask, 4u);
    return shaders;
}

AgcDriver::Graphics::ColorTarget Target(std::uint64_t address, VkFormat format, std::uint32_t samples) {
    const AgcDriver::Graphics::ColorTargetLayout layout(Extent, Extent, ColorTileMode::RenderTarget, 4u);
    AgcDriver::Graphics::ColorTarget target {address, {Extent, Extent}, format, layout.Bytes(), 0xe4u};
    target.tileMode = ColorTileMode::RenderTarget;
    target.elementBytes = 4u;
    target.surfaceAddress = address;
    target.surfaceExtent = {Extent, Extent};
    target.samples = samples;
    return target;
}

AgcDriver::Graphics::State BaseState(const AgcDriver::Graphics::ColorTarget& target) {
    AgcDriver::Graphics::State state {};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    state.renderExtent = {Extent, Extent};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0.0f, static_cast<float>(Extent), static_cast<float>(Extent), -static_cast<float>(Extent), 0.0f, 1.0f};
    state.scissor = {{0, 0}, {Extent, Extent}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.negativeOneToOne = false;
    state.blend = {};
    state.blend.colorWriteMask = 15u;
    state.blends = {state.blend};
    state.blendConstants = {};
    state.hasColorTarget = true;
    state.samples = target.samples;
    state.color = target;
    state.colors = {target};
    return state;
}

void Draw(AgcDriver::VulkanDevice& device, const AgcDriver::Graphics::State& state, const Shaders& shaders, std::uint32_t first, std::uint32_t count) {
    const std::array<AgcDriver::Graphics::CompiledShader, 2> compiled {{{ShaderRecompiler::ShaderStage::Vertex, &shaders.vertex, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &shaders.fragment, 128u}}};
    AgcDriver::Pm4::DrawParameters draw {0u, count, 0u, 1u, 0u, false};
    draw.firstVertex = first;
    device.Draw(state, draw, compiled);
}

// The CB resolve of `source` into the tiled single-sample `destination`, read back as linear texels.
std::vector<std::uint32_t> Resolve(AgcDriver::VulkanDevice& device, const AgcDriver::Graphics::ColorTarget& source, GuestBlock& destination) {
    AgcDriver::Graphics::ColorMetadataPass pass {AgcDriver::Graphics::ColorMetadataPass::Mode::Resolve, {Target(destination.Address(), source.format, 1u)}, source};
    device.ColorMetadataPass(pass);
    device.WaitIdle();
    const AgcDriver::Graphics::ColorTargetLayout layout(Extent, Extent, ColorTileMode::RenderTarget, 4u);
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::Graphics::StorageTexture::FlushPending(destination.Address(), layout.Bytes(), nullptr, "test");
        device.WaitIdle();
    }
    std::vector<std::uint32_t> texels(Extent * Extent);
    layout.Detile(std::span<const std::byte>(destination.Data(), layout.Bytes()), std::as_writable_bytes(std::span(texels)));
    return texels;
}

void WriteVertices(GuestBlock& block, std::span<const std::array<float, 4>> vertices) {
    std::memcpy(block.Data(), vertices.data(), vertices.size_bytes());
}

// Standard 4x positions (PA_SC_AA_SAMPLE_LOCS defaults, 1/16 pixel from the center).
constexpr std::array<std::array<int, 2>, 4> Standard4 {{{-2, -6}, {6, -2}, {-6, 2}, {2, 6}}};

// Framebuffer coordinates of an NDC position under the test viewport (y down).
std::array<double, 2> Framebuffer(const std::array<float, 4>& v) {
    return {(v[0] + 1.0) * Extent / 2.0, Extent - (v[1] + 1.0) * Extent / 2.0};
}

double EdgeFunction(const std::array<double, 2>& a, const std::array<double, 2>& b, double x, double y) {
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
}

std::uint32_t Channel(std::uint32_t texel, std::uint32_t channel) {
    return (texel >> (channel * 8u)) & 0xffu;
}

bool Near(std::uint32_t actual, double expected) {
    return std::fabs(static_cast<double>(actual) - expected) <= 1.0;
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

void CheckEdges(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    GuestBlock source(Extent * Extent * 4u * Samples);
    GuestBlock destination(DestinationBytes());
    // A diagonal triangle: every pixel's resolved red is 255 * (covered samples) / 4.
    constexpr std::array<std::array<float, 4>, 3> triangle {{{-0.83f, -0.91f, 0.5f, 0.0f}, {0.87f, -0.29f, 0.5f, 0.0f}, {-0.41f, 0.93f, 0.5f, 0.0f}}};
    WriteVertices(vertices, triangle);
    const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
    const auto state = BaseState(target);
    Draw(device, state, MakeShaders(vertices.Address(), 3u, 0xff0000ffu), 0u, 3u);
    const auto texels = Resolve(device, target, destination);
    const auto a = Framebuffer(triangle[0]);
    const auto b = Framebuffer(triangle[1]);
    const auto c = Framebuffer(triangle[2]);
    const double area = EdgeFunction(a, b, c[0], c[1]);
    std::uint32_t partial = 0;
    std::uint32_t checked = 0;
    for (std::uint32_t y = 0; y < Extent; ++y) {
        for (std::uint32_t x = 0; x < Extent; ++x) {
            std::uint32_t covered = 0;
            bool ambiguous = false;
            for (const auto& offset : Standard4) {
                const double sx = x + 0.5 + offset[0] / 16.0;
                const double sy = y + 0.5 + offset[1] / 16.0;
                const std::array<double, 3> e {EdgeFunction(a, b, sx, sy) / area, EdgeFunction(b, c, sx, sy) / area, EdgeFunction(c, a, sx, sy) / area};
                for (const auto value : e) ambiguous |= std::fabs(value) < 1e-4;
                covered += e[0] > 0.0 && e[1] > 0.0 && e[2] > 0.0 ? 1u : 0u;
            }
            if (ambiguous) continue;
            ++checked;
            if (covered != 0u && covered != Samples) ++partial;
            const auto texel = texels[y * Extent + x];
            const double expected = 255.0 * covered / Samples;
            Require(Near(Channel(texel, 0), expected) && Channel(texel, 1) == 0u && Near(Channel(texel, 3), expected), "pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") resolved to " + Hex(texel) + ", expected red/alpha " + std::to_string(expected) + " (" + std::to_string(covered) + " of 4 samples)");
        }
    }
    Require(partial >= 20u, "the triangle's edges cover only " + std::to_string(partial) + " partially covered pixels");

    // Custom positions (PA_SC_AA_SAMPLE_LOCS): all four samples at x - 6/16. A rectangle whose right
    // edge is at x = 10.25 covers pixel 10's samples at 10.125 only: 1 of 4 standard, 4 of 4 custom.
    GuestBlock standardSource(Extent * Extent * 4u * Samples);
    GuestBlock customSource(Extent * Extent * 4u * Samples);
    const float right = 10.25f / Extent * 2.0f - 1.0f;
    const std::array<std::array<float, 4>, 6> rectangle {{{-1.0f, -1.0f, 0.5f, 0.0f}, {right, -1.0f, 0.5f, 0.0f}, {-1.0f, 1.0f, 0.5f, 0.0f}, {right, -1.0f, 0.5f, 0.0f}, {right, 1.0f, 0.5f, 0.0f}, {-1.0f, 1.0f, 0.5f, 0.0f}}};
    WriteVertices(vertices, rectangle);
    for (const bool custom : {false, true}) {
        const auto rectTarget = Target(custom ? customSource.Address() : standardSource.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
        auto rectState = BaseState(rectTarget);
        if (custom) {
            rectState.customSampleLocations = true;
            for (std::uint32_t sample = 0; sample < Samples; ++sample) {
                const auto y = static_cast<std::uint8_t>(Standard4[sample][1] & 0xf);
                rectState.sampleLocations[sample] = static_cast<std::uint8_t>((static_cast<std::uint8_t>(-6) & 0xfu) | (y << 4u));
            }
        }
        Draw(device, rectState, MakeShaders(vertices.Address(), 6u, custom ? 0xff00ff00u : 0xffff0000u), 0u, 6u);
        const auto rect = Resolve(device, rectTarget, destination);
        const auto channel = custom ? 1u : 2u;
        const double expected = custom ? 255.0 : 255.0 * 0.25;
        const auto edge = rect[5u * Extent + 10u];
        Require(Near(Channel(edge, channel), expected) && Channel(rect[5u * Extent + 9u], channel) == 255u && Channel(rect[5u * Extent + 11u], channel) == 0u, std::string(custom ? "custom" : "standard") + " sample positions: pixels 9/10/11 of the edge resolved to " + Hex(rect[5u * Extent + 9u]) + " " + Hex(edge) + " " + Hex(rect[5u * Extent + 11u]));
    }
    std::printf("edges: %u pixels checked (%u partially covered), custom positions passed\n", checked, partial);
}

constexpr std::array<std::array<float, 4>, 3> FullScreen {{{-1.0f, -1.0f, 0.5f, 0.0f}, {3.0f, -1.0f, 0.5f, 0.0f}, {-1.0f, 3.0f, 0.5f, 0.0f}}};

void RequireUniform(const std::vector<std::uint32_t>& texels, std::uint32_t channel, double expected, const std::string& what) {
    for (std::uint32_t index = 0; index < texels.size(); ++index) {
        Require(Near(Channel(texels[index], channel), expected), what + ": pixel " + std::to_string(index % Extent) + ", " + std::to_string(index / Extent) + " resolved to " + Hex(texels[index]) + ", expected channel " + std::to_string(channel) + " = " + std::to_string(expected));
    }
}

void CheckMasks(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    GuestBlock destination(DestinationBytes());
    // An exported sample mask (samples 0 and 2): half of every pixel.
    {
        GuestBlock source(Extent * Extent * 4u * Samples);
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0xff0000ffu, "#define MASK 1\n", 0x5u), 0u, 3u);
        RequireUniform(Resolve(device, target, destination), 0u, 127.5, "sample mask 0x5");
    }
    // Alpha-to-coverage as B6 emulates it (the guest's undithered rule: all samples at alpha >= 0.5):
    // the driver keeps Metal's own A2C off, which would cover 3 of 4 samples at alpha 0.75.
    for (const auto& [alpha, expected] : std::array<std::pair<std::uint32_t, double>, 2>{{{0xbfu, 255.0}, {0x40u, 0.0}}}) {
        GuestBlock source(Extent * Extent * 4u * Samples);
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
        auto state = BaseState(target);
        state.alphaToCoverage = true;
        Draw(device, state, MakeShaders(vertices.Address(), 3u, (alpha << 24u) | 0xffu, "#define A2C 1\n"), 0u, 3u);
        RequireUniform(Resolve(device, target, destination), 0u, expected, "alpha-to-coverage at alpha " + std::to_string(alpha));
    }
    // Per-sample shading: only sample 0 is red.
    {
        GuestBlock source(Extent * Extent * 4u * Samples);
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0xff0000ffu, "#define PER_SAMPLE 1\n"), 0u, 3u);
        const auto texels = Resolve(device, target, destination);
        RequireUniform(texels, 0u, 63.75, "per-sample shading");
        RequireUniform(texels, 3u, 255.0, "per-sample shading alpha");
    }
    std::puts("masks: exported sample mask, emulated alpha-to-coverage, per-sample shading passed");
}

void CheckDepth(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    GuestBlock source(Extent * Extent * 4u * Samples);
    GuestBlock destination(DestinationBytes());
    // A near rectangle (z 0.25) whose right edge is at x = 10.25 (pixel 10: 1 of 4 samples), then a
    // full-screen far one (z 0.5) that the LESS test keeps off the near samples.
    const float right = 10.25f / Extent * 2.0f - 1.0f;
    const std::array<std::array<float, 4>, 9> geometry {{{-1.0f, -1.0f, 0.25f, 0.0f}, {right, -1.0f, 0.25f, 0.0f}, {-1.0f, 1.0f, 0.25f, 0.0f}, {right, -1.0f, 0.25f, 0.0f}, {right, 1.0f, 0.25f, 0.0f}, {-1.0f, 1.0f, 0.25f, 0.0f}, {-1.0f, -1.0f, 0.5f, 0.0f}, {3.0f, -1.0f, 0.5f, 0.0f}, {-1.0f, 3.0f, 0.5f, 0.0f}}};
    WriteVertices(vertices, geometry);
    const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
    auto state = BaseState(target);
    std::vector<float> depthMemory(Extent * Extent * Samples, 0.0f);
    state.depth = AgcDriver::Graphics::DepthTarget {reinterpret_cast<std::uint64_t>(depthMemory.data()), 0u, {Extent, Extent}, VK_FORMAT_D32_SFLOAT, 1.0f, 0u};
    state.depth->samples = Samples;
    state.depthTest = true;
    state.depthWrite = true;
    state.depthCompare = VK_COMPARE_OP_LESS;
    Draw(device, state, MakeShaders(vertices.Address(), 9u, 0xff0000ffu), 0u, 6u);
    Draw(device, state, MakeShaders(vertices.Address(), 9u, 0xff00ff00u), 6u, 3u);
    const auto texels = Resolve(device, target, destination);
    const auto at = [&](std::uint32_t x) { return texels[7u * Extent + x]; };
    Require(Channel(at(9u), 0u) == 255u && Channel(at(9u), 1u) == 0u, "the near rectangle lost pixel 9 to the far draw: " + Hex(at(9u)));
    Require(Near(Channel(at(10u), 0u), 63.75) && Near(Channel(at(10u), 1u), 191.25), "the edge pixel's samples were not depth-tested one by one: " + Hex(at(10u)));
    Require(Channel(at(11u), 0u) == 0u && Channel(at(11u), 1u) == 255u, "the far draw did not cover pixel 11: " + Hex(at(11u)));
    std::puts("depth: 4x depth surface, per-sample depth test passed");
}

void CheckInteger(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    GuestBlock destination(DestinationBytes());
    // Samples 1..3 only: the sample-0 resolve sees the cleared 0; sample 0 alone: the value.
    for (const auto& [mask, expected] : std::array<std::pair<std::uint32_t, std::uint32_t>, 2>{{{0xeu, 0u}, {0x1u, 0x12345678u}}}) {
        GuestBlock source(Extent * Extent * 4u * Samples);
        const auto target = Target(source.Address(), VK_FORMAT_R32_UINT, Samples);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0x12345678u, "#define UINT_TARGET 1\n#define MASK 1\n", mask), 0u, 3u);
        const auto texels = Resolve(device, target, destination);
        for (std::uint32_t index = 0; index < texels.size(); ++index) Require(texels[index] == expected, "R32_UINT resolve with sample mask " + std::to_string(mask) + ": pixel " + std::to_string(index) + " is " + Hex(texels[index]) + ", expected " + Hex(expected));
    }
    std::puts("integer: R32_UINT resolve takes sample 0 passed");
}

// A guest compute program reads a 4x target through a 2D MSAA T# (type 14, DECISIONS D27):
//   v_and_b32 v6, 3, v0; v_lshrrev_b32 v4, 2, v0; v_mov_b32 v5, 5
//   image_load v[8:11], v[4:6], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm; s_waitcnt vmcnt(0)
//   v_lshlrev_b32 v1, 4, v0; buffer_store_dwordx4 v[8:11], v1, s[0:3], 0 offen; s_endpgm
// (llvm-mc -arch=amdgcn -mcpu=gfx1030): thread t reads sample t & 3 of pixel (t >> 2, 5).
alignas(256) constexpr std::array<std::uint32_t, 10> MsaaLoadCode {0x360c0083u, 0x2c080082u, 0x7e0a0285u, 0xf0001f30u, 0x00010804u, 0xbf8c3f70u, 0x34020084u, 0xe0781000u, 0x80000801u, 0xbf810000u};
alignas(256) std::array<float, 32 * 4> MsaaLoadResults {};

void CheckSampled(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    GuestBlock source(Extent * Extent * 4u * Samples);
    const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
    // Sample s of every pixel is red (64 s + 32) / 255.
    Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0u, "#define SAMPLE_COLOR 1\n"), 0u, 3u);
    MsaaLoadResults.fill(-1.0f);
    std::vector<std::uint32_t> userData(16, 0u);
    const auto results = reinterpret_cast<std::uint64_t>(MsaaLoadResults.data());
    const std::array<std::uint32_t, 4> buffer {static_cast<std::uint32_t>(results), static_cast<std::uint32_t>((results >> 32u) & 0xffffu), static_cast<std::uint32_t>(sizeof(MsaaLoadResults)), 0x31016facu};
    // T#: 8_8_8_8_UNORM (56), 32x32, identity swizzle, LAST_LEVEL = log2(samples) = 2, type 14, MAX_MIP 2.
    const auto address = source.Address();
    const std::array<std::uint32_t, 8> texture {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (((Extent - 1u) & 3u) << 30u), ((Extent - 1u) >> 2u) | ((Extent - 1u) << 14u), 0xfacu | (2u << 16u) | (14u << 28u), 0u, 2u << 4u, 0u, 0u};
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(MsaaLoadCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory {{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute {{32u, 1u, 1u}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request {
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}};
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    for (std::uint32_t thread = 0; thread < 32u; ++thread) {
        const auto sample = thread & 3u;
        const float expected = static_cast<float>(sample * 64u + 32u) / 255.0f;
        const auto* texel = MsaaLoadResults.data() + thread * 4u;
        Require(std::fabs(texel[0] - expected) < 1e-6f && texel[1] == 0.0f && texel[2] == 0.0f && texel[3] == 1.0f, "image_load of sample " + std::to_string(sample) + " at pixel (" + std::to_string(thread >> 2u) + ", 5) read " + std::to_string(texel[0]) + " " + std::to_string(texel[1]) + " " + std::to_string(texel[2]) + " " + std::to_string(texel[3]) + ", expected red " + std::to_string(expected));
    }
    // Memory that is no multisampled target has no modeled sample layout: refused.
    GuestBlock plain(Extent * Extent * 4u * Samples);
    const auto other = plain.Address();
    userData[4] = static_cast<std::uint32_t>(other >> 8u);
    userData[5] = static_cast<std::uint32_t>((other >> 40u) & 0xffu) | (56u << 20u) | (((Extent - 1u) & 3u) << 30u);
    std::string refusal;
    try {
        device.Dispatch(ShaderRecompiler::Recompile(request), 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
        device.WaitIdle();
    } catch (const std::exception& error) {
        refusal = error.what();
    }
    Require(refusal.find("no multisampled color target") != std::string::npos, "an MSAA T# over plain memory was not refused: " + refusal);
    std::puts("sampled: image_load of each sample through a 2D MSAA T#, refusal over plain memory passed");
}

}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "edges";
    std::unique_ptr<AgcDriver::VulkanDevice> device;
    try {
        device = std::make_unique<AgcDriver::VulkanDevice>();
    } catch (const std::exception& error) {
        std::printf("skipped, no usable Metal device: %s\n", error.what());
        return Skipped;
    }
    try {
        if (mode == "edges") CheckEdges(*device);
        else if (mode == "masks") CheckMasks(*device);
        else if (mode == "depth") CheckDepth(*device);
        else if (mode == "integer") CheckInteger(*device);
        else if (mode == "sampled") CheckSampled(*device);
        else {
            std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 2;
        }
        device->WaitIdle();
        std::printf("metal msaa %s tests passed\n", mode.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        try {
            if (device != nullptr) device->WaitIdle();
        } catch (const std::exception& idle) {
            std::fprintf(stderr, "%s\n", idle.what());
        }
        return 1;
    }
}
