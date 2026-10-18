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
// MSAA2 (DECISIONS D28, M2_DESIGN "MSAA2 results"):
//   agc_metal_msaa_tests emulated  (8x as two 4x slices: coverage, slice masks, per-sample shading,
//                                   integer resolve, depth, pass sharing; 4x == 2 x 2x emulation)
//   agc_metal_msaa_tests sampled8  (8x type 14 and 4x/8x type 15 T# reads, image_get_resinfo)
//   agc_metal_msaa_tests interp    (recompiled PERSP_SAMPLE vs PERSP_CENTER pixel programs, per sample)
//   agc_metal_msaa_tests interp_locations (DECISIONS D38: one input read at sample, center and
//                                   centroid through interpolant<>, against CPU positions)
//   agc_metal_msaa_tests a2c       (recompiled alpha-to-coverage: the measured proportional rule)
//   agc_metal_msaa_tests storage   (guest image_store / image_atomic_add on rendered 4x/8x targets)
//   agc_metal_msaa_tests ranges    (a build binding an MSAA T# before single-sample T#s: each surface
//                                   range keeps its own texture, so the pre-sync list is right)
#include "../src/MetalDeviceState.h"
#include "../src/MetalImages.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/tests/execution/VulkanTestDevice.hpp"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
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

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    // The image cache's Context, the one the device's dispatches build their image bindings with
    // (MetalTextures.mm ImageBindings::Prepare).
    static const Graphics::Context& ImageContextOf(const VulkanDevice& device) {
        const auto* context = device.state->images != nullptr ? device.state->images->ImageContext() : nullptr;
        if (context == nullptr) throw std::runtime_error("the device's image cache is not attached");
        return *context;
    }
};

}

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

// Location 0 (float4 x, y, z, w) pulled through the VertexFetch slot; p0 is 1, or with P0_X the
// framebuffer x / 32 (0 at the left edge, 1 at the right edge of the 32-pixel targets).
const char* VertexSource = R"(#include <metal_stdlib>
using namespace metal;
struct Fetch { ulong address; uint stride; uint bytes; uint word3; uint rate; uint components; uint reserved; };
struct VertexOut { float4 position [[position]]; float4 p0 [[user(locn0)]]; };
vertex VertexOut aps5_main(uint vertexId [[vertex_id]], device const Fetch* fetch [[buffer(4)]]) {
    device const float4* data = reinterpret_cast<device const float4*>(fetch[0].address + ulong(vertexId) * fetch[0].stride);
    VertexOut out;
    out.position = float4(data->x, data->y, data->z, 1.0f);
#ifdef P0_X
    out.p0 = float4((data->x + 1.0f) * 0.5f);
#else
    out.p0 = float4(1.0f);
#endif
    return out;
}
)";

// Push dword 32 is the RGBA8 color; dword 33 the sample mask (MASK), alpha >= 0.5 decides an
// all-or-nothing mask (A2C, B6's emulation), sample 0 alone is lit (PER_SAMPLE). As the emitter's
// functions (MslAbi Version 9), sample ids add APS5_SAMPLE_BASE and the exported mask is shifted
// right by it (the slices of an emulated 8x target, DECISIONS D28).
std::string FragmentSource(const std::string& defines) {
    return "#include <metal_stdlib>\nusing namespace metal;\n#ifndef APS5_SAMPLE_BASE\n#define APS5_SAMPLE_BASE 0\n#endif\n" + defines + R"(
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
    out.mask = push[33] >> APS5_SAMPLE_BASE;
#endif
#ifdef A2C
    out.mask = !(out.color.w >= 0.5f) ? 0u : 0xffffffffu;
#endif
#ifdef PER_SAMPLE
    if (sampleId + APS5_SAMPLE_BASE != 0u) out.color = float4(0.0f, 0.0f, 0.0f, out.color.w);
#endif
#ifdef SAMPLE_COLOR
    out.color = float4((float(sampleId + APS5_SAMPLE_BASE) * 32.0f + 16.0f) / 255.0f, 0.0f, 0.0f, 1.0f);
#endif
    return out;
}
)";
}

struct Shaders {
    ShaderRecompiler::RecompileResult vertex;
    ShaderRecompiler::RecompileResult fragment;
};

Shaders MakeShaders(std::uint64_t vertices, std::uint32_t records, std::uint32_t color, const std::string& defines = "", std::uint32_t mask = ~0u, bool p0x = false) {
    Shaders shaders;
    Abi::Reflection vertex;
    vertex.stage = Abi::Stage::Vertex;
    vertex.group = RuntimeAbi::Stage::Main;
    vertex.outputLocations = 1u;
    vertex.vertexFetchLocations = 1u;
    shaders.vertex.msl = ShaderRecompiler::SharedMsl(ShaderRecompiler::MslShader{std::string(p0x ? "#define P0_X 1\n" : "") + VertexSource, vertex});
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

// Multisampled targets live for the device's lifetime, keyed by address (upstream's model): every
// target of a check keeps its own live guest block, so no later one reuses an earlier one's samples.
// Cleared before the check returns (static destruction would outlive GuestAllocations).
std::vector<std::unique_ptr<GuestBlock>> sources;

void CheckMasks(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    GuestBlock destination(DestinationBytes());
    // An exported sample mask (samples 0 and 2): half of every pixel.
    {
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * Samples));
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0xff0000ffu, "#define MASK 1\n", 0x5u), 0u, 3u);
        RequireUniform(Resolve(device, target, destination), 0u, 127.5, "sample mask 0x5");
    }
    // Alpha-to-coverage as B6 emulates it (the guest's undithered rule: all samples at alpha >= 0.5):
    // the driver keeps Metal's own A2C off, which would cover 3 of 4 samples at alpha 0.75.
    for (const auto& [alpha, expected] : std::array<std::pair<std::uint32_t, double>, 2>{{{0xbfu, 255.0}, {0x40u, 0.0}}}) {
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * Samples));
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
        auto state = BaseState(target);
        state.alphaToCoverage = true;
        Draw(device, state, MakeShaders(vertices.Address(), 3u, (alpha << 24u) | 0xffu, "#define A2C 1\n"), 0u, 3u);
        RequireUniform(Resolve(device, target, destination), 0u, expected, "alpha-to-coverage at alpha " + std::to_string(alpha));
    }
    // Per-sample shading: only sample 0 is red.
    {
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * Samples));
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0xff0000ffu, "#define PER_SAMPLE 1\n"), 0u, 3u);
        const auto texels = Resolve(device, target, destination);
        RequireUniform(texels, 0u, 63.75, "per-sample shading");
        RequireUniform(texels, 3u, 255.0, "per-sample shading alpha");
    }
    sources.clear();
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
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * Samples));
        const auto target = Target(source.Address(), VK_FORMAT_R32_UINT, Samples);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0x12345678u, "#define UINT_TARGET 1\n#define MASK 1\n", mask), 0u, 3u);
        const auto texels = Resolve(device, target, destination);
        for (std::uint32_t index = 0; index < texels.size(); ++index) Require(texels[index] == expected, "R32_UINT resolve with sample mask " + std::to_string(mask) + ": pixel " + std::to_string(index) + " is " + Hex(texels[index]) + ", expected " + Hex(expected));
    }
    sources.clear();
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
    // Sample s of every pixel is red (32 s + 16) / 255.
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
        const float expected = static_cast<float>(sample * 32u + 16u) / 255.0f;
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


// --- MSAA2 (DECISIONS D28): emulated sample counts, MSAA array T#s, size queries, sample-rate
// interpolation, proportional alpha-to-coverage ---------------------------------------------------

using AgcDriver::Graphics::Metal::EmulatedSampleCounts;

// The guest's sample positions of a state as offsets in [0, 1) from the pixel's top-left corner.
std::vector<std::array<double, 2>> Positions(const AgcDriver::Graphics::State& state) {
    const auto locations = AgcDriver::Graphics::SampleLocations(state);
    std::vector<std::array<double, 2>> positions;
    for (std::uint32_t sample = 0; sample < state.samples; ++sample) positions.push_back({locations[sample].x, locations[sample].y});
    return positions;
}

std::uint32_t CoveredSamples(const std::vector<std::array<double, 2>>& positions, const std::array<std::array<double, 2>, 3>& triangle, double area, std::uint32_t x, std::uint32_t y, bool& ambiguous) {
    std::uint32_t covered = 0;
    for (const auto& position : positions) {
        const double sx = x + position[0];
        const double sy = y + position[1];
        const std::array<double, 3> e {EdgeFunction(triangle[0], triangle[1], sx, sy) / area, EdgeFunction(triangle[1], triangle[2], sx, sy) / area, EdgeFunction(triangle[2], triangle[0], sx, sy) / area};
        for (const auto value : e) ambiguous |= std::fabs(value) < 1e-4;
        covered += e[0] > 0.0 && e[1] > 0.0 && e[2] > 0.0 ? 1u : 0u;
    }
    return covered;
}

constexpr std::array<std::array<float, 4>, 3> Diagonal {{{-0.83f, -0.91f, 0.5f, 0.0f}, {0.87f, -0.29f, 0.5f, 0.0f}, {-0.41f, 0.93f, 0.5f, 0.0f}}};

// Draws the diagonal triangle into a target of `samples` samples and checks every unambiguous pixel
// against the analytic coverage at the state's positions; returns the resolved texels.
std::vector<std::uint32_t> CheckDiagonal(AgcDriver::VulkanDevice& device, std::uint32_t samples, const std::string& what) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, Diagonal);
    auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * samples));
    GuestBlock destination(DestinationBytes());
    const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, samples);
    const auto state = BaseState(target);
    Draw(device, state, MakeShaders(vertices.Address(), 3u, 0xff0000ffu), 0u, 3u);
    const auto texels = Resolve(device, target, destination);
    const std::array<std::array<double, 2>, 3> triangle {Framebuffer(Diagonal[0]), Framebuffer(Diagonal[1]), Framebuffer(Diagonal[2])};
    const double area = EdgeFunction(triangle[0], triangle[1], triangle[2][0], triangle[2][1]);
    const auto positions = Positions(state);
    std::uint32_t partial = 0;
    for (std::uint32_t y = 0; y < Extent; ++y) {
        for (std::uint32_t x = 0; x < Extent; ++x) {
            bool ambiguous = false;
            const auto covered = CoveredSamples(positions, triangle, area, x, y, ambiguous);
            if (ambiguous) continue;
            if (covered != 0u && covered != samples) ++partial;
            const auto texel = texels[y * Extent + x];
            const double expected = 255.0 * covered / samples;
            Require(Near(Channel(texel, 0), expected) && Near(Channel(texel, 3), expected), what + ": pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") resolved to " + Hex(texel) + ", expected " + std::to_string(covered) + " of " + std::to_string(samples) + " samples");
        }
    }
    Require(partial >= 20u, what + ": only " + std::to_string(partial) + " partially covered pixels");
    return texels;
}

void CheckEmulated(AgcDriver::VulkanDevice& device) {
    id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
    if ([metal supportsTextureSampleCount:8]) std::puts("note: the device renders 8 samples natively; 8x is not emulated here (the 2 x 2x check still is)");
    // 8x coverage at the guest's standard 8x positions, two 4x slices on Apple9.
    const auto before = EmulatedSampleCounts();
    CheckDiagonal(device, 8u, "8x diagonal");
    // Exported masks: each slice sees its own 4 bits (APS5_SAMPLE_BASE).
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    GuestBlock destination(DestinationBytes());
    for (const auto& [mask, expected] : std::array<std::pair<std::uint32_t, double>, 4>{{{0x0fu, 127.5}, {0xf0u, 127.5}, {0x81u, 63.75}, {0x7eu, 191.25}}}) {
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * 8u));
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, 8u);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0xff0000ffu, "#define MASK 1\n", mask), 0u, 3u);
        RequireUniform(Resolve(device, target, destination), 0u, expected, "8x sample mask " + Hex(mask));
    }
    // Per-sample shading: only guest sample 0 (slice 0, host sample 0) is red.
    {
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * 8u));
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, 8u);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0xff0000ffu, "#define PER_SAMPLE 1\n"), 0u, 3u);
        RequireUniform(Resolve(device, target, destination), 0u, 255.0 / 8.0, "8x per-sample shading");
    }
    // Integer resolves take guest sample 0 (slice 0).
    for (const auto& [mask, expected] : std::array<std::pair<std::uint32_t, std::uint32_t>, 2>{{{0xfeu, 0u}, {0x1u, 0x12345678u}}}) {
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * 8u));
        const auto target = Target(source.Address(), VK_FORMAT_R32_UINT, 8u);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0x12345678u, "#define UINT_TARGET 1\n#define MASK 1\n", mask), 0u, 3u);
        const auto texels = Resolve(device, target, destination);
        for (std::uint32_t index = 0; index < texels.size(); ++index) Require(texels[index] == expected, "8x R32_UINT resolve with mask " + Hex(mask) + ": pixel " + std::to_string(index) + " is " + Hex(texels[index]));
    }
    // 8x depth: a near rectangle whose right edge is at x = 10.25, then a far full-screen draw.
    {
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * 8u));
        GuestBlock geometryBlock(16384);
        const float right = 10.25f / Extent * 2.0f - 1.0f;
        const std::array<std::array<float, 4>, 9> geometry {{{-1.0f, -1.0f, 0.25f, 0.0f}, {right, -1.0f, 0.25f, 0.0f}, {-1.0f, 1.0f, 0.25f, 0.0f}, {right, -1.0f, 0.25f, 0.0f}, {right, 1.0f, 0.25f, 0.0f}, {-1.0f, 1.0f, 0.25f, 0.0f}, {-1.0f, -1.0f, 0.5f, 0.0f}, {3.0f, -1.0f, 0.5f, 0.0f}, {-1.0f, 3.0f, 0.5f, 0.0f}}};
        WriteVertices(geometryBlock, geometry);
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, 8u);
        auto state = BaseState(target);
        std::vector<float> depthMemory(Extent * Extent * 8u, 0.0f);
        state.depth = AgcDriver::Graphics::DepthTarget {reinterpret_cast<std::uint64_t>(depthMemory.data()), 0u, {Extent, Extent}, VK_FORMAT_D32_SFLOAT, 1.0f, 0u};
        state.depth->samples = 8u;
        state.depthTest = true;
        state.depthWrite = true;
        state.depthCompare = VK_COMPARE_OP_LESS;
        Draw(device, state, MakeShaders(geometryBlock.Address(), 9u, 0xff0000ffu), 0u, 6u);
        Draw(device, state, MakeShaders(geometryBlock.Address(), 9u, 0xff00ff00u), 6u, 3u);
        const auto texels = Resolve(device, target, destination);
        std::uint32_t near = 0;
        for (const auto& position : Positions(state)) near += position[0] < 0.25 ? 1u : 0u;
        const auto edge = texels[7u * Extent + 10u];
        Require(Near(Channel(edge, 0), 255.0 * near / 8.0) && Near(Channel(edge, 1), 255.0 * (8u - near) / 8.0), "8x depth: the edge pixel resolved to " + Hex(edge) + ", expected " + std::to_string(near) + " of 8 samples near");
        Require(Channel(texels[7u * Extent + 9u], 0) == 255u && Channel(texels[7u * Extent + 11u], 1) == 255u, "8x depth: pixels 9/11 resolved to " + Hex(texels[7u * Extent + 9u]) + " " + Hex(texels[7u * Extent + 11u]));
        device.WaitIdle();
    }
    // The cost: consecutive draws into one 8x target share slice passes (a draw begins at the slice
    // the recorder holds open), so n draws need about n + 1 passes, not 2n.
    {
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * 8u));
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, 8u);
        const auto shaders = MakeShaders(vertices.Address(), 3u, 0xff0000ffu);
        device.WaitIdle();
        const auto first = EmulatedSampleCounts();
        constexpr std::uint32_t draws = 8u;
        for (std::uint32_t index = 0; index < draws; ++index) Draw(device, BaseState(target), shaders, 0u, 3u);
        const auto last = EmulatedSampleCounts();
        const auto begun = last.begun - first.begun;
        std::printf("emulated: %u 8x draws recorded %llu slice passes, %llu of them began a render pass (native 8x: %u draws, 1 pass)\n", draws, static_cast<unsigned long long>(last.slicePasses - first.slicePasses), static_cast<unsigned long long>(begun), draws);
        if (![metal supportsTextureSampleCount:8]) {
            Require(last.draws - first.draws == draws && last.slicePasses - first.slicePasses == 2u * draws, "the 8x draws were not recorded once per slice");
            Require(begun <= draws + 1u, "consecutive 8x draws began " + std::to_string(begun) + " passes (expected at most " + std::to_string(draws + 1u) + ")");
        }
        RequireUniform(Resolve(device, target, destination), 0u, 255.0, "8x consecutive draws");
    }
    const auto after = EmulatedSampleCounts();
    if (![metal supportsTextureSampleCount:8]) Require(after.draws > before.draws, "no 8x draw was emulated");
    // 4x emulated as 2 x 2x (APS5_MSAA_HOST_SAMPLES=2) against native 4x: the same resolve.
    const auto native = CheckDiagonal(device, 4u, "4x native diagonal");
    setenv("APS5_MSAA_HOST_SAMPLES", "2", 1);
    std::vector<std::uint32_t> split;
    try {
        split = CheckDiagonal(device, 4u, "4x as 2 x 2x diagonal");
    } catch (...) {
        unsetenv("APS5_MSAA_HOST_SAMPLES");
        throw;
    }
    unsetenv("APS5_MSAA_HOST_SAMPLES");
    std::uint32_t differing = 0;
    for (std::size_t index = 0; index < native.size(); ++index) differing += native[index] != split[index] ? 1u : 0u;
    Require(differing == 0u, std::to_string(differing) + " pixels of the 2 x 2x emulation differ from native 4x");
    sources.clear();
    std::puts("emulated: 8x coverage, slice masks, per-sample shading, integer resolve, 8x depth, pass sharing, 4x == 2 x 2x passed");
}

// Guest compute programs (llvm-mc -arch=amdgcn -mcpu=gfx1030), T# in s[4:11], results in s[0:3]:
// thread t reads sample t & 7 of pixel (t >> 3, 5):
//   v_and_b32 v7, 7, v0; v_lshrrev_b32 v4, 3, v0; v_mov_b32 v5, 5; v_mov_b32 v6, v7
//   image_load v[8:11], v[4:6], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm; ...store
alignas(256) constexpr std::array<std::uint32_t, 11> Load8Code {0x360e0087u, 0x2c080083u, 0x7e0a0285u, 0x7e0c0307u, 0xf0001f30u, 0x00010804u, 0xbf8c3f70u, 0x34020084u, 0xe0781000u, 0x80000801u, 0xbf810000u};
//   the same through dim:SQ_RSRC_IMG_2D_MSAA_ARRAY, layer v6 = 0, sample v7
alignas(256) constexpr std::array<std::uint32_t, 11> LoadArrayCode {0x360e0087u, 0x2c080083u, 0x7e0a0285u, 0x7e0c0280u, 0xf0001f38u, 0x00010804u, 0xbf8c3f70u, 0x34020084u, 0xe0781000u, 0x80000801u, 0xbf810000u};
//   v_mov_b32 v4, 0; image_get_resinfo v[8:11], v4, s[4:11] dmask:0xf dim:2D_MSAA / 2D_MSAA_ARRAY
alignas(256) constexpr std::array<std::uint32_t, 8> ResinfoCode {0x7e080280u, 0xf0381f30u, 0x00010804u, 0xbf8c3f70u, 0x34020084u, 0xe0781000u, 0x80000801u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 8> ResinfoArrayCode {0x7e080280u, 0xf0381f38u, 0x00010804u, 0xbf8c3f70u, 0x34020084u, 0xe0781000u, 0x80000801u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 32 * 4> GuestResults {};

std::uint32_t Log2(std::uint32_t value) {
    std::uint32_t log = 0;
    while ((1u << log) < value) ++log;
    return log;
}

// Runs `code` (32 threads) with an 8_8_8_8_UNORM 32x32 MSAA T# over `address`.
void RunGuest(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::uint64_t address, std::uint32_t samples, std::uint32_t type, std::uint32_t format = 56u) {
    GuestResults.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(16, 0u);
    const auto results = reinterpret_cast<std::uint64_t>(GuestResults.data());
    const std::array<std::uint32_t, 4> buffer {static_cast<std::uint32_t>(results), static_cast<std::uint32_t>((results >> 32u) & 0xffffu), static_cast<std::uint32_t>(sizeof(GuestResults)), 0x31016facu};
    std::array<std::uint32_t, 8> texture {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (((Extent - 1u) & 3u) << 30u), ((Extent - 1u) >> 2u) | ((Extent - 1u) << 14u), 0xfacu | (Log2(samples) << 16u) | (type << 28u), 0u, Log2(samples) << 4u, 0u, 0u};
    texture[1] = (texture[1] & ~(0x1ffu << 20u)) | (format << 20u);
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory {{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute {{32u, 1u, 1u}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request {
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}};
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

float ResultFloat(std::uint32_t thread, std::uint32_t component) {
    float value;
    std::memcpy(&value, &GuestResults[thread * 4u + component], 4u);
    return value;
}

void CheckSampledEmulated(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    for (const std::uint32_t samples : {4u, 8u}) {
        auto& source = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * samples));
        const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, samples);
        // Sample s of every pixel is red (32 s + 16) / 255.
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0u, "#define SAMPLE_COLOR 1\n"), 0u, 3u);
        // Type 14 (8x only: the 4x read is agc_metal_msaa_sampled) and type 15 T#s: guest sample t & 7.
        for (const std::uint32_t type : {14u, 15u}) {
            if (type == 14u && samples == 4u) continue;
            RunGuest(device, type == 14u ? std::span<const std::uint32_t>(Load8Code) : std::span<const std::uint32_t>(LoadArrayCode), source.Address(), samples, type);
            for (std::uint32_t thread = 0; thread < 32u; ++thread) {
                const auto sample = thread & 7u;
                const float expected = sample < samples ? static_cast<float>(sample * 32u + 16u) / 255.0f : 0.0f;
                const float alpha = sample < samples ? 1.0f : 0.0f;
                Require(std::fabs(ResultFloat(thread, 0) - expected) < 1e-6f && ResultFloat(thread, 3) == alpha, std::to_string(samples) + "x type " + std::to_string(type) + " T#: sample " + std::to_string(sample) + " of pixel " + std::to_string(thread >> 3u) + " read red " + std::to_string(ResultFloat(thread, 0)) + " alpha " + std::to_string(ResultFloat(thread, 3)) + ", expected " + std::to_string(expected));
            }
        }
        // Size queries: width, height, layers (type 15), the T#'s level count log2(samples) + 1.
        for (const std::uint32_t type : {14u, 15u}) {
            RunGuest(device, type == 14u ? std::span<const std::uint32_t>(ResinfoCode) : std::span<const std::uint32_t>(ResinfoArrayCode), source.Address(), samples, type);
            const std::array<std::uint32_t, 4> expected {Extent, Extent, type == 15u ? 1u : 0u, Log2(samples) + 1u};
            for (std::uint32_t component = 0; component < 4u; ++component) Require(GuestResults[component] == expected[component], std::to_string(samples) + "x type " + std::to_string(type) + " image_get_resinfo component " + std::to_string(component) + " is " + std::to_string(GuestResults[component]) + ", expected " + std::to_string(expected[component]));
        }
    }
    sources.clear();
    std::puts("sampled8: 8x type 14, 4x/8x type 15 reads of every guest sample, image_get_resinfo passed");
}

// Guest pixel programs, recompiled for the device (tests/MslPerVertexCheck.mm's): input 0 by the
// first enabled I/J pair, exported to MRT0 in all four channels.
//   v_interp_p1_f32 v2, v0, attr0.x; v_interp_p2_f32 v2, v1, attr0.x; exp mrt0 v2, v2, v2, v2 done vm
alignas(256) constexpr std::array<std::uint32_t, 5> InterpolatedPixelCode {0xc8080000u, 0xc8090001u, 0xf800180fu, 0x02020202u, 0xbf810000u};

ShaderRecompiler::RecompileResult RecompilePixel(const AgcDriver::VulkanDevice& device, ShaderRecompiler::PixelInput input, bool alphaToCoverage) {
    ShaderRecompiler::ShaderPixelStageInfo pixel {};
    pixel.wave32 = false;
    pixel.interpolatorCount = 1;
    pixel.executeOnNoop = true;
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(input);
    pixel.hasPerspectiveCenterVgpr = input == ShaderRecompiler::PixelInput::PerspectiveCenter;
    pixel.targetOutputMode[0] = 9u;
    pixel.alphaToCoverage = alphaToCoverage;
    pixel.targetExportMapping.fill(0xe4u);
    const std::span<const std::uint32_t> code(InterpolatedPixelCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory {{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const auto target = device.Target();
    ShaderRecompiler::RecompileRequest request {
        {ShaderRecompiler::ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64u, 0, {}, std::nullopt, pixel, std::nullopt, memory},
        target,
        PixelPushLayout(0u, target)};
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

void DrawRecompiled(AgcDriver::VulkanDevice& device, const AgcDriver::Graphics::State& state, const Shaders& vertex, const ShaderRecompiler::RecompileResult& pixel) {
    const std::array<AgcDriver::Graphics::CompiledShader, 2> compiled {{{ShaderRecompiler::ShaderStage::Vertex, &vertex.vertex, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &pixel, PixelPushOffset(0u, device.Target())}}};
    AgcDriver::Pm4::DrawParameters draw {0u, 3u, 0u, 1u, 0u, false};
    device.Draw(state, draw, compiled);
}

// The guest's PERSP_SAMPLE pair interpolates input 0 (framebuffer x / 32) at each sample's
// position, PERSP_CENTER at the pixel center: read back per sample through a 2D MSAA T#.
void CheckInterpolation(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    const auto vertex = MakeShaders(vertices.Address(), 3u, 0u, "", ~0u, true);
    for (const std::uint32_t samples : {4u, 8u}) {
        for (const bool atSample : {true, false}) {
            const auto pixel = RecompilePixel(device, atSample ? ShaderRecompiler::PixelInput::PerspectiveSample : ShaderRecompiler::PixelInput::PerspectiveCenter, false);
            const std::string source(pixel.msl.Source());
            Require(source.find(atSample ? "sample_perspective" : "center_perspective") != std::string::npos, "the recompiled pixel program does not interpolate at the " + std::string(atSample ? "sample" : "center"));
            auto& block = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * samples));
            const auto target = Target(block.Address(), VK_FORMAT_R8G8B8A8_UNORM, samples);
            const auto state = BaseState(target);
            DrawRecompiled(device, state, vertex, pixel);
            RunGuest(device, samples == 8u ? std::span<const std::uint32_t>(Load8Code) : std::span<const std::uint32_t>(MsaaLoadCode), block.Address(), samples, 14u);
            const auto positions = Positions(state);
            for (std::uint32_t thread = 0; thread < 32u; ++thread) {
                const auto sample = samples == 8u ? thread & 7u : thread & 3u;
                const auto x = samples == 8u ? thread >> 3u : thread >> 2u;
                const double expected = (x + (atSample ? positions[sample][0] : 0.5)) / Extent;
                Require(std::fabs(ResultFloat(thread, 0) - expected) <= 0.6 / 255.0, std::to_string(samples) + "x " + (atSample ? "sample" : "center") + " interpolation: sample " + std::to_string(sample) + " of pixel " + std::to_string(x) + " is " + std::to_string(ResultFloat(thread, 0) * 255.0f) + "/255, expected " + std::to_string(expected * 255.0) + "/255");
            }
        }
    }
    sources.clear();
    std::puts("interp: PERSP_SAMPLE inputs interpolate at each 4x/8x sample, PERSP_CENTER at the center passed");
}

// One input read at several locations (DECISIONS D38): the interpolant<> path. Guest programs
// (llvm-mc -arch=amdgcn -mcpu=gfx1030), I/J pairs in SPI_PS_INPUT_ADDR order from v0:
//   sample, center, centroid (PERSP_SAMPLE v[0:1], PERSP_CENTER v[2:3], PERSP_CENTROID v[4:5]):
//     v_interp_p1_f32 v6, v0, attr0.x; v_interp_p2_f32 v6, v1, attr0.x
//     v_interp_p1_f32 v7, v2, attr0.x; v_interp_p2_f32 v7, v3, attr0.x
//     v_interp_p1_f32 v8, v4, attr0.x; v_interp_p2_f32 v8, v5, attr0.x
//     exp mrt0 v6, v7, v8, v7 done vm; s_endpgm
alignas(256) constexpr std::array<std::uint32_t, 9> ThreeLocationsPixelCode {0xc8180000u, 0xc8190001u, 0xc81c0002u, 0xc81d0003u, 0xc8200004u, 0xc8210005u, 0xf800180fu, 0x07080706u, 0xbf810000u};
//   center, centroid (PERSP_CENTER v[0:1], PERSP_CENTROID v[2:3]), no per-sample execution:
//     the first four interp instructions above; exp mrt0 v6, v7, v6, v7 done vm; s_endpgm
alignas(256) constexpr std::array<std::uint32_t, 7> CenterCentroidPixelCode {0xc8180000u, 0xc8190001u, 0xc81c0002u, 0xc81d0003u, 0xf800180fu, 0x07060706u, 0xbf810000u};

ShaderRecompiler::RecompileResult RecompileLocations(const AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::uint32_t inputs) {
    using ShaderRecompiler::PixelInput;
    using ShaderRecompiler::PixelInputBit;
    ShaderRecompiler::ShaderPixelStageInfo pixel {};
    pixel.wave32 = false;
    pixel.interpolatorCount = 1;
    pixel.executeOnNoop = true;
    pixel.inputAddr = inputs;
    pixel.hasPerspectiveCenterVgpr = (inputs & PixelInputBit(PixelInput::PerspectiveCenter)) != 0u;
    pixel.perspectiveCentroid = (inputs & PixelInputBit(PixelInput::PerspectiveCentroid)) != 0u;
    pixel.targetOutputMode[0] = 9u;
    pixel.targetExportMapping.fill(0xe4u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory {{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const auto target = device.Target();
    ShaderRecompiler::RecompileRequest request {
        {ShaderRecompiler::ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64u, 0, {}, std::nullopt, pixel, std::nullopt, memory},
        target,
        PixelPushLayout(0u, target)};
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

// A triangle covering x < Edge (framebuffer pixels) over the whole height: pixel 2 of row 5 is
// partially covered, its center (2.5) is not.
constexpr double LocationsEdge = 2.4375 + 1.0 / 32.0;
constexpr std::array<std::array<float, 4>, 3> LocationsTriangle {{
    {static_cast<float>(LocationsEdge / Extent * 2.0 - 1.0), -3.0f, 0.5f, 0.0f},
    {static_cast<float>(LocationsEdge / Extent * 2.0 - 1.0), 3.0f, 0.5f, 0.0f},
    {-5.0f, 0.0f, 0.5f, 0.0f},
}};

// Input 0 is framebuffer x / 32 (linear in x, so perspective and linear agree); channels: sample,
// center, centroid (first program), center, centroid (second). Each is checked against the CPU's location. Sample reads are the sample's position; center reads x + 0.5,
// covered or not; centroid reads lie within the covered samples' x range (Vulkan and Metal leave
// the point inside the covered area to the implementation), and equal the center for a fully
// covered pixel when the function runs per pixel.
void CheckInterpolationLocations(AgcDriver::VulkanDevice& device) {
    using ShaderRecompiler::PixelInput;
    using ShaderRecompiler::PixelInputBit;
    GuestBlock vertices(16384);
    WriteVertices(vertices, LocationsTriangle);
    const auto vertex = MakeShaders(vertices.Address(), 3u, 0u, "", ~0u, true);
    struct Program {
        const char* name;
        std::span<const std::uint32_t> code;
        std::uint32_t inputs;
        bool perSample;
    };
    const std::array<Program, 2> programs {{
        {"sample+center+centroid", ThreeLocationsPixelCode, PixelInputBit(PixelInput::PerspectiveSample) | PixelInputBit(PixelInput::PerspectiveCenter) | PixelInputBit(PixelInput::PerspectiveCentroid), true},
        {"center+centroid", CenterCentroidPixelCode, PixelInputBit(PixelInput::PerspectiveCenter) | PixelInputBit(PixelInput::PerspectiveCentroid), false},
    }};
    std::uint32_t partial = 0;
    for (const auto& program : programs) {
        const auto pixel = RecompileLocations(device, program.code, program.inputs);
        const std::string source(pixel.msl.Source());
        const std::string what(program.name);
        Require(source.find("interpolant<float4, interpolation::perspective> p0 [[user(locn0)]]") != std::string::npos, what + ": input 0 is not an interpolant");
        Require(source.find(".interpolate_at_centroid()") != std::string::npos && source.find(".interpolate_at_center()") != std::string::npos, what + ": no center and centroid reads");
        Require((source.find(".interpolate_at_sample(aps5_sample_id)") != std::string::npos) == program.perSample, what + ": the sample read does not match");
        Require(pixel.msl.Reflection().perSample == program.perSample, what + ": Reflection::perSample does not match");
        for (const std::uint32_t samples : {4u, 8u}) {
            auto& block = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * samples));
            const auto target = Target(block.Address(), VK_FORMAT_R8G8B8A8_UNORM, samples);
            const auto state = BaseState(target);
            DrawRecompiled(device, state, vertex, pixel);
            RunGuest(device, samples == 8u ? std::span<const std::uint32_t>(Load8Code) : std::span<const std::uint32_t>(MsaaLoadCode), block.Address(), samples, 14u);
            const auto positions = Positions(state);
            constexpr double tolerance = 0.6 / 255.0;
            for (std::uint32_t thread = 0; thread < 32u; ++thread) {
                const auto sample = samples == 8u ? thread & 7u : thread & 3u;
                const auto x = samples == 8u ? thread >> 3u : thread >> 2u;
                if (x + positions[sample][0] >= LocationsEdge) continue;
                double low = 1.0;
                double high = 0.0;
                std::uint32_t covered = 0;
                for (const auto& position : positions) {
                    if (x + position[0] >= LocationsEdge) continue;
                    low = std::min(low, position[0]);
                    high = std::max(high, position[0]);
                    ++covered;
                }
                const auto label = what + " " + std::to_string(samples) + "x sample " + std::to_string(sample) + " of pixel " + std::to_string(x);
                const double center = (x + 0.5) / Extent;
                if (program.perSample) {
                    const double expected = (x + positions[sample][0]) / Extent;
                    Require(std::fabs(ResultFloat(thread, 0) - expected) <= tolerance, label + ": the sample read is " + std::to_string(ResultFloat(thread, 0) * 255.0f) + "/255, expected " + std::to_string(expected * 255.0) + "/255");
                } else {
                    Require(std::fabs(ResultFloat(thread, 0) - center) <= tolerance, label + ": the center read (red) is " + std::to_string(ResultFloat(thread, 0) * 255.0f) + "/255, expected " + std::to_string(center * 255.0) + "/255");
                }
                if (program.perSample) Require(std::fabs(ResultFloat(thread, 1) - center) <= tolerance, label + ": the center read (green) is " + std::to_string(ResultFloat(thread, 1) * 255.0f) + "/255, expected " + std::to_string(center * 255.0) + "/255");
                const auto centroid = ResultFloat(thread, program.perSample ? 2u : 1u);
                const double lowest = (x + low) / Extent - tolerance;
                const double highest = (x + high) / Extent + tolerance;
                Require(centroid >= lowest && centroid <= highest, label + ": the centroid read is " + std::to_string(centroid * 255.0f) + "/255, outside the covered samples' " + std::to_string((x + low) / Extent * 255.0) + ".." + std::to_string((x + high) / Extent * 255.0) + "/255");
                if (covered == samples && !program.perSample) Require(std::fabs(centroid - center) <= tolerance, label + ": the centroid read of a fully covered pixel is " + std::to_string(centroid * 255.0f) + "/255, not the center");
                if (covered != samples) {
                    ++partial;
                    if (sample == 0u) std::printf("%s: %u of %u samples covered, centroid %.3f/255 (center %.3f/255)\n", label.c_str(), covered, samples, centroid * 255.0, center * 255.0);
                }
            }
        }
    }
    Require(partial != 0u, "no partially covered pixel was checked");
    sources.clear();
    std::puts("interp_locations: one input at sample, center and centroid (interpolant<>), 4x/8x, passed");
}

// Alpha to coverage of the recompiled program (alpha = x / 32 at the pixel center): the measured
// 0xaa01 rule (DECISIONS D28): floor(alpha * n + 0.5) samples, 8x in a fixed mask per 1/16.
constexpr std::array<std::uint32_t, 16> AlphaMasks8 {0x00u, 0x10u, 0x04u, 0x14u, 0x12u, 0x32u, 0x16u, 0x36u, 0x35u, 0x75u, 0x3du, 0x7du, 0x77u, 0xf7u, 0x7fu, 0xffu};

std::uint32_t AlphaMask(double alpha, std::uint32_t samples) {
    if (!(alpha > 0.0)) return 0u;
    if (alpha >= 1.0) return (1u << samples) - 1u;
    if (samples == 8u) return AlphaMasks8[std::min<std::uint32_t>(static_cast<std::uint32_t>(alpha * 16.0), 15u)];
    const auto count = std::min<std::uint32_t>(static_cast<std::uint32_t>(alpha * samples + 0.5), samples);
    return (1u << count) - 1u;
}

void CheckAlphaToCoverage(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    GuestBlock destination(DestinationBytes());
    const auto vertex = MakeShaders(vertices.Address(), 3u, 0u, "", ~0u, true);
    const auto pixel = RecompilePixel(device, ShaderRecompiler::PixelInput::PerspectiveCenter, true);
    Require(std::string(pixel.msl.Source()).find("aps5_alpha_to_mask(") != std::string::npos, "the recompiled pixel program has no alpha-to-coverage mask");
    for (const std::uint32_t samples : {2u, 4u, 8u}) {
        auto& block = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * samples));
        const auto target = Target(block.Address(), VK_FORMAT_R8G8B8A8_UNORM, samples);
        auto state = BaseState(target);
        state.alphaToCoverage = true;
        DrawRecompiled(device, state, vertex, pixel);
        if (samples == 8u) {
            // Which guest samples: sample s of pixel x is lit iff mask bit s.
            RunGuest(device, Load8Code, block.Address(), samples, 14u);
            for (std::uint32_t thread = 0; thread < 32u; ++thread) {
                const auto sample = thread & 7u;
                const auto x = thread >> 3u;
                const double alpha = (x + 0.5) / Extent;
                const bool lit = ((AlphaMask(alpha, 8u) >> sample) & 1u) != 0u;
                Require(std::fabs(ResultFloat(thread, 3) - (lit ? alpha : 0.0)) <= 0.6 / 255.0, "8x alpha to coverage: sample " + std::to_string(sample) + " of pixel " + std::to_string(x) + " has alpha " + std::to_string(ResultFloat(thread, 3)) + ", expected it " + (lit ? "lit" : "uncovered"));
            }
        }
        const auto texels = Resolve(device, target, destination);
        for (std::uint32_t x = 0; x < Extent; ++x) {
            const double alpha = (x + 0.5) / Extent;
            const auto covered = static_cast<std::uint32_t>(__builtin_popcount(AlphaMask(alpha, samples)));
            const auto expected = std::round(alpha * 255.0) * covered / samples;
            const auto texel = texels[9u * Extent + x];
            Require(std::fabs(static_cast<double>(Channel(texel, 0)) - expected) <= 1.5, std::to_string(samples) + "x alpha to coverage: pixel " + std::to_string(x) + " resolved to " + Hex(texel) + ", expected red " + std::to_string(expected) + " (" + std::to_string(covered) + " samples)");
        }
    }
    sources.clear();
    std::puts("a2c: proportional alpha-to-coverage at 2x, 4x and emulated 8x (per-sample bits) passed");
}

// Guest storage access to rendered MSAA targets (DECISIONS D22, D28): stores and atomics go to the
// target's storage shadow (texture2d_array of every sample), which the host image takes back before
// its next use. Programs (llvm-mc gfx1030), T# in s[4:11]:
//   4x: v_and_b32 v6, 2, v0; v_lshrrev_b32 v4, 2, v0  (samples 0 and 2 of pixel (t >> 2, 5))
//   8x: v_and_b32 v6, 6, v0; v_lshrrev_b32 v4, 3, v0  (samples 0, 2, 4, 6 of pixel (t >> 3, 5))
//   v_mov_b32 v5, 5; v[8:11] = (0, 1.0, 0, 1.0); image_store v[8:11], v[4:6], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm
alignas(256) constexpr std::array<std::uint32_t, 10> Store4Code {0x360c0082u, 0x2c080082u, 0x7e0a0285u, 0x7e100280u, 0x7e1202f2u, 0x7e140280u, 0x7e1602f2u, 0xf0201f30u, 0x00010804u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 10> Store8Code {0x360c0086u, 0x2c080083u, 0x7e0a0285u, 0x7e100280u, 0x7e1202f2u, 0x7e140280u, 0x7e1602f2u, 0xf0201f30u, 0x00010804u, 0xbf810000u};
//   v_and_b32 v6, 3, v0; v_lshrrev_b32 v4, 2, v0; v_mov_b32 v5, 5; v_mov_b32 v8, 1
//   image_atomic_add v8, v[4:6], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm (every sample of pixels 0-7)
alignas(256) constexpr std::array<std::uint32_t, 7> AtomicAddCode {0x360c0083u, 0x2c080082u, 0x7e0a0285u, 0x7e100281u, 0xf0441130u, 0x00010804u, 0xbf810000u};

void CheckStorage(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    GuestBlock destination(DestinationBytes());
    for (const std::uint32_t samples : {4u, 8u}) {
        auto& block = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * samples));
        const auto target = Target(block.Address(), VK_FORMAT_R8G8B8A8_UNORM, samples);
        // Sample s red (32 s + 16) / 255, then green stores to the even samples of row 5's first pixels.
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0u, "#define SAMPLE_COLOR 1\n"), 0u, 3u);
        RunGuest(device, samples == 8u ? std::span<const std::uint32_t>(Store8Code) : std::span<const std::uint32_t>(Store4Code), block.Address(), samples, 14u);
        // The sampled view (host image, after the shadow's samples came back): stored samples green,
        // the others still the draw's.
        RunGuest(device, samples == 8u ? std::span<const std::uint32_t>(Load8Code) : std::span<const std::uint32_t>(MsaaLoadCode), block.Address(), samples, 14u);
        const auto pixels = samples == 8u ? 4u : 8u;
        for (std::uint32_t thread = 0; thread < 32u; ++thread) {
            const auto sample = thread % samples;
            const auto x = thread / samples;
            const bool stored = sample % 2u == 0u && x < pixels;
            const float red = stored ? 0.0f : static_cast<float>(sample * 32u + 16u) / 255.0f;
            const float green = stored ? 1.0f : 0.0f;
            Require(std::fabs(ResultFloat(thread, 0) - red) < 1e-6f && ResultFloat(thread, 1) == green, std::to_string(samples) + "x storage: sample " + std::to_string(sample) + " of pixel " + std::to_string(x) + " reads " + std::to_string(ResultFloat(thread, 0)) + " " + std::to_string(ResultFloat(thread, 1)) + ", expected " + (stored ? "the stored green" : "the drawn red"));
        }
        // The resolve averages the stored samples with the drawn ones.
        const auto texels = Resolve(device, target, destination);
        double drawnRed = 0.0;
        for (std::uint32_t sample = 1; sample < samples; sample += 2u) drawnRed += sample * 32.0 + 16.0;
        const auto texel = texels[5u * Extent + 0u];
        Require(Near(Channel(texel, 0), drawnRed / samples) && Near(Channel(texel, 1), 255.0 / 2.0), std::to_string(samples) + "x storage: the resolve of a stored pixel is " + Hex(texel));
        // Drawing again after the stores: the draw renders over the host image the stores reached.
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0xff0000ffu, "#define MASK 1\n", 0x2u), 0u, 3u);
        const auto again = Resolve(device, target, destination);
        Require(Near(Channel(again[5u * Extent + 0u], 1), 255.0 / 2.0), std::to_string(samples) + "x storage: a draw after the stores lost them: " + Hex(again[5u * Extent + 0u]));
        const auto counts = AgcDriver::Graphics::Metal::MultisampleShadowCounts(block.Address());
        std::printf("storage %ux: %llu host-to-shadow and %llu shadow-to-host copies\n", samples, static_cast<unsigned long long>(counts.hostToShadow), static_cast<unsigned long long>(counts.shadowToHost));
        Require(counts.hostToShadow == 1u && counts.shadowToHost == 1u, std::to_string(samples) + "x storage: expected one copy each way");
    }
    // 32-bit atomics on every sample of an R32_UINT 4x target (T# format 20).
    {
        auto& block = *sources.emplace_back(std::make_unique<GuestBlock>(Extent * Extent * 4u * 4u));
        const auto target = Target(block.Address(), VK_FORMAT_R32_UINT, 4u);
        Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0x10u, "#define UINT_TARGET 1\n"), 0u, 3u);
        RunGuest(device, AtomicAddCode, block.Address(), 4u, 14u, 20u);
        RunGuest(device, MsaaLoadCode, block.Address(), 4u, 14u, 20u);
        for (std::uint32_t thread = 0; thread < 32u; ++thread) Require(GuestResults[thread * 4u] == 0x11u, "4x atomic add: sample " + std::to_string(thread & 3u) + " of pixel " + std::to_string(thread >> 2u) + " is " + Hex(GuestResults[thread * 4u]));
        const auto texels = Resolve(device, target, destination);
        Require(texels[5u * Extent + 7u] == 0x11u && texels[5u * Extent + 8u] == 0x10u && texels[6u * Extent] == 0x10u, "4x atomic add: the sample-0 resolve reads " + Hex(texels[5u * Extent + 7u]) + " " + Hex(texels[5u * Extent + 8u]));
    }
    sources.clear();
    std::puts("storage: 4x/8x guest stores and 4x atomics reach the rendered target's samples passed");
}

// Surface ranges and textures of a build (TechnicalDebt metal-msaa gap 6): an MSAA view is kept in
// the build's textures without a surface range, so pairing "texture" ranges with textures by
// position shifted every later range onto the previous texture. Bound here: an MSAA T#, then an
// R32_UINT surface in host-imported memory (a view of its storage image, served GPU-direct) and a
// BC1 surface in the same import (a CPU snapshot). The pre-sync list (PresyncSurfaces, the
// surfaces the build's lookups read on the CPU) must hold the BC1 surface and not the R32 one;
// by position the BC1 range took the R32 view's texture and was left out.
class ImportedSurfaces {
public:
    static constexpr std::uint32_t Side = 256u;
    static constexpr std::size_t SurfaceBytes = std::size_t{Side} * Side * 4u;
    explicit ImportedSurfaces(const AgcDriver::Graphics::Context& context) : context(context) {
        memory = std::aligned_alloc(65536, 2u * SurfaceBytes);
        Require(memory != nullptr, "cannot allocate the imported surfaces");
        std::memset(memory, 0x5a, 2u * SurfaceBytes);
        GuestAllocations::Mutation().Add(memory, 2u * SurfaceBytes, true, true, true);
    }
    ~ImportedSurfaces() {
        AgcDriver::Graphics::ClearCachedTextures(context.device);
        GuestAllocations::Mutation().Remove(memory);
        // Reconciles the import away before the memory goes.
        AgcDriver::Graphics::HostImportFor(context, Surface(0), 2u * SurfaceBytes);
        std::free(memory);
    }
    ImportedSurfaces(const ImportedSurfaces&) = delete;
    ImportedSurfaces& operator=(const ImportedSurfaces&) = delete;
    std::uint64_t Surface(std::size_t index) const { return reinterpret_cast<std::uint64_t>(memory) + index * SurfaceBytes; }

private:
    const AgcDriver::Graphics::Context& context;
    void* memory = nullptr;
};

// A 2D single-sample T# (R64KB_X tiling) of `side` x `side` texels of guest format `format`.
std::array<std::uint32_t, 8> SingleSampleDescriptor(std::uint64_t address, std::uint32_t format, std::uint32_t side) {
    return {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((side - 1u) & 3u) << 30u), ((side - 1u) >> 2u) | ((side - 1u) << 14u), 0xfacu | (0x1bu << 20u) | (9u << 28u), 0u, 0u, 0u, 0u};
}

ShaderRecompiler::DescriptorBinding SampledBinding(std::uint32_t slot, ShaderRecompiler::DescriptorImageShape shape, std::span<const std::array<std::uint32_t, 8>> elements) {
    ShaderRecompiler::DescriptorBinding binding {};
    binding.kind = ShaderRecompiler::DescriptorKind::SampledImage;
    binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
    binding.descriptorSet = 0;
    binding.binding = RuntimeAbi::FirstImageBinding + AgcDriver::Graphics::PlaceholderDimensionsPerClass + slot;
    binding.count = static_cast<std::uint32_t>(elements.size());
    binding.imageShape = shape;
    binding.imageWritten.assign(elements.size(), false);
    binding.imageSamplers.assign(elements.size(), 0u);
    for (const auto& words : elements) binding.guestDescriptor.insert(binding.guestDescriptor.end(), words.begin(), words.end());
    return binding;
}

void CheckRanges(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    GuestBlock source(Extent * Extent * 4u * Samples);
    const auto target = Target(source.Address(), VK_FORMAT_R8G8B8A8_UNORM, Samples);
    Draw(device, BaseState(target), MakeShaders(vertices.Address(), 3u, 0u, "#define SAMPLE_COLOR 1\n"), 0u, 3u);
    device.WaitIdle();
    const auto& context = AgcDriver::VulkanDeviceTestAccess::ImageContextOf(device);
    Require(context.hostImportAlignment != 0u, "the Metal device's Context has no host imports");
    {
        ImportedSurfaces surfaces(context);
        Require(AgcDriver::Graphics::HostImportFor(context, surfaces.Surface(0), 2u * ImportedSurfaces::SurfaceBytes) != nullptr, "the Metal device refused the host import of the test surfaces");
        const auto viewed = surfaces.Surface(0);
        const auto compressed = surfaces.Surface(1);
        // The MSAA T# as agc_metal_msaa_sampled's (8_8_8_8_UNORM 32x32, 4x, type 14).
        const auto address = source.Address();
        const std::array<std::array<std::uint32_t, 8>, 1> msaa {{{static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (((Extent - 1u) & 3u) << 30u), ((Extent - 1u) >> 2u) | ((Extent - 1u) << 14u), 0xfacu | (2u << 16u) | (14u << 28u), 0u, 2u << 4u, 0u, 0u}}};
        const std::array<std::array<std::uint32_t, 8>, 2> plain {SingleSampleDescriptor(viewed, 20u, ImportedSurfaces::Side), SingleSampleDescriptor(compressed, 169u, ImportedSurfaces::Side)};
        ShaderRecompiler::RecompileResult program;
        program.bindings.push_back(SampledBinding(3u, ShaderRecompiler::DescriptorImageShape::Image2DMultisample, msaa));
        program.bindings.push_back(SampledBinding(2u, ShaderRecompiler::DescriptorImageShape::Image2D, plain));
        const AgcDriver::Graphics::CompiledShader shader {ShaderRecompiler::ShaderStage::Compute, &program, 0u};
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::Graphics::ShaderResources resources(context, shader, AgcDriver::Graphics::ShaderResources::MetalImagesOnly{});
        Require(resources.Completed(), "the test build did not complete");
        const auto& textures = resources.SampledTextures();
        Require(textures.size() == 3u, "the build holds " + std::to_string(textures.size()) + " sampled textures, not the MSAA view and two single-sample textures");
        Require(textures[1]->ViewsStorageImage(), "the R32_UINT surface in host-imported memory is not a view of its storage image");
        Require(!textures[2]->ViewsStorageImage(), "the BC1 surface is not a CPU snapshot");
        const auto presync = resources.PresyncSurfaces();
        const auto listed = [&](std::uint64_t surface) { return std::any_of(presync.begin(), presync.end(), [&](const auto& range) { return range.address == surface; }); };
        Require(listed(compressed), "the pre-sync list leaves out the BC1 snapshot surface (its range paired with another texture)");
        Require(!listed(viewed), "the pre-sync list holds the R32_UINT surface its storage image serves GPU-direct");
        Require(presync.size() == 1u, "the pre-sync list holds " + std::to_string(presync.size()) + " surfaces, not the BC1 one alone");
    }
    std::puts("ranges: an MSAA T# before single-sample T#s keeps each surface range on its texture passed");
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
        else if (mode == "emulated") CheckEmulated(*device);
        else if (mode == "sampled8") CheckSampledEmulated(*device);
        else if (mode == "interp") CheckInterpolation(*device);
        else if (mode == "interp_locations") CheckInterpolationLocations(*device);
        else if (mode == "a2c") CheckAlphaToCoverage(*device);
        else if (mode == "storage") CheckStorage(*device);
        else if (mode == "ranges") CheckRanges(*device);
        else {
            std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 2;
        }
        device->WaitIdle();
        std::printf("metal msaa %s tests passed\n", mode.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        sources.clear();
        try {
            if (device != nullptr) device->WaitIdle();
        } catch (const std::exception& idle) {
            std::fprintf(stderr, "%s\n", idle.what());
        }
        return 1;
    }
}
