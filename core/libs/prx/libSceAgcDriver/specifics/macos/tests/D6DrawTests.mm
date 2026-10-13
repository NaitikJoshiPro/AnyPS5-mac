// Metal draw tests of work package D6 (docs/mac/M2_DESIGN.md "DRAW results"): the draw path of
// MetalDraw.mm with hand-written vertex and fragment functions against the MslAbi Version 3
// contract (B6/D6 interface), so the driver side is proven independently of B6's emitter.
//   agc_metal_draw_tests targets   (agc_metal_draw_targets: linear targets in place, scissor, a
//                                   continued pass, tiled RenderTarget-mode targets loaded and
//                                   written back, cull mode and front-face winding as Vulkan)
//   agc_metal_draw_tests primitives (agc_metal_draw_primitives: depth surfaces and the depth test,
//                                   indexed triangle fans, CPU-read indirect draws, instancing)
//   agc_metal_draw_tests offscreen  (agc_metal_offscreen_frames: headless presentation, a window
//                                   without a Metal layer: pixels scaled aspect-fit over black,
//                                   clears, slots retired; read back via APS5_METAL_PRESENT_READBACK)
#include "../src/MetalDeviceState.h"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
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
namespace Abi = ShaderRecompiler::MslAbi;
namespace RuntimeAbi = ShaderRecompiler::RuntimeAbi;
using AgcDriver::Graphics::ColorTileMode;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Guest memory registered with GuestAllocations, as the guest's own allocations are.
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

constexpr std::uint32_t Extent = 64u;

// The vertex function pulls location 0 (float4 x, y, z, w) through the VertexFetch slot and
// passes white to the fragment function, which writes push dword 32 (byte 128, the pixel slot)
// unpacked as unorm RGBA8 times its input.
const char* VertexSource = R"(#include <metal_stdlib>
using namespace metal;
struct Fetch { ulong address; uint stride; uint bytes; uint word3; uint rate; uint components; uint reserved; };
struct VertexOut { float4 position [[position]]; float4 p0 [[user(locn0)]]; };
vertex VertexOut aps5_main(uint vertexId [[vertex_id]], uint instanceId [[instance_id]], device const Fetch* fetch [[buffer(4)]], constant uint* push [[buffer(1)]]) {
    const uint index = fetch[0].rate == 0u ? vertexId : instanceId;
    device const float4* data = reinterpret_cast<device const float4*>(fetch[0].address + ulong(index) * fetch[0].stride);
    const float4 value = ulong(index) * fetch[0].stride + 16u <= fetch[0].bytes ? *data : float4(0.0f);
    VertexOut out;
    // Instances shift right by w (the instancing check).
    out.position = float4(value.x + float(instanceId) * value.w, value.y, value.z, 1.0f);
    out.p0 = float4(1.0f);
    return out;
}
)";

const char* FragmentSource = R"(#include <metal_stdlib>
using namespace metal;
struct FragmentIn { float4 p0 [[user(locn0)]]; };
struct FragmentOut { float4 color [[color(0)]]; };
fragment FragmentOut aps5_main(FragmentIn in [[stage_in]], constant uint* push [[buffer(1)]]) {
    FragmentOut out;
    out.color = unpack_unorm4x8_to_float(push[32]) * in.p0;
    return out;
}
)";

struct Shaders {
    ShaderRecompiler::RecompileResult vertex;
    ShaderRecompiler::RecompileResult fragment;
};

std::array<std::uint32_t, 4> VertexDescriptor(std::uint64_t address, std::uint32_t stride, std::uint32_t records) {
    // Format 77 (32_32_32_32 float), dst_sel xyzw.
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) | (stride << 16u), records, 0x00000facu | (77u << 12u)};
}

Shaders MakeShaders(std::uint64_t vertices, std::uint32_t records, std::uint32_t color, std::uint32_t rate = 0u) {
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
    attribute.resource.fields = VertexDescriptor(vertices, 16u, records);
    attribute.fetchIndex = rate;
    attribute.formatComponents = 4u;
    shaders.vertex.vertexAttributes.push_back(attribute);
    Abi::Reflection fragment;
    fragment.stage = Abi::Stage::Fragment;
    fragment.group = RuntimeAbi::Stage::Fragment;
    fragment.colorOutputs = 1u;
    fragment.usesPushData = true;
    shaders.fragment.msl = ShaderRecompiler::SharedMsl(ShaderRecompiler::MslShader{FragmentSource, fragment});
    shaders.fragment.pushConstants.resize(4u);
    std::memcpy(shaders.fragment.pushConstants.data(), &color, 4u);
    return shaders;
}

AgcDriver::Graphics::State BaseState() {
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
    return state;
}

void SetTarget(AgcDriver::Graphics::State& state, std::uint64_t address, VkFormat format, ColorTileMode mode, std::uint32_t elementBytes) {
    const AgcDriver::Graphics::ColorTargetLayout layout(Extent, Extent, mode, elementBytes);
    state.color = {address, {Extent, Extent}, format, layout.Bytes(), 0xe4u};
    state.color.tileMode = mode;
    state.color.elementBytes = elementBytes;
    state.colors = {state.color};
}

void Draw(AgcDriver::VulkanDevice& device, const AgcDriver::Graphics::State& state, const Shaders& shaders, AgcDriver::Pm4::DrawParameters draw) {
    const std::array<AgcDriver::Graphics::CompiledShader, 2> compiled {{{ShaderRecompiler::ShaderStage::Vertex, &shaders.vertex, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &shaders.fragment, 128u}}};
    device.Draw(state, draw, compiled);
}

// Tiled targets are resident images (upstream image path): their results reach guest memory when a
// reader flushes them, as the PM4 core's readers do.
void StoreResults(AgcDriver::VulkanDevice& device, std::uint64_t address, std::size_t bytes) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    AgcDriver::Graphics::StorageTexture::FlushPending(address, bytes, nullptr, "test");
    device.WaitIdle();
}

void WriteVertices(GuestBlock& block, std::span<const std::array<float, 4>> vertices) {
    std::memcpy(block.Data(), vertices.data(), vertices.size_bytes());
}

std::uint32_t PixelRgba8(const GuestBlock& target, std::uint32_t x, std::uint32_t y) {
    std::uint32_t value;
    std::memcpy(&value, target.Data() + (static_cast<std::size_t>(y) * Extent + x) * 4u, 4u);
    return value;
}

// A full-screen triangle (counter-clockwise in Vulkan framebuffer space with the test viewport).
constexpr std::array<std::array<float, 4>, 3> FullScreen {{{-1.0f, -1.0f, 0.5f, 0.0f}, {3.0f, -1.0f, 0.5f, 0.0f}, {-1.0f, 3.0f, 0.5f, 0.0f}}};

void CheckTargets(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    WriteVertices(vertices, FullScreen);
    // Linear RGBA8: the texture aliases the guest bytes.
    GuestBlock linear(Extent * Extent * 4u);
    auto state = BaseState();
    SetTarget(state, linear.Address(), VK_FORMAT_R8G8B8A8_UNORM, ColorTileMode::Linear, 4u);
    const auto red = MakeShaders(vertices.Address(), 3u, 0xff0000ffu);
    Draw(device, state, red, {0u, 3u, 0u, 1u, 0u, false});
    // A second draw into the same targets continues the pass: green on the left half (scissor).
    auto half = state;
    half.scissor = {{0, 0}, {Extent / 2u, Extent}};
    const auto green = MakeShaders(vertices.Address(), 3u, 0xff00ff00u);
    Draw(device, half, green, {0u, 3u, 0u, 1u, 0u, false});
    device.WaitIdle();
    for (std::uint32_t y = 0; y < Extent; ++y) {
        for (std::uint32_t x = 0; x < Extent; ++x) {
            const auto expected = x < Extent / 2u ? 0xff00ff00u : 0xff0000ffu;
            const auto actual = PixelRgba8(linear, x, y);
            Require(actual == expected, "linear target pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + std::to_string(actual) + ", expected " + std::to_string(expected));
        }
    }

    // Cull mode and winding: the triangle is counter-clockwise in Vulkan's framebuffer space, so
    // back-face culling keeps it with a CCW front face and drops it with a CW one.
    std::memset(linear.Data(), 0, linear.Size());
    auto culled = state;
    culled.cullMode = VK_CULL_MODE_BACK_BIT;
    culled.frontFace = VK_FRONT_FACE_CLOCKWISE;
    Draw(device, culled, red, {0u, 3u, 0u, 1u, 0u, false});
    device.WaitIdle();
    Require(PixelRgba8(linear, 5u, 5u) == 0u, "a counter-clockwise triangle survived back-face culling with a clockwise front face");
    culled.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    Draw(device, culled, red, {0u, 3u, 0u, 1u, 0u, false});
    device.WaitIdle();
    Require(PixelRgba8(linear, 5u, 5u) == 0xff0000ffu, "a counter-clockwise triangle was culled as a back face with a counter-clockwise front face");

    // Tiled RenderTarget-mode RGBA16F: loaded from guest memory (the untouched half keeps its
    // bytes), written back retiled when the batch completes.
    const AgcDriver::Graphics::ColorTargetLayout layout(Extent, Extent, ColorTileMode::RenderTarget, 8u);
    GuestBlock tiled(layout.Bytes());
    std::vector<std::byte> linearBytes(static_cast<std::size_t>(Extent) * Extent * 8u);
    const std::uint16_t quarter = 0x3400u;
    for (std::size_t texel = 0; texel < static_cast<std::size_t>(Extent) * Extent; ++texel) {
        for (std::size_t component = 0; component < 4u; ++component) std::memcpy(linearBytes.data() + texel * 8u + component * 2u, &quarter, 2u);
    }
    layout.Tile(linearBytes, std::span<std::byte>(tiled.Data(), layout.Bytes()));
    auto tiledState = BaseState();
    SetTarget(tiledState, tiled.Address(), VK_FORMAT_R16G16B16A16_SFLOAT, ColorTileMode::RenderTarget, 8u);
    tiledState.scissor = {{0, 0}, {Extent, Extent / 2u}};
    Draw(device, tiledState, red, {0u, 3u, 0u, 1u, 0u, false});
    device.WaitIdle();
    StoreResults(device, tiled.Address(), layout.Bytes());
    layout.Detile(std::span<const std::byte>(tiled.Data(), layout.Bytes()), linearBytes);
    for (std::uint32_t y = 0; y < Extent; ++y) {
        for (std::uint32_t x = 0; x < Extent; ++x) {
            std::array<std::uint16_t, 4> texel;
            std::memcpy(texel.data(), linearBytes.data() + (static_cast<std::size_t>(y) * Extent + x) * 8u, 8u);
            const std::array<std::uint16_t, 4> expected = y < Extent / 2u ? std::array<std::uint16_t, 4>{0x3c00u, 0u, 0u, 0x3c00u} : std::array<std::uint16_t, 4>{quarter, quarter, quarter, quarter};
            Require(texel == expected, "tiled target texel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + std::to_string(texel[0]) + " " + std::to_string(texel[1]) + " " + std::to_string(texel[2]) + " " + std::to_string(texel[3]));
        }
    }
    // A second pass reloads nothing stale: green over the lower half keeps the red upper half.
    tiledState.scissor = {{0, static_cast<std::int32_t>(Extent / 2u)}, {Extent, Extent / 2u}};
    Draw(device, tiledState, green, {0u, 3u, 0u, 1u, 0u, false});
    device.WaitIdle();
    StoreResults(device, tiled.Address(), layout.Bytes());
    layout.Detile(std::span<const std::byte>(tiled.Data(), layout.Bytes()), linearBytes);
    std::array<std::uint16_t, 4> top;
    std::array<std::uint16_t, 4> bottom;
    std::memcpy(top.data(), linearBytes.data(), 8u);
    std::memcpy(bottom.data(), linearBytes.data() + (static_cast<std::size_t>(Extent - 1u) * Extent) * 8u, 8u);
    Require(top == std::array<std::uint16_t, 4>{0x3c00u, 0u, 0u, 0x3c00u} && bottom == std::array<std::uint16_t, 4>{0u, 0x3c00u, 0u, 0x3c00u}, "the second tiled pass lost the first pass's texels");
    std::puts("targets: linear in place, scissor, continued pass, culling/winding, tiled load and write-back passed");
}

void CheckPrimitives(AgcDriver::VulkanDevice& device) {
    GuestBlock vertices(16384);
    GuestBlock target(Extent * Extent * 4u);
    // Depth: a near triangle (z 0.25) on the left half first, then a full-screen one at z 0.5 that
    // fails the LESS test there.
    constexpr std::array<std::array<float, 4>, 6> twoDepths {{{-1.0f, -1.0f, 0.5f, 0.0f}, {3.0f, -1.0f, 0.5f, 0.0f}, {-1.0f, 3.0f, 0.5f, 0.0f}, {-1.0f, -1.0f, 0.25f, 0.0f}, {3.0f, -1.0f, 0.25f, 0.0f}, {-1.0f, 3.0f, 0.25f, 0.0f}}};
    WriteVertices(vertices, twoDepths);
    auto state = BaseState();
    SetTarget(state, target.Address(), VK_FORMAT_R8G8B8A8_UNORM, ColorTileMode::Linear, 4u);
    std::vector<float> depthMemory(Extent * Extent, 0.0f);
    state.depth = AgcDriver::Graphics::DepthTarget {reinterpret_cast<std::uint64_t>(depthMemory.data()), 0u, {Extent, Extent}, VK_FORMAT_D32_SFLOAT, 1.0f, 0u};
    state.depthTest = true;
    state.depthWrite = true;
    state.depthCompare = VK_COMPARE_OP_LESS;
    const auto blue = MakeShaders(vertices.Address(), 6u, 0xffff0000u);
    const auto white = MakeShaders(vertices.Address(), 6u, 0xffffffffu);
    auto left = state;
    left.scissor = {{0, 0}, {Extent / 2u, Extent}};
    AgcDriver::Pm4::DrawParameters nearDraw {0u, 3u, 0u, 1u, 0u, false};
    nearDraw.firstVertex = 3u;
    Draw(device, left, blue, nearDraw);
    Draw(device, state, white, {0u, 3u, 0u, 1u, 0u, false});
    device.WaitIdle();
    Require(PixelRgba8(target, 2u, 2u) == 0xffff0000u && PixelRgba8(target, Extent - 2u, 2u) == 0xffffffffu, "the depth test did not keep the nearer triangle (" + std::to_string(PixelRgba8(target, 2u, 2u)) + ", " + std::to_string(PixelRgba8(target, Extent - 2u, 2u)) + ")");

    // Indexed triangle fan (16-bit indices): a quad over the right half from 4 vertices.
    std::memset(target.Data(), 0, target.Size());
    constexpr std::array<std::array<float, 4>, 4> quad {{{0.0f, -1.0f, 0.5f, 0.0f}, {1.0f, -1.0f, 0.5f, 0.0f}, {1.0f, 1.0f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.5f, 0.0f}}};
    WriteVertices(vertices, quad);
    GuestBlock indices(16384);
    const std::array<std::uint16_t, 4> fanIndices {0u, 1u, 2u, 3u};
    std::memcpy(indices.Data(), fanIndices.data(), sizeof(fanIndices));
    auto fan = BaseState();
    SetTarget(fan, target.Address(), VK_FORMAT_R8G8B8A8_UNORM, ColorTileMode::Linear, 4u);
    fan.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    const auto quadShaders = MakeShaders(vertices.Address(), 4u, 0xff0000ffu);
    Draw(device, fan, quadShaders, {indices.Address(), 4u, 2u, 1u, 0u, true});
    device.WaitIdle();
    Require(PixelRgba8(target, Extent - 2u, 2u) == 0xff0000ffu && PixelRgba8(target, Extent - 2u, Extent - 2u) == 0xff0000ffu && PixelRgba8(target, 2u, 2u) == 0u, "the indexed triangle fan did not cover exactly the right half");

    // CPU-read indirect draw (two records, the first empty) plus instancing: the instance at w
    // = 1 shifts the half-width quad's triangles by one unit.
    std::memset(target.Data(), 0, target.Size());
    constexpr std::array<std::array<float, 4>, 3> leftHalf {{{-1.0f, -1.0f, 0.5f, 1.0f}, {1.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};
    WriteVertices(vertices, leftHalf);
    GuestBlock arguments(16384);
    const std::array<std::uint32_t, 8> records {0u, 1u, 0u, 0u, 3u, 2u, 0u, 0u};
    std::memcpy(arguments.Data(), records.data(), sizeof(records));
    AgcDriver::Pm4::DrawParameters indirect {0u, 0u, 0u, 0u, 0u, false};
    AgcDriver::Pm4::DrawParameters::IndirectDraw args {};
    args.arguments = arguments.Address();
    args.opcode = 0x2cu;
    args.recordBytes = 16u;
    args.stride = 16u;
    args.count = 2u;
    args.vertexRule = AgcDriver::Pm4::DrawParameters::IndirectDraw::Rule::InPlace;
    args.instanceRule = AgcDriver::Pm4::DrawParameters::IndirectDraw::Rule::InPlace;
    indirect.indirect = args;
    auto plain = BaseState();
    SetTarget(plain, target.Address(), VK_FORMAT_R8G8B8A8_UNORM, ColorTileMode::Linear, 4u);
    const auto shifted = MakeShaders(vertices.Address(), 3u, 0xff00ff00u);
    Draw(device, plain, shifted, indirect);
    device.WaitIdle();
    Require(PixelRgba8(target, 2u, 2u) == 0xff00ff00u && PixelRgba8(target, Extent - 2u, 2u) == 0xff00ff00u, "the indirect draw's second instance did not reach the right half");
    std::puts("primitives: depth test, indexed fan, CPU indirect, instancing passed");
}

// Headless presentation: a 32x16 source with four colored quadrants presented into a 64x48
// offscreen target is scaled 2x and centered vertically (8 black rows above and below).
void CheckOffscreen(AgcDriver::VulkanDevice& device) {
    Require(device.Presentable() && device.Window() != nullptr, "a window without a Metal layer is not presentable headless");
    constexpr std::uint32_t Width = 32u;
    constexpr std::uint32_t Height = 16u;
    constexpr std::array<std::uint32_t, 4> quadrants {0xffff0000u, 0xff00ff00u, 0xff0000ffu, 0xffffffffu};
    std::vector<std::byte> pixels(Width * Height * 4u);
    for (std::uint32_t y = 0; y < Height; ++y) {
        for (std::uint32_t x = 0; x < Width; ++x) {
            const auto color = quadrants[(y >= Height / 2u ? 2u : 0u) + (x >= Width / 2u ? 1u : 0u)];
            std::memcpy(pixels.data() + (static_cast<std::size_t>(y) * Width + x) * 4u, &color, 4u);
        }
    }
    device.PresentPixels(Width, Height, pixels);
    device.WaitIdle();
    auto frame = AgcDriver::Metal::LastPresentedFrame();
    Require(frame.frames == 1u && frame.width == 64u && frame.height == 48u, "the headless present read back " + std::to_string(frame.frames) + " frames of " + std::to_string(frame.width) + "x" + std::to_string(frame.height));
    const auto at = [&](std::uint32_t x, std::uint32_t y) {
        std::uint32_t value;
        std::memcpy(&value, frame.pixels.data() + (static_cast<std::size_t>(y) * frame.width + x) * 4u, 4u);
        return value;
    };
    Require(at(32u, 2u) == 0xff000000u && at(32u, 45u) == 0xff000000u, "the letterbox rows are not opaque black");
    Require(at(8u, 14u) == quadrants[0] && at(56u, 14u) == quadrants[1] && at(8u, 34u) == quadrants[2] && at(56u, 34u) == quadrants[3], "the scaled quadrants are wrong (" + std::to_string(at(8u, 14u)) + " " + std::to_string(at(56u, 14u)) + " " + std::to_string(at(8u, 34u)) + " " + std::to_string(at(56u, 34u)) + ")");
    // A clear-only present; the slot retires through FinishPresent.
    Require(device.PresentClear(Width, Height, true), "a headless clear was not presented");
    device.FinishPresent();
    device.QueuePresent();
    frame = AgcDriver::Metal::LastPresentedFrame();
    Require(frame.frames == 2u && at(8u, 14u) == 0xff000000u && at(56u, 34u) == 0xff000000u, "the clear-only present did not clear the target");
    Require(AgcDriver::VulkanDevice::PresentCounts().presents >= 2u, "presents were not counted");
    std::puts("offscreen: aspect-fit scaling, letterbox, clear, slot retirement passed");
}

void NoDrawableSize(void*, std::uint32_t* width, std::uint32_t* height) {
    *width = 64u;
    *height = 48u;
}

}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "targets";
    std::unique_ptr<AgcDriver::VulkanDevice> device;
    static int windowToken = 0;
    AgcDriver::PresentationWindow window {&windowToken, {}, nullptr, &NoDrawableSize, 64u, 48u, nullptr};
    try {
        device = mode == "offscreen" ? std::make_unique<AgcDriver::VulkanDevice>(&window) : std::make_unique<AgcDriver::VulkanDevice>();
    } catch (const std::exception& error) {
        std::printf("skipped, no usable Metal device: %s\n", error.what());
        return Skipped;
    }
    try {
        if (mode == "targets") {
            CheckTargets(*device);
        } else if (mode == "primitives") {
            CheckPrimitives(*device);
        } else if (mode == "offscreen") {
            CheckOffscreen(*device);
        } else {
            std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 2;
        }
        device->WaitIdle();
        std::printf("metal draw %s tests passed\n", mode.c_str());
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
