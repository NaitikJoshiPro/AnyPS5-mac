// agc_driver_stage_image_bindings: a draw whose vertex and pixel programs both read guest images, each
// from its own T#. The vertex program loads texel (0, 0) of its image and adds it to the clip x of
// every vertex (shifting the triangle right); the pixel program loads texel (0, 0) of its image and
// exports it to all four channels. Both stages' image bindings must come from their own descriptor
// build: the Metal device once wrote the pixel function's entries from the vertex function's build
// and threw "Metal descriptors (work package D5): the build has no binding 66" (ASTRO BOT's
// worldmap; DECISIONS D76). Checked: the shifted triangle's coverage and the pixel image's value.

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <sys/mman.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::PixelInput;
using ShaderRecompiler::PixelInputBit;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 48;
constexpr float Unwritten = -7.0f;
constexpr float Shift = 0.75f;          // the vertex image's texel: clip x added to every vertex
constexpr float PixelValue = 0.3125f;   // the pixel image's texel
constexpr std::uint32_t Format32Float = 22;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t ImageWidth = 64;
alignas(256) std::array<float, Width * Height * 4> Pixels{};

// buffer_load_format_xyzw v[0:3] by the vertex index (v5) from s[0:3]; v7 = v8 = 0; image_load v6,
// v[7:8], s[4:11] dmask:0x1 unorm dim 2D; v_add_f32 v0, v6, v0; param0 = pos0 = v[0:3].
alignas(256) constexpr std::array<std::uint32_t, 15> VertexCode{
    0xe0382000, 0x80000005,
    0x7e0e0280, 0x7e100280,
    0xf0001108, 0x00010607,
    0xbf8c3f70,
    0x06000106,
    0xf800020f, 0x03020100,
    0xf80008cf, 0x03020100,
    0xbf810000, 0xbf810000, 0xbf810000,
};

// v0 = v1 = 0; image_load v2, v[0:1], s[0:7] dmask:0x1 unorm dim 2D; exp mrt0 v2 v2 v2 v2 done.
alignas(256) constexpr std::array<std::uint32_t, 9> PixelCode{
    0x7e000280, 0x7e020280,
    0xf0001108, 0x00000200,
    0xbf8c3f70,
    0xf800180f, 0x02020202,
    0xbf810000, 0xbf810000,
};

struct Vertex {
    float x, y, z, w;
};

// A triangle in the left half of the target; shifted by `Shift` it covers the middle.
alignas(256) constexpr std::array<Vertex, 3> Corners{{{-0.9f, -0.8f, 0.5f, 1.0f}, {-0.3f, -0.8f, 0.5f, 1.0f}, {-0.6f, 0.8f, 0.5f, 1.0f}}};

class GuestImages {
public:
    GuestImages() {
        void* mapped = mmap(nullptr, Bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        Require(mapped != MAP_FAILED, "stage image bindings: cannot map the images");
        block = static_cast<std::uint8_t*>(mapped);
        GuestAllocations::Mutation().Add(block, Bytes, true, true, true);
    }
    ~GuestImages() {
        GuestAllocations::Mutation().Remove(block);
        munmap(block, Bytes);
    }
    GuestImages(const GuestImages&) = delete;
    GuestImages& operator=(const GuestImages&) = delete;
    float* Image(std::uint32_t index) { return reinterpret_cast<float*>(block + index * ImageBytes); }

    static constexpr std::size_t ImageBytes = 65536;
    static constexpr std::size_t Bytes = 2 * ImageBytes;

private:
    std::uint8_t* block = nullptr;
};

std::array<std::uint32_t, 8> TextureDescriptor(const void* data) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    constexpr std::uint32_t height = 1;
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32Float << 20u) | (((ImageWidth - 1u) & 3u) << 30u),
        ((ImageWidth - 1u) >> 2u) | ((height - 1u) << 14u),
        0xfacu | (Type2D << 28u),
        0u, 0u, 0u, 0u,
    };
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, 0x01016facu};
}

// Whether the pixel centre (x + 0.5, y + 0.5) lies inside the triangle shifted by `shift` in NDC x,
// with a margin of `margin` pixels to every edge.
bool Inside(std::uint32_t x, std::uint32_t y, float shift, double margin) {
    std::array<std::array<double, 2>, 3> window{};
    for (std::size_t i = 0; i < 3; ++i) window[i] = {(Corners[i].x + shift + 1.0) * Width / 2.0, (1.0 - Corners[i].y) * Height / 2.0};
    const double px = x + 0.5;
    const double py = y + 0.5;
    const auto edge = [](const std::array<double, 2>& a, const std::array<double, 2>& b, double cx, double cy) { return (b[0] - a[0]) * (cy - a[1]) - (b[1] - a[1]) * (cx - a[0]); };
    const double area = edge(window[0], window[1], window[2][0], window[2][1]);
    for (std::size_t i = 0; i < 3; ++i) {
        const auto& a = window[i];
        const auto& b = window[(i + 1) % 3];
        const double length = std::hypot(b[0] - a[0], b[1] - a[1]);
        if (edge(a, b, px, py) / area * std::fabs(area) / length < margin) return false;
    }
    return true;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestImages images;
        std::fill(images.Image(0), images.Image(0) + ImageWidth, Shift);
        std::fill(images.Image(1), images.Image(1) + ImageWidth, PixelValue);
        Pixels.fill(Unwritten);
        const auto target = device->Target();

        std::vector<std::uint32_t> vertexUserData;
        const auto vertexBuffer = BufferDescriptor(Corners.data(), sizeof(Vertex), static_cast<std::uint32_t>(Corners.size()));
        const auto vertexImage = TextureDescriptor(images.Image(0));
        vertexUserData.insert(vertexUserData.end(), vertexBuffer.begin(), vertexBuffer.end());
        vertexUserData.insert(vertexUserData.end(), vertexImage.begin(), vertexImage.end());
        const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
        ShaderRecompiler::RecompileRequest vertex{
            {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
            {64u, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
            target,
            {0, 0, 0, 64}
        };
        vertex.useCache = false;
        const auto vertexResult = ShaderRecompiler::Recompile(vertex);
        const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());

        const auto pixelImage = TextureDescriptor(images.Image(1));
        const std::vector<std::uint32_t> pixelUserData(pixelImage.begin(), pixelImage.end());
        ShaderRecompiler::ShaderPixelStageInfo pixel{};
        pixel.interpolatorCount = 1;
        pixel.inputAddr = PixelInputBit(PixelInput::PerspectiveCenter);
        pixel.hasPerspectiveCenterVgpr = true;
        pixel.executeOnNoop = true;
        pixel.targetOutputMode[0] = 9;
        pixel.targetExportMapping.fill(0xe4u);
        const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
        ShaderRecompiler::RecompileRequest fragment{
            {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
            {64u, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
            target,
            PixelPushLayout(vertexPush, target)
        };
        fragment.useCache = false;
        const auto pixelResult = ShaderRecompiler::Recompile(fragment);
        const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
            {ShaderStage::Vertex, &vertexResult, 0},
            {ShaderStage::Fragment, &pixelResult, PixelPushOffset(vertexPush, target)},
        }};

        AgcDriver::Graphics::State state{};
        state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
        state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(Pixels), 0xe4u, AgcDriver::Graphics::ColorTileMode::Linear, 16u};
        state.colors = {state.color};
        state.hasColorTarget = true;
        state.renderExtent = {Width, Height};
        state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
        state.negativeOneToOne = false;
        state.scissor = {{0, 0}, {Width, Height}};
        state.cullMode = VK_CULL_MODE_NONE;
        state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        state.blend.colorWriteMask = 15;
        state.blends = {state.blend};
        state.blendConstants = {};
        const AgcDriver::Pm4::DrawParameters draw{0, 3u, 0, 1, 0, false};
        device->Draw(state, draw, shaders);
        device->WaitIdle();

        std::uint32_t inside = 0;
        std::uint32_t outside = 0;
        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = 0; x < Width; ++x) {
                const float* value = &Pixels[(y * Width + x) * 4u];
                const std::string at = "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
                if (Inside(x, y, Shift, 0.75)) {
                    for (std::uint32_t channel = 0; channel < 4; ++channel) Require(value[channel] == PixelValue, "pixel " + at + " of the shifted triangle reads " + std::to_string(value[channel]) + " in channel " + std::to_string(channel) + ", expected the pixel image's " + std::to_string(PixelValue));
                    ++inside;
                } else if (Inside(x, y, 0.0f, -0.75) && !Inside(x, y, Shift, -0.75)) {
                    Require(value[0] == Unwritten, "pixel " + at + " of the unshifted triangle was drawn: the vertex program did not read its own image");
                    ++outside;
                }
            }
        }
        Require(inside > 50u && outside > 50u, "stage image bindings: too few pixels checked (" + std::to_string(inside) + " inside, " + std::to_string(outside) + " outside)");
        std::printf("stage image bindings: the vertex image shifted the triangle (%u pixels left unwritten) and %u pixels read the pixel image\n", outside, inside);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
