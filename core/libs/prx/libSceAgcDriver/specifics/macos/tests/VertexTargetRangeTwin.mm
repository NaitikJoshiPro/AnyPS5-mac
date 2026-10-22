// Metal twin of tests/Recorder.cpp's --vertex-target-range-copied and --vertex-target-range-imported modes
// (agc_driver_vertex_target_range_copied / _imported, vertexTargetRangeTests, upstream fdefd7660). Upstream draws a
// float colour target (the producer), then a draw whose vertex records start inside that target (firstVertex or
// firstInstance past a descriptor based at the consumer's own colour target) and checks that the consumer reads
// the GPU-produced offset, that a reused recipe sees a new producer value, and that ranges which really alias the
// render target (indirect, firstVertex 0, zero stride, an index buffer inside it) are refused.
// Metal differences:
// - The draws go through the Metal device (MetalDraw.mm), whose render-target alias check
//   (CheckInputsClearOfTarget) ports upstream's prepareDrawInputs one; the shaders are hand-written MSL
//   against MslAbi (as D6DrawTests.mm), standing in for upstream's GLSL VertexRange/VertexRangeProducer/Cover.
// - Metal has no draw recipes (TechnicalDebt metal-d6-bring-up-draws): the "recipe reuse" step is a second
//   draw with the same parameters after the producer changed, which must read the new value.
// - Copied and imported: Metal draws read guest vertex and index memory in place (no-copy wrappers or host
//   imports, DECISIONS D19), so both modes run the same checks; the mode only names the upstream variant.

#include "RecorderTwins.h"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

namespace Abi = ShaderRecompiler::MslAbi;
namespace RuntimeAbi = ShaderRecompiler::RuntimeAbi;
using AgcDriver::Graphics::ColorTileMode;

constexpr std::size_t Unit = 65536;
constexpr std::size_t Bytes = Unit * 3;
constexpr std::uint32_t Extent = 64u;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Cover.vert: a triangle over the whole target.
const char* CoverSource = R"(#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
vertex VertexOut aps5_main(uint vertexId [[vertex_id]], constant uint* push [[buffer(1)]]) {
    const float2 position = float2(float((vertexId << 1u) & 2u), float(vertexId & 2u));
    VertexOut out;
    out.position = float4(position * 2.0f - 1.0f, 0.5f, 1.0f);
    return out;
}
)";

// VertexRangeProducer.frag: (value, 0, 0, 1) with value from push dword 32 (0.5, or 0.75 for the second producer).
const char* ProducerSource = R"(#include <metal_stdlib>
using namespace metal;
struct FragmentOut { float4 color [[color(0)]]; };
fragment FragmentOut aps5_main(constant uint* push [[buffer(1)]]) {
    FragmentOut out;
    out.color = float4(as_type<float>(push[32]), 0.0f, 0.0f, 1.0f);
    return out;
}
)";

// VertexRange.vert: the big triangle shifted right by the x of location 0, pulled through the VertexFetch slot
// (vertex_id and instance_id include the draw's first vertex and first instance, as gl_VertexIndex does).
const char* VertexRangeSource = R"(#include <metal_stdlib>
using namespace metal;
struct Fetch { ulong address; uint stride; uint bytes; uint word3; uint rate; uint components; uint reserved; };
struct VertexOut { float4 position [[position]]; };
vertex VertexOut aps5_main(uint vertexId [[vertex_id]], uint instanceId [[instance_id]], device const Fetch* fetch [[buffer(4)]], constant uint* push [[buffer(1)]]) {
    const uint index = fetch[0].rate == 0u ? vertexId : instanceId;
    const ulong offset = fetch[0].stride == 0u ? 0ul : ulong(index) * fetch[0].stride;
    const float4 value = offset + 16u <= fetch[0].bytes ? *reinterpret_cast<device const float4*>(fetch[0].address + offset) : float4(0.0f);
    const float2 positions[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    VertexOut out;
    out.position = float4(positions[vertexId % 3u] + float2(value.x, 0.0f), 0.5f, 1.0f);
    return out;
}
)";

// Cover.frag: white.
const char* WhiteSource = R"(#include <metal_stdlib>
using namespace metal;
struct FragmentOut { float4 color [[color(0)]]; };
fragment FragmentOut aps5_main(constant uint* push [[buffer(1)]]) {
    FragmentOut out;
    out.color = float4(1.0f);
    return out;
}
)";

ShaderRecompiler::RecompileResult VertexProgram(const char* source, bool fetches) {
    Abi::Reflection reflection;
    reflection.stage = Abi::Stage::Vertex;
    reflection.group = RuntimeAbi::Stage::Main;
    reflection.vertexFetchLocations = fetches ? 1u : 0u;
    ShaderRecompiler::RecompileResult result;
    result.msl = ShaderRecompiler::SharedMsl(ShaderRecompiler::MslShader{source, reflection});
    return result;
}

ShaderRecompiler::RecompileResult FragmentProgram(const char* source, float value) {
    Abi::Reflection reflection;
    reflection.stage = Abi::Stage::Fragment;
    reflection.group = RuntimeAbi::Stage::Fragment;
    reflection.colorOutputs = 1u;
    reflection.usesPushData = true;
    ShaderRecompiler::RecompileResult result;
    result.msl = ShaderRecompiler::SharedMsl(ShaderRecompiler::MslShader{source, reflection});
    result.pushConstants.resize(4u);
    std::memcpy(result.pushConstants.data(), &value, 4u);
    return result;
}

ShaderRecompiler::VertexAttribute AttributeFor(std::uint64_t base, std::uint32_t stride, std::uint32_t fetchIndex, std::uint32_t outOfBoundsSelect = 0u) {
    ShaderRecompiler::VertexAttribute attribute{};
    attribute.location = 0u;
    attribute.components = 4u;
    attribute.formatComponents = 4u;
    attribute.fetchIndex = fetchIndex;
    // Format 77 (32_32_32_32 float), dst_sel xyzw; Bytes / 16 records as upstream.
    attribute.resource.fields = {static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(base >> 32u) | (stride << 16u), static_cast<std::uint32_t>(Bytes / 16u), 0x00000facu | (77u << 12u) | (outOfBoundsSelect << 28u)};
    return attribute;
}

AgcDriver::Graphics::State StateFor(std::uint64_t base, VkFormat format, ColorTileMode mode, std::uint32_t elementBytes) {
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    const AgcDriver::Graphics::ColorTargetLayout layout(Extent, Extent, mode, elementBytes);
    state.color = {base, {Extent, Extent}, format, layout.Bytes(), 0xe4u};
    state.color.tileMode = mode;
    state.color.elementBytes = elementBytes;
    state.colors = {state.color};
    state.hasColorTarget = true;
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
    return state;
}

int Run(bool imported) {
    std::unique_ptr<AgcDriver::VulkanDevice> device;
    try {
        device = std::make_unique<AgcDriver::VulkanDevice>();
    } catch (const std::exception& error) {
        std::printf("skipped, no usable Metal device: %s\n", error.what());
        return 77;
    }
    void* raw = std::aligned_alloc(Unit, Bytes);
    Require(raw != nullptr, "cannot allocate the guest block");
    auto* block = static_cast<std::byte*>(raw);
    std::memset(block, 0, Bytes);
    GuestAllocations::Mutation().Add(block, Bytes, true, true, true);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    int result = 0;
    try {
        auto producerVertex = VertexProgram(CoverSource, false);
        auto vertex = VertexProgram(VertexRangeSource, true);
        const auto pixel = FragmentProgram(WhiteSource, 0.0f);
        // Producer: RGBA32F, RenderTarget-tiled (a resident image written back when a reader flushes it), at
        // address + Unit. Consumer: linear RGBA8 at the start of the block, which the vertex descriptors also start at.
        const auto producer = StateFor(address + Unit, VK_FORMAT_R32G32B32A32_SFLOAT, ColorTileMode::RenderTarget, 16u);
        const auto consumer = StateFor(address, VK_FORMAT_R8G8B8A8_UNORM, ColorTileMode::Linear, 4u);
        const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 1, 0, false};
        const auto drawWith = [&](const AgcDriver::Graphics::State& state, const ShaderRecompiler::RecompileResult& first, const ShaderRecompiler::RecompileResult& second, const AgcDriver::Pm4::DrawParameters& draw) {
            const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{{ShaderRecompiler::ShaderStage::Vertex, &first, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &second, 128u}}};
            device->Draw(state, draw, shaders);
        };
        const auto produce = [&](float value) {
            const auto producerPixel = FragmentProgram(ProducerSource, value);
            drawWith(producer, producerVertex, producerPixel, triangle);
            const std::array range{std::pair{address + Unit, address + 2 * Unit}};
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            Require(AgcDriver::Graphics::StorageTexture::AnyPendingOverlaps(range), "the color producer did not leave GPU results pending");
        };
        const auto check = [&](std::uint32_t edge, const std::string& what) {
            device->WaitIdle();
            {
                std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
                AgcDriver::Graphics::StorageTexture::FlushPending(address, Unit, nullptr, "vertex target range test");
            }
            device->WaitIdle();
            for (std::uint32_t y = 0; y < Extent; ++y) {
                for (std::uint32_t x = 0; x < Extent; ++x) {
                    const auto expected = x < edge ? std::byte{0} : std::byte{255};
                    for (std::uint32_t channel = 0; channel < 4; ++channel) {
                        const auto actual = block[(static_cast<std::size_t>(y) * Extent + x) * 4u + channel];
                        Require(actual == expected, what + ": the draw did not consume the GPU-produced vertex offset (texel " + std::to_string(x) + ", " + std::to_string(y) + " channel " + std::to_string(channel) + " is " + std::to_string(std::to_integer<unsigned>(actual)) + ")");
                    }
                }
            }
        };
        const auto clearConsumer = [&] {
            device->WaitIdle();
            std::memset(block, 0, Unit);
        };
        const auto reject = [&](auto action, const char* expected) {
            try {
                action();
            } catch (const std::runtime_error& error) {
                Require(std::string(error.what()).find(expected) != std::string::npos, error.what());
                return;
            }
            throw std::runtime_error(std::string("an actual render-target alias was accepted (expected: ") + expected + ")");
        };
        for (const std::uint32_t fetchIndex : {0u, 1u}) {
            const std::string rate = fetchIndex == 0 ? "per-vertex" : "per-instance";
            vertex.vertexAttributes = {AttributeFor(address, 16, fetchIndex)};
            auto draw = triangle;
            if (fetchIndex == 0) draw.firstVertex = Unit / 16;
            else draw.firstInstance = Unit / 16;
            clearConsumer();
            produce(0.5f);
            drawWith(consumer, vertex, pixel, draw);
            check(16, rate + " records past the consumer's target");
            // Upstream reuses the draw's recipe here; Metal has none, so the same draw is recorded again.
            clearConsumer();
            produce(0.75f);
            drawWith(consumer, vertex, pixel, draw);
            check(24, rate + " draw after the producer changed");
            clearConsumer();
            vertex.vertexAttributes = {AttributeFor(address + Unit, 16, fetchIndex)};
            drawWith(consumer, vertex, pixel, triangle);
            check(24, rate + " descriptor based at the producer");
            vertex.vertexAttributes = {AttributeFor(address, 16, fetchIndex)};
            auto indirect = draw;
            indirect.indirect.emplace();
            reject([&] { drawWith(consumer, vertex, pixel, indirect); }, "vertex buffer aliases the render target");
            if (fetchIndex == 0) {
                const std::array<std::uint16_t, 3> indices{0, 1, 2};
                std::memcpy(block + 2 * Unit, indices.data(), sizeof(indices));
                auto indexed = draw;
                indexed.indexed = true;
                indexed.indexSize = 2;
                indexed.indexAddress = address + 2 * Unit;
                clearConsumer();
                produce(0.75f);
                drawWith(consumer, vertex, pixel, indexed);
                check(24, "indexed draw with a base vertex past the consumer's target");
                indexed.indexAddress = address;
                reject([&] { drawWith(consumer, vertex, pixel, indexed); }, "index buffer aliases the render target");
            }
            reject([&] { drawWith(consumer, vertex, pixel, triangle); }, "vertex buffer aliases the render target");
            // OOB_SELECT 3 (raw): a zero-stride fetch still reads its record (upstream bea48f5ee0, 58afce1ad1).
            vertex.vertexAttributes = {AttributeFor(address, 0, fetchIndex, 3u)};
            reject([&] { drawWith(consumer, vertex, pixel, draw); }, "vertex buffer aliases the render target");
        }
        // OOB_SELECT 0 with stride 0: RDNA 2's range check reads nothing, so the draw does not alias its target.
        vertex.vertexAttributes = {AttributeFor(address, 0, 0)};
        drawWith(consumer, vertex, pixel, triangle);
        device->WaitIdle();
        std::printf("Vertex range and pending color producer tests passed on Metal (%s: guest inputs are read in place)\n", imported ? "imported" : "copied");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        result = 1;
    }
    try {
        device->WaitIdle();
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        AgcDriver::Graphics::StorageTexture::FlushPending(address, Bytes, nullptr, "vertex target range teardown");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "teardown: %s\n", error.what());
        result = 1;
    }
    device.reset();
    GuestAllocations::Mutation().Remove(block);
    std::free(raw);
    return result;
}

const bool copied = AgcDriver::MetalTwins::RegisterRecorderMode("--vertex-target-range-copied", [] { return Run(false); });
const bool imported = AgcDriver::MetalTwins::RegisterRecorderMode("--vertex-target-range-imported", [] { return Run(true); });

}
