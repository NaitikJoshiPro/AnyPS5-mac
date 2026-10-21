// Metal twin of tests/execution/MultisampledPassOrder.cpp (agc_driver_multisampled_pass_order), swapped in by
// twins/SKIPFIX5-msaa.cmake. Upstream draws twice into one four-sample color target while switching depth targets
// and captures vkCmdPipelineBarrier through Context.deviceProc: the draw back on the first depth target must record a
// barrier ordering the first draw's color writes before it. Metal has no vkCmdPipelineBarrier to capture. Its form
// of that order: a draw whose depth target differs from the open pass's begins a new render encoder, and every render
// encoder begins with a queue barrier after all stages of the work before it (MetalCommandStream.mm BeginRender), so
// the order is kept exactly when no such draw continues an earlier pass. The twin draws the same sequence through the
// Metal device (MetalDraw.mm, the device's own recorder) and counts the render encoders of the open batch: a draw on
// the same targets continues the pass (one encoder), each depth switch begins one. Like upstream, the multisampled
// color is not resolved to guest memory; the draws then complete on the GPU.

#include "../src/MetalCommandStream.h"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "../../../tests/execution/DepthFastClearHarness.hpp"
#include "../../../tests/execution/VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>

namespace {

using namespace AgcDriver::Graphics;
using DepthFastClearHarness::Compile;
using DepthFastClearHarness::Height;
using DepthFastClearHarness::Shaders;
using DepthFastClearHarness::Triangles;
using DepthFastClearHarness::Width;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Samples = 4;

alignas(4096) std::array<std::byte, Width * Height * 4 * Samples> Multisampled{};
alignas(4096) std::array<float, Width * Height> DepthA{};
alignas(4096) std::array<float, Width * Height> DepthB{};

void Draw(AgcDriver::VulkanDevice& device, const Shaders& compiled, std::uint64_t depth) {
    const std::array<CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &compiled.vertex, 0},
        {ShaderStage::Fragment, &compiled.pixel, PixelPushOffset(compiled.vertexPush, device.Target())}
    }};
    State state{};
    state.stages = {ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Multisampled.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Multisampled.size(), 0xe4u};
    state.color.samples = Samples;
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.depth = DepthTarget{depth, 0, {Width, Height}, VK_FORMAT_D32_SFLOAT, 1.0f, 0, 0, false, Samples};
    state.depthTest = true;
    state.depthWrite = true;
    state.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL;
    state.samples = Samples;
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
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Triangles().size()), 0, 1, 0, false};
    device.Draw(state, draw, shaders);
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = device->graphicsContext();
        Require((context.limits.framebufferColorSampleCounts & Samples) != 0 && (context.limits.framebufferDepthSampleCounts & Samples) != 0, "the Metal device does not report four-sample color and depth targets");
        Require(context.recorder != nullptr, "the Metal device has no recorder");
        const auto shaders = Compile(*device);
        device->WaitIdle();
        const auto depthA = reinterpret_cast<std::uintptr_t>(DepthA.data());
        const auto depthB = reinterpret_cast<std::uintptr_t>(DepthB.data());
        // The depth surfaces are made (and cleared) by a first draw of each, outside the counted sequence.
        Draw(*device, shaders, depthA);
        Draw(*device, shaders, depthB);
        device->WaitIdle();
        std::uint64_t encoders = 0;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            // Commands() opens the batch and ends any open pass, so every draw below starts from no pass.
            auto& stream = Metal::Stream(context.recorder->Commands());
            const auto before = stream.Encoders();
            Draw(*device, shaders, depthA);
            Draw(*device, shaders, depthA);
            Draw(*device, shaders, depthB);
            Draw(*device, shaders, depthA);
            auto& after = Metal::Stream(context.recorder->Commands());
            Require(&after == &stream, "the recorder submitted its batch inside the counted draws");
            encoders = after.Encoders() - before;
        }
        device->WaitIdle();
        std::printf("four-sample draws on depth A, A, B, A began %llu render encoders\n", static_cast<unsigned long long>(encoders));
        Require(encoders == 3u, "multisampled pass order: draws sharing a multisampled color target began " + std::to_string(encoders) + " render encoders, expected 3 (the second draw on depth A continues the first's pass; the switch to depth B and the draw back on depth A each begin a pass, whose queue barrier orders the earlier color writes before it)");
        std::puts("multisampled pass order tests passed (ordering asserted on the render encoders, each begun after a queue barrier; the multisampled image is not resolved to guest memory)");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
