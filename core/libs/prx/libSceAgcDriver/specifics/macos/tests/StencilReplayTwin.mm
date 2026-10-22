// Metal twin of tests/execution/StencilReplay.cpp (agc_driver_stencil_replay and agc_driver_stencil_replay_no_gpl),
// swapped in by twins/UPSYNC3-tests.cmake. The upstream file is compiled here with its main renamed, so its programs,
// Draw and Check keep tracking upstream.
//
// Upstream removes CapabilityStencilExportEXT from the target to force the Vulkan stencil replay (one draw per proven
// reference value with a filtering shader), counts vkCmdBindPipeline calls and inspects the replay SPIR-V. The Metal
// target always exports stencil natively: Metal fragment functions return [[stencil]] (MslStageIo.cpp StencilRef, on
// every Apple GPU family this port runs on), so no replay exists to count or inspect. The macro below turns upstream's
// capability removal into a no-op (CapabilityMax is never listed), so the same Draw compiles the same guest programs
// with native export, and this main runs every observable stencil check of upstream's main against the GPU results.
//
// Left out, replay-only: the vertex subgroup-size contract checks (they guard the replay's re-draws), and the refusals
// of draws the replay cannot express (strips/restart, unbounded references, early or ordered tests, indirect draws, an
// export skipped on a kept path): natively those draws are legal. The unbounded-reference draw is instead checked to
// export each pixel's column as its stencil value, and the strip draw to export the same references as the list.

#include "../../../tests/execution/VulkanTestDevice.hpp"

#define CapabilityStencilExportEXT CapabilityMax
#define main AgcDriverStencilReplayVulkanMain
#include "../../../tests/execution/StencilReplay.cpp"
#undef main
#undef CapabilityStencilExportEXT

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        // Upstream's bind trace and replay reference checks have no Metal input; they stay compiled for its main.
        (void)&CountBind;
        constexpr std::uint32_t exported = 0x52u;
        constexpr std::uint32_t plain = 0x50u;
        constexpr std::array<std::uint32_t, 12> alternating{
            0x7e040f00u, 0x36060481u,
            0x7c000000u | (0xc2u << 17u) | (3u << 9u) | 0x80u,
            0x7e0802ffu, 0xffu,
            (1u << 25u) | (5u << 17u) | (4u << 9u) | 0x80u,
            0x7e0e02f2u, 0xf8000082u, 5u << 8u,
            0xf800180fu, 0x07070707u, 0xbf810000u
        };
        constexpr std::array<std::uint32_t, 10> ordered{
            0x7c000000u | (4u << 17u) | (2u << 9u) | 0xf0u,
            0x7e0802ffu, 0xffu,
            (1u << 25u) | (5u << 17u) | (4u << 9u) | 0x80u,
            0x7e0e02f2u, 0xf8000082u, 5u << 8u,
            0xf800180fu, 0x07070707u, 0xbf810000u
        };
        const auto replace = [](std::uint32_t reference = 0u, std::uint32_t mask = 0xffu) { return VkStencilOpState{VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xffu, mask, reference}; };
        const auto equal = [](std::uint32_t reference, std::uint32_t mask = 0xffu) { return VkStencilOpState{VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL, mask, 0u, reference}; };
        const auto all = [](std::uint32_t, std::uint32_t) { return true; };
        const auto even = [](std::uint32_t x, std::uint32_t) { return x % 2u == 0u; };
        const auto odd = [](std::uint32_t x, std::uint32_t) { return x % 2u != 0u; };
        Draw(*device, alternating, exported, replace());
        Check(all, "finite stencil export");
        Draw(*device, PlainPixelCode, plain, equal(255u));
        Check(even, "even columns exported 255");
        Draw(*device, PlainPixelCode, plain, equal(0u));
        Check(odd, "odd columns exported zero");
        Draw(*device, alternating, exported, equal(91u));
        Check(all, "exported references replace compare reference");
        Draw(*device, PlainPixelCode, plain, replace(0u));
        Draw(*device, ordered, exported, replace(), 6u);
        Draw(*device, PlainPixelCode, plain, equal(0u));
        Check(all, "overlapping primitives retain original order");
        Draw(*device, PlainPixelCode, plain, replace(0u));
        const VkStencilOpState instanceOrder{VK_STENCIL_OP_INCREMENT_AND_WRAP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_GREATER, 0xffu, 0xffu, 0u};
        Draw(*device, ordered, exported, instanceOrder, 6u, 2u);
        Draw(*device, PlainPixelCode, plain, equal(0u));
        Check(all, "primitive order within every instance");
        alignas(256) constexpr std::array<std::uint32_t, 6> indices{0u, 1u, 2u, 3u, 4u, 5u};
        alignas(256) std::array<std::array<float, 4>, 7> indexedVertices{};
        std::copy(Vertices.begin(), Vertices.end(), indexedVertices.begin() + 1);
        Draw(*device, PlainPixelCode, plain, replace(0u));
        Draw(*device, ordered, exported, instanceOrder, 6u, 2u, false, {}, [&](AgcDriver::Pm4::DrawParameters& draw) {
            draw.indexed = true;
            draw.indexAddress = reinterpret_cast<std::uintptr_t>(indices.data());
            draw.indexSize = 4u;
            draw.firstVertex = 1u;
            draw.firstInstance = 7u;
        }, indexedVertices);
        Draw(*device, PlainPixelCode, plain, equal(0u));
        Check(all, "indexed primitives preserve base vertex and nonzero first instance");
        alignas(256) std::array<std::array<float, 4>, 72> instanceVertices{};
        std::copy(Vertices.begin(), Vertices.end(), instanceVertices.begin() + 56);
        std::copy(Vertices.begin() + 3, Vertices.end(), instanceVertices.begin() + 64);
        std::copy(Vertices.begin(), Vertices.begin() + 3, instanceVertices.begin() + 67);
        constexpr std::array<std::uint32_t, 8> instanceVertexCode{0x34101083u, 0x4a0a0b08u, 0xe0382000u, 0x80000005u, 0xbf8c3f70u, 0xf80008cfu, 0x03020100u, 0xbf810000u};
        Draw(*device, PlainPixelCode, plain, replace(0u));
        Draw(*device, ordered, exported, instanceOrder, 6u, 2u, false, {}, [](AgcDriver::Pm4::DrawParameters& draw) { draw.firstInstance = 7u; }, instanceVertices, instanceVertexCode);
        Check(all, "nonzero instance indices select live vertices");
        Draw(*device, PlainPixelCode, plain, equal(255u));
        Check(all, "each instance preserves its observable instance index");
        Draw(*device, PlainPixelCode, plain, replace(0x55u));
        Draw(*device, alternating, exported, replace(0u, 0xaau));
        Draw(*device, PlainPixelCode, plain, equal(0xffu));
        Check(even, "stencil write mask preserves even columns");
        Draw(*device, PlainPixelCode, plain, equal(0x55u));
        Check(odd, "stencil write mask preserves odd columns");
        auto highBits = alternating;
        highBits[4] = 0x1234ffu;
        Draw(*device, highBits, exported, replace());
        Draw(*device, PlainPixelCode, plain, equal(255u));
        Check(even, "stencil reference truncates to eight attachment bits");
        Draw(*device, PlainPixelCode, plain, replace(0u));
        Draw(*device, PlainPixelCode, plain, replace(0u), 3u, 1u, false, [](AgcDriver::Graphics::State& state) { state.cullMode = VK_CULL_MODE_BACK_BIT; });
        const auto facing = Pixels[0] == std::byte{255} ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
        for (const bool back : {false, true}) {
            Draw(*device, PlainPixelCode, plain, replace(0u));
            Draw(*device, alternating, exported, replace(), 3u, 1u, false, [&](AgcDriver::Graphics::State& state) {
                state.frontFace = back ? (facing == VK_FRONT_FACE_COUNTER_CLOCKWISE ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE) : facing;
                state.stencilFront.writeMask = 0xaau;
                state.stencilBack.writeMask = 0x55u;
                state.stencilBack.passOp = VK_STENCIL_OP_INVERT;
            });
            Draw(*device, PlainPixelCode, plain, equal(back ? 0x55u : 0xaau));
            Check(back ? all : even, "distinct front and back stencil operations and write masks");
        }
        Draw(*device, PlainPixelCode, plain, replace(0x0fu));
        Draw(*device, alternating, exported, equal(1u, 0x0fu));
        Check(even, "stencil compare masks exported reference");
        const VkStencilOpState depthFailure{VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_COMPARE_OP_ALWAYS, 0xffu, 0xffu, 0u};
        Draw(*device, alternating, exported, depthFailure, 3u, 1u, true);
        Check([](std::uint32_t, std::uint32_t) { return false; }, "depth failure rejects colors");
        Draw(*device, PlainPixelCode, plain, equal(255u));
        Check(even, "depth-fail replace uses exported reference");
        auto depthStencil = std::vector<std::uint32_t>(alternating.begin(), alternating.end());
        const auto mrtz = std::find(depthStencil.begin(), depthStencil.end(), 0xf8000082u);
        const auto mrtzOffset = static_cast<std::size_t>(mrtz - depthStencil.begin());
        depthStencil.insert(mrtz, {0x7e0c02ffu, 0x3f400000u});
        depthStencil[mrtzOffset + 2u] = 0xf8000083u;
        depthStencil[mrtzOffset + 3u] = (5u << 8u) | 6u;
        Draw(*device, depthStencil, 0x53u, replace(), 3u, 1u, false, [](AgcDriver::Graphics::State& state) {
            state.depthTest = true;
            state.depthWrite = true;
            state.depthCompare = VK_COMPARE_OP_ALWAYS;
        });
        auto midDepth = Vertices;
        for (auto& vertex : midDepth) vertex[2] = 0.5f;
        Draw(*device, PlainPixelCode, plain, replace(), 3u, 1u, false, [](AgcDriver::Graphics::State& state) {
            state.depthTest = true;
            state.depthCompare = VK_COMPARE_OP_LESS;
        }, {}, midDepth);
        Check(all, "simultaneous depth and stencil exports preserve shader depth replacement");
        // Native export draws what the replay refuses: an unbounded reference (each pixel's column) and a strip.
        Draw(*device, ColumnPixelCode, exported, replace());
        Draw(*device, PlainPixelCode, plain, equal(37u));
        Check([](std::uint32_t x, std::uint32_t) { return x == 37u; }, "an unbounded exported reference (the pixel column) reaches the attachment");
        Draw(*device, PlainPixelCode, plain, replace(0u));
        Draw(*device, alternating, exported, replace(), 3u, 1u, false, [](AgcDriver::Graphics::State& state) { state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; });
        Draw(*device, PlainPixelCode, plain, equal(255u));
        Check(even, "a triangle strip exports the same references as the list");
        auto killed = std::vector<std::uint32_t>(alternating.begin(), alternating.end());
        killed.insert(std::find(killed.begin(), killed.end(), 0xf800180fu), 0x7c000000u | (0xd2u << 17u) | (3u << 9u) | 0x80u);
        Draw(*device, killed, exported, replace());
        Check(even, "shader kill remains effective before reference filtering");
        auto userColor = std::vector<std::uint32_t>(alternating.begin(), alternating.end());
        *std::find(userColor.begin(), userColor.end(), 0x7e0e02f2u) = 0x7e0e0200u;
        const std::array<std::uint32_t, 1> colorData{0x3f800000u};
        Draw(*device, userColor, exported, replace(), 3u, 1u, false, {}, {}, Vertices, VertexCode, colorData);
        Check(all, "stencil export preserves the shader-data push payload");
        auto guarded = std::vector<std::uint32_t>(alternating.begin(), alternating.end());
        *std::find(guarded.begin(), guarded.end(), 0x7e0e02f2u) = 0x7e0e020cu;
        guarded.insert(guarded.begin(), {0xf4040280u, 0xfa000018u, 0xf4000305u, 0xfa000000u, 0xbf8cc07fu});
        alignas(256) std::array<std::uint32_t, 16> root{};
        const std::uint32_t payload = 0x3f800000u;
        const auto rootAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(root.data()));
        const std::array<std::uint32_t, 2> rootData{static_cast<std::uint32_t>(rootAddress), static_cast<std::uint32_t>(rootAddress >> 32u)};
        const auto payloadAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&payload));
        root[6] = static_cast<std::uint32_t>(payloadAddress);
        root[7] = static_cast<std::uint32_t>(payloadAddress >> 32u);
        const std::array<ShaderRecompiler::MemoryRegion, 2> guardedMemory{{{rootAddress, std::as_bytes(std::span(root))}, {payloadAddress, std::as_bytes(std::span(&payload, 1))}}};
        Draw(*device, guarded, exported, replace(), 6u, 2u, false, {}, {}, Vertices, VertexCode, rootData, guardedMemory);
        Check(all, "readonly guarded SRT survives repeated references, primitives and instances");
        std::puts("stencil replay tests passed (native [[stencil]] export)");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
