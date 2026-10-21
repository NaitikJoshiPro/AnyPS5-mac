// Metal twin of tests/execution/RectListCorner.cpp (agc_driver_rect_list_corner), swapped in by
// twins/SKIPFIX5-tests.cmake. Upstream draws one rect list with hand-built SPIR-V vertex and fragment shaders
// through the rect-list tessellation stages (ShaderRecompiler::BuildRectListShaders), which the SPIR-V path adds.
// The Metal device has no such stages: it expands rect lists in the vertex function (MslAbi::RectListMacro,
// MetalDraw.mm, Execution Draw.cpp rectListStages), so the twin draws the same rect list with guest programs
// recompiled for the Metal target: a vertex program that fetches each corner's position from a vertex buffer and
// exports it as position and parameter 0, and a pixel program that writes the interpolated parameter. Upstream's
// corners, identities and checks (VertexOf, the four corners, the coverage bound) run unchanged: the parameter of
// corner v0 (-1, -1) clamps to identity 0, v1 (1, -1) to 1 (red), v2 (-1, 1) to 2 (green) and the fourth corner
// the expansion adds, (1, 1), to 3.

#define main AgcDriverRectListCornerVulkanMain
#include "../../../tests/execution/RectListCorner.cpp"
#undef main

namespace {

// buffer_load_format_xyzw v[0:3], v5, s[0:3] idxen; s_waitcnt; exp param0 v[0:3]; exp pos0 v[0:3] done; s_endpgm
// (the vertex program of tests/execution/PixelImplicitDerivatives.cpp).
alignas(256) constexpr std::array<std::uint32_t, 8> TwinVertexCode{
    0xe0382000u, 0x80000005u, 0xbf8c3f70u, 0xf800020fu, 0x03020100u, 0xf80008cfu, 0x03020100u, 0xbf810000u,
};

// s_wqm_b64 exec; v_interp_p1/p2_f32 v2 = attr0.x, v3 = attr0.y; v_mov_b32 v4, 0; v_mov_b32 v5, 1.0;
// exp mrt0 v2, v3, v4, v5 done vm; s_endpgm
alignas(256) constexpr std::array<std::uint32_t, 10> TwinPixelCode{
    0xbefe0a7eu, 0xc8080000u, 0xc8090001u, 0xc80c0100u, 0xc80d0101u, 0x7e080280u, 0x7e0a02f2u,
    0xf800180fu, 0x05040302u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::array<float, 4>, 3> Corners{{{-1.0f, -1.0f, 0.0f, 1.0f}, {1.0f, -1.0f, 0.0f, 1.0f}, {-1.0f, 1.0f, 0.0f, 1.0f}}};

int RunMetal(AgcDriver::VulkanDevice& device) {
    std::cout << "device: " << device.DeviceName() << "\n";
    const auto target = device.Target();
    Require(target.backend == ShaderRecompiler::ShaderBackend::Msl, "the rect-list corner twin runs on the Metal target");
    const auto address = reinterpret_cast<std::uintptr_t>(Corners.data());
    const std::vector<std::uint32_t> vertexUserData{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (16u << 16u), static_cast<std::uint32_t>(Corners.size()), 0x01016facu};
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(TwinVertexCode.data()), std::as_bytes(std::span(TwinVertexCode))}}};
    ShaderRecompiler::RecompileRequest vertexRequest{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(TwinVertexCode.data()), TwinVertexCode, 0, {}},
        {32u, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory}, target, {0, 0, 0, 64}
    };
    vertexRequest.useCache = false;
    const auto vertex = ShaderRecompiler::Recompile(vertexRequest);
    const auto vertexPush = static_cast<std::uint32_t>(vertex.pushConstants.size());

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.wave32 = true;
    pixel.interpolatorCount = 1;
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PerspectiveCenter);
    pixel.hasPerspectiveCenterVgpr = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const std::vector<std::uint32_t> pixelUserData(4, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(TwinPixelCode.data()), std::as_bytes(std::span(TwinPixelCode))}}};
    ShaderRecompiler::RecompileRequest fragmentRequest{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(TwinPixelCode.data()), TwinPixelCode, 0, {}},
        {32u, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory}, target, PixelPushLayout(vertexPush, target)
    };
    fragmentRequest.useCache = false;
    const auto fragment = ShaderRecompiler::Recompile(fragmentRequest);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertex, 0u},
        {ShaderStage::Fragment, &fragment, PixelPushOffset(vertexPush, target)},
    }};
    // Upstream's state, but for the rect-list stages the Metal device does not add.
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(pixels.data()), {Extent, Extent}, VK_FORMAT_R8G8B8A8_UNORM, pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.rectList = true;
    state.renderExtent = {Extent, Extent};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0.0f, static_cast<float>(Extent), static_cast<float>(Extent), -static_cast<float>(Extent), 0.0f, 1.0f};
    state.scissor = {{0, 0}, {Extent, Extent}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15u;
    state.blends = {state.blend};
    pixels.fill(std::byte{0});
    device.Draw(state, {0u, 3u, 0u, 1u, 0u, false}, shaders);
    device.WaitIdle();

    // Upstream's checks, unchanged.
    const std::array<std::array<std::uint32_t, 3>, 4> corners{{
        {0u, Extent - 1u, 0u},
        {Extent - 1u, Extent - 1u, 1u},
        {0u, 0u, 2u},
        {Extent - 1u, 0u, 3u},
    }};
    for (const auto& corner : corners) {
        const auto selected = VertexOf(corner[0], corner[1]);
        std::cout << "corner (" << corner[0] << ", " << corner[1] << ") vertex " << selected << "\n";
        Require(selected == corner[2], "rect-list corner selected vertex " + std::to_string(selected) + ", expected " + std::to_string(corner[2]));
    }
    std::uint32_t covered = 0;
    for (std::size_t offset = 3; offset < pixels.size(); offset += 4) {
        if (pixels[offset] != std::byte{0}) ++covered;
    }
    std::cout << "covered " << covered << " of " << Extent * Extent << "\n";
    Require(covered * 5u > Extent * Extent * 4u, "rect-list rectangle covered " + std::to_string(covered) + " pixels");
    return 0;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto result = RunMetal(*device);
        if (result != 0) return result;
        std::cout << "rect-list corner tests passed (Metal: the vertex function's rect-list expansion)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
