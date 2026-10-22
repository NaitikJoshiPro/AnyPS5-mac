// Metal twin of tests/execution/LayeredVolumeDraw.cpp (agc_driver_layered_volume_draw and its
// --vertex-only form), swapped in by twins/B6-stage-io.cmake. The upstream file is compiled unchanged
// (main renamed) so its draws, shaders and checks keep tracking upstream; only the mesh half differs:
// upstream inspects the SPIR-V mesh program for BuiltIn Layer and runs the SPIR-V validator without
// drawing, which has no MSL form. Here the MSL mesh function (NGG primitive shader, DECISIONS D23)
// must carry the layer in its primitive record ([[render_target_array_index]], each primitive taking
// its provoking vertex's layer) and must then draw the same bands into the same volume slices as the
// vertex draw.

#define main AgcDriverLayeredVolumeDrawVulkanMain
#include "../../../tests/execution/LayeredVolumeDraw.cpp"
#undef main

namespace {

void RunMesh(AgcDriver::VulkanDevice& device, const AgcDriver::Graphics::ColorTarget& color) {
    const auto target = device.Target();
    const auto vertexBuffer = BufferDescriptor(Vertices.data(), sizeof(Vertex), static_cast<std::uint32_t>(Vertices.size()));
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Vertices.size()), 0, 1, 0, false};
    ShaderRecompiler::ShaderVertexStageInfo vertexInfo{};
    vertexInfo.paClVsOutCntl = LayerExport;
    std::vector<std::uint32_t> userData(12u, 0u);
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), userData.begin() + 8);
    const std::span<const std::uint32_t> code(GeometryCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> geometryMemory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    ShaderRecompiler::RecompileRequest geometry{
        {ShaderStage::Mesh, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, 0, userData, std::nullopt, std::nullopt, vertexInfo, geometryMemory},
        target,
        {0, 0, 0, ShaderRecompiler::MeshDrawPushOffsetBytes}
    };
    geometry.graphics = ShaderRecompiler::GraphicsCompileContext{0, {}, Subgroup, std::nullopt, {draw.indexAddress, draw.indexCount, draw.indexSize, draw.instanceCount}};
    geometry.useCache = false;
    const auto geometryResult = ShaderRecompiler::Recompile(geometry);
    Require(!geometryResult.msl.empty() && geometryResult.msl.Reflection().stage == ShaderRecompiler::MslAbi::Stage::Mesh, "the geometry program is not an MSL mesh function");
    const std::string source(geometryResult.msl.Source());
    Require(source.find("struct Aps5MeshPrimitive {") != std::string::npos && source.find("uint aps5_layer [[render_target_array_index]];") != std::string::npos, "the mesh function's primitive record has no [[render_target_array_index]] layer");
    const auto geometryPush = static_cast<std::uint32_t>(geometryResult.pushConstants.size());

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const std::vector<std::uint32_t> pixelUserData(8, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
        {64, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(geometryPush, target)
    };
    fragment.useCache = false;
    const auto pixelResult = ShaderRecompiler::Recompile(fragment);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Mesh, &geometryResult, 0},
        {ShaderStage::Fragment, &pixelResult, PixelPushOffset(geometryPush, target)}
    }};

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Geometry, 0x20u, 64, 64, Subgroup, std::nullopt};
    state.color = color;
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.renderLayers = Layers;
    state.layerExports = LayerExport;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
    AgcDriver::Graphics::StorageTexture::FlushAllPending("layered volume mesh draw test");
    device.WaitIdle();
}

}

int main(int argc, char** argv) {
    try {
        const bool vertexOnly = argc == 2 && std::strcmp(argv[1], "--vertex-only") == 0;
        Require(argc == 1 || vertexOnly, "usage: layered volume draw [--vertex-only]");
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto target = device->Target();
        if (std::find(target.supportedCapabilities.begin(), target.supportedCapabilities.end(), ShaderViewportIndexLayerCapability) == target.supportedCapabilities.end()) {
            std::puts("skipped, the device has no VK_EXT_shader_viewport_index_layer");
            return VulkanTestSkipped;
        }
        BuildBands();
        const GuestVolume volume((VolumeTarget(nullptr).bytes + 65535u) & ~std::size_t{65535u});
        const auto color = VolumeTarget(volume.Data());
        Run(*device, color, false, PixelCode, false);
        Check(volume.Data(), color, "vertex draw", WhiteBands);
        // Upstream fdefd7660: a pixel shader reading the layer from its packed ancillary word.
        const GuestVolume fieldVolume((VolumeTarget(nullptr).bytes + 65535u) & ~std::size_t{65535u});
        const auto fieldColor = VolumeTarget(fieldVolume.Data());
        Run(*device, fieldColor, false, LayerFieldPixelCode, true);
        Check(fieldVolume.Data(), fieldColor, "a pixel shader reading the render target index of its packed ancillary word", LayerFieldBands);
        // Upstream b2d988070: a thick SW_4KB_S_X volume (thick 4 KiB standard XOR detiling and write-back).
        const GuestVolume thickVolume((VolumeTarget(nullptr, ColorTileMode::S4KBX).bytes + 65535u) & ~std::size_t{65535u});
        const auto thickColor = VolumeTarget(thickVolume.Data(), ColorTileMode::S4KBX);
        Run(*device, thickColor, false, PixelCode, false);
        Check(thickVolume.Data(), thickColor, "thick SW_4KB_S_X vertex draw and writeback", WhiteBands);
        if (vertexOnly) {
            std::puts("layered volume vertex draw tests passed");
        } else if (target.mesh.has_value()) {
            // A second volume, so the mesh draw is checked on its own bands.
            const GuestVolume meshVolume((VolumeTarget(nullptr).bytes + 65535u) & ~std::size_t{65535u});
            const auto meshColor = VolumeTarget(meshVolume.Data());
            RunMesh(*device, meshColor);
            Check(meshVolume.Data(), meshColor, "mesh draw", WhiteBands);
        } else {
            std::puts("mesh case skipped, the device has no VK_EXT_mesh_shader");
        }
        std::puts("layered volume draw tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
