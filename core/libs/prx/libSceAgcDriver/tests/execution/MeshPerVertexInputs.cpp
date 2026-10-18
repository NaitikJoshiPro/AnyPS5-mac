// agc_driver_mesh_per_vertex_inputs: per-vertex fragment inputs and emulated barycentrics of mesh
// (NGG) draws. The Metal device draws them in the mesh function's triangle-vertices mode (MslAbi
// Version 13; DECISIONS D46): every output triangle has three
// unshared vertices with one-hot corner weights and its corners' values in the primitive record, the
// subgroup split over the grid's z when 3 x its primitives exceed Metal's 256 mesh vertices. The
// same triangles are drawn through a vertex program in the vertex function's triangle-vertices mode
// (DECISIONS D20) and through tests/Mesh.cpp's NGG geometry program, first with a host-interpolated
// pixel program (same coverage on both paths), then with three pixel programs:
// v_interp_mov P10/P20/P0 plus the raw PERSP_CENTER I (SuperTuxKart's radeonsi shape: per-vertex
// location 0 and emulated perspective barycentrics), raw PERSP_CENTER I/J and raw LINEAR_CENTER I/J.
// The two paths must produce bit-identical R32G32B32A32 images (same positions, same corner order,
// same one-hot weights through the same rasterizer), for triangle lists (wave64 one- and two-wave
// subgroups, wave32, indexed, 64-primitive subgroups split over two threadgroups), the last-vertex
// provoking convention and strips; the per-vertex values of the lists are also checked against the
// corners. tests/Mesh.cpp's program outputs three vertices per primitive (maxVertices 3 x the
// primitives), so a split comes from a mesh declared for more primitives than a subgroup allocates.
// Strips run with the last vertex provoking: the translator hands an odd strip triangle's NGG
// program the vertices (i + 1, i, i + 2) (InstructionTranslator.cpp, the mesh entry setup), which
// with the last-vertex rotation is the vertex path's (i + 2, i + 1, i); under the first-vertex
// convention the vertex path uses Vulkan's (i, i + 2, i + 1) and the mesh path the guest's export
// order (i + 1, i, i + 2) (TechnicalDebt metal-mesh-per-vertex).
// Line and point lists (DECISIONS D75, MslAbi Version 16): the mesh function outputs lines and points
// from the primitive export's first two or one indices, and must match the vertex path's lines and
// points (D69) bit for bit.
// Metal only: the Vulkan device's mesh path has its own barycentric emulation (upstream).

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::PixelInput;
using ShaderRecompiler::PixelInputBit;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 160;
constexpr std::uint32_t Height = 96;
constexpr float Unwritten = -7.0f;
alignas(256) std::array<float, Width * Height * 4> Pixels{};

// tests/Mesh.cpp's NGG geometry program: index fetch, LDS vertex exchange, GS_ALLOC_REQ, primitive,
// position and parameter exports (param0 = the vertex buffer's dwords 4-7).
alignas(256) constexpr std::array<std::uint32_t, 104> GeometryCode{
    0x8f6a9003, 0x94fe6ac1, 0xbf88000b, 0xd7650006, 0x000100c1, 0xd7660006, 0x00020cc1, 0x93ebff03,
    0x00040018, 0xd7460006, 0x04190c6b, 0x340c0c82, 0xd8340000, 0x00000506, 0xbf8cc07f, 0xbefe04c1,
    0xbf8a0000, 0x938dff02, 0x00090016, 0x9382ff03, 0x00040018, 0x938cff03, 0x00080008, 0xd7650009,
    0x000100c1, 0xd7660009, 0x000212c1, 0xd746000a, 0x04250c02, 0x7da8120c, 0xbf88002e, 0x361600ff,
    0x0000ffff, 0x2c180090, 0x361a02ff, 0x0000ffff, 0xd8d80000, 0x0b00000b, 0xd8d80000, 0x0c00000c,
    0xd8d80000, 0x0d00000d, 0xbf8cc07f, 0xe0382000, 0x8002100b, 0xe0382010, 0x8002140b, 0xe0382000,
    0x8002180c, 0xe0382010, 0x80021c0c, 0xe0382000, 0x8002200d, 0xe0382010, 0x8002240d, 0x161c1483,
    0x161e14ff, 0x00000060, 0xbf8c3f70, 0xdb7c0400, 0x0000100f, 0xdb7c0410, 0x0000140f, 0xdb7c0420,
    0x0000180f, 0xdb7c0430, 0x00001c0f, 0xdb7c0440, 0x0000200f, 0xdb7c0450, 0x0000240f, 0x4a501c81,
    0x4a521c82, 0x3450508a, 0x34525294, 0xd772002a, 0x04a6510e, 0xbf8cc07f, 0xbefe04c1, 0xbf8a0000,
    0x930e830d, 0xbf078002, 0xbf850003, 0x8f0f8c0d, 0x887c0f0e, 0xbf900009, 0x7da8140d, 0xbf880002,
    0xf8000941, 0x0000002a, 0xbefe04c1, 0x7da8140e, 0xbf88000a, 0x34561485, 0xdbfc0400, 0x2c00002b,
    0xdbfc0410, 0x3000002b, 0xbf8cc07f, 0xf80008cf, 0x2f2e2d2c, 0xf800020f, 0x33323130, 0xbf810000,
};

alignas(256) constexpr auto Wave32GeometryCode = [] {
    auto code = GeometryCode;
    code[10] = 0x04190a6bu;
    code[28] = 0x04250a02u;
    return code;
}();

// The vertex program with the same interface: buffer_load_format_xyzw v[0:3] (offset 0) and v[4:7]
// (offset 16) by the vertex index, param0 = v[4:7], pos0 = v[0:3].
alignas(256) constexpr std::array<std::uint32_t, 10> VertexCode{
    0xe0382000, 0x80000005, 0xe0382010, 0x80000405, 0xbf8c3f70, 0xf800020f, 0x07060504, 0xf80008cf, 0x03020100, 0xbf810000,
};

// v_interp_mov P10, P20, P0 of attr0.x into v2-v4 and the raw I (v0), exported as (P10, P20, P0, I).
alignas(256) constexpr std::array<std::uint32_t, 6> PerVertexPixelCode{
    0xc80a0000, 0xc80e0001, 0xc8120002, 0xf800180f, 0x00040302, 0xbf810000,
};

// The raw I/J exported as (I, J, I, J).
alignas(256) constexpr std::array<std::uint32_t, 3> BarycentricPixelCode{
    0xf800180f, 0x01000100, 0xbf810000,
};

// v_interp_p1/p2 of attr0.x with the PERSP_CENTER I/J, exported to all four channels: host
// interpolation (no emulation), which checks that both paths cover the same pixels.
alignas(256) constexpr std::array<std::uint32_t, 5> InterpolatedPixelCode{
    0xc8080000, 0xc8090001, 0xf800180f, 0x02020202, 0xbf810000,
};

struct Vertex {
    std::array<float, 4> position;
    std::array<float, 4> color;
};

constexpr std::uint32_t Columns = 8;
constexpr std::uint32_t Rows = 8;
constexpr std::uint32_t Triangles = Columns * Rows;
constexpr std::uint32_t StripPrimitives = 30;

alignas(256) std::array<Vertex, 3 * Triangles> List{};
alignas(256) std::array<Vertex, 3 * Triangles> Scrambled{};
alignas(256) std::array<std::uint16_t, 3 * Triangles> Indices{};
alignas(256) std::array<Vertex, StripPrimitives + 2> Strip{};

float WeightOf(std::uint32_t vertex) {
    constexpr std::array<float, 3> weights{1.0f, 1.75f, 0.625f};
    return weights[vertex % 3u];
}

Vertex Make(float x, float y, std::uint32_t id) {
    const float w = WeightOf(id);
    return {{x * w, y * w, 0.5f * w, w}, {0.03125f * static_cast<float>(id + 1u), 0.25f, 0.5f, 1.0f}};
}

void Build() {
    for (std::uint32_t triangle = 0; triangle < Triangles; ++triangle) {
        const float cellWidth = 2.0f / Columns;
        const float cellHeight = 2.0f / Rows;
        const float x = -1.0f + cellWidth * static_cast<float>(triangle % Columns);
        const float y = -1.0f + cellHeight * static_cast<float>(triangle / Columns);
        // A skewed triangle per cell, alternating its winding.
        const std::array<std::array<float, 2>, 3> corners{{{0.08f, 0.1f}, {0.93f, 0.3f}, {0.35f, 0.92f}}};
        for (std::uint32_t k = 0; k < 3; ++k) {
            const auto corner = corners[triangle % 2u == 0u ? k : 2u - k];
            List[3 * triangle + k] = Make(x + corner[0] * cellWidth, y + corner[1] * cellHeight, 3 * triangle + k);
        }
    }
    for (std::uint32_t triangle = 0; triangle < Triangles; ++triangle) {
        for (std::uint32_t k = 0; k < 3; ++k) {
            const auto slot = 3 * (Triangles - 1 - triangle) + (k + 1) % 3;
            Scrambled[slot] = List[3 * triangle + k];
            Indices[3 * triangle + k] = static_cast<std::uint16_t>(slot);
        }
    }
    for (std::uint32_t i = 0; i < Strip.size(); ++i) {
        const float x = -0.95f + 1.9f * static_cast<float>(i / 2u) / static_cast<float>(Strip.size() / 2u - 1u);
        const float y = i % 2u == 0u ? -0.7f + 0.05f * static_cast<float>(i % 7u) : 0.75f - 0.04f * static_cast<float>(i % 5u);
        Strip[i] = Make(x, y, i);
    }
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (32u << 16u), count, 0x01016facu};
}

struct PixelProgram {
    const char* name;
    std::span<const std::uint32_t> code;
    std::uint32_t inputs;
};

struct Case {
    std::string name;
    ShaderRecompiler::MeshConfiguration mesh;
    AgcDriver::Pm4::DrawParameters draw;
    std::array<std::uint32_t, 4> vertexBuffer;
    std::uint32_t waveSize;
    bool last;
};

ShaderRecompiler::RecompileResult Fragment(AgcDriver::VulkanDevice& device, const PixelProgram& program, std::uint32_t previousPush) {
    const auto target = device.Target();
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = 1;
    pixel.inputAddr = program.inputs;
    pixel.hasPerspectiveCenterVgpr = (program.inputs & PixelInputBit(PixelInput::PerspectiveCenter)) != 0u;
    pixel.noPerspective = (program.inputs & PixelInputBit(PixelInput::LinearCenter)) != 0u;
    pixel.executeOnNoop = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(program.code.data()), std::as_bytes(program.code)}}};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(program.code.data()), program.code, 0, {}},
        {64u, 0, {}, std::nullopt, pixel, std::nullopt, memory},
        target,
        PixelPushLayout(previousPush, target),
        std::nullopt
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

AgcDriver::Graphics::State BaseState(VkPrimitiveTopology topology, bool last) {
    AgcDriver::Graphics::State state{};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(Pixels), 0xe4u, AgcDriver::Graphics::ColorTileMode::Linear, 16u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = topology;
    state.provokingVertexMode = last ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    return state;
}

VkPrimitiveTopology TopologyOf(const Case& test) {
    if (test.mesh.inputPrimitive == 1u) return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    if (test.mesh.inputPrimitive == 2u) return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    return test.mesh.inputPrimitive == 6u ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

ShaderRecompiler::RecompileResult DrawVertex(AgcDriver::VulkanDevice& device, const Case& test, const PixelProgram& program) {
    Pixels.fill(Unwritten);
    const auto target = device.Target();
    std::vector<std::uint32_t> userData(test.vertexBuffer.begin(), test.vertexBuffer.end());
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {64u, 0, userData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, memory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());
    auto pixelResult = Fragment(device, program, vertexPush);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, &pixelResult, PixelPushOffset(vertexPush, target)},
    }};
    auto state = BaseState(TopologyOf(test), test.last);
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    device.Draw(state, test.draw, shaders);
    device.WaitIdle();
    return pixelResult;
}

ShaderRecompiler::RecompileResult DrawMesh(AgcDriver::VulkanDevice& device, const Case& test, const PixelProgram& program) {
    Pixels.fill(Unwritten);
    const auto target = device.Target();
    std::vector<std::uint32_t> userData(12, 0u);
    const auto index = AgcDriver::Graphics::MeshIndexBufferDescriptor(test.draw);
    std::copy(index.begin(), index.end(), userData.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    std::copy(test.vertexBuffer.begin(), test.vertexBuffer.end(), userData.begin() + 8);
    const auto& code = test.waveSize == 32u ? Wave32GeometryCode : GeometryCode;
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(std::span(code))}}};
    ShaderRecompiler::RecompileRequest geometry{
        {ShaderStage::Mesh, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {test.waveSize, 0, userData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, memory},
        target,
        {0, 0, 0, ShaderRecompiler::MeshDrawPushOffsetBytes},
        ShaderRecompiler::GraphicsCompileContext{0, {}, test.mesh, std::nullopt, {test.draw.indexAddress, test.draw.indexCount, test.draw.indexSize, test.draw.instanceCount}}
    };
    geometry.useCache = false;
    const auto meshResult = ShaderRecompiler::Recompile(geometry);
    const auto meshPush = static_cast<std::uint32_t>(meshResult.pushConstants.size());
    auto pixelResult = Fragment(device, program, meshPush);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Mesh, &meshResult, 0},
        {ShaderStage::Fragment, &pixelResult, PixelPushOffset(meshPush, target)},
    }};
    auto state = BaseState(TopologyOf(test), test.last);
    state.stages = {AgcDriver::Graphics::ShaderPath::Geometry, 0x20u, test.waveSize, 64, test.mesh, std::nullopt};
    device.Draw(state, test.draw, shaders);
    device.WaitIdle();
    return pixelResult;
}

std::string At(std::size_t pixel) {
    return "(" + std::to_string(pixel % Width) + ", " + std::to_string(pixel / Width) + ")";
}

// The corner values a triangle-list pixel's (P10, P20, P0) must be: corner 0 is the provoking vertex
// (the first, or the third under the last-vertex convention), then the triangle's order.
void CheckListValues(const std::string& what, bool last) {
    std::uint32_t checked = 0;
    for (std::size_t pixel = 0; pixel < Width * Height; ++pixel) {
        const float* value = &Pixels[pixel * 4u];
        if (value[0] == Unwritten) continue;
        bool found = false;
        for (std::uint32_t triangle = 0; triangle < Triangles && !found; ++triangle) {
            std::array<float, 3> c{};
            for (std::uint32_t k = 0; k < 3; ++k) c[k] = List[3 * triangle + (last ? (k + 2u) % 3u : k)].color[0];
            found = value[2] == c[0] && value[0] == c[1] - c[0] && value[1] == c[2] - c[0];
        }
        Require(found, what + ": pixel " + At(pixel) + " reads P10/P20/P0 (" + std::to_string(value[0]) + ", " + std::to_string(value[1]) + ", " + std::to_string(value[2]) + "), which are no triangle's corner values");
        Require(value[3] >= -1e-5f && value[3] <= 1.0f + 1e-5f, what + ": pixel " + At(pixel) + " reads I = " + std::to_string(value[3]));
        ++checked;
    }
    Require(checked > 400u, what + ": only " + std::to_string(checked) + " pixels were drawn");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (device->Target().backend != ShaderRecompiler::ShaderBackend::Msl || !device->Target().mesh) {
            std::puts("skipped, the mesh triangle-vertices mode is the Metal device's (MslAbi Version 13)");
            return VulkanTestSkipped;
        }
        Build();
        const auto perspective = PixelInputBit(PixelInput::PerspectiveCenter);
        const auto linear = PixelInputBit(PixelInput::LinearCenter);
        const std::array<PixelProgram, 4> programs{{
            {"host interpolation", InterpolatedPixelCode, perspective},
            {"P10/P20/P0 and PERSP_CENTER I", PerVertexPixelCode, perspective},
            {"PERSP_CENTER I/J", BarycentricPixelCode, perspective},
            {"LINEAR_CENTER I/J", BarycentricPixelCode, linear},
        }};
        const auto listDraw = AgcDriver::Pm4::DrawParameters{0, static_cast<std::uint32_t>(List.size()), 0, 1, 0, false};
        const auto indexedDraw = AgcDriver::Pm4::DrawParameters{reinterpret_cast<std::uintptr_t>(Indices.data()), static_cast<std::uint32_t>(Indices.size()), 2, 1, 0, true};
        const auto stripDraw = AgcDriver::Pm4::DrawParameters{0, static_cast<std::uint32_t>(Strip.size()), 0, 1, 0, false};
        // {input primitive, primitives and vertices per group, max vertices, max primitives,
        // threads, LDS dwords, provoking vertex, ESGS item size}, as tests/Mesh.cpp.
        const std::array<Case, 11> cases{{
            {"triangle list, one-wave subgroups, wave64", {4u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u}, listDraw, BufferDescriptor(List.data(), static_cast<std::uint32_t>(List.size())), 64u, false},
            {"triangle list, two-wave subgroups, wave64", {4u, 32u, 96u, 96u, 32u, 128u, 2048u, 0u, 4u}, listDraw, BufferDescriptor(List.data(), static_cast<std::uint32_t>(List.size())), 64u, false},
            {"triangle list, wave32", {4u, 4u, 12u, 12u, 4u, 32u, 1024u, 0u, 4u}, listDraw, BufferDescriptor(List.data(), static_cast<std::uint32_t>(List.size())), 32u, false},
            {"indexed triangle list", {4u, 32u, 96u, 96u, 32u, 128u, 2048u, 0u, 4u}, indexedDraw, BufferDescriptor(Scrambled.data(), static_cast<std::uint32_t>(Scrambled.size())), 64u, false},
            {"triangle list, last vertex provoking", {4u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u}, listDraw, BufferDescriptor(List.data(), static_cast<std::uint32_t>(List.size())), 64u, true},
            // 64-primitive subgroups in a mesh declared for 96 (192 output vertices, three waves): the
            // triangle-vertices mode splits it into two threadgroups of 48 (0-47, then 48-63).
            {"triangle list, 64-primitive subgroups of a 96-primitive mesh (two splits)", {4u, 64u, 192u, 192u, 96u, 192u, 4096u, 0u, 4u}, listDraw, BufferDescriptor(List.data(), static_cast<std::uint32_t>(List.size())), 64u, false},
            {"triangle list, two splits, last vertex provoking", {4u, 64u, 192u, 192u, 96u, 192u, 4096u, 0u, 4u}, listDraw, BufferDescriptor(List.data(), static_cast<std::uint32_t>(List.size())), 64u, true},
            // Strips with the last vertex provoking (OpenGL's default, radeonsi's): see the header.
            {"triangle strip, last vertex provoking", {6u, 3u, 5u, 9u, 3u, 64u, 1024u, 0u, 4u}, stripDraw, BufferDescriptor(Strip.data(), static_cast<std::uint32_t>(Strip.size())), 64u, true},
            // Points and lines (MslAbi Version 16, DECISIONS D75): the program still exports three
            // vertices per primitive; a line takes the export's first two indices, a point its first.
            {"line list", {2u, 4u, 8u, 12u, 4u, 64u, 1024u, 0u, 4u}, listDraw, BufferDescriptor(List.data(), static_cast<std::uint32_t>(List.size())), 64u, false},
            {"line list, last vertex provoking", {2u, 4u, 8u, 12u, 4u, 64u, 1024u, 0u, 4u}, listDraw, BufferDescriptor(List.data(), static_cast<std::uint32_t>(List.size())), 64u, true},
            {"point list", {1u, 4u, 4u, 12u, 4u, 64u, 1024u, 0u, 4u}, listDraw, BufferDescriptor(List.data(), static_cast<std::uint32_t>(List.size())), 64u, false},
        }};
        std::vector<float> reference(Pixels.size());
        std::uint32_t compared = 0;
        for (const auto& test : cases) {
            for (const auto& program : programs) {
                const auto what = test.name + ", " + program.name;
                const auto vertexPixel = DrawVertex(*device, test, program);
                const bool emulated = program.code.data() != InterpolatedPixelCode.data();
                Require(vertexPixel.barycentricEmulation.active == emulated, what + ": the pixel program " + (emulated ? "does not run" : "runs") + " the barycentric emulation");
                std::copy(Pixels.begin(), Pixels.end(), reference.begin());
                const auto meshPixel = DrawMesh(*device, test, program);
                Require(meshPixel.barycentricEmulation.active == emulated, what + ": the mesh draw's pixel program " + (emulated ? "does not run" : "runs") + " the barycentric emulation");
                std::uint32_t drawn = 0;
                std::uint32_t differing = 0;
                std::size_t first = 0;
                std::uint32_t low = Width, high = 0;
                for (std::size_t index = 0; index < Pixels.size(); ++index) {
                    if (std::memcmp(&Pixels[index], &reference[index], sizeof(float)) != 0) {
                        if (differing++ == 0u) first = index;
                        low = std::min<std::uint32_t>(low, static_cast<std::uint32_t>(index / 4u % Width));
                        high = std::max<std::uint32_t>(high, static_cast<std::uint32_t>(index / 4u % Width));
                    }
                    drawn += index % 4u == 0u && Pixels[index] != Unwritten ? 1u : 0u;
                }
                Require(differing == 0u, what + ": " + std::to_string(differing) + " channels differ (x " + std::to_string(low) + " to " + std::to_string(high) + "), first pixel " + At(first / 4u) + " channel " + std::to_string(first % 4u) + " is " + std::to_string(Pixels[first]) + " on the mesh path, " + std::to_string(reference[first]) + " on the vertex path");
                const std::uint32_t least = test.mesh.inputPrimitive == 1u ? 150u : 400u;
                Require(drawn > least, what + ": only " + std::to_string(drawn) + " pixels were drawn");
                if (program.code.data() == PerVertexPixelCode.data() && test.mesh.inputPrimitive == 4u) CheckListValues(what, test.last);
                std::printf("%s: %u pixels identical on the mesh and vertex paths\n", what.c_str(), drawn);
                ++compared;
            }
        }
        std::printf("mesh per-vertex input tests passed: %u draws compared\n", compared);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
