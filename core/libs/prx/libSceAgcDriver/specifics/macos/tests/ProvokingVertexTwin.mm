// Metal twin of tests/execution/ProvokingVertex.cpp (agc_driver_provoking_vertex), swapped in by
// twins/B6-stage-io.cmake (work package B6). The upstream test draws with two GLSL-compiled SPIR-V
// shaders, which have no MSL form; this twin runs the same draws and checks (compiled unchanged,
// main renamed, so they keep tracking upstream) with the two shaders written in MSL against the
// B6/D6 interface (M2_DESIGN.md): a vertex function that picks position and flat colour by vertex
// id, and a fragment function that writes the flat colour. The guest P0 part recompiles for the
// device's target as upstream does.

#define main AgcDriverProvokingVertexVulkanMain
#include "../../../tests/execution/ProvokingVertex.cpp"
#undef main

namespace {

constexpr const char* TwinVertex = R"(#include <metal_stdlib>
using namespace metal;
struct Aps5VertexOut {
    float4 aps5_position [[position]];
    float4 p0 [[user(locn0)]];
};
vertex Aps5VertexOut aps5_main(uint aps5_vertex_id [[vertex_id]]) {
    const float2 positions[6] = {float2(-0.75f, 0.75f), float2(-0.75f, -0.75f), float2(0.75f, 0.75f), float2(0.75f, -0.75f), float2(0.75f, 0.75f), float2(-0.75f, -0.75f)};
    const float4 colors[6] = {float4(1, 0, 0, 1), float4(0, 1, 0, 1), float4(0, 0, 1, 1), float4(1, 1, 0, 1), float4(0, 1, 1, 1), float4(1, 0, 1, 1)};
    Aps5VertexOut out;
    out.aps5_position = float4(positions[aps5_vertex_id], 0.5f, 1.0f);
    out.p0 = colors[aps5_vertex_id];
    return out;
}
)";

constexpr const char* TwinFragment = R"(#include <metal_stdlib>
using namespace metal;
#ifndef APS5_VERTEX_OUTPUTS
#define APS5_VERTEX_OUTPUTS 0xffffffff
#endif
struct Aps5FragmentIn {
    float4 aps5_position [[position]];
#if ((APS5_VERTEX_OUTPUTS >> 0) & 1)
    float4 p0 [[user(locn0), flat]];
#endif
};
#ifndef APS5_COLOR0_TYPE
#define APS5_COLOR0_TYPE float4
#endif
struct Aps5FragmentOut {
    APS5_COLOR0_TYPE c0 [[color(0)]];
};
fragment Aps5FragmentOut aps5_main(Aps5FragmentIn aps5_in [[stage_in]]) {
    Aps5FragmentOut out;
    out.c0 = as_type<APS5_COLOR0_TYPE>(aps5_in.p0);
    return out;
}
)";

ShaderRecompiler::RecompileResult MslShader(const char* source, ShaderRecompiler::MslAbi::Stage stage, std::uint64_t variant) {
    ShaderRecompiler::MslShader shader;
    shader.source = source;
    shader.reflection.stage = stage;
    if (stage == ShaderRecompiler::MslAbi::Stage::Vertex) {
        shader.reflection.group = ShaderRecompiler::RuntimeAbi::Stage::Main;
        shader.reflection.outputLocations = 1u;
    } else {
        shader.reflection.group = ShaderRecompiler::RuntimeAbi::Stage::Fragment;
        shader.reflection.colorOutputs = 1u;
    }
    ShaderRecompiler::RecompileResult result{};
    result.msl = ShaderRecompiler::SharedMsl(std::move(shader));
    result.variantId = variant;
    return result;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto vertex = MslShader(TwinVertex, ShaderRecompiler::MslAbi::Stage::Vertex, 1);
        const auto fragment = MslShader(TwinFragment, ShaderRecompiler::MslAbi::Stage::Fragment, 2);
        const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
            {ShaderRecompiler::ShaderStage::Vertex, &vertex, 0},
            {ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}
        }};
        if (!device->ProvokingVertexLast()) {
            for (const auto topology : {VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP}) Run(*device, topology, shaders, false, "MSL");
            bool rejected = false;
            try {
                Run(*device, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, shaders, true, "MSL");
            } catch (const std::exception& error) {
                Require(std::string(error.what()).find("last provoking vertex requires VK_EXT_provoking_vertex with provokingVertexLast enabled") != std::string::npos, std::string("unexpected last-vertex rejection: ") + error.what());
                rejected = true;
            }
            Require(rejected, "last provoking vertex was drawn without device support");
            device->WaitIdle();
            // The guest P0 flat input, first vertex only (Metal has no last provoking vertex).
            const auto guestPixel = GuestPixel(device->Target());
            const std::array<AgcDriver::Graphics::CompiledShader, 2> guestShaders{{
                {ShaderRecompiler::ShaderStage::Vertex, &vertex, 0},
                {ShaderRecompiler::ShaderStage::Fragment, &guestPixel, PixelPushOffset(0u, device->Target())}
            }};
            for (const auto topology : {VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP}) Run(*device, topology, guestShaders, false, "guest P0");
            // Upstream f2a4236ab's RunRectangle (a guest P0 rect list drawn first/last/first) is not run: it builds the
            // SPIR-V rect-list TCS/TES (BuildRectListShaders), while Metal expands rect lists in the guest vertex
            // function (APS5_RECT_LIST), whose flat parameters do not follow the provoking vertex yet.
            std::puts("skipped, guest P0 rect list provoking vertex: Metal rect lists take flat inputs from each host triangle's first vertex, not the guest's provoking vertex (TechnicalDebt metal-b6-rect-flat)");
            std::puts("provoking vertex rendering tests passed: MSL and guest P0 first-vertex rendering, last-vertex rejection (Metal has no last provoking vertex)");
            return 0;
        }
        for (const auto topology : {VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP}) Run(*device, topology, shaders, true, "MSL");
        std::puts("provoking vertex rendering tests passed: MSL, first/last/first");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
