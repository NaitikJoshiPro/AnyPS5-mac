// agc_metal_msl_stage_io (work package B6): vertex and fragment functions of MetalBackend checked on
// the CPU, without a Metal device. Recompile requests of vertex and pixel programs (seeded from the
// GPU suite's dumped corpus into tests/msl-stage-io) are recompiled for the MSL backend, and the
// emitted source is checked against its reflection and the B6/D6 interface (M2_DESIGN.md):
// - vertex: `vertex Aps5VertexOut aps5_main`, the [[vertex_id]] and DriverData parameters, the
//   APS5_RECT_LIST expansion, Reflection::outputLocations equal to the user(locnN) outputs,
//   Reflection::vertexFetchLocations equal to the VertexFetch prologue loads;
// - fragment: a [[stage_in]] struct whose user(locnN) inputs are each guarded by APS5_VERTEX_OUTPUTS,
//   Reflection::colorOutputs equal to the [[color(n)]] outputs, exportsDepth/usesDiscard/
//   earlyFragmentTests matching the source.
// MSL compilation of the same shapes is the corpus check's job (agc_metal_msl_corpus).

#include "ControlFlow/RequestSerializer.hpp"
#include "MslAbi.hpp"
#include "MetalBackend/MslConservative.hpp"
#include "Recompiler.hpp"
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool Has(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

std::uint64_t Mask(const std::string& text, const std::string& prefix, const std::string& suffix, std::uint32_t limit) {
    std::uint64_t mask = 0;
    for (std::uint32_t index = 0; index < limit; ++index) {
        if (Has(text, prefix + std::to_string(index) + suffix)) mask |= std::uint64_t {1u} << index;
    }
    return mask;
}

void Check(const fs::path& file, std::uint32_t& vertex, std::uint32_t& fragment) {
    std::ifstream stream(file, std::ios::binary);
    std::stringstream text;
    text << stream.rdbuf();
    auto replay = ShaderRecompiler::RequestSerializer{}.Deserialize(text.str());
    auto& request = replay.request;
    request.useCache = false;
    request.target.backend = ShaderRecompiler::ShaderBackend::Msl;
    const auto result = ShaderRecompiler::Recompile(request);
    const auto name = file.filename().string();
    Require(!result.msl.empty(), name + ": no MSL");
    const auto& reflection = result.msl.Reflection();
    std::string source(result.msl.Source());
    // The conservative-rasterization discard (APS5_CONSERVATIVE, MslAbi Version 5) is the driver's,
    // not the guest program's: Reflection::usesDiscard does not count it.
    const auto conservative = ShaderRecompiler::MslConservative::FragmentText();
    if (const auto at = source.find(conservative); at != std::string::npos) source.erase(at, conservative.size());
    // So is the uniform sample mask's discard (APS5_SAMPLE_MASK, MslAbi Version 15).
    const std::string hostMask(ShaderRecompiler::MslAbi::SampleMaskDiscardText);
    if (const auto at = source.find(hostMask); at != std::string::npos) source.erase(at, hostMask.size());
    Require(reflection.abiVersion == ShaderRecompiler::MslAbi::Version, name + ": unexpected MslAbi version");
    if (reflection.stage == ShaderRecompiler::MslAbi::Stage::Vertex) {
        ++vertex;
        Require(Has(source, "vertex Aps5VertexOut aps5_main("), name + ": no vertex function returning Aps5VertexOut");
        Require(Has(source, "uint aps5_vertex_id [[vertex_id]]") && Has(source, "constant Aps5DriverData& aps5_driver [[buffer(3)]]"), name + ": the vertex function lacks the vertex id or DriverData the rect-list expansion needs");
        Require(Has(source, "#if defined(APS5_RECT_LIST) && APS5_RECT_LIST") && Has(source, "aps5_driver.workgroupBase[0] + aps5_rect * 3u"), name + ": no APS5_RECT_LIST expansion");
        Require(Has(source, "float4 aps5_position [[position]];") && Has(source, "[[clip_distance]]"), name + ": no position or clip distances");
        Require(reflection.outputLocations == Mask(source, "[[user(locn", ")]]", 64u), name + ": Reflection::outputLocations differs from the user(locnN) outputs");
        Require(reflection.vertexFetchLocations == static_cast<std::uint32_t>(Mask(source, "const uint4 aps5_va", " = aps5_vertex_fetch(", 32u)), name + ": Reflection::vertexFetchLocations differs from the fetches");
        Require(reflection.vertexFetchLocations == 0u || Has(source, "device const Aps5VertexFetch* aps5_fetch [[buffer(4)]]"), name + ": fetches without the VertexFetch buffer");
        Require(!Has(source, "aps5_out.aps5_position.y = -aps5_out.aps5_position.y;\n#endif") || Has(source, "#if defined(APS5_FLIP_Y)"), name + ": an unconditional y flip (the driver flips the viewport)");
    } else if (reflection.stage == ShaderRecompiler::MslAbi::Stage::Fragment) {
        ++fragment;
        Require(Has(source, "fragment ") && Has(source, "Aps5FragmentIn aps5_in [[stage_in]]"), name + ": no fragment function with a stage_in struct");
        for (std::uint32_t location = 0; location < 64u; ++location) {
            const auto field = "float4 p" + std::to_string(location) + " [[user(locn" + std::to_string(location) + ")";
            const auto at = source.find(field);
            if (at == std::string::npos) continue;
            const auto guard = source.rfind("#if ((APS5_VERTEX_OUTPUTS", at);
            Require(guard != std::string::npos && source.find("#endif", guard) > at, name + ": fragment input " + std::to_string(location) + " is not guarded by APS5_VERTEX_OUTPUTS");
        }
        Require(reflection.colorOutputs == static_cast<std::uint32_t>(Mask(source, "[[color(", ")]]", 8u)), name + ": Reflection::colorOutputs differs from the color(n) outputs");
        Require(reflection.exportsDepth == Has(source, "[[depth("), name + ": Reflection::exportsDepth differs from the source");
        Require(reflection.usesDiscard == Has(source, "discard_fragment()"), name + ": Reflection::usesDiscard differs from the source");
        Require(reflection.earlyFragmentTests == Has(source, "[[early_fragment_tests]]"), name + ": Reflection::earlyFragmentTests differs from the source");
    } else {
        throw std::runtime_error(name + ": not a vertex or fragment request");
    }
}

}

int main(int argc, char** argv) {
    try {
        const fs::path directory = argc > 1 ? fs::path(argv[1]) : fs::path("core/libs/prx/libSceAgcDriver/specifics/macos/tests/msl-stage-io");
        std::uint32_t vertex = 0, fragment = 0;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.path().extension() != ".req") continue;
            Check(entry.path(), vertex, fragment);
        }
        Require(vertex != 0u && fragment != 0u, "the sample set needs vertex and fragment requests");
        std::printf("msl stage io check passed: %u vertex, %u fragment\n", vertex, fragment);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
