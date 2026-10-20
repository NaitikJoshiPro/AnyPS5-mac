// Metal twin of tests/execution/Tessellation.cpp (agc_driver_tessellation), swapped in by twins/B7-tessellation.cmake.
// The upstream file is compiled here with the functions that read result.spirv renamed, so its programs, draws,
// refusals and pixel expectations keep tracking upstream; main below repeats upstream's order over this file's
// versions of those functions. For an MSL target result.spirv is empty, so:
// - "was not compiled" checks (single-control-point hulls, compare branches) require the MSL source instead;
// - HasSpacing / HasLevelBound read OpExecutionMode Spacing* and the 63/64 OpConstant factor bound of the SPIR-V
//   hull and domain. On Metal neither is in a stage library: the partitioning is an argument of the linked mesh
//   function's software tessellator (LinkMslTessellation, aps5_tess_process(domain, partitioning, topology, ...),
//   DECISIONS D24) and the tessellator core bounds the factors by it (MslTessellatorCore.inc: 63 for fractional_odd,
//   64 otherwise, D26). The twin links the draw's stages for the patch draw's configuration and checks that call
//   and that bound instead. Every draw and pixel check after them is upstream's.

#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "MetalBackend/MslEmitter.hpp"
#include "SceShaders.hpp"
#include "../../../tests/execution/VulkanTestDevice.hpp"
#include <spirv/unified1/spirv.hpp>
#include <string>

#define RequireCompareBranches UpstreamRequireCompareBranches
#define RequireLevelClamps UpstreamRequireLevelClamps
#define RequireIntegerPartitioning UpstreamRequireIntegerPartitioning
#define RequireFractionalEvenPartitioning UpstreamRequireFractionalEvenPartitioning
#define RequireSingleControlPointHulls UpstreamRequireSingleControlPointHulls
#define main AgcDriverTessellationVulkanMain
#include "../../../tests/execution/Tessellation.cpp"
#undef main
#undef RequireSingleControlPointHulls
#undef RequireFractionalEvenPartitioning
#undef RequireIntegerPartitioning
#undef RequireLevelClamps
#undef RequireCompareBranches

namespace {

bool Compiled(const ShaderRecompiler::RecompileResult& result) {
    return !result.msl.empty() && !result.msl.Source().empty();
}

ShaderRecompiler::MslTessellationProgram TwinProgram(const ShaderRecompiler::RecompileResult& result) {
    Require(Compiled(result), "tessellation twin: a stage has no MSL source");
    ShaderRecompiler::MslTessellationProgram program;
    program.shader = &result.msl.Shader();
    for (const auto& constant : result.specialization) program.constants.emplace_back(constant.id, constant.value);
    return program;
}

// The partitioning the linked mesh function tessellates with (MslTessellation.cpp: aps5_tess_process(domain,
// partitioning, topology, factors)), and the factor bound the tessellator core gives it.
void RequireLinkedPartitioning(const Stages& stages, const PatchDraw& patches, std::uint32_t partitioning, float bound, const std::string& what) {
    const auto& config = patches.configuration;
    Require(config.partitioning == partitioning, what + ": the patch draw's configuration names partitioning " + std::to_string(config.partitioning));
    const auto linked = ShaderRecompiler::LinkMslTessellation(TwinProgram(stages.local), TwinProgram(stages.hull), TwinProgram(stages.domain), config);
    const auto call = "aps5_tess_process(" + std::to_string(config.domain) + ", " + std::to_string(partitioning) + ", " + std::to_string(config.outputTopology) + ", ";
    Require(linked.mesh.source.find(call) != std::string::npos, what + " did not link the software tessellator with that partitioning");
    const auto& source = linked.mesh.source;
    Require(source.find("if (partitioning == 2) upper = 63.0f;") != std::string::npos && source.find("float upper = 64.0f;") != std::string::npos, what + ": the tessellator core lacks the 63 (fractional_odd) / 64 factor bound");
    Require((partitioning == 2u ? 63.0f : 64.0f) == bound, what + ": unexpected factor bound");
}

void RequireCompareBranches(const ShaderRecompiler::SpirvTarget& target) {
    std::uint32_t cursor = 0;
    Require(Compiled(Compile(target, ShaderStage::TessellationEvaluation, DomainCompareBranch, cursor)), "a domain program branching on a compare copied to VCC was not compiled");
    Require(Compiled(Compile(target, ShaderStage::TessellationControl, HullCompareBranch, cursor)), "a hull program branching on a compare copied to VCC was not compiled");
}

void RequireSingleControlPointHulls(const ShaderRecompiler::SpirvTarget& target) {
    Require(Compiled(CompileHull(target, HullSingleInput, {1u, 3u, 1u, 2u, 2u})), "a hull that reads only the one input control point of its patch was not compiled");
    Require(Compiled(CompileHull(target, HullSingleOutput, {1u, 1u, 1u, 2u, 2u})), "a hull with one output control point and unguarded patch outputs was not compiled");
}

// Upstream's RequireLevelClamps, RequireIntegerPartitioning and RequireFractionalEvenPartitioning with their
// HasSpacing/HasLevelBound checks replaced by RequireLinkedPartitioning.
void RequireLevelClamps(AgcDriver::VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target) {
    const auto center = Row(-0.5f + 1.0f / 3.0f);
    const auto stages = CompileTriangles(target, Triangles);
    RequireLinkedPartitioning(stages, Triangles, 2u, 63.0f, "fractional-odd partitioning");
    const PatchDraw flattened{Patches, Indices, 1.0f, 1.0f};
    Draw(device, stages, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, flattened);
    Require(CoveredPixels(2, false) > 100 && CoveredPixels(2, true) == 0, "VGT_HOS_MAX_TESS_LEVEL 1 did not flatten the level-5 patch, centre " + PixelText(PixelAt(Column(2, 0.25f), center)));
    Require(CoveredPixels(1, false) == 0, "VGT_HOS_MAX_TESS_LEVEL 1 kept the patch with level 0");
    Draw(device, stages, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, Triangles);
    Require(CoveredPixels(2, false) > 100 && PixelAt(Column(2, 0.25f), center)[1] > 32, "the same hull kept the previous draw's VGT_HOS_MAX_TESS_LEVEL 1 for a draw with level 64, centre " + PixelText(PixelAt(Column(2, 0.25f), center)));
    const PatchDraw raised{Patches, Indices, 5.0f, 64.0f};
    Draw(device, stages, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, raised);
    Require(PixelAt(Column(0, 0.25f), center)[1] > 32, "VGT_HOS_MIN_TESS_LEVEL 5 did not tessellate the level-1 patch, centre " + PixelText(PixelAt(Column(0, 0.25f), center)));
    Require(CoveredPixels(1, false) == 0, "VGT_HOS_MIN_TESS_LEVEL 5 raised the patch with level 0 instead of culling it");
}

void RequireIntegerPartitioning(AgcDriver::VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target) {
    const auto center = Row(-0.5f + 1.0f / 3.0f);
    const PatchDraw patches{{3u, 3u, 1u, 0u, 2u}, Indices};
    const auto stages = CompileTriangles(target, patches);
    RequireLinkedPartitioning(stages, patches, 0u, 64.0f, "integer partitioning");
    Draw(device, stages, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, patches);
    Require(CoveredPixels(0, false) > 100 && CoveredPixels(0, true) == 0, "integer partitioning: the level-1 patch was not drawn as one flat triangle, centre " + PixelText(PixelAt(Column(0, 0.25f), center)));
    Require(CoveredPixels(1, false) == 0, "integer partitioning: the patch with level 0 was not culled");
    Require(CoveredPixels(2, false) > 100 && PixelAt(Column(2, 0.25f), center)[1] > 32, "integer partitioning: the level-5 patch has no interior domain points at its centre, " + PixelText(PixelAt(Column(2, 0.25f), center)));
}

void RequireFractionalEvenPartitioning(AgcDriver::VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target) {
    const auto below = Row(-0.75f);
    const PatchDraw integer{{3u, 3u, 1u, 0u, 2u}, Indices};
    Draw(device, CompileLoweredInterior(target, integer), VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, integer);
    Require(CoveredPixels(0, false) > 100 && PixelAt(Column(0, 0.25f), below) == Background, "integer partitioning: the level-1 patch has a domain point inside it, " + PixelText(PixelAt(Column(0, 0.25f), below)));
    Require(PixelAt(Column(2, 0.25f), below) != Background, "integer partitioning: the level-5 patch has no domain point inside it");
    const PatchDraw patches{{3u, 3u, 1u, 3u, 2u}, Indices};
    const auto stages = CompileLoweredInterior(target, patches);
    RequireLinkedPartitioning(stages, patches, 3u, 64.0f, "fractional-even partitioning");
    Draw(device, stages, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, patches);
    Require(CoveredPixels(0, false) > 100 && PixelAt(Column(0, 0.25f), below) != Background, "fractional-even partitioning: the level-1 patch was not raised to level 2 with a domain point inside it");
    Require(CoveredPixels(1, false) == 0 && PixelAt(Column(1, 0.25f), below) == Background, "fractional-even partitioning: the patch with level 0 was not culled");
    Require(CoveredPixels(2, false) > 100 && PixelAt(Column(2, 0.25f), below) != Background, "fractional-even partitioning: the level-5 patch has no domain point inside it");
}

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto target = device->Target();
        if (!target.tessellation.has_value()) {
            std::puts("tessellation: skipped, the device has no tessellation shaders");
            return VulkanTestSkipped;
        }
        RequireSingleControlPointHulls(target);
        ExpectRefusal(target, ShaderStage::TessellationControl, HullUnguardedFactors, "not written by control point 0 alone");
        ExpectRefusal(target, ShaderStage::TessellationControl, HullSquaredAddress, "not a whole number of control points");
        ExpectRefusal(target, ShaderStage::TessellationControl, HullHalfSlotIndex, "not a whole number of control points");
        ExpectRefusal(target, ShaderStage::TessellationControl, HullLaneCount, "lane mask");
        ExpectRefusal(target, ShaderStage::TessellationEvaluation, DomainNeighbourRecord, "more than one patch-constant record", RecordPatches);
        ExpectRefusal(target, ShaderStage::TessellationEvaluation, DomainFixedPatchRead, "not indexed by its relative patch", RecordPatches);
        ExpectRefusal(target, ShaderStage::TessellationEvaluation, DomainNeighbourPoint, "neither its patch's control points nor a patch-constant record", RecordPatches);
        ExpectRefusal(target, ShaderStage::TessellationEvaluation, DomainCulledPrimitive, "passthrough primitive");
        ExpectRefusal(target, ShaderStage::Local, LocalIndexAsData, "outside a tessellation ring or LDS address");
        ExpectRefusal(target, ShaderStage::TessellationControl, HullRelativeData, "GE_CNTL.PRIM_GRP_SIZE is 0", RecordPatches);
        ExpectRefusal(target, ShaderStage::TessellationEvaluation, DomainRelativeData, "GE_CNTL.PRIM_GRP_SIZE is 0", RecordPatches);
        ExpectRefusal(target, ShaderStage::TessellationControl, HullCode, "outside its patch's quad factors", {{3u, 3u, 2u, 2u, 2u}, Indices});
        ExpectRefusal(target, ShaderStage::TessellationControl, HullDescendingRecords, "not affine");
        ExpectRefusal(target, ShaderStage::TessellationControl, HullDescendingRecords, "not affine", {{3u, 3u, 1u, 2u, 2u, 158u}, Indices});
        ExpectRefusal(target, ShaderStage::TessellationControl, HullDescendingRecords, "outside its control points and patch constants", {{3u, 3u, 1u, 2u, 2u, 129u}, Indices});

        RequirePatches(*device, target, HullCode, "hull program");
        RequirePatches(*device, target, HullRestoredExecSaved, "hull program saving its restored exec");
        RequirePatches(*device, target, HullNextInput, "hull program reading the next of 4 input control points of 24 bytes", NextInputTriangles, LocalCodeStride24);
        RequirePatches(*device, target, HullDynamicIndex, "hull program reading a control point at a computed index");
        RequirePatches(*device, target, HullAndSaveexec, "hull program narrowing exec with s_and_saveexec_b64");
        RequirePatches(*device, target, HullSaveexecLoop, "hull program restoring exec through loop phis");
        RequirePatches(*device, target, HullDescendingRecords, "hull program with descending patch records in 128-patch groups", {{3u, 3u, 1u, 2u, 2u, 128u}, Indices});
        RequireCounterClockwise(*device, target);
        RequireLevelClamps(*device, target);
        RequireIntegerPartitioning(*device, target);
        RequireFractionalEvenPartitioning(*device, target);
        RequireQuads(*device, target);
        RequirePatchConstants(*device, target);
        RequireRelativePatches(*device, target);
        RequireInstancedRelativePatches(*device, target);
        RequireHullBdaReads(*device, target);
        RequireCompareBranches(target);
        device.reset();
        RequireRegisteredStages();
        std::puts("Tessellation tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
