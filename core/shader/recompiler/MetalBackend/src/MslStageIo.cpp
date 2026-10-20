#include "MetalBackend/MslEmitterState.hpp"
#include "MetalBackend/MslConservative.hpp"
#include "PipelineSpecialization.hpp"
#include "IntermediateRepresentation/IrMetadata/BufferFormat.hpp"
#include <algorithm>
#include <bit>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

// Work package B6: stage interfaces. Compute builtins and user data for every stage; for vertex and
// pixel programs the vertex/fragment function interface of M2_DESIGN "B6/D6 interface":
// - vertex attributes are pulled from the VertexFetch records (BufferIndex::VertexFetch) and
//   converted by the BufferFormat helpers (B5);
// - exports (SetAttribute) write the stage function's output struct `aps5_out`: position, parameters
//   as float4 user(locnN), point size, clip distances (plus the all-zero position cull distance),
//   layer and viewport for vertex programs; MRT colours (typed by APS5_COLOR<n>_TYPE, default from
//   pixelInfo), depth, stencil and sample mask for pixel programs;
// - pixel inputs read the fragment stage_in struct `aps5_in`, each parameter guarded by the
//   APS5_VERTEX_OUTPUTS macros, with the SPIR-V path's interpolation choice.
// The guest body runs in a helper function (MslEntryPoint.cpp), so its `return` statements keep
// working; the stage function fills `aps5_out` through it and applies the epilogue.

std::string TriangleVertexMacroDefaults() {
    return "#ifndef APS5_PER_VERTEX_LOCATIONS\n#define APS5_PER_VERTEX_LOCATIONS 0\n#endif\n"
           "#ifndef APS5_PER_VERTEX_LOCATIONS_HI\n#define APS5_PER_VERTEX_LOCATIONS_HI 0\n#endif\n"
           "#ifndef APS5_BARYCENTRICS\n#define APS5_BARYCENTRICS 0\n#endif\n";
}

std::string TriangleVerticesCondition() {
    return "defined(APS5_TRIANGLE_VERTICES) && APS5_TRIANGLE_VERTICES";
}

std::string WindowSpaceEpilogue(const std::string& position, const std::string& indent) {
    // PA_CL_VTE_CNTL 0x300: x and y are pixels, z is depth, nothing divides them by w, and w is only
    // the perspective term of attribute interpolation. Multiplying the NDC of a (0, 0, width, height,
    // 0, 1) viewport by w lands on the same pixel and depth after the host's divide and keeps w.
    const std::string condition = std::string("defined(") + MslAbi::WindowSpaceMacro + ") && " + MslAbi::WindowSpaceMacro;
    std::string text = "#if " + condition + "\n";
    text += indent + "{\n";
    text += indent + "    const float4 aps5_ws = " + position + ";\n";
    text += indent + "    " + position + " = float4((2.0f * aps5_ws.x / float(" + MslAbi::WindowSpaceWidthMacro + ") - 1.0f) * aps5_ws.w, (2.0f * aps5_ws.y / float(" + MslAbi::WindowSpaceHeightMacro + ") - 1.0f) * aps5_ws.w, aps5_ws.z * aps5_ws.w, aps5_ws.w);\n";
    text += indent + "}\n";
    text += "#endif\n";
    return text;
}

std::string PerVertexLocationCondition(std::uint32_t location) {
    return std::string("((") + (location < 32u ? "APS5_PER_VERTEX_LOCATIONS" : "APS5_PER_VERTEX_LOCATIONS_HI") + " >> " + std::to_string(location % 32u) + ") & 1)";
}

std::string PerVertexField(std::uint32_t location, std::uint32_t vertex) {
    return "pv" + std::to_string(location) + "_" + std::to_string(vertex);
}

std::uint32_t MslCullClipSlot(const MslEmitterState& state) {
    std::uint32_t count = 0u;
    for (const auto& output : state.program.Info().outputs) {
        if (output.kind == StageOutputKind::ClipDistance) count = std::max(count, output.index + 1u);
    }
    return count + 1u;
}

namespace {

std::string QuadPixel() {
    return "((uint(aps5_in.aps5_position.x) & 1u) | ((uint(aps5_in.aps5_position.y) & 1u) << 1u))";
}

}

std::string MslQuadCoverage(const MslEmitterState& state) {
    const auto& pixel = *state.inputInfo.pixel;
    if (!pixel.QuadSampleMaskPartial()) return "0xffffffffu";
    // 16 guest samples per quad pixel, the top row's two pixels in one word, the bottom row's in
    // the other (x selects the half word).
    const auto& masks = pixel.quadSampleMask;
    const auto top = static_cast<std::uint32_t>(masks[0]) | (static_cast<std::uint32_t>(masks[1]) << 16u);
    const auto bottom = static_cast<std::uint32_t>(masks[2]) | (static_cast<std::uint32_t>(masks[3]) << 16u);
    return "((((" + QuadPixel() + " & 2u) != 0u ? " + std::to_string(bottom) + "u : " + std::to_string(top) + "u) >> ((" + QuadPixel() + " & 1u) << 4u)) & 0xffffu)";
}

std::string MslQuadUncovered(const MslEmitterState& state) {
    const auto& pixel = *state.inputInfo.pixel;
    // The host mask's bits are guest samples APS5_SAMPLE_BASE and up (DECISIONS D28).
    // aps5_sample_coverage: the stage function's coverage, a per-sample invocation's own sample (MslAbi::SampleMaskDiscardText).
    if (pixel.QuadSampleMaskPartial()) return "(((aps5_sample_coverage << APS5_SAMPLE_BASE) & " + MslQuadCoverage(state) + ") == 0u)";
    return "(((" + std::to_string(static_cast<std::uint32_t>(pixel.quadPixelMask)) + "u >> " + QuadPixel() + ") & 1u) == 0u)";
}

std::uint32_t MslCullDistanceCount(const MslEmitterState& state) {
    std::uint32_t count = 0u;
    for (const auto& output : state.program.Info().outputs) {
        if (output.kind == StageOutputKind::CullDistance) count = std::max(count, output.index + 1u);
    }
    return count;
}

namespace {

constexpr std::uint32_t PsInputFlatShade = 0x00000400u;
constexpr std::uint32_t PixelParameterLimit = 32u;
constexpr std::uint32_t MaxClipDistances = 8u;

std::string U32(std::uint32_t value) {
    return std::to_string(value) + "u";
}

IrShaderStage StageOf(const MslEmitterState& state) {
    return state.program.Resources().stage;
}

const ShaderPixelInputInfo& PixelInfo(MslEmitterState& state) {
    if (state.inputInfo.pixel == nullptr) state.Fail("pixel input info is missing (work package B6)");
    return *state.inputInfo.pixel;
}

const ShaderVertexInputInfo& VertexInfo(MslEmitterState& state) {
    if (state.inputInfo.vertex == nullptr) state.Fail("vertex input info is missing (work package B6)");
    return *state.inputInfo.vertex;
}

// Invocation ids come from the core (MslEmitterState), which rebuilds them per guest lane half for
// wave64 programs on 32-wide SIMD groups.
std::string ComputeBuiltin(MslEmitterState& state, StageInputKind kind, std::uint32_t component) {
    switch (kind) {
    case StageInputKind::LocalInvocationId:
        return state.GuestLocalInvocationId(component);
    case StageInputKind::LocalInvocationIndex:
        return state.GuestInvocationIndex();
    case StageInputKind::WorkgroupId:
        return state.WorkgroupId(component);
    case StageInputKind::GlobalInvocationId:
        return state.GuestGlobalInvocationId(component);
    case StageInputKind::DispatchThreadLimit:
        return state.ShaderDataDword(state.program.Metadata().bindings.DispatchThreadLimitDword() + component);
    default:
        state.Fail("compute builtin " + std::to_string(static_cast<unsigned>(kind)) + " is not implemented in the MSL backend yet");
    }
}

std::string VertexId(MslEmitterState& state) {
    state.AddParameter("uint aps5_vertex_id [[vertex_id]]");
    return "aps5_vertex_id";
}

std::string InstanceId(MslEmitterState& state) {
    state.AddParameter("uint aps5_instance_id [[instance_id]]");
    return "aps5_instance_id";
}

const char* LocationName(InterpolationLocation location) {
    switch (location) {
    case InterpolationLocation::Centroid: return "centroid";
    case InterpolationLocation::Sample: return "sample";
    default: return "center";
    }
}

// A read at the sample's location runs the fragment function per sample (MSL specification 5.5:
// sample_* inputs and [[sample_id]]); interpolate_at_sample takes the sample's (host) index.
void ReadAtSample(MslEmitterState& state) {
    state.AddParameter("uint aps5_sample_id [[sample_id]]");
    state.reflection.perSample = true;
}

// The interpolation location of one read (DECISIONS D38): the IrValue flags the front end set
// (InterpolationLocationFlags), else the center for barycentrics, and for attributes the only
// location the program's metadata names for the input.
InterpolationLocation ReadLocation(MslEmitterState& state, const IrValue& instruction, std::optional<std::uint32_t> attribute) {
    const auto flags = instruction.Flags<std::uint32_t>();
    if ((flags & InterpolationLocationFlag) != 0u) {
        const auto location = flags & ~InterpolationLocationFlag;
        if (location > static_cast<std::uint32_t>(InterpolationLocation::Sample)) state.Fail(instruction, "invalid interpolation location " + std::to_string(location));
        return static_cast<InterpolationLocation>(location);
    }
    if (!attribute.has_value()) return InterpolationLocation::Center;
    const auto& metadata = state.program.Metadata();
    const auto bit = 1u << *attribute;
    const bool atSample = (metadata.pixelSampleInputs & bit) != 0u;
    const bool atCentroid = (metadata.pixelCentroidInputs & bit) != 0u;
    if (static_cast<unsigned>(atSample) + static_cast<unsigned>(atCentroid) + static_cast<unsigned>((metadata.pixelCenterInputs & bit) != 0u) > 1u) state.Fail(instruction, "reads pixel input " + std::to_string(*attribute) + ", interpolated at more than one of center, centroid and sample, without naming the location of this read");
    return atSample ? InterpolationLocation::Sample : atCentroid ? InterpolationLocation::Centroid : InterpolationLocation::Center;
}

// Emulated per-vertex values (MslAbi Version 5, the triangle-vertices expansion): the corner weights
// interpolated perspective-correct (bit 0) or linearly (bit 1), as gl_BaryCoordKHR / NoPerspKHR, at
// the read's location (DECISIONS D38: aps5_bw_<location>, defined by MslGraphicsInterfaceOf).
std::string EmulatedBarycentric(MslEmitterState& state, bool linear, InterpolationLocation location) {
    state.reflection.emulatedBarycentrics |= linear ? 2u : 1u;
    state.barycentricLocations[linear ? 1u : 0u] |= 1u << static_cast<std::uint32_t>(location);
    if (location == InterpolationLocation::Sample) ReadAtSample(state);
    return std::string(linear ? "aps5_bwl_" : "aps5_bw_") + LocationName(location);
}

// One channel of corner `vertex`'s value of a per-vertex pixel parameter at Vulkan location
// `location`, as a float (flat input user(aps5_pv<L>_<vertex>), zero when the vertex function does
// not write the location).
std::string PerVertexValue(MslEmitterState& state, std::uint32_t location, std::uint32_t vertex, std::uint32_t channel) {
    state.reflection.perVertexLocations |= std::uint64_t {1u} << location;
    return "aps5_pv_" + std::to_string(location) + "(" + std::to_string(vertex) + ", " + std::to_string(channel & 3u) + ")";
}

bool InputIsPerVertex(MslEmitterState& state, std::uint32_t attribute) {
    return std::ranges::any_of(state.program.Info().inputs, [&](const StageInput& input) { return input.kind == StageInputKind::Parameter && input.location == attribute && input.perVertex; });
}

std::uint32_t PixelLocation(MslEmitterState& state, const IrValue& instruction, std::uint32_t attribute) {
    const auto& pixel = PixelInfo(state);
    if (attribute >= PixelParameterLimit || attribute >= pixel.inputNum) state.Fail(instruction, "pixel interpolator index is out of range");
    const auto location = pixel.InputSlot(attribute);
    if (location >= 64u) state.Fail(instruction, "pixel input location " + std::to_string(location) + " is beyond 63");
    return location;
}

// Alpha to coverage (DECISIONS D28): DB_ALPHA_TO_MASK 0xaa01, the undithered setting titles were
// seen writing (offsets 2, no OFFSET_ROUND), as upstream measured it on RDNA2 hardware
// (boykopovar/AnyPS5 PR 3228: 1 sample covers iff alpha >= 0.5, NaN and negatives cover nothing;
// PR 4451: 8 samples cover floor(alpha * 8 + 0.5) samples in a fixed mask per 1/16 of alpha, not
// the lowest bits). Other counts take floor(alpha * n + 0.5) of their lowest samples (not measured).
// The guest sample count is APS5_SAMPLE_COUNT (the pass's own count unless it is emulated).
constexpr std::string_view AlphaToMaskText = R"(#ifndef APS5_SAMPLE_COUNT
#define APS5_SAMPLE_COUNT get_num_samples()
#endif
static uint aps5_alpha_to_mask(float alpha, uint samples) {
    if (!(alpha > 0.0f)) return 0u;
    if (samples <= 1u) return alpha >= 0.5f ? 0xffffffffu : 0u;
    if (alpha >= 1.0f) return 0xffffffffu;
    if (samples == 8u) {
        const uint masks[16] = {0x00u, 0x10u, 0x04u, 0x14u, 0x12u, 0x32u, 0x16u, 0x36u, 0x35u, 0x75u, 0x3du, 0x7du, 0x77u, 0xf7u, 0x7fu, 0xffu};
        return masks[min(uint(alpha * 16.0f), 15u)];
    }
    const uint count = min(uint(alpha * float(samples) + 0.5f), samples);
    return count >= 32u ? 0xffffffffu : (1u << count) - 1u;
}
)";

// Dithered alpha to coverage (upstream c8b3a3753, StoreDitheredAlphaCoverage): DB_ALPHA_TO_MASK
// 0x18701 at eight samples (the driver accepts it only there), 64 alpha bins whose packed word
// holds one 8-sample mask per pixel of the 2x2 quad, selected by ((x & 1) << 1) | (y & 1) of the
// window position; NaN and negative alphas take bin 0, alphas above 63/64 bin 63.
constexpr std::string_view DitheredAlphaToMaskText = R"(static uint aps5_dithered_alpha_to_mask(float alpha, float4 position) {
    const uint masks[64] = {
        0x00000000u, 0x00000010u, 0x00000004u, 0x10000004u, 0x10000002u, 0x10100002u, 0x04100002u, 0x04101002u,
        0x04101001u, 0x04101011u, 0x04041011u, 0x14041011u, 0x12041011u, 0x12141011u, 0x12140411u, 0x12141411u,
        0x12141411u, 0x12141431u, 0x12141415u, 0x32141415u, 0x32121415u, 0x32321415u, 0x16321415u, 0x16323415u,
        0x15323415u, 0x15323435u, 0x15163435u, 0x35163435u, 0x35163235u, 0x35363235u, 0x35361635u, 0x35363635u,
        0x35363635u, 0x35363675u, 0x3536363du, 0x7536363du, 0x75363637u, 0x75763637u, 0x3d763637u, 0x3d767637u,
        0x3d757637u, 0x3d757677u, 0x3d3d7677u, 0x7d3d7677u, 0x773d7677u, 0x777d7677u, 0x777d3e77u, 0x777d7e77u,
        0x777d7e77u, 0x777d7ef7u, 0x777d7e7fu, 0xf77d7e7fu, 0xf7777e7fu, 0xf7f77e7fu, 0x7ff77e7fu, 0x7ff7fe7fu,
        0x7ff7fd7fu, 0x7ff7fdffu, 0x7f7ffdffu, 0xff7ffdffu, 0xff7ff7ffu, 0xfffff7ffu, 0xffff7fffu, 0xffffffffu,
    };
    const float positive = alpha > 0.0f ? alpha : 0.0f;
    const float bounded = positive > 0.984375f ? 0.984375f : positive;
    const uint pixel = ((uint(position.x) & 1u) << 1u) | (uint(position.y) & 1u);
    return (masks[uint(bounded * 64.0f)] >> (pixel << 3u)) & 0xffu;
}
)";

// Reversed-order blending (upstream 16ea32c97 / c434467f6, FragmentOutputSpecialization.hpp): each
// component of a ReversedBlendCode route field (RouteBits per component) picks the export's own
// component, component 0, 0 or 1.
constexpr std::string_view ReversedBlendRouteText = R"(static float4 aps5_reversed_route(float4 color, uint routes) {
    float4 result;
    for (uint c = 0u; c < 4u; ++c) {
        const uint route = (routes >> (2u * c)) & 3u;
        result[c] = route == 0u ? color[c] : route == 1u ? color[0] : route == 2u ? 0.0f : 1.0f;
    }
    return result;
}
)";

// EmitBuiltinU32 for vertex and pixel programs: the value's bits as a uint.
std::string GraphicsBuiltin(MslEmitterState& state, const IrValue& instruction, StageInputKind kind, std::uint32_t component) {
    const auto stage = StageOf(state);
    if (MslTessellationStage(stage)) {
        if (auto value = MslTessellationBuiltin(state, instruction, kind, component); !value.empty()) return value;
    }
    if (stage == IrShaderStage::Vertex || stage == IrShaderStage::Local) {
        switch (kind) {
        case StageInputKind::VertexIndex: return VertexId(state);
        case StageInputKind::InstanceIndex: return InstanceId(state);
        default: break;
        }
    }
    if (stage == IrShaderStage::Pixel) {
        switch (kind) {
        case StageInputKind::FragCoord:
            if (component > 3u) state.Fail(instruction, "FragCoord component is out of range");
            return "as_type<uint>(aps5_in.aps5_position[" + std::to_string(component) + "])";
        case StageInputKind::FrontFacing:
            state.AddParameter("bool aps5_front_facing [[front_facing]]");
            return "(aps5_front_facing ? 0x3f800000u : 0xbf800000u)";
        case StageInputKind::SampleId:
            state.AddParameter("uint aps5_sample_id [[sample_id]]");
            state.reflection.perSample = true;
            // MslAbi Version 9: APS5_SAMPLE_BASE is the guest index of host sample 0 (4 in the second
            // half of an 8x target emulated as two 4x halves, DECISIONS D28).
            return "(aps5_sample_id + APS5_SAMPLE_BASE)";
        case StageInputKind::PrimitiveId:
            state.AddParameter("uint aps5_primitive_id [[primitive_id]]");
            return "aps5_primitive_id";
        case StageInputKind::Layer:
            state.AddParameter("uint aps5_layer_in [[render_target_array_index]]");
            return "aps5_layer_in";
        case StageInputKind::HelperInvocation:
            // Upstream 90b20afac: a pixel the quad mask demoted reads as a helper. Metal's discard_fragment
            // demotes, but simd_is_helper_thread() stays false after it, so the stage function's demotions
            // (quad and sample masks, conservative coverage) reach the body as aps5_demoted.
            return "((simd_is_helper_thread() || aps5_demoted) ? 1u : 0u)";
        case StageInputKind::LayerExported:
            // Upstream #5097 (EmitBuiltinU32): 1 when the draw's last geometry stage exports a render
            // target index (PipelineSpecialization::LayerExported, default 0); the ancillary word's
            // bits 16-26 read Layer times this (ConstantFolder::ComposePackedAncillary).
            return state.FunctionConstant(PipelineSpecialization::LayerExported, 0u);
        case StageInputKind::RasterPrimitive:
            // Upstream #4031 (EmitBuiltinU32): the rasterized primitive class of the pipeline
            // (Recompiler.hpp RasterPrimitive), PipelineSpecialization::RasterPrimitive, default
            // Triangle; the ancillary word's bits 0-1 (ConstantFolder::ComposePackedAncillary).
            return state.FunctionConstant(PipelineSpecialization::RasterPrimitive, static_cast<std::uint32_t>(RasterPrimitive::Triangle));
        case StageInputKind::BaryCoordSmooth:
        case StageInputKind::BaryCoordNoPerspective:
            // Exact barycentrics only exist in programs the recompiler translated for the emulated
            // per-vertex values (ProgramMetadata::barycentricEmulation): the triangle-vertices vertex
            // function emits one-hot corner weights, interpolated as the hardware interpolates the
            // guest's I/J (EmulatedBarycentric).
            if (component > 1u) state.Fail(instruction, "barycentric component is out of range");
            return "as_type<uint>(" + EmulatedBarycentric(state, kind == StageInputKind::BaryCoordNoPerspective, ReadLocation(state, instruction, std::nullopt)) + "[" + std::to_string(component + 1u) + "])";
        default:
            break;
        }
    }
    state.Fail(instruction, "stage input " + std::to_string(static_cast<unsigned>(kind)) + " of stage " + std::to_string(static_cast<unsigned>(stage)) + " is not implemented in the MSL backend yet (work package B6)");
}

// One vertex attribute location, fetched once per invocation (prologue of the guest body).
std::string VertexAttribute(MslEmitterState& state, const IrValue& instruction, std::uint32_t attribute, std::uint32_t channel) {
    if (attribute >= 32u) state.Fail(instruction, "vertex attribute " + std::to_string(attribute) + " is beyond the 32 VertexFetch records");
    state.RequirePrelude(MslPreludeChunk::StageIo);
    state.AddParameter("device const Aps5VertexFetch* aps5_fetch [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::VertexFetch)) + ")]]");
    const auto name = "aps5_va" + std::to_string(attribute);
    if ((state.reflection.vertexFetchLocations & (1u << attribute)) == 0u) {
        state.reflection.vertexFetchLocations |= 1u << attribute;
        state.AddPrologue("const uint4 " + name + " = aps5_vertex_fetch(aps5_fetch, " + U32(attribute) + ", " + VertexId(state) + ", " + InstanceId(state) + ");");
    }
    return name + "[" + std::to_string(channel & 3u) + "]";
}

// A pixel input component (EmitAttributeValue): the interpolator's default, or the parameter at its
// Vulkan location through the APS5_VERTEX_OUTPUTS-guarded accessor.
std::string PixelAttribute(MslEmitterState& state, const IrValue& instruction, std::uint32_t attribute, std::uint32_t channel) {
    const auto& pixel = PixelInfo(state);
    if (attribute >= PixelParameterLimit || attribute >= pixel.inputNum) state.Fail(instruction, "pixel interpolator index is out of range");
    if (pixel.InputIsDefault(attribute)) return MslHex(pixel.InputDefaultBits(attribute, channel & 3u)) + "u";
    const auto location = PixelLocation(state, instruction, attribute);
    const bool flat = (pixel.interpolatorSettings[attribute] & PsInputFlatShade) != 0u && !pixel.InputIsCustom(attribute);
    const auto& metadata = state.program.Metadata();
    const bool linear = pixel.InputIsLinear(attribute, metadata.pixelLinearInputs, metadata.pixelPerspectiveInputs);
    if (InputIsPerVertex(state, attribute)) {
        // A slot read per vertex is interpolated from its corner values, as the SPIR-V path's
        // EmitAttributeValue: corner 0 when flat, else the barycentric-weighted sum (weights at the
        // read's location).
        if (flat) return "as_type<uint>(" + PerVertexValue(state, location, 0u, channel) + ")";
        const auto weights = EmulatedBarycentric(state, linear, ReadLocation(state, instruction, attribute));
        std::string sum = PerVertexValue(state, location, 0u, channel) + " * " + weights + "[0]";
        for (std::uint32_t vertex = 1u; vertex < 3u; ++vertex) sum = "(" + sum + " + " + PerVertexValue(state, location, vertex, channel) + " * " + weights + "[" + std::to_string(vertex) + "])";
        return "as_type<uint>(" + sum + ")";
    }
    // The interpolation location is the I/J pair this read's v_interp_p2 reads (SPI_PS_INPUT_ENA
    // PERSP/LINEAR_CENTER, _CENTROID, _SAMPLE; DECISIONS D28, D38): an input read at one location
    // takes that location's qualifier (sample_* inputs make Metal run the fragment function per
    // sample, at that sample's position), an input read at several is an interpolant<> read with
    // interpolate_at_<location> (MslGraphicsInterfaceOf defines aps5_input_<L>_<location>).
    using Mode = MslEmitterState::FragmentInput::Mode;
    const auto mode = flat ? Mode::Flat : linear ? Mode::Linear : Mode::Perspective;
    const auto input = state.fragmentInputs.try_emplace(location, MslEmitterState::FragmentInput {mode, 0u}).first;
    if (input->second.mode != mode) state.Fail(instruction, "pixel input location " + std::to_string(location) + " is read both flat and interpolated, or both perspective-correct and linearly");
    if (flat) return "aps5_input_" + std::to_string(location) + "(" + std::to_string(channel & 3u) + ")";
    const auto at = ReadLocation(state, instruction, attribute);
    input->second.locations |= 1u << static_cast<std::uint32_t>(at);
    if (at == InterpolationLocation::Sample) ReadAtSample(state);
    return "aps5_input_" + std::to_string(location) + "_" + LocationName(at) + "(" + std::to_string(channel & 3u) + ")";
}

void GetAttribute(MslEmitterState& state, const IrValue& instruction) {
    const auto attribute = instruction.Argument(0)->Resolve()->ImmediateU32();
    const auto channel = instruction.Argument(1)->Resolve()->ImmediateU32();
    const auto stage = StageOf(state);
    if (stage == IrShaderStage::Pixel) {
        state.Define(instruction, PixelAttribute(state, instruction, attribute, channel));
        return;
    }
    if (stage != IrShaderStage::Vertex && stage != IrShaderStage::Local) state.Fail(instruction, "attributes of this stage are not implemented in the MSL backend yet (work package B6)");
    if (instruction.Flags<std::uint32_t>() != 1u) {
        state.Define(instruction, VertexAttribute(state, instruction, attribute, channel));
        return;
    }
    // A prepared embedded-fetch attribute: the channel selector (0 zero, 1 one, 4-7 x-w) and the
    // "one" bits are function constants the pipeline specialises (as the SPIR-V path does without
    // registration modules).
    if (!VertexInfo(state).fetchEmbedded || channel >= 4u) state.Fail(instruction, "invalid prepared vertex attribute");
    const auto first = PipelineSpecialization::VertexBase + attribute * PipelineSpecialization::VertexWords;
    const auto selector = state.FunctionConstant(first + channel, channel + 4u);
    const auto one = state.FunctionConstant(first + 4u, 0x3f800000u);
    std::string value = "(" + selector + " == 1u ? " + one + " : 0u)";
    for (std::uint32_t component = 0; component < 4u; ++component) {
        value = "(" + selector + " == " + U32(component + 4u) + " ? " + VertexAttribute(state, instruction, attribute, component) + " : " + value + ")";
    }
    state.Define(instruction, value);
}

// v_interp_mov / barycentric-mode v_interp_p1/p2 (GetInterpolationParameter: mode 0 P10, 1 P20,
// 2 P0) and their 16-bit forms (GetInterpolationParameterF16, the low or high half), as the SPIR-V
// path's EmitInterpolationParameterValue / EmitInterpolationParameterF16Value: corner (mode + 1) % 3
// minus corner 0 for the deltas (not for pass-through inputs; the 16-bit delta rounds to half).
void GetInterpolationParameter(MslEmitterState& state, const IrValue& instruction, bool half) {
    if (StageOf(state) != IrShaderStage::Pixel) state.Fail(instruction, "interpolation parameters outside pixel programs");
    const auto attribute = instruction.Argument(0)->Resolve()->ImmediateU32();
    const auto channel = instruction.Argument(1)->Resolve()->ImmediateU32();
    const auto mode = instruction.Argument(2)->Resolve()->ImmediateU32();
    if (mode > 2u) state.Fail(instruction, "interpolation parameter mode is out of range");
    const auto& pixel = PixelInfo(state);
    if (!half) {
        if (pixel.InputIsDefault(attribute)) {
            state.Define(instruction, mode == 2u ? PixelAttribute(state, instruction, attribute, channel) : std::string("0u"));
            return;
        }
        if (!InputIsPerVertex(state, attribute)) {
            state.Define(instruction, PixelAttribute(state, instruction, attribute, channel));
            return;
        }
        const auto location = PixelLocation(state, instruction, attribute);
        std::string value = PerVertexValue(state, location, (mode + 1u) % 3u, channel);
        if (!pixel.InputIsCustom(attribute) && mode < 2u) value = "(" + value + " - " + PerVertexValue(state, location, 0u, channel) + ")";
        state.Define(instruction, "as_type<uint>(" + value + ")");
        return;
    }
    const bool high = instruction.Argument(3)->Resolve()->ImmediateU32() != 0u;
    if (pixel.InputHalfIsDefault(attribute, high)) {
        state.Define(instruction, "as_type<float>(" + MslHex(mode == 2u ? pixel.InputHalfDefaultBits(attribute, channel & 3u, high) : 0u) + "u)");
        return;
    }
    if (!InputIsPerVertex(state, attribute)) state.Fail(instruction, "16-bit interpolation parameter is not read per vertex");
    const auto location = PixelLocation(state, instruction, attribute);
    const auto corner = [&](std::uint32_t vertex) {
        return "float(as_type<half>(ushort(as_type<uint>(" + PerVertexValue(state, location, vertex, channel) + ")" + (high ? " >> 16u" : "") + ")))";
    };
    if (mode == 2u) state.Define(instruction, corner(0u));
    else state.Define(instruction, "float(half(" + corner(mode + 1u) + " - " + corner(0u) + "))");
}

std::uint32_t MrtOutputMode(MslEmitterState& state, const ExportInfo& exp) {
    if (StageOf(state) != IrShaderStage::Pixel || exp.kind != ExportTargetKind::Mrt) return 0u;
    const auto& pixel = PixelInfo(state);
    if (exp.index >= std::size(pixel.targetOutputMode)) return 0u;
    return pixel.targetOutputMode[exp.index];
}

const StageOutput* FindOutput(MslEmitterState& state, StageOutputKind kind, std::uint32_t index) {
    for (const auto& output : state.program.Info().outputs) {
        if (output.kind == kind && output.index == index) return &output;
    }
    return nullptr;
}

bool HasOutput(MslEmitterState& state, StageOutputKind kind) {
    return std::ranges::any_of(state.program.Info().outputs, [kind](const StageOutput& output) { return output.kind == kind; });
}

std::uint32_t ClipDistanceCount(MslEmitterState& state) {
    std::uint32_t count = 0u;
    for (const auto& output : state.program.Info().outputs) {
        if (output.kind == StageOutputKind::ClipDistance) count = std::max(count, output.index + 1u);
    }
    return count;
}

// ExportVector: the export's four components as uint bits (float bits unless uintOutput).
std::string ExportVector(MslEmitterState& state, const std::string& data, const ExportInfo& exp, bool uintOutput) {
    const auto mode = MrtOutputMode(state, exp);
    const auto name = state.Temporary("aps5_ev");
    if (exp.compr && !uintOutput) {
        const auto unpack = [&](const std::string& word) {
            if (mode == 5u) return "unpack_unorm2x16_to_float(" + word + ")";
            if (mode == 6u) return "unpack_snorm2x16_to_float(" + word + ")";
            return "float2(as_type<half2>(" + word + "))";
        };
        state.Line("float4 " + name + "f = float4(0.0f, 0.0f, 0.0f, 1.0f);");
        for (std::uint32_t pair = 0; pair < 2u; ++pair) {
            if ((exp.en & (3u << (pair * 2u))) == 0u) continue;
            const auto unpacked = state.Temporary("aps5_up");
            state.Line("const float2 " + unpacked + " = " + unpack(data + "[" + std::to_string(pair) + "]") + ";");
            for (std::uint32_t lane = 0; lane < 2u; ++lane) {
                const auto component = pair * 2u + lane;
                if (((exp.en >> component) & 1u) != 0u) state.Line(name + "f[" + std::to_string(component) + "] = " + unpacked + "[" + std::to_string(lane) + "];");
            }
        }
        state.Line("const uint4 " + name + " = as_type<uint4>(" + name + "f);");
        return name;
    }
    state.Line("uint4 " + name + " = uint4(0u, 0u, 0u, " + std::string(uintOutput ? "1u" : "0x3f800000u") + ");");
    // Upstream 26a746a43: 32_AR (mode 3) exports channels 0 and 1, and channel 1 is the alpha (GFX10).
    const auto exported = exp.en & (mode == 1u ? 0x1u : mode == 2u || mode == 3u ? 0x3u : 0xfu);
    if (exp.compr) {
        for (std::uint32_t pair = 0; pair < 2u; ++pair) {
            if ((exp.en & (3u << (pair * 2u))) == 0u) continue;
            for (std::uint32_t lane = 0; lane < 2u; ++lane) {
                const auto component = pair * 2u + lane;
                if (((exp.en >> component) & 1u) == 0u) continue;
                state.Line(name + "[" + std::to_string(component) + "] = extract_bits(" + data + "[" + std::to_string(pair) + "], " + U32(lane * 16u) + ", 16u);");
            }
        }
    } else {
        for (std::uint32_t component = 0; component < 4u; ++component) {
            if (((exported >> component) & 1u) != 0u) state.Line(name + "[" + std::to_string(mode == 3u && component == 1u ? 3u : component) + "] = " + data + "[" + std::to_string(component) + "];");
        }
    }
    return name;
}

// The output record an export writes: the vertex function's aps5_out, or in a mesh function this
// lane half's staged vertex (EmitMslEntryPoint copies it into the mesh after the guest program).
std::string OutputRecord(MslEmitterState& state) {
    if (StageOf(state) == IrShaderStage::Mesh) return "aps5_mv[" + std::to_string(state.half) + "]";
    return "aps5_out";
}

void AuxPositionExport(MslEmitterState& state, const std::string& data, const ExportInfo& exp) {
    const auto& vertex = VertexInfo(state);
    const bool mesh = StageOf(state) == IrShaderStage::Mesh;
    for (std::uint32_t component = 0; component < 4u; ++component) {
        if ((exp.en & (1u << component)) == 0u) continue;
        const auto output = DecodePositionExportComponent(vertex.paClVsOutCntl, exp.index, component);
        const auto raw = data + "[" + std::to_string(component) + "]";
        // DefineMeshOutputs: mesh functions output the position, parameters, the layer and (upstream
        // f7a2903ed) clip distances; cull distances are staged per vertex in threadgroup memory
        // (aps5_mcull) and turned into the primitive's cull flag by the mesh epilogue.
        if (mesh && (output.viewport || output.pointSize)) state.Fail("unsupported mesh output kind (viewport index or point size)");
        if (output.layer && mesh) {
            // Per vertex in threadgroup memory; each primitive takes its provoking vertex's layer.
            const auto index = state.GuestInvocationIndex();
            state.Line("if (" + index + " < " + U32(vertex.mesh.maxVertices) + ") aps5_mlayer[" + index + "] = " + raw + " & 0x7ffu;");
        } else if (output.layer) {
            state.Line("aps5_out.aps5_layer = " + raw + " & 0x7ffu;");
        }
        if (output.viewport) state.Line("aps5_out.aps5_viewport = extract_bits(" + raw + ", 16u, 4u);");
        if (output.layer || output.viewport) continue;
        if (output.pointSize) {
            state.Line("aps5_out.aps5_point_size = as_type<float>(" + raw + ");");
            continue;
        }
        if (output.clipDistance != std::numeric_limits<std::uint32_t>::max()) state.Line(OutputRecord(state) + ".aps5_clip[" + std::to_string(output.clipDistance) + "] = as_type<float>(" + raw + ");");
        // Upstream #423: Metal has no cull distance output. The guest program's distances go to the
        // vertex function's per-vertex record (aps5_cull, EmitMslEntryPoint), which turns them into
        // one clip distance per primitive (MslAbi Version 10).
        // Mesh functions: no per-vertex record survives to the primitive, so the distance goes to
        // threadgroup memory, read by the epilogue's per-primitive verdict (MeshCullTest).
        if (output.cullDistance != std::numeric_limits<std::uint32_t>::max() && mesh) {
            const auto index = state.GuestInvocationIndex();
            state.Line("if (" + index + " < " + U32(vertex.mesh.maxVertices) + ") aps5_mcull[" + index + " * " + U32(MslCullDistanceCount(state)) + " + " + U32(output.cullDistance) + "] = as_type<float>(" + raw + ");");
        } else if (output.cullDistance != std::numeric_limits<std::uint32_t>::max()) {
            state.Line("aps5_cull[" + std::to_string(output.cullDistance) + "] = as_type<float>(" + raw + ");");
        }
    }
}

// GS_ALLOC_REQ (EmitMeshAllocate): the first invocation stores M0's vertex count (bits 0-9) and
// primitive count (bits 12-22) for the outputs written after the program.
void MeshAllocate(MslEmitterState& state, const IrValue& instruction) {
    if (StageOf(state) != IrShaderStage::Mesh) state.Fail(instruction, "GS_ALLOC_REQ outside a mesh function");
    const auto value = state.Arg(instruction, 0u);
    state.Line("if (" + state.GuestInvocationIndex() + " == 0u) {");
    state.Indent();
    state.Line("aps5_malloc[0] = extract_bits(uint(" + value + "), 0u, 10u);");
    state.Line("aps5_malloc[1] = extract_bits(uint(" + value + "), 12u, 11u);");
    state.Outdent();
    state.Line("}");
}

// Mesh draw words: push dwords MeshDrawPushOffsetBytes / 4 + i of the stage's push data (slot 0, as
// the Vulkan push range): {index count, first vertex, first instance, index size | restart table,
// argument address low, high}.
std::string MeshDrawWord(MslEmitterState& state, const IrValue& instruction, std::uint32_t index, std::uint32_t count) {
    if (StageOf(state) != IrShaderStage::Mesh || index >= count) state.Fail(instruction, "invalid mesh draw parameter");
    state.usesPushData = true;
    return "aps5_push[" + U32(MeshDrawPushOffsetBytes / 4u + index) + "]";
}

std::string MeshArgumentAddress(MslEmitterState& state, const IrValue& instruction) {
    const auto low = MeshDrawWord(state, instruction, MeshArgumentAddressDword, PushData::MeshDrawDwordCount);
    const auto high = MeshDrawWord(state, instruction, MeshArgumentAddressDword + 1u, PushData::MeshDrawDwordCount);
    const auto name = state.Temporary("aps5_ma");
    state.Line("const ulong " + name + " = ulong(" + low + ") | (ulong(" + high + ") << 32u);");
    return name;
}

// The indirect draw's argument record (MeshArguments helper), 0 without one (EmitMeshArgument).
void MeshArgument(MslEmitterState& state, const IrValue& instruction) {
    const auto index = instruction.Argument(0)->Resolve()->ImmediateU32();
    if (StageOf(state) != IrShaderStage::Mesh || index >= MeshArgumentBytes / 4u) state.Fail(instruction, "invalid mesh argument");
    const auto address = MeshArgumentAddress(state, instruction);
    state.Define(instruction, address + " != 0ul ? reinterpret_cast<device const uint*>(" + address + ")[" + U32(index) + "] : 0u");
}

// Strip restart table (EmitMeshRestartStart): entry min(position, length - 1) when the condition holds.
void MeshRestartStart(MslEmitterState& state, const IrValue& instruction) {
    if (StageOf(state) != IrShaderStage::Mesh) state.Fail(instruction, "mesh restart table read outside a mesh shader");
    const auto address = MeshArgumentAddress(state, instruction);
    const auto table = "reinterpret_cast<device const uint*>(" + address + ")";
    state.Define(instruction, "(" + state.Arg(instruction, 1u) + ") ? " + table + "[min(uint(" + state.Arg(instruction, 0u) + "), " + table + "[" + U32(MeshRestartLengthDword) + "] - 1u) + " + U32(MeshRestartTableDword) + "] : 0u");
}

// SelectExportPacking: 10_11_11 UNORM and (upstream 49b0e805f) 11_11_10 float targets take the
// colour packed into the red channel of a 32-bit uint attachment.
std::string ExportPacking(MslEmitterState& state, std::uint32_t target, const std::string& color) {
    state.RequirePrelude(MslPreludeChunk::StageIo);
    const auto packing = state.FunctionConstant(PipelineSpecialization::ExportPackingBase + target, static_cast<std::uint32_t>(ColorExportPacking::None));
    const auto name = state.Temporary("aps5_pk");
    const auto floating = packing + " == " + U32(static_cast<std::uint32_t>(ColorExportPacking::Float11_11_10));
    state.Line("const uint4 " + name + " = " + packing + " != " + U32(static_cast<std::uint32_t>(ColorExportPacking::None)) + " ? uint4(aps5_pack_export(" + color + ", " + floating + " ? APS5_FORMAT_10_11_11_FLOAT : APS5_FORMAT_11_11_10_UNORM), 0u, 0u, 0u) : " + color + ";");
    return name;
}

// Upstream 16ea32c97 / c434467f6 (SpecializeFragmentOutputs): with APS5_REVERSED_BLEND (the driver
// sets it when PipelineSpecialization::ReversedBlend is not 0) the MRT0 store also writes the second
// blend source color(0) index(1) from the SecondShift routes and, when the ColorShift routes are not
// all Own, stores the ColorShift routing of the export instead; both come from copies of the export.
void EmitReversedBlend(MslEmitterState& state) {
    if (state.declarations.find("aps5_reversed_route") == std::string::npos) state.AddDeclaration(ReversedBlendRouteText);
    const auto code = state.FunctionConstant(PipelineSpecialization::ReversedBlend, 0u);
    const auto source = state.Temporary("aps5_rs");
    state.Line(std::string("#if defined(") + MslAbi::ReversedBlendMacro + ") && " + MslAbi::ReversedBlendMacro);
    state.Line("const float4 " + source + " = float4(aps5_out.c0);");
    state.Line("aps5_out.aps5_rb = aps5_reversed_route(" + source + ", (" + code + " >> " + U32(ReversedBlendCode::SecondShift) + ") & " + U32(ReversedBlendCode::Routes) + ");");
    state.Line("if (((" + code + " >> " + U32(ReversedBlendCode::ColorShift) + ") & " + U32(ReversedBlendCode::Routes) + ") != 0u) aps5_out.c0 = APS5_COLOR0_TYPE(aps5_reversed_route(" + source + ", (" + code + " >> " + U32(ReversedBlendCode::ColorShift) + ") & " + U32(ReversedBlendCode::Routes) + "));");
    state.Line("#endif");
}

void SetAttribute(MslEmitterState& state, const IrValue& instruction) {
    const auto stage = StageOf(state);
    if (stage != IrShaderStage::Vertex && stage != IrShaderStage::Pixel && stage != IrShaderStage::Mesh && stage != IrShaderStage::TessellationEvaluation) state.Fail(instruction, "exports of this stage are not implemented in the MSL backend yet (work package B6)");
    const auto& exports = state.program.Metadata().exportInfo;
    const auto index = instruction.Flags<ExportFlags>().index;
    if (index >= exports.size()) state.Fail(instruction, "export metadata index is out of range");
    const auto& exp = exports[index];
    const auto exec = state.Arg(instruction, 1u);
    if (stage == IrShaderStage::Pixel && exp.vm && state.requirements.pixelValidMask) {
        state.usesValidMask = true;
        state.Line("aps5_valid = " + exec + " ? 1u : 0u;");
    }
    if (exp.kind == ExportTargetKind::Null || exp.en == 0u || (exp.kind == ExportTargetKind::Primitive && stage != IrShaderStage::Mesh)) return;
    // A pixel program's pos0 export has no fragment output to write (SpirvModuleEmitter.cpp finds
    // no output variable for it and drops it the same way).
    if (stage == IrShaderStage::Pixel && exp.kind == ExportTargetKind::Position && exp.index == 0u) return;
    state.Line("if (" + exec + ") {");
    state.Indent();
    const auto data = state.Temporary("aps5_ed");
    state.Line("const uint4 " + data + " = uint4(" + state.Arg(instruction, 0u) + ");");
    const auto record = OutputRecord(state);
    if (exp.kind == ExportTargetKind::Primitive) {
        // The packed primitive (three 10-bit vertex fields, bit 31 null), unpacked after the program.
        state.Line("aps5_mprim[" + std::to_string(state.half) + "] = " + data + ".x;");
        state.Outdent();
        state.Line("}");
        return;
    }
    if (exp.kind == ExportTargetKind::Position && exp.index != 0u) {
        AuxPositionExport(state, data, exp);
    } else if (exp.kind == ExportTargetKind::MrtZ) {
        if ((exp.en & 1u) != 0u && HasOutput(state, StageOutputKind::Depth)) state.Line("aps5_out.aps5_depth = as_type<float>(" + data + ".x);");
        if ((exp.en & 2u) != 0u && HasOutput(state, StageOutputKind::StencilRef)) state.Line("aps5_out.aps5_stencil = " + data + ".y;");
        if ((exp.en & 4u) != 0u && HasOutput(state, StageOutputKind::SampleMask)) state.Line("aps5_out.aps5_sample_mask = " + data + ".z;");
    } else if (exp.kind == ExportTargetKind::Position) {
        auto position = "as_type<float4>(" + ExportVector(state, data, exp, false) + ")";
        const auto& vertex = VertexInfo(state);
        if (vertex.clipSpace.enabled) {
            const auto& transform = vertex.clipSpace;
            const auto converted = state.Temporary("aps5_cs");
            state.Line("float4 " + converted + " = " + position + ";");
            for (std::uint32_t c = 0; c < 2u; ++c) {
                const auto f = [](float value) { return "as_type<float>(" + MslHex(std::bit_cast<std::uint32_t>(value)) + "u)"; };
                state.Line(converted + "[" + std::to_string(c) + "] = (" + converted + "[" + std::to_string(c) + "] * " + f(transform.scale[c]) + " + " + f(transform.offset[c]) + ") / " + f(transform.halfExtent[c]) + " - 1.0f;");
            }
            position = converted;
        }
        state.Line(record + ".aps5_position = " + position + ";");
        // The all-zero position cull distance (StoreZeroPositionClipDistance; not on mesh outputs).
        if (stage != IrShaderStage::Mesh) state.Line("aps5_out.aps5_clip[" + std::to_string(ClipDistanceCount(state)) + "] = ((as_type<uint4>(aps5_out.aps5_position).x | as_type<uint4>(aps5_out.aps5_position).y | as_type<uint4>(aps5_out.aps5_position).z | as_type<uint4>(aps5_out.aps5_position).w) & 0x7fffffffu) == 0u ? -1.0f : 0.0f;");
    } else {
        const auto kind = exp.kind == ExportTargetKind::Mrt ? StageOutputKind::Mrt : StageOutputKind::Parameter;
        const auto* output = FindOutput(state, kind, exp.index);
        if (output != nullptr) {
            const auto mode = MrtOutputMode(state, exp);
            const bool uintOutput = mode == 7u;
            auto value = ExportVector(state, data, exp, uintOutput);
            if (stage == IrShaderStage::Pixel && exp.kind == ExportTargetKind::Mrt) {
                if (exp.index >= 8u) state.Fail(instruction, "fragment export target exceeds the runtime mapping table");
                std::string mapped = "uint4(";
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    const auto selector = state.FunctionConstant(PipelineSpecialization::ExportBase + exp.index * 4u + component, component);
                    mapped += (component == 0u ? "" : ", ") + value + "[" + selector + " & 3u]";
                }
                const auto name = state.Temporary("aps5_em");
                state.Line("const uint4 " + name + " = " + mapped + ");");
                value = name;
                if (exp.index == 0u && !uintOutput && PixelInfo(state).psAlphaToCoverage) {
                    if (PixelInfo(state).psDitheredAlphaToCoverage) {
                        if (state.declarations.find("aps5_dithered_alpha_to_mask") == std::string::npos) state.AddDeclaration(DitheredAlphaToMaskText);
                        state.Line("aps5_out.aps5_sample_mask = aps5_dithered_alpha_to_mask(as_type<float>(" + value + ".w), aps5_in.aps5_position);");
                    } else {
                        if (state.declarations.find("aps5_alpha_to_mask") == std::string::npos) state.AddDeclaration(AlphaToMaskText);
                        state.Line("aps5_out.aps5_sample_mask = aps5_alpha_to_mask(as_type<float>(" + value + ".w), APS5_SAMPLE_COUNT);");
                    }
                }
                if (!uintOutput && mode != 5u && mode != 6u) value = ExportPacking(state, exp.index, value);
                state.Line("aps5_out.c" + std::to_string(output->location) + " = as_type<APS5_COLOR" + std::to_string(output->location) + "_TYPE>(" + value + ");");
                if (output->location == 0u && !uintOutput) EmitReversedBlend(state);
            } else {
                state.Line(record + ".p" + std::to_string(output->location) + " = as_type<float4>(" + value + ");");
            }
        }
    }
    state.Outdent();
    state.Line("}");
}

constexpr std::string_view StageIoText = R"(// Vertex pulling (MslAbi::VertexFetchRecord): one attribute location of the vertex or instance, in
// the format of the record's V# dword 3; components the format lacks read (0, 0, 0, 1), a null record
// (bytes 0) reads zeros.
struct Aps5VertexFetch {
    ulong address;
    uint stride;
    uint bytes;
    uint word3;
    uint rate;
    uint components;
    uint reserved;
};

static uint4 aps5_vertex_fetch(device const Aps5VertexFetch* records, uint location, uint vertexIndex, uint instanceIndex) {
    const Aps5VertexFetch record = records[location];
    const uint index = record.rate != 0u ? instanceIndex : vertexIndex;
    const uint format = extract_bits(record.word3, 12u, 7u);
    const Aps5Format info = aps5_format(format);
    if (record.bytes == 0u || info.count == 0u) return uint4(0u);
    uint dstSel = 0u;
    for (uint output = 0u; output < 4u; ++output) dstSel |= (output < info.count ? 4u + output : (output == 3u ? 1u : 0u)) << (output * 3u);
    const uint misalignment = uint(record.address & 3ul);
    const Aps5Buffer buffer {record.address - ulong(misalignment), record.bytes + misalignment, 0u};
    const Aps5BufferMemory memory {buffer, 0u, misalignment, false};
    const uint records_ = record.stride != 0u ? record.bytes / record.stride : 0xffffffffu;
    // Index-bounded (out-of-bounds select 1), stride in dword 1.
    const uint word3 = (record.word3 & ~(3u << 28u)) | (1u << 28u);
    bool stop = false;
    return aps5_format_load(memory, info, record.stride << 16u, records_, word3, true, index, 0u, 0u, dstSel, 4u, false, stop);
}

// SelectExportPacking's PackExport: the 10_11_11 UNORM or 11_11_10 float packing of an RGB colour
// (float bits).
static uint aps5_pack_export(uint4 color, uint format) {
    const Aps5Format info = aps5_format(format);
    uint packed = 0u;
    for (uint component = 0u; component < info.count; ++component) packed |= aps5_format_store(info.type, info.bits[component], color[component]) << info.offsets[component];
    return packed;
}

)";

constexpr std::array<MslPreludeChunk, 2> OnBufferFormat {MslPreludeChunk::RuntimeBuffer, MslPreludeChunk::BufferFormat};

}

bool EmitMslStageInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::GetUserData: {
        const auto registerIndex = instruction.Argument(0)->Resolve()->Register().index;
        const auto& registers = state.program.Metadata().bindings.userDataRegisters;
        const auto found = std::lower_bound(registers.begin(), registers.end(), registerIndex);
        if (found == registers.end() || *found != registerIndex) state.Define(instruction, "0u");
        else state.Define(instruction, state.ShaderDataDword(static_cast<std::uint32_t>(found - registers.begin())));
        return true;
    }
    case IrOpcode::GetBuiltin: {
        const auto kind = static_cast<StageInputKind>(instruction.Argument(0)->Resolve()->ImmediateU32());
        const auto component = instruction.Argument(1)->Resolve()->ImmediateU32();
        if (state.ThreadgroupStage()) state.Define(instruction, ComputeBuiltin(state, kind, component));
        else state.Define(instruction, GraphicsBuiltin(state, instruction, kind, component));
        return true;
    }
    case IrOpcode::GetAttribute:
        GetAttribute(state, instruction);
        return true;
    case IrOpcode::SetAttribute:
        SetAttribute(state, instruction);
        return true;
    case IrOpcode::MeshAllocate:
        MeshAllocate(state, instruction);
        return true;
    case IrOpcode::MeshDrawParameter:
        state.Define(instruction, MeshDrawWord(state, instruction, instruction.Argument(0)->Resolve()->ImmediateU32(), PushData::MeshDrawDwordCount));
        return true;
    case IrOpcode::MeshArgument:
        MeshArgument(state, instruction);
        return true;
    case IrOpcode::MeshRestartStart:
        MeshRestartStart(state, instruction);
        return true;
    case IrOpcode::GetTessellationAttribute:
    case IrOpcode::SetTessellationAttribute:
        return EmitMslTessellationInstruction(state, instruction);
    case IrOpcode::GetInterpolationParameter:
        GetInterpolationParameter(state, instruction, false);
        return true;
    case IrOpcode::GetInterpolationParameterF16:
        GetInterpolationParameter(state, instruction, true);
        return true;
    default:
        return false;
    }
}

MslGraphicsInterface MslGraphicsInterfaceOf(MslEmitterState& state) {
    MslGraphicsInterface result;
    const auto stage = StageOf(state);
    std::string& text = result.declarations;
    if (stage == IrShaderStage::Vertex || stage == IrShaderStage::TessellationEvaluation) {
        // Guest clip distances, the all-zero position distance, then (cull distances written) the
        // primitive's cull distance (MslAbi Version 10, EmitMslEntryPoint).
        const auto culls = MslCullDistanceCount(state);
        if (culls != 0u && stage == IrShaderStage::TessellationEvaluation) state.Fail("domain programs that write cull distances (the tessellation mesh function has no cull distance emulation; TechnicalDebt metal-tessellation)");
        if (culls > MaxClipDistances) state.Fail("more than 8 guest cull distances");
        const auto clips = ClipDistanceCount(state) + 1u + (culls != 0u ? 1u : 0u);
        if (clips > MaxClipDistances) state.Fail(culls != 0u ? "guest clip distances plus the all-zero clip position distance and the emulated cull distance exceed 8 clip distances" : "guest clip distances plus the all-zero clip position distance exceed 8 clip distances");
        state.reflection.cullDistances = culls;
        text += "struct Aps5VertexOut {\n    float4 aps5_position [[position]];\n";
        std::uint32_t scalars = 4u;
        for (const auto& output : state.program.Info().outputs) {
            switch (output.kind) {
            case StageOutputKind::Parameter:
                if (output.location >= 64u) state.Fail("vertex parameter location " + std::to_string(output.location) + " is beyond 63");
                text += "    float4 p" + std::to_string(output.location) + " [[user(locn" + std::to_string(output.location) + ")]];\n";
                state.reflection.outputLocations |= std::uint64_t {1u} << output.location;
                scalars += 4u;
                break;
            case StageOutputKind::PointSize: text += "    float aps5_point_size [[point_size]];\n"; break;
            case StageOutputKind::Layer:
            case StageOutputKind::ViewportIndex:
                // Domain programs run in mesh functions, whose layer and viewport are per primitive.
                if (stage == IrShaderStage::TessellationEvaluation) state.Fail("domain programs that write the layer or viewport index (Metal mesh vertices have neither; TechnicalDebt metal-tessellation)");
                text += output.kind == StageOutputKind::Layer ? "    uint aps5_layer [[render_target_array_index]];\n" : "    uint aps5_viewport [[viewport_array_index]];\n";
                break;
            default: break;
            }
        }
        // Tessellation point output (LinkMslTessellation defines APS5_TESS_POINTS): Metal needs a
        // point size, and a domain program without one draws 1-pixel points (TechnicalDebt
        // metal-tessellation: PA_SU_POINT_SIZE is not applied).
        const bool pointSize = std::any_of(state.program.Info().outputs.begin(), state.program.Info().outputs.end(), [](const auto& output) { return output.kind == StageOutputKind::PointSize; });
        if (stage == IrShaderStage::TessellationEvaluation && !pointSize) text += "#ifdef APS5_TESS_POINTS\n    float aps5_tess_point_size [[point_size]];\n#define APS5_TESS_POINT_SIZE_FIELD 1\n#endif\n";
        if (stage == IrShaderStage::Vertex && !pointSize) text += "#if defined(" + std::string(MslAbi::PointsMacro) + ") && " + std::string(MslAbi::PointsMacro) + "\n    float aps5_default_point_size [[point_size]];\n#endif\n";
        // Metal's compiler fails past 124 scalar varyings (M2_DESIGN §4).
        if (scalars > 124u) state.Fail("the vertex function exceeds 124 scalar varyings");
        // MslAbi Version 5, triangle-vertices draws: each per-vertex location the fragment function
        // reads (APS5_PER_VERTEX_LOCATIONS) also goes out as its triangle's three corner values,
        // flat; the corner weights stand in for the barycentrics (APS5_BARYCENTRICS); conservative
        // rasterization adds its triangle (MslConservative). The driver keeps the total within
        // Metal's varying limit (the library compile fails otherwise).
        text += TriangleVertexMacroDefaults();
        text += "#if " + TriangleVerticesCondition() + "\n";
        for (const auto& output : state.program.Info().outputs) {
            if (output.kind != StageOutputKind::Parameter) continue;
            text += "#if " + PerVertexLocationCondition(output.location) + "\n";
            for (std::uint32_t vertex = 0; vertex < 3u; ++vertex) text += "    float4 " + PerVertexField(output.location, vertex) + " [[user(aps5_" + PerVertexField(output.location, vertex) + ")]];\n";
            text += "#endif\n";
        }
        text += "#if (APS5_BARYCENTRICS & 1)\n    float3 aps5_bw [[user(aps5_bw)]];\n#endif\n";
        text += "#if (APS5_BARYCENTRICS & 2)\n    float3 aps5_bwl [[user(aps5_bwl)]];\n#endif\n";
        text += "#if defined(APS5_CONSERVATIVE) && APS5_CONSERVATIVE\n" + MslConservative::VertexOutputFields() + "#endif\n";
        text += "#endif\n";
        text += "    float aps5_clip [[clip_distance]] [" + std::to_string(clips) + "];\n};\n\n";
        result.outputType = "Aps5VertexOut";
        // Metal has no depth clip control: a [-w, w] guest depth range is remapped to [0, w]. The y
        // flip between Vulkan and Metal lives in the driver's viewport (MetalDraw.mm); APS5_FLIP_Y is
        // a debug aid the driver never defines.
        result.epilogue = WindowSpaceEpilogue("aps5_out.aps5_position", "    ") +
                          "#if defined(APS5_DEPTH_NEGATIVE_ONE_TO_ONE) && APS5_DEPTH_NEGATIVE_ONE_TO_ONE\n"
                          "    aps5_out.aps5_position.z = (aps5_out.aps5_position.z + aps5_out.aps5_position.w) * 0.5f;\n"
                          "#endif\n"
                          "#if defined(APS5_FLIP_Y) && APS5_FLIP_Y\n"
                          "    aps5_out.aps5_position.y = -aps5_out.aps5_position.y;\n"
                          "#endif\n";
        return result;
    }
    if (stage == IrShaderStage::Mesh) {
        // Mesh functions (NGG primitive shaders, as DefineMeshOutputs): the vertex record carries the
        // position and the parameters, the primitive record the cull flag and the layer.
        const auto meshOutput = MslMeshOutputOf(VertexInfo(state).mesh);
        const auto subgroup = meshOutput.rectList ? std::string(" (rect list: guest vertices plus one per rect)") : std::string();
        if (meshOutput.vertices == 0u || meshOutput.vertices > 256u) state.Fail("mesh functions output 1 to 256 vertices (Metal mesh<> limit), the subgroup has " + std::to_string(meshOutput.vertices) + subgroup);
        if (meshOutput.primitives == 0u || meshOutput.primitives > 512u) state.Fail("mesh functions output 1 to 512 primitives (Metal mesh<> limit), the subgroup has " + std::to_string(meshOutput.primitives) + subgroup);
        text += "struct Aps5MeshVertex {\n    float4 aps5_position [[position]];\n";
        std::uint32_t scalars = 4u;
        bool layer = false;
        for (const auto& output : state.program.Info().outputs) {
            switch (output.kind) {
            case StageOutputKind::Position: break;
            case StageOutputKind::Parameter:
                if (output.location >= 64u) state.Fail("mesh parameter location " + std::to_string(output.location) + " is beyond 63");
                text += "    float4 p" + std::to_string(output.location) + " [[user(locn" + std::to_string(output.location) + ")]];\n";
                state.reflection.outputLocations |= std::uint64_t {1u} << output.location;
                scalars += 4u;
                break;
            case StageOutputKind::Layer: layer = true; break;
            case StageOutputKind::ClipDistance: break;
            case StageOutputKind::CullDistance:
                // Upstream f7a2903ed writes gl_CullDistance per mesh vertex; Metal has no cull
                // distance output: the mesh epilogue culls a primitive whose vertices all have a
                // negative distance in one slot (as D30 does for vertex functions; MeshCullTest).
                if (MslCullDistanceCount(state) > MaxClipDistances) state.Fail("more than 8 guest cull distances");
                break;
            default: state.Fail("mesh functions output the position, parameters, layer and clip distances only: point size and viewport index of NGG programs are not implemented (TechnicalDebt metal-mesh-functions)");
            }
        }
        // Upstream f7a2903ed: guest clip distances per mesh vertex (no all-zero-position distance on
        // mesh outputs, as the SPIR-V path).
        if (const auto clips = MslCullClipSlot(state) - 1u; clips != 0u) {
            if (clips > MaxClipDistances) state.Fail("more than 8 guest clip distances");
            text += "    float aps5_clip [[clip_distance]] [" + std::to_string(clips) + "];\n";
            scalars += clips;
        }
        if (scalars > 124u) state.Fail("the mesh function exceeds 124 scalar varyings");
        // Point lists (MslAbi Version 16, DECISIONS D75): NGG programs export no point size here.
        text += "#if defined(" + std::string(MslAbi::PointsMacro) + ") && " + std::string(MslAbi::PointsMacro) + "\n    float aps5_default_point_size [[point_size]];\n#endif\n";
        // MslAbi Version 13, the triangle-vertices mode (APS5_TRIANGLE_VERTICES, as the vertex
        // function's D20 emulation): every output triangle has its own three vertices, which carry
        // one-hot corner weights as the barycentrics; the corners' values of the per-vertex locations
        // go in the primitive record (flat by definition, read by the fragment function's flat
        // user(aps5_pv<L>_<j>) inputs), so each is stored once per triangle, not per vertex.
        text += TriangleVertexMacroDefaults();
        text += "#if " + TriangleVerticesCondition() + "\n";
        text += "#if (APS5_BARYCENTRICS & 1)\n    float3 aps5_bw [[user(aps5_bw)]];\n#endif\n";
        text += "#if (APS5_BARYCENTRICS & 2)\n    float3 aps5_bwl [[user(aps5_bwl)]];\n#endif\n";
        text += "#endif\n";
        text += "};\n\nstruct Aps5MeshPrimitive {\n    bool aps5_culled [[primitive_culled]];\n";
        if (layer) text += "    uint aps5_layer [[render_target_array_index]];\n";
        text += "#if " + TriangleVerticesCondition() + "\n";
        for (const auto& output : state.program.Info().outputs) {
            if (output.kind != StageOutputKind::Parameter) continue;
            text += "#if " + PerVertexLocationCondition(output.location) + "\n";
            for (std::uint32_t vertex = 0; vertex < 3u; ++vertex) text += "    float4 " + PerVertexField(output.location, vertex) + " [[user(aps5_" + PerVertexField(output.location, vertex) + ")]];\n";
            text += "#endif\n";
        }
        text += "#endif\n";
        text += "};\n\n";
        if (meshOutput.rectList) {
            text += "using Aps5Mesh = metal::mesh<Aps5MeshVertex, Aps5MeshPrimitive, " + std::to_string(meshOutput.vertices) + ", " + std::to_string(meshOutput.primitives) + ", metal::topology::triangle>;\n\n";
        } else {
            const auto split = MslAbi::MeshTriangleVerticesPrimitives(meshOutput.primitives);
            // MslAbi Version 16 (DECISIONS D75): point and line draws output points and lines, one or
            // two of the primitive export's indices per primitive (PrimitiveVerticesMacro).
            const std::string primitiveVertices = MslAbi::PrimitiveVerticesMacro;
            // Upstream a0347cfed: an NGG geometry program with line-strip output
            // (ShaderMeshInputInfo::outputPrimitiveSize 2) exports lines whatever the draw's topology.
            const auto outputVertices = VertexInfo(state).mesh.outputPrimitiveSize;
            if (outputVertices != 2u && outputVertices != 3u) state.Fail("mesh functions output lines or triangles, the program outputs primitives of " + std::to_string(outputVertices) + " vertices");
            if (outputVertices == 2u) text += "#undef " + primitiveVertices + "\n#define " + primitiveVertices + " 2u\n";
            text += "#ifndef " + primitiveVertices + "\n#define " + primitiveVertices + " 3u\n#endif\n";
            text += "#if " + primitiveVertices + " == 1u\n#define APS5_MESH_TOPOLOGY metal::topology::point\n#elif " + primitiveVertices + " == 2u\n#define APS5_MESH_TOPOLOGY metal::topology::line\n#else\n#define APS5_MESH_TOPOLOGY metal::topology::triangle\n#endif\n";
            text += "#if " + TriangleVerticesCondition() + "\n";
            text += "using Aps5Mesh = metal::mesh<Aps5MeshVertex, Aps5MeshPrimitive, " + std::to_string(3u * split) + ", " + std::to_string(split) + ", APS5_MESH_TOPOLOGY>;\n";
            text += "#else\n";
            text += "using Aps5Mesh = metal::mesh<Aps5MeshVertex, Aps5MeshPrimitive, " + std::to_string(meshOutput.vertices) + ", " + std::to_string(meshOutput.primitives) + ", APS5_MESH_TOPOLOGY>;\n";
            text += "#endif\n\n";
        }
        result.outputType = "Aps5MeshVertex";
        return result;
    }
    if (stage != IrShaderStage::Pixel) state.Fail("only vertex, mesh and pixel graphics stages are implemented in the MSL backend so far (work package B6)");
    const auto& pixel = PixelInfo(state);
    text += "#ifndef APS5_VERTEX_OUTPUTS\n#define APS5_VERTEX_OUTPUTS 0xffffffff\n#endif\n#ifndef APS5_VERTEX_OUTPUTS_HI\n#define APS5_VERTEX_OUTPUTS_HI 0xffffffff\n#endif\n";
    text += "#ifndef APS5_SAMPLE_BASE\n#define APS5_SAMPLE_BASE 0\n#endif\n";
    text += "struct Aps5FragmentIn {\n    float4 aps5_position [[position]];\n";
    std::string accessors;
    using Mode = MslEmitterState::FragmentInput::Mode;
    constexpr std::array locations {InterpolationLocation::Center, InterpolationLocation::Centroid, InterpolationLocation::Sample};
    for (const auto& [location, input] : state.fragmentInputs) {
        const auto macro = location < 32u ? std::string("APS5_VERTEX_OUTPUTS") : std::string("APS5_VERTEX_OUTPUTS_HI");
        const auto guard = "#if ((" + macro + " >> " + std::to_string(location % 32u) + ") & 1)\n";
        const auto field = "p" + std::to_string(location);
        const auto name = "aps5_input_" + std::to_string(location);
        if (input.mode == Mode::Flat) {
            text += guard + "    float4 " + field + " [[user(locn" + std::to_string(location) + "), flat]];\n#endif\n";
            accessors += guard + "#define " + name + "(c) as_type<uint>(aps5_in." + field + "[c])\n#else\n#define " + name + "(c) 0u\n#endif\n";
            continue;
        }
        const std::string perspective = input.mode == Mode::Linear ? "no_perspective" : "perspective";
        // DECISIONS D38: several locations need pull-model interpolation (MSL specification 2.18,
        // 6.12; an interpolant<> stage_in member takes no interpolation qualifier).
        const bool pull = std::popcount(input.locations) > 1;
        if (pull) text += guard + "    interpolant<float4, interpolation::" + perspective + "> " + field + " [[user(locn" + std::to_string(location) + ")]];\n#endif\n";
        std::string defined;
        std::string zero;
        for (const auto at : locations) {
            if ((input.locations & (1u << static_cast<std::uint32_t>(at))) == 0u) continue;
            const auto accessor = name + "_" + LocationName(at);
            if (!pull) text += guard + "    float4 " + field + " [[user(locn" + std::to_string(location) + "), " + LocationName(at) + "_" + perspective + "]];\n#endif\n";
            const auto value = !pull ? "aps5_in." + field : at == InterpolationLocation::Sample ? "aps5_in." + field + ".interpolate_at_sample(aps5_sample_id)" : "aps5_in." + field + ".interpolate_at_" + LocationName(at) + "()";
            defined += "#define " + accessor + "(c) as_type<uint>(" + value + "[c])\n";
            zero += "#define " + accessor + "(c) 0u\n";
        }
        accessors += guard + defined + "#else\n" + zero + "#endif\n";
    }
    // MslAbi Version 5: per-vertex values and emulated barycentrics (triangle-vertices draws).
    for (std::uint32_t location = 0; location < 64u; ++location) {
        if (((state.reflection.perVertexLocations >> location) & 1u) == 0u) continue;
        const auto macro = location < 32u ? std::string("APS5_VERTEX_OUTPUTS") : std::string("APS5_VERTEX_OUTPUTS_HI");
        const auto guard = "#if ((" + macro + " >> " + std::to_string(location % 32u) + ") & 1)\n";
        text += guard;
        for (std::uint32_t vertex = 0; vertex < 3u; ++vertex) text += "    float4 " + PerVertexField(location, vertex) + " [[user(aps5_" + PerVertexField(location, vertex) + "), flat]];\n";
        text += "#endif\n";
        accessors += guard + "#define aps5_pv_" + std::to_string(location) + "(v, c) aps5_in.pv" + std::to_string(location) + "_##v[c]\n#else\n#define aps5_pv_" + std::to_string(location) + "(v, c) 0.0f\n#endif\n";
    }
    // The corner weights at each location read (DECISIONS D38): one location takes its qualifier,
    // several make the weights an interpolant<>.
    for (std::uint32_t linear = 0u; linear < 2u; ++linear) {
        if ((state.reflection.emulatedBarycentrics & (1u << linear)) == 0u) continue;
        const std::string field = linear != 0u ? "aps5_bwl" : "aps5_bw";
        const std::string perspective = linear != 0u ? "no_perspective" : "perspective";
        const auto read = state.barycentricLocations[linear];
        const bool pull = std::popcount(read) > 1;
        if (pull) text += "    interpolant<float3, interpolation::" + perspective + "> " + field + " [[user(" + field + ")]];\n";
        for (const auto at : locations) {
            if ((read & (1u << static_cast<std::uint32_t>(at))) == 0u) continue;
            if (!pull) text += "    float3 " + field + " [[user(" + field + "), " + LocationName(at) + "_" + perspective + "]];\n";
            const auto value = !pull ? "aps5_in." + field : at == InterpolationLocation::Sample ? "aps5_in." + field + ".interpolate_at_sample(aps5_sample_id)" : "aps5_in." + field + ".interpolate_at_" + LocationName(at) + "()";
            accessors += "#define " + field + "_" + LocationName(at) + " " + value + "\n";
        }
    }
    text += "#if defined(APS5_CONSERVATIVE) && APS5_CONSERVATIVE\n" + MslConservative::FragmentInputFields() + "#endif\n";
    text += "};\n" + accessors + "\n";
    std::string members;
    for (const auto& output : state.program.Info().outputs) {
        switch (output.kind) {
        case StageOutputKind::Mrt: {
            if (output.location >= 8u) state.Fail("colour output location " + std::to_string(output.location) + " is beyond 7");
            const auto location = std::to_string(output.location);
            const bool uintOutput = output.index < std::size(pixel.targetOutputMode) && pixel.targetOutputMode[output.index] == 7u;
            text += "#ifndef APS5_COLOR" + location + "_TYPE\n#define APS5_COLOR" + location + "_TYPE " + (uintOutput ? "uint4" : "float4") + "\n#endif\n";
            if (output.location == 1u) {
                // Dual-source blending (APS5_DUAL_SOURCE, the PipelineSpecialization::DualSourceBlend
                // flag as a macro): the second colour is color(0) index(1), as the SPIR-V path's
                // fragment output specialization moves location 1 to location 0 index 1.
                members += "#if defined(APS5_DUAL_SOURCE) && APS5_DUAL_SOURCE\n    APS5_COLOR1_TYPE c1 [[color(0), index(1)]];\n#else\n    APS5_COLOR1_TYPE c1 [[color(1)]];\n#endif\n";
                state.reflection.dualSource = true;
            } else {
                members += "    APS5_COLOR" + location + "_TYPE c" + location + " [[color(" + location + ")]];\n";
            }
            state.reflection.colorOutputs |= 1u << output.location;
            break;
        }
        case StageOutputKind::Depth: {
            const char* mode = pixel.psConservativeZExport == ConservativeZExport::LessThanZ ? "less" : pixel.psConservativeZExport == ConservativeZExport::GreaterThanZ ? "greater" : "any";
            members += std::string("    float aps5_depth [[depth(") + mode + ")]];\n";
            state.reflection.exportsDepth = true;
            break;
        }
        case StageOutputKind::StencilRef:
            members += "    uint aps5_stencil [[stencil]];\n";
            state.reflection.exportsStencil = true;
            break;
        case StageOutputKind::SampleMask:
            members += "    uint aps5_sample_mask [[sample_mask]];\n";
            state.reflection.exportsSampleMask = true;
            break;
        default:
            break;
        }
    }
    // Reversed-order blending (EmitReversedBlend): the second blend source of MRT0. Upstream refuses
    // it with dual-source blending, for a non-float MRT0 and next to other colour outputs.
    if ((state.reflection.colorOutputs & 1u) != 0u) {
        const std::string guard = std::string("#if defined(") + MslAbi::ReversedBlendMacro + ") && " + MslAbi::ReversedBlendMacro + "\n";
        const bool uintOutput0 = std::ranges::any_of(state.program.Info().outputs, [&](const StageOutput& output) { return output.kind == StageOutputKind::Mrt && output.location == 0u && output.index < std::size(pixel.targetOutputMode) && pixel.targetOutputMode[output.index] == 7u; });
        members += guard;
        if (uintOutput0) members += "#error \"reversed-order blending needs a float4 MRT0 output\"\n";
        if ((state.reflection.colorOutputs & ~1u) != 0u) members += "#error \"reversed-order blending needs MRT0 to be the only color output\"\n";
        members += "#if defined(APS5_DUAL_SOURCE) && APS5_DUAL_SOURCE\n#error \"reversed-order blending cannot be combined with dual-source blending\"\n#endif\n";
        members += "    float4 aps5_rb [[color(0), index(1)]];\n#endif\n";
    }
    // Upstream 1ffc5f672: a partial per-pixel PA_SC_AA_MASK writes its coverage (EmitMslEntryPoint).
    if ((pixel.psAlphaToCoverage || pixel.QuadSampleMaskPartial()) && !state.reflection.exportsSampleMask) {
        members += "    uint aps5_sample_mask [[sample_mask]];\n";
        state.reflection.exportsSampleMask = true;
    }
    // Upstream 1ffc5f672: a PA_SC_AA_MASK that is the same for the four quad pixels but leaves
    // samples out is the Vulkan pipeline's sample mask; MTLRenderPipelineDescriptor has none, so the
    // driver compiles the function with APS5_SAMPLE_MASK (the guest samples kept) and the function
    // ANDs its coverage with it (MTLCompileOptions.preprocessorMacros, as APS5_SAMPLE_BASE). A function without other
    // outputs returns a struct whose only member is that mask under the macro (empty without it, which Metal takes as
    // no outputs), so a depth- or stencil-writing draw of an output-less program keeps the mask off those writes.
    const bool hostMaskMember = !state.reflection.exportsSampleMask;
    if (hostMaskMember) members += "#if defined(APS5_SAMPLE_MASK)\n    uint aps5_sample_mask [[sample_mask]];\n#endif\n";
    if (!members.empty()) {
        text += "struct Aps5FragmentOut {\n" + members + "};\n\n";
        result.outputType = "Aps5FragmentOut";
    }
    // PS_ITER_SAMPLE: the stage function takes [[sample_id]] (EmitMslEntryPoint).
    if (pixel.psSampleShading) state.reflection.perSample = true;
    // MslAbi Version 9: the guest mask's bits of this pass's host samples (APS5_SAMPLE_BASE,
    // DECISIONS D28).
    const std::string toHost = "#if APS5_SAMPLE_BASE\n    aps5_out.aps5_sample_mask = aps5_out.aps5_sample_mask >> APS5_SAMPLE_BASE;\n#endif\n";
    if (state.reflection.exportsSampleMask) {
        result.epilogue += "#if defined(APS5_SAMPLE_MASK)\n    aps5_out.aps5_sample_mask &= APS5_SAMPLE_MASK;\n#endif\n" + toHost;
    } else if (hostMaskMember) {
        result.epilogue += "#if defined(APS5_SAMPLE_MASK)\n    aps5_out.aps5_sample_mask = APS5_SAMPLE_MASK;\n" + toHost + "#endif\n";
    }
    // Primitive-ordered pixel shaders (the SPIR-V path's PixelInterlockOrderedEXT around the whole
    // program): the stage function brackets the guest body with raster-order-group accesses
    // (EmitMslEntryPoint, BufferIndex::RasterOrder).
    if (pixel.psOrderedPixelShader) state.reflection.rasterOrder = true;
    // Upstream #4210 (SpirvModuleSetup DefineModule): the front end decides earlyZ (DB_SHADER_CONTROL
    // DEPTH_BEFORE_SHADER, or Z_ORDER 1/3 without kill and exports) and the backend keeps only the
    // alpha-to-coverage exception, so a killing or mask-exporting shader can test early. Metal also
    // refuses [[early_fragment_tests]] on a function that returns [[depth]] or [[stencil]] (MSL
    // specification 5.1.2), so depth and stencil exports keep late tests here (Vulkan ignores those
    // writes under EarlyFragmentTests; TechnicalDebt metal-early-depth-export).
    // Upstream 72518f64f: quad-masked pixel functions (they discard uncovered pixels) test late.
    if (pixel.psEarlyZ && !pixel.psAlphaToCoverage && !pixel.QuadMasked() && !pixel.psDepthExportEnable && !pixel.psStencilRefExportEnable) {
        // Conservative rasterization discards in the stage function (MslConservative), which
        // early fragment tests would not wait for.
        // MslAbi Version 15: a uniform PA_SC_AA_MASK (APS5_SAMPLE_MASK) is applied by the function's
        // [[sample_mask]], which Metal applies after early tests; Vulkan's pipeline mask removes the
        // samples before them, so those draws test late (MetalDraw refuses the memory-writing case).
        result.attributes = "#if !(defined(APS5_CONSERVATIVE) && APS5_CONSERVATIVE) && !defined(APS5_SAMPLE_MASK)\n[[early_fragment_tests]]\n#endif\n";
        state.reflection.earlyFragmentTests = true;
    }
    if (state.usesValidMask) {
        state.reflection.usesDiscard = true;
        result.epilogue += "    if (aps5_valid == 0u && !simd_is_helper_thread()) discard_fragment();\n";
    }
    return result;
}

MslPreludeEntry MslStagePrelude(MslPreludeChunk chunk) {
    if (chunk != MslPreludeChunk::StageIo) throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not a stage chunk (work package B6)");
    static const std::string text = "#define APS5_FORMAT_11_11_10_UNORM " + std::to_string(static_cast<std::uint32_t>(IrBufferFormat::Format11_11_10UNorm)) + "u\n#define APS5_FORMAT_10_11_11_FLOAT " + std::to_string(static_cast<std::uint32_t>(IrBufferFormat::Format10_11_11Float)) + "u\n" + std::string(StageIoText);
    return {text, OnBufferFormat};
}

}
