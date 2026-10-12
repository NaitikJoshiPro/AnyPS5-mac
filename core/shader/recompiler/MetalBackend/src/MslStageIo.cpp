#include "MetalBackend/MslEmitterState.hpp"
#include "PipelineSpecialization.hpp"
#include "IntermediateRepresentation/IrMetadata/BufferFormat.hpp"
#include <algorithm>
#include <bit>
#include <array>
#include <limits>
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

// EmitBuiltinU32 for vertex and pixel programs: the value's bits as a uint.
std::string GraphicsBuiltin(MslEmitterState& state, const IrValue& instruction, StageInputKind kind, std::uint32_t component) {
    const auto stage = StageOf(state);
    if (stage == IrShaderStage::Vertex) {
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
            return "aps5_sample_id";
        case StageInputKind::PrimitiveId:
            state.AddParameter("uint aps5_primitive_id [[primitive_id]]");
            return "aps5_primitive_id";
        case StageInputKind::Layer:
            state.AddParameter("uint aps5_layer_in [[render_target_array_index]]");
            return "aps5_layer_in";
        case StageInputKind::HelperInvocation:
            return "(simd_is_helper_thread() ? 1u : 0u)";
        case StageInputKind::BaryCoordSmooth:
            if (component > 1u) state.Fail(instruction, "barycentric component is out of range");
            state.AddParameter("float3 aps5_bary [[barycentric_coord, center_perspective]]");
            return "as_type<uint>(aps5_bary[" + std::to_string(component + 1u) + "])";
        case StageInputKind::BaryCoordNoPerspective:
            if (component > 1u) state.Fail(instruction, "barycentric component is out of range");
            state.AddParameter("float3 aps5_bary_linear [[barycentric_coord, center_no_perspective]]");
            return "as_type<uint>(aps5_bary_linear[" + std::to_string(component + 1u) + "])";
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
    for (const auto& input : state.program.Info().inputs) {
        if (input.kind == StageInputKind::Parameter && input.location == attribute && input.perVertex) state.Fail(instruction, "per-vertex (barycentric emulation) pixel inputs are not implemented in the MSL backend (work package B6)");
    }
    const auto location = pixel.InputSlot(attribute);
    if (location >= 64u) state.Fail(instruction, "pixel input location " + std::to_string(location) + " is beyond 63");
    const bool flat = (pixel.interpolatorSettings[attribute] & PsInputFlatShade) != 0u && !pixel.InputIsCustom(attribute);
    const auto& metadata = state.program.Metadata();
    const bool linear = pixel.InputIsLinear(attribute, metadata.pixelLinearInputs, metadata.pixelPerspectiveInputs);
    state.fragmentInputs.emplace(location, flat ? ", flat" : linear ? ", center_no_perspective" : ", center_perspective");
    return "aps5_input_" + std::to_string(location) + "(" + std::to_string(channel & 3u) + ")";
}

void GetAttribute(MslEmitterState& state, const IrValue& instruction) {
    const auto attribute = instruction.Argument(0)->Resolve()->ImmediateU32();
    const auto channel = instruction.Argument(1)->Resolve()->ImmediateU32();
    const auto stage = StageOf(state);
    if (stage == IrShaderStage::Pixel) {
        state.Define(instruction, PixelAttribute(state, instruction, attribute, channel));
        return;
    }
    if (stage != IrShaderStage::Vertex) state.Fail(instruction, "attributes of this stage are not implemented in the MSL backend yet (work package B6)");
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
    const auto exported = exp.en & (mode == 1u ? 0x1u : mode == 2u ? 0x3u : mode == 3u ? 0x9u : 0xfu);
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
            if (((exported >> component) & 1u) != 0u) state.Line(name + "[" + std::to_string(component) + "] = " + data + "[" + std::to_string(component) + "];");
        }
    }
    return name;
}

void AuxPositionExport(MslEmitterState& state, const std::string& data, const ExportInfo& exp) {
    const auto& vertex = VertexInfo(state);
    for (std::uint32_t component = 0; component < 4u; ++component) {
        if ((exp.en & (1u << component)) == 0u) continue;
        const auto output = DecodePositionExportComponent(vertex.paClVsOutCntl, exp.index, component);
        const auto raw = data + "[" + std::to_string(component) + "]";
        if (output.layer) state.Line("aps5_out.aps5_layer = " + raw + " & 0x7ffu;");
        if (output.viewport) state.Line("aps5_out.aps5_viewport = extract_bits(" + raw + ", 16u, 4u);");
        if (output.layer || output.viewport) continue;
        if (output.pointSize) {
            state.Line("aps5_out.aps5_point_size = as_type<float>(" + raw + ");");
            continue;
        }
        if (output.clipDistance != std::numeric_limits<std::uint32_t>::max()) state.Line("aps5_out.aps5_clip[" + std::to_string(output.clipDistance) + "] = as_type<float>(" + raw + ");");
        if (output.cullDistance != std::numeric_limits<std::uint32_t>::max()) state.Fail("cull distances have no Metal vertex output (work package B6)");
    }
}

// SelectExportPacking: 10_11_11 targets take the colour packed into the red channel.
std::string ExportPacking(MslEmitterState& state, std::uint32_t target, const std::string& color) {
    state.RequirePrelude(MslPreludeChunk::StageIo);
    const auto packing = state.FunctionConstant(PipelineSpecialization::ExportPackingBase + target, static_cast<std::uint32_t>(ColorExportPacking::None));
    const auto name = state.Temporary("aps5_pk");
    state.Line("const uint4 " + name + " = " + packing + " == " + U32(static_cast<std::uint32_t>(ColorExportPacking::Unorm10_11_11)) + " ? uint4(aps5_pack_unorm_10_11_11(" + color + "), 0u, 0u, 0u) : " + color + ";");
    return name;
}

void SetAttribute(MslEmitterState& state, const IrValue& instruction) {
    const auto stage = StageOf(state);
    if (stage != IrShaderStage::Vertex && stage != IrShaderStage::Pixel) state.Fail(instruction, "exports of this stage are not implemented in the MSL backend yet (work package B6)");
    const auto& exports = state.program.Metadata().exportInfo;
    const auto index = instruction.Flags<ExportFlags>().index;
    if (index >= exports.size()) state.Fail(instruction, "export metadata index is out of range");
    const auto& exp = exports[index];
    const auto exec = state.Arg(instruction, 1u);
    if (stage == IrShaderStage::Pixel && exp.vm && state.requirements.pixelValidMask) {
        state.usesValidMask = true;
        state.Line("aps5_valid = " + exec + " ? 1u : 0u;");
    }
    if (exp.kind == ExportTargetKind::Null || exp.en == 0u || exp.kind == ExportTargetKind::Primitive) return;
    state.Line("if (" + exec + ") {");
    state.Indent();
    const auto data = state.Temporary("aps5_ed");
    state.Line("const uint4 " + data + " = uint4(" + state.Arg(instruction, 0u) + ");");
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
        state.Line("aps5_out.aps5_position = " + position + ";");
        // The all-zero position cull distance (StoreZeroPositionClipDistance).
        state.Line("aps5_out.aps5_clip[" + std::to_string(ClipDistanceCount(state)) + "] = ((as_type<uint4>(aps5_out.aps5_position).x | as_type<uint4>(aps5_out.aps5_position).y | as_type<uint4>(aps5_out.aps5_position).z | as_type<uint4>(aps5_out.aps5_position).w) & 0x7fffffffu) == 0u ? -1.0f : 0.0f;");
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
                    state.Line("aps5_out.aps5_sample_mask = !(as_type<float>(" + value + ".w) >= 0.5f) ? 0u : 0xffffffffu;");
                }
                if (!uintOutput && mode != 5u && mode != 6u) value = ExportPacking(state, exp.index, value);
                state.Line("aps5_out.c" + std::to_string(output->location) + " = as_type<APS5_COLOR" + std::to_string(output->location) + "_TYPE>(" + value + ");");
            } else {
                state.Line("aps5_out.p" + std::to_string(output->location) + " = as_type<float4>(" + value + ");");
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

// SelectExportPacking's 10_11_11 UNORM packing of an RGB colour (float bits).
static uint aps5_pack_unorm_10_11_11(uint4 color) {
    const Aps5Format info = aps5_format(APS5_FORMAT_11_11_10_UNORM);
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
        if (StageOf(state) == IrShaderStage::Compute) state.Define(instruction, ComputeBuiltin(state, kind, component));
        else state.Define(instruction, GraphicsBuiltin(state, instruction, kind, component));
        return true;
    }
    case IrOpcode::GetAttribute:
        GetAttribute(state, instruction);
        return true;
    case IrOpcode::SetAttribute:
        SetAttribute(state, instruction);
        return true;
    default:
        return false;
    }
}

MslGraphicsInterface MslGraphicsInterfaceOf(MslEmitterState& state) {
    MslGraphicsInterface result;
    const auto stage = StageOf(state);
    std::string& text = result.declarations;
    if (stage == IrShaderStage::Vertex) {
        const auto clips = ClipDistanceCount(state) + 1u;
        if (clips > MaxClipDistances) state.Fail("guest clip distances plus the all-zero clip position distance exceed 8 clip distances");
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
            case StageOutputKind::Layer: text += "    uint aps5_layer [[render_target_array_index]];\n"; break;
            case StageOutputKind::ViewportIndex: text += "    uint aps5_viewport [[viewport_array_index]];\n"; break;
            case StageOutputKind::CullDistance: state.Fail("cull distances have no Metal vertex output (work package B6)");
            default: break;
            }
        }
        // Metal's compiler fails past 124 scalar varyings (M2_DESIGN §4).
        if (scalars > 124u) state.Fail("the vertex function exceeds 124 scalar varyings");
        text += "    float aps5_clip [[clip_distance]] [" + std::to_string(clips) + "];\n};\n\n";
        result.outputType = "Aps5VertexOut";
        // Metal has no depth clip control: a [-w, w] guest depth range is remapped to [0, w]. The y
        // flip between Vulkan and Metal lives in the driver's viewport (MetalDraw.mm); APS5_FLIP_Y is
        // a debug aid the driver never defines.
        result.epilogue = "#if defined(APS5_DEPTH_NEGATIVE_ONE_TO_ONE) && APS5_DEPTH_NEGATIVE_ONE_TO_ONE\n"
                          "    aps5_out.aps5_position.z = (aps5_out.aps5_position.z + aps5_out.aps5_position.w) * 0.5f;\n"
                          "#endif\n"
                          "#if defined(APS5_FLIP_Y) && APS5_FLIP_Y\n"
                          "    aps5_out.aps5_position.y = -aps5_out.aps5_position.y;\n"
                          "#endif\n";
        return result;
    }
    if (stage != IrShaderStage::Pixel) state.Fail("only vertex and pixel graphics stages are implemented in the MSL backend so far (work package B6)");
    const auto& pixel = PixelInfo(state);
    text += "#ifndef APS5_VERTEX_OUTPUTS\n#define APS5_VERTEX_OUTPUTS 0xffffffff\n#endif\n#ifndef APS5_VERTEX_OUTPUTS_HI\n#define APS5_VERTEX_OUTPUTS_HI 0xffffffff\n#endif\n";
    text += "struct Aps5FragmentIn {\n    float4 aps5_position [[position]];\n";
    std::string accessors;
    for (const auto& [location, qualifier] : state.fragmentInputs) {
        const auto macro = location < 32u ? std::string("APS5_VERTEX_OUTPUTS") : std::string("APS5_VERTEX_OUTPUTS_HI");
        const auto guard = "#if ((" + macro + " >> " + std::to_string(location % 32u) + ") & 1)\n";
        text += guard + "    float4 p" + std::to_string(location) + " [[user(locn" + std::to_string(location) + ")" + qualifier + "]];\n#endif\n";
        accessors += guard + "#define aps5_input_" + std::to_string(location) + "(c) as_type<uint>(aps5_in.p" + std::to_string(location) + "[c])\n#else\n#define aps5_input_" + std::to_string(location) + "(c) 0u\n#endif\n";
    }
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
    if (pixel.psAlphaToCoverage && !state.reflection.exportsSampleMask) {
        members += "    uint aps5_sample_mask [[sample_mask]];\n";
        state.reflection.exportsSampleMask = true;
    }
    if (!members.empty()) {
        text += "struct Aps5FragmentOut {\n" + members + "};\n\n";
        result.outputType = "Aps5FragmentOut";
    }
    if (pixel.psSampleShading) state.reflection.perSample = true;
    // Primitive-ordered pixel shaders (the SPIR-V path's PixelInterlockOrderedEXT around the whole
    // program): the stage function brackets the guest body with raster-order-group accesses
    // (EmitMslEntryPoint, BufferIndex::RasterOrder).
    if (pixel.psOrderedPixelShader) state.reflection.rasterOrder = true;
    if (pixel.psEarlyZ && !pixel.psPixelKillEnable && !pixel.psDepthExportEnable && !pixel.psSampleMaskExportEnable && !pixel.psStencilRefExportEnable && !pixel.psAlphaToCoverage) {
        result.attributes = "[[early_fragment_tests]] ";
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
    static const std::string text = "#define APS5_FORMAT_11_11_10_UNORM " + std::to_string(static_cast<std::uint32_t>(IrBufferFormat::Format11_11_10UNorm)) + "u\n" + std::string(StageIoText);
    return {text, OnBufferFormat};
}

}
