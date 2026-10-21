#include "MetalBackend/MslEmitterState.hpp"
#include "MetalBackend/MslTessellator.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Tessellation on Metal (DECISIONS D24, MslAbi Version 7). MTL4RenderCommandEncoder has no patch
// draws, so the guest's local (LS), hull (HS) and domain (DS) programs run in mesh pipelines:
// - each program is emitted as a library without a stage function: `aps5_stage(...)` with the fixed
//   signature of its role runs the guest body (EmitMslTessellationStage below);
// - LinkMslTessellation puts the local and hull libraries in namespaces aps5_ls / aps5_hs under an
//   object function (one threadgroup per patch and instance: the local program per input control
//   point into threadgroup memory, a barrier, the hull program per output control point into the
//   object payload, then thread 0 runs the tessellator's factor processing and launches one mesh
//   threadgroup per slice of the patch's primitives), and the domain library in aps5_ds under a mesh
//   function (the slice's primitives from the software tessellator, the domain program once per
//   primitive corner);
// - the tessellator is MslTessellatorCore.inc, a port of the D3D11 reference tessellator, checked
//   bit for bit against it on the CPU (agc_metal_tessellator).
// The tessellation rings and LDS accesses arrive lowered (TessellationLowering): Get/Set
// TessellationAttribute with byte addresses inside one vertex's LDS slot (local outputs), the patch's
// LDS (hull inputs), one control point (hull control outputs), the patch-constant record (hull
// patch outputs, domain patch inputs; offsets modulo the record stride), the patch's control points
// (domain inputs) or the factor record (factors).

namespace ShaderRecompiler {

namespace {

constexpr std::uint32_t HeaderWords = MslAbi::TessellationPayloadHeaderWords;

std::string U32(std::uint32_t value) {
    return std::to_string(value) + "u";
}

IrShaderStage StageOf(const MslEmitterState& state) {
    return state.program.Resources().stage;
}

const ShaderTessellationInputInfo& Tessellation(const MslEmitterState& state) {
    if (state.inputInfo.vertex == nullptr) state.Fail("tessellation programs need the vertex input info");
    return state.inputInfo.vertex->tess;
}

std::uint32_t LocalWords(const MslEmitterState& state) {
    return state.program.Metadata().tessellationLocalStride / 4u;
}

std::uint32_t ControlWords(const MslEmitterState& state) {
    return state.program.Metadata().tessellationControlStride / 4u;
}

// Patch-constant record dwords (upstream #4163, SpirvModuleEmitter DefineTessellationInterfaces):
// TessellationLowering places every hull patch output and domain patch input at its off-chip address
// modulo the record stride (ProgramMetadata::tessellationPatchStride), so the hull and the domain
// program, compiled apart, agree on each constant's slot.
std::uint32_t PatchWords(const MslEmitterState& state) {
    return (state.program.Metadata().tessellationPatchStride + 3u) / 4u;
}

std::uint32_t ControlPointWords(const MslEmitterState& state) {
    return Tessellation(state).outputControlPoints * ControlWords(state);
}

// Hull outputs in payload dwords after the header: every control point, then the patch-constant
// record.
std::uint32_t PayloadWords(const MslEmitterState& state) {
    return ControlPointWords(state) + PatchWords(state);
}

std::string Index(MslEmitterState& state, const IrValue& instruction) {
    return "(uint(" + state.Arg(instruction, 1u) + ") >> 2u)";
}

void Get(MslEmitterState& state, const IrValue& instruction) {
    const auto kind = static_cast<TessellationAttribute>(instruction.Argument(0)->Resolve()->ImmediateU32());
    const auto predicate = state.Arg(instruction, 2u);
    const auto index = state.Temporary("aps5_ti");
    if (kind == TessellationAttribute::ControlInput) {
        if (StageOf(state) != IrShaderStage::TessellationControl) state.Fail(instruction, "control inputs outside the hull program");
        state.AddParameter("threadgroup uint* aps5_tess_local");
        // Upstream #4727 (SpirvModuleEmitter TessellationPointer): the control point of a hull input
        // address, which may be computed at run time, is clamped to the patch's last input control
        // point; the offset within the control point stays.
        const auto stride = state.program.Metadata().tessellationLocalStride;
        if (stride == 0u || stride % 4u != 0u) state.Fail(instruction, "hull input reads need a whole-dword local stride");
        const auto address = state.Temporary("aps5_ta");
        state.Line("const uint " + address + " = uint(" + state.Arg(instruction, 1u) + ");");
        state.Line("const uint " + index + " = min(" + address + " / " + U32(stride) + ", " + U32(Tessellation(state).inputControlPoints - 1u) + ") * " + U32(stride / 4u) + " + (" + address + " % " + U32(stride) + ") / 4u;");
        state.Define(instruction, "(" + predicate + ") ? aps5_tess_local[" + index + "] : 0u");
        return;
    }
    if (kind == TessellationAttribute::EvaluationInput) {
        if (StageOf(state) != IrShaderStage::TessellationEvaluation) state.Fail(instruction, "evaluation inputs outside the domain program");
        state.AddParameter("const object_data uint* aps5_tess_payload");
        // Upstream f95471ec6a (SpirvModuleEmitter TessellationPointer): the control point of a
        // domain input address, which may be computed from data, is clamped to the patch's last
        // output control point; the offset within the control point stays.
        const auto stride = state.program.Metadata().tessellationControlStride;
        if (stride != 0u && stride % 4u == 0u) {
            const auto address = state.Temporary("aps5_ta");
            state.Line("const uint " + address + " = uint(" + state.Arg(instruction, 1u) + ");");
            state.Line("const uint " + index + " = min(" + address + " / " + U32(stride) + ", " + U32(Tessellation(state).outputControlPoints - 1u) + ") * " + U32(stride / 4u) + " + (" + address + " % " + U32(stride) + ") / 4u;");
        } else {
            state.Line("const uint " + index + " = " + Index(state, instruction) + ";");
        }
        state.Define(instruction, "(" + predicate + ") && " + index + " < " + U32(ControlPointWords(state)) + " ? aps5_tess_payload[" + U32(HeaderWords) + " + " + index + "] : 0u");
        return;
    }
    state.Line("const uint " + index + " = " + Index(state, instruction) + ";");
    if (kind == TessellationAttribute::InstanceInput) {
        // Upstream 30e85a3a2: the draw-relative instance of the patch's control points (the local
        // program's InstanceIndex - BaseInstance on the SPIR-V path), so a hull numbers the patches
        // of instances that share hull threadgroups across the draw. The object function runs one
        // threadgroup per patch and instance and passes its instance (LinkMslTessellation).
        if (StageOf(state) != IrShaderStage::TessellationControl) state.Fail(instruction, "instance inputs outside the hull program");
        state.AddParameter("uint aps5_hull_instance");
        state.Define(instruction, "(" + predicate + ") ? aps5_hull_instance : 0u");
        return;
    }
    if (kind == TessellationAttribute::PatchInput) {
        // Upstream #4163: the domain program reads the hull's patch constants from the record after
        // the control points.
        if (StageOf(state) != IrShaderStage::TessellationEvaluation) state.Fail(instruction, "patch inputs outside the domain program");
        state.AddParameter("const object_data uint* aps5_tess_payload");
        state.Define(instruction, "(" + predicate + ") && " + index + " < " + U32(PatchWords(state)) + " ? aps5_tess_payload[" + U32(HeaderWords + ControlPointWords(state)) + " + " + index + "] : 0u");
        return;
    }
    state.Fail(instruction, "unsupported tessellation attribute read");
}

void Set(MslEmitterState& state, const IrValue& instruction) {
    const auto kind = static_cast<TessellationAttribute>(instruction.Argument(0)->Resolve()->ImmediateU32());
    const auto value = "uint(" + state.Arg(instruction, 2u) + ")";
    state.Line("if (" + state.Arg(instruction, 3u) + ") {");
    state.Indent();
    const auto index = state.Temporary("aps5_ti");
    state.Line("const uint " + index + " = " + Index(state, instruction) + ";");
    switch (kind) {
    case TessellationAttribute::LocalOutput:
        if (StageOf(state) != IrShaderStage::Local) state.Fail(instruction, "local outputs outside the local program");
        state.AddParameter("threadgroup uint* aps5_tess_local");
        state.Line("if (" + index + " < " + U32(LocalWords(state)) + ") aps5_tess_local[" + index + "] = " + value + ";");
        break;
    case TessellationAttribute::ControlOutput:
        if (StageOf(state) != IrShaderStage::TessellationControl) state.Fail(instruction, "control outputs outside the hull program");
        state.AddParameter("object_data uint* aps5_tess_payload");
        state.AddParameter("uint aps5_invocation_id");
        // Hull invocations past the output control points (object threadgroups as wide as the input
        // patch) run the program for its barriers only.
        state.Line("if (aps5_invocation_id < " + U32(Tessellation(state).outputControlPoints) + " && " + index + " < " + U32(ControlWords(state)) + ") aps5_tess_payload[" + U32(HeaderWords) + " + aps5_invocation_id * " + U32(ControlWords(state)) + " + " + index + "] = " + value + ";");
        break;
    case TessellationAttribute::InstanceOutput:
        // Upstream 30e85a3a2: the local program's draw-relative instance for the hull. The object
        // function already knows each patch's instance and hands it to the hull (InstanceInput).
        if (StageOf(state) != IrShaderStage::Local) state.Fail(instruction, "instance outputs outside the local program");
        state.Line("(void)" + value + ";");
        break;
    case TessellationAttribute::PatchOutput:
        if (StageOf(state) != IrShaderStage::TessellationControl) state.Fail(instruction, "patch outputs outside the hull program");
        state.AddParameter("object_data uint* aps5_tess_payload");
        state.Line("if (" + index + " < " + U32(PatchWords(state)) + ") aps5_tess_payload[" + U32(HeaderWords + ControlPointWords(state)) + " + " + index + "] = " + value + ";");
        break;
    case TessellationAttribute::Factor: {
        if (StageOf(state) != IrShaderStage::TessellationControl) state.Fail(instruction, "tessellation factors outside the hull program");
        const auto* address = instruction.Argument(1)->Resolve();
        if (!address->HasImmediate()) state.Fail(instruction, "tessellation factor index must be immediate");
        const auto factor = address->ImmediateU32() / 4u;
        const auto domain = Tessellation(state).domain;
        const auto count = domain == 0u ? 2u : domain == 1u ? 4u : 6u;
        if (factor >= count) state.Fail(instruction, "tessellation factor index out of range");
        // Payload factors in the tessellator's order: outer 0-3, inner 0-1. The triangle's inner factor
        // follows its 3 outer ones in the ring; the isoline ring record is (detail, density), the
        // reverse of the tessellator's (density, detail) (radeonsi and r600 swap GL's outer[0] and
        // outer[1] for isolines; DECISIONS D26).
        const auto slot = domain == 1u && factor == 3u ? 4u : domain == 0u ? factor ^ 1u : factor;
        state.AddParameter("object_data uint* aps5_tess_payload");
        state.Line("aps5_tess_payload[" + U32(slot) + "] = " + value + ";");
        break;
    }
    default:
        state.Fail(instruction, "unsupported tessellation attribute write");
    }
    state.Outdent();
    state.Line("}");
}

struct Role {
    const char* name;
    std::vector<std::string> signature;
};

Role RoleOf(const MslEmitterState& state) {
    switch (StageOf(state)) {
    case IrShaderStage::Local:
        return {"local", {"device const Aps5Table& aps5_table", "constant uint* aps5_push", "constant Aps5DriverData& aps5_driver", "device const Aps5VertexFetch* aps5_fetch", "uint aps5_vertex_id", "uint aps5_instance_id", "uint aps5_base_instance", "uint aps5_simd_lane", "threadgroup uint* aps5_tess_local"}};
    case IrShaderStage::TessellationControl:
        return {"hull", {"device const Aps5Table& aps5_table", "constant uint* aps5_push", "constant Aps5DriverData& aps5_driver", "uint aps5_invocation_id", "uint aps5_patch_id", "uint aps5_hull_instance", "uint aps5_simd_lane", "threadgroup uint* aps5_tess_local", "object_data uint* aps5_tess_payload"}};
    case IrShaderStage::TessellationEvaluation:
        return {"domain", {"device const Aps5Table& aps5_table", "constant uint* aps5_push", "constant Aps5DriverData& aps5_driver", "uint aps5_patch_id", "float3 aps5_tess_coord", "uint aps5_simd_lane", "const object_data uint* aps5_tess_payload", "thread Aps5VertexOut& aps5_out"}};
    default:
        state.Fail("not a tessellation stage");
    }
}

std::string NameOf(const std::string& declaration) {
    return declaration.substr(declaration.find_last_of(" &*") + 1u);
}

}

bool MslTessellationStage(IrShaderStage stage) {
    return stage == IrShaderStage::Local || stage == IrShaderStage::TessellationControl || stage == IrShaderStage::TessellationEvaluation;
}

bool EmitMslTessellationInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::GetTessellationAttribute:
        Get(state, instruction);
        return true;
    case IrOpcode::SetTessellationAttribute:
        Set(state, instruction);
        return true;
    default:
        return false;
    }
}

std::string MslTessellationBuiltin(MslEmitterState& state, const IrValue& instruction, StageInputKind kind, std::uint32_t component) {
    const auto stage = StageOf(state);
    if ((kind == StageInputKind::InvocationId || kind == StageInputKind::PrimitiveId) && stage == IrShaderStage::TessellationControl) {
        state.AddParameter(kind == StageInputKind::InvocationId ? "uint aps5_invocation_id" : "uint aps5_patch_id");
        return kind == StageInputKind::InvocationId ? "aps5_invocation_id" : "aps5_patch_id";
    }
    if (kind == StageInputKind::BaseInstance && stage == IrShaderStage::Local) {
        // Upstream 30e85a3a2 (TessellationLowering writeInstance): the draw's first instance.
        state.AddParameter("uint aps5_base_instance");
        return "aps5_base_instance";
    }
    if (kind == StageInputKind::PrimitiveId && stage == IrShaderStage::TessellationEvaluation) {
        state.AddParameter("uint aps5_patch_id");
        return "aps5_patch_id";
    }
    if (kind == StageInputKind::TessCoord && stage == IrShaderStage::TessellationEvaluation) {
        if (component > 2u) state.Fail(instruction, "TessCoord component is out of range");
        state.AddParameter("float3 aps5_tess_coord");
        return "as_type<uint>(aps5_tess_coord[" + std::to_string(component) + "])";
    }
    return {};
}

std::string MslTessellationInvocation(MslEmitterState& state) {
    state.AddParameter("uint aps5_invocation_id");
    return "aps5_invocation_id";
}

std::string EmitMslTessellationStage(MslEmitterState& state, std::vector<std::string> parameters, bool hasTable) {
    const auto role = RoleOf(state);
    const auto stage = StageOf(state);
    std::string text;
    // The fixed signature names these types even when the program does not use them.
    if (!hasTable) text += "struct Aps5Table {\n    uint aps5_none;\n};\n\n";
    std::string epilogue;
    std::vector<std::string> helperParameters;
    std::vector<std::string> arguments;
    for (const auto& parameter : parameters) {
        const auto declaration = parameter.substr(0, parameter.find(" [["));
        helperParameters.push_back(declaration);
        arguments.push_back(NameOf(declaration));
    }
    if (stage == IrShaderStage::TessellationEvaluation) {
        const auto graphics = MslGraphicsInterfaceOf(state);
        text += graphics.declarations;
        epilogue = graphics.epilogue;
        helperParameters.push_back("thread Aps5VertexOut& aps5_out");
        arguments.push_back("aps5_out");
    }
    const auto& tess = Tessellation(state);
    text += "constexpr constant uint aps5_tess_inputs = " + U32(tess.inputControlPoints) + ";\n";
    text += "constexpr constant uint aps5_tess_outputs = " + U32(tess.outputControlPoints) + ";\n";
    text += "constexpr constant uint aps5_tess_local_words = " + U32(LocalWords(state)) + ";\n";
    text += "constexpr constant uint aps5_tess_control_words = " + U32(ControlWords(state)) + ";\n";
    text += "constexpr constant uint aps5_tess_payload_words = " + U32(stage == IrShaderStage::Local ? 0u : PayloadWords(state)) + ";\n\n";
    text += "static void aps5_body(";
    for (std::size_t index = 0; index < helperParameters.size(); ++index) text += (index == 0u ? "\n    " : ",\n    ") + helperParameters[index];
    text += ")\n{\n" + MslFunctionStatements(state, state.prologue) + "}\n\n";
    std::vector<std::string> names;
    for (const auto& declaration : role.signature) names.push_back(NameOf(declaration));
    for (std::size_t index = 0; index < helperParameters.size(); ++index) {
        if (std::find(names.begin(), names.end(), arguments[index]) == names.end()) state.Fail("the " + std::string(role.name) + " program needs " + helperParameters[index] + ", which tessellation stages do not provide");
        // The signature's declaration must be the body's (a type mismatch would not compile anyway).
    }
    // Every tessellation library carries the types its signature names.
    text += "// Tessellation " + std::string(role.name) + " program (MslAbi Version 7, LinkMslTessellation).\n";
    text += "static void aps5_stage(";
    for (std::size_t index = 0; index < role.signature.size(); ++index) text += (index == 0u ? "\n    " : ",\n    ") + role.signature[index];
    text += ")\n{\n    aps5_body(";
    for (std::size_t index = 0; index < arguments.size(); ++index) text += (index == 0u ? "" : ", ") + arguments[index];
    text += ");\n" + epilogue + "}\n";
    return text;
}

void PrepareMslTessellationStage(MslEmitterState& state) {
    // aps5_stage names Aps5DriverData (Common) and, for the local program, Aps5VertexFetch (StageIo).
    state.RequirePrelude(MslPreludeChunk::Common);
    if (StageOf(state) == IrShaderStage::Local) state.RequirePrelude(MslPreludeChunk::StageIo);
    // The fixed signature's builtins: the SIMD lane is a parameter, not a stage attribute.
    for (std::size_t index = 0; index < state.builtins.size(); ++index) {
        if (state.builtins.test(index) && static_cast<MslBuiltin>(index) != MslBuiltin::ThreadIndexInSimdgroup) state.Fail("threadgroup builtins are not available to tessellation programs");
    }
}

namespace {

// The library text without its header, function constants replaced by their pipeline values (the
// three programs share an object or mesh library, and their PushDataOffset constants differ).
std::string Namespaced(const MslTessellationProgram& program, const char* name) {
    if (program.shader == nullptr) throw std::runtime_error("MSL tessellation link: the " + std::string(name) + " program has no MSL");
    const auto& source = program.shader->source;
    static constexpr std::string_view Header = "#include <metal_stdlib>\nusing namespace metal;\n";
    std::string body = source.starts_with(Header) ? source.substr(Header.size()) : source;
    std::string out;
    std::size_t start = 0;
    while (start < body.size()) {
        auto end = body.find('\n', start);
        if (end == std::string::npos) end = body.size();
        const auto line = body.substr(start, end - start);
        start = end + 1u;
        static constexpr std::string_view Prefix = "constant uint aps5_fc";
        if (line.starts_with(Prefix)) {
            const auto idEnd = line.find_first_not_of("0123456789", Prefix.size());
            const auto id = static_cast<std::uint32_t>(std::stoul(line.substr(Prefix.size(), idEnd - Prefix.size())));
            if (line.compare(idEnd, 6, "_value") == 0) continue;
            const auto colon = line.rfind(" : ");
            std::string value = line.substr(colon + 3u, line.size() - colon - 4u);
            for (const auto& [constant, supplied] : program.constants) {
                if (constant == id) value = std::to_string(supplied) + "u";
            }
            out += "constant uint aps5_fc" + std::to_string(id) + " = " + value + ";\n";
            continue;
        }
        out += line + "\n";
    }
    return "namespace " + std::string(name) + " {\n" + out + "}\n\n";
}

std::uint32_t Declared(const MslTessellationProgram& program, std::string_view constant) {
    const auto key = "constexpr constant uint " + std::string(constant) + " = ";
    const auto& source = program.shader->source;
    const auto at = source.find(key);
    if (at == std::string::npos) throw std::runtime_error("MSL tessellation link: a tessellation library lacks " + std::string(constant));
    return static_cast<std::uint32_t>(std::stoul(source.substr(at + key.size())));
}

void Expect(const MslTessellationProgram& program, MslAbi::Stage stage, const char* name) {
    if (program.shader == nullptr || program.shader->reflection.stage != stage) throw std::runtime_error("MSL tessellation link: the " + std::string(name) + " program is not a " + name + " tessellation library");
}

}

MslTessellationPipeline LinkMslTessellation(const MslTessellationProgram& local, const MslTessellationProgram& hull, const MslTessellationProgram& domain, const TessellationConfiguration& config) {
    Expect(local, MslAbi::Stage::Local, "local");
    Expect(hull, MslAbi::Stage::Hull, "hull");
    Expect(domain, MslAbi::Stage::Domain, "domain");
    namespace T = MslTessellator;
    if (config.domain > 2u || config.partitioning > 3u || config.outputTopology > 3u) throw std::runtime_error("MSL tessellation link: invalid tessellation domain, partitioning or output topology");
    if (config.domain == T::Isoline && config.outputTopology >= T::TriangleCw) throw std::runtime_error("MSL tessellation link: isolines cannot output triangles");
    if (config.inputControlPoints == 0u || config.inputControlPoints > 32u || config.outputControlPoints == 0u || config.outputControlPoints > 32u) throw std::runtime_error("MSL tessellation link: invalid control-point counts");
    const auto topology = static_cast<int>(config.outputTopology);
    const auto slice = static_cast<std::uint32_t>(T::SlicePrimitives(topology));
    const auto corners = topology == T::Point ? 1u : topology == T::Line ? 2u : 3u;
    const auto localWords = std::max(Declared(local, "aps5_tess_local_words"), Declared(hull, "aps5_tess_local_words"));
    const auto payloadWords = HeaderWords + std::max(Declared(hull, "aps5_tess_payload_words"), Declared(domain, "aps5_tess_payload_words"));
    if (payloadWords * 4u > MslAbi::TessellationPayloadBytes) throw std::runtime_error("MSL tessellation link: the hull outputs (" + std::to_string(payloadWords * 4u) + " bytes) exceed Metal's 16 KiB object payload (the compute pre-pass fallback is not implemented, TechnicalDebt metal-tessellation)");
    const auto threads = std::max(config.inputControlPoints, config.outputControlPoints);
    const auto configuration = std::to_string(config.domain) + ", " + std::to_string(config.partitioning) + ", " + std::to_string(config.outputTopology);
    const std::string common = "#include <metal_stdlib>\nusing namespace metal;\n\n" + T::Source() + "\nstruct Aps5TessPayload {\n    uint words[" + U32(payloadWords) + "];\n};\n\n";
    const auto factors = "    Aps5TessFactors aps5_factors;\n"
                         "    for (uint i = 0u; i < 4u; ++i) aps5_factors.outer[i] = as_type<float>(aps5_payload.words[i]);\n"
                         "    for (uint i = 0u; i < 2u; ++i) aps5_factors.inner[i] = as_type<float>(aps5_payload.words[4u + i]);\n"
                         "    Aps5TessPatch aps5_patch = aps5_tess_process(" + configuration + ", aps5_factors);\n";
    // VGT_HOS_MIN/MAX_TESS_LEVEL (the TessellationDraw record) clamp the factors as upstream's hull
    // programs do (#4082, SpirvModuleEmitter EmitSetTessellationAttribute; MslAbi Version 10): an
    // outer factor above 0 to [min, max], an inner factor above 1 to [max(min, 1), max]; an outer
    // factor at or below 0 or NaN stays (it culls the patch), an inner one becomes max(min, 1).
    // Upstream caps at 63 for fractional_odd and 64 otherwise (f8e8aca5c, d7818c9e5 also accept
    // integer and fractional_even); the tessellator clamps every partitioning to that same range
    // after this (DECISIONS D26). The clamped factors replace the raw
    // ones in the payload, so the mesh threadgroups tessellate exactly what the object stage counted.
    const auto clamp = "        for (uint i = 0u; i < " + U32(MslAbi::TessellationPayloadFactorWords) + "; ++i) {\n"
                       "            const float aps5_raw = as_type<float>(aps5_payload.words[i]);\n"
                       "            const float aps5_lowest = i < 4u ? aps5_draw.minLevel : fmax(aps5_draw.minLevel, 1.0f);\n"
                       "            const bool aps5_inside = aps5_raw > (i < 4u ? 0.0f : 1.0f);\n"
                       "            const float aps5_level = aps5_inside ? fmin(fmax(aps5_raw, aps5_lowest), aps5_draw.maxLevel) : (i < 4u ? aps5_raw : aps5_lowest);\n"
                       "            aps5_payload.words[i] = as_type<uint>(aps5_level);\n"
                       "        }\n";

    MslTessellationPipeline result;
    result.objectThreads = threads;
    result.meshThreads = 256u;
    result.payloadBytes = payloadWords * 4u;
    result.maxMeshThreadgroups = 1024u;
    result.topology = config.outputTopology;

    std::string object = common + Namespaced(local, "aps5_ls") + Namespaced(hull, "aps5_hs");
    object += "struct Aps5TessDraw {\n    ulong indexAddress;\n    uint indexSize;\n    uint firstIndex;\n    int vertexOffset;\n    uint firstInstance;\n    float minLevel;\n    float maxLevel;\n};\n\n";
    object += "[[object]] void " + std::string(MslAbi::EntryPoint) + "(\n";
    object += "    object_data Aps5TessPayload& aps5_payload [[payload]],\n    mesh_grid_properties aps5_grid,\n";
    object += "    device const aps5_ls::Aps5Table& aps5_ls_table [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::DescriptorTable)) + ")]],\n";
    object += "    constant uint* aps5_push [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::PushData)) + ")]],\n";
    object += "    constant aps5_ls::Aps5DriverData& aps5_driver [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::DriverData)) + ")]],\n";
    object += "    device const aps5_ls::Aps5VertexFetch* aps5_fetch [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::VertexFetch)) + ")]],\n";
    object += "    device const aps5_hs::Aps5Table& aps5_hs_table [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::TessellationHullTable)) + ")]],\n";
    object += "    device const Aps5TessDraw& aps5_draw [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::TessellationDraw)) + ")]],\n";
    object += "    uint3 aps5_group [[threadgroup_position_in_grid]],\n    uint aps5_tid [[thread_index_in_threadgroup]],\n    uint aps5_lane [[thread_index_in_simdgroup]])\n{\n";
    object += "    threadgroup uint aps5_local[" + U32(std::max(config.inputControlPoints * localWords, 1u)) + "];\n";
    object += "    threadgroup int aps5_none[1];\n";
    object += "    const uint aps5_patch_id = aps5_group.x;\n";
    object += "    if (aps5_tid < " + U32(config.inputControlPoints) + ") {\n";
    object += "        const uint aps5_position = aps5_draw.firstIndex + aps5_patch_id * " + U32(config.inputControlPoints) + " + aps5_tid;\n";
    object += "        uint aps5_vertex = aps5_position;\n";
    object += "        if (aps5_draw.indexSize == 2u) aps5_vertex = uint(int(reinterpret_cast<device const ushort*>(aps5_draw.indexAddress)[aps5_position]) + aps5_draw.vertexOffset);\n";
    object += "        else if (aps5_draw.indexSize == 4u) aps5_vertex = uint(int(reinterpret_cast<device const uint*>(aps5_draw.indexAddress)[aps5_position]) + aps5_draw.vertexOffset);\n";
    object += "        aps5_ls::aps5_stage(aps5_ls_table, aps5_push, aps5_driver, aps5_fetch, aps5_vertex, aps5_draw.firstInstance + aps5_group.y, aps5_draw.firstInstance, aps5_lane, aps5_local + aps5_tid * " + U32(localWords) + ");\n";
    object += "    }\n";
    object += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    object += "    aps5_hs::aps5_stage(aps5_hs_table, aps5_push, *reinterpret_cast<constant aps5_hs::Aps5DriverData*>(&aps5_driver), aps5_tid, aps5_patch_id, aps5_group.y, aps5_lane, aps5_local, aps5_payload.words);\n";
    object += "    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_object_data);\n";
    object += "    if (aps5_tid == 0u) {\n" + std::string(clamp) + factors;
    object += "        const int aps5_count = aps5_tess_primitive_count(aps5_patch, aps5_none);\n";
    object += "        aps5_payload.words[" + U32(MslAbi::TessellationPayloadPatchWord) + "] = aps5_patch_id;\n";
    object += "        aps5_payload.words[" + U32(MslAbi::TessellationPayloadPrimitivesWord) + "] = uint(aps5_count);\n";
    // A culled patch (no primitives) leaves the mesh grid unset, which launches no mesh threadgroup
    // (MSL specification 4.1, section 5.1.8: mesh_grid_properties).
    object += "    }\n    threadgroup_barrier(mem_flags::mem_object_data);\n";
    object += "    const uint aps5_primitives = aps5_payload.words[" + U32(MslAbi::TessellationPayloadPrimitivesWord) + "];\n";
    object += "    if (aps5_tid == 0u && aps5_primitives != 0u) aps5_grid.set_threadgroups_per_grid(uint3((aps5_primitives + " + U32(slice - 1u) + ") / " + U32(slice) + ", 1u, 1u));\n}\n";

    const char* topologyName = topology == T::Point ? "point" : topology == T::Line ? "line" : "triangle";
    std::string mesh = common + (topology == T::Point ? "#define APS5_TESS_POINTS 1\n" : "") + Namespaced(domain, "aps5_ds");
    mesh += "struct Aps5TessPrimitive {\n    bool aps5_culled [[primitive_culled]];\n};\n\n";
    mesh += "using Aps5TessMesh = metal::mesh<aps5_ds::Aps5VertexOut, Aps5TessPrimitive, " + std::to_string(slice * corners) + ", " + std::to_string(slice) + ", metal::topology::" + topologyName + ">;\n\n";
    mesh += "[[mesh]] void " + std::string(MslAbi::EntryPoint) + "(\n    Aps5TessMesh aps5_mesh,\n";
    mesh += "    const object_data Aps5TessPayload& aps5_payload [[payload]],\n";
    mesh += "    device const aps5_ds::Aps5Table& aps5_ds_table [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::DescriptorTable)) + ")]],\n";
    mesh += "    constant uint* aps5_push [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::PushData)) + ")]],\n";
    mesh += "    constant aps5_ds::Aps5DriverData& aps5_driver [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::DriverData)) + ")]],\n";
    mesh += "    uint3 aps5_slice [[threadgroup_position_in_grid]],\n    uint aps5_tid [[thread_index_in_threadgroup]],\n    uint aps5_lane [[thread_index_in_simdgroup]])\n{\n";
    mesh += "    threadgroup int aps5_corners[" + U32(slice * corners) + "];\n    threadgroup uint aps5_count[1];\n";
    // Per corner: its owner corner (bits 0-15) and, for an owner, its vertex (bits 16-23). One 32-bit
    // word per corner: sub-word threadgroup arrays here had their stores land in aps5_corners on the
    // M3 (agc_metal_tessellation_domains, macOS 27.0.1).
    mesh += "    threadgroup uint aps5_sharing[" + U32(slice * corners) + "];\n    threadgroup atomic_uint aps5_vertices[1];\n";
    mesh += factors;
    mesh += "    if (aps5_tid == 0u) {\n";
    mesh += "        const int aps5_total = int(aps5_payload.words[" + U32(MslAbi::TessellationPayloadPrimitivesWord) + "]);\n";
    mesh += "        const int aps5_begin = int(aps5_slice.x) * " + std::to_string(slice) + ";\n";
    mesh += "        aps5_count[0] = uint(aps5_tess_window(aps5_patch, aps5_begin, min(" + std::to_string(slice) + ", aps5_total - aps5_begin), aps5_corners));\n";
    mesh += "        atomic_store_explicit(&aps5_vertices[0], 0u, memory_order_relaxed);\n";
    mesh += "        aps5_mesh.set_primitive_count(aps5_count[0]);\n    }\n";
    mesh += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    mesh += "    const uint aps5_primitives = aps5_count[0];\n";
    // Vertex sharing within the slice: the first corner that uses a domain point owns its vertex
    // (the domain program runs once per unique point); later corners index the owner's vertex.
    mesh += "    const uint aps5_used = aps5_primitives * " + U32(corners) + ";\n";
    mesh += "    const int aps5_corner = aps5_tid < aps5_used ? aps5_corners[aps5_tid] : 0;\n";
    mesh += "    if (aps5_tid < aps5_used) {\n";
    mesh += "        uint aps5_owner = aps5_tid;\n";
    mesh += "        for (uint s = 0u; s < aps5_tid; ++s) {\n            if (aps5_corners[s] == aps5_corner) { aps5_owner = s; break; }\n        }\n";
    mesh += "        aps5_sharing[aps5_tid] = aps5_owner == aps5_tid ? aps5_owner | (atomic_fetch_add_explicit(&aps5_vertices[0], 1u, memory_order_relaxed) << 16u) : aps5_owner;\n";
    mesh += "    }\n";
    mesh += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    mesh += "    const uint aps5_owned = aps5_tid < aps5_used ? aps5_sharing[aps5_tid] : 0u;\n";
    mesh += "    const uint aps5_index = aps5_tid < aps5_used ? aps5_sharing[aps5_owned & 0xffffu] >> 16u : 0u;\n";
    mesh += "    if (aps5_tid < aps5_used && (aps5_owned & 0xffffu) == aps5_tid) {\n";
    mesh += "        const Aps5TessPoint aps5_point = aps5_tess_point(aps5_patch, aps5_corner);\n";
    // Domain locations as the D3D11 reference: (u, v) and, for triangles, w = 1 - u - v.
    mesh += "        const float3 aps5_coord = float3(aps5_point.u, aps5_point.v, " + std::string(config.domain == T::Triangle ? "1.0f - aps5_point.u - aps5_point.v" : "0.0f") + ");\n";
    mesh += "        aps5_ds::Aps5VertexOut aps5_vertex = {};\n";
    mesh += "#ifdef APS5_TESS_POINT_SIZE_FIELD\n        aps5_vertex.aps5_tess_point_size = 1.0f;\n#endif\n";
    mesh += "        aps5_ds::aps5_stage(aps5_ds_table, aps5_push, aps5_driver, aps5_payload.words[" + U32(MslAbi::TessellationPayloadPatchWord) + "], aps5_coord, aps5_lane, aps5_payload.words, aps5_vertex);\n";
    mesh += "        aps5_mesh.set_vertex(aps5_index, aps5_vertex);\n    }\n";
    mesh += "    if (aps5_tid < aps5_used) aps5_mesh.set_index(aps5_tid, uchar(aps5_index));\n";
    mesh += "    if (aps5_tid < aps5_primitives) {\n        Aps5TessPrimitive aps5_primitive = {};\n        aps5_primitive.aps5_culled = false;\n        aps5_mesh.set_primitive(aps5_tid, aps5_primitive);\n    }\n}\n";

    result.object.source = std::move(object);
    result.object.reflection.stage = MslAbi::Stage::Object;
    result.object.reflection.threadsPerThreadgroup = {threads, 1u, 1u};
    result.object.reflection.usesPushData = true;
    result.object.reflection.usesDriverData = true;
    result.object.reflection.vertexFetchLocations = local.shader->reflection.vertexFetchLocations;
    result.mesh.source = std::move(mesh);
    result.mesh.reflection.stage = MslAbi::Stage::Mesh;
    result.mesh.reflection.threadsPerThreadgroup = {result.meshThreads, 1u, 1u};
    result.mesh.reflection.usesPushData = true;
    result.mesh.reflection.usesDriverData = true;
    result.mesh.reflection.outputLocations = domain.shader->reflection.outputLocations;
    return result;
}

}
