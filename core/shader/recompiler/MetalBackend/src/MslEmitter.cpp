#include "MetalBackend/MslEmitterState.hpp"
#include "BdaAbi.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ShaderRecompiler {

namespace {

const ShaderWorkgroupInputInfo* WorkgroupInput(const IrProgram& program, const ShaderStageInputInfo& inputInfo) {
    switch (program.Resources().stage) {
    case IrShaderStage::Compute:
        if (inputInfo.compute == nullptr) throw std::runtime_error("MSL emission failed: compute input info is missing");
        return inputInfo.compute;
    case IrShaderStage::Mesh:
        if (inputInfo.vertex == nullptr) throw std::runtime_error("MSL emission failed: vertex input info is missing");
        return &inputInfo.vertex->mesh;
    default:
        return nullptr;
    }
}

MslAbi::Stage ReflectedStage(IrShaderStage stage) {
    switch (stage) {
    case IrShaderStage::Compute:
        return MslAbi::Stage::Kernel;
    case IrShaderStage::Vertex:
    case IrShaderStage::Local:
    case IrShaderStage::TessellationEvaluation:
        return MslAbi::Stage::Vertex;
    case IrShaderStage::Pixel:
        return MslAbi::Stage::Fragment;
    case IrShaderStage::Mesh:
        return MslAbi::Stage::Mesh;
    default:
        throw std::runtime_error("MSL emission failed: unsupported shader stage " + std::to_string(static_cast<unsigned>(stage)));
    }
}

RuntimeAbi::Stage StageGroup(IrShaderStage stage) {
    switch (stage) {
    case IrShaderStage::Pixel:
        return RuntimeAbi::Stage::Fragment;
    case IrShaderStage::TessellationControl:
        return RuntimeAbi::Stage::TessellationControl;
    case IrShaderStage::TessellationEvaluation:
        return RuntimeAbi::Stage::TessellationEvaluation;
    default:
        return RuntimeAbi::Stage::Main;
    }
}

}

std::string MslHex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::string MslType(IrType type) {
    switch (type) {
    case IrType::Bool:
        return "bool";
    case IrType::U8:
        return "uchar";
    case IrType::U16:
        return "ushort";
    case IrType::U32:
        return "uint";
    case IrType::S32:
        return "int";
    case IrType::F32:
        return "float";
    case IrType::U64:
        return "ulong";
    case IrType::S64:
        return "long";
    case IrType::F64:
        return "ulong";
    case IrType::F16:
        return "half";
    case IrType::Vec2U32:
        return "uint2";
    case IrType::Vec3U32:
        return "uint3";
    case IrType::Vec4U32:
        return "uint4";
    case IrType::Vec2F32:
        return "float2";
    case IrType::Vec3F32:
        return "float3";
    case IrType::Vec4F32:
        return "float4";
    default:
        throw std::runtime_error("MSL emission failed: IR type " + TypeToString(type) + " has no MSL value type");
    }
}

std::string MslTableField(DescriptorBindingKind kind) {
    switch (kind) {
    case DescriptorBindingKind::Buffers:
        return "buffers";
    case DescriptorBindingKind::Samplers:
        return "samplers";
    case DescriptorBindingKind::Gds:
        return "gds";
    case DescriptorBindingKind::BdaPagetable:
        return "bda_pagetable";
    case DescriptorBindingKind::FaultBuffer:
        return "fault_buffer";
    case DescriptorBindingKind::FlattenedSrt:
        return "flattened_srt";
    case DescriptorBindingKind::ShaderData:
        return "shader_data";
    default:
        return "images" + std::to_string(static_cast<std::uint32_t>(kind));
    }
}

MslEmitterState::MslEmitterState(const IrProgram& program, const ShaderStageInputInfo& inputInfo, const CompiledBindingLayout& bindings, const MslTargetOptions& target)
    : program(program), inputInfo(inputInfo), bindings(bindings), target(target), requirements(AnalyzeProgramRequirements(program)) {
}

void MslEmitterState::Line(std::string_view text) {
    body.append(static_cast<std::size_t>(depth) * 4u, ' ');
    body.append(text);
    body.push_back('\n');
}

void MslEmitterState::Outdent() {
    if (depth == 0u) Fail("unbalanced MSL indentation");
    --depth;
}

void MslEmitterState::DeclareLocal(std::string_view type, std::string_view name, std::string_view initializer) {
    locals.append("    ");
    locals.append(type);
    locals.push_back(' ');
    locals.append(name);
    if (!initializer.empty()) {
        locals.append(" = ");
        locals.append(initializer);
    }
    locals.append(";\n");
}

void MslEmitterState::RequirePrelude(MslPreludeChunk chunk) {
    const auto index = static_cast<std::size_t>(chunk);
    if (prelude.test(index)) return;
    for (const auto dependency : MslPrelude(chunk).dependencies) RequirePrelude(dependency);
    prelude.set(index);
}

std::string MslEmitterState::Prelude() const {
    std::string text;
    for (std::size_t index = 0; index < prelude.size(); ++index) {
        if (prelude.test(index)) text.append(MslPrelude(static_cast<MslPreludeChunk>(index)).text);
    }
    return text;
}

std::string MslEmitterState::FunctionConstant(std::uint32_t id, std::uint32_t defaultValue) {
    if (id > 0xffffu) Fail("function constant id " + std::to_string(id) + " exceeds the Metal limit");
    const auto [found, inserted] = functionConstants.emplace(id, defaultValue);
    if (!inserted && found->second != defaultValue) Fail("function constant " + std::to_string(id) + " has two defaults");
    return "aps5_fc" + std::to_string(id);
}

std::string MslEmitterState::Builtin(MslBuiltin builtin) {
    builtins.set(static_cast<std::size_t>(builtin));
    switch (builtin) {
    case MslBuiltin::ThreadPositionInThreadgroup:
        return "aps5_local_id";
    case MslBuiltin::ThreadgroupPositionInGrid:
        return "aps5_group_id";
    case MslBuiltin::ThreadgroupsPerGrid:
        return "aps5_group_count";
    case MslBuiltin::ThreadIndexInThreadgroup:
        return "aps5_local_index";
    case MslBuiltin::ThreadIndexInSimdgroup:
        return "aps5_simd_lane";
    default:
        Fail("unknown builtin");
    }
}

std::string MslEmitterState::Value(const IrValue* value) {
    if (value == nullptr) Fail("received a null value");
    const IrValue* resolved = value->Resolve();
    if (!resolved->HasImmediate()) return Name(*resolved);
    switch (resolved->Type()) {
    case IrType::Bool:
        return resolved->ImmediateBool() ? "true" : "false";
    case IrType::U8:
        return "uchar(" + MslHex(resolved->ImmediateU8()) + "u)";
    case IrType::U16:
        return "ushort(" + MslHex(resolved->ImmediateU16()) + "u)";
    case IrType::U32:
        return MslHex(resolved->ImmediateU32()) + "u";
    case IrType::S32:
        return "as_type<int>(" + MslHex(resolved->ImmediateU32()) + "u)";
    case IrType::F32:
        return "as_type<float>(" + MslHex(std::bit_cast<std::uint32_t>(resolved->ImmediateF32())) + "u)";
    case IrType::U64:
    case IrType::F64:
        return MslHex(resolved->ImmediateU64()) + "ul";
    case IrType::S64:
        return "as_type<long>(" + MslHex(resolved->ImmediateU64()) + "ul)";
    case IrType::F16:
        return "as_type<half>(ushort(" + MslHex(resolved->ImmediateF16Bits()) + "u))";
    default:
        Fail("immediate of IR type " + TypeToString(resolved->Type()) + " has no MSL literal");
    }
}

std::string MslEmitterState::Name(const IrValue& instruction) {
    const IrValue* resolved = instruction.Resolve();
    std::string name = "v" + std::to_string(resolved->Id());
    if (declared.insert(resolved).second) {
        const auto type = MslType(resolved->Type());
        DeclareLocal(type, name, type + "(0)");
    }
    return name;
}

void MslEmitterState::Define(const IrValue& instruction, std::string_view expression) {
    Line(Name(instruction) + " = " + std::string(expression) + ";");
}

const MemoryInfo& MslEmitterState::Memory(const IrValue& instruction) const {
    const auto index = instruction.Flags<MemoryFlags>().index;
    const auto& memory = program.Resources().memoryInfo;
    if (index >= memory.size()) Fail(instruction, "memory metadata index is out of range");
    return memory[index];
}

void MslEmitterState::Fail(std::string_view reason) const {
    throw std::runtime_error("MSL emission failed: hash=0x" + std::to_string(program.Resources().shaderHash) + " stage=" + std::to_string(static_cast<unsigned>(program.Resources().stage)) + " reason=" + std::string(reason));
}

void MslEmitterState::Fail(const IrValue& instruction, std::string_view reason) const {
    throw std::runtime_error("MSL emission failed: hash=0x" + std::to_string(program.Resources().shaderHash) + " stage=" + std::to_string(static_cast<unsigned>(program.Resources().stage)) + " opcode=" + std::string(IrOpcodeName(instruction.Opcode())) + " reason=" + std::string(reason));
}

std::string MslEmitterState::ShaderDataDword(std::uint32_t dword) {
    const IrBindingLayout& layout = program.Metadata().bindings;
    if (layout.UsesPushData()) {
        usesPushData = true;
        const auto base = FunctionConstant(PipelineSpecialization::PushDataOffset, layout.pushDataStartDword);
        return "aps5_push[" + base + " + " + std::to_string(dword) + "u]";
    }
    return "reinterpret_cast<device const uint*>(" + TableBuffer(DescriptorBindingKind::ShaderData, 0u) + ".address)[" + std::to_string(dword) + "]";
}

std::string MslEmitterState::MemoryByteOffset(std::uint32_t element) {
    const IrBindingLayout& layout = program.Metadata().bindings;
    if (element >= layout.memoryOffsetCount) Fail("memory offset element " + std::to_string(element) + " is outside the layout");
    return "((" + ShaderDataDword(layout.memoryOffsetDword + element / 4u) + " >> " + std::to_string((element % 4u) * 8u) + "u) & 0xffu)";
}

std::string MslEmitterState::TableBuffer(DescriptorBindingKind kind, std::uint32_t element) {
    if (MslAbi::KindOf(kind) != MslAbi::EntryKind::Buffer) Fail("binding " + MslTableField(kind) + " is not a buffer binding");
    const auto& descriptors = program.Metadata().bindings.descriptors;
    const auto found = std::find_if(descriptors.begin(), descriptors.end(), [&](const IrDescriptorBinding& descriptor) { return descriptor.kind == kind; });
    if (found == descriptors.end()) Fail("descriptor binding " + MslTableField(kind) + " was not allocated");
    if (element >= std::max<std::size_t>(found->resources.size(), 1u)) Fail("descriptor element " + std::to_string(element) + " of " + MslTableField(kind) + " is out of range");
    usesTable = true;
    return "aps5_table." + MslTableField(kind) + "[" + std::to_string(element) + "]";
}

std::uint32_t MslEmitterState::DescriptorElement(DescriptorBindingKind kind, std::uint32_t resource) const {
    for (const auto& descriptor : program.Metadata().bindings.descriptors) {
        if (descriptor.kind != kind) continue;
        const auto found = std::find(descriptor.resources.begin(), descriptor.resources.end(), resource);
        if (found == descriptor.resources.end()) Fail("resource " + std::to_string(resource) + " is absent from descriptor binding " + MslTableField(kind));
        return static_cast<std::uint32_t>(found - descriptor.resources.begin());
    }
    Fail("descriptor binding " + MslTableField(kind) + " was not allocated");
}

std::string MslEmitterState::LoopGuardLimit() {
    usesDriverData = true;
    return "aps5_driver.loopGuardLimit";
}

MslShader MslEmitter::Emit(const IrProgram& program, const ShaderStageInputInfo& inputInfo, const CompiledBindingLayout& bindings, const MslTargetOptions& target) const {
    const auto stage = program.Resources().stage;
    if (!program.Resources().srtPlanComplete || !program.Resources().resourceTrackingComplete || !program.Metadata().shaderInfoComplete || !program.Metadata().bindingLayoutComplete) {
        throw std::runtime_error("MSL emission failed: the emitter requires a fully planned native shader program");
    }
    ValidateProgram(program, true);
    if ((program.Info().usesFaultBuffer || program.Info().usesDma) && target.bdaAbiVersion != BdaAbi::Version) throw std::runtime_error("MSL emission failed: the program needs the BDA ABI, which the target lacks");
    MslEmitterState state(program, inputInfo, bindings, target);
    state.reflection.stage = ReflectedStage(stage);
    state.reflection.group = StageGroup(stage);
    state.hostSubgroupSize = target.subgroupSize;
    state.workgroup = WorkgroupInput(program, inputInfo);
    state.laneCount = state.workgroup != nullptr && program.WaveSize() == 64u && state.workgroup->hostSubgroupSize == 32u ? 2u : 1u;
    if (stage != IrShaderStage::Compute) state.Fail("only compute shaders are implemented in the MSL backend so far");
    if (state.laneCount != 1u) state.Fail("wave64 programs on 32-wide SIMD groups (laneCount 2) are not implemented in the MSL backend yet");
    if (program.Info().sharedMemoryBytes != 0u) state.Fail("device-memory LDS is not implemented in the MSL backend yet");
    EmitMslFunctionBody(state);
    const auto entry = EmitMslEntryPoint(state);
    MslShader shader;
    shader.reflection = std::move(state.reflection);
    shader.reflection.usesPushData = state.usesPushData;
    shader.reflection.usesDriverData = state.usesDriverData;
    for (const auto& [id, value] : state.functionConstants) shader.reflection.functionConstants.push_back(id);
    shader.source = "#include <metal_stdlib>\nusing namespace metal;\n\n" + state.Prelude() + entry;
    static const bool dump = std::getenv("APS5_MSL_DUMP") != nullptr;
    if (dump) std::fprintf(stderr, "[msl] shader 0x%llx:\n%s\n[msl] IR:\n%s\n", static_cast<unsigned long long>(program.Resources().shaderHash), shader.source.c_str(), ProgramToString(program).c_str());
    return shader;
}

}
