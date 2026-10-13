#include "MetalBackend/MslEmitterState.hpp"
#include "BdaAbi.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ShaderRecompiler {

namespace {

// Threads per SIMD group on every Apple GPU (MTLComputePipelineState.threadExecutionWidth).
constexpr std::uint32_t AppleSimdWidth = 32u;

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
        return MslAbi::Stage::Vertex;
    case IrShaderStage::Local:
        return MslAbi::Stage::Local;
    case IrShaderStage::TessellationControl:
        return MslAbi::Stage::Hull;
    case IrShaderStage::TessellationEvaluation:
        return MslAbi::Stage::Domain;
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

std::string Vote(const std::string& predicate) {
    return "uint(static_cast<simd_vote::vote_t>(simd_ballot(" + predicate + ")))";
}

std::string Component(std::string_view vector, std::uint32_t component) {
    static constexpr const char* Names[] = {".x", ".y", ".z"};
    if (component > 2u) throw std::runtime_error("MSL emission failed: builtin component " + std::to_string(component) + " is out of range");
    return std::string(vector) + Names[component];
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

bool MslEmitterState::SharedAcrossHalves(const IrValue& instruction) const {
    if (laneCount == 1u) return true;
    const IrValue* resolved = instruction.Resolve();
    return resolved->Type() != IrType::Void && !IrOpcodeHasSideEffects(resolved->Opcode()) && waveUniform.contains(resolved);
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

std::string MslEmitterState::Temporary(std::string_view prefix) {
    return std::string(prefix) + std::to_string(nextTemporary++);
}

void MslEmitterState::AddDeclaration(std::string_view text) {
    declarations.append(text);
    if (!text.empty() && text.back() != '\n') declarations.push_back('\n');
}

void MslEmitterState::AddParameter(std::string_view text) {
    if (parameterSet.emplace(text).second) parameters.emplace_back(text);
}

void MslEmitterState::AddPrologue(std::string_view line) {
    prologue.append("    ");
    prologue.append(line);
    prologue.push_back('\n');
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

std::string MslEmitterState::HalfValue(const IrValue* value, std::uint32_t laneHalf) {
    if (value == nullptr) Fail("received a null value");
    const IrValue* resolved = value->Resolve();
    if (!resolved->HasImmediate()) return HalfName(*resolved, laneHalf);
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

std::string MslEmitterState::HalfName(const IrValue& instruction, std::uint32_t laneHalf) {
    if (laneHalf >= laneCount) Fail("lane half " + std::to_string(laneHalf) + " does not exist (laneCount " + std::to_string(laneCount) + ")");
    const IrValue* resolved = instruction.Resolve();
    const bool second = laneHalf == 1u && !SharedAcrossHalves(*resolved);
    std::string name = "v" + std::to_string(resolved->Id()) + (second ? "_h" : "");
    if (declared[second ? 1u : 0u].insert(resolved).second) {
        const auto type = MslType(resolved->Type());
        DeclareLocal(type, name, type + "(0)");
    }
    return name;
}

void MslEmitterState::Define(const IrValue& instruction, std::string_view expression) {
    Line(Name(instruction) + " = " + std::string(expression) + ";");
}

std::string MslEmitterState::GuestLane() {
    if (laneCount == 2u) {
        const auto lane = Builtin(MslBuiltin::ThreadIndexInSimdgroup);
        return half == 0u ? lane : "(" + lane + " + 32u)";
    }
    // Graphics stages have no threadgroup index: their wave64 programs keep the host SIMD lane, and
    // cross-lane reads past it fail (WaveBallot).
    if (hostSubgroupSize < program.WaveSize() && ThreadgroupStage()) return "(" + Builtin(MslBuiltin::ThreadIndexInThreadgroup) + " & " + std::to_string(program.WaveSize() - 1u) + "u)";
    return Builtin(MslBuiltin::ThreadIndexInSimdgroup);
}

std::string MslEmitterState::GuestInvocationIndex() {
    const auto index = Builtin(MslBuiltin::ThreadIndexInThreadgroup);
    if (laneCount == 1u) return index;
    // Host invocation h runs guest lanes h % 32 and h % 32 + 32 of guest wave h / 32 (as
    // EmitLocalInvocationIndex on the SPIR-V path).
    return "(" + index + " + (" + index + " & ~31u)" + (half == 0u ? std::string() : " + 32u") + ")";
}

std::string MslEmitterState::GuestLocalInvocationId(std::uint32_t component) {
    if (component > 2u) Fail("invocation id component " + std::to_string(component) + " is out of range");
    if (laneCount == 1u) return Component(Builtin(MslBuiltin::ThreadPositionInThreadgroup), component);
    if (workgroup == nullptr) Fail("invocation ids of a wave64 program need its workgroup size");
    std::uint32_t divisor = 1u;
    for (std::uint32_t axis = 0; axis < component; ++axis) divisor *= std::max(workgroup->threadsNum[axis], 1u);
    const auto size = std::max(workgroup->threadsNum[component], 1u);
    return "((" + GuestInvocationIndex() + " / " + std::to_string(divisor) + "u) % " + std::to_string(size) + "u)";
}

std::string MslEmitterState::GuestGlobalInvocationId(std::uint32_t component) {
    if (component > 2u) Fail("invocation id component " + std::to_string(component) + " is out of range");
    const auto size = workgroup != nullptr ? std::max(workgroup->threadsNum[component], 1u) : 1u;
    return "(" + WorkgroupId(component) + " * " + std::to_string(size) + "u + " + GuestLocalInvocationId(component) + ")";
}

std::string MslEmitterState::WorkgroupId(std::uint32_t component) {
    // Mesh draws are never split: the grid is (mesh groups, instances, 1), as gl_WorkGroupID.
    if (program.Resources().stage == IrShaderStage::Mesh) return Component(Builtin(MslBuiltin::ThreadgroupPositionInGrid), component);
    usesDriverData = true;
    return "(" + Component(Builtin(MslBuiltin::ThreadgroupPositionInGrid), component) + " + aps5_driver.workgroupBase[" + std::to_string(component) + "])";
}

// Graphics stages (no threadgroups): a wave64 program runs at one lane per invocation and its guest
// wave spans two host SIMD groups. Each SIMD group then acts as a 32-lane view of the wave: lane ids
// are the SIMD lane, ballots hold the group's lanes in the low word, votes cover the group. EXEC
// masks saved and restored through ballots stay per lane, and branch votes stay correct because the
// lanes' EXEC bits still guard every effect; operations that need the other half fail (ReadLane of a
// lane past 31) or see only this group (bit counts of a whole-wave mask: TechnicalDebt).
bool MslGraphicsHalfWave(const MslEmitterState& state) {
    return state.laneCount == 1u && state.program.WaveSize() > state.hostSubgroupSize && !state.ThreadgroupStage();
}

std::string MslEmitterState::WaveBallot(const IrValue* predicate) {
    if (laneCount == 2u) return "uint2(" + Vote(HalfValue(predicate, 0u)) + ", " + Vote(HalfValue(predicate, 1u)) + ")";
    if (MslGraphicsHalfWave(*this)) return "uint2(" + Vote(Value(predicate)) + ", 0u)";
    if (program.WaveSize() > hostSubgroupSize) Fail("a ballot of a wave64 program at one lane per invocation spans two host SIMD groups");
    return "uint2(" + Vote(Value(predicate)) + ", 0u)";
}

std::string MslEmitterState::WaveAny(const IrValue* predicate) {
    if (laneCount == 2u) return "bool(uint(simd_any(" + HalfValue(predicate, 0u) + ")) | uint(simd_any(" + HalfValue(predicate, 1u) + ")))";
    if (program.WaveSize() > hostSubgroupSize && !MslGraphicsHalfWave(*this)) Fail("a wave vote of a wave64 program at one lane per invocation spans two host SIMD groups");
    return "simd_any(" + Value(predicate) + ")";
}

std::string MslEmitterState::WaveAll(const IrValue* predicate) {
    if (laneCount == 2u) return "bool(uint(simd_all(" + HalfValue(predicate, 0u) + ")) & uint(simd_all(" + HalfValue(predicate, 1u) + ")))";
    if (program.WaveSize() > hostSubgroupSize && !MslGraphicsHalfWave(*this)) Fail("a wave vote of a wave64 program at one lane per invocation spans two host SIMD groups");
    return "simd_all(" + Value(predicate) + ")";
}

const MemoryInfo& MslEmitterState::Memory(const IrValue& instruction) const {
    const auto index = instruction.Flags<MemoryFlags>().index;
    const auto& memory = program.Resources().memoryInfo;
    if (index >= memory.size()) Fail(instruction, "memory metadata index is out of range");
    return memory[index];
}

void MslEmitterState::Fail(std::string_view reason) const {
    throw std::runtime_error("MSL emission failed: hash=" + MslHex(program.Resources().shaderHash) + " stage=" + std::to_string(static_cast<unsigned>(program.Resources().stage)) + " reason=" + std::string(reason));
}

void MslEmitterState::Fail(const IrValue& instruction, std::string_view reason) const {
    throw std::runtime_error("MSL emission failed: hash=" + MslHex(program.Resources().shaderHash) + " stage=" + std::to_string(static_cast<unsigned>(program.Resources().stage)) + " opcode=" + std::string(IrOpcodeName(instruction.Opcode())) + " reason=" + std::string(reason));
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
    return TableResource(kind, element);
}

std::string MslEmitterState::TableResource(DescriptorBindingKind kind, std::uint32_t element) {
    const auto& descriptors = program.Metadata().bindings.descriptors;
    const auto found = std::find_if(descriptors.begin(), descriptors.end(), [&](const IrDescriptorBinding& descriptor) { return descriptor.kind == kind; });
    if (found == descriptors.end()) Fail("descriptor binding " + MslTableField(kind) + " was not allocated");
    if (element >= std::max<std::size_t>(found->resources.size(), 1u)) Fail("descriptor element " + std::to_string(element) + " of " + MslTableField(kind) + " is out of range");
    usesTable = true;
    usedKinds.insert(kind);
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
    // MslAbi::DriverData: 0 means no guard.
    if (!loopGuardDeclared) AddPrologue("const uint aps5_loop_limit = aps5_driver.loopGuardLimit != 0u ? aps5_driver.loopGuardLimit : 0xffffffffu;");
    loopGuardDeclared = true;
    return "aps5_loop_limit";
}

bool MslEmitterState::ThreadgroupStage() const {
    const auto stage = program.Resources().stage;
    return stage == IrShaderStage::Compute || stage == IrShaderStage::Mesh;
}

bool MslEmitterState::LdsInDeviceMemory() const {
    return program.Info().sharedMemoryBytes != 0u;
}

std::string MslEmitterState::LdsMemoryFlags() const {
    return LdsInDeviceMemory() ? "mem_flags::mem_device | mem_flags::mem_threadgroup" : "mem_flags::mem_threadgroup";
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
    // Apple SIMD groups are 32 wide whatever width the target claims, and the front end lays MSL
    // programs out for 32 lanes (Recompiler.cpp HostSubgroupSize). Only the APS5_SINGLE_LANE debug
    // aid lays a wave64 program out at one lane per invocation for 64 lanes: the emitter still maps
    // it to the 32-lane hardware, so graphics stages take the half-wave view and compute wave ops
    // across two SIMD groups throw, instead of shuffling lanes 32-63 that do not exist (undefined in
    // MSL; they read 0 on the M3).
    state.hostSubgroupSize = std::min(target.subgroupSize, AppleSimdWidth);
    state.workgroup = WorkgroupInput(program, inputInfo);
    state.laneCount = state.workgroup != nullptr && program.WaveSize() == 64u && state.workgroup->hostSubgroupSize == 32u ? 2u : 1u;
    // Wave-uniform values: computed once for both lane halves at laneCount 2; at any lane count the
    // stores whose address is wave-uniform keep only the highest active lane's data (MslBufferShared).
    if (state.ThreadgroupStage()) state.waveUniform = WaveUniformValues(program);
    static const bool survey = std::getenv("APS5_MSL_SURVEY") != nullptr;
    state.survey = survey;
    if (stage != IrShaderStage::Compute && stage != IrShaderStage::Vertex && stage != IrShaderStage::Pixel && stage != IrShaderStage::Mesh && !MslTessellationStage(stage)) state.Fail("unsupported shader stage for the MSL backend");
    if ((stage == IrShaderStage::Vertex || stage == IrShaderStage::Mesh || MslTessellationStage(stage)) && inputInfo.vertex == nullptr) state.Fail("vertex input info is missing (work package B6)");
    state.waveLdsScope = MslWaveLdsScope(state);
    EmitMslFunctionBody(state);
    const auto entry = EmitMslEntryPoint(state);
    if (!state.surveyFailures.empty()) {
        std::string text = "survey: " + std::to_string(state.surveyFailures.size()) + " instruction family failures:";
        for (const auto& failure : state.surveyFailures) text += " [" + failure + "]";
        state.Fail(text);
    }
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
