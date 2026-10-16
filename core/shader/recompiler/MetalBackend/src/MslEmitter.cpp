#include "MetalBackend/MslEmitterState.hpp"
#include "MetalBackend/MslLaneLocal.hpp"
#include "BdaAbi.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

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

namespace {

bool IdentifierStart(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool IdentifierChar(char c) {
    return IdentifierStart(c) || (c >= '0' && c <= '9');
}

// `vN` or `vN_h`: the names HalfName gives SSA values.
bool SsaName(std::string_view name) {
    if (name.size() < 2u || name[0] != 'v') return false;
    std::size_t index = 1u;
    while (index < name.size() && name[index] >= '0' && name[index] <= '9') ++index;
    if (index == 1u) return false;
    return index == name.size() || name.substr(index) == "_h";
}

}

std::size_t MslSinkDeclarations(std::string& locals, const std::string& prologue, std::string& body) {
    struct Candidate {
        std::string type;
        std::size_t localBegin = 0;
        std::size_t localEnd = 0;
        std::int64_t scope = -1;
        std::size_t definition = std::string::npos;
        bool eligible = false;
    };
    std::unordered_map<std::string, Candidate> candidates;
    // Locals of the exact form "    T vN = T(0);".
    for (std::size_t begin = 0; begin < locals.size();) {
        std::size_t end = locals.find('\n', begin);
        end = end == std::string::npos ? locals.size() : end + 1u;
        const std::string_view line(locals.data() + begin, end - begin);
        if (line.starts_with("    ") && line.ends_with(";\n")) {
            const std::string_view rest = line.substr(4u, line.size() - 6u);
            const auto space = rest.find(' ');
            const auto equals = rest.find(" = ");
            if (space != std::string_view::npos && equals != std::string_view::npos && space < equals) {
                const std::string_view type = rest.substr(0, space);
                const std::string_view name = rest.substr(space + 1u, equals - space - 1u);
                const std::string_view initializer = rest.substr(equals + 3u);
                if (SsaName(name) && initializer.size() == type.size() + 3u && initializer.starts_with(type) && initializer.ends_with("(0)")) {
                    Candidate candidate;
                    candidate.type = type;
                    candidate.localBegin = begin;
                    candidate.localEnd = end;
                    candidates.emplace(std::string(name), std::move(candidate));
                }
            }
        }
        begin = end;
    }
    if (candidates.empty()) return 0u;

    // Block structure: every '{' opens a scope, '}' closes it; a scope with a case label is a switch
    // body (a declaration there could be jumped over).
    std::vector<std::int64_t> stack{0};
    std::vector<bool> active{true};
    std::vector<bool> switchScope{false};
    bool failed = false;
    auto scan = [&](const std::string& text, bool isBody) {
        char previousEnd = ';';
        for (std::size_t begin = 0; begin < text.size() && !failed;) {
            std::size_t end = text.find('\n', begin);
            if (end == std::string::npos) end = text.size();
            std::size_t first = begin;
            while (first < end && (text[first] == ' ' || text[first] == '\t')) ++first;
            if (first < end && text[first] == '#') {
                failed = true;
                return;
            }
            const std::string_view trimmed(text.data() + first, end - first);
            if (trimmed.starts_with("case ") || trimmed.starts_with("default:")) switchScope[static_cast<std::size_t>(stack.back())] = true;
            // A definition statement: "vN = expression;" on a line of its own, after a complete statement.
            const bool statement = isBody && trimmed.ends_with(";") && (previousEnd == ';' || previousEnd == '{' || previousEnd == '}');
            bool firstToken = true;
            std::size_t codeEnd = end;
            for (std::size_t index = first; index < end;) {
                const char c = text[index];
                if (c == '/' && index + 1u < end && text[index + 1u] == '/') {
                    codeEnd = index;
                    break;
                }
                if (c == '{') {
                    stack.push_back(static_cast<std::int64_t>(active.size()));
                    active.push_back(true);
                    switchScope.push_back(false);
                    ++index;
                    firstToken = false;
                    continue;
                }
                if (c == '}') {
                    if (stack.size() <= 1u) {
                        failed = true;
                        return;
                    }
                    active[static_cast<std::size_t>(stack.back())] = false;
                    stack.pop_back();
                    ++index;
                    firstToken = false;
                    continue;
                }
                if (IdentifierChar(c)) {
                    std::size_t tokenEnd = index;
                    while (tokenEnd < end && IdentifierChar(text[tokenEnd])) ++tokenEnd;
                    if (IdentifierStart(c)) {
                        const std::string_view token(text.data() + index, tokenEnd - index);
                        if (const auto found = token[0] == 'v' ? candidates.find(std::string(token)) : candidates.end(); found != candidates.end()) {
                            auto& candidate = found->second;
                            if (candidate.scope < 0) {
                                candidate.scope = stack.back();
                                const std::string_view after(text.data() + tokenEnd, end - tokenEnd);
                                candidate.eligible = statement && firstToken && after.starts_with(" = ");
                                if (candidate.eligible) {
                                    // The expression must not read the value it defines.
                                    const std::string_view expression = after.substr(3u);
                                    for (std::size_t at = expression.find(token); at != std::string_view::npos; at = expression.find(token, at + 1u)) {
                                        const bool before = at == 0u || !IdentifierChar(expression[at - 1u]);
                                        const bool behind = at + token.size() >= expression.size() || !IdentifierChar(expression[at + token.size()]);
                                        if (before && behind) candidate.eligible = false;
                                    }
                                }
                                candidate.definition = index;
                            } else if (!active[static_cast<std::size_t>(candidate.scope)]) {
                                candidate.eligible = false;
                            }
                        }
                    }
                    index = tokenEnd;
                    firstToken = false;
                    continue;
                }
                if (c != ' ' && c != '\t') firstToken = false;
                ++index;
            }
            // The last code character of the line (comments and blank lines leave it unchanged).
            std::size_t last = codeEnd;
            while (last > first && (text[last - 1u] == ' ' || text[last - 1u] == '\t')) --last;
            if (last > first) previousEnd = text[last - 1u];
            begin = end + 1u;
        }
    };
    scan(prologue, false);
    scan(body, true);
    if (failed) return 0u;

    std::vector<std::pair<std::size_t, const Candidate*>> sunk;
    for (const auto& [name, candidate] : candidates) {
        if (candidate.eligible && candidate.scope >= 0 && !switchScope[static_cast<std::size_t>(candidate.scope)]) sunk.emplace_back(candidate.definition, &candidate);
    }
    if (sunk.empty()) return 0u;
    std::sort(sunk.begin(), sunk.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
    std::string rewritten;
    rewritten.reserve(body.size() + sunk.size() * 8u);
    std::size_t copied = 0;
    for (const auto& [offset, candidate] : sunk) {
        rewritten.append(body, copied, offset - copied);
        rewritten.append(candidate->type);
        rewritten.push_back(' ');
        copied = offset;
    }
    rewritten.append(body, copied, std::string::npos);
    body = std::move(rewritten);
    std::vector<std::pair<std::size_t, std::size_t>> removed;
    removed.reserve(sunk.size());
    for (const auto& entry : sunk) removed.emplace_back(entry.second->localBegin, entry.second->localEnd);
    std::sort(removed.begin(), removed.end());
    std::string kept;
    kept.reserve(locals.size());
    std::size_t from = 0;
    for (const auto& [begin, end] : removed) {
        kept.append(locals, from, begin - from);
        from = end;
    }
    kept.append(locals, from, std::string::npos);
    locals = std::move(kept);
    return sunk.size();
}

std::string MslFunctionStatements(const MslEmitterState& state, const std::string& prologue) {
    static const bool hoistAll = [] {
        const char* value = std::getenv("APS5_MSL_HOIST_ALL");
        return value != nullptr && value[0] == '1';
    }();
    if (hoistAll) return state.locals + prologue + state.body;
    std::string locals = state.locals;
    std::string body = state.body;
    static_cast<void>(MslSinkDeclarations(locals, prologue, body));
    return locals + prologue + body;
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
    // One-lane waves (upstream #4470, SingleLaneDefines): every invocation is lane 0.
    if (hostSubgroupSize == 1u) return "0u";
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
    // Mesh draws: the grid is (mesh groups, instances, splits), gl_WorkGroupID is (x, y, 0). The z
    // dimension splits a group's primitives in the triangle-vertices mode (MslAbi Version 13).
    if (program.Resources().stage == IrShaderStage::Mesh) return component == 2u ? std::string("0u") : Component(Builtin(MslBuiltin::ThreadgroupPositionInGrid), component);
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

// Threadgroup exchange (DECISIONS D62): `threadgroup uint aps5_wx[128]` in two alternating halves of 64
// (aps5_wp), so one barrier per exchange suffices: a half is written again only after the next
// exchange's barrier, which every invocation reaches after reading this one.
static void DeclareExchange(MslEmitterState& state) {
    if (!state.threadgroupExchange) state.Fail("a wave64 program at one lane per invocation exchanges values without threadgroup exchange");
    if (state.prologue.find("aps5_wx[") != std::string::npos) return;
    state.AddPrologue("threadgroup uint aps5_wx[128];");
    state.AddPrologue("uint aps5_wp = 0u;");
}

std::string MslEmitterState::ExchangeRead(const std::string& value, const std::string& lane) {
    DeclareExchange(*this);
    Line("aps5_wx[aps5_wp + " + GuestLane() + "] = " + value + ";");
    Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    const auto result = Temporary("aps5_xr");
    Line("const uint " + result + " = aps5_wx[aps5_wp + ((" + lane + ") & 63u)];");
    Line("aps5_wp ^= 64u;");
    return result;
}

// One word per host SIMD group (guest lanes 0-31 and 32-63: the SIMD group of thread_index_in_threadgroup
// / 32, as the two-half layout's lane mapping).
static std::string ExchangeHalves(MslEmitterState& state, const std::string& perGroup) {
    DeclareExchange(state);
    const auto mine = state.Temporary("aps5_xg");
    state.Line("const uint " + mine + " = " + perGroup + ";");
    state.Line("if (simd_is_first()) aps5_wx[aps5_wp + (" + state.GuestLane() + " >> 5u)] = " + mine + ";");
    state.Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    const auto words = state.Temporary("aps5_xw");
    state.Line("const uint2 " + words + " = uint2(aps5_wx[aps5_wp], aps5_wx[aps5_wp + 1u]);");
    state.Line("aps5_wp ^= 64u;");
    return words;
}

std::string MslEmitterState::ExchangeVote(const IrValue* predicate, bool all) {
    const auto words = ExchangeHalves(*this, std::string("uint(") + (all ? "simd_all(" : "simd_any(") + Value(predicate) + "))");
    return all ? "((" + words + ".x & " + words + ".y) != 0u)" : "((" + words + ".x | " + words + ".y) != 0u)";
}

std::string MslEmitterState::ExchangeBallot(const IrValue* predicate) {
    return ExchangeHalves(*this, "uint(static_cast<simd_vote::vote_t>(simd_ballot(" + Value(predicate) + ")))");
}

std::string MslEmitterState::HalfBallot(const IrValue* predicate) {
    if (!laneLocal) return WaveBallot(predicate);
    const auto bits = Temporary("aps5_hb");
    Line("const uint " + bits + " = uint(static_cast<simd_vote::vote_t>(simd_ballot(" + Value(predicate) + ")));");
    return "uint2(" + bits + ", " + bits + ")";
}

std::string MslEmitterState::WaveBallot(const IrValue* predicate) {
    if (laneLocal) {
        if (fullBallots) return ExchangeBallot(predicate);
        // The lane's own bit (MslLaneLocal.cpp (b): only own-bit consumers read it).
        const auto lane = GuestLane();
        const auto value = Value(predicate);
        return "uint2((" + value + ") && " + lane + " < 32u ? 1u << " + lane + " : 0u, (" + value + ") && " + lane + " >= 32u ? 1u << (" + lane + " - 32u) : 0u)";
    }
    if (laneCount == 2u) return "uint2(" + Vote(HalfValue(predicate, 0u)) + ", " + Vote(HalfValue(predicate, 1u)) + ")";
    if (MslGraphicsHalfWave(*this)) return "uint2(" + Vote(Value(predicate)) + ", 0u)";
    if (program.WaveSize() > hostSubgroupSize) Fail("a ballot of a wave64 program at one lane per invocation spans two host SIMD groups");
    return "uint2(" + Vote(Value(predicate)) + ", 0u)";
}

std::string MslEmitterState::WaveAny(const IrValue* predicate) {
    // Branches lanes decide alone never get here (MslFlowEmitter Condition); the others vote.
    if (laneLocal) return ExchangeVote(predicate, false);
    if (laneCount == 2u) return "bool(uint(simd_any(" + HalfValue(predicate, 0u) + ")) | uint(simd_any(" + HalfValue(predicate, 1u) + ")))";
    if (program.WaveSize() > hostSubgroupSize && !MslGraphicsHalfWave(*this)) Fail("a wave vote of a wave64 program at one lane per invocation spans two host SIMD groups");
    return "simd_any(" + Value(predicate) + ")";
}

std::string MslEmitterState::WaveAll(const IrValue* predicate) {
    if (laneLocal) return ExchangeVote(predicate, true);
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

// Upstream #4470: vertex and local programs of a target without vertexSubgroups are laid out as
// one-lane waves (Recompiler.cpp HostSubgroupSize 1), as SpirvSubgroup.cpp's single-lane helpers
// emit them: a ballot is the invocation's own bit, a shuffle, broadcast or reduction its own value,
// an election true, the lane 0. The library maps the SIMD-group functions it calls to those values
// (prelude helpers included) and drops the mapping at its end, so a linked tessellation library's
// other programs keep the real functions (MslAbi Version 10).
constexpr std::string_view SingleLaneDefines = R"(// One-lane waves (MslAbi Version 10): SIMD-group functions over this invocation alone.
#define simd_ballot(p) simd_vote(simd_vote::vote_t((p) ? 1u : 0u))
#define simd_any(p) bool(p)
#define simd_all(p) bool(p)
#define simd_is_first() true
#define simd_broadcast_first(v) (v)
#define simd_broadcast(v, l) (v)
#define simd_shuffle(v, l) (v)
#define simd_shuffle_xor(v, m) (v)
#define simd_shuffle_up(v, d) (v)
#define simd_shuffle_down(v, d) (v)
#define simd_min(v) (v)
#define simd_max(v) (v)
#define simd_sum(v) (v)
#define simd_product(v) (v)
#define simd_and(v) (v)
#define simd_or(v) (v)
#define simd_xor(v) (v)
#define simd_prefix_inclusive_sum(v) (v)
#define simd_prefix_exclusive_sum(v) (decltype(v)(0))

)";

constexpr std::string_view SingleLaneUndefines = R"(
#undef simd_ballot
#undef simd_any
#undef simd_all
#undef simd_is_first
#undef simd_broadcast_first
#undef simd_broadcast
#undef simd_shuffle
#undef simd_shuffle_xor
#undef simd_shuffle_up
#undef simd_shuffle_down
#undef simd_min
#undef simd_max
#undef simd_sum
#undef simd_product
#undef simd_and
#undef simd_or
#undef simd_xor
#undef simd_prefix_inclusive_sum
#undef simd_prefix_exclusive_sum
)";

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
    if (state.hostSubgroupSize == 1u && stage != IrShaderStage::Vertex && stage != IrShaderStage::Local) state.Fail("one-lane waves are only laid out for vertex and local programs (a target without vertexSubgroups)");
    state.workgroup = WorkgroupInput(program, inputInfo);
    state.laneCount = state.workgroup != nullptr && program.WaveSize() == 64u && state.workgroup->hostSubgroupSize == 32u ? 2u : 1u;
    if (state.laneCount == 2u && stage == IrShaderStage::Compute && (target.wave64LaneLocal || std::getenv("APS5_MSL_LANE_SURVEY") != nullptr)) {
        const auto threads = std::max(state.workgroup->threadsNum[0], 1u) * std::max(state.workgroup->threadsNum[1], 1u) * std::max(state.workgroup->threadsNum[2], 1u);
        // Twice the host invocations of the two-half layout: kept within what every pipeline
        // supports (TechnicalDebt metal-wave64-lane-local).
        const auto plan = MslPlanWave64LaneLocal(program, threads);
        // The exchange buffer takes 512 bytes of threadgroup memory beside LDS kept there.
        const auto ldsBytes = program.Info().sharedMemoryBytes != 0u ? 0u : (state.workgroup->ldsSizeDwords + 1u) * 4u;
        const bool exchangeFits = !plan.threadgroupExchange || ldsBytes + 512u <= target.maxThreadgroupMemoryBytes;
        if (target.wave64LaneLocal && plan.eligible && threads <= MslLaneLocalMaxThreads && exchangeFits) {
            state.laneCount = 1u;
            state.laneLocal = true;
            state.threadgroupExchange = plan.threadgroupExchange;
            state.fullBallots = plan.fullBallots;
            state.votedBranches = plan.votedBranches;
        }
    }
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
    const bool singleLane = state.hostSubgroupSize == 1u;
    shader.source = "#include <metal_stdlib>\nusing namespace metal;\n\n" + (singleLane ? std::string(SingleLaneDefines) : std::string()) + state.Prelude() + entry + (singleLane ? std::string(SingleLaneUndefines) : std::string());
    if (const char* survey = std::getenv("APS5_MSL_LANE_SURVEY"); survey != nullptr && (state.laneCount == 2u || state.laneLocal) && stage == IrShaderStage::Compute) {
        if (FILE* out = std::fopen(survey, "a")) {
            std::fprintf(out, "[lane-size] hash=0x%llx bytes=%zu local=%d\n", static_cast<unsigned long long>(program.Resources().shaderHash), shader.source.size(), state.laneLocal ? 1 : 0);
            std::fclose(out);
        }
    }
    static const bool dump = std::getenv("APS5_MSL_DUMP") != nullptr;
    if (dump) std::fprintf(stderr, "[msl] shader 0x%llx:\n%s\n[msl] IR:\n%s\n", static_cast<unsigned long long>(program.Resources().shaderHash), shader.source.c_str(), ProgramToString(program).c_str());
    return shader;
}

}
