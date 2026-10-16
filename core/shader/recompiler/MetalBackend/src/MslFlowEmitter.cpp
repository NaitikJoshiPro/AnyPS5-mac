#include "MetalBackend/MslEmitterState.hpp"
#include "ControlFlow/include/ControlFlow/ControlFlowGraph.hpp"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler {

namespace {

// Instructions a host invocation runs once even when it holds two guest lanes.
bool OncePerInvocation(IrOpcode opcode) {
    return opcode == IrOpcode::Barrier || opcode == IrOpcode::MeshAllocate || opcode == IrOpcode::Unreachable;
}

bool DispatchInstruction(MslEmitterState& state, const IrValue& instruction) {
    return EmitMslControlInstruction(state, instruction) || EmitMslAluInstruction(state, instruction) || EmitMslFloat64Instruction(state, instruction) || EmitMslWaveInstruction(state, instruction) || EmitMslMemoryInstruction(state, instruction) || EmitMslBufferInstruction(state, instruction) || EmitMslImageInstruction(state, instruction) || EmitMslStageInstruction(state, instruction);
}

// Rebuilds nested MSL control flow from the structurizer's SPIR-V-style annotations
// (IrProgramMetadata::blockInfo, paired with BlockOrder by position):
// - a ConditionalBranch with a merge block is `if (c) { true arm } else { false arm }`, both arms
//   ending at the merge;
// - a loop header is a guarded `for` loop: a branch to its merge block is `break`, a branch to its
//   continue block runs the continue construct (the latch: its instructions and the back edge,
//   `continue` or `break`), emitted at the body's natural end and repeated at every earlier
//   branch to it, so `continue` never skips the latch;
// - a conditional branch without a merge must have an exit (break, continue or the region's end)
//   on at least one side.
// Phis are function-scope locals assigned on each incoming edge (parallel copies through
// temporaries). Every SSA value is hoisted to function scope, so dominance carries over.
// Branch conditions are wave-uniform: a wave-mask branch (EXEC/VCC zero or non-zero) votes over the
// guest wave, so both guest lanes of a host invocation (laneCount 2) always take the same path.
class StructuredEmitter {
public:
    explicit StructuredEmitter(MslEmitterState& state) : state(state) {
        const auto& blocks = state.program.BlockOrder();
        const auto& blockInfo = state.program.Metadata().blockInfo;
        if (blocks.size() != blockInfo.size()) state.Fail("control flow block metadata is inconsistent");
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            if (blocks[index] == nullptr) state.Fail("control flow block is null");
            info.emplace(blocks[index], &blockInfo[index]);
            byId.emplace(blockInfo[index].id, blocks[index]);
        }
    }

    void Run() {
        const auto& blocks = state.program.BlockOrder();
        if (blocks.empty()) {
            Return();
            return;
        }
        Region(blocks.front(), nullptr);
    }

private:
    struct Loop {
        const IrBlock* header;
        const IrBlock* merge;
        const IrBlock* continueBlock;
    };

    MslEmitterState& state;
    std::unordered_map<const IrBlock*, const BlockInfo*> info;
    std::unordered_map<std::uint32_t, const IrBlock*> byId;
    std::vector<Loop> loops;
    std::unordered_set<const IrBlock*> emitted;

    const BlockInfo& Info(const IrBlock* block) const {
        const auto found = info.find(block);
        if (found == info.end()) state.Fail("control flow block has no metadata");
        return *found->second;
    }

    const IrBlock* Target(std::uint32_t id) const {
        if (id == InvalidControlFlowId) return nullptr;
        const auto found = byId.find(id);
        if (found == byId.end()) state.Fail("control flow target block " + std::to_string(id) + " is missing");
        return found->second;
    }

    const Loop* CurrentLoop() const {
        return loops.empty() ? nullptr : &loops.back();
    }

    bool IsExit(const IrBlock* target, const IrBlock* stop) const {
        if (target == stop) return true;
        const auto* loop = CurrentLoop();
        return loop != nullptr && (target == loop->merge || target == loop->continueBlock);
    }

    // Wave LDS ordering (see MslWaveLdsScope): a barrier separates an LDS write from the next LDS
    // access and a read from the next write. A block may be entered right after a write.
    void WaveLdsBarrier(const IrValue& instruction, bool& written, bool& read) {
        if (state.waveLdsScope == MslBarrierScope::None) return;
        if (instruction.Opcode() == IrOpcode::Barrier) {
            written = false;
            read = false;
            return;
        }
        const auto access = SharedAccessOf(instruction.Opcode());
        if (access == SharedAccess::None) return;
        const bool writes = access != SharedAccess::Read;
        if (written || (writes && read)) {
            const auto function = state.waveLdsScope == MslBarrierScope::Threadgroup ? "threadgroup_barrier(" : "simdgroup_barrier(";
            state.Line(std::string(function) + state.LdsMemoryFlags() + ");");
            written = false;
            read = false;
        }
        written |= writes;
        read |= access != SharedAccess::Write;
    }

    void Instruction(const IrValue& instruction) {
        if (!state.survey) {
            if (!DispatchInstruction(state, instruction)) state.Fail(instruction, "has no MSL emitter");
            return;
        }
        // Debug aid: record the failure, keep going (the emission still fails at the end).
        const auto depth = state.Depth();
        std::string failure;
        try {
            if (!DispatchInstruction(state, instruction)) failure = "opcode=" + std::string(IrOpcodeName(instruction.Opcode())) + " has no MSL emitter";
        } catch (const std::runtime_error& error) {
            failure = error.what();
            const auto reason = failure.find(" opcode=");
            if (reason != std::string::npos) failure = failure.substr(reason + 1u);
            state.RestoreDepth(depth);
        }
        if (!failure.empty() && std::find(state.surveyFailures.begin(), state.surveyFailures.end(), failure) == state.surveyFailures.end()) state.surveyFailures.push_back(failure);
    }

    void Body(const IrBlock* block, bool repeat) {
        if (!repeat && !emitted.insert(block).second) state.Fail("control flow block " + std::to_string(Info(block).id) + " would be emitted twice");
        bool ldsWritten = true;
        bool ldsRead = false;
        for (const auto* instruction : block->Instructions()) {
            if (instruction->IsPhi()) continue;
            WaveLdsBarrier(*instruction, ldsWritten, ldsRead);
            for (std::uint32_t half = 0; half < state.laneCount; ++half) {
                if (half != 0u && (OncePerInvocation(instruction->Opcode()) || state.SharedAcrossHalves(*instruction))) continue;
                state.half = half;
                Instruction(*instruction);
            }
            state.half = 0u;
        }
    }

    void PhiCopies(const IrBlock* from, const IrBlock* to) {
        struct Copy {
            std::string phi;
            std::string type;
            std::string value;
        };
        std::vector<Copy> copies;
        for (const auto* instruction : to->Instructions()) {
            if (!instruction->IsPhi()) continue;
            const IrValue* incoming = nullptr;
            for (std::size_t index = 0; index < instruction->PhiBlockCount(); ++index) {
                if (instruction->PhiBlock(index) == from) incoming = instruction->Argument(index);
            }
            if (incoming == nullptr) state.Fail(*instruction, "has no incoming value for block " + std::to_string(Info(from).id));
            for (std::uint32_t half = 0; half < state.laneCount; ++half) {
                if (half != 0u && state.SharedAcrossHalves(*instruction)) continue;
                copies.push_back({state.HalfName(*instruction, half), MslType(instruction->Type()), state.HalfValue(incoming, half)});
            }
        }
        if (copies.size() == 1u) {
            state.Line(copies.front().phi + " = " + copies.front().value + ";");
            return;
        }
        std::vector<std::string> temporaries;
        for (const auto& copy : copies) {
            temporaries.push_back(state.Temporary());
            state.Line("const " + copy.type + " " + temporaries.back() + " = " + copy.value + ";");
        }
        for (std::size_t index = 0; index < copies.size(); ++index) state.Line(copies[index].phi + " = " + temporaries[index] + ";");
    }

    std::string Condition(const BlockInfo& blockInfo) {
        if (blockInfo.condition == nullptr) state.Fail("conditional branch of block " + std::to_string(blockInfo.id) + " has no condition");
        const auto kind = blockInfo.terminator.condition;
        const auto* condition = blockInfo.condition;
        if (kind == BranchCondition::ScalarInstruction || kind == BranchCondition::GotoVariable || kind == BranchCondition::IndirectTarget) return state.HalfValue(condition, 0u);
        if (state.laneCount == 1u && !IsWaveMaskBranch(kind)) return state.HalfValue(condition, 0u);
        // Wave64 at one lane per invocation (DECISIONS D62): the lane's own condition where
        // MslLaneLocal.cpp proved that exact, else a vote through threadgroup memory.
        if (state.laneLocal && !state.votedBranches.contains(blockInfo.id)) return state.HalfValue(condition, 0u);
        // A wave-uniform condition is the same in every lane: no vote needed.
        if (state.laneCount == 2u && (condition->Resolve()->HasImmediate() || state.SharedAcrossHalves(*condition))) return state.HalfValue(condition, 0u);
        // EXEC/VCC (and, at laneCount 2, every other per-lane condition as the SPIR-V path does):
        // the zero forms branch when every lane's predicate holds, the others when any lane's does.
        const bool zero = kind == BranchCondition::ExecZero || kind == BranchCondition::VccZero || kind == BranchCondition::SccZero;
        return zero ? state.WaveAll(condition) : state.WaveAny(condition);
    }

    void Return() {
        state.Line("return;");
    }

    // The edge from -> target when target leaves the current construct: the region's end, the
    // loop's merge (break) or its continue block (the continue construct). False otherwise.
    bool Exit(const IrBlock* from, const IrBlock* target, const IrBlock* stop) {
        if (target == stop) {
            PhiCopies(from, target);
            return true;
        }
        const auto* loop = CurrentLoop();
        if (loop == nullptr) return false;
        if (target == loop->merge) {
            PhiCopies(from, target);
            state.Line("break;");
            return true;
        }
        if (target == loop->continueBlock) {
            PhiCopies(from, target);
            ContinueConstruct(*loop, true);
            return true;
        }
        return false;
    }

    // The latch: its instructions (unless it is the header, emitted once at the loop's top) and the
    // back edge, which always ends with `continue` or `break`.
    void ContinueConstruct(const Loop& loop, bool emitBody) {
        const auto* latch = loop.continueBlock;
        if (emitBody && latch != loop.header) Body(latch, true);
        const auto& blockInfo = Info(latch);
        const auto& term = blockInfo.terminator;
        const auto edge = [&](const IrBlock* target) {
            PhiCopies(latch, target);
            if (target == loop.header) state.Line("continue;");
            else if (target == loop.merge) state.Line("break;");
            else state.Fail("continue block " + std::to_string(blockInfo.id) + " branches outside its loop's header and merge");
        };
        switch (term.kind) {
        case TerminatorKind::Branch: {
            const auto* target = Target(term.trueBlock);
            if (target == nullptr) {
                Return();
                return;
            }
            edge(target);
            return;
        }
        case TerminatorKind::ConditionalBranch:
            state.Line("if (" + Condition(blockInfo) + ") {");
            state.Indent();
            edge(Target(term.trueBlock));
            state.Outdent();
            state.Line("} else {");
            state.Indent();
            edge(Target(term.falseBlock));
            state.Outdent();
            state.Line("}");
            return;
        case TerminatorKind::Return:
            Return();
            return;
        default:
            Unstructured(blockInfo);
        }
    }

    [[noreturn]] void Unstructured(const BlockInfo& blockInfo) const {
        // The front end lowers indirect branches to IndirectTarget conditional chains and refuses
        // unsupported ones, so neither reaches an emitter.
        state.Fail("block " + std::to_string(blockInfo.id) + " ends in an indirect or unsupported branch, which the front end lowers before emission");
    }

    void EmitLoop(const IrBlock* header) {
        const auto& blockInfo = Info(header);
        const auto* merge = Target(blockInfo.terminator.mergeBlock);
        const auto* latch = Target(blockInfo.terminator.continueBlock);
        if (merge == nullptr || latch == nullptr) state.Fail("loop header " + std::to_string(blockInfo.id) + " lacks a merge or continue block");
        state.RequirePrelude(MslPreludeChunk::LoopGuard);
        const auto guard = "aps5_loop" + std::to_string(blockInfo.id);
        state.DeclareLocal("uint", guard, "0u");
        const auto limit = state.LoopGuardLimit();
        // A bounded loop: the Metal compiler mishandles constant-true loop conditions (Mesa
        // KosmicKrisp workaround list), and the guard keeps a runaway guest loop from hanging the GPU.
        // The guard counts per entry; every invocation of a SIMD group counts alike (the loop is
        // wave-uniform), so a tripped guard ends the whole guest wave.
        state.Line("for (" + guard + " = 0u; " + guard + " < " + limit + "; ++" + guard + ") {");
        state.Indent();
        loops.push_back({header, merge, latch});
        if (latch == header) {
            Body(header, false);
            ContinueConstruct(loops.back(), false);
        } else if (Region(header, latch)) {
            ContinueConstruct(loops.back(), true);
        }
        loops.pop_back();
        state.Outdent();
        state.Line("}");
        state.Line("if (" + guard + " >= " + limit + ") {");
        state.Indent();
        state.Line("aps5_loop_guard_tripped(aps5_driver, " + MslHex(state.program.Resources().shaderHash) + "ul, " + std::to_string(static_cast<unsigned>(state.program.Resources().stage)) + "u, " + std::to_string(blockInfo.startPc) + "u);");
        Return();
        state.Outdent();
        state.Line("}");
    }

    // Emits from `block` until control reaches `stop`. Returns whether it does (false: every path
    // ended with break, continue or return).
    bool Region(const IrBlock* block, const IrBlock* stop) {
        bool entering = !loops.empty() && loops.back().header == block;
        while (block != nullptr && block != stop) {
            const auto* loop = CurrentLoop();
            if (!entering && loop != nullptr) {
                if (block == loop->continueBlock) {
                    ContinueConstruct(*loop, true);
                    return false;
                }
                if (block == loop->merge) {
                    state.Line("break;");
                    return false;
                }
            }
            const auto& blockInfo = Info(block);
            if (blockInfo.terminator.loopHeader && !entering) {
                EmitLoop(block);
                block = Target(blockInfo.terminator.mergeBlock);
                continue;
            }
            entering = false;
            Body(block, false);
            const auto& term = blockInfo.terminator;
            switch (term.kind) {
            case TerminatorKind::Branch: {
                const auto* target = Target(term.trueBlock);
                if (target == nullptr) {
                    Return();
                    return false;
                }
                if (Exit(block, target, stop)) return target == stop;
                PhiCopies(block, target);
                block = target;
                continue;
            }
            case TerminatorKind::ConditionalBranch: {
                const auto* trueBlock = Target(term.trueBlock);
                const auto* falseBlock = Target(term.falseBlock);
                if (trueBlock == nullptr || falseBlock == nullptr) {
                    Return();
                    return false;
                }
                if (trueBlock == falseBlock) {
                    // Both edges lead to the same block (a branch to the next instruction): the
                    // structurizer gives it no merge, and the condition does not matter.
                    if (Exit(block, trueBlock, stop)) return trueBlock == stop;
                    PhiCopies(block, trueBlock);
                    block = trueBlock;
                    continue;
                }
                const auto condition = Condition(blockInfo);
                const auto* merge = term.loopHeader ? nullptr : Target(term.mergeBlock);
                if (merge != nullptr) {
                    state.Line("if (" + condition + ") {");
                    state.Indent();
                    Arm(block, trueBlock, merge);
                    state.Outdent();
                    state.Line("} else {");
                    state.Indent();
                    Arm(block, falseBlock, merge);
                    state.Outdent();
                    state.Line("}");
                    block = merge;
                    continue;
                }
                const bool trueExit = IsExit(trueBlock, stop);
                const bool falseExit = IsExit(falseBlock, stop);
                if (!trueExit && !falseExit) state.Fail("conditional branch of block " + std::to_string(blockInfo.id) + " has no merge and no exit");
                if (trueExit && falseExit) {
                    state.Line("if (" + condition + ") {");
                    state.Indent();
                    const bool trueFalls = trueBlock == stop;
                    Exit(block, trueBlock, stop);
                    state.Outdent();
                    state.Line("} else {");
                    state.Indent();
                    const bool falseFalls = falseBlock == stop;
                    Exit(block, falseBlock, stop);
                    state.Outdent();
                    state.Line("}");
                    return trueFalls || falseFalls;
                }
                const auto* exit = trueExit ? trueBlock : falseBlock;
                const auto* next = trueExit ? falseBlock : trueBlock;
                state.Line(trueExit ? "if (" + condition + ") {" : "if (!(" + condition + ")) {");
                state.Indent();
                Exit(block, exit, stop);
                state.Outdent();
                if (exit == stop) {
                    // An early exit to the region's end: the rest of the region is the other arm.
                    state.Line("} else {");
                    state.Indent();
                    PhiCopies(block, next);
                    Region(next, stop);
                    state.Outdent();
                    state.Line("}");
                    return true;
                }
                state.Line("}");
                PhiCopies(block, next);
                block = next;
                continue;
            }
            case TerminatorKind::Return:
                Return();
                return false;
            default:
                Unstructured(blockInfo);
            }
        }
        return block == stop;
    }

    void Arm(const IrBlock* from, const IrBlock* target, const IrBlock* merge) {
        if (Exit(from, target, merge)) return;
        PhiCopies(from, target);
        Region(target, merge);
    }
};

bool CommunicatesThroughMemory(IrOpcode opcode) {
    const auto buffer = BufferAccessOf(opcode);
    const auto address = AddressOpcodeInfoOf(opcode).access;
    const auto image = ImageOpcodeInfoOf(opcode).access;
    return SharedAccessOf(opcode) != SharedAccess::None || buffer == BufferAccess::Write || buffer == BufferAccess::Atomic || address == AddressAccess::Write || address == AddressAccess::Atomic || image == ImageAccess::Write || image == ImageAccess::Atomic;
}

// s_memtime / s_memrealtime: MSL has no shader or device clock, so both read an emulated clock,
// the 64-bit count of clock reads in the driver's table (MslAbi::ClockCounterWord, low then high
// word). One elected invocation per SIMD group takes the next count and broadcasts it, so the value
// is uniform over the guest wave (over each 32-lane view of a wave64 graphics program), nonzero,
// and increases from read to read across waves, dispatches and draws of the device. It counts
// reads, not time (TechnicalDebt metal-shader-clock). Without the table (atomicLocks 0) the count
// is the invocation's own.
void EmitClock(MslEmitterState& state, const IrValue& instruction) {
    state.usesDriverData = true;
    static constexpr const char* Local = "aps5_clock_reads";
    if (state.locals.find(std::string(" ") + Local + " ") == std::string::npos) state.DeclareLocal("uint", Local, "0u");
    const auto value = state.Temporary("aps5_clk");
    state.Line(std::string(Local) + " += 1u;");
    state.Line("uint2 " + value + " = uint2(" + Local + ", 0u);");
    state.Line("if (aps5_driver.atomicLocks != 0ul) {");
    state.Indent();
    state.Line("if (simd_is_first()) {");
    state.Indent();
    state.Line("device atomic_uint* " + value + "_c = reinterpret_cast<device atomic_uint*>(aps5_driver.atomicLocks) + " + std::to_string(MslAbi::ClockCounterWord) + "u;");
    state.Line(value + ".x = atomic_fetch_add_explicit(&" + value + "_c[0], 1u, memory_order_relaxed) + 1u;");
    state.Line(value + ".y = " + value + ".x == 0u ? atomic_fetch_add_explicit(&" + value + "_c[1], 1u, memory_order_relaxed) + 1u : atomic_load_explicit(&" + value + "_c[1], memory_order_relaxed);");
    state.Outdent();
    state.Line("}");
    state.Line(value + " = simd_broadcast_first(" + value + ");");
    state.Outdent();
    state.Line("}");
    state.Define(instruction, "as_type<ulong>(" + value + ")");
}

void EmitBarrier(MslEmitterState& state, const IrValue& instruction) {
    const auto stage = state.program.Resources().stage;
    // Local and domain programs run per control point / corner in object and mesh functions, inside
    // branches no barrier may sit in: as vertex programs, their barriers order nothing.
    if (stage == IrShaderStage::Vertex || stage == IrShaderStage::Local || stage == IrShaderStage::TessellationEvaluation) {
        // One guest wave per primitive group on the SPIR-V path too: an s_barrier between vertex
        // invocations orders nothing unless they communicate through memory.
        for (const auto* block : state.program.BlockOrder()) {
            for (const auto* other : block->Instructions()) {
                if (CommunicatesThroughMemory(other->Opcode())) state.Fail(instruction, "s_barrier in a vertex program that accesses LDS or GDS or writes memory is not implemented");
            }
        }
        return;
    }
    if (stage == IrShaderStage::TessellationControl) {
        state.Line("threadgroup_barrier(mem_flags::mem_none);");
        return;
    }
    state.Line("threadgroup_barrier(" + state.LdsMemoryFlags() + ");");
}

}

MslBarrierScope MslWaveLdsScope(const MslEmitterState& state) {
    const auto& program = state.program;
    if (program.Resources().stage != IrShaderStage::Compute || state.workgroup == nullptr) return MslBarrierScope::None;
    bool writes = false;
    for (const auto* block : program.BlockOrder()) {
        for (const auto* instruction : block->Instructions()) {
            const auto access = SharedAccessOf(instruction->Opcode());
            writes |= access != SharedAccess::None && access != SharedAccess::Read;
        }
    }
    if (!writes) return MslBarrierScope::None;
    const auto* workgroup = state.workgroup;
    const auto threads = std::max(workgroup->threadsNum[0], 1u) * std::max(workgroup->threadsNum[1], 1u) * std::max(workgroup->threadsNum[2], 1u);
    // As the SPIR-V path: a single-wave workgroup gets threadgroup barriers (a subgroup-scope barrier
    // did not make LDS writes visible on NVIDIA; APS5_WAVE_LDS_SUBGROUP=1 restores it for comparison),
    // a guest wave held by one host SIMD group a SIMD-group barrier.
    static const bool preferSubgroup = std::getenv("APS5_WAVE_LDS_SUBGROUP") != nullptr;
    if (!preferSubgroup && threads <= program.WaveSize()) return MslBarrierScope::Threadgroup;
    if (state.laneCount == 2u || (program.WaveSize() == workgroup->hostSubgroupSize && workgroup->hostSubgroupSize <= 32u)) return MslBarrierScope::Simdgroup;
    return threads <= program.WaveSize() ? MslBarrierScope::Threadgroup : MslBarrierScope::None;
}

void EmitMslFunctionBody(MslEmitterState& state) {
    StructuredEmitter(state).Run();
}

bool EmitMslControlInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::Identity:
    case IrOpcode::Void:
    case IrOpcode::Reference:
    case IrOpcode::ReferenceU32:
    case IrOpcode::ControlNop:
    case IrOpcode::Waitcnt:
    case IrOpcode::Sendmsg:
    case IrOpcode::TtraceData:
    case IrOpcode::InstPrefetch:
    case IrOpcode::GetBufferResource:
    case IrOpcode::GetAddressResource:
    case IrOpcode::GetScratchResource:
    case IrOpcode::GetSrtResource:
        return true;
    case IrOpcode::UndefU1:
    case IrOpcode::UndefU8:
    case IrOpcode::UndefU16:
    case IrOpcode::UndefU32:
    case IrOpcode::UndefU64:
        // Any value will do; the hoisted local already holds zero.
        static_cast<void>(state.Name(instruction));
        return true;
    case IrOpcode::GetShaderBase:
        state.Define(instruction, "0ul");
        return true;
    case IrOpcode::ShaderClock:
    case IrOpcode::RealtimeClock:
        EmitClock(state, instruction);
        return true;
    case IrOpcode::Barrier:
        EmitBarrier(state, instruction);
        return true;
    case IrOpcode::Unreachable:
        state.Line("return;");
        return true;
    default:
        return false;
    }
}

}
