#include "MetalBackend/MslEmitterState.hpp"
#include "ControlFlow/include/ControlFlow/ControlFlowGraph.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler {

namespace {

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
        if (blocks.empty()) state.Fail("the program has no blocks");
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
    std::uint32_t nextTemporary = 0;

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

    void Body(const IrBlock* block, bool repeat) {
        if (!repeat && !emitted.insert(block).second) state.Fail("control flow block " + std::to_string(Info(block).id) + " would be emitted twice");
        for (const auto* instruction : block->Instructions()) {
            if (instruction->IsPhi()) continue;
            if (EmitMslControlInstruction(state, *instruction) || EmitMslAluInstruction(state, *instruction) || EmitMslFloat64Instruction(state, *instruction) || EmitMslWaveInstruction(state, *instruction) || EmitMslMemoryInstruction(state, *instruction) || EmitMslBufferInstruction(state, *instruction) || EmitMslImageInstruction(state, *instruction) || EmitMslStageInstruction(state, *instruction)) continue;
            state.Fail(*instruction, "has no MSL emitter");
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
            copies.push_back({state.Name(*instruction), MslType(instruction->Type()), state.Value(incoming)});
        }
        if (copies.size() == 1u) {
            state.Line(copies.front().phi + " = " + copies.front().value + ";");
            return;
        }
        std::vector<std::string> temporaries;
        for (const auto& copy : copies) {
            temporaries.push_back("aps5_t" + std::to_string(nextTemporary++));
            state.Line("const " + copy.type + " " + temporaries.back() + " = " + copy.value + ";");
        }
        for (std::size_t index = 0; index < copies.size(); ++index) state.Line(copies[index].phi + " = " + temporaries[index] + ";");
    }

    std::string Condition(const BlockInfo& blockInfo) {
        if (blockInfo.condition == nullptr) state.Fail("conditional branch of block " + std::to_string(blockInfo.id) + " has no condition");
        const auto value = state.Value(blockInfo.condition);
        if (!IsWaveMaskBranch(blockInfo.terminator.condition)) return value;
        if (state.laneCount != 1u) state.Fail("wave-mask branches of wave64 programs on 32-wide SIMD groups are not implemented in the MSL backend yet");
        const auto kind = blockInfo.terminator.condition;
        if (kind == BranchCondition::ExecNonZero || kind == BranchCondition::VccNonZero) return "simd_any(" + value + ")";
        return "!simd_any(!(" + value + "))";
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
        case TerminatorKind::Branch:
            edge(Target(term.trueBlock));
            return;
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
        default:
            Return();
            return;
        }
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
            default:
                Return();
                return false;
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

}

void EmitMslFunctionBody(MslEmitterState& state) {
    StructuredEmitter(state).Run();
}

bool EmitMslControlInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::Identity:
    case IrOpcode::ControlNop:
    case IrOpcode::Waitcnt:
    case IrOpcode::TtraceData:
    case IrOpcode::InstPrefetch:
    case IrOpcode::GetBufferResource:
    case IrOpcode::GetAddressResource:
    case IrOpcode::GetScratchResource:
    case IrOpcode::GetSrtResource:
        return true;
    case IrOpcode::Unreachable:
        state.Line("return;");
        return true;
    default:
        return false;
    }
}

}
