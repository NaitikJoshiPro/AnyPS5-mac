#include "MetalBackend/MslLaneLocal.hpp"
#include "IntermediateRepresentation/IrBlock.hpp"
#include "IntermediateRepresentation/IrMetadata.hpp"
#include "IntermediateRepresentation/IrOpcode.hpp"
#include "SpirvBackend/SpirvAnalysis.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler {

// Wave64 compute at one lane per invocation. The two-half layout (laneCount 2) runs guest lanes l
// and l + 32 of a wave in one host invocation so that wave-wide operations (ballots, votes, lane
// reads) see the whole wave inside one 32-wide SIMD group. A program whose observable results do
// not depend on the other lanes of its wave can run each guest lane as its own host invocation
// instead (half the MSL). This file decides that.
//
// Model. The front end gives every lane its own EXEC bit (GetExec is per lane; vector register
// writes are Select(exec, new, old); every memory access takes its lane's active bit as its last
// argument) and builds the 64-bit masks the guest's scalar code reads with Ballot. At one lane per
// invocation, lane L computes everything itself:
//   (a) no instruction may read another lane's value: ReadFirstLane, ReadLane, DPP, permlane,
//       ds_permute/bpermute, ds_swizzle, ds_append/consume block the layout;
//   (b) Ballot becomes L's own bit (other bits 0). That is exact for every consumer that reads
//       only L's bit of the mask: masks may flow through bitwise and/or/xor/not, selects and phis
//       (never mixing word 0 with word 1), and leave the mask domain only through the own-bit
//       extraction the front end emits (threadBit: Select(LaneId < 32, lo, hi) >> (LaneId & 31)
//       & 1). Any other use (a mask compared with zero for SCC, s_bcnt1, s_ff1, a 64-bit mask, a
//       store of the mask, ...) needs the other lanes' bits and blocks the layout;
//   (c) a wave-mask branch (EXEC/VCC zero or non-zero) is decided by L's own condition instead of
//       a vote. L then deviates from the wave only while L is inactive: for the zero forms when L's
//       condition holds but another lane's does not, for the non-zero forms the other way round.
//       Lane L skips the region R the wave runs meanwhile (the blocks reachable from the wave's
//       successor before the branch's immediate post-dominator M, which must be the successor L
//       takes). This is exact when, evaluated for L with the deviation's fact (L's condition) and
//       everything else symbolic, (1) every effect in R is masked off for L (memory writes and
//       atomics whose active bit is false; no barrier and no other effect), and (2) every value
//       that leaves R (phis at M, values defined before the deviation and used after M) equals the
//       value L carries on its own path. The evaluation is an optimistic fixpoint over R: a value is
//       a constant, "the value x had when L deviated" (x outside R, or x itself when x may be read
//       in R before R redefines it), or unknown; selects with a known condition pick their arm,
//       phis join their incoming values from R, and own-bit extractions of masks evaluate the
//       ballot predicates. By induction over L's path, L's state after M equals its state in the
//       wave, and every deviation is covered by one branch's proof;
//   (d) no LDS hazard: in lockstep a wave sees its own LDS writes without s_barrier, which a wave
//       spread over two SIMD groups would need a threadgroup barrier for (and none may sit in
//       divergent code). Every LDS access must be separated from every earlier LDS write (and every
//       LDS write from every earlier LDS read) by an s_barrier on every path.

namespace {

// Cross-lane reads that stay in the reading lane's 32-lane half of the wave (DPP rows and DPP8,
// v_permlane16/x16, ds_swizzle, ds_bpermute/ds_permute): at one lane per invocation the half is one
// host SIMD group (guest lane = thread index in the threadgroup & 63), so a SIMD-group shuffle
// reads them, from every invocation of the group (a convergence point).
bool HalfLaneOpcode(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::DppMoveU32:
    case IrOpcode::DppUpdateU32:
    case IrOpcode::Permlane16U32:
    case IrOpcode::BpermuteU32:
    case IrOpcode::PermuteU32:
    case IrOpcode::SwizzleU32:
        return true;
    default:
        return false;
    }
}

// Reads of any lane of the wave: through threadgroup memory with a barrier (a synchronization
// point), only when the threadgroup is exactly one wave.
bool WaveLaneOpcode(IrOpcode opcode) {
    return opcode == IrOpcode::ReadFirstLane || opcode == IrOpcode::ReadLane;
}

const IrValue* R(const IrValue* value) {
    return value != nullptr ? value->Resolve() : nullptr;
}

bool IsImmediate(const IrValue* value, std::uint64_t bits) {
    return value != nullptr && value->HasImmediate() && value->ImmediateU64() == bits;
}

// LaneId & 31.
bool IsLaneBitIndex(const IrValue* value) {
    if (value->Opcode() != IrOpcode::BitwiseAnd32) return false;
    const auto* a = R(value->Argument(0));
    const auto* b = R(value->Argument(1));
    return (a->Opcode() == IrOpcode::LaneId && IsImmediate(b, 31u)) || (b->Opcode() == IrOpcode::LaneId && IsImmediate(a, 31u));
}

// LaneId < 32.
bool IsLowHalfTest(const IrValue* value) {
    return value->Opcode() == IrOpcode::ULessThan32 && R(value->Argument(0))->Opcode() == IrOpcode::LaneId && IsImmediate(R(value->Argument(1)), 32u);
}

// Mask kinds of (b). Word0/Word1: a word of a 64-bit lane mask whose bit (L & 31) is exact for the
// lanes L of that word. Own: the word holding L's bit (Select(LaneId < 32, lo, hi)). Shifted: Own
// >> (L & 31), whose bit 0 is L's bit. Bit: Shifted & 1 (a 0/1 value, exact). Vector: a Ballot.
enum class MaskKind : std::uint8_t { None, Vector, Word0, Word1, Own, Shifted, Bit };

bool IsWord(MaskKind kind) {
    return kind == MaskKind::Word0 || kind == MaskKind::Word1 || kind == MaskKind::Own;
}

// The abstract value of (c).
struct Abs {
    enum class Kind : std::uint8_t { Top, Const, Dev, Bottom };
    Kind kind = Kind::Top;
    std::uint64_t bits = 0;
    const IrValue* dev = nullptr;

    static Abs Top() { return {}; }
    static Abs Bottom() { return {Kind::Bottom, 0, nullptr}; }
    static Abs Constant(std::uint64_t bits) { return {Kind::Const, bits, nullptr}; }
    static Abs Bool(bool value) { return Constant(value ? 1u : 0u); }
    static Abs Deviation(const IrValue* value) { return {Kind::Dev, 0, value}; }
    bool operator==(const Abs& other) const { return kind == other.kind && bits == other.bits && dev == other.dev; }
    [[nodiscard]] bool IsBool(bool value) const { return kind == Kind::Const && bits == (value ? 1u : 0u); }
};

Abs Join(const Abs& a, const Abs& b) {
    if (a.kind == Abs::Kind::Top) return b;
    if (b.kind == Abs::Kind::Top) return a;
    return a == b ? a : Abs::Bottom();
}

Abs And(const Abs& a, const Abs& b) {
    if (a.IsBool(false) || b.IsBool(false)) return Abs::Bool(false);
    if (a.IsBool(true)) return b;
    if (b.IsBool(true)) return a;
    if (a.kind == Abs::Kind::Top || b.kind == Abs::Kind::Top) return Abs::Top();
    return a == b ? a : Abs::Bottom();
}

Abs Or(const Abs& a, const Abs& b) {
    if (a.IsBool(true) || b.IsBool(true)) return Abs::Bool(true);
    if (a.IsBool(false)) return b;
    if (b.IsBool(false)) return a;
    if (a.kind == Abs::Kind::Top || b.kind == Abs::Kind::Top) return Abs::Top();
    return a == b ? a : Abs::Bottom();
}

Abs Not(const Abs& a) {
    if (a.kind == Abs::Kind::Const) return Abs::Bool(a.bits == 0u);
    return a.kind == Abs::Kind::Top ? Abs::Top() : Abs::Bottom();
}

Abs Xor(const Abs& a, const Abs& b) {
    if (a.kind == Abs::Kind::Const && b.kind == Abs::Kind::Const) return Abs::Bool((a.bits != 0u) != (b.bits != 0u));
    if (a.kind == Abs::Kind::Top || b.kind == Abs::Kind::Top) return Abs::Top();
    return Abs::Bottom();
}

Abs Choose(const Abs& condition, const Abs& a, const Abs& b) {
    if (condition.IsBool(true)) return a;
    if (condition.IsBool(false)) return b;
    if (condition.kind == Abs::Kind::Top) return Abs::Top();
    return Join(a, b);
}

// Dominators by Cooper, Harvey and Kennedy over nodes 0..count-1 (a node unreachable from `entry`
// gets -1).
std::vector<int> Dominators(int count, int entry, const std::vector<std::vector<int>>& successors, const std::vector<std::vector<int>>& predecessors) {
    std::vector<int> order;
    std::vector<int> number(static_cast<std::size_t>(count), -1);
    std::vector<char> seen(static_cast<std::size_t>(count), 0);
    std::vector<std::pair<int, std::size_t>> stack {{entry, 0u}};
    seen[static_cast<std::size_t>(entry)] = 1;
    while (!stack.empty()) {
        const int node = stack.back().first;
        const auto& out = successors[static_cast<std::size_t>(node)];
        if (stack.back().second < out.size()) {
            const int target = out[stack.back().second++];
            if (!seen[static_cast<std::size_t>(target)]) {
                seen[static_cast<std::size_t>(target)] = 1;
                stack.emplace_back(target, 0u);
            }
            continue;
        }
        number[static_cast<std::size_t>(node)] = static_cast<int>(order.size());
        order.push_back(node);
        stack.pop_back();
    }
    std::vector<int> idom(static_cast<std::size_t>(count), -1);
    idom[static_cast<std::size_t>(entry)] = entry;
    const auto intersect = [&](int a, int b) {
        while (a != b) {
            while (number[static_cast<std::size_t>(a)] < number[static_cast<std::size_t>(b)]) a = idom[static_cast<std::size_t>(a)];
            while (number[static_cast<std::size_t>(b)] < number[static_cast<std::size_t>(a)]) b = idom[static_cast<std::size_t>(b)];
        }
        return a;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const int node = *it;
            if (node == entry) continue;
            int candidate = -1;
            for (const int predecessor : predecessors[static_cast<std::size_t>(node)]) {
                if (number[static_cast<std::size_t>(predecessor)] < 0 || idom[static_cast<std::size_t>(predecessor)] < 0) continue;
                candidate = candidate < 0 ? predecessor : intersect(predecessor, candidate);
            }
            if (candidate >= 0 && idom[static_cast<std::size_t>(node)] != candidate) {
                idom[static_cast<std::size_t>(node)] = candidate;
                changed = true;
            }
        }
    }
    return idom;
}

bool Dominates(const std::vector<int>& idom, int a, int b) {
    if (idom[static_cast<std::size_t>(b)] < 0) return false;
    for (int node = b;; node = idom[static_cast<std::size_t>(node)]) {
        if (node == a) return true;
        if (idom[static_cast<std::size_t>(node)] == node) return false;
    }
}

class Planner {
public:
    Planner(const IrProgram& program, std::uint32_t threads) : program(program), threads(threads), blocks(program.BlockOrder()), infos(program.Metadata().blockInfo) {}

    MslLaneLocalPlan Run() {
        MslLaneLocalPlan plan;
        if (blocks.empty() || infos.size() != blocks.size()) return Blocked(plan, "control flow metadata");
        Index();
        bool waveReads = false;
        for (const auto* block : blocks) {
            for (const auto* instruction : block->Instructions()) {
                const auto opcode = instruction->Opcode();
                if (opcode == IrOpcode::DataAppend || opcode == IrOpcode::DataConsume || opcode == IrOpcode::DataOrderedCount) return Blocked(plan, "cross-lane " + std::string(IrOpcodeName(opcode)));
                if (HalfLaneOpcode(opcode) && threads % 64u != 0u) return Blocked(plan, "cross-lane " + std::string(IrOpcodeName(opcode)) + " in a partial wave");
                if (WaveLaneOpcode(opcode)) {
                    if (!SingleWave()) return Blocked(plan, "cross-lane " + std::string(IrOpcodeName(opcode)) + " outside a one-wave workgroup");
                    waveReads = true;
                }
            }
        }
        if (auto reason = ClassifyMasks(); !reason.empty()) {
            if (!SingleWave()) return Blocked(plan, reason);
            // Exact ballots through threadgroup memory instead of the lane's own bit.
            fullBallots = true;
            mask.clear();
        }
        if (auto reason = LdsHazard(); !reason.empty()) return Blocked(plan, reason);
        if (auto reason = Branches(); !reason.empty()) return Blocked(plan, reason);
        plan.eligible = true;
        plan.laneMaskBranches = laneMaskBranches;
        plan.fullBallots = fullBallots;
        for (const auto& branch : branches) {
            if (branch.voted) plan.votedBranches.insert(infos[static_cast<std::size_t>(branch.block)].id);
        }
        plan.threadgroupExchange = fullBallots || waveReads || !plan.votedBranches.empty();
        return plan;
    }

private:
    const IrProgram& program;
    std::uint32_t threads;
    // A single-wave workgroup that writes LDS: the emitter orders the wave's LDS accesses with
    // threadgroup barriers (MslWaveLdsScope), which must stay out of code a lane may skip.
    bool waveLdsBarriers = false;
    // Ballots read beyond the lane's own bit: exact ballots through threadgroup memory.
    bool fullBallots = false;
    struct Branch {
        int block = 0;
        std::vector<int> region;
        bool voted = false;
    };
    std::vector<Branch> branches;

    // The threadgroup is exactly one wave: wave-wide votes, ballots and lane reads can go through
    // threadgroup memory with threadgroup barriers.
    bool SingleWave() const {
        return threads == 64u;
    }

    // An instruction every invocation of its SIMD group (half reads) or of the threadgroup
    // (synchronization points) must reach together: none may sit in code a lane may skip.
    bool Convergent(const IrValue* instruction) const {
        const auto opcode = instruction->Opcode();
        if (HalfLaneOpcode(opcode) || WaveLaneOpcode(opcode) || opcode == IrOpcode::Barrier) return true;
        if (opcode == IrOpcode::Ballot && fullBallots) return true;
        return waveLdsBarriers && SharedAccessOf(opcode) != SharedAccess::None;
    }
    const std::vector<IrBlock*>& blocks;
    const std::vector<BlockInfo>& infos;
    std::unordered_map<const IrBlock*, int> blockIndex;
    std::unordered_map<std::uint32_t, int> idIndex;
    std::unordered_map<const IrValue*, std::pair<int, int>> position;
    std::vector<std::vector<int>> successors;
    std::vector<std::vector<int>> predecessors;
    std::unordered_map<const IrValue*, MaskKind> mask;
    std::uint32_t laneMaskBranches = 0;

    // " (branch of block <id>, value <id> <opcode>)" for blockers.
    std::string Where(int branch, const IrValue* value) const {
        return " (branch of block " + std::to_string(infos[static_cast<std::size_t>(branch)].id) + ", value " + std::to_string(value->Id()) + " " + std::string(IrOpcodeName(value->Opcode())) + ")";
    }

    static MslLaneLocalPlan Blocked(MslLaneLocalPlan& plan, std::string reason) {
        plan.eligible = false;
        plan.blocker = std::move(reason);
        return plan;
    }

    void Index() {
        const int count = static_cast<int>(blocks.size());
        successors.assign(static_cast<std::size_t>(count), {});
        predecessors.assign(static_cast<std::size_t>(count), {});
        for (int index = 0; index < count; ++index) {
            blockIndex.emplace(blocks[static_cast<std::size_t>(index)], index);
            idIndex.emplace(infos[static_cast<std::size_t>(index)].id, index);
            int order = 0;
            for (const auto* instruction : blocks[static_cast<std::size_t>(index)]->Instructions()) position.emplace(instruction, std::pair<int, int>{index, order++});
        }
        for (int index = 0; index < count; ++index) {
            for (const auto* successor : blocks[static_cast<std::size_t>(index)]->Successors()) {
                const int target = blockIndex.at(successor);
                successors[static_cast<std::size_t>(index)].push_back(target);
                predecessors[static_cast<std::size_t>(target)].push_back(index);
            }
        }
    }

    int BlockOf(const IrValue* value) const {
        const auto found = position.find(value);
        return found == position.end() ? -1 : found->second.first;
    }

    MaskKind KindOf(const IrValue* value) const {
        const auto found = mask.find(R(value));
        return found == mask.end() ? MaskKind::None : found->second;
    }

    // (b): the mask kind an instruction gets from its operands, or a blocker.
    std::optional<MaskKind> Derive(const IrValue* instruction, std::string& reason) const {
        const auto opcode = instruction->Opcode();
        if (opcode == IrOpcode::Ballot) return MaskKind::Vector;
        std::vector<MaskKind> kinds;
        bool any = false;
        for (std::size_t index = 0; index < instruction->ArgumentCount(); ++index) {
            const auto* argument = instruction->Argument(index);
            kinds.push_back(argument != nullptr ? KindOf(argument) : MaskKind::None);
            any |= kinds.back() != MaskKind::None;
        }
        if (!any) return MaskKind::None;
        const auto fail = [&](std::string_view what) -> std::optional<MaskKind> {
            reason = "lane mask used by " + std::string(IrOpcodeName(opcode)) + (what.empty() ? std::string() : " (" + std::string(what) + ")");
            return std::nullopt;
        };
        // One word kind over the mask operands from `first` on (non-mask operands are exact in
        // every bit).
        const auto common = [&](std::size_t first) -> std::optional<MaskKind> {
            MaskKind result = MaskKind::None;
            for (std::size_t index = first; index < kinds.size(); ++index) {
                const auto kind = kinds[index];
                // A 0/1 own bit is an exact value like any other.
                if (kind == MaskKind::None || kind == MaskKind::Bit) continue;
                if (!IsWord(kind)) return std::nullopt;
                if (result != MaskKind::None && result != kind) return std::nullopt;
                result = kind;
            }
            return result;
        };
        switch (opcode) {
        case IrOpcode::CompositeExtractU32x4: {
            if (kinds[0] != MaskKind::Vector) return fail("");
            const auto* component = R(instruction->Argument(1));
            if (IsImmediate(component, 0u)) return MaskKind::Word0;
            if (IsImmediate(component, 1u)) return MaskKind::Word1;
            return fail("ballot words 2-3");
        }
        case IrOpcode::BitwiseAnd32: {
            const auto* a = R(instruction->Argument(0));
            const auto* b = R(instruction->Argument(1));
            if ((kinds[0] == MaskKind::Shifted && kinds[1] == MaskKind::None && IsImmediate(b, 1u)) || (kinds[1] == MaskKind::Shifted && kinds[0] == MaskKind::None && IsImmediate(a, 1u))) return MaskKind::Bit;
            if (auto kind = common(0)) return *kind;
            return fail("mixed words");
        }
        case IrOpcode::BitwiseOr32:
        case IrOpcode::BitwiseXor32:
        case IrOpcode::BitwiseNot32:
            if (auto kind = common(0)) return *kind;
            return fail("mixed words");
        case IrOpcode::Select:
        case IrOpcode::SelectU32: {
            if (kinds[0] != MaskKind::None) return fail("condition");
            if (IsLowHalfTest(R(instruction->Argument(0)))) {
                const bool low = kinds[1] == MaskKind::Word0 || kinds[1] == MaskKind::None;
                const bool high = kinds[2] == MaskKind::Word1 || kinds[2] == MaskKind::None;
                if (low && high) return MaskKind::Own;
                return fail("own word");
            }
            if (auto kind = common(1)) return *kind;
            return fail("mixed words");
        }
        case IrOpcode::Phi:
            if (auto kind = common(0)) return *kind;
            return fail("mixed words");
        case IrOpcode::ShiftRightLogical32:
        case IrOpcode::IShiftRightLogical32:
            if (kinds[0] == MaskKind::Own && kinds[1] == MaskKind::None && IsLaneBitIndex(R(instruction->Argument(1)))) return MaskKind::Shifted;
            return fail("shift");
        case IrOpcode::INotEqual32:
        case IrOpcode::IEqual32:
            if ((kinds[0] == MaskKind::Bit && IsImmediate(R(instruction->Argument(1)), 0u)) || (kinds[1] == MaskKind::Bit && IsImmediate(R(instruction->Argument(0)), 0u))) return MaskKind::None;
            return fail("");
        default:
            // A 0/1 own bit is an exact value: any other use is fine.
            if (std::all_of(kinds.begin(), kinds.end(), [](MaskKind kind) { return kind == MaskKind::None || kind == MaskKind::Bit; })) return MaskKind::None;
            return fail("");
        }
    }

    std::string ClassifyMasks() {
        for (bool changed = true; changed;) {
            changed = false;
            for (const auto* block : blocks) {
                for (const auto* instruction : block->Instructions()) {
                    std::string reason;
                    const auto kind = Derive(instruction, reason);
                    if (!kind) return reason;
                    if (*kind == MaskKind::None) continue;
                    auto [entry, inserted] = mask.emplace(instruction, *kind);
                    if (inserted) {
                        changed = true;
                    } else if (entry->second != *kind) {
                        return "lane mask kind changes at " + std::string(IrOpcodeName(instruction->Opcode()));
                    }
                }
            }
        }
        // A Bit value is only exact as a 0/1 value: phis of it stay exact (None), all fine. A mask
        // word as a branch condition or reference would be a whole-wave test.
        for (const auto& info : infos) {
            if (info.condition != nullptr && KindOf(info.condition) != MaskKind::None) return "lane mask as a branch condition";
            if (info.indirectTarget != nullptr && KindOf(info.indirectTarget) != MaskKind::None) return "lane mask as a branch target";
        }
        return {};
    }

    // (d): a forward "LDS written / read since the last barrier" analysis over the CFG.
    std::string LdsHazard() {
        bool any = false;
        bool writes = false;
        for (const auto* block : blocks) {
            for (const auto* instruction : block->Instructions()) {
                const auto access = SharedAccessOf(instruction->Opcode());
                any |= access != SharedAccess::None;
                writes |= access != SharedAccess::None && access != SharedAccess::Read;
            }
        }
        if (!any) return {};
        // One wave per workgroup: the threadgroup is the wave, and the emitter's wave LDS barriers
        // (threadgroup_barrier between an LDS write and the next access, a read and the next write)
        // give the lockstep order, as on the two-half path.
        if (threads <= 64u) {
            waveLdsBarriers = writes;
            return {};
        }
        const int count = static_cast<int>(blocks.size());
        std::vector<std::uint8_t> out(static_cast<std::size_t>(count), 0);
        std::vector<char> visited(static_cast<std::size_t>(count), 0);
        constexpr std::uint8_t Written = 1u;
        constexpr std::uint8_t Read = 2u;
        for (bool changed = true; changed;) {
            changed = false;
            for (int index = 0; index < count; ++index) {
                std::uint8_t state = 0;
                bool reached = index == 0;
                for (const int predecessor : predecessors[static_cast<std::size_t>(index)]) {
                    if (!visited[static_cast<std::size_t>(predecessor)]) continue;
                    reached = true;
                    state |= out[static_cast<std::size_t>(predecessor)];
                }
                if (!reached) continue;
                const bool first = !visited[static_cast<std::size_t>(index)];
                visited[static_cast<std::size_t>(index)] = 1;
                for (const auto* instruction : blocks[static_cast<std::size_t>(index)]->Instructions()) {
                    if (instruction->Opcode() == IrOpcode::Barrier) {
                        state = 0;
                        continue;
                    }
                    const auto access = SharedAccessOf(instruction->Opcode());
                    if (access == SharedAccess::None) continue;
                    const bool writes = access != SharedAccess::Read;
                    if ((state & Written) != 0u) return writes ? "LDS write after an LDS write without a barrier" : "LDS read after an LDS write without a barrier";
                    if (writes && (state & Read) != 0u) return "LDS write after an LDS read without a barrier";
                    if (writes) state |= Written;
                    if (access != SharedAccess::Write) state |= Read;
                }
                if (first || out[static_cast<std::size_t>(index)] != state) {
                    out[static_cast<std::size_t>(index)] = state;
                    changed = true;
                }
            }
        }
        return {};
    }

    // (c).
    std::string Branches() {
        const int count = static_cast<int>(blocks.size());
        // Post-dominators: the reverse graph from a virtual exit (node `count`).
        std::vector<std::vector<int>> reverseSuccessors(static_cast<std::size_t>(count + 1));
        std::vector<std::vector<int>> reversePredecessors(static_cast<std::size_t>(count + 1));
        for (int index = 0; index < count; ++index) {
            const auto& out = successors[static_cast<std::size_t>(index)];
            if (out.empty()) {
                reverseSuccessors[static_cast<std::size_t>(count)].push_back(index);
                reversePredecessors[static_cast<std::size_t>(index)].push_back(count);
            }
            for (const int target : out) {
                reverseSuccessors[static_cast<std::size_t>(target)].push_back(index);
                reversePredecessors[static_cast<std::size_t>(index)].push_back(target);
            }
        }
        const auto ipdom = Dominators(count + 1, count, reverseSuccessors, reversePredecessors);
        for (int index = 0; index < count; ++index) {
            const auto& info = infos[static_cast<std::size_t>(index)];
            if (info.terminator.kind != TerminatorKind::ConditionalBranch || !IsWaveMaskBranch(info.terminator.condition)) continue;
            if (info.condition == nullptr) return "wave-mask branch without a condition";
            const auto trueBlock = idIndex.find(info.terminator.trueBlock);
            const auto falseBlock = idIndex.find(info.terminator.falseBlock);
            if (trueBlock == idIndex.end() || falseBlock == idIndex.end()) return "wave-mask branch target unknown";
            const auto kind = info.terminator.condition;
            const bool zero = kind == BranchCondition::ExecZero || kind == BranchCondition::VccZero;
            // Zero forms: the wave branches to `true` when the condition holds in every lane, so L
            // (condition true) leaves the wave when another lane's is false. Non-zero forms: `true`
            // when any lane's holds; L (false) leaves when another lane's is true.
            const int off = zero ? trueBlock->second : falseBlock->second;
            const int on = zero ? falseBlock->second : trueBlock->second;
            if (off == on) continue;
            const int merge = ipdom[static_cast<std::size_t>(index)];
            Branch branch;
            branch.block = index;
            std::string reason;
            if (merge < 0 || merge == count) {
                reason = "wave-mask branch without a post-dominator";
            } else if (merge != off) {
                reason = "wave-mask branch whose lane path is not its post-dominator";
            } else {
                reason = Region(index, on, off, R(info.condition), zero, branch.region);
            }
            if (!reason.empty()) {
                // A one-wave threadgroup votes through threadgroup memory instead.
                if (!SingleWave()) return reason;
                branch.voted = true;
            }
            branches.push_back(std::move(branch));
        }
        // Lanes decide a branch alone only when nothing in what they may skip needs the other
        // invocations; a branch that votes is such a point for the branches around it.
        std::unordered_set<int> votedBlocks;
        for (const auto& branch : branches) {
            if (branch.voted) votedBlocks.insert(branch.block);
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (auto& branch : branches) {
                if (branch.voted) continue;
                std::string reason;
                for (const int block : branch.region) {
                    if (votedBlocks.contains(block)) reason = "a voting wave-mask branch in code a lane may skip";
                    for (const auto* instruction : blocks[static_cast<std::size_t>(block)]->Instructions()) {
                        if (reason.empty() && Convergent(instruction)) reason = std::string(IrOpcodeName(instruction->Opcode())) + " in code a lane may skip";
                    }
                    if (!reason.empty()) break;
                }
                if (reason.empty()) continue;
                if (!SingleWave()) return reason;
                branch.voted = true;
                votedBlocks.insert(branch.block);
                changed = true;
            }
        }
        for (const auto& branch : branches) laneMaskBranches += branch.voted ? 0u : 1u;
        return {};
    }

    std::string Region(int branch, int on, int merge, const IrValue* condition, bool fact, std::vector<int>& regionOut) {
        const int count = static_cast<int>(blocks.size());
        constexpr int End = std::numeric_limits<int>::max();
        // R: reachable from `on` before `merge`.
        std::vector<char> inRegion(static_cast<std::size_t>(count), 0);
        std::vector<int> region;
        std::vector<int> stack {on};
        inRegion[static_cast<std::size_t>(on)] = 1;
        while (!stack.empty()) {
            const int node = stack.back();
            stack.pop_back();
            region.push_back(node);
            for (const int target : successors[static_cast<std::size_t>(node)]) {
                if (target == merge || inRegion[static_cast<std::size_t>(target)]) continue;
                inRegion[static_cast<std::size_t>(target)] = 1;
                stack.push_back(target);
            }
        }
        std::sort(region.begin(), region.end());
        regionOut = region;
        const auto inR = [&](const IrValue* value) {
            const int block = BlockOf(value);
            return block >= 0 && inRegion[static_cast<std::size_t>(block)];
        };
        // Dominators of R rooted at `on` (local numbering).
        std::unordered_map<int, int> local;
        for (std::size_t index = 0; index < region.size(); ++index) local.emplace(region[index], static_cast<int>(index));
        std::vector<std::vector<int>> rs(region.size());
        std::vector<std::vector<int>> rp(region.size());
        for (std::size_t index = 0; index < region.size(); ++index) {
            for (const int target : successors[static_cast<std::size_t>(region[index])]) {
                const auto found = local.find(target);
                if (found == local.end()) continue;
                rs[index].push_back(found->second);
                rp[static_cast<std::size_t>(found->second)].push_back(static_cast<int>(index));
            }
        }
        const auto rdom = Dominators(static_cast<int>(region.size()), local.at(on), rs, rp);
        // After the deviation, a use reads the value R itself computed when the definition comes
        // first on every path from `on` (an earlier instruction of the same block, or a dominating
        // block in R); otherwise it may read the value L had when it deviated.
        const auto dominatedUse = [&](const IrValue* definition, int useBlock, int useOrder) {
            const auto [defBlock, defOrder] = position.at(definition);
            if (defBlock == useBlock) return defOrder < useOrder && !(useBlock == on && false);
            return Dominates(rdom, local.at(defBlock), local.at(useBlock));
        };

        // Facts L's own condition gives at the deviation.
        std::unordered_map<const IrValue*, bool> facts;
        std::function<void(const IrValue*, bool)> learn = [&](const IrValue* value, bool truth) {
            value = R(value);
            if (value == nullptr || !facts.emplace(value, truth).second) return;
            switch (value->Opcode()) {
            case IrOpcode::LogicalNot:
                learn(value->Argument(0), !truth);
                break;
            case IrOpcode::LogicalAnd:
                if (truth) {
                    learn(value->Argument(0), true);
                    learn(value->Argument(1), true);
                }
                break;
            case IrOpcode::LogicalOr:
                if (!truth) {
                    learn(value->Argument(0), false);
                    learn(value->Argument(1), false);
                }
                break;
            // Own-bit extraction backwards: a fact on L's bit is a fact on the masks it came from
            // (for a mask word, "L's bit" is vacuous when L is in the other word, so a fact on the
            // own word holds for both words).
            case IrOpcode::INotEqual32:
            case IrOpcode::IEqual32:
                for (std::size_t index = 0; index < 2u; ++index) {
                    if (KindOf(value->Argument(index)) == MaskKind::Bit && IsImmediate(R(value->Argument(1u - index)), 0u)) learn(value->Argument(index), value->Opcode() == IrOpcode::INotEqual32 ? truth : !truth);
                }
                break;
            case IrOpcode::BitwiseAnd32:
                if (KindOf(value) == MaskKind::Bit) {
                    learn(KindOf(value->Argument(0)) == MaskKind::Shifted ? value->Argument(0) : value->Argument(1), truth);
                } else if (IsWord(KindOf(value)) && truth) {
                    for (std::size_t index = 0; index < 2u; ++index) {
                        if (IsWord(KindOf(value->Argument(index)))) learn(value->Argument(index), true);
                    }
                }
                break;
            case IrOpcode::BitwiseOr32:
                if (IsWord(KindOf(value)) && !truth) {
                    for (std::size_t index = 0; index < 2u; ++index) {
                        if (IsWord(KindOf(value->Argument(index)))) learn(value->Argument(index), false);
                    }
                }
                break;
            case IrOpcode::BitwiseNot32:
                if (IsWord(KindOf(value))) learn(value->Argument(0), !truth);
                break;
            case IrOpcode::ShiftRightLogical32:
            case IrOpcode::IShiftRightLogical32:
                if (KindOf(value) == MaskKind::Shifted) learn(value->Argument(0), truth);
                break;
            case IrOpcode::Select:
            case IrOpcode::SelectU32:
                if (KindOf(value) == MaskKind::Own && IsLowHalfTest(R(value->Argument(0)))) {
                    learn(value->Argument(1), truth);
                    learn(value->Argument(2), truth);
                }
                break;
            case IrOpcode::CompositeExtractU32x4:
                if (KindOf(value) != MaskKind::None) learn(value->Argument(0), truth);
                break;
            case IrOpcode::Ballot:
                learn(value->Argument(0), truth);
                break;
            default:
                break;
            }
        };
        learn(condition, fact);

        // Values of R that may be read in R before R redefines them (they hold what L had when it
        // deviated, or R's later definitions).
        std::vector<const IrValue*> regionValues;
        for (const int block : region) {
            for (const auto* instruction : blocks[static_cast<std::size_t>(block)]->Instructions()) regionValues.push_back(instruction);
        }
        std::unordered_set<const IrValue*> preDeviation;
        for (const auto* value : regionValues) {
            for (const auto* user : value->Uses()) {
                const auto* use = R(user);
                const auto found = position.find(use);
                if (found == position.end()) continue;
                const auto [useBlock, useOrder] = found->second;
                if (use->IsPhi()) {
                    for (std::size_t index = 0; index < use->PhiBlockCount(); ++index) {
                        if (R(use->Argument(index)) != value) continue;
                        const int edge = blockIndex.at(use->PhiBlock(index));
                        if (!inRegion[static_cast<std::size_t>(edge)] || !dominatedUse(value, edge, End)) preDeviation.insert(value);
                    }
                } else if (!inRegion[static_cast<std::size_t>(useBlock)] || !dominatedUse(value, useBlock, useOrder)) {
                    preDeviation.insert(value);
                }
            }
        }
        for (const int block : region) {
            const auto* blockCondition = R(infos[static_cast<std::size_t>(block)].condition);
            if (blockCondition != nullptr && inR(blockCondition) && !dominatedUse(blockCondition, block, End)) preDeviation.insert(blockCondition);
        }

        std::unordered_map<const IrValue*, Abs> table;
        std::unordered_map<const IrValue*, Abs> outside;
        std::function<Abs(const IrValue*)> value = [&](const IrValue* v) -> Abs {
            v = R(v);
            if (v == nullptr) return Abs::Bottom();
            if (v->HasImmediate()) return Abs::Constant(v->ImmediateU64());
            if (inR(v)) {
                const auto found = table.find(v);
                return found == table.end() ? Abs::Top() : found->second;
            }
            if (const auto found = facts.find(v); found != facts.end()) return Abs::Bool(found->second);
            if (const auto found = outside.find(v); found != outside.end()) return found->second;
            // A value outside R does not change in R: a constant derived from the facts, or itself.
            outside.emplace(v, Abs::Deviation(v));
            Abs result = Transfer(v, value);
            if (result.kind != Abs::Kind::Const) result = Abs::Deviation(v);
            outside[v] = result;
            return result;
        };
        const auto initial = [&](const IrValue* v) {
            if (!preDeviation.contains(v)) return Abs::Top();
            if (const auto found = facts.find(v); found != facts.end()) return Abs::Bool(found->second);
            return Abs::Deviation(v);
        };
        for (const auto* v : regionValues) table[v] = initial(v);
        for (bool changed = true; changed;) {
            changed = false;
            for (const auto* v : regionValues) {
                Abs next = Abs::Top();
                if (v->IsPhi()) {
                    for (std::size_t index = 0; index < v->PhiBlockCount(); ++index) {
                        const int edge = blockIndex.at(v->PhiBlock(index));
                        // The incoming values of edges taken while L is off: edges from R, and the
                        // branch's own edge into `on`.
                        if (inRegion[static_cast<std::size_t>(edge)] || (edge == branch && blockIndex.at(v->Parent()) == on)) next = Join(next, value(v->Argument(index)));
                    }
                } else {
                    next = Transfer(v, value);
                }
                next = Join(initial(v), next);
                const Abs current = table[v];
                // Monotone descent: Top -> one value -> Bottom.
                const Abs merged = current.kind == Abs::Kind::Top ? next : (next.kind == Abs::Kind::Top || current == next ? current : Abs::Bottom());
                if (!(merged == current)) {
                    table[v] = merged;
                    changed = true;
                }
            }
        }

        // (1) Effects in R are masked off for L.
        for (const auto* v : regionValues) {
            if (!v->MayHaveSideEffects()) continue;
            const auto opcode = v->Opcode();
            switch (opcode) {
            case IrOpcode::Reference:
            case IrOpcode::ReferenceU32:
            case IrOpcode::Waitcnt:
            case IrOpcode::InstPrefetch:
                continue;
            case IrOpcode::Barrier:
                // A convergence point (Branches).
                continue;
            default:
                break;
            }
            const bool memory = BufferAccessOf(opcode) != BufferAccess::None || SharedAccessOf(opcode) != SharedAccess::None || AddressOpcodeInfoOf(opcode).access != AddressAccess::None || ImageOpcodeInfoOf(opcode).access != ImageAccess::None;
            if (!memory || v->ArgumentCount() == 0u) return "effect " + std::string(IrOpcodeName(opcode)) + " in code a lane may skip";
            const auto* active = v->Argument(v->ArgumentCount() - 1u);
            if (active == nullptr || active->Type() != IrType::Bool) return "effect " + std::string(IrOpcodeName(opcode)) + " without an active bit in code a lane may skip";
            if (!value(active).IsBool(false)) return "effect " + std::string(IrOpcodeName(opcode)) + " not masked off in code a lane may skip";
        }
        // (2) Values leaving R equal L's own. Phis at the post-dominator: L arrives on the branch's
        // edge with what the incoming value was when it deviated.
        const auto atDeviation = [&](const IrValue* v) -> Abs {
            v = R(v);
            if (v->HasImmediate()) return Abs::Constant(v->ImmediateU64());
            if (const auto found = facts.find(v); found != facts.end()) return Abs::Bool(found->second);
            if (inR(v)) return Abs::Deviation(v);
            return value(v);
        };
        for (const auto* phi : blocks[static_cast<std::size_t>(merge)]->Instructions()) {
            if (!phi->IsPhi()) continue;
            const IrValue* lanePath = nullptr;
            for (std::size_t index = 0; index < phi->PhiBlockCount(); ++index) {
                if (blockIndex.at(phi->PhiBlock(index)) == branch) lanePath = phi->Argument(index);
            }
            if (lanePath == nullptr) return "post-dominator phi without the branch's edge";
            const Abs expected = atDeviation(lanePath);
            for (std::size_t index = 0; index < phi->PhiBlockCount(); ++index) {
                const int edge = blockIndex.at(phi->PhiBlock(index));
                if (!inRegion[static_cast<std::size_t>(edge)]) continue;
                if (!(value(phi->Argument(index)) == expected)) return "a value leaving code a lane may skip differs for that lane" + Where(branch, phi);
            }
        }
        // Values of R defined before the deviation and read after M (or on M's phis from the
        // branch's edge, covered above when the branch is in R): unchanged for L in R.
        for (const auto* v : preDeviation) {
            bool usedOutside = false;
            for (const auto* user : v->Uses()) {
                const int block = BlockOf(R(user));
                if (block >= 0 && !inRegion[static_cast<std::size_t>(block)] && !(R(user)->IsPhi() && block == merge)) usedOutside = true;
            }
            if (!usedOutside) continue;
            const Abs all = table[v];
            if (!(all == Abs::Deviation(v)) && !(all == atDeviation(v))) return "a value leaving code a lane may skip changes for that lane" + Where(branch, v);
        }
        return {};
    }

    // One instruction's abstract value from its operands' (c).
    Abs Transfer(const IrValue* v, const std::function<Abs(const IrValue*)>& value) const {
        const auto opcode = v->Opcode();
        const auto kind = KindOf(v);
        const auto argument = [&](std::size_t index) { return value(v->Argument(index)); };
        if (kind != MaskKind::None) {
            // Masks are abstracted by L's bit.
            const auto ownBit = [&](std::size_t index) -> Abs {
                const auto* operand = R(v->Argument(index));
                if (IsWord(KindOf(operand))) return value(operand);
                if (operand->HasImmediate()) {
                    if (operand->ImmediateU32() == 0u) return Abs::Bool(false);
                    if (operand->ImmediateU32() == 0xffffffffu) return Abs::Bool(true);
                }
                return Abs::Bottom();
            };
            switch (opcode) {
            case IrOpcode::Ballot:
            case IrOpcode::CompositeExtractU32x4:
                return argument(0);
            case IrOpcode::BitwiseAnd32:
                if (kind == MaskKind::Bit) return KindOf(v->Argument(0)) == MaskKind::Shifted ? argument(0) : argument(1);
                return And(ownBit(0), ownBit(1));
            case IrOpcode::BitwiseOr32: return Or(ownBit(0), ownBit(1));
            case IrOpcode::BitwiseXor32: return Xor(ownBit(0), ownBit(1));
            case IrOpcode::BitwiseNot32: return Not(ownBit(0));
            case IrOpcode::Select:
            case IrOpcode::SelectU32:
                if (IsLowHalfTest(R(v->Argument(0)))) return Join(ownBit(1), ownBit(2));
                return Choose(argument(0), ownBit(1), ownBit(2));
            case IrOpcode::ShiftRightLogical32:
            case IrOpcode::IShiftRightLogical32:
                return argument(0);
            default:
                return Abs::Bottom();
            }
        }
        switch (opcode) {
        case IrOpcode::Select:
        case IrOpcode::SelectU1:
        case IrOpcode::SelectU32:
        case IrOpcode::SelectF32:
            return Choose(argument(0), argument(1), argument(2));
        case IrOpcode::LogicalNot: return Not(argument(0));
        case IrOpcode::LogicalAnd: return And(argument(0), argument(1));
        case IrOpcode::LogicalOr: return Or(argument(0), argument(1));
        case IrOpcode::INotEqual32:
        case IrOpcode::IEqual32:
            // L's own bit as a bool.
            for (std::size_t index = 0; index < 2u; ++index) {
                if (KindOf(v->Argument(index)) != MaskKind::Bit) continue;
                const auto bit = argument(index);
                return opcode == IrOpcode::INotEqual32 ? Choose(bit, Abs::Bool(true), Abs::Bool(false)) : Not(Choose(bit, Abs::Bool(true), Abs::Bool(false)));
            }
            return Abs::Bottom();
        default:
            return Abs::Bottom();
        }
    }
};

}

MslLaneLocalPlan MslPlanWave64LaneLocal(const IrProgram& program, std::uint32_t threads) {
    auto plan = Planner(program, threads).Run();
    if (const char* survey = std::getenv("APS5_MSL_LANE_SURVEY"); survey != nullptr) {
        if (FILE* out = std::fopen(survey, "a")) {
            std::fprintf(out, "[lane-survey] hash=0x%llx threads=%u blocks=%zu eligible=%d branches=%u blocker=%s\n", static_cast<unsigned long long>(program.Resources().shaderHash), threads, program.BlockOrder().size(), plan.eligible ? 1 : 0, plan.laneMaskBranches, plan.blocker.c_str());
            std::fclose(out);
        }
    }
    return plan;
}

}
