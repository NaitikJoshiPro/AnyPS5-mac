#include "MetalBackend/MslEmitterState.hpp"
#include "IntermediateRepresentation/IrBlock.hpp"
#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <string>

namespace ShaderRecompiler {

// Work package B3: cross-lane operations (the wave parts of SpirvAluEmitterMath.cpp and
// SpirvSubgroup.cpp): lane ids, readlane/readfirstlane/writelane, DPP16/DPP8, permlane16/x16,
// ds_swizzle, ds_bpermute/ds_permute and WQM masks.
//
// Lanes of a guest wave map to one 32-wide Apple SIMD group: wave32 one lane per invocation, wave64
// compute (laneCount 2) the guest lanes `simd_lane` and `simd_lane + 32` in one invocation. A guest
// lane read is a `simd_shuffle` of the half that holds the lane (bit 5 of the lane picks the half).
// Every shuffle is a statement of its own, emitted where all invocations of the SIMD group run: guest
// branches vote over the whole wave (MslFlowEmitter.cpp), and nothing here sits in a per-lane `if`.
// Metal leaves a shuffle from an invocation that does not exist (a workgroup smaller than the SIMD
// group) undefined, so every read is masked with the lane's EXEC bit or the SIMD group's live mask.

namespace {

std::string U32(std::uint32_t value) {
    return std::to_string(value) + "u";
}

bool OneLanePerInvocationNarrow(const MslEmitterState& state) {
    return state.laneCount == 1u && state.program.WaveSize() > state.hostSubgroupSize;
}

void RequireHostWave(MslEmitterState& state, const IrValue& instruction) {
    if (OneLanePerInvocationNarrow(state) && !MslGraphicsHalfWave(state) && !state.laneLocal) state.Fail(instruction, "a cross-lane read of a wave64 program at one lane per invocation spans two host SIMD groups (work package B3)");
}

// A lane of the guest wave as a statement-level temporary.
std::string Constant(MslEmitterState& state, std::string_view type, std::string_view expression, std::string_view prefix = "aps5_w") {
    const auto name = state.Temporary(prefix);
    state.Line("const " + std::string(type) + " " + name + " = " + std::string(expression) + ";");
    return name;
}

// The value of argument `index` in guest lane `lane` (a uint expression, 0..63; lanes of a wave32
// program must be below 32). Shuffles both halves for laneCount 2 and selects by bit 5.
std::string Shuffle(MslEmitterState& state, const IrValue& instruction, std::size_t index, const std::string& lane) {
    const IrType type = instruction.Argument(index)->Type();
    const bool flag = type == IrType::Bool;
    const auto wrap = [&](const std::string& value) { return flag ? "uint(" + value + ")" : value; };
    const auto valueType = flag ? std::string("uint") : MslType(type);
    std::string result;
    if (state.laneCount == 2u) {
        const auto host = Constant(state, "ushort", "ushort(" + lane + " & 31u)", "aps5_wl");
        const auto low = Constant(state, valueType, "simd_shuffle(" + wrap(state.HalfArg(instruction, index, 0u)) + ", " + host + ")");
        const auto high = Constant(state, valueType, "simd_shuffle(" + wrap(state.HalfArg(instruction, index, 1u)) + ", " + host + ")");
        result = Constant(state, valueType, "((" + lane + ") & 32u) != 0u ? " + high + " : " + low);
    } else if (state.laneLocal) {
        // Wave64 at one lane per invocation: callers read lanes of the reading lane's 32-lane half,
        // which is this SIMD group (WaveRead reads any lane).
        result = Constant(state, valueType, "simd_shuffle(" + wrap(state.Arg(instruction, index)) + ", ushort((" + lane + ") & 31u))");
    } else {
        result = Constant(state, valueType, "simd_shuffle(" + wrap(state.Arg(instruction, index)) + ", ushort(" + lane + "))");
    }
    return flag ? "(" + result + " != 0u)" : result;
}

// The value of argument `index` in any guest lane `lane`: Shuffle, or at one lane per invocation of
// a wave64 program (DECISIONS D62) a read through threadgroup memory.
std::string WaveRead(MslEmitterState& state, const IrValue& instruction, std::size_t index, const std::string& lane) {
    if (!state.laneLocal) return Shuffle(state, instruction, index, lane);
    const IrType type = instruction.Argument(index)->Type();
    if (type != IrType::U32 && type != IrType::F32 && type != IrType::S32) state.Fail(instruction, "reads a lane of a " + MslType(type) + " value through threadgroup memory (only 32-bit values)");
    const auto bits = state.ExchangeRead("as_type<uint>(" + state.Arg(instruction, index) + ")", lane);
    return "as_type<" + MslType(type) + ">(" + bits + ")";
}

// The bit of `lane` in a 64-lane ballot (uint2), false past the wave.
std::string LaneActive(MslEmitterState& state, const std::string& ballot, const std::string& lane) {
    const auto waveSize = state.program.WaveSize();
    const auto word = waveSize == 64u ? "((" + lane + ") >= 32u ? " + ballot + ".y : " + ballot + ".x)" : ballot + ".x";
    return "((((" + word + ") >> ((" + lane + ") & 31u)) & 1u) != 0u && (" + lane + ") < " + U32(waveSize) + ")";
}

// The invocations of this SIMD group that exist (bit per host lane).
std::string LiveMask(MslEmitterState& state) {
    return Constant(state, "uint", "uint(static_cast<simd_vote::vote_t>(simd_ballot(true)))", "aps5_live");
}

struct TargetLane {
    std::string lane;
    std::string valid;
};

// EmitDppTargetLane: the source lane of a DPP16 / DPP8 control for the current guest lane.
TargetLane DppTarget(MslEmitterState& state, const IrValue& instruction, std::uint32_t control) {
    const auto subid = Constant(state, "uint", state.GuestLane(), "aps5_sub");
    const auto make = [&](const std::string& lane, const std::string& valid) {
        return TargetLane{Constant(state, "uint", lane, "aps5_tl"), valid};
    };
    const auto groupPerm = [&](std::uint32_t selectors, std::uint32_t laneBits) {
        const std::uint32_t laneMask = (1u << laneBits) - 1u;
        return make("(" + subid + " & " + U32(~laneMask) + ") | ((" + U32(selectors) + " >> ((" + subid + " & " + U32(laneMask) + ") * " + U32(laneBits) + ")) & " + U32(laneMask) + ")", "true");
    };
    if ((control & DppMoveFlags::Lanes8) != 0u) return groupPerm(control & ~DppMoveFlags::Lanes8, 3u);
    if (control <= 0xffu) return groupPerm(control, 2u);
    const auto row = "(" + subid + " & 0xfffffff0u)";
    const auto inRow = "(" + subid + " & 15u)";
    if (control >= 0x101u && control <= 0x10fu) {
        const auto amount = control & 0xfu;
        return make(row + " | (" + inRow + " + " + U32(amount) + ")", "(" + inRow + " < " + U32(16u - amount) + ")");
    }
    if (control >= 0x111u && control <= 0x11fu) {
        const auto amount = control & 0xfu;
        return make(row + " | (" + inRow + " - " + U32(amount) + ")", "(" + inRow + " >= " + U32(amount) + ")");
    }
    if (control >= 0x121u && control <= 0x12fu) {
        const auto amount = control & 0xfu;
        return make(row + " | (" + inRow + " >= " + U32(amount) + " ? " + inRow + " - " + U32(amount) + " : " + inRow + " + " + U32(16u - amount) + ")", "true");
    }
    if (control == 0x140u) return make(row + " | (15u - " + inRow + ")", "true");
    if (control == 0x141u) return make("(" + subid + " & 0xfffffff8u) | (7u - (" + subid + " & 7u))", "true");
    if (control >= 0x150u && control <= 0x15fu) return make(row + " | " + U32(control & 0xfu), "true");
    if (control >= 0x160u && control <= 0x16fu) return make(subid + " ^ " + U32(control & 0xfu), "true");
    state.Fail(instruction, "DPP control " + std::to_string(control) + " is not supported");
}

// EmitDsSwizzleTargetLane: the ds_swizzle_b32 source lane (within the lane's 32-lane group).
std::string SwizzleTarget(MslEmitterState& state, std::uint32_t control) {
    const auto subid = Constant(state, "uint", state.GuestLane(), "aps5_sub");
    const auto lane = "(" + subid + " & 31u)";
    const auto base = "(" + subid + " & 0xffffffe0u)";
    if ((control & 0xc000u) == 0xc000u) {
        const std::uint32_t mask = control & 0x1fu;
        const std::uint32_t rotate = (control >> 5u) & 0x1fu;
        const std::uint32_t delta = (control & 0x400u) != 0u ? ((32u - rotate) & 0x1fu) : rotate;
        return Constant(state, "uint", base + " | (" + lane + " & " + U32(mask) + ") | (((" + lane + " + " + U32(delta) + ") & 31u) & " + U32((~mask) & 31u) + ")", "aps5_tl");
    }
    if ((control & 0x8000u) != 0u) {
        return Constant(state, "uint", "(" + subid + " & 0xfffffffcu) | ((" + U32(control) + " >> ((" + subid + " & 3u) << 1u)) & 3u)", "aps5_tl");
    }
    return Constant(state, "uint", base + " | (((" + lane + " & " + U32(control & 0x1fu) + ") | " + U32((control >> 5u) & 0x1fu) + ") ^ " + U32((control >> 10u) & 0x1fu) + ")", "aps5_tl");
}

// EmitDsMaskedLaneRead: argument `source` of `target` (which stays in the reading lane's 32-lane
// group, so in the current half), zero unless that lane exists and its `exec` argument is set.
std::string DsMaskedLaneRead(MslEmitterState& state, const IrValue& instruction, std::size_t source, std::size_t exec, const std::string& target) {
    const auto host = Constant(state, "ushort", "ushort(" + target + " & 31u)", "aps5_wl");
    const auto live = LiveMask(state);
    const auto value = Constant(state, "uint", "simd_shuffle(" + state.Arg(instruction, source) + ", " + host + ")");
    const auto active = Constant(state, "uint", "simd_shuffle(uint(" + state.Arg(instruction, exec) + "), " + host + ")");
    return "((" + active + " != 0u && ((" + live + " >> uint(" + host + ")) & 1u) != 0u) ? " + value + " : 0u)";
}

// Half-wave reduction scans (SpirvAluEmitterMath.cpp MatchHalfWaveScan, IR-level matching copied
// as is): a wave64 graphics program at one lane per invocation reads lane 31/63 of a DPP row scan;
// each host SIMD group then reduces its own 32 keys, and the upper half's lane reads the identity
// when the scan's keys are proven zero/identity there (UpperLaneWalk, EntryLaneWalk).
struct WaveReduction {
    IrOpcode opcode;
    std::uint32_t identity;
    const char* reduce;
    bool floating = false;
};

constexpr std::array<WaveReduction, 9> WaveReductions{{
    {IrOpcode::UMin32, 0xffffffffu, "simd_min(uint("},
    {IrOpcode::UMax32, 0u, "simd_max(uint("},
    {IrOpcode::SMin32, 0x7fffffffu, "as_type<uint>(simd_min(as_type<int>("},
    {IrOpcode::SMax32, 0x80000000u, "as_type<uint>(simd_max(as_type<int>("},
    {IrOpcode::IAdd32, 0u, "simd_sum(uint("},
    {IrOpcode::BitwiseAnd32, 0xffffffffu, "simd_and(uint("},
    {IrOpcode::BitwiseOr32, 0u, "simd_or(uint("},
    {IrOpcode::FPMin32, 0x7f800000u, "", true},
    {IrOpcode::FPMax32, 0xff800000u, "", true},
}};

struct HalfWaveScan {
    const WaveReduction* reduction = nullptr;
    const IrValue* source = nullptr;
};

// Upstream #4162 (SpirvAluEmitterMath StoredWindowValue): a relatively addressed vector register
// read (LoadVectorWindow) at an immediate slot stands for the value the same block last stored to
// that slot, so half-wave scans over v_movrel registers still match.
const IrValue* StoredWindowValue(const IrValue* load) {
    const auto* slot = load->Argument(0)->Resolve();
    const auto* block = load->Parent();
    if (slot == nullptr || !slot->HasImmediate() || block == nullptr) return nullptr;
    const auto& instructions = block->Instructions();
    auto at = std::find(instructions.begin(), instructions.end(), load);
    if (at == instructions.end()) return nullptr;
    while (at != instructions.begin()) {
        const auto* previous = *--at;
        if (previous->HasImmediate() || previous->Opcode() != IrOpcode::StoreVectorWindow) continue;
        const auto* stored = previous->Argument(0)->Resolve();
        if (stored == nullptr || !stored->HasImmediate()) return nullptr;
        if (stored->ImmediateU32() == slot->ImmediateU32()) return previous->Argument(1)->Resolve();
    }
    return nullptr;
}

const IrValue* Resolved(const IrValue* value) {
    value = value != nullptr ? value->Resolve() : nullptr;
    while (value != nullptr && !value->HasImmediate() && value->Opcode() == IrOpcode::LoadVectorWindow) {
        const auto* stored = StoredWindowValue(value);
        if (stored == nullptr) break;
        value = stored;
    }
    return value;
}

bool Is(const IrValue* value, IrOpcode opcode) {
    return value != nullptr && !value->HasImmediate() && value->Opcode() == opcode;
}

bool IsU32(const IrValue* value, std::uint32_t expected) {
    const auto* resolved = Resolved(value);
    return resolved != nullptr && resolved->HasImmediate() && resolved->Type() == IrType::U32 && resolved->ImmediateU32() == expected;
}

template<typename TMatch>
bool EitherOrder(const IrValue* value, TMatch&& match) {
    return match(Resolved(value->Argument(0)), Resolved(value->Argument(1))) || match(Resolved(value->Argument(1)), Resolved(value->Argument(0)));
}

template<typename TMatch>
bool EitherOrder(const std::array<const IrValue*, 2>& operands, TMatch&& match) {
    return match(operands[0], operands[1]) || match(operands[1], operands[0]);
}

std::optional<std::array<const IrValue*, 2>> LaneBitWords(const IrValue* bit) {
    const IrValue* masked = nullptr;
    const IrValue* shifted = nullptr;
    if (!Is(bit, IrOpcode::INotEqual32) || !EitherOrder(bit, [&](const IrValue* value, const IrValue* zero) { masked = value; return IsU32(zero, 0u) && Is(value, IrOpcode::BitwiseAnd32); })) return std::nullopt;
    if (!EitherOrder(masked, [&](const IrValue* value, const IrValue* one) { shifted = value; return IsU32(one, 1u) && Is(value, IrOpcode::ShiftRightLogical32); })) return std::nullopt;
    const auto* index = Resolved(shifted->Argument(1));
    if (!Is(index, IrOpcode::BitwiseAnd32) || !EitherOrder(index, [](const IrValue* lane, const IrValue* mask) { return Is(lane, IrOpcode::LaneId) && IsU32(mask, 31u); })) return std::nullopt;
    const auto* word = Resolved(shifted->Argument(0));
    if (!Is(word, IrOpcode::SelectU32)) return std::array<const IrValue*, 2>{word, word};
    const auto* low = Resolved(word->Argument(0));
    if (!Is(low, IrOpcode::ULessThan32) || !Is(Resolved(low->Argument(0)), IrOpcode::LaneId) || !IsU32(low->Argument(1), 32u)) return std::nullopt;
    return std::array<const IrValue*, 2>{Resolved(word->Argument(1)), Resolved(word->Argument(2))};
}

bool AllOnesWord(const IrValue* word) {
    if (IsU32(word, 0xffffffffu)) return true;
    return Is(word, IrOpcode::BitwiseOr32) && EitherOrder(word, [](const IrValue* inverted, const IrValue* other) {
        return AllOnesWord(inverted) || (Is(inverted, IrOpcode::BitwiseNot32) && Resolved(inverted->Argument(0)) == other);
    });
}

bool AllLanesBit(const IrValue* bit) {
    // Upstream 859dcae2a: a saved EXEC set to every lane (`!x || x`) is every lane too.
    bit = Resolved(bit);
    if (Is(bit, IrOpcode::LogicalOr) && EitherOrder(bit, [](const IrValue* inverted, const IrValue* other) { return Is(inverted, IrOpcode::LogicalNot) && Resolved(inverted->Argument(0)) == other; })) return true;
    const auto words = LaneBitWords(bit);
    return words && AllOnesWord((*words)[0]) && AllOnesWord((*words)[1]);
}

// Upstream 34e7c309a (SpirvAluEmitterMath): the not-helper test (IEqual32 of HelperInvocation and
// 0) counts as a lane bit, and a conjunction with a false or helper conjunct is zero before any
// operand is walked (the walk budget is spent on loop phis otherwise).
bool HelperTest(const IrValue* predicate) {
    return Is(predicate, IrOpcode::IEqual32) && EitherOrder(predicate, [](const IrValue* builtin, const IrValue* zero) { return IsU32(zero, 0u) && Is(builtin, IrOpcode::GetBuiltin) && IsU32(builtin->Argument(0), static_cast<std::uint32_t>(StageInputKind::HelperInvocation)) && IsU32(builtin->Argument(1), 0u); });
}

bool HelperOrFalseConjunct(const IrValue* bit) {
    bit = Resolved(bit);
    if (bit == nullptr) return false;
    if (bit->HasImmediate()) return bit->Type() == IrType::Bool && !bit->ImmediateBool();
    if (Is(bit, IrOpcode::LogicalAnd)) return HelperOrFalseConjunct(bit->Argument(0)) || HelperOrFalseConjunct(bit->Argument(1));
    return HelperTest(bit);
}

bool EveryInvocationEnters(IrShaderStage stage) {
    return stage == IrShaderStage::Vertex || stage == IrShaderStage::Local || stage == IrShaderStage::TessellationControl || stage == IrShaderStage::TessellationEvaluation;
}

class EntryLaneWalk {
public:
    explicit EntryLaneWalk(IrShaderStage stage) : everyInvocationEnters(EveryInvocationEnters(stage)) {}

    bool ZeroBit(const IrValue* bit) {
        bit = Resolved(bit);
        if (bit == nullptr || !Spend()) return false;
        if (bit->HasImmediate()) return bit->Type() == IrType::Bool && !bit->ImmediateBool();
        if (Is(bit, IrOpcode::LogicalAnd)) return HelperOrFalseConjunct(bit) || ZeroBit(bit->Argument(0)) || ZeroBit(bit->Argument(1));
        if (Is(bit, IrOpcode::SelectU1)) return ZeroBit(bit->Argument(1)) && ZeroBit(bit->Argument(2));
        if (Is(bit, IrOpcode::Phi)) return Incoming(bit, [this](const IrValue* incoming) { return ZeroBit(incoming); });
        if (Is(bit, IrOpcode::IEqual32)) return HelperTest(bit);
        const auto words = LaneBitWords(bit);
        return words && Word((*words)[0], 0u) && Word((*words)[1], 1u);
    }

private:
    bool Spend() {
        if (budget == 0u) return false;
        --budget;
        return true;
    }

    bool NotHelper(const IrValue* predicate) {
        predicate = Resolved(predicate);
        if (predicate == nullptr || !Spend()) return false;
        if (predicate->HasImmediate()) return everyInvocationEnters && predicate->Type() == IrType::Bool && predicate->ImmediateBool();
        switch (predicate->Opcode()) {
            case IrOpcode::LogicalAnd: return NotHelper(predicate->Argument(0)) || NotHelper(predicate->Argument(1));
            case IrOpcode::IEqual32: return HelperTest(predicate);
            default: return ZeroBit(predicate);
        }
    }

    bool Word(const IrValue* word, std::uint32_t half) {
        word = Resolved(word);
        if (word == nullptr || !Spend()) return false;
        if (word->HasImmediate()) return IsU32(word, 0u);
        switch (word->Opcode()) {
            case IrOpcode::CompositeExtractU32x4: {
                const auto* ballot = Resolved(word->Argument(0));
                return IsU32(word->Argument(1), half) && Is(ballot, IrOpcode::Ballot) && NotHelper(ballot->Argument(0));
            }
            case IrOpcode::CompositeExtractU64: return IsU32(word->Argument(1), half) && Word64(word->Argument(0), half);
            case IrOpcode::BitwiseAnd32: return Word(word->Argument(0), half) || Word(word->Argument(1), half);
            case IrOpcode::BitwiseOr32: return Word(word->Argument(0), half) && Word(word->Argument(1), half);
            case IrOpcode::Phi: return Incoming(word, [this, half](const IrValue* incoming) { return Word(incoming, half); });
            default: return false;
        }
    }

    bool Word64(const IrValue* mask, std::uint32_t half) {
        mask = Resolved(mask);
        if (mask == nullptr || !Spend()) return false;
        if (mask->HasImmediate()) return mask->Type() == IrType::U64 && static_cast<std::uint32_t>(mask->ImmediateU64() >> (32u * half)) == 0u;
        switch (mask->Opcode()) {
            case IrOpcode::CompositeConstructU64: return Word(mask->Argument(half), half);
            case IrOpcode::Phi: return Incoming(mask, [this, half](const IrValue* incoming) { return Word64(incoming, half); });
            default: return false;
        }
    }

    template<typename TCheck>
    bool Incoming(const IrValue* phi, TCheck&& check) {
        if (!phis.insert(phi).second) return true;
        bool result = phi->ArgumentCount() != 0u;
        for (std::size_t index = 0; result && index < phi->ArgumentCount(); ++index) result = check(phi->Argument(index));
        phis.erase(phi);
        return result;
    }

    bool everyInvocationEnters;
    std::unordered_set<const IrValue*> phis;
    // Upstream 99500f9d4: 2048 steps (was 512).
    std::uint32_t budget = 2048u;
};

bool EveryLaneBit(const IrValue* bit, bool constantTrue) {
    const auto* resolved = Resolved(bit);
    if (constantTrue && resolved != nullptr && resolved->HasImmediate()) return resolved->Type() == IrType::Bool && resolved->ImmediateBool();
    return AllLanesBit(resolved);
}

bool WritesEveryLane(const IrValue& update) {
    const auto flags = update.Flags<DppMoveFlags>();
    return flags.boundControl && flags.rowMask == 0xfu && flags.bankMask == 0xfu && EveryLaneBit(update.Argument(2), true);
}

bool KeepsNoOldValue(const IrValue& permlane) {
    const auto flags = permlane.Flags<PermlaneFlags>();
    return flags.fetchInactive || flags.boundControl || EveryLaneBit(permlane.Argument(3), true);
}

bool UpperLaneTest(const IrValue* bit) {
    const auto* compare = Resolved(bit);
    return Is(compare, IrOpcode::UGreaterThanEqual32) && Is(Resolved(compare->Argument(0)), IrOpcode::LaneId) && IsU32(compare->Argument(1), 32u);
}

class UpperLaneWalk {
public:
    explicit UpperLaneWalk(IrShaderStage stage) : enters(EveryInvocationEnters(stage)), laneIndexed(enters && stage != IrShaderStage::TessellationControl) {}

    bool ZeroValue(const IrValue* value) {
        value = Resolved(value);
        if (!enters || value == nullptr || !spend()) return false;
        if (value->HasImmediate()) return IsU32(value, 0u);
        if (zeroValues.contains(value)) return true;
        bool zero = false;
        switch (value->Opcode()) {
            case IrOpcode::SelectU32: zero = ZeroValue(value->Argument(2)) && (zeroBit(value->Argument(0)) || ZeroValue(value->Argument(1))); break;
            case IrOpcode::BitwiseOr32: zero = ZeroValue(value->Argument(0)) && ZeroValue(value->Argument(1)); break;
            case IrOpcode::DppMoveU32: zero = ZeroValue(value->Argument(0)); break;
            case IrOpcode::DppUpdateU32: zero = ZeroValue(value->Argument(0)) && (WritesEveryLane(*value) || ZeroValue(value->Argument(1))); break;
            case IrOpcode::Permlane16U32: zero = ZeroValue(value->Argument(0)) && (KeepsNoOldValue(*value) || ZeroValue(value->Argument(4))); break;
            case IrOpcode::Phi: zero = incoming(value, [this](const IrValue* source) { return ZeroValue(source); }); break;
            default: break;
        }
        if (zero && phis.empty()) zeroValues.insert(value);
        return zero;
    }

private:
    bool spend() {
        if (budget == 0u) return false;
        --budget;
        return true;
    }

    bool zeroBit(const IrValue* bit) {
        bit = Resolved(bit);
        if (bit == nullptr || !spend()) return false;
        if (bit->HasImmediate()) return bit->Type() == IrType::Bool && !bit->ImmediateBool();
        switch (bit->Opcode()) {
            case IrOpcode::LogicalAnd: return zeroBit(bit->Argument(0)) || zeroBit(bit->Argument(1));
            case IrOpcode::SelectU1: return zeroBit(bit->Argument(2)) && (zeroBit(bit->Argument(0)) || zeroBit(bit->Argument(1)));
            case IrOpcode::Phi: return incoming(bit, [this](const IrValue* source) { return zeroBit(source); });
            case IrOpcode::INotEqual32: return EitherOrder(bit, [this](const IrValue* value, const IrValue* zero) { return IsU32(zero, 0u) && ZeroValue(value); });
            case IrOpcode::ULessThan32: return belowUpperLanes(bit->Argument(0), bit->Argument(1));
            case IrOpcode::UGreaterThan32: return belowUpperLanes(bit->Argument(1), bit->Argument(0));
            default: return false;
        }
    }

    bool belowUpperLanes(const IrValue* index, const IrValue* bound) {
        const auto* limit = Resolved(bound);
        return limit != nullptr && limit->HasImmediate() && limit->Type() == IrType::U32 && limit->ImmediateU32() <= 32u && upperLaneIndex(index);
    }

    bool upperLaneIndex(const IrValue* index) {
        index = Resolved(index);
        if (!laneIndexed || index == nullptr || !spend()) return false;
        if (Is(index, IrOpcode::LaneId)) return true;
        if (Is(index, IrOpcode::BitCount32)) {
            const auto* mask = Resolved(index->Argument(0));
            return Is(mask, IrOpcode::SelectU32) && IsU32(mask->Argument(1), 0xffffffffu) && UpperLaneTest(mask->Argument(0));
        }
        return Is(index, IrOpcode::IAdd32) && EitherOrder(index, [this](const IrValue* counted, const IrValue* added) { return Is(added, IrOpcode::BitCount32) && upperLaneIndex(counted); });
    }

    template<typename TCheck>
    bool incoming(const IrValue* phi, TCheck&& check) {
        if (!phis.insert(phi).second) return true;
        bool result = phi->ArgumentCount() != 0u;
        for (std::size_t index = 0; result && index < phi->ArgumentCount(); ++index) result = check(phi->Argument(index));
        phis.erase(phi);
        return result;
    }

    bool enters;
    bool laneIndexed;
    std::unordered_set<const IrValue*> zeroValues;
    std::unordered_set<const IrValue*> phis;
    std::uint32_t budget = 1024u;
};

bool RowShiftFlags(const IrValue* value, std::uint32_t control, bool zeroFill) {
    const auto flags = value->Flags<DppMoveFlags>();
    return flags.control == control && flags.rowMask == 0xfu && flags.bankMask == 0xfu && (!flags.boundControl || zeroFill) && !flags.fetchInactive;
}

const IrValue* Unflushed(const IrValue* value) {
    if (!Is(value, IrOpcode::SelectU32)) return value;
    const auto* kept = Resolved(value->Argument(2));
    const auto masked = [kept](const IrValue* candidate, std::uint32_t mask) {
        return Is(candidate, IrOpcode::BitwiseAnd32) && EitherOrder(candidate, [&](const IrValue* source, const IrValue* constant) { return source == kept && IsU32(constant, mask); });
    };
    const auto* condition = Resolved(value->Argument(0));
    const bool exponentZero = Is(condition, IrOpcode::IEqual32) && EitherOrder(condition, [&](const IrValue* exponent, const IrValue* zero) { return IsU32(zero, 0u) && masked(exponent, 0x7f800000u); });
    return exponentZero && masked(Resolved(value->Argument(1)), 0x80000000u) ? kept : value;
}

std::optional<std::array<const IrValue*, 2>> ReductionOperands(const IrValue* combined, const WaveReduction& reduction) {
    if (reduction.floating) {
        combined = Unflushed(combined);
        if (!Is(combined, IrOpcode::BitCastU32F32)) return std::nullopt;
        combined = Resolved(combined->Argument(0));
    }
    if (!Is(combined, reduction.opcode)) return std::nullopt;
    std::array<const IrValue*, 2> operands{Resolved(combined->Argument(0)), Resolved(combined->Argument(1))};
    if (!reduction.floating) return operands;
    for (auto& operand : operands) {
        if (!Is(operand, IrOpcode::BitCastF32U32)) return std::nullopt;
        operand = Unflushed(Resolved(operand->Argument(0)));
    }
    return operands;
}

const IrValue* RowScanStepInput(const IrValue* step, const WaveReduction& reduction, std::uint32_t control, bool narrow) {
    const bool zeroFill = narrow && reduction.identity == 0u && !reduction.floating;
    if (!Is(step, IrOpcode::DppUpdateU32) || !RowShiftFlags(step, control, zeroFill) || !EveryLaneBit(step->Argument(2), narrow)) return nullptr;
    const auto* previous = zeroFill && WritesEveryLane(*step) ? nullptr : Resolved(step->Argument(1));
    const IrValue* input = nullptr;
    const auto operands = ReductionOperands(Resolved(step->Argument(0)), reduction);
    const bool matched = operands && EitherOrder(*operands, [&](const IrValue* moved, const IrValue* own) {
        input = own;
        return (previous == nullptr || own == previous) && Is(moved, IrOpcode::DppMoveU32) && RowShiftFlags(moved, control, zeroFill) && Resolved(moved->Argument(0)) == own && EveryLaneBit(moved->Argument(1), narrow);
    });
    return matched ? input : nullptr;
}

const IrValue* Unmasked(const IrValue* value, bool narrow) {
    while (Is(value, IrOpcode::SelectU32) && EveryLaneBit(value->Argument(0), narrow)) value = Resolved(value->Argument(1));
    return value;
}

bool CrossesRowsOf(const IrValue* value, const IrValue* scan, bool narrow) {
    const auto* permlane = Unmasked(value, narrow);
    if (!Is(permlane, IrOpcode::Permlane16U32)) return false;
    const auto flags = permlane->Flags<PermlaneFlags>();
    return flags.x16 && !flags.fetchInactive && Resolved(permlane->Argument(0)) == scan && IsU32(permlane->Argument(1), 0xffffffffu) && IsU32(permlane->Argument(2), 0xffffffffu) && EveryLaneBit(permlane->Argument(3), narrow);
}

std::optional<HalfWaveScan> MatchHalfWaveScan(const IrValue* value, IrShaderStage stage, bool narrow) {
    const auto* combined = Unmasked(Resolved(value), narrow);
    std::optional<std::array<const IrValue*, 2>> operands;
    const auto reduction = std::find_if(WaveReductions.begin(), WaveReductions.end(), [&](const WaveReduction& candidate) { return (operands = ReductionOperands(combined, candidate)).has_value(); });
    const IrValue* scan = nullptr;
    if (reduction == WaveReductions.end() || !EitherOrder(*operands, [&](const IrValue* own, const IrValue* crossed) { scan = own; return CrossesRowsOf(crossed, own, narrow); })) return std::nullopt;
    for (const auto control : {0x118u, 0x114u, 0x112u, 0x111u}) {
        scan = RowScanStepInput(scan, *reduction, control, narrow);
        if (scan == nullptr) return std::nullopt;
    }
    const auto* source = Unmasked(scan, narrow);
    const bool entryKeys = Is(source, IrOpcode::SelectU32) && IsU32(source->Argument(2), reduction->identity) && EntryLaneWalk{stage}.ZeroBit(source->Argument(0));
    const bool upperKeys = narrow && reduction->identity == 0u && !reduction->floating && UpperLaneWalk{stage}.ZeroValue(source);
    if (!entryKeys && !upperKeys) return std::nullopt;
    return HalfWaveScan{&*reduction, source};
}

// Upstream 859dcae2a (SpirvAluEmitterMath MatchRowScan / EmitRowReduction): a row scan (DPP row_shr
// 1, 2, 4 and 8 over keys holding the identity on lanes the wave did not enter with, or on lanes an
// all-active ballot masks off) read at a row's last lane would shuffle from lanes that already left
// the surrounding loop, inactive on the host with undefined values; the read becomes a SIMD
// reduction over the active invocations of that row.
bool ActiveMaskedWord(const IrValue* word, std::uint32_t depth = 0u) {
    word = Resolved(word);
    if (word == nullptr || depth > 8u) return false;
    if (Is(word, IrOpcode::CompositeExtractU32x4)) {
        const auto* ballot = Resolved(word->Argument(0));
        const auto* active = ballot != nullptr && Is(ballot, IrOpcode::Ballot) ? Resolved(ballot->Argument(0)) : nullptr;
        return active != nullptr && active->HasImmediate() && active->Type() == IrType::Bool && active->ImmediateBool() && IsU32(word->Argument(1), 0u);
    }
    return Is(word, IrOpcode::BitwiseAnd32) && (ActiveMaskedWord(word->Argument(0), depth + 1u) || ActiveMaskedWord(word->Argument(1), depth + 1u));
}

bool ActiveMaskedBit(const IrValue* bit) {
    const auto words = LaneBitWords(Resolved(bit));
    return words && ActiveMaskedWord((*words)[0]) && ActiveMaskedWord((*words)[1]);
}

std::optional<HalfWaveScan> MatchRowScan(const IrValue* value, IrShaderStage stage, bool narrow) {
    const auto* top = Unmasked(Resolved(value), narrow);
    for (const auto& reduction : WaveReductions) {
        if (reduction.floating) continue;
        const IrValue* scan = top;
        for (const auto control : {0x118u, 0x114u, 0x112u, 0x111u}) {
            scan = RowScanStepInput(scan, reduction, control, narrow);
            if (scan == nullptr) break;
        }
        if (scan == nullptr) continue;
        const auto* source = Unmasked(scan, narrow);
        if (Is(source, IrOpcode::SelectU32) && IsU32(source->Argument(2), reduction.identity) && (ActiveMaskedBit(source->Argument(0)) || EntryLaneWalk{stage}.ZeroBit(source->Argument(0)))) return HalfWaveScan{&reduction, source};
    }
    return std::nullopt;
}

// The row reduction of a v_readlane_b32 at immediate lane `lane` of a graphics program whose wave
// one host SIMD group holds at one lane per invocation, or nullopt.
std::optional<std::string> RowReduction(MslEmitterState& state, const IrValue& instruction, std::uint32_t lane) {
    if ((lane & 15u) != 15u || state.laneCount != 1u || state.laneLocal || state.hostSubgroupSize == 1u || state.hostSubgroupSize != state.program.WaveSize()) return std::nullopt;
    const auto stage = state.program.Resources().stage;
    auto scan = MatchRowScan(instruction.Argument(0u), stage, false);
    if (!scan) scan = MatchRowScan(instruction.Argument(0u), stage, true);
    if (!scan) return std::nullopt;
    const auto keys = "((" + state.GuestLane() + " >> 4u) == " + U32(lane >> 4u) + " ? " + state.Value(scan->source) + " : " + MslHex(scan->reduction->identity) + "u)";
    std::string reduce = scan->reduction->reduce;
    const bool signedReduce = reduce.find("as_type<int>") != std::string::npos;
    return reduce + keys + (signedReduce ? ")))" : "))");
}

void Ballot(MslEmitterState& state, const IrValue& instruction) {
    // Both guest lane halves for wave64 on 32-wide SIMD groups (core helper).
    const auto type = MslType(instruction.Type());
    const auto bits = state.WaveBallot(instruction.Argument(0u));
    if (type == "uint2") state.Define(instruction, bits);
    else if (type == "uint4") state.Define(instruction, "uint4(" + bits + ", 0u, 0u)");
    else if (type == "ulong") state.Define(instruction, "as_type<ulong>(" + bits + ")");
    else state.Fail(instruction, "has an unexpected result type " + type);
}

void ReadFirstLane(MslEmitterState& state, const IrValue& instruction) {
    RequireHostWave(state, instruction);
    const auto ballot = Constant(state, "uint2", state.laneLocal ? state.ExchangeBallot(instruction.Argument(1u)) : state.WaveBallot(instruction.Argument(1u)), "aps5_b");
    const auto first = Constant(state, "uint", ballot + ".x != 0u ? ctz(" + ballot + ".x) : (" + ballot + ".y != 0u ? 32u + ctz(" + ballot + ".y) : 0u)", "aps5_fl");
    state.Define(instruction, WaveRead(state, instruction, 0u, first));
}

void ReadLane(MslEmitterState& state, const IrValue& instruction) {
    RequireHostWave(state, instruction);
    if (MslGraphicsHalfWave(state)) {
        const IrValue* selector = instruction.Argument(1u)->Resolve();
        const auto stage = state.program.Resources().stage;
        if (!selector->HasImmediate()) {
            // A lane chosen at run time (waterfall loops: s_ff1 of an EXEC copy). The half-wave
            // view's EXEC masks and ballots hold only this SIMD group's lanes (low word), so lanes
            // picked from them are below 32; a lane past 31 reads 0, as a shuffle past the host
            // subgroup on the SPIR-V path (TechnicalDebt metal-b6-half-wave).
            const auto lane = Constant(state, "uint", "(" + state.Arg(instruction, 1u) + ") & 63u", "aps5_rl");
            const auto value = Shuffle(state, instruction, 0u, "(" + lane + " & 31u)");
            // One-lane waves: any lane reads the invocation's own value (SpirvSubgroup EmitSubgroupShuffle).
            if (state.hostSubgroupSize == 1u) state.Define(instruction, value);
            else state.Define(instruction, "(" + lane + " < 32u ? " + value + " : " + MslType(instruction.Type()) + "(0))");
            return;
        }
        const auto lane = selector->ImmediateU32() & 63u;
        if ((lane & 31u) == 31u) {
            // As EmitHalfWaveReduction: the scan laid out for the wave first, then for 32-lane hosts.
            auto scan = MatchHalfWaveScan(instruction.Argument(0u), stage, false);
            if (!scan) scan = MatchHalfWaveScan(instruction.Argument(0u), stage, true);
            if (scan) {
                if (lane >= 32u) {
                    state.Define(instruction, MslHex(scan->reduction->identity) + "u");
                    return;
                }
                if (!scan->reduction->floating) {
                    std::string reduce = scan->reduction->reduce;
                    const bool signedReduce = reduce.find("as_type<int>") != std::string::npos;
                    state.Define(instruction, reduce + state.Value(scan->source) + (signedReduce ? ")))" : "))"));
                    return;
                }
            }
        }
        if (lane >= 32u && UpperLaneWalk{stage}.ZeroValue(instruction.Argument(0u))) {
            state.Define(instruction, "0u");
            return;
        }
        if (lane >= 32u && state.hostSubgroupSize != 1u) state.Fail(instruction, "v_readlane_b32 of lane " + std::to_string(lane) + " is outside the 32-lane host SIMD group that runs this wave64 graphics program at one lane per invocation (work package B6)");
    } else if (const IrValue* selector = instruction.Argument(1u)->Resolve(); selector->HasImmediate() && state.program.Resources().stage != IrShaderStage::Compute) {
        if (const auto reduced = RowReduction(state, instruction, selector->ImmediateU32() & (state.program.WaveSize() - 1u))) {
            state.Define(instruction, *reduced);
            return;
        }
    }
    const auto lane = Constant(state, "uint", "(" + state.Arg(instruction, 1u) + ") & " + U32(state.program.WaveSize() - 1u), "aps5_rl");
    state.Define(instruction, WaveRead(state, instruction, 0u, lane));
}

void WriteLane(MslEmitterState& state, const IrValue& instruction) {
    const auto lane = "((" + state.Arg(instruction, 1u) + ") & " + U32(state.program.WaveSize() - 1u) + ")";
    state.Define(instruction, "(" + state.GuestLane() + " == " + lane + ") ? " + state.Arg(instruction, 0u) + " : " + state.Arg(instruction, 2u));
}

void DppMove(MslEmitterState& state, const IrValue& instruction) {
    RequireHostWave(state, instruction);
    const auto flags = instruction.Flags<DppMoveFlags>();
    const auto target = DppTarget(state, instruction, flags.control);
    const auto shuffled = Shuffle(state, instruction, 0u, target.lane);
    if (flags.fetchInactive) {
        state.Define(instruction, "(" + target.valid + ") ? " + shuffled + " : 0u");
        return;
    }
    const auto ballot = Constant(state, "uint2", state.HalfBallot(instruction.Argument(1u)), "aps5_b");
    state.Define(instruction, "((" + target.valid + ") && " + LaneActive(state, ballot, target.lane) + ") ? " + shuffled + " : 0u");
}

// EmitDppWriteCondition + select: the moved value where the row/bank masks, the bound control and
// the source lane allow the write, else the old value.
void DppUpdate(MslEmitterState& state, const IrValue& instruction) {
    const auto flags = instruction.Flags<DppMoveFlags>();
    const auto lane = Constant(state, "uint", state.GuestLane(), "aps5_sub");
    std::string writable = "((" + U32(flags.bankMask) + " >> ((" + lane + " >> 2u) & 3u)) & 1u) != 0u && ((" + U32(flags.rowMask) + " >> ((" + lane + " >> 4u) & 3u)) & 1u) != 0u";
    if (!flags.boundControl) {
        const auto target = DppTarget(state, instruction, flags.control);
        writable += " && (" + target.valid + ")";
        if (!flags.fetchInactive && (flags.control & DppMoveFlags::Lanes8) == 0u) {
            RequireHostWave(state, instruction);
            const auto ballot = Constant(state, "uint2", state.HalfBallot(instruction.Argument(2u)), "aps5_b");
            writable += " && " + LaneActive(state, ballot, target.lane);
        }
    }
    state.Define(instruction, "(" + state.Arg(instruction, 2u) + " && " + writable + ") ? " + state.Arg(instruction, 0u) + " : " + state.Arg(instruction, 1u));
}

// v_permlane16_b32 / v_permlanex16_b32: each lane reads a lane of its own (or, x16, the other) row
// of 16 selected by a nibble of the two selector words.
void Permlane16(MslEmitterState& state, const IrValue& instruction) {
    RequireHostWave(state, instruction);
    const auto flags = instruction.Flags<PermlaneFlags>();
    const auto subid = Constant(state, "uint", state.GuestLane(), "aps5_sub");
    const auto row = "(" + subid + " & 0xfffffff0u)" + std::string(flags.x16 ? " ^ 16u" : "");
    const auto inRow = "(" + subid + " & 15u)";
    const auto selector = "(" + inRow + " >= 8u ? " + state.Arg(instruction, 2u) + " : " + state.Arg(instruction, 1u) + ")";
    const auto target = Constant(state, "uint", "(" + row + ") | ((" + selector + " >> ((" + inRow + " & 7u) << 2u)) & 15u)", "aps5_tl");
    const auto shuffled = Shuffle(state, instruction, 0u, target);
    if (flags.fetchInactive) {
        state.Define(instruction, shuffled);
        return;
    }
    const auto ballot = Constant(state, "uint2", state.HalfBallot(instruction.Argument(3u)), "aps5_b");
    state.Define(instruction, LaneActive(state, ballot, target) + " ? " + shuffled + " : " + (flags.boundControl ? std::string("0u") : state.Arg(instruction, 4u)));
}

// ds_bpermute_b32: each lane reads the lane its address names, within its 32-lane group.
void Bpermute(MslEmitterState& state, const IrValue& instruction) {
    RequireHostWave(state, instruction);
    const auto target = Constant(state, "uint", "(" + state.Arg(instruction, 1u) + " >> 2u) & 31u", "aps5_tl");
    state.Define(instruction, DsMaskedLaneRead(state, instruction, 0u, 2u, target));
}

// ds_permute_b32: each active lane sends its value to the lane its address names (within its 32-lane
// group); a lane nobody writes reads 0, the highest sending lane wins.
void Permute(MslEmitterState& state, const IrValue& instruction) {
    RequireHostWave(state, instruction);
    const auto result = state.Temporary("aps5_pm");
    const auto own = Constant(state, "uint", "(" + state.GuestLane() + ") & 31u", "aps5_own");
    const auto live = LiveMask(state);
    const auto source = state.Temporary("aps5_src");
    state.Line("uint " + result + " = 0u;");
    state.Line("for (ushort " + source + " = 0; " + source + " < 32; ++" + source + ") {");
    state.Indent();
    const auto value = Constant(state, "uint", "simd_shuffle(" + state.Arg(instruction, 0u) + ", " + source + ")");
    const auto address = Constant(state, "uint", "simd_shuffle(" + state.Arg(instruction, 1u) + ", " + source + ")");
    const auto active = Constant(state, "uint", "simd_shuffle(uint(" + state.Arg(instruction, 2u) + "), " + source + ")");
    state.Line("if (" + active + " != 0u && ((" + live + " >> uint(" + source + ")) & 1u) != 0u && ((" + address + " >> 2u) & 31u) == " + own + ") " + result + " = " + value + ";");
    state.Outdent();
    state.Line("}");
    state.Define(instruction, result);
}

void Swizzle(MslEmitterState& state, const IrValue& instruction) {
    RequireHostWave(state, instruction);
    const IrValue* control = instruction.Argument(1u)->Resolve();
    const auto target = SwizzleTarget(state, control != nullptr && control->HasImmediate() ? control->ImmediateU32() : 0u);
    state.Define(instruction, DsMaskedLaneRead(state, instruction, 0u, 2u, target));
}

// EmitWqmU64: every quad with an active lane becomes fully active.
void Wqm(MslEmitterState& state, const IrValue& instruction) {
    const auto value = Constant(state, "ulong", state.Arg(instruction, 0u), "aps5_wq");
    const auto one = Constant(state, "ulong", value + " | (" + value + " >> 1u)", "aps5_wq");
    const auto two = Constant(state, "ulong", one + " | (" + one + " >> 2u)", "aps5_wq");
    state.Define(instruction, "(" + two + " & 0x1111111111111111ul) * 0xful");
}

}

bool EmitMslWaveInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::Ballot:
        Ballot(state, instruction);
        return true;
    case IrOpcode::LaneId:
        // The hull program's lane is its control point, as the SPIR-V path's EmitLaneId.
        state.Define(instruction, state.program.Resources().stage == IrShaderStage::TessellationControl ? MslTessellationInvocation(state) : state.GuestLane());
        return true;
    case IrOpcode::ReadFirstLane:
        ReadFirstLane(state, instruction);
        return true;
    case IrOpcode::ReadLane:
        ReadLane(state, instruction);
        return true;
    case IrOpcode::WriteLane:
        WriteLane(state, instruction);
        return true;
    case IrOpcode::DppMoveU32:
        DppMove(state, instruction);
        return true;
    case IrOpcode::DppUpdateU32:
        DppUpdate(state, instruction);
        return true;
    case IrOpcode::Permlane16U32:
        Permlane16(state, instruction);
        return true;
    case IrOpcode::BpermuteU32:
        Bpermute(state, instruction);
        return true;
    case IrOpcode::PermuteU32:
        Permute(state, instruction);
        return true;
    case IrOpcode::SwizzleU32:
        Swizzle(state, instruction);
        return true;
    case IrOpcode::WqmU64:
        Wqm(state, instruction);
        return true;
    default:
        return false;
    }
}

MslPreludeEntry MslWavePrelude(MslPreludeChunk chunk) {
    throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not used by the wave family (work package B3)");
}

}
