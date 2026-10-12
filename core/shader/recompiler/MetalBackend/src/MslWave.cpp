#include "MetalBackend/MslEmitterState.hpp"
#include <stdexcept>
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
    if (OneLanePerInvocationNarrow(state)) state.Fail(instruction, "a cross-lane read of a wave64 program at one lane per invocation spans two host SIMD groups (work package B3)");
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
    } else {
        result = Constant(state, valueType, "simd_shuffle(" + wrap(state.Arg(instruction, index)) + ", ushort(" + lane + "))");
    }
    return flag ? "(" + result + " != 0u)" : result;
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
    const auto ballot = Constant(state, "uint2", state.WaveBallot(instruction.Argument(1u)), "aps5_b");
    const auto first = Constant(state, "uint", ballot + ".x != 0u ? ctz(" + ballot + ".x) : (" + ballot + ".y != 0u ? 32u + ctz(" + ballot + ".y) : 0u)", "aps5_fl");
    state.Define(instruction, Shuffle(state, instruction, 0u, first));
}

void ReadLane(MslEmitterState& state, const IrValue& instruction) {
    RequireHostWave(state, instruction);
    const auto lane = Constant(state, "uint", "(" + state.Arg(instruction, 1u) + ") & " + U32(state.program.WaveSize() - 1u), "aps5_rl");
    state.Define(instruction, Shuffle(state, instruction, 0u, lane));
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
    const auto ballot = Constant(state, "uint2", state.WaveBallot(instruction.Argument(1u)), "aps5_b");
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
            const auto ballot = Constant(state, "uint2", state.WaveBallot(instruction.Argument(2u)), "aps5_b");
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
    const auto ballot = Constant(state, "uint2", state.WaveBallot(instruction.Argument(3u)), "aps5_b");
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
        if (state.program.Resources().stage == IrShaderStage::TessellationControl) state.Fail(instruction, "the TCS lane id is the invocation id (work package B6)");
        state.Define(instruction, state.GuestLane());
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
