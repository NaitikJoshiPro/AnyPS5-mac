#include "MetalBackend/MslEmitterState.hpp"
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

bool EmitMslWaveInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::Ballot: {
        if (state.laneCount != 1u || state.program.WaveSize() != state.hostSubgroupSize) state.Fail(instruction, "ballots of wave64 programs on 32-wide SIMD groups are not implemented in the MSL backend yet");
        const auto type = MslType(instruction.Type());
        const auto bits = "uint(static_cast<simd_vote::vote_t>(simd_ballot(" + state.Arg(instruction, 0u) + ")))";
        if (type == "uint2") state.Define(instruction, "uint2(" + bits + ", 0u)");
        else if (type == "uint4") state.Define(instruction, "uint4(" + bits + ", 0u, 0u, 0u)");
        else state.Fail(instruction, "has an unexpected result type " + type);
        return true;
    }
    default:
        return false;
    }
}

MslPreludeEntry MslWavePrelude(MslPreludeChunk chunk) {
    throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not implemented yet");
}

}
