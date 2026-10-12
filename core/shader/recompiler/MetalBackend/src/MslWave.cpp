#include "MetalBackend/MslEmitterState.hpp"
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

bool EmitMslWaveInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::Ballot: {
        // Both guest lane halves for wave64 on 32-wide SIMD groups (core helper).
        const auto type = MslType(instruction.Type());
        const auto bits = state.WaveBallot(instruction.Argument(0u));
        if (type == "uint2") state.Define(instruction, bits);
        else if (type == "uint4") state.Define(instruction, "uint4(" + bits + ", 0u, 0u)");
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
