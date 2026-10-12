#include "MetalBackend/MslEmitterState.hpp"
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

bool EmitMslImageInstruction(MslEmitterState& state, const IrValue& instruction) {
    static_cast<void>(state);
    static_cast<void>(instruction);
    return false;
}

MslPreludeEntry MslImagePrelude(MslPreludeChunk chunk) {
    throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not implemented yet");
}

}
