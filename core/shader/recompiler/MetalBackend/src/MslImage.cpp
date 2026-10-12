#include "MetalBackend/MslEmitterState.hpp"

namespace ShaderRecompiler {

bool EmitMslImageInstruction(MslEmitterState& state, const IrValue& instruction) {
    static_cast<void>(state);
    static_cast<void>(instruction);
    return false;
}

}
