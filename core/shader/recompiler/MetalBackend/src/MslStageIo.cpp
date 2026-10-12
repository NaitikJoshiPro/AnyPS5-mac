#include "MetalBackend/MslEmitterState.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

// Invocation ids come from the core (MslEmitterState), which rebuilds them per guest lane half for
// wave64 programs on 32-wide SIMD groups.
std::string ComputeBuiltin(MslEmitterState& state, StageInputKind kind, std::uint32_t component) {
    switch (kind) {
    case StageInputKind::LocalInvocationId:
        return state.GuestLocalInvocationId(component);
    case StageInputKind::LocalInvocationIndex:
        return state.GuestInvocationIndex();
    case StageInputKind::WorkgroupId:
        return state.WorkgroupId(component);
    case StageInputKind::GlobalInvocationId:
        return state.GuestGlobalInvocationId(component);
    case StageInputKind::DispatchThreadLimit:
        return state.ShaderDataDword(state.program.Metadata().bindings.DispatchThreadLimitDword() + component);
    default:
        state.Fail("compute builtin " + std::to_string(static_cast<unsigned>(kind)) + " is not implemented in the MSL backend yet");
    }
}

}

bool EmitMslStageInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::GetUserData: {
        const auto registerIndex = instruction.Argument(0)->Resolve()->Register().index;
        const auto& registers = state.program.Metadata().bindings.userDataRegisters;
        const auto found = std::lower_bound(registers.begin(), registers.end(), registerIndex);
        if (found == registers.end() || *found != registerIndex) state.Define(instruction, "0u");
        else state.Define(instruction, state.ShaderDataDword(static_cast<std::uint32_t>(found - registers.begin())));
        return true;
    }
    case IrOpcode::GetBuiltin: {
        const auto kind = static_cast<StageInputKind>(instruction.Argument(0)->Resolve()->ImmediateU32());
        const auto component = instruction.Argument(1)->Resolve()->ImmediateU32();
        if (state.program.Resources().stage != IrShaderStage::Compute) state.Fail(instruction, "builtins of graphics stages are not implemented in the MSL backend yet");
        state.Define(instruction, ComputeBuiltin(state, kind, component));
        return true;
    }
    default:
        return false;
    }
}

MslPreludeEntry MslStagePrelude(MslPreludeChunk chunk) {
    throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not implemented yet");
}

}
