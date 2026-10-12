#include "MetalBackend/MslEmitterState.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

std::string Component(std::string_view vector, std::uint32_t component) {
    static constexpr const char* Names[] = {".x", ".y", ".z"};
    if (component > 2u) throw std::runtime_error("MSL emission failed: builtin component " + std::to_string(component) + " is out of range");
    return std::string(vector) + Names[component];
}

std::string WorkgroupId(MslEmitterState& state, std::uint32_t component) {
    state.usesDriverData = true;
    return "(" + Component(state.Builtin(MslBuiltin::ThreadgroupPositionInGrid), component) + " + aps5_driver.workgroupBase[" + std::to_string(component) + "])";
}

std::string ComputeBuiltin(MslEmitterState& state, StageInputKind kind, std::uint32_t component) {
    if (state.laneCount != 1u && (kind == StageInputKind::LocalInvocationId || kind == StageInputKind::GlobalInvocationId || kind == StageInputKind::LocalInvocationIndex)) {
        state.Fail("invocation builtins of wave64 programs on 32-wide SIMD groups are not implemented in the MSL backend yet");
    }
    switch (kind) {
    case StageInputKind::LocalInvocationId:
        return Component(state.Builtin(MslBuiltin::ThreadPositionInThreadgroup), component);
    case StageInputKind::LocalInvocationIndex:
        return state.Builtin(MslBuiltin::ThreadIndexInThreadgroup);
    case StageInputKind::WorkgroupId:
        return WorkgroupId(state, component);
    case StageInputKind::GlobalInvocationId: {
        const auto size = state.workgroup != nullptr ? std::max(state.workgroup->threadsNum[component], 1u) : 1u;
        return "(" + WorkgroupId(state, component) + " * " + std::to_string(size) + "u + " + Component(state.Builtin(MslBuiltin::ThreadPositionInThreadgroup), component) + ")";
    }
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
