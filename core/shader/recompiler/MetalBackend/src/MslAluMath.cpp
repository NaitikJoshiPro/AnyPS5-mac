#include "MetalBackend/MslEmitterState.hpp"
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

// Integer, logic, compare, select, composite and bit-cast operations. U64 is MSL ulong (the SPIR-V
// path uses a uvec2 pair); shift counts are masked to the operand width, as RDNA2 hardware does.
// Float semantics (RDNA denormal, NaN, -0 and rounding behaviour, SpirvAluEmitterMath.cpp) are work
// package B2.

namespace {

void Binary(MslEmitterState& state, const IrValue& instruction, std::string_view operation) {
    state.Define(instruction, "(" + state.Arg(instruction, 0u) + " " + std::string(operation) + " " + state.Arg(instruction, 1u) + ")");
}

void Signed32(MslEmitterState& state, const IrValue& instruction, std::string_view operation) {
    state.Define(instruction, "(as_type<int>(" + state.Arg(instruction, 0u) + ") " + std::string(operation) + " as_type<int>(" + state.Arg(instruction, 1u) + "))");
}

void Signed64(MslEmitterState& state, const IrValue& instruction, std::string_view operation) {
    state.Define(instruction, "(as_type<long>(" + state.Arg(instruction, 0u) + ") " + std::string(operation) + " as_type<long>(" + state.Arg(instruction, 1u) + "))");
}

void Call(MslEmitterState& state, const IrValue& instruction, std::string_view function) {
    std::string expression = std::string(function) + "(";
    for (std::size_t index = 0; index < instruction.ArgumentCount(); ++index) expression += (index == 0u ? "" : ", ") + state.Arg(instruction, index);
    state.Define(instruction, expression + ")");
}

void Construct(MslEmitterState& state, const IrValue& instruction) {
    Call(state, instruction, MslType(instruction.Type()));
}

void Extract(MslEmitterState& state, const IrValue& instruction) {
    const auto* index = instruction.Argument(1u)->Resolve();
    if (!index->HasImmediate()) state.Fail(instruction, "needs an immediate component index");
    state.Define(instruction, state.Arg(instruction, 0u) + "[" + std::to_string(index->ImmediateU32()) + "]");
}

}

bool EmitMslAluInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::IAdd32:
    case IrOpcode::IAdd64:
        Binary(state, instruction, "+");
        return true;
    case IrOpcode::ISub32:
    case IrOpcode::ISub64:
        Binary(state, instruction, "-");
        return true;
    case IrOpcode::IMul32:
    case IrOpcode::IMul64:
        Binary(state, instruction, "*");
        return true;
    case IrOpcode::UDiv32:
        state.Define(instruction, "(" + state.Arg(instruction, 1u) + " == 0u ? 0xffffffffu : " + state.Arg(instruction, 0u) + " / " + state.Arg(instruction, 1u) + ")");
        return true;
    case IrOpcode::UMulHi:
        Call(state, instruction, "mulhi");
        return true;
    case IrOpcode::SMulHi:
        state.Define(instruction, "as_type<uint>(mulhi(as_type<int>(" + state.Arg(instruction, 0u) + "), as_type<int>(" + state.Arg(instruction, 1u) + ")))");
        return true;
    case IrOpcode::IAddCarry32:
        state.Define(instruction, "uint2(" + state.Arg(instruction, 0u) + " + " + state.Arg(instruction, 1u) + ", (" + state.Arg(instruction, 0u) + " + " + state.Arg(instruction, 1u) + ") < " + state.Arg(instruction, 0u) + " ? 1u : 0u)");
        return true;
    case IrOpcode::IAbs32:
        state.Define(instruction, "as_type<uint>(abs(as_type<int>(" + state.Arg(instruction, 0u) + ")))");
        return true;
    case IrOpcode::ShiftLeftLogical32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " << (" + state.Arg(instruction, 1u) + " & 31u))");
        return true;
    case IrOpcode::ShiftRightLogical32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " >> (" + state.Arg(instruction, 1u) + " & 31u))");
        return true;
    case IrOpcode::ShiftRightArithmetic32:
        state.Define(instruction, "as_type<uint>(as_type<int>(" + state.Arg(instruction, 0u) + ") >> (" + state.Arg(instruction, 1u) + " & 31u))");
        return true;
    case IrOpcode::ShiftLeftLogical64:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " << (" + state.Arg(instruction, 1u) + " & 63u))");
        return true;
    case IrOpcode::ShiftRightLogical64:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " >> (" + state.Arg(instruction, 1u) + " & 63u))");
        return true;
    case IrOpcode::ShiftRightArithmetic64:
        state.Define(instruction, "as_type<ulong>(as_type<long>(" + state.Arg(instruction, 0u) + ") >> (" + state.Arg(instruction, 1u) + " & 63u))");
        return true;
    case IrOpcode::BitwiseAnd32:
    case IrOpcode::BitwiseAnd64:
        Binary(state, instruction, "&");
        return true;
    case IrOpcode::BitwiseOr32:
        Binary(state, instruction, "|");
        return true;
    case IrOpcode::BitwiseXor32:
        Binary(state, instruction, "^");
        return true;
    case IrOpcode::BitwiseNot32:
        state.Define(instruction, "~" + state.Arg(instruction, 0u));
        return true;
    case IrOpcode::BitReverse32:
        Call(state, instruction, "reverse_bits");
        return true;
    case IrOpcode::BitCount32:
    case IrOpcode::BitCount64:
        state.Define(instruction, "uint(popcount(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::FindILsb32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " == 0u ? 0xffffffffu : ctz(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::FindUMsb32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " == 0u ? 0xffffffffu : 31u - clz(" + state.Arg(instruction, 0u) + "))");
        return true;
    case IrOpcode::BitFieldUExtract:
        Call(state, instruction, "extract_bits");
        return true;
    case IrOpcode::BitFieldSExtract:
        state.Define(instruction, "as_type<uint>(extract_bits(as_type<int>(" + state.Arg(instruction, 0u) + "), " + state.Arg(instruction, 1u) + ", " + state.Arg(instruction, 2u) + "))");
        return true;
    case IrOpcode::BitFieldInsert:
        Call(state, instruction, "insert_bits");
        return true;
    case IrOpcode::UMin32:
        Call(state, instruction, "min");
        return true;
    case IrOpcode::UMax32:
        Call(state, instruction, "max");
        return true;
    case IrOpcode::SMin32:
        state.Define(instruction, "as_type<uint>(min(as_type<int>(" + state.Arg(instruction, 0u) + "), as_type<int>(" + state.Arg(instruction, 1u) + ")))");
        return true;
    case IrOpcode::SMax32:
        state.Define(instruction, "as_type<uint>(max(as_type<int>(" + state.Arg(instruction, 0u) + "), as_type<int>(" + state.Arg(instruction, 1u) + ")))");
        return true;
    case IrOpcode::IEqual32:
    case IrOpcode::IEqual64:
        Binary(state, instruction, "==");
        return true;
    case IrOpcode::INotEqual32:
    case IrOpcode::INotEqual64:
        Binary(state, instruction, "!=");
        return true;
    case IrOpcode::ULessThan32:
    case IrOpcode::ULessThan64:
        Binary(state, instruction, "<");
        return true;
    case IrOpcode::ULessThanEqual32:
        Binary(state, instruction, "<=");
        return true;
    case IrOpcode::UGreaterThan32:
    case IrOpcode::UGreaterThan64:
        Binary(state, instruction, ">");
        return true;
    case IrOpcode::UGreaterThanEqual32:
        Binary(state, instruction, ">=");
        return true;
    case IrOpcode::SLessThan32:
        Signed32(state, instruction, "<");
        return true;
    case IrOpcode::SLessThan64:
        Signed64(state, instruction, "<");
        return true;
    case IrOpcode::SLessThanEqual32:
        Signed32(state, instruction, "<=");
        return true;
    case IrOpcode::SGreaterThan32:
        Signed32(state, instruction, ">");
        return true;
    case IrOpcode::SGreaterThanEqual32:
        Signed32(state, instruction, ">=");
        return true;
    case IrOpcode::LogicalAnd:
        Binary(state, instruction, "&&");
        return true;
    case IrOpcode::LogicalOr:
        Binary(state, instruction, "||");
        return true;
    case IrOpcode::LogicalXor:
        Binary(state, instruction, "!=");
        return true;
    case IrOpcode::LogicalNot:
        state.Define(instruction, "!" + state.Arg(instruction, 0u));
        return true;
    case IrOpcode::Select:
    case IrOpcode::SelectU1:
    case IrOpcode::SelectU32:
    case IrOpcode::SelectF32:
        state.Define(instruction, "(" + state.Arg(instruction, 0u) + " ? " + state.Arg(instruction, 1u) + " : " + state.Arg(instruction, 2u) + ")");
        return true;
    case IrOpcode::CompositeConstructU32x2:
    case IrOpcode::CompositeConstructU32x3:
    case IrOpcode::CompositeConstructU32x4:
    case IrOpcode::CompositeConstructF32x2:
        Construct(state, instruction);
        return true;
    case IrOpcode::CompositeConstructU64:
        state.Define(instruction, "(ulong(" + state.Arg(instruction, 0u) + ") | (ulong(" + state.Arg(instruction, 1u) + ") << 32))");
        return true;
    case IrOpcode::CompositeExtractU32x2:
    case IrOpcode::CompositeExtractU32x3:
    case IrOpcode::CompositeExtractU32x4:
        Extract(state, instruction);
        return true;
    case IrOpcode::CompositeExtractU64: {
        const auto* index = instruction.Argument(1u)->Resolve();
        if (!index->HasImmediate() || index->ImmediateU32() > 1u) state.Fail(instruction, "needs an immediate half index");
        state.Define(instruction, "uint(" + state.Arg(instruction, 0u) + " >> " + std::to_string(index->ImmediateU32() * 32u) + ")");
        return true;
    }
    case IrOpcode::ConvertU16U32:
        state.Define(instruction, "ushort(" + state.Arg(instruction, 0u) + " & 0xffffu)");
        return true;
    case IrOpcode::ConvertU8U32:
        state.Define(instruction, "uchar(" + state.Arg(instruction, 0u) + " & 0xffu)");
        return true;
    case IrOpcode::ConvertU32U16:
    case IrOpcode::ConvertU32U8:
        state.Define(instruction, "uint(" + state.Arg(instruction, 0u) + ")");
        return true;
    case IrOpcode::BitCastF32U32:
    case IrOpcode::BitCastU32F32:
    case IrOpcode::BitCastF16U16:
    case IrOpcode::BitCastU16F16:
        state.Define(instruction, "as_type<" + MslType(instruction.Type()) + ">(" + state.Arg(instruction, 0u) + ")");
        return true;
    case IrOpcode::UndefU1:
    case IrOpcode::UndefU8:
    case IrOpcode::UndefU16:
    case IrOpcode::UndefU32:
    case IrOpcode::UndefU64:
        state.Define(instruction, MslType(instruction.Type()) + "(0)");
        return true;
    default:
        return false;
    }
}

MslPreludeEntry MslAluPrelude(MslPreludeChunk chunk) {
    throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not implemented yet");
}

}
