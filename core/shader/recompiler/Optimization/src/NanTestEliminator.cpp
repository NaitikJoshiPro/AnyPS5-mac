#include "Optimization/NanTestEliminator.hpp"
#include "IntermediateRepresentation/IrBuilder.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ShaderRecompiler {

namespace {

// What is known about a float value (or the f32 bits held by a U32 value). `maxExponent`: |x| <=
// 2^maxExponent; Unbounded when x may be an infinity (every finite f32 is below 2^128).
constexpr int Unbounded = 1000;
constexpr int LargestFinite = 127;

struct Facts {
    bool notNan = false;
    int maxExponent = Unbounded;
    // Never below zero (a -0 counts as not below zero).
    bool notNegative = false;

    [[nodiscard]] bool Finite() const { return maxExponent <= LargestFinite; }
};

const Facts Unknown {};

Facts Join(const Facts& a, const Facts& b) {
    return {a.notNan && b.notNan, std::max(a.maxExponent, b.maxExponent), a.notNegative && b.notNegative};
}

int Bound(int exponent) {
    return exponent > LargestFinite ? Unbounded : exponent;
}

Facts OfBits(std::uint32_t bits) {
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return Unknown;
    Facts facts;
    facts.notNan = true;
    facts.notNegative = (bits & 0x80000000u) == 0u || magnitude == 0u;
    if (magnitude == 0x7f800000u) {
        facts.maxExponent = Unbounded;
    } else {
        // |x| < 2^(exponent - 126) for a biased exponent field.
        facts.maxExponent = static_cast<int>(magnitude >> 23u) - 126;
    }
    return facts;
}

Facts Add(const Facts& a, const Facts& b) {
    // inf + -inf is the only NaN of two non-NaN operands.
    return {a.notNan && b.notNan && (a.Finite() || b.Finite()), a.Finite() && b.Finite() ? Bound(std::max(a.maxExponent, b.maxExponent) + 1) : Unbounded, a.notNegative && b.notNegative};
}

Facts Multiply(const Facts& a, const Facts& b) {
    // 0 * inf is the only NaN of two non-NaN operands.
    return {a.notNan && b.notNan && a.Finite() && b.Finite(), a.Finite() && b.Finite() ? Bound(a.maxExponent + b.maxExponent) : Unbounded, a.notNegative && b.notNegative};
}

bool IsU32(const IrValue* value, std::uint32_t bits) {
    return value->HasImmediate() && value->ImmediateU32() == bits;
}

// The operand of `value` = op(operand, mask) or op(mask, operand).
const IrValue* Masked(const IrValue* value, IrOpcode opcode, std::uint32_t mask) {
    if (value->HasImmediate() || value->Opcode() != opcode || value->ArgumentCount() != 2u) return nullptr;
    const auto* lhs = value->Argument(0)->Resolve();
    const auto* rhs = value->Argument(1)->Resolve();
    if (IsU32(rhs, mask)) return lhs;
    if (IsU32(lhs, mask)) return rhs;
    return nullptr;
}

class Analysis {
public:
    const Facts& Of(const IrValue* value) {
        value = value->Resolve();
        if (const auto found = known.find(value); found != known.end()) return found->second;
        // A cycle (a loop's phi) reads Unknown: conservative.
        known.emplace(value, Unknown);
        const Facts facts = Compute(*value);
        return known[value] = facts;
    }

private:
    Facts Compute(const IrValue& value) {
        if (value.HasImmediate()) {
            if (value.Type() == IrType::F32) return OfBits(std::bit_cast<std::uint32_t>(value.ImmediateF32()));
            if (value.Type() == IrType::U32) return OfBits(value.ImmediateU32());
            return Unknown;
        }
        const auto argument = [&](std::size_t index) -> const Facts& { return Of(value.Argument(index)); };
        switch (value.Opcode()) {
        // f32 bits in a U32 and back.
        case IrOpcode::BitCastU32F32:
        case IrOpcode::BitCastF32U32:
            return argument(0);
        case IrOpcode::BitwiseAnd32:
            if (const auto* operand = Masked(&value, IrOpcode::BitwiseAnd32, 0x7fffffffu)) {
                Facts facts = Of(operand);
                facts.notNegative = true;
                return facts;
            }
            return Unknown;
        case IrOpcode::BitwiseXor32:
            if (const auto* operand = Masked(&value, IrOpcode::BitwiseXor32, 0x80000000u)) {
                Facts facts = Of(operand);
                facts.notNegative = false;
                return facts;
            }
            return Unknown;
        case IrOpcode::BitwiseOr32:
            if (const auto* operand = Masked(&value, IrOpcode::BitwiseOr32, 0x80000000u)) {
                Facts facts = Of(operand);
                facts.notNegative = false;
                return facts;
            }
            return Unknown;
        case IrOpcode::Select:
        case IrOpcode::SelectU32:
        case IrOpcode::SelectF32:
            return Join(argument(1), argument(2));
        case IrOpcode::Phi: {
            Facts facts = argument(0);
            for (std::size_t index = 1; index < value.ArgumentCount(); ++index) facts = Join(facts, argument(index));
            return facts;
        }
        // Integer conversions: finite, at most 2^32.
        case IrOpcode::ConvertIToF32:
        case IrOpcode::ConvertF32S32:
            return {true, 32, false};
        case IrOpcode::ConvertUToF32:
        case IrOpcode::ConvertF32U32:
            return {true, 32, true};
        case IrOpcode::FPAbs32: {
            Facts facts = argument(0);
            facts.notNegative = true;
            return facts;
        }
        case IrOpcode::FPNeg32: {
            Facts facts = argument(0);
            facts.notNegative = false;
            return facts;
        }
        // aps5_fsat: NaN and negatives give +0, the rest at most 1.
        case IrOpcode::FPSaturate32:
            return {true, 0, true};
        case IrOpcode::FPAdd32:
            return Add(argument(0), argument(1));
        case IrOpcode::FPSub32: {
            Facts negated = argument(1);
            negated.notNegative = false;
            Facts sum = Add(argument(0), negated);
            sum.notNegative = false;
            return sum;
        }
        case IrOpcode::FPMul32:
            return Multiply(argument(0), argument(1));
        case IrOpcode::FPFma32:
        case IrOpcode::FPMad32:
            return Add(Multiply(argument(0), argument(1)), argument(2));
        // aps5_fmin/aps5_fmax: a NaN operand gives the other one.
        case IrOpcode::FPMin32:
        case IrOpcode::FPMax32: {
            const auto& a = argument(0);
            const auto& b = argument(1);
            return {a.notNan || b.notNan, std::max(a.maxExponent, b.maxExponent), a.notNegative && b.notNegative};
        }
        case IrOpcode::FPMinTri32:
        case IrOpcode::FPMaxTri32: {
            const auto& a = argument(0);
            const auto& b = argument(1);
            const auto& c = argument(2);
            const bool firstNotNan = a.notNan || b.notNan;
            return {firstNotNan || c.notNan, std::max({a.maxExponent, b.maxExponent, c.maxExponent}), a.notNegative && b.notNegative && c.notNegative};
        }
        case IrOpcode::FPFloor32:
        case IrOpcode::FPCeil32:
        case IrOpcode::FPTrunc32:
        case IrOpcode::FPRoundEven32: {
            const auto& a = argument(0);
            return {a.notNan, a.Finite() ? Bound(std::max(a.maxExponent, 0) + 1) : Unbounded, a.notNegative};
        }
        // aps5_fract: [0, 1) of finite values; fract(inf) is a NaN.
        case IrOpcode::FPFract32: {
            const auto& a = argument(0);
            return {a.notNan && a.Finite(), 0, true};
        }
        case IrOpcode::FPSqrt: {
            const auto& a = argument(0);
            return {a.notNan && a.notNegative, a.Finite() ? Bound(a.maxExponent / 2 + 1) : Unbounded, true};
        }
        case IrOpcode::FPRecipSqrt32: {
            const auto& a = argument(0);
            return {a.notNan && a.notNegative, Unbounded, false};
        }
        // 1 / +-0 is an infinity, 1 / inf a zero: never a NaN of a non-NaN.
        case IrOpcode::FPRecip32:
        case IrOpcode::FPRecipIFlag32:
            return {argument(0).notNan, Unbounded, false};
        case IrOpcode::FPExp2: {
            const auto& a = argument(0);
            // 2^x <= 2^(2^maxExponent).
            return {a.notNan, a.Finite() && a.maxExponent <= 6 ? (1 << std::max(a.maxExponent, 0)) : Unbounded, true};
        }
        default:
            return Unknown;
        }
    }

public:
    // Survey aid (APS5_MSL_NAN_SURVEY=<file>): the opcode a possible NaN comes from.
    std::string Source(const IrValue* value, std::unordered_set<const IrValue*>& seen) {
        value = value->Resolve();
        if (Of(value).notNan) return "not-nan";
        if (!seen.insert(value).second) return "cycle";
        if (value->HasImmediate()) return "immediate-nan";
        switch (value->Opcode()) {
        case IrOpcode::BitCastU32F32: case IrOpcode::BitCastF32U32: case IrOpcode::FPAbs32: case IrOpcode::FPNeg32:
        case IrOpcode::BitwiseAnd32: case IrOpcode::BitwiseXor32: case IrOpcode::BitwiseOr32:
        case IrOpcode::Select: case IrOpcode::SelectU32: case IrOpcode::SelectF32: case IrOpcode::Phi:
        case IrOpcode::FPAdd32: case IrOpcode::FPSub32: case IrOpcode::FPMul32: case IrOpcode::FPFma32: case IrOpcode::FPMad32:
        case IrOpcode::FPMin32: case IrOpcode::FPMax32: case IrOpcode::FPFloor32: case IrOpcode::FPCeil32: case IrOpcode::FPTrunc32:
        case IrOpcode::FPFract32: case IrOpcode::FPSqrt: case IrOpcode::FPRecipSqrt32: case IrOpcode::FPRecip32: case IrOpcode::FPExp2: {
            for (std::size_t index = value->Opcode() == IrOpcode::Select || value->Opcode() == IrOpcode::SelectU32 || value->Opcode() == IrOpcode::SelectF32 ? 1u : 0u; index < value->ArgumentCount(); ++index) {
                const auto* argument = value->Argument(index)->Resolve();
                if (argument->HasImmediate() && Of(argument).notNan) continue;
                if (!Of(argument).notNan) return Source(argument, seen);
            }
            return std::string("bound:") + std::string(IrOpcodeName(value->Opcode()));
        }
        default:
            return std::string(IrOpcodeName(value->Opcode()));
        }
    }

private:
    std::unordered_map<const IrValue*, Facts> known;
};

// (bits & 0x7fffffff) > 0x7f800000: the bits' NaN test the front end emits.
const IrValue* NanTestOperand(const IrValue& value) {
    if (value.Opcode() == IrOpcode::FPIsNan32) return value.Argument(0)->Resolve();
    if (value.Opcode() != IrOpcode::UGreaterThan32) return nullptr;
    if (!IsU32(value.Argument(1)->Resolve(), 0x7f800000u)) return nullptr;
    return Masked(value.Argument(0)->Resolve(), IrOpcode::BitwiseAnd32, 0x7fffffffu);
}

}

NanTestEliminationStats NanTestEliminator::Eliminate(IrProgram& program) const {
    NanTestEliminationStats stats;
    Analysis analysis;
    static const char* survey = std::getenv("APS5_MSL_NAN_SURVEY");
    std::map<std::string, std::uint32_t> sources;
    IrBuilder builder(program);
    for (auto* block : program.BlockOrder()) {
        for (auto* instruction : block->Instructions()) {
            if (instruction->HasImmediate() || !instruction->HasUses()) continue;
            const IrValue* operand = NanTestOperand(*instruction);
            if (operand == nullptr) continue;
            ++stats.tests;
            if (!analysis.Of(operand).notNan) {
                if (survey != nullptr) {
                    std::unordered_set<const IrValue*> seen;
                    ++sources[analysis.Source(operand, seen)];
                }
                continue;
            }
            instruction->ReplaceUsesWith(&builder.ConstantBool(false), true);
            ++stats.removedTests;
        }
    }
    if (survey != nullptr) {
        if (FILE* out = std::fopen(survey, "a")) {
            std::fprintf(out, "[nan] tests=%u removed=%u", stats.tests, stats.removedTests);
            for (const auto& [source, count] : sources) std::fprintf(out, " %s=%u", source.c_str(), count);
            std::fprintf(out, "\n");
            std::fclose(out);
        }
    }
    return stats;
}

}
