#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_NANTESTELIMINATOR_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_NANTESTELIMINATOR_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <cstdint>

namespace ShaderRecompiler {

struct NanTestEliminationStats {
    std::uint32_t removedTests = 0;
    std::uint32_t tests = 0;
};

// DECISIONS D63: the front end emulates RDNA's NaN results (nanResultF32 and friends: a quieted NaN
// source, else the default NaN 0xffc00000) with explicit NaN tests on the operands and results of
// float operations, because the host's NaN sign and payload differ. A conservative dataflow analysis
// proves values that can never be a NaN (constants, integer conversions, clamps, and arithmetic over
// such values that cannot make a NaN: no inf - inf, no 0 * inf, ...); their NaN tests
// (FPIsNan32, (bits & 0x7fffffff) > 0x7f800000) become false, and ConstantFolder/DeadCodeEliminator
// then drop the emulation around them. Never wrong for a value that may be a NaN: every unknown
// value (memory, inputs, anything the analysis does not model) may be a NaN.
class NanTestEliminator {
public:
    [[nodiscard]] NanTestEliminationStats Eliminate(IrProgram& program) const;
};

}

#endif
