#ifndef CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLEMITTER_HPP
#define CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLEMITTER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "Recompiler.hpp"
#include <cstdint>

namespace ShaderRecompiler {

struct MslTargetOptions {
    // Host SIMD-group width (32 on every Apple GPU).
    std::uint32_t subgroupSize = 32u;
    std::uint32_t bdaAbiVersion = 0u;
    std::uint32_t maxThreadgroupMemoryBytes = 32768u;
};

// IR -> Metal Shading Language, the MetalBackend counterpart of SpirvEmitter (same inputs, same
// preconditions). The result is one library whose stage function is MslAbi::EntryPoint and whose
// bindings follow MslAbi.hpp.
class MslEmitter {
public:
    [[nodiscard]] MslShader Emit(const IrProgram& program, const ShaderStageInputInfo& inputInfo, const CompiledBindingLayout& bindings, const MslTargetOptions& target) const;
};

}

#endif
