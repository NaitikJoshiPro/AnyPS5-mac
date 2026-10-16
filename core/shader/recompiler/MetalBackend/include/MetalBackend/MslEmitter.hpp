#ifndef CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLEMITTER_HPP
#define CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLEMITTER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "Recompiler.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

struct MslTargetOptions {
    // Host SIMD-group width (32 on every Apple GPU).
    std::uint32_t subgroupSize = 32u;
    std::uint32_t bdaAbiVersion = 0u;
    std::uint32_t maxThreadgroupMemoryBytes = 32768u;
    // FLOAT_MODE keeps f32 denormals (FP_DENORM_32 == 3). Apple GPUs flush f32 denormals in every float
    // operation, so the ALU emits exact (slower) f32 arithmetic in this mode (MslAluMath.cpp, B2).
    bool keepF32Denormals = false;
    // The target advertises spv::CapabilityStorageImageMultisample: storage views of multisampled
    // images are emulated as 2D arrays of layers * samples slices (MslAbi Version 6). Without it they
    // fail as the SPIR-V backend does.
    bool storageImageMultisample = false;
    // Upstream #3988 (SpirvTarget::float16DenormPreserve): the GPU's half to float conversion keeps
    // half denormals, so ConvertF32F16 converts in hardware (aps5_f16_to_f32_native: NaN bits rebuilt
    // from the half) instead of decoding the bits with integer operations. Halves of fields narrower
    // than 16 bits keep the decode, as on the SPIR-V path.
    bool float16DenormPreserve = false;
    // SpirvTarget::mslWave64LaneLocal (DECISIONS D62).
    bool wave64LaneLocal = false;
};

// IR -> Metal Shading Language, the MetalBackend counterpart of SpirvEmitter (same inputs, same
// preconditions). The result is one library whose stage function is MslAbi::EntryPoint and whose
// bindings follow MslAbi.hpp.
class MslEmitter {
public:
    [[nodiscard]] MslShader Emit(const IrProgram& program, const ShaderStageInputInfo& inputInfo, const CompiledBindingLayout& bindings, const MslTargetOptions& target) const;
};

// Tessellation (DECISIONS D24, MslAbi Version 7): one tessellation stage library (Stage::Local,
// Hull or Domain) with the function constant values of its pipeline.
struct MslTessellationProgram {
    const MslShader* shader = nullptr;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> constants;
};

// The object function (local + hull programs per patch) and the mesh function (software
// tessellator + domain program) of a tessellated draw, each one self-contained library whose
// function constants are already resolved, and the draw's launch shape: object threadgroups of
// objectThreads threads (one per patch and instance), mesh threadgroups of meshThreads threads.
struct MslTessellationPipeline {
    MslShader object;
    MslShader mesh;
    std::uint32_t objectThreads = 0;
    std::uint32_t meshThreads = 0;
    std::uint32_t payloadBytes = 0;
    std::uint32_t maxMeshThreadgroups = 0;
    std::uint32_t topology = 0;
};

[[nodiscard]] MslTessellationPipeline LinkMslTessellation(const MslTessellationProgram& local, const MslTessellationProgram& hull, const MslTessellationProgram& domain, const TessellationConfiguration& config);

}

#endif
