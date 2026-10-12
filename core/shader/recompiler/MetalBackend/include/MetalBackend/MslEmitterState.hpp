#ifndef CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLEMITTERSTATE_HPP
#define CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLEMITTERSTATE_HPP

#include "MetalBackend/MslEmitter.hpp"
#include "IntermediateRepresentation/IrMetadata.hpp"
#include "SpirvBackend/SpirvAnalysis.hpp"
#include <bitset>
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ShaderRecompiler {

// Helper code an emitted library may need, pasted before the stage function in this order. Each
// instruction family owns its chunks: the text and dependencies live in the family's source file
// (MslPreludeEntry functions below); MslPrelude.cpp only dispatches by chunk.
enum class MslPreludeChunk : std::uint32_t {
    // Core (MslPrelude.cpp).
    Common,
    FaultRecord,
    LoopGuard,
    // ALU (MslAluMath.cpp, MslFloat64.cpp).
    Alu,
    Float16,
    Float64,
    // Wave operations, LDS and atomics (MslWave.cpp, MslMemory.cpp).
    Wave,
    LdsLock,
    Atomics64,
    // Images (MslImage.cpp).
    Image,
    // Buffers, typed formats, BDA (MslRuntimeBuffer.cpp).
    RuntimeBuffer,
    BufferFormat,
    Bda,
    // Stage interfaces (MslStageIo.cpp).
    StageIo,
    Count
};

struct MslPreludeEntry {
    std::string_view text;
    std::span<const MslPreludeChunk> dependencies;
};

[[nodiscard]] MslPreludeEntry MslPrelude(MslPreludeChunk chunk);
[[nodiscard]] MslPreludeEntry MslCorePrelude(MslPreludeChunk chunk);
[[nodiscard]] MslPreludeEntry MslAluPrelude(MslPreludeChunk chunk);
[[nodiscard]] MslPreludeEntry MslFloat64Prelude(MslPreludeChunk chunk);
[[nodiscard]] MslPreludeEntry MslWavePrelude(MslPreludeChunk chunk);
[[nodiscard]] MslPreludeEntry MslMemoryPrelude(MslPreludeChunk chunk);
[[nodiscard]] MslPreludeEntry MslImagePrelude(MslPreludeChunk chunk);
[[nodiscard]] MslPreludeEntry MslBufferPrelude(MslPreludeChunk chunk);
[[nodiscard]] MslPreludeEntry MslStagePrelude(MslPreludeChunk chunk);

// Stage-function parameters with Metal attributes ([[thread_position_in_threadgroup]], ...).
enum class MslBuiltin : std::uint32_t {
    ThreadPositionInThreadgroup,
    ThreadgroupPositionInGrid,
    ThreadgroupsPerGrid,
    ThreadIndexInThreadgroup,
    ThreadIndexInSimdgroup,
    Count
};

class MslEmitterState {
public:
    MslEmitterState(const IrProgram& program, const ShaderStageInputInfo& inputInfo, const CompiledBindingLayout& bindings, const MslTargetOptions& target);

    const IrProgram& program;
    const ShaderStageInputInfo& inputInfo;
    const CompiledBindingLayout& bindings;
    const MslTargetOptions& target;
    const SpirvRequirements requirements;
    MslAbi::Reflection reflection;

    // Guest lanes per host invocation: 2 for a wave64 program on 32-wide SIMD groups (compute), as
    // SpirvEmitterState::laneCount.
    std::uint32_t laneCount = 1u;
    std::uint32_t hostSubgroupSize = 32u;
    const ShaderWorkgroupInputInfo* workgroup = nullptr;

    // Function body: statements at the current indentation.
    void Line(std::string_view text);
    void Indent() { ++depth; }
    void Outdent();
    std::string body;

    // Function-scope locals, declared before the body (SSA values are hoisted: a definition inside a
    // loop is used after it).
    void DeclareLocal(std::string_view type, std::string_view name, std::string_view initializer);
    std::string locals;

    void RequirePrelude(MslPreludeChunk chunk);
    [[nodiscard]] std::string Prelude() const;

    // A [[function_constant(id)]] uint with the default the emitter would bake in (the value used
    // when the driver supplies none); returns the expression to read it.
    std::string FunctionConstant(std::uint32_t id, std::uint32_t defaultValue);
    std::map<std::uint32_t, std::uint32_t> functionConstants;

    // Stage-function parameters.
    std::string Builtin(MslBuiltin builtin);
    std::bitset<static_cast<std::size_t>(MslBuiltin::Count)> builtins;
    bool usesTable = false;
    bool usesPushData = false;
    bool usesDriverData = false;

    // Values: an immediate is a literal, every other value the hoisted local v<id>.
    [[nodiscard]] std::string Value(const IrValue* value);
    [[nodiscard]] std::string Arg(const IrValue& instruction, std::size_t index) { return Value(instruction.Argument(index)); }
    std::string Name(const IrValue& instruction);
    void Define(const IrValue& instruction, std::string_view expression);

    [[nodiscard]] const MemoryInfo& Memory(const IrValue& instruction) const;
    [[noreturn]] void Fail(std::string_view reason) const;
    [[noreturn]] void Fail(const IrValue& instruction, std::string_view reason) const;

    // A dword of the shader data (RuntimeAbi::ShaderData): from the push data when the layout puts it
    // there, else from the ShaderData buffer.
    std::string ShaderDataDword(std::uint32_t dword);
    // Byte i of the memory offsets (the bound buffer element's byte offset, see IrBindingLayout).
    std::string MemoryByteOffset(std::uint32_t element);
    // The descriptor table entry `element` of a buffer-kind binding: "aps5_table.<binding>[element]".
    std::string TableBuffer(DescriptorBindingKind kind, std::uint32_t element);
    // Index of `resource` within the `kind` binding (as ResourceForDescriptor on the SPIR-V path).
    std::uint32_t DescriptorElement(DescriptorBindingKind kind, std::uint32_t resource) const;

    // The loop guard's limit expression (DriverData), see MslFlowEmitter.cpp.
    std::string LoopGuardLimit();

private:
    std::uint32_t depth = 1u;
    std::bitset<static_cast<std::size_t>(MslPreludeChunk::Count)> prelude;
    std::unordered_set<const IrValue*> declared;
};

[[nodiscard]] std::string MslType(IrType type);
[[nodiscard]] std::string MslHex(std::uint64_t value);
[[nodiscard]] std::string MslTableField(DescriptorBindingKind kind);

// The stage function's signature and module-scope declarations (MslEntryPoint.cpp).
[[nodiscard]] std::string EmitMslEntryPoint(MslEmitterState& state);
// The stage function's body (MslFlowEmitter.cpp): structured control flow over the IR blocks.
void EmitMslFunctionBody(MslEmitterState& state);

// Per-family instruction emitters: each returns true when it emitted `instruction`. The flow
// emitter tries them in turn; an opcode no family handles fails the emission with its name.
bool EmitMslControlInstruction(MslEmitterState& state, const IrValue& instruction);
bool EmitMslAluInstruction(MslEmitterState& state, const IrValue& instruction);
bool EmitMslFloat64Instruction(MslEmitterState& state, const IrValue& instruction);
bool EmitMslWaveInstruction(MslEmitterState& state, const IrValue& instruction);
bool EmitMslMemoryInstruction(MslEmitterState& state, const IrValue& instruction);
bool EmitMslBufferInstruction(MslEmitterState& state, const IrValue& instruction);
bool EmitMslImageInstruction(MslEmitterState& state, const IrValue& instruction);
bool EmitMslStageInstruction(MslEmitterState& state, const IrValue& instruction);

}

#endif
