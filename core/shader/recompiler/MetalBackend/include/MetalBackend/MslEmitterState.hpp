#ifndef CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLEMITTERSTATE_HPP
#define CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLEMITTERSTATE_HPP

#include "MetalBackend/MslEmitter.hpp"
#include "IntermediateRepresentation/IrMetadata.hpp"
#include "SpirvBackend/SpirvAnalysis.hpp"
#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

// Execution scope of the barriers that keep one guest wave's LDS accesses in program order across
// host invocations (lanes of a guest wave run in lockstep, host invocations do not): see
// MslWaveLdsScope in MslFlowEmitter.cpp.
enum class MslBarrierScope : std::uint32_t {
    None,
    Simdgroup,
    Threadgroup
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

    // Guest lanes per host invocation: 2 for a wave64 compute program on 32-wide SIMD groups (as
    // SpirvEmitterState::laneCount). Each host invocation then runs guest lanes `simd_lane` (half 0)
    // and `simd_lane + 32` (half 1) of guest wave `thread_index_in_threadgroup / 32`: the flow emitter
    // offers every instruction once per half with `half` set, and Value/Name/Define resolve to that
    // half's names. Wave-uniform values (WaveUniformValues) without side effects are computed once,
    // in half 0, and both halves read the same name; such an emitter reaches the other half's
    // operands through HalfValue/HalfArg (ballots, readfirstlane, ...). Workgroup barriers and
    // MeshAllocate are emitted once.
    std::uint32_t laneCount = 1u;
    // A wave64 compute program at one guest lane per invocation (laneCount 1, DECISIONS D62): its
    // ballots are the lane's own bit and its wave-mask branches follow the lane's own condition
    // (MslLaneLocal.cpp proves both exact).
    bool laneLocal = false;
    // Lane-local in a one-wave threadgroup that exchanges values through threadgroup memory
    // (MslLaneLocalPlan::threadgroupExchange): exact ballots when fullBallots (else the lane's own
    // bit), votes for the wave-mask branches of the blocks in votedBranches (the others follow the
    // lane's own condition), lane reads.
    bool threadgroupExchange = false;
    bool fullBallots = false;
    std::unordered_set<std::uint32_t> votedBranches;
    // Threadgroup exchange (one-wave threadgroups, every invocation at the same point): the value of
    // guest lane `lane` (uint expressions), a vote over the wave, an exact 64-lane ballot (uint2).
    [[nodiscard]] std::string ExchangeRead(const std::string& value, const std::string& lane);
    [[nodiscard]] std::string ExchangeVote(const IrValue* predicate, bool all);
    [[nodiscard]] std::string ExchangeBallot(const IrValue* predicate);
    // The ballot of this invocation's SIMD group in both words (bit = lane & 31): for cross-lane
    // reads that stay in the reading lane's 32-lane half at one lane per invocation.
    [[nodiscard]] std::string HalfBallot(const IrValue* predicate);
    std::uint32_t half = 0u;
    // Relative-addressing window slots kept in the window array (the rest are locals of their own,
    // MslAluMath.cpp VectorWindow): unset until the first window access computes it, 0 for the whole
    // window in the array.
    std::optional<std::uint32_t> vectorWindowIndexed;
    std::uint32_t hostSubgroupSize = 32u;
    const ShaderWorkgroupInputInfo* workgroup = nullptr;
    std::unordered_set<const IrValue*> waveUniform;
    // Upstream d422c760c/3265cb6d7 (SpirvBdaRead.cpp BdaAnchor): the user-data register pairs whose
    // BDA range the program searches once at entry (MslAddress.cpp), collected at the first address
    // read; `name` is the prologue's ulong3 (usable size, device address, offset of the base in the
    // range), empty until a read uses the anchor.
    struct BdaAnchor {
        std::uint32_t lowRegister = 0;
        std::uint32_t highRegister = 0;
        std::string name;
    };
    std::optional<std::vector<BdaAnchor>> bdaAnchors;
    // True when `instruction` is computed once for both halves (always true when laneCount is 1).
    [[nodiscard]] bool SharedAcrossHalves(const IrValue& instruction) const;

    // Function body: statements at the current indentation.
    void Line(std::string_view text);
    void Indent() { ++depth; }
    void Outdent();
    std::string body;

    // Function-scope locals, declared before the body (SSA values are hoisted: a definition inside a
    // loop is used after it).
    void DeclareLocal(std::string_view type, std::string_view name, std::string_view initializer);
    std::string locals;
    // A fresh function-unique identifier for an emitter's temporary (`<prefix><n>`); the caller
    // declares it (DeclareLocal or a block-scope declaration).
    [[nodiscard]] std::string Temporary(std::string_view prefix = "aps5_t");

    // Hooks for the instruction families, all emitted by MslEntryPoint.cpp:
    // - module-scope declarations after the prelude (types, constant tables, function constants);
    void AddDeclaration(std::string_view text);
    std::string declarations;
    // - extra stage-function parameters, each a full declaration with its attribute
    //   ("threadgroup uint* x [[threadgroup(0)]]"); added once per text;
    void AddParameter(std::string_view text);
    std::vector<std::string> parameters;
    // - statements run at the start of the function, after the locals and before the first block
    //   (threadgroup variable declarations, the LDS lock reset and its barrier, ...).
    void AddPrologue(std::string_view line);
    std::string prologue;

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
    // Graphics stages (MslStageIo.cpp, B6): fragment inputs read (Vulkan location -> how: flat,
    // perspective or linear, and the interpolation locations read, one bit per
    // InterpolationLocation: one location takes its qualifier, several make the input an
    // interpolant<> read with interpolate_at_*, DECISIONS D38), the emulated barycentrics' locations
    // (perspective, linear), and whether the pixel valid mask (kill) is used.
    struct FragmentInput {
        enum class Mode : std::uint32_t { Flat, Perspective, Linear } mode = Mode::Flat;
        std::uint32_t locations = 0;
    };
    std::map<std::uint32_t, FragmentInput> fragmentInputs;
    std::array<std::uint32_t, 2> barycentricLocations {};
    bool usesValidMask = false;
    // Images (MslImage.cpp): the direct runtime-mode switch case being emitted, {selector function
    // constant id, mode index}, for Reflection::samplerModeUses; {0, ~0u} outside one.
    std::pair<std::uint32_t, std::uint32_t> imageModeCase {0u, ~0u};
    // Images (MslImage.cpp): the image/sampler table access being emitted (MslAbi Version 15), as
    // SpirvEmitterState::tableAccess: MSL expressions of the texture array element, the record's T#
    // dword 3 (swizzle) and the sampler array element.
    struct TableAccess {
        bool active = false;
        std::string element;
        std::string dword3;
        std::string samplerElement;
    };
    TableAccess tableAccess;
    // Descriptor kinds the emitted code references (TableResource): the table holds only these ranges,
    // as the SPIR-V path drops unused descriptor variables (so the binding plan drops them too).
    std::set<DescriptorBindingKind> usedKinds;
    bool usesPushData = false;
    bool usesDriverData = false;

    // Values: an immediate is a literal, every other value the hoisted local v<id> (half 1 of a
    // value that is not shared across halves: v<id>_h).
    [[nodiscard]] std::string Value(const IrValue* value) { return HalfValue(value, half); }
    [[nodiscard]] std::string HalfValue(const IrValue* value, std::uint32_t laneHalf);
    [[nodiscard]] std::string Arg(const IrValue& instruction, std::size_t index) { return Value(instruction.Argument(index)); }
    [[nodiscard]] std::string HalfArg(const IrValue& instruction, std::size_t index, std::uint32_t laneHalf) { return HalfValue(instruction.Argument(index), laneHalf); }
    std::string Name(const IrValue& instruction) { return HalfName(instruction, half); }
    std::string HalfName(const IrValue& instruction, std::uint32_t laneHalf);
    void Define(const IrValue& instruction, std::string_view expression);

    // Guest lane of the current half within its guest wave (the value of v_mbcnt-style lane ids):
    // the host SIMD lane, plus 32 in half 1; for a wave64 program kept at one lane per invocation on
    // a narrower host (graphics stages), the local index modulo the wave size.
    [[nodiscard]] std::string GuestLane();
    // Guest LocalInvocationIndex of the current half (the host's for laneCount 1).
    [[nodiscard]] std::string GuestInvocationIndex();
    // Guest LocalInvocationId / GlobalInvocationId component of the current half: the Metal builtins
    // for laneCount 1, rebuilt from GuestInvocationIndex for laneCount 2 (the host threadgroup is then
    // one-dimensional, see MslEntryPoint.cpp).
    [[nodiscard]] std::string GuestLocalInvocationId(std::uint32_t component);
    [[nodiscard]] std::string GuestGlobalInvocationId(std::uint32_t component);
    // The guest workgroup id component (Metal's threadgroup position plus the split-dispatch base).
    [[nodiscard]] std::string WorkgroupId(std::uint32_t component);
    // A 64-lane ballot of a per-lane predicate as a uint2 (lanes 0-31, lanes 32-63): both halves for
    // laneCount 2, the host SIMD group (high word 0) for a wave32 program.
    [[nodiscard]] std::string WaveBallot(const IrValue* predicate);
    // simd_any / simd_all of a per-lane predicate over the whole guest wave.
    [[nodiscard]] std::string WaveAny(const IrValue* predicate);
    [[nodiscard]] std::string WaveAll(const IrValue* predicate);

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
    // The descriptor table entry `element` of an image heap or the sampler heap (a texture or sampler
    // expression, "aps5_table.<binding>[element]"); the member type is MslTableMemberType.
    std::string TableResource(DescriptorBindingKind kind, std::uint32_t element);
    // Index of `resource` within the `kind` binding (as ResourceForDescriptor on the SPIR-V path).
    std::uint32_t DescriptorElement(DescriptorBindingKind kind, std::uint32_t resource) const;

    // The loop guard's limit expression (DriverData), see MslFlowEmitter.cpp.
    std::string LoopGuardLimit();

    // LDS: device memory when it does not fit threadgroup memory (RuntimeAbi's workgroup memory
    // arena, BufferIndex::WorkgroupMemory), else threadgroup memory. The memory flags workgroup
    // barriers need so the LDS accesses around them are ordered.
    [[nodiscard]] bool LdsInDeviceMemory() const;
    // Compute and mesh programs run in threadgroups (workgroup LDS, invocation ids, wave64 at two
    // lanes per invocation); the other graphics stages do not.
    [[nodiscard]] bool ThreadgroupStage() const;
    [[nodiscard]] std::string LdsMemoryFlags() const;
    MslBarrierScope waveLdsScope = MslBarrierScope::None;

    // Debug aid (APS5_MSL_SURVEY=1): failures of the instruction families are collected instead of
    // ending the emission at the first one, and reported together at the end, so one run lists every
    // missing opcode of a shader. Failures of the core (control flow, entry point) still end it.
    bool survey = false;
    std::vector<std::string> surveyFailures;
    [[nodiscard]] std::uint32_t Depth() const { return depth; }
    void RestoreDepth(std::uint32_t value) { depth = value; }

private:
    std::uint32_t depth = 1u;
    std::uint32_t nextTemporary = 0u;
    bool loopGuardDeclared = false;
    std::bitset<static_cast<std::size_t>(MslPreludeChunk::Count)> prelude;
    std::array<std::unordered_set<const IrValue*>, 2> declared;
    std::set<std::string, std::less<>> parameterSet;
};

// A wave64 graphics program at one lane per invocation: each host SIMD group is a 32-lane view of
// the guest wave (MslEmitter.cpp).
[[nodiscard]] bool MslGraphicsHalfWave(const MslEmitterState& state);
[[nodiscard]] std::string MslType(IrType type);
[[nodiscard]] std::string MslHex(std::uint64_t value);
[[nodiscard]] std::string MslTableField(DescriptorBindingKind kind);
// The MSL type of one descriptor table entry of `kind` (MslAbi.hpp §Descriptor table): Aps5Buffer
// for buffer kinds, a texture type fixed by the image heap's class, numeric class and dimension, or
// `sampler`.
[[nodiscard]] std::string MslTableMemberType(DescriptorBindingKind kind);
// `struct Aps5Table` for a descriptor table layout (needs the Common prelude chunk), with static
// assertions that the Metal compiler lays it out as MslAbi::LayoutTable does.
[[nodiscard]] std::string MslTableDeclaration(const MslAbi::TableLayout& layout);

// Tessellation stages (MslTessellation.cpp, MslAbi Version 7): the local, hull and domain programs
// as libraries defining aps5_stage(...) (no stage function), their tessellation attribute accesses
// and builtins (empty: not a tessellation builtin).
[[nodiscard]] bool MslTessellationStage(IrShaderStage stage);
bool EmitMslTessellationInstruction(MslEmitterState& state, const IrValue& instruction);
[[nodiscard]] std::string MslTessellationBuiltin(MslEmitterState& state, const IrValue& instruction, StageInputKind kind, std::uint32_t component);
[[nodiscard]] std::string MslTessellationInvocation(MslEmitterState& state);
void PrepareMslTessellationStage(MslEmitterState& state);
[[nodiscard]] std::string EmitMslTessellationStage(MslEmitterState& state, std::vector<std::string> parameters, bool hasTable);
// Mesh output of an NGG rect list (VGT_PRIMITIVE_TYPE 17, DECISIONS D41): the guest's vertices
// (at most its group's), then one synthesized fourth vertex per rect, and two triangles per rect.
struct MslMeshOutput {
    bool rectList = false;
    std::uint32_t guestVertices = 0u;
    std::uint32_t guestPrimitives = 0u;
    std::uint32_t vertices = 0u;
    std::uint32_t primitives = 0u;
};
[[nodiscard]] MslMeshOutput MslMeshOutputOf(const ShaderMeshInputInfo& mesh);
// The stage function's signature and module-scope declarations (MslEntryPoint.cpp).
[[nodiscard]] std::string EmitMslEntryPoint(MslEmitterState& state);
// Graphics stages (MslStageIo.cpp, B6): the stage-function output/input struct declarations and the
// statements after the guest body (vertex z remap and y flip; fragment discard), for EmitMslEntryPoint.
struct MslGraphicsInterface {
    std::string declarations;
    std::string outputType;   // empty: the function returns void
    std::string attributes;   // before the stage keyword ([[early_fragment_tests]])
    std::string epilogue;
};
[[nodiscard]] MslGraphicsInterface MslGraphicsInterfaceOf(MslEmitterState& state);
// MslAbi Version 5 triangle-vertices helpers (MslStageIo.cpp): defaults of the per-vertex macros, the
// preprocessor condition of the mode, of one per-vertex location, and corner fields (pv<L>_<vertex>).
[[nodiscard]] std::string TriangleVertexMacroDefaults();
[[nodiscard]] std::string TriangleVerticesCondition();
// The window-space position remap (MslAbi::WindowSpaceMacro) of `position`, a float4 lvalue.
[[nodiscard]] std::string WindowSpaceEpilogue(const std::string& position, const std::string& indent);
[[nodiscard]] std::string PerVertexLocationCondition(std::uint32_t location);
// The guest cull distances a vertex program writes (highest index + 1; MslAbi Version 10).
[[nodiscard]] std::uint32_t MslCullDistanceCount(const MslEmitterState& state);
// The clip distance that carries the emulated cull verdict: after the guest clip distances and the
// all-zero position distance (when there is room for it, MslZeroPositionDistance).
[[nodiscard]] std::uint32_t MslCullClipSlot(const MslEmitterState& state);
[[nodiscard]] std::uint32_t MslGuestClipDistanceCount(const MslEmitterState& state);
[[nodiscard]] bool MslZeroPositionDistance(const MslEmitterState& state);
// Upstream 72518f64f/1ffc5f672: a pixel function's PA_SC_AA_MASK coverage of its quad pixel, for
// quad-masked programs (ShaderPixelInputInfo::QuadMasked): the guest samples the pixel covers
// (MslQuadCoverage, 16 bits; all ones without a partial sample mask) and whether the pixel is
// uncovered (MslQuadUncovered, a bool expression over aps5_in and, for a partial sample mask, the
// `uint aps5_sample_mask_in [[sample_mask]]` parameter EmitMslEntryPoint adds).
[[nodiscard]] std::string MslQuadCoverage(const MslEmitterState& state);
[[nodiscard]] std::string MslQuadUncovered(const MslEmitterState& state);
[[nodiscard]] std::string PerVertexField(std::uint32_t location, std::uint32_t vertex);
// The host threadgroup size a workgroup maps to (MslEntryPoint.cpp): the guest workgroup for
// laneCount 1; for laneCount 2 one-dimensional, 32 host invocations per guest wave64.
[[nodiscard]] std::array<std::uint32_t, 3> MslThreadsPerThreadgroup(const ShaderWorkgroupInputInfo& workgroup, std::uint32_t laneCount);
// The stage function's body (MslFlowEmitter.cpp): structured control flow over the IR blocks.
void EmitMslFunctionBody(MslEmitterState& state);
// The statements of a function: the locals, `prologue` and the body (MslEmitter.cpp). An SSA local
// whose first occurrence is its definition statement, with every other occurrence inside that
// statement's block, is declared at the definition instead of zero-initialized at function scope
// (MslSinkDeclarations; APS5_MSL_HOIST_ALL=1 keeps every one hoisted).
[[nodiscard]] std::string MslFunctionStatements(const MslEmitterState& state, const std::string& prologue);
// The declaration sinking on its own: `locals` lines `    T vN = T(0);` (and `vN_h`) whose value
// qualifies lose their declaration, and their definition line `vN = e;` in `body` becomes
// `T vN = e;`. Nothing changes when the text has a preprocessor line. Returns the number sunk.
std::size_t MslSinkDeclarations(std::string& locals, const std::string& prologue, std::string& body);
// The wave-LDS barrier scope of a program (MslFlowEmitter.cpp, the SPIR-V path's WaveLdsScope).
[[nodiscard]] MslBarrierScope MslWaveLdsScope(const MslEmitterState& state);

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
