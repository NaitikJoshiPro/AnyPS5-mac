#ifndef CORE_SHADER_RECOMPILIER_INTERMEDIATEREPRESENTATION_INCLUDE_INTERMEDIATEREPRESENTATION_IRMETADATA_STAGEIO_HPP
#define CORE_SHADER_RECOMPILIER_INTERMEDIATEREPRESENTATION_INCLUDE_INTERMEDIATEREPRESENTATION_IRMETADATA_STAGEIO_HPP

#include <cstdint>
#include <limits>
#include <string>

namespace ShaderRecompiler {

enum class TessellationAttribute {
    LocalOutput,
    ControlInput,
    ControlOutput,
    EvaluationInput,
    PatchOutput,
    Factor,
    PatchInput,
    InstanceOutput,
    InstanceInput,
    ControlShared
};

enum class TessellationBaseKind {
    OffChip,
    Factor,
    RelativeIndex,
    PassthroughPrimitive
};

enum class StageInputKind {
    VertexIndex,
    InvocationId,
    PrimitiveId,
    TessCoord,
    InstanceIndex,
    FragCoord,
    FrontFacing,
    PackedAncillary,
    Layer,
    SampleId,
    BaryCoordSmooth,
    BaryCoordNoPerspective,
    WorkgroupId,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    DispatchThreadLimit,
    Parameter,
    HelperInvocation,
    RasterPrimitive,
    LayerExported,
    BaseInstance,
    BaryCoordSmoothCenter,
    BaryCoordNoPerspectiveCenter,
    BaryCoordSmoothSample,
    BaryCoordNoPerspectiveSample,
    SampleMask,
};

// The I/J pair (SPI_PS_INPUT_ADDR PERSP/LINEAR_CENTER, _CENTROID, _SAMPLE) one interpolated pixel
// read uses, carried per instruction (DECISIONS D38) in the IrValue flags of pixel GetAttribute
// values (host interpolation) and of BaryCoordSmooth/BaryCoordNoPerspective GetBuiltin values
// (emulated barycentrics) as InterpolationLocationFlag | location. Flags without the bit: the pixel
// center for barycentrics, the program's per-input metadata for attributes.
enum class InterpolationLocation : std::uint32_t {
    Center = 0,
    Centroid = 1,
    Sample = 2,
};
inline constexpr std::uint32_t InterpolationLocationFlag = 0x100u;

inline constexpr std::uint32_t InterpolationLocationFlags(InterpolationLocation location) {
    return InterpolationLocationFlag | static_cast<std::uint32_t>(location);
}

constexpr StageInputKind BarycentricInputOf(StageInputKind kind) {
    switch (kind) {
    case StageInputKind::BaryCoordSmoothCenter:
    case StageInputKind::BaryCoordSmoothSample:
        return StageInputKind::BaryCoordSmooth;
    case StageInputKind::BaryCoordNoPerspectiveCenter:
    case StageInputKind::BaryCoordNoPerspectiveSample:
        return StageInputKind::BaryCoordNoPerspective;
    default:
        return kind;
    }
}

constexpr bool IsBarycentricInput(StageInputKind kind) {
    const auto input = BarycentricInputOf(kind);
    return input == StageInputKind::BaryCoordSmooth || input == StageInputKind::BaryCoordNoPerspective;
}

enum class StageOutputKind {
    Position,
    Parameter,
    Mrt,
    Depth,
    SampleMask,
    StencilRef,
    PointSize,
    ClipDistance,
    CullDistance,
    Layer,
    ViewportIndex
};

struct PositionExportComponent {
    std::uint32_t clipDistance = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t cullDistance = std::numeric_limits<std::uint32_t>::max();
    bool pointSize = false;
    bool layer = false;
    bool viewport = false;
};

struct StageInput {
    StageInputKind kind = StageInputKind::VertexIndex;
    std::uint32_t location = 0;
    std::uint32_t componentCount = 1;
    std::string debugName;
    bool perVertex = false;

    bool operator==(const StageInput& other) const = default;
};

struct StageOutput {
    StageOutputKind kind = StageOutputKind::Parameter;
    std::uint32_t index = 0;
    std::uint32_t location = 0;
    std::string debugName;

    bool operator==(const StageOutput& other) const = default;
};

}

#endif
