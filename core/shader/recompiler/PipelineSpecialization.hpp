#ifndef CORE_SHADER_RECOMPILER_PIPELINESPECIALIZATION_HPP
#define CORE_SHADER_RECOMPILER_PIPELINESPECIALIZATION_HPP

#include <cstdint>

namespace ShaderRecompiler {

struct PipelineSpecializationConstant {
    std::uint32_t id;
    std::uint32_t value;

    bool operator==(const PipelineSpecializationConstant&) const = default;
};

namespace PipelineSpecialization {

inline constexpr std::uint32_t CompareBase = 49152u;
inline constexpr std::uint32_t CompareWords = 6u;
inline constexpr std::uint32_t BufferBase = 0u;
inline constexpr std::uint32_t BufferWords = 4u;
inline constexpr std::uint32_t ImageBase = 1024u;
inline constexpr std::uint32_t ImageWords = 5u;
inline constexpr std::uint32_t VertexBase = 4096u;
inline constexpr std::uint32_t VertexWords = 6u;
inline constexpr std::uint32_t PushDataOffset = 8188u;
inline constexpr std::uint32_t ExportBase = 8192u;
inline constexpr std::uint32_t ExportPackingBase = 8224u;
inline constexpr std::uint32_t DualSourceBlend = 8232u;
inline constexpr std::uint32_t RasterPrimitive = 8233u;
inline constexpr std::uint32_t LayerExported = 8234u;
inline constexpr std::uint32_t StencilReferenceOffset = 8235u;
inline constexpr std::uint32_t ReversedBlend = 8236u;
inline constexpr std::uint32_t HeapCountBase = 8256u;
inline constexpr std::uint32_t MipCountBase = 8320u;
// Multisampled storage images on MSL backends (MslAbi Version 6): Metal shaders cannot write
// multisampled textures, so a storage view of a 2D MSAA (array) image is a 2D-array texture of
// layers * samples slices (slice = layer * samples + sample). Per direct image resource i, the sample
// count (1 << T# word 3 LAST_LEVEL) is constant SampleCountBase + i; SPIR-V modules have no such
// constants.
inline constexpr std::uint32_t SampleCountBase = 12288u;
inline constexpr std::uint32_t DescriptorIndexBase = 16384u;
inline constexpr std::uint32_t DescriptorIndexStride = 128u;
inline constexpr std::uint32_t FoldOffsetBase = 57344u;
// Unnormalized samplers that take the base level for implicit-LOD sampling, per sampler s at
// UnnormalizedBaseLevelBase + s (upstream #1418).
inline constexpr std::uint32_t UnnormalizedBaseLevelBase = 57408u;
// Min/max reduction samplers (Metal ignores MTLSamplerDescriptor.reductionMode): per sampler s,
// word 0 = reduction | mip filter << 2 | min LOD raw << 4 | max LOD raw << 16 and word 1 = the raw
// LOD bias | XY mag filter << 16 | XY min filter << 18, at ReductionBase + s * ReductionWords; supplied only for reducing samplers (MSL
// backends emulate the reduction, SPIR-V modules have no such constants).
inline constexpr std::uint32_t ReductionBase = 57472u;
inline constexpr std::uint32_t ReductionWords = 2u;

inline constexpr std::uint32_t DescriptorIndex(std::uint32_t binding, std::uint32_t element) {
    return DescriptorIndexBase + binding * DescriptorIndexStride + element;
}

}

}

#endif
