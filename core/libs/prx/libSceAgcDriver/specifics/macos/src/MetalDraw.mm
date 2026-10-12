// Draw members and graphics feature queries of the Metal device (work package D6,
// docs/mac/M2_DESIGN.md): render passes rebuilt from PM4 state, render pipelines, attachments.

#include "MetalDeviceState.h"
#include <optional>
#include <string>

namespace AgcDriver {

using Metal::Unimplemented;

std::optional<ShaderRecompiler::GeometryStageLimits> VulkanDevice::GeometryLimits() const {
    return std::nullopt;
}

bool VulkanDevice::DualSrcBlend() const {
    return true;
}

bool VulkanDevice::PrimitiveListRestart() const {
    return false;
}

bool VulkanDevice::SamplerFilterMinmax() const {
    // MTLSamplerDescriptor.reductionMode (min/max filtering) exists on macOS 26+ (D5's samplers).
    return true;
}

bool VulkanDevice::ConservativeRasterization() const {
    return false;
}

VkShaderStageFlags VulkanDevice::SubgroupStages() const {
    // SIMD-group functions in vertex functions are undocumented: D1's startup self-test decides.
    const bool vertex = state->queue != nil && Metal::Profile(state->device).vertexSimdGroups;
    return VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | (vertex ? VK_SHADER_STAGE_VERTEX_BIT : 0u);
}

bool VulkanDevice::ProvokingVertexLast() const {
    return false;
}

bool VulkanDevice::GraphicsPipelineLibraries() const {
    return false;
}

void VulkanDevice::NoteDrawRecipeMiss(DrawRecipePrecheck) {
}

void VulkanDevice::Draw(const Graphics::State&, const Pm4::DrawParameters&, std::span<const Graphics::CompiledShader>, std::span<const Graphics::GuestMemorySnapshot>, std::shared_ptr<const DrawRecipe>*) {
    Unimplemented("Draw", "work package D5");
}

std::optional<std::string> VulkanDevice::KnownDrawRejection(const Graphics::State&, std::span<const Graphics::CompiledShader>) const {
    return std::string("draws are not implemented on the Metal device yet (work package D5)");
}

void VulkanDevice::ColorMetadataPass(const Graphics::ColorMetadataPass&) {
    Unimplemented("ColorMetadataPass", "work package D4");
}

RecipeOutcome VulkanDevice::DrawFromRecipe(const Graphics::State&, const Pm4::DrawParameters&, std::span<const Graphics::CompiledShader>, std::span<const Graphics::GuestMemorySnapshot>, const std::shared_ptr<const DrawRecipe>&) {
    return RecipeOutcome::Rebuild;
}

VulkanDevice::IndirectDrawSupport VulkanDevice::DrawIndirectSupport() const {
    return {false, false, false};
}

bool VulkanDevice::PrebuildGraphicsLibraries(const Graphics::ShaderStages&, std::span<const Graphics::CompiledShader>) {
    return false;
}

}
