#include "SceShaders.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparationScope.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"
#include "prx/libSceAgc/Shader/include/ShaderUtils.hpp"
#include <array>
#include <algorithm>

extern "C" int APS5_VABI sceAgcCreatePrimState(ShaderRegister*, ShaderRegister*, const Shader*, const Shader*, std::uint32_t);
extern "C" int APS5_VABI sceAgcCreateInterpolantMapping_0100(ShaderRegister*, const Shader*, const Shader*);

extern "C" int APS5_VABI sceAgcLinkShaders(ShaderRegister* context, ShaderRegister* primitive,
    const void* reserved, const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType) {
    using namespace ShaderRegs;
    if (!context || !primitive || reserved || !vertex) return GRAPHICS5_ERROR_INVALID_SHADER_PROGRAM;
    // Validate through the headers' host views (displaced memory, D31, D33); the helpers and the driver get the guest headers.
    const Shader vertexView = HostShader(vertex);
    Shader pixelView{};
    if (pixel) pixelView = HostShader(pixel);
    if (vertexView.type != static_cast<std::uint8_t>(ShaderBinaryType::Gs) ||
        (vertexView.num_output_semantics && !vertexView.output_semantics) ||
        (pixel && (pixelView.type != static_cast<std::uint8_t>(ShaderBinaryType::Ps) ||
                   pixelView.num_input_semantics > 32 ||
                   (pixelView.num_input_semantics && !pixelView.input_semantics))))
        return GRAPHICS5_ERROR_INVALID_SHADER_PROGRAM;

    // The simple vertex/pixel pipeline combines two primitive context registers
    // and 32 interpolant registers. Publish only after both helpers succeed.
    AgcDriver::ShaderPreparationScope transaction;
    std::array<ShaderRegister, 34> contextValues{};
    std::array<ShaderRegister, 3> primitiveValues{};
    sceAgcCreatePrimState(contextValues.data(), primitiveValues.data(), nullptr, vertex, primitiveType);
    sceAgcCreateInterpolantMapping_0100(contextValues.data() + 2, vertex, pixel);
    if (pixel != nullptr) {
        const std::array<const Shader*, 2> stages{vertex, pixel};
        AgcDriverResolveGraphicsStagesAbi_nid_postfix(stages, contextValues, primitiveValues);
    }
    AgcDriverResolveGraphicsAbi_nid_postfix(vertex, pixel, primitiveType);
    transaction.Commit();
    std::copy(contextValues.begin(), contextValues.end(), GuestArena::HostPointer(context));
    std::copy(primitiveValues.begin(), primitiveValues.end(), GuestArena::HostPointer(primitive));
    return 0;
}
