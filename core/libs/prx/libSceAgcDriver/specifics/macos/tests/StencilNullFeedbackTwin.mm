// Metal twin of tests/execution/StencilNullFeedback.cpp (agc_driver_stencil_null_feedback; upstream e25bb140f9, a
// typed image heap with an empty element must not break a stencil-exporting draw), swapped in by
// twins/UPSYNC3-tests.cmake. The upstream file is compiled here with its main renamed, so its programs, image words,
// PixelInfo and Check keep tracking upstream.
//
// Upstream forces the Vulkan stencil replay (it removes CapabilityStencilExportEXT), counts vkCmdBindPipeline calls,
// and attaches the image heap by patching an OpTypeImage array into the SPIR-V module. On Metal there is no SPIR-V
// module (an MSL target leaves result.spirv empty) and no replay: fragment functions export [[stencil]] natively
// (MslStageIo.cpp StencilRef). This twin's draw is upstream's Draw with the capability kept and the image heap
// attached as the same guest binding (two elements, the second an all-zero null descriptor) without the module
// patch, so the Metal driver resolves the heap, null element included, on the same draws, and the stencil results
// are checked as upstream checks them.
//
// Left out, replay-only: the refusal of a non-null image aliasing the depth surface. Upstream refuses it because the
// replay draws the primitive once per reference and would read its own depth/stencil writes back; a native export
// draws once, so that draw is not a feedback hazard on Metal.

#include "../../../tests/execution/VulkanTestDevice.hpp"

#undef ANYPS5_ENABLE_SPIRV_TOOLS
#define ANYPS5_ENABLE_SPIRV_TOOLS 0
#define main AgcDriverStencilNullFeedbackVulkanMain
#include "../../../tests/execution/StencilNullFeedback.cpp"
#undef main

namespace {

void AttachMetalNullImages(ShaderRecompiler::RecompileResult& result, const ShaderRecompiler::SpirvTarget& target) {
    namespace Runtime = ShaderRecompiler::RuntimeAbi;
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(NullBindingTexture.data()));
    const std::array<std::uint32_t, 8> words{static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u) | (56u << 20u) | (((Width - 1u) & 3u) << 30u), ((Width - 1u) >> 2u) | ((Height - 1u) << 14u), 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
    const auto set = target.registrationModules ? Runtime::FragmentDescriptorSet : Runtime::DescriptorSet;
    const auto binding = Runtime::BindingNumber(Runtime::Stage::Fragment, static_cast<Runtime::Binding>(StorageFloat2DHeap));
    ShaderRecompiler::DescriptorBinding images{ShaderRecompiler::DescriptorKind::StorageImage, ShaderRecompiler::DescriptorRole::GuestImages, set, binding, ImageElements, std::vector<std::uint32_t>(ImageElements * words.size())};
    std::copy(words.begin(), words.end(), images.guestDescriptor.begin());
    images.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
    images.imageWritten = {false, false};
    images.imageAtomic = {false, false};
    images.imageAtomic64 = {false, false};
    result.bindings.push_back(std::move(images));
}

void DrawNative(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> pixelCode, std::uint32_t shaderControl, const VkStencilOpState& stencil, std::uint32_t vertices, std::uint32_t instances, bool nullImages) {
    Pixels.fill(Kept);
    const auto target = device.Target();
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(Vertices.data(), 16u, static_cast<std::uint32_t>(Vertices.size()));
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {WaveSize, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());

    std::vector<std::uint32_t> pixelUserData(8, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(pixelCode.data()), std::as_bytes(pixelCode)}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(pixelCode.data()), pixelCode, 0, {}},
        {WaveSize, 0, pixelUserData, std::nullopt, PixelInfo(shaderControl), std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    fragment.useCache = false;
    auto pixelResult = ShaderRecompiler::Recompile(fragment);
    Require(pixelResult.stencilReferences.empty(), "the Metal target compiled a stencil replay although it exports stencil natively");
    if (nullImages) AttachMetalNullImages(pixelResult, target);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, &pixelResult, PixelPushOffset(vertexPush, target)}
    }};

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, WaveSize, WaveSize, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.depth = AgcDriver::Graphics::DepthTarget{reinterpret_cast<std::uintptr_t>(DepthMemory.data()), reinterpret_cast<std::uintptr_t>(StencilMemory.data()), {Width, Height}, VK_FORMAT_D32_SFLOAT_S8_UINT, 1.0f, 0};
    state.depthTest = false;
    state.depthCompare = VK_COMPARE_OP_ALWAYS;
    state.stencilTest = true;
    state.stencilFront = stencil;
    state.stencilBack = stencil;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, vertices, 0, instances, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        // Upstream's bind trace and SPIR-V heap patch have no Metal input; they stay compiled for its main.
        (void)&CountBind;
        constexpr std::uint32_t exported = 0x52u;
        constexpr std::uint32_t plain = 0x50u;
        constexpr std::array<std::uint32_t, 12> alternating{
            0x7e040f00u, 0x36060481u,
            0x7c000000u | (0xc2u << 17u) | (3u << 9u) | 0x80u,
            0x7e0802ffu, 0xffu,
            (1u << 25u) | (5u << 17u) | (4u << 9u) | 0x80u,
            0x7e0e02f2u, 0xf8000082u, 5u << 8u,
            0xf800180fu, 0x07070707u, 0xbf810000u
        };
        const auto replace = [](std::uint32_t reference = 0u, std::uint32_t mask = 0xffu) { return VkStencilOpState{VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xffu, mask, reference}; };
        const auto equal = [](std::uint32_t reference, std::uint32_t mask = 0xffu) { return VkStencilOpState{VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL, mask, 0u, reference}; };
        const auto all = [](std::uint32_t, std::uint32_t) { return true; };
        const auto even = [](std::uint32_t x, std::uint32_t) { return x % 2u == 0u; };
        const auto odd = [](std::uint32_t x, std::uint32_t) { return x % 2u != 0u; };
        DrawNative(*device, PlainPixelCode, plain, replace(), 3u, 1u, true);
        Check(all, "an empty typed image element remains supported without stencil export");
        DrawNative(*device, alternating, exported, replace(), 6u, 2u, true);
        Check(all, "an empty typed image element preserves stencil export colors");
        DrawNative(*device, PlainPixelCode, plain, equal(255u), 3u, 1u, false);
        Check(even, "an empty typed image element preserves the even stencil reference");
        DrawNative(*device, PlainPixelCode, plain, equal(0u), 3u, 1u, false);
        Check(odd, "an empty typed image element preserves the odd stencil reference");
        std::puts("stencil null feedback tests passed (native [[stencil]] export)");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
