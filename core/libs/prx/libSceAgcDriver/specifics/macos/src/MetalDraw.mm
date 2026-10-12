// Draw members and graphics feature queries of the Metal device (work package D6,
// docs/mac/M2_DESIGN.md "DRAW results"): the bring-up draw path, new code beside the device like
// D1's compute path rather than a twin of Graphics/src/Draw.cpp (which stays compiled and
// unreachable on Metal). A draw records into the recorder's open batch:
// - bindings: one MslAbi descriptor table per stage (MetalDevice.mm's StageBindings), the shared
//   256-byte push block (vertex push constants at their offset, pixel ones at theirs), DriverData,
//   and for vertex pulling the VertexFetch records (B6/D6 interface);
// - targets: linear color targets are textures over the guest memory itself (no copies); tiled
//   ones are private textures loaded from guest memory (CPU detile, ColorTargetLayout) when the
//   pass begins and written back (CPU retile) when the batch completes; depth/stencil surfaces are
//   GPU-only, as upstream's DepthSurface (never written back to guest memory);
// - passes: the recorder's StartDrawPass/LeaveRenderPassOpen keep one render encoder open across
//   draws with the same attachments (load/store actions Load/Store: correctness first);
// - pipelines and depth-stencil states from D5's PipelineCache; viewport, scissor, cull, winding,
//   depth bias, stencil references and blend constants as dynamic encoder state;
// - primitives: lists and strips draw directly, triangle fans become index lists (Vulkan order),
//   rect lists use B6's vertex-ID expansion (6 host vertices per rect); indirect draws read their
//   records on the CPU (DrawIndirectSupport reports no GPU path yet).
// Unsupported states throw std::runtime_error naming D6 (docs/dev/TechnicalDebt.md metal-d6-*).

#include "MetalDeviceState.h"
#include "MetalCommandStream.h"
#include "MetalPipelineCache.h"
#include "MetalImages.h"
#include "MetalVulkanImages.h"
#include "MetalDepthSurfaces.h"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "MslAbi.hpp"
#include "PipelineSpecialization.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver {

using Metal::Unimplemented;

namespace {

namespace Abi = ShaderRecompiler::MslAbi;
using Graphics::Recorder;

[[noreturn]] void Fail(const std::string& what) {
    throw std::runtime_error("Metal draw (work package D6): " + what);
}

// Debug aid (as upstream): APS5_NO_RESIDENT_TARGETS=1 keeps tiled targets out of the image cache
// (the bring-up load/write-back path).
bool ResidentTargets() {
    static const bool resident = std::getenv("APS5_NO_RESIDENT_TARGETS") == nullptr;
    return resident;
}

bool Trace() {
    static const bool trace = std::getenv("APS5_METAL_TRACE") != nullptr;
    return trace;
}

MTLBlendFactor BlendFactor(VkBlendFactor factor) {
    switch (factor) {
    case VK_BLEND_FACTOR_ZERO: return MTLBlendFactorZero;
    case VK_BLEND_FACTOR_ONE: return MTLBlendFactorOne;
    case VK_BLEND_FACTOR_SRC_COLOR: return MTLBlendFactorSourceColor;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
    case VK_BLEND_FACTOR_DST_COLOR: return MTLBlendFactorDestinationColor;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
    case VK_BLEND_FACTOR_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
    case VK_BLEND_FACTOR_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
    case VK_BLEND_FACTOR_CONSTANT_COLOR: return MTLBlendFactorBlendColor;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return MTLBlendFactorOneMinusBlendColor;
    case VK_BLEND_FACTOR_CONSTANT_ALPHA: return MTLBlendFactorBlendAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return MTLBlendFactorOneMinusBlendAlpha;
    case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return MTLBlendFactorSourceAlphaSaturated;
    case VK_BLEND_FACTOR_SRC1_COLOR: return MTLBlendFactorSource1Color;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: return MTLBlendFactorOneMinusSource1Color;
    case VK_BLEND_FACTOR_SRC1_ALPHA: return MTLBlendFactorSource1Alpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: return MTLBlendFactorOneMinusSource1Alpha;
    default: Fail("blend factor " + std::to_string(factor) + " has no Metal counterpart");
    }
}

MTLBlendOperation BlendOperation(VkBlendOp operation) {
    switch (operation) {
    case VK_BLEND_OP_ADD: return MTLBlendOperationAdd;
    case VK_BLEND_OP_SUBTRACT: return MTLBlendOperationSubtract;
    case VK_BLEND_OP_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
    case VK_BLEND_OP_MIN: return MTLBlendOperationMin;
    case VK_BLEND_OP_MAX: return MTLBlendOperationMax;
    default: Fail("blend operation " + std::to_string(operation) + " has no Metal counterpart");
    }
}

MTLColorWriteMask WriteMask(VkColorComponentFlags mask) {
    MTLColorWriteMask result = MTLColorWriteMaskNone;
    if (mask & VK_COLOR_COMPONENT_R_BIT) result |= MTLColorWriteMaskRed;
    if (mask & VK_COLOR_COMPONENT_G_BIT) result |= MTLColorWriteMaskGreen;
    if (mask & VK_COLOR_COMPONENT_B_BIT) result |= MTLColorWriteMaskBlue;
    if (mask & VK_COLOR_COMPONENT_A_BIT) result |= MTLColorWriteMaskAlpha;
    return result;
}

MTLCompareFunction Compare(VkCompareOp operation) {
    switch (operation) {
    case VK_COMPARE_OP_NEVER: return MTLCompareFunctionNever;
    case VK_COMPARE_OP_LESS: return MTLCompareFunctionLess;
    case VK_COMPARE_OP_EQUAL: return MTLCompareFunctionEqual;
    case VK_COMPARE_OP_LESS_OR_EQUAL: return MTLCompareFunctionLessEqual;
    case VK_COMPARE_OP_GREATER: return MTLCompareFunctionGreater;
    case VK_COMPARE_OP_NOT_EQUAL: return MTLCompareFunctionNotEqual;
    case VK_COMPARE_OP_GREATER_OR_EQUAL: return MTLCompareFunctionGreaterEqual;
    case VK_COMPARE_OP_ALWAYS: return MTLCompareFunctionAlways;
    default: Fail("compare operation " + std::to_string(operation));
    }
}

MTLStencilOperation StencilOperation(VkStencilOp operation) {
    switch (operation) {
    case VK_STENCIL_OP_KEEP: return MTLStencilOperationKeep;
    case VK_STENCIL_OP_ZERO: return MTLStencilOperationZero;
    case VK_STENCIL_OP_REPLACE: return MTLStencilOperationReplace;
    case VK_STENCIL_OP_INCREMENT_AND_CLAMP: return MTLStencilOperationIncrementClamp;
    case VK_STENCIL_OP_DECREMENT_AND_CLAMP: return MTLStencilOperationDecrementClamp;
    case VK_STENCIL_OP_INVERT: return MTLStencilOperationInvert;
    case VK_STENCIL_OP_INCREMENT_AND_WRAP: return MTLStencilOperationIncrementWrap;
    case VK_STENCIL_OP_DECREMENT_AND_WRAP: return MTLStencilOperationDecrementWrap;
    default: Fail("stencil operation " + std::to_string(operation));
    }
}

Graphics::Metal::StencilFaceState StencilFace(const VkStencilOpState& face) {
    Graphics::Metal::StencilFaceState result;
    result.compare = Compare(face.compareOp);
    result.fail = StencilOperation(face.failOp);
    result.depthFail = StencilOperation(face.depthFailOp);
    result.pass = StencilOperation(face.passOp);
    result.readMask = face.compareMask & 0xffu;
    result.writeMask = face.writeMask & 0xffu;
    return result;
}

// How a Vulkan topology draws on Metal.
struct Primitive {
    MTLPrimitiveType type;
    MTLPrimitiveTopologyClass topologyClass;
    // Triangle fans: index lists (i + 1, i + 2, 0) per triangle, Vulkan's vertex order (first
    // vertex provoking).
    bool fan = false;
};

Primitive PrimitiveOf(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return {MTLPrimitiveTypePoint, MTLPrimitiveTopologyClassPoint};
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return {MTLPrimitiveTypeLine, MTLPrimitiveTopologyClassLine};
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return {MTLPrimitiveTypeLineStrip, MTLPrimitiveTopologyClassLine};
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return {MTLPrimitiveTypeTriangle, MTLPrimitiveTopologyClassTriangle};
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: return {MTLPrimitiveTypeTriangleStrip, MTLPrimitiveTopologyClassTriangle};
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN: return {MTLPrimitiveTypeTriangle, MTLPrimitiveTopologyClassTriangle, true};
    default: Fail("primitive topology " + std::to_string(topology) + " (adjacency and patch lists need the geometry or tessellation path)");
    }
}

MTLPixelFormat DepthFormat(const Graphics::DepthTarget& target) {
    const bool stencil = target.stencilAddress != 0u;
    switch (target.format) {
    case VK_FORMAT_D32_SFLOAT: return stencil ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth32Float;
    case VK_FORMAT_D16_UNORM: return stencil ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth16Unorm;
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D16_UNORM_S8_UINT: return MTLPixelFormatDepth32Float_Stencil8;
    case VK_FORMAT_S8_UINT: return MTLPixelFormatStencil8;
    default: Fail("depth format " + std::to_string(target.format));
    }
}

// The MSL type a fragment function writes to a color attachment of `format` (B6/D6 interface:
// APS5_COLOR<n>_TYPE): uint4/int4 for integer formats, float4 otherwise.
const char* ColorOutputType(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Uint: case MTLPixelFormatRG8Uint: case MTLPixelFormatRGBA8Uint: case MTLPixelFormatR16Uint: case MTLPixelFormatRG16Uint: case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatR32Uint: case MTLPixelFormatRG32Uint: case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGB10A2Uint:
        return "uint4";
    case MTLPixelFormatR8Sint: case MTLPixelFormatRG8Sint: case MTLPixelFormatRGBA8Sint: case MTLPixelFormatR16Sint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatR32Sint: case MTLPixelFormatRG32Sint: case MTLPixelFormatRGBA32Sint:
        return "int4";
    default:
        return "float4";
    }
}

bool HasStencil(MTLPixelFormat format) {
    return format == MTLPixelFormatDepth32Float_Stencil8 || format == MTLPixelFormatStencil8;
}

bool HasDepth(MTLPixelFormat format) {
    return format != MTLPixelFormatStencil8;
}

// Objects a recorded draw's batch keeps alive (Metal 4 command buffers retain nothing).
struct Held {
    std::vector<id> objects;
};

}

namespace Metal {

// The device's draw objects (one per device, made at the first draw, destroyed by Teardown after
// the recorder is idle and before the image cache, whose residency set holds the textures).
struct DrawCache {
    id<MTLDevice> device = nil;
    Graphics::Metal::ImageCache* images = nullptr;
    id<MTL4ArgumentTable> vertexArguments = nil;
    id<MTL4ArgumentTable> fragmentArguments = nil;

    // The depth/stencil surface of one draw (DepthSurface.mm's registry holds the surfaces: GPU-only,
    // cleared at creation and after HTILE fast clears, never written back to guest memory).
    struct DepthSurface {
        Graphics::DepthTarget target;
        MTLPixelFormat format;
        id<MTLTexture> texture;
        bool clearDepth = false;
        bool clearStencil = false;
    };

    // Private textures of tiled color targets. `gpuNewer`: passes whose write-back has not landed
    // yet (the texture then holds the newest contents and is not reloaded from guest memory).
    struct TiledTarget {
        std::uint64_t address;
        VkExtent2D extent;
        VkFormat format;
        Graphics::ColorTileMode tileMode;
        std::uint32_t elementBytes;
        std::uint32_t pipeBankXor;
        id<MTLTexture> texture;
        std::uint32_t gpuNewer = 0;
    };
    std::vector<std::shared_ptr<TiledTarget>> tiled;

    // CMASK codes other than 0 (cleared) and 15 (expanded) a materialization met (bit per code),
    // reported by the next one (as upstream's detiler reports its GPU pass's codes).
    std::uint32_t cmaskErrors = 0;

    // The attachments of the render pass the recorder holds open (valid while its key matches).
    std::uint64_t passKey = 0;
    std::vector<id<MTLTexture>> passTextures;

    ~DrawCache() {
        std::vector<id<MTLTexture>> textures = Graphics::Metal::DepthSurfaceTextures(device);
        Graphics::ClearDepthSurfaces(Graphics::Metal::DeviceHandle(device));
        for (const auto& target : tiled) textures.push_back(target->texture);
        if (images != nullptr && !textures.empty()) images->Release(textures);
    }
};

}

namespace {

using DrawCache = Metal::DrawCache;

}

Metal::DrawCache& VulkanDevice::State::Draws() {
    auto& state = *this;
    if (state.draws == nullptr) {
        auto cache = std::make_shared<DrawCache>();
        cache->device = state.device;
        cache->images = state.images.get();
        MTL4ArgumentTableDescriptor* table = [MTL4ArgumentTableDescriptor new];
        table.maxBufferBindCount = Abi::Index(Abi::BufferIndex::VertexFetch) + 1u;
        NSError* error = nil;
        cache->vertexArguments = [state.device newArgumentTableWithDescriptor:table error:&error];
        cache->fragmentArguments = [state.device newArgumentTableWithDescriptor:table error:&error];
        if (cache->vertexArguments == nil || cache->fragmentArguments == nil) Fail("cannot create the draw argument tables: " + Metal::Describe(error));
        state.draws = std::move(cache);
    }
    return *state.draws;
}

namespace {

// One color attachment of a draw.
struct ColorAttachment {
    const Graphics::ColorTarget* target;
    MTLPixelFormat format;
    std::uint64_t begin;
    std::uint64_t end;
    // Linear targets: a texture over the guest memory (made per pass). Tiled: the image cache's
    // resident image of the surface (upstream image path), else the bring-up cache entry.
    std::shared_ptr<DrawCache::TiledTarget> tiled;
    bool load = false;
    std::shared_ptr<Graphics::StorageTexture> resident;
};

// The color buffer as a single-mip 2D surface descriptor (upstream Draw.cpp SurfaceForTarget).
Graphics::GuestTextureResource SurfaceForTarget(const Graphics::ColorTarget& color) {
    const bool chain = color.mipCount > 1;
    Graphics::GuestTextureResource surface {};
    surface.baseAddress = chain ? color.surfaceAddress : color.address;
    surface.pipeBankXor = color.pipeBankXor;
    surface.width = chain ? color.surfaceExtent.width : color.extent.width;
    surface.height = chain ? color.surfaceExtent.height : color.extent.height;
    surface.depthOrLastArray = color.depth - 1u;
    surface.baseArray = 0;
    surface.mipCount = color.mipCount;
    surface.baseLevel = 0;
    surface.lastLevel = color.mipCount - 1;
    surface.tileMode = Graphics::ColorTextureTileMode(color.tileMode);
    surface.dimension = color.depth > 1 ? Graphics::TextureDimension::k3D : Graphics::TextureDimension::k2D;
    const auto guest = Graphics::FindGuestColorTargetFormat(color.format, color.elementBytes);
    if (!guest) Fail("no guest texture format matches the color buffer format " + std::to_string(static_cast<int>(color.format)));
    surface.format = *guest;
    surface.dstSelX = 4;
    surface.dstSelY = 5;
    surface.dstSelZ = 6;
    surface.dstSelW = 7;
    surface.dccAddress = color.dccAddress;
    surface.dccAlphaOnMsb = color.dccAlphaOnMsb;
    surface.dccPipeAligned = color.dccPipeAligned;
    return surface;
}

// A slice of a 2D array color target as the whole array (upstream ArraySurfaceForTarget).
Graphics::GuestTextureResource ArraySurfaceForTarget(const Graphics::ColorTarget& color) {
    auto surface = SurfaceForTarget(color);
    surface.baseAddress = color.arrayAddress;
    surface.width = color.surfaceExtent.width;
    surface.height = color.surfaceExtent.height;
    surface.depthOrLastArray = color.arrayLastSlice;
    surface.dimension = Graphics::TextureDimension::k2DArray;
    return surface;
}

std::uint64_t Mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    return hash;
}

struct DrawCall {
    std::uint32_t count;
    std::uint32_t instances;
    std::uint32_t first;
    std::int32_t vertexOffset;
    std::uint32_t firstInstance;
};

}

void VulkanDevice::State::SettleRange(std::uint64_t address, std::uint64_t bytes) {
    auto& state = *this;
    if (state.recorder->PendingWriteOverlaps(address, bytes) || state.recorder->PendingLabelIn(address, bytes)) {
        Recorder::CountSync(4);
        state.recorder->SyncThrough(address, bytes);
    }
}

// The GPU buffer and offset holding guest memory [address, address + bytes) (wrapped in place).
// CMASK fast clears (upstream Draw.cpp materializeCmaskClear, CPU path): the 8x8 tiles whose CMASK
// code is 0 (fast-cleared) get the clear register's texel in guest memory, and the CMASK is left
// expanded (all 0xff), before a draw or a CB fast-clear eliminate reads the target. Codes other than
// 0 and 15 are recorded and reported by the next materialization, as upstream reports its GPU
// pass's codes.
void VulkanDevice::State::MaterializeCmask(const Graphics::ColorTarget& color, Graphics::StorageTexture* resident) {
    auto& cache = Draws();
    if (const auto codes = std::exchange(cache.cmaskErrors, 0u); codes != 0u) {
        if ((codes & 1u) != 0u) Fail("a CMASK fast clear of a DCC color target, whose clear belongs in its DCC keys, was recorded");
        std::string listed;
        for (std::uint32_t code = 1; code < 15; ++code) {
            if ((codes >> code) & 1u) listed += (listed.empty() ? "" : ", ") + std::to_string(code);
        }
        Fail("CMASK codes " + listed + " were recorded, which are not a single-sample fast-clear code (0 cleared, 15 expanded)");
    }
    if (color.cmaskAddress == 0u) return;
    const Graphics::CmaskLayout layout(color.extent.width, color.extent.height);
    const auto bytes = layout.Bytes();
    if (color.cmaskBytes != bytes) Fail("CMASK size differs from the color target's CMASK layout");
    SettleRange(color.cmaskAddress, bytes);
    const auto* mask = reinterpret_cast<const std::uint8_t*>(color.cmaskAddress);
    if (std::all_of(mask, mask + bytes, [](std::uint8_t value) { return value == 0xffu; })) return;
    std::vector<std::uint32_t> cleared;
    for (std::uint32_t tileY = 0; tileY < layout.TilesY(); ++tileY) {
        for (std::uint32_t tileX = 0; tileX < layout.TilesX(); ++tileX) {
            const auto nibble = layout.Nibble(tileX, tileY);
            const auto code = (static_cast<std::uint32_t>(mask[nibble / 2u]) >> ((nibble % 2u) * 4u)) & 0xfu;
            if (code == 0u) cleared.push_back(tileY * layout.TilesX() + tileX);
            else if (code != 0xfu) cache.cmaskErrors |= 1u << code;
        }
    }
    if (!cleared.empty()) {
        if (color.dccAddress != 0u) Fail("CMASK fast clear of a DCC color target, whose clear belongs in its DCC keys");
        if (color.elementBytes == 0u || color.elementBytes > sizeof(color.clearWords)) Fail("the CMASK register clear of a texel over 64 bits is not modeled");
        // The texels are patched in guest memory (upstream storeCmaskClearTexels): pending
        // resident-image results and recorded GPU work over the target (and a tiled target's
        // pending write-back) land first.
        Graphics::StorageTexture::FlushPending(color.address, color.bytes, nullptr, "fast-clear materialization");
        if (recorder->PendingWriteOverlaps(color.address, color.bytes)) {
            Recorder::CountSync(2);
            recorder->SyncThrough(color.address, color.bytes);
        }
        const Graphics::ColorTargetLayout surface(color.extent.width, color.extent.height, color.tileMode, color.elementBytes, color.pipeBankXor);
        std::vector<std::byte> texels(surface.Bytes());
        GuestMemory::Read(color.address, texels, surface.Alignment());
        for (const auto tile : cleared) {
            const auto x0 = (tile % layout.TilesX()) * 8u;
            const auto y0 = (tile / layout.TilesX()) * 8u;
            for (auto y = y0; y < std::min(y0 + 8u, color.extent.height); ++y) {
                for (auto x = x0; x < std::min(x0 + 8u, color.extent.width); ++x) std::memcpy(texels.data() + surface.Offset(x, y), color.clearWords.data(), color.elementBytes);
            }
        }
        GuestMemory::Write(color.address, texels, surface.Alignment());
        if (resident != nullptr) resident->Refresh();
    }
    const std::vector<std::byte> expanded(bytes, std::byte{0xff});
    GuestMemory::Write(color.cmaskAddress, expanded, Graphics::CmaskLayout::Alignment);
}

id<MTLBuffer> VulkanDevice::State::BufferAt(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset) {
    auto& state = *this;
    state.Wrap(address, bytes);
    if (id<MTLBuffer> import = state.ImportOf(address, bytes, offset)) return import;
    if (id<MTLBuffer> wrapper = state.WrapperOf(address, bytes, offset)) return wrapper;
    Fail("guest memory at " + std::to_string(address) + " has no GPU mapping");
}

void VulkanDevice::State::RecordDraw(const Graphics::State& graphics, const Pm4::DrawParameters& draw, const Graphics::CompiledShader& vertex, const Graphics::CompiledShader* fragment, std::span<const Graphics::GuestMemorySnapshot> snapshots) {
    auto& state = *this;
    auto& cache = Draws();
    auto& recorder = *state.recorder;
    // A batch over its kept-byte budget is submitted first (upstream's Draw does the same).
    recorder.BoundKeptBytes();
    const auto& vertexResult = *vertex.program;
    const auto& vertexReflection = vertexResult.msl.Reflection();
    const auto* fragmentResult = fragment != nullptr ? fragment->program : nullptr;
    const Abi::Reflection* fragmentReflection = fragmentResult != nullptr ? &fragmentResult->msl.Reflection() : nullptr;
    if (vertexReflection.stage != Abi::Stage::Vertex) Fail("the vertex stage's MSL is not a vertex function");
    if (fragmentReflection != nullptr && fragmentReflection->stage != Abi::Stage::Fragment) Fail("the pixel stage's MSL is not a fragment function");
    if (vertexReflection.workgroupMemoryStrideDwords != 0u || (fragmentReflection != nullptr && fragmentReflection->workgroupMemoryStrideDwords != 0u)) Fail("device-memory LDS in a draw stage");
    if (graphics.samples > 1u) Fail("multisampled draws (" + std::to_string(graphics.samples) + " samples)");
    if (graphics.conservativeRasterization != VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT) Fail("conservative rasterization requires VK_EXT_conservative_rasterization, which has no Metal counterpart (the Metal API has no conservative rasterization)");
    if (graphics.depthBoundsTest) Fail("the depth bounds test");

    // The draws to record: direct, or the indirect records read on the CPU (through the checked
    // guest memory path, which waits for recorded GPU work that writes them).
    std::vector<DrawCall> calls;
    std::uint32_t highestIndexCount = draw.indexCount;
    if (draw.indirect) {
        using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
        const auto& args = *draw.indirect;
        const auto count = args.countIndirect ? std::min(args.count, Pm4::ReadDrawCount(args)) : args.count;
        for (std::uint32_t record = 0; record < count; ++record) {
            const auto values = Pm4::ReadDrawArguments(args, record);
            if (values.count == 0u || values.instances == 0u) continue;
            const auto firstInstance = args.instanceRule == Rule::InPlace ? values.firstInstance : args.instanceConstant;
            if (draw.indexed) {
                if (values.firstVertexOrIndex >= draw.indexCount) continue;
                const auto vertexOffset = args.vertexRule == Rule::InPlace ? values.vertexOffset : args.vertexConstant;
                calls.push_back({std::min(values.count, draw.indexCount - values.firstVertexOrIndex), values.instances, values.firstVertexOrIndex, static_cast<std::int32_t>(vertexOffset), firstInstance});
            } else {
                const auto firstVertex = args.vertexRule == Rule::InPlace ? values.firstVertexOrIndex : args.vertexConstant;
                calls.push_back({values.count, values.instances, firstVertex, 0, firstInstance});
            }
        }
    } else {
        calls.push_back({draw.indexCount, draw.instanceCount, 0u, draw.indexed ? static_cast<std::int32_t>(draw.firstVertex) : 0, draw.firstInstance});
        if (!draw.indexed) calls.back().first = draw.firstVertex;
    }
    if (calls.empty()) return;

    const auto primitive = graphics.rectList ? Primitive{MTLPrimitiveTypeTriangle, MTLPrimitiveTopologyClassTriangle} : PrimitiveOf(graphics.topology);
    if (graphics.rectList && draw.indexed) Fail("indexed rect lists");
    if (draw.indexed && draw.indexSize != 2u && draw.indexSize != 4u) Fail("index size " + std::to_string(draw.indexSize));

    // Guest ranges of both stages, the vertex fetches, the indices and the color targets.
    StageBindings vertexPlan;
    StageBindings fragmentPlan;
    state.PlanBindings(vertexResult, vertexPlan);
    if (fragmentResult != nullptr) state.PlanBindings(*fragmentResult, fragmentPlan);
    struct Fetch {
        std::uint32_t location;
        Abi::VertexFetchRecord record;
        std::uint64_t begin;
        std::uint64_t end;
    };
    std::vector<Fetch> fetches;
    for (const auto& attribute : vertexResult.vertexAttributes) {
        if (attribute.location >= 32u || (vertexReflection.vertexFetchLocations & (1u << attribute.location)) == 0u) continue;
        Abi::VertexFetchRecord record {};
        record.word3 = attribute.resource.fields[3];
        record.rate = attribute.fetchIndex;
        record.components = attribute.formatComponents == 0u ? attribute.components : attribute.formatComponents;
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
        if (!Graphics::NullVertexDescriptor(attribute)) {
            const auto& fields = attribute.resource.fields;
            begin = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
            record.stride = (fields[1] >> 16u) & 0x3fffu;
            const auto bytes = record.stride == 0u ? Graphics::ZeroStrideAvailableBytes(attribute) : static_cast<std::uint64_t>(fields[2]) * record.stride;
            if (bytes > 0xffffffffu) Fail("a vertex buffer exceeds 4 GiB");
            record.bytes = static_cast<std::uint32_t>(bytes);
            end = begin + bytes;
            if (bytes != 0u) vertexPlan.reads.push_back({begin, end});
        }
        fetches.push_back({attribute.location, record, begin, end});
    }
    std::uint64_t indexBegin = 0;
    std::uint64_t indexBytes = 0;
    if (draw.indexed) {
        indexBegin = draw.indexAddress;
        indexBytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
        if (indexBegin == 0u || indexBytes == 0u) Fail("an indexed draw without an index buffer");
        vertexPlan.reads.push_back({indexBegin, indexBegin + indexBytes});
    }

    // Color attachments by Vulkan location (exportIndex), written as the blend states allow.
    std::vector<ColorAttachment> colors;
    std::uint32_t colorOutputs = fragmentReflection != nullptr ? fragmentReflection->colorOutputs : 0u;
    StageBindings targetPlan;
    for (const auto& target : graphics.colors) {
        if (target.exportIndex >= 8u) Fail("color target location " + std::to_string(target.exportIndex));
        if (target.samples > 1u) Fail("multisampled color targets");
        if (target.dccAddress != 0u) Fail("DCC-compressed color targets");
        // Tiled targets are the image cache's resident images (upstream image path): their writes
        // stay on the GPU until a reader of the memory flushes them (StorageTexture::MarkDirty).
        std::shared_ptr<Graphics::StorageTexture> resident;
        if (target.tileMode != Graphics::ColorTileMode::Linear && state.images->ImageContext() != nullptr && ResidentTargets()) {
            const auto& imageContext = *state.images->ImageContext();
            if (Graphics::AttachmentProxyFormat(imageContext, target.format) != VK_FORMAT_UNDEFINED) Fail("color target format " + std::to_string(target.format) + " needs an attachment proxy");
            if (target.arrayLastSlice != 0u) {
                if (target.cmaskAddress != 0u) Fail("a slice of a 2D array color target with CMASK");
                const auto array = ArraySurfaceForTarget(target);
                if (target.surfaceAddress != target.arrayAddress + target.arraySlice * Graphics::DescribeSurface(array).layerBytes) Fail("the array image's layer pitch differs from the color view's slice offset");
                resident = Graphics::CachedStorageSurface(imageContext, array);
            } else {
                resident = Graphics::CachedStorageSurface(imageContext, SurfaceForTarget(target));
            }
            if (resident == nullptr || !resident->Attachable()) Fail("storage format cannot be a color attachment");
            // CMASK fast clears land in guest memory, which the image then picks up (Refresh).
            state.MaterializeCmask(target, resident.get());
        } else if (target.depth > 1u || target.arrayAddress != 0u || target.mip != 0u || target.mipCount > 1u) {
            Fail("linear color targets with mips, slices or array layers");
        }
        const auto format = Graphics::Metal::PixelFormatOf(target.format);
        if (format.format == MTLPixelFormatInvalid) Fail("color target format " + std::to_string(target.format) + " has no Metal pixel format");
        if (format.blockBytes != target.elementBytes) Fail("color target format " + std::to_string(target.format) + " is " + std::to_string(format.blockBytes) + " bytes a texel in Metal, " + std::to_string(target.elementBytes) + " in guest memory");
        const Graphics::ColorTargetLayout layout(target.extent.width, target.extent.height, target.tileMode, target.elementBytes, target.pipeBankXor);
        ColorAttachment attachment {&target, format.format, target.address, target.address + (resident != nullptr ? target.bytes : layout.Bytes()), nullptr, false, resident};
        const bool written = target.exportIndex < graphics.blends.size() && graphics.blends[target.exportIndex].colorWriteMask != 0u && (colorOutputs & (1u << target.exportIndex)) != 0u;
        // A resident image's guest memory is not touched by the pass (its results are pending in
        // the storage registry instead).
        if (resident == nullptr) {
            targetPlan.reads.push_back({attachment.begin, attachment.end});
            if (written) targetPlan.writes.push_back({attachment.begin, attachment.end});
        }
        colors.push_back(attachment);
    }

    // CMASK fast clears land in guest memory before the targets load from it.
    for (const auto& attachment : colors) {
        if (attachment.resident == nullptr) state.MaterializeCmask(*attachment.target, nullptr);
    }

    const StageBindings* plans[] = {&vertexPlan, &fragmentPlan, &targetPlan};
    state.SettleBindings(plans);
    Graphics::Metal::ImageBindings imageBindings(*state.images, [&state](std::uint64_t address, std::uint64_t bytes) { state.SettleRange(address, bytes); });
    state.PlanImages(vertexResult, imageBindings, vertexPlan);
    if (fragmentResult != nullptr) state.PlanImages(*fragmentResult, imageBindings, fragmentPlan);

    // Tiled targets: their cache entries, and the guest bytes they load from (settled now, read
    // when the pass begins).
    std::uint64_t passKey = Mix(0x6d65746c64726177ull, graphics.renderExtent.width | (static_cast<std::uint64_t>(graphics.renderExtent.height) << 32u));
    for (auto& attachment : colors) {
        const auto& target = *attachment.target;
        passKey = Mix(Mix(Mix(passKey, target.address), target.format | (static_cast<std::uint64_t>(target.exportIndex) << 32u)), static_cast<std::uint64_t>(target.tileMode) | (static_cast<std::uint64_t>(target.extent.width) << 16u) | (static_cast<std::uint64_t>(target.extent.height) << 40u));
        if (attachment.resident != nullptr) {
            passKey = Mix(passKey, reinterpret_cast<std::uint64_t>(attachment.resident.get()) ^ (static_cast<std::uint64_t>(target.mip) << 48u) ^ (static_cast<std::uint64_t>(target.arraySlice | target.depthSlice) << 32u));
            continue;
        }
        if (target.tileMode == Graphics::ColorTileMode::Linear) continue;
        for (const auto& entry : cache.tiled) {
            if (entry->address == target.address && entry->extent.width == target.extent.width && entry->extent.height == target.extent.height && entry->format == target.format && entry->tileMode == target.tileMode && entry->elementBytes == target.elementBytes && entry->pipeBankXor == target.pipeBankXor) {
                attachment.tiled = entry;
                break;
            }
        }
        if (attachment.tiled == nullptr) {
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:attachment.format width:target.extent.width height:target.extent.height mipmapped:NO];
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            id<MTLTexture> texture = [state.device newTextureWithDescriptor:descriptor];
            if (texture == nil) Fail("cannot create a tiled color target texture");
            const std::array<id<MTLTexture>, 1> resident {texture};
            state.images->MakeResident(resident);
            attachment.tiled = std::make_shared<DrawCache::TiledTarget>(DrawCache::TiledTarget{target.address, target.extent, target.format, target.tileMode, target.elementBytes, target.pipeBankXor, texture, 0u});
            cache.tiled.push_back(attachment.tiled);
        }
        attachment.load = attachment.tiled->gpuNewer == 0u;
        if (attachment.load) state.SettleRange(attachment.begin, attachment.end - attachment.begin);
    }

    // Depth/stencil surface (with the clears its pass must perform: a new surface, an HTILE clear).
    std::optional<DrawCache::DepthSurface> depthSurface;
    DrawCache::DepthSurface* depth = nullptr;
    if (graphics.depth) {
        const auto& target = *graphics.depth;
        if (target.samples > 1u) Fail("multisampled depth targets");
        const auto format = DepthFormat(target);
        bool created = false;
        const auto use = Graphics::Metal::AcquireDepthSurface(state.device, target, format, &created);
        if (created) {
            const std::array<id<MTLTexture>, 1> resident {use.texture};
            state.images->MakeResident(resident);
        }
        depthSurface = DrawCache::DepthSurface{target, format, use.texture, use.clearDepth, use.clearStencil};
        depth = &*depthSurface;
        passKey = Mix(Mix(passKey, target.address), static_cast<std::uint64_t>(format));
    }

    // Pipelines: a second channel pass for alternate-alpha 8_8 targets, a back-face pass for
    // per-face depth bias (as upstream's secondChannel and backFaces pipelines).
    Graphics::Metal::RenderPipelineRequest request;
    request.vertex = Graphics::Metal::PipelineFunction::From(vertexResult);
    if (graphics.rectList) request.vertex.macros.push_back({Abi::RectListMacro, "1"});
    if (graphics.negativeOneToOne) request.vertex.macros.push_back({Abi::DepthNegativeOneToOneMacro, "1"});
    if (fragmentResult != nullptr) {
        auto function = Graphics::Metal::PipelineFunction::From(*fragmentResult);
        function.macros.push_back({Abi::VertexOutputsMacro, std::to_string(static_cast<std::uint32_t>(vertexReflection.outputLocations)) + "u"});
        function.macros.push_back({Abi::VertexOutputsHighMacro, std::to_string(static_cast<std::uint32_t>(vertexReflection.outputLocations >> 32u)) + "u"});
        for (const auto& target : graphics.colors) {
            const auto format = Graphics::Metal::PixelFormatOf(target.format).format;
            if (target.exportIndex < 8u && format != MTLPixelFormatInvalid) function.macros.push_back({std::string(Abi::ColorTypeMacroPrefix) + std::to_string(target.exportIndex) + "_TYPE", ColorOutputType(format)});
        }
        // Dual-source blending: the fragment function writes color(0) index(1) too.
        const bool dualSource = graphics.dualSourceBlend || std::any_of(fragmentResult->specialization.begin(), fragmentResult->specialization.end(), [](const auto& constant) { return constant.id == ShaderRecompiler::PipelineSpecialization::DualSourceBlend && constant.value == 1u; });
        if (dualSource) function.macros.push_back({Abi::DualSourceMacro, "1"});
        request.fragment = std::move(function);
    }
    for (const auto& attachment : colors) {
        const auto index = attachment.target->exportIndex;
        auto& color = request.colors[index];
        color.format = attachment.format;
        if (index < graphics.blends.size()) {
            const auto& blend = graphics.blends[index];
            color.blending = blend.blendEnable == VK_TRUE;
            if (color.blending) {
            color.sourceRgb = BlendFactor(blend.srcColorBlendFactor);
            color.destinationRgb = BlendFactor(blend.dstColorBlendFactor);
            color.rgbOperation = BlendOperation(blend.colorBlendOp);
            color.sourceAlpha = BlendFactor(blend.srcAlphaBlendFactor);
            color.destinationAlpha = BlendFactor(blend.dstAlphaBlendFactor);
            color.alphaOperation = BlendOperation(blend.alphaBlendOp);
            }
            // Vulkan leaves attachments a pixel shader has no output for undefined: masked, as
            // upstream's maskedState.
            color.writeMask = (colorOutputs & (1u << index)) != 0u ? WriteMask(blend.colorWriteMask) : MTLColorWriteMaskNone;
        } else {
            color.writeMask = MTLColorWriteMaskNone;
        }
    }
    if (depth != nullptr) {
        if (HasDepth(depth->format)) request.depthFormat = depth->format;
        if (HasStencil(depth->format)) request.stencilFormat = depth->format;
    }
    request.rasterSampleCount = 1u;
    request.alphaToCoverage = graphics.alphaToCoverage;
    request.topology = primitive.topologyClass;
    struct PipelinePass {
        id<MTLRenderPipelineState> pipeline;
        MTLCullMode cull;
        float biasConstant;
        float biasSlope;
    };
    std::vector<PipelinePass> pipelinePasses;
    const auto cullOf = [](VkCullModeFlags mode) {
        return mode == VK_CULL_MODE_FRONT_BIT ? MTLCullModeFront : mode == VK_CULL_MODE_BACK_BIT ? MTLCullModeBack : MTLCullModeNone;
    };
    if (graphics.cullMode == VK_CULL_MODE_FRONT_AND_BACK) return;
    const float biasConstant = graphics.depthBias ? graphics.depthBiasConstant : 0.0f;
    const float biasSlope = graphics.depthBias ? graphics.depthBiasSlope : 0.0f;
    if (Graphics::SplitsAlternateAlpha(graphics)) {
        auto first = request;
        auto second = request;
        for (const auto& attachment : colors) {
            const auto index = attachment.target->exportIndex;
            const bool split = attachment.target->componentMapping == 0xecu && index < graphics.blends.size() && graphics.blends[index].blendEnable;
            if (!split) {
                second.colors[index].writeMask = MTLColorWriteMaskNone;
                continue;
            }
            first.colors[index].writeMask &= ~MTLColorWriteMaskGreen;
            second.colors[index].writeMask &= MTLColorWriteMaskGreen;
            second.colors[index].sourceRgb = second.colors[index].sourceAlpha;
            second.colors[index].destinationRgb = second.colors[index].destinationAlpha;
            second.colors[index].rgbOperation = second.colors[index].alphaOperation;
        }
        pipelinePasses.push_back({state.pipelines->Render(first), cullOf(graphics.cullMode), biasConstant, biasSlope});
        pipelinePasses.push_back({state.pipelines->Render(second), cullOf(graphics.cullMode), biasConstant, biasSlope});
    } else if (graphics.depthBiasPerFace) {
        if (graphics.cullMode != VK_CULL_MODE_NONE) Fail("per-face depth bias with culling");
        id<MTLRenderPipelineState> pipeline = state.pipelines->Render(request);
        pipelinePasses.push_back({pipeline, MTLCullModeBack, biasConstant, biasSlope});
        pipelinePasses.push_back({pipeline, MTLCullModeFront, graphics.backDepthBiasConstant, graphics.backDepthBiasSlope});
    } else {
        pipelinePasses.push_back({state.pipelines->Render(request), cullOf(graphics.cullMode), biasConstant, biasSlope});
    }
    id<MTLDepthStencilState> depthStencil = nil;
    if (depth != nullptr) {
        Graphics::Metal::DepthStencilRequest depthRequest;
        if (graphics.depthTest) {
            depthRequest.depthCompare = Compare(graphics.depthCompare);
            depthRequest.depthWrite = graphics.depthWrite;
        }
        if (graphics.stencilTest && HasStencil(depth->format)) {
            depthRequest.stencil = true;
            depthRequest.front = StencilFace(graphics.stencilFront);
            depthRequest.back = StencilFace(graphics.stencilBack);
        }
        depthStencil = state.pipelines->DepthStencil(depthRequest);
    }

    // Index-list expansion: triangle fans always, and lists and strips whose last vertex provokes
    // (Metal's flat shading takes each primitive's first vertex): every primitive in Vulkan's vertex
    // order, rotated (winding kept) so its provoking vertex comes first. Indexed draws read the guest
    // indices (settled); non-indexed ones count from the call's first vertex (baseVertex adds it).
    const bool lastVertex = graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT && !graphics.rectList;
    const bool expand = primitive.fan || (lastVertex && primitive.topologyClass != MTLPrimitiveTopologyClassPoint);
    std::vector<std::uint32_t> fanIndices;
    std::vector<DrawCall> fanCalls;
    if (expand) {
        std::vector<std::byte> guest;
        if (draw.indexed) {
            state.SettleRange(indexBegin, indexBytes);
            guest.resize(indexBytes);
            std::memcpy(guest.data(), reinterpret_cast<const void*>(indexBegin), indexBytes);
        }
        const auto indexAt = [&](std::uint32_t position) -> std::uint32_t {
            if (draw.indexSize == 2u) {
                std::uint16_t value;
                std::memcpy(&value, guest.data() + position * 2u, 2u);
                return value;
            }
            std::uint32_t value;
            std::memcpy(&value, guest.data() + position * 4u, 4u);
            return value;
        };
        const std::uint32_t restart = draw.indexSize == 2u ? 0xffffu : 0xffffffffu;
        const auto topology = graphics.topology;
        const bool strips = topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN || topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        std::vector<std::uint32_t> segment;
        const auto flush = [&] {
            const auto& v = segment;
            const auto n = v.size();
            const auto triangle = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
                // (a, b, c) in Vulkan order, a provoking first-vertex, c last-vertex.
                if (lastVertex) fanIndices.insert(fanIndices.end(), {c, a, b});
                else fanIndices.insert(fanIndices.end(), {a, b, c});
            };
            const auto line = [&](std::uint32_t a, std::uint32_t b) {
                if (lastVertex) fanIndices.insert(fanIndices.end(), {b, a});
                else fanIndices.insert(fanIndices.end(), {a, b});
            };
            switch (topology) {
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
                for (std::size_t k = 0; k + 2 < n; k += 3) triangle(v[k], v[k + 1], v[k + 2]);
                break;
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
                for (std::size_t i = 0; i + 2 < n; ++i) {
                    // Odd triangles: (i, i + 2, i + 1) with the first vertex provoking, (i + 1, i,
                    // i + 2) with the last (VK_EXT_provoking_vertex): same winding.
                    if (i % 2u == 0u) triangle(v[i], v[i + 1], v[i + 2]);
                    else if (lastVertex) triangle(v[i + 1], v[i], v[i + 2]);
                    else triangle(v[i], v[i + 2], v[i + 1]);
                }
                break;
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
                // Vulkan's fan triangle i is (i + 1, i + 2, 0) with the first vertex provoking, and
                // (0, i + 1, i + 2) with the last (VK_EXT_provoking_vertex): same winding.
                for (std::size_t i = 0; i + 2 < n; ++i) {
                    if (lastVertex) fanIndices.insert(fanIndices.end(), {v[i + 2], v[0], v[i + 1]});
                    else fanIndices.insert(fanIndices.end(), {v[i + 1], v[i + 2], v[0]});
                }
                break;
            case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
                for (std::size_t k = 0; k + 1 < n; k += 2) line(v[k], v[k + 1]);
                break;
            case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
                for (std::size_t i = 0; i + 1 < n; ++i) line(v[i], v[i + 1]);
                break;
            default:
                Fail("index expansion of topology " + std::to_string(topology));
            }
            segment.clear();
        };
        for (const auto& call : calls) {
            const auto first = static_cast<std::uint32_t>(fanIndices.size());
            for (std::uint32_t position = 0; position < call.count; ++position) {
                if (!draw.indexed) {
                    segment.push_back(position);
                    continue;
                }
                const auto value = indexAt(call.first + position);
                if (strips && graphics.primitiveRestart && value == restart) {
                    flush();
                    continue;
                }
                segment.push_back(value);
            }
            flush();
            const auto vertexOffset = draw.indexed ? call.vertexOffset : static_cast<std::int32_t>(call.first);
            fanCalls.push_back({static_cast<std::uint32_t>(fanIndices.size()) - first, call.instances, first, vertexOffset, call.firstInstance});
        }
    }

    state.WrapUnions(plans);

    // The pass: continued when the recorder holds a pass with the same attachments open and no
    // hazard separates this draw from it.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> passReads;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> passWrites;
    for (const auto* plan : plans) {
        passReads.insert(passReads.end(), plan->reads.begin(), plan->reads.end());
        passWrites.insert(passWrites.end(), plan->writes.begin(), plan->writes.end());
    }
    Graphics::PassAccess access;
    access.reads = passReads;
    access.writes = passWrites;
    access.anyReads = !passReads.empty();
    access.anyWrites = !passWrites.empty();
    // A pending depth/stencil clear is a load action: it needs a new pass.
    const bool clears = depth != nullptr && (depth->clearDepth || depth->clearStencil);
    const auto start = recorder.StartDrawPass(passKey, clears, false, access);
    auto& stream = Graphics::Metal::Stream(start.commands);
    const bool continued = start.continued && cache.passKey == passKey && !cache.passTextures.empty();
    auto held = std::make_shared<Held>();
    id<MTL4RenderCommandEncoder> encoder = nil;
    std::vector<std::shared_ptr<DrawCache::TiledTarget>> passTiled;
    if (continued) {
        encoder = stream.Render();
        if (encoder == nil) Fail("the recorder continued a render pass that has no encoder");
    } else {
        // Tiled targets load from guest memory (CPU detile into upload memory, then a copy).
        std::vector<id<MTLTexture>> textures;
        cache.passTextures.clear();
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        std::uint32_t width = graphics.renderExtent.width;
        std::uint32_t height = graphics.renderExtent.height;
        for (auto& attachment : colors) {
            const auto& target = *attachment.target;
            id<MTLTexture> texture = nil;
            if (attachment.resident != nullptr) {
                texture = Graphics::Metal::TextureOfView(attachment.resident->AttachmentView(target.format, target.mip, target.depthSlice, target.arraySlice));
                if (texture == nil) Fail("the resident color target has no attachment view");
            } else if (attachment.tiled != nullptr) {
                texture = attachment.tiled->texture;
                if (attachment.load) {
                    const Graphics::ColorTargetLayout layout(target.extent.width, target.extent.height, target.tileMode, target.elementBytes, target.pipeBankXor);
                    const auto pitch = static_cast<std::size_t>(target.extent.width) * target.elementBytes;
                    const auto linearBytes = pitch * target.extent.height;
                    auto upload = stream.Allocate(linearBytes, 256u);
                    layout.Detile(std::span<const std::byte>(reinterpret_cast<const std::byte*>(target.address), layout.Bytes()), std::span<std::byte>(upload.host, linearBytes));
                    id<MTL4ComputeCommandEncoder> copy = stream.Compute();
                    [copy copyFromBuffer:upload.buffer sourceOffset:upload.offset sourceBytesPerRow:pitch sourceBytesPerImage:linearBytes sourceSize:MTLSizeMake(target.extent.width, target.extent.height, 1) toTexture:texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                    stream.CountCommand();
                }
                passTiled.push_back(attachment.tiled);
            } else {
                const Graphics::ColorTargetLayout layout(target.extent.width, target.extent.height, target.tileMode, target.elementBytes, target.pipeBankXor);
                std::uint64_t offset = 0;
                id<MTLBuffer> buffer = state.BufferAt(attachment.begin, attachment.end - attachment.begin, &offset);
                const auto pitch = layout.BlocksPerRow() * static_cast<std::size_t>(target.elementBytes);
                const auto alignment = [state.device minimumLinearTextureAlignmentForPixelFormat:attachment.format];
                if (offset % alignment != 0u || pitch % alignment != 0u) Fail("a linear color target at offset " + std::to_string(offset) + " with pitch " + std::to_string(pitch) + " breaks the linear texture alignment " + std::to_string(alignment));
                MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:attachment.format width:target.extent.width height:target.extent.height mipmapped:NO];
                descriptor.storageMode = buffer.storageMode;
                descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
                descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
                texture = [buffer newTextureWithDescriptor:descriptor offset:offset bytesPerRow:pitch];
                if (texture == nil) Fail("cannot alias a linear color target");
                state.KeepWrapped(attachment.begin, attachment.end - attachment.begin);
                textures.push_back(texture);
            }
            width = std::min<std::uint32_t>(width, target.extent.width);
            height = std::min<std::uint32_t>(height, target.extent.height);
            auto* color = pass.colorAttachments[attachment.target->exportIndex];
            color.texture = texture;
            color.loadAction = MTLLoadActionLoad;
            color.storeAction = MTLStoreActionStore;
            cache.passTextures.push_back(texture);
        }
        if (depth != nullptr) {
            width = std::min<std::uint32_t>(width, depth->target.extent.width);
            height = std::min<std::uint32_t>(height, depth->target.extent.height);
            if (HasDepth(depth->format)) {
                pass.depthAttachment.texture = depth->texture;
                pass.depthAttachment.loadAction = depth->clearDepth ? MTLLoadActionClear : MTLLoadActionLoad;
                pass.depthAttachment.clearDepth = depth->target.clearDepth;
                pass.depthAttachment.storeAction = MTLStoreActionStore;
            }
            if (HasStencil(depth->format)) {
                pass.stencilAttachment.texture = depth->texture;
                pass.stencilAttachment.loadAction = depth->clearStencil ? MTLLoadActionClear : MTLLoadActionLoad;
                pass.stencilAttachment.clearStencil = depth->target.clearStencil;
                pass.stencilAttachment.storeAction = MTLStoreActionStore;
            }
        }
        if (width == 0u || height == 0u) Fail("a render pass without an extent");
        pass.renderTargetWidth = width;
        pass.renderTargetHeight = height;
        if (colors.empty() && depth == nullptr) pass.defaultRasterSampleCount = 1u;
        if (void* visibility = recorder.SampleVisibilityBuffer()) {
            pass.visibilityResultBuffer = (__bridge id<MTLBuffer>)visibility;
            pass.visibilityResultType = MTLVisibilityResultTypeAccumulate;
        }
        cache.passKey = passKey;
        if (!textures.empty() || !cache.passTextures.empty()) {
            const std::vector<id<MTLTexture>> passTextures = cache.passTextures;
            state.images->MakeResident(passTextures);
            held->objects.insert(held->objects.end(), passTextures.begin(), passTextures.end());
            auto* images = state.images.get();
            recorder.OnComplete([images, passTextures] { images->Release(passTextures); });
        }
        encoder = stream.BeginRender(pass);
        if (encoder == nil) Fail("cannot begin the render pass");
    }
    recorder.NoteSampledDraw(start.commands);

    // Bindings after the stream is taken: descriptor tables, push data, driver data, fetches.
    std::array<std::byte, Abi::PushDataBytes> push {};
    const auto copyPush = [&](const ShaderRecompiler::RecompileResult& result, std::uint32_t offset) {
        if (result.pushConstants.empty()) return std::span<std::byte>();
        if (offset > push.size() || result.pushConstants.size() > push.size() - offset) Fail("push constants exceed the push data block");
        std::memcpy(push.data() + offset, result.pushConstants.data(), result.pushConstants.size());
        return std::span<std::byte>(push).subspan(offset, result.pushConstants.size());
    };
    const auto vertexTable = state.WriteTable(vertexResult, snapshots, stream, vertexPlan, copyPush(vertexResult, vertex.pushConstantOffset));
    const auto fragmentTable = fragmentResult != nullptr ? state.WriteTable(*fragmentResult, snapshots, stream, fragmentPlan, copyPush(*fragmentResult, fragment->pushConstantOffset)) : 0u;
    const auto pushAddress = stream.Write(std::span<const std::byte>(push), 16u).address;
    const auto driverAddress = state.WriteDriverData(stream, {0u, 0u, 0u});
    std::uint64_t fetchAddress = driverAddress;
    if (!fetches.empty()) {
        std::uint32_t highest = 0;
        for (const auto& fetch : fetches) highest = std::max(highest, fetch.location);
        std::vector<Abi::VertexFetchRecord> records(highest + 1u);
        for (auto& fetch : fetches) {
            auto record = fetch.record;
            if (fetch.end > fetch.begin) {
                record.address = state.WrapOrCopy(fetch.begin, fetch.end - fetch.begin, stream, false);
                state.KeepWrapped(fetch.begin, fetch.end - fetch.begin);
            }
            records[fetch.location] = record;
        }
        fetchAddress = stream.Write(std::as_bytes(std::span(records)), 16u).address;
    }
    [cache.vertexArguments setAddress:vertexTable != 0u ? vertexTable : driverAddress atIndex:Abi::Index(Abi::BufferIndex::DescriptorTable)];
    [cache.vertexArguments setAddress:pushAddress atIndex:Abi::Index(Abi::BufferIndex::PushData)];
    [cache.vertexArguments setAddress:driverAddress atIndex:Abi::Index(Abi::BufferIndex::WorkgroupMemory)];
    [cache.vertexArguments setAddress:driverAddress atIndex:Abi::Index(Abi::BufferIndex::DriverData)];
    [cache.vertexArguments setAddress:fetchAddress atIndex:Abi::Index(Abi::BufferIndex::VertexFetch)];
    [cache.fragmentArguments setAddress:fragmentTable != 0u ? fragmentTable : driverAddress atIndex:Abi::Index(Abi::BufferIndex::DescriptorTable)];
    [cache.fragmentArguments setAddress:pushAddress atIndex:Abi::Index(Abi::BufferIndex::PushData)];
    [cache.fragmentArguments setAddress:driverAddress atIndex:Abi::Index(Abi::BufferIndex::WorkgroupMemory)];
    [cache.fragmentArguments setAddress:driverAddress atIndex:Abi::Index(Abi::BufferIndex::DriverData)];
    [cache.fragmentArguments setAddress:driverAddress atIndex:Abi::Index(Abi::BufferIndex::VertexFetch)];
    [encoder setArgumentTable:cache.vertexArguments atStages:MTLRenderStageVertex];
    [encoder setArgumentTable:cache.fragmentArguments atStages:MTLRenderStageFragment];

    // Dynamic state. Vulkan's viewport transform is y_fb = y + (1 + y_ndc) h / 2, Metal's
    // originY + (1 - y_ndc) height / 2: the same transform with originY = y + h, height = -h.
    const auto& viewport = graphics.viewport;
    [encoder setViewport:MTLViewport{viewport.x, static_cast<double>(viewport.y) + viewport.height, viewport.width, -static_cast<double>(viewport.height), viewport.minDepth, viewport.maxDepth}];
    const auto passWidth = static_cast<std::int64_t>(colors.empty() ? (depth != nullptr ? depth->target.extent.width : graphics.renderExtent.width) : graphics.renderExtent.width);
    const auto passHeight = static_cast<std::int64_t>(colors.empty() ? (depth != nullptr ? depth->target.extent.height : graphics.renderExtent.height) : graphics.renderExtent.height);
    std::int64_t limitWidth = passWidth;
    std::int64_t limitHeight = passHeight;
    for (const auto& attachment : colors) {
        limitWidth = std::min<std::int64_t>(limitWidth, attachment.target->extent.width);
        limitHeight = std::min<std::int64_t>(limitHeight, attachment.target->extent.height);
    }
    if (depth != nullptr) {
        limitWidth = std::min<std::int64_t>(limitWidth, depth->target.extent.width);
        limitHeight = std::min<std::int64_t>(limitHeight, depth->target.extent.height);
    }
    const auto x0 = std::clamp<std::int64_t>(graphics.scissor.offset.x, 0, limitWidth);
    const auto y0 = std::clamp<std::int64_t>(graphics.scissor.offset.y, 0, limitHeight);
    const auto x1 = std::clamp<std::int64_t>(static_cast<std::int64_t>(graphics.scissor.offset.x) + graphics.scissor.extent.width, x0, limitWidth);
    const auto y1 = std::clamp<std::int64_t>(static_cast<std::int64_t>(graphics.scissor.offset.y) + graphics.scissor.extent.height, y0, limitHeight);
    const bool empty = x1 == x0 || y1 == y0;
    [encoder setScissorRect:MTLScissorRect{static_cast<NSUInteger>(x0), static_cast<NSUInteger>(y0), static_cast<NSUInteger>(std::max<std::int64_t>(x1 - x0, 1)), static_cast<NSUInteger>(std::max<std::int64_t>(y1 - y0, 1))}];
    [encoder setFrontFacingWinding:graphics.frontFace == VK_FRONT_FACE_COUNTER_CLOCKWISE ? MTLWindingCounterClockwise : MTLWindingClockwise];
    [encoder setDepthClipMode:graphics.depthClamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
    [encoder setBlendColorRed:graphics.blendConstants[0] green:graphics.blendConstants[1] blue:graphics.blendConstants[2] alpha:graphics.blendConstants[3]];
    [encoder setTriangleFillMode:MTLTriangleFillModeFill];
    if (depthStencil != nil) {
        [encoder setDepthStencilState:depthStencil];
        [encoder setStencilFrontReferenceValue:graphics.stencilFront.reference & 0xffu backReferenceValue:graphics.stencilBack.reference & 0xffu];
        held->objects.push_back(depthStencil);
    } else {
        [encoder setDepthStencilState:nil];
    }

    // Index buffers: the guest indices in place (4-byte aligned), else a copy; fans their list.
    std::uint64_t indexAddress = 0;
    std::uint64_t indexLength = 0;
    MTLIndexType indexType = draw.indexSize == 2u ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
    const bool expanded = expand;
    const auto drawType = expand ? (primitive.topologyClass == MTLPrimitiveTopologyClassLine ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle) : primitive.type;
    const auto& drawCalls = expanded ? fanCalls : calls;
    const bool indexedDraw = draw.indexed || expanded;
    if (expanded) {
        indexType = MTLIndexTypeUInt32;
        if (!fanIndices.empty()) {
            const auto upload = stream.Write(std::as_bytes(std::span(fanIndices)), 16u);
            indexAddress = upload.address;
            indexLength = fanIndices.size() * 4u;
        }
    } else if (draw.indexed) {
        if (indexBegin % 4u == 0u) {
            indexAddress = state.WrapOrCopy(indexBegin, indexBytes, stream, false);
            state.KeepWrapped(indexBegin, indexBytes);
        } else {
            state.SettleRange(indexBegin, indexBytes);
            const auto upload = stream.Write(std::span<const std::byte>(reinterpret_cast<const std::byte*>(indexBegin), indexBytes), 16u);
            indexAddress = upload.address;
        }
        indexLength = indexBytes;
    }

    std::uint64_t recorded = 0;
    if (!empty) {
        for (const auto& pass : pipelinePasses) {
            [encoder setRenderPipelineState:pass.pipeline];
            [encoder setCullMode:pass.cull];
            [encoder setDepthBias:pass.biasConstant slopeScale:pass.biasSlope clamp:graphics.depthBias ? graphics.depthBiasClamp : 0.0f];
            for (const auto& call : drawCalls) {
                if (call.count == 0u || call.instances == 0u) continue;
                if (graphics.rectList) {
                    // Rect lists (B6/D6 interface): 6 host vertices per rect from vertex 0, the
                    // guest first vertex in the vertex stage's DriverData.workgroupBase[0].
                    const auto rects = call.count / 3u;
                    if (rects == 0u) continue;
                    [cache.vertexArguments setAddress:state.WriteDriverData(stream, {call.first, 0u, 0u}) atIndex:Abi::Index(Abi::BufferIndex::DriverData)];
                    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:static_cast<NSUInteger>(rects) * 6u instanceCount:call.instances baseInstance:call.firstInstance];
                } else if (indexedDraw) {
                    const auto size = indexType == MTLIndexTypeUInt16 ? 2u : 4u;
                    [encoder drawIndexedPrimitives:drawType indexCount:call.count indexType:indexType indexBuffer:indexAddress + static_cast<std::uint64_t>(call.first) * size indexBufferLength:indexLength - static_cast<std::uint64_t>(call.first) * size instanceCount:call.instances baseVertex:call.vertexOffset baseInstance:call.firstInstance];
                } else {
                    [encoder drawPrimitives:drawType vertexStart:call.first vertexCount:call.count instanceCount:call.instances baseInstance:call.firstInstance];
                }
                stream.CountCommand();
                ++recorded;
            }
            held->objects.push_back(pass.pipeline);
        }
    }

    // Tiled targets written by this pass go back to guest memory when the batch completes: the
    // pass's end copies them into upload memory, the completion retiles them.
    std::function<void(VkCommandBuffer)> afterPass;
    if (!continued && !passTiled.empty()) {
        std::vector<std::pair<std::shared_ptr<DrawCache::TiledTarget>, bool>> targets;
        for (const auto& tiled : passTiled) {
            bool written = false;
            for (const auto& attachment : colors) {
                if (attachment.tiled == tiled) written = std::any_of(targetPlan.writes.begin(), targetPlan.writes.end(), [&](const auto& range) { return range.first == attachment.begin; });
            }
            targets.push_back({tiled, written});
        }
        for (auto& [tiled, written] : targets) {
            if (written) ++tiled->gpuNewer;
        }
        afterPass = [targets](VkCommandBuffer commands) {
            auto& passStream = Graphics::Metal::Stream(commands);
            auto* active = Recorder::Active();
            for (const auto& [tiled, written] : targets) {
                if (!written) continue;
                const auto pitch = static_cast<std::size_t>(tiled->extent.width) * tiled->elementBytes;
                const auto linearBytes = pitch * tiled->extent.height;
                auto upload = passStream.Allocate(linearBytes, 256u);
                id<MTL4ComputeCommandEncoder> copy = passStream.Compute();
                [copy copyFromTexture:tiled->texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(tiled->extent.width, tiled->extent.height, 1) toBuffer:upload.buffer destinationOffset:upload.offset destinationBytesPerRow:pitch destinationBytesPerImage:linearBytes];
                passStream.CountCommand();
                if (active == nullptr) continue;
                auto target = tiled;
                std::byte* host = upload.host;
                active->OnComplete([target, host, linearBytes] {
                    const Graphics::ColorTargetLayout layout(target->extent.width, target->extent.height, target->tileMode, target->elementBytes, target->pipeBankXor);
                    layout.Tile(std::span<const std::byte>(host, linearBytes), std::span<std::byte>(reinterpret_cast<std::byte*>(target->address), layout.Bytes()));
                    if (target->gpuNewer != 0u) --target->gpuNewer;
                });
            }
            passStream.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        };
    }
    recorder.LeaveRenderPassOpen(passKey, Recorder::NoTiming, false, access, std::move(afterPass));

    // Resident targets: the results stay on the GPU until something reads the target's memory.
    for (const auto& attachment : colors) {
        if (attachment.resident == nullptr) continue;
        recorder.Keep(attachment.resident);
        attachment.resident->MarkDirty();
    }

    // Lifetimes and pending accesses.
    held->objects.push_back(cache.vertexArguments);
    held->objects.push_back(cache.fragmentArguments);
    recorder.Keep(held);
    if (!imageBindings.Empty()) imageBindings.Commit(recorder);
    state.NoteBindings(vertexPlan);
    state.NoteBindings(fragmentPlan);
    state.NoteBindings(targetPlan);
    state.CheckFaultsOnComplete();
    if (Trace()) std::fprintf(stderr, "[metal] draw %s %zu calls (%llu recorded, %zu pipeline passes), %zu color targets, depth %s, pass %s\n", indexedDraw ? "indexed" : "direct", drawCalls.size(), static_cast<unsigned long long>(recorded), pipelinePasses.size(), colors.size(), depth != nullptr ? "yes" : "no", continued ? "continued" : "begun");
}

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
    // Draws whose last vertex provokes are expanded into index lists rotated so that vertex comes
    // first (MetalDraw.mm RecordDraw).
    return true;
}

bool VulkanDevice::GraphicsPipelineLibraries() const {
    return false;
}

void VulkanDevice::NoteDrawRecipeMiss(DrawRecipePrecheck) {
}

void VulkanDevice::Draw(const Graphics::State& graphics, const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> shaders, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>* recipe) {
    if (recipe != nullptr) *recipe = nullptr;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) throw std::runtime_error("Metal device: a draw stage has no program");
        ShaderRecompiler::RuntimeAbi::RequireVersion(shader.program->runtimeAbiVersion);
    }
    if (state->recorder == nullptr) throw std::runtime_error("Metal device: a compile-only device cannot draw");
    const Graphics::CompiledShader* vertex = nullptr;
    const Graphics::CompiledShader* fragment = nullptr;
    for (const auto& shader : shaders) {
        if (shader.stage == ShaderRecompiler::ShaderStage::Vertex && vertex == nullptr) vertex = &shader;
        else if (shader.stage == ShaderRecompiler::ShaderStage::Fragment && fragment == nullptr) fragment = &shader;
        else Fail("draws with a " + std::to_string(static_cast<int>(shader.stage)) + " stage (geometry, tessellation and mesh paths are not implemented)");
    }
    if (vertex == nullptr) Fail("a draw without a vertex stage");
    if (graphics.stages.path != Graphics::ShaderPath::Vertex || graphics.stages.mesh || graphics.stages.tessellation) Fail("the geometry, tessellation and mesh shader paths");
    for (const auto* shader : {vertex, fragment}) {
        if (shader == nullptr) continue;
        if (shader->program->msl.empty()) throw std::runtime_error("Metal device: the shader has no MSL form (recompile it for the device's Target(), whose backend is Msl)");
        if (shader->program->msl.Reflection().abiVersion != ShaderRecompiler::MslAbi::Version) throw std::runtime_error("Metal device: the shader was emitted for MslAbi version " + std::to_string(shader->program->msl.Reflection().abiVersion) + ", the device binds version " + std::to_string(ShaderRecompiler::MslAbi::Version));
    }
    if (draw.flags != 0u && draw.flags != 0x20u) Fail("draw flags " + std::to_string(draw.flags));
    if (!draw.indirect && (draw.indexCount == 0u || draw.instanceCount == 0u)) return;
    @autoreleasepool {
        state->RecordDraw(graphics, draw, *vertex, fragment, snapshots);
    }
}

std::optional<std::string> VulkanDevice::KnownDrawRejection(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> shaders) const {
    if (graphics.stages.path != Graphics::ShaderPath::Vertex || graphics.stages.mesh || graphics.stages.tessellation) return std::string("Metal draw (work package D6): the geometry, tessellation and mesh shader paths are not implemented");
    for (const auto& shader : shaders) {
        if (shader.stage != ShaderRecompiler::ShaderStage::Vertex && shader.stage != ShaderRecompiler::ShaderStage::Fragment) return std::string("Metal draw (work package D6): only vertex and pixel stages are implemented");
    }
    if (graphics.samples > 1u) return std::string("Metal draw (work package D6): multisampled draws are not implemented");
    return std::nullopt;
}

// CB metadata passes: a fast-clear eliminate (or a DCC decompress of a target without DCC) expands
// the CMASK into guest memory as a draw would; DCC and MSAA resolves are not implemented.
void VulkanDevice::ColorMetadataPass(const Graphics::ColorMetadataPass& pass) {
    if (state->recorder == nullptr) throw std::runtime_error("Metal device: a compile-only device has no color metadata passes");
    if (pass.mode == Graphics::ColorMetadataPass::Mode::Resolve) Fail("CB resolves (multisampled targets)");
    for (const auto& color : pass.targets) {
        if (color.dccAddress != 0u) Fail("CB metadata passes over DCC-compressed targets");
        state->MaterializeCmask(color, nullptr);
    }
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
