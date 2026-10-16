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
#include "MetalMesh.h"
#include "MetalTessellation.h"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "MslAbi.hpp"
#include "PipelineSpecialization.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "MetalResidency.h"
#include <algorithm>
#include <bit>
#include <array>
#include <atomic>
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
    // Tessellated draws (MetalTessellation.h): the object stage's table.
    id<MTL4ArgumentTable> objectArguments = nil;
    // MslAbi v4: the buffer of primitive-ordered fragment functions' raster order group (contents
    // ignored), made on first use, resident through the device's residency set.
    id<MTLBuffer> rasterOrder = nil;

    // The depth/stencil surface of one draw (DepthSurface.mm's registry holds the surfaces: GPU-only,
    // cleared at creation and after HTILE fast clears, never written back to guest memory).
    struct DepthSurface {
        Graphics::DepthTarget target;
        MTLPixelFormat format;
        id<MTLTexture> texture;
        bool clearDepth = false;
        bool clearStencil = false;
        // The clear values (AcquireDepthSurface: a plane fill's, else DB_DEPTH_CLEAR/DB_STENCIL_CLEAR).
        float depthValue = 0.0f;
        std::uint8_t stencilValue = 0;
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
        // A mip in its surface's packed mip tail (upstream #4328): `address` is the tail block, the
        // mip's texels sit at (tailX, tailY) of it, and the rest of the block belongs to other mips.
        bool tail = false;
        std::uint32_t tailX = 0;
        std::uint32_t tailY = 0;
    };
    std::vector<std::shared_ptr<TiledTarget>> tiled;

    // Bit 0: a CMASK fast clear of a DCC target was met (as upstream's detiler reports its GPU
    // pass's), reported by the next materialization.
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
        table.maxBufferBindCount = Abi::Index(Abi::BufferIndex::Count);
        NSError* error = nil;
        cache->vertexArguments = [state.device newArgumentTableWithDescriptor:table error:&error];
        cache->fragmentArguments = [state.device newArgumentTableWithDescriptor:table error:&error];
        cache->objectArguments = [state.device newArgumentTableWithDescriptor:table error:&error];
        if (cache->vertexArguments == nil || cache->fragmentArguments == nil || cache->objectArguments == nil) Fail("cannot create the draw argument tables: " + Metal::Describe(error));
        state.draws = std::move(cache);
    }
    return *state.draws;
}

namespace {

std::atomic<std::uint64_t> emulatedDraws{0};
// D81: index expansions gathered on the GPU, and those that read indices recorded work still
// wrote on the CPU (each a recorder sync).
std::atomic<std::uint64_t> indexGathers{0};
std::atomic<std::uint64_t> indexSettles{0};

bool GpuIndexGather() {
    static const bool enabled = std::getenv("APS5_CPU_INDEX_EXPANSION") == nullptr;
    return enabled;
}
std::atomic<std::uint64_t> emulatedSlicePasses{0};
std::atomic<std::uint64_t> emulatedBegun{0};

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
    // Multisampled targets: the MultisampleTarget host image (samples live only on the host, the
    // guest memory is guarded, as upstream's MultisampleTarget.cpp).
    id<MTLTexture> multisample = nil;
};

// Tiled targets outside the image cache move between guest memory and their private texture as
// tightly packed rows. A packed mip tail moves only its own texels, addressed in the tail block's
// layout, so the other mips sharing the block keep their bytes (upstream renders such a mip
// through the GPU detiler with the mip's ComputeColorTargetMipLayout).
Graphics::ColorTargetLayout TiledGuestLayout(const DrawCache::TiledTarget& target) {
    return Graphics::ColorTargetLayout(target.tail ? target.tailX + target.extent.width : target.extent.width, target.tail ? target.tailY + target.extent.height : target.extent.height, target.tileMode, target.elementBytes, target.pipeBankXor);
}

void DetileTarget(const DrawCache::TiledTarget& target, std::span<std::byte> linear) {
    const auto layout = TiledGuestLayout(target);
    // Whole-surface native access: a displaced target (D31, D33) at its alias, not through the fault path.
    const auto* guest = reinterpret_cast<const std::byte*>(GuestArena::HostRange(target.address, layout.Bytes()));
    if (!target.tail) {
        layout.Detile(std::span<const std::byte>(guest, layout.Bytes()), linear);
        return;
    }
    const std::size_t element = target.elementBytes;
    for (std::uint32_t y = 0; y < target.extent.height; ++y) {
        for (std::uint32_t x = 0; x < target.extent.width; ++x) std::memcpy(linear.data() + (static_cast<std::size_t>(y) * target.extent.width + x) * element, guest + layout.Offset(target.tailX + x, target.tailY + y), element);
    }
}

void TileTarget(const DrawCache::TiledTarget& target, std::span<const std::byte> linear) {
    const auto layout = TiledGuestLayout(target);
    auto* guest = reinterpret_cast<std::byte*>(GuestArena::HostRange(target.address, layout.Bytes()));
    if (!target.tail) {
        layout.Tile(linear, std::span<std::byte>(guest, layout.Bytes()));
        return;
    }
    const std::size_t element = target.elementBytes;
    for (std::uint32_t y = 0; y < target.extent.height; ++y) {
        for (std::uint32_t x = 0; x < target.extent.width; ++x) std::memcpy(guest + layout.Offset(target.tailX + x, target.tailY + y), linear.data() + (static_cast<std::size_t>(y) * target.extent.width + x) * element, element);
    }
}

// A slice of a 2D array color target as the whole array (upstream ArraySurfaceForTarget).
Graphics::GuestTextureResource ArraySurfaceForTarget(const Graphics::ColorTarget& color) {
    auto surface = Graphics::SurfaceForTarget(color);
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

// Upstream Draw.cpp prepareDrawInputs (fdefd7660): the index buffer and the vertex records a draw
// reads must not alias its color target. A vertex buffer counts from the first record the draw
// fetches (firstVertex, or firstInstance for instance-rate inputs), so a descriptor based below the
// target is accepted when its fetched records lie past it; indirect draws (records unknown on the
// CPU) count the whole descriptor, as upstream. Indices are scanned for the highest one only when
// the descriptor reaches the target.
template <typename Settle>
static void CheckInputsClearOfTarget(Settle settle, const Graphics::State& graphics, const Pm4::DrawParameters& draw, std::span<const ShaderRecompiler::VertexAttribute> attributes, std::uint64_t indexBegin, std::uint64_t indexBytes) {
    if (!graphics.hasColorTarget) return;
    const auto targetBegin = graphics.color.address;
    const auto targetEnd = targetBegin + graphics.color.bytes;
    const auto overlaps = [&](std::uint64_t begin, std::uint64_t end) { return begin < targetEnd && targetBegin < end; };
    if (draw.indexed && overlaps(indexBegin, indexBegin + indexBytes)) throw std::runtime_error("AGC graphics: index buffer aliases the render target");
    std::optional<std::uint32_t> maxIndex;
    bool scanned = false;
    for (const auto& attribute : attributes) {
        if (Graphics::NullVertexDescriptor(attribute)) continue;
        const auto& fields = attribute.resource.fields;
        const auto address = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
        const auto stride = (fields[1] >> 16u) & 0x3fffu;
        const std::uint64_t extent = stride == 0u ? Graphics::ZeroStrideAvailableBytes(attribute) : static_cast<std::uint64_t>(fields[2]) * stride;
        if (extent == 0u || !overlaps(address, address + extent)) continue;
        std::uint64_t firstByte = 0;
        std::uint64_t bytes = extent;
        if (!draw.indirect) {
            if (attribute.fetchIndex == 0u && !scanned) {
                scanned = true;
                if (!draw.indexed) {
                    maxIndex = draw.firstVertex + draw.indexCount - 1u;
                } else {
                    settle(indexBegin, indexBytes);
                    // Lists restart too (PrimitiveListRestart: the index expansion below).
                    const auto highest = Graphics::HighestDrawIndex(std::span(reinterpret_cast<const std::byte*>(GuestArena::HostRange(indexBegin, static_cast<std::size_t>(indexBytes))), static_cast<std::size_t>(indexBytes)), draw.indexSize, graphics.primitiveRestart);
                    if (highest) maxIndex = *highest + draw.firstVertex;
                }
            }
            if (attribute.fetchIndex == 0u && !maxIndex) continue;
            bytes = Graphics::VertexBufferReadSize(attribute, maxIndex.value_or(0u), draw.instanceCount, draw.firstInstance);
            firstByte = Graphics::VertexBufferReadOffset(attribute, draw.firstVertex, draw.firstInstance);
        }
        if (overlaps(address + firstByte, address + bytes)) throw std::runtime_error("AGC graphics: vertex buffer aliases the render target");
    }
}

// Upstream Draw.cpp prepareDrawInputs: the index buffer and the vertex records a draw reads must not
// overlap a multisampled color target, whose samples are only on the host (RequireNoMultisampleTarget).
// A vertex descriptor is checked whole first (nothing is guarded, or it is clear of every target);
// only one that reaches a target is narrowed to the records the draw fetches, as upstream's copy plan.
template <typename Settle>
static void CheckInputsClearOfMultisampleTargets(Settle settle, const Graphics::State& graphics, const Pm4::DrawParameters& draw, std::span<const ShaderRecompiler::VertexAttribute> attributes, std::uint64_t indexBegin, std::uint64_t indexBytes) {
    if (draw.indexed) Graphics::RequireNoMultisampleTarget(indexBegin, static_cast<std::size_t>(indexBytes), "an index buffer read");
    std::optional<std::uint32_t> maxIndex;
    bool scanned = false;
    for (const auto& attribute : attributes) {
        if (Graphics::NullVertexDescriptor(attribute)) continue;
        const auto& fields = attribute.resource.fields;
        const auto address = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
        const auto stride = (fields[1] >> 16u) & 0x3fffu;
        const std::uint64_t extent = stride == 0u ? Graphics::ZeroStrideAvailableBytes(attribute) : static_cast<std::uint64_t>(fields[2]) * stride;
        if (extent == 0u) continue;
        try {
            Graphics::RequireNoMultisampleTarget(address, static_cast<std::size_t>(extent), "a vertex buffer read");
            continue;
        } catch (const std::runtime_error&) {
            if (draw.indirect) throw;
        }
        if (attribute.fetchIndex == 0u && !scanned) {
            scanned = true;
            if (!draw.indexed) {
                maxIndex = draw.firstVertex + draw.indexCount - 1u;
            } else {
                settle(indexBegin, indexBytes);
                const auto highest = Graphics::HighestDrawIndex(std::span(reinterpret_cast<const std::byte*>(GuestArena::HostRange(indexBegin, static_cast<std::size_t>(indexBytes))), static_cast<std::size_t>(indexBytes)), draw.indexSize, graphics.primitiveRestart);
                if (highest) maxIndex = *highest + draw.firstVertex;
            }
        }
        if (attribute.fetchIndex == 0u && !maxIndex) continue;
        const auto end = Graphics::VertexBufferReadSize(attribute, maxIndex.value_or(0u), draw.instanceCount, draw.firstInstance);
        const auto first = Graphics::VertexBufferReadOffset(attribute, draw.firstVertex, draw.firstInstance);
        if (end > first) Graphics::RequireNoMultisampleTarget(address + first, static_cast<std::size_t>(end - first), "a vertex buffer read");
    }
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
// expanded (all 0xff), before a draw or a CB fast-clear eliminate reads the target. Tiles with any
// other code keep their texels (upstream 51c043ba5).
void VulkanDevice::State::MaterializeCmask(const Graphics::ColorTarget& color, Graphics::StorageTexture* resident) {
    auto& cache = Draws();
    if (const auto codes = std::exchange(cache.cmaskErrors, 0u); codes != 0u) {
        if ((codes & 1u) != 0u) Fail("a CMASK fast clear of a DCC color target, whose clear belongs in its DCC keys, was recorded");
    }
    if (color.cmaskAddress == 0u) return;
    const Graphics::CmaskLayout layout(color.extent.width, color.extent.height);
    const auto bytes = layout.Bytes();
    if (color.cmaskBytes != bytes) Fail("CMASK size differs from the color target's CMASK layout");
    SettleRange(color.cmaskAddress, bytes);
    const auto* mask = reinterpret_cast<const std::uint8_t*>(GuestArena::HostRange(color.cmaskAddress, bytes)); // alias if displaced (D31, D33)
    if (std::all_of(mask, mask + bytes, [](std::uint8_t value) { return value == 0xffu; })) return;
    std::vector<std::uint32_t> cleared;
    for (std::uint32_t tileY = 0; tileY < layout.TilesY(); ++tileY) {
        for (std::uint32_t tileX = 0; tileX < layout.TilesX(); ++tileX) {
            const auto nibble = layout.Nibble(tileX, tileY);
            const auto code = (static_cast<std::uint32_t>(mask[nibble / 2u]) >> ((nibble % 2u) * 4u)) & 0xfu;
            // Upstream 51c043ba5: a code other than 0 (cleared) keeps its texels like an expanded tile
            // (titles fast clear targets whose CMASK they never initialized).
            if (code == 0u) cleared.push_back(tileY * layout.TilesX() + tileX);
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

// DCC color targets (upstream Draw.cpp storeClearTexels / storeSingleTexels, CPU path): keys that
// encode a fast clear (the clear register or a clear code) get the clear texel in guest memory,
// comp-to-single keys get each 256-byte block's first texel, and the keys are left uncompressed
// (0xff) before the draw loads the target, as upstream leaves them after its draw. Per-block
// compressed keys are not modeled (upstream rejects them too).
void VulkanDevice::State::MaterializeDcc(const Graphics::ColorTarget& color, Graphics::StorageTexture* resident) {
    if (color.dccAddress == 0u) return;
    const auto keyBytes = Graphics::DccKeyBytes(color.bytes);
    const auto keyCount = color.mipCount != 1u || color.depth != 1u || !color.dccPipeAligned ? keyBytes : Graphics::DccKeyCount(Graphics::ColorTextureTileMode(color.tileMode), color.elementBytes, color.extent.width, color.extent.height, color.bytes);
    SettleRange(color.dccAddress, keyBytes);
    const auto keys = Graphics::CurrentDccKeys(color.dccAddress, color.bytes);
    if (keys == Graphics::DccKeys::Uncompressed) return;
    const bool single = keys == Graphics::DccKeys::ClearSingle;
    if (!single && !Graphics::IsDccClear(keys)) Fail(std::string("DCC keys that are ") + Graphics::DccKeysName(keys) + " (per-block metadata is not modeled)");
    const auto elementBytes = static_cast<std::size_t>(color.elementBytes);
    if (elementBytes == 0u || elementBytes > 16u) Fail("unexpected color element size");
    Graphics::StorageTexture::FlushPending(color.address, color.bytes, nullptr, "DCC materialization");
    if (recorder->PendingWriteOverlaps(color.address, color.bytes)) {
        Recorder::CountSync(2);
        recorder->SyncThrough(color.address, color.bytes);
    }
    if (single) {
        if (color.tileMode == Graphics::ColorTileMode::Linear) Fail("comp-to-single DCC keys of a linear color target");
        const Graphics::ColorTargetLayout surface(color.extent.width, color.extent.height, color.tileMode, color.elementBytes, color.pipeBankXor);
        if (surface.Bytes() != color.bytes) Fail("comp-to-single DCC keys over a color target whose layout differs from its surface");
        std::vector<std::byte> texels(color.bytes);
        GuestMemory::Read(color.address, texels, surface.Alignment());
        constexpr std::size_t BlockBytes = 256;
        for (std::uint32_t y = 0; y < color.extent.height; ++y) {
            for (std::uint32_t x = 0; x < color.extent.width; ++x) {
                const auto offset = surface.Offset(x, y);
                const auto first = offset / BlockBytes * BlockBytes;
                if (offset != first) std::memcpy(texels.data() + offset, texels.data() + first, elementBytes);
            }
        }
        GuestMemory::Write(color.address, texels, surface.Alignment());
    } else {
        std::array<std::byte, 16> texel {};
        if (keys == Graphics::DccKeys::ClearRegister) {
            if (elementBytes > sizeof(color.clearWords)) Fail("the DCC register clear of a texel over 64 bits is not modeled");
            std::memcpy(texel.data(), color.clearWords.data(), elementBytes);
        } else if (!Graphics::FillDccClear(color.format, keys, color.dccAlphaOnMsb, std::span(texel.data(), elementBytes))) {
            Fail("the target's format has no encoding of its DCC clear code");
        }
        std::vector<std::byte> texels(color.bytes);
        for (std::size_t offset = 0; offset + elementBytes <= texels.size(); offset += elementBytes) std::memcpy(texels.data() + offset, texel.data(), elementBytes);
        GuestMemory::Write(color.address, texels);
    }
    Graphics::MarkDccUncompressed(context, color.dccAddress, color.bytes, keyCount);
    if (resident != nullptr) resident->Refresh();
}

id<MTLBuffer> VulkanDevice::State::BufferAt(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset) {
    auto& state = *this;
    state.Wrap(address, bytes);
    if (id<MTLBuffer> import = state.ImportOf(address, bytes, offset)) return import;
    if (id<MTLBuffer> wrapper = state.WrapperOf(address, bytes, offset)) return wrapper;
    Fail("guest memory at " + std::to_string(address) + " has no GPU mapping");
}

void VulkanDevice::State::RecordDraw(const Graphics::State& graphics, const Pm4::DrawParameters& draw, const Graphics::CompiledShader& vertex, const Graphics::CompiledShader* fragment, std::span<const Graphics::GuestMemorySnapshot> snapshots, const Graphics::Metal::TessellationStages* tessellation) {
    auto& state = *this;
    auto& cache = Draws();
    auto& recorder = *state.recorder;
    // A batch over its kept-byte budget is submitted first (upstream's Draw does the same).
    recorder.BoundKeptBytes();
    const auto& vertexResult = *vertex.program;
    // Before the indirect records are read, as upstream validates its inputs first.
    CheckInputsClearOfTarget([&](std::uint64_t begin, std::uint64_t bytes) { state.SettleRange(begin, bytes); }, graphics, draw, (tessellation != nullptr ? *tessellation->local->program : vertexResult).vertexAttributes, draw.indexed ? draw.indexAddress : 0u, draw.indexed ? static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize : 0u);
    CheckInputsClearOfMultisampleTargets([&](std::uint64_t begin, std::uint64_t bytes) { state.SettleRange(begin, bytes); }, graphics, draw, (tessellation != nullptr ? *tessellation->local->program : vertexResult).vertexAttributes, draw.indexed ? draw.indexAddress : 0u, draw.indexed ? static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize : 0u);
    const auto& vertexReflection = vertexResult.msl.Reflection();
    const auto* fragmentResult = fragment != nullptr ? fragment->program : nullptr;
    const Abi::Reflection* fragmentReflection = fragmentResult != nullptr ? &fragmentResult->msl.Reflection() : nullptr;
    // Upstream 1ffc5f672 / 72518f64f (PA_SC_AA_MASK). A mask that differs between the four quad
    // pixels is compiled into the fragment function (pixel.quadSampleMask / quadPixelMask from
    // DrawDecode, part of the shader key): it writes its coverage or discards uncovered pixels and
    // tests late (MslQuadCoverage). A mask that is the same for the whole quad is the Vulkan
    // pipeline's pSampleMask; MTLRenderPipelineDescriptor has none, so the fragment function ANDs
    // its [[sample_mask]] with APS5_SAMPLE_MASK (guest sample bits; the function shifts them by
    // APS5_SAMPLE_BASE itself), a library macro and so part of the library and pipeline keys. Under
    // the macro the function discards pixels whose covered samples the mask removes before its body
    // (a mask of 0 runs no body, as Vulkan runs no invocation) and tests late (MslAbi Version 15).
    // The remaining refusals are TechnicalDebt metal-pipeline-sample-mask.
    const bool partialSampleMask = Graphics::PartialSampleMask(graphics);
    const bool hostSampleMask = partialSampleMask && Graphics::UniformSampleMask(graphics);
    if (partialSampleMask) {
        if (fragmentReflection == nullptr) Fail("a sample mask (PA_SC_AA_MASK) that removes samples from a draw without a pixel program: Metal has no pipeline sample mask, the fragment function applies it (TechnicalDebt metal-pipeline-sample-mask)");
        const bool tested = graphics.depth.has_value() && (graphics.depthTest || graphics.stencilTest || graphics.depthBoundsTest);
        const bool earlyWithMemoryWrites = tested && Graphics::ShadersWriteMemory(std::span(fragment, 1u));
        if (hostSampleMask) {
            // Every fragment function carries [[sample_mask]] under the macro, an output-less one in an
            // otherwise empty output struct (MslStageIo.cpp hostMaskMember), so the mask also keeps the
            // masked samples of a covered pixel off the depth and stencil writes.
            // The function drops [[early_fragment_tests]] under APS5_SAMPLE_MASK (Metal applies its
            // [[sample_mask]] after early tests); a memory-writing program would then also run for
            // pixels the tests reject.
            if (fragmentReflection->earlyFragmentTests && earlyWithMemoryWrites) Fail("a sample mask (PA_SC_AA_MASK) that removes samples from a pixel program with early depth or stencil tests that writes memory: Metal applies the fragment function's [[sample_mask]] after early tests, so the program tests late (TechnicalDebt metal-pipeline-sample-mask)");
        } else {
            // Upstream ValidateSampleMask: the masked pixels are removed in the function, after the
            // tests, which an early-tested program that writes memory cannot tolerate.
            if (graphics.pixelEarlyZ && earlyWithMemoryWrites) Fail("a sample mask that differs between the pixels of a quad, from a pixel program with early depth or stencil tests that writes memory: the masked pixels are removed in the shader, after the tests (TechnicalDebt metal-pipeline-sample-mask)");
        }
    }
    // Mesh draws (B6 stage 5, MetalMesh.h): the "vertex" stage is the NGG primitive shader's mesh function.
    const bool mesh = vertexReflection.stage == Abi::Stage::Mesh && graphics.stages.mesh.has_value();
    // Tessellated draws (MetalTessellation.h): the "vertex" stage is the domain program; the local
    // and hull programs run in the object function.
    const bool tess = tessellation != nullptr && vertexReflection.stage == Abi::Stage::Domain && graphics.stages.tessellation.has_value();
    if (vertexReflection.stage != Abi::Stage::Vertex && !mesh && !tess) Fail("the vertex stage's MSL is not a vertex function");
    if (fragmentReflection != nullptr && fragmentReflection->stage != Abi::Stage::Fragment) Fail("the pixel stage's MSL is not a fragment function");
    if (vertexReflection.workgroupMemoryStrideDwords != 0u || (fragmentReflection != nullptr && fragmentReflection->workgroupMemoryStrideDwords != 0u)) Fail("device-memory LDS in a draw stage");
    // Multisampled draws (DECISIONS D27): the pipeline rasterizes graphics.samples samples into
    // MultisampleTarget host images and multisampled depth surfaces, at the guest's sample positions.
    // Sample counts the device cannot render (8x on Apple9) are emulated (DECISIONS D28): the targets
    // are 2D multisample arrays of `slices` slices of hostSamples.samples samples, and the draw is
    // recorded once per slice, each pass at its slice's guest sample positions with the fragment
    // function's sample ids and exported masks offset by the slice's first guest sample.
    const auto hostSamples = graphics.samples > 1u ? Graphics::Metal::HostSamplesOf(state.device, graphics.samples) : Graphics::Metal::HostSamples{};
    const std::uint32_t slices = hostSamples.slices;
    // Conservative rasterization is emulated (MslConservative.hpp, DECISIONS "conservative
    // rasterization emulated"), and so are per-vertex fragment inputs and barycentrics (B6): both
    // need every triangle's three vertices in the vertex function (the triangle-vertices mode).
    const std::uint32_t conservativeMode = graphics.conservativeRasterization == VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT ? 1u : graphics.conservativeRasterization == VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT ? 2u : 0u;
    const bool perVertex = fragmentReflection != nullptr && (fragmentReflection->perVertexLocations != 0u || fragmentReflection->emulatedBarycentrics != 0u);
    // Tessellated draws take patch lists, whose primitives the tessellator makes (the mesh pipeline
    // has no topology class).
    const auto primitive = graphics.rectList || tess ? Primitive{MTLPrimitiveTypeTriangle, MTLPrimitiveTopologyClassTriangle} : PrimitiveOf(graphics.topology);
    // Guest cull distances (upstream #423, MslAbi v10): Metal has none, so the vertex function
    // emulates them in one more clip distance, which needs the whole primitive: triangles draw in
    // the triangle-vertices mode, rect lists and points have their own macro, lines are refused.
    const bool cullDistances = vertexReflection.cullDistances != 0u && !mesh && !tess;
    if (cullDistances && !graphics.rectList && primitive.topologyClass == MTLPrimitiveTopologyClassLine) Fail("cull distances on lines have no Metal emulation");
    const bool cullTriangles = cullDistances && !graphics.rectList && primitive.topologyClass == MTLPrimitiveTopologyClassTriangle;
    const bool cullPoints = cullDistances && !graphics.rectList && primitive.topologyClass == MTLPrimitiveTopologyClassPoint;
    const bool triangleVertices = conservativeMode != 0u || perVertex || cullTriangles;
    // Mesh draws run per-vertex fragment inputs in the mesh function's triangle-vertices mode (MslAbi
    // Version 13: unshared vertices, the subgroup split over the grid's z). Its conservative
    // rasterization (D21's corner math needs the viewport record) is not written; mesh cull
    // distances throw in the recompiler.
    if ((mesh && conservativeMode != 0u) || (tess && triangleVertices)) {
        // Name what the draw needed.
        char detail[160];
        std::snprintf(detail, sizeof(detail), " (per-vertex locations 0x%llx, emulated barycentrics 0x%x, conservative mode %u, cull distances %u)", static_cast<unsigned long long>(fragmentReflection != nullptr ? fragmentReflection->perVertexLocations : 0u), fragmentReflection != nullptr ? fragmentReflection->emulatedBarycentrics : 0u, conservativeMode, vertexReflection.cullDistances);
        Fail(std::string(conservativeMode != 0u ? "conservative rasterization" : "per-vertex fragment inputs") + (tess ? " of tessellated draws" : " of mesh draws") + detail);
    }
    if (tess && (draw.indirect || graphics.rectList)) Fail(draw.indirect ? "indirect tessellated draws (refused as upstream)" : "tessellated rect lists");
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
    Graphics::Metal::MeshDrawPlan meshPlan;
    std::uint32_t meshSplits = 1u;
    std::uint32_t pointSplitVertices = 0u;
    if (mesh) {
        std::span<const std::byte> indices;
        if (draw.indexed) {
            const auto bytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
            state.SettleRange(draw.indexAddress, bytes);
            indices = std::span<const std::byte>(reinterpret_cast<const std::byte*>(GuestArena::HostRange(draw.indexAddress, bytes)), bytes);
        }
        meshPlan = Graphics::Metal::PlanMeshDraw(graphics, draw, indices);
        if (meshPlan.groups == 0u || meshPlan.instances == 0u) return;
        if (triangleVertices && primitive.topologyClass == MTLPrimitiveTopologyClassPoint) {
            // Points need no staging; their vertices are split by Metal's mesh output size (D78).
            const auto outputs = vertexReflection.outputLocations;
            pointSplitVertices = Abi::MeshPointSplitVertices(graphics.stages.mesh->maxVertices, graphics.stages.mesh->maxPrimitives, static_cast<std::uint32_t>(std::popcount(outputs)), static_cast<std::uint32_t>(std::popcount(fragmentReflection != nullptr ? fragmentReflection->perVertexLocations & outputs : 0u)));
            meshSplits = Abi::MeshPointSplits(graphics.stages.mesh->maxVertices, pointSplitVertices);
            if (static_cast<std::uint64_t>(meshPlan.groups) * meshPlan.instances * meshSplits > Graphics::Metal::MaxMeshThreadgroupsPerGrid) Fail("a point mesh draw of " + std::to_string(meshPlan.groups) + " groups x " + std::to_string(meshPlan.instances) + " instances x " + std::to_string(meshSplits) + " splits exceeds the mesh grid");
        } else if (triangleVertices) {
            meshSplits = Abi::MeshTriangleVerticesSplits(graphics.stages.mesh->maxPrimitives);
            if (static_cast<std::uint64_t>(meshPlan.groups) * meshPlan.instances * meshSplits > Graphics::Metal::MaxMeshThreadgroupsPerGrid) Fail("a triangle-vertices mesh draw of " + std::to_string(meshPlan.groups) + " groups x " + std::to_string(meshPlan.instances) + " instances x " + std::to_string(meshSplits) + " splits exceeds the mesh grid");
        }
    }

    // Per-vertex fragment inputs of points and lines draw in the triangle-vertices mode with one or
    // two guest vertices per primitive (MslAbi Version 14, PrimitiveVerticesMacro; DECISIONS D69),
    // mesh draws too (Version 16, D75). Conservative rasterization and cull distances stay
    // triangle-only.
    const bool pointsOrLines = primitive.topologyClass != MTLPrimitiveTopologyClassTriangle;
    if (triangleVertices && (graphics.rectList || (pointsOrLines && conservativeMode != 0u))) {
        Fail(std::string(conservativeMode != 0u ? "conservative rasterization" : "per-vertex fragment inputs") + " of " + (graphics.rectList ? "rect lists" : "points or lines"));
    }
    const std::uint32_t primitiveVertices = primitive.topologyClass == MTLPrimitiveTopologyClassPoint ? 1u : primitive.topologyClass == MTLPrimitiveTopologyClassLine ? 2u : 3u;
    // The mesh path fetches rect-list indices like any list (DECISIONS D41); the vertex function's
    // expansion counts vertex ids.
    if (graphics.rectList && draw.indexed && !mesh) Fail("indexed rect lists");
    if (draw.indexed && draw.indexSize != 2u && draw.indexSize != 4u) Fail("index size " + std::to_string(draw.indexSize));

    // Guest ranges of both stages, the vertex fetches, the indices and the color targets.
    StageBindings vertexPlan;
    StageBindings fragmentPlan;
    StageBindings localPlan;
    StageBindings hullPlan;
    state.PlanBindings(vertexResult, vertexPlan);
    if (fragmentResult != nullptr) state.PlanBindings(*fragmentResult, fragmentPlan);
    if (tess) {
        state.PlanBindings(*tessellation->local->program, localPlan);
        state.PlanBindings(*tessellation->hull->program, hullPlan);
    }
    // Vertex pulling and the guest indices: the vertex function's, or a tessellated draw's local
    // program in the object function.
    const auto& fetchResult = tess ? *tessellation->local->program : vertexResult;
    const auto& fetchReflection = fetchResult.msl.Reflection();
    auto& fetchPlan = tess ? localPlan : vertexPlan;
    struct Fetch {
        std::uint32_t location;
        Abi::VertexFetchRecord record;
        std::uint64_t begin;
        std::uint64_t end;
    };
    std::vector<Fetch> fetches;
    for (const auto& attribute : fetchResult.vertexAttributes) {
        if (attribute.location >= 32u || (fetchReflection.vertexFetchLocations & (1u << attribute.location)) == 0u) continue;
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
            if (bytes != 0u) fetchPlan.reads.push_back({begin, end});
        }
        fetches.push_back({attribute.location, record, begin, end});
    }
    std::uint64_t indexBegin = 0;
    std::uint64_t indexBytes = 0;
    if (draw.indexed) {
        indexBegin = draw.indexAddress;
        indexBytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
        if (indexBegin == 0u || indexBytes == 0u) Fail("an indexed draw without an index buffer");
        fetchPlan.reads.push_back({indexBegin, indexBegin + indexBytes});
    }

    // Color attachments by Vulkan location (exportIndex), written as the blend states allow.
    std::vector<ColorAttachment> colors;
    std::uint32_t colorOutputs = fragmentReflection != nullptr ? fragmentReflection->colorOutputs : 0u;
    StageBindings targetPlan;
    for (const auto& target : graphics.colors) {
        if (target.exportIndex >= 8u) Fail("color target location " + std::to_string(target.exportIndex));
        if (target.samples > 1u) {
            // Upstream Draw.cpp: a multisampled target is its host image (MultisampleTargetView,
            // which also applies a pending CMASK fast clear); its guest memory is not read or written.
            const auto* imageContext = state.images->ImageContext();
            if (imageContext == nullptr) Fail("a multisampled color target without the device's image context");
            if (target.samples != graphics.samples) Fail("a color target's sample count differs from the rasterizer's");
            const auto format = Graphics::Metal::PixelFormatOf(target.format);
            if (format.format == MTLPixelFormatInvalid) Fail("color target format " + std::to_string(target.format) + " has no Metal pixel format");
            ColorAttachment attachment {&target, format.format, target.address, target.address, nullptr, false, nullptr};
            attachment.multisample = Graphics::Metal::TextureOfView(Graphics::MultisampleTargetView(*imageContext, target));
            if (attachment.multisample == nil || attachment.multisample.sampleCount != hostSamples.samples || attachment.multisample.arrayLength != slices) Fail("the multisampled color target has no host image of its sample count");
            colors.push_back(attachment);
            continue;
        }
        // Tiled targets are the image cache's resident images (upstream image path): their writes
        // stay on the GPU until a reader of the memory flushes them (StorageTexture::MarkDirty).
        std::shared_ptr<Graphics::StorageTexture> resident;
        // Upstream Draw.cpp (linearResident): a linear target over memory that already has a host
        // import is a resident image too, so the draw is recorded instead of synced (debug aid
        // APS5_NO_RESIDENT_LINEAR=1 keeps it in place).
        static const bool residentLinear = std::getenv("APS5_NO_RESIDENT_LINEAR") == nullptr;
        const bool linearResident = residentLinear && target.tileMode == Graphics::ColorTileMode::Linear && target.depth <= 1u && target.arrayAddress == 0u && !target.mipTail && state.images->ImageContext() != nullptr && Graphics::ExistingHostImportFor(*state.images->ImageContext(), target.address, Graphics::ColorTargetLayout(target.extent.width, target.extent.height, target.tileMode, target.elementBytes, target.pipeBankXor).Bytes()) != nullptr;
        if ((target.tileMode != Graphics::ColorTileMode::Linear || linearResident) && state.images->ImageContext() != nullptr && ResidentTargets()) {
            const auto& imageContext = *state.images->ImageContext();
            if (Graphics::AttachmentProxyFormat(imageContext, target.format) != VK_FORMAT_UNDEFINED) Fail("color target format " + std::to_string(target.format) + " needs an attachment proxy");
            if (target.arrayLastSlice != 0u) {
                if (target.cmaskAddress != 0u) Fail("a slice of a 2D array color target with CMASK");
                const auto array = ArraySurfaceForTarget(target);
                if (target.surfaceAddress != target.arrayAddress + target.arraySlice * Graphics::DescribeSurface(array).layerBytes) Fail("the array image's layer pitch differs from the color view's slice offset");
                resident = Graphics::CachedStorageSurface(imageContext, array);
            } else {
                resident = Graphics::CachedStorageSurface(imageContext, Graphics::SurfaceForTarget(target));
            }
            if (resident == nullptr || !resident->Attachable()) Fail("storage format cannot be a color attachment");
            // CMASK fast clears and DCC clear keys land in guest memory, which the image then picks
            // up (Refresh).
            state.MaterializeDcc(target, resident.get());
            state.MaterializeCmask(target, resident.get());
        } else if (target.depth > 1u || target.arrayAddress != 0u) {
            // Linear targets, and tiled ones without the image cache (APS5_NO_RESIDENT_TARGETS):
            // a mip is its own surface at the view's address (as upstream's transfer and GPU tiled
            // fallbacks), a packed tail mip a window of the tail block (TileTarget).
            Fail("color targets with slices or array layers outside the image cache (linear, or APS5_NO_RESIDENT_TARGETS)");
        } else if (target.mipTail && target.tileMode == Graphics::ColorTileMode::Linear) {
            Fail("a packed mip tail of a linear color target");
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

    // Layered draws (upstream #3763, State::renderLayers) render into a resident 3D target's slice
    // range; DecodeState already refuses a depth target and mixed slice counts.
    if (graphics.renderLayers > 1u) {
        for (const auto& attachment : colors) {
            if (attachment.resident == nullptr || attachment.multisample != nil || attachment.target->depth <= 1u) Fail("a layered draw into a color target that is not a resident single-sample 3D image");
        }
    }

    // CMASK fast clears land in guest memory before the targets load from it.
    for (const auto& attachment : colors) {
        if (attachment.resident != nullptr || attachment.multisample != nil) continue;
        state.MaterializeDcc(*attachment.target, nullptr);
        state.MaterializeCmask(*attachment.target, nullptr);
    }

    const StageBindings* plans[] = {&vertexPlan, &fragmentPlan, &targetPlan, &localPlan, &hullPlan};
    state.SettleBindings(plans);
    Graphics::Metal::ImageBindings imageBindings(*state.images, [&state](std::uint64_t address, std::uint64_t bytes) { state.SettleRange(address, bytes); });
    state.PlanImages(vertexResult, imageBindings, vertexPlan);
    if (fragmentResult != nullptr) state.PlanImages(*fragmentResult, imageBindings, fragmentPlan);
    if (tess) {
        state.PlanImages(*tessellation->local->program, imageBindings, localPlan);
        state.PlanImages(*tessellation->hull->program, imageBindings, hullPlan);
    }

    // Tiled targets: their cache entries, and the guest bytes they load from (settled now, read
    // when the pass begins).
    std::uint64_t passKey = Mix(0x6d65746c64726177ull, graphics.renderExtent.width | (static_cast<std::uint64_t>(graphics.renderExtent.height) << 32u));
    for (auto& attachment : colors) {
        const auto& target = *attachment.target;
        passKey = Mix(Mix(Mix(passKey, target.address), target.format | (static_cast<std::uint64_t>(target.exportIndex) << 32u)), static_cast<std::uint64_t>(target.tileMode) | (static_cast<std::uint64_t>(target.extent.width) << 16u) | (static_cast<std::uint64_t>(target.extent.height) << 40u));
        if (attachment.multisample != nil) {
            passKey = Mix(passKey, reinterpret_cast<std::uint64_t>((__bridge void*)attachment.multisample));
            continue;
        }
        if (attachment.resident != nullptr) {
            passKey = Mix(passKey, reinterpret_cast<std::uint64_t>(attachment.resident.get()) ^ (static_cast<std::uint64_t>(target.mip) << 48u) ^ (static_cast<std::uint64_t>(target.arraySlice | target.depthSlice) << 32u) ^ (static_cast<std::uint64_t>(target.layers) << 56u));
            continue;
        }
        if (target.tileMode == Graphics::ColorTileMode::Linear) continue;
        std::uint32_t tailX = 0;
        std::uint32_t tailY = 0;
        if (target.mipTail) {
            const auto mip = Graphics::ComputeColorTargetMipLayout(target.tileMode, target.elementBytes, target.surfaceExtent.width, target.surfaceExtent.height, target.mipCount, target.mip);
            if (!mip.tail || mip.width != target.extent.width || mip.height != target.extent.height) Fail("the packed mip tail layout of a color target differs from its view");
            tailX = mip.tailX;
            tailY = mip.tailY;
            passKey = Mix(passKey, (static_cast<std::uint64_t>(tailX) << 32u) | tailY | (1ull << 63u));
        }
        for (const auto& entry : cache.tiled) {
            if (entry->address == target.address && entry->extent.width == target.extent.width && entry->extent.height == target.extent.height && entry->format == target.format && entry->tileMode == target.tileMode && entry->elementBytes == target.elementBytes && entry->pipeBankXor == target.pipeBankXor && entry->tail == target.mipTail && entry->tailX == tailX && entry->tailY == tailY) {
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
            attachment.tiled = std::make_shared<DrawCache::TiledTarget>(DrawCache::TiledTarget{target.address, target.extent, target.format, target.tileMode, target.elementBytes, target.pipeBankXor, texture, 0u, target.mipTail, tailX, tailY});
            if (target.mipTail && TiledGuestLayout(*attachment.tiled).Bytes() != Graphics::ColorTargetLayout(target.extent.width, target.extent.height, target.tileMode, target.elementBytes, target.pipeBankXor).Bytes()) Fail("a packed mip tail of a color target spans more than its tail block");
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
        if (target.samples != graphics.samples) Fail("the depth target's sample count differs from the rasterizer's");
        const auto format = DepthFormat(target);
        bool created = false;
        const auto use = Graphics::Metal::AcquireDepthSurface(state.device, target, format, &created);
        if (created) {
            const std::array<id<MTLTexture>, 1> resident {use.texture};
            state.images->MakeResident(resident);
        }
        depthSurface = DrawCache::DepthSurface{target, format, use.texture, use.clearDepth, use.clearStencil, use.depth, use.stencil};
        depth = &*depthSurface;
        passKey = Mix(Mix(passKey, target.address), static_cast<std::uint64_t>(format));
    }
    // The sample count and positions are pass state.
    std::array<MTLSamplePosition, 8> samplePositions {};
    if (graphics.samples > 1u) {
        // Upstream b6cde5698 passes patterns that differ between the pixels of the guest's 2x2 grid
        // as a VK_EXT_sample_locations 2x2 grid; Metal's setSamplePositions takes one pattern for
        // every pixel (docs/dev/TechnicalDebt.md). Pixel 0's pattern leads the array either way.
        if (graphics.sampleLocationGrid) Fail("sample locations that vary between the pixels of the 2x2 grid (PA_SC_AA_SAMPLE_LOCS_PIXEL_*): Metal sets one sample pattern for every pixel");
        const auto locations = Graphics::SampleLocations(graphics);
        passKey = Mix(passKey, graphics.samples);
        for (std::uint32_t sample = 0; sample < graphics.samples; ++sample) {
            samplePositions[sample] = MTLSamplePositionMake(locations[sample].x, locations[sample].y);
            passKey = Mix(passKey, static_cast<std::uint64_t>(locations[sample].x * 16.0f) | (static_cast<std::uint64_t>(locations[sample].y * 16.0f) << 8u));
        }
    }
    // Upstream 9bf98992ff: a multisampled rasterizer over one-sample targets draws one sample at sample
    // 0's location (Graphics::State keeps it as a custom one-sample location). Without programmable
    // positions the draw samples the pixel centre, as upstream's device without sample locations does.
    const bool oneSamplePosition = graphics.samples == 1u && graphics.customSampleLocations && !graphics.sampleLocationGrid && state.device.programmableSamplePositionsSupported;
    if (oneSamplePosition) {
        const auto locations = Graphics::SampleLocations(graphics);
        samplePositions[0] = MTLSamplePositionMake(locations[0].x, locations[0].y);
        passKey = Mix(Mix(passKey, 1u), static_cast<std::uint64_t>(locations[0].x * 16.0f) | (static_cast<std::uint64_t>(locations[0].y * 16.0f) << 8u));
    }

    // Pipelines: a second channel pass for alternate-alpha 8_8 targets, a back-face pass for
    // per-face depth bias (as upstream's secondChannel and backFaces pipelines).
    Graphics::Metal::RenderPipelineRequest request;
    request.vertex = Graphics::Metal::PipelineFunction::From(vertexResult);
    request.mesh = mesh;
    request.meshThreads = mesh ? vertexReflection.threadsPerThreadgroup[0] : 0u;
    std::shared_ptr<const ShaderRecompiler::MslTessellationPipeline> linked;
    if (tess) {
        // The linked libraries carry their function constants resolved (LinkMslTessellation).
        linked = Graphics::Metal::LinkTessellation(*tessellation, *graphics.stages.tessellation);
        request.vertex = Graphics::Metal::PipelineFunction{std::make_shared<const ShaderRecompiler::MslShader>(linked->mesh), {}, {}};
        request.object = Graphics::Metal::PipelineFunction{std::make_shared<const ShaderRecompiler::MslShader>(linked->object), {}, {}};
        request.mesh = true;
        request.meshThreads = linked->meshThreads;
        request.objectThreads = linked->objectThreads;
        request.meshGroups = linked->maxMeshThreadgroups;
    }
    // Mesh draws expand rect lists from the mesh configuration (DECISIONS D41), not the macro.
    if (graphics.rectList && !mesh) request.vertex.macros.push_back({Abi::RectListMacro, "1"});
    if (cullPoints) request.vertex.macros.push_back({Abi::CullPointsMacro, "1"});
    // Mesh draws of points and lines output that topology in either mode (MslAbi Version 16, D75).
    if (mesh && !graphics.rectList && primitiveVertices != 3u) request.vertex.macros.push_back({Abi::PrimitiveVerticesMacro, std::to_string(primitiveVertices) + "u"});
    // Point lists write a point size when the guest program exports none (MslAbi Version 16, D75).
    if (!tess && !graphics.rectList && primitiveVertices == 1u) request.vertex.macros.push_back({Abi::PointsMacro, "1"});
    // A mesh line under the last-vertex convention runs from its second vertex in the plain mode too.
    if (mesh && !triangleVertices && primitiveVertices == 2u && graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT) request.vertex.macros.push_back({Abi::ProvokingLastMacro, "1"});
    if (triangleVertices) {
        request.vertex.macros.push_back({Abi::TriangleVerticesMacro, "1"});

        if (primitiveVertices != 3u && !mesh) request.vertex.macros.push_back({Abi::PrimitiveVerticesMacro, std::to_string(primitiveVertices) + "u"});
        if (pointSplitVertices != 0u) request.vertex.macros.push_back({Abi::PointSplitVerticesMacro, std::to_string(pointSplitVertices) + "u"});
        if (conservativeMode != 0u) request.vertex.macros.push_back({Abi::ConservativeMacro, std::to_string(conservativeMode)});
        if (perVertex) {
            request.vertex.macros.push_back({Abi::PerVertexLocationsMacro, std::to_string(static_cast<std::uint32_t>(fragmentReflection->perVertexLocations)) + "u"});
            request.vertex.macros.push_back({Abi::PerVertexLocationsHighMacro, std::to_string(static_cast<std::uint32_t>(fragmentReflection->perVertexLocations >> 32u)) + "u"});
            request.vertex.macros.push_back({Abi::BarycentricsMacro, std::to_string(fragmentReflection->emulatedBarycentrics) + "u"});
        }
        // The vertex path rotates its expansion on the CPU; the mesh function rotates each primitive.
        if (mesh && graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT) request.vertex.macros.push_back({Abi::ProvokingLastMacro, "1"});
    }
    if (graphics.negativeOneToOne) request.vertex.macros.push_back({Abi::DepthNegativeOneToOneMacro, "1"});
    if (graphics.windowSpacePosition) {
        // The window-space epilogue is in the vertex and mesh stage functions; the tessellation
        // pipeline's domain output and the triangle-vertices corner math work on clip positions.
        // The mesh function's triangle-vertices mode remaps each staged vertex before it is copied.
        if (tess || (triangleVertices && !mesh)) throw std::runtime_error(std::string("Metal draw: window-space positions (PA_CL_VTE_CNTL 0x300) with ") + (tess ? "tessellation" : "a triangle-vertices draw") + " are not implemented");
        request.vertex.macros.push_back({Abi::WindowSpaceMacro, "1"});
        request.vertex.macros.push_back({Abi::WindowSpaceWidthMacro, std::to_string(static_cast<std::uint32_t>(graphics.viewport.width))});
        request.vertex.macros.push_back({Abi::WindowSpaceHeightMacro, std::to_string(static_cast<std::uint32_t>(graphics.viewport.height))});
    }
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
        // Blending into a target with a reversed component order (upstream 16ea32c97d/c434467f6d,
        // State::reversedBlend): MRT0's SRC1 blend factors read the second source aps5_rb, which the
        // function writes at color(0) index(1) under ReversedBlendMacro (MslAbi Version 15). The
        // library refuses it beside APS5_DUAL_SOURCE.
        const bool reversedBlend = graphics.reversedBlend != 0u || std::any_of(fragmentResult->specialization.begin(), fragmentResult->specialization.end(), [](const auto& constant) { return constant.id == ShaderRecompiler::PipelineSpecialization::ReversedBlend && constant.value != 0u; });
        if (reversedBlend) {
            if (dualSource) throw std::runtime_error("Metal draw: a reversed component order blend beside dual-source blending is not supported (one color(0) index(1) output)");
            function.macros.push_back({Abi::ReversedBlendMacro, "1"});
        }
        if (conservativeMode != 0u) function.macros.push_back({Abi::ConservativeMacro, std::to_string(conservativeMode)});
        // The uniform PA_SC_AA_MASK (upstream HostSampleMask, guest sample bits; none when full).
        if (hostSampleMask) function.macros.push_back({Abi::SampleMaskMacro, std::to_string(static_cast<std::uint32_t>(Graphics::HostSampleMask(graphics))) + "u"});
        // Point and line draws output only their own corners' per-vertex values (DECISIONS D78).
        if (triangleVertices && primitiveVertices != 3u) function.macros.push_back({Abi::PrimitiveVerticesMacro, std::to_string(primitiveVertices) + "u"});
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
    request.rasterSampleCount = hostSamples.samples;
    // B6 emulates alpha-to-coverage in the fragment function's [[sample_mask]] (the guest's rule);
    // Metal's own would apply on top.
    request.alphaToCoverage = graphics.alphaToCoverage && !(fragmentReflection != nullptr && fragmentReflection->exportsSampleMask);
    request.topology = mesh || tess ? MTLPrimitiveTopologyClassUnspecified : primitive.topologyClass;
    struct PipelinePass {
        id<MTLRenderPipelineState> pipeline;
        MTLCullMode cull;
        float biasConstant;
        float biasSlope;
    };
    const auto cullOf = [](VkCullModeFlags mode) {
        return mode == VK_CULL_MODE_FRONT_BIT ? MTLCullModeFront : mode == VK_CULL_MODE_BACK_BIT ? MTLCullModeBack : MTLCullModeNone;
    };
    if (graphics.cullMode == VK_CULL_MODE_FRONT_AND_BACK) return;
    // Upstream 7d34a6aa8: a 16-bit depth with stencil is stored in a wider host format, whose
    // constant-bias unit differs (HostDepthBiasConstant refuses what no constant reproduces).
    const auto hostBias = [&](float constant) {
        if (constant == 0.0f || !graphics.depth || state.images->ImageContext() == nullptr) return constant;
        return Graphics::HostDepthBiasConstant(*state.images->ImageContext(), graphics.depth->format, constant);
    };
    const float biasConstant = graphics.depthBias ? hostBias(graphics.depthBiasConstant) : 0.0f;
    const float biasSlope = graphics.depthBias ? graphics.depthBiasSlope : 0.0f;
    const auto buildPipelinePasses = [&](const Graphics::Metal::RenderPipelineRequest& request) {
        std::vector<PipelinePass> pipelinePasses;
        if (Graphics::SplitsAlternateAlpha(graphics)) {
            // Upstream Pipeline.cpp's limits of the second-channel split (the second pass would repeat
            // the draw's other effects).
            const auto keepsStencil = [](const VkStencilOpState& face) { return face.writeMask == 0 || (face.failOp == VK_STENCIL_OP_KEEP && face.passOp == VK_STENCIL_OP_KEEP && face.depthFailOp == VK_STENCIL_OP_KEEP); };
            if ((graphics.depth && (graphics.depthWrite || (graphics.stencilTest && (!keepsStencil(graphics.stencilFront) || !keepsStencil(graphics.stencilBack))))) || graphics.depthBiasPerFace) throw std::runtime_error("AGC graphics: blending into an 8_8 color target with the alternate component order and a separate alpha blend from a draw that writes depth or stencil is not implemented");
            const std::array<Graphics::CompiledShader, 2> stages {{vertex, fragment != nullptr ? *fragment : vertex}};
            if (Graphics::ShadersWriteMemory(std::span(stages.data(), fragment != nullptr ? 2u : 1u))) throw std::runtime_error("AGC graphics: blending into an 8_8 color target with the alternate component order and a separate alpha blend from a draw whose shaders write memory is not implemented");
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
            pipelinePasses.push_back({pipeline, MTLCullModeFront, hostBias(graphics.backDepthBiasConstant), graphics.backDepthBiasSlope});
        } else {
            pipelinePasses.push_back({state.pipelines->Render(request), cullOf(graphics.cullMode), biasConstant, biasSlope});
        }
        return pipelinePasses;
    };
    // One pipeline set per slice of an emulated sample count (DECISIONS D28): slice k's fragment
    // function sees guest samples k * h .. k * h + h - 1 (SampleBaseMacro); the guest sample count
    // is SampleCountMacro (alpha-to-coverage). Functions that read no sample id and export no mask
    // share slice 0's pipelines.
    std::vector<std::vector<PipelinePass>> slicePipelines;
    if (slices > 1u && request.fragment.has_value()) request.fragment->macros.push_back({Abi::SampleCountMacro, std::to_string(graphics.samples)});
    slicePipelines.push_back(buildPipelinePasses(request));
    // APS5_SAMPLE_MASK makes the function write [[sample_mask]] (shifted by APS5_SAMPLE_BASE).
    const bool sliceSpecific = fragmentReflection != nullptr && (fragmentReflection->exportsSampleMask || fragmentReflection->perSample || hostSampleMask);
    for (std::uint32_t slice = 1; slice < slices; ++slice) {
        if (!sliceSpecific) {
            slicePipelines.push_back(slicePipelines.front());
            continue;
        }
        auto sliceRequest = request;
        sliceRequest.fragment->macros.push_back({Abi::SampleBaseMacro, std::to_string(slice * hostSamples.samples)});
        slicePipelines.push_back(buildPipelinePasses(sliceRequest));
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
    // Upstream f2a4236ab draws rect lists with the last vertex provoking by taking the flat inputs from
    // the rectangle's third vertex. The MSL rect-list paths cannot tell which parameters are flat and
    // would hand every corner the first vertex's: refused while the pixel shader has a flat input
    // (TechnicalDebt metal-rect-list-last-vertex).
    if (graphics.rectList && graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT && fragment != nullptr && fragment->program != nullptr) {
        const auto& parameters = fragment->program->fragmentParameters;
        if (std::any_of(parameters.begin(), parameters.end(), [](const ShaderRecompiler::FragmentParameter& parameter) { return parameter.flat; })) Fail("rect lists with the last vertex provoking and flat pixel inputs (TechnicalDebt metal-rect-list-last-vertex)");
    }
    // Upstream 99edf9435 (Pipeline.cpp ValidateProvokingVertex): P10 of a flat input reads the
    // provoking vertex, measured only with the last vertex provoking.
    if (graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT && fragment != nullptr && fragment->program != nullptr) {
        const auto& parameters = fragment->program->fragmentParameters;
        if (std::any_of(parameters.begin(), parameters.end(), [](const ShaderRecompiler::FragmentParameter& parameter) { return parameter.flatP10; })) Fail("v_interp_mov p10 of a flat input with the first provoking vertex is not measured");
    }
    // Indexed lists with primitive restart (PrimitiveListRestart) expand too: Metal restarts strips
    // only, and a restart index ends a list's current primitive (its incomplete vertices dropped,
    // as VK_EXT_primitive_topology_list_restart and the guest's VGT reset index do).
    const bool listTopology = graphics.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST || graphics.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || graphics.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    const bool listRestart = draw.indexed && graphics.primitiveRestart && listTopology && !graphics.rectList;
    const bool expand = !mesh && !tess && (primitive.fan || triangleVertices || listRestart || (lastVertex && primitive.topologyClass != MTLPrimitiveTopologyClassPoint));
    std::vector<std::uint32_t> fanIndices;
    std::vector<DrawCall> fanCalls;
    // D81: indices that recorded GPU work still writes are not read on the CPU (that waited for the
    // whole recorder, Recorder.Sync inside the draw). Without a primitive restart that matters, the
    // expansion depends on positions alone: it is planned over positions, and a helper kernel
    // gathers the guest indices at those positions into the draw's index list on the GPU, ordered
    // after the writers in queue order. Restarts (values decide the segments), the triangle-vertices
    // table (its ids add each call's base vertex on the CPU), unwrappable or misaligned memory and
    // a pending CPU write-back over the indices (FlushForGpuRead waited then) read them on the CPU
    // as before. APS5_CPU_INDEX_EXPANSION=1 always reads them on the CPU.
    const bool restartMatters = graphics.primitiveRestart && (listTopology || graphics.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || graphics.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN || graphics.topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
    std::uint64_t gatherSource = 0;
    if (expand && draw.indexed && !triangleVertices && !restartMatters && GpuIndexGather() && (draw.indexSize == 2u ? indexBegin % 2u == 0u : indexBegin % 4u == 0u) && (state.recorder->PendingWriteOverlaps(indexBegin, indexBytes) || state.recorder->PendingLabelIn(indexBegin, indexBytes))) {
        gatherSource = state.TryWrap(indexBegin, indexBytes);
        if (gatherSource != 0u && !Graphics::Metal::FlushForGpuRead(indexBegin, indexBytes)) gatherSource = 0u;
    }
    const bool gather = gatherSource != 0u;
    if (expand) {
        std::vector<std::byte> guest;
        if (draw.indexed && !gather) {
            if (state.recorder->PendingWriteOverlaps(indexBegin, indexBytes) || state.recorder->PendingLabelIn(indexBegin, indexBytes)) indexSettles.fetch_add(1u, std::memory_order_relaxed);
            state.SettleRange(indexBegin, indexBytes);
            guest.resize(indexBytes);
            std::memcpy(guest.data(), reinterpret_cast<const void*>(GuestArena::HostRange(indexBegin, indexBytes)), indexBytes);
        }
        const auto indexAt = [&](std::uint32_t position) -> std::uint32_t {
            if (gather) return position;
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
            case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
                fanIndices.insert(fanIndices.end(), v.begin(), v.end());
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
                if ((strips || listTopology) && graphics.primitiveRestart && value == restart) {
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
    // D81: the positions become guest indices on the GPU, before the pass (a compute encoder ends an
    // open one; the render encoder's queue barrier orders the index fetch after the kernel).
    std::uint64_t gatheredAddress = 0;
    if (gather && !fanIndices.empty()) {
        auto& stream = Graphics::Metal::Stream(recorder.Commands());
        const auto positions = stream.Write(std::as_bytes(std::span(fanIndices)), 16u);
        const auto gathered = stream.Allocate(fanIndices.size() * 4u, 16u);
        struct {
            std::uint64_t source;
            std::uint64_t positions;
            std::uint64_t target;
            std::uint32_t count;
            std::uint32_t indexSize;
        } parameters{gatherSource, positions.address, gathered.address, static_cast<std::uint32_t>(fanIndices.size()), draw.indexSize};
        static_assert(sizeof(parameters) == 32);
        // Earlier recorded writers of the indices (this encoder's dispatches and blits) first.
        stream.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        id<MTLComputePipelineState> pipeline = Graphics::Metal::HelperPipeline(state.device, Graphics::Metal::HelperKernel::IndexGather);
        const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256u);
        stream.DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake((fanIndices.size() + width - 1u) / width, 1u, 1u), MTLSizeMake(width, 1u, 1u));
        state.KeepWrapped(indexBegin, indexBytes);
        recorder.NotePendingRead(indexBegin, static_cast<std::size_t>(indexBytes), Recorder::ReadKind::DrawInput);
        gatheredAddress = gathered.address;
        indexGathers.fetch_add(1u, std::memory_order_relaxed);
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
    auto held = std::make_shared<Held>();
    std::uint64_t recorded = 0;
    bool anyContinued = false;
    // One pass per slice of the targets (one unless the sample count is emulated, DECISIONS D28).
    const auto slicePassKeyOf = [&](std::uint32_t slice) { return slices > 1u ? Mix(passKey, 0x736c696365ull + slice) : passKey; };
    const auto recordSlice = [&](std::uint32_t slice) {
        const auto slicePassKey = slicePassKeyOf(slice);
        // A pending depth/stencil clear is a load action: it needs a new pass.
        const bool clears = depth != nullptr && (depth->clearDepth || depth->clearStencil);
        const auto start = recorder.StartDrawPass(slicePassKey, clears, false, access);
        auto& stream = Graphics::Metal::Stream(start.commands);
        const bool continued = start.continued && cache.passKey == slicePassKey && !cache.passTextures.empty();
        id<MTL4RenderCommandEncoder> encoder = nil;
        std::vector<std::shared_ptr<DrawCache::TiledTarget>> passTiled;
        if (continued) {
            encoder = stream.Render();
            if (encoder == nil) Fail("the recorder continued a render pass that has no encoder");
        } else {
            // The occlusion counter slot of the pass's visibility buffer (an occlusion query open in
            // the guest's command stream: ASTRO BOT's first sampled draw threw "an occlusion segment
            // begins without a reset query slot", nothing on this path took a slot).
            recorder.PrepareSamplePass();
            // Tiled targets load from guest memory (CPU detile into upload memory, then a copy).
            std::vector<id<MTLTexture>> textures;
            cache.passTextures.clear();
            MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
            std::uint32_t width = graphics.renderExtent.width;
            std::uint32_t height = graphics.renderExtent.height;
            for (auto& attachment : colors) {
                const auto& target = *attachment.target;
                id<MTLTexture> texture = nil;
                if (attachment.multisample != nil) {
                    texture = attachment.multisample;
                } else if (attachment.resident != nullptr) {
                    texture = Graphics::Metal::TextureOfView(attachment.resident->AttachmentView(target.format, target.mip, target.depthSlice, target.layers, target.arraySlice));
                    if (texture == nil) Fail("the resident color target has no attachment view");
                } else if (attachment.tiled != nullptr) {
                    texture = attachment.tiled->texture;
                    if (attachment.load) {
                        const auto pitch = static_cast<std::size_t>(target.extent.width) * target.elementBytes;
                        const auto linearBytes = pitch * target.extent.height;
                        auto upload = stream.Allocate(linearBytes, 256u);
                        DetileTarget(*attachment.tiled, std::span<std::byte>(upload.host, linearBytes));
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
                // A volume target's attachment view is the whole 3D mip (StorageTexture::AttachmentView):
                // the pass starts at the view's first slice, and a layered draw's render target array
                // index counts from it (upstream #3763; MoltenVK maps a 2D-array view of a 3D image the
                // same way, depthPlane = baseArrayLayer).
                if (texture.textureType == MTLTextureType3D) color.depthPlane = target.depthSlice;
                if (attachment.multisample != nil) color.slice = slice;
                color.loadAction = MTLLoadActionLoad;
                color.storeAction = MTLStoreActionStore;
                cache.passTextures.push_back(texture);
            }
            if (depth != nullptr) {
                width = std::min<std::uint32_t>(width, depth->target.extent.width);
                height = std::min<std::uint32_t>(height, depth->target.extent.height);
                cache.passTextures.push_back(depth->texture);
                if (HasDepth(depth->format)) {
                    pass.depthAttachment.texture = depth->texture;
                    if (graphics.samples > 1u) pass.depthAttachment.slice = slice;
                    pass.depthAttachment.loadAction = depth->clearDepth ? MTLLoadActionClear : MTLLoadActionLoad;
                    pass.depthAttachment.clearDepth = depth->depthValue;
                    pass.depthAttachment.storeAction = MTLStoreActionStore;
                }
                if (HasStencil(depth->format)) {
                    pass.stencilAttachment.texture = depth->texture;
                    if (graphics.samples > 1u) pass.stencilAttachment.slice = slice;
                    pass.stencilAttachment.loadAction = depth->clearStencil ? MTLLoadActionClear : MTLLoadActionLoad;
                    pass.stencilAttachment.clearStencil = depth->stencilValue;
                    pass.stencilAttachment.storeAction = MTLStoreActionStore;
                }
            }
            if (width == 0u || height == 0u) Fail("a render pass without an extent");
            pass.renderTargetWidth = width;
            pass.renderTargetHeight = height;
            // Layered draws (upstream #3763): the vertex stage's render_target_array_index picks
            // one of the color view's slices.
            if (graphics.renderLayers > 1u) pass.renderTargetArrayLength = graphics.renderLayers;
            if (colors.empty() && depth == nullptr) pass.defaultRasterSampleCount = hostSamples.samples;
            // The guest's sample positions (standard or PA_SC_AA_SAMPLE_LOCS custom ones), as upstream's
            // VK_EXT_sample_locations: Metal's are in [0, 1) from the pixel's top-left corner in 1/16
            // steps, the same grid (Graphics::SampleLocations).
            if (graphics.samples > 1u) {
                if (!state.device.programmableSamplePositionsSupported) Fail("the device has no programmable sample positions");
                [pass setSamplePositions:samplePositions.data() + slice * hostSamples.samples count:hostSamples.samples];
            } else if (oneSamplePosition) {
                [pass setSamplePositions:samplePositions.data() count:1];
            }
            if (void* visibility = recorder.SampleVisibilityBuffer()) {
                pass.visibilityResultBuffer = (__bridge id<MTLBuffer>)visibility;
                pass.visibilityResultType = MTLVisibilityResultTypeAccumulate;
            }
            cache.passKey = slicePassKey;
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
        const auto localTable = tess ? state.WriteTable(*tessellation->local->program, snapshots, stream, localPlan, copyPush(*tessellation->local->program, tessellation->local->pushConstantOffset)) : 0u;
        const auto hullTable = tess ? state.WriteTable(*tessellation->hull->program, snapshots, stream, hullPlan, copyPush(*tessellation->hull->program, tessellation->hull->pushConstantOffset)) : 0u;
        if (mesh) {
            // The mesh draw words after the mesh function's own push data (pushDrawConstants).
            if (vertexResult.pushConstants.size() > ShaderRecompiler::MeshDrawPushOffsetBytes) Fail("mesh push constants overlap the mesh draw words");
            auto words = meshPlan.words;
            if (!meshPlan.arguments.empty()) {
                const auto arguments = stream.Write(std::span<const std::byte>(meshPlan.arguments), 16u).address;
                words[4] = static_cast<std::uint32_t>(arguments);
                words[5] = static_cast<std::uint32_t>(arguments >> 32u);
            }
            std::memcpy(push.data() + ShaderRecompiler::MeshDrawPushOffsetBytes, words.data(), sizeof(words));
        }
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
        std::uint64_t rasterOrderAddress = driverAddress;
        if (fragmentReflection != nullptr && fragmentReflection->rasterOrder) {
            if (cache.rasterOrder == nil) {
                cache.rasterOrder = [state.device newBufferWithLength:Abi::RasterOrderBytes options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
                if (cache.rasterOrder == nil) Fail("cannot allocate the raster order buffer");
                ::AgcDriver::Metal::Residency::Add(state.residency, cache.rasterOrder);
                ::AgcDriver::Metal::Residency::Commit(state.residency);
            }
            rasterOrderAddress = cache.rasterOrder.gpuAddress;
        }
        [cache.fragmentArguments setAddress:rasterOrderAddress atIndex:Abi::Index(Abi::BufferIndex::RasterOrder)];
        if (tess) {
            // The object stage (MetalTessellation.h): the local program's bindings, the hull program's
            // table and, per call below, the TessellationDraw record.
            [cache.objectArguments setAddress:localTable != 0u ? localTable : driverAddress atIndex:Abi::Index(Abi::BufferIndex::DescriptorTable)];
            [cache.objectArguments setAddress:pushAddress atIndex:Abi::Index(Abi::BufferIndex::PushData)];
            [cache.objectArguments setAddress:driverAddress atIndex:Abi::Index(Abi::BufferIndex::WorkgroupMemory)];
            [cache.objectArguments setAddress:driverAddress atIndex:Abi::Index(Abi::BufferIndex::DriverData)];
            [cache.objectArguments setAddress:fetchAddress atIndex:Abi::Index(Abi::BufferIndex::VertexFetch)];
            [cache.objectArguments setAddress:hullTable != 0u ? hullTable : driverAddress atIndex:Abi::Index(Abi::BufferIndex::TessellationHullTable)];
            [cache.objectArguments setAddress:driverAddress atIndex:Abi::Index(Abi::BufferIndex::TessellationDraw)];
            [encoder setArgumentTable:cache.objectArguments atStages:MTLRenderStageObject];
        }
        [encoder setArgumentTable:cache.vertexArguments atStages:mesh || tess ? MTLRenderStageMesh : MTLRenderStageVertex];
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
        const auto drawType = expand ? (primitive.topologyClass == MTLPrimitiveTopologyClassLine ? MTLPrimitiveTypeLine : primitive.topologyClass == MTLPrimitiveTopologyClassPoint ? MTLPrimitiveTypePoint : MTLPrimitiveTypeTriangle) : primitive.type;
        const auto& drawCalls = expanded ? fanCalls : calls;
        const bool indexedDraw = (draw.indexed || expanded) && !triangleVertices;
        if (triangleVertices && !mesh) {
            // The triangle-vertices table (MslAbi v5, vertex slot TriangleVertices): the header, then
            // every triangle's guest vertex ids (the expansion's lists with each call's base vertex),
            // drawn non-indexed: host vertex v of a call starting at table index f is v = f + i.
            std::vector<std::byte> table(sizeof(Abi::TriangleVerticesHeader) + fanIndices.size() * 4u);
            Abi::TriangleVerticesHeader header {};
            header.viewport[0] = graphics.viewport.x;
            header.viewport[1] = graphics.viewport.y;
            header.viewport[2] = graphics.viewport.width;
            header.viewport[3] = graphics.viewport.height;
            header.depthRange[0] = graphics.viewport.minDepth;
            header.depthRange[1] = graphics.viewport.maxDepth;
            header.triangles = static_cast<std::uint32_t>(fanIndices.size() / primitiveVertices);
            header.flags = (conservativeMode != 0u ? Abi::TriangleFlagConservative : 0u) | (perVertex ? Abi::TriangleFlagPerVertex : 0u);
            std::memcpy(table.data(), &header, sizeof(header));
            auto* ids = reinterpret_cast<std::uint32_t*>(table.data() + sizeof(header));
            for (const auto& call : fanCalls) {
                for (std::uint32_t index = call.first; index < call.first + call.count; ++index) ids[index] = fanIndices[index] + static_cast<std::uint32_t>(call.vertexOffset);
            }
            const auto upload = stream.Write(table, 16u);
            [cache.vertexArguments setAddress:upload.address atIndex:Abi::Index(Abi::BufferIndex::TriangleVertices)];
        } else if (expanded) {
            indexType = MTLIndexTypeUInt32;
            if (gatheredAddress != 0u) {
                indexAddress = gatheredAddress;
                indexLength = fanIndices.size() * 4u;
            } else if (!fanIndices.empty()) {
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
                const auto upload = stream.Write(std::span<const std::byte>(reinterpret_cast<const std::byte*>(GuestArena::HostRange(indexBegin, indexBytes)), indexBytes), 16u);
                indexAddress = upload.address;
            }
            indexLength = indexBytes;
        }

        if (!empty) {
            for (const auto& pass : slicePipelines[slice]) {
                [encoder setRenderPipelineState:pass.pipeline];
                [encoder setCullMode:pass.cull];
                [encoder setDepthBias:pass.biasConstant slopeScale:pass.biasSlope clamp:graphics.depthBias ? graphics.depthBiasClamp : 0.0f];
                if (tess) {
                    // One object threadgroup per patch and instance (MetalTessellation.h).
                    const auto controlPoints = graphics.stages.tessellation->inputControlPoints;
                    for (const auto& call : drawCalls) {
                        const auto patches = call.count / controlPoints;
                        if (patches == 0u || call.instances == 0u) continue;
                        Abi::TessellationDraw record {};
                        record.indexAddress = draw.indexed ? indexAddress : 0u;
                        record.indexSize = draw.indexed ? draw.indexSize : 0u;
                        record.firstIndex = call.first;
                        record.vertexOffset = call.vertexOffset;
                        record.firstInstance = call.firstInstance;
                        record.minLevel = graphics.stages.minTessellationLevel;
                        record.maxLevel = graphics.stages.maxTessellationLevel;
                        [cache.objectArguments setAddress:stream.Write(std::as_bytes(std::span(&record, 1)), 16u).address atIndex:Abi::Index(Abi::BufferIndex::TessellationDraw)];
                        [encoder drawMeshThreadgroups:MTLSizeMake(patches, call.instances, 1) threadsPerObjectThreadgroup:MTLSizeMake(linked->objectThreads, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(linked->meshThreads, 1, 1)];
                        stream.CountCommand();
                        ++recorded;
                    }
                    held->objects.push_back(pass.pipeline);
                    continue;
                }
                if (mesh) {
                    // Grid (mesh groups, instances, splits): the mesh function's WorkgroupId x and y;
                    // z splits each subgroup's triangles in the triangle-vertices mode (MslAbi v13).
                    [encoder drawMeshThreadgroups:MTLSizeMake(meshPlan.groups, meshPlan.instances, meshSplits) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(vertexReflection.threadsPerThreadgroup[0], 1, 1)];
                    stream.CountCommand();
                    ++recorded;
                    held->objects.push_back(pass.pipeline);
                    continue;
                }
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
                        TileTarget(*target, std::span<const std::byte>(host, linearBytes));
                        if (target->gpuNewer != 0u) --target->gpuNewer;
                    });
                }
                passStream.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            };
        }
        recorder.LeaveRenderPassOpen(slicePassKey, Recorder::NoTiming, false, access, std::move(afterPass));
        anyContinued |= continued;
        if (slices > 1u) {
            emulatedSlicePasses.fetch_add(1u, std::memory_order_relaxed);
            if (!continued) emulatedBegun.fetch_add(1u, std::memory_order_relaxed);
        }
    };
    // Begun at the slice whose pass the recorder holds open: consecutive draws then share passes
    // (slices are independent targets, so their order between slices is free).
    std::uint32_t firstSlice = 0;
    for (std::uint32_t slice = 0; slice < slices; ++slice) {
        if (cache.passKey == slicePassKeyOf(slice)) firstSlice = slice;
    }
    for (std::uint32_t index = 0; index < slices; ++index) recordSlice((firstSlice + index) % slices);
    if (slices > 1u) emulatedDraws.fetch_add(1u, std::memory_order_relaxed);

    // Resident targets: the results stay on the GPU until something reads the target's memory.
    for (const auto& attachment : colors) {
        if (attachment.resident == nullptr) continue;
        recorder.Keep(attachment.resident);
        attachment.resident->MarkDirty();
    }

    // Lifetimes and pending accesses.
    held->objects.push_back(cache.vertexArguments);
    held->objects.push_back(cache.fragmentArguments);
    if (tess) held->objects.push_back(cache.objectArguments);
    recorder.Keep(held);
    if (!imageBindings.Empty()) imageBindings.Commit(recorder);
    state.NoteBindings(vertexPlan);
    state.NoteBindings(fragmentPlan);
    state.NoteBindings(targetPlan);
    if (tess) {
        state.NoteBindings(localPlan);
        state.NoteBindings(hullPlan);
    }
    state.CheckFaultsOnComplete(true, imageBindings.TableRanges());
    // Upstream 88ee39183 (keepRecordedDraw): a resident linear target's results go back to guest
    // memory right after the draw with a recorded GPU store, so CPU readers of the import see them
    // without a wait.
    for (const auto& attachment : colors) {
        if (attachment.resident == nullptr || attachment.resident->Descriptor().tileMode != Graphics::TextureTileMode::kLinear) continue;
        Graphics::StorageTexture::FlushPending(attachment.resident->Descriptor().baseAddress, static_cast<std::size_t>(attachment.resident->GuestBytes()), nullptr, "linear target");
    }
    // As upstream (Draw.cpp, SyncNotResident): a draw over a colour target that is not a resident
    // image writes guest memory in place, which the CPU may read without a flush hook (a queue
    // reported idle does not wait for the GPU), so it is submitted and waited for here.
    // Multisampled targets live only in their host image (guest memory guarded): not in place.
    const bool inPlaceTargets = std::any_of(colors.begin(), colors.end(), [](const ColorAttachment& attachment) { return attachment.resident == nullptr && attachment.multisample == nil; });
    if (inPlaceTargets) state.Sync();
    if (Trace()) {
        std::fprintf(stderr, "[metal] draw state: render %ux%u viewport (%.1f %.1f %.1f %.1f) scissor (%d %d %u %u) colorOutputs 0x%x\n", graphics.renderExtent.width, graphics.renderExtent.height, graphics.viewport.x, graphics.viewport.y, graphics.viewport.width, graphics.viewport.height, graphics.scissor.offset.x, graphics.scissor.offset.y, graphics.scissor.extent.width, graphics.scissor.extent.height, colorOutputs);
        for (const auto& attachment : colors) {
            const auto& target = *attachment.target;
            std::fprintf(stderr, "[metal]   color %u: 0x%llx %ux%u format %d tile %d element %u bytes 0x%llx mask 0x%x %s\n", target.exportIndex, static_cast<unsigned long long>(target.address), target.extent.width, target.extent.height, static_cast<int>(target.format), static_cast<int>(target.tileMode), target.elementBytes, static_cast<unsigned long long>(target.bytes), target.exportIndex < graphics.blends.size() ? graphics.blends[target.exportIndex].colorWriteMask : 0u, attachment.resident != nullptr ? "resident" : "in place");
        }
    }
    if (Trace()) std::fprintf(stderr, "[metal] draw %llu commands recorded in %u slice pass(es), %zu color targets, depth %s, pass %s\n", static_cast<unsigned long long>(recorded), slices, colors.size(), depth != nullptr ? "yes" : "no", anyContinued ? "continued" : "begun");
}

namespace Graphics::Metal {

EmulatedSampleStatistics EmulatedSampleCounts() {
    return {emulatedDraws.load(std::memory_order_relaxed), emulatedSlicePasses.load(std::memory_order_relaxed), emulatedBegun.load(std::memory_order_relaxed)};
}

IndexExpansionStatistics IndexExpansionCounts() {
    return {indexGathers.load(std::memory_order_relaxed), indexSettles.load(std::memory_order_relaxed)};
}

}

std::optional<ShaderRecompiler::GeometryStageLimits> VulkanDevice::GeometryLimits() const {
    return std::nullopt;
}

bool VulkanDevice::DualSrcBlend() const {
    return true;
}

bool VulkanDevice::PrimitiveListRestart() const {
    // Indexed list draws with restart go through the CPU index expansion (MetalDraw.mm `listRestart`).
    return true;
}

bool VulkanDevice::SamplerFilterMinmax() const {
    // MTLSamplerDescriptor.reductionMode (min/max filtering) exists on macOS 26+ (D5's samplers).
    return true;
}

bool VulkanDevice::ConservativeRasterization() const {
    // Emulated: the triangle-vertices mode with MslConservative.hpp's expansion and exact coverage
    // test (the Metal API has no conservative rasterization).
    return true;
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
    Graphics::Metal::TessellationStages tessellation;
    for (const auto& shader : shaders) {
        if ((shader.stage == ShaderRecompiler::ShaderStage::Vertex || shader.stage == ShaderRecompiler::ShaderStage::Mesh) && vertex == nullptr) vertex = &shader;
        else if (shader.stage == ShaderRecompiler::ShaderStage::Fragment && fragment == nullptr) fragment = &shader;
        else if (shader.stage == ShaderRecompiler::ShaderStage::Local && tessellation.local == nullptr) tessellation.local = &shader;
        else if (shader.stage == ShaderRecompiler::ShaderStage::TessellationControl && tessellation.hull == nullptr) tessellation.hull = &shader;
        else if (shader.stage == ShaderRecompiler::ShaderStage::TessellationEvaluation && tessellation.domain == nullptr) tessellation.domain = &shader;
        else Fail("draws with a " + std::to_string(static_cast<int>(shader.stage)) + " stage");
    }
    // Tessellated draws (MetalTessellation.h): local + hull in the object function, the domain
    // program (the stage feeding the fragment function) in the mesh function.
    const bool tessellated = graphics.stages.path == Graphics::ShaderPath::Tessellation;
    if (tessellated) {
        if (vertex != nullptr || tessellation.local == nullptr || tessellation.hull == nullptr || tessellation.domain == nullptr || !graphics.stages.tessellation) Fail("a tessellated draw without exactly a local, hull and domain stage and a tessellation configuration");
        vertex = tessellation.domain;
    } else if (tessellation.local != nullptr || tessellation.hull != nullptr || tessellation.domain != nullptr || graphics.stages.tessellation) {
        Fail("tessellation stages outside the tessellation path");
    }
    if (vertex == nullptr) Fail("a draw without a vertex or mesh stage");
    // Mesh draws: the geometry path's NGG primitive shader as a mesh function (MetalMesh.h).
    const bool meshDraw = vertex->stage == ShaderRecompiler::ShaderStage::Mesh;
    if (!tessellated && (meshDraw ? (graphics.stages.path != Graphics::ShaderPath::Geometry || !graphics.stages.mesh) : (graphics.stages.path != Graphics::ShaderPath::Vertex || graphics.stages.mesh))) Fail("the geometry shader path without a mesh configuration, or a mesh configuration on the vertex path");
    // Upstream Draw.cpp prepareDrawInputs: fast-launch subgroups (VGT_SHADER_STAGES_EN bits 19-20) generate
    // their vertex ids from the draw's vertex range, so an index buffer has no place in them.
    if (meshDraw && ((graphics.stages.registerValue >> 19u) & 3u) != 0u && draw.indexed) Fail("indexed fast-launch draws are unsupported");
    for (const auto* shader : {vertex, fragment, tessellation.local, tessellation.hull}) {
        if (shader == nullptr) continue;
        if (shader->program->msl.empty()) throw std::runtime_error("Metal device: the shader has no MSL form (recompile it for the device's Target(), whose backend is Msl)");
        if (shader->program->msl.Reflection().abiVersion != ShaderRecompiler::MslAbi::Version) throw std::runtime_error("Metal device: the shader was emitted for MslAbi version " + std::to_string(shader->program->msl.Reflection().abiVersion) + ", the device binds version " + std::to_string(ShaderRecompiler::MslAbi::Version));
    }
    if (draw.flags != 0u && draw.flags != 0x20u) Fail("draw flags " + std::to_string(draw.flags));
    if (!draw.indirect && (draw.indexCount == 0u || draw.instanceCount == 0u)) return;
    @autoreleasepool {
        state->RecordDraw(graphics, draw, *vertex, fragment, snapshots, tessellated ? &tessellation : nullptr);
    }
}

std::optional<std::string> VulkanDevice::KnownDrawRejection(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> shaders) const {
    const bool meshPath = graphics.stages.path == Graphics::ShaderPath::Geometry && graphics.stages.mesh;
    const bool tessellationPath = graphics.stages.path == Graphics::ShaderPath::Tessellation && graphics.stages.tessellation;
    if (graphics.stages.path != Graphics::ShaderPath::Vertex && !meshPath && !tessellationPath) return std::string("Metal draw (work package D6): the geometry path without a mesh configuration and tessellation plus geometry are not implemented");
    for (const auto& shader : shaders) {
        const auto stage = shader.stage;
        const bool tessellationStage = stage == ShaderRecompiler::ShaderStage::Local || stage == ShaderRecompiler::ShaderStage::TessellationControl || stage == ShaderRecompiler::ShaderStage::TessellationEvaluation;
        if (stage != ShaderRecompiler::ShaderStage::Vertex && stage != ShaderRecompiler::ShaderStage::Fragment && stage != ShaderRecompiler::ShaderStage::Mesh && !(tessellationPath && tessellationStage)) return std::string("Metal draw (work package D6): only vertex, mesh, tessellation and pixel stages are implemented");
    }
    // Sample counts the device cannot render are emulated as slices (DECISIONS D28); only counts with
    // no split into renderable slices are refused.
    if (graphics.samples > 1u && state->queue != nil) {
        try {
            static_cast<void>(Graphics::Metal::HostSamplesOf(state->device, graphics.samples));
        } catch (const std::exception& error) {
            return std::string("Metal draw (work package D6): ") + error.what();
        }
    }
    return std::nullopt;
}

// CB metadata passes: a fast-clear eliminate (or a DCC decompress of a target without DCC) expands
// the CMASK into guest memory as a draw would; a CB resolve is a resolve pass (DECISIONS D27).
void VulkanDevice::ColorMetadataPass(const Graphics::ColorMetadataPass& pass) {
    if (state->recorder == nullptr) throw std::runtime_error("Metal device: a compile-only device has no color metadata passes");
    if (pass.mode == Graphics::ColorMetadataPass::Mode::Resolve) {
        // Upstream Draw.cpp RunColorMetadataPass: the destination is the image cache's resident image
        // of the surface, written by a resolve pass from the source's MultisampleTarget host image.
        if (!pass.source.has_value() || pass.targets.size() != 1u) Fail("a CB resolve without its source and destination");
        const auto& destination = pass.targets.front();
        const auto* imageContext = state->images->ImageContext();
        if (destination.tileMode == Graphics::ColorTileMode::Linear || imageContext == nullptr) Fail("a CB resolve into a linear target (upstream resolves into resident images only)");
        const auto resident = Graphics::CachedStorageSurface(*imageContext, Graphics::SurfaceForTarget(destination));
        if (resident == nullptr || !resident->Attachable()) Fail("a CB resolve into a storage format that cannot be a color attachment");
        Graphics::ResolveMultisampleTarget(*imageContext, *pass.source, *resident, resident->AttachmentView(destination.format, destination.mip, destination.depthSlice, 1u, destination.arraySlice));
        return;
    }
    // DCC targets (upstream Draw.cpp RunColorMetadataPass): fast-clear keys store the clear texel and comp-to-single
    // keys each block's first texel in guest memory, and the keys are left uncompressed, exactly as before a draw
    // (MaterializeDcc). The image cache's resident image of the target, if any, reloads from guest memory.
    const auto* imageContext = state->images->ImageContext();
    for (const auto& color : pass.targets) {
        std::shared_ptr<Graphics::StorageTexture> resident;
        if (color.tileMode != Graphics::ColorTileMode::Linear && imageContext != nullptr) {
            resident = Graphics::CachedStorageSurface(*imageContext, Graphics::SurfaceForTarget(color));
            if (resident != nullptr && resident->GuestBytes() != color.bytes) resident = nullptr;
        }
        if (color.dccAddress != 0u) {
            state->SettleRange(color.dccAddress, Graphics::DccKeyBytes(color.bytes));
            if (pass.mode == Graphics::ColorMetadataPass::Mode::EliminateFastClear && Graphics::CurrentDccKeys(color.dccAddress, color.bytes) == Graphics::DccKeys::ClearSingle)
                Fail("CB fast-clear eliminate over comp-to-single DCC keys (whether it expands them is not modeled)");
            state->MaterializeDcc(color, resident.get());
        }
        state->MaterializeCmask(color, resident.get());
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
