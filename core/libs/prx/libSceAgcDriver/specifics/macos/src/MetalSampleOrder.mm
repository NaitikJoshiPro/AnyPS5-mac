// The guest sample order's reorder passes on Metal: see overlay MetalSampleOrder.hpp.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Graphics/include/MetalSampleOrder.hpp"
#include "MetalCommandStream.h"
#include "MetalImages.h"
#include "MetalVulkanImages.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace AgcDriver::Graphics::Metal {
namespace {

[[noreturn]] void FailOrder(const std::string& what) {
    throw std::runtime_error("Metal sample order: " + what);
}

std::atomic<std::uint64_t> reorderCount{0};
std::atomic<bool> suppressed{false};

// Per-sample fragment functions (one invocation per sample: they read [[sample_id]]). Destination
// slice `slice` holds guest samples (slice % p) * h + sample_id of layer slice / p (DECISIONS D28);
// its source is the guest sample the pixel's map word names, wherever its slice is.
constexpr const char* ReorderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct ReorderArgs {
    uint4 map;
    uint guestSamples;
    uint hostSamples;
    uint slicesPerLayer;
    uint slice;
    texture2d_ms_array<float> f;
    texture2d_ms_array<uint> u;
    texture2d_ms_array<int> i;
    depth2d_ms_array<float> d;
    texture2d_ms_array<uint> s;
};
vertex float4 aps5_reorder_vertex(uint id [[vertex_id]]) {
    const float2 corner = float2(float((id << 1u) & 2u), float(id & 2u));
    return float4(corner * 2.0f - 1.0f, 0.0f, 1.0f);
}
// (slice, sample) of the source of this destination sample.
static uint2 aps5_reorder_source(constant ReorderArgs& a, uint2 at, uint sample) {
    const uint quad = (at.x & 1u) | ((at.y & 1u) << 1u);
    const uint layer = a.slice / a.slicesPerLayer;
    const uint guest = (a.slice % a.slicesPerLayer) * a.hostSamples + sample;
    const uint source = (a.map[quad] >> (guest * 4u)) & 7u;
    return uint2(layer * a.slicesPerLayer + source / a.hostSamples, source % a.hostSamples);
}
fragment float4 aps5_reorder_float(float4 position [[position]], uint sample [[sample_id]], constant ReorderArgs& a [[buffer(0)]]) {
    const uint2 at = uint2(position.xy);
    const uint2 from = aps5_reorder_source(a, at, sample);
    return a.f.read(at, from.x, from.y);
}
fragment uint4 aps5_reorder_uint(float4 position [[position]], uint sample [[sample_id]], constant ReorderArgs& a [[buffer(0)]]) {
    const uint2 at = uint2(position.xy);
    const uint2 from = aps5_reorder_source(a, at, sample);
    return a.u.read(at, from.x, from.y);
}
fragment int4 aps5_reorder_sint(float4 position [[position]], uint sample [[sample_id]], constant ReorderArgs& a [[buffer(0)]]) {
    const uint2 at = uint2(position.xy);
    const uint2 from = aps5_reorder_source(a, at, sample);
    return a.i.read(at, from.x, from.y);
}
struct Aps5ReorderDepth {
    float depth [[depth(any)]];
};
struct Aps5ReorderDepthStencil {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};
struct Aps5ReorderStencil {
    uint stencil [[stencil]];
};
fragment Aps5ReorderDepth aps5_reorder_depth(float4 position [[position]], uint sample [[sample_id]], constant ReorderArgs& a [[buffer(0)]]) {
    const uint2 at = uint2(position.xy);
    const uint2 from = aps5_reorder_source(a, at, sample);
    Aps5ReorderDepth out;
    out.depth = a.d.read(at, from.x, from.y);
    return out;
}
fragment Aps5ReorderDepthStencil aps5_reorder_depth_stencil(float4 position [[position]], uint sample [[sample_id]], constant ReorderArgs& a [[buffer(0)]]) {
    const uint2 at = uint2(position.xy);
    const uint2 from = aps5_reorder_source(a, at, sample);
    Aps5ReorderDepthStencil out;
    out.depth = a.d.read(at, from.x, from.y);
    out.stencil = a.s.read(at, from.x, from.y).r;
    return out;
}
fragment Aps5ReorderStencil aps5_reorder_stencil(float4 position [[position]], uint sample [[sample_id]], constant ReorderArgs& a [[buffer(0)]]) {
    const uint2 at = uint2(position.xy);
    const uint2 from = aps5_reorder_source(a, at, sample);
    Aps5ReorderStencil out;
    out.stencil = a.s.read(at, from.x, from.y).r;
    return out;
}
)MSL";

// The MSL ReorderArgs (map, counts, then the texture resource IDs; 16-byte struct alignment).
struct ReorderArgs {
    std::uint32_t map[4];
    std::uint32_t guestSamples;
    std::uint32_t hostSamples;
    std::uint32_t slicesPerLayer;
    std::uint32_t slice;
    std::uint64_t textures[5];
    std::uint64_t pad;
};
static_assert(sizeof(ReorderArgs) == 80u);

enum class Kind : std::uint32_t {
    Float,
    Uint,
    Sint,
    Depth,
    DepthStencil,
    Stencil
};

const char* FunctionOf(Kind kind) {
    switch (kind) {
    case Kind::Float: return "aps5_reorder_float";
    case Kind::Uint: return "aps5_reorder_uint";
    case Kind::Sint: return "aps5_reorder_sint";
    case Kind::Depth: return "aps5_reorder_depth";
    case Kind::DepthStencil: return "aps5_reorder_depth_stencil";
    case Kind::Stencil: return "aps5_reorder_stencil";
    }
    return "";
}

struct ReorderPipelines {
    std::mutex mutex;
    std::unordered_map<void*, id<MTLLibrary>> libraries;
    std::unordered_map<void*, id<MTL4ArgumentTable>> tables;
    std::unordered_map<std::string, id<MTLRenderPipelineState>> pipelines;
    std::unordered_map<std::string, id<MTLDepthStencilState>> depthStates;
};

ReorderPipelines& Pipelines() {
    static ReorderPipelines pipelines;
    return pipelines;
}

bool HasDepth(MTLPixelFormat format) {
    return format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float || format == MTLPixelFormatDepth32Float_Stencil8;
}

bool HasStencil(MTLPixelFormat format) {
    return format == MTLPixelFormatDepth32Float_Stencil8 || format == MTLPixelFormatStencil8;
}

// The numeric class of a color pixel format: integer formats move exactly through their own type.
Kind ColorKind(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Uint: case MTLPixelFormatRG8Uint: case MTLPixelFormatRGBA8Uint: case MTLPixelFormatR16Uint: case MTLPixelFormatRG16Uint: case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatR32Uint: case MTLPixelFormatRG32Uint: case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGB10A2Uint:
        return Kind::Uint;
    case MTLPixelFormatR8Sint: case MTLPixelFormatRG8Sint: case MTLPixelFormatRGBA8Sint: case MTLPixelFormatR16Sint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatR32Sint: case MTLPixelFormatRG32Sint: case MTLPixelFormatRGBA32Sint:
        return Kind::Sint;
    default:
        return Kind::Float;
    }
}

// The pixel format the samples of a color texture move through: an integer view of the texel size
// that Apple GPUs render multisampled (Metal Feature Set Tables: R8Uint, R16Uint, RGBA8Uint and
// RGBA16Uint have MSAA on every Apple family; R32Uint, RG32Uint and RGBA32Uint do not), so the
// bits move unchanged. Integer textures keep their own format; 16-byte float texels (RGBA32Float,
// MSAA on Apple7+) have no MSAA integer view of their size and move as float4 (a read returned as
// is, no arithmetic).
MTLPixelFormat MoveFormat(id<MTLTexture> texture, std::uint32_t elementBytes, Kind& kind) {
    kind = ColorKind(texture.pixelFormat);
    if (kind != Kind::Float) return texture.pixelFormat;
    kind = Kind::Uint;
    switch (elementBytes) {
    case 1: return MTLPixelFormatR8Uint;
    case 2: return MTLPixelFormatR16Uint;
    case 4: return MTLPixelFormatRGBA8Uint;
    case 8: return MTLPixelFormatRGBA16Uint;
    case 16:
        kind = Kind::Float;
        return texture.pixelFormat;
    default:
        FailOrder("a color target of " + std::to_string(elementBytes) + "-byte texels has no reorder format");
    }
}

std::string DeviceKey(id<MTLDevice> device) {
    return std::to_string(reinterpret_cast<std::uintptr_t>((__bridge void*)device));
}

id<MTLLibrary> LibraryFor(ReorderPipelines& objects, id<MTLDevice> device) {
    auto& library = objects.libraries[(__bridge void*)device];
    if (library == nil) {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_1;
        NSError* error = nil;
        library = [device newLibraryWithSource:@(ReorderSource) options:options error:&error];
        if (library == nil) FailOrder("the reorder shaders do not compile: " + Describe(error));
    }
    return library;
}

struct PassObjects {
    id<MTLRenderPipelineState> pipeline = nil;
    id<MTLDepthStencilState> depthState = nil;
    id<MTL4ArgumentTable> table = nil;
};

// The pipeline drawing `kind` into an attachment of `format` at `samples` samples.
PassObjects ObjectsFor(id<MTLDevice> device, Kind kind, MTLPixelFormat format, NSUInteger samples) {
    auto& objects = Pipelines();
    std::lock_guard lock(objects.mutex);
    PassObjects result;
    NSError* error = nil;
    auto& table = objects.tables[(__bridge void*)device];
    if (table == nil) {
        MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 1;
        table = [device newArgumentTableWithDescriptor:descriptor error:&error];
        if (table == nil) FailOrder("cannot create the reorder argument table: " + Describe(error));
    }
    result.table = table;
    const auto key = DeviceKey(device) + ":" + std::to_string(static_cast<std::uint32_t>(kind)) + ":" + std::to_string(static_cast<std::uint64_t>(format)) + ":" + std::to_string(samples);
    auto& pipeline = objects.pipelines[key];
    if (pipeline == nil) {
        id<MTLLibrary> library = LibraryFor(objects, device);
        MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = [library newFunctionWithName:@"aps5_reorder_vertex"];
        descriptor.fragmentFunction = [library newFunctionWithName:@(FunctionOf(kind))];
        if (kind == Kind::Float || kind == Kind::Uint || kind == Kind::Sint) {
            descriptor.colorAttachments[0].pixelFormat = format;
        } else {
            if (HasDepth(format)) descriptor.depthAttachmentPixelFormat = format;
            if (HasStencil(format)) descriptor.stencilAttachmentPixelFormat = format;
        }
        descriptor.rasterSampleCount = samples;
        pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if (pipeline == nil) FailOrder("cannot build the reorder pipeline " + std::string(FunctionOf(kind)) + " for pixel format " + std::to_string(static_cast<std::uint64_t>(format)) + " at " + std::to_string(samples) + " samples: " + Describe(error));
    }
    result.pipeline = pipeline;
    if (kind == Kind::Depth || kind == Kind::DepthStencil || kind == Kind::Stencil) {
        const auto stateKey = DeviceKey(device) + ":" + std::to_string(static_cast<std::uint32_t>(kind));
        auto& state = objects.depthStates[stateKey];
        if (state == nil) {
            // Every sample takes the exported depth and stencil: no test rejects it.
            MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
            descriptor.depthCompareFunction = MTLCompareFunctionAlways;
            descriptor.depthWriteEnabled = kind != Kind::Stencil;
            if (kind != Kind::Depth) {
                MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
                stencil.stencilCompareFunction = MTLCompareFunctionAlways;
                stencil.stencilFailureOperation = MTLStencilOperationReplace;
                stencil.depthFailureOperation = MTLStencilOperationReplace;
                stencil.depthStencilPassOperation = MTLStencilOperationReplace;
                stencil.readMask = 0xffu;
                stencil.writeMask = 0xffu;
                descriptor.frontFaceStencil = stencil;
                descriptor.backFaceStencil = stencil;
            }
            state = [device newDepthStencilStateWithDescriptor:descriptor];
            if (state == nil) FailOrder("cannot build the reorder depth-stencil state");
        }
        result.depthState = state;
    }
    return result;
}

// A retained Objective-C object the stream keeps until the GPU finished with it.
void KeepObject(CommandStream& stream, id object) {
    stream.Keep(std::shared_ptr<void>((__bridge_retained void*)object, [](void* held) { CFRelease(held); }));
}

// A scratch multisampled texture shaped like `like`, resident (image cache) while the stream keeps it.
id<MTLTexture> ScratchLike(CommandStream& stream, id<MTLTexture> like, MTLPixelFormat format, bool formatView) {
    id<MTLDevice> device = like.device;
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = MTLTextureType2DMultisampleArray;
    descriptor.pixelFormat = format;
    descriptor.width = like.width;
    descriptor.height = like.height;
    descriptor.sampleCount = like.sampleCount;
    descriptor.arrayLength = like.arrayLength;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | (formatView ? MTLTextureUsagePixelFormatView : 0u);
    id<MTLTexture> scratch = [device newTextureWithDescriptor:descriptor];
    if (scratch == nil) FailOrder("cannot create the scratch texture of a reorder");
    auto* cache = ImageCache::For(device);
    if (cache != nullptr) cache->Track(scratch);
    stream.Keep(std::shared_ptr<void>((__bridge_retained void*)scratch, [cache](void* held) {
        id<MTLTexture> texture = (__bridge_transfer id<MTLTexture>)held;
        if (cache != nullptr) cache->Untrack(texture);
    }));
    return scratch;
}

// A view of a 2D multisample array in another pixel format (kept by the stream).
id<MTLTexture> FormatView(CommandStream& stream, id<MTLTexture> texture, MTLPixelFormat format, const char* what) {
    if (texture.pixelFormat == format) return texture;
    if ((texture.usage & MTLTextureUsagePixelFormatView) == 0u) FailOrder(std::string("the ") + what + " was made without MTLTextureUsagePixelFormatView, so its samples have no " + std::to_string(static_cast<std::uint64_t>(format)) + " view to move through");
    id<MTLTexture> view = [texture newTextureViewWithPixelFormat:format textureType:MTLTextureType2DMultisampleArray levels:NSMakeRange(0, 1) slices:NSMakeRange(0, texture.arrayLength)];
    if (view == nil) FailOrder(std::string("cannot view the ") + what + " in pixel format " + std::to_string(static_cast<std::uint64_t>(format)));
    KeepObject(stream, view);
    return view;
}

struct Sources {
    id<MTLTexture> color = nil;
    id<MTLTexture> depth = nil;
    id<MTLTexture> stencil = nil;
};

// One render pass per slice of `destination`: every covered sample of the region takes its source
// sample from `sources` through `map`.
void EncodePasses(CommandStream& stream, Kind kind, id<MTLTexture> destination, const Sources& sources, const SampleOrderMap& map, std::uint32_t guestSamples, MTLSize region, bool load) {
    const auto host = static_cast<std::uint32_t>(destination.sampleCount);
    const auto objects = ObjectsFor(destination.device, kind, destination.pixelFormat, destination.sampleCount);
    ReorderArgs args{};
    for (std::size_t quad = 0; quad < 4u; ++quad) args.map[quad] = map[quad];
    args.guestSamples = guestSamples;
    args.hostSamples = host;
    args.slicesPerLayer = guestSamples / host;
    const auto slot = [&](std::size_t index, id<MTLTexture> texture) {
        if (texture != nil) args.textures[index] = texture.gpuResourceID._impl;
    };
    if (kind == Kind::Float) slot(0, sources.color);
    else if (kind == Kind::Uint) slot(1, sources.color);
    else if (kind == Kind::Sint) slot(2, sources.color);
    slot(3, sources.depth);
    slot(4, sources.stencil);
    const bool color = kind == Kind::Float || kind == Kind::Uint || kind == Kind::Sint;
    for (std::uint32_t slice = 0; slice < destination.arrayLength; ++slice) {
        args.slice = slice;
        const auto upload = stream.Write(std::as_bytes(std::span(&args, 1)), 16u);
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.renderTargetWidth = destination.width;
        pass.renderTargetHeight = destination.height;
        const auto loadAction = load ? MTLLoadActionLoad : MTLLoadActionDontCare;
        if (color) {
            pass.colorAttachments[0].texture = destination;
            pass.colorAttachments[0].slice = slice;
            pass.colorAttachments[0].loadAction = loadAction;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        } else {
            if (HasDepth(destination.pixelFormat)) {
                pass.depthAttachment.texture = destination;
                pass.depthAttachment.slice = slice;
                pass.depthAttachment.loadAction = loadAction;
                pass.depthAttachment.storeAction = MTLStoreActionStore;
            }
            if (HasStencil(destination.pixelFormat)) {
                pass.stencilAttachment.texture = destination;
                pass.stencilAttachment.slice = slice;
                pass.stencilAttachment.loadAction = loadAction;
                pass.stencilAttachment.storeAction = MTLStoreActionStore;
            }
        }
        id<MTL4RenderCommandEncoder> encoder = stream.BeginRender(pass);
        if (encoder == nil) FailOrder("cannot begin a reorder pass");
        [objects.table setAddress:upload.address atIndex:0];
        [encoder setRenderPipelineState:objects.pipeline];
        if (objects.depthState != nil) [encoder setDepthStencilState:objects.depthState];
        [encoder setArgumentTable:objects.table atStages:MTLRenderStageFragment];
        [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<double>(destination.width), static_cast<double>(destination.height), 0.0, 1.0}];
        [encoder setScissorRect:MTLScissorRect{0, 0, region.width, region.height}];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        stream.CountCommand();
        stream.EndEncoder();
    }
}

}

VkFormat SampleOrderStorageFormat(const Context& context, VkFormat format, std::uint32_t elementBytes, std::uint32_t samples) {
    if (!context.storageImageMultisample || (samples != 2u && samples != 4u && samples != 8u)) return VK_FORMAT_UNDEFINED;
    const auto info = PixelFormatOf(format);
    if (info.format == MTLPixelFormatInvalid || info.blockWidth != 1u || info.blockBytes != elementBytes || HasDepth(info.format) || HasStencil(info.format)) return VK_FORMAT_UNDEFINED;
    switch (elementBytes) {
    case 1: return VK_FORMAT_R8_UINT;
    case 2: return VK_FORMAT_R16_UINT;
    case 4: return VK_FORMAT_R32_UINT;
    case 8: return VK_FORMAT_R32G32_UINT;
    case 16: return VK_FORMAT_R32G32B32A32_UINT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

bool SampleOrderDepthExact(VkFormat format) {
    switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

SampleOrderMap SampleOrderMapFor(const SampleOrder& order, bool toHost) {
    SampleOrderMap map{};
    for (std::size_t pixel = 0; pixel < map.size(); ++pixel) {
        const auto word = order[pixel] == 0u ? KeptSampleOrder : order[pixel];
        map[pixel] = toHost ? word : InvertSampleOrder(word);
    }
    return map;
}

void ReorderSamples(VkCommandBuffer commands, std::span<const SampleOrderTexture> textures, const SampleOrderMap& map, VkExtent2D region, std::uint32_t guestSamples) {
    auto& stream = Stream(commands);
    const SampleOrderMap identity{KeptSampleOrder, KeptSampleOrder, KeptSampleOrder, KeptSampleOrder};
    for (const auto& entry : textures) {
        id<MTLTexture> texture = TextureOfImage(entry.image);
        if (texture == nil) FailOrder("a reordered target has no texture");
        if (texture.textureType != MTLTextureType2DMultisampleArray) FailOrder("a reordered target is not a 2D multisample array (DECISIONS D28)");
        const auto host = static_cast<std::uint32_t>(texture.sampleCount);
        if (guestSamples < 2u || guestSamples > 8u || host < 2u || guestSamples % host != 0u || texture.arrayLength % (guestSamples / host) != 0u) FailOrder(std::to_string(guestSamples) + " guest samples do not fit a target of " + std::to_string(host) + " samples in " + std::to_string(texture.arrayLength) + " slices");
        const MTLSize size = MTLSizeMake(std::min<NSUInteger>(region.width, texture.width), std::min<NSUInteger>(region.height, texture.height), 1u);
        if (size.width == 0u || size.height == 0u) continue;
        const auto format = texture.pixelFormat;
        if (HasDepth(format) || HasStencil(format)) {
            const bool depth = HasDepth(format);
            const bool stencil = HasStencil(format);
            const auto kind = depth ? (stencil ? Kind::DepthStencil : Kind::Depth) : Kind::Stencil;
            // Stencil is read through an X32_Stencil8 view of a combined texture (the surface is
            // made with MTLTextureUsagePixelFormatView, DepthSurface.mm MakeSurfaceTexture).
            const auto stencilOf = [&](id<MTLTexture> of) -> id<MTLTexture> {
                if (!stencil) return nil;
                return format == MTLPixelFormatDepth32Float_Stencil8 ? FormatView(stream, of, MTLPixelFormatX32_Stencil8, "multisampled depth-stencil surface") : of;
            };
            id<MTLTexture> scratch = ScratchLike(stream, texture, format, format == MTLPixelFormatDepth32Float_Stencil8);
            EncodePasses(stream, kind, scratch, {nil, depth ? texture : nil, stencilOf(texture)}, identity, guestSamples, size, false);
            EncodePasses(stream, kind, texture, {nil, depth ? scratch : nil, stencilOf(scratch)}, map, guestSamples, size, true);
        } else {
            if (entry.elementBytes == 0u) FailOrder("a reordered color target without its texel size");
            Kind kind = Kind::Float;
            const auto move = MoveFormat(texture, entry.elementBytes, kind);
            id<MTLTexture> target = FormatView(stream, texture, move, "multisampled color target");
            id<MTLTexture> scratch = ScratchLike(stream, texture, move, false);
            EncodePasses(stream, kind, scratch, {target, nil, nil}, identity, guestSamples, size, false);
            EncodePasses(stream, kind, target, {scratch, nil, nil}, map, guestSamples, size, true);
        }
        reorderCount.fetch_add(1u, std::memory_order_relaxed);
    }
}

void RecordSampleOrder(const Context&, VkCommandBuffer commands, const SampleOrderPass& pass, std::span<const SampleOrderTarget> targets, bool toHost, std::vector<std::shared_ptr<void>>& kept) {
    if (targets.empty() || !ReordersSamples(pass.order)) return;
    const auto map = SampleOrderMapFor(pass.order, toHost);
    for (const auto& target : targets) {
        if (target.image == VK_NULL_HANDLE) FailOrder("a reordered attachment has no image");
        const VkExtent2D region{std::min(pass.extent.width, target.extent.width), std::min(pass.extent.height, target.extent.height)};
        if (region.width == 0u || region.height == 0u) continue;
        const std::uint32_t elementBytes = target.method == SampleOrderMethod::Storage ? PixelFormatOf(target.format).blockBytes : 0u;
        const SampleOrderTexture texture{target.image, elementBytes};
        ReorderSamples(commands, std::span(&texture, 1u), map, region, target.samples);
        if (target.keep != nullptr) kept.push_back(target.keep);
    }
}

std::uint64_t SampleOrderReorders() {
    return reorderCount.load(std::memory_order_relaxed);
}

void SetSampleOrderSuppressed(bool value) {
    suppressed.store(value, std::memory_order_release);
}

bool SampleOrderSuppressed() {
    return suppressed.load(std::memory_order_acquire);
}

}
