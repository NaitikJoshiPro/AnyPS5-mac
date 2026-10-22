// Metal twin of tests/execution/SampleOrder.cpp (agc_driver_sample_order, agc_driver_sample_order_grid_reference,
// agc_driver_sample_order_recipe_depth_locations; upstream 57ceb9e370, adc46f3ea0, f8e8dc6f67, d9bd0f9b4d, b6cde5698),
// swapped in by twins/SKIPFIX5-msaa.cmake. Upstream builds its reference images with its own Vulkan images, storage
// views and pipelines; here the same checks run on Metal textures, with per-sample fill passes and a readback kernel of
// the twin's own (bit-exact: integer views of the texel size, as MetalSampleOrder.mm moves samples), and the draws go
// through the Metal device (MetalDraw.mm) with the guest programs, state and expectations of upstream's file.
// Default mode: the reorder passes round-trip every color format and depth format bit for bit (SwapPasses; 8x too,
// emulated as two 4x slices on Apple9, DECISIONS D28), the pixel shader's SampleID / exported mask / quad mask follow
// the order (ShaderRemaps), and scenes keep each sample's own index (SceneChecks). --grid-reference: Village's 2x
// checkerboard through the exact path matches four uniform renders composed by pixel parity bit for bit, and pixel
// (0,0)'s locations alone (the device with its sample order suppressed, upstream's fallback context) do not.
// --recipe-depth-locations: the Metal device builds no draw recipes (TechnicalDebt metal-d6-bring-up-draws:
// DrawFromRecipe rebuilds), so the recorded draw is compared with the same draw recorded again as the first draw of a
// pass (the rebuild a recipe miss takes), and a reordered draw must build no recipe and keep its depth at the guest's
// sample locations.

#include "../src/MetalCommandStream.h"
#include "../src/MetalImages.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MetalSampleOrder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/SampleOrder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "../../../tests/execution/VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace Metal = AgcDriver::Graphics::Metal;
using AgcDriver::Graphics::Context;
using AgcDriver::Graphics::KeptSampleOrder;
using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::SampleOrder;
using AgcDriver::Graphics::SampleOrderMethod;
using AgcDriver::Graphics::SampleOrderPass;
using AgcDriver::Graphics::SampleOrderTarget;
using ShaderRecompiler::PixelInput;
using ShaderRecompiler::PixelInputBit;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t WaveSize = 64;

SampleOrder OrderFor(std::uint32_t samples) {
    if (samples == 2u) return {KeptSampleOrder, KeptSampleOrder, 0x76543201u, 0x76543201u};
    if (samples == 4u) return {KeptSampleOrder, 0x76542301u, 0x76540132u, 0x76541023u};
    return {0x01234567u, KeptSampleOrder, 0x10325476u, 0x45670123u};
}

std::uint32_t Pick(std::uint32_t word, std::uint32_t index) {
    return (word >> (index * 4u)) & 7u;
}

std::uint32_t Quad(std::uint32_t x, std::uint32_t y) {
    return (x & 1u) | ((y & 1u) << 1u);
}

std::string At(std::uint32_t x, std::uint32_t y, std::uint32_t sample) {
    return "sample " + std::to_string(sample) + " of pixel (" + std::to_string(x) + ", " + std::to_string(y) + ")";
}

using Samples = std::vector<std::array<std::uint32_t, 4>>;

template<typename Record>
void Submit(const Context& context, Record record) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    AgcDriver::Graphics::CommandBatch batch(context);
    record(batch.Handle());
    batch.SubmitAndWait();
}

// The twin's fill and readback programs. A sample's value is up to four words of its texel's bits (words beyond the
// texel size are zero, upstream's AliasMask); color moves through an integer view of the texel size that Apple GPUs
// render multisampled (R8Uint, R16Uint, RGBA8Uint, RGBA16Uint; 16-byte texels as their own format), depth as float
// bits in word 0 and stencil in word 1. Guest sample g of a 2D multisample array of h samples per slice is sample
// g % h of slice g / h (DECISIONS D28).
constexpr const char* TestSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct TestArgs {
    uint width;
    uint height;
    uint guestSamples;
    uint hostSamples;
    uint slicesPerLayer;
    uint slice;
    uint bytes;
    uint mode;
    device uint4* values;
    texture2d_ms_array<float> f;
    texture2d_ms_array<uint> u;
    depth2d_ms_array<float> d;
    texture2d_ms_array<uint> s;
};
static uint4 to_move(uint4 w, uint bytes) {
    switch (bytes) {
    case 1u: return uint4(w.x & 0xffu, 0u, 0u, 0u);
    case 2u: return uint4(w.x & 0xffffu, 0u, 0u, 0u);
    case 4u: return uint4(w.x & 0xffu, (w.x >> 8u) & 0xffu, (w.x >> 16u) & 0xffu, w.x >> 24u);
    case 8u: return uint4(w.x & 0xffffu, w.x >> 16u, w.y & 0xffffu, w.y >> 16u);
    default: return w;
    }
}
static uint4 from_move(uint4 c, uint bytes) {
    switch (bytes) {
    case 1u: return uint4(c.x & 0xffu, 0u, 0u, 0u);
    case 2u: return uint4(c.x & 0xffffu, 0u, 0u, 0u);
    case 4u: return uint4((c.x & 0xffu) | ((c.y & 0xffu) << 8u) | ((c.z & 0xffu) << 16u) | (c.w << 24u), 0u, 0u, 0u);
    case 8u: return uint4((c.x & 0xffffu) | (c.y << 16u), (c.z & 0xffffu) | (c.w << 16u), 0u, 0u);
    default: return c;
    }
}
vertex float4 test_vertex(uint id [[vertex_id]]) {
    const float2 corner = float2(float((id << 1u) & 2u), float(id & 2u));
    return float4(corner * 2.0f - 1.0f, 0.0f, 1.0f);
}
static uint test_index(constant TestArgs& a, float4 position, uint sample) {
    const uint2 at = uint2(position.xy);
    return (at.y * a.width + at.x) * a.guestSamples + (a.slice % a.slicesPerLayer) * a.hostSamples + sample;
}
fragment uint4 test_fill_uint(float4 position [[position]], uint sample [[sample_id]], constant TestArgs& a [[buffer(0)]]) {
    return to_move(a.values[test_index(a, position, sample)], a.bytes);
}
fragment float4 test_fill_float(float4 position [[position]], uint sample [[sample_id]], constant TestArgs& a [[buffer(0)]]) {
    return as_type<float4>(a.values[test_index(a, position, sample)]);
}
struct TestDepth {
    float depth [[depth(any)]];
};
struct TestDepthStencil {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};
fragment TestDepth test_fill_depth(float4 position [[position]], uint sample [[sample_id]], constant TestArgs& a [[buffer(0)]]) {
    TestDepth out;
    out.depth = as_type<float>(a.values[test_index(a, position, sample)].x);
    return out;
}
fragment TestDepthStencil test_fill_depth_stencil(float4 position [[position]], uint sample [[sample_id]], constant TestArgs& a [[buffer(0)]]) {
    const uint4 value = a.values[test_index(a, position, sample)];
    TestDepthStencil out;
    out.depth = as_type<float>(value.x);
    out.stencil = value.y & 0xffu;
    return out;
}
// mode 0: float color (bits), 1: integer color through to/from_move, 2: depth, 3: depth and stencil.
kernel void test_read(constant TestArgs& a [[buffer(0)]], uint3 id [[thread_position_in_grid]]) {
    if (id.x >= a.width || id.y >= a.height || id.z >= a.guestSamples) return;
    const uint slice = id.z / a.hostSamples;
    const uint sample = id.z % a.hostSamples;
    uint4 value = uint4(0u);
    if (a.mode == 0u) value = as_type<uint4>(a.f.read(id.xy, slice, sample));
    else if (a.mode == 1u) value = from_move(a.u.read(id.xy, slice, sample), a.bytes);
    else {
        value.x = as_type<uint>(a.d.read(id.xy, slice, sample));
        if (a.mode == 3u) value.y = a.s.read(id.xy, slice, sample).r;
    }
    a.values[(id.y * a.width + id.x) * a.guestSamples + id.z] = value;
}
)MSL";

struct TestArgs {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t guestSamples;
    std::uint32_t hostSamples;
    std::uint32_t slicesPerLayer;
    std::uint32_t slice;
    std::uint32_t bytes;
    std::uint32_t mode;
    std::uint64_t values;
    std::uint64_t textures[4];
};
static_assert(sizeof(TestArgs) == 72u);

struct Programs {
    id<MTLLibrary> library = nil;
    id<MTL4ArgumentTable> table = nil;
    id<MTLComputePipelineState> read = nil;
    id<MTLDepthStencilState> writeAll = nil;
};

Programs& ProgramsFor(id<MTLDevice> device) {
    static Programs programs;
    if (programs.library != nil) return programs;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_1;
    NSError* error = nil;
    programs.library = [device newLibraryWithSource:@(TestSource) options:options error:&error];
    Require(programs.library != nil, "the sample order test programs do not compile: " + Metal::Describe(error));
    MTL4ArgumentTableDescriptor* table = [MTL4ArgumentTableDescriptor new];
    table.maxBufferBindCount = 1;
    programs.table = [device newArgumentTableWithDescriptor:table error:&error];
    Require(programs.table != nil, "cannot create the sample order test's argument table");
    programs.read = [device newComputePipelineStateWithFunction:[programs.library newFunctionWithName:@"test_read"] error:&error];
    Require(programs.read != nil, "cannot build the sample order test's readback kernel: " + Metal::Describe(error));
    MTLDepthStencilDescriptor* depth = [MTLDepthStencilDescriptor new];
    depth.depthCompareFunction = MTLCompareFunctionAlways;
    depth.depthWriteEnabled = YES;
    MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
    stencil.stencilCompareFunction = MTLCompareFunctionAlways;
    stencil.depthStencilPassOperation = MTLStencilOperationReplace;
    stencil.writeMask = 0xffu;
    stencil.readMask = 0xffu;
    depth.frontFaceStencil = stencil;
    depth.backFaceStencil = stencil;
    programs.writeAll = [device newDepthStencilStateWithDescriptor:depth];
    return programs;
}

bool HasDepth(MTLPixelFormat format) {
    return format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float || format == MTLPixelFormatDepth32Float_Stencil8;
}

bool HasStencil(MTLPixelFormat format) {
    return format == MTLPixelFormatDepth32Float_Stencil8;
}

bool IntegerFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Uint: case MTLPixelFormatR16Uint: case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatR8Sint: case MTLPixelFormatR16Sint: case MTLPixelFormatRGBA8Sint: case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA32Sint:
        return true;
    default:
        return false;
    }
}

// The view the twin fills and reads a color texture through (float only for 16-byte float texels).
MTLPixelFormat MoveFormat(MTLPixelFormat format, std::uint32_t bytes) {
    if (IntegerFormat(format)) return format;
    switch (bytes) {
    case 1: return MTLPixelFormatR8Uint;
    case 2: return MTLPixelFormatR16Uint;
    case 4: return MTLPixelFormatRGBA8Uint;
    case 8: return MTLPixelFormatRGBA16Uint;
    default: return format;
    }
}

id<MTLTexture> ViewOf(id<MTLTexture> texture, MTLPixelFormat format) {
    if (texture.pixelFormat == format) return texture;
    id<MTLTexture> view = [texture newTextureViewWithPixelFormat:format textureType:MTLTextureType2DMultisampleArray levels:NSMakeRange(0, 1) slices:NSMakeRange(0, texture.arrayLength)];
    Require(view != nil, "cannot view a test texture in pixel format " + std::to_string(static_cast<std::uint64_t>(format)));
    return view;
}

// A multisampled test texture of `samples` guest samples (HostSamplesOf's slices), resident while it lives.
class TestTexture {
public:
    TestTexture(id<MTLDevice> device, MTLPixelFormat format, std::uint32_t samples, VkExtent2D extent) : samples(samples) {
        const auto host = Metal::HostSamplesOf(device, samples);
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
        descriptor.textureType = MTLTextureType2DMultisampleArray;
        descriptor.pixelFormat = format;
        descriptor.width = extent.width;
        descriptor.height = extent.height;
        descriptor.sampleCount = host.samples;
        descriptor.arrayLength = host.slices;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
        texture = [device newTextureWithDescriptor:descriptor];
        Require(texture != nil, "cannot create a sample order test texture");
        cache = Metal::ImageCache::For(device);
        if (cache != nullptr) cache->Track(texture);
    }
    ~TestTexture() {
        if (cache != nullptr) cache->Untrack(texture);
    }
    TestTexture(const TestTexture&) = delete;
    TestTexture& operator=(const TestTexture&) = delete;
    VkImage Image() const { return reinterpret_cast<VkImage>((__bridge void*)texture); }
    VkImageView View() const { return reinterpret_cast<VkImageView>((__bridge void*)texture); }
    id<MTLTexture> texture = nil;
    std::uint32_t samples;

private:
    Metal::ImageCache* cache = nullptr;
};

TestArgs ArgsFor(id<MTLTexture> texture, VkExtent2D extent, std::uint32_t guestSamples, const AgcDriver::Graphics::Buffer& values) {
    TestArgs args{};
    args.width = extent.width;
    args.height = extent.height;
    args.guestSamples = guestSamples;
    args.hostSamples = static_cast<std::uint32_t>(texture.sampleCount);
    args.slicesPerLayer = guestSamples / args.hostSamples;
    args.values = values.DeviceAddress();
    return args;
}

// Fills every sample of `texture` (a color texture through `move`, `bytes` per texel; or a depth texture) from `values`.
void Fill(VkCommandBuffer commands, id<MTLTexture> texture, std::uint32_t bytes, std::uint32_t guestSamples, VkExtent2D extent, const AgcDriver::Graphics::Buffer& values) {
    id<MTLDevice> device = texture.device;
    auto& programs = ProgramsFor(device);
    const bool depth = HasDepth(texture.pixelFormat);
    id<MTLTexture> target = depth ? texture : ViewOf(texture, MoveFormat(texture.pixelFormat, bytes));
    const bool floatMove = !depth && !IntegerFormat(target.pixelFormat);
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [programs.library newFunctionWithName:@"test_vertex"];
    descriptor.fragmentFunction = [programs.library newFunctionWithName:depth ? (HasStencil(texture.pixelFormat) ? @"test_fill_depth_stencil" : @"test_fill_depth") : (floatMove ? @"test_fill_float" : @"test_fill_uint")];
    if (depth) {
        descriptor.depthAttachmentPixelFormat = texture.pixelFormat;
        if (HasStencil(texture.pixelFormat)) descriptor.stencilAttachmentPixelFormat = texture.pixelFormat;
    } else {
        descriptor.colorAttachments[0].pixelFormat = target.pixelFormat;
    }
    descriptor.rasterSampleCount = texture.sampleCount;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    Require(pipeline != nil, "cannot build the sample order test's fill pipeline: " + Metal::Describe(error));
    auto& stream = Metal::Stream(commands);
    auto args = ArgsFor(texture, extent, guestSamples, values);
    args.bytes = bytes;
    for (std::uint32_t slice = 0; slice < texture.arrayLength; ++slice) {
        args.slice = slice;
        const auto upload = stream.Write(std::as_bytes(std::span(&args, 1)), 16u);
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.renderTargetWidth = texture.width;
        pass.renderTargetHeight = texture.height;
        if (depth) {
            pass.depthAttachment.texture = texture;
            pass.depthAttachment.slice = slice;
            pass.depthAttachment.loadAction = MTLLoadActionDontCare;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            if (HasStencil(texture.pixelFormat)) {
                pass.stencilAttachment.texture = texture;
                pass.stencilAttachment.slice = slice;
                pass.stencilAttachment.loadAction = MTLLoadActionDontCare;
                pass.stencilAttachment.storeAction = MTLStoreActionStore;
            }
        } else {
            pass.colorAttachments[0].texture = target;
            pass.colorAttachments[0].slice = slice;
            pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        }
        id<MTL4RenderCommandEncoder> encoder = stream.BeginRender(pass);
        [programs.table setAddress:upload.address atIndex:0];
        [encoder setRenderPipelineState:pipeline];
        if (depth) [encoder setDepthStencilState:programs.writeAll];
        [encoder setArgumentTable:programs.table atStages:MTLRenderStageFragment];
        [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<double>(texture.width), static_cast<double>(texture.height), 0.0, 1.0}];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        stream.EndEncoder();
    }
    if (target != texture) stream.Keep(std::shared_ptr<void>((__bridge_retained void*)target, [](void* held) { CFRelease(held); }));
}

// Every sample of `texture` (guest sample order of the array's layout) as upstream's ReadValues layout.
Samples Read(const Context& context, id<MTLTexture> texture, std::uint32_t bytes, std::uint32_t guestSamples, VkExtent2D extent) {
    const auto count = static_cast<std::size_t>(extent.width) * extent.height * guestSamples;
    AgcDriver::Graphics::Buffer values(context, count * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memset(values.Bytes().data(), 0xcd, count * 16u);
    auto& programs = ProgramsFor(texture.device);
    auto args = ArgsFor(texture, extent, guestSamples, values);
    args.bytes = bytes;
    id<MTLTexture> view = nil;
    id<MTLTexture> stencil = nil;
    if (HasDepth(texture.pixelFormat)) {
        args.mode = HasStencil(texture.pixelFormat) ? 3u : 2u;
        args.textures[2] = texture.gpuResourceID._impl;
        if (args.mode == 3u) {
            stencil = ViewOf(texture, MTLPixelFormatX32_Stencil8);
            args.textures[3] = stencil.gpuResourceID._impl;
        }
    } else {
        view = ViewOf(texture, MoveFormat(texture.pixelFormat, bytes));
        args.mode = IntegerFormat(view.pixelFormat) ? 1u : 0u;
        args.textures[args.mode == 1u ? 1 : 0] = view.gpuResourceID._impl;
    }
    Submit(context, [&](VkCommandBuffer commands) {
        auto& stream = Metal::Stream(commands);
        stream.DispatchHelper(programs.read, std::as_bytes(std::span(&args, 1)), MTLSizeMake((extent.width + 7u) / 8u, (extent.height + 7u) / 8u, guestSamples), MTLSizeMake(8, 8, 1));
    });
    values.Invalidate();
    Samples read(count);
    std::memcpy(read.data(), values.Bytes().data(), count * sizeof(read[0]));
    return read;
}

void CheckOrdered(const Samples& before, const Samples& after, VkExtent2D extent, VkExtent2D region, std::uint32_t samples, const SampleOrder& order, bool toHost, const std::string& what) {
    std::uint32_t moved = 0;
    std::string first;
    std::array<std::uint32_t, 4> wrong{};
    for (std::uint32_t y = 0; y < extent.height; ++y) {
        for (std::uint32_t x = 0; x < extent.width; ++x) {
            const bool inside = x < region.width && y < region.height;
            const auto word = order[Quad(x, y)];
            const auto map = toHost ? word : AgcDriver::Graphics::InvertSampleOrder(word);
            for (std::uint32_t sample = 0; sample < samples; ++sample) {
                const auto base = (y * extent.width + x) * samples;
                const auto source = inside ? Pick(map, sample) : sample;
                if (after[base + sample] != before[base + source]) {
                    if (first.empty()) {
                        char held[96];
                        std::snprintf(held, sizeof(held), " holds 0x%08x 0x%08x, expected 0x%08x 0x%08x (sample %u)", after[base + sample][0], after[base + sample][1], before[base + source][0], before[base + source][1], source);
                        first = At(x, y, sample) + held;
                    }
                    ++wrong[Quad(x, y)];
                }
                if (source != sample) ++moved;
            }
        }
    }
    if (!first.empty()) {
        std::string classes;
        for (std::uint32_t quad = 0; quad < wrong.size(); ++quad) {
            const bool kept = order[quad] == 0u || order[quad] == KeptSampleOrder;
            classes += (quad == 0u ? "" : ", ") + std::to_string(wrong[quad]) + " in pixels with x " + ((quad & 1u) != 0u ? "odd" : "even") + " and y " + ((quad & 2u) != 0u ? "odd" : "even") + (kept ? " (kept)" : " (reordered)");
        }
        throw std::runtime_error(what + ": " + first + "; wrong samples: " + classes);
    }
    Require(moved != 0, what + ": the order moved no sample");
}

const std::array<std::pair<VkFormat, std::uint32_t>, 5> ColorFormats{{
    {VK_FORMAT_R8_UNORM, 1u}, {VK_FORMAT_R16_SFLOAT, 2u}, {VK_FORMAT_R8G8B8A8_SRGB, 4u}, {VK_FORMAT_R16G16B16A16_SFLOAT, 8u}, {VK_FORMAT_R32G32B32A32_SFLOAT, 16u}
}};

std::uint32_t AliasMask(std::uint32_t elementBytes, std::uint32_t component) {
    if (elementBytes == 1u) return component == 0u ? 0xffu : 0u;
    if (elementBytes == 2u) return component == 0u ? 0xffffu : 0u;
    if (elementBytes == 4u) return component == 0u ? ~0u : 0u;
    if (elementBytes == 8u) return component < 2u ? ~0u : 0u;
    return ~0u;
}

constexpr VkExtent2D SwapExtent{20u, 12u};
constexpr VkExtent2D SwapRegion{16u, 10u};

bool ColorSwaps(const Context& context, VkFormat format, std::uint32_t elementBytes, std::uint32_t samples) {
    const auto alias = AgcDriver::Graphics::SampleOrderStorageFormat(context, format, elementBytes, samples);
    const auto name = "the " + std::to_string(samples) + "x color format " + std::to_string(format);
    if (alias == VK_FORMAT_UNDEFINED) {
        std::printf("%s has no %u-sample storage alias here, not swapped\n", name.c_str(), samples);
        return false;
    }
    const auto order = OrderFor(samples);
    TestTexture image(Metal::DeviceOf(context), Metal::PixelFormatOf(format).format, samples, SwapExtent);
    const auto count = static_cast<std::size_t>(SwapExtent.width) * SwapExtent.height * samples;
    AgcDriver::Graphics::Buffer values(context, count * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Samples before(count);
    for (std::size_t index = 0; index < count; ++index) {
        for (std::uint32_t component = 0; component < 4u; ++component) before[index][component] = static_cast<std::uint32_t>(index * 0x9e3779b1u + component * 0x85ebca6bu + 0x1000u * (index % samples)) & AliasMask(elementBytes, component);
        before[index][0] = (before[index][0] & ~7u & AliasMask(elementBytes, 0u)) | static_cast<std::uint32_t>(index % samples);
    }
    std::memcpy(values.Bytes().data(), before.data(), count * sizeof(before[0]));
    Submit(context, [&](VkCommandBuffer commands) { Fill(commands, image.texture, elementBytes, samples, SwapExtent, values); });
    Require(Read(context, image.texture, elementBytes, samples, SwapExtent) == before, name + ": the stored samples did not read back");
    const SampleOrderTarget target{image.Image(), image.View(), alias, SwapExtent, samples, SampleOrderMethod::Storage, nullptr};
    const SampleOrderPass pass{order, SwapRegion, {}, false};
    std::vector<std::shared_ptr<void>> kept;
    const auto reordersBefore = Metal::SampleOrderReorders();
    Submit(context, [&](VkCommandBuffer commands) { AgcDriver::Graphics::RecordSampleOrder(context, commands, pass, std::span(&target, 1u), true, kept); });
    Require(Metal::SampleOrderReorders() == reordersBefore + 1u, name + ": RecordSampleOrder did not take the Metal reorder passes");
    CheckOrdered(before, Read(context, image.texture, elementBytes, samples, SwapExtent), SwapExtent, SwapRegion, samples, order, true, name + " put in the host's order");
    Submit(context, [&](VkCommandBuffer commands) { AgcDriver::Graphics::RecordSampleOrder(context, commands, pass, std::span(&target, 1u), false, kept); });
    Require(Read(context, image.texture, elementBytes, samples, SwapExtent) == before, name + ": putting the samples back in the guest's order did not restore every sample bit for bit");
    std::printf("%s round-trips through its %u-byte texels\n", name.c_str(), elementBytes);
    return true;
}

bool DepthSwaps(const Context& context, VkFormat format, std::uint32_t samples) {
    // The Metal surface of the guest format (DepthSurface.mm SurfaceFormat: 16-bit depth with stencil is widened).
    const MTLPixelFormat surface = format == VK_FORMAT_D32_SFLOAT ? MTLPixelFormatDepth32Float : format == VK_FORMAT_D16_UNORM ? MTLPixelFormatDepth16Unorm : MTLPixelFormatDepth32Float_Stencil8;
    const bool stencil = HasStencil(surface);
    const auto method = stencil ? SampleOrderMethod::DepthStencilExport : SampleOrderMethod::Depth;
    const auto name = "the " + std::to_string(samples) + "x depth format " + std::to_string(format) + (stencil ? " (depth and [[stencil]] export)" : " (depth)");
    AgcDriver::Graphics::State probe{};
    probe.samples = samples;
    probe.depth = AgcDriver::Graphics::DepthTarget{0x100000u, stencil ? 0x200000u : 0u, SwapExtent, format, 1.0f, 0};
    probe.depth->samples = samples;
    const auto refusal = AgcDriver::Graphics::SampleOrderRefusal(context, probe);
    Require(refusal.empty(), name + ": the Metal device refuses the order: " + refusal);
    const auto order = OrderFor(samples);
    TestTexture image(Metal::DeviceOf(context), surface, samples, SwapExtent);
    const auto count = static_cast<std::size_t>(SwapExtent.width) * SwapExtent.height * samples;
    AgcDriver::Graphics::Buffer values(context, count * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Samples filled(count);
    const bool unorm16 = format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D16_UNORM_S8_UINT;
    for (std::size_t index = 0; index < count; ++index) {
        const auto key = static_cast<std::uint32_t>(index + 1u);
        filled[index] = {std::bit_cast<std::uint32_t>(unorm16 ? static_cast<float>(key) / 65535.0f : static_cast<float>(key) / 4096.0f), stencil ? (key * 37u) & 0xffu : 0u, 0u, 0u};
    }
    std::memcpy(values.Bytes().data(), filled.data(), count * sizeof(filled[0]));
    Submit(context, [&](VkCommandBuffer commands) { Fill(commands, image.texture, 0u, samples, SwapExtent, values); });
    const auto before = Read(context, image.texture, 0u, samples, SwapExtent);
    for (std::size_t index = 0; index < count; ++index) {
        const auto key = static_cast<std::uint32_t>(index + 1u);
        const auto depth = std::bit_cast<float>(before[index][0]);
        const bool depthHeld = unorm16 ? static_cast<std::uint32_t>(depth * 65535.0f + 0.5f) == key : before[index][0] == filled[index][0];
        Require(depthHeld && (!stencil || before[index][1] == filled[index][1]), name + ": the filled sample " + std::to_string(index) + " did not read back");
    }
    const SampleOrderTarget target{image.Image(), image.View(), format, SwapExtent, samples, method, nullptr};
    const SampleOrderPass pass{order, SwapRegion, {}, false};
    std::vector<std::shared_ptr<void>> kept;
    Submit(context, [&](VkCommandBuffer commands) { AgcDriver::Graphics::RecordSampleOrder(context, commands, pass, std::span(&target, 1u), true, kept); });
    CheckOrdered(before, Read(context, image.texture, 0u, samples, SwapExtent), SwapExtent, SwapRegion, samples, order, true, name + " put in the host's order");
    Submit(context, [&](VkCommandBuffer commands) { AgcDriver::Graphics::RecordSampleOrder(context, commands, pass, std::span(&target, 1u), false, kept); });
    Require(Read(context, image.texture, 0u, samples, SwapExtent) == before, name + ": putting the samples back in the guest's order did not restore every sample bit for bit");
    std::printf("%s round-trips\n", name.c_str());
    return true;
}

// 2x and 4x natively; 8x as two 4x slices on Apple9 (HostSamplesOf, DECISIONS D28), which the reorder crosses.
std::uint32_t SwapPasses(const Context& context) {
    std::uint32_t swapped = 0;
    for (const std::uint32_t samples : {2u, 4u, 8u}) {
        for (const auto& [format, bytes] : ColorFormats) swapped += ColorSwaps(context, format, bytes, samples) ? 1u : 0u;
        for (const auto format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT}) swapped += DepthSwaps(context, format, samples) ? 1u : 0u;
    }
    return swapped;
}

using Vertices = std::vector<std::array<float, 4>>;

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 3> NullExportCode{0xf8001890, 0x00000000, 0xbf810000};

constexpr std::uint32_t RemapWidth = 64;
constexpr std::uint32_t RemapHeight = 8;
alignas(256) std::array<std::uint32_t, RemapWidth * RemapHeight * 8> Invocations{};
alignas(256) std::array<std::array<std::uint32_t, 17>, 4> StoreCodes{};
alignas(256) std::array<std::array<std::uint32_t, 5>, 8> MaskCodes{};
alignas(256) constexpr std::array<std::uint32_t, 7> SampleColorCode{0xd5480007, 0x02111106, 0x7e0e0d07, 0xf800180f, 0x07070100, 0xbf810000, 0xbf810000};
alignas(256) std::array<std::byte, 10 * 256 * 1024> DepthMemory{};

std::span<const std::uint32_t> StoreCode(std::uint32_t samples) {
    const auto shift = static_cast<std::uint32_t>(std::countr_zero(samples));
    auto& code = StoreCodes[shift];
    code = {0x7e100f04, 0x7e120f05, 0x34121286, 0x4a101308, 0x34101080u | shift, 0xd548000a, 0x02111106, 0x4a101508, 0x34101082, 0x7e160281, 0xe0705000, 0x80000b08, 0xbf8c3f70, 0xf8001890, 0x00000000, 0xbf810000, 0xbf810000};
    return code;
}

std::span<const std::uint32_t> MaskCode(std::size_t slot, std::uint32_t mask) {
    auto& code = MaskCodes[slot];
    code = {0x7e1002ffu, mask, 0xf8001884u, 0x00080000u, 0xbf810000u};
    return code;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, stride == 0u ? 0x31016facu : 0x01016facu};
}

struct Pass {
    const Vertices* vertices = nullptr;
    const void* vertexMemory = nullptr;
    std::span<const std::uint32_t> code;
    bool perSample = false;
    std::uint32_t shaderControl = 0u;
    std::uint32_t colorFormat = 0u;
    std::uint32_t samples = 4u;
    VkExtent2D extent{RemapWidth, RemapHeight};
    std::optional<AgcDriver::Graphics::DepthTarget> depth;
    bool depthWrite = false;
    VkCompareOp depthCompare = VK_COMPARE_OP_ALWAYS;
    std::optional<AgcDriver::Graphics::ColorTarget> color;
    std::optional<std::array<std::uint8_t, 32>> locations;
    std::optional<SampleOrder> order;
    std::array<std::uint16_t, 4> quadSampleMask{0xffffu, 0xffffu, 0xffffu, 0xffffu};
    bool storesInvocations = false;
    bool buildsNoRecipe = false;
};

// Restores the device's sample order when a draw under a fallback context returns or throws.
struct SuppressedOrder {
    explicit SuppressedOrder(bool suppress) { Metal::SetSampleOrderSuppressed(suppress); }
    ~SuppressedOrder() { Metal::SetSampleOrderSuppressed(false); }
};

// Upstream's Draw through the Metal device. `context` names the path as upstream's contexts do: the device's own takes
// the exact path; one without multisampled storage (upstream's fallback) draws with pixel (0,0)'s locations alone,
// which on Metal is the device with its sample order suppressed (MetalSampleOrder.hpp test seam).
void Draw(AgcDriver::VulkanDevice& device, const Context& context, const Pass& pass) {
    const auto target = device.Target();
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(pass.vertexMemory != nullptr ? pass.vertexMemory : pass.vertices->data(), 16u, static_cast<std::uint32_t>(pass.vertices->size()));
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

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, WaveSize, WaveSize, std::nullopt, std::nullopt};
    state.hasColorTarget = pass.color.has_value();
    if (pass.color) {
        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask = 0xf;
        state.colors = {*pass.color};
        state.color = *pass.color;
        state.blends = {blend};
        state.blend = blend;
    }
    state.renderExtent = pass.extent;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(pass.extent.height), static_cast<float>(pass.extent.width), -static_cast<float>(pass.extent.height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, pass.extent};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blendConstants = {};
    state.samples = pass.samples;
    state.perSampleShading = pass.perSample;
    state.quadSampleMask = pass.quadSampleMask;
    state.pixelEarlyZ = AgcDriver::Graphics::PixelEarlyZ(pass.shaderControl);
    if (pass.locations) {
        state.customSampleLocations = true;
        state.sampleLocations = *pass.locations;
        for (std::uint32_t pixel = 1; pixel < 4u; ++pixel) {
            for (std::uint32_t sample = 0; sample < pass.samples; ++sample) state.sampleLocationGrid |= state.sampleLocations[pixel * 8u + sample] != state.sampleLocations[sample];
        }
    }
    if (pass.depth) {
        state.depth = pass.depth;
        state.depthTest = true;
        state.depthWrite = pass.depthWrite;
        state.depthCompare = pass.depthCompare;
    }

    const std::uint32_t inputs = pass.perSample ? PixelInputBit(PixelInput::PerspectiveSample) | PixelInputBit(PixelInput::LinearSample) | PixelInputBit(PixelInput::PositionX) | PixelInputBit(PixelInput::PositionY) | PixelInputBit(PixelInput::Ancillary) : 0u;
    const AgcDriver::Registers registers{{0x1b3, inputs}, {0x1b4, inputs}, {0x1b6, 0u}, {0x1c5, pass.colorFormat}, {0x203, pass.shaderControl}};
    auto pixel = AgcDriver::Graphics::DecodePixelStageInfo(registers, AgcDriver::Graphics::ExportMappings(state));
    Require(pixel.sampleShading == pass.perSample, "the pixel inputs did not select the requested shading rate");
    pixel.iteratesSamples = state.perSampleShading;
    pixel.sampleLocations = AgcDriver::Graphics::PixelSampleLocations(state, pixel);
    const auto order = pass.order ? *pass.order : state.sampleLocationGrid ? AgcDriver::Graphics::HostSampleLocationsFor(context, state).order : SampleOrder{};
    pixel.sampleOrder = order;
    pixel.quadPixelMask = AgcDriver::Graphics::QuadPixelMask(state);
    pixel.quadSampleMask = AgcDriver::Graphics::QuadSampleMasks(state, order);
    std::vector<std::uint32_t> pixelUserData(4, 0u);
    if (pass.storesInvocations) {
        const auto invocations = BufferDescriptor(Invocations.data(), 0u, static_cast<std::uint32_t>(sizeof(Invocations)));
        std::copy(invocations.begin(), invocations.end(), pixelUserData.begin());
    }
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(pass.code.data()), std::as_bytes(pass.code)}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(pass.code.data()), pass.code, 0, {}},
        {WaveSize, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    fragment.useCache = false;
    const auto pixelResult = ShaderRecompiler::Recompile(fragment);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{{ShaderStage::Vertex, &vertexResult, 0}, {ShaderStage::Fragment, &pixelResult, PixelPushOffset(vertexPush, target)}}};
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(pass.vertices->size()), 0, 1, 0, false};
    SuppressedOrder suppressed(!context.storageImageMultisample);
    if (!pass.buildsNoRecipe) {
        device.Draw(state, draw, shaders);
        return;
    }
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    device.Draw(state, draw, shaders, {}, &recipe);
    Require(recipe == nullptr, "a draw that reorders its samples built a recipe");
}

void Settle(AgcDriver::VulkanDevice& device) {
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        device.SubmitRecorded(true);
    }
    device.WaitIdle();
}

Vertices Rectangle(float left, float right, float z) {
    return {{left, -1.0f, z, 1.0f}, {right, -1.0f, z, 1.0f}, {right, 1.0f, z, 1.0f}, {left, -1.0f, z, 1.0f}, {right, 1.0f, z, 1.0f}, {left, 1.0f, z, 1.0f}};
}

Vertices Screen(float z) {
    return {{-1.0f, -1.0f, z, 1.0f}, {3.0f, -1.0f, z, 1.0f}, {-1.0f, 3.0f, z, 1.0f}};
}

float StandardX(std::uint32_t samples, std::uint32_t sample) {
    static constexpr std::array<float, 2> two{0.75f, 0.25f};
    static constexpr std::array<float, 4> four{0.375f, 0.875f, 0.125f, 0.625f};
    static constexpr std::array<float, 8> eight{0.5625f, 0.4375f, 0.8125f, 0.3125f, 0.1875f, 0.0625f, 0.6875f, 0.9375f};
    return samples == 2u ? two[sample] : samples == 4u ? four[sample] : eight[sample];
}

void CheckStored(std::uint32_t samples, const std::string& what, const std::function<std::uint32_t(std::uint32_t, std::uint32_t)>& expected) {
    for (std::uint32_t y = 0; y < RemapHeight; ++y) {
        for (std::uint32_t x = 0; x < RemapWidth; ++x) {
            const auto mask = expected(x, y);
            for (std::uint32_t sample = 0; sample < samples; ++sample) {
                const auto stored = Invocations[(y * RemapWidth + x) * samples + sample];
                Require(stored == ((mask >> sample) & 1u), what + ": guest " + At(x, y, sample) + (stored != 0u ? " ran" : " did not run") + ", expected guest mask " + std::to_string(mask));
            }
        }
    }
}

std::uint32_t GuestBits(std::uint32_t hostBits, std::uint32_t word, std::uint32_t samples) {
    std::uint32_t guest = 0;
    for (std::uint32_t host = 0; host < samples; ++host) {
        if (((hostBits >> host) & 1u) != 0u) guest |= 1u << Pick(word, host);
    }
    return guest;
}

void SampleIndexRemap(AgcDriver::VulkanDevice& device, const Context& context, std::uint32_t samples) {
    const auto order = OrderFor(samples);
    const auto toNdc = [](float pixel) { return pixel * 2.0f / static_cast<float>(RemapWidth) - 1.0f; };
    const auto rectangle = Rectangle(toNdc(2.5f), toNdc(5.5f), 0.5f);
    for (const bool remapped : {false, true}) {
        Invocations.fill(0u);
        Pass pass;
        pass.vertices = &rectangle;
        pass.code = StoreCode(samples);
        pass.perSample = true;
        pass.samples = samples;
        pass.order = remapped ? order : SampleOrder{};
        pass.storesInvocations = true;
        Draw(device, context, pass);
        Settle(device);
        CheckStored(samples, std::to_string(samples) + "x SampleID " + (remapped ? "under the test's sample order" : "in the host's order"), [&](std::uint32_t x, std::uint32_t y) {
            std::uint32_t host = 0;
            for (std::uint32_t sample = 0; sample < samples; ++sample) {
                const auto at = static_cast<float>(x) + StandardX(samples, sample);
                if (at > 2.5f && at < 5.5f) host |= 1u << sample;
            }
            return remapped ? GuestBits(host, order[Quad(x, y)], samples) : host;
        });
    }
    std::printf("%ux SampleID reads the guest's sample index in all four pixels of the 2x2 grid\n", samples);
}

void CoverageRemap(AgcDriver::VulkanDevice& device, const Context& context, std::uint32_t samples) {
    const auto order = OrderFor(samples);
    const auto back = Screen(0.25f), middle = Screen(0.5f), probe = Screen(0.375f);
    const AgcDriver::Graphics::DepthTarget depth = [&] {
        AgcDriver::Graphics::DepthTarget target{reinterpret_cast<std::uintptr_t>(DepthMemory.data()) + 256u * 1024u * static_cast<std::uint32_t>(std::countr_zero(samples)), 0u, {RemapWidth, RemapHeight}, VK_FORMAT_D32_SFLOAT, 0.0f, 0};
        target.samples = samples;
        return target;
    }();
    const std::array<std::uint32_t, 2> masks{1u, samples == 2u ? 2u : samples == 4u ? 6u : 0x96u};
    std::size_t slot = 0;
    for (const auto mask : masks) {
        Pass reset;
        reset.vertices = &back;
        reset.code = NullExportCode;
        reset.samples = samples;
        reset.depth = depth;
        reset.depthWrite = true;
        Draw(device, context, reset);
        Pass masked = reset;
        masked.vertices = &middle;
        masked.code = MaskCode(slot++ + (samples == 2u ? 0u : samples == 4u ? 2u : 4u), mask);
        masked.shaderControl = 0x100u;
        masked.order = order;
        Draw(device, context, masked);
        Settle(device);
        Invocations.fill(0u);
        Pass check;
        check.vertices = &probe;
        check.code = StoreCode(samples);
        check.perSample = true;
        check.shaderControl = 0x10u;
        check.samples = samples;
        check.depth = depth;
        check.depthCompare = VK_COMPARE_OP_LESS;
        check.order = order;
        check.storesInvocations = true;
        Draw(device, context, check);
        Settle(device);
        CheckStored(samples, std::to_string(samples) + "x samples a pixel shader's exported guest mask " + std::to_string(mask) + " wrote depth into", [&](std::uint32_t, std::uint32_t) { return mask; });
    }
    std::printf("%ux exported sample masks reach the host samples at the guest samples' locations\n", samples);
}

void CoverageOrder(AgcDriver::VulkanDevice& device, const Context& context, std::uint32_t samples) {
    const auto order = OrderFor(samples);
    const auto toNdc = [](float pixel) { return pixel * 2.0f / static_cast<float>(RemapWidth) - 1.0f; };
    const auto rectangle = Rectangle(toNdc(2.5f), toNdc(5.5f), 0.5f);
    const auto back = Screen(0.25f), probe = Screen(0.375f);
    const AgcDriver::Graphics::DepthTarget depth = [&] {
        AgcDriver::Graphics::DepthTarget target{reinterpret_cast<std::uintptr_t>(DepthMemory.data()) + 256u * 1024u * (static_cast<std::uint32_t>(std::countr_zero(samples)) + 4u), 0u, {RemapWidth, RemapHeight}, VK_FORMAT_D32_SFLOAT, 0.0f, 0};
        target.samples = samples;
        return target;
    }();
    const std::array<std::uint16_t, 4> mask{2u, 2u, 2u, 4u};
    const auto full = (1u << samples) - 1u;
    const auto covered = [&](std::uint32_t x, std::uint32_t y) {
        std::uint32_t host = 0u;
        for (std::uint32_t sample = 0; sample < samples; ++sample) {
            const auto at = static_cast<float>(x) + StandardX(samples, sample);
            if (at > 2.5f && at < 5.5f) host |= 1u << sample;
        }
        return GuestBits(host, order[Quad(x, y)], samples) & (mask[Quad(x, y)] & full);
    };
    Pass reset;
    reset.vertices = &back;
    reset.code = NullExportCode;
    reset.samples = samples;
    reset.depth = depth;
    reset.depthWrite = true;
    Draw(device, context, reset);
    Invocations.fill(0u);
    Pass masked;
    masked.vertices = &rectangle;
    masked.code = StoreCode(samples);
    masked.perSample = true;
    masked.samples = samples;
    masked.depth = depth;
    masked.depthWrite = true;
    masked.order = order;
    masked.quadSampleMask = mask;
    masked.storesInvocations = true;
    Draw(device, context, masked);
    Settle(device);
    CheckStored(samples, std::to_string(samples) + "x a quad sample mask reached the pixel shader in the guest's order", covered);
    Invocations.fill(0u);
    Pass check;
    check.vertices = &probe;
    check.code = StoreCode(samples);
    check.perSample = true;
    check.shaderControl = 0x10u;
    check.samples = samples;
    check.depth = depth;
    check.depthCompare = VK_COMPARE_OP_LESS;
    check.order = order;
    check.storesInvocations = true;
    Draw(device, context, check);
    Settle(device);
    CheckStored(samples, std::to_string(samples) + "x a quad sample mask exported through the sample order landed on the host's samples", covered);
    std::printf("%ux a quad sample mask reaches the pixel shader in the guest's order and exports coverage in the host's\n", samples);
}

std::uint32_t ShaderRemaps(AgcDriver::VulkanDevice& device, const Context& context) {
    std::uint32_t checked = 0;
    for (const std::uint32_t samples : {2u, 4u, 8u}) {
        if ((context.limits.framebufferDepthSampleCounts & samples) == 0u || (context.limits.framebufferNoAttachmentsSampleCounts & samples) == 0u) continue;
        if (context.limits.standardSampleLocations) {
            SampleIndexRemap(device, context, samples);
            ++checked;
        } else {
            std::printf("skipped the %ux SampleID check: the device has no standard sample locations\n", samples);
        }
        CoverageRemap(device, context, samples);
        CoverageOrder(device, context, samples);
        ++checked;
    }
    return checked;
}

constexpr std::size_t BlockBytes = 65536;

class GuestBlock {
public:
    explicit GuestBlock(std::size_t bytes) : bytes((bytes + BlockBytes - 1u) & ~(BlockBytes - 1u)) {
        block = static_cast<std::byte*>(std::aligned_alloc(BlockBytes, this->bytes));
        Require(block != nullptr, "sample order test: cannot allocate a guest block");
        std::memset(block, 0, this->bytes);
        GuestAllocations::Mutation().Add(block, this->bytes, true, true, true);
    }
    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;
    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }

private:
    std::size_t bytes;
    std::byte* block = nullptr;
};

GuestBlock& NewBlock(std::size_t bytes) {
    static auto* blocks = new std::deque<GuestBlock>();
    return blocks->emplace_back(bytes);
}

constexpr VkExtent2D ReferenceExtent{64u, 32u};
constexpr std::array<std::uint8_t, 32> VillageLocations{0x04, 0x0c, 0, 0, 0, 0, 0, 0, 0x04, 0x0c, 0, 0, 0, 0, 0, 0, 0x0c, 0x04, 0, 0, 0, 0, 0, 0, 0x0c, 0x04, 0, 0, 0, 0, 0, 0};
constexpr float ClearDepth = 0.9375f;
constexpr std::size_t SceneSlots = 6u;
const Vertices Slopes{{-1.0f, -1.0f, 0.125f, 1.0f}, {1.0f, -0.8125f, 0.875f, 1.0f}, {-0.5625f, 1.0f, 0.5f, 1.0f}, {1.0f, -1.0f, 0.75f, 1.0f}, {0.6875f, 1.0f, 0.0625f, 1.0f}, {-1.0f, 0.4375f, 0.3125f, 1.0f}};

struct Rendered {
    Samples color;
    Samples depth;
};

double SlopeDepth(std::size_t triangle, double x, double y) {
    std::array<std::array<double, 3>, 3> corner{};
    for (std::size_t vertex = 0; vertex < corner.size(); ++vertex) {
        const auto& at = Slopes[triangle * 3u + vertex];
        corner[vertex] = {(at[0] + 1.0) * ReferenceExtent.width / 2.0, (1.0 - at[1]) * ReferenceExtent.height / 2.0, at[2]};
    }
    const auto edge = [](const std::array<double, 3>& from, const std::array<double, 3>& to, double px, double py) { return (to[0] - from[0]) * (py - from[1]) - (to[1] - from[1]) * (px - from[0]); };
    const auto area = edge(corner[0], corner[1], corner[2][0], corner[2][1]);
    return (edge(corner[1], corner[2], x, y) * corner[0][2] + edge(corner[2], corner[0], x, y) * corner[1][2] + edge(corner[0], corner[1], x, y) * corner[2][2]) / area;
}

struct DepthLocations {
    std::size_t sloped = 0;
    std::size_t guest = 0;
    std::size_t standard = 0;
    std::array<std::size_t, 2> slopedByRow{};
    std::array<std::size_t, 2> guestByRow{};
};

DepthLocations LocateDepth(const Samples& depth, const std::array<std::uint8_t, 32>& locations = VillageLocations) {
    constexpr std::uint32_t samples = 2u;
    constexpr std::array<std::array<double, 2>, 2> standard{{{0.75, 0.75}, {0.25, 0.25}}};
    DepthLocations located;
    for (std::size_t index = 0; index < depth.size(); ++index) {
        const auto value = static_cast<double>(std::bit_cast<float>(depth[index][0]));
        if (value == ClearDepth) continue;
        ++located.sloped;
        const auto pixel = static_cast<std::uint32_t>(index / samples);
        const auto sample = static_cast<std::uint32_t>(index % samples);
        const auto x = pixel % ReferenceExtent.width;
        const auto y = pixel / ReferenceExtent.width;
        ++located.slopedByRow[y & 1u];
        const auto bits = static_cast<std::uint32_t>(locations[Quad(x, y) * 8u + sample]);
        const std::array<double, 2> guest{0.5 + (static_cast<int>(bits & 7u) - static_cast<int>(bits & 8u)) / 16.0, 0.5 + (static_cast<int>((bits >> 4u) & 7u) - static_cast<int>((bits >> 4u) & 8u)) / 16.0};
        const auto holds = [&](const std::array<double, 2>& at) {
            for (std::size_t triangle = 0; triangle < 2u; ++triangle) {
                if (std::abs(SlopeDepth(triangle, x + at[0], y + at[1]) - value) < 1.0e-5) return true;
            }
            return false;
        };
        if (holds(guest)) {
            ++located.guest;
            ++located.guestByRow[y & 1u];
        } else if (holds(standard[sample])) {
            ++located.standard;
        }
    }
    return located;
}

DepthLocations PrintDepthLocations(const std::string& what, const DepthLocations& located) {
    std::printf("%s: of %zu sloped depth samples, %zu hold the slope's depth at the guest's sample locations (%zu of %zu on even rows, %zu of %zu on odd rows) and %zu at the standard 2x locations\n", what.c_str(), located.sloped, located.guest, located.guestByRow[0], located.slopedByRow[0], located.guestByRow[1], located.slopedByRow[1], located.standard);
    return located;
}

DepthLocations ReportDepthLocations(const std::string& what, const Rendered& rendered, const std::array<std::uint8_t, 32>& locations = VillageLocations) {
    return PrintDepthLocations(what, LocateDepth(rendered.depth, locations));
}

// A device target (MultisampleTargetView, DepthSurfaceOrderTarget) read sample by sample in the guest's order.
Samples ReadTarget(const Context& context, VkImageView view, bool color, std::uint32_t samples) {
    id<MTLTexture> texture = Metal::TextureOfView(view);
    Require(texture != nil && texture.textureType == MTLTextureType2DMultisampleArray, "a device target the test reads is not a 2D multisample array");
    // Color targets of these scenes are RGBA16F: their samples read as float bits, as upstream's ReadColor does.
    return Read(context, texture, color ? 16u : 0u, samples, ReferenceExtent);
}

Rendered RenderCheckerboard(AgcDriver::VulkanDevice& device, const Context& context, std::uint32_t samples, std::size_t run, const std::optional<std::array<std::uint8_t, 32>>& locations) {
    AgcDriver::Graphics::ColorTarget color{};
    color.extent = ReferenceExtent;
    color.surfaceExtent = ReferenceExtent;
    color.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    color.componentMapping = 0xe4u;
    color.tileMode = AgcDriver::Graphics::ColorTileMode::RenderTarget;
    color.elementBytes = 8u;
    color.samples = samples;
    color.bytes = AgcDriver::Graphics::ColorTargetLayout(ReferenceExtent.width, ReferenceExtent.height, color.tileMode, color.elementBytes).Bytes();
    auto& block = NewBlock(AgcDriver::Graphics::MultisampledColorBytes(color));
    color.address = block.Address();
    color.surfaceAddress = color.address;
    AgcDriver::Graphics::DepthTarget depth{reinterpret_cast<std::uintptr_t>(DepthMemory.data()) + 256u * 1024u * run, 0u, ReferenceExtent, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    depth.samples = samples;
    const Vertices clear = Screen(ClearDepth);
    const Vertices shade{{-0.9375f, -0.875f, 0.25f, 1.0f}, {0.9375f, -0.625f, 0.6875f, 1.0f}, {0.0625f, 0.9375f, 0.4375f, 1.0f}};
    Pass depthOnly;
    depthOnly.vertices = &clear;
    depthOnly.code = NullExportCode;
    depthOnly.samples = samples;
    depthOnly.extent = ReferenceExtent;
    depthOnly.depth = depth;
    depthOnly.depthWrite = true;
    depthOnly.locations = locations;
    Draw(device, context, depthOnly);
    depthOnly.vertices = &Slopes;
    depthOnly.depthCompare = VK_COMPARE_OP_LESS;
    Draw(device, context, depthOnly);
    Pass colorPass;
    colorPass.vertices = &clear;
    colorPass.code = SampleColorCode;
    colorPass.perSample = true;
    colorPass.colorFormat = 9u;
    colorPass.samples = samples;
    colorPass.extent = ReferenceExtent;
    colorPass.depth = depth;
    colorPass.depthCompare = VK_COMPARE_OP_ALWAYS;
    colorPass.color = color;
    colorPass.locations = locations;
    Draw(device, context, colorPass);
    colorPass.vertices = &shade;
    colorPass.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL;
    Draw(device, context, colorPass);
    Settle(device);
    Rendered rendered;
    rendered.color = ReadTarget(context, AgcDriver::Graphics::MultisampleTargetView(context, color), true, samples);
    rendered.depth = ReadTarget(context, AgcDriver::Graphics::DepthSurfaceOrderTarget(context, depth).view, false, samples);
    return rendered;
}

std::size_t Differences(const Samples& reference, const Samples& rendered, std::uint32_t samples, const std::string& what, bool report) {
    std::size_t differences = 0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        if (reference[index] == rendered[index]) continue;
        if (report && differences < 4u) {
            const auto pixel = static_cast<std::uint32_t>(index / samples);
            std::printf("%s: %s is 0x%08x 0x%08x 0x%08x 0x%08x, the reference's is 0x%08x 0x%08x 0x%08x 0x%08x\n", what.c_str(), At(pixel % ReferenceExtent.width, pixel / ReferenceExtent.width, static_cast<std::uint32_t>(index % samples)).c_str(), rendered[index][0], rendered[index][1], rendered[index][2], rendered[index][3], reference[index][0], reference[index][1], reference[index][2], reference[index][3]);
        }
        ++differences;
    }
    return differences;
}

struct ClearTargets {
    AgcDriver::VulkanDevice& device;
    VkDevice handle;
    ~ClearTargets() {
        Metal::SetSampleOrderSuppressed(false);
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        device.SubmitRecorded(true);
        device.WaitIdle();
        AgcDriver::Graphics::ClearMultisampleTargets(handle);
        AgcDriver::Graphics::ClearCachedTextures(handle);
    }
};

void CheckScene(const Rendered& rendered, std::uint32_t samples, const std::string& what) {
    std::size_t cleared = 0;
    for (std::size_t index = 0; index < rendered.depth.size(); ++index) {
        const auto sample = static_cast<std::uint32_t>(index % samples);
        const auto pixel = static_cast<std::uint32_t>(index / samples);
        const auto at = At(pixel % ReferenceExtent.width, pixel / ReferenceExtent.width, sample);
        const auto id = static_cast<float>(sample);
        const auto z = std::bit_cast<float>(rendered.color[index][2]);
        const auto w = std::bit_cast<float>(rendered.color[index][3]);
        Require(z == id && w == id, what + ": " + at + " does not hold its own sample index in color (z " + std::to_string(z) + ", w " + std::to_string(w) + ")");
        const auto depth = std::bit_cast<float>(rendered.depth[index][0]);
        Require(depth >= 0.0f && depth <= ClearDepth, what + ": " + at + " holds depth " + std::to_string(depth));
        if (depth == ClearDepth) ++cleared;
    }
    Require(cleared != 0u && cleared != rendered.depth.size(), what + ": the slopes did not leave depth both at and below the clear value");
}

// Every Metal depth surface is shader-readable (DepthSurface.mm), so upstream's DepthSampleable gate is the sample count.
std::uint32_t SceneChecks(AgcDriver::VulkanDevice& device, const Context& context) {
    ClearTargets clearTargets{device, context.device};
    std::uint32_t checked = 0;
    for (const std::uint32_t samples : {2u, 4u, 8u}) {
        if ((context.limits.framebufferColorSampleCounts & samples) == 0u || (context.limits.framebufferDepthSampleCounts & samples) == 0u) continue;
        CheckScene(RenderCheckerboard(device, context, samples, SceneSlots + static_cast<std::size_t>(std::countr_zero(samples)), std::nullopt), samples, std::to_string(samples) + "x scene with the standard locations");
        ++checked;
    }
    return checked;
}

std::array<std::uint8_t, 32> UniformLocations(const std::array<std::uint8_t, 32>& pattern, std::uint32_t pixel) {
    std::array<std::uint8_t, 32> uniform{};
    for (std::uint32_t target = 0; target < 4u; ++target) std::copy_n(pattern.begin() + 8u * pixel, 8u, uniform.begin() + 8u * target);
    return uniform;
}

Rendered QuadComposite(AgcDriver::VulkanDevice& device, const Context& context, std::uint32_t samples, std::size_t run, const std::array<std::uint8_t, 32>& pattern) {
    std::array<Rendered, 4> parts;
    for (std::uint32_t pixel = 0; pixel < parts.size(); ++pixel) parts[pixel] = RenderCheckerboard(device, context, samples, run + pixel, UniformLocations(pattern, pixel));
    auto composite = parts[0];
    for (std::size_t index = 0; index < composite.color.size(); ++index) {
        const auto pixel = static_cast<std::uint32_t>(index / samples);
        const auto& part = parts[Quad(pixel % ReferenceExtent.width, pixel / ReferenceExtent.width)];
        composite.color[index] = part.color[index];
        composite.depth[index] = part.depth[index];
    }
    return composite;
}

// Metal has no 2x2 location grid (setSamplePositions is one pattern per pass): the device's own context is the
// exact path, and the composite of four uniform renders is the reference, as upstream on such a device.
int GridReference(AgcDriver::VulkanDevice& device) {
    const auto context = device.graphicsContext();
    ClearTargets clearTargets{device, context.device};
    constexpr std::uint32_t samples = 2u;
    Require((context.limits.framebufferColorSampleCounts & samples) != 0u && (context.limits.framebufferDepthSampleCounts & samples) != 0u, "the Metal device does not report 2x color and depth targets");
    Require((context.sampleLocationSampleCounts & samples) != 0u, "the Metal device does not report custom sample locations at 2 samples (programmable sample positions)");
    Require(context.sampleLocationGridSampleCounts == 0u, "the Metal device reports a 2x2 sample location grid, which setSamplePositions cannot draw");
    const auto& emulated = context;
    auto fallback = context;
    fallback.storageImageMultisample = false;
    AgcDriver::Graphics::State probe{};
    probe.samples = samples;
    probe.customSampleLocations = true;
    probe.sampleLocationGrid = true;
    probe.sampleLocations = VillageLocations;
    AgcDriver::Graphics::ColorTarget color{};
    color.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    color.elementBytes = 8u;
    color.samples = samples;
    probe.colors = {color};
    probe.depth = AgcDriver::Graphics::DepthTarget{0x100000u, 0u, ReferenceExtent, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    probe.depth->samples = samples;
    const auto refusal = AgcDriver::Graphics::SampleOrderRefusal(emulated, probe);
    Require(refusal.empty(), "the Metal device does not put the checkerboard's samples in the guest's order: " + refusal);
    Require(AgcDriver::Graphics::ReordersSamples(AgcDriver::Graphics::HostSampleLocationsFor(emulated, probe).order) && !AgcDriver::Graphics::ReordersSamples(AgcDriver::Graphics::HostSampleLocationsFor(fallback, probe).order), "the exact and fallback contexts did not take the exact and approximate paths");
    Require(AgcDriver::Graphics::ReordersSamples(device.HostSampleOrder(probe)), "the Metal device names no sample order for the checkerboard");
    const auto reorders = Metal::SampleOrderReorders();
    const auto reference = QuadComposite(device, context, samples, 3u, VillageLocations);
    Require(Metal::SampleOrderReorders() == reorders, "the uniform renders of the composite reordered samples");
    const auto exact = RenderCheckerboard(device, emulated, samples, 1u, VillageLocations);
    const auto exactReorders = Metal::SampleOrderReorders() - reorders;
    Require(exactReorders != 0u, "the checkerboard through the exact path recorded no reorder pass");
    const auto approximate = RenderCheckerboard(device, fallback, samples, 2u, VillageLocations);
    Require(Metal::SampleOrderReorders() == reorders + exactReorders, "the checkerboard with its sample order suppressed reordered samples");
    CheckScene(reference, samples, "the composite of uniform renders");
    CheckScene(exact, samples, "the exact path");
    CheckScene(approximate, samples, "pixel (0,0)'s locations alone");
    const auto referenceDepth = ReportDepthLocations("the composite of uniform renders", reference);
    ReportDepthLocations("the exact path", exact);
    ReportDepthLocations("pixel (0,0)'s locations alone", approximate);
    Require(referenceDepth.guest == referenceDepth.sloped, "the Metal device does not keep depth drawn at custom sample locations at those locations: the composite of uniform renders holds the guest's depth in " + std::to_string(referenceDepth.guest) + " of " + std::to_string(referenceDepth.sloped) + " sloped samples");
    std::printf("the device has no 2x2 sample location grid at 2 samples: the composite of four uniform renders is the reference (the exact path took %llu reorder passes)\n", static_cast<unsigned long long>(exactReorders));
    const auto colorDifferences = Differences(reference.color, exact.color, samples, "color with the samples put in the guest's order", true);
    const auto depthDifferences = Differences(reference.depth, exact.depth, samples, "depth with the samples put in the guest's order", true);
    const auto approximateDifferences = Differences(reference.color, approximate.color, samples, "color", false) + Differences(reference.depth, approximate.depth, samples, "depth", false);
    std::printf("2x checkerboard at %ux%u: the exact path differs from the reference in %zu color and %zu depth samples; pixel (0,0)'s locations alone differ in %zu samples\n", ReferenceExtent.width, ReferenceExtent.height, colorDifferences, depthDifferences, approximateDifferences);
    Require(approximateDifferences != 0u, "pixel (0,0)'s locations alone rendered the checkerboard like the reference: the scene does not tell the paths apart");
    Require(colorDifferences == 0u && depthDifferences == 0u, "the exact path did not render the checkerboard bit for bit like the reference");
    std::puts("the exact path renders Village's 2x checkerboard like four uniform renders composed by pixel parity, bit for bit");
    return 0;
}

// Upstream's recipe replay on the Metal device: the device builds no recipe (DrawFromRecipe rebuilds), so the replay is
// the rebuild a recipe miss takes, the same draw recorded again as the first draw of a pass after a clear.
void CheckRecipeDepth(AgcDriver::VulkanDevice& device, const Context& context, const std::array<std::uint8_t, 32>& locations, std::size_t run, const std::string& what, bool reordered) {
    constexpr std::uint32_t samples = 2u;
    AgcDriver::Graphics::DepthTarget depth{reinterpret_cast<std::uintptr_t>(DepthMemory.data()) + 256u * 1024u * run, 0u, ReferenceExtent, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    depth.samples = samples;
    const Vertices clear = Screen(ClearDepth);
    Pass clearPass;
    clearPass.vertices = &clear;
    clearPass.code = NullExportCode;
    clearPass.samples = samples;
    clearPass.extent = ReferenceExtent;
    clearPass.depth = depth;
    clearPass.depthWrite = true;
    clearPass.locations = locations;
    auto slopes = clearPass;
    slopes.vertices = &Slopes;
    auto& vertexBlock = NewBlock(Slopes.size() * sizeof(Slopes[0]));
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(vertexBlock.Address())), Slopes.data(), Slopes.size() * sizeof(Slopes[0]));
    slopes.vertexMemory = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(vertexBlock.Address()));
    const auto read = [&] { return ReadTarget(context, AgcDriver::Graphics::DepthSurfaceOrderTarget(context, depth).view, false, samples); };
    if (reordered) {
        slopes.buildsNoRecipe = true;
        const auto reorders = Metal::SampleOrderReorders();
        Draw(device, context, clearPass);
        Draw(device, context, slopes);
        Settle(device);
        Require(Metal::SampleOrderReorders() != reorders, what + ": the draws recorded no reorder pass");
        const auto located = PrintDepthLocations(what, LocateDepth(read(), locations));
        Require(located.sloped != 0u && located.guest == located.sloped, what + ": the reordered depth does not hold the slope's depth at the guest's sample locations in every sloped sample");
        std::printf("%s: builds no recipe, since a draw that reorders its samples is built in full each time, with its sample swaps\n", what.c_str());
        return;
    }
    Draw(device, context, clearPass);
    Draw(device, context, slopes);
    Settle(device);
    const auto recorded = read();
    Draw(device, context, clearPass);
    Settle(device);
    const auto cleared = read();
    Require(std::all_of(cleared.begin(), cleared.end(), [](const std::array<std::uint32_t, 4>& value) { return std::bit_cast<float>(value[0]) == ClearDepth; }), what + ": the clear between the recorded draw and its rebuild left depth other than the clear value");
    Draw(device, context, slopes);
    Settle(device);
    const auto replayed = read();
    const auto first = PrintDepthLocations(what + ", the recorded draw", LocateDepth(recorded, locations));
    PrintDepthLocations(what + ", the rebuild as the first draw of a pass", LocateDepth(replayed, locations));
    Require(first.sloped != 0u, what + ": the recorded draw left no sloped depth");
    Require(first.guest == first.sloped, what + ": the Metal device did not keep depth drawn at custom sample locations at those locations");
    const auto differences = Differences(recorded, replayed, samples, what + ", depth rebuilt", true);
    std::printf("%s, %ux%u: the rebuild differs from the recorded draw in %zu of %zu samples\n", what.c_str(), ReferenceExtent.width, ReferenceExtent.height, differences, recorded.size());
    Require(differences == 0u, what + ": the rebuild did not leave the depth the recorded draw left at the guest's sample locations");
    std::printf("%s: a depth-only 2x draw rebuilt as the first draw of a pass leaves the depth the recorded draw left, bit for bit\n", what.c_str());
}

int RecipeDepthLocations(AgcDriver::VulkanDevice& device) {
    const auto context = device.graphicsContext();
    ClearTargets clearTargets{device, context.device};
    constexpr std::uint32_t samples = 2u;
    Require((context.limits.framebufferDepthSampleCounts & samples) != 0u, "the Metal device does not report 2x depth targets");
    Require((context.sampleLocationSampleCounts & samples) != 0u, "the Metal device does not report custom sample locations at 2 samples (programmable sample positions)");
    CheckRecipeDepth(device, context, UniformLocations(VillageLocations, 0u), 0u, "pixel (0,0)'s locations in every pixel", false);
    AgcDriver::Graphics::State probe{};
    probe.samples = samples;
    probe.customSampleLocations = true;
    probe.sampleLocationGrid = true;
    probe.sampleLocations = VillageLocations;
    probe.depth = AgcDriver::Graphics::DepthTarget{0x100000u, 0u, ReferenceExtent, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    probe.depth->samples = samples;
    const auto refusal = AgcDriver::Graphics::SampleOrderRefusal(context, probe);
    Require(refusal.empty(), "the Metal device does not put the checkerboard's samples in the guest's order: " + refusal);
    Require(AgcDriver::Graphics::ReordersSamples(AgcDriver::Graphics::HostSampleLocationsFor(context, probe).order), "the Metal device did not take the exact path");
    CheckRecipeDepth(device, context, VillageLocations, 2u, "the 2x checkerboard through the exact path", true);
    std::puts("the Metal device builds no draw recipes (TechnicalDebt metal-d6-bring-up-draws): the rebuild a recipe miss takes was checked in place of a replay");
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? std::string_view(argv[1]) : std::string_view();
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto context = device->graphicsContext();
        Require(context.sampleRateShading, "the Metal device does not report sampleRateShading ([[sample_id]] fragment functions run per sample)");
        Require(context.storageImageMultisample, "the Metal device does not report multisampled storage (D22/D28 storage shadows)");
        Require(Metal::SampleOrderContext(context), "the device's context is not a Metal context");
        if (mode == "--grid-reference") return GridReference(*device);
        if (mode == "--recipe-depth-locations") return RecipeDepthLocations(*device);
        Require(mode.empty(), "unknown mode " + std::string(mode));
        const auto swapped = SwapPasses(context);
        const auto remapped = ShaderRemaps(*device, context);
        const auto scenes = SceneChecks(*device, context);
        Require(swapped != 0u && remapped != 0u && scenes != 0u, "the Metal device ran no swap, shader remap or scene check (" + std::to_string(swapped) + ", " + std::to_string(remapped) + ", " + std::to_string(scenes) + ")");
        std::printf("sample order tests passed: %u swap round trips, %u shader remap checks, %u scene checks\n", swapped, remapped, scenes);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
