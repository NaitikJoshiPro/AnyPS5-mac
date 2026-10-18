// Work package D4: Metal twin of Graphics/src/TextureDetiler.cpp. The detile/retile kernels of
// Graphics/shaders/TextureDetile.comp (buffer to buffer, and buffer to image: LINEAR_IMAGE) and
// CmaskClear.comp in MSL, recorded into the command stream through its helper dispatch. The class
// keeps its header (Vulkan-typed members carry Metal objects, D-M2-3): `module` holds the library,
// `pipelines` the specialized pipeline states (CF-retained), `cmaskPipeline` the CMASK kernel.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "MetalCommandStream.h"
#include "MetalImages.h"
#include "MetalVulkanImages.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

[[noreturn]] void Fail(const std::string& message) {
    throw std::runtime_error("Metal texture detiler (work package D4): " + message);
}

constexpr const char* DetileSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint ELEMENT_BYTES [[function_constant(0)]];
constant uint BLOCK_BYTES [[function_constant(1)]];
constant uint TILE_FAMILY [[function_constant(2)]];
constant uint RETILE [[function_constant(3)]];
constant uint EQ0 [[function_constant(4)]];
constant uint EQ1 [[function_constant(5)]];
constant uint EQ2 [[function_constant(6)]];
constant uint EQ3 [[function_constant(7)]];
constant uint EQ4 [[function_constant(8)]];
constant uint EQ5 [[function_constant(9)]];
constant uint EQ6 [[function_constant(10)]];
constant uint EQ7 [[function_constant(11)]];
constant uint EQ8 [[function_constant(12)]];
constant uint EQ9 [[function_constant(13)]];
constant uint EQ10 [[function_constant(14)]];
constant uint EQ11 [[function_constant(15)]];
constant uint EQ12 [[function_constant(16)]];
constant uint EQ13 [[function_constant(17)]];
constant uint EQ14 [[function_constant(18)]];
constant uint EQ15 [[function_constant(19)]];
constant uint BLOCK_WIDTH [[function_constant(20)]];
constant uint BLOCK_HEIGHT [[function_constant(21)]];

struct Push {
    uint srcBase;
    uint dstBase;
    uint width;
    uint height;
    uint pitchBytes;
    uint blocksPerRow;
    uint tail;
    uint tailX;
    uint tailY;
    uint elementBytes;
    uint slice;
    uint rangeBegin;
    uint rangeEnd;
    uint tiledBase;
    uint linearBase;
    uint columnBegin;
    uint rowBegin;
    uint pipeBankXor;
};

struct BufferParams {
    Push push;
    device uint* input;
    device uint* output;
};

struct ImageParams {
    Push push;
    device uint* tiled;
    texture2d<uint, access::read_write> image;
};

static uint equationBit(uint mask, uint2 p, uint slice) {
    const uint selected = (p.x & (mask & 0xfffu)) ^ ((p.y << 12) & (mask & 0xfff000u)) ^ ((slice << 24) & (mask & 0xff000000u));
    return popcount(selected) & 1u;
}

static uint equationOffset(uint2 p, uint slice) {
    const uint eq[16] = {EQ0, EQ1, EQ2, EQ3, EQ4, EQ5, EQ6, EQ7, EQ8, EQ9, EQ10, EQ11, EQ12, EQ13, EQ14, EQ15};
    uint offset = 0u;
    for (uint bit = 0u; bit < 16u; ++bit) offset |= equationBit(eq[bit], p, slice) << bit;
    return offset;
}

static uint standardOffset(uint x, uint y) {
    switch (ELEMENT_BYTES) {
    case 1u:
        return ((y << 4) & 0x1f0u) ^ ((y << 5) & 0x400u) ^ (x & 0x00fu) ^ ((x << 5) & 0x200u) ^ ((x << 6) & 0x800u);
    case 2u:
        return ((y << 4) & 0x070u) ^ ((y << 5) & 0x100u) ^ ((y << 6) & 0x400u) ^ ((x << 1) & 0x00eu) ^ ((x << 4) & 0x080u) ^ ((x << 5) & 0x200u) ^ ((x << 6) & 0x800u);
    case 4u:
        return ((y << 4) & 0x070u) ^ ((y << 5) & 0x100u) ^ ((y << 6) & 0x400u) ^ ((x << 2) & 0x00cu) ^ ((x << 5) & 0x080u) ^ ((x << 6) & 0x200u) ^ ((x << 7) & 0x800u);
    case 8u:
        return ((y << 4) & 0x030u) ^ ((y << 6) & 0x100u) ^ ((y << 7) & 0x400u) ^ ((x << 3) & 0x008u) ^ ((x << 5) & 0x0c0u) ^ ((x << 6) & 0x200u) ^ ((x << 7) & 0x800u);
    default:
        return ((y << 4) & 0x030u) ^ ((y << 6) & 0x100u) ^ ((y << 7) & 0x400u) ^ ((x << 6) & 0x0c0u) ^ ((x << 7) & 0x200u) ^ ((x << 8) & 0x800u);
    }
}

static uint standard64Extra(uint x, uint y) {
    switch (ELEMENT_BYTES) {
    case 1u: return ((x << 7) & 0x2000u) ^ ((x << 8) & 0x8000u) ^ ((y << 6) & 0x1000u) ^ ((y << 7) & 0x4000u);
    case 2u: return ((x << 7) & 0x2000u) ^ ((x << 8) & 0x8000u) ^ ((y << 7) & 0x1000u) ^ ((y << 8) & 0x4000u);
    case 4u: return ((x << 8) & 0x2000u) ^ ((x << 9) & 0x8000u) ^ ((y << 7) & 0x1000u) ^ ((y << 8) & 0x4000u);
    case 8u: return ((x << 8) & 0x2000u) ^ ((x << 9) & 0x8000u) ^ ((y << 8) & 0x1000u) ^ ((y << 9) & 0x4000u);
    default: return ((x << 9) & 0x2000u) ^ ((x << 10) & 0x8000u) ^ ((y << 8) & 0x1000u) ^ ((y << 9) & 0x4000u);
    }
}

static uint blockOffset(uint2 p, constant Push& push) {
    if (TILE_FAMILY == 2u) return equationOffset(p, push.slice) ^ push.pipeBankXor;
    uint offset = standardOffset(p.x, p.y);
    if (BLOCK_BYTES > 4096u) offset ^= standard64Extra(p.x, p.y);
    return offset & (BLOCK_BYTES - 1u);
}

static uint2 blockExtent() {
    if (BLOCK_WIDTH != 0u) return uint2(BLOCK_WIDTH, BLOCK_HEIGHT);
    uint width;
    if (BLOCK_BYTES <= 256u) width = ELEMENT_BYTES <= 2u ? 16u : (ELEMENT_BYTES <= 8u ? 8u : 4u);
    else if (BLOCK_BYTES <= 4096u) width = ELEMENT_BYTES <= 2u ? 64u : (ELEMENT_BYTES <= 8u ? 32u : 16u);
    else width = ELEMENT_BYTES <= 2u ? 256u : (ELEMENT_BYTES <= 8u ? 128u : 64u);
    return uint2(width, BLOCK_BYTES / (width * ELEMENT_BYTES));
}

// The tiled byte offset of element p (UINT_MAX: outside the window's range).
static uint tiledOffsetOf(uint2 p, constant Push& push, thread uint& linearOffset) {
    linearOffset = p.y * push.pitchBytes + p.x * ELEMENT_BYTES;
    if (TILE_FAMILY == 0u) return linearOffset;
    uint2 swizzle = p;
    uint2 block = uint2(0u);
    if (push.tail != 0u) swizzle += uint2(push.tailX, push.tailY);
    else block = p / blockExtent();
    const uint blockIndex = block.y * push.blocksPerRow + block.x;
    return blockIndex * BLOCK_BYTES + blockOffset(swizzle, push);
}

static void copyElement(device uint* input, device uint* output, uint src, uint dst) {
    if (ELEMENT_BYTES >= 4u) {
        for (uint i = 0u; i < ELEMENT_BYTES; i += 4u) output[(dst + i) >> 2] = input[(src + i) >> 2];
    } else {
        const uint mask = ELEMENT_BYTES == 1u ? 0xffu : 0xffffu;
        const uint value = (input[src >> 2] >> ((src & 3u) * 8u)) & mask;
        const uint shift = (dst & 3u) * 8u;
        device atomic_uint* word = reinterpret_cast<device atomic_uint*>(output + (dst >> 2));
        atomic_fetch_and_explicit(word, ~(mask << shift), memory_order_relaxed);
        atomic_fetch_or_explicit(word, value << shift, memory_order_relaxed);
    }
}

kernel void detile(constant BufferParams& params [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
    constant Push& push = params.push;
    const uint2 p = id + uint2(push.columnBegin, push.rowBegin);
    if (p.x >= push.width || p.y >= push.height) return;
    uint linearOffset;
    uint tiledOffset = tiledOffsetOf(p, push, linearOffset);
    if (tiledOffset < push.rangeBegin || tiledOffset >= push.rangeEnd) return;
    if (TILE_FAMILY == 0u) {
        if (RETILE != 0u) copyElement(params.input, params.output, push.srcBase + linearOffset - push.linearBase, push.dstBase + linearOffset - push.tiledBase);
        else copyElement(params.input, params.output, push.srcBase + linearOffset - push.tiledBase, push.dstBase + linearOffset - push.linearBase);
        return;
    }
    tiledOffset -= push.tiledBase;
    linearOffset -= push.linearBase;
    if (RETILE != 0u) copyElement(params.input, params.output, push.srcBase + linearOffset, push.dstBase + tiledOffset);
    else copyElement(params.input, params.output, push.srcBase + tiledOffset, push.dstBase + linearOffset);
}

kernel void detile_image(constant ImageParams& params [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {
    constant Push& push = params.push;
    const uint2 p = id + uint2(push.columnBegin, push.rowBegin);
    if (p.x >= push.width || p.y >= push.height) return;
    uint linearOffset;
    uint tiledOffset = tiledOffsetOf(p, push, linearOffset);
    if (tiledOffset < push.rangeBegin || tiledOffset >= push.rangeEnd) return;
    const uint at = (TILE_FAMILY == 0u ? linearOffset : tiledOffset) - push.tiledBase;
    device uint* tiled = params.tiled;
    if (RETILE != 0u) {
        const uint4 value = params.image.read(p);
        const uint dst = push.dstBase + at;
        const uint word = dst >> 2;
        if (ELEMENT_BYTES >= 4u) {
            tiled[word] = value.x;
            if (ELEMENT_BYTES >= 8u) tiled[word + 1u] = value.y;
            if (ELEMENT_BYTES >= 16u) {
                tiled[word + 2u] = value.z;
                tiled[word + 3u] = value.w;
            }
        } else {
            const uint mask = ELEMENT_BYTES == 1u ? 0xffu : 0xffffu;
            const uint shift = (dst & 3u) * 8u;
            device atomic_uint* target = reinterpret_cast<device atomic_uint*>(tiled + word);
            atomic_fetch_and_explicit(target, ~(mask << shift), memory_order_relaxed);
            atomic_fetch_or_explicit(target, (value.x & mask) << shift, memory_order_relaxed);
        }
    } else {
        const uint src = push.srcBase + at;
        const uint word = src >> 2;
        uint4 value = uint4(0u);
        if (ELEMENT_BYTES == 1u) value.x = (tiled[word] >> ((src & 3u) * 8u)) & 0xffu;
        else if (ELEMENT_BYTES == 2u) value.x = (tiled[word] >> ((src & 3u) * 8u)) & 0xffffu;
        else {
            value.x = tiled[word];
            if (ELEMENT_BYTES >= 8u) value.y = tiled[word + 1u];
            if (ELEMENT_BYTES >= 16u) {
                value.z = tiled[word + 2u];
                value.w = tiled[word + 3u];
            }
        }
        params.image.write(value, p);
    }
}

struct CmaskParams {
    uint base;
    uint width;
    uint height;
    uint blocksPerRow;
    uint clear0;
    uint clear1;
    uint write;
    uint pad;
    device const uint* cmask;
    device atomic_uint* errors;
    texture2d<uint, access::write> target;
};

constant uint xMasks[13] = {0x008u, 0x000u, 0x040u, 0x000u, 0x080u, 0x000u, 0x100u, 0x000u, 0x200u, 0x008u, 0x010u, 0x040u, 0x020u};
constant uint yMasks[13] = {0x000u, 0x010u, 0x000u, 0x040u, 0x000u, 0x080u, 0x000u, 0x100u, 0x000u, 0x008u, 0x010u, 0x020u, 0x040u};

// Graphics/shaders/TextureRepack.comp: guest 10_11_11 float words (R in bits 0-9) as B10G11R11
// (R in bits 0-10), lossless; the view's BGRA swizzle restores the channels (upstream 571fecadd).
struct RepackParams {
    uint words;
    uint pad;
    device uint* source;
    device uint* destination;
};

kernel void texel_repack(constant RepackParams& params [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index >= params.words) return;
    const uint word = params.source[index];
    params.destination[index] = ((word >> 21u) & 0x7ffu) | (((word >> 10u) & 0x7ffu) << 11u) | ((word & 0x3ffu) << 22u);
}

kernel void cmask_clear(constant CmaskParams& params [[buffer(0)]], uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= params.width || pixel.y >= params.height) return;
    const uint x = pixel.x & ~7u;
    const uint y = pixel.y & ~7u;
    uint offset = 0u;
    for (uint bit = 0u; bit < 13u; ++bit) offset |= (popcount((x & xMasks[bit]) ^ (y & yMasks[bit])) & 1u) << bit;
    const uint nibble = ((y / 512u) * params.blocksPerRow + x / 1024u) * 8192u + offset;
    const uint byteAddress = params.base + nibble / 2u;
    const uint code = (params.cmask[byteAddress / 4u] >> ((byteAddress % 4u) * 8u + (nibble % 2u) * 4u)) & 0xfu;
    if (code != 0u) return;
    if (params.write != 0u) {
        params.target.write(uint4(params.clear0, params.clear1, 0u, 0u), pixel);
        return;
    }
    if ((pixel.x & 7u) == 0u && (pixel.y & 7u) == 0u) atomic_fetch_or_explicit(params.errors, 1u, memory_order_relaxed);
}
)MSL";

struct Push {
    std::uint32_t srcBase;
    std::uint32_t dstBase;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pitchBytes;
    std::uint32_t blocksPerRow;
    std::uint32_t tail;
    std::uint32_t tailX;
    std::uint32_t tailY;
    std::uint32_t elementBytes;
    std::uint32_t slice;
    std::uint32_t rangeBegin;
    std::uint32_t rangeEnd;
    std::uint32_t tiledBase;
    std::uint32_t linearBase;
    std::uint32_t columnBegin;
    std::uint32_t rowBegin;
    std::uint32_t pipeBankXor;
};

struct BufferParams {
    Push push;
    std::uint64_t input;
    std::uint64_t output;
};
static_assert(sizeof(Push) == 72 && offsetof(BufferParams, input) == 72);

struct ImageParams {
    Push push;
    std::uint64_t tiled;
    std::uint64_t image;
};

struct RepackParams {
    std::uint32_t words;
    std::uint32_t pad;
    std::uint64_t source;
    std::uint64_t destination;
};
static_assert(sizeof(RepackParams) == 24 && offsetof(RepackParams, source) == 8);

struct CmaskParams {
    std::uint32_t base;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t blocksPerRow;
    std::uint32_t clear0;
    std::uint32_t clear1;
    std::uint32_t write;
    std::uint32_t pad;
    std::uint64_t cmask;
    std::uint64_t errors;
    std::uint64_t target;
};

std::uint32_t BlockBytesFor(TextureTileMode tileMode) {
    switch (tileMode) {
    case TextureTileMode::kLinear: return 0u;
    case TextureTileMode::kStandard256B:
    case TextureTileMode::kD256B: return 256u;
    case TextureTileMode::kStandard4KB:
    case TextureTileMode::kD4KB:
    case TextureTileMode::kS4KBX:
    case TextureTileMode::kD4KBX: return 4096u;
    case TextureTileMode::kStandard64KB:
    case TextureTileMode::kD64KB:
    case TextureTileMode::kS64KBT:
    case TextureTileMode::kD64KBT:
    case TextureTileMode::kZ64KBX:
    case TextureTileMode::kS64KBX:
    case TextureTileMode::kD64KBX:
    case TextureTileMode::kR64KBX: return 65536u;
    }
    Fail("unknown tile mode");
}

std::uint32_t PipelineKey(TextureTileMode tileMode, std::uint32_t elementBytes, bool retile, bool thick, bool image) {
    return (image ? 1u << 18 : 0u) | (thick ? 1u << 17 : 0u) | (retile ? 1u << 16 : 0u) | (static_cast<std::uint32_t>(tileMode) << 8) | elementBytes;
}

id<MTLDevice> DeviceFor(const Context& context) {
    id<MTLDevice> device = Metal::DeviceOf(context);
    if (device == nil) Fail("the detiler needs a Metal device context");
    return device;
}

id<MTLLibrary> LibraryOf(VkShaderModule module) {
    return (__bridge id<MTLLibrary>)reinterpret_cast<void*>(module);
}

id<MTLComputePipelineState> PipelineOf(VkPipeline pipeline) {
    return (__bridge id<MTLComputePipelineState>)reinterpret_cast<void*>(pipeline);
}

std::uint64_t ImageResource(VkImageView view) {
    return Metal::TextureOfView(view).gpuResourceID._impl;
}

}

TextureDetiler::TextureDetiler(const Context& context) : context(context) {}

TextureDetiler::~TextureDetiler() {
    release();
}

void TextureDetiler::release() noexcept {
    for (const auto& entry : pipelines) {
        if (entry.second != VK_NULL_HANDLE) CFRelease(reinterpret_cast<CFTypeRef>(entry.second));
    }
    pipelines.clear();
    if (cmaskPipeline != VK_NULL_HANDLE) CFRelease(reinterpret_cast<CFTypeRef>(cmaskPipeline));
    cmaskPipeline = VK_NULL_HANDLE;
    if (module != VK_NULL_HANDLE) CFRelease(reinterpret_cast<CFTypeRef>(module));
    module = VK_NULL_HANDLE;
    cmaskErrors.reset();
    if (repackPipeline != VK_NULL_HANDLE) CFRelease(reinterpret_cast<CFTypeRef>(repackPipeline));
    repackPipeline = VK_NULL_HANDLE;
}

VkFormat TextureDetiler::ImageElementFormat(std::uint32_t elementBytes) {
    switch (elementBytes) {
    case 1: return VK_FORMAT_R8_UINT;
    case 2: return VK_FORMAT_R16_UINT;
    case 4: return VK_FORMAT_R32_UINT;
    case 8: return VK_FORMAT_R32G32_UINT;
    case 16: return VK_FORMAT_R32G32B32A32_UINT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

void TextureDetiler::BeginBatch() {}

VkDescriptorSet TextureDetiler::allocateSet(bool) {
    return VK_NULL_HANDLE;
}

VkPipeline TextureDetiler::pipeline(TextureTileMode tileMode, std::uint32_t elementBytes, bool retile, bool thick, bool image) {
    if (!(std::has_single_bit(elementBytes) && elementBytes <= 16u)) Fail("unsupported element size for texture detiling");
    const auto key = PipelineKey(tileMode, elementBytes, retile, thick, image);
    for (const auto& entry : pipelines) {
        if (entry.first == key) return entry.second;
    }
    id<MTLDevice> device = DeviceFor(context);
    if (module == VK_NULL_HANDLE) {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_1;
        options.mathMode = MTLMathModeSafe;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(DetileSource) options:options error:&error];
        if (library == nil) Fail("the detile kernels do not compile: " + Metal::Describe(error));
        module = reinterpret_cast<VkShaderModule>(const_cast<void*>(CFBridgingRetain(library)));
    }
    std::array<std::uint32_t, 22> values{elementBytes, BlockBytesFor(tileMode), tileMode == TextureTileMode::kLinear ? 0u : 1u, retile ? 1u : 0u};
    if (thick) {
        const auto thickMode = tileMode == TextureTileMode::kStandard4KB ? 0x105u : tileMode == TextureTileMode::kS4KBX ? 0x115u : tileMode == TextureTileMode::kStandard64KB ? 0x109u : tileMode == TextureTileMode::kS64KBX ? 0x119u : tileMode == TextureTileMode::kD64KBX ? 0x11au : 0u;
        if (thickMode == 0u) Fail("3D textures are only detiled thick from SW_4KB_S, SW_4KB_S_X, SW_64KB_S, SW_64KB_S_X or SW_64KB_D_X");
        const auto* equation = FindTextureSwizzleEquation(thickMode, elementBytes);
        if (equation == nullptr) Fail("no thick swizzle equation for the element size");
        values[2] = 2u;
        std::copy(equation->bits.begin(), equation->bits.end(), values.begin() + 4);
        const auto extent = ThickBlockExtent(tileMode, elementBytes);
        values[20] = extent[0];
        values[21] = extent[1];
    } else if (const auto mode = EquationSwizzleMode(tileMode); mode != 0) {
        const auto* equation = FindTextureSwizzleEquation(mode, elementBytes);
        if (equation == nullptr) Fail("no swizzle equation for tile mode " + std::to_string(mode) + " at " + std::to_string(elementBytes) + " bytes per element");
        values[2] = 2u;
        std::copy(equation->bits.begin(), equation->bits.end(), values.begin() + 4);
    }
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    for (std::uint32_t index = 0; index < values.size(); ++index) [constants setConstantValue:&values[index] type:MTLDataTypeUInt atIndex:index];
    NSError* error = nil;
    id<MTLFunction> function = [LibraryOf(module) newFunctionWithName:(image ? @"detile_image" : @"detile") constantValues:constants error:&error];
    if (function == nil) Fail("cannot specialize the detile kernel: " + Metal::Describe(error));
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
    if (state == nil) Fail("cannot build the detile pipeline: " + Metal::Describe(error));
    const auto handle = reinterpret_cast<VkPipeline>(const_cast<void*>(CFBridgingRetain(state)));
    pipelines.emplace_back(key, handle);
    return handle;
}

void TextureDetiler::checkWindow(const TileMipLayout& layout, const DetileWindow& window, std::uint32_t& columnEnd, std::uint32_t& rowEnd) const {
    Require(layout.width != 0 && layout.height != 0, "texture detiling requires a non-empty mip layout");
    Require(layout.tiledSize != 0 && layout.linearSize != 0, "texture detiling requires a non-empty mip layout");
    columnEnd = window.columnEnd != 0 ? std::min(window.columnEnd, layout.width) : layout.width;
    rowEnd = window.rowEnd != 0 ? std::min(window.rowEnd, layout.height) : layout.height;
    if (!(window.columnBegin < columnEnd && window.rowBegin < rowEnd)) Require(false, "texture detiling window lies outside the mip: columns " + std::to_string(window.columnBegin) + ".." + std::to_string(columnEnd) + ", rows " + std::to_string(window.rowBegin) + ".." + std::to_string(rowEnd) + " of " + std::to_string(layout.width) + "x" + std::to_string(layout.height));
}

namespace {

Push MakePush(const TileMipLayout& layout, std::uint32_t elementBytes, std::uint32_t slice, const DetileWindow& window) {
    Push push{};
    push.width = layout.width;
    push.height = layout.height;
    push.pitchBytes = layout.pitchBytes;
    push.blocksPerRow = layout.blocksPerRow;
    push.tail = layout.tail ? 1u : 0u;
    push.tailX = layout.tailX;
    push.tailY = layout.tailY;
    push.elementBytes = elementBytes;
    push.slice = slice;
    push.rangeBegin = window.rangeBegin;
    push.rangeEnd = window.rangeEnd;
    push.tiledBase = window.tiledBase;
    push.linearBase = window.linearBase;
    push.columnBegin = window.columnBegin;
    push.rowBegin = window.rowBegin;
    push.pipeBankXor = window.pipeBankXor;
    return push;
}

}

void TextureDetiler::Dispatch(VkCommandBuffer commands, TextureTileMode tileMode, std::uint32_t elementBytes, VkBuffer source, std::uint64_t sourceOffset, VkBuffer destination, std::uint64_t destinationOffset, const TileMipLayout& layout, bool retile, std::uint32_t slice, bool thick, const DetileWindow& window) {
    Require(commands != VK_NULL_HANDLE, "texture detiling requires an active command buffer");
    Require(window.rangeBegin < window.rangeEnd && window.tiledBase <= window.rangeBegin, "texture detiling window is empty");
    Require(source != VK_NULL_HANDLE && destination != VK_NULL_HANDLE, "texture detiling requires source and destination buffers");
    std::uint32_t columnEnd = 0, rowEnd = 0;
    checkWindow(layout, window, columnEnd, rowEnd);
    const auto target = PipelineOf(pipeline(tileMode, elementBytes, retile, thick));
    // The kernel addresses dword-aligned bases (as the Vulkan descriptor offsets) plus the remainder.
    const auto sourceBase = sourceOffset % 4u;
    const auto destinationBase = destinationOffset % 4u;
    BufferParams params{};
    params.push = MakePush(layout, elementBytes, slice, window);
    params.push.srcBase = static_cast<std::uint32_t>(sourceBase);
    params.push.dstBase = static_cast<std::uint32_t>(destinationBase);
    params.input = Metal::BufferOf(source).gpuAddress + (sourceOffset - sourceBase);
    params.output = Metal::BufferOf(destination).gpuAddress + (destinationOffset - destinationBase);
    auto& stream = Metal::Stream(commands);
    stream.DispatchHelper(target, std::as_bytes(std::span(&params, 1)), MTLSizeMake((columnEnd - window.columnBegin + 7u) / 8u, (rowEnd - window.rowBegin + 7u) / 8u, 1), MTLSizeMake(8, 8, 1));
}

void TextureDetiler::DispatchImage(VkCommandBuffer commands, TextureTileMode tileMode, std::uint32_t elementBytes, VkBuffer tiled, std::uint64_t tiledOffset, VkImageView view, const TileMipLayout& layout, bool retile, std::uint32_t slice, const DetileWindow& window) {
    Require(commands != VK_NULL_HANDLE, "texture detiling requires an active command buffer");
    Require(window.rangeBegin < window.rangeEnd && window.tiledBase <= window.rangeBegin, "texture detiling window is empty");
    Require(tiled != VK_NULL_HANDLE && view != VK_NULL_HANDLE, "texture detiling requires a tiled buffer and an image view");
    Require(ImageElementFormat(elementBytes) != VK_FORMAT_UNDEFINED, "unsupported element size for texture detiling through an image");
    std::uint32_t columnEnd = 0, rowEnd = 0;
    checkWindow(layout, window, columnEnd, rowEnd);
    const auto target = PipelineOf(pipeline(tileMode, elementBytes, retile, false, true));
    const auto base = tiledOffset % 4u;
    ImageParams params{};
    params.push = MakePush(layout, elementBytes, slice, window);
    params.push.srcBase = static_cast<std::uint32_t>(retile ? 0 : base);
    params.push.dstBase = static_cast<std::uint32_t>(retile ? base : 0);
    params.push.linearBase = 0;
    params.tiled = Metal::BufferOf(tiled).gpuAddress + (tiledOffset - base);
    params.image = ImageResource(view);
    auto& stream = Metal::Stream(commands);
    stream.DispatchHelper(target, std::as_bytes(std::span(&params, 1)), MTLSizeMake((columnEnd - window.columnBegin + 7u) / 8u, (rowEnd - window.rowBegin + 7u) / 8u, 1), MTLSizeMake(8, 8, 1));
}

void TextureDetiler::DispatchRepack(VkCommandBuffer commands, VkBuffer source, std::uint64_t sourceOffset, VkBuffer destination, std::uint64_t destinationOffset, std::uint64_t bytes) {
    Require(commands != VK_NULL_HANDLE && source != VK_NULL_HANDLE && destination != VK_NULL_HANDLE, "a texel repack needs a command buffer and its buffers");
    Require(bytes != 0 && bytes % 4u == 0 && sourceOffset % 4u == 0 && destinationOffset % 4u == 0, "a texel repack needs whole aligned words");
    const auto words = bytes / 4u;
    Require(words <= std::uint64_t{0xffffffffu} - 63u, "a texel repack exceeds the device's dispatch limits");
    if (repackPipeline == VK_NULL_HANDLE) {
        pipeline(TextureTileMode::kLinear, 4u, false, false);  // compiles the library
        NSError* error = nil;
        id<MTLFunction> function = [LibraryOf(module) newFunctionWithName:@"texel_repack"];
        id<MTLComputePipelineState> state = function == nil ? nil : [DeviceFor(context) newComputePipelineStateWithFunction:function error:&error];
        if (state == nil) Fail("cannot build the texel repack pipeline: " + Metal::Describe(error));
        repackPipeline = reinterpret_cast<VkPipeline>(const_cast<void*>(CFBridgingRetain(state)));
    }
    RepackParams params{};
    params.words = static_cast<std::uint32_t>(words);
    params.source = Metal::BufferOf(source).gpuAddress + sourceOffset;
    params.destination = Metal::BufferOf(destination).gpuAddress + destinationOffset;
    auto& stream = Metal::Stream(commands);
    stream.DispatchHelper(PipelineOf(repackPipeline), std::as_bytes(std::span(&params, 1)), MTLSizeMake((words + 63u) / 64u, 1, 1), MTLSizeMake(64, 1, 1));
}

std::uint32_t TextureDetiler::CmaskErrors() {
    if (cmaskErrors == nullptr) return 0;
    std::uint32_t codes = 0;
    std::memcpy(&codes, cmaskErrors->Bytes().data(), sizeof(codes));
    return codes;
}

void TextureDetiler::DispatchCmaskClear(VkCommandBuffer commands, VkBuffer cmask, std::uint64_t cmaskOffset, std::size_t cmaskBytes, VkImageView view, std::uint32_t width, std::uint32_t height, std::uint32_t elementBytes, const std::array<std::uint32_t, 2>& clearWords, bool write) {
    Require(commands != VK_NULL_HANDLE && cmask != VK_NULL_HANDLE && view != VK_NULL_HANDLE, "a CMASK clear needs a command buffer, the CMASK buffer and the target view");
    Require(elementBytes == 1 || elementBytes == 2 || elementBytes == 4 || elementBytes == 8, "a CMASK clear of this element size is not modeled");
    Require(cmaskOffset % 4 == 0 && cmaskBytes % 4 == 0 && cmaskBytes != 0, "a CMASK clear needs a word-aligned CMASK");
    id<MTLDevice> device = DeviceFor(context);
    if (cmaskPipeline == VK_NULL_HANDLE) {
        pipeline(TextureTileMode::kLinear, 4u, false, false);  // compiles the library
        NSError* error = nil;
        id<MTLFunction> function = [LibraryOf(module) newFunctionWithName:@"cmask_clear"];
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
        if (state == nil) Fail("cannot build the CMASK clear pipeline: " + Metal::Describe(error));
        cmaskPipeline = reinterpret_cast<VkPipeline>(const_cast<void*>(CFBridgingRetain(state)));
        cmaskErrors = std::make_unique<Buffer>(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memset(cmaskErrors->Bytes().data(), 0, 4);
    }
    auto& stream = Metal::Stream(commands);
    stream.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const std::uint32_t mask = elementBytes >= 4 ? 0xffffffffu : (1u << (elementBytes * 8u)) - 1u;
    CmaskParams params{};
    params.base = 0;
    params.width = width;
    params.height = height;
    params.blocksPerRow = (width + 1023u) / 1024u;
    params.clear0 = clearWords[0] & mask;
    params.clear1 = elementBytes == 8 ? clearWords[1] : 0u;
    params.write = write ? 1u : 0u;
    params.cmask = Metal::BufferOf(cmask).gpuAddress + cmaskOffset;
    params.errors = Metal::BufferOf(cmaskErrors->Handle()).gpuAddress;
    params.target = ImageResource(view);
    stream.DispatchHelper(PipelineOf(cmaskPipeline), std::as_bytes(std::span(&params, 1)), MTLSizeMake((width + 7u) / 8u, (height + 7u) / 8u, 1), MTLSizeMake(8, 8, 1));
    stream.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    stream.FillBuffer(Metal::BufferOf(cmask), cmaskOffset, cmaskBytes, 0xffffffffu);
    stream.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}

}
