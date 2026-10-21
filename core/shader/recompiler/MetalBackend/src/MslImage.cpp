// Work package B4 (docs/mac/M2_DESIGN.md §9): image and sampler instructions in MSL. The semantics
// follow SpirvBackend/src/SpirvImageEmitter.cpp (the hardware-measured reference): runtime image
// modes selected by a function constant, dst_sel swizzles on converted texels, dmask/d16 result
// packing, packed and BY2/BY4 loads, sRGB decode, cube faces as 2D array layers, 1D images as 2D
// textures one texel high (MslEntryPoint.cpp), 32-bit image atomics on R32Uint textures.
#include "MetalBackend/MslEmitterState.hpp"
#include "MslBufferShared.hpp"
#include "BdaAbi.hpp"
#include "ImageTableAbi.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "PipelineSpecialization.hpp"
#include "RuntimeAbi.hpp"
#include <cstddef>
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "RdnaDecoder/RdnaImageOpDecoder.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include "SpirvBackend/SpirvMemory/SpirvDescriptors.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ShaderRecompiler {

namespace {

constexpr std::array<MslPreludeChunk, 3> OnBufferFormat{MslPreludeChunk::Common, MslPreludeChunk::RuntimeBuffer, MslPreludeChunk::BufferFormat};

// MSL helpers of the Image chunk. aps5_srgb_decode: the SPIR-V path's 256-entry LUT of an sRGB
// code (unorm8 of the view value) to linear.
constexpr std::string_view ImageText = R"MSL(
constant uint aps5_srgb_to_linear[256] = {
    0x00000000u, 0x399f0000u, 0x3a1f0000u, 0x3a6f0000u, 0x3a9f0000u, 0x3ac70000u, 0x3aef0000u, 0x3b0b0000u,
    0x3b1f0000u, 0x3b330000u, 0x3b470000u, 0x3b5b0000u, 0x3b710000u, 0x3b840000u, 0x3b900000u, 0x3b9d0000u,
    0x3baa0000u, 0x3bb80000u, 0x3bc60000u, 0x3bd50000u, 0x3be50000u, 0x3bf60000u, 0x3c030000u, 0x3c0c0000u,
    0x3c160000u, 0x3c1f0000u, 0x3c290000u, 0x3c340000u, 0x3c3e0000u, 0x3c490000u, 0x3c550000u, 0x3c600000u,
    0x3c6d0000u, 0x3c790000u, 0x3c830000u, 0x3c8a0000u, 0x3c910000u, 0x3c980000u, 0x3c9f0000u, 0x3ca60000u,
    0x3cae0000u, 0x3cb60000u, 0x3cbe0000u, 0x3cc60000u, 0x3cce0000u, 0x3cd70000u, 0x3ce00000u, 0x3ce90000u,
    0x3cf20000u, 0x3cfc0000u, 0x3d030000u, 0x3d080000u, 0x3d0d0000u, 0x3d120000u, 0x3d170000u, 0x3d1c0000u,
    0x3d220000u, 0x3d280000u, 0x3d2d0000u, 0x3d330000u, 0x3d390000u, 0x3d3f0000u, 0x3d450000u, 0x3d4c0000u,
    0x3d520000u, 0x3d590000u, 0x3d5f0000u, 0x3d660000u, 0x3d6d0000u, 0x3d740000u, 0x3d7b0000u, 0x3d810000u,
    0x3d850000u, 0x3d880000u, 0x3d8c0000u, 0x3d900000u, 0x3d940000u, 0x3d980000u, 0x3d9c0000u, 0x3da00000u,
    0x3da40000u, 0x3da90000u, 0x3dad0000u, 0x3db10000u, 0x3db60000u, 0x3dba0000u, 0x3dbf0000u, 0x3dc30000u,
    0x3dc80000u, 0x3dcd0000u, 0x3dd10000u, 0x3dd60000u, 0x3ddb0000u, 0x3de00000u, 0x3de50000u, 0x3dea0000u,
    0x3df00000u, 0x3df50000u, 0x3dfa0000u, 0x3e000000u, 0x3e020000u, 0x3e050000u, 0x3e080000u, 0x3e0b0000u,
    0x3e0e0000u, 0x3e110000u, 0x3e140000u, 0x3e170000u, 0x3e1a0000u, 0x3e1d0000u, 0x3e200000u, 0x3e230000u,
    0x3e260000u, 0x3e290000u, 0x3e2c0000u, 0x3e300000u, 0x3e330000u, 0x3e360000u, 0x3e3a0000u, 0x3e3d0000u,
    0x3e400000u, 0x3e440000u, 0x3e470000u, 0x3e4b0000u, 0x3e4e0000u, 0x3e520000u, 0x3e560000u, 0x3e590000u,
    0x3e5d0000u, 0x3e610000u, 0x3e650000u, 0x3e680000u, 0x3e6c0000u, 0x3e700000u, 0x3e740000u, 0x3e780000u,
    0x3e7c0000u, 0x3e800000u, 0x3e820000u, 0x3e840000u, 0x3e860000u, 0x3e880000u, 0x3e8a0000u, 0x3e8d0000u,
    0x3e8f0000u, 0x3e910000u, 0x3e930000u, 0x3e950000u, 0x3e980000u, 0x3e9a0000u, 0x3e9c0000u, 0x3e9e0000u,
    0x3ea10000u, 0x3ea30000u, 0x3ea50000u, 0x3ea80000u, 0x3eaa0000u, 0x3ead0000u, 0x3eaf0000u, 0x3eb20000u,
    0x3eb40000u, 0x3eb60000u, 0x3eb90000u, 0x3ebc0000u, 0x3ebe0000u, 0x3ec10000u, 0x3ec30000u, 0x3ec60000u,
    0x3ec80000u, 0x3ecb0000u, 0x3ece0000u, 0x3ed10000u, 0x3ed30000u, 0x3ed60000u, 0x3ed90000u, 0x3edb0000u,
    0x3ede0000u, 0x3ee10000u, 0x3ee40000u, 0x3ee70000u, 0x3eea0000u, 0x3eed0000u, 0x3ef00000u, 0x3ef20000u,
    0x3ef50000u, 0x3ef80000u, 0x3efb0000u, 0x3efe0000u, 0x3f010000u, 0x3f020000u, 0x3f040000u, 0x3f050000u,
    0x3f070000u, 0x3f090000u, 0x3f0a0000u, 0x3f0c0000u, 0x3f0d0000u, 0x3f0f0000u, 0x3f110000u, 0x3f120000u,
    0x3f140000u, 0x3f160000u, 0x3f170000u, 0x3f190000u, 0x3f1b0000u, 0x3f1c0000u, 0x3f1e0000u, 0x3f200000u,
    0x3f210000u, 0x3f230000u, 0x3f250000u, 0x3f270000u, 0x3f290000u, 0x3f2a0000u, 0x3f2c0000u, 0x3f2e0000u,
    0x3f300000u, 0x3f320000u, 0x3f330000u, 0x3f350000u, 0x3f370000u, 0x3f390000u, 0x3f3b0000u, 0x3f3d0000u,
    0x3f3f0000u, 0x3f410000u, 0x3f430000u, 0x3f450000u, 0x3f470000u, 0x3f490000u, 0x3f4b0000u, 0x3f4d0000u,
    0x3f4f0000u, 0x3f510000u, 0x3f530000u, 0x3f550000u, 0x3f570000u, 0x3f590000u, 0x3f5b0000u, 0x3f5d0000u,
    0x3f5f0000u, 0x3f610000u, 0x3f630000u, 0x3f650000u, 0x3f680000u, 0x3f6a0000u, 0x3f6c0000u, 0x3f6e0000u,
    0x3f700000u, 0x3f730000u, 0x3f750000u, 0x3f770000u, 0x3f790000u, 0x3f7b0000u, 0x3f7e0000u, 0x3f800000u,
};

static inline float4 aps5_srgb_decode(float4 color) {
    float4 result;
    for (uint index = 0u; index < 4u; ++index) {
        const uint code = min(uint(color[index] * 255.0f + 0.5f), 255u);
        result[index] = as_type<float>(aps5_srgb_to_linear[code]);
    }
    return result;
}

// RuntimeSwizzledComponent: dst_sel 0 → 0, 1 → `one`, 4..7 → component sel-4.
static inline uint aps5_swizzle_select(uint selector, uint4 components, uint one) {
    if (selector == 1u) return one;
    if (selector >= 4u && selector <= 7u) return components[selector - 4u];
    return 0u;
}

static inline uint aps5_f16_bits(float value) {
    return uint(as_type<ushort>(half(value)));
}

static inline float aps5_f16_value(uint bits) {
    return float(as_type<half>(ushort(bits & 0xffffu)));
}

static inline uint aps5_pack_half2(float low, float high) {
    return aps5_f16_bits(low) | (aps5_f16_bits(high) << 16u);
}

// A float array layer as Vulkan picks it: round to nearest even, clamped to the layers.
static inline uint aps5_layer(float layer, uint layers) {
    return uint(clamp(rint(layer), 0.0f, float(max(layers, 1u) - 1u)));
}

// CubeLayer: the guest face/layer value (face + 8 * cube) as a 2D array layer (face + 6 * cube).
static inline float aps5_cube_layer(float value) {
    const uint guest = uint(value);
    return float(guest - ((guest >> 3u) << 1u));
}
// Emulated depth comparison of a color texture (SpirvImageEmitter.cpp EmitEmulatedCompareSample):
// p = (function, linear, clamp x, clamp y), q = (reference mode, border white).
static inline int aps5_ec_address(int index, int extent, uint mode) {
    if (mode == 0u) {
        const int remainder = index % extent;
        return remainder < 0 ? remainder + extent : remainder;
    }
    return clamp(index, 0, extent - 1);
}

static inline float4 aps5_ec_fetch(texture2d<float> t, uint x, uint y, uint layer) {
    return t.read(uint2(x, y), 0u);
}

static inline float4 aps5_ec_fetch(texture2d_array<float> t, uint x, uint y, uint layer) {
    return t.read(uint2(x, y), layer, 0u);
}

template <typename T>
static inline float aps5_ec_texel(T t, int x, int y, uint layer, int width, int height, uint4 p, uint2 q, uint4 swizzle, float reference) {
    const float4 texel = aps5_ec_fetch(t, uint(aps5_ec_address(x, width, p.z)), uint(aps5_ec_address(y, height, p.w)), layer);
    float red = texel.w;
    for (uint channel = 3u; channel-- > 0u;) {
        if (swizzle[channel] == 4u) red = texel[channel];
    }
    const float border = q.y == 1u ? 1.0f : 0.0f;
    if (p.z == 2u && !(x >= 0 && x < width)) red = border;
    if (p.w == 2u && !(y >= 0 && y < height)) red = border;
    float result = p.x == 7u ? 1.0f : 0.0f;
    switch (p.x) {
    case 1u: result = reference < red ? 1.0f : 0.0f; break;
    case 2u: result = reference == red ? 1.0f : 0.0f; break;
    case 3u: result = reference <= red ? 1.0f : 0.0f; break;
    case 4u: result = reference > red ? 1.0f : 0.0f; break;
    case 5u: result = (reference < red || reference > red) ? 1.0f : 0.0f; break;
    case 6u: result = reference >= red ? 1.0f : 0.0f; break;
    default: break;
    }
    return q.x == 3u ? 0.0f : result;
}

template <typename T>
static inline float4 aps5_ec_sample(T t, float2 coord, uint layer, int2 offset, uint4 p, uint2 q, uint4 swizzle, float reference) {
    if (q.x == 1u) reference = clamp(reference, 0.0f, 1.0f);
    if (q.x == 2u) reference = clamp(reference, -1.0f, 1.0f);
    const int width = int(t.get_width(0));
    const int height = int(t.get_height(0));
    const float u = coord.x * float(width);
    const float v = coord.y * float(height);
    float result;
    if (p.y == 0u) {
        result = aps5_ec_texel(t, int(floor(u)) + offset.x, int(floor(v)) + offset.y, layer, width, height, p, q, swizzle, reference);
    } else {
        const float cu = u - 0.5f;
        const float cv = v - 0.5f;
        const float fu = floor(cu);
        const float fv = floor(cv);
        const float wu = cu - fu;
        const float wv = cv - fv;
        const int x0 = int(fu) + offset.x;
        const int y0 = int(fv) + offset.y;
        const float top = mix(aps5_ec_texel(t, x0, y0, layer, width, height, p, q, swizzle, reference), aps5_ec_texel(t, x0 + 1, y0, layer, width, height, p, q, swizzle, reference), wu);
        const float bottom = mix(aps5_ec_texel(t, x0, y0 + 1, layer, width, height, p, q, swizzle, reference), aps5_ec_texel(t, x0 + 1, y0 + 1, layer, width, height, p, q, swizzle, reference), wu);
        result = mix(top, bottom, wv);
    }
    float4 value;
    for (uint channel = 0u; channel < 4u; ++channel) value[channel] = swizzle[channel] == 0u ? 0.0f : (swizzle[channel] == 1u ? 1.0f : result);
    return value;
}
// Min/max sampler reduction (Metal ignores MTLSamplerDescriptor.reductionMode): the 2x2 bilinear
// footprint of the selected level read through the sampler at texel centres (exact for every
// filter and address mode), reduced per component. `packed`/`filters` are the
// PipelineSpecialization::ReductionBase words; linear mip filtering with a reduction is refused by
// the sampler decoder.
static inline float4 aps5_rs_fetch(texture2d<float> t, sampler s, float2 c, uint slice, uint l) {
    return t.sample(s, c, level(float(l)));
}

static inline float4 aps5_rs_fetch(texture2d_array<float> t, sampler s, float2 c, uint slice, uint l) {
    return t.sample(s, c, slice, level(float(l)));
}

template <typename T>
static float4 aps5_reduce_sample(T t, sampler s, float2 coord, uint slice, float lod, int2 offset, uint packed, uint filters) {
    const uint reduction = packed & 3u;
    const uint mip = (packed >> 2u) & 3u;
    float lambda = 0.0f;
    if (mip != 0u) {
        const float bias = float(int(((filters & 0x3fffu) ^ 0x2000u) - 0x2000u)) / 256.0f;
        lambda = clamp(lod + bias, float((packed >> 4u) & 0xfffu) / 256.0f, float((packed >> 16u) & 0xfffu) / 256.0f);
    }
    const uint levels = max(t.get_num_mip_levels(), 1u);
    const uint level = mip == 0u ? 0u : min(uint(max(ceil(lambda + 0.5f) - 1.0f, 0.0f)), levels - 1u);
    const uint filter = lambda > 0.0f ? (filters >> 18u) & 3u : (filters >> 16u) & 3u;
    if (filter == 0u) return aps5_rs_fetch(t, s, coord + float2(offset) / float2(float(t.get_width(level)), float(t.get_height(level))), slice, level);
    const float w = float(t.get_width(level));
    const float h = float(t.get_height(level));
    const float x0 = floor(coord.x * w - 0.5f) + float(offset.x);
    const float y0 = floor(coord.y * h - 0.5f) + float(offset.y);
    float4 result = aps5_rs_fetch(t, s, float2((x0 + 0.5f) / w, (y0 + 0.5f) / h), slice, level);
    for (uint corner = 1u; corner < 4u; ++corner) {
        const float4 texel = aps5_rs_fetch(t, s, float2((x0 + float(corner & 1u) + 0.5f) / w, (y0 + float(corner >> 1u) + 0.5f) / h), slice, level);
        result = reduction == 1u ? min(result, texel) : max(result, texel);
    }
    return result;
}
)MSL";

struct Access {
    const IrValue& inst;
    const MemoryInfo& mem;
    const ImageResource& image;
    const IrValue& address;
    // The mode whose heap binding holds the texture when `image` is a view of it with another
    // dimension (a remapped 1D depth-comparison image, OneDimensionalView); null: `image`.
    const ImageResource* binding = nullptr;
};

bool HasFlag(const MemoryInfo& mem, std::uint32_t flag) {
    return (mem.imageSampleFlags & flag) != 0u;
}

std::uint32_t EffectiveDmask(const MemoryInfo& mem) {
    return mem.dmask != 0u ? mem.dmask : 1u;
}

std::uint32_t DmaskComponentIndex(std::uint32_t dmask, std::uint32_t component) {
    std::uint32_t index = 0;
    for (std::uint32_t i = 0; i < component; i++) index += (dmask >> i) & 1u;
    return index;
}

std::uint32_t DmaskComponent(std::uint32_t dmask, std::uint32_t index) {
    for (std::uint32_t component = 0; component < 4u; component++) {
        if (((dmask >> component) & 1u) != 0u && index-- == 0u) return component;
    }
    throw std::runtime_error("MSL emission failed: image dmask has fewer components than requested");
}

std::uint32_t ImageGatherComponent(MslEmitterState& state, const IrValue& inst, std::uint32_t dmask) {
    switch (dmask) {
    case 0x1u: return 0;
    case 0x2u: return 1;
    case 0x4u: return 2;
    case 0x8u: return 3;
    default: state.Fail(inst, "image gather dmask must select exactly one component");
    }
}

std::string U(std::uint32_t value) {
    return std::to_string(value) + "u";
}

std::string F(float value) {
    char text[64];
    std::snprintf(text, sizeof(text), "as_type<float>(0x%08xu)", std::bit_cast<std::uint32_t>(value));
    return text;
}

const RdnaImageDimensionInfo& Dimension(RdnaImageDimension dimension) {
    return RdnaImageDimensionInfoFor(dimension);
}

const RdnaImageDimensionInfo& AddressDimension(const Access& access) {
    return Dimension(access.mem.imageDimension);
}

std::string ScalarType(IrTextureNumericClass numeric) {
    switch (numeric) {
    case IrTextureNumericClass::Float: return "float";
    case IrTextureNumericClass::Uint: return "uint";
    case IrTextureNumericClass::Sint: return "int";
    default: throw std::runtime_error("MSL emission failed: invalid image numeric class");
    }
}

std::string VectorType(IrTextureNumericClass numeric) {
    return ScalarType(numeric) + "4";
}

// The uint bits of a typed scalar.
std::string Bits(const std::string& value, IrTextureNumericClass numeric) {
    if (numeric == IrTextureNumericClass::Uint) return value;
    return "as_type<uint>(" + value + ")";
}

std::string AddressU32(MslEmitterState& state, const Access& access, std::uint32_t component) {
    const auto layout = GetRdnaImageAddressComponentLayout(access.mem.imageSampleFlags, component);
    const auto packed = layout.bitOffset / 32u;
    if (packed >= access.address.ArgumentCount()) state.Fail(access.inst, "has an image address component outside of the address operands");
    auto value = state.Value(access.address.Argument(packed));
    if (layout.bitWidth == 16u) {
        if ((layout.bitOffset & 31u) != 0u) value = "(" + value + " >> 16u)";
        value = "(" + value + " & 0xffffu)";
    }
    return value;
}

std::string AddressF32(MslEmitterState& state, const Access& access, std::uint32_t component) {
    const auto value = AddressU32(state, access, component);
    if (GetRdnaImageAddressComponentLayout(access.mem.imageSampleFlags, component).bitWidth == 16u) return "aps5_f16_value(" + value + ")";
    return "as_type<float>(" + value + ")";
}

ImageSampleLayout Layout(const MemoryInfo& mem) {
    ImageSampleLayout layout;
    std::uint32_t cursor = 0;
    const auto& info = Dimension(mem.imageDimension);
    if (HasFlag(mem, RdnaImageSampleFlagOffset)) layout.offset = cursor++;
    if (HasFlag(mem, RdnaImageSampleFlagBias)) layout.bias = cursor++;
    if (HasFlag(mem, RdnaImageSampleFlagCompare)) layout.dref = cursor++;
    if (HasFlag(mem, RdnaImageSampleFlagDerivative)) {
        layout.gradX = cursor;
        cursor += info.spatialComponents;
        layout.gradY = cursor;
        cursor += info.spatialComponents;
    }
    layout.coord = cursor;
    cursor += info.coordinateComponents;
    if (HasFlag(mem, RdnaImageSampleFlagLod)) layout.lod = cursor++;
    if (HasFlag(mem, RdnaImageSampleFlagLodClamp)) layout.clamp = cursor++;
    return layout;
}

// Element `element` of an image heap or the sampler heap. The prepared binding plan compacts the
// elements a snapshot binds and supplies their positions as DescriptorIndex specialization
// constants (default: the emission-time element), as DescriptorElementPointer on the SPIR-V path.
std::string HeapElement(MslEmitterState& state, DescriptorBindingKind kind, std::uint32_t element) {
    const auto checked = state.TableResource(kind, element);
    static_cast<void>(checked);
    const auto index = state.FunctionConstant(PipelineSpecialization::DescriptorIndex(static_cast<std::uint32_t>(kind), element), element);
    return "aps5_table." + MslTableField(kind) + "[" + index + "]";
}

// The texture (heap element) a mode's access reads or writes. A table mode (MslAbi Version 15)
// reads element tableAccess.element of the image table's array, typed as the mode's heap binding;
// a storage table mode (Version 18, upstream 603dc01db) the same element of the storage table's.
std::string Texture(MslEmitterState& state, const Access& access, std::uint32_t elementOffset = 0u) {
    const auto& bound = access.binding != nullptr ? *access.binding : access.image;
    const auto kind = DescriptorBindingForImage(bound);
    if (access.image.table != NoTable) {
        if (!state.tableAccess.active) state.Fail(access.inst, "reads an image table texture outside a table access");
        const auto binding = static_cast<std::uint32_t>(kind);
        if (access.image.resourceClass == ImageResourceClass::Storage) {
            // Float (29-35) and uint (36-42) storage views; atomic modes are never table modes.
            constexpr auto storageEnd = RuntimeAbi::FirstStorageImageBinding + 14u;
            if (elementOffset != 0u || access.image.mipMode == ImageMipMode::DynamicStorage || binding < RuntimeAbi::FirstStorageImageBinding || binding >= storageEnd || !TableModeEmitted(bound)) state.Fail(access.inst, "selects an image table storage mode that is not a non-atomic, single-sampled, static-mip storage view (an image table store to a multisampled image or through dynamic mips is not implemented)");
            static_cast<void>(state.TableResource(DescriptorBindingKind::ImageStorageTable, 0u));
            return "aps5_table." + MslTableField(DescriptorBindingKind::ImageStorageTable) + ".k" + std::to_string(binding) + ".address[" + state.tableAccess.element + "].value";
        }
        if (elementOffset != 0u || access.image.resourceClass != ImageResourceClass::Sampled || binding < RuntimeAbi::FirstImageBinding || binding >= RuntimeAbi::FirstStorageImageBinding || !TableModeEmitted(bound)) state.Fail(access.inst, "selects an image table mode that is not a sampled, single-sampled heap view");
        static_cast<void>(state.TableResource(DescriptorBindingKind::ImageTable, 0u));
        return "aps5_table." + MslTableField(DescriptorBindingKind::ImageTable) + ".k" + std::to_string(binding) + ".address[" + state.tableAccess.element + "].value";
    }
    return HeapElement(state, kind, state.DescriptorElement(kind, access.mem.resource) + elementOffset);
}

std::string Sampler(MslEmitterState& state, const Access& access) {
    // LoadSamplerDescriptor: element 2s+1 is the point-filtered twin for integer, converted and
    // depth-bit reads (TablePointSampled, which adds uint, when the image or the sampler is a table).
    const auto& samplerResource = state.program.Info().samplers.at(access.mem.sampler);
    const bool table = samplerResource.table != NoTable || access.image.table != NoTable;
    const bool point = table ? TablePointSampled(access.image) : access.image.numericClass == IrTextureNumericClass::Sint || access.image.conversionFormat != IrBufferFormat::Invalid || access.image.depthBits;
    if (samplerResource.table != NoTable) {
        // The sampler table entry's four elements: plain, point, compare, compare point.
        if (!state.tableAccess.active) state.Fail(access.inst, "reads a sampler table entry outside a table access");
        static_cast<void>(state.TableResource(DescriptorBindingKind::SamplerTable, 0u));
        const auto variant = (access.image.depthCompare ? 2u : 0u) + (point ? 1u : 0u);
        return "aps5_table." + MslTableField(DescriptorBindingKind::SamplerTable) + "[0].address[" + state.tableAccess.samplerElement + " + " + U(variant) + "].value";
    }
    // Reflection::samplerModeUses: the mode case this reference sits in, or an unconditional use.
    if (state.imageModeCase.second != ~0u) {
        state.reflection.samplerModeUses.push_back(state.imageModeCase.first);
        state.reflection.samplerModeUses.push_back(state.imageModeCase.second);
    } else {
        state.reflection.samplerUnconditional = true;
    }
    return HeapElement(state, DescriptorBindingKind::Samplers, state.DescriptorElement(DescriptorBindingKind::Samplers, access.mem.sampler) + (point ? 1u : 0u));
}

std::string SwizzleSelector(MslEmitterState& state, const Access& access, std::uint32_t component) {
    // RuntimeImageSwizzle: a table mode's dst_sel comes from the selected record's T# dword 3.
    if (access.image.table != NoTable) {
        if (!state.tableAccess.active) state.Fail(access.inst, "reads an image table swizzle outside a table access");
        return "((" + state.tableAccess.dword3 + " >> " + U(component * 3u) + ") & 7u)";
    }
    return state.FunctionConstant(PipelineSpecialization::ImageBase + access.mem.resource * PipelineSpecialization::ImageWords + 1u + component, component + 4u);
}

bool Arrayed(RdnaImageDimension dimension) {
    return dimension == RdnaImageDimension::Dim1DArray || dimension == RdnaImageDimension::Dim2DArray || dimension == RdnaImageDimension::Dim2DMsaaArray;
}

bool Multisampled(RdnaImageDimension dimension) {
    return dimension == RdnaImageDimension::Dim2DMsaa || dimension == RdnaImageDimension::Dim2DMsaaArray;
}

// Integer texel coordinates of the mode's view: `coord` (uint2 or uint3) and, for arrays, `slice`.
struct TexelCoord {
    std::string coord;
    std::string slice;
};

TexelCoord CoordU32(MslEmitterState& state, const Access& access) {
    const auto dimension = access.image.dimension;
    const auto components = Dimension(dimension).coordinateComponents;
    const auto encoded = std::min(components, AddressDimension(access).coordinateComponents);
    if (access.mem.imageAddressComponents < encoded) state.Fail(access.inst, "has an image address with too few coordinate components");
    const auto component = [&](std::uint32_t index) { return index < encoded ? AddressU32(state, access, index) : std::string("0u"); };
    switch (dimension) {
    case RdnaImageDimension::Dim1D:
        return {"uint2(" + component(0) + ", 0u)", ""};
    case RdnaImageDimension::Dim1DArray:
        return {"uint2(" + component(0) + ", 0u)", component(1)};
    case RdnaImageDimension::Dim2D:
    case RdnaImageDimension::Dim2DMsaa:
        return {"uint2(" + component(0) + ", " + component(1) + ")", ""};
    case RdnaImageDimension::Dim2DArray:
    case RdnaImageDimension::Dim2DMsaaArray:
        return {"uint2(" + component(0) + ", " + component(1) + ")", component(2)};
    case RdnaImageDimension::Dim3D:
        return {"uint3(" + component(0) + ", " + component(1) + ", " + component(2) + ")", ""};
    default:
        state.Fail(access.inst, "has an invalid image dimension");
    }
}

std::string LodU32(MslEmitterState& state, const Access& access) {
    if (!access.mem.imageHasMip) return "0u";
    const auto component = AddressDimension(access).coordinateComponents;
    if (access.mem.imageAddressComponents <= component) state.Fail(access.inst, "has an image address without the mip level component");
    return AddressU32(state, access, component);
}

std::string SampleIndex(MslEmitterState& state, const Access& access) {
    if (!Multisampled(access.image.dimension)) return "0u";
    const auto& addressInfo = AddressDimension(access);
    if (addressInfo.multisampled == 0u || access.mem.imageAddressComponents <= addressInfo.coordinateComponents) state.Fail(access.inst, "has no sample index in the image address");
    return AddressU32(state, access, addressInfo.coordinateComponents);
}

// --- multisampled storage views (MslAbi Version 6) -----------------------------------------------
// Metal shaders cannot write multisampled textures (MSL: texture2d_ms supports access::read only),
// so the storage heaps of the 2D MSAA and 2D MSAA array dimensions are texture2d_array textures of
// layers * samples slices: sample s of layer l is slice l * samples + s. The sample count is a
// function constant for direct images (SampleCountBase + resource, supplied by the binding plan from
// the T#) and the T#'s LAST_LEVEL (log2 samples on MSAA types) of the resource metadata for table
// images. Sampled multisampled views are texture2d_ms_array (Version 9, SampledMultisample).
bool EmulatedMultisample(const Access& access) {
    return access.image.resourceClass == ImageResourceClass::Storage && Multisampled(access.image.dimension);
}

std::string StorageSampleCount(MslEmitterState& state, const Access& access) {
    return state.FunctionConstant(PipelineSpecialization::SampleCountBase + access.mem.resource, 1u);
}

// MslAbi Version 9: sampled views of 2D MSAA (array) images are texture2d_ms_array heaps whose
// guest layer l, sample s is host slice l * p + s / h, host sample s % h, where n is the guest
// sample count (the T#'s, the SampleCountBase function constant or the table metadata), h the
// texture's own sample count and p = n / h host slices per guest layer: 1 for a native target, 2
// for an 8x target emulated as two 4x halves on a device that renders 4 at most (DECISIONS D28).
// A missing constant (default 1) reads as the texture's own count.
struct SampledMultisampleCoord {
    TexelCoord coord;
    std::string sample;
    std::string inside;
    std::string guestSamples;
    std::string slicesPerLayer;
};

SampledMultisampleCoord SampledMultisample(MslEmitterState& state, const Access& access, const std::string& texture, bool withSample) {
    const auto name = state.Temporary("aps5_mr");
    state.Line("const uint " + name + "_h = " + texture + ".get_num_samples();");
    state.Line("const uint " + name + "_n = max(" + StorageSampleCount(state, access) + ", " + name + "_h);");
    state.Line("const uint " + name + "_p = max(" + name + "_n / max(" + name + "_h, 1u), 1u);");
    SampledMultisampleCoord result;
    result.guestSamples = name + "_n";
    result.slicesPerLayer = name + "_p";
    if (!withSample) return result;
    const auto base = CoordU32(state, access);
    state.Line("const uint2 " + name + "_xy = " + base.coord + ";");
    state.Line("const uint " + name + "_s = " + SampleIndex(state, access) + ";");
    state.Line("const uint " + name + "_l = " + (base.slice.empty() ? std::string("0u") : base.slice) + ";");
    state.Line("const bool " + name + "_in = " + name + "_xy.x < " + texture + ".get_width() && " + name + "_xy.y < " + texture + ".get_height() && " + name + "_s < " + name + "_n && " + name + "_l < " + texture + ".get_array_size() / " + name + "_p;");
    result.coord = {name + "_xy", name + "_l * " + name + "_p + " + name + "_s / " + name + "_h"};
    result.sample = name + "_s % " + name + "_h";
    result.inside = name + "_in";
    return result;
}

// The 2D-array coordinate of the access's sample and whether x, y, the sample and the layer lie
// inside the view.
struct MultisampleStorageCoord {
    TexelCoord coord;
    std::string inside;
};

MultisampleStorageCoord EmulatedMultisampleCoord(MslEmitterState& state, const Access& access, const std::string& texture) {
    if (!state.target.storageImageMultisample) state.Fail(access.inst, "accesses a multisampled storage image: storage image multisampling is unavailable on the target device");
    if (access.image.mipMode == ImageMipMode::DynamicStorage) state.Fail(access.inst, "is a multisampled storage image with dynamic mip levels");
    const auto base = CoordU32(state, access);
    const auto sample = SampleIndex(state, access);
    const auto name = state.Temporary("aps5_ms");
    state.Line("const uint2 " + name + "_xy = " + base.coord + ";");
    state.Line("const uint " + name + "_n = " + StorageSampleCount(state, access) + ";");
    state.Line("const uint " + name + "_s = " + sample + ";");
    state.Line("const uint " + name + "_l = " + (base.slice.empty() ? std::string("0u") : base.slice) + ";");
    state.Line("const bool " + name + "_in = " + name + "_xy.x < " + texture + ".get_width() && " + name + "_xy.y < " + texture + ".get_height() && " + name + "_s < " + name + "_n && " + name + "_l < " + texture + ".get_array_size() / " + name + "_n;");
    return {{name + "_xy", name + "_l * " + name + "_n + " + name + "_s"}, name + "_in"};
}

// Declares `name` = whether the texel coordinate (at mip `lod` for sampled views) lies inside the
// view, and returns the read expression arguments after the coordinate.
std::string BoundsCheck(MslEmitterState& state, const std::string& texture, const RdnaImageDimension dimension, const TexelCoord& coord, const std::string& lod, bool storage, const std::string& sample) {
    const auto name = state.Temporary("aps5_in");
    const auto lodArg = storage ? std::string() : lod;
    const auto size = [&](const char* axis) { return texture + ".get_" + axis + "(" + lodArg + ")"; };
    std::string inside;
    if (!storage && !Multisampled(dimension)) inside = lod + " < " + texture + ".get_num_mip_levels() && ";
    const auto c = coord.coord;
    if (dimension == RdnaImageDimension::Dim3D) {
        inside += c + ".x < " + size("width") + " && " + c + ".y < " + size("height") + " && " + c + ".z < " + size("depth");
    } else if (Multisampled(dimension)) {
        inside += c + ".x < " + texture + ".get_width() && " + c + ".y < " + texture + ".get_height() && " + sample + " < " + texture + ".get_num_samples()";
    } else {
        inside += c + ".x < " + size("width") + " && " + c + ".y < " + size("height");
    }
    if (!coord.slice.empty()) inside += " && " + coord.slice + " < " + texture + ".get_array_size()";
    state.Line("const bool " + name + " = " + inside + ";");
    return name;
}

std::string ReadCall(const std::string& texture, RdnaImageDimension dimension, const TexelCoord& coord, const std::string& lod, bool storage, const std::string& sample) {
    if (Multisampled(dimension)) return texture + ".read(" + coord.coord + (coord.slice.empty() ? "" : ", " + coord.slice) + ", " + sample + ")";
    if (storage) return texture + ".read(" + coord.coord + (coord.slice.empty() ? "" : ", " + coord.slice) + ")";
    return texture + ".read(" + coord.coord + (coord.slice.empty() ? "" : ", " + coord.slice) + ", " + lod + ")";
}

SpirvBufferFormatInfo ImageConversionFormat(const ImageResource& image) {
    const auto format = image.conversionFormat;
    if (format == IrBufferFormat::Invalid) return {};
    const auto info = GetFormatInfo(format);
    if (SampledTextureNumericClass(format) != IrTextureNumericClass::Uint || RemapTextureFormat(format) == format || (info.type != SpirvFormatComponentType::Uint && info.type != SpirvFormatComponentType::Unorm && info.type != SpirvFormatComponentType::Float) || !info.packedBitfield || info.byteSize != sizeof(std::uint32_t) || info.componentCount == 0u || info.componentCount > 4u) {
        throw std::runtime_error("MSL emission failed: image conversion format is not a packed 32-bit unsigned integer, unorm or float format");
    }
    return info;
}

void RequireConvertedUnormAccess(MslEmitterState& state, const Access& access, const SpirvBufferFormatInfo& info) {
    if (access.mem.dataBits == 16u) state.Fail(access.inst, info.type == SpirvFormatComponentType::Float ? "reads or writes a converted float image with 16-bit data, which is not implemented" : "reads or writes a converted unorm image with 16-bit data, which is not implemented");
    for (std::uint32_t component = 0; component < 4u; component++) {
        const auto selector = (access.image.shaderSwizzle >> (component * 3u)) & 7u;
        if (selector >= 4u && selector - 4u >= info.componentCount) state.Fail(access.inst, "selects a channel the converted image format does not have");
    }
}

std::uint32_t FormatTypeCode(SpirvFormatComponentType type) {
    return static_cast<std::uint32_t>(type);
}

// UnpackImageTexel: `texel` (a typed vec4 of the view) → the shader-visible vec4 expression of the
// same numeric class (converted formats: uint bits of the unpacked channels).
std::string UnpackImageTexel(MslEmitterState& state, const Access& access, const std::string& texel) {
    if (access.image.srgbDecode) return "aps5_srgb_decode(" + texel + ")";
    const auto info = ImageConversionFormat(access.image);
    if (info.format == IrBufferFormat::Invalid) return texel;
    state.RequirePrelude(MslPreludeChunk::BufferFormat);
    const bool unorm = info.type == SpirvFormatComponentType::Unorm || info.type == SpirvFormatComponentType::Float;
    if (unorm) RequireConvertedUnormAccess(state, access, info);
    const auto packed = state.Temporary("aps5_pk");
    state.Line("const uint " + packed + " = " + texel + ".x;");
    std::array<std::string, 4> components{"0u", "0u", "0u", "0u"};
    for (std::uint32_t component = 0; component < info.componentCount; component++) {
        auto raw = "extract_bits(" + packed + ", " + U(info.componentBitOffset[component]) + ", " + U(info.componentBits[component]) + ")";
        if (unorm) raw = "aps5_format_normalize(" + U(FormatTypeCode(info.type)) + ", " + U(info.componentBits[component]) + ", " + raw + ")";
        components[component] = raw;
    }
    for (std::uint32_t component = info.componentCount; component < 4u; component++) components[component] = components[component % info.componentCount];
    const auto unpacked = state.Temporary("aps5_up");
    state.Line("const uint4 " + unpacked + " = uint4(" + components[0] + ", " + components[1] + ", " + components[2] + ", " + components[3] + ");");
    const auto one = U(FormattedConstantBits(info, SpirvFormattedSourceKind::One));
    const auto result = state.Temporary("aps5_cv");
    std::string channels;
    for (std::uint32_t component = 0; component < 4u; component++) {
        if (component != 0u) channels += ", ";
        channels += "aps5_swizzle_select(" + SwizzleSelector(state, access, component) + ", " + unpacked + ", " + one + ")";
    }
    state.Line("const uint4 " + result + " = uint4(" + channels + ");");
    return result;
}

// ResultVector: the 32-bit result dwords of a read/sample/gather. `value` is a typed vec4 (or a
// float for a depth comparison).
std::string ResultVector(MslEmitterState& state, const Access& access, const std::string& value, IrTextureNumericClass numeric, bool dref, bool gather) {
    const auto& mem = access.mem;
    auto valueClass = dref ? IrTextureNumericClass::Float : numeric;
    const bool integer = valueClass == IrTextureNumericClass::Uint || valueClass == IrTextureNumericClass::Sint;
    if (mem.dataBits == 16u && access.image.depthBits) state.Fail(access.inst, "reads the bits of a depth plane as 16-bit results");
    const auto result = state.Temporary("aps5_rv");
    if (mem.dataBits == 16u) {
        if (mem.dataDwords > 4u) state.Fail(access.inst, "has more than four packed image result dwords");
        const auto dmask = EffectiveDmask(mem);
        const auto zero = valueClass == IrTextureNumericClass::Float ? std::string("0.0f") : valueClass == IrTextureNumericClass::Sint ? std::string("0") : std::string("0u");
        const auto scalar = [&](std::uint32_t index) -> std::string {
            if (dref) return value;
            const auto component = gather ? index : DmaskComponent(dmask, index);
            return value + "[" + std::to_string(component) + "]";
        };
        std::array<std::string, 4> packed{"0u", "0u", "0u", "0u"};
        for (std::uint32_t word = 0; word < mem.dataDwords; word++) {
            const auto low = scalar(word * 2u);
            const auto high = word * 2u + 1u < mem.componentCount ? scalar(word * 2u + 1u) : zero;
            if (integer) {
                packed[word] = "((" + Bits(low, valueClass) + " & 0xffffu) | ((" + Bits(high, valueClass) + " & 0xffffu) << 16u))";
            } else {
                packed[word] = "aps5_pack_half2(" + low + ", " + high + ")";
            }
        }
        state.Line("const uint4 " + result + " = uint4(" + packed[0] + ", " + packed[1] + ", " + packed[2] + ", " + packed[3] + ");");
        return result;
    }
    std::array<std::string, 4> components;
    for (std::uint32_t index = 0; index < 4u; index++) {
        if (dref) {
            components[index] = "as_type<uint>(" + value + ")";
            continue;
        }
        const auto scalar = value + "[" + std::to_string(index) + "]";
        if (access.image.depthUnorm16) {
            components[index] = "uint(" + scalar + " * 65535.0f + 0.5f)";
        } else {
            components[index] = Bits(scalar, valueClass);
        }
        if (access.image.depthBits && !gather) {
            const auto selector = SwizzleSelector(state, access, index);
            components[index] = "(" + selector + " == 4u ? " + components[index] + " : ((" + selector + " == 1u || " + selector + " == 7u) ? 1u : 0u))";
        }
    }
    state.Line("const uint4 " + result + " = uint4(" + components[0] + ", " + components[1] + ", " + components[2] + ", " + components[3] + ");");
    return result;
}

SpirvBufferFormatInfo PackedTexelFormat(MslEmitterState& state, const Access& access) {
    const auto format = access.image.packedFormat;
    if (format == IrBufferFormat::Invalid || access.image.depthBits || access.image.conversionFormat != IrBufferFormat::Invalid) state.Fail(access.inst, "accesses packed texels of an image without a packed format");
    const auto info = GetFormatInfo(format);
    if (info.packedBitfield) state.Fail(access.inst, "accesses packed texels of a bitfield format");
    return info;
}

// PackedImageTexel: the raw texel bits (uint4) of a packed read.
std::string PackedImageTexel(MslEmitterState& state, const Access& access, const std::string& texel) {
    const auto info = PackedTexelFormat(state, access);
    std::uint32_t texelBits = 0;
    for (std::uint32_t component = 0; component < info.componentCount; component++) {
        const auto bits = info.componentBits[component];
        const bool exact = info.type == SpirvFormatComponentType::Uint || info.type == SpirvFormatComponentType::Sint || (info.type == SpirvFormatComponentType::Unorm && bits <= 16u) || (info.type == SpirvFormatComponentType::Float && bits == 32u);
        if (!exact) state.Fail(access.inst, "reads packed texels of a format whose bits are not recoverable from the view");
        texelBits += bits;
    }
    const auto numeric = access.image.numericClass;
    std::array<std::string, 4> words{"0u", "0u", "0u", "0u"};
    std::uint32_t position = 0;
    for (std::uint32_t component = 0; component < info.componentCount; component++) {
        const auto bits = info.componentBits[component];
        const auto scalar = texel + "[" + std::to_string(component) + "]";
        std::string value = info.type == SpirvFormatComponentType::Unorm ? "uint(" + scalar + " * " + F(static_cast<float>((1u << bits) - 1u)) + " + 0.5f)" : Bits(scalar, numeric);
        if (bits < 32u) value = "(" + value + " & " + U((1u << bits) - 1u) + ")";
        if (position % 32u != 0u) value = "(" + value + " << " + U(position % 32u) + ")";
        auto& word = words[position / 32u];
        word = word == "0u" ? value : "(" + word + " | " + value + ")";
        position += bits;
    }
    if (access.mem.dataSigned && texelBits < 32u) {
        words[0] = "as_type<uint>(as_type<int>(" + words[0] + " << " + U(32u - texelBits) + ") >> " + U(32u - texelBits) + ")";
    }
    const auto result = state.Temporary("aps5_pt");
    state.Line("const uint4 " + result + " = uint4(" + words[0] + ", " + words[1] + ", " + words[2] + ", " + words[3] + ");");
    return result;
}

// --- reads ---------------------------------------------------------------------------------------

void EmitByRead(MslEmitterState& state, const Access& access) {
    const auto elements = access.mem.imageByElements;
    const bool packed = access.mem.imagePacked;
    const auto components = packed ? 1u : access.mem.dataDwords / elements;
    if (access.image.dimension != RdnaImageDimension::Dim2D || components == 0u || (packed ? access.mem.dataDwords != 1u : components * elements != access.mem.dataDwords) || access.mem.dataBits != 32u) {
        state.Fail(access.inst, "is a MIMG BY2/BY4 load outside the measured 2D, 32-bit data subset");
    }
    const auto result = state.Name(access.inst);
    state.Line(result + " = uint4(0u);");
    state.Line("if (" + state.Arg(access.inst, 2) + ") {");
    state.Indent();
    const auto texture = Texture(state, access);
    const auto x = state.Temporary("aps5_bx");
    const auto y = state.Temporary("aps5_by");
    const auto lod = state.Temporary("aps5_bl");
    const auto inside = state.Temporary("aps5_bi");
    state.Line("const uint " + x + " = " + AddressU32(state, access, 0) + ";");
    state.Line("const uint " + y + " = " + AddressU32(state, access, 1) + ";");
    state.Line("const uint " + lod + "_raw = " + LodU32(state, access) + ";");
    state.Line("const bool " + lod + "_in = " + lod + "_raw < " + texture + ".get_num_mip_levels();");
    state.Line("const uint " + lod + " = " + lod + "_in ? " + lod + "_raw : 0u;");
    state.Line("const uint " + lod + "_w = max(" + texture + ".get_width(0) >> " + lod + ", 1u);");
    state.Line("const uint " + lod + "_h = max(" + texture + ".get_height(0) >> " + lod + ", 1u);");
    state.Line("const bool " + inside + " = " + lod + "_in && " + x + " < " + lod + "_w && (" + lod + "_w - " + x + ") >= " + U(elements) + " && " + y + " < " + lod + "_h;");
    state.Line("const uint " + x + "_first = " + inside + " ? (" + x + " & " + U(~(elements - 1u)) + ") : 0u;");
    state.Line("const uint " + y + "_row = " + inside + " ? " + y + " : 0u;");
    MemoryInfo texelMemory = access.mem;
    texelMemory.dmask = (1u << components) - 1u;
    texelMemory.dataDwords = components;
    texelMemory.componentCount = components;
    const Access texelAccess{access.inst, texelMemory, access.image, access.address, access.binding};
    std::array<std::string, 4> words{"0u", "0u", "0u", "0u"};
    const auto numeric = access.image.numericClass;
    for (std::uint32_t element = 0; element < elements; ++element) {
        const auto color = state.Temporary("aps5_bc");
        state.Line("const " + VectorType(numeric) + " " + color + " = " + texture + ".read(uint2(" + x + "_first + " + U(element) + ", " + y + "_row), " + lod + ");");
        if (packed) {
            const auto info = PackedTexelFormat(state, access);
            std::uint32_t texelBits = 0;
            for (std::uint32_t component = 0; component < info.componentCount; ++component) texelBits += info.componentBits[component];
            if (texelBits * elements > 32u) state.Fail(access.inst, "is a MIMG PCK2/PCK4 load whose elements do not fit one dword");
            const auto raw = PackedImageTexel(state, access, color);
            auto bits = "(" + raw + ".x & " + U(texelBits == 32u ? 0xffffffffu : (1u << texelBits) - 1u) + ")";
            if (element != 0u) bits = "(" + bits + " << " + U(element * texelBits) + ")";
            words[0] = "(" + words[0] + " | (" + inside + " ? " + bits + " : 0u))";
            continue;
        }
        const auto texel = ResultVector(state, texelAccess, UnpackImageTexel(state, texelAccess, color), numeric, false, false);
        for (std::uint32_t component = 0; component < components; ++component) {
            words[element * components + component] = "(" + inside + " ? " + texel + "[" + std::to_string(component) + "] : 0u)";
        }
    }
    state.Line(result + " = uint4(" + words[0] + ", " + words[1] + ", " + words[2] + ", " + words[3] + ");");
    state.Outdent();
    state.Line("}");
}

void EmitRead(MslEmitterState& state, const Access& access) {
    if (access.mem.imageByElements != 0u) {
        EmitByRead(state, access);
        return;
    }
    const auto numeric = access.image.numericClass;
    auto condition = state.Arg(access.inst, 2);
    // A 3D address on a 2D or (upstream bd09862e5) 1D mode reads slice 0 only; a 2D-array address
    // on a 3D mode (upstream 945b97094) is bounded by the volume depth in BoundsCheck.
    if (access.mem.imageDimension == RdnaImageDimension::Dim3D && (access.image.dimension == RdnaImageDimension::Dim2D || access.image.dimension == RdnaImageDimension::Dim1D)) {
        condition = "(" + condition + " && " + AddressU32(state, access, 2u) + " == 0u)";
    }
    const auto result = state.Name(access.inst);
    state.Line(result + " = uint4(0u);");
    state.Line("if (" + condition + ") {");
    state.Indent();
    const bool storage = access.image.resourceClass == ImageResourceClass::Storage;
    const auto texture = Texture(state, access);
    // Reads are sampled-class resources (ResourceTracker: only writes and atomics are storage), so a
    // multisampled storage read never reaches the emitter.
    if (EmulatedMultisample(access)) state.Fail(access.inst, "reads a multisampled storage image");
    if (Multisampled(access.image.dimension)) {
        // MslAbi Version 9: texture2d_ms_array, emulated sample counts (SampledMultisample).
        const auto ms = SampledMultisample(state, access, texture, true);
        state.Line("if (" + ms.inside + ") {");
        state.Indent();
        const auto color = state.Temporary("aps5_tx");
        state.Line("const " + VectorType(numeric) + " " + color + " = " + texture + ".read(" + ms.coord.coord + ", " + ms.coord.slice + ", " + ms.sample + ");");
        if (access.mem.imagePacked) {
            state.Line(result + " = " + PackedImageTexel(state, access, color) + ";");
        } else {
            state.Line(result + " = " + ResultVector(state, access, UnpackImageTexel(state, access, color), numeric, false, false) + ";");
        }
        state.Outdent();
        state.Line("}");
        state.Outdent();
        state.Line("}");
        return;
    }
    const auto coord = CoordU32(state, access);
    const auto lod = state.Temporary("aps5_lod");
    state.Line("const uint " + lod + " = " + LodU32(state, access) + ";");
    const auto sample = state.Temporary("aps5_smp");
    state.Line("const uint " + sample + " = " + SampleIndex(state, access) + ";");
    const auto inside = BoundsCheck(state, texture, access.image.dimension, coord, lod, storage, sample);
    state.Line("if (" + inside + ") {");
    state.Indent();
    const auto color = state.Temporary("aps5_tx");
    state.Line("const " + VectorType(numeric) + " " + color + " = " + ReadCall(texture, access.image.dimension, coord, lod, storage, sample) + ";");
    if (access.mem.imagePacked) {
        state.Line(result + " = " + PackedImageTexel(state, access, color) + ";");
    } else {
        state.Line(result + " = " + ResultVector(state, access, UnpackImageTexel(state, access, color), numeric, false, false) + ";");
    }
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
}

void EmitQueryDimensions(MslEmitterState& state, const Access& access) {
    const auto dimension = access.image.dimension;
    const auto texture = Texture(state, access);
    const bool storage = access.image.resourceClass == ImageResourceClass::Storage;
    const auto lod = storage ? std::string() : AddressU32(state, access, 0);
    std::array<std::string, 4> result{"0u", "0u", "0u", "0u"};
    const bool ms = Multisampled(dimension);
    if (EmulatedMultisample(access)) state.Fail(access.inst, "queries a multisampled storage image (queries are sampled-class resources)");
    const auto lodArg = ms ? std::string() : lod;
    result[0] = texture + ".get_width(" + lodArg + ")";
    if (ms) {
        // MslAbi Version 9 (DECISIONS D28): w is the T#'s level count LAST_LEVEL - BASE_LEVEL + 1,
        // which MSAA T#s fill with log2(samples) (BASE_LEVEL is 0: MultisampleSampledTexture), i.e.
        // log2(samples) + 1. RDNA has no sample-count query; compilers read 1 << LAST_LEVEL from the
        // T# (Mesa ac_nir_lower_resinfo query_samples). Layers are guest layers (emulated 8x targets
        // hold two host slices per layer).
        const auto samples = SampledMultisample(state, access, texture, false);
        result[1] = texture + ".get_height()";
        if (dimension == RdnaImageDimension::Dim2DMsaaArray) result[2] = "(" + texture + ".get_array_size() / " + samples.slicesPerLayer + ")";
        result[3] = "(ctz(" + samples.guestSamples + ") + 1u)";
        state.Define(access.inst, "uint4(" + result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ")");
        return;
    }
    switch (dimension) {
    case RdnaImageDimension::Dim1D:
        break;
    case RdnaImageDimension::Dim1DArray:
        result[1] = texture + ".get_array_size()";
        break;
    case RdnaImageDimension::Dim2D:
    case RdnaImageDimension::Dim2DMsaa:
        result[1] = texture + ".get_height(" + lodArg + ")";
        break;
    case RdnaImageDimension::Dim2DArray:
    case RdnaImageDimension::Dim2DMsaaArray:
        result[1] = texture + ".get_height(" + lodArg + ")";
        result[2] = texture + ".get_array_size()";
        break;
    case RdnaImageDimension::Dim3D:
        result[1] = texture + ".get_height(" + lodArg + ")";
        result[2] = texture + ".get_depth(" + lodArg + ")";
        break;
    default:
        state.Fail(access.inst, "queries an invalid image dimension");
    }
    if (access.mem.imageDimension == RdnaImageDimension::Dim3D && dimension == RdnaImageDimension::Dim2D) result[2] = "1u";
    if (!ms) result[3] = storage ? std::string("1u") : texture + ".get_num_mip_levels()";
    state.Define(access.inst, "uint4(" + result[0] + ", " + result[1] + ", " + result[2] + ", " + result[3] + ")");
}

// --- writes and atomics --------------------------------------------------------------------------

SpirvBufferFormatInfo SintStorageFormat(const ImageResource& image) {
    const auto format = image.conversionFormat;
    if (image.resourceClass != ImageResourceClass::Storage || format == IrBufferFormat::Invalid || SampledTextureNumericClass(format) != IrTextureNumericClass::Sint) return {};
    const auto info = GetFormatInfo(format);
    if (image.numericClass != IrTextureNumericClass::Uint || info.type != SpirvFormatComponentType::Sint || info.packedBitfield || info.componentCount == 0u || info.byteSize == 12u) {
        throw std::runtime_error("MSL emission failed: storage image conversion format is not a SINT format written through a UINT view");
    }
    return info;
}

// PackImageTexel: a converted store packs the (uint or float bit) channels into dword 0.
std::string PackImageTexel(MslEmitterState& state, const Access& access, const std::string& texel) {
    const auto info = ImageConversionFormat(access.image);
    if (info.format == IrBufferFormat::Invalid) return texel;
    state.RequirePrelude(MslPreludeChunk::BufferFormat);
    const bool unorm = info.type == SpirvFormatComponentType::Unorm || info.type == SpirvFormatComponentType::Float;
    if (unorm) RequireConvertedUnormAccess(state, access, info);
    std::string packed = "0u";
    for (std::uint32_t component = 0; component < info.componentCount; component++) {
        const auto value = texel + "[" + std::to_string(component) + "]";
        std::string clamped;
        if (unorm) {
            clamped = "aps5_format_store(" + U(FormatTypeCode(info.type)) + ", " + U(info.componentBits[component]) + ", " + value + ")";
        } else {
            clamped = "min(" + value + ", " + U(info.componentBits[component] == 32u ? 0xffffffffu : (1u << info.componentBits[component]) - 1u) + ")";
        }
        if (info.componentBitOffset[component] != 0u) clamped = "(" + clamped + " << " + U(info.componentBitOffset[component]) + ")";
        packed = "(" + packed + " | " + clamped + ")";
    }
    const auto result = state.Temporary("aps5_wp");
    state.Line("const uint4 " + result + " = uint4(" + packed + ", 0u, 0u, 0u);");
    return result;
}

// StoreTexel: the vec4 (uint4 for uint views, float4 otherwise) a store writes.
std::string StoreTexel(MslEmitterState& state, const Access& access, const std::string& data, bool integer) {
    const auto& mem = access.mem;
    const auto sint = SintStorageFormat(access.image);
    if (sint.format != IrBufferFormat::Invalid) state.RequirePrelude(MslPreludeChunk::BufferFormat);
    const auto dmask = EffectiveDmask(mem);
    std::array<std::string, 4> values;
    for (std::uint32_t component = 0; component < 4u; component++) {
        std::string raw = "0u";
        for (std::uint32_t reverse = 4u; reverse != 0u; --reverse) {
            const auto source = reverse - 1u;
            const auto packedIndex = DmaskComponentIndex(dmask, source);
            std::string candidate = "0u";
            if (((dmask >> source) & 1u) != 0u) {
                candidate = data + "[" + std::to_string(mem.dataBits == 16u ? packedIndex / 2u : packedIndex) + "]";
                if (mem.dataBits == 16u) {
                    if ((packedIndex & 1u) != 0u) candidate = "(" + candidate + " >> 16u)";
                    candidate = "(" + candidate + " & 0xffffu)";
                }
            }
            raw = "(" + SwizzleSelector(state, access, source) + " == " + U(component + 4u) + " ? " + candidate + " : " + raw + ")";
        }
        values[component] = integer ? raw : mem.dataBits == 16u ? "aps5_f16_value(" + raw + ")" : "as_type<float>(" + raw + ")";
        if (component < sint.componentCount) {
            // Upstream a18d6e9a6: D16 data takes EmitD16StoreComponent (sign-extended, then
            // saturated to the component width).
            if (mem.dataBits == 16u) values[component] = "aps5_format_d16_store(" + U(FormatTypeCode(sint.type)) + ", " + U(sint.componentBits[component]) + ", " + raw + ")";
            else values[component] = "aps5_format_store(" + U(FormatTypeCode(sint.type)) + ", " + U(sint.componentBits[component]) + ", " + values[component] + ")";
        }
    }
    const auto texel = state.Temporary("aps5_st");
    state.Line(std::string(integer ? "const uint4 " : "const float4 ") + texel + " = " + (integer ? "uint4(" : "float4(") + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ");");
    return sint.format != IrBufferFormat::Invalid ? texel : PackImageTexel(state, access, texel);
}

std::string PackedStoreTexel(MslEmitterState& state, const Access& access, const std::string& data) {
    const auto info = PackedTexelFormat(state, access);
    const auto numeric = access.image.numericClass;
    if (info.byteSize == 12u) state.Fail(access.inst, "stores packed texels of a format the hardware does not write");
    std::array<std::string, 4> words;
    for (std::uint32_t word = 0; word < 4u; word++) {
        words[word] = ((access.mem.dmask >> word) & 1u) == 0u ? std::string("0u") : data + "[" + std::to_string(DmaskComponentIndex(access.mem.dmask, word)) + "]";
    }
    std::array<std::string, 4> values;
    for (std::uint32_t component = 0; component < 4u; component++) {
        if (component >= info.componentCount) {
            values[component] = numeric == IrTextureNumericClass::Float ? "0.0f" : "0u";
            continue;
        }
        const auto bits = info.componentBits[component];
        const auto offset = info.componentBitOffset[component];
        const bool exact = (info.type == SpirvFormatComponentType::Uint && numeric == IrTextureNumericClass::Uint) || (bits == 32u && ((info.type == SpirvFormatComponentType::Sint && numeric == IrTextureNumericClass::Uint) || (info.type == SpirvFormatComponentType::Float && numeric == IrTextureNumericClass::Float)));
        if (!exact || offset % 32u + bits > 32u) state.Fail(access.inst, "stores packed texels of a format whose bits are not reproducible through the view");
        auto value = words[offset / 32u];
        if (offset % 32u != 0u) value = "(" + value + " >> " + U(offset % 32u) + ")";
        if (bits < 32u) value = "(" + value + " & " + U((1u << bits) - 1u) + ")";
        values[component] = numeric == IrTextureNumericClass::Float ? "as_type<float>(" + value + ")" : value;
    }
    const auto texel = state.Temporary("aps5_ps");
    const auto type = numeric == IrTextureNumericClass::Float ? std::string("float4") : std::string("uint4");
    state.Line("const " + type + " " + texel + " = " + type + "(" + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ");");
    return texel;
}

std::string WriteCall(const std::string& texture, RdnaImageDimension dimension, const TexelCoord& coord, const std::string& texel) {
    return texture + ".write(" + texel + ", " + coord.coord + (coord.slice.empty() ? "" : ", " + coord.slice) + ")";
}

void EmitWrite(MslEmitterState& state, const Access& access) {
    if (Multisampled(access.image.dimension) && !EmulatedMultisample(access)) state.Fail(access.inst, "stores to a multisampled image that is not a storage image");
    const bool uintImage = access.image.numericClass == IrTextureNumericClass::Uint;
    state.Line("if (" + state.Arg(access.inst, 3) + ") {");
    state.Indent();
    const auto data = state.Arg(access.inst, 2);
    if (EmulatedMultisample(access)) {
        // Metal cannot write texture2d_ms: the sample is a slice of the 2D-array backing.
        const auto texture = Texture(state, access);
        const auto ms = EmulatedMultisampleCoord(state, access, texture);
        const auto texel = access.mem.imagePacked ? PackedStoreTexel(state, access, data) : StoreTexel(state, access, data, uintImage);
        state.Line("if (" + ms.inside + ") " + WriteCall(texture, RdnaImageDimension::Dim2DArray, ms.coord, texel) + ";");
        state.Outdent();
        state.Line("}");
        return;
    }
    const auto coord = CoordU32(state, access);
    const auto texel = access.mem.imagePacked ? PackedStoreTexel(state, access, data) : StoreTexel(state, access, data, uintImage);
    const auto write = [&](const std::string& texture) {
        const auto inside = BoundsCheck(state, texture, access.image.dimension, coord, "0u", true, "0u");
        state.Line("if (" + inside + ") " + WriteCall(texture, access.image.dimension, coord, texel) + ";");
    };
    if (access.image.mipMode != ImageMipMode::DynamicStorage) {
        write(Texture(state, access));
    } else {
        if (access.image.mipCount == 0u) state.Fail(access.inst, "dynamic storage image has no mip descriptors");
        const auto count = state.FunctionConstant(PipelineSpecialization::MipCountBase + access.mem.resource, access.image.mipCount);
        const auto mip = state.Temporary("aps5_mip");
        state.Line("const uint " + mip + " = " + LodU32(state, access) + ";");
        state.Line("switch (" + mip + ") {");
        for (std::uint32_t level = 0; level < access.image.mipCount; level++) {
            state.Line("case " + U(level) + ":");
            state.Indent();
            state.Line("if (" + U(level) + " < " + count + ") {");
            state.Indent();
            write(Texture(state, access, level));
            state.Outdent();
            state.Line("}");
            state.Line("break;");
            state.Outdent();
        }
        state.Line("default:");
        state.Line("    break;");
        state.Line("}");
    }
    state.Outdent();
    state.Line("}");
}

std::optional<MslAtomicOp> AtomicOp(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::ImageAtomicSwap32: return MslAtomicOp::Swap;
    case IrOpcode::ImageAtomicCmpSwap32: return MslAtomicOp::CmpSwap;
    case IrOpcode::ImageAtomicIAdd32: return MslAtomicOp::Add;
    case IrOpcode::ImageAtomicISub32: return MslAtomicOp::Sub;
    case IrOpcode::ImageAtomicSMin32: return MslAtomicOp::SMin;
    case IrOpcode::ImageAtomicUMin32: return MslAtomicOp::UMin;
    case IrOpcode::ImageAtomicSMax32: return MslAtomicOp::SMax;
    case IrOpcode::ImageAtomicUMax32: return MslAtomicOp::UMax;
    case IrOpcode::ImageAtomicAnd32: return MslAtomicOp::And;
    case IrOpcode::ImageAtomicOr32: return MslAtomicOp::Or;
    case IrOpcode::ImageAtomicXor32: return MslAtomicOp::Xor;
    case IrOpcode::ImageAtomicInc32: return MslAtomicOp::Inc;
    case IrOpcode::ImageAtomicDec32: return MslAtomicOp::Dec;
    case IrOpcode::ImageAtomicFMin32: return MslAtomicOp::FMin;
    case IrOpcode::ImageAtomicFMax32: return MslAtomicOp::FMax;
    case IrOpcode::ImageAtomicFCmpSwap32: return MslAtomicOp::FCmpSwap;
    default: return std::nullopt;
    }
}

std::optional<std::uint32_t> Atomic64Op(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::ImageAtomicSwap64: return 0u;
    case IrOpcode::ImageAtomicCmpSwap64: return 1u;
    case IrOpcode::ImageAtomicIAdd64: return 2u;
    case IrOpcode::ImageAtomicISub64: return 3u;
    case IrOpcode::ImageAtomicSMin64: return 4u;
    case IrOpcode::ImageAtomicUMin64: return 5u;
    case IrOpcode::ImageAtomicSMax64: return 6u;
    case IrOpcode::ImageAtomicUMax64: return 7u;
    case IrOpcode::ImageAtomicAnd64: return 8u;
    case IrOpcode::ImageAtomicOr64: return 9u;
    case IrOpcode::ImageAtomicXor64: return 10u;
    default: return std::nullopt;
    }
}

// 64-bit image atomics: Apple9 has only atomic_max/min (no result) on 64-bit (RG32Uint) textures,
// so every read-modify-write of a texel takes a lock word of the driver's lock table
// (DriverData.atomicLocks, the Atomics64 chunk of MslMemory.cpp) in the shared lock loop
// (MslLockLoop: one elected invocation per SIMD group per round, as MslDeviceAtomic64); the new
// value is committed with a max or min so a texel is never seen half-written.
void EmitAtomic64(MslEmitterState& state, const Access& access) {
    const auto op = Atomic64Op(access.inst.Opcode());
    if (!op) state.Fail(access.inst, "is a 64-bit image atomic the MSL backend does not know");
    if (Multisampled(access.image.dimension) && !EmulatedMultisample(access)) state.Fail(access.inst, "is an atomic on a multisampled image that is not a storage image");
    state.RequirePrelude(MslPreludeChunk::Atomics64);
    state.usesDriverData = true;
    const auto condition = state.Arg(access.inst, access.inst.ArgumentCount() - 1u);
    const auto value = state.Arg(access.inst, 2);
    const auto comparator = *op == 1u ? state.Arg(access.inst, 3) : std::string("0ul");
    const auto result = state.Name(access.inst);
    state.Line(result + " = 0ul;");
    const auto texture = Texture(state, access);
    std::optional<MultisampleStorageCoord> ms;
    if (EmulatedMultisample(access)) ms = EmulatedMultisampleCoord(state, access, texture);
    const auto coord = ms ? ms->coord : CoordU32(state, access);
    const auto where = coord.coord + (coord.slice.empty() ? "" : ", " + coord.slice);
    const auto tag = state.Temporary("aps5_ia64");
    state.Line("bool " + tag + "_done = !(" + condition + ");");
    state.Line("if (!" + tag + "_done) {");
    state.Indent();
    const auto inside = ms ? ms->inside : BoundsCheck(state, texture, access.image.dimension, coord, "0u", true, "0u");
    state.Line("if (!" + inside + ") " + tag + "_done = true;");
    state.Outdent();
    state.Line("}");
    // The lock is hashed from the image resource and the texel.
    std::string key = "((ulong(" + U(access.mem.resource) + ") << 40u) ^ ulong(" + coord.coord + ".x) ^ (ulong(" + coord.coord + ".y) << 16u)";
    if (!coord.slice.empty()) key += " ^ (ulong(" + coord.slice + ") << 28u)";
    if (access.image.dimension == RdnaImageDimension::Dim3D) key += " ^ (ulong(" + coord.coord + ".z) << 28u)";
    key += ")";
    state.Line("device atomic_uint* " + tag + "_lock = aps5_atomic64_lock(aps5_driver.atomicLocks, " + key + " << 3u);");
    state.Line("if (aps5_driver.atomicLocks == 0ul) " + tag + "_done = true;");
    const std::string fence = "atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_device);";
    MslLockLoop(state, access.inst, {tag + "_lock", fence, fence}, tag + "_done", [&] {
        state.Line(result + " = " + texture + ".read(" + where + ").x;");
        state.Line("const ulong " + tag + "_next = aps5_atomic64_next(" + U(*op) + ", " + result + ", ulong(" + value + "), ulong(" + comparator + "));");
        state.Line("if (" + tag + "_next > " + result + ") " + texture + ".atomic_max(" + where + ", " + tag + "_next);");
        state.Line("else if (" + tag + "_next < " + result + ") " + texture + ".atomic_min(" + where + ", " + tag + "_next);");
    });
}

void EmitAtomic(MslEmitterState& state, const Access& access) {
    const auto opcode = access.inst.Opcode();
    if (IsImageAtomic64Opcode(opcode)) {
        EmitAtomic64(state, access);
        return;
    }
    const auto op = AtomicOp(opcode);
    if (!op) state.Fail(access.inst, "is an image atomic the MSL backend does not know");
    if (Multisampled(access.image.dimension) && !EmulatedMultisample(access)) state.Fail(access.inst, "is an atomic on a multisampled image that is not a storage image");
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    const auto condition = state.Arg(access.inst, access.inst.ArgumentCount() - 1u);
    const auto value = state.Arg(access.inst, 2);
    const bool compare = *op == MslAtomicOp::CmpSwap || *op == MslAtomicOp::FCmpSwap;
    const auto comparator = compare ? state.Arg(access.inst, 3) : std::string("0u");
    const auto result = state.Name(access.inst);
    state.Line(result + " = 0u;");
    state.Line("if (" + condition + ") {");
    state.Indent();
    const auto texture = Texture(state, access);
    std::optional<MultisampleStorageCoord> ms;
    if (EmulatedMultisample(access)) ms = EmulatedMultisampleCoord(state, access, texture);
    const auto coord = ms ? ms->coord : CoordU32(state, access);
    const auto inside = ms ? ms->inside : BoundsCheck(state, texture, access.image.dimension, coord, "0u", true, "0u");
    state.Line("if (" + inside + ") {");
    state.Indent();
    // The invocation's earlier image stores come first (Vulkan orders them by program order; Metal
    // only after a texture fence).
    state.Line("atomic_thread_fence(mem_flags::mem_texture, memory_order_seq_cst, thread_scope_device);");
    const auto where = coord.coord + (coord.slice.empty() ? "" : ", " + coord.slice);
    const auto native = [&](const char* method) { state.Line(result + " = " + texture + "." + method + "(" + where + ", " + value + ").x;"); };
    switch (*op) {
    case MslAtomicOp::Swap: native("atomic_exchange"); break;
    case MslAtomicOp::Add: native("atomic_fetch_add"); break;
    case MslAtomicOp::Sub: native("atomic_fetch_sub"); break;
    case MslAtomicOp::UMin: native("atomic_fetch_min"); break;
    case MslAtomicOp::UMax: native("atomic_fetch_max"); break;
    case MslAtomicOp::And: native("atomic_fetch_and"); break;
    case MslAtomicOp::Or: native("atomic_fetch_or"); break;
    case MslAtomicOp::Xor: native("atomic_fetch_xor"); break;
    default: {
        // Compare-exchange loop on the texel (aps5_atomic_next of the RuntimeBuffer chunk computes the
        // new value); compare-swap returns the observed value without writing when it differs.
        const auto observed = state.Temporary("aps5_ob");
        state.Line("uint4 " + observed + " = " + texture + ".atomic_load(" + where + ");");
        state.Line("for (uint aps5_i = 0u; aps5_i < 1048576u; ++aps5_i) {");
        state.Indent();
        state.Line("const uint aps5_next = aps5_atomic_next(" + U(static_cast<std::uint32_t>(*op)) + ", " + observed + ".x, " + value + ", " + comparator + ");");
        if (compare) state.Line("if (aps5_next == " + observed + ".x) break;");
        state.Line("if (" + texture + ".atomic_compare_exchange_weak(" + where + ", &" + observed + ", uint4(aps5_next))) break;");
        state.Outdent();
        state.Line("}");
        state.Line(result + " = " + observed + ".x;");
        break;
    }
    }
    state.Line("atomic_thread_fence(mem_flags::mem_texture, memory_order_seq_cst, thread_scope_device);");
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
}

// --- sampling ------------------------------------------------------------------------------------

struct SampleCoord {
    // float2 / float3 coordinate of the view, the array slice (uint) when arrayed.
    std::string coord;
    std::string slice;
};

std::string Float2(const std::string& x, const std::string& y) {
    return "float2(" + x + ", " + y + ")";
}

// CoordF32 for the view: 1D images are 2D textures one texel high (y at the texel centre).
SampleCoord CoordF32(MslEmitterState& state, const Access& access, std::uint32_t first, const std::string& texture) {
    const auto& info = Dimension(access.image.dimension);
    const auto components = info.coordinateComponents;
    const auto encoded = AddressDimension(access).coordinateComponents;
    const bool cube = access.image.cube;
    const auto present = std::min(encoded, components);
    // A cube T# read through DIM 2D (upstream 3b96840e6): face 0, the missing face coordinate reads as 0.
    const bool cubeFace = cube && access.mem.imageDimension == RdnaImageDimension::Dim2D;
    if (first == NoImageComponent || (encoded < components && ((cube && !cubeFace) || encoded == 0u)) || access.mem.imageAddressComponents < first + present) state.Fail(access.inst, "has an image address with too few coordinate components");
    const auto component = [&](std::uint32_t index) { return index < present ? AddressF32(state, access, first + index) : std::string("0.0f"); };
    auto x = component(0);
    switch (access.image.dimension) {
    case RdnaImageDimension::Dim1D:
        return {Float2(x, "0.5f"), ""};
    case RdnaImageDimension::Dim1DArray:
        return {Float2(x, "0.5f"), "aps5_layer(" + component(1) + ", " + texture + ".get_array_size())"};
    case RdnaImageDimension::Dim2D:
        return {Float2(x, component(1)), ""};
    case RdnaImageDimension::Dim2DArray: {
        auto y = component(1);
        auto z = component(2);
        if (cube) {
            x = "(" + x + " - 1.0f)";
            y = "(" + y + " - 1.0f)";
            z = "aps5_cube_layer(" + z + ")";
        }
        return {Float2(x, y), "aps5_layer(" + z + ", " + texture + ".get_array_size())"};
    }
    case RdnaImageDimension::Dim3D:
        return {"float3(" + x + ", " + component(1) + ", " + component(2) + ")", ""};
    default:
        state.Fail(access.inst, "samples an image dimension Metal cannot sample");
    }
}

// A gradient vector (spatial components) of the view.
std::string Gradient(MslEmitterState& state, const Access& access, std::uint32_t first) {
    const auto spatial = Dimension(access.image.dimension).spatialComponents;
    const auto encoded = AddressDimension(access).spatialComponents;
    const auto component = [&](std::uint32_t index) { return index < std::min(spatial, encoded) ? AddressF32(state, access, first + index) : std::string("0.0f"); };
    if (spatial == 1u) return Float2(component(0), "0.0f");
    if (spatial == 2u) return Float2(component(0), component(1));
    return "float3(" + component(0) + ", " + component(1) + ", " + component(2) + ")";
}

// The texel offset as int2/int3 (6-bit signed fields at an 8-bit stride).
std::string PackedOffset(MslEmitterState& state, const Access& access, const ImageSampleLayout& layout) {
    if (layout.offset == NoImageComponent || access.mem.imageAddressComponents <= layout.offset) state.Fail(access.inst, "has no texel offset in the image address");
    const auto spatial = Dimension(access.image.dimension).spatialComponents;
    const auto packed = "as_type<int>(" + AddressU32(state, access, layout.offset) + ")";
    const auto field = [&](std::uint32_t index) { return "extract_bits(" + packed + ", " + U(index * 8u) + ", 6u)"; };
    if (spatial == 1u) return "int2(" + field(0) + ", 0)";
    if (spatial == 2u) return "int2(" + field(0) + ", " + field(1) + ")";
    return "int3(" + field(0) + ", " + field(1) + ", " + field(2) + ")";
}

void EmitConstantSwizzleSample(MslEmitterState& state, const Access& access) {
    const auto numeric = access.image.numericClass;
    const bool gather = access.inst.Opcode() == IrOpcode::ImageGatherRaw;
    const auto scalar = ScalarType(numeric);
    const auto one = numeric == IrTextureNumericClass::Float ? std::string("1.0f") : numeric == IrTextureNumericClass::Sint ? std::string("1") : std::string("1u");
    const auto zero = numeric == IrTextureNumericClass::Float ? std::string("0.0f") : numeric == IrTextureNumericClass::Sint ? std::string("0") : std::string("0u");
    std::array<std::string, 4> channels;
    for (std::uint32_t channel = 0; channel < 4u; channel++) {
        const auto selector = SwizzleSelector(state, access, gather ? ImageGatherComponent(state, access.inst, EffectiveDmask(access.mem)) : channel);
        channels[channel] = "(" + selector + " == 1u ? " + one + " : " + zero + ")";
    }
    const auto value = state.Temporary("aps5_cs");
    state.Line("const " + VectorType(numeric) + " " + value + " = " + VectorType(numeric) + "(" + channels[0] + ", " + channels[1] + ", " + channels[2] + ", " + channels[3] + ");");
    state.Define(access.inst, ResultVector(state, access, value, numeric, false, gather));
}

void EmitPackedHorizontalGather(MslEmitterState& state, const Access& access) {
    const auto& mem = access.mem;
    const auto dimension = access.image.dimension;
    if (dimension != RdnaImageDimension::Dim1D && dimension != RdnaImageDimension::Dim2D) state.Fail(access.inst, "is a packed horizontal gather of an image dimension whose row and slice rules are not measured");
    if (Layout(mem).clamp != NoImageComponent || HasFlag(mem, RdnaImageSampleFlagCompare)) state.Fail(access.inst, "is a packed horizontal gather with depth comparison or an LOD clamp, which is not implemented");
    if (AddressDimension(access).coordinateComponents < Dimension(dimension).coordinateComponents) state.Fail(access.inst, "has an image address with too few coordinate components");
    const auto info = PackedTexelFormat(state, access);
    std::uint32_t texelBits = 0;
    for (std::uint32_t component = 0; component < info.componentCount; component++) texelBits += info.componentBits[component];
    std::array<std::string, 4> stream{"0u", "0u", "0u", "0u"};
    std::string rowInside;
    const bool twoDimensional = dimension == RdnaImageDimension::Dim2D;
    if (texelBits <= 32u) {
        const auto texture = Texture(state, access);
        const auto tag = state.Temporary("aps5_hg");
        state.Line("const int " + tag + "_w = int(" + texture + ".get_width(0));");
        state.Line("const int " + tag + "_h = int(" + texture + ".get_height(0));");
        const auto texelFloor = [&](std::uint32_t address, const std::string& length, bool centre) {
            std::string scaled = "(" + AddressF32(state, access, address) + " * float(" + length + "))";
            if (centre) scaled = "(" + scaled + " - 0.5f)";
            return "int(floor(" + scaled + "))";
        };
        state.Line("const int " + tag + "_x = " + texelFloor(0u, tag + "_w", true) + ";");
        std::string row = "0";
        if (twoDimensional) {
            state.Line("const int " + tag + "_ry = " + texelFloor(1u, tag + "_h", false) + ";");
            rowInside = tag + "_ri";
            state.Line("const bool " + rowInside + " = " + tag + "_ry < " + tag + "_h;");
            row = "clamp(" + tag + "_ry, 0, " + tag + "_h - 1)";
        }
        for (std::uint32_t element = 0; element < 4u; element++) {
            const auto color = state.Temporary("aps5_hc");
            const auto column = "clamp(" + tag + "_x + " + std::to_string(static_cast<int>(element) - 1) + ", 0, " + tag + "_w - 1)";
            state.Line("const " + VectorType(access.image.numericClass) + " " + color + " = " + texture + ".read(uint2(uint(" + column + "), uint(" + row + ")), 0u);");
            const auto bits = PackedImageTexel(state, access, color) + ".x";
            const auto position = element * texelBits;
            const auto word = position / 32u;
            const auto shift = position % 32u;
            stream[word] = "(" + stream[word] + " | " + (shift == 0u ? bits : "(" + bits + " << " + U(shift) + ")") + ")";
            if (shift != 0u && shift + texelBits > 32u) stream[word + 1u] = "(" + stream[word + 1u] + " | (" + bits + " >> " + U(32u - shift) + "))";
        }
    }
    std::array<std::string, 4> selected{"0u", "0u", "0u", "0u"};
    std::uint32_t next = 0;
    for (std::uint32_t bit = 0; bit < 4u; bit++) {
        if (((mem.dmask >> bit) & 1u) == 0u) continue;
        if (texelBits <= 32u) selected[next] = rowInside.empty() ? stream[bit] : "(" + rowInside + " ? " + stream[bit] + " : 0u)";
        next++;
    }
    state.Define(access.inst, "uint4(" + selected[0] + ", " + selected[1] + ", " + selected[2] + ", " + selected[3] + ")");
}

std::string UnpackImageGather(MslEmitterState& state, const Access& access, const std::string& gathered) {
    const auto info = ImageConversionFormat(access.image);
    if (info.format == IrBufferFormat::Invalid) return gathered;
    const auto component = ImageGatherComponent(state, access.inst, EffectiveDmask(access.mem));
    const auto selector = SwizzleSelector(state, access, component);
    std::array<std::string, 4> values;
    for (std::uint32_t lane = 0; lane < 4u; lane++) {
        std::array<std::string, 4> components;
        for (std::uint32_t index = 0u; index < 4u; ++index) {
            const auto physical = index % info.componentCount;
            components[index] = "extract_bits(" + gathered + "[" + std::to_string(lane) + "], " + U(info.componentBitOffset[physical]) + ", " + U(info.componentBits[physical]) + ")";
        }
        values[lane] = "aps5_swizzle_select(" + selector + ", uint4(" + components[0] + ", " + components[1] + ", " + components[2] + ", " + components[3] + "), 0u)";
    }
    const auto result = state.Temporary("aps5_ug");
    state.Line("const uint4 " + result + " = uint4(" + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ");");
    return result;
}

const char* ComponentName(std::uint32_t component) {
    static constexpr const char* Names[] = {"component::x", "component::y", "component::z", "component::w"};
    return Names[component];
}

void EmitOneDimensionalGatherLz(MslEmitterState& state, const Access& access, const ImageSampleLayout& layout, const SampleCoord& coord, const std::string& texture, const std::string& sampler) {
    const auto numeric = access.image.numericClass;
    const bool arrayed = access.image.dimension == RdnaImageDimension::Dim1DArray;
    const auto tag = state.Temporary("aps5_g1");
    state.Line("const float " + tag + "_w = float(" + texture + ".get_width(0));");
    std::string left = "floor(" + coord.coord + ".x * " + tag + "_w - 0.5f)";
    if (layout.offset != NoImageComponent) left = "(" + left + " + float(" + PackedOffset(state, access, layout) + ".x))";
    state.Line("const float " + tag + "_l = " + left + ";");
    const auto component = ImageConversionFormat(access.image).format == IrBufferFormat::Invalid ? ImageGatherComponent(state, access.inst, EffectiveDmask(access.mem)) : 0u;
    std::array<std::string, 2> values;
    for (std::uint32_t index = 0; index < 2u; index++) {
        const auto at = Float2("(" + tag + "_l + " + (index == 0u ? std::string("0.5f") : std::string("1.5f")) + ") / " + tag + "_w", "0.5f");
        values[index] = texture + ".sample(" + sampler + ", " + at + (arrayed ? ", " + coord.slice : "") + ", level(0.0f))[" + std::to_string(component) + "]";
    }
    const auto result = state.Temporary("aps5_g1r");
    state.Line("const " + ScalarType(numeric) + " " + result + "_a = " + values[0] + ";");
    state.Line("const " + ScalarType(numeric) + " " + result + "_b = " + values[1] + ";");
    state.Line("const " + VectorType(numeric) + " " + result + " = " + VectorType(numeric) + "(" + result + "_a, " + result + "_b, " + result + "_b, " + result + "_a);");
    state.Define(access.inst, ResultVector(state, access, UnpackImageGather(state, access, result), numeric, false, true));
}

void EmitGather(MslEmitterState& state, const Access& access, const ImageSampleLayout& layout, bool dref, const SampleCoord& coord, const std::string& texture, const std::string& sampler) {
    const auto& mem = access.mem;
    const auto dimension = access.image.dimension;
    if (layout.clamp != NoImageComponent) state.Fail(access.inst, "is a gather with an LOD clamp, which is not implemented");
    if (dref && (access.image.emulatedCompare & EmulatedCompare::Enabled) != 0u) state.Fail(access.inst, "is a comparison gather of a color texture, which is not implemented");
    if (dimension == RdnaImageDimension::Dim1D || dimension == RdnaImageDimension::Dim1DArray) {
        if (dref || !HasFlag(mem, RdnaImageSampleFlagLevelZero) || HasFlag(mem, RdnaImageSampleFlagGatherHorizontal)) state.Fail(access.inst, "has an unsupported 1D gather variant");
        EmitOneDimensionalGatherLz(state, access, layout, coord, texture, sampler);
        return;
    }
    if (dimension == RdnaImageDimension::Dim3D) state.Fail(access.inst, "gathers a 3D image, which Metal cannot gather");
    const auto slice = coord.slice.empty() ? std::string() : ", " + coord.slice;
    std::string drefValue;
    if (dref) {
        if (layout.dref == NoImageComponent || mem.imageAddressComponents <= layout.dref) state.Fail(access.inst, "has no depth reference in the image address");
        drefValue = AddressF32(state, access, layout.dref);
    }
    std::uint32_t component = 0;
    if (!dref && ImageConversionFormat(access.image).format == IrBufferFormat::Invalid) component = ImageGatherComponent(state, access.inst, EffectiveDmask(mem));
    const auto numeric = dref ? IrTextureNumericClass::Float : access.image.numericClass;
    const auto gathered = state.Temporary("aps5_ga");
    const auto call = [&](const std::string& offset) {
        if (dref) return texture + ".gather_compare(" + sampler + ", " + coord.coord + slice + ", " + drefValue + ", " + offset + ")";
        return texture + ".gather(" + sampler + ", " + coord.coord + slice + ", " + offset + ", " + ComponentName(component) + ")";
    };
    if (HasFlag(mem, RdnaImageSampleFlagGatherHorizontal)) {
        // ConstOffsets (-1,0), (0,0), (1,0), (2,0): texel i is the (i0, j0) texel (.w) of the gather
        // at offset i.
        state.Line("const " + VectorType(numeric) + " " + gathered + " = " + VectorType(numeric) + "(" + call("int2(-1, 0)") + ".w, " + call("int2(0, 0)") + ".w, " + call("int2(1, 0)") + ".w, " + call("int2(2, 0)") + ".w);");
    } else {
        const auto offset = layout.offset != NoImageComponent ? PackedOffset(state, access, layout) : std::string("int2(0)");
        state.Line("const " + VectorType(numeric) + " " + gathered + " = " + call(offset) + ";");
    }
    state.Define(access.inst, ResultVector(state, access, UnpackImageGather(state, access, gathered), numeric, false, true));
}

void EmitEmulatedCompareSample(MslEmitterState& state, const Access& access, const ImageSampleLayout& layout, const std::string& texture) {
    const auto& mem = access.mem;
    const auto& image = access.image;
    if (HasFlag(mem, RdnaImageSampleFlagDerivative) || HasFlag(mem, RdnaImageSampleFlagLod)) state.Fail(access.inst, "compares against a color texture with gradients or an explicit LOD, which is not implemented");
    const bool arrayed = image.dimension == RdnaImageDimension::Dim2DArray;
    if (image.dimension != RdnaImageDimension::Dim2D && !arrayed) state.Fail(access.inst, "compares against a color texture that is not a 2D or 2D array view, which is not implemented");
    if (layout.dref == NoImageComponent || mem.imageAddressComponents <= layout.dref) state.Fail(access.inst, "has no depth reference in the image address");
    const auto parameter = [&](std::uint32_t word) { return state.FunctionConstant(PipelineSpecialization::CompareBase + mem.resource * PipelineSpecialization::CompareWords + word, 0u); };
    const auto coordinate = [&](std::uint32_t index) { return AddressF32(state, access, layout.coord + index); };
    std::string layer = "0u";
    if (arrayed) layer = "uint(clamp(int(floor(" + coordinate(2) + " + 0.5f)), 0, int(" + texture + ".get_array_size()) - 1))";
    const auto offset = layout.offset != NoImageComponent ? "int2(" + PackedOffset(state, access, layout) + ".xy)" : std::string("int2(0)");
    std::string swizzle = "uint4(";
    for (std::uint32_t channel = 0; channel < 4u; ++channel) swizzle += (channel != 0u ? ", " : "") + SwizzleSelector(state, access, channel);
    swizzle += ")";
    const auto value = state.Temporary("aps5_ec");
    state.Line("const float4 " + value + " = aps5_ec_sample(" + texture + ", float2(" + coordinate(0) + ", " + coordinate(1) + "), " + layer + ", " + offset + ", uint4(" + parameter(0) + ", " + parameter(1) + ", " + parameter(2) + ", " + parameter(3) + "), uint2(" + parameter(4) + ", " + parameter(5) + "), " + swizzle + ", " + AddressF32(state, access, layout.dref) + ");");
    state.Define(access.inst, ResultVector(state, access, value, IrTextureNumericClass::Float, false, false));
}

void EmitSample(MslEmitterState& state, const Access& access, const ImageSampleLayout& layout, bool dref, const SampleCoord& coord, const std::string& texture, const std::string& sampler) {
    const auto& mem = access.mem;
    const auto& image = access.image;
    if (dref && (image.emulatedCompare & EmulatedCompare::Enabled) != 0u) {
        EmitEmulatedCompareSample(state, access, layout, texture);
        return;
    }
    const bool explicitLod = ImageSampleExplicitLod(mem.imageSampleFlags, state.program.Resources().stage);
    const auto spatial = Dimension(image.dimension).spatialComponents;
    std::string options;
    std::string clamp;
    if (layout.clamp != NoImageComponent) clamp = AddressF32(state, access, layout.clamp);
    if (HasFlag(mem, RdnaImageSampleFlagDerivative)) {
        const auto gx = Gradient(state, access, layout.gradX);
        const auto gy = Gradient(state, access, layout.gradY);
        options = (spatial == 3u ? "gradient3d(" : "gradient2d(") + gx + ", " + gy + ")";
        if (!clamp.empty()) options += ", min_lod_clamp(" + clamp + ")";
    } else if (explicitLod) {
        auto lod = HasFlag(mem, RdnaImageSampleFlagLod) ? AddressF32(state, access, layout.lod) : std::string("0.0f");
        if (!clamp.empty()) lod = "max(" + lod + ", " + clamp + ")";
        options = "level(" + lod + ")";
    } else if (layout.bias != NoImageComponent) {
        options = "bias(" + AddressF32(state, access, layout.bias) + ")";
        if (!clamp.empty()) options += ", min_lod_clamp(" + clamp + ")";
    } else if (!clamp.empty()) {
        options = "min_lod_clamp(" + clamp + ")";
    }
    const bool hasOffset = layout.offset != NoImageComponent;
    const bool foldOffset = hasOffset && FoldsTexelOffsets(state.program.Info().samplers.at(mem.sampler)) && ResourceMaterializer::UnnormalizedSampleMismatch(image) == nullptr;
    const auto slice = coord.slice.empty() ? std::string() : ", " + coord.slice;
    std::string drefValue;
    if (dref) {
        if (layout.dref == NoImageComponent || mem.imageAddressComponents <= layout.dref) state.Fail(access.inst, "has no depth reference in the image address");
        drefValue = AddressF32(state, access, layout.dref);
    }
    const auto call = [&](const std::string& at, const std::string& offset) {
        std::string arguments = sampler + ", " + at + slice;
        if (dref) arguments += ", " + drefValue;
        if (!options.empty()) arguments += ", " + options;
        if (!offset.empty()) arguments += ", " + offset;
        return texture + (dref ? ".sample_compare(" : ".sample(") + arguments + ")";
    };
    const auto numeric = dref ? IrTextureNumericClass::Float : access.image.numericClass;
    const auto sample = state.Temporary("aps5_sa");
    const auto type = dref ? std::string("float") : VectorType(numeric);
    const auto offset = hasOffset ? PackedOffset(state, access, layout) : std::string();
    const auto folded = [&]() {
        // FoldedOffsetCoord: the offset added to the (unnormalized) coordinate.
        const auto offsetF = spatial == 3u ? "float3(" + offset + ")" : "float2(" + offset + ")";
        return "(" + coord.coord + " + " + (spatial == 1u ? "float2(float(" + offset + ".x), 0.0f)" : offsetF) + ")";
    };
    const bool reducible = !dref && access.image.numericClass == IrTextureNumericClass::Float && explicitLod && !HasFlag(mem, RdnaImageSampleFlagDerivative) && spatial == 2u && (image.dimension == RdnaImageDimension::Dim1D || image.dimension == RdnaImageDimension::Dim2D || image.dimension == RdnaImageDimension::Dim2DArray);
    std::string expression;
    if (reducible && !foldOffset) {
        // A min/max reduction sampler (PipelineSpecialization::ReductionBase, supplied only for those).
        const auto first = PipelineSpecialization::ReductionBase + mem.sampler * PipelineSpecialization::ReductionWords;
        const auto packed = state.FunctionConstant(first, 0u);
        const auto filters = state.FunctionConstant(first + 1u, 0u);
        auto lod = HasFlag(mem, RdnaImageSampleFlagLod) ? AddressF32(state, access, layout.lod) : std::string("0.0f");
        if (!clamp.empty()) lod = "max(" + lod + ", " + clamp + ")";
        const auto offset2 = offset.empty() ? std::string("int2(0)") : "int2(" + offset + ".xy)";
        expression = "(" + packed + " & 3u) != 0u ? aps5_reduce_sample(" + texture + ", " + sampler + ", " + coord.coord + ", " + (coord.slice.empty() ? std::string("0u") : coord.slice) + ", " + lod + ", " + offset2 + ", " + packed + ", " + filters + ") : " + call(coord.coord, offset);
    } else if (!foldOffset) {
        expression = call(coord.coord, offset);
    } else {
        const auto selector = state.FunctionConstant(PipelineSpecialization::FoldOffsetBase + mem.sampler, 0u);
        expression = selector + " != 0u ? " + call(folded(), "") + " : " + call(coord.coord, offset);
    }
    // Upstream #1418 (SpirvImageEmitter EmitSampleOp): a sampler that only samples with implicit LODs
    // or gradients (SamplesBaseLevelIfUnnormalized) and that the binding plan proves unnormalized
    // (function constant UnnormalizedBaseLevelBase + s, 1) samples the base level: an explicit LOD 0
    // and no other operand, as the console's unnormalized samplers do. Upstream 7c19c66de: a texel
    // offset is folded into the (unnormalized) coordinate there (FoldedOffsetCoord).
    if (!dref && SamplesBaseLevelIfUnnormalized(state.program.Info().samplers.at(mem.sampler))) {
        const auto selector = state.FunctionConstant(PipelineSpecialization::UnnormalizedBaseLevelBase + mem.sampler, 0u);
        expression = selector + " != 0u ? " + texture + ".sample(" + sampler + ", " + (hasOffset ? folded() : coord.coord) + slice + ", level(0.0f)) : " + expression;
    }
    state.Line("const " + type + " " + sample + " = " + expression + ";");
    const auto result = dref ? sample : UnpackImageTexel(state, access, sample);
    state.Define(access.inst, ResultVector(state, access, result, access.image.numericClass, dref, false));
}

void EmitQueryLod(MslEmitterState& state, const Access& access) {
    const auto info = ImageConversionFormat(access.image);
    if (info.type == SpirvFormatComponentType::Unorm) state.Fail(access.inst, "queries the level of detail of a converted unorm image, which is not implemented");
    if (info.type == SpirvFormatComponentType::Float) state.Fail(access.inst, "queries the level of detail of a converted float image, which is not implemented");
    const auto texture = Texture(state, access);
    const auto sampler = Sampler(state, access);
    const auto coord = CoordF32(state, access, 0, texture);
    if (state.program.Resources().stage != IrShaderStage::Pixel) {
        // Upstream 1ef97e517: a compute LOD query takes its derivatives from quads of four
        // consecutive lanes (GFX10, DerivativeGroupLinearKHR). MSL's calculate_*_lod exist in
        // fragment functions only, so the LOD is computed here from the quad's coordinates
        // (quad_shuffle_xor: lane ^ 1 is the x neighbour, lane ^ 2 the y neighbour; guest lanes are
        // SIMD lanes in order, DECISIONS D25) as the isotropic log2 of the larger texel-space
        // footprint. The clamped LOD is bounded by the view's levels only: MSL does not expose the
        // sampler's LOD range or bias (TechnicalDebt metal-compute-lod).
        if (state.program.Resources().stage != IrShaderStage::Compute) state.Fail(access.inst, "queries an implicit level of detail outside pixel and compute programs");
        if (access.image.cube) state.Fail(access.inst, "queries the level of detail of a cube image in a compute program, which is not implemented");
        const bool volume = access.image.dimension == RdnaImageDimension::Dim3D;
        const auto vector = volume ? std::string("float3") : std::string("float2");
        const auto at = state.Temporary("aps5_lq");
        state.Line("const " + vector + " " + at + " = " + coord.coord + ";");
        const auto size = vector + "(float(" + texture + ".get_width(0u)), float(" + texture + ".get_height(0u))" + (volume ? ", float(" + texture + ".get_depth(0u))" : std::string()) + ")";
        state.Line("const float " + at + "_rho = max(length((quad_shuffle_xor(" + at + ", 1u) - " + at + ") * " + size + "), length((quad_shuffle_xor(" + at + ", 2u) - " + at + ") * " + size + "));");
        state.Line("const float " + at + "_lod = log2(" + at + "_rho);");
        state.Define(access.inst, "uint4(as_type<uint>(clamp(" + at + "_lod, 0.0f, float(" + texture + ".get_num_mip_levels() - 1u))), as_type<uint>(" + at + "_lod), 0u, 0u)");
        return;
    }
    state.Define(access.inst, "uint4(as_type<uint>(" + texture + ".calculate_clamped_lod(" + sampler + ", " + coord.coord + ")), as_type<uint>(" + texture + ".calculate_unclamped_lod(" + sampler + ", " + coord.coord + ")), 0u, 0u)");
}

void EmitSampling(MslEmitterState& state, const Access& access) {
    if (access.image.constantSwizzle) {
        EmitConstantSwizzleSample(state, access);
        return;
    }
    if (access.image.srgbDecode) state.Fail(access.inst, "samples or gathers an sRGB image the device cannot sample, which is not implemented");
    if (access.mem.imagePacked) {
        EmitPackedHorizontalGather(state, access);
        return;
    }
    const auto layout = Layout(access.mem);
    const bool dref = HasFlag(access.mem, RdnaImageSampleFlagCompare);
    if (dref && access.image.conversionFormat != IrBufferFormat::Invalid) state.Fail(access.inst, "uses depth comparison with a packed integer image");
    if (ImageConversionFormat(access.image).type == SpirvFormatComponentType::Unorm) state.Fail(access.inst, "samples or gathers a converted unorm image, which needs filtering in the shader and is not implemented");
    if (ImageConversionFormat(access.image).type == SpirvFormatComponentType::Float) state.Fail(access.inst, "samples or gathers a converted float image, which needs filtering in the shader and is not implemented");
    const auto texture = Texture(state, access);
    if (dref && access.inst.Opcode() != IrOpcode::ImageGatherRaw && (access.image.emulatedCompare & EmulatedCompare::Enabled) != 0u) {
        // The emulated comparison reads texels itself: it references no sampler (the binding plan
        // then drops the sampler binding and its image associations, as the SPIR-V module does).
        EmitEmulatedCompareSample(state, access, layout, texture);
        return;
    }
    const auto sampler = Sampler(state, access);
    const auto coord = CoordF32(state, access, layout.coord, texture);
    if (access.inst.Opcode() == IrOpcode::ImageGatherRaw) {
        EmitGather(state, access, layout, dref, coord, texture, sampler);
    } else {
        EmitSample(state, access, layout, dref, coord, texture, sampler);
    }
}

const IrValue& ImageAddress(MslEmitterState& state, const IrValue& inst, const IrValue* value) {
    const IrValue* resolved = value != nullptr ? value->Resolve() : nullptr;
    if (resolved == nullptr || resolved->Opcode() != IrOpcode::MakeImageAddress) state.Fail(inst, "has a typed image address that MakeImageAddress did not construct");
    return *resolved;
}

// Upstream c97e5e0338: a depth-comparison 1D (array) image is a 2D (array) binding with
// ImageResource::remappedDepth1D, so Vulkan drivers without 1D depth images can back it. MSL heaps
// already hold 1D images as 2D textures one texel high (depth2d, depth2d_array; MslEntryPoint
// DepthShapes), so the function samples it as the 1D view it was: y at the centre of row 0, the
// slice from the second address component, the offset's y 0, no gather (MslAbi Version 21).
ImageResource OneDimensionalView(const ImageResource& mode) {
    auto view = mode;
    if (mode.remappedDepth1D) {
        if (mode.dimension == RdnaImageDimension::Dim2D) view.dimension = RdnaImageDimension::Dim1D;
        else if (mode.dimension == RdnaImageDimension::Dim2DArray) view.dimension = RdnaImageDimension::Dim1DArray;
    }
    return view;
}

void EmitMode(MslEmitterState& state, const IrValue& inst, const ImageResource& resource) {
    const auto& mem = state.Memory(inst);
    const auto info = ImageOpcodeInfoOf(inst.Opcode());
    const auto& address = ImageAddress(state, inst, inst.Argument(info.needsSampler ? 2 : 1));
    const auto view = OneDimensionalView(resource);
    const auto& mode = resource.remappedDepth1D ? view : resource;
    const Access access{inst, mem, mode, address, &resource};
    // Upstream caae92c12: FMASK images are uint images read from guest memory (Fmask32_S8_F8); the
    // identity Fmask8_S2_F1 mode is gone.
    if (mode.depthBits && mem.dataBits == 16u) state.Fail(inst, "depth bits require a 32-bit image result");
    if (info.access == ImageAccess::Atomic) {
        EmitAtomic(state, access);
        return;
    }
    switch (inst.Opcode()) {
    case IrOpcode::ImageQueryDimensions:
        EmitQueryDimensions(state, access);
        return;
    case IrOpcode::ImageQueryLod:
        EmitQueryLod(state, access);
        return;
    case IrOpcode::ImageRead:
        EmitRead(state, access);
        return;
    case IrOpcode::ImageWrite:
        EmitWrite(state, access);
        return;
    case IrOpcode::ImageSampleRaw:
    case IrOpcode::ImageGatherRaw:
        EmitSampling(state, access);
        return;
    default:
        state.Fail(inst, "has no MSL image emitter");
    }
}

// --- image and sampler tables (upstream ImageTableAbi, MslAbi Version 15) ------------------------
// SpirvImageEmitter.cpp EmitTableWord/EmitTableAccess: a table handle's dword offset selects a key of
// the driver's table map; its word is a record of the map (null, or a poison code with the flag
// set). Selected records give the mode, the T# dword 3 and the element of the texture (sampler)
// array. Every failure records fault reason ImageTable with the poison index and yields zero.

std::string TableMap(MslEmitterState& state) {
    if (state.prologue.find("aps5_tmap ") == std::string::npos) {
        const auto map = state.TableBuffer(DescriptorBindingKind::ImageTableMap, 0u);
        state.AddPrologue("device const uint* aps5_tmap = reinterpret_cast<device const uint*>(" + map + ".address);");
    }
    return "aps5_tmap";
}

struct TableWord {
    std::string code;
    std::string address;
    std::string poisoned;
};

std::string Poisoned(const std::string& value) {
    return "((" + value + " & " + U(ImageTableAbi::PoisonFlag) + ") != 0u)";
}

TableWord EmitTableWord(MslEmitterState& state, const IrValue& inst, std::uint32_t source, std::uint32_t table, std::size_t argument) {
    namespace Abi = ImageTableAbi;
    const IrValue* handle = inst.Argument(argument)->Resolve();
    const auto& sources = state.program.Resources().descriptorSources;
    if (handle->ArgumentCount() == 0u || source >= sources.size() || !sources[source].tableColumn.has_value()) state.Fail(inst, "has an image table handle without its column");
    const auto& column = *sources[source].tableColumn;
    const auto map = TableMap(state);
    const auto field = [&](std::uint32_t index) { return map + "[" + U(Abi::TableHeader(table) + index) + "]"; };
    const auto t = state.Temporary("aps5_tw");
    state.Line("const uint " + t + "_off = " + state.Value(handle->Argument(0)) + ";");
    state.Line("const uint " + t + "_rel = " + t + "_off - " + U(column.addend) + ";");
    state.Line("const uint " + t + "_rec = " + t + "_rel / " + U(column.stride) + ";");
    state.Line("const bool " + t + "_in = " + t + "_rel - " + t + "_rec * " + U(column.stride) + " == 0u && " + t + "_rec < " + field(Abi::TableKeyCount) + ";");
    state.Line("const uint " + t + "_mapped = " + map + "[" + t + "_in ? " + field(Abi::TableMapStart) + " + " + t + "_rec : 0u];");
    state.Line("const uint " + t + "_fault = " + field(Abi::TableFault) + ";");
    const auto base = "(ulong(" + field(Abi::TableBaseLow) + ") | (ulong(" + field(Abi::TableBaseHigh) + ") << 32))";
    const auto outsideSnapshot = U(Abi::PoisonCode(Abi::OutsideSnapshotPoison));
    TableWord word{t + "_code", t + "_addr", t + "_poisoned"};
    if (column.address) {
        // The record's address: base + (offset & ~3) + the sign-extended immediate (64-bit wrap).
        const auto immediate = static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(column.offset & ~3u)));
        state.Line("const uint " + word.code + " = " + t + "_fault != 0u ? " + t + "_fault : (" + t + "_in ? " + t + "_mapped : " + outsideSnapshot + ");");
        state.Line("const ulong " + word.address + " = " + base + " + ulong(" + t + "_off & ~3u) + " + MslHex(immediate) + "ul;");
    } else {
        // EmitScalarBufferDword: the dword at offset + immediate, inside the V#'s size.
        state.Line("const ulong " + t + "_sum = ulong(" + t + "_off) + " + MslHex(column.offset) + "ul;");
        state.Line("const uint " + t + "_byte = uint(" + t + "_sum) & ~3u;");
        state.Line("const bool " + t + "_range = (" + t + "_sum >> 32) == 0ul && (" + field(Abi::TableSizeHigh) + " != 0u || " + t + "_byte < (" + field(Abi::TableSizeLow) + " & ~3u));");
        state.Line("const uint " + word.code + " = " + t + "_fault != 0u ? " + t + "_fault : (" + t + "_in ? " + t + "_mapped : (" + t + "_range ? " + outsideSnapshot + " : " + field(Abi::TableOutsideCode) + "));");
        state.Line("const ulong " + word.address + " = " + base + " + ulong(" + t + "_byte);");
    }
    state.Line("const bool " + word.poisoned + " = " + Poisoned(word.code) + ";");
    return word;
}

void EmitTableFault(MslEmitterState& state, const IrValue& inst, const TableWord& word, const std::string& poison, const std::string& condition) {
    const auto reason = static_cast<std::uint32_t>(BdaAbi::FaultReason::ImageTable);
    state.Line("if (" + condition + ") " + MslRecordFault(state, reason, word.address, "(" + poison + " & " + U(~ImageTableAbi::PoisonFlag) + ")", MslB5Pc(inst)));
}

std::string TableZero(MslEmitterState& state, const IrValue& inst) {
    switch (inst.Type()) {
    case IrType::U32:
    case IrType::U64:
    case IrType::U32x4:
        return MslType(inst.Type()) + "(0u)";
    default:
        state.Fail(inst, "has no zero result for an image table access");
    }
}

struct TableAccessResult {
    std::string valid = "true";
    std::string flags = "0u";
    bool unsupported = false;
};

TableAccessResult EmitTableAccess(MslEmitterState& state, const IrValue& inst, const MemoryInfo& mem, const ImageResource& base, bool tableSampler) {
    namespace Abi = ImageTableAbi;
    const auto map = TableMap(state);
    const auto header = [&](std::uint32_t index) { return map + "[" + U(index) + "]"; };
    const auto exec = "bool(" + state.Arg(inst, inst.ArgumentCount() - 1u) + ")";
    TableAccessResult result;
    std::string live = "true";
    state.tableAccess = {};
    if (base.table != NoTable) {
        const auto word = EmitTableWord(state, inst, base.source, base.table, 0u);
        EmitTableFault(state, inst, word, word.code, exec + " && " + word.poisoned);
        if (base.tableOperation == TableOperation::Unsupported || state.program.Info().runtimeImageModes.at(mem.resource).empty()) {
            if (inst.Type() != IrType::Void) state.Define(inst, TableZero(state, inst));
            result.unsupported = true;
            return result;
        }
        const auto t = state.Temporary("aps5_ti");
        state.Line("const bool " + t + "_sel = " + word.code + " != " + U(Abi::NullCode) + " && !" + word.poisoned + ";");
        state.Line("const uint " + t + "_rec = " + t + "_sel ? " + word.code + " : 0u;");
        state.Line("const uint " + t + "_flags = " + map + "[" + t + "_rec + " + U(Abi::ImageRecordFlags) + "];");
        state.Line("const uint " + t + "_d3 = " + map + "[" + t + "_rec + " + U(Abi::ImageRecordDword3) + "];");
        state.Line("const uint " + t + "_elem = " + map + "[" + t + "_rec + " + U(Abi::ImageRecordElement) + "];");
        state.Line("const bool " + t + "_rej = " + t + "_sel && " + Poisoned(t + "_flags") + ";");
        EmitTableFault(state, inst, word, t + "_flags", exec + " && " + t + "_rej");
        state.Line("const bool " + t + "_bounds = " + t + "_elem < " + header(Abi::HeaderElementCount) + ";");
        EmitTableFault(state, inst, word, U(Abi::OutsideSnapshotPoison), exec + " && " + t + "_sel && !" + t + "_rej && !" + t + "_bounds");
        state.Line("const bool " + t + "_chk = " + t + "_sel && !" + t + "_rej && " + t + "_bounds;");
        state.Line("const uint " + t + "_drv = " + map + "[" + t + "_chk ? " + header(Abi::HeaderElementFaults) + " + " + t + "_elem : 0u];");
        state.Line("const bool " + t + "_drej = " + t + "_chk && " + Poisoned(t + "_drv") + ";");
        EmitTableFault(state, inst, word, t + "_drv", exec + " && " + t + "_drej");
        state.Line("const uint " + t + "_at = " + header(Abi::HeaderElementBase) + " + " + t + "_elem;");
        result.valid = "(" + t + "_chk && !" + t + "_drej)";
        result.flags = t + "_flags";
        live = t + "_sel";
        state.tableAccess.element = t + "_at";
        state.tableAccess.dword3 = t + "_d3";
    }
    if (tableSampler) {
        const auto& sampler = state.program.Info().samplers.at(mem.sampler);
        const auto word = EmitTableWord(state, inst, sampler.source, sampler.table, 1u);
        const auto active = exec + " && " + live;
        EmitTableFault(state, inst, word, word.code, active + " && " + word.poisoned);
        const auto t = state.Temporary("aps5_ts");
        state.Line("const bool " + t + "_sel = !" + word.poisoned + ";");
        state.Line("const uint " + t + "_rec = " + t + "_sel ? " + word.code + " : 0u;");
        state.Line("const uint " + t + "_flags = " + map + "[" + t + "_rec + " + U(Abi::SamplerRecordFlags) + "];");
        state.Line("const uint " + t + "_elem = " + map + "[" + t + "_rec + " + U(Abi::SamplerRecordElement) + "];");
        state.Line("const bool " + t + "_rej = " + t + "_sel && " + Poisoned(t + "_flags") + ";");
        EmitTableFault(state, inst, word, t + "_flags", active + " && " + t + "_rej");
        state.Line("const bool " + t + "_bounds = " + t + "_elem + " + U(Abi::SamplerEntryElements) + " <= " + header(Abi::HeaderSamplerCount) + ";");
        EmitTableFault(state, inst, word, U(Abi::OutsideSnapshotPoison), active + " && " + t + "_sel && !" + t + "_rej && !" + t + "_bounds");
        state.Line("const bool " + t + "_chk = " + t + "_sel && !" + t + "_rej && " + t + "_bounds;");
        state.Line("const uint " + t + "_drv = " + map + "[" + t + "_chk ? " + header(Abi::HeaderSamplerFaults) + " + (" + t + "_elem >> 2u) : 0u];");
        state.Line("const bool " + t + "_drej = " + t + "_chk && " + Poisoned(t + "_drv") + ";");
        EmitTableFault(state, inst, word, t + "_drv", active + " && " + t + "_drej");
        result.valid = "(" + result.valid + " && " + t + "_chk && !" + t + "_drej)";
        state.tableAccess.samplerElement = t + "_elem";
    }
    state.tableAccess.active = true;
    return result;
}

// A direct image's runtime modes, selected by its function constant (ImageBase + resource).
void EmitDirectModes(MslEmitterState& state, const IrValue& inst, const MemoryInfo& mem, const ImageResource& base) {
    const auto modes = ResourceMaterializer::RuntimeImageModes(base);
    if (modes.size() == 1u) {
        state.Line("{");
        state.Indent();
        EmitMode(state, inst, modes.front());
        state.Outdent();
        state.Line("}");
        return;
    }
    const auto selectorId = PipelineSpecialization::ImageBase + mem.resource * PipelineSpecialization::ImageWords;
    const auto selector = state.FunctionConstant(selectorId, 0u);
    // The result keeps its previous value for an invalid selector (the SPIR-V path: undefined).
    state.Line("switch (" + selector + ") {");
    for (std::uint32_t index = 0u; index < modes.size(); ++index) {
        state.Line("case " + U(index) + ": {");
        state.Indent();
        state.imageModeCase = {selectorId, index};
        EmitMode(state, inst, modes[index]);
        state.imageModeCase = {0u, ~0u};
        state.Line("break;");
        state.Outdent();
        state.Line("}");
    }
    state.Line("default:");
    state.Line("    break;");
    state.Line("}");
}

// SpirvImageEmitter.cpp EmitImage with tables: a table image switches on the selected record's mode
// (an invalid access takes no case and reads zero); a direct image read through a table sampler
// runs only when the sampler entry is valid.
void EmitTableImage(MslEmitterState& state, const IrValue& inst, const MemoryInfo& mem, const ImageResource& base, bool tableSampler) {
    state.Line("{");
    state.Indent();
    const auto access = EmitTableAccess(state, inst, mem, base, tableSampler);
    if (!access.unsupported) {
        if (inst.Type() != IrType::Void) state.Define(inst, TableZero(state, inst));
        if (base.table != NoTable) {
            const auto& modes = state.program.Info().runtimeImageModes.at(mem.resource);
            const auto mode = "((" + access.flags + " >> " + U(ImageTableAbi::ModeShift) + ") & " + U(ImageTableAbi::ModeMask) + ")";
            state.Line("switch (" + access.valid + " ? " + mode + " : 0xffffffffu) {");
            for (std::uint32_t index = 0u; index < modes.size(); ++index) {
                if (!TableModeEmitted(modes[index])) continue;
                state.Line("case " + U(index) + ": {");
                state.Indent();
                EmitMode(state, inst, modes[index]);
                state.Line("break;");
                state.Outdent();
                state.Line("}");
            }
            state.Line("default:");
            state.Line("    break;");
            state.Line("}");
        } else {
            state.Line("if (" + access.valid + ") {");
            state.Indent();
            EmitDirectModes(state, inst, mem, base);
            state.Outdent();
            state.Line("}");
        }
    }
    state.tableAccess = {};
    state.Outdent();
    state.Line("}");
}

void EmitImage(MslEmitterState& state, const IrValue& inst) {
    state.RequirePrelude(MslPreludeChunk::Image);
    const auto& mem = state.Memory(inst);
    const auto& base = state.program.Resources().info.images.at(mem.resource);
    // Upstream d9e47d1e9/34525fdf7/c794273ab/55eefe9c5 replaced the bindless image search
    // (ImageResource::indirectRoot) with image and sampler tables: a table map (ImageTableAbi) the
    // driver materializes from the guest records, selecting array elements and image modes at run
    // time and reporting bad entries through the fault buffer (MslAbi Version 15).
    const auto& samplers = state.program.Resources().info.samplers;
    const bool tableSampler = ImageOpcodeInfoOf(inst.Opcode()).needsSampler && mem.sampler < samplers.size() && samplers[mem.sampler].table != NoTable;
    if (base.table != NoTable || tableSampler) {
        EmitTableImage(state, inst, mem, base, tableSampler);
        return;
    }
    EmitDirectModes(state, inst, mem, base);
}


// image_bvh_intersect_ray / image_bvh64_intersect_ray (SpirvImageEmitter.cpp EmitImageBvhIntersectRay):
// one node of a BVH read through the BDA page table, a triangle (types 0, 1) or a box node of
// fp16 (4) or fp32 (5) bounds; inactive or invalid nodes return four 0xffffffff.
constexpr std::string_view BvhText = R"MSL(
static inline float aps5_bvh_f16(uint packed, uint index) {
    return float(as_type<half>(ushort(index % 2u == 0u ? (packed & 0xffffu) : (packed >> 16u))));
}

static inline void aps5_bvh_read(Aps5Bda bda, ulong address, uint count, uint pc, thread uint* d) {
    const ulong physical = aps5_bda_lookup(bda, address, count * 4u, pc, 1u, true);
    const bool mapped = physical != 0ul && (physical & 3ul) == 0ul;
    for (uint i = 0u; i < count; ++i) d[i] = mapped ? reinterpret_cast<device const uint*>(physical)[i] : 0u;
}

static inline float aps5_bvh_nanmax(float a, float b) { return (isnan(a) || a > b) ? a : b; }
static inline float aps5_bvh_nanmin(float a, float b) { return (isnan(a) || a < b) ? a : b; }

static uint4 aps5_bvh_intersect(Aps5Bda bda, uint4 word, ulong node, uint nodeLow, bool node64, float extent, float3 origin, float3 direction, float3 inverse, uint pc) {
    const uint invalid = 0xffffffffu;
    const uint type = nodeLow & 7u;
    const ulong nodeIndex = node >> 3u;
    const ulong baseUnits = ulong(word.x) | (ulong(word.y & 0xffu) << 32u);
    const ulong lastNode = ulong(word.z) | (ulong(word.w & 0x3ffu) << 32u);
    const uint boxGrow = (word.y >> 23u) & 0xffu;
    const bool boxSort = (word.y & 0x80000000u) != 0u;
    const bool barycentrics = (word.w & (1u << 24u)) != 0u;
    const ulong nodeAddress = (baseUnits << 8u) + ((node & ~7ul) << 3u);
    const bool baseValid = node64 || baseUnits != 0ul;
    const bool valid = baseValid && nodeIndex <= lastNode;
    const bool isTriangle = valid && type <= 1u;
    const bool isBox16 = valid && type == 4u;
    const bool isBox32 = baseValid && nodeIndex < lastNode && type == 5u;
    if (isBox32 || isBox16) {
        uint d[28];
        const bool fp16 = !isBox32;
        aps5_bvh_read(bda, nodeAddress, fp16 ? 16u : 28u, pc, d);
        const float growFactor = 1.0f + float(boxGrow) * as_type<float>(0x33800000u);
        uint4 children;
        float4 keys;
        for (uint child = 0u; child < 4u; ++child) {
            float enter = 0.0f;
            float leave = 0.0f;
            for (uint axis = 0u; axis < 3u; ++axis) {
                const uint low = child * 6u + axis;
                const uint high = low + 3u;
                const float boundLow = fp16 ? aps5_bvh_f16(d[4u + low / 2u], low) : as_type<float>(d[4u + low]);
                const float boundHigh = fp16 ? aps5_bvh_f16(d[4u + high / 2u], high) : as_type<float>(d[4u + high]);
                const float planeMin = (boundLow - origin[axis]) * inverse[axis];
                const float planeMax = (boundHigh - origin[axis]) * inverse[axis];
                const bool positive = inverse[axis] >= 0.0f;
                const float near = positive ? planeMin : planeMax;
                const float far = positive ? planeMax : planeMin;
                enter = axis == 0u ? near : aps5_bvh_nanmax(enter, near);
                leave = axis == 0u ? far : aps5_bvh_nanmin(leave, far);
            }
            const bool nan = isnan(enter) || isnan(leave);
            const float minT = nan ? as_type<float>(0x7f800000u) : (enter > 0.0f ? enter : 0.0f);
            const float maxT = nan ? as_type<float>(0xff800000u) : (leave < extent ? leave : extent);
            const bool hit = minT <= maxT * growFactor;
            children[child] = hit ? d[child] : invalid;
            keys[child] = minT;
        }
        uint4 sorted = children;
        const uint2 pairs[5] = {uint2(0u, 2u), uint2(1u, 3u), uint2(0u, 1u), uint2(2u, 3u), uint2(1u, 2u)};
        for (uint index = 0u; index < 5u; ++index) {
            const uint a = pairs[index].x;
            const uint b = pairs[index].y;
            const bool swap = (sorted[b] != invalid && keys[b] < keys[a]) || sorted[a] == invalid;
            const uint sa = swap ? sorted[b] : sorted[a];
            const uint sb = swap ? sorted[a] : sorted[b];
            const float ka = swap ? keys[b] : keys[a];
            const float kb = swap ? keys[a] : keys[b];
            sorted[a] = sa;
            sorted[b] = sb;
            keys[a] = ka;
            keys[b] = kb;
        }
        return boxSort ? sorted : children;
    }
    if (!isTriangle) return uint4(invalid);
    uint d[16];
    aps5_bvh_read(bda, nodeAddress, 16u, pc, d);
    const bool second = type == 1u;
    float3 v1;
    float3 v2;
    float3 v3;
    for (uint axis = 0u; axis < 3u; ++axis) {
        v1[axis] = second ? as_type<float>(d[3u + axis]) : as_type<float>(d[axis]);
        v2[axis] = second ? as_type<float>(d[9u + axis]) : as_type<float>(d[3u + axis]);
        v3[axis] = as_type<float>(d[6u + axis]);
    }
    const float3 e1 = v2 - v1;
    const float3 e2 = v3 - v1;
    const float3 e3 = origin - v1;
    const float3 s1 = float3(direction.y * e2.z - direction.z * e2.y, direction.z * e2.x - direction.x * e2.z, direction.x * e2.y - direction.y * e2.x);
    const float3 s2 = float3(e3.y * e1.z - e3.z * e1.y, e3.z * e1.x - e3.x * e1.z, e3.x * e1.y - e3.y * e1.x);
    float tNum = (e2.x * s2.x + e2.y * s2.y) + e2.z * s2.z;
    float tDenom = (s1.x * e1.x + s1.y * e1.y) + s1.z * e1.z;
    const float iNum = (e3.x * s1.x + e3.y * s1.y) + e3.z * s1.z;
    const float jNum = (direction.x * s2.x + direction.y * s2.y) + direction.z * s2.z;
    const float t = tNum / tDenom;
    const float u = iNum / tDenom;
    const float v = jNum / tDenom;
    const bool missed = u < 0.0f || u > 1.0f || v < 0.0f || u + v > 1.0f || t < 0.0f;
    tNum = missed ? as_type<float>(0x7f800000u) : tNum;
    tDenom = missed ? 1.0f : tDenom;
    const uint triangleId = d[15];
    const uint shift = type << 3u;
    const float rest = (tDenom - iNum) - jNum;
    const uint source0 = (triangleId >> shift) & 3u;
    const uint source2 = (triangleId >> (shift + 2u)) & 3u;
    const float bary0 = source0 == 1u ? iNum : (source0 == 2u ? jNum : rest);
    const float bary2 = source2 == 1u ? iNum : (source2 == 2u ? jNum : rest);
    return uint4(as_type<uint>(tNum), as_type<uint>(tDenom), barycentrics ? as_type<uint>(bary0) : triangleId, barycentrics ? as_type<uint>(bary2) : (missed ? 0u : 1u));
}
)MSL";

void EmitBvhIntersectRay(MslEmitterState& state, const IrValue& inst) {
    const auto& mem = state.Memory(inst);
    const auto& address = ImageAddress(state, inst, inst.Argument(1));
    const auto bda = MslBdaContext(state);
    if (state.declarations.find("aps5_bvh_intersect") == std::string::npos) state.AddDeclaration(BvhText);
    const auto component = [&](std::uint32_t index) { return state.Value(address.Argument(index)); };
    const bool a16 = (mem.imageSampleFlags & RdnaImageSampleFlagA16) != 0u;
    const bool node64 = mem.addressIsFull;
    const std::uint32_t ray = node64 ? 2u : 1u;
    const auto node = node64 ? "(ulong(" + component(0) + ") | (ulong(" + component(1) + ") << 32u))" : "ulong(" + component(0) + ")";
    std::array<std::string, 3> origin;
    std::array<std::string, 3> direction;
    std::array<std::string, 3> inverse;
    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
        origin[axis] = "as_type<float>(" + component(ray + 1u + axis) + ")";
        if (!a16) {
            direction[axis] = "as_type<float>(" + component(ray + 4u + axis) + ")";
            inverse[axis] = "as_type<float>(" + component(ray + 7u + axis) + ")";
            continue;
        }
        const auto half = [&](std::uint32_t index) { return "aps5_f16_value(" + std::string(index % 2u == 0u ? "" : "(") + component(ray + 4u + index / 2u) + (index % 2u == 0u ? "" : " >> 16u)") + ")"; };
        direction[axis] = half(axis);
        inverse[axis] = half(3u + axis);
    }
    const auto vector = [](const std::array<std::string, 3>& values) { return "float3(" + values[0] + ", " + values[1] + ", " + values[2] + ")"; };
    state.Define(inst, state.Arg(inst, 2) + " ? aps5_bvh_intersect(" + bda + ", " + state.Arg(inst, 0) + ", " + node + ", " + component(0) + ", " + MslBool(node64) + ", as_type<float>(" + component(ray) + "), " + vector(origin) + ", " + vector(direction) + ", " + vector(inverse) + ", " + MslB5Pc(inst) + ") : uint4(0xffffffffu)");
}

}

bool EmitMslImageInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::GetImageResource:
    case IrOpcode::GetSamplerResource:
    case IrOpcode::MakeImageAddress:
        // Handles and packed addresses: the image instructions read their operands directly.
        return true;
    case IrOpcode::ImageBvhIntersectRay:
        state.RequirePrelude(MslPreludeChunk::Image);
        EmitBvhIntersectRay(state, instruction);
        return true;
    default:
        break;
    }
    const auto info = ImageOpcodeInfoOf(instruction.Opcode());
    if (info.access == ImageAccess::None) return false;
    EmitImage(state, instruction);
    return true;
}

MslPreludeEntry MslImagePrelude(MslPreludeChunk chunk) {
    if (chunk != MslPreludeChunk::Image) throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not an image chunk (work package B4)");
    static constexpr std::array<MslPreludeChunk, 1> OnCommon{MslPreludeChunk::Common};
    return {ImageText, OnCommon};
}

}
