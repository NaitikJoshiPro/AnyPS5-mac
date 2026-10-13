// agc_metal_msl_ms_storage (MslAbi Version 6): emulated multisampled storage images. Metal shaders
// cannot write texture2d_ms, so storage views of 2D MSAA (array) images are texture2d_array textures
// of layers * samples slices, sample s of layer l at slice l * samples + s (MslImage.cpp,
// EmulatedMultisampleCoord; DECISIONS D21).
// - CPU: the MIMG stores and atomics of tests/execution/PreparedShaders.cpp (PrepareMultisampledStorage)
//   plus a 64-bit atomic are recompiled for an MSL target that advertises
//   CapabilityStorageImageMultisample; the storage heap must be a read_write texture2d_array, the
//   sample index the program's constant, and the slice layer * samples + sample. Without the
//   capability preparation fails as on the SPIR-V backend.
// - GPU (tiny): the emitted libraries compile with Metal; the store and 32-bit atomic kernels (image
//   mode 0) run one 8x8 workgroup on 8x8 2D arrays of 2 layers * 4 samples bound through a minimal
//   descriptor table, and only the addressed sample's slice may change. The 64-bit atomic (driver
//   lock table) is compiled only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "BdaAbi.hpp"
#include "MslAbi.hpp"
#include "PipelineSpecialization.hpp"
#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace Abi = ShaderRecompiler::MslAbi;
constexpr int Skipped = 77;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool Has(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

ShaderRecompiler::SpirvTarget MetalTarget(bool multisample) {
    // SpirvTarget::supportedCapabilities is a span: the lists outlive every request.
    static std::array<std::vector<std::uint32_t>, 2> lists;
    auto& capabilities = lists[multisample ? 1u : 0u];
    capabilities.clear();
    for (const auto capability : {spv::CapabilityShader, spv::CapabilityInt64, spv::CapabilityInt64Atomics, spv::CapabilityInt64ImageEXT, spv::CapabilityInt16, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformArithmetic, spv::CapabilitySignedZeroInfNanPreserve, spv::CapabilityImageQuery, spv::CapabilityFloat64}) {
        capabilities.push_back(static_cast<std::uint32_t>(capability));
    }
    if (multisample) capabilities.push_back(static_cast<std::uint32_t>(spv::CapabilityStorageImageMultisample));
    std::ranges::sort(capabilities);
    ShaderRecompiler::SpirvTarget target {0u, 0u, 32u, ShaderRecompiler::BdaAbi::Version, capabilities, {}, false, {1024u, 1024u, 1024u}, 1024u, 32768u, {}, {}};
    target.backend = ShaderRecompiler::ShaderBackend::Msl;
    return target;
}

// tests/execution/PreparedShaders.cpp PrepareMultisampledStorage: v0..v3 = x, y, (layer,) sample
// (v_bfe_u32 of the local id for x/y, v2/v3 constants), v4..v7 = s8..s11, then the MIMG op on
// T# s[0:7] (user data, zero at recompile time).
struct Variant {
    const char* name;
    std::uint32_t mimg;
    bool layered;
    bool atomic;
    std::uint32_t layer;
    std::uint32_t sample;
    bool atomic64 = false;
};

std::vector<std::uint32_t> Code(const Variant& variant) {
    std::vector<std::uint32_t> code{0xd7460000u, 0x0401060cu, 0xd7460001u, 0x0405060du, 0x7e04020eu, 0x7e060280u, 0x7e080208u, 0x7e0a0209u, 0x7e0c020au, 0x7e0e020bu};
    // Address: x, y, (layer,) sample. v_mov_b32 v2/v3, <inline constant>.
    code[4] = 0x7e040280u + (variant.layered ? variant.layer : variant.sample);
    code[5] = 0x7e060280u + (variant.layered ? variant.sample : 0u);
    code.insert(code.end(), {variant.mimg, 0x00000400u, 0xbf810000u});
    return code;
}

ShaderRecompiler::RecompileResult Recompile(std::span<const std::uint32_t> code, bool multisample) {
    ShaderRecompiler::RecompileRequest request{};
    request.shader = {ShaderRecompiler::ShaderStage::Compute, 0x30000u, code, 0, {}};
    static const std::array<std::uint32_t, 16> users{};
    request.context.userData = users;
    request.context.waveSize = 32;
    request.context.compute = ShaderRecompiler::ShaderComputeStageInfo{{8, 8, 1}, 0, {true, true, false}, false, 2, {}};
    request.target = MetalTarget(multisample);
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

std::string Check(const Variant& variant, const ShaderRecompiler::RecompileResult& result) {
    const std::string name(variant.name);
    Require(!result.msl.empty(), name + ": no MSL");
    Require(result.msl.Reflection().abiVersion == Abi::Version && Abi::Version >= 6u, name + ": not an MslAbi v6 library");
    const std::string source(result.msl.Source());
    if (std::getenv("APS5_MS_STORAGE_DUMP") != nullptr) std::cout << "=== " << name << "\n" << source << '\n';
    // The storage heap of the MSAA dimension is a writable 2D array; no texture2d_ms is declared
    // for a storage heap (sampled heaps may still be texture2d_ms).
    const std::string scalar = variant.atomic64 ? "ulong" : variant.atomic ? "uint" : "float";
    Require(Has(source, "texture2d_array<" + scalar + ", access::read_write>"), name + ": the multisampled storage heap is not a read_write texture2d_array");
    Require(!Has(source, "texture2d_ms<" + scalar + ", access::read>") && !Has(source, "texture2d_ms_array<" + scalar + ", access::read>"), name + ": a storage heap is still a read-only texture2d_ms");
    // Sample count: function constant SampleCountBase + resource 0, default 1.
    const auto samples = "aps5_fc" + std::to_string(ShaderRecompiler::PipelineSpecialization::SampleCountBase);
    Require(Has(source, samples), name + ": the sample count is not the SampleCountBase function constant");
    Require(Has(source, "_n = " + samples + ";"), name + ": the sample count is not read from the function constant");
    // The sample index is the program's constant (the emitter prints literals as 0x<hex>u).
    const auto literal = [](std::uint32_t value) {
        char text[16];
        std::snprintf(text, sizeof(text), "0x%xu;", value);
        return std::string(text);
    };
    Require(Has(source, "_s = " + literal(variant.sample)), name + ": the sample index is not the program's constant " + std::to_string(variant.sample));
    Require(Has(source, "_l * ") && Has(source, "_n + "), name + ": no layer * samples + sample slice");
    Require(Has(source, "get_array_size() / "), name + ": the layer is not bounded by slices / samples");
    if (variant.atomic64) Require(Has(source, "aps5_atomic64_lock(") && Has(source, ".atomic_max("), name + ": the 64-bit atomic does not take the lock table on the backing");
    else if (variant.atomic) Require(Has(source, ".atomic_exchange("), name + ": the atomic swap is not a texture atomic on the backing");
    else Require(Has(source, ".write("), name + ": no texture write");
    return source;
}

// --- GPU part --------------------------------------------------------------------------------------

std::string Describe(NSError* error) {
    return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

id<MTLFunction> Compile(id<MTLDevice> device, const std::string& name, const ShaderRecompiler::RecompileResult& result, std::uint32_t samples) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = static_cast<MTLLanguageVersion>((Abi::LanguageMajor << 16u) | Abi::LanguageMinor);
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    const std::string source(result.msl.Source());
    id<MTLLibrary> library = [device newLibraryWithSource:@(source.c_str()) options:options error:&error];
    if (library == nil) {
        std::cerr << source << '\n';
        throw std::runtime_error(name + ": MSL does not compile: " + Describe(error));
    }
    // The recompile-time constants, then image mode 0 of resource 0 and its sample count.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> values;
    for (const auto& constant : result.specialization) values.emplace_back(constant.id, constant.value);
    values.emplace_back(ShaderRecompiler::PipelineSpecialization::ImageBase, 0u);
    values.emplace_back(ShaderRecompiler::PipelineSpecialization::SampleCountBase, samples);
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    for (const auto& [id, value] : values) {
        const std::uint32_t word = value;
        [constants setConstantValue:&word type:MTLDataTypeUInt atIndex:id];
    }
    id<MTLFunction> function = [library newFunctionWithName:@(Abi::EntryPoint) constantValues:constants error:&error];
    Require(function != nil, name + ": no specializable entry point: " + Describe(error));
    return function;
}

// The Aps5Table fields (heap kinds, in declaration order) of the library, and the field image mode 0
// writes.
std::vector<std::uint32_t> TableKinds(const std::string& source) {
    std::vector<std::uint32_t> kinds;
    const auto begin = source.find("struct Aps5Table {");
    const auto end = source.find("};", begin);
    Require(begin != std::string::npos && end != std::string::npos, "no Aps5Table");
    for (auto at = source.find("> images", begin); at < end; at = source.find("> images", at + 1)) kinds.push_back(static_cast<std::uint32_t>(std::stoul(source.substr(at + 8))));
    return kinds;
}

std::uint32_t ModeZeroKind(const std::string& source) {
    auto at = source.find("case 0u: {");
    at = source.find("aps5_table.images", at == std::string::npos ? source.find("kernel void aps5_main") : at);
    Require(at != std::string::npos, "no image access in mode 0");
    return static_cast<std::uint32_t>(std::stoul(source.substr(at + 17)));
}

constexpr std::uint32_t Extent = 8u;
constexpr std::uint32_t Layers = 2u;
constexpr std::uint32_t Samples = 4u;

// Runs mode 0 of the store / 32-bit atomic program on one 8x8 workgroup over 2D arrays of
// Layers * Samples slices and checks that exactly the addressed sample's slice holds the data.
void Run(id<MTLDevice> device, id<MTLCommandQueue> queue, const Variant& variant, const ShaderRecompiler::RecompileResult& result) {
    const std::string name(variant.name);
    const std::string source(result.msl.Source());
    id<MTLFunction> function = Compile(device, name, result, Samples);
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    Require(pipeline != nil, name + ": no compute pipeline: " + Describe(error));
    const auto kinds = TableKinds(source);
    const auto written = ModeZeroKind(source);
    const auto storageIndex = written - ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
    const bool arrayed = storageIndex % 7u == 5u;
    const bool isFloat = storageIndex / 7u == 0u;
    Require(storageIndex % 7u == 4u || arrayed, name + ": mode 0 does not access a multisampled storage heap");
    std::vector<id<MTLTexture>> textures;
    id<MTLTexture> target = nil;
    for (const auto kind : kinds) {
        const auto index = kind - ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
        descriptor.textureType = MTLTextureType2DArray;
        descriptor.pixelFormat = index / 7u == 0u ? MTLPixelFormatRGBA32Float : MTLPixelFormatR32Uint;
        descriptor.width = Extent;
        descriptor.height = Extent;
        descriptor.arrayLength = Layers * Samples;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | (index / 7u == 0u ? 0u : MTLTextureUsageShaderAtomic);
        id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
        Require(texture != nil, name + ": cannot create a 2D-array texture");
        const std::vector<std::uint8_t> zero(Extent * Extent * 16u, 0u);
        for (std::uint32_t slice = 0; slice < Layers * Samples; ++slice) [texture replaceRegion:MTLRegionMake2D(0, 0, Extent, Extent) mipmapLevel:0 slice:slice withBytes:zero.data() bytesPerRow:Extent * (index / 7u == 0u ? 16u : 4u) bytesPerImage:0];
        textures.push_back(texture);
        if (kind == written) target = texture;
    }
    Require(target != nil, name + ": the written heap is not in the table");
    std::vector<std::uint64_t> table;
    for (id<MTLTexture> texture : textures) table.push_back(texture.gpuResourceID._impl);
    // User data s0..s15: s8..s11 = the stored data (v4..v7), s12/s13 = 0 (workgroup origin).
    // The user data start at push dword PushDataOffset (function constant, recompile-time value).
    std::uint32_t base = 0u;
    for (const auto& constant : result.specialization) {
        if (constant.id == ShaderRecompiler::PipelineSpecialization::PushDataOffset) base = constant.value;
    }
    Require(base + 16u <= 64u, name + ": unexpected push data offset");
    std::array<std::uint32_t, 64> push{};
    const std::array<float, 4> floats{1.5f, 2.5f, 3.5f, 4.5f};
    const std::array<std::uint32_t, 4> uints{0x1234u, 0x5678u, 0x9abcu, 0xdef0u};
    for (std::uint32_t component = 0; component < 4u; ++component) push[base + 8u + component] = isFloat ? std::bit_cast<std::uint32_t>(floats[component]) : uints[component];
    id<MTLBuffer> tableBuffer = [device newBufferWithBytes:table.data() length:table.size() * sizeof(std::uint64_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> pushBuffer = [device newBufferWithBytes:push.data() length:sizeof(push) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:tableBuffer offset:0 atIndex:Abi::Index(Abi::BufferIndex::DescriptorTable)];
    [encoder setBuffer:pushBuffer offset:0 atIndex:Abi::Index(Abi::BufferIndex::PushData)];
    for (id<MTLTexture> texture : textures) [encoder useResource:texture usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(Extent, Extent, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    Require(commands.status == MTLCommandBufferStatusCompleted, name + ": the dispatch failed: " + Describe(commands.error));
    const auto slice = (arrayed ? variant.layer * Samples : 0u) + variant.sample;
    for (std::uint32_t check = 0; check < Layers * Samples; ++check) {
        std::vector<std::uint32_t> texels(Extent * Extent * 4u, 0u);
        [target getBytes:texels.data() bytesPerRow:Extent * (isFloat ? 16u : 4u) bytesPerImage:0 fromRegion:MTLRegionMake2D(0, 0, Extent, Extent) mipmapLevel:0 slice:check];
        for (std::uint32_t texel = 0; texel < Extent * Extent; ++texel) {
            for (std::uint32_t component = 0; component < (isFloat ? 4u : 1u); ++component) {
                const auto value = texels[texel * (isFloat ? 4u : 1u) + component];
                const auto expected = check != slice ? 0u : push[base + 8u + component];
                if (value != expected) {
                    throw std::runtime_error(name + ": heap " + std::to_string(written) + " slice " + std::to_string(check) + " texel " + std::to_string(texel) + " component " + std::to_string(component) + " is 0x" + std::to_string(value) + ", expected " + std::to_string(expected) + " (sample " + std::to_string(variant.sample) + " of layer " + std::to_string(variant.layer) + " is slice " + std::to_string(slice) + ")");
                }
            }
        }
    }
    std::printf("%s: heap %u, only slice %u written (%s)\n", variant.name, written, slice, arrayed ? "2D MSAA array view" : "2D MSAA view");
}

}

int main() {
    try {
        const std::array<Variant, 6> variants{{
            {"store, 2D MSAA array, layer 0 sample 0", 0xf0200f38u, true, false, 0u, 0u},
            {"store, 2D MSAA array, layer 1 sample 3", 0xf0200f38u, true, false, 1u, 3u},
            {"store, 2D MSAA, sample 2", 0xf0200f30u, false, false, 0u, 2u},
            {"atomic swap, 2D MSAA array, layer 1 sample 3", 0xf03c0138u, true, true, 1u, 3u},
            {"atomic swap, 2D MSAA, sample 2", 0xf03c0130u, false, true, 0u, 2u},
            {"64-bit atomic swap, 2D MSAA array, layer 1 sample 3", 0xf03c0338u, true, true, 1u, 3u, true},
        }};
        std::vector<ShaderRecompiler::RecompileResult> results;
        for (const auto& variant : variants) {
            const auto code = Code(variant);
            results.push_back(Recompile(code, true));
            Check(variant, results.back());
            try {
                static_cast<void>(Recompile(code, false));
                throw std::logic_error(std::string(variant.name) + ": recompiled without CapabilityStorageImageMultisample");
            } catch (const std::runtime_error& error) {
                Require(Has(error.what(), "storage image multisampling is unavailable on the target device"), std::string(variant.name) + ": unexpected failure without the capability: " + error.what());
            }
        }
        std::puts("multisampled storage MSL shape: 6 programs ok");
        if (std::getenv("APS5_MS_STORAGE_CPU_ONLY") != nullptr) return 0;
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
                std::puts("skipped, the device has no Metal 4 support");
                return Skipped;
            }
            for (std::size_t index = 0; index < variants.size(); ++index) static_cast<void>(Compile(device, variants[index].name, results[index], Samples));
            std::puts("multisampled storage MSL compiles with Metal (4 samples)");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            Require(queue != nil, "no command queue");
            // The 64-bit atomic needs the driver's lock table: covered by the compile above only.
            for (std::size_t index = 0; index < variants.size(); ++index) {
                if (!variants[index].atomic64) Run(device, queue, variants[index], results[index]);
            }
        }
        std::puts("msl multisampled storage check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
