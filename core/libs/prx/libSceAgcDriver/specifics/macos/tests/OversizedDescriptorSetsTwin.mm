// Metal twin of tests/execution/OversizedDescriptorSets.cpp (agc_driver_oversized_descriptor_sets), swapped in by
// twins/D5-cache-twins.cmake. The upstream file is compiled here unchanged with its main renamed, for its constants
// (Buffers = 4097 storage buffers, one past a Vulkan chain pool's 4096, Spacing) and expected values (Input).
// Upstream checks two things, both carried over with its expected values:
// 1. DescriptorCache: a set of 4097 storage buffers gets a dedicated pool of its own (two such sets, two pools, no
//    chain pool), and again after the others were freed. On Metal the pools and sets are tokens
//    (MetalVulkanImages.mm) under the same bookkeeping (ShaderResources.mm); the limits are the ones the Metal image
//    context gives every ShaderResources build (MetalTextures.mm ImageCache::Attach).
// 2. A dispatch reads the last two of the 4097 buffers through the set and writes their sum through the first. The
//    Metal device does not bind Vulkan sets: it writes each dispatch's descriptor table (MslAbi, dense BufferEntry
//    records) into the batch's upload ring and binds it at BufferIndex::DescriptorTable (MetalDevice.mm WriteTable).
//    So the twin lays out a 4097-entry guest-buffer table with MslAbi::LayoutTable, uploads it through a one-shot
//    batch's CommandStream (64 KiB of table, past any setBytes limit), and runs a kernel made of the MSL backend's own
//    text: its Common and RuntimeBuffer prelude chunks and MslTableDeclaration's Aps5Table, reading with
//    aps5_buffer_read and writing with aps5_buffer_write as emitted guest buffer loads and stores do.

#define main AgcDriverOversizedDescriptorSetsVulkanMain
#include "../../../tests/execution/OversizedDescriptorSets.cpp"
#undef main

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalCommandStream.h"
#include "../src/MetalImages.h"
#include "MetalBackend/MslEmitterState.hpp"
#include "MetalTestDevice.h"
#include "MslAbi.hpp"
#include "RuntimeAbi.hpp"
#include <array>

namespace MetalOversizedTwin {

using namespace AgcDriver::Graphics;
using ShaderRecompiler::MslPrelude;
using ShaderRecompiler::MslPreludeChunk;
namespace MslAbi = ShaderRecompiler::MslAbi;
namespace RuntimeAbi = ShaderRecompiler::RuntimeAbi;

// A prelude chunk and its dependencies, each once, dependencies first (as the emitter assembles them).
void AddChunk(std::string& source, std::array<bool, static_cast<std::size_t>(MslPreludeChunk::Count)>& added, MslPreludeChunk chunk) {
    auto& done = added[static_cast<std::size_t>(chunk)];
    if (done) return;
    done = true;
    const auto entry = MslPrelude(chunk);
    for (const auto dependency : entry.dependencies) AddChunk(source, added, dependency);
    source += entry.text;
}

std::string SumLastTwoKernel(const MslAbi::TableLayout& layout) {
    std::string source = "#include <metal_stdlib>\nusing namespace metal;\n";
    std::array<bool, static_cast<std::size_t>(MslPreludeChunk::Count)> added {};
    AddChunk(source, added, MslPreludeChunk::Common);
    AddChunk(source, added, MslPreludeChunk::RuntimeBuffer);
    source += ShaderRecompiler::MslTableDeclaration(layout);
    const auto field = "aps5_table." + ShaderRecompiler::MslTableField(RuntimeAbi::Binding::Buffers);
    source += "kernel void " + std::string(MslAbi::EntryPoint) + "(device const Aps5Table& aps5_table [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::DescriptorTable)) + ")]]) {\n";
    source += "    const uint last = aps5_buffer_read(" + field + "[" + std::to_string(Buffers - 1u) + "], 0u, 0u, 0u, 32u, false);\n";
    source += "    const uint beforeLast = aps5_buffer_read(" + field + "[" + std::to_string(Buffers - 2u) + "], 0u, 0u, 0u, 32u, false);\n";
    source += "    aps5_buffer_write(" + field + "[0], 0u, 0u, 0u, last + beforeLast, 32u, false);\n";
    source += "}\n";
    return source;
}

// Upstream's DescriptorCache checks, unchanged, over the Metal image context.
void DescriptorCacheChecks(const Context& context) {
    DescriptorCache cache(context);
    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, Buffers, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    const std::array<std::uint32_t, 4> key{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, Buffers, VK_SHADER_STAGE_COMPUTE_BIT};
    auto setLayout = cache.Layout(key, std::span(&binding, 1));
    const std::array<VkDescriptorPoolSize, 1> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, Buffers}}};
    const auto first = cache.Allocate(setLayout, sizes);
    const auto second = cache.Allocate(setLayout, sizes);
    Require(first.set != VK_NULL_HANDLE && second.set != VK_NULL_HANDLE, "a set of " + std::to_string(Buffers) + " storage buffers, above the chain pool's capacity, got no descriptor set");
    Require(first.pool != second.pool && cache.Counters().pools == 0, "the oversized sets do not have a pool each");
    Require(cache.Counters().dedicatedPools == 2 && cache.Counters().sets == 2, "the oversized sets were not counted as two dedicated pools");
    cache.Free(first);
    cache.Free(second);
    const auto again = cache.Allocate(setLayout, sizes);
    Require(again.set != VK_NULL_HANDLE && again.pool != VK_NULL_HANDLE && cache.Counters().pools == 0, "an oversized set after the others were freed got no descriptor set");
    cache.Free(again);
}

// The dispatch through a 4097-entry descriptor table, as the Metal device binds guest buffers.
void DispatchChecks(const Context& context) {
    const std::array<MslAbi::TableBinding, 1> bindings{{{RuntimeAbi::Binding::Buffers, Buffers}}};
    const auto layout = MslAbi::LayoutTable(bindings);
    const auto* range = layout.Find(RuntimeAbi::Binding::Buffers);
    Require(range != nullptr && range->count == Buffers && layout.bytes == Buffers * sizeof(MslAbi::BufferEntry), "the MslAbi table of " + std::to_string(Buffers) + " guest buffers is not " + std::to_string(Buffers) + " dense BufferEntry records");

    id<MTLDevice> device = Metal::DeviceOf(context);
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = static_cast<MTLLanguageVersion>((MslAbi::LanguageMajor << 16u) | MslAbi::LanguageMinor);
    NSError* error = nil;
    const auto source = SumLastTwoKernel(layout);
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
    Require(library != nil, "the 4097-buffer table kernel does not compile: " + Metal::Describe(error));
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:MslAbi::EntryPoint]];
    id<MTLComputePipelineState> pipeline = function != nil ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    Require(pipeline != nil, "no pipeline for the 4097-buffer table kernel: " + Metal::Describe(error));

    Buffer storage(context, static_cast<std::size_t>(Buffers) * Spacing, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto* words = reinterpret_cast<std::uint32_t*>(storage.Bytes().data());
    for (std::uint32_t buffer = 0; buffer < Buffers; ++buffer) words[buffer * (Spacing / 4u)] = Input(buffer);
    std::vector<MslAbi::BufferEntry> table(Buffers);
    for (std::uint32_t buffer = 0; buffer < Buffers; ++buffer) table[buffer] = {storage.DeviceAddress() + static_cast<std::uint64_t>(buffer) * Spacing, 4u, 0u};
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        CommandBatch batch(context);
        auto& stream = Metal::Stream(batch.Handle());
        stream.DispatchHelper(pipeline, std::as_bytes(std::span(table)), MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
    }
    storage.Invalidate();
    const auto expected = Input(Buffers - 1u) + Input(Buffers - 2u);
    Require(words[0] == expected, "the dispatch through the oversized table wrote " + std::to_string(words[0]) + ", expected " + std::to_string(expected) + " from its last two storage buffers");
    Require(words[Spacing / 4u] == Input(1u), "the dispatch through the oversized table wrote past its first buffer's 4 bytes");
}

}

int main() {
    try {
        AgcDriver::Graphics::MetalTest::Device test;
        AgcDriver::Graphics::Metal::ImageCache images(test.device, test.queue);
        images.Attach(test.context);
        const auto& context = *images.ImageContext();
        const auto& limits = context.limits;
        std::printf("device %s\n", test.device.name.UTF8String);
        // Metal argument tables bound the heaps, not the device: the image context's limits admit the set.
        AgcDriver::Graphics::Require(limits.maxPerStageDescriptorStorageBuffers >= Buffers && limits.maxPerStageResources >= Buffers && limits.maxDescriptorSetStorageBuffers >= Buffers, "the Metal image context binds fewer than " + std::to_string(Buffers) + " storage buffers per stage or set");
        MetalOversizedTwin::DescriptorCacheChecks(context);
        MetalOversizedTwin::DispatchChecks(context);
        images.Teardown();
        std::puts("oversized descriptor set tests passed on Metal");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
