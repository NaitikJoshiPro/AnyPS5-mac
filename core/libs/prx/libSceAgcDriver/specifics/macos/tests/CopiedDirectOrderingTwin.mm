// Metal twin of tests/Recorder.cpp --copied-direct-only (agc_copied_direct_ordering, copiedDirectOrderingTests),
// registered with RecorderTwin.mm (RecorderTwins.h). Upstream dispatches tests/shaders/BufferUpdate.comp
// (values[gl_GlobalInvocationID.x] += delta) through a pipeline layout of the descriptor set ShaderResources
// binds, so it checks that a direct (host-imported) consumer, fresh or revalidated, runs after a copied producer's
// CPU write-back, that disjoint copied and direct writes keep apart, that a CPU completion label orders before a
// later direct consumer, that ordinary direct reuse waits for nothing, and that cached sampled images (colour and
// DCC metadata) see a copied producer's results. On Metal, Bind is a no-op: guest buffers are bound by address at
// dispatch (MslAbi tables), so the same update runs as an MSL kernel recorded with CommandStream::DispatchHelper
// into the recorder's batch, addressing the Metal buffer that the object's guest-buffer Descriptor names
// (ShaderResources::BufferMemory, a test seam) at the descriptor's offset. Barriers, Keep, MarkGpuWrites,
// OnComplete write-backs, Revalidate and the image copies are upstream's calls; the image context is the image
// cache's (its detiler and texture cache), which upstream builds itself. Expected values are upstream's.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "../src/MetalCommandStream.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "RuntimeAbi.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr const char* UpdateSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct UpdateParameters {
    ulong values;
    uint delta;
};
kernel void aps5_buffer_update(constant UpdateParameters& parameters [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    device uint* values = reinterpret_cast<device uint*>(parameters.values);
    values[index] += parameters.delta;
}
)MSL";

struct UpdateParameters {
    std::uint64_t values;
    std::uint32_t delta;
    std::uint32_t padding;
};

id<MTLComputePipelineState> UpdatePipeline(const Context& context) {
    id<MTLDevice> device = Metal::DeviceOf(context);
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = static_cast<MTLLanguageVersion>((ShaderRecompiler::MslAbi::LanguageMajor << 16u) | ShaderRecompiler::MslAbi::LanguageMinor);
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(UpdateSource) options:options error:&error];
    Require(library != nil, "the ordering update kernel does not compile: " + Metal::Describe(error));
    id<MTLFunction> function = [library newFunctionWithName:@"aps5_buffer_update"];
    id<MTLComputePipelineState> pipeline = function != nil ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    Require(pipeline != nil, "no pipeline for the ordering update kernel: " + Metal::Describe(error));
    return pipeline;
}

void copiedDirectOrderingTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    // Metal binds no buffer through descriptors, so a null binding cannot fault (upstream needs VK_EXT_robustness2).
    context.nullDescriptors = true;
    if (context.hostImportAlignment == 0 || !context.nullDescriptors || !AgcDriver::GuestMemory::WriteWatched()) {
        std::cout << "Host imports, null descriptors or write watching unavailable: copied/direct ordering test skipped\n";
        return;
    }
    recorder.Sync();
    const auto bytes = static_cast<std::size_t>(std::max<VkDeviceSize>(131072, context.hostImportAlignment));
    auto* block = static_cast<std::uint32_t*>(AllocateWatched(bytes, bytes));
    Require(block != nullptr, "allocate copied/direct ordering memory");
    const auto address = reinterpret_cast<std::uint64_t>(block);
    std::vector<std::shared_ptr<ShaderResources>> writers;
    context.copiedWriters = &writers;
    struct Release {
        const Context& context;
        Recorder& recorder;
        std::uint32_t* block;
        std::size_t bytes;
        std::vector<std::shared_ptr<ShaderResources>>& writers;
        std::shared_ptr<ShaderResources> layoutOwner;
        ~Release() {
            recorder.Sync();
            writers.clear();
            DrawCopiedWriters()->clear();
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            static_cast<void>(HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes));
            recorder.Sync();
            ReleaseWatched(block, bytes);
        }
    } release{context, recorder, block, bytes, writers};
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    std::memset(block, 0, bytes);
    Require(HostImportFor(context, address, bytes) != nullptr, "import copied/direct ordering memory");
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.guestDescriptor = {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, 1024, 0x31000000u};
    binding.bufferWritten = {true};
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    auto direct = std::make_shared<ShaderResources>(context, compute);
    release.layoutOwner = direct;
    Require(!direct->HasCopiedWrites(), "ordering consumer did not bind directly");
    const id<MTLComputePipelineState> pipeline = UpdatePipeline(context);
    // The guest range each program binds (its V#'s base and record bytes), whose first dwords the update adds to.
    const auto update = [&](const std::shared_ptr<ShaderResources>& resources, std::uint64_t base, std::size_t boundBytes, std::uint32_t delta, std::uint32_t count = 1) {
        const auto commands = recorder.Commands();
        recorder.Keep(resources);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        std::uint32_t adjustment = 0;
        const auto descriptor = resources->BufferMemory().Descriptor(base, boundBytes, adjustment);
        const id<MTLBuffer> metal = Metal::BufferOf(descriptor.buffer);
        Require(metal != nil && descriptor.offset + adjustment + static_cast<VkDeviceSize>(count) * 4u <= metal.length, "the ordering update's guest buffer is not a Metal buffer covering its words");
        const UpdateParameters parameters{metal.gpuAddress + descriptor.offset + adjustment, delta, 0};
        Metal::Stream(commands).DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake(count, 1, 1), MTLSizeMake(1, 1, 1));
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
        resources->MarkGpuWrites(recorder);
    };
    for (int mode = 0; mode < 3; ++mode) {
        *block = 10;
        auto copiedContext = context;
        copiedContext.hostImportAlignment = 0;
        copiedContext.dmaBufImport = false;
        auto copied = std::make_shared<ShaderResources>(copiedContext, compute);
        Require(copied->HasCopiedWrites(), "ordering producer did not take the copied path");
        auto& pending = mode == 2 ? *DrawCopiedWriters() : writers;
        pending.push_back(copied);
        update(copied, address, 1024, 1);
        recorder.OnComplete([copied, &pending] {
            copied->WriteBackBuffers();
            std::erase(pending, copied);
        });
        Require(!recorder.Idle() && *block == 10, "copied producer was completed before the consumer test");
        if (mode == 0) direct = std::make_shared<ShaderResources>(context, compute);
        else Require(direct->Revalidate(std::span(&compute, 1)), "the imported template could not be revalidated");
        update(direct, address, 1024, 2);
        recorder.Sync();
        Require(*block == 13, mode == 0 ? "fresh direct consumer ran before copied write-back" : "reused direct consumer ran before copied write-back");
    }
    *block = 20;
    block[1024] = 30;
    auto copiedContext = context;
    copiedContext.hostImportAlignment = 0;
    copiedContext.dmaBufImport = false;
    auto copied = std::make_shared<ShaderResources>(copiedContext, compute);
    writers.push_back(copied);
    update(copied, address, 1024, 1);
    recorder.OnComplete([copied, &writers] {
        copied->WriteBackBuffers();
        std::erase(writers, copied);
    });
    auto separateProgram = program;
    separateProgram.bindings[0].guestDescriptor[0] += 4096;
    const CompiledShader separateCompute{ShaderRecompiler::ShaderStage::Compute, &separateProgram, 0};
    auto separate = std::make_shared<ShaderResources>(context, separateCompute);
    Require(!writers.empty() && *block == 20, "a disjoint consumer waited for copied writes");
    update(separate, address + 4096, 1024, 2);
    recorder.Sync();
    Require(*block == 21 && block[1024] == 32, "disjoint copied and direct writes changed each other");
    const std::uint32_t label = 40;
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(address, std::as_bytes(std::span(&label, 1)), 1000, 0, false);
    Require(direct->Revalidate(std::span(&compute, 1)), "template reuse after a CPU completion label failed");
    update(direct, address, 1024, 2);
    recorder.Sync();
    Require(*block == 42, "direct consumer preceded an overlapping CPU completion label");
    update(direct, address, 1024, 1);
    const auto submissions = recorder.Submissions();
    Require(direct->Revalidate(std::span(&compute, 1)) && recorder.Submissions() == submissions, "ordinary direct work incurred a CPU write-back wait");
    update(direct, address, 1024, 2);
    recorder.Sync();
    Require(*block == 45, "direct consumers lost GPU-ordered updates");
    {
        // Upstream builds a TextureDetiler and a TextureCache for its image context; the image cache's Context
        // already carries the Metal ones (Metal::ImageCache::Attach).
        auto imageContext = context;
        Require(imageContext.detiler != nullptr && imageContext.textureCache != nullptr, "the Metal image context has no detiler or texture cache");
        Buffer readback(context, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        struct Clear {
            const Context& context;
            Recorder& recorder;
            ~Clear() { recorder.Sync(); ClearCachedTextures(context.device); }
        } clear{context, recorder};
        ShaderRecompiler::RecompileResult imageProgram;
        ShaderRecompiler::DescriptorBinding imageBinding;
        imageBinding.kind = ShaderRecompiler::DescriptorKind::SampledImage;
        imageBinding.role = ShaderRecompiler::DescriptorRole::GuestImages;
        imageBinding.descriptorSet = 0;
        imageBinding.binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding;
        imageBinding.count = 1;
        imageBinding.imageSamplers = {0};
        imageBinding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
        imageBinding.guestDescriptor = {static_cast<std::uint32_t>(address >> 8), static_cast<std::uint32_t>(address >> 40) | (56u << 20) | (3u << 30), 15u | (63u << 14), 0x91b00fac, 0, 0, 0, 0};
        imageProgram.bindings.push_back(imageBinding);
        const CompiledShader imageCompute{ShaderRecompiler::ShaderStage::Compute, &imageProgram, 0};
        auto sampled = std::make_shared<ShaderResources>(imageContext, imageCompute);
        recorder.Sync();
        const auto copyFirstTexel = [&](const std::vector<std::pair<VkImage, bool>>& images) {
            const auto commands = recorder.Commands();
            recorder.Keep(sampled);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {1, 1, 1};
            Metal::Vk::CmdCopyImageToBuffer(commands, images.front().first, VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Sync();
            std::uint32_t value = 0;
            std::memcpy(&value, readback.Bytes().data(), sizeof(value));
            return value;
        };
        for (int reuse = 0; reuse < 2; ++reuse) {
            auto producer = std::make_shared<ShaderResources>(copiedContext, compute);
            writers.push_back(producer);
            update(producer, address, 1024, 1);
            recorder.OnComplete([producer, &writers] {
                producer->WriteBackBuffers();
                std::erase(writers, producer);
            });
            if (reuse == 0) sampled = std::make_shared<ShaderResources>(imageContext, imageCompute);
            else Require(sampled->Revalidate(std::span(&imageCompute, 1)), "sampled image template did not revalidate");
            const auto images = sampled->StorageImages();
            Require(images.size() == 1, "sampled image did not use imported texture storage");
            Require(copyFirstTexel(images) == static_cast<std::uint32_t>(46 + reuse), "cached sampled image preceded a copied producer");
        }
        sampled.reset();
        ClearCachedTextures(context.device);
        const auto keys = address + 65536;
        const auto keyBytes = DccKeyBytes(65536);
        imageProgram.bindings[0].guestDescriptor[6] = (1u << 21) | (static_cast<std::uint32_t>((keys >> 8) & 0xffu) << 24);
        imageProgram.bindings[0].guestDescriptor[7] = static_cast<std::uint32_t>(keys >> 16);
        auto keyProgram = program;
        keyProgram.bindings[0].guestDescriptor = {static_cast<std::uint32_t>(keys), static_cast<std::uint32_t>(keys >> 32u) & 0xffffu, static_cast<std::uint32_t>(keyBytes), 0x31000000u};
        const CompiledShader keyCompute{ShaderRecompiler::ShaderStage::Compute, &keyProgram, 0};
        for (int reuse = 0; reuse < 2; ++reuse) {
            std::memset(reinterpret_cast<void*>(keys), 0, keyBytes);
            AgcDriver::GuestMemory::MarkWritten(keys, keyBytes);
            sampled = std::make_shared<ShaderResources>(imageContext, imageCompute);
            recorder.Sync();
            auto producer = std::make_shared<ShaderResources>(copiedContext, keyCompute);
            writers.push_back(producer);
            update(producer, keys, keyBytes, 0x40404040, static_cast<std::uint32_t>(keyBytes / 4));
            recorder.OnComplete([producer, &writers] {
                producer->WriteBackBuffers();
                std::erase(writers, producer);
            });
            if (reuse == 0) sampled = std::make_shared<ShaderResources>(imageContext, imageCompute);
            else Require(sampled->Revalidate(std::span(&imageCompute, 1)), "metadata-only producer invalidated the sampled template");
            const auto images = sampled->StorageImages();
            Require(images.size() == 1, "metadata texture is not resident");
            Require(copyFirstTexel(images) == 0xff000000u, "cached sampled image ignored copied compression-metadata writes");
        }
    }
    std::cout << "Copied producers precede fresh and reused direct consumers\n";
}

const bool copiedDirect = AgcDriver::MetalTwins::RegisterRecorderMode("--copied-direct-only", [] {
    return RunMode([](const Device& device, Recorder& recorder) { copiedDirectOrderingTests(device, recorder); });
});

}
