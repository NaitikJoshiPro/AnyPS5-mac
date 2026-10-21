// Metal twin of tests/Recorder.cpp's --multisample-only mode (agc_multisample_guest_access). The test function
// (multisampleGuestAccessTests, Recorder.cpp at upstream 0f2683cb7) is copied unchanged but for its first check: a
// multisampled color target's samples live only in a host image (here a Metal 2D multisample texture made by
// the MultisampleTarget twin, specifics/macos/src/MultisampleTarget.mm), so a CPU read (the GuestMemory read
// guard), a DMA copy and a texture read through the Metal ShaderResources twin of its memory must throw, also
// in the padding of a later sample plane, while memory past the planes and, after ClearMultisampleTargets, the
// target's own memory read normally. This is not about guest multisampled T#s (types 14/15, which
// DecodeTextureResource rejects on every platform, TechnicalDebt metal-ms-storage): the texture read here is a
// 2D T# over the target's memory. Upstream skips the checks without 4x color targets or nullDescriptor; every
// Metal 4 GPU renders 4x color, and the Metal image path has no null descriptors (it binds placeholders), so the
// twin requires the sample count and does not look at nullDescriptor. Upstream sync 3 cases ported as written: a
// 2D multisample T# (type 14) over the rendered target is sampled as a view of the target's texture (one Texture
// per lookup, MultisampleSources, not reusable), mismatched sample count/width, an unrendered linear T# and storage
// access are refused (upstream 5c42e97f47: storage access moves the target to a storable image when the device has
// shaderStorageImageMultisample; the Metal device sets Context::storageImageMultisample, D22/D28), and a remapped or removed allocation retires the target (a new view, a new sampled texture);
// clearTargets submits and waits before ClearMultisampleTargets. Expected values are upstream's.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "RuntimeAbi.hpp"
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

void multisampleGuestAccessTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    Require((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0, "the Metal device does not render 4x color targets");
    constexpr std::uint32_t side = 64;
    constexpr std::size_t targetBytes = 65536u;
    constexpr std::size_t bytes = 5u * 65536u;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the multisampled target block");
    std::memset(block, 0, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            ClearMultisampleTargets(context.device);
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
#ifdef _WIN32
            VirtualFree(block, 0, MEM_RELEASE);
#else
            std::free(block);
#endif
        }
    } unregister{base, block};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    context.nullDescriptors = true;
    TextureCache cache(context);
    context.textureCache = &cache;
    const auto clearTargets = [&] {
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        ClearMultisampleTargets(context.device);
    };
    const auto refused = [](const std::function<void()>& access, const std::string& expected, const std::string& what) {
        std::string message;
        try {
            access();
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find(expected) != std::string::npos && message.find("multisampled color target") != std::string::npos, what + " of a resolved-only multisampled target's memory did not throw: " + message);
    };
    ColorTarget target{};
    target.address = address;
    target.surfaceAddress = address;
    target.extent = {side, side};
    target.surfaceExtent = target.extent;
    target.format = VK_FORMAT_R8G8B8A8_UNORM;
    target.bytes = targetBytes;
    target.samples = 4;
    {
        std::lock_guard gpu(GpuMutex());
        Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the multisampled target has no view");
        Require(!MultisampleTargetStorable(context, target), "a multisampled target nothing stored to was created with storage usage");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::array<std::byte, 64> texels{};
    refused([&] { AgcDriver::GuestMemory::Read(address + 256u, texels); }, "a CPU read", "a CPU read");
    refused([&] { AgcDriver::GuestMemory::ReadCommitted(address + 256u, texels); }, "a CPU read", "a committed CPU read");
    refused([&] { AgcDriver::GuestMemory::ReadCommittedUnsynced(address + 256u, texels); }, "a CPU read", "a committed CPU read without the flush hook");
    refused([&] { RequireNoMultisampleTarget(address, 64u, "a DMA copy"); }, "a DMA copy", "a DMA copy");
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::SampledImage;
    binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
    binding.descriptorSet = 0;
    binding.binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding;
    binding.count = 1;
    binding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
    binding.imageSamplers = {0};
    binding.guestDescriptor = {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (9u << 28u),
        0u,
        0u,
        0u,
        0u,
    };
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    refused([&] {
        std::lock_guard gpu(GpuMutex());
        ShaderResources resources(context, compute);
    }, "a texture read", "a texture read");
    const auto multisampledWords = [&](std::uint32_t width, std::uint32_t sampleShift, std::uint64_t base) {
        return std::vector<std::uint32_t>{
            static_cast<std::uint32_t>(base >> 8u),
            static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (56u << 20u) | (((width - 1u) & 3u) << 30u),
            ((width - 1u) >> 2u) | ((side - 1u) << 14u),
            0xfacu | (sampleShift << 16u) | (14u << 28u),
            0u,
            sampleShift << 4u,
            0u,
            0u,
        };
    };
    const auto sampledProgram = [&](ShaderRecompiler::DescriptorKind kind, std::vector<std::uint32_t> words) {
        ShaderRecompiler::RecompileResult result;
        auto element = binding;
        element.kind = kind;
        if (kind == ShaderRecompiler::DescriptorKind::StorageImage) element.binding = ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
        element.imageShape = ShaderRecompiler::DescriptorImageShape::Image2DMultisample;
        element.guestDescriptor = std::move(words);
        result.bindings.push_back(std::move(element));
        return result;
    };
    const auto refusedSample = [&](ShaderRecompiler::DescriptorKind kind, std::vector<std::uint32_t> words, const std::string& expected, const std::string& what) {
        const auto refusedProgram = sampledProgram(kind, std::move(words));
        const CompiledShader refusedCompute{ShaderRecompiler::ShaderStage::Compute, &refusedProgram, 0};
        std::string message;
        try {
            std::lock_guard gpu(GpuMutex());
            ShaderResources resources(context, refusedCompute);
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find(expected) != std::string::npos, what + " did not throw: " + message);
    };
    {
        const auto sampled = sampledProgram(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address));
        const CompiledShader sampledCompute{ShaderRecompiler::ShaderStage::Compute, &sampled, 0};
        std::lock_guard gpu(GpuMutex());
        ShaderResources first(context, sampledCompute);
        ShaderResources second(context, sampledCompute);
        Require(first.SampledTextures().size() == 1 && first.SampledTextures().front() != nullptr, "the multisampled target was not bound as a sampled texture");
        Require(first.SampledTextures().front()->Layout() == VK_IMAGE_LAYOUT_GENERAL && first.SampledTextures().front()->ViewFormat() == target.format, "the sampled multisampled target is not a view of the target's image in its layout");
        Require(second.SampledTextures().front() == first.SampledTextures().front(), "a second lookup of the multisampled target made another texture");
        Require(first.MultisampleSources() == std::vector<std::uint64_t>{address} && !first.Reusable(), "resources sampling a multisampled target are kept for reuse without the barrier their lookup records");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 3u, address), "as a 64x64 texture of 8 samples", "sampling a 4x target as an 8x texture");
    refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side / 2u, 2u, address), "as a 32x64 texture of 4 samples", "sampling a 4x target as a narrower texture");
    refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address + 4u * 65536u), "which no draw rendered to and whose tile mode 0 has no modeled multisampled color footprint", "sampling a linear multisampled texture no draw rendered to");
    if (context.storageImageMultisample) {
        const auto stored = sampledProgram(ShaderRecompiler::DescriptorKind::StorageImage, multisampledWords(side, 2u, address));
        const CompiledShader storedCompute{ShaderRecompiler::ShaderStage::Compute, &stored, 0};
        std::lock_guard gpu(GpuMutex());
        const auto rendered = MultisampleTargetView(context, target);
        Require(!MultisampleTargetStorable(context, target), "sampling the multisampled target gave it storage usage");
        ShaderResources resources(context, storedCompute);
        Require(MultisampleTargetStorable(context, target) && MultisampleTargetView(context, target) != rendered, "a storage binding did not move the multisampled target to an image with storage usage");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    } else {
        // Upstream's refusal comes from the MultisampleTarget storage path; the Metal ShaderResources twin refuses
        // a multisampled storage view before it reaches the target (ShaderResources.mm), with its own message.
        std::string message;
        try {
            const auto refusedProgram = sampledProgram(ShaderRecompiler::DescriptorKind::StorageImage, multisampledWords(side, 2u, address));
            const CompiledShader refusedCompute{ShaderRecompiler::ShaderStage::Compute, &refusedProgram, 0};
            std::lock_guard gpu(GpuMutex());
            ShaderResources resources(context, refusedCompute);
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find("as a storage image at its sample count") != std::string::npos || message.find("storage image access to a multisampled texture") != std::string::npos, "storage access to a multisampled target without shaderStorageImageMultisample did not throw: " + message);
    }
    refused([&] { AgcDriver::GuestMemory::Read(address + 3u * 65536u + 256u, texels); }, "a CPU read", "a CPU read in the padding of a later sample plane");
    AgcDriver::GuestMemory::Read(address + 4u * 65536u, texels);
    clearTargets();
    // Upstream 0f2683cb7: a tiled (RenderTarget) multisampled target guards its whole surface, including the
    // tile-height padding of 2x and 8x targets.
    target.tileMode = ColorTileMode::RenderTarget;
    {
        std::lock_guard gpu(GpuMutex());
        Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the tiled multisampled target has no view");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    refused([&] { AgcDriver::GuestMemory::Read(address + targetBytes - texels.size(), texels); }, "a CPU read", "a CPU read inside the tiled multisampled target");
    AgcDriver::GuestMemory::Read(address + targetBytes, texels);
    clearTargets();
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_2_BIT) != 0) {
        target.extent = {128, 64};
        target.surfaceExtent = target.extent;
        target.samples = 2;
        {
            std::lock_guard gpu(GpuMutex());
            Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the 2x tiled multisampled target has no view");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        refused([&] { AgcDriver::GuestMemory::Read(address + 2u * targetBytes - texels.size(), texels); }, "a CPU read", "a CPU read inside the 2x tile-height padding");
        AgcDriver::GuestMemory::Read(address + 2u * targetBytes, texels);
        clearTargets();
    }
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_8_BIT) != 0) {
        target.extent = {64, 32};
        target.surfaceExtent = target.extent;
        target.samples = 8;
        {
            std::lock_guard gpu(GpuMutex());
            Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the 8x tiled multisampled target has no view");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        refused([&] { AgcDriver::GuestMemory::Read(address + 2u * targetBytes - texels.size(), texels); }, "a CPU read", "a CPU read inside the 8x tile-height padding");
        AgcDriver::GuestMemory::Read(address + 2u * targetBytes, texels);
        clearTargets();
    }
    AgcDriver::GuestMemory::Read(address + 256u, texels);
    target.extent = {side, side};
    target.surfaceExtent = target.extent;
    target.samples = 4;
    VkImageView previous = VK_NULL_HANDLE;
    std::shared_ptr<Texture> previousTexture;
    for (const bool removeAllocation : {false, true}) {
        VkImageView view = VK_NULL_HANDLE;
        std::shared_ptr<Texture> texture;
        {
            const auto sampled = sampledProgram(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address));
            const CompiledShader sampledCompute{ShaderRecompiler::ShaderStage::Compute, &sampled, 0};
            std::lock_guard gpu(GpuMutex());
            view = MultisampleTargetView(context, target);
            Require(view != VK_NULL_HANDLE && view != previous, "a remapped allocation reused its retired multisampled image");
            ShaderResources resources(context, sampledCompute);
            texture = resources.SampledTextures().front();
            Require(texture != nullptr && texture != previousTexture, "a remapped allocation sampled its retired multisampled image");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        {
            GuestAllocations::Mutation mutation;
            mutation.Protect(block, bytes, true, true, true, [] {});
        }
        refused([&] { RequireNoMultisampleTarget(address, 64u, "a CPU read"); }, "a CPU read", "a read after protection changed");
        bool failed = false;
        try {
            GuestAllocations::Mutation mutation;
            mutation.Unmap(block, bytes, [](const void*, std::size_t, const void*, bool) { throw std::runtime_error("unmap failed"); });
        } catch (const std::runtime_error&) {
            failed = true;
        }
        Require(failed, "the failed host unmap did not throw");
        refused([&] { RequireNoMultisampleTarget(address, 64u, "a CPU read"); }, "a CPU read", "a read after a failed unmap");
        {
            GuestAllocations::Mutation mutation;
            if (removeAllocation) mutation.Remove(block);
            else mutation.Unmap(block, bytes, [](const void*, std::size_t, const void*, bool) {});
            mutation.Add(block, bytes, true, true, true);
        }
        RequireNoMultisampleTarget(address, bytes, "a texture read");
        AgcDriver::GuestMemory::Read(address + 256u, texels);
        refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address), "which no draw rendered to", "sampling a multisampled target whose allocation was unmapped");
        previous = view;
        previousTexture = texture;
    }
    clearTargets();
    std::cout << "Multisampled target guest access tests passed\n";
}

const bool registered = AgcDriver::MetalTwins::RegisterRecorderMode("--multisample-only", [] {
    return RunMode([](const Device& device, Recorder& recorder) { multisampleGuestAccessTests(device, recorder); });
});

}
