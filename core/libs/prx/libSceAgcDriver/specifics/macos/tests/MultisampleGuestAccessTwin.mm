// Metal twin of tests/Recorder.cpp's --multisample-only mode (agc_multisample_guest_access). The test function
// (multisampleGuestAccessTests, Recorder.cpp:4504-4605) is copied unchanged but for its first check: a
// multisampled color target's samples live only in a host image (here a Metal 2D multisample texture made by
// the MultisampleTarget twin, specifics/macos/src/MultisampleTarget.mm), so a CPU read (the GuestMemory read
// guard), a DMA copy and a texture read through the Metal ShaderResources twin of its memory must throw, also
// in the padding of a later sample plane, while memory past the planes and, after ClearMultisampleTargets, the
// target's own memory read normally. This is not about guest multisampled T#s (types 14/15, which
// DecodeTextureResource rejects on every platform, TechnicalDebt metal-ms-storage): the texture read here is a
// 2D T# over the target's memory. Upstream skips the checks without 4x color targets or nullDescriptor; every
// Metal 4 GPU renders 4x color, and the Metal image path has no null descriptors (it binds placeholders), so the
// twin requires the sample count and does not look at nullDescriptor. Expected values are upstream's.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

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
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::array<std::byte, 64> texels{};
    refused([&] { AgcDriver::GuestMemory::Read(address + 256u, texels); }, "a CPU read", "a CPU read");
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
    refused([&] { AgcDriver::GuestMemory::Read(address + 3u * 65536u + 256u, texels); }, "a CPU read", "a CPU read in the padding of a later sample plane");
    AgcDriver::GuestMemory::Read(address + 4u * 65536u, texels);
    ClearMultisampleTargets(context.device);
    AgcDriver::GuestMemory::Read(address + 256u, texels);
    std::cout << "Multisampled target guest access tests passed\n";
}

const bool registered = AgcDriver::MetalTwins::RegisterRecorderMode("--multisample-only", [] {
    return RunMode([](const Device& device, Recorder& recorder) { multisampleGuestAccessTests(device, recorder); });
});

}
