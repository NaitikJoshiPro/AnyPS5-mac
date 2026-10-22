// Metal twin of tests/Recorder.cpp's --split-writeback-only mode (agc_storage_split_writeback,
// splitWriteBackTests; upstream ea3a2152d): a storage image whose memory spans two adjacent registered
// ranges (one allocation re-protected at the split, or two adjacent allocations) writes back on the GPU
// through each range's own import (StorageTexture::importSegments), stores every texel and nothing
// outside, and a second store outside the write watch keeps the bytes the CPU wrote after the first.
// Copied from upstream but for the clear (Metal::Vk::CmdClearColorImage) and the page size: Metal imports
// whole 16 KiB pages, so upstream's 4 KiB offset and 132 KiB splits are one 16 KiB page here (16 KiB and
// 144 KiB). The import window stays as upstream sets it (64 MiB), so the two ranges import apart and the
// adjacent-allocation join (D65) does not apply to them: neither range crosses a window boundary.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>

namespace {

void splitWriteBackTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    // Metal imports whole 16 KiB pages: upstream's 4 KiB offsets and splits become one page.
    constexpr std::uint64_t page = 16384;
    if (base.hostImportAlignment == 0 || 65536 % base.hostImportAlignment != 0 || page % base.hostImportAlignment != 0) {
        std::cout << "host imports unavailable: the split write-back not tested\n";
        return;
    }
    struct Case {
        const char* name;
        TextureTileMode tileMode;
        std::uint32_t mipCount;
        std::uint64_t offset;
        std::uint64_t split;
        bool gpu;
        bool direct;
        bool twoAllocations;
        bool secondStore;
    };
    const Case cases[] = {
        {"64 KiB R_X 200x150, 3 mips, ranges split at 128 KiB", TextureTileMode::kR64KBX, 3, 0, 131072, true, true, false, false},
        {"64 KiB R_X 200x150, 3 mips, off a 64 KiB boundary, ranges split at 144 KiB", TextureTileMode::kR64KBX, 3, page, 131072 + page, true, true, false, false},
        {"linear 200x150, ranges split at 64 KiB", TextureTileMode::kLinear, 1, 0, 65536, true, true, false, false},
        {"64 KiB R_X 200x150, 3 mips, second range of another access", TextureTileMode::kR64KBX, 3, 0, 131072, false, false, false, false},
        {"64 KiB R_X 200x150, 3 mips, ranges split inside a 64 KiB unit", TextureTileMode::kR64KBX, 3, 0, 131072 + page, true, !UnitShadowEnabled(), false, false},
        {"64 KiB R_X 200x150, 3 mips, two adjacent allocations split at 128 KiB", TextureTileMode::kR64KBX, 3, 0, 131072, true, true, true, false},
        {"64 KiB R_X 200x150, 3 mips, ranges split at 128 KiB, a second store outside the write watch", TextureTileMode::kR64KBX, 3, 0, 131072, true, true, false, true},
        {"linear 200x150, ranges split at 64 KiB, a second store outside the write watch", TextureTileMode::kLinear, 1, 0, 65536, true, true, false, true},
    };
    SetHostImportWindow(64ull << 20u);
    struct Window {
        ~Window() { SetHostImportWindow(0); }
    } window;
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    std::string failures;
    for (const auto& test : cases) {
        GuestTextureResource resource{};
        resource.width = 200;
        resource.height = 150;
        resource.mipCount = test.mipCount;
        resource.tileMode = test.tileMode;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        const auto surfaceBytes = static_cast<std::size_t>(geometry.guestBytes);
        Require(test.offset < test.split && test.split < test.offset + surfaceBytes, "the split test's ranges do not divide its surface");
        std::size_t texelCount = 0;
        for (const auto& mip : geometry.mips) texelCount += static_cast<std::size_t>(mip.width) * mip.height;
        const auto bytes = (static_cast<std::size_t>(test.offset) + surfaceBytes + 65535) / 65536 * 65536 + 65536;
        // The block sits at an odd multiple of 64 KiB: two ranges that fill an aligned power-of-two block are
        // imported as that one block (upstream a24aafb75c), so a block of 512 KiB that macOS malloc happened to
        // place 512 KiB-aligned imported both ranges at once and the case failed on the address alone (the
        // 144 KiB case's ranges fill 512 KiB). Upstream's 4 KiB-page allocations do not land that way.
        void* allocation = std::aligned_alloc(65536, bytes + 65536);
        Require(allocation != nullptr, "cannot allocate the split write-back block");
        void* block = static_cast<std::uint8_t*>(allocation) + ((reinterpret_cast<std::uintptr_t>(allocation) / 65536u) % 2u == 0u ? 65536u : 0u);
        auto* memory = static_cast<std::uint8_t*>(block);
        for (std::size_t i = 0; i < bytes; ++i) memory[i] = pattern(i);
        const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
        const auto address = blockAddress + test.offset;
        {
            GuestAllocations::Mutation mutation;
            if (test.twoAllocations) {
                mutation.Add(block, static_cast<std::size_t>(test.split), true, true, true);
                mutation.Add(memory + test.split, static_cast<std::size_t>(bytes - test.split), true, true, test.gpu);
            } else {
                mutation.Add(block, bytes, true, true, true);
                mutation.Protect(memory + test.split, static_cast<std::size_t>(bytes - test.split), true, true, test.gpu, [] {});
            }
        }
        struct Unregister {
            const Context& context;
            void* allocation;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            void* second;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                    if (second != nullptr) mutation.Remove(second);
                }
                HostImportFor(context, address, bytes);
                std::free(allocation);
            }
        } unregister{base, allocation, block, blockAddress, bytes, test.twoAllocations ? static_cast<void*>(memory + test.split) : nullptr};
        Require(HostImportFor(base, address, surfaceBytes) == nullptr, std::string(test.name) + ": one import covers both ranges");
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        resource.baseAddress = address;
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, test.mipCount, 0, 1};
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
            image->WriteBack();
            const bool direct = !image->GuestSnapshotValid();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            std::size_t texels = 0, garbage = 0, outside = 0;
            for (std::size_t i = 0; i < bytes; i += 4) {
                const bool inside = i >= test.offset && i < test.offset + surfaceBytes;
                bool isTexel = inside, isPattern = true;
                for (std::size_t c = 0; c < 4; ++c) {
                    isTexel = isTexel && memory[i + c] == red[c];
                    isPattern = isPattern && memory[i + c] == pattern(i + c);
                }
                if (isTexel) ++texels;
                else if (!isPattern) ++(inside ? garbage : outside);
            }
            if (direct != test.direct) failures += std::string(test.name) + ": the write-back took the " + (direct ? "GPU-direct" : "CPU") + " path\n";
            if (direct && !(HostImportCovers(base, blockAddress, static_cast<std::size_t>(test.split)) && HostImportCovers(base, blockAddress + test.split, static_cast<std::size_t>(bytes - test.split)))) failures += std::string(test.name) + ": a range has no import after the GPU-direct write-back\n";
            if (garbage != 0 || outside != 0 || texels != texelCount) failures += std::string(test.name) + ": the write-back stored " + std::to_string(texels) + " texels of " + std::to_string(texelCount) + " and changed " + std::to_string(garbage) + " words no texel holds (" + std::to_string(outside) + " past the surface)\n";
            else if (test.secondStore) {
                const auto cpuPage = static_cast<std::size_t>(test.split) + 16384;
                Require(cpuPage + 4096 <= test.offset + surfaceBytes, std::string(test.name) + ": the CPU's page lies past the surface");
                if (AgcDriver::GuestMemory::Watched(address, surfaceBytes)) failures += std::string(test.name) + ": the surface is write-watched\n";
                if (!image->GuestSnapshotValid()) failures += std::string(test.name) + ": the first write-back left no saved bytes\n";
                for (std::size_t i = cpuPage; i < cpuPage + 4096; ++i) memory[i] = static_cast<std::uint8_t>(~pattern(i));
                const auto again = recorder.Commands();
                const VkClearColorValue greenValue{{0.0f, 1.0f, 0.0f, 1.0f}};
                RecordMemoryBarrier(context, again, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                Metal::Vk::CmdClearColorImage(again, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &greenValue, 1, &range);
                RecordMemoryBarrier(context, again, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                constexpr std::array<std::uint8_t, 4> green{0, 255, 0, 255};
                std::size_t kept = 0, greenBelow = 0, greenAbove = 0;
                for (std::size_t i = cpuPage; i < cpuPage + 4096; ++i) kept += memory[i] == static_cast<std::uint8_t>(~pattern(i)) ? 1 : 0;
                for (std::size_t i = test.offset; i < test.offset + surfaceBytes; i += 4) {
                    if (i >= cpuPage && i < cpuPage + 4096) continue;
                    if (std::equal(green.begin(), green.end(), memory + i)) ++(i < test.split ? greenBelow : greenAbove);
                }
                if (kept != 4096) failures += std::string(test.name) + ": the second write-back overwrote " + std::to_string(4096 - kept) + " of the 4096 bytes the CPU wrote after the first\n";
                if (greenBelow == 0 || greenAbove == 0) failures += std::string(test.name) + ": the second write-back stored " + std::to_string(greenBelow) + " texels below the split and " + std::to_string(greenAbove) + " above it\n";
            }
        }
        recorder.Sync();
    }
    if (!failures.empty()) throw std::runtime_error("split write-back:\n" + failures);
    std::cout << "split write-back: ok\n";
}

const bool registered = [] {
    AgcDriver::MetalTwins::RegisterRecorderMode("--split-writeback-only", [] { return RunMode([](const Device& device, Recorder& recorder) { splitWriteBackTests(device, recorder); }); });
    return true;
}();

}
