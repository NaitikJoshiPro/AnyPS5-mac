// Metal twin of three tests/Recorder.cpp modes (upstream 661f8b7171, UPSYNC4 owed tests):
//   --read-written-staging-only (agc_read_written_staging, readWrittenStagingTests): which written and read+written
//     storage elements are bound in place and which are staged, through GuestBufferMemory and a compute
//     ShaderResources.
//   --storage-write-back-only (agc_storage_write_back): storageRefreshTests, writeBackPaddingTests and
//     singlePassTests, each unwatched then watched.
//   --writeback-blocks-only (agc_storage_writeback_blocks, writeBackBlockTests): a 4 MiB write-back over 64 KiB
//     pieces imports one or two segments, or none when every other piece is not GPU mapped.
// Copied from upstream but for the Vulkan entry points (vkCmdClearColorImage, vkCmdCopyImageToBuffer and
// vkCmdCopyBufferToImage become the Metal::Vk shims), the Windows allocation branches, and singlePassTests'
// gate: upstream returns early unless the device set Context::singlePassStorage (shaderStorageImage{Read,Write}
// WithoutFormat); each case sets the flag on its own context copy, so the twin runs the cases whenever host
// imports exist (the Metal device sets the flag, MetalDevice.mm; the detile kernels take the image with read or
// write access, which RG32Uint, the 8-byte element view, has and read_write does not: TextureDetiler.mm).

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool readWrittenStagingTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: read and written staging not tested\n";
        return false;
    }
    constexpr std::size_t bytes = 8u << 20u;
    void* block = std::aligned_alloc(65536, bytes);
    Require(block != nullptr, "cannot allocate the test block");
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the test block refused: read and written staging not tested\n";
        return false;
    }
    const auto boundInPlace = [&](std::uint64_t element, std::size_t elementBytes, bool read) {
        GuestBufferMemory memory(context);
        memory.AllowDeviceStaging();
        memory.AddWritable(element, elementBytes, false, read);
        memory.Upload(false);
        const auto reads = memory.InPlaceReads();
        const bool direct = reads.size() == 1 && reads[0].first == element && reads[0].second == element + elementBytes;
        Require(direct || reads.empty(), "an element was bound neither in place nor staged");
        memory.RecordCopyBacks(recorder);
        recorder.Sync();
        return direct;
    };
    Require(!boundInPlace(address + 4096, 1u << 20u, false), "a written 1 MiB element was not staged");
    Require(boundInPlace(address + 4096, 4u << 20u, false), "a written-only 4 MiB element was staged past the written window");
    Require(!boundInPlace(address + 4096, 4u << 20u, true), "a read and written 4 MiB element was bound in place");
    const auto buildInPlace = [&](std::uint64_t dispatchThreads, bool read, std::uint32_t stride, std::uint32_t extent) {
        const auto element = address + 4096;
        constexpr std::uint32_t elementBytes = 4u << 20u;
        const std::uint32_t records = stride != 0 ? elementBytes / stride : elementBytes;
        ShaderRecompiler::RecompileResult program;
        ShaderRecompiler::DescriptorBinding binding;
        binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
        binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
        binding.descriptorSet = 0;
        binding.binding = 0;
        binding.count = 1;
        binding.guestDescriptor = {static_cast<std::uint32_t>(element), (static_cast<std::uint32_t>(element >> 32u) & 0xffffu) | (stride << 16u), records, 0x31000000u};
        binding.bufferWritten = {true};
        binding.bufferRead = {read};
        binding.bufferExtent = {extent};
        program.bindings.push_back(binding);
        const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
        ShaderResources resources(context, compute, {}, false, dispatchThreads);
        const auto reads = resources.InPlaceReads();
        const bool direct = std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return range.first <= element && range.second >= element + elementBytes; });
        resources.MarkGpuWrites(recorder);
        recorder.Sync();
        return direct;
    };
    Require(!buildInPlace((4u << 20u) / 16u, true, 16, 0), "a dispatch sweeping a read and written 4 MiB element bound it in place");
    Require(buildInPlace(1024, true, 16, 0), "a dispatch touching 16 KiB of a read and written 4 MiB element staged all of it");
    Require(buildInPlace(0, true, 16, 0), "a dispatch of unknown size staged a read and written 4 MiB element");
    Require(buildInPlace((4u << 20u) / 16u, false, 16, 0), "a dispatch sweeping a written-only 4 MiB element staged it");
    Require(!buildInPlace((4u << 20u) / 32u, true, 0, 32), "a dispatch sweeping a read and written 4 MiB raw element bound it in place");
    Require(buildInPlace(1024, true, 0, 32), "a dispatch touching 32 KiB of a read and written 4 MiB raw element staged all of it");
    Require(buildInPlace((4u << 20u) / 32u, true, 0, 0), "a raw element without an access extent was staged");
    Require(std::all_of(static_cast<const std::byte*>(block), static_cast<const std::byte*>(block) + bytes, [](std::byte value) { return value == std::byte{0x11}; }), "a staged copy-back changed the guest bytes");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    }
    HostImportFor(context, address, bytes);
    return true;
}

void storageRefreshTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: storage refresh not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = nullptr;
    if (watched) {
        block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: storage refresh in watched memory not tested\n";
            return;
        }
    } else {
        block = std::aligned_alloc(65536, bytes);
    }
    Require(block != nullptr, "cannot allocate the storage refresh block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    const auto* import = HostImportFor(base, address, bytes);
    if (import == nullptr) {
        std::cout << "host import of the storage refresh block refused: storage refresh not tested\n";
        return;
    }
    if (watched && !AgcDriver::GuestMemory::Watched(address, bytes)) std::cout << "host imports are compared, not watched: storage refresh in watched memory runs as unwatched\n";
    watched = watched && AgcDriver::GuestMemory::Watched(address, bytes);
    Require(watched || !ShadowDestinationFor(base, *import, address, address + 65536).has_value(), "a unit shadow was offered for memory the write tracker does not watch");
    if (!AgcDriver::GuestMemory::WriteWatched()) Require(!UnitShadowEnabled(), "unit shadows are on without write watching");
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    Require(DescribeSurface(resource).guestBytes == surfaceBytes, "the test surface has an unexpected size");
    DccKeyProof proof;
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000, "0x00 keys are not a 0000 clear");
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto draw = [&](VkClearColorValue value) {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        };
        const auto holds = [&](std::array<std::uint8_t, 4> texel) {
            Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            Metal::Vk::CmdCopyImageToBuffer(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                if (std::to_integer<std::uint8_t>(pixels[i]) != texel[i % 4]) return false;
            }
            return true;
        };
        const auto memoryHolds = [&](std::array<std::uint8_t, 4> texel, std::size_t first = 0, std::uint8_t other = 0) {
            std::vector<std::byte> read(surfaceBytes);
            if (watched) AgcDriver::GuestMemory::Read(address, read);
            else std::memcpy(read.data(), texels, surfaceBytes);
            for (std::size_t i = 0; i < surfaceBytes; ++i) {
                if (std::to_integer<std::uint8_t>(read[i]) != (i < first ? other : texel[i % 4])) return false;
            }
            return true;
        };
        const auto holdsMixed = [&](std::array<std::uint8_t, 4> texel, std::size_t count, std::uint8_t other) {
            Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            Metal::Vk::CmdCopyImageToBuffer(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            std::size_t others = 0;
            for (std::size_t i = 0; i < pixels.size(); i += 4) {
                bool isOther = true, isTexel = true;
                for (std::size_t c = 0; c < 4; ++c) {
                    const auto value = std::to_integer<std::uint8_t>(pixels[i + c]);
                    isOther = isOther && value == other;
                    isTexel = isTexel && value == texel[c];
                }
                if (isOther) ++others;
                else if (!isTexel) return false;
            }
            return others == count;
        };
        Require(holds({0, 0, 0, 0}), "a surface under 0000 clear keys was not cleared");
        draw({{1.0f, 0.0f, 0.0f, 1.0f}});
        image->Refresh();
        Require(holds({255, 0, 0, 255}), "a refresh under the image's own clear keys dropped the draw's results");
        if (watched) {
            Require(memoryHolds({255, 0, 0, 255}), "the refresh's store of the results did not reach watched guest memory");
        } else {
            Require(StorageTexture::FindPending(address, surfaceBytes) == image, "an unchanged untracked refresh discarded the pending image");
            Require(StorageTexture::FlushPending(address, surfaceBytes, nullptr, "storage refresh visibility test"), "the pending untracked image was not flushed");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            Require(memoryHolds({255, 0, 0, 255}), "an explicit flush did not make pending results visible in untracked guest memory");
        }
        if (watched) {
            draw({{0.0f, 0.0f, 1.0f, 1.0f}});
            std::memset(texels, 0x33, 65536);
            image->Refresh();
            Require(holdsMixed({0, 0, 255, 255}, 65536 / 4, 0x33), "a CPU store into a unit with results pending was not seen by the refresh");
            Require(memoryHolds({0, 0, 255, 255}, 65536, 0x33), "guest memory lost the CPU store or the other units' results");
            image->Refresh();
            Require(holdsMixed({0, 0, 255, 255}, 65536 / 4, 0x33), "an unchanged surface changed at a refresh");
        }
        draw({{0.0f, 1.0f, 0.0f, 1.0f}});
        std::memset(keys, 0x40, keyCount);
        image->Refresh();
        Require(holds({0, 0, 0, 255}), "new 0001 clear keys did not clear the image");
        Require(watched ? memoryHolds({0, 0, 255, 255}, 65536, 0x33) : memoryHolds({255, 0, 0, 255}), "results dead under new clear keys were stored");
    }
    recorder.Sync();
}

void writeBackPaddingTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: write-back padding not tested\n";
        return;
    }
    struct Case {
        const char* name;
        TextureTileMode tileMode;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t mipCount;
        std::uint64_t offset;
        bool direct = false;
    };
    constexpr Case cases[] = {
        {"64 KiB R_X 200x150", TextureTileMode::kR64KBX, 200, 150, 1, 0},
        {"64 KiB R_X 200x150, 3 mips", TextureTileMode::kR64KBX, 200, 150, 3, 0},
        {"64 KiB R_X 520x260, 6 mips", TextureTileMode::kR64KBX, 520, 260, 6, 0, true},
        {"64 KiB R_X 200x150 off a 64 KiB boundary, 3 mips", TextureTileMode::kR64KBX, 200, 150, 3, 4096},
        {"4 KiB S 100x70, 2 mips", TextureTileMode::kStandard4KB, 100, 70, 2, 0},
        {"linear 200x150", TextureTileMode::kLinear, 200, 150, 1, 0},
        {"64 KiB R_X 512x512, 3 mips, no padding below the tail", TextureTileMode::kR64KBX, 512, 512, 3, 0, true},
        {"64 KiB R_X 1000x700", TextureTileMode::kR64KBX, 1000, 700, 1, 0, true},
        {"64 KiB R_X 1900x1100, across two 8 MiB slabs", TextureTileMode::kR64KBX, 1900, 1100, 1, 0, true},
    };
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    constexpr std::array<std::array<std::uint8_t, 4>, 2> colours{{{255, 0, 0, 255}, {0, 255, 0, 255}}};
    std::string failures;
    for (const auto& test : cases) {
        const std::string what = std::string(watched ? "(watched) " : "") + test.name;
        GuestTextureResource resource{};
        resource.width = test.width;
        resource.height = test.height;
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
        std::size_t texelCount = 0;
        for (const auto& mip : geometry.mips) texelCount += static_cast<std::size_t>(mip.width) * mip.height;
        // The surface, then a 64 KiB unit past it that no store may reach.
        const auto bytes = (static_cast<std::size_t>(test.offset) + surfaceBytes + 65535) / 65536 * 65536 + 65536;
        void* block = nullptr;
        if (watched) {
            block = AllocateWatched(bytes, 65536);
            if (block == nullptr) {
                std::cout << "no write watching: write-back padding in watched memory not tested\n";
                return;
            }
        } else {
            block = std::aligned_alloc(65536, bytes);
        }
        Require(block != nullptr, "cannot allocate the write-back padding block");
        auto* memory = static_cast<std::uint8_t*>(block);
        for (std::size_t i = 0; i < bytes; ++i) memory[i] = pattern(i);
        const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
        const auto address = blockAddress + test.offset;
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(block, bytes, true, true, true);
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            bool watched;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                HostImportFor(context, address, bytes);
                if (watched) ReleaseWatched(block, bytes);
            }
        } unregister{base, block, blockAddress, bytes, watched};
        if (HostImportFor(base, blockAddress, bytes) == nullptr) {
            std::cout << "host import of the write-back padding block refused: write-back padding not tested\n";
            return;
        }
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        resource.baseAddress = address;
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, test.mipCount, 0, 1};
            for (std::size_t pass = 0; pass < (test.direct ? colours.size() : 1u); ++pass) {
                const auto& colour = colours[pass];
                const auto commands = recorder.Commands();
                recorder.Keep(image);
                const VkClearColorValue value{{colour[0] / 255.0f, colour[1] / 255.0f, colour[2] / 255.0f, colour[3] / 255.0f}};
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                const auto slabRetilesBefore = StorageTexture::SlabRetiles();
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                const auto slabRetiled = StorageTexture::SlabRetiles() - slabRetilesBefore;
                if (test.direct && watched && UnitShadowEnabled() && AgcDriver::GuestMemory::Watched(address, surfaceBytes) && slabRetiled == 0) failures += what + ": no window was retiled straight into its unit-shadow slab\n";
                std::vector<std::byte> read(bytes);
                if (watched) AgcDriver::GuestMemory::Read(blockAddress, read);
                else std::memcpy(read.data(), memory, bytes);
                std::size_t texels = 0, garbage = 0, outside = 0;
                for (std::size_t i = 0; i < bytes; i += 4) {
                    const bool inside = i >= test.offset && i < test.offset + surfaceBytes;
                    bool isTexel = inside, isPattern = true;
                    for (std::size_t c = 0; c < 4; ++c) {
                        const auto byte = std::to_integer<std::uint8_t>(read[i + c]);
                        isTexel = isTexel && byte == colour[c];
                        isPattern = isPattern && byte == pattern(i + c);
                    }
                    if (isTexel) ++texels;
                    else if (!isPattern) ++(inside ? garbage : outside);
                }
                if (garbage != 0 || outside != 0 || texels != texelCount) {
                    failures += what + ", write-back " + std::to_string(pass + 1) + ": stored " + std::to_string(texels) + " texels of " + std::to_string(texelCount) + " and changed " + std::to_string(garbage) + " words no texel holds (" + std::to_string(outside) + " past the surface)\n";
                }
            }
        }
        recorder.Sync();
    }
    if (!failures.empty()) throw std::runtime_error("write-back padding:\n" + failures);
    std::cout << "write-back padding" << (watched ? " (watched)" : "") << ": ok\n";
}

void writeBackBlockTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0 || 65536 % base.hostImportAlignment != 0) {
        std::cout << "host imports unavailable: write-back blocks not tested\n";
        return;
    }
    constexpr std::size_t piece = 65536;
    constexpr std::size_t window = std::size_t{8} << 20u;
    struct Case {
        const char* name;
        std::size_t offset;
        std::size_t registeredEnd;
        bool alternate;
        bool direct;
        std::size_t imports;
    };
    const Case cases[] = {
        {"a 4 MiB image over 64 adjacent 64 KiB pieces", 0, 4u << 20u, false, true, 1},
        {"a 4 MiB image at 64 KiB over 66 adjacent 64 KiB pieces", piece, (4u << 20u) + 2 * piece, false, true, 2},
        {"a 4 MiB image over 64 KiB pieces of which every other one is not GPU mapped", 0, 4u << 20u, true, false, 0},
    };
    SetHostImportWindow(window);
    struct Window {
        ~Window() { SetHostImportWindow(0); }
    } restore;
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    std::string failures;
    for (const auto& test : cases) {
        GuestTextureResource resource{};
        resource.width = 1024;
        resource.height = 1024;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kLinear;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto surfaceBytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
        Require(surfaceBytes == (4u << 20u) && test.offset + surfaceBytes <= test.registeredEnd, std::string(test.name) + ": the surface does not fit its pieces");
        void* reserved = std::aligned_alloc(window, 2 * window);
        Require(reserved != nullptr, "cannot allocate the write-back block");
        auto* memory = static_cast<std::uint8_t*>(reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(reserved) + window - 1) / window * window));
        for (std::size_t i = 0; i < window; ++i) memory[i] = pattern(i);
        std::vector<std::size_t> pieces;
        for (std::size_t offset = 0; offset < test.registeredEnd; offset += piece) pieces.push_back(offset);
        {
            GuestAllocations::Mutation mutation;
            for (const auto offset : pieces) mutation.Add(memory + offset, piece, true, true, !test.alternate || offset / piece % 2 == 0);
        }
        ClearHostImports(base.device);
        const auto address = reinterpret_cast<std::uint64_t>(memory) + test.offset;
        const auto before = HostImportCounters();
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        resource.baseAddress = address;
        std::size_t texels = 0;
        bool direct = false;
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
            image->WriteBack();
            direct = !image->GuestSnapshotValid();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            for (std::size_t i = test.offset; i < test.offset + surfaceBytes; i += 4) texels += std::equal(red.begin(), red.end(), memory + i) ? 1 : 0;
        }
        const auto after = HostImportCounters();
        std::size_t made = 0;
        std::size_t segments = 0;
        for (std::size_t index = 0; index < ImportRequesterCount; ++index) made += after.requesters[index].made - before.requesters[index].made;
        segments = after.requesters[static_cast<std::size_t>(ImportRequester::Segment)].made - before.requesters[static_cast<std::size_t>(ImportRequester::Segment)].made;
        std::printf("%s: %zu imports made (%zu by write-back segments), %s write-back, %zu of %zu texels stored\n", test.name, made, segments, direct ? "GPU-direct" : "CPU", texels, surfaceBytes / 4);
        if (made != test.imports) failures += std::string(test.name) + ": the write-back made " + std::to_string(made) + " imports, expected " + std::to_string(test.imports) + "\n";
        if (direct != test.direct) failures += std::string(test.name) + ": the write-back took the " + (direct ? "GPU-direct" : "CPU") + " path\n";
        if (texels != surfaceBytes / 4) failures += std::string(test.name) + ": the write-back stored " + std::to_string(texels) + " of " + std::to_string(surfaceBytes / 4) + " texels\n";
        {
            GuestAllocations::Mutation mutation;
            for (const auto offset : pieces) mutation.Remove(memory + offset);
        }
        ClearHostImports(base.device);
        recorder.Sync();
        std::free(reserved);
    }
    if (!failures.empty()) throw std::runtime_error("write-back blocks:\n" + failures);
    std::cout << "write-back blocks: ok\n";
}

void singlePassTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    // Metal: the gate on the device's singlePassStorage is dropped. Each case sets the flag on its own context
    // copy (copies, then single pass); the twin's Device mirrors MetalDevice.mm's limits but not this flag.
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: single-pass storage moves not tested\n";
        return;
    }
    struct Case {
        const char* name;
        std::uint32_t format;
        TextureTileMode tileMode;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t mipCount;
        std::uint32_t pipeBankXor = 0;
    };
    constexpr Case cases[] = {
        {"R8 64 KiB R_X 200x150", 1, TextureTileMode::kR64KBX, 200, 150, 1},
        {"R16 64 KiB R_X 200x150, 3 mips", 7, TextureTileMode::kR64KBX, 200, 150, 3},
        {"RGBA8 64 KiB R_X 520x260, 6 mips", 56, TextureTileMode::kR64KBX, 520, 260, 6},
        {"RGBA16F 64 KiB R_X 384x256", 71, TextureTileMode::kR64KBX, 384, 256, 1},
        {"RGBA16F 64 KiB R_X 256x200, 4 mips", 71, TextureTileMode::kR64KBX, 256, 200, 4},
        {"RGBA32F 64 KiB R_X 136x72, 2 mips", 77, TextureTileMode::kR64KBX, 136, 72, 2},
        {"RGBA8 64 KiB S 300x170", 56, TextureTileMode::kStandard64KB, 300, 170, 1},
        {"RGBA16F 4 KiB S 100x70, 2 mips", 71, TextureTileMode::kStandard4KB, 100, 70, 2},
        {"R8 256 B S 90x40", 1, TextureTileMode::kStandard256B, 90, 40, 1},
        {"RGBA8 64 KiB R_X 520x260, 2 mips, pipe/bank XOR 5", 56, TextureTileMode::kR64KBX, 520, 260, 2, 5},
    };
    std::string failures;
    std::size_t tested = 0;
    for (const auto& test : cases) {
        if (!StorageFormatAvailable(base, test.format)) continue;
        const std::string what = std::string(watched ? "(watched) " : "") + test.name;
        GuestTextureResource resource{};
        resource.width = test.width;
        resource.height = test.height;
        resource.mipCount = test.mipCount;
        resource.tileMode = test.tileMode;
        resource.dimension = TextureDimension::k2D;
        resource.format = test.format;
        resource.pipeBankXor = test.pipeBankXor;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        const auto elementBytes = BytesPerElement(test.format);
        const auto bytes = (static_cast<std::size_t>(geometry.guestBytes) + 65535) / 65536 * 65536 + 65536;
        struct Result {
            std::vector<std::uint8_t> uploaded, memory, reuploaded;
            std::uint64_t moves = 0;
        };
        std::array<Result, 2> results;
        bool skipped = false;
        for (int mode = 0; mode < 2 && !skipped; ++mode) {
            auto context = base;
            context.singlePassStorage = mode == 1;
            void* block = nullptr;
            if (watched) {
                block = AllocateWatched(bytes, 65536);
            } else {
                block = std::aligned_alloc(65536, bytes);
            }
            if (block == nullptr) {
                std::cout << "no write watching: single-pass storage moves in watched memory not tested\n";
                return;
            }
            auto* memory = static_cast<std::uint8_t*>(block);
            std::uint32_t seed = 0x13579bdfu;
            const auto random = [&] {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                return static_cast<std::uint8_t>(seed >> 9);
            };
            for (std::size_t i = 0; i < bytes; ++i) memory[i] = random();
            const auto address = reinterpret_cast<std::uint64_t>(block);
            {
                GuestAllocations::Mutation mutation;
                mutation.Add(block, bytes, true, true, true);
            }
            struct Unregister {
                const Context& context;
                void* block;
                std::uint64_t address;
                std::size_t bytes;
                bool watched;
                ~Unregister() {
                    {
                        GuestAllocations::Mutation mutation;
                        mutation.Remove(block);
                    }
                    HostImportFor(context, address, bytes);
                    if (watched) ReleaseWatched(block, bytes);
                    else std::free(block);
                }
            } unregister{base, block, address, bytes, watched};
            if (HostImportFor(base, address, bytes) == nullptr) {
                std::cout << "host import refused: single-pass storage moves not tested\n";
                return;
            }
            TextureDetiler detiler(base);
            context.detiler = &detiler;
            resource.baseAddress = address;
            std::size_t linearBytes = 0;
            for (const auto& mip : geometry.mips) linearBytes += static_cast<std::size_t>(mip.width) * mip.height * elementBytes;
            const auto readImage = [&](const StorageTexture& image) {
                Buffer readback(context, linearBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                const auto commands = recorder.Commands();
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                std::vector<VkBufferImageCopy> copies;
                std::size_t at = 0;
                for (std::uint32_t level = 0; level < geometry.mips.size(); ++level) {
                    VkBufferImageCopy copy{};
                    copy.bufferOffset = at;
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                    copy.imageExtent = {geometry.mips[level].width, geometry.mips[level].height, 1};
                    copies.push_back(copy);
                    at += static_cast<std::size_t>(geometry.mips[level].width) * geometry.mips[level].height * elementBytes;
                }
                Metal::Vk::CmdCopyImageToBuffer(commands, image.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), static_cast<std::uint32_t>(copies.size()), copies.data());
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                const auto read = readback.Bytes();
                return std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(read.data()), reinterpret_cast<const std::uint8_t*>(read.data()) + linearBytes);
            };
            std::vector<std::uint8_t> texels(linearBytes);
            for (auto& value : texels) value = random();
            const auto movesBefore = StorageTexture::SinglePassMoves();
            {
                auto image = StorageTexture::Create(context, detiler, resource, 0);
                recorder.Keep(image);
                results[mode].uploaded = readImage(*image);
                Buffer staging(context, linearBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                std::memcpy(staging.Bytes().data(), texels.data(), linearBytes);
                const auto commands = recorder.Commands();
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                std::vector<VkBufferImageCopy> copies;
                std::size_t at = 0;
                for (std::uint32_t level = 0; level < geometry.mips.size(); ++level) {
                    VkBufferImageCopy copy{};
                    copy.bufferOffset = at;
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                    copy.imageExtent = {geometry.mips[level].width, geometry.mips[level].height, 1};
                    copies.push_back(copy);
                    at += static_cast<std::size_t>(geometry.mips[level].width) * geometry.mips[level].height * elementBytes;
                }
                Metal::Vk::CmdCopyBufferToImage(commands, staging.Handle(), image->Image(), VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(copies.size()), copies.data());
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                auto again = StorageTexture::Create(context, detiler, resource, 0);
                recorder.Keep(again);
                results[mode].reuploaded = readImage(*again);
                std::vector<std::byte> read(bytes);
                if (watched) AgcDriver::GuestMemory::Read(address, read);
                else {
                    StorageTexture::FlushPending(address, bytes, nullptr, "test", PublishScope::Whole);
                    recorder.Submit();
                    device.WaitQueue();
                    recorder.Sync();
                    std::memcpy(read.data(), memory, bytes);
                }
                results[mode].memory.assign(reinterpret_cast<const std::uint8_t*>(read.data()), reinterpret_cast<const std::uint8_t*>(read.data()) + bytes);
            }
            recorder.Sync();
            results[mode].moves = StorageTexture::SinglePassMoves() - movesBefore;
            if (results[mode].reuploaded != texels) failures += what + (mode == 1 ? " (single pass)" : " (copies)") + ": the second image does not hold the texels written back\n";
        }
        if (results[0].moves != 0 || results[1].moves < 3) failures += what + ": single-pass moves " + std::to_string(results[0].moves) + " with the copies and " + std::to_string(results[1].moves) + " in single pass (want 0 and at least 3)\n";
        if (results[0].uploaded != results[1].uploaded) failures += what + ": the first upload differs between the copies and single pass\n";
        if (results[0].memory != results[1].memory) {
            std::size_t differ = 0;
            for (std::size_t i = 0; i < results[0].memory.size(); ++i) differ += results[0].memory[i] != results[1].memory[i];
            failures += what + ": guest memory after the write-back differs in " + std::to_string(differ) + " bytes between the copies and single pass\n";
        }
        ++tested;
    }
    if (!failures.empty()) throw std::runtime_error("single-pass storage moves:\n" + failures);
    std::cout << "single-pass storage moves" << (watched ? " (watched)" : "") << ": ok (" << tested << " surfaces)\n";
}

// The test Device always enables host imports (RecorderTwinHarness.h), so upstream's "skipped, host imports are
// unavailable" exit (77) is unreachable here; the tests' own checks stay.
const bool registered = [] {
    AgcDriver::MetalTwins::RegisterRecorderMode("--read-written-staging-only", [] {
        int code = 0;
        const int failed = RunMode([&](const Device& device, Recorder& recorder) {
            if (!readWrittenStagingTests(device, recorder)) {
                code = 77;
                return;
            }
            std::cout << "read and written staging tests passed\n";
        });
        return failed != 0 ? failed : code;
    });
    AgcDriver::MetalTwins::RegisterRecorderMode("--storage-write-back-only", [] {
        return RunMode([](const Device& device, Recorder& recorder) {
            storageRefreshTests(device, recorder, false);
            storageRefreshTests(device, recorder, true);
            writeBackPaddingTests(device, recorder, false);
            writeBackPaddingTests(device, recorder, true);
            singlePassTests(device, recorder, false);
            singlePassTests(device, recorder, true);
        });
    });
    AgcDriver::MetalTwins::RegisterRecorderMode("--writeback-blocks-only", [] {
        return RunMode([](const Device& device, Recorder& recorder) { writeBackBlockTests(device, recorder); });
    });
    return true;
}();

}
