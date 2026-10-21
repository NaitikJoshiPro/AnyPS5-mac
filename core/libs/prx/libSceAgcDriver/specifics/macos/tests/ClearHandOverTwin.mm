// Metal twin of the tests/Recorder.cpp mode upstream added with the DCC clear-key hand-over (UPSYNC5, upstream
// 92063916d2 / b176274da1):
//   --clear-hand-over-only   agc_storage_clear_hand_over   clearHandOverTests (unwatched, then watched)
// The hand-over itself is in the Metal storage image twin (specifics/macos/src/Texture.mm clearHandOver /
// supersedeHandedOver, ported with the rest of UPSYNC5's Texture.cpp). The test function is upstream's, expected
// values included, but for:
// - its Vulkan calls: vkCmdClearColorImage -> Metal::Vk::CmdClearColorImage, vkCmdCopyImageToBuffer ->
//   Metal::Vk::CmdCopyImageToBuffer (as D3RecorderModes.mm and Upsync3RecorderModesA.mm);
// - the watched form. Upstream runs it over a watched host import and returns false (main skips) when the device
//   compares its imports. Metal imports are always compared (DECISIONS D19, TechnicalDebt metal-d3-import-watch:
//   keeping an import watched would have GPU write-backs store into write-protected pages), so on Metal the
//   memory a hand-over can happen in is watched guest memory without a host import. The twin's watched form runs
//   there: the same watched block, the images built with host imports off (Context::hostImportAlignment 0), so
//   their results are written back by the CPU path and the block stays watched; every check and expected count
//   (one hand-over per refresh, A pending, two units superseded, the same guest bytes and texels as with the
//   hand-over off) is upstream's. The unwatched form runs over a real Metal host import, as upstream's does.
// - `Device`, AllocateWatched and ReleaseWatched come from RecorderTwinHarness.h. The _WIN32 branches are dropped.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

bool clearHandOverTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    const std::string what = watched ? "clear-key hand-over (watched)" : "clear-key hand-over";
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: " << what << " not tested\n";
        return false;
    }
    constexpr std::size_t surfaceA = 256 * 256 * 4;
    constexpr std::size_t surfaceB = 128 * 128 * 8;
    constexpr std::size_t keyCount = surfaceB / 256;
    constexpr std::size_t bytes = surfaceA + 65536;
    static_assert(surfaceB <= surfaceA && keyCount <= bytes - surfaceA, "B lies over the first units of A, and B's DCC keys follow A inside the block");
    const auto decided = PrepareImportWatch(base);
    // Metal: imports are always compared (decided == Unwatch, D19), so upstream's watched form would stop here. The
    // twin runs it over watched memory that is not imported (see the file comment).
    Require(decided == ImportWatch::Unwatch, "Metal host imports stayed watched by default");
    void* block = nullptr;
    if (watched) {
        block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: " << what << " not tested\n";
            return false;
        }
    } else {
        block = std::aligned_alloc(65536, bytes);
    }
    Require(block != nullptr && reinterpret_cast<std::uint64_t>(block) % 65536 == 0, "cannot allocate the clear-key hand-over block");
    struct RestoreWatch {
        const Context& context;
        ImportWatch decided;
        ~RestoreWatch() { SetImportWatch(context, decided); }
    } restoreWatch{base, decided};
    auto* memory = static_cast<std::uint8_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        bool watched;
        ~Unregister() {
            StorageTexture::SetClearHandOver(true);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
            if (watched) ReleaseWatched(block, bytes);
        }
    } unregister{base, block, address, watched};
    if (!watched && HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the clear-key hand-over block refused: " << what << " not tested\n";
        return false;
    }
    if (watched && !AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "the watched block left the write watch: " << what << " not tested\n";
        return false;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    // Metal, watched form: no host imports for the images, so their results reach the watched block through the
    // CPU write-back and the block stays watched.
    if (watched) context.hostImportAlignment = 0;
    const auto surface = [&](std::uint32_t width, std::uint32_t height, std::uint32_t format, std::uint64_t dcc) {
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = width;
        resource.height = height;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kR64KBX;
        resource.dimension = TextureDimension::k2D;
        resource.format = format;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        resource.dccAddress = dcc;
        return resource;
    };
    const auto resourceA = surface(256, 256, 56, 0);
    const auto resourceB = surface(128, 128, 71, address + surfaceA);
    Require(DescribeSurface(resourceA).guestBytes == surfaceA && DescribeSurface(resourceB).guestBytes == surfaceB, "the clear-key hand-over surfaces have unexpected sizes");
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const auto draw = [&](const std::shared_ptr<StorageTexture>& image, VkClearColorValue value) {
        const auto commands = recorder.Commands();
        recorder.Keep(image);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        image->MarkDirty();
    };
    const auto readImage = [&](const std::shared_ptr<StorageTexture>& image, std::uint32_t width, std::uint32_t height, std::size_t texelBytes) {
        Buffer readback(context, static_cast<std::size_t>(width) * height * texelBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        Metal::Vk::CmdCopyImageToBuffer(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        const auto pixels = readback.Bytes();
        return std::vector<std::byte>(pixels.begin(), pixels.end());
    };
    const auto readMemory = [&](std::uint64_t from, std::size_t length) {
        std::vector<std::byte> read(length);
        AgcDriver::GuestMemory::Read(from, read);
        recorder.Sync();
        if (!watched) std::memcpy(read.data(), memory + (from - address), length);
        return read;
    };
    const auto filled = [](std::span<const std::byte> read, std::span<const std::uint8_t> texel) {
        for (std::size_t i = 0; i < read.size(); ++i) {
            if (std::to_integer<std::uint8_t>(read[i]) != texel[i % texel.size()]) return false;
        }
        return true;
    };
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    constexpr std::array<std::uint8_t, 4> blue{0, 0, 255, 255};
    constexpr std::array<std::uint8_t, 8> green{0, 0, 0x00, 0x3c, 0, 0, 0x00, 0x3c};
    constexpr std::array<std::uint8_t, 8> cleared{};
    enum class Order { Superseded, ReadFirst, Rewritten, CpuStore };
    struct Outcome {
        std::vector<std::byte> bytes;
        std::vector<std::byte> imageB;
        std::uint64_t handOvers = 0;
        std::uint64_t superseded = 0;
    };
    const auto run = [&](bool on, Order order) {
        const std::string where = what + (order == Order::Superseded ? ", A never read again" : order == Order::ReadFirst ? ", A read before B wrote" : order == Order::Rewritten ? ", A written again after the hand-over" : ", a CPU store before B wrote") + (on ? ", on" : ", off");
        StorageTexture::SetClearHandOver(on);
        recorder.Sync();
        std::memset(memory, 0x55, surfaceA);
        std::memset(memory + surfaceA, 0x00, keyCount);
        Outcome outcome{};
        const auto handOvers = StorageTexture::ClearHandOvers();
        const auto superseded = StorageTexture::HandedOverSuperseded();
        {
            auto imageA = StorageTexture::Create(context, detiler, resourceA, 0);
            auto imageB = StorageTexture::Create(context, detiler, resourceB, 0);
            draw(imageA, {{1.0f, 0.0f, 0.0f, 1.0f}});
            imageB->Refresh();
            if (on && watched) {
                Require(StorageTexture::ClearHandOvers() == handOvers + 1, where + ": B's refresh under clear keys did not hand A's results over");
                Require(StorageTexture::FindPending(address, surfaceA) == imageA, where + ": A's results did not stay pending");
                Require(std::all_of(memory, memory + 2 * 65536, [](std::uint8_t value) { return value == 0x55; }), where + ": B's refresh stored A's results over B");
            } else {
                Require(StorageTexture::ClearHandOvers() == handOvers, where + ": the hand-over ran while off");
            }
            outcome.imageB = readImage(imageB, 128, 128, 8);
            Require(filled(outcome.imageB, cleared), where + ": B does not hold its keys' clear");
            if (order == Order::ReadFirst) {
                Require(filled(readMemory(address, 65536), red), where + ": a read of A's memory before B wrote it did not see A's results");
                imageA->Refresh();
                Require(filled(readImage(imageA, 256, 256, 4), red), where + ": A read again as A lost its results");
            }
            if (order == Order::Rewritten) {
                draw(imageA, {{0.0f, 0.0f, 1.0f, 1.0f}});
                Require(filled(readMemory(address, 65536), blue), where + ": a read of A's memory did not see A's newer results");
                imageB->Refresh();
                Require(filled(readImage(imageB, 128, 128, 8), cleared), where + ": B refreshed again does not hold its keys' clear");
            }
            if (order == Order::CpuStore) {
                std::memset(memory + 65536, 0x77, 65536);
            }
            draw(imageB, {{0.0f, 1.0f, 0.0f, 1.0f}});
            Require(StorageTexture::FlushPending(address, bytes, nullptr, "clear-key hand-over test", PublishScope::Whole), where + ": nothing was pending at the end");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            outcome.bytes = readMemory(address, surfaceA);
            Require(filled(readImage(imageB, 128, 128, 8), green), where + ": B lost its results");
        }
        outcome.handOvers = StorageTexture::ClearHandOvers() - handOvers;
        outcome.superseded = StorageTexture::HandedOverSuperseded() - superseded;
        recorder.Sync();
        return outcome;
    };
    for (const auto order : {Order::Superseded, Order::ReadFirst, Order::Rewritten, Order::CpuStore}) {
        const std::string name = what + (order == Order::Superseded ? ", A never read again" : order == Order::ReadFirst ? ", A read before B wrote" : order == Order::Rewritten ? ", A written again after the hand-over" : ", a CPU store before B wrote");
        const auto off = run(false, order);
        const auto on = run(true, order);
        const auto& lastA = order == Order::Rewritten ? blue : red;
        if (order != Order::CpuStore) Require(filled(std::span(off.bytes).first(2 * 65536), green), name + ": B's results did not reach guest memory with the hand-over off");
        Require(filled(std::span(off.bytes).subspan(2 * 65536), lastA), name + ": A's results did not reach guest memory with the hand-over off");
        Require(on.imageB == off.imageB, name + ": B held other texels with the hand-over on than off");
        Require(on.bytes == off.bytes, name + ": the hand-over left other guest bytes than the old order");
        Require(off.handOvers == 0 && off.superseded == 0, name + ": the hand-over ran while off");
        if (!watched) {
            Require(on.handOvers == 0, name + ": results were handed over in memory the write tracker does not watch");
            continue;
        }
        if (order == Order::Superseded) Require(on.superseded == 2, name + ": B's results did not drop A's two units under B (dropped " + std::to_string(on.superseded) + ")");
        if (order == Order::ReadFirst) Require(on.superseded <= 1, name + ": a unit stored by the read was dropped again");
    }
    std::cout << what << ": ok\n";
    return true;
}

const bool registered = AgcDriver::MetalTwins::RegisterRecorderMode("--clear-hand-over-only", [] {
    return RunMode([](const Device& device, Recorder& recorder) {
        Require(clearHandOverTests(device, recorder, false), "the unwatched clear-key hand-over did not run");
        Require(clearHandOverTests(device, recorder, true), "the watched clear-key hand-over did not run");
    });
});

}
