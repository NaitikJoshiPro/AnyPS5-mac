// Metal twins of the tests/Recorder.cpp modes upstream added between e7c70a2df and 0f2683cb7 (v0.1.2 sync, D29):
// --integer-clear-only (agc_storage_integer_clear, integerClearTests), --written-buffer-order-only
// (agc_storage_written_buffer_order, writtenBufferOrderTests), --parked-import-unmap-only (agc_parked_import_unmap,
// parkedImportUnmapTests, upstream #4291) and --host-import-retirement-only (agc_host_import_retirement,
// hostImportRetirementTests). The test functions are copied unchanged but for their Vulkan calls, which go
// through the Metal command stream (vkCmdClearColorImage: Metal::Vk::CmdClearColorImage; vkCmdFillBuffer:
// CommandStream::FillBuffer on the buffer the guest-memory descriptor names). Expected values are upstream's.
// Metal notes:
// - parkedImportUnmapTests replaces the device: on Metal both test devices wrap the one system MTLDevice, so the
//   replacement's Context names the same device handle and its imports are the first device's; the test then
//   checks that a guest unmap of a range two device generations imported still runs, which is the property
//   upstream guards (the parked import must not block the unmap).
// - --copied-direct-only (copiedDirectOrderingTests) is ported in CopiedDirectOrderingTwin.mm (an MSL kernel stands
//   in for its SPIR-V pipeline).
// - --collect-mapping-history-only (agc_collect_mapping_history_{map,ring,shared,uncached}, upstream b8f6d316a9 and
//   c19dcecfec, UPSYNC6): collectMappingHistoryTests copied unchanged; it drives GuestMemory only, no device.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "../src/MetalCommandStream.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

bool collectMappingHistoryTests() {
    constexpr std::size_t bytes = 131072;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes, 65536));
    if (memory == nullptr) return false;
    GuestAllocations::Mutation().Add(memory, bytes, true, true, false);
    struct Cleanup {
        std::uint8_t* memory;
        ~Cleanup() {
            GuestAllocations::Mutation().Remove(memory);
            ReleaseWatched(memory, bytes);
        }
    } cleanup{memory};
    const auto address = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    const auto before = AgcDriver::GuestMemory::CollectWrites(address, bytes);
    Require(before != 0, "the mapping fixture has no complete write history");
    memory[4096] = 0x22;
    GuestAllocations::Mutation().Remove(memory);
    GuestAllocations::Mutation().Add(memory, bytes, true, true, false);
    Require(AgcDriver::GuestMemory::CollectWrites(address, bytes) != 0, "the replacement mapping lost write tracking");
    Require(!AgcDriver::GuestMemory::UnchangedSince(address, bytes, before), "a cached collect hid a CPU store across allocation replacement");
    Require(memory[4096] == 0x22, "collecting the replacement mapping changed its CPU bytes");
    return true;
}

void integerClearTests(const Device& device) {
    const auto& context = device.GetContext();
    std::size_t tested = 0;
    for (const std::uint32_t format : {5u, 11u, 20u}) {
        if (!StorageFormatAvailable(context, format)) continue;
        ++tested;
        Require(StorageClearAvailable(context, format, DccKeys::Clear0000), "an integer storage format refused the 0000 clear");
        for (const auto keys : {DccKeys::Clear0001, DccKeys::Clear1111, DccKeys::ClearRegister, DccKeys::Uncompressed}) {
            Require(!StorageClearAvailable(context, format, keys), "an integer storage format took a clear code other than 0000");
        }
    }
    Require(tested == 3, "an integer storage format has no storage form");
    std::cout << "integer storage clears: " << tested << " formats ok\n";
}

bool writtenBufferOrderTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: written buffer order not tested\n";
        return false;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t bytes = side * side * 4;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: written buffer order not tested\n";
        return false;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } releaseBlock{block};
    std::memset(block, 0x55, bytes);
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
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the written buffer order block refused: written buffer order not tested\n";
        return false;
    }
    std::cout << "written buffer order: host imports are " << (AgcDriver::GuestMemory::Watched(address, bytes) ? "watched" : "compared") << "\n";
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
    Require(DescribeSurface(resource).guestBytes == bytes, "the written buffer order surface has an unexpected size");
    const auto settle = [&] {
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    };
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
        {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        }
        constexpr std::size_t written = 40800;
        GuestBufferMemory memory(context);
        memory.AcquireRegistered();
        memory.AddWritable(address, written, false, false);
        memory.Upload(true);
        std::uint32_t adjustment = 0;
        const auto view = memory.Descriptor(address, written, adjustment);
        Require(adjustment == 0, "the written buffer view is not aligned to its element");
        {
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            Metal::Stream(commands).FillBuffer(Metal::BufferOf(view.buffer), static_cast<NSUInteger>(view.offset), written, 0xa5a5a5a5u);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        memory.RecordCopyBacks(recorder);
        memory.MarkDirectWrites();
        settle();
        StorageTexture::FlushAllPending("written buffer order test");
        settle();
        std::vector<std::byte> read(bytes);
        AgcDriver::GuestMemory::Read(address, read);
        Require(std::all_of(read.begin(), read.begin() + written, [](std::byte value) { return value == std::byte{0xa5}; }), "storage results pending under a buffer a later use wrote were stored over the buffer's contents");
        for (std::size_t i = written; i < bytes; ++i) Require(std::to_integer<std::uint8_t>(read[i]) == (i % 4 == 1 || i % 4 == 2 ? 0 : 255), "the storage results outside the written buffer did not reach guest memory");
    }
    recorder.Sync();
    return true;
}

void parkedImportUnmapTests(const Device& device) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: parked import unmap not tested\n";
        return;
    }
    const auto alignment = static_cast<std::size_t>(std::max<VkDeviceSize>(65536u, context.hostImportAlignment));
    const auto bytes = alignment * 4u;
    void* block = std::aligned_alloc(alignment, bytes);
    Require(block != nullptr, "cannot allocate the parked import unmap block");
    const auto release = [&] { std::free(block); };
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    std::unique_ptr<Device> replacement;
    bool imported = false;
    {
        std::lock_guard gpu(GpuMutex());
        imported = HostImportFor(context, address, bytes) != nullptr;
    }
    if (imported) {
        replacement = std::make_unique<Device>();
        std::lock_guard gpu(GpuMutex());
        imported = HostImportFor(replacement->GetContext(), address, bytes) != nullptr;
    }
    if (!imported) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        if (replacement) ClearHostImports(replacement->GetContext().device);
        release();
        std::cout << "host import refused: parked import unmap not tested\n";
        return;
    }
    bool unmapped = false;
    try {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(block, bytes, [&](const void*, std::size_t, const void*, bool) { unmapped = true; });
    } catch (const std::exception& error) {
        ClearHostImports(replacement->GetContext().device);
        ClearHostImports(context.device);
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        release();
        throw std::runtime_error(std::string("guest unmap after a device replacement: ") + error.what());
    }
    ClearHostImports(replacement->GetContext().device);
    ClearImageMirrors(replacement->GetContext().device);
    replacement.reset();
    release();
    Require(unmapped, "guest unmap of a range a replaced device imported did not run");
    std::cout << "parked import unmap: ok\n";
}

void hostImportRetirementTests(const Device& device) {
    const auto& context = device.GetContext();
    constexpr std::size_t UnitBytes = 65536;
    constexpr std::size_t AllocationBytes = 4 * UnitBytes;
    constexpr std::size_t CopyBytes = 256;
    void* block = std::aligned_alloc(UnitBytes, AllocationBytes);
    Require(block != nullptr, "cannot allocate the host import retirement block");
    const auto release = [](void* pointer) { std::free(pointer); };
    std::unique_ptr<void, decltype(release)> allocation(block, release);
    auto* memory = static_cast<std::byte*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(context);
    Recorder recorder(context);
    recorder.Activate();
    Buffer readback(context, CopyBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::array<bool, 4> registered{};
    const auto cleanup = [&] {
        if (!gpu.owns_lock()) gpu.lock();
        recorder.Sync();
        recorder.FlushDeferredReleases();
        ClearHostImports(context.device);
        GuestAllocations::Mutation mutation;
        for (std::size_t index = 0; index < registered.size(); ++index) {
            if (!registered[index]) continue;
            mutation.Remove(memory + index * UnitBytes);
            registered[index] = false;
        }
        SetHostImportWindow(0);
    };
    const auto add = [&](std::size_t index) {
        GuestAllocations::Mutation mutation;
        mutation.Add(memory + index * UnitBytes, UnitBytes, true, true, true);
        registered[index] = true;
    };
    const auto mapped = [&](std::size_t index) {
        GuestAllocations::Mapped result;
        for (const auto& range : GuestAllocations::GuestAllocationsAcquire_nid_postfix()) {
            if (range->address == address + index * UnitBytes) result.push_back(range);
        }
        Require(result.size() == 1, "host import retirement lost its registered range");
        return result;
    };
    bool deferredEveryRetirement = true;
    bool preservedEveryRecorder = true;
    try {
        const std::array<const char*, 3> shapes{"same-base growth", "earlier-base overlap", "predecessor split"};
        for (std::size_t shape = 0; shape < shapes.size(); ++shape) {
            SetHostImportWindow(0);
            Require(ImportMappedRanges(context, {}, GuestAllocations::GuestAllocationsGeneration_nid_postfix(), true), "host import retirement could not adopt the device");
            const std::size_t first = shape == 1 ? 1 : 0;
            add(first);
            if (shape == 2) add(1);
            for (std::size_t index = 0; index < AllocationBytes; ++index) memory[index] = static_cast<std::byte>((index * 37u + shape * 19u + 11u) & 0xffu);
            const auto initial = mapped(first);
            const auto initialGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
            gpu.unlock();
            const bool fresh = ImportMappedRanges(context, initial, initialGeneration, false);
            const bool unchanged = ImportMappedRanges(context, initial, initialGeneration, false);
            gpu.lock();
            Require(fresh && unchanged, "host import retirement blocked a fresh or unchanged fast import");
            const auto sourceAddress = address + (shape == 2 ? UnitBytes : first * UnitBytes);
            const auto* imported = HostImportFor(context, sourceAddress, CopyBytes);
            Require(imported != nullptr, "host import retirement could not import its source");
            const auto source = imported->buffer;
            const auto sourceOffset = sourceAddress - imported->base;
            std::array<std::byte, CopyBytes> expected{};
            std::memcpy(expected.data(), reinterpret_cast<const void*>(sourceAddress), CopyBytes);
            recorder.Commands();
            recorder.Submit();
            // Upstream waits for the queue (vkQueueWaitIdle) and leaves the batch unreaped; the twin harness's
            // WaitQueue is Recorder::Sync, which reaps it, so wait on the batch's serial instead.
            recorder.WaitSerial(recorder.NewestSubmitted());
            Require(!recorder.Recording() && recorder.InFlightBatches() == 1, ("host import retirement requires a submitted batch without an open batch (shape " + std::to_string(shape) + ", recording " + std::to_string(recorder.Recording()) + ", in flight " + std::to_string(recorder.InFlightBatches()) + ")").c_str());
            std::size_t changed = 0;
            if (shape == 0) {
                changed = 1;
                add(changed);
            } else if (shape == 1) {
                add(changed);
            } else {
                SetHostImportWindow(UnitBytes);
                add(3);
                changed = 1;
            }
            const auto ranges = mapped(changed);
            const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
            gpu.unlock();
            const bool deferred = !ImportMappedRanges(context, ranges, generation, false);
            gpu.lock();
            const bool preserved = !recorder.Recording() && recorder.InFlightBatches() == 1;
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            CopyBuffer(context, commands, source, sourceOffset, readback.Handle(), 0, CopyBytes);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            Require(ImportMappedRanges(context, ranges, generation, true), "host import retirement refused its retry under the GPU lock");
            Require(HostImportFor(context, sourceAddress, CopyBytes) != nullptr, "host import retirement lost the replacement import");
            recorder.Sync();
            readback.Invalidate();
            Require(std::equal(expected.begin(), expected.end(), readback.Bytes().begin()), "host import retirement changed bytes copied from the retired import");
            std::cout << "host import retirement: " << shapes[shape] << " GPU readback verified; deferred=" << deferred << "; recorder preserved=" << preserved << '\n';
            deferredEveryRetirement = deferredEveryRetirement && deferred;
            preservedEveryRecorder = preservedEveryRecorder && preserved;
            cleanup();
        }
    } catch (...) {
        cleanup();
        throw;
    }
    Require(deferredEveryRetirement, "unlocked mapping retired an import instead of requesting the GPU lock");
    Require(preservedEveryRecorder, "unlocked mapping changed recorder batches");
    std::cout << "Host import retirement locking and GPU readback tests passed\n";
}

// Runs a mode with the device alone, as upstream main runs --host-import-retirement-only and --integer-clear-only
// (before it takes the GPU lock and activates a recorder for the other modes; integerClearTests runs after, but
// reads nothing the recorder holds).
template <typename Test>
int RunDevice(Test test) {
    @autoreleasepool {
        try {
            Device device;
            return test(device);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}

// Runs a mode as upstream main runs --parked-import-unmap-only: an active recorder, then the GPU lock released.
template <typename Test>
int RunUnlocked(Test test) {
    @autoreleasepool {
        try {
            Device device;
            std::unique_lock gpu(GpuMutex());
            std::cout << "host imports " << (PrepareImportWatch(device.GetContext()) == ImportWatch::Unwatch ? "are compared" : "stay watched") << '\n';
            Recorder recorder(device.GetContext());
            recorder.Activate();
            gpu.unlock();
            test(device);
            gpu.lock();
            recorder.Sync();
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}

const bool integerClear = AgcDriver::MetalTwins::RegisterRecorderMode("--integer-clear-only", [] {
    return RunDevice([](const Device& device) {
        integerClearTests(device);
        return 0;
    });
});

const bool writtenBufferOrder = AgcDriver::MetalTwins::RegisterRecorderMode("--written-buffer-order-only", [] {
    int code = 0;
    const int failed = RunMode([&](const Device& device, Recorder& recorder) {
        if (!writtenBufferOrderTests(device, recorder)) {
            code = 77;
            return;
        }
        std::cout << "written buffer order tests passed\n";
    });
    return failed != 0 ? failed : code;
});

const bool parkedImportUnmap = AgcDriver::MetalTwins::RegisterRecorderMode("--parked-import-unmap-only", [] {
    return RunUnlocked([](const Device& device) { parkedImportUnmapTests(device); });
});

const bool hostImportRetirement = AgcDriver::MetalTwins::RegisterRecorderMode("--host-import-retirement-only", [] {
    return RunDevice([](const Device& device) {
        const auto alignment = device.GetContext().hostImportAlignment;
        if (alignment == 0 || alignment > 65536) {
            std::cout << "skipped, host import retirement requires host imports aligned within 64 KiB; alignment=" << alignment << '\n';
            return 77;
        }
        hostImportRetirementTests(device);
        return 0;
    });
});

const bool collectMappingHistory = AgcDriver::MetalTwins::RegisterRecorderMode("--collect-mapping-history-only", [] {
    if (!collectMappingHistoryTests()) return 77;
    std::cout << "collect mapping history tests passed\n";
    return 0;
});

}
