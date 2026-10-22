// Metal twins of the tests/Recorder.cpp modes upstream added after 0214069e3 (UPSYNC3, part B):
// --sparse-run-import-only and --sparse-run-copied-only (agc_sparse_run_import, agc_sparse_run_import_copied;
// sparseRunImportTests), --draw-snapshot-epoch-only (agc_draw_snapshot_epoch; drawSnapshotEpochTests, upstream
// 11c4a257c), --tiled-resolve-only (agc_tiled_resolve; tiledResolveTests), --slice-dirty-only (agc_slice_dirty;
// sliceDirtyTests), --metadata-pass-only (agc_metadata_pass; metadataPassTests), --pipeline-cache-file-only
// (agc_pipeline_cache_file; pipelineCacheFileTests) and --resident-present-only (agc_resident_present; upstream's
// tests/ResidentPresent.cpp RunResidentPresentTests, which T1-tests.cmake drops into RecorderTwin.mm).
// The test functions are upstream's, copied unchanged but for these deviations:
// - Vulkan entry points (context.Function<PFN_vkX>("vkX")) are the Metal twins of the same signature,
//   Metal::Vk::X (MetalVulkanImages.h); vkCmdFillBuffer is CommandStream::FillBuffer on the Metal buffer of the
//   host import; the Win32 branches are dropped (unifdef -U_WIN32).
// - Upstream's "... not tested" exits for a device without host imports or write watching throw instead (as in
//   DrawSnapshotCapTwin.mm): the Metal device has both (16 KiB no-copy imports, GuestWriteWatch), so a pass means
//   every check ran. sparseRunImportTests' refused import therefore fails instead of returning 77.
// - tiledResolveTests: the Metal device always has a graphics queue (upstream's device.Graphics() check is gone);
//   the 4x check stays (Apple GPUs render 1, 2 and 4 samples).
// - metadataPassTests / tiledResolveTests call upstream's Graphics::RunColorMetadataPass (Graphics/src/Draw.cpp,
//   compiled unchanged on macOS) over the test Context: everything it reaches is a Metal twin (CachedStorageSurface,
//   StorageTexture::FillClear/FindPending, ResolveMultisampleTarget, TextureDetiler). The Metal device itself runs
//   its own VulkanDevice::ColorMetadataPass (MetalDraw.mm), which these tests do not reach.
// - conversionTest (resident present): VkDeviceMemory is a token on Metal, so the image takes memory type 0
//   instead of a device-local type looked up in context.memory. What it checks exists in the Metal driver: the
//   presentation decisions and slices (Execution/src/DisplayFormat.cpp, compiled unchanged) and GpuColorTransfer's
//   GPU detile of guest memory and of a resident image (the D4 twin GpuColorTransfer.mm, PQ mode included). The
//   Metal presenter does not take the resident path yet (it decodes on the CPU: TechnicalDebt metal-d6-present,
//   metal-present-pass), so this checks the pieces, not a present.
// - pipelineCacheFileTests: upstream checks that VkPipelineCache objects sharing ANYPS5_SHADER_CACHE_DIR merge
//   into one vk-pipelines-* file whichever saves first, that an object adding no pipelines does not rewrite it, and
//   that an invalid file is replaced. The Metal device's pipeline cache is Metal::PipelineCache (D5, the
//   per-device MTL4Archive generations under an archive directory, DECISIONS D51; D5RecorderModes.mm maps
//   --pipeline-cache-only the same way): "holds pipeline i" is "building it compiles nothing" (Statistics()
//   .pipelinesCompiled unchanged, an archive hit), the two pipelines are MetalPipelineCacheChecks.h's kernel with
//   function constant 3 and 5 (upstream: SampleLod and SampleArray), "the cache file" is the set of
//   *.mtl4archive generations (each writing cache saves its own, so there is not exactly one), and the invalid
//   file is every generation overwritten. Under MTL_SHADER_VALIDATION the archive is off: the test checks that
//   instead, as D5's does. agc_pipeline_cache_file has no SKIP_RETURN_CODE, so a machine without a Metal 4 device
//   fails it (as agc_pipeline_cache).
// Expected values are upstream's.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "MetalPipelineCacheChecks.h"
#include "../src/MetalCommandStream.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/PqToSdr.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuColorTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <sys/mman.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {


bool sparseRunImportTests(const Device& device, bool copied) {
    const auto& context = device.GetContext();
    constexpr std::size_t unit = 65536;
    constexpr std::size_t heap = 8 * unit;
    struct Heap {
        const Context& context;
        void* area = nullptr;
        std::uint8_t* guest = nullptr;
        std::size_t mapped = 0;
        ~Heap() {
            ClearHostImports(context.device);
            {
                GuestAllocations::Mutation mutation;
                for (std::size_t index = 0; index < mapped; ++index) mutation.Remove(guest + index * unit);
            }
            SetHostImportWindow(0);
            if (area != nullptr) munmap(area, heap + unit);
        }
    } state{context};
    void* area = mmap(nullptr, heap + unit, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(area != MAP_FAILED, "cannot reserve the sparse run heap");
    state.area = area;
    state.guest = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(area) + unit - 1) & ~std::uintptr_t{unit - 1});
    auto* guest = state.guest;
    const auto address = reinterpret_cast<std::uint64_t>(guest);
    const auto map = [&] {
        auto* piece = guest + state.mapped * unit;
        Require(mprotect(piece, unit, PROT_READ | PROT_WRITE) == 0, "cannot map a piece of the sparse run heap");
        for (std::size_t at = 0; at < unit; ++at) piece[at] = static_cast<std::uint8_t>((state.mapped * unit + at) * 7u + 3u);
        GuestAllocations::Mutation mutation;
        mutation.Add(piece, unit, true, true, true);
        ++state.mapped;
    };
    for (int piece = 0; piece < 3; ++piece) map();
    PrepareImportWatch(context);
    SetHostImportWindow(unit);
    if (HostImportFor(context, address, unit) == nullptr) {
        throw std::runtime_error("required on Metal (host imports and write watching exist): host import of the sparse run heap refused: sparse run imports not tested");
    }
    Recorder recorder(context);
    recorder.Activate();
    const auto run = [&] { return HostImportFor(context, address, state.mapped * unit); };
    const auto bound = [&](GuestBufferMemory& memory, std::uint64_t begin, std::uint64_t bytes) {
        std::uint32_t adjustment = 0;
        const auto view = memory.Descriptor(begin, bytes, adjustment);
        const auto* import = run();
        return import != nullptr && view.buffer == import->buffer;
    };
    const auto upload = [&](bool written, std::uint64_t pastRun, const std::function<void(GuestBufferMemory&)>& check) {
        GuestBufferMemory memory(context);
        if (written) memory.AddWritable(address, heap);
        else memory.AddReadable(address, heap);
        if (pastRun != 0) memory.AddReadable(address + pastRun, 16);
        memory.Upload(true);
        check(memory);
        memory.WriteBack();
    };
    const auto committed = 3 * unit;
    upload(false, 0, [&](GuestBufferMemory& memory) {
        Require(!bound(memory, address, heap) && memory.InPlaceReads().empty(), "a sparse run heap was joined the first time its run was seen");
    });
    if (copied) {
        upload(false, 5 * unit, [&](GuestBufferMemory& memory) {
            std::uint32_t adjustment = 0;
            const auto view = memory.Descriptor(address, heap, adjustment);
            Require(!bound(memory, address, heap) && memory.InPlaceReads().empty(), "a sparse run heap with a view past its stable run is bound in place");
            Buffer readback(context, committed, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            {
                std::unique_lock gpu(GpuMutex());
                const auto commands = recorder.Commands();
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                CopyBuffer(context, commands, view.buffer, view.offset, readback.Handle(), 0, committed);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
                recorder.Sync();
            }
            readback.Invalidate();
            Require(std::memcmp(readback.Bytes().data(), guest, committed) == 0, "the GPU does not read a copied sparse run heap's mapped bytes");
        });
        recorder.Sync();
        std::cout << "sparse run import tests passed (copied)\n";
        return true;
    }
    upload(false, 0, [&](GuestBufferMemory& memory) {
        std::uint32_t adjustment = 0;
        const auto view = memory.Descriptor(address, heap, adjustment);
        const auto ranges = memory.AddressRanges();
        Require(adjustment == 0 && view.range == committed && ranges.size() == 1 && ranges[0].begin == address && ranges[0].end == address + committed, "a sparse run heap's view or address range is not cut at its last committed byte");
        const auto* import = run();
        Require(import != nullptr && view.buffer == import->buffer && view.offset == address - import->base && ranges[0].deviceAddress == import->address + (address - import->base), "a sparse run heap is not bound in place from its run's import");
        const auto direct = memory.DirectRegions();
        const auto reads = memory.InPlaceReads();
        const auto cuts = memory.CutRegions();
        Require(direct.has_value() && direct->size() == 1 && direct->front() == std::pair<std::uint64_t, std::uint64_t>{address, address + committed} && reads.size() == 1 && reads.front() == direct->front(), "a sparse run heap bound in place is not named up to its import's end for reuse");
        Require(cuts.size() == 1 && cuts.front().begin == address && cuts.front().end == address + heap && cuts.front().committedEnd == address + committed, "a sparse run heap bound in place does not report its cut");
        Buffer readback(context, committed, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        {
            std::unique_lock gpu(GpuMutex());
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            CopyBuffer(context, commands, view.buffer, view.offset, readback.Handle(), 0, committed);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Sync();
        }
        readback.Invalidate();
        Require(std::memcmp(readback.Bytes().data(), guest, committed) == 0, "the GPU does not read a sparse run heap's bytes through its in-place view");
    });
    upload(true, 0, [&](GuestBufferMemory& memory) {
        Require(!bound(memory, address, heap), "a written sparse run heap is bound in place");
    });
    upload(false, 5 * unit, [&](GuestBufferMemory& memory) {
        Require(!bound(memory, address, heap), "a sparse run heap with a view past its run is bound in place");
    });
    {
        auto regionContext = context;
        DescriptorCache cache(regionContext);
        regionContext.descriptorCache = &cache;
        Recorder regionRecorder(regionContext);
        regionRecorder.Activate();
        ShaderRecompiler::RecompileResult program;
        ShaderRecompiler::DescriptorBinding binding;
        binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
        binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
        binding.descriptorSet = 0;
        binding.binding = 0;
        binding.count = 1;
        binding.guestDescriptor = {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, static_cast<std::uint32_t>(heap), 0x31000000u};
        binding.bufferWritten = {false};
        program.bindings.push_back(binding);
        program.pushConstants.resize(16);
        program.memoryOffsetDword = 0;
        const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
        {
            ShaderResources resources(regionContext, compute);
            const auto reads = resources.InPlaceReads();
            Require(resources.Reusable() && reads.size() == 1 && reads.front() == std::pair<std::uint64_t, std::uint64_t>{address, address + committed}, "a set over a sparse run heap bound in place is not reusable");
            Require(resources.Revalidate(compute), "a set over an unchanged sparse run heap failed its proof");
            map();
            Require(!resources.Revalidate(compute), "a set over a sparse run heap mapped past its cut was reused");
            regionRecorder.Submit();
            device.WaitQueue();
            regionRecorder.Sync();
        }
        recorder.Activate();
        upload(false, 0, [&](GuestBufferMemory& memory) {
            Require(!bound(memory, address, heap), "a grown sparse run heap was joined again the first time its grown run was seen");
        });
        upload(false, 0, [&](GuestBufferMemory& memory) {
            std::uint32_t adjustment = 0;
            const auto view = memory.Descriptor(address, heap, adjustment);
            Require(bound(memory, address, heap) && view.range == 4 * unit, "a grown sparse run heap is not joined again over its grown run");
        });
    }
    recorder.Sync();
    std::cout << "sparse run import tests passed (in place)\n";
    return true;
}

void drawSnapshotEpochTests(const Device& device) {
    const auto& context = device.GetContext();
    constexpr std::size_t bytes = 4 * 65536;
    std::unique_lock gpu(GpuMutex());
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        throw std::runtime_error("required on Metal (host imports and write watching exist): host imports unavailable: draw snapshot reuse within a collect epoch not tested");
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    auto* guest = static_cast<std::uint8_t*>(block);
    for (std::size_t at = 0; at < bytes; ++at) guest[at] = static_cast<std::uint8_t>(at * 7u + 3u);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        throw std::runtime_error("required on Metal (host imports and write watching exist): host import of the block refused: draw snapshot reuse within a collect epoch not tested");
    }
    constexpr std::size_t elementOffset = 4096;
    constexpr std::size_t elementBytes = 2 * 65536;
    const auto element = address + elementOffset;
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.guestDescriptor = {static_cast<std::uint32_t>(element), static_cast<std::uint32_t>(element >> 32u) & 0xffffu, static_cast<std::uint32_t>(elementBytes), 0x31000000u};
    binding.bufferWritten = {false};
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    auto snapshotContext = context;
    DescriptorCache cache(snapshotContext);
    snapshotContext.descriptorCache = &cache;
    Recorder snapshotRecorder(snapshotContext);
    snapshotRecorder.Activate();
    {
        ShaderResources resources(snapshotContext, compute);
        const auto snapshot = [&] {
            auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
            Require(bindings != nullptr && bindings->snapshots.size() == 1, "read-only draw input was not snapshotted");
            auto buffer = bindings->snapshots[0].buffer;
            const auto contents = buffer->Bytes();
            Require(contents.size() == elementBytes && std::memcmp(contents.data(), reinterpret_cast<const void*>(element), elementBytes) == 0, "a draw snapshot does not hold the guest bytes of its draw");
            return buffer;
        };
        AgcDriver::GuestMemory::BumpCollectEpoch();
        const auto first = snapshot();
        Require(snapshot() == first, "a second draw in the same collect epoch copied its unchanged input again");
        guest[elementOffset + 65536 + 17] ^= 0xffu;
        AgcDriver::GuestMemory::BumpCollectEpoch();
        const auto changed = snapshot();
        Require(changed != first, "a draw after a store and a new collect epoch reused the snapshot taken before the store");
        Require(snapshot() == changed, "a second draw after the store copied its input again");
        snapshotRecorder.Sync();
    }
    std::cout << "draw snapshot reuse within a collect epoch: ok\n";
}

void metadataPassTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        throw std::runtime_error("required on Metal (host imports and write watching exist): host imports unavailable: resident metadata passes not tested");
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = std::aligned_alloc(65536, bytes);
    Require(block != nullptr, "cannot allocate the metadata pass block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
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
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        throw std::runtime_error("required on Metal (host imports and write watching exist): host import of the metadata pass block refused: resident metadata passes not tested");
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    ColorTarget color{};
    color.address = address;
    color.extent = {side, side};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.bytes = surfaceBytes;
    color.componentMapping = 0xe4u;
    color.tileMode = ColorTileMode::RenderTarget;
    color.elementBytes = 4;
    color.dccAddress = address + surfaceBytes;
    color.dccPipeAligned = true;
    color.clearWords = {0x80402010u, 0};
    const ColorMetadataPass pass{ColorMetadataPass::Mode::EliminateFastClear, {color}};
    const auto memoryHolds = [&](std::array<std::uint8_t, 4> texel) {
        StorageTexture::FlushPending(address, surfaceBytes, nullptr, "test");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        std::vector<std::uint8_t> stored(surfaceBytes);
        AgcDriver::GuestMemory::Read(address, std::as_writable_bytes(std::span(stored)), 1);
        for (std::size_t i = 0; i < surfaceBytes; ++i) {
            if (stored[i] != texel[i % 4]) return false;
        }
        return true;
    };
    const auto keysUncompressed = [&] { return ReadDccKeys(color.dccAddress, surfaceBytes) == DccKeys::Uncompressed; };
    const auto extent = DccKeyCount(TextureTileMode::kR64KBX, 4, side, side, surfaceBytes);
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0x20, extent);
    RunColorMetadataPass(context, pass);
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr && texels[0] == 0x55, "the register fast clear eliminate did not stay in the resident image");
    Require(keysUncompressed(), "the register fast clear eliminate left the keys compressed");
    Require(CurrentDccKeys(color.dccAddress, surfaceBytes, extent) == DccKeys::Uncompressed, "the register fast clear eliminate did not store uncompressed keys over the target's console DCC extent");
    Require(memoryHolds({0x10, 0x20, 0x40, 0x80}), "the register fast clear eliminate did not store CB_COLOR_CLEAR_WORD");
    Require(std::all_of(keys, keys + extent, [](std::uint8_t key) { return key == 0xff; }), "the stored keys are not uncompressed");
    std::memset(keys, 0xc0, keyCount);
    RunColorMetadataPass(context, {ColorMetadataPass::Mode::DccDecompress, {color}});
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr, "the DCC decompress did not stay in the resident image");
    Require(keysUncompressed(), "the DCC decompress left the keys compressed");
    Require(memoryHolds({0xff, 0xff, 0xff, 0xff}), "the DCC decompress did not store the 1111 value");
    {
        const auto* import = HostImportFor(base, address, bytes);
        Require(import != nullptr, "the metadata pass block lost its import");
        const auto commands = recorder.Commands();
        Metal::Stream(commands).FillBuffer(Metal::BufferOf(import->buffer), static_cast<NSUInteger>(color.dccAddress - import->base), keyCount, 0x20202020u);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
        recorder.NotePendingWrite(color.dccAddress, keyCount);
        Require(keys[0] == 0xff, "the recorded key fill landed before its batch ran");
        Require(CurrentDccKeys(color.dccAddress, surfaceBytes) == DccKeys::ClearRegister, "the keys read past a pending fast-clear fill");
    }
    std::memset(texels, 0x77, surfaceBytes);
    RunColorMetadataPass(context, pass);
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr && texels[0] == 0x77, "a fast clear recorded before the pass fell back to CPU write-back");
    Require(keysUncompressed() && memoryHolds({0x10, 0x20, 0x40, 0x80}), "a fast clear recorded before the pass was not eliminated");
    std::memset(texels, 0x66, surfaceBytes);
    RunColorMetadataPass(context, pass);
    Require(memoryHolds({0x66, 0x66, 0x66, 0x66}), "a pass over uncompressed keys changed the texels");
    ClearCachedTextures(context.device);
    GuestTextureResource plain{};
    plain.baseAddress = address;
    plain.width = side;
    plain.height = side;
    plain.mipCount = 1;
    plain.tileMode = TextureTileMode::kR64KBX;
    plain.dimension = TextureDimension::k2D;
    plain.format = 56;
    plain.dstSelX = 4;
    plain.dstSelY = 5;
    plain.dstSelZ = 6;
    plain.dstSelW = 7;
    const auto unkeyed = CachedStorageSurface(context, plain);
    std::memset(keys, 0xc0, keyCount);
    RunColorMetadataPass(context, pass);
    const auto keyed = StorageTexture::FindPending(address, surfaceBytes);
    Require(keyed != nullptr && keyed != unkeyed && keyed->Descriptor().dccAddress == color.dccAddress, "a pass over a resident image under other keys did not remake it under the target's keys");
    Require(keysUncompressed() && memoryHolds({0xff, 0xff, 0xff, 0xff}), "a pass over a remade resident image did not store the 1111 value");
    {
        const auto* import = HostImportFor(base, address, bytes);
        Require(import != nullptr, "the metadata pass block lost its import");
        const auto commands = recorder.Commands();
        Metal::Stream(commands).FillBuffer(Metal::BufferOf(import->buffer), static_cast<NSUInteger>(color.dccAddress - import->base), keyCount, 0x20202020u);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
        recorder.NotePendingWrite(color.dccAddress, keyCount);
        NoteKeysFillOnGpu(color.dccAddress, keyCount, DccKeys::ClearRegister);
    }
    Require(KnownPendingDccKeys(color.dccAddress, surfaceBytes) == DccKeys::ClearRegister, "the keys of a pending register clear store were not known");
    const std::array<std::uint32_t, 4> clearPattern{0x80402010u, 0x80402010u, 0x80402010u, 0x80402010u};
    const char* refusal = nullptr;
    Require(!keyed->FillClear(clearPattern, StorageTexture::WholeImage, refusal) && refusal != nullptr && std::strcmp(refusal, "keys pending") == 0, "a clear over pending keys was not refused without the proof");
    refusal = nullptr;
    Require(keyed->FillClear(clearPattern, StorageTexture::WholeImage, refusal, true), "a clear over known pending keys was refused");
    Require(recorder.PendingWriteOverlaps(color.dccAddress, keyCount) && keys[0] == 0xff, "the clear over known pending keys waited for the key store");
    MarkDccUncompressed(context, color.dccAddress, surfaceBytes);
    Require(memoryHolds({0x10, 0x20, 0x40, 0x80}), "the clear recorded behind a pending key store did not store CB_COLOR_CLEAR_WORD");
    Require(std::all_of(keys, keys + keyCount, [](std::uint8_t key) { return key == 0xff; }), "the keys after a clear behind a pending key store are not uncompressed");
}

void tiledResolveTests(const Device& device, Recorder& recorder) {
    namespace GuestMemory = AgcDriver::GuestMemory;
    auto context = device.GetContext();
    if ((context.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0) {
        std::cout << "4x color targets or graphics unavailable: CB resolves into tiled targets not tested\n";
        return;
    }
    context.hostImportAlignment = 0;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    constexpr std::uint32_t side = 128;
    constexpr std::size_t targetBytes = side * side * 4u;
    constexpr std::size_t sourceBytes = targetBytes * 4u;
    constexpr std::size_t bytes = targetBytes + sourceBytes + 65536u;
    void* block = std::aligned_alloc(65536, bytes);
    Require(block != nullptr, "cannot allocate the tiled resolve block");
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Release {
        const Context& context;
        void* block;
        ~Release() {
            ClearMultisampleTargets(context.device);
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            std::free(block);
        }
    } release{context, block};
    ColorTarget source{};
    source.address = address + targetBytes;
    source.surfaceAddress = source.address;
    source.extent = {side, side};
    source.surfaceExtent = source.extent;
    source.format = VK_FORMAT_R8G8B8A8_UNORM;
    source.bytes = targetBytes;
    source.componentMapping = 0xe4u;
    source.tileMode = ColorTileMode::RenderTarget;
    source.samples = 4;
    source.cmaskAddress = address + targetBytes + sourceBytes;
    source.cmaskBytes = 4096;
    source.cmaskFastClear = true;
    const std::array<std::pair<ColorTileMode, std::uint32_t>, 3> modes{{{ColorTileMode::Standard4KB, 0x80402010u}, {ColorTileMode::Standard64KB, 0x10204080u}, {ColorTileMode::RenderTarget, 0x40801020u}}};
    for (const auto& [mode, clear] : modes) {
        const auto name = std::to_string(static_cast<std::uint32_t>(mode));
        std::memset(block, 0x55, targetBytes);
        std::memset(static_cast<std::byte*>(block) + targetBytes + sourceBytes, 0, 4096);
        ColorTarget destination{};
        destination.address = address;
        destination.surfaceAddress = address;
        destination.extent = {side, side};
        destination.surfaceExtent = destination.extent;
        destination.format = VK_FORMAT_R8G8B8A8_UNORM;
        destination.bytes = targetBytes;
        destination.componentMapping = 0xe4u;
        destination.tileMode = mode;
        source.clearWords = {clear, 0};
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination}, source});
        std::vector<std::uint32_t> resolved(targetBytes / 4);
        GuestMemory::Read(address, std::as_writable_bytes(std::span(resolved)));
        const auto texels = std::ranges::count(resolved, clear);
        Require(texels == static_cast<std::ptrdiff_t>(resolved.size()), "a CB resolve into a tile mode " + name + " target left " + std::to_string(resolved.size() - static_cast<std::size_t>(texels)) + " texels without the resolved color");
        recorder.Sync();
        ClearMultisampleTargets(context.device);
        ClearCachedTextures(context.device);
    }
    {
        ColorTarget destination{};
        destination.address = address;
        destination.extent = {side, side};
        destination.format = VK_FORMAT_R8G8B8A8_UNORM;
        destination.bytes = targetBytes;
        destination.tileMode = ColorTileMode::Linear;
        std::string message;
        try {
            RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination}, source});
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find("without a resident image is not implemented") != std::string::npos, "a CB resolve into a linear target did not throw: " + message);
    }
    recorder.Sync();
}

void sliceDirtyTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    constexpr std::uint32_t side = 128;
    constexpr std::size_t sliceBytes = 65536;
    constexpr std::size_t bytes = 4 * sliceBytes;
    constexpr std::uint8_t untouched = 0x11;
    for (const auto tileMode : {TextureTileMode::kLinear, TextureTileMode::kR64KBX}) {
        const std::string what = tileMode == TextureTileMode::kLinear ? "a linear two-slice image" : "a tiled two-slice image";
        void* block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            throw std::runtime_error("required on Metal (host imports and write watching exist): no write watching: slice dirty marking not tested");
        }
        auto* memory = static_cast<std::uint8_t*>(block);
        std::memset(memory, untouched, bytes);
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
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                HostImportFor(context, address, bytes);
                ReleaseWatched(block, bytes);
            }
        } unregister{base, block, address, bytes};
        AgcDriver::GuestMemory::CollectWritesUncached(address, bytes);
        TextureDetiler detiler(base);
        auto context = base;
        context.hostImportAlignment = 0;
        context.detiler = &detiler;
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = side;
        resource.height = side;
        resource.depthOrLastArray = 1;
        resource.mipCount = 1;
        resource.tileMode = tileMode;
        resource.dimension = TextureDimension::k2DArray;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        Require(geometry.imageLayers == 2 && geometry.layerBytes == sliceBytes && geometry.guestBytes == 2 * sliceBytes, what + " does not hold two 64 KiB slices");
        const auto stored = [&](std::uint32_t slice, std::array<std::uint8_t, 4> color) {
            std::vector<std::byte> read(sliceBytes);
            AgcDriver::GuestMemory::Read(address + geometry.GuestLayerOffset(slice), read);
            for (std::size_t at = 0; at < read.size(); ++at) {
                if (std::to_integer<std::uint8_t>(read[at]) != color[at % 4]) return false;
            }
            return true;
        };
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
            const VkClearColorValue green{{0.0f, 1.0f, 0.0f, 1.0f}};
            const VkImageSubresourceRange first{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const VkImageSubresourceRange second{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1, 1};
            Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &first);
            Metal::Vk::CmdClearColorImage(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &green, 1, &second);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkSliceDirty(0);
            for (std::uint32_t slice = 0; slice < 2; ++slice) {
                const auto display = address + geometry.GuestLayerOffset(slice);
                const auto found = StorageTexture::FindPendingSlice(display, sliceBytes);
                Require(found == image, what + ": a display buffer that is slice " + std::to_string(slice) + " does not find the pending image");
                Require(AgcDriver::ResidentPresentSlice(found->Descriptor().baseAddress, found->GuestBytes(), found->ImageLayers(), display, sliceBytes) == slice, what + ": a display buffer that is slice " + std::to_string(slice) + " is not presented from that layer");
            }
            Require(StorageTexture::FindPending(address + geometry.GuestLayerOffset(1), sliceBytes) == nullptr, what + ": the lookup by base address finds a display buffer in slice 1");
            Require(StorageTexture::FindPendingSlice(address + 2 * sliceBytes, sliceBytes) == nullptr && StorageTexture::FindPendingSlice(address + geometry.GuestLayerOffset(1), sliceBytes + 1) == nullptr, what + ": a range past the image's last slice finds the pending image");
            image->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            Require(stored(0, {255, 0, 0, 255}), what + ": the slice a draw wrote was not stored");
            Require(stored(1, {untouched, untouched, untouched, untouched}), what + ": a slice no draw wrote was stored over guest memory");
            static_cast<void>(recorder.Commands());
            recorder.Keep(image);
            image->MarkSliceDirty(1);
            image->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            Require(stored(0, {255, 0, 0, 255}) && stored(1, {0, 255, 0, 255}), what + ": the second slice was not stored once a draw wrote it");
            auto other = StorageTexture::Create(context, detiler, resource, 0);
            static_cast<void>(recorder.Commands());
            recorder.Keep(image);
            recorder.Keep(other);
            image->MarkSliceDirty(1);
            other->MarkSliceDirty(1);
            std::string ambiguous;
            try {
                static_cast<void>(StorageTexture::FindPendingSlice(address + geometry.GuestLayerOffset(1), sliceBytes));
            } catch (const std::runtime_error& error) {
                ambiguous = error.what();
            }
            Require(ambiguous.find("is a slice of the pending array images") != std::string::npos, what + ": a slice of two pending array images did not throw: " + ambiguous);
            image->WriteBack();
            other->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        recorder.Sync();
    }
    std::cout << "slice dirty marking: ok\n";
}

// tests/ResidentPresent.cpp

using namespace AgcDriver::Graphics;
using AgcDriver::DisplayBuffer;
using AgcDriver::ResidentPresent;
using AgcDriver::ResidentPresentSlice;

constexpr std::uint64_t Bgra8 = 0x8000000000000000ull;
constexpr std::uint64_t Rgba8 = 0x8000000022000000ull;
constexpr std::uint64_t TenBit = 0x0100000000000000ull;
constexpr std::uint64_t Rgba16Float = 0xc001000622000000ull;
constexpr std::uint64_t Rgb10Pq = 0x8100070422000000ull;
constexpr std::uint64_t Bgr10Pq = 0x8100070400000000ull;

void decisionTests() {
    using AgcDriver::DisplayTexelFormat;
    using AgcDriver::ResidentPresentPath;
    Require(DisplayTexelFormat(Bgra8) == VK_FORMAT_B8G8R8A8_UNORM && DisplayTexelFormat(Rgba8) == VK_FORMAT_R8G8B8A8_UNORM, "8-bit display texel formats are wrong");
    Require(DisplayTexelFormat(Bgra8 | TenBit) == VK_FORMAT_A2R10G10B10_UNORM_PACK32 && DisplayTexelFormat(Rgba8 | TenBit) == VK_FORMAT_A2B10G10R10_UNORM_PACK32, "10-bit display texel formats are wrong");
    Require(ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, Bgra8, true) == ResidentPresent::Blit && ResidentPresentPath(VK_FORMAT_R8G8B8A8_UNORM, Rgba8, true) == ResidentPresent::Blit, "an 8-bit image in its display's order is not blitted");
    Require(ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, Bgra8, false) == ResidentPresent::Convert, "an 8-bit image that cannot be a blit source is not converted");
    Require(ResidentPresentPath(VK_FORMAT_R8G8B8A8_UNORM, Bgra8, true) == ResidentPresent::Convert && ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, Rgba8, true) == ResidentPresent::Convert, "an 8-bit image in the other order is not converted");
    Require(ResidentPresentPath(VK_FORMAT_R8G8B8A8_SRGB, Rgba8, true) == ResidentPresent::Convert, "an sRGB image is blitted (the blit would decode it)");
    Require(ResidentPresentPath(VK_FORMAT_R8G8B8A8_UINT, Rgba8, true) == ResidentPresent::Convert && ResidentPresentPath(VK_FORMAT_B8G8R8A8_UINT, Bgra8, true) == ResidentPresent::Convert, "an 8-bit integer image is not converted");
    Require(ResidentPresentPath(VK_FORMAT_R8G8B8A8_UINT, Rgba8 | TenBit, true) == ResidentPresent::None, "an 8-bit integer image presents a 10-bit display");
    for (const auto display : {Bgra8 | TenBit, Rgba8 | TenBit}) {
        for (const auto storage : {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32}) {
            Require(ResidentPresentPath(storage, display, true) == ResidentPresent::Convert, "a 10-bit image is blitted (the blit rounds where the guest path truncates)");
        }
        Require(ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, display, true) == ResidentPresent::None, "an 8-bit image presents a 10-bit display");
    }
    Require(DisplayTexelFormat(Bgr10Pq) == VK_FORMAT_A2R10G10B10_UNORM_PACK32 && DisplayTexelFormat(Rgb10Pq) == VK_FORMAT_A2B10G10R10_UNORM_PACK32, "PQ display texel formats are wrong");
    for (const auto display : {Bgr10Pq, Rgb10Pq}) {
        for (const auto storage : {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32}) {
            Require(ResidentPresentPath(storage, display, true) == ResidentPresent::Convert, "a PQ image is blitted (it must be converted to SDR)");
        }
        Require(ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, display, true) == ResidentPresent::None, "an 8-bit image presents a PQ display");
    }
    Require(ResidentPresentPath(VK_FORMAT_A2B10G10R10_UNORM_PACK32, Bgra8, true) == ResidentPresent::None && ResidentPresentPath(VK_FORMAT_R16G16_SFLOAT, Bgra8, true) == ResidentPresent::None, "an image of another type presents an 8-bit display");
    const auto guest = FindGuestColorTargetFormat(VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4);
    Require(guest.has_value() && guest == FindGuestTextureFormat(VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4), "the blue-low 10-bit color target has no storage format");
}

void sliceTests() {
    constexpr std::uint64_t image = 0x208000000ull;
    constexpr std::uint64_t slice = 0x1fe0000ull;
    Require(ResidentPresentSlice(image, slice, 1, image, slice) == 0u, "a display buffer that is the whole image is not its slice 0");
    Require(ResidentPresentSlice(image, 2 * slice, 2, image, slice) == 0u, "a display buffer at the start of a two-slice image is not its slice 0");
    Require(ResidentPresentSlice(image, 2 * slice, 2, image + slice, slice) == 1u, "a display buffer one slice into a two-slice image is not its slice 1");
    Require(ResidentPresentSlice(image, 6 * slice, 6, image + 5 * slice, slice) == 5u, "a display buffer at the last of six slices is not slice 5");
    Require(!ResidentPresentSlice(image, 2 * slice, 2, image + 2 * slice, slice), "a display buffer past the image's last slice has a slice");
    Require(!ResidentPresentSlice(image, 2 * slice, 2, image + 0x10000, slice), "a display buffer between two slices has a slice");
    Require(!ResidentPresentSlice(image, 2 * slice, 2, image - slice, slice), "a display buffer before the image has a slice");
    Require(!ResidentPresentSlice(image, 2 * slice, 2, image, 2 * slice), "a display buffer the size of two slices has a slice");
    Require(!ResidentPresentSlice(image, 2 * slice, 2, image, slice / 2), "a display buffer of half a slice has a slice");
    Require(!ResidentPresentSlice(image, 2 * slice + 1, 2, image, slice), "an image whose bytes do not divide into its slices has a slice");
    Require(!ResidentPresentSlice(image, 0, 0, image, slice) && !ResidentPresentSlice(image, slice, 1, image, 0), "an empty image or display buffer has a slice");
}

void conversionTest(const Context& context, std::uint64_t pixelFormat, VkFormat format, std::uint32_t layer = 0, float whiteNits = AgcDriver::SdrToneMapLuminanceNits) {
    constexpr std::uint32_t width = 200;
    constexpr std::uint32_t height = 130;
    const auto texelBytes = AgcDriver::DisplayTexelBytes(pixelFormat);
    std::vector<std::byte> words(static_cast<std::size_t>(width) * height * texelBytes);
    std::uint32_t seed = 0x9e3779b9u ^ static_cast<std::uint32_t>(format);
    for (std::size_t i = 0; i < words.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        words[i] = static_cast<std::byte>(seed >> 24u);
    }
    const ColorTargetLayout layout(width, height, ColorTileMode::RenderTarget, texelBytes);
    std::vector<std::byte> storage(layout.Bytes() + 65536);
    auto* aligned = reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(storage.data()) + 65535u) & ~std::uintptr_t{65535});
    std::span<std::byte> guest(aligned, layout.Bytes());
    const DisplayBuffer display{reinterpret_cast<std::uint64_t>(aligned), pixelFormat, width, height};
    Require(AgcDriver::DisplayBufferSize(display) == guest.size(), "test display buffer size differs from the color layout");
    if (texelBytes == 4u) {
        layout.Tile(words, guest);
    } else {
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) std::memcpy(guest.data() + AgcDriver::DisplayBufferOffset(display, x, y), words.data() + (static_cast<std::size_t>(y) * width + x) * texelBytes, texelBytes);
        }
    }
    auto expected = AgcDriver::DecodeDisplayBuffer(display, guest);
    if (AgcDriver::DisplayPq(pixelFormat) && whiteNits != AgcDriver::SdrToneMapLuminanceNits) {
        for (std::size_t i = 0; i < expected.size(); i += 4) {
            std::uint32_t word = 0;
            std::memcpy(&word, words.data() + i, 4);
            const auto pixel = AgcDriver::PqToSdrPixel(word, AgcDriver::DisplayRedLow(pixelFormat), whiteNits);
            for (std::size_t channel = 0; channel < pixel.size(); ++channel) expected[i + channel] = static_cast<std::byte>(pixel[channel]);
        }
    }

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = layer + 1u;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    Check(Metal::Vk::CreateImage(context.device, &imageInfo, nullptr, &image), "vkCreateImage resident present test");
    struct Release {
        const Context& context;
        VkImage& image;
        VkDeviceMemory& memory;
        ~Release() {
            if (image) Metal::Vk::DestroyImage(context.device, image, nullptr);
            if (memory) Metal::Vk::FreeMemory(context.device, memory, nullptr);
        }
    } release{context, image, memory};
    VkMemoryRequirements requirements{};
    Metal::Vk::GetImageMemoryRequirements(context.device, image, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = 0; // Metal: VkDeviceMemory is a token (textures own their memory), no memory types to pick from
    Check(Metal::Vk::AllocateMemory(context.device, &allocation, nullptr, &memory), "vkAllocateMemory resident present test");
    Check(Metal::Vk::BindImageMemory(context.device, image, memory, 0), "vkBindImageMemory resident present test");

    Buffer upload(context, words.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(upload.Bytes().data(), words.data(), words.size());
    Buffer fromImage(context, expected.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer fromGuest(context, expected.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    GpuColorTransfer transfer(context);
    const bool redLow = AgcDriver::DisplayRedLow(pixelFormat);
    const bool pq = AgcDriver::DisplayPq(pixelFormat);
    const bool tenBit = AgcDriver::DisplayTenBit(pixelFormat) && !pq;
    const bool halfFloat = AgcDriver::DisplayHalfFloat(pixelFormat);
    const auto pqWhiteNits = pq ? std::optional<float>(whiteNits) : std::nullopt;
    const auto barrier = Metal::Vk::CmdPipelineBarrier;
    const auto readback = [&](VkCommandBuffer commands, Buffer& destination) {
        const VkBufferCopy copy{0, 0, expected.size()};
        Metal::Vk::CmdCopyBuffer(commands, transfer.LinearBuffer(), destination.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
    };
    transfer.Upload(display.address, width, height, ColorTileMode::RenderTarget, texelBytes);
    {
        CommandBatch batch(context);
        const auto commands = batch.Handle();
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = image;
        toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layer + 1u};
        barrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransfer);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
        region.imageExtent = {width, height, 1};
        Metal::Vk::CmdCopyBufferToImage(commands, upload.Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        if (layer != 0) {
            const VkClearColorValue other{};
            const VkImageSubresourceRange lower{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layer};
            Metal::Vk::CmdClearColorImage(commands, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &other, 1, &lower);
        }
        auto toGeneral = toTransfer;
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        transfer.Detile(commands, redLow, tenBit, halfFloat, pqWhiteNits);
        readback(commands, fromGuest);
        if (!pq) {
            transfer.DetileImage(commands, image, VK_IMAGE_LAYOUT_GENERAL, width, height, ColorTileMode::RenderTarget, redLow, tenBit, halfFloat, layer);
            readback(commands, fromImage);
        }
        batch.SubmitAndWait();
    }
    const auto name = std::string(halfFloat ? "16-bit float " : pq ? "PQ " : tenBit ? "10-bit " : "8-bit ") + (redLow ? "red-low" : "blue-low") + " display from layer " + std::to_string(layer) + " of VkFormat " + std::to_string(static_cast<int>(format));
    const auto matches = [&](std::span<const std::byte> actual) {
        return std::equal(expected.begin(), expected.end(), actual.begin(), [&](std::byte want, std::byte got) { return std::abs(std::to_integer<int>(want) - std::to_integer<int>(got)) <= (pq ? 1 : 0); });
    };
    Require(matches(fromGuest.Bytes()), "the GPU detile differs from the CPU decode: " + name);
    Require(pq || matches(fromImage.Bytes()), "the resident image presents other pixels than guest memory: " + name);
}

void RunResidentPresentTests(const Context& context) {
    decisionTests();
    sliceTests();
    conversionTest(context, Bgra8 | TenBit, VK_FORMAT_A2B10G10R10_UNORM_PACK32);
    conversionTest(context, Rgba8 | TenBit, VK_FORMAT_A2B10G10R10_UNORM_PACK32);
    conversionTest(context, Bgra8 | TenBit, VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    conversionTest(context, Bgra8, VK_FORMAT_R8G8B8A8_UNORM);
    conversionTest(context, Rgba8, VK_FORMAT_B8G8R8A8_UNORM);
    conversionTest(context, Rgba16Float, VK_FORMAT_R16G16B16A16_SFLOAT);
    conversionTest(context, Rgba8, VK_FORMAT_B8G8R8A8_UNORM, 1);
    conversionTest(context, Bgra8 | TenBit, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 2);
    conversionTest(context, Rgb10Pq, VK_FORMAT_A2B10G10R10_UNORM_PACK32);
    conversionTest(context, Bgr10Pq, VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    conversionTest(context, Rgb10Pq, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 0, 1000.0f);
    conversionTest(context, Bgr10Pq, VK_FORMAT_A2R10G10B10_UNORM_PACK32, 0, 150.0f);
    std::cout << "resident present format and conversion tests passed\n";
}

// pipelineCacheFileTests on the Metal pipeline cache (see the header comment for the mapping).
int pipelineCacheFileTests() {
    namespace Checks = AgcDriver::Graphics::MetalPipelineCacheChecks;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
        std::printf("skipped, the machine has no Metal 4 device\n");
        return 77;
    }
    const auto directory = std::filesystem::temp_directory_path() / ("aps5-pipeline-cache-file-test-" + std::to_string(std::random_device{}()));
    struct RemoveDirectory {
        std::filesystem::path directory;
        ~RemoveDirectory() {
            std::error_code error;
            std::filesystem::remove_all(directory, error);
        }
    } removeDirectory{directory};
    const auto build = [&](Metal::PipelineCache& cache, std::size_t index) {
        Require(cache.Compute(Checks::ComputeFunction(index == 0 ? 3u : 5u)) != nil, "no compute pipeline");
    };
    const auto holds = [&](Metal::PipelineCache& cache, std::size_t index) {
        const auto before = cache.Statistics().pipelinesCompiled;
        build(cache, index);
        return cache.Statistics().pipelinesCompiled == before;
    };
    const auto cacheFiles = [&] {
        std::vector<std::filesystem::path> found;
        std::error_code error;
        for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) {
            if (it->path().extension() == ".mtl4archive") found.push_back(it->path());
        }
        std::sort(found.begin(), found.end());
        Require(!found.empty(), "the pipeline cache directory holds no archive");
        return found;
    };
    const auto contents = [](const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    };
    const auto snapshot = [&] {
        std::map<std::filesystem::path, std::vector<char>> files;
        for (const auto& path : cacheFiles()) files.emplace(path, contents(path));
        return files;
    };
    {
        Metal::PipelineCache probe(device, directory);
        const auto statistics = probe.Statistics();
        if (Checks::ShaderValidation()) {
            Require(statistics.archivePath.empty() && !statistics.archiveDisabled.empty(), "the archive must be disabled under MTL_SHADER_VALIDATION");
            Require(!holds(probe, 0), "a cache without an archive did not compile its first pipeline");
            std::cout << "Pipeline cache file merge tests passed (shader validation: archive disabled)\n";
            return 0;
        }
        Require(!statistics.archivePath.empty() && statistics.archiveDisabled.empty(), "the pipeline cache has no archive in its directory: " + statistics.archiveDisabled);
    }
    for (const bool firstSavesFirst : {true, false}) {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
        auto first = std::make_unique<Metal::PipelineCache>(device, directory);
        auto second = std::make_unique<Metal::PipelineCache>(device, directory);
        build(*first, 0);
        build(*second, 1);
        if (firstSavesFirst) {
            first.reset();
            second.reset();
        } else {
            second.reset();
            first.reset();
        }
        Metal::PipelineCache third(device, directory);
        Require(holds(third, 0), firstSavesFirst ? "the cache file lost the pipeline of the object that saved first" : "the cache file lost the pipeline of the object that saved last");
        Require(holds(third, 1), firstSavesFirst ? "the cache file lost the pipeline of the object that saved last" : "the cache file lost the pipeline of the object that saved first");
    }
    const auto stored = snapshot();
    {
        Metal::PipelineCache idle(device, directory);
    }
    Require(snapshot() == stored, "an object that added no pipelines rewrote the cache file");
    for (const auto& [path, bytes] : stored) {
        std::ofstream corrupt(path, std::ios::binary | std::ios::trunc);
        corrupt << "not a pipeline cache";
    }
    {
        Metal::PipelineCache replacing(device, directory);
        build(replacing, 1);
    }
    {
        Metal::PipelineCache reloaded(device, directory);
        Require(holds(reloaded, 1), "an invalid cache file was not replaced by the saving object's pipelines");
    }
    // Upstream e37e5f0aba/d65cc07a1b: a cache that loaded the file, kept it unchanged and added a pipeline saves
    // both. Upstream's unseeded caches (PipelineCache(context, properties, false), SaveNow) have no Metal
    // counterpart: every Metal::PipelineCache loads the archive generations (D51) and saves its own generation
    // at teardown, so those two cases are not run (upstream runs them after its idle check, here they are absent).
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
    {
        Metal::PipelineCache writer(device, directory);
        build(writer, 0);
    }
    {
        Metal::PipelineCache extender(device, directory);
        Require(holds(extender, 0), "the cache file lost a pipeline");
        build(extender, 1);
    }
    {
        Metal::PipelineCache reloaded(device, directory);
        Require(holds(reloaded, 0) && holds(reloaded, 1), "a cache that saved over an unchanged file lost pipelines");
    }
    std::cout << "Pipeline cache file merge tests passed\n";
    return 0;
}

// Upstream main runs the sparse run modes on a fresh device, before it takes the GPU mutex, prepares the import
// watch or activates a recorder: sparseRunImportTests does all three itself.
int RunSparseRun(bool copied) {
    @autoreleasepool {
        try {
            Device device;
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, sparse run imports require host imports\n";
                return 77;
            }
            return sparseRunImportTests(device, copied) ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}

const bool sparseRunImport = AgcDriver::MetalTwins::RegisterRecorderMode("--sparse-run-import-only", [] { return RunSparseRun(false); });

const bool sparseRunCopied = AgcDriver::MetalTwins::RegisterRecorderMode("--sparse-run-copied-only", [] { return RunSparseRun(true); });

const bool drawSnapshotEpoch = AgcDriver::MetalTwins::RegisterRecorderMode("--draw-snapshot-epoch-only", [] {
    return RunMode([](const Device& device, Recorder&) { drawSnapshotEpochTests(device); });
});

const bool tiledResolve = AgcDriver::MetalTwins::RegisterRecorderMode("--tiled-resolve-only", [] {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil || ![device supportsTextureSampleCount:4]) {
            std::cout << "skipped, the device lacks 4x color targets or a graphics queue\n";
            return 77;
        }
    }
    return RunMode([](const Device& device, Recorder& recorder) {
        tiledResolveTests(device, recorder);
        std::cout << "CB resolves into tiled targets tests passed\n";
    });
});

const bool sliceDirty = AgcDriver::MetalTwins::RegisterRecorderMode("--slice-dirty-only", [] {
    return RunMode([](const Device& device, Recorder& recorder) { sliceDirtyTests(device, recorder); });
});

const bool metadataPass = AgcDriver::MetalTwins::RegisterRecorderMode("--metadata-pass-only", [] {
    return RunMode([](const Device& device, Recorder& recorder) {
        metadataPassTests(device, recorder);
        std::cout << "Color metadata pass tests passed\n";
    });
});

const bool pipelineCacheFile = AgcDriver::MetalTwins::RegisterRecorderMode("--pipeline-cache-file-only", [] {
    @autoreleasepool {
        try {
            return pipelineCacheFileTests();
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
});

const bool residentPresent = AgcDriver::MetalTwins::RegisterRecorderMode("--resident-present-only", [] {
    return RunMode([](const Device& device, Recorder&) { RunResidentPresentTests(device.GetContext()); });
});

}
