// Metal twin of tests/Recorder.cpp's --draw-snapshot-cap-only mode (agc_draw_snapshot_cap). The test function
// (drawSnapshotCapTests, Recorder.cpp:1778-1921) is copied unchanged: it drives the Metal ShaderResources twin
// (specifics/macos/src/ShaderResources.mm PrepareDrawBindings, which applies APS5_DRAW_SNAPSHOT_MAX_MIB) and the
// Metal recorder's snapshot cache (Recorder.mm ReusableDrawSnapshot) over a host import of write-watched memory
// (GuestBufferMemory.mm: no-copy Metal buffers over 16 KiB pages). An input over the cap is bound in place and
// sees CPU writes, one under it is snapshotted and does not, and the cap is cumulative over the layout's bindings.
// Upstream's exits for a device without host imports or write watching are requirements here: the Metal device
// has both, so a pass means every check ran. Its imports are compared, not watched (DECISIONS D19), so upstream's
// watched-only reuse check becomes the check that no snapshot of the unwatched block is kept.
// Upstream sync 3: the case that built resources keep the cap they were built with (raising the cap to 64 MiB
// leaves the large input in place on a second PrepareDrawBindings) is ported as written; ShaderResources.mm keeps the
// cap in maxDrawSnapshotBytes like upstream. Expected values are upstream's.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void drawSnapshotCapTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    const char* previous = std::getenv("APS5_DRAW_SNAPSHOT_MAX_MIB");
    const std::string saved = previous != nullptr ? previous : "";
#ifdef _WIN32
    Require(_putenv_s("APS5_DRAW_SNAPSHOT_MAX_MIB", "1") == 0, "cannot set snapshot cap");
#else
    Require(setenv("APS5_DRAW_SNAPSHOT_MAX_MIB", "1", 1) == 0, "cannot set snapshot cap");
#endif
    struct RestoreCap {
        const char* previous;
        std::string saved;
        ~RestoreCap() {
#ifdef _WIN32
            _putenv_s("APS5_DRAW_SNAPSHOT_MAX_MIB", saved.c_str());
#else
            if (previous != nullptr) {
                setenv("APS5_DRAW_SNAPSHOT_MAX_MIB", saved.c_str(), 1);
            } else {
                unsetenv("APS5_DRAW_SNAPSHOT_MAX_MIB");
            }
#endif
        }
    } restore{previous, saved};
    constexpr std::size_t bytes = 3 * 1024 * 1024;
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    // The Metal device always has host imports and write watching: upstream's "not tested" exits fail here.
    Require(block != nullptr, "host imports or write watching unavailable: draw snapshot cap not tested");
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    std::memset(block, 0x11, bytes);
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
    Require(HostImportFor(context, address, bytes) != nullptr, "host import of the watched block refused: draw snapshot cap not tested");
    const auto smallAddress = address + 4096;
    constexpr std::size_t smallBytes = 1024 * 1024 - 4096;
    const auto largeAddress = address + 1024 * 1024 + 8192;
    constexpr std::size_t largeBytes = 1024 * 1024 + 4096;
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding smallBinding;
    smallBinding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    smallBinding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    smallBinding.descriptorSet = 0;
    smallBinding.binding = 0;
    smallBinding.count = 1;
    smallBinding.guestDescriptor = {static_cast<std::uint32_t>(smallAddress), static_cast<std::uint32_t>(smallAddress >> 32u) & 0xffffu, static_cast<std::uint32_t>(smallBytes), 0x31000000u};
    smallBinding.bufferWritten = {false};
    program.bindings.push_back(smallBinding);
    ShaderRecompiler::DescriptorBinding largeBinding;
    largeBinding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    largeBinding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    largeBinding.descriptorSet = 0;
    largeBinding.binding = 1;
    largeBinding.count = 1;
    largeBinding.guestDescriptor = {static_cast<std::uint32_t>(largeAddress), static_cast<std::uint32_t>(largeAddress >> 32u) & 0xffffu, static_cast<std::uint32_t>(largeBytes), 0x31000000u};
    largeBinding.bufferWritten = {false};
    program.bindings.push_back(largeBinding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    {
        auto snapshotContext = context;
        DescriptorCache cache(snapshotContext);
        snapshotContext.descriptorCache = &cache;
        Recorder snapshotRecorder(snapshotContext);
        snapshotRecorder.Activate();
        ShaderResources resources(snapshotContext, compute);
        const auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
        Require(bindings != nullptr, "read-only draw bindings were not prepared");
        Require(bindings->snapshots.size() == 1, "an input exceeding the draw snapshot cap was snapshotted");
        Require(bindings->snapshots[0].address == smallAddress, "the snapshotted input does not match the smaller buffer below the cap");
        Require(snapshotRecorder.ReusableDrawSnapshot(largeAddress, largeBytes) == nullptr, "an input exceeding the draw snapshot cap was kept in the snapshot cache");
        if (Watched(address, bytes)) {
            Require(snapshotRecorder.ReusableDrawSnapshot(smallAddress, smallBytes) != nullptr, "an input below the draw snapshot cap was not kept in the snapshot cache");
        } else {
            // Metal host imports are compared, not watched (DECISIONS D19): the import unwatched the block, so no
            // write generation proves a kept snapshot current and none may be kept.
            Require(snapshotRecorder.ReusableDrawSnapshot(smallAddress, smallBytes) == nullptr, "a snapshot of unwatched memory was kept in the snapshot cache, where no write generation proves it current");
        }
        const auto contents = bindings->snapshots[0].buffer->Bytes();
        Require(contents.size() == smallBytes && std::all_of(contents.begin(), contents.end(), [](std::byte value) { return value == std::byte{0x11}; }), "a draw snapshot does not hold the guest bytes of its draw");
        std::memset(reinterpret_cast<void*>(smallAddress), 0x22, smallBytes);
        std::memset(reinterpret_cast<void*>(largeAddress), 0x33, largeBytes);
        Require(std::all_of(contents.begin(), contents.end(), [](std::byte value) { return value == std::byte{0x11}; }), "snapshotted input observed CPU overwrite");
        const auto* inPlace = reinterpret_cast<const std::byte*>(largeAddress);
        Require(inPlace[0] == std::byte{0x33} && std::all_of(inPlace, inPlace + largeBytes, [](std::byte value) { return value == std::byte{0x33}; }), "in-place bound input did not observe CPU write");
        std::memset(block, 0x11, bytes);
        snapshotRecorder.Sync();
#ifdef _WIN32
        Require(_putenv_s("APS5_DRAW_SNAPSHOT_MAX_MIB", "64") == 0, "cannot raise snapshot cap");
#else
        Require(setenv("APS5_DRAW_SNAPSHOT_MAX_MIB", "64", 1) == 0, "cannot raise snapshot cap");
#endif
        const auto again = resources.PrepareDrawBindings(snapshotRecorder);
        Require(again != nullptr && again->snapshots.size() == 1 && again->snapshots[0].address == smallAddress, "built resources read the draw snapshot cap again instead of keeping the one they were built with");
        snapshotRecorder.Sync();
#ifdef _WIN32
        Require(_putenv_s("APS5_DRAW_SNAPSHOT_MAX_MIB", "1") == 0, "cannot restore snapshot cap");
#else
        Require(setenv("APS5_DRAW_SNAPSHOT_MAX_MIB", "1", 1) == 0, "cannot restore snapshot cap");
#endif
    }
    const auto firstAddress = address + 4096;
    constexpr std::size_t firstBytes = 600 * 1024;
    const auto secondAddress = address + 600 * 1024 + 8192;
    constexpr std::size_t secondBytes = 600 * 1024;
    ShaderRecompiler::RecompileResult cumulativeProgram;
    ShaderRecompiler::DescriptorBinding firstBinding;
    firstBinding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    firstBinding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    firstBinding.descriptorSet = 0;
    firstBinding.binding = 0;
    firstBinding.count = 1;
    firstBinding.guestDescriptor = {static_cast<std::uint32_t>(firstAddress), static_cast<std::uint32_t>(firstAddress >> 32u) & 0xffffu, static_cast<std::uint32_t>(firstBytes), 0x31000000u};
    firstBinding.bufferWritten = {false};
    cumulativeProgram.bindings.push_back(firstBinding);
    ShaderRecompiler::DescriptorBinding secondBinding;
    secondBinding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    secondBinding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    secondBinding.descriptorSet = 0;
    secondBinding.binding = 1;
    secondBinding.count = 1;
    secondBinding.guestDescriptor = {static_cast<std::uint32_t>(secondAddress), static_cast<std::uint32_t>(secondAddress >> 32u) & 0xffffu, static_cast<std::uint32_t>(secondBytes), 0x31000000u};
    secondBinding.bufferWritten = {false};
    cumulativeProgram.bindings.push_back(secondBinding);
    const CompiledShader cumulativeCompute{ShaderRecompiler::ShaderStage::Compute, &cumulativeProgram, 0};
    {
        auto snapshotContext = context;
        DescriptorCache cache(snapshotContext);
        snapshotContext.descriptorCache = &cache;
        Recorder snapshotRecorder(snapshotContext);
        snapshotRecorder.Activate();
        ShaderResources resources(snapshotContext, cumulativeCompute);
        const auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
        Require(bindings != nullptr, "read-only draw bindings were not prepared for cumulative test");
        Require(bindings->snapshots.size() == 1, "cumulative cap permitted both inputs to snapshot");
        Require(bindings->snapshots[0].address == firstAddress, "the earlier binding in the layout was not prioritised for the snapshot budget");
        snapshotRecorder.Sync();
    }
    recorder.Activate();
}


const bool registered = AgcDriver::MetalTwins::RegisterRecorderMode("--draw-snapshot-cap-only", [] {
    return RunMode([](const Device& device, Recorder& recorder) {
        drawSnapshotCapTests(device, recorder);
        std::cout << "Draw snapshot cap tests passed\n";
    });
});

}
