// Metal tests of the recorder and its command stream (work package D2): what tests/Recorder.cpp
// and the label/wait paths of the driver need from Recorder.mm, on the GPU with tiny inputs. Store
// runs (labels) land in program order (joined, replaced, WAW inside a run, after earlier work);
// DCC key stores fill and seed; SyncThrough finishes only up to the batch that writes a range;
// completions run in batch order; the timeline serves unlocked waits; the pending-write snapshot
// follows the batches; the fill and mesh-argument kernels compute what their GLSL originals do;
// occlusion samples counted by a render pass reach a ZPASS_DONE dump and SamplesTotal; the flush
// hook waits for a batch's writes; hundreds of random batches keep program order.
// `--timing` (APS5_PROFILE_GPU=1): batch and range timestamps from MTL4 counter heaps.

#include "MetalTestDevice.h"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace AgcDriver::Graphics;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

using Lock = std::lock_guard<AgcDriver::GuestMemory::GpuMutexType>;

std::span<const std::byte> Words(const std::vector<std::uint32_t>& words) {
    return std::as_bytes(std::span(words));
}

std::uint32_t* WordsOf(Buffer& buffer) {
    return reinterpret_cast<std::uint32_t*>(buffer.Bytes().data());
}

std::uint64_t AddressOf(Buffer& buffer, std::size_t offset = 0) {
    return reinterpret_cast<std::uint64_t>(buffer.Bytes().data()) + offset;
}

// Labels: stores of one batch form a run recorded at Submit; a contiguous store joins the last
// queued one, one inside it replaces its bytes, and a store over an earlier (not last) queued one
// takes a WAW barrier. Memory must end as program order leaves it.
void StoreRuns(const Context& context) {
    Buffer target(context, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(target.Bytes().data(), 0, 4096);
    const auto before = Recorder::StoreCounts();
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        Recorder recorder(context);
        recorder.RecordStore(target.Handle(), 0, Words({1, 2, 3, 4}), AddressOf(target));
        recorder.RecordStore(target.Handle(), 16, Words({5, 6}), AddressOf(target, 16));
        recorder.RecordStore(target.Handle(), 4, Words({9}), AddressOf(target, 4));
        recorder.RecordStore(target.Handle(), 256, Words({0x100}), AddressOf(target, 256));
        recorder.RecordStore(target.Handle(), 8, Words({0x77, 0x78}), AddressOf(target, 8));
        Expect(recorder.HasQueuedStores() && recorder.QueuedStoreOverlaps(AddressOf(target, 8), 4), "the queued stores are not visible to QueuedStoreOverlaps");
        recorder.Sync();
        Expect(recorder.Submissions() == 1, "one batch of stores made " + std::to_string(recorder.Submissions()) + " submissions");
    }
    const auto* words = WordsOf(target);
    const std::array<std::uint32_t, 6> expected{1, 9, 0x77, 0x78, 5, 6};
    for (std::size_t i = 0; i < expected.size(); ++i) Expect(words[i] == expected[i], "store run word " + std::to_string(i) + " is " + Hex(words[i]) + ", expected " + Hex(expected[i]));
    Expect(words[64] == 0x100 && words[6] == 0 && words[65] == 0, "the separate store landed wrongly or a store spilled (" + Hex(words[64]) + ", " + Hex(words[6]) + ")");
    const auto after = Recorder::StoreCounts();
    Expect(after.stores - before.stores == 5 && after.joined - before.joined == 1 && after.replaced - before.replaced == 1 && after.wawBarriers - before.wawBarriers == 1, "store run counters: stores " + std::to_string(after.stores - before.stores) + ", joined " + std::to_string(after.joined - before.joined) + ", replaced " + std::to_string(after.replaced - before.replaced) + ", WAW " + std::to_string(after.wawBarriers - before.wawBarriers));
}

// A store run is recorded behind the work before it: a fill recorded first never lands over it.
void StoresAfterEarlierWork(const Context& context) {
    Buffer target(context, 65536, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(target.Bytes().data(), 0, 65536);
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        Recorder recorder(context);
        for (int round = 0; round < 4; ++round) {
            recorder.Stream().FillBuffer(Metal::BufferOf(target.Handle()), 0, 65536, 0x5a5a5a5au + static_cast<std::uint32_t>(round) * 0x01010101u);
            recorder.RecordStore(target.Handle(), 4096 * static_cast<std::size_t>(round) + 12, Words({0xc0ffee00u + static_cast<std::uint32_t>(round)}), AddressOf(target, 4096 * static_cast<std::size_t>(round) + 12));
            recorder.Submit();
        }
        recorder.Sync();
    }
    const auto* words = WordsOf(target);
    // The last fill (0x5d5d5d5d) covers everything recorded before it, the last store lands after it.
    Expect(words[3 * 1024 + 3] == 0xc0ffee03u, "the store after the last fill is " + Hex(words[3 * 1024 + 3]) + ", the fill landed over it");
    Expect(words[3] == 0x5d5d5d5du && words[1024 + 3] == 0x5d5d5d5du && words[16383] == 0x5d5d5d5du, "a later fill did not cover an earlier store (" + Hex(words[3]) + ", " + Hex(words[1027]) + ")");
}

// DCC key stores: the dword-aligned middle is filled with 0xff, the unaligned head and tail come
// from the seed's first bytes.
void KeyStores(const Context& context) {
    Buffer target(context, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(target.Bytes().data(), 0, 256);
    auto seed = std::make_shared<Buffer>(context, 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    for (std::size_t i = 0; i < 16; ++i) seed->Bytes()[i] = static_cast<std::byte>(0xa0 + i);
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        Recorder recorder(context);
        recorder.QueueKeyStore(target.Handle(), 6, 41, seed, AddressOf(target, 6), AddressOf(target, 41));
        recorder.QueueKeyStore(target.Handle(), 6, 41, seed, AddressOf(target, 6), AddressOf(target, 41));
        Expect(recorder.HasQueuedKeyStores() && recorder.QueuedKeyStoreOverlaps(AddressOf(target, 40), 1), "the queued key store is not visible");
        recorder.FlushKeyStores();
        Expect(!recorder.HasQueuedKeyStores(), "FlushKeyStores left a queued key store");
        recorder.Sync();
    }
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(target.Bytes().data());
    bool good = bytes[5] == 0 && bytes[6] == 0xa0 && bytes[7] == 0xa1 && bytes[41] == 0;
    for (std::size_t i = 8; i < 40; ++i) good = good && bytes[i] == 0xff;
    good = good && bytes[40] == 0xa0;
    Expect(good, "the key store's fill or seeded head and tail are wrong (bytes 6..8: " + std::to_string(bytes[6]) + " " + std::to_string(bytes[7]) + " " + std::to_string(bytes[8]) + ", 40: " + std::to_string(bytes[40]) + ")");
}

// SyncThrough waits only through the newest batch that writes the range; completions of the
// finished batches run in batch order.
void SyncThroughAndCompletions(const Context& context) {
    Buffer target(context, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::vector<int> order;
    Lock gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    for (int batch = 0; batch < 3; ++batch) {
        recorder.RecordStore(target.Handle(), 64 * static_cast<std::size_t>(batch), Words({static_cast<std::uint32_t>(batch + 1)}), AddressOf(target, 64 * static_cast<std::size_t>(batch)));
        recorder.NotePendingWrite(AddressOf(target, 64 * static_cast<std::size_t>(batch)), 4);
        recorder.OnComplete([&order, batch] { order.push_back(batch); });
        recorder.Submit();
    }
    Expect(recorder.InFlightBatches() == 3 && recorder.HasCompletions(), "three submitted batches with completions are not in flight");
    Expect(recorder.PendingWriteOverlaps(AddressOf(target, 64), 4), "the second batch's write is not pending");
    recorder.SyncThrough(AddressOf(target, 64), 4);
    Expect(recorder.InFlightBatches() == 1 && order == std::vector<int>{0, 1}, "SyncThrough of the second batch's range left " + std::to_string(recorder.InFlightBatches()) + " batches in flight and ran " + std::to_string(order.size()) + " completions");
    Expect(WordsOf(target)[16] == 2, "the second batch's store is not in memory after SyncThrough");
    recorder.Sync();
    Expect(order == std::vector<int>{0, 1, 2} && !recorder.HasCompletions(), "completions did not run once each in batch order");
    Expect(!recorder.PendingWriteOverlaps(AddressOf(target), 4096), "a finished batch's write is still pending");
}

// The timeline: a serial waited for without the GPU mutex on another thread, then finished.
void UnlockedWaits(const Context& context) {
    Buffer target(context, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(target.Bytes().data(), 0, 4096);
    Recorder* recorder = nullptr;
    std::unique_ptr<Recorder> owned;
    std::uint64_t serial = 0;
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        owned = std::make_unique<Recorder>(context);
        recorder = owned.get();
        Expect(recorder->HasTimeline(), "the Metal recorder has no timeline");
        recorder->RecordStore(target.Handle(), 0, Words({0xfeedu}), AddressOf(target));
        serial = recorder->SubmitAndEpoch();
        Expect(serial == 1, "the first epoch is " + std::to_string(serial));
    }
    std::thread waiter([&] { recorder->WaitSerial(serial); });
    waiter.join();
    Expect(WordsOf(target)[0] == 0xfeedu, "the store is not in memory after the unlocked wait");
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        Expect(recorder->InFlightBatches() == 1, "the waited batch left the in-flight list before FinishUpTo");
        recorder->FinishUpTo(serial);
        Expect(recorder->Idle(), "FinishUpTo did not finish the waited batch");
        // Reap: a batch the GPU finished retires without a wait.
        recorder->RecordStore(target.Handle(), 4, Words({0xbeefu}), AddressOf(target, 4));
        recorder->Submit();
        bool reaped = false;
        for (int attempt = 0; attempt < 2000 && !reaped; ++attempt) {
            reaped = recorder->Reap();
            if (!reaped) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Expect(reaped && WordsOf(target)[1] == 0xbeefu, "Reap never retired a finished batch");
        owned.reset();
    }
}

// The pending-write snapshot the flush hook reads: published while a batch writes the range, gone
// once the batch finished; labels enter the table with their batch and leave with it.
void SnapshotAndLabels(const Context& context) {
    Buffer target(context, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Lock gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    recorder.Activate();
    const auto address = AddressOf(target, 128);
    Expect(!Recorder::SnapshotWriteOverlaps(address, 8), "an idle recorder publishes a pending write");
    const std::vector<std::uint32_t> label{0x11223344u, 0x55667788u};
    recorder.RecordStore(target.Handle(), 128, Words(label), address);
    recorder.NoteLabel(address, Words(label), 7, 0);
    Expect(Recorder::SnapshotWriteOverlaps(address, 8) && recorder.OpenWriteOverlaps(address, 4), "a recorded label is not a pending write of the open batch");
    const auto hit = recorder.PendingLabel(address, 8, 6);
    Expect(hit.has_value() && hit->value == 0x5566778811223344ull && hit->stamp == 7, "the label table does not serve the recorded label");
    Expect(!recorder.PendingLabel(address, 8, 7).has_value(), "a label recorded before the wait's stamp was served without the late rule");
    recorder.Sync();
    Expect(!Recorder::SnapshotWriteOverlaps(address, 8) && recorder.PendingLabels() == 0, "a finished label is still pending or tabled");
    Expect(WordsOf(target)[32] == 0x11223344u && WordsOf(target)[33] == 0x55667788u, "the label did not land");
}

// The stream's fills: byte-uniform dwords by Metal fills, other dwords and 16-byte patterns by the
// fill kernel; dependent dispatches ordered by Barrier.
void Fills(const Context& context) {
    Buffer target(context, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(target.Bytes().data(), 0, 4096);
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        Recorder recorder(context);
        auto& stream = recorder.Stream();
        id<MTLBuffer> buffer = Metal::BufferOf(target.Handle());
        const std::array<std::uint32_t, 4> pattern{1, 2, 3, 4};
        stream.FillPattern(buffer, 4, 40, pattern);
        stream.FillBuffer(buffer, 64, 36, 0x12345678u);
        stream.FillBuffer(buffer, 128, 32, 0x7f7f7f7fu);
        // Two kernel fills of one range: the barrier orders the second after the first.
        stream.FillBuffer(buffer, 256, 1024, 0x0badf00du);
        stream.Barrier(MTLStageDispatch, MTLStageDispatch);
        stream.FillBuffer(buffer, 256, 1024, 0x600dcafeu);
        recorder.Sync();
    }
    const auto* words = WordsOf(target);
    const std::array<std::uint32_t, 12> head{0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 0};
    for (std::size_t i = 0; i < head.size(); ++i) Expect(words[i] == head[i], "pattern fill word " + std::to_string(i) + " is " + Hex(words[i]));
    for (std::size_t i = 16; i < 25; ++i) Expect(words[i] == 0x12345678u, "dword fill word " + std::to_string(i) + " is " + Hex(words[i]));
    Expect(words[25] == 0 && words[15] == 0, "the dword fill spilled");
    for (std::size_t i = 32; i < 40; ++i) Expect(words[i] == 0x7f7f7f7fu, "byte fill word " + std::to_string(i) + " is " + Hex(words[i]));
    bool ordered = true;
    for (std::size_t i = 64; i < 320; ++i) ordered = ordered && words[i] == 0x600dcafeu;
    Expect(ordered, "the second kernel fill did not land after the first (word 64 is " + Hex(words[64]) + ")");
}

// The MSL twin of MeshArguments.comp: the task group count of an indirect mesh draw record.
void MeshArguments(const Context& context) {
    struct Case {
        std::array<std::uint32_t, 3> record;
        std::array<std::uint32_t, 5> expected;
    };
    const std::array<std::uint32_t, 7> rules{1000, 3, 1, 64, 1024, 16, 4096};
    const std::array<Case, 4> cases{{
        {{300, 2, 10}, {5, 2, 1, 300, 10}},
        {{300, 0, 10}, {0, 0, 0, 300, 10}},
        {{50, 1, 990}, {1, 1, 1, 10, 990}},
        {{300, 17, 0}, {0, 0, 0, 300, 0}},
    }};
    for (const auto& item : cases) {
        Buffer record(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        Buffer arguments(context, 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        std::memcpy(record.Bytes().data(), item.record.data(), 12);
        std::memset(arguments.Bytes().data(), 0xee, 32);
        {
            Lock gpu(AgcDriver::GuestMemory::GpuMutex());
            Recorder recorder(context);
            Expect(recorder.RecordMeshArguments(recorder.Commands(), record.DeviceAddress(), arguments.DeviceAddress(), rules), "the mesh argument kernel is unavailable");
            recorder.Sync();
        }
        const auto* words = WordsOf(arguments);
        for (std::size_t i = 0; i < 5; ++i) Expect(words[i] == item.expected[i], "mesh arguments of record {" + std::to_string(item.record[0]) + ", " + std::to_string(item.record[1]) + ", " + std::to_string(item.record[2]) + "} word " + std::to_string(i) + " is " + std::to_string(words[i]) + ", expected " + std::to_string(item.expected[i]));
    }
}

constexpr const char* SampleShaders = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Position { float4 position [[position]]; };
vertex Position FullScreen(uint vertexId [[vertex_id]]) {
    const float2 corner = float2(float((vertexId << 1u) & 2u), float(vertexId & 2u));
    Position out;
    out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return out;
}
fragment float4 Solid() { return float4(1.0, 0.0, 0.0, 1.0); }
)MSL";

// Occlusion samples: a render pass counts its sampled draw into the recorder's visibility slot;
// a queued ZPASS_DONE dump folds it with the SampleDumps kernel at Submit, and SamplesTotal agrees.
void OcclusionSamples(const MetalTest::Device& device) {
    const auto& context = device.context;
    NSError* error = nil;
    id<MTLLibrary> library = [device.device newLibraryWithSource:@(SampleShaders) options:nil error:&error];
    Expect(library != nil, "the occlusion test shaders do not compile: " + Metal::Describe(error));
    if (library == nil) return;
    MTLRenderPipelineDescriptor* pipelineDescriptor = [MTLRenderPipelineDescriptor new];
    pipelineDescriptor.vertexFunction = [library newFunctionWithName:@"FullScreen"];
    pipelineDescriptor.fragmentFunction = [library newFunctionWithName:@"Solid"];
    pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline = [device.device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    Expect(pipeline != nil, "the occlusion test pipeline: " + Metal::Describe(error));
    if (pipeline == nil) return;
    MTLTextureDescriptor* textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:64 height:64 mipmapped:NO];
    textureDescriptor.usage = MTLTextureUsageRenderTarget;
    textureDescriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texture = [device.device newTextureWithDescriptor:textureDescriptor];
    MTLResidencySetDescriptor* residencyDescriptor = [MTLResidencySetDescriptor new];
    id<MTLResidencySet> residency = [device.device newResidencySetWithDescriptor:residencyDescriptor error:&error];
    [residency addAllocation:texture];
    [residency commit];
    [device.queue addResidencySet:residency];
    Buffer dump(context, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    std::memset(dump.Bytes().data(), 0, 256);
    std::uint64_t total = 0;
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        Recorder recorder(context);
        recorder.CountSamples();
        Expect(recorder.QueuesSampleDumps(), "the Metal recorder does not count samples in segment mode");
        const auto start = recorder.StartDrawPass(1, false, false, PassAccess{});
        recorder.PrepareSampleSlot();
        auto* visibility = recorder.SampleVisibilityBuffer();
        Expect(visibility != nullptr, "no visibility buffer is ready after PrepareSampleSlot");
        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.visibilityResultBuffer = (__bridge id<MTLBuffer>)visibility;
        pass.visibilityResultType = MTLVisibilityResultTypeAccumulate;
        auto& stream = Metal::Stream(start.commands);
        id<MTL4RenderCommandEncoder> render = stream.BeginRender(pass);
        [render setRenderPipelineState:pipeline];
        recorder.NoteSampledDraw(start.commands);
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        recorder.LeaveRenderPassOpen(1, Recorder::NoTiming, false, PassAccess{});
        Expect(recorder.DumpSamples(dump.DeviceAddress(), AddressOf(dump)), "the dump was refused");
        recorder.Sync();
        total = recorder.SamplesTotal();
    }
    [device.queue removeResidencySet:residency];
    const auto* words = WordsOf(dump);
    Expect(words[0] == 64u * 64u && words[1] == 0x80000000u, "the dump's first entry is " + Hex(words[1]) + ":" + Hex(words[0]) + ", expected 4096 samples with the valid bit");
    bool rest = true;
    for (std::uint32_t db = 1; db < 16; ++db) rest = rest && words[db * 4] == 0 && words[db * 4 + 1] == 0x80000000u;
    Expect(rest, "the dump's other entries are not empty valid entries");
    Expect(total == 64u * 64u, "SamplesTotal is " + std::to_string(total) + ", expected 4096");
}

// Occlusion samples over many passes, as MetalDraw records them (DECISIONS D57): each new pass takes
// its counter slot with PrepareSamplePass before it begins (the previous pass's segment ends there),
// and 130 passes outrun one visibility buffer's 128 slots. Every pass counts its draw.
void OcclusionPasses(const MetalTest::Device& device) {
    const auto& context = device.context;
    NSError* error = nil;
    id<MTLLibrary> library = [device.device newLibraryWithSource:@(SampleShaders) options:nil error:&error];
    Expect(library != nil, "the occlusion test shaders do not compile: " + Metal::Describe(error));
    if (library == nil) return;
    MTLRenderPipelineDescriptor* pipelineDescriptor = [MTLRenderPipelineDescriptor new];
    pipelineDescriptor.vertexFunction = [library newFunctionWithName:@"FullScreen"];
    pipelineDescriptor.fragmentFunction = [library newFunctionWithName:@"Solid"];
    pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline = [device.device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    Expect(pipeline != nil, "the occlusion test pipeline: " + Metal::Describe(error));
    if (pipeline == nil) return;
    MTLTextureDescriptor* textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
    textureDescriptor.usage = MTLTextureUsageRenderTarget;
    textureDescriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texture = [device.device newTextureWithDescriptor:textureDescriptor];
    id<MTLResidencySet> residency = [device.device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    [residency addAllocation:texture];
    [residency commit];
    [device.queue addResidencySet:residency];
    constexpr std::uint32_t Passes = 130;
    std::uint64_t total = 0;
    std::string failure;
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        Recorder recorder(context);
        recorder.CountSamples();
        // The counter is the device's (shared with the recorders of earlier checks): a delta.
        const auto before = recorder.SamplesTotal();
        try {
            for (std::uint32_t index = 0; index < Passes; ++index) {
                const auto start = recorder.StartDrawPass(100 + index, false, false, PassAccess{});
                recorder.PrepareSamplePass();
                auto* visibility = recorder.SampleVisibilityBuffer();
                if (visibility == nullptr) throw std::runtime_error("no visibility buffer before pass " + std::to_string(index));
                MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
                pass.colorAttachments[0].texture = texture;
                pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                pass.visibilityResultBuffer = (__bridge id<MTLBuffer>)visibility;
                pass.visibilityResultType = MTLVisibilityResultTypeAccumulate;
                id<MTL4RenderCommandEncoder> render = Metal::Stream(start.commands).BeginRender(pass);
                [render setRenderPipelineState:pipeline];
                recorder.NoteSampledDraw(start.commands);
                [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                recorder.LeaveRenderPassOpen(100 + index, Recorder::NoTiming, false, PassAccess{});
            }
            recorder.Sync();
            total = recorder.SamplesTotal() - before;
        } catch (const std::exception& thrown) {
            failure = thrown.what();
        }
    }
    [device.queue removeResidencySet:residency];
    Expect(failure.empty(), "sampled passes threw: " + failure);
    Expect(total == std::uint64_t{Passes} * 16u * 16u, "SamplesTotal over " + std::to_string(Passes) + " passes is " + std::to_string(total) + ", expected " + std::to_string(Passes * 256u));
}

// A one-shot command batch alongside the recorder: recorded work goes first in queue order.
void CommandBatchAfterRecorded(const Context& context) {
    Buffer target(context, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(target.Bytes().data(), 0, 4096);
    Lock gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    recorder.Activate();
    recorder.Stream().FillBuffer(Metal::BufferOf(target.Handle()), 0, 4096, 0x01010101u);
    {
        CommandBatch batch(context);
        Expect(!recorder.Recording() && recorder.Submissions() == 1, "a command batch did not submit the open recorded batch first");
        Metal::Stream(batch.Handle()).FillBuffer(Metal::BufferOf(target.Handle()), 0, 64, 0x02020202u);
        batch.SubmitAndWait();
    }
    Expect(WordsOf(target)[0] == 0x02020202u && WordsOf(target)[16] == 0x01010101u, "the command batch did not land after the recorded fill (" + Hex(WordsOf(target)[0]) + ", " + Hex(WordsOf(target)[16]) + ")");
    Expect(recorder.Idle(), "the command batch's wait did not reap the finished recorded batch");
}

// The flush hook (GuestMemory::FlushGpuWrites, which every guest read makes): a CPU access to a
// range a submitted batch writes waits for that batch, with the GPU mutex released, and then sees
// the GPU's bytes. This is what lets the PM4 core read results without draining the device.
void FlushHookWaitsForGpuWrites(const Context& context) {
    constexpr std::size_t bytes = std::size_t{32} << 20u;
    Buffer target(context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(target.Bytes().data(), 0, bytes);
    const auto address = AddressOf(target);
    std::unique_ptr<Recorder> recorder;
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder = std::make_unique<Recorder>(context);
        recorder->Activate();
        auto& stream = recorder->Stream();
        // Eight ordered fills of 32 MiB: long enough that the batch usually still runs at the read.
        for (std::uint32_t fill = 1; fill <= 8; ++fill) {
            if (fill != 1) stream.Barrier(MTLStageBlit, MTLStageBlit);
            stream.FillBuffer(Metal::BufferOf(target.Handle()), 0, bytes, 0x11111111u * fill);
        }
        recorder->NotePendingWrite(address, bytes);
        recorder->Submit();
        Expect(Recorder::SnapshotWriteOverlaps(address + bytes - 64, 64), "the submitted fill is not in the pending-write snapshot");
    }
    const auto waitsBefore = Recorder::ThreadHookWaits();
    AgcDriver::GuestMemory::FlushGpuWrites(address + bytes - 64, 64);
    const auto* words = WordsOf(target);
    Expect(words[bytes / 4 - 1] == 0x88888888u && words[0] == 0x88888888u, "the CPU read after the flush hook saw " + Hex(words[bytes / 4 - 1]) + ", not the last fill");
    std::printf("flush hook: the read %s for the GPU\n", Recorder::ThreadHookWaits() != waitsBefore ? "waited" : "found the batch finished and did not wait");
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        Expect(recorder->Idle() && !Recorder::SnapshotWriteOverlaps(address, bytes), "the hook's sync did not finish the batch and retire its write");
        recorder.reset();
    }
}

// Hundreds of batches of label stores and fills in random program order, with reaps, targeted
// syncs and full syncs between them: memory ends as program order leaves it, every completion runs
// once, and the batch objects and upload memory are recycled without corruption.
void ManyBatches(const Context& context) {
    constexpr std::size_t dwords = 16384;
    constexpr int batches = 600;
    Buffer target(context, dwords * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(target.Bytes().data(), 0, dwords * 4);
    std::vector<std::uint32_t> expected(dwords, 0);
    std::uint64_t seed = 0x2545f4914f6cdd1dull;
    const auto next = [&](std::uint64_t bound) {
        seed ^= seed << 13u;
        seed ^= seed >> 7u;
        seed ^= seed << 17u;
        return seed % bound;
    };
    int completions = 0;
    std::uint32_t value = 1;
    {
        Lock gpu(AgcDriver::GuestMemory::GpuMutex());
        Recorder recorder(context);
        for (int batch = 0; batch < batches; ++batch) {
            const auto stores = 1 + next(4);
            for (std::uint64_t store = 0; store < stores; ++store) {
                const auto first = static_cast<std::size_t>(next(dwords - 8));
                const auto count = static_cast<std::size_t>(1 + next(6));
                std::vector<std::uint32_t> words(count);
                for (auto& word : words) word = value++;
                recorder.RecordStore(target.Handle(), first * 4, Words(words), AddressOf(target, first * 4));
                std::copy(words.begin(), words.end(), expected.begin() + static_cast<std::ptrdiff_t>(first));
            }
            if (next(3) == 0) {
                const auto first = static_cast<std::size_t>(next(dwords - 256));
                const auto count = static_cast<std::size_t>(1 + next(256));
                const auto byte = static_cast<std::uint32_t>(1 + next(255));
                // A later command over queued stores records them first (as the device's fills do).
                recorder.FlushStoresOverlapping(AddressOf(target, first * 4), count * 4);
                auto& stream = recorder.Stream();
                stream.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                stream.FillBuffer(Metal::BufferOf(target.Handle()), first * 4, count * 4, byte * 0x01010101u);
                std::fill(expected.begin() + static_cast<std::ptrdiff_t>(first), expected.begin() + static_cast<std::ptrdiff_t>(first + count), byte * 0x01010101u);
            }
            recorder.OnComplete([&completions] { ++completions; });
            recorder.Submit();
            switch (next(4)) {
            case 0:
                recorder.Reap();
                break;
            case 1: {
                const auto first = static_cast<std::size_t>(next(dwords - 64));
                recorder.NotePendingWrite(AddressOf(target, first * 4), 256);
                recorder.Submit();
                recorder.SyncThrough(AddressOf(target, first * 4), 256);
                break;
            }
            case 2:
                if (next(8) == 0) recorder.Sync();
                break;
            default:
                break;
            }
        }
        recorder.Sync();
        Expect(recorder.Idle() && completions == batches, "after " + std::to_string(batches) + " batches " + std::to_string(completions) + " completions ran and the recorder is " + (recorder.Idle() ? "idle" : "busy"));
    }
    const auto* words = WordsOf(target);
    std::size_t wrong = 0, firstWrong = dwords;
    for (std::size_t i = 0; i < dwords; ++i) {
        if (words[i] == expected[i]) continue;
        if (wrong++ == 0) firstWrong = i;
    }
    Expect(wrong == 0, std::to_string(wrong) + " dwords differ from program order after many batches (first at " + std::to_string(firstWrong) + ": " + (firstWrong < dwords ? Hex(words[firstWrong]) + " instead of " + Hex(expected[firstWrong]) : std::string("-")) + ")");
}

// An overrunning update fails at record time with the package's message, not on the GPU.
void RecordTimeChecks(const Context& context) {
    Buffer target(context, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Lock gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    bool refused = false;
    try {
        recorder.Stream().UpdateBuffer(Metal::BufferOf(target.Handle()), 252, Words({1, 2}));
    } catch (const std::runtime_error& error) {
        refused = std::string_view(error.what()).find("work package D2") != std::string_view::npos;
    }
    Expect(refused, "an update overrunning its buffer was recorded");
    recorder.Sync();
}

// APS5_PROFILE_GPU=1: every batch carries whole-batch stamps from its counter heap, and a timed
// range gets two stamps around its work.
void Timing(const Context& context) {
    Expect(Recorder::GpuTimingEnabled() && Recorder::BatchStampsEnabled(), "the timing run needs APS5_PROFILE_GPU=1");
    Buffer target(context, 1u << 20u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Lock gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    const auto range = recorder.BeginGpuTiming(Recorder::CommandClass::Fill);
    Expect(range != Recorder::NoTiming, "BeginGpuTiming gave no range");
    recorder.Stream().FillBuffer(Metal::BufferOf(target.Handle()), 0, 1u << 20u, 0x33333333u);
    recorder.EndGpuTiming(range, 1u << 20u);
    recorder.Sync();
    std::size_t missing = 0;
    const auto completed = recorder.CompletedBatches(0, recorder.Submissions(), missing);
    Expect(completed.size() == 1 && missing == 0, "the finished batch is not in the completed ring");
    if (!completed.empty()) Expect(completed.front().gpuStartNs > 0 && completed.front().gpuEndNs >= completed.front().gpuStartNs, "the batch stamps are " + std::to_string(completed.front().gpuStartNs) + " .. " + std::to_string(completed.front().gpuEndNs));
}

// A timestamp the GPU never wrote resolves to MTLCounterErrorValue (all ones); such a pair, a zero
// stamp or a pair out of order measures nothing (before, the [gputime] line summed their unsigned
// differences, ~1.8e19 ticks each).
void TimestampPairs() {
    constexpr std::uint64_t unwritten = ~std::uint64_t{0};
    Expect(Metal::TimestampPairValid(100u, 100u) && Metal::TimestampPairValid(100u, 250u), "an ordered pair of written stamps is refused");
    Expect(!Metal::TimestampPairValid(100u, unwritten) && !Metal::TimestampPairValid(unwritten, 100u) && !Metal::TimestampPairValid(unwritten, unwritten), "an unwritten stamp is accepted");
    Expect(!Metal::TimestampPairValid(250u, 100u) && !Metal::TimestampPairValid(0u, 100u), "a reversed pair or a zero stamp is accepted");
}

}

int main(int argc, char** argv) {
    @autoreleasepool {
        MetalTest::Device device;
        const bool timing = argc == 2 && std::string_view(argv[1]) == "--timing";
        try {
            TimestampPairs();
            if (timing) {
                Timing(device.context);
            } else {
                StoreRuns(device.context);
                StoresAfterEarlierWork(device.context);
                KeyStores(device.context);
                SyncThroughAndCompletions(device.context);
                UnlockedWaits(device.context);
                SnapshotAndLabels(device.context);
                Fills(device.context);
                MeshArguments(device.context);
                OcclusionSamples(device);
                OcclusionPasses(device);
                CommandBatchAfterRecorded(device.context);
                FlushHookWaitsForGpuWrites(device.context);
                ManyBatches(device.context);
                RecordTimeChecks(device.context);
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            return 1;
        }
        if (failures != 0) {
            std::fprintf(stderr, "%d Metal recorder checks failed\n", failures);
            return 1;
        }
        std::printf("Metal recorder tests passed\n");
    }
    return 0;
}
