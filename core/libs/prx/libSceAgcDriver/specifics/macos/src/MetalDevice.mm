// The AGC driver's device API (VulkanDevice.hpp, overlaid by
// specifics/macos/overlay/.../VulkanDevice.hpp) implemented on Metal 4 (work package D1,
// docs/mac/M2_DESIGN.md): device setup, the Metal shader target, compute dispatches (direct, split,
// indirect) of MetalBackend shaders over wrapped guest memory with every MslAbi binding the compute
// path has (guest buffers, GDS, BDA page table, fault buffer, flattened SRT, shader data, the
// device-memory LDS arena), GPU labels, fills and copies, and the submission the PM4 core drives.
// Every command records into D2's recorder (specifics/macos/src/Recorder.mm), as upstream
// VulkanDevice.cpp records into the Vulkan one: batches are MTL4 command buffers on the device's
// queue, waits and completions are the recorder's, and its pending writes and label table serve
// the GuestMemory flush hook and WAIT_REG_MEM.

#include "MetalDeviceState.h"
#include "MetalCommandStream.h"
#include "MetalPipelineCache.h"
#include "MetalImages.h"
#include "BdaAbi.hpp"
#include "MslAbi.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include <spirv/unified1/spirv.hpp>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace AgcDriver {

struct PreparedDispatch {};
struct RecipeHit {};
struct RecordedDispatch {};

using Metal::Describe;
using Metal::Unimplemented;

namespace {

namespace Abi = ShaderRecompiler::MslAbi;
namespace Bda = ShaderRecompiler::BdaAbi;
using Graphics::Recorder;

constexpr std::uint64_t HostPage = 16384u;
constexpr std::uint64_t ArenaGranule = 1u << 20u;
// Workgroups per grid dimension the device reports (WorkGroupCountLimit). Metal ran 2^24
// threadgroups in x and 2^20 in y and z on the M3 (no limit is documented); dispatches beyond this
// one are split into chunks whose first workgroup reaches the shader in DriverData.workgroupBase,
// as upstream splits with vkCmdDispatchBase.
constexpr std::uint32_t GroupLimit = 65535u;
constexpr std::uint64_t MaxSplitChunks = 1u << 16u;

// The Vulkan stage and access masks of upstream's barriers (the command stream maps the stages).
constexpr VkPipelineStageFlags AnyStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT;
constexpr VkAccessFlags ShaderAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
constexpr VkAccessFlags DispatchedAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
constexpr VkAccessFlags TransferredAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_HOST_READ_BIT;

std::uint64_t NextSerial() {
    static std::atomic<std::uint64_t> serials{0};
    return serials.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::uint32_t LoopGuardLimit() {
    static const std::uint32_t limit = [] {
        const char* value = std::getenv("APS5_LOOP_GUARD");
        if (value == nullptr) return 1u << 22u;
        const auto parsed = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 0));
        return parsed == 0u ? 0xffffffffu : parsed;
    }();
    return limit;
}

// Debug aid (APS5_NO_OPPORTUNISTIC_REAP=1), as upstream: no reap of finished batches before a
// label, fill, copy or dispatch decision.
bool OpportunisticReap() {
    static const bool enabled = std::getenv("APS5_NO_OPPORTUNISTIC_REAP") == nullptr;
    return enabled;
}

// APS5_METAL_TRACE=1: one stderr line per command the device records (debug aid).
bool Trace() {
    static const bool trace = std::getenv("APS5_METAL_TRACE") != nullptr;
    return trace;
}

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

// A guest V#'s base and the bytes its records span (RDNA2 buffer resource, words 0-2).
struct GuestBufferRange {
    std::uint64_t base;
    std::uint64_t bytes;
};

GuestBufferRange DecodeBuffer(std::span<const std::uint32_t> words) {
    const auto base = static_cast<std::uint64_t>(words[0]) | (static_cast<std::uint64_t>(words[1] & 0xffffu) << 32u);
    const auto stride = (words[1] >> 16u) & 0x3fffu;
    const auto records = static_cast<std::uint64_t>(words[2]);
    return {base, stride == 0u ? records : records * stride};
}

// Why [pageBegin, pageEnd) cannot be wrapped in place, empty when it can: every page must be
// mapped, readable and not executable (code-signed pages fault when wrapped).
std::string Unwrappable(std::uint64_t pageBegin, std::uint64_t pageEnd) {
    mach_vm_address_t cursor = pageBegin;
    while (cursor < pageEnd) {
        mach_vm_address_t region = cursor;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info {};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || region > cursor) {
            return "guest memory at " + Hex(cursor) + " is not mapped";
        }
        if ((info.protection & VM_PROT_EXECUTE) != 0 || (info.protection & VM_PROT_READ) == 0) {
            return "guest memory at " + Hex(cursor) + " is executable or unreadable and cannot be wrapped (copy fallback: work package D3)";
        }
        cursor = region + size;
    }
    return {};
}

std::uint32_t WorkgroupStride(const ShaderRecompiler::RecompileResult& shader) {
    return shader.msl.Reflection().workgroupMemoryStrideDwords;
}

}

// Registered guest memory is reached through its host import, so every GPU access to it (dispatch
// V#s, BDA table, labels, fills, copies) goes through one MTLBuffer: two no-copy buffers over the
// same pages have different GPU addresses, and ordering stores through one against loads through
// the other was seen to fail (agc_driver_copied_writer_copy). Other memory gets page wrappers.
std::uint64_t VulkanDevice::State::TryWrap(std::uint64_t address, std::uint64_t bytes) {
    const auto end = address + std::max<std::uint64_t>(bytes, 1u);
    if (end < address) return 0;
    if (context.hostImportAlignment != 0u) {
        std::uint64_t offset = 0;
        if (id<MTLBuffer> import = ImportOf(address, bytes, &offset)) return import.gpuAddress + offset;
    }
    for (const auto& wrapper : wrappers) {
        if (address >= wrapper.begin && end <= wrapper.end) return wrapper.buffer.gpuAddress + (address - wrapper.begin);
    }
    const auto pageBegin = address & ~(HostPage - 1u);
    const auto pageEnd = (end + HostPage - 1u) & ~(HostPage - 1u);
    if (!Unwrappable(pageBegin, pageEnd).empty()) return 0;
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(pageBegin) length:pageEnd - pageBegin options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
    if (buffer == nil) return 0;
    [residency addAllocation:buffer];
    [residency commit];
    wrappers.push_back({pageBegin, pageEnd, buffer});
    return buffer.gpuAddress + (address - pageBegin);
}

// The host import (GuestBufferMemory.mm, D3) holding guest memory [address, address + bytes): a
// no-copy buffer over a registered guest allocation run, nil when it has none (unregistered or
// unwrappable memory). Labels, fills and copies record on the GPU only into imports, as upstream.
id<MTLBuffer> VulkanDevice::State::ImportOf(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset) const {
    const auto* import = Graphics::HostImportFor(context, address, bytes);
    if (import == nullptr) return nil;
    *offset = address - import->base;
    return Graphics::Metal::BufferOf(import->buffer);
}

// The GPU address of guest (host) memory [address, address + bytes): the 16 KiB pages around it
// wrapped in place (newBufferWithBytesNoCopy) within its VM regions, never executable pages.
std::uint64_t VulkanDevice::State::Wrap(std::uint64_t address, std::uint64_t bytes) {
    if (const auto gpu = TryWrap(address, bytes); gpu != 0) return gpu;
    const auto pageBegin = address & ~(HostPage - 1u);
    const auto pageEnd = (address + std::max<std::uint64_t>(bytes, 1u) + HostPage - 1u) & ~(HostPage - 1u);
    const auto reason = Unwrappable(pageBegin, pageEnd);
    throw std::runtime_error("Metal device: " + (reason.empty() ? "newBufferWithBytesNoCopy refused guest memory at " + Hex(pageBegin) : reason));
}

id<MTLBuffer> VulkanDevice::State::WrapperOf(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset) const {
    const auto end = address + std::max<std::uint64_t>(bytes, 1u);
    for (const auto& wrapper : wrappers) {
        if (address >= wrapper.begin && end <= wrapper.end) {
            *offset = address - wrapper.begin;
            return wrapper.buffer;
        }
    }
    return nil;
}

void VulkanDevice::State::DropWrappersIfIdle() {
    if (recorder == nullptr || !recorder->Idle() || wrappers.empty()) return;
    for (const auto& wrapper : wrappers) [residency removeAllocation:wrapper.buffer];
    [residency commit];
    wrappers.clear();
    ++wrapperEpoch;
    // The cached BDA table points into the dropped wrappers and its lease pins the registry.
    addressTable = {};
}

id<MTLBuffer> VulkanDevice::State::WorkgroupArena(std::uint64_t groups, std::uint32_t strideDwords) {
    const auto groupBytes = static_cast<std::uint64_t>(strideDwords) * 4u;
    const auto limit = static_cast<std::uint64_t>(device.maxBufferLength);
    if (groups != 0u && groupBytes > limit / groups) {
        throw std::runtime_error("Metal device: the device-memory LDS of " + std::to_string(groups) + " workgroups of " + std::to_string(groupBytes) + " bytes each is over maxBufferLength " + std::to_string(limit));
    }
    const auto bytes = std::max<std::uint64_t>(groups, 1u) * groupBytes;
    if (workgroupArena != nil && workgroupArena.length >= bytes) return workgroupArena;
    // Grown only while idle: recorded and in-flight dispatches still index the old arena.
    if (!recorder->Idle()) Sync();
    const auto size = std::min<std::uint64_t>((bytes + ArenaGranule - 1u) / ArenaGranule * ArenaGranule, limit);
    id<MTLBuffer> arena = [device newBufferWithLength:size options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    if (arena == nil) throw std::runtime_error("Metal device: cannot allocate " + std::to_string(size) + " bytes of device-memory LDS");
    // The LDS lock dword of every workgroup slot starts unlocked.
    std::memset(arena.contents, 0, arena.length);
    if (workgroupArena != nil) [residency removeAllocation:workgroupArena];
    [residency addAllocation:arena];
    [residency commit];
    workgroupArena = arena;
    return arena;
}

const std::vector<std::byte>& VulkanDevice::State::AddressTableBytes(std::shared_ptr<const GuestAllocations::Lease>* lease) {
    // The registry generation is read before the lease: a mutation bumps it before it releases the
    // registry, so an unchanged generation means the lease is the one a build would make now.
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (addressTable.generation != generation || addressTable.wrapperEpoch != wrapperEpoch) {
        auto acquired = std::make_shared<GuestAllocations::Lease>(GuestAllocations::GuestAllocationsAcquire_nid_postfix());
        std::vector<Bda::Range> ranges;
        ranges.reserve(acquired->size());
        std::vector<std::pair<std::uint64_t, std::uint64_t>> writable;
        std::size_t skipped = 0;
        for (const auto& range : *acquired) {
            const auto gpu = TryWrap(range->address, range->bytes);
            if (gpu == 0) {
                ++skipped;
                continue;
            }
            const auto permissions = Bda::Read | (range->writable ? Bda::Write : 0u);
            if ((permissions & Bda::Write) != 0u) {
                if (range->address + range->bytes > Bda::WrittenAddressLimit) throw std::runtime_error("Metal device: a writable BDA range ends past the addresses the written-page record can hold");
                writable.push_back({range->address, range->address + range->bytes});
            }
            ranges.push_back({range->address, range->address + range->bytes, gpu, permissions, 0u});
        }
        if (skipped != 0u && addressTable.skipped != skipped) {
            std::fprintf(stderr, "[metal] BDA page table: %zu of %zu guest allocations cannot be wrapped in place and are left out (mirrors: work package D3)\n", skipped, acquired->size());
        }
        // TryWrap may have created wrappers but never drops any, so the epoch is still current.
        const Bda::Header header {Bda::Version, static_cast<std::uint32_t>(ranges.size()), static_cast<std::uint32_t>(sizeof(Bda::Range)), 0u};
        addressTable.bytes.resize(sizeof(header) + ranges.size() * sizeof(Bda::Range));
        std::memcpy(addressTable.bytes.data(), &header, sizeof(header));
        if (!ranges.empty()) std::memcpy(addressTable.bytes.data() + sizeof(header), ranges.data(), ranges.size() * sizeof(Bda::Range));
        if (addressTable.bytes.size() > 0xffffffffu) throw std::runtime_error("Metal device: the BDA page table exceeds 4 GiB");
        addressTable.generation = generation;
        addressTable.wrapperEpoch = wrapperEpoch;
        addressTable.lease = acquired;
        *lease = std::move(acquired);
        addressTable.ranges = std::move(ranges);
        addressTable.writable = std::move(writable);
        addressTable.skipped = skipped;
    }
    *lease = addressTable.lease.lock();
    // Nothing in flight held it: the registry is unchanged (same generation), so a new lease pins
    // the same ranges.
    if (*lease == nullptr) {
        auto renewed = std::make_shared<const GuestAllocations::Lease>(GuestAllocations::GuestAllocationsAcquire_nid_postfix());
        addressTable.lease = renewed;
        *lease = std::move(renewed);
    }
    return addressTable.bytes;
}

void VulkanDevice::State::Sync() {
    if (recorder == nullptr) return;
    if (!recorder->Idle()) Recorder::CountSync(0, __builtin_return_address(0));
    recorder->Sync();
    DropWrappersIfIdle();
    RaisePendingFault();
}

void VulkanDevice::State::RaisePendingFault() {
    std::exception_ptr fault;
    {
        std::lock_guard lock(faultMutex);
        fault = std::exchange(pendingFault, nullptr);
    }
    if (fault != nullptr) std::rethrow_exception(fault);
}

// The BdaAbi fault buffer after a batch with dispatches completed: GPU stores through BDA addresses
// are reported to the write tracker (written-page record), then a published fault record is raised.
void VulkanDevice::State::CheckFaults() {
    auto* words = static_cast<std::uint32_t*>(faults.contents);
    if (words[Bda::WrittenOverflowWord] != 0u) {
        std::memset(faults.contents, 0, Bda::WrittenListWord * sizeof(std::uint32_t));
        throw std::runtime_error("Metal device: more than " + std::to_string(Bda::WrittenPageSlots) + " chunks of " + std::to_string(Bda::WrittenChunkPages) + " pages stored to through the BDA table in one use are not implemented");
    }
    if (const auto count = words[Bda::WrittenCountWord]; count != 0u) {
        const bool valid = Bda::ForEachWrittenRange(words, [](std::uint64_t address, std::size_t bytes) { GuestMemory::MarkWritten(address, bytes); });
        for (std::uint32_t claim = 0; claim < std::min(count, Bda::WrittenPageSlots); ++claim) {
            const auto slot = words[Bda::WrittenListWord + claim];
            if (slot >= Bda::WrittenPageSlots) continue;
            words[Bda::WrittenSlotsWord + 2u * slot] = 0u;
            words[Bda::WrittenSlotsWord + 2u * slot + 1u] = 0u;
        }
        words[Bda::WrittenCountWord] = 0u;
        if (!valid) throw std::runtime_error("Metal device: invalid BDA written-page record");
    }
    Bda::Fault report {};
    std::memcpy(&report, faults.contents, sizeof(report));
    if (report.state == Bda::FaultState::Empty) return;
    std::memset(faults.contents, 0, sizeof(report));
    if (report.state != Bda::FaultState::Ready) throw std::runtime_error("Metal device: incomplete GPU fault record (state " + std::to_string(static_cast<std::uint32_t>(report.state)) + ")");
    if (report.reason == Bda::FaultReason::LoopLimit) {
        throw std::runtime_error("Metal device: GPU loop guard: the loop exit at pc " + Hex(report.instruction) + " of shader " + Hex(report.address) + " ran past " + std::to_string(report.bytes) + " evaluations (APS5_LOOP_GUARD)");
    }
    std::string message = "Metal device: GPU fault: BDA access failed: address=" + Hex(report.address) + " instruction=" + Hex(report.instruction) + " bytes=" + std::to_string(report.bytes) + " stage=" + std::to_string(report.stage) + " reason=" + std::to_string(static_cast<std::uint32_t>(report.reason));
    if (report.reason == Bda::FaultReason::Unmapped) message += " (no guest allocation in the BDA page table covers it)";
    if (report.reason == Bda::FaultReason::Permission) message += " (a store to a read-only range of the BDA page table)";
    throw std::runtime_error(message);
}


// Devices alive in the process, torn down by an exit handler when exit() runs with one still
// alive (a test returning from main through exit(), a title quitting): the GPU driver must never
// see a terminating process with residency sets on a queue, batches in flight or no-copy wrappers
// of static pages (the 2026-10-12 panic, DECISIONS D19.5). Leaked so no static destructor races it;
// the handler is registered after the first device finished construction, so it runs before the
// destructors of statics that construction created.
struct LiveDevices {
    struct Entry {
        const void* state;
        void (*teardown)(const void*);
    };
    std::mutex mutex;
    std::vector<Entry> states;
    std::once_flag handler;
};

LiveDevices& Live() {
    static LiveDevices* const live = new LiveDevices;
    return *live;
}

std::mutex& TeardownReportMutex() {
    static std::mutex* const mutex = new std::mutex;
    return *mutex;
}

Metal::TeardownReport& TeardownReportValue() {
    static Metal::TeardownReport* const report = new Metal::TeardownReport;
    return *report;
}

void TeardownLiveDevicesAtExit() {
    auto& live = Live();
    std::vector<LiveDevices::Entry> states;
    {
        std::lock_guard lock(live.mutex);
        states.swap(live.states);
    }
    for (const auto& entry : states) entry.teardown(entry.state);
}

void RegisterLiveDevice(const void* state, void (*teardown)(const void*)) {
    auto& live = Live();
    {
        std::lock_guard lock(live.mutex);
        live.states.push_back({state, teardown});
    }
    std::call_once(live.handler, [] { std::atexit(TeardownLiveDevicesAtExit); });
}

void UnregisterLiveDevice(const void* state) {
    auto& live = Live();
    std::lock_guard lock(live.mutex);
    std::erase_if(live.states, [&](const LiveDevices::Entry& entry) { return entry.state == state; });
}

std::vector<Bda::Range> Metal::DisjointRanges(std::span<const Bda::Range> ranges) {
    struct Edge {
        std::uint64_t address;
        bool opens;
        std::size_t index;
    };
    std::vector<Edge> edges;
    edges.reserve(ranges.size() * 2u);
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        if (ranges[index].end <= ranges[index].begin) continue;
        edges.push_back({ranges[index].begin, true, index});
        edges.push_back({ranges[index].end, false, index});
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& left, const Edge& right) { return left.address < right.address; });
    std::vector<Bda::Range> result;
    std::set<std::size_t> active;
    for (std::size_t edge = 0; edge < edges.size();) {
        const auto address = edges[edge].address;
        for (; edge < edges.size() && edges[edge].address == address; ++edge) {
            if (edges[edge].opens) active.insert(edges[edge].index);
            else active.erase(edges[edge].index);
        }
        if (active.empty() || edge == edges.size()) continue;
        const auto next = edges[edge].address;
        const auto& owner = ranges[*active.begin()];
        std::uint32_t permissions = 0;
        for (const auto index : active) permissions |= ranges[index].permissions;
        const auto device = owner.deviceAddress + (address - owner.begin);
        if (!result.empty() && result.back().end == address && result.back().permissions == permissions && result.back().deviceAddress + (address - result.back().begin) == device) {
            result.back().end = next;
        } else {
            result.push_back({address, next, device, permissions, 0u});
        }
    }
    return result;
}

Metal::TeardownReport Metal::LastTeardown() {
    std::lock_guard lock(TeardownReportMutex());
    return TeardownReportValue();
}

void VulkanDevice::State::Teardown(bool atExit) noexcept {
    if (tornDown) return;
    tornDown = true;
    tearingDown = true;
    static const bool log = std::getenv("APS5_METAL_TEARDOWN_LOG") != nullptr;
    Metal::TeardownReport report;
    report.atExit = atExit;
    // Sync() drops the wrappers itself once the recorder is idle: count them before it runs.
    const auto heldWrappers = wrappers.size();
    @autoreleasepool {
        // 1. Every submitted batch completes before anything it may use is released. A Sync that
        //    throws (a fault record, a failed completion) has still retired the batches before the
        //    failing one; retry until the recorder is idle (bounded: a hung GPU fails each wait
        //    after the recorder's batch timeout).
        for (std::uint32_t attempt = 0; recorder != nullptr && attempt < 64u; ++attempt) {
            try {
                Sync();
                if (recorder->Idle()) break;
            } catch (const std::exception& error) {
                ++report.syncFailures;
                std::fprintf(stderr, "[metal] teardown: %s\n", error.what());
            }
        }
        report.idle = recorder == nullptr || recorder->Idle();
        // 2. The recorder: its destructor syncs again, joins the release thread and removes its
        //    residency set (and its command stream's) from the queue.
        recorder.reset();
        context.recorder = nullptr;
        // 2b. The host imports (no-copy buffers over guest allocations) and their residency set on
        //     the queue (GuestBufferMemory.mm), once nothing recorded can use them.
        if (context.device != VK_NULL_HANDLE) {
            try {
                Graphics::ClearImageMirrors(context.device);
                Graphics::ClearHostImports(context.device);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[metal] teardown: host imports: %s\n", error.what());
            }
        }
        // 3. The guest-memory wrappers (no-copy buffers over guest and static __DATA pages) and
        //    the cached BDA table holding their addresses.
        report.wrappersDropped = std::max(heldWrappers, wrappers.size());
        if (residency != nil) {
            for (const auto& wrapper : wrappers) [residency removeAllocation:wrapper.buffer];
        }
        wrappers.clear();
        ++wrapperEpoch;
        addressTable = {};
        // 4. The device's own residency set leaves the queue before it is released.
        if (queue != nil && residency != nil) {
            [queue removeResidencySet:residency];
            [residency removeAllAllocations];
            [residency commit];
            report.residencyRemoved = true;
        }
        // The image cache's residency set leaves the queue (D4/D5, MetalImages.h).
        if (images != nullptr) images->Teardown();
        images.reset();
        // The pipeline cache after the recorder: joins its background workers, saves the archive.
        pipelines.reset();
        compiler = nil;
        arguments = nil;
        workgroupArena = nil;
        faults = nil;
        atomicLocks = nil;
        residency = nil;
        // 5. The buffer pool (its residency set leaves the queue in its destructor) and the queue.
        context.bufferPool.reset();
    }
    {
        std::lock_guard lock(TeardownReportMutex());
        report.teardowns = TeardownReportValue().teardowns + 1u;
        TeardownReportValue() = report;
    }
    if (log) {
        std::fprintf(stderr, "[metal] teardown%s: recorder %s, %u failed syncs, %llu wrappers dropped, device residency set %s\n", atExit ? " at exit" : "",
                     report.idle ? "idle" : "NOT idle", report.syncFailures, static_cast<unsigned long long>(report.wrappersDropped), report.residencyRemoved ? "removed from the queue" : "not attached");
    }
}

VulkanDevice::VulkanDevice(const PresentationWindow* window) : VulkanDevice(window, false) {
}

VulkanDevice::VulkanDevice(CompileOnly) : VulkanDevice(nullptr, true) {
}

VulkanDevice::VulkanDevice(const PresentationWindow* window, bool compileOnly) : state(std::make_unique<State>()), serial(NextSerial()) {
    try {
        if (window != nullptr) Unimplemented("presentation to a window", "work package D6");
        @autoreleasepool {
            state->device = MTLCreateSystemDefaultDevice();
            if (state->device == nil) throw std::runtime_error("Metal device: no Metal device");
            if (![state->device supportsFamily:MTLGPUFamilyMetal4] || ![state->device supportsFamily:MTLGPUFamilyApple7]) throw std::runtime_error("Metal device: the GPU lacks the Metal 4 / Apple7 feature set this driver needs");
            NSError* error = nil;
            state->pipelines = std::make_unique<Graphics::Metal::PipelineCache>(state->device);
            state->compiler = state->pipelines->Compiler();
            if (state->compiler == nil) throw std::runtime_error("Metal device: no shader compiler");
            for (const auto capability : {spv::CapabilityShader, spv::CapabilityInt64, spv::CapabilityInt64Atomics, spv::CapabilityInt16, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformArithmetic, spv::CapabilitySignedZeroInfNanPreserve, spv::CapabilityImageQuery}) {
                state->capabilities.push_back(static_cast<std::uint32_t>(capability));
            }
            // f64 is IEEE binary64 in software in the MSL backend (MslFloat64.cpp, package B2).
            state->capabilities.push_back(static_cast<std::uint32_t>(spv::CapabilityFloat64));
            // half arithmetic rounds to nearest even and keeps f16 denormals on Apple GPUs (probe_float_b2),
            // so packed f16 add/mul run natively (FPAdd16x2/FPMul16x2, MslAluMath.cpp).
            for (const auto capability : {spv::CapabilityFloat16, spv::CapabilityRoundingModeRTE, spv::CapabilityDenormPreserve}) state->capabilities.push_back(static_cast<std::uint32_t>(capability));
            std::ranges::sort(state->capabilities);
            state->extensions = {"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
            if (compileOnly) return;
            state->queue = [state->device newMTL4CommandQueue];
            MTLResidencySetDescriptor* residency = [MTLResidencySetDescriptor new];
            residency.initialCapacity = 64;
            state->residency = [state->device newResidencySetWithDescriptor:residency error:&error];
            if (state->queue == nil || state->residency == nil) throw std::runtime_error("Metal device: queue setup failed: " + Describe(error));
            state->faults = [state->device newBufferWithLength:Bda::FaultBufferBytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            if (state->faults == nil) throw std::runtime_error("Metal device: cannot allocate the fault buffer");
            std::memset(state->faults.contents, 0, state->faults.length);
            [state->residency addAllocation:state->faults];
            state->atomicLocks = [state->device newBufferWithLength:Abi::AtomicLockTableBytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            if (state->atomicLocks == nil) throw std::runtime_error("Metal device: cannot allocate the 64-bit atomic lock table");
            // Every lock word starts released; only the GPU touches the table after this.
            std::memset(state->atomicLocks.contents, 0, state->atomicLocks.length);
            [state->residency addAllocation:state->atomicLocks];
            [state->residency commit];
            [state->queue addResidencySet:state->residency];
            MTL4ArgumentTableDescriptor* table = [MTL4ArgumentTableDescriptor new];
            table.maxBufferBindCount = Abi::Index(Abi::BufferIndex::Count);
            state->images = std::make_unique<Graphics::Metal::ImageCache>(state->device, state->queue);
            state->arguments = [state->device newArgumentTableWithDescriptor:table error:&error];
            if (state->arguments == nil) throw std::runtime_error("Metal device: argument table setup failed: " + Describe(error));

            // The Context of the D2-D6 twins (D2's integration steps, docs/mac/M2_DESIGN.md §9 D2).
            auto& context = state->context;
            context.device = Graphics::Metal::DeviceHandle(state->device);
            context.queue = Graphics::Metal::QueueHandle(state->queue);
            context.bufferDeviceAddress = true;
            // Host imports (D3): no-copy buffers over whole 16 KiB pages of guest allocations.
            context.hostImportAlignment = HostPage;
            const auto bufferLength = static_cast<std::uint64_t>(state->device.maxBufferLength);
            context.limits.maxStorageBufferRange = static_cast<std::uint32_t>(std::min<std::uint64_t>(bufferLength, 0xffffffffu));
            context.limits.maxUniformBufferRange = context.limits.maxStorageBufferRange;
            context.limits.maxPushConstantsSize = Abi::PushDataBytes;
            context.limits.maxBoundDescriptorSets = 4u;
            context.limits.minStorageBufferOffsetAlignment = 4u;
            context.limits.nonCoherentAtomSize = 1u;
            const auto threads = state->device.maxThreadsPerThreadgroup;
            context.limits.maxComputeWorkGroupCount[0] = context.limits.maxComputeWorkGroupCount[1] = context.limits.maxComputeWorkGroupCount[2] = GroupLimit;
            context.limits.maxComputeWorkGroupSize[0] = static_cast<std::uint32_t>(threads.width);
            context.limits.maxComputeWorkGroupSize[1] = static_cast<std::uint32_t>(threads.height);
            context.limits.maxComputeWorkGroupSize[2] = static_cast<std::uint32_t>(threads.depth);
            context.limits.maxComputeWorkGroupInvocations = 1024u;
            context.limits.maxComputeSharedMemorySize = static_cast<std::uint32_t>(state->device.maxThreadgroupMemoryLength);
            context.subgroup.subgroupSize = 32u;
            context.bufferPool = std::make_shared<Graphics::BufferPool>(context);
            state->recorder = std::make_unique<Graphics::Recorder>(context, true);
            context.recorder = state->recorder.get();
            state->recorder->Activate();
            Graphics::PrepareImportWatch(context);
        }
    } catch (...) {
        // A half-built device releases what it attached (residency sets on the queue) before unwinding.
        state->Teardown(false);
        throw;
    }
    RegisterLiveDevice(state.get(), [](const void* live) { const_cast<State*>(static_cast<const State*>(live))->Teardown(true); });
}

VulkanDevice::~VulkanDevice() {
    if (state == nullptr) return;
    state->Teardown(false);
    UnregisterLiveDevice(state.get());
}

ComputePipelineReplayResult VulkanDevice::ReplayComputePipeline(const ComputePipelineCapture&, std::span<const std::byte>) {
    Unimplemented("ReplayComputePipeline (Vulkan pipeline captures)", "work package D1");
}

std::string VulkanDevice::DeviceName() const {
    return state->device.name.UTF8String;
}

ShaderRecompiler::SpirvTarget VulkanDevice::Target() const {
    const auto threadgroupMemory = static_cast<std::uint32_t>(state->device.maxThreadgroupMemoryLength);
    const auto threads = state->device.maxThreadsPerThreadgroup;
    // fragmentShaderBarycentricEnabled stays false: the front end's barycentric mode interpolates
    // from per-vertex attribute values (VK_KHR_fragment_shader_barycentric), which Metal has only
    // on Apple10 (vertex_value); Apple7+ has [[barycentric_coord]] alone (DeviceProfile, B6).
    ShaderRecompiler::SpirvTarget target {0u, 0u, 32u, Bda::Version, state->capabilities, state->extensions, false, {static_cast<std::uint32_t>(threads.width), static_cast<std::uint32_t>(threads.height), static_cast<std::uint32_t>(threads.depth)}, 1024u, threadgroupMemory, {}, {}};
    target.backend = ShaderRecompiler::ShaderBackend::Msl;
    return target;
}

const std::uint32_t* VulkanDevice::WorkGroupCountLimit() const {
    static constexpr std::uint32_t limits[3] = {GroupLimit, GroupLimit, GroupLimit};
    return limits;
}

std::uint32_t VulkanDevice::VendorId() const {
    return 0x106bu;
}

ShaderRecompiler::SpirvTarget VulkanDevice::ComputeTarget(std::uint32_t) const {
    return Target();
}

VkDevice VulkanDevice::Device() const {
    return state->context.device;
}

void VulkanDevice::WaitIdle() {
    state->Sync();
}

void VulkanDevice::PrepareForReplacement() {
    GuestMemory::AssertGpuLockHeld("VulkanDevice::PrepareForReplacement");
    WaitIdle();
}

bool VulkanDevice::ImportGuestMemory(const GuestAllocations::Mapped& ranges, std::uint64_t generation, bool adoptDevice) {
    if (state->recorder == nullptr) return true;
    return Graphics::ImportMappedRanges(state->context, ranges, generation, adoptDevice);
}

void VulkanDevice::SubmitRecorded(bool reapFirst) {
    if (state->recorder == nullptr) return;
    if (reapFirst && OpportunisticReap()) {
        state->recorder->Reap();
        state->DropWrappersIfIdle();
    }
    state->recorder->Submit();
}

void VulkanDevice::StoreIdleResults(bool) {
    // Pending storage-image results are D4's (StorageTexture); the device has none yet.
}

bool VulkanDevice::CanWaitUnlocked() const {
    return state->recorder != nullptr && state->recorder->HasTimeline();
}

std::uint64_t VulkanDevice::SubmitAndEpoch() {
    if (state->recorder == nullptr) return 0;
    if (OpportunisticReap()) {
        state->recorder->Reap();
        state->DropWrappersIfIdle();
    }
    return state->recorder->SubmitAndEpoch();
}

void VulkanDevice::WaitRecorded(std::uint64_t serial) {
    if (state->recorder != nullptr) state->recorder->WaitSerial(serial);
}

void VulkanDevice::ReapRecorded(std::uint64_t serial) {
    if (state->recorder == nullptr) return;
    Recorder::AnnounceSyncSite(__builtin_return_address(0));
    state->recorder->FinishUpTo(serial);
    state->DropWrappersIfIdle();
    state->RaisePendingFault();
}

void VulkanDevice::ReapRecorded() {
    if (state->recorder == nullptr) return;
    state->recorder->Reap();
    state->DropWrappersIfIdle();
}

// Upstream's rule: a store behind the recorded work (into the wrapped guest memory, through the
// recorder's store runs and label table), or the CPU's when nothing is recorded (1). The device has
// no copied writers (every guest buffer is wrapped in place), so no label waits behind a write-back
// completion (5).
int VulkanDevice::WriteLabelOnGpu(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue, bool reapFirst) {
    if (state->recorder == nullptr || bytes.empty() || bytes.size() > 65536u || address % 4u != 0u || bytes.size() % 4u != 0u) return 4;
    auto& recorder = *state->recorder;
    static const bool reapAllQueues = std::getenv("APS5_LABEL_REAP_ALL_QUEUES") != nullptr;
    if (reapFirst && (queue == 0 || reapAllQueues) && OpportunisticReap()) {
        recorder.Reap();
        state->DropWrappersIfIdle();
    }
    if (recorder.Idle()) return 1;
    @autoreleasepool {
        std::uint64_t offset = 0;
        id<MTLBuffer> wrapper = state->ImportOf(address, bytes.size(), &offset);
        if (wrapper == nil) return 3;
        recorder.RecordStore(Graphics::Metal::BufferHandle(wrapper), offset, bytes, address);
    }
    recorder.NoteLabel(address, bytes, stamp, queue);
    GuestMemory::MarkWritten(address, bytes.size());
    if (Trace()) std::fprintf(stderr, "[metal] label 0x%llx+0x%zx stamp %llu queue %u\n", static_cast<unsigned long long>(address), bytes.size(), static_cast<unsigned long long>(stamp), queue);
    static const bool submitNow = std::getenv("APS5_LABEL_SUBMIT_NOW") != nullptr;
    if (submitNow) recorder.Submit();
    return 0;
}

bool VulkanDevice::AfterRecordedWork(std::function<void()> action, bool reapFirst) {
    if (state->recorder == nullptr) return false;
    if (reapFirst && OpportunisticReap()) {
        state->recorder->Reap();
        state->DropWrappersIfIdle();
    }
    return state->recorder->AfterRecordedWork(std::move(action));
}

std::optional<Graphics::Recorder::LabelHit> VulkanDevice::PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, Graphics::Recorder::LabelRefusal* refusal) const {
    if (state->recorder == nullptr) return std::nullopt;
    return state->recorder->PendingLabel(address, bytes, afterStamp, refusal);
}

bool VulkanDevice::OpenWriteOverlaps(std::uint64_t address, std::size_t bytes) const {
    return state->recorder != nullptr && state->recorder->OpenWriteOverlaps(address, bytes);
}

bool VulkanDevice::RecordedWritesSettled(std::uint64_t address, std::size_t bytes) const {
    if (state->recorder == nullptr) return true;
    const auto& recorder = *state->recorder;
    if (!recorder.PendingWriteSettled(address, bytes)) return false;
    return Recorder::PendingCompletionLabels() == 0 || !recorder.CompletionLabelIn(address, bytes);
}

bool VulkanDevice::StoresPendingOver(std::uint64_t address, std::size_t bytes) const {
    return state->recorder != nullptr && (state->recorder->PendingWriteOverlaps(address, bytes) || state->recorder->PendingLabelIn(address, bytes));
}

// Upstream's rule over the recorder: earlier recorded stores into the range are ordered before the
// fill by its barrier; a completion label pending over the range is waited for (it lands by a CPU
// store at reap). A dword-uniform pattern is the stream's fill, any other the fill kernel.
bool VulkanDevice::FillBuffer(std::uint64_t address, std::size_t bytes, std::span<const std::uint32_t, 4> pattern) {
    if (state->recorder == nullptr || bytes == 0u || bytes % 16u != 0u || address % 16u != 0u) return false;
    auto& recorder = *state->recorder;
    if (OpportunisticReap()) {
        recorder.Reap();
        state->DropWrappersIfIdle();
    }
    @autoreleasepool {
        std::uint64_t offset = 0;
        if (state->ImportOf(address, bytes, &offset) == nil) return false;
        static const bool alwaysSync = std::getenv("APS5_FILL_SYNC") != nullptr;
        if (recorder.PendingWriteOverlaps(address, bytes) && (alwaysSync || (Recorder::PendingCompletionLabels() != 0 && recorder.PendingLabelIn(address, bytes)))) {
            Recorder::CountSync(4);
            recorder.SyncThrough(address, bytes);
        }
        recorder.FlushKeyStoresOverlapping(address, bytes);
        recorder.FlushStoresOverlapping(address, bytes);
        id<MTLBuffer> wrapper = state->ImportOf(address, bytes, &offset);
        if (wrapper == nil) throw std::runtime_error("Metal device: the fill's host import vanished while it waited");
        VkAccessFlags covered = 0;
        auto& stream = recorder.Stream(&covered);
        if ((covered & VK_ACCESS_TRANSFER_WRITE_BIT) != 0u && Recorder::MergeBarriers()) {
            Recorder::CountMerged(Recorder::CommandClass::Fill);
        } else {
            stream.PipelineBarrier(AnyStage, VK_PIPELINE_STAGE_TRANSFER_BIT);
            Recorder::CountBarriers(Recorder::CommandClass::Fill);
        }
        const bool uniform = pattern[0] == pattern[1] && pattern[1] == pattern[2] && pattern[2] == pattern[3];
        if (uniform) {
            stream.FillBuffer(wrapper, offset, bytes, pattern[0]);
        } else {
            stream.FillPattern(wrapper, offset, bytes, pattern);
        }
        stream.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, AnyStage);
        Recorder::CountBarriers(Recorder::CommandClass::Fill);
        recorder.MarkCovered(TransferredAccess);
        if (uniform && pattern[0] == (pattern[0] & 0xffu) * 0x01010101u) {
            recorder.NotePendingFill(address, bytes, static_cast<std::uint8_t>(pattern[0]));
        } else {
            recorder.NotePendingWrite(address, bytes);
        }
    }
    GuestMemory::MarkWritten(address, bytes);
    if (Trace()) std::fprintf(stderr, "[metal] fill 0x%llx+0x%zx pattern %08x %08x %08x %08x\n", static_cast<unsigned long long>(address), bytes, pattern[0], pattern[1], pattern[2], pattern[3]);
    return true;
}

bool VulkanDevice::DumpSamplesOnGpu(std::uint64_t) {
    return false;
}

// Upstream's rule over the recorder (VulkanDevice.cpp CopyBuffer) without the parts other
// packages own (storage images D4, unit shadows D3, copied writers D5): path 0 copies on the CPU
// when neither range has an unsettled recorded write, a pending label or an in-place reader;
// path 1 records a copy between the wrappers behind the recorded work.
VulkanDevice::CopyOutcome VulkanDevice::CopyBuffer(std::uint64_t destination, std::uint64_t source, std::size_t bytes, std::size_t cpuMax, std::size_t gpuMax, std::size_t, std::uint64_t, std::uint32_t, const CopyWriterNote& noteWriter) {
    CopyOutcome outcome {2, false, 0, CopyOutcome::None, false, false, 0, 0, 0, 0, false};
    if (state->recorder == nullptr || bytes == 0u) return outcome;
    auto& recorder = *state->recorder;
    const auto completionStoreOverlaps = [&](std::uint64_t address) { return Recorder::PendingCompletionLabels() != 0 && recorder.PendingLabelIn(address, bytes); };
    bool settledBySignal = false;
    const auto refusal = [&]() -> int {
        settledBySignal = false;
        if (bytes > cpuMax) return CopyOutcome::Size;
        if (recorder.PendingWriteOverlaps(source, bytes)) {
            if (!recorder.PendingWriteSettled(source, bytes)) return CopyOutcome::SourcePending;
            if (completionStoreOverlaps(source) || recorder.PendingLabelIn(source, bytes)) return CopyOutcome::SourceUnsettled;
            settledBySignal = true;
        }
        if (recorder.PendingWriteOverlaps(destination, bytes)) {
            static const bool strict = std::getenv("APS5_COPY_DESTINATION_STRICT") != nullptr;
            if (strict || !recorder.PendingWriteSettled(destination, bytes)) return CopyOutcome::DestinationPending;
            if (completionStoreOverlaps(destination)) return CopyOutcome::DestinationUnsettled;
        }
        if (recorder.PendingLabelIn(destination, bytes)) return CopyOutcome::Label;
        if (Recorder::ReadTracking() ? recorder.PendingReadOverlaps(destination, bytes) : !recorder.Idle()) return CopyOutcome::Reader;
        return CopyOutcome::None;
    };
    outcome.reason = refusal();
    if (outcome.reason == CopyOutcome::None) {
        GuestMemory::StoreOwnBytes(destination, bytes, [&] { std::memmove(reinterpret_cast<void*>(destination), reinterpret_cast<const void*>(source), bytes); });
        outcome.path = 0;
        outcome.sourceSettledBySignal = settledBySignal;
        return outcome;
    }
    if (bytes > gpuMax) {
        outcome.path = 8;
        return outcome;
    }
    @autoreleasepool {
        std::uint64_t probeOffset = 0;
        if (state->ImportOf(destination, bytes, &probeOffset) == nil || state->ImportOf(source, bytes, &probeOffset) == nil) return outcome;
        const bool waitSource = recorder.PendingWriteOverlaps(source, bytes) && completionStoreOverlaps(source);
        const bool waitDestination = recorder.PendingWriteOverlaps(destination, bytes) && completionStoreOverlaps(destination);
        if (waitSource || waitDestination) {
            outcome.synced = true;
            const auto start = std::chrono::steady_clock::now();
            if (waitSource) {
                Recorder::CountSync(4);
                recorder.SyncThrough(source, bytes);
            }
            if (waitDestination) {
                Recorder::CountSync(4);
                recorder.SyncThrough(destination, bytes);
            }
            outcome.syncMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        recorder.FlushKeyStoresOverlapping(source, bytes);
        recorder.FlushKeyStoresOverlapping(destination, bytes);
        recorder.FlushStoresOverlapping(source, bytes);
        recorder.FlushStoresOverlapping(destination, bytes);
        std::uint64_t destinationOffset = 0;
        std::uint64_t sourceOffset = 0;
        id<MTLBuffer> destinationBuffer = state->ImportOf(destination, bytes, &destinationOffset);
        id<MTLBuffer> sourceBuffer = state->ImportOf(source, bytes, &sourceOffset);
        if (destinationBuffer == nil || sourceBuffer == nil) throw std::runtime_error("Metal device: the copy's host imports vanished while it waited");
        VkAccessFlags covered = 0;
        auto& stream = recorder.Stream(&covered);
        constexpr VkAccessFlags transferAccess = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        if ((covered & transferAccess) == transferAccess && Recorder::MergeBarriers()) {
            Recorder::CountMerged(Recorder::CommandClass::Copy);
        } else {
            stream.PipelineBarrier(AnyStage, VK_PIPELINE_STAGE_TRANSFER_BIT);
            Recorder::CountBarriers(Recorder::CommandClass::Copy);
        }
        stream.CopyBuffer(sourceBuffer, sourceOffset, destinationBuffer, destinationOffset, bytes);
        stream.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, AnyStage);
        Recorder::CountBarriers(Recorder::CommandClass::Copy);
        recorder.MarkCovered(TransferredAccess);
        recorder.NotePendingRead(source, bytes, Recorder::ReadKind::CopySource);
    }
    GuestMemory::MarkWritten(destination, bytes);
    noteWriter({}, 0);
    recorder.NotePendingWrite(destination, bytes);
    if (Trace()) std::fprintf(stderr, "[metal] copy 0x%llx <- 0x%llx +0x%zx\n", static_cast<unsigned long long>(destination), static_cast<unsigned long long>(source), bytes);
    outcome.path = 1;
    return outcome;
}

VulkanDevice::CopyVerification VulkanDevice::CopyVerifyCounts() {
    return {};
}

VulkanDevice::CopyAliasing VulkanDevice::CopyAliasCounts() {
    return {};
}

std::shared_ptr<PreparedDispatch> VulkanDevice::PrepareDispatch(const ShaderRecompiler::RecompileResult& shader, std::span<const Graphics::GuestMemorySnapshot>, std::uint64_t) {
    // The pipeline starts building on the cache's workers; Dispatch waits for it.
    if (state->pipelines != nullptr && state->recorder != nullptr && !shader.msl.empty() && shader.msl.Reflection().stage == Abi::Stage::Kernel) state->pipelines->PrepareCompute(shader);
    return nullptr;
}

std::span<const double, 5> VulkanDevice::PreparePhaseMs(const PreparedDispatch&) {
    static constexpr std::array<double, 5> zero {};
    return zero;
}

void VulkanDevice::State::RequireDispatchable(const ShaderRecompiler::RecompileResult& shader) const {
    if (recorder == nullptr) throw std::runtime_error("Metal device: a compile-only device cannot dispatch");
    if (shader.msl.empty()) throw std::runtime_error("Metal device: the shader has no MSL form (recompile it for the device's Target(), whose backend is Msl)");
    ShaderRecompiler::RuntimeAbi::RequireVersion(shader.runtimeAbiVersion);
    const auto& reflection = shader.msl.Reflection();
    if (reflection.abiVersion != Abi::Version) throw std::runtime_error("Metal device: the shader was emitted for MslAbi version " + std::to_string(reflection.abiVersion) + ", the device binds version " + std::to_string(Abi::Version));
    if (reflection.stage != Abi::Stage::Kernel) throw std::runtime_error("Metal device: Dispatch needs a kernel");
}

// Records one compute dispatch of `shader` into the recorder's open batch (see the binding
// contract in MslAbi.hpp):
// - DescriptorTable: guest buffers (wrapped V# ranges), Gds (the 64 KiB global data share, guest
//   memory at Pm4::GdsAddress()), BdaPagetable (BdaAbi::Header + Range[] of every readable guest
//   allocation, wrapped in place: begin/end are guest addresses, deviceAddress the GPU address of
//   `begin`, Write when the allocation is writable), FaultBuffer (the device's BdaAbi fault and
//   written-page record, BdaAbi::FaultBufferBytes), FlattenedSrt and ShaderData (copied).
// - PushData: the 256-byte block, the shader's push constants at byte 0.
// - WorkgroupMemory: the device-memory LDS arena when the reflection has a stride (workgroup w of
//   the grid, linear over threadgroups_per_grid, uses dwords [w*stride, (w+1)*stride), the last
//   one the LDS lock, zero before first use); otherwise the DriverData record (never read).
// - DriverData: workgroupBase (the chunk's first workgroup of a split dispatch, else 0),
//   loopGuardLimit, faultBuffer (the fault buffer's GPU address).
// Barriers as upstream: a leading one (unless the last command's trailing barrier covered shader
// access) and a trailing one; the written V# ranges, GDS and every writable range of a bound BDA
// table are noted as pending writes, the bound V# ranges as in-place reads.
std::vector<Bda::Range> VulkanDevice::State::AddressExtras(const ShaderRecompiler::RecompileResult& shader, std::span<const Graphics::GuestMemorySnapshot> snapshots, Graphics::Metal::CommandStream& stream) {
    std::vector<Bda::Range> extras;
    for (const auto& binding : shader.bindings) {
        if (binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
            const auto elements = std::min<std::size_t>(binding.count, binding.guestDescriptor.size() / 4u);
            for (std::uint32_t element = 0; element < elements; ++element) {
                const auto guest = DecodeBuffer(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 4u, 4u));
                if (guest.base == 0u || guest.bytes == 0u) continue;
                const auto aligned = guest.base & ~std::uint64_t{3u};
                const auto bytes = (guest.base - aligned + guest.bytes + 3u) & ~std::uint64_t{3u};
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                extras.push_back({aligned, aligned + bytes, Wrap(aligned, bytes), Bda::Read | (written ? Bda::Write : 0u), 0u});
            }
        } else if (binding.role == ShaderRecompiler::DescriptorRole::Gds) {
            const auto gds = Pm4::GdsAddress();
            extras.push_back({gds, gds + Pm4::GdsBytes, Wrap(gds, Pm4::GdsBytes), Bda::Read | Bda::Write, 0u});
        }
    }
    for (const auto& snapshot : snapshots) {
        if (snapshot.bytes.empty()) continue;
        const auto end = snapshot.address + snapshot.bytes.size();
        if (const auto gpu = TryWrap(snapshot.address, snapshot.bytes.size()); gpu != 0u) {
            extras.push_back({snapshot.address, end, gpu, Bda::Read, 0u});
        } else {
            // Unwrappable (executable) pages: the GPU reads the bytes captured for this dispatch.
            const auto copy = stream.Write(snapshot.bytes, 16u);
            extras.push_back({snapshot.address, end, copy.address, Bda::Read, 0u});
        }
    }
    return extras;
}

void VulkanDevice::State::RecordCompute(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint64_t indirectArguments, std::span<const Graphics::GuestMemorySnapshot> snapshots) {
    const auto& reflection = shader.msl.Reflection();
    const auto stride = WorkgroupStride(shader);
    const bool split = indirectArguments == 0u && (x > GroupLimit || y > GroupLimit || z > GroupLimit);
    if (split) {
        if (stride != 0u) {
            throw std::runtime_error("Metal device: workgroup count " + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + " exceeds the device limits " + std::to_string(GroupLimit) + " per dimension and a split aliases the device-memory LDS slots, whose base is computed from threadgroups_per_grid of a single chunk");
        }
        std::uint64_t chunks = 1;
        for (const std::uint64_t count : {x, y, z}) chunks *= (count + GroupLimit - 1u) / GroupLimit;
        if (chunks > MaxSplitChunks) throw std::runtime_error("Metal device: a dispatch of " + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + " workgroups needs more than " + std::to_string(MaxSplitChunks) + " chunks");
    }
    auto& recorder = *this->recorder;
    // The arena is sized for the whole grid (an indirect dispatch with an arena reads its counts on
    // the CPU first, so the grid is direct here); growing it waits for idle, before the open batch
    // is taken below.
    id<MTLBuffer> arena = stride != 0u ? WorkgroupArena(static_cast<std::uint64_t>(x) * y * z, stride) : nil;
    id<MTLComputePipelineState> pipeline = pipelines->Compute(shader);
    const auto& threads = reflection.threadsPerThreadgroup;
    if (static_cast<NSUInteger>(threads[0]) * threads[1] * threads[2] > pipeline.maxTotalThreadsPerThreadgroup) {
        throw std::runtime_error("Metal device: the workgroup of " + std::to_string(threads[0] * threads[1] * threads[2]) + " threads exceeds the pipeline's maximum threadgroup size " + std::to_string(pipeline.maxTotalThreadsPerThreadgroup));
    }

    // Guest ranges first (queued label and key stores into them must precede the dispatch).
    std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    bool addressBased = false;
    for (const auto& binding : shader.bindings) {
        if (binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
            const auto elements = std::min<std::size_t>(binding.count, binding.guestDescriptor.size() / 4u);
            for (std::uint32_t element = 0; element < elements; ++element) {
                const auto guest = DecodeBuffer(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 4u, 4u));
                if (guest.base == 0u || guest.bytes == 0u) continue;
                const auto aligned = guest.base & ~std::uint64_t{3u};
                const auto end = aligned + ((guest.base - aligned + guest.bytes + 3u) & ~std::uint64_t{3u});
                reads.push_back({aligned, end});
                // An element the recompiler did not classify counts as written.
                if (element >= binding.bufferWritten.size() || binding.bufferWritten[element]) writes.push_back({aligned, end});
            }
        } else if (binding.role == ShaderRecompiler::DescriptorRole::Gds) {
            const auto gds = Pm4::GdsAddress();
            reads.push_back({gds, gds + Pm4::GdsBytes});
            writes.push_back({gds, gds + Pm4::GdsBytes});
        } else if (binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable) {
            addressBased = true;
        }
    }
    if (indirectArguments != 0u) reads.push_back({indirectArguments, indirectArguments + 12u});
    const auto touches = [&](std::uint64_t begin, std::uint64_t end) {
        const auto overlaps = [&](const auto& ranges) { return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.first < end && begin < range.second; }); };
        return overlaps(reads) || overlaps(writes);
    };
    if (recorder.HasQueuedKeyStores() && (addressBased || recorder.AnyQueuedKeyStore(touches))) recorder.FlushKeyStores();
    if (recorder.HasQueuedStores() && (addressBased || recorder.AnyQueuedStore(touches))) recorder.FlushStores();

    // Images and samplers (D4/D5, MetalImages.h) before the open batch is taken: uploading a
    // surface reads guest memory on the CPU, and the settle hook may sync (submit) the batch.
    Graphics::Metal::ImageBindings imageBindings(*images, [this](std::uint64_t address, std::uint64_t bytes) {
        if (this->recorder->PendingWriteOverlaps(address, bytes) || this->recorder->PendingLabelIn(address, bytes)) {
            Recorder::CountSync(4);
            this->recorder->SyncThrough(address, bytes);
        }
    });
    std::vector<std::pair<std::uint32_t, std::vector<std::uint64_t>>> imageEntries;
    if (reflection.table.bytes != 0u) {
        for (const auto& binding : shader.bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages && binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers) continue;
            const auto kind = static_cast<ShaderRecompiler::RuntimeAbi::Binding>(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count));
            const auto* range = reflection.table.Find(kind);
            if (range == nullptr) continue;
            std::vector<std::uint64_t> entries(std::min(binding.count, range->count), 0u);
            imageBindings.Write(binding, entries);
            imageEntries.push_back({range->offset, std::move(entries)});
        }
    }

    VkAccessFlags covered = 0;
    auto& stream = recorder.Stream(&covered);
    std::array<std::byte, Abi::PushDataBytes> push {};
    if (!shader.pushConstants.empty()) {
        if (shader.pushConstants.size() > push.size()) throw std::runtime_error("Metal device: push constants exceed the push data block");
        std::memcpy(push.data(), shader.pushConstants.data(), shader.pushConstants.size());
    }
    std::uint64_t tableAddress = 0;
    if (reflection.table.bytes != 0u) {
        const auto upload = stream.Allocate(reflection.table.bytes, 16u);
        tableAddress = upload.address;
        std::memset(upload.host, 0, reflection.table.bytes);
        auto* table = upload.host;
        for (const auto& binding : shader.bindings) {
            const auto group = binding.binding / static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
            const auto kind = static_cast<ShaderRecompiler::RuntimeAbi::Binding>(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count));
            if (group != static_cast<std::uint32_t>(reflection.group)) throw std::runtime_error("Metal device: a binding belongs to another stage group");
            const auto* range = reflection.table.Find(kind);
            if (range == nullptr) continue;
            auto* entries = reinterpret_cast<Abi::BufferEntry*>(table + range->offset);
            switch (binding.role) {
            case ShaderRecompiler::DescriptorRole::GuestBuffers:
                if (binding.guestDescriptor.size() < static_cast<std::size_t>(std::min(binding.count, range->count)) * 4u) throw std::runtime_error("Metal device: a guest buffer binding has fewer than four words per element");
                for (std::uint32_t element = 0; element < std::min(binding.count, range->count); ++element) {
                    const auto guest = DecodeBuffer(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 4u, 4u));
                    if (guest.base == 0u || guest.bytes == 0u) continue;
                    const auto aligned = guest.base & ~std::uint64_t{3u};
                    const auto bytes = (guest.base - aligned + guest.bytes + 3u) & ~std::uint64_t{3u};
                    if (bytes > 0xffffffffu) throw std::runtime_error("Metal device: a guest buffer exceeds 4 GiB");
                    entries[element] = {Wrap(aligned, bytes), static_cast<std::uint32_t>(bytes), 0u};
                    const auto position = static_cast<std::size_t>(shader.memoryOffsetDword) * 4u + element;
                    if (position < shader.pushConstants.size()) push[position] = std::byte{0};
                }
                break;
            case ShaderRecompiler::DescriptorRole::ShaderData:
            case ShaderRecompiler::DescriptorRole::FlattenedSrt: {
                const auto bytes = binding.guestDescriptor.size() * sizeof(std::uint32_t);
                const auto data = stream.Allocate(std::max<std::size_t>(bytes, 4u), 16u);
                if (bytes != 0u) std::memcpy(data.host, binding.guestDescriptor.data(), bytes);
                entries[0] = {data.address, static_cast<std::uint32_t>(bytes), 0u};
                break;
            }
            case ShaderRecompiler::DescriptorRole::Gds: {
                if (binding.count != 1u || !binding.guestDescriptor.empty()) throw std::runtime_error("Metal device: invalid GDS descriptor contract");
                entries[0] = {Wrap(Pm4::GdsAddress(), Pm4::GdsBytes), static_cast<std::uint32_t>(Pm4::GdsBytes), 0u};
                break;
            }
            case ShaderRecompiler::DescriptorRole::BdaPagetable: {
                std::shared_ptr<const GuestAllocations::Lease> lease;
                const auto& registered = AddressTableBytes(&lease);
                recorder.Keep(std::const_pointer_cast<GuestAllocations::Lease>(lease));
                writes.insert(writes.end(), addressTable.writable.begin(), addressTable.writable.end());
                // As upstream's GuestBufferMemory::AddressRanges: the registered ranges plus every
                // bound V# range (Write when the shader stores to it), GDS, and the captured
                // snapshots (live in place when wrappable, else their captured bytes read-only).
                const auto extras = AddressExtras(shader, snapshots, stream);
                const auto covered = [&](const Bda::Range& extra) {
                    const auto& ranges = addressTable.ranges;
                    const auto found = std::upper_bound(ranges.begin(), ranges.end(), extra.begin, [](std::uint64_t value, const Bda::Range& range) { return value < range.begin; });
                    return found != ranges.begin() && extra.end <= std::prev(found)->end && (std::prev(found)->permissions & extra.permissions) == extra.permissions;
                };
                if (std::all_of(extras.begin(), extras.end(), covered)) {
                    const auto data = stream.Write(std::span<const std::byte>(registered), 16u);
                    entries[0] = {data.address, static_cast<std::uint32_t>(registered.size()), 0u};
                } else {
                    std::vector<Bda::Range> all = addressTable.ranges;
                    all.insert(all.end(), extras.begin(), extras.end());
                    const auto ranges = Metal::DisjointRanges(all);
                    const Bda::Header header {Bda::Version, static_cast<std::uint32_t>(ranges.size()), static_cast<std::uint32_t>(sizeof(Bda::Range)), 0u};
                    const auto bytes = sizeof(header) + ranges.size() * sizeof(Bda::Range);
                    if (bytes > 0xffffffffu) throw std::runtime_error("Metal device: the BDA page table exceeds 4 GiB");
                    const auto data = stream.Allocate(bytes, 16u);
                    std::memcpy(data.host, &header, sizeof(header));
                    if (!ranges.empty()) std::memcpy(data.host + sizeof(header), ranges.data(), ranges.size() * sizeof(Bda::Range));
                    entries[0] = {data.address, static_cast<std::uint32_t>(bytes), 0u};
                }
                break;
            }
            case ShaderRecompiler::DescriptorRole::FaultBuffer:
                entries[0] = {faults.gpuAddress, static_cast<std::uint32_t>(faults.length), 0u};
                break;
            case ShaderRecompiler::DescriptorRole::GuestImages:
            case ShaderRecompiler::DescriptorRole::GuestSamplers:
                break;
            }
        }
        for (const auto& [offset, values] : imageEntries) {
            if (!values.empty()) std::memcpy(table + offset, values.data(), values.size() * sizeof(std::uint64_t));
        }
    }
    const auto indirect = indirectArguments != 0u ? Wrap(indirectArguments, 12u) : 0u;
    const auto pushAddress = stream.Write(std::span<const std::byte>(push), 16u).address;
    std::vector<std::uint64_t> driverAddresses;
    std::vector<MTLSize> chunkGroups;
    const auto addChunk = [&](std::uint32_t xBase, std::uint32_t yBase, std::uint32_t zBase, MTLSize groups) {
        Abi::DriverData driver {};
        driver.workgroupBase = {xBase, yBase, zBase};
        driver.loopGuardLimit = LoopGuardLimit();
        driver.faultBuffer = faults.gpuAddress;
        driver.atomicLocks = atomicLocks.gpuAddress;
        driverAddresses.push_back(stream.Write(std::as_bytes(std::span(&driver, 1)), 16u).address);
        chunkGroups.push_back(groups);
    };
    if (split) {
        for (std::uint64_t zBase = 0; zBase < z; zBase += GroupLimit) {
            for (std::uint64_t yBase = 0; yBase < y; yBase += GroupLimit) {
                for (std::uint64_t xBase = 0; xBase < x; xBase += GroupLimit) {
                    addChunk(static_cast<std::uint32_t>(xBase), static_cast<std::uint32_t>(yBase), static_cast<std::uint32_t>(zBase), MTLSizeMake(std::min<std::uint64_t>(GroupLimit, x - xBase), std::min<std::uint64_t>(GroupLimit, y - yBase), std::min<std::uint64_t>(GroupLimit, z - zBase)));
                }
            }
        }
    } else {
        addChunk(0u, 0u, 0u, MTLSizeMake(x, y, z));
    }

    if ((covered & ShaderAccess) == ShaderAccess && Recorder::MergeBarriers()) {
        Recorder::CountMerged(Recorder::CommandClass::DispatchLeading);
    } else {
        stream.PipelineBarrier(AnyStage, indirectArguments != 0u ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::DispatchLeading);
    }
    auto encoder = stream.Compute();
    const auto threadsPerGroup = MTLSizeMake(threads[0], threads[1], threads[2]);
    for (std::size_t chunk = 0; chunk < driverAddresses.size(); ++chunk) {
        if (chunk != 0u) stream.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        // The stream's helper kernels bind their own table: set ours for every dispatch (bindings
        // are snapshotted per dispatch, so each chunk sees its own DriverData).
        [encoder setComputePipelineState:pipeline];
        [encoder setArgumentTable:this->arguments];
        [this->arguments setAddress:tableAddress != 0u ? tableAddress : driverAddresses.front() atIndex:Abi::Index(Abi::BufferIndex::DescriptorTable)];
        [this->arguments setAddress:pushAddress atIndex:Abi::Index(Abi::BufferIndex::PushData)];
        [this->arguments setAddress:arena != nil ? arena.gpuAddress : driverAddresses.front() atIndex:Abi::Index(Abi::BufferIndex::WorkgroupMemory)];
        [this->arguments setAddress:driverAddresses[chunk] atIndex:Abi::Index(Abi::BufferIndex::DriverData)];
        if (indirect != 0u) {
            [encoder dispatchThreadgroupsWithIndirectBuffer:indirect threadsPerThreadgroup:threadsPerGroup];
        } else {
            [encoder dispatchThreadgroups:chunkGroups[chunk] threadsPerThreadgroup:threadsPerGroup];
        }
        stream.CountCommand();
    }
    stream.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, AnyStage);
    Recorder::CountBarriers(Recorder::CommandClass::DispatchTrailing);
    recorder.MarkCovered(DispatchedAccess);
    if (!imageBindings.Empty()) imageBindings.Commit(recorder);
    if (indirectArguments != 0u) recorder.NotePendingRead(indirectArguments, 12u, Recorder::ReadKind::Indirect);
    if (!reads.empty()) recorder.NotePendingReads(reads, Recorder::ReadKind::DispatchElement);
    if (!writes.empty()) recorder.NotePendingWrites(writes);
    // Faults and written pages of this batch's dispatches (one shared record) when it completes.
    recorder.OnComplete([this] {
        try {
            CheckFaults();
        } catch (const std::exception& error) {
            if (tearingDown) {
                std::fprintf(stderr, "[metal] %s\n", error.what());
                return;
            }
            std::lock_guard lock(faultMutex);
            if (pendingFault == nullptr) pendingFault = std::current_exception();
        }
    });
    if (Trace()) std::fprintf(stderr, "[metal] dispatch %ux%ux%u%s (%zu chunks, arena %s, %zu reads, %zu writes; stream: %llu encoders, %llu barriers, %llu commands; covered 0x%x)\n", x, y, z, indirect != 0u ? " indirect" : "", driverAddresses.size(), arena != nil ? "yes" : "no", reads.size(), writes.size(), static_cast<unsigned long long>(stream.Encoders()), static_cast<unsigned long long>(stream.Barriers()), static_cast<unsigned long long>(stream.CommandCount()), covered);
}

void VulkanDevice::Dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>* recipe) {
    if (recipe != nullptr) *recipe = nullptr;
    state->RequireDispatchable(shader);
    if (x == 0u || y == 0u || z == 0u) return;
    @autoreleasepool {
        state->RecordCompute(shader, x, y, z, 0u, snapshots);
    }
}

// DISPATCH_INDIRECT as upstream decides it (decideIndirect): the GPU reads the three group counts
// from the wrapped guest memory behind the recorded work (cpuReason 0), so a dispatch or label
// writing them earlier is seen; otherwise the counts are read on the CPU (GuestMemory::Read, whose
// flush hook waits for the batch that writes them) and dispatched directly: 8 the shader uses the
// device-memory LDS arena, sized from the counts; 2 a label is pending over the counts; 3 they
// cannot be wrapped (unmapped, executable or misaligned memory).
VulkanDevice::IndirectOutcome VulkanDevice::DispatchIndirect(const ShaderRecompiler::RecompileResult& shader, std::uint64_t arguments, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>* recipe) {
    if (recipe != nullptr) *recipe = nullptr;
    state->RequireDispatchable(shader);
    IndirectOutcome outcome {0, 0.0};
    auto& recorder = *state->recorder;
    @autoreleasepool {
        const auto labelPending = [&] {
            for (std::uint64_t dword = arguments; dword < arguments + 12u; dword += 4u) {
                if (recorder.PendingLabel(dword, 4, 0).has_value()) return true;
            }
            return false;
        };
        if (WorkgroupStride(shader) != 0u || shader.workgroupMemoryDwords != 0u) {
            outcome.cpuReason = 8;
        } else if (labelPending()) {
            outcome.cpuReason = 2;
        } else if (arguments % 4u != 0u || state->TryWrap(arguments, 12u) == 0u) {
            outcome.cpuReason = 3;
        }
        if (outcome.cpuReason == 0) {
            state->RecordCompute(shader, 1u, 1u, 1u, arguments, snapshots);
            return outcome;
        }
        const auto start = std::chrono::steady_clock::now();
        std::array<std::uint32_t, 3> groups {};
        GuestMemory::Read(arguments, std::as_writable_bytes(std::span(groups)), 4u);
        outcome.argumentReadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (groups[0] != 0u && groups[1] != 0u && groups[2] != 0u) state->RecordCompute(shader, groups[0], groups[1], groups[2], 0u, snapshots);
    }
    return outcome;
}

std::shared_ptr<RecipeHit> VulkanDevice::PrepareRecipe(const std::shared_ptr<const Recipe>&, bool) {
    return nullptr;
}

RecipeOutcome VulkanDevice::DispatchRecipe(const ShaderRecompiler::RecompileResult&, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t, std::uint64_t, const std::shared_ptr<RecipeHit>&, IndirectOutcome&, const std::shared_ptr<PreparedDispatch>&, bool) {
    Unimplemented("DispatchRecipe (dispatch recipes are off on Metal: DispatchRecipes() is false)", "work package D1");
}

bool VulkanDevice::DispatchRecipes() {
    return false;
}

bool VulkanDevice::VerifyRecipes() {
    return false;
}

bool VulkanDevice::TemplateDataRefresh() {
    return false;
}

void VulkanDevice::NoteRecipe(RecipeEvent, bool) {
}

void VulkanDevice::NoteRecipe(RecipeEvent, RecipeKind) {
}

Graphics::Context VulkanDevice::graphicsContext() const {
    if (state->recorder == nullptr) Unimplemented("the Graphics::Context of a compile-only device", "work package D1");
    return state->context;
}

}
