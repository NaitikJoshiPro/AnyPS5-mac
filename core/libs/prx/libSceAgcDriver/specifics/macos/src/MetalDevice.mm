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
#include "MetalMemoryBudget.h"
#include "MetalCommandStream.h"
#include "MetalPipelineCache.h"
#include "MetalImages.h"
#include "MetalNoCopy.h"
#include "MetalVmTrace.h"
#include "BdaAbi.hpp"
#include "MslAbi.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "MetalResidency.h"

#include <spirv/unified1/spirv.hpp>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/mach_time.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
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

// The host-import join window (SetHostImportWindow): 8 MiB as upstream's Vulkan device, or
// APS5_HOST_IMPORT_WINDOW_MIB (0 = unbounded spans, the behaviour before D44).
std::uint64_t HostImportWindowBytes() {
    const char* value = std::getenv("APS5_HOST_IMPORT_WINDOW_MIB");
    return (value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : 8ull) << 20u;
}
constexpr std::uint64_t ArenaGranule = 1u << 20u;
// Workgroups per grid dimension the device reports (WorkGroupCountLimit). Metal ran 2^24
// threadgroups in x and 2^20 in y and z on the M3 (no limit is documented); dispatches beyond this
// one are split into chunks whose first workgroup reaches the shader in DriverData.workgroupBase,
// as upstream splits with vkCmdDispatchBase.
constexpr std::uint32_t GroupLimit = 65535u;
constexpr std::uint64_t MaxSplitChunks = 1u << 16u;
// The largest read-only range copied into upload memory when it cannot be wrapped (WrapOrCopy).
constexpr std::uint64_t MaxCopiedBytes = std::uint64_t{64} << 20u;

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

// The guest memory the BDA page table may add through new page wrappers (D52): wrappers are no-copy
// buffers Metal wires in full when a command buffer uses them, outside the live-import budget.
std::uint64_t BdaWrapperBudget() {
    static const std::uint64_t bytes = [] {
        if (const char* value = std::getenv("APS5_BDA_WRAP_MIB"); value != nullptr && *value != '\0') return std::strtoull(value, nullptr, 10) << 20u;
        return std::uint64_t{1024} << 20u;
    }();
    return bytes;
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

// Whether every page of guest [pageBegin, pageEnd) is mapped readable (executable pages included). A displaced range
// (GuestDisplaced.hpp, D31/D33) is queried at its alias; its guest address is a guard.
bool Readable(std::uint64_t pageBegin, std::uint64_t pageEnd) {
    mach_vm_address_t cursor = GuestArena::HostAddress(pageBegin);
    pageEnd = cursor + (pageEnd - pageBegin);
    while (cursor < pageEnd) {
        mach_vm_address_t region = cursor;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info {};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || region > cursor) return false;
        if ((info.protection & VM_PROT_READ) == 0) return false;
        cursor = region + size;
    }
    return true;
}

// Why [pageBegin, pageEnd) cannot be wrapped in place, empty when it can: every page must be
// mapped, readable and not executable (code-signed pages fault when wrapped).
std::string Unwrappable(std::uint64_t pageBegin, std::uint64_t pageEnd) {
    mach_vm_address_t cursor = GuestArena::HostAddress(pageBegin);
    pageEnd = cursor + (pageEnd - pageBegin);
    while (cursor < pageEnd) {
        mach_vm_address_t region = cursor;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info {};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || region > cursor) {
            return "guest memory at " + Hex(GuestArena::GuestAddress(cursor)) + " is not mapped";
        }
        if ((info.protection & VM_PROT_EXECUTE) != 0 || (info.protection & VM_PROT_READ) == 0) {
            return "guest memory at " + Hex(GuestArena::GuestAddress(cursor)) + " is executable or unreadable and cannot be wrapped (ranges a command only reads are copied instead)";
        }
        cursor = region + size;
    }
    return {};
}

// The Metal device: the system default, or with ANYPS5_GPU set the first device whose name
// contains it (case-insensitive), as upstream picks a Vulkan device.
id<MTLDevice> SelectDevice() {
    const char* requested = std::getenv("ANYPS5_GPU");
    if (requested == nullptr || *requested == '\0') return MTLCreateSystemDefaultDevice();
    const auto lower = [](std::string text) {
        for (auto& character : text) character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
        return text;
    };
    const auto request = lower(requested);
    std::string names;
    for (id<MTLDevice> candidate in MTLCopyAllDevices()) {
        const std::string name = candidate.name.UTF8String;
        if (!names.empty()) names += ", ";
        names += name;
        if (lower(name).find(request) != std::string::npos) return candidate;
    }
    throw std::runtime_error(std::string("Metal device: no usable device whose name contains ANYPS5_GPU=\"") + requested + "\"; devices: " + names);
}

// One Metal 4 command queue per MTLDevice, shared by every VulkanDevice made on it. The driver
// replaces its windowless device with a windowed one when VideoOut attaches a window
// (Driver::AttachWindow), and state keyed by the MTLDevice outlives the replacement: the host
// imports (GuestBufferMemory.mm) and the objects the first device made attach their residency sets
// to the queue that was current when they were made. With a queue per device, the replacement's
// command buffers ran without those sets: its label stores and every other access through such an
// import went to non-resident memory (reads 0, writes lost; PPSA99005's completion markers never
// reached the CPU). Upstream's VkDevice differs per VulkanDevice, which resets that state instead.
// Never destroyed: the exit handler tears down devices still alive at exit() after static destructors ran.
id<MTL4CommandQueue> SharedQueue(id<MTLDevice> device) {
    struct Queues {
        std::mutex mutex;
        std::vector<std::pair<id<MTLDevice>, id<MTL4CommandQueue>>> queues;
    };
    static Queues* const shared = new Queues;
    std::lock_guard lock(shared->mutex);
    for (const auto& [owner, queue] : shared->queues) {
        if (owner == device) return queue;
    }
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    if (queue != nil) shared->queues.emplace_back(device, queue);
    return queue;
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
    // Newest first: a wrapper made over overlapping ones covers all of them and serves their
    // ranges from then on, so overlapping V#s of a dispatch share one GPU mapping.
    for (auto wrapper = wrappers.rbegin(); wrapper != wrappers.rend(); ++wrapper) {
        if (address >= wrapper->begin && end <= wrapper->end) return wrapper->buffer.gpuAddress + (address - wrapper->begin);
    }
    auto pageBegin = address & ~(HostPage - 1u);
    auto pageEnd = (end + HostPage - 1u) & ~(HostPage - 1u);
    if (!Unwrappable(pageBegin, pageEnd).empty()) return 0;
    for (bool grown = true; grown;) {
        grown = false;
        for (const auto& wrapper : wrappers) {
            if (wrapper.begin < pageEnd && pageBegin < wrapper.end && (wrapper.begin < pageBegin || wrapper.end > pageEnd)) {
                pageBegin = std::min(pageBegin, wrapper.begin);
                pageEnd = std::max(pageEnd, wrapper.end);
                grown = true;
            }
        }
    }
    id<MTLBuffer> buffer = Metal::NewNoCopyBuffer(device, pageBegin, pageEnd - pageBegin, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked);
    if (buffer == nil) return 0;
    ::AgcDriver::Metal::Residency::Add(residency, buffer);
    ::AgcDriver::Metal::Residency::Commit(residency);
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

// Guest memory a command only reads that cannot be wrapped in place (executable pages: constant
// data in __TEXT, the guest's code) is copied into the stream's upload memory when the command is
// recorded (D3's copy fallback; such pages do not change). Written ranges must be wrapped: a copy
// would lose the GPU's stores, so they throw as Wrap does.
std::uint64_t VulkanDevice::State::WrapOrCopy(std::uint64_t address, std::uint64_t bytes, Graphics::Metal::CommandStream& stream, bool written) {
    if (const auto gpu = TryWrap(address, bytes); gpu != 0u) return gpu;
    const auto pageBegin = address & ~(HostPage - 1u);
    const auto pageEnd = (address + std::max<std::uint64_t>(bytes, 1u) + HostPage - 1u) & ~(HostPage - 1u);
    if (written || bytes > MaxCopiedBytes || !Readable(pageBegin, pageEnd)) {
        const auto reason = Unwrappable(pageBegin, pageEnd);
        throw std::runtime_error("Metal device: " + (reason.empty() ? "newBufferWithBytesNoCopy refused guest memory at " + Hex(pageBegin) : reason) + (written ? " (the command writes it: a copy would lose the stores)" : bytes > MaxCopiedBytes ? " (too large to copy: " + std::to_string(bytes) + " bytes)" : ""));
    }
    return stream.Write(std::span<const std::byte>(reinterpret_cast<const std::byte*>(GuestArena::HostAddress(address)), bytes), 16u).address;
}

id<MTLBuffer> VulkanDevice::State::WrapperOf(std::uint64_t address, std::uint64_t bytes, std::uint64_t* offset) const {
    const auto end = address + std::max<std::uint64_t>(bytes, 1u);
    for (auto wrapper = wrappers.rbegin(); wrapper != wrappers.rend(); ++wrapper) {
        if (address >= wrapper->begin && end <= wrapper->end) {
            *offset = address - wrapper->begin;
            return wrapper->buffer;
        }
    }
    return nil;
}

// Guest memory a command reads through a page wrapper (unregistered memory) stays pinned and
// resident while the batch is in flight: its pages count against the batch's kept-byte budget
// (Recorder::BoundKeptBytes), where upstream counts the copies it makes of such memory.
void VulkanDevice::State::KeepWrapped(std::uint64_t address, std::uint64_t bytes) {
    std::uint64_t offset = 0;
    if (context.hostImportAlignment != 0u && ImportOf(address, bytes, &offset) != nil) return;
    if (id<MTLBuffer> wrapper = WrapperOf(address, bytes, &offset)) recorder->KeepBytes((__bridge const void*)wrapper, wrapper.length);
}

void VulkanDevice::State::DropWrappersIfIdle() {
    if (recorder == nullptr || !recorder->Idle() || wrappers.empty()) return;
    for (const auto& wrapper : wrappers) ::AgcDriver::Metal::Residency::Remove(residency, wrapper.buffer);
    ::AgcDriver::Metal::Residency::Commit(residency);
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
    if (workgroupArena != nil) ::AgcDriver::Metal::Residency::Remove(residency, workgroupArena);
    ::AgcDriver::Metal::Residency::Add(residency, arena);
    ::AgcDriver::Metal::Residency::Commit(residency);
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
        static const bool traceTable = std::getenv("APS5_TRACE_GPU") != nullptr;
        std::uint64_t importedBytes = 0;
        std::uint64_t wrappedBytes = 0;
        std::uint64_t refusedBytes = 0;
        // Every page wrapper the device holds counts (the V# wrappers of earlier dispatches too):
        // they live until the recorder is idle (DropWrappersIfIdle).
        std::uint64_t heldWrapperBytes = 0;
        for (const auto& wrapper : wrappers) heldWrapperBytes += wrapper.end - wrapper.begin;
        const auto add = [&](const GuestAllocations::Range& range, std::uint64_t begin, std::uint64_t end, std::uint64_t gpu) {
            const auto permissions = Bda::Read | (range.writable ? Bda::Write : 0u);
            if ((permissions & Bda::Write) != 0u) {
                if (end > Bda::WrittenAddressLimit) throw std::runtime_error("Metal device: a writable BDA range ends past the addresses the written-page record can hold");
                writable.push_back({begin, end});
            }
            ranges.push_back({begin, end, gpu, permissions, 0u});
        };
        for (const auto& range : *acquired) {
            const auto end = range->address + range->bytes;
            if (context.hostImportAlignment != 0u && range->bytes > Graphics::HostImportPieceBytes) {
                // A long allocation is served piece by piece through host imports (D42 windows,
                // within the live-import budget), never by one page wrapper: Metal wires a no-copy
                // buffer in full when a command buffer uses it, and a table holding ASTRO BOT's
                // 3.1 + 4.6 GiB direct memory as wrappers beside the imports made the GPU fail the
                // first BDA dispatch with kIOGPUCommandBufferCallbackErrorOutOfMemory (reported as
                // MTL4CommandQueueErrorTimeout, D52). A piece the budget refuses is left out of the
                // table: an access there records an Unmapped fault instead.
                for (auto begin = range->address; begin < end;) {
                    const auto pieceEnd = std::min(end, (begin / Graphics::HostImportPieceBytes + 1u) * Graphics::HostImportPieceBytes);
                    std::uint64_t offset = 0;
                    if (id<MTLBuffer> import = ImportOf(begin, pieceEnd - begin, &offset)) {
                        add(*range, begin, pieceEnd, import.gpuAddress + offset);
                        importedBytes += pieceEnd - begin;
                    } else {
                        refusedBytes += pieceEnd - begin;
                    }
                    begin = pieceEnd;
                }
                continue;
            }
            std::uint64_t offset = 0;
            if (context.hostImportAlignment != 0u) {
                if (id<MTLBuffer> import = ImportOf(range->address, range->bytes, &offset)) {
                    add(*range, range->address, end, import.gpuAddress + offset);
                    importedBytes += range->bytes;
                    continue;
                }
            }
            // Not imported (the budget, or memory imports cannot hold): a page wrapper, which the
            // import budget does not see, so the wrappers the table adds are bounded on their own
            // (BdaWrapperBudget, over all the device's wrappers); one that already holds the range
            // costs nothing more.
            if (WrapperOf(range->address, range->bytes, &offset) == nil) {
                if (heldWrapperBytes + range->bytes > BdaWrapperBudget()) {
                    refusedBytes += range->bytes;
                    continue;
                }
                wrappedBytes += range->bytes;
                heldWrapperBytes += range->bytes;
            }
            const auto gpu = TryWrap(range->address, range->bytes);
            if (gpu == 0) {
                ++skipped;
                continue;
            }
            add(*range, range->address, end, gpu);
        }
        if (refusedBytes != 0u && addressTable.refusedBytes != refusedBytes) {
            std::fprintf(stderr, "[metal] BDA page table: %.0f MiB of guest allocations left out (live-import budget APS5_HOST_IMPORT_MIB, page-wrapper budget APS5_BDA_WRAP_MIB; an access there records an Unmapped fault)\n", refusedBytes / 1048576.0);
        }
        if (skipped != 0u && addressTable.skipped != skipped) {
            std::fprintf(stderr, "[metal] BDA page table: %zu of %zu guest allocations cannot be wrapped in place and are left out (mirrors: work package D3)\n", skipped, acquired->size());
        }
        if (traceTable) {
            std::uint64_t wrapperBytes = 0;
            std::uint64_t largest = 0;
            for (const auto& wrapper : wrappers) {
                wrapperBytes += wrapper.end - wrapper.begin;
                largest = std::max(largest, wrapper.end - wrapper.begin);
            }
            std::fprintf(stderr, "[gpu] BDA page table: %zu ranges, %.1f MiB of them in host imports, %.1f MiB in new page wrappers, %.1f MiB left out; %zu page wrappers hold %.1f MiB (largest %.1f MiB)\n", ranges.size(), importedBytes / 1048576.0, wrappedBytes / 1048576.0, refusedBytes / 1048576.0, wrappers.size(), wrapperBytes / 1048576.0, largest / 1048576.0);
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
        addressTable.refusedBytes = refusedBytes;
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
        // As upstream: the dispatch's result is kept and the guard is reported, not raised.
        std::fprintf(stderr, "[gpu] loop guard: the loop exit at pc 0x%x of shader 0x%llx ran past %u evaluations\n", report.instruction, static_cast<unsigned long long>(report.address), report.bytes);
        return;
    }
    std::string message = "Metal device: GPU fault: BDA access failed: address=" + Hex(report.address) + " instruction=" + Hex(report.instruction) + " bytes=" + std::to_string(report.bytes) + " stage=" + std::to_string(report.stage) + " reason=" + std::to_string(static_cast<std::uint32_t>(report.reason));
    lastFaultStage = report.stage;
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
            for (const auto& wrapper : wrappers) ::AgcDriver::Metal::Residency::Remove(residency, wrapper.buffer);
        }
        wrappers.clear();
        ++wrapperEpoch;
        addressTable = {};
        // 4. The device's own residency set leaves the queue before it is released.
        if (queue != nil && residency != nil) {
            ::AgcDriver::Metal::Residency::Detach(queue, residency);
            ::AgcDriver::Metal::Residency::RemoveAll(residency);
            ::AgcDriver::Metal::Residency::Commit(residency);
            report.residencyRemoved = true;
        }
        // The image cache's residency set leaves the queue (D4/D5, MetalImages.h).
        // D6's draw objects (depth surfaces, tiled targets) release their textures from the
        // image cache's residency set before the cache goes.
        draws.reset();
        presenter.reset();
        if (images != nullptr) images->Teardown();
        images.reset();
        // Allocations removed from residency sets since the last queue commit (MetalResidency.h):
        // the queue is idle and its sets are gone, so they go now.
        ::AgcDriver::Metal::Residency::Drain();
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
        @autoreleasepool {
            state->device = SelectDevice();
            if (state->device == nil) throw std::runtime_error("Metal device: no Metal device");
            if (![state->device supportsFamily:MTLGPUFamilyMetal4] || ![state->device supportsFamily:MTLGPUFamilyApple7]) throw std::runtime_error("Metal device: the GPU lacks the Metal 4 / Apple7 feature set this driver needs");
            NSError* error = nil;
            state->pipelines = std::make_unique<Graphics::Metal::PipelineCache>(state->device);
            state->compiler = state->pipelines->Compiler();
            if (state->compiler == nil) throw std::runtime_error("Metal device: no shader compiler");
            // ImageGatherExtended + nonConstantImageOffsets (Target()): MSL's sample/gather take a
            // run-time int2 offset (MslImage.cpp PackedOffset), as Vulkan's maintenance8 does.
            for (const auto capability : {spv::CapabilityShader, spv::CapabilityInt64, spv::CapabilityInt64Atomics, spv::CapabilityInt64ImageEXT, spv::CapabilityStencilExportEXT, spv::CapabilityFragmentShaderPixelInterlockEXT, spv::CapabilityInt16, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformArithmetic, spv::CapabilitySignedZeroInfNanPreserve, spv::CapabilityImageQuery, spv::CapabilityMinLod, spv::CapabilityImageGatherExtended}) {
                state->capabilities.push_back(static_cast<std::uint32_t>(capability));
            }
            // Multisampled storage images are emulated as 2D arrays of layers * samples slices (DECISIONS D22, MslAbi v6).
            state->capabilities.push_back(static_cast<std::uint32_t>(spv::CapabilityStorageImageMultisample));
            // s_memtime / s_memrealtime read an emulated clock: a device-wide count of clock reads after
            // the atomic lock words (MslAbi::ClockCounterWord, MslFlowEmitter.cpp EmitClock; CORPUS).
            state->capabilities.push_back(static_cast<std::uint32_t>(spv::CapabilityShaderClockKHR));
            // f64 is IEEE binary64 in software in the MSL backend (MslFloat64.cpp, package B2).
            state->capabilities.push_back(static_cast<std::uint32_t>(spv::CapabilityFloat64));
            // half arithmetic rounds to nearest even and keeps f16 denormals on Apple GPUs (probe_float_b2),
            // so packed f16 add/mul run natively (FPAdd16x2/FPMul16x2, MslAluMath.cpp).
            for (const auto capability : {spv::CapabilityFloat16, spv::CapabilityRoundingModeRTE, spv::CapabilityDenormPreserve}) state->capabilities.push_back(static_cast<std::uint32_t>(capability));
            // Vertex and mesh functions write [[render_target_array_index]] (layered draws, upstream #3763; the SPIR-V
            // header names the layer export through SPV_EXT_shader_viewport_index_layer, as upstream's devices do).
            state->capabilities.push_back(static_cast<std::uint32_t>(spv::CapabilityShaderViewportIndexLayerEXT));
            std::ranges::sort(state->capabilities);
            state->extensions = {"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
            if (compileOnly) return;
            state->queue = SharedQueue(state->device);
            MTLResidencySetDescriptor* residency = [MTLResidencySetDescriptor new];
            residency.initialCapacity = 64;
            state->residency = [state->device newResidencySetWithDescriptor:residency error:&error];
            if (state->queue == nil || state->residency == nil) throw std::runtime_error("Metal device: queue setup failed: " + Describe(error));
            state->faults = [state->device newBufferWithLength:Bda::FaultBufferBytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            if (state->faults == nil) throw std::runtime_error("Metal device: cannot allocate the fault buffer");
            std::memset(state->faults.contents, 0, state->faults.length);
            ::AgcDriver::Metal::Residency::Add(state->residency, state->faults);
            state->atomicLocks = [state->device newBufferWithLength:Abi::AtomicLockAllocationBytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            if (state->atomicLocks == nil) throw std::runtime_error("Metal device: cannot allocate the 64-bit atomic lock table");
            // Every lock word starts released; only the GPU touches the table after this.
            std::memset(state->atomicLocks.contents, 0, state->atomicLocks.length);
            ::AgcDriver::Metal::Residency::Add(state->residency, state->atomicLocks);
            ::AgcDriver::Metal::Residency::Commit(state->residency);
            ::AgcDriver::Metal::Residency::Attach(state->queue, state->residency);
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
            // Adjacent allocations join one import only over whole aligned 8 MiB windows, as on
            // upstream's Vulkan device (a94730373): a heap growing 16 KiB at a time otherwise re-imports
            // the whole chain at every growth, and the retired no-copy buffers stay in the import
            // residency set until their batch completes (SuperTuxKart: ~20 GiB of them in one batch,
            // which Metal refused at commit with MTL4CommandQueueErrorTimeout; DECISIONS D44).
            Graphics::SetHostImportWindow(HostImportWindowBytes());
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
            // Upstream #423 / #3763 device features (they gate SPIR-V validation only; Metal draws
            // never validate SPIR-V): MSL vertex functions write [[clip_distance]] and
            // [[render_target_array_index]]; cull distances are emulated in one more clip distance
            // (MslAbi v10, MetalDraw.mm: triangle-vertices draws, points; lines refused).
            context.shaderClipDistance = true;
            context.shaderCullDistance = true;
            context.viewportIndexLayer = true;
            // The device-local heap (the recommended working set) and its VK_EXT_memory_budget report, which size the
            // sampled and storage texture caches (D5) and shrink them under OS memory pressure (MetalMemoryBudget.h).
            Graphics::Metal::DescribeMemory(context, state->device);
            context.bufferPool = std::make_shared<Graphics::BufferPool>(context);
            state->recorder = std::make_unique<Graphics::Recorder>(context, true);
            context.recorder = state->recorder.get();
            state->recorder->Activate();
            Graphics::PrepareImportWatch(context);
            // The image cache's Graphics twins (Texture/ShaderResources, D4/D5) take the complete Context.
            state->images->Attach(state->context);
            // Presentation (D6): the window's Metal layer, or headless targets when it has none.
            if (window != nullptr) state->AttachPresentation(*window);
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

// Upstream reports VK_KHR_driver_properties' name and info: Metal has no driver version, so the
// OS build (the Metal framework ships with it) and the GPU family name it.
std::string VulkanDevice::DriverInfo() const {
    NSOperatingSystemVersion version = [NSProcessInfo processInfo].operatingSystemVersion;
    std::string info = "Metal (macOS " + std::to_string(version.majorVersion) + "." + std::to_string(version.minorVersion) + "." + std::to_string(version.patchVersion) + ")";
    if ([state->device supportsFamily:MTLGPUFamilyMetal4]) info += " Metal 4";
    return info;
}

// Upstream calibrates through VK_KHR_calibrated_timestamps (device + CLOCK_MONOTONIC). Metal samples
// a CPU timestamp (mach_absolute_time) and a GPU timestamp together; the CPU stamp's age gives the
// steady_clock point, the GPU stamp is converted like the recorder's counter-heap timestamps.
bool VulkanDevice::CalibrateClock(double& gpuNs, std::chrono::steady_clock::time_point& cpu) const {
    if (state->device == nil) return false;
    MTLTimestamp cpuStamp = 0;
    MTLTimestamp gpuStamp = 0;
    [state->device sampleTimestamps:&cpuStamp gpuTimestamp:&gpuStamp];
    const auto nowTicks = mach_absolute_time();
    const auto now = std::chrono::steady_clock::now();
    if (cpuStamp == 0u && gpuStamp == 0u) return false;
    mach_timebase_info_data_t timebase {};
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.denom == 0u) return false;
    const double nsPerTick = static_cast<double>(timebase.numer) / static_cast<double>(timebase.denom);
    // The GPU stamp in the unit the recorder's counter-heap ranges use (Recorder.mm: 1e9 /
    // queryTimestampFrequency per tick); PerformanceTrace rescales between two samples anyway.
    const auto frequency = [state->device queryTimestampFrequency];
    gpuNs = static_cast<double>(gpuStamp) * (frequency != 0u ? 1e9 / static_cast<double>(frequency) : nsPerTick);
    const double sinceNs = static_cast<double>(static_cast<std::int64_t>(nowTicks - cpuStamp)) * nsPerTick;
    cpu = now - std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double, std::nano>(sinceNs));
    return true;
}

ShaderRecompiler::SpirvTarget VulkanDevice::Target() const {
    const auto threadgroupMemory = static_cast<std::uint32_t>(state->device.maxThreadgroupMemoryLength);
    const auto threads = state->device.maxThreadsPerThreadgroup;
    // fragmentShaderBarycentricEnabled: per-vertex fragment values are emulated (MslAbi Version 5
    // triangle-vertices draws; Metal's own per-vertex values are Apple10-only). On the MSL backend
    // the recompiler still starts every pixel program with host interpolation and switches to the
    // emulation only for programs that need per-vertex values (Recompiler.cpp).
    ShaderRecompiler::SpirvTarget target {0u, 0u, 32u, Bda::Version, state->capabilities, state->extensions, true, {static_cast<std::uint32_t>(threads.width), static_cast<std::uint32_t>(threads.height), static_cast<std::uint32_t>(threads.depth)}, 1024u, threadgroupMemory, {}, {}};
    target.backend = ShaderRecompiler::ShaderBackend::Msl;
    // Formats whose sRGB decode the image path does in the texture view (D4, MetalImages.h).
    target.srgbDecodeFormats = Graphics::Metal::SrgbDecodeFormats();
    // MslAbi: pixel push constants start at byte 128 of the push block (slot 1), whatever the
    // vertex stage uses.
    target.fixedPushSlots = true;
    // MSL mesh functions (B6 stage 5, NGG primitive shaders): mesh<V, P, ...> holds at most 256
    // vertices and 512 primitives, a mesh threadgroup as many threads as a compute one, 124 scalar
    // varyings (Metal Feature Set Tables: mesh shading Apple7+, 1,048,575 mesh threadgroups per grid
    // on Apple9). Pipelines and draws: M2_DESIGN "SHD4 results".
    target.mesh = ShaderRecompiler::MeshTargetLimits{{static_cast<std::uint32_t>(threads.width), 1u, 1u}, static_cast<std::uint32_t>(threads.width), threadgroupMemory, 256u, 512u, 124u, threadgroupMemory, 1u, 1u};
    // Tessellation runs on object + mesh functions (DECISIONS D24, MetalTessellation.h): 32 control
    // points, hull outputs bounded by the 16 KiB object payload (LinkMslTessellation checks it).
    // The software tessellator draws every domain, partitioning and output topology and applies
    // the VGT_HOS level clamps (DECISIONS D26).
    target.tessellation = ShaderRecompiler::TessellationTargetLimits{32u, 128u, 128u, 120u, 4096u, 128u, 124u, ShaderRecompiler::TessellationFeatures::All()};
    // Upstream 0bcff53b2 runs vertex programs as one-lane waves on devices without vertex-stage
    // subgroups: on Metal the startup self-test decides (as SubgroupStages); a compile-only device
    // has no profile and keeps 32-lane vertex programs (DECISIONS D25).
    target.vertexSubgroups = state->queue == nil || Metal::Profile(state->device).vertexSimdGroups;
    // Upstream e4f212b2e: float16DenormPreserve converts f16 operands in hardware (MSL
    // aps5_f16_to_f32_native). Apple GPUs keep f16 denormals in the conversion: agc_metal_alu_helpers'
    // "f16->f32 (native)" op matched the exact decode for 20000 halves, a quarter of them denormal
    // (gpu-suite 20261013T140350Z).
    target.float16DenormPreserve = true;
    // Texel offsets computed at run time (ASTRO BOT: "normalized guest sampler is used with a texel
    // offset that is not a constant"): MSL sample/gather/sample_compare take a run-time int2 offset,
    // the equivalent of Vulkan's maintenance8 + shaderImageGatherExtended. Metal is reported to clamp offsets to
    // [-8, 7] where RDNA2 takes 6-bit fields ([-32, 31]); TechnicalDebt metal-texel-offset-range.
    target.nonConstantImageOffsets = true;
    // DECISIONS D54: Apple GPUs flush f32 denormal operands and results of add, subtract, multiply
    // and compares to signed zeros, as RDNA's FLOAT_MODE f32 flush does (agc_metal_native_f32_flush
    // compares both translations bit for bit), so the front end drops its explicit flushes there.
    // APS5_MSL_NATIVE_F32_FLUSH=0 keeps them (debug aid).
    static const bool nativeF32Flush = [] {
        const char* value = std::getenv("APS5_MSL_NATIVE_F32_FLUSH");
        return value == nullptr || value[0] != '0';
    }();
    target.float32DenormFlushNative = nativeF32Flush;
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

namespace {

// Debug aid (APS5_TRACE_GDS=1): GDS dwords 4-7 (the append/consume counters of the GDS tests).
void TraceGds(const char* where) {
    static const bool enabled = std::getenv("APS5_TRACE_GDS") != nullptr;
    if (!enabled) return;
    const auto* words = reinterpret_cast<const std::uint32_t*>(Pm4::GdsAddress());
    std::fprintf(stderr, "[gds] %s: %u %u %u %u\n", where, words[4], words[5], words[6], words[7]);
}

}

void VulkanDevice::WaitIdle() {
    state->Sync();
    std::exception_ptr fault;
    {
        std::lock_guard lock(state->drawFaultMutex);
        fault = std::exchange(state->drawFault, nullptr);
    }
    if (fault != nullptr) std::rethrow_exception(fault);
    TraceGds("after WaitIdle");
    if (std::getenv("APS5_TRACE_GDS") != nullptr) {
        const auto gds = Pm4::GdsAddress();
        std::uint64_t offset = 0;
        id<MTLBuffer> import = state->ImportOf(gds, Pm4::GdsBytes, &offset);
        id<MTLBuffer> wrapper = state->WrapperOf(gds, Pm4::GdsBytes, &offset);
        for (id<MTLBuffer> buffer : {import, wrapper}) {
            if (buffer == nil) continue;
            const auto* contents = static_cast<const std::uint8_t*>(buffer.contents);
            std::fprintf(stderr, "[gds] %s contents=%p guest=0x%llx gpu=0x%llx dword4=%u\n", buffer == import ? "import" : "wrapper", contents, static_cast<unsigned long long>(gds), static_cast<unsigned long long>(buffer.gpuAddress), *reinterpret_cast<const std::uint32_t*>(contents + offset + 0x10u));
        }
    }
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

void VulkanDevice::StoreIdleResults(bool reapFirst) {
    // Pending storage-image results of the upstream image path (D4's Texture twin), as upstream
    // (VulkanDevice.cpp): the write-back is recorded into the open batch, which is submitted here.
    if (!Graphics::StorageTexture::AnyPending()) return;
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    Graphics::StorageTexture::FlushAllPending("idle");
    SubmitRecorded(reapFirst);
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
        // Pending resident-image results over the range land before the fill overwrites it.
        Graphics::StorageTexture::FlushPending(address, bytes, nullptr, "fill destination", Graphics::PublishScope::PartialUnits);
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
    // Upstream 396b5189a (#4496): only a label a completion will store counts here; a label stored by
    // the GPU stream is ordered by the copy's barrier.
    const auto completionStoreOverlaps = [&](std::uint64_t address) { return Recorder::PendingCompletionLabels() != 0 && recorder.CompletionLabelIn(address, bytes); };
    bool settledBySignal = false;
    // Pending resident-image results over either range are stored first (upstream copies between
    // the storage images instead, CopyOutcome path 3; not ported: TechnicalDebt metal-d6-bring-up-draws).
    Graphics::StorageTexture::FlushPending(source, bytes, nullptr, "buffer copy source", Graphics::PublishScope::Whole);
    Graphics::StorageTexture::FlushPending(destination, bytes, nullptr, "buffer copy destination", Graphics::PublishScope::PartialUnits);
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
        GuestMemory::StoreOwnBytes(destination, bytes, [&] { std::memmove(reinterpret_cast<void*>(GuestArena::HostAddress(destination)), reinterpret_cast<const void*>(GuestArena::HostAddress(source)), bytes); });
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
    // As upstream: an incompatible runtime ABI is refused before anything is prepared.
    ShaderRecompiler::RuntimeAbi::RequireVersion(shader.runtimeAbiVersion);
    // The pipeline starts building on the cache's workers; Dispatch waits for it.
    if (state->pipelines != nullptr && state->recorder != nullptr && !shader.msl.empty() && shader.msl.Reflection().stage == Abi::Stage::Kernel) state->pipelines->PrepareCompute(shader);
    return nullptr;
}

std::span<const double, 5> VulkanDevice::PreparePhaseMs(const PreparedDispatch&) {
    static constexpr std::array<double, 5> zero {};
    return zero;
}

void VulkanDevice::State::RequireDispatchable(const ShaderRecompiler::RecompileResult& shader) const {
    // As upstream, an incompatible runtime ABI is the first refusal.
    ShaderRecompiler::RuntimeAbi::RequireVersion(shader.runtimeAbiVersion);
    if (recorder == nullptr) throw std::runtime_error("Metal device: a compile-only device cannot dispatch");
    if (shader.msl.empty()) throw std::runtime_error("Metal device: the shader has no MSL form (recompile it for the device's Target(), whose backend is Msl)");
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
                extras.push_back({aligned, aligned + bytes, WrapOrCopy(aligned, bytes, stream, written), Bda::Read | (written ? Bda::Write : 0u), 0u});
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

void VulkanDevice::State::PlanBindings(const ShaderRecompiler::RecompileResult& shader, StageBindings& plan) {
    for (const auto& binding : shader.bindings) {
        if (binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
            const auto elements = std::min<std::size_t>(binding.count, binding.guestDescriptor.size() / 4u);
            for (std::uint32_t element = 0; element < elements; ++element) {
                const auto guest = DecodeBuffer(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 4u, 4u));
                if (guest.base == 0u || guest.bytes == 0u) continue;
                const auto aligned = guest.base & ~std::uint64_t{3u};
                const auto end = aligned + ((guest.base - aligned + guest.bytes + 3u) & ~std::uint64_t{3u});
                plan.reads.push_back({aligned, end});
                // An element the recompiler did not classify counts as written.
                if (element >= binding.bufferWritten.size() || binding.bufferWritten[element]) plan.writes.push_back({aligned, end});
            }
        } else if (binding.role == ShaderRecompiler::DescriptorRole::Gds) {
            const auto gds = Pm4::GdsAddress();
            plan.reads.push_back({gds, gds + Pm4::GdsBytes});
            plan.writes.push_back({gds, gds + Pm4::GdsBytes});
        } else if (binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable) {
            plan.addressBased = true;
        }
    }
}

void VulkanDevice::State::PlanImages(const ShaderRecompiler::RecompileResult& shader, Graphics::Metal::ImageBindings& imageBindings, StageBindings& plan) {
    const auto& reflection = shader.msl.Reflection();
    // Pairs samplers with images (ANISO_OVERRIDE, FORCE_DEGAMMA) and checks bindless metadata.
    imageBindings.Prepare(shader);
    if (reflection.table.bytes != 0u) {
        for (const auto& binding : shader.bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages && binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers) continue;
            const auto kind = static_cast<ShaderRecompiler::RuntimeAbi::Binding>(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count));
            const auto* range = reflection.table.Find(kind);
            if (range == nullptr) continue;
            std::vector<std::uint64_t> entries(std::min(binding.count, range->count), 0u);
            imageBindings.Write(binding, entries);
            plan.imageEntries.push_back({range->offset, std::move(entries)});
        }
    }
}

void VulkanDevice::State::SettleBindings(std::span<const StageBindings* const> plans) {
    auto& recorder = *this->recorder;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    bool addressBased = false;
    for (const auto* plan : plans) {
        ranges.insert(ranges.end(), plan->reads.begin(), plan->reads.end());
        ranges.insert(ranges.end(), plan->writes.begin(), plan->writes.end());
        addressBased = addressBased || plan->addressBased;
    }
    const auto touches = [&](std::uint64_t begin, std::uint64_t end) {
        return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.first < end && begin < range.second; });
    };
    if (recorder.HasQueuedKeyStores() && (addressBased || recorder.AnyQueuedKeyStore(touches))) recorder.FlushKeyStores();
    if (recorder.HasQueuedStores() && (addressBased || recorder.AnyQueuedStore(touches))) recorder.FlushStores();
    // Resident images (render targets, storage images of the upstream image path) whose results are
    // still on the GPU store them into guest memory before a command reads or writes that memory as
    // a buffer, as upstream's buffer bindings do.
    if (!ranges.empty() && Graphics::StorageTexture::AnyPendingOverlaps(ranges)) {
        for (const auto& [begin, end] : ranges) Graphics::StorageTexture::FlushPending(begin, static_cast<std::size_t>(end - begin), nullptr, "buffer binding");
    }
}

void VulkanDevice::State::WrapUnions(std::span<const StageBindings* const> plans) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto* plan : plans) ranges.insert(ranges.end(), plan->reads.begin(), plan->reads.end());
    // Overlapping guest ranges (V#s over one array, GDS) are wrapped as their union first, so every
    // V# binds as an offset into one mapping (upstream merges overlapping descriptor regions into
    // one region the same way).
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t index = 0; index < ranges.size();) {
        const auto begin = ranges[index].first;
        auto end = ranges[index].second;
        std::size_t count = 1;
        for (++index; index < ranges.size() && ranges[index].first < end; ++index, ++count) end = std::max(end, ranges[index].second);
        if (count > 1u) TryWrap(begin, end - begin);
    }
}

std::uint64_t VulkanDevice::State::WriteTable(const ShaderRecompiler::RecompileResult& shader, std::span<const Graphics::GuestMemorySnapshot> snapshots, Graphics::Metal::CommandStream& stream, StageBindings& plan, std::span<std::byte> push) {
    const auto& reflection = shader.msl.Reflection();
    if (reflection.table.bytes == 0u) return 0u;
    auto& recorder = *this->recorder;
    const auto upload = stream.Allocate(reflection.table.bytes, 16u);
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
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                entries[element] = {WrapOrCopy(aligned, bytes, stream, written), static_cast<std::uint32_t>(bytes), 0u};
                KeepWrapped(aligned, bytes);
                const auto position = static_cast<std::size_t>(shader.memoryOffsetDword) * 4u + element;
                if (position < push.size()) push[position] = std::byte{0};
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
            if (Trace()) {
                std::uint64_t offset = 0;
                id<MTLBuffer> import = ImportOf(Pm4::GdsAddress(), Pm4::GdsBytes, &offset);
                id<MTLBuffer> wrapper = WrapperOf(Pm4::GdsAddress(), Pm4::GdsBytes, &offset);
                const auto* first = reinterpret_cast<const std::uint32_t*>(Pm4::GdsAddress());
                std::fprintf(stderr, "[metal] GDS pages: %s\n", Metal::VmRegionInfo(Pm4::GdsAddress()).c_str());
                std::fprintf(stderr, "[metal] GDS wiring: %s\n", Metal::VmWiredInfo(Pm4::GdsAddress()).c_str());
                std::fprintf(stderr, "[metal] GDS 0x%llx bound at 0x%llx (%s, buffer contents %p), dwords 4-7 now %u %u %u %u\n", static_cast<unsigned long long>(Pm4::GdsAddress()), static_cast<unsigned long long>(entries[0].address), import != nil ? "host import" : wrapper != nil ? "page wrapper" : "neither", import != nil ? import.contents : wrapper != nil ? wrapper.contents : nullptr, first[4], first[5], first[6], first[7]);
            }
            break;
        }
        case ShaderRecompiler::DescriptorRole::BdaPagetable: {
            std::shared_ptr<const GuestAllocations::Lease> lease;
            const auto& registered = AddressTableBytes(&lease);
            recorder.Keep(std::const_pointer_cast<GuestAllocations::Lease>(lease));
            plan.writes.insert(plan.writes.end(), addressTable.writable.begin(), addressTable.writable.end());
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
    for (const auto& [offset, values] : plan.imageEntries) {
        if (!values.empty()) std::memcpy(table + offset, values.data(), values.size() * sizeof(std::uint64_t));
    }
    return upload.address;
}

std::uint64_t VulkanDevice::State::WriteDriverData(Graphics::Metal::CommandStream& stream, std::array<std::uint32_t, 3> workgroupBase) {
    Abi::DriverData driver {};
    driver.workgroupBase = workgroupBase;
    driver.loopGuardLimit = LoopGuardLimit();
    driver.faultBuffer = faults.gpuAddress;
    driver.atomicLocks = atomicLocks.gpuAddress;
    return stream.Write(std::as_bytes(std::span(&driver, 1)), 16u).address;
}

void VulkanDevice::State::NoteBindings(const StageBindings& plan) {
    if (!plan.reads.empty()) recorder->NotePendingReads(plan.reads, Recorder::ReadKind::DispatchElement);
    if (plan.writes.empty()) return;
    recorder->NotePendingWrites(plan.writes);
    // The GPU's stores through in-place V# bindings, linear color targets and GDS are the driver's
    // own writes: stamped at record time like labels, fills and copies (readers sync on the pending
    // set until they land). Over write-watched memory (page wrappers, watched imports: DECISIONS D74)
    // the watch never sees them, and caches over the range would keep stale bytes after the batch
    // retires; over unwatched memory the stamp records an unstamped store for NewestStamp.
    for (const auto& [begin, end] : plan.writes) {
        if (end > begin) GuestMemory::MarkWritten(begin, static_cast<std::size_t>(end - begin));
    }
}

void VulkanDevice::State::CheckFaultsOnComplete(bool draw) {
    // Faults and written pages of this batch's commands (one shared record) when it completes. As
    // upstream (BdaResources::CheckFault in a completion): a fault record throws, which the recorder
    // reports on stderr ("[gpu] deferred write-back failed: ...") without failing the wait. A
    // draw's graphics-stage fault is also kept for WaitIdle: upstream's draw over non-resident
    // targets runs synchronously and throws it from Draw (agc_driver_pixel_interlock).
    recorder->OnComplete([this, draw] {
        if (!tearingDown) {
            try {
                CheckFaults();
            } catch (...) {
                constexpr std::uint32_t ComputeStage = 4u;  // IrShaderStage::Compute in Bda::Fault::stage
                if (draw && lastFaultStage != ComputeStage) {
                    std::lock_guard lock(drawFaultMutex);
                    if (drawFault == nullptr) drawFault = std::current_exception();
                }
                throw;
            }
            return;
        }
        try {
            CheckFaults();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[metal] %s\n", error.what());
        }
    });
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
    // A batch over its kept-byte budget is submitted first (upstream's dispatch does the same).
    recorder.BoundKeptBytes();
    // The arena is sized for the whole grid (an indirect dispatch with an arena reads its counts on
    // the CPU first, so the grid is direct here); growing it waits for idle, before the open batch
    // is taken below.
    id<MTLBuffer> arena = stride != 0u ? WorkgroupArena(static_cast<std::uint64_t>(x) * y * z, stride) : nil;
    id<MTLComputePipelineState> pipeline = pipelines->Compute(shader);
    const auto& threads = reflection.threadsPerThreadgroup;
    if (static_cast<NSUInteger>(threads[0]) * threads[1] * threads[2] > pipeline.maxTotalThreadsPerThreadgroup) {
        throw std::runtime_error("Metal device: the workgroup of " + std::to_string(threads[0] * threads[1] * threads[2]) + " threads exceeds the pipeline's maximum threadgroup size " + std::to_string(pipeline.maxTotalThreadsPerThreadgroup));
    }

    // Guest ranges and image/sampler entries first (queued label and key stores into the ranges
    // must precede the dispatch; uploading a surface reads guest memory on the CPU, and the settle
    // hook may sync the batch), before the open batch is taken.
    Graphics::Metal::ImageBindings imageBindings(*images, [this](std::uint64_t address, std::uint64_t bytes) {
        if (this->recorder->PendingWriteOverlaps(address, bytes) || this->recorder->PendingLabelIn(address, bytes)) {
            Recorder::CountSync(4);
            this->recorder->SyncThrough(address, bytes);
        }
    });
    StageBindings plan;
    PlanBindings(shader, plan);
    if (indirectArguments != 0u) plan.reads.push_back({indirectArguments, indirectArguments + 12u});
    const StageBindings* plans[] = {&plan};
    SettleBindings(plans);
    PlanImages(shader, imageBindings, plan);
    WrapUnions(plans);

    VkAccessFlags covered = 0;
    auto& stream = recorder.Stream(&covered);
    std::array<std::byte, Abi::PushDataBytes> push {};
    if (!shader.pushConstants.empty()) {
        if (shader.pushConstants.size() > push.size()) throw std::runtime_error("Metal device: push constants exceed the push data block");
        std::memcpy(push.data(), shader.pushConstants.data(), shader.pushConstants.size());
    }
    const std::uint64_t tableAddress = WriteTable(shader, snapshots, stream, plan, std::span<std::byte>(push).first(shader.pushConstants.size()));
    const auto indirect = indirectArguments != 0u ? Wrap(indirectArguments, 12u) : 0u;
    const auto pushAddress = stream.Write(std::span<const std::byte>(push), 16u).address;
    std::vector<std::uint64_t> driverAddresses;
    std::vector<MTLSize> chunkGroups;
    const auto addChunk = [&](std::uint32_t xBase, std::uint32_t yBase, std::uint32_t zBase, MTLSize groups) {
        driverAddresses.push_back(WriteDriverData(stream, {xBase, yBase, zBase}));
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
    if (indirectArguments != 0u) {
        recorder.NotePendingRead(indirectArguments, 12u, Recorder::ReadKind::Indirect);
        plan.reads.pop_back();
    }
    NoteBindings(plan);
    CheckFaultsOnComplete();
    if (Trace() && std::any_of(shader.bindings.begin(), shader.bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::Gds; })) {
        // After the dispatch completes: what the CPU sees in GDS dwords 4-7, and what the GPU sees
        // through the bound mapping (copied into a fresh buffer behind the dispatch).
        std::uint64_t offset = 0;
        id<MTLBuffer> mapping = ImportOf(Pm4::GdsAddress(), Pm4::GdsBytes, &offset);
        if (mapping == nil) mapping = WrapperOf(Pm4::GdsAddress(), Pm4::GdsBytes, &offset);
        id<MTLBuffer> probe = [device newBufferWithLength:32 options:MTLResourceStorageModeShared];
        // CPU -> GPU direction: a CPU store after the import exists, read back by the GPU copy.
        auto* last = reinterpret_cast<volatile std::uint32_t*>(Pm4::GdsAddress() + Pm4::GdsBytes - 4u);
        *last = 0xc0ffee00u;
        if (mapping != nil && probe != nil) {
            ::AgcDriver::Metal::Residency::Add(residency, probe);
            ::AgcDriver::Metal::Residency::Commit(residency);
            stream.Barrier();
            [stream.Compute() copyFromBuffer:mapping sourceOffset:offset + 16u toBuffer:probe destinationOffset:0 size:16];
            [stream.Compute() copyFromBuffer:mapping sourceOffset:offset + Pm4::GdsBytes - 4u toBuffer:probe destinationOffset:16 size:4];
            stream.Barrier();
        }
        if (std::getenv("APS5_METAL_GDS_WATCH") != nullptr) Metal::WatchWrites(Pm4::GdsAddress());
        recorder.OnComplete([probe] {
            const auto* first = reinterpret_cast<const std::uint32_t*>(Pm4::GdsAddress());
            std::fprintf(stderr, "[metal] GDS wiring after: %s\n", Metal::VmWiredInfo(Pm4::GdsAddress()).c_str());
            const auto* gpu = probe != nil ? static_cast<const std::uint32_t*>(probe.contents) : first;
            std::fprintf(stderr, "[metal] GDS pages after: %s\n", Metal::VmRegionInfo(Pm4::GdsAddress()).c_str());
            std::fprintf(stderr, "[metal] GDS after the batch: CPU dwords 4-7 %u %u %u %u, GPU copy %u %u %u %u\n", first[4], first[5], first[6], first[7], gpu[0], gpu[1], gpu[2], gpu[3]);
            if (probe != nil) std::fprintf(stderr, "[metal] GDS last dword: CPU stored 0xc0ffee00 after the import, GPU copy reads 0x%08x\n", gpu[4]);
        });
    }
    if (Trace()) std::fprintf(stderr, "[metal] dispatch %ux%ux%u%s (%zu chunks, arena %s, %zu reads, %zu writes; stream: %llu encoders, %llu barriers, %llu commands; covered 0x%x)\n", x, y, z, indirect != 0u ? " indirect" : "", driverAddresses.size(), arena != nil ? "yes" : "no", plan.reads.size(), plan.writes.size(), static_cast<unsigned long long>(stream.Encoders()), static_cast<unsigned long long>(stream.Barriers()), static_cast<unsigned long long>(stream.CommandCount()), covered);
}

void VulkanDevice::Dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::span<const Graphics::GuestMemorySnapshot> snapshots, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>* recipe) {
    if (recipe != nullptr) *recipe = nullptr;
    state->RequireDispatchable(shader);
    if (x == 0u || y == 0u || z == 0u) return;
    TraceGds("before Dispatch");
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
