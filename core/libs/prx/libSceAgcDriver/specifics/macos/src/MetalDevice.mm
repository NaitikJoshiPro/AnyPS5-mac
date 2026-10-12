// The AGC driver's device API (VulkanDevice.hpp, overlaid by
// specifics/macos/overlay/.../VulkanDevice.hpp) implemented on Metal 4 (work package D1,
// docs/mac/M2_DESIGN.md): device setup, the Metal shader target, compute dispatches (direct, split,
// indirect) of MetalBackend shaders over wrapped guest memory with every MslAbi binding the compute
// path has (guest buffers, GDS, BDA page table, fault buffer, flattened SRT, shader data, the
// device-memory LDS arena), GPU labels, fills and copies, completion actions and the
// synchronization the PM4 core needs.
// Command model: one MTL4 command queue, a MTLSharedEvent timeline, every command buffer ordered
// after all earlier work by a queue barrier (bring-up policy) and committed at once; D2's recorder
// replaces the immediate commits with batches.

#include "MetalDeviceState.h"
#include "BdaAbi.hpp"
#include "MslAbi.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
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
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

constexpr std::uint64_t HostPage = 16384u;
constexpr std::size_t RingBytes = 8u << 20u;
// Ring room a command needs at its start; less waits for idle and starts the ring over (a larger
// single allocation gets a buffer of its own).
constexpr std::size_t RingHeadroom = 1u << 20u;
constexpr std::uint64_t WaitTimeoutMs = 20000u;
constexpr std::uint64_t ArenaGranule = 1u << 20u;
// Workgroups per grid dimension the device reports (WorkGroupCountLimit). Metal ran 2^24
// threadgroups in x and 2^20 in y and z on the M3 (no limit is documented); dispatches beyond this
// one are split into chunks whose first workgroup reaches the shader in DriverData.workgroupBase,
// as upstream splits with vkCmdDispatchBase.
constexpr std::uint32_t GroupLimit = 65535u;
constexpr std::uint64_t MaxSplitChunks = 1u << 16u;

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

// The MSL version MetalBackend emits (MslAbi::LanguageMajor/LanguageMinor).
MTLLanguageVersion ShaderLanguage() {
    static_assert(Abi::LanguageMajor == 4u, "the Metal device compiles MSL 4.x");
    return Abi::LanguageMinor >= 1u ? MTLLanguageVersion4_1 : MTLLanguageVersion4_0;
}

// The driver's own kernels. Their arguments are a constant block at the push data slot.
constexpr const char* HelperSource = R"(#include <metal_stdlib>
using namespace metal;

struct Aps5FillArguments {
    ulong address;
    uint count;
    uint reserved;
    uint4 pattern;
};

// Stores a 16-byte pattern `count` times from `address` (16-byte aligned).
kernel void aps5_fill16(constant Aps5FillArguments& arguments [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    if (index < arguments.count) reinterpret_cast<device uint4*>(arguments.address)[index] = arguments.pattern;
}
)";

struct FillArguments {
    std::uint64_t address;
    std::uint32_t count;
    std::uint32_t reserved;
    std::array<std::uint32_t, 4> pattern;
};

static_assert(sizeof(FillArguments) == 32u);

}

VulkanDevice::State::Upload VulkanDevice::State::Stage(std::size_t bytes, std::size_t alignment) {
    const auto offset = (ringUsed + alignment - 1u) & ~(alignment - 1u);
    if (offset + bytes <= RingBytes) {
        ringUsed = offset + bytes;
        return {ring.gpuAddress + offset, static_cast<std::byte*>(ring.contents) + offset, ring, offset};
    }
    // A buffer of its own, kept by the batch of the command being built (Commit).
    id<MTLBuffer> buffer = [device newBufferWithLength:std::max<std::size_t>(bytes, 16u) options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    if (buffer == nil) throw std::runtime_error("Metal device: cannot allocate " + std::to_string(bytes) + " bytes of upload memory");
    [residency addAllocation:buffer];
    [residency commit];
    uploads.push_back(buffer);
    return {buffer.gpuAddress, buffer.contents, buffer, 0u};
}

std::uint64_t VulkanDevice::State::Allocate(std::size_t bytes, std::size_t alignment, void** host) {
    const auto upload = Stage(bytes, alignment);
    *host = upload.host;
    return upload.address;
}

void VulkanDevice::State::Reserve(std::size_t bytes) {
    if (RingBytes - ringUsed >= std::max(bytes, RingHeadroom)) return;
    if (!inflight.empty()) Retire(true);
    ringUsed = 0;
}

std::uint64_t VulkanDevice::State::TryWrap(std::uint64_t address, std::uint64_t bytes) {
    const auto end = address + std::max<std::uint64_t>(bytes, 1u);
    if (end < address) return 0;
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

id<MTLLibrary> VulkanDevice::State::Library(const ShaderRecompiler::MslShader& shader) {
    std::lock_guard lock(pipelinesMutex);
    if (const auto found = libraries.find(shader.source); found != libraries.end()) return found->second;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = ShaderLanguage();
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
    descriptor.source = [NSString stringWithUTF8String:shader.source.c_str()];
    descriptor.options = options;
    NSError* error = nil;
    id<MTLLibrary> library = [compiler newLibraryWithDescriptor:descriptor error:&error];
    if (library == nil) {
        std::fprintf(stderr, "[metal] MSL compile failed:\n%s\n", shader.source.c_str());
        throw std::runtime_error("Metal device: MSL compile failed: " + Describe(error));
    }
    libraries.emplace(shader.source, library);
    return library;
}

id<MTLComputePipelineState> VulkanDevice::State::ComputePipeline(const ShaderRecompiler::RecompileResult& shader) {
    const auto& msl = shader.msl.Shader();
    std::string key = msl.source;
    for (const auto& constant : shader.specialization) key += "|" + std::to_string(constant.id) + "=" + std::to_string(constant.value);
    {
        std::lock_guard lock(pipelinesMutex);
        if (const auto found = computePipelines.find(key); found != computePipelines.end()) return found->second;
    }
    id<MTLLibrary> library = Library(msl);
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.name = @(Abi::EntryPoint);
    function.library = library;
    MTL4FunctionDescriptor* entry = function;
    if (!shader.specialization.empty()) {
        MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
        for (const auto& constant : shader.specialization) {
            const std::uint32_t value = constant.value;
            [values setConstantValue:&value type:MTLDataTypeUInt atIndex:constant.id];
        }
        MTL4SpecializedFunctionDescriptor* specialized = [MTL4SpecializedFunctionDescriptor new];
        specialized.functionDescriptor = function;
        specialized.constantValues = values;
        entry = specialized;
    }
    MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = entry;
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if (pipeline == nil) throw std::runtime_error("Metal device: compute pipeline creation failed: " + Describe(error));
    std::lock_guard lock(pipelinesMutex);
    computePipelines.emplace(std::move(key), pipeline);
    return pipeline;
}

id<MTLComputePipelineState> VulkanDevice::State::HelperPipeline(const char* name) {
    std::lock_guard lock(pipelinesMutex);
    if (const auto found = helperPipelines.find(name); found != helperPipelines.end()) return found->second;
    NSError* error = nil;
    if (helpers == nil) {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
        descriptor.source = @(HelperSource);
        descriptor.options = options;
        helpers = [compiler newLibraryWithDescriptor:descriptor error:&error];
        if (helpers == nil) throw std::runtime_error("Metal device: the driver's helper kernels did not compile: " + Describe(error));
    }
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.name = @(name);
    function.library = helpers;
    MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if (pipeline == nil) throw std::runtime_error(std::string("Metal device: helper pipeline ") + name + " creation failed: " + Describe(error));
    helperPipelines.emplace(name, pipeline);
    return pipeline;
}

id<MTLBuffer> VulkanDevice::State::WorkgroupArena(std::uint64_t groups, std::uint32_t strideDwords) {
    const auto groupBytes = static_cast<std::uint64_t>(strideDwords) * 4u;
    const auto limit = static_cast<std::uint64_t>(device.maxBufferLength);
    if (groups != 0u && groupBytes > limit / groups) {
        throw std::runtime_error("Metal device: the device-memory LDS of " + std::to_string(groups) + " workgroups of " + std::to_string(groupBytes) + " bytes each is over maxBufferLength " + std::to_string(limit));
    }
    const auto bytes = std::max<std::uint64_t>(groups, 1u) * groupBytes;
    if (workgroupArena != nil && workgroupArena.length >= bytes) return workgroupArena;
    // Grown only while idle: in-flight dispatches still index the old arena.
    if (!inflight.empty()) Retire(true);
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

std::uint64_t VulkanDevice::State::BuildAddressTable(std::uint32_t* tableBytes, GuestAllocations::Lease* lease) {
    // The registry generation is read before the lease: a mutation bumps it before it releases the
    // registry, so an unchanged generation means the lease is the one a build would make now.
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (addressTable.generation != generation || addressTable.wrapperEpoch != wrapperEpoch) {
        auto acquired = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
        std::vector<Bda::Range> ranges;
        ranges.reserve(acquired.size());
        std::size_t skipped = 0;
        for (const auto& range : acquired) {
            const auto gpu = TryWrap(range->address, range->bytes);
            if (gpu == 0) {
                ++skipped;
                continue;
            }
            const auto permissions = Bda::Read | (range->writable ? Bda::Write : 0u);
            if ((permissions & Bda::Write) != 0u && range->address + range->bytes > Bda::WrittenAddressLimit) {
                throw std::runtime_error("Metal device: a writable BDA range ends past the addresses the written-page record can hold");
            }
            ranges.push_back({range->address, range->address + range->bytes, gpu, permissions, 0u});
        }
        if (skipped != 0u && addressTable.skipped != skipped) {
            std::fprintf(stderr, "[metal] BDA page table: %zu of %zu guest allocations cannot be wrapped in place and are left out (mirrors: work package D3)\n", skipped, acquired.size());
        }
        // TryWrap may have created wrappers but never drops any, so the epoch is still current.
        const Bda::Header header {Bda::Version, static_cast<std::uint32_t>(ranges.size()), static_cast<std::uint32_t>(sizeof(Bda::Range)), 0u};
        addressTable.bytes.resize(sizeof(header) + ranges.size() * sizeof(Bda::Range));
        std::memcpy(addressTable.bytes.data(), &header, sizeof(header));
        if (!ranges.empty()) std::memcpy(addressTable.bytes.data() + sizeof(header), ranges.data(), ranges.size() * sizeof(Bda::Range));
        addressTable.generation = generation;
        addressTable.wrapperEpoch = wrapperEpoch;
        addressTable.lease = std::move(acquired);
        addressTable.skipped = skipped;
    }
    if (addressTable.bytes.size() > 0xffffffffu) throw std::runtime_error("Metal device: the BDA page table exceeds 4 GiB");
    void* host = nullptr;
    const auto address = Allocate(addressTable.bytes.size(), 16u, &host);
    std::memcpy(host, addressTable.bytes.data(), addressTable.bytes.size());
    *tableBytes = static_cast<std::uint32_t>(addressTable.bytes.size());
    *lease = addressTable.lease;
    return address;
}

VulkanDevice::State::Encoding VulkanDevice::State::Begin() {
    Encoding encoding;
    if (!freeAllocators.empty()) {
        encoding.allocator = freeAllocators.back();
        freeAllocators.pop_back();
    } else {
        encoding.allocator = [device newCommandAllocator];
    }
    encoding.commands = [device newCommandBuffer];
    if (encoding.allocator == nil || encoding.commands == nil) throw std::runtime_error("Metal device: cannot create a command buffer");
    [encoding.commands beginCommandBufferWithAllocator:encoding.allocator];
    encoding.encoder = [encoding.commands computeCommandEncoder];
    // Bring-up ordering: everything committed before, any stage, completes before this encoder.
    [encoding.encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    [encoding.encoder setArgumentTable:arguments];
    return encoding;
}

std::uint64_t VulkanDevice::State::Commit(Encoding& encoding, Batch batch) {
    [encoding.encoder endEncoding];
    [encoding.commands endCommandBuffer];
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    std::shared_ptr<Errors> sink = errors;
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        if (feedback.error == nil) return;
        std::lock_guard lock(sink->mutex);
        sink->messages.push_back(Describe(feedback.error));
    }];
    id<MTL4CommandBuffer> list[1] = {encoding.commands};
    [queue commit:list count:1 options:options];
    [queue signalEvent:event value:++submitted];
    batch.value = submitted;
    batch.allocator = encoding.allocator;
    batch.uploads = std::move(uploads);
    uploads.clear();
    inflight.push_back(std::move(batch));
    encoding = {};
    return submitted;
}

void VulkanDevice::State::Retire(bool wait) {
    if (queue == nil) return;
    if (wait && submitted != 0u && ![event waitUntilSignaledValue:submitted timeoutMS:WaitTimeoutMs]) {
        throw std::runtime_error("Metal device: the GPU did not finish within " + std::to_string(WaitTimeoutMs) + " ms");
    }
    const auto done = event.signaledValue;
    std::vector<std::function<void()>> actions;
    bool released = false;
    while (!inflight.empty() && inflight.front().value <= done) {
        auto& batch = inflight.front();
        [batch.allocator reset];
        freeAllocators.push_back(batch.allocator);
        for (auto& action : batch.completions) actions.push_back(std::move(action));
        for (id<MTLBuffer> buffer : batch.uploads) {
            [residency removeAllocation:buffer];
            released = true;
        }
        inflight.pop_front();
    }
    std::string fault;
    if (inflight.empty()) {
        ringUsed = 0;
        for (const auto& wrapper : wrappers) [residency removeAllocation:wrapper.buffer];
        released = released || !wrappers.empty();
        if (!wrappers.empty()) ++wrapperEpoch;
        wrappers.clear();
        // The cached BDA table points into the dropped wrappers and its lease pins the registry.
        addressTable = {};
        try {
            CheckFaults();
        } catch (const std::exception& error) {
            fault = error.what();
        }
    }
    if (released) [residency commit];
    // Completion actions (EOP interrupts, deferred stores) run in batch order, after the batch's
    // GPU stores are visible; a failure in one does not drop the others.
    std::string actionFailure;
    for (auto& action : actions) {
        try {
            action();
        } catch (const std::exception& error) {
            if (actionFailure.empty()) actionFailure = error.what();
        }
    }
    std::vector<std::string> failures;
    {
        std::lock_guard lock(errors->mutex);
        failures.swap(errors->messages);
    }
    if (!failures.empty()) throw std::runtime_error("Metal device: command buffer failed: " + failures.front());
    if (!fault.empty()) throw std::runtime_error(fault);
    if (!actionFailure.empty()) throw std::runtime_error("Metal device: a completion action failed: " + actionFailure);
}

// The BdaAbi fault buffer after the GPU went idle: GPU stores through BDA addresses are reported
// to the write tracker (written-page record), then a published fault record is raised.
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

VulkanDevice::VulkanDevice(const PresentationWindow* window) : VulkanDevice(window, false) {
}

VulkanDevice::VulkanDevice(CompileOnly) : VulkanDevice(nullptr, true) {
}

VulkanDevice::VulkanDevice(const PresentationWindow* window, bool compileOnly) : state(std::make_unique<State>()), serial(NextSerial()) {
    if (window != nullptr) Unimplemented("presentation to a window", "work package D6");
    @autoreleasepool {
        state->device = MTLCreateSystemDefaultDevice();
        if (state->device == nil) throw std::runtime_error("Metal device: no Metal device");
        if (![state->device supportsFamily:MTLGPUFamilyMetal4] || ![state->device supportsFamily:MTLGPUFamilyApple7]) throw std::runtime_error("Metal device: the GPU lacks the Metal 4 / Apple7 feature set this driver needs");
        NSError* error = nil;
        state->compiler = [state->device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        if (state->compiler == nil) throw std::runtime_error("Metal device: no shader compiler: " + Describe(error));
        for (const auto capability : {spv::CapabilityShader, spv::CapabilityInt64, spv::CapabilityInt16, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformArithmetic, spv::CapabilitySignedZeroInfNanPreserve, spv::CapabilityImageQuery}) {
            state->capabilities.push_back(static_cast<std::uint32_t>(capability));
        }
        std::ranges::sort(state->capabilities);
        state->extensions = {"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
        if (compileOnly) return;
        state->queue = [state->device newMTL4CommandQueue];
        state->event = [state->device newSharedEvent];
        MTLResidencySetDescriptor* residency = [MTLResidencySetDescriptor new];
        residency.initialCapacity = 64;
        state->residency = [state->device newResidencySetWithDescriptor:residency error:&error];
        if (state->queue == nil || state->event == nil || state->residency == nil) throw std::runtime_error("Metal device: queue setup failed: " + Describe(error));
        state->ring = [state->device newBufferWithLength:RingBytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        state->faults = [state->device newBufferWithLength:Bda::FaultBufferBytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        if (state->ring == nil || state->faults == nil) throw std::runtime_error("Metal device: cannot allocate the upload ring or the fault buffer");
        std::memset(state->faults.contents, 0, state->faults.length);
        [state->residency addAllocation:state->ring];
        [state->residency addAllocation:state->faults];
        [state->residency commit];
        [state->queue addResidencySet:state->residency];
        MTL4ArgumentTableDescriptor* table = [MTL4ArgumentTableDescriptor new];
        table.maxBufferBindCount = Abi::Index(Abi::BufferIndex::Count);
        state->arguments = [state->device newArgumentTableWithDescriptor:table error:&error];
        if (state->arguments == nil) throw std::runtime_error("Metal device: argument table setup failed: " + Describe(error));
    }
}

VulkanDevice::~VulkanDevice() {
    try {
        if (state != nullptr) state->Retire(true);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[metal] %s\n", error.what());
    }
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
    return VK_NULL_HANDLE;
}

void VulkanDevice::WaitIdle() {
    state->Retire(true);
}

void VulkanDevice::PrepareForReplacement() {
    WaitIdle();
}

bool VulkanDevice::ImportGuestMemory(const GuestAllocations::Mapped&, std::uint64_t, bool) {
    return true;
}

void VulkanDevice::SubmitRecorded(bool reapFirst) {
    if (reapFirst) state->Retire(false);
}

void VulkanDevice::StoreIdleResults(bool reapFirst) {
    if (reapFirst) state->Retire(false);
}

bool VulkanDevice::CanWaitUnlocked() const {
    return true;
}

std::uint64_t VulkanDevice::SubmitAndEpoch() {
    return state->inflight.empty() ? 0u : state->submitted;
}

void VulkanDevice::WaitRecorded(std::uint64_t serial) {
    if (serial != 0u && ![state->event waitUntilSignaledValue:serial timeoutMS:WaitTimeoutMs]) throw std::runtime_error("Metal device: the GPU did not finish within the timeout");
}

void VulkanDevice::ReapRecorded(std::uint64_t) {
    state->Retire(false);
}

void VulkanDevice::ReapRecorded() {
    state->Retire(false);
}

// The store is a copy from the upload ring into the wrapped label memory in a command buffer of its
// own, ordered after everything committed before it; the CPU writes the label (1) when nothing is
// in flight. Without D2's recorder there is no pending-label table: waiters poll memory.
int VulkanDevice::WriteLabelOnGpu(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t, std::uint32_t, bool reapFirst) {
    if (state->queue == nil || bytes.empty() || bytes.size() > 65536u || address % 4u != 0u || bytes.size() % 4u != 0u) return 4;
    @autoreleasepool {
        if (reapFirst) state->Retire(false);
        if (state->inflight.empty()) return 1;
        state->Reserve(bytes.size());
        if (state->inflight.empty()) return 1;
        if (state->TryWrap(address, bytes.size()) == 0u) return 3;
        std::uint64_t destinationOffset = 0;
        id<MTLBuffer> destination = state->WrapperOf(address, bytes.size(), &destinationOffset);
        const auto source = state->Stage(bytes.size(), 16u);
        std::memcpy(source.host, bytes.data(), bytes.size());
        auto encoding = state->Begin();
        [encoding.encoder copyFromBuffer:source.buffer sourceOffset:source.offset toBuffer:destination destinationOffset:destinationOffset size:bytes.size()];
        state->Commit(encoding, {});
    }
    GuestMemory::MarkWritten(address, bytes.size());
    return 0;
}

bool VulkanDevice::AfterRecordedWork(std::function<void()> action, bool reapFirst) {
    if (state->queue == nil) return false;
    if (reapFirst) state->Retire(false);
    if (state->inflight.empty()) return false;
    state->inflight.back().completions.push_back(std::move(action));
    return true;
}

std::optional<Graphics::Recorder::LabelHit> VulkanDevice::PendingLabel(std::uint64_t, std::size_t, std::uint64_t, Graphics::Recorder::LabelRefusal*) const {
    return std::nullopt;
}

bool VulkanDevice::OpenWriteOverlaps(std::uint64_t, std::size_t) const {
    return false;
}

bool VulkanDevice::RecordedWritesSettled(std::uint64_t, std::size_t) const {
    return state->inflight.empty();
}

bool VulkanDevice::StoresPendingOver(std::uint64_t, std::size_t) const {
    return !state->inflight.empty();
}

// A uniform byte pattern is a fillBuffer; any other 16-byte pattern runs the aps5_fill16 kernel.
bool VulkanDevice::FillBuffer(std::uint64_t address, std::size_t bytes, std::span<const std::uint32_t, 4> pattern) {
    if (state->queue == nil || bytes == 0u || bytes % 16u != 0u || address % 16u != 0u) return false;
    @autoreleasepool {
        state->Retire(false);
        state->Reserve(sizeof(FillArguments));
        const auto gpu = state->TryWrap(address, bytes);
        if (gpu == 0u) return false;
        std::array<std::byte, 16> patternBytes {};
        std::memcpy(patternBytes.data(), pattern.data(), patternBytes.size());
        const bool uniform = std::all_of(patternBytes.begin(), patternBytes.end(), [&](std::byte value) { return value == patternBytes[0]; });
        id<MTLComputePipelineState> pipeline = uniform ? nil : state->HelperPipeline("aps5_fill16");
        std::uint64_t argumentsAddress = 0;
        if (!uniform) {
            if (bytes / 16u > 0xffffffffu) throw std::runtime_error("Metal device: a fill of " + std::to_string(bytes) + " bytes exceeds the fill kernel's range");
            const FillArguments arguments {gpu, static_cast<std::uint32_t>(bytes / 16u), 0u, {pattern[0], pattern[1], pattern[2], pattern[3]}};
            void* host = nullptr;
            argumentsAddress = state->Allocate(sizeof(arguments), 16u, &host);
            std::memcpy(host, &arguments, sizeof(arguments));
        }
        std::uint64_t offset = 0;
        id<MTLBuffer> wrapper = state->WrapperOf(address, bytes, &offset);
        auto encoding = state->Begin();
        if (uniform) {
            [encoding.encoder fillBuffer:wrapper range:NSMakeRange(offset, bytes) value:static_cast<std::uint8_t>(patternBytes[0])];
        } else {
            [encoding.encoder setComputePipelineState:pipeline];
            [state->arguments setAddress:argumentsAddress atIndex:Abi::Index(Abi::BufferIndex::PushData)];
            const auto count = bytes / 16u;
            const auto width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256u);
            [encoding.encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(width, count), 1, 1)];
        }
        state->Commit(encoding, {});
    }
    GuestMemory::MarkWritten(address, bytes);
    return true;
}

bool VulkanDevice::DumpSamplesOnGpu(std::uint64_t) {
    return false;
}

// Path 0 (CPU copy) only while nothing is in flight: without D2's recorder the device cannot tell
// which ranges in-flight work reads or writes. Path 1 records a copy between the two wrappers,
// ordered after everything committed before it.
VulkanDevice::CopyOutcome VulkanDevice::CopyBuffer(std::uint64_t destination, std::uint64_t source, std::size_t bytes, std::size_t cpuMax, std::size_t gpuMax, std::size_t, std::uint64_t, std::uint32_t, const CopyWriterNote& noteWriter) {
    CopyOutcome outcome {2, false, 0, CopyOutcome::None, false, false, 0, 0, 0, 0, false};
    if (state->queue == nil || bytes == 0u) return outcome;
    @autoreleasepool {
        state->Retire(false);
        if (bytes <= cpuMax && state->inflight.empty()) {
            GuestMemory::StoreOwnBytes(destination, bytes, [&] { std::memmove(reinterpret_cast<void*>(destination), reinterpret_cast<const void*>(source), bytes); });
            outcome.path = 0;
            return outcome;
        }
        outcome.reason = bytes > cpuMax ? CopyOutcome::Size : CopyOutcome::SourcePending;
        if (bytes > gpuMax) {
            outcome.path = 8;
            return outcome;
        }
        state->Reserve(0);
        if (state->TryWrap(destination, bytes) == 0u || state->TryWrap(source, bytes) == 0u) return outcome;
        std::uint64_t destinationOffset = 0;
        std::uint64_t sourceOffset = 0;
        id<MTLBuffer> destinationBuffer = state->WrapperOf(destination, bytes, &destinationOffset);
        id<MTLBuffer> sourceBuffer = state->WrapperOf(source, bytes, &sourceOffset);
        auto encoding = state->Begin();
        [encoding.encoder copyFromBuffer:sourceBuffer sourceOffset:sourceOffset toBuffer:destinationBuffer destinationOffset:destinationOffset size:bytes];
        state->Commit(encoding, {});
    }
    GuestMemory::MarkWritten(destination, bytes);
    noteWriter({}, 0);
    outcome.path = 1;
    return outcome;
}

VulkanDevice::CopyVerification VulkanDevice::CopyVerifyCounts() {
    return {};
}

VulkanDevice::CopyAliasing VulkanDevice::CopyAliasCounts() {
    return {};
}

std::shared_ptr<PreparedDispatch> VulkanDevice::PrepareDispatch(const ShaderRecompiler::RecompileResult&, std::span<const Graphics::GuestMemorySnapshot>, std::uint64_t) {
    return nullptr;
}

std::span<const double, 5> VulkanDevice::PreparePhaseMs(const PreparedDispatch&) {
    static constexpr std::array<double, 5> zero {};
    return zero;
}

namespace {

std::uint32_t WorkgroupStride(const ShaderRecompiler::RecompileResult& shader) {
    return shader.msl.Reflection().workgroupMemoryStrideDwords;
}

}

void VulkanDevice::State::RequireDispatchable(const ShaderRecompiler::RecompileResult& shader) const {
    if (queue == nil) throw std::runtime_error("Metal device: a compile-only device cannot dispatch");
    if (shader.msl.empty()) throw std::runtime_error("Metal device: the shader has no MSL form (recompile it for the device's Target(), whose backend is Msl)");
    ShaderRecompiler::RuntimeAbi::RequireVersion(shader.runtimeAbiVersion);
    const auto& reflection = shader.msl.Reflection();
    if (reflection.abiVersion != Abi::Version) throw std::runtime_error("Metal device: the shader was emitted for MslAbi version " + std::to_string(reflection.abiVersion) + ", the device binds version " + std::to_string(Abi::Version));
    if (reflection.stage != Abi::Stage::Kernel) throw std::runtime_error("Metal device: Dispatch needs a kernel");
}

// Records one compute dispatch of `shader` in a command buffer of its own (see the binding
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
void VulkanDevice::State::RecordCompute(const ShaderRecompiler::RecompileResult& shader, std::uint32_t gridX, std::uint32_t gridY, std::uint32_t gridZ, std::uint64_t indirectArguments) {
    struct {
        std::uint32_t x, y, z;
        std::uint64_t indirect;
    } const grid {gridX, gridY, gridZ, indirectArguments};
    auto& state = *this;
    const auto& reflection = shader.msl.Reflection();
    const auto stride = WorkgroupStride(shader);
    const bool split = grid.indirect == 0u && (grid.x > GroupLimit || grid.y > GroupLimit || grid.z > GroupLimit);
    std::uint64_t chunks = 1;
    if (split) {
        if (stride != 0u) {
            throw std::runtime_error("Metal device: workgroup count " + std::to_string(grid.x) + "x" + std::to_string(grid.y) + "x" + std::to_string(grid.z) + " exceeds the device limits " + std::to_string(GroupLimit) + " per dimension and a split aliases the device-memory LDS slots, whose base is computed from threadgroups_per_grid of a single chunk");
        }
        for (const std::uint64_t count : {grid.x, grid.y, grid.z}) chunks *= (count + GroupLimit - 1u) / GroupLimit;
        if (chunks > MaxSplitChunks) throw std::runtime_error("Metal device: a dispatch of " + std::to_string(grid.x) + "x" + std::to_string(grid.y) + "x" + std::to_string(grid.z) + " workgroups needs more than " + std::to_string(MaxSplitChunks) + " chunks");
    }
    // Ring room before anything is wrapped: a full ring waits for idle, which drops wrappers.
    std::size_t upload = reflection.table.bytes + Abi::PushDataBytes + chunks * sizeof(Abi::DriverData) + 64u;
    for (const auto& binding : shader.bindings) {
        if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData || binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt) upload += binding.guestDescriptor.size() * sizeof(std::uint32_t) + 16u;
    }
    state.Reserve(upload);
    // The arena is sized for the whole grid (an indirect dispatch with an arena reads its counts on
    // the CPU first, so `grid` is direct here).
    id<MTLBuffer> arena = stride != 0u ? state.WorkgroupArena(static_cast<std::uint64_t>(grid.x) * grid.y * grid.z, stride) : nil;
    id<MTLComputePipelineState> pipeline = state.ComputePipeline(shader);
    // Wrapped after the ring check, which may wait for idle and drop wrappers.
    const auto indirect = grid.indirect != 0u ? state.Wrap(grid.indirect, 12u) : 0u;
    const auto& threads = reflection.threadsPerThreadgroup;
    if (static_cast<NSUInteger>(threads[0]) * threads[1] * threads[2] > pipeline.maxTotalThreadsPerThreadgroup) {
        throw std::runtime_error("Metal device: the workgroup of " + std::to_string(threads[0] * threads[1] * threads[2]) + " threads exceeds the pipeline's maximum threadgroup size " + std::to_string(pipeline.maxTotalThreadsPerThreadgroup));
    }

    std::array<std::byte, Abi::PushDataBytes> push {};
    if (!shader.pushConstants.empty()) {
        if (shader.pushConstants.size() > push.size()) throw std::runtime_error("Metal device: push constants exceed the push data block");
        std::memcpy(push.data(), shader.pushConstants.data(), shader.pushConstants.size());
    }
    Batch batch {};
    std::uint64_t tableAddress = 0;
    if (reflection.table.bytes != 0u) {
        void* host = nullptr;
        tableAddress = state.Allocate(reflection.table.bytes, 16u, &host);
        std::memset(host, 0, reflection.table.bytes);
        auto* table = static_cast<std::byte*>(host);
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
                    entries[element] = {state.Wrap(aligned, bytes), static_cast<std::uint32_t>(bytes), 0u};
                    const auto position = static_cast<std::size_t>(shader.memoryOffsetDword) * 4u + element;
                    if (position < shader.pushConstants.size()) push[position] = std::byte{0};
                }
                break;
            case ShaderRecompiler::DescriptorRole::ShaderData:
            case ShaderRecompiler::DescriptorRole::FlattenedSrt: {
                const auto bytes = binding.guestDescriptor.size() * sizeof(std::uint32_t);
                void* data = nullptr;
                const auto address = state.Allocate(std::max<std::size_t>(bytes, 4u), 16u, &data);
                if (bytes != 0u) std::memcpy(data, binding.guestDescriptor.data(), bytes);
                entries[0] = {address, static_cast<std::uint32_t>(bytes), 0u};
                break;
            }
            case ShaderRecompiler::DescriptorRole::Gds: {
                if (binding.count != 1u || !binding.guestDescriptor.empty()) throw std::runtime_error("Metal device: invalid GDS descriptor contract");
                const auto gds = Pm4::GdsAddress();
                entries[0] = {state.Wrap(gds, Pm4::GdsBytes), static_cast<std::uint32_t>(Pm4::GdsBytes), 0u};
                break;
            }
            case ShaderRecompiler::DescriptorRole::BdaPagetable: {
                std::uint32_t bytes = 0;
                const auto address = state.BuildAddressTable(&bytes, &batch.lease);
                entries[0] = {address, bytes, 0u};
                break;
            }
            case ShaderRecompiler::DescriptorRole::FaultBuffer:
                entries[0] = {state.faults.gpuAddress, static_cast<std::uint32_t>(state.faults.length), 0u};
                break;
            case ShaderRecompiler::DescriptorRole::GuestImages:
            case ShaderRecompiler::DescriptorRole::GuestSamplers:
                Unimplemented("image and sampler descriptors in compute dispatches", "work packages D4/D5");
            }
        }
    }
    void* pushHost = nullptr;
    const auto pushAddress = state.Allocate(push.size(), 16u, &pushHost);
    std::memcpy(pushHost, push.data(), push.size());
    std::vector<std::uint64_t> driverAddresses;
    std::vector<MTLSize> chunkGroups;
    const auto addChunk = [&](std::uint32_t xBase, std::uint32_t yBase, std::uint32_t zBase, MTLSize groups) {
        Abi::DriverData driver {};
        driver.workgroupBase = {xBase, yBase, zBase};
        driver.loopGuardLimit = LoopGuardLimit();
        driver.faultBuffer = state.faults.gpuAddress;
        void* host = nullptr;
        driverAddresses.push_back(state.Allocate(sizeof(driver), 16u, &host));
        std::memcpy(host, &driver, sizeof(driver));
        chunkGroups.push_back(groups);
    };
    if (split) {
        for (std::uint64_t zBase = 0; zBase < grid.z; zBase += GroupLimit) {
            for (std::uint64_t yBase = 0; yBase < grid.y; yBase += GroupLimit) {
                for (std::uint64_t xBase = 0; xBase < grid.x; xBase += GroupLimit) {
                    addChunk(static_cast<std::uint32_t>(xBase), static_cast<std::uint32_t>(yBase), static_cast<std::uint32_t>(zBase), MTLSizeMake(std::min<std::uint64_t>(GroupLimit, grid.x - xBase), std::min<std::uint64_t>(GroupLimit, grid.y - yBase), std::min<std::uint64_t>(GroupLimit, grid.z - zBase)));
                }
            }
        }
    } else {
        addChunk(0u, 0u, 0u, MTLSizeMake(grid.x, grid.y, grid.z));
    }

    auto encoding = state.Begin();
    [encoding.encoder setComputePipelineState:pipeline];
    [state.arguments setAddress:tableAddress != 0u ? tableAddress : driverAddresses.front() atIndex:Abi::Index(Abi::BufferIndex::DescriptorTable)];
    [state.arguments setAddress:pushAddress atIndex:Abi::Index(Abi::BufferIndex::PushData)];
    [state.arguments setAddress:arena != nil ? arena.gpuAddress : driverAddresses.front() atIndex:Abi::Index(Abi::BufferIndex::WorkgroupMemory)];
    const auto threadsPerGroup = MTLSizeMake(threads[0], threads[1], threads[2]);
    for (std::size_t chunk = 0; chunk < driverAddresses.size(); ++chunk) {
        // Argument table bindings are snapshotted per dispatch, so each chunk sees its own base.
        [state.arguments setAddress:driverAddresses[chunk] atIndex:Abi::Index(Abi::BufferIndex::DriverData)];
        if (indirect != 0u) {
            [encoding.encoder dispatchThreadgroupsWithIndirectBuffer:indirect threadsPerThreadgroup:threadsPerGroup];
        } else {
            [encoding.encoder dispatchThreadgroups:chunkGroups[chunk] threadsPerThreadgroup:threadsPerGroup];
        }
    }
    state.Commit(encoding, std::move(batch));
}

void VulkanDevice::Dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::span<const Graphics::GuestMemorySnapshot>, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>* recipe) {
    if (recipe != nullptr) *recipe = nullptr;
    state->RequireDispatchable(shader);
    if (x == 0u || y == 0u || z == 0u) return;
    @autoreleasepool {
        state->Retire(false);
        state->RecordCompute(shader, x, y, z, 0u);
    }
}

// DISPATCH_INDIRECT: the GPU reads the three group counts from guest memory (wrapped in place)
// behind everything committed before, so a dispatch or label writing them earlier is seen.
// cpuReason (the counts are read on the CPU after the device went idle, then dispatched directly):
// 8 the shader uses the device-memory LDS arena, sized from the counts; 3 the arguments cannot be
// wrapped (unmapped, executable or misaligned memory).
VulkanDevice::IndirectOutcome VulkanDevice::DispatchIndirect(const ShaderRecompiler::RecompileResult& shader, std::uint64_t arguments, std::span<const Graphics::GuestMemorySnapshot>, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>* recipe) {
    if (recipe != nullptr) *recipe = nullptr;
    state->RequireDispatchable(shader);
    IndirectOutcome outcome {0, 0.0};
    @autoreleasepool {
        state->Retire(false);
        if (WorkgroupStride(shader) != 0u || shader.workgroupMemoryDwords != 0u) {
            outcome.cpuReason = 8;
        } else if (arguments % 4u != 0u || state->TryWrap(arguments, 12u) == 0u) {
            outcome.cpuReason = 3;
        }
        if (outcome.cpuReason == 0) {
            state->RecordCompute(shader, 1u, 1u, 1u, arguments);
            return outcome;
        }
        const auto start = std::chrono::steady_clock::now();
        state->Retire(true);
        std::array<std::uint32_t, 3> groups {};
        GuestMemory::Read(arguments, std::as_writable_bytes(std::span(groups)), 4u);
        outcome.argumentReadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (groups[0] != 0u && groups[1] != 0u && groups[2] != 0u) state->RecordCompute(shader, groups[0], groups[1], groups[2], 0u);
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
    Unimplemented("the Vulkan Graphics::Context", "Vulkan-only test hook");
}

}
