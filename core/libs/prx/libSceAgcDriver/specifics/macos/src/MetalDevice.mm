// The AGC driver's device API (VulkanDevice.hpp, overlaid by
// specifics/macos/overlay/.../VulkanDevice.hpp) implemented on Metal 4. M2 foundation: device
// setup, the Metal shader target, compute dispatches of MetalBackend shaders over wrapped guest
// memory, and the synchronization the PM4 core needs; the rest throws until its work package
// lands (docs/mac/M2_DESIGN.md). Command model: one MTL4 command queue, a MTLSharedEvent timeline,
// every command buffer ordered after all earlier work by a queue barrier (bring-up policy).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "BdaAbi.hpp"
#include "MslAbi.hpp"
#include <spirv/unified1/spirv.hpp>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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

namespace {

constexpr std::uint64_t HostPage = 16384u;
constexpr std::size_t RingBytes = 8u << 20u;
constexpr std::uint64_t WaitTimeoutMs = 20000u;

[[noreturn]] void Unimplemented(const char* what, const char* package) {
    throw std::runtime_error(std::string("Metal device: ") + what + " is not implemented yet (" + package + ", docs/mac/M2_DESIGN.md)");
}

std::uint64_t NextSerial() {
    static std::atomic<std::uint64_t> serials{0};
    return serials.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::string Describe(NSError* error) {
    return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
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

}

struct VulkanDevice::State {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTLSharedEvent> event = nil;
    id<MTLResidencySet> residency = nil;
    id<MTL4ArgumentTable> arguments = nil;
    id<MTLBuffer> ring = nil;
    id<MTLBuffer> faults = nil;
    std::size_t ringUsed = 0;
    std::uint64_t submitted = 0;
    std::vector<std::uint32_t> capabilities;
    std::vector<std::string_view> extensions;

    struct Batch {
        std::uint64_t value;
        id<MTL4CommandAllocator> allocator;
    };
    std::deque<Batch> inflight;
    std::vector<id<MTL4CommandAllocator>> freeAllocators;
    // Command buffer errors reported by commit feedback, which may arrive after the device is gone.
    struct Errors {
        std::mutex mutex;
        std::vector<std::string> messages;
    };
    std::shared_ptr<Errors> errors = std::make_shared<Errors>();

    struct Wrapper {
        std::uint64_t begin;
        std::uint64_t end;
        id<MTLBuffer> buffer;
    };
    std::vector<Wrapper> wrappers;

    std::mutex pipelinesMutex;
    std::unordered_map<std::string, id<MTLLibrary>> libraries;
    std::unordered_map<std::string, id<MTLComputePipelineState>> computePipelines;

    std::uint64_t Allocate(std::size_t bytes, std::size_t alignment, void** host);
    std::uint64_t Wrap(std::uint64_t address, std::uint64_t bytes);
    id<MTLLibrary> Library(const ShaderRecompiler::MslShader& shader);
    id<MTLComputePipelineState> ComputePipeline(const ShaderRecompiler::RecompileResult& shader);
    void Retire(bool wait);
    void CheckFaults();
};

std::uint64_t VulkanDevice::State::Allocate(std::size_t bytes, std::size_t alignment, void** host) {
    auto offset = (ringUsed + alignment - 1u) & ~(alignment - 1u);
    if (offset + bytes > RingBytes) {
        Retire(true);
        if (!inflight.empty()) throw std::runtime_error("Metal device: the transient ring is exhausted with work in flight");
        offset = 0;
    }
    if (bytes > RingBytes) throw std::runtime_error("Metal device: a transient allocation of " + std::to_string(bytes) + " bytes exceeds the ring");
    ringUsed = offset + bytes;
    *host = static_cast<std::byte*>(ring.contents) + offset;
    return ring.gpuAddress + offset;
}

// The GPU address of guest (host) memory [address, address + bytes): the 16 KiB pages around it
// wrapped in place (newBufferWithBytesNoCopy) within its VM regions, never executable pages.
std::uint64_t VulkanDevice::State::Wrap(std::uint64_t address, std::uint64_t bytes) {
    const auto end = address + std::max<std::uint64_t>(bytes, 1u);
    for (const auto& wrapper : wrappers) {
        if (address >= wrapper.begin && end <= wrapper.end) return wrapper.buffer.gpuAddress + (address - wrapper.begin);
    }
    const auto pageBegin = address & ~(HostPage - 1u);
    const auto pageEnd = (end + HostPage - 1u) & ~(HostPage - 1u);
    mach_vm_address_t cursor = pageBegin;
    while (cursor < pageEnd) {
        mach_vm_address_t region = cursor;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info {};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || region > cursor) {
            throw std::runtime_error("Metal device: guest memory at " + std::to_string(cursor) + " is not mapped");
        }
        if ((info.protection & VM_PROT_EXECUTE) != 0 || (info.protection & VM_PROT_READ) == 0) {
            throw std::runtime_error("Metal device: guest memory at " + std::to_string(cursor) + " is executable or unreadable and cannot be wrapped (copy fallback: work package D2)");
        }
        cursor = region + size;
    }
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(pageBegin) length:pageEnd - pageBegin options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
    if (buffer == nil) throw std::runtime_error("Metal device: newBufferWithBytesNoCopy refused guest memory at " + std::to_string(pageBegin));
    [residency addAllocation:buffer];
    [residency commit];
    wrappers.push_back({pageBegin, pageEnd, buffer});
    return buffer.gpuAddress + (address - pageBegin);
}

id<MTLLibrary> VulkanDevice::State::Library(const ShaderRecompiler::MslShader& shader) {
    std::lock_guard lock(pipelinesMutex);
    if (const auto found = libraries.find(shader.source); found != libraries.end()) return found->second;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
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
    function.name = @(ShaderRecompiler::MslAbi::EntryPoint);
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

void VulkanDevice::State::Retire(bool wait) {
    if (queue == nil) return;
    if (wait && submitted != 0u && ![event waitUntilSignaledValue:submitted timeoutMS:WaitTimeoutMs]) {
        throw std::runtime_error("Metal device: the GPU did not finish within " + std::to_string(WaitTimeoutMs) + " ms");
    }
    const auto done = event.signaledValue;
    while (!inflight.empty() && inflight.front().value <= done) {
        [inflight.front().allocator reset];
        freeAllocators.push_back(inflight.front().allocator);
        inflight.pop_front();
    }
    if (inflight.empty()) {
        ringUsed = 0;
        for (const auto& wrapper : wrappers) [residency removeAllocation:wrapper.buffer];
        if (!wrappers.empty()) [residency commit];
        wrappers.clear();
        CheckFaults();
    }
    std::vector<std::string> failures;
    {
        std::lock_guard lock(errors->mutex);
        failures.swap(errors->messages);
    }
    if (!failures.empty()) throw std::runtime_error("Metal device: command buffer failed: " + failures.front());
}

void VulkanDevice::State::CheckFaults() {
    auto* fault = static_cast<ShaderRecompiler::BdaAbi::Fault*>(faults.contents);
    if (fault->state != ShaderRecompiler::BdaAbi::FaultState::Ready) return;
    const auto reason = static_cast<std::uint32_t>(fault->reason);
    const auto address = fault->address;
    const auto instruction = fault->instruction;
    std::memset(faults.contents, 0, faults.length);
    throw std::runtime_error("Metal device: GPU fault reason " + std::to_string(reason) + " address 0x" + [&] {
        char text[24];
        std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(address));
        return std::string(text);
    }() + " pc " + std::to_string(instruction));
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
        state->faults = [state->device newBufferWithLength:ShaderRecompiler::BdaAbi::FaultBufferBytes options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        std::memset(state->faults.contents, 0, state->faults.length);
        [state->residency addAllocation:state->ring];
        [state->residency addAllocation:state->faults];
        [state->residency commit];
        [state->queue addResidencySet:state->residency];
        MTL4ArgumentTableDescriptor* table = [MTL4ArgumentTableDescriptor new];
        table.maxBufferBindCount = ShaderRecompiler::MslAbi::Index(ShaderRecompiler::MslAbi::BufferIndex::Count);
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
    ShaderRecompiler::SpirvTarget target {0u, 0u, 32u, ShaderRecompiler::BdaAbi::Version, state->capabilities, state->extensions, false, {static_cast<std::uint32_t>(threads.width), static_cast<std::uint32_t>(threads.height), static_cast<std::uint32_t>(threads.depth)}, 1024u, threadgroupMemory, {}, {}};
    target.backend = ShaderRecompiler::ShaderBackend::Msl;
    return target;
}

std::optional<ShaderRecompiler::GeometryStageLimits> VulkanDevice::GeometryLimits() const {
    return std::nullopt;
}

const std::uint32_t* VulkanDevice::WorkGroupCountLimit() const {
    static constexpr std::uint32_t limits[3] = {65535u, 65535u, 65535u};
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

int VulkanDevice::WriteLabelOnGpu(std::uint64_t, std::span<const std::byte>, std::uint64_t, std::uint32_t, bool) {
    state->Retire(true);
    return 1;
}

bool VulkanDevice::AfterRecordedWork(std::function<void()>, bool) {
    Unimplemented("AfterRecordedWork (completion actions)", "work package D1");
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

bool VulkanDevice::FillBuffer(std::uint64_t, std::size_t, std::span<const std::uint32_t, 4>) {
    state->Retire(true);
    return false;
}

bool VulkanDevice::DumpSamplesOnGpu(std::uint64_t) {
    return false;
}

VulkanDevice::CopyOutcome VulkanDevice::CopyBuffer(std::uint64_t, std::uint64_t, std::size_t, std::size_t, std::size_t, std::size_t, std::uint64_t, std::uint32_t, const CopyWriterNote&) {
    CopyOutcome outcome {};
    outcome.path = 2;
    return outcome;
}

VulkanDevice::CopyVerification VulkanDevice::CopyVerifyCounts() {
    return {};
}

VulkanDevice::CopyAliasing VulkanDevice::CopyAliasCounts() {
    return {};
}

void* VulkanDevice::Window() const {
    return nullptr;
}

void VulkanDevice::Resize(std::uint32_t, std::uint32_t) {
}

bool VulkanDevice::Presentable() const {
    return false;
}

bool VulkanDevice::DualSrcBlend() const {
    return true;
}

bool VulkanDevice::PrimitiveListRestart() const {
    return false;
}

bool VulkanDevice::SamplerFilterMinmax() const {
    return false;
}

bool VulkanDevice::ConservativeRasterization() const {
    return false;
}

VkShaderStageFlags VulkanDevice::SubgroupStages() const {
    return VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
}

bool VulkanDevice::ProvokingVertexLast() const {
    return false;
}

bool VulkanDevice::GraphicsPipelineLibraries() const {
    return false;
}

std::size_t VulkanDevice::FlipInFlight() {
    return 1u;
}

double VulkanDevice::RetirePresents(std::size_t) {
    return 0.0;
}

bool VulkanDevice::PresentWaitsForSlots(const DisplayBuffer*) const {
    return false;
}

bool VulkanDevice::AcquireImage() {
    Unimplemented("presentation (AcquireImage)", "work package D6");
}

bool VulkanDevice::PresentClear(std::uint32_t, std::uint32_t, bool) {
    Unimplemented("presentation (PresentClear)", "work package D6");
}

void VulkanDevice::PresentPixels(std::uint32_t, std::uint32_t, std::span<const std::byte>) {
    Unimplemented("presentation (PresentPixels)", "work package D6");
}

bool VulkanDevice::PresentDisplayBuffer(const DisplayBuffer&) {
    Unimplemented("presentation (PresentDisplayBuffer)", "work package D6");
}

double VulkanDevice::FinishPresent() {
    return 0.0;
}

void VulkanDevice::QueuePresent() {
    Unimplemented("presentation (QueuePresent)", "work package D6");
}

void VulkanDevice::FlipBatches(std::uint64_t& submissions, std::uint64_t& unsignaled) const {
    submissions = state->submitted;
    unsignaled = state->inflight.size();
}

VulkanDevice::PresentStatistics VulkanDevice::PresentCounts() {
    return {};
}

std::shared_ptr<PreparedDispatch> VulkanDevice::PrepareDispatch(const ShaderRecompiler::RecompileResult&, std::span<const Graphics::GuestMemorySnapshot>, std::uint64_t) {
    return nullptr;
}

std::span<const double, 5> VulkanDevice::PreparePhaseMs(const PreparedDispatch&) {
    static constexpr std::array<double, 5> zero {};
    return zero;
}

void VulkanDevice::Dispatch(const ShaderRecompiler::RecompileResult& shader, std::uint32_t x, std::uint32_t y, std::uint32_t z, std::span<const Graphics::GuestMemorySnapshot>, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>* recipe) {
    if (recipe != nullptr) *recipe = nullptr;
    if (state->queue == nil) throw std::runtime_error("Metal device: a compile-only device cannot dispatch");
    if (shader.msl.empty()) throw std::runtime_error("Metal device: the shader has no MSL form (recompile it for the device's Target(), whose backend is Msl)");
    ShaderRecompiler::RuntimeAbi::RequireVersion(shader.runtimeAbiVersion);
    if (x == 0u || y == 0u || z == 0u) return;
    const auto& reflection = shader.msl.Reflection();
    if (reflection.abiVersion != ShaderRecompiler::MslAbi::Version) throw std::runtime_error("Metal device: the shader was emitted for another MslAbi version");
    if (reflection.stage != ShaderRecompiler::MslAbi::Stage::Kernel) throw std::runtime_error("Metal device: Dispatch needs a kernel");
    @autoreleasepool {
        state->Retire(false);
        id<MTLComputePipelineState> pipeline = state->ComputePipeline(shader);
        const auto& threads = reflection.threadsPerThreadgroup;
        if (static_cast<NSUInteger>(threads[0]) * threads[1] * threads[2] > pipeline.maxTotalThreadsPerThreadgroup) throw std::runtime_error("Metal device: the workgroup exceeds the pipeline's maximum threadgroup size");

        std::array<std::byte, ShaderRecompiler::MslAbi::PushDataBytes> push {};
        if (!shader.pushConstants.empty()) {
            if (shader.pushConstants.size() > push.size()) throw std::runtime_error("Metal device: push constants exceed the push data block");
            std::memcpy(push.data(), shader.pushConstants.data(), shader.pushConstants.size());
        }
        std::uint64_t tableAddress = 0;
        if (reflection.table.bytes != 0u) {
            void* host = nullptr;
            tableAddress = state->Allocate(reflection.table.bytes, 16u, &host);
            std::memset(host, 0, reflection.table.bytes);
            auto* table = static_cast<std::byte*>(host);
            for (const auto& binding : shader.bindings) {
                const auto group = binding.binding / static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
                const auto kind = static_cast<ShaderRecompiler::RuntimeAbi::Binding>(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count));
                if (group != static_cast<std::uint32_t>(reflection.group)) throw std::runtime_error("Metal device: a binding belongs to another stage group");
                const auto* range = reflection.table.Find(kind);
                if (range == nullptr) continue;
                auto* entries = reinterpret_cast<ShaderRecompiler::MslAbi::BufferEntry*>(table + range->offset);
                switch (binding.role) {
                case ShaderRecompiler::DescriptorRole::GuestBuffers:
                    for (std::uint32_t element = 0; element < std::min(binding.count, range->count); ++element) {
                        const auto guest = DecodeBuffer(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 4u, 4u));
                        if (guest.base == 0u || guest.bytes == 0u) continue;
                        const auto aligned = guest.base & ~std::uint64_t{3u};
                        const auto bytes = (guest.base - aligned + guest.bytes + 3u) & ~std::uint64_t{3u};
                        if (bytes > 0xffffffffu) throw std::runtime_error("Metal device: a guest buffer exceeds 4 GiB");
                        entries[element] = {state->Wrap(aligned, bytes), static_cast<std::uint32_t>(bytes), 0u};
                        const auto position = static_cast<std::size_t>(shader.memoryOffsetDword) * 4u + element;
                        if (position < shader.pushConstants.size()) push[position] = std::byte{0};
                    }
                    break;
                case ShaderRecompiler::DescriptorRole::ShaderData:
                case ShaderRecompiler::DescriptorRole::FlattenedSrt: {
                    const auto bytes = binding.guestDescriptor.size() * sizeof(std::uint32_t);
                    void* data = nullptr;
                    const auto address = state->Allocate(std::max<std::size_t>(bytes, 4u), 16u, &data);
                    if (bytes != 0u) std::memcpy(data, binding.guestDescriptor.data(), bytes);
                    entries[0] = {address, static_cast<std::uint32_t>(bytes), 0u};
                    break;
                }
                case ShaderRecompiler::DescriptorRole::FaultBuffer:
                    entries[0] = {state->faults.gpuAddress, static_cast<std::uint32_t>(state->faults.length), 0u};
                    break;
                default:
                    Unimplemented("this descriptor role in compute dispatches", "work packages D2-D4");
                }
            }
        }
        void* pushHost = nullptr;
        const auto pushAddress = state->Allocate(push.size(), 16u, &pushHost);
        std::memcpy(pushHost, push.data(), push.size());
        void* driverHost = nullptr;
        const auto driverAddress = state->Allocate(sizeof(ShaderRecompiler::MslAbi::DriverData), 16u, &driverHost);
        ShaderRecompiler::MslAbi::DriverData driver {};
        driver.loopGuardLimit = LoopGuardLimit();
        driver.faultBuffer = state->faults.gpuAddress;
        std::memcpy(driverHost, &driver, sizeof(driver));

        id<MTL4CommandAllocator> allocator = nil;
        if (!state->freeAllocators.empty()) {
            allocator = state->freeAllocators.back();
            state->freeAllocators.pop_back();
        } else {
            allocator = [state->device newCommandAllocator];
        }
        id<MTL4CommandBuffer> commands = [state->device newCommandBuffer];
        [commands beginCommandBufferWithAllocator:allocator];
        id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder setComputePipelineState:pipeline];
        [encoder setArgumentTable:state->arguments];
        using ShaderRecompiler::MslAbi::BufferIndex;
        [state->arguments setAddress:tableAddress != 0u ? tableAddress : driverAddress atIndex:ShaderRecompiler::MslAbi::Index(BufferIndex::DescriptorTable)];
        [state->arguments setAddress:pushAddress atIndex:ShaderRecompiler::MslAbi::Index(BufferIndex::PushData)];
        [state->arguments setAddress:driverAddress atIndex:ShaderRecompiler::MslAbi::Index(BufferIndex::WorkgroupMemory)];
        [state->arguments setAddress:driverAddress atIndex:ShaderRecompiler::MslAbi::Index(BufferIndex::DriverData)];
        [encoder dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(threads[0], threads[1], threads[2])];
        [encoder endEncoding];
        [commands endCommandBuffer];
        MTL4CommitOptions* options = [MTL4CommitOptions new];
        std::shared_ptr<State::Errors> sink = state->errors;
        [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            if (feedback.error == nil) return;
            std::lock_guard lock(sink->mutex);
            sink->messages.push_back(Describe(feedback.error));
        }];
        id<MTL4CommandBuffer> list[1] = {commands};
        [state->queue commit:list count:1 options:options];
        [state->queue signalEvent:state->event value:++state->submitted];
        state->inflight.push_back({state->submitted, allocator});
    }
}

VulkanDevice::IndirectOutcome VulkanDevice::DispatchIndirect(const ShaderRecompiler::RecompileResult&, std::uint64_t, std::span<const Graphics::GuestMemorySnapshot>, std::uint64_t, std::shared_ptr<PreparedDispatch>, std::shared_ptr<const Recipe>*) {
    Unimplemented("DispatchIndirect", "work package D1");
}

std::shared_ptr<RecipeHit> VulkanDevice::PrepareRecipe(const std::shared_ptr<const Recipe>&, bool) {
    return nullptr;
}

RecipeOutcome VulkanDevice::DispatchRecipe(const ShaderRecompiler::RecompileResult&, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t, std::uint64_t, const std::shared_ptr<RecipeHit>&, IndirectOutcome&, const std::shared_ptr<PreparedDispatch>&, bool) {
    Unimplemented("DispatchRecipe (dispatch recipes)", "work package D1");
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

void VulkanDevice::NoteDrawRecipeMiss(DrawRecipePrecheck) {
}

void VulkanDevice::Draw(const Graphics::State&, const Pm4::DrawParameters&, std::span<const Graphics::CompiledShader>, std::span<const Graphics::GuestMemorySnapshot>, std::shared_ptr<const DrawRecipe>*) {
    Unimplemented("Draw", "work package D5");
}

std::optional<std::string> VulkanDevice::KnownDrawRejection(const Graphics::State&, std::span<const Graphics::CompiledShader>) const {
    return std::string("draws are not implemented on the Metal device yet (work package D5)");
}

void VulkanDevice::ColorMetadataPass(const Graphics::ColorMetadataPass&) {
    Unimplemented("ColorMetadataPass", "work package D4");
}

RecipeOutcome VulkanDevice::DrawFromRecipe(const Graphics::State&, const Pm4::DrawParameters&, std::span<const Graphics::CompiledShader>, std::span<const Graphics::GuestMemorySnapshot>, const std::shared_ptr<const DrawRecipe>&) {
    return RecipeOutcome::Rebuild;
}

VulkanDevice::IndirectDrawSupport VulkanDevice::DrawIndirectSupport() const {
    return {false, false, false};
}

bool VulkanDevice::PrebuildGraphicsLibraries(const Graphics::ShaderStages&, std::span<const Graphics::CompiledShader>) {
    return false;
}

Graphics::Context VulkanDevice::graphicsContext() const {
    Unimplemented("the Vulkan Graphics::Context", "Vulkan-only test hook");
}

}
