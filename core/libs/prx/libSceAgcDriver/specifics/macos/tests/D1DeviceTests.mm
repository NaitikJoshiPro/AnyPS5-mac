// Metal device tests of work package D1 (docs/mac/M2_DESIGN.md §9): the dispatch bindings (GDS,
// BDA page table, fault buffer and written-page record, device-memory LDS arena, DriverData),
// split and indirect dispatches, GPU labels, fills, copies and completion actions, and the startup
// self-tests. The kernels are hand-written MSL against the MslAbi contract, so these tests prove the
// driver side independently of the MetalBackend families.
//   agc_metal_device_tests bindings   (agc_metal_device_bindings)
//   agc_metal_device_tests submission (agc_metal_device_submission)
//   agc_metal_device_tests profile    (agc_metal_device_profile)
//   agc_metal_device_tests memory     (agc_metal_device_memory: the Context's device-local heap and its
//                                      VK_EXT_memory_budget report, MetalMemoryBudget.h)
//   agc_metal_device_tests teardown   (agc_metal_device_teardown: D19.5 teardown after a failed MSL
//                                      compile with work in flight, through ~VulkanDevice)
//   agc_metal_device_tests teardown_exit (agc_metal_device_teardown_exit: the same, the device still
//                                      alive at exit(); the exit handler tears it down)

#include "../src/MetalDeviceState.h"
#include "../src/MetalCommandStream.h"
#include "../src/MetalMemoryBudget.h"
#include "../src/MetalResidency.h"
#include "BdaAbi.hpp"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/WaitMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <atomic>
#include <chrono>
#include <thread>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

constexpr int Skipped = 77;

namespace Abi = ShaderRecompiler::MslAbi;
namespace RuntimeAbi = ShaderRecompiler::RuntimeAbi;
using ShaderRecompiler::DescriptorRole;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

// Guest memory registered with GuestAllocations, as the guest's own allocations are.
class GuestBlock {
public:
    GuestBlock(std::size_t bytes, bool writable) : size(bytes) {
        block = static_cast<std::uint32_t*>(std::aligned_alloc(16384, bytes));
        Require(block != nullptr, "cannot allocate a guest block");
        std::memset(block, 0, bytes);
        GuestAllocations::Mutation().Add(block, bytes, true, writable, true);
    }
    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
        std::free(block);
    }
    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;
    std::uint32_t* Words() const { return block; }
    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(block); }
    std::size_t Bytes() const { return size; }

private:
    std::uint32_t* block = nullptr;
    std::size_t size = 0;
};

std::vector<std::uint32_t> Buffer(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

struct TableSpec {
    RuntimeAbi::Binding binding;
    DescriptorRole role;
    std::uint32_t count;
    std::vector<std::uint32_t> guest;
};

const char* FieldName(RuntimeAbi::Binding binding) {
    switch (binding) {
    case RuntimeAbi::Binding::Buffers: return "buffers";
    case RuntimeAbi::Binding::Gds: return "gds";
    case RuntimeAbi::Binding::BdaPagetable: return "bda_pagetable";
    case RuntimeAbi::Binding::FaultBuffer: return "fault_buffer";
    case RuntimeAbi::Binding::FlattenedSrt: return "flattened_srt";
    case RuntimeAbi::Binding::ShaderData: return "shader_data";
    default: throw std::runtime_error("test: unexpected binding");
    }
}

// A kernel as MetalBackend would emit it: the MslAbi slots, a descriptor table laid out by
// MslAbi::LayoutTable, push data, the arena and DriverData. `body` sees `table`, `push`, `lds`,
// `driver`, `group`, `groups`, `local` and the helper `out(n)` (buffers[n] as device uint*).
ShaderRecompiler::RecompileResult Kernel(const std::string& body, std::vector<TableSpec> specs, std::array<std::uint32_t, 3> threads, std::uint32_t strideDwords = 0, std::vector<std::uint32_t> push = {}) {
    std::sort(specs.begin(), specs.end(), [](const TableSpec& left, const TableSpec& right) { return left.binding < right.binding; });
    std::vector<Abi::TableBinding> layout;
    std::string table = "struct Table {\n";
    for (const auto& spec : specs) {
        layout.push_back({spec.binding, spec.count});
        table += "    Buffer " + std::string(FieldName(spec.binding)) + "[" + std::to_string(spec.count) + "];\n";
    }
    table += "};\n";
    std::string source = R"(#include <metal_stdlib>
using namespace metal;
struct Buffer { ulong address; uint bytes; uint flags; };
struct DriverData { uint workgroupBase[3]; uint loopGuardLimit; ulong faultBuffer; ulong atomicLocks; };
)" + table + R"(#define out(n) reinterpret_cast<device uint*>(table.buffers[n].address)
kernel void aps5_main(device const Table& table [[buffer(0)]], constant uint* push [[buffer(1)]], device uint* lds [[buffer(2)]], constant DriverData& driver [[buffer(3)]],
                      uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]], uint local [[thread_index_in_threadgroup]]) {
)" + body + "\n}\n";
    Abi::Reflection reflection;
    reflection.stage = Abi::Stage::Kernel;
    reflection.group = RuntimeAbi::Stage::Main;
    reflection.threadsPerThreadgroup = threads;
    reflection.workgroupMemoryStrideDwords = strideDwords;
    reflection.usesPushData = true;
    reflection.usesDriverData = true;
    reflection.table = Abi::LayoutTable(layout);
    ShaderRecompiler::RecompileResult result;
    result.msl = ShaderRecompiler::SharedMsl(ShaderRecompiler::MslShader{source, reflection});
    for (const auto& spec : specs) {
        ShaderRecompiler::DescriptorBinding binding {ShaderRecompiler::DescriptorKind::StorageBuffer, spec.role, RuntimeAbi::DescriptorSet, RuntimeAbi::BindingNumber(RuntimeAbi::Stage::Main, spec.binding), spec.count, spec.guest};
        result.bindings.push_back(std::move(binding));
    }
    push.resize(RuntimeAbi::PushConstantDwords, 0u);
    result.pushConstants.resize(push.size() * 4u);
    std::memcpy(result.pushConstants.data(), push.data(), result.pushConstants.size());
    // The guest-buffer offset bytes (zeroed by the driver) live in the last push dword.
    result.memoryOffsetDword = RuntimeAbi::PushConstantDwords - 1u;
    result.workgroupMemoryDwords = strideDwords;
    return result;
}

std::uint32_t Low(std::uint64_t value) {
    return static_cast<std::uint32_t>(value);
}

std::uint32_t High(std::uint64_t value) {
    return static_cast<std::uint32_t>(value >> 32u);
}

template<typename TCall>
std::string Throws(TCall&& call) {
    try {
        call();
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

// Overlapping V#s over one unregistered array (one of them past the page the other was wrapped
// in by an earlier dispatch): every V# binds as an offset into one mapping, so sub-dword stores
// through either land in the same dword.
void CheckOverlappingBuffers(AgcDriver::VulkanDevice& device) {
    alignas(16384) static std::array<std::uint32_t, 8192> memory {};
    memory.fill(0u);
    alignas(16384) static std::array<std::uint32_t, 16> output {};
    output.fill(0xdeadbeefu);
    // An earlier dispatch wraps the first page alone.
    const auto first = Kernel("out(0)[0] = 1u;", {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(memory.data(), 64)}}, {1, 1, 1});
    device.Dispatch(first, 1, 1, 1);
    const auto kernel = Kernel(R"(
    if (local == 0u) {
        out(2)[0] = uint(table.buffers[1].address - table.buffers[0].address);
        out(2)[1] = uint((table.buffers[1].address - table.buffers[0].address) >> 32);
    }
    device atomic_uint* low = reinterpret_cast<device atomic_uint*>(table.buffers[0].address);
    device atomic_uint* high = reinterpret_cast<device atomic_uint*>(table.buffers[1].address);
    atomic_fetch_or_explicit(&low[10], 1u << local, memory_order_relaxed);
    atomic_fetch_or_explicit(&high[2], 0x10000u << local, memory_order_relaxed);)",
        {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 3, [&] {
            auto words = Buffer(memory.data(), 64);
            const auto second = Buffer(reinterpret_cast<const std::byte*>(memory.data()) + 32, 16384 + 64);
            const auto third = Buffer(output.data(), sizeof(output));
            words.insert(words.end(), second.begin(), second.end());
            words.insert(words.end(), third.begin(), third.end());
            return words;
        }()}}, {16, 1, 1});
    device.Dispatch(kernel, 1, 1, 1);
    device.WaitIdle();
    const auto delta = static_cast<std::uint64_t>(output[0]) | (static_cast<std::uint64_t>(output[1]) << 32u);
    Require(delta == 32u, "overlapping V#s: the second V# is not an offset into the first one's mapping (address delta " + Hex(delta) + ")");
    Require(memory[0] == 1u && memory[10] == 0xffffffffu, "overlapping V#s: the dword stored through both V#s is " + Hex(memory[10]));
    std::puts("overlapping V#s: one mapping, stores through both land in one dword");
}

// The BDA page table merge (CPU only): overlapping registered, V# and GDS ranges become sorted,
// disjoint entries; the first range listed keeps its device address, permissions are ORed.
void CheckDisjointRanges() {
    namespace Bda = ShaderRecompiler::BdaAbi;
    using Range = Bda::Range;
    const auto same = [](const std::vector<Range>& got, const std::vector<Range>& want) {
        if (got.size() != want.size()) return false;
        for (std::size_t index = 0; index < got.size(); ++index) {
            if (got[index].begin != want[index].begin || got[index].end != want[index].end || got[index].deviceAddress != want[index].deviceAddress || got[index].permissions != want[index].permissions) return false;
        }
        return true;
    };
    constexpr auto R = Bda::Read;
    constexpr auto RW = Bda::Read | Bda::Write;
    // Disjoint input, unsorted: sorted, unchanged; an empty range is dropped.
    const std::vector<Range> disjoint {{0x3000, 0x4000, 0x93000, R, 0}, {0x1000, 0x2000, 0x71000, RW, 0}, {0x5000, 0x5000, 0x1, R, 0}};
    Require(same(AgcDriver::Metal::DisjointRanges(disjoint), {{0x1000, 0x2000, 0x71000, RW, 0}, {0x3000, 0x4000, 0x93000, R, 0}}), "bda ranges: disjoint input changed");
    // A written V# inside a read-only registered range: split in three, W only in the middle,
    // device addresses of the registered (first) range throughout.
    const std::vector<Range> inside {{0x1000, 0x5000, 0xa1000, R, 0}, {0x2000, 0x3000, 0xf2000, RW, 0}};
    Require(same(AgcDriver::Metal::DisjointRanges(inside), {{0x1000, 0x2000, 0xa1000, R, 0}, {0x2000, 0x3000, 0xa2000, RW, 0}, {0x3000, 0x5000, 0xa3000, R, 0}}), "bda ranges: a range inside another is wrong");
    // A V# contained in a writable range with the same permissions disappears into it.
    const std::vector<Range> absorbed {{0x1000, 0x5000, 0xa1000, RW, 0}, {0x2000, 0x3000, 0xf2000, R, 0}};
    Require(same(AgcDriver::Metal::DisjointRanges(absorbed), {{0x1000, 0x5000, 0xa1000, RW, 0}}), "bda ranges: a contained range was not absorbed");
    // Partial overlap of two unrelated wrappers: the overlap keeps the first one's mapping, the
    // tail continues with the second's.
    const std::vector<Range> partial {{0x1000, 0x3000, 0xa1000, R, 0}, {0x2000, 0x4000, 0xb2000, R, 0}};
    Require(same(AgcDriver::Metal::DisjointRanges(partial), {{0x1000, 0x3000, 0xa1000, R, 0}, {0x3000, 0x4000, 0xb3000, R, 0}}), "bda ranges: a partial overlap is wrong");
    // Many random ranges: output sorted, disjoint, and every input address resolves to an output
    // range with at least its permissions.
    std::uint64_t seed = 12345;
    const auto next = [&] { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return seed >> 33u; };
    std::vector<Range> random;
    for (int index = 0; index < 200; ++index) {
        const auto begin = (next() % 4096u) * 4u;
        random.push_back({begin, begin + 4u + (next() % 256u) * 4u, 0x100000u * (index + 1u), (next() & 1u) != 0u ? RW : R, 0});
    }
    const auto merged = AgcDriver::Metal::DisjointRanges(random);
    for (std::size_t index = 1; index < merged.size(); ++index) Require(merged[index - 1u].end <= merged[index].begin, "bda ranges: the random output overlaps or is unsorted");
    for (const auto& range : random) {
        for (auto address = range.begin; address < range.end; address += 4u) {
            const auto found = std::upper_bound(merged.begin(), merged.end(), address, [](std::uint64_t value, const Range& entry) { return value < entry.begin; });
            Require(found != merged.begin() && address < std::prev(found)->end && (std::prev(found)->permissions & range.permissions) == range.permissions, "bda ranges: a random input address is not covered");
        }
    }
    std::printf("bda range merge: contained, partial and %zu random ranges -> %zu disjoint entries\n", random.size(), merged.size());
}

// GDS: the binding is the 64 KiB global data share in guest memory (Pm4::GdsAddress()), so PM4
// DMA packets and shaders see the same words.
void CheckGds(AgcDriver::VulkanDevice& device) {
    alignas(16384) static std::array<std::uint32_t, 256> output {};
    output.fill(0xdeadbeefu);
    auto* gds = reinterpret_cast<std::uint32_t*>(AgcDriver::Pm4::GdsAddress());
    gds[4] = 1000u;
    const auto kernel = Kernel(R"(
    device atomic_uint* gds = reinterpret_cast<device atomic_uint*>(table.gds[0].address);
    const uint index = group.x * 64u + local;
    out(0)[index] = atomic_fetch_add_explicit(&gds[push[0]], 1u, memory_order_relaxed);
    if (index == 0u) out(0)[128] = table.gds[0].bytes;)",
        {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(output.data(), sizeof(output))}, {RuntimeAbi::Binding::Gds, DescriptorRole::Gds, 1, {}}}, {64, 1, 1}, 0, {4u});
    device.Dispatch(kernel, 2, 1, 1);
    device.WaitIdle();
    Require(gds[4] == 1128u, "gds: the counter is " + std::to_string(gds[4]) + ", expected 1128");
    std::vector<std::uint32_t> values(output.begin(), output.begin() + 128);
    std::sort(values.begin(), values.end());
    for (std::uint32_t index = 0; index < 128u; ++index) Require(values[index] == 1000u + index, "gds: the returned values are not 1000..1127");
    Require(output[128] == AgcDriver::Pm4::GdsBytes, "gds: the binding has " + std::to_string(output[128]) + " bytes");
    std::puts("gds binding: 128 atomics on the guest GDS words");
}

// D3 host imports: a CPU seed, GPU atomics through the import of registered guest memory, the CPU
// reads the results after WaitIdle. Round 0 makes the import while the guest write watch has the
// page write-protected (agc_driver_gds_append_consume: the GPU's stores never reached the CPU's
// pages); rounds 1-2 reseed from the CPU once the import exists (the import is no longer watched).
// The last block is
// imported while the CPU maps it read-only (a GPU-writable, CPU read-only guest mapping).
int PageProtection(const void* page) {
    mach_vm_address_t region = reinterpret_cast<std::uintptr_t>(page);
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info {};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS) return -1;
    return info.protection;
}

class MappedGuestBlock {
public:
    explicit MappedGuestBlock(std::size_t bytes) : size(bytes) {
        void* mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        Require(mapped != MAP_FAILED, "imports: cannot map a guest block");
        block = static_cast<std::uint32_t*>(mapped);
        GuestAllocations::Mutation().Add(block, bytes, true, true, true);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
    }
    ~MappedGuestBlock() {
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, size);
        GuestAllocations::Mutation().Remove(block);
        munmap(block, size);
    }
    MappedGuestBlock(const MappedGuestBlock&) = delete;
    MappedGuestBlock& operator=(const MappedGuestBlock&) = delete;
    std::uint32_t* Words() const { return block; }
    void Arm() const {
        // The watch re-arms every page it reports as written (Collect), as it does between batches.
        GuestWriteWatch::GuestWriteWatchCollect_nid_postfix(reinterpret_cast<std::uintptr_t>(block), size, [](void*, std::uintptr_t, std::uintptr_t) {}, nullptr);
    }

private:
    std::uint32_t* block = nullptr;
    std::size_t size = 0;
};

void RunImportAtomics(AgcDriver::VulkanDevice& device, std::uint32_t* words, std::uint32_t seed, const std::string& what) {
    const auto kernel = Kernel(R"(
    device atomic_uint* counter = reinterpret_cast<device atomic_uint*>(out(0));
    const uint index = group.x * 64u + local;
    out(0)[16u + index] = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);)",
        {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(words, 4096)}}, {64, 1, 1});
    device.Dispatch(kernel, 2, 1, 1);
    device.WaitIdle();
    Require(words[0] == seed + 128u, "imports: " + what + ": the CPU reads the counter as " + std::to_string(words[0]) + " after WaitIdle, expected " + std::to_string(seed + 128u));
    std::vector<std::uint32_t> values(words + 16, words + 16 + 128);
    std::sort(values.begin(), values.end());
    for (std::uint32_t index = 0; index < 128u; ++index) Require(values[index] == seed + index, "imports: " + what + ": the atomics returned " + std::to_string(values[index]) + " at rank " + std::to_string(index) + ", expected " + std::to_string(seed + index));
}

void CheckImports(AgcDriver::VulkanDevice& device) {
    Require(GuestWriteWatch::GuestWriteWatchAvailable_nid_postfix(), "imports: the guest write watch is not available");
    {
        MappedGuestBlock block(65536);
        for (std::uint32_t round = 0; round < 3; ++round) {
            const std::uint32_t seed = 1000u * (round + 1u);
            block.Words()[0] = seed;
            if (round == 0u) {
                block.Arm();
                Require((PageProtection(block.Words()) & VM_PROT_WRITE) == 0, "imports: the write watch did not write-protect the page before the import");
            } else {
                // Metal imports are compared, not watched (GuestBufferMemory decideImportWatch).
                Require(!GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(reinterpret_cast<std::uintptr_t>(block.Words()), 65536), "imports: the imported block is still write-watched");
            }
            RunImportAtomics(device, block.Words(), seed, "write-watched round " + std::to_string(round));
        }
    }
    {
        MappedGuestBlock block(65536);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block.Words(), 65536);
        block.Words()[0] = 7000u;
        Require(mprotect(block.Words(), 65536, PROT_READ) == 0, "imports: cannot map the block read-only");
        RunImportAtomics(device, block.Words(), 7000u, "CPU read-only block");
        Require(PageProtection(block.Words()) == VM_PROT_READ, "imports: the CPU read-only block did not keep its protection");
        Require(mprotect(block.Words(), 65536, PROT_READ | PROT_WRITE) == 0, "imports: cannot restore the block");
    }
    std::puts("host imports: GPU atomics reach the CPU's pages (made over write-watched pages, reseeded, CPU read-only)");
}

// D44 (1): residency-set changes from another thread while the queue commits. The recorder's release
// thread removes and releases retired imports' buffers while the queue worker commits; a commit walks
// the attached sets (under the debug layer: applyResidencySets, which type-checks each allocation),
// so an unserialized release crashed it (SuperTuxKart under --validation). MetalResidency.h
// serializes both and keeps removed allocations alive until the next queue commit returned.
void CommitEmpty(id<MTLDevice> device, id<MTL4CommandQueue> queue, id<MTL4CommandAllocator> allocator, id<MTLSharedEvent> event, std::uint64_t value) {
    [allocator reset];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands endCommandBuffer];
    id<MTL4CommandBuffer> list[1] = {commands};
    AgcDriver::Metal::Residency::CommitQueue([&] {
        [queue commit:list count:1];
        [queue signalEvent:event value:value];
    });
    Require([event waitUntilSignaledValue:value timeoutMS:5000], "residency: an empty command buffer did not complete in 5 s");
}

// D67: a finished batch calls the recorder's completion listener (the driver's wake), and
// CompletionWakeNotify wakes a PollSleep-style waiter at once.
std::atomic<std::uint32_t> completionCalls{0};

void CheckCompletionWake(AgcDriver::VulkanDevice& device) {
    AgcDriver::Graphics::Recorder::SetCompletionListener(+[] { completionCalls.fetch_add(1, std::memory_order_relaxed); });
    GuestBlock block(16384, true);
    const std::array<std::uint32_t, 4> pattern {0x01020304u, 0xa0b0c0d0u, 0x55aa55aau, 0x00ff00ffu};
    Require(device.FillBuffer(block.Address(), 4096, pattern), "completion wake: the fill was refused");
    device.WaitIdle();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (completionCalls.load() == 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    AgcDriver::Graphics::Recorder::SetCompletionListener(nullptr);
    Require(completionCalls.load() != 0, "completion wake: a finished batch did not call the completion listener");
    Require(block.Words()[0] == pattern[0] && block.Words()[3] == pattern[3], "completion wake: the fill did not land");
    auto& wake = AgcDriver::DriverDetail::CompletionWakeState();
    std::uint64_t seen = 0;
    {
        std::lock_guard lock(wake.mutex);
        seen = wake.epoch;
    }
    std::atomic<bool> woke{false};
    std::thread waiter([&] {
        std::unique_lock lock(wake.mutex);
        woke = wake.changed.wait_for(lock, std::chrono::seconds(5), [&] { return wake.epoch != seen; });
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto notified = std::chrono::steady_clock::now();
    AgcDriver::DriverDetail::CompletionWakeNotify();
    waiter.join();
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - notified).count();
    Require(woke.load() && ms < 1000.0, "completion wake: the waiter was not woken by CompletionWakeNotify");
    std::printf("completion wake: %u listener calls; a waiter woke %.2f ms after the notify\n", completionCalls.load(), ms);
}

void CheckResidency(AgcDriver::VulkanDevice& device) {
    namespace Metal = AgcDriver::Graphics::Metal;
    namespace Residency = AgcDriver::Metal::Residency;
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    id<MTLDevice> metal = Metal::DeviceOf(context);
    id<MTL4CommandQueue> queue = Metal::QueueOf(context);
    Require(metal != nil && queue != nil, "residency: the device's Context carries no MTLDevice or queue");
    id<MTL4CommandAllocator> allocator = [metal newCommandAllocator];
    id<MTLSharedEvent> event = [metal newSharedEvent];
    NSError* error = nil;
    id<MTLResidencySet> set = [metal newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    Require(set != nil && allocator != nil && event != nil, "residency: cannot create the test objects");
    Residency::Attach(queue, set);
    std::uint64_t value = 0;
    // A removed allocation outlives its last driver reference until the next queue commit returned.
    __weak id<MTLBuffer> watched = nil;
    @autoreleasepool {
        id<MTLBuffer> buffer = [metal newBufferWithLength:16384 options:MTLResourceStorageModeShared];
        watched = buffer;
        Residency::Add(set, buffer);
        Residency::Commit(set);
        CommitEmpty(metal, queue, allocator, event, ++value);
        Residency::Remove(set, buffer);
        Residency::Commit(set);
        buffer = nil;
    }
    Require(watched != nil, "residency: a removed allocation was released before the next queue commit");
    CommitEmpty(metal, queue, allocator, event, ++value);
    Require(watched == nil, "residency: a removed allocation outlived the queue commit after its removal");
    // RemoveAll keeps every allocation of the set the same way; Drain releases them without a commit.
    __weak id<MTLBuffer> first = nil;
    __weak id<MTLBuffer> second = nil;
    @autoreleasepool {
        id<MTLBuffer> a = [metal newBufferWithLength:16384 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b = [metal newBufferWithLength:16384 options:MTLResourceStorageModeShared];
        first = a;
        second = b;
        Residency::Add(set, a);
        Residency::Add(set, b);
        Residency::Commit(set);
        Residency::RemoveAll(set);
        Residency::Commit(set);
    }
    Require(first != nil && second != nil, "residency: RemoveAll released its allocations before the next queue commit");
    Residency::Drain();
    Require(first == nil && second == nil, "residency: Drain did not release the removed allocations");
    // D67: CommitBeforeQueue defers a set's commit to the next queue commit (coalesced per set).
    {
        const auto before = Residency::Snapshot();
        id<MTLBuffer> deferredBuffer = [metal newBufferWithLength:16384 options:MTLResourceStorageModeShared];
        Residency::Add(set, deferredBuffer);
        Residency::CommitBeforeQueue(set);
        Residency::CommitBeforeQueue(set);
        Require(Residency::Snapshot().deferred == 1, "residency: a deferred set commit was not queued once");
        CommitEmpty(metal, queue, allocator, event, ++value);
        const auto after = Residency::Snapshot();
        Require(after.deferred == 0 && after.deferredCommits == before.deferredCommits + 1, "residency: the queue commit did not commit the deferred set");
        Require([set containsAllocation:deferredBuffer], "residency: the deferred allocation is not in its set");
        Residency::Remove(set, deferredBuffer);
        Residency::CommitBeforeQueue(set);
        CommitEmpty(metal, queue, allocator, event, ++value);
    }
    // The race itself: one thread adds, removes and releases buffers in an attached set while this
    // one commits. Without the lock the debug layer traps within seconds (scratch probe, D44).
    std::atomic<bool> stop{false};
    std::atomic<std::uint32_t> churned{0};
    std::thread churn([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
        while (std::chrono::steady_clock::now() < deadline) {
            @autoreleasepool {
                id<MTLBuffer> buffer = [metal newBufferWithLength:16384 options:MTLResourceStorageModeShared];
                Residency::Add(set, buffer);
                Residency::Commit(set);
                Residency::Remove(set, buffer);
                Residency::Commit(set);
            }
            churned.fetch_add(1, std::memory_order_relaxed);
        }
        stop.store(true, std::memory_order_release);
    });
    std::uint32_t commits = 0;
    while (!stop.load(std::memory_order_acquire)) {
        @autoreleasepool {
            CommitEmpty(metal, queue, allocator, event, ++value);
        }
        ++commits;
    }
    churn.join();
    CommitEmpty(metal, queue, allocator, event, ++value);
    Require(Residency::Snapshot().graveyard == 0, "residency: the graveyard is not empty after a queue commit");
    Residency::Detach(queue, set);
    std::printf("residency: removed allocations live until the next queue commit (Remove, RemoveAll, Drain); %u commits beside %u add/remove/release rounds on another thread\n", commits, churned.load());
}

// D44 (2): host imports join adjacent allocations only over whole 8 MiB windows on the Metal device,
// as on upstream's Vulkan device. A heap growing 16 KiB at a time re-imported the whole chain at every
// growth before (SuperTuxKart: 106 GB of imports in 60 s, ~20 GiB of retired no-copy buffers in one
// batch's residency, which Metal refused at commit with MTL4CommandQueueErrorTimeout). A retired
// import's bytes now also count against the open batch's kept-byte budget.
void CheckImportWindow(AgcDriver::VulkanDevice& device) {
    namespace Graphics = AgcDriver::Graphics;
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    Require(context.hostImportAlignment == 16384u, "import window: the Metal device does not import host memory in 16 KiB pages");
    constexpr std::size_t window = std::size_t{8} << 20u;
    constexpr std::size_t piece = 16384u;
    constexpr std::size_t pieces = window / piece + 64u;
    const std::size_t mapped = 3u * window;
    void* region = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    Require(region != MAP_FAILED, "import window: cannot map the heap");
    const auto base = (reinterpret_cast<std::uint64_t>(region) + window - 1u) / window * window;
    std::uint64_t imported = 0;
    std::uint64_t keptBefore = 0;
    std::uint64_t keptAfter = 0;
    const Graphics::HostImport* joined = nullptr;
    const Graphics::HostImport* next = nullptr;
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        auto* recorder = Graphics::Recorder::Active();
        Require(recorder != nullptr, "import window: the device has no active recorder");
        keptBefore = recorder->OpenKeptBytes();
        std::uint64_t lastBase = 0;
        std::uint64_t lastBytes = 0;
        const void* lastBuffer = nullptr;
        for (std::size_t index = 0; index < pieces; ++index) {
            auto* range = reinterpret_cast<void*>(base + index * piece);
            GuestAllocations::Mutation().Add(range, piece, true, true, true);
            // The newest range and the heap's first byte are both used by the GPU after each growth.
            const auto* newest = Graphics::HostImportFor(context, base + index * piece, 16);
            Require(newest != nullptr, "import window: range " + std::to_string(index) + " was not imported");
            const auto* front = Graphics::HostImportFor(context, base, 16);
            Require(front != nullptr, "import window: the heap's first range lost its import");
            if (front->base != lastBase || front->bytes != lastBytes || reinterpret_cast<const void*>(front->buffer) != lastBuffer) {
                imported += front->bytes;
                lastBase = front->base;
                lastBytes = front->bytes;
                lastBuffer = reinterpret_cast<const void*>(front->buffer);
            }
        }
        joined = Graphics::HostImportFor(context, base, 16);
        next = Graphics::HostImportFor(context, base + window, 16);
        Require(joined != nullptr && joined->base == base && joined->bytes == window, "import window: a full 8 MiB window is not one import (" + std::to_string(joined != nullptr ? joined->bytes : 0) + " bytes)");
        Require(next != nullptr && next->base == base + window && next->bytes == piece, "import window: a range of the next, partly filled window joined a span (" + std::to_string(next != nullptr ? next->bytes : 0) + " bytes)");
        keptAfter = recorder->OpenKeptBytes();
    }
    // Unbounded spans re-import the front's whole chain at each growth: ~pieces^2 / 2 * 16 KiB (> 2 GiB).
    Require(imported <= 2u * window, "import window: the heap's front was re-imported " + std::to_string(imported >> 20u) + " MiB over " + std::to_string(pieces) + " growths");
    Require(keptAfter >= keptBefore + window / 2u, "import window: retired imports did not count against the open batch's kept bytes (" + std::to_string(keptBefore) + " -> " + std::to_string(keptAfter) + ")");
    device.WaitIdle();
    for (std::size_t index = 0; index < pieces; ++index) GuestAllocations::Mutation().Remove(reinterpret_cast<void*>(base + index * piece));
    munmap(region, mapped);
    // Two adjacent allocations that each cross a window boundary (a 3840x2160 surface allocated in two
    // GL buffers, Showcase 1.0.2) join one import, so a resource spanning both has one (D65): with
    // each imported alone, nothing held the surface and its results never reached guest memory.
    {
        const std::size_t large = window * 3u / 2u;
        const std::size_t span = 2u * large;
        void* bigRegion = mmap(nullptr, span + window, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        Require(bigRegion != MAP_FAILED, "import window: cannot map the two large allocations");
        const auto bigBase = (reinterpret_cast<std::uint64_t>(bigRegion) + window - 1u) / window * window;
        GuestAllocations::Mutation().Add(reinterpret_cast<void*>(bigBase), large, true, true, true);
        GuestAllocations::Mutation().Add(reinterpret_cast<void*>(bigBase + large), large, true, true, true);
        const Graphics::HostImport* spanning = nullptr;
        const Graphics::HostImport* fromSecond = nullptr;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            spanning = Graphics::HostImportFor(context, bigBase + large - 4096u, 8192u);
            fromSecond = Graphics::HostImportFor(context, bigBase + large + 4096u, 16u);
        }
        Require(spanning != nullptr && spanning->base == bigBase && spanning->bytes == span, "import window: a range across two adjacent large allocations has no single import (" + std::to_string(spanning != nullptr ? spanning->bytes : 0) + " bytes)");
        Require(fromSecond == spanning, "import window: the second large allocation is not served by the joined import");
        device.WaitIdle();
        GuestAllocations::Mutation().Remove(reinterpret_cast<void*>(bigBase));
        GuestAllocations::Mutation().Remove(reinterpret_cast<void*>(bigBase + large));
        munmap(bigRegion, span + window);
    }
    std::printf("import window: %zu ranges of 16 KiB: the front re-imported %llu KiB in all, a full window is one 8 MiB import, the next window's range stays alone, adjacent large allocations share one import; retired imports kept %llu KiB in the open batch\n", pieces, static_cast<unsigned long long>(imported >> 10u), static_cast<unsigned long long>((keptAfter - keptBefore) >> 10u));
}

// BDA page table: BdaAbi::Header + Range[] over the registered guest allocations, translated in
// the shader like MetalBackend's lookup, with permissions from the registry.
void CheckAddressTable(AgcDriver::VulkanDevice& device) {
    alignas(16384) static std::array<std::uint32_t, 64> output {};
    output.fill(0xdeadbeefu);
    GuestBlock writable(65536, true);
    GuestBlock readOnly(16384, false);
    writable.Words()[100] = 0x1234u;
    readOnly.Words()[7] = 0x5678u;
    const auto guest = writable.Address() + 400u;
    const auto guestRo = readOnly.Address() + 28u;
    const auto kernel = Kernel(R"(
    struct Range { ulong begin; ulong end; ulong gpu; uint permissions; uint reserved; };
    device const uint* header = reinterpret_cast<device const uint*>(table.bda_pagetable[0].address);
    device const Range* ranges = reinterpret_cast<device const Range*>(header + 4);
    const ulong guest = ulong(push[0]) | (ulong(push[1]) << 32);
    const ulong guestRo = ulong(push[2]) | (ulong(push[3]) << 32);
    out(0)[0] = header[0];
    out(0)[1] = header[1];
    out(0)[2] = header[2];
    out(0)[3] = table.bda_pagetable[0].bytes;
    uint found = 0u;
    for (uint index = 0u; index < header[1]; ++index) {
        if (index != 0u && ranges[index].begin < ranges[index - 1u].end) found |= 4u;
        if (guest >= ranges[index].begin && guest < ranges[index].end) {
            device uint* word = reinterpret_cast<device uint*>(ranges[index].gpu + (guest - ranges[index].begin));
            out(0)[4] = word[0];
            word[1] = word[0] + 1u;
            out(0)[5] = ranges[index].permissions;
            found |= 1u;
        }
        if (guestRo >= ranges[index].begin && guestRo < ranges[index].end) {
            out(0)[6] = reinterpret_cast<device const uint*>(ranges[index].gpu + (guestRo - ranges[index].begin))[0];
            out(0)[7] = ranges[index].permissions;
            found |= 2u;
        }
    }
    // The bound V# (a static, unregistered array) is in the table too, writable (upstream's
    // AddressRanges): store through its translation.
    const ulong bound = ulong(push[4]) | (ulong(push[5]) << 32);
    for (uint index = 0u; index < header[1]; ++index) {
        if (bound >= ranges[index].begin && bound < ranges[index].end) {
            reinterpret_cast<device uint*>(ranges[index].gpu + (bound - ranges[index].begin))[9] = 0x77u + ranges[index].permissions;
            found |= 8u;
        }
    }
    out(0)[8] = found;)",
        {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(output.data(), sizeof(output))}, {RuntimeAbi::Binding::BdaPagetable, DescriptorRole::BdaPagetable, 1, {}}}, {1, 1, 1}, 0, {Low(guest), High(guest), Low(guestRo), High(guestRo), Low(reinterpret_cast<std::uintptr_t>(output.data())), High(reinterpret_cast<std::uintptr_t>(output.data()))});
    device.Dispatch(kernel, 1, 1, 1);
    device.WaitIdle();
    namespace Bda = ShaderRecompiler::BdaAbi;
    Require(output[0] == Bda::Version && output[2] == sizeof(Bda::Range), "bda table: bad header (version " + std::to_string(output[0]) + ", entry bytes " + std::to_string(output[2]) + ")");
    Require(output[3] == sizeof(Bda::Header) + output[1] * sizeof(Bda::Range), "bda table: the binding's size disagrees with the header's count");
    Require(output[8] == 11u, "bda table: the guest blocks or the bound V# were not found or the ranges are unsorted (flags " + std::to_string(output[8]) + ")");
    Require(output[9] == 0x77u + (Bda::Read | Bda::Write), "bda table: the store through the bound V#'s translation read " + Hex(output[9]));
    Require(output[4] == 0x1234u && writable.Words()[101] == 0x1235u, "bda table: the translated load/store of the writable block failed");
    Require(output[5] == (Bda::Read | Bda::Write), "bda table: the writable block has permissions " + std::to_string(output[5]));
    Require(output[6] == 0x5678u && output[7] == Bda::Read, "bda table: the read-only block read " + Hex(output[6]) + " with permissions " + std::to_string(output[7]));
    std::printf("bda page table: %u ranges, translated load and store, read-only permissions\n", output[1]);
}

// MslAbi v2: DriverData.atomicLocks is the device's zero-filled AtomicLockTableBytes table, the
// same for every dispatch; a kernel can take and release a lock word in it. Only the lock words
// (AtomicLockWords) must be free: the emulated shader clock counts in the words after them
// (AtomicLockAllocationBytes).
void CheckAtomicLocks(AgcDriver::VulkanDevice& device) {
    alignas(16384) static std::array<std::uint32_t, 64> output {};
    output.fill(0xdeadbeefu);
    const auto kernel = Kernel(R"(
    device atomic_uint* locks = reinterpret_cast<device atomic_uint*>(driver.atomicLocks);
    if (local == 0u) {
        out(0)[group.x * 4u] = uint(driver.atomicLocks);
        out(0)[group.x * 4u + 1u] = uint(driver.atomicLocks >> 32);
        uint nonzero = 0u;
        for (uint index = 0u; index < )" + std::to_string(Abi::AtomicLockWords) + R"(u; ++index) nonzero += atomic_load_explicit(&locks[index], memory_order_relaxed) != 0u ? 1u : 0u;
        out(0)[group.x * 4u + 2u] = nonzero;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (local == 0u && group.x == 0u) {
        uint expected = 0u;
        const bool taken = atomic_compare_exchange_weak_explicit(&locks[7], &expected, 1u, memory_order_relaxed, memory_order_relaxed);
        atomic_store_explicit(&locks[7], 0u, memory_order_relaxed);
        out(0)[3] = taken ? 1u : 0u;
    })",
        {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(output.data(), sizeof(output))}}, {32, 1, 1});
    device.Dispatch(kernel, 1, 1, 1);
    device.WaitIdle();
    device.Dispatch(kernel, 2, 1, 1);
    device.WaitIdle();
    const auto address = static_cast<std::uint64_t>(output[0]) | (static_cast<std::uint64_t>(output[1]) << 32u);
    const auto second = static_cast<std::uint64_t>(output[4]) | (static_cast<std::uint64_t>(output[5]) << 32u);
    Require(address != 0u && address == second, "atomic locks: DriverData.atomicLocks is " + Hex(address) + " / " + Hex(second));
    Require(output[2] == 0u && output[6] == 0u, "atomic locks: the lock table is not all released (" + std::to_string(output[2]) + " busy words)");
    Require(output[3] == 1u, "atomic locks: a free lock word could not be taken");
    std::puts("atomic lock table: bound in DriverData, zero-filled, released after use");
}

// Fault buffer: a published fault record is reported (stderr, as upstream) and the buffer is reset;
// a written-page record is reported to the write tracker and cleared; DriverData.faultBuffer and the
// table entry agree; a tripped loop guard is reported, not raised.
void CheckFaults(AgcDriver::VulkanDevice& device) {
    namespace Bda = ShaderRecompiler::BdaAbi;
    alignas(16384) static std::array<std::uint32_t, 64> output {};
    output.fill(0xdeadbeefu);
    const auto record = [&](std::uint32_t reason, std::uint64_t address) {
        return Kernel(R"(
    device uint* words = reinterpret_cast<device uint*>(table.fault_buffer[0].address);
    out(0)[0] = driver.faultBuffer == table.fault_buffer[0].address ? 1u : 0u;
    out(0)[1] = table.fault_buffer[0].bytes;
    out(0)[2] = words[0] | words[)" + std::to_string(Bda::WrittenCountWord) + R"(];
    if (push[0] != 0u) {
        words[1] = push[0];
        words[2] = push[1];
        words[3] = push[2];
        words[4] = 4u;
        words[5] = 4u;
        words[6] = 0x123u;
        words[0] = 2u;
    }
    if (push[3] != 0u) {
        words[)" + std::to_string(Bda::WrittenSlotsWord) + R"( + 2u * 5u] = push[3];
        words[)" + std::to_string(Bda::WrittenSlotsWord) + R"( + 2u * 5u + 1u] = 1u;
        words[)" + std::to_string(Bda::WrittenListWord) + R"(] = 5u;
        words[)" + std::to_string(Bda::WrittenCountWord) + R"(] = 1u;
    })",
            {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(output.data(), sizeof(output))}, {RuntimeAbi::Binding::FaultBuffer, DescriptorRole::FaultBuffer, 1, {}}}, {1, 1, 1}, 0, {reason, Low(address), High(address), 0u});
    };
    device.Dispatch(record(static_cast<std::uint32_t>(Bda::FaultReason::Unmapped), 0xabc0001000ull), 1, 1, 1);
    const auto fault = Throws([&] { device.WaitIdle(); });
    // As upstream: the record is reported on stderr by the batch's completion, the wait succeeds.
    Require(fault.empty(), "faults: the unmapped fault failed the wait: " + fault);
    Require(output[0] == 1u && output[1] == Bda::FaultBufferBytes, "faults: DriverData.faultBuffer or the table entry is wrong");
    device.Dispatch(record(static_cast<std::uint32_t>(Bda::FaultReason::LoopLimit), 0x5eedull), 1, 1, 1);
    const auto guard = Throws([&] { device.WaitIdle(); });
    Require(guard.empty(), "faults: the loop guard failed the wait: " + guard);
    // A written-page record over a guest block: reported and cleared, so the next use starts empty.
    GuestBlock block(65536, true);
    auto written = record(0u, 0u);
    reinterpret_cast<std::uint32_t*>(written.pushConstants.data())[3] = Bda::WrittenChunkKey(block.Address());
    device.Dispatch(written, 1, 1, 1);
    device.WaitIdle();
    device.Dispatch(record(0u, 0u), 1, 1, 1);
    device.WaitIdle();
    Require(output[2] == 0u, "faults: the fault buffer was not reset after a fault and a written-page record (" + Hex(output[2]) + ")");
    std::puts("fault buffer: fault and loop-guard records reported, written pages reported, buffer reset");
}

// Device-memory LDS: workgroup w of the grid owns dwords [w*stride, (w+1)*stride), the last one the
// lock (zero before use); the arena grows past its first 1 MiB granule.
void CheckWorkgroupArena(AgcDriver::VulkanDevice& device) {
    alignas(16384) static std::array<std::uint32_t, 256> output {};
    for (const auto& [stride, grid] : {std::pair<std::uint32_t, std::array<std::uint32_t, 3>>{11u, {3u, 2u, 2u}}, {8193u, {5u, 4u, 2u}}}) {
        output.fill(0xdeadbeefu);
        const auto kernel = Kernel(R"(
    const uint stride = )" + std::to_string(stride) + R"(u;
    const uint w = (group.z * groups.y + group.y) * groups.x + group.x;
    device uint* slot = lds + w * stride;
    if (local == 0u) out(0)[w * 2u + 1u] = slot[stride - 1u];
    threadgroup_barrier(mem_flags::mem_device);
    for (uint index = local; index < stride - 1u; index += 32u) slot[index] = w * 100u + index;
    threadgroup_barrier(mem_flags::mem_device);
    if (local == 0u) {
        uint sum = 0u;
        for (uint index = 0u; index < 10u; ++index) sum += slot[index];
        sum += slot[stride - 2u];
        out(0)[w * 2u] = sum;
    })",
            {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(output.data(), sizeof(output))}}, {32, 1, 1}, stride);
        device.Dispatch(kernel, grid[0], grid[1], grid[2]);
        device.WaitIdle();
        const auto groups = grid[0] * grid[1] * grid[2];
        for (std::uint32_t w = 0; w < groups; ++w) {
            const auto expected = 1000u * w + 45u + w * 100u + (stride - 2u);
            Require(output[w * 2u] == expected, "arena: workgroup " + std::to_string(w) + " of stride " + std::to_string(stride) + " summed " + std::to_string(output[w * 2u]) + ", expected " + std::to_string(expected));
            Require(output[w * 2u + 1u] == 0u, "arena: the lock dword of workgroup " + std::to_string(w) + " started at " + Hex(output[w * 2u + 1u]));
        }
    }
    // A split would alias the arena slots: refused like upstream.
    const auto kernel = Kernel("", {}, {32, 1, 1}, 11u);
    const auto refused = Throws([&] { device.Dispatch(kernel, 1, device.WorkGroupCountLimit()[1] + 1u, 1); });
    device.WaitIdle();
    Require(refused.find("aliases the device-memory LDS") != std::string::npos, "arena: a split dispatch with device-memory LDS gave '" + refused + "'");
    std::puts("device-memory LDS arena: per-workgroup slots, zeroed locks, growth, split refused");
}

ShaderRecompiler::RecompileResult CounterKernel(std::array<std::uint32_t, 4>& counter) {
    return Kernel(R"(
    device atomic_uint* counter = reinterpret_cast<device atomic_uint*>(out(0));
    atomic_fetch_add_explicit(&counter[0], 1u, memory_order_relaxed);
    const uint x = group.x + driver.workgroupBase[0];
    const uint y = group.y + driver.workgroupBase[1];
    atomic_fetch_max_explicit(&counter[1], x, memory_order_relaxed);
    atomic_fetch_max_explicit(&counter[2], y, memory_order_relaxed);)",
        {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(counter.data(), sizeof(counter))}}, {1, 1, 1});
}

// Grids over WorkGroupCountLimit are split; each chunk's first workgroup is DriverData.workgroupBase.
void CheckSplit(AgcDriver::VulkanDevice& device) {
    alignas(16384) static std::array<std::uint32_t, 4> counter {};
    const auto limit = device.WorkGroupCountLimit()[0];
    for (const auto& [x, y] : {std::pair<std::uint32_t, std::uint32_t>{limit * 2u + 7u, 1u}, {3u, limit + 2u}}) {
        counter.fill(0u);
        device.Dispatch(CounterKernel(counter), x, y, 1);
        device.WaitIdle();
        Require(counter[0] == x * y && counter[1] == x - 1u && counter[2] == y - 1u, "split: " + std::to_string(x) + "x" + std::to_string(y) + " ran " + std::to_string(counter[0]) + " workgroups, max x " + std::to_string(counter[1]) + " y " + std::to_string(counter[2]));
    }
    std::puts("split dispatch: grids over the workgroup limit run every workgroup once");
}

// DISPATCH_INDIRECT: counts written by an earlier dispatch are read by the GPU (cpuReason 0); a
// shader with device-memory LDS reads them on the CPU (8).
void CheckIndirect(AgcDriver::VulkanDevice& device) {
    alignas(16384) static std::array<std::uint32_t, 4> arguments {};
    alignas(16384) static std::array<std::uint32_t, 4> counter {};
    arguments = {1u, 1u, 1u, 0u};
    counter.fill(0u);
    const auto writer = Kernel("out(0)[0] = 4u; out(0)[1] = 3u; out(0)[2] = 2u;", {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(arguments.data(), sizeof(arguments))}}, {1, 1, 1});
    device.Dispatch(writer, 1, 1, 1);
    const auto outcome = device.DispatchIndirect(CounterKernel(counter), reinterpret_cast<std::uintptr_t>(arguments.data()));
    device.WaitIdle();
    Require(outcome.cpuReason == 0, "indirect: the arguments were read on the CPU (reason " + std::to_string(outcome.cpuReason) + ")");
    Require(counter[0] == 24u && counter[1] == 3u && counter[2] == 2u, "indirect: ran " + std::to_string(counter[0]) + " workgroups, expected the 4x3x2 an earlier dispatch wrote (the arguments hold " + std::to_string(arguments[0]) + "x" + std::to_string(arguments[1]) + "x" + std::to_string(arguments[2]) + ")");
    alignas(16384) static std::array<std::uint32_t, 64> output {};
    output.fill(0u);
    const auto lds = Kernel("if (local == 0u) atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(out(0)), 1u, memory_order_relaxed);", {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(output.data(), sizeof(output))}}, {32, 1, 1}, 11u);
    const auto cpu = device.DispatchIndirect(lds, reinterpret_cast<std::uintptr_t>(arguments.data()));
    device.WaitIdle();
    Require(cpu.cpuReason == 8 && output[0] == 24u, "indirect: a device-memory LDS shader gave reason " + std::to_string(cpu.cpuReason) + " and " + std::to_string(output[0]) + " workgroups");
    std::puts("indirect dispatch: GPU-read counts after a producer dispatch; CPU-read counts for device-memory LDS");
}

// Labels, fills, copies and completion actions are ordered after the work committed before them.
void CheckSubmission(AgcDriver::VulkanDevice& device) {
    // Labels, fills and copies go to the GPU only in registered guest allocations (upstream's host
    // imports); other memory (a static page here) is left to the PM4 core's drain-and-CPU path.
    alignas(16384) static std::array<std::uint32_t, 1024> unregistered {};
    GuestBlock block(16384, true);
    const std::span<std::uint32_t, 1024> memory(block.Words(), 1024);
    const auto writeAll = Kernel("for (uint index = 0u; index < 1024u; ++index) out(0)[index] = 0x11111111u;", {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(memory.data(), 4096u)}}, {1, 1, 1});
    const std::array<std::uint32_t, 2> label {0xbbbb0001u, 0xbbbb0002u};
    const auto labelAddress = reinterpret_cast<std::uintptr_t>(&memory[8]);
    device.WaitIdle();
    Require(device.WriteLabelOnGpu(labelAddress, std::as_bytes(std::span(label)), 1, 0) == 1, "labels: an idle device did not leave the label to the CPU");
    Require(device.WriteLabelOnGpu(labelAddress + 2u, std::as_bytes(std::span(label)), 1, 0) == 4, "labels: a misaligned label was accepted");
    device.Dispatch(writeAll, 1, 1, 1);
    Require(device.WriteLabelOnGpu(labelAddress, std::as_bytes(std::span(label)), 2, 0, false) == 0, "labels: a label behind recorded work was not stored on the GPU");
    const auto staticAddress = reinterpret_cast<std::uintptr_t>(unregistered.data());
    Require(device.WriteLabelOnGpu(staticAddress, std::as_bytes(std::span(label)), 3, 0, false) == 3, "labels: a label into unregistered memory was not left to the drain path");
    const std::array<std::uint32_t, 4> zeros {};
    Require(!device.FillBuffer(staticAddress, 64, zeros), "fills: a fill into unregistered memory was not refused");
    Require(device.CopyBuffer(staticAddress, reinterpret_cast<std::uintptr_t>(&memory[64]), 64, 0, 1u << 20u, 0, 0, 0, [](std::span<const std::byte>, std::uint64_t) {}).path == 2, "copies: a copy into unregistered memory was not refused");
    const std::array<std::uint32_t, 4> pattern {0x01020304u, 0xa0b0c0d0u, 0x55aa55aau, 0x00ff00ffu};
    Require(device.FillBuffer(reinterpret_cast<std::uintptr_t>(&memory[64]), 256, pattern), "fills: the pattern fill was refused");
    const std::array<std::uint32_t, 4> uniform {0x05050505u, 0x05050505u, 0x05050505u, 0x05050505u};
    Require(device.FillBuffer(reinterpret_cast<std::uintptr_t>(&memory[192]), 64, uniform), "fills: the uniform fill was refused");
    Require(!device.FillBuffer(reinterpret_cast<std::uintptr_t>(&memory[1]), 64, uniform), "fills: a misaligned fill was accepted");
    const auto copy = device.CopyBuffer(reinterpret_cast<std::uintptr_t>(&memory[512]), reinterpret_cast<std::uintptr_t>(&memory[64]), 256, 0, 1u << 20u, 0, 0, 0, [](std::span<const std::byte>, std::uint64_t) {});
    Require(copy.path == 1, "copies: a copy behind recorded work took path " + std::to_string(copy.path));
    const auto oddCopy = device.CopyBuffer(reinterpret_cast<std::uintptr_t>(&memory[700]) + 1u, reinterpret_cast<std::uintptr_t>(&memory[64]) + 3u, 9, 0, 1u << 20u, 0, 0, 0, [](std::span<const std::byte>, std::uint64_t) {});
    Require(oddCopy.path == 1, "copies: an unaligned copy took path " + std::to_string(oddCopy.path));
    // A CPU read through GuestMemory waits for the GPU stores into its range (the recorder's
    // flush hook over the pending writes the device noted), without WaitIdle.
    std::array<std::uint32_t, 4> read {};
    AgcDriver::GuestMemory::Read(reinterpret_cast<std::uintptr_t>(&memory[512]), std::as_writable_bytes(std::span(read)), 4);
    Require(read == pattern, "flush hook: a GuestMemory read did not wait for the GPU copy into it (" + Hex(read[0]) + ")");
    // A completion action (an EOP interrupt) runs when the work recorded before it completed.
    Require(device.FillBuffer(reinterpret_cast<std::uintptr_t>(&memory[960]), 64, pattern), "fills: the fill before the completion was refused");
    bool completed = false;
    std::uint32_t seenAtCompletion = 0;
    Require(device.AfterRecordedWork([&] {
        completed = true;
        seenAtCompletion = memory[960];
    }, false), "completions: an action behind recorded work was not deferred");
    device.WaitIdle();
    Require(completed && seenAtCompletion == pattern[0], "completions: the action did not run after the work before it (" + Hex(seenAtCompletion) + ")");
    Require(memory[8] == label[0] && memory[9] == label[1] && memory[7] == 0x11111111u && memory[10] == 0x11111111u, "labels: the label did not land after the dispatch (" + Hex(memory[8]) + ")");
    for (std::uint32_t index = 0; index < 64u; ++index) Require(memory[64u + index] == pattern[index % 4u], "fills: dword " + std::to_string(index) + " of the pattern fill is " + Hex(memory[64u + index]));
    for (std::uint32_t index = 0; index < 16u; ++index) Require(memory[192u + index] == 0x05050505u, "fills: the uniform fill is wrong");
    Require(memory[208] == 0x11111111u, "fills: the uniform fill overran");
    for (std::uint32_t index = 0; index < 64u; ++index) Require(memory[512u + index] == pattern[index % 4u], "copies: the GPU copy is wrong at dword " + std::to_string(index));
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(memory.data());
    const auto* source = bytes + 64u * 4u + 3u;
    Require(std::memcmp(bytes + 700u * 4u + 1u, source, 9) == 0 && bytes[700u * 4u] == 0x11u && bytes[700u * 4u + 10u] == 0x11u, "copies: the unaligned GPU copy is wrong");
    Require(!device.AfterRecordedWork([] {}, true), "completions: an idle device deferred an action");
    const auto cpuCopy = device.CopyBuffer(reinterpret_cast<std::uintptr_t>(&memory[800]), reinterpret_cast<std::uintptr_t>(&memory[64]), 64, 1024, 1u << 20u, 0, 0, 0, [](std::span<const std::byte>, std::uint64_t) {});
    Require(cpuCopy.path == 0 && std::memcmp(&memory[800], &memory[64], 64) == 0, "copies: an idle small copy took path " + std::to_string(cpuCopy.path));
    const auto bigCopy = device.CopyBuffer(reinterpret_cast<std::uintptr_t>(&memory[800]), reinterpret_cast<std::uintptr_t>(&memory[64]), 64, 0, 32, 0, 0, 0, [](std::span<const std::byte>, std::uint64_t) {});
    Require(bigCopy.path == 8, "copies: a copy over gpuMax took path " + std::to_string(bigCopy.path));
    std::puts("submission: GPU labels, pattern and uniform fills, aligned and unaligned copies, flush-hook reads and completions ordered after recorded work");
}

void CheckProfile(AgcDriver::VulkanDevice& device) {
    id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
    const auto& profile = AgcDriver::Metal::Profile(metal);
    std::printf("%s", profile.summary.c_str());
    Require(profile.measured, "profile: the self-tests did not run: " + profile.failure);
    Require(profile.barycentrics, "profile: [[barycentric_coord]] did not weight a triangle list in vertex order");
    Require(profile.oddStripOrder != AgcDriver::Metal::DeviceProfile::StripOrder::Unknown, "profile: the strip order was not decided");
    const bool vertex = (device.SubgroupStages() & VK_SHADER_STAGE_VERTEX_BIT) != 0u;
    Require(vertex == profile.vertexSimdGroups, "profile: SubgroupStages() disagrees with the vertex SIMD-group self-test");
    std::printf("device profile: vertex SIMD groups %s, odd strip order %s\n", profile.vertexSimdGroups ? "yes" : "no", profile.oddStripOrder == AgcDriver::Metal::DeviceProfile::StripOrder::I_I2_I1 ? "(v_i, v_i+2, v_i+1)" : "other");
}

// DECISIONS D19.5: a device torn down after a failed MSL compile, with a dispatch in flight and a
// static (__DATA) page wrapped, waits for its batches, drops its wrappers and removes its residency
// set from the queue. With `viaExit` the device is still alive when the process calls exit(); the
// driver's exit handler tears it down (checked by the test's PASS_REGULAR_EXPRESSION on its log).
void CheckTeardown(std::unique_ptr<AgcDriver::VulkanDevice>& device, bool viaExit) {
    alignas(16384) static std::array<std::uint32_t, 64> output {};
    output.fill(0u);
    const std::vector<TableSpec> specs {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(output.data(), sizeof(output))}};
    const auto good = Kernel("if (local == 0u) atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(out(0)), 1u, memory_order_relaxed);", specs, {32, 1, 1});
    device->Dispatch(good, 4, 1, 1);
    device->WaitIdle();
    Require(output[0] == 4u, "teardown: the first dispatch ran " + std::to_string(output[0]) + " workgroups, expected 4");
    const auto broken = Kernel("out(0)[0] = aps5_teardown_undefined_identifier;", specs, {32, 1, 1});
    const auto failure = Throws([&] { device->Dispatch(broken, 1, 1, 1); });
    Require(!failure.empty(), "teardown: a kernel whose MSL does not compile was accepted");
    std::printf("teardown: the broken kernel was refused: %.120s\n", failure.c_str());
    // Left in flight for the teardown to wait for.
    device->Dispatch(good, 8, 1, 1);
    if (viaExit) {
        std::puts("teardown: exiting with the device alive");
        std::fflush(stdout);
        std::exit(0);
    }
    const auto before = AgcDriver::Metal::LastTeardown().teardowns;
    device.reset();
    const auto report = AgcDriver::Metal::LastTeardown();
    Require(report.teardowns == before + 1u && !report.atExit, "teardown: ~VulkanDevice did not tear the device down");
    Require(report.idle && report.syncFailures == 0u, "teardown: the recorder was not idle before the release (" + std::to_string(report.syncFailures) + " failed syncs)");
    Require(report.residencyRemoved, "teardown: the device's residency set was not removed from the queue");
    Require(report.wrappersDropped >= 1u, "teardown: no guest-memory wrapper was dropped (the static page was not wrapped?)");
    Require(output[0] == 12u, "teardown: the dispatch in flight at teardown did not complete (" + std::to_string(output[0]) + " workgroups)");
    std::printf("teardown: waited for the in-flight dispatch, dropped %llu wrapper(s), removed the residency set\n", static_cast<unsigned long long>(report.wrappersDropped));
}

}

// The device's Context describes the device-local heap of a unified-memory Mac (the recommended working set) and
// answers upstream's VK_EXT_memory_budget query, so the texture caches size themselves by it (D5).
void CheckMemory(AgcDriver::VulkanDevice& device) {
    namespace Metal = AgcDriver::Graphics::Metal;
    using Metal::MemoryBudget;
    using Metal::MemoryPressure;
    constexpr std::uint64_t GiB = 1ull << 30u;
    // The budget rule: the working set bounds it, usage plus the system's headroom (halved at WARN, none at CRITICAL).
    Require(MemoryBudget(10 * GiB, 1 * GiB, 4 * GiB, MemoryPressure::Normal) == 5 * GiB, "the budget is not usage + headroom");
    Require(MemoryBudget(10 * GiB, 1 * GiB, 4 * GiB, MemoryPressure::Warn) == 3 * GiB, "memory pressure WARN does not halve the headroom");
    Require(MemoryBudget(10 * GiB, 1 * GiB, 4 * GiB, MemoryPressure::Critical) == 1 * GiB, "memory pressure CRITICAL leaves headroom");
    Require(MemoryBudget(10 * GiB, 1 * GiB, UINT64_MAX, MemoryPressure::Normal) == 10 * GiB, "the working set does not bound the budget");
    Require(MemoryBudget(10 * GiB, 12 * GiB, 4 * GiB, MemoryPressure::Normal) == 10 * GiB, "usage past the working set raises the budget");

    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    id<MTLDevice> metal = Metal::DeviceOf(context);
    Require(metal != nil, "the device's Context carries no MTLDevice");
    const auto workingSet = static_cast<std::uint64_t>(metal.recommendedMaxWorkingSetSize);
    Require(context.memory.memoryHeapCount == 1u, "the Context describes " + std::to_string(context.memory.memoryHeapCount) + " memory heaps, not the one unified heap");
    const auto& heap = context.memory.memoryHeaps[0];
    Require((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0, "the Context's heap is not device-local");
    Require(heap.size == workingSet && workingSet > 0u, "the Context's heap of " + std::to_string(heap.size) + " bytes is not the recommended working set of " + std::to_string(workingSet));
    Require(context.memoryProperties2 != nullptr && context.physical != VK_NULL_HANDLE, "the Context has no VK_EXT_memory_budget report");

    const auto before = static_cast<std::uint64_t>(metal.currentAllocatedSize);
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    const auto after = static_cast<std::uint64_t>(metal.currentAllocatedSize);
    Require(properties.memoryProperties.memoryHeapCount == 1u && properties.memoryProperties.memoryHeaps[0].size == workingSet, "the report's memory properties are not the Context's heap");
    const auto usage = reported.heapUsage[0];
    const auto budget = reported.heapBudget[0];
    Require(usage > 0u, "the report counts no usage on a device that has allocated its tables");
    Require(usage >= std::min(before, after) && usage <= std::max(before, after), "the reported usage " + std::to_string(usage) + " is not the device's allocated size");
    Require(budget > 0u && budget <= workingSet, "the reported budget " + std::to_string(budget) + " is not within the working set " + std::to_string(workingSet));
    Require(budget >= std::min(usage, workingSet), "the reported budget is below the device's usage within the working set");
    for (std::uint32_t other = 1; other < VK_MAX_MEMORY_HEAPS; ++other) Require(reported.heapBudget[other] == 0u && reported.heapUsage[other] == 0u, "the report describes a heap the Context does not have");
    const auto sampled = AgcDriver::Graphics::SampledTextureBudget(context.memory, &reported, 0);
    const auto storage = AgcDriver::Graphics::TextureCacheBudget(context.memory);
    Require(storage == std::max<std::uint64_t>(2 * GiB, workingSet / 4u), "the storage image cache budget is not a quarter of the working set (2 GiB floor)");
    Require(sampled >= 2 * GiB && sampled <= std::max<std::uint64_t>(2 * GiB, budget), "the sampled texture cache budget " + std::to_string(sampled) + " is outside [2 GiB, budget]");
    const char* level = Metal::CurrentMemoryPressure() == MemoryPressure::Normal ? "normal" : Metal::CurrentMemoryPressure() == MemoryPressure::Warn ? "warn" : "critical";
    std::printf("heap %llu MiB, budget %llu MiB, usage %llu MiB, system headroom %llu MiB, pressure %s; sampled cache budget %llu MiB, storage image cache budget %llu MiB\n", static_cast<unsigned long long>(workingSet >> 20u), static_cast<unsigned long long>(budget >> 20u), static_cast<unsigned long long>(usage >> 20u), static_cast<unsigned long long>(Metal::SystemMemoryHeadroom() >> 20u), level, static_cast<unsigned long long>(sampled >> 20u), static_cast<unsigned long long>(storage >> 20u));
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "bindings";
    std::unique_ptr<AgcDriver::VulkanDevice> device;
    try {
        device = std::make_unique<AgcDriver::VulkanDevice>();
    } catch (const std::exception& error) {
        std::printf("skipped, no usable Metal device: %s\n", error.what());
        return Skipped;
    }
    try {
        if (mode == "bindings") {
            CheckDisjointRanges();
            CheckGds(*device);
            CheckAddressTable(*device);
            CheckFaults(*device);
            CheckWorkgroupArena(*device);
            CheckAtomicLocks(*device);
            CheckOverlappingBuffers(*device);
        } else if (mode == "imports") {
            CheckImports(*device);
        } else if (mode == "completion_wake") {
            CheckCompletionWake(*device);
        } else if (mode == "residency") {
            CheckResidency(*device);
        } else if (mode == "import_window") {
            CheckImportWindow(*device);
        } else if (mode == "submission") {
            CheckSplit(*device);
            CheckIndirect(*device);
            CheckSubmission(*device);
        } else if (mode == "profile") {
            CheckProfile(*device);
        } else if (mode == "memory") {
            CheckMemory(*device);
        } else if (mode == "teardown" || mode == "teardown_exit") {
            CheckTeardown(device, mode == "teardown_exit");
            std::printf("metal device %s tests passed\n", mode.c_str());
            return 0;
        } else {
            std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 2;
        }
        device->WaitIdle();
        std::printf("metal device %s tests passed\n", mode.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        try {
            if (device != nullptr) device->WaitIdle();
        } catch (const std::exception& idle) {
            std::fprintf(stderr, "%s\n", idle.what());
        }
        return 1;
    }
}
