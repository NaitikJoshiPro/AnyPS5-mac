// Metal device tests of work package D1 (docs/mac/M2_DESIGN.md §9): the dispatch bindings (GDS,
// BDA page table, fault buffer and written-page record, device-memory LDS arena, DriverData),
// split and indirect dispatches, GPU labels, fills, copies and completion actions, and the startup
// self-tests. The kernels are hand-written MSL against the MslAbi contract, so these tests prove the
// driver side independently of the MetalBackend families.
//   agc_metal_device_tests bindings   (agc_metal_device_bindings)
//   agc_metal_device_tests submission (agc_metal_device_submission)
//   agc_metal_device_tests profile    (agc_metal_device_profile)
//   agc_metal_device_tests teardown   (agc_metal_device_teardown: D19.5 teardown after a failed MSL
//                                      compile with work in flight, through ~VulkanDevice)
//   agc_metal_device_tests teardown_exit (agc_metal_device_teardown_exit: the same, the device still
//                                      alive at exit(); the exit handler tears it down)

#include "../src/MetalDeviceState.h"
#include "BdaAbi.hpp"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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
struct DriverData { uint workgroupBase[3]; uint loopGuardLimit; ulong faultBuffer; uint reserved[2]; };
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
    out(0)[8] = found;)",
        {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(output.data(), sizeof(output))}, {RuntimeAbi::Binding::BdaPagetable, DescriptorRole::BdaPagetable, 1, {}}}, {1, 1, 1}, 0, {Low(guest), High(guest), Low(guestRo), High(guestRo)});
    device.Dispatch(kernel, 1, 1, 1);
    device.WaitIdle();
    namespace Bda = ShaderRecompiler::BdaAbi;
    Require(output[0] == Bda::Version && output[2] == sizeof(Bda::Range), "bda table: bad header (version " + std::to_string(output[0]) + ", entry bytes " + std::to_string(output[2]) + ")");
    Require(output[3] == sizeof(Bda::Header) + output[1] * sizeof(Bda::Range), "bda table: the binding's size disagrees with the header's count");
    Require(output[8] == 3u, "bda table: the guest blocks were not found or the ranges are unsorted (flags " + std::to_string(output[8]) + ")");
    Require(output[4] == 0x1234u && writable.Words()[101] == 0x1235u, "bda table: the translated load/store of the writable block failed");
    Require(output[5] == (Bda::Read | Bda::Write), "bda table: the writable block has permissions " + std::to_string(output[5]));
    Require(output[6] == 0x5678u && output[7] == Bda::Read, "bda table: the read-only block read " + Hex(output[6]) + " with permissions " + std::to_string(output[7]));
    std::printf("bda page table: %u ranges, translated load and store, read-only permissions\n", output[1]);
}

// Fault buffer: a published fault record is raised once the device is idle and the buffer is reset;
// a written-page record is reported to the write tracker and cleared; DriverData.faultBuffer and the
// table entry agree; a tripped loop guard names the shader and pc.
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
    Require(fault.find("BDA access failed: address=0xabc0001000") != std::string::npos && fault.find("reason=1") != std::string::npos, "faults: the unmapped fault was raised as '" + fault + "'");
    Require(output[0] == 1u && output[1] == Bda::FaultBufferBytes, "faults: DriverData.faultBuffer or the table entry is wrong");
    device.Dispatch(record(static_cast<std::uint32_t>(Bda::FaultReason::LoopLimit), 0x5eedull), 1, 1, 1);
    const auto guard = Throws([&] { device.WaitIdle(); });
    Require(guard.find("loop guard") != std::string::npos && guard.find("0x5eed") != std::string::npos, "faults: the loop guard fault was raised as '" + guard + "'");
    // A written-page record over a guest block: reported and cleared, so the next use starts empty.
    GuestBlock block(65536, true);
    auto written = record(0u, 0u);
    reinterpret_cast<std::uint32_t*>(written.pushConstants.data())[3] = Bda::WrittenChunkKey(block.Address());
    device.Dispatch(written, 1, 1, 1);
    device.WaitIdle();
    device.Dispatch(record(0u, 0u), 1, 1, 1);
    device.WaitIdle();
    Require(output[2] == 0u, "faults: the fault buffer was not reset after a fault and a written-page record (" + Hex(output[2]) + ")");
    std::puts("fault buffer: fault and loop-guard records raised, written pages reported, buffer reset");
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
    alignas(16384) static std::array<std::uint32_t, 1024> memory {};
    memory.fill(0u);
    const auto writeAll = Kernel("for (uint index = 0u; index < 1024u; ++index) out(0)[index] = 0x11111111u;", {{RuntimeAbi::Binding::Buffers, DescriptorRole::GuestBuffers, 1, Buffer(memory.data(), sizeof(memory))}}, {1, 1, 1});
    const std::array<std::uint32_t, 2> label {0xbbbb0001u, 0xbbbb0002u};
    const auto labelAddress = reinterpret_cast<std::uintptr_t>(&memory[8]);
    device.WaitIdle();
    Require(device.WriteLabelOnGpu(labelAddress, std::as_bytes(std::span(label)), 1, 0) == 1, "labels: an idle device did not leave the label to the CPU");
    Require(device.WriteLabelOnGpu(labelAddress + 2u, std::as_bytes(std::span(label)), 1, 0) == 4, "labels: a misaligned label was accepted");
    device.Dispatch(writeAll, 1, 1, 1);
    Require(device.WriteLabelOnGpu(labelAddress, std::as_bytes(std::span(label)), 2, 0, false) == 0, "labels: a label behind recorded work was not stored on the GPU");
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
            CheckGds(*device);
            CheckAddressTable(*device);
            CheckFaults(*device);
            CheckWorkgroupArena(*device);
        } else if (mode == "submission") {
            CheckSplit(*device);
            CheckIndirect(*device);
            CheckSubmission(*device);
        } else if (mode == "profile") {
            CheckProfile(*device);
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
