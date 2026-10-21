// Metal twin of tests/BdaDevice.cpp and tests/BdaExecution.cpp (agc_driver_bda_device), swapped in by
// twins/T1-tests.cmake. Upstream runs hand-built SPIR-V around the SpirvBda helpers on a Vulkan device
// against BdaAbi page tables over Vulkan buffers. On Metal the same helpers are MSL functions in the
// MetalBackend's Bda prelude chunk (MslBda.cpp, exactly the text MslPrelude() hands the emitter), so:
// 1. RunBdaContractTests and the written-page decode checks run as upstream (CPU only);
// 2. execution: one kernel per upstream case calls the prelude helper the MSL emitter emits for that
//    access (LoadAddress 8/16 bits: aps5_bda_read_small; 32 bits: aps5_bda_read_span with one dword;
//    LoadAddressWide: aps5_bda_read_span; StoreAddressWide: aps5_bda_store_dwords; StoreAddress:
//    aps5_bda_store; immediate offsets: aps5_bda_add) against the same page tables, over Metal
//    buffers (newBufferWithLength, never guest memory), with upstream's expected bytes, fault
//    records and written-page records;
// 3. device: the Metal driver's own BDA page table for a dispatch (registered guest allocations
//    wrapped in place by the device through Metal::NewNoCopyBuffer, the bound V# ranges merged by
//    Metal::DisjointRanges) read and written through the same helpers: in-place loads and stores,
//    the permissions of registered and bound ranges, Permission and Unmapped faults, written pages.
// The upstream device test also runs RunColorTransferTests; on Metal that is agc_metal_color_transfer.
// GPU selection (upstream 4cb397a4c9, 8c301fb5c7, 3bac1f0b43): the Metal driver picks its MTLDevice as
// MetalDevice.mm's SelectDevice does (ANYPS5_GPU: the first device whose name contains it, case-insensitive; unset:
// the system default), so SelectMetalDevice below states that rule over device names and DeviceSelectionChecks
// checks it with upstream's cases (--device-selection-cpu, agc_driver_bda_device_selection_cpu). Metal devices have
// no Vulkan API version or device type: upstream's discrete-over-integrated ranking is the system default here and
// its Vulkan 1.1 filter has no counterpart. The execution checks run on the selected device, the driver's device
// must carry the requested name, and a requested GPU that is missing fails instead of skipping, as upstream.
// --host-import-failures (upstream 89bd3f19b6, agc_driver_host_import_failure_handles) skips: it injects
// vkCreateBuffer/vkAllocateMemory/vkBindBufferMemory failures through the device resolver, which a Metal import never
// calls; the Metal form of the check (an import over unreadable pages publishes nothing, a retry imports) is
// HostImportRetireTwin.mm's FailedImportPublishesNothing (agc_host_import_retire; TechnicalDebt
// metal-host-import-failure-seams). BdaExecution.cpp's RunBdaAnchorTests (with a20968ee25's halfword anchor case)
// runs SPIR-V anchored-read programs (MakeBdaAnchoredReadTestShader, SpirvBdaRead.cpp); the MSL Bda prelude has no
// anchored read, so it has no Metal counterpart here.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define main AgcDriverBdaExecutionUnused
#include "../../../tests/BdaExecution.cpp"
#undef main

#include "BdaAbi.hpp"
#include "MetalBackend/MslEmitterState.hpp"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include "../src/MetalCommandStream.h"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace MetalBdaTwin {

namespace Abi = ShaderRecompiler::BdaAbi;
namespace MslAbi = ShaderRecompiler::MslAbi;
namespace RuntimeAbi = ShaderRecompiler::RuntimeAbi;
using ShaderRecompiler::DescriptorRole;
using ShaderRecompiler::MslPrelude;
using ShaderRecompiler::MslPreludeChunk;

constexpr int Skipped = 77;
constexpr std::uint32_t Pc = 0x1234u;
constexpr auto ComputeStage = static_cast<std::uint32_t>(ShaderRecompiler::IrShaderStage::Compute);

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

// The Bda chunk and its dependencies, each once, dependencies first (as the emitter assembles them).
void AddChunk(std::string& source, std::array<bool, static_cast<std::size_t>(MslPreludeChunk::Count)>& added, MslPreludeChunk chunk) {
    auto& done = added[static_cast<std::size_t>(chunk)];
    if (done) return;
    done = true;
    const auto entry = MslPrelude(chunk);
    for (const auto dependency : entry.dependencies) AddChunk(source, added, dependency);
    source += entry.text;
}

std::string BdaPrelude() {
    std::string source;
    std::array<bool, static_cast<std::size_t>(MslPreludeChunk::Count)> added {};
    AddChunk(source, added, MslPreludeChunk::Bda);
    return source;
}

// --- 2. Execution: the prelude helpers against hand-built page tables -------------------------------

enum class Mode : std::uint32_t { Read, DwordRead, SpanRead, DwordWrite, Store, Write };

struct Parameters {
    std::uint64_t table;
    std::uint64_t faults;
    std::uint64_t address;
    std::uint64_t offsetMagnitude;
    std::uint64_t output;
    std::uint32_t tableBytes;
    std::uint32_t mode;
    std::uint32_t negative;
    std::uint32_t bits;
    std::uint32_t dwords;
    std::uint32_t used;
    std::uint32_t glcAccess;
    std::uint32_t stops;
    std::uint32_t spanOffset;
    std::uint32_t stage;
    std::uint32_t values[4];
};
static_assert(sizeof(Parameters) == 96);

constexpr const char* ExecutionKernel = R"(
struct Parameters {
    ulong table;
    ulong faults;
    ulong address;
    ulong offsetMagnitude;
    ulong output;
    uint tableBytes;
    uint mode;
    uint negative;
    uint bits;
    uint dwords;
    uint used;
    uint glcAccess;
    uint stops;
    uint spanOffset;
    uint stage;
    uint4 values;
};

kernel void aps5_bda_check(constant Parameters& p [[buffer(0)]], uint local [[thread_index_in_threadgroup]]) {
    Aps5Bda bda;
    bda.table.address = p.table;
    bda.table.bytes = p.tableBytes;
    bda.table.flags = 0u;
    bda.faults = p.faults;
    bda.stage = p.stage;
    const uint pc = 0x1234u;
    const bool stops = p.stops != 0u;
    const bool glc = p.glcAccess != 0u;
    device uint* output = reinterpret_cast<device uint*>(p.output);
    bool stop = false;
    // Upstream's non-stopping shaders add LocalInvocationIndex to the address (one invocation: 0).
    ulong address = stops ? p.address : p.address + ulong(local);
    if (p.mode == 0u) {
        if (p.offsetMagnitude != 0ul) {
            bool overflow = false;
            address = aps5_bda_add(bda, address, p.offsetMagnitude, p.negative != 0u, pc, overflow);
            if (overflow && stops) return;
        }
        const uint value = p.bits == 32u ? aps5_bda_read_span(bda, address, 0u, 1u, pc, glc, stops, stop).x : aps5_bda_read_small(bda, address, p.bits, pc, stops, stop);
        if (stop) return;
        output[0] = value;
    } else if (p.mode == 1u) {
        const uint4 values = aps5_bda_read_span(bda, address, 0u, (1u << p.dwords) - 1u, pc, glc, stops, stop);
        if (stop) return;
        for (uint dword = 0u; dword < p.dwords; ++dword) output[dword] = values[dword];
    } else if (p.mode == 2u) {
        const uint4 values = aps5_bda_read_span(bda, address, p.spanOffset, p.used, pc, glc, stops, stop);
        if (stop) return;
        for (uint dword = 0u; dword < 4u; ++dword) output[dword] = values[dword];
    } else if (p.mode == 3u) {
        aps5_bda_store_dwords(bda, address, 0u, p.dwords, p.values, pc, glc, stops, stop);
        if (stop) return;
        output[0] = 1u;
    } else if (p.mode == 5u) {
        aps5_bda_write(bda, address, p.values.x, p.bits, pc, glc, stops, stop);
        if (stop) return;
        output[0] = 1u;
    } else {
        aps5_bda_store(bda, address, p.values.x, p.bits, pc, glc);
        output[0] = 1u;
    }
}
)";

class Gpu {
public:
    explicit Gpu(id<MTLDevice> device) : device(device) {
        std::string source = "#include <metal_stdlib>\nusing namespace metal;\n" + BdaPrelude() + ExecutionKernel;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
        Check(library != nil, std::string("the Bda prelude kernel does not compile: ") + (error != nil ? error.localizedDescription.UTF8String : "no error"));
        pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"aps5_bda_check"] error:&error];
        queue = [device newCommandQueue];
        Check(pipeline != nil && queue != nil, std::string("no Bda kernel pipeline or queue: ") + (error != nil ? error.localizedDescription.UTF8String : "no error"));
    }

    id<MTLBuffer> Allocate(std::size_t bytes) {
        id<MTLBuffer> buffer = [device newBufferWithLength:std::max<std::size_t>(bytes, 16u) options:MTLResourceStorageModeShared];
        Check(buffer != nil, "cannot allocate a Metal buffer");
        std::memset(buffer.contents, 0, buffer.length);
        [buffers addObject:buffer];
        return buffer;
    }

    // One dispatch of `groups` threadgroups of one invocation; every buffer of this Gpu is resident.
    void Run(const Parameters& parameters, std::uint32_t groups) {
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        for (id<MTLBuffer> buffer in buffers) [encoder useResource:buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [commands addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(done); }];
        [commands commit];
        const bool finished = dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10ll * NSEC_PER_SEC)) == 0;
        Check(finished, "a Bda kernel did not complete within 10 s");
        Check(commands.status == MTLCommandBufferStatusCompleted, std::string("a Bda kernel failed: ") + (commands.error != nil ? commands.error.localizedDescription.UTF8String : "no error"));
    }

private:
    id<MTLDevice> device;
    id<MTLComputePipelineState> pipeline;
    id<MTLCommandQueue> queue;
    NSMutableArray<id<MTLBuffer>>* buffers = [NSMutableArray new];
};

std::byte* Bytes(id<MTLBuffer> buffer) {
    return static_cast<std::byte*>(buffer.contents);
}

template<std::size_t Count>
void WriteTable(id<MTLBuffer> table, std::uint32_t count, const std::array<Abi::Range, Count>& ranges) {
    const Abi::Header header {Abi::Version, count, sizeof(Abi::Range), 0};
    std::memcpy(Bytes(table), &header, sizeof(header));
    std::memcpy(Bytes(table) + sizeof(header), ranges.data(), sizeof(ranges));
}

template<std::size_t Count>
std::uint32_t TableBytes() {
    return static_cast<std::uint32_t>(sizeof(Abi::Header) + Count * sizeof(Abi::Range));
}

Abi::Fault FaultOf(id<MTLBuffer> fault) {
    Abi::Fault report {};
    std::memcpy(&report, fault.contents, sizeof(report));
    return report;
}

Parameters Base(id<MTLBuffer> table, std::uint32_t tableBytes, id<MTLBuffer> fault, id<MTLBuffer> output, Mode mode, std::uint64_t address) {
    Parameters parameters {};
    parameters.table = table.gpuAddress;
    parameters.tableBytes = tableBytes;
    parameters.faults = fault.gpuAddress;
    parameters.output = output.gpuAddress;
    parameters.mode = static_cast<std::uint32_t>(mode);
    parameters.address = address;
    parameters.stops = 1u;
    parameters.stage = ComputeStage;
    return parameters;
}

// RunBdaDwordWriteTests (BdaExecution.cpp): 2-4 dword stores through aps5_bda_store_dwords.
void DwordWriteChecks(Gpu& gpu) {
    constexpr std::uint64_t guest = 0x7fff12340f80ULL;
    constexpr std::size_t firstBytes = 160;
    id<MTLBuffer> first = gpu.Allocate(firstBytes);
    id<MTLBuffer> second = gpu.Allocate(16);
    id<MTLBuffer> readOnly = gpu.Allocate(16);
    const std::array<Abi::Range, 3> ranges {{
        {guest, guest + firstBytes, first.gpuAddress, Abi::Read | Abi::Write, 0},
        {guest + firstBytes, guest + firstBytes + 16, second.gpuAddress, Abi::Read | Abi::Write, 0},
        {guest + firstBytes + 16, guest + firstBytes + 32, readOnly.gpuAddress, Abi::Read, 0},
    }};
    id<MTLBuffer> table = gpu.Allocate(TableBytes<3>());
    id<MTLBuffer> fault = gpu.Allocate(Abi::FaultBufferBytes);
    id<MTLBuffer> output = gpu.Allocate(4);
    const std::array<std::uint32_t, 4> values {0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
    const auto bytesAt = [&](std::uint64_t address) -> std::byte* {
        for (std::size_t index = 0; index < ranges.size(); ++index) {
            if (address < ranges[index].begin || address >= ranges[index].end) continue;
            id<MTLBuffer> buffer = index == 0 ? first : index == 1 ? second : readOnly;
            return Bytes(buffer) + (address - ranges[index].begin);
        }
        return nullptr;
    };
    const auto notedBytes = [&](std::uint64_t low, std::uint64_t high) {
        std::uint64_t covered = 0, total = 0;
        const auto* words = static_cast<const std::uint32_t*>(fault.contents);
        Check(Abi::ForEachWrittenRange(words, [&](std::uint64_t begin, std::size_t bytes) {
            total += bytes;
            for (const auto address : {low, high}) covered += address >= begin && address < begin + bytes ? 1u : 0u;
        }), "a BDA dword write left an invalid written-page record");
        return std::pair {covered, total};
    };
    const auto write = [&](std::uint64_t address, std::uint32_t dwords, std::uint32_t storedBytes, Abi::FaultReason reason, std::uint32_t groups = 1) {
        const std::string label = "BDA dword write of " + std::to_string(dwords) + " at guest + " + std::to_string(address - guest) + ": ";
        WriteTable(table, 3, ranges);
        std::memset(fault.contents, 0, fault.length);
        for (id<MTLBuffer> buffer : {first, second, readOnly}) std::memset(buffer.contents, 0xa5, buffer.length);
        auto parameters = Base(table, TableBytes<3>(), fault, output, Mode::DwordWrite, address);
        parameters.dwords = dwords;
        std::memcpy(parameters.values, values.data(), sizeof(parameters.values));
        gpu.Run(parameters, groups);
        const auto report = FaultOf(fault);
        for (std::uint32_t byte = 0; byte < dwords * 4u; ++byte) {
            const auto* stored = bytesAt(address + byte);
            const auto expected = byte < storedBytes ? static_cast<std::byte>(values[byte / 4u] >> ((byte % 4u) * 8u)) : std::byte {0xa5};
            Check(stored != nullptr && *stored == expected, label + "incorrect byte at offset " + std::to_string(byte));
        }
        Check(*bytesAt(address - 1) == std::byte {0xa5}, label + "touched the byte before it");
        if (address + dwords * 4u < guest + firstBytes + 32) Check(*bytesAt(address + dwords * 4u) == std::byte {0xa5}, label + "touched the byte after it");
        if (static_cast<std::uint32_t>(reason) == 0) {
            Check(report.state == Abi::FaultState::Empty, label + "a mapped write published a fault");
        } else {
            Check(report.state == Abi::FaultState::Ready && report.reason == reason && report.instruction == Pc, label + "did not publish its fault");
        }
        const auto [covered, total] = notedBytes(address, address + (storedBytes != 0 ? storedBytes - 1u : 0u));
        if (storedBytes == 0) {
            Check(total == 0, label + "stored nothing but noted pages");
        } else {
            const auto pages = ((address + storedBytes - 1u) >> Abi::WrittenPageShift) - (address >> Abi::WrittenPageShift) + 1u;
            Check(covered == 2 && total == pages << Abi::WrittenPageShift, label + "did not note exactly its written pages");
            const auto chunks = ((address + storedBytes - 1u) >> Abi::WrittenChunkShift) - (address >> Abi::WrittenChunkShift) + 1u;
            std::uint32_t claims = 0;
            std::memcpy(&claims, Bytes(fault) + Abi::WrittenCountWord * sizeof(std::uint32_t), sizeof(claims));
            Check(claims == chunks, label + "claimed a written-page slot more than once per chunk");
        }
    };
    write(guest + 16, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 20, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 8, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 12, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 32, 3, 12, static_cast<Abi::FaultReason>(0));
    write(guest + 0x78, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 0x7c, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 41, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + firstBytes - 8, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + firstBytes + 12, 4, 4, Abi::FaultReason::Permission);
    write(guest + 0x7c, 2, 8, static_cast<Abi::FaultReason>(0), 64);
}

// RunBdaExecutionTests (BdaExecution.cpp): reads, dword and span reads, faults.
void ReadChecks(Gpu& gpu) {
    id<MTLBuffer> first = gpu.Allocate(3);
    id<MTLBuffer> second = gpu.Allocate(2);
    Bytes(first)[0] = std::byte {0x11};
    Bytes(first)[1] = std::byte {0x22};
    Bytes(first)[2] = std::byte {0x33};
    Bytes(second)[0] = std::byte {0x44};
    Bytes(second)[1] = std::byte {0x55};
    constexpr std::uint64_t guest = 0x7fff12340001ULL;
    std::array<Abi::Range, 2> ranges {{{guest, guest + 3, first.gpuAddress, Abi::Read, 0}, {guest + 3, guest + 5, second.gpuAddress, Abi::Read, 0}}};
    id<MTLBuffer> table = gpu.Allocate(TableBytes<2>());
    id<MTLBuffer> fault = gpu.Allocate(Abi::FaultBufferBytes);
    id<MTLBuffer> output = gpu.Allocate(4);
    constexpr std::uint32_t sentinel = 0xdeadbeef;
    const auto run = [&](std::uint64_t address, std::uint32_t bits, std::uint32_t expected, Abi::FaultReason reason, std::uint32_t count = 2, std::uint32_t groups = 1, std::int64_t offset = 0) {
        const std::string label = "BDA read of " + std::to_string(bits) + " bits at " + Hex(address) + (offset != 0 ? " + " + std::to_string(offset) : std::string()) + ": ";
        WriteTable(table, count, ranges);
        std::memset(fault.contents, 0, fault.length);
        std::memcpy(output.contents, &sentinel, sizeof(sentinel));
        auto parameters = Base(table, TableBytes<2>(), fault, output, Mode::Read, address);
        parameters.bits = bits;
        parameters.offsetMagnitude = offset < 0 ? std::uint64_t {0} - static_cast<std::uint64_t>(offset) : static_cast<std::uint64_t>(offset);
        parameters.negative = offset < 0 ? 1u : 0u;
        gpu.Run(parameters, groups);
        const auto report = FaultOf(fault);
        std::uint32_t result = 0;
        std::memcpy(&result, output.contents, sizeof(result));
        if (static_cast<std::uint32_t>(reason) == 0) {
            Check(report.state == Abi::FaultState::Empty && result == expected, label + "incorrect data (" + Hex(result) + ") or a fault");
        } else {
            Check(report.state == Abi::FaultState::Ready && report.reason == reason && report.instruction == Pc, label + "the fault was not published correctly (reason " + std::to_string(static_cast<std::uint32_t>(report.reason)) + ")");
            Check(result == sentinel, label + "the faulting kernel continued to output a substitute value");
        }
    };
    run(guest, 8, 0x11, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 16, 0x3322, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 32, 0x55443322, static_cast<Abi::FaultReason>(0));
    run(guest + 5, 8, 0, Abi::FaultReason::Unmapped, 2, 64);
    run(guest - 1, 8, 0, Abi::FaultReason::Unmapped);
    run(std::numeric_limits<std::uint64_t>::max() - 1, 32, 0, Abi::FaultReason::Overflow);
    run(guest, 8, 0, Abi::FaultReason::InvalidTable, 3);
    run(std::numeric_limits<std::uint64_t>::max() - 2, 8, 0, Abi::FaultReason::Overflow, 2, 1, 4);
    run(1, 8, 0, Abi::FaultReason::Overflow, 2, 1, -4);
    id<MTLBuffer> words = gpu.Allocate(16);
    const std::array<std::uint32_t, 4> sentinels {sentinel, sentinel, sentinel, sentinel};
    const auto read = [&](std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops, std::uint32_t value, bool faults) {
        const std::string label = "BDA dword read of " + std::to_string(dwords) + " at " + Hex(address) + (coherent ? " coherent" : "") + (stops ? " stopping" : "") + ": ";
        WriteTable(table, 2, ranges);
        std::memset(fault.contents, 0, fault.length);
        std::memcpy(words.contents, sentinels.data(), sizeof(sentinels));
        auto parameters = Base(table, TableBytes<2>(), fault, words, Mode::DwordRead, address);
        parameters.dwords = dwords;
        parameters.glcAccess = coherent ? 1u : 0u;
        parameters.stops = stops ? 1u : 0u;
        gpu.Run(parameters, 1);
        const auto report = FaultOf(fault);
        std::array<std::uint32_t, 4> result {};
        std::memcpy(result.data(), words.contents, sizeof(result));
        auto expected = sentinels;
        if (!faults || !stops) {
            expected[0] = value;
            for (std::uint32_t dword = 1; dword < dwords; ++dword) expected[dword] = 0;
        }
        if (faults) {
            Check(report.state == Abi::FaultState::Ready && report.reason == Abi::FaultReason::Unmapped && report.address == guest + 5 && report.bytes == 1 && report.instruction == Pc, label + "did not publish its first fault");
        } else {
            Check(report.state == Abi::FaultState::Empty, label + "a mapped read published a fault");
        }
        Check(result == expected, label + (faults && stops ? "the faulting read continued to output a substitute value" : "incorrect data"));
    };
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            read(guest + 1, 1, coherent, stops, 0x55443322, false);
            read(guest + 3, 1, coherent, stops, 0x5544, true);
            read(guest + 1, 4, coherent, stops, 0x55443322, true);
        }
    }
    id<MTLBuffer> wide = gpu.Allocate(32);
    for (std::uint32_t dword = 0; dword < 8u; ++dword) {
        const std::uint32_t value = 0xa0b0c000u + dword;
        std::memcpy(Bytes(wide) + dword * 4u, &value, sizeof(value));
    }
    constexpr std::uint64_t spanGuest = 0x7fff56780000ULL;
    const std::array<Abi::Range, 3> spanRanges {{ranges[0], ranges[1], {spanGuest, spanGuest + 32, wide.gpuAddress, Abi::Read, 0}}};
    id<MTLBuffer> spanTable = gpu.Allocate(TableBytes<3>());
    const auto span = [&](std::uint64_t address, std::int32_t offset, std::uint32_t extracted, bool coherent, bool stops, std::array<std::uint32_t, 4> values, Abi::FaultReason reason = {}, std::uint64_t faultAddress = 0, std::uint32_t faultBytes = 0) {
        const std::string label = "BDA span read at " + Hex(address) + " offset " + std::to_string(offset) + " dwords " + std::to_string(extracted) + (coherent ? " coherent" : "") + (stops ? " stopping" : "") + ": ";
        WriteTable(spanTable, 3, spanRanges);
        std::memset(fault.contents, 0, fault.length);
        std::memcpy(words.contents, sentinels.data(), sizeof(sentinels));
        auto parameters = Base(spanTable, TableBytes<3>(), fault, words, Mode::SpanRead, address);
        parameters.spanOffset = static_cast<std::uint32_t>(offset);
        parameters.used = extracted;
        parameters.glcAccess = coherent ? 1u : 0u;
        parameters.stops = stops ? 1u : 0u;
        gpu.Run(parameters, 1);
        const auto report = FaultOf(fault);
        std::array<std::uint32_t, 4> result {};
        std::memcpy(result.data(), words.contents, sizeof(result));
        const bool faults = static_cast<std::uint32_t>(reason) != 0u;
        if (faults) {
            Check(report.state == Abi::FaultState::Ready && report.reason == reason && report.address == faultAddress && report.bytes == faultBytes && report.instruction == Pc, label + "did not publish its first fault (reason " + std::to_string(static_cast<std::uint32_t>(report.reason)) + " at " + Hex(report.address) + ")");
        } else {
            Check(report.state == Abi::FaultState::Empty, label + "a mapped read published a fault");
        }
        Check(result == (faults && stops ? sentinels : values), label + (faults && stops ? "the faulting read continued to output a substitute value" : "incorrect data"));
    };
    const auto at = [](std::uint32_t dword) { return 0xa0b0c000u + dword; };
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            span(spanGuest + 16, -12, 0b1010u, coherent, stops, {0, at(2), 0, at(4)});
            span(spanGuest, 4, 0b0101u, coherent, stops, {at(1), 0, at(3), 0});
            span(spanGuest + 24, 0, 0b0011u, coherent, stops, {at(6), at(7), 0, 0});
            span(spanGuest + 20, 8, 0b0001u, coherent, stops, {at(7), 0, 0, 0});
            span(spanGuest + 28, 0, 0b0101u, coherent, stops, {at(7), 0, 0, 0}, Abi::FaultReason::Unmapped, spanGuest + 36, 1);
            span(guest, 1, 0b1001u, coherent, stops, {0x55443322, 0, 0, 0}, Abi::FaultReason::Unmapped, guest + 13, 1);
            span(4, -8, 0b0001u, coherent, stops, {0, 0, 0, 0}, Abi::FaultReason::Overflow, 4, 0);
        }
    }
    ranges[0].permissions = 0;
    run(guest, 8, 0, Abi::FaultReason::Permission);
}

// RunBdaStoreExecutionTests (BdaExecution.cpp): 8/16/32-bit stores across ranges, permissions, holes; each case
// also as a runtime write through a V# in GPU memory (upstream 0f2683cb7 MakeBdaWriteTestShader, EmitBdaWrite:
// aps5_bda_write, stopping), with the same expected bytes, faults and written pages.
void StoreChecks(Gpu& gpu) {
    id<MTLBuffer> first = gpu.Allocate(2);
    id<MTLBuffer> second = gpu.Allocate(3);
    id<MTLBuffer> table = gpu.Allocate(TableBytes<2>());
    id<MTLBuffer> fault = gpu.Allocate(Abi::FaultBufferBytes);
    id<MTLBuffer> output = gpu.Allocate(4);
    constexpr std::uint64_t guest = 0x7fff1235fffdULL;
    constexpr std::uint32_t writable = Abi::Read | Abi::Write;
    std::array<Abi::Range, 2> ranges {};
    bool runtime = false;
    const auto run = [&](const char* name, std::uint64_t address, std::uint32_t bits, bool coherent, const std::array<std::uint8_t, 5>& expected, std::initializer_list<std::uint64_t> writtenAddresses, Abi::FaultReason reason = {}, std::uint64_t faultAddress = 0, std::uint32_t faultBytes = 0) {
        const std::string label = std::string(runtime ? "BDA runtime write " : "BDA store ") + name + (coherent ? " coherent: " : ": ");
        std::memset(first.contents, 0xcd, first.length);
        std::memset(second.contents, 0xcd, second.length);
        WriteTable(table, 2, ranges);
        std::memset(fault.contents, 0, fault.length);
        std::memset(output.contents, 0, output.length);
        auto parameters = Base(table, TableBytes<2>(), fault, output, runtime ? Mode::Write : Mode::Store, address);
        parameters.bits = bits;
        parameters.glcAccess = coherent ? 1u : 0u;
        parameters.values[0] = 0xa1b2c3d4u;
        gpu.Run(parameters, 1);
        std::array<std::uint8_t, 5> actual {};
        std::memcpy(actual.data(), first.contents, 2);
        std::memcpy(actual.data() + 2, second.contents, 3);
        Check(actual == expected, label + "incorrect stored bytes or writes outside the mapped span");
        const auto report = FaultOf(fault);
        if (reason == Abi::FaultReason {}) {
            Check(report.state == Abi::FaultState::Empty && report.reason == Abi::FaultReason {} && report.address == 0 && report.bytes == 0 && report.stage == 0 && report.instruction == 0 && report.otherFaults == 0, label + "mapped store published a fault");
        } else {
            Check(report.state == Abi::FaultState::Ready && report.reason == reason && report.address == faultAddress && report.bytes == faultBytes && report.stage == ComputeStage && report.instruction == Pc && report.otherFaults == 0, label + "incorrect first fault metadata");
        }
        std::array<std::uint32_t, Abi::FaultBufferBytes / sizeof(std::uint32_t)> words {};
        std::memcpy(words.data(), fault.contents, sizeof(words));
        Check(words[Abi::WrittenOverflowWord] == 0, label + "written-page table overflowed");
        std::set<std::uint64_t> recordedPages;
        Check(Abi::ForEachWrittenRange(words.data(), [&](std::uint64_t written, std::size_t bytes) {
            for (std::size_t offset = 0; offset < bytes; offset += std::size_t {1} << Abi::WrittenPageShift) {
                Check(recordedPages.insert((written + offset) >> Abi::WrittenPageShift).second, label + "written-page table contains duplicate pages");
            }
        }), label + "invalid written-page record");
        std::set<std::uint64_t> expectedPages;
        std::set<std::uint32_t> expectedChunks;
        for (const auto writtenAddress : writtenAddresses) {
            expectedPages.insert(writtenAddress >> Abi::WrittenPageShift);
            expectedChunks.insert(Abi::WrittenChunkKey(writtenAddress));
        }
        Check(recordedPages == expectedPages, label + "incorrect written-page set");
        Check(words[Abi::WrittenCountWord] == expectedChunks.size(), label + "incorrect written-chunk claim count");
        std::uint32_t completed = 0;
        std::memcpy(&completed, output.contents, sizeof(completed));
        Check(completed == 1, label + "store did not finish its per-byte accesses");
    };
    for (const auto& [coherent, writes] : std::initializer_list<std::pair<bool, bool>>{{false, false}, {false, true}, {true, false}, {true, true}}) {
        runtime = writes;
        ranges = {{{guest, guest + 2, first.gpuAddress, writable, 0}, {guest + 2, guest + 5, second.gpuAddress, writable, 0}}};
        run("byte", guest + 1, 8, coherent, {0xcd, 0xd4, 0xcd, 0xcd, 0xcd}, {guest + 1});
        run("split short", guest + 1, 16, coherent, {0xcd, 0xd4, 0xc3, 0xcd, 0xcd}, {guest + 1, guest + 2});
        run("split unaligned dword", guest, 32, coherent, {0xd4, 0xc3, 0xb2, 0xa1, 0xcd}, {guest, guest + 3});
        ranges[1].permissions = Abi::Read;
        run("read-only byte", guest + 2, 8, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {}, Abi::FaultReason::Permission, guest + 2, 1);
        run("read-only short suffix", guest + 1, 16, coherent, {0xcd, 0xd4, 0xcd, 0xcd, 0xcd}, {guest + 1}, Abi::FaultReason::Permission, guest + 2, 1);
        run("read-only suffix", guest, 32, coherent, {0xd4, 0xc3, 0xcd, 0xcd, 0xcd}, {guest}, Abi::FaultReason::Permission, guest + 2, 1);
        ranges[0].end = guest + 1;
        ranges[1].permissions = writable;
        run("unmapped middle byte", guest, 32, coherent, {0xd4, 0xcd, 0xb2, 0xa1, 0xcd}, {guest, guest + 3}, Abi::FaultReason::Unmapped, guest + 1, 1);
        ranges = {{{guest + 3, guest + 5, first.gpuAddress, writable, 0}, {guest + 5, guest + 8, second.gpuAddress, writable, 0}}};
        run("split aligned dword", guest + 3, 32, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {}, Abi::FaultReason::Unmapped, guest + 3, 4);
    }
}

// --- 3. Device: the Metal driver's page table for a dispatch ----------------------------------------

// Guest memory registered with GuestAllocations, as the guest's own allocations are.
class GuestBlock {
public:
    GuestBlock(std::size_t bytes, bool writable) : size(bytes) {
        block = static_cast<std::uint32_t*>(std::aligned_alloc(16384, bytes));
        Check(block != nullptr, "cannot allocate a guest block");
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

private:
    std::uint32_t* block = nullptr;
    std::size_t size = 0;
};

std::vector<std::uint32_t> BufferDescriptor(std::uint64_t address, std::uint32_t bytes) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

// A kernel as MetalBackend emits one (MslAbi slots, the descriptor table laid out by
// MslAbi::LayoutTable, push data, DriverData) with the Bda prelude: buffers[0] is the output V#,
// buffers[1] (optional) a V# the kernel never touches but the driver adds to the page table as a
// writable range; `body` sees `bda` (the bound page table and fault buffer), `push`, `out`.
ShaderRecompiler::RecompileResult DriverKernel(const std::string& body, std::vector<std::uint32_t> buffers, std::uint32_t bufferCount, std::vector<std::uint32_t> push) {
    const std::vector<MslAbi::TableBinding> layout {{RuntimeAbi::Binding::Buffers, bufferCount}, {RuntimeAbi::Binding::BdaPagetable, 1}, {RuntimeAbi::Binding::FaultBuffer, 1}};
    std::string source = "#include <metal_stdlib>\nusing namespace metal;\n" + BdaPrelude() + R"(
struct Table {
    Aps5Buffer buffers[)" + std::to_string(bufferCount) + R"(];
    Aps5Buffer bda_pagetable[1];
    Aps5Buffer fault_buffer[1];
};
kernel void aps5_main(device const Table& table [[buffer(0)]], constant uint* push [[buffer(1)]], device uint* lds [[buffer(2)]], constant Aps5DriverData& driver [[buffer(3)]]) {
    Aps5Bda bda;
    bda.table = table.bda_pagetable[0];
    bda.faults = table.fault_buffer[0].address;
    bda.stage = )" + std::to_string(ComputeStage) + R"(u;
    device uint* out = reinterpret_cast<device uint*>(table.buffers[0].address);
    const uint pc = 0x1234u;
    bool stop = false;
    device const uint* fault = reinterpret_cast<device const uint*>(bda.faults);
)" + body + R"(
    for (uint word = 0u; word < 8u; ++word) out[24u + word] = fault[word];
    out[32] = fault[)" + std::to_string(Abi::WrittenCountWord) + R"(];
    device const uint* header = reinterpret_cast<device const uint*>(bda.table.address);
    out[33] = header[1];
    uint sorted = 1u;
    for (uint index = 1u; index < header[1]; ++index) {
        if (aps5_bda_word64(header, 4u + index * 8u) < aps5_bda_word64(header, 4u + (index - 1u) * 8u + 2u)) sorted = 0u;
    }
    out[34] = sorted;
}
)";
    MslAbi::Reflection reflection;
    reflection.stage = MslAbi::Stage::Kernel;
    reflection.group = RuntimeAbi::Stage::Main;
    reflection.threadsPerThreadgroup = {1, 1, 1};
    reflection.usesPushData = true;
    reflection.usesDriverData = true;
    reflection.table = MslAbi::LayoutTable(layout);
    ShaderRecompiler::RecompileResult result;
    result.msl = ShaderRecompiler::SharedMsl(ShaderRecompiler::MslShader {source, reflection});
    result.bindings.push_back({ShaderRecompiler::DescriptorKind::StorageBuffer, DescriptorRole::GuestBuffers, RuntimeAbi::DescriptorSet, RuntimeAbi::BindingNumber(RuntimeAbi::Stage::Main, RuntimeAbi::Binding::Buffers), bufferCount, std::move(buffers)});
    result.bindings.push_back({ShaderRecompiler::DescriptorKind::StorageBuffer, DescriptorRole::BdaPagetable, RuntimeAbi::DescriptorSet, RuntimeAbi::BindingNumber(RuntimeAbi::Stage::Main, RuntimeAbi::Binding::BdaPagetable), 1, {}});
    result.bindings.push_back({ShaderRecompiler::DescriptorKind::StorageBuffer, DescriptorRole::FaultBuffer, RuntimeAbi::DescriptorSet, RuntimeAbi::BindingNumber(RuntimeAbi::Stage::Main, RuntimeAbi::Binding::FaultBuffer), 1, {}});
    push.resize(RuntimeAbi::PushConstantDwords, 0u);
    result.pushConstants.resize(push.size() * 4u);
    std::memcpy(result.pushConstants.data(), push.data(), result.pushConstants.size());
    result.memoryOffsetDword = RuntimeAbi::PushConstantDwords - 1u;
    return result;
}

std::uint32_t Low(std::uint64_t value) { return static_cast<std::uint32_t>(value); }
std::uint32_t High(std::uint64_t value) { return static_cast<std::uint32_t>(value >> 32u); }

struct DriverFault {
    Abi::FaultState state;
    Abi::FaultReason reason;
    std::uint64_t address;
    std::uint32_t bytes;
    std::uint32_t stage;
    std::uint32_t instruction;
};

DriverFault FaultSeen(const std::array<std::uint32_t, 64>& out) {
    return {static_cast<Abi::FaultState>(out[24]), static_cast<Abi::FaultReason>(out[25]), static_cast<std::uint64_t>(out[26]) | (static_cast<std::uint64_t>(out[27]) << 32u), out[28], out[29], out[30]};
}

void DeviceChecks(AgcDriver::VulkanDevice& device) {
    GuestBlock readOnly(65536, false);
    GuestBlock writable(65536, true);
    for (std::uint32_t index = 0; index < 64u; ++index) readOnly.Words()[index] = 0x5a000000u + index;
    alignas(16384) static std::array<std::uint32_t, 64> out {};
    const auto ro = readOnly.Address();
    const auto rw = writable.Address();
    const auto output = BufferDescriptor(reinterpret_cast<std::uintptr_t>(out.data()), sizeof(out));
    const auto dispatch = [&](const std::string& body, std::vector<std::uint32_t> push, std::vector<std::uint32_t> extra = {}) {
        out.fill(0xdeadbeefu);
        auto buffers = output;
        buffers.insert(buffers.end(), extra.begin(), extra.end());
        const auto count = static_cast<std::uint32_t>(buffers.size() / 4u);
        device.Dispatch(DriverKernel(body, std::move(buffers), count, std::move(push)), 1, 1, 1);
        device.WaitIdle();
    };
    // In-place reads of a registered read-only allocation and stores into a writable one.
    dispatch(R"(
    const ulong ro = ulong(push[0]) | (ulong(push[1]) << 32u);
    const ulong rw = ulong(push[2]) | (ulong(push[3]) << 32u);
    out[0] = aps5_bda_read_span(bda, ro, 0u, 1u, pc, false, true, stop).x;
    const uint4 four = aps5_bda_read_span(bda, ro, 16u, 0xfu, pc, true, true, stop);
    out[1] = four.x; out[2] = four.y; out[3] = four.z; out[4] = four.w;
    out[5] = aps5_bda_read_small(bda, ro + 11ul, 16u, pc, true, stop);
    aps5_bda_store_dwords(bda, rw, 256u, 4u, uint4(0x10u, 0x20u, 0x30u, 0x40u), pc, false, true, stop);
    aps5_bda_store(bda, rw + 513ul, 0xc3b2a1u, 16u, pc, false);
    out[6] = stop ? 1u : 0u;)", {Low(ro), High(ro), Low(rw), High(rw)});
    Check(out[0] == 0x5a000000u, "device: the in-place read of a registered read-only allocation returned " + Hex(out[0]));
    Check(out[1] == 0x5a000004u && out[2] == 0x5a000005u && out[3] == 0x5a000006u && out[4] == 0x5a000007u, "device: the 4-dword in-place read is wrong");
    Check(out[5] == 0x035au, "device: the unaligned 16-bit read across two dwords is " + Hex(out[5]));
    Check(out[6] == 0u && FaultSeen(out).state == Abi::FaultState::Empty, "device: mapped accesses published a fault (reason " + std::to_string(static_cast<std::uint32_t>(FaultSeen(out).reason)) + " at " + Hex(FaultSeen(out).address) + ")");
    Check(writable.Words()[64] == 0x10u && writable.Words()[65] == 0x20u && writable.Words()[66] == 0x30u && writable.Words()[67] == 0x40u, "device: the dword stores did not land in the writable allocation in place");
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(writable.Words());
    Check(bytes[512] == 0 && bytes[513] == 0xa1 && bytes[514] == 0xb2 && bytes[515] == 0, "device: the unaligned 16-bit store landed wrong");
    Check(out[32] >= 1u, "device: the stores noted no written page");
    Check(out[33] >= 2u && out[34] == 1u, "device: the page table has " + std::to_string(out[33]) + " ranges, sorted " + std::to_string(out[34]));
    std::printf("device page table: %u ranges, in-place reads of a read-only and stores into a writable allocation\n", out[33]);

    // A store into the registered read-only allocation: Permission, nothing stored.
    dispatch(R"(
    const ulong ro = ulong(push[0]) | (ulong(push[1]) << 32u);
    aps5_bda_store(bda, ro + 64ul, 0x77u, 32u, pc, false);)", {Low(ro), High(ro)});
    auto fault = FaultSeen(out);
    Check(fault.state == Abi::FaultState::Ready && fault.reason == Abi::FaultReason::Permission && fault.address == ro + 64u && fault.bytes == 4u && fault.stage == ComputeStage && fault.instruction == Pc, "device: a store into a read-only allocation did not fault with Permission (reason " + std::to_string(static_cast<std::uint32_t>(fault.reason)) + ")");
    Check(readOnly.Words()[16] == 0x5a000010u, "device: the refused store reached the read-only allocation");

    // An address no allocation covers: Unmapped, the stopping read outputs nothing.
    dispatch(R"(
    const uint value = aps5_bda_read_small(bda, 0x1000ul, 8u, pc, true, stop);
    out[7] = stop ? 0xfeedu : value;)", {});
    fault = FaultSeen(out);
    Check(fault.state == Abi::FaultState::Ready && fault.reason == Abi::FaultReason::Unmapped && fault.address == 0x1000u && fault.bytes == 1u, "device: a read of unmapped guest memory did not fault with Unmapped (reason " + std::to_string(static_cast<std::uint32_t>(fault.reason)) + ")");
    Check(out[7] == 0xfeedu, "device: the faulting read did not stop");

    // A written V# inside the read-only allocation: the page table merges the registered range with
    // the bound one (Metal::DisjointRanges), so [ro + 4096, ro + 4160) is writable and the rest of
    // the allocation stays read-only.
    dispatch(R"(
    const ulong ro = ulong(push[0]) | (ulong(push[1]) << 32u);
    aps5_bda_store(bda, ro + 4100ul, 0x6116u, 32u, pc, false);
    out[8] = fault[0];
    aps5_bda_store(bda, ro + 4160ul, 0x7117u, 32u, pc, false);)", {Low(ro), High(ro)}, BufferDescriptor(ro + 4096u, 64));
    fault = FaultSeen(out);
    Check(out[8] == 0u && readOnly.Words()[1025] == 0x6116u, "device: a store into a bound writable V# inside a read-only allocation was refused");
    Check(fault.state == Abi::FaultState::Ready && fault.reason == Abi::FaultReason::Permission && fault.address == ro + 4160u && readOnly.Words()[1040] == 0u, "device: the read-only remainder after a bound writable V# accepted a store (reason " + std::to_string(static_cast<std::uint32_t>(fault.reason)) + " at " + Hex(fault.address) + ")");
    Check(out[34] == 1u, "device: the merged page table is not sorted and disjoint");
    std::puts("device faults: Permission for read-only memory, Unmapped for unknown addresses, bound V# ranges merged with registered ones");
}

// MetalDevice.mm's SelectDevice over device names: with a request the first name containing it (case-insensitive),
// without one the system default.
std::optional<std::size_t> SelectMetalDevice(const std::vector<std::string>& names, std::optional<std::size_t> systemDefault, std::string_view request) {
    if (request.empty()) return systemDefault;
    for (std::size_t index = 0; index < names.size(); ++index) {
        const std::string_view name(names[index]);
        if (std::search(name.begin(), name.end(), request.begin(), request.end(), [](char left, char right) {
            return std::tolower(static_cast<unsigned char>(left)) == std::tolower(static_cast<unsigned char>(right));
        }) != name.end()) return index;
    }
    return std::nullopt;
}

// Upstream's canSkipVulkanFailure: only an unrequested, optional device may be missing.
bool CanSkipDeviceFailure(const char* requestedName, const char* requireDevice) {
    return (requestedName == nullptr || *requestedName == '\0') && requireDevice == nullptr;
}

// Upstream's deviceSelectionTests over SelectMetalDevice (the ranking by device type is the system default here).
void DeviceSelectionChecks() {
    const std::vector<std::string> devices{"AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 5080"};
    Check(!SelectMetalDevice({}, std::nullopt, {}).has_value(), "an empty Metal device list selected a device");
    Check(SelectMetalDevice(devices, 1u, {}) == 1u, "default selection did not take the system default device");
    Check(SelectMetalDevice(devices, 1u, "AMD") == 0u, "an AMD request lost to the system default device");
    Check(SelectMetalDevice(devices, 1u, "aMd rAdEoN") == 0u, "GPU name matching was not case-insensitive");
    Check(SelectMetalDevice(devices, 0u, "RTX 5080") == 1u, "partial GPU name matching failed");
    Check(!SelectMetalDevice(devices, 0u, "missing device").has_value(), "an unmatched GPU request selected another device");
    Check(CanSkipDeviceFailure(nullptr, nullptr), "an optional device test without a device request cannot skip");
    Check(CanSkipDeviceFailure("", nullptr), "an empty GPU request prevents an optional device skip");
    Check(!CanSkipDeviceFailure("missing device", nullptr), "an unmatched GPU request can be reported as a skip");
    Check(!CanSkipDeviceFailure(nullptr, "1"), "a required device test can be reported as a skip");
    std::puts("BDA device selection CPU tests passed");
}

// The MTLDevice the driver selects (SelectMetalDevice over MTLCopyAllDevices), or nil without a usable one; a
// requested name that no device carries throws, listing the devices.
id<MTLDevice> SelectedDevice() {
    const char* requested = std::getenv("ANYPS5_GPU");
    const std::string_view request = requested != nullptr ? requested : "";
    if (request.empty()) return MTLCreateSystemDefaultDevice();
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    std::vector<std::string> names;
    for (id<MTLDevice> candidate in all) names.emplace_back(candidate.name.UTF8String);
    const auto selected = SelectMetalDevice(names, std::nullopt, request);
    if (!selected.has_value()) {
        std::string candidates;
        for (const auto& name : names) candidates += (candidates.empty() ? "" : ", ") + name;
        throw std::runtime_error("no Metal device whose name contains ANYPS5_GPU=\"" + std::string(request) + "\"; devices: " + candidates);
    }
    return all[*selected];
}

int Run() {
    DeviceSelectionChecks();
    RunBdaContractTests();
    RunWrittenRangeDecodeTests();
    std::puts("BDA contract and written-page decode tests passed");
    const bool canSkip = CanSkipDeviceFailure(std::getenv("ANYPS5_GPU"), std::getenv("ANYPS5_REQUIRE_VULKAN"));
    id<MTLDevice> metal = SelectedDevice();
    if (metal == nil) {
        if (!canSkip) throw std::runtime_error("no usable Metal device");
        std::puts("skipped, no usable Metal device");
        return Skipped;
    }
    std::printf("Using \"%s\"\n", metal.name.UTF8String);
    {
        Gpu gpu(metal);
        ReadChecks(gpu);
        DwordWriteChecks(gpu);
        std::puts("BDA read, dword read, span read and dword write execution tests passed");
        StoreChecks(gpu);
        std::puts("BDA store boundary tests passed");
    }
    std::unique_ptr<AgcDriver::VulkanDevice> device;
    try {
        device = std::make_unique<AgcDriver::VulkanDevice>();
    } catch (const std::exception& error) {
        if (!canSkip) throw;
        std::printf("skipped, no usable Metal device: %s\n", error.what());
        return Skipped;
    }
    // The driver's device is the one the execution checks ran on (MetalDevice.mm's SelectDevice).
    if (id<MTLDevice> driver = AgcDriver::Graphics::Metal::DeviceOf(device->graphicsContext())) {
        Check(driver.registryID == metal.registryID, std::string("the driver selected \"") + driver.name.UTF8String + "\", not \"" + metal.name.UTF8String + "\"");
    }
    try {
        DeviceChecks(*device);
    } catch (...) {
        try {
            device->WaitIdle();
        } catch (const std::exception& idle) {
            std::fprintf(stderr, "%s\n", idle.what());
        }
        throw;
    }
    std::puts("Metal BDA allocation and execution tests passed");
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--device-selection-cpu") {
            MetalBdaTwin::DeviceSelectionChecks();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-failures") {
            std::puts("skipped, Metal host imports have no vkCreateBuffer/vkAllocateMemory/vkBindBufferMemory steps to fail; the Metal form of the check runs in agc_host_import_retire (HostImportRetireTwin.mm FailedImportPublishesNothing)");
            return MetalBdaTwin::Skipped;
        }
        return MetalBdaTwin::Run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
