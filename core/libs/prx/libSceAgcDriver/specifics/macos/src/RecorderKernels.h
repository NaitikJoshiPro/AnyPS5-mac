#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_RECORDERKERNELS_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_RECORDERKERNELS_H

// MSL twins of the driver's helper compute shaders (work package D2): the GLSL originals live in
// Graphics/shaders (MeshArguments.comp, SampleCounter.comp, SampleDumps.comp) and the parameter
// blocks keep their push-constant layouts, now uploaded and bound at buffer index 0 (Metal 4 has
// no push constants). FillPattern is new: Metal fills bytes only, so dword and 16-byte pattern
// fills (vkCmdFillBuffer, the device's pattern fills) run it. Every kernel runs as one threadgroup
// of one thread except FillPattern (one thread per dword). Compiled at first use per device
// (Metal::HelperPipeline).

namespace AgcDriver::Graphics::Metal {

inline constexpr const char* RecorderKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// A 64-bit occlusion counter kept as two dwords (low, high), like the GLSL's uaddCarry sums.
static inline void AddSamples(thread uint& low, thread uint& high, device const uint* value) {
    const uint sum = low + value[0];
    high = high + value[1] + (sum < low ? 1u : 0u);
    low = sum;
}

struct FillParameters {
    device uint* target;
    uint pattern[4];
    uint dwords;
    uint first;
};

kernel void FillPattern(constant FillParameters& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index < p.dwords) p.target[index] = p.pattern[(p.first + index) & 3u];
}

struct MeshArgumentParameters {
    device const uint* record;
    device uint* arguments;
    uint indexCount;
    uint inputSize;
    uint step;
    uint primitivesPerGroup;
    uint maxGroups;
    uint maxInstances;
    uint maxTotal;
    uint unused;
};

kernel void MeshArguments(constant MeshArgumentParameters& p [[buffer(0)]]) {
    const uint count = p.record[0];
    const uint instances = p.record[1];
    const uint first = p.record[2];
    const uint effective = first < p.indexCount ? min(count, p.indexCount - first) : 0u;
    uint groups = 0u;
    if (effective >= p.inputSize && instances != 0u && p.step != 0u && p.primitivesPerGroup != 0u) {
        groups = (effective - p.inputSize) / p.step / p.primitivesPerGroup + 1u;
        if (groups > p.maxGroups || instances > p.maxInstances || groups > p.maxTotal / instances) groups = 0u;
    }
    p.arguments[0] = groups;
    p.arguments[1] = groups != 0u ? instances : 0u;
    p.arguments[2] = groups != 0u ? 1u : 0u;
    p.arguments[3] = effective;
    p.arguments[4] = first;
}

struct SampleCounterParameters {
    device uint* counter;
    device uint* target;
    uint count;
    uint store;
};

kernel void SampleCounter(constant SampleCounterParameters& p [[buffer(0)]]) {
    // A ZPASS_DONE dump is 16 render-backend entries: entry 0 holds the total, the others 0, each
    // with bit 63 (valid) set. store 1, 3, 2 publish a dump in three dispatches (barriers between): the high word without
    // the ready bit, then the low word (and the counter), then the ready bit, so the CPU never
    // sees the ready marker before the counts (upstream 6b164f21af).
    if (p.store == 2u) {
        for (uint db = 0u; db < 16u; ++db) p.target[db * 4u + 1u] |= 0x80000000u;
        return;
    }
    uint low = p.counter[0];
    uint high = p.counter[1];
    for (uint i = 0u; i < p.count; ++i) AddSamples(low, high, p.counter + 4u + 2u * i);
    if (p.store == 1u) {
        for (uint db = 0u; db < 16u; ++db) p.target[db * 4u + 1u] = db == 0u ? (high & 0x7fffffffu) : 0u;
        return;
    }
    if (p.store == 3u) {
        for (uint db = 0u; db < 16u; ++db) p.target[db * 4u] = db == 0u ? low : 0u;
    }
    p.counter[0] = low;
    p.counter[1] = high;
}

struct SampleDumpParameters {
    device uint* counter;
    device const uint* dumps;
    uint firstDump;
    uint dumpCount;
    uint firstSegment;
    uint lastSegment;
    uint phase;
};

kernel void SampleDumps(constant SampleDumpParameters& p [[buffer(0)]]) {
    uint low = p.counter[0];
    uint high = p.counter[1];
    uint segment = p.firstSegment;
    for (uint dump = p.firstDump; dump < p.firstDump + p.dumpCount; ++dump) {
        const uint end = p.dumps[dump * 4u + 2u];
        for (; segment < end; ++segment) AddSamples(low, high, p.counter + 4u + 2u * segment);
        const ulong address = ulong(p.dumps[dump * 4u]) | (ulong(p.dumps[dump * 4u + 1u]) << 32u);
        device uint* target = reinterpret_cast<device uint*>(address);
        // Phases 0, 1, 2 (separate dispatches): high word without the ready bit, low word, ready bit.
        for (uint db = 0u; db < 16u; ++db) {
            if (p.phase == 0u) target[db * 4u + 1u] = db == 0u ? (high & 0x7fffffffu) : 0u;
            else if (p.phase == 1u) target[db * 4u] = db == 0u ? low : 0u;
            else target[db * 4u + 1u] = (db == 0u ? high : 0u) | 0x80000000u;
        }
    }
    if (p.phase != 2u && p.dumpCount != 0u) return;
    for (; segment < p.lastSegment; ++segment) AddSamples(low, high, p.counter + 4u + 2u * segment);
    p.counter[0] = low;
    p.counter[1] = high;
}
)MSL";

}

#endif
