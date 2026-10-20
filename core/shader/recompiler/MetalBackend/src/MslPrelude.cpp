#include "MetalBackend/MslEmitterState.hpp"
#include <array>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

constexpr std::string_view CommonText = R"(struct Aps5Buffer {
    ulong address;
    uint bytes;
    uint flags;
};

struct Aps5DriverData {
    uint workgroupBase[3];
    uint loopGuardLimit;
    ulong faultBuffer;
    ulong atomicLocks;
};

)";

// BdaAbi::Fault: claim the record (Empty -> Writing), fill it, publish it (Ready). The publishing
// fence is seq_cst: MSL 4.0 has only memory_order_relaxed and memory_order_seq_cst (release and
// acquire need MSL 4.1's mem_flags overloads), and the fault path is rare.
constexpr std::string_view FaultRecordText = R"(static void aps5_record_fault(ulong faultBuffer, uint reason, ulong address, uint bytes, uint stage, uint instruction) {
    if (faultBuffer == 0ul) return;
    device atomic_uint* state = reinterpret_cast<device atomic_uint*>(faultBuffer);
    // BdaAbi::Fault::otherFaults (upstream fcb48b774): sticky for every fault that is not Unaligned,
    // so a first unaligned record cannot hide a later access fault.
    if (reason != 7u) atomic_fetch_or_explicit(state + 7, 1u, memory_order_relaxed);
    uint expected = 0u;
    if (!atomic_compare_exchange_weak_explicit(state, &expected, 1u, memory_order_relaxed, memory_order_relaxed)) return;
    device uint* words = reinterpret_cast<device uint*>(faultBuffer);
    words[1] = reason;
    words[2] = uint(address);
    words[3] = uint(address >> 32);
    words[4] = bytes;
    words[5] = stage;
    words[6] = instruction;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    atomic_store_explicit(state, 2u, memory_order_relaxed);
}

)";

constexpr std::string_view LoopGuardText = R"(static void aps5_loop_guard_tripped(constant Aps5DriverData& driver, ulong shaderHash, uint stage, uint pc) {
    aps5_record_fault(driver.faultBuffer, 6u, shaderHash, driver.loopGuardLimit, stage, pc);
}

)";

constexpr std::array<MslPreludeChunk, 1> OnCommon {MslPreludeChunk::Common};
constexpr std::array<MslPreludeChunk, 2> OnFault {MslPreludeChunk::Common, MslPreludeChunk::FaultRecord};

}

MslPreludeEntry MslCorePrelude(MslPreludeChunk chunk) {
    switch (chunk) {
    case MslPreludeChunk::Common:
        return {CommonText, {}};
    case MslPreludeChunk::FaultRecord:
        return {FaultRecordText, OnCommon};
    case MslPreludeChunk::LoopGuard:
        return {LoopGuardText, OnFault};
    default:
        throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not a core chunk");
    }
}

MslPreludeEntry MslPrelude(MslPreludeChunk chunk) {
    switch (chunk) {
    case MslPreludeChunk::Common:
    case MslPreludeChunk::FaultRecord:
    case MslPreludeChunk::LoopGuard:
        return MslCorePrelude(chunk);
    case MslPreludeChunk::Alu:
    case MslPreludeChunk::Float16:
        return MslAluPrelude(chunk);
    case MslPreludeChunk::Float64:
        return MslFloat64Prelude(chunk);
    case MslPreludeChunk::Wave:
        return MslWavePrelude(chunk);
    case MslPreludeChunk::LdsLock:
    case MslPreludeChunk::Atomics64:
        return MslMemoryPrelude(chunk);
    case MslPreludeChunk::Image:
        return MslImagePrelude(chunk);
    case MslPreludeChunk::RuntimeBuffer:
    case MslPreludeChunk::BufferFormat:
    case MslPreludeChunk::Bda:
        return MslBufferPrelude(chunk);
    case MslPreludeChunk::StageIo:
        return MslStagePrelude(chunk);
    default:
        throw std::runtime_error("MSL emission failed: unknown prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)));
    }
}

}
