#include "MetalBackend/MslEmitterState.hpp"
#include <array>
#include <stdexcept>

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
    uint reserved[2];
};

)";

// BdaAbi::Fault: claim the record (Empty -> Writing), fill it, publish it (Ready).
constexpr std::string_view FaultRecordText = R"(static void aps5_record_fault(ulong faultBuffer, uint reason, ulong address, uint bytes, uint stage, uint instruction) {
    if (faultBuffer == 0ul) return;
    device atomic_uint* state = reinterpret_cast<device atomic_uint*>(faultBuffer);
    uint expected = 0u;
    if (!atomic_compare_exchange_weak_explicit(state, &expected, 1u, memory_order_relaxed, memory_order_relaxed)) return;
    device uint* words = reinterpret_cast<device uint*>(faultBuffer);
    words[1] = reason;
    words[2] = uint(address);
    words[3] = uint(address >> 32);
    words[4] = bytes;
    words[5] = stage;
    words[6] = instruction;
    atomic_thread_fence(mem_flags::mem_device, memory_order_release, thread_scope_device);
    atomic_store_explicit(state, 2u, memory_order_relaxed);
}

)";

constexpr std::string_view LoopGuardText = R"(static void aps5_loop_guard_tripped(constant Aps5DriverData& driver, ulong shaderHash, uint stage, uint pc) {
    aps5_record_fault(driver.faultBuffer, 6u, shaderHash, driver.loopGuardLimit, stage, pc);
}

)";

// MUBUF addressing and bounds (SpirvRuntimeBuffer.cpp address()) and the direct access through a
// descriptor-table buffer (readBuffer/writeBuffer there).
constexpr std::string_view RuntimeBufferText = R"(struct Aps5BufferAccess {
    uint byte;
    bool inBounds;
};

static inline bool aps5_buffer_fits(uint offset, uint size, uint bytes) {
    return size >= bytes && offset <= size - bytes;
}

static inline Aps5BufferAccess aps5_buffer_access(uint word1, uint records, uint word3, bool valid, uint index, uint offset, uint soffset, uint bytes) {
    const uint stride = extract_bits(word1, 16u, 14u);
    const bool swizzle = stride != 0u && extract_bits(word1, 31u, 1u) != 0u;
    const uint indexShift = extract_bits(word3, 21u, 2u) + 3u;
    const uint indices = 1u << indexShift;
    const uint indexMsb = index >> indexShift;
    const uint indexLsb = index & (indices - 1u);
    const uint swizzled = (indexMsb * stride + (offset & ~3u)) * indices + ((indexLsb << 2u) + (offset & 3u));
    const uint linear = index * stride + offset;
    Aps5BufferAccess access;
    access.byte = (swizzle ? swizzled : linear) + soffset;
    const uint mode = extract_bits(word3, 28u, 2u);
    const bool indexInBounds = index < records;
    const bool structured = indexInBounds && offset < stride;
    const uint rawRecords = records - soffset;
    const bool raw = soffset <= records && (swizzle ? (index < rawRecords && aps5_buffer_fits(offset, stride, bytes)) : aps5_buffer_fits(offset, rawRecords, bytes));
    bool inBounds = mode == 0u ? structured : indexInBounds;
    inBounds = mode < 2u ? inBounds : (mode == 2u ? records != 0u : raw);
    access.inBounds = valid && extract_bits(word3, 12u, 7u) != 0u && inBounds;
    return access;
}

static inline uint aps5_buffer_read(Aps5Buffer buffer, uint byteOffset, uint misalignment, uint byte, uint bits) {
    device const uint* words = reinterpret_cast<device const uint*>(buffer.address);
    const uint index = (byte >> 2u) + (byteOffset >> 2u);
    const bool unaligned = misalignment != 0u;
    if ((unaligned ? index + 1u : index) >= (buffer.bytes >> 2u)) return 0u;
    const uint shift = misalignment << 3u;
    uint value = words[index] >> shift;
    if (unaligned) value |= words[index + 1u] << (32u - shift);
    if (bits == 32u) return value;
    return (value >> ((byte & 3u) << 3u)) & ((1u << bits) - 1u);
}

static inline void aps5_update_bits(device uint* word, uint mask, uint value) {
    device atomic_uint* bits = reinterpret_cast<device atomic_uint*>(word);
    atomic_fetch_and_explicit(bits, ~mask, memory_order_relaxed);
    atomic_fetch_or_explicit(bits, value & mask, memory_order_relaxed);
}

static inline void aps5_buffer_write(Aps5Buffer buffer, uint byteOffset, uint misalignment, uint byte, uint value, uint bits) {
    device uint* words = reinterpret_cast<device uint*>(buffer.address);
    const uint index = (byte >> 2u) + (byteOffset >> 2u);
    const bool unaligned = misalignment != 0u;
    if ((unaligned ? index + 1u : index) >= (buffer.bytes >> 2u)) return;
    if (bits == 32u && !unaligned) {
        words[index] = value;
        return;
    }
    uint mask = bits == 32u ? 0xffffffffu : (1u << bits) - 1u;
    uint masked = value & mask;
    if (bits != 32u) {
        const uint subword = (byte & 3u) << 3u;
        mask <<= subword;
        masked <<= subword;
    }
    const uint shift = misalignment << 3u;
    aps5_update_bits(words + index, mask << shift, masked << shift);
    if (unaligned) aps5_update_bits(words + index + 1u, mask >> (32u - shift), masked >> (32u - shift));
}

)";

constexpr std::array<std::string_view, static_cast<std::size_t>(MslPreludeChunk::Count)> Texts {CommonText, FaultRecordText, LoopGuardText, RuntimeBufferText};

constexpr std::array<MslPreludeChunk, 1> OnCommon {MslPreludeChunk::Common};
constexpr std::array<MslPreludeChunk, 2> OnFault {MslPreludeChunk::Common, MslPreludeChunk::FaultRecord};

}

std::string_view MslPreludeText(MslPreludeChunk chunk) {
    const auto index = static_cast<std::size_t>(chunk);
    if (index >= Texts.size()) throw std::runtime_error("MSL emission failed: unknown prelude chunk");
    return Texts[index];
}

std::span<const MslPreludeChunk> MslPreludeDependencies(MslPreludeChunk chunk) {
    switch (chunk) {
    case MslPreludeChunk::Common:
        return {};
    case MslPreludeChunk::FaultRecord:
    case MslPreludeChunk::RuntimeBuffer:
        return OnCommon;
    case MslPreludeChunk::LoopGuard:
        return OnFault;
    default:
        throw std::runtime_error("MSL emission failed: unknown prelude chunk");
    }
}

}
