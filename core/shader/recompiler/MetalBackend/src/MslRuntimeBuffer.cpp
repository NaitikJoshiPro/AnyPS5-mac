// Work package B5: MUBUF/MTBUF buffer accesses (SpirvRuntimeBuffer.cpp), typed formats
// (SpirvBufferFormat.cpp tables, SpirvFormatConvert.cpp conversions), SMEM buffer loads and the
// flattened SRT (ReadConstBuffer, ReadConst), 32-bit buffer atomics (the buffer parts of
// SpirvMemoryInstructions.cpp). A V# the driver resolved at pipeline time is a descriptor-table
// buffer whose words 1 and 3, presence and base misalignment are function constants; a V# that lives
// in GPU memory (MemoryInfo::gpuDescriptor) is read from its registers and accessed through the
// BdaAbi page table (Bda chunk, MslBda.cpp). Flat/global/scratch accesses are MslAddress.cpp.

#include "MslBufferShared.hpp"
#include "PipelineSpecialization.hpp"
#include "SpirvBackend/SpirvBufferFormat.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

// MUBUF addressing and bounds (SpirvRuntimeBuffer.cpp address()), the direct access through a
// descriptor-table buffer (readBuffer/writeBuffer there), the flattened SRT, and the 32-bit atomic
// helpers shared with flat/global atomics.
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

// One dword of a descriptor-table buffer; a coherent (GLC) read is an atomic load, which reaches
// stores of other threadgroups of the dispatch.
static inline uint aps5_buffer_word(Aps5Buffer buffer, uint index, bool glc) {
    if (glc) return atomic_load_explicit(reinterpret_cast<device atomic_uint*>(buffer.address) + index, memory_order_relaxed);
    return reinterpret_cast<device const uint*>(buffer.address)[index];
}

static inline uint aps5_buffer_read(Aps5Buffer buffer, uint byteOffset, uint misalignment, uint byte, uint bits, bool glc) {
    const uint index = (byte >> 2u) + (byteOffset >> 2u);
    const bool unaligned = misalignment != 0u;
    if ((unaligned ? index + 1u : index) >= (buffer.bytes >> 2u)) return 0u;
    const uint shift = (misalignment << 3u) + (bits == 32u ? 0u : (byte & 3u) << 3u);
    // A narrow component may end in the next dword (see aps5_buffer_write); past the buffer it reads 0.
    const bool crosses = shift + bits > 32u;
    ulong window = aps5_buffer_word(buffer, index, glc);
    if (crosses && index + 1u < (buffer.bytes >> 2u)) window |= ulong(aps5_buffer_word(buffer, index + 1u, glc)) << 32u;
    const uint value = uint(window >> shift);
    return bits == 32u ? value : value & ((1u << bits) - 1u);
}

// A masked store as one compare-exchange (AtomicUpdate): two lanes storing the same byte must leave
// one lane's value, not the OR of both (a separate clear and set would).
static inline void aps5_update_bits(device uint* word, uint mask, uint value) {
    device atomic_uint* bits = reinterpret_cast<device atomic_uint*>(word);
    uint observed = atomic_load_explicit(bits, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(bits, &observed, (observed & ~mask) | (value & mask), memory_order_relaxed, memory_order_relaxed)) {
    }
}

// An 8- or 16-bit component lands at byte (byte & 3) of its dword plus the base misalignment, so it
// can cross into the next dword (a 16-bit store at byte 3, or at byte 2 of a base misaligned by 1..3):
// the mask is built 64 bits wide over the two dwords. Plain (non-GLC) narrow stores are byte stores
// (upstream 066a8fbe3 stores them through an 8-bit view in compute); Apple GPUs store bytes without
// touching the neighbours, and a concurrent compare-exchange on the same dword retries.
static inline void aps5_buffer_write(Aps5Buffer buffer, uint byteOffset, uint misalignment, uint byte, uint value, uint bits, bool glc) {
    device uint* words = reinterpret_cast<device uint*>(buffer.address);
    const uint count = buffer.bytes >> 2u;
    const uint index = (byte >> 2u) + (byteOffset >> 2u);
    const bool unaligned = misalignment != 0u;
    if ((unaligned ? index + 1u : index) >= count) return;
    if (bits == 32u && !unaligned) {
        if (glc) atomic_store_explicit(reinterpret_cast<device atomic_uint*>(words + index), value, memory_order_relaxed);
        else words[index] = value;
        return;
    }
    const uint shift = (misalignment << 3u) + (bits == 32u ? 0u : (byte & 3u) << 3u);
    if (bits != 32u && !glc) {
        device uchar* bytes = reinterpret_cast<device uchar*>(words + index);
        for (uint part = 0u; part < bits / 8u; ++part) {
            const uint at = (shift >> 3u) + part;
            if (index + (at >> 2u) < count) bytes[at] = uchar(value >> (part * 8u));
        }
        return;
    }
    const ulong mask = (bits == 32u ? 0xfffffffful : ((1ul << bits) - 1ul)) << shift;
    const ulong masked = (ulong(value) << shift) & mask;
    aps5_update_bits(words + index, uint(mask), uint(masked));
    if (uint(mask >> 32u) != 0u && index + 1u < count) aps5_update_bits(words + index + 1u, uint(mask >> 32u), uint(masked >> 32u));
}

// Memory of a V# the driver bound in the descriptor table (byte offsets from the V# base).
struct Aps5BufferMemory {
    Aps5Buffer buffer;
    uint byteOffset;
    uint misalignment;
    bool glc;

    uint read(uint byte, uint bits, thread bool& stop) const {
        return aps5_buffer_read(buffer, byteOffset, misalignment, byte, bits, glc);
    }

    void write(uint byte, uint value, uint bits, thread bool& stop) const {
        aps5_buffer_write(buffer, byteOffset, misalignment, byte, value, bits, glc);
    }

    // A coherent (GLC) dwordx2 at an 8-byte aligned address reads a pair one 64-bit store wrote, never
    // half of a 64-bit atomic (which commits with one 64-bit atomic_max/min, see the Atomics64 chunk).
    uint2 read2(uint byte, thread bool& stop) const {
        const ulong address = buffer.address + ulong(byteOffset) + ulong(byte);
        const uint index = (byte >> 2u) + (byteOffset >> 2u);
        if (glc && misalignment == 0u && (address & 7ul) == 0ul && index + 1u < (buffer.bytes >> 2u)) {
            // Atomic loads reach other threadgroups' stores (a plain or coherent(device) load may hit a
            // stale cache line in a spin loop); re-reading the high dword until it is stable gives a
            // pair some 64-bit store wrote.
            device atomic_uint* words = reinterpret_cast<device atomic_uint*>(address);
            uint high = atomic_load_explicit(words + 1, memory_order_relaxed);
            uint low = 0u;
            for (uint attempt = 0u; attempt < 64u; ++attempt) {
                low = atomic_load_explicit(words, memory_order_relaxed);
                const uint again = atomic_load_explicit(words + 1, memory_order_relaxed);
                if (again == high) break;
                high = again;
            }
            return uint2(low, high);
        }
        return uint2(read(byte, 32u, stop), read(byte + 4u, 32u, stop));
    }
};

// A dword of the flattened SRT (ReadConst); zero past its end.
static inline uint aps5_srt_word(Aps5Buffer srt, uint index) {
    return index < (srt.bytes >> 2u) ? reinterpret_cast<device const uint*>(srt.address)[index] : 0u;
}

// A strong compare-exchange returning the old value (OpAtomicCompareExchange).
template <typename A>
static inline uint aps5_atomic_cas(A object, uint expected, uint desired) {
    uint observed = expected;
    while (!atomic_compare_exchange_weak_explicit(object, &observed, desired, memory_order_relaxed, memory_order_relaxed)) {
        if (observed != expected) return observed;
    }
    return expected;
}

static inline uint aps5_atomic_smin(device atomic_uint* object, uint value) {
    return as_type<uint>(atomic_fetch_min_explicit(reinterpret_cast<device atomic_int*>(object), as_type<int>(value), memory_order_relaxed));
}

static inline uint aps5_atomic_smin(threadgroup atomic_uint* object, uint value) {
    return as_type<uint>(atomic_fetch_min_explicit(reinterpret_cast<threadgroup atomic_int*>(object), as_type<int>(value), memory_order_relaxed));
}

static inline uint aps5_atomic_smax(device atomic_uint* object, uint value) {
    return as_type<uint>(atomic_fetch_max_explicit(reinterpret_cast<device atomic_int*>(object), as_type<int>(value), memory_order_relaxed));
}

static inline uint aps5_atomic_smax(threadgroup atomic_uint* object, uint value) {
    return as_type<uint>(atomic_fetch_max_explicit(reinterpret_cast<threadgroup atomic_int*>(object), as_type<int>(value), memory_order_relaxed));
}

// f32 atomics on bits (AtomicFloatBits): NaN-aware min/max and compare that treats +0 == -0.
static inline bool aps5_atomic_f32_nan(uint value) {
    return (value & 0x7fffffffu) > 0x7f800000u;
}

static inline uint aps5_f32_order(uint value) {
    return (value & 0x80000000u) != 0u ? ~value : (value | 0x80000000u);
}

static uint aps5_f32_min_max(uint old, uint source, bool maximum) {
    const bool better = maximum ? aps5_f32_order(source) > aps5_f32_order(old) : aps5_f32_order(source) < aps5_f32_order(old);
    uint result = better ? source : old;
    result = aps5_atomic_f32_nan(old) ? source : result;
    result = aps5_atomic_f32_nan(source) ? old : result;
    result = aps5_atomic_f32_nan(source) && aps5_atomic_f32_nan(old) ? source : result;
    result = aps5_atomic_f32_nan(old) && (old & 0x00400000u) == 0u ? (old | 0x00400000u) : result;
    return aps5_atomic_f32_nan(source) && (source & 0x00400000u) == 0u ? (source | 0x00400000u) : result;
}

static inline bool aps5_f32_equal(uint left, uint right) {
    return (left == right && !aps5_atomic_f32_nan(left)) || ((left | right) & 0x7fffffffu) == 0u;
}

// The value a 32-bit atomic stores (FlatAtomicNext32); `op` is MslAtomicOp. `value` is the
// operand (the desired value of the compare-exchange forms).
static uint aps5_atomic_next(uint op, uint old, uint value, uint comparator) {
    switch (op) {
    case 0u: return value;
    case 1u: return old == comparator ? value : old;
    case 2u: return old + value;
    case 3u: return old - value;
    case 4u: return as_type<int>(old) < as_type<int>(value) ? old : value;
    case 5u: return old < value ? old : value;
    case 6u: return as_type<int>(old) > as_type<int>(value) ? old : value;
    case 7u: return old > value ? old : value;
    case 8u: return old & value;
    case 9u: return old | value;
    case 10u: return old ^ value;
    case 11u: return old >= value ? 0u : old + 1u;
    case 12u: return (old == 0u || old > value) ? value : old - 1u;
    case 13u: return old >= value ? old - value : 0u;
    case 14u: return aps5_f32_min_max(old, value, false);
    case 15u: return aps5_f32_min_max(old, value, true);
    default: return aps5_f32_equal(old, comparator) ? value : old;
    }
}

// AtomicUpdate: a compare-exchange loop for the operations Metal has no atomic for.
template <typename A>
static uint aps5_atomic_update(A object, uint op, uint value, uint comparator) {
    uint observed = atomic_load_explicit(object, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(object, &observed, aps5_atomic_next(op, observed, value, comparator), memory_order_relaxed, memory_order_relaxed)) {
    }
    return observed;
}

)";

// Typed buffer formats: SpirvFormatConvert.cpp as MSL functions over the component type (the
// SpirvFormatComponentType value), its width and its raw bits, and the formatted access of
// SpirvRuntimeBuffer.cpp formatted() over any memory with read/write (Aps5BufferMemory,
// Aps5BdaMemory). The format table follows; with a constant format (typed instructions, V#s the
// driver resolved) the compiler folds the table and the type switches.
constexpr std::string_view BufferFormatText = R"(struct Aps5Format {
    uint type;
    uint count;
    uint packed;
    uint bits[4];
    uint offsets[4];
};

// UnormToF32Bits: value / (2^bits - 1), correctly rounded, in integer arithmetic.
static uint aps5_unorm_to_f32(uint value, uint bits) {
    const uint maximum = (1u << bits) - 1u;
    const bool zero = value == 0u;
    const uint safe = zero ? 1u : value;
    const uint msb = 31u - clz(safe);
    const uint leading = bits - 1u - msb;
    const uint high = safe << leading;
    const uint low = safe >> (bits - leading);
    const uint rotated = (high | low) & maximum;
    uint pattern = rotated << (32u - bits);
    for (uint shift = bits; shift < 32u; shift *= 2u) pattern |= pattern >> shift;
    const uint roundBit = (pattern >> 7u) & 1u;
    const uint rounded = (pattern >> 8u) + roundBit;
    const uint exponent = (msb + 126u - bits) << 23u;
    return zero ? 0u : exponent + rounded;
}

// EmitUFloatToF32Bits: 11- and 10-bit unsigned floats.
static uint aps5_ufloat_to_f32(uint raw, uint bits) {
    const uint mantissaBits = bits == 11u ? 6u : 5u;
    const uint mantissa = raw & ((1u << mantissaBits) - 1u);
    const uint exponent = (raw >> mantissaBits) & 0x1fu;
    const uint mantissa32 = mantissa << (23u - mantissaBits);
    const uint normal = ((exponent + 112u) << 23u) | mantissa32;
    const uint special = 0x7f800000u | mantissa32;
    const uint subnormal = as_type<uint>(float(mantissa) * (bits == 11u ? 0x1p-20f : 0x1p-19f));
    const uint finite = exponent == 0u ? subnormal : normal;
    return exponent == 31u ? special : finite;
}

// EmitF16BitsToF32.
static uint aps5_f16_to_f32(uint bits) {
    const uint signBits = (bits & 0x8000u) << 16u;
    const uint exponent = (bits >> 10u) & 0x1fu;
    const uint mantissa = bits & 0x3ffu;
    const uint mantissa32 = mantissa << 13u;
    const uint normal = ((exponent + 112u) << 23u) | mantissa32;
    const uint special = 0x7f800000u | mantissa32;
    const uint subnormal = as_type<uint>(float(mantissa) * 0x1p-24f);
    const uint finite = exponent == 0u ? subnormal : normal;
    return signBits | (exponent == 0x1fu ? special : finite);
}

static inline bool aps5_format_signed(uint type) {
    return type == 2u || type == 4u || type == 6u;
}

// NormalizeFormatComponent: types 1 Uint, 2 Sint, 3 Unorm, 4 Snorm, 5 Uscaled, 6 Sscaled, 7 Float;
// signed raw values are sign-extended.
static uint aps5_format_normalize(uint type, uint bits, uint raw) {
    switch (type) {
    case 1u:
    case 2u:
        return raw;
    case 3u:
        return aps5_unorm_to_f32(raw, bits);
    case 4u: {
        const uint maximum = (1u << (bits - 1u)) - 1u;
        const bool negative = as_type<int>(raw) < 0;
        const uint magnitude = negative ? 0u - raw : raw;
        return (negative ? 0x80000000u : 0u) | aps5_unorm_to_f32(min(magnitude, maximum), bits - 1u);
    }
    case 5u:
        return as_type<uint>(float(raw));
    case 6u:
        return as_type<uint>(float(as_type<int>(raw)));
    case 7u:
        if (bits == 32u) return raw;
        if (bits == 16u) return aps5_f16_to_f32(raw);
        return aps5_ufloat_to_f32(raw, bits);
    default:
        return 0u;
    }
}

// NormToF16Bits.
static uint aps5_norm_to_f16(uint value, uint maximum) {
    const uint width = 32u - clz(maximum);
    const bool zero = value == 0u;
    const uint safe = zero ? 1u : value;
    const uint first = 9u + width - (31u - clz(safe));
    const bool small = (safe << first) < 1024u * maximum;
    const uint shift = min(small ? first + 1u : first, 24u);
    const uint scaled = safe << shift;
    const uint quotient = scaled / maximum;
    const uint remainder = scaled % maximum;
    const uint rounded = quotient + (remainder > (maximum >> 1u) ? 1u : 0u);
    const uint exponent = (25u - shift) << 10u;
    return zero ? 0u : exponent + rounded - 1024u;
}

// IntegerToF16Bits.
static uint aps5_integer_to_f16(uint value) {
    const bool zero = value == 0u;
    const uint msb = 31u - clz(zero ? 1u : value);
    const bool wide = msb > 10u;
    const uint left = wide ? 0u : 10u - msb;
    const uint right = wide ? msb - 10u : 0u;
    const uint quotient = (value << left) >> right;
    const uint unit = 1u << right;
    const uint remainder = value & (unit - 1u);
    const uint halfUnit = unit >> 1u;
    const bool tie = remainder == halfUnit && (quotient & 1u) != 0u;
    const bool up = wide && (remainder > halfUnit || tie);
    const uint rounded = quotient + (up ? 1u : 0u);
    const uint bits = ((msb + 15u) << 10u) + rounded - 1024u;
    return zero ? 0u : min(bits, 0x7bffu);
}

// F32BitsToSmallFloatRtz: f32 bits to a 16-, 11- or 10-bit float, rounding toward zero.
static uint aps5_f32_to_small_float(uint bits, uint mantissaBits, bool signedFormat) {
    const uint infinity = 0x1fu << mantissaBits;
    const bool negative = (bits & 0x80000000u) != 0u;
    const uint signBits = signedFormat && negative ? 1u << (5u + mantissaBits) : 0u;
    const uint exponent = (bits >> 23u) & 0xffu;
    const uint mantissa = bits & 0x7fffffu;
    const uint top = mantissa >> (23u - mantissaBits);
    const bool lostPayload = top == 0u && mantissa != 0u;
    const uint special = infinity | (lostPayload ? 1u : top);
    const int biased = int(exponent) - 112;
    const uint biasedBits = as_type<uint>(biased);
    const uint normal = (biasedBits << mantissaBits) | top;
    const uint subnormal = (mantissa | 0x800000u) >> min((24u - mantissaBits) - biasedBits, 31u);
    uint magnitude = biased >= -int(mantissaBits) ? subnormal : 0u;
    magnitude = biased >= 1 ? normal : magnitude;
    magnitude = biased >= 31 ? infinity - 1u : magnitude;
    const bool isSpecial = exponent == 0xffu;
    magnitude = isSpecial ? special : magnitude;
    if (!signedFormat) magnitude = negative && !(isSpecial && mantissa != 0u) ? 0u : magnitude;
    return signBits | magnitude;
}

// NormStoreBits: f32 bits scaled to [0, maximum] (or [-maximum, maximum]) with round to nearest
// even, in integer arithmetic; the float comparisons are made on the bits.
static uint aps5_norm_store(uint bits, uint maximum, bool signedFormat) {
    const uint magnitudeBits = bits & 0x7fffffffu;
    const uint exponent = (bits >> 23u) & 0xffu;
    const bool denormal = exponent == 0u;
    const uint mantissa = (bits & 0x7fffffu) | (denormal ? 0u : 0x800000u);
    const uint shift = denormal ? 149u : 150u - exponent;
    const uint extra = shift - 24u;
    const ulong product = ulong(mantissa) * ulong(maximum);
    const uint low = uint(product);
    const uint high = uint(product >> 32u);
    const uint integer = (low >> 24u) | (high << 8u);
    const uint fraction = low & 0xffffffu;
    const uint fixedPoint = (integer << 16u) | (fraction >> 8u);
    const bool sticky = (fraction & 0xffu) != 0u;
    const uint safeExtra = min(extra, 16u);
    const uint quotient = (fixedPoint >> 16u) >> safeExtra;
    const uint remainder = fixedPoint - ((quotient << 16u) << safeExtra);
    const uint halfway = (1u << 15u) << safeExtra;
    const bool tie = remainder == halfway && (sticky || (quotient & 1u) != 0u);
    const uint rounded = quotient + (remainder > halfway || tie ? 1u : 0u);
    uint result = extra > 16u ? 0u : rounded;
    const bool full = magnitudeBits >= 0x3f800000u && magnitudeBits <= 0x7f800000u;
    result = full ? maximum : result;
    const bool positiveMagnitude = magnitudeBits != 0u && magnitudeBits <= 0x7f800000u;
    const bool positive = signedFormat ? positiveMagnitude : (positiveMagnitude && (bits & 0x80000000u) == 0u);
    result = positive ? result : 0u;
    if (!signedFormat) return result;
    return (bits & 0x80000000u) != 0u ? 0u - result : result;
}

// EmitD16FormatComponent: the 16-bit result of a d16 load.
static uint aps5_format_d16(uint type, uint bits, uint raw) {
    if (type != 7u && bits > 16u) {
        if (type == 1u) return min(raw, 0xffffu);
        if (type == 2u) return as_type<uint>(clamp(as_type<int>(raw), -32768, 32767)) & 0xffffu;
        return 0u;
    }
    switch (type) {
    case 1u:
    case 2u:
        return raw & 0xffffu;
    case 3u:
        return aps5_norm_to_f16(raw, (1u << bits) - 1u);
    case 4u: {
        const bool negative = as_type<int>(raw) < 0;
        const uint magnitude = negative ? 0u - raw : raw;
        const uint maximum = (1u << (bits - 1u)) - 1u;
        return (negative ? 0x8000u : 0u) | aps5_norm_to_f16(min(magnitude, maximum), maximum);
    }
    case 5u:
        return aps5_integer_to_f16(raw);
    case 6u: {
        const bool negative = as_type<int>(raw) < 0;
        return (negative ? 0x8000u : 0u) | aps5_integer_to_f16(negative ? 0u - raw : raw);
    }
    case 7u:
        if (bits == 16u) return raw & 0xffffu;
        return aps5_f32_to_small_float(aps5_format_normalize(type, bits, raw), 10u, true);
    default:
        return 0u;
    }
}

// EmitFormatStoreComponent.
static uint aps5_format_store(uint type, uint bits, uint data) {
    const uint mask = bits >= 32u ? 0xffffffffu : (1u << bits) - 1u;
    switch (type) {
    case 1u:
        return bits >= 32u ? data : min(data, mask);
    case 2u: {
        if (bits >= 32u) return data;
        const int low = as_type<int>(~(mask >> 1u));
        const int high = as_type<int>(mask >> 1u);
        return as_type<uint>(min(max(as_type<int>(data), low), high)) & mask;
    }
    case 3u:
    case 5u:
        return aps5_norm_store(data, mask, false);
    case 4u:
    case 6u:
        return aps5_norm_store(data, mask >> 1u, true) & mask;
    case 7u:
        if (bits == 32u) return data;
        if (bits == 16u) return aps5_f32_to_small_float(data, 10u, true);
        if (bits == 11u || bits == 10u) return aps5_f32_to_small_float(data, bits - 5u, false);
        return 0u;
    default:
        return 0u;
    }
}

// EmitD16StoreComponent: a 16-bit source (low bits of `h`) through the format's store conversion.
static uint aps5_format_d16_store(uint type, uint bits, uint h) {
    if (type == 1u) return aps5_format_store(type, bits, h);
    if (type == 2u) return aps5_format_store(type, bits, as_type<uint>(as_type<int>(h << 16u) >> 16));
    if (type == 7u && bits == 16u) return h;
    const uint signBits = (h & 0x8000u) << 16u;
    const uint exponent = (h >> 10u) & 0x1fu;
    const uint mantissa = h & 0x3ffu;
    const uint normal = ((exponent + 112u) << 23u) | (mantissa << 13u);
    const uint special = 0x7f800000u | (mantissa << 13u);
    const uint msb = 31u - clz(mantissa == 0u ? 1u : mantissa);
    const uint subnormal = mantissa == 0u ? 0u : (((msb + 103u) << 23u) | ((mantissa << (23u - msb)) & 0x7fffffu));
    uint magnitude = exponent == 0u ? subnormal : normal;
    magnitude = exponent == 31u ? special : magnitude;
    return aps5_format_store(type, bits, signBits | magnitude);
}

// SpirvRuntimeBuffer.cpp formatted() for a load: `dstSel` holds the four 3-bit destination
// selectors (0 zero, 1 one, 4-7 memory components); a component is read only when a selector of
// the loaded outputs names it, and nothing is read unless every needed component is in bounds.
template <typename Memory>
static uint4 aps5_format_load(thread const Memory& memory, thread const Aps5Format& info, uint word1, uint records, uint word3, bool valid, uint index, uint offset, uint soffset, uint dstSel, uint components, bool d16, thread bool& stop) {
    uint4 result = uint4(0u);
    if (info.count == 0u) return result;
    uint selectors[4];
    uint selected[4];
    for (uint output = 0u; output < 4u; ++output) {
        selectors[output] = extract_bits(dstSel, output * 3u, 3u);
        selected[output] = (selectors[output] - 4u) % info.count;
    }
    bool required[4] = {false, false, false, false};
    Aps5BufferAccess accesses[4];
    bool inBounds = true;
    for (uint component = 0u; component < info.count; ++component) {
        for (uint output = 0u; output < components; ++output) required[component] = required[component] || (selectors[output] >= 4u && selected[output] == component);
        const uint byteOffset = info.packed != 0u ? 0u : info.offsets[component] / 8u;
        const uint bytes = info.packed != 0u ? 4u : info.bits[component] / 8u;
        accesses[component] = aps5_buffer_access(word1, records, word3, valid, index, offset + byteOffset, soffset, bytes);
        inBounds = inBounds && (!required[component] || accesses[component].inBounds);
    }
    uint values[4] = {0u, 0u, 0u, 0u};
    for (uint component = 0u; component < info.count; ++component) {
        if (!inBounds || !required[component]) continue;
        const uint bits = info.bits[component];
        uint raw = memory.read(accesses[component].byte, info.packed != 0u ? 32u : bits, stop);
        if (stop) return uint4(0u);
        if (bits != 32u) {
            const uint at = info.packed != 0u ? info.offsets[component] : 0u;
            raw = aps5_format_signed(info.type) ? as_type<uint>(extract_bits(as_type<int>(raw), at, bits)) : extract_bits(raw, at, bits);
        }
        values[component] = d16 ? aps5_format_d16(info.type, bits, raw) : aps5_format_normalize(info.type, bits, raw);
    }
    const bool integer = info.type == 1u || info.type == 2u;
    const uint one = integer ? 1u : (d16 ? 0x3c00u : 0x3f800000u);
    for (uint output = 0u; output < components; ++output) {
        uint value = values[0];
        for (uint component = 1u; component < info.count; ++component) value = selected[output] == component ? values[component] : value;
        value = selectors[output] == 1u ? one : value;
        result[output] = selectors[output] == 0u ? 0u : value;
    }
    return result;
}

// formatted() for a store: every format component must be in bounds; components past the stored
// ones are written as zero bits.
template <typename Memory>
static void aps5_format_store_data(thread const Memory& memory, thread const Aps5Format& info, uint word1, uint records, uint word3, bool valid, uint index, uint offset, uint soffset, uint4 data, uint components, bool d16, thread bool& stop) {
    if (info.count == 0u) return;
    Aps5BufferAccess accesses[4];
    bool inBounds = true;
    for (uint component = 0u; component < info.count; ++component) {
        const uint byteOffset = info.packed != 0u ? 0u : info.offsets[component] / 8u;
        const uint bytes = info.packed != 0u ? 4u : info.bits[component] / 8u;
        accesses[component] = aps5_buffer_access(word1, records, word3, valid, index, offset + byteOffset, soffset, bytes);
        inBounds = inBounds && accesses[component].inBounds;
    }
    if (!inBounds) return;
    uint packed = 0u;
    for (uint component = 0u; component < info.count; ++component) {
        const uint bits = info.bits[component];
        const uint encoded = component >= components ? 0u : (d16 ? aps5_format_d16_store(info.type, bits, data[component]) : aps5_format_store(info.type, bits, data[component]));
        if (info.packed != 0u) {
            packed |= (encoded & ((1u << bits) - 1u)) << info.offsets[component];
        } else {
            memory.write(accesses[component].byte, encoded, bits, stop);
            if (stop) return;
        }
    }
    if (info.packed != 0u) memory.write(accesses[0].byte, packed, 32u, stop);
}

)";

constexpr std::array<MslPreludeChunk, 1> OnCommon {MslPreludeChunk::Common};
constexpr std::array<MslPreludeChunk, 2> OnRuntimeBuffer {MslPreludeChunk::Common, MslPreludeChunk::RuntimeBuffer};
constexpr std::array<MslPreludeChunk, 3> OnFaultAndBuffer {MslPreludeChunk::Common, MslPreludeChunk::FaultRecord, MslPreludeChunk::RuntimeBuffer};

constexpr std::uint32_t FormatCount = static_cast<std::uint32_t>(IrBufferFormat::Format32_32_32_32Float);

// The BufferFormat chunk: the conversion functions and GetFormatInfo of every unified format
// 0-77 (0 and anything past 77 have no components: such accesses read zero and write nothing).
const std::string& BufferFormatChunk() {
    static const std::string text = [] {
        std::string table(BufferFormatText);
        table += "constant Aps5Format aps5_formats[" + std::to_string(FormatCount + 1u) + "] = {\n";
        for (std::uint32_t code = 0; code <= FormatCount; ++code) {
            const auto info = GetFormatInfo(static_cast<IrBufferFormat>(code));
            const auto list = [](const std::uint32_t (&values)[4]) {
                return "{" + std::to_string(values[0]) + "u, " + std::to_string(values[1]) + "u, " + std::to_string(values[2]) + "u, " + std::to_string(values[3]) + "u}";
            };
            table += "    {" + std::to_string(static_cast<std::uint32_t>(info.type)) + "u, " + std::to_string(info.componentCount) + "u, " + (info.packedBitfield ? "1u" : "0u") + ", " + list(info.componentBits) + ", " + list(info.componentBitOffset) + "},\n";
        }
        table += "};\n\n";
        table += "static inline Aps5Format aps5_format(uint format) {\n    return aps5_formats[format <= " + std::to_string(FormatCount) + "u ? format : 0u];\n}\n\n";
        return table;
    }();
    return text;
}

static_assert(static_cast<std::uint32_t>(SpirvFormatComponentType::Uint) == 1u && static_cast<std::uint32_t>(SpirvFormatComponentType::Sint) == 2u && static_cast<std::uint32_t>(SpirvFormatComponentType::Unorm) == 3u && static_cast<std::uint32_t>(SpirvFormatComponentType::Snorm) == 4u && static_cast<std::uint32_t>(SpirvFormatComponentType::Uscaled) == 5u && static_cast<std::uint32_t>(SpirvFormatComponentType::Sscaled) == 6u && static_cast<std::uint32_t>(SpirvFormatComponentType::Float) == 7u, "the BufferFormat chunk spells the component types as literals");
static_assert(static_cast<std::uint32_t>(MslAtomicOp::FCmpSwap) == 16u && static_cast<std::uint32_t>(MslAtomicOp::USubSat) == 13u, "the RuntimeBuffer chunk spells the atomic operations as literals");

std::string ActiveArgument(MslEmitterState& state, const IrValue& instruction) {
    return state.Arg(instruction, instruction.ArgumentCount() - 1u);
}

// The parts of a MUBUF access the shader needs, as SpirvRuntimeBuffer.cpp descriptor() builds them.
// For a V# the driver resolved at pipeline time, words 1 and 3, presence and the base's
// misalignment are function constants (BufferBase + 4 * resource) and the record count stays a
// runtime value (the handle's word 2); a V# in GPU memory is its four registers, its base an
// address in the BDA page table.
struct B5Descriptor {
    std::string word1;
    std::string records;
    std::string word3;
    std::string valid;
    std::string index;
    std::string base;
    std::string memory;
    std::string bda;
    std::string pc;
    bool gpu = false;
    bool stops = false;
};

// Emits the descriptor's locals (inside the caller's block) and returns their names.
B5Descriptor MakeDescriptor(MslEmitterState& state, const IrValue& instruction, bool needsBase) {
    const auto& memory = state.Memory(instruction);
    const auto* handle = instruction.Argument(0)->Resolve();
    if (handle->Opcode() != IrOpcode::GetBufferResource || handle->ArgumentCount() != 4u) state.Fail(instruction, "requires a runtime four-dword V#");
    if (memory.kind != ResourceKind::Buffer && memory.kind != ResourceKind::ScalarBuffer) state.Fail(instruction, "must access a buffer resource");
    B5Descriptor descriptor;
    descriptor.gpu = memory.gpuDescriptor;
    descriptor.pc = MslB5Pc(instruction);
    const auto first = PipelineSpecialization::BufferBase + memory.resource * PipelineSpecialization::BufferWords;
    if (descriptor.gpu) {
        descriptor.word1 = state.Arg(*handle, 1u);
        descriptor.word3 = state.Arg(*handle, 3u);
    } else {
        descriptor.word1 = state.FunctionConstant(first, 0u);
        descriptor.word3 = state.FunctionConstant(first + 1u, 0u);
    }
    descriptor.records = state.Arg(*handle, 2u);
    const auto addThreadId = "(extract_bits(" + descriptor.word3 + ", 23u, 1u) != 0u)";
    std::string invalid = "(extract_bits(" + descriptor.word3 + ", 30u, 2u) != 0u)";
    descriptor.index = "0u";
    if (memory.kind == ResourceKind::Buffer) {
        if (state.ThreadgroupStage()) {
            // The guest lane of the half being emitted (B1 core helper; laneCount 2 adds 32 in half 1).
            descriptor.index = "(" + state.Arg(instruction, 1u) + " + (" + addThreadId + " ? " + state.GuestLane() + " : 0u))";
        } else {
            descriptor.index = state.Arg(instruction, 1u);
            invalid = "(" + invalid + " || " + addThreadId + ")";
        }
    }
    if (descriptor.gpu || needsBase) {
        descriptor.base = state.Temporary("aps5_vb");
        state.Line("const ulong " + descriptor.base + " = ulong(" + state.Arg(*handle, 0u) + ") | (ulong(extract_bits(" + state.Arg(*handle, 1u) + ", 0u, 16u)) << 32u);");
    }
    if (!descriptor.gpu) {
        descriptor.valid = "(" + state.FunctionConstant(first + 2u, 0u) + " != 0u)";
        const auto element = state.DescriptorElement(DescriptorBindingKind::Buffers, memory.resource);
        const auto misalignment = memory.kind == ResourceKind::ScalarBuffer ? std::string("0u") : state.FunctionConstant(first + 3u, 0u);
        state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
        descriptor.memory = state.Temporary("aps5_vm");
        state.Line("const Aps5BufferMemory " + descriptor.memory + " {" + state.TableBuffer(DescriptorBindingKind::Buffers, element) + ", " + state.MemoryByteOffset(element) + ", " + misalignment + ", " + MslBool(memory.coherent) + "};");
        return descriptor;
    }
    // A V# in GPU memory: an invalid one (reserved type bits, or add_tid outside compute) records an
    // InvalidDescriptor fault; a null base reads zero.
    descriptor.bda = MslBdaContext(state);
    descriptor.stops = MslBdaStops(state);
    const auto bad = state.Temporary("aps5_vi");
    state.Line("const bool " + bad + " = " + invalid + ";");
    state.Line("if (" + bad + ") aps5_bda_fault(" + descriptor.bda + ", 8u, " + descriptor.base + ", 0u, " + descriptor.pc + ");");
    MslB5StopIf(state, bad);
    descriptor.valid = "(!" + bad + " && " + descriptor.base + " != 0ul)";
    descriptor.memory = state.Temporary("aps5_vm");
    state.Line("const Aps5BdaMemory " + descriptor.memory + " {" + descriptor.bda + ", " + descriptor.base + ", " + descriptor.pc + ", " + MslBool(memory.coherent) + ", " + MslBool(descriptor.stops) + "};");
    return descriptor;
}

std::string OffsetExpression(MslEmitterState& state, const IrValue& instruction) {
    return "(" + state.Arg(instruction, 2u) + " + " + MslHex(state.Memory(instruction).offset) + "u)";
}

std::string AccessExpression(MslEmitterState& state, const IrValue& instruction, const B5Descriptor& descriptor, std::uint32_t componentOffset, std::uint32_t bytes) {
    const auto offset = "(" + state.Arg(instruction, 2u) + " + " + MslHex(state.Memory(instruction).offset + componentOffset) + "u)";
    return "aps5_buffer_access(" + descriptor.word1 + ", " + descriptor.records + ", " + descriptor.word3 + ", " + descriptor.valid + ", " + descriptor.index + ", " + offset + ", " + state.Arg(instruction, 3u) + ", " + std::to_string(bytes) + "u)";
}

// The format of a formatted access: a typed instruction's own (a literal), a V# the driver resolved
// (its word 3, a function constant), or a V# in GPU memory (its runtime word 3; a literal when the
// register is an immediate). Also the destination selectors of the loaded outputs.
struct B5Format {
    std::string code;
    std::string dstSel;
    bool runtime = false;
};

B5Format FormatOf(MslEmitterState& state, const IrValue& instruction, const B5Descriptor& descriptor) {
    const auto& memory = state.Memory(instruction);
    B5Format format;
    if (memory.typed) {
        const auto value = DecodeTBufferFormat(memory.dataFormat, memory.numberFormat);
        const auto info = GetFormatInfo(value);
        if (info.type == SpirvFormatComponentType::Unknown) state.Fail(instruction, "typed buffer instruction has an invalid format");
        std::uint32_t dstSel = 0;
        for (std::uint32_t output = 0; output < 4u; ++output) dstSel |= (output < info.componentCount ? 4u + output : 0u) << (output * 3u);
        format.code = std::to_string(static_cast<std::uint32_t>(value)) + "u";
        format.dstSel = MslHex(dstSel) + "u";
        return format;
    }
    format.dstSel = "extract_bits(" + descriptor.word3 + ", 0u, 12u)";
    const auto* word = instruction.Argument(0)->Resolve()->Argument(3)->Resolve();
    if (descriptor.gpu && word->HasImmediate()) {
        const auto value = (word->ImmediateU32() >> 12u) & 0x7fu;
        if (value == 0u || value > FormatCount) state.Fail(instruction, "formatted buffer instruction has an invalid constant format");
        format.code = std::to_string(value) + "u";
        return format;
    }
    format.code = "extract_bits(" + descriptor.word3 + ", 12u, 7u)";
    format.runtime = descriptor.gpu;
    return format;
}

// For a V# in GPU memory, faults the formatted access cannot run with (formattedDispatch,
// formatSwitch): reserved destination selectors of the loaded outputs, an invalid format.
void CheckGpuFormat(MslEmitterState& state, const IrValue& instruction, const B5Descriptor& descriptor, const B5Format& format, std::uint32_t components, bool store) {
    if (!descriptor.gpu) return;
    const auto& memory = state.Memory(instruction);
    if (!store && !memory.typed) {
        std::string invalid = "false";
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto selector = "extract_bits(" + descriptor.word3 + ", " + std::to_string(component * 3u) + "u, 3u)";
            invalid += " || " + selector + " == 2u || " + selector + " == 3u";
        }
        const auto bad = state.Temporary("aps5_fi");
        state.Line("const bool " + bad + " = " + invalid + ";");
        state.Line("if (" + bad + ") aps5_bda_fault(" + descriptor.bda + ", 8u, " + descriptor.base + ", 0u, " + descriptor.pc + ");");
        MslB5StopIf(state, bad);
    }
    if (format.runtime) {
        const auto unsupported = state.Temporary("aps5_fu");
        state.Line("const bool " + unsupported + " = " + format.code + " == 0u || " + format.code + " > " + std::to_string(FormatCount) + "u;");
        state.Line("if (" + unsupported + ") aps5_bda_fault(" + descriptor.bda + ", 8u, " + descriptor.base + ", 0u, " + descriptor.pc + ");");
        MslB5StopIf(state, unsupported);
    }
}

std::string Widen(const std::string& value, std::uint32_t components) {
    switch (components) {
    case 1u:
        return "uint4(uint(" + value + "), 0u, 0u, 0u)";
    case 2u:
        return "uint4(" + value + ", 0u, 0u)";
    case 3u:
        return "uint4(" + value + ", 0u)";
    default:
        return value;
    }
}

std::string Narrow(const std::string& value, std::uint32_t components) {
    static constexpr const char* Swizzles[] = {"", ".x", ".xy", ".xyz", ""};
    return value + Swizzles[components];
}

void Load(MslEmitterState& state, const IrValue& instruction, std::uint32_t components) {
    const auto& memory = state.Memory(instruction);
    if (memory.kind != ResourceKind::Buffer) state.Fail(instruction, "must access a buffer resource");
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    const auto result = state.Name(instruction);
    const auto type = MslType(instruction.Type());
    state.Line(result + " = " + type + "(0);");
    state.Line("if (" + ActiveArgument(state, instruction) + ") {");
    state.Indent();
    const auto descriptor = MakeDescriptor(state, instruction, false);
    const auto stop = state.Temporary("aps5_st");
    state.Line("bool " + stop + " = false;");
    if (memory.formatted) {
        state.RequirePrelude(MslPreludeChunk::BufferFormat);
        const auto format = FormatOf(state, instruction, descriptor);
        state.Line("if (" + descriptor.valid + ") {");
        state.Indent();
        CheckGpuFormat(state, instruction, descriptor, format, components, false);
        const auto info = state.Temporary("aps5_fm");
        state.Line("const Aps5Format " + info + " = aps5_format(" + format.code + ");");
        const auto value = "aps5_format_load(" + descriptor.memory + ", " + info + ", " + descriptor.word1 + ", " + descriptor.records + ", " + descriptor.word3 + ", " + descriptor.valid + ", " + descriptor.index + ", " + OffsetExpression(state, instruction) + ", " + state.Arg(instruction, 3u) + ", " + format.dstSel + ", " + std::to_string(components) + "u, " + MslBool(memory.d16) + ", " + stop + ")";
        state.Line(result + " = " + (components == 1u ? type + "(" + value + ".x)" : Narrow(value, components)) + ";");
        if (descriptor.gpu) MslB5StopIf(state, stop);
        state.Outdent();
        state.Line("}");
    } else if (components == 2u && memory.coherent && memory.dataBits == 32u) {
        // GLC dwordx2: one 64-bit load when both dwords are in bounds (a 64-bit atomic's value whole).
        const auto first = state.Temporary("aps5_a");
        const auto second = state.Temporary("aps5_a");
        state.Line("const Aps5BufferAccess " + first + " = " + AccessExpression(state, instruction, descriptor, 0u, 4u) + ";");
        state.Line("const Aps5BufferAccess " + second + " = " + AccessExpression(state, instruction, descriptor, 4u, 4u) + ";");
        state.Line("if (" + first + ".inBounds && " + second + ".inBounds && " + second + ".byte == " + first + ".byte + 4u) " + result + " = " + descriptor.memory + ".read2(" + first + ".byte, " + stop + ");");
        state.Line("else {");
        state.Indent();
        state.Line("if (" + first + ".inBounds) " + result + "[0] = " + descriptor.memory + ".read(" + first + ".byte, 32u, " + stop + ");");
        state.Line("if (" + second + ".inBounds) " + result + "[1] = " + descriptor.memory + ".read(" + second + ".byte, 32u, " + stop + ");");
        state.Outdent();
        state.Line("}");
        if (descriptor.gpu) MslB5StopIf(state, stop);
    } else {
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto access = state.Temporary("aps5_a");
            state.Line("const Aps5BufferAccess " + access + " = " + AccessExpression(state, instruction, descriptor, component * 4u, memory.dataBits / 8u) + ";");
            const auto target = components == 1u ? result : result + "[" + std::to_string(component) + "]";
            const auto read = descriptor.memory + ".read(" + access + ".byte, " + std::to_string(memory.dataBits) + "u, " + stop + ")";
            state.Line("if (" + access + ".inBounds) " + target + " = " + (components == 1u ? type + "(" + read + ")" : read) + ";");
            if (descriptor.gpu) MslB5StopIf(state, stop);
        }
    }
    state.Outdent();
    state.Line("}");
}

void Store(MslEmitterState& state, const IrValue& instruction, std::uint32_t components) {
    const auto& memory = state.Memory(instruction);
    if (memory.kind != ResourceKind::Buffer) state.Fail(instruction, "must access a buffer resource");
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    const auto data = state.Arg(instruction, instruction.ArgumentCount() - 2u);
    const auto lane = MslUniformStoreLane(state, instruction, 0u, 3u);
    state.Line("if (" + ActiveArgument(state, instruction) + (lane.empty() ? std::string() : " && " + lane) + ") {");
    state.Indent();
    const auto descriptor = MakeDescriptor(state, instruction, false);
    const auto stop = state.Temporary("aps5_st");
    state.Line("bool " + stop + " = false;");
    if (memory.formatted) {
        state.RequirePrelude(MslPreludeChunk::BufferFormat);
        const auto format = FormatOf(state, instruction, descriptor);
        state.Line("if (" + descriptor.valid + ") {");
        state.Indent();
        CheckGpuFormat(state, instruction, descriptor, format, components, true);
        const auto info = state.Temporary("aps5_fm");
        state.Line("const Aps5Format " + info + " = aps5_format(" + format.code + ");");
        state.Line("aps5_format_store_data(" + descriptor.memory + ", " + info + ", " + descriptor.word1 + ", " + descriptor.records + ", " + descriptor.word3 + ", " + descriptor.valid + ", " + descriptor.index + ", " + OffsetExpression(state, instruction) + ", " + state.Arg(instruction, 3u) + ", " + Widen(data, components) + ", " + std::to_string(components) + "u, " + MslBool(memory.d16) + ", " + stop + ");");
        if (descriptor.gpu) MslB5StopIf(state, stop);
        state.Outdent();
        state.Line("}");
    } else {
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto access = state.Temporary("aps5_a");
            state.Line("const Aps5BufferAccess " + access + " = " + AccessExpression(state, instruction, descriptor, component * 4u, memory.dataBits / 8u) + ";");
            const auto value = components == 1u ? "uint(" + data + ")" : data + "[" + std::to_string(component) + "]";
            state.Line("if (" + access + ".inBounds) " + descriptor.memory + ".write(" + access + ".byte, " + value + ", " + std::to_string(memory.dataBits) + "u, " + stop + ");");
            if (descriptor.gpu) MslB5StopIf(state, stop);
        }
    }
    state.Outdent();
    state.Line("}");
}

// EmitRuntimeScalarBufferLoad (s_buffer_load_dword): the dword at (offset + imm) & ~3, inside the V#'s
// stride * records bytes (records bytes for stride 0); the base's low bits are ignored.
void ReadConstBuffer(MslEmitterState& state, const IrValue& instruction) {
    const auto& memory = state.Memory(instruction);
    if (memory.planningOnly) return;
    if (memory.kind != ResourceKind::ScalarBuffer) state.Fail(instruction, "must read a scalar buffer resource");
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    const auto result = state.Name(instruction);
    state.Line(result + " = 0u;");
    state.Line("{");
    state.Indent();
    const auto descriptor = MakeDescriptor(state, instruction, false);
    const auto stride = "extract_bits(" + descriptor.word1 + ", 16u, 14u)";
    const auto address = state.Temporary("aps5_sa");
    const auto byte = state.Temporary("aps5_sb");
    state.Line("const uint " + address + " = " + state.Arg(instruction, 1u) + " + " + MslHex(memory.offset) + "u;");
    state.Line("const uint " + byte + " = " + address + " & ~3u;");
    const auto size = "(" + stride + " == 0u ? ulong(" + descriptor.records + ") : ulong(" + stride + ") * ulong(" + descriptor.records + "))";
    const auto stop = state.Temporary("aps5_st");
    state.Line("bool " + stop + " = false;");
    state.Line("if (" + address + " >= " + state.Arg(instruction, 1u) + " && " + descriptor.valid + " && ulong(" + byte + ") + 4ul <= " + size + ") " + result + " = " + descriptor.memory + ".read(" + byte + ", 32u, " + stop + ");");
    if (descriptor.gpu) MslB5StopIf(state, stop);
    state.Outdent();
    state.Line("}");
}

// EmitReadConst: a dword of the flattened SRT. A guarded slot (an SRT read whose source pointer the
// planner could not prove mapped) holds a poison flag: a poisoned read records the Unmapped fault of
// the original access (its pc and address are in the SRT's guard records) and ends the invocation.
void ReadConst(MslEmitterState& state, const IrValue& instruction) {
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    const auto srt = state.TableBuffer(DescriptorBindingKind::FlattenedSrt, 0u);
    const auto& resources = state.program.Resources();
    const auto& guarded = resources.guardedSrtSlots;
    if (!guarded.empty()) {
        const IrValue* slot = instruction.Argument(1)->Resolve();
        if (!slot->HasImmediate()) state.Fail(instruction, "reads the flattened SRT at a runtime slot while some slots are guarded");
        const auto found = std::lower_bound(guarded.begin(), guarded.end(), slot->ImmediateU32());
        if (found != guarded.end() && *found == slot->ImmediateU32()) {
            const auto guard = static_cast<std::uint32_t>(found - guarded.begin());
            const auto records = resources.srtGuardOffset + static_cast<std::uint32_t>(guarded.size());
            const auto flag = state.Temporary("aps5_pf");
            const auto record = state.Temporary("aps5_pr");
            state.Line("{");
            state.Indent();
            state.Line("const uint " + flag + " = aps5_srt_word(" + srt + ", " + std::to_string(resources.srtGuardOffset + guard) + "u);");
            state.Line("if (" + flag + " != 0u) {");
            state.Indent();
            state.Line("const uint " + record + " = " + std::to_string(records) + "u + (" + flag + " - 1u) * 3u;");
            const auto word = [&](std::uint32_t offset) { return "aps5_srt_word(" + srt + ", " + record + " + " + std::to_string(offset) + "u)"; };
            state.Line(MslRecordFault(state, 1u, "(ulong(" + word(1u) + ") | (ulong(" + word(2u) + ") << 32u))", "4u", word(0u)));
            state.Outdent();
            state.Line("}");
            MslB5StopIf(state, flag + " != 0u");
            state.Outdent();
            state.Line("}");
        }
    }
    state.Define(instruction, "aps5_srt_word(" + srt + ", " + state.Arg(instruction, 1u) + ")");
}

std::optional<MslAtomicOp> BufferAtomicOp(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::BufferAtomicSwap32:
        return MslAtomicOp::Swap;
    case IrOpcode::BufferAtomicCmpSwap32:
        return MslAtomicOp::CmpSwap;
    case IrOpcode::BufferAtomicIAdd32:
        return MslAtomicOp::Add;
    case IrOpcode::BufferAtomicISub32:
        return MslAtomicOp::Sub;
    case IrOpcode::BufferAtomicSMin32:
        return MslAtomicOp::SMin;
    case IrOpcode::BufferAtomicUMin32:
        return MslAtomicOp::UMin;
    case IrOpcode::BufferAtomicSMax32:
        return MslAtomicOp::SMax;
    case IrOpcode::BufferAtomicUMax32:
        return MslAtomicOp::UMax;
    case IrOpcode::BufferAtomicAnd32:
        return MslAtomicOp::And;
    case IrOpcode::BufferAtomicOr32:
        return MslAtomicOp::Or;
    case IrOpcode::BufferAtomicXor32:
        return MslAtomicOp::Xor;
    case IrOpcode::BufferAtomicInc32:
        return MslAtomicOp::Inc;
    case IrOpcode::BufferAtomicDec32:
        return MslAtomicOp::Dec;
    case IrOpcode::BufferAtomicUSubSat32:
        return MslAtomicOp::USubSat;
    case IrOpcode::BufferAtomicFMin32:
        return MslAtomicOp::FMin;
    case IrOpcode::BufferAtomicFMax32:
        return MslAtomicOp::FMax;
    case IrOpcode::BufferAtomicFCmpSwap32:
        return MslAtomicOp::FCmpSwap;
    default:
        return std::nullopt;
    }
}

std::optional<MslAtomicOp> BufferAtomic64Op(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::BufferAtomicSwap64: return MslAtomicOp::Swap;
    case IrOpcode::BufferAtomicCmpSwap64: return MslAtomicOp::CmpSwap;
    case IrOpcode::BufferAtomicIAdd64: return MslAtomicOp::Add;
    case IrOpcode::BufferAtomicISub64: return MslAtomicOp::Sub;
    case IrOpcode::BufferAtomicSMin64: return MslAtomicOp::SMin;
    case IrOpcode::BufferAtomicUMin64: return MslAtomicOp::UMin;
    case IrOpcode::BufferAtomicSMax64: return MslAtomicOp::SMax;
    case IrOpcode::BufferAtomicUMax64: return MslAtomicOp::UMax;
    case IrOpcode::BufferAtomicAnd64: return MslAtomicOp::And;
    case IrOpcode::BufferAtomicOr64: return MslAtomicOp::Or;
    case IrOpcode::BufferAtomicXor64: return MslAtomicOp::Xor;
    case IrOpcode::BufferAtomicFCmpSwap64: return MslAtomicOp::FCmpSwap;
    case IrOpcode::BufferAtomicFMin64: return MslAtomicOp::FMin;
    case IrOpcode::BufferAtomicFMax64: return MslAtomicOp::FMax;
    case IrOpcode::BufferAtomicInc64: return MslAtomicOp::Inc;
    case IrOpcode::BufferAtomicDec64: return MslAtomicOp::Dec;
    default: return std::nullopt;
    }
}

[[maybe_unused]] bool IsBufferAtomic64(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::BufferAtomicSwap64:
    case IrOpcode::BufferAtomicCmpSwap64:
    case IrOpcode::BufferAtomicIAdd64:
    case IrOpcode::BufferAtomicISub64:
    case IrOpcode::BufferAtomicSMin64:
    case IrOpcode::BufferAtomicUMin64:
    case IrOpcode::BufferAtomicSMax64:
    case IrOpcode::BufferAtomicUMax64:
    case IrOpcode::BufferAtomicAnd64:
    case IrOpcode::BufferAtomicOr64:
    case IrOpcode::BufferAtomicXor64:
    case IrOpcode::BufferAtomicFCmpSwap64:
    case IrOpcode::BufferAtomicFMin64:
    case IrOpcode::BufferAtomicFMax64:
    case IrOpcode::BufferAtomicInc64:
    case IrOpcode::BufferAtomicDec64:
        return true;
    default:
        return false;
    }
}

// Buffer atomics whose operand is the identity element (add, sub, or, xor of 0) leave memory
// unchanged (SpirvMemoryInstructions.cpp Atomic32): an unused result skips the operation, a used one
// takes an atomic load of the current value instead. APS5_NO_ATOMIC_ZERO_SKIP=1 keeps the atomics.
bool HasZeroIdentity(MslAtomicOp op) {
    return op == MslAtomicOp::Add || op == MslAtomicOp::Sub || op == MslAtomicOp::Or || op == MslAtomicOp::Xor;
}

bool AtomicZeroSkipEnabled() {
    static const bool disabled = std::getenv("APS5_NO_ATOMIC_ZERO_SKIP") != nullptr;
    return !disabled;
}

// EmitRuntimeBufferAtomic: arguments {V#, index, offset, soffset, [desired,] value, exec}. 64-bit
// atomics (`wide`) run under the lock table of package B3 (MslDeviceAtomic64).
void BufferAtomic(MslEmitterState& state, const IrValue& instruction, MslAtomicOp op, bool wide = false) {
    const auto& memory = state.Memory(instruction);
    if (memory.kind != ResourceKind::Buffer) state.Fail(instruction, "must access a buffer resource");
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    const auto count = instruction.ArgumentCount();
    const bool compares = op == MslAtomicOp::CmpSwap || op == MslAtomicOp::FCmpSwap;
    const auto value = state.Arg(instruction, compares ? count - 3u : count - 2u);
    const auto comparator = compares ? state.Arg(instruction, count - 2u) : std::string("0u");
    const IrValue* operand = instruction.Argument(count - 2u)->Resolve();
    const bool nonZeroImmediate = operand->HasImmediate() && operand->Type() == IrType::U32 && operand->ImmediateU32() != 0u;
    const bool zeroIdentity = !wide && AtomicZeroSkipEnabled() && HasZeroIdentity(op) && !nonZeroImmediate;
    const bool zeroSkip = zeroIdentity && !instruction.HasUses();
    const auto result = state.Name(instruction);
    const std::uint32_t bytes = wide ? 8u : 4u;
    const std::string mask = wide ? "7u" : "3u";
    const std::string shift = wide ? "3u" : "2u";
    state.Line(result + " = " + MslType(instruction.Type()) + "(0);");
    auto active = ActiveArgument(state, instruction);
    if (zeroSkip) active = "(" + active + " && " + value + " != 0u)";
    state.Line("if (" + active + ") {");
    state.Indent();
    const auto descriptor = MakeDescriptor(state, instruction, true);
    const auto access = state.Temporary("aps5_a");
    state.Line("const Aps5BufferAccess " + access + " = " + AccessExpression(state, instruction, descriptor, 0u, bytes) + ";");
    state.Line("if (" + access + ".inBounds) {");
    state.Indent();
    const auto apply = [&](const std::string& object) {
        const auto fence = MslAtomicFence();
        if (wide) {
            const auto address = object.substr(object.find('(') + 1u, object.rfind(')') - object.find('(') - 1u);
            state.Line(result + " = " + MslDeviceAtomic64(state, instruction, op, address, value, compares ? comparator : std::string()) + ";");
            if (!fence.empty()) state.Line(fence);
            return;
        }
        const auto operation = MslAtomic32Expression(op, object, value, comparator);
        if (!zeroIdentity || zeroSkip) {
            state.Line(result + " = " + operation + ";");
        } else {
            state.Line(result + " = " + value + " != 0u ? " + operation + " : atomic_load_explicit(" + object + ", memory_order_relaxed);");
        }
        if (!fence.empty()) state.Line(fence);
    };
    if (!descriptor.gpu) {
        // A descriptor-table buffer, bound from the V# base rounded down to a dword (the base's misalignment is a
        // function constant): the element is base + offset, so an unaligned base with an offset that brings the
        // element back to a dword boundary runs, and an element off a dword boundary records an Unaligned fault and
        // does nothing (RDNA2 stops the wave). Upstream 7e4ce1ab3: a 32-bit atomic ignores the two low address bits
        // (RDNA2 ISA 8.1.7, alignment mode DWORD) and acts on the dword holding its address; only 64-bit atomics
        // off an 8-byte boundary fault.
        const auto element = state.DescriptorElement(DescriptorBindingKind::Buffers, memory.resource);
        const auto buffer = state.TableBuffer(DescriptorBindingKind::Buffers, element);
        const auto byte = state.Temporary("aps5_ab");
        state.Line("const uint " + byte + " = " + access + ".byte + " + state.MemoryByteOffset(element) + " + " + descriptor.memory + ".misalignment;");
        if (wide) {
            state.Line("if ((" + byte + " & " + mask + ") != 0u) {");
            state.Indent();
            state.Line(MslRecordFault(state, 7u, descriptor.base + " + ulong(" + access + ".byte)", std::to_string(bytes) + "u", MslB5Pc(instruction)));
            if (MslBdaStops(state)) state.Line(MslBdaStopStatement(state));
            state.Outdent();
            state.Line("} else if ((" + byte + " >> " + shift + ") < (" + buffer + ".bytes >> " + shift + ")) {");
        } else {
            // Upstream 96c692c887: with the V# base off a dword boundary, an atomic whose containing dword lies
            // before the first aligned dword of the view is out of bounds (returns 0, writes nothing); the view
            // starts at byte MemoryByteOffset + misalignment of the bound buffer.
            const auto first = "((" + state.MemoryByteOffset(element) + " + " + descriptor.memory + ".misalignment + 3u) >> 2u)";
            state.Line("if ((" + byte + " >> " + shift + ") < (" + buffer + ".bytes >> " + shift + ") && (" + byte + " >> 2u) >= " + first + ") {");
        }
        state.Indent();
        if (wide) apply("(" + buffer + ".address + ulong(" + byte + "))");
        else apply("(reinterpret_cast<device atomic_uint*>(" + buffer + ".address) + (" + byte + " >> 2u))");
        state.Outdent();
        state.Line("}");
    } else {
        // A V# in GPU memory: the element through the page table, mapped readable and writable.
        const auto overflow = state.Temporary("aps5_ov");
        const auto address = state.Temporary("aps5_aa");
        const auto physical = state.Temporary("aps5_pa");
        state.Line("bool " + overflow + " = false;");
        state.Line(std::string(wide ? "const " : "") + "ulong " + address + " = aps5_bda_add(" + descriptor.bda + ", " + descriptor.base + ", ulong(" + access + ".byte), false, " + descriptor.pc + ", " + overflow + ");");
        MslB5StopIf(state, overflow);
        if (wide) {
            state.Line("if ((" + address + " & " + mask + "l) != 0ul) {");
            state.Indent();
            state.Line("aps5_bda_fault(" + descriptor.bda + ", 7u, " + address + ", " + std::to_string(bytes) + "u, " + descriptor.pc + ");");
            if (descriptor.stops) state.Line(MslBdaStopStatement(state));
            state.Outdent();
            state.Line("} else {");
        } else {
            // Upstream 7e4ce1ab3: the dword holding the address.
            state.Line(address + " &= ~3ul;");
            state.Line("{");
        }
        state.Indent();
        state.Line("const ulong " + physical + " = aps5_bda_lookup(" + descriptor.bda + ", " + address + ", " + std::to_string(bytes) + "u, " + descriptor.pc + ", 3u, true);");
        state.Line("if (" + physical + " != 0ul) {");
        state.Indent();
        if (wide) apply("(" + physical + ")");
        else apply("reinterpret_cast<device atomic_uint*>(" + physical + ")");
        state.Line("aps5_bda_note_write(" + descriptor.bda + ".faults, " + address + ");");
        state.Outdent();
        state.Line("}");
        state.Outdent();
        state.Line("}");
    }
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
}

}

bool EmitMslBufferInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::LoadBufferU8:
    case IrOpcode::LoadBufferU16:
    case IrOpcode::LoadBufferU32:
        Load(state, instruction, 1u);
        return true;
    case IrOpcode::LoadBufferU32x2:
        Load(state, instruction, 2u);
        return true;
    case IrOpcode::LoadBufferU32x3:
        Load(state, instruction, 3u);
        return true;
    case IrOpcode::LoadBufferU32x4:
        Load(state, instruction, 4u);
        return true;
    case IrOpcode::StoreBufferU8:
    case IrOpcode::StoreBufferU16:
    case IrOpcode::StoreBufferU32:
        Store(state, instruction, 1u);
        return true;
    case IrOpcode::StoreBufferU32x2:
        Store(state, instruction, 2u);
        return true;
    case IrOpcode::StoreBufferU32x3:
        Store(state, instruction, 3u);
        return true;
    case IrOpcode::StoreBufferU32x4:
        Store(state, instruction, 4u);
        return true;
    case IrOpcode::ReadConst:
        ReadConst(state, instruction);
        return true;
    case IrOpcode::ReadConstBuffer:
        ReadConstBuffer(state, instruction);
        return true;
    default:
        break;
    }
    if (const auto op = BufferAtomicOp(instruction.Opcode())) {
        BufferAtomic(state, instruction, *op);
        return true;
    }
    if (const auto op = BufferAtomic64Op(instruction.Opcode())) {
        BufferAtomic(state, instruction, *op, true);
        return true;
    }
    return EmitMslAddressInstruction(state, instruction);
}

MslPreludeEntry MslBufferPrelude(MslPreludeChunk chunk) {
    switch (chunk) {
    case MslPreludeChunk::RuntimeBuffer:
        return {RuntimeBufferText, OnCommon};
    case MslPreludeChunk::BufferFormat:
        return {BufferFormatChunk(), OnRuntimeBuffer};
    case MslPreludeChunk::Bda:
        return {MslBdaPreludeText(), OnFaultAndBuffer};
    default:
        throw std::runtime_error("MSL emission failed: buffer prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not a buffer chunk (work package B5)");
    }
}

}
