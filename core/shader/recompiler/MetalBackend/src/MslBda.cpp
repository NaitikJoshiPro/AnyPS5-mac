// Work package B5: guest addresses through the BdaAbi page table (SpirvBdaLookup.cpp,
// SpirvBdaRead.cpp, SpirvBdaFault.cpp on the SPIR-V path) and the flat apertures
// (SpirvMemoryInstructions.cpp LoadAperture*/StoreAperture*), as MSL helper functions in the Bda
// prelude chunk, and the emitter-side helpers shared by the buffer and address emitters.

#include "MslBufferShared.hpp"
#include "BdaAbi.hpp"
#include <cstdlib>
#include <string>

namespace ShaderRecompiler {

namespace {

static_assert(static_cast<std::uint32_t>(BdaAbi::FaultReason::Unmapped) == 1u && static_cast<std::uint32_t>(BdaAbi::FaultReason::Permission) == 2u && static_cast<std::uint32_t>(BdaAbi::FaultReason::Overflow) == 3u && static_cast<std::uint32_t>(BdaAbi::FaultReason::InvalidTable) == 4u && static_cast<std::uint32_t>(BdaAbi::FaultReason::Unaligned) == 7u && static_cast<std::uint32_t>(BdaAbi::FaultReason::InvalidDescriptor) == 8u, "the Bda chunk spells the BdaAbi fault reasons as literals");
static_assert(BdaAbi::Version == 1u && sizeof(BdaAbi::Header) == 16u && sizeof(BdaAbi::Range) == 32u && BdaAbi::Read == 1u && BdaAbi::Write == 2u, "the Bda chunk spells the BdaAbi table layout as literals");
static_assert(BdaAbi::WrittenOverflowWord == 8u && BdaAbi::WrittenCountWord == 9u && BdaAbi::WrittenSlotsWord == 10u && BdaAbi::WrittenListWord == 4106u && BdaAbi::WrittenPageSlots == 2048u && BdaAbi::WrittenPageProbes == 8u, "the Bda chunk spells the written-page record layout as literals");
static_assert(BdaAbi::WrittenChunkShift == 17u && BdaAbi::WrittenPageShift == 12u && BdaAbi::WrittenChunkPages == 32u, "the Bda chunk spells the written-page granularity as literals");

// The faults recorded here follow BdaAbi.hpp and the SPIR-V functions they port (named in the
// comments); reasons: 1 Unmapped, 2 Permission, 3 Overflow, 4 InvalidTable, 7 Unaligned,
// 8 InvalidDescriptor. `stops` selects the invocation-ending variant (StopBdaInvocationIf): the
// helper then returns at the first fault with `stop` set and the caller ends the invocation.
constexpr std::string_view BdaText = R"(struct Aps5Bda {
    Aps5Buffer table;
    ulong faults;
    uint stage;
};

static inline ulong aps5_bda_word64(device const uint* words, uint index) {
    return ulong(words[index]) | (ulong(words[index + 1u]) << 32u);
}

static inline void aps5_bda_fault(Aps5Bda bda, uint reason, ulong address, uint bytes, uint pc) {
    aps5_record_fault(bda.faults, reason, address, bytes, bda.stage, pc);
}

// 0 with the device address of [address, address + bytes) in `result` when one page-table range maps
// all of it with `permission`, else the fault reason (get_bda_pointer).
static uint aps5_bda_find(Aps5Bda bda, ulong address, uint bytes, uint permission, thread ulong& result) {
    const uint length = bda.table.bytes >> 2u;
    if (bda.table.address == 0ul || length < 4u) return 4u;
    device const uint* words = reinterpret_cast<device const uint*>(bda.table.address);
    if (words[0] != 1u || words[2] != 32u || words[3] != 0u) return 4u;
    const uint count = words[1];
    const uint available = length - 4u;
    if ((available & 7u) != 0u || count != (available >> 3u)) return 4u;
    if (bytes == 0u) return 3u;
    const ulong end = address + ulong(bytes);
    if (end <= address) return 3u;
    uint low = 0u;
    uint high = count;
    for (uint step = 0u; step < 33u && low < high; ++step) {
        const uint middle = low + ((high - low) >> 1u);
        if (address < aps5_bda_word64(words, 4u + middle * 8u)) high = middle;
        else low = middle + 1u;
    }
    if (low == 0u) return 1u;
    const uint entry = 4u + (low - 1u) * 8u;
    const ulong begin = aps5_bda_word64(words, entry);
    const ulong finish = aps5_bda_word64(words, entry + 2u);
    const ulong base = aps5_bda_word64(words, entry + 4u);
    if (words[entry + 7u] != 0u || begin >= finish) return 4u;
    if (end > finish) return 1u;
    if ((words[entry + 6u] & permission) != permission) return 2u;
    if (base == 0ul) return 4u;
    const ulong mapped = base + (address - begin);
    if (mapped < base || mapped + ulong(bytes) <= mapped) return 3u;
    result = mapped;
    return 0u;
}

// The device address, or 0 after recording the fault when `record` (probe_bda_pointer records none).
static ulong aps5_bda_lookup(Aps5Bda bda, ulong address, uint bytes, uint pc, uint permission, bool record) {
    ulong mapped = 0ul;
    const uint reason = aps5_bda_find(bda, address, bytes, permission, mapped);
    if (reason == 0u) return mapped;
    if (record) aps5_bda_fault(bda, reason, address, bytes, pc);
    return 0ul;
}

// Upstream d422c760c/3265cb6d7 (bda_anchor_range): for the range a read lookup of `base` selects,
// (usable size, device address, offset of base in the range), where the range ends at the next
// range's begin when that comes first (ranges ascend by begin but may overlap). Size 0 when nothing
// maps base, the device addresses wrap, the range ends above 0xffffffff00000000 (base plus a 32-bit
// offset could wrap) or is longer than 2^62 bytes.
static ulong3 aps5_bda_anchor(Aps5Bda bda, ulong base) {
    ulong mapped = 0ul;
    if (aps5_bda_find(bda, base, 1u, 1u, mapped) != 0u) return ulong3(0ul);
    device const uint* words = reinterpret_cast<device const uint*>(bda.table.address);
    const uint count = words[1];
    uint low = 0u;
    uint high = count;
    for (uint step = 0u; step < 33u && low < high; ++step) {
        const uint middle = low + ((high - low) >> 1u);
        if (base < aps5_bda_word64(words, 4u + middle * 8u)) high = middle;
        else low = middle + 1u;
    }
    const uint entry = 4u + (low - 1u) * 8u;
    const ulong begin = aps5_bda_word64(words, entry);
    ulong bound = aps5_bda_word64(words, entry + 2u);
    if (low < count) bound = min(bound, aps5_bda_word64(words, 4u + low * 8u));
    const ulong device = aps5_bda_word64(words, entry + 4u);
    const ulong span = bound - begin;
    const bool usable = device != 0ul && device + span > device && bound <= 0xffffffff00000000ul && span <= (1ul << 62u);
    return ulong3(usable ? span : 0ul, device, base - begin);
}

// address + offset (or - offset) with an Overflow fault when it wraps (AddBdaAddress).
static inline ulong aps5_bda_add(Aps5Bda bda, ulong address, ulong offset, bool subtract, uint pc, thread bool& overflow) {
    const ulong result = subtract ? address - offset : address + offset;
    if (subtract ? offset > address : result < address) {
        aps5_bda_fault(bda, 3u, address, 0u, pc);
        overflow = true;
    }
    return result;
}

// address + a signed 32-bit offset (AddBdaImmediate, AddBdaSignedOffset).
static inline ulong aps5_bda_add_signed(Aps5Bda bda, ulong address, uint offset, uint pc, thread bool& overflow) {
    const bool negative = int(offset) < 0;
    return aps5_bda_add(bda, address, ulong(negative ? 0u - offset : offset), negative, pc, overflow);
}

// An Overflow fault when [address, address + bytes) wraps (EmitBdaOverflowCheck).
static inline bool aps5_bda_overflows(Aps5Bda bda, ulong address, uint bytes, uint pc) {
    if (address <= 0xfffffffffffffffful - ulong(bytes)) return false;
    aps5_bda_fault(bda, 3u, address, bytes, pc);
    return true;
}

// 1-4 bytes at address, one lookup each (EmitBdaBytes): an unmapped byte reads zero after its fault
// is recorded.
static uint aps5_bda_read_bytes(Aps5Bda bda, ulong address, uint bytes, uint pc, bool stops, thread bool& stop) {
    uint value = 0u;
    for (uint byte = 0u; byte < bytes; ++byte) {
        const ulong physical = aps5_bda_lookup(bda, address + ulong(byte), 1u, pc, 1u, true);
        if (physical == 0ul) {
            if (stops) {
                stop = true;
                return 0u;
            }
            continue;
        }
        value |= uint(*reinterpret_cast<device const uchar*>(physical)) << (byte * 8u);
    }
    return value;
}

// An 8- or 16-bit read (EmitBdaRead).
static uint aps5_bda_read_small(Aps5Bda bda, ulong address, uint bits, uint pc, bool stops, thread bool& stop) {
    if (aps5_bda_overflows(bda, address, bits / 8u, pc) && stops) {
        stop = true;
        return 0u;
    }
    return aps5_bda_read_bytes(bda, address, bits / 8u, pc, stops, stop);
}

// One dword by bytes (read_bda_dword_bytes); a coherent read of a mapped, aligned dword is one atomic
// load.
static uint aps5_bda_read_dword(Aps5Bda bda, ulong address, uint pc, bool glc, bool stops, thread bool& stop) {
    if (aps5_bda_overflows(bda, address, 4u, pc) && stops) {
        stop = true;
        return 0u;
    }
    if (glc) {
        const ulong physical = aps5_bda_lookup(bda, address, 4u, pc, 1u, false);
        if (physical != 0ul && (physical & 3ul) == 0ul) return atomic_load_explicit(reinterpret_cast<device atomic_uint*>(physical), memory_order_relaxed);
    }
    return aps5_bda_read_bytes(bda, address, 4u, pc, stops, stop);
}

// The dwords of `used` (bit i: the dword at address + offset + 4i) of a 1-4 dword load
// (read_bda_span): one probe of the span from the first to the last used dword; a span inside one
// range at a 4-aligned device address is read dword-wise, anything else falls back to the per-dword
// byte reads, which record every fault exactly as separate dword loads would.
static uint4 aps5_bda_read_span(Aps5Bda bda, ulong address, uint offset, uint used, uint pc, bool glc, bool stops, thread bool& stop) {
    uint4 values = uint4(0u);
    if (used == 0u) return values;
    const uint first = ctz(used);
    const uint last = 31u - clz(used);
    const uint firstBytes = first * 4u;
    bool overflow = false;
    const ulong firstAddress = aps5_bda_add_signed(bda, address, offset + firstBytes, pc, overflow);
    if (overflow && stops) {
        stop = true;
        return values;
    }
    const ulong physical = aps5_bda_lookup(bda, firstAddress, (last + 1u) * 4u - firstBytes, pc, 1u, false);
    if (physical != 0ul && (physical & 3ul) == 0ul) {
        for (uint dword = first; dword <= last; ++dword) {
            if ((used & (1u << dword)) == 0u) continue;
            const ulong at = physical + ulong(dword * 4u - firstBytes);
            values[dword] = glc ? atomic_load_explicit(reinterpret_cast<device atomic_uint*>(at), memory_order_relaxed) : *reinterpret_cast<device const uint*>(at);
        }
        return values;
    }
    for (uint dword = first; dword <= last; ++dword) {
        if ((used & (1u << dword)) == 0u) continue;
        ulong dwordAddress = firstAddress;
        if (dword != first) {
            dwordAddress = aps5_bda_add_signed(bda, address, offset + dword * 4u, pc, overflow);
            if (overflow && stops) {
                stop = true;
                return values;
            }
        }
        values[dword] = aps5_bda_read_dword(bda, dwordAddress, pc, glc, stops, stop);
        if (stop) return values;
    }
    return values;
}

// The written-page record of the fault buffer (note_bda_write): the guest page of `address` joins its
// 128 KiB chunk's bit mask in an open-addressed table of 2048 slots; a full probe sequence sets the
// overflow word, after which the driver treats every page as written.
static void aps5_bda_note_write(ulong faults, ulong address) {
    if (faults == 0ul) return;
    device atomic_uint* words = reinterpret_cast<device atomic_uint*>(faults);
    const uint key = uint(address >> 17u) + 1u;
    const uint bit = 1u << (uint(address >> 12u) & 31u);
    const uint hash = (key * 0x9e3779b1u) >> 21u;
    bool done = false;
    for (uint probe = 0u; probe < 8u && !done; ++probe) {
        const uint slot = (hash + probe) & 2047u;
        const uint keyWord = 10u + slot * 2u;
        done = atomic_load_explicit(words + keyWord, memory_order_relaxed) == key;
        if (!done) {
            const uint previous = aps5_atomic_cas(words + keyWord, 0u, key);
            const bool claimed = previous == 0u;
            done = claimed || previous == key;
            if (claimed) {
                const uint index = atomic_fetch_add_explicit(words + 9u, 1u, memory_order_relaxed);
                if (index < 2048u) atomic_store_explicit(words + 4106u + index, slot, memory_order_relaxed);
            }
        }
        if (done && (atomic_load_explicit(words + keyWord + 1u, memory_order_relaxed) & bit) == 0u) atomic_fetch_or_explicit(words + keyWord + 1u, bit, memory_order_relaxed);
    }
    if (!done) atomic_store_explicit(words + 8u, 1u, memory_order_relaxed);
}

// One lookup for `bytes` (1 or 4) at address, the store and its written page (EmitBdaStoreAt).
static void aps5_bda_store_at(Aps5Bda bda, ulong address, uint bytes, uint value, uint pc, bool glc) {
    const ulong physical = aps5_bda_lookup(bda, address, bytes, pc, 2u, true);
    if (physical == 0ul) return;
    if (bytes == 4u) {
        if (glc) atomic_store_explicit(reinterpret_cast<device atomic_uint*>(physical), value, memory_order_relaxed);
        else *reinterpret_cast<device uint*>(physical) = value;
    } else {
        *reinterpret_cast<device uchar*>(physical) = uchar(value);
    }
    aps5_bda_note_write(bda.faults, address);
}

// A flat/global store of 8, 16 or 32 bits (EmitBdaStore): bytewise unless an aligned dword.
static void aps5_bda_store(Aps5Bda bda, ulong address, uint value, uint bits, uint pc, bool glc) {
    if (bits == 32u && (address & 3ul) == 0ul) {
        aps5_bda_store_at(bda, address, 4u, value, pc, glc);
        return;
    }
    for (uint byte = 0u; byte < bits / 8u; ++byte) aps5_bda_store_at(bda, address + ulong(byte), 1u, value >> (byte * 8u), pc, glc);
}

// A write through a V# in GPU memory (EmitBdaWrite): 8- and 16-bit writes and, since upstream #4518,
// unaligned dwords are stored byte by byte (after the overflow check); aligned dwords at once.
static void aps5_bda_write(Aps5Bda bda, ulong address, uint value, uint bits, uint pc, bool glc, bool stops, thread bool& stop) {
    if (bits != 32u || (address & 3ul) != 0ul) {
        if (aps5_bda_overflows(bda, address, bits / 8u, pc) && stops) {
            stop = true;
            return;
        }
        for (uint byte = 0u; byte < bits / 8u; ++byte) aps5_bda_store_at(bda, address + ulong(byte), 1u, value >> (byte * 8u), pc, glc);
        return;
    }
    aps5_bda_store_at(bda, address, 4u, value, pc, glc);
}

// A 2-4 dword flat/global store at address + offset (EmitBdaDwordWrites): one probe of the whole span;
// a span inside one range with aligned guest and device addresses is stored dword-wise and noted
// once per written page, anything else dword by dword.
static void aps5_bda_store_dwords(Aps5Bda bda, ulong address, uint offset, uint dwords, uint4 values, uint pc, bool glc, bool stops, thread bool& stop) {
    ulong addresses[4] = {0ul, 0ul, 0ul, 0ul};
    for (uint dword = 0u; dword < dwords; ++dword) {
        bool overflow = false;
        addresses[dword] = aps5_bda_add_signed(bda, address, offset + dword * 4u, pc, overflow);
        if (overflow && stops) {
            stop = true;
            return;
        }
    }
    const ulong physical = aps5_bda_lookup(bda, addresses[0], dwords * 4u, pc, 2u, false);
    if (physical != 0ul && (addresses[0] & 3ul) == 0ul && (physical & 3ul) == 0ul) {
        for (uint dword = 0u; dword < dwords; ++dword) {
            device uint* word = reinterpret_cast<device uint*>(physical) + dword;
            if (glc) atomic_store_explicit(reinterpret_cast<device atomic_uint*>(word), values[dword], memory_order_relaxed);
            else *word = values[dword];
        }
        aps5_bda_note_write(bda.faults, addresses[0]);
        if ((addresses[0] >> 12u) != (addresses[dwords - 1u] >> 12u)) aps5_bda_note_write(bda.faults, addresses[dwords - 1u]);
        return;
    }
    for (uint dword = 0u; dword < dwords; ++dword) aps5_bda_store(bda, addresses[dword], values[dword], 32u, pc, glc);
}

// The device address of an aligned 4-byte atomic element (EmitBdaAtomic with `permission` Write,
// EmitRuntimeBufferAtomic with Read | Write): an unaligned element records an Unaligned fault and
// yields 0, as does an unmapped one (after its lookup fault).
static ulong aps5_bda_atomic_address(Aps5Bda bda, ulong address, uint bytes, uint permission, uint pc) {
    if ((address & ulong(bytes - 1u)) != 0ul) {
        aps5_bda_fault(bda, 7u, address, bytes, pc);
        return 0ul;
    }
    return aps5_bda_lookup(bda, address, bytes, pc, permission, true);
}

// Memory of a V# that lives in GPU memory: byte offsets from the V# base through the page table (the
// BufferFormat templates and the buffer emitters read and write through it).
struct Aps5BdaMemory {
    Aps5Bda bda;
    ulong base;
    uint pc;
    bool glc;
    bool stops;

    uint read(uint byte, uint bits, thread bool& stop) const {
        bool overflow = false;
        const ulong address = aps5_bda_add(bda, base, ulong(byte), false, pc, overflow);
        if (overflow && stops) {
            stop = true;
            return 0u;
        }
        if (bits == 32u) return aps5_bda_read_span(bda, address, 0u, 1u, pc, glc, stops, stop).x;
        return aps5_bda_read_small(bda, address, bits, pc, stops, stop);
    }

    // A coherent dwordx2 (Aps5BufferMemory::read2): dword by dword through the page table.
    uint2 read2(uint byte, thread bool& stop) const {
        const uint low = read(byte, 32u, stop);
        return uint2(low, stop ? 0u : read(byte + 4u, 32u, stop));
    }

    void write(uint byte, uint value, uint bits, thread bool& stop) const {
        bool overflow = false;
        const ulong address = aps5_bda_add(bda, base, ulong(byte), false, pc, overflow);
        if (overflow && stops) {
            stop = true;
            return;
        }
        aps5_bda_write(bda, address, value, bits, pc, glc, stops, stop);
    }
};

// Flat accesses to the shared (LDS) and private (scratch) apertures (SpirvMemoryInstructions.cpp
// LoadAperture*/StoreAperture*): byte offsets into `words[length]`. A read outside the storage (or
// whose byte offset wraps) is zero, a store there is dropped. LDS sub-dword stores merge with
// atomics, scratch is the invocation's own.
static inline void aps5_aperture_merge(threadgroup uint* word, uint mask, uint value) {
    threadgroup atomic_uint* bits = reinterpret_cast<threadgroup atomic_uint*>(word);
    uint observed = atomic_load_explicit(bits, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(bits, &observed, (observed & ~mask) | (value & mask), memory_order_relaxed, memory_order_relaxed)) {
    }
}

// Device-memory LDS (the WorkgroupMemory arena).
static inline void aps5_aperture_merge(device uint* word, uint mask, uint value) {
    device atomic_uint* bits = reinterpret_cast<device atomic_uint*>(word);
    uint observed = atomic_load_explicit(bits, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(bits, &observed, (observed & ~mask) | (value & mask), memory_order_relaxed, memory_order_relaxed)) {
    }
}

static inline void aps5_aperture_merge(thread uint* word, uint mask, uint value) {
    *word = (*word & ~mask) | (value & mask);
}

template <typename Words>
static inline uint aps5_aperture_byte(Words words, uint length, uint byteOffset, uint offset) {
    const uint address = byteOffset + offset;
    if (byteOffset > 0xffffffffu - offset || (address >> 2u) >= length) return 0u;
    return (words[address >> 2u] >> ((address & 3u) << 3u)) & 0xffu;
}

template <typename Words>
static inline void aps5_aperture_store_byte(Words words, uint length, uint byteOffset, uint offset, uint data) {
    const uint address = byteOffset + offset;
    if (byteOffset > 0xffffffffu - offset || (address >> 2u) >= length) return;
    const uint shift = (address & 3u) << 3u;
    aps5_aperture_merge(words + (address >> 2u), 0xffu << shift, (data & 0xffu) << shift);
}

// LoadApertureElement: an element inside one dword is read from it, anything else byte by byte.
template <typename Words>
static uint aps5_aperture_load(Words words, uint length, uint byteOffset, uint bits) {
    if (bits == 8u) return aps5_aperture_byte(words, length, byteOffset, 0u);
    const uint last = bits / 8u - 1u;
    if (byteOffset > 0xffffffffu - last || ((byteOffset + last) >> 2u) >= length) return 0u;
    if ((byteOffset & 3u) + bits / 8u <= 4u) {
        const uint word = words[byteOffset >> 2u];
        return bits == 32u ? word : (word >> ((byteOffset & 3u) << 3u)) & 0xffffu;
    }
    uint value = 0u;
    for (uint byte = 0u; byte < bits / 8u; ++byte) value |= aps5_aperture_byte(words, length, byteOffset, byte) << (byte * 8u);
    return value;
}

// StoreApertureElement.
template <typename Words>
static void aps5_aperture_store(Words words, uint length, uint byteOffset, uint bits, uint data) {
    if (bits == 8u) {
        aps5_aperture_store_byte(words, length, byteOffset, 0u, data);
        return;
    }
    const uint last = bits / 8u - 1u;
    if (byteOffset > 0xffffffffu - last || ((byteOffset + last) >> 2u) >= length) return;
    if ((byteOffset & 3u) + bits / 8u <= 4u) {
        if (bits == 32u) {
            words[byteOffset >> 2u] = data;
            return;
        }
        const uint shift = (byteOffset & 3u) << 3u;
        aps5_aperture_merge(words + (byteOffset >> 2u), 0xffffu << shift, (data & 0xffffu) << shift);
        return;
    }
    for (uint byte = 0u; byte < bits / 8u; ++byte) aps5_aperture_store_byte(words, length, byteOffset, byte, data >> (byte * 8u));
}

// A 32-bit flat atomic on the private aperture (ApertureAtomic32 for scratch): the invocation's own
// memory, read, combined and written back. The caller checks alignment and bounds.
static inline uint aps5_scratch_atomic(thread uint* word, uint op, uint value, uint comparator) {
    const uint old = *word;
    *word = aps5_atomic_next(op, old, value, comparator);
    return old;
}

)";

std::uint32_t StageLiteral(const MslEmitterState& state) {
    return static_cast<std::uint32_t>(state.program.Resources().stage);
}

}

std::string MslB5Tag(const MslEmitterState& state, const IrValue& instruction) {
    return "_" + std::to_string(instruction.Resolve()->Id()) + (state.half != 0u ? "h" : "");
}

std::string MslB5Pc(const IrValue& instruction) {
    return MslHex(instruction.Flags<MemoryFlags>().pc) + "u";
}

std::string MslFaultBuffer(MslEmitterState& state) {
    state.usesDriverData = true;
    return "aps5_driver.faultBuffer";
}

std::string MslRecordFault(MslEmitterState& state, std::uint32_t reason, std::string_view address, std::string_view bytes, std::string_view pc) {
    state.RequirePrelude(MslPreludeChunk::FaultRecord);
    return "aps5_record_fault(" + MslFaultBuffer(state) + ", " + std::to_string(reason) + "u, " + std::string(address) + ", " + std::string(bytes) + ", " + std::to_string(StageLiteral(state)) + "u, " + std::string(pc) + ");";
}

std::string MslBdaContext(MslEmitterState& state) {
    state.RequirePrelude(MslPreludeChunk::Bda);
    if (state.prologue.find("Aps5Bda aps5_bda ") == std::string::npos) {
        const auto table = state.TableBuffer(DescriptorBindingKind::BdaPagetable, 0u);
        state.AddPrologue("const Aps5Bda aps5_bda {" + table + ", " + MslFaultBuffer(state) + ", " + std::to_string(StageLiteral(state)) + "u};");
    }
    return "aps5_bda";
}

bool MslBdaStops(const MslEmitterState& state) {
    // SpirvEmitter: bdaStopsInvocations for every stage but mesh (its output writes are shared by
    // the workgroup) when no wave-LDS barrier or workgroup barrier would be left incomplete.
    if (state.program.Resources().stage == IrShaderStage::Mesh || state.waveLdsScope != MslBarrierScope::None) return false;
    // BdaInvocationsMayStop: programs with workgroup barriers keep every invocation running.
    for (const auto* block : state.program.BlockOrder()) {
        for (const auto* instruction : block->Instructions()) {
            if (instruction->Opcode() == IrOpcode::Barrier) return false;
        }
    }
    return true;
}

std::string MslBdaStopStatement(const MslEmitterState& state) {
    // StopBdaInvocationIf: pixel invocations are killed (OpKill), so the faulting fragment writes no
    // colour, depth or sample mask; after discard_fragment its buffer and texture writes are
    // inhibited, which also covers the raster-order ticket write-back of the stage function (raster
    // order groups order the accesses themselves, so no later fragment waits on that value).
    return state.program.Resources().stage == IrShaderStage::Pixel ? "{ discard_fragment(); return; }" : "return;";
}

void MslB5StopIf(MslEmitterState& state, std::string_view flag) {
    if (MslBdaStops(state)) state.Line("if (" + std::string(flag) + ") " + MslBdaStopStatement(state));
}

std::uint32_t MslUsedDwords(const IrValue& instruction, std::uint32_t dwords) {
    const auto all = (1u << dwords) - 1u;
    if (dwords == 1u) return all;
    std::uint32_t used = 0;
    for (const IrValue* user : instruction.Uses()) {
        const auto opcode = user->Opcode();
        if (opcode != IrOpcode::CompositeExtractU32x2 && opcode != IrOpcode::CompositeExtractU32x3 && opcode != IrOpcode::CompositeExtractU32x4) return all;
        const IrValue* index = user->ArgumentCount() > 1u ? user->Argument(1) : nullptr;
        if (index == nullptr || !index->HasImmediate() || index->ImmediateU32() >= dwords) return all;
        used |= 1u << index->ImmediateU32();
    }
    return used;
}

MslB5Storage MslB5Lds(MslEmitterState& state) {
    return MslLdsStorage(state);
}

MslB5Storage MslB5Scratch(MslEmitterState& state) {
    const auto dwords = state.program.Info().scratchDwords;
    if (dwords == 0u) return {};
    const std::string name = state.half != 0u ? "aps5_scratch_h" : "aps5_scratch";
    if (state.prologue.find(" " + name + "[") == std::string::npos) state.AddPrologue("uint " + name + "[" + std::to_string(dwords) + "] = {};");
    return {name, dwords};
}

std::string MslAtomic32Expression(MslAtomicOp op, std::string_view object, std::string_view value, std::string_view comparator) {
    const std::string target(object);
    const std::string operand(value);
    switch (op) {
    case MslAtomicOp::Swap:
        return "atomic_exchange_explicit(" + target + ", " + operand + ", memory_order_relaxed)";
    case MslAtomicOp::CmpSwap:
        return "aps5_atomic_cas(" + target + ", " + std::string(comparator) + ", " + operand + ")";
    case MslAtomicOp::Add:
        return "atomic_fetch_add_explicit(" + target + ", " + operand + ", memory_order_relaxed)";
    case MslAtomicOp::Sub:
        return "atomic_fetch_sub_explicit(" + target + ", " + operand + ", memory_order_relaxed)";
    case MslAtomicOp::SMin:
        return "aps5_atomic_smin(" + target + ", " + operand + ")";
    case MslAtomicOp::UMin:
        return "atomic_fetch_min_explicit(" + target + ", " + operand + ", memory_order_relaxed)";
    case MslAtomicOp::SMax:
        return "aps5_atomic_smax(" + target + ", " + operand + ")";
    case MslAtomicOp::UMax:
        return "atomic_fetch_max_explicit(" + target + ", " + operand + ", memory_order_relaxed)";
    case MslAtomicOp::And:
        return "atomic_fetch_and_explicit(" + target + ", " + operand + ", memory_order_relaxed)";
    case MslAtomicOp::Or:
        return "atomic_fetch_or_explicit(" + target + ", " + operand + ", memory_order_relaxed)";
    case MslAtomicOp::Xor:
        return "atomic_fetch_xor_explicit(" + target + ", " + operand + ", memory_order_relaxed)";
    case MslAtomicOp::Inc:
    case MslAtomicOp::Dec:
    case MslAtomicOp::USubSat:
    case MslAtomicOp::FMin:
    case MslAtomicOp::FMax:
    case MslAtomicOp::FCmpSwap:
        return "aps5_atomic_update(" + target + ", " + std::to_string(static_cast<std::uint32_t>(op)) + "u, " + operand + ", " + std::string(comparator.empty() ? "0u" : comparator) + ")";
    }
    throw std::runtime_error("MSL emission failed: unknown 32-bit atomic operation (work package B5)");
}

std::string MslAtomicFence() {
    // GCN atomics imply no fence; the SPIR-V path follows every buffer and flat atomic with a
    // device-scope acquire-release barrier (the acquire side of lock/publish patterns), which MSL 4.0
    // spells as a seq_cst fence. APS5_NO_ATOMIC_BARRIER=1 drops it, as on the SPIR-V path.
    static const bool disabled = std::getenv("APS5_NO_ATOMIC_BARRIER") != nullptr;
    if (disabled) return {};
    return "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);";
}

std::string_view MslBdaPreludeText() {
    return BdaText;
}

}
