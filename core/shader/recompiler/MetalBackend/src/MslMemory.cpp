// Work package B3: LDS and GDS (ds_* reads, writes, atomics, append/consume) and the 64-bit atomic
// emulation (SpirvMemoryInstructions.cpp: word access, the atomic core, LockedLdsUpdate, the 64-bit
// LDS atomics, AppendConsume).
//
// LDS is `threadgroup uint aps5_lds[N (+1 lock dword)]` when it fits Apple's 32 KiB of threadgroup
// memory, else the workgroup's slice of the device arena at MslAbi::BufferIndex::WorkgroupMemory
// (`device uint* aps5_lds`, stride Reflection::workgroupMemoryStrideDwords). GDS is the descriptor
// table's Gds buffer. Apple GPUs have no 64-bit atomics on threadgroup memory and only min/max on
// device memory, so a program with 64-bit LDS atomics (SpirvRequirements::ldsLock) serialises every
// LDS atomic through the lock dword after the LDS: one elected invocation per SIMD group tries the
// lock per round (never spins alone), the whole SIMD group loops until each of its invocations was
// served, and a round limit records a LoopLimit fault instead of hanging the GPU.

#include "MslBufferShared.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace ShaderRecompiler {

namespace {

// Rounds of the LDS / atomic lock loop before it gives up with a LoopLimit fault (a correct program
// needs at most one round per contending SIMD group of the dispatch).
constexpr std::uint32_t LockRoundLimit = 0x100000u;

constexpr std::string_view LdsLockText = R"(// Float atomics on bits (AtomicFloatBits): FLOAT_MODE denormal flush, NaN-aware min/max, compare
// with -0 == +0 (work package B3).
static inline uint aps5_fbits_flush(uint value, bool flush) {
    return flush && (value & 0x7fffffffu) < 0x00800000u ? (value & 0x80000000u) : value;
}

// ds_add_f32 as measured on gfx1010/gfx1031 (upstream 731575c80): operands flushed to a signed zero
// when FLOAT_MODE denormal bit 0 is clear, the result when bit 1 is clear; a NaN data operand stores
// quieted, else a NaN old value stores quieted, and +inf + -inf stores 0xffc00000.
static inline uint aps5_lds_fadd32(uint old, uint data, bool flushInputs, bool flushResult) {
    if ((data & 0x7fffffffu) > 0x7f800000u) return data | 0x00400000u;
    if ((old & 0x7fffffffu) > 0x7f800000u) return old | 0x00400000u;
    const uint sum = aps5_fbits_flush(as_type<uint>(as_type<float>(aps5_fbits_flush(old, flushInputs)) + as_type<float>(aps5_fbits_flush(data, flushInputs))), flushResult);
    return (sum & 0x7fffffffu) > 0x7f800000u ? 0xffc00000u : sum;
}

static inline ulong aps5_fbits_flush(ulong value, bool flush) {
    return flush && (value & 0x7ffffffffffffffful) < 0x0010000000000000ul ? (value & 0x8000000000000000ul) : value;
}

static inline bool aps5_f64_bits_nan(ulong value) {
    return (value & 0x7ffffffffffffffful) > 0x7ff0000000000000ul;
}

static inline ulong aps5_f64_bits_order(ulong value) {
    return (value & 0x8000000000000000ul) != 0ul ? ~value : (value | 0x8000000000000000ul);
}

static ulong aps5_f64_bits_min_max(ulong old, ulong source, bool maximum) {
    const bool better = maximum ? aps5_f64_bits_order(source) > aps5_f64_bits_order(old) : aps5_f64_bits_order(source) < aps5_f64_bits_order(old);
    ulong result = better ? source : old;
    result = aps5_f64_bits_nan(old) ? source : result;
    result = aps5_f64_bits_nan(source) ? old : result;
    result = aps5_f64_bits_nan(source) && aps5_f64_bits_nan(old) ? source : result;
    result = aps5_f64_bits_nan(old) && (old & 0x0008000000000000ul) == 0ul ? (old | 0x0008000000000000ul) : result;
    return aps5_f64_bits_nan(source) && (source & 0x0008000000000000ul) == 0ul ? (source | 0x0008000000000000ul) : result;
}

static inline bool aps5_f64_bits_equal(ulong left, ulong right) {
    return (left == right && !aps5_f64_bits_nan(left)) || ((left | right) & 0x7ffffffffffffffful) == 0ul;
}

)";

// 64-bit atomics on device memory (buffer, flat and global *_x2 atomics). Apple9 has 64-bit device
// atomics only as atomic_max/min without a result, so every 64-bit read-modify-write takes a lock
// word of the driver's lock table (DriverData.atomicLocks, MslAbi::AtomicLockWords dwords hashed by address): one
// elected invocation per SIMD group per round, as the LDS lock.
static_assert(MslAbi::AtomicLockWords == 1024u, "aps5_atomic64_lock masks the hash with AtomicLockWords - 1");
constexpr std::string_view Atomics64Text = R"(static ulong aps5_atomic64_next(uint op, ulong old, ulong value, ulong comparator) {
    switch (op) {
    case 0u: return value;
    case 1u: return old == comparator ? value : old;
    case 2u: return old + value;
    case 3u: return old - value;
    case 4u: return as_type<long>(old) < as_type<long>(value) ? old : value;
    case 5u: return old < value ? old : value;
    case 6u: return as_type<long>(old) > as_type<long>(value) ? old : value;
    case 7u: return old > value ? old : value;
    case 8u: return old & value;
    case 9u: return old | value;
    case 10u: return old ^ value;
    case 11u: return old >= value ? 0ul : old + 1ul;
    case 12u: return (old == 0ul || old > value) ? value : old - 1ul;
    case 13u: return old >= value ? old - value : 0ul;
    case 14u: return aps5_f64_bits_min_max(old, value, false);
    case 15u: return aps5_f64_bits_min_max(old, value, true);
    default: return aps5_f64_bits_equal(old, comparator) ? value : old;
    }
}

static inline device atomic_uint* aps5_atomic64_lock(ulong locks, ulong address) {
    const uint hash = uint(address >> 3u) ^ uint(address >> 13u) ^ uint(address >> 32u);
    return reinterpret_cast<device atomic_uint*>(locks) + (hash & 1023u);
}

)";

constexpr std::array<MslPreludeChunk, 1> OnLdsLock {MslPreludeChunk::LdsLock};

std::string U32(std::uint32_t value) {
    return std::to_string(value) + "u";
}

std::string Active(MslEmitterState& state, const IrValue& instruction) {
    return state.Arg(instruction, instruction.ArgumentCount() - 1u);
}

// Fences: MSL 4.1 has acquire/release orders on the mem_flags fences.
std::string Fence(std::string_view flags, std::string_view order, std::string_view scope) {
    return "atomic_thread_fence(" + std::string(flags) + ", " + std::string(order) + ", " + std::string(scope) + ");";
}

// LDS or GDS as seen by one instruction.
struct Shared {
    std::string words;   // uint array or pointer expression
    std::string length;  // dwords, an MSL expression
    std::string space;   // "threadgroup" or "device"
    std::string flags;   // fence flags for this memory
    std::string scope;   // fence scope
    bool lds = true;
    bool locked = false; // LDS atomics go through the lock (SpirvRequirements::ldsLock)
};

Shared SharedOf(MslEmitterState& state, const IrValue& instruction) {
    const auto& memory = state.Memory(instruction);
    Shared shared;
    if (memory.kind == ResourceKind::Lds && !state.ThreadgroupStage()) {
        // Function LDS (SpirvRequirements::functionLds): graphics stages keep LDS per invocation, at the
        // constant lane addresses the front end assigns (FunctionLdsLaneAddresses), as on the SPIR-V path.
        if (!state.requirements.functionLds) state.Fail(instruction, "LDS in a graphics stage without function LDS");
        const auto dwords = state.requirements.functionLdsDwords != 0u ? state.requirements.functionLdsDwords : FunctionLdsDwordLimit;
        if (state.prologue.find(" aps5_flds[") == std::string::npos) state.AddPrologue("uint aps5_flds[" + std::to_string(dwords) + "] = {};");
        shared.words = "aps5_flds";
        shared.length = U32(dwords);
        shared.space = "thread";
        shared.flags = "mem_flags::mem_none";
        shared.scope = "thread_scope_thread";
        return shared;
    }
    if (memory.kind == ResourceKind::Lds) {
        const auto storage = MslLdsStorage(state);
        if (storage.words.empty()) state.Fail(instruction, "accesses LDS, but the workgroup has none");
        shared.words = storage.words;
        shared.length = U32(storage.dwords);
        shared.space = state.LdsInDeviceMemory() ? "device" : "threadgroup";
        shared.flags = state.LdsMemoryFlags();
        shared.scope = "thread_scope_threadgroup";
        shared.locked = state.requirements.ldsLock;
        return shared;
    }
    if (memory.kind != ResourceKind::Gds) state.Fail(instruction, "must access LDS or GDS");
    // One GDS buffer per program (the SPIR-V path's single gdsVariable), whatever resource id the
    // front end gave the access.
    static_cast<void>(memory.resource);
    const std::uint32_t element = 0u;
    const auto name = "aps5_gds" + std::to_string(element);
    if (state.prologue.find(" " + name + " ") == std::string::npos) {
        const auto table = state.TableBuffer(DescriptorBindingKind::Gds, element);
        state.AddPrologue("device uint* " + name + " = reinterpret_cast<device uint*>(" + table + ".address);");
        state.AddPrologue("const uint " + name + "_dwords = " + table + ".bytes >> 2u;");
    }
    shared.words = name;
    shared.length = name + "_dwords";
    shared.space = "device";
    shared.flags = "mem_flags::mem_device";
    shared.scope = "thread_scope_device";
    shared.lds = false;
    return shared;
}

std::string AtomicPointer(const Shared& shared, const std::string& index) {
    return "(reinterpret_cast<" + shared.space + " atomic_uint*>(" + shared.words + ") + (" + index + "))";
}

// Upstream #4279 (SpirvEmitterHelpers DefineDescriptors: the gds buffer is Coherent): GDS is one
// memory for the whole GPU, so a plain ds_read / ds_write on GDS sees other threadgroups' writes.
// Plain device loads may hit a stale cache line (MslRuntimeBuffer.cpp, GLC reads), so GDS words go
// through relaxed atomic loads and stores; LDS words stay plain.
std::string ReadWord(const Shared& shared, const std::string& index) {
    if (!shared.lds) return "atomic_load_explicit(" + AtomicPointer(shared, index) + ", memory_order_relaxed)";
    return shared.words + "[" + index + "]";
}

std::string WriteWord(const Shared& shared, const std::string& index, const std::string& value) {
    if (!shared.lds) return "atomic_store_explicit(" + AtomicPointer(shared, index) + ", uint(" + value + "), memory_order_relaxed);";
    return shared.words + "[" + index + "] = " + value + ";";
}

// ByteAddress: the address register plus the instruction offset (function-LDS addresses are
// constants).
std::string ByteAddress(MslEmitterState& state, const IrValue& instruction) {
    const auto& memory = state.Memory(instruction);
    if (memory.kind == ResourceKind::Lds) {
        if (const auto found = state.requirements.functionLdsAddresses.find(&instruction); found != state.requirements.functionLdsAddresses.end()) return U32(found->second + memory.offset);
    }
    const auto name = state.Temporary("aps5_da");
    state.Line("const uint " + name + " = " + state.Arg(instruction, 0u) + (memory.offset != 0u ? " + " + U32(memory.offset) : std::string()) + ";");
    return name;
}

void Load(MslEmitterState& state, const IrValue& instruction, std::uint32_t bits, std::uint32_t components) {
    const auto result = state.Name(instruction);
    const auto type = MslType(instruction.Type());
    state.Line(result + " = " + type + "(0);");
    state.Line("if (" + Active(state, instruction) + ") {");
    state.Indent();
    const auto shared = SharedOf(state, instruction);
    const auto address = ByteAddress(state, instruction);
    if (bits < 32u) {
        const auto index = state.Temporary("aps5_di");
        state.Line("const uint " + index + " = " + address + " >> 2u;");
        state.Line("if (" + index + " < " + shared.length + ") " + result + " = " + type + "((" + ReadWord(shared, index) + " >> ((" + address + " & 3u) << 3u)) & " + (bits == 8u ? "0xffu" : "0xffffu") + ");");
    } else if (components == 1u) {
        const auto index = state.Temporary("aps5_di");
        state.Line("const uint " + index + " = " + address + " >> 2u;");
        state.Line("if (" + index + " < " + shared.length + ") " + result + " = " + ReadWord(shared, index) + ";");
    } else {
        static constexpr const char* Components[] = {".x", ".y", ".z", ".w"};
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto index = state.Temporary("aps5_di");
            state.Line("const uint " + index + " = (" + address + " + " + U32(component * 4u) + ") >> 2u;");
            state.Line("if (" + index + " < " + shared.length + ") " + result + Components[component] + " = " + ReadWord(shared, index) + ";");
        }
    }
    state.Outdent();
    state.Line("}");
}

void Store(MslEmitterState& state, const IrValue& instruction, std::uint32_t bits, std::uint32_t components) {
    const auto shared = SharedOf(state, instruction);
    // Per-invocation LDS (upstream 6337c8afb, pixel programs) has a copy per lane: every active lane writes its own, so
    // a wave-uniform store is not narrowed to the highest active lane there.
    const auto lane = shared.space == "thread" ? std::string() : MslUniformStoreLane(state, instruction, 0u, 0u);
    state.Line("if (" + Active(state, instruction) + (lane.empty() ? std::string() : " && " + lane) + ") {");
    state.Indent();
    const auto address = ByteAddress(state, instruction);
    if (bits < 32u && shared.space == "thread") {
        const auto index = state.Temporary("aps5_di");
        const auto shift = state.Temporary("aps5_ds");
        const auto mask = bits == 8u ? std::string("0xffu") : std::string("0xffffu");
        state.Line("const uint " + index + " = " + address + " >> 2u;");
        state.Line("const uint " + shift + " = (" + address + " & 3u) << 3u;");
        state.Line("if (" + index + " < " + shared.length + ") " + shared.words + "[" + index + "] = (" + shared.words + "[" + index + "] & ~(" + mask + " << " + shift + ")) | ((uint(" + state.Arg(instruction, instruction.ArgumentCount() - 2u) + ") & " + mask + ") << " + shift + ");");
    } else if (bits < 32u) {
        // Neighbouring lanes write the other bytes of the dword: one compare-exchange per field.
        const auto index = state.Temporary("aps5_di");
        const auto shift = state.Temporary("aps5_ds");
        const auto mask = bits == 8u ? std::string("0xffu") : std::string("0xffffu");
        state.Line("const uint " + index + " = " + address + " >> 2u;");
        state.Line("const uint " + shift + " = (" + address + " & 3u) << 3u;");
        state.Line("if (" + index + " < " + shared.length + ") {");
        state.Indent();
        const auto old = state.Temporary("aps5_dw");
        const auto field = "((uint(" + state.Arg(instruction, instruction.ArgumentCount() - 2u) + ") & " + mask + ") << " + shift + ")";
        state.Line("uint " + old + " = atomic_load_explicit(" + AtomicPointer(shared, index) + ", memory_order_relaxed);");
        state.Line("while (!atomic_compare_exchange_weak_explicit(" + AtomicPointer(shared, index) + ", &" + old + ", (" + old + " & ~(" + mask + " << " + shift + ")) | " + field + ", memory_order_relaxed, memory_order_relaxed)) {}");
        state.Outdent();
        state.Line("}");
    } else if (components == 1u) {
        const auto index = state.Temporary("aps5_di");
        state.Line("const uint " + index + " = " + address + " >> 2u;");
        state.Line("if (" + index + " < " + shared.length + ") " + WriteWord(shared, index, state.Arg(instruction, instruction.ArgumentCount() - 2u)));
    } else {
        for (std::uint32_t component = 0; component < components; ++component) {
            const auto index = state.Temporary("aps5_di");
            state.Line("const uint " + index + " = (" + address + " + " + U32(component * 4u) + ") >> 2u;");
            state.Line("if (" + index + " < " + shared.length + ") " + WriteWord(shared, index, state.Arg(instruction, component + 1u)));
        }
    }
    state.Outdent();
    state.Line("}");
}

// LockedLdsUpdate: `critical` runs under the LDS lock, once per invocation that reaches here.
void Locked(MslEmitterState& state, const IrValue& instruction, const Shared& shared, const std::function<void()>& critical) {
    const auto done = state.Temporary("aps5_done");
    state.Line("bool " + done + " = false;");
    MslLockLoop(state, instruction, {AtomicPointer(shared, shared.length), Fence(shared.flags, "memory_order_acquire", shared.scope), Fence(shared.flags, "memory_order_release", shared.scope)}, done, critical);
}

// One 32-bit LDS/GDS atomic: `native` when Metal has the operation, else a compare-exchange loop
// over `next(old)`; under the LDS lock a plain read-modify-write.
struct Atomic32Op {
    std::optional<MslAtomicOp> native;
    std::function<std::string(const std::string&)> next;
    std::string value;
    std::string comparator;
};

Atomic32Op Atomic32Of(MslEmitterState& state, const IrValue& instruction) {
    const bool flush = state.Memory(instruction).flushDenormals;
    const auto last = [&](std::size_t fromEnd) { return state.Arg(instruction, instruction.ArgumentCount() - fromEnd); };
    const auto op = [&](MslAtomicOp code) {
        state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
        const auto value = last(2u);
        return Atomic32Op{code, [code, value](const std::string& old) { return "aps5_atomic_next(" + std::to_string(static_cast<std::uint32_t>(code)) + "u, " + old + ", " + value + ", 0u)"; }, value, ""};
    };
    const auto custom = [&](std::function<std::string(const std::string&)> next) {
        state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
        state.RequirePrelude(MslPreludeChunk::LdsLock);
        return Atomic32Op{std::nullopt, std::move(next), "", ""};
    };
    const std::string f = flush ? "true" : "false";
    switch (instruction.Opcode()) {
    case IrOpcode::SharedAtomicSwap32: return op(MslAtomicOp::Swap);
    case IrOpcode::SharedAtomicIAdd32: return op(MslAtomicOp::Add);
    case IrOpcode::SharedAtomicISub32: return op(MslAtomicOp::Sub);
    case IrOpcode::SharedAtomicSMin32: return op(MslAtomicOp::SMin);
    case IrOpcode::SharedAtomicUMin32: return op(MslAtomicOp::UMin);
    case IrOpcode::SharedAtomicSMax32: return op(MslAtomicOp::SMax);
    case IrOpcode::SharedAtomicUMax32: return op(MslAtomicOp::UMax);
    case IrOpcode::SharedAtomicAnd32: return op(MslAtomicOp::And);
    case IrOpcode::SharedAtomicOr32: return op(MslAtomicOp::Or);
    case IrOpcode::SharedAtomicXor32: return op(MslAtomicOp::Xor);
    case IrOpcode::SharedAtomicInc32: return op(MslAtomicOp::Inc);
    case IrOpcode::SharedAtomicDec32: return op(MslAtomicOp::Dec);
    case IrOpcode::SharedAtomicCmpst32: {
        const auto compare = state.Arg(instruction, 1u);
        const auto source = state.Arg(instruction, 2u);
        auto result = op(MslAtomicOp::CmpSwap);
        result.value = source;
        result.comparator = compare;
        result.next = [compare, source](const std::string& old) { return "(" + old + " == " + compare + " ? " + source + " : " + old + ")"; };
        return result;
    }
    case IrOpcode::SharedAtomicFMin32:
    case IrOpcode::SharedAtomicFMax32: {
        const auto data = state.Arg(instruction, 1u);
        const auto maximum = instruction.Opcode() == IrOpcode::SharedAtomicFMax32 ? "true" : "false";
        return custom([=](const std::string& old) { return "aps5_f32_min_max(aps5_fbits_flush(" + old + ", " + f + "), aps5_fbits_flush(uint(" + data + "), " + f + "), " + maximum + ")"; });
    }
    case IrOpcode::SharedAtomicCmpstF32: {
        const auto compare = state.Arg(instruction, 1u);
        const auto source = state.Arg(instruction, 2u);
        return custom([=](const std::string& old) { return "(aps5_f32_equal(aps5_fbits_flush(" + old + ", " + f + "), aps5_fbits_flush(uint(" + compare + "), " + f + ")) ? " + source + " : " + old + ")"; });
    }
    case IrOpcode::SharedAtomicRsub32: {
        const auto data = state.Arg(instruction, 1u);
        return custom([=](const std::string& old) { return "(" + data + " - " + old + ")"; });
    }
    case IrOpcode::SharedAtomicFAdd32: {
        const auto data = state.Arg(instruction, 1u);
        const std::string r = state.Memory(instruction).flushDenormalResults ? "true" : "false";
        return custom([=](const std::string& old) { return "aps5_lds_fadd32(" + old + ", uint(" + data + "), " + f + ", " + r + ")"; });
    }
    case IrOpcode::SharedAtomicMskor32: {
        const auto mask = state.Arg(instruction, 1u);
        const auto source = state.Arg(instruction, 2u);
        return custom([=](const std::string& old) { return "((" + old + " & ~" + mask + ") | " + source + ")"; });
    }
    case IrOpcode::SharedAtomicWrap32: {
        const auto limit = state.Arg(instruction, 1u);
        const auto step = state.Arg(instruction, 2u);
        return custom([=](const std::string& old) { return "(" + old + " >= " + limit + " ? " + old + " - " + limit + " : " + old + " + " + step + ")"; });
    }
    default:
        state.Fail(instruction, "is not a 32-bit LDS/GDS atomic");
    }
}

void SharedAtomic32(MslEmitterState& state, const IrValue& instruction) {
    const auto result = state.Name(instruction);
    state.Line(result + " = 0u;");
    const auto atomic = Atomic32Of(state, instruction);
    state.Line("if (" + Active(state, instruction) + ") {");
    state.Indent();
    const auto shared = SharedOf(state, instruction);
    const auto address = ByteAddress(state, instruction);
    const auto index = state.Temporary("aps5_di");
    state.Line("const uint " + index + " = " + address + " >> 2u;");
    state.Line("if (" + index + " < " + shared.length + ") {");
    state.Indent();
    if (shared.space == "thread") {
        state.Line(result + " = " + shared.words + "[" + index + "];");
        state.Line(shared.words + "[" + index + "] = " + atomic.next(result) + ";");
    } else if (shared.locked) {
        Locked(state, instruction, shared, [&] {
            state.Line(result + " = " + shared.words + "[" + index + "];");
            state.Line(shared.words + "[" + index + "] = " + atomic.next(result) + ";");
        });
    } else {
        const auto pointer = AtomicPointer(shared, index);
        if (atomic.native) {
            state.Line(result + " = " + MslAtomic32Expression(*atomic.native, pointer, atomic.value, atomic.comparator) + ";");
        } else {
            state.Line(result + " = atomic_load_explicit(" + pointer + ", memory_order_relaxed);");
            state.Line("while (!atomic_compare_exchange_weak_explicit(" + pointer + ", &" + result + ", " + atomic.next(result) + ", memory_order_relaxed, memory_order_relaxed)) {}");
        }
        state.Line(Fence(shared.flags, "memory_order_acq_rel", shared.scope));
    }
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
}

// SharedAtomic64: a 64-bit read-modify-write of two LDS dwords, under the LDS lock.
void SharedAtomic64(MslEmitterState& state, const IrValue& instruction) {
    const auto& memory = state.Memory(instruction);
    const std::string f = memory.flushDenormals ? "true" : "false";
    const auto last = [&](std::size_t fromEnd) { return "ulong(" + state.Arg(instruction, instruction.ArgumentCount() - fromEnd) + ")"; };
    std::function<std::string(const std::string&)> next;
    const auto data = last(2u);
    const auto select = [&](std::string_view compare, bool sign) {
        const std::string wrap = sign ? "as_type<long>" : "";
        next = [=](const std::string& old) { return "(" + wrap + "(" + data + ") " + std::string(compare) + " " + wrap + "(" + old + ") ? " + data + " : " + old + ")"; };
    };
    switch (instruction.Opcode()) {
    case IrOpcode::SharedAtomicSwap64: next = [=](const std::string&) { return data; }; break;
    case IrOpcode::SharedAtomicIAdd64: next = [=](const std::string& old) { return "(" + old + " + " + data + ")"; }; break;
    case IrOpcode::SharedAtomicISub64: next = [=](const std::string& old) { return "(" + old + " - " + data + ")"; }; break;
    case IrOpcode::SharedAtomicRsub64: next = [=](const std::string& old) { return "(" + data + " - " + old + ")"; }; break;
    case IrOpcode::SharedAtomicInc64: next = [=](const std::string& old) { return "(" + old + " >= " + data + " ? 0ul : " + old + " + 1ul)"; }; break;
    case IrOpcode::SharedAtomicDec64: next = [=](const std::string& old) { return "((" + old + " == 0ul || " + old + " > " + data + ") ? " + data + " : " + old + " - 1ul)"; }; break;
    case IrOpcode::SharedAtomicSMin64: select("<", true); break;
    case IrOpcode::SharedAtomicUMin64: select("<", false); break;
    case IrOpcode::SharedAtomicSMax64: select(">", true); break;
    case IrOpcode::SharedAtomicUMax64: select(">", false); break;
    case IrOpcode::SharedAtomicAnd64: next = [=](const std::string& old) { return "(" + old + " & " + data + ")"; }; break;
    case IrOpcode::SharedAtomicOr64: next = [=](const std::string& old) { return "(" + old + " | " + data + ")"; }; break;
    case IrOpcode::SharedAtomicXor64: next = [=](const std::string& old) { return "(" + old + " ^ " + data + ")"; }; break;
    case IrOpcode::SharedAtomicFMin64:
    case IrOpcode::SharedAtomicFMax64: {
        const auto maximum = instruction.Opcode() == IrOpcode::SharedAtomicFMax64 ? "true" : "false";
        next = [=](const std::string& old) { return "aps5_f64_bits_min_max(aps5_fbits_flush(" + old + ", " + f + "), aps5_fbits_flush(" + data + ", " + f + "), " + maximum + ")"; };
        break;
    }
    case IrOpcode::SharedAtomicCmpst64: {
        const auto compare = last(3u);
        next = [=](const std::string& old) { return "(" + old + " == " + compare + " ? " + data + " : " + old + ")"; };
        break;
    }
    case IrOpcode::SharedAtomicCmpstF64: {
        const auto compare = last(3u);
        next = [=](const std::string& old) { return "(aps5_f64_bits_equal(aps5_fbits_flush(" + old + ", " + f + "), aps5_fbits_flush(" + compare + ", " + f + ")) ? " + data + " : " + old + ")"; };
        break;
    }
    case IrOpcode::SharedAtomicMskor64: {
        const auto mask = last(3u);
        next = [=](const std::string& old) { return "((" + old + " & ~" + mask + ") | " + data + ")"; };
        break;
    }
    default:
        state.Fail(instruction, "is not a 64-bit LDS atomic");
    }
    state.RequirePrelude(MslPreludeChunk::LdsLock);
    const auto result = state.Name(instruction);
    const auto type = MslType(instruction.Type());
    state.Line(result + " = " + type + "(0);");
    state.Line("if (" + Active(state, instruction) + ") {");
    state.Indent();
    const auto shared = SharedOf(state, instruction);
    if (!shared.lds) state.Fail(instruction, "64-bit GDS atomics are not implemented in the MSL backend yet (work package B3)");
    const auto address = ByteAddress(state, instruction);
    const auto low = state.Temporary("aps5_di");
    state.Line("const uint " + low + " = " + address + " >> 2u;");
    state.Line("if (" + low + " + 1u < " + shared.length + ") {");
    state.Indent();
    const auto update = [&] {
        const auto old = state.Temporary("aps5_o64");
        const auto updated = state.Temporary("aps5_n64");
        state.Line("const ulong " + old + " = ulong(" + shared.words + "[" + low + "]) | (ulong(" + shared.words + "[" + low + " + 1u]) << 32u);");
        state.Line("const ulong " + updated + " = " + next(old) + ";");
        state.Line(shared.words + "[" + low + "] = uint(" + updated + ");");
        state.Line(shared.words + "[" + low + " + 1u] = uint(" + updated + " >> 32u);");
        state.Line(result + " = " + type + "(" + old + ");");
    };
    if (shared.locked) Locked(state, instruction, shared, update);
    else update();
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
}

// AppendConsume: one atomic per wave adds (subtracts) the number of active lanes; every lane gets
// the old counter value. Emitted in half 0; half 1 copies it. ds_ordered_count (`ordered`, upstream
// 46a1cc0fc/3a7e82a17) adds the first active lane's data operand to the GDS counter at offset0 alone
// (no M0 base or size).
void AppendConsume(MslEmitterState& state, const IrValue& instruction, bool append, bool ordered = false) {
    if (state.half == 1u) {
        state.Define(instruction, state.HalfName(instruction, 0u));
        return;
    }
    const auto& memory = state.Memory(instruction);
    const auto shared = SharedOf(state, instruction);
    const bool gds = memory.kind == ResourceKind::Gds;
    const bool m0Address = gds && !ordered;
    const auto index = state.Temporary("aps5_di");
    std::string bounds;
    if (m0Address) {
        const auto m0 = state.Arg(instruction, 0u);
        state.Line("const uint " + index + " = ((" + m0 + " >> 16u) + " + U32(memory.offset) + ") >> 2u;");
        bounds = "(" + index + " < " + shared.length + " && (" + m0 + " & 0xffffu) != 0u)";
    } else {
        state.Line("const uint " + index + " = " + U32(memory.offset >> 2u) + ";");
        bounds = "(" + index + " < " + shared.length + ")";
    }
    const auto ballot = state.Temporary("aps5_b");
    const auto count = state.Temporary("aps5_cnt");
    const auto first = state.Temporary("aps5_fl");
    const auto value = state.Temporary("aps5_ac");
    state.Line("const uint2 " + ballot + " = " + state.WaveBallot(instruction.Argument(1u)) + ";");
    state.Line("const uint " + count + " = popcount(" + ballot + ".x) + popcount(" + ballot + ".y);");
    state.Line("const uint " + first + " = (" + ballot + ".x != 0u ? ctz(" + ballot + ".x) : 32u + ctz(" + ballot + ".y)) & 31u;");
    const auto lanesActive = state.laneCount == 2u ? "(" + count + " != 0u)" : std::string(state.Arg(instruction, 1u));
    std::string amount = count;
    if (ordered) {
        amount = state.Temporary("aps5_amt");
        state.Line("const uint " + amount + " = simd_shuffle(" + state.Arg(instruction, 4u) + ", ushort(" + first + "));");
    }
    state.Line("uint " + value + " = 0u;");
    const auto condition = state.Builtin(MslBuiltin::ThreadIndexInSimdgroup) + " == " + first + " && " + lanesActive + " && " + bounds;
    if (shared.space == "thread") {
        // Per-invocation LDS (upstream 6337c8afb): the first active lane updates its own copy.
        state.Line("if (" + condition + ") { " + value + " = " + shared.words + "[" + index + "]; " + shared.words + "[" + index + "] = " + value + (append ? " + " : " - ") + count + "; }");
    } else {
        const auto pointer = AtomicPointer(shared, index);
        state.Line("if (" + condition + ") " + value + " = " + (append ? "atomic_fetch_add_explicit(" : "atomic_fetch_sub_explicit(") + pointer + ", " + amount + ", memory_order_relaxed);");
    }
    state.Define(instruction, "simd_shuffle(" + value + ", ushort(" + first + "))");
}

}

MslB5Storage MslLdsStorage(MslEmitterState& state) {
    if (state.workgroup == nullptr) state.Fail("LDS needs a workgroup stage (work package B3)");
    const auto dwords = state.workgroup->ldsSizeDwords;
    if (dwords == 0u) return {};
    if (state.prologue.find(" aps5_lds") != std::string::npos) return {"aps5_lds", dwords};
    if (state.LdsInDeviceMemory()) {
        const auto stride = WorkgroupMemoryStrideDwords(state.program.Info());
        if (state.program.Info().sharedMemoryBytes != dwords * 4u) state.Fail("device memory LDS does not match the compute program's LDS size");
        state.reflection.workgroupMemoryStrideDwords = stride;
        state.AddParameter("device uint* aps5_wgm [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::WorkgroupMemory)) + ")]]");
        const auto group = state.Builtin(MslBuiltin::ThreadgroupPositionInGrid);
        const auto groups = state.Builtin(MslBuiltin::ThreadgroupsPerGrid);
        state.AddPrologue("device uint* aps5_lds = aps5_wgm + ((" + group + ".z * " + groups + ".y + " + group + ".y) * " + groups + ".x + " + group + ".x) * " + std::to_string(stride) + "u;");
    } else {
        state.AddPrologue("threadgroup uint aps5_lds[" + std::to_string(dwords + (state.requirements.ldsLock ? 1u : 0u)) + "];");
    }
    if (state.requirements.ldsLock) {
        // The lock dword starts free; nobody takes it before every invocation passed the barrier.
        state.AddPrologue("aps5_lds[" + std::to_string(dwords) + "] = 0u;");
        state.AddPrologue("threadgroup_barrier(" + state.LdsMemoryFlags() + ");");
    }
    return {"aps5_lds", dwords};
}

void MslLockLoop(MslEmitterState& state, const IrValue& instruction, const MslLock& lock, std::string_view done, const std::function<void()>& critical) {
    const auto rounds = state.Temporary("aps5_round");
    const auto expected = state.Temporary("aps5_lk");
    const std::string flag(done);
    state.Line("for (uint " + rounds + " = 0u;; ++" + rounds + ") {");
    state.Indent();
    // The election is its own branch: simd_is_first() picks the first lane among those still waiting.
    state.Line("if (!" + flag + ") {");
    state.Indent();
    state.Line("if (simd_is_first()) {");
    state.Indent();
    state.Line("uint " + expected + " = 0u;");
    state.Line("if (atomic_compare_exchange_weak_explicit(" + lock.pointer + ", &" + expected + ", 1u, memory_order_relaxed, memory_order_relaxed)) {");
    state.Indent();
    state.Line(lock.acquire);
    critical();
    state.Line(lock.release);
    state.Line("atomic_store_explicit(" + lock.pointer + ", 0u, memory_order_relaxed);");
    state.Line(flag + " = true;");
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
    state.Line("if (!simd_any(!" + flag + ")) break;");
    state.Line("if (" + rounds + " == " + U32(LockRoundLimit) + ") {");
    state.Indent();
    state.Line("if (simd_is_first()) " + MslRecordFault(state, 6u, MslHex(state.program.Resources().shaderHash) + "ul", U32(LockRoundLimit), MslB5Pc(instruction)));
    state.Line("break;");
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
}

void MslLdsLocked(MslEmitterState& state, const IrValue& instruction, const std::function<void()>& critical) {
    if (!state.requirements.ldsLock) state.Fail(instruction, "an LDS update under the lock in a program without the LDS lock (work package B3)");
    const auto storage = MslLdsStorage(state);
    Shared shared;
    shared.words = storage.words;
    shared.length = U32(storage.dwords);
    shared.space = state.LdsInDeviceMemory() ? "device" : "threadgroup";
    shared.flags = state.LdsMemoryFlags();
    shared.scope = "thread_scope_threadgroup";
    shared.locked = true;
    Locked(state, instruction, shared, critical);
}

std::string MslLdsAtomicPointer(MslEmitterState& state, std::string_view index) {
    const auto storage = MslLdsStorage(state);
    return "(reinterpret_cast<" + std::string(state.LdsInDeviceMemory() ? "device" : "threadgroup") + " atomic_uint*>(" + storage.words + ") + (" + std::string(index) + "))";
}

std::string MslDeviceAtomic64(MslEmitterState& state, const IrValue& instruction, MslAtomicOp op, std::string_view address, std::string_view value, std::string_view comparator) {
    state.RequirePrelude(MslPreludeChunk::Atomics64);
    state.usesDriverData = true;
    const auto result = state.Temporary("aps5_r64");
    state.Line("ulong " + result + " = 0ul;");
    const Shared lockSpace{"", "", "device", "mem_flags::mem_device", "thread_scope_device", false, true};
    const auto done = state.Temporary("aps5_done");
    const auto lock = state.Temporary("aps5_lock");
    const auto words = state.Temporary("aps5_w64");
    state.Line("device atomic_uint* " + lock + " = aps5_atomic64_lock(aps5_driver.atomicLocks, " + std::string(address) + ");");
    state.Line("device uint* " + words + " = reinterpret_cast<device uint*>(" + std::string(address) + ");");
    state.Line("bool " + done + " = aps5_driver.atomicLocks == 0ul;");
    state.Line("if (" + done + ") " + MslRecordFault(state, 6u, std::string(address), "8u", MslB5Pc(instruction)));
    MslLockLoop(state, instruction, {lock, Fence(lockSpace.flags, "memory_order_acquire", lockSpace.scope), Fence(lockSpace.flags, "memory_order_release", lockSpace.scope)}, done, [&] {
        state.Line(result + " = ulong(atomic_load_explicit(reinterpret_cast<device atomic_uint*>(" + words + "), memory_order_relaxed)) | (ulong(atomic_load_explicit(reinterpret_cast<device atomic_uint*>(" + words + ") + 1, memory_order_relaxed)) << 32u);");
        const auto updated = state.Temporary("aps5_n64");
        state.Line("const ulong " + updated + " = aps5_atomic64_next(" + std::to_string(static_cast<std::uint32_t>(op)) + "u, " + result + ", ulong(" + std::string(value) + "), ulong(" + std::string(comparator.empty() ? "0ul" : comparator) + "));");
        // Commit the new value with one 64-bit atomic (max when it grows, min when it shrinks: the lock
        // holder knows the old value), so 64-bit loads elsewhere never see half of it.
        state.Line("if (" + updated + " > " + result + ") atomic_max_explicit(reinterpret_cast<device atomic_ulong*>(" + words + "), " + updated + ", memory_order_relaxed);");
        state.Line("else if (" + updated + " < " + result + ") atomic_min_explicit(reinterpret_cast<device atomic_ulong*>(" + words + "), " + updated + ", memory_order_relaxed);");
    });
    return result;
}

std::string MslUniformStoreLane(MslEmitterState& state, const IrValue& instruction, std::size_t first, std::size_t last) {
    if (state.program.WaveSize() > state.hostSubgroupSize && state.laneCount == 1u) return {};
    for (std::size_t index = first; index <= last && index < instruction.ArgumentCount(); ++index) {
        const IrValue* argument = instruction.Argument(index)->Resolve();
        if (!argument->HasImmediate() && !state.waveUniform.contains(argument)) return {};
    }
    const auto ballot = state.Temporary("aps5_sb");
    state.Line("const uint2 " + ballot + " = " + state.WaveBallot(instruction.Argument(instruction.ArgumentCount() - 1u)) + ";");
    const auto highest = state.Temporary("aps5_hl");
    state.Line("const uint " + highest + " = " + ballot + ".y != 0u ? 63u - clz(" + ballot + ".y) : 31u - clz(" + ballot + ".x);");
    return "(" + state.GuestLane() + " == " + highest + ")";
}

bool EmitMslMemoryInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::LoadSharedU8: Load(state, instruction, 8u, 1u); return true;
    case IrOpcode::LoadSharedU16: Load(state, instruction, 16u, 1u); return true;
    case IrOpcode::LoadSharedU32: Load(state, instruction, 32u, 1u); return true;
    case IrOpcode::LoadSharedU32x2: Load(state, instruction, 32u, 2u); return true;
    case IrOpcode::LoadSharedU32x3: Load(state, instruction, 32u, 3u); return true;
    case IrOpcode::LoadSharedU32x4: Load(state, instruction, 32u, 4u); return true;
    case IrOpcode::WriteSharedU8: Store(state, instruction, 8u, 1u); return true;
    case IrOpcode::WriteSharedU16: Store(state, instruction, 16u, 1u); return true;
    case IrOpcode::WriteSharedU32: Store(state, instruction, 32u, 1u); return true;
    case IrOpcode::WriteSharedU32x2: Store(state, instruction, 32u, 2u); return true;
    case IrOpcode::WriteSharedU32x3: Store(state, instruction, 32u, 3u); return true;
    case IrOpcode::WriteSharedU32x4: Store(state, instruction, 32u, 4u); return true;
    case IrOpcode::SharedAtomicFMin32:
    case IrOpcode::SharedAtomicFMax32:
    case IrOpcode::SharedAtomicSwap32:
    case IrOpcode::SharedAtomicIAdd32:
    case IrOpcode::SharedAtomicISub32:
    case IrOpcode::SharedAtomicInc32:
    case IrOpcode::SharedAtomicDec32:
    case IrOpcode::SharedAtomicSMin32:
    case IrOpcode::SharedAtomicUMin32:
    case IrOpcode::SharedAtomicSMax32:
    case IrOpcode::SharedAtomicUMax32:
    case IrOpcode::SharedAtomicAnd32:
    case IrOpcode::SharedAtomicOr32:
    case IrOpcode::SharedAtomicXor32:
    case IrOpcode::SharedAtomicRsub32:
    case IrOpcode::SharedAtomicFAdd32:
    case IrOpcode::SharedAtomicCmpst32:
    case IrOpcode::SharedAtomicCmpstF32:
    case IrOpcode::SharedAtomicMskor32:
    case IrOpcode::SharedAtomicWrap32:
        SharedAtomic32(state, instruction);
        return true;
    case IrOpcode::SharedAtomicSwap64:
    case IrOpcode::SharedAtomicIAdd64:
    case IrOpcode::SharedAtomicISub64:
    case IrOpcode::SharedAtomicRsub64:
    case IrOpcode::SharedAtomicInc64:
    case IrOpcode::SharedAtomicDec64:
    case IrOpcode::SharedAtomicSMin64:
    case IrOpcode::SharedAtomicUMin64:
    case IrOpcode::SharedAtomicSMax64:
    case IrOpcode::SharedAtomicUMax64:
    case IrOpcode::SharedAtomicAnd64:
    case IrOpcode::SharedAtomicOr64:
    case IrOpcode::SharedAtomicXor64:
    case IrOpcode::SharedAtomicFMin64:
    case IrOpcode::SharedAtomicFMax64:
    case IrOpcode::SharedAtomicCmpst64:
    case IrOpcode::SharedAtomicCmpstF64:
    case IrOpcode::SharedAtomicMskor64:
        SharedAtomic64(state, instruction);
        return true;
    case IrOpcode::DataAppend:
        AppendConsume(state, instruction, true);
        return true;
    case IrOpcode::DataConsume:
        AppendConsume(state, instruction, false);
        return true;
    case IrOpcode::DataOrderedCount:
        AppendConsume(state, instruction, true, true);
        return true;
    default:
        return false;
    }
}

MslPreludeEntry MslMemoryPrelude(MslPreludeChunk chunk) {
    switch (chunk) {
    case MslPreludeChunk::LdsLock:
        return {LdsLockText, {}};
    case MslPreludeChunk::Atomics64:
        return {Atomics64Text, OnLdsLock};
    default:
        throw std::runtime_error("MSL emission failed: prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not a memory chunk (work package B3)");
    }
}

}
