#ifndef CORE_SHADER_RECOMPILER_METALBACKEND_SRC_MSLBUFFERSHARED_HPP
#define CORE_SHADER_RECOMPILER_METALBACKEND_SRC_MSLBUFFERSHARED_HPP

// Work package B5 (buffers, typed formats, BDA, flat/global/scratch): declarations shared by
// MslRuntimeBuffer.cpp (MUBUF/MTBUF, SMEM buffer loads, ReadConst, buffer atomics), MslBda.cpp
// (the BdaAbi page table, fault and written-page records, flat apertures) and MslAddress.cpp
// (flat/global/scalar-address/scratch accesses). Not part of the MetalBackend interface: other
// families reach these accesses through the IR opcodes only.

#include "MetalBackend/MslEmitterState.hpp"
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace ShaderRecompiler {

// A suffix that names an emitter's block-scoped temporaries uniquely per instruction and lane half
// (a wave64 program on 32-wide SIMD groups offers each instruction once per half).
[[nodiscard]] std::string MslB5Tag(const MslEmitterState& state, const IrValue& instruction);

// The instruction's guest program counter as an MSL literal (fault records name it).
[[nodiscard]] std::string MslB5Pc(const IrValue& instruction);

// "true"/"false" literal for an MSL bool parameter.
[[nodiscard]] inline std::string MslBool(bool value) {
    return value ? "true" : "false";
}

// The fault buffer's GPU address expression (DriverData.faultBuffer, 0 when the dispatch has none).
[[nodiscard]] std::string MslFaultBuffer(MslEmitterState& state);
// A statement that records a BdaAbi fault (the core FaultRecord chunk) of `reason` at `address`.
[[nodiscard]] std::string MslRecordFault(MslEmitterState& state, std::uint32_t reason, std::string_view address, std::string_view bytes, std::string_view pc);
// The Aps5Bda context of the program (Bda chunk): the BdaPagetable descriptor-table entry, the
// fault buffer and the stage, declared once in the prologue as `aps5_bda`. Requires the program to
// use BDA (the binding allocator then allocates the page table).
[[nodiscard]] std::string MslBdaContext(MslEmitterState& state);
// Whether a faulting BDA access ends its invocation, as SpirvEmitter decides it: programs with
// workgroup barriers or wave-LDS barriers keep every invocation running (the faulting access reads
// zero instead); mesh invocations never stop.
[[nodiscard]] bool MslBdaStops(const MslEmitterState& state);
// The statement that ends a stopping invocation: `return;`, or discard_fragment() and return in a
// pixel shader (OpKill on the SPIR-V path).
[[nodiscard]] std::string MslBdaStopStatement(const MslEmitterState& state);
// `if (<flag>) <MslBdaStopStatement>` when the invocation may stop (MslBdaStops), nothing otherwise.
void MslB5StopIf(MslEmitterState& state, std::string_view flag);

// The dwords of a 1-4 dword load that the program extracts (all of them when the composite is used
// whole), as UsedBdaDwords on the SPIR-V path.
[[nodiscard]] std::uint32_t MslUsedDwords(const IrValue& instruction, std::uint32_t dwords);

// Per-lane private and workgroup storage reached through flat apertures and scratch accesses:
// `words` names the array (empty when the program has none of it), `dwords` its length.
struct MslB5Storage {
    std::string words;
    std::uint32_t dwords = 0;
};
// The workgroup's LDS as `threadgroup uint aps5_lds[N]` (function scope, declared once). Package B3
// owns LDS: its declaration must keep this name so flat accesses through the shared aperture and
// ds_* instructions see the same memory.
[[nodiscard]] MslB5Storage MslB5Lds(MslEmitterState& state);
// The invocation's scratch (private segment) of the half being emitted, zero-initialised.
[[nodiscard]] MslB5Storage MslB5Scratch(MslEmitterState& state);

// Package B3 (MslMemory.cpp): the workgroup's LDS, declared once: `threadgroup uint aps5_lds[N]`
// (plus the lock dword when SpirvRequirements::ldsLock), or the workgroup's slice of the device
// arena (`device uint* aps5_lds`) when it exceeds threadgroup memory. MslB5Lds forwards here.
[[nodiscard]] MslB5Storage MslLdsStorage(MslEmitterState& state);
// Package B3: runs `critical` (a plain read-modify-write of LDS) under the LDS lock, once per
// invocation that reaches this point (SpirvRequirements::ldsLock programs: LockedLdsUpdate).
void MslLdsLocked(MslEmitterState& state, const IrValue& instruction, const std::function<void()>& critical);
// An `atomic_uint*` expression for LDS dword `index` in the LDS's address space.
[[nodiscard]] std::string MslLdsAtomicPointer(MslEmitterState& state, std::string_view index);

// 32-bit atomic operations shared by buffer (MUBUF) and flat/global atomics; the values are the
// `op` codes of aps5_atomic_next in the RuntimeBuffer chunk.
enum class MslAtomicOp : std::uint32_t {
    Swap,
    CmpSwap,
    Add,
    Sub,
    SMin,
    UMin,
    SMax,
    UMax,
    And,
    Or,
    Xor,
    Inc,
    Dec,
    USubSat,
    FMin,
    FMax,
    FCmpSwap
};
// The MSL expression that applies `op` to `*object` (a `device` or `threadgroup` `atomic_uint*`
// expression) and yields the old value: native atomics where Metal has them, else the CAS loop of
// the RuntimeBuffer chunk. `comparator` is used by the compare-exchange forms only.
[[nodiscard]] std::string MslAtomic32Expression(MslAtomicOp op, std::string_view object, std::string_view value, std::string_view comparator);
// Package B3: a 64-bit atomic read-modify-write of the 8-byte aligned device address `address` (a
// ulong expression already checked and translated by the caller), emitted at the current position
// where every invocation of the SIMD group that reaches the caller's `if` runs it. Apple GPUs have no
// 64-bit device atomics beyond min/max, so the update runs under a lock word of the driver's lock
// table (DriverData.atomicLocks). `op` codes as MslAtomicOp (comparator: compare-exchange forms).
// Returns the name of a ulong local holding the old value.
[[nodiscard]] std::string MslDeviceAtomic64(MslEmitterState& state, const IrValue& instruction, MslAtomicOp op, std::string_view address, std::string_view value, std::string_view comparator);
// Stores of several lanes to one address keep the highest lane's data on RDNA; Apple GPUs pick an
// unspecified lane. When arguments [first, last] (the address operands) of `instruction` are
// wave-uniform, every active lane stores to the same place: this emits, at the current position
// (outside any per-lane `if`), the ballot of the instruction's EXEC argument and returns the
// condition "this lane is the highest active one" to AND into the store's guard; empty otherwise.
[[nodiscard]] std::string MslUniformStoreLane(MslEmitterState& state, const IrValue& instruction, std::size_t first, std::size_t last);
// The device-scope fence that follows every buffer and flat/global atomic
// (EmitDeviceAtomicMemoryBarrier; APS5_NO_ATOMIC_BARRIER=1 drops it). Empty when disabled.
[[nodiscard]] std::string MslAtomicFence();

// Flat, global, scalar-address and scratch accesses (MslAddress.cpp): Load/StoreAddress*,
// AddressAtomic*.
bool EmitMslAddressInstruction(MslEmitterState& state, const IrValue& instruction);

// Prelude chunk text of MslBda.cpp (the Bda chunk).
[[nodiscard]] std::string_view MslBdaPreludeText();

}

#endif
