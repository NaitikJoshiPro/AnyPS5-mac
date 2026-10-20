// Work package B5: flat, global, scalar-address and scratch accesses (the address parts of
// SpirvMemoryInstructions.cpp: LoadAddress*, StoreAddress*, AddressAtomic*). Guest addresses go
// through the BdaAbi page table (Bda chunk, MslBda.cpp); flat accesses of compute programs route the
// shared aperture (high dword 0x80000000) to LDS and the private aperture (0x70000000) to scratch.

#include "MslBufferShared.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

namespace {

// SharedApertureHigh / PrivateApertureHigh: RdnaInstructionDecoder.hpp, shared with the aperture source reads.

bool RoutesApertures(const MslEmitterState& state, const MemoryInfo& memory) {
    return memory.kind == ResourceKind::Flat && state.ThreadgroupStage();
}

std::string Active(MslEmitterState& state, const IrValue& instruction) {
    return state.Arg(instruction, instruction.ArgumentCount() - 1u);
}

std::string Components(std::uint32_t count) {
    static constexpr const char* Swizzles[] = {"", ".x", ".xy", ".xyz", ""};
    return Swizzles[count];
}

// The parts of an address access every path needs.
struct B5Address {
    std::string bda;
    std::string pc;
    std::string stops;
    std::string coherent;
};

B5Address Prepare(MslEmitterState& state, const IrValue& instruction, const MemoryInfo& memory) {
    return {MslBdaContext(state), MslB5Pc(instruction), MslBool(MslBdaStops(state)), MslBool(memory.coherent)};
}

// GuestAddressBase: the full 64-bit address of the instruction's registers, or the address
// resource's base plus the unsigned 32-bit register offset (an Overflow fault when it wraps).
std::string GuestAddressBase(MslEmitterState& state, const IrValue& instruction, const MemoryInfo& memory, const B5Address& access) {
    auto low = state.Arg(instruction, 1u);
    if (memory.kind == ResourceKind::ScalarAddress) low = "(" + low + " & ~3u)";
    const auto name = state.Temporary("aps5_ga");
    if (memory.addressIsFull) {
        state.Line("const ulong " + name + " = ulong(" + low + ") | (ulong(" + state.Arg(instruction, 2u) + ") << 32u);");
        return name;
    }
    const IrValue* argument = instruction.Argument(0);
    const IrValue* handle = argument != nullptr ? argument->Resolve() : nullptr;
    if (handle == nullptr || handle->Opcode() != IrOpcode::GetAddressResource || handle->ArgumentCount() != 2u) state.Fail(instruction, "has no address base pair");
    const auto overflow = state.Temporary("aps5_ov");
    state.Line("bool " + overflow + " = false;");
    state.Line("const ulong " + name + " = aps5_bda_add(" + access.bda + ", ulong(" + state.Arg(*handle, 0u) + ") | (ulong(" + state.Arg(*handle, 1u) + ") << 32u), ulong(" + low + "), false, " + access.pc + ", " + overflow + ");");
    MslB5StopIf(state, overflow);
    return name;
}

// AddBdaImmediate: base plus a signed immediate (an Overflow fault when it wraps), in a new local.
std::string AddImmediate(MslEmitterState& state, const std::string& base, std::int32_t immediate, const B5Address& access) {
    if (immediate == 0) return base;
    const auto magnitude = immediate < 0 ? -static_cast<std::int64_t>(immediate) : static_cast<std::int64_t>(immediate);
    const auto overflow = state.Temporary("aps5_ov");
    const auto name = state.Temporary("aps5_ga");
    state.Line("bool " + overflow + " = false;");
    state.Line("const ulong " + name + " = aps5_bda_add(" + access.bda + ", " + base + ", " + MslHex(static_cast<std::uint64_t>(magnitude)) + "ul, " + MslBool(immediate < 0) + ", " + access.pc + ", " + overflow + ");");
    MslB5StopIf(state, overflow);
    return name;
}

std::int32_t Immediate(const MemoryInfo& memory) {
    auto immediate = static_cast<std::int32_t>(memory.offset);
    if (memory.kind == ResourceKind::ScalarAddress) immediate = static_cast<std::int32_t>(static_cast<std::uint32_t>(immediate) & ~3u);
    return immediate;
}

// Upstream d422c760c/3265cb6d7 (SpirvBdaRead.cpp CollectBdaAnchors, EmitAnchoredBdaRead): reads
// whose 64-bit base is a pair of user-data registers search the BDA table at every read although
// the base is one value for the whole dispatch or draw. For the (at most four) bases with two or
// more read sites, the range holding the base is searched once at entry (by the SIMD group's first
// lane in compute and pixel programs), and each read loads directly when that range holds all its
// bytes (dwords: at a 4-aligned device address); otherwise it takes the lookup path unchanged, so
// data, faults, stops and wraparound are the same.
constexpr std::size_t MaxBdaAnchors = 4u;

std::optional<std::pair<std::uint32_t, std::uint32_t>> UserDataBase(const MslEmitterState& state, const IrValue& instruction) {
    if (AddressOpcodeInfoOf(instruction.Opcode()).access != AddressAccess::Read || instruction.ArgumentCount() < 2u) return std::nullopt;
    const auto index = instruction.Flags<MemoryFlags>().index;
    const auto& memory = state.program.Resources().memoryInfo;
    if (index >= memory.size()) return std::nullopt;
    const auto& info = memory[index];
    if (info.kind != ResourceKind::ScalarAddress && info.kind != ResourceKind::Flat && info.kind != ResourceKind::Global) return std::nullopt;
    if (info.planningOnly || info.addressIsFull || RoutesApertures(state, info)) return std::nullopt;
    const IrValue* handle = instruction.Argument(0) != nullptr ? instruction.Argument(0)->Resolve() : nullptr;
    if (handle == nullptr || handle->Opcode() != IrOpcode::GetAddressResource || handle->ArgumentCount() != 2u) return std::nullopt;
    const auto userData = [](const IrValue* value) -> std::optional<std::uint32_t> {
        value = value != nullptr ? value->Resolve() : nullptr;
        if (value == nullptr || value->Opcode() != IrOpcode::GetUserData || value->ArgumentCount() == 0u || value->Argument(0) == nullptr) return std::nullopt;
        return value->Argument(0)->Register().index;
    };
    const auto low = userData(handle->Argument(0));
    const auto high = userData(handle->Argument(1));
    if (!low || !high) return std::nullopt;
    return std::pair {*low, *high};
}

void CollectBdaAnchors(MslEmitterState& state) {
    state.bdaAnchors.emplace();
    if (!state.program.Info().usesDma) return;
    std::vector<std::pair<std::pair<std::uint32_t, std::uint32_t>, std::size_t>> bases;
    for (const IrBlock* block : state.program.BlockOrder()) {
        for (const IrValue* instruction : block->Instructions()) {
            const auto base = UserDataBase(state, *instruction);
            if (!base) continue;
            const auto found = std::find_if(bases.begin(), bases.end(), [&](const auto& entry) { return entry.first == *base; });
            if (found != bases.end()) ++found->second;
            else bases.push_back({*base, 1u});
        }
    }
    std::stable_sort(bases.begin(), bases.end(), [](const auto& left, const auto& right) { return left.second > right.second; });
    for (const auto& [base, reads] : bases) {
        if (reads < 2u || state.bdaAnchors->size() == MaxBdaAnchors) break;
        state.bdaAnchors->push_back({base.first, base.second, {}});
    }
}

std::string UserDataWord(MslEmitterState& state, std::uint32_t registerIndex) {
    const auto& registers = state.program.Metadata().bindings.userDataRegisters;
    const auto found = std::lower_bound(registers.begin(), registers.end(), registerIndex);
    if (found == registers.end() || *found != registerIndex) return "0u";
    return state.ShaderDataDword(static_cast<std::uint32_t>(found - registers.begin()));
}

// The anchor of the instruction's base (its prologue search emitted on first use), or empty.
std::string FindBdaAnchor(MslEmitterState& state, const IrValue& instruction, const std::string& bda) {
    if (!state.bdaAnchors) CollectBdaAnchors(state);
    if (state.bdaAnchors->empty()) return {};
    const auto base = UserDataBase(state, instruction);
    if (!base) return {};
    for (std::size_t index = 0; index < state.bdaAnchors->size(); ++index) {
        auto& anchor = (*state.bdaAnchors)[index];
        if (anchor.lowRegister != base->first || anchor.highRegister != base->second) continue;
        if (anchor.name.empty()) {
            anchor.name = "aps5_anc" + std::to_string(index);
            const auto address = "(ulong(" + UserDataWord(state, anchor.lowRegister) + ") | (ulong(" + UserDataWord(state, anchor.highRegister) + ") << 32u))";
            const auto stage = state.program.Resources().stage;
            if (state.ThreadgroupStage() || stage == IrShaderStage::Pixel) {
                state.AddPrologue("ulong3 " + anchor.name + " = ulong3(0ul);");
                state.AddPrologue("if (simd_is_first()) " + anchor.name + " = aps5_bda_anchor(" + bda + ", " + address + ");");
                state.AddPrologue(anchor.name + " = ulong3(as_type<ulong>(simd_broadcast_first(as_type<uint2>(" + anchor.name + ".x))), as_type<ulong>(simd_broadcast_first(as_type<uint2>(" + anchor.name + ".y))), as_type<ulong>(simd_broadcast_first(as_type<uint2>(" + anchor.name + ".z))));");
            } else {
                state.AddPrologue("const ulong3 " + anchor.name + " = aps5_bda_anchor(" + bda + ", " + address + ");");
            }
        }
        return anchor.name;
    }
    return {};
}

// BdaAnchorHolds: whether the anchor's range holds [low + immediate, + bytes) (its device address in
// `translated`, a new local).
std::string AnchorHolds(MslEmitterState& state, const std::string& anchor, const std::string& low, std::int32_t immediate, std::uint32_t bytes, std::string& translated) {
    const auto offset = state.Temporary("aps5_ao");
    state.Line("const ulong " + offset + " = " + anchor + ".z + ulong(" + low + ")" + (immediate != 0 ? " + " + MslHex(static_cast<std::uint64_t>(static_cast<std::int64_t>(immediate))) + "ul" : std::string()) + ";");
    translated = state.Temporary("aps5_ad");
    state.Line("const ulong " + translated + " = " + anchor + ".y + " + offset + ";");
    return "(" + anchor + ".x >= " + std::to_string(bytes) + "ul && " + offset + " <= " + anchor + ".x - " + std::to_string(bytes) + "ul)";
}

std::string AnchorLow(MslEmitterState& state, const IrValue& instruction, const MemoryInfo& memory) {
    auto low = state.Arg(instruction, 1u);
    if (memory.kind == ResourceKind::ScalarAddress) low = "(" + low + " & ~3u)";
    return low;
}

// GuestAddress: GuestAddressBase plus the instruction's immediate offset.
std::string GuestAddress(MslEmitterState& state, const IrValue& instruction, const MemoryInfo& memory, const B5Address& access) {
    return AddImmediate(state, GuestAddressBase(state, instruction, memory, access), Immediate(memory), access);
}

// Aperture routing of a flat address (RouteFlatAccess): `shared`, `scratch` and `global` emit the
// three arms; an aperture without storage in the program is skipped (reads stay zero).
template <typename TShared, typename TPrivate, typename TGlobal>
void RouteFlat(MslEmitterState& state, const std::string& address, TShared&& shared, TPrivate&& priv, TGlobal&& global) {
    const auto high = state.Temporary("aps5_hi");
    state.Line("const uint " + high + " = uint(" + address + " >> 32u);");
    state.Line("if (" + high + " == " + MslHex(SharedApertureHigh) + "u) {");
    state.Indent();
    shared();
    state.Outdent();
    state.Line("} else if (" + high + " == " + MslHex(PrivateApertureHigh) + "u) {");
    state.Indent();
    priv();
    state.Outdent();
    state.Line("} else {");
    state.Indent();
    global();
    state.Outdent();
    state.Line("}");
}

// ScratchByteAddress: the 64-bit register pair plus the signed immediate; anything past 4 GiB is out
// of range (0xffffffff).
std::string ScratchByte(MslEmitterState& state, const IrValue& instruction, const MemoryInfo& memory) {
    const auto sum = state.Temporary("aps5_sa");
    const auto byte = state.Temporary("aps5_sb");
    const auto immediate = static_cast<std::int64_t>(static_cast<std::int32_t>(memory.offset));
    state.Line("const ulong " + sum + " = ((ulong(" + state.Arg(instruction, 2u) + ") << 32u) | ulong(" + state.Arg(instruction, 1u) + ")) + " + MslHex(static_cast<std::uint64_t>(immediate)) + "ul;");
    state.Line("const uint " + byte + " = (" + sum + " >> 32u) == 0ul ? uint(" + sum + ") : 0xffffffffu;");
    return byte;
}

// Scratch-segment loads (LoadWord/LoadSubword on the per-invocation scratch array).
void LoadScratch(MslEmitterState& state, const IrValue& instruction, const MemoryInfo& memory, std::uint32_t bits) {
    const auto result = state.Name(instruction);
    const auto type = MslType(instruction.Type());
    state.Line("if (" + Active(state, instruction) + ") {");
    state.Indent();
    const auto scratch = MslB5Scratch(state);
    if (scratch.words.empty()) state.Fail(instruction, "scratch operation has no per-thread storage");
    const auto byte = ScratchByte(state, instruction, memory);
    const auto word = scratch.words + "[" + byte + " >> 2u]";
    const auto value = bits == 32u ? word : "((" + word + " >> ((" + byte + " & 3u) << 3u)) & " + (bits == 8u ? "0xffu" : "0xffffu") + ")";
    state.Line("if ((" + byte + " >> 2u) < " + std::to_string(scratch.dwords) + "u) " + result + " = " + type + "(" + value + ");");
    state.Outdent();
    state.Line("}");
}

void StoreScratch(MslEmitterState& state, const IrValue& instruction, const MemoryInfo& memory, std::uint32_t bits) {
    const auto data = "uint(" + state.Arg(instruction, instruction.ArgumentCount() - 2u) + ")";
    state.Line("if (" + Active(state, instruction) + ") {");
    state.Indent();
    const auto scratch = MslB5Scratch(state);
    if (scratch.words.empty()) state.Fail(instruction, "scratch operation has no per-thread storage");
    const auto byte = ScratchByte(state, instruction, memory);
    const auto word = scratch.words + "[" + byte + " >> 2u]";
    state.Line("if ((" + byte + " >> 2u) < " + std::to_string(scratch.dwords) + "u) {");
    state.Indent();
    if (bits == 32u) {
        state.Line(word + " = " + data + ";");
    } else {
        const auto shift = state.Temporary("aps5_sh");
        const auto mask = bits == 8u ? "0xffu" : "0xffffu";
        state.Line("const uint " + shift + " = (" + byte + " & 3u) << 3u;");
        state.Line(word + " = (" + word + " & ~(" + std::string(mask) + " << " + shift + ")) | ((" + data + " & " + mask + ") << " + shift + ");");
    }
    state.Outdent();
    state.Line("}");
    state.Outdent();
    state.Line("}");
}

// One element of an aperture as an MSL expression (zero when the program has no such storage).
std::string ApertureLoad(const MslB5Storage& storage, const std::string& byteOffset, std::uint32_t bits) {
    if (storage.words.empty()) return "0u";
    return "aps5_aperture_load(" + storage.words + ", " + std::to_string(storage.dwords) + "u, " + byteOffset + ", " + std::to_string(bits) + "u)";
}

void ApertureStore(MslEmitterState& state, const MslB5Storage& storage, const std::string& byteOffset, std::uint32_t bits, const std::string& data) {
    if (storage.words.empty()) return;
    state.Line("aps5_aperture_store(" + storage.words + ", " + std::to_string(storage.dwords) + "u, " + byteOffset + ", " + std::to_string(bits) + "u, " + data + ");");
}

void LoadAddress(MslEmitterState& state, const IrValue& instruction, std::uint32_t bits) {
    const auto& memory = state.Memory(instruction);
    if (bits == 32u && memory.planningOnly) return;
    const auto result = state.Name(instruction);
    const auto type = MslType(instruction.Type());
    state.Line(result + " = " + type + "(0);");
    if (memory.kind == ResourceKind::Scratch) {
        LoadScratch(state, instruction, memory, bits);
        return;
    }
    if (memory.kind != ResourceKind::ScalarAddress && memory.kind != ResourceKind::Flat && memory.kind != ResourceKind::Global) state.Fail(instruction, "must read a scratch or physical address resource");
    state.Line("if (" + Active(state, instruction) + ") {");
    state.Indent();
    const auto access = Prepare(state, instruction, memory);
    // A sub-dword coherent read keeps the lookup path (its byte loads are not atomic).
    if (const auto anchor = memory.coherent && bits != 32u ? std::string() : FindBdaAnchor(state, instruction, access.bda); !anchor.empty()) {
        std::string translated;
        auto direct = AnchorHolds(state, anchor, AnchorLow(state, instruction, memory), Immediate(memory), bits / 8u, translated);
        if (bits == 32u) direct = "(" + direct + " && (" + translated + " & 3ul) == 0ul)";
        state.Line("if (" + direct + ") {");
        state.Indent();
        if (bits == 32u) {
            state.Line(result + " = " + (memory.coherent ? "atomic_load_explicit(reinterpret_cast<device atomic_uint*>(" + translated + "), memory_order_relaxed)" : "*reinterpret_cast<device const uint*>(" + translated + ")") + ";");
        } else {
            std::string value = "0u";
            for (std::uint32_t byte = 0; byte < bits / 8u; ++byte) value += " | (uint(reinterpret_cast<device const uchar*>(" + translated + ")[" + std::to_string(byte) + "]) << " + std::to_string(byte * 8u) + "u)";
            state.Line(result + " = " + type + "(" + value + ");");
        }
        state.Outdent();
        state.Line("} else {");
        state.Indent();
        const auto address = GuestAddress(state, instruction, memory, access);
        const auto stop = state.Temporary("aps5_st");
        state.Line("bool " + stop + " = false;");
        if (bits == 32u) state.Line(result + " = aps5_bda_read_span(" + access.bda + ", " + address + ", 0u, 1u, " + access.pc + ", " + access.coherent + ", " + access.stops + ", " + stop + ").x;");
        else state.Line(result + " = " + type + "(aps5_bda_read_small(" + access.bda + ", " + address + ", " + std::to_string(bits) + "u, " + access.pc + ", " + access.stops + ", " + stop + "));");
        MslB5StopIf(state, stop);
        state.Outdent();
        state.Line("}");
        state.Outdent();
        state.Line("}");
        return;
    }
    const auto address = GuestAddress(state, instruction, memory, access);
    const auto global = [&] {
        const auto stop = state.Temporary("aps5_st");
        state.Line("bool " + stop + " = false;");
        if (bits == 32u) state.Line(result + " = aps5_bda_read_span(" + access.bda + ", " + address + ", 0u, 1u, " + access.pc + ", " + access.coherent + ", " + access.stops + ", " + stop + ").x;");
        else state.Line(result + " = " + type + "(aps5_bda_read_small(" + access.bda + ", " + address + ", " + std::to_string(bits) + "u, " + access.pc + ", " + access.stops + ", " + stop + "));");
        MslB5StopIf(state, stop);
    };
    if (!RoutesApertures(state, memory)) {
        global();
    } else {
        RouteFlat(state, address, [&] {
            state.Line(result + " = " + type + "(" + ApertureLoad(MslB5Lds(state), "uint(" + address + ")", bits) + ");");
        }, [&] {
            state.Line(result + " = " + type + "(" + ApertureLoad(MslB5Scratch(state), "uint(" + address + ")", bits) + ");");
        }, global);
    }
    state.Outdent();
    state.Line("}");
}

void LoadAddressWide(MslEmitterState& state, const IrValue& instruction, std::uint32_t components) {
    const auto& memory = state.Memory(instruction);
    if (memory.planningOnly) return;
    if (memory.kind != ResourceKind::Flat && memory.kind != ResourceKind::Global) state.Fail(instruction, "must read a physical address resource");
    const auto result = state.Name(instruction);
    const auto type = MslType(instruction.Type());
    state.Line(result + " = " + type + "(0);");
    state.Line("if (" + Active(state, instruction) + ") {");
    state.Indent();
    const auto access = Prepare(state, instruction, memory);
    const auto used = MslUsedDwords(instruction, components);
    if (const auto anchor = used == 0u ? std::string() : FindBdaAnchor(state, instruction, access.bda); !anchor.empty()) {
        const auto first = static_cast<std::uint32_t>(std::countr_zero(used));
        const auto last = 31u - static_cast<std::uint32_t>(std::countl_zero(used));
        std::string translated;
        const auto holds = AnchorHolds(state, anchor, AnchorLow(state, instruction, memory), static_cast<std::int32_t>(memory.offset + first * 4u), (last - first + 1u) * 4u, translated);
        state.Line("if (" + holds + " && (" + translated + " & 3ul) == 0ul) {");
        state.Indent();
        for (std::uint32_t dword = first; dword <= last; ++dword) {
            if ((used & (1u << dword)) == 0u) continue;
            const auto element = "(" + translated + " + " + std::to_string((dword - first) * 4u) + "ul)";
            state.Line(result + "[" + std::to_string(dword) + "] = " + (memory.coherent ? "atomic_load_explicit(reinterpret_cast<device atomic_uint*>" + element + ", memory_order_relaxed)" : "*reinterpret_cast<device const uint*>" + element) + ";");
        }
        state.Outdent();
        state.Line("} else {");
        state.Indent();
        const auto base = GuestAddressBase(state, instruction, memory, access);
        const auto stop = state.Temporary("aps5_st");
        state.Line("bool " + stop + " = false;");
        state.Line(result + " = aps5_bda_read_span(" + access.bda + ", " + base + ", " + MslHex(memory.offset) + "u, " + MslHex(used) + "u, " + access.pc + ", " + access.coherent + ", " + access.stops + ", " + stop + ")" + Components(components) + ";");
        MslB5StopIf(state, stop);
        state.Outdent();
        state.Line("}");
        state.Outdent();
        state.Line("}");
        return;
    }
    const auto base = GuestAddressBase(state, instruction, memory, access);
    const auto global = [&] {
        const auto stop = state.Temporary("aps5_st");
        state.Line("bool " + stop + " = false;");
        state.Line(result + " = aps5_bda_read_span(" + access.bda + ", " + base + ", " + MslHex(memory.offset) + "u, " + MslHex(used) + "u, " + access.pc + ", " + access.coherent + ", " + access.stops + ", " + stop + ")" + Components(components) + ";");
        MslB5StopIf(state, stop);
    };
    if (!RoutesApertures(state, memory)) {
        global();
    } else {
        const auto address = AddImmediate(state, base, static_cast<std::int32_t>(memory.offset), access);
        const auto aperture = [&](const MslB5Storage& storage) {
            for (std::uint32_t component = 0; component < components; ++component) {
                state.Line(result + "[" + std::to_string(component) + "] = " + ApertureLoad(storage, "(uint(" + address + ") + " + std::to_string(component * 4u) + "u)", 32u) + ";");
            }
        };
        RouteFlat(state, address, [&] { aperture(MslB5Lds(state)); }, [&] { aperture(MslB5Scratch(state)); }, global);
    }
    state.Outdent();
    state.Line("}");
}

void StoreAddress(MslEmitterState& state, const IrValue& instruction, std::uint32_t bits) {
    const auto& memory = state.Memory(instruction);
    if (memory.kind == ResourceKind::Scratch) {
        StoreScratch(state, instruction, memory, bits);
        return;
    }
    if (memory.kind != ResourceKind::Flat && memory.kind != ResourceKind::Global) state.Fail(instruction, "must write a scratch or physical address resource");
    const auto lane = MslUniformStoreLane(state, instruction, 0u, 2u);
    state.Line("if (" + Active(state, instruction) + (lane.empty() ? std::string() : " && " + lane) + ") {");
    state.Indent();
    const auto access = Prepare(state, instruction, memory);
    const auto address = GuestAddress(state, instruction, memory, access);
    const auto data = "uint(" + state.Arg(instruction, instruction.ArgumentCount() - 2u) + ")";
    const auto global = [&] {
        state.Line("aps5_bda_store(" + access.bda + ", " + address + ", " + data + ", " + std::to_string(bits) + "u, " + access.pc + ", " + access.coherent + ");");
    };
    if (!RoutesApertures(state, memory)) {
        global();
    } else {
        RouteFlat(state, address, [&] { ApertureStore(state, MslB5Lds(state), "uint(" + address + ")", bits, data); }, [&] { ApertureStore(state, MslB5Scratch(state), "uint(" + address + ")", bits, data); }, global);
    }
    state.Outdent();
    state.Line("}");
}

void StoreAddressWide(MslEmitterState& state, const IrValue& instruction, std::uint32_t components) {
    const auto& memory = state.Memory(instruction);
    if (memory.kind != ResourceKind::Flat && memory.kind != ResourceKind::Global) state.Fail(instruction, "must write a physical address resource");
    const auto lane = MslUniformStoreLane(state, instruction, 0u, 2u);
    state.Line("if (" + Active(state, instruction) + (lane.empty() ? std::string() : " && " + lane) + ") {");
    state.Indent();
    const auto access = Prepare(state, instruction, memory);
    const auto base = GuestAddressBase(state, instruction, memory, access);
    const auto data = state.Arg(instruction, instruction.ArgumentCount() - 2u);
    const auto global = [&] {
        static constexpr const char* Widen[] = {"", "", "uint4(%, 0u, 0u)", "uint4(%, 0u)", "%"};
        std::string values = Widen[components];
        values.replace(values.find('%'), 1u, data);
        const auto stop = state.Temporary("aps5_st");
        state.Line("bool " + stop + " = false;");
        state.Line("aps5_bda_store_dwords(" + access.bda + ", " + base + ", " + MslHex(memory.offset) + "u, " + std::to_string(components) + "u, " + values + ", " + access.pc + ", " + access.coherent + ", " + access.stops + ", " + stop + ");");
        MslB5StopIf(state, stop);
    };
    if (!RoutesApertures(state, memory)) {
        global();
    } else {
        const auto address = AddImmediate(state, base, static_cast<std::int32_t>(memory.offset), access);
        const auto aperture = [&](const MslB5Storage& storage) {
            for (std::uint32_t component = 0; component < components; ++component) {
                ApertureStore(state, storage, "(uint(" + address + ") + " + std::to_string(component * 4u) + "u)", 32u, data + "[" + std::to_string(component) + "]");
            }
        };
        RouteFlat(state, address, [&] { aperture(MslB5Lds(state)); }, [&] { aperture(MslB5Scratch(state)); }, global);
    }
    state.Outdent();
    state.Line("}");
}

std::optional<MslAtomicOp> AddressAtomicOp(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::AddressAtomicSwap32:
        return MslAtomicOp::Swap;
    case IrOpcode::AddressAtomicCmpSwap32:
        return MslAtomicOp::CmpSwap;
    case IrOpcode::AddressAtomicIAdd32:
        return MslAtomicOp::Add;
    case IrOpcode::AddressAtomicISub32:
        return MslAtomicOp::Sub;
    case IrOpcode::AddressAtomicSMin32:
        return MslAtomicOp::SMin;
    case IrOpcode::AddressAtomicUMin32:
        return MslAtomicOp::UMin;
    case IrOpcode::AddressAtomicSMax32:
        return MslAtomicOp::SMax;
    case IrOpcode::AddressAtomicUMax32:
        return MslAtomicOp::UMax;
    case IrOpcode::AddressAtomicAnd32:
        return MslAtomicOp::And;
    case IrOpcode::AddressAtomicOr32:
        return MslAtomicOp::Or;
    case IrOpcode::AddressAtomicXor32:
        return MslAtomicOp::Xor;
    case IrOpcode::AddressAtomicInc32:
        return MslAtomicOp::Inc;
    case IrOpcode::AddressAtomicDec32:
        return MslAtomicOp::Dec;
    case IrOpcode::AddressAtomicUSubSat32:
        return MslAtomicOp::USubSat;
    case IrOpcode::AddressAtomicFMin32:
        return MslAtomicOp::FMin;
    case IrOpcode::AddressAtomicFMax32:
        return MslAtomicOp::FMax;
    case IrOpcode::AddressAtomicFCmpSwap32:
        return MslAtomicOp::FCmpSwap;
    default:
        return std::nullopt;
    }
}

std::optional<MslAtomicOp> AddressAtomic64Op(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::AddressAtomicSwap64: return MslAtomicOp::Swap;
    case IrOpcode::AddressAtomicCmpSwap64: return MslAtomicOp::CmpSwap;
    case IrOpcode::AddressAtomicIAdd64: return MslAtomicOp::Add;
    case IrOpcode::AddressAtomicISub64: return MslAtomicOp::Sub;
    case IrOpcode::AddressAtomicSMin64: return MslAtomicOp::SMin;
    case IrOpcode::AddressAtomicUMin64: return MslAtomicOp::UMin;
    case IrOpcode::AddressAtomicSMax64: return MslAtomicOp::SMax;
    case IrOpcode::AddressAtomicUMax64: return MslAtomicOp::UMax;
    case IrOpcode::AddressAtomicAnd64: return MslAtomicOp::And;
    case IrOpcode::AddressAtomicOr64: return MslAtomicOp::Or;
    case IrOpcode::AddressAtomicXor64: return MslAtomicOp::Xor;
    case IrOpcode::AddressAtomicFCmpSwap64: return MslAtomicOp::FCmpSwap;
    case IrOpcode::AddressAtomicFMin64: return MslAtomicOp::FMin;
    case IrOpcode::AddressAtomicFMax64: return MslAtomicOp::FMax;
    case IrOpcode::AddressAtomicInc64: return MslAtomicOp::Inc;
    case IrOpcode::AddressAtomicDec64: return MslAtomicOp::Dec;
    default: return std::nullopt;
    }
}

// AddressAtomic (64-bit), FlatAtomicNext64 / ApertureAtomic64: global memory through the lock table
// of package B3, the shared aperture under the LDS lock, scratch as a plain update.
void AddressAtomic64(MslEmitterState& state, const IrValue& instruction, MslAtomicOp op) {
    const auto& memory = state.Memory(instruction);
    if (memory.kind != ResourceKind::Flat && memory.kind != ResourceKind::Global) state.Fail(instruction, "must access a physical address resource");
    const auto result = state.Name(instruction);
    state.Line(result + " = " + MslType(instruction.Type()) + "(0);");
    state.Line("if (" + Active(state, instruction) + ") {");
    state.Indent();
    const auto access = Prepare(state, instruction, memory);
    const auto address = GuestAddress(state, instruction, memory, access);
    const bool compares = op == MslAtomicOp::CmpSwap || op == MslAtomicOp::FCmpSwap;
    const auto value = "ulong(" + state.Arg(instruction, 3u) + ")";
    const auto comparator = compares ? "ulong(" + state.Arg(instruction, 4u) + ")" : std::string("0ul");
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    state.RequirePrelude(MslPreludeChunk::Atomics64);
    const auto global = [&] {
        const auto physical = state.Temporary("aps5_pa");
        state.Line("const ulong " + physical + " = aps5_bda_atomic_address(" + access.bda + ", " + address + ", 8u, 2u, " + access.pc + ");");
        state.Line("if (" + physical + " != 0ul) {");
        state.Indent();
        state.Line(result + " = " + MslDeviceAtomic64(state, instruction, op, physical, value, compares ? comparator : std::string()) + ";");
        if (const auto fence = MslAtomicFence(); !fence.empty()) state.Line(fence);
        state.Line("aps5_bda_note_write(" + access.bda + ".faults, " + address + ");");
        state.Outdent();
        state.Line("}");
    };
    if (!RoutesApertures(state, memory)) {
        global();
    } else {
        const auto aperture = [&](bool shared) {
            const auto low = state.Temporary("aps5_lo");
            state.Line("const uint " + low + " = uint(" + address + ");");
            state.Line("if ((" + low + " & 7u) != 0u) " + MslRecordFault(state, 7u, address, "8u", access.pc));
            const auto storage = shared ? MslB5Lds(state) : MslB5Scratch(state);
            if (storage.words.empty()) return;
            state.Line("else if ((" + low + " >> 2u) + 1u < " + std::to_string(storage.dwords) + "u) {");
            state.Indent();
            const auto update = [&] {
                const auto index = "(" + low + " >> 2u)";
                state.Line(result + " = ulong(" + storage.words + "[" + index + "]) | (ulong(" + storage.words + "[" + index + " + 1u]) << 32u);");
                const auto next = state.Temporary("aps5_n64");
                state.Line("const ulong " + next + " = aps5_atomic64_next(" + std::to_string(static_cast<std::uint32_t>(op)) + "u, " + result + ", " + value + ", " + comparator + ");");
                state.Line(storage.words + "[" + index + "] = uint(" + next + ");");
                state.Line(storage.words + "[" + index + " + 1u] = uint(" + next + " >> 32u);");
            };
            if (shared) MslLdsLocked(state, instruction, update);
            else update();
            state.Outdent();
            state.Line("}");
        };
        RouteFlat(state, address, [&] { aperture(true); }, [&] { aperture(false); }, global);
    }
    state.Outdent();
    state.Line("}");
}

// AddressAtomic (32-bit): arguments {address, low, high, value, [comparator,] exec}.
void AddressAtomic(MslEmitterState& state, const IrValue& instruction, MslAtomicOp op) {
    const auto& memory = state.Memory(instruction);
    if (memory.kind != ResourceKind::Flat && memory.kind != ResourceKind::Global) state.Fail(instruction, "must access a physical address resource");
    const auto result = state.Name(instruction);
    state.Line(result + " = 0u;");
    state.Line("if (" + Active(state, instruction) + ") {");
    state.Indent();
    const auto access = Prepare(state, instruction, memory);
    const auto address = GuestAddress(state, instruction, memory, access);
    const auto value = state.Arg(instruction, 3u);
    const bool compares = op == MslAtomicOp::CmpSwap || op == MslAtomicOp::FCmpSwap;
    const auto comparator = compares ? state.Arg(instruction, 4u) : std::string("0u");
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    const auto global = [&] {
        // EmitBdaAtomic: the element must be aligned and mapped writable; the store is noted for
        // write-back.
        const auto physical = state.Temporary("aps5_pa");
        state.Line("const ulong " + physical + " = aps5_bda_atomic_address(" + access.bda + ", " + address + ", 4u, 2u, " + access.pc + ");");
        state.Line("if (" + physical + " != 0ul) {");
        state.Indent();
        state.Line(result + " = " + MslAtomic32Expression(op, "reinterpret_cast<device atomic_uint*>(" + physical + ")", value, comparator) + ";");
        if (const auto fence = MslAtomicFence(); !fence.empty()) state.Line(fence);
        state.Line("aps5_bda_note_write(" + access.bda + ".faults, " + address + ");");
        state.Outdent();
        state.Line("}");
    };
    if (!RoutesApertures(state, memory)) {
        global();
    } else {
        // ApertureAtomic32: an unaligned element records an Unaligned fault and yields zero.
        const auto aperture = [&](bool shared) {
            const auto low = state.Temporary("aps5_lo");
            state.Line("const uint " + low + " = uint(" + address + ");");
            state.Line("if ((" + low + " & 3u) != 0u) " + MslRecordFault(state, 7u, address, "4u", access.pc));
            const auto storage = shared ? MslB5Lds(state) : MslB5Scratch(state);
            if (storage.words.empty()) return;
            state.Line("else if ((" + low + " >> 2u) < " + std::to_string(storage.dwords) + "u) {");
            state.Indent();
            if (shared && state.requirements.ldsLock) {
                // Every LDS atomic of a program with 64-bit LDS atomics runs under the LDS lock (B3).
                MslLdsLocked(state, instruction, [&] {
                    state.Line(result + " = " + storage.words + "[" + low + " >> 2u];");
                    state.Line(storage.words + "[" + low + " >> 2u] = aps5_atomic_next(" + std::to_string(static_cast<std::uint32_t>(op)) + "u, " + result + ", " + value + ", " + comparator + ");");
                });
            } else if (shared) {
                state.Line(result + " = " + MslAtomic32Expression(op, MslLdsAtomicPointer(state, low + " >> 2u"), value, comparator) + ";");
                state.Line("atomic_thread_fence(" + state.LdsMemoryFlags() + ", memory_order_acq_rel, thread_scope_threadgroup);");
            } else {
                state.Line(result + " = aps5_scratch_atomic(&" + storage.words + "[" + low + " >> 2u], " + std::to_string(static_cast<std::uint32_t>(op)) + "u, " + value + ", " + comparator + ");");
            }
            state.Outdent();
            state.Line("}");
        };
        RouteFlat(state, address, [&] { aperture(true); }, [&] { aperture(false); }, global);
    }
    state.Outdent();
    state.Line("}");
}

}

bool EmitMslAddressInstruction(MslEmitterState& state, const IrValue& instruction) {
    switch (instruction.Opcode()) {
    case IrOpcode::LoadAddressU8:
        LoadAddress(state, instruction, 8u);
        return true;
    case IrOpcode::LoadAddressU16:
        LoadAddress(state, instruction, 16u);
        return true;
    case IrOpcode::LoadAddressU32:
        LoadAddress(state, instruction, 32u);
        return true;
    case IrOpcode::LoadAddressU32x2:
        LoadAddressWide(state, instruction, 2u);
        return true;
    case IrOpcode::LoadAddressU32x3:
        LoadAddressWide(state, instruction, 3u);
        return true;
    case IrOpcode::LoadAddressU32x4:
        LoadAddressWide(state, instruction, 4u);
        return true;
    case IrOpcode::StoreAddressU8:
        StoreAddress(state, instruction, 8u);
        return true;
    case IrOpcode::StoreAddressU16:
        StoreAddress(state, instruction, 16u);
        return true;
    case IrOpcode::StoreAddressU32:
        StoreAddress(state, instruction, 32u);
        return true;
    case IrOpcode::StoreAddressU32x2:
        StoreAddressWide(state, instruction, 2u);
        return true;
    case IrOpcode::StoreAddressU32x3:
        StoreAddressWide(state, instruction, 3u);
        return true;
    case IrOpcode::StoreAddressU32x4:
        StoreAddressWide(state, instruction, 4u);
        return true;
    default:
        break;
    }
    if (const auto op = AddressAtomicOp(instruction.Opcode())) {
        AddressAtomic(state, instruction, *op);
        return true;
    }
    if (const auto op = AddressAtomic64Op(instruction.Opcode())) {
        AddressAtomic64(state, instruction, *op);
        return true;
    }
    return false;
}

}
