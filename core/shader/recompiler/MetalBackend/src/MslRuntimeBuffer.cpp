#include "MetalBackend/MslEmitterState.hpp"
#include "PipelineSpecialization.hpp"
#include <array>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

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

constexpr std::array<MslPreludeChunk, 1> OnCommon {MslPreludeChunk::Common};

// The parts of a MUBUF access the shader needs, as SpirvRuntimeBuffer.cpp descriptor() builds them
// for a V# the driver resolved at pipeline time: words 1 and 3, presence and the base's
// misalignment are function constants (BufferBase + 4 * resource), the record count stays a
// runtime value (the handle's word 2).
struct StaticDescriptor {
    std::string word1;
    std::string records;
    std::string word3;
    std::string valid;
    std::string misalignment;
    std::string index;
    std::string buffer;
    std::string byteOffset;
};

StaticDescriptor Descriptor(MslEmitterState& state, const IrValue& instruction) {
    const auto& memory = state.Memory(instruction);
    const auto* handle = instruction.Argument(0)->Resolve();
    if (handle->Opcode() != IrOpcode::GetBufferResource || handle->ArgumentCount() != 4u) state.Fail(instruction, "requires a runtime four-dword V#");
    if (memory.gpuDescriptor) state.Fail(instruction, "buffers whose V# lives in GPU memory (BDA access) are not implemented in the MSL backend yet");
    if (memory.kind != ResourceKind::Buffer) state.Fail(instruction, "scalar buffer loads are not implemented in the MSL backend yet");
    const auto first = PipelineSpecialization::BufferBase + memory.resource * PipelineSpecialization::BufferWords;
    StaticDescriptor descriptor;
    descriptor.word1 = state.FunctionConstant(first, 0u);
    descriptor.word3 = state.FunctionConstant(first + 1u, 0u);
    descriptor.valid = "(" + state.FunctionConstant(first + 2u, 0u) + " != 0u)";
    descriptor.misalignment = state.FunctionConstant(first + 3u, 0u);
    descriptor.records = state.Value(handle->Argument(2));
    const auto addThreadId = "(extract_bits(" + descriptor.word3 + ", 23u, 1u) != 0u)";
    if (state.program.Resources().stage != IrShaderStage::Compute) state.Fail(instruction, "buffer access outside compute is not implemented in the MSL backend yet");
    if (state.laneCount != 1u) state.Fail(instruction, "buffer access of wave64 programs on 32-wide SIMD groups is not implemented in the MSL backend yet");
    const auto lane = state.hostSubgroupSize < state.program.WaveSize() ? "(" + state.Builtin(MslBuiltin::ThreadIndexInThreadgroup) + " & " + std::to_string(state.program.WaveSize() - 1u) + "u)" : state.Builtin(MslBuiltin::ThreadIndexInSimdgroup);
    descriptor.index = "(" + state.Arg(instruction, 1u) + " + (" + addThreadId + " ? " + lane + " : 0u))";
    const auto element = state.DescriptorElement(DescriptorBindingKind::Buffers, memory.resource);
    descriptor.buffer = state.TableBuffer(DescriptorBindingKind::Buffers, element);
    descriptor.byteOffset = state.MemoryByteOffset(element);
    return descriptor;
}

std::string Access(MslEmitterState& state, const IrValue& instruction, const StaticDescriptor& descriptor, std::uint32_t component) {
    const auto& memory = state.Memory(instruction);
    const auto offset = "(" + state.Arg(instruction, 2u) + " + " + std::to_string(memory.offset + component * 4u) + "u)";
    return "aps5_buffer_access(" + descriptor.word1 + ", " + descriptor.records + ", " + descriptor.word3 + ", " + descriptor.valid + ", " + descriptor.index + ", " + offset + ", " + state.Arg(instruction, 3u) + ", " + std::to_string(memory.dataBits / 8u) + "u)";
}

void Load(MslEmitterState& state, const IrValue& instruction, std::uint32_t components) {
    const auto& memory = state.Memory(instruction);
    if (memory.formatted) state.Fail(instruction, "typed (formatted) buffer access is not implemented in the MSL backend yet");
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    const auto result = state.Name(instruction);
    const auto active = state.Arg(instruction, instruction.ArgumentCount() - 1u);
    state.Line(result + " = " + MslType(instruction.Type()) + "(0);");
    state.Line("if (" + active + ") {");
    state.Indent();
    const auto descriptor = Descriptor(state, instruction);
    for (std::uint32_t component = 0; component < components; ++component) {
        const auto access = "aps5_a" + std::to_string(instruction.Id()) + "_" + std::to_string(component);
        state.Line("const Aps5BufferAccess " + access + " = " + Access(state, instruction, descriptor, component) + ";");
        const auto target = components == 1u ? result : result + "[" + std::to_string(component) + "]";
        state.Line("if (" + access + ".inBounds) " + target + " = aps5_buffer_read(" + descriptor.buffer + ", " + descriptor.byteOffset + ", " + descriptor.misalignment + ", " + access + ".byte, " + std::to_string(memory.dataBits) + "u);");
    }
    state.Outdent();
    state.Line("}");
}

void Store(MslEmitterState& state, const IrValue& instruction, std::uint32_t components) {
    const auto& memory = state.Memory(instruction);
    if (memory.formatted) state.Fail(instruction, "typed (formatted) buffer access is not implemented in the MSL backend yet");
    state.RequirePrelude(MslPreludeChunk::RuntimeBuffer);
    const auto active = state.Arg(instruction, instruction.ArgumentCount() - 1u);
    const auto data = state.Arg(instruction, instruction.ArgumentCount() - 2u);
    state.Line("if (" + active + ") {");
    state.Indent();
    const auto descriptor = Descriptor(state, instruction);
    for (std::uint32_t component = 0; component < components; ++component) {
        const auto access = "aps5_a" + std::to_string(instruction.Id()) + "_" + std::to_string(component);
        state.Line("const Aps5BufferAccess " + access + " = " + Access(state, instruction, descriptor, component) + ";");
        const auto value = components == 1u ? data : data + "[" + std::to_string(component) + "]";
        state.Line("if (" + access + ".inBounds) aps5_buffer_write(" + descriptor.buffer + ", " + descriptor.byteOffset + ", " + descriptor.misalignment + ", " + access + ".byte, " + value + ", " + std::to_string(memory.dataBits) + "u);");
    }
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
    default:
        return false;
    }
}

MslPreludeEntry MslBufferPrelude(MslPreludeChunk chunk) {
    switch (chunk) {
    case MslPreludeChunk::RuntimeBuffer:
        return {RuntimeBufferText, OnCommon};
    default:
        throw std::runtime_error("MSL emission failed: buffer prelude chunk " + std::to_string(static_cast<std::uint32_t>(chunk)) + " is not implemented yet");
    }
}

}
