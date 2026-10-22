// libSceAgc's shader functions on headers, code and tables in displaced guest memory (D31, D33): ASTRO BOT keeps its
// shaders at 0x5xxxxxxxx in the shared region tail, where every native load through the guest address faults. Each
// function reads through host views (HostShader, GuestArena::HostRange), so a call takes no fault at all, gives the
// same results as the same shader graph in ordinary memory, and hands the driver and the guest guest addresses only.
// The driver entry points are stubs here (no GPU): agc_driver_displaced_guest_memory covers the driver's own reads.
#include "SceShaders.hpp"
#include "SceTypes.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include "prx/libSceAgc/Misc/include/ShaderFusion.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
int APS5_VABI sceAgcCreateShader(Shader** dst, void* header, const volatile void* code);
int APS5_VABI sceAgcCreateInterpolantMapping_0100(ShaderRegister* regs, const Shader* gs, const Shader* ps);
int APS5_VABI sceAgcCreatePrimState(ShaderRegister* cx_regs, ShaderRegister* uc_regs, const Shader* hs, const Shader* gs, std::uint32_t prim_type);
int APS5_VABI sceAgcLinkShaders(ShaderRegister* context, ShaderRegister* primitive, const void* reserved, const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType);
int APS5_VABI sceAgcGetGsPrimPayload(std::uint32_t* payload, const Shader* gs);
int APS5_VABI sceAgcGetGsOversubscription(ShaderRegister* regs, const Shader* gs, std::uint32_t budget, float factor);
std::uint32_t APS5_VABI sceAgcDriverWaitUntilSafeForRendering(std::uint32_t** command, std::uint32_t capacity, std::uint32_t mode, std::uint32_t handle, int index);
std::uint32_t APS5_VABI sceAgcDriverGetWaitRenderingPacketSizeInDwords();
}

namespace {

using namespace ShaderRegs;

constexpr std::uintptr_t CarveoutBase = 0x4000000000ull;
constexpr std::uintptr_t TailBase = 0x400000000ull;
constexpr std::size_t Bytes = 0x200000;
constexpr int CpuGpuReadWrite = 0x33;
constexpr int MapFixed = 0x10;
constexpr std::uint32_t SCodeEnd = 0xbf9f0000u;
constexpr std::uint32_t SSetpcS6 = 0xbe802006u;
constexpr std::uint32_t SMovS0 = 0xbe800380u;
constexpr std::uint32_t VMovV0 = 0x7e000280u;
constexpr std::uint32_t SEndpgm = 0xbf810000u;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t Faults() {
    std::uint64_t faults = 0, patches = 0;
    GuestArena::GuestDisplacedStats_nid_no_patch(&faults, &patches);
    return faults;
}

// What the driver stubs were handed.
struct Seen {
    std::vector<std::uintptr_t> registered;
    std::vector<std::uintptr_t> resolved;
} g_seen;

// Lays a shader graph out in [guest, guest + bytes), written through `host` (the alias, or the same address).
class Arena {
public:
    Arena(std::uintptr_t guest, std::uint8_t* host, std::size_t bytes) : guest_(guest), host_(host), bytes_(bytes) {}

    template <typename T>
    T* Place(std::span<const T> values, std::size_t alignment = alignof(T)) {
        cursor_ = (cursor_ + alignment - 1) & ~(alignment - 1);
        Require(cursor_ + values.size_bytes() <= bytes_, "arena full");
        std::memcpy(host_ + cursor_, values.data(), values.size_bytes());
        auto* placed = reinterpret_cast<T*>(guest_ + cursor_);
        cursor_ += values.size_bytes();
        return placed;
    }
    template <typename T>
    T* Place(const T& value, std::size_t alignment = alignof(T)) { return Place(std::span<const T>(&value, 1), alignment); }
    void* Reserve(std::size_t bytes, std::size_t alignment) {
        cursor_ = (cursor_ + alignment - 1) & ~(alignment - 1);
        Require(cursor_ + bytes <= bytes_, "arena full");
        std::memset(host_ + cursor_, 0xcd, bytes);
        auto* placed = reinterpret_cast<void*>(guest_ + cursor_);
        cursor_ += bytes;
        return placed;
    }
    template <typename T>
    T* Host(T* guest) const { return reinterpret_cast<T*>(host_ + (reinterpret_cast<std::uintptr_t>(guest) - guest_)); }
    bool Contains(std::uintptr_t address) const { return address >= guest_ && address < guest_ + bytes_; }

private:
    std::uintptr_t guest_;
    std::uint8_t* host_;
    std::size_t bytes_;
    std::size_t cursor_ = 0;
};

template <typename T>
std::intptr_t Relative(const T* target, const void* field) {
    return reinterpret_cast<std::intptr_t>(target) - reinterpret_cast<std::intptr_t>(field);
}

struct Graph {
    Shader* compute = nullptr;   // SDK layout: relative table offsets, resolved by sceAgcCreateShader
    const volatile void* computeCode = nullptr;
    ShaderRegister* computeRegisters = nullptr;
    Shader* front = nullptr;     // resolved layouts
    Shader* back = nullptr;
    Shader* vertex = nullptr;
    Shader* pixel = nullptr;
    void* scratch = nullptr;
    ShaderRegister* interpolants = nullptr;
    ShaderRegister* context = nullptr;
    ShaderRegister* primitive = nullptr;
};

Shader Header(ShaderBinaryType type) {
    Shader shader{};
    shader.file_header = SHADER_FILE_HEADER_MAGIC;
    shader.version = SHADER_VERSION;
    shader.header_size = sizeof(Shader);
    shader.type = static_cast<std::uint8_t>(type);
    return shader;
}

Graph Build(Arena& arena) {
    Graph graph;
    // A compute shader as the SDK ships it: the header stores each table as an offset from the field.
    const std::array<std::uint32_t, 4> computeCode{SMovS0, SEndpgm, SCodeEnd, SCodeEnd};
    graph.computeCode = arena.Place(std::span<const std::uint32_t>(computeCode), 256);
    const std::array<ShaderRegister, 3> computeSh{{{COMPUTE_PGM_LO, 0x10u}, {COMPUTE_PGM_LO + 1u, 0u}, {0x212u, 0x55u}}};
    graph.computeRegisters = arena.Place(std::span<const ShaderRegister>(computeSh));
    graph.compute = arena.Place(Header(ShaderBinaryType::Cs), 16);
    Shader compute = Header(ShaderBinaryType::Cs);
    compute.shader_size = sizeof(computeCode);
    compute.num_sh_registers = static_cast<std::uint8_t>(computeSh.size());
    compute.sh_registers = reinterpret_cast<ShaderRegister*>(Relative(graph.computeRegisters, &graph.compute->sh_registers));
    std::memcpy(arena.Host(graph.compute), &compute, sizeof(compute));

    // GS halves (the shapes of agc_shader_fusion).
    std::vector<std::uint32_t> frontCode{SMovS0, VMovV0, SSetpcS6, SCodeEnd, SCodeEnd};
    const auto codeBytes = static_cast<std::uint32_t>(frontCode.size() * 4);
    std::array<std::uint32_t, 2> magic{};
    std::memcpy(magic.data(), "barefoot", 8);
    frontCode.insert(frontCode.end(), {magic[0], magic[1], 0xabcdu, 0u, 1u, codeBytes, 0u});
    const std::array<std::uint32_t, 3> backCode{SMovS0, SEndpgm, SCodeEnd};
    const std::array<ShaderRegister, 4> frontSh{{{0x080u, 0x1234u}, {0x08Au, (2u << 6u) | 3u}, {0x08Bu, (6u << 1u) | (1u << 7u)}, {0x0B2u, 0x55u}}};
    const std::array<ShaderRegister, 2> frontCx{{{0x29Bu, 7u}, {0x2D5u, 0x24u}}};
    const std::array<ShaderRegister, 6> backSh{{{0x088u, 0u}, {0x089u, 0u}, {0x08Au, (1u << 24u) | (1u << 6u) | 5u}, {0x08Bu, (2u << 1u) | (1u << 7u) | (1u << 15u)}, {0x0C8u, 0u}, {0x0C9u, 0u}}};
    const std::array<ShaderRegister, 1> backCx{{{0x29Bu, 2u}}};
    std::array<ShaderSemantic, 2> inputs{};
    inputs[0].semantic = 1;
    inputs[1].semantic = 2;
    std::array<ShaderSemantic, 2> outputs{};
    outputs[0].semantic = 2;
    outputs[0].hardware_mapping = 5;
    outputs[1].semantic = 3;
    outputs[1].hardware_mapping = 6;
    const std::array<std::uint16_t, 2> direct{3u, 9u};
    ShaderSharp sharp{};
    sharp.offset_dw = 12;
    ShaderSpecialRegs frontSpecials{};
    frontSpecials.vgt_shader_stages_en = {VGT_SHADER_STAGES_EN, 1u << 22u};
    ShaderSpecialRegs backSpecials{};
    backSpecials.vgt_shader_stages_en = {VGT_SHADER_STAGES_EN, 1u << 22u};
    backSpecials.ge_cntl = {GE_CNTL, 0x77u};
    backSpecials.vgt_gs_out_prim_type = {VGT_GS_OUT_PRIM_TYPE, 2u};
    backSpecials.ge_user_vgpr_en = {GE_USER_VGPR_EN, 0u};

    ShaderUserData userData{};
    userData.direct_resource_offset = arena.Place(std::span<const std::uint16_t>(direct));
    userData.direct_resource_count = static_cast<std::uint16_t>(direct.size());
    userData.sharp_resource_offset[1] = arena.Place(sharp);
    userData.sharp_resource_count[1] = 1;
    userData.eud_size_dw = 4;

    Shader front = Header(ShaderBinaryType::GsFront);
    front.code = arena.Place(std::span<const std::uint32_t>(frontCode), 256);
    front.shader_size = static_cast<std::uint32_t>(frontCode.size() * 4);
    front.sh_registers = arena.Place(std::span<const ShaderRegister>(frontSh));
    front.num_sh_registers = static_cast<std::uint8_t>(frontSh.size());
    front.cx_registers = arena.Place(std::span<const ShaderRegister>(frontCx));
    front.num_cx_registers = static_cast<std::uint8_t>(frontCx.size());
    front.input_semantics = arena.Place(std::span<const ShaderSemantic>(inputs));
    front.num_input_semantics = static_cast<std::uint32_t>(inputs.size());
    front.specials = arena.Place(frontSpecials);
    front.scratch_size_dw_per_thread = 16;
    graph.front = arena.Place(front, 16);

    Shader back = Header(ShaderBinaryType::GsBack);
    back.code = arena.Place(std::span<const std::uint32_t>(backCode), 256);
    back.shader_size = sizeof(backCode);
    back.sh_registers = arena.Place(std::span<const ShaderRegister>(backSh));
    back.num_sh_registers = static_cast<std::uint8_t>(backSh.size());
    back.cx_registers = arena.Place(std::span<const ShaderRegister>(backCx));
    back.num_cx_registers = static_cast<std::uint8_t>(backCx.size());
    back.output_semantics = arena.Place(std::span<const ShaderSemantic>(outputs));
    back.num_output_semantics = static_cast<std::uint16_t>(outputs.size());
    back.user_data = arena.Place(userData);
    back.specials = arena.Place(backSpecials);
    back.special_sizes_bytes = sizeof(ShaderSpecialRegs);
    back.scratch_size_dw_per_thread = 8;
    graph.back = arena.Place(back, 16);

    // A linked vertex (GS) and pixel pair.
    const std::array<ShaderRegister, 6> vertexCx{{{VGT_GS_ONCHIP_CNTL, 0x80u | (64u << 11u)}, {GE_NGG_SUBGRP_CNTL, 0x80u}, {SPI_VS_OUT_CONFIG, 0x4u},
        {PA_CL_VS_OUT_CNTL, 1u << 21u}, {GE_MAX_OUTPUT_PER_SUBGROUP, 0x100u}, {SPI_SHADER_IDX_FORMAT, 2u}}};
    Shader vertex = Header(ShaderBinaryType::Gs);
    vertex.code = back.code;
    vertex.shader_size = back.shader_size;
    vertex.cx_registers = arena.Place(std::span<const ShaderRegister>(vertexCx));
    vertex.num_cx_registers = static_cast<std::uint8_t>(vertexCx.size());
    vertex.output_semantics = back.output_semantics;
    vertex.num_output_semantics = back.num_output_semantics;
    vertex.specials = back.specials;
    vertex.special_sizes_bytes = sizeof(ShaderSpecialRegs);
    graph.vertex = arena.Place(vertex, 16);
    Shader pixel = Header(ShaderBinaryType::Ps);
    pixel.code = back.code;
    pixel.shader_size = back.shader_size;
    pixel.input_semantics = front.input_semantics;
    pixel.num_input_semantics = front.num_input_semantics;
    graph.pixel = arena.Place(pixel, 16);

    graph.scratch = arena.Reserve(0x2000, 256);
    graph.interpolants = static_cast<ShaderRegister*>(arena.Reserve(32 * sizeof(ShaderRegister), 8));
    graph.context = static_cast<ShaderRegister*>(arena.Reserve(34 * sizeof(ShaderRegister), 8));
    graph.primitive = static_cast<ShaderRegister*>(arena.Reserve(3 * sizeof(ShaderRegister), 8));
    return graph;
}

// The results of every call, read back through the arena's host view, with guest addresses made relative to it.
struct Results {
    std::vector<std::uint8_t> bytes;
    template <typename T>
    void Add(const T& value) {
        const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
        bytes.insert(bytes.end(), raw, raw + sizeof(T));
    }
};

Results Exercise(const Arena& arena, Graph& graph, std::uintptr_t guestBase) {
    Results results;
    g_seen = {};
    const auto offset = [&](const volatile void* pointer) {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        Require(pointer == nullptr || arena.Contains(address), "a result points outside the shader memory (a host view escaped?)");
        return pointer == nullptr ? std::uint64_t{0} : static_cast<std::uint64_t>(address - guestBase);
    };

    Shader* created = nullptr;
    Require(sceAgcCreateShader(&created, graph.compute, graph.computeCode) == 0, "sceAgcCreateShader failed");
    Require(created == graph.compute, "sceAgcCreateShader returned another header");
    Require(g_seen.registered.size() == 1 && g_seen.registered[0] == reinterpret_cast<std::uintptr_t>(graph.compute), "the driver did not get the guest header");
    const Shader compute = *arena.Host(graph.compute);
    Require(compute.sh_registers == graph.computeRegisters, "the resolved register table is not the guest's");
    Require(compute.code == graph.computeCode, "the code pointer is not the guest's");
    const auto* registers = arena.Host(graph.computeRegisters);
    const auto program = (std::uint64_t{registers[0].value} << 8u) | (std::uint64_t{registers[1].value & 0xffu} << 40u);
    Require(program == reinterpret_cast<std::uint64_t>(graph.computeCode) + (0x10u << 8u), "the program address was not patched to the guest code");
    results.Add(registers[0].value - static_cast<std::uint32_t>(guestBase >> 8u));
    results.Add(registers[2]);

    SizeAlign size{};
    Require(sceAgcGetFusedShaderSize_0080(&size, graph.front, graph.back) == 0, "sceAgcGetFusedShaderSize failed");
    results.Add(size.m_size);
    Shader fused{};
    Require(sceAgcFuseShaderHalves_0200(&fused, graph.front, graph.back, graph.scratch) == 0, "sceAgcFuseShaderHalves failed");
    // A geometry fusion keeps its program in memory libSceAgc owns (upstream d6f67d1be), host memory outside the
    // displaced window whose host addresses are its guest addresses; only the fused register list goes to the
    // caller's memory, written through its host view.
    Require(g_seen.registered.size() == 2, "the driver did not get the fused header");
    const auto owned = g_seen.registered[1];
    Require(!arena.Contains(owned) && !GuestArena::IsDisplaced(owned), "the fused program is not in libSceAgc-owned host memory");
    Require(fused.sh_registers == graph.scratch, "the fused register list is not in the caller's memory");
    const auto fusedBase = reinterpret_cast<std::uintptr_t>(fused.code);
    const auto inProgram = [&](const volatile void* pointer) {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        Require(pointer == nullptr || (!arena.Contains(address) && address >= fusedBase && address < owned + fused.header_size),
            "a fused table is outside the libSceAgc-owned program");
        return pointer == nullptr ? std::uint64_t{0} : static_cast<std::uint64_t>(address - fusedBase);
    };
    for (const volatile void* pointer : {static_cast<const volatile void*>(fused.cx_registers), static_cast<const volatile void*>(fused.specials),
             static_cast<const volatile void*>(fused.input_semantics), static_cast<const volatile void*>(fused.output_semantics),
             static_cast<const volatile void*>(fused.user_data)})
        results.Add(inProgram(pointer));
    results.Add(inProgram(reinterpret_cast<const Shader*>(owned)));
    results.Add(offset(fused.sh_registers));
    const Shader stored = *reinterpret_cast<const Shader*>(owned);
    Shader expectedHeader = fused;
    expectedHeader.sh_registers = stored.sh_registers;
    Require(std::memcmp(&stored, &expectedHeader, sizeof(Shader)) == 0, "the registered fused header differs from the result");
    const auto* fusedSh = arena.Host(fused.sh_registers);
    Require(std::memcmp(stored.sh_registers, fusedSh, fused.num_sh_registers * sizeof(ShaderRegister)) == 0, "the caller's register list differs from the program's");
    const auto* fusedUserData = fused.user_data;
    results.Add(inProgram(fusedUserData->direct_resource_offset));
    results.Add(inProgram(fusedUserData->sharp_resource_offset[1]));
    results.Add(fusedUserData->direct_resource_offset[1]);
    const auto* fusedCode = static_cast<const std::uint32_t*>(const_cast<const void*>(fused.code));
    for (std::size_t i = 0; i < fused.shader_size / 4; ++i) results.Add(fusedCode[i]);
    for (std::size_t i = 0; i < fused.num_sh_registers; ++i) {
        // the program address registers hold the owned program's address
        // (bits 8-15 of the address in the hi register differ between programs: left out)
        const bool address = fusedSh[i].offset == SPI_SHADER_PGM_LO_ES || fusedSh[i].offset == SPI_SHADER_PGM_LO_GS;
        const bool high = fusedSh[i].offset == SPI_SHADER_PGM_LO_ES + 1u || fusedSh[i].offset == SPI_SHADER_PGM_LO_GS + 1u;
        results.Add(fusedSh[i].offset);
        results.Add(address ? fusedSh[i].value - static_cast<std::uint32_t>(fusedBase >> 8u) : high ? fusedSh[i].value & 0xffffff00u : fusedSh[i].value);
    }
    results.Add(fused.shader_size);
    results.Add(fused.header_size);

    Require(sceAgcCreateInterpolantMapping_0100(graph.interpolants, graph.vertex, graph.pixel) == 0, "sceAgcCreateInterpolantMapping failed");
    for (std::size_t i = 0; i < 32; ++i) results.Add(arena.Host(graph.interpolants)[i]);
    Require(sceAgcCreatePrimState(graph.context, graph.primitive, nullptr, graph.vertex, 4) == 0, "sceAgcCreatePrimState failed");
    for (std::size_t i = 0; i < 2; ++i) results.Add(arena.Host(graph.context)[i]);
    for (std::size_t i = 0; i < 3; ++i) results.Add(arena.Host(graph.primitive)[i]);
    Require(sceAgcLinkShaders(graph.context, graph.primitive, nullptr, graph.vertex, graph.pixel, 4) == 0, "sceAgcLinkShaders failed");
    for (std::size_t i = 0; i < 34; ++i) results.Add(arena.Host(graph.context)[i]);
    std::uint32_t payload = 0xffffffffu;
    Require(sceAgcGetGsPrimPayload(&payload, graph.vertex) == 0, "sceAgcGetGsPrimPayload failed");
    results.Add(payload);
    std::array<ShaderRegister, 2> oversubscription{};
    Require(sceAgcGetGsOversubscription(oversubscription.data(), graph.vertex, 4096u, 0.5f) == 0, "sceAgcGetGsOversubscription failed");
    results.Add(oversubscription);
    for (const auto shader : g_seen.resolved) Require(arena.Contains(shader), "the driver was handed a header outside the shader memory (a host view escaped?)");
    Require(!g_seen.resolved.empty(), "the driver resolved no shader ABI");
    return results;
}

}

// Driver entry points (stubs: this test runs without a GPU).
extern "C" void AgcDriverRegisterShader_nid_postfix(const Shader* shader) {
    g_seen.registered.push_back(reinterpret_cast<std::uintptr_t>(shader));
}
extern "C" void AgcDriverResolveShaderAbi_nid_postfix(const Shader* shader, std::span<const ShaderRegister>, std::span<const ShaderRegister>) {
    g_seen.resolved.push_back(reinterpret_cast<std::uintptr_t>(shader));
}
extern "C" void AgcDriverResolveGraphicsStagesAbi_nid_postfix(std::span<const Shader* const> stages, std::span<const ShaderRegister>, std::span<const ShaderRegister>) {
    for (const auto* stage : stages) g_seen.resolved.push_back(reinterpret_cast<std::uintptr_t>(stage));
}
extern "C" void AgcDriverResolveGraphicsAbi_nid_postfix(const Shader* vertex, const Shader* pixel, std::uint32_t) {
    g_seen.resolved.push_back(reinterpret_cast<std::uintptr_t>(vertex));
    if (pixel != nullptr) g_seen.resolved.push_back(reinterpret_cast<std::uintptr_t>(pixel));
}

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("displaced shaders: no displaced window on this host, skipped");
    return 0;
#else
    try {
        Require(GuestArena::GuestDisplacedActive_nid_no_patch(), "the displaced window is not active on macOS arm64");
        std::vector<std::uint8_t> ordinary(Bytes + 0x1000);
        const auto ordinaryBase = (reinterpret_cast<std::uintptr_t>(ordinary.data()) + 0xfff) & ~std::uintptr_t{0xfff};
        Arena reference(ordinaryBase, reinterpret_cast<std::uint8_t*>(ordinaryBase), Bytes);
        auto referenceGraph = Build(reference);
        const auto expected = Exercise(reference, referenceGraph, ordinaryBase);

        std::int64_t physical = 0;
        Require(sceKernelAllocateDirectMemory(0, 0x100000000ll, Bytes, 0x200000, 3, &physical) == 0, "allocate direct memory");
        for (const std::uintptr_t base : {TailBase, CarveoutBase}) {
            Require(GuestArena::IsDisplaced(base), "0x" + std::to_string(base) + " is not displaced");
            void* fixed = reinterpret_cast<void*>(base);
            Require(sceKernelMapDirectMemory(&fixed, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && fixed == reinterpret_cast<void*>(base), "the fixed mapping failed");
            Arena displaced(base, reinterpret_cast<std::uint8_t*>(GuestArena::HostAddress(base)), Bytes);
            auto graph = Build(displaced);
            const auto before = Faults();
            const auto results = Exercise(displaced, graph, base);
            const auto faults = Faults() - before;
            char where[32];
            std::snprintf(where, sizeof(where), "0x%llx: ", static_cast<unsigned long long>(base));
            Require(faults == 0, std::string(where) + std::to_string(faults) + " displaced faults: a shader function read guest memory through the guest address");
            Require(results.bytes == expected.bytes, std::string(where) + "results differ from the same shaders in ordinary memory");
            // The driver's rendering-wait packet: the command slot and the packet both in displaced memory.
            {
                auto** slot = reinterpret_cast<std::uint32_t**>(base + Bytes - 0x100);
                auto* packet = reinterpret_cast<std::uint32_t*>(base + Bytes - 0x80);
                auto** slotHost = GuestArena::HostPointer(slot);
                *slotHost = packet;
                const auto words = sceAgcDriverGetWaitRenderingPacketSizeInDwords();
                const auto waitBefore = Faults();
                Require(sceAgcDriverWaitUntilSafeForRendering(slot, words, 0, 7, 2) == 0, "sceAgcDriverWaitUntilSafeForRendering failed");
                Require(Faults() == waitBefore, std::string(where) + "sceAgcDriverWaitUntilSafeForRendering faulted on displaced memory");
                Require(*slotHost == packet + words, "the command pointer did not advance by the packet (as a guest pointer)");
                const auto* packetHost = GuestArena::HostPointer(packet);
                Require(packetHost[1] == 7 && packetHost[2] == 2 && packetHost[3] == 0, "the rendering-wait packet has the wrong payload");
            }
            Require(sceKernelMunmap(fixed, Bytes) == 0, "munmap");
            std::printf("displaced shaders at 0x%llx: 0 faults, results match ordinary memory\n", static_cast<unsigned long long>(base));
        }
        Require(sceKernelReleaseDirectMemory(physical, Bytes) == 0, "release direct memory");
        std::puts("displaced shader tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "displaced shaders: %s\n", error.what());
        return 1;
    }
#endif
}
