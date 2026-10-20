#include "prx/libSceAgc/Misc/include/ShaderFusion.hpp"
#include "prx/libSceAgcDriver/tests/execution/VulkanTestDevice.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t SCodeEnd = 0xbf9f0000u;
constexpr std::uint32_t SNop = 0xbf800000u;
constexpr std::uint32_t SSetpcS6 = 0xbe802006u;
constexpr std::uint32_t SSwappcNullS6 = 0xbefd2106u;
constexpr std::uint32_t SMovS0 = 0xbe800380u;
constexpr std::uint32_t VMovV0 = 0x7e000280u;
constexpr std::uint32_t SEndpgm = 0xbf810000u;
constexpr int InvalidShaderHalves = static_cast<int>(0x8a6c0008u);
constexpr std::uint32_t GsStage = 1u << 22u;
constexpr std::uint32_t HsStage = 1u << 21u;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename TAction>
void expectFailure(TAction action) {
    try {
        action();
    } catch (const std::exception& error) {
        check(error.what()[0] != '\0', "empty exception message");
        return;
    }
    throw std::runtime_error("expected an exception");
}

std::uint32_t word(const void* memory, std::size_t index) {
    std::uint32_t value = 0;
    std::memcpy(&value, static_cast<const std::uint8_t*>(memory) + index * 4, sizeof(value));
    return value;
}

const ShaderRegister* find(const Shader& shader, std::uint32_t offset) {
    for (std::uint32_t i = 0; i < shader.num_sh_registers; ++i) {
        if (shader.sh_registers[i].offset == offset) return &shader.sh_registers[i];
    }
    return nullptr;
}

struct Halves {
    std::vector<std::uint32_t> frontCode;
    std::array<std::uint32_t, 3> backCode{SMovS0, SEndpgm, SCodeEnd};
    std::array<ShaderRegister, 4> frontSh{{{0x080u, 0x1234u}, {0x08Au, (2u << 6u) | 3u}, {0x08Bu, (6u << 1u) | (1u << 7u)}, {0x0B2u, 0x55u}}};
    std::array<ShaderRegister, 2> frontCx{{{0x29Bu, 7u}, {0x2D5u, 0x24u}}};
    std::array<ShaderRegister, 6> backSh{{{0x088u, 0u}, {0x089u, 0u}, {0x08Au, (1u << 24u) | (1u << 6u) | 5u}, {0x08Bu, (2u << 1u) | (1u << 7u) | (1u << 15u)}, {0x0C8u, 0u}, {0x0C9u, 0u}}};
    std::array<ShaderRegister, 1> backCx{{{0x29Bu, 2u}}};
    std::array<ShaderSemantic, 2> inputs{};
    std::array<ShaderSemantic, 1> outputs{};
    std::array<std::uint16_t, 2> direct{3u, 9u};
    std::array<ShaderSharp, 1> sharps{};
    ShaderUserData userData{};
    ShaderSpecialRegs frontSpecials{};
    ShaderSpecialRegs backSpecials{};
    Shader front{};
    Shader back{};

    explicit Halves(std::uint32_t terminator = SSetpcS6) {
        frontCode = {SMovS0, VMovV0, terminator, SCodeEnd, SCodeEnd};
        const auto codeBytes = static_cast<std::uint32_t>(frontCode.size() * 4);
        std::array<std::uint32_t, 2> magic{};
        std::memcpy(magic.data(), "barefoot", 8);
        frontCode.insert(frontCode.end(), {magic[0], magic[1], 0xabcdu, 0u, 1u, codeBytes, 0u});
        inputs[0].semantic = 1;
        inputs[1].semantic = 2;
        outputs[0].semantic = 3;
        sharps[0].offset_dw = 12;
        userData.direct_resource_offset = direct.data();
        userData.direct_resource_count = static_cast<std::uint16_t>(direct.size());
        userData.sharp_resource_offset[1] = sharps.data();
        userData.sharp_resource_count[1] = 1;
        userData.eud_size_dw = 4;
        frontSpecials.vgt_shader_stages_en.value = GsStage;
        backSpecials.vgt_shader_stages_en.value = GsStage;
        backSpecials.ge_cntl.value = 0x77u;
        front.file_header = ShaderRegs::SHADER_FILE_HEADER_MAGIC;
        front.version = ShaderRegs::SHADER_VERSION;
        front.code = frontCode.data();
        front.shader_size = static_cast<std::uint32_t>(frontCode.size() * 4);
        front.header_size = sizeof(Shader);
        front.sh_registers = frontSh.data();
        front.num_sh_registers = static_cast<std::uint8_t>(frontSh.size());
        front.cx_registers = frontCx.data();
        front.num_cx_registers = static_cast<std::uint8_t>(frontCx.size());
        front.input_semantics = inputs.data();
        front.num_input_semantics = static_cast<std::uint32_t>(inputs.size());
        front.specials = &frontSpecials;
        front.scratch_size_dw_per_thread = 16;
        front.type = static_cast<std::uint8_t>(ShaderRegs::ShaderBinaryType::GsFront);
        back.file_header = ShaderRegs::SHADER_FILE_HEADER_MAGIC;
        back.version = ShaderRegs::SHADER_VERSION;
        back.code = backCode.data();
        back.shader_size = static_cast<std::uint32_t>(backCode.size() * 4);
        back.header_size = sizeof(Shader);
        back.sh_registers = backSh.data();
        back.num_sh_registers = static_cast<std::uint8_t>(backSh.size());
        back.cx_registers = backCx.data();
        back.num_cx_registers = static_cast<std::uint8_t>(backCx.size());
        back.output_semantics = outputs.data();
        back.num_output_semantics = static_cast<std::uint16_t>(outputs.size());
        back.user_data = &userData;
        back.specials = &backSpecials;
        back.special_sizes_bytes = sizeof(ShaderSpecialRegs);
        back.scratch_size_dw_per_thread = 8;
        back.type = static_cast<std::uint8_t>(ShaderRegs::ShaderBinaryType::GsBack);
    }
};

std::uint8_t* aligned(std::vector<std::uint8_t>& storage, std::size_t alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
    return storage.data() + ((alignment - address % alignment) % alignment);
}

void testFusion(std::size_t misalignment, std::uint32_t terminator = SSetpcS6) {
    Halves halves(terminator);
    SizeAlign size{};
    check(sceAgcGetFusedShaderSize_0080(&size, &halves.front, &halves.back) == 0, "fused size query failed");
    check(size.m_size == 7 * sizeof(ShaderRegister) && size.m_align == 4, "fused size is not the fused register list");
    std::vector<std::uint8_t> storage(size.m_size + 512, 0xcd);
    auto* scratch = aligned(storage, 256) + misalignment;
    Shader fused{};
    check(sceAgcFuseShaderHalves_0200(&fused, &halves.front, &halves.back, scratch) == 0, "fusion failed");
    for (std::size_t i = 0; i < 16; ++i) check(scratch[size.m_size + i] == 0xcd, "fusion wrote past the reported size");
    const auto base = reinterpret_cast<std::uintptr_t>(fused.code);
    const auto* memory = reinterpret_cast<const std::uint8_t*>(base);
    check(fused.type == static_cast<std::uint8_t>(ShaderRegs::ShaderBinaryType::Gs), "fused shader is not a GS");
    check(base % 256 == 0 && fused.shader_size == 256 + halves.back.shader_size, "fused code placement is wrong");
    check(word(memory, 0) == SMovS0 && word(memory, 1) == VMovV0, "front program was not copied");
    for (std::size_t i = 2; i < 64; ++i) check(word(memory, i) == SNop, "front program does not fall through to the back half");
    check(std::memcmp(memory + 256, halves.backCode.data(), halves.back.shader_size) == 0, "back program was not copied");
    check(reinterpret_cast<std::uintptr_t>(fused.sh_registers) == reinterpret_cast<std::uintptr_t>(scratch), "fused registers are not in the caller's memory");
    const auto* backLo = find(fused, 0x088u);
    const auto* frontLo = find(fused, 0x0C8u);
    check(backLo != nullptr && backLo->value == static_cast<std::uint32_t>((base + 256) >> 8u), "back program address was not patched");
    check(frontLo != nullptr && frontLo->value == static_cast<std::uint32_t>(base >> 8u), "fused program address was not patched");
    check(find(fused, 0x08Au)->value == ((1u << 24u) | (2u << 6u) | 5u), "RSRC1 register counts were not merged");
    check(find(fused, 0x08Bu)->value == ((6u << 1u) | (1u << 7u) | (1u << 15u)), "RSRC2 user SGPRs were not taken from the front half");
    check(find(fused, 0x0B2u) != nullptr && find(fused, 0x0B2u)->value == 0x55u, "front-only SH register was lost");
    check(find(fused, 0x080u) == nullptr, "front program checksum was kept");
    check(fused.num_sh_registers == 7 && fused.num_cx_registers == 2, "fused register counts are wrong");
    check(fused.cx_registers[0].offset == 0x29Bu && fused.cx_registers[0].value == 2u && fused.cx_registers[1].offset == 0x2D5u, "context registers were not merged");
    check(fused.num_input_semantics == 2 && fused.input_semantics[1].semantic == 2, "front inputs were not kept");
    check(fused.num_output_semantics == 1 && fused.output_semantics[0].semantic == 3, "back outputs were not kept");
    check(fused.scratch_size_dw_per_thread == 16, "scratch size is not the larger half's");
    check(fused.specials != nullptr && fused.specials != halves.back.specials && fused.specials->ge_cntl.value == 0x77u, "back specials were not copied");
    const auto inside = [&](std::uintptr_t address) {
        const auto begin = reinterpret_cast<std::uintptr_t>(storage.data());
        return address >= begin && address < begin + storage.size();
    };
    const auto program = [&](const void* pointer) {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        return !inside(address) && address >= base + fused.shader_size && address < base + fused.shader_size + fused.header_size + alignof(Shader);
    };
    check(!inside(base) && program(fused.cx_registers) && program(fused.specials) && program(fused.input_semantics) && program(fused.output_semantics), "fused tables are not beside the fused code");
    check(program(fused.user_data) && program(fused.user_data->direct_resource_offset) && program(fused.user_data->sharp_resource_offset[1]), "fused user data is not beside the fused code");
    check(fused.user_data->direct_resource_count == 2 && fused.user_data->direct_resource_offset[1] == 9u && fused.user_data->sharp_resource_offset[1][0].offset_dw == 12 && fused.user_data->sharp_resource_offset[0] == nullptr && fused.user_data->eud_size_dw == 4, "fused user data is wrong");
}

void testScratchReuse() {
    Halves first;
    Halves second;
    second.backCode[0] = VMovV0;
    SizeAlign size{};
    check(sceAgcGetFusedShaderSize_0080(&size, &first.front, &first.back) == 0, "fused size query failed");
    std::vector<std::uint8_t> scratch(size.m_size);
    Shader firstFused{};
    Shader secondFused{};
    check(sceAgcFuseShaderHalves_0200(&firstFused, &first.front, &first.back, scratch.data()) == 0, "first fusion failed");
    const auto firstCode = reinterpret_cast<std::uintptr_t>(firstFused.code);
    check(sceAgcFuseShaderHalves_0200(&secondFused, &second.front, &second.back, scratch.data()) == 0, "second fusion failed");
    const auto secondCode = reinterpret_cast<std::uintptr_t>(secondFused.code);
    check(firstCode != secondCode, "two fused programs share one code address");
    check(word(reinterpret_cast<const void*>(firstCode), 64) == SMovS0, "fusing into reused memory overwrote the first fused program");
    check(word(reinterpret_cast<const void*>(secondCode), 64) == VMovV0, "second fused program has the wrong back half");
    Shader again{};
    check(sceAgcFuseShaderHalves_0200(&again, &first.front, &first.back, scratch.data()) == 0, "repeated fusion failed");
    check(reinterpret_cast<std::uintptr_t>(again.code) == firstCode, "the same halves were fused into a second program");
    first.backCode[0] = VMovV0;
    Shader changed{};
    check(sceAgcFuseShaderHalves_0200(&changed, &first.front, &first.back, scratch.data()) == 0, "fusion of changed halves failed");
    check(reinterpret_cast<std::uintptr_t>(changed.code) != firstCode && word(reinterpret_cast<const void*>(firstCode), 64) == SMovS0, "changed halves reused the earlier fused program");
}

void testRejections() {
    Halves halves;
    SizeAlign size{};
    Shader fused{};
    expectFailure([&] { sceAgcGetFusedShaderSize_0080(nullptr, &halves.front, &halves.back); });
    expectFailure([&] { sceAgcFuseShaderHalves_0200(&fused, &halves.front, &halves.back, nullptr); });
    {
        Halves swapped;
        check(sceAgcGetFusedShaderSize_0080(&size, &swapped.back, &swapped.front) == InvalidShaderHalves, "swapped halves were accepted by the size query");
        check(sceAgcFuseShaderHalves_0200(&fused, &swapped.back, &swapped.front, nullptr) == InvalidShaderHalves, "swapped halves were fused");
    }
    {
        Halves stages;
        stages.backSpecials.vgt_shader_stages_en.value = 0;
        std::vector<std::uint8_t> storage(4096);
        check(sceAgcFuseShaderHalves_0200(&fused, &stages.front, &stages.back, storage.data()) == InvalidShaderHalves, "halves with different GS stages were fused");
    }
    {
        Halves noJump;
        noJump.frontCode[2] = SNop;
        expectFailure([&] { sceAgcGetFusedShaderSize_0080(&size, &noJump.front, &noJump.back); });
    }
    {
        Halves noTrailer;
        noTrailer.frontCode[5] = 0;
        expectFailure([&] { sceAgcGetFusedShaderSize_0080(&size, &noTrailer.front, &noTrailer.back); });
    }
    {
        Halves hardware;
        hardware.frontSh[2].value |= 1u << 16u;
        expectFailure([&] { sceAgcGetFusedShaderSize_0080(&size, &hardware.front, &hardware.back); });
    }
    {
        Halves constants;
        constants.back.embedded_constant_buffer_size_dqw = 1;
        expectFailure([&] { sceAgcGetFusedShaderSize_0080(&size, &constants.front, &constants.back); });
    }
}

void testHullHalves() {
    std::array<std::uint32_t, 2> frontCode{SEndpgm, SCodeEnd};
    std::array<std::uint32_t, 2> backCode{SEndpgm, SCodeEnd};
    std::array<ShaderRegister, 4> frontSh{{{0x100u, 0x11u}, {0x100u, 0x22u}, {0x10Au, 7u}, {0x10Bu, 6u << 1u}}};
    std::array<ShaderRegister, 6> backSh{{{0x100u, 0u}, {0x100u, 0u}, {0x10Au, 3u}, {0x10Bu, 2u << 1u}, {0x148u, 0u}, {0x149u, 0xab00u}}};
    ShaderSpecialRegs specials{};
    specials.vgt_shader_stages_en.value = HsStage;
    Shader front{};
    front.code = frontCode.data();
    front.sh_registers = frontSh.data();
    front.num_sh_registers = static_cast<std::uint8_t>(frontSh.size());
    front.specials = &specials;
    front.type = static_cast<std::uint8_t>(ShaderRegs::ShaderBinaryType::HsFront);
    Shader back{};
    back.code = backCode.data();
    back.sh_registers = backSh.data();
    back.num_sh_registers = static_cast<std::uint8_t>(backSh.size());
    back.specials = &specials;
    back.type = static_cast<std::uint8_t>(ShaderRegs::ShaderBinaryType::HsBack);
    SizeAlign size{};
    check(sceAgcGetFusedShaderSize_0080(&size, &front, &back) == 0 && size.m_size == backSh.size() * sizeof(ShaderRegister) && size.m_align == 4, "hull size query changed");
    std::vector<ShaderRegister> scratch(backSh.size());
    Shader fused{};
    check(sceAgcFuseShaderHalves_0200(&fused, &front, &back, scratch.data()) == 0, "hull fusion failed");
    check(fused.type == static_cast<std::uint8_t>(ShaderRegs::ShaderBinaryType::Hs) && fused.code == back.code && fused.sh_registers == scratch.data() && fused.user_data == nullptr, "hull halves were not fused at the register level");
    const auto address = reinterpret_cast<std::uint64_t>(front.code);
    check(scratch[0].value == 0x11u && scratch[1].value == 0x22u, "hull checksums were not taken from the front half");
    check(scratch[2].value == 7u && scratch[3].value == (6u << 1u), "hull resource registers were not merged");
    check(scratch[4].value == static_cast<std::uint32_t>(address >> 8u) && scratch[5].value == (0xab00u | static_cast<std::uint32_t>((address >> 40u) & 0xffu)), "hull local program address does not point at the front half");
    check(backSh[4].value == 0u, "hull fusion wrote the back half's registers");
}

// Copies a shader graph into guest memory at `guest` (written through its host view), rebasing every table pointer.
class Relocation {
public:
    explicit Relocation(std::uintptr_t guest) : guest_(guest), host_(GuestArena::HostAddress(guest)) {}
    template <typename T>
    T* Copy(const T* source, std::size_t bytes, std::size_t alignment = alignof(T)) {
        if (source == nullptr) return nullptr;
        cursor_ = (cursor_ + alignment - 1) & ~(alignment - 1);
        std::memcpy(reinterpret_cast<void*>(host_ + cursor_), static_cast<const void*>(source), bytes);
        auto* placed = reinterpret_cast<T*>(guest_ + cursor_);
        cursor_ += bytes;
        return placed;
    }
    Shader* Place(const Shader& shader) {
        Shader moved = shader;
        moved.code = Copy(static_cast<const std::uint8_t*>(const_cast<const void*>(shader.code)), shader.shader_size, 256);
        moved.sh_registers = Copy(shader.sh_registers, shader.num_sh_registers * sizeof(ShaderRegister));
        moved.cx_registers = Copy(shader.cx_registers, shader.num_cx_registers * sizeof(ShaderRegister));
        moved.specials = Copy(shader.specials, sizeof(ShaderSpecialRegs));
        moved.input_semantics = Copy(shader.input_semantics, shader.num_input_semantics * sizeof(ShaderSemantic));
        moved.output_semantics = Copy(shader.output_semantics, shader.num_output_semantics * sizeof(ShaderSemantic));
        if (shader.user_data != nullptr) {
            ShaderUserData userData = *shader.user_data;
            userData.direct_resource_offset = Copy(userData.direct_resource_offset, userData.direct_resource_count * sizeof(std::uint16_t));
            for (std::size_t i = 0; i < 4; ++i) userData.sharp_resource_offset[i] = Copy(userData.sharp_resource_offset[i], userData.sharp_resource_count[i] * sizeof(ShaderSharp));
            moved.user_data = Copy(&userData, sizeof(userData));
        }
        return Copy(&moved, sizeof(moved), 16);
    }
    void* Reserve(std::size_t bytes, std::size_t alignment) {
        cursor_ = (cursor_ + alignment - 1) & ~(alignment - 1);
        auto* placed = reinterpret_cast<void*>(guest_ + cursor_);
        cursor_ += bytes;
        return placed;
    }
    template <typename T>
    T* Host(T* guest) const { return reinterpret_cast<T*>(GuestArena::HostAddress(reinterpret_cast<std::uintptr_t>(guest))); }

private:
    std::uintptr_t guest_;
    std::uintptr_t host_;
    std::size_t cursor_ = 0;
};

// Halves, scratch and the fused result in displaced guest memory (D31, D33; ASTRO BOT's shaders live at 0x5xxxxxxxx):
// libSceAgc and the driver's registration read and write them through host views, so fusing and registering takes
// no displaced fault, and the fused shader is the one ordinary memory gives (at its own address).
void testDisplacedFusion() {
#if defined(__APPLE__) && defined(__aarch64__)
    constexpr std::uintptr_t Base = 0x400000000ull;
    constexpr std::size_t Bytes = 0x40000;
    check(GuestArena::GuestDisplacedActive_nid_no_patch() && GuestArena::IsDisplaced(Base), "0x400000000 is not displaced");
    GuestArena::GuestArenaCommit_nid_postfix(reinterpret_cast<void*>(Base), Bytes, 0x04, 0x4000);
    GuestAllocations::Mutation().Add(reinterpret_cast<void*>(Base), Bytes, true, true, true);
    Halves halves;
    Relocation memory(Base);
    auto* front = memory.Place(halves.front);
    auto* back = memory.Place(halves.back);
    const auto faults = [] {
        std::uint64_t count = 0, patches = 0;
        GuestArena::GuestDisplacedStats_nid_no_patch(&count, &patches);
        return count;
    };
    const auto before = faults();
    SizeAlign size{};
    check(sceAgcGetFusedShaderSize_0080(&size, front, back) == 0, "displaced fused size query failed");
    auto* scratch = static_cast<std::uint8_t*>(memory.Reserve(size.m_size, 256));
    Shader fused{};
    check(sceAgcFuseShaderHalves_0200(&fused, front, back, scratch) == 0, "displaced fusion failed");
    const auto taken = faults() - before;
    if (taken != 0) throw std::runtime_error("displaced fusion took " + std::to_string(taken) + " displaced faults (a native read through the guest address)");
    // Upstream 0d078f75d3: the fused program (code, tables, user data) lives in host memory libSceAgc owns, which is
    // never displaced; only the fused register list is written to the caller's (displaced) scratch memory.
    const auto code = reinterpret_cast<std::uintptr_t>(fused.code);
    check(!GuestArena::IsDisplaced(code) && code % 256 == 0 && fused.shader_size == 256 + halves.back.shader_size, "displaced fused code placement is wrong");
    check(word(reinterpret_cast<const void*>(code), 0) == SMovS0 && word(reinterpret_cast<const void*>(code), 1) == VMovV0 && word(reinterpret_cast<const void*>(code), 64) == halves.backCode[0], "displaced fused program was not copied");
    check(reinterpret_cast<std::uintptr_t>(fused.sh_registers) == reinterpret_cast<std::uintptr_t>(scratch), "displaced fused registers are not at their guest address");
    check(fused.user_data != nullptr && !GuestArena::IsDisplaced(reinterpret_cast<std::uintptr_t>(fused.user_data)) && fused.user_data->direct_resource_offset[1] == 9u, "displaced fused user data is wrong");
    const auto* lo = memory.Host(fused.sh_registers);
    bool patched = false;
    for (std::size_t i = 0; i < fused.num_sh_registers; ++i) patched |= lo[i].offset == 0x0C8u && lo[i].value == static_cast<std::uint32_t>(code >> 8u);
    check(patched, "displaced fused program address is not the fused code address");
    GuestAllocations::Mutation().Remove(reinterpret_cast<void*>(Base));
#endif
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        testFusion(0);
        testFusion(0xa0);
        testFusion(4);
        testFusion(0, SSwappcNullS6);
        testScratchReuse();
        testRejections();
        testHullHalves();
        testDisplacedFusion();
        LibcRunShutdown_nid_postfix();
        std::puts("AGC shader fusion tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        try { LibcRunShutdown_nid_postfix(); }
        catch (const std::exception& shutdown) { std::fprintf(stderr, "shutdown: %s\n", shutdown.what()); }
        return 1;
    }
}
