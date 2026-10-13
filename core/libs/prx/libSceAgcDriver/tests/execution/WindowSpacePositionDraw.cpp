#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace {

using AgcDriver::Graphics::Require;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 32;
constexpr std::uint32_t SetContextReg = 0x69u;
constexpr std::uint32_t SetShReg = 0x76u;
constexpr std::uint32_t SetUconfigReg = 0x79u;
constexpr std::uint32_t DrawIndexAuto = 0x2du;

constexpr std::size_t Block = 65536;
constexpr std::size_t ColorBytes = Width * Height * 4;
std::byte* ColorMemory = nullptr;
alignas(64) volatile std::uint32_t EndOfPipe = 0;

std::byte* AllocateWatched(std::size_t bytes) {
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + Block, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + Block - 1) & ~static_cast<std::uintptr_t>(Block - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + Block) munmap(reinterpret_cast<void*>(aligned + bytes), begin + Block - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    return static_cast<std::byte*>(block);
}

// buffer_load_format_xyzw v[0:3] of the vertex (the position, from the buffer in user data), exported
// as pos0 alone (tests/execution/VertexClipCullDraw.cpp's program without its pos1 export).
alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80020005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

// White to mrt0.
alignas(256) constexpr std::array<std::uint32_t, 7> PixelCode{
    0x7e0002f2, 0x7e0202f2, 0x7e0402f2, 0x7e0602f2, 0xf800080f, 0x03020100, 0xbf810000,
};

// A right triangle in pixels: legs on x = 8 and y = 4, hypotenuse x + 2y = 64. No pixel centre lies on
// an edge, so coverage does not depend on the fill convention.
constexpr std::array<std::array<float, 2>, 3> Corners{{{8.0f, 4.0f}, {56.0f, 4.0f}, {8.0f, 28.0f}}};

struct Step {
    const char* name;
    float w;           // the position's w: window space never divides by it
    std::uint32_t clip; // PA_CL_CLIP_CNTL
};

// radeonsi sets CLIP_DISABLE (bit 16) for window-space shaders (si_emit_clip_regs).
const std::array<Step, 3> Steps{{
    {"w = 1", 1.0f, 0x80000u},
    {"w = 2", 2.0f, 0x80000u},
    {"w = 0.5, clipping disabled", 0.5f, 0x90000u},
}};

struct RegisteredShader {
    Shader shader;
    ShaderUserData userData;
};
RegisteredShader VertexShader{};
RegisteredShader PixelShader{};

// tests/execution/VertexClipCullDraw.cpp's baseline with PA_CL_VTE_CNTL 0x300 (VTX_XY_FMT | VTX_Z_FMT,
// no viewport transform) and a viewport transform that would put the triangle elsewhere if applied.
const std::vector<std::pair<std::uint32_t, std::uint32_t>> ContextBaseline{
    {0x2d5, 0x2000},
    {0x1b6, 0}, {0x200, 0}, {0x203, 0x800}, {0x207, 0},
    {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
    {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
    {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
    {0x206, 0x300}, {0x205, 0x240},
    {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
    {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
    {0x31b, 0}, {0x31d, 0}, {0x3b8, 0x9000000}, {0x1e0, 0},
    {0xc, 0},
    {0x81, 0x80000000}, {0x90, 0x80000000}, {0x94, 0x80000000},
    {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
    {0x10f, std::bit_cast<std::uint32_t>(3.0f)}, {0x110, std::bit_cast<std::uint32_t>(-20.0f)},
    {0x111, std::bit_cast<std::uint32_t>(-5.0f)}, {0x112, std::bit_cast<std::uint32_t>(7.0f)},
    {0x113, std::bit_cast<std::uint32_t>(0.5f)}, {0x114, std::bit_cast<std::uint32_t>(0.5f)},
};

void Register(RegisteredShader& registered, const void* code, std::uint32_t bytes, std::uint8_t type) {
    registered.shader.file_header = 0x34333231u;
    registered.shader.version = 0x18u;
    registered.shader.user_data = &registered.userData;
    registered.shader.code = code;
    registered.shader.header_size = sizeof(RegisteredShader);
    registered.shader.shader_size = bytes;
    registered.shader.type = type;
    AgcDriverRegisterShader_nid_postfix(&registered.shader);
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, 0x01016facu};
}

void SetRegister(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::uint32_t offset, std::uint32_t value) {
    words.push_back(0xc0010000u | (opcode << 8u));
    words.push_back(offset);
    words.push_back(value);
}

std::vector<std::uint32_t> Frame(const Step& step, const std::array<std::array<float, 4>, 3>& vertices) {
    const auto colorAddress = reinterpret_cast<std::uintptr_t>(ColorMemory);
    const auto vertexAddress = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    const auto pixelAddress = reinterpret_cast<std::uintptr_t>(PixelCode.data());
    std::vector<std::uint32_t> words;
    for (const auto& [offset, value] : ContextBaseline) SetRegister(words, SetContextReg, offset, value);
    SetRegister(words, SetContextReg, 0x204, step.clip);
    SetRegister(words, SetContextReg, 0x318, static_cast<std::uint32_t>(colorAddress >> 8u));
    SetRegister(words, SetContextReg, 0x390, static_cast<std::uint32_t>(colorAddress >> 40u));
    SetRegister(words, SetContextReg, 0x3b0, ((Width - 1u) << 14u) | (Height - 1u));
    const auto extent = (Height << 16u) | Width;
    SetRegister(words, SetContextReg, 0xd, extent);
    SetRegister(words, SetContextReg, 0x82, extent);
    SetRegister(words, SetContextReg, 0x91, extent);
    SetRegister(words, SetContextReg, 0x95, extent);
    SetRegister(words, SetUconfigReg, 0x242, 4u);
    SetRegister(words, SetShReg, 0x008, static_cast<std::uint32_t>(pixelAddress >> 8u));
    SetRegister(words, SetShReg, 0x009, static_cast<std::uint32_t>(pixelAddress >> 40u));
    SetRegister(words, SetShReg, 0x00b, 0u);
    SetRegister(words, SetShReg, 0x0c8, static_cast<std::uint32_t>(vertexAddress >> 8u));
    SetRegister(words, SetShReg, 0x0c9, static_cast<std::uint32_t>(vertexAddress >> 40u));
    SetRegister(words, SetShReg, 0x08b, 4u << 1u);
    const auto descriptor = BufferDescriptor(vertices.data(), sizeof(vertices[0]), static_cast<std::uint32_t>(vertices.size()));
    for (std::uint32_t i = 0; i < descriptor.size(); ++i) SetRegister(words, SetShReg, 0x08c + i, descriptor[i]);
    words.insert(words.end(), {0xc0000000u | (1u << 16u) | (DrawIndexAuto << 8u), static_cast<std::uint32_t>(vertices.size()), 2u});
    const auto labelAddress = reinterpret_cast<std::uintptr_t>(&EndOfPipe);
    words.insert(words.end(), {0xc0064900u, 0x514u, (1u << 29u) | (2u << 24u), static_cast<std::uint32_t>(labelAddress), static_cast<std::uint32_t>(static_cast<std::uint64_t>(labelAddress) >> 32u), 1u, 0u, 0u});
    return words;
}

void Check(const Step& step, const std::vector<std::byte>& colors) {
    const auto what = std::string("window-space positions, ") + step.name;
    std::uint32_t covered = 0;
    for (std::uint32_t pixel = 0; pixel < Width * Height; ++pixel) {
        const auto value = std::to_integer<std::uint8_t>(colors[pixel * 4u]);
        Require(value == 0u || value == 255u, what + ": pixel " + std::to_string(pixel) + " stored " + std::to_string(value));
        const double x = (pixel % Width) + 0.5;
        const double y = (pixel / Width) + 0.5;
        const bool expected = x > 8.0 && y > 4.0 && x + 2.0 * y < 64.0;
        const bool written = value == 255u;
        covered += written ? 1u : 0u;
        Require(written == expected, what + ": pixel (" + std::to_string(pixel % Width) + ", " + std::to_string(pixel / Width) + ") is " + (written ? "written" : "clear") + ", expected " + (expected ? "written" : "clear"));
    }
    Require(covered > 500u, what + ": the triangle covered only " + std::to_string(covered) + " pixels");
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        ColorMemory = AllocateWatched(Block);
        Register(VertexShader, VertexCode.data(), sizeof(VertexCode), 2);
        Register(PixelShader, PixelCode.data(), sizeof(PixelCode), 1);
        for (const auto& step : Steps) {
            alignas(16) std::array<std::array<float, 4>, 3> vertices{};
            for (std::size_t i = 0; i < vertices.size(); ++i) vertices[i] = {Corners[i][0], Corners[i][1], 0.5f, step.w};
            std::fill_n(ColorMemory, ColorBytes, std::byte{0});
            EndOfPipe = 0;
            auto words = Frame(step, vertices);
            Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
            Require(sceAgcDriverSubmitDcb(&packet) == 0, std::string("DCB submission failed for ") + step.name);
            AgcDriverWaitIdle_nid_postfix();
            Require(EndOfPipe == 1u, std::string("the end-of-pipe label was not written for ") + step.name);
            std::vector<std::byte> colors(ColorBytes);
            AgcDriver::GuestMemory::Read(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(ColorMemory)), colors);
            Check(step, colors);
        }
        AgcDriverShutdown_nid_postfix();
        std::puts("window-space position draws passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
