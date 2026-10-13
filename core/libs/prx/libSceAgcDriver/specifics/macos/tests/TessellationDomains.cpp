// agc_metal_tessellation_domains (DECISIONS D26): guest tessellated draws of every domain and of
// the partitionings and output topologies the Metal target opened beyond the reference path's
// triangle / fractional_odd / triangle_cw set, checked against the software tessellator core
// (MslTessellator, bit-exact with Microsoft's D3D11 reference in agc_metal_tessellator).
//
// One patch per draw, one control point per patch. The hull program writes literal factors to the
// factor ring (isoline records are (detail, density) as on RDNA2); the domain program appends
// (u, v, patch) to a record buffer through a buffer atomic, places (u, v) on a square of the render
// target and exports white-red. Per case:
// - the domain program's invocations are exactly the unique domain points of each mesh slice (the
//   vertex sharing of the mesh function), with the tessellator's (u, v) bit for bit;
// - triangle outputs cover the domain's square or triangle with the winding the topology names
//   (drawn under one front face, culled under the other); line and point outputs light the
//   expected rows and pixels; a culled patch runs no domain program and draws nothing;
// - VGT_HOS level clamps change the tessellation as the CPU expectation clamps it.
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "MetalBackend/MslTessellator.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderStage = ShaderRecompiler::ShaderStage;
namespace T = ShaderRecompiler::MslTessellator;
constexpr ShaderRecompiler::TessellationFeatures Everything{0x7u, 0xfu, 0xfu, true};

constexpr std::uint32_t Size = 128;
constexpr std::array<std::uint8_t, 4> Background{16, 24, 40, 255};
alignas(256) std::array<std::byte, Size * Size * 4> Pixels{};
constexpr std::uint32_t RecordCapacity = 4096;
alignas(256) std::array<std::uint32_t, 4 + RecordCapacity * 4> Records{};

// tests/execution/Tessellation.cpp's local program (control point data into LDS, then s_swappc).
alignas(256) constexpr std::array<std::uint32_t, 29> LocalCode{
    0xd5480004, 0x02090502, 0xd5480005, 0x02090102, 0xd5480006, 0x02110902, 0xd5480007, 0x02111102,
    0x7e080d04, 0x7e0a0d05, 0x7e0c0d06, 0x7e0e0d07, 0x10100cff, 0x3f19999a, 0x061010ff, 0xbf666666,
    0x3e1008ff, 0x3e800000, 0xd54b0009, 0x03c5e105, 0x161406a0, 0xd9340000, 0x0000080a, 0xd8340008,
    0x0000070a, 0xd8340010, 0x0000070a, 0xbf8cc07f, 0xbefd2106,
};

// Hull programs (llvm-mc -mcpu=gfx1030): v20 = invocation, v18 = relative patch; exec = invocation 0;
// v24..v29 = the literal factors (placeholders 0xf000000n); v10 = patch * record bytes;
// buffer_store_dwordx4/x2 to the factor ring (s[12:15], soffset s4).
alignas(256) constexpr std::array<std::uint32_t, 21> HullTriangle{
    0xd5480014, 0x02151101, 0x362402ff, 0x000000ff, 0x7da82881, 0x7e3002ff, 0xf0000000, 0x7e3202ff, 0xf0000001, 0x7e3402ff, 0xf0000002,
    0x7e3602ff, 0xf0000003, 0x7e3802ff, 0xf0000004, 0x7e3a02ff, 0xf0000005, 0x16142490, 0xe0781000, 0x0403180a, 0xbf810000,
};
alignas(256) constexpr std::array<std::uint32_t, 23> HullQuad{
    0xd5480014, 0x02151101, 0x362402ff, 0x000000ff, 0x7da82881, 0x7e3002ff, 0xf0000000, 0x7e3202ff, 0xf0000001, 0x7e3402ff, 0xf0000002,
    0x7e3602ff, 0xf0000003, 0x7e3802ff, 0xf0000004, 0x7e3a02ff, 0xf0000005, 0x16142498, 0xe0781000, 0x0403180a, 0xe0741010, 0x04031c0a,
    0xbf810000,
};
alignas(256) constexpr std::array<std::uint32_t, 21> HullIsoline{
    0xd5480014, 0x02151101, 0x362402ff, 0x000000ff, 0x7da82881, 0x7e3002ff, 0xf0000000, 0x7e3202ff, 0xf0000001, 0x7e3402ff, 0xf0000002,
    0x7e3602ff, 0xf0000003, 0x7e3802ff, 0xf0000004, 0x7e3a02ff, 0xf0000005, 0x16142488, 0xe0741000, 0x0403180a, 0xbf810000,
};

// Domain program: exp prim; v1 = buffer_atomic_add(s[8:11] + 0, 1) glc; store (u, v, primitive id, 0)
// at 16 + 16 * v1; position (1.5 u - 0.77, 1.5 v - 0.77, 0, 1); param0 = (1, 0, 0, 1).
alignas(256) constexpr std::array<std::uint32_t, 28> DomainCode{
    0xf8000941, 0x00000000, 0x7e020281, 0xe0c84000, 0x80020100, 0x7e140305, 0x7e160306, 0x7e180308, 0x7e1a0280, 0xbf8c3f70,
    0x34040284, 0xe0781010, 0x80020a02, 0x10200aff, 0x3fc00000, 0x062020ff, 0xbf451eb8, 0x10220cff, 0x3fc00000, 0x062222ff,
    0xbf451eb8, 0x7e260280, 0x7e2802f2, 0xf80008cf, 0x14131110, 0xf800020f, 0x14131314, 0xbf810000,
};
constexpr float Scale = 1.5f;
constexpr float Offset = -0.77f;

alignas(256) constexpr std::array<std::uint32_t, 7> PixelCode{
    0xc8020002, 0xc8060102, 0xc80a0202, 0xc80e0302, 0xf800180f, 0x03020100, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint16_t, 1> Indices{0};

struct Case {
    const char* name;
    std::uint32_t domain;
    std::uint32_t partitioning;
    std::uint32_t topology;
    std::array<float, 6> ring; // as the hull program writes the factor ring record
    float minLevel = 0.0f;
    float maxLevel = 64.0f;
};

// The case's hull program: its template with the placeholder literals replaced by the factors.
alignas(256) std::array<std::uint32_t, 32> HullBuffer{};

std::span<const std::uint32_t> HullFor(const Case& test) {
    const std::span<const std::uint32_t> source = test.domain == T::Isoline ? std::span<const std::uint32_t>(HullIsoline) : test.domain == T::Triangle ? std::span<const std::uint32_t>(HullTriangle) : std::span<const std::uint32_t>(HullQuad);
    for (std::size_t i = 0; i < source.size(); ++i) {
        const bool literal = i > 0 && (source[i - 1] & 0xfe01ffffu) == 0x7e0002ffu && (source[i] & 0xfffffff0u) == 0xf0000000u;
        HullBuffer[i] = literal ? std::bit_cast<std::uint32_t>(test.ring[source[i] & 0xfu]) : source[i];
    }
    return std::span<const std::uint32_t>(HullBuffer.data(), source.size());
}

ShaderRecompiler::RecompileResult Compile(const ShaderRecompiler::SpirvTarget& target, const ShaderRecompiler::TessellationConfiguration& config, ShaderStage stage, std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, std::uint32_t& pushCursor) {
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const bool hull = stage == ShaderStage::TessellationControl;
    std::optional<ShaderRecompiler::ShaderPixelStageInfo> pixel;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex;
    std::optional<ShaderRecompiler::GraphicsCompileContext> graphics;
    if (stage == ShaderStage::Fragment) {
        ShaderRecompiler::ShaderPixelStageInfo info{};
        info.interpolatorCount = 1;
        info.interpolatorSettings[0] = 0x400u;
        info.targetOutputMode[0] = 9;
        info.targetExportMapping.fill(0xe4u);
        pixel = info;
    } else {
        vertex = ShaderRecompiler::ShaderVertexStageInfo{};
        graphics = ShaderRecompiler::GraphicsCompileContext{hull ? 0u : 8u, {}, std::nullopt, config, {reinterpret_cast<std::uintptr_t>(Indices.data()), static_cast<std::uint32_t>(Indices.size()), 2, 1}};
    }
    ShaderRecompiler::RecompileRequest request{
        {stage, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, hull ? 0u : 8u, userData, std::nullopt, pixel, vertex, memory},
        target,
        stage == ShaderStage::Fragment ? PixelPushLayout(pushCursor, target) : ShaderRecompiler::BindingLayout{0, 0, pushCursor, AgcDriver::Graphics::PipelinePushSlotBytes - pushCursor},
        graphics
    };
    request.useCache = false;
    auto result = ShaderRecompiler::Recompile(request);
    pushCursor += static_cast<std::uint32_t>(result.pushConstants.size());
    return result;
}

std::array<std::uint8_t, 4> PixelAt(std::uint32_t x, std::uint32_t y) {
    const auto offset = (static_cast<std::size_t>(y) * Size + x) * 4u;
    return {std::to_integer<std::uint8_t>(Pixels[offset]), std::to_integer<std::uint8_t>(Pixels[offset + 1]), std::to_integer<std::uint8_t>(Pixels[offset + 2]), std::to_integer<std::uint8_t>(Pixels[offset + 3])};
}

bool Lit(std::uint32_t x, std::uint32_t y) {
    const auto pixel = PixelAt(x, y);
    if (pixel == Background) return false;
    Require(pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0 && pixel[3] == 255, "pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") has an unexpected colour");
    return true;
}

// Domain (u, v) of a pixel centre, through the domain program's placement and the flipped viewport.
std::pair<float, float> DomainAt(std::uint32_t x, std::uint32_t y) {
    const float nx = (static_cast<float>(x) + 0.5f) / Size * 2.0f - 1.0f;
    const float ny = 1.0f - (static_cast<float>(y) + 0.5f) / Size * 2.0f;
    return {(nx - Offset) / Scale, (ny - Offset) / Scale};
}

std::pair<float, float> PixelOf(float u, float v) {
    return {((u * Scale + Offset) + 1.0f) * 0.5f * Size, (1.0f - (v * Scale + Offset)) * 0.5f * Size};
}

struct Expectation {
    T::Aps5TessPatch patch{};
    int primitives = 0;
    std::uint32_t corners = 0;               // domain program runs if every corner owned its vertex
    std::vector<std::pair<std::uint32_t, std::uint32_t>> points; // one per domain program run
    std::vector<std::array<int, 3>> triangles;
};

Expectation Expect(const Case& test) {
    const int topology = static_cast<int>(test.topology);
    T::Aps5TessFactors factors{};
    // The tessellator's isoline order is (density, detail), the ring's (detail, density).
    if (test.domain == T::Isoline) factors = {{test.ring[1], test.ring[0], 0.0f, 0.0f}, {0.0f, 0.0f}};
    else if (test.domain == T::Triangle) factors = {{test.ring[0], test.ring[1], test.ring[2], 0.0f}, {test.ring[3], 0.0f}};
    else factors = {{test.ring[0], test.ring[1], test.ring[2], test.ring[3]}, {test.ring[4], test.ring[5]}};
    Expectation result;
    result.patch = T::aps5_tess_process(static_cast<int>(test.domain), static_cast<int>(test.partitioning), topology, factors);
    if (result.patch.culled == 0) {
        const auto clamp = [&](float f) { return std::isnan(f) ? f : std::fmin(std::fmax(f, test.minLevel), test.maxLevel); };
        for (auto& f : factors.outer) f = clamp(f);
        for (auto& f : factors.inner) f = clamp(f);
        result.patch = T::aps5_tess_process(static_cast<int>(test.domain), static_cast<int>(test.partitioning), topology, factors);
    }
    result.primitives = T::aps5_tess_primitive_count(result.patch, nullptr);
    const int slice = T::SlicePrimitives(topology);
    const int corners = topology == T::Point ? 1 : topology == T::Line ? 2 : 3;
    std::vector<int> window(static_cast<std::size_t>(slice * corners));
    for (int first = 0; first < result.primitives; first += slice) {
        const int count = T::aps5_tess_window(result.patch, first, std::min(slice, result.primitives - first), window.data());
        std::set<int> unique;
        for (int c = 0; c < count * corners; ++c) {
            ++result.corners;
            if (!unique.insert(window[static_cast<std::size_t>(c)]).second) continue;
            const auto point = T::aps5_tess_point(result.patch, window[static_cast<std::size_t>(c)]);
            result.points.emplace_back(std::bit_cast<std::uint32_t>(point.u), std::bit_cast<std::uint32_t>(point.v));
        }
        if (corners == 3) {
            for (int k = 0; k < count; ++k) result.triangles.push_back({window[static_cast<std::size_t>(k * 3)], window[static_cast<std::size_t>(k * 3 + 1)], window[static_cast<std::size_t>(k * 3 + 2)]});
        }
    }
    return result;
}

// +1 when every triangle is counter-clockwise in NDC (y up), -1 when every one is clockwise.
int Winding(const Expectation& expected) {
    int sign = 0;
    for (const auto& triangle : expected.triangles) {
        std::array<std::pair<float, float>, 3> p;
        for (int i = 0; i < 3; ++i) {
            const auto point = T::aps5_tess_point(expected.patch, triangle[static_cast<std::size_t>(i)]);
            p[static_cast<std::size_t>(i)] = {point.u * Scale + Offset, point.v * Scale + Offset};
        }
        const float area = (p[1].first - p[0].first) * (p[2].second - p[0].second) - (p[2].first - p[0].first) * (p[1].second - p[0].second);
        if (area == 0.0f) continue;
        const int here = area > 0.0f ? 1 : -1;
        Require(sign == 0 || sign == here, "the tessellator's triangles mix windings");
        sign = here;
    }
    return sign;
}

void Draw(AgcDriver::VulkanDevice& device, const Case& test, VkCullModeFlags cull, VkFrontFace front) {
    for (std::size_t i = 0; i < Pixels.size(); i += 4) {
        for (std::size_t c = 0; c < 4; ++c) Pixels[i + c] = std::byte{Background[c]};
    }
    Records.fill(0u);
    const auto target = device.Target();
    const ShaderRecompiler::TessellationConfiguration config{1u, 1u, test.domain, test.partitioning, test.topology};
    const auto address = reinterpret_cast<std::uintptr_t>(Records.data());
    const std::array<std::uint32_t, 8> domainUserData{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(Records.size() * 4u), 0x31016facu, 0, 0, 0, 0};
    const std::array<std::uint32_t, 16> hullUserData{};
    std::uint32_t cursor = 0;
    const auto local = Compile(target, config, ShaderStage::Local, LocalCode, {}, cursor);
    const auto hull = Compile(target, config, ShaderStage::TessellationControl, HullFor(test), hullUserData, cursor);
    const auto domain = Compile(target, config, ShaderStage::TessellationEvaluation, DomainCode, domainUserData, cursor);
    const auto fragment = Compile(target, config, ShaderStage::Fragment, PixelCode, {}, cursor);
    std::uint32_t push = 0;
    const auto offset = [&](const ShaderRecompiler::RecompileResult& result) {
        const auto at = result.pushConstants.empty() ? 0u : push;
        push += static_cast<std::uint32_t>(result.pushConstants.size());
        return at;
    };
    const std::array<AgcDriver::Graphics::CompiledShader, 4> shaders{{
        {ShaderStage::Local, &local, offset(local)},
        {ShaderStage::TessellationControl, &hull, offset(hull)},
        {ShaderStage::TessellationEvaluation, &domain, offset(domain)},
        {ShaderStage::Fragment, &fragment, PixelPushOffset(push, target)},
    }};
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Tessellation, 0x0200210du, 64, 64, std::nullopt, config, test.minLevel, test.maxLevel};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Size, Size}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {Size, Size};
    state.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    state.viewport = {0, static_cast<float>(Size), static_cast<float>(Size), -static_cast<float>(Size), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Size, Size}};
    state.cullMode = cull;
    state.frontFace = front;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    device.Draw(state, {reinterpret_cast<std::uintptr_t>(Indices.data()), static_cast<std::uint32_t>(Indices.size()), 2, 1, 0, true}, shaders);
    device.WaitIdle();
}

void RequireRecords(const Case& test, const Expectation& expected) {
    const std::string name = test.name;
    const auto runs = Records[0];
    Require(runs == expected.points.size(), name + ": the domain program ran " + std::to_string(runs) + " times, expected " + std::to_string(expected.points.size()) + " unique slice points (" + std::to_string(expected.corners) + " corners)");
    std::vector<std::pair<std::uint32_t, std::uint32_t>> seen;
    for (std::uint32_t i = 0; i < runs; ++i) {
        const auto* record = &Records[4u + i * 4u];
        Require(record[2] == 0u, name + ": a domain record names another patch");
        seen.emplace_back(record[0], record[1]);
    }
    auto wanted = expected.points;
    std::sort(seen.begin(), seen.end());
    std::sort(wanted.begin(), wanted.end());
    if (seen != wanted) {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> extra;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> missing;
        std::set_difference(seen.begin(), seen.end(), wanted.begin(), wanted.end(), std::back_inserter(extra));
        std::set_difference(wanted.begin(), wanted.end(), seen.begin(), seen.end(), std::back_inserter(missing));
        std::string detail;
        for (std::size_t i = 0; i < std::min<std::size_t>(extra.size(), 4); ++i) detail += " ran (" + std::to_string(std::bit_cast<float>(extra[i].first)) + ", " + std::to_string(std::bit_cast<float>(extra[i].second)) + ")";
        for (std::size_t i = 0; i < std::min<std::size_t>(missing.size(), 4); ++i) detail += " expected (" + std::to_string(std::bit_cast<float>(missing[i].first)) + ", " + std::to_string(std::bit_cast<float>(missing[i].second)) + ")";
        for (std::uint32_t i = 0; i < runs; ++i) {
            const auto* record = &Records[4u + i * 4u];
            if (std::find(extra.begin(), extra.end(), std::pair{record[0], record[1]}) != extra.end()) detail += " [record " + std::to_string(i) + " of " + std::to_string(runs) + "]";
        }
        for (std::size_t i = 1; i < seen.size(); ++i) {
            if (seen[i] == seen[i - 1]) detail += " duplicate (" + std::to_string(std::bit_cast<float>(seen[i].first)) + ", " + std::to_string(std::bit_cast<float>(seen[i].second)) + ")";
        }
        Require(false, name + ": the domain locations differ from the tessellator's (" + std::to_string(extra.size()) + " unexpected, " + std::to_string(missing.size()) + " missing):" + detail);
    }
}

std::uint32_t LitPixels() {
    std::uint32_t count = 0;
    for (std::uint32_t y = 0; y < Size; ++y) {
        for (std::uint32_t x = 0; x < Size; ++x) count += Lit(x, y) ? 1u : 0u;
    }
    return count;
}

// Triangle outputs: the domain's region is covered (inside by a margin) and nothing outside it.
void RequireCoverage(const Case& test, bool drawn) {
    const std::string name = test.name;
    for (std::uint32_t y = 0; y < Size; ++y) {
        for (std::uint32_t x = 0; x < Size; ++x) {
            const auto [u, v] = DomainAt(x, y);
            constexpr float Margin = 0.03f;
            const bool inside = test.domain == T::Triangle ? u > Margin && v > Margin && u + v < 1.0f - Margin : u > Margin && v > Margin && u < 1.0f - Margin && v < 1.0f - Margin;
            const bool outside = test.domain == T::Triangle ? u < -Margin || v < -Margin || u + v > 1.0f + Margin : u < -Margin || v < -Margin || u > 1.0f + Margin || v > 1.0f + Margin;
            if (outside) Require(!Lit(x, y), name + ": pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") outside the domain is drawn");
            if (inside) Require(Lit(x, y) == drawn, name + ": pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") inside the domain is " + (drawn ? "not drawn" : "drawn under the culled winding"));
        }
    }
}

void RequirePoints(const Case& test, const Expectation& expected) {
    // Each domain point lights one pixel (1-pixel points) at its place; nothing else is drawn.
    std::set<std::pair<std::uint32_t, std::uint32_t>> places;
    for (const auto& [ub, vb] : expected.points) {
        const auto [fx, fy] = PixelOf(std::bit_cast<float>(ub), std::bit_cast<float>(vb));
        bool found = false;
        for (int dy = -1; dy <= 1 && !found; ++dy) {
            for (int dx = -1; dx <= 1 && !found; ++dx) {
                const int x = static_cast<int>(std::floor(fx)) + dx;
                const int y = static_cast<int>(std::floor(fy)) + dy;
                if (x >= 0 && y >= 0 && x < static_cast<int>(Size) && y < static_cast<int>(Size) && Lit(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y))) found = true;
            }
        }
        Require(found, std::string(test.name) + ": no pixel at a domain point");
    }
    const auto lit = LitPixels();
    Require(lit >= expected.points.size() && lit <= 2u * expected.points.size(), std::string(test.name) + ": " + std::to_string(lit) + " pixels lit for " + std::to_string(expected.points.size()) + " points");
}

void RequireLines(const Case& test, const Expectation& expected) {
    // Isolines run along u at v = line / density: the middle column crosses each line once.
    const auto column = static_cast<std::uint32_t>(PixelOf(0.5f, 0.0f).first);
    std::uint32_t runs = 0;
    bool previous = false;
    for (std::uint32_t y = 0; y < Size; ++y) {
        const bool here = Lit(column, y);
        if (here && !previous) ++runs;
        previous = here;
    }
    Require(static_cast<int>(runs) == expected.patch.lines, std::string(test.name) + ": the middle column crosses " + std::to_string(runs) + " lines, expected " + std::to_string(expected.patch.lines));
}

void Run(AgcDriver::VulkanDevice& device, const Case& test) {
    const auto expected = Expect(test);
    std::printf("%s: %d primitives, domain program runs %zu (corners %u)\n", test.name, expected.primitives, expected.points.size(), expected.corners);
    if (test.topology >= T::TriangleCw) {
        const int winding = Winding(expected);
        Require(winding != 0 || expected.primitives == 0, std::string(test.name) + ": no triangle has an area");
        for (const auto front : {VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FRONT_FACE_CLOCKWISE}) {
            Draw(device, test, VK_CULL_MODE_BACK_BIT, front);
            RequireRecords(test, expected);
            const bool drawn = expected.primitives != 0 && (front == VK_FRONT_FACE_COUNTER_CLOCKWISE) == (winding > 0);
            RequireCoverage(test, drawn);
        }
        return;
    }
    Draw(device, test, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    RequireRecords(test, expected);
    if (expected.primitives == 0) Require(LitPixels() == 0u, std::string(test.name) + ": a culled patch drew pixels");
    else if (test.topology == T::Point) RequirePoints(test, expected);
    else if (test.domain == T::Isoline) RequireLines(test, expected);
}

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto target = device->Target();
        if (!target.tessellation.has_value() || !(target.tessellation->features == Everything)) {
            std::puts("tessellation domains: skipped, the target does not draw every tessellation domain, partitioning and topology");
            return VulkanTestSkipped;
        }
        const std::vector<Case> cases{
            // Tiny inputs first.
            {"quad integer cw, level 1", T::Quad, T::Integer, T::TriangleCw, {1, 1, 1, 1, 1, 1}},
            {"quad integer cw", T::Quad, T::Integer, T::TriangleCw, {3, 4, 5, 2, 4, 3}},
            {"quad integer ccw", T::Quad, T::Integer, T::TriangleCcw, {3, 4, 5, 2, 4, 3}},
            {"quad fractional_even cw", T::Quad, T::FractionalEven, T::TriangleCw, {2.5f, 3.7f, 5.2f, 4.1f, 3.3f, 6.6f}},
            {"quad fractional_even ccw", T::Quad, T::FractionalEven, T::TriangleCcw, {2.5f, 3.7f, 5.2f, 4.1f, 3.3f, 6.6f}},
            {"quad pow2 cw", T::Quad, T::Pow2, T::TriangleCw, {3, 5, 2, 7, 6, 3}},
            {"quad fractional_odd cw, 2 slices", T::Quad, T::FractionalOdd, T::TriangleCw, {9.5f, 8.2f, 7.7f, 10.1f, 9.9f, 8.8f}},
            {"triangle fractional_odd ccw", T::Triangle, T::FractionalOdd, T::TriangleCcw, {5, 5, 5, 5, 0, 0}},
            {"triangle integer cw", T::Triangle, T::Integer, T::TriangleCw, {2, 3, 4, 3, 0, 0}},
            {"triangle fractional_even ccw", T::Triangle, T::FractionalEven, T::TriangleCcw, {2.2f, 3.3f, 4.4f, 5.5f, 0, 0}},
            {"isoline integer line (detail 4, density 3)", T::Isoline, T::Integer, T::Line, {4, 3, 0, 0, 0, 0}},
            {"isoline fractional_even line", T::Isoline, T::FractionalEven, T::Line, {5.5f, 2.0f, 0, 0, 0, 0}},
            {"isoline integer point", T::Isoline, T::Integer, T::Point, {3, 2, 0, 0, 0, 0}},
            {"quad integer point", T::Quad, T::Integer, T::Point, {3, 4, 5, 2, 4, 3}},
            {"triangle fractional_odd point", T::Triangle, T::FractionalOdd, T::Point, {3, 3, 3, 3, 0, 0}},
            {"quad culled (outer 0)", T::Quad, T::Integer, T::TriangleCw, {3, 0, 3, 3, 3, 3}},
            {"isoline culled (density 0)", T::Isoline, T::Integer, T::Line, {4, 0, 0, 0, 0, 0}},
            {"quad integer cw, clamped to [2, 3]", T::Quad, T::Integer, T::TriangleCw, {1, 4, 5, 2, 4, 1}, 2.0f, 3.0f},
            {"triangle fractional_odd cw, clamped to [1, 4]", T::Triangle, T::FractionalOdd, T::TriangleCw, {7, 9, 5, 9, 0, 0}, 1.0f, 4.0f},
        };
        for (const auto& test : cases) Run(*device, test);
        std::puts("Tessellation domain tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
