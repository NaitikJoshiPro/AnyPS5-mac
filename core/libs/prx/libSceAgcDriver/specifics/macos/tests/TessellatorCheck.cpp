// agc_metal_tessellator (DECISIONS D24): the Metal path's software tessellator (MslTessellatorCore.inc,
// the text the mesh functions run) checked on the CPU against Microsoft's D3D11 reference tessellator
// (3rdparty/d3d11-tessellator): for every domain, partitioning and output topology and many factor
// sets (edge cases, exhaustive small integer sets, seeded random ones), the same point count, every
// domain point bit for bit, the same primitives in the same order, and the same primitives when the
// patch is cut into the mesh-threadgroup slices the object stage plans. Pure CPU, no Metal device.

#include "MetalBackend/MslTessellator.hpp"
#include "tessellator.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

namespace T = ShaderRecompiler::MslTessellator;

int failures = 0;
std::uint64_t cases = 0;
std::uint64_t primitivesChecked = 0;
int largestSlices = 0;

std::string Describe(int domain, int partitioning, int topology, const T::Aps5TessFactors& factors) {
    char text[256];
    std::snprintf(text, sizeof(text), "domain %d partitioning %d topology %d outer (%a %a %a %a) inner (%a %a)", domain, partitioning, topology, factors.outer[0], factors.outer[1], factors.outer[2], factors.outer[3], factors.inner[0], factors.inner[1]);
    return text;
}

void Fail(const std::string& what, const std::string& reason) {
    if (++failures <= 20) std::fprintf(stderr, "FAIL %s: %s\n", what.c_str(), reason.c_str());
}

int Corners(int topology) {
    return topology == T::Point ? 1 : topology == T::Line ? 2 : 3;
}

void Check(int domain, int partitioning, int topology, const T::Aps5TessFactors& factors) {
    ++cases;
    auto reference = std::make_unique<CHWTessellator>();
    reference->Init(static_cast<PIPE_TESSELLATOR_PARTITIONING>(partitioning), static_cast<PIPE_TESSELLATOR_OUTPUT_PRIMITIVE>(topology));
    if (domain == T::Isoline) reference->TessellateIsoLineDomain(factors.outer[0], factors.outer[1]);
    else if (domain == T::Triangle) reference->TessellateTriDomain(factors.outer[0], factors.outer[1], factors.outer[2], factors.inner[0]);
    else reference->TessellateQuadDomain(factors.outer[0], factors.outer[1], factors.outer[2], factors.outer[3], factors.inner[0], factors.inner[1]);
    const auto what = Describe(domain, partitioning, topology, factors);

    const auto patch = T::aps5_tess_process(domain, partitioning, topology, factors);
    const int points = patch.culled != 0 ? 0 : patch.points;
    if (points != reference->GetPointCount()) {
        Fail(what, "point count " + std::to_string(points) + ", reference " + std::to_string(reference->GetPointCount()));
        return;
    }
    const auto* referencePoints = reference->GetPoints();
    for (int index = 0; index < points; ++index) {
        const auto point = T::aps5_tess_point(patch, index);
        if (std::bit_cast<std::uint32_t>(point.u) != std::bit_cast<std::uint32_t>(referencePoints[index].u) || std::bit_cast<std::uint32_t>(point.v) != std::bit_cast<std::uint32_t>(referencePoints[index].v)) {
            char text[160];
            std::snprintf(text, sizeof(text), "point %d is (%a, %a), reference (%a, %a)", index, point.u, point.v, referencePoints[index].u, referencePoints[index].v);
            Fail(what, text);
            return;
        }
    }
    int unused = 0;
    const int count = T::aps5_tess_primitive_count(patch, &unused);
    const int corners = Corners(topology);
    if (count * corners != reference->GetIndexCount()) {
        Fail(what, "primitive count " + std::to_string(count) + " x " + std::to_string(corners) + ", reference index count " + std::to_string(reference->GetIndexCount()));
        return;
    }
    std::vector<int> all(static_cast<std::size_t>(std::max(count, 1)) * corners, -1);
    if (T::aps5_tess_window(patch, 0, count, all.data()) != count) {
        Fail(what, "the whole-patch window is short");
        return;
    }
    const auto* referenceIndices = reference->GetIndices();
    for (int index = 0; index < count * corners; ++index) {
        if (all[index] != referenceIndices[index]) {
            Fail(what, "index " + std::to_string(index) + " is " + std::to_string(all[index]) + ", reference " + std::to_string(referenceIndices[index]));
            return;
        }
        if (all[index] < 0 || all[index] >= points) {
            Fail(what, "index " + std::to_string(index) + " is outside the points");
            return;
        }
    }
    primitivesChecked += static_cast<std::uint64_t>(count);
    // The mesh-threadgroup slices of the object stage, and an odd window size.
    for (const int slice : {T::SlicePrimitives(topology), 7}) {
        int slices = 0;
        std::vector<int> window(static_cast<std::size_t>(slice) * corners);
        for (int first = 0; first < count; first += slice, ++slices) {
            std::fill(window.begin(), window.end(), -1);
            const int written = T::aps5_tess_window(patch, first, slice, window.data());
            if (written != std::min(slice, count - first)) {
                Fail(what, "slice at " + std::to_string(first) + " wrote " + std::to_string(written));
                return;
            }
            if (!std::equal(window.begin(), window.begin() + written * corners, all.begin() + static_cast<std::ptrdiff_t>(first) * corners)) {
                Fail(what, "slice at " + std::to_string(first) + " of " + std::to_string(slice) + " differs from the whole patch");
                return;
            }
        }
        if (slice == T::SlicePrimitives(topology)) largestSlices = std::max(largestSlices, slices);
    }
}

std::vector<int> Topologies(int domain) {
    if (domain == T::Isoline) return {T::Point, T::Line};
    return {T::Point, T::Line, T::TriangleCw, T::TriangleCcw};
}

T::Aps5TessFactors Uniform(float value) {
    return T::Aps5TessFactors{{value, value, value, value}, {value, value}};
}

}

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const std::vector<float> special{nan, -nan, -infinity, -1.0f, -0.0f, 0.0f, std::bit_cast<float>(1u), 1e-30f, 0.25f, 0.999f, 1.0f, std::bit_cast<float>(0x3f800001u), 1.0000076f, 1.0000153f, 1.2f, 1.5f, 1.99f, 2.0f, 2.01f, 2.5f, 3.0f, 3.3f, 4.0f, 5.0f, 6.7f, 7.0f, 8.0f, 9.5f, 15.5f, 16.0f, 17.0f, 31.9f, 32.0f, 33.0f, 47.25f, 62.5f, 63.0f, 63.5f, 64.0f, 65.0f, 1000.0f, infinity};
    for (int domain = 0; domain < 3; ++domain) {
        for (int partitioning = 0; partitioning < 4; ++partitioning) {
            for (const int topology : Topologies(domain)) {
                // Every special value on all factors, then each special value on one factor at a time.
                for (const float value : special) Check(domain, partitioning, topology, Uniform(value));
                const int slots = domain == T::Isoline ? 2 : domain == T::Triangle ? 4 : 6;
                for (int slot = 0; slot < slots; ++slot) {
                    for (const float value : special) {
                        auto factors = Uniform(5.5f);
                        if (domain == T::Triangle && slot == 3) factors.inner[0] = value;
                        else if (domain == T::Quad && slot >= 4) factors.inner[slot - 4] = value;
                        else factors.outer[slot] = value;
                        Check(domain, partitioning, topology, factors);
                    }
                }
            }
        }
    }
    // Exhaustive small integer factor sets with triangle output (both windings in the quad check).
    for (int partitioning = 0; partitioning < 4; ++partitioning) {
        for (int a = 1; a <= 6; ++a) {
            for (int b = 1; b <= 6; ++b) {
                for (int c = 1; c <= 6; ++c) {
                    for (int i = 1; i <= 6; ++i) Check(T::Triangle, partitioning, T::TriangleCw, {{float(a), float(b), float(c), 0.0f}, {float(i), 0.0f}});
                    for (int d = 1; d <= 4; ++d) {
                        for (int i = 1; i <= 4; ++i) {
                            for (int j = 1; j <= 4; ++j) Check(T::Quad, partitioning, (a + i) % 2 == 0 ? T::TriangleCw : T::TriangleCcw, {{float(a), float(b), float(c), float(d)}, {float(i), float(j)}});
                        }
                    }
                }
            }
            for (int b = 1; b <= 64; ++b) Check(T::Isoline, partitioning, T::Line, {{float(a), float(b), 0.0f, 0.0f}, {0.0f, 0.0f}});
        }
    }
    // Seeded random factors: continuous values over the whole range, and mixed tiny/huge sets.
    std::mt19937 random(0x7e55u);
    std::uniform_real_distribution<float> wide(-2.0f, 70.0f);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(special.size()) - 1);
    std::uniform_int_distribution<int> coin(0, 3);
    for (int round = 0; round < 60000; ++round) {
        const int domain = round % 3;
        const int partitioning = (round / 3) % 4;
        const auto topologies = Topologies(domain);
        const int topology = topologies[(round / 12) % topologies.size()];
        T::Aps5TessFactors factors{};
        for (auto& value : factors.outer) value = coin(random) == 0 ? special[pick(random)] : wide(random);
        for (auto& value : factors.inner) value = coin(random) == 0 ? special[pick(random)] : wide(random);
        Check(domain, partitioning, topology, factors);
    }
    std::printf("tessellator: %llu cases, %llu primitives checked, at most %d mesh threadgroups per patch, %d failures\n", static_cast<unsigned long long>(cases), static_cast<unsigned long long>(primitivesChecked), largestSlices, failures);
    if (largestSlices > 1024) {
        std::fprintf(stderr, "FAIL: a patch needs more than 1024 mesh threadgroups (Metal's per-object grid budget)\n");
        return 1;
    }
    if (failures != 0) return 1;
    std::puts("Tessellator tests passed");
    return 0;
}
