#ifndef CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLCONSERVATIVE_HPP
#define CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLCONSERVATIVE_HPP

// Conservative rasterization emulation (work package D6, owner DRAW; DECISIONS "conservative
// rasterization is emulated"): the Metal API has none, so draws that ask for it run in the
// triangle-vertices mode (APS5_TRIANGLE_VERTICES: every host vertex knows its triangle's three
// guest vertices) and add:
// - vertex stage: each corner moves outward in framebuffer space until its two edges have moved
//   by the half-pixel extent along their normals (the offset lines of GPU Gems 2, ch. 42), so the
//   rasterizer visits every pixel whose square the triangle touches; the triangle's three
//   framebuffer-space vertices go to the fragment stage as flat outputs;
// - fragment stage: the exact test (separating axes x, y and the three edge normals) discards the
//   fragments whose pixel square does not meet the triangle (overestimation), or that the triangle
//   does not cover entirely (underestimation).
// Degenerate triangles (zero area: segments and points) are rasterized as the bounding box of
// their vertices grown by half a pixel and tested on the axes they have. The functions return MSL
// text for MslEntryPoint.cpp (B6), which owns the vertex/fragment functions.
//
// Names the text expects (set by the caller):
// - vertex: `aps5_v[3]` the three corners' Aps5VertexOut (field `aps5_position`, clip space),
//   `aps5_corner` this host vertex's corner (0-2), `aps5_out` the output (already aps5_v[corner]),
//   `aps5_tv` the TriangleVertices header (field `viewport`: Vulkan x, y, width, height).
// - fragment: `aps5_in` the stage input (fields `aps5_position` and the flat fields below).
// The mode macro APS5_CONSERVATIVE is 1 for overestimation, 2 for underestimation.

#include <string>

namespace ShaderRecompiler::MslConservative {

inline constexpr const char* Macro = "APS5_CONSERVATIVE";

// Fields of the vertex output struct.
inline std::string VertexOutputFields() {
    return "    float4 aps5_cr_01 [[user(aps5_cr_01)]];\n"
           "    float2 aps5_cr_2 [[user(aps5_cr_2)]];\n";
}

// Fields of the fragment input struct (the same user names, flat).
inline std::string FragmentInputFields() {
    return "    float4 aps5_cr_01 [[user(aps5_cr_01), flat]];\n"
           "    float2 aps5_cr_2 [[user(aps5_cr_2), flat]];\n";
}

// Vertex stage, after aps5_out = aps5_v[aps5_corner].
inline std::string VertexText() {
    return R"(    {
        // Framebuffer-space corners (the driver's Metal viewport reproduces Vulkan's transform:
        // x_fb = x + w (1 + x_ndc) / 2, y_fb = y + h (1 + y_ndc) / 2).
        const float4 aps5_vp = aps5_tv.viewport;
        float2 aps5_s[3];
        bool aps5_projected = true;
        for (uint i = 0u; i < 3u; ++i) {
            const float4 p = aps5_v[i].aps5_position;
            aps5_projected = aps5_projected && p.w > 0.0f;
            const float2 ndc = p.xy / p.w;
            aps5_s[i] = aps5_vp.xy + aps5_vp.zw * (1.0f + ndc) * 0.5f;
        }
        aps5_out.aps5_cr_01 = float4(aps5_s[0], aps5_s[1]);
        aps5_out.aps5_cr_2 = aps5_s[2];
#if APS5_CONSERVATIVE == 1
        if (aps5_projected) {
            const float2 e1 = aps5_s[1] - aps5_s[0];
            const float2 e2 = aps5_s[2] - aps5_s[0];
            const float area = e1.x * e2.y - e1.y * e2.x;
            float2 moved;
            if (abs(area) > 1.0e-6f) {
                // Outward unit normals of the corner's two edges, each line moved out by the
                // projection of the half-pixel square onto its normal; the corner goes to the
                // offset lines' intersection (bounded for very sharp corners).
                const uint a = aps5_corner, b = (aps5_corner + 1u) % 3u, c = (aps5_corner + 2u) % 3u;
                const float orient = area > 0.0f ? 1.0f : -1.0f;
                const float2 dNext = aps5_s[b] - aps5_s[a];
                const float2 dPrev = aps5_s[a] - aps5_s[c];
                const float2 nNext = normalize(float2(dNext.y, -dNext.x) * orient);
                const float2 nPrev = normalize(float2(dPrev.y, -dPrev.x) * orient);
                const float oNext = 0.5f * (abs(nNext.x) + abs(nNext.y)) + 1.0f / 512.0f;
                const float oPrev = 0.5f * (abs(nPrev.x) + abs(nPrev.y)) + 1.0f / 512.0f;
                const float det = nNext.x * nPrev.y - nNext.y * nPrev.x;
                float2 shift = abs(det) > 1.0e-3f ? float2((oNext * nPrev.y - oPrev * nNext.y) / det, (nNext.x * oPrev - nPrev.x * oNext) / det) : nNext * oNext + nPrev * oPrev;
                const float length2 = dot(shift, shift);
                if (length2 > 4096.0f) shift *= 64.0f / sqrt(length2);
                moved = aps5_s[a] + shift;
            } else {
                // Degenerate: a triangle over the vertices' box grown by half a pixel (corner 0 the
                // box's top-left, 1 and 2 twice its width and height away).
                const float2 low = min(min(aps5_s[0], aps5_s[1]), aps5_s[2]) - 0.5f - 1.0f / 512.0f;
                const float2 high = max(max(aps5_s[0], aps5_s[1]), aps5_s[2]) + 0.5f + 1.0f / 512.0f;
                const float2 size = high - low;
                moved = aps5_corner == 0u ? low : aps5_corner == 1u ? float2(low.x + 2.0f * size.x, low.y) : float2(low.x, low.y + 2.0f * size.y);
            }
            const float4 p = aps5_out.aps5_position;
            const float2 ndc = (moved - aps5_vp.xy) / aps5_vp.zw * 2.0f - 1.0f;
            aps5_out.aps5_position = float4(ndc * p.w, p.z, p.w);
        }
#endif
    }
)";
}

// Fragment stage, before the guest body: discard the fragments the mode excludes.
inline std::string FragmentText() {
    return R"(    {
        const float2 aps5_t[3] = {aps5_in.aps5_cr_01.xy, aps5_in.aps5_cr_01.zw, aps5_in.aps5_cr_2};
        const float2 aps5_c = aps5_in.aps5_position.xy;
        const float2 e1 = aps5_t[1] - aps5_t[0];
        const float2 e2 = aps5_t[2] - aps5_t[0];
        const float area = e1.x * e2.y - e1.y * e2.x;
        const float orient = area >= 0.0f ? 1.0f : -1.0f;
        bool aps5_keep = true;
#if APS5_CONSERVATIVE == 1
        // Overestimation: the pixel square meets the triangle (no separating axis among x, y and
        // the edge normals; a degenerate triangle has only its nonzero edges' normals).
        const float2 low = min(min(aps5_t[0], aps5_t[1]), aps5_t[2]);
        const float2 high = max(max(aps5_t[0], aps5_t[1]), aps5_t[2]);
        aps5_keep = all(high >= aps5_c - 0.5f) && all(low <= aps5_c + 0.5f);
        for (uint i = 0u; i < 3u && aps5_keep; ++i) {
            const float2 a = aps5_t[i];
            const float2 d = aps5_t[(i + 1u) % 3u] - a;
            if (dot(d, d) <= 1.0e-12f) continue;
            float2 n = float2(d.y, -d.x);
            if (abs(area) > 1.0e-6f) {
                n *= orient;
                const float radius = 0.5f * (abs(n.x) + abs(n.y));
                // The triangle lies where dot(n, x - a) <= 0; the square's nearest point is radius
                // closer along n than its centre.
                if (dot(n, aps5_c - a) - radius > 0.0f) aps5_keep = false;
            } else {
                // A segment: separated when the square lies entirely on one side of its line.
                const float radius = 0.5f * (abs(n.x) + abs(n.y));
                if (abs(dot(n, aps5_c - a)) - radius > 0.0f) aps5_keep = false;
            }
        }
#else
        // Underestimation: the triangle covers the whole pixel square (all four corners inside).
        if (abs(area) <= 1.0e-6f) aps5_keep = false;
        for (uint i = 0u; i < 3u && aps5_keep; ++i) {
            const float2 a = aps5_t[i];
            const float2 d = aps5_t[(i + 1u) % 3u] - a;
            const float2 n = float2(d.y, -d.x) * orient;
            const float radius = 0.5f * (abs(n.x) + abs(n.y));
            if (dot(n, aps5_c - a) + radius > 0.0f) aps5_keep = false;
        }
#endif
        if (!aps5_keep) { discard_fragment(); aps5_demoted = true; }
    }
)";
}

// The names SHD3's MslEntryPoint.cpp calls (coord/m2.md ~06:20).
inline std::string MslConservativeVertex() { return VertexText(); }
inline std::string MslConservativeFragment() { return FragmentText(); }
inline std::string MslConservativeVertexOutputFields() { return VertexOutputFields(); }
inline std::string MslConservativeFragmentInputFields() { return FragmentInputFields(); }

}

#endif
