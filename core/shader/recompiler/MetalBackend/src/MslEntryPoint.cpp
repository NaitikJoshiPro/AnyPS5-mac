#include "MetalBackend/MslEmitterState.hpp"
#include "MetalBackend/MslConservative.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace ShaderRecompiler {

namespace {

std::string BuiltinParameter(MslBuiltin builtin) {
    switch (builtin) {
    case MslBuiltin::ThreadPositionInThreadgroup:
        return "uint3 aps5_local_id [[thread_position_in_threadgroup]]";
    case MslBuiltin::ThreadgroupPositionInGrid:
        return "uint3 aps5_group_id [[threadgroup_position_in_grid]]";
    case MslBuiltin::ThreadgroupsPerGrid:
        return "uint3 aps5_group_count [[threadgroups_per_grid]]";
    case MslBuiltin::ThreadIndexInThreadgroup:
        return "uint aps5_local_index [[thread_index_in_threadgroup]]";
    case MslBuiltin::ThreadIndexInSimdgroup:
        return "uint aps5_simd_lane [[thread_index_in_simdgroup]]";
    default:
        throw std::runtime_error("MSL emission failed: unknown builtin parameter");
    }
}

// Image heaps follow DescriptorBindingForImage: per class 7 dimensions in this order. 1D images are
// 2D textures one texel high (Metal's texture1d has no mipmaps and no sampling with LOD).
constexpr std::uint32_t HeapDimensions = 7u;
// MslAbi Version 9: both multisampled dimensions are texture2d_ms_array (a 2D MSAA view is one
// guest layer of host slices; emulated 8x targets hold two 4x slices per layer, DECISIONS D28).
constexpr const char* TextureShapes[HeapDimensions] = {"texture2d", "texture2d_array", "texture2d", "texture2d_array", "texture2d_ms_array", "texture2d_ms_array", "texture3d"};
constexpr const char* DepthShapes[HeapDimensions] = {"depth2d", "depth2d_array", "depth2d", "depth2d_array", "depth2d_ms", "depth2d_ms_array", "texture3d"};

}

std::string MslTableMemberType(DescriptorBindingKind kind) {
    switch (MslAbi::KindOf(kind)) {
    case MslAbi::EntryKind::Buffer:
        return "Aps5Buffer";
    case MslAbi::EntryKind::Sampler:
        return "sampler";
    case MslAbi::EntryKind::Texture:
        break;
    }
    const auto value = static_cast<std::uint32_t>(kind);
    if (value < RuntimeAbi::FirstComparisonImageBinding) {
        // Sampled float (1-7), uint (8-14), sint (15-21).
        static constexpr const char* Scalars[] = {"float", "uint", "int"};
        const auto index = value - RuntimeAbi::FirstImageBinding;
        return std::string(TextureShapes[index % HeapDimensions]) + "<" + Scalars[index / HeapDimensions] + ">";
    }
    if (value < RuntimeAbi::FirstStorageImageBinding) {
        // Depth-compare images (22-28).
        return std::string(DepthShapes[(value - RuntimeAbi::FirstComparisonImageBinding) % HeapDimensions]) + "<float>";
    }
    // Storage float (29-35), uint (36-42), 32-bit atomic uint (43-49), 64-bit atomic uint (50-56).
    // MslAbi Version 6: Metal cannot write multisampled textures from a shader, so storage views of
    // 2D MSAA (array) images are 2D arrays of layers * samples slices (MslImage.cpp,
    // EmulatedMultisampleCoord).
    static constexpr const char* Scalars[] = {"float", "uint", "uint", "ulong"};
    const auto index = value - RuntimeAbi::FirstStorageImageBinding;
    const auto dimension = index % HeapDimensions;
    const bool multisampled = dimension == 4u || dimension == 5u;
    return std::string(multisampled ? "texture2d_array" : TextureShapes[dimension]) + "<" + Scalars[index / HeapDimensions] + ", access::read_write>";
}

std::string MslTableDeclaration(const MslAbi::TableLayout& layout) {
    std::string text = "struct Aps5Table {\n";
    for (const auto& range : layout.ranges) {
        const auto type = MslTableMemberType(range.binding);
        if (range.kind == MslAbi::EntryKind::Buffer) text += "    " + type + " " + MslTableField(range.binding) + "[" + std::to_string(range.count) + "];\n";
        else text += "    array<" + type + ", " + std::to_string(range.count) + "> " + MslTableField(range.binding) + ";\n";
    }
    text += "};\n";
    // The driver writes the table by MslAbi's layout: a mismatch must not compile.
    text += "static_assert(sizeof(Aps5Table) == " + std::to_string(layout.bytes) + ", \"MslAbi descriptor table size\");\n";
    for (const auto& range : layout.ranges) {
        text += "static_assert(__builtin_offsetof(Aps5Table, " + MslTableField(range.binding) + ") == " + std::to_string(range.offset) + ", \"MslAbi descriptor table offset\");\n";
    }
    return text + "\n";
}

std::array<std::uint32_t, 3> MslThreadsPerThreadgroup(const ShaderWorkgroupInputInfo& workgroup, std::uint32_t laneCount) {
    const std::array<std::uint32_t, 3> guest {std::max(workgroup.threadsNum[0], 1u), std::max(workgroup.threadsNum[1], 1u), std::max(workgroup.threadsNum[2], 1u)};
    if (laneCount == 1u) return guest;
    // Wave64 on 32-wide SIMD groups: one host SIMD group per guest wave, 32 invocations of two
    // lanes each, as the SPIR-V path's LocalSize (the guest ids are rebuilt from the linear index).
    const auto threads = guest[0] * guest[1] * guest[2];
    return {(threads + 63u) / 64u * 32u, 1u, 1u};
}

MslMeshOutput MslMeshOutputOf(const ShaderMeshInputInfo& mesh) {
    MslMeshOutput output;
    output.rectList = mesh.inputPrimitive == 17u;
    if (!output.rectList) {
        output.guestVertices = output.vertices = mesh.maxVertices;
        output.guestPrimitives = output.primitives = mesh.maxPrimitives;
        return output;
    }
    // A rect-list NGG vertex program allocates at most its group's vertices and rects (the
    // translator's s2 counts), so the host mesh holds those, one synthesized vertex per rect after
    // them and two triangles per rect.
    output.guestVertices = std::min(mesh.maxVertices, mesh.verticesPerGroup);
    output.guestPrimitives = std::min(mesh.maxPrimitives, mesh.primitivesPerGroup);
    output.vertices = output.guestVertices + output.guestPrimitives;
    output.primitives = output.guestPrimitives * 2u;
    return output;
}

namespace {

// Mesh functions (NGG primitive shaders, mirroring EmitMeshEntryPoint on the SPIR-V path): the guest
// program runs in aps5_body with each lane half's exports staged in aps5_mv / aps5_mprim and the
// GS_ALLOC_REQ counts in threadgroup memory; after a threadgroup barrier invocation i writes vertex
// i and primitive i (indices: three 9-bit fields at bits 0, 10 and 20, clamped to the last vertex;
// bit 31 culls). Threadgroup variables live in the mesh function and reach aps5_body as pointers.
// Rect lists (VGT_PRIMITIVE_TYPE 17, GS_OUT_PRIM_TYPE RECTLIST; DECISIONS D41): the guest exports
// one primitive of three vertices per rect and the primitive assembler adds the fourth corner. As
// the vertex function's APS5_RECT_LIST (and the SPIR-V rect-list TCS), the right-angle vertex c is
// the one whose neighbours share its x and y; the fourth vertex is the other two minus c, for the
// position and every parameter alike (one plane over the rect, as the hardware interpolates it);
// the rect draws as (c, n, p) and (p, n, v3), both in the guest's winding. A rect without exactly
// one right angle, a w that differs between its vertices or is not positive, or a non-finite fourth
// position is culled. Vertices are staged raw in threadgroup memory (another invocation's export
// feeds each rect), then the window-space / depth epilogue runs on every written vertex.
std::string MeshRectListEpilogue(MslEmitterState& state, const MslMeshOutput& output, bool layer, std::uint32_t ldsBytes) {
    const auto& mesh = state.inputInfo.vertex->mesh;
    std::uint32_t parameters = 0u;
    for (const auto& stageOutput : state.program.Info().outputs) parameters += stageOutput.kind == StageOutputKind::Parameter ? 1u : 0u;
    const auto staging = output.guestVertices * 16u * (1u + parameters);
    const auto total = staging + ldsBytes + 8u + (layer ? mesh.maxVertices * 4u : 0u);
    if (total > state.target.maxThreadgroupMemoryBytes) state.Fail("a rect-list mesh function stages " + std::to_string(output.guestVertices) + " vertices of " + std::to_string(16u * (1u + parameters)) + " bytes in threadgroup memory: " + std::to_string(total) + " bytes with LDS exceed the " + std::to_string(state.target.maxThreadgroupMemoryBytes) + "-byte limit");
    const auto local = state.Builtin(MslBuiltin::ThreadIndexInThreadgroup);
    const auto guestVertices = std::to_string(output.guestVertices) + "u";
    std::string text = "    threadgroup Aps5MeshVertex aps5_rv[" + std::to_string(output.guestVertices) + "];\n";
    text += "    const uint aps5_vertices = min(aps5_malloc[0], " + guestVertices + ");\n";
    text += "    const uint aps5_rects = min(aps5_malloc[1], " + std::to_string(output.guestPrimitives) + "u);\n";
    text += "    if (" + local + " == 0u) aps5_mesh.set_primitive_count(aps5_rects * 2u);\n";
    text += "    const uint aps5_last = max(aps5_vertices, 1u) - 1u;\n";
    for (std::uint32_t half = 0; half < state.laneCount; ++half) {
        state.half = half;
        text += "    {\n        const uint aps5_i = " + state.GuestInvocationIndex() + ";\n";
        text += "        if (aps5_i < aps5_vertices) aps5_rv[aps5_i] = aps5_mv[" + std::to_string(half) + "];\n    }\n";
    }
    text += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    const auto finish = [](const std::string& name, const std::string& indent) {
        return WindowSpaceEpilogue(name + ".aps5_position", indent) +
               "#if defined(APS5_DEPTH_NEGATIVE_ONE_TO_ONE) && APS5_DEPTH_NEGATIVE_ONE_TO_ONE\n" + indent + name + ".aps5_position.z = (" + name + ".aps5_position.z + " + name + ".aps5_position.w) * 0.5f;\n#endif\n";
    };
    for (std::uint32_t half = 0; half < state.laneCount; ++half) {
        state.half = half;
        const auto h = std::to_string(half);
        text += "    {\n        const uint aps5_i = " + state.GuestInvocationIndex() + ";\n";
        text += "        if (aps5_i < aps5_vertices) {\n            Aps5MeshVertex aps5_v = aps5_rv[aps5_i];\n" + finish("aps5_v", "            ");
        text += "            aps5_mesh.set_vertex(aps5_i, aps5_v);\n        }\n";
        text += "        if (aps5_i < aps5_rects) {\n            const uint aps5_packed = aps5_mprim[" + h + "];\n";
        text += "            uint aps5_k[3];\n            Aps5MeshVertex aps5_r[3];\n";
        text += "            for (uint c = 0u; c < 3u; ++c) {\n                aps5_k[c] = min(extract_bits(aps5_packed, c * 10u, 9u), aps5_last);\n                aps5_r[c] = aps5_rv[aps5_k[c]];\n            }\n";
        text += "            bool aps5_edge[3];\n            float aps5_weight[3];\n            uint aps5_edges = 0u;\n";
        text += "            for (uint c = 0u; c < 3u; ++c) {\n";
        text += "                const uint next = (c + 1u) % 3u, previous = (c + 2u) % 3u;\n";
        text += "                const bool xy = aps5_r[c].aps5_position.x == aps5_r[next].aps5_position.x && aps5_r[c].aps5_position.y == aps5_r[previous].aps5_position.y;\n";
        text += "                const bool yx = aps5_r[c].aps5_position.y == aps5_r[next].aps5_position.y && aps5_r[c].aps5_position.x == aps5_r[previous].aps5_position.x;\n";
        text += "                aps5_edge[c] = xy || yx;\n                aps5_weight[c] = aps5_edge[c] ? -1.0f : 1.0f;\n                aps5_edges += aps5_edge[c] ? 1u : 0u;\n            }\n";
        text += "            const uint aps5_first = aps5_edge[1] ? 1u : (aps5_edge[2] ? 2u : 0u);\n";
        text += "            Aps5MeshVertex aps5_v3 = aps5_r[aps5_first];\n";
        text += "            aps5_v3.aps5_position = aps5_r[0].aps5_position * aps5_weight[0] + aps5_r[1].aps5_position * aps5_weight[1] + aps5_r[2].aps5_position * aps5_weight[2];\n";
        for (const auto& stageOutput : state.program.Info().outputs) {
            if (stageOutput.kind != StageOutputKind::Parameter) continue;
            const auto field = "p" + std::to_string(stageOutput.location);
            text += "            aps5_v3." + field + " = aps5_r[0]." + field + " * aps5_weight[0] + aps5_r[1]." + field + " * aps5_weight[1] + aps5_r[2]." + field + " * aps5_weight[2];\n";
        }
        text += "            const float aps5_w = aps5_r[0].aps5_position.w;\n";
        text += "            const bool aps5_valid_rect = aps5_edges == 1u && aps5_w > 0.0f && aps5_r[1].aps5_position.w == aps5_w && aps5_r[2].aps5_position.w == aps5_w && all(isfinite(aps5_v3.aps5_position));\n";
        text += finish("aps5_v3", "            ");
        text += "            const uint aps5_extra = " + guestVertices + " + aps5_i;\n";
        text += "            aps5_mesh.set_vertex(aps5_extra, aps5_v3);\n";
        text += "            const uint aps5_c = aps5_k[aps5_first], aps5_n = aps5_k[(aps5_first + 1u) % 3u], aps5_p = aps5_k[(aps5_first + 2u) % 3u];\n";
        text += "            const uint aps5_corners[6] = {aps5_c, aps5_n, aps5_p, aps5_p, aps5_n, aps5_extra};\n";
        text += "            for (uint c = 0u; c < 6u; ++c) aps5_mesh.set_index(aps5_i * 6u + c, uchar(aps5_corners[c]));\n";
        text += "            Aps5MeshPrimitive aps5_prim = {};\n";
        text += "            aps5_prim.aps5_culled = (aps5_packed & 0x80000000u) != 0u || !aps5_valid_rect;\n";
        if (layer) text += "            aps5_prim.aps5_layer = aps5_mlayer[aps5_k[" + std::to_string(std::min(mesh.provokingVertex, 2u)) + "]];\n";
        text += "            aps5_mesh.set_primitive(aps5_i * 2u, aps5_prim);\n            aps5_mesh.set_primitive(aps5_i * 2u + 1u, aps5_prim);\n        }\n    }\n";
    }
    state.half = 0u;
    return text;
}

// The triangle-vertices mode of mesh functions (APS5_TRIANGLE_VERTICES, MslAbi Version 13; DECISIONS
// D46), the mesh counterpart of the vertex function's D20
// emulation: per-vertex fragment values need each triangle's three corners, and the emulated
// barycentrics need one-hot corner weights, which a vertex shared between triangles cannot carry.
// So every output triangle gets three unshared vertices (weights (1,0,0), (0,1,0), (0,0,1)) and the
// corners' values of the per-vertex locations in its primitive record. A mesh<> holds 256 vertices,
// so the subgroup's primitives are split over the grid's z dimension in equal parts of at most 85
// (MslAbi::MeshTriangleVerticesPrimitives); every split threadgroup runs the guest program in full
// and writes its part. The guest's vertices and primitive words are staged in threadgroup memory
// (any invocation's export feeds any triangle). Corner 0 is the provoking vertex, then the guest
// primitive's order (rotated for last-vertex provoking, which keeps the winding), as the vertex
// function's TriangleVerticesHeader order, so P0/P10/P20 and the weights agree between the paths.
// The triangle-vertices mode's staging in the guest's LDS (MeshTriangleVerticesEpilogue): every
// vertex's position and parameters (16 bytes each), then every primitive's packed indices.
std::uint32_t MeshTriangleVerticesStagingBytes(const MslEmitterState& state) {
    const auto& mesh = state.inputInfo.vertex->mesh;
    std::uint32_t parameters = 0u;
    for (const auto& stageOutput : state.program.Info().outputs) parameters += stageOutput.kind == StageOutputKind::Parameter ? 1u : 0u;
    return mesh.maxVertices * 16u * (1u + parameters) + mesh.maxPrimitives * 4u;
}

// Point draws in the triangle-vertices mode (MslAbi Version 17, DECISIONS D78): no staging. Every
// vertex record carries its own per-vertex values (user(aps5_pv<L>_0)) and corner weight; threadgroup
// z outputs the vertices [z x APS5_POINT_SPLIT_VERTICES, ...) and the points whose vertex is among
// them (Metal's mesh output holds 32 KiB: MslAbi::MeshPointSplitVertices).
std::string MeshPointVerticesEpilogue(MslEmitterState& state, bool layer) {
    const auto& mesh = state.inputInfo.vertex->mesh;
    const auto local = state.Builtin(MslBuiltin::ThreadIndexInThreadgroup);
    const auto group = state.Builtin(MslBuiltin::ThreadgroupPositionInGrid);
    const std::string split = MslAbi::PointSplitVerticesMacro;
    std::string text = "    threadgroup atomic_uint aps5_point_count;\n";
    text += "    if (" + local + " == 0u) atomic_store_explicit(&aps5_point_count, 0u, memory_order_relaxed);\n";
    text += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    text += "    const uint aps5_vertices = min(aps5_malloc[0], " + std::to_string(mesh.maxVertices) + "u);\n";
    text += "    const uint aps5_primitives = min(aps5_malloc[1], " + std::to_string(mesh.maxPrimitives) + "u);\n";
    text += "    const uint aps5_last = max(aps5_vertices, 1u) - 1u;\n";
    text += "    const uint aps5_base = " + group + ".z * " + split + ";\n";
    for (std::uint32_t half = 0; half < state.laneCount; ++half) {
        state.half = half;
        const auto h = std::to_string(half);
        text += "    {\n        const uint aps5_i = " + state.GuestInvocationIndex() + ";\n";
        text += "        if (aps5_i < aps5_vertices && aps5_i >= aps5_base && aps5_i - aps5_base < " + split + ") {\n            Aps5MeshVertex aps5_v = aps5_mv[" + h + "];\n";
        text += "#if defined(" + std::string(MslAbi::PointsMacro) + ") && " + std::string(MslAbi::PointsMacro) + "\n            aps5_v.aps5_default_point_size = 1.0f;\n#endif\n";
        text += WindowSpaceEpilogue("aps5_v.aps5_position", "            ");
        text += "#if defined(APS5_DEPTH_NEGATIVE_ONE_TO_ONE) && APS5_DEPTH_NEGATIVE_ONE_TO_ONE\n";
        text += "            aps5_v.aps5_position.z = (aps5_v.aps5_position.z + aps5_v.aps5_position.w) * 0.5f;\n#endif\n";
        text += "#if (APS5_BARYCENTRICS & 1)\n            aps5_v.aps5_bw = float3(1.0f, 0.0f, 0.0f);\n#endif\n";
        text += "#if (APS5_BARYCENTRICS & 2)\n            aps5_v.aps5_bwl = float3(1.0f, 0.0f, 0.0f);\n#endif\n";
        for (const auto& stageOutput : state.program.Info().outputs) {
            if (stageOutput.kind != StageOutputKind::Parameter) continue;
            text += "#if " + PerVertexLocationCondition(stageOutput.location) + "\n            aps5_v." + PerVertexField(stageOutput.location, 0u) + " = aps5_v.p" + std::to_string(stageOutput.location) + ";\n#endif\n";
        }
        text += "            aps5_mesh.set_vertex(aps5_i - aps5_base, aps5_v);\n        }\n";
        text += "        if (aps5_i < aps5_primitives) {\n            const uint aps5_packed = aps5_mprim[" + h + "];\n";
        text += "            const uint aps5_k = min(extract_bits(aps5_packed, 0u, 9u), aps5_last);\n";
        text += "            if (aps5_k >= aps5_base && aps5_k - aps5_base < " + split + ") {\n";
        text += "                const uint aps5_slot = atomic_fetch_add_explicit(&aps5_point_count, 1u, memory_order_relaxed);\n";
        text += "                aps5_mesh.set_index(aps5_slot, uchar(aps5_k - aps5_base));\n";
        text += "                Aps5MeshPrimitive aps5_p = {};\n                aps5_p.aps5_culled = (aps5_packed & 0x80000000u) != 0u;\n";
        if (layer) text += "                aps5_p.aps5_layer = aps5_mlayer[aps5_k];\n";
        text += "                aps5_mesh.set_primitive(aps5_slot, aps5_p);\n            }\n        }\n    }\n";
    }
    state.half = 0u;
    text += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    text += "    if (" + local + " == 0u) aps5_mesh.set_primitive_count(atomic_load_explicit(&aps5_point_count, memory_order_relaxed));\n";
    return text;
}

std::string MeshTriangleVerticesEpilogue(MslEmitterState& state, bool layer, std::uint32_t ldsBytes) {
    const auto& mesh = state.inputInfo.vertex->mesh;
    const auto split = MslAbi::MeshTriangleVerticesPrimitives(mesh.maxPrimitives);
    const auto threadCounts = MslThreadsPerThreadgroup(mesh, state.laneCount);
    const auto threads = threadCounts[0] * threadCounts[1] * threadCounts[2];
    std::uint32_t parameters = 0u;
    for (const auto& stageOutput : state.program.Info().outputs) parameters += stageOutput.kind == StageOutputKind::Parameter ? 1u : 0u;
    // Aps5MeshVertex: position and parameters, plus at most two float3 weight sets (16 bytes each).
    const auto vertexBytes = 16u * (1u + parameters) + 32u;
    // The guest's LDS is dead once the body returned, so the staging reuses it (aps5_lds_words is
    // declared at the larger of the two in this mode; DECISIONS D77): ASTRO BOT's worldmap meshes
    // stage 80 vertices of 224 bytes beside 16 KiB of LDS.
    const auto total = (ldsBytes != 0u ? std::max(MeshTriangleVerticesStagingBytes(state), ldsBytes) : mesh.maxVertices * vertexBytes + mesh.maxPrimitives * 4u) + 8u + (layer ? mesh.maxVertices * 4u : 0u);
    if (total > state.target.maxThreadgroupMemoryBytes) {
        return "#error \"the triangle-vertices mesh function (per-vertex fragment inputs of a mesh draw) stages " + std::to_string(mesh.maxVertices) + " vertices of " + std::to_string(vertexBytes) + " bytes in threadgroup memory: " + std::to_string(total) + " bytes with LDS exceed the " + std::to_string(state.target.maxThreadgroupMemoryBytes) + "-byte limit\"\n";
    }
    const auto local = state.Builtin(MslBuiltin::ThreadIndexInThreadgroup);
    const auto group = state.Builtin(MslBuiltin::ThreadgroupPositionInGrid);
    const auto provoking = std::to_string(std::min(mesh.provokingVertex, 2u));
    // The provoking corner: the draw's last-vertex convention (APS5_PROVOKING_LAST, PA_SU_SC_MODE_CNTL
    // PROVOKING_VTX_LAST; State.cpp leaves the mesh configuration's provokingVertex at 0), else the
    // configuration's.
    std::string text = "#if defined(" + std::string(MslAbi::ProvokingLastMacro) + ") && " + std::string(MslAbi::ProvokingLastMacro) + "\n    constexpr uint aps5_provoking = 2u;\n#else\n    constexpr uint aps5_provoking = " + provoking + "u;\n#endif\n";
    // In the LDS the records are raw 16-byte words, position then the parameters in output order (a
    // threadgroup Aps5MeshVertex pointer cast from the words crashed MTLCompilerService).
    std::vector<std::uint32_t> locations;
    for (const auto& stageOutput : state.program.Info().outputs) {
        if (stageOutput.kind == StageOutputKind::Parameter) locations.push_back(stageOutput.location);
    }
    const auto stride = std::to_string(1u + locations.size());
    const auto store = [&](const std::string& slot, const std::string& vertex) {
        std::string out = "            aps5_lds_words[" + slot + " * " + stride + "u] = as_type<uint4>(" + vertex + ".aps5_position);\n";
        for (std::size_t j = 0; j < locations.size(); ++j) out += "            aps5_lds_words[" + slot + " * " + stride + "u + " + std::to_string(1u + j) + "u] = as_type<uint4>(" + vertex + ".p" + std::to_string(locations[j]) + ");\n";
        return out;
    };
    const auto load = [&](const std::string& slot, const std::string& vertex) {
        std::string out = "            Aps5MeshVertex " + vertex + " = {};\n            " + vertex + ".aps5_position = as_type<float4>(aps5_lds_words[" + slot + " * " + stride + "u]);\n";
        for (std::size_t j = 0; j < locations.size(); ++j) out += "            " + vertex + ".p" + std::to_string(locations[j]) + " = as_type<float4>(aps5_lds_words[" + slot + " * " + stride + "u + " + std::to_string(1u + j) + "u]);\n";
        out += "#if defined(" + std::string(MslAbi::PointsMacro) + ") && " + std::string(MslAbi::PointsMacro) + "\n            " + vertex + ".aps5_default_point_size = 1.0f;\n#endif\n";
        return out;
    };
    const bool raw = ldsBytes != 0u;
    if (raw) {
        text += "    threadgroup uint* aps5_tv_p = aps5_lds + " + std::to_string(mesh.maxVertices * 4u * (1u + locations.size())) + "u;\n";
    } else {
        text += "    threadgroup Aps5MeshVertex aps5_tv_v[" + std::to_string(mesh.maxVertices) + "];\n";
        text += "    threadgroup uint aps5_tv_p[" + std::to_string(mesh.maxPrimitives) + "];\n";
    }
    text += "    const uint aps5_vertices = min(aps5_malloc[0], " + std::to_string(mesh.maxVertices) + "u);\n";
    text += "    const uint aps5_primitives = min(aps5_malloc[1], " + std::to_string(mesh.maxPrimitives) + "u);\n";
    text += "    const uint aps5_last = max(aps5_vertices, 1u) - 1u;\n";
    text += "    const uint aps5_first = " + group + ".z * " + std::to_string(split) + "u;\n";
    text += "    const uint aps5_count = aps5_primitives > aps5_first ? min(aps5_primitives - aps5_first, " + std::to_string(split) + "u) : 0u;\n";
    text += "    if (" + local + " == 0u) aps5_mesh.set_primitive_count(aps5_count);\n";
    for (std::uint32_t half = 0; half < state.laneCount; ++half) {
        state.half = half;
        const auto h = std::to_string(half);
        text += "    {\n        const uint aps5_i = " + state.GuestInvocationIndex() + ";\n";
        text += "        if (aps5_i < aps5_vertices) {\n            Aps5MeshVertex aps5_v = aps5_mv[" + h + "];\n#if defined(" + std::string(MslAbi::PointsMacro) + ") && " + std::string(MslAbi::PointsMacro) + "\n            aps5_v.aps5_default_point_size = 1.0f;\n#endif\n";
        text += WindowSpaceEpilogue("aps5_v.aps5_position", "            ");
        text += "#if defined(APS5_DEPTH_NEGATIVE_ONE_TO_ONE) && APS5_DEPTH_NEGATIVE_ONE_TO_ONE\n";
        text += "            aps5_v.aps5_position.z = (aps5_v.aps5_position.z + aps5_v.aps5_position.w) * 0.5f;\n#endif\n";
        text += raw ? store("aps5_i", "aps5_v") + "        }\n" : std::string("            aps5_tv_v[aps5_i] = aps5_v;\n        }\n");
        text += "        if (aps5_i < aps5_primitives) aps5_tv_p[aps5_i] = aps5_mprim[" + h + "];\n    }\n";
    }
    state.half = 0u;
    text += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    text += "    for (uint aps5_t = " + local + "; aps5_t < aps5_count; aps5_t += " + std::to_string(threads) + "u) {\n";
    text += "        const uint aps5_packed = aps5_tv_p[aps5_first + aps5_t];\n";
    // Points and lines (MslAbi Version 16, DECISIONS D75): the export's first one or two indices,
    // rotated within them; corners past them repeat the last.
    text += "        uint aps5_k[3];\n";
    text += "        constexpr uint aps5_n = " + std::string(MslAbi::PrimitiveVerticesMacro) + ";\n";
    text += "        for (uint c = 0u; c < 3u; ++c) aps5_k[c] = min(extract_bits(aps5_packed, ((min(c, aps5_n - 1u) + min(aps5_provoking, aps5_n - 1u)) % aps5_n) * 10u, 9u), aps5_last);\n";
    text += "        Aps5MeshPrimitive aps5_p = {};\n        aps5_p.aps5_culled = (aps5_packed & 0x80000000u) != 0u;\n";
    if (layer) text += "        aps5_p.aps5_layer = aps5_mlayer[aps5_k[0]];\n";
    for (std::uint32_t corner = 0; corner < 3u; ++corner) {
        const auto c = std::to_string(corner);
        const auto weights = std::string("float3(") + (corner == 0u ? "1.0f" : "0.0f") + ", " + (corner == 1u ? "1.0f" : "0.0f") + ", " + (corner == 2u ? "1.0f" : "0.0f") + ")";
        text += "        {\n" + (raw ? load("aps5_k[" + c + "]", "aps5_v") : "            Aps5MeshVertex aps5_v = aps5_tv_v[aps5_k[" + c + "]];\n");
        text += "#if (APS5_BARYCENTRICS & 1)\n            aps5_v.aps5_bw = " + weights + ";\n#endif\n";
        text += "#if (APS5_BARYCENTRICS & 2)\n            aps5_v.aps5_bwl = " + weights + ";\n#endif\n";
        for (const auto& stageOutput : state.program.Info().outputs) {
            if (stageOutput.kind != StageOutputKind::Parameter) continue;
            text += "#if " + PerVertexLocationCondition(stageOutput.location) + "\n";
            // Points and lines output only their own corners (DECISIONS D78).
            const auto assignment = "            aps5_p." + PerVertexField(stageOutput.location, corner) + " = aps5_v.p" + std::to_string(stageOutput.location) + ";\n";
            text += (corner == 0u ? assignment : "#if APS5_PRIMITIVE_VERTICES > " + std::to_string(corner) + "u\n" + assignment + "#endif\n") + "#endif\n";
        }
        text += "            if (" + c + "u < aps5_n) {\n";
        text += "                aps5_mesh.set_vertex(aps5_t * aps5_n + " + c + "u, aps5_v);\n";
        text += "                aps5_mesh.set_index(aps5_t * aps5_n + " + c + "u, uchar(aps5_t * aps5_n + " + c + "u));\n            }\n        }\n";
    }
    text += "        aps5_mesh.set_primitive(aps5_t, aps5_p);\n    }\n";
    return text;
}

std::string EmitMeshFunction(MslEmitterState& state, std::vector<std::string> parameters) {
    const auto graphics = MslGraphicsInterfaceOf(state);
    const auto& mesh = state.inputInfo.vertex->mesh;
    const auto output = MslMeshOutputOf(mesh);
    std::uint32_t lds = 0u;
    if (state.LdsInDeviceMemory()) state.Fail("mesh functions keep LDS in threadgroup memory; device-memory LDS is not implemented for them");
    const bool layer = std::any_of(state.program.Info().outputs.begin(), state.program.Info().outputs.end(), [](const StageOutput& output) { return output.kind == StageOutputKind::Layer; });
    std::string text = graphics.declarations;
    std::vector<std::string> helperParameters;
    std::vector<std::string> arguments;
    for (const auto& parameter : parameters) {
        const auto attribute = parameter.find(" [[");
        const auto declaration = parameter.substr(0, attribute);
        helperParameters.push_back(declaration);
        arguments.push_back(declaration.substr(declaration.find_last_of(" &*") + 1u));
    }
    std::string prologue;
    std::string hoisted;
    std::size_t start = 0;
    while (start < state.prologue.size()) {
        auto end = state.prologue.find('\n', start);
        if (end == std::string::npos) end = state.prologue.size() - 1u;
        const auto line = state.prologue.substr(start, end + 1u - start);
        start = end + 1u;
        const auto first = line.find_first_not_of(' ');
        if (first != std::string::npos && line.compare(first, 12u, "threadgroup ") == 0) {
            if (line.compare(first, 26u, "threadgroup uint aps5_lds[") != 0) state.Fail("threadgroup declaration in a mesh program's prologue: " + line);
            lds = static_cast<std::uint32_t>(std::stoul(line.substr(first + 26u))) * 4u;
            // 16-byte words, so the triangle-vertices mode can stage its vertex records in them once
            // the body returned (MeshTriangleVerticesEpilogue, DECISIONS D77).
            const auto words = [](std::uint32_t bytes) { return std::to_string((bytes + 15u) / 16u); };
            hoisted += "#if (" + TriangleVerticesCondition() + ") && APS5_PRIMITIVE_VERTICES > 1u\n    threadgroup uint4 aps5_lds_words[" + words(std::max(lds, MeshTriangleVerticesStagingBytes(state))) + "];\n#else\n";
            hoisted += "    threadgroup uint4 aps5_lds_words[" + words(lds) + "];\n#endif\n";
            hoisted += "    threadgroup uint* aps5_lds = reinterpret_cast<threadgroup uint*>(aps5_lds_words);\n";
            helperParameters.push_back("threadgroup uint* aps5_lds");
            arguments.push_back("aps5_lds");
            continue;
        }
        prologue += line;
    }
    const auto lanes = std::to_string(state.laneCount);
    helperParameters.push_back("thread Aps5MeshVertex* aps5_mv");
    helperParameters.push_back("thread uint* aps5_mprim");
    helperParameters.push_back("threadgroup uint* aps5_malloc");
    arguments.insert(arguments.end(), {"aps5_mv", "aps5_mprim", "aps5_malloc"});
    if (layer) {
        helperParameters.push_back("threadgroup uint* aps5_mlayer");
        arguments.push_back("aps5_mlayer");
    }
    text += "static void aps5_body(";
    for (std::size_t index = 0; index < helperParameters.size(); ++index) text += (index == 0u ? "\n    " : ",\n    ") + helperParameters[index];
    text += ")\n{\n" + MslFunctionStatements(state, prologue) + "}\n\n";
    text += "[[mesh]] void " + std::string(MslAbi::EntryPoint) + "(\n    Aps5Mesh aps5_mesh";
    for (const auto& parameter : parameters) text += ",\n    " + parameter;
    text += ")\n{\n" + hoisted;
    text += "    threadgroup uint aps5_malloc[2];\n";
    if (layer) text += "    threadgroup uint aps5_mlayer[" + std::to_string(mesh.maxVertices) + "];\n";
    text += "    Aps5MeshVertex aps5_mv[" + lanes + "] = {};\n";
    text += "    uint aps5_mprim[" + lanes + "] = {" + (state.laneCount == 2u ? std::string("0x80000000u, 0x80000000u") : std::string("0x80000000u")) + "};\n";
    const auto local = state.Builtin(MslBuiltin::ThreadIndexInThreadgroup);
    text += "    if (" + local + " == 0u) {\n        aps5_malloc[0] = 0u;\n        aps5_malloc[1] = 0u;\n    }\n";
    text += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    std::string call = "    aps5_body(";
    for (std::size_t index = 0; index < arguments.size(); ++index) call += (index == 0u ? "" : ", ") + arguments[index];
    text += call + ");\n";
    text += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    if (output.rectList) return text + MeshRectListEpilogue(state, output, layer, lds) + "}\n";
    text += "#if (" + TriangleVerticesCondition() + ") && APS5_PRIMITIVE_VERTICES == 1u\n" + MeshPointVerticesEpilogue(state, layer) + "#elif " + TriangleVerticesCondition() + "\n" + MeshTriangleVerticesEpilogue(state, layer, lds) + "#else\n";
    text += "#if defined(" + std::string(MslAbi::ProvokingLastMacro) + ") && " + std::string(MslAbi::ProvokingLastMacro) + "\n#define APS5_LINE_PROVOKING_LAST true\n#else\n#define APS5_LINE_PROVOKING_LAST false\n#endif\n";
    text += "    const uint aps5_vertices = min(aps5_malloc[0], " + std::to_string(mesh.maxVertices) + "u);\n";
    text += "    const uint aps5_primitives = min(aps5_malloc[1], " + std::to_string(mesh.maxPrimitives) + "u);\n";
    text += "    if (" + local + " == 0u) aps5_mesh.set_primitive_count(aps5_primitives);\n";
    text += "    const uint aps5_last = max(aps5_vertices, 1u) - 1u;\n";
    for (std::uint32_t half = 0; half < state.laneCount; ++half) {
        state.half = half;
        const auto h = std::to_string(half);
        text += "    {\n        const uint aps5_i = " + state.GuestInvocationIndex() + ";\n";
        text += "        if (aps5_i < aps5_vertices) {\n            Aps5MeshVertex aps5_v = aps5_mv[" + h + "];\n#if defined(" + std::string(MslAbi::PointsMacro) + ") && " + std::string(MslAbi::PointsMacro) + "\n            aps5_v.aps5_default_point_size = 1.0f;\n#endif\n";
        // Metal has no depth clip control: a [-w, w] guest depth range is remapped to [0, w].
        text += WindowSpaceEpilogue("aps5_v.aps5_position", "            ");
        text += "#if defined(APS5_DEPTH_NEGATIVE_ONE_TO_ONE) && APS5_DEPTH_NEGATIVE_ONE_TO_ONE\n";
        text += "            aps5_v.aps5_position.z = (aps5_v.aps5_position.z + aps5_v.aps5_position.w) * 0.5f;\n#endif\n";
        text += "            aps5_mesh.set_vertex(aps5_i, aps5_v);\n        }\n";
        text += "        if (aps5_i < aps5_primitives) {\n            const uint aps5_packed = aps5_mprim[" + h + "];\n";
        // A line drawn with the last vertex provoking runs from its second vertex (Metal's flat
        // values come from the first), as the vertex path's expansion draws it (DECISIONS D75).
        text += "            constexpr bool aps5_reverse = " + std::string(MslAbi::PrimitiveVerticesMacro) + " == 2u && " + "APS5_LINE_PROVOKING_LAST;\n";
        text += "            uint aps5_k[3] = {};\n";
        text += "            for (uint c = 0u; c < " + std::string(MslAbi::PrimitiveVerticesMacro) + "; ++c) {\n                aps5_k[c] = min(extract_bits(aps5_packed, (aps5_reverse ? 1u - c : c) * 10u, 9u), aps5_last);\n";
        text += "                aps5_mesh.set_index(aps5_i * " + std::string(MslAbi::PrimitiveVerticesMacro) + " + c, uchar(aps5_k[c]));\n            }\n";
        text += "            Aps5MeshPrimitive aps5_p = {};\n            aps5_p.aps5_culled = (aps5_packed & 0x80000000u) != 0u;\n";
        if (layer) text += "            aps5_p.aps5_layer = aps5_mlayer[aps5_k[min(" + std::to_string(std::min(mesh.provokingVertex, 2u)) + "u, " + std::string(MslAbi::PrimitiveVerticesMacro) + " - 1u)]];\n";
        text += "            aps5_mesh.set_primitive(aps5_i, aps5_p);\n        }\n    }\n";
    }
    state.half = 0u;
    text += "#endif\n}\n";
    return text;
}

}

std::string EmitMslEntryPoint(MslEmitterState& state) {
    // The mesh function's epilogue needs the invocation index whether or not the guest reads it.
    // (and the grid position: the triangle-vertices mode's split index is its z, MslAbi Version 13).
    if (state.reflection.stage == MslAbi::Stage::Mesh) {
        static_cast<void>(state.Builtin(MslBuiltin::ThreadIndexInThreadgroup));
        static_cast<void>(state.Builtin(MslBuiltin::ThreadgroupPositionInGrid));
    }
    std::vector<MslAbi::TableBinding> tableBindings;
    for (const auto& descriptor : state.program.Metadata().bindings.descriptors) {
        if (!state.usedKinds.contains(descriptor.kind)) continue;
        tableBindings.push_back({descriptor.kind, std::max<std::uint32_t>(static_cast<std::uint32_t>(descriptor.resources.size()), 1u)});
    }
    state.reflection.table = MslAbi::LayoutTable(tableBindings);
    if (state.workgroup != nullptr) state.reflection.threadsPerThreadgroup = MslThreadsPerThreadgroup(*state.workgroup, state.laneCount);

    std::string text;
    for (const auto& [id, defaultValue] : state.functionConstants) {
        const auto name = "aps5_fc" + std::to_string(id);
        text += "constant uint " + name + "_value [[function_constant(" + std::to_string(id) + ")]];\n";
        text += "constant uint " + name + " = is_function_constant_defined(" + name + "_value) ? " + name + "_value : " + MslHex(defaultValue) + "u;\n";
    }
    if (!state.functionConstants.empty()) text += "\n";
    if (!state.declarations.empty()) text += state.declarations + "\n";

    std::vector<std::string> parameters;
    if (state.reflection.table.bytes != 0u) {
        state.RequirePrelude(MslPreludeChunk::Common);
        text += MslTableDeclaration(state.reflection.table);
        parameters.push_back("device const Aps5Table& aps5_table [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::DescriptorTable)) + ")]]");
    }
    if (state.usesPushData) parameters.push_back("constant uint* aps5_push [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::PushData)) + ")]]");
    if (state.usesDriverData) {
        state.RequirePrelude(MslPreludeChunk::Common);
        parameters.push_back("constant Aps5DriverData& aps5_driver [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::DriverData)) + ")]]");
    }
    const bool tessellation = MslTessellationStage(state.program.Resources().stage);
    if (tessellation) PrepareMslTessellationStage(state);
    if (state.reflection.stage == MslAbi::Stage::Vertex && state.builtins.any()) {
        // Vertex functions have no [[thread_index_in_simdgroup]]: the lane is the number of active
        // lanes below this one (the hardware lane while the SIMD group is full, as vertex groups are
        // except at the draw's end).
        if (state.builtins.count() != 1u || !state.builtins.test(static_cast<std::size_t>(MslBuiltin::ThreadIndexInSimdgroup))) state.Fail("threadgroup builtins are not available to vertex functions (work package B6)");
        state.builtins.reset();
        state.prologue = "    const uint aps5_simd_lane = simd_prefix_exclusive_sum(1u);\n" + state.prologue;
    }
    for (std::size_t index = 0; index < state.builtins.size(); ++index) {
        if (state.builtins.test(index)) parameters.push_back(BuiltinParameter(static_cast<MslBuiltin>(index)));
    }
    // PS_ITER_SAMPLE: a [[sample_id]] input makes Metal run the fragment function once per sample.
    if (state.reflection.stage == MslAbi::Stage::Fragment && state.inputInfo.pixel != nullptr && state.inputInfo.pixel->psSampleShading) state.AddParameter("uint aps5_sample_id [[sample_id]]");
    parameters.insert(parameters.end(), state.parameters.begin(), state.parameters.end());

    if (state.reflection.stage == MslAbi::Stage::Kernel) {
        text += "kernel void " + std::string(MslAbi::EntryPoint) + "(";
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            text += (index == 0u ? "\n    " : ",\n    ") + parameters[index];
        }
        text += ")\n{\n" + MslFunctionStatements(state, state.prologue) + "}\n";
        return text;
    }
    if (state.reflection.stage == MslAbi::Stage::Mesh) return text + EmitMeshFunction(state, std::move(parameters));
    if (tessellation) return text + EmitMslTessellationStage(state, std::move(parameters), state.reflection.table.bytes != 0u);
    if (state.reflection.stage != MslAbi::Stage::Vertex && state.reflection.stage != MslAbi::Stage::Fragment) state.Fail("only kernel, vertex and fragment stage functions are implemented in the MSL backend so far (work package B6)");
    // Vertex and fragment functions (B6): the guest body runs in aps5_body, which takes every stage
    // parameter (without its attribute) and the output struct by reference, so the body's returns
    // end the guest program and the stage function applies the epilogue (MslGraphicsInterfaceOf).
    const bool fragment = state.reflection.stage == MslAbi::Stage::Fragment;
    if (fragment) parameters.insert(parameters.begin(), "Aps5FragmentIn aps5_in [[stage_in]]");
    if (!fragment) {
        // Rect lists (APS5_RECT_LIST) need the host vertex id and the guest first vertex.
        const std::string vertexId = "uint aps5_vertex_id [[vertex_id]]";
        if (std::find(parameters.begin(), parameters.end(), vertexId) == parameters.end()) parameters.push_back(vertexId);
        const std::string driver = "constant Aps5DriverData& aps5_driver [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::DriverData)) + ")]]";
        if (std::find(parameters.begin(), parameters.end(), driver) == parameters.end()) {
            state.RequirePrelude(MslPreludeChunk::Common);
            state.reflection.usesDriverData = true;
            parameters.push_back(driver);
        }
    }
    const auto graphics = MslGraphicsInterfaceOf(state);
    text += graphics.declarations;
    std::vector<std::string> helperParameters;
    std::vector<std::string> arguments;
    for (const auto& parameter : parameters) {
        const auto attribute = parameter.find(" [[");
        const auto declaration = parameter.substr(0, attribute);
        const auto name = declaration.substr(declaration.find_last_of(" &*") + 1u);
        helperParameters.push_back(declaration);
        arguments.push_back(name);
    }
    if (!graphics.outputType.empty()) {
        helperParameters.push_back("thread " + graphics.outputType + "& aps5_out");
        arguments.push_back("aps5_out");
    }
    if (state.usesValidMask) {
        helperParameters.push_back("thread uint& aps5_valid");
        arguments.push_back("aps5_valid");
    }
    // MslAbi Version 10: the guest cull distances of each guest vertex the function runs (rect lists
    // and triangle-vertices draws run three).
    const auto culls = fragment ? 0u : MslCullDistanceCount(state);
    if (culls != 0u) {
        helperParameters.push_back("thread float* aps5_cull");
        arguments.push_back("aps5_cull");
    }
    // Primitive-ordered pixel shaders: the raster order group buffer is a stage-function parameter
    // only (the guest body never touches it).
    const bool rasterOrder = fragment && state.reflection.rasterOrder;
    if (rasterOrder) parameters.push_back("device uint* aps5_rog [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::RasterOrder)) + "), raster_order_group(0)]]");
    // Triangle-vertices draws (MslAbi Version 5): the TriangleVerticesHeader and the guest vertex ids,
    // a stage-function parameter only in that mode (other draws bind nothing at the slot).
    if (!fragment) {
        text += "struct Aps5TriangleVertices {\n    float4 viewport;\n    float2 depthRange;\n    uint triangles;\n    uint flags;\n};\n";
        text += "static_assert(sizeof(Aps5TriangleVertices) == 32, \"MslAbi::TriangleVerticesHeader\");\n\n";
        parameters.back() += "\n#if " + TriangleVerticesCondition() + "\n    , device const uint* aps5_triangles [[buffer(" + std::to_string(MslAbi::Index(MslAbi::BufferIndex::TriangleVertices)) + ")]]\n#endif\n";
    }
    text += "static void aps5_body(";
    for (std::size_t index = 0; index < helperParameters.size(); ++index) text += (index == 0u ? "\n    " : ",\n    ") + helperParameters[index];
    text += ")\n{\n" + MslFunctionStatements(state, state.prologue) + "}\n\n";
    const auto returnType = graphics.outputType.empty() ? std::string("void") : graphics.outputType;
    text += graphics.attributes + (fragment ? "fragment " : "vertex ") + returnType + " " + std::string(MslAbi::EntryPoint) + "(";
    for (std::size_t index = 0; index < parameters.size(); ++index) text += (index == 0u ? "\n    " : ",\n    ") + parameters[index];
    text += ")\n{\n";
    if (!graphics.outputType.empty()) text += "    " + graphics.outputType + " aps5_out = {};\n";
    if (state.usesValidMask) text += "    uint aps5_valid = 1u;\n";
    if (culls != 0u) text += "    float aps5_cull[3][" + std::to_string(culls) + "] = {};\n";
    // Alpha to coverage starts from full coverage, as the SPIR-V path's StoreCoverage(-1): only a
    // float MRT0 export lowers it, so uint exports keep every sample.
    if (fragment && state.inputInfo.pixel != nullptr && state.inputInfo.pixel->psAlphaToCoverage) text += "    aps5_out.aps5_sample_mask = 0xffffffffu;\n";
    if (fragment) text += "#if defined(APS5_CONSERVATIVE) && APS5_CONSERVATIVE\n" + MslConservative::FragmentText() + "#endif\n";
    const auto call = [&](const std::string& vertexId, const std::string& output, std::uint32_t vertex = 0u) {
        std::string line = "aps5_body(";
        for (std::size_t index = 0; index < arguments.size(); ++index) {
            auto argument = arguments[index] == "aps5_vertex_id" ? vertexId : arguments[index] == "aps5_out" ? output : arguments[index] == "aps5_cull" ? "aps5_cull[" + std::to_string(vertex) + "]" : arguments[index];
            line += (index == 0u ? "" : ", ") + argument;
        }
        return line + ");";
    };
    if (!fragment) {
        // APS5_RECT_LIST: 6 host vertices per rect (two triangles); each runs the guest program for
        // the rect's 3 guest vertices (DriverData.workgroupBase[0] = the guest first vertex) and
        // emits one corner, the fourth derived as on the SPIR-V path's rect-list TCS: the right-angle
        // vertex gets weight -1, the others +1; an invalid rect is culled (zero position).
        text += "#if defined(APS5_RECT_LIST) && APS5_RECT_LIST\n";
        text += "    const uint aps5_rect = aps5_vertex_id / 6u;\n";
        text += "    constexpr uint aps5_corners[6] = {0u, 1u, 2u, 2u, 1u, 3u};\n";
        text += "    const uint aps5_corner = aps5_corners[aps5_vertex_id % 6u];\n";
        text += "    Aps5VertexOut aps5_v[3] = {};\n";
        for (std::uint32_t vertex = 0; vertex < 3u; ++vertex) {
            text += "    " + call("aps5_driver.workgroupBase[0] + aps5_rect * 3u + " + std::to_string(vertex) + "u", "aps5_v[" + std::to_string(vertex) + "]", vertex) + "\n";
        }
        text += "    bool aps5_edge[3];\n    float aps5_weight[3];\n    uint aps5_edges = 0u;\n";
        text += "    for (uint i = 0u; i < 3u; ++i) {\n";
        text += "        const uint next = (i + 1u) % 3u, previous = (i + 2u) % 3u;\n";
        text += "        const bool xy = aps5_v[i].aps5_position.x == aps5_v[next].aps5_position.x && aps5_v[i].aps5_position.y == aps5_v[previous].aps5_position.y;\n";
        text += "        const bool yx = aps5_v[i].aps5_position.y == aps5_v[next].aps5_position.y && aps5_v[i].aps5_position.x == aps5_v[previous].aps5_position.x;\n";
        text += "        aps5_edge[i] = xy || yx;\n        aps5_weight[i] = aps5_edge[i] ? -1.0f : 1.0f;\n        aps5_edges += aps5_edge[i] ? 1u : 0u;\n    }\n";
        text += "    const uint aps5_first = aps5_edge[1] ? 1u : (aps5_edge[2] ? 2u : 0u);\n";
        text += "    if (aps5_corner < 3u) {\n        aps5_out = aps5_v[(aps5_first + aps5_corner) % 3u];\n    } else {\n        aps5_out = aps5_v[aps5_first];\n";
        text += "        aps5_out.aps5_position = aps5_v[0].aps5_position * aps5_weight[0] + aps5_v[1].aps5_position * aps5_weight[1] + aps5_v[2].aps5_position * aps5_weight[2];\n";
        for (const auto& output : state.program.Info().outputs) {
            if (output.kind != StageOutputKind::Parameter) continue;
            const auto field = "p" + std::to_string(output.location);
            text += "        aps5_out." + field + " = aps5_v[0]." + field + " * aps5_weight[0] + aps5_v[1]." + field + " * aps5_weight[1] + aps5_v[2]." + field + " * aps5_weight[2];\n";
        }
        text += "    }\n";
        text += "    const float aps5_w = aps5_v[0].aps5_position.w;\n";
        text += "    const bool aps5_valid_rect = aps5_edges == 1u && aps5_w > 0.0f && aps5_v[1].aps5_position.w == aps5_w && aps5_v[2].aps5_position.w == aps5_w && all(isfinite(aps5_out.aps5_position));\n";
        text += "    if (!aps5_valid_rect) aps5_out.aps5_position = float4(0.0f);\n";
        // APS5_TRIANGLE_VERTICES (MslAbi Version 5): 3 host vertices per guest triangle, non-indexed;
        // each runs the guest program for its triangle's 3 guest vertices (TriangleVerticesHeader,
        // then the ids) and emits its corner, plus the corner values of the per-vertex locations,
        // one-hot barycentric weights and the conservative-rasterization outputs the macros ask for.
        text += "#elif " + TriangleVerticesCondition() + "\n";
        text += "    device const Aps5TriangleVertices& aps5_tv = *reinterpret_cast<device const Aps5TriangleVertices*>(aps5_triangles);\n";
        text += "    device const uint* aps5_tv_ids = aps5_triangles + 8u;\n";
        // MslAbi Version 14: points and lines draw in the same mode with PrimitiveVerticesMacro guest
        // vertices per primitive (1 or 2, default 3); the corners past them repeat the last vertex.
        const std::string primitiveVertices = MslAbi::PrimitiveVerticesMacro;
        text += "#ifndef " + primitiveVertices + "\n#define " + primitiveVertices + " 3u\n#endif\n";
        text += "    const uint aps5_triangle = aps5_vertex_id / " + primitiveVertices + ";\n";
        text += "    const uint aps5_corner = aps5_vertex_id % " + primitiveVertices + ";\n";
        text += "    Aps5VertexOut aps5_v[3] = {};\n";
        for (std::uint32_t vertex = 0; vertex < 3u; ++vertex) {
            const auto slot = "aps5_v[" + std::to_string(vertex) + "]";
            const auto guest = call("aps5_tv_ids[aps5_triangle * " + primitiveVertices + " + " + std::to_string(vertex) + "u]", slot, vertex);
            if (vertex == 0u) {
                text += "    " + guest + "\n";
                continue;
            }
            text += "#if " + primitiveVertices + " > " + std::to_string(vertex) + "u\n    " + guest + "\n#else\n    " + slot + " = aps5_v[" + std::to_string(vertex - 1u) + "];\n#endif\n";
        }
        text += "    aps5_out = aps5_v[aps5_corner];\n";
        for (const auto& output : state.program.Info().outputs) {
            if (output.kind != StageOutputKind::Parameter) continue;
            text += "#if " + PerVertexLocationCondition(output.location) + "\n";
            for (std::uint32_t vertex = 0; vertex < 3u; ++vertex) {
                const auto assignment = "    aps5_out." + PerVertexField(output.location, vertex) + " = aps5_v[" + std::to_string(vertex) + "].p" + std::to_string(output.location) + ";\n";
                text += vertex == 0u ? assignment : "#if APS5_PRIMITIVE_VERTICES > " + std::to_string(vertex) + "u\n" + assignment + "#endif\n";
            }
            text += "#endif\n";
        }
        text += "    const float3 aps5_weights = float3(aps5_corner == 0u ? 1.0f : 0.0f, aps5_corner == 1u ? 1.0f : 0.0f, aps5_corner == 2u ? 1.0f : 0.0f);\n";
        text += "#if (APS5_BARYCENTRICS & 1)\n    aps5_out.aps5_bw = aps5_weights;\n#endif\n";
        text += "#if (APS5_BARYCENTRICS & 2)\n    aps5_out.aps5_bwl = aps5_weights;\n#endif\n";
        text += "#if defined(APS5_CONSERVATIVE) && APS5_CONSERVATIVE\n" + MslConservative::VertexText() + "#endif\n";
        text += "#else\n    " + call("aps5_vertex_id", "aps5_out") + "\n#endif\n";
        if (culls != 0u) {
            // Upstream #423 cull distances (MslAbi Version 10): Metal has none, so the primitive's
            // verdict becomes its last clip distance, -1 at every vertex of a culled primitive. A
            // triangle (triangle-vertices draws, and rect lists over their three guest vertices) is
            // culled when all its vertices have a negative distance on one plane, a point (draws
            // the driver compiles with APS5_CULL_POINTS) when its vertex has; NaN is not negative.
            // Lines have no exact emulation: the driver refuses them, and the library does not
            // compile without one of the modes.
            const auto count = std::to_string(culls);
            text += "    bool aps5_culled = false;\n";
            text += "#if (" + TriangleVerticesCondition() + ") || (defined(APS5_RECT_LIST) && APS5_RECT_LIST)\n";
            text += "    for (uint i = 0u; i < " + count + "u; ++i) aps5_culled = aps5_culled || (aps5_cull[0][i] < 0.0f && aps5_cull[1][i] < 0.0f && aps5_cull[2][i] < 0.0f);\n";
            text += "#elif defined(" + std::string(MslAbi::CullPointsMacro) + ") && " + std::string(MslAbi::CullPointsMacro) + "\n";
            text += "    for (uint i = 0u; i < " + count + "u; ++i) aps5_culled = aps5_culled || aps5_cull[0][i] < 0.0f;\n";
            text += "#else\n#error \"cull distances need a triangle-vertices, rect-list or APS5_CULL_POINTS draw (MslAbi Version 10)\"\n#endif\n";
            text += "    aps5_out.aps5_clip[" + std::to_string(MslCullClipSlot(state)) + "] = aps5_culled ? -1.0f : 0.0f;\n";
        }
    } else if (rasterOrder) {
        // The guest program is the critical section, as the SPIR-V path's Begin/End invocation
        // interlock around it: the read of the pixel's raster-order-group dword waits until every
        // earlier overlapping fragment wrote it back, and the write-back after the body releases the
        // next one. The guest's own memory accesses are device pointers made from addresses, which
        // the compiler cannot prove apart from aps5_rog, so they stay between the two accesses; the
        // fences make the previous fragment's stores visible (MslAbi::RasterOrderBytes).
        text += "    const uint aps5_rog_slot = (uint(aps5_in.aps5_position.x) & 63u) | ((uint(aps5_in.aps5_position.y) & 63u) << 6u);\n";
        text += "    const uint aps5_rog_ticket = aps5_rog[aps5_rog_slot];\n";
        text += "    atomic_thread_fence(mem_flags::mem_device, memory_order_acquire, thread_scope_device);\n";
        text += "    " + call("aps5_vertex_id", "aps5_out") + "\n";
        text += "    atomic_thread_fence(mem_flags::mem_device, memory_order_release, thread_scope_device);\n";
        text += "    aps5_rog[aps5_rog_slot] = aps5_rog_ticket + 1u;\n";
    } else {
        text += "    " + call("aps5_vertex_id", "aps5_out") + "\n";
    }
    if (state.reflection.stage == MslAbi::Stage::Vertex && !graphics.outputType.empty() && state.program.Resources().stage == IrShaderStage::Vertex && std::none_of(state.program.Info().outputs.begin(), state.program.Info().outputs.end(), [](const StageOutput& output) { return output.kind == StageOutputKind::PointSize; })) {
        text += "#if defined(" + std::string(MslAbi::PointsMacro) + ") && " + std::string(MslAbi::PointsMacro) + "\n    aps5_out.aps5_default_point_size = 1.0f;\n#endif\n";
    }
    text += graphics.epilogue;
    if (!graphics.outputType.empty()) text += "    return aps5_out;\n";
    text += "}\n";
    return text;
}

}
