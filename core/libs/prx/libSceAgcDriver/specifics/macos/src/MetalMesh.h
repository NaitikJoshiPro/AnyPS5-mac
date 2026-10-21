#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALMESH_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALMESH_H

// Mesh draws on Metal (B6 stage 5, owner SHD4; M2_DESIGN "SHD4 results"): the guest's NGG
// primitive shader runs as an MSL mesh function (MslEntryPoint.cpp EmitMeshFunction) drawn with
// drawMeshThreadgroups over (mesh groups, instances, 1), as upstream's vkCmdDrawMeshTasksEXT path
// (Graphics/src/Draw.cpp). The mesh function reads six draw words at push dwords
// MeshDrawPushOffsetBytes / 4.. (pushDrawConstants): index count, first vertex, first instance,
// index size | MeshIndexRestartTable, and the GPU address of a MeshArguments record followed by the
// strip restart table (length, then one start per index), 0 when the draw needs neither.
// Indirect mesh draws read their first record on the CPU (the Metal draw path's rule for every
// indirect draw) and resolve it with upstream's ResolveMeshArguments.

#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace AgcDriver::Graphics::Metal {

// Metal Feature Set Tables (Apple9): 1,048,575 threadgroups per mesh grid.
inline constexpr std::uint32_t MaxMeshThreadgroupsPerGrid = 1048575u;

struct MeshDrawPlan {
    std::uint32_t groups = 0;
    std::uint32_t instances = 0;
    // Push dwords MeshDrawPushOffsetBytes / 4 + 0..5; words 4 and 5 (the argument address) are
    // filled by the caller once `arguments` is uploaded.
    std::array<std::uint32_t, ShaderRecompiler::MeshDrawPushBytes / 4u> words{};
    // MeshArguments, then the restart table (length, starts); empty when the draw needs neither.
    std::vector<std::byte> arguments;
};

// `indices`: the draw's guest index buffer (settled), empty for non-indexed draws.
inline MeshDrawPlan PlanMeshDraw(const Graphics::State& graphics, const Pm4::DrawParameters& draw, std::span<const std::byte> indices) {
    if (!graphics.stages.mesh) throw std::runtime_error("Metal device: a mesh draw without a mesh configuration");
    const auto& mesh = *graphics.stages.mesh;
    const auto inputSize = mesh.inputPrimitive == 1 ? 1u : mesh.inputPrimitive == 2 ? 2u : 3u;
    const auto step = mesh.inputPrimitive == 5 || mesh.inputPrimitive == 6 ? 1u : inputSize;
    if (mesh.primitivesPerGroup == 0u) throw std::runtime_error("AGC graphics: mesh pipeline groups no primitive (zero primitives per group)");
    MeshDrawPlan plan;
    std::vector<std::uint32_t> restartTable;
    if (draw.indexed) {
        const auto restart = draw.indexSize == 2u ? 0xffffu : 0xffffffffu;
        bool restarts = false;
        for (std::size_t position = 0; graphics.primitiveRestart && position + draw.indexSize <= indices.size(); position += draw.indexSize) {
            std::uint32_t value = 0;
            if (draw.indexSize == 2u) {
                std::uint16_t half = 0;
                std::memcpy(&half, indices.data() + position, 2u);
                value = half;
            } else {
                std::memcpy(&value, indices.data() + position, 4u);
            }
            restarts |= value == restart;
        }
        // As upstream (Draw.cpp prepareDrawInputs): fans cannot restart; strips use a restart table.
        if (restarts && mesh.inputPrimitive == 5) throw std::runtime_error("AGC graphics: primitive restart in a triangle fan geometry draw is unsupported");
        if (restarts && mesh.inputPrimitive == 6) restartTable = MeshRestartTable(indices, draw.indexSize);
        // Lists pass the draw precheck with restart enabled (PrimitiveListRestart: the vertex path
        // expands them on the CPU); the mesh function has no list restart, so a list that holds a
        // restart index throws here instead of fetching vertex 0xffff(ffff).
        if (restarts && step != 1u) throw std::runtime_error("AGC graphics: a restart index inside an indexed list geometry (mesh) draw is unsupported");
    }
    MeshArguments resolved{};
    bool needsArguments = !restartTable.empty();
    if (draw.indirect) {
        const MeshArgumentRules rules{draw.indexCount, inputSize, step, mesh.primitivesPerGroup, MaxMeshThreadgroupsPerGrid, MaxMeshThreadgroupsPerGrid, MaxMeshThreadgroupsPerGrid};
        resolved = ResolveMeshArguments(Pm4::ReadDrawArguments(*draw.indirect, 0), rules);
        plan.groups = resolved.groups;
        plan.instances = resolved.instances;
        needsArguments = true;
    } else {
        // Upstream e2a4d109c0: a draw shorter than one primitive draws nothing (no groups; MetalDraw skips it).
        if (draw.indexCount < inputSize) return plan;
        const auto primitives = (draw.indexCount - inputSize) / step + 1u;
        plan.groups = (primitives - 1u) / mesh.primitivesPerGroup + 1u;
        plan.instances = draw.instanceCount;
        if (plan.groups > MaxMeshThreadgroupsPerGrid || static_cast<std::uint64_t>(plan.groups) * plan.instances > MaxMeshThreadgroupsPerGrid) throw std::runtime_error("AGC graphics: mesh draw exceeds workgroup count limits");
        resolved = MeshArguments{0u, 0u, 0u, draw.indexCount, 0u};
    }
    const auto firstVertex = draw.indirect ? draw.indirect->vertexConstant : draw.firstVertex;
    const auto firstInstance = draw.indirect ? draw.indirect->instanceConstant : draw.firstInstance;
    plan.words = {draw.indexCount, firstVertex, firstInstance, draw.indexed ? draw.indexSize | (restartTable.empty() ? 0u : ShaderRecompiler::MeshIndexRestartTable) : 0u, 0u, 0u};
    if (needsArguments) {
        const auto tableBytes = restartTable.empty() ? 0u : sizeof(std::uint32_t) * (restartTable.size() + 1u);
        plan.arguments.resize(ShaderRecompiler::MeshArgumentBytes + tableBytes);
        std::memcpy(plan.arguments.data(), &resolved, sizeof(resolved));
        if (!restartTable.empty()) {
            const auto length = static_cast<std::uint32_t>(restartTable.size());
            std::memcpy(plan.arguments.data() + ShaderRecompiler::MeshArgumentBytes, &length, sizeof(length));
            std::memcpy(plan.arguments.data() + ShaderRecompiler::MeshArgumentBytes + sizeof(length), restartTable.data(), restartTable.size() * sizeof(std::uint32_t));
        }
    }
    return plan;
}

}

#endif
