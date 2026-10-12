#include "MetalBackend/MslEmitterState.hpp"
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
constexpr const char* TextureShapes[HeapDimensions] = {"texture2d", "texture2d_array", "texture2d", "texture2d_array", "texture2d_ms", "texture2d_ms_array", "texture3d"};
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
    // Metal cannot write multisampled textures from a shader: those heaps are read-only.
    static constexpr const char* Scalars[] = {"float", "uint", "uint", "ulong"};
    const auto index = value - RuntimeAbi::FirstStorageImageBinding;
    const auto dimension = index % HeapDimensions;
    const bool multisampled = dimension == 4u || dimension == 5u;
    return std::string(TextureShapes[dimension]) + "<" + Scalars[index / HeapDimensions] + (multisampled ? ", access::read>" : ", access::read_write>");
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

std::string EmitMslEntryPoint(MslEmitterState& state) {
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
    parameters.insert(parameters.end(), state.parameters.begin(), state.parameters.end());

    if (state.reflection.stage == MslAbi::Stage::Kernel) {
        text += "kernel void " + std::string(MslAbi::EntryPoint) + "(";
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            text += (index == 0u ? "\n    " : ",\n    ") + parameters[index];
        }
        text += ")\n{\n" + state.locals + state.prologue + state.body + "}\n";
        return text;
    }
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
    text += "static void aps5_body(";
    for (std::size_t index = 0; index < helperParameters.size(); ++index) text += (index == 0u ? "\n    " : ",\n    ") + helperParameters[index];
    text += ")\n{\n" + state.locals + state.prologue + state.body + "}\n\n";
    const auto returnType = graphics.outputType.empty() ? std::string("void") : graphics.outputType;
    text += graphics.attributes + (fragment ? "fragment " : "vertex ") + returnType + " " + std::string(MslAbi::EntryPoint) + "(";
    for (std::size_t index = 0; index < parameters.size(); ++index) text += (index == 0u ? "\n    " : ",\n    ") + parameters[index];
    text += ")\n{\n";
    if (!graphics.outputType.empty()) text += "    " + graphics.outputType + " aps5_out = {};\n";
    if (state.usesValidMask) text += "    uint aps5_valid = 1u;\n";
    const auto call = [&](const std::string& vertexId, const std::string& output) {
        std::string line = "aps5_body(";
        for (std::size_t index = 0; index < arguments.size(); ++index) {
            auto argument = arguments[index] == "aps5_vertex_id" ? vertexId : arguments[index] == "aps5_out" ? output : arguments[index];
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
            text += "    " + call("aps5_driver.workgroupBase[0] + aps5_rect * 3u + " + std::to_string(vertex) + "u", "aps5_v[" + std::to_string(vertex) + "]") + "\n";
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
        text += "#else\n    " + call("aps5_vertex_id", "aps5_out") + "\n#endif\n";
    } else {
        text += "    " + call("aps5_vertex_id", "aps5_out") + "\n";
    }
    text += graphics.epilogue;
    if (!graphics.outputType.empty()) text += "    return aps5_out;\n";
    text += "}\n";
    return text;
}

}
