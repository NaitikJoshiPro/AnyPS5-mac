#ifndef CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLTESSELLATOR_HPP
#define CORE_SHADER_RECOMPILER_METALBACKEND_INCLUDE_METALBACKEND_MSLTESSELLATOR_HPP

// The software tessellator of the Metal tessellation path (DECISIONS D24): MslTessellatorCore.inc as
// C++ (ShaderRecompiler::MslTessellator, for the CPU unit test and the driver's planning) and as MSL
// text (MslTessellator::Source, for the generated mesh and object functions). One text, two
// compilers, so the CPU checks cover exactly what the GPU runs.

#include <bit>
#include <cmath>
#include <cstdint>
#include <string>

namespace ShaderRecompiler::MslTessellator {

using uint = std::uint32_t;

#define APS5_TESS_SOURCE(...) __VA_ARGS__
#define APS5_TESS_FN static inline
#define APS5_TESS_TABLE static constexpr
#define APS5_TESS_THREAD
#define APS5_TESS_OUT
#define APS5_TESS_AS_UINT(f) std::bit_cast<uint>(f)
#define APS5_TESS_AS_FLOAT(u) std::bit_cast<float>(static_cast<uint>(u))
#define APS5_TESS_CEIL(f) std::ceil(f)
#include "MetalBackend/MslTessellatorCore.inc"
#undef APS5_TESS_SOURCE
#undef APS5_TESS_FN
#undef APS5_TESS_TABLE
#undef APS5_TESS_THREAD
#undef APS5_TESS_OUT
#undef APS5_TESS_AS_UINT
#undef APS5_TESS_AS_FLOAT
#undef APS5_TESS_CEIL

// VGT_TF_PARAM encodings (TessellationConfiguration): domain TYPE, PARTITIONING, TOPOLOGY.
enum Domain : int { Isoline = 0, Triangle = 1, Quad = 2 };
enum Partitioning : int { Integer = 0, Pow2 = 1, FractionalOdd = 2, FractionalEven = 3 };
enum Topology : int { Point = 0, Line = 1, TriangleCw = 2, TriangleCcw = 3 };

// Primitives one mesh threadgroup emits: Metal's mesh<> holds at most 256 vertices and 512
// primitives (WWDC22 "Transform your geometry with Metal mesh shaders"), and every primitive owns its
// corners' vertices, so 85 triangles (255 vertices), 128 lines or 256 points.
inline constexpr int SlicePrimitives(int topology) {
    return topology == Point ? 256 : topology == Line ? 128 : 85;
}

// The MSL text: the core with its address spaces and bit casts (no preprocessor state leaks: the
// macros are #undef'd after it).
inline std::string Source() {
    static const std::string text = std::string("#define APS5_TESS_FN static inline\n#define APS5_TESS_TABLE constant\n#define APS5_TESS_THREAD thread\n#define APS5_TESS_OUT threadgroup\n"
                                                 "#define APS5_TESS_AS_UINT(f) as_type<uint>(f)\n#define APS5_TESS_AS_FLOAT(u) as_type<float>(uint(u))\n#define APS5_TESS_CEIL(f) ceil(f)\n") +
#define APS5_TESS_SOURCE(...) #__VA_ARGS__
#include "MetalBackend/MslTessellatorCore.inc"
#undef APS5_TESS_SOURCE
        + std::string("\n#undef APS5_TESS_FN\n#undef APS5_TESS_TABLE\n#undef APS5_TESS_THREAD\n#undef APS5_TESS_OUT\n#undef APS5_TESS_AS_UINT\n#undef APS5_TESS_AS_FLOAT\n#undef APS5_TESS_CEIL\n");
    return text;
}

}

#endif
