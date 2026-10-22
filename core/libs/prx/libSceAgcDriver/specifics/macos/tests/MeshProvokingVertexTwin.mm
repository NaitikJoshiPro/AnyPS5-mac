// Metal twin of tests/execution/MeshProvokingVertex.cpp (agc_driver_mesh_provoking_vertex; upstream 8f03282f5e,
// geometry programs draw with the last provoking vertex), swapped in by twins/UPSYNC3-tests.cmake. The upstream file
// runs unchanged except for its device gate: it asks the target for CapabilityMeshShadingEXT, which the Metal target
// does not list (it is a SPIR-V capability), although the Metal device runs guest NGG primitive shaders as MSL mesh
// functions (DECISIONS D23). The wrapper below answers that capability for the Metal device; every other capability
// (FragmentBarycentricKHR, which decides the smooth half) is answered as the target lists it.
//
// Risk to check when this test first runs: upstream 8f03282f5e makes State.cpp set the mesh configuration's
// provokingVertex to outputSize - 1 when PA_SU_SC_MODE_CNTL.PROVOKING_VTX_LAST is set. MslEntryPoint.cpp (the
// provoking corner, around its ProvokingLastMacro use) was written for provokingVertex 0, and MetalDraw.mm also
// defines ProvokingLastMacro for last-vertex mesh draws, so the last-vertex case may rotate twice. A failure of the
// "from vertex 2 with the last" check points there, not at this twin.
//
// Upstream 0bd8f6188a adds StripFlatInputs (a two-triangle strip through the geometry program: flat inputs from
// vertex 2 and vertex 3 with the last provoking vertex, from vertex 0 with the first), which runs here unchanged.
// Its driver half lifts the strip refusal from Pipeline.cpp's ValidateProvokingVertex, which the Metal draw does
// not call (MetalDraw.mm keeps its own provoking-vertex refusals); the second triangle's provoking corner on Metal
// is the MSL mesh function's (odd strip triangles reach the program as (i+1, i, i+2)), so a failure of the
// "triangle 1 (the color of vertex 3)" check points at MslEntryPoint.cpp's provoking corner for strips.

#include "../../../tests/execution/VulkanTestDevice.hpp"

namespace MetalMeshProvokingVertexTwin {

inline bool TargetHasCapability(const ShaderRecompiler::SpirvTarget& target, spv::Capability capability) {
    // Guest mesh programs are MSL mesh functions on the Metal device (D23).
    if (capability == spv::CapabilityMeshShadingEXT) return true;
    return ::TargetHasCapability(target, capability);
}

}

#define TargetHasCapability MetalMeshProvokingVertexTwin::TargetHasCapability
#include "../../../tests/execution/MeshProvokingVertex.cpp"
#undef TargetHasCapability
