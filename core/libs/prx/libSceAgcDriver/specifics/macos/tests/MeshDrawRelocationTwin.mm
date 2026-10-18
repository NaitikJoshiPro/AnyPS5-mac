// Metal twin of tests/execution/MeshDrawRelocation.cpp (agc_driver_mesh_draw_relocation; upstream 032ee506b2, a
// relocated draw keeps its mesh index descriptor), swapped in by twins/UPSYNC3-tests.cmake. The upstream file runs
// unchanged except for its device gate: it asks the target for CapabilityMeshShadingEXT, which the Metal target does
// not list (it is a SPIR-V capability), although the Metal device runs guest NGG primitive shaders as MSL mesh
// functions (DECISIONS D23, MslEntryPoint.cpp EmitMeshFunction). The wrapper below answers that capability for the
// Metal device and every other one as the target lists it.

#include "../../../tests/execution/VulkanTestDevice.hpp"

namespace MetalMeshDrawRelocationTwin {

inline bool TargetHasCapability(const ShaderRecompiler::SpirvTarget& target, spv::Capability capability) {
    // Guest mesh programs are MSL mesh functions on the Metal device (D23).
    if (capability == spv::CapabilityMeshShadingEXT) return true;
    return ::TargetHasCapability(target, capability);
}

}

#define TargetHasCapability MetalMeshDrawRelocationTwin::TargetHasCapability
#include "../../../tests/execution/MeshDrawRelocation.cpp"
#undef TargetHasCapability
