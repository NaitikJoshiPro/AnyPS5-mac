// Metal twin of tests/execution/MeshIndexCache.cpp (agc_driver_mesh_index_cache; upstream d92ebfa882, cached mesh
// draws validate their index descriptors), swapped in by twins/UPSYNC3-tests.cmake. The upstream file runs unchanged
// except for its device gate: it asks the target for CapabilityMeshShadingEXT, which the Metal target does not list (it
// is a SPIR-V capability), so the test returned 77 without a reason on Metal although the Metal device runs guest NGG
// primitive shaders as MSL mesh functions (DECISIONS D23, MslEntryPoint.cpp EmitMeshFunction). The wrapper below
// answers that capability for the Metal device and every other one as the target lists it (as MeshDrawRelocationTwin.mm).

#include "../../../tests/execution/VulkanTestDevice.hpp"

namespace MetalMeshIndexCacheTwin {

inline bool TargetHasCapability(const ShaderRecompiler::SpirvTarget& target, spv::Capability capability) {
    // Guest mesh programs are MSL mesh functions on the Metal device (D23).
    if (capability == spv::CapabilityMeshShadingEXT) return true;
    return ::TargetHasCapability(target, capability);
}

}

#define TargetHasCapability MetalMeshIndexCacheTwin::TargetHasCapability
#include "../../../tests/execution/MeshIndexCache.cpp"
#undef TargetHasCapability
