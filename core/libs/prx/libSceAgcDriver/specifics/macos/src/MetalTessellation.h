#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALTESSELLATION_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALTESSELLATION_H

// Tessellated draws on Metal (DECISIONS D24, owner TESS): MTL4RenderCommandEncoder has no patch
// draws, so the guest's local, hull and domain programs (MslAbi Version 7 tessellation libraries)
// are linked by LinkMslTessellation into the object and mesh functions of a mesh render pipeline
// (MetalMesh.mm buildMeshRender with RenderPipelineRequest::object). The draw launches one object
// threadgroup per patch and instance (drawMeshThreadgroups over (patches, instances, 1)); each
// launches the mesh threadgroups of its patch's tessellated primitives. The object stage's argument
// table carries the local program's bindings, the hull program's descriptor table
// (BufferIndex::TessellationHullTable) and the TessellationDraw record; the mesh stage's the domain
// program's. Indirect tessellated draws are refused, as upstream.

#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "MetalBackend/MslEmitter.hpp"
#include "Recompiler.hpp"
#include <cstdint>
#include <memory>

namespace AgcDriver::Graphics::Metal {

struct TessellationStages {
    const Graphics::CompiledShader* local = nullptr;
    const Graphics::CompiledShader* hull = nullptr;
    const Graphics::CompiledShader* domain = nullptr;
};

// The linked object and mesh libraries of a stage combination (made once per combination and
// configuration; thread-safe).
std::shared_ptr<const ShaderRecompiler::MslTessellationPipeline> LinkTessellation(const TessellationStages& stages, const ShaderRecompiler::TessellationConfiguration& config);

}

#endif
