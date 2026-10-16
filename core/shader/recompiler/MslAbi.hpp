#ifndef CORE_SHADER_RECOMPILER_MSLABI_HPP
#define CORE_SHADER_RECOMPILER_MSLABI_HPP

// The binding contract between the Metal Shading Language backend (MetalBackend) and the Metal
// driver (libSceAgcDriver/specifics/macos), and the one the SPIRV-Cross bridge must follow too.
// It mirrors RuntimeAbi (the Vulkan descriptor contract) so the front end, the binding allocator and
// the driver's binding plan stay shared; only the host representation changes:
//
// - Every stage function gets the same [[buffer(n)]] slots (BufferIndex). A render pipeline binds
//   its vertex-side and fragment-side slots separately, so each function sees its own stage group.
// - RuntimeAbi descriptor bindings (guest buffers, image heaps, samplers, GDS, BDA page table,
//   fault buffer, flattened SRT, shader data) live in one argument buffer per stage function, the
//   descriptor table at BufferIndex::DescriptorTable. Its layout is dense: only the bindings the
//   shader declares, in ascending RuntimeAbi binding order, each `count` entries long (TableLayout).
//   Buffers are BufferEntry records (GPU address + size), textures and samplers MTLResourceID
//   values (Tier 2 argument buffers, written directly by the CPU).
// - The RuntimeAbi push constant block (two 128-byte slots: pre-raster stages, pixel stage) is the
//   256 bytes at BufferIndex::PushData (setBytes on Metal 3, a ring-buffer address on Metal 4).
// - Specialization constants keep their PipelineSpecialization ids as [[function_constant(id)]]
//   (uint). A constant the driver does not supply takes the default the emitter recorded.
// - Device-memory LDS (an LDS larger than threadgroup memory, RuntimeAbi's
//   WorkgroupMemoryDescriptorSet) is the arena at BufferIndex::WorkgroupMemory: workgroup w uses
//   the dwords [w * stride, (w + 1) * stride), stride = Reflection::workgroupMemoryStrideDwords.
// - Values Metal does not give a shader as builtins (split dispatch bases, the loop guard) are the
//   DriverData record at BufferIndex::DriverData.
//
// The emitted MSL reports what it declares in a Reflection; the driver binds by it, never by
// re-deriving the shader's interface. Bump Version on any change to this file's layouts.

#include "RuntimeAbi.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ShaderRecompiler::MslAbi {

// Version 6: storage image heaps of the 2D MSAA and 2D MSAA array dimensions are
// texture2d_array<T, access::read_write> textures of layers * samples slices (sample s of layer l at
// slice l * samples + s; Metal shaders cannot write texture2d_ms). The sample count is function
// constant PipelineSpecialization::SampleCountBase + resource (direct images) or the T# LAST_LEVEL of
// the resource metadata (table images). Sampled multisampled heaps stay texture2d_ms (until v9).
// Version 7: tessellation (DECISIONS D24). The local, hull and domain programs are Stage::Local,
// Stage::Hull and Stage::Domain libraries without a stage function: each defines
// `aps5_stage(...)` with the fixed signature of its role (MslTessellation.cpp), and
// LinkMslTessellation wraps them in namespaces into an object function (local + hull per patch) and
// a mesh function (software tessellator + domain program). The object stage binds the hull
// program's descriptor table at BufferIndex::TessellationHullTable and the TessellationDraw record
// at BufferIndex::TessellationDraw; both stages read the object payload (TessellationPayload*).
// Version 8: every tessellation domain, partitioning and output topology (DECISIONS D26): the
// TessellationDraw record carries the VGT_HOS level clamps (minLevel/maxLevel, were reserved), the
// isoline payload factors are in the tessellator's (density, detail) order, and a culled patch
// launches no mesh threadgroup.
// Version 9: multisampling (DECISIONS D28). Sampled heaps of both MSAA dimensions are
// texture2d_ms_array: guest layer l, sample s of a view whose texture holds h samples per slice is
// slice l * (n / h) + s / h, sample s % h, n the guest sample count (SampleCountBase + resource,
// now supplied for sampled MSAA views too, or the table metadata's T#); size queries return log2(n) + 1
// (the T#'s level count) in w and guest layers in z.
// Fragment functions take SampleBaseMacro (an 8x target emulated as two 4x halves draws each half
// with base 0 and 4), interpolate inputs at the location the guest's I/J pair names
// (center/centroid/sample) and take [[sample_id]] under PS_ITER_SAMPLE.
// Version 10 (upstream sync to 0f2683cb7, the SPIR-V backend's behaviour changes mirrored):
// - vertex functions that write guest cull distances (PA_CL_VS_OUT_CNTL, upstream #423) report
//   Reflection::cullDistances and emulate them in one more clip distance, after the guest clip
//   distances and the all-zero position distance: -1 at every vertex of a culled primitive. Their
//   draws must be triangle-vertices (TriangleVerticesMacro), rect-list (RectListMacro) or point
//   (CullPointsMacro) draws; the library does not compile otherwise (lines are refused);
// - the TessellationDraw level clamps follow upstream's hull programs (#4082): outer factors above
//   0 clamp to [minLevel, maxLevel] (others stay and cull), inner factors above 1 to
//   [max(minLevel, 1), maxLevel] (others become max(minLevel, 1)), before the tessellator's own
//   clamp; hull input control points are clamped to the patch (#4727);
// - new function constants: PipelineSpecialization::RasterPrimitive (pixel functions reading the
//   ancillary word's primitive type, #4031) and UnnormalizedBaseLevelBase + s (#1418); the MSL-only
//   ReductionBase moved from 57408 to 57472 to make room;
// - vertex and local programs of a target without vertexSubgroups (#4470) run as one-lane waves
//   (the front end's host subgroup size 1): the library maps the SIMD-group functions to their
//   one-lane values (MslEmitter.cpp SingleLaneDefines).
// Version 11 (upstream sync to fdefd7660):
// - the hull's patch constants are a record of ceil(tessellationPatchStride / 4) dwords after the
//   control points in the object payload, each constant at its off-chip address modulo the record
//   stride (#4163; was the hull's written range, compacted), and domain programs read it (patch
//   inputs);
// - new function constant PipelineSpecialization::LayerExported (pixel functions reading the
//   ancillary word's render target index, #5097: [[render_target_array_index]] times the constant);
// - GDS plain reads and writes are relaxed atomics (the coherent gds buffer, #4279);
// - pixel functions take [[early_fragment_tests]] whenever the request's earlyZ is set without
//   alpha-to-coverage, depth or stencil export (#4210: kill and sample-mask exports no longer keep
//   tests late).
// Version 12: vertex and mesh functions take the window-space macros (APS5_WINDOW_SPACE,
//   APS5_WINDOW_SPACE_WIDTH/HEIGHT): positions in pixels and depth (PA_CL_VTE_CNTL 0x300) are mapped
//   to clip space over a viewport covering the render area.
// Version 13: mesh functions take the triangle-vertices macros (TriangleVerticesMacro with
//   PerVertexLocationsMacro / BarycentricsMacro; DECISIONS D46): each output triangle gets three unshared vertices with one-hot corner weights and its
//   corners' per-vertex values in the primitive record (user(aps5_pv<L>_<j>)). A mesh holds at most
//   MeshTriangleVerticesPrimitives(maxPrimitives) triangles, so the driver draws the grid (mesh
//   groups, instances, MeshTriangleVerticesSplits(maxPrimitives)); WorkgroupId z reads 0.
// Version 15 (upstream sync to 5d9dcc42d2):
// - fragment: APS5_SAMPLE_MASK (SampleMaskMacro), the uniform PA_SC_AA_MASK in guest sample bits
//   (upstream 1ffc5f672 HostSampleMask, the Vulkan pipeline's pSampleMask): the function ANDs its
//   [[sample_mask]] with it before the APS5_SAMPLE_BASE shift, and a function with other outputs
//   but no mask export gains a [[sample_mask]] member that writes it; unset when the mask is full.
//   Every fragment function reads [[sample_mask]] (aps5_sample_mask_in) and, under the macro,
//   discards a pixel whose covered samples the mask removes before the guest body
//   (SampleMaskDiscardText: Vulkan runs no invocation for it, so its memory writes do not happen),
//   and drops [[early_fragment_tests]] (Metal applies the function's mask after early tests).
// - image and sampler tables (upstream d9e47d1e9/34525fdf7/c794273ab/55eefe9c5, ImageTableAbi) replace
//   the bindless image search. ImageTableMap is an Aps5Buffer entry holding the driver's finalized
//   table map words. SamplerTable is one Aps5TableArray<sampler> entry (16 bytes: the GPU address of
//   an array of 8-byte sampler MTLResourceIDs, its size in bytes, flags 0) indexed by the map's
//   sampler element (+2 depth compare, +1 point twin). ImageTable is ImageTableSlots such entries
//   (struct Aps5ImageTable, member k<b> typed as heap binding b = FirstImageBinding + slot; the
//   multisampled slots are Aps5Buffer placeholders): the driver writes the same array of texture
//   MTLResourceIDs (one per table element of the stage) to every slot, and an access reads element
//   HeaderElementBase + element (the Metal driver's per-shader builds keep the base 0). A missed key,
//   poisoned word, out-of-range element or driver rejection records BdaAbi fault reason ImageTable
//   (bytes = poison index) as the SPIR-V path does, and the access yields zero.
// - fragment: APS5_REVERSED_BLEND (ReversedBlendMacro, upstream 16ea32c97/c434467f6): a function with
//   an MRT0 output gains a float4 aps5_rb [[color(0), index(1)]] member under the macro, and each MRT0
//   store routes copies of the export through the ReversedBlendCode in function constant
//   PipelineSpecialization::ReversedBlend: SecondShift routes into aps5_rb, ColorShift routes (when
//   not all Own) into c0. The library does not compile (#error) for a uint MRT0, other colour
//   outputs or APS5_DUAL_SOURCE alongside it.
// - images: a cube T# read through DIM 2D image_sample / image_load (upstream 3b96840e6, the
//   front end's cubeFaceCompatible runtime mode Dim2DArray + cube) reads face 0: the missing face
//   coordinate is 0 instead of a "too few coordinate components" failure. Direct 2D images whose
//   accesses allow it gain that mode in their ImageBase selector switch.
inline constexpr std::uint32_t Version = 15u;

// Version 13: the triangle-vertices mode of mesh functions. Metal's mesh<> holds at most 256
// vertices (MSL specification, set_index takes a uchar), so a mesh of unshared triangles holds 85;
// a subgroup's primitives are split over the grid's z dimension in equal parts.
inline constexpr std::uint32_t MeshMaxVertices = 256u;
inline constexpr std::uint32_t MeshTriangleVerticesSplits(std::uint32_t maxPrimitives) {
    constexpr std::uint32_t limit = MeshMaxVertices / 3u;
    return maxPrimitives == 0u ? 1u : (maxPrimitives + limit - 1u) / limit;
}
inline constexpr std::uint32_t MeshTriangleVerticesPrimitives(std::uint32_t maxPrimitives) {
    const auto splits = MeshTriangleVerticesSplits(maxPrimitives);
    return std::max((maxPrimitives + splits - 1u) / splits, 1u);
}
static_assert(MeshTriangleVerticesPrimitives(85u) == 85u && MeshTriangleVerticesSplits(86u) == 2u && MeshTriangleVerticesPrimitives(128u) == 64u && MeshTriangleVerticesPrimitives(256u) == 64u && MeshTriangleVerticesSplits(256u) == 4u && MeshTriangleVerticesPrimitives(512u) * 3u <= MeshMaxVertices);

// Metal Shading Language version the backend emits (MTLLanguageVersion4_1, macOS 27 and later: the
// acquire/release orders of the mem_flags fences).
inline constexpr std::uint32_t LanguageMajor = 4u;
inline constexpr std::uint32_t LanguageMinor = 1u;

// Name of the stage function in every emitted library.
inline constexpr const char* EntryPoint = "aps5_main";

// The [[buffer(n)]] slots every stage function uses. Slots at or above Count are free for the
// driver's own helper pipelines (detile, clears, scaler).
enum class BufferIndex : std::uint32_t {
    DescriptorTable = 0u,
    PushData = 1u,
    WorkgroupMemory = 2u,
    DriverData = 3u,
    // Version 3: vertex functions only, the VertexFetchRecord array indexed by Vulkan location
    // (Reflection::vertexFetchLocations), for vertex pulling.
    VertexFetch = 4u,
    // Version 4: fragment functions with Reflection::rasterOrder only (primitive-ordered pixel
    // shaders): any device buffer of at least RasterOrderBytes, declared [[raster_order_group(0)]].
    // Its contents are ignored and never read back; it may be shared by every draw.
    RasterOrder = 5u,
    // Version 5: vertex functions compiled with TriangleVerticesMacro only: a TriangleVerticesHeader
    // followed by `uint guestVertex[3 * triangles]` (see TriangleVerticesHeader).
    TriangleVertices = 6u,
    // Version 7: tessellation object functions only (LinkMslTessellation): the hull program's
    // descriptor table (the local program's is at DescriptorTable), and the TessellationDraw record.
    TessellationHullTable = 7u,
    TessellationDraw = 8u,
    Count = 9u
};

inline constexpr std::uint32_t Index(BufferIndex index) {
    return static_cast<std::uint32_t>(index);
}

// Two RuntimeAbi push constant slots of 32 dwords each.
inline constexpr std::uint32_t PushDataBytes = 2u * RuntimeAbi::PushConstantDwords * sizeof(std::uint32_t);

// A buffer in the descriptor table: GPU address of the bound range's first byte and the range's
// size in bytes (the shader's bounds, like OpArrayLength on the Vulkan path). `flags` is reserved
// (0). Unused or null entries are all zero.
struct BufferEntry {
    std::uint64_t address;
    std::uint32_t bytes;
    std::uint32_t flags;
};

// Textures and samplers are MTLResourceID values (gpuResourceID); 8 bytes each.
using ResourceEntry = std::uint64_t;

static_assert(std::is_standard_layout_v<BufferEntry> && sizeof(BufferEntry) == 16u && alignof(BufferEntry) == 8u);
static_assert(sizeof(ResourceEntry) == 8u);

enum class EntryKind : std::uint32_t {
    Buffer,
    Texture,
    Sampler
};

// Version 15: the ImageTable range holds one Aps5TableArray entry per sampled heap binding
// (FirstImageBinding .. FirstStorageImageBinding - 1), all pointing at the same texture array.
inline constexpr std::uint32_t ImageTableSlots = RuntimeAbi::FirstStorageImageBinding - RuntimeAbi::FirstImageBinding;

// The entry kind of a RuntimeAbi binding (its number within a stage group).
// RuntimeAbi 12 (upstream d9e47d1e9) adds the runtime-indexed tables: SamplerTable, ImageTableMap and
// ImageTable (the one binding numbered past Count, shared by every stage on the SPIR-V path). On Metal
// all three are buffer entries (Version 15): the map's words, and arrays of MTLResourceIDs.
inline EntryKind KindOf(RuntimeAbi::Binding binding) {
    const auto value = static_cast<std::uint32_t>(binding);
    if (binding == RuntimeAbi::Binding::ImageTable) return EntryKind::Buffer;
    if (value >= static_cast<std::uint32_t>(RuntimeAbi::Binding::Count)) throw std::runtime_error("MslAbi: invalid RuntimeAbi binding " + std::to_string(value));
    if (binding == RuntimeAbi::Binding::SamplerTable || binding == RuntimeAbi::Binding::ImageTableMap) return EntryKind::Buffer;
    if (binding == RuntimeAbi::Binding::Samplers) return EntryKind::Sampler;
    if (value >= RuntimeAbi::FirstImageBinding && value < static_cast<std::uint32_t>(RuntimeAbi::Binding::Samplers)) return EntryKind::Texture;
    return EntryKind::Buffer;
}

inline std::uint32_t EntryBytes(EntryKind kind) {
    return kind == EntryKind::Buffer ? static_cast<std::uint32_t>(sizeof(BufferEntry)) : static_cast<std::uint32_t>(sizeof(ResourceEntry));
}

// One RuntimeAbi binding's place in a descriptor table.
struct TableRange {
    RuntimeAbi::Binding binding;
    EntryKind kind;
    std::uint32_t count;
    std::uint32_t offset;

    bool operator==(const TableRange& other) const = default;
};

struct TableLayout {
    std::vector<TableRange> ranges;
    std::uint32_t bytes = 0;

    [[nodiscard]] const TableRange* Find(RuntimeAbi::Binding binding) const {
        const auto found = std::find_if(ranges.begin(), ranges.end(), [&](const TableRange& range) { return range.binding == binding; });
        return found == ranges.end() ? nullptr : &*found;
    }

    bool operator==(const TableLayout& other) const = default;
};

struct TableBinding {
    RuntimeAbi::Binding binding;
    std::uint32_t count;
};

// The dense layout: bindings in ascending RuntimeAbi order, each range 8-byte aligned (every entry
// is a multiple of 8 bytes, so ranges are packed). Duplicate bindings and empty ranges are refused.
inline TableLayout LayoutTable(std::span<const TableBinding> bindings) {
    std::vector<TableBinding> sorted(bindings.begin(), bindings.end());
    std::sort(sorted.begin(), sorted.end(), [](const TableBinding& left, const TableBinding& right) { return left.binding < right.binding; });
    TableLayout layout;
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        if (sorted[index].count == 0u) throw std::runtime_error("MslAbi: a descriptor table binding has no entries");
        if (index != 0u && sorted[index].binding == sorted[index - 1u].binding) throw std::runtime_error("MslAbi: a descriptor table binding is listed twice");
        const auto kind = KindOf(sorted[index].binding);
        layout.ranges.push_back({sorted[index].binding, kind, sorted[index].count, layout.bytes});
        layout.bytes += sorted[index].count * EntryBytes(kind);
    }
    return layout;
}

// Driver-provided values at BufferIndex::DriverData.
struct DriverData {
    // First workgroup of this dispatch within the guest grid (a split or partial dispatch).
    std::array<std::uint32_t, 3> workgroupBase;
    // Loop iterations per loop entry before the loop guard ends the invocation (0: no guard).
    std::uint32_t loopGuardLimit;
    // GPU address of the BdaAbi fault buffer, 0 when the dispatch has none.
    std::uint64_t faultBuffer;
    // Version 2: GPU address of the 64-bit atomic lock table (AtomicLockTableBytes, zero-filled
    // once, never written by the CPU afterwards): Apple GPUs lack 64-bit device atomics beyond
    // min/max, so MetalBackend serialises each 64-bit read-modify-write on a lock word hashed from
    // the address. 0: the dispatch has none (a 64-bit atomic then records a LoopLimit fault and
    // leaves memory unchanged).
    std::uint64_t atomicLocks;
};

// Version 3: one vertex attribute location as the vertex function pulls it (MSL Aps5VertexFetch).
// address = the V# base (0 for a null V#, with bytes 0: reads give zeros), stride = the V# stride,
// bytes = the readable bytes from address (VertexBufferExtent, ZeroStrideAvailableBytes for stride 0),
// word3 = V# dword 3 (dst_sel and format), rate = the attribute's fetchIndex (0 per vertex,
// 1 per instance), components = its format component count.
struct VertexFetchRecord {
    std::uint64_t address;
    std::uint32_t stride;
    std::uint32_t bytes;
    std::uint32_t word3;
    std::uint32_t rate;
    std::uint32_t components;
    std::uint32_t reserved;
};

static_assert(std::is_standard_layout_v<VertexFetchRecord> && sizeof(VertexFetchRecord) == 32u && offsetof(VertexFetchRecord, stride) == 8u);

// Version 3: preprocessor macros the driver sets when it compiles a render pipeline's libraries
// (MTLCompileOptions.preprocessorMacros):
// - fragment: the vertex function's written user(locnN) locations (Reflection::outputLocations),
//   decimal, bits 0-31 and 32-63; a fragment input whose location is not written reads its default;
inline constexpr const char* VertexOutputsMacro = "APS5_VERTEX_OUTPUTS";
inline constexpr const char* VertexOutputsHighMacro = "APS5_VERTEX_OUTPUTS_HI";
// - vertex: 1 when the draw is a rect list expanded to 6 host vertices per rect;
inline constexpr const char* RectListMacro = "APS5_RECT_LIST";
// - vertex: 1 when the guest clip space has z in [-w, w] (the function remaps z = (z + w) / 2).
inline constexpr const char* DepthNegativeOneToOneMacro = "APS5_DEPTH_NEGATIVE_ONE_TO_ONE";
// - vertex/mesh: 1 when the guest position is in window space (PA_CL_VTE_CNTL 0x300: x, y in pixels,
//   z depth, w only for perspective-correct interpolation); the function writes
//   ((2x / width - 1) w, (2y / height - 1) w, z w, w) for a Vulkan-style viewport (0, 0, width,
//   height, 0, 1), width and height (integers) from the two size macros.
inline constexpr const char* WindowSpaceMacro = "APS5_WINDOW_SPACE";
inline constexpr const char* WindowSpaceWidthMacro = "APS5_WINDOW_SPACE_WIDTH";
inline constexpr const char* WindowSpaceHeightMacro = "APS5_WINDOW_SPACE_HEIGHT";
// - fragment: 1 when the pipeline blends with two sources (PipelineSpecialization::DualSourceBlend):
//   colour location 1 becomes color(0) index(1) (Reflection::dualSource: the function writes it);
inline constexpr const char* DualSourceMacro = "APS5_DUAL_SOURCE";
// - fragment: 1 when PipelineSpecialization::ReversedBlend is not 0 (Version 15, upstream 16ea32c97):
//   the function writes MRT0's second blend source aps5_rb at color(0) index(1);
inline constexpr const char* ReversedBlendMacro = "APS5_REVERSED_BLEND";
// - fragment: the MSL type of colour output n (float4, uint4 or int4; default from pixelInfo),
//   APS5_COLOR<n>_TYPE.
inline constexpr const char* ColorTypeMacroPrefix = "APS5_COLOR";
// Version 9: fragment: the guest index of host sample 0 (decimal, default 0): [[sample_id]] reads
// add it and the exported [[sample_mask]] is shifted right by it, so one pass covers guest samples
// base..base+h-1 of an emulated sample count (DECISIONS D28).
inline constexpr const char* SampleBaseMacro = "APS5_SAMPLE_BASE";
// Version 9: fragment: the guest sample count when the pass renders fewer samples than the guest's
// (an emulated count, DECISIONS D28; default: the pass's get_num_samples()), for the alpha-to-
// coverage mask.
inline constexpr const char* SampleCountMacro = "APS5_SAMPLE_COUNT";
// Version 15: fragment: the guest samples a uniform PA_SC_AA_MASK keeps (decimal with a u suffix,
// guest sample bits; unset when every sample is kept): the function ANDs its [[sample_mask]] with
// it (MTLRenderPipelineDescriptor has no sample mask). The driver refuses it for a function without
// outputs when the draw writes depth or stencil.
inline constexpr const char* SampleMaskMacro = "APS5_SAMPLE_MASK";
// Version 15: the stage function's discard of pixels without samples left by APS5_SAMPLE_MASK
// (before the guest body; the host coverage bits are guest samples APS5_SAMPLE_BASE and up).
inline constexpr const char* SampleMaskDiscardText = "#if defined(APS5_SAMPLE_MASK)\n    if (((aps5_sample_mask_in << APS5_SAMPLE_BASE) & (APS5_SAMPLE_MASK)) == 0u) discard_fragment();\n#endif\n";
// Version 5: the triangle-vertices expansion, shared by conservative rasterization and the per-vertex
// fragment value emulation (no Metal geometry stage; per-vertex fragment inputs are Apple10-only):
// - vertex: 1 when the draw is non-indexed with 3 host vertices per guest triangle and binds
//   BufferIndex::TriangleVertices; host vertex v runs the guest program for the 3 guest vertices
//   guestVertex[3 * (v / 3) + 0..2] and emits corner v % 3;
//   Version 13, mesh: 1 when the draw runs the mesh function's triangle-vertices mode (three
//   unshared vertices per output triangle, MeshTriangleVerticesSplits threadgroups per subgroup);
//   no buffer: the corners come from the guest's own primitive export;
inline constexpr const char* TriangleVerticesMacro = "APS5_TRIANGLE_VERTICES";
// - vertex and fragment: 1 for overestimating conservative rasterization (needs triangle vertices);
inline constexpr const char* ConservativeMacro = "APS5_CONSERVATIVE";
// - vertex and mesh (with TriangleVerticesMacro): the fragment function's
//   Reflection::perVertexLocations, decimal, bits 0-31 and 32-63: each location is also emitted as
//   three flat values (one per corner, user(aps5_pv<L>_<j>); per primitive in mesh functions);
inline constexpr const char* PerVertexLocationsMacro = "APS5_PER_VERTEX_LOCATIONS";
inline constexpr const char* PerVertexLocationsHighMacro = "APS5_PER_VERTEX_LOCATIONS_HI";
// - vertex and mesh (with TriangleVerticesMacro): the fragment function's Reflection::emulatedBarycentrics
//   (bit 0 perspective, bit 1 linear): one-hot corner weights interpolated as the barycentrics.
inline constexpr const char* BarycentricsMacro = "APS5_BARYCENTRICS";
// Version 13: mesh (with TriangleVerticesMacro): 1 when the draw's provoking vertex is the last
// (PA_SU_SC_MODE_CNTL PROVOKING_VTX_LAST): corner 0 is the guest primitive's third vertex, then the
// first and second (the vertex function's TriangleVerticesHeader order).
inline constexpr const char* ProvokingLastMacro = "APS5_PROVOKING_LAST";
// Version 10: vertex functions with Reflection::cullDistances drawn as points (not triangle-vertices
// or rect-list draws): 1, each point is culled when one of its cull distances is negative.
inline constexpr const char* CullPointsMacro = "APS5_CULL_POINTS";

// Version 5: the buffer at BufferIndex::TriangleVertices. guestVertex follows the header: per guest
// triangle its 3 guest vertex ids (index buffer, base vertex, strips, fans and restart resolved),
// corner 0 being the provoking vertex, then the primitive's order (Vulkan's, rotated for
// last-vertex provoking), so flat outputs and P0 come from corner 0.
inline constexpr std::uint32_t TriangleFlagConservative = 1u;
inline constexpr std::uint32_t TriangleFlagPerVertex = 2u;
struct TriangleVerticesHeader {
    float viewport[4];   // Vulkan viewport x, y, width, height (pixels)
    float depthRange[2]; // minDepth, maxDepth
    std::uint32_t triangles;
    std::uint32_t flags; // TriangleFlag*
};
static_assert(std::is_standard_layout_v<TriangleVerticesHeader> && sizeof(TriangleVerticesHeader) == 32u && offsetof(TriangleVerticesHeader, triangles) == 24u);

// Version 4: minimum size of the buffer at BufferIndex::RasterOrder. A primitive-ordered fragment
// function reads one dword of it (picked by the pixel position) before the guest program and writes
// it back after, so Metal orders the guest program's memory accesses between overlapping fragments
// as the guest's ordered (POPS) pixel shaders and SPV_EXT_fragment_shader_interlock do.
inline constexpr std::uint32_t RasterOrderBytes = 16384u;

// Size of the lock table DriverData::atomicLocks points at: AtomicLockWords 64-bit atomic lock
// words, hashed by address. The driver allocates AtomicLockAllocationBytes, zero-filled: after the
// lock words, the two words at ClockCounterWord (low, high) count the emulated shader clock reads
// (s_memtime, s_memrealtime: MSL has no shader clock, MslFlowEmitter.cpp EmitClock).
inline constexpr std::uint32_t AtomicLockTableBytes = 4096u;
inline constexpr std::uint32_t AtomicLockWords = AtomicLockTableBytes / 4u;
inline constexpr std::uint32_t ClockCounterWord = AtomicLockWords;
inline constexpr std::uint32_t AtomicLockAllocationBytes = AtomicLockTableBytes + 16u;
static_assert((AtomicLockWords & (AtomicLockWords - 1u)) == 0u && (ClockCounterWord + 2u) * 4u <= AtomicLockAllocationBytes);

static_assert(std::is_standard_layout_v<DriverData> && sizeof(DriverData) == 32u && offsetof(DriverData, faultBuffer) == 16u);

enum class Stage : std::uint32_t {
    Kernel,
    Vertex,
    Fragment,
    Object,
    Mesh,
    // Version 7: tessellation stage libraries (no stage function, see Version).
    Local,
    Hull,
    Domain
};

// Version 7: the record at BufferIndex::TessellationDraw. Object threadgroup (p, i) runs patch p of
// instance i: its local invocations fetch control point c's vertex id from the guest index buffer at
// indexAddress (element firstIndex + p * inputControlPoints + c, plus vertexOffset) or, for a
// non-indexed draw (indexSize 0), as firstIndex + p * inputControlPoints + c; the instance id is
// firstInstance + i. Version 8: minLevel and maxLevel are VGT_HOS_MIN/MAX_TESS_LEVEL; the object
// stage clamps the factors of every patch its raw factors do not cull to [minLevel, maxLevel] (NaN
// factors stay NaN) before the tessellator's own per-partitioning clamp.
struct TessellationDraw {
    std::uint64_t indexAddress;
    std::uint32_t indexSize;
    std::uint32_t firstIndex;
    std::int32_t vertexOffset;
    std::uint32_t firstInstance;
    float minLevel;
    float maxLevel;
};
static_assert(std::is_standard_layout_v<TessellationDraw> && sizeof(TessellationDraw) == 32u);

// Version 7: the object payload, in dwords: the tessellation factors as the hull program wrote
// them in the tessellator's order (outer 0-3, inner 0-1; triangle inner at 4; isoline density at 0
// and detail at 1, the reverse of the ring; Version 8: clamped by the object stage), the patch id, the patch's tessellated primitive
// count, then the hull outputs at TessellationPayloadHeaderWords + byte address / 4 (control point c
// at c * control stride; Version 11: then the patch-constant record, offsets modulo its stride).
inline constexpr std::uint32_t TessellationPayloadFactorWords = 6u;
inline constexpr std::uint32_t TessellationPayloadPatchWord = 6u;
inline constexpr std::uint32_t TessellationPayloadPrimitivesWord = 7u;
inline constexpr std::uint32_t TessellationPayloadHeaderWords = 8u;
// Metal Feature Set Tables (2026-05-21): 16,384 bytes of payload in a mesh shader pipeline.
inline constexpr std::uint32_t TessellationPayloadBytes = 16384u;

// What an emitted library declares, for the driver's pipeline and binding setup.
struct Reflection {
    std::uint32_t abiVersion = Version;
    Stage stage = Stage::Kernel;
    // The RuntimeAbi stage group whose bindings the descriptor table holds (absolute binding
    // numbers are RuntimeAbi::BindingNumber(group, range.binding)).
    RuntimeAbi::Stage group = RuntimeAbi::Stage::Main;
    // Kernels: the threadgroup size the guest's workgroup maps to (threadsPerThreadgroup).
    std::array<std::uint32_t, 3> threadsPerThreadgroup {1u, 1u, 1u};
    // Bytes of threadgroup memory the function declares (the compiler reserves them; reported for
    // the 32 KiB limit check and statistics).
    std::uint32_t threadgroupMemoryBytes = 0;
    // Device-memory LDS: dwords per workgroup in the arena at BufferIndex::WorkgroupMemory (0: unused).
    std::uint32_t workgroupMemoryStrideDwords = 0;
    bool usesPushData = false;
    bool usesDriverData = false;
    TableLayout table;
    // [[function_constant]] ids the function declares; the driver supplies values for these only.
    std::vector<std::uint32_t> functionConstants;
    // Version 3, vertex functions: user(locnN) outputs written (bit N), locations pulled through
    // BufferIndex::VertexFetch (bit N).
    std::uint64_t outputLocations = 0;
    std::uint32_t vertexFetchLocations = 0;
    // Version 3, fragment functions: color(n) outputs written (bit n); depth, stencil and sample
    // mask exports; a dual-source second colour (color(0), index(1)); per-sample shading (reads
    // sample_id); [[early_fragment_tests]]; the function can discard. dualSource: the function writes
    // colour location 1, which APS5_DUAL_SOURCE turns into the second blend source.
    std::uint32_t colorOutputs = 0;
    bool exportsDepth = false;
    bool exportsStencil = false;
    bool exportsSampleMask = false;
    bool dualSource = false;
    bool perSample = false;
    bool earlyFragmentTests = false;
    bool usesDiscard = false;
    // Version 3: sampler-heap references and the runtime image modes they sit in. A reference inside
    // the case `mode` of a direct runtime-mode switch (function constant `id`, the image's mode
    // selector) is the pair (id, mode); any other reference sets samplerUnconditional. A pipeline
    // whose supplied constants select no recorded mode does not bind the sampler heap (as the
    // SPIR-V path's optimizer drops the unused sampler variable).
    std::vector<std::uint32_t> samplerModeUses;
    bool samplerUnconditional = false;
    // Version 4, fragment functions: a primitive-ordered pixel shader; the driver binds a buffer of
    // at least RasterOrderBytes at BufferIndex::RasterOrder. Vertex and fragment functions never use
    // BufferIndex::WorkgroupMemory (graphics-stage LDS is per invocation, in thread memory, as the
    // SPIR-V path's Function-storage LDS), so workgroupMemoryStrideDwords stays 0 for them.
    bool rasterOrder = false;
    // Version 5, fragment functions: parameter locations read per vertex (v_interp_mov, 16-bit
    // interpolation, raw P0/P10/P20 reads: StageInput::perVertex), and the barycentrics the function
    // interpolates them with (bit 0 perspective, bit 1 linear). Either nonzero: the driver draws
    // with TriangleVerticesMacro and passes these as PerVertexLocationsMacro / BarycentricsMacro to
    // the vertex function.
    std::uint64_t perVertexLocations = 0;
    std::uint32_t emulatedBarycentrics = 0;
    // Version 10, vertex functions: the guest cull distances written (highest index + 1, 0: none).
    // Nonzero: the driver draws triangles as triangle-vertices draws (or rect lists), points with
    // CullPointsMacro, and refuses lines (Metal has no cull distances; the emulation needs every
    // vertex of the primitive).
    std::uint32_t cullDistances = 0;

    bool operator==(const Reflection& other) const = default;
};

}

#endif
