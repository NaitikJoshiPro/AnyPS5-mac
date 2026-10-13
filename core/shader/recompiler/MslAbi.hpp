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

inline constexpr std::uint32_t Version = 5u;

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
    Count = 7u
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

// The entry kind of a RuntimeAbi binding (its number within a stage group).
inline EntryKind KindOf(RuntimeAbi::Binding binding) {
    const auto value = static_cast<std::uint32_t>(binding);
    if (value >= static_cast<std::uint32_t>(RuntimeAbi::Binding::Count)) throw std::runtime_error("MslAbi: invalid RuntimeAbi binding " + std::to_string(value));
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
// - fragment: 1 when the pipeline blends with two sources (PipelineSpecialization::DualSourceBlend):
//   colour location 1 becomes color(0) index(1) (Reflection::dualSource: the function writes it);
inline constexpr const char* DualSourceMacro = "APS5_DUAL_SOURCE";
// - fragment: the MSL type of colour output n (float4, uint4 or int4; default from pixelInfo),
//   APS5_COLOR<n>_TYPE.
inline constexpr const char* ColorTypeMacroPrefix = "APS5_COLOR";
// Version 5: the triangle-vertices expansion, shared by conservative rasterization and the per-vertex
// fragment value emulation (no Metal geometry stage; per-vertex fragment inputs are Apple10-only):
// - vertex: 1 when the draw is non-indexed with 3 host vertices per guest triangle and binds
//   BufferIndex::TriangleVertices; host vertex v runs the guest program for the 3 guest vertices
//   guestVertex[3 * (v / 3) + 0..2] and emits corner v % 3;
inline constexpr const char* TriangleVerticesMacro = "APS5_TRIANGLE_VERTICES";
// - vertex and fragment: 1 for overestimating conservative rasterization (needs triangle vertices);
inline constexpr const char* ConservativeMacro = "APS5_CONSERVATIVE";
// - vertex (with TriangleVerticesMacro): the fragment function's Reflection::perVertexLocations,
//   decimal, bits 0-31 and 32-63: each location is also emitted as three flat values (one per
//   corner, user(aps5_pv<L>_<j>));
inline constexpr const char* PerVertexLocationsMacro = "APS5_PER_VERTEX_LOCATIONS";
inline constexpr const char* PerVertexLocationsHighMacro = "APS5_PER_VERTEX_LOCATIONS_HI";
// - vertex (with TriangleVerticesMacro): the fragment function's Reflection::emulatedBarycentrics
//   (bit 0 perspective, bit 1 linear): one-hot corner weights interpolated as the barycentrics.
inline constexpr const char* BarycentricsMacro = "APS5_BARYCENTRICS";

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

// Size of the lock table DriverData::atomicLocks points at (1024 lock words).
inline constexpr std::uint32_t AtomicLockTableBytes = 4096u;

static_assert(std::is_standard_layout_v<DriverData> && sizeof(DriverData) == 32u && offsetof(DriverData, faultBuffer) == 16u);

enum class Stage : std::uint32_t {
    Kernel,
    Vertex,
    Fragment,
    Object,
    Mesh
};

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

    bool operator==(const Reflection& other) const = default;
};

}

#endif
