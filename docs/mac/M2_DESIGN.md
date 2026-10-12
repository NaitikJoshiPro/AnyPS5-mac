# M2 design: Metal shader backend and Metal driver

Status: architecture plus foundation, 2026-10-12, branch `mac-native-metal` (worktree `.mac-work/wt/metal`, build dir `build-metal`).
Inputs: `docs/mac/MISSION.md` §5, §6.3, §8 (M2); decision D8 in `docs/mac/DECISIONS.md`; research in `.mac-work/research/` (`m2-scope.md`, `metal-limits.md`, `metal4-vs-metal3.md`, probes in `metal-probes/`).

M2 exit criteria (MISSION §8): MetalBackend produces valid MSL for every shader in the corpus; the libSceAgcDriver GPU suite passes with 0 failures and only documented Apple-hardware skips; presentation through `CAMetalLayer` works; the pipeline cache persists and a second run compiles 0 pipelines.

---

## 0. Summary

- **Command model: Metal 4** (`MTL4CommandQueue`, allocators, argument tables, residency sets, explicit barriers, `MTL4Compiler`), with a small **Metal 3 side queue** reserved for tessellated draws and as a bisection fallback (presentation, Metal 4 bugs). Evidence in §1 D-M2-1.
- **Binding contract: `core/shader/recompiler/MslAbi.hpp`.** Every stage function gets the same four buffer slots: a dense argument-buffer descriptor table mirroring RuntimeAbi bindings, the 256-byte push-data block, the device-memory LDS arena and a `DriverData` record. Specialization constants keep their ids as function constants. §4.
- **Shader side:** `SpirvTarget::backend = ShaderBackend::Msl` makes `compileVariant` call `MslEmitter` instead of `SpirvEmitter` at the single call site; the artifact carries `SharedMsl {source, reflection}`; specialization becomes function constants. SpirvBackend is untouched and still built. §5.
- **Driver side:** the `VulkanDevice` class keeps its name and API; on Apple it is implemented in Objective-C++ under `libSceAgcDriver/specifics/macos`, and the 23 Vulkan-calling upstream files are replaced one by one by Metal twins listed per work package in `specifics/macos/twins/*.cmake`. §6.
- **Foundation (implemented, verified):** the backend switch and MSL artifact plumbing (incl. recompile and disk caches), `MslAbi.hpp`, the MetalBackend skeleton (structured control flow, guarded loops, phis, integer ALU, MUBUF loads/stores, compute builtins, user data, wave32 ballot), the Metal device skeleton (Metal 4 queue, residency set, event timeline, guest memory wrapped in place, descriptor table, push data, DriverData, loop guard, fault buffer), `agc_metal_msl_compile` (compile-only check) and **`agc_driver_vop1_control` passing on the M3 GPU through MetalBackend**. §11.
- **13 work packages** (6 backend, 6 driver, 1 tooling), disjoint file ownership, three waves of 4–6 agents. §9.

---

## 1. Decision drafts (for `DECISIONS.md`)

### D-M2-1: Metal 4 command model, Metal 3 side queue for tessellation
Context: D8 left Metal 3 vs Metal 4 open. Criterion: the fastest route to a passing GPU suite without blocking the M4 TBDR work.
Evidence:
- Probe `metal-probes/probe_api_m3_m4.mm` (this M3, macOS 27.0.1): 1000 dispatches of 64 threads over a `newBufferWithBytesNoCopy` buffer reached only through a GPU address inside an argument buffer, per-dispatch constants. Both APIs correct. CPU encode ~1.0–1.3 µs/dispatch for Metal 3 serial + `setBytes` and for Metal 4 argument table + upload ring; a Metal 4 encoder barrier after every dispatch adds 0.6–0.8 µs; Metal 3 concurrent + `memoryBarrier` 2.5–3.0 µs. GPU time for 1000 serialized dispatches ~3–4 ms in both; Metal 4 without barriers overlaps them (2 ms wall vs 5–8 ms). Argument-table bindings are snapshotted per dispatch.
- Residency sets also work with Metal 3 command buffers (`useResidencySet:`), so residency is not a differentiator; a missing residency entry silently drops GPU writes in both APIs (probes `probe_residency_missing_m3/m4`).
- `MTL_SHADER_VALIDATION=1` + `MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1` caught an out-of-bounds device store in both a Metal 3 and a Metal 4 pipeline (`probe_validation_m4.mm`); `MTL_DEBUG_LAYER` works with both. GPU capture of Metal 4 work works without Xcode (`/usr/bin/gpucapture`, `gpudebug`; `probe_capture_m4`).
- Mesa KosmicKrisp is a Vulkan 1.4 conformant driver on M1–M5 including the base M3 on macOS 27 using Metal 4 only (Khronos conformance entry 997, 2026-09-25; https://docs.mesa3d.org/drivers/kosmickrisp.html). Its maintainers: "Metal4 simplifies a lot of things since we can just pass pointers around" (Mesa MR 42268).
- Metal 4 maps one-to-one onto the upstream Vulkan-shaped recorder: stage-mask barriers, allocators ≈ command pools, untracked and non-retained resources ≈ Vulkan, address bindings ≈ BDA, `MTLSharedEvent` ≈ timeline semaphore (`metal4-vs-metal3.md` §9).
- Metal 4-only features M4 needs: colour-attachment mapping, render pass suspend/resume within a commit, producer barriers, unified compute+blit encoder, texture view pools. Flexible pipelines and the QoS compiler also work from Metal 3, and pipelines/resources cross between the two queues (`probe_pso_crossuse`).
- Gap: `MTL4RenderCommandEncoder` has no tessellation (Feature Set Tables; Apple recommends mesh shaders). Mixing queues is supported ("Events work with any combination of MTLCommandQueue and MTL4CommandQueue", WWDC25-254).
Choice: Metal 4 for every queue the driver records on. One `MTL4CommandQueue` and one `MTL4Compiler` per device (a process holding >100 `MTL4CommandQueue`s panicked an M3 Max on 27.0.1, mokume #1999). Bring-up barrier policy: a consumer queue barrier (`All → All`, visibility Device) at the start of every encoder and an encoder barrier (`Dispatch|Blit → Dispatch|Blit`) between dependent commands; M4 relaxes them from guest acquire/flush packets. Tessellation: phase 1 skip with a documented reason; phase 2 a Metal 3 side queue with `drawPatches` and event hand-off (the MoltenVK model, which passes the suite); phase 3 (M4) mesh-shader tessellation. Presentation through Metal 4 (`waitForDrawable`/`signalDrawable`), with a Metal 3 present queue behind `APS5_METAL3_PRESENT=1` (KosmicKrisp saw a WindowServer crash presenting through Metal 4).
Risk: a young API (known: `MTL4PipelineDataSetSerializer` with `.captureDescriptors` fails on 27.0.1; concurrent `MTL4Compiler` heap corruption fixed in 27; 16-bit index-buffer bounds FB23291220) and silent failures (missing residency or barrier). Mitigations: one residency set every allocation goes through, regular `MTL_SHADER_VALIDATION` runs, a Metal 3 encoder path kept in the probes for bisection.

### D-M2-2: `MslAbi.hpp` binding contract
Argument buffers (Tier 2, written directly by the CPU with `gpuAddress`/`gpuResourceID`), not direct bindings: RuntimeAbi needs up to 128 buffers, image heaps of 64 and samplers of 32 per stage, beyond the 31/128/16 root slots. One descriptor table per stage function, dense: only the bindings the shader declares, ascending RuntimeAbi binding order, `count` entries each (16-byte `BufferEntry {address, bytes, flags}` for buffers, 8-byte `MTLResourceID` for textures and samplers). Buffer bounds travel in the entry (MSL has no `OpArrayLength`). The emitted shader reports its layout in `MslAbi::Reflection`; the driver binds by the reflection. Details in §4.

### D-M2-3: Driver twins keep the class names, swap per work package, transitional handle bridge
The class keeps the name `VulkanDevice` (PM4 core calls `std::make_shared<VulkanDevice>()` in 5 places; two headers forward-declare it); `MetalDevice` is an alias for macOS-only code. Twin headers live in `specifics/macos/overlay/prx/libSceAgcDriver/...` and are searched first by **every** target of the libSceAgcDriver directory (one definition per class). Twin sources are listed as `original=twin` pairs in `specifics/macos/twins/<package>.cmake`; `Metal.cmake` swaps them into every target that compiled the original (the library and the test targets that compile driver sources directly), enables ARC and links Metal. An original may have several twins (the device API is split by package). During the transition a twin header may keep Vulkan handle types in its public signatures as opaque carriers of Metal objects (`VkBuffer` holding a retained `id<MTLBuffer>`, `VkCommandBuffer` pointing at the D2 command stream), so un-twinned originals keep compiling; each package removes those carriers from the headers it owns once every user is twinned. Un-twinned originals stay compiled but are never executed by the Metal device (their Vulkan function tables are null). Exit check at M2 end: no `Graphics/src/*.cpp` original that calls Vulkan remains in the Apple build (`grep` of the swap lists + `nm -u libSceAgcDriver | grep ' _vk'` empty).

### D-M2-4: Guest memory is wrapped in place, per allocation, never the whole arena
`newBufferWithBytesNoCopy` over 16 KiB page ranges of guest memory, shared storage, untracked, added to the device residency set; GPU address = `gpuAddress + (address − pageBegin)`. Never the whole arena (first GPU use wires and zero-fills the entire range: 2 GiB took 1.39 s and pinned 2.07 GiB), never PROT_NONE (rejected), never executable/code-signed pages (SIGBUS), never `.storageModePrivate` (abort). Memory that cannot be wrapped (code segments, PROT_NONE) is mirrored by copy with write-back, as upstream does for Windows `MEM_IMAGE`. GPU writes bypass CPU `mprotect`, so GPU write tracking stays in the driver (BdaAbi written pages + recorder pending writes). The foundation wraps per page range and drops wrappers whenever the GPU is idle; D3 turns this into per-allocation imports invalidated by `GuestAllocations` unmap notifications.

### D-M2-5: Shader compilation and caches
MSL source and reflection are cached by the recompiler's disk cache (format 28 stores the MSL form). Libraries and pipelines are built by the device's `MTL4Compiler` on background threads (pool size `maximumConcurrentCompilationTaskCount`, 8 on this M3, below-interactive QoS), at shader registration where the PM4 core already prepares shaders. Pipelines persist in a per-title, per-GPU, per-OS `MTL4Archive` written by a binaries-only `MTL4PipelineDataSetSerializer` (binaries+descriptors fails on 27.0.1), with `MTLBinaryArchive` as the fallback. "Second run compiles 0 pipelines" is measured as pipeline creations satisfied by archive lookups that cannot fall back to compiling (`MTL4Archive` lookups never compile; `MTLPipelineOptionFailOnBinaryArchiveMiss` for Metal 3). MSL front-end compiles on the second run hit the OS Metal cache (~5 ms each) and are reported separately.

### D-M2-6: MetalBackend generates MSL text from the IR with a chunked MSL prelude
RDNA semantics that are the same for every shader (MUBUF addressing, typed formats, BDA lookup, f64, 64-bit atomics, wave64 helpers) are written once as MSL helper functions in prelude chunks owned by the instruction families; the emitter emits calls and only the chunks a shader uses. U64 is MSL `ulong` (SPIR-V used a uvec2 pair). F64 is IEEE binary64 in software over `ulong` (MSL has no `double`). Every IR value is a function-scope local (hoisted; MSL block scopes would hide a loop header's definitions). Loops are bounded `for` loops whose limit comes from `DriverData.loopGuardLimit` (never a constant-true condition: KosmicKrisp workaround list), and a tripped guard records a `LoopLimit` fault and ends the invocation, which also keeps a runaway guest loop from wedging the GPU. AIR/metallib generation is out of scope until M4.

### D-M2-7: No SPIRV-Cross bridge on the critical path
The bridge (research P3) was meant to run the suite before MetalBackend existed; the foundation already runs MetalBackend shaders end to end, and the hardware-measured expected values in the tests are the oracle. SPIRV-Cross stays an optional triage tool (T1 may vendor it dev-only under its MIT option if a package needs a differential check); it is not built by default and never shipped.

### D-M2-8: Skips only for documented Apple hardware limits, through an allowlist
`tools/check_skips.py --job macos` (T1) accepts a skip only if the test is in `tools/mac/m2_skip_allowlist.txt` with the expected reason and a `docs/dev/TechnicalDebt.md` anchor. Allowed classes (from `metal-limits.md`): sampler min/max reduction, depth-bounds test, sampler LOD bias in state (emulated in shader where possible), 8x MSAA, conservative rasterization, shader clock (unless emulated), geometry shaders, D24S8, logic ops (unless emulated with programmable blending), MSAA/depth shader writes, graphics pipeline libraries, Vulkan-internal tests replaced by Metal twins. f64 and 64-bit atomics are emulated, not skipped.

---

## 2. Architecture

```
RDNA2 code ─► RdnaDecoder ─► ControlFlow (structurizer) ─► Translation ─► IR (SSA) ─► Optimization
                                                                                 │ compileVariant (Recompiler.cpp)
                                              target.backend == Spirv ─► SpirvEmitter (unchanged)
                                              target.backend == Msl   ─► MslEmitter ─► SharedMsl {source, MslAbi::Reflection}
                                                                                 │ materializeResult: constants → result.specialization
PM4 driver core (Execution/src/Driver, unchanged) ─► VulkanDevice API (overlay header)
   └─ macOS: specifics/macos/src/MetalDevice.mm (+ MetalPresentation.mm, MetalDraw.mm, twins)
        MTL4Compiler: library (MSL) + function constants ─► pipeline (MTL4Archive lookup first)
        descriptor table (MslAbi) + push data + DriverData in an upload ring, argument table
        guest memory wrapped in place ─► residency set ─► MTL4 command buffer ─► MTLSharedEvent
```

---

## 3. (a) Metal 4 command model in the driver

| Vulkan concept (upstream) | Metal 4 (this driver) | Notes |
|---|---|---|
| `VkQueue` | one `MTL4CommandQueue` per device | never per object or test (panic risk) |
| command pool / buffer | `MTL4CommandAllocator` per batch (reset after its event value) / `MTL4CommandBuffer` reused after `beginCommandBufferWithAllocator:` | allocators are not thread-safe: one per recording thread |
| timeline semaphore, fences | `MTLSharedEvent`: `signalEvent:value:` after each commit; CPU `waitUntilSignaledValue:timeoutMS:` | the batch serial is the event value |
| `vkCmdPipelineBarrier` | consumer queue barrier at encoder start; encoder barrier between dependent dispatches; producer barrier at encoder end when the next encoder is known | bring-up: always full barriers; M4: precise |
| image layouts | none | drop transitions |
| descriptor sets | MslAbi descriptor table in the upload ring; `setAddress` into the argument table | snapshotted per dispatch/draw |
| push constants | 256-byte block in the upload ring, `setAddress` at `BufferIndex::PushData` | no `setBytes` in Metal 4 |
| memory / lifetime | residency set for every resource (queue-attached for long-lived, command-buffer `useResidencySet:` for transient); objects kept alive by the batch until its event value | command buffers retain nothing |
| buffer copy/fill/update | `MTL4ComputeCommandEncoder` copy/fill (unified compute+blit) | barriers between them as for dispatches |
| render pass begin/end | `MTL4RenderPassDescriptor` + render encoder; keep the encoder open across draws with the same pass key (upstream `StartDrawPass`/`LeaveRenderPassOpen`) | TBDR: choose load/store actions from PM4 state; `MTLStoreActionUnknown` when undecided |
| pipeline cache | `MTL4Archive` (+ binaries-only serializer) | §7 |
| graphics pipeline libraries | none; flexible render pipelines for format/blend variants | `GraphicsPipelineLibraries()` is false |
| timestamp / occlusion queries | commit feedback GPU times; `MTL4CounterHeap` timestamps; visibility result buffer | Metal 3 counter timestamps reported zero on 26 |
| tessellation | Metal 3 side queue (phase 2), mesh emulation (phase 3) | events between queues, pass split |
| present | `waitForDrawable`/`signalDrawable`, layer residency set; headless offscreen texture without a layer | Metal 3 fallback switch |

Error detection: every commit gets a `MTL4CommitOptions` feedback handler that records `feedback.error` into a sink shared with the device (the handler may run after the device is gone); the next wait raises it. GPU faults the shaders record (BdaAbi fault buffer, `LoopLimit`, `Unmapped`, `Permission`) are checked whenever the device goes idle and raised as `std::runtime_error`. Waits time out after 20 s (`WaitTimeoutMs`).

TBDR (M4 work, enabled by this model): render passes rebuilt from consecutive draws with the same colour/depth targets; load action clear for fast clears, dontCare when the pass fully overwrites; store dontCare for targets never read nor CPU-visible; memoryless MSAA when the guest resolves and discards (4x memoryless + resolve works in Metal 4, `probe_memoryless_msaa_m4`); never sample a bound attachment (programmable blending `[[color(n)]]` or a copy instead).

---

## 4. (b) MslAbi: the binding contract (`core/shader/recompiler/MslAbi.hpp`, Version 1)

Buffer slots, identical for every stage function (`MslAbi::BufferIndex`):

| Slot | Name | Contents | Bound by |
|---|---|---|---|
| 0 | `DescriptorTable` | this function's descriptor table (argument buffer), layout = `Reflection::table` | upload ring + argument table |
| 1 | `PushData` | 256 bytes: RuntimeAbi push slot 0 (pre-raster, compute) at byte 0, slot 1 (pixel) at byte 128 | upload ring |
| 2 | `WorkgroupMemory` | device-memory LDS arena: workgroup `w` uses dwords `[w*stride, (w+1)*stride)`, `stride = Reflection::workgroupMemoryStrideDwords` (LDS + lock dword) | device-owned buffer |
| 3 | `DriverData` | `{workgroupBase[3], loopGuardLimit, faultBuffer (GPU address), reserved[2]}` (32 bytes) | upload ring |
| ≥4 | free | driver helper pipelines (detile, clears, scaler) | |

Descriptor table:
- Ranges in ascending `RuntimeAbi::Binding` order, each `count` entries, 8-byte aligned and packed (`MslAbi::LayoutTable`). The count is the emission-time element count of the binding (`IrDescriptorBinding::resources.size()`, at least 1); the driver fills `min(DescriptorBinding::count, range.count)` entries and leaves the rest zero (null).
- `BufferEntry {ulong address; uint bytes; uint flags}` for `Buffers`, `Gds`, `BdaPagetable`, `FaultBuffer`, `FlattenedSrt`, `ShaderData`. `address` is the GPU address of the bound range's first byte, `bytes` its size (the shader's bounds check, the role `OpArrayLength` plays on the Vulkan path). For guest buffers the driver binds `V#.base & ~3` and writes 0 into the shader-data memory-offset bytes; the shader handles the base misalignment from the `BufferBase + 4r + 3` function constant, like the Vulkan path.
- `MTLResourceID` (8 bytes) for image heaps (bindings 1–56) and `Samplers` (57). MSL types per heap: the binding number fixes the class (sampled/comparison/storage) and `DescriptorImageShape`/numeric class → `texture1d/2d/2d_array/3d/cube<T, access>` or `depth2d<float>`; 1D images are emulated as 2D (KosmicKrisp). Samplers are created with `supportArgumentBuffers = YES`. Unbound elements are zero; shaders test `is_null_texture()` where the Vulkan path relied on `nullDescriptor`.
- Raster order groups are allowed on table members (MSL 4.1 §5.2.1.2; MISSION §5's statement was wrong).

Push data: the shader reads dword `PushDataOffset + i` of the 256-byte block (`PushDataOffset` = function constant 8188 with the emission-time default `pushDataStartDword`); the driver copies `RecompileResult::pushConstants` at byte `pushConstantOffset` (compute: 0). Shader data that does not fit in push data is the `ShaderData` table buffer.

Function constants: every `PipelineSpecialization` id is a `uint` `[[function_constant(id)]]` (Metal allows 65,536; the largest id is 57344+). The emitter records the default it would have baked in (`is_function_constant_defined` fallback); `Reflection::functionConstants` lists the ids; `materializeResult` passes the supplied values in `RecompileResult::specialization` and the driver sets them with `setConstantValue:type:atIndex:`. Values that change MSL *types* (fragment export packing and render-target numeric class, dual-source blending, vertex numeric class where vertex pulling does not absorb it) use preprocessor macros (`MTLCompileOptions.preprocessorMacros`, one library per macro set); array lengths (`HeapCountBase`) are declared at their emission-time maximum.

Stage interfaces (B6/D6 fill these in, the contract is fixed now):
- Compute: `[[thread_position_in_threadgroup]]`, `[[threadgroup_position_in_grid]]` (+ `DriverData.workgroupBase` for split dispatches), `[[thread_index_in_threadgroup]]`, `[[thread_index_in_simdgroup]]`; threads per threadgroup = guest workgroup (x halved when `laneCount == 2`), reported in `Reflection::threadsPerThreadgroup`.
- Vertex: vertex pulling only (no `MTLVertexDescriptor`): guest vertex buffers are descriptor-table buffers read with `[[vertex_id]]`/`[[instance_id]]` + `[[base_vertex]]`/`[[base_instance]]` and converted by the typed-format helpers (Apple GPUs fetch vertices in shader code anyway). Outputs: `[[position]]` (+ `[[invariant]]` with `preserveInvariance` when depth equality matters), parameters as `[[user(locnN)]]`, `[[clip_distance]]`, `[[render_target_array_index]]`, `[[viewport_array_index]]`, point size. ≤124 scalar varyings (the compiler crashes at 128: check before compiling).
- Fragment: parameters with `[[flat]]`/`[[center_no_perspective]]`/`[[center_perspective]]`/`[[sample_*]]`, `[[barycentric_coord]]` (native on Apple7+; no geometry-shader emulation), `[[front_facing]]`, `[[sample_id]]`, `[[position]]`; outputs `[[color(n)]]` typed by macro, `[[depth(any|greater|less)]]`, `[[stencil]]`, `[[sample_mask]]`; `discard_fragment()` for kill and the pixel valid mask.
- Mesh/object (NGG, rect lists, tessellation phase 3): `mesh<V, P, ≤256, ≤512, topology>`, 16 KB payload; rect lists expand in a mesh function or by vertex-ID expansion, never tessellation.

Reflection (`MslAbi::Reflection`): `abiVersion`, `stage`, `group` (RuntimeAbi stage group of the table's bindings), `threadsPerThreadgroup`, `threadgroupMemoryBytes`, `workgroupMemoryStrideDwords`, `usesPushData`, `usesDriverData`, `table`, `functionConstants`. Stored in the disk cache. Change rule: `MslAbi.hpp` is append-only, owned by B1; any change bumps `Version` and lands together with the driver side (D1).

---

## 5. (c) MetalBackend structure (`core/shader/recompiler/MetalBackend`)

Entry: `MslEmitter::Emit(program, inputInfo, bindings, MslTargetOptions)` (same inputs and preconditions as `SpirvEmitter::Emit`), called from `compileVariant` (`Recompiler.cpp`). Dispatch: `MslFlowEmitter.cpp` walks the structured CFG and offers each instruction to the family emitters in turn (`EmitMsl{Control,Alu,Float64,Wave,Memory,Buffer,Image,Stage}Instruction`); an opcode nobody handles fails with its name. Families never touch each other's files; the seams are `MslEmitterState` (B1) and the prelude registry.

File map (SpirvBackend file → MSL counterpart → package). Sizes from the SpirvBackend survey; MSL estimates exclude prelude text.

| SpirvBackend source | Lines | MetalBackend file | Pkg |
|---|---|---|---|
| `SpirvEmitter.cpp`, `SpirvEmitterState.hpp`, `SpirvModule.*`, `SpirvMemory/SpirvTypes.*`, `SpirvConstants.*` | 270+245+521+193+72 | `MslEmitter.cpp`, `MslEmitterState.hpp` (value naming, literals, hoisted locals, function constants, builtins) | B1 |
| `SpirvFlowEmitter.cpp` (CFG walk, terminators, phis, loop guard, wave-LDS barriers, dual-lane emission, dispatch) | 894 | `MslFlowEmitter.cpp` | B1 |
| `SpirvModuleSetup.cpp`, `SpirvModuleEmitter.cpp` 999-1124, `SpirvEmitterHelpers.cpp` 17-153/384-464 | ~700 | `MslEntryPoint.cpp` (signature, table struct, resources) | B1 |
| `SpirvAnalysis.cpp` (IR-level: requirements, function LDS, wave-uniform values) | 488 | reused as is (`SpirvBackend/SpirvAnalysis.hpp`) | — |
| `SpirvAlu/SpirvAluEmitterMath.cpp` scalar parts 22-140, 562-933, 1095-1568 | ~1.3k | `MslAluMath.cpp` (+ `Alu`, `Float16` prelude chunks) | B2 |
| `SpirvAlu/SpirvAluEmitterFloat64.cpp` | 868 | `MslFloat64.cpp` + `Float64` soft-float prelude | B2 |
| `SpirvAlu/SpirvAluEmitterRegisters.cpp` (vector window) | 137 | `MslAluMath.cpp` | B2 |
| wave/DPP/readlane parts of `SpirvAluEmitterMath.cpp` 141-560, 934-1094; `SpirvMemory/SpirvSubgroup.cpp`; cross-half ops `SpirvEmitter.cpp` 113-165 | ~1.1k | `MslWave.cpp` (+ `Wave` chunk) | B3 |
| `SpirvMemory/SpirvMemoryInstructions.cpp` word access, atomic core, LDS lock, 64-bit LDS atomics, append/consume; `SpirvBufferAccess.cpp` 62-188 | ~1.4k | `MslMemory.cpp` (+ `LdsLock`, `Atomics64` chunks) | B3 |
| `SpirvImageEmitter.cpp`, `SpirvMemory/SpirvDescriptors.cpp` | 2097+296 | `MslImage.cpp` (+ `Image` chunk); split into `MslImageSample.cpp`, `MslBvh.cpp` as it grows | B4 |
| `SpirvRuntimeBuffer.cpp`, `SpirvBufferFormat.cpp` (tables reusable), `SpirvMemory/SpirvFormatConvert.cpp`, `SpirvBda{Lookup,Read,Fault}.cpp`, address/flat parts of `SpirvMemoryInstructions.cpp`, `ReadConst` | ~2.5k | `MslRuntimeBuffer.cpp` (+ `RuntimeBuffer`, `BufferFormat`, `Bda` chunks); split into `MslBda.cpp`, `MslAddress.cpp` | B5 |
| `SpirvMemory/SpirvInputOutput.cpp`, I/O, exports, mesh/tess parts of `SpirvModuleEmitter.cpp`, `RectListShaders.cpp` | ~2.1k | `MslStageIo.cpp` (+ `StageIo` chunk); `MslExports.cpp`, `MslMeshTess.cpp`, `MslRectList.cpp` | B6 |
| `BarycentricGeometryShader.cpp` | 204 | none: `[[barycentric_coord]]` is native | — |
| `SpirvOptimizer.cpp`, `SpirvSpecialization.cpp` | 175+634 | none: Metal compiler; function constants | — |

Backend-resident behaviours that must be ported (owner, design):
- **Structured control flow** (B1, done for the basic shapes): `if/else` from selection merges; guarded `for` loops; `break` to the loop merge; branches to the continue block re-emit the latch then `continue`; phi copies on edges via temporaries. To do: goto-variable conditions, indirect branches (already lowered by the front end; assert), header conditional branches without an exit, wave-mask branches for `laneCount == 2`.
- **Loop guard** (B1, done): bounded loops, `DriverData.loopGuardLimit` (default 2²², `APS5_LOOP_GUARD=n`, 0 = unbounded), `LoopLimit` fault with shader hash and PC, invocation ends.
- **Wave64 on 32-wide SIMD groups, `laneCount = 2`** (B1 infrastructure first, B3 semantics): every wave64 compute shader takes this mode on Apple GPUs. Each host invocation runs both halves: values get a per-half name unless wave-uniform (`WaveUniformValues`), each instruction is emitted per half (barriers and `MeshAllocate` once), lane id = `simd_lane + 32·half`, the threadgroup's x size is halved, LocalInvocationId/GlobalInvocationId are rebuilt from the linear index (`SpirvModuleEmitter.cpp:48-65`), ballots combine both halves into 64 bits, cross-half shuffles select by lane bit 5. Graphics stages: wave64 runs at one lane per invocation with cross-lane reads limited to the host SIMD group (fail like `SpirvAluEmitterMath.cpp:195-202`).
- **Wave-LDS ordering barriers** (B1/B3): `threadgroup_barrier(mem_flags::mem_threadgroup)` where `WaveLdsScope` (`SpirvEmitter.cpp:46-68`) requires; simdgroup scope when one guest wave = one SIMD group.
- **LDS** (B3): `threadgroup uint lds[N]` when LDS + lock ≤ 32 KiB, else the device arena at `BufferIndex::WorkgroupMemory` (Apple's hard 32 KiB limit; RDNA allows 64 KiB) with `coherent(device)` accesses; the LDS lock dword for 64-bit/float LDS atomics (`LockedLdsUpdate`, elected lane per SIMD group).
- **64-bit atomics** (B3 core, B5 addressing): Apple9 has only `atomic_max/min_explicit` on `device atomic_ulong` (void, relaxed). umin/umax native; smin/smax via bias when all accesses agree, else lock; add/sub without return: 32-bit add + carry; and/or/xor: two 32-bit atomics; CAS/exchange/returning forms: hashed lock table in device memory, one elected lane per SIMD group; LDS 64-bit: threadgroup lock. `agc_driver_lds_atomics64` runs only after the lock is proven not to hang (supervised, timeout).
- **Typed buffer formats** (B5): `SpirvBufferFormat.cpp` tables reused; conversion helpers in the `BufferFormat` chunk.
- **BDA page table, fault and written-page records** (B5): binary search over `BdaAbi::Range[]` in the `BdaPagetable` table buffer, permissions, overflow, fault record (claim/fill/publish, done in the core `FaultRecord` chunk), written-page tracking for write-back.
- **f64** (B2): IEEE binary64 on `ulong` in the `Float64` chunk (add, mul, fma, div via Newton iteration with exact correction, sqrt, conversions, frexp/ldexp, min/max/compare with RDNA NaN and denormal rules, `FPTrigPreop64` table). Candidate source: Berkeley SoftFloat 3e (BSD-3-Clause, GPL-2.0 compatible; record the import) translated to MSL, or own code. Tests: the ~17 f64 tests; the target then advertises `CapabilityFloat64`.
- **Float semantics** (B2): `MTLMathModeSafe`, precise functions, `#pragma METAL fp contract(off)` except explicit FMA; never `fma(a, b, 0)` (−0); NaN tests on bits; RDNA denormal/clamp/omod/med3/class semantics from `SpirvAluEmitterMath.cpp`.
- **Subgroup ops outside compute/fragment** (B3/D1): they compile and run in vertex/mesh/object functions on this M3 but are undocumented: D1 runs a startup self-test and reports the stages in `SubgroupStages()`; the target gains a stage mask for the backend.
- **Pixel valid mask, helper lanes, WQM** (B6/B3).

Foundation coverage (verified): compute stage, wave32 (`laneCount == 1`), integer ALU subset, MUBUF dword loads/stores with static V# (function constants) incl. sub-dword and misaligned paths, user data from push data or ShaderData, compute builtins, wave32 ballot, guarded loops, phis. Everything else throws `MSL emission failed: ... reason=...` naming the opcode or feature.

---

## 6. (d) Metal driver structure (`core/libs/prx/libSceAgcDriver/specifics/macos`)

Layout:
- `overlay/prx/libSceAgcDriver/...` — twin headers (searched before `core/libs` by every target of the directory).
- `src/` — Objective-C++ twins (ARC). `MetalDeviceState.h` (D1) declares the device state shared by `MetalDevice.mm` (D1), `MetalPresentation.mm` (D6) and `MetalDraw.mm` (D6).
- `twins/<package>.cmake` — `original=twin` pairs per package; `Metal.cmake` (D1) applies them.
- `tests/` — Metal-only tests; each package registers its own in `tests/<package>.cmake` (`B1-msl.cmake` registers `agc_metal_msl_compile` from `MslCompileCheck.mm`), included by `Metal.cmake` before the overlay is applied.

The 23 Vulkan-calling upstream files (survey with line anchors in `.mac-work/research/m2-scope.md` §2 and the driver map summarised below) and their twins:

| Upstream file (Lines / Vk lines) | Twin | Metal counterpart | Pkg |
|---|---|---|---|
| `Execution/src/VulkanDevice.cpp` (4391/170) | `src/MetalDevice.mm`, `src/MetalPresentation.mm`, `src/MetalDraw.mm` | `MTLDevice`, MTL4 queue, compiler, events, compute dispatch, present | D1 (+D6) |
| `Execution/src/BdaFeatures.cpp` (27) | folded into `MetalDevice.mm` | GPU family queries | D1 |
| `Graphics/include/Context.hpp` (207/47) | overlay `Context.hpp` | device/queue/capabilities, no function table | D1 |
| `Execution/include/Recipe.hpp` | overlay `Recipe.hpp` | Metal pipeline/template objects | D1 |
| `Graphics/src/Recorder.cpp` (3737/94) + `.hpp` (828) | `src/Recorder.mm` + overlay | batches = MTL4 command buffers + allocators + event values; barrier API; labels; pending writes; completion actions; counter heaps; 3 helper kernels (MeshArguments, SampleCounter, SampleDumps) in MSL | D2 |
| `Graphics/src/Resources.cpp` (400/82), `BufferPool.cpp` (331/11), `BufferAddress.cpp` (17) | `src/Resources.mm`, `src/BufferPool.mm`, `src/BufferAddress.mm` | `MTLBuffer`/`MTLTexture` wrappers, one-shot batches, heap sub-allocation | D2 |
| `Graphics/src/GuestBufferMemory.cpp` (3702/56) | `src/GuestBufferMemory.mm` | host imports = no-copy wrappers per guest allocation + residency; mirrors for unwrappable memory; write-back | D3 |
| `Graphics/src/UnitShadow.cpp` (859/9), `BdaResources` (page table) | `src/UnitShadow.mm`, `src/BdaResources.mm` | private slabs + copies; BdaAbi range table built from wrappers | D3 |
| `Graphics/src/Texture.cpp` (4371/109), `TextureDetiler.cpp` (425/44), `TextureDetilerDescriptors.cpp` (55), `TextureCache.cpp` | `src/Texture.mm`, `src/TextureDetiler.mm`, … | `MTLTexture` + views (pre-validate view compatibility: bad views abort), detile/CMASK kernels in MSL, no layouts | D4 |
| `Graphics/src/TextureFormat.cpp` (201), `GpuColorTransfer.cpp` (192/20) | `src/TextureFormat.mm`, `src/GpuColorTransfer.mm` | `VkFormat` → `MTLPixelFormat` table (no D24S8: D32F_S8), colour transfer kernel | D4 |
| `Graphics/src/ShaderResources.cpp` (3688/29), `ShaderAddressResources.cpp` (49) | `src/ShaderResources.mm`, … | binding plan → MslAbi descriptor tables; resource/descriptor caches | D5 |
| `Graphics/src/Sampler.cpp` (140), `PlaceholderImages.cpp` (111) | `src/Sampler.mm`, `src/PlaceholderImages.mm` | `MTLSamplerState` (argument buffers; 3 border colours; LOD bias and min/max reduction emulated or skipped) ; dummy textures | D5 |
| `Graphics/src/Pipeline.cpp` (948/50), `PipelineLibrary.cpp` (277), `PipelineCache.cpp` (134) | `src/Pipeline.mm`, `src/PipelineLibrary.mm`, `src/PipelineCache.mm` | render pipelines + depth-stencil states (flexible pipelines), no libraries, `MTL4Archive` | D5 |
| `Graphics/src/Draw.cpp` (2590/26), `DepthSurface.cpp` (303), `MultisampleTarget.cpp` (332) | `src/Draw.mm`, `src/DepthSurface.mm`, `src/MultisampleTarget.mm` | render encoders per pass key, indirect draws (ICB or compute), mesh draws, MSAA (≤4x) + resolve | D6 |
| `Execution/src/PresentationScaler.cpp` (158/14), `Execution/include/Presentation.hpp`, `libSceVideoOut` window code | `src/PresentationScaler.mm`, overlay `Presentation.hpp`, `libSceVideoOut` `__APPLE__` path | `CAMetalLayer` via `SDL_Metal_CreateView`, headless offscreen target, scaler pass | D6 |

Not Vulkan-calling despite the survey list (keep the originals): `SubgroupClock.cpp`, `ShaderAddressResources.cpp` (types only), `ParallelCompare.cpp`, `ColorTargetTransfer.cpp`. The PM4 core (`Execution/src/Driver/**`, 47 files) has no Vulkan calls and is reused verbatim; Recorder's CPU-side statics (label table, pending-write snapshots, statistics; ~30 functions, ~80 call sites) must keep their behaviour in the D2 twin.

Guest memory (D3, D-M2-4): `ImportGuestMemory(ranges, generation, adopt)` imports `GuestAllocations` ranges as no-copy wrappers (16 KiB granules, within one or adjacent non-executable VM regions, `mach_vm_region` checked), adds them to the residency set and records the GPU address per range; `GuestAllocations` unmap invalidates them (`agc_host_import_unmap`). Large allocations are pre-wired off the critical path (first use costs ~0.2–0.7 ms/MiB). Total wrapped memory stays below `recommendedMaxWorkingSetSize` (12.7 GB here). The BDA page table (`BdaAbi::Range{begin, end, deviceAddress, permissions}`) is built from the wrappers.

Presentation (D6): `PresentationWindow` gains a Metal layer accessor; `DisplayWindow` creates the SDL window with `SDL_WINDOW_METAL` + `SDL_Metal_CreateView`; the device configures the `CAMetalLayer` (BGRA8 or RGBA16F, `framebufferOnly`, 2–3 drawables, `displaySyncEnabled`), acquires drawables on the presenter thread, runs the scaler, presents, and retires slots on event values. With SDL's offscreen driver (the `video_out_*` tests) there is no layer: the device renders into an offscreen texture (headless mode) so the flip/display tests run without a window. One supervised windowed run closes the criterion.

Foundation state of the device (`MetalDevice.mm`): working — construction (normal and `CompileOnly`), `Target`/`ComputeTarget` (backend Msl, subgroup 32, capability vocabulary without Float64/Int64Atomics/clock), `Dispatch` (pipeline from MSL + function constants, descriptor table for guest buffers/ShaderData/FlattenedSrt/FaultBuffer, push data, DriverData, one command buffer per dispatch with a full consumer barrier), waits/reaps on the event timeline, conservative answers for labels/fills/copies (`WriteLabelOnGpu` waits and returns 1, `FillBuffer` waits and returns false, `CopyBuffer` returns path 2, `PrepareDispatch`/`PrepareRecipe` return null). Throwing with the package name: `DispatchIndirect`, `DispatchRecipe`, `AfterRecordedWork`, `ReplayComputePipeline`, `Draw`, `ColorMetadataPass`, presentation (`AcquireImage`, `PresentClear`, `PresentPixels`, `PresentDisplayBuffer`, `QueuePresent`), GDS/BDA page table/images/samplers in dispatch bindings.

---

## 7. (e) Shader compilation and caching

1. Recompile (front end + `MslEmitter`) → `SharedMsl` cached in memory (variant cache) and on disk (`ShaderDiskCache`, format 28; `ANYPS5_NO_SHADER_CACHE=1` in tests).
2. Library: `MTL4Compiler newLibraryWithDescriptor` (MSL 4.0, `MTLMathModeSafe`, precise functions) keyed by source hash (+ macro set). Measured cold cost 0.05–4.4 s front end plus 0.25–20 s pipeline under load; the OS Metal cache serves later processes in ~5 ms.
3. Pipeline: function descriptor (+ `MTL4SpecializedFunctionDescriptor` with the function-constant values) → archive lookup → compile on miss with the serializer attached. Keyed by (source hash, constants, render state for graphics).
4. Background compilation (D5): a pool of `maximumConcurrentCompilationTaskCount` workers at utility QoS on the efficiency cores, fed at shader registration (`ShaderRegistry` prepares shaders ahead of use); a draw or dispatch whose pipeline is not ready waits for that one task (no skip in M2).
5. Persistence (D5): `<ShaderCacheDirectory>/metal/<gpu>-<os-build>.mtl4archive` + `MTLBinaryArchive` fallback; serialized at clean shutdown and every N new pipelines, atomically renamed.
6. Exit test (D5 + T1): `agc_metal_pipeline_cache` runs a fixed compute and graphics workload in two child processes sharing a cache directory; the second must report 0 compiled pipelines (all from archive lookups) and identical results.

---

## 8. (f) Test strategy

GPU safety (CLAUDE.md, MISSION §5): one GPU test at a time (`ctest -j1 --timeout 30`; every test of the libSceAgcDriver directory, Metal-only ones included, carries the `gpu` label, so `ctest -L gpu`/`-LE gpu` split GPU from CPU runs), never under a debugger, tiny inputs first; never `agc_driver_lds_atomics64` or window/flip/display tests until their stage; the loop guard is always on; device waits time out at 20 s. Shared build directory: wrap every `ninja`/`ctest` in `tools/mac/buildlock.py build-metal --`.

Bring-up order (counts from `m2-scope.md` §3):
| Stage | Tests | Needs |
|---|---|---|
| G0 | CPU-only tests stay green | — |
| G1 | 182 compute-only execution tests (ALU, LDS, buffer/global/flat memory, 32-bit atomics, wave ops, control flow) | B1, B2, B3, B5, D1 (+D3 for flat/global) |
| G2 | 46 compute + image/sampler tests | B4, D4, D5 |
| G3 | 34 draw tests (vertex fetch, interpolation, MRT/blend/depth/stencil, MSAA, provoking vertex, NGG/mesh, tessellation) | B6, D5, D6 |
| G4 | PM4/driver tests (`agc_driver`, `agc_driver_pm4*`, `agc_driver_wait_shutdown`, `agc_driver_mesh`, `agc_driver_cmask_fast_clear`, `agc_shared_write*`, prepared-shader variants) | D1, D2 |
| G5 | presentation: `agc_driver_flip{,_compute}` (mock), `video_out_*` headless, then one supervised windowed run | D6 |
| G6 | f64 (~17) and 64-bit atomics (~9), `agc_driver_lds_atomics64` last and supervised | B2, B3 |
| G7 | Metal twins of the 25 Vulkan-internal tests, pipeline cache test, MSL corpus check | T1, D5 |

Mechanics (T1):
- `tools/mac/gpu_suite.sh <group-regex>`: runs the group with `ctest -j1 --timeout 30 --output-junit`, excludes the held-back tests, prints pass/fail/skip and the first error line per failure (`MSL emission failed ... opcode=X` lines aggregate into an opcode histogram).
- Skip allowlist: `tools/mac/m2_skip_allowlist.txt` (`test | expected skip reason | TechnicalDebt anchor`); `tools/check_skips.py --job macos --allowlist ...` fails on any other skip and warns on stale entries. Metal skip messages keep the upstream wording (`skipped, the device has no …`) so the existing patterns apply.
- MSL corpus check (exit criterion 1): an `APS5_DUMP_REQUESTS=<dir>` hook in `Recompile` serializes every request (`RequestSerializer`) during a full suite run; `agc_metal_msl_corpus` replays them against the Metal target, compiles each MSL with `newLibraryWithSource` and builds the pipeline (no GPU work). `agc_metal_msl_compile` is its built-in seed.
- Vulkan-internal tests (25: `tests/Recorder.cpp`, `Graphics.cpp` MockVulkan, `agc_driver_bda_device`, resource caches, descriptor pools, pipeline cache/library): each gets a Metal twin under `specifics/macos/tests` exercising the twin's behaviour, or an allowlisted skip with a reason when the behaviour is Vulkan-only (descriptor pool recycling, pipeline libraries).
- Every new driver behaviour gets a Metal test; every backend family gets per-instruction execution coverage through the existing hardware-oracle tests (expected values measured on AMD hardware, `docs/dev/HW_ORACLE.md`).
- Differential checks: SPIRV-Cross is not needed (D-M2-7); the hardware oracle values are the reference.
- Shader validation runs: every package runs its group once under `MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1` before declaring it done (catches non-resident memory and out-of-bounds accesses).

---

## 9. (g) Work packages

Rules for every package:
- Own only the files listed; anything else needs the owner's agreement through the coordinator. `MslAbi.hpp` changes: append-only, bump `MslAbi::Version`, land with the D1 driver side.
- Branch `mac-native-metal`, worktree `.mac-work/wt/metal`, build dir `build-metal` shared: wrap `ninja`/`ctest` in `tools/mac/buildlock.py build-metal --`; build only your targets; logs to `/Users/naitikjoshi/PS5/.mac-work/logs/metal-<pkg>-*.log`.
- GPU tests one at a time, `ctest -j1 --timeout 30 -R '^<exact>$'`; never `agc_driver_lds_atomics64` or window/flip/display tests unless your package is G5/G6 and the coordinator agrees.
- Unsupported states throw `std::runtime_error` naming the package; known gaps go to `docs/dev/TechnicalDebt.md` (one entry per gap, owner package).
- One topic per commit, `Co-Authored-By` line; report pass counts of your test group before/after.

### Backend (core/shader/recompiler/MetalBackend)

| Pkg | Owns | Depends on | Proven by | Size |
|---|---|---|---|---|
| **B1 Core, flow, resources** | `include/MetalBackend/MslEmitter.hpp`, `MslEmitterState.hpp`; `src/MslEmitter.cpp`, `MslFlowEmitter.cpp`, `MslEntryPoint.cpp`, `MslPrelude.cpp` (core chunks + registry); `core/shader/recompiler/MslAbi.hpp`; MSL plumbing in `Recompiler.cpp/.hpp`, `CacheKey.hpp`, `ShaderDiskCache.*` | — | control-flow execution tests (`agc_driver_(loop|branch|scalar_program_end|scalar_sethalt|wave_sleep|...)`), every G1 test reaching emission, `agc_metal_msl_compile` | ~2k |
| **B2 ALU and f64** | `src/MslAluMath.cpp`, `src/MslFloat64.cpp` (+ `Alu`, `Float16`, `Float64` chunks) | B1 | VOP1/2/3/C, SOP*, SDWA, F16/F32/packed, conversions, compares (G1); the ~17 f64 tests (G6) | ~1.7k + ~2k prelude |
| **B3 Wave, LDS, atomics** | `src/MslWave.cpp`, `src/MslMemory.cpp` (+ `Wave`, `LdsLock`, `Atomics64` chunks) | B1 (laneCount 2 infrastructure first) | wave/DPP/readlane/permlane/bpermute/swizzle, LDS and LDS atomics, GDS, 32-bit and 64-bit atomics tests; `agc_driver_lds_atomics64` last | ~1.6k |
| **B4 Images and samplers** | `src/MslImage.cpp` (+ `Image` chunk; may add `MslImageSample.cpp`, `MslBvh.cpp`) | B1, D4/D5 for execution | 46 image tests (G2) | ~1.5k |
| **B5 Buffers, typed formats, BDA, flat/global** | `src/MslRuntimeBuffer.cpp` (+ `RuntimeBuffer`, `BufferFormat`, `Bda` chunks; may add `MslBda.cpp`, `MslAddress.cpp`) | B1, D3 (page table) | buffer load/store/formatted/d16/scalar tests, global/flat/scratch, buffer/address atomics with B3, BDA fault tests | ~1.7k |
| **B6 Stage interfaces** | `src/MslStageIo.cpp` (+ `StageIo` chunk; may add `MslExports.cpp`, `MslMeshTess.cpp`, `MslRectList.cpp`) | B1, D6 | 34 draw tests (G3) | ~1.4k |

### Driver (core/libs/prx/libSceAgcDriver/specifics/macos)

| Pkg | Owns (twins + overlay headers + `twins/<pkg>.cmake`) | Exposes | Proven by | Size |
|---|---|---|---|---|
| **D1 Device, submission integration, compute** | `src/MetalDevice.mm`, `src/MetalDeviceState.h`, overlay `Execution/include/VulkanDevice.hpp`, `Recipe.hpp`, `Graphics/include/Context.hpp`; `specifics/macos/Metal.cmake`; `twins/D1-device.cmake`, `tests/D1-*.cmake` | the VulkanDevice API, Context, pipeline cache hooks; wires D2/D3/D5 twins into dispatch as they land; GDS buffer, device-LDS arena, BDA page table and fault buffer in dispatch bindings (first, they unblock B3/B5); `DispatchIndirect`; labels/copies/fills on the GPU; startup self-tests (vertex SIMD ops, barycentric strip order) | G1 with the B packages, G4 | ~4.5k |
| **D2 Recorder and buffer primitives** | `src/Recorder.mm` + overlay `Recorder.hpp`, `PassHazards.hpp`; `src/Resources.mm`, `src/BufferPool.mm`, `src/BufferAddress.mm` + overlays; MSL for MeshArguments/SampleCounter/SampleDumps; `twins/D2-recorder.cmake` | `Commands()` → command stream (MTL4 command buffer + current encoder), barriers, labels, pending writes, completion actions, timing; CPU-side statics unchanged | G4, label/wait execution tests, `agc_shared_write*`, Metal twins of `tests/Recorder.cpp` | ~4.5k |
| **D3 Guest memory, imports, BDA** | `src/GuestBufferMemory.mm`, `src/UnitShadow.mm`, `src/BdaResources.mm` + overlays; `twins/D3-memory.cmake` | `HostImport`, `HostImportFor`, `ImportMappedRanges`, write-back, shadows, page table | host-import/unmap, write-back, flat/global and BDA tests (with B5) | ~4.6k |
| **D4 Textures, detiling, formats** | `src/Texture.mm`, `src/TextureDetiler.mm`, `src/TextureDetilerDescriptors.mm`, `src/TextureCache.mm`, `src/TextureFormat.mm`, `src/GpuColorTransfer.mm` + overlays; MSL for TextureDetile (2 variants), CmaskClear, ColorTransfer; `twins/D4-textures.cmake` | texture/storage texture API and statics, detile dispatches, format table | G2 (with B4/D5), DCC/CMASK/detile tests | ~5.2k |
| **D5 Descriptors, samplers, pipelines, caches** | `src/ShaderResources.mm`, `src/Sampler.mm`, `src/PlaceholderImages.mm`, `src/Pipeline.mm`, `src/PipelineLibrary.mm`, `src/PipelineCache.mm` + overlays; `twins/D5-pipelines.cmake` | binding plan → MslAbi tables, resource caches, render/compute pipeline creation (background, archives) | resource-cache tests, G2/G3 bindings, `agc_metal_pipeline_cache` (exit criterion 4) | ~5.3k |
| **D6 Draws, attachments, presentation** | `src/MetalDraw.mm`, `src/MetalPresentation.mm`, `src/Draw.mm`, `src/DepthSurface.mm`, `src/MultisampleTarget.mm`, `src/PresentationScaler.mm` + overlays (`Draw.hpp`, `Presentation.hpp`, …); `libSceVideoOut` `__APPLE__` window path; `twins/D6-draw-present.cmake` | draws, passes (TBDR-ready), MSAA, depth, presentation + headless mode | G3 (with B6/D5), G5 | ~4.3k |

### Tooling

| Pkg | Owns | Proven by | Size |
|---|---|---|---|
| **T1 Test infrastructure** | `tools/mac/gpu_suite.sh`, `tools/mac/m2_skip_allowlist.txt`, `tools/check_skips.py` (`--job macos`), `specifics/macos/tests/*` except `MslCompileCheck.mm` growth by B1, the `APS5_DUMP_REQUESTS` hook (in `Recompiler.cpp`, coordinate with B1), `agc_metal_msl_corpus`, Metal twins of the Vulkan-internal tests (with D2/D3/D5), a port-ledger script listing upstream changes to twinned originals after each `git fetch upstream` | its own tests; the suite summary format | ~3k |

### Waves (4–6 agents at a time)
- **Wave 1:** B1, B2, B5, D1, D2, T1. The foundation's compute path already runs MSL shaders, so B1/B2/B5 iterate on G1 tests at once; D1 first adds GDS, device LDS, BDA page table and fault buffer bindings; D2 ports the recorder; T1 builds the runner, allowlist and corpus check.
- **Wave 2:** B3 (after B1's laneCount 2), D3, D4 + B4, D5.
- **Wave 3:** B6 + D6 (draws, presentation), then triage owners per group G1–G7 until the exit criteria hold; D1 integrates D2/D3/D5 into the dispatch path, and every package removes its Vulkan-handle carriers.
Critical path: B1 (laneCount 2) → B3 → G1 complete; D2 → D1 integration → G4; D4/D5 → G2; B6/D6 → G3/G5.

---

## 10. Risks and open items

- Metal 4 maturity (D-M2-1): keep the probes; bisect suspected driver bugs on a Metal 3 encoder path; file Feedback with minimal repros.
- Silent residency/barrier failures: one residency path; shader-validation runs per package.
- MSL compile time for large shaders (0.05–4.4 s front end, up to 20 s pipeline under load): background compilation, archives, OS cache; measure in G3.
- wave64 `laneCount = 2` doubles register pressure in every wave64 compute shader; watch occupancy in M4.
- 64-bit atomic emulation can hang if a lock is held across divergent control flow: elected-lane locking, loop guard, `agc_driver_lds_atomics64` last.
- Tessellation without Metal 4 support: phase 2 side queue costs a pass split per tessellated draw.
- Fragment inputs >124 scalars crash the compiler service: B6 checks the count first.
- Texture view incompatibility aborts the process (`_mtlValidateArgumentsForTextureViewOnDevice`): D4 pre-validates.
- Upstream drift of twinned files: T1's port ledger at each upstream merge.
- MISSION §5 corrections to apply (orchestrator): raster order groups in argument buffers are allowed; subgroup ops work in vertex/mesh/object functions on M3 (undocumented, self-test); 64-bit atomics exact set; Apple10-only list; 4x MSAA max; no-copy wrapping facts (`metal-limits.md` §3).

---

## 11. Foundation: what exists and how it was verified (2026-10-12)

Commits on `mac-native-metal`: `91b78455` (MetalBackend + plumbing), `aa2abcf9` (Metal device + compile check), `c90e5ccc` (prelude chunks owned by families), `649a457f` (device split by package), `d7522a51` (per-package test registration), `d7fba2a4` (merge of `mac-native`), and the commit that adds this file.

| Item | Files |
|---|---|
| Backend switch + MSL artifact | `Recompiler.hpp` (`ShaderBackend`, `SpirvTarget::backend`, `MslShader`, `SharedMsl`, `CompiledShaderArtifact::msl`), `Recompiler.cpp` (`compileVariant`, `buildMslModule`, `materializeResult`), `CacheKey.hpp`, `ShaderDiskCache.*` |
| Binding contract | `core/shader/recompiler/MslAbi.hpp` |
| MetalBackend skeleton | `core/shader/recompiler/MetalBackend/**` |
| Metal device skeleton | `specifics/macos/src/MetalDevice.mm`, `MetalDeviceState.h`, `MetalPresentation.mm`, `MetalDraw.mm`, overlay `VulkanDevice.hpp` |
| CMake switch | `libSceAgcDriver/CMakeLists.txt` (`enable_language(OBJCXX)`, `include(specifics/macos/Metal.cmake)`), `specifics/macos/Metal.cmake`, `twins/*.cmake`, `tests/*.cmake`, recompiler `CMakeLists.txt` |
| Tests | `agc_metal_msl_compile` (new), `agc_driver_vop1_control` (existing, now on Metal) |

Verification (build dir `.mac-work/wt/metal/build-metal`, configured with `-DCMAKE_OSX_ARCHITECTURES=arm64 -DBUILD_TESTING=ON`, ccache, `-DFFMPEG_PREBUILT_DIR=...`):
- `buildlock.py build-metal -- ninja -C build-metal libSceAgcDriver agc_driver_vop1_control_tests agc_metal_msl_compile_tests` → exit 0.
- `buildlock.py build-metal -- ctest --test-dir build-metal -j1 --timeout 30 -R '^agc_driver_vop1_control$'` → 1/1 passed (0.15 s; MSL: guarded prelude, function constants 0–7 and 8188, descriptor table with 2 guest buffers; run with `APS5_MSL_DUMP=1` to print MSL and IR).
- `buildlock.py build-metal -- ctest --test-dir build-metal -j1 --timeout 60 -R '^agc_metal_msl_compile$'` → 1/1 passed.
- `python3 tools/check_test_registration.py build-metal` → 692 test executables, none unregistered.
- The M0 arm64 fixes (`fix(macos)`, `fix(libc)`, `fix(libkernel)`, `fix(agc-driver)`) are merged from `mac-native` (`d7fba2a4`); the foundation tests were re-run on the merged tree with the same results.

### G1 baseline (2026-10-12, after the foundation)

All 286 execution test executables build against the overlay (`ninja -k 0` over `agc_driver_*_tests`: 0 failures). The 196 compute-class tests (heuristic: no draw, no image, no PM4 submission; `agc_driver_lds_atomics64` excluded) ran one at a time: `buildlock.py build-metal -- ctest --test-dir build-metal -j1 --timeout 30 -R '^(<compute tests>)$' --output-junit .mac-work/logs/metal-survey-compute.xml` → 115 s, **29 passed**, 29 skipped, 138 failed, no hang, no timeout.
- Passing: `buffer_descriptor_runtime cache_control div_fixup_f16 dpp_no_form f16_min_max_nan float16_min_max_bits float16_ternary_clamp integer_dot ldexp_f32_denormal packed_integer_clamp prepared_publication registration_prepare_shutdown scalar_getreg scalar_relative_indexing scalar_sethalt scalar_sopk_compare scalar_sopk_misc scalar_sopk_waitcnt vcvt_pknorm_f16_source_modifiers vop1_control vop3_add_sub_clamp vop3_align_op_sel vop3_integer vop3_integer_clamp vop3_lerp vop3_mad_clamp vop3_quad_sad vop3_sad wave_sleep`.
- Skips: f64 14 (B2 soft-f64 removes them), 64-bit buffer atomics 8 (B3/B5), "the device does not import guest memory" 3 (D3), "cannot load Vulkan" 2 (Vulkan-internal: T1 twins), native f16 RTE 1 (B2 decides), shader clock 1 (allowlist or emulate).
- First blocker per failing test (each test stops at its first unsupported feature, so counts are lower bounds): `LaneId` 19 (B3), `ConvertF32F16` 12 + `ConvertF16F32` 2 + other f16/f32 conversions 4 (B2), wave64 `laneCount 2` 10 (B1 then B3), `LoadAddress*`/`StoreAddress*`/address atomics 14 (B5 + D1/D3 page table), `FPMul32`/`FPAdd32`/other f32 ALU 22 (B2), LDS `WriteShared*`/`SharedAtomic*` 11 (B3), `ReadLane`/`ReadFirstLane`/`WriteLane`/DPP/permlane/swizzle 16 (B3), `Reference` 4 (B1), typed buffer formats 6 (B5), `ReadConst`/`ReadConstBuffer` 5 (B5), vector window 2 (B2), images 1 (B4), device-memory LDS 1 (B3 + D1), non-compute stages 3 (B6).

### B1 results: core, flow, resources (2026-10-12)

Commits `b4298a5b`, `5d6a7e4c`, `c520b119`, `2545afda` (+ the docs commit with this section).

What the core does now (`MetalBackend/src/MslEmitter.cpp`, `MslFlowEmitter.cpp`, `MslEntryPoint.cpp`, `MslEmitterState.hpp`):
- **Wave64 on 32-wide SIMD groups (laneCount 2)**, as the SPIR-V path: a wave64 compute program runs guest lanes `simd_lane` (half 0) and `simd_lane + 32` (half 1) of guest wave `thread_index_in_threadgroup / 32` in one host invocation. The flow emitter offers each instruction once per half with `MslEmitterState::half` set, and `Value`/`Arg`/`Name`/`Define` resolve to that half (`v<id>`, `v<id>_h`). Wave-uniform values without side effects (`WaveUniformValues`) are computed once, in half 0, and shared; their emitter reaches the other half through `HalfValue`/`HalfArg`. Barriers, `MeshAllocate` and `Unreachable` are emitted once; phi copies run per half. The host threadgroup is one-dimensional with `ceil(N/64)*32` invocations (`Reflection::threadsPerThreadgroup`; this replaces "x halved" in §4), and the guest `LocalInvocationIndex`/`LocalInvocationId`/`GlobalInvocationId` are rebuilt from the linear index (`index + (index & ~31) + 32*half`, the SPIR-V path's formula).
- **Branch conditions**: scalar, goto-variable and indirect-target conditions read half 0; EXEC/VCC conditions (and at laneCount 2 every per-lane condition) vote over the guest wave, `simd_all` for the zero forms and `simd_any` for the others, combined over both halves; a wave-uniform condition needs no vote. A conditional branch with both edges to one block is a plain branch; indirect or unsupported terminators fail (the front end lowers them). Loop headers are always empty blocks with an unconditional branch (Structurizer), so the "header conditional branch" case of §5 cannot occur.
- **Wave-LDS ordering barriers** (the SPIR-V path's `WaveLdsScope`): `threadgroup_barrier` for single-wave workgroups, `simdgroup_barrier` when a guest wave is one SIMD group, between an LDS write and the next LDS access and between a read and the next write.
- Core opcodes: `Barrier` (threadgroup, LDS memory flags; vertex and TCS as on the SPIR-V path), `Reference`, `ReferenceU32`, `Void`, `Sendmsg`, `Undef*`, `GetShaderBase`. Loop guard: `DriverData.loopGuardLimit` read once per invocation, 0 = no guard.
- **Descriptor table**: image heaps and the sampler heap have member types (`MslTableMemberType`: 1D as 2D, depth-compare heaps `depth2d*`, MSAA storage heaps read-only, 64-bit atomic heaps `ulong`), declared `array<T, count>`; every emitted table carries `static_assert`s that Metal's layout equals `MslAbi::LayoutTable` (size and each range offset). All 63 binding kinds compile on the M3 (`agc_metal_msl_compile`).
- Failures name the shader hash in hex. `APS5_MSL_SURVEY=1` (debug aid) collects every instruction-family failure of a shader and reports them together (`reason=survey: N instruction family failures: [opcode=X ...] ...`): one run lists all missing opcodes per test, and an emission failure without `survey:` is the core's.

Interfaces for the families (`MslEmitterState.hpp`):
- Halves: `half`, `laneCount`, `SharedAcrossHalves`, `HalfValue`, `HalfArg`, `HalfName`. Per-lane private storage (vector window, scratch) needs one copy per `half`.
- Lanes and ids: `GuestLane()` (lane id of the half being emitted: `LaneId`, MUBUF `add_tid`, swizzles), `GuestInvocationIndex()`, `GuestLocalInvocationId(c)`, `GuestGlobalInvocationId(c)`, `WorkgroupId(c)`.
- Votes: `WaveBallot(p)` (uint2 over both halves), `WaveAny(p)`, `WaveAll(p)`; they throw for a wave64 program at one lane per invocation (graphics stages), as the SPIR-V path does.
- Resources: `TableBuffer`, `TableResource` (textures, samplers), `MslTableMemberType`, `MslTableDeclaration`.
- Entry-point hooks: `AddDeclaration` (module scope), `AddParameter` (stage-function parameters, deduplicated), `AddPrologue` (function-scope statements before the first block: threadgroup arrays, the LDS lock reset), `Temporary(prefix)`. LDS: `LdsInDeviceMemory()`, `LdsMemoryFlags()`.
- Small edits in other packages' files: `MslWave.cpp` (Ballot through `WaveBallot`), `MslStageIo.cpp` (compute invocation ids through the core), `MslRuntimeBuffer.cpp` (`add_tid` lane through `GuestLane`).

Results (G1 group = the 196 compute tests of §11 plus `agc_metal_msl_compile`, `buildlock.py build-metal -- ctest --test-dir build-metal -j1 --timeout 30 -R '^(<197 names>)$'`):
- Before B1: 30 passed, 29 skipped, 138 failed.
- After B1 (with B2/B5/D1 work landed in parallel; list in `.mac-work/logs/metal-B1-g1-tests.txt`, junit `metal-B1-final1.xml`): 75 passed, 26 skipped, 96 failed. B1's own gains are `branch_past_endpgm`, `loop_ending_exits`, `scalar_program_end`, `scalar_program_end_saved` (core opcodes, loop guard prelude), `scalar_static_call` and `wave_id_in_workgroup` (wave64 at laneCount 2), and `agc_metal_msl_compile` with the wave64, loop and table-type checks.
- Survey (`APS5_MSL_SURVEY=1`): every G1 test reaches emission except the core's documented gaps: graphics stages (3 tests, B6) and device-memory LDS (2 tests, B3 + D1); see TechnicalDebt "MetalBackend core (macOS)".
- Control-flow tests: `branch_past_endpgm`, `loop_ending_exits`, `scalar_program_end`, `scalar_program_end_saved`, `scalar_sethalt`, `scalar_static_call`, `wave_sleep` pass; `scalar_program_flow` (ReadFirstLane, LaneId), `scalar_setpc` and `scalar_swappc` (ReadLane, WriteLane), `subvector_loop` (LaneId) wait for B3's wave opcodes; `compute_user_data_calls` emits correct MSL (checked with `APS5_MSL_DUMP=1`) but, like every PM4-submitted G1 test (`dword_pattern_fill`, `pattern_buffer_fill` at wave32 too), finds its output untouched after `Submit` (driver path, D1/D2/D3).

### D2 results: recorder and buffer primitives (2026-10-12)

Commits `ec6343b5` (command stream, helper kernels), `41f3bc80` (recorder and buffer twins, Metal twins of the two mock tests), `e70f3eae` (Metal recorder tests), and the docs commit with this section.

What exists (`core/libs/prx/libSceAgcDriver/specifics/macos`):
- **Command stream** (`src/MetalCommandStream.h/.mm`, `Metal::CommandStream`): one MTL4 command buffer, one encoder at a time: `Compute()` (dispatches, copies and fills: Metal 4's compute encoder does blits too), `BeginRender(descriptor)`/`Render()`, `EndEncoder()`. Bring-up barriers (D-M2-1): every encoder starts with a consumer queue barrier over all earlier queue stages (device visibility); `Barrier(after, before)` (Metal stages) is an encoder barrier inside a compute encoder, ends the pass inside a render pass (no intra-pass fragment barriers on Apple GPUs), and is implied between encoders; `PipelineBarrier(source, destination)` takes Vulkan stage masks, mapped through `StagesOf` (TRANSFER = Blit|Dispatch, since fills may be kernels; HOST = no GPU stage). Debug aid: `APS5_METAL_SERIAL_COMMANDS=1` records a full barrier at every `Compute()` that returns an open encoder. Inline data: `Allocate`/`Write` (upload arena memory valid until the batch completed), `UpdateBuffer` (vkCmdUpdateBuffer), `CopyBuffer`, `FillBuffer` (byte-uniform dwords are Metal fills, other dwords the fill kernel), `FillPattern` (16-byte patterns), `WriteTimestamp`, `Arguments()`/`DispatchHelper` for the driver's kernels. Bridges (D-M2-3): `Metal::DeviceOf(context)`/`QueueOf(context)` read the id<MTLDevice>/id<MTL4CommandQueue> a Context's `device`/`queue` carry (nil for a Vulkan or mock-Vulkan Context: `deviceProc` set), `DeviceHandle`/`QueueHandle` make them, `BufferOf`/`BufferHandle` convert VkBuffer ↔ id<MTLBuffer>, `Stream(VkCommandBuffer)`/`Handle(stream)`.
- **Recorder** (`src/Recorder.mm`, overlay `Graphics/include/Recorder.hpp` + `MetalObjects.hpp`): the upstream API and CPU-side logic unchanged (pending writes/reads and their snapshot, label table, completion actions, kept objects, deferred releases, every statistic and report). A batch = MTL4 command buffer + allocator + command stream; `Commands()` returns the open batch's stream handle, `Stream()` the typed stream. `Submit` commits with a feedback handler and signals the recorder's MTLSharedEvent with the batch serial (fence = timeline: `HasTimeline()` is always true, `WaitSerial` waits without the GPU mutex, `signaled()` reads `signaledValue`). Store runs (labels): bytes uploaded and copied into the import (`RecordStore` takes the import's VkBuffer carrier and offset), with the upstream run, join, replace and WAW rules. DCC key stores: fills + seed blits (unaligned 1-3 byte heads and tails work on the M3). [gputime] and batch stamps: MTL4 counter heaps (24 MHz on this M3). Occlusion: segment mode only, through `SampleVisibilityBuffer()` (contract for D6 below). A batch still running after `APS5_METAL_BATCH_TIMEOUT_MS` (20 s) fails the wait. Commit-feedback errors are raised at the next finish (the handler lags the event value).
- **Buffers** (`src/Resources.mm`, `src/BufferPool.mm`, `src/BufferAddress.mm`, overlays `Resources.hpp`, `BufferPool.hpp`): `Buffer`/`DeviceBuffer` hold pool allocations (shared storage when host-visible, private when device-local; every one resident through the pool's residency set on the device queue, ~3 µs to create, ~0.4 µs to add); the upstream size classes, three tiers, budgets (device tier = max(512 MiB, recommendedMaxWorkingSetSize / 8)) and LRU eviction; no slabs. `CommandBatch`: one-shot command buffer with its own event, after the recorder's open batch in queue order. `CopyBuffer`, `RecordMemoryBarrier` record into a stream; `FillDeviceFunctions` is empty; `RenderTarget` throws (D4/D6).
- **Kernels** (`src/RecorderKernels.h`, compiled once per device by `Metal::HelperPipeline`): MSL twins of MeshArguments, SampleCounter, SampleDumps (parameters at buffer 0 in the push-constant layouts), and FillPattern.
- `PassHazards.hpp` needs no overlay (types only).

For D1 (device integration, `MetalDevice.mm`), the steps that put the recorder under the device as upstream `VulkanDevice.cpp` does:
1. Context and recorder at setup: `Graphics::Context context{}; context.device = Graphics::Metal::DeviceHandle(device); context.queue = Graphics::Metal::QueueHandle(queue); context.bufferDeviceAddress = true; context.bufferPool = std::make_shared<Graphics::BufferPool>(context);` then `recorder = std::make_unique<Graphics::Recorder>(context); context.recorder = recorder.get(); recorder->Activate();` and `graphicsContext()` returns that context (twins of D3-D6 take it).
2. Recording a dispatch: `auto& stream = recorder->Stream(&covered); auto encoder = stream.Compute(); [encoder setArgumentTable:...]` (re-set per dispatch: the helper kernels bind their own table); transient tables, push data and DriverData from `stream.Write`/`Allocate` (valid until the batch completed, no ring reset); every object the batch uses through `recorder->Keep(...)` (Metal 4 command buffers retain nothing); `recorder->NotePendingWrites(written ranges)`, `NotePendingReads`, `Graphics::Recorder::CountRecordedWork()`; leading/trailing barriers through `stream.PipelineBarrier` (or `Graphics::RecordMemoryBarrier`) where upstream records them, since the dispatches of a batch share one compute encoder and run concurrently without them; no commit per dispatch.
3. Submission and waits as upstream (`VulkanDevice.cpp` 1680-1760): `WaitIdle` → `CountSync(0)` + `recorder->Sync()`; `SubmitRecorded` → `Reap` + `Submit`; `CanWaitUnlocked` → `HasTimeline()`; `SubmitAndEpoch`, `WaitRecorded` → `WaitSerial`, `ReapRecorded(serial)` → `FinishUpTo`, `ReapRecorded()` → `Reap`; `PendingLabel`, `OpenWriteOverlaps`, `RecordedWritesSettled`, `StoresPendingOver`, `AfterRecordedWork` forward to the recorder.
4. `WriteLabelOnGpu`: upstream's logic with `recorder.RecordStore(Graphics::Metal::BufferHandle(wrapper), address - wrapperBegin, bytes, address)` then `NoteLabel(address, bytes, stamp, queue)` (or `AfterCompletions` behind a copied writer); `FillBuffer`/`CopyBuffer`: `stream.FillPattern`/`FillBuffer`/`CopyBuffer` between the recorder's barriers and notes.
5. Guest-memory wrappers (until D3's imports) must outlive every batch that reads them: drop them only when `recorder->Idle()`, or keep them per batch with `Keep`.
For D6: a render pass begun after `Recorder::PrepareSampleSlot()` sets `descriptor.visibilityResultBuffer = (__bridge id<MTLBuffer>)recorder->SampleVisibilityBuffer()` with `MTLVisibilityResultTypeAccumulate` whenever that is non-null; draws call `NoteSampledDraw(commands)` as upstream; `RecordMeshArguments` dispatches, so it runs before the pass begins; open passes with `stream.BeginRender` and leave them to `LeaveRenderPassOpen`/the recorder to end.

Results (D2 group, 45 tests: G4, the label/wait execution tests `unimported_label_order`, `pattern_buffer_fill`, `dword_pattern_fill`, `copied_writer_copy`, `gated_dispatch_capture*`, `agc_shared_write*`, the tests/Recorder.cpp tests and the two mock tests; `buildlock.py build-metal -- ctest --test-dir build-metal -j1 --timeout 30 -R '^(<45 names>)$'`, names in `.mac-work/logs/metal-D2-baseline.log`):
- Before D2: 11 passed, 16 skipped, 18 failed (`metal-D2-baseline.xml`).
- After D2 (with D1/B work landed in parallel): 12 passed, 15 skipped, 18 failed (`metal-D2-after1.xml`); `agc_recorder_kept_bytes` and `agc_buffer_pool` pass as Metal twins; new `agc_metal_recorder` (now also the flush hook waiting unlocked for a batch's writes, and 600 random batches keeping program order) and `agc_metal_recorder_timing` pass, also under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` and `APS5_METAL_SERIAL_COMMANDS=1`; `agc_metal_recorder_logic{,_late,_stamps}` run the recorder-logic cases of tests/Recorder.cpp unchanged on the Metal recorder and pass. The 18 failures stop before the recorder: graphics stages in MSL (12, B6), `ReadConst`/`LoadAddress*` opcodes (3, B5), `Draw` (1, D5/D6), and `dword_pattern_fill`/`pattern_buffer_fill`, which read their output before the GPU wrote it: the device commits the fill (`APS5_METAL_TRACE=1`: event 1 retires after the check), but `Driver::WaitIdle` only waits for the queue workers and `GuestMemory::FlushGpuWrites` syncs through the recorder's pending-write snapshot, which is empty until the device records into the recorder (step 2 above: `NotePendingWrite(s)` + `Activate`). The same gap is B1's "output untouched after Submit" of the PM4-submitted G1 tests. The 14 tests/Recorder.cpp tests skip (Vulkan-internal: "no usable Vulkan device"); their Metal counterparts are `agc_metal_recorder*`.
- `agc_driver_graphics` (tests/Graphics.cpp, MockVulkan, not in the group) now stops at its first case that builds a `Buffer`, `BufferPool` or `Recorder` on a mock Vulkan Context (TechnicalDebt `metal-d2-mock-tests`): its Vulkan-internal cases need T1's split.

### SHD results: B2 ALU/f64, B3 wave/LDS/atomics, B5 buffers (2026-10-13)

Commits: `fe0e2727` (B3 wave family, LDS/GDS, LDS lock, device-memory LDS, MSL 4.1), `2e487d33` (64-bit buffer/flat/global atomics under a lock table; MslAbi Version 2: `DriverData.reserved` became `atomicLocks`, `AtomicLockTableBytes` = 4 KiB, filled by D1 in `c3f611d9`), `bc600396` (ordered f32 not-equal on NaN bits; sub-dword stores as one compare-exchange), `42a0f443` (stores to a wave-uniform address keep the highest active lane's data), `5bc9f74b` (64-bit atomics commit with one 64-bit `atomic_max/min`, GLC dwordx2 is one 64-bit coherent load), `e8bf6bd4` (the descriptor table holds only the referenced binding kinds), `06fa3dfb`, `7629e2d3`/`a3744691` (`agc_metal_msl_wave_lds`, a CPU-only check of the emitted wave/LDS/lock shape), `2e6f803c` (`gpu_suite --rerun-held`), `cec87019` (TechnicalDebt "MetalBackend wave, LDS and atomics").

Design points (details in TechnicalDebt `metal-b3-*`):
- Cross-lane reads are `simd_shuffle` statements outside any per-lane `if`; wave64 at two lanes per invocation shuffles both halves and picks by lane bit 5; reads of lanes that do not exist are masked with the EXEC bit or `simd_ballot(true)`.
- LDS is `threadgroup uint aps5_lds[N]` (+1 lock dword) or the workgroup's slice of the WorkgroupMemory arena (`device uint* aps5_lds`, stride N+1, linear over `threadgroups_per_grid`, as D1 binds it). Programs with 64-bit LDS atomics serialise every LDS atomic through the lock: one elected invocation per SIMD group tries the lock per round, never spinning alone; 2^20 rounds end in a LoopLimit fault instead of a hang. 64-bit device atomics use the same pattern on the driver's lock table and commit with one 64-bit `atomic_max/min`.
- The backend emits MSL 4.1 (`MslAbi::LanguageMinor` 1): acquire/release fences. Every MSL compiler call must use the MslAbi version (the D5 pipeline cache did not; fixed by TEX).

Results (gpu_suite):
- Before (`.mac-work/logs/gpu-suite/shd-baseline`, HEAD 6273496c): G1 104/182 passed, 77 failed, 1 skipped; G6 15/24 passed, 9 skipped (no Int64Atomics).
- After: G1 167/182 (`shd-r2`, HEAD bc600396; 0 timeouts, and the 77 earlier failures rerun under `--validation` in `shd-r1`), G6 23/24 under `--validation` (`shd-r5`, HEAD 46b3059b), and `agc_driver_lds_atomics64` passed its supervised `--rerun-held` run (`shd-lds64`, 15 s); it stays in HOLD.
- Remaining G1 failures and their owners: `d16_memory`, `global_load_addtid` (the BDA table lacks the dispatch's V# ranges: DRV/D3); `guarded_null_pointer`, `flat_store` (fault records are raised instead of reported since `6bb020c8`: DRV); `buffer_load_store` (two V#s aliasing one array: DRV/D3; it passed in `shd-r4` after DRV's import changes); `host_import_at_map`, `host_import_teardown` (D3); `device_selection` (D1); `image_by`, `image_pck_n`, `runtime_images`, `typed_heaps` (B4/D5: TEX); `depth_fast_clear`, `agc_shader_fusion` (graphics stages: B6). G6: `image_atomics64` skips without `shaderImageInt64Atomics` (B4 + D1). `shader_clock` stays allowlisted.

### DRV results: D1 recorder integration, D3 guest memory (2026-10-13)

Commits `c3f611d9` (MslAbi v2 lock table, Int64Atomics, labels/fills/copies only into registered memory), `e71bf683` (per-dispatch BDA table, D5 pipeline cache, D4/D5 image bindings, D3 host imports), `1b7067f7` + `4633c51d` (registered memory only through its import; weak BDA-table lease), `7e57fcfd` (faults reported as upstream; overlapping V#s share one mapping), `f5fe28b3` (TEX hooks: `ImageBindings::Prepare`, sRGB decode formats, MinLod, min/max samplers), `48392234` (imports compared, no probe), the ANYPS5_GPU commit, `24ba6a74` (quarantine release). The recorder integration itself (Context with Metal device/queue, BufferPool, `Recorder::Activate`, dispatches into `Stream().Compute()` with Keep and pending-write notes, submission/waits forwarded, labels through `RecordStore`/`NoteLabel`, wrappers dropped only when idle, D19.5 teardown) was in `9a3cba4a`/`1d91615c` and is now GPU-verified.

What the device does now (`MetalDevice.mm`):
- Dispatch bindings: guest V#s (in place), GDS (guest memory at `Pm4::GdsAddress()`), device-memory LDS arena, fault buffer, `DriverData.atomicLocks` (4 KiB zeroed table, MslAbi v2), images/samplers through D4/D5's `ImageBindings` (written before the open batch is taken, committed after). `DispatchIndirect` reads the counts on the GPU from wrapped memory (CPU read when a label is pending, the arena is used, or the memory cannot be wrapped).
- BDA page table per dispatch: registered allocations (cached per registry generation, lease held weakly; batches Keep it) + bound V#s (Write when stored to) + GDS + captured snapshots (in place, or their bytes when unwrappable), made sorted and disjoint by `Metal::DisjointRanges` (first range keeps its mapping, permissions ORed). Faults and written pages are read when the batch completes; faults are reported on stderr as upstream (`[gpu] deferred write-back failed: ... BDA access failed`), loop-guard trips printed, nothing thrown.
- Memory: registered guest memory goes through D3 host imports (`GuestBufferMemory.mm`, upstream's registry with no-copy MTLBuffers in a per-device residency set; `ImportMappedRanges`, `HostImportFor`, `HostImportCovers`, `ClearHostImports` work as upstream); other memory (static `__DATA`, unregistered heaps) through page wrappers dropped when idle; a wrapper over overlapping ones covers their union. Labels, fills and copies are recorded on the GPU only into imports (else the PM4 core's drain + CPU path, as upstream for unimported memory).
- Pipelines through D5's `PipelineCache` (`PrepareDispatch` starts background builds). `ANYPS5_GPU` selects the device by name.
- Teardown (D19.5): sync until idle, destroy the recorder, clear image mirrors and host imports (their residency set leaves the queue), drop wrappers, `images->Teardown()`, pipeline cache reset, device residency set removed, buffer pool.

Results (all under `--validation`):
- G4: 14/39 passed before (`gpu-suite/20261012T183153Z-G4`), 18/39 after (`20261012T193710Z-G1_G4_G7`). Remaining: 16 graphics-stage MSL emission (B6), `cmask_fast_clear` (Draw, D6), `gds_append_consume` (B3, fixed since in `d2883a00`), `store_idle_results` (storage images, D4; also aborts at exit through upstream's static order, TechnicalDebt).
- G1: 178/182 (`20261012T193710Z-G1_G4_G7`); driver failures fixed in this package: pattern/dword fills, host_import_at_map/teardown, d16_memory, global_load_addtid, guarded_null_pointer, flat_store, buffer_load_store, copied_writer_copy, device_selection. Remaining: `depth_fast_clear`, `shader_fusion` (B6).
- G7: 24 passed, 2 failed (`msl_corpus` B6, `sampler_reduction` D5), 29 skipped (Vulkan-internal tests; D3's `agc_host_import_unmap`, `agc_unit_shadow`, `agc_unchanged_cpu_stamps`, `agc_storage_baseline_settled` still need Metal twins).
- New/extended tests: `agc_metal_device_bindings` (lock table, overlapping V#s, BDA merge CPU cases, V# through the BDA table, faults reported not thrown), `agc_metal_device_submission` (unregistered memory refused for labels/fills/copies).

### B6/D6 interface: vertex and fragment functions (SHD with DRAW, 2026-10-13)

Driver side from DRAW: `.mac-work/coord/b6-d6-interface-draw.md`, agreed except where noted. MslAbi grows to Version 3 (append-only).

Bindings: each stage function has slots 0-3 as for kernels: its own descriptor table (Reflection::table, group Main for vertex, Fragment for pixel), the shared 256-byte PushData block (vertex push constants at byte 0, pixel at byte 128; each function's PushDataOffset constant 8188 defaults to its emission offset, so the driver need not set it), DriverData (workgroupBase 0). New slot `BufferIndex::VertexFetch = 4` (vertex functions only, `Reflection::vertexFetchLocations != 0`): an array indexed by Vulkan location of
`Aps5VertexFetch { ulong address; uint stride; uint bytes; uint word3; uint rate; uint components; uint reserved; }` (32 bytes: `address` = the V# base plus the attribute offset, `bytes` = the readable bytes from `address` for the bounds check, `word3` = V# dword 3 (dst_sel, format), `rate` 0 per vertex, 1 per instance, `components` = formatComponents), built from `ShaderInvocation::vertexAttributes` as `Graphics/include/VertexInput.hpp` builds Vulkan vertex input. The shader converts with B5's BufferFormat helpers. Embedded-fetch (`fetchEmbedded`) programs fetch through their own buffer loads and need no records.

Vertex function: `vertex Aps5VertexOut aps5_main(uint [[vertex_id]], uint [[instance_id]], uint [[base_vertex]], uint [[base_instance]], ...)`; VertexIndex/InstanceIndex are vertex_id/instance_id (they include the bases, as in Vulkan). `Aps5VertexOut`: `float4 position [[position]]` (+`[[invariant]]`), `float4 p<N> [[user(locn<N>)]]` for every written parameter location N (all float4, integer data bit-cast), `float clip [K] [[clip_distance]]` (incl. the zero-position cull distance), `float point_size [[point_size]]`, `uint layer [[render_target_array_index]]`, `uint viewport [[viewport_array_index]]` when written. Cull distances beyond that throw (TechnicalDebt). The 124-scalar varying limit is checked at emission.

Fragment function: `fragment Aps5FragmentOut aps5_main(Aps5FragmentIn in [[stage_in]], ...)`. Input N is declared only `#if ((APS5_VERTEX_OUTPUTS >> N) & 1)` (locations 32-63: `APS5_VERTEX_OUTPUTS_HI`), else it reads the SPI_PS_INPUT_CNTL default; the macros default to all-written when undefined. Interpolation: `[[flat]]` when SPI_PS_INPUT_CNTL flat (bit 0x400, not custom), `[[center_no_perspective]]` for `pixelLinearInputs`, else `[[center_perspective]]`; centroid folds into center as on the SPIR-V path. Builtins: `[[position]]`, `[[front_facing]]`, `[[sample_id]]` (sets per-sample shading, reported as `Reflection::perSample`), `[[primitive_id]]`, `[[render_target_array_index]]`, `[[barycentric_coord]]`, helper lanes via `simd_is_helper_thread()`. Per-vertex (barycentric-emulation) inputs throw for now.
Outputs: `[[color(n)]]` for each exported MRT n (n = the Vulkan location, after the RuntimeState remap), typed from `pixelInfo` (targetOutputMode: float4, half4 for compressed fp16, uint4/int4, unorm16/snorm16 as float4), dual source `[[color(0), index(1)]]`; `float depth [[depth(any|greater|less)]]`, `uint stencil [[stencil]]`, `uint sample_mask [[sample_mask]]` (also alpha-to-coverage emulation). Kill: the pixel-valid mask local; `discard_fragment()` at the end when no lane is valid and not a helper. `[[early_fragment_tests]]` under the SPIR-V path's condition; ordered pixel shaders use `[[raster_order_group(0)]]` on their storage resources.

Reflection fields appended (Version 3):
- `std::uint64_t outputLocations` (vertex: user(locnN) written, bit N): (a) for DRAW's APS5_VERTEX_OUTPUTS.
- `std::uint32_t vertexFetchLocations` (vertex: locations read through the VertexFetch slot).
- `std::uint32_t colorOutputs` (fragment: color(n) written, bit n), `bool exportsDepth`, `bool exportsStencil`, `bool exportsSampleMask`, `bool dualSource`: (b) for masking unwritten attachments.
- `bool perSample`, `bool earlyFragmentTests`, `bool usesDiscard`.

Rect lists: DRAW's vertex-ID expansion under `APS5_RECT_LIST=1` (6 host vertices per rect, the guest program runs for the rect's 3 guest vertices and the 4th corner is derived per parameter: v3 = v1 + v2 - v0 for non-flat ones), not a mesh function: the emitter wraps the guest body in a function called once per guest vertex.
Pixel LDS (device arena for pixel stages): not supported in the first cut (throws naming B6); no G3 test needs it before the basic draws pass.
Geometry/tessellation/mesh: rejected (KnownDrawRejection) until after G3.

### TEX results: D4 textures, B4 MSL images, D5 descriptors/samplers/pipelines (2026-10-13)

- **Image path:** the upstream twins are the default: Texture.mm, ShaderResources.mm (with a `MetalImagesOnly` mode), Sampler.mm, PlaceholderImages.mm, TextureDetiler.mm (MSL kernels), UnitShadow.mm and GpuColorTransfer.mm. They run on Metal through the `Metal::Vk` shim (MetalVulkanImages.mm). `ImageCache::Attach` builds the image Context, and `ImageBindings` prepares, writes and commits each dispatch's table entries. `APS5_METAL_UPSTREAM_IMAGES=0` keeps the bring-up path (MetalTextures.mm's CPU detile and re-upload) until the flipped default is verified on G1/G2/G4. After that, the bring-up path is deleted in its own commit.
- **MSL images (B4):** sampling, gathers, loads and stores, queries, 32-bit atomics and 64-bit atomics (atomicLocks table) are implemented, as are bindless tables, BVH, emulated depth compare and min/max reduction. Min/max is emulated in the shader because Metal ignores `reductionMode` on M3 (ReductionBase function constants). Heap elements are remapped by `DescriptorIndex` function constants.
- **Pipelines (D5):** `PipelineCache` (MetalPipelineCache.h) gives `Compute`, `Render(RenderPipelineRequest)` and `DepthStencil(DepthStencilRequest)`, plus `Prepare*` for background compiles. It keeps an MTL4Archive per GPU and OS build, which is skipped under shader validation. On the second run, 0 pipelines compile (agc_metal_pipeline_cache).
- **Last verified (upstream path, `--validation`, log .mac-work/logs/gpu-suite/20261012T214250Z-G2_re…):**
  - G2: 38/43 (baseline 0/43, bring-up path 32–33/43).
  - G1 image tests: 4/4.
  - agc_driver_image_atomics64: passes.
  - D3 twins: 3 pass; agc_unit_shadow then skipped (test-context fix committed, not yet run).
- **Open:**
  - dcc_keys_first_write, depth_compare_copy and storage_one_slice_alias: pending-storage interplay.
  - The two emulated_color_compare tests: MSL keeps unselected modes' sampler references, so a reflection field is needed.
  - The flipped default and the test-context fixes are not yet GPU-verified (tests paused by the user).
