# Mission: native Apple Silicon port of AnyPS5

This file is the knowledge base for the project. `CLAUDE.md` at the repo root holds the operating rules; this file holds the goal, the architecture, the milestones with exit criteria, and everything learned before execution started (2026-10-12). Keep it current: when a fact here turns out wrong, fix it here and note it in `docs/mac/DECISIONS.md`.

## 1. Goal

Run PS5 titles **natively on Apple Silicon macOS**, starting from AnyPS5 (`https://github.com/boykopovar/AnyPS5`, GPL-2.0-only, cloned here, branch `mac-native`, remote `upstream`). Not "playable": **very good** performance, close to what native Mac ports get on the same chip.

Hard constraints from the user, not negotiable:

1. **No Rosetta in the product path.** The shipped game process is arm64 only. (Rosetta may be used as a *test oracle* only; see §6.4.)
2. **No MoltenVK, no Vulkan on macOS.** The GPU path is a new Metal backend and Metal driver.
3. **No per-game reverse engineering or decompilation.** The CPU layer is a *generic* x86-64 → ARM64 binary translator that works from metadata every PS5 executable already carries. Per-title patches of game code are out of scope. (Per-title *settings* such as an internal resolution override, applied through our own driver layer, are allowed in M4.)
4. Never download, link to, or request commercial game dumps. Test with freely distributed PS5 homebrew (verify its licence) and with titles the user provides from their own legally obtained copies.

## 2. Target machine (verified with `system_profiler`)

| | |
|---|---|
| Model | MacBook Air 13" M3 (Mac15,12), **fanless** |
| CPU | 8 cores: 4 performance + 4 efficiency |
| GPU | 10-core Apple M3 (Apple9 family), Metal 4, ~3.58 TFLOPS FP32 peak, SIMD-group width 32 |
| Memory | 16 GB unified, ~100 GB/s |
| Display | 2560×1664 (16:10, not 16:9) |
| OS | macOS 27.0.1 (26A434) |
| Toolchain at start | Command Line Tools only (Apple clang 21, CMake 4.4.3, Ninja 1.13.2, Python 3.14.7, Homebrew). **No full Xcode**, so no offline `metal` compiler, no Metal debugger, no Instruments GPU counters, until the user installs it. |

## 3. Background facts

### 3.1 Why AnyPS5 works on PC and why Mac is different
- PS5 CPU = AMD Zen 2, **x86-64**. On PC the game's machine code already runs natively; AnyPS5 only relinks the ELF, rewrites AMD-only instructions, and reimplements Sony's system libraries. On Apple Silicon (ARM64), the x86 code must be translated. That translator is the core new work.
- **PS5 pages are 16 KiB** (`PS5_PAGE_SIZE = 0x4000`), the same as Apple Silicon. Page size is *not* a mismatch. The 16 KiB crashes community testers hit came from dyld sliding images in 4 KiB steps (see §5).
- Apple has said macOS 27 is the final release to support Rosetta 2 (https://developer.apple.com/news/?id=w5ngl9k2). This is one reason for the no-Rosetta rule.

### 3.2 AnyPS5 layout (upstream `main` at `9aa5fd29`)
Total ~368k lines of C/C++/asm.

| Area | Lines | Role | Plan |
|---|---|---|---|
| `core/relinker` | ~22k | PS5 ELF → Linux ELF / Windows PE / x86-64 Mach-O (`--macos`), NID import binding, `--to-intel` / `--to-rosetta` instruction lowering | Keep the ELF/NID/relocation/unwind analysis; add an arm64 output path that hands code to the translator |
| `core/libs/prx` | ~237k | ~130 reimplemented Sony libraries (libkernel, libc, libSceAgc*, audio, pad, save data, ...) built as shared libraries; `nid_patcher` renames exports to NIDs (Mach-O patcher exists, `core/libs/nid/src/MachONidPatcher.cpp`) | Build natively for arm64; replace x86-only bits |
| `core/shader/recompiler` | ~58k | RDNA2 shader → SPIR-V. Stages: `RdnaDecoder` (7.6k) → `ControlFlow` (4.4k) → `Translation` (10k) → `IntermediateRepresentation` (4.7k) / `Optimization` (9.5k) → `SpirvBackend` (17k) | **Keep everything up to the IR** (it encodes thousands of measured RDNA2 behaviours, see `docs/dev/TechnicalDebt.md`). **Replace `SpirvBackend` with a new `MetalBackend`** |
| `core/libs/prx/libSceAgcDriver` | Execution ~21k, Graphics ~33k (75 files use Vulkan), tests ~95k | PM4 command processing, state, draws, dispatches, Vulkan pipelines, memory tracking | Keep the PM4/state logic; **replace the Vulkan layer with Metal**. The tests directory is our GPU conformance suite |
| `core/libs/prx/libSceVideoOut` | ~3.7k | Presentation | Move to `CAMetalLayer` |
| `core/libs/tests` | ~46k | Guest/runtime tests | Must pass on arm64 |

Key docs: `docs/dev/ARCHITECTURE.md`, `BUILD.md`, `TechnicalDebt.md`, `CONVENTIONS.md`, `HW_ORACLE.md`, `docs/user/USAGE.md`.

Upstream behaviour worth knowing: unsupported states throw `std::runtime_error` and the process ends. Upstream `main` now also **ends the process on any draw it cannot execute**. On a new backend this blocks any title with one unsupported effect, so add a documented skip-and-report switch for our port.

### 3.3 Upstream position on macOS
The owner (boykopovar, issue #1) accepts macOS PRs only if (1) the rendering logic isn't reworked, (2) MoltenVK stays out of the Linux and Windows builds, (3) the port stays x86-64. **Our port breaks (1) and (3), so it is a fork.** Do not open upstream PRs or push anywhere without asking the user. Small, platform-neutral fixes could go upstream later, but only if the user asks.

## 4. Existing community work to reuse (read before building anything)

All from issue #1 (https://github.com/boykopovar/AnyPS5/issues/1), #1553, and discussion #1098. Check what has merged into `main` since 2026-10-12 before redoing any of it.

| Who | What | Where |
|---|---|---|
| mugurc | Most complete macOS port (x86-64 Mach-O under Rosetta + MoltenVK): Mach-O guest TLS, image registration, guest start on the main thread, memory protection, `VirtualQuery`, unwind info, write-watch consumers, app packager, `mprotect-slide` fixture. Already merged into `main`: #1597 (`relinker --macos`), #2247 (Mach-O NID patcher). #2119 (`--to-rosetta`) was open. | branch `macos-port` on https://github.com/mugurc/AnyPS5 |
| arvindfroi | **Libraries building natively for arm64**: guest `va_list` read by hand (as on Windows), x87 `long double` converted in software (`strtold`, `wcstold`, `%L`), guest-state ops (`setjmp`) throw until a runner provides them, `@loader_path` instead of `$ORIGIN`, re-signing after NID patching. On arm64 the 431-test suite gave 260 pass, 171 skip (GPU). Also an in-process FEXCore runner `aps5-fex` (willfaust/FEX at MIT commit `ac555dd8`) that runs ten guest programs, with imports calling arm64 libraries through x86 stubs. | mugurc/AnyPS5#3 (libs on arm64), mugurc/AnyPS5#4 (runner); branch `apple-native/macos-native` and `native-arm64` on https://github.com/arvindfroi/AnyPS5 |
| Cloud-Multiplayer | macOS write watch (page protection + SIGBUS/SIGSEGV handler; hooks `GuestArena::HostWrite`; took one title from ~20 to 60 fps) as PR #1686; loader-based port (Linux ELF mapped by an in-process loader); APFS `getdents64` fix; a wall-clock sampling profiler (`APS5_LOADER_PROFILE`) | https://github.com/Cloud-Multiplayer/AnyPS5/tree/macos-loader |
| morten-holm | PIE-slide findings; PRs #1640 (`s_cselect_b64` of masks as lane bit), #1653 (no fragment barycentrics on MoltenVK), #1658 (guest exit handlers before host exit handlers) | issue #1 |
| Zheorgyan | M1 fixes #1565, #1571 (`exp_f32` 2^-126 ULP fix), #1580 (many buffers), #2163 (LibAtrac9 `Gradient[31]` overflow); SPIRV-Cross -0 fix upstream (KhronosGroup/SPIRV-Cross#2704) | issue #1 |
| krishalivaibhav | Vampire Survivors reaching gameplay on M2; geometry-shader-free barycentric fallback through tessellation; clear/LOAD/blend barrier fix | issue #1 |
| yabanana | Proposal: AOT-only LLVM route to ARM64 Mach-O reusing AnyPS5's ELF/unwind/relocation analysis; on three scalar functions it produced code identical to a source-compiled reference | issue #1553 |
| Vladislav-Kalinkin | PS4 static recompiler for Apple Silicon (reached a UE4 error in Hogwarts Legacy) | https://github.com/wie-project/PS4Recomp |
| others | Wine-style PS5 layer `mattias800/prosper`; FEX on iOS `willfaust/madeira`; iPad port `buberlo/madeira-anyps5` (has MoltenVK shader patches); fork `RINNECODER/AnyPS5` | |

Licence: reused code must be GPL-2.0-compatible. FEX is MIT only up to `ac555dd8` of willfaust/FEX; later commits there are GPL-3.0-or-later, which is **not** compatible with GPL-2.0-only. Check the licence of anything you pull in.

## 5. macOS pitfalls already found by others (avoid re-debugging these)

Runtime / loader:
- dyld slides images in 4 KiB steps; the guest rounds `mprotect` to 16 KiB on absolute addresses. Count guest pages from where ELF address 0 of the containing image landed (mugurc `40cabefd`), or load at a fixed 16 KiB-aligned base (an in-process loader can over-reserve and trim).
- `GuestAllocationsRegisterMainImage` skipped `initprot == NONE` filler segments ("guest protection or unmap range has a hole"); those segments also need `maxprot` read+write for later `mprotect` (but `__PAGEZERO` must stay 0). Register only the guest's own segments, not the Mach-O header, `__APS5DATA` or `__LINKEDIT`.
- macOS cannot set the fs base: guest TLS must go through rewritten `%fs` accesses. In our translator this is just a register mapping.
- APFS: `d_seekoff` is 0, so naive `getdirentries` resumes from 0 and loops forever. APFS is case-insensitive by default (`Settings.ini` and `SETTINGS.INI` are one file).
- No `/proc/self/exe` (the shader disk cache was silently disabled; use `_NSGetExecutablePath`), no `/proc/self/maps` (use `mach_vm_region` + `dladdr`; `sceSysmoduleGetModuleInfoForUnwind` is hot, thousands of calls/s in Unity titles).
- `sceKernelGetModuleInfoForUnwind` must fill `eh_frame_addr`/`eh_frame_size`, or every guest C++ throw ends in `terminate` (kills all Unity IL2CPP titles).
- `sceKernelRaiseException` is Windows-only upstream; IL2CPP's Boehm GC needs it to stop the world (deliver with a host signal on the target thread).
- libSystem's `__cxa_thread_atexit` returns `void`. `socketpair` has no `AF_UNIX SOCK_SEQPACKET`. `feof_unlocked`/`getc_unlocked` are macros. No `ppoll`/`POLLRDHUP`. Mach-O assembly symbols need a leading underscore. dyld ignores `$ORIGIN`: use `@loader_path`. Re-sign binaries after NID patching. Older libc++ needs `-fexperimental-library` for `std::stop_token`. With CMake 4.x set `CMAKE_APPLE_SILICON_PROCESSOR` before `project()`; `CMAKE_SYSTEM_PROCESSOR` gets reset.
- Write watch has no backend on macOS in `main`. It is essential for performance: without it the driver reads render targets back every frame.
- Flaky tests seen on Mac: `guest_tcp` (ECONNRESET vs EPIPE), `host_thread_local` (libc++ readies a future before `thread_local` destructors run).
- **Safety:** a wedged GPU once took a Mac down with a WindowServer watchdog panic. Run GPU tests one at a time with a timeout (`ctest -j1 --timeout 30`), never under a debugger, and leave out `agc_driver_lds_atomics64` and window-opening flip/display tests until the backend is stable.

Shader recompiler performance:
- `Structurizer::computeDominatorTree` / `computePostDominators` intersect sorted vectors until a fixed point and are recomputed after every structurizing step. Bit-row fixed point halved a 6-minute UE4 start. `PrepareRegistered` builds and structurizes a graph per registered shader and then doesn't use it.

GPU features that Apple hardware or Metal lacks (MoltenVK failures that map onto real Metal limits the backend must design around):
- No subgroup ops in **vertex** shaders on Apple GPUs. Emit single-lane equivalents, or run vertex work as compute/mesh.
- No f64 in shaders. 64-bit atomics are limited on Apple GPUs (verify exactly what Apple9 supports in the Metal Feature Set Tables before designing).
- Threadgroup memory 32 KiB (a test needs 40 KiB). Upstream already has a fallback: LDS in a device buffer at a descriptor set (`WorkgroupMemoryDescriptorSet`); port that idea.
- No geometry shaders: use mesh/object shaders or compute emulation. Barycentrics: Metal has `[[barycentric_coord]]` natively, so check that before porting the tessellation fallback.
- Raster order groups exist in Metal but must not sit inside argument-buffer fields.
- `fma(l, r, 0)` turns -0 into +0. Emit plain multiplies where sign matters.
- RDNA2 wave64 vs Apple SIMD width 32: see upstream discussion #2013 (wave64 programs at one lane per invocation on 32-wide hosts).
- Storage buffers per stage, MSAA storage images, `nullDescriptor`: in Metal use argument buffers / bindless resources (Metal 3+, residency sets in Metal 4) instead of fixed binding slots.

## 6. Architecture

### 6.1 CPU: generic x86-64 → ARM64 translator
- **Discovery from metadata only:** `.eh_frame` FDEs (function boundaries for nearly all compiled code), relocations, NID import/export tables, init/fini arrays, entry point, then recursive descent. Jump tables found by pattern analysis. Nothing per-game.
- **AOT at relink/install time:** lift x86-64 into our own IR, then emit optimized ARM64. Evaluate LLVM as the AOT backend (quality) against a direct emitter (control, compile speed); yabanana's #1553 result suggests LLVM is viable. Record the choice in DECISIONS.md with measurements.
- **JIT fallback** (`MAP_JIT`, `pthread_jit_write_protect_np`, W^X) for code not found ahead of time. Persist what it translates so the next launch is AOT (a persistent translation cache per title).
- **Performance design:** pin the 16 x86 GPRs to ARM64 registers; XMM/YMM onto NEON (YMM = two 128-bit halves); lazy or dead-eliminated flags (AOT liveness); x86 TSO memory ordering via LDAPR/STLR (FEAT_LRCPC/LRCPC2 on M3) only where needed, skipping provably thread-local and stack accesses. Apple's hardware TSO mode is not available to third-party processes, so measure barrier cost early, because it decides CPU performance. Watch for AVX2 → NEON lowering quality and `x87` (rare; software).
- **Native library calls:** imports bind directly to native arm64 Sony libraries through calling-convention thunks generated at build time from the library declarations. Callbacks into guest code (thread entry, qsort comparators, destructors, signal handlers) need reverse thunks. Guest `setjmp`/`longjmp`, exceptions/unwinding (the guest's own `.eh_frame` describes x86 frames) and signals need explicit design.
- `%fs`-based TLS maps to a pinned register or a `TPIDR_EL0`-relative block.
- Prior art to study, but check licences before copying: FEX-Emu (MIT up to the commit above), Rosetta 2 public analyses, box64, yabanana's approach, PS4Recomp.

### 6.2 Libraries
Build all prx libraries natively for arm64 (start from arvindfroi's work). Replace x86 inline asm/intrinsics (about 6 files in `core/libs/prx` use them; `GuestArena.cpp`, `RegisterContext.cpp` etc.). Keep the Mach-O NID patcher.

### 6.3 GPU
- Keep `RdnaDecoder` → `ControlFlow` → `Translation` → `IR` → `Optimization`. Add `MetalBackend` (MSL source from IR at first; consider AIR/metallib later). Keep `SpirvBackend` compiling until the Metal path passes the conformance suite, so tests can compare.
- New Metal driver in libSceAgcDriver: command encoding, pipelines, argument buffers, residency, synchronisation, memory import (unified memory: guest memory can be wrapped as `MTLBuffer` with `newBufferWithBytesNoCopy` when 16 KiB aligned), presentation via `CAMetalLayer`.
- **Design for tile-based deferred rendering (TBDR) from day one:** rebuild render passes from PM4 state, choose load/store actions precisely, use memoryless targets for transient attachments, merge passes. At ~100 GB/s this is the biggest GPU lever.
- **No shader compilation stutter:** compile at registration on background threads (efficiency cores), persist `MTLBinaryArchive` / pipeline caches per title.

### 6.4 Test oracle for the CPU translator
There is no x86 hardware. Options: (a) expected values from the Intel SDM / known semantics; (b) a small, simple x86 reference interpreter for differential testing; (c) run the x86 fixture binaries under Rosetta **only inside the test harness** and compare. (c) is a test tool, not a product route. Use it only if the user approves; record the answer in DECISIONS.md.

## 7. Performance facts and targets

| | PS5 | M3 Air (10-core GPU) |
|---|---|---|
| GPU FP32 | 10.28 TFLOPS (36 CU @ 2.23 GHz, RDNA2) | ~3.58 TFLOPS |
| Memory bandwidth | 448 GB/s GDDR6 | ~100 GB/s LPDDR5 |
| CPU | 8× Zen 2 @ 3.5 GHz | 4P + 4E; P-cores much faster per core |

- Native Mac ports on M3 Air-class hardware (2024 reports): Resident Evil 4 ~40 fps at 1710×1068 low/medium, ~50 with FSR; Death Stranding ~45–50 fps with resolution below native; Cyberpunk 2077 on base M2 ~36 fps low native, ~50–58 with MetalFX. Elden Ring via CrossOver: base M4 Mac mini ~50 fps 900p medium (community report).
- Realistic AAA target on this Air: ~720p–900p internal upscaled (MetalFX) to the panel or a 1080p-class output, tuned settings, 40–60 fps. Frame interpolation (MetalFX) is optional and must be labelled (it doesn't reduce input latency). The user wants performance pushed hard; the Minecraft evidence (55 → 380 fps average with an optimised renderer, chip ran *cooler*) shows how much a bad API path wastes.
- **Fanless chassis:** CPU and GPU share one power/thermal budget. Wasted CPU watts steal GPU clocks. No busy-waits; background work on efficiency cores; thin submission path. Measure with `sudo powermetrics --samplers cpu_power,gpu_power -i 2000` (needs the user's password, so ask once, or use `powermetrics` only when the user runs it) and the Metal performance HUD (`MTL_HUD_ENABLED=1`).
- Every performance claim needs a measurement: frame time median and 1% low, CPU/GPU watts, sustained over ≥10 minutes (thermal steady state).

## 8. Milestones and exit criteria

Each milestone ends with: all its exit criteria met and verified by commands whose output is recorded in `docs/mac/PROGRESS.md`; build directories from the previous milestone deleted; disk budget check passing; a commit on `mac-native`.

**M0: native arm64 build.**
- Relinker-only build passes natively (`-DANYPS5_RELINKER_ONLY=ON -DBUILD_TESTING=ON`, all tests).
- All prx libraries build as arm64 Mach-O (`file` confirms arm64), NID-patched and signed.
- Full non-GPU `ctest` on arm64: 0 failures; every skip has a recorded reason; flaky tests run 20× and documented.
- No x86 inline asm or x86 intrinsics left on the arm64 build path (grep proves it).

**M1: CPU translator.**
- Translator unit suite: per-instruction semantic tests covering every x86-64 instruction class used by the relinker fixtures and test guests (integer, flags, SSE through AVX2, BMI1/2, FMA, F16C, MOVBE, string ops, x87 subset), compared against the chosen oracle.
- The repo's guest fixtures (relinker test programs: imports, TLS, threads, C++ exceptions, C cleanup, variadic printf, callbacks such as qsort comparators, `long double`) run translated, natively, with correct output.
- Discovery report per binary: % of bytes covered AOT vs JIT; JIT results persisted and reused.
- Microbenchmarks: translated vs native-compiled C for integer loops, memcpy, SIMD kernels, atomics-heavy code. Record the ratios; target ≥0.7× native on integer/SIMD loops.

**M2: Metal backend.**
- `MetalBackend` produces valid MSL for every shader in the existing recompiler test corpus.
- Metal driver passes the libSceAgcDriver GPU test suite: 0 failures; skips only for documented Apple hardware limits (f64, etc.) with a workaround or a TechnicalDebt entry.
- Presentation via `CAMetalLayer` works (flip/display tests pass, run carefully).
- Pipeline cache persisted; second run compiles 0 pipelines for a cached workload.

**M3: first titles.**
- At least one freely distributed PS5 homebrew title (licence verified, recorded) and any title the user provides boot, render correct frames (screenshots checked), take input, play audio, and run ≥10 minutes without a crash, fully native (`file` + `vmmap`/`lsof` show no x86 code and no Rosetta).
- A compatibility table in `docs/mac/COMPATIBILITY.md`.

**M4: performance.**
- Profiling harness: frame time median/1% low, CPU/GPU watts, GPU pass timings (Instruments/Metal counters once Xcode is present), sustained 10-minute runs.
- Implement and measure: TBDR render-pass reconstruction and memoryless targets; persistent pipeline/binary archives; translator hot-path optimisation (barriers, flags, register allocation); efficiency-core placement of background work; internal resolution override + MetalFX upscaling (spatial generic; temporal where motion vectors can be identified generically through the driver).
- 2D titles: locked 60 fps at panel resolution, sustained, with lower power than at the M3 start.
- A 3D title (if the user provides one): results against the §7 targets, with a written profile of the remaining bottlenecks.
- `docs/mac/PERFORMANCE.md` with before/after numbers for every optimisation.
