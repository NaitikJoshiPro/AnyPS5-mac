# Decisions

One entry per decision. Newest at the bottom. Format: context, options, evidence, choice, date.

## D1: Fork, not upstream (2026-10-12)
Upstream accepts macOS only as x86-64 + MoltenVK with unchanged rendering (issue #1). Our goals (arm64, Metal, reworked rendering) conflict, so we work on branch `mac-native` and push nothing without the user's approval.

## D2: No Rosetta in the product path (2026-10-12, user decision)
The game process is arm64 only. Reasons: the user wants native performance; Apple ends Rosetta support after macOS 27. Rosetta as a *test-only* oracle for the CPU translator is still an open question for the user (MISSION §6.4).

## D3: Metal, not MoltenVK (2026-10-12, user decision)
Rewrite the GPU path: new `MetalBackend` in the shader recompiler and a Metal driver in libSceAgcDriver, designed for TBDR. Keep the recompiler front end (decoder, control flow, translation, IR, optimization) because it encodes measured RDNA2 behaviour. MoltenVK failure list in MISSION §5 is the requirements list.

## D4: Generic binary translation, no per-game reverse engineering (2026-10-12, user decision)
The CPU layer is a game-agnostic x86-64 → ARM64 translator: AOT from executable metadata (`.eh_frame`, relocations, NID tables) plus a JIT fallback with a persistent cache. Open sub-decision for M1: LLVM AOT backend vs a direct emitter, and whether to borrow from FEX (MIT ≤ `ac555dd8`). Decide with measurements.

## D5: Storage budget 100 GB (2026-10-12, user decision)
Everything the port creates stays ≤ 100 GB; `tools/mac/disk_budget.sh` enforces it. Only storage exceptions require asking the user.

## D6: FEXCore is the M1 translator engine; our own AOT discovery and per-title cache on top (2026-10-12)
Context: D4 left open LLVM AOT vs a direct emitter vs borrowing FEX. arvindfroi's runner `aps5-fex` (mugurc/AnyPS5#4, merged here) already runs guest programs through FEXCore in-process.
Options and evidence (`.mac-work/research/translator-perf.md`, `runner-arch.md`):
- (a) FEXCore JIT + our AOT discovery (`.eh_frame`, relocations, exports, init arrays) + persistent cache: est. 0.7–0.9× native on integer loops, 0.6–0.9× SSE, 0.4–0.7× AVX2 (no SVE on M3: upper YMM halves live in memory); 2–4 agent-days.
- (b) LLVM lifter + JIT fallback: published lifters run 1.5–4.8× slower than native (Lasagne, Instrew, Elevator); yabanana's #1553 covers 3 scalar functions and relies on per-game annotations (forbidden by MISSION §1). 3–6+ agent-weeks, and still needs a JIT.
- (c) Own emitter: months to reach FEX's coverage (AVX2, x87 80-bit via SoftFloat, TSO modes).
- Measured on this M3 (cycles/op, `.mac-work/research/tsobench/`): LDAPR/LDAPUR 0.60 vs LDR 0.35; STLR 1.0 vs STR 0.5; ordered vector accesses 11–21×. Taken under heavy load: ratios valid, absolute ns not.
Licence: arvindfroi/FEX `cad0c096` audited (`.mac-work/research/fex-licence.md`): FEX-2607 + 55 willfaust commits ending at `ac555dd8` (MIT) + 15 FEX-Emu backports (MIT, byte-identical to upstream) + 6 arvind commits; no willfaust commit after `ac555dd8` (his GPL-3 switch is `188e8224`, 2026-08-28). Nested deps: fmt (MIT), xxHash lib (BSD-2), range-v3 (BSL-1.0), unordered_dense (MIT), cephes (BSD), SoftFloat-3e (BSD-3). All GPL-2.0-compatible.
Choice: (a). Pin `3rdparty/FEX` at `cad0c096` (no `branch =` drift), guard it with a licence test, keep a git bundle backup in `.mac-work/ref/`, take future FEX fixes only from FEX-Emu (MIT), never from willfaust after `ac555dd8`. Keep FEX patches local (FEX-Emu's CONTRIBUTING forbids AI-generated contributions, and we push nothing without the user). Revisit only if M1 microbenchmarks stay below 0.7× native on integer/SIMD loops after tuning; LLVM stays an option for measured hot functions in M4.

## D7: CPU translator test oracle: FEX's x86 ASM unit tests, no Rosetta (2026-10-12)
Context: MISSION §6.4 (no x86 hardware here). FEX ships ~2,222 nasm unit tests whose expected register/memory values are checked on real x86 hardware in FEX's CI (MIT). Choice: port them to run under `aps5-fex` as the per-instruction oracle, plus SDM-derived expectations for anything they miss. Rosetta is not needed as an oracle, so the user question in §6.4 is moot. 1,020 tests use scratch addresses below 4 GiB, which macOS arm64 can't map: the harness moves 1,007 of them up by `0x10_0000_0000`; 13 can't be moved (recorded skips). All 2,222 assemble with nasm 3.02.

## D8: M2 shape: MetalBackend from IR + the VulkanDevice API implemented on Metal (2026-10-12)
Evidence (`.mac-work/research/m2-scope.md`): the IR (SSA, 469 opcodes, structured if/loop markers) has almost no SPIR-V dependency; SpirvBackend has one call site (`Recompiler.cpp:427-436`). The driver has no abstraction layer, but only 23 files (~27.4k lines) call Vulkan, and ~329 of the 336 `agc_driver_*` tests use only the public `VulkanDevice` API. Upstream notes the suite already passes under MoltenVK on Apple GPUs except 16 hardware-limit tests.
Choice: `MetalBackend` emits MSL at the SpirvBackend call site (SpirvBackend keeps compiling for differential tests). The `VulkanDevice` API is implemented on Metal in Objective-C++ under `libSceAgcDriver/specifics/macos`, swapped in by CMake on Apple with an overlay include directory, so upstream files stay untouched except CMake. Vulkan-Headers stays as header-only enum/struct types; nothing loads or calls Vulkan. SPIRV-Cross (core files dual "Apache-2.0 OR MIT", used under MIT) may be used in dev builds only, as a bridge and differential oracle, never shipped. Rejected: a Vulkan-on-Metal layer (violates D3), a new abstraction layer (rewrites a Vulkan path we can't test here), a clean-slate driver (loses behaviour the tests check). Metal 3 vs Metal 4 command API: decided in `docs/mac/M2_DESIGN.md`.

## D9: Two parallel build streams; milestones still close in order (2026-10-12)
Context: CLAUDE.md allows at most two live build directories; the M3 Air has 8 cores. The CPU stream (M0 → M1 → M3 integration) and the GPU stream (M2) share almost no files. Choice: main worktree + `build-arm64` for M0/M1; a git worktree `.mac-work/wt/metal` (branch `mac-native-metal`) + `build-metal` for M2; agents sharing a build directory serialise ninja/ctest with `tools/mac/buildlock.py`. Prep work that needs no build (tools, design, title preparation) runs in build-less worktrees. Each milestone is still closed in order with its exit criteria verified.

## D10: No hardware TSO; emulate x86 ordering with LRCPC (2026-10-12)
Evidence: the TSO enable bit is set only by the kernel (Rosetta); no API, entitlement or sysctl in the macOS 27 SDK exposes it, and the TSOEnabler kext needs SIP off. Hypervisor.framework lets a VM vCPU enable TSO (probed here with an ad-hoc-signed binary: worked; one VM exit ≥3 µs), which is too slow for a game process calling native libraries constantly. Choice: FEX's default ordering (integer loads/stores LDAPR/STLR, vector ordering off, stack accesses unordered); further elision (frame-pointer locals, TLS, read-only data) only with measurements in M4.

## D11: M3 homebrew title: DOOM Shareware for PS5 (2026-10-12)
`lowbit/ps5-doom` release v1.1.1 (`PPSA99666.zip`, sha256 `4d8e5b785b61696fbf1988c2d0d87c5c977734ff60b1ce11cc49134b0e711b5f`, matches the release's `.sha256`). Licence: GPL-2.0 (repo `LICENSE`, id's linuxdoom-1.10 source); the shareware `doom1.wad` is freely redistributable; the bundled `sce_module/libc.prx` is GPL-3.0 (test data, never linked into our code). Already documented upstream in `docs/user/COMPATIBILITY.md` (in game on Windows and Linux), and it presents through AGC compute, so it exercises the CPU translator, compute shaders, presentation, audio and pad input.

## D12: Community code merged into the fork (2026-10-12)
`mugurc/macos-port` (46 commits, macOS port under Rosetta + MoltenVK) and `arvindfroi/native-arm64` (arm64 libraries + FEXCore runner, mugurc/AnyPS5#3 and #4) are contributions to this GPL-2.0 project and were merged with history (`692016bf`, `faf0a695`). Their Rosetta/MoltenVK-specific parts stay for now (x86-64 build) but are not on our arm64 product path.

## D13: Vendor FEX at FEX-Emu FEX-2610 plus our macOS series (supersedes D6's pin) (2026-10-12)
Context: M1 needs JIT results persisted and reused. Evidence (`docs/mac/M1_DESIGN.md` "Code cache: verdict", file:line checked): the pinned FEX's FXCC cache is off by default (`EnableCodeCachingWIP=false`), has never worked on macOS (`CodeCache.cpp:607-612`), is invalidated by every FEX rebuild, never checks guest bytes, and needs a Linux-side generator; the `.eh_frame` AOT generator is never called. FEX-2610's DiskCache lives entirely in FEXCore, caches ordinary JIT output, checks guest bytes on every hit, is address-independent, capped at 1 GiB and uses `flock`. A trial rebase of the macOS delta (61 commits after dropping the 15 backports already upstream) stopped 14 times with 45 hunks (30 in code we build; the code-buffer cluster onto upstream's `SharedCodeBufferManager` is the hard part). Staying on the pin would mean 400–700 runner lines on the broken FXCC path; the rebase is ~3 agent-days but also brings ~3 months of upstream fixes.
Options for hosting the rebased FEX: (1) a fork on GitHub (needs a push, i.e. the user's approval); (2) FEX-Emu as the submodule plus patch files applied at configure time (pristine upstream, but every FEX edit during active development must be re-exported); (3) vendor it in our repo.
Choice: (3). `git subtree add --squash` FEX-Emu `FEX-2610` at `3rdparty/FEX`, then apply the slim macOS series (willfaust ≤ `ac555dd8` + arvind's, dropping Windows/iOS-only commits) with `git am --directory=3rdparty/FEX`, preserving authorship; FEX's own `External/` dependencies become submodules of our repo. Updates come with `git subtree pull --squash` from FEX-Emu only. The licence guard (`tools/mac/check_fex_licence.py`) checks the recorded upstream base and scans for GPL-3 markers. No push is needed, so the user question in M1_DESIGN R is moot. Keep the git bundle of `cad0c096` until M1 closes.

## D14: Additional homebrew test titles (2026-10-12)
Context: the user provides no titles ("make do with whatever you have"); commercial dumps are off-limits. A licence-checked search (catalog `blackbearreloaded/ps5-homebrew-catalog`, upstream issues, `project-oops/oops-apps`) gave three titles beyond DOOM (D11), all fSELF `eboot.bin` app folders, downloaded into `.mac-work/titles/dl-*` with verified sha256:
- PS5 OpenGL Showcase `PPSA99005` v1.0.1 (blackbearreloaded/ps5-opengl), GPL-3.0-or-later code, the author's own art/music, Mesa MIT. Real 3D through AGC draws: vertex + pixel + compute, depth, RGBA16F targets, `glMultiDrawElementsIndirectCount`, bloom. Upstream ran it (#3581, #4291) with #3512 (merged) and #4030 (open: flat shader inputs). sha256 `d16eb7f7…33c9f9`.
- Homebrew UI Lab `PPSA99050` 01.000.010, GPL-3.0-or-later, fonts OFL-1.1/Bitstream Vera. 2D UI through ps5-opengl; upstream home screen with #4030; quits if a GPU submission isn't done within ~2 s (#4399). sha256 `420a3fb5…cdaf`.
- SuperTuxKart `STKT00001` (project-oops/oops-apps `c590060e`), port code Apache-2.0 OR MIT, STK GPL-3.0+, stk-assets with per-folder licences. Mesa radeonsi command buffers via `sceAgcDriverSubmitDcb`. Never run upstream. sha256 `09bbc989…ec49` (GitHub digest; tag is rebuilt, digest saved beside it).
Excluded for licence/provenance: ccleste, Yamagi Quake II demo data, Quake3e, vkQuake (no data/licence), Unity-based titles, FSR4 Showcase, anything needing commercial data or ROMs.
These are test data run as guests, never linked into our code, so their GPL-3.0 licences don't affect the project's GPL-2.0-only code. Consequences: merge upstream PR #4030 when M3 starts; shader compilation must not stall a submission for ~2 s (compile in the background at registration, persistent cache); `tools/unself.py` must accept either fSELF magic for either file (the two SDKs swap them).

## D15: Build and test conventions for the arm64 fork (2026-10-12)
- The full macOS build defaults to arm64 (top-level CMakeLists.txt); x86_64 stays selectable with `-DCMAKE_OSX_ARCHITECTURES=x86_64`. Reason: this fork's product is arm64-only (D2).
- GPU tests carry an explicit per-test `gpu` label (468 tests in libSceAgcDriver, libSceAgc, libSceVideoOut): `ctest -LE gpu` is the non-GPU suite; `-L gpu` is run one at a time with timeouts only (GPU safety). A directory `LABELS` property is not enough: CTest ignored it for tests that set `WORKING_DIRECTORY`.
- `no_x86_build_path` (the M0 "no x86 on the arm64 path" proof, ~12 min on all cores) carries the `slow` label: routine runs use `-LE "gpu|slow"`, milestone ends run it.
- Guest-state operations (getcontext/setcontext/makecontext/swapcontext, sigsetjmp/siglongjmp, the sceKernelRaiseException register snapshot) on arm64 go through runner hooks; without a runner they validate arguments and throw a `std::runtime_error` naming the function, and their tests skip with that reason (M1 package G wires them).
- Agents sharing one worktree commit only named paths (`git commit -- <paths>`), because the index is shared.
