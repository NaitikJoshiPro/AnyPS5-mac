# Progress log

Append-only. Read the tail on every start or resume. Each entry: date, milestone, done, verification (commands + summarised results), disk usage, next step.

## 2026-10-12: handoff (planning session)
- Cloned upstream `main` (`9aa5fd29`) into `/Users/naitikjoshi/PS5`, branch `mac-native`, remote `upstream`. Partial clone (`--filter=blob:none`); submodules **not** initialised yet.
- Installed Ninja 1.13.2 via Homebrew.
- Wrote `CLAUDE.md`, `docs/mac/MISSION.md`, `docs/mac/DECISIONS.md`, `docs/mac/KICKOFF_PROMPT.md`, `tools/mac/disk_budget.sh`; added `.mac-work/` to `.gitignore`.
- Disk: 0.33 GB used of 100 GB budget; 233.8 GB free on disk.
- Not done yet: nothing built. Full Xcode not installed (Command Line Tools only).
- **Next:** M0. Fetch upstream, check what macOS work has merged since `9aa5fd29`, pull in arvindfroi's arm64 library work (mugurc/AnyPS5#3) and the relevant parts of mugurc `macos-port`, init submodules, relinker-only native build + tests.

## 2026-10-12: M0 started (execution session 1)
- Toolchain: Xcode still installing (CLT only: `xcrun --find metal` fails). Installed ccache 4.14.1 (`ccache -M 10G`). Rosetta present (only relevant as a possible test oracle, see MISSION §6.4). Homebrew LLVM 22 present (no lld).
- Merged `upstream/main` `e7c70a2d` (2 CI commits), then `mugurc/macos-port` (46 commits; 2 trivial conflicts), then `arvind/native-arm64` (arvindfroi's arm64 libraries, mugurc/AnyPS5#3, + FEXCore runner, #4; 18 hunks in 15 files). The libc conflicts were semantic: kept main's newer formatting/scanning code and routed arm64 through the hand-read guest `va_list` path (`VarArgsAbi.hpp`, `!APS5_GUEST_VA_LIST_IS_HOST` = Windows or arm64). Two merge bugs fixed during resolution: numbered `%N$Lf` arguments stored as host `long double` (8 bytes on arm64) now `GuestLongDouble`; `dl_phdr_info` grew `dlpi_adds/dlpi_subs` on main but `core/fex/src/Bridge.cpp` passes the 4-field struct, so only those fields are copied.
- Known follow-ups found during the merge (main-only code): `libc/src/ScanningWide.cpp:272` `static_assert(sizeof(long double) == 16)`; guest `va_list` still behind `#ifdef _WIN32` in `ScanningWide.cpp:383`, `Syslog.cpp:57,104`, `libSceLibcInternal/UserContext.cpp:39`.
- Submodules initialised shallow (`--depth 1`), except `3rdparty/FEX` (arvindfroi/FEX `cad0c096`), held until its licence history is audited.
- Verification: relinker-only arm64 build `cmake -S . -B build-relinker -G Ninja -DANYPS5_RELINKER_ONLY=ON -DBUILD_TESTING=ON -DCMAKE_OSX_ARCHITECTURES=arm64` + `ninja` → 169/169; `ctest -j6` → **121/121 passed** (35.7 s); `file build-relinker/core/relinker/relinker` → `Mach-O 64-bit executable arm64`. (Run on the tree with mugurc merged; re-run at M0 end.)
- M3 title candidate: DOOM Shareware homebrew `lowbit/ps5-doom` v1.1.1 (`PPSA99666.zip`, sha256 `4d8e5b78…711b5f` verified against the release's `.sha256`). Licence: GPL-2.0 (id's linuxdoom source, repo `LICENSE`), shareware `doom1.wad` freely distributable; bundled `sce_module/libc.prx` is GPL-3.0 (test data only, not linked into our code). Extracted to `.mac-work/titles/dl-ps5-doom-v1.1.1/extracted/`. `eboot.bin` and `libc.prx` are fSELF (`4f 15 3d 1d`): the relinker rejects SELF containers, so an unwrap step is needed (upstream #2599).
- Disk: ~0.5 GB.
- **Next:** full arm64 build (`build-arm64`, running), fix build breaks, non-GPU ctest; decide FEX licence from the audit.

## 2026-10-12: recon done, three streams launched
- Recon workflow (6 agents) reports in `.mac-work/research/` (summaries in `research/summaries/`): FEX licence audit (clean), runner architecture + gaps, M0 inventory (11 arm64 compile blockers, `check_no_x86.py`), M2 scope, Metal/Apple9 limits (probed on this M3), translator performance (TSO cost measured). Decisions D6–D12 recorded; MISSION §4/§5/§6.1/§3.2 corrected.
- Full arm64 build first attempt (`build-arm64`): configure OK (287 s), 1232/3011 steps, 10 failed objects (main-only x86 code), as predicted by the inventory.
- Removed `build-relinker` (superseded; to be re-verified at M0 end). Added `tools/mac/buildlock.py` (serialises ninja/ctest in a shared build dir).
- User answer on titles: no titles from the user; use freely distributed homebrew only ("make do with whatever you have"). Commercial dumps stay off-limits (CLAUDE.md legal rule).
- Running agents: (A) M0 completion in main worktree/`build-arm64`; (B) M1 prep in worktree `.mac-work/wt/m1prep` (branch `mac-native-m1prep`, no builds): FEX pin + licence test, code-cache verdict, `tools/unself.py`, DOOM unwrap + relink, `docs/mac/M1_DESIGN.md`; (C) M2 architect in worktree `.mac-work/wt/metal` (branch `mac-native-metal`, `build-metal`): `docs/mac/M2_DESIGN.md`, MslAbi, MetalBackend + MetalDevice skeletons, one compute test end to end; (D) search for more licensed PS5 homebrew (3D draws wanted).
- **Next:** on A's completion verify M0 exit criteria myself and close M0; then merge B's branch and start M1 implementation (work packages from M1_DESIGN.md); on C's completion fan out M2 work packages.

## 2026-10-12: network outage, agents resumed
- A DNS/API outage killed all four agents and their sub-agents mid-task. State at the time: (A) M0: 16 files fixed but uncommitted in the main worktree, plus `tools/mac/check_no_x86.py` and its test; (B) M1 prep: 3 commits on `mac-native-m1prep` (`986586b8` FEX branch tracking removed, `cf215cc4` FEX licence guard test, `e47605dc` `tools/unself.py`); was relinking DOOM; (C) M2 architect: worktree and `build-metal` configured, `MslAbi.hpp` uncommitted, plus temporary copies of A's WIP fixes (to be dropped when A's commits land); (D) homebrew search: ccleste excluded (unlicensed assets), checking ProsperoPuzzles and Homebrew UI Lab.
- All four resumed with context intact (SendMessage); each told to resume or re-launch its own sub-agents.

## 2026-10-12: Xcode and sudo setup (user-authorised)
- The user supplied their sudo password to end password prompts. It is not stored anywhere (repo, logs, memory, agent prompts); it was used once for:
  - `DevToolsSecurity -enable` (debugger attach no longer raises the "Developer Tools Access" dialog);
  - Xcode 27.0 (27A266a) at `/Applications/Xcode.app`: licence accepted, `-runFirstLaunch` OK. `xcode-select` still points at the Command Line Tools on purpose (switching mid-build would swap compilers under live build dirs); use `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun <tool>` for Xcode tools. The Metal toolchain (`metal`, `metallib`) is a separate component: `xcodebuild -downloadComponent MetalToolchain` running;
  - `/etc/sudoers.d/anyps5-powermetrics` (0440 root:wheel, `visudo -c` OK): `naitikjoshi ALL=(root) NOPASSWD: /usr/bin/powermetrics` only. Verified: `sudo -n /usr/bin/powermetrics --samplers cpu_power -i 200 -n 1` → CPU 1167 mW, combined 1244 mW (idle-ish).
- **Cleanup at project end:** delete `/etc/sudoers.d/anyps5-powermetrics`, optionally `DevToolsSecurity -disable`; the user will rotate the password.

## 2026-10-12: M0 complete (native arm64 build)
Done by the M0 agent on `mac-native` (`b1c5aaf9`..`e65e3cf3`), exit criteria re-checked by me from its artifacts:
- Compile fixes: DirectMemory physical-span index + stack-protect test (any macOS build), NEON `WriteChanged` that only stores changed bytes (fuzzed 200,000 cases vs scalar), libc long double / guest `va_list` on arm64, libkernel guest register state through the runner hooks (`sigsetjmp`/`siglongjmp` via the bridge; `getcontext`/`setcontext`/`sceKernelRaiseException` validate then throw naming the missing runner; listed in TechnicalDebt.md). Test-found fixes: libkernel imports libc's register hooks by `_nid_no_patch` names (NID patching had made libkernel.prx unloadable: `Symbol not found: _LibcCaptureRegisters`), guest stack mprotect in 16 KiB host pages, `swscanf_s %Lf`, `guest_udp` byte compare, AMM tests need the guest arena, `fcntl`/`open` guest va_list. Tooling: per-test `gpu` labels (directory LABELS was ignored for 353 tests with WORKING_DIRECTORY), `check_skips --list` prints the test's own reason, `tools/mac/check_no_x86.py` + ctest `no_x86_build_path` (labelled `slow`, ~12 min, all cores).
- Note: my docs commit `e77fb099` accidentally carried the agent's staged arm64-default hunk (shared index in one worktree). Content correct; from now on every commit names its paths (`git commit -- <paths>`).
- Verification:
  - Relinker-only: `cmake -S . -B build-relinker -G Ninja -DANYPS5_RELINKER_ONLY=ON -DBUILD_TESTING=ON -DCMAKE_OSX_ARCHITECTURES=arm64` + ninja → 169/169; `ctest -j6 --timeout 120` → **122/122 passed**, 0 skipped (`.mac-work/logs/relinker-m0-ctest.xml`); dir deleted afterwards.
  - Full build: `ninja -C build-arm64 -k 0` → 0 `FAILED:`.
  - prx: `ninja -C build-arm64 libs`; 123 libraries (+4 test fixtures) all `Mach-O 64-bit dynamically linked shared library arm64` (re-checked: 0 mismatches), `codesign --verify --strict` 0 failures (re-checked), `_nid_postfix` exports 1219 → 2 (libkernel `exit`/`rename`, intentional), 28,959 NID-shaped exports.
  - Non-GPU: `ctest --test-dir build-arm64 -LE gpu -j6 --timeout 120 --output-junit .mac-work/logs/ctest-nongpu-2.xml` → **453 tests: 449 passed, 4 skipped, 0 failed** (790 s; re-read from the JUnit). `ctest -N -L gpu` = 468 / `-LE gpu` = 453.
  - Skips (`tools/check_skips.py --list`): `guest_amm_placement` (guest arena is Linux/Windows only), `guest_raise_exception` (needs the arm64 runner to deliver), `guest_user_context`, `guest_user_context_exit` (getcontext/setcontext need the arm64 runner). Runner-dependent ones run their argument checks first. All three runner skips are M1 package G.
  - Flaky: `guest_tcp` 20/20, `host_thread_local` 20/20 under load ~75; no intermittent failures seen.
  - No x86 on the arm64 build path: `python3 -I tools/mac/check_no_x86.py build-arm64` → 1,632 TUs, **0 findings**, 0 not preprocessed.
- Known gaps carried to M1: 16 long-double math exports (`acosl`…`truncl`, `frexpl`) return wrong values to a guest (the bridge passes no x87 arguments); AMM calls throw on macOS (no guest arena); arm64 `sigsetjmp`/`siglongjmp` untested until the runner runs.
- Build dirs: `build-arm64` (main), `.mac-work/wt/metal/build-metal` (M2). Disk: 6.81 GB of 100 GB (+1.8 GB Homebrew `llvm` 23 in the Cellar, pulled in by `lld`, not measured by the script).
- **Next:** M1. Merge `mac-native-m1prep`, check out pinned FEX, build `aps5-fex`, baseline `fex_*` tests; wave 1 (A ASM oracle, D1 bridge signatures, E image registration/dlsym, H benchmark kernels) in parallel; R phase 2 (vendor FEX-2610) when its series is ready; then B, C, F, D2, G.

## 2026-10-12: M1 started
- Merged `mac-native-m1prep` (`2010f6d2`; one CMake conflict: kept both `check_no_x86_tool` and `unself` tests); removed its worktree and branch. M1-prep results: FEX pin without branch tracking + `fex_licence`/`fex_licence_checker` tests; `tools/unself.py` + 19 tests (ctest `unself`); DOOM unwrapped (`.mac-work/titles/PPSA99666/`) and relinked to a Linux x86-64 ELF for `aps5-fex` (`.mac-work/titles/PPSA99666-relinked/`, `.eh_frame` covers 98.2% of the eboot's code, 1,163 FDEs; the title's libc has an all-zero `.eh_frame`); `docs/mac/M1_DESIGN.md` (packages R, A–H, schedule). Tools: lld 23.1.3, nasm 3.02.
- Checked out the pinned FEX (`cad0c096`, shallow) + fmt, xxhash, unordered_dense, range-v3 in the main worktree; first FEX-enabled build + baseline `fex_*`/`unself` tests running (`.mac-work/logs/m1-base-*.log`).
- Running: workflow `m1-wave1` (A ASM oracle with pin baseline, D1 bridge signature generator, E image registration/dlsym/arena, H microbenchmarks; each independently verified, one fix round); package R phase 1 (FEX-2610 slim series in `.mac-work/ref/FEX-rebase`, no build) — phase 2 (vendor into `mac-native` per D13, build, compare A's pass list) on my go; M2 architect in `.mac-work/wt/metal`.
- M1 baseline (pinned FEX `cad0c096`): first FEX-enabled build failed to link `aps5-fex` (bundled `libxxhash.a` never built: the external FEX build only built FEXCore/FEXCore_Base/JemallocLibs) → fixed in `a11655be` (`-DCMAKE_DISABLE_FIND_PACKAGE_xxhash=TRUE` + build `xxhash`). Then `ctest -R "^fex_(fixtures|bridge_tables)$"`: **`fex_fixtures` passed** (12.7 s: arvind's 21 guest programs run natively through FEXCore on this M3); `fex_bridge_tables` **failed** as expected: main added a variadic `makecontext` and long-double math exports (`acosl`, `frexpl`, `atanl`, `sinl`, …) that the positional bridge can't pass/return (packages D1/D2). `fex_licence`, `fex_licence_checker`, `unself` passed.

## 2026-10-12: M2 foundation done, M2 wave 1 launched
- M2 architect finished on `mac-native-metal` (9 commits incl. merge of `mac-native` M0 fixes `d7fba2a4`): `docs/mac/M2_DESIGN.md` (decisions → D16; work packages B1–B6, D1–D6, T1; waves), `MetalBackend` emitting MSL at the SpirvBackend call site (structured flow, guarded loops, integer ALU, buffers, compute builtins, wave32 ballot; MSL cached in disk cache format 28; SpirvBackend unchanged), Metal 4 device behind the `VulkanDevice` API (residency set, event timeline, guest memory wrapped in place, descriptor table, push data, loop-guard faults), `Metal.cmake` twin swapping.
- Verification (logs `.mac-work/logs/metal-*`): all 286 execution test executables build against the overlays; `ctest -j1 --timeout 30 -R '^(agc_driver_vop1_control|agc_metal_msl_compile)$'` → 2/2 passed (first real RDNA2 test executed on the M3 GPU via MSL); CPU-only recompiler tests 5/5; G1 baseline (196 compute-class tests, one at a time, 115 s): **29 passed, 29 skipped, 138 failed, 0 hangs/timeouts**.
- Launched workflow `m2-wave1`: B1 (core/flow, wave64 first), B2 (ALU, f64 in software), B5 (buffers, typed formats, BDA, flat/global), D1 (device: GDS/LDS arena/page table bindings, indirect, self-tests), D2 (recorder → MTL4 command stream), T1 (gpu_suite.sh, skip allowlist, MSL corpus check, Vulkan-internal test twins, port ledger); each verified independently with one fix round. GPU tests serialised machine-wide through the `build-metal` buildlock.
- Process incident: SendMessage to agents running inside a workflow does not reach them; it starts a duplicate from the agent's transcript. Relaying B5's MSL 4.0 finding this way created duplicate B1/D1/B5 instances in the metal worktree (the D1 duplicate noticed the original's edits and held off). The duplicates were told to stop and revert only their own uncommitted hunks; the duplicate B1's fence fix `caed64fc` stays. From now on, coordinator notices go to `.mac-work/coord/{m1,m2}.md`, and workflow prompts tell agents to read them; agents reach the coordinator with SendMessage to "main". MSL 4.1 switch deferred to wave 2 (seq_cst fence already fixes the compile failure).
- M1 R phase 1 done (`.mac-work/ref/FEX-rebase/`, bundle `.mac-work/ref/fex-2610-macos.bundle`): 36-commit series on FEX-2610 (`14c92681f`): 28 Will Faust + 5 arvind commits (authorship kept, each cites its source sha) + 3 ours (keep Apple `-march`; `AllocatorHooks.cpp` include gap; `readdir` instead of Linux `getdents64` for DiskCache). Windows/iOS-only code stripped from kept commits (macOS-compiled code identical to the pin); 43 of 76 commits dropped (15 backports, `e941d55cf`, 26 empty after stripping, 1 comment-only). All 68 FEXCore TUs pass `-fsyntax-only` with the real flags; the vendoring dry run (`subtree add --squash` + `git am --directory=3rdparty/FEX`) is byte-identical. Licence: FEX-2607..2610 756 commits/26 authors, MIT, no GPL-3 markers; `-DENABLE_GDB_SYMBOLS=OFF` added (GPL-3 `jit-reader.h`). Pin problems carried over for later packages: ~900 lines of iOS diagnostics in the JIT compile path (error logs, reads at hardcoded Windows addresses), macOS signal-mask guard never restores, unaligned backpatch writes the read-only alias (WP B), macOS mutex/condvar spin with `usleep` and ignore timeouts (WP H). Phase 2 (vendoring + build + parity) waits for wave 1 to finish (A's pin baseline; GuestCpu.cpp is being edited).
- M1 wave 1 so far on `mac-native`: E `11fedbb8` (runner images in libc's registry; x86 stubs for native symbols), `60eee40d` + `e59d6c50` (guest arena reserved on macOS; AMM works; `guest_amm_placement` no longer skips); H `d4fb332a`…`304b7175` (microbenchmarks translated vs native, `APS5_FEX_CONFIG`/`APS5_FEX_QOS` runner options). A and D1 in progress.

## 2026-10-12 ~20:20: KERNEL PANIC caused by our GPU test; GPU work frozen
- The Mac panicked (purple screen, reboot) while T1's serialized full-suite run executed `agc_driver_f16_denormal_flush_tests` (pid 20052): `Kernel data abort` in `com.apple.iokit.IOGPUFamily(162.11)`, `far 0x10` (NULL+0x10), macOS 27.0.1 26A434. Only one GPU process was running (B5 waited on the lock), so a single unprivileged Metal 4 compute workload panicked the kernel: an Apple driver bug that our workload triggered. Panic file and a ready-to-file Feedback report: `.mac-work/feedback/` (the user files it in Feedback Assistant; it needs their Apple ID).
- Both workflows died with the reboot. Committed work survived; in-progress uncommitted M2 edits remain in `.mac-work/wt/metal` (MetalDevice.mm, MetalDeviceState.h, MetalPresentation.mm, D1DeviceTests.mm, recompiler CMakeLists, MslAluMath.cpp, MslRuntimeBuffer.cpp, new MslAddress/MslBda/MslBufferShared).
- **GPU work frozen** until (a) the shader of F16DenormalFlush is analysed offline (no re-run), the test is quarantined, and (b) new GPU rules are in place (single dedicated GPU runner, validation on, nothing GPU while builds load the machine, new shader families on minimal inputs first). The user asked to continue the work; CPU-only M1 resumes.
- M1 wave 1 results (implemented and committed; the verify rounds were cut off by the panic):
  - A (`ace93c10`, `0dade124`, `161f7c8a`, `3b1673cf`): FEX's ASM suite under `aps5-fex-asm`: **2,121/2,222 pass, 73 known failures (listed with reasons), 28 skip, 0 unexpected failures** on the pin (`core/fex/tests/asm/baseline-cad0c096.txt`).
  - D1 (`3c340812`, `2af1b441`, `60bab994`): bridge signatures generated from the prx sources at build time; five more variadic exports bridged.
  - E (above) and H (`d4fb332a`…`624b4688`): native time / translated time (CPU time, geomean, avx2): integer 0.57 (0.71 with TSO off), memory 0.44 (0.56), SIMD 0.57, one-thread atomics 0.97. **0.7× gate not met** under default TSO. Costs: TSO ldapur/stlur on every GPR access, indirect control flow, VEX.128 upper-half zeroing, guest→native call 105 ns (native 1.3 ns), native→guest callback 124–129 ns. int_lexer generator hit the unaligned-TSO SIGBUS (package B).
- M1 R phase 2 done (`f8e268e7`…`9e4b6b65`): FEX-Emu FEX-2610 (`14c92681f`) vendored at `3rdparty/FEX` (subtree-compatible squash + merge), 36-commit macOS series via `git am --directory` (tree `9afa22ef`, identical to phase 1), FEX deps as root submodules, runner on the FEX-2610 API (`d4553901`, `-DENABLE_GDB_SYMBOLS=OFF`), licence guard for the vendored layout (`14e2a450`), `docs/user/THIRD-PARTY-NOTICES.md` (`9e4b6b65`). Verification: full `ninja -j4` OK; `ctest -j1` fex tests 11/11 passed; ASM suite vs `baseline-cad0c096.txt`: **0 regressions**, default 2165 pass / 73 xfail / 28 skip / 4 fail (the 4 = AVX-VNNI tests, harness lacks the `AVX_VNNI` host-feature name; 96 new tests in 2610, 88 pass). Benchmark (wall clock, avx2, alternating runs, same session): integer 0.68 (pin 0.64), memory 0.52 (0.52), SIMD 0.52 (0.53). Gate (0.7×) still not met. Disk 7.29 GB.
- Next: M1 wave 2 (CPU only, low priority): agent 1 = iOS diagnostics removal + AVX_VNNI harness name + B (unaligned SIGBUS backpatch via RW alias) + C (code pool reclamation) + F (DiskCache wiring, AOT discovery, coverage report, run-twice test); agent 2 = D2 (bridge marshalling from signatures + direct reverse thunks; bridge call cost) + G (guest-state ops through the runner).
- Panic analysis (CPU only, `.mac-work/feedback/`): the F16DenormalFlush shader never reached the GPU. All 84 generated MSL sources failed to compile (uncommitted mid-edit MetalBackend code: `coherent` used as an identifier, duplicate `aps5_f32_nan`), the test threw before encoding any work, and the panic hit during process teardown (terminated snapshot, IOGPUFamily). Contributing: an untested half-edited prx (linked 20:18:18 with D1's in-progress recorder integration) and a device residency set added to the queue but never removed (`MetalDevice.mm:390`). Same test had passed at 19:54 and 20:10 under shader validation. Feedback report corrected accordingly. Quarantine mechanism `8c5b8e2d` (`tools/mac/gpu_quarantine.txt`, fail-closed). Safety proposal: `.mac-work/feedback/gpu-safety-proposal.md` (GPU runs only from a clean committed runner worktree, refuse under load/compiles, canaries first, MSL must compile before launch, validation on, teardown fixes). GPU stays frozen pending the user's OK.
- M1 wave 2a done (`d3b103ef`..`26929a58`): iOS diagnostics removed from the JIT path (~2,270 lines); signal-mask restore; harness knows AVX_VNNI and the runner reports I8MM/DotProd; FEX ExternalProject BUILD_ALWAYS (edits under 3rdparty/FEX previously never reached libFEXCore.a!); B: backpatch via RW alias + BUS_ADRALN → HandleUnalignedAccess + `unaligned` fixture; strict in-process split locks on by default (4 copies: 12/12 correct; without: 11/12 torn); C: Darwin MADV_DONTNEED doesn't zero → MADV_ZERO+memset for lookup cache/callret stack, `CodePool` reclamation (`fex_code_pool`, `fex_code_churn`); F: DiskCache wiring, `--precompile` AOT seeds, coverage recorder + `tools/mac/fex_coverage.py` (`fex_disk_cache`, `fex_aot_seeds`, `fex_coverage_tool`).
  - ASM suite (`baseline-fex2610.txt`): **default 2237 pass / 5 xfail / 28 skip / 0 fail**; full 2238/4/28/0; 0 regressions.
  - Run twice on one cache: second run compiles **0 blocks**, identical output. AOT coverage of executed code: callbacks 93.0%, heap 97.0%, crypto 100%, floats 100%, zen2 95.0%, exception 82.1%, jump 93.1%. DOOM `app.elf` precompile: 82.3% of executable bytes (~98% of .text), 2.4 s, 4.8 MB cache.
  - Bug found: DOOM's unwrapped guest `libc.prx` has every dynsym export at value 0x50 (1,874 exports → 7 seeds): relinker/unself bug that would also break import resolution at run time (M3 blocker). Agent assigned.
  - Benchmark (avx2, wall clock, median of 3, load 3–4): integer 0.68, memory 0.53, SIMD 0.51, atomics 0.80; guest→native call 45 ns (was 105), callback 66–77 ns (was ~125). Gate 0.7× not met.
  - Open: FEX JIT temp-buffer 4 KiB guard can't be set on 16 KiB pages (CodeBufferOverflow.asm known failure); one unreproduced `fibers` segfault under -j4 load (16/16 after).
- "libc exports at 0x50" is not a bug: DOOM/Showcase/UI Lab ship an SDK placeholder `libc.prx` (127 non-zero bytes of code, all exports → a `pxor; xor eax; ret` stub at 0x50); titles take libc from HLE `libSceLibcInternal` (0 `#guest` imports). `950f05cf2`: unself notes placeholder modules, accepts either fSELF magic for either file (SuperTuxKart swaps them), 22/22 tests. All four titles re-unwrapped; DOOM, Showcase, UI Lab relink; **SuperTuxKart doesn't relink: `R_X86_64_TPOFF64` (13 relocs) not in the relinker allowlist** (`ValidationPolicy.cpp:18-34`) → M3 work item.

## 2026-10-12: M1 complete (CPU translator), speed target not yet met
- Wave 2b (`6d04bc6b`, `7dd8018f`, `5624e54a`, `620b043f`): bridge marshalling from generated signatures (stack args, HFAs, >16-byte structs by reference, hidden-pointer results, x87 long double ↔ double for 16 math exports, strtold/wcstold in st(0)); new test library `libAps5BridgeAbi` + fixtures abi-registers/stack/memory/long-double/variadic/callbacks; 4,096 static arm64 callback thunks; imports via FEX's thunk instruction instead of `syscall`; guest-state ops through the runner (get/set/makecontext, siglongjmp, sceKernelRaiseException on the calling thread): `guest_user_context*` and `guest_raise_exception` now run instead of skipping.
- M1 exit verification (me): `ninja -C build-arm64 -j4` OK; `ctest --test-dir build-arm64 -LE "gpu|slow" -j4 --timeout 300` → **2739 tests, 0 failures, 28 skipped (all known ASM-harness skips)** (`.mac-work/logs/m1-exit-nongpu.xml`).
- Exit criteria:
  1. Per-instruction suite vs oracle (FEX ASM tests, hardware-recorded): 2237 pass / 5 xfail / 28 skip / 0 fail. ✔
  2. Guest fixtures (imports, TLS, threads, C++ exceptions, C cleanup, variadic printf, qsort callbacks, long double, contexts, fibers, unaligned atomics): 34 programs pass translated. ✔
  3. Discovery report per binary + persisted JIT reused: coverage tool, 82–100% AOT on fixtures, DOOM eboot 82.3% of executable bytes precompiled; second run compiles 0 blocks. ✔
  4. Microbenchmarks recorded: integer 0.68, memory 0.53, SIMD 0.51, atomics 0.80 (avx2, wall clock); guest→native call 41 ns, callback 66 ns. ✔ recorded, **target ≥0.7× on integer/SIMD not met**. Known costs: TSO ordering on every GPR access, VEX.128 upper zeroing, indirect branches, FPCR switching around host calls (~1/3 of bridge cost). Carried to M4 (translator hot-path optimisation); D6 revisit only if M4 tuning fails.
- Open (TechnicalDebt): positional callbacks via fault path, flags lost on fast-path resume, cross-thread raise, long double at double precision, no swapcontext export, FEX JIT temp-buffer guard on 16 KiB pages, STK TPOFF64 relocations.
- Disk 8.01 GB.
- **Next:** M2, gated on the user's OK of the GPU safety proposal (GPU frozen since the panic).
- SuperTuxKart relinks now (`36a74c44d` executable TLS relocations TPOFF64/DTPOFF64/DTPMOD64 → Linux `.rela.dyn`, applied by the runner's variant II layout; static resolution on Windows/macOS outputs; test `executable_tls_relocations`, fixture `tls-exec`; `5dc07a7e6` grow the program header table into unused zero padding). Relinker tests 103/103, fex fixtures 35/35. Not run (needs GPU).

## 2026-10-12 ~22:50: paused at the user's request
- M2 restart agent stopped while bucketing failures per package (its work up to then is committed on `mac-native-metal`: `c666cccdd`…`6273496cb`: merge of mac-native, D19 guards in gpu_suite, teardown hygiene + tests, f16 test released from quarantine with validation evidence). Last validated run (canaries passed): G2 0/43, G3 0/32, G4 14/39, G6 10/24, G7 17/49 (total 41/187 excluding G1). No GPU/build processes left running.
- **Resume:** finish the per-package failure buckets + refreshed work list in M2_DESIGN.md (restart agent's step 5), G1 count, then fan out the M2 packages (textures D4/B4, pipelines D5, draws B6/D6, presentation, atomics/LDS B3, guest memory D3, D1 recorder integration, PR #4030).

## 2026-10-13: M2 resumed (wave R)
- User said continue. Disk 8.58 GB / 100 GB. Metal Toolchain download still pending (offline `metal` unavailable; runtime compile gate in use).
- Instead of a separate bucketing pass, each package agent baselines and buckets its own group. Agents (Agent tool, worktree `.mac-work/wt/metal`, shared `build-metal` under buildlock):
  - **DRV**: D1 recorder integration and D3 guest memory/BDA page table → G4, plus the driver side of G1/G7.
  - **SHD**: B2/B3/B5 and the MSL 4.1 switch → G1, G6.
  - **TEX**: D4/B4/D5 textures, images, pipelines and the pipeline cache → G2.
  - B6/D6 (draws, presentation → G3/G5) start once TEX's render pipelines land.
- Protocol (coord/m2.md): agents commit their own files before every gpu_suite run, because D19 refuses while the tree is dirty.
- 2026-10-13 wave R status (all counts under `--validation`, logs `.mac-work/logs/gpu-suite/`):
  - **DRV done**: D1 recorder integration and D3 guest memory (M2_DESIGN "DRV results", `b6cd25d1c`; hand-off `.mac-work/coord/drv-handoff.md`).
    - BDA table built per dispatch; MslAbi v2 atomic lock table; overlapping V#s share one mapping; faults reported (not thrown) as upstream; host imports are no-copy buffers; pipeline cache wired.
    - G1 **178/182**, G4 **18/39** (was 14), G7 24 pass / 2 fail / 29 skip.
  - **SHD**: G1 104 → 178/182; G6 10 → **23/24**. Only image int64 atomics left, now with TEX.
    - MSL 4.1, wave/LDS/GDS, 64-bit atomics through the lock table.
    - `agc_driver_lds_atomics64` passed one supervised validation run (`--rerun-held`, `2e6f803c`); still held for normal runs.
  - **TEX**: G2 0 → **33/43**.
    - Pipeline cache: second run compiles 0 pipelines (exit criterion 4).
    - MSL min/max sampler reduction emulated (Metal ignores reductionMode on the M3).
    - Texture.mm and ShaderResources.mm twins of upstream.
    - Re-applying `images->Attach(context)` failed validation twice (depth placeholder texture type, then the placeholder clear); reverted both times, being fixed with a dedicated Attach-path test.
  - Coordinator fixed the shader-cache reviewed-names build break (`9e932a592`).
  - **Next**:
    - DRAW (fresh agent, D6 draws and presentation) started; SHD moves to B6 (stage interfaces); TEX finishes Attach, the pending-storage family and image int64 atomics.
    - Open: D3 Metal twins still missing (`agc_host_import_unmap`, `agc_unit_shadow`, `agc_unchanged_cpu_stamps`, `agc_storage_baseline_settled`).
- 2026-10-13 ~03:15: **user order: no GPU or CPU tests until the user greenlights** (gaming on the laptop). Agents code and build-check only (`nice -n 15`, -j2). Earlier, ~02:07–03:15, GPU runs were refused anyway because the user's Docker Desktop VM held load at 45–88.
  - Last validated runs before the freeze:
    - G3 **7/34** (from 0; `shd-g3a`).
    - G1 181/182 (depth_fast_clear and shader_fusion now pass).
    - G2 **38/43** on the upstream image path (`20261012T214250Z-G2_re…`).
    - agc_driver_image_atomics64 passes, so G6 has everything but the quarantined look_back_scan.
  - TEX handed back (`.mac-work/coord/tex-handoff.md`).
    - The upstream image path is the default (`12dddd4ce`, not yet GPU-verified); the bring-up path is deleted after G1/G2/G4 verification.
    - Left over: 3 pending-storage G2 tests (need GPU diagnosis), and emulated_color_compare (now SHD's: per-mode sampler uses in the reflection).
  - DRAW: draw path, presentation (CAMetalLayer + headless) and CMASK landed. Being fixed offline: copy fallback for executable/unreadable guest memory (13 G3 tests), layered targets, the vertex_fetch ABI check, dual source, stencil export.
  - SHD: B6 vertex/fragment emission plus rect lists; corpus 313/359 compile. Being fixed offline: pixel LDS, wave64 pixel reduction, draw offsets. Also the GDS append bug.
  - Quarantined: agc_driver_look_back_scan (20 s GPU batch timeout, machine recovered; fix `28a79003` awaiting a supervised rerun).
- 2026-10-13 (test freeze continues):
  - SHD handed back (`.mac-work/coord/shd-handoff.md`). SHD2 did the B6 remainder offline, build-checked only (`51a07a73f`, `5b828b370`, `104f66274`):
    - wave64 readlane and pixel half-wave fixes;
    - raster-order pixel shaders (MslAbi v4, fragment slot 5);
    - graphics LDS per invocation, no arena;
    - interpolation_f16 allowlisted as an Apple9 limit (Feature Set Tables 2026-05-21: per-vertex values are Apple10).
    - Its test queue is in M2_DESIGN "SHD2 results".
  - DRAW (still active): resident colour targets, pending flushes, depth-only passes, M3 macOS window/input/audio survey and fixes (HiDPI drawable, title throttling).
    - Doing next offline: main-thread event pump, Cmd+Ctrl+F fullscreen, mach_wait_until vblank.
    - After the freeze: SDL_X11/SDL_VULKAN off on Apple, raster-order slot 5 binding, gds_append_consume trace.
  - **When the user greenlights tests:**
    - Run DRAW's queue: draw tests, cmask, G3, G4, G1.
    - Run SHD2's queue.
    - Verify the upstream image default on G1/G2/G4, then delete the bring-up path.
    - look_back_scan supervised rerun.
    - First windowed run (supervised).

## 2026-10-13 ~07:00 — M2: user greenlit tests again; no "Apple limits"
- User lifted the test freeze, and ordered that missing Metal features be EMULATED rather than allowlisted (conservative raster, per-vertex fragment values; later mesh and tessellation, M3 backlog in M2_DESIGN).
- Shared "triangle vertices" mode (MslAbi v5, vertex slot 6): header 7fb900ce2 (SHD3), driver side 9ec6a7ffb, conservative snippets 07fe3aac8/a8aa31286 (DRAW). Built, not run; the SHD3 emitter is pending.
- Results, all under --validation:
  - draw_targets/primitives/cmask 3/3 (gpu-suite/20261013T002138Z)
  - G3 21/34, up from 7 (20261013T002224Z-G3_G4)
  - G4 25/39, up from 18
  - G1 181/182 + 1 skip (20261013T003936Z-G1)
  - SHD2 queue 7/7 (shd3-t1)
  - vertex_fetch now passes
- GDS bug narrowed down: GPU atomics land in the D3 host import's pages, but the CPU still reads the seed after WaitIdle. The VM object is unchanged. Owned by D3, now DRAW2.
- Agents: DRAW handed off (context full). Now running: DRAW2 (D3 GDS, 9 driver G3/G4 failures, SDL options, windowed-run prep, which needs approval) and SHD3 (per-vertex emulation, alpha_to_coverage, dst_sel, interp param, wave64 ReadLane, MSAA ImageWrite, upstream image-path verification, pixel_interlock one-shot).
- Next: collect the results, then M2 exit verification.

## 2026-10-13 ~08:30 — M2: SHD3 done, bring-up image path deleted
- Clean runs at b91753d6a (--validation): upstream image path 293/303 (G1 181/182, G2 43/43, G3 33/33, G4 30/39, G7 6/6), bring-up path 285/303. Bring-up path deleted (d2a2b0827).
- Per-vertex fragment values emulated (D20, 3f14bf8ad): interpolation_f16 + pixel_barycentric_emulation pass and are off the allowlist.
- DRAW2 fixes:
  - GDS / no-copy over write-watched read-only pages (1497d1225, test agc_metal_device_imports) fixed most "destination unchanged" G3 failures.
  - Offscreen VideoOut segfault fixed (b8395ffb9).
  - First windowed run passed.
  - G5 headless pass: 12/13. frame_dump_queue still HOLD: the hang is fixed (frame dump worker) and the remaining FIFO write is being debugged.
- Agents now: DRAW2 (frame_dump_queue, windowed decode/overlay, msaa_enable_single_sample), SHD4 (pixel_interlock fault stop, conservative raster verification, G7 unbuilt tests, MSL mesh functions), MSIMG (multisampled storage-write emulation, 4 prepared_shaders tests).
- Remaining to emulate: mesh (SHD4), tessellation (design pending), MS storage writes (MSIMG).

## 2026-10-13 ~10:30 — M2: mesh shaders, conservative raster and G5 done
- DRAW2: G3 34/34, G4 35/39, G5 headless 14/14 (frame_dump_queue fixed: a0a16d65d, 8d9b4e68e), windowed present/decode/overlay rc=0 with 0 validation errors, msaa_enable_single_sample fixed (2f43c6ab9). G5 released from HOLD and gpu_suite now forces SDL_VIDEODRIVER=offscreen (d3e582be7); HOLD is now only lds_atomics64.
- SHD4: pixel_interlock passes (d80f1643a). Conservative raster is emulated and passes (D21, f27bde0f7). MSL mesh functions + MTL4 mesh pipelines (D23: 76b008405, f940c9b7d) make agc_driver_mesh pass. G3 34/34, G4 36/39 (gpu-suite/shd4-g3g4-v2). D19 slip: one direct test-binary run (0.04 s, harmless).
- Tessellation: MTL4 has no drawPatches. Design: object stage (LS+HS) + mesh stage (software tessellator + DS). Agent TESS is implementing it.
- Running: TESS (tessellation), MSIMG (MS storage writes, D22), T1 (agc_driver_graphics Vulkan-internal split + BdaDeviceTwin.mm).

## 2026-10-13 ~13:00 — M2: tessellation, MS storage, corpus, T1 twins done
- TESS: tessellation on MTL4 object/mesh pipelines (D24, MslAbi v7). The tessellator is a port of the D3D11 reference (MIT) and is bit-exact against it (agc_metal_tessellator: 131,088 factor sets, 0 failures). agc_driver_tessellation and prepared_graphics pass (tess-set1 6/6). G3 34/34, G4 38/39 + 1 allowlisted skip (tess-g3g4). Next, TESS2: open the shared front end for quad/isoline/all partitionings, capability-gated; the Vulkan target keeps its current set.
- MSIMG: multisampled storage writes (D22, MslAbi v6), the 4 prepared_shaders tests pass. Gap: guest MSAA T# 14/15 are rejected upstream too (M3 backlog).
- T1: agc_driver_graphics split (72/80 checks run), BdaDeviceTwin passes, CapabilityStorageImageMultisample advertised (c1b4b133e). G1 181/182.
- CORPUS: 338/338 SpirvBackend-accepted requests produce valid MSL (D25). shader_clock is emulated as a read counter (TechnicalDebt metal-shader-clock; later: real GPU timestamps). The corpus needs a re-dump now that mesh/tess have landed.
- REGRESSION: agc_driver_global_atomics64 and buffer_atomics_or_swap64 time out. Quarantined (bd73a0a1a). ATOM is on it (suspect 28a79003e, lock-table spin forward progress), along with look_back_scan.
- gpu_suite now holds a build lease for the whole run (065898379), so runs are no longer voided by other agents' rebuilds.
- M2 exit still open: ATOM fixes; lds_atomics64 HOLD release; f16_denormal_flush rerun (it caused the 2026-10-12 kernel panic; waiting for the user to be away from the laptop); corpus re-dump; full-suite verification.

## 2026-10-13 evening — M2 exit verification: all four criteria PASS (EXIT, 5415f5dfd at dba2dbd7d)
1. Corpus: 2,827/2,827 valid (fresh dump exit-dump1 + seed). 795 front-end-rejected, 54 spirv-rejected, 4 host-width-rejected (M3).
2. gpu-suite/exit-final1 (all, --validation --build-first): 502 run, 488 pass, 0 fail, 0 timeout, 14 allowlisted skips (pipeline-library Vulkan internals x12, descriptor pools x2). No binaries changed during the run.
3. G5 14/14 headless. Earlier windowed present/decode/overlay rc=0.
4. Pipeline cache: second run 0 compiled / 3 archive hits.
- CPU ctest -LE gpu 5,015/5,015 (56 known FEX skips). tools/mac unittests 68/68.
- Since the last entry: ATOM (atomics64 timeouts were slow validated compiles; MslLockLoop; look_back_scan fixed; quarantine empty), lds_atomics64 released from HOLD (HOLD now empty), TESS2 (all domains/partitionings, D26, MslAbi v8), TWIN-A/B (12 pending twins written, CmdClearColorImage keep-alive fix c1d3a0e15).
- Excluded: agc_driver_f16_denormal_flush_tests (caused the 2026-10-12 kernel panic). It needs a supervised run with the user warned.
- Running: MSAA (MSAA render targets + resolves; draws still refuse samples>1, a gap the suite didn't catch), BUDGET (Metal device memory heap and budget).
- Next: the f16 supervised run, merge mac-native-metal into mac-native, M3.

## 2026-10-13 — supervised rerun of agc_driver_f16_denormal_flush_tests (user approved: "go")
- This test caused the 2026-10-12 IOGPUFamily kernel panic. It now runs once, alone, via gpu_suite --validation, with the result in .mac-work/logs/gpu-suite/main-f16-supervised.
- IF THE MAC REBOOTED and this is the last entry: the test panicked again. Quarantine it in tools/mac/gpu_quarantine.txt, keep .mac-work/feedback/FEEDBACK_REPORT.md current (a second occurrence), and tell the user.
- 2026-10-13 ATOM (M2, 64-bit atomics timeouts): the quarantined agc_driver_global_atomics64 and buffer_atomics_or_swap64 30 s timeouts were CPU-side Metal compile stalls, not GPU hangs. There was no recorder "still running" line; the validated compile is 3.4 s of a 4 s test; in the same run corpus took 35 s instead of 4.5 s. 28a79003e is not the cause and also fixed look_back_scan's real hang. Commits: f858f772f (MslLockLoop, one forward-progress-safe lock loop for the LDS, 64-bit device and image atomics; CheckLockLoops in agc_metal_msl_wave_lds, negative-tested), f242e54a7 (PipelineCache reports compiles running longer than 5 s), ab52cce19 (released from quarantine, TechnicalDebt). Verification: `gpu_suite --validation --build-first --rerun-quarantined` atom-ga64-r3 / atom-or64-r2 / atom-lbs-r2 pass; `gpu_suite --validation --build-first G6` → 24/24 (atom-g6b, HEAD 1d3739b56); agc_metal_msl_wave_lds passes; check_test_registration clean. lds_atomics64 is still on HOLD (proposed to main). Disk 7.6 GB.

## 2026-10-13 — M2 TESS2: tessellation front end for every domain, partitioning and topology (MslAbi v8)
- Done (DECISIONS D26): `TessellationFeatures` target capability; capability-driven DecodeShaderStages/DecodeState (Metal: all domains/partitionings/topologies + VGT_HOS level clamps; Vulkan: unchanged reference set and messages); per-domain factor records (isoline 8, tri 16, quad 24 B); MslAbi v8 (clamps in TessellationDraw, isoline factor order swap, no mesh threadgroup for culled patches); vertex sharing per mesh slice; 1-px points for point output.
- Bug found: sub-word threadgroup arrays in the mesh function corrupted its corner array on the M3; fixed with 32-bit words (83ed2d518).
- New tests: agc_metal_tessellation_domains (19 guest draws vs the tessellator core), agc_metal_tessellator_gpu (641 factor sets, GPU core == CPU core).
- Verification: `python3 tools/mac/gpu_suite.py --validation --busy-wait 300 --build-first <tess set> agc_driver_graphics G3 G4` → G3 34/34, G4 38/39 + allowlisted skip, tessellation set 8/8; agc_driver_graphics re-run after a test-expectation fix → pass (gpu-suite/20261013T101928Z-..., 20261013T102451Z-...).
- Counts: domain program runs 502 vs 1,744 corners over the test draws (gap 3); mesh threadgroups 19 vs 21 (gap 6, CPU count).
- Disk: 8.0 GB of 100 GB.
- Next: SPIR-V side of D26 stays unverifiable here; remaining gaps in TechnicalDebt metal-tessellation (payload fallback, per-slice sharing only, PA_SU_POINT_SIZE, clamp order unmeasured).

## 2026-10-13 — M2 MSAA: multisampled Metal draws, CB resolves, sampled MSAA T#s (D27)
- Done: draws with samples > 1 render into the MultisampleTarget host image and 2DMultisample depth surfaces at the guest's sample positions; CB resolves (MTLStoreActionMultisampleResolve; integer formats sample 0); guest 2D MSAA T#s (type 14) read a rendered target through texture2d_ms on MSL targets only (Image2DMultisample shape; SPIR-V unchanged). Commits 4cbbc423b, 4eb587c25, dee6bce48, 52b10f41b; MslAbi unchanged (v8).
- Verification: `python3 tools/mac/gpu_suite.py --validation --busy-wait 1800 --build-first 're:^agc_metal_msaa_'` → gpu-suite/msaa-r3 5/5; `... G3 G4 G7` → gpu-suite/msaa-g347: G3 39/39, G4 38/39 + allowlisted skip, G7 55 + 13 allowlisted skips; check_skips and check_test_registration clean.
- Gaps: 8x on Apple9 (throws; design in TechnicalDebt metal-msaa-8x), 2D MSAA array T#s, MSAA T#s over unrendered memory, sample-rate interpolation, proportional A2C, MSAA size query sample count (TechnicalDebt metal-msaa).
- Disk 8.48 GB of 100 GB.

## 2026-10-13 ~18:40 — M2: MSAA2, emulated 8x and the rest of MSAA (D28, MslAbi v9)
- 8x on Apple9 is emulated as two 4x slices of a 2D multisample array (per-slice passes at the guest's positions, APS5_SAMPLE_BASE). 8 consecutive draws take 9 render passes. 4x rendered as 2 x 2x resolves bit-identically to native 4x.
- Also done: type-15 T#s (sampled heaps are texture2d_ms_array); resinfo w = log2(samples)+1, inferred from Mesa (main asked for the sample count, see D28); guest storage stores/atomics to rendered targets through a coherent storage shadow; sample/centroid interpolation; PS_ITER_SAMPLE; measured proportional A2C (0xaa01).
- Bug found and fixed: multisampled draws Sync()ed after every draw (8e64856a3).
- Commits 07af885b6, f722a7f2a, 767f5b97e, 8e64856a3, ddb35b87d + docs. New G3 tests agc_metal_msaa_{emulated,sampled8,interp,a2c,storage}.
- Verification (--validation): gpu-suite/msaa2-g347: G3 44/44, G4 38/39 + 1 allowlisted skip, G7 55 + 13 allowlisted skips. CPU ctest -LE gpu 5,015/5,015. check_test_registration clean. Disk 8.94 GB.
- Next: the f16 supervised run, merge mac-native-metal into mac-native, M3.


## 2026-10-13 ~20:00 — M3: DOOM (PPSA99666) meets every M3 exit criterion (TITLEQA)
- Boots to E1M1. Frames correct (58 soak frames, 12 windowed). Scripted input works (886 pad commands). Audio non-silent in all 20 × 30 s windows. Headless soak 11.5 min (690 s cap), RSS 95–100 MiB (+0.13 MiB/min), about 47% of one core. Native proof: P_TRANSLATED clear, ARM64 code type, no Rosetta runtime mapped. Windowed 60 s run passes after the main-thread fix (af22a8284: on macOS the guest runs on a secondary thread via Aps5StartGuest). Evidence in .mac-work/logs/m3-qa/; COMPATIBILITY.md row: Playable.
- Tools: tools/mac/title_run.py (4a8755987), Mach-O import_audit (03136f2a9).
- LIBS: STK goes from nothing to loads (4 modules a82c1c808, scePthreadAttrSetprio d11764f8b, console syscall gadget 6d087bca5, FEX disk-cache fix e7185c008). Blocker: fixed mapping at 0x4000000000 falls inside macOS 27's 385 GiB guard region. Agent VA is investigating; this is foundational for retail titles.
- UPSYNC (merge of upstream/main, 937 commits) in progress in .mac-work/wt/sync.
- Open: windowed audio not confirmed by ear; headless audio capture 578 s vs 680 s run (not investigated); the f16 supervised test still needs the user (auto-mode classifier denied main).
- Origin: public fork NaitikJoshiPro/AnyPS5. All our commits rewritten to the NaitikJoshiPro identity before the first push.
