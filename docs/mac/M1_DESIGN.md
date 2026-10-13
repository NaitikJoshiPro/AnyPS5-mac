# M1 design: CPU translator work packages

Scope: MISSION §8 M1 on top of decisions D6 (FEXCore engine, our own AOT discovery and per-title cache), D7 (FEX's ASM unit tests as the oracle) and D10 (x86 ordering through LRCPC, no hardware TSO). Everything runs in `aps5-fex` (`core/fex/`), which is built only on Apple arm64 when `3rdparty/FEX` is checked out (`CMakeLists.txt:179`).

Path prefixes: R = `core/fex/src`, T = `core/fex/tests`, L = `core/libs/prx`, FC = `3rdparty/FEX/FEXCore`, `FEX-2610:` = FEX-Emu tag FEX-2610 (`14c92681f`). Line numbers are at `mac-native` `f7948fbe` (pin `cad0c096`).

## State at the start of M1

- `3rdparty/FEX` is pinned at arvindfroi/FEX `cad0c096` (FEX-2607 + 55 willfaust commits ≤ `ac555dd8` + 15 FEX-Emu backports + 6 arvindfroi commits) with no tracked branch. Its `unittests/`, `Scripts/`, `Source/Tools/` and `.github/` are identical to FEX-2607.
- Nested submodules are shallow and non-recursive:
  - fmt, xxhash and unordered_dense are linked by FEXCore.
  - range-v3 is required because FEX's top-level `CMakeLists.txt:444-447` adds it when the system has none.
  - range-v3's `doc/gh-pages` stays uninitialised.
- Backup: `.mac-work/ref/fex-cad0c096.bundle` holds the complete history of the pin plus tag FEX-2607.
- ctest `fex_licence` (`tools/mac/check_fex_licence.py`) guards the pin and the tree. `fex_licence_checker` tests the guard.
- `tools/unself.py` unwraps fSELF containers. DOOM (`PPSA99666`, see "DOOM" below) is unwrapped and relinked for `aps5-fex`.
- Tools: `ld.lld` 23.1.3 and `nasm` 3.02 (Homebrew).

## Exit criteria → work packages

| M1 exit criterion (MISSION §8) | Work packages |
|---|---|
| Per-instruction semantic tests against the oracle | A |
| Guest fixtures run translated with correct output (imports, TLS, threads, C++ exceptions, C cleanup, variadic printf, callbacks, `long double`) | B, C, D, E, G |
| Discovery report per binary (% of bytes AOT vs JIT); JIT results persisted and reused | R, F |
| Microbenchmarks vs native-compiled C, ≥0.7× native on integer/SIMD loops | H |

## Code cache: verdict

The pin has an old, unfinished cache (FXCC). FEX-2609 and later have a working one (DiskCache). Evidence:

**FXCC at the pin: inside FEXCore, but WIP and unusable on macOS.**
- Built from FC/Source/CMakeLists.txt:21 (`CodeCache.cpp`); interface in FC/include/FEXCore/Core/CodeCache.h:228-288.
- Off by default: `EnableCodeCachingWIP` and `EnableLazyCodeCachingWIP` are `false` (FC/Source/Interface/Config/Config.json.in:19-28).
- Every FEX rebuild invalidates it: the header carries the git hash, checked at CodeCache.cpp:574. The guest bytes are never checked.
- `FileId` is XXH3 of the path (:265-272).
- It is relocatable through `FileStartVA` (:328, :336, :887-893).
- A separate generation run (`IsGeneratingCache`) has to produce it. That run is fed by code maps through the Linux-only FEXServer and FEXOfflineCompiler (`Source/Tools/FEXOfflineCompiler/Main.cpp:389-541`), which `aps5-fex` does not build.
- The macOS load path has never worked. `LoadCache` reserves `PROT_NONE` memory (CodeCache.cpp:607-612) that nothing makes executable, and willfaust's comment at :840-843 says it "isn't used on the iOS host".

**The `.eh_frame`/symbol AOT generator is dead code.** It lives in `Source/Tools/FEXInterpreter/AOT/AOTGenerator.cpp:20-174`, nothing calls it, it persists nothing, and it is Linux frontend code.

**The runner opts into none of this.** `LookupExecutableFileSection` returns `std::nullopt` (R/GuestCpu.cpp:149-151), and the context setup sets no cache option (:386-404).

**DiskCache in FEX-2610 lives entirely in FEXCore.**
- First commit is `561c32b45`, 2026-08-09, which is not an ancestor of the pin. Source is `FEX-2610:FEXCore/Source/Interface/Core/DiskCache.cpp`.
- It stores ordinary JIT output after each compile (`FEX-2610:FEXCore/Source/Interface/Core/Core.cpp:993-1002`). A hit is copied into the code buffer (`JIT/JIT.cpp:1202-1215`).
- Every hit is checked against an XXH128 of the live guest bytes. Entries are keyed by `RIP − FileStartVA` and FileId, or by a hash of the code bytes for anonymous code. The key has no git hash.
- Relocations are applied at the live RIP.
- Concurrency and size: `flock`, a 1 GiB cap (`DiskCacheMaxFileSize`, Config.json.in:225-227), and pruning of stale buckets.
- Off by default (`DiskCache`, Config.json.in:161-163).

**What an embedder must provide:**
- `DiskCache=1` and a cache directory before `CreateNewContext`.
- `QueryGuestExecutableRange`.
- Optionally, `LookupExecutableFileSection` with a FileId that is stable across runs.
- A redirect of the copy at JIT.cpp:1208 through the dual-map write alias. FEX-2610 has no `__APPLE__`/`DualMap` code.

**Verdicts on the three recon claims:**
1. "File cache + `.eh_frame` AOT + offline compiler at the pin" is partly true. They exist, but the cache is WIP and broken on macOS, the AOT generator is dead code, and the offline compiler is Linux-only.
2. "Relocatable cache is WIP/off, its generator is in the Linux frontend" is true for FXCC.
3. "The pin lacks the FEX-2609+ disk cache" is true.

**Cost on each base:**
- On the pin, persisting JIT output means about 400-600 runner lines plus 60-100 FEXCore lines. It needs a generation step, a repaired macOS load path, and it still has no content check and is invalidated by every rebuild.
- On FEX-2610 it is 50-150 runner lines plus a one-line FEXCore fix, after the rebase (WP R).

**Recommendation:** rebase onto FEX-2610 (WP R, about 3 agent-days) before wiring the cache (WP F). Do not build on FXCC.

## Work packages

Sizes are new or changed lines (LOC); agent-days (d) assume one agent. "Overlap" names the other packages that edit the same files.

### R. Rebase the macOS FEX delta onto FEX-2610
**Trial results** (scratch clone, deleted):
- The 15 backports are already in FEX-2610, which leaves 61 commits to replay.
- 14 stops, 27 conflicted file instances and 45 hunks. 30 hunks are in code we build; 15 are in `Source/Windows`, which we don't build.
- Resolving mechanically by taking the replayed side does not compile. It brings back the old `CreateThread`, `DetermineVASize` and the old `CodeBufferManager`, so every hunk needs a hand merge.

**Hot spots:**
- CPUBackend.cpp/.h: willfaust `fce78cefd` and `156653c06` against upstream `d2c92808f` "Split out CodeBuffer management to its own file" and `6734c9ed3`.
- Core.cpp: 6 hunks.
- JIT.cpp: 5 hunks (`dfcbdac34`).
- Frontend.cpp: DiskCache relocation patching.
- PassManager.cpp: willfaust `d6d49752e` against `2893d2b64`.
- Drop arvind's `e941d55cf`: upstream `91bbce5a9` already ends a block at `syscall` on every host.

**Runner changes:**

| Site | Change |
|---|---|
| R/GuestCpu.cpp:79-80 | `DCacheLineSize`/`ICacheLineSize` become `DCacheLineLog2` |
| :137 | `OSABI` member removed |
| :140-142 | becomes `void HandleSyscall(CpuStateFrame*)`; it reads `gregs[REG_RAX/RDI]` and writes RAX |
| :214 | `CreateThread()`, then set `rip` and RSP |

All other calls the runner uses are unchanged.

**Approach:**
- Port the series as a slim macOS series. Drop the commits that are only `Source/Windows` or `FEX_IOS_HOST`, and keep authorship. This is about 1 day less than a full replay.
- Show with `git range-diff` that the kept commits match the audited ones.
- The 756 new FEX-Emu commits (463 non-merge, 23 authors) are MIT, with no Will Faust commits and no GPL-3.0 markers. Re-audit, then add the new SHA to `AUDITED` in `tools/mac/check_fex_licence.py`.

**User decision: where the rebased FEX lives.**
- Option 1: a fork under the user's account. Pushing needs approval.
- Option 2: point the submodule at FEX-Emu/FEX `FEX-2610` and carry the macOS series as patch files applied when the external project is configured.
- Until the user decides, keep the branch local, with a git bundle in `.mac-work/ref/`.

**Package summary:**
- **Files:** `3rdparty/FEX` (new commit), `.gitmodules` (option 2), R/GuestCpu.cpp (the four sites above), `tools/mac/check_fex_licence.py`.
- **Tests:** `fex_fixtures` (21 programs), `fex_bridge_tables`, `fex_licence`, and the A suite pass list equal to the pin baseline.
- **Size:** about 3 d (2-4.5): the rebase 1-2 d (the code-buffer cluster onto `SharedCodeBufferManager` is 4-8 h), the runner 1-2 h plus 2-4 h of contingency (the syscall handler now runs from the dispatcher), and verification.
- **Risks:**
  - The dual-map W^X pool and code-buffer lifetime can only be checked at run time.
  - FEX-2610 has never been built on Darwin.
  - Keep `DiskCache` off until F.
- **Dependencies:** an A baseline on the pin, recommended first. **Overlap:** B, C, F, H (GuestCpu.cpp).

### A. FEX ASM unit tests as the per-instruction oracle
**Inventory:**
- Source: `3rdparty/FEX/unittests/ASM`, 2,222 tests. FEX-2610 has 2,270 (+48).
- Largest directories: VEX 478, TwoByte 221, SSELanePreservation 219, X87 214, Primary 201, PrimaryGroup 152, X87_F64 150, OpSize 110, FEX_bugs 107, H0F38 86, Secondary 80, H0F3A 45, REP 40, 3DNow 27, REPNE 27.
- Skip `32Bit_ASM` (245 tests, 32-bit mode).

**Expectation header:** a block `%ifdef CONFIG {json} %endif` (`Scripts/json_asm_config_parse.py:12-18`). Keys and the number of tests that use them:

| Key | Tests | Content |
|---|---|---|
| `RegData` | 2,198 | `RAX`..`R15`, `RIP` as hex strings; `XMMn` as 2 qwords, or 4 for YMM width; `MMn` as `[mantissa, exp16]`; `FS`/`GS` |
| `HostFeatures` | 916 | e.g. AVX 694, SSE4.1 65, BMI1/2 10 each |
| `Env` | 155 | e.g. `FEX_X87REDUCEDPRECISION=1` ×151 |
| `MemoryRegions` | 45 | `{"0xADDR": "size"}` |
| `Match` | 19 | always `"All"` |
| `MemoryData` | 4 | `{"0xADDR": "AA BB …"}` |
| `Ignore` | 3 | |

- There are no RFLAGS or MXCSR fields: tests copy flags into GPRs.
- Every test ends in `hlt`. The 24 tests without `RegData` pass by reaching `hlt` without a fault.

**Oracle:** `.github/workflows/hostrunner.yml` runs the same `.bin` files natively on a self-hosted x86-64 machine (Zen+) and checks the same `RegData` (`TestHarnessRunner.cpp:269-271, :397, :405`). About 45-50 tests are not hardware-checked: disabled tests, host known failures, and 3DNow/EMMI.

**Harness behaviour to reimplement** (FEX's is MIT; ours is our own code, and copied code keeps FEX's notice):
1. Prepend `BITS 64`, append `ret`, run `nasm -i unittests/ASM/Includes/ X.asm -o X.bin` (flat binary, origin 0; `unittests/ASM/CMakeLists.txt:30-40`). Do this in Python: FEX uses GNU `sed -i`.
2. Map scratch at `0xe000'0000` (10 pages), the SIB8 window at `0xe800'0000` ± 1 page, code at `0x1'0000`, then `MemoryRegions` (`Source/Tools/CommonTools/HarnessHelpers.h:465-512`).
3. Write `MemoryData`. Apply `Env` as a FEX config layer (`TestHarnessRunner.cpp:225`).
4. If a host feature is missing, exit 125 (skip). FEX silently returns pass (:264-266).
5. Start the thread at the code base with RSP = 0 and run to `hlt`. A fault anywhere other than `hlt` fails the test (:124-130).
6. Compare `RegData` with the thread state; XMM/YMM come from the JIT state (`HarnessHelpers.h:216-266`; TestHarnessRunner.cpp:363-375).
7. Lists: `Known_Failures*` invert the result and `Disabled_Tests*` skip. The `XeSS_quadratic` entry lacks the `Test_` prefix, so it is never disabled; disable it or give it a long label.

**Addresses below 4 GiB, which macOS arm64 cannot map (`__PAGEZERO`):**
- 1,020 tests use low addresses beyond the code placement: 1,009 load a `0xE…` literal into a 64-bit register, 1 into a 32-bit register, and 2 hold one in a `dq`.
- 10 tests declare regions below 4 GiB, and 38 declare regions at `0x1_0000_0000`-`0x5_0000_0000`, where the macOS executable may sit.
- **Relocation:** move everything by Δ = `0x10_0000_0000`, a multiple of 4 GiB, so the low 32 bits are kept:
  - Reserve the window with `mach_vm_allocate(VM_FLAGS_FIXED)`; it fails rather than overwrites.
  - At build time, a rewrite adds Δ to address literals and `dq` values, widens `mov r32, ADDR`, and moves the code base, the `MemoryRegions`/`MemoryData` keys and the 107 GPR expectations that point into mapped ranges.
  - Round regions to 16 KiB: 16 bases and 43 sizes are not multiples.
  - Take the `cmpxchg8b`/`cmpxchg16b` code-address expectations from a nasm map file.
- **13 tests cannot be relocated mechanically.** They go to a `Disabled_Tests_macos` list with reasons:
  - 5 use 32-bit address size: `a32_lods`, `a32_rep_lods`, the two `call [esp]` tests (`5_FF_05_03_o32*`), and `vmovq_vmovd_reg`.
  - 6 use a raw `db 0x67`: `Primary_A0`, `A2`, `A6_addrmod`, `A6_REP_addrmod`, `AE_addrmod`, `mov_address_size_override`.
  - 2 use an absolute disp32: `Displacement_Encoding`, `MemoryData`.
- Runnable: 2,209/2,222 (99.4%). The classifier and the scripts are in `.mac-work/research/fex-asm-tools/`; the per-test table is in `.mac-work/logs/m1prep-asm-lowaddr.tsv`.

**nasm 3.02:** all 2,222 tests at the pin and all 2,270 at FEX-2610 assemble. The one warning (`FEX_bugs/o16_leave.asm:20`) still produces the intended `66 C9`.

**Package summary:**
- **Files:**
  - `T/asm/AsmHarness.cpp`, a harness executable that shares the runner's FEX setup through a static library made from R/GuestCpu.cpp.
  - `T/asm/prepare_asm_tests.py` (relocation, config JSON, nasm).
  - `T/asm/Disabled_Tests_macos`.
  - `core/fex/CMakeLists.txt`: `find_program(nasm)`; without nasm the tests are not registered, the same rule as Python.
- **Registration:** one ctest per file, `fex_asm/<dir>/<file>`, default config (`MAXINST=500`, multiblock), `TIMEOUT 30`, `SKIP_RETURN_CODE 125`, label `fex_asm`. The `MAXINST=1` config goes behind label `fex_asm_full`. A per-directory test would break the 30 s rule (VEX alone is 478 processes).
- **Tests:** the suite itself, plus `T/test_prepare_asm_tests.py` for the rewrite on synthetic inputs.
- **Size:** harness 500-700 LOC, prepare 400 LOC, tests 200, CMake 60; 2-3 d.
- **Dependencies:** none. Run it on the pin first as the R baseline. **Overlap:** R (GuestCpu.cpp API), `core/fex/CMakeLists.txt` with C, D and H.
- **Later (M4):** `unittests/InstructionCountCI` (101 files, 7,677 entries) is a code-size regression oracle. Its expected disassembly needs a Darwin baseline because the pin doesn't use x18 on Apple.

**Results (2026-10-12, pin `cad0c096`, commits `ace93c10`, `0dade124`, `161f7c8a`):**
- **Built as designed, with these corrections:**
  - Δ is `0x100_0000_0000`, not `0x10_0000_0000`: macOS 27 reserves a 384 GiB guard region at `0x10_0000_0000`-`0x70_0000_0000` (`vmmap`: "Guard"), the GPU carveout at `0x73_ec00_0000`-`0x79_ec00_0000` and malloc regions in between; `mach_vm_allocate(FIXED)` there returns `KERN_NO_SPACE` even in an empty process. A small `__PAGEZERO` (`-pagezero_size`) is no way out: the kernel kills such an arm64 binary at exec (exit 137).
  - The shared setup is `RunBareGuest` in R/GuestCpu.cpp (declared in R/BareGuest.hpp): `PrepareFex` (log handlers, pool, allocator hooks, config, `APS5_FEX_CONFIG`) is shared with `GuestCpu`'s constructor; a bare `SyscallHandler` and `GuestThread` without TLS or bridge. `aps5-fex` is `Main.cpp` over the static library `aps5-fex-runtime`, which the harness links too.
  - The harness forks per test so crashes count as failures and known failures invert them; it captures the child's output and prints the verdict first (CTest truncates passed-test output).
  - Relocation is a source-order dataflow over `nasm -E` output: movs whose constant becomes a pointer, followed through copies, arithmetic, push/pop, `cmp`/`sub` with another moved pointer, and unscaled bases whose sum is an address (`SIBScaleTranspose`). 1,024 tests are rewritten. Expectations accept `V + Δ` within 1 MiB of a mapped range and the moved code address through nasm listings (`cmpxchg8b`, `cmpxchg16b`).
  - Tests that cannot be moved: the 13 predicted, plus the three `5_FF_05*` far jumps through `lea edi, [rel label]` (a 32-bit code address).
- **Suite (`ctest -L fex_asm`, both variants, 4,444 tests + prepare, about 15 s on 8 cores after a 70-100 s first prepare):** default (MAXINST=500, multiblock, TSO on as in the runner) 2,121 pass, 73 known failures, 28 skips, 0 unexpected; full (MAXINST=1) 2,122 / 72 / 28 / 0. Baseline: `T/asm/baseline-cad0c096.txt`; `T/asm/asm_suite.py compare <junit> <baseline>` checks a rerun.
- **Skips (28):** 18 `Disabled_Tests_macos` (16 unmovable, `TwoByte/0F_02.asm` needs the Linux `time` syscall, `Secondary/07_XX_04.asm` with nasm 3), 5 FEX `Disabled_Tests`, 3 CLZERO (the runner's host features leave it out), 2 RDRAND/RDSEED (no FEAT_RNG on M3).
- **Known failures (73): FEX's own `FEX_bugs/32bit_syscall.asm` (`Known_Failures_jit`) and 72 runner gaps, none in FEXCore's x86 semantics:**
  - 69 SIGBUS (`BUS_ADRALN`) on atomics or TSO-ordered accesses that cross a 16-byte granule (fault offsets 9-15). The plain ones pass with `FEX_TSOENABLED=0`; lock-prefixed ones fault either way. Package B's handler should turn them into passes (XPASS then asks to delete the list entries).
  - `FEX_bugs/CodeBufferOverflow.asm` (default variant only): FEXCore restarts a block that outgrows its temporary JIT buffer from the frontend's SIGSEGV handler on `Thread->JITGuardPage` (FEX `Source/Tools/LinuxEmulation/LinuxSyscalls/SignalDelegator.cpp:585`, `UncheckedLongJump::ManuallyLoadJumpBuf(Thread->RestartJump, ...)`); `OnFault` does not.
  - `Primary/Primary_8C.asm`, `Primary_8C_2.asm`: `mov es, ax` with 0x4142 makes FEXCore read GDT entry 0x828; `GuestThread::gdt` has 32 entries on the stack.
- **Found and fixed in the bare guest only** (the runner still lacks them, TechnicalDebt): the pinned FEX ends a block at `syscall` and resumes at the stored RIP, so a handler must advance RIP by 2 (`Bridge::Call` does); FEXCore leaves `InvalidateGuestCodeRange` to the frontend, so with `SMCChecks=full` a changed block loops forever and with the default MTRACK self-modifying code is never seen.
- **nasm 3.02 vs 2.16.03** (built from source in `.mac-work/tools/`, sha256 from Homebrew's 2.16.03 formula): 26 tests assemble to different bytes (prefix order such as `66 F3 A5` for `F3 66 A5`, `cdqe` for `movsxd rax, eax`, dropped no-op REX.W on `pinsrw`/`pextrw`/`crc32`, swapped `xchg` operands). Only `smsw r64/r16` changes meaning (nasm 3 drops REX.W and 0x66). With nasm 2.16.03 the default variant gives 2,122 pass, 73 known failures, 27 skips. `-DAPS5_FEX_ASM_NASM=<nasm>` selects the assembler; the baseline uses nasm 3.02.
- **Coverage, default variant** (`asm_suite.py classes`; a test counts for every class among its instructions other than mov/lea/push/pop/xor/jmp/call/ret/nop/hlt):

| Class | Tests | Pass | Known failures | Skips | Pass rate of run tests |
|---|---|---|---|---|---|
| integer | 831 | 762 | 65 | 4 | 92.1% |
| flags (pushf/lahf/setcc/cmovcc/jcc/adc/sbb/rcl/rcr readers) | 417 | 400 | 15 | 2 | 96.4% |
| string ops | 99 | 98 | 1 | 0 | 99.0% |
| atomics (lock, cmpxchg8b/16b) | 90 | 45 | 45 | 0 | 50.0% |
| x87 | 388 | 381 | 7 | 0 | 98.2% |
| MMX / 3DNow! | 145 / 28 | 144 / 28 | 1 / 0 | 0 | 99.3% / 100% |
| SSE / SSE2 / SSE3 / SSSE3 | 312 / 368 / 37 / 54 | 308 / 364 / 37 / 54 | 4 / 4 / 0 / 0 | 0 | 98.7% / 98.9% / 100% / 100% |
| SSE4.1 / SSE4.2 / SSE4a | 105 / 37 / 12 | all pass | 0 | 0 | 100% |
| AES / PCLMUL / SHA | 31 | 31 | 0 | 0 | 100% |
| AVX / AVX2 | 702 / 258 | all pass | 0 | 0 | 100% |
| FMA / F16C | 21 / 11 | all pass | 0 | 0 | 100% |
| BMI1 / BMI2 / LZCNT, POPCNT | 13 / 11 / 7 | 13 / 11 / 5 | 0 / 0 / 2 | 0 | 100% / 100% / 71.4% |
| MOVBE | 3 | 3 | 0 | 0 | 100% |
| system (cpuid, rdtsc, xsave, fxsave, rdrand, ...) | 86 | 73 | 8 | 5 | 90.1% |
| all | 2,222 | 2,121 | 73 | 28 | 96.7% |

### B. SIGBUS handler for unaligned ordered accesses
**Problem:** the M3 raises SIGBUS (BUS_ADRALN) when an LDAPR/LDAPUR/STLR/STLUR or atomic crosses a 16-byte boundary. `OnFault` (R/GuestCpu.cpp:353-371) handles only native→guest calls (:356-359) and passes everything else on, so the process dies.

**Fix** (FEX Linux does the same at `Source/Tools/LinuxEmulation/LinuxSyscalls/SignalDelegator.cpp:890-894, :966-985`):
- If `si_code == BUS_ADRALN` and the PC is in the code buffer (`IsAddressInCodeBuffer`, FC/include/FEXCore/Core/Context.h:160), call `FEXCore::ArchHelpers::Arm64::HandleUnalignedAccess(thread, type, pc, &uc_mcontext->__ss.__x[0])` (FC/include/FEXCore/Utils/ArchHelpers/Arm64.h:13-18, :35-36), then advance the PC by the result.
- The handler type is `HalfBarrier` if `HalfBarrierTSOEnabled`, else `NonAtomic`.
- On Darwin, `__ss.__x[0..28], __fp, __lr, __sp, __pc` are contiguous.
- FEX backpatches the instruction by storing to PC[0]/PC[1] (FC/Source/Utils/ArchHelpers/Arm64.cpp:2382-2440), but our code view is RX (R/GuestCpu.cpp:49-51). Either hand FEX the RW alias (`pc + DualMap::WriteOffset`) or patch Arm64.cpp to write through `DualMap::WriteAddr`, as block linking does (JIT.cpp:536-542).

**Package summary:**
- **Files:** R/GuestCpu.cpp (`OnFault`), possibly FC/Source/Utils/ArchHelpers/Arm64.cpp.
- **Tests:** fixture `T/unaligned.cpp`, added to `COMPILED` in T/run_fex_fixtures.py:101-130 and run by ctest `fex_fixtures`. It does plain and `lock xadd`/`cmpxchg` accesses of 2, 4 and 8 bytes at offsets 12, 14 and 15 within a 16-byte line, twice (the second pass runs the patched code), on 2 threads.
- **Size:** 40 + 90 LOC; 0.5 d.
- **Dependencies:** after R; the `NonAtomic` variant needs H's config hook. **Overlap:** C, F, H (different functions of GuestCpu.cpp).

### C. JIT code pool reclamation
**Problem:**
- The 1 GiB dual-mapped pool (R/GuestCpu.cpp:38-55) is a bump allocator (`PoolMmap`, :57-63). `PoolMunmap` ignores pool ranges (:65-69).
- FEX allocates code buffers with `Allocator::VirtualAlloc` (FC/Source/Interface/Core/CPUBackend.cpp:457-459) and frees them with `VirtualFree` (:499-501); both reach the runner's mmap/munmap hooks. Buffers start at 16 MiB, double up to 128 MiB, and are replaced on `ClearCodeCache`.
- After about 16+32+64+128 MiB plus 6 more 128 MiB buffers, `PoolMmap` returns `MAP_FAILED`. That value passes FEX's `!!Ptr` check (CPUBackend.cpp:481), and FEX writes through a wild pointer.

**Fix:**
- New R/CodePool.{hpp,cpp}: a free list at 16 KiB granularity with coalescing, keeping the single `WriteOffset`.
- Freeing a range does `madvise(MADV_FREE_REUSABLE)` on the RW alias.
- Exhaustion throws `std::runtime_error`.
- In FEX-2610 this goes through `SharedCodeBufferManager`, so C comes after R.

**Package summary:**
- **Files:** R/CodePool.{hpp,cpp}, R/GuestCpu.cpp (`Pool*`, hooks at :389-391), `core/fex/CMakeLists.txt`.
- **Tests:**
  - Unit test `T/CodePool.cpp` (`add_test_executable` plus `add_test(fex_code_pool)`, as `tools/check_test_registration.py` requires).
  - Fixture `code-churn`: with `APS5_FEX_CLEAR_CODE=1`, the runner calls `ClearCodeCache` at guest depth 0 (R/GuestCpu.cpp:316), and the guest creates and joins 64 threads. It fails today and passes after the fix.
- **Size:** 150 + 150 LOC; 1 d.
- **Dependencies:** R. **Overlap:** B, F, H.

### D. Bridge ABI from build-time signatures, and direct reverse thunks
**Today's forward path:**
- Import stub (R/Bridge.cpp:298-311) → spill rdi..r9 and xmm0-7, then `syscall` (:184-212) → `Bridge::Call` (:354-421).
- Arguments are mapped by position: rdi..r9→x0-5, two stack slots→x6/x7, xmm0-7→v0-7 (:378-384), and 8 more 8-byte stack slots (Trampolines.S:19-31).
- The only per-function knowledge is `VariadicExports` (:148-166) and `X87Results` (:168).

**Three mappings are wrong** (checked with `clang -S`):

| Case | Apple arm64 | x86-64 SysV | Example |
|---|---|---|---|
| Stack arguments | packed at natural size | 8-byte slots | `sceAgcCbReleaseMem` (12 arguments, the last four u16,u16,u8,u32; L/libSceAgc/Cb/src/Sync.cpp:25) |
| Struct of floats by value | `{float x,y,z}` in s0-s2 | xmm0 holds x,y; xmm1 holds z | `sceAudioOut2GetSpeakerArrayCoefficients` (L/libSceAudioOut/src/AudioOut2SpeakerArray.cpp:102) |
| Hidden-pointer struct return | x8 | rdi, with rax = rdi on return | `_ZSt17current_exceptionv` (L/libc/src/exception/ExceptionPointer.cpp:133) |

For stack arguments: 14 exports take more than 8 integer-class arguments (`.mac-work/logs/m1prep-abi-scan.log`, 3,559 exports scanned). 11 of them put narrow types on the stack, including `sceAgcCbReleaseMem`, `sceAgcDcbDmaData` and `sceAgcCbBranch`. D1 found that positional mapping is wrong for 4 of them (see "D1 result").

**Signature table:**
- No file declares the export signatures; the NID patcher renames exports, so the bridge resolves them by `dladdr` name (R/Bridge.cpp:329-341).
- A generator runs `clang++ -fsyntax-only -Xclang -ast-dump=json` over the libs sources, using `compile_commands.json` from `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.
- It classifies each export under both ABIs and writes only the shapes that positional mapping gets wrong into a checked-in R/BridgeSignatures.inc. (D1 instead generates every export's shape into the build tree with libclang; see "D1 result".)
- `Resolve` attaches the shape. `Call` then sets x8 to the guest's rdi buffer (and rax = rdi), splits SSE eightbytes into s/d registers, and repacks the stack at Apple offsets.

**Reverse direction:**
- **Today:** a native call to a guest address faults and is redirected (R/GuestCpu.cpp:356-359 → Trampolines.S:52-77 → `CallGuest` :297-326), which costs microseconds per call.
- **Fault-path users:**
  - thread entry (L/libkernel/Pthread/src/Thread.cpp:426)
  - cleanup handlers (:768)
  - TSD destructors (Tsd.cpp:57)
  - `atexit` (L/libc/src/Process.cpp:37-38)
  - guest personality routines (L/libc/src/exception/Unwind.cpp:43-46)
- **Direct-path users:** only qsort/bsearch (L/libc/src/FileAndHeap.cpp:376, :401-402) and the application heap, through `CallProgram`, which takes at most 6 integer arguments (L/libc/include/ProgramCall.hpp:19-33).
- **Fix:**
  - A static table of N arm64 thunks in Trampolines.S (`adrp/ldr x16, slot[k]; b _Aps5GuestCallEntry`), so there is no runtime code generation and no W^X issue.
  - Bridge keeps guest↔thunk maps, and `Call` swaps function-pointer arguments for thunks.
  - `CallGuest` takes a per-thunk shape for more than 6 arguments, floats and hidden-pointer returns.

**Package summary:**
- **Files:** R/Bridge.{cpp,hpp}, R/Trampolines.S, R/GuestCpu.cpp (`CallGuest`), new generator `core/fex/tools/gen_bridge_signatures.py`, R/BridgeSignatures.inc, `core/fex/CMakeLists.txt`.
- **Tests:**
  - ctest `fex_bridge_signatures`: the generator in `--check` mode fails when the libs change without regenerating.
  - Fixture `T/abi.cpp` plus a test-only NID-patched library `T/abi_exports.cpp` that echoes every shape and calls back. It is staged into a per-test `libs` directory; run_fex_fixtures.py:153 symlinks the build's `libs` today.
  - A byte check of the `sceAgcCbReleaseMem` packet.
- **Size:** generator 350, Bridge 250, thunks 150, tests 300 LOC; 3 d. Split into D1 (generator and table) and D2 (marshalling and thunks).
- **Dependencies:** D2 after D1 and after E's `StubFor`. **Overlap:** E, G (Bridge.cpp), C, F (`CallGuest` is close to the pool code).

**D1 result (2026-10-12):**
- **Where the table lives:** generated into the build tree at build time (`build-arm64/core/fex/signatures/BridgeSignatures.{inc,json}`), not checked in. Every derived file in this repository is a build product (SPIR-V headers, the shader-cache version header, the NID-patched prx), and none is checked in. A table kept beside its sources can lag a library change, and a stale table hands a function wrong arguments with no error. The command depends on every prx library target, so a rebuilt library regenerates it.
- **Generator** (`core/fex/tools/gen_bridge_signatures.py`): `-ast-dump=json` was dropped: one unit (libSceAgc `Sync.cpp`) dumps 669 MB in 25 s. libclang through ctypes, with skipped function bodies, takes about 1 s of CPU per unit. A definition is told from a declaration by the text after its declarator; `--full-parse` asks libclang instead and finds the same exports. Per-unit cache keyed by arguments, libclang, the parser source and the mtimes of every included repository file: 439 units, about 430 s of CPU cold (2:28 to 4:43 at load averages 35-98), 0.8 s warm.
- **Exports:** extern "C" definitions with external linkage and default visibility in the units of prx library targets (`-D<target>_EXPORTS`, not in `tests/`): 3,541, plus 76 host-only `*_nid_no_patch`. One record per export: kinds (signed/unsigned with size, `__int128`, pointer, function pointer with the callback's own signature, float, double, `long double`, GuestLongDouble, vector, empty, struct with size, alignment, flattened leaves and whether C++ passes it by hidden reference), variadic and fixed count, the SysV and Apple arm64 locations, and the reasons positional mapping is wrong (`BridgeReason` in R/BridgeSignatures.hpp).
- **Needs more than positional marshalling (53):** 22 variadic (handled by VariadicExports, now 22 entries: `makecontext`, `openat`, `sceKernelFcntl`, `swscanf_s` and `syslog` were missing and read their variable arguments from the native stack); `strtold`/`wcstold` (X87Results); 16 `long double` math exports (`acosl`…`truncl`, `frexpl`) and `__signbitl` (x87 argument); 4 of the 14 exports with more than 8 arguments (`sceAgcCbReleaseMem`, `sceAgcCbBranch`, `sceAgcDcbDmaData`, `sceAgcAcbDmaData`: narrow arguments after the ninth); `sceAudioOut2GetSpeakerArrayCoefficients` (HFA, then every later float shifts); 7 hidden-pointer results (`_ZSt17current_exceptionv`, plus 6 host-side `*_nid_postfix` helpers). The other 10 exports with more than 8 arguments are positional: their stack arguments land at the offsets the bridge copies to. 83 function-pointer arguments in 72 exports, all callbacks positional; 5 exports return function pointers.
- **Compiler check:** `verify` compiles every distinct export and callback shape (563) for `x86_64-sie-ps5` and `arm64-apple-macos14` and compares the generator's locations with the MIR after instruction selection (`liveins`, fixed stack objects, `RET`): no difference. With `--compile-commands` it compiles the 12 library units whose exports pass structs, GuestLongDouble or `__int128` by value and checks those 12 exports on arm64: no difference. This caught `std::shared_ptr`, `std::vector` and `std::stop_token` results classified as passed in registers; a template instantiation's special members are now read from its template.
- **Bridge:** R/BridgeSignatures.{hpp,cpp} hold the table types and lookup. With `-DAPS5_FEX_BRIDGE_SIGNATURES=ON` (off by default) the table is compiled into the runner and `StubFor` attaches `Function::signature`; marshalling is unchanged (D2).
- **Tests** (replace `fex_bridge_tables`): `fex_bridge_signatures` (no unclassified export, no conflicting definitions, VariadicExports and X87Results equal to the generated data, prints the list above), `fex_bridge_signatures_table` (the C++ table and lookups), `fex_bridge_signatures_abi` (`verify`, 1 s), `fex_bridge_signatures_units` (`verify` with library units, `slow`), `fex_bridge_signatures_generator` (11 unit tests: the models, a generated tree, cache, `check`, `verify`, the `.inc` compiling). `fex_fixtures` passes with the flag off and on.

**D2 result (2026-10-12, `6d04bc6b`, `7dd8018f`):**
- **Marshalling:** the generator also emits per-value layouts for both conventions (`BridgePiece`/`BridgeLayout`: register, vector byte, stack offset, st(0), hidden pointer). Exports with a reason other than `variadic` go through `MarshalArguments`/`UnmarshalResult` (R/Bridge.cpp); the rest stay positional, copying only the stack slots and vector registers their signature uses. `long double` is converted x87 ↔ double at the boundary (R/X87Convert.hpp, round to nearest even; `fex_x87_convert`). The table is on by default.
- **Correction to D1:** Apple's arm64 convention does not start `__int128` or a 16-byte aligned struct at an even register (clang `-target arm64-apple-macos14`: `f(int, __int128, int)` takes x1:x2, w3); `verify` caught the old rule on the test library.
- **Tests:** test-only library `T/prx/libAps5BridgeAbi` (NID-patched, staged per fixture, its exports in the table); fixtures `abi-registers`, `abi-stack` (incl. a `sceAgcCbReleaseMem` packet), `abi-memory`, `abi-long-double` (all 16 math exports, `strtold`, `wcstold`, `__signbitl`), `abi-variadic` (`swprintf` now bridged), `abi-callbacks`.
- **Reverse thunks:** 4,096 static thunks (`mov w17, #k; b _Aps5GuestThunkEntry`), R/GuestCall.cpp. Only function pointers whose callback shape positional mapping gets wrong are swapped: libSceFiber hands a fiber's entry back to the guest as its rip, so swapping every guest pointer broke `fibers`. Thunks coming back as results or resumed rips map to the guest function. Positional callbacks keep the fault path (or `CallProgram`).
- **Bridge cost** (`run_fex_bench.py --variants avx2 --kernels '^bridge_(call|qsort)$'`, CPU time, load 2-4): guest→native was 45 ns at the start of this wave (105 ns was the pin under load 40-99). The trampolines now call the bridge through FEX's thunk instruction (`0f 3f` + 32-byte name, `ThunkHandler`): FEXCore calls it from JIT code with spilled static registers, no dispatcher exit, no block lookups. Same build, `APS5_FEX_BRIDGE_SYSCALL=1` against default: 46-54 → 41-42 ns. native→guest through `qsort`: 66-68 ns, unchanged. `sample` of `bridge_call`: about a third of the samples sit on the two `msr FPCR` of FEXCore's spill/fill around host calls (AFP's AH/NEP/FIZ off for host code, on again), FEXCore code we do not change (D13); the bridge itself is about 15%.

### E. Guest image registration, `sceKernelDlsym`, low-address mappings
**Guest image registration:**
- **Problem:**
  - libc's memory registry (L/libc/include/GuestAllocations.hpp:28-40) only registers the main image outside Windows (GuestAllocations.cpp:238). The Apple branch scans dyld images for `__APS5DATA` (:248-283).
  - So images the runner maps (R/GuestImage.cpp:126-137) are never registered, and `DoMprotect` trims their pages away (L/libkernel/DirectMemory/DirectMemory.cpp:1015, :1053-1054).
  - The `dl_iterate_phdr` list is already fed by the runner (R/Bridge.cpp:279-287).
- **Fix:** the Apple branch also walks the runner-registered list (L/libc/src/specifics/macos/GuestImages.cpp:39-55) through a libc-internal accessor. It maps `PT_LOAD` flags to read/write and marks the ranges not releasable.

**`sceKernelDlsym`:**
- **Problem:** the call chain (L/libkernel/Module/src/Module.cpp:133-141 → `FindSymbol`, DynamicLoader.cpp:104-115) returns host symbols as raw arm64 addresses (:63).
- **Fix:**
  - Add a third callback to `Aps5SetGuestLoader` (:120-123, set at R/GuestProgram.cpp:73-76). It calls a new `Bridge::StubFor(address, name)`, factored out of `Resolve` (R/Bridge.cpp:313-352). Code gets an x86 stub; data stays raw.
  - The same applies to the native defaults returned by `get_terminate` (L/libc/src/exception/Runtime.cpp:349) and `get_new_handler` (StdioExtras.cpp:74).

**Below 4 GiB:**
- Nothing in our code needs low addresses: the relinker emits ET_DYN, the runner maps anywhere, and GuestImage throws on 32-bit relocations (:256).
- The remaining risks are guest `MAP_FIXED` below 4 GiB, which throws "Fixed mmap failed" (DirectMemory.cpp:717-723), and the PS5 arena layout at or above 8 GiB, which is not reserved on macOS (L/libc/src/GuestArena.cpp:40-43, :56-58).
- **Fix:** reserve the arena at startup. Keep the throw, name the 4 GiB `__PAGEZERO` limit in its message, and record it in TechnicalDebt.

**Package summary:**
- **Files:** L/libc/src/GuestAllocations.cpp, L/libc/src/specifics/macos/GuestImages.cpp, L/libkernel/Module/src/DynamicLoader.cpp, L/libc/src/GuestArena.cpp, R/Bridge.cpp (`StubFor`), R/GuestProgram.cpp.
- **Tests:** fixtures `image-protect` (`sceKernelMprotect` and `sceKernelQueryMemoryProtection` on the fixture's own `.data`) and `dlsym-host` (`sceKernelDlsym("strlen")`, then a call to it).
- **Size:** libc 60, libkernel and runner 60, tests 100 LOC; 1 d.
- **Dependencies:** none. **Overlap:** D, G (Bridge.cpp).

**Result (2026-10-12):**
- **Images:** registered eagerly, not at `RegisterMainImage`: `Aps5RegisterGuestImage` calls libc's `GuestAllocations::RegisterRunnerImage` (L/libc/src/GuestAllocations.cpp), so registration does not depend on when the registry first registers the dyld images. Pages are 16 KiB host pages; segments sharing one get the union of their access (the fixtures' 4 KiB-linked ELFs put `.text` and `.data` in one page).
- **Stubs:** one hook in libc instead of a third `Aps5SetGuestLoader` callback, because libc needs it too: the runner sets `Aps5SetBridgeCallable_nid_no_patch` (R/GuestProgram.cpp), and `LibcGuestCallable_nid_no_patch` (L/libc/src/specifics/macos/GuestImages.cpp) serves libkernel's `dlsym` path (DynamicLoader.cpp) and libc's `set_terminate`/`get_terminate` (`std::abort` maps to libc's own `abort` export). `get_new_handler` needs nothing: its default is null.
- **`Bridge::StubFor(address, name)`:** one stub per native address, so `dlsym` and imports agree. Data comes back as its address. 0 for anything outside the PS5 libraries (any `*.prx`, not only the ones the runner opened), so `dlsym(0, "strlen")` skips libSystem's `strlen` and finds libc.prx's by NID. Stubs can now be added while guest threads run: a recursive lock, `functions.reserve(StubCapacity)`, and an acquire-loaded `stubCount` as `Call`'s bound. D2 can use `StubFor` for native function pointers it returns to the guest.
- **Arena:** reserved on macOS as on Linux (`SeedAppleUsedRanges`: `mach_vm_region` over [8 GiB, 0xFC_0000_0000), free gaps reserved in 1 GiB `mach_vm_map(VM_FLAGS_FIXED, VM_PROT_NONE)` chunks, about 2 ms). On this M3 (macOS 27.0.1) the host already holds the shared region (6 to 63 GiB), guard ranges up to 448 GiB and malloc's regions between 448 and 512 GiB; 536 GiB stays for the guest. DirectMemory's Linux arena paths (fixed, hinted, anywhere, unmap) now also run on Apple. A fixed mapping below 4 GiB throws naming `__PAGEZERO`. AMM works: `guest_amm_placement` (which skipped before) and `guest_ampr_markers`'s AMM checks now run on every host with no macOS skip or gate, and pass on arm64.
- **Extra:** on arm64, guest `PROT_CPU_EXEC` no longer makes host pages executable (`LinuxProtFromSce`). x86 bytes must fault into the runner, and macOS refuses W+X pages outside MAP_JIT (EACCES, measured).
- **Tests:**
  - fixtures `image-protect` and `dlsym-host` (T/image_protect.cpp, T/dlsym_host.cpp); `fex_fixtures` passes with 23 programs.
  - `ctest -LE "gpu|slow"`: 458 tests, 2 failures. `fex_bridge_tables` was already failing (D1/D2). `write_watch_threads` timed out at load average 66; it links neither libc nor libkernel.

### F. Code cache, AOT discovery, coverage report
**DiskCache wiring (after R):**
- In R/GuestCpu.cpp:392-397, set `DiskCache=1` and the cache directory before `CreateNewContext`. The directory is `~/Library/Caches/AnyPS5/<title id>/fex`, or a test directory under the build.
- Implement `LookupExecutableFileSection` (R/GuestCpu.cpp:149-151).
  - FileId is XXH3 of the module file. R/GuestImage.cpp:98-100 already has its bytes; FEX's build-id code is Linux-only.
  - `FileStartVA`, `BeginVA` and `EndVA` come from `GuestImage`.
- Narrow `QueryGuestExecutableRange` (:144-147) to the executable segments.
- Redirect the copy at `FEX-2610:JIT.cpp:1208` through the write alias.

**AOT discovery (our code):**
- Seeds:
  - the entry point (R/GuestImage.cpp:145) and exports (:156-165);
  - `DT_INIT` and the init arrays (:185-199);
  - 64-bit and RELATIVE relocation targets in executable segments (:236-261);
  - `.eh_frame` FDEs through `PT_GNU_EH_FRAME`. The relinker keeps it when a program header slot is free (`core/relinker/elfpatcher/src/general/ProgramHeaderLayoutBuilder.cpp:145-152`), and GuestImage keeps the program headers (R/GuestImage.cpp:112-113). Walk `.eh_frame` directly when the `.eh_frame_hdr` table is empty.
- `aps5-fex --precompile <elf>` calls `CompileRIP` (FC/include/FEXCore/Core/Context.h:77) for each seed on worker threads. DiskCache stores each result.

**Measured on the relinked DOOM:**

| Image | `.eh_frame` | Other seeds |
|---|---|---|
| `app.elf` | 1,163 FDEs cover 98.2% of the 0x7a11d-byte text | 173 RELATIVE targets in text, no exports |
| guest `libc.prx` | present but all zero (24,080 bytes, 0 FDEs) | 1,874 exported functions (their `st_size` covers 1.0% of text, so entry points only), 3 RELATIVE targets |

So discovery must combine all the seeds, and libraries without unwind tables depend on exports, block following and the JIT.

**Coverage report:**
- A FEXCore hook records each compiled block's guest byte range and origin: precompiled, cache hit, or runtime JIT. FEX-2610 only counts hits, misses and compiles (`Core.cpp:908, 916, 920`).
- At exit the runner writes `coverage.json` with per-module text bytes, bytes in FDE ranges, AOT bytes, runtime-JIT bytes and cache-hit bytes.
- `tools/mac/fex_coverage.py` prints the report.

**Package summary:**
- **Files:** R/GuestCpu.cpp (constructor, `LookupExecutableFileSection`), R/GuestImage.{cpp,hpp} (identity, seeds, recorded relocations), R/Main.cpp (`--precompile`), FC JIT.cpp and the coverage hook (FEX patch), `tools/mac/fex_coverage.py`.
- **Tests:**
  - A fixture run twice with a fresh cache directory. The second run must compile 0 blocks at run time and print the same output.
  - Seed extraction on a synthetic ELF, including an empty `.eh_frame_hdr` table.
  - The coverage JSON schema.
- **Size:** runner 330, cache wiring 50-150, coverage 100-200, tests 150 LOC; 2-3 d.
- **Dependencies:** R, C. **Overlap:** B, C, H (GuestCpu.cpp).

### G. Guest-state operations wired to the runner
**State on `mac-native` `f8119cd3`:** these throw on arm64.
- `getcontext`/`setcontext` (L/libkernel/System/src/UserContext.cpp:78 `NeedsRunner`, reached at :175 and :215).
- `siglongjmp` when the x87 control word changes (SignalSets.cpp:194).
- `sceKernelRaiseException` (Exception.cpp:311).
- Their tests exit 77, which counts as a skip.
- `makecontext` points `rip` at a host function (L/libSceLibcInternal/UserContext.cpp:63).

**Hooks:**
- **Available today:**
  - `LibcCaptureRegisters` → `CaptureGuestFrame` (R/Bridge.cpp:72-87; 16 GPRs and RIP of the innermost bridge call).
  - `LibcRestoreRegisters` → the catch at :405-410.
  - `LibcGuestControl` (read-only).
- **Missing:** RFLAGS, xmm0-15, x87 state, writable MXCSR/FCW, and the fs base.
- **Fix:** add `L/libc/include/GuestMachineState.hpp` and `Aps5SetBridgeMachineState_nid_no_patch(get, set)`.
  - get: the full CPU state (FC/include/FEXCore/Core/CoreState.h:121, 127, 132, 195), with RFLAGS from `ReconstructCompactedEFLAGS` (FC/include/FEXCore/Core/Context.h:103).
  - set: a full-state frame applied at R/Bridge.cpp:405, using `SetFlagsFromCompactedEFLAGS` (:105).
  - `makecontext` goes through a Bridge stub plus `LibcCallGuest`.
- Asynchronous `sceKernelRaiseException` (`pthread_kill`, rebuild the guest state from the host registers, `CallGuest`) is a separate package, G2.

**Package summary:**
- **Files:** L/libc/src/specifics/x86_64/RegisterContext.cpp, the new header, L/libkernel/System/src/{UserContext,SignalSets,Exception}.cpp, L/libSceLibcInternal/UserContext.cpp, R/Bridge.cpp.
- **Tests:** fixtures `T/context.cpp` (`getcontext`/`setcontext`/`makecontext`/`swapcontext` round trips, including xmm and flags) and `T/raise.cpp`. The libkernel tests that exit 77 run under `aps5-fex`.
- **Size:** 350 LOC; G2 another 300. 1.5 d.
- **Dependencies:** M0's guest-state commits on `mac-native` (`4823f6e8`, `33c36514`), D2 and E. **Overlap:** D, E.

**G result (2026-10-12, `5624e54a`):**
- `L/libc/include/GuestMachineState.hpp`: gregs (DWARF order), rip, rflags, FXSAVE image, fs base; libc exports `LibcHasMachineState/LibcGetMachineState/LibcSetMachineState/LibcMakecontextEntry_nid_no_patch`, the runner sets them with `Aps5SetBridgeMachineState_nid_no_patch`. Get: the innermost bridge call (xmm0-7 from the trampoline block, RFLAGS from `ReconstructCompactedEFLAGS`). Set: a `MachineFrame` exception caught in `Bridge::Invoke`, applied with `SetFlagsFromCompactedEFLAGS`; on the fast path FEXCore does not reload NZCV after the thunk call, so CF/ZF/SF/OF of a resumed context are lost (TechnicalDebt).
- `makecontext`: rip is 48 bytes of x86 at stub page offset 0x200 that call the function with the six arguments, then `Aps5MakecontextFinish_nid_no_patch` (L/libSceLibcInternal). No library exports `swapcontext`; the fixture builds it from `getcontext`/`setcontext`.
- `siglongjmp` to another x87 control word resumes the whole state; `sceKernelRaiseException` on the calling thread delivers (G2, another thread, still throws).
- Tests: fixtures `context`, `context-exit`, `raise`; ctest `guest_user_context`, `guest_user_context_exit`, `guest_raise_exception` run the host argument checks, then these fixtures (`T/run_host_and_fixtures.py`).

### H. Microbenchmarks
**Kernels** (`T/bench/*.cpp`):
- integer loops (sum, hash, branchy code)
- memcpy: libc's (native), `rep movsb`, and a hand-written loop
- SSE2/SSE4.1 and AVX2: saxpy, dot product, 4×4 matrix, byte histogram, string search
- atomics: `lock xadd` and `cmpxchg` on 1, 2 and 4 threads
- the bridge round trip: guest→native trivial export, and native→guest through a qsort comparator

**Driver** (`T/run_fex_bench.py`):
- It builds each kernel as a guest program, the same way as the fixtures: Apple clang `--target=x86_64-unknown-freebsd13 -O2 -march=znver2`, nidify, `ld.lld`, relinker (T/fex_toolchain.py:14, :50-52; T/run_fex_fixtures.py:111-154).
- It builds the same source natively with `xcrun clang++ -O2 -arch arm64`.
- It writes JSON to `.mac-work/bench/`.
- Timing: RDTSC becomes `CNTVCT_EL0` × 64 (FC/Source/Interface/Core/JIT/ALUOps.cpp:78-91), which is enough for ratios.

**TSO on/off:**
- FEX config `TSOEnabled`, `VectorTSOEnabled`, `MemcpySetTSOEnabled` and `HalfBarrierTSOEnabled` (Config.json.in:450-473).
- The runner reads no `FEX_*` variables: `Config::Initialize` adds only the meta layer. Add `APS5_FEX_CONFIG="Key=Value,…"` and apply it with `FEXCore::Config::Set` before R/GuestCpu.cpp:397.

**Runs:** P-cores and E-cores through QoS classes. Sustained runs last at least 10 minutes on this fanless Air; report the median and the 1% low.

**Package summary:**
- **Files:** `T/bench/*`, `T/run_fex_bench.py`, R/GuestCpu.cpp (config hook), `core/fex/CMakeLists.txt`.
- **Tests:** only `fex_bench_smoke` is in ctest (tiny iteration counts, checks results, `TIMEOUT 30`). The ratios are recorded in PROGRESS.md and later PERFORMANCE.md.
- **Gate:** at least 0.7× native on integer and SIMD loops (D6 revisit trigger).
- **Size:** kernels 300, driver 200, config 50, CMake 30 LOC; 1.5 d.
- **Dependencies:** none for the kernels; the config hook comes after R. **Overlap:** B, C, F, R (GuestCpu.cpp).

**Result (2026-10-12):**
- **Kernels:** 37 in T/bench/kernels.cpp: 10 integer, 6 memory, 6 SIMD, 12 atomic (`lock xadd`, `cmpxchg`, an `xchg` spinlock and private counters on 1, 2 and 4 threads), 3 bridge (guest→native call, libc `qsort` comparator, thread create+join).
  - One source is built natively (`xcrun clang++ -arch arm64 -O2 -ffreestanding -fno-builtin -fno-stack-protector`) and for three guest targets: sse2 (`-march=x86-64`), sse4 (`x86-64-v2`) and avx2 (`znver2`).
  - The driver refuses kernel objects that call a library function, such as a loop turned into `memcpy`.
- **Changes to the plan:**
  - Time comes from libkernel's `sceKernelGetProcessTimeCounter` (the guest) and `clock_gettime(CLOCK_MONOTONIC)` (native, the same clock, 1 µs), not RDTSC. Each repetition also records its thread's CPU time (guest `clock_gettime(14)`).
  - The native `qsort` of `bridge_qsort` copies libc.prx's heapsort (L/libc/src/FileAndHeap.cpp:384), so only the callbacks differ.
  - The config hook is R/Options.cpp, not inline in R/GuestCpu.cpp. `APS5_FEX_CONFIG` takes FEX's JSON names and throws on an unknown one; it is applied after `ReloadMetaLayer`. `APS5_FEX_QOS` is set on every thread that runs guest code.
- **Commands:**
  - `python3 core/fex/tests/run_fex_bench.py build-arm64/core/relinker/relinker build-arm64/core/fex/aps5-fex build-arm64/core/libs/libs [--variants avx2] [--config "" --config TSOEnabled=0] [--qos background]`
  - `--compare <json>...`
  - ctest `fex_bench_smoke`: 3 s.
- **Conditions:**
  - Other agents' builds kept the load average at 40-99 for an hour. The main run started when it fell to 5.9 and saw 6.6-11 (`.mac-work/bench/20261012-201230-idle.json`: three targets, TSO on and off, interleaved per kernel).
  - With `APS5_FEX_QOS=user-interactive`, both programs still ran at priority 31 (default QoS, `ps -M`). macOS has no core affinity.
  - The table uses CPU time, which held under load: the integer, memory and SIMD means of the two runs at load 40-99 (`-192209-loaded.json`, `-192403-loaded-tso.json`) are within 0.02 of it, the one-thread atomics within 0.13. Clock speeds in the idle run differ from it by up to 40% for single kernels.
- **Speed, native time / guest time, CPU time, geometric means:**

| group | sse2 | sse4 | avx2 | avx2, `TSOEnabled=0` |
|---|---|---|---|---|
| integer (10) | 0.60 | 0.60 | 0.57 | 0.71 |
| memory (6) | 0.45 | 0.46 | 0.44 | 0.56 |
| SIMD (6) | 0.42 | 0.60 | 0.57 | 0.57 |
| atomics, one thread (4) | 0.96 | 0.93 | 0.97 | 1.24 |

- **Gate: not met** under FEX's default TSO: integer 0.57, SIMD 0.57 on avx2.
  - At or above 0.8 on avx2: int_hash 0.99, int_mix 0.81, int_sort 0.87, simd_saxpy 0.88, simd_dot 0.85, libc `memcpy`/`memset` 1.0. Atomics: 0.70-1.15 on one thread, 0.80-2.27 by the clock on two and four.
- **Costs, avx2** (TSO off in parentheses):
  - **TSO.** Every guest GPR load becomes `ldapur` plus `nop`, every store `nop` plus `stlur` (FEXOpt/MultiInst_TSO.json).
    - Affected: int_histogram 0.32 (1.00), mem_copy_scalar 0.25 (1.04), int_interp 0.40 (0.56), int_sum 0.38 (0.60), int_crc32 0.74 (0.84).
    - The spinlock on 2 and 4 threads gives wrong counts with TSO off, so TSO stays on (D10).
  - **Indirect control flow**, with TSO off: int_interp (jump-table dispatch) 0.56, int_sort_fnptr (indirect calls) 0.55, int_fib (calls and returns) 0.60, int_lexer (branches) 0.50.
  - **VEX.** Every VEX.128 instruction also zeroes the upper YMM half in the context (`stp xzr, xzr, [x28, #off]`, AVX128/VEX_map*.json).
    - Results: simd_mat4 0.32, simd_blur 0.38, simd_memchr 0.55 (`vpmovmskb` is 7 instructions).
    - avx2 trails sse2/sse4 on simd_dot (0.85 against 1.0) and mem_copy_vector (0.85 against 1.01).
  - **Scalar SSE.** The sse2 simd_pcm16 is 0.10: clang emits `maxss`/`minss`/`cvttss2si`/`pinsrw` per element. sse4 and avx2 vectorise it (0.77, 0.69).
  - **Tuning.** With `-march=znver2` clang does not unroll int_sum's loop (5 instructions per element), unlike the x86-64 and arm64 builds: avx2 0.38 against sse2 0.66.
- **Bridge, every target:**
  - guest→native call 105 ns (a native call is 1.3 ns);
  - native→guest callback through libc's `qsort` 124-129 ns (native 5.5 ns);
  - thread create+join 175-206 µs by the clock (native 33 µs);
  - 64-byte `memcpy` through the bridge (mem_memcpy_small) at 0.05.
  - FEXCore's log shows a 16 MB lookup cache allocated per guest thread.
- **Found:** int_lexer's text generator died with SIGBUS under the default TSO (package B). Clang had merged two byte stores into a `movw`, which crossed a 16-byte boundary. The generator now writes single bytes.
- **Not done:** E-core runs (`--qos background`) and the 10-minute sustained run. Cold times (the first repetition, including translation) are only in the JSON.

## Schedule

| Wave | In parallel (disjoint files) | Then |
|---|---|---|
| 1 | R (`3rdparty/FEX`, four GuestCpu.cpp call sites); A on the pin (T/asm, harness), which gives the R baseline; D1 (generator, `.inc`, check test); E (libs, `StubFor`); H kernels and driver | — |
| 2 (after R) | B (`OnFault`) and C (`Pool*`, CodePool); H config hook and runs; rerun A on the rebased FEX and compare pass lists | F after C |
| 3 | D2 (Bridge marshalling, thunks, `CallGuest`) after D1 and E | G after D2 and M0's guest-state commits |

- **Files several packages edit:**
  - R/GuestCpu.cpp: R, B, C, F, H, D2.
  - R/Bridge.cpp: D, E, G.
  - T/run_fex_fixtures.py: every package that adds a fixture.
  - `core/fex/CMakeLists.txt`: A, C, D, H.
- **Merging:** land R first, then C, then the others rebased. Keep fixtures in separate files so only the `COMPILED` list conflicts.
- **Total:** about 15-18 agent-days of work. With 3 agents in parallel, about 6-8 days.
- **M1 exit run:**
  - the full `ctest -L fex_asm` plus the `fex_*` tests;
  - the coverage report for every fixture, plus the static AOT part for DOOM (its run-time part needs M2's GPU path, because `app.elf` imports `libSceAgc`/`libSceVideoOut`);
  - the microbenchmark ratios with TSO on and off.

## DOOM (`PPSA99666`, lowbit/ps5-doom v1.1.1)

**Unwrap:** `python3 tools/unself.py .mac-work/titles/dl-ps5-doom-v1.1.1/extracted/PPSA99666 .mac-work/titles/PPSA99666`.
- `eboot.bin`: 0xd9fd5 → 0xef7b0 bytes. PT_NOTE 13 (0x18 bytes) is not stored in the SELF and is zero-filled.
- `sce_module/libc.prx` (54 14 F5 EE variant): 0x139a42 → 0x14629a bytes. Its 38 section headers are dropped.
- The download directory is byte-identical before and after (sha256 list).

**llvm-readelf** (Homebrew `llvm@22`): no warnings for either file.
- ELF64 x86-64, FreeBSD ABI, types `0xfe10`/`0xfe18`, 14 program headers.
- `eboot.bin`: 88 dynamic entries with 15 `NEEDED` (`libSceAgc.prx`…`libkernel.prx`) and 171 dynamic symbols.
- `libc.prx`: `SONAME` `libc.prx`.

**Relink**, with a snapshot of `build-arm64/core/relinker/relinker` (sha256 `e9ff6c08…`):
```
relinker .mac-work/titles/PPSA99666/eboot.bin .mac-work/titles/PPSA99666-relinked/app.elf
```
- Output: `System: Linux`, 171 NID references, guest module `app0/sce_module/libc.prx.guest.prx`; 0.5 s.
- `app.elf`: ET_DYN x86-64, interpreter `/lib64/ld-linux-x86-64.so.2`, `RUNPATH $ORIGIN/libs`, `BIND_NOW`, 17 `NEEDED`.
- `app0/` also holds `wads/`, `sce_sys/` and `licenses/`.

**To run:** put `libs/` next to `app.elf` (the arm64 prx libraries) and run `build-arm64/core/fex/aps5-fex app.elf` once a FEX-enabled arm64 build exists.
