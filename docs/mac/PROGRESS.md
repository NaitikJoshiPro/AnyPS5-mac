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
