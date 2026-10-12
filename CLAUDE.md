# CLAUDE.md: native Apple Silicon port of AnyPS5

You are porting AnyPS5 (PS5 executable relinker + reimplemented Sony system libraries + RDNA2 shader recompiler) to run **natively on Apple Silicon**: arm64 only, Metal only, no Rosetta, no MoltenVK, no per-game reverse engineering. Read `docs/mac/MISSION.md` before anything else. It holds the goal, the architecture, prior community work, known pitfalls, and the milestones M0–M4 with exit criteria.

## Files you maintain
- `docs/mac/MISSION.md`: knowledge base. Fix it when a fact turns out wrong.
- `docs/mac/PROGRESS.md`: append-only log. After every meaningful step: date, milestone, what was done, exact verification commands and their summarised results, disk usage, next step. **This file is your memory across context compaction and new sessions.** On every start or resume, read the last ~100 lines first and continue from there.
- `docs/mac/DECISIONS.md`: one entry per design decision (context, options, measurements, choice, date). Never re-litigate a recorded decision without new evidence.
- `docs/mac/COMPATIBILITY.md`, `docs/mac/PERFORMANCE.md`: created in M3/M4.

## Autonomy
- Execute M0 → M4 **fully autonomously**, in order. Do not stop to ask permission for code, builds, tests, refactors, installs from Homebrew, or design choices you can settle with evidence. Decide, record in DECISIONS.md, continue.
- **The only mandatory stop is storage** (see below).
- You *may* ask the user's opinion (AskUserQuestion) when a choice is genuinely theirs: product direction, licence questions, using Rosetta as a test-only oracle (MISSION §6.4), anything needing their password or hardware action (e.g. `sudo powermetrics`, installing Xcode), or providing a game title. Batch questions; keep working on something unblocked meanwhile; never ask about things you can measure.
- A milestone is done only when every exit criterion in MISSION §8 is met **and verified by running it**. No "should work". Report failures honestly in PROGRESS.md.

## Storage budget: hard rule
- Everything you create or download (repo, submodules, build dirs, caches, Homebrew downloads, ccache, DerivedData, test titles, traces, captures, scratch) must stay **≤ 100 GB total**. The disk has ~250 GB usable, shared with the user.
- Run `tools/mac/disk_budget.sh` at the start of every session, before and after every large download or build, and at each milestone end. Exit 1 (≥85 GB): clean up before continuing. Exit 2 (>100 GB): stop and clean; if you truly cannot fit, **ask the user for a storage exception**. That is the one question you must ask rather than decide.
- Keep all project artifacts under the repo: build dirs `build-*` (gitignored), everything else in `.mac-work/` (gitignored). Don't scatter files elsewhere; if a tool insists (Xcode DerivedData, Homebrew cache), clean it with that tool's own command.
- Clean up after yourself: at most **two** live build directories at a time; delete a build dir once it is superseded; `brew cleanup` after installs; delete downloaded archives after extraction; trim ccache (`ccache -M 10G`); delete GPU captures/traces once their findings are written down. Shallow-clone or `--filter=blob:none` reference repos into `.mac-work/ref/` and delete them when no longer needed.
- Never delete anything outside the repo that you didn't create, and never `git clean -fdx` the repo root (it would delete `.mac-work/`). Before deleting, look at what you're deleting.

## Context hygiene: don't ruin your own context
- **Use subagents aggressively** (Agent tool): `Explore` for codebase sweeps, `general-purpose` for parallel investigations, research and long build/test-triage loops, `Plan` for design reviews. Send independent subagents in one message so they run in parallel. Ask them for conclusions, file:line references, and short excerpts, never whole files or full logs.
- **Use web search fully** for Metal / MSL / Metal 4 docs and the Metal Feature Set Tables, ARM64 ISA and Apple CPU behaviour, x86-64 semantics (Intel SDM), LLVM, FEX, box64, Rosetta analyses, RDNA2 ISA, prior PS4/PS5 projects, and the upstream issues listed in MISSION §4. Prefer primary sources; cite URLs in DECISIONS.md.
- Never print large outputs into context. Redirect builds and tests to log files under `.mac-work/logs/` and read them with `tail`, `grep`, or a subagent. Use `ctest --output-on-failure`, `ninja ... 2>&1 | tail -40`, `grep -c`, `wc -l`.
- Read files with offsets/limits; don't re-read files you just wrote.
- Keep PROGRESS.md current so compaction loses nothing.

## Testing and simulation: you run everything yourself
- Build: CMake + Ninja, native arm64 (`-DCMAKE_OSX_ARCHITECTURES=arm64`). Use `ccache` if available.
- After every change: build, then run the relevant tests. At milestone ends, run the full suite.
- **GPU safety:** run GPU tests one at a time with a timeout (`ctest -j1 --timeout 30 -R <pattern>`), never under a debugger, and leave out `agc_driver_lds_atomics64` and window-opening flip/display tests until the Metal backend is stable. A wedged GPU can panic WindowServer and reboot the Mac, so wrap any new GPU workload in a timeout and test it on tiny inputs first.
- Write new tests for everything you add: translator per-instruction tests, differential tests, MSL backend tests, Metal driver tests. Follow the repo's test registration conventions (`tools/check_test_registration.py`).
- Long runs (titles, 10-minute soak tests, benchmarks) go in the background with timeouts and logs; check them with Monitor or subagents.
- Thermals: this is a fanless MacBook Air. For performance numbers use sustained runs (≥10 min) and report median + 1% low frame time. Don't run heavy GPU soak tests back to back without a cool-down.

## Code rules
- Follow upstream conventions (`docs/dev/CONVENTIONS.md`, `CONTRIBUTING.md`): unsupported states throw `std::runtime_error` with a clear message; record known gaps in `docs/dev/TechnicalDebt.md`.
- Platform code goes behind `__APPLE__` / `__aarch64__` or in `specifics/macos` directories, matching the existing structure. Don't break the build structure for other platforms unnecessarily, but Linux/Windows support is not a goal of this fork.
- Licences: the project is GPL-2.0-only. Only bring in GPL-2.0-compatible code (MIT/BSD/Apache-2.0 with care, LGPL ok). FEX is MIT only up to willfaust/FEX `ac555dd8`; anything GPL-3.0 is not allowed. Record every third-party import in DECISIONS.md.
- Legal: never download, search for, or link commercial game dumps, firmware, or keys. Test titles: freely distributed PS5 homebrew (verify and record the licence) or titles the user provides.

## Git
- Work on branch `mac-native`. Commit early and often with clear messages; one topic per commit. End commit messages with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (replace the model name with your own if different).
- Remote `upstream` is boykopovar/AnyPS5. **Never push** anywhere or open PRs/issues/comments on GitHub without asking the user. Fetching upstream to merge new fixes is fine (`git fetch upstream`), and worth doing at each milestone start.

## Machine
MacBook Air M3, 4P+4E CPU, 10-core GPU (Metal 4), 16 GB, macOS 27.0.1, fanless. At handoff only the Command Line Tools were installed (no full Xcode). Check `xcode-select -p` and `xcrun --find metal` at start; if Xcode is still missing, ask the user once to install it, and keep working with runtime MSL compilation (`newLibraryWithSource`) in the meantime.
