# Kickoff prompt

Paste the block below as the first message of a new session started in `/Users/naitikjoshi/PS5`. To resume after a break, paste the same block; the agent picks up from `docs/mac/PROGRESS.md`.

---

```
You are taking over a long-running engineering mission: porting AnyPS5 so PS5 titles run natively on Apple Silicon macOS, arm64 only, Metal only, with no Rosetta, no MoltenVK, and no per-game reverse engineering. The previous session did the research and planning. Your job is execution, milestones M0 through M4, fully autonomously.

Start by reading, in this order:
1. CLAUDE.md (operating rules; they override your defaults),
2. docs/mac/MISSION.md (goal, architecture, prior community work, known pitfalls, milestone exit criteria),
3. docs/mac/DECISIONS.md (settled decisions; don't reopen them without new evidence),
4. the tail of docs/mac/PROGRESS.md (where work stands; continue from its "Next" line).
Then run tools/mac/disk_budget.sh and check the toolchain (xcode-select -p, xcrun --find metal).

How to work:
- Go through M0 → M1 → M2 → M3 → M4 in order without waiting for approval. A milestone is complete only when every exit criterion in MISSION §8 has been verified by commands you actually ran, with results logged in PROGRESS.md. Then clean up superseded build directories, run the disk budget check, commit, and start the next milestone.
- Run every build, test, benchmark and simulation yourself. Write new tests for everything you add. Never claim something works without running it; log failures honestly.
- Storage: everything you create stays within 100 GB (tools/mac/disk_budget.sh). Clean up after yourself continuously. If you cannot fit within 100 GB, stop and ask me for a storage exception. That is the only thing you must ask about.
- You may ask my opinion (batched, while continuing unblocked work) on choices that are genuinely mine: product direction, licensing, using Rosetta purely as a test oracle, anything needing my password or a hardware action, or a game title to test. Decide everything else yourself from evidence and record it in DECISIONS.md.
- Use your web search heavily for primary sources (Metal/MSL/Metal 4 docs and feature set tables, ARM64 and x86-64 ISA references, LLVM, FEX, RDNA2 ISA, the upstream issues and community branches listed in MISSION §4). Check what has merged upstream before rebuilding anything the community already did.
- Use subagents heavily and in parallel: Explore for codebase sweeps, general-purpose for research, build/test triage and long investigations, Plan for design reviews. Have them return conclusions and file:line references, not dumps.
- Protect your context: send build and test output to log files under .mac-work/logs/ and read only tails or grep results; never paste whole logs or large files into the conversation; keep PROGRESS.md up to date so context compaction loses nothing.
- GPU safety: run GPU tests one at a time with timeouts; never run unbounded GPU workloads (a wedged GPU can panic WindowServer).
- Git: commit often on branch mac-native. Never push, open PRs, or post on GitHub without asking me.

Begin now with the "Next" step in PROGRESS.md.
```
