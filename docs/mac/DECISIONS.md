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
