# PERFORMANCE: M4 measurements

Every optimisation is logged here with before/after numbers on at least two workloads (one a 3D title), as MISSION §8 M4 requires. Newest measurements at the bottom of each section.

## Measurement rules

- **Workloads:** PS5 OpenGL Showcase 1.0.2 (PPSA99005, the 3D title: 262,144 particles, 8,192 instances, HDR bloom, compute SSBO, multi-draw-indirect-count at 3840x2160) and DOOM Shareware (PPSA99666, the CPU baseline). ASTRO BOT joins once it renders. No per-title code.
- **Frame time:** the driver's performance trace (`APS5_TRACE_FRAMES=<start>:<count>` or `APS5_TRACE_SECONDS=<start>:<length>`, `APS5_TRACE_DIR`): `frame_ms` p50 (median) and p99 (the 1% low frame time) of present-to-present intervals, plus `top_phases_ms_per_frame` and `worker_busy_pct`. Showcase: frames 70:50 (the scene, after the SDK's 30 s start and the pipeline warm-up); DOOM: seconds 20:30.
- **Runs:** `tools/mac/title_run.py run` (GPU lock, headless, no `--validation` for timing; gate runs use it), build type RelWithDebInfo, `--max-load 8`. Two workloads, A/B interleaved (A B A B or B A B A) in one session, because thermals drift between runs.
- **Thermals (fanless Air, warm room):** every number records the P-cluster and GPU frequency and GPU active residency, sampled mid-run with `sudo -n powermetrics --samplers cpu_power,gpu_power -n 5 -i 1000`. "P-cluster 0 MHz" means the P cores were idle in the samples (the busy threads ran on E cores). Sustained 10-minute numbers wait for a cool machine.
- **Profiles:** host `sample <pid> 10-15`, the guest sampler (`APS5_FEX_SAMPLE=2`), `APS5_PROFILE_DRAW=1` (driver counters every 10 s; inflates frame time), `APS5_PROFILE_GPU=1` (`[gputime]`, valid since D59).

## Resolution target (M4)

Measure and optimise for an **internal render resolution of about 1920x1080 or lower, upscaled with MetalFX to a 16:9 output** (letterboxing is fine), not native 4K. Showcase renders at its own 3840x2160 today because the driver has no resolution override yet. Its numbers below are a GPU-bound worst case: about 84-90% GPU active residency, with about 6 s of GPU batch time per 10 s. The internal-resolution override plus MetalFX (MISSION §8 M4) is the main lever for it.

## Baselines (2026-10-14/15, build-perf1 at 4c337f14a, headless)

| Workload | p50 ms | p99 ms | Conditions | Notes |
|---|---|---|---|---|
| Showcase 4K | 523.9 | 650.7 | P-cluster throttled (1.6 GHz), load ~27 | worker busy 34%; Draw 162 ms (Recorder.Sync 125), Present 64.8 per frame |
| Showcase 4K | 418-434 | 478-519 | load 1-6, GPU 0.85-1.29 GHz, 84-90% active | worker busy 38-41%; Draw 134-146 ms, Present 54-57 |
| DOOM 1080p | 34.6 | 37.5 | throttled, load ~27 | worker 99.8%: presenter holds the GPU mutex 8.6 s per 10 s |
| DOOM 1080p | 33.4 | 41.6 | cool, load 1-5 | paced at 30 fps by the title (900 presents in 30 s); dispatches 50 ms per 10 s |

Profile of Showcase (base4, `sample` 15 s, guest sampler):
- Not CPU-bound. The process uses about 0.6 of a core. The guest's main thread sleeps about 98% of the time, polling with `sceKernelUsleep(1000)`.
- Queue worker: 55% idle. Of its busy time, `StorageTexture::Refresh` takes about 48%:
  - GPU drains about 1,070 samples;
  - `compareUntracked` memcmp 743;
  - CPU uploads 810.
  - The GPU mutex wait is 472.
- Presenter: GPU wait plus the CPU decode of the 4K display buffer, under the GPU mutex.
- `GuestAllocationsAcquire`: 5 samples, which is negligible.
- GPU (D59): storage uploads and write-backs move about 1.6 GB each way per 10 s, yet take only about 180 ms of GPU time. The rest of the about 6 s is the scene.

## Changes

| Decision | Change | Showcase p50 / p99 (A -> B) | DOOM p50 / p99 (A -> B) | Adopted |
|---|---|---|---|---|
| D59 | `[gputime]` leaves out unwritten timestamp pairs (profiling fix) | not a frame-time change | not a frame-time change | yes |
| D60 | Resident present (present from the pending render target) | 424 / 478 -> 454 / 532; 418 / 519 -> 490 / 584 (GPU 0.83-1.07 GHz, 86-91% active) | not applicable (display buffer not pending) | no |
| D60 | Display-buffer GPU wait before the present takes the GPU mutex (full flush hook) | 391 / 432 -> 585 / 633 | 47.2 / 56.9 -> 44.9 / 49.7; 46.6 / 61.1 -> 36.0 / 103.1 (throttled, P 1.36-1.57 GHz); 33.4 / 56.1 -> 33.4 / 37.4; 33.4 / 41.6 -> 33.4 / 40.9 (cool) | no |
| D60 | Same, waiting for GPU writes only (no store outside the mutex) | 433 / 493 -> 418 / 517 | 33.4 / 41.6 -> 33.4 / 41.6 (x2, cool) | no |
