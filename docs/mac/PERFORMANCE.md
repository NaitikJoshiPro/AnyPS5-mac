# PERFORMANCE: M4 measurements

Every optimisation is logged here with before/after numbers on at least two workloads (one a 3D title), as MISSION §8 M4 requires. Newest measurements at the bottom of each section.

## Measurement rules

- **Workloads:** PS5 OpenGL Showcase 1.0.2 (PPSA99005, the 3D title: 262,144 particles, 8,192 instances, HDR bloom, compute SSBO, multi-draw-indirect-count at 3840x2160) and DOOM Shareware (PPSA99666, the CPU baseline). ASTRO BOT joins once it renders. No per-title code.
- **Frame time:** the driver's performance trace (`APS5_TRACE_FRAMES=<start>:<count>` or `APS5_TRACE_SECONDS=<start>:<length>`, `APS5_TRACE_DIR`): `frame_ms` p50 (median) and p99 (the 1% low frame time) of present-to-present intervals, plus `top_phases_ms_per_frame` and `worker_busy_pct`. Showcase: frames 70:50 (the scene, after the SDK's 30 s start and the pipeline warm-up); DOOM: seconds 20:30.
- **Runs:** `tools/mac/title_run.py run` (GPU lock, headless, no `--validation` for timing; gate runs use it), build type RelWithDebInfo, `--max-load 8`. Two workloads, A/B interleaved (A B A B or B A B A) in one session, because thermals drift between runs.
- **Thermals (fanless Air, warm room):** every number records the P-cluster and GPU frequency and GPU active residency, sampled mid-run with `sudo -n powermetrics --samplers cpu_power,gpu_power -n 5 -i 1000`. "P-cluster 0 MHz" means the P cores were idle in the samples (the busy threads ran on E cores). Sustained 10-minute numbers wait for a cool machine.
- **Profiles:** host `sample <pid> 10-15`, the guest sampler (`APS5_FEX_SAMPLE=2`), `APS5_PROFILE_DRAW=1` (driver counters every 10 s; inflates frame time), `APS5_PROFILE_GPU=1` (`[gputime]`, valid since D59).

## Resolution target (M4)

Measure and optimise for an **internal render resolution of about 1920x1080 or lower, upscaled with MetalFX to a 16:9 output** (letterboxing is fine), not native 4K. Showcase renders at its own 3840x2160 because the title fixes it (eglSetDisplayModePS5(3840, 2160)); the console display mode (D61, default 1080p) cannot lower it. Its numbers below are a GPU-bound worst case: about 84-90% GPU active residency, with about 6 s of GPU batch time per 10 s. The internal-resolution override plus MetalFX (MISSION §8 M4) is the main lever for it.

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
| D61 | Console display mode 1080p (reported by VideoOut) instead of 2160p | 203 / 259, 215 / 268 -> 219 / 232, 215 / 233 (2560x1440 target, GPU 1.15-1.20 GHz, ~70% active): fixed 4K title, no change | not applicable (DOOM registers 1080p) | yes (system behaviour; ASTRO 303 / 1598 (load 22), 323 / 437 -> 347 / 438, 336 / 395 ms, P 4.05 GHz, GPU 0.66-0.73 GHz: no change, its scene is already 1920x1080 / 2432x1368) |
| D61 | MetalFX spatial upscale in the present (1080p -> 2560x1440) instead of bilinear | not applicable (4K source, downscale) | 33.38 / 41.62, 33.37 / 41.63 -> 33.36 / 41.63, 33.37 / 41.60 (P 3.97-4.0 GHz, cool; paced 30 fps); Present 10.2-10.3 -> 10.7 ms per frame, GPU 420 -> 482 MHz | yes (image quality; ~0.5 ms per present) |
| D66 | Display buffer decoded on the GPU from its host import, no CPU wait under the GPU mutex (vs the CPU decode) | 616 / 1579, 562 / 621 -> 279 / 557, 524 / 572 (Heavy thermal pressure: P 0-3.1 GHz, GPU 350-680 MHz); present_display_buffer 130, 92 -> 4.1, 3.8 ms per frame | 33.37 / 38.5, 39.1 / 45.4 -> 33.36 / 37.1, 33.37 / 37.5 (paced; throttled GPU 338-660 MHz); Present 9.2, 38.6 -> 3.6, 5.8 ms per frame | yes (ASTRO 412 / 1044 (P 3.4 GHz, load 2), 499 / 1059, 951 / 1988 (load 12-17) -> 333 / 502 (P 3.7 GHz, load 7), 436 / 806 (P 2.0 GHz); present_display_buffer 41-94 -> 3.7-4.4 ms, queue 0 GPU-mutex wait 51-99 -> 12-14 ms per frame) |
| D68 | Game preset "Performance Mode" (GetGamePresets priority 1) instead of Game Default (0); system answers from one set of console settings | not applicable (Showcase does not read presets) | not applicable (DOOM does not import GetGamePresets) | yes (system behaviour; ASTRO A/B/B/A 308.6 / 424.1, 435.3 / 883.9, 328.9 / 388.2, 394.3 / 1232.2 ms with P 3.7, 1.8, 3.3, 2.2 GHz and GPU 0.6-0.87 GHz: frame time follows the P-cluster clock, not the preset; display buffers 3840x2160 and scene 1920x1080 / 2432x1368 in every run) |

## Display resolution and upscaling (D61, PERF2, 2026-10-15)

What the 3D workloads render at (`APS5_TRACE_VIDEOOUT=1`):
- **Showcase 1.0.2** is fixed at 4K. It calls `eglSetDisplayModePS5(3840, 2160)` and registers 3840x2160 display buffers before it asks the system anything. The console display mode cannot lower it, and there is no safe general render-target scale (D61). Its frame time at 4K is the GPU-bound worst case.
- **ASTRO BOT** reads the output class and then registers 3840x2160 display buffers under every mode. Its scene renders at 1920x1080 and 2432x1368 (dynamic resolution), and the title upscales to 4K itself. Its frame time is dominated by `Driver.Packet.memory_wait` (600-900 ms per present), not by resolution.
- **DOOM** registers 1920x1080 display buffers. A 1080p source in a 16:9 output larger than 1080p is where MetalFX applies.

Headless runs present into `APS5_PRESENT_SIZE=2560x1440` (the user's 2560x1664 panel letterboxed to 16:9). Without it, title_run's offscreen drawable is 614x345.

## ASTRO BOT: what `Driver.Packet.memory_wait` is (PERF3, 2026-10-15)

`top_phases_ms_per_frame` sums every worker thread. In PERF2's aA4 trace (60 s), memory_wait was 50.0 s on async compute queue 0x40 and 57.3 s on 0x48, but only 4.5 s on graphics queue 0. The async queues sleep in WAIT_REG_MEM most of the frame, waiting for queue 0's labels: that is idle time, not a cost. Queue 0 is the critical path (100% busy). Its memory_wait (24-34 ms per frame) waits for queue 0x40's dispatches: about 45 ms of CPU recording, which contends with queue 0 for the GPU mutex. The label handoff itself (the last 0x40 dispatch to queue 0 waking) takes about 5 ms, so commit and label latency is not the lever.
Queue 0's real costs per frame (aB3, after D66): Recorder.Sync about 70 ms, WaitSerial about 20 ms, GPU-mutex waits about 16 ms. Before D66 the present's mutex hold (41-94 ms) added to these.
A host `sample` of queue 0 (asamp1, during a pipeline-compile stretch) shows recording at about 68% of samples:
- memcmp and copies of guest memory for StorageTexture revalidation (`compareUntracked`, `captureGuestBytes`)
- CMASK materialisation (`ColorTargetLayout::Offset`)
- `Residency::Commit` from `TryWrap`
- pipeline-key string building in `PipelineCache::PrepareCompute`

GPU waits are about 10% of samples. Two thirds of those are `StorageTexture::Refresh` -> `upload` -> `captureGuestBytes` -> the flush hook's SyncThrough.

## What ASTRO BOT asks before it sizes its output (D68, PERF4, 2026-10-15)

- Before RegisterBuffers2 it calls only `sceSystemServiceParamGetInt(1)` (language), `sceSystemServiceGetStatus` and `sceVideoOutGetOutputStatus` once. It imports no GetResolutionStatus, IsOutputSupported, GetDeviceCapabilityInfo or ConfigureOutput.
- It uses the output status's dynamic range: an HDR answer (`APS5_HDR=1`) makes it register 3840x2160 with pixel format 0x8100070400000000 instead of 0x8100000000000000 (not decoded yet). It ignores the resolution class: 3840x2160 under class 1 and 2. Its 4K display buffers are its scan-out size on any TV; the scene renders at 1920x1080 / 2432x1368.
- After registering it polls GetOutputStatus, GetHdrToneMapLuminance and GetGamePresets every frame. The "Performance Mode" preset (now the default, `APS5_GAME_PRESET=default` for Game Default) changed neither its render sizes nor its frame time in four interleaved runs; frame time tracked the P-cluster clock (1.7-3.7 GHz on a shared machine).
- What would lower ASTRO's cost: the per-present `Driver.Packet.memory_wait` (560-800 ms) and the driver waits PERF3 is working on (D66 already moved the 4K display-buffer decode to the GPU), not the system answers.
## Waste audit (PERF5, 2026-10-15)

Tool: `tools/mac/profile_title.py run <ID> --tag T --seconds S --window START:LEN [--fex-sample 2] [--env K=V]` (one headless title_run under the GPU lock; `sample`, per-thread P/E counters, powermetrics, the driver trace, `[gputime]`, the guest sampler; report `<ID>-<TAG>-profile.md/json`). `profile_title.py report <log-dir> <ID> <TAG>` re-reads a run. "Cores" below is samples divided by samples per thread: the CPU kept busy (or the threads parked) on average. "ms/frame" multiplies that by the mean frame time.

Sources so far (no new title runs were allowed after ~14:15 because the user needed the Mac):
- DOOM: perf5 p0, build-metal before its d227f1e36 rebuild, 60 s run, window 20:30, sample 10 s, powermetrics 60 samples, Nominal pressure.
- ASTRO: PERF3's baseline trace aA4 (a09b099c06d4, APS5_PRESENT_GPU_DECODE=0, seconds 80:60, 125 frames) and PERF3's `sample` asamp1 (14:10, 981 samples per thread, 113 threads), both re-read with profile_parse.
- Showcase 1.0.2: PERF3's baseline trace sA2 (49 frames) and PERF1's `sample` base3.

### ASTRO BOT (PPSA21564): frame mean 479 ms (p50 412, p99 1044)

The critical path is queue worker 0x0. It is in a phase 99.9% of the time, 228 ms/frame of it waiting. The GPU is busy 243 ms/frame (about 50%). The async compute queues 0x40 and 0x48 wait on queue 0 for 365 and 451 ms/frame in `memory_wait`: they are idle because they depend on queue 0, which is not waste of their own.

| # | Hog | ms/frame or share | Evidence | Owner | Proposed fix |
|---|---|---|---|---|---|
| 1 | `Recorder.Sync` inside draws (88 ms) and dispatches (63 ms) on queue 0 | 150 ms/frame of the critical path; meanwhile the GPU idles | trace aA4: Sync under Driver.Draw.draw_and_resource_release 55, Driver.Draw 32, Driver.Dispatch 63 | AGC driver (PERF3/DRV-D) | Find which resource forces each full GPU drain (storage-image refresh, CPU readback) and make it a ranged wait or a GPU-side copy |
| 2 | Guest audio thread at 100% of a core in translated code | 1.0 core (98% of that thread's samples) | asamp1: 960 of 981 samples in JIT, hot PCs in two small clusters; only 16 bridge calls; AudioOut2MixPorts 9 | FEX/translation or HLE audio (PERF5, open) | Guest sampler run to name the loop: a free-running mixer (no pacing from our AudioOut2/Ngs2), a guest spin on a flag our HLE never sets, or real DSP made expensive by translation |
| 3 | Host CPU work on queue 0: `EqualsCommittedUnsynced` and `copyCommitted` (snapshot compare and copy of untracked guest memory) | about 0.35 core, about 170 ms/frame on the critical path | asamp1: 242 + 104 samples of queue 0's 981 | AGC driver (DRV-D: write-tracked imports) | Write-tracked host imports so that Refresh skips the memcmp |
| 4 | `GpuMutex.wait.label` in prepare_memory | 51 ms/frame | trace aA4 | AGC driver (PERF3) | Shorter GPU-mutex holds (presenter, queues 0x40 and 0x48 polling) |
| 5 | `ColorTargetLayout::Offset` on the CPU | about 0.08 core, about 38 ms/frame on queue 0 | asamp1: 78 samples | AGC driver | Per-row or per-tile addressing instead of per-texel, or a GPU detile |
| 6 | `Recorder.WaitSerial` + `memory_wait` on queue 0 | 26 + 47 ms/frame | trace aA4 | AGC driver (PERF3) | GPU completion pushed to the worker (MTLSharedEvent listener) instead of 1 ms polls |
| 7 | File-loader guest thread: stat/open system calls under ResolveHostPath | about 0.26 core | asamp1: 185 ResolveHostPath (83 stat), 67 open | HLE libc/libkernel | Cache resolved host paths and their existence (the guest opens the same files again) |
| 8 | Pipeline saver: SHA-256 and archive Save | about 0.10 core | asamp1: 79 SHA256 + 21 Save on "pipeline saver" | AGC driver / Metal pipeline cache | Save less often, or hash only new pipelines (D51 keeps earlier pipelines) |
| 9 | Metal submission (`iokit_user_client_trap` on com.Metal4.SubmissionQueue) and Residency::Commit | 0.14 + 0.06 core | asamp1 | Metal backend | Fewer, larger batches; residency changes batched (D44) |
| 10 | Translation (FEX compile, dispatcher, signals, thunks) | about 0.05 core in all | asamp1: compile 14, dispatcher 18, signals 6, thunks 11 samples (warm disk cache) | FEX | Not a hog after D49/D58; keep the disk cache warm |

### Showcase 1.0.2 (PPSA99005 v1.0.2, fixed 4K): frame mean 558 ms (p50 562)

The GPU is busy 420 ms/frame (75%) at 1.15-1.2 GHz (PERF2), so the title is GPU-bound. Queue 0 waits 316 ms/frame for submissions (Driver.WorkerWait) and 112 ms in `Recorder.Sync`; the presenter waits 58 ms in `Recorder.Sync`. The guest main thread sleeps about 98% of the time in `sceKernelUsleep(1000)` polling. The CPU totals about 0.6 of a core.

| # | Hog | ms/frame or share | Evidence | Owner | Proposed fix |
|---|---|---|---|---|---|
| 1 | GPU scene time at 4K | 420 ms/frame | trace sA2 gpu track | MSL/recompiler (SHD9), driver | Internal-resolution override + MetalFX (MISSION M4); shader cost (SHD9 wave64 halving) |
| 2 | CPU-GPU serialization: `Recorder.Sync` on queue 0 (112 ms) and the presenter (58 ms) | GPU idle gaps of about 140 ms/frame (25%) | trace sA2 | AGC driver (PERF3) | Same as ASTRO #1; present without a full drain (D60 revisited after write tracking) |
| 3 | `StorageTexture::Refresh` compare and upload (memcmp + upload of 1.6 GB per 10 s) | about 48% of the worker's busy time | PERF1 base3 | DRV-D (write-tracked imports) | Write tracking on imported memory |
| 4 | 4K display-buffer CPU decode under the GPU mutex | about 400 of 2649 samples on the presenter | PERF1 base3 | PERF3 (D66, GPU decode) | Landing |
| 5 | Guest polling with `sceKernelUsleep(1000)` | wake-ups only, no CPU cost; adds up to 1 ms of latency per poll | PERF1 base3 | none (title behaviour) | Leave it: a 1 ms nanosleep costs about 1.1 ms on macOS |
| 6 | Write-watch ticket lock yield (`GuestWriteWatchProtection`) and lease waits | about 3% of a core | PERF1 base3: 80 + 23 samples | libc write watch (PERF5) | Shorter hold or a reader path; small |

### DOOM (PPSA99666, CPU baseline): p50 34.3 ms, p99 37.8 ms (30 fps title pacing)

The process uses 0.17 cores, all on P cores (per-thread counters). The CPU is idle. Per frame, the presenter spends 15.8 ms in vblank_wait and 17.6 ms on work; the queue worker spends 21.7 ms in rendering_wait. The GPU is 82.7% active but at its lowest state (338-378 MHz) because the title paces itself at 30 fps. The GPU track shows 20.6 ms/frame for one compute dispatch.

| # | Hog | ms/frame or share | Evidence | Owner | Proposed fix |
|---|---|---|---|---|---|
| 1 | `DecodeDisplayBuffer` on the presenter (CPU) | 0.13 core, 4.5 ms/frame | p0 sample: 888 of 6668 | PERF3 (D66) | GPU decode |
| 2 | The GPU mutex held around the present: the queue worker waits | 0.62 core-equivalent parked (21 ms/frame), not CPU | p0 sample: 4118 GpuMutexType::lock on the queue worker | PERF3 | Shorter present section (D66) |
| 3 | The guest game thread | 2.3% of a core | p0 counters | none | none: translation is not a factor for DOOM |
| 4 | GPU DVFS at its lowest state while 83% active | no frame cost at 30 fps | p0 powermetrics | none | Only matters if the title wanted 60 fps |
