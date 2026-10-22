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

Sources:
- DOOM: perf5 p0, build-metal before its d227f1e36 rebuild, 60 s run, window 20:30, sample 10 s, powermetrics 60 samples, Nominal pressure.
- ASTRO: PERF5 runs on build-perf5 (logs .mac-work/logs/perf5/ab: awarm, bq*, bg1, cq*, cl2, ct0), plus PERF3's asamp1 sample.
- Showcase 1.0.2: PERF3's baseline trace sA2 (49 frames) and PERF1's `sample` base3.

### ASTRO BOT (PPSA21564)

**Run cq1:** build-perf5 at 933f3325c, SDL dummy audio, warm FEX and shader caches, seconds 180:60 of 270. p50 398 ms, p99 1002 ms, 139 presents. P cores ran at 4.0 GHz, E cores at 2.2 GHz; the GPU at 733 MHz, 44.5% active; Heavy thermal pressure. The process used 0.8 cores, 96% of it on P cores.

**What the frame waits on.** Queue worker 0x0 is the critical path: it is in a phase for 432 of 434 ms per frame, and 313 ms of that is not in a wait. The GPU is busy 144 ms per frame (33%). The async queues 0x40 and 0x48 wait for queue 0 for 367 and 417 ms in `memory_wait`. That is idleness caused by the dependency, not their own cost.

**The frame follows the P-core clock.** Queue 0's non-wait time per frame was 656 ms at 1.4 GHz (awarm, Heavy) and 363 ms at 3.1 GHz (bq0b), and the frame time moves with it:

| P GHz | pressure | p50 / p99 ms | run |
|---|---|---|---|
| 1.4 | Heavy | 712 / 2002 | awarm (device audio) |
| 2.27 | Moderate | 586 / 1539 | bg1 (device audio) |
| 3.1 | Heavy/Moderate | 455 / 874 | bq0b (device audio) |
| 3.36 | Heavy | 385 / 825 | cq0 (dummy audio) |
| 4.0 | Heavy | 398 / 1002 | cq1 (dummy audio) |
| 4.05 | Moderate/Nominal | 346 / 580 | cl2 (dummy audio) |

On the fanless Air in a 38 °C room, the P cluster throttled to 1.4 GHz for minutes at a time. At that point E cores (2.27 GHz) do as much per thread as P cores, so where a thread is placed stops mattering. Thermals are the largest single factor in ASTRO's frame time on this machine.

| # | Hog | Cost | Evidence (cq1 unless noted) | Owner | Proposed fix |
|---|---|---|---|---|---|
| 1 | Queue 0's CPU work in the driver | about 313 ms/frame on the critical path; CPU-clock bound | trace: work 313 of 432 ms; sample on queue 0 (bq0b): AGC driver 54% of samples | AGC driver (PERF3/DRV-D) | Items 2-5 |
| 2 | `EqualsCommittedUnsynced` + `copyCommitted`: snapshot compare and copy of untracked guest memory | 22% + 5% of queue 0 (bq0b); 990 + 187 samples process-wide | sample | DRV-D | Write-tracked host imports |
| 3 | `Recorder.Sync` inside draws and dispatches | 63 ms/frame on queue 0, while the GPU idles | trace | PERF3 | Ranged waits instead of full drains |
| 4 | `Metal::Residency::Commit` | 10% of queue 0 (bq0b); 450 samples | sample | Metal backend (PERF3/DRV-D) | Batch residency-set changes per submission |
| 5 | `MaterializeDcc` / `MaterializeCmask` on the CPU | about 10% of queue 0; 286 + 55 samples | sample | AGC driver | GPU-side DCC materialisation, or skip DCC that nothing samples |
| 6 | TBB worker threads spinning in `scePthreadYield` | 0.26 core (3 threads × 8%) when present | sample cq1: 247-264 samples each | PERF5 (libkernel) | Yield back-off (D73) |
| 7 | GPU-mutex waits (`GpuMutex.wait.label`, gpu_mutex_wait) | 726 samples parked process-wide | sample | PERF3 (D66 cut the present's part) | Shorter holds |
| 8 | AvPlayer h264 (FFmpeg) decode of the intro movies | 0.15 core, all on its own threads | sample: receiveFrames, cabac | HLE libs | Hardware decode (VideoToolbox) |
| 9 | Guest audio thread (SceSndzAudioOutMain) | 8% of a core at 3.6 GHz (cq1); 19-23% at 1.4 GHz; 96-100% in earlier scenes (PERF3 asamp1, a1) | counters; AudioOut2 trace: 93.9 pushes/s = real time (512-sample grains at 48 kHz), so its work is paced, not a spin | FEX/translation | Translation cost of the title's mixer. TSOEnabled=0 cannot be measured on the title (it hangs, see below) |
| 10 | FEX lookup-cache misses and translation | 20-28k misses per 30 s, 50-230 ms per 30 s inside CompileCode; translation otherwise about 0.01 core with a warm disk cache | APS5_FEX_STATS | PERF5 (FEX) | L2 lookup table on (D71) |
| 11 | Profiling's own cost: `beginTiming` / `WriteTimestamp` (APS5_PROFILE_GPU=1) | about 1% of queue 0 | sample | tool | Off in timing runs (`profile_title.py run --no-gputime`; the driver enables it for any value of the variable) |

Translation-specific findings:
- With a cold disk cache FEX compiles about 28,700 blocks in the first 30 s (3.5 s of JIT, 122 us per block). Warm, JIT time is 0.2-0.8% of a core and comes only from lookup misses.
- `TSOEnabled=0` (APS5_FEX_CONFIG) deadlocks ASTRO about 35 s in. FEX's TSO emulation is required. Hardware TSO (ACTLR_EL1.TSOEN) is per-thread state that macOS gives only to Rosetta; on macOS 14.4+ only a Hypervisor.framework guest can set it.
- The guest sampler filed samples outside a thread's code buffer under one stale RIP, so on ASTRO 78-92% of the audio thread's samples looked like `libc.prx.guest.prx+0x6dbb0 host`. It now separates translated code, FEX's dispatcher, displaced-access stubs and native code per thread.
- A fresh build directory has a cold shader cache (`build-*/core/fex/shader_cache`). ASTRO then shows no frames for 150+ s ("a compute pipeline still compiling after 5 s"). Warm it before measuring.

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
