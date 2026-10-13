# Compatibility: native Apple Silicon (arm64, Metal)

Titles run through `relinker` → `aps5-fex` (FEXCore, arm64), with the arm64 prx libraries and the Metal backend. The machine is a MacBook Air M3 (10-core GPU), macOS 27.0.1. Only freely distributed homebrew is listed here, plus titles the user provides. Never commercial dumps, firmware or keys (CLAUDE.md).

Status levels: **Nothing** (doesn't load), **Loads** (relinks and starts, no frame), **Menu** (first correct frames), **In game** (gameplay reached), **Playable** (10 min, input and audio, correct frames).

## Titles

| Title | ID | Version | Status | Last tested | Top blocker | Run log |
|---|---|---|---|---|---|---|
| DOOM Shareware for PS5 | PPSA99666 | v1.1.1 | **In game** (launcher → skill menu → E1M1 rendered, pad input works, audio plays), alive at the 60 s cap | 2026-10-13, `build-metal` (libs 18:07, aps5-fex 18:11) | none at 60 s. 10-min soak, windowed run and Rosetta/x86 check (`vmmap`) not yet done | `logs/m3-boot/PPSA99666-pad1.log` |
| PS5 OpenGL Showcase | PPSA99005 | v1.0.1 | **Loads** (GL 4.6 context ready, first 3 submissions executed, then guest fail-stop) | 2026-10-13 | `[ps5-agc] submission retirement failed`: the guest's completion marker isn't seen by the CPU within the SDK's 2 s bound (SDK ≤1.0.1). GPU trace shows serials 5–7 done by 1.27 s. Suspects: completion-label visibility, or first-frame pipeline compiles (0.6 s per submission) | `logs/m3-boot/PPSA99005-noval-gpu*.log` |
| Homebrew UI Lab | PPSA99050 | 01.000.010 | **Loads** (fonts and shaders load, 3 submissions, then fail-stop) | 2026-10-13 | `std::runtime_error: AGC graphics: texture detiling requires source and destination buffers` (`TextureDetiler.mm:475`, from `StorageTexture::upload`, `Texture.mm:2145`, `import->buffer` null for a tile-mode-7 480x270/512x512 storage texture), also thrown on the VideoOut presentation path. With `APS5_TRACE_GPU` it segfaults in `TextureDetiler::Dispatch` instead | `logs/m3-boot/PPSA99050-trace2*.log` |
| SuperTuxKart (oops-apps) | STKT00001 | `latest-main` @ c590060e | **Nothing** (relinks; the loader stops) | 2026-10-13 | `missing library libSceAppInstUtil.prx`; also needed but missing: `libSceGnmDriver`, `libSceSysUtil` (`sceSysUtilSendSystemNotificationWithText`), `libSceVideoRecording`. None of these exist upstream either | `logs/m3-boot/STKT00001-try1.log` |

Logs are under `.mac-work/logs/m3-boot/`. Run directories (relinked `app.elf`, `app0` symlinked to the title, `libs` → `build-metal/core/libs/libs`) are under `.mac-work/runs/m3/<ID>/`, made by `prepare.sh`. `boot.sh` makes one headless attempt: `SDL_VIDEODRIVER=offscreen`, `APS5_DUMP_FRAMES`, `MTL_DEBUG_LAYER=1`, `MTL_SHADER_VALIDATION=1`, a 60 s alarm, all under `buildlock.py build-metal`. Note that ps5-opengl titles `freopen` stdout/stderr to `/download0/...` (`<run>/download0/`), so host errors appear there and not on the console.

## Licences and provenance

Each was verified against its primary source (repo `LICENSE`, release page, GitHub's asset digest) on 2026-10-13. The titles are test data run as guests and are never linked into our GPL-2.0-only code.

| Title | Official source | Asset | sha256 (matches GitHub's asset digest) | Licence | Redistribution / use |
|---|---|---|---|---|---|
| DOOM Shareware for PS5 | https://github.com/lowbit/ps5-doom/releases/tag/v1.1.1 | `PPSA99666.zip` | `4d8e5b785b61696fbf1988c2d0d87c5c977734ff60b1ce11cc49134b0e711b5f` | GPL-2.0 (`LICENSE`). `wads/doom1.wad` is the unmodified shareware IWAD v1.9 under id's shareware terms (free to share unmodified, free of charge; `THIRD_PARTY_NOTICES.md`). The bundled `sce_module/libc.prx` is GPL-3.0-or-later (ps5-native-app-boilerplate `b1315a9`). Also bundled: libarchive (BSD-2), liblzma (0BSD), zlib, qrcodegen (MIT); the texts are in `licenses/` | Yes |
| PS5 OpenGL Showcase | https://github.com/blackbearreloaded/ps5-opengl/releases/tag/v1.0.1 | `ps5-opengl-showcase-1.0.1-PPSA99005.zip` | `d16eb7f779d382d21e7f3374a3871e54cce3dea150a9e7de1603a9011833c9f9` | GPL-3.0-or-later (`LICENSE`). Mesa is mostly MIT (`LICENSES/Mesa`). The icon, backgrounds and music have no stated author: **art/music licence unverified** (presumed to fall under the repo licence) | Yes (code). Don't redistribute the art separately |
| Homebrew UI Lab | https://github.com/blackbearreloaded/ps5-homebrew-ui/releases/tag/01.000.010 (also in `blackbearreloaded/ps5-homebrew-catalog` `apps/PPSA99050.json`) | `PPSA99050.zip` | `420a3fb50ad6451c0d63f5e9a5b6ca3e8579a9e29b63f2eb187e740ce853cdaf` | GPL-3.0-or-later (`LICENSE`, `THIRD_PARTY_NOTICES.md`). Fonts: OFL-1.1 (Inter, Montserrat, Press Start 2P, Patrick Hand) and Bitstream Vera (DejaVu Sans Mono). stb_vorbis is public domain or MIT. Music and SFX are © BlackBearReloaded under the project licence | Yes |
| SuperTuxKart (Project Oops) | https://github.com/project-oops/oops-apps/releases/tag/latest-main, built from commit `c590060e1e501314afbd1def18ae68aadce7a86e` (a rolling pre-release whose assets get replaced; our digest is saved beside the download) | `supertuxkart-title-prospero.zip` | `09bbc9898633adf673e7866ac908c887990b4d8bea1a61f1fed6c9fac29dec49` | Port code: `LICENSE-APACHE` + `LICENSE-MIT` (a dual licence is inferred from the two files, not stated). STK 1.5 code is GPL-3.0-or-later (`COPYING`). Data (stk-assets-mobile 1.5) is mixed GPL/CC-BY/CC-BY-SA/PD with per-folder `licenses.txt` | Yes, with the per-asset attribution caveats. Never fetch that release's `pltauth-patch`/`sandbox-daemon` assets |

Excluded for licence or provenance reasons (DECISIONS D14): ccleste, Yamagi Quake II demo data, Quake3e, vkQuake, Unity-based titles, FSR4 Showcase, and anything that needs commercial data or ROMs.
