# AnyPS5 for Apple Silicon (native arm64 + Metal)

This repository is a port of [AnyPS5](https://github.com/boykopovar/AnyPS5) that runs natively on Apple Silicon Macs: arm64 only, Metal only, no Rosetta, no MoltenVK. Plan and status: [mission](docs/mac/MISSION.md), [progress log](docs/mac/PROGRESS.md), [decisions](docs/mac/DECISIONS.md), [compatibility](docs/mac/COMPATIBILITY.md).

[![port milestones](https://naitikjoshipro.github.io/AnyPS5-mac/badge-milestones.svg)](https://naitikjoshipro.github.io/AnyPS5-mac/) [![Metal GPU tests](https://naitikjoshipro.github.io/AnyPS5-mac/badge-metal-tests.svg)](https://naitikjoshipro.github.io/AnyPS5-mac/#metal-gpu-tests) [![titles](https://naitikjoshipro.github.io/AnyPS5-mac/badge-titles.svg)](https://naitikjoshipro.github.io/AnyPS5-mac/#titles) [![libraries](https://naitikjoshipro.github.io/AnyPS5-mac/badge-libraries.svg)](https://naitikjoshipro.github.io/AnyPS5-mac/progress.html) [![shaders](https://naitikjoshipro.github.io/AnyPS5-mac/badge-shaders.svg)](https://naitikjoshipro.github.io/AnyPS5-mac/progress.html)

<sub>Computed on every push to `mac-native` by [tools/mac/mac_progress.py](tools/mac/mac_progress.py) ([progress page](https://naitikjoshipro.github.io/AnyPS5-mac/)).
**Port milestones:** share of the [MISSION §8](docs/mac/MISSION.md) exit criteria (M0–M4) verified, from [docs/mac/milestones.toml](docs/mac/milestones.toml); each milestone weighs the same, partly done criteria count as 0.
**Metal GPU tests:** GPU tests that passed in the last full `tools/mac/gpu_suite.py` run, out of all registered GPU tests; allowlisted skips (Vulkan-only features) and held-back tests count as not passed. Snapshot in [docs/mac/gpu-status.json](docs/mac/gpu-status.json), refreshed on the Mac.
**Titles:** average stage of the titles in [COMPATIBILITY.md](docs/mac/COMPATIBILITY.md) (Nothing 0, Loads 20, Intro 40, Menu 60, In game 80, Playable 100).
**Libraries / shaders:** upstream's metrics (functions implemented in core/libs/prx, RDNA instructions decoded) measured on this tree; they describe the shared code, not whether it works on the Mac.</sub>

The upstream README follows.

# About

Tool for automatic executables porting to Linux and Windows.

The relinker-only build runs on macOS, including Apple silicon.

[**Web converter**](https://boykopovar.github.io/AnyPS5/) - files stay on your device.

![anyps5_collage1](https://github.com/user-attachments/assets/ed2f0b2d-33d3-46d7-ab88-767ca872fc99)

Includes a [relinker](core/relinker) that converts executable to the target system's native format and implementations of [system prx libraries](core/libs/prx) suitable for dynamic linking. No emulation or separate runtime process.

[**Discord**](https://discord.gg/BHFztBPUe), [Usage](docs/user/USAGE.md), [Build instructions](docs/dev/BUILD.md), [Architecture](docs/dev/ARCHITECTURE.md), [Technical debt of the project](docs/dev/TechnicalDebt.md), [code style conventions](docs/dev/CONVENTIONS.md), [contributing](CONTRIBUTING.md)

## Status

[![libraries](https://boykopovar.github.io/AnyPS5/badge-libraries.svg)](https://boykopovar.github.io/AnyPS5/progress.html) [![shaders](https://boykopovar.github.io/AnyPS5/badge-shaders.svg)](https://boykopovar.github.io/AnyPS5/progress.html)

[![progress map](https://boykopovar.github.io/AnyPS5/progress.svg)](https://boykopovar.github.io/AnyPS5/progress.html)

[Progress history](https://boykopovar.github.io/AnyPS5/history.html)

<sub>* System libraries: percentage of the functions known to the project so far (declared in [core/libs/prx](core/libs/prx)), not of every PS5 system function. The total grows as more functions are declared.</sub>

Unsupported or unexpected states strictly throw `std::runtime_error`. `what()` is printed to stderr and the process terminates.

The [shader recompiler](core/shader/recompiler/Recompiler.cpp) successfully produces SPIR-V (validated via [Spirv-Tools](3rdparty/SPIRV-Tools) when built with `ANYPS5_ENABLE_SPIRV_TOOLS`).

## Compatibility

Per-game progress (stage reached, open blockers, libraries and GPU coverage) is tracked on the [blocker map](https://alxcbtw.github.io/anyps5-roadmap/), a community project built from public GitHub data ([source](https://github.com/AlxcBTW/anyps5-roadmap)). Test reports go in the [Game bring-up](https://github.com/boykopovar/AnyPS5/discussions/categories/game-bring-up) discussions, one thread per title.

## Input mapping

SDL-mapped game controllers are supported, including analog sticks and triggers. Keyboard and mouse controls can be configured with an `anyps5-input.ini` file. See [input mapping](docs/user/INPUT_MAPPING.md) for the supported devices and configuration format.

## Disclaimer

This project is intended for interoperability, research, preservation, and compatibility purposes. It does not include, distribute, or require copyrighted software, firmware, cryptographic keys, or proprietary libraries. Users are responsible for ensuring that any binaries used with this project are obtained and used in accordance with applicable laws and their respective license terms.

AnyPS5 is an independent project and is not affiliated with, authorized or endorsed by Sony Interactive Entertainment. "PlayStation" and "PS5" are trademarks of Sony Interactive Entertainment Inc.

## License

This project is licensed under the GNU General Public License version 2 only.
