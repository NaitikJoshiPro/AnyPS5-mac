#!/bin/sh
# Runs GPU test groups (G0..G7 in tools/mac/gpu_groups.txt) one test per ctest process under the build lock, never
# the held-back or quarantined tests, and writes junit.xml + summary.txt/json with pass counts per group.
# D19 guards (DECISIONS.md): refuses (exit 3) with uncommitted changes under core/ or specifics/, stale test binaries
# or prx (ninja -n), load average > 6 or a running compiler/linker; runs the canaries (tools/mac/gpu_canaries.txt)
# first and stops on any failure; timeout <= 30 s per test; 2 s pause between tests; records HEAD + prx sha256.
# Example: tools/mac/gpu_suite.sh report            (G1..G4)
#          tools/mac/gpu_suite.sh G1 --validation   (MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1)
# Options: tools/mac/gpu_suite.sh --help
exec python3 -I "$(dirname "$0")/gpu_suite.py" "$@"
