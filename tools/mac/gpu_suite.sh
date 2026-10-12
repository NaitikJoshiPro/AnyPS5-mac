#!/bin/sh
# Runs GPU test groups (G0..G7 in tools/mac/gpu_groups.txt) one test at a time under tools/mac/buildlock.py,
# never the held-back tests, and writes junit.xml + summary.txt with pass counts per group.
# Example: tools/mac/gpu_suite.sh report            (G1..G4)
#          tools/mac/gpu_suite.sh G1 --validation   (Metal shader validation on)
# Options: tools/mac/gpu_suite.sh --help
exec python3 -I "$(dirname "$0")/gpu_suite.py" "$@"
