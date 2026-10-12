#!/bin/sh
# Measures disk used by the native-Mac port against the storage budget.
# Exit status: 0 = under the warning line, 1 = over the warning line, 2 = over the budget.
# Usage: tools/mac/disk_budget.sh [budget_gb] [warn_gb]
BUDGET_GB=${1:-100}
WARN_GB=${2:-85}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

paths="$ROOT
$HOME/Library/Caches/Homebrew
$HOME/Library/Caches/ccache
$HOME/.cache/ccache
$HOME/.ccache
$HOME/Library/Developer/Xcode/DerivedData
$HOME/Library/Caches/pip
$HOME/.cache/pip
$HOME/.cache/huggingface
/private/tmp/claude-$(id -u)"

total_kb=0
printf '%-60s %10s\n' "PATH" "SIZE"
echo "$paths" | while IFS= read -r p; do
  [ -e "$p" ] || continue
  kb=$(du -sk "$p" 2>/dev/null | awk '{print $1}')
  printf '%-60s %8.2fG\n' "$p" "$(echo "$kb / 1048576" | bc -l)"
done
total_kb=$(echo "$paths" | while IFS= read -r p; do [ -e "$p" ] && du -sk "$p" 2>/dev/null; done | awk '{s+=$1} END {print s+0}')
total_gb=$(echo "$total_kb / 1048576" | bc -l)
free_gb=$(df -k / | awk 'NR==2 {printf "%.1f", $4/1048576}')
printf '\nTOTAL used by project: %.2fG of %sG budget (warn at %sG). Disk free: %sG\n' "$total_gb" "$BUDGET_GB" "$WARN_GB" "$free_gb"

if [ "$(echo "$total_gb > $BUDGET_GB" | bc -l)" = 1 ]; then echo "OVER BUDGET: stop and clean up, or ask the user for an exception."; exit 2; fi
if [ "$(echo "$total_gb > $WARN_GB" | bc -l)" = 1 ]; then echo "WARNING: above warning line, clean up before continuing."; exit 1; fi
exit 0
