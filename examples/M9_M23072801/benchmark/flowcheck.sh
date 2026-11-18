#!/usr/bin/env bash
# Compare the water flow params fits (<magnet>-flow_params.json) with the measured hydraulics
# of the M9_M23072801 campaign runs: pump speed, flow, HP and BP vs current, per operating mode.
# Uses the viewer's flowcheck view (magnet_scipy.viewer): prints the RMS table per run, circuit
# and mode, and draws one figure per magnet.
#
# Usage:  examples/M9_M23072801/benchmark/flowcheck.sh [selector ...]
#   selector  campaign group (C0, C4, ...) or run tag (230916-175522, ...); default: every run
# Env:    THRESHOLD [A] ignore |I| below (default 300)
#         SAVE      write <stem>_<magnet>.png instead of opening windows (e.g. SAVE=flowcheck.png)
#         PYTHON    (default <repo>/.venv/bin/python)
#
# Needs the <tag>_hydraulics_GR*.csv written by examples/build_M23072801_campaign.py.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
THRESHOLD=${THRESHOLD:-300}

shopt -s nullglob
CONFIGS=()
declare -A SEEN=()
for cfg in "$REPO"/examples/M9_M23072801_C*/circuit*.json; do
    group=$(basename "$(dirname "$cfg")" | cut -d_ -f3)  # M9_M23072801_C4_long_plateaus -> C4
    tag=$(basename "$cfg" .json | grep -oE '[0-9]{6}-[0-9]{6}$' || true)
    if [ $# -gt 0 ]; then
        match=0
        for sel in "$@"; do
            [ "$sel" = "$group" ] || [ "$sel" = "$tag" ] && match=1
        done
        [ $match -eq 1 ] || continue
    fi
    # A run listed in two groups would be counted twice.
    [ -n "${SEEN[$tag]:-}" ] && continue
    SEEN[$tag]=1
    CONFIGS+=("$cfg")
done
[ ${#CONFIGS[@]} -gt 0 ] || { echo "no campaign config matches: $*" >&2; exit 1; }
echo "${#CONFIGS[@]} runs: $(printf '%s\n' "${!SEEN[@]}" | sort | tr '\n' ' ')"

VIEWER=("$PYTHON" -m magnet_scipy.viewer)
[ -n "${SAVE:-}" ] && VIEWER+=(--save "$SAVE")
exec "${VIEWER[@]}" flowcheck --threshold "$THRESHOLD" "${CONFIGS[@]}"
