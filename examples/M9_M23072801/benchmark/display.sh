#!/usr/bin/env bash
# View the M9 PID sweep results with the interactive viewer (magnet_scipy.viewer).
#
# Usage:  examples/M9_M23072801/benchmark/display.sh [view] [sweep_dir]
#   view       overview  (default) one run: currents, voltages, power, temperatures vs measurements
#              residuals one run: simulated - measured per circuit
#              compare   every method of CONFIG (sweep) or every recorded config, one quantity
#   sweep_dir  directory written by run_pid_sweep.sh (<config>_<method>.npz);
#              default: the .npz recorded in examples/M9_M23072801 (<config>.npz)
# Env:    CONFIG   (default circuits-pid)
#         METHOD   (default LSODA; sweep only)
#         CIRCUIT  (default M23072801; compare only, or M9Bitters_18MW)
#         QUANTITY (default current; compare only)
#         SAVE     write the figure to this PNG instead of opening a window
#         PYTHON   (default <repo>/.venv/bin/python)
set -euo pipefail

VIEW=${1:-overview}
SWEEP=${2:-}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
WD=$REPO/examples/M9_M23072801
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CONFIG=${CONFIG:-circuits-pid}
METHOD=${METHOD:-LSODA}
CIRCUIT=${CIRCUIT:-M23072801}
QUANTITY=${QUANTITY:-current}

VIEWER=("$PYTHON" -m magnet_scipy.viewer)
[ -n "${SAVE:-}" ] && VIEWER+=(--save "$SAVE")

if [ -n "$SWEEP" ]; then
    SWEEP=$(cd "$SWEEP" && pwd)
    NPZ=$SWEEP/${CONFIG}_${METHOD}.npz
else
    NPZ=$WD/$CONFIG.npz
fi

case "$VIEW" in
    overview|residuals)
        [ -f "$NPZ" ] || { echo "missing $NPZ" >&2; exit 1; }
        exec "${VIEWER[@]}" "$VIEW" "$NPZ" --config "$WD/$CONFIG.json" --wd "$WD"
        ;;
    compare)
        if [ -n "$SWEEP" ]; then
            shopt -s nullglob
            FILES=("$SWEEP/${CONFIG}"_*.npz)
        else
            FILES=("$WD"/circuits*.npz)
        fi
        [ ${#FILES[@]} -gt 0 ] || { echo "no .npz to compare" >&2; exit 1; }
        exec "${VIEWER[@]}" compare -c "$CIRCUIT" -q "$QUANTITY" "${FILES[@]}"
        ;;
    *)
        echo "unknown view '$VIEW' (overview, residuals, compare)" >&2
        exit 2
        ;;
esac
