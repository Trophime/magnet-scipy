#!/usr/bin/env bash
# Run the M9 PID solver sweep recorded in ../summary.md: each config x each solve_ivp method,
# full CLI run timed with /usr/bin/time (as in the M2 student's v0 report).
#
# Usage:  examples/M9_M23072801/benchmark/run_pid_sweep.sh <outdir> [jobs]
# Env:    CONFIGS   (default "circuits-pid circuits-pid-temp")
#         METHODS   (default "RK45 RK23 DOP853 Radau BDF LSODA")
#         MAGNET_SCIPY_CLI (default <repo>/.venv/bin/magnet-scipy-coupled)
#         RUN_TIMEOUT [s]  (default 7200; needs `timeout`, skipped if absent)
set -euo pipefail

OUT=${1:?usage: run_pid_sweep.sh <outdir> [jobs]}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
JOBS=${2:-6}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
CLI=${MAGNET_SCIPY_CLI:-$REPO/.venv/bin/magnet-scipy-coupled}
CONFIGS=${CONFIGS:-"circuits-pid circuits-pid-temp"}
METHODS=${METHODS:-"RK45 RK23 DOP853 Radau BDF LSODA"}
RUN_TIMEOUT=${RUN_TIMEOUT:-7200}

# GNU time on Linux (-v), BSD time on macOS (-l); build_summary.py parses both.
if [ "$(uname)" = "Darwin" ]; then TIME_FLAG=-l; else TIME_FLAG=-v; fi
TIMEOUT_CMD=""
command -v timeout >/dev/null 2>&1 && TIMEOUT_CMD="timeout $RUN_TIMEOUT"

run_one() {
    local cfg=$1 method=$2 tag="$1_$2" status
    cd "$REPO"
    set +e
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
        $TIMEOUT_CMD /usr/bin/time $TIME_FLAG "$CLI" \
        --wd examples/M9_M23072801 --config-file "$cfg.json" --strategy pid \
        --value_start -0.609178 0.236709 --time_start 0 --time_end 1100 --time_step 0.1 \
        --method "$method" --save_results "$OUT/$tag.npz" > "$OUT/$tag.log" 2>&1
    status=$?
    set -e
    echo "Exit status: $status" >> "$OUT/$tag.log"
    echo "$tag exit=$status"
}
export -f run_one
export REPO OUT CLI TIME_FLAG TIMEOUT_CMD

for cfg in $CONFIGS; do
    for m in $METHODS; do echo "$cfg $m"; done
done | xargs -P "$JOBS" -L 1 bash -c 'run_one "$0" "$1"'
