#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C6_drift \
    --config-file circuits-230914-214307.json \
    --value_start -6.4701 4.1241 \
    --time_start 0 --time_end 11848 --time_step 0.1 \
    --save_plots 230914-214307.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C6_drift \
    --config-file circuits-240726-112211.json \
    --value_start 4742.5249 1646.7797 \
    --time_start 0 --time_end 45852 --time_step 0.1 \
    --save_plots 240726-112211.png --show_analytics
