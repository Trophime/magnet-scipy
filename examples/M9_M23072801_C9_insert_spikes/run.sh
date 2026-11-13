#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C9_insert_spikes \
    --config-file circuits-240804-091254.json \
    --value_start -5.2046 4.0109 \
    --time_start 0 --time_end 14873 --time_step 0.1 \
    --save_plots 240804-091254.png --show_analytics
