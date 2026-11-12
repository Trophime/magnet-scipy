#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C3_eco_vs_normal \
    --config-file circuits-230927-145406.json \
    --value_start -5.4322 3.9428 \
    --time_start 0 --time_end 940 --time_step 0.01 \
    --save_plots 230927-145406.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C3_eco_vs_normal \
    --config-file circuits-231004-121211.json \
    --value_start -5.3922 3.9515 \
    --time_start 0 --time_end 1251 --time_step 0.01 \
    --save_plots 231004-121211.png --show_analytics
