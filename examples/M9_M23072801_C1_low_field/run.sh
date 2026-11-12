#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.main_refactor \
    --wd examples/M9_M23072801_C1_low_field \
    --config-file circuit-GR2-230906-111551.json \
    --value_start 4.0815 \
    --time_start 0 --time_end 3193 --time_step 0.01 \
    --save_plots 230906-111551.png --show_analytics

python -m magnet_scipy.main_refactor \
    --wd examples/M9_M23072801_C1_low_field \
    --config-file circuit-GR2-230906-161112.json \
    --value_start 3.9228 \
    --time_start 0 --time_end 1856 --time_step 0.01 \
    --save_plots 230906-161112.png --show_analytics

python -m magnet_scipy.main_refactor \
    --wd examples/M9_M23072801_C1_low_field \
    --config-file circuit-GR2-231012-163131.json \
    --value_start 3.7767 \
    --time_start 0 --time_end 1041 --time_step 0.01 \
    --save_plots 231012-163131.png --show_analytics

python -m magnet_scipy.main_refactor \
    --wd examples/M9_M23072801_C1_low_field \
    --config-file circuit-GR2-231013-090926.json \
    --value_start 5005.0674 \
    --time_start 0 --time_end 573 --time_step 0.01 \
    --save_plots 231013-090926.png --show_analytics
