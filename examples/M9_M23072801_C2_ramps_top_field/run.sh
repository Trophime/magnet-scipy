#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C2_ramps_top_field \
    --config-file circuits-230911-142159.json \
    --value_start -6.4807 4.0701 \
    --time_start 0 --time_end 857 --time_step 0.01 \
    --save_plots 230911-142159.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C2_ramps_top_field \
    --config-file circuits-240416-152804.json \
    --value_start -5.3675 3.7339 \
    --time_start 0 --time_end 1282 --time_step 0.01 \
    --save_plots 240416-152804.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C2_ramps_top_field \
    --config-file circuits-240716-114136.json \
    --value_start -5.2671 3.9016 \
    --time_start 0 --time_end 917 --time_step 0.01 \
    --save_plots 240716-114136.png --show_analytics
