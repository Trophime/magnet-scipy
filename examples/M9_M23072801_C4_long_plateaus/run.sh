#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C4_long_plateaus \
    --config-file circuits-240510-144107.json \
    --value_start -5.3341 4.1099 \
    --time_start 0 --time_end 35754 --time_step 0.1 \
    --save_plots 240510-144107.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C4_long_plateaus \
    --config-file circuits-240715-090712.json \
    --value_start -5.2207 3.9094 \
    --time_start 0 --time_end 29335 --time_step 0.1 \
    --save_plots 240715-090712.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C4_long_plateaus \
    --config-file circuits-230915-200701.json \
    --value_start -6.4147 3.8852 \
    --time_start 0 --time_end 41825 --time_step 0.1 \
    --save_plots 230915-200701.png --show_analytics
