#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C7_default_events \
    --config-file circuits-230919-190811.json \
    --value_start -6.4509 3.9472 \
    --time_start 0 --time_end 1842 --time_step 0.01 \
    --save_plots 230919-190811.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C7_default_events \
    --config-file circuits-240719-171041.json \
    --value_start -5.3493 3.9591 \
    --time_start 0 --time_end 21667 --time_step 0.1 \
    --save_plots 240719-171041.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C7_default_events \
    --config-file circuits-240716-210814.json \
    --value_start -5.3142 3.995 \
    --time_start 0 --time_end 10673 --time_step 0.1 \
    --save_plots 240716-210814.png --show_analytics
