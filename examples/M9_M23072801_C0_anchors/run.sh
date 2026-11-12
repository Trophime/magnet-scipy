#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C0_anchors \
    --config-file circuits-230916-175522.json \
    --value_start -6.3259 4.0381 \
    --time_start 0 --time_end 1554 --time_step 0.01 \
    --save_plots 230916-175522.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C0_anchors \
    --config-file circuits-231017-122455.json \
    --value_start -5.4261 3.9736 \
    --time_start 0 --time_end 3965 --time_step 0.1 \
    --save_plots 231017-122455.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C0_anchors \
    --config-file circuits-240405-152426.json \
    --value_start -4.9798 3.7376 \
    --time_start 0 --time_end 1393 --time_step 0.01 \
    --save_plots 240405-152426.png --show_analytics
