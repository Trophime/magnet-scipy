#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C8_pupitre_only \
    --config-file circuits-240717-174346.json \
    --value_start 0.0 0.0 \
    --time_start 0 --time_end 29867 --time_step 0.1 \
    --save_plots 240717-174346.png --show_analytics
