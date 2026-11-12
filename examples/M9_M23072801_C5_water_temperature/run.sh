#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C5_water_temperature \
    --config-file circuits-240417-144138.json \
    --value_start -5.3925 3.8745 \
    --time_start 0 --time_end 7590 --time_step 0.1 \
    --save_plots 240417-144138.png --show_analytics

python -m magnet_scipy.coupled_main_refactor \
    --wd examples/M9_M23072801_C5_water_temperature \
    --config-file circuits-240725-104414.json \
    --value_start -5.3766 4.1797 \
    --time_start 0 --time_end 48537 --time_step 0.1 \
    --save_plots 240725-104414.png --show_analytics
