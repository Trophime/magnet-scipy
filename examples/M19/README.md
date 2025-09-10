# Running test cases

## Voltage mode for 2 circuits

```bash
python -m magnet_scipy.coupled_main \
  --wd examples/M19/ \
  --config-file 2circuits.json \
  --value_start 200 200 \
  --time_end 6000 --time_step 0.01 \
  --save_plots tyty-2circuits.png
```
## Voltage mode for single circuit

```bash
python -m magnet_scipy.main \
  --wd examples/M19/ \
  --config-file M9_M9Bitters_18MW.json \
  --value_start 200 \
  --time_end 6000 --time_step 0.01 \
  --save_plots tyty.png
```