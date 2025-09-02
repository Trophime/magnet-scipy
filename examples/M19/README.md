```bash
python -m magnet_scipy.coupled_main \
  --wd examples/M19/ \
  --config-file 2circuits.json \
  --value_start 200 200 \
  --time_end 6000 --time_step 0.01 \
  --experimental_csv M19_Overview_240208-0941_current.csv M19_Overview_240208-0941_current.csv
```