# Benchmarking the M9 PID cases

How to regenerate [`summary.md`](summary.md) and `summary.csv`. These compare `circuits-pid.json` (constant 25 °C)
and `circuits-pid-temp.json` (measured inlet temperature) across the six `scipy.integrate.solve_ivp` methods. They cover:

- **cost:** wall time, CPU time, peak RAM, solution points;
- **electrical results per magnet:** tracking error, and current and voltage compared with the measurements;
- **outlet temperature:** a calorimetric estimate of Tout from `python_magnetcooling`, compared with the measured Tout.

Every run uses the same settings as the M2 student's v0 report, which `circuits-pid.json` + RK45 reproduces exactly:
the full CLI is timed with `/usr/bin/time`, over 0–1100 s, with `max_step` = 0.1 s, initial currents −0.609178 / 0.236709 A,
and the defaults rtol = 1e-6, atol = 1e-9.

## Prerequisites

1. **Data.** `examples/.gitignore` excludes `*.txt`, `*.tdms` and `*.csv`, so these files must be present in this
   directory:
   - pupitre record `M9_2023.09.16---17:55:22.txt` (Tin, Tout, flows, pressures);
   - pigbrother `M9_Overview_230916-1755.tdms` and its exported `M9_Overview_230916-1755_*_vs_Time.csv`;
   - the reference, resistance and temperature CSVs used by the two configs.
2. **magnet-scipy**, installed in the repo's `.venv` (`uv sync`). The runner calls `.venv/bin/magnet-scipy-coupled`;
   set `MAGNET_SCIPY_CLI` to use another executable.
3. **Analysis environment**, needed only for the summary and the clock-offset steps:
   ```bash
   uv venv ~/venvs/m9-analysis
   VIRTUAL_ENV=~/venvs/m9-analysis uv pip install -e . -e ../python_magnetcooling nptdms
   ```
   `python_magnetcooling` provides the Tout calculation; it pulls in `iapws`, `ht` and `pint`.
   As of 2026-11-12, its `main` branch fails to import
   (see `python_magnetcooling/prompts/fix_import_regressions.md`). Until that is fixed, point
   `build_summary.py` at the source directory and it loads the module it needs directly:
   ```bash
   export PYTHON_MAGNETCOOLING_SRC=$PWD/../python_magnetcooling/python_magnetcooling
   ```

## Running

All commands are run from the repository root.

### 1. Simulations

```bash
examples/M9_M23072801/benchmark/run_pid_sweep.sh /path/to/sweep 6     # output dir, parallel jobs
```

Each run writes `<config>_<method>.npz` (results) and `<config>_<method>.log` (CLI output plus `/usr/bin/time`) to the output directory.

| Solvers | Time per run | Peak RAM per run | Size of `.npz` |
|---|---|---|---|
| Radau, BDF, LSODA | 5–25 s | ~175 MiB | ~1 MB |
| RK45, RK23, DOP853 | 30–50 min (stability-limited) | 1.1–2.2 GiB | 75–220 MB |

With 6 jobs in parallel, the whole sweep of 12 runs takes about 50 min, needs about 11 GiB of RAM at peak and about
1 GB of disk.

- **Quick check:** `METHODS="Radau BDF LSODA" examples/M9_M23072801/benchmark/run_pid_sweep.sh /path/to/sweep`.
- **One config only:** `CONFIGS=circuits-pid`.
- **Timeout:** each run is capped at `RUN_TIMEOUT` seconds (default 7200) when `timeout` is available. On macOS
  without GNU coreutils there is no cap.
- **Timing format:** Linux uses GNU `time -v`, macOS uses BSD `time -l`; `build_summary.py` reads both.

### 2. Clock offset between the data sources (optional)

```bash
~/venvs/m9-analysis/bin/python examples/M9_M23072801/benchmark/lag_sources.py
```

The pupitre and pigbrother files each start their own t = 0. In the exported CSVs, pigbrother time t matches pupitre
time t + **8.4 s**:
- the timestamps predict 8.91 s (`.tdms` starts at 17:55:32.41 local; the pupitre's first row is 17:55:23; the CSV
  export adds 0.5 s);
- cross-correlating the four signals both systems record (I_H, I_B, U_H, U_B) gives 8.06–8.99 s.

Only the Tout comparison mixes the two sources, so only it uses this offset. The value goes to `--pig-shift` in step 3.

### 3. Summary tables

```bash
~/venvs/m9-analysis/bin/python examples/M9_M23072801/benchmark/build_summary.py /path/to/sweep \
    [--output DIR] [--pig-shift 8.4] [--no-plots]
```

This writes `summary.md` and `summary.csv` to `--output` (default: the sweep directory). A missing or failed run
shows as `—`. Unless `--no-plots` is given, it also writes:

- `summary.png`: the current and voltage RMSE vs measurements for each magnet, and the Tout RMSE (no lag, and after
  the lag fit), against the wall time of the CLI run. Colour and marker show the method; filled markers are
  `circuits-pid`, hollow ones `circuits-pid-temp`. The dashed line is the Tout error of the measured-U·I baseline.
- `plots/<config>_<method>.png`: each run vs the measurements, one column per magnet: current with its reference,
  current error, and controller output vs measured voltage.

 To update the recorded files, use `--output examples/M9_M23072801`, then restore the provenance
header at the top of `summary.md`.

## Reading the results

- **Wall time depends on the load.** With parallel jobs, wall time is inflated by contention: RK45 took 2716 s CPU
  with 6 jobs, but 1631 s alone. CPU time and peak RAM are comparable between runs; run with `jobs = 1` for clean wall
  times.
- **Tracking RMSE/MAE are averaged over solver output points**, as the package computes them, so they depend on each
  solver's step distribution (0.82–0.89 A for the helix). Weighted by time, they agree within 0.2 % (0.855 A). See
  `prompts/time_weighted_metrics.md`.
- **Voltage is sensitive to numerical noise.** The modelled voltage is the controller output Kp·e + Ki·∫e, and Kp = 12
  amplifies 0.02 A of numerical noise into about 0.25 V. RK45 is the outlier here (3.46 V against 3.71 V for the five
  other solvers on the helix).
- **Tout fit:** Tout = Tin_mix + k·F_τ[cTout − Tin_mix](t − L), where F_τ is a first-order low-pass filter.
  `measured U·I` is a baseline that uses measured power instead of a simulation. The Tout rows are the same for all
  solvers.

## Solver-only benchmarks

To compare solvers rather than reproduce the student's setup, use [`benchmarks/bench_solvers.py`](../../benchmarks/README.md).
It times only the solve (no CSV loading, no plotting), counts RHS evaluations and LU decompositions, sweeps rtol, and
separates numerical error from model error with a tight-tolerance reference solution:

```bash
python benchmarks/bench_solvers.py --cases M9_coupled_pid M9_coupled_pid_temp --methods Radau BDF LSODA \
    --rtol 1e-4 1e-6 1e-8 --repeat 3
```

Its M9 cases (`benchmarks/cases.json`) don't use the settings above. They run from 5 to 1200 s with `max_step` = 0.01 s,
starting from the measured currents at t = 5 s. It does not compute Tout.
