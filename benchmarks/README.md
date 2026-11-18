# ODE solver benchmarks

`bench_solvers.py` runs the `examples/` cases listed in `cases.json` with each
`scipy.integrate.solve_ivp` method and reports cost, memory and precision.

The experimental CSV files in `examples/` are git-ignored, so the benchmark only
runs on a checkout where they are present.

## Running

```bash
# quick pass: every case/method, first 300 s of each shot
python benchmarks/bench_solvers.py --max-duration 300

# work-precision sweep on the stiff PID case, implicit solvers only
python benchmarks/bench_solvers.py --cases M9_coupled_pid \
    --methods Radau BDF LSODA --rtol 1e-4 1e-6 1e-8 --repeat 3

# remove the max_step cap to compare the solvers' own step-size control
python benchmarks/bench_solvers.py --max-step inf --max-duration 600
```

`--timeout` (default 1800 s) kills a run that takes too long. On full shots with
PID control the explicit methods (RK45, RK23, DOP853) are stability-limited and
take tens of minutes.

## What is measured

Each run is executed in a freshly spawned process, so timings and peak memory
do not depend on earlier runs. Only `runner.run_simulation()` is timed; loading
the CSV files and building the interpolators are not.

`performance.csv`, one row per (case, method, rtol):

| column | meaning |
|---|---|
| `wall_s_median`, `wall_s_min`, `cpu_s_median` | solve time over `--repeat` runs [s] |
| `nfev`, `njev`, `nlu` | RHS evaluations, Jacobian evaluations, LU decompositions |
| `n_steps` | accepted steps |
| `rss_delta_mb` | peak RSS growth during the solve [MB] |
| `tracemalloc_peak_mb` | Python-level allocation peak [MB], with `--tracemalloc` only (extra, slower run) |

`accuracy.csv` has one row per (case, method, rtol, circuit, quantity). The
quantity is the current, plus the voltage in PID mode, where the controller
output `Kp e + Ki ∫e + Kd ė` is the modelled supply voltage. The simulation is
sampled at the experimental time stamps inside the simulated window.

| column | meaning |
|---|---|
| `rmse_exp`, `mae_exp`, `max_abs_error_exp`, `bias_exp` | vs measurements [A] or [V] |
| `nrmse_exp` | `rmse_exp / max(abs(measured))` |
| `rmse_ref`, `max_abs_error_ref` | vs reference solution [A] or [V] |

### Why two precision measures

The error vs measurements combines **model error** (R(I, T) table, L, M,
inlet temperature, the time alignment of the data) and **numerical error**. Once
a solver has converged, the model error dominates by several orders of
magnitude: every method then gives the same `rmse_exp`, which tells you nothing
about the solver.

The reference solution (default Radau, `rtol = atol = 1e-10`, with the case's
own `max_step`) separates the two. `*_ref` is the solver's numerical error
alone, and `*_exp` is the model's quality. A solver is good enough when its
`rmse_ref` is negligible compared with `rmse_exp`.

References are cached in `results/references/`, keyed on the case definition,
the JSON config and the reference settings. Delete that directory after
changing a CSV input.

### Plots

Unless `--no-plots` is given, the benchmark also writes PNG files:

| file | content |
|---|---|
| `work_precision.png` | one panel per case: wall time vs current `rmse_ref` (worst circuit), one line per method through its rtol values; the dashed line is the median `rmse_exp` |
| `plots/<case>_<method>_rtol<r>.png` | results per method: current (simulated, measured, reference current in PID mode), current error, and voltage (controller output vs measured in PID mode, input voltage in voltage mode), one column per circuit |

Each successful run is saved to `runs/<case>_<method>_rtol<r>.npz` (solution
thinned to 20 000 points, plus the samples on the experimental time grid).
`python benchmarks/plot_results.py <output-dir>` redraws every plot from these
files and the CSVs without solving again. `work_precision.png` needs the
reference solution, so it is skipped with `--no-reference`.

`environment.json` records the CPU, the Python/NumPy/SciPy versions and the
command-line options. BLAS is pinned to one thread.

## M23072801 test campaign

`bench_campaign.py` runs the same benchmark on every run of the
`examples/M9_M23072801_C*` campaign written by
`examples/build_M23072801_campaign.py`. It builds one case per run from each
group's `provenance.json` (named `<group>_<tag>`, e.g. `C0_230916-175522`, with
the run's config, initial currents, window and max step) and writes them to
`<output-dir>/cases.json`. It then runs `bench_solvers.py` on them and adds a
`summary.md` with one row per run and one column per method: wall time, steps,
and current RMSE vs measurements and vs the reference solution. On top of the
`bench_solvers.py` plots, `campaign_rtol<r>.png` shows the wall time and current
RMSE vs measurements of every run, one marker per method, grouped by C-group.
Runs that did not finish are marked ✕. `summary.md` needs `tabulate`, which
`uv sync` installs with the dev group.

```bash
python benchmarks/bench_campaign.py --list                       # runs and their windows
python benchmarks/bench_campaign.py --groups C0 C1 C2 C3         # short runs (< 1 h)
python benchmarks/bench_campaign.py --max-t-end 2000 --rtol 1e-4 1e-6 1e-8
python benchmarks/bench_campaign.py --runs 230916-175522 --methods Radau BDF LSODA RK45
```

Only Radau, BDF and LSODA run by default. The C4–C9 runs cover 3–13 h of data
at `max_step` = 0.1 s, which the explicit methods cannot solve within the
timeout. `--groups`, `--runs` and `--max-t-end` select the runs. Any other option
is passed on to `bench_solvers.py`. Results go to `results/campaign/` by default.
On the long runs the reference solution (Radau, rtol = 1e-10) can take longer
than `--timeout`. Increase the timeout or use `--no-reference`.

## Adding a case

Add an entry to `cases.json`:

```json
"my_case": {
  "wd": "examples/M19",
  "config": "M9_M9Bitters_18MW.json",
  "kind": "single",
  "initial_values": [200.0],
  "t_start": 0.0,
  "t_end": 6000.0,
  "max_step": 0.01
}
```

`kind` is `single` (single-circuit JSON) or `coupled` (JSON with a
`circuits` list). The voltage or PID strategy is auto-detected as in the CLI.
Only circuits with `experiment_data` entries contribute to `accuracy.csv`.
