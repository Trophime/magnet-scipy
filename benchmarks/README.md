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

`environment.json` records the CPU, the Python/NumPy/SciPy versions and the
command-line options. BLAS is pinned to one thread.

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
