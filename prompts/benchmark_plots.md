# PNG plots for the solver benchmarks

Status: implemented (2026-11-16). `build_summary.py` plots were tested with a
stand-in for `python_magnetcooling`; still to run in the real analysis env.

## Goal

Every benchmark writes PNG figures next to its CSV / markdown output, by default
(`--no-plots` to skip). Plotting runs after the CSVs are written, so a plotting
error never loses benchmark results.

## Plots

| Benchmark | File | Content |
|---|---|---|
| `bench_solvers.py` | `work_precision.png` | one panel per case: wall time vs current `rmse_ref`, one line per method across rtol; `rmse_exp` (model error) as a vertical band |
| `bench_solvers.py` / `bench_campaign.py` | `plots/<case>_<method>_rtol<r>.png` | results per method: layout of `examples/plot_lsoda.py` (current + reference + measured, current error, voltage) |
| `bench_campaign.py` | `campaign.png` | wall time and current `rmse_exp` per run, bars per method, runs grouped by C-group |
| M9 PID sweep (`build_summary.py`) | `summary.png` | wall time vs current RMSE / voltage RMSE per magnet and Tout RMSE, colour = method, marker = config |
| M9 PID sweep (`build_summary.py`) | `plots/<config>_<method>.png` | results per method, from the `.npz` files |

## Implementation

1. `benchmarks/plot_results.py` (new): shared plotting functions; standalone
   CLI `python benchmarks/plot_results.py <results dir>` regenerates the plots
   of a `bench_solvers` / `bench_campaign` output directory.
2. `bench_solvers.py`: the worker also returns a thinned trajectory
   (≤ 20 000 points: time, currents, reference current in PID mode, input
   voltage in voltage mode). The trajectory and the experiment samples are saved
   to `<output>/runs/<case>_<method>_rtol<r>.npz`, so plots can be regenerated
   without solving again.
3. `bench_campaign.py`: `campaign.png` after `summary.md`.
4. `build_summary.py`: `summary.png` and per-method plots.
5. README / BENCHMARK.md: list the PNG outputs.

`examples/plot_lsoda.py` becomes redundant with the per-method plots
(LSODA is one of the methods); left in place, to be removed by the user if wanted.
