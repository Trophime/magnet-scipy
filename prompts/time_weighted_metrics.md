# Time-weighted analytics metrics

Status: plan, not implemented. Drafted 2026-11-12.

## Problem

The analytics in `magnet_scipy/plotting_core.py` average over **solver output points** `t_k`, giving each point equal weight:

$$\mathrm{RMSE}_{\text{points}} = \sqrt{\tfrac{1}{N}\textstyle\sum_k e(t_k)^2}$$

Adaptive solvers space `t_k` unevenly, and each solver does it differently. The metrics are therefore properties of the
step distribution, not of the solution. The physically meaningful quantity is the time average:

$$\mathrm{RMSE}_{\text{time}} = \sqrt{\tfrac{1}{T}\int_{t_0}^{t_1} e(t)^2\,dt}, \qquad T = t_1 - t_0$$

### Evidence: M9 PID sweep (`examples/M9_M23072801/summary.md`)

| Tracking RMSE, H / B [A] | Averaged over points (6 solvers) | Weighted by time (6 solvers) |
|---|---|---|
| `circuits-pid` | 0.823–0.887 / 0.335–0.359 | 0.8546–0.8564 / 0.3458–0.3463 |
| `circuits-pid-temp` | 0.820–0.882 / 0.333–0.357 | 0.8497–0.8515 / 0.3436–0.3441 |

The spread between solvers drops from about 7.5 % to about 0.2 %. Resampling on a uniform 1 s grid agrees with the
trapezoidal integral. The mechanism: the reference ramps during 60.3 % of the run. RK45, RK23 and DOP853 put 64.2 % of
their points there, where the error is largest, so they overstate the RMSE. Radau, BDF and LSODA step almost uniformly at
`max_step`, and their few extra steps fall at kinks, so they understate it. The student's v0 value of 0.8861 A for
RK45 is 0.855 A weighted by time.

## Affected code

| Location | Quantity | Current (point-weighted) | Fix |
|---|---|---|---|
| `DataProcessor.compute_error_metrics` | `mae`, `mse`, `rmse`, `bias` | `np.mean(...)` over points | time average |
| `DataProcessor.compute_error_metrics` | `max_error` | `max(abs(e))` | unchanged (no weighting) |
| `DataProcessor.compute_statistics` | `mean`, `std`, `rms` | over points | time average |
| `DataProcessor.compute_statistics` | `min`, `max`, `range` | | unchanged |
| `PlottingAnalytics._analyze_region_usage` | % of run in each PID region | counts points | % of **time** in each region |
| `PlottingAnalytics.print_detailed_analytics` | "Total Energy (approx)" | `mean(P) * 1`, comment "Assuming 1s simulation" (wrong for any real run) | `E = ∫P dt` |

The only caller is `PlottingAnalytics.analyze_circuit_performance`, which already has `results.time`.
No test covers these functions today.

Out of scope (separate issue): `utils.exp_metrics` resamples onto 200 evenly spaced points over the overlap. That is
already weighted by time, but coarsely. Its window-labelling and no-overlap crash are tracked separately.

## Design

1. One private helper in `plotting_core.py`:
   ```python
   def _time_mean(y: np.ndarray, t: np.ndarray) -> float:
       """Time average of y over t using the trapezoidal rule."""
   ```
   Use `scipy.integrate.trapezoid`, not `np.trapezoid`: `pyproject.toml` allows `numpy>=1.21`, while
   `np.trapezoid` needs NumPy ≥ 2.0 and `np.trapz` is deprecated there. If `T == 0` or `len(t) < 2`, fall back to the plain mean.
2. Add an optional `time: np.ndarray | None = None` keyword to `compute_error_metrics` and `compute_statistics`.
   - `time is None`: current behaviour, unchanged, so external callers and saved-result comparisons keep working.
   - `time` given: time-weighted `mean`, `std = sqrt(<(y - <y>)^2>)`, `rms`, `mae`, `mse`, `rmse` and `bias`.
   This is not a rename, so no deprecation shim is needed (CLAUDE.md).
3. `_analyze_region_usage(regions, time=None)`: when `time` is given, weight each sample by its trapezoid half-interval
   `w_k = (t_{k+1} - t_{k-1}) / 2` (half-intervals at the ends), so the weights sum to T.
4. `analyze_circuit_performance` passes `time=results.time` everywhere.
   Add `'energy': trapezoid(P, t)` to `power_stats`, print `Total Energy: … MJ`, and delete the `* 1` approximation.
5. NumPy-style docstrings with units on every touched function. Say explicitly that metrics are time-weighted when `time` is given.

## Steps

1. **Tests first** (`tests/test_time_weighted_metrics.py`) → verify: they fail on the current code.
   - *Sampling invariance*: take an analytic e(t), e.g. a ramp plus a sinusoid, on [0, 10] s. Sample it on a uniform grid
     and on a strongly clustered grid (90 % of points in [0, 1] s). With `time=` given, `rmse`, `mae`, `mean` and `rms`
     agree within 1e-3 between the two grids and match the closed-form values. With `time=None`, the old point-weighted
     values are unchanged (regression).
   - *Region usage*: two regions, 30 % / 70 % of the time, with the point density inverted. The time-weighted shares
     return 30 / 70.
   - *Energy*: constant P = 2 MW over 5 s gives E = 10 MJ, for any grid.
2. Implement the helper and the keyword arguments → verify: the new tests pass and the existing suite stays green.
3. Wire up `analyze_circuit_performance` and the printout → verify on the M9 sweep that the six solvers'
   tracking RMSE agree within 0.3 % (expected 0.855 / 0.346 A for `circuits-pid`).
4. Update `examples/M9_M23072801/summary.md` once the package reports time-weighted values.
5. Tell the student: the v0 figures (RMSE 0.8861 / 0.3588 A, MAE 0.7006 / 0.2525 A) were point-weighted;
   for v1 use 0.855 / 0.346 A, or rerun.

## Decision needed

**Default behaviour of `analyze_circuit_performance`:**
- (a) Switch to time-weighted metrics everywhere. Recommended: these are the physically meaningful values, and
  the old ones were only consistent within a single solver.
- (b) Report both, e.g. `rmse` (time) and `rmse_points` (legacy), for one release. This is noisier, but lets the
  student compare with v0 directly.
