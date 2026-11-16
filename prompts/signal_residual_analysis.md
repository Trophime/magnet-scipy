# Signal residual analysis (ROADMAP V4)

Status: plan, not implemented. Drafted 2026-11-12.
Replaces [signal-difference.md](signal-difference.md) and
[signal-difference-dtw.md](signal-difference-dtw.md), which are kept as background.

## Goal

Find **where** and **why** the simulated signals deviate from the measured ones:

- time windows where a residual exceeds the noise by a set factor, per channel and per GR;
- a likely cause for each window: offset, R, L/M or time skew, sensor, actuator, unexplained;
- per-channel, per-regime metrics, time-weighted, stored so they can later feed D3 (`simulations` table).

## Why not the two earlier notes

| Note | Method | Problem here |
|---|---|---|
| signal-difference | min-max / z-score normalisation, then pointwise threshold | Removes offset and gain, which are the diagnosis (tap offset, ΔR, k ≈ 0.90 on Tout). Global normalisation couples distant times. A threshold as a share of the range is about 6 kA on a 30 kA run. No noise model. `interp1d(..., 'extrapolate')` invents data. |
| signal-difference-dtw | `fastdtw` warping | The delays here are parametric: acquisition offset, transport delay plus first-order lag on Tout. On a ramp, an amplitude error equals a time shift, so DTW turns ΔR, ΔL and gain errors into warping. No band, sample indices instead of time, O(N·M), and a distance with no unit. |

Both ideas survive only as optional diagnostics (see the end of this note).

## Existing code

| Where | What | Matters because |
|---|---|---|
| `utils.exp_metrics` | RMS/MAE on 200 evenly spaced points over the overlap | coarse; crashes with `UnboundLocalError` when the ranges don't overlap (tracked separately) |
| `plotting_strategies._process_pid_data` | simulated U = `Kp·e + Ki·∫e + Kd·ė`, compared with measured U and I via `exp_metrics` | the only model-vs-measurement comparison today |
| `RLCircuitPID._load_experimental_data` | config `experiment_data` → `experimental_functions["{type}_{key}"]` with `time_data` and `values_data` | measured series are already wired per circuit |
| `ResultsFileManager.save_results` | `<circuit>_{current,reference,voltage,resistance,temperature}` and `time` in an `.npz` (`--save_results`; automatic with `--config-file`) | the residual analysis reads this file, so no change to the solver or plotting path is needed |
| PID model | **no voltage limit** (no clip or saturation in `pid_controller.py` / `rlcircuitpid.py`) | a closed-loop replay of a trip asks for any voltage needed to follow the reference to 0; see Decision 1 |
| [time_weighted_metrics.md](time_weighted_metrics.md) | time-weighted RMSE/MAE | reuse its `_time_mean` helper for every metric here |

## Residual model

Per GR, on the measurement grid:

$$r_U(t) = U_\text{meas}(t) - U_\text{sim}(t), \qquad r_I(t) = I_\text{meas}(t) - I_\text{sim}(t)$$

**Which channel carries the information.** In closed loop (PID driven by the reference) the simulated current follows
the reference whatever the plant error, so `r_I` mostly measures the controller and **`r_U` carries the plant
error**. In open loop, driven by the measured U (V5 method C), it is the other way round. v1 is closed loop.

**Signatures in `r_U`.** I is the current, a = dI/dt on a ramp, δ is a time skew:

| Cause | Plateau | Linear ramp | Ramp start/stop |
|---|---|---|---|
| Tap offset ΔU₀ | ΔU₀ | ΔU₀ | — |
| Resistance ΔR (incl. R(T) error) | ΔR·I | ΔR·I | — |
| Inductance ΔL (or ΔM·dI_other/dt) | 0 | ΔL·a | step |
| Time skew δ of U vs the reference | 0 | −δ·R·a | spike −δ·L·d²I/dt² |
| Tin lag δ_T (pupitre vs Overview) | ≈ I·∂R/∂T·δ_T·dT/dt | idem | — |

ΔL and skew are **collinear on ramps**. They separate only at the kinks, or when δ is estimated independently: by
cross-correlation between sources, or from the Default capture's absolute time stamps (see the C7 case below). For
the generated campaign cases, U, I and the reference come from the same Overview file, so δ ≈ 0. Only Tin (pupitre)
carries the uncorrected ±1 s lag.

**Model-free consistency residual (sensor check).** Computed on the measured signals only:

$$c_U(t) = U_\text{meas} - R(I_\text{meas}, T_\text{in})\,I_\text{meas} - L\,\frac{dI_\text{meas}}{dt} - M\,\frac{dI_\text{other}}{dt}$$

A large `c_U` means the measured signals contradict each other: a sensor fault, not a model error.
Example: 230919 t = 1807–1833 s, `I_GR1` frozen at 20 905 A while U ≈ −0.1 V gives c_U ≈ −250 V.

## Pipeline

1. **Align.** Use the measurement grid, restricted to the overlap with the simulation (never extrapolate).
   Interpolate the simulation onto it with `np.interp`: the solver's max step (0.01–0.1 s) is well below the 1 Hz grid.
   Record which source each series comes from, and its time offset, in the output.
2. **Residuals** in physical units: `r_U`, `r_I` per GR, `c_U` per GR.
3. **Segment by regime**, using the reference (clean), not the measurement:
   off (I_ref < 100 A), plateau (|dI_ref/dt| < ε), ramp, kink (± 2 samples around a slope change), and PID region
   switch (`low_threshold` / `high_threshold`).
4. **Noise σ per channel**: robust MAD of the first difference / √2 on plateau segments (removes the slow trend).
   Keep it per run, and also store it as the C0 baseline.
5. **Fit signatures per segment** with time-weighted least squares, `r_U ≈ a₀ + a₁·I + a₂·dI/dt`, with 95 % CIs.
   Then fit globally per regime type to report ΔU₀ [V], ΔR [Ω] and ΔL [H] (or δ [s] when δ is known).
6. **Detect** on both the raw residual and the residual after the fit: enter when |r| > k·σ (default k = 5), exit
   below k·σ/2 (hysteresis), minimum 2 samples, merge gaps < 5 s.
7. **Label** each window, first match wins:
   1. `sensor`: |c_U| > k·σ_c over the window.
   2. `actuator`: |I_meas − I_ref| ≫ tracking σ while U_meas ≈ 0, or while the reference steps to 0 (converter off).
   3. `offset` / `resistance` / `inductance_or_skew`: the signature term that explains most of the window variance
      (> 70 %).
   4. `unexplained`: everything else.
8. **Metrics**: time-weighted RMSE, MAE and bias of `r_U` and `r_I`, per channel and per regime.
9. **Output**: `<tag>_residuals.csv` (one row per window: `gr, channel, t_start, t_end, regime, peak, peak_z,
   integral, label`), `<tag>_residuals.json` (σ, fitted coefficients with CIs, metrics, sources), and a plot of the
   residuals with flagged windows shaded by label.

## Code layout

New module `magnet_scipy/residuals.py` (PEP 8, NumPy docstrings with units):

| Function | Returns |
|---|---|
| `align_series(t_sim, y_sim, t_meas, y_meas)` | common time grid, both series on it (overlap only) |
| `segment_regimes(t, i_ref, low_threshold, high_threshold, eps)` | :class:`~pandas.DataFrame` of segments |
| `noise_sigma(r, mask)` | robust σ in the channel's unit |
| `consistency_residual(t, u, i, resistance, inductance, mutual, i_other)` | `c_U` [V] |
| `fit_signatures(t, r, i, didt, segments)` | coefficients with CIs, per segment and per regime |
| `detect_deviations(t, r, sigma, k, k_exit, min_samples, merge_gap)` | windows |
| `label_windows(windows, ...)` | windows with `label` |

Entry point: `python -m magnet_scipy.residuals --wd <dir> --config-file <cfg> --results <npz>`. It reads the saved
`.npz`, plus the config for `experiment_data`, L, M and the R tables. No change to the solver, the CLI or plotting.
Leave fixing `exp_metrics` to its own issue.

## Demonstrator cases

All built by `examples/build_M23072801_campaign.py`. Event details (from the raw Default/Spike tdms):

| Case | Window | Expected result |
|---|---|---|
| **C0** 230916-175522 | full | Baseline. No window above k·σ; ΔU₀, ΔR, ΔL near 0. Sets σ and checks k. |
| **C7** 230919-190811 | t ∈ [1700, 1842] s (`value_start` from the CSV at 1700 s) | **1805–1810 s, `actuator`**: converter A1 fault (`A1_Def12` at 17:38:18.106 UTC); I_H decays freely (τ ≈ 0.2 s), I_B decays τ ≈ 1.2 s while the reference ramps to 0. **1805.29 s**: I_B 99 A above the reference, the M coupling transient (+0.9 kA at 4.8 kHz). **1807–1833 s, `sensor`**: I_GR1 frozen at 20 905 A, c_U ≈ −250 V. **1834–1842 s, `sensor`**: 1 kA → 477 A with U ≈ 0. |
| **C7** 240716-210814 | full | Negative control. 65 Default captures, but no fault line, mains dip or anomaly at the trigger (4 of 65 checked). Expect **no window**; any window is a false positive. |
| **C9** 240804-091254 | full | Spike-only run. 936 Spike captures, all on the insert tap `Interne7` (1.5–3.3 V, ms long, no coincident HT disturbance), 528 of them between t = 5400 and 7200 s; median I_GR1 at a spike 27.4 kA (90 % above 23.6 kA). Expect **no window at 1 Hz**: this case documents the detection limit of the 1 Hz lumped comparison. |

C7 240719-171041 behaves like C9 (659 spikes, also on `Interne7`) plus 3 Default captures with no fault line.

**Timing check from C7 230919.** The Default capture puts the first fault at t ≈ 1806.1 s on the run grid, but the
1 Hz CSV already shows the reference dropping at t = 1805.29 s. Use this sharp event to measure the
Overview-vs-Default offset before interpreting any window shorter than 2 s.

## Steps

1. **Tests first**, `tests/test_residuals.py`, on synthetic coupled RL signals with known faults → verify: they fail
   (module missing).
   - ΔU₀ = 0.5 V, ΔR = +2 %, ΔL = +5 % on a plateau–ramp–plateau reference: the fit recovers each one within its CI,
     and the labels are `offset`, `resistance` and `inductance_or_skew`.
   - Frozen current sensor over 20 samples: `sensor`, start and end within ±1 sample.
   - Converter off (U = 0, free RL decay) while the reference steps to 0: `actuator`.
   - Pure Gaussian noise, 10⁴ samples, k = 5: no window.
   - Partial overlap between simulation and measurement: no extrapolation; no overlap: a clear `ValueError`.
2. Implement `align_series`, `segment_regimes`, `noise_sigma` and `detect_deviations` → verify: the noise and window
   tests pass.
3. Implement `consistency_residual`, `fit_signatures` and `label_windows` → verify: all tests pass, and the existing
   suite stays green.
4. Add the entry point and run it on C0 → verify: no window; report σ per channel and the fitted coefficients.
5. Run it on C7 230919 (cropped), C7 240716 and C9 → verify: the outcomes in the table above. Write
   `examples/M9_M23072801_C7_default_events/residuals.md` with the result.
6. Add a MyST docs page `docs/residual_analysis.md` and the API entry; update ROADMAP V4 to point here.

## Later (not v1)

- **Open loop** (V5 method C): with measured U driving the model, `r_I` becomes the plant residual, and the C7
  trip, including the M coupling overshoot, becomes a quantitative test of M.
- **High-rate path**: run the same detector on Default/Spike captures (4.8 kHz) and per tap (`Interne1–7`). Needs
  per-helix R(I, T) to predict tap voltages. For C9, correlate spike rate with I, dI/dt and Tin.
- **Thermal channels**: a delay + first-order lag model for Tout (≈ 22 s + τ ≈ 19 s, gain ≈ 0.90; see M5).
- **Diagnostics**: shape comparison normalised **per segment**; DTW with a Sakoe-Chiba band used only to plot lag(t).
- **D3**: load `<tag>_residuals.json` into the DuckDB `simulations` table.

## Decisions needed

1. **Closed-loop replay of a trip.** The PID model has no voltage limit, so in closed loop the simulated U goes very
   negative during the C7 trip, and `r_U` there is meaningless.
   - (a) Detect and label actuator windows from the measurements alone (I_meas vs I_ref, U_meas ≈ 0), and mask them
     before fitting signatures. Recommended for v1.
   - (b) Add a converter model (voltage limit, off state) to the PID circuit. More faithful, but it changes the solver
     path that the student project depends on.
2. **Threshold**: k = 5 with σ per run, or σ fixed from the C0 baseline? Per run is more robust to sensor changes; the
   baseline is more comparable across runs.
3. **Output location**: next to the `.npz` in the case directory (proposed), or a central results directory for D3?
