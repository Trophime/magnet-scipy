# M9 PID solver sweep: `circuits-pid.json` vs `circuits-pid-temp.json`

Generated 2026-11-12 from magnet-scipy branch `claude_cleanup` (uncommitted working tree).
To reproduce, see [BENCHMARK.md](BENCHMARK.md).

- Run: `magnet-scipy-coupled --wd examples/M9_M23072801 --config-file <cfg> --strategy pid --value_start -0.609178 0.236709 --time_start 0 --time_end 1100 --time_step 0.1 --method <solver>`.
  The defaults rtol = 1e-6 and atol = 1e-9 apply. `circuits-pid.json` with RK45 reproduces the M2 student's v0 report exactly.
- Timing: `/usr/bin/time -v`, 6 runs in parallel, 1 BLAS thread each. CPU time and peak RAM are comparable between runs.
  Wall times are inflated by the parallel load: RK45 alone took 1631 s CPU on this machine and 992 s wall on the student's Mac.
- Experimental comparison: sim − exp evaluated at the 1 s pigbrother samples over 0.5–1099.5 s.
  Experimental data: `M9_Overview_230916-1755_Courants_Alimentations_Courant_GR{1,2}_vs_Time.csv` and `M9_Overview_230916-1755_Tensions_Aimant_ALL_{internes,externes}_vs_Time.csv`.
- **Tracking RMSE/MAE are averaged over solver output points, as the package computes them.** They therefore depend on each solver's step distribution.
  Weighted by time, the tracking RMSE is the same for all solvers to within 0.2 %:
  0.8546–0.8564 A (H) and 0.3458–0.3463 A (B) for `circuits-pid`; 0.8497–0.8515 A (H) and 0.3436–0.3441 A (B) for `circuits-pid-temp`.
  See `prompts/time_weighted_metrics.md`.
- Tout: calorimetric cTout from `python_magnetcooling.heatexchanger_primary` (`estimate_temperature_elevation` per magnet, then `mixing_temp`).
  Inputs are the measured Flow1/2, Tin1/2, HP1/2 and BP from the pupitre record `M9_2023.09.16---17:55:22.txt`.
  The time bases are aligned before comparison: pigbrother CSV t == pupitre t + 8.4 s, from cross-correlation of I_H, I_B, U_H and U_B.
  Fit model: Tout = Tin_mix + k·F_τ[cTout − Tin_mix](t − L), where F_τ is a first-order low-pass filter.
  "measured U·I" uses measured power instead of a simulation, as a baseline independent of the circuit model.
- H = M23072801 (helix insert, GR1, index 1); B = M9Bitters_18MW (Bitter, GR2, index 2).

## Performance (0–1100 s, max_step = 0.1 s, 6 runs in parallel, 1 thread each)

| Config | Solver | Status | Wall [s] | CPU [s] | Peak RAM [MiB] | Solution points |
|---|---|---|---:|---:|---:|---:|
| circuits-pid | RK45 | ok | 2846.4 | 2715.7 | 1801 | 1,877,463 |
| circuits-pid | RK23 | ok | 1973.0 | 1972.5 | 2247 | 2,473,785 |
| circuits-pid | DOP853 | ok | 2814.2 | 2757.4 | 1081 | 972,230 |
| circuits-pid | Radau | ok | 18.8 | 18.8 | 174 | 11,245 |
| circuits-pid | BDF | ok | 9.5 | 9.5 | 175 | 11,802 |
| circuits-pid | LSODA | ok | 8.1 | 8.1 | 174 | 12,364 |
| circuits-pid-temp | RK45 | ok | 3002.6 | 2880.1 | 1802 | 1,877,452 |
| circuits-pid-temp | RK23 | ok | 2193.4 | 2176.1 | 2247 | 2,473,764 |
| circuits-pid-temp | DOP853 | ok | 3026.2 | 2949.2 | 1081 | 972,218 |
| circuits-pid-temp | Radau | ok | 22.1 | 22.1 | 176 | 11,230 |
| circuits-pid-temp | BDF | ok | 12.6 | 12.6 | 175 | 11,774 |
| circuits-pid-temp | LSODA | ok | 8.8 | 8.8 | 176 | 12,334 |

## M23072801: tracking and comparison with experiment

| Config | Solver | P_max [MW] | Tracking RMSE / MAE / max [A] | Current vs exp RMSE / MAE [A] | Voltage vs exp RMSE / MAE [V] |
|---|---|---:|---|---|---|
| circuits-pid | RK45 | 10.52 | 0.8861 / 0.7006 / 1.5736 | 44.5 / 34.0 | 3.46 / 2.86 |
| circuits-pid | RK23 | 10.52 | 0.8872 / 0.7109 / 1.6377 | 44.5 / 34.0 | 3.72 / 3.09 |
| circuits-pid | DOP853 | 10.52 | 0.8859 / 0.7040 / 1.6143 | 44.5 / 34.0 | 3.71 / 3.10 |
| circuits-pid | Radau | 10.52 | 0.8479 / 0.6404 / 1.5837 | 44.5 / 34.0 | 3.71 / 3.10 |
| circuits-pid | BDF | 10.52 | 0.8343 / 0.6223 / 1.5860 | 44.5 / 34.0 | 3.71 / 3.10 |
| circuits-pid | LSODA | 10.52 | 0.8230 / 0.6074 / 1.5858 | 44.5 / 34.0 | 3.71 / 3.10 |
| circuits-pid-temp | RK45 | 10.45 | 0.8809 / 0.6967 / 1.5637 | 44.5 / 34.0 | 1.75 / 1.39 |
| circuits-pid-temp | RK23 | 10.45 | 0.8821 / 0.7076 / 1.6274 | 44.5 / 34.0 | 2.04 / 1.62 |
| circuits-pid-temp | DOP853 | 10.45 | 0.8808 / 0.7008 / 1.6039 | 44.5 / 34.0 | 2.00 / 1.61 |
| circuits-pid-temp | Radau | 10.45 | 0.8432 / 0.6379 / 1.5755 | 44.5 / 34.0 | 1.99 / 1.61 |
| circuits-pid-temp | BDF | 10.45 | 0.8301 / 0.6208 / 1.5756 | 44.5 / 34.0 | 1.99 / 1.61 |
| circuits-pid-temp | LSODA | 10.45 | 0.8199 / 0.6075 / 1.5755 | 44.5 / 34.0 | 1.99 / 1.61 |

## M9Bitters_18MW: tracking and comparison with experiment

| Config | Solver | P_max [MW] | Tracking RMSE / MAE / max [A] | Current vs exp RMSE / MAE [A] | Voltage vs exp RMSE / MAE [V] |
|---|---|---:|---|---|---|
| circuits-pid | RK45 | 4.00 | 0.3588 / 0.2525 / 0.8565 | 32.3 / 24.4 | 6.48 / 5.32 |
| circuits-pid | RK23 | 4.00 | 0.3585 / 0.2541 / 0.8598 | 32.3 / 24.4 | 6.46 / 5.30 |
| circuits-pid | DOP853 | 4.00 | 0.3584 / 0.2526 / 0.8538 | 32.3 / 24.4 | 6.46 / 5.30 |
| circuits-pid | Radau | 4.00 | 0.3423 / 0.2294 / 0.8462 | 32.3 / 24.4 | 6.46 / 5.30 |
| circuits-pid | BDF | 4.00 | 0.3388 / 0.2247 / 0.8463 | 32.3 / 24.4 | 6.46 / 5.30 |
| circuits-pid | LSODA | 4.00 | 0.3349 / 0.2196 / 0.8463 | 32.3 / 24.4 | 6.46 / 5.30 |
| circuits-pid-temp | RK45 | 3.97 | 0.3566 / 0.2515 / 0.8552 | 32.4 / 24.4 | 5.44 / 4.40 |
| circuits-pid-temp | RK23 | 3.97 | 0.3562 / 0.2525 / 0.8585 | 32.4 / 24.4 | 5.42 / 4.38 |
| circuits-pid-temp | DOP853 | 3.97 | 0.3561 / 0.2510 / 0.8531 | 32.4 / 24.4 | 5.42 / 4.38 |
| circuits-pid-temp | Radau | 3.97 | 0.3404 / 0.2287 / 0.8450 | 32.4 / 24.4 | 5.42 / 4.38 |
| circuits-pid-temp | BDF | 3.97 | 0.3369 / 0.2241 / 0.8450 | 32.4 / 24.4 | 5.42 / 4.38 |
| circuits-pid-temp | LSODA | 3.97 | 0.3333 / 0.2194 / 0.8450 | 32.4 / 24.4 | 5.42 / 4.38 |

## Outlet temperature: calorimetric cTout (python_magnetcooling) vs measured Tout, sources aligned (8.4 s)

| Power source | Solver | No lag RMSE / MAE [K] | Plateau offset [K] | cTout max [°C] | Fit L [s] | k | τ [s] | Fit RMSE [K] |
|---|---|---|---:|---:|---:|---:|---:|---:|
| measured U·I | n/a | 2.02 / 1.79 | +1.41 | 34.44 | 21.8 | 0.899 | 19.3 | 0.14 |
| circuits-pid | RK45 | 2.20 / 1.94 | +1.69 | 34.70 | 22.5 | 0.882 | 19.5 | 0.14 |
| circuits-pid | RK23 | 2.20 / 1.94 | +1.69 | 34.70 | 22.5 | 0.882 | 19.5 | 0.14 |
| circuits-pid | DOP853 | 2.20 / 1.94 | +1.69 | 34.70 | 22.5 | 0.882 | 19.5 | 0.14 |
| circuits-pid | Radau | 2.20 / 1.94 | +1.69 | 34.70 | 22.5 | 0.882 | 19.5 | 0.14 |
| circuits-pid | BDF | 2.20 / 1.94 | +1.69 | 34.70 | 22.5 | 0.882 | 19.5 | 0.14 |
| circuits-pid | LSODA | 2.20 / 1.94 | +1.69 | 34.70 | 22.5 | 0.882 | 19.5 | 0.14 |
| circuits-pid-temp | RK45 | 2.13 / 1.89 | +1.62 | 34.64 | 22.5 | 0.887 | 18.9 | 0.14 |
| circuits-pid-temp | RK23 | 2.13 / 1.89 | +1.62 | 34.64 | 22.5 | 0.887 | 18.9 | 0.14 |
| circuits-pid-temp | DOP853 | 2.13 / 1.89 | +1.62 | 34.64 | 22.5 | 0.887 | 18.9 | 0.14 |
| circuits-pid-temp | Radau | 2.13 / 1.89 | +1.62 | 34.64 | 22.5 | 0.887 | 18.9 | 0.14 |
| circuits-pid-temp | BDF | 2.13 / 1.89 | +1.62 | 34.64 | 22.5 | 0.887 | 18.9 | 0.14 |
| circuits-pid-temp | LSODA | 2.13 / 1.89 | +1.62 | 34.64 | 22.5 | 0.887 | 18.9 | 0.14 |
