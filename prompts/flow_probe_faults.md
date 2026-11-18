# Flow probe faults in pupitre files: detection and reconstruction

Status: plan, not implemented. Drafted 2026-11-17.
Related: [flow_params_check.md](flow_params_check.md) (pump/flow model),
[signal_residual_analysis.md](signal_residual_analysis.md) (residual
windows, "sensor" cause).

## Trigger

`M10_2019.05.30---17:22:17.txt` (`python_magnetrun/srvdata`, same file in
`python_magnetsetup/data/mrecords`): `Flow2` drops from 74.5 to exactly
0.0 l/s at t = 407–496 s (plateau at 14 kA) and at t = 1142–1159 s
(ramp-down). Meanwhile `Rpm2` = 1487 rpm and `HP2` = 9.1 bar don't move.
The water flows, so the probe output is wrong.

## Survey (13 398 srvdata files, 22 190 group-files with `Rpm > 900`)

Throwaway scripts in the session scratchpad: `scan.py`, `scan2.py`,
`detect.py`, `census.py`, `bench.py`, `zero2.py`.

### Which witness predicts the flow

- **`Rpm1/2` is the pump set point, not the measured speed.** It leads the
  flow by 10–30 s on every ramp, so flow vs `Rpm` is a hysteresis loop. This
  matches flow_params_check: the rpm follows `Vp0 + Vpmax (I/Imax)^2` to
  0.25 rpm.
- `HP−BP` moves in step with the flow (same water, measured), with
  `Q ∝ (HP−BP)^n`, n = 0.42–0.57 (median ≈ 0.5, i.e. ΔP ∝ Q²).

Flow residual over 2 234 group-files with ramps [l/s]:

| predictor | median MAD | median p99 |
|---|---|---|
| `a + b·Rpm` (static) | 0.59 | 4.5 |
| `a + b·L_τ(Rpm)`, 1st-order lag, τ = 10–15 s | **0.56** | **2.8** |
| `c·(HP−BP)^n` | 0.89 | 4.8 |

The lagged-rpm line has the form of the `WaterFlow` model,
`F = F0 + Fmax·Vp/(Vpmax+Vp0)`, so `flow_params` gives its prior: for
M23072801, b = 128.9/2800 = 0.046 l/s/rpm, and the per-file median on M9
is 0.049.

### Fault families (census, events ≥ 3 s)

| family | events | group-files | where | nature |
|---|---|---|---|---|
| whole file at 0, `HP` ≈ 0 too | 3 627 | — | all sites | group not running or not recorded: **skip** |
| **dropout to 0, head kept** (> 1 bar, median 13 bar) | 3 990 | 682 | **97% M10 `Flow2`** | probe |
| flow 0 and head 0, `Rpm` set point unchanged | 220 | 150 | | **real** pump stop: never repair |
| stuck at full scale, 165–172 l/s | 241 | 73 | 94% M10 | probe |
| spikes / oscillations, head smooth | (inside "deviation", 3 114) | | e.g. M9_2015.04.27 GR1 | probe |
| staircase: flow held 10–20 s on fast ramps, lags the head 20–40 s | thousands, ≤ 10 s, ≤ 6 l/s | | all sites | slow flowmeter update: **informational** |
| `Rpm` channel frozen for the whole file | | | e.g. M10_2017.11.17/21 GR2 | rpm witness unusable |

M10 `Flow2` dropouts per year: 2017: 14, 2018: 42, **2019: 571**, 2020: 97,
2021: 772, 2022: 1059, 2023: 638, 2024: 667. The probe or its connection
has been degrading since early 2019 (operator information, not just a
data-cleaning item).

## M9 (2026-11-17)

Prototype v2 (`detect2.py`, `campaign.py`, `census_m9.py`, `bench_m9.py` in
the scratchpad) uses the rules updated below.

### M9_M23072801 campaign (24 runs, whole pupitre files)

- **No probe fault.** There are no dropouts and nothing stuck. Every flag
  is a known event or a real lag: the C7 230919 trip (7 s of pump
  slow-down) and C2 240716 GR2 (`hydraulic`: the pump can't follow a fast
  set-point ramp). C1 GR1 (insert off) is `pump_off`: `Rpm1` stays at its
  1000 rpm set point with no head.
- `Rpm` vs the `flow_params` set-point law: RMS 0.2–0.45 rpm, except
  C1 230906 GR2 at 240–256 rpm (Bitters alone, NORMAL, pumps saturate at
  ~22 kA). **The rpm check needs a mode-aware law.**
- Lags, constant per magnet: rpm set point → flow τ = 10 s on GR1 (insert)
  and 5 s on GR2 (Bitters); flowmeter behind the head τ_m = 5–10 s on GR1
  and 3 s on GR2.
- Per-file slope vs prior: GR1 0.046–0.052 l/s/rpm (prior 0.046), GR2
  0.041–0.045 (prior 0.042). The GR1 offset vs prior varies from −1.7 to
  +4.6 l/s between runs (insert epoch shift, flow_params_check). The head
  exponent n drops from 0.53–0.56 (2023 → 2024-05) to 0.50–0.52 (2024-07 →):
  another sign of the insert hydraulics change.
- C7 230919 trip: after the trip the flow settles at 79 l/s =
  `F(Vp(15.6 kA))`, not 50 l/s at idle, while the pupitre DCCT is frozen at
  15 594 A. **The pump set point seems to use the same frozen current**, so
  the rpm witness is not independent of the DCCT. Only 40 s of data after
  the trip: to confirm.

Gap filling on the campaign (2 640 artificial gaps, RMS [l/s], median / p90):

| gap | interp | per-file `F_rpm` + offset | `flow_params` alone | `flow_params` + offset | `F_hp` + offset |
|---|---|---|---|---|---|
| 5 s flat | 0.11 / 0.27 | 0.18 / 0.45 | 0.91 / 4.04 | 0.18 / 0.45 | 0.19 / 0.44 |
| 90 s ramp | 1.77 / 6.24 | 0.40 / 0.77 | 0.94 / 4.74 | 0.43 / 0.86 | 0.38 / 0.76 |
| 300 s ramp | 8.79 / 19.9 | 0.60 / 1.13 | 1.18 / 4.41 | 0.73 / 1.88 | 0.57 / 1.89 |

`flow_params` alone is off by up to ~4 l/s (the run-to-run offset).
Anchored on the gap edges, it is as good as the per-file fit, so a magnet
with a `flow_params` file can be filled without calibrating each file. The
head is an equivalent fallback when rpm is invalid.

### M9 archive (`srv-data-install/M9`, 6 427 files 2011–2026, 8 703 running group-files)

Baseline: 1–3 repairable events per 100 running hours on GR1, 0.2–2.5 on
GR2. Exceptions:

| episode | what | class |
|---|---|---|
| **2025-06 → 2025-08, GR1 (insert)**: 358 probe + 44 stuck events, 44.5 / 100 h | at high flow (≥ 128 l/s) the flow jumps erratically, then **clamps at 150 l/s** (full scale) for 30–120 s while rpm and head stay flat. 2025-08-01: bursts +20–35 l/s on each cycle. Worst week: 2025-06-14 → 06-23 | probe / stuck |
| 2015-04-27, GR1 | three files with spikes up to 150 l/s, head smooth | probe |
| 2016, GR1 (13.5 / 100 h on only 163 h) | spikes | probe |
| 2026-07-02, GR2 | the only clear M9 dropout: zeros and erratic values 0–110 l/s for ~350 s at 47–60 l/s, head 3.7 bar | dropout / probe |
| 2011–2013, mostly 2013-07 | flow updated every 10–20 s on ramps (staircase) | stale (info) |

Unlike M10 (3 887 `Flow2` dropouts), M9 has essentially no dropouts to
zero. Its failure mode is saturation/noise of the insert flowmeter at high
flow, concentrated in summer 2025.

## Detection (per group g, per file)

Rules updated after M9 (v2):

- `dropout`: zero flow and head > 0.6 × the file's *idle head* (head at
  `Vp0` with water flowing: ~2.5 bar M10 GR2, ~3.5 bar M9 GR2). A fixed
  1 bar was not enough: M9 GR2 keeps a static ~1.1–1.2 bar after a pump
  stop.
- Head witness lagged by the flowmeter's τ_m (0–15 s); otherwise fast
  ramp-downs are flagged.
- τ fixed per magnet when known (M9: 10 s GR1, 5 s GR2): plateau-only files
  otherwise pick τ = 0.
- `stale` vs `stuck` by whether the witnesses ramp (max |dF_hp/dt| over
  ±10 s > 0.3 l/s/s), not by how long the value is held. A held value while
  the witnesses ramp is `stale`; held ≥ 20 s while they are flat is `stuck`.

Original v1 description:

Inputs: `Flow_g`, `Rpm_g`, `HP_g`, `BP`, `I_g` (`Idcct` sums), optionally
the magnet `flow_params`.

0. **Active**: `Rpm_g > Vp0 − 100` and `HP_g − BP > 0.3 bar`. Otherwise the
   sample is `off` (not tested).
1. **Validate the witnesses**
   - rpm: with `flow_params`, `|Rpm − Vp(I)| < 20 rpm`. Without them, `Rpm`
     must vary where `I` varies (no frozen channel).
   - head: `HP−BP > 0.3 bar`, not frozen.
2. **Calibrate** (robust, MAD-trimmed least squares, on active samples with
   flow > 0.5):
   - `F_rpm = a + b·L_τ(Rpm)`, τ ∈ {0, 5, 10, 15, 20, 30} s chosen by the
     smallest p95 residual. Prior/fallback `(a, b)` from `flow_params` when
     the file has < 300 rpm of range, or when the fitted slope is far from
     the prior.
   - `F_hp = c·(HP−BP)^n`.
   - noise `σ = 1.4826·MAD`, threshold `thr = max(6σ, 5 l/s)`.
3. **Classify** each sample:

   | flag | rule |
   |---|---|
   | `dropout` | flow ≤ 0.5 and head > 1 bar |
   | `stuck` | flow constant at full scale (≥ 165 l/s) and > thr from both witnesses |
   | `probe` | `|flow − F_rpm| > thr` **and** `|flow − F_hp| > thr` |
   | `hydraulic` | flow agrees with `F_hp`, both disagree with `F_rpm` (pump trip, valve): real |
   | `pump_off` | flow ≈ 0 and head ≈ 0: real |
   | `stale` | flow piecewise constant ≥ 5 s while `F_hp` moves > thr (informational) |

   Merge runs separated by ≤ 2 s, drop runs < 3 s (except `dropout`), and
   widen probe runs by 1 s on each side (edge samples can be partial).

On `M10_2019.05.30---17:22:17.txt` the prototype flags exactly 407–496 and
1142–1159 on GR2 and nothing on GR1. Every gap injected into 400 clean
group-files was found (100%).

## Reconstruction

Only on `dropout` / `stuck` / `probe` samples. `hydraulic` and `pump_off`
are real.

**Model plus edge-anchored offset**: `F_fill(t) = F_model(t) + r(t)`, where
`r` is linearly interpolated between the median residual
`flow − F_model` over the 10 s before and after the gap (one-sided at file
edges). `F_model = F_rpm` when rpm is valid, otherwise `F_hp`. The local
offset absorbs drift such as the insert epoch shift in flow_params_check.

Benchmark, 6 396 artificial gaps in 400 real group-files, RMS
[l/s], median / p90:

| gap | rpm | linear interp | `F_rpm` | **`F_rpm` + offset** | rpm+HP regression |
|---|---|---|---|---|---|
| 5 s | flat | 0.10 / 0.29 | 0.32 / 0.90 | 0.19 / 0.50 | 0.31 / 0.88 |
| 20 s | ramp | 0.33 / 1.07 | 0.62 / 1.84 | 0.39 / 1.19 | 0.58 / 1.69 |
| 90 s | ramp | 0.98 / 3.73 | 0.76 / 1.87 | 0.59 / 1.82 | 0.71 / 1.73 |
| 300 s | ramp | 3.83 / 14.4 | 0.88 / 2.13 | 0.77 / 2.29 | 0.78 / 2.07 |

One method covers every case: linear interpolation is marginally better
only on short flat gaps, and adding HP as a second regressor gains nothing.

Output: keep the raw `Flow_g`. Add `Flow_g_filled`, `Flow_g_flag`, and a
per-sample uncertainty (≈ 0.5 l/s on plateaus, ≈ 2 l/s p90 on ramps).
Downstream (`hydraulics_csv`, the cooling model, the
`Tout = Tin + P/(ρ c_p Q)` check of parameter_identification) reads the
filled column and can mask on the flag.

## Proposed steps

1. `magnet_scipy/flow_probe.py`: `lag_first_order`, `calibrate_flow_witnesses`,
   `classify_flow_samples`, `reconstruct_flow`, and a `check_flow_probe(df, group,
   flow_params=None)` that returns a DataFrame with the filled flow and flags.
   → verify: unit tests on synthetic series (dropout with head kept, pump
   stop, stuck high, frozen rpm, gap at file edge) and a regression test on
   `M10_2019.05.30---17:22:17.txt` (two runs, nothing on GR1).
2. Viewer `flowprobe FILE...`: flow, `F_rpm`, `F_hp`, flagged and filled
   samples; per-file event table on stdout.
3. `build_M23072801_campaign.py`: write `flow_filled` / `flow_flag` in
   `<tag>_hydraulics_GR*.csv`.
   → verify: no flags on the M9 campaign except known events (C7 trip).
4. Optional: a census CLI over a records directory (per site/group/year),
   for the operators.

## Open

- Package location: `magnet_scipy` (next to `flow_params`, viewer, cooling
  model) or `python_magnetrun` (where pupitre files are read)? Proposal:
  here first, upstream once stable.
- Full-scale value of the flowmeters: 171–172 l/s on M10, and a few files
  stuck at 484. Per-site value needed to make `stuck` exact.
- `stale` (slow update on ramps): fill it as well, or only flag it? It
  biases a transient energy balance by up to ~6 l/s for 10–40 s.
- No `flow_params` for M10 yet: the per-file fit is the only calibration
  there, so files where rpm is frozen *and* the flow is faulty fall back to
  the head alone.
- Report the M10 `Flow2` probe degradation (2019 →) to operations, and the
  M9 GR1 flowmeter saturating/bursting at high flow (2025-06 → 2025-08).
- Confirm on a longer post-trip record that the pumps take their set point
  from the same DCCT reading as the pupitre (C7 230919).
- Mode-aware rpm law (Bitters alone / NORMAL saturates at ~22 kA) before
  the rpm-vs-`Vp(I)` check is used as a witness validator.
