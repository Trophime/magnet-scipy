# Water flow params per magnet, checked against the campaign data

Goal: every M23072801 benchmark example carries one `python_magnetcooling`
flow params file per magnet, and a viewer command shows how well each file's
pump/flow/pressure model matches the measured pupitre data.

## Findings (2026-11-16, 24 campaign runs, 319k pupitre rows)

- magnetdb has no `flow_params` (empty in all 1952 overview records).
- Model (`python_magnetcooling.waterflow.WaterFlow`): `Vp = Vpmax (I/Imax)^2 + Vp0`,
  `F = F0 + Fmax Vp/(Vpmax+Vp0)`, `P = Pmin + Pmax (Vp/(Vpmax+Vp0))^2`,
  clamped at `Imax`; back pressure `BP`/`Pout` is a constant.
- Pupitre channels: GR1 (insert) `Rpm1`, `Flow1`, `HP1`; GR2 (Bitters)
  `Rpm2`, `Flow2`, `HP2`; `BP` shared. Current: `Idcct1+2`, `Idcct3+4`.
- `M19061901-flow_params.json`: does not fit M23072801; `Vp0`/`Vpmax` look
  swapped (1800/1000 rpm, the pump runs 1000 -> ~2950 rpm).
- `M19071101-flow_params.json`: pump speed fits M23072801 (RMS 57 rpm) but
  flow is ~4.5 l/s high (F0 = 6.9) and HP ~2.5 bar high near Imax.
- Measured BP rises with current (3.5 -> 4.6 bar); the model holds it constant.

### Operating mode

- Rule: magnetdb-duckdb-demo `to_duckdb/crud.py::infer_operating_mode`, the
  one that fills `overview_records.mode` (one value per Overview file): fit
  IH against IB where `50 A < IB < 0.25 max(IB)`; slope in `[0.66, 1.5]` is
  NORMAL, any other slope ECO; IH or IB never above 50 A is NORMAL (one supply
  powered, a convention, not an inference). No pupitre column holds the mode.
- Ported to `magnet_scipy/operating_mode.py`; `shot_modes` applies it per
  shot (stretch where |IH| or |IB| > 50 A) and labels the rest samples `rest`,
  so a record with several shots is split. Pupitre IH = `Idcct1+2`,
  IB = `Idcct3+4`.
- Per shot vs `provenance.json` `modes` (24 runs): same answer except
  - 231004-121211 (C3): provenance NORMAL, shots ECO (slope 2.40, as its C3
    twin 230927). Its record `M9_Overview_231004-1023` starts ~2 h before the
    pupitre file; the Overview currents restricted to the run window also give
    ECO (2.40). The record's NORMAL comes from the earlier part of the file:
    **C3 is ECO vs ECO on the simulated window.**
  - 230919-190811 (C7): provenance NORMAL, shots ECO; slope 1.44 on the
    Overview window, 1.54 / 1.75 on the two pupitre shots: on the 1.5 border.
  - 240717-174346 (C8, no overview record): ECO (both shots).
- No run changes mode between shots. All insert samples are ECO; the only
  NORMAL samples are the four Bitters-only C1 runs (by the one-supply
  convention).

## Decisions

1. `M23072801-flow_params.json` is fitted from the campaign pupitre data on
   the ECO samples (I > 300 A): Imax detected as the breakpoint of a
   2-segment piecewise fit, then the clamped pump model at that Imax, then
   flow and pressure with `fit_flow_rate` / `fit_pressure` (see Results).
   `M9Bitters_18MW-flow_params.json` the same way (`--group GR2`); it
   replaces the original file (Imax 28 kA, still at the repo root).
2. One file per magnet; the check reports the modes separately.
3. The mode of each sample is inferred from the supply currents (per shot);
   `provenance.json` `modes` is only the fallback for shots the rule cannot
   classify and for hydraulics CSVs without `ih`/`ib`.
4. The builder writes the measured hydraulics; configs reference the files.

## Steps

1. `examples/fit_M23072801_flow_params.py [--group GR1|GR2]`: read the
   pupitre files of all campaign runs (`provenance.json`), infer the mode per
   sample, fit the group on the ECO samples, write
   `examples/M9_M23072801/{M23072801,M9Bitters_18MW}-flow_params.json`, print
   the samples per run and mode and the fit statistics.
   -> verify: RMS of the new file on ECO data below the M19071101 one.
2. `build_M23072801_campaign.py`: per active GR write
   `<tag>_hydraulics_GR*.csv` (time, current, rpm, flow, hp, bp, and both
   supply currents ih, ib; whole pupitre file, same time 0), copy both flow
   params files into each group, add `"flow_params"` and `"hydraulics_csv"`
   to each circuit; README documents it. Base configs of `M9_M23072801` get
   `"flow_params"` too.
   -> verify: rerun the builder; config diffs only add the two keys.
3. `cli_core.CIRCUIT_CONFIG_KEYS`: accept both keys (used by the viewer and
   the future cooling model, ignored by the solver), so runs do not warn.
4. `magnet_scipy/operating_mode.py`: `infer_operating_mode` (port) and
   `shot_modes`.
   -> verify: unit tests; same answer as magnetdb on runs whose window is
   their whole Overview file.
5. Viewer `flowcheck CONFIG...`: per circuit, rpm/flow/HP/BP vs |I|, data
   coloured by the mode of each sample, model curve, RMS per mode in the
   legend; bias/RMS table per run and mode on stdout.
   -> verify: tests on synthetic files (per-sample split, provenance
   fallback); on the campaign, the numbers below.

## Results (`magnet-scipy-viewer flowcheck examples/M9_M23072801_C*/circuit*-*.json`)

### Insert fit, Imax detected (current file)

`examples/fit_M23072801_flow_params.py` (20 runs including C8, 257k ECO
samples > 300 A):

1. Imax = breakpoint of `fit_pump_speed_piecewise` (2 segments) on the median
   pump speed per 100 A bin: **27 066 A**. A free-Imax least-squares fit of
   the clamped model gives the same value (27 066 A).
2. Pump: clamped model `Vpmax (min(I, Imax)/Imax)^2 + Vp0` at that Imax:
   Vp0 = 999.5 rpm, Vpmax = 1800.5 rpm (full speed 2800 rpm),
   **RMS 0.25 rpm, R^2 = 1.00000**: the insert pump speed is a set point
   computed from the current, ~1000 + 1800 (I/27.07 kA)^2 rpm.
3. Flow / HP with `fit_flow_rate` / `fit_pressure` on that pump fit:
   F0 = 3.76 l/s, Fmax = 128.9 l/s, Pmin = 4.10 bar, Pmax = 18.19 bar;
   BP = 4.00 ± 0.27 bar.

`python_magnetcooling` issues met on the way (only the breakpoint of its
piecewise fit is used):

- `fit_pump_speed_piecewise` returns `vpmax` = speed at Imax, while
  `PumpSpeedFit.pump_speed` and `WaterFlow` use it as the rise above Vp0, so
  `fit_hydraulic_system(method="piecewise")` saturates at ~Vp0 + Vp(Imax);
  its extracted segment equations are also wrong here (`vpmax` = -9.7e7 rpm).
- Its 1-vs-2-segment check compares the fit with `current[-1]`, i.e. assumes
  data sorted by current.
- `fit_pump_speed_simple` fits the unclamped parabola: with Imax = 27 066 A
  the samples above Imax pull it to Vp0 = 1021, full speed 2759 rpm.

Previous fit (Imax fixed at 28 kA, unclamped `fit_hydraulic_system` simple):
Vp0 = 1021, Vpmax = 1860 rpm, F0 = 1.05, Fmax = 137.4 l/s, Pmin = 3.85,
Pmax = 20.1 bar. Same ECO samples, RMS:

| insert fit | rpm | flow [l/s] | HP [bar] | > 26 kA: rpm | flow | HP |
|---|---|---|---|---|---|---|
| Imax 28 kA fixed | 37.4 | 3.37 | 0.50 | 52.7 | 4.58 | 0.68 |
| Imax 27.07 kA detected | **0.25** | **2.59** | 0.54 | **0.20** | **3.13** | 0.74 |

The flow/HP residual left is an epoch shift, not a curve-shape error: flow
bias (model - measured) is -2.5 to -4.5 l/s for every run from 2023-09 to
2024-05-10 and +0.2 to +0.8 l/s from 2024-07-15 on, HP bias ~+0.2 then
-0.3 to -0.9 bar: at the same pump speed the insert passes less water after
mid-2024 (between 2024-05-10 and 2024-07-15). One file sits in between.

### Bitters fit, Imax detected (current file)

`fit_M23072801_flow_params.py --group GR2` (20 runs, 256k ECO samples
> 300 A; the C1 Bitters-alone runs are NORMAL and left out):
Imax = **26 916 A** (breakpoint), Vp0 = 999.5 rpm, Vpmax = 1780.8 rpm (full
speed 2780 rpm), pump **RMS 0.30 rpm, R^2 = 1.00000** (set point, as for the
insert); F0 = -0.61 l/s (intercept only: 41 l/s at Vp0), Fmax = 115.6 l/s,
Pmin = 3.80 bar, Pmax = 22.72 bar, BP = 4.00 ± 0.27 bar; R^2 flow 0.9994,
HP 0.9991.

Original file: Vp0 = 999.6, Vpmax = 1799.8 rpm, F0 = 0.65, Fmax = 116.4 l/s,
Pmin = 3.61, Pmax = 23.12 bar, Pout = 3.54 bar, Imax = 28 kA. Same samples:

| Bitters file | mode | rpm | flow [l/s] | HP [bar] | BP [bar] | > 26 kA: rpm | flow | HP |
|---|---|---|---|---|---|---|---|---|
| original (Imax 28 kA) | ECO | 54.3 | 1.73 | 0.87 | 0.54 | 108 | 3.52 | 1.78 |
| Imax 26.92 kA detected | ECO | **0.30** | **0.55** | **0.13** | 0.27 | **0.40** | **0.68** | **0.14** |
| Imax 26.92 kA detected | NORMAL | 214 | 10.3 | 2.31 | 0.39 | 55.5 | 4.47 | 0.74 |

No epoch shift on the Bitters: flow bias stays within ±0.8 l/s on every ECO
run from 2023-09 to 2024-08, so the mid-2024 flow change is the insert's own
hydraulics, not the plant. NORMAL (Bitters alone) is unchanged: pumps
saturate at ~22 kA in the 230906 runs (bias -245 to -265 rpm, -10 to -11 l/s).

### Check over all runs

RMS model - measurement, |I| > 300 A, mode per sample:

| magnet | mode | samples | rpm | flow [l/s] | HP [bar] | BP [bar] |
|---|---|---|---|---|---|---|
| M23072801 | ECO | 257k | 0.25 | 2.6 | 0.54 | 0.27 |
| M9Bitters_18MW | ECO | 256k | 0.30 | 0.55 | 0.13 | 0.27 |
| M9Bitters_18MW | NORMAL (Bitters alone) | 6.6k | 214 | 10.3 | 2.3 | 0.39 |

- Bitters NORMAL = Bitters alone (C1): pumps saturate at ~22 kA instead of
  ~27 kA (230906 runs); the two 4 T runs (231012, 231013) stay below that
  and fit. Whether this is the NORMAL
  pump law or a single-supply law cannot be told from this campaign: it has
  no two-supply NORMAL shot.
- Artefact: the vertical streak at 15.6 kA (flow 80 -> 117 l/s, HP 10 -> 17.5
  bar) is C7 230919 after the trip: the pupitre current stays frozen at
  15594 A while the hydraulics keep moving.
- BP rises with I (3.5 -> 4.6 bar); the constant model cannot follow
  (RMS 0.27 bar for both magnets with the fitted mean of 4.00 bar).

## Open

- Insert hydraulics change between 2024-05-10 and 2024-07-15 (flow -3 to
  -5 l/s at the same pump speed): one file per epoch, or a dated
  `flow_params`? Check the operation log for that period.
- Bitters alone (NORMAL by convention): a second Bitters file with its own
  Imax (~22 kA), if single-supply runs are to be simulated with the cooling
  model; only 2 runs (230906) reach the saturation.
- Report the three `python_magnetcooling` fitting issues above upstream.

- C3 is meant as "ECO vs NORMAL"; with the per-shot mode both runs are ECO.
  Pick a two-supply NORMAL run (the 231004 Overview's earlier shot?) if the
  comparison is still wanted, and recheck `test_campaign_M23072801.md`.
- `examples/M19` (M9 Bitters and M10 Bitters in the M19 housing): needs M19
  hydraulics data and an M10 Bitters file; the M9-housing fit does not apply
  (pumps are per housing).
- A mode-dependent Imax (NORMAL) and a current-dependent BP are not in the
  `WaterFlow` model.
