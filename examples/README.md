# Examples

## `energy_balance.py` — energy balance of one run

Splits the electrical power of one magnet run into what is dissipated in the
magnets, the DC busbars, the power supplies (Graetz bridges) and the pumps,
compares the measured cooling temperatures `Tout` and `TAlimout` with
estimates computed from those powers, and balances the heat exchanger
(primary side against secondary/river-water side).

The data sources are complementary:

- **pigbrother** (`.tdms`) records the power installation: converter and
  magnet currents and voltages, bridge currents. It has no temperature or flow.
- **pupitre** (`.txt`) records the cooling: temperatures, flows, pressures,
  pump speeds, river-water temperatures and flow. It has no converter data.
- **SUPERVISION** (`bdd.csv` export, optional) records the site-level
  exchanger temperatures every ~5 s: `teb`, `tsb` (river water in/out) and
  `ted`, `tsd` (primary water at the same points as `Tout` and
  `Tin = (Tin1 + Tin2)/2`, with other sensors). The export available in
  `magnetdb-duckdb-demo/Data/bdd.csv` covers 2026-06-18 to 2026-11-06 only.

### Usage

```bash
python examples/energy_balance.py \
    "~/LNCMIG-Data/records/srv-data-install/M9/2024.07.15 - 09:07:12.txt" \
    ~/LNCMIG-Data/records/pbsurv/M9/Overview/M9_Overview_240715-0907.tdms \
    --delay 22 --tau 19 --csv balance.csv --plot balance.png
```

With the SUPERVISION export (2026 runs):

```bash
python examples/energy_balance.py \
    "~/LNCMIG-Data/records/srv-data-install/M9/2026.10.02 - 09:10:07.txt" \
    ~/LNCMIG-Data/records/pbsurv/M9/Overview/M9_Overview_261002-0909.tdms \
    --bdd ~/github/magnetdb-duckdb-demo/Data/bdd.csv --plot balance.png
```

The first argument is the pupitre file; the following ones are the pigbrother
Overview (1 Hz) or Archive (120 Hz) files covering the run. Archive channels
are averaged to 1 s. Requires `numpy`, `pandas`, `nptdms`, `duckdb` for
`--bdd` and `matplotlib` for `--plot`.

| Option | Default | Meaning |
|---|---|---|
| `--vt0` | `1.0` | Thyristor threshold voltage [V] of the bridge loss model |
| `--rt` | `1e-4` | Thyristor slope resistance [Ω] of the bridge loss model |
| `--pump-coef` | `0.023` | Electrical power of one magnet pump at 1000 rpm [MW], scaled as rpm³ |
| `--conv-flow` | `30` | Total cooling flow of the power-supply circuits [l/s]; `0` leaves the branch out of the heat exchanger balance and disables the `TAlimout` estimate |
| `--delay` | `0` | Transport delay applied to the `Tout` estimate [s] |
| `--tau` | `0` | First-order time constant of the `Tout` estimate [s] |
| `--talim-tau` | `0` | First-order time constant of the `TAlimout` estimate [s] |
| `--max-lag` | `120` | Search range of the pupitre/pigbrother time offset [s] |
| `--bdd` | none | SUPERVISION `bdd.csv` export; adds `ted`/`tsd` and finer `teb`/`tsb` to the heat exchanger balance |
| `--housing` | `M9` | Housing (`M5`, `M7`, `M8`, `M9`, `M10`): which magnet each supply group feeds |
| `--ptot-includes-pumps` / `--no-ptot-includes-pumps` | on | `Ptot` includes the magnet pumps: subtract `P_pump_el` from it when computing `P_alim` |
| `--csv` | none | Write the 1 s balance to this CSV file |
| `--plot` | none | Save a figure (powers, `Tout`, `TAlimout`) to this file |

`--vt0`, `--rt`, `--pump-coef` and `--conv-flow` are model parameters, not
measured values (see [Limitations](#limitations)). `--delay 22
--tau 19` are the `Tout` dynamics found on M9 run 2023.09.16 (22 s transport,
19 s first-order).

### Channels used

| Source | Group / column | Use |
|---|---|---|
| pigbrother | `Courants_Alimentations`: `Courant_A1..A4`, `Courant_GR1`, `Courant_GR2` | converter and magnet currents [A] (`GR1 = A1 + A2`, `GR2 = A3 + A4`) |
| pigbrother | `Tensions_Alimentations`: `Tension_A1..A4` | converter DC output voltage [V] (`= Tension_P_A + Tension_N_A`) |
| pigbrother | `Tensions_Aimant`: `ALL_internes`, `ALL_externes` | voltage across the insert and the Bitters [V] |
| pigbrother | `Courants_ponts`: `A1_p1..A4_p4` | DC current of each of the 4 bridges per converter [A] |
| pupitre | `Tin1`, `Tin2`, `Tout`, `TAlimout` | temperatures [°C] |
| pupitre | `Flow1`, `Flow2` [l/s], `HP1`, `HP2`, `BP` [bar], `Rpm1`, `Rpm2` | hydraulics of the two magnet branches |
| pupitre | `Pmagnet` [MW], `Idcct1..4` [A] | time alignment, group on/off detection |
| pupitre | `teb`, `tsb` [°C], `debitbrut` [m³/h] | river-water (cold) side of the exchanger |
| pupitre | `Ptot` [MW] | `Pmagnet` + busbars + magnet pumps + power supplies (transformers, converters); only computed while the supplies run |
| SUPERVISION | `teb`, `tsb` [°C] | river-water side in/out (same as pupitre, finer resolution) |
| SUPERVISION | `ted`, `tsd` [°C] | primary side, same points as `Tout` and `Tin`; second sensor pair |

Pupitre `Q` is reactive power [Mvar], not a flow; it is not used.

**Housing wiring.** Supply group GR1 (A1 + A2) and GR2 (A3 + A4) feed
different magnets depending on the housing: on M9 the insert is on GR1 and the
Bitters on GR2; on M8 and M10 the reverse; M5 (Bitters) and M7 (insert) use
GR1 only. `ALL_internes` (insert taps) and `ALL_externes` (Bitter taps) are
paired with the current of the group feeding that magnet. When
`python_magnetrun` is installed, the wiring comes from its housing
configuration, at the version in force at the start of the run (e.g. M10
had its supplies swapped from 2021-03-25 to 2021-09-29); otherwise a built-in
copy of the base wiring is used and the script says so. The wiring used is
printed.

Channel names with spaces (2023 files, e.g. `Courant GR1`) are accepted. A
group whose channels are missing from the pigbrother files (single-group
recordings) is set to 0 while its pupitre current is below 100 A, NaN
otherwise.

### What is computed

All powers are in MW, on a 1 s UTC grid.

| Term | Formula | Nature |
|---|---|---|
| Magnets `P_magnet` | `ALL_internes·I_insert + ALL_externes·I_Bitter`, currents `Courant_GR1`/`Courant_GR2` per the housing wiring; pigbrother equivalent of the pupitre `Pmagnet` (`= Σ Ucoil_k·Icoil_k`) | measured |
| Busbars `P_busbar_GRx` | `Puissance_Ax + Puissance_Ay − P_magnet` of the magnet that group feeds (`Puissance_A = Tension_A·Courant_A`) | measured |
| Power supplies `P_alim` | `Ptot − P_magnet − P_busbar_GR1 − P_busbar_GR2 − P_pump_el`, i.e. `Ptot − Σ Puissance_A − P_pump_el` (no pump term with `--no-ptot-includes-pumps`): transformer + converter losses | derived |
| Power supplies `P_conv` | `Σ_bridges 2·(V_T0·I_p + r_T·I_p²)` — two thyristors always conduct in series in a Graetz bridge | model |
| Pumps, hydraulic `P_hyd` | `Σ_i (HP_i − BP)·Flow_i`: flow work dissipated by friction between the HP and BP taps, i.e. in the magnets | measured |
| Pumps, electrical `P_pump_el` | `pump_coef·((Rpm1/1000)³ + (Rpm2/1000)³)`, magnet pumps only | estimate |
| `Tout_est` | `Tin + (P_magnet + P_hyd)/(ρcp·(Flow1 + Flow2))`, then delay and first-order lag | estimate |
| `k_cal` | `ρcp·(Flow1 + Flow2)·(Tout − Tin)/(P_magnet + P_hyd)`: fraction of the heat seen by the water | measured |
| `TAlim_est` | `Tin + P_conv/(ρcp·Flow_conv)`, bridge losses only (disabled with `--conv-flow 0`) | estimate |
| `F_alim_implied` | `P_alim/(ρcp·(TAlimout − Tin))` [l/s]: power-supply cooling flow that closes the balance with the measured losses | derived |
| `Q_hx_hot` | `ρcp·(F_mag·(Tout − Tin) + F_conv·(TAlimout − Tin))`, `F_mag = Flow1 + Flow2`, `F_conv` = `--conv-flow` | measured (pupitre) |
| `Q_hx_hot_bdd` | same with `ted` for `Tout` and `tsd` for `Tin`, only with `--bdd` | measured (SUPERVISION) |
| `Q_hx_cold` | `ρcp·debitbrut/3600·(tsb − teb)`, `debitbrut` = river-water flow [m³/h]; `teb`/`tsb` from SUPERVISION when given, pupitre otherwise | measured |
| `dT_ted`, `dT_tsd` | `ted − Tout`, `tsd − Tin`: agreement of the two sensor pairs [K], only with `--bdd` | sensor check |

`Tin = (Tin1 + Tin2)/2`: both sensors see the same exchanger outlet water
(the magnet pumps are after the exchanger), so their steady ~0.5 K difference
is a sensor offset and they are not flow-weighted. `ρcp = 4.17 MJ/(m³·K)`
(water at about 20 °C, where the flowmeters are). `P_hyd` belongs in the
magnet balance because a pressure drop `Δp` raises the water temperature by
about `Δp/(ρcp)` (0.48 K for 20 bar).

**Time alignment.** pupitre timestamps are site local time, pigbrother is UTC.
The remaining offset is found by matching pupitre `Pmagnet` with the
pigbrother magnet power within `±--max-lag` seconds and is printed. It differs
between runs (+2 to +43 s against Overview files); on 2024.07.15 it is +1 s
against the Archive and +24 s against the Overview, i.e. it comes from the
Overview time stamps. SUPERVISION timestamps are naive UTC (the build
exports `bdd.csv` with `mysql_connect.py --utc`) from the same clock as the
pupitre (checked on 2026.10.02 and 2026.10.28: `teb`/`tsb` agree within
0.05 K), so they get the pupitre offset. They are interpolated from ~5 s to
1 s, not across gaps longer than 15 s. A `bdd.csv` exported before the
`--utc` option is in local time and gives a 1–2 h shift: refresh it from the
current build (e.g. calcul22).

**Heat exchanger topology.** The hot side receives the magnet return
(`Tout`, measured downstream of the magnets only) and the power-supply
cooling return (`TAlimout`, fixed flow of its own pumps, joining before the
exchanger); both leave at `Tin`. The cold side is river water: `teb` in, `tsb` out, flow
`debitbrut`.

### Output

- **stdout**: the time offset, the wiring, the model parameters, then three
  tables with one row per steady plateau (power stable within 0.2 MW over
  5 min, at least 5 min long, above 1 MW; medians after 2 min of settling):
  power split (`P_mag`, `P_pup`, `P_bus_GR1`, `P_bus_GR2`, `P_conv`, `P_hyd`,
  `P_pump_el`, `Ptot`, `P_alim`), temperatures, and heat exchanger
  (`Q_hot`, `Q_hot_bdd`, `Q_cold`, `cold/hot`, `ted-Tout`, `tsd-Tin`); finally
  the energy of each term over the whole run [MWh].
- **`--csv`**: every term above at 1 s, including per-group columns
  (`P_magnet_GR1`, `P_busbar_GR2`, `P_conv_GR1`, …) and `Pmagnet_pupitre`.
- **`--plot`**: powers (log scale), `Tout` measured vs estimated, `TAlimout`
  measured vs estimated (with `Tin` on both temperature panels), and the heat
  exchanger hot and cold sides against `P_magnet` and `Ptot`.

Power split per plateau, default options [MW]:

| Run | `P_mag` | `P_pup` | `P_bus_GR1` | `P_bus_GR2` | `P_pump_el` | `Ptot` | `P_alim` | `F_alim_impl` |
|---|---|---|---|---|---|---|---|---|
| M9 2024.07.15 | 21.77 | 21.64 | 0.21 | 0.13 | 0.95 | 24.09 | 1.03 | 26.0 l/s |
| M9 2026.10.02 | 20.08 | 19.97 | 0.23 | 0.07 | 0.81 | 22.42 | 1.23 | 33.2 l/s |
| M10 2026.09.21 (`--housing M10`) | 17.70 | 17.66 | 0.13 (Bitters) | 0.20 (insert) | 0.96 | 20.16 | 1.19 | 32.5 l/s |

The M10 `P_pump_el` uses the M9 pump coefficient and is only indicative.

Example, M9 run 2024.07.15, first plateau (Overview, `--delay 22 --tau 19`):

| `P_mag` | `P_conv` | `P_hyd` | `P_pump_el` | `Tout` | `Tout_est` | `k_cal` | `TAlim` | `TAlim_est` | `F_alim_impl` |
|---|---|---|---|---|---|---|---|---|---|
| 21.77 | 0.15 | 0.48 | 0.95 | 39.9 °C | 42.4 °C | 0.89 | 30.0 °C | 21.6 °C | 26.0 l/s |

Heat exchanger, M9 run 2026.10.02 with `--bdd`, default options (branch at
30 l/s), two plateaus [MW, K]:

| `P_mag` | `Ptot` | `Q_hot` | `Q_hot_bdd` | `Q_cold` | `cold/hot` | `ted-Tout` | `tsd-Tin` |
|---|---|---|---|---|---|---|---|
| 15.67 | 17.71 | 15.18 | 14.91 | 20.64 | 1.36 | −0.37 | −0.05 |
| 20.08 | 22.42 | 19.23 | 18.87 | 25.77 | 1.34 | −0.42 | −0.04 |

### Limitations

- **`k_cal ≈ 0.89`**: on every M9 run checked (2023.09.14 to 2024.07.25), the
  water carries only 88–89 % of `P_magnet + P_hyd` above 10 MW, and 84–87 %
  at 3–9 MW. `Tout_est` is therefore 1–2.5 K above the measured `Tout`. The
  data fit `ΔT_measured ≈ 0.90·ΔT_true − 0.3 K`; whether the gain comes from
  the flowmeters or the temperature sensors is not resolved.
- **Power supplies**: their losses are only known as the remainder `P_alim`
  of `Ptot`, for both groups together (about 1–1.2 MW at full field). The HT
  channels (`Haute_Tension`, `HT_Courant`) are RMS values only, with no power
  factor; the `P1` active power of the `Fichiers_stats` files has offsets
  (−3.4 MW at zero current, +2.3 MW jumps at constant load on 2024.07.15) and
  is not used. `P_conv` is a conduction-loss model of the bridges only, with
  placeholder `V_T0`, `r_T`.
- **`TAlimout` at zero current** stays 4–5 K above `Tin`; whether the
  auxiliary pumps keep running at rest is not known.
- **`Ptot` includes the magnet pumps**, as the site staff state: at constant
  current, M9 run 2026.09.04 had pump 2 moved 2247 → 2780 → 1535 rpm and
  `Ptot − Pmagnet` followed by +0.25 / −0.16 MW (the rpm³ law predicts
  +0.27 / −0.21 MW); M9 2014.08.11 gives the same ratio. These are the only
  such events found in 5329 files. `Ptot` reads 0 at rest because it is
  only computed while the supplies run (at 0.5 MW it is even below
  `Pmagnet`), not because the pumps are excluded.
- **Pump electrical power** is not measured directly. `--pump-coef 0.023`
  (0.021–0.025 MW per pump at 1000 rpm) comes from those pump-speed steps:
  about 0.95 MW for the two M9 pumps at full field, against 0.50–0.53 MW of
  hydraulic power, i.e. an overall efficiency of about 0.55. A fit of
  `Ptot − Pmagnet` on currents and pump hydraulic power over 9500 plateaus
  cannot separate the pump term (pump speed follows the current: 96–99 % of
  its variance is explained by the currents).
- **Magnet voltage taps differ between pigbrother and pupitre**: currents
  agree within 0.03 %, but `ALL_internes` is 0.2–0.3 % above
  `Σ Ucoil1–14` and `ALL_externes` 0.8–1.0 % above `Ucoil15 + Ucoil16`
  (M9). About 0.1 MW of Bitter power at full field is therefore "magnet" or
  "busbar" depending on which taps are used; `P_busbar` uses pigbrother's.
- **Busbars** are not water-cooled: they lose their heat to the hall air by
  natural convection and radiation. Their power (0.2–0.35 MW at full field)
  is therefore part of `Ptot` but of no water balance, and the script does
  not add it to any temperature or heat exchanger estimate.
- **Heat exchanger does not close**: with the power-supply branch at
  30 l/s, the river-water side carries 1.32–1.40 times the hot side at
  11–22 MW (M9 2024.07.15, 2026.10.02); without the branch, 1.4–1.5 at
  15–20 MW and up to 2.2 at 3 MW (also 2026.07.22, 2026.09.30). On 2026.07.22 the plateaus
  fit `Q_cold ≈ 1.08·Ptot + 1.7 MW`; over 5000 M9/M10 plateaus,
  `Q_cold ≈ 1.06–1.08·Ptot + 1.3–1.5 MW`. Likely causes, from the data:
  - **`tsb − teb` offset of ≈ 0.8 K**: at rest (1196 windows, 2021–2026, no
    current, pumps on), `tsb − teb` is 0.80 K (P10–P90 0.7–0.9) on both
    housings, every year, whatever `debitbrut` (980–1300 m³/h), while the
    primary water is only 0.45 K above `teb`: constant in K, not in MW, and
    impossible as heat crossing the exchanger. Worth 0.9–1.5 MW at running
    flows.
  - **cold side ≈ 9 % high** once that offset is removed (`debitbrut` scale
    or `tsb − teb` gain), against the ≈ 0.98·`Ptot` the water receives.
  - **hot side ≈ 11 % low** (`k_cal ≈ 0.89`): single-magnet runs (idle branch
    stopped, so no mixing at the `Tout` sensor) give the same `k_cal` as
    two-magnet runs, with either branch, so neither poor mixing nor one
    faulty flowmeter explains it; both flowmeters ≈ 11 % low is the most
    likely cause (the SUPERVISION `ted`/`tsd` pair agrees with `Tout`/`Tin`
    within 2 %).
- **The two primary sensor pairs disagree on `Tout`**: `tsd` matches
  `(Tin1 + Tin2)/2` within 0.01–0.2 K, but `ted − Tout` depends on the load:
  about −0.4 K at 15–20 MW, +0.4 K at 3 MW and +0.9 K at rest. This is a gain
  (or linearity) difference between the two sensors, not a fixed offset, so
  `Q_hot` and `Q_hot_bdd` differ by 2 % at full power and up to 10 % at low
  power.
- **Power-supply circuit flow is not measured**: the auxiliary circuit (own
  pumps PA1–PA3, two parallel circuits A1/A2 and A3/A4, one outlet sensor
  `TAlimout` after they merge) has no flowmeter. `F_alim_implied` gives
  25–26 l/s on M9 2024.07.15, 33–38 l/s on M9 2026.10.02 and 32–34 l/s on
  M10 2026.09.21, stable within each run; `--conv-flow` defaults to 30 l/s
  (python_magnetcooling's `debit_alim` is 60 m³/h per circuit, i.e. 33 l/s
  in total). Over 294 runs with pigbrother data (2021-11 to 2026-10, 30 min
  plateaus above 5 MW), the implied flow scatters by 1.5 l/s within a run
  but 5 l/s (M9) to 9 l/s (M10) between runs, in levels that hold for days
  to weeks and then switch: M9 mostly 26–28 l/s, sometimes 34–38 l/s; M10
  28–36 l/s, sometimes 14–20 l/s, with `TAlimout − Tin` doubling at the same
  `P_alim` in those low periods. The flow therefore really changes, most
  likely with the auxiliary-pump configuration (PA1–PA3, VA9/VA10); scripts
  and results in [`cooling_investigations/`](cooling_investigations/). The branch carries about 1–1.2 MW at full field, far more
  than the bridge conduction losses `P_conv` (≈ 0.15 MW with the default
  `V_T0`, `r_T`): the circuit also cools the transformers (water/oil
  exchangers), whose losses are part of `Ptot`. `TAlim_est`, based on
  `P_conv` only, therefore stays far below the measured `TAlimout`.
