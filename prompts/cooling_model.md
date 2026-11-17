# Cooling model for magnet water loops

## Context

`magnet_scipy` simulates RL circuits for resistive magnets, where resistance is
`R(I, Tin)` — a function of current and the cooling-water **inlet** temperature
(`RLCircuitPID.get_resistance`, `magnet_scipy/rlcircuitpid.py:346`). Today `Tin` is
always exogenous: a constant or a measured `temperature_csv` time series
(`load_temperature_from_csv`, `rlcircuitpid.py:272`). The README's own todo list
names the gap directly: *"couple with cooling model for estimate of Tin"*.

Per the attached schematic and the user's clarifications, the real installation is
more than "N magnets on a loop through an exchanger":
- Each magnet branch's water flow is **controlled by that magnet's own current**,
  not a free input.
- The closed loop also feeds a **derivation** that cools the power installation
  (AC/DC Graetz bridges): **two circuits in parallel, one per group** (A1/A2 and
  A3/A4), each with a **fixed** water flow and a heat load modeled as a fraction of
  the electrical power the group delivers (`Q_conv_g = k_g·P_g`).
- The heat exchanger's transfer characteristic (UA) is **not reliably known**, so
  the loop supply temperature `T_supply` (exchanger primary cold outlet) should be
  computed from a **calorimetric balance** using the secondary loop's own sensors
  (`Teb`, `Tsb`, secondary flow) — `Q = ρ·Cp·Flow_secondary·(Tsb−Teb)`, then apply
  that measured heat removal to the primary side — rather than assumed from an
  NTU/effectiveness model that needs an unknown UA.
- The primary purpose is **calorimetry-first**: compute expected `Tout_i`, `Tout`
  (mixed), `Talim`, etc. from energy balances so they can be checked against the
  real `T*` sensor logs (reusing the existing experimental-data/rms-mae comparison
  pattern already used for current/voltage). Closing the loop back into `R(I,Tin)`
  for genuinely predictive (no-sensor-data) scenarios is a secondary capability,
  falling back through progressively less-informed estimates when sensor data is
  unavailable.
- Cooling-loop dynamics remain a first-order relaxation (time constant τ) toward
  the calorimetric equilibrium — confirmed earlier — and the model generalizes to
  an arbitrary number of magnets sharing an arbitrary number of loops.

## Physical model

### Loop topology and sensors (per the installation schematic)

```
HX cold outlet ─┬─ Pump/Flow1 ─ Tin1,HP1 ─ Magnet1 ─┐
 (T_supply,     ├─ Pump/Flow2 ─ Tin2,HP2 ─ Magnet2 ─┴─ Tout,BP ─┐   (expansion
  no sensor)    ├─ fixed flow ─ Graetz A1/A2 ─┐                 │    vessel on
                └─ fixed flow ─ Graetz A3/A4 ─┴─ Talim ─────────┴─►  hot header)
                                                                 HX hot inlet
Secondary (blue): Teb, Flow ─► HX ─► Tsb
```

- `Tin1 ≈ Tin2 ≈ T_supply`: all branches are fed from the exchanger's cold outlet,
  which has no dedicated sensor.
- `Tout` (pupitre) sees **only the magnet branches**: the converter return joins
  the hot header *downstream* of the `Tout` sensor, just before the exchanger.
- `Talim` (pupitre `Talimout`) is the **outlet temperature of the converter
  derivation** (assumed measured after the two group circuits merge), not a loop
  supply/return temperature.
- `HPi − BP` gives each magnet's pressure drop — an independent hydraulic check on
  `Flow_i` (not modeled here; see `flow_params_check.md` / `flow_probe_faults.md`).

### Branches (unifies magnets and the converter-derivation branches)

Every heat-loaded branch on a loop has a **heat source** `Q(t)` and a **flow**
`Flow(t)`, and produces `Tout_branch(t) = T_supply(t) + Q(t)/(ρ·Cp·Flow(t))` — purely
algebraic, no extra ODE state per branch.

- **Magnet branch** `i`: `Q_i(t) = R_i(I_i, T_supply)·I_i(t)²` (existing
  `get_resistance`). `Flow_i(t) = flow_curve_i(I_i(t))` — a current→flow lookup
  (1D CSV, same interpolation utility as everything else), since flow is
  current-controlled, not independent. Constant/CSV(t) flow remains supported as a
  simpler fallback for setups without a flow-curve.
- **Derivation branches** (0..N per loop; 2 at LNCMI, one per Graetz group
  A1/A2 and A3/A4, in parallel): each `g` has a heat source `Q_conv_g(t)` (from
  pigbrother bridge currents, see "Converter heat source from pigbrother"; fallback
  `k_g · Σ_{i∈g} Q_i(t)`) and a fixed `Flow_conv_g` (CSV kept only as an option). The two
  outlets merge before the `Talim` sensor:
  `Tout_conv = Σ_g Flow_conv_g·Tout_conv_g / Flow_conv`, `Flow_conv = Σ_g Flow_conv_g`
  — `Tout_conv` is the model counterpart of `Talim`.
  With fixed flows and `Talim`, `Tin` measured, the **total** converter heat is
  directly measurable: `Q_conv_meas = ρ·Cp·Flow_conv·(Talim − Tin)`. A single
  sensor only gives `Σ_g Q_conv_g`; `k_1`, `k_2` are still separable by
  regressing `Q_conv_meas` on the two group powers, provided the two magnet
  currents are not collinear over the record (true for runs with independent
  ramps/plateaus, e.g. the M9 Bitter/helix pairs).

### Converter heat source from pigbrother (power data is not in the pupitre)

Cooling data (`Tin1`, `Tin2`, `Tout`, `TAlimout`, flows, pressures, `teb`/`tsb`) come
from the pupitre only; power-installation data come from pigbrother only, so a run
needs both sources, aligned on absolute time (see the M9 alignment note:
pigbrother t = pupitre t + ~8.4 s; on 2024.07.15 the pupitre matches the
Archive within 1 s but the Overview time stamps are 23 s off). Feeder mapping (pigbrother doc + run 2024.07.15, where a
GR2 current step moves HT2 and A3+A4 only):
`HT1 → MC1 → A1/A2 → GR1 (insert)` **+ inner circuit pumps**,
`HT2 → MC2 → A3/A4 → GR2 (Bitters)` **+ river pumps + hybrid**.
The converters dominate each feeder (~12.7 MW at full field); the inner pumps are
~0.7–1.1 MW electrical at full speed (regression on HT1, see below).

Inner-pump heat is a loop heat source: the part dissipated between the HP and BP
taps, `Σ(HP_i − BP)·Flow_i` (≈ 0.48 MW at full field), shows up in `Tout − Tin`;
the rest of the pump work and the pump losses heat the water elsewhere in the loop
(exchanger, piping, pump), i.e. upstream of `Tin`, and must be included in the
loop-level balance against `Q_hx`.

What each pigbrother group can and cannot give (Overview and Stats are 1 Hz):

| Stage | Group | Content (1 Hz) | Use for the heat source |
|---|---|---|---|
| Grid AC | `Haute_Tension`, `HT_Courant` | RMS line voltage ~15 kV, RMS current ~590 A (unit mislabelled "Volts") | RMS only, no phase → no real power |
| Grid AC | `Fichiers_stats` `P1`, `S`, `cos_phi` | fundamental active power per feeder, **kW** (unit mislabelled "Volts") | `P_HT − P_DC` would be the total AC→DC loss (transformer + converter + filters) — but **not reliable on 2024.07.15**: `P1_HT2` = −3.4 MW at zero current, and both feeders step +2.3 MW mid-plateau with constant DC load (offset or shared feeder). Use only after a per-run zero-current check. |
| AC/DC | `Tensions_MC1`, `Tensions_MC2` | DC common-mode voltages of each converter, positive (`Tension_P_A*`) and negative (`Tension_N_A*`) parts; `Tension_A = P + N` (pigbrother doc) | no AC-side quantity → no converter input power |
| AC/DC | `Courants_ponts` | 4 bridge DC currents per converter (`A1_p1..A1_p4`, ~3.5 kA each) + sums | **bottom-up semiconductor losses** (below) |
| AC/DC | `Tensions_Alimentations`, `Puissances` | converter DC output voltage / power (`Puissance_A = Tension_A·Courant_A`) | DC power delivered per converter |
| Magnet | `Courants_Alimentations`, `Tensions_Aimant` | GR currents, magnet voltages | magnet power `P_H = ALL_internes·I_GR1`, `P_B = ALL_externes·I_GR2` |

Heat-source options for derivation branch `g`, in order of preference:

1. **Bridge loss model** from `Courants_ponts`: each 6-pulse bridge always has two
   thyristors in series, so
   `Q_g(t) = Σ_{p∈g} 2·(V_T0·I_p + r_T·I_p²) + Q_g,0`,
   with `V_T0`, `r_T` (datasheet values or fitted) and an idle term `Q_g,0`. Fit
   against the measured converter heat `ρ·Cp·Flow_conv·(TAlimout − Tin)` once the
   fixed flows are known. Order of magnitude at 28 kA / 26 kA (V_T0 ≈ 1 V,
   r_T ≈ 0.1 mΩ): ~0.15 MW for the 16 bridges.
2. **Fraction of delivered power** `Q_g = k_g·(Puissance_Ag1 + Puissance_Ag2)` —
   the fallback when no bridge currents are available (pupitre-only runs, e.g. C8).
3. **AC − DC balance** `P1_HTg − ΣPuissance_A` — upper bound only, includes
   transformer and filter losses that are probably not in this water circuit.

`examples/energy_balance.py` implements options 1 (with `--vt0`/`--rt`), the busbar
term, the pump terms, and the `Tout` / `TAlimout` estimates for one run.

DC busbar losses between converter and magnet are measurable and clean:
`(Tension_A − U_magnet)·I`, ≈ 0.21 MW (GR1) and 0.13 MW (GR2) at full field.
The busbars are not water-cooled (natural convection and radiation to the
hall air), so this power belongs to no water balance: not in `Q_g`, not in
`Q_hx`.

### Mixing — two distinct mixes

```
Tout_mag(t) = [Σ_magnets Flow_i·Tout_i] / [Σ_magnets Flow_i]            # ≈ "Tout" sensor
T_hx_in(t)  = [Σ_magnets Flow_i·Tout_i + Flow_conv·Tout_conv] / Flow_total  # HX hot inlet, unmeasured
```

### Supply temperature — calorimetric balance, tiered by available data

`CoolingLoop` picks the best available method for `T_supply_eq` (exchanger primary
cold outlet), in order:

1. **Calorimetric (preferred)** — if `Teb`, `Tsb`, and secondary flow are all
   available (constant/CSV/measured):
   `Q_hx(t) = ρ·Cp·Flow_secondary(t)·(Tsb(t) − Teb(t))`
   `T_supply_eq(t) = T_hx_in(t) − Q_hx(t) / (ρ·Cp·Flow_total(t))`
   No UA needed. Checked against the `Tin1`/`Tin2` sensors.
2. **Effectiveness-NTU fallback** — if a `HeatExchanger` (UA or fixed effectiveness)
   *is* supplied but full secondary-side calorimetry isn't (e.g. `Tsb` unmeasured):
   predict both `T_supply_eq` and `Tsb_eq` from `T_hx_in`, `Teb`, and both flows.
3. **Exogenous fallback** — neither of the above: `T_supply` is a directly
   supplied constant/CSV (today's status quo `Tin`, scoped to the loop).

### Loop relaxation (the one ODE state per loop, unchanged from earlier discussion)

```
dT_supply/dt = (T_supply_eq(t) - T_supply) / τ      # T_supply is this loop's shared Tin
```

Only **one new ODE state per cooling loop**, regardless of branch count — everything
else is algebraic given `T_supply` and `t`.

### Sensor-checking

`CoolingLoop` computes, at every solved time point: `Tout_i` per magnet branch,
`Tout_conv` (≈ `Talim`), `Tout_mag` (≈ `Tout` sensor), `T_hx_in`, `Q_hx`,
`Q_conv`, `T_supply_eq`, `Tsb_eq` (mode 2 only), `T_supply` (≈ `Tin1`/`Tin2`).
Also a primary-vs-secondary energy-balance residual:
`ρ·Cp·[Σ Flow_i·Tout_i + Flow_conv·Talim − Flow_total·Tin] − Q_hx`, built from
measured sensors only. These are compared against real sensor CSVs the same way
`RLCircuitPID.experimental_data`/`exp_metrics` already compares simulated vs
measured current/voltage (`rlcircuitpid.py:106-219`, `utils.py`'s `exp_metrics`).

## New module: `magnet_scipy/cooling_model.py`

```python
@dataclass
class WaterProperties:
    rho: float = 1000.0   # kg/m^3
    cp: float = 4186.0    # J/(kg*K)

class FlowControl:
    """Constant, CSV(time), or CSV(current) flow — same three-way pattern as
    RLCircuitPID's temperature/resistance inputs."""
    def __init__(self, constant=None, csv=None, csv_x="time", column="flow"): ...
    def __call__(self, t: float = None, current: float = None) -> float: ...

class CoolingBranch:
    """One heat-loaded branch on a loop: a magnet or the converter derivation."""
    def __init__(self, branch_id: str, heat_source: Callable[[float], float],
                 flow: FlowControl, circuit_id: str = None): ...
    def outlet_temperature(self, t, T_supply) -> float: ...

class HeatExchanger:                                    # optional, fallback #2 only
    def __init__(self, UA: float = None, effectiveness: float = None,
                 water: WaterProperties = None): ...
    def effectiveness(self, mdot_hot, mdot_cold) -> float: ...
    def outlet_temperatures(self, T_hot_in, T_cold_in, mdot_hot, mdot_cold): ...

class CoolingLoop:
    """N magnet branches + 0..M parallel derivation branches, calorimetric T_supply (tiered),
    one relaxation ODE state (T_supply). Independently testable: takes plain
    circuit_ids/currents dicts at each vector_field evaluation, not live objects."""
    def __init__(
        self, loop_id: str, circuit_ids: List[str],
        flow_curves: Dict[str, str] = None,          # circuit_id -> CSV(current,flow)
        flow_rates: Dict[str, float] = None,          # circuit_id -> constant fallback
        derivations: List[dict] = None,                # one per Graetz group, in parallel:
                                                       # {"branch_id", "circuit_ids", "heat_fraction", "flow" | "flow_csv"}
        secondary_supply_temperature: float = None, secondary_supply_temperature_csv: str = None,  # Teb
        secondary_return_temperature: float = None, secondary_return_temperature_csv: str = None,   # Tsb (drives mode 1)
        secondary_flow_rate: float = None, secondary_flow_rate_csv: str = None,
        heat_exchanger: HeatExchanger = None,          # mode 2 fallback
        supply_temperature: float = None, supply_temperature_csv: str = None,  # mode 3 fallback
        tau: float = 30.0, T_supply_initial: float = None,
        water: WaterProperties = None,
        experimental_data: List[dict] = None,          # sensor-checking, same shape as RLCircuitPID's
    ): ...
    def supply_mode(self) -> str: ...                  # "calorimetric" | "effectiveness" | "exogenous", resolved once at init
    def equilibrium_supply_temperature(self, t, circuits: Dict[str, RLCircuitPID], currents: Dict[str, float], T_supply: float) -> float: ...
    def dT_supply_dt(self, t, T_supply, circuits, currents) -> float: ...
    def get_diagnostics(self, t, circuits, currents, T_supply) -> Dict[str, float]:  # Tout_i, Tout_conv, Tout_mag, T_hx_in, Q_hx, Q_conv, T_supply_eq, Tsb_eq
    def load_experimental_data(self) -> None: ...        # mirrors RLCircuitPID._load_experimental_data
    def get_experimental_data(self, key: str, t: float) -> float: ...

def create_cooling_loop(**flat_kwargs) -> CoolingLoop: ...   # flat-kwargs factory, same pattern as create_adaptive_pid_controller
```

`CoolingLoop` takes plain `circuit_ids`, not circuit objects — the caller passes live
circuit/current dicts into `dT_supply_dt` at each `vector_field` evaluation. Keeps
the module self-contained/unit-testable and avoids import cycles with
`rlcircuitpid.py`. All CSV loading reuses `create_function_from_csv` from
`magnet_scipy/csv_utils.py:7` (1D, including current→flow curves) — identical
convention to `RLCircuitPID.load_temperature_from_csv`.

## Wiring into the electrical model

**`RLCircuitPID`** (`magnet_scipy/rlcircuitpid.py`) — single-circuit ownership:
- `__init__` gains `cooling_loop: CoolingLoop = None`, stored as `self.cooling_loop`.
  Used only when this circuit is simulated standalone (`single_circuit_adapter.py`).
- `vector_field`/`voltage_vector_field`: when set, `y` gains one trailing element
  (`T_supply`); use it as `Tin` instead of `self.get_temperature(t)`, and append
  `cooling_loop.dT_supply_dt(t, T_supply, {self.circuit_id: self}, {self.circuit_id: i})`
  to the returned derivative array. Without a `cooling_loop`, behavior is unchanged
  (backward compatible).

**`CoupledRLCircuitsPID`** (`magnet_scipy/coupled_circuits.py`) — system-level
ownership, parallel to `mutual_inductances`:
- `__init__` gains `cooling_loops: List[CoolingLoop] = None`; validates every
  `circuit_id` referenced exists and isn't claimed by more than one loop.
  `self.n_thermal_states = len(cooling_loops or [])`.
- `vector_field`/`voltage_vector_field`: `y` layout becomes
  `[currents(n), integral_errors(n), T_supply(n_loops)]` (voltage strategy omits the
  integral-error block). Per circuit, use its loop's `T_supply` for the resistance
  lookup (fall back to `circuit.get_temperature(t)` if unassigned to any loop).
  Append `[loop.dT_supply_dt(...) for loop in self.cooling_loops]`.

Circuits inside a `CoupledRLCircuitsPID` cooling setup should not also set their own
`cooling_loop` — loop membership is declared once at the coupled-system level.

**State-vector construction** (mirrors existing `[currents, integral_errors]`):
- `simulation_strategies.py`: `VoltageInputStrategy`/`PIDControlStrategy`
  `get_initial_conditions` (`:72`, `:163`) append
  `[loop.T_supply_initial for loop in system.cooling_loops]` when present.
- `single_circuit_adapter.py`: both strategies' `get_initial_conditions` (`:34`,
  `:99`) append `[circuit.cooling_loop.T_supply_initial]` when set.

## Plotting & sensor-checking (`magnet_scipy/plotting_strategies.py`)

- Slice the trailing thermal rows the same way `vector_field` builds them.
- `_get_temperature_over_time` (defined identically in both plotting strategy
  classes): when a circuit has a cooling loop, source `temperature_over_time` from
  the loop's solved `T_supply` row instead of calling `get_temperature(t)`; keep the
  existing branch unchanged for circuits without a loop.
- New: for each loop present, compute `get_diagnostics(...)` over the solved time
  grid and, for any `experimental_data` entries the loop has loaded, compute
  `rms_diff`/`mae_diff` via `exp_metrics` (`utils.py`) — identical pattern to the
  current/voltage comparison already in `_process_pid_data`
  (`plotting_strategies.py:406-475`). Surface these in `ProcessedResults` so
  `--show-analytics` reports temperature sensor-check metrics alongside the
  existing current/voltage ones, and the temperature subplot can overlay measured
  vs computed `Tout`/`Talim`/`Tin1`/`Tin2` the same way it already overlays experimental
  current.

## Config schema (`magnet_scipy/cli_core.py`, `ConfigurationLoader`)

New optional sections, parsed with `create_cooling_loop(**cfg)`:
- Single circuit (`load_single_circuit`, `cli_core.py:207`): top-level
  `"cooling_loop"` (singular object) → `RLCircuitPID(cooling_loop=...)`.
- Coupled (`load_coupled_circuits`, `cli_core.py:234`): top-level `"cooling_loops"`
  (list) → `CoupledRLCircuitsPID(cooling_loops=...)`.

Example (2-magnet case matching the schematic, calorimetric supply mode):
```json
"cooling_loops": [{
  "loop_id": "loop_1",
  "circuit_ids": ["magnet_1", "magnet_2"],
  "flow_curves": {"magnet_1": "flow_vs_current_1.csv", "magnet_2": "flow_vs_current_2.csv"},
  "derivations": [
    {"branch_id": "graetz_A1A2", "circuit_ids": ["magnet_1"], "heat_fraction": 0.02, "flow": 0.0025},
    {"branch_id": "graetz_A3A4", "circuit_ids": ["magnet_2"], "heat_fraction": 0.02, "flow": 0.0025}
  ],
  "secondary_supply_temperature_csv": "Teb.csv",
  "secondary_return_temperature_csv": "Tsb.csv",
  "secondary_flow_rate_csv": "Flow_secondary.csv",
  "tau": 30.0,
  "T_supply_initial": 20.0,
  "experimental_data": [
    {"type": "temperature", "key": "Tout", "file": "Tout_measured.csv"},
    {"type": "temperature", "key": "Talim", "file": "Talimout_measured.csv"},
    {"type": "temperature", "key": "Tin", "file": "Tin_measured.csv"}
  ]
}]
```
`flow_curves` CSV format: `current,flow` (mirrors `resistance_csv`'s `current,...`
column convention). `tau` should be estimated from the loop's water volume / total
flow rate as a starting point — note this in the README next to the example.

No schema-validation library exists in this repo (confirmed — plain `dict.get` with
defaults everywhere); the new sections follow the same convention.

## Small adjacent fix

`magnet_scipy/cli_simulation.py:222` prints a final current via
`result.solution[-1, i * 2]`, already wrong for the actual
`[currents(n), integral_errors(n)]` block layout with un-transposed
`(n_states, n_times)` storage (should be `result.solution[i, -1]`). Fix this one line
alongside, since the plan depends on precisely documenting the state layout and
leaving it wrong while adding more rows makes it more misleading. Print-only, not
load-bearing for plots/saved results.

## Tests: `tests/test_cooling_model.py`

Following `tests/test_csv_utils.py`'s style (fixture-based, `tmp_path` for CSV
cases, `pytest.mark.unit`, one `Test*` class per feature):
- `TestFlowControl`: constant / CSV(time) / CSV(current) modes each return expected
  values; CSV(current) mode used by a magnet branch driven by a ramping current.
- `TestCoolingBranch`: `Tout_branch` algebra against hand calculations.
- `TestCoolingLoopMixing`: multi-branch (2 magnets + 2 parallel derivations)
  flow-weighted `Tout_mag` (magnets only), `Tout_conv` (derivations only) and
  `T_hx_in` (all) against hand calculations; each derivation's heat scales with
  `k_g · Σ_{i∈g} Q_i`.
- `TestSupplyModes`: each of the three tiers (calorimetric / effectiveness-NTU /
  exogenous) produces the expected `T_supply_eq`, and the tier actually selected
  (`supply_mode()`) matches what data was supplied. Calorimetric-mode energy
  conservation: `Q_hx` computed from secondary side matches the primary-side heat
  removed implied by `T_hx_in → T_supply_eq`.
- `TestCoolingLoopDynamics`: integrating `dT_supply_dt` to steady state converges to
  `equilibrium_supply_temperature` (τ affects only approach speed, not the fixed
  point). Use small stub circuit objects (`get_resistance(I, T) -> constant`) rather
  than full `RLCircuitPID`.
- One light integration test: single `RLCircuitPID(cooling_loop=...)` run through
  `SingleCircuitPIDStrategy` for a few seconds with a step current reference;
  assert `Tin` rises monotonically toward the expected equilibrium and stays
  bounded.

## README updates

- Document `cooling_loop`/`cooling_loops` config fields with the example above,
  including the three supply-temperature tiers and when each applies.
- Check off the README todo: *"couple with cooling model for estimate of Tin"*.

## Verification

1. `pytest tests/test_cooling_model.py -v` — physics/calorimetry unit tests pass in
   isolation.
2. `pytest` (full suite) — confirm no regression in existing single/coupled circuit
   and plotting tests from the state-vector layout change.
3. Build one example config (single circuit + `cooling_loop` in exogenous-supply
   mode, using an existing `examples/testcase/test_resistance.csv`-style table) and
   run `magnet-scipy --config-file <cfg> --strategy pid --show-analytics
   --save-results out.npz` — confirm `Tin` in the results rises with current and
   settles.
4. Build a 2-magnet coupled config with synthetic `Teb`/`Tsb`/secondary-flow CSVs
   (calorimetric mode) plus a synthetic `Tout_measured.csv`/`Talimout_measured.csv` for
   `experimental_data` — confirm `--show-analytics` reports rms/mae for the
   temperature checks and `--show-plots` overlays measured vs computed curves.

## Open items for next discussion

- Derivation heat fraction `k_g`: confirmed as "fraction of magnet power"
  (`Q_conv_g = k_g · P_g`) rather than a lookup table or direct exogenous
  input — a reasonable starting proxy, refine once real converter loss data exists.
- Derivation flows: confirmed **fixed**, not tied to magnet current. Values still
  to be supplied (one per group circuit).
- Derivation: confirmed as **two parallel circuits, one per Graetz group** (A1/A2,
  A3/A4), whose combined outlet is `Talim` (pupitre `Talimout`), returning to the
  hot header downstream of `Tout`, before the HX.
- To confirm: which magnet(s) each group powers (example assumes A1/A2 → magnet_1,
  A3/A4 → magnet_2), and that `Talimout` is measured after the two group circuits
  merge (no per-group outlet sensor).
- Model purpose: confirmed calorimetry-first, doubling as both sensor-checking and
  (secondarily) predictive `Tin` for `R(I,Tin)`.
- Not yet decided: exact `k_g` values (identifiable from `Talim`/`Tin`, the fixed
  flows and the two group powers), real UA/effectiveness data availability (if any), and which
  sensor CSVs actually exist for a first real test case.
