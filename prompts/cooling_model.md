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
- The closed loop also feeds a **derivation branch** that cools part of the power
  installation (AC/DC Graetz bridge converters) — its own flow, and a heat load
  modeled as a fraction of the magnets' electrical power (`Q_conv = k·ΣP_magnet`),
  independent flow (constant or CSV).
- The heat exchanger's transfer characteristic (UA) is **not reliably known**, so
  `Talim` (the loop's return/supply temperature) should be computed from a
  **calorimetric balance** using the secondary loop's own sensors (`Teb`, `Tsb`,
  secondary flow) — `Q = ρ·Cp·Flow_secondary·(Tsb−Teb)`, then apply that measured
  heat removal to the primary side — rather than assumed from an NTU/effectiveness
  model that needs an unknown UA.
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

### Branches (unifies magnets and the converter-derivation branch)

Every heat-loaded branch on a loop has a **heat source** `Q(t)` and a **flow**
`Flow(t)`, and produces `Tout_branch(t) = T_supply(t) + Q(t)/(ρ·Cp·Flow(t))` — purely
algebraic, no extra ODE state per branch.

- **Magnet branch** `i`: `Q_i(t) = R_i(I_i, T_supply)·I_i(t)²` (existing
  `get_resistance`). `Flow_i(t) = flow_curve_i(I_i(t))` — a current→flow lookup
  (1D CSV, same interpolation utility as everything else), since flow is
  current-controlled, not independent. Constant/CSV(t) flow remains supported as a
  simpler fallback for setups without a flow-curve.
- **Derivation branch** (0 or 1 per loop): `Q_conv(t) = k_conv · Σ_i Q_i(t)`
  (`k_conv` a configured fraction). `Flow_conv(t)` independent — constant or CSV(t).

### Mixing (this is the diagram's "Tout" — needs no exchanger model)

```
T_mix(t) = [Σ_branches Flow_b(t)·Tout_b(t)] / [Σ_branches Flow_b(t)]
```

### Talim — calorimetric balance, tiered by available data

`CoolingLoop` picks the best available method, in order:

1. **Calorimetric (preferred)** — if `Teb`, `Tsb`, and secondary flow are all
   available (constant/CSV/measured):
   `Q_hx(t) = ρ·Cp·Flow_secondary(t)·(Tsb(t) − Teb(t))`
   `Talim_eq(t) = T_mix(t) − Q_hx(t) / (ρ·Cp·Flow_primary_total(t))`
   No UA needed — this is also exactly the check against the real `Talim` sensor.
2. **Effectiveness-NTU fallback** — if a `HeatExchanger` (UA or fixed effectiveness)
   *is* supplied but full secondary-side calorimetry isn't (e.g. `Tsb` unmeasured):
   predict both `Talim_eq` and `Tsb_eq` from `T_mix`, `Teb`, and both flows.
3. **Exogenous fallback** — neither of the above: `Talim` is a directly supplied
   constant/CSV (today's status quo, renamed/scoped to the loop).

### Loop relaxation (the one ODE state per loop, unchanged from earlier discussion)

```
dT_supply/dt = (Talim_eq(t) - T_supply) / τ         # T_supply is this loop's shared Tin
```

Only **one new ODE state per cooling loop**, regardless of branch count — everything
else is algebraic given `T_supply` and `t`.

### Sensor-checking

`CoolingLoop` computes, at every solved time point: `Tout_i` per magnet branch,
`Tout_conv`, `T_mix` (≈ `Tout` sensor), `Q_hx`, `Talim_eq`, `Tsb_eq` (mode 2 only),
`T_supply` (≈ `Tin`). These are compared against real sensor CSVs the same way
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
    """N magnet branches + optional derivation branch, calorimetric Talim (tiered),
    one relaxation ODE state (T_supply). Independently testable: takes plain
    circuit_ids/currents dicts at each vector_field evaluation, not live objects."""
    def __init__(
        self, loop_id: str, circuit_ids: List[str],
        flow_curves: Dict[str, str] = None,          # circuit_id -> CSV(current,flow)
        flow_rates: Dict[str, float] = None,          # circuit_id -> constant fallback
        derivation_heat_fraction: float = None, derivation_flow: float = None,
        derivation_flow_csv: str = None,
        secondary_supply_temperature: float = None, secondary_supply_temperature_csv: str = None,  # Teb
        secondary_return_temperature: float = None, secondary_return_temperature_csv: str = None,   # Tsb (drives mode 1)
        secondary_flow_rate: float = None, secondary_flow_rate_csv: str = None,
        heat_exchanger: HeatExchanger = None,          # mode 2 fallback
        talim: float = None, talim_csv: str = None,    # mode 3 fallback
        tau: float = 30.0, T_supply_initial: float = None,
        water: WaterProperties = None,
        experimental_data: List[dict] = None,          # sensor-checking, same shape as RLCircuitPID's
    ): ...
    def talim_mode(self) -> str: ...                   # "calorimetric" | "effectiveness" | "exogenous", resolved once at init
    def equilibrium_supply_temperature(self, t, circuits: Dict[str, RLCircuitPID], currents: Dict[str, float], T_supply: float) -> float: ...
    def dT_supply_dt(self, t, T_supply, circuits, currents) -> float: ...
    def get_diagnostics(self, t, circuits, currents, T_supply) -> Dict[str, float]:  # Tout_i, Tout_conv, T_mix, Q_hx, Talim_eq, Tsb_eq
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
  vs computed `Tout`/`Talim`/`Tin` the same way it already overlays experimental
  current.

## Config schema (`magnet_scipy/cli_core.py`, `ConfigurationLoader`)

New optional sections, parsed with `create_cooling_loop(**cfg)`:
- Single circuit (`load_single_circuit`, `cli_core.py:207`): top-level
  `"cooling_loop"` (singular object) → `RLCircuitPID(cooling_loop=...)`.
- Coupled (`load_coupled_circuits`, `cli_core.py:234`): top-level `"cooling_loops"`
  (list) → `CoupledRLCircuitsPID(cooling_loops=...)`.

Example (2-magnet case matching the schematic, calorimetric Talim mode):
```json
"cooling_loops": [{
  "loop_id": "loop_1",
  "circuit_ids": ["magnet_1", "magnet_2"],
  "flow_curves": {"magnet_1": "flow_vs_current_1.csv", "magnet_2": "flow_vs_current_2.csv"},
  "derivation_heat_fraction": 0.02,
  "derivation_flow": 0.005,
  "secondary_supply_temperature_csv": "Teb.csv",
  "secondary_return_temperature_csv": "Tsb.csv",
  "secondary_flow_rate_csv": "Flow_secondary.csv",
  "tau": 30.0,
  "T_supply_initial": 20.0,
  "experimental_data": [
    {"type": "temperature", "key": "Tout", "file": "Tout_measured.csv"},
    {"type": "temperature", "key": "Talim", "file": "Talim_measured.csv"}
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
- `TestCoolingLoopMixing`: multi-branch (2 magnets + derivation) flow-weighted
  `T_mix` against hand calculations; derivation branch heat scales correctly with
  `k_conv · ΣQ_i`.
- `TestTalimModes`: each of the three tiers (calorimetric / effectiveness-NTU /
  exogenous) produces the expected `Talim_eq`, and the tier actually selected
  (`talim_mode()`) matches what data was supplied. Calorimetric-mode energy
  conservation: `Q_hx` computed from secondary side matches the primary-side heat
  removed implied by `T_mix → Talim_eq`.
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
  including the three Talim tiers and when each applies.
- Check off the README todo: *"couple with cooling model for estimate of Tin"*.

## Verification

1. `pytest tests/test_cooling_model.py -v` — physics/calorimetry unit tests pass in
   isolation.
2. `pytest` (full suite) — confirm no regression in existing single/coupled circuit
   and plotting tests from the state-vector layout change.
3. Build one example config (single circuit + `cooling_loop` in exogenous-Talim
   mode, using an existing `examples/testcase/test_resistance.csv`-style table) and
   run `magnet-scipy --config-file <cfg> --strategy pid --show-analytics
   --save-results out.npz` — confirm `Tin` in the results rises with current and
   settles.
4. Build a 2-magnet coupled config with synthetic `Teb`/`Tsb`/secondary-flow CSVs
   (calorimetric mode) plus a synthetic `Tout_measured.csv`/`Talim_measured.csv` for
   `experimental_data` — confirm `--show-analytics` reports rms/mae for the
   temperature checks and `--show-plots` overlays measured vs computed curves.

## Open items for next discussion

- Derivation branch heat fraction `k_conv`: confirmed as "fraction of magnet power"
  (`Q_conv = k_conv · ΣP_magnet`) rather than a lookup table or direct exogenous
  input — a reasonable starting proxy, refine once real converter loss data exists.
- Derivation branch flow: confirmed independent (constant or CSV), not tied to
  magnet current.
- Model purpose: confirmed calorimetry-first, doubling as both sensor-checking and
  (secondarily) predictive `Tin` for `R(I,Tin)`.
- Not yet decided: exact `k_conv` value(s), real UA/effectiveness data availability
  (if any), which sensor CSVs actually exist for a first real test case, and
  whether the derivation branch should ever be per-converter (one per Graetz
  bridge) rather than a single aggregate branch per loop.
