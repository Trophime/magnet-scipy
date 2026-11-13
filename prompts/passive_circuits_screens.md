# Passive circuits (screens) and a general coupled RHS

Screens are conducting cylinders with no supply: no applied voltage and no
reference current. Their eddy currents still change the transient of the
driven circuits through the mutual inductances. To capture the skin depth, a
screen is split into `n_r × n_z` elements. Each element is an independent
short-circuited loop and gets its own row in the inductance matrix
(`my-magnettools` `e0ab675`, `mt.SplitScreen`, labels `Sc<i>_<k>`).

The simulator cannot represent such a row today. A coupled system must be all
PID or all voltage-driven. `HybridStrategy` was meant for mixed systems but
`run_simulation` raises `NotImplementedError`. Auto-detection never chose it
for a system with a screen (see ROADMAP N1).

This plan replaces `HybridStrategy` with one general coupled RHS in which each
row has a role, and adds screens as passive rows.

Depends on:
- [element_voltages.md](element_voltages.md) (M1), step 3: element matrix
  `M_elem`, incidence `P`, magnettools CSV loader. Screen elements are
  extra columns of `P`.
- [inductance_matrix_followups.md](inductance_matrix_followups.md) (N2),
  steps 3 and 6: `eigvalsh` SPD check, no `pinv` fallback. With finely split
  screens, M is ill-conditioned and these checks matter.

## Model

N rows (driven circuits, plus one row per screen element). Each row k has a
role: 𝒫 (PID), 𝒱 (imposed voltage) or 𝒮 (passive). The same circuit
equation holds for every row:

```
M·di/dt + R(|i|, T) ⊙ i = u
```

Only the driving voltage depends on the role:

| role | `u_k` |
|---|---|
| 𝒫 | `Kp·e_k + Ki·z_k + Kd·(di_ref,k/dt − di_k/dt)`, `e = i_ref − i`, `z = ∫e` |
| 𝒱 | `u_k(t)` from the CSV |
| 𝒮 | `0` |

Moving `Kd·di/dt` to the left gives one linear solve per RHS call:

```
(M + D)·di/dt = b,      D = diag(Kd on 𝒫 rows, 0 elsewhere)

b = −R ⊙ i
  + u(t)                                    on 𝒱 rows
  + Kp·(i_ref − i) + Ki·z + Kd·di_ref/dt    on 𝒫 rows

dz/dt = i_ref − i                           on 𝒫 rows only
```

State: `y = [i (N), z (|𝒫|)]`. M is SPD and Kd ≥ 0, so `M + D` is SPD and
Cholesky always works.

Screen element resistance: a geometry factor times the resistivity. The
geometry factor assumes uniform current density, the same assumption as
`UnifMagnet::Inductance`; the product equals `UnifMagnet::Resistance(1/ρ)`:

```
R_k = g_k · ρ,      g_k = π(r₁ + r₂) / ((r₂ − r₁)(z₂ − z₁))
```

This plan is **stage A**: ρ is a constant from the config. Screens usually run
at low temperature, where ρ depends strongly on T, B and RRR (stages B and C
below). Until then, run each case twice, with the lowest and the highest
plausible ρ: the spread between the two runs shows how much ρ matters.

Sizing guide for the split (order of magnitude): `δ ≈ √(2ρτ/μ₀)` for a
transient of time scale τ. Copper at room temperature: τ = 10 ms → 1.6 cm,
100 ms → 5 cm, 1 s → 16 cm. δ scales as √ρ: copper with RRR = 100 at 4 K
gives δ ≈ 1.6 mm at 10 ms. Size the split (`N_R × N_Z` in magnettools) for
the lowest ρ of the pair.

## Design

New module `magnet_scipy/coupled_model.py`:

```python
@dataclass
class CoupledModel:
    M: np.ndarray              # (N, N) [H]
    roles: np.ndarray          # (N,) "pid" | "voltage" | "passive"
    labels: list[str]          # circuit ids, then Sc<i>_<k>
    resistance: Callable       # (|i|, T) -> R, shape (N,) [Ω]
    temperature: Callable      # t -> T, shape (N,) [°C]
    voltage: Callable          # t -> u on voltage rows [V]
    reference: Callable        # t -> (i_ref, di_ref/dt) on PID rows
    gains: Callable            # i_ref -> (Kp, Ki, Kd, region_key) on PID rows

    def rhs(self, t, y) -> np.ndarray: ...
    def jac(self, t, y) -> np.ndarray: ...
    def voltages(self, t, y) -> np.ndarray: ...   # u = M·di/dt + R ⊙ i, all rows
```

- `rhs`: assembles `b` as above. Solves with `cho_solve` on a Cholesky
  factor of `M + D`, cached per `region_key` (the tuple of PID region names).
  The factor only changes when a PID row switches gain region.
- `jac`: `∂(di/dt)/∂i = −(M + D)⁻¹·diag(R + Kp on 𝒫)`,
  `∂(di/dt)/∂z = (M + D)⁻¹·Ki` on the 𝒫 columns, `∂(dz/dt)/∂i = −E`,
  `∂(dz/dt)/∂z = 0`. The `|i|·R'(|i|)` term is left out: it only affects
  Newton convergence, not the solution.
- `voltages`: one formula for every role. On 𝒫 rows it equals the control
  law (N1 tests); on 𝒮 rows it is 0 (a consistency check).
- `CoupledModel.from_system(system, screens=None)`: builds the callables
  from the existing `RLCircuitPID` / `CoupledRLCircuitsPID` objects (R(I, T)
  tables, `Tin`, references, `PIDController`). Screen elements get a constant
  R vector and the temperature is unused for them. No `RLCircuitPID` is
  created per screen element.
- Role of a driven circuit: inferred as today (reference + PID → `pid`,
  voltage only → `voltage`, both → `pid`). No `role` config key (decision 3).

Strategies: one `CoupledStrategy` runs `solve_ivp(model.rhs, …, jac=model.jac)`.
`detect_strategy` returns `"coupled"` when the roles differ or any row is
passive. `"pid"` and `"voltage"` stay as CLI names and run the same strategy
after checking that every row has that role. `HybridStrategy` is removed from
`SimulationRunner.strategies`. The class stays as a deprecated subclass of
`CoupledStrategy` until the next major version (CLAUDE.md shim pattern).

The legacy `vector_field` / `voltage_vector_field` methods stay unchanged in
this plan. The step-1 tests prove they give the same RHS. Deprecating them
belongs to H1.

## Config

Top-level `screens` list. The inductances come from the magnettools element
CSV (M1), which must have been generated with the same `--screen` arguments:

```json
"mutual_inductances": "M9_M23072801_inductances.csv",
"screens": [
  {"screen_id": "Sc1", "r_int": 0.20, "r_ext": 0.21,
   "z_inf": -0.30, "z_sup": 0.30, "rho": 1.7e-8, "n_r": 2, "n_z": 20}
]
```

- Element order: `k = i_r·n_z + i_z + 1`, the same as `SplitScreen`.
- Checks: the CSV has exactly `n_r·n_z` labels `Sc<i>_*` for each screen, and
  the screens are numbered in config order.
- M1 interaction: M1 step 3 drops CSV groups that no circuit uses. `Sc*`
  labels must be kept when the config has `screens`, and dropped with the
  usual warning otherwise.
- Initial screen currents: 0. Warn when `t_start` is not on a plateau (any
  PID row has `di_ref/dt ≠ 0` at `t_start`): the eddy currents are not 0
  there, so the start is inconsistent.

## Plan

Tests go in `tests/test_coupled_model.py`. Each step ends with that file and
the existing suite green.

1. **Tests** (fail first):
   - N = 1 PID: `model.rhs` equals `RLCircuitPID.vector_field(t, y, di_ref_dt)`
     at a sample of points, to 1e-12 relative (theory-review reference case,
     `R = 0.5 Ω`, `L = 0.2 H`).
   - N = 1 voltage: equals `voltage_vector_field`.
   - Coupled PID (`M₁₂ = 0.1 H`) and coupled voltage: equal
     `CoupledRLCircuitsPID.vector_field` / `voltage_vector_field`.
   - Mixed PID + voltage: hand-computed `b` and `di/dt` at one point.
   - Passive only, constant R: the solution equals
     `expm(−M⁻¹R·t)·i₀` (`scipy.linalg.expm`) to 1e-8.
   - PID + screen: `model.voltages` is 0 on every screen row at the output
     points (1e-9 V). High-ρ limit: driven currents match the no-screen run
     to 1e-6 relative (Radau).
   - `jac` matches a finite-difference Jacobian of `rhs` (constant R, 1e-6).
   - Cholesky cache: number of factorisations equals the number of distinct
     region keys seen.
   - Screen config: expands to `n_r·n_z` rows labelled `Sc1_1 … Sc1_40`;
     `R_k` matches the formula above; label-count mismatch raises; `Sc*`
     labels kept / dropped as described in "Config".
   - `detect_strategy`: PID + screen → `"coupled"`; all PID → `"pid"`;
     PID + voltage → `"coupled"`; `--strategy pid` on a system with a screen
     → error naming the passive rows.

2. **`CoupledModel` core**: `rhs`, `jac`, `voltages`, Cholesky cache, NumPy
   docstrings with units.
   → verify: equivalence, mixed and passive tests pass on hand-built models.

3. **`from_system` builder** for `RLCircuitPID` and `CoupledRLCircuitsPID`
   (no screens yet).
   → verify: N = 1 and coupled equivalence tests pass through the builder.

4. **`CoupledStrategy`** in `simulation_strategies.py`; new
   `detect_strategy` rule; `HybridStrategy` deprecated; `"pid"` and
   `"voltage"` coupled runs go through `CoupledStrategy`. Single-circuit
   runs are unchanged (decision 4).
   → verify: `examples/M9_M23072801/circuits-pid-temp.json` gives the same
   final currents as before to 1e-6 relative; full suite green.

5. **Screens**: `screens` config block, the R vector, `Sc*` rows from the
   M1 element matrix, the initial-value warning, and `"screens"` added to the
   top-level config keys.
   → verify: screen config tests pass.

6. **Post-processing**: a `"coupled"` plotting strategy. Driven circuits are
   plotted as today: current, voltage from `model.voltages`, R, power. Each
   screen gets one panel: total Joule loss `Σ R_k·i_k²` and the voltage it
   induces on each driven circuit, `Σ_k M_ak·di_k/dt`. Per-element screen
   currents are saved to the results file (npz/csv), not plotted. The N1 exact
   `de/dt` uses `model.rhs`.
   → verify: run M9 + a synthetic screen; by eye, the induced voltage is
   largest at the ramp corners and decays on the plateaus.

7. **Docs**: `theory_documentation.md`: the general model with roles, the
   screen element R, and the skin-depth sizing guide. README: the `screens`
   config block.

## Decisions

Settled (2026-11-12):

1. A general RHS replaces `HybridStrategy`.
2. Screens are passive rows; this item depends on M1 and N2.
3. This plan is stage A: constant ρ per screen from the config. ρ(T, B) and
   the screen temperature are stages B and C (see "Later stages").

Still open (default applies unless changed):

4. **Explicit `role` key on circuits**: *not added* (default; roles are
   inferred, ties go to PID). Alternative: optional `"role"` to force
   `voltage` when both CSVs are present, or to declare a passive coil that is
   not a screen.
5. **Single-circuit runs through `CoupledModel` (N = 1)**: *not in this plan*
   (default; the single path keeps its own strategies). Alternative: route it
   too, which removes `SingleCircuitPIDStrategy`/`…VoltageStrategy`.
6. **Source of the screen geometry factor `g_k`**: *computed in magnet-scipy
   from the config geometry* (default; no change in my-magnettools).
   Alternative: `inductances.py` writes a `<stem>_screens.csv` sidecar (label,
   r₁, r₂, z₁, z₂, g), so the geometry has one source and the config only
   names the screens.
7. **Default solver when passive rows are present**: *`Radau` with `jac`,
   unless `--method` is given* (default). Alternative: keep `RK45` and only
   warn.

## Later stages (not planned yet)

Recorded here so that stage A does not close the door on them.

**Stage B: ρ(T_op, |B(t)|) at a fixed screen temperature.**
- Resistivity law, for example copper: `ρ = ρ₂₇₃/RRR + ρ_i(T)` plus Kohler
  magnetoresistance `Δρ/ρ = f(B·ρ₂₇₃/ρ(T, 0))`. With azimuthal currents, both
  `Bz` and `Br` are transverse, so `|B|` is what matters.
- Field at each element: linear in the currents, `B_k = Σ_j G_kj·i_j`.
  `G` comes from a new magnettools export at the element centroids (or
  averaged over each element), next to the inductance CSV, so the same split
  defines M and G.
- RHS change: `resistance(i, T_states, t)` instead of `(|i|, T)`, with `B`
  from `G @ i`. Jacobian: ∂R/∂B is left out, like `R'(I)`.

**Stage C: thermal state per screen.**
- One temperature per screen (high-RRR copper is nearly isothermal at low T),
  with enthalpy as the state because `c_p` varies by orders of magnitude:
  `dH_s/dt = Σ_k R_k·i_k² − Q_cool(T_s)`, `T_s = T(H_s)`.
- Adiabatic first (`Q_cool = 0`: upper bound on the heating), then a cooling
  link `G(T)·(T_s − T_bath)`.
- State: `y = [i, z, H_screens]`. This is the first case where T is an ODE
  state rather than an input; M5 tier 1 can reuse the machinery, but the
  physics differs (cryogenic, not forced water convection).

**Open questions for B and C:**
1. Which screens: material, temperature level (4 K, 77 K, …), is RRR known?
   (B, C; for stage A only to choose the ρ bounds and the split.)
2. Cooling: bath immersion, conduction to a cryocooler, or attached to a
   thermal shield? (C)
3. Where the laws ρ(T, B), `c_p(T)`, `k(T)` are stored. The materials table
   holds scalars only (`electrical_conductivity` at `t_ref`, linear `alpha`).
   Proposal for the DB side: a `material_properties` table (material,
   property, model `linear_alpha` | `cu_rrr_kohler` | `table` | `expression`,
   params JSON, validity range, source). D1 exports the rows into the case
   directory, so magnet-scipy stays DB-free. Alternative: start from
   `materials.metadata` JSON in `python_magnetdb`. (B, C)

RRR is often known only within a factor of 2. Identifying it from the
eddy-current decay seen in the driven-circuit voltages after a ramp stops
(a V5 parameter) may beat a database value.

## Out of scope (noticed, not changed)

- Restarting the integration at the known breakpoints (reference corners,
  gain switches): N3.
- JAX/diffrax port of `CoupledModel` with masks instead of index sets: N4.
- Superconducting coils (current source, Thevenin ↔ Norton) and protection
  circuits: listed in `Examples.md`, not covered by the three roles.
- `RLCircuitPID` silently builds a default PID when it has neither
  `voltage_csv` nor `pid_controller`
  ([rlcircuitpid.py:87](../magnet_scipy/rlcircuitpid.py#L87)). Screens avoid it
  (no `RLCircuitPID` per element), but a passive coil declared as a circuit
  would hit it (see decision 3).
- Eddy currents also affect V3 (inductance identification on ramp corners):
  without screen rows, the identified `K` absorbs their effect.
