# Per-element voltages for series-connected circuits

A circuit (power supply group) is several elements in series — e.g. the
M23072801 insert as 7 helix pairs `H1H2 … H13H14`, the M9 Bitters as
`M9_Be`, `M9_Bi`. Goal: keep solving for the N circuit currents, but
post-process the voltage across each element, split into three terms for
debugging.

## Model

Element `a` of circuit A, current `I_A`; `P` is the n_elem × N incidence
matrix (`P[e, k] = 1` if element `e` is in circuit `k`):

```
U_a = R_a(I_A, T_A)·I_A  +  Σ_{a'∈A} M_aa'·dI_A/dt  +  Σ_{B≠A} Σ_{b∈B} M_ab·dI_B/dt
      └─── U_R ───────┘     └──────── U_L_own ─────┘   └────────── U_M_other ────────┘
```

Circuit level (what the ODE uses, unchanged):

- `M_circuit = Pᵀ · M_elem · P` (so `L_A = Σ_{a,a'∈A} M_aa'`, `M_AB = Σ_{a∈A,b∈B} M_ab`)
- `R_A = Σ_a R_a` — already the `R[ohm]` column of `R_<circuit>.csv`
  (checked: `max |Σ R_e − R| ≈ 2e-18` for both example files)
- `T_A` is the circuit inlet temperature (`Tinit[C]`), shared by its elements

Identity: `Σ_{a∈A} U_a = u_A`. Exact in voltage mode (same `dI/dt` as the
ODE). In PID mode it is exact against `u = R·I + M·dI/dt`, but only
approximate against the plotted PID output, which uses `np.gradient`
([plotting_strategies.py:431](../magnet_scipy/plotting_strategies.py#L431)).

## Input formats

### Element resistances (existing file, no change)

`examples/M9_M23072801/R_14Helices.csv`:

```
I[A],R_H1H2[ohm],…,R_H13H14[ohm],R[ohm],Tinit[C]
```

This file goes in the existing `resistance_csv` key and is auto-detected.
The element format is recognised when the columns include `I[A]`, `Tinit[C]`,
`R[ohm]` and at least one `R_<id>[ohm]`. Otherwise the file is read in the
current `current,temperature,resistance` format (e.g. `Rtot_14Helices.csv`).
Element ids are taken from the `R_<id>[ohm]` headers, in column order. Each
column goes through `create_2d_function_from_csv(csv, "I[A]", "Tinit[C]",
"R_<id>[ohm]")`, the same gridded loader `resistance_csv` already uses.

### Element mutual inductances (from python3-magnettools)

One CSV for the whole site, written by
`compute_inductance_matrix(Tubes, Helices, BMagnets, UMagnets, csvfile=…)`
(`my-magnettools/Python/inductances.py:25`). It holds the full symmetric
element matrix [H]. Rows and columns are helix tubes `H_<i>`, then Bitters
`B_<i>`, then supras `S_<i>`; the header has an empty first cell, and the
values are written as `.6e`:

```
,H_0,H_1,…,B_0,B_1
H_0,1.234567e-04,…
…
```

`H_i` is already a tube aggregate (elementary helices + their mutuals inside
the tube). The same call also writes `<stem>_summary.csv` with the
group totals `L_H`, `L_B`, `L_S`, `M_HB`, `M_HS`, `M_BS`.

Config: `mutual_inductances` becomes a path. Each circuit names its
magnettools group with `matrix_group`:

```json
"circuits": [
  {"circuit_id": "M23072801", "matrix_group": "H",
   "resistance_csv": "R_14Helices.csv", …},
  {"circuit_id": "M9Bitters_18MW", "matrix_group": "B",
   "resistance_csv": "R_M9Bitters_18MW.csv", …}
],
"mutual_inductances": "M9_M23072801_inductances.csv"
```

Mapping element ↔ matrix label is positional inside the group: the k-th
`R_<id>[ohm]` column ↔ `<group>_k`. Required check: the number of `<group>_*`
labels equals the number of element resistances, for each circuit
(tubes ↔ `R_14Helices.csv`, B magnets ↔ `R_M9Bitters_18MW.csv`, supras ↔
their file if any). The loader dispatches on type: str → element mode,
numbers → current behaviour (packed vector or N×N matrix). Existing configs
are untouched.

Groups present in the CSV but used by no circuit (e.g. `S_*` when only H and
B are simulated) are dropped with a warning. This is equivalent to treating
that coil's current as constant.

No real CSV exists yet for M9_M23072801. How to produce one from the duckdb
demo is planned in [inductance_csv_from_duckdb.md](inductance_csv_from_duckdb.md).

## Plan

Tests go in `tests/test_element_voltages.py` and use small synthetic CSVs in
`tmp_path` (2 circuits: 2 + 1 elements), plus the real `R_14Helices.csv`.
Each step ends with `pytest tests/test_element_voltages.py
tests/test_inductance_config.py` green.

Depends on step 2 of [inductance_matrix_followups.md](inductance_matrix_followups.md)
(`_build_inductance_matrix`). Do that step first, or as part of step 3 here.

1. **Tests** (fail first):
   - Element R loader on `R_14Helices.csv`: auto-detected as element format;
     ids `["H1H2", …, "H13H14"]`; `Σ R_e(I, T) == R(I, T)` at grid points and
     at an off-grid point. `Rtot_14Helices.csv` still loads as a single R with
     no elements.
   - Assembly from a synthetic magnettools-format CSV (`H_0, H_1, B_0`):
     `M_elem` symmetric, `M_circuit == Pᵀ M_elem P`, element `k` ↔ `<group>_k`.
   - Cross-check against `<stem>_summary.csv`: `M_circuit` equals
     `[[L_H, M_HB], [M_HB, L_B]]` to the CSV's `.6e` rounding.
   - Rejections: label count of a group ≠ number of element resistances of
     its circuit; unknown `matrix_group`; two circuits on the same group;
     asymmetric or non-SPD `M_elem`; config `inductance` ≠ derived `L_A`
     beyond `rtol=1e-5` (decision 6; the tolerance covers the `.6e` rounding).
   - Unused group (`S_*` in CSV, no circuit) → warning, dropped.
   - Post-processing, voltage mode: `U == U_R + U_L_own + U_M_other` per
     element; `Σ_a U_a == u_A` to 1e-9 relative.
   - Backward compat: `examples/M9_M23072801/circuits.json` and
     `examples/M19/*.json` load and give the same `M` as before.

2. **Element resistances on `RLCircuitPID`.** In
   `load_resistance_from_csv`, detect the format from the header. For the
   element format: set `element_ids` and `element_resistance_funcs`, and
   build the circuit `resistance_func` from the `R[ohm]` column (single
   interpolator in the RHS, no per-element cost in the ODE). For the legacy
   format: `element_ids = []`, behaviour unchanged. Update the docstring
   (NumPy style).
   → verify: R-loader tests pass.

3. **Element inductance matrix in `CoupledRLCircuitsPID`.** Loader: read
   the CSV with `pandas.read_csv(path, index_col=0)`, check
   `index == columns`. In `_build_inductance_matrix`, the str branch selects
   for each circuit the rows/columns `<group>_*` in label order (count check
   here), drops unused groups with a warning, and assembles `self.M_elem`,
   `self.P` and `self.element_index` (`(circuit_id, element_id, label)` per
   row). It then sets `self.M = Pᵀ M_elem P`. Validate `M_elem` (symmetric,
   SPD with `eigvalsh`) before validating `M`. Add `matrix_group` to
   `CIRCUIT_CONFIG_KEYS`.
   → verify: assembly/rejection/summary tests pass.

4. **`element_voltages` method.** On `CoupledRLCircuitsPID`:
   `element_voltages(t, I, dI_dt) -> dict[circuit_id, dict[element_id,
   dict[str, ndarray]]]` with keys `U`, `U_R`, `U_L_own`, `U_M_other` [V].
   Vectorised over time: `U_L_own` uses the block-diagonal part of
   `M_elem P`, `U_M_other` the off-block part.
   → verify: identity tests pass on hand-built `I`, `dI/dt`.

5. **Wire into post-processing.** In both coupled branches of
   `VoltageInputPlottingStrategy.prepare_data` and
   `PIDControlPlottingStrategy.prepare_data`, when the system has
   `M_elem`: get `dI/dt` from the vector field at each `sol.t`
   (`voltage_vector_field(t, I)`, or `vector_field(t, y, di_ref_dt)[:N]`).
   Do not use `np.gradient`. Store the result under
   `circuits_data[cid]["elements"]`. Add these arrays to the saved results
   (npz/csv) under `"<cid>/<eid>/U"` etc.
   → verify: voltage-mode identity test end-to-end through `prepare_data`.

6. **Figure.** One figure, one subplot per circuit, `U_a(t)` for each
   element plus a dashed `Σ U_a` vs `u_A`. The three-term breakdown goes in
   the saved results only (decision 4).
   → verify: run on the M9 example (once real M CSVs exist), check
   `Σ U_a ≈ u_A` by eye.

7. **Docs.** README config section (`element_resistance_csv`, object form
   of `mutual_inductances`, with the CSV layouts above); theory doc: the
   `Pᵀ M_elem P` reduction and the three-term split.

## Decisions

Settled (2026-10-21):

1. Matrix input: one magnettools CSV for the whole site.
2. Group key: `matrix_group` (`"H"`/`"B"`/`"S"`).
3. Element resistances: auto-detected in `resistance_csv`.
4. Three-term breakdown: saved data only for now, no plot.
5. Group in the CSV that no circuit uses: warn and drop.

Still open (default applies unless changed):

6. **Circuit `inductance` in element mode** — default: *optional; if
   given it must equal the derived `L_A` within `rtol=1e-5`*. The
   alternative is *forbidden*.

## Out of scope (noticed, not changed)

- Comparing against measured voltage taps
  (`M9_Overview_*_Tensions_Aimant_ALL_internes_*.csv`). That needs a
  tap-to-element mapping; it could be added later through `experiment_data`.
- `load_single_circuit` path: element mode only for the coupled loader (a
  1-circuit config already goes through it, e.g. `M9Bitters_18MW.json`).
- The magnettools inductance CSV for M9_M23072801 isn't in the repo; step 6
  needs it (`inductances.py <site>.d --csv …`). Tube count vs. resistance count
  can only be confirmed then (magnetdb/duckdb query); the loader's count check
  covers it.
- One group split over several circuits (e.g. `B_0` and `B_1` on different
  supplies): not supported, one group per circuit.
