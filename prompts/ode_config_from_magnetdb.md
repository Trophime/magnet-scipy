# magnet-scipy configuration from magnetdb-duckdb-demo

## Goal

Pick **one overview record** (or one assembly at a date) in the DuckDB
database and produce a self-contained magnet-scipy case directory: circuits
json, R(I, T) tables, inductance matrix, reference currents, Tin series and
measured U/I. Uses:

- **replay**: simulate the run and compare with the measurements;
- **parameter identification**: the same directory gives the inputs of
  [parameter_identification.md](parameter_identification.md) (records, taps,
  element ↔ part mapping) and the a-priori values that the identified ones
  are compared with.

First move, as requested:

1. circuits json from the DB;
2. inductance matrix from the magnettools bindings;
3. R(I, T) from python_magnetworkflows.

## Relation to the existing DB design

`magnetdb-duckdb-demo/PLAN_digital_twin_schema.md` already has this as its
acceptance target: phase 4, `build_ode_config(<record>)` rebuilds
`examples/M9_M23072801/circuits-pid-temp.json` from the DB. That phase is
blocked on phases 1–3 (L1 PID tables, L2 run configuration, L3
`electrical_characteristics`) and S2 (signatures, `sync_info`). None of them
is implemented.

This plan is a **prototype of phase 4 that does not wait for them**. It reads
what the DB has today and computes or imports the rest. Every place where a
future table will replace a computation is marked **→ L1/L2/L3**. The output
format is the contract, so magnet-scipy doesn't change when the source
moves to the DB tables.

## What the DB has today (`magnetdb-2026-11-05.duckdb`, read-only check 2026-11-09)

| Need | Table / source | State |
|---|---|---|
| record → assembly, housing, t0, files | `overview_records` (1952) | ✓ e.g. `M9_Overview_230916-1755` → `M9_A230801_00`, M9, ECO, t0 2023-09-16 15:55 UTC, 1609 s, `sources_archive = M9_Archive_230916-1755.tdms`, `sources_pupitre = 2023.09.16 - 17:55:22.txt` |
| assembly → magnets | `assembly_magnets` | ✓ `M23072801` + `M9Bitters-newBi08` |
| magnet → parts, order | `magnet_parts.rank`, `coil_index` | ✓ insert: 14 helices `coil_index` 1–14 (+ 13 rings, no index); Bitters: `M9_newBi08` (1), `M9Be` (2) |
| GR role, channels at t0 | `housing_config_versions` + python_magnetrun `M*-housing-config.json` | ✓ (M9: insert GR1 = `IH`, Bitters GR2 = `IB`) |
| geometry → magnettools | `load_assembly_config_from_duckdb` → `prepare_geometry_directory` → `load_magnettools` (hoop-stress chain) | ✓ |
| flow law | `overview_records.flow_params` | **empty** (`'{}'`); `*-flow_params.json` in `apps/notebooks/` |
| reference currents, lags | `overview_records.signatures`, `sync_info` | **empty** (S2 not done) |
| PID gains | L1 tables | **missing**; 4 CIRRUS XML samples in `examples/M9_M23072801/`, parser `python_magnetrun/configAlims/convertxml.py` |
| sections (tap ↔ parts) | L2 `v_config_sections` | **missing**; rule known: insert section k = helices `coil_index` 2k−1, 2k |
| R, L, M | L3 `electrical_characteristics` | **missing** |

## Output: case directory (the contract)

```
<out>/<record>/
  circuits.json             # circuit level, loads with today's loader
  circuits-elements.json    # element mode, once element_voltages.md lands
  R_<magnet>.csv            # I[A], R_<elem>[ohm]..., R[ohm], Tinit[C]  (existing format)
  <assembly>_inductances.csv (+ _summary.csv, _labels.csv)   # magnettools
  reference_GR<n>.csv       # Overview Référence_GR<n>
  Tin_GR<n>.csv             # pupitre TinH/TinB, shifted by the clock lag
  measured/                 # Overview (1 Hz) and Archive (120 Hz) U, I, taps
  provenance.json           # see below
```

- **Circuit ids = DB magnet names** (`M23072801`, `M9Bitters-newBi08`), not
  hand-made names (`M9Bitters_18MW`).
- **Element ids**: Bitters → part names (`M9_newBi08`, `M9Be`). That
  removes the Be/Bi positional ambiguity noted in
  [parameter_identification.md](parameter_identification.md). Insert →
  section labels `H1H2 … H13H14` from `coil_index`, with the part names of
  each section in `provenance.json`.
- Each element also records its tap channel (`Ucoil<k>` / `Interne<k>` /
  `Externe<k>`) from the housing config, for the identification.
- `provenance.json`: DB file + sha256, record, assembly, housing-config
  version, source files, magnettools / python_magnetworkflows versions and
  options (cooling, heat correlation, friction, flow_params), and the clock
  lag used. Same purpose as L5 `simulations.inputs`.

## Steps

The code lives in `magnetdb-duckdb-demo/to_duckdb/` (decision 1). Tests
there run against `test-magnetdb.duckdb`; the DB is always opened
`read_only=True`.

1. **Record resolver (DB only, no data files).**
   `resolve_ode_case(con, record) -> OdeCase`: assembly, housing version at
   t0, magnets with GR role, parts per magnet with `coil_index`, sections,
   tap channels, current/Tin channel names, source file names.
   → verify: `M9_Overview_230916-1755` gives 2 circuits (insert GR1,
   Bitters GR2), 7 + 2 elements, taps `Ucoil1..7` / `Ucoil15..16`.

2. **Inductance (magnettools).** Implement
   [inductance_csv_from_duckdb.md](inductance_csv_from_duckdb.md) (not done
   yet) as a function called by the builder. Circuit level:
   `inductance` = `L_H`, `L_B`; `mutual_inductances` = `[M_HB]`. Element
   level: the full CSV + labels.
   → verify: matrix symmetric, SPD; compare with the hand values in
   `circuits-pid-temp.json` (`L = 0.002315`, `0.01254`, `M = 0.001965` H).
   → L3: later read from `electrical_characteristics` (source `magnettools`).

3. **R(I, T) (python_magnetworkflows).** This is the expensive step: a
   Feel++ run per (current step × Tin), in the singularity container, from a
   python_magnetsetup case built from the same geometry. It needs a
   `flow_params` per magnet, and `overview_records.flow_params` is empty.
   So it is a **separate, cached step**, not part of the builder:
   - `run_r_table(assembly, magnet, I_grid, Tin_grid, cooling, …)` → runs
     `python -m python_magnetworkflows.commissioning` (steplist = I grid)
     once per Tin, then converts the `measures.csv` files into
     `R_<magnet>.csv` (per element `U/I` → `R_<elem>[ohm]`, sum → `R[ohm]`,
     `Tinit[C]`).
   - Cache key: (magnet, geometry sha, I/Tin grids, model options,
     flow_params). The builder only looks the table up.
   - Before writing the converter, check the `measures.csv` column names
     (per-part `U_<part>`?) and which Tin parameter the cfg exposes.
   → verify: on `M9Bitters-newBi08` (Bitters case already in the README),
   rebuild `Rtot_M9Bitters_18MW.csv` within the FEM tolerance.
   Until this works, the builder **imports an existing CSV**
   (`--resistance-csv`), or uses the pupitre-header model (decision 2).
   → L3: tables in S3 + `electrical_characteristics` rows.

4. **PID.** Parse the CIRRUS XML in force at t0 with `convertxml.py`.
   Supplies come from the housing config (GR1 = A1 + A2, GR2 = A3 + A4),
   the jeu from the housing (17–20 = M7–M10). Write the magnet-scipy
   `pid_params`, with explicit thresholds (60 A; 1000 vs 800 A, DB open
   question #18).
   Checked on the 4 samples (current loop `numero=1`, jeu 19, seuils 1/2/3):

   | Supply | Kp | Ki | Kd (raw) | XML `version_config`, date |
   |---|---|---|---|---|
   | A1 (GR1) | 4 / 4 / 6 | 2 / 2 / 2 | −1 | 22, 2017-04-11 (file name says 24) |
   | A2 (GR1) | 1 / 0.05 / 0.05 | 0.2 | 1 / 5 / 5 | 24, 2017-04-11 |
   | A3 (GR2) | 5 / 12 / 12 | 0.2 / 1 / 1 | −1 | 44, 2019-06-25 (file name says 45) |
   | A4 (GR2) | 1 / 0.1 / 0.1 | 0.2 | 5 / 5 / 5 | 41, 2019-06-25 (file name says 42) |

   `circuits-pid-temp.json` gives the **insert (GR1)** A3's gains and the
   **Bitters (GR2)** A1's gains, so it looks swapped with respect to the
   XML (decision 4).
   → verify: the builder reproduces the table above for M9 at 2023-09-16.
   → L1.

5. **Time series (needs the data files).** From the record's sources, with
   python_magnetrun: Overview `Référence_GR<n>` → `reference_GR<n>.csv`;
   pupitre TinH/TinB → `Tin_GR<n>.csv`; Overview and Archive U, I, taps →
   `measured/`. The pupitre ↔ pigbrother clock lag is per file (+33 s seen on
   M9 2026-05-08). Estimate it by cross-correlating `IH` with
   `Courant_GR1` (python_magnetrun `_compute_lag_correlation`) and store
   it in `provenance.json`.
   → verify: |lag| reported; the shifted `IH` matches `Courant_GR1` to
   < 0.1 % on plateaus.
   → S2: later read from `sync_info`.

6. **Builder + CLI.**
   `magnetdb.py ode-config --record NAME --out DIR [--resistance-csv …]
   [--no-timeseries]`. Assembles 1–5 and writes the case directory.
   → **acceptance**: for `M9_Overview_230916-1755`, the case directory runs
   with `python -m magnet_scipy.coupled_main_refactor` and its
   `circuits.json` matches `examples/M9_M23072801/circuits-pid-temp.json` up
   to names and the R source.

magnet-scipy side: nothing new for `circuits.json`. `circuits-elements.json`
waits for [element_voltages.md](element_voltages.md). Check that the loader
tolerates a `provenance` key (it warns on unknown circuit keys), or keep
provenance in its own file only (as above).

## Decisions

Open:

1. **Where the code lives**: `to_duckdb/` script + CLI subcommand
   (recommended: same choice as the inductance plan; magnet-scipy stays free
   of DB/magnettools/Feel++ dependencies and only reads files), or a
   magnet-scipy optional module importing `to_duckdb`?
2. **R source for the first move**:
   (a) existing CSVs, imported;
   (b) python_magnetworkflows FEM tables (step 3, needs flow_params + container);
   (c) the pupitre-header monitoring model
   `(r0 + a0|I| + b0 I²)(1 + ct·(Tin + Tout)/2)`, per section, already in
   every pupitre file's line 1. It costs nothing, but it depends on `Tout`
   and magnet-scipy's R takes `(I, Tin)` only.
   Proposal: (a) now, (b) as the reference, (c) as a cross-check (it is
   also what method A of the identification should be compared with).
3. **Insert element ids**: section labels `H1H2…` (matches existing CSVs)
   or part names joined (`H15011301+H21091001`)?
4. **PID ↔ circuit on M9**: the XML puts `Kp 4/4/6, Ki 2` on A1 (GR1 =
   insert on M9) and `Kp 5/12/12, Ki 0.2/1/1` on A3 (GR2 = Bitters).
   `circuits-pid-temp.json` and `examples/M9_M23072801/README.md`
   ("Groupe1"/"Groupe2") have them the other way round. Which is right?
   Also: within a group, which supply's regulator sets the group current?
   A2/A4 have different gains and Kd ≠ −1 (`rapport_entre_boucle` = 0.45).
   And were these XML versions (2017/2019) still in force in 2023 (CIRRUS
   history not mirrored)?
5. **M19 is not in the DB**: no overview record on 2024-02-08, and the DB
   has M9 = `M23072801` + `M9Bitters-newBi08` and M10 = `M19071101` +
   `M10Bitters` active on that date, not Bitters-only assemblies. Proposal:
   `magnetdb-duckdb-demo/PLAN_m19_joint_operation.md`. It adds no M19
   housing or assembly; it uses housing-config versions for the test window
   and one record per housing. Waiting for the data volume.

## Out of scope

- The L1/L2/L3 tables themselves (PLAN_digital_twin_schema phases 1–3).
- Writing anything to the DB (no `simulations` rows yet).
- Supra parts (M8): R/L of supras depend on `IS`, not covered here.
