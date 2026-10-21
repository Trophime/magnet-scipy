# Inductance matrix CSV from the duckdb demo

Follow-up to [element_voltages.md](element_voltages.md). magnet-scipy's
element mode needs the site inductance matrix written by
`magnettools.inductances.compute_inductance_matrix` (`H_i`, `B_i`, `S_i`
labels, [H]). No such CSV exists yet for M9_M23072801. This plan produces it
from `magnetdb-duckdb-demo`, reusing the DB → magnettools chain that the hoop
stress pipeline already uses.

Work happens in `~/github/magnetdb-duckdb-demo/to_duckdb/`. magnet-scipy only
consumes the resulting CSV.

## What already exists

Hoop stress builds magnettools objects from a DB assembly in three calls
([compute_hoop_stats.py:761-802](../../magnetdb-duckdb-demo/to_duckdb/compute_hoop_stats.py),
same sequence in `stress_map._load_assembly`, `stress_map.py:1588`):

```python
housing, magnet_configs = load_assembly_config_from_duckdb(assembly_name, db_path, con=con)
assembly_config = {"name": assembly_name,
                   "magnets": [cfg for _name, cfg, _geo in magnet_configs]}
tmpdir = prepare_geometry_directory(assembly_name, assembly_config, db_path, con=con)
Tubes, Helices, OHelices, BMagnets, UMagnets, Shims = load_magnettools(assembly_config, tmpdir)
# … finally: shutil.rmtree(tmpdir, ignore_errors=True)
```

The magnettools objects are exactly the inputs of
`compute_inductance_matrix(Tubes, Helices, BMagnets, UMagnets, csvfile=…)`,
which also writes `<stem>_summary.csv` (`L_H`, `L_B`, `L_S`, `M_HB`, `M_HS`,
`M_BS`).

`build_part_column_map(assembly_name, db_path, con=con)`
([compute_hoop_stats.py:103](../../magnetdb-duckdb-demo/to_duckdb/compute_hoop_stats.py))
returns `{H1_fast: part_name, B1_fast: …}`. Parts are ordered by magnet
`commissioned_at DESC`, then `rank`, which is the same order
`load_assembly_config_from_duckdb` uses.

## Risks to check first

- **Positional ordering.** magnet-scipy maps the k-th `R_<id>[ohm]` column
  to `<group>_k`. The count check catches a wrong number of tubes or Bitters,
  but not a swap (e.g. `B_0`/`B_1` vs `M9_Be`/`M9_Bi`). Group totals
  (`L_H`, `M_HB`, …) are invariant under such swaps, so the summary
  cross-check won't catch it either. → write the label ↔ part-name sidecar
  (step 2) and check it by hand against the `R_*.csv` column order.
- **One H label per part?** `H_i` is one per Tube. The hoop code assumes
  one helix part per Tube (`stress_map.py:781`). Compare `len(Tubes)` with
  the number of helix parts from `build_part_column_map`, and likewise
  `len(BMagnets)` with the Bitter parts. A mismatch means magnettools
  splits differently from the DB parts; stop and look.
- **Assembly content.** `load_assembly_config_from_duckdb` returns all
  magnets of the assembly, commissioned or not. Check that the M9_M23072801
  assembly holds only the insert and the M9 Bitters.
- **magnettools version.** `magnettools.inductances` comes from
  `my-magnettools` commit 27c6652 (2026-10-12), which is in no tag yet. The
  released `python3-magnettools` may only have `magnettools.Mutual`
  (now a deprecation shim). Import `inductances`, fall back to `Mutual`.
- **Stale demo.** `demos/test-hoopstress.py` calls
  `compute_hoop_stress_history(site_name=…)`, but the parameter is now
  `assembly_name`. Don't copy it as a template.

## Plan

0. **Find the assembly.** Query `assemblies` / `assembly_magnets` for the
   name pairing M23072801 with M9Bitters_18MW (the demo DB uses `M9_%`
   names). Print each magnet's type and its parts in rank order.
   → verify: exactly one insert + one Bitter magnet; helix part count = 14
   helices; note how they group into the 7 resistance elements `H1H2…`.

1. **Script `to_duckdb/export_inductance_matrix.py`**:
   - CLI: `--assembly NAME --db PATH --csv OUT.csv [--geometries DIR]
     [--quiet]`.
   - Body: the three-call chain above, then
     `compute_inductance_matrix(..., csvfile=OUT, verbose=not quiet)`.
     Clean up tmpdir in `finally`. Open the DB `read_only=True`, since
     nothing is written.
   - Print `len(Tubes)`, `len(BMagnets)`, `len(UMagnets)`.
   - NumPy docstrings; PEP 8 names.
   → verify: running on M9_M23072801 writes `OUT.csv` and
   `OUT_summary.csv`; the matrix is symmetric and its eigenvalues are > 0.

2. **Label sidecar `<stem>_labels.csv`**: columns `label, magnet, part`,
   built from `build_part_column_map`, remapping `H1_fast → H_0`,
   `B1_fast → B_0`, `Supra1_fast → S_0`. If the H/B/S counts differ from
   `len(Tubes)/len(BMagnets)/len(UMagnets)`, warn and write no sidecar
   (risk 2).
   → verify: by hand, the sidecar order matches the `R_14Helices.csv` /
   `R_M9Bitters_18MW.csv` column order.

3. **Cross-check against the current magnet-scipy config.**
   `examples/M9_M23072801/circuits.json` has `L = 0.002315` (insert),
   `0.01254` (Bitters), `M = 0.001965` [H]. Compare with `L_H`, `L_B`,
   `M_HB` from `OUT_summary.csv`.
   → verify: agreement to about 1e-3 relative. If it doesn't agree, find out
   where the json values came from (other geometry or version) before going
   further.

4. **Test** in `to_duckdb/tests/test_export_inductance_matrix.py`, run
   against `test-magnetdb.duckdb`. Skip if magnettools can't be imported
   (see `checks.py`). Run the export on one assembly with geometry; check
   the labels' prefixes and order (`H_*`, then `B_*`, then `S_*`), that
   the matrix is symmetric, and that the diagonal is > 0.
   → verify: `pytest tests/test_export_inductance_matrix.py`.

5. **Hand-off to magnet-scipy.** Copy `OUT.csv` (+ `_summary`, `_labels`)
   to `examples/M9_M23072801/`, then add `circuits-elements.json`
   (`matrix_group`, `resistance_csv: R_*.csv`,
   `mutual_inductances: "<OUT>.csv"`). This unblocks step 6 of
   [element_voltages.md](element_voltages.md).

## Decisions

Settled (2026-10-21):

1. The code goes in a standalone `to_duckdb/export_inductance_matrix.py`,
   using only the public helpers. It is not added as a subcommand of
   `stress_map.py`.
2. Output is a CSV only for now. No DB table yet; revisit when several
   assemblies need it.

## Out of scope

- Generating the `R_*.csv` element resistance files from the DB. They
  already exist for M9_M23072801.
- Fixing `demos/test-hoopstress.py` (stale `site_name=` keyword).
- Tagging/releasing magnettools with the `inductances` module.
