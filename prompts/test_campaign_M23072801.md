# Test campaign — M23072801 (assembly `M9_A230801_00`)

## Goal

Pick a small set of **representative runs** of insert `M23072801` (with
Bitters `M9Bitters-newBi08`, housing M9). They are used to test magnet-scipy
against measurements:

- **replay**: closed-loop ODE driven by the GR references, compared with
  measured I, U, Tout;
- **identification**: R from plateaus (method A) and L from ramp corners
  (method B) in [parameter_identification.md](parameter_identification.md);
- **regression set**: a fixed list of cases that can be rerun when the model
  changes (R(I, T), cooling, PID, inductance).

Each case is meant to become one case directory produced by
`ode-config --record NAME` ([ode_config_from_magnetdb.md](ode_config_from_magnetdb.md), roadmap D1).

## Source

`magnetdb-duckdb-demo/to_duckdb/magnetdb-2026-11-05.duckdb`, opened read-only
on 2026-11-12. The SQL is in the [appendix](#appendix--selection-query).

| Table | Rows for `M9_A230801_00` | Used for |
|---|---|---|
| `experiments` | 282 (all `STATS DONE`), 2023-09-06 → 2024-08-04 | one row per pupitre file; key to `exp_run_scalars` and `exp_assembly_bin_stats` |
| `operationaldata` | 160 Overview, 282 Pupitre, 230 Archive, 114 Default, 5755 Spike (all `pending`) | which raw files exist |
| `overview_records` | 160 (92 ECO, 68 NORMAL); 16 have `merged_into` set | mode, t0, source file lists |

**Join rule.** `experiments.file` = one of `overview_records.sources_pupitre`,
keeping only records with `merged_into IS NULL`. One run can still map to two
records when the mode changes during the run (e.g. 2023.09.27 14:26:39,
NORMAL → ECO). 23 runs above 1 T have **no overview record** (pupitre only).

**Time zones.** `overview_records.t0` is UTC. Pupitre and Default file names
are local time (`M9_Overview_240716-2039` → t0 `2024-07-16 18:39`).

### What the DB does not give (yet)

- `overview_records.metrics`, `plateaux`, `debitbrut`, `signatures`,
  `sync_info` and `flow_params` are **empty** (`{}`) for all 160 records. The
  selection therefore uses `exp_run_scalars` (field, duration, energy) and
  `exp_assembly_bin_stats` (Field, Pmagnet, Ptot, teb, tsb, debitbrut per
  field bin of 5 T).
- `exp_part_bin_stats` is **not usable** here. It lists only helices
  `coil_index` 1–7 plus the two Bitters, and every part has the same `Icoil`
  max (29 981.6 A). The part → channel mapping needs checking first.
- Only one housing config version (`M9-181109`) covers the whole period, so
  the channel wiring is constant across all the runs below.

## Selection axes

| Axis | Range in the data | Why it matters for the model |
|---|---|---|
| Field setpoint | 1.9 → 36.1 T; plateaus at 29.99, 30.99, 31.99, 33.99, 34.99, 35.99 T | R(I) (heating, magnetoresistance), PID gains by threshold |
| Field-on duration | 4 min → 14.5 h | transient vs thermal steady state |
| Inlet water `teb` | 6 → 17 °C (≈ 9 °C in April 2024, ≈ 15 °C in Sept 2023 and July 2024) | R(T) through Tin |
| Mode | ECO / NORMAL | flow and cooling conditions |
| Epoch | commissioning (Sept–Oct 2023), April–May 2024, July–Aug 2024 | drift of R |
| Data | Archive 120 Hz, Default 50 Hz captures, triggers | needed for L (method B) and fast transients |

## Campaign

`B` = setpoint reached, `t≥30` = time with B ≥ 30 T [s], `teb` = mean
inlet water temperature with field on [°C], `Pmag` = max magnet power [MW].
Archive / Default / Spike = number of files listed by the overview record(s).

### C0 — Anchors (data already in `examples/M9_M23072801/`)

| Run (pupitre) | Overview | Mode | B [T] | on [s] | teb | Archive | Notes |
|---|---|---|---|---|---|---|---|
| 2023.09.16 - 17:55:22 | 230916-1755 | ECO | 31.1 | 1001 | 14.8 | 1 | D1 acceptance case; method A/B reference record |
| 2023.10.17 - 12:24:55 | 231017-1224 | ECO | 30.0 | 2015 | 13.9 | 1 | IB/IH, UB/UH CSV in examples |
| 2024.04.05 - 15:24:26 | 240405-1524 | ECO | 35.0 | 1054 | 10.8 | 1 | IB/IH, UB/UH CSV in examples |

Start here: these runs show whether a pipeline change reproduces the
existing example results.

### C1 — Low field, little heating (R0, L, small-signal PID)

| Run | Overview | Mode | B [T] | on [s] | teb | Archive | Default |
|---|---|---|---|---|---|---|---|
| 2023.09.06 - 11:15:51 | 230906-1017 | NORMAL | 6.2 | 3036 | 14.8 | 3 | 0 |
| 2023.09.06 - 16:11:12 | 230906-1459 | NORMAL | 10.2 | 1732 | 16.9 | 1 | 1 |
| 2023.10.12 - 16:31:31 | 231012-1629 | NORMAL | 4.1 | 908 | 16.2 | 1 | 0 |
| 2023.10.13 - 09:09:26 | 231013-0916 | NORMAL | 4.25 | 944 | 14.3 | 1 | 0 |

- 2023.09.06: first energisation, a staircase 1.9 → 6.2 → 10.2 T.
- 12–13 Oct 2023: about 12 short runs at ≈ 4.1 T. Two are kept to check
  **repeatability** (same setpoint, different days and teb).
- **All four runs are Bitters only**: the pupitre shows IH = 0, and their
  Overview files only record GR2 (`Courant_A3/A4`). They are single-circuit
  cases. 41 of the 160 Overview files of the assembly are GR2-only like
  these, and 5 are GR1-only. Open point: UH reaches 13 V on 12–13 Oct with
  IH = 0, more than M·dI_B/dt at these ramp rates.

### C2 — Short ramps to top field (inductive transient, PID, ramp heating)

| Run | Overview | Mode | B [T] | on [s] | teb | Archive | Notes |
|---|---|---|---|---|---|---|---|
| 2023.09.11 - 14:21:59 | 230911-1343 | ECO | 35.0 | 755 | 15.9 | 1 | first 35 T; pupitre in examples |
| 2024.04.16 - 15:28:04 | 240416-1528 | ECO | 36.1 | 1074 | 9.5 | 1 | highest field of the assembly's life, 8 triggers |
| 2024.07.16 - 11:41:36 | 240716-1141 | ECO | 36.0 | 769 | 14.5 | 1 | same setpoint as above, warm water |

### C3 — ECO vs NORMAL at the same setpoint (34.61 T)

| Run | Overview | Mode | B [T] | on [s] | teb | Archive | P/B² in 30–35 T bin |
|---|---|---|---|---|---|---|---|
| 2023.09.27 - 14:54:06 | 230927-1453 | ECO | 34.61 | 851 | 15.1 | 1 | 0.01665 |
| 2023.10.04 - 12:12:11 | 231004-1023 | NORMAL | 34.61 | 1047 | 15.8 | 1 | 0.01680 |

One week apart, same setpoint, teb within 0.7 K: the cleanest mode
comparison in the data.

### C4 — Long plateaus (thermal steady state, Tin/Tout drift)

| Run | Overview | Mode | B [T] | t≥30 [s] | teb | Archive | Notes |
|---|---|---|---|---|---|---|---|
| 2024.05.10 - 14:41:07 | 240510-1441 | ECO | 32.0 | 34 885 | 12.3 | 2 | 9.7 h at 32 T, almost no ramps: the purest plateau |
| 2024.07.15 - 09:07:12 | 240715-0907 | ECO | 35.2 | 18 454 | 14.2 | 2 | long and hot (tsb max 33.6 °C) |
| 2023.09.15 - 20:07:01 | 230915-2006 | ECO | 34.0 | 4 199 | 15.1 | 1 | 7.4 h with field on, many setpoint changes; pupitre in examples |

These runs test the cooling model ([cooling_model.md](cooling_model.md)). At 1 Hz
the pupitre is enough (method A); the Archive is only needed for the ramp
corners.

### C5 — Inlet water temperature at fixed setpoint (R(T))

| Run | Overview | B in 35–40 T bin | teb | Pmag (bin mean) | P/B² |
|---|---|---|---|---|---|
| 2024.04.17 - 14:41:38 | 240417-1441 | 35.57 | 9.0 | 22.38 | 0.01769 |
| 2024.07.25 - 10:44:14 | 240725-1044 | 35.50 | 14.1 | 22.68 | 0.01800 |

ΔP/B² ≈ +1.8 % for Δteb ≈ +5 K, which is the order expected from R(T) alone.
The model with measured Tin must reproduce this difference **without**
re-fitting R. Season and epoch are confounded here: no epoch has a wide teb
range.

### C6 — Drift between epochs at equal teb

| Run | Overview | B in 30–35 T bin | teb | Pmag (bin mean) | P/B² |
|---|---|---|---|---|---|
| 2023.09.14 - 21:43:07 | 230914-2143 | 32.47 | 15.3 | 17.12 | 0.01624 |
| 2024.07.26 - 11:22:11 | 240726-1132, 240726-1334 | 31.79 | 15.7 | 16.30 | 0.01613 |

Same 33.99 T setpoint, ten months apart. The bin means show **no clear
drift** (−0.7 %). The 5 T bin mixes ramps and plateaus, so method A on the
plateau tails has to confirm it. If confirmed, one R(I, T) table is valid for
the whole life of the assembly.

### C7 — Events with Default (50 Hz) captures

| Run | Overview | Mode | B [T] | Default files (local time) | Archive |
|---|---|---|---|---|---|
| 2023.09.19 - 19:08:11 | 230919-1906 | NORMAL | 26.4 | 2 (19:38:27, 19:38:57) | 1 |
| 2024.07.19 - 17:10:41 | 240719-1710 | ECO | 35.9 | 3 (20:08:49 → 20:11:34) | 2 |
| 2024.07.16 - 21:08:14 | 240716-2039 | ECO | 35.9 | **65** from 22:15:13 | 1 |

A Default file is a 50 Hz current capture. It does **not** by itself prove a
trip. Check each event in the Archive (I, U around the capture time) before
it becomes a test. 2024.07.16 needs a look first: 65 captures in one run.

### C8 — Pupitre only (no overview record)

| Run | B [T] | on [s] | teb | Notes |
|---|---|---|---|---|
| 2024.07.17 - 17:43:46 | 35.9 | 28 059 | 14.8 | no references, no Archive: tests the fallback where the reference is rebuilt from `Idcct` |

### C9 — Spikes on the insert, no Default

| Run | Overview | Mode | B [T] | on [s] | teb | Pmag | Archive | Default | Spike |
|---|---|---|---|---|---|---|---|---|---|
| 2024.08.04 - 09:12:54 | 240804-0912 | ECO | 33.14 | 14 714 | 15.0 | 18.4 | 1 | 0 | 936 |

A Spike file is a 1 s capture at 4.8 kHz. Spike recording starts on
2024-07-18, so only the last 19 overview records of the assembly can have
any. Records with Spike captures and **no** Default: 240718-1428,
240721-0950 (3 spikes), 240724-1636, 240725-1044 (C5), 240731-0925,
240801-0911, 240802-0914, 240803-0859, 240804-0912.

How the spikes were attributed (2026-11-12, raw tdms):

- Each capture: every analog channel high-passed (minus a 5 ms moving mean),
  score = max|hp| / robust σ (MAD). From 240724 on, **every** capture peaks on
  the insert tap `Interne7` (median z ≈ 75–150). 240718-1428 is mixed
  (several taps, some weak), so it is not used.
- 25 random 240804 captures: `Interne7` jumps by 1.5–3.3 V (almost always
  positive), ms long; the HT phases at the same instant stay at noise level
  (median z 2.4, none above 10). The spikes are not mains-induced.

Why 240804: most spikes (936), shortest clean candidate, last run of the
assembly. 528 spikes fall in t = 5400–7200 s; median I_GR1 at a spike is
27.4 kA (90 % above 23.6 kA). The list of Spike files is in
`provenance.json` (`spike`).

Expected use: the **detection limit** of the 1 Hz lumped comparison in
[signal_residual_analysis.md](signal_residual_analysis.md). On 240719 (C7,
same `Interne7` spikes) neither U/I nor I − ref at 1 Hz shows anything during
a burst of 186 spikes in 10 min, so no 1 Hz window is expected here either.

Caveats:

- Spike timing: the file `datetime` is ≈ 6.8 s before `wf_start_time`, and
  `wf_start_time` differs by up to 0.6 s between groups. Use the
  `Tensions_Aimant` start time for the spike instant. Placing spikes in the
  run from the (local-time) file name is good to about ±7 s.
- C5 240725-1044 (544) and C6 240726-1334 (186) carry the same `Interne7`
  spikes: they are not spike-free references.

## Excluded — check before use

| Runs | Observation | Question |
|---|---|---|
| 2024.02.08 - 09:07:52, 12:54:21 (no overview record) | Ptot 27.9 / 33.8 MW but Pmagnet 13.1 / 17.0 MW at 10.7 / 12.2 T; teb 6 °C; debitbrut ≈ 1575 | Is the assembly attribution right, or was another load on the converters? |
| 10 morning runs 2024.04.10 → 04.26 at 5.38 T (no overview record) | 2–5.6 h each; Pmagnet 5.9 MW **>** Ptot 3.7 MW | Single-supply operation or a channel issue? |
| 2024.06.25 - 15:00:10, 15:19:17 (no overview record) | 24.6 / 23.4 T, short | Usable as C8 if C8 needs a second case |
| `exp_part_bin_stats` | see above | Part → `Icoil`/`Ucoil` mapping |

## Run list (for scripts)

```
2023.09.16 - 17:55:22    C0
2023.10.17 - 12:24:55    C0
2024.04.05 - 15:24:26    C0
2023.09.06 - 11:15:51    C1
2023.09.06 - 16:11:12    C1
2023.10.12 - 16:31:31    C1
2023.10.13 - 09:09:26    C1
2023.09.11 - 14:21:59    C2
2024.04.16 - 15:28:04    C2
2024.07.16 - 11:41:36    C2
2023.09.27 - 14:54:06    C3
2023.10.04 - 12:12:11    C3
2024.05.10 - 14:41:07    C4
2024.07.15 - 09:07:12    C4
2023.09.15 - 20:07:01    C4
2024.04.17 - 14:41:38    C5
2024.07.25 - 10:44:14    C5
2023.09.14 - 21:43:07    C6
2024.07.26 - 11:22:11    C6
2023.09.19 - 19:08:11    C7
2024.07.19 - 17:10:41    C7
2024.07.16 - 21:08:14    C7  (check first)
2024.07.17 - 17:43:46    C8
2024.08.04 - 09:12:54    C9
```

24 runs, 23 with an overview record.

## Case directories

`examples/build_M23072801_campaign.py` writes one directory per group,
`examples/M9_M23072801_C<k>_<name>/`, with for each run (`<tag>` =
`YYMMDD-HHMMSS` of the pupitre file):

- `circuits-<tag>.json` (coupled, both GRs powered) or
  `circuit-GR2-<tag>.json` (single circuit, C1);
- `<tag>_reference_GR*.csv`, `<tag>_Tin_GR*.csv`,
  `<tag>_measured_{I,U}_GR*.csv`;
- per group: `README.md` (table, inputs, commands), `run.sh`,
  `provenance.json` (DB sha256, source files, UTC window, gaps), `Rtot_*.csv`.

```bash
../magnetdb-duckdb-demo/.venv/bin/python examples/build_M23072801_campaign.py \
    --db ../magnetdb-duckdb-demo/to_duckdb/magnetdb-2026-11-05.duckdb \
    --data-root ~/LNCMIG-Data/records [--groups C0 C2]
```

Choices made by the builder:

- R tables, L, M and PID gains are those of
  `examples/M9_M23072801/circuits-pid-temp.json` for every run; circuit ids
  stay `M23072801` / `M9Bitters_18MW`.
- Pupitre (local time) and Overview (UTC) are aligned on absolute time;
  time 0 = start of their overlap. The ±1 s residual lag is not corrected.
- Tin: the pupitre rounds `Tin1`/`Tin2` to 0.1 °C, which made R(I, T)
  jump by 3.6 × 10⁻⁴ at each 0.1 °C flip. The file holds the taut string
  inside the ±0.05 °C band around the recorded values (smoothest curve
  within the resolution). A fixed Savitzky–Golay window does not fit all
  runs: 301 samples still wiggles on quiet runs (C1, C4) and departs by up
  to 0.6 K from the data on the fast Tin rises (C0, C2, C3, C5). Duplicate
  pupitre timestamps (same second written twice) are dropped.
- Max solver step: 0.01 s up to a 3600 s window, 0.1 s above. Measured
  cost: C2 230911-142159 (857 s, coupled, 0.01 s) takes 42 min.
- C8 and any run whose Overview misses an active GR: reference = measured
  pupitre current.

Known limits: C6 2024.07.26 starts at 4.7/1.6 kA (Overview starts 10 min
after the pupitre file) and has 5 gaps up to 131 s between its Overview
files; C1 2023.10.13 starts at 5 kA for the same reason.

## Steps

1. Pull the raw files for the 23 runs (pupitre, Overview, Archive, Default)
   → verify: every file listed in `operationaldata` is readable.
2. Run D1 `ode-config --record` on C0 → verify: rebuilds the existing
   `examples/M9_M23072801/circuits-pid-temp.json`.
3. Method A on the C1, C4, C5, C6 plateaus → verify: R per GR with CI; C6
   answers the drift question.
4. Method B on C1, C2 and C3 (Archive) → verify: L, M vs magnettools.
5. Closed-loop replay of every case with fixed R, L → per-case error
   report; this list becomes the regression set.
6. C7 only after each event has been checked in the Archive.
7. C9 with the residual analysis ([signal_residual_analysis.md](signal_residual_analysis.md))
   → verify: no 1 Hz window during the spike bursts (detection limit).

## Open decisions

1. **Pass criteria** for the replay (error on U per GR on plateaus, on I
   during ramps, on Tout). To fix once C0 gives a baseline.
2. **Mode ECO vs NORMAL**: what does the mode change physically (flow, pump
   configuration)? It decides whether C3 is a cooling test or a
   converter test.
3. The two excluded groups (2024.02.08, April 2024 at 5.38 T): ask the
   operators before including or dropping them.
4. Add a roadmap line (V-section) for this campaign?

## Appendix — selection query

Run with the `magnetdb-duckdb-demo` venv (`duckdb` 1.5.6), read-only.

```sql
with e as (select * from experiments where assembly_name = 'M9_A230801_00'),
sc as (
  select experiment_id,
         max(value) filter (where channel = 'duration_s')          as dur,
         max(value) filter (where channel = 'duration_field_on_s') as dur_on,
         max(value) filter (where channel = 'energy_j') / 3.6e9    as e_mwh,
         max(value) filter (where channel = 'field_max')           as bmax
  from exp_run_scalars group by 1),
b as (
  select experiment_id,
         max(max_x) filter (where channel = 'Pmagnet') as pmag_max,
         max(max_x) filter (where channel = 'Ptot')    as ptot_max,
         sum(sum_dt) filter (where channel = 'Field' and field_bin_low >= 30) as t_ge30,
         sum(sum_dt) filter (where channel = 'Field' and field_bin_low >= 20) as t_ge20,
         sum(sum_x_dt) filter (where channel = 'teb' and field_bin_low >= 0.1)
           / nullif(sum(sum_dt) filter (where channel = 'teb' and field_bin_low >= 0.1), 0) as teb_on,
         max(max_x) filter (where channel = 'tsb') as tsb_max,
         sum(sum_x_dt) filter (where channel = 'debitbrut' and field_bin_low >= 0.1)
           / nullif(sum(sum_dt) filter (where channel = 'debitbrut' and field_bin_low >= 0.1), 0) as q_on
  from exp_assembly_bin_stats group by 1),
o as (
  select filename as ovr, mode, t0, unnest(sources_pupitre) as pup,
         len(sources_archive) as n_arch, len(sources_default) as n_def,
         len(sources_trigger) as n_trg
  from overview_records
  where assembly_name = 'M9_A230801_00' and merged_into is null)
select e.name, string_agg(o.ovr, ',') as ovr, string_agg(distinct o.mode, '/') as mode,
       sum(o.n_arch) as n_arch, sum(o.n_def) as n_def, sum(o.n_trg) as n_trg,
       any_value(sc.dur) as dur, any_value(sc.dur_on) as dur_on,
       any_value(sc.bmax) as bmax, any_value(sc.e_mwh) as e_mwh,
       any_value(b.pmag_max) as pmag, any_value(b.ptot_max) as ptot,
       any_value(b.t_ge20) as t20, any_value(b.t_ge30) as t30,
       any_value(b.teb_on) as teb_on, any_value(b.tsb_max) as tsb_max,
       any_value(b.q_on) as q_on
from e join sc on sc.experiment_id = e.id
       join b on b.experiment_id = e.id
       left join o on o.pup = e.file
group by e.name
order by e.name;
```

The bin means of C3, C5 and C6 come from the same table, restricted to the
highest field bin reached by each run (`field_bin_low` = 30 or 35).
