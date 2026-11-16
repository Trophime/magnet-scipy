"""Build the M23072801 test-campaign case directories.

One directory per group of ``prompts/test_campaign_M23072801.md`` is written
under ``examples/``. Each run of a group gets its reference currents, inlet
temperatures, measured currents/voltages (Overview 1 Hz and, when the run
has one, Archive 120 Hz) and a coupled-circuit config that
``python -m magnet_scipy.coupled_main_refactor`` loads as is.

Requires ``duckdb``, ``nptdms``, ``pandas`` (e.g. the magnetdb-duckdb-demo
venv). The DB is opened read-only.

Example
-------
::

    ../magnetdb-duckdb-demo/.venv/bin/python examples/build_M23072801_campaign.py \\
        --db ../magnetdb-duckdb-demo/to_duckdb/magnetdb-2026-11-05.duckdb \\
        --data-root ~/LNCMIG-Data/records
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from nptdms import TdmsFile

ASSEMBLY = "M9_A230801_00"
SITE_TZ = "Europe/Paris"  # pupitre timestamps are local time; pigbrother is UTC
EXAMPLES = Path(__file__).resolve().parent
BASE_CASE = EXAMPLES / "M9_M23072801"
BASE_CONFIG = "circuits-pid-temp.json"
R_TABLES = ("Rtot_14Helices.csv", "Rtot_M9Bitters_18MW.csv")
LONG_RUN_S = 3600.0  # above this, use a coarser max solver step
ACTIVE_CURRENT_A = 100.0  # a GR whose pupitre current never exceeds this is off
GAP_S = 5.0  # holes in the time series longer than this are reported
ARCHIVE_FLOAT_FORMAT = "%.10g"  # keeps the 1/120 s time step up to ~1e5 s

# GR1 = insert (circuit 0 of the base config), GR2 = Bitters (circuit 1).
GROUPS = ("GR1", "GR2")
# Overview channel names, after replacing spaces by "_" (both spellings occur).
OVERVIEW_REFERENCE = {
    "GR1": ("Référence_A1", "Référence_A2"),
    "GR2": ("Référence_A3", "Référence_A4"),
}
OVERVIEW_CURRENT = {"GR1": "Courant_GR1", "GR2": "Courant_GR2"}
OVERVIEW_VOLTAGE = {"GR1": "ALL_internes", "GR2": "ALL_externes"}
PUPITRE_CURRENT = {"GR1": ("Idcct1", "Idcct2"), "GR2": ("Idcct3", "Idcct4")}
PUPITRE_VOLTAGE = {
    "GR1": tuple(f"Ucoil{k}" for k in range(1, 15)),
    "GR2": ("Ucoil15", "Ucoil16"),
}
PUPITRE_TIN = {"GR1": "Tin1", "GR2": "Tin2"}


@dataclass
class Group:
    """One group of the campaign."""

    key: str
    dirname: str
    title: str
    purpose: str
    runs: list[str] = field(default_factory=list)


CAMPAIGN = [
    Group("C0", "M9_M23072801_C0_anchors", "Anchors",
          "Runs whose data is already in `M9_M23072801`; checks that the pipeline "
          "reproduces the existing example.",
          ["2023.09.16 - 17:55:22", "2023.10.17 - 12:24:55", "2024.04.05 - 15:24:26"]),
    Group("C1", "M9_M23072801_C1_low_field", "Low field, little heating",
          "Bitters alone: the insert is not powered (IH = 0) in these runs, so they "
          "are single-circuit cases for R0, L and small-signal PID of the Bitters. "
          "The two 4.1 T runs check repeatability.",
          ["2023.09.06 - 11:15:51", "2023.09.06 - 16:11:12",
           "2023.10.12 - 16:31:31", "2023.10.13 - 09:09:26"]),
    Group("C2", "M9_M23072801_C2_ramps_top_field", "Short ramps to top field",
          "Inductive transient, PID and ramp heating up to 35-36 T.",
          ["2023.09.11 - 14:21:59", "2024.04.16 - 15:28:04", "2024.07.16 - 11:41:36"]),
    Group("C3", "M9_M23072801_C3_eco_vs_normal", "ECO vs NORMAL at 34.61 T",
          "Same setpoint one week apart, teb within 0.7 K; only the mode differs.",
          ["2023.09.27 - 14:54:06", "2023.10.04 - 12:12:11"]),
    Group("C4", "M9_M23072801_C4_long_plateaus", "Long plateaus",
          "Thermal steady state and Tin/Tout drift (cooling model).",
          ["2024.05.10 - 14:41:07", "2024.07.15 - 09:07:12", "2023.09.15 - 20:07:01"]),
    Group("C5", "M9_M23072801_C5_water_temperature", "Inlet water temperature",
          "Same 36 T setpoint at teb 9 and 14 C: R(T) must explain dP/B2 = +1.8 % "
          "without re-fitting R.",
          ["2024.04.17 - 14:41:38", "2024.07.25 - 10:44:14"]),
    Group("C6", "M9_M23072801_C6_drift", "Drift between epochs",
          "Same 33.99 T setpoint and teb, ten months apart: is one R(I, T) table "
          "valid for the whole life of the assembly?",
          ["2023.09.14 - 21:43:07", "2024.07.26 - 11:22:11"]),
    Group("C7", "M9_M23072801_C7_default_events", "Default (50 Hz) events",
          "Runs with Default captures. Check each event in the Archive before "
          "using it as a test; 2024.07.16 has 65 captures.",
          ["2023.09.19 - 19:08:11", "2024.07.19 - 17:10:41", "2024.07.16 - 21:08:14"]),
    Group("C8", "M9_M23072801_C8_pupitre_only", "Pupitre only",
          "No overview record: the reference is the measured pupitre current "
          "(IH, IB), so this tests the fallback path, not the PID.",
          ["2024.07.17 - 17:43:46"]),
    Group("C9", "M9_M23072801_C9_insert_spikes", "Spikes on the insert, no Default",
          "936 Spike captures (1 s, 4.8 kHz), all on the insert helix tap `Interne7` "
          "(1.5-3.3 V, ms long, no coincident HT disturbance), and no Default capture: "
          "spike-only case. Spike times are in `provenance.json`.",
          ["2024.08.04 - 09:12:54"]),
]


def run_tag(pupitre_name: str) -> str:
    """Return a filesystem-safe tag ``YYMMDD-HHMMSS`` for a pupitre run name."""
    return pd.to_datetime(pupitre_name, format="%Y.%m.%d - %H:%M:%S").strftime("%y%m%d-%H%M%S")


def to_utc(local: pd.Series) -> pd.Series:
    """Convert naive site-local timestamps to naive UTC."""
    aware = local.dt.tz_localize(SITE_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return aware.dt.tz_convert("UTC").dt.tz_localize(None)


def sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pupitre(path: Path) -> pd.DataFrame:
    """Read a pupitre file and add the absolute UTC time.

    Parameters
    ----------
    path : Path
        Pupitre ``.txt`` file (tab separated, ``Date`` and ``Time`` columns).

    Returns
    -------
    :class:`~pandas.DataFrame`
        Pupitre columns plus ``utc`` (naive UTC timestamps), rows without a
        valid timestamp dropped. The pupitre sometimes writes the same second
        twice; only the first row is kept.
    """
    df = pd.read_csv(path, sep="\t", skiprows=1)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    local = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")
    df["utc"] = to_utc(local)
    df = df.dropna(subset=["utc"]).drop_duplicates("utc")
    return df.reset_index(drop=True)


def _funnel(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Shortest path from (x[0], lo[0]) to (x[-1], lo[-1]) through the gates [lo, hi]."""
    px, py = [x[0]], [lo[0]]
    ax, ay = x[0], lo[0]
    i, smax, smin, iu, il = 1, np.inf, -np.inf, 0, 0
    while i < len(x):
        dx = x[i] - ax
        su, sl = (hi[i] - ay) / dx, (lo[i] - ay) / dx
        if sl > smax:  # gate i is above the funnel: bend on the upper vertex
            ax, ay, i = x[iu], hi[iu], iu + 1
        elif su < smin:  # gate i is below the funnel: bend on the lower vertex
            ax, ay, i = x[il], lo[il], il + 1
        else:
            if su <= smax:
                smax, iu = su, i
            if sl >= smin:
                smin, il = sl, i
            i += 1
            continue
        px.append(ax)
        py.append(ay)
        smax, smin = np.inf, -np.inf
    px.append(x[-1])
    py.append(lo[-1])
    return np.interp(x, px, py)


def dequantize(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Reconstruct a slowly varying signal recorded with a coarse resolution.

    The recorded values are rounded to a step ``q``, read from the data as
    the smallest non-zero change. The true signal is then within ``q / 2``
    of every sample; the taut string is the shortest (and smoothest, for any
    convex measure of the slope) curve inside that band. Free ends are
    obtained by mirroring the series on both sides, which gives zero slope
    at the ends.

    Parameters
    ----------
    x : :class:`~numpy.ndarray`
        Strictly increasing sample times [s].
    y : :class:`~numpy.ndarray`
        Recorded values, e.g. pupitre ``Tin`` [°C] (0.1 °C steps).

    Returns
    -------
    :class:`~numpy.ndarray`
        Reconstructed values at ``x``, within ``q / 2`` of ``y``.
    float
        Half band ``q / 2`` used; 0 if ``y`` is constant (returned as is).
    """
    steps = np.abs(np.diff(y))
    steps = steps[steps > 1e-9]
    if steps.size == 0:
        return y.astype(float), 0.0
    delta = round(float(steps.min()), 6) / 2
    xe = np.concatenate([2 * x[0] - x[:0:-1], x, 2 * x[-1] - x[-2::-1]])
    ye = np.concatenate([y[:0:-1], y, y[-2::-1]]).astype(float)
    n = len(y)
    return _funnel(xe, ye - delta, ye + delta)[n - 1:2 * n - 1], delta


def read_pigbrother(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate the channels needed from pigbrother tdms files.

    Overview (1 Hz) and Archive (120 Hz) files have the same groups and
    channel names; only the channels used are read from disk.

    Parameters
    ----------
    paths : list of Path
        Overview or Archive ``.tdms`` files of the run.

    Returns
    -------
    :class:`~pandas.DataFrame`
        Columns ``utc``, ``ref_GR*``, ``I_GR*`` [A] and ``U_GR*`` [V], sorted
        by time, duplicated timestamps removed. A GR whose channels are not in
        a file (single-group recordings) is NaN for that file. Files without
        channels are skipped.
    """
    frames = []
    for path in paths:
        with TdmsFile.open(path) as tdms:
            if "Courants_Alimentations" not in [g.name for g in tdms.groups()]:
                continue  # empty file (aborted acquisition)
            cur = {c.name.replace(" ", "_"): c for c in tdms["Courants_Alimentations"].channels()}
            mag = {c.name.replace(" ", "_"): c for c in tdms["Tensions_Aimant"].channels()}
            ch0 = next(cur[OVERVIEW_CURRENT[gr]] for gr in GROUPS if OVERVIEW_CURRENT[gr] in cur)
            start = pd.Timestamp(ch0.properties["wf_start_time"])
            dt = float(ch0.properties["wf_increment"])
            n = len(ch0)
            data = {"utc": start + pd.to_timedelta(np.arange(n) * dt, unit="s")}
            for gr in GROUPS:
                if OVERVIEW_CURRENT[gr] not in cur:
                    for k in ("ref", "I", "U"):
                        data[f"{k}_{gr}"] = np.full(n, np.nan)
                    continue
                data[f"ref_{gr}"] = sum(cur[c][:n].astype(float) for c in OVERVIEW_REFERENCE[gr])
                data[f"I_{gr}"] = cur[OVERVIEW_CURRENT[gr]][:n].astype(float)
                data[f"U_{gr}"] = mag[OVERVIEW_VOLTAGE[gr]][:n].astype(float)
        frames.append(pd.DataFrame(data))
    df = pd.concat(frames).sort_values("utc")
    return df.drop_duplicates("utc").reset_index(drop=True)


def write_series(path: Path, t: np.ndarray, y: np.ndarray, name: str,
                 float_format: str = "%.6g") -> None:
    """Write a ``time,<name>`` CSV."""
    pd.DataFrame({"time": t, name: y}).to_csv(path, index=False, float_format=float_format)


def _archive_starts(files: dict[str, str]) -> pd.Series:
    """Return the naive UTC start of every Archive file, read from its name (minute, local time)."""
    names = [n for n in files if "_Archive_" in n]
    stamps = [n.rsplit("_", 1)[1].removesuffix(".tdms") for n in names]
    return to_utc(pd.Series(pd.to_datetime(stamps, format="%y%m%d-%H%M"), index=names))


def _gaps(t: np.ndarray) -> list[list[float]]:
    """Return ``[t_start, length]`` [s] of the holes longer than ``GAP_S`` in ``t``."""
    dt = np.diff(t)
    return [[float(t[i]), float(dt[i])] for i in np.flatnonzero(dt > GAP_S)]


@dataclass
class RunCase:
    """What a group README and run script need about one built run."""

    name: str
    tag: str
    config: str
    active: list[str]
    value_start: list[float]
    time_end: float
    time_step: float
    sources: dict


def _pupitre_series(pup: pd.DataFrame, gr: str) -> dict[str, np.ndarray]:
    """Return reference (= measured current), current [A] and voltage [V] of a GR from the pupitre."""
    i = pup[list(PUPITRE_CURRENT[gr])].sum(axis=1).to_numpy()
    u = pup[list(PUPITRE_VOLTAGE[gr])].sum(axis=1).to_numpy()
    return {"ref": i, "I": i, "U": u}


def build_run(con, run: str, data_root: Path, out: Path, base: dict) -> RunCase:
    """Write the input files and config of one run.

    A GR is active when its pupitre current exceeds ``ACTIVE_CURRENT_A``.
    With two active GRs the config is the coupled one; with one, it is the
    single-circuit config of that GR. Signals come from the Overview when it
    records every active GR, otherwise from the pupitre (the reference is
    then the measured current). When the overview records list Archive
    files, the 120 Hz measured current and voltage are also written, on the
    same window and time 0; the config does not use them.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Read-only connection to the magnetdb DuckDB file.
    run : str
        Pupitre run name, e.g. ``"2024.05.10 - 14:41:07"``.
    data_root : Path
        Root of the raw records (``operationaldata.file`` is relative to it).
    out : Path
        Group directory.
    base : dict
        Base coupled-circuit config (L, M, PID, R tables).

    Returns
    -------
    RunCase
        Config name, active GRs, initial currents [A], time window [s] and
        sources.
    """
    tag = run_tag(run)
    files = dict(con.execute(
        "select regexp_extract(file, '[^/]+$'), file from operationaldata where assembly_name = ?",
        [ASSEMBLY]).fetchall())
    records = con.execute(
        "select filename, sources_overview, sources_archive, sources_default, mode, sources_spike "
        "from overview_records where merged_into is null and list_contains(sources_pupitre, ?) "
        "order by t0", [f"{run}.txt"]).fetchall()

    pup = read_pupitre(data_root / files[f"{run}.txt"])
    active = [gr for gr in GROUPS
              if pup[list(PUPITRE_CURRENT[gr])].sum(axis=1).max() > ACTIVE_CURRENT_A]
    ov_names = [n for r in records for n in r[1]]

    ov = read_pigbrother([data_root / files[n] for n in ov_names]) if ov_names else None
    use_overview = ov is not None and all(ov[f"I_{gr}"].notna().any() for gr in active)
    if use_overview:
        ov = ov.dropna(subset=[f"I_{gr}" for gr in active])
        t0 = max(pup.utc.iloc[0], ov.utc.iloc[0])
        t1 = min(pup.utc.iloc[-1], ov.utc.iloc[-1])
        ov = ov[(ov.utc >= t0) & (ov.utc <= t1)]
        t = (ov.utc - t0).dt.total_seconds().to_numpy()
        series = {gr: {k: ov[f"{k}_{gr}"].to_numpy() for k in ("ref", "I", "U")}
                  for gr in active}
    else:
        t0, t1 = pup.utc.iloc[0], pup.utc.iloc[-1]
        t = (pup.utc - t0).dt.total_seconds().to_numpy()
        series = {gr: _pupitre_series(pup, gr) for gr in active}

    # Tin: whole pupitre file, so interpolation covers the window edges.
    t_pup = (pup.utc - t0).dt.total_seconds().to_numpy()

    tin_delta = {}
    circuits = []
    for circuit, gr in zip(json.loads(json.dumps(base["circuits"])), GROUPS):
        if gr not in active:
            continue
        ref, tin = f"{tag}_reference_{gr}.csv", f"{tag}_Tin_{gr}.csv"
        i_meas, u_meas = f"{tag}_measured_I_{gr}.csv", f"{tag}_measured_U_{gr}.csv"
        write_series(out / ref, t, series[gr]["ref"], "current")
        tin_values, tin_delta[gr] = dequantize(t_pup, pup[PUPITRE_TIN[gr]].to_numpy())
        write_series(out / tin, t_pup, tin_values, "temperature")
        write_series(out / i_meas, t, series[gr]["I"], "current")
        write_series(out / u_meas, t, series[gr]["U"], "voltage")
        circuit["reference_csv"] = ref
        circuit["temperature_csv"] = tin
        circuit["experiment_data"] = [
            {"type": "voltage", "key": "voltage", "file": u_meas},
            {"type": "current", "key": "current", "file": i_meas},
        ]
        circuits.append(circuit)

    # Archives: those of the overview records, plus those the DB does not link
    # to a record but that start inside the window (continuation files).
    arc_linked = [n for r in records for n in r[2]]
    arc_start = _archive_starts(files)
    arc_window = sorted(n for n, s in arc_start.items()
                        if t0 - pd.Timedelta(minutes=1) <= s <= t1 and n not in arc_linked)
    arc_names = arc_linked + arc_window
    arc_span, arc_gaps = None, []
    if arc_names:
        arc = read_pigbrother([data_root / files[n] for n in arc_names])
        arc = arc.dropna(subset=[f"I_{gr}" for gr in active])
        arc = arc[(arc.utc >= t0) & (arc.utc <= t1)]
        if len(arc):
            t_arc = (arc.utc - t0).dt.total_seconds().to_numpy()
            for gr in active:
                write_series(out / f"{tag}_archive_I_{gr}.csv", t_arc, arc[f"I_{gr}"].to_numpy(),
                             "current", ARCHIVE_FLOAT_FORMAT)
                write_series(out / f"{tag}_archive_U_{gr}.csv", t_arc, arc[f"U_{gr}"].to_numpy(),
                             "voltage", ARCHIVE_FLOAT_FORMAT)
            arc_span, arc_gaps = [float(t_arc[0]), float(t_arc[-1])], _gaps(t_arc)

    if len(circuits) == 2:
        config = {"name": f"M9_M23072801_{tag}", "circuits": circuits,
                  "mutual_inductances": base["mutual_inductances"]}
        config_name = f"circuits-{tag}.json"
    else:
        config = circuits[0]
        config_name = f"circuit-{active[0]}-{tag}.json"
    (out / config_name).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    time_end = float(np.floor(t[-1]))
    gaps = _gaps(t)
    sources = {
        "pupitre": files[f"{run}.txt"],
        "overview_records": [r[0] for r in records],
        "modes": sorted({r[4] for r in records}),
        "overview": [files[n] for n in ov_names],
        "archive": [files.get(n, n) for r in records for n in r[2]],
        "archive_from_window": [files[n] for n in arc_window],  # not linked to a record in the DB
        "default": [files.get(n, n) for r in records for n in r[3]],
        "spike": [files.get(n, n) for r in records for n in (r[5] or [])],
        "window_utc": [str(t0), str(t1)],
        "active_groups": active,
        "signals_from": "overview" if use_overview else "pupitre (reference = measured current)",
        "gaps_s": gaps,  # [t_start, length] of holes > GAP_S, linearly interpolated
        "archive_span_s": arc_span,  # [first, last] Archive sample in the window, None if no Archive
        "archive_gaps_s": arc_gaps,  # holes > GAP_S in the Archive files, not interpolated
        "tin_reconstruction": {"method": "taut string in +/- q/2 band, free ends",
                               "half_band_K": tin_delta},
    }
    return RunCase(
        name=run, tag=tag, config=config_name, active=active,
        value_start=[round(float(series[gr]["I"][0]), 4) for gr in active],
        time_end=time_end,
        time_step=0.01 if time_end <= LONG_RUN_S else 0.1,
        sources=sources,
    )


def command(group: Group, case: RunCase) -> str:
    """Return the simulation command line of one run."""
    module = "coupled_main_refactor" if len(case.active) == 2 else "main_refactor"
    return (
        f"python -m magnet_scipy.{module} \\\n"
        f"    --wd examples/{group.dirname} \\\n"
        f"    --config-file {case.config} \\\n"
        f"    --value_start {' '.join(str(v) for v in case.value_start)} \\\n"
        f"    --time_start 0 --time_end {case.time_end:g} --time_step {case.time_step:g} \\\n"
        f"    --save_plots {case.tag}.png --show_analytics"
    )


def _notes(case: RunCase) -> str:
    """Return the README notes of a run: non-zero start, gaps, partial Archive."""
    notes = []
    if any(abs(v) > ACTIVE_CURRENT_A for v in case.value_start):
        notes.append("starts at " + " / ".join(f"{v:.0f} A" for v in case.value_start)
                     + " (Overview starts after the pupitre file; PID starts from rest)")
    if case.sources["gaps_s"]:
        longest = max(g[1] for g in case.sources["gaps_s"])
        notes.append(f"{len(case.sources['gaps_s'])} gaps > {GAP_S:g} s (max {longest:.0f} s), "
                     "interpolated")
    if case.sources["archive_from_window"]:
        notes.append(f"{len(case.sources['archive_from_window'])} Archive(s) found by time, "
                     "not linked in the DB")
    span = case.sources["archive_span_s"]
    if span and (span[0] > 1.0 or span[1] < case.time_end - 1.0):
        notes.append(f"Archive covers {span[0]:.0f}-{span[1]:.0f} s only")
    if case.sources["archive_gaps_s"]:
        notes.append(f"{len(case.sources['archive_gaps_s'])} Archive gaps > {GAP_S:g} s")
    return "; ".join(notes)


def write_group_files(group: Group, cases: list[RunCase], provenance: dict, out: Path) -> None:
    """Write README.md, run.sh and provenance.json of a group."""
    rows = "\n".join(
        f"| {c.name} | `{c.config}` | {'+'.join(c.active)} "
        f"| {', '.join(c.sources['overview_records']) or '—'} "
        f"| {'/'.join(c.sources['modes']) or '—'} | {c.sources['signals_from'].split(' ')[0]} "
        f"| {c.time_end:g} | {c.time_step:g} "
        f"| {len(c.sources['archive'])} | {len(c.sources['default'])} "
        f"| {len(c.sources['spike'])} | {_notes(c)} |"
        for c in cases)
    commands = "\n\n".join(f"```bash\n{command(group, c)}\n```" for c in cases)
    readme = f"""# {group.key} — {group.title}

{group.purpose}

Generated by `examples/build_M23072801_campaign.py`; see
[test_campaign_M23072801.md](../../prompts/test_campaign_M23072801.md). The CSV
files are not tracked (`examples/.gitignore`): rerun the builder to recreate them.

| Run (pupitre) | Config | Active | Overview record(s) | Mode | Signals | window [s] | max step [s] | Archive | Default | Spike | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
{rows}

GR1 = insert `M23072801`, GR2 = Bitters. A GR is active when its pupitre
current exceeds {ACTIVE_CURRENT_A:g} A. One active GR → single-circuit config
(`circuit-GR*-<tag>.json`, `main_refactor`); two → coupled config
(`circuits-<tag>.json`, `coupled_main_refactor`).

## Inputs per run (`<tag>` = `YYMMDD-HHMMSS` of the pupitre file)

Signals = `overview` (1 Hz pigbrother):

- `<tag>_reference_GR*.csv`: `Référence_A1 + A2` (GR1), `Référence_A3 + A4`
  (GR2).
- `<tag>_measured_{{I,U}}_GR*.csv`: `Courant GR*`, `ALL internes` (GR1),
  `ALL externes` (GR2).

Signals = `pupitre` (no usable Overview): reference and measured current are
both `Idcct1 + Idcct2` (GR1) or `Idcct3 + Idcct4` (GR2); voltage is
`Ucoil1..14` (GR1) or `Ucoil15 + Ucoil16` (GR2).

Archive (120 Hz pigbrother), when the run has one:

- `<tag>_archive_{{I,U}}_GR*.csv`: same channels as the Overview measured
  files, same window and time 0. The Archive files of a run are the
  `sources_archive` of its overview records in the DB (`archive`), plus the
  DB Archive files whose name time falls in the window but that no record
  links (`archive_from_window`, e.g. 5 h continuation files). The configs do not
  use them (for L identification and Default events). Covered span and
  holes (not interpolated) per run in `provenance.json` (`archive_span_s`,
  `archive_gaps_s`).

Always:

- `<tag>_Tin_GR*.csv`: pupitre `Tin1` (insert) and `Tin2` (Bitters). The
  pupitre rounds them to 0.1 °C; the file holds the taut string inside the
  ±0.05 °C band around the recorded values (smoothest curve compatible with
  the resolution, no window to tune). Half band per run in `provenance.json`.
- `Rtot_*.csv`, L, M and PID gains: copied from `M9_M23072801/{BASE_CONFIG}`.

Time 0 is the start of the window where the pupitre file and the Overview
overlap. The two clocks are aligned on absolute time (pupitre local time →
UTC); their residual lag (±1 s) is not corrected.

## Run

From the repository root (`run.sh` runs them all):

{commands}
"""
    (out / "README.md").write_text(readme)
    script = "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/../..\"\n\n" + \
        "\n\n".join(command(group, c) for c in cases) + "\n"
    (out / "run.sh").write_text(script)
    (out / "run.sh").chmod(0o755)
    provenance = dict(provenance, group=group.key,
                      runs={c.name: dict(c.sources, tag=c.tag, config=c.config,
                                         value_start=c.value_start, time_end=c.time_end,
                                         time_step=c.time_step) for c in cases})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    """Parse arguments and build the selected groups."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", type=Path, required=True, help="magnetdb DuckDB file")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="root of the raw records (pbsurv/, srv-data-install/)")
    parser.add_argument("--groups", nargs="*", default=None, help="subset, e.g. C0 C2")
    args = parser.parse_args()

    base = json.loads((BASE_CASE / BASE_CONFIG).read_text())
    for circuit, table in zip(base["circuits"], R_TABLES):
        circuit["resistance_csv"] = table
    provenance = {
        "db": args.db.name, "db_sha256": sha256(args.db),
        "assembly": ASSEMBLY, "base_config": f"M9_M23072801/{BASE_CONFIG}",
        "data_root": str(args.data_root.expanduser().resolve()).replace(str(Path.home()), "~", 1),
    }

    con = duckdb.connect(str(args.db), read_only=True)
    for group in CAMPAIGN:
        if args.groups and group.key not in args.groups:
            continue
        out = EXAMPLES / group.dirname
        out.mkdir(exist_ok=True)
        for table in R_TABLES:
            shutil.copy(BASE_CASE / table, out / table)
        cases = [build_run(con, run, args.data_root.expanduser(), out, base) for run in group.runs]
        write_group_files(group, cases, provenance, out)
        print(f"{group.key}: {out.name} ({len(cases)} runs)")


if __name__ == "__main__":
    main()
