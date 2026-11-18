"""Interactive matplotlib viewer for simulation results and experiment comparison.

Reads the ``.npz`` files written by ``magnet-scipy`` / ``magnet-scipy-coupled``
(``--save_results``) and, when the matching circuit JSON is found, overlays the
experimental series listed under ``experiment_data``.

Views
-----
overview
    One run: current / voltage / power / resistance / temperature against time,
    experiment overlaid, pigbrother Default/Spike captures as vertical lines,
    linked time axes, check-boxes per circuit and event kind.
residuals
    One run: simulation minus experiment for each experimental series.
compare
    Several runs: one quantity of one circuit overlaid, experiment included.
flow
    Pump speed / flow / pressure against current with the
    ``python_magnetcooling`` fit (optional dependency).
flowcheck
    Flow params files (``flow_params`` of each circuit) against the measured
    hydraulics (``hydraulics_csv``), per operating mode inferred for each
    sample (:mod:`magnet_scipy.operating_mode`), with an RMS table.

Command line::

    python -m magnet_scipy.viewer overview run.npz [--config run.json]
    python -m magnet_scipy.viewer compare a.npz b.npz -q current -c M23072801
    python -m magnet_scipy.viewer flow records.csv --imax 28000
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .csv_utils import _read_csv
from .experiment_io import read_experiment

DISPLAY_POINTS = 20000  # experimental traces longer than this are decimated for display
QUANTITIES = ("current", "voltage", "power", "resistance", "temperature")
UNITS = {
    "current": "A",
    "voltage": "V",
    "power": "W",
    "resistance": "Ohm",
    "temperature": "°C",  # inlet water temperature Tin, as in the R(I, Tin) tables
    "reference": "A",
    "error": "A",
}


EVENT_KINDS = {"Default": "default", "Spikes": "spike"}  # pigbrother mode -> kind
EVENT_COLORS = {"default": "red", "spike": "purple"}


@dataclass
class Event:
    """A pigbrother Default or Spike capture.

    Attributes
    ----------
    kind : str
        ``"default"`` or ``"spike"``.
    t : float
        Capture time on the simulation time axis [s].
    subtype : str
        Default subtype from the filename (e.g. ``"Courants50Hz"``), or ``""``.
    file : str
        Source filename.
    """

    kind: str
    t: float
    subtype: str
    file: str


def parse_event_file(
    filename: str, time_zone: str = "Europe/Paris"
) -> tuple[str, pd.Timestamp, str] | None:
    """Parse kind, time and subtype from a pigbrother Default/Spike filename.

    The file itself is never opened. Names follow
    ``{site}_{Default|Spikes}_{YYMMDD-HHMM[SS]}[_{subtype}].tdms`` in local
    time. Local time is converted to naive UTC like
    ``python_magnetrun.utils.timezone.local_to_utc_naive``: an ambiguous time
    (autumn fall-back) resolves as the first (DST) occurrence, a non-existent
    one is shifted forward.

    Parameters
    ----------
    filename : str
        Path or bare name, e.g. ``"M9_Default_230919-193827_Courants50Hz.tdms"``.
    time_zone : str
        IANA time zone of the timestamp in the name.

    Returns
    -------
    tuple or None
        ``(kind, utc, subtype)`` with ``kind`` in ``{"default", "spike"}`` and
        ``utc`` a naive-UTC :class:`~pandas.Timestamp`; ``None`` if the name is
        not a Default/Spike capture or cannot be parsed.
    """
    path = Path(filename)
    parts = [p for p in path.stem.split("_") if p]
    if path.suffix.lower() != ".tdms" or len(parts) < 3 or parts[1] not in EVENT_KINDS:
        return None
    date, _, hms = parts[2].partition("-")
    fmt = "%y%m%d%H%M%S" if len(hms) >= 6 else "%y%m%d%H%M"
    try:
        local = pd.to_datetime(date + hms[:6], format=fmt)
    except ValueError:
        return None
    utc = (
        local.tz_localize(time_zone, ambiguous=True, nonexistent="shift_forward")
        .tz_convert("UTC")
        .tz_localize(None)
    )
    return EVENT_KINDS[parts[1]], utc, parts[3] if len(parts) > 3 else ""


def load_events(source: dict) -> list[Event]:
    """Place the Default/Spike captures of a record on the simulation time axis.

    Parameters
    ----------
    source : dict
        Overview-record-like mapping: ``t0`` (naive UTC of simulation t = 0;
        ``window_utc[0]`` is used if ``t0`` is absent) and the file lists
        ``sources_default`` / ``sources_spike`` (or ``default`` / ``spike``,
        as in the campaign ``provenance.json``).

    Returns
    -------
    list of Event
        Events sorted by time; unparseable names are skipped.
    """
    t0 = pd.Timestamp(source.get("t0") or source["window_utc"][0])
    events = []
    for key in ("default", "spike"):
        for f in source.get(f"sources_{key}") or source.get(key) or []:
            parsed = parse_event_file(f)
            if parsed is not None:
                kind, utc, subtype = parsed
                events.append(Event(kind, (utc - t0).total_seconds(), subtype, f))
    return sorted(events, key=lambda e: e.t)


def _provenance_run(cfg: Path) -> dict | None:
    """Return the ``provenance.json`` run entry whose ``config`` is ``cfg``, if any."""
    prov = cfg.parent / "provenance.json"
    if not prov.is_file():
        return None
    runs = json.loads(prov.read_text()).get("runs", {})
    return next((r for r in runs.values() if r.get("config") == cfg.name), None)


@dataclass
class Run:
    """Simulation results of one run plus its experimental series.

    Attributes
    ----------
    name : str
        Label used in legends (the ``.npz`` stem).
    time : :class:`~numpy.ndarray`
        Simulation time [s].
    series : dict
        ``series[circuit_id][quantity]`` -> :class:`~numpy.ndarray`.
    experiments : dict
        ``experiments[circuit_id][key]`` -> ``(time [s], values)`` tuple.
    events : list of Event
        pigbrother Default/Spike captures on the simulation time axis.
    """

    name: str
    time: np.ndarray
    series: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    experiments: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    events: list[Event] = field(default_factory=list)

    @property
    def circuits(self) -> list[str]:
        """Circuit identifiers present in the run."""
        return list(self.series)


def _decimate(
    t: np.ndarray, y: np.ndarray, max_points: int = DISPLAY_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimation for display: keeps the envelope of long traces."""
    k = 2 * len(t) // max_points
    if k < 2:
        return t, y
    m = len(t) // k * k
    yb = y[:m].reshape(-1, k)
    idx = np.sort(np.stack([yb.argmin(1), yb.argmax(1)], 1), 1)
    idx = (idx + k * np.arange(len(yb))[:, None]).ravel()
    return t[idx], y[idx]


def _experiment_entries(config: dict) -> dict[str, list[dict]]:
    """Map circuit id -> ``experiment_data`` list, for multi- or single-circuit JSON."""
    if "circuits" in config:
        return {
            c["circuit_id"]: c.get("experiment_data", []) for c in config["circuits"]
        }
    return {config.get("circuit_id", ""): config.get("experiment_data", [])}


def load_run(
    npz_path: str | Path,
    config_path: str | Path | None = None,
    wd: str | Path | None = None,
    events: str | Path | None = None,
) -> Run:
    """Load a results ``.npz``, its experimental series and its events.

    Parameters
    ----------
    npz_path : str or :class:`~pathlib.Path`
        Results file written by the simulation.
    config_path : str or :class:`~pathlib.Path`, optional
        Circuit JSON. Defaults to the ``.json`` with the same stem next to the
        ``.npz``; experiments are skipped if it does not exist.
    wd : str or :class:`~pathlib.Path`, optional
        Directory the experiment file names (CSV or TDMS, see
        :mod:`magnet_scipy.experiment_io`) are relative to. Defaults to the
        directory of the JSON. Unreadable sources are skipped with a warning.
    events : str or :class:`~pathlib.Path`, optional
        JSON with the Default/Spike captures, see :func:`load_events`.
        Defaults to the run of ``provenance.json`` (next to the circuit JSON)
        whose ``config`` is the circuit JSON; no events if there is none.

    Returns
    -------
    Run
        Simulation series and, when available, experimental series and events.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    run = Run(name=npz_path.stem, time=np.asarray(data["time"], float))

    for key in data.files:
        if key.endswith("_current") and data[key].shape == run.time.shape:
            cid = key[: -len("_current")]
            run.series[cid] = {
                k[len(cid) + 1 :]: np.asarray(data[k], float)
                for k in data.files
                if k.startswith(cid + "_")
                and data[k].shape == run.time.shape
                and k[len(cid) + 1 :] in (*QUANTITIES, "reference", "error")
            }

    cfg = Path(config_path) if config_path else npz_path.with_suffix(".json")
    if cfg.exists():
        base = Path(wd) if wd else cfg.parent
        for cid, entries in _experiment_entries(json.loads(cfg.read_text())).items():
            for e in entries:
                if cid not in run.series:
                    continue
                try:
                    series = read_experiment(e, base)
                except (FileNotFoundError, ImportError, KeyError) as exc:
                    warnings.warn(f"{cid}: experiment {e['file']} skipped: {exc}")
                    continue
                run.experiments.setdefault(cid, {})[e.get("key", e["type"])] = series

    source = json.loads(Path(events).read_text()) if events else _provenance_run(cfg)
    if source:
        run.events = load_events(source)
    return run


def residual(run: Run, circuit: str, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Simulation minus experiment, evaluated at the experimental times.

    Parameters
    ----------
    run : Run
        Loaded run.
    circuit : str
        Circuit identifier.
    key : str
        Quantity name (``"current"``, ``"voltage"``, ...).

    Returns
    -------
    tuple of :class:`~numpy.ndarray` or None
        ``(t_exp [s], sim - exp)`` or ``None`` if no experiment exists for
        ``key``. Experimental points outside the simulated window are dropped.
    """
    exp = run.experiments.get(circuit, {}).get(key)
    if exp is None or key not in run.series[circuit]:
        return None
    t, y = exp
    m = (t >= run.time[0]) & (t <= run.time[-1])
    return t[m], np.interp(t[m], run.time, run.series[circuit][key]) - y[m]


def _color(i: int) -> str:
    return f"C{i % 10}"


def _rms_mae(r: np.ndarray) -> tuple[float, float]:
    return float(np.sqrt(np.mean(r**2))), float(np.mean(np.abs(r)))


def _draw_events(axes, events: list[Event]) -> dict[str, list]:
    """Draw one dashed vertical line per event on every axis; return artists per kind."""
    from matplotlib.transforms import blended_transform_factory

    artists: dict[str, list] = {}
    for j, ax in enumerate(axes):
        labelled = set()
        for ev in events:
            ln = ax.axvline(
                ev.t,
                color=EVENT_COLORS[ev.kind],
                ls="--",
                lw=0.8,
                alpha=0.7,
                label=None if ev.kind in labelled else ev.kind,
            )
            labelled.add(ev.kind)
            artists.setdefault(ev.kind, []).append(ln)
            if j == 0 and ev.subtype:
                tr = blended_transform_factory(ax.transData, ax.transAxes)
                txt = ax.text(
                    ev.t,
                    0.02,
                    ev.subtype,
                    transform=tr,
                    rotation=90,
                    fontsize=6,
                    va="bottom",
                    ha="left",  # right of the line: hidden with it when off-axis
                    color=EVENT_COLORS[ev.kind],
                    clip_on=True,
                )
                artists[ev.kind].append(txt)
    return artists


def _add_toggle(fig: Figure, labels: list[str], artists: dict) -> object:
    """Add check-boxes toggling the artists (``artists[label]``) per label."""
    from matplotlib.widgets import CheckButtons

    ax = fig.add_axes((0.005, 0.80, 0.12, 0.025 + 0.04 * len(labels)))
    ax.set_title("show", fontsize=8)
    cb = CheckButtons(ax, labels, [True] * len(labels))

    def _toggle(label):
        for a in artists[label]:
            a.set_visible(not a.get_visible())
        fig.canvas.draw_idle()

    cb.on_clicked(_toggle)
    return cb  # caller must keep a reference, or the widget is garbage-collected


def plot_overview(run: Run, fig: Figure | None = None) -> Figure:
    """Plot every available quantity of a run, experiment overlaid.

    Parameters
    ----------
    run : Run
        Loaded run.
    fig : :class:`~matplotlib.figure.Figure`, optional
        Figure to draw into; a new one is created if omitted.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
        Figure with linked time axes, Default/Spike event lines and
        check-boxes per circuit and per event kind.
    """
    import matplotlib.pyplot as plt

    quantities = [q for q in QUANTITIES if any(q in s for s in run.series.values())]
    fig = fig or plt.figure(figsize=(13, 2.2 * len(quantities) + 1))
    axes = fig.subplots(len(quantities), 1, sharex=True, squeeze=False)[:, 0]
    artists: dict[str, list] = {c: [] for c in run.circuits}

    for ax, q in zip(axes, quantities, strict=True):
        for i, cid in enumerate(run.circuits):
            if q in run.series[cid]:
                (ln,) = ax.plot(
                    run.time, run.series[cid][q], color=_color(i), label=f"{cid} sim"
                )
                artists[cid].append(ln)
            if q == "current" and "reference" in run.series[cid]:
                (ln,) = ax.plot(
                    run.time,
                    run.series[cid]["reference"],
                    color=_color(i),
                    ls="--",
                    lw=0.8,
                    label=f"{cid} reference",
                )
                artists[cid].append(ln)
            exp = run.experiments.get(cid, {}).get(q)
            if exp is not None:
                (ln,) = ax.plot(
                    *_decimate(*exp),
                    color=_color(i),
                    ls=":",
                    lw=1.5,
                    label=f"{cid} exp",
                )
                artists[cid].append(ln)
    artists.update(_draw_events(axes, run.events))
    for ax, q in zip(axes, quantities, strict=True):
        ax.set_ylabel(f"{q} [{UNITS[q]}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1].set_xlabel("t [s]")
    axes[-1].set_xlim(run.time[0], run.time[-1])
    fig.suptitle(run.name)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.95, hspace=0.12)
    fig._toggle = _add_toggle(fig, list(artists), artists)  # type: ignore[attr-defined]
    return fig


def plot_residuals(run: Run, fig: Figure | None = None) -> Figure:
    """Plot simulation minus experiment, with RMS/MAE in each legend.

    Parameters
    ----------
    run : Run
        Loaded run.
    fig : :class:`~matplotlib.figure.Figure`, optional
        Figure to draw into; a new one is created if omitted.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
        One panel per experimental quantity (empty note if there is none),
        with the Default/Spike event lines.
    """
    import matplotlib.pyplot as plt

    keys = sorted({k for e in run.experiments.values() for k in e})
    fig = fig or plt.figure(figsize=(13, 2.5 * max(len(keys), 1) + 1))
    if not keys:
        fig.text(0.5, 0.5, "no experimental data found", ha="center")
        return fig
    axes = fig.subplots(len(keys), 1, sharex=True, squeeze=False)[:, 0]
    for ax, k in zip(axes, keys, strict=True):
        for i, cid in enumerate(run.circuits):
            res = residual(run, cid, k)
            if res is None:
                continue
            rms, mae = _rms_mae(res[1])
            ax.plot(
                *_decimate(*res),
                color=_color(i),
                lw=0.8,
                label=f"{cid}: RMS={rms:.3g} MAE={mae:.3g}",
            )
        ax.axhline(0, color="k", lw=0.5)
    _draw_events(axes, run.events)
    for ax, k in zip(axes, keys, strict=True):
        ax.set_ylabel(f"sim - exp {k} [{UNITS.get(k, '')}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("t [s]")
    axes[-1].set_xlim(run.time[0], run.time[-1])
    fig.suptitle(f"{run.name} - residuals")
    fig.tight_layout()
    return fig


def plot_compare(
    runs: list[Run], circuit: str, quantity: str, fig: Figure | None = None
) -> Figure:
    """Overlay one quantity of one circuit across several runs.

    Parameters
    ----------
    runs : list of Run
        Runs to overlay; those lacking ``circuit`` are skipped.
    circuit : str
        Circuit identifier.
    quantity : str
        Quantity name, e.g. ``"current"``.
    fig : :class:`~matplotlib.figure.Figure`, optional
        Figure to draw into; a new one is created if omitted.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
        Top panel: curves (experiment drawn once, from the first run having
        it); bottom panel: residual of each run against that experiment.
    """
    import matplotlib.pyplot as plt

    fig = fig or plt.figure(figsize=(13, 7))
    ax, axr = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    exp = None
    for i, run in enumerate(runs):
        if circuit not in run.series or quantity not in run.series[circuit]:
            continue
        ax.plot(
            run.time, run.series[circuit][quantity], color=_color(i), label=run.name
        )
        exp = exp or run.experiments.get(circuit, {}).get(quantity)
        res = residual(run, circuit, quantity)
        if res is not None:
            axr.plot(*_decimate(*res), color=_color(i), lw=0.8)
    if exp is not None:
        ax.plot(*_decimate(*exp), "k:", lw=1.5, label="experiment")
    ends = [r.time[[0, -1]] for r in runs if circuit in r.series]
    if ends:
        ax.set_xlim(min(e[0] for e in ends), max(e[1] for e in ends))
    ax.set_ylabel(f"{quantity} [{UNITS.get(quantity, '')}]")
    axr.set_ylabel("sim - exp")
    axr.set_xlabel("t [s]")
    for a in (ax, axr):
        a.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"{circuit} - {quantity}")
    fig.tight_layout()
    return fig


def plot_flow_fit(
    csv_path: str | Path,
    imax: float | None = None,
    method: str = "simple",
    current_threshold: float = 300.0,
    fig: Figure | None = None,
) -> Figure:
    """Plot pump speed, flow and pressure against current with the fitted model.

    Needs the optional ``python_magnetcooling`` package; the fit is done by
    :func:`python_magnetcooling.fitting.fit_hydraulic_system`.

    Parameters
    ----------
    csv_path : str or :class:`~pathlib.Path`
        CSV with columns ``current`` [A], ``pump_speed`` [rpm],
        ``flow_rate`` [l/s], ``pressure`` [bar] and optionally
        ``back_pressure`` [bar].
    imax : float, optional
        Maximum current [A]; required for ``method="simple"``.
    method : str
        ``"simple"`` or ``"piecewise"``.
    current_threshold : float
        Samples below this current [A] are ignored by the fit.
    fig : :class:`~matplotlib.figure.Figure`, optional
        Figure to draw into; a new one is created if omitted.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
        Three panels with measured points and the fitted curve.

    Raises
    ------
    ImportError
        If ``python_magnetcooling`` is not installed.
    """
    import matplotlib.pyplot as plt

    try:
        from python_magnetcooling.fitting import fit_hydraulic_system
    except ImportError as exc:
        raise ImportError("the flow view needs python_magnetcooling") from exc

    df = _read_csv(csv_path).dropna()
    cur, vp, fl, pr = (
        df[c].to_numpy(float)
        for c in ("current", "pump_speed", "flow_rate", "pressure")
    )
    bp = (
        df["back_pressure"].to_numpy(float)
        if "back_pressure" in df
        else np.zeros_like(pr)
    )
    pump, fp = fit_hydraulic_system(
        cur,
        vp,
        fl,
        pr,
        bp,
        imax=imax,
        method=method,
        current_threshold=current_threshold,
    )
    i = np.linspace(cur.min(), cur.max(), 300)
    vpm = np.array([pump.pump_speed(x) for x in i])
    x = vpm / (pump.vpmax + pump.vp0)
    models = {
        "pump_speed": (vp, vpm, "pump speed [rpm]"),
        "flow_rate": (fl, fp.f0 + fp.fmax * x, "flow [l/s]"),
        "pressure": (pr, fp.pmin + fp.pmax * x**2, "pressure [bar]"),
    }
    fig = fig or plt.figure(figsize=(14, 4.5))
    for ax, (meas, fit, label) in zip(fig.subplots(1, 3), models.values(), strict=True):
        ax.plot(cur, meas, ".", ms=3, alpha=0.4, label="measured")
        ax.plot(i, fit, "r-", label="fit")
        ax.axvline(current_threshold, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("I [A]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Vpmax={pump.vpmax:.0f} Vp0={pump.vp0:.0f} Imax={pump.imax:.0f} | "
        f"F0={fp.f0:.2f} Fmax={fp.fmax:.2f} | Pmin={fp.pmin:.2f} Pmax={fp.pmax:.2f}"
    )
    fig.tight_layout()
    return fig


FLOW_QUANTITIES = {
    "rpm": "pump speed [rpm]",
    "flow": "flow [l/s]",
    "hp": "inlet pressure HP [bar]",
    "bp": "back pressure BP [bar]",
}


def flow_model(params: dict, current: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate a water flow params file against current.

    Same model as :class:`python_magnetcooling.waterflow.WaterFlow`:
    ``Vp = Vpmax (I/Imax)^2 + Vp0``, ``F = F0 + Fmax Vp/(Vpmax+Vp0)``,
    ``P = Pmin + Pmax (Vp/(Vpmax+Vp0))^2``, constant back pressure, all held
    at their ``Imax`` value above ``Imax``.

    Parameters
    ----------
    params : dict
        Flow params, ``{"Vp0": {"value": ..., "unit": "rpm"}, ...}``; back
        pressure under ``BP`` or ``Pout`` (optional).
    current : :class:`~numpy.ndarray`
        Current [A]; only its magnitude is used.

    Returns
    -------
    dict of :class:`~numpy.ndarray`
        ``rpm`` [rpm], ``flow`` [l/s], ``hp`` [bar] and ``bp`` [bar] (NaN when
        the file has no back pressure).
    """
    v = {k: p["value"] for k, p in params.items()}
    i = np.minimum(np.abs(np.asarray(current, float)), v["Imax"])
    rpm = v["Vpmax"] * (i / v["Imax"]) ** 2 + v["Vp0"]
    x = rpm / (v["Vpmax"] + v["Vp0"])
    bp = v.get("BP", v.get("Pout", np.nan))
    return {
        "rpm": rpm,
        "flow": v["F0"] + v["Fmax"] * x,
        "hp": v["Pmin"] + v["Pmax"] * x**2,
        "bp": np.full_like(i, bp),
    }


@dataclass
class Hydraulics:
    """Measured hydraulics of one circuit in one run, with its flow params.

    Attributes
    ----------
    circuit : str
        Circuit identifier.
    run : str
        Config file stem, e.g. ``"circuits-230919-190811"``.
    params_file : str
        Flow params file name.
    params : dict
        Flow params.
    data : :class:`~pandas.DataFrame`
        Columns ``time`` [s], ``current`` [A], ``rpm``, ``flow``, ``hp``,
        ``bp`` and ``mode`` (operating mode of each sample, see
        :func:`load_hydraulics`).
    """

    circuit: str
    run: str
    params_file: str
    params: dict
    data: pd.DataFrame


def load_hydraulics(config_paths: list[str | Path]) -> list[Hydraulics]:
    """Load the circuits that have both ``flow_params`` and ``hydraulics_csv``.

    The operating mode of each sample is inferred shot by shot from the
    ``ih`` / ``ib`` columns with :func:`magnet_scipy.operating_mode.shot_modes`
    (``"ECO"``, ``"NORMAL"``, or ``"rest"`` when both supplies are off). A
    shot the rule cannot classify, or a CSV without ``ih`` / ``ib``, falls back
    to the ``modes`` of the run of ``provenance.json`` whose ``config`` is the
    JSON (``overview_records.mode``), when there is exactly one; otherwise
    ``"unknown"``.

    Parameters
    ----------
    config_paths : list of str or :class:`~pathlib.Path`
        Circuit JSON files (single circuit or ``circuits`` list); file names
        are relative to each JSON.

    Returns
    -------
    list of Hydraulics
        One entry per circuit and config; circuits whose files are missing
        are skipped with a warning.
    """
    from .operating_mode import shot_modes

    out = []
    for cfg in map(Path, config_paths):
        config = json.loads(cfg.read_text())
        prov_modes = (_provenance_run(cfg) or {}).get("modes") or []
        fallback = prov_modes[0] if len(prov_modes) == 1 else "unknown"
        for c in config.get("circuits", [config]):
            if not (c.get("flow_params") and c.get("hydraulics_csv")):
                continue
            try:
                params = json.loads((cfg.parent / c["flow_params"]).read_text())
                data = _read_csv(str(cfg.parent / c["hydraulics_csv"]))
            except FileNotFoundError as exc:
                warnings.warn(f"{cfg.name} {c['circuit_id']}: skipped: {exc}")
                continue
            if {"ih", "ib"} <= set(data.columns):
                mode = shot_modes(data["ih"].to_numpy(), data["ib"].to_numpy())
                mode[mode == "UNKNOWN"] = fallback
            else:
                mode = np.full(len(data), fallback, dtype=object)
            out.append(
                Hydraulics(
                    c["circuit_id"],
                    cfg.stem,
                    c["flow_params"],
                    params,
                    data.assign(mode=mode),
                )
            )
    return out


def _with_residuals(r: Hydraulics) -> pd.DataFrame:
    """Data of a record plus ``res_<q>`` = model - measurement columns."""
    model = flow_model(r.params, r.data["current"].to_numpy())
    return r.data.assign(
        **{f"res_{q}": model[q] - r.data[q].to_numpy(float) for q in FLOW_QUANTITIES}
    )


def flow_check_table(
    records: list[Hydraulics], threshold: float = 300.0
) -> pd.DataFrame:
    """Bias and RMS of model minus measurement, per run, circuit and mode.

    Parameters
    ----------
    records : list of Hydraulics
        Output of :func:`load_hydraulics`.
    threshold : float
        Samples with ``|I|`` at or below this current [A] are ignored.

    Returns
    -------
    :class:`~pandas.DataFrame`
        One row per record and operating mode present above the threshold:
        ``circuit``, ``run``, ``mode``, ``params``, ``n``, then ``<q>_bias``
        and ``<q>_rms`` for ``q`` in rpm, flow, hp, bp.
    """
    rows = []
    for r in records:
        d = _with_residuals(r)
        for mode, g in d[d["current"].abs() > threshold].groupby("mode"):
            row = {
                "circuit": r.circuit,
                "run": r.run,
                "mode": mode,
                "params": r.params_file,
                "n": len(g),
            }
            for q in FLOW_QUANTITIES:
                res = g[f"res_{q}"].to_numpy()
                row[f"{q}_bias"] = float(np.mean(res))
                row[f"{q}_rms"] = float(np.sqrt(np.mean(res**2)))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_flow_check(
    records: list[Hydraulics], threshold: float = 300.0
) -> list[Figure]:
    """Plot measured pump speed, flow, HP and BP against the flow params model.

    Parameters
    ----------
    records : list of Hydraulics
        Output of :func:`load_hydraulics`.
    threshold : float
        Current [A] below which samples are left out of the RMS (drawn as a
        dotted vertical line).

    Returns
    -------
    list of :class:`~matplotlib.figure.Figure`
        One figure per circuit: measurements coloured by the mode of each
        sample, with the RMS of each mode in the legend, and one model curve
        per flow params file.
    """
    import matplotlib.pyplot as plt

    figs = []
    for cid in dict.fromkeys(r.circuit for r in records):
        recs = [r for r in records if r.circuit == cid]
        data = pd.concat([_with_residuals(r) for r in recs], ignore_index=True)
        fig, axes = plt.subplots(1, len(FLOW_QUANTITIES), figsize=(18, 4.8))
        fig._label = cid  # type: ignore[attr-defined]
        i_max = float(data["current"].abs().max())
        for ax, (q, label) in zip(axes, FLOW_QUANTITIES.items(), strict=True):
            for k, (mode, g) in enumerate(data.groupby("mode")):
                res = g.loc[g["current"].abs() > threshold, f"res_{q}"].to_numpy()
                name = (
                    f"{mode}: RMS={np.sqrt(np.mean(res**2)):.3g}" if len(res) else mode
                )
                step = max(1, len(g) // DISPLAY_POINTS)
                ax.plot(
                    g["current"].abs()[::step],
                    g[q][::step],
                    ".",
                    ms=1.5,
                    alpha=0.4,
                    color=_color(k),
                    label=name,
                )
            ii = np.linspace(0.0, 1.05 * i_max, 300)
            for j, f in enumerate(dict.fromkeys(r.params_file for r in recs)):
                params = next(r.params for r in recs if r.params_file == f)
                ax.plot(
                    ii,
                    flow_model(params, ii)[q],
                    "k",
                    ls=("-", "--", ":")[j % 3],
                    lw=1.8,
                    label=f,
                )
            ax.axvline(threshold, color="gray", ls=":", lw=0.8)
            ax.set_xlabel("|I| [A]")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, markerscale=5)
        n_runs = len({r.run for r in recs})
        fig.suptitle(f"{cid}: flow params vs measured hydraulics ({n_runs} runs)")
        fig.tight_layout()
        figs.append(fig)
    return figs


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (``python -m magnet_scipy.viewer``)."""
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--save", help="write the figure to this file instead of showing it")
    sub = p.add_subparsers(dest="view", required=True)
    for name in ("overview", "residuals"):
        s = sub.add_parser(name)
        s.add_argument("npz")
        s.add_argument("--config", help="circuit JSON (default: <npz stem>.json)")
        s.add_argument("--wd", help="directory of the experiment CSV files")
        s.add_argument(
            "--events",
            help="JSON with t0 (UTC) and sources_default/sources_spike "
            "(default: matching run of provenance.json)",
        )
    s = sub.add_parser("compare")
    s.add_argument("npz", nargs="+")
    s.add_argument("-c", "--circuit", required=True)
    s.add_argument("-q", "--quantity", default="current")
    s = sub.add_parser("flow")
    s.add_argument("csv")
    s.add_argument("--imax", type=float)
    s.add_argument("--method", default="simple", choices=("simple", "piecewise"))
    s.add_argument("--threshold", type=float, default=300.0)
    s = sub.add_parser("flowcheck", help="flow params files vs measured hydraulics")
    s.add_argument(
        "config", nargs="+", help="circuit JSON(s) with flow_params and hydraulics_csv"
    )
    s.add_argument(
        "--threshold", type=float, default=300.0, help="ignore |I| below [A]"
    )
    a = p.parse_args(argv)

    if a.view == "flowcheck":
        records = load_hydraulics(a.config)
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(
                flow_check_table(records, a.threshold).round(3).to_string(index=False)
            )
        figs = plot_flow_check(records, a.threshold)
        if a.save:
            for fig in figs:
                path = Path(a.save)
                fig.savefig(path.with_stem(f"{path.stem}_{fig._label}"), dpi=150)
        else:
            plt.show()
        return

    if a.view == "overview":
        plot_overview(load_run(a.npz, a.config, a.wd, a.events))
    elif a.view == "residuals":
        plot_residuals(load_run(a.npz, a.config, a.wd, a.events))
    elif a.view == "compare":
        plot_compare([load_run(f) for f in a.npz], a.circuit, a.quantity)
    else:
        plot_flow_fit(a.csv, a.imax, a.method, a.threshold)

    if a.save:
        plt.gcf().savefig(a.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
