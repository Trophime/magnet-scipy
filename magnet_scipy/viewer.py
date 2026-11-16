"""Interactive matplotlib viewer for simulation results and experiment comparison.

Reads the ``.npz`` files written by ``magnet-scipy`` / ``magnet-scipy-coupled``
(``--save_results``) and, when the matching circuit JSON is found, overlays the
experimental series listed under ``experiment_data``.

Views
-----
overview
    One run: current / voltage / power / resistance / temperature against time,
    experiment overlaid, linked time axes, circuit check-boxes.
residuals
    One run: simulation minus experiment for each experimental series.
compare
    Several runs: one quantity of one circuit overlaid, experiment included.
flow
    Pump speed / flow / pressure against current with the
    ``python_magnetcooling`` fit (optional dependency).

Command line::

    python -m magnet_scipy.viewer overview run.npz [--config run.json]
    python -m magnet_scipy.viewer compare a.npz b.npz -q current -c M23072801
    python -m magnet_scipy.viewer flow records.csv --imax 28000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .csv_utils import _read_csv

QUANTITIES = ("current", "voltage", "power", "resistance", "temperature")
UNITS = {
    "current": "A",
    "voltage": "V",
    "power": "W",
    "resistance": "Ohm",
    "temperature": "K",
    "reference": "A",
    "error": "A",
}


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
    """

    name: str
    time: np.ndarray
    series: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    experiments: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )

    @property
    def circuits(self) -> list[str]:
        """Circuit identifiers present in the run."""
        return list(self.series)


def _read_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column CSV (``time`` plus one value column, any order)."""
    df = _read_csv(path)
    tcol = "time" if "time" in df.columns else df.columns[0]
    vcol = next(c for c in df.columns if c != tcol)
    return df[tcol].to_numpy(float), df[vcol].to_numpy(float)


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
) -> Run:
    """Load a results ``.npz`` and its experimental series.

    Parameters
    ----------
    npz_path : str or :class:`~pathlib.Path`
        Results file written by the simulation.
    config_path : str or :class:`~pathlib.Path`, optional
        Circuit JSON. Defaults to the ``.json`` with the same stem next to the
        ``.npz``; experiments are skipped if it does not exist.
    wd : str or :class:`~pathlib.Path`, optional
        Directory the experiment CSV names are relative to. Defaults to the
        directory of the JSON.

    Returns
    -------
    Run
        Simulation series and, when available, experimental series.
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
                f = base / e["file"]
                if f.exists() and cid in run.series:
                    run.experiments.setdefault(cid, {})[e.get("key", e["type"])] = (
                        _read_series(f)
                    )
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


def _add_circuit_toggle(fig: Figure, circuits: list[str], artists: dict) -> object:
    """Add check-boxes toggling the artists (``artists[circuit]``) per circuit."""
    from matplotlib.widgets import CheckButtons

    ax = fig.add_axes((0.005, 0.80, 0.12, 0.025 + 0.04 * len(circuits)))
    ax.set_title("circuits", fontsize=8)
    cb = CheckButtons(ax, circuits, [True] * len(circuits))

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
        Figure with linked time axes and circuit check-boxes.
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
                    *exp, color=_color(i), ls=":", lw=1.5, label=f"{cid} exp"
                )
                artists[cid].append(ln)
        ax.set_ylabel(f"{q} [{UNITS[q]}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1].set_xlabel("t [s]")
    axes[-1].set_xlim(run.time[0], run.time[-1])
    fig.suptitle(run.name)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.95, hspace=0.12)
    fig._toggle = _add_circuit_toggle(fig, run.circuits, artists)  # type: ignore[attr-defined]
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
        One panel per experimental quantity (empty note if there is none).
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
                *res,
                color=_color(i),
                lw=0.8,
                label=f"{cid}: RMS={rms:.3g} MAE={mae:.3g}",
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"sim - exp {k} [{UNITS.get(k, '')}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("t [s]")
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
            axr.plot(*res, color=_color(i), lw=0.8)
    if exp is not None:
        ax.plot(*exp, "k:", lw=1.5, label="experiment")
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
    s = sub.add_parser("compare")
    s.add_argument("npz", nargs="+")
    s.add_argument("-c", "--circuit", required=True)
    s.add_argument("-q", "--quantity", default="current")
    s = sub.add_parser("flow")
    s.add_argument("csv")
    s.add_argument("--imax", type=float)
    s.add_argument("--method", default="simple", choices=("simple", "piecewise"))
    s.add_argument("--threshold", type=float, default=300.0)
    a = p.parse_args(argv)

    if a.view == "overview":
        plot_overview(load_run(a.npz, a.config, a.wd))
    elif a.view == "residuals":
        plot_residuals(load_run(a.npz, a.config, a.wd))
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
