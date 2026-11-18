"""PNG plots for the solver benchmarks.

Used by ``bench_solvers.py``, ``bench_campaign.py`` and
``examples/M9_M23072801/benchmark/build_summary.py``. Run on its own, it
regenerates the plots of a ``bench_solvers`` / ``bench_campaign`` output
directory from its CSV files and ``runs/*.npz``, without solving again.

Usage
-----
::

    python benchmarks/plot_results.py benchmarks/results
    python benchmarks/plot_results.py benchmarks/results/campaign
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed colour and marker per method: a method keeps its look in every figure.
METHOD_STYLE = {
    "RK45": ("#2a78d6", "o"),
    "RK23": ("#eb6834", "s"),
    "DOP853": ("#1baf7a", "^"),
    "Radau": ("#eda100", "D"),
    "BDF": ("#e87ba4", "v"),
    "LSODA": ("#008300", "P"),
}
SIM, MEAS, AUX = "#2a78d6", "#eb6834", "#7a7974"
MAX_PLOT_POINTS = 20000


def _style(ax) -> None:
    """Recessive grid, no top/right spines, legend if anything is labelled."""
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9)


def _plain_ticks(ax) -> None:
    """Label a log y axis with plain numbers (``5.2``, not ``5.2×10⁰``)."""
    fmt = ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)


def _method_style(method: str) -> tuple[str, str]:
    """Colour and marker of a method (grey circle for an unknown one)."""
    return METHOD_STYLE.get(method, (AUX, "o"))


def _metrics(simulated: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """RMSE, bias and range-normalised RMSE of ``simulated - measured``."""
    err = simulated - measured
    rmse = float(np.sqrt(np.mean(err**2)))
    scale = float(np.max(np.abs(measured)))
    return {"rmse": rmse, "bias": float(np.mean(err)),
            "nrmse": rmse / scale if scale > 0 else np.nan}


def _thin(*arrays: np.ndarray) -> list[np.ndarray]:
    """Keep at most :data:`MAX_PLOT_POINTS` samples of each array (same stride)."""
    stride = max(1, len(arrays[0]) // MAX_PLOT_POINTS)
    return [a[::stride] for a in arrays]


def plot_run(run: dict[str, Any], title: str, path: Path) -> Path:
    """Plot one solution against the measurements.

    One column per circuit. Rows: current (simulated, measured and, in PID
    mode, the reference current), current error simulated - measured, and
    voltage (controller output vs measured in PID mode, input voltage in
    voltage mode).

    Parameters
    ----------
    run : dict
        ``label`` (legend of the simulated curve, e.g. the method name),
        ``time`` (:class:`~numpy.ndarray`, [s]) and ``circuits``, a list of
        dicts with ``id``, ``current`` [A] on ``time``, and optionally
        ``reference`` [A] and ``voltage_input`` [V] on ``time``,
        ``current_exp`` and ``voltage_exp``: tuples ``(t, simulated,
        measured)`` sampled on the experimental time grid, and
        ``voltage_label`` (legend of the simulated voltage).
    title : str
        Figure title.
    path : Path
        PNG file to write; its directory is created if needed.

    Returns
    -------
    Path
        ``path``.
    """
    circuits = run["circuits"]
    fig, axes = plt.subplots(3, len(circuits), figsize=(7 * len(circuits), 9),
                             sharex=True, squeeze=False)
    t = np.asarray(run["time"])
    for k, c in enumerate(circuits):
        ax_i, ax_e, ax_u = axes[:, k]
        if c.get("reference") is not None:
            td, ref = _thin(t, np.asarray(c["reference"]))
            ax_i.plot(td, ref, color=AUX, lw=1, ls="--", label="reference current")
        td, cur = _thin(t, np.asarray(c["current"]))
        ax_i.plot(td, cur, color=SIM, lw=1.5, label=run["label"])
        head = c["id"]
        if c.get("current_exp") is not None:
            te, vs, ve = (np.asarray(a) for a in c["current_exp"])
            m = _metrics(vs, ve)
            te, vs, ve = _thin(te, vs, ve)
            ax_i.plot(te, ve, color=MEAS, lw=1.5, label="measured")
            ax_e.plot(te, vs - ve, color=SIM, lw=1.5)
            head += (f"\ncurrent: RMSE={m['rmse']:.3g} A, bias={m['bias']:.3g} A, "
                     f"NRMSE={m['nrmse']:.2%}")
        ax_i.set_title(head, fontsize=10)
        ax_i.set_ylabel("current [A]")
        ax_e.axhline(0, color=AUX, lw=0.8)
        ax_e.set_ylabel("current error sim − meas [A]")

        if c.get("voltage_exp") is not None:
            te, vs, ve = (np.asarray(a) for a in c["voltage_exp"])
            m = _metrics(vs, ve)
            te, vs, ve = _thin(te, vs, ve)
            ax_u.plot(te, vs, color=SIM, lw=1.5,
                      label=c.get("voltage_label", "controller output"))
            ax_u.plot(te, ve, color=MEAS, lw=1.5, label="measured")
            ax_u.set_title(f"voltage: RMSE={m['rmse']:.3g} V, bias={m['bias']:.3g} V",
                           fontsize=10)
        elif c.get("voltage_input") is not None:
            td, u = _thin(t, np.asarray(c["voltage_input"]))
            ax_u.plot(td, u, color=AUX, lw=1.5, label="input voltage")
        ax_u.set_ylabel("voltage [V]")
        ax_u.set_xlabel("time [s]")
        for ax in (ax_i, ax_e, ax_u):
            _style(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_file_name(case: str, method: str, rtol: float) -> str:
    """Base name shared by a run's ``.npz`` and ``.png`` files.

    Parameters
    ----------
    case : str
        Case name.
    method : str
        Solver method.
    rtol : float
        Relative tolerance.

    Returns
    -------
    str
        ``<case>_<method>_rtol<rtol>``, without extension.
    """
    return f"{case}_{method}_rtol{rtol:g}"


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a run saved by ``bench_solvers.py`` in the format of :func:`plot_run`.

    Parameters
    ----------
    path : Path
        ``runs/<case>_<method>_rtol<r>.npz`` file.

    Returns
    -------
    dict
        Run description for :func:`plot_run`.
    dict
        Settings: ``case``, ``method``, ``rtol``, ``max_step`` [s], ``strategy``.
    """
    with np.load(path) as d:
        meta = {k: d[k].item() for k in ("case", "method", "rtol", "max_step", "strategy")}
        circuits = []
        for cid in d["circuits"].tolist():
            c = {"id": cid, "current": d[f"{cid}/current"]}
            for key in ("reference", "voltage_input"):
                if f"{cid}/{key}" in d:
                    c[key] = d[f"{cid}/{key}"]
            for quantity in ("current", "voltage"):
                prefix = f"{cid}/exp_{quantity}"
                if f"{prefix}_time" in d:
                    c[f"{quantity}_exp"] = tuple(
                        d[f"{prefix}_{x}"] for x in ("time", "simulated", "measured"))
            circuits.append(c)
        run = {"label": meta["method"], "time": d["time"], "circuits": circuits}
    return run, meta


def plot_saved_run(path: Path, plots_dir: Path) -> Path:
    """Plot a run saved by ``bench_solvers.py``.

    Parameters
    ----------
    path : Path
        ``runs/<case>_<method>_rtol<r>.npz`` file.
    plots_dir : Path
        Directory of the PNG, named after the ``.npz`` file.

    Returns
    -------
    Path
        Path of the PNG written.
    """
    run, meta = load_run(path)
    title = (f"{meta['case']} — {meta['method']} rtol={meta['rtol']:g}, "
             f"max_step={meta['max_step']:g} s ({meta['strategy']} mode)")
    return plot_run(run, title, plots_dir / f"{path.stem}.png")


def plot_reference(case: str, reference: list[dict], runs: list[Path], path: Path) -> Path:
    """Plot a case's reference solution and each method's deviation from it.

    One column per circuit. Rows: current (reference solution vs measured),
    current deviation method - reference, voltage (reference solution vs
    measured), voltage deviation method - reference. All on the experimental
    time grid, where ``bench_solvers.py`` samples the reference.

    Parameters
    ----------
    case : str
        Case name, for the title.
    reference : list of dict
        Reference samples as cached in ``references/<case>-<digest>.json``:
        ``circuit``, ``quantity`` (``"current"`` or ``"voltage"``), ``time``
        [s], ``simulated`` and ``measured`` ([A] or [V]).
    runs : list of Path
        ``runs/<case>_<method>_rtol<r>.npz`` files of the case.
    path : Path
        PNG file to write; its directory is created if needed.

    Returns
    -------
    Path
        ``path``.
    """
    ref = {(s["circuit"], s["quantity"]): s for s in reference}
    circuits = list(dict.fromkeys(s["circuit"] for s in reference))
    loaded = [load_run(r) for r in runs]
    units = {"current": "A", "voltage": "V"}
    fig, axes = plt.subplots(4, len(circuits), figsize=(7 * len(circuits), 12),
                             sharex=True, squeeze=False)
    for k, cid in enumerate(circuits):
        for row, quantity in enumerate(units):
            ax_v, ax_d = axes[2 * row, k], axes[2 * row + 1, k]
            unit = units[quantity]
            s = ref.get((cid, quantity))
            if s is None:
                continue
            t, vr, vm = (np.asarray(s[x], dtype=float) for x in ("time", "simulated", "measured"))
            m = _metrics(vr, vm)
            td, vrd, vmd = _thin(t, vr, vm)
            ax_v.plot(td, vrd, color=SIM, lw=1.5, label="reference solution")
            ax_v.plot(td, vmd, color=MEAS, lw=1.5, label="measured")
            ax_v.set_title(f"{cid} — {quantity}: reference vs measured RMSE={m['rmse']:.3g} "
                           f"{unit}, bias={m['bias']:.3g} {unit}", fontsize=10)
            ax_v.set_ylabel(f"{quantity} [{unit}]")
            for run, meta in loaded:
                c = next((c for c in run["circuits"] if c["id"] == cid), None)
                exp = c.get(f"{quantity}_exp") if c else None
                if exp is None or len(exp[1]) != len(vr):
                    continue
                dev = np.asarray(exp[1], dtype=float) - vr
                rmse = float(np.sqrt(np.mean(dev**2)))
                color, _ = _method_style(meta["method"])
                td, dd = _thin(t, dev)
                ax_d.plot(td, dd, color=color, lw=1,
                          label=f"{meta['method']} rtol={meta['rtol']:g}: RMSE={rmse:.3g} {unit}")
            ax_d.axhline(0, color=AUX, lw=0.8)
            ax_d.set_ylabel(f"{quantity} − reference [{unit}]")
        axes[-1, k].set_xlabel("time [s]")
        for ax in axes[:, k]:
            _style(ax)
    fig.suptitle(f"{case} — reference solution and deviation of each method", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_references(output_dir: Path) -> list[Path]:
    """Plot every cached reference solution of a benchmark output directory.

    Parameters
    ----------
    output_dir : Path
        Directory holding ``references/*.json`` and ``runs/*.npz``.

    Returns
    -------
    list of Path
        ``plots/reference_<case>.png`` files written, one per case with a
        reference (the newest cache file if a case has several).
    """
    latest: dict[str, Path] = {}
    for f in sorted((output_dir / "references").glob("*.json"), key=lambda p: p.stat().st_mtime):
        latest[f.stem.rsplit("-", 1)[0]] = f
    written = []
    for case, f in latest.items():
        runs = sorted((output_dir / "runs").glob(f"{case}_*.npz"))
        written.append(plot_reference(case, json.loads(f.read_text()), runs,
                                      output_dir / "plots" / f"reference_{case}.png"))
    return written


def _current_errors(acc: pd.DataFrame) -> pd.DataFrame:
    """Worst current error over circuits per (case, method, rtol)."""
    current = acc[acc["quantity"] == "current"]
    return current.groupby(["case", "method", "rtol"], as_index=False)[
        ["rmse_exp", "rmse_ref"]].max()


def plot_work_precision(perf: pd.DataFrame, acc: pd.DataFrame, path: Path) -> Path | None:
    """Plot solve time against numerical error, one panel per case.

    The numerical error is the current RMSE vs the reference solution
    (``rmse_ref``), the worst over the circuits of the case. Each method is
    one line through its rtol values. The dashed vertical line is the median
    current RMSE vs measurements (``rmse_exp``): a solver whose numerical
    error lies well left of it is accurate enough for this model.

    Parameters
    ----------
    perf : :class:`~pandas.DataFrame`
        Contents of ``performance.csv``.
    acc : :class:`~pandas.DataFrame`
        Contents of ``accuracy.csv``.
    path : Path
        PNG file to write.

    Returns
    -------
    Path or None
        ``path``, or None if no run has a finite, positive ``rmse_ref``
        (e.g. run with ``--no-reference``).
    """
    if acc.empty:
        return None
    ok = perf[perf["status"] == "ok"][["case", "method", "rtol", "wall_s_median"]]
    data = ok.merge(_current_errors(acc), on=["case", "method", "rtol"])
    data = data[data["rmse_ref"] > 0]
    if data.empty:
        return None
    cases = list(dict.fromkeys(data["case"]))
    ncols = min(3, len(cases))
    nrows = -(-len(cases) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    for ax, case in zip(axes.flat, cases):
        sub = data[data["case"] == case]
        for method, g in sub.groupby("method", sort=False):
            g = g.sort_values("rtol")
            color, marker = _method_style(method)
            ax.plot(g["rmse_ref"], g["wall_s_median"], color=color, marker=marker,
                    ms=8, lw=2, label=method)
        ax.axvline(sub["rmse_exp"].median(), color=AUX, lw=1, ls="--",
                   label="RMSE vs measurements")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_ticks(ax)
        ax.set_title(case, fontsize=10)
        ax.set_xlabel("current RMSE vs reference solution [A]")
        ax.set_ylabel("wall time [s]")
        _style(ax)
    for ax in axes.flat[len(cases):]:
        ax.set_visible(False)
    fig.suptitle("Work-precision: solve time vs numerical error "
                 "(worst circuit; one point per rtol)", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_campaign(perf: pd.DataFrame, acc: pd.DataFrame,
                  cases: dict[str, dict[str, Any]], output_dir: Path) -> list[Path]:
    """Plot wall time and current RMSE vs measurements for every campaign run.

    One figure per rtol, ``campaign_rtol<rtol>.png``. Runs are ordered and
    labelled by campaign group. Each method is a dot in the run's slot (log wall time, linear
    RMSE). Runs that did not finish are marked ✕ at the bottom of the
    wall-time panel.

    Parameters
    ----------
    perf : :class:`~pandas.DataFrame`
        Contents of ``performance.csv``.
    acc : :class:`~pandas.DataFrame`
        Contents of ``accuracy.csv`` (may be empty).
    cases : dict of str to dict
        Campaign cases with their ``group`` and ``tag`` keys, as written by
        ``bench_campaign.py`` to ``cases.json``.
    output_dir : Path
        Directory of the PNG files.

    Returns
    -------
    list of Path
        PNG files written.
    """
    names = [n for n in cases if n in set(perf["case"])]
    methods = [m for m in METHOD_STYLE if m in set(perf["method"])]
    errors = _current_errors(acc) if not acc.empty else pd.DataFrame(
        columns=["case", "method", "rtol", "rmse_exp"])
    x = np.arange(len(names))
    width = 0.8 / max(len(methods), 1)
    paths = []
    for rtol in sorted(perf["rtol"].unique()):
        fig, (ax_t, ax_e) = plt.subplots(2, 1, figsize=(max(8, 0.5 * len(names) * len(methods)), 8),
                                         sharex=True)
        p = perf[perf["rtol"] == rtol].set_index(["case", "method"])
        e = errors[errors["rtol"] == rtol].set_index(["case", "method"])["rmse_exp"]
        for j, method in enumerate(methods):
            color, marker = _method_style(method)
            xj = x - 0.4 + (j + 0.5) * width
            wall = [p["wall_s_median"].get((n, method), np.nan)
                    if p["status"].get((n, method)) == "ok" else np.nan for n in names]
            ax_t.plot(xj, wall, ls="none", marker=marker, ms=8, color=color, label=method)
            failed = [xi for xi, n in zip(xj, names)
                      if (n, method) in p.index and p["status"][(n, method)] != "ok"]
            ax_t.plot(failed, [0.03] * len(failed), ls="none", marker="x", ms=8, mew=2,
                      color=color, transform=ax_t.get_xaxis_transform())
            ax_e.plot(xj, [e.get((n, method), np.nan) for n in names], ls="none",
                      marker=marker, ms=8, color=color)
        ax_t.set_yscale("log")
        ax_t.set_ylabel("wall time [s]")
        _plain_ticks(ax_t)
        ax_e.set_ylim(bottom=0)
        ax_e.set_ylabel("current RMSE vs measurements [A]\n(worst circuit)")
        ax_e.set_xticks(x, [cases[n]["tag"] for n in names], rotation=90, fontsize=8)
        groups = [cases[n]["group"] for n in names]
        for ax in (ax_t, ax_e):
            for i in range(1, len(names)):
                if groups[i] != groups[i - 1]:
                    ax.axvline(i - 0.5, color=AUX, lw=0.8)
            _style(ax)
        for g in dict.fromkeys(groups):
            idx = [i for i, gi in enumerate(groups) if gi == g]
            ax_t.text((idx[0] + idx[-1]) / 2, 1.01, g, ha="center", va="bottom",
                      fontsize=9, transform=ax_t.get_xaxis_transform())
        ax_t.legend(frameon=False, fontsize=9, ncols=len(methods),
                    loc="lower center", bbox_to_anchor=(0.5, 1.06))
        fig.suptitle(f"Campaign runs — rtol={rtol:g} (✕: run did not finish)", fontsize=12)
        fig.tight_layout()
        path = output_dir / f"campaign_rtol{rtol:g}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_results_dir(output_dir: Path, runs: list[Path] | None = None) -> list[Path]:
    """Write every plot of a ``bench_solvers`` / ``bench_campaign`` output directory.

    Parameters
    ----------
    output_dir : Path
        Directory holding ``performance.csv``, ``accuracy.csv`` and ``runs/``.
        A ``cases.json`` whose cases have a ``group`` key marks a campaign
        directory and adds the campaign figures.
    runs : list of Path, optional
        Run files to plot; default: every ``runs/*.npz``.

    Returns
    -------
    list of Path
        PNG files written.
    """
    perf = pd.read_csv(output_dir / "performance.csv")
    acc_path = output_dir / "accuracy.csv"
    acc = pd.read_csv(acc_path) if acc_path.stat().st_size > 1 else pd.DataFrame()
    written = []
    if (p := plot_work_precision(perf, acc, output_dir / "work_precision.png")) is not None:
        written.append(p)
    else:
        print("work_precision.png skipped: no run has an error vs the reference solution")
    cases_path = output_dir / "cases.json"
    if cases_path.exists():
        cases = json.loads(cases_path.read_text())
        if cases and all("group" in c for c in cases.values()):
            written += plot_campaign(perf, acc, cases, output_dir)
    written += plot_references(output_dir)
    for run in runs if runs is not None else sorted((output_dir / "runs").glob("*.npz")):
        written.append(plot_saved_run(run, output_dir / "plots"))
    return written


def main(argv: list[str] | None = None) -> int:
    """Regenerate the plots of a benchmark output directory.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", type=Path,
                        help="Directory written by bench_solvers.py or bench_campaign.py")
    args = parser.parse_args(argv)
    for path in plot_results_dir(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
