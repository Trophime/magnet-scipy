"""Summarise the M9 PID solver sweep produced by ``run_pid_sweep.sh``.

Writes ``summary.md`` and ``summary.csv`` (performance, electrical metrics per magnet,
calorimetric outlet temperature) into the output directory, plus ``summary.png``
(cost vs error) and ``plots/<config>_<method>.png`` (each run vs the measurements)
unless ``--no-plots`` is given.

Usage
-----
::

    python build_summary.py <sweep dir> [--output DIR] [--pig-shift 8.4]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
import plot_results  # noqa: E402


def _import_magnetcooling():
    """Import the Tout helpers from python_magnetcooling.

    ``main`` of python_magnetcooling (2026-11-12) fails in its package ``__init__``
    (see python_magnetcooling/prompts/fix_import_regressions.md). If so, and if
    ``PYTHON_MAGNETCOOLING_SRC`` points to the ``python_magnetcooling/python_magnetcooling``
    source directory, the submodule is loaded without running ``__init__``.
    """
    try:
        from python_magnetcooling.heatexchanger_primary import (
            estimate_temperature_elevation,
            mixing_temp,
        )
    except (ImportError, NameError):
        src = os.environ.get("PYTHON_MAGNETCOOLING_SRC")
        if not src:
            raise
        for name in [m for m in sys.modules if m.startswith("python_magnetcooling")]:
            del sys.modules[name]
        pkg = types.ModuleType("python_magnetcooling")
        pkg.__path__ = [src]
        sys.modules["python_magnetcooling"] = pkg
        from python_magnetcooling.heatexchanger_primary import (
            estimate_temperature_elevation,
            mixing_temp,
        )
    return estimate_temperature_elevation, mixing_temp


estimate_temperature_elevation, mixing_temp = _import_magnetcooling()

T_END = 1100.0
MAG = {"M23072801": ("GR1", "internes", "1"), "M9Bitters_18MW": ("GR2", "externes", "2")}
CONFIGS = ["circuits-pid", "circuits-pid-temp"]
METHODS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]


def _load_inputs(wd: Path):
    """Pupitre record (Tin, Tout, flows, pressures) and pigbrother current/voltage per magnet."""
    pup = pd.read_csv(wd / "M9_2023.09.16---17:55:22.txt", sep="\t", skiprows=1)
    ts = pd.to_datetime(pup["Date"] + " " + pup["Time"], format="%Y.%m.%d %H:%M:%S")
    pup["t"] = (ts - ts.iloc[0]).dt.total_seconds()
    pup = pup[pup["t"] <= T_END].reset_index(drop=True)
    exp = {}
    for cid, (gr, tag, _) in MAG.items():
        ci = pd.read_csv(wd / f"M9_Overview_230916-1755_Courants_Alimentations_Courant_{gr}_vs_Time.csv", sep="\t")
        vu = pd.read_csv(wd / f"M9_Overview_230916-1755_Tensions_Aimant_ALL_{tag}_vs_Time.csv", sep="\t")
        exp[cid] = {
            "current": (ci["time"].to_numpy(), ci["current"].to_numpy()),
            "voltage": (vu["time"].to_numpy(), vu["voltage"].to_numpy()),
        }
    return pup, exp


def _err_stats(t, y, te, ye):
    """RMSE and MAE of sim - exp at experimental samples inside the simulated window."""
    m = (te >= t.min()) & (te <= t.max())
    e = np.interp(te[m], t, y) - ye[m]
    return np.sqrt(np.mean(e**2)), np.mean(np.abs(e))


def _ctout(pup, power, pig_shift):
    """Calorimetric mixed outlet temperature at pupitre times; ``power[cid](t)`` is on pigbrother time."""
    tp = pup["t"].to_numpy()
    tout = {}
    for cid, (_, _, k) in MAG.items():
        p_w = np.clip(power[cid](tp - pig_shift), 0.0, None)
        tout[cid] = np.array([
            estimate_temperature_elevation(p, q * 1e-3, ti, bp, hp)
            for p, q, ti, bp, hp in zip(p_w, pup[f"Flow{k}"], pup[f"Tin{k}"], pup["BP"], pup[f"HP{k}"])
        ])
    return np.array([
        mixing_temp(a, bp, ta, b, bp, tb)
        for a, b, bp, ta, tb in zip(pup["Flow1"], pup["Flow2"], pup["BP"], tout["M23072801"], tout["M9Bitters_18MW"])
    ])


def _lowpass(y, tau, dt):
    """First-order low-pass filter on a uniform grid of step dt."""
    if tau <= 1e-6:
        return y
    a = 1.0 - np.exp(-dt / tau)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = out[i - 1] + a * (y[i] - out[i - 1])
    return out


def _tout_stats(pup, mix):
    """No-lag error and fit of Tout = Tin_mix + k * F_tau[cTout - Tin_mix](t - L)."""
    tp, meas = pup["t"].to_numpy(), pup["Tout"].to_numpy()
    q1, q2 = pup["Flow1"].to_numpy(), pup["Flow2"].to_numpy()
    tin_mix = (q1 * pup["Tin1"] + q2 * pup["Tin2"]).to_numpy() / (q1 + q2)
    tg = np.arange(0.0, tp[-1] + 1e-9, 0.5)  # uniform grid: the pupitre record has an 11 s gap
    rise = np.interp(tg, tp, mix - tin_mix)
    win = tp >= 120

    def model(p):
        return np.interp(tp, tg + p[0], _lowpass(rise, p[2], tg[1] - tg[0]), left=0.0) * p[1] + tin_mix

    r = least_squares(lambda p: (model(p) - meas)[win], [20, 1.0, 20], bounds=([0, 0.5, 0], [120, 1.5, 300]))
    e0, e = (mix - meas)[win], (model(r.x) - meas)[win]
    plateau = (tp >= 550) & (tp <= 800)
    return {
        "tout_rmse0": np.sqrt(np.mean(e0**2)), "tout_mae0": np.mean(np.abs(e0)),
        "tout_plateau": np.mean((mix - meas)[plateau]), "tout_max": mix.max(),
        "L": r.x[0], "k": r.x[1], "tau": r.x[2], "tout_rmse_fit": np.sqrt(np.mean(e**2)),
    }


def _parse_log(path: Path) -> dict:
    """Wall, CPU, peak RSS and status from a run log (GNU ``time -v`` or BSD ``time -l``)."""
    txt = path.read_text(errors="replace")
    out = {"status": "ok" if "simulation completed successfully" in txt else "failed"}
    if m := re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", txt):  # GNU
        parts = [float(x) for x in m.group(1).split(":")]
        out["wall_s"] = sum(v * 60**i for i, v in enumerate(reversed(parts)))
        u = re.search(r"User time \(seconds\): (\S+)", txt)
        s = re.search(r"System time \(seconds\): (\S+)", txt)
        out["cpu_s"] = float(u.group(1)) + float(s.group(1))
        if m := re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt):
            out["rss_mib"] = int(m.group(1)) / 1024
    elif m := re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", txt):  # BSD / macOS
        out["wall_s"], out["cpu_s"] = float(m.group(1)), float(m.group(2)) + float(m.group(3))
        if m := re.search(r"(\d+)\s+maximum resident set size", txt):
            out["rss_mib"] = int(m.group(1)) / 1024**2
    if (m := re.findall(r"Exit status: (\d+)", txt)) and int(m[-1]) == 124:
        out["status"] = "timeout"
    if "Traceback" in txt:
        out["status"] = "error"
    return out


def _f(v, fmt):
    """Format a value, or a dash when missing."""
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


def build(sweep: Path, wd: Path, pig_shift: float) -> pd.DataFrame:
    """Collect one row per (config, method), plus the measured-power Tout baseline."""
    pup, exp = _load_inputs(wd)
    rows = []
    pw = {c: (lambda t, d=exp[c]: np.interp(t, *d["voltage"]) * np.interp(t, *d["current"])) for c in MAG}
    rows.append(dict(config="measured U·I", method="n/a", status="ok", **_tout_stats(pup, _ctout(pup, pw, pig_shift))))
    for cfg in CONFIGS:
        for meth in METHODS:
            tag = f"{cfg}_{meth}"
            row = {"config": cfg, "method": meth}
            if (sweep / f"{tag}.log").exists():
                row.update(_parse_log(sweep / f"{tag}.log"))
            npz = sweep / f"{tag}.npz"
            if row.get("status") == "ok" and npz.exists():
                d = np.load(npz, allow_pickle=True)
                t = d["time"]
                row["points"] = t.size
                for cid in MAG:
                    i, iref, u, p = (d[f"{cid}_{k}"] for k in ("current", "reference", "voltage", "power"))
                    e = i - iref
                    row[f"{cid}_trk_rmse"], row[f"{cid}_trk_mae"] = np.sqrt(np.mean(e**2)), np.mean(np.abs(e))
                    row[f"{cid}_trk_max"] = np.abs(e).max()
                    row[f"{cid}_I_rmse"], row[f"{cid}_I_mae"] = _err_stats(t, i, *exp[cid]["current"])
                    row[f"{cid}_U_rmse"], row[f"{cid}_U_mae"] = _err_stats(t, u, *exp[cid]["voltage"])
                    row[f"{cid}_Pmax"] = p.max() / 1e6
                pw = {c: (lambda tt, t0=t, pp=d[f"{c}_power"]: np.interp(tt, t0, pp)) for c in MAG}
                row.update(_tout_stats(pup, _ctout(pup, pw, pig_shift)))
            rows.append(row)
            print(f"{tag:28s} {row.get('status', 'missing')}", flush=True)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, pig_shift: float) -> str:
    """Render the three summary tables."""
    sims = df[df.config != "measured U·I"]
    md = ["## Performance (0–1100 s, max_step = 0.1 s)\n",
          "| Config | Solver | Status | Wall [s] | CPU [s] | Peak RAM [MiB] | Solution points |",
          "|---|---|---|---:|---:|---:|---:|"]
    for _, r in sims.iterrows():
        md.append(f"| {r.config} | {r.method} | {r.get('status', 'missing')} | {_f(r.get('wall_s'), '.1f')} | "
                  f"{_f(r.get('cpu_s'), '.1f')} | {_f(r.get('rss_mib'), '.0f')} | {_f(r.get('points'), ',.0f')} |")
    for cid in MAG:
        md += [f"\n## {cid}: tracking and comparison with experiment\n",
               "| Config | Solver | P_max [MW] | Tracking RMSE / MAE / max [A] | Current vs exp RMSE / MAE [A] "
               "| Voltage vs exp RMSE / MAE [V] |",
               "|---|---|---:|---|---|---|"]
        for _, r in sims.iterrows():
            g = lambda k, fmt: _f(r.get(f"{cid}_{k}"), fmt)  # noqa: E731
            md.append(f"| {r.config} | {r.method} | {g('Pmax', '.2f')} | "
                      f"{g('trk_rmse', '.4f')} / {g('trk_mae', '.4f')} / {g('trk_max', '.4f')} | "
                      f"{g('I_rmse', '.1f')} / {g('I_mae', '.1f')} | {g('U_rmse', '.2f')} / {g('U_mae', '.2f')} |")
    md += [f"\n## Outlet temperature: calorimetric cTout (python_magnetcooling) vs measured Tout, "
           f"sources aligned ({pig_shift} s)\n",
           "| Power source | Solver | No lag RMSE / MAE [K] | Plateau offset [K] | cTout max [°C] | Fit L [s] | k "
           "| τ [s] | Fit RMSE [K] |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        md.append(f"| {r.config} | {r.method} | {_f(r.get('tout_rmse0'), '.2f')} / {_f(r.get('tout_mae0'), '.2f')} | "
                  f"{_f(r.get('tout_plateau'), '+.2f')} | {_f(r.get('tout_max'), '.2f')} | {_f(r.get('L'), '.1f')} | "
                  f"{_f(r.get('k'), '.3f')} | {_f(r.get('tau'), '.1f')} | {_f(r.get('tout_rmse_fit'), '.2f')} |")
    return "\n".join(md) + "\n"


def plot_summary(df: pd.DataFrame, path: Path) -> Path:
    """Plot cost vs error for every (config, method) of the sweep.

    Panels: current and voltage RMSE vs measurements per magnet, and Tout RMSE
    (no lag, and after the lag fit), each against the wall time of the full
    CLI run. Colour and marker give the method, filled markers are
    ``circuits-pid`` and hollow ones ``circuits-pid-temp``. The Tout panels
    show the measured-U·I baseline as a dashed line.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Output of :func:`build`.
    path : Path
        PNG file to write.

    Returns
    -------
    Path
        ``path``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    plain = FuncFormatter(lambda v, _: f"{v:g}")
    sims = df[(df.config != "measured U·I") & (df["status"] == "ok")]
    base = df[df.config == "measured U·I"].iloc[0]
    cids = list(MAG)
    panels = [
        [(f"{cids[0]}_I_rmse", f"{cids[0]}: current RMSE vs exp [A]"),
         (f"{cids[1]}_I_rmse", f"{cids[1]}: current RMSE vs exp [A]"),
         ("tout_rmse0", "Tout RMSE, no lag [K]")],
        [(f"{cids[0]}_U_rmse", f"{cids[0]}: voltage RMSE vs exp [V]"),
         (f"{cids[1]}_U_rmse", f"{cids[1]}: voltage RMSE vs exp [V]"),
         ("tout_rmse_fit", "Tout RMSE, lag fit [K]")],
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    for row, specs in zip(axes, panels):
        for ax, (col, label) in zip(row, specs):
            for _, r in sims.iterrows():
                color, marker = plot_results.METHOD_STYLE.get(r.method, (plot_results.AUX, "o"))
                filled = r.config == CONFIGS[0]
                ax.plot(r.get("wall_s"), r.get(col), ls="none", marker=marker, ms=9, mew=2,
                        color=color, markerfacecolor=color if filled else "none")
            top = pd.to_numeric(sims.get(col), errors="coerce").max()
            if col.startswith("tout"):
                ax.axhline(base[col], color=plot_results.AUX, lw=1, ls="--")
                top = np.nanmax([top, base[col]])
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(plain)
            ax.xaxis.set_minor_formatter(plain)
            if np.isfinite(top) and top > 0:
                ax.set_ylim(0, 1.12 * top)
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("wall time of the CLI run [s]")
    methods = [Line2D([], [], ls="none", marker=mk, ms=9, color=c, label=m)
               for m, (c, mk) in plot_results.METHOD_STYLE.items() if m in set(sims.method)]
    configs = [Line2D([], [], ls="none", marker="o", ms=9, mew=2, color="k",
                      markerfacecolor="k" if i == 0 else "none", label=cfg)
               for i, cfg in enumerate(CONFIGS)]
    configs.append(Line2D([], [], color=plot_results.AUX, ls="--", label="measured U·I (Tout)"))
    fig.legend(handles=methods + configs, loc="lower center", ncols=len(methods) + len(configs),
               frameon=False, fontsize=9)
    fig.suptitle(f"M9 PID sweep: cost vs error (0–{T_END:g} s, max_step = 0.1 s)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_runs(df: pd.DataFrame, sweep: Path, wd: Path, plots_dir: Path) -> list[Path]:
    """Plot each successful run of the sweep against the measurements.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Output of :func:`build`.
    sweep : Path
        Directory written by ``run_pid_sweep.sh`` (``<config>_<method>.npz``).
    wd : Path
        ``examples/M9_M23072801`` data directory.
    plots_dir : Path
        Directory of the ``<config>_<method>.png`` files.

    Returns
    -------
    list of Path
        PNG files written.
    """
    _, exp = _load_inputs(wd)
    written = []
    for _, r in df[(df.config != "measured U·I") & (df["status"] == "ok")].iterrows():
        npz = sweep / f"{r.config}_{r.method}.npz"
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=True)
        t = d["time"]
        circuits = []
        for cid in MAG:
            c = {"id": cid, "current": d[f"{cid}_current"], "reference": d[f"{cid}_reference"]}
            for quantity in ("current", "voltage"):
                te, ve = exp[cid][quantity]
                m = (te >= t.min()) & (te <= t.max())
                c[f"{quantity}_exp"] = (te[m], np.interp(te[m], t, d[f"{cid}_{quantity}"]), ve[m])
            circuits.append(c)
        title = f"{r.config} — {r.method} rtol=1e-6, max_step=0.1 s (pid mode)"
        written.append(plot_results.plot_run({"label": r.method, "time": t, "circuits": circuits},
                                             title, plots_dir / f"{r.config}_{r.method}.png"))
    return written


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep", type=Path, help="directory written by run_pid_sweep.sh")
    ap.add_argument("--wd", type=Path, default=here.parent, help="examples/M9_M23072801 data directory")
    ap.add_argument("--output", type=Path, help="where to write summary.md/csv (default: the sweep directory)")
    ap.add_argument("--pig-shift", type=float, default=8.4,
                    help="pigbrother CSV t == pupitre t + shift [s] (see lag_sources.py)")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip summary.png and the per-run plots/<config>_<method>.png")
    args = ap.parse_args(argv)
    out = args.output or args.sweep
    out.mkdir(parents=True, exist_ok=True)
    df = build(args.sweep, args.wd, args.pig_shift)
    df.to_csv(out / "summary.csv", index=False)
    md = to_markdown(df, args.pig_shift)
    (out / "summary.md").write_text(md)
    print(md)
    if not args.no_plots:
        plots = [plot_summary(df, out / "summary.png")]
        plots += plot_runs(df, args.sweep, args.wd, out / "plots")
        print(f"{len(plots)} plots written to {out}")


if __name__ == "__main__":
    main()
