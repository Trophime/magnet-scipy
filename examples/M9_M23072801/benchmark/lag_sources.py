"""Clock offset between the pupitre (.txt) and pigbrother (.tdms / exported CSV) records of the 2023-09-16 17:55 run.

Prints the lag L [s] such that ``pigbrother(t_csv) == pupitre(t_csv + L)``, each file on its own t = 0,
from four signals recorded by both systems. Pass L to ``build_summary.py --pig-shift``.

Usage
-----
::

    python lag_sources.py [--wd examples/M9_M23072801]

Requires ``nptdms``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile

# pupitre column(s) <-> pigbrother tdms channel <-> exported CSV
SIGNALS = {
    "I_H": (["Icoil1"], ("Courants_Alimentations", "Courant GR1"),
            "M9_Overview_230916-1755_Courants_Alimentations_Courant_GR1_vs_Time.csv"),
    "I_B": (["Icoil15"], ("Courants_Alimentations", "Courant GR2"),
            "M9_Overview_230916-1755_Courants_Alimentations_Courant_GR2_vs_Time.csv"),
    "U_H": ([f"Ucoil{i}" for i in range(1, 8)], ("Tensions_Aimant", "ALL internes"),
            "M9_Overview_230916-1755_Tensions_Aimant_ALL_internes_vs_Time.csv"),
    "U_B": (["Ucoil15", "Ucoil16"], ("Tensions_Aimant", "ALL externes"),
            "M9_Overview_230916-1755_Tensions_Aimant_ALL_externes_vs_Time.csv"),
}


def best_lag(t_pig, y_pig, t_pup, y_pup, lags):
    """Lag minimising the residual of a gain+offset fit ``y_pig ~ a*y_pup(t+L) + b``.

    Returns the lag refined by a parabola through the minimum [s], the residual RMSE there,
    the residual RMSE at zero lag, and the fitted gain.
    """
    res = []
    for lag in lags:
        m = (t_pig + lag >= t_pup[0]) & (t_pig + lag <= t_pup[-1])
        x = np.interp(t_pig[m] + lag, t_pup, y_pup)
        a = np.c_[x, np.ones_like(x)]
        coef, *_ = np.linalg.lstsq(a, y_pig[m], rcond=None)
        res.append((np.sqrt(np.mean((y_pig[m] - a @ coef) ** 2)), coef[0]))
    rmse = np.array([r[0] for r in res])
    i = int(np.argmin(rmse))
    lag = lags[i]
    if 0 < i < len(lags) - 1:
        y0, y1, y2 = rmse[i - 1 : i + 2]
        lag += 0.5 * (lags[1] - lags[0]) * (y0 - y2) / (y0 - 2 * y1 + y2)
    return lag, rmse[i], rmse[int(np.argmin(np.abs(lags)))], res[i][1]


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wd", type=Path, default=Path(__file__).resolve().parent.parent)
    wd = ap.parse_args(argv).wd

    pup = pd.read_csv(wd / "M9_2023.09.16---17:55:22.txt", sep="\t", skiprows=1)  # line 1: magnet metadata
    ts = pd.to_datetime(pup["Date"] + " " + pup["Time"], format="%Y.%m.%d %H:%M:%S")
    tp = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    gaps = np.flatnonzero(np.diff(tp) > 1.5)
    print(f"pupitre:    first row {ts.iloc[0]} (local, whole seconds), {len(tp)} rows, "
          f"gaps: {[(float(tp[g]), float(np.diff(tp)[g])) for g in gaps]} (t, length) [s]")

    tdms = TdmsFile.read(wd / "M9_Overview_230916-1755.tdms")
    ch0 = tdms[SIGNALS["I_H"][1][0]][SIGNALS["I_H"][1][1]]
    start = pd.Timestamp(ch0.properties["wf_start_time"]).tz_localize("UTC").tz_convert("Europe/Paris")
    t0_csv = pd.read_csv(wd / SIGNALS["I_H"][2], sep="\t")["time"].iloc[0]
    nominal = (start.tz_localize(None) - ts.iloc[0]).total_seconds()
    print(f"pigbrother: wf_start_time {start}, CSV export time = sample index + {t0_csv} s")
    print(f"expected from timestamps: L = {nominal:.2f} - {t0_csv} = {nominal - t0_csv:+.2f} s (±1 s pupitre resolution)\n")

    lags = np.round(np.arange(-30, 30.001, 0.1), 3)
    print("signal   L (CSV time)   residual@L   residual@0   gain")
    found = []
    for name, (cols, (grp, chan), csv) in SIGNALS.items():
        y_pup = pup[cols].sum(axis=1).to_numpy()
        d = pd.read_csv(wd / csv, sep="\t")
        lag, r, r0, gain = best_lag(d["time"].to_numpy(), d.iloc[:, 0].to_numpy(), tp, y_pup, lags)
        unit = "A" if name.startswith("I") else "V"
        print(f"{name:6s}   {lag:+8.2f} s    {r:8.2f} {unit}   {r0:8.2f} {unit}   {gain:.4f}")
        found.append(lag)
    print(f"\nmean L = {np.mean(found):+.2f} s (spread {np.ptp(found):.2f} s) -> build_summary.py --pig-shift "
          f"{np.mean(found):.1f}")


if __name__ == "__main__":
    main()
