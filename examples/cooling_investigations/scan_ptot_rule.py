"""Evaluate the ``Ptot_not_recorded`` check rule on pupitre files (one row per file).

Statistics of ``Pmagnet * (Ptot == 0)`` are computed exactly as
``python_magnetrun.runetl._rule_checks`` does (``RULE_STATS`` on
``DataFrame.eval``, NaN ignored). As an independent reference, the longest
run of consecutive powered rows (``Pmagnet`` > 0.5 MW) with ``Ptot == 0`` is
also recorded.

Example
-------
::

    python examples/cooling_investigations/scan_ptot_rule.py ptot_rule_scan.csv \\
        ~/LNCMIG-Data/records/srv-data-install/M9/*.txt \\
        ~/LNCMIG-Data/records/srv-data-install/M10/*.txt
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from python_magnetrun.runetl import RULE_STATS

EXPR = "Pmagnet * (Ptot == 0)"
POWERED_MW = 0.5


def scan(path: str) -> dict:
    """Return the rule statistics and the longest powered ``Ptot == 0`` run of one file.

    Parameters
    ----------
    path : str
        Pupitre ``.txt`` file; its parent directory name is the housing.

    Returns
    -------
    dict
        ``housing``, ``file``, ``evaluated`` and, when evaluated, ``max_abs``,
        ``p99_abs``, ``mean_abs`` [MW], ``rows``, ``powered_rows``,
        ``powered_zero_rows``, ``longest_zero_run_s`` [s], ``Pmagnet_max`` [MW].
    """
    row = {"housing": Path(path).parent.name, "file": Path(path).name}
    try:
        header = pd.read_csv(path, sep="\t", skiprows=1, nrows=0).columns
        if not {"Pmagnet", "Ptot"} <= set(header):
            return row | {"evaluated": False}
        d = pd.read_csv(path, sep="\t", skiprows=1, usecols=["Pmagnet", "Ptot"])
    except Exception:
        return row | {"evaluated": False}
    d = d.apply(pd.to_numeric, errors="coerce")
    if d.empty:
        return row | {"evaluated": False}
    values = d.eval(EXPR)
    row |= {k: float(f(values)) for k, f in RULE_STATS.items()}
    z = ((d.Pmagnet > POWERED_MW) & (d.Ptot == 0)).to_numpy()
    longest = 0
    if z.any():  # longest run of consecutive True
        edges = np.diff(np.concatenate(([0], z.astype(np.int8), [0])))
        longest = int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())
    return row | {
        "evaluated": True,
        "rows": len(d),
        "powered_rows": int((d.Pmagnet > POWERED_MW).sum()),
        "powered_zero_rows": int(z.sum()),
        "longest_zero_run_s": longest,
        "Pmagnet_max": float(d.Pmagnet.max()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="CSV to write, one row per file")
    ap.add_argument("files", nargs="+", help="pupitre .txt files")
    args = ap.parse_args()
    out = args.out
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(scan, args.files, chunksize=16))
    keys = [
        "housing",
        "file",
        "evaluated",
        "max_abs",
        "p99_abs",
        "mean_abs",
        "rows",
        "powered_rows",
        "powered_zero_rows",
        "longest_zero_run_s",
        "Pmagnet_max",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} files -> {out}")
