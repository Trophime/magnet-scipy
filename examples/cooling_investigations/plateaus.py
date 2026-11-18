"""Collect steady plateaus (pupitre only) from M9/M10 files where ``Ptot`` is recorded.

Input is the output of ``scan_ptot_rule.py``; files are kept when evaluated,
with at least 600 powered rows and ``max |Pmagnet * (Ptot == 0)|`` ≤ 1 MW.
A plateau is at least 7 min with ``Pmagnet`` > 1 MW, stable within 0.05 MW
and both pump speeds within 5 rpm (5 min windows); its 2 min medians are
averaged after 2 min of settling.

Example
-------
::

    python examples/cooling_investigations/plateaus.py ptot_rule_scan.csv plateaus.csv
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

COLS = [
    "Pmagnet",
    "Ptot",
    "Idcct1",
    "Idcct2",
    "Idcct3",
    "Idcct4",
    "Rpm1",
    "Rpm2",
    "Flow1",
    "Flow2",
    "HP1",
    "HP2",
    "BP",
    "Tin1",
    "Tin2",
    "Tout",
    "TAlimout",
    "teb",
    "tsb",
    "debitbrut",
]


def plateaus(args: tuple[Path, str, str]) -> list[dict]:
    """Return the steady plateaus of one pupitre file.

    Parameters
    ----------
    args : tuple
        ``(records, housing, name)``: pupitre root directory
        (:class:`~pathlib.Path`), housing and file name.

    Returns
    -------
    list of dict
        One dict per plateau: medians of :data:`COLS` plus ``housing``,
        ``file``, ``start`` (row index) and ``dur_s`` [s].
    """
    records, housing, name = args
    try:
        d = pd.read_csv(
            records / housing / name, sep="\t", skiprows=1, usecols=lambda c: c in COLS
        )
    except Exception:
        return []
    if not set(COLS[:12]) <= set(d.columns):
        return []
    d = d.apply(pd.to_numeric, errors="coerce")
    m = d.rolling(120, center=True, min_periods=60).median()
    steady = (
        (m.Pmagnet > 1)
        & (d.Pmagnet.rolling(300, center=True).std() < 0.05)
        & (d.Rpm1.rolling(300, center=True).std() < 5)
        & (d.Rpm2.rolling(300, center=True).std() < 5)
    )
    seg = (steady != steady.shift()).cumsum()
    out = []
    for _, s in steady[steady].groupby(seg[steady]):
        if len(s) < 420:
            continue
        p = m.loc[s.index[120] : s.index[-1]].median()
        p["housing"], p["file"], p["start"], p["dur_s"] = (
            housing,
            name,
            int(s.index[0]),
            len(s),
        )
        out.append(p.to_dict())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scan", type=Path, help="output of scan_ptot_rule.py")
    ap.add_argument("out", type=Path, help="plateaus CSV to write")
    ap.add_argument(
        "--records",
        type=Path,
        default=Path.home() / "LNCMIG-Data/records/srv-data-install",
        help="pupitre root directory (default: ~/LNCMIG-Data/records/srv-data-install)",
    )
    args = ap.parse_args()
    scan = pd.read_csv(args.scan)
    ok = scan[
        scan.evaluated.astype(bool)
        & (scan.powered_rows >= 600)
        & (scan.max_abs <= 1.0)
        & scan.housing.isin(["M9", "M10"])
    ]
    jobs = [(args.records, h, f) for h, f in zip(ok.housing, ok.file)]
    with ProcessPoolExecutor() as ex:
        rows = [r for rs in ex.map(plateaus, jobs, chunksize=8) for r in rs]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(ok)} files -> {len(rows)} plateaus -> {args.out}")
