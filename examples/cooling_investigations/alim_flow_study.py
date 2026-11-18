"""Implied power-supply cooling flow per long plateau, over all M9/M10 runs.

For each run with a plateau of at least 30 min above 5 MW (from
``plateaus.py``) and pigbrother Overview files covering it, the energy
balance of ``examples/energy_balance.py`` is computed with its defaults
(``Ptot`` includes the magnet pumps, 0.023 MW per pump per (1000 rpm)³).
For each plateau of at least 30 min, the medians of its last 10 min (the
``TAlimout`` response is slow) give
``F_alim_implied = P_alim / (rho*cp*(TAlimout - Tin))`` [l/s] with
``P_alim = Ptot - sum Puissance_A - P_pumps`` (transformer + converter
losses).

Example
-------
::

    python examples/cooling_investigations/alim_flow_study.py plateaus.csv alim_flow_study.csv

Needs ``nptdms``, ``pandas`` and, for dated housing wiring, ``python_magnetrun``.
"""

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import energy_balance as eb  # noqa: E402

warnings.filterwarnings("ignore")


def overview_index(records: Path, housing: str) -> pd.DataFrame:
    """Return the pigbrother Overview files of a housing with their naive UTC start.

    Parameters
    ----------
    records : :class:`~pathlib.Path`
        Records root (contains ``pbsurv/<housing>/Overview``).
    housing : str
        Housing name, e.g. ``"M9"``.

    Returns
    -------
    :class:`~pandas.DataFrame`
        Columns ``path`` and ``start`` (from the file name, site local time
        converted to naive UTC).
    """
    files = sorted(
        (records / "pbsurv" / housing / "Overview").glob(f"{housing}_Overview_*.tdms")
    )
    stamps = pd.to_datetime(
        [f.stem.rsplit("_", 1)[1] for f in files], format="%y%m%d-%H%M", errors="coerce"
    )
    utc = pd.Series(stamps).dt.tz_localize(
        eb.SITE_TZ, ambiguous="NaT", nonexistent="shift_forward"
    )
    return pd.DataFrame(
        {"path": files, "start": utc.dt.tz_convert("UTC").dt.tz_localize(None)}
    ).dropna()


def run(args: tuple[Path, str, str, pd.DataFrame]) -> list[dict]:
    """Return the implied flow and power terms of each long plateau of one run.

    Parameters
    ----------
    args : tuple
        ``(records, housing, name, overviews)``: records root, housing,
        pupitre file name, and :func:`overview_index` of the housing.

    Returns
    -------
    list of dict
        One dict per plateau ≥ 30 min above 5 MW: ``start`` (UTC),
        ``dur_min``, ``lag_s``, powers [MW] (``P_mag``, ``P_pup``, ``Ptot``,
        ``P_busbar``, ``P_pump_el``, ``P_alim``), ``Tin``, ``TAlim`` [°C],
        ``F_alim_implied`` [l/s], pump speeds and flows. Empty when the run
        has no pigbrother data or fails to load.
    """
    records, housing, name, ovw = args
    try:
        pup = eb.read_pupitre(records / "srv-data-install" / housing / name)
        t0, t1 = pup.index[0], pup.index[-1]
        sel = ovw[
            (ovw.start <= t1) & (ovw.start >= t0 - pd.Timedelta("36h"))
        ].path.tolist()
        if not sel:
            return []
        pb = eb.read_pigbrother(sel)
        pb = pb[
            (pb.index >= t0 - pd.Timedelta("5min"))
            & (pb.index <= t1 + pd.Timedelta("5min"))
        ]
        if len(pb) < 1800:
            return []
        wiring, _ = eb.housing_wiring(housing, t0)
        for gr in eb.CONVERTERS:
            pb = pb.reindex(
                columns=pb.columns.union(
                    [c for _, c in eb._group_channels(gr, wiring.get(gr))]
                )
            )
        p_ref = sum(
            (pb[tap] * pb[f"Courant_{gr}"]).fillna(0) for gr, tap in wiring.items()
        )
        lag = eb.estimate_lag(p_ref / 1e6, pup.Pmagnet, 120)
        pup.index = pup.index + pd.Timedelta(seconds=lag)
        df = pb.join(pup, how="inner").asfreq("1s")
        for gr in eb.CONVERTERS:
            off = (
                df[list(eb.PUPITRE_CURRENT[gr])].abs().sum(axis=1) < eb.ACTIVE_CURRENT_A
            )
            cols = [c for _, c in eb._group_channels(gr, wiring.get(gr))]
            df.loc[off, cols] = df.loc[off, cols].fillna(0.0)
        bal = eb.energy_balance(df, wiring, 1.0, 1e-4, 0.023, 30.0, 0, 0, 0, True)
    except Exception:
        return []
    out = []
    for a, b in eb.steady_segments(bal.P_magnet.fillna(bal.Pmagnet_pupitre)):
        if (b - a) < pd.Timedelta("30min"):
            continue
        s = bal.loc[b - pd.Timedelta("10min") : b].median()
        if not s.P_magnet > 5 or s.Ptot_pupitre <= 0:
            continue
        r = df.loc[
            b - pd.Timedelta("10min") : b, ["Rpm1", "Rpm2", "Flow1", "Flow2"]
        ].median()
        out.append(
            {
                "housing": housing,
                "file": name,
                "start": a,
                "dur_min": (b - a).total_seconds() / 60,
                "lag_s": lag,
                "P_mag": s.P_magnet,
                "P_pup": s.Pmagnet_pupitre,
                "Ptot": s.Ptot_pupitre,
                "P_busbar": s.P_busbar,
                "P_pump_el": s.P_pump_el,
                "P_alim": s.P_alim,
                "Tin": s.Tin,
                "TAlim": s.TAlimout,
                "F_alim_implied": s.F_alim_implied,
                "Rpm1": r.Rpm1,
                "Rpm2": r.Rpm2,
                "Flow1": r.Flow1,
                "Flow2": r.Flow2,
            }
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plateaus", type=Path, help="output of plateaus.py")
    ap.add_argument("out", type=Path, help="CSV to write, one row per long plateau")
    ap.add_argument(
        "--records",
        type=Path,
        default=Path.home() / "LNCMIG-Data/records",
        help="records root with srv-data-install/ and pbsurv/ (default: ~/LNCMIG-Data/records)",
    )
    ap.add_argument(
        "--workers", type=int, default=12, help="parallel processes (default: 12)"
    )
    args = ap.parse_args()
    pl = pd.read_csv(args.plateaus)
    cand = pl[(pl.dur_s >= 1800) & (pl.Pmagnet > 5)][
        ["housing", "file"]
    ].drop_duplicates()
    jobs = []
    for housing, g in cand.groupby("housing"):
        ovw = overview_index(args.records, housing)
        jobs += [(args.records, housing, f, ovw) for f in g.file]
    print(f"{len(jobs)} candidate runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for rs in ex.map(run, jobs, chunksize=2) for r in rs]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(rows)} plateaus -> {args.out}")
