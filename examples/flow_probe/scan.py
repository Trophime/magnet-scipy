"""Scan pupitre files for flow-probe faults (survey, not package code)."""
import sys, os, glob
import numpy as np, pandas as pd
from multiprocessing import Pool

D = os.path.expanduser("~/github/python_magnetrun/srvdata")


def load(f):
    df = pd.read_csv(f, sep=r"\s+", skiprows=1, engine="c", on_bad_lines="skip")
    return df


def scan(f):
    try:
        return _scan(f)
    except Exception as e:
        return [dict(file=os.path.basename(f), err=repr(e)[:80])]


def _scan(f):
    try:
        df = load(f)
    except Exception as e:
        return [dict(file=os.path.basename(f), err=str(e)[:60])]
    out = []
    for g in ("1", "2"):
        F, R, P = f"Flow{g}", f"Rpm{g}", f"HP{g}"
        if not {F, R, P, "BP"} <= set(df.columns):
            continue
        d = df[[F, R, P, "BP"]].apply(pd.to_numeric, errors="coerce").dropna()
        on = d[R] > 900
        n = int(on.sum())
        if n < 30:
            continue
        d = d[on]
        zero = d[F] <= 0.5
        # zero runs
        grp = (zero != zero.shift()).cumsum()
        runs = zero.groupby(grp).agg(["first", "size"])
        zr = runs[runs["first"]]["size"].tolist()
        # robust linear flow~rpm on non-zero samples
        ok = ~zero
        x, y = d.loc[ok, R].values, d.loc[ok, F].values
        if len(x) < 30:
            out.append(dict(file=os.path.basename(f), grp=g, n=n, nzero=int(zero.sum()), zruns=len(zr), zmax=max(zr) if zr else 0))
            continue
        if np.ptp(x) > 50:
            b, a = np.polyfit(x, y, 1)
            for _ in range(3):
                r = y - (a + b * x)
                s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
                m = np.abs(r) < 5 * s
                b, a = np.polyfit(x[m], y[m], 1)
            r = y - (a + b * x)
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        else:
            a, b = np.median(y), 0.0
            r = y - a
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        nout = int((np.abs(r) > max(6 * s, 3.0)).sum())
        # frozen: long constant run while rpm varies
        fz = (d[F].diff() == 0)
        g2 = (fz != fz.shift()).cumsum()
        maxfrozen = int(fz.groupby(g2).sum().max())
        out.append(dict(file=os.path.basename(f), grp=g, n=n, nzero=int(zero.sum()),
                        zruns=len(zr), zmax=max(zr) if zr else 0,
                        a=a, b=b, mad=s, nout=nout, maxfrozen=maxfrozen,
                        rpm_ptp=float(np.ptp(d[R]))))
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(D, "*.txt")))
    with Pool(16) as p:
        res = p.map(scan, files, chunksize=50)
    rows = [r for rr in res for r in rr]
    pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
