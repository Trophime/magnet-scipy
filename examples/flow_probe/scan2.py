"""Compare flow predictors per file: static rpm, lagged rpm, head HP-BP (survey)."""
import sys, os, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import lfilter

D = os.path.expanduser("~/github/python_magnetrun/srvdata")
TAUS = [0, 5, 10, 15, 20, 30, 40, 60]


def lag(u, tau, dt=1.0):
    if tau == 0:
        return u
    a = dt / (tau + dt)
    y, _ = lfilter([a], [1, a - 1], u, zi=[u[0] * (1 - a)])
    return y


def mad(r):
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def robust_lin(x, y):
    A = np.c_[np.ones_like(x), x]
    m = np.ones_like(y, bool)
    for _ in range(4):
        c, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        r = y - A @ c
        m = np.abs(r) < max(5 * mad(r), 1.0)
    return c, r


def scan(f):
    try:
        df = pd.read_csv(f, sep=r"\s+", skiprows=1)
    except Exception:
        return []
    out = []
    for g in ("1", "2"):
        F, R, P = f"Flow{g}", f"Rpm{g}", f"HP{g}"
        if not {F, R, P, "BP"} <= set(df.columns):
            continue
        d = df[[F, R, P, "BP"]].apply(pd.to_numeric, errors="coerce").ffill().fillna(0)
        on = (d[R] > 900).values
        ok = on & (d[F] > 1).values
        if ok.sum() < 200 or np.ptp(d[R][on]) < 300 or ok.sum() < 0.95 * on.sum():
            continue
        flow = d[F].values
        res = dict(file=os.path.basename(f), grp=g, n=int(ok.sum()))
        for tau in TAUS:
            x = lag(d[R].values.astype(float), tau)
            c, r = robust_lin(x[ok], flow[ok])
            res[f"rpm{tau}"] = mad(r)
            res[f"rpm{tau}_p99"] = np.percentile(np.abs(r), 99)
        dp = (d[P] - d["BP"]).values
        okp = ok & (dp > 0.2)
        c, r = robust_lin(np.log(dp[okp]), np.log(flow[okp]))
        res["hp_n"] = c[1]
        rr = flow[okp] - np.exp(c[0]) * dp[okp] ** c[1]
        res["hp"] = mad(rr)
        res["hp_p99"] = np.percentile(np.abs(rr), 99)
        res["hp_max"] = np.abs(rr).max()
        out.append(res)
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(D, "*.txt")))
    rng = np.random.default_rng(0)
    files = list(rng.choice(files, 2500, replace=False))
    with Pool(16) as p:
        res = p.map(scan, files, chunksize=20)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(sys.argv[1], index=False)
