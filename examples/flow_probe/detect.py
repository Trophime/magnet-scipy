"""Prototype flow-probe fault detector + reconstruction (survey, not package code)."""
import os
import numpy as np, pandas as pd
from scipy.signal import lfilter

TAUS = (0, 5, 10, 15, 20, 30)


def lag(u, tau, dt=1.0):
    if tau == 0:
        return u.astype(float)
    a = dt / (tau + dt)
    y, _ = lfilter([a], [1, a - 1], u.astype(float), zi=[u[0] * (1 - a)])
    return y


def mad(r):
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def robust_lin(X, y, m=None):
    A = np.c_[np.ones(len(y)), X]
    m = np.ones(len(y), bool) if m is None else m.copy()
    for _ in range(5):
        c, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        r = y - A @ c
        s = mad(r[m])
        m = m & (np.abs(r) < max(5 * s, 1.5)) if _ else (np.abs(r) < max(5 * mad(r), 1.5)) & m
    return c, r, s


def runs(mask):
    """Return list of (start, stop) index pairs of True runs."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def analyse(df, g, thr_floor=5.0, k=6.0, min_len=3):
    F, R, P = f"Flow{g}", f"Rpm{g}", f"HP{g}"
    flow = df[F].values.astype(float)
    rpm = df[R].values.astype(float)
    dp = (df[P] - df["BP"]).values.astype(float)
    on = rpm > 900
    zero = on & (flow <= 0.5)
    good = on & ~zero
    out = dict(on=on, zero=zero)
    if good.sum() < 100:
        return out
    # flow ~ lagged rpm, tau picked on this file
    best = None
    for tau in TAUS:
        x = lag(rpm, tau)
        c, r, s = robust_lin(x[:, None], flow, good)
        q = np.percentile(np.abs(r[good]), 95)
        if best is None or q < best[3]:
            best = (tau, c, s, q)
    tau, c, s, _ = best
    f_rpm = c[0] + c[1] * lag(rpm, tau)
    # flow ~ head (log-log)
    okh = good & (dp > 0.2)
    ch, _, _ = robust_lin(np.log(np.clip(dp[okh], 0.2, None))[:, None], np.log(flow[okh]))
    f_hp = np.exp(ch[0]) * np.clip(dp, 0.05, None) ** ch[1]
    rh = flow - f_hp
    sh = mad(rh[okh])
    thr = max(k * s, thr_floor)
    thrh = max(k * sh, thr_floor)
    dev = on & (np.abs(flow - f_rpm) > thr)
    devh = on & (np.abs(rh) > thrh)
    sensor = zero | (dev & devh)          # flow disagrees with both witnesses -> flow probe
    hydraul = dev & ~devh                 # flow and head agree, rpm set point not -> real
    hpfault = devh & ~dev                 # flow agrees rpm, head does not -> HP probe
    out.update(tau=tau, c=c, s=s, ch=ch, sh=sh, f_rpm=f_rpm, f_hp=f_hp,
               sensor=sensor, hydraul=hydraul, hpfault=hpfault, dev=dev, thr=thr)
    return out


def reconstruct(df, g, bad, res, method="rpm_bias"):
    """Fill flow on `bad` samples."""
    F = f"Flow{g}"
    y = df[F].values.astype(float).copy()
    t = np.arange(len(y))
    if method == "interp":
        y[bad] = np.interp(t[bad], t[~bad], y[~bad])
        return y
    if method == "rpm":
        y[bad] = res["f_rpm"][bad]
        return y
    if method == "rpm_hp":
        R, P = f"Rpm{g}", f"HP{g}"
        x1 = lag(df[R].values, res["tau"])
        x2 = np.sqrt(np.clip((df[P] - df["BP"]).values, 0.05, None))
        ok = res["on"] & ~bad & ~res["zero"]
        c, _, _ = robust_lin(np.c_[x1, x2], y, ok)
        y[bad] = (np.c_[np.ones(len(y)), x1, x2] @ c)[bad]
        return y
    if method == "rpm_bias":
        # model + offset linearly interpolated between gap edges
        f = res["f_rpm"]
        for a, b in runs(bad):
            lo = max(a - 10, 0); hi = min(b + 10, len(y))
            left = (y[lo:a] - f[lo:a]); right = (y[b:hi] - f[b:hi])
            l = np.median(left) if len(left) else (np.median(right) if len(right) else 0)
            r = np.median(right) if len(right) else l
            w = (np.arange(a, b) - a + 0.5) / (b - a)
            y[a:b] = f[a:b] + (1 - w) * l + w * r
        return y
    raise ValueError(method)
