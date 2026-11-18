"""Flow-probe fault detector v2 (prototype, survey only)."""
import numpy as np, pandas as pd
from scipy.signal import lfilter

TAUS = (0, 5, 10, 15, 20, 30)
CURRENT = {"1": ("Idcct1", "Idcct2"), "2": ("Idcct3", "Idcct4")}


def lag(u, tau, dt=1.0):
    u = np.asarray(u, float)
    if tau == 0:
        return u
    a = dt / (tau + dt)
    y, _ = lfilter([a], [1, a - 1], u, zi=[u[0] * (1 - a)])
    return y


def mad(r):
    return 1.4826 * np.median(np.abs(r - np.median(r))) if len(r) else np.nan


def robust_lin(X, y, m):
    A = np.c_[np.ones(len(y)), X]
    m = m.copy()
    for _ in range(5):
        c, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        r = y - A @ c
        m = m & (np.abs(r) < max(5 * mad(r[m]), 1.5))
    return c, y - A @ c, mad((y - A @ c)[m])


def runs(mask):
    m = np.r_[False, np.asarray(mask, bool), False].astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def hold_length(x):
    """Length of the constant run each sample belongs to."""
    x = np.asarray(x)
    new = np.r_[True, x[1:] != x[:-1]]
    gid = np.cumsum(new)
    return np.bincount(gid)[gid]


def flow_params_values(fp):
    return {k: p["value"] for k, p in fp.items()} if fp else None


def analyse(df, g, fp=None, k=6.0, thr_floor=5.0, taus=TAUS):
    F, R, P = f"Flow{g}", f"Rpm{g}", f"HP{g}"
    flow = df[F].to_numpy(float)
    rpm = df[R].to_numpy(float)
    head = (df[P] - df["BP"]).to_numpy(float)
    n = len(flow)
    on = rpm > 900
    zero = flow <= 0.5
    out = dict(on=on, head=head)
    out["pump_off"] = on & zero & (head < 0.5)
    out["dropout"] = on & zero & (head > 1.0)
    good = on & ~zero & (head > 0.3)
    out["n_good"] = int(good.sum())
    v = flow_params_values(fp)
    # rpm vs set-point law
    if v is not None and set(CURRENT[g]) <= set(df.columns):
        i = np.abs(df[CURRENT[g][0]].to_numpy(float) + df[CURRENT[g][1]].to_numpy(float))
        vp = v["Vpmax"] * (np.minimum(i, v["Imax"]) / v["Imax"]) ** 2 + v["Vp0"]
        out["rpm_law_res"] = rpm - vp
    if good.sum() < 100 or np.ptp(rpm[good]) < 100:
        out["calibrated"] = False
        return out
    best = None
    for tau in taus:
        x = lag(rpm, tau)
        c, r, s = robust_lin(x[:, None], flow, good)
        q = np.percentile(np.abs(r[good]), 95)
        if best is None or q < best[3]:
            best = (tau, c, s, q)
    tau, c, s, _ = best
    xr = lag(rpm, tau)
    f_rpm = c[0] + c[1] * xr
    okh = good & (head > 0.3)
    ch, _, _ = robust_lin(np.log(head[okh])[:, None], np.log(flow[okh]), np.ones(okh.sum(), bool))
    f_hp0 = np.exp(ch[0]) * np.clip(head, 0.05, None) ** ch[1]
    # flowmeter response: flow vs lagged head, tau_m picked like tau
    bm = None
    for tm in (0, 3, 5, 10, 15):
        rr = (flow - lag(f_hp0, tm))[okh]
        q = np.percentile(np.abs(rr), 95)
        if bm is None or q < bm[1]:
            bm = (tm, q)
    tau_m = bm[0]
    f_hp = lag(f_hp0, tau_m)
    sh = mad((flow - f_hp)[okh])
    thr, thrh = max(k * s, thr_floor), max(k * sh, thr_floor)
    dev = on & ~zero & (np.abs(flow - f_rpm) > thr)
    devh = on & ~zero & (np.abs(flow - f_hp) > thrh)
    both = dev & devh
    hold = hold_length(flow)
    slope = pd.Series(np.abs(np.gradient(f_hp0))).rolling(21, center=True, min_periods=1).max()
    ramp = slope.to_numpy() > 0.3
    out["stale"] = both & (hold >= 3) & ramp            # slow update while witnesses move
    out["stuck"] = both & (hold >= 20) & ~ramp          # held while witnesses are flat
    out["probe"] = both & ~out["stuck"] & ~out["stale"]
    out["hydraulic"] = dev & ~devh
    # zero is a probe dropout only if the head says the pump delivers
    idle = good & (rpm < rpm[good].min() + 20)
    idle_head = np.median(head[idle]) if idle.sum() >= 10 else 1.7
    out["idle_head"] = idle_head
    out["dropout"] = on & zero & (head > 0.6 * idle_head)
    out["pump_off"] = on & zero & ~out["dropout"]
    out.update(calibrated=True, tau=tau, tau_m=tau_m, a=c[0], b=c[1], s=s, n_hp=ch[1], sh=sh,
               thr=thr, f_rpm=f_rpm, f_hp=f_hp)
    if v is not None:
        f_prior = v["F0"] + v["Fmax"] * xr / (v["Vpmax"] + v["Vp0"])
        out["f_prior"] = f_prior
        out["a_prior"] = v["F0"]
        out["b_prior"] = v["Fmax"] / (v["Vpmax"] + v["Vp0"])
        out["bias_prior"] = float(np.median((flow - f_prior)[good]))
    return out


REPAIR = ("dropout", "stuck", "probe")


def events(res, min_len=3, gap=2):
    """List flagged events per class (merged across `gap` s, kept if >= min_len)."""
    ev = []
    for kind in ("dropout", "stuck", "probe", "stale", "hydraulic", "pump_off"):
        if kind not in res:
            continue
        m = res[kind].copy()
        for a, b in runs(~m):  # close short holes
            if 0 < a and b < len(m) and b - a <= gap:
                m[a:b] = True
        for a, b in runs(m):
            if b - a >= (1 if kind == "dropout" else min_len):
                ev.append((kind, int(a), int(b)))
    return ev


def reconstruct(flow, bad, model, edge=10):
    """Model plus residual linearly interpolated between gap edges."""
    y = np.asarray(flow, float).copy()
    for a, b in runs(bad):
        L = (y[max(a - edge, 0):a] - model[max(a - edge, 0):a])
        R = (y[b:b + edge] - model[b:b + edge])
        l = np.median(L) if len(L) else (np.median(R) if len(R) else 0.0)
        r = np.median(R) if len(R) else l
        w = (np.arange(a, b) - a + 0.5) / (b - a)
        y[a:b] = model[a:b] + (1 - w) * l + w * r
    return y
