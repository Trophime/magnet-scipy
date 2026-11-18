"""Flow-probe faults of the pupitre hydraulics: detection and repair.

The measured flow of a hydraulic group is checked against two independent
witnesses:

- the pump set point ``Rpm``, through a first-order lag (the pupitre records
  the set point, which leads the flow by 5–30 s on ramps):
  ``F_rpm = a + b·L_τ(Rpm)``, the linear form of
  :class:`python_magnetcooling.waterflow.WaterFlow`;
- the pump head ``HP − BP``: ``F_head = c·(HP − BP)^n`` (n ≈ 0.5), lagged by
  the flowmeter response ``τ_m``.

A sample disagreeing with both witnesses is a probe fault; a sample whose
flow and head agree but disagree with the set point is a real hydraulic
event and is left alone. A zero flow is a probe dropout only while the head
says the pump delivers water; otherwise the pump is stopped. Probe faults
are filled with the head witness shifted by the measured offset at the gap
edges.

The functions work on one group at a time (``Flow{g}``, ``Rpm{g}``,
``HP{g}``, shared ``BP``); which magnet a group feeds depends on the housing.
Background and validation: ``prompts/flow_probe_faults.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FLAGS = ("ok", "off", "pump_off", "dropout", "stuck", "probe", "stale", "hydraulic")
REPAIRED = ("dropout", "stuck", "probe")  # flags whose flow is replaced

RPM_ON = 900.0  # [rpm] pump commanded above this set point (idle ~1000 rpm)
ZERO_FLOW = 0.5  # [l/s] flow at or below this reads as zero
MIN_HEAD = 0.3  # [bar] head with water flowing
IDLE_HEAD_FRACTION = 0.6  # dropout: head above this fraction of the idle head
IDLE_HEAD_FALLBACK = 1.7  # [bar] idle head when the file has no idle samples
MIN_SAMPLES = 100  # running samples needed to calibrate
MIN_RPM_RANGE = 100.0  # [rpm] set-point range needed to fit the slope
FROZEN_RPM_FLOW_RANGE = 10.0  # [l/s] head-flow range proving a frozen set point
RPM_TAUS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)  # [s] candidate set-point lags
METER_TAUS = (0.0, 3.0, 5.0, 10.0, 15.0)  # [s] candidate flowmeter lags
LAG_SCORE_CAP = 15.0  # [l/s] residual cap when scoring a lag
K_SIGMA = 6.0  # threshold in robust standard deviations
THRESHOLD_FLOOR = 5.0  # [l/s] minimum threshold
STALE_HOLD = 3.0  # [s] held value on a ramp: slow meter update
STUCK_HOLD = 20.0  # [s] held value while the witnesses are flat
STALE_MAX_HOLD = 30.0  # [s] longer holds are stuck even on a ramp
CEILING_FRACTION = (
    0.9  # held values above this fraction of the maximum: meter full scale
)
RAMP_SLOPE = 0.1  # [l/s per s] witness slope that counts as a ramp
RAMP_HALF_WINDOW = 10.0  # [s]
EDGE_WINDOW = 10.0  # [s] window for the offset at each gap edge


@dataclass
class FlowWitnesses:
    """Calibration of the two flow witnesses of one group in one file.

    Attributes
    ----------
    tau : float
        Set-point to flow lag [s].
    a : float
        Offset of the set-point witness [l/s].
    b : float
        Slope of the set-point witness [l/s per rpm].
    sigma_rpm : float
        Robust standard deviation of flow minus set-point witness [l/s].
    rpm_usable : bool
        False when the set point is frozen while the hydraulics move.
    tau_m : float
        Flowmeter lag behind the head [s].
    c : float
        Head-witness coefficient [l/s per bar^n].
    n : float
        Head-witness exponent.
    sigma_head : float
        Robust standard deviation of flow minus head witness [l/s].
    idle_head : float
        Head at the idle set point with water flowing [bar].
    """

    tau: float
    a: float
    b: float
    sigma_rpm: float
    rpm_usable: bool
    tau_m: float
    c: float
    n: float
    sigma_head: float
    idle_head: float

    def flow_from_rpm(self, time: np.ndarray, rpm: np.ndarray) -> np.ndarray:
        """Flow predicted from the pump set point.

        Parameters
        ----------
        time : :class:`~numpy.ndarray`
            Time [s], increasing.
        rpm : :class:`~numpy.ndarray`
            Pump set point [rpm].

        Returns
        -------
        :class:`~numpy.ndarray`
            Flow [l/s]; NaN everywhere when the set point is not usable.
        """
        if not self.rpm_usable:
            return np.full(len(rpm), np.nan)
        return self.a + self.b * lag_first_order(time, rpm, self.tau)

    def flow_from_head(
        self, time: np.ndarray, head: np.ndarray, lagged: bool = True
    ) -> np.ndarray:
        """Flow predicted from the pump head.

        Parameters
        ----------
        time : :class:`~numpy.ndarray`
            Time [s], increasing.
        head : :class:`~numpy.ndarray`
            Pump head ``HP − BP`` [bar].
        lagged : bool, optional
            Apply the flowmeter lag ``tau_m`` (default), to compare with the
            measured flow.

        Returns
        -------
        :class:`~numpy.ndarray`
            Flow [l/s].
        """
        q = self.c * np.clip(head, 0.05, None) ** self.n
        return lag_first_order(time, q, self.tau_m) if lagged else q


def lag_first_order(time: np.ndarray, u: np.ndarray, tau: float) -> np.ndarray:
    """First-order lag of a sampled signal, exact for piecewise-constant input.

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Sample times [s], increasing (uneven steps allowed).
    u : :class:`~numpy.ndarray`
        Input signal.
    tau : float
        Time constant [s]; 0 returns the input.

    Returns
    -------
    :class:`~numpy.ndarray`
        Lagged signal, starting at ``u[0]``.
    """
    u = np.asarray(u, float)
    if tau <= 0 or len(u) == 0:
        return u.copy()
    alpha = 1.0 - np.exp(-np.diff(np.asarray(time, float)) / tau)
    y = np.empty_like(u)
    y[0] = u[0]
    for k in range(1, len(u)):
        y[k] = y[k - 1] + alpha[k - 1] * (u[k] - y[k - 1])
    return y


def _mad(r: np.ndarray) -> float:
    """Robust standard deviation (1.4826 × median absolute deviation)."""
    return float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(r) else np.nan


def _robust_line(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, slope: float | None = None
):
    """Trimmed least-squares ``y = a + b x`` on ``mask`` (``b`` fixed when ``slope`` is given)."""
    keep = mask.copy()
    for _ in range(5):
        if slope is None:
            b, a = np.polyfit(x[keep], y[keep], 1)
        else:
            b, a = slope, float(np.median(y[keep] - slope * x[keep]))
        r = y - a - b * x
        keep = keep & (np.abs(r) < max(5 * _mad(r[keep]), 1.5))
    return a, b, _mad(r[keep]), r


def _lag_score(r: np.ndarray) -> float:
    """Mean of ``|r|`` clipped at ``LAG_SCORE_CAP``: probe faults weigh the same for every lag."""
    return float(np.mean(np.minimum(np.abs(r), LAG_SCORE_CAP)))


def _prior_slope(flow_params: dict) -> float:
    """Slope ``Fmax / (Vpmax + Vp0)`` of a flow params file [l/s per rpm]."""
    v = {k: p["value"] for k, p in flow_params.items()}
    return v["Fmax"] / (v["Vpmax"] + v["Vp0"])


def _samples(time: np.ndarray, seconds: float) -> int:
    """Number of samples spanning ``seconds`` at the median time step."""
    dt = np.median(np.diff(time)) if len(time) > 1 else 1.0
    return max(int(round(seconds / dt)), 1)


def calibrate_flow_witnesses(
    time: np.ndarray,
    flow: np.ndarray,
    rpm: np.ndarray,
    head: np.ndarray,
    *,
    flow_params: dict | None = None,
    tau: float | None = None,
) -> FlowWitnesses | None:
    """Fit the set-point and head witnesses of one group on one file.

    Fits are robust (trimmed least squares) on running samples: set point
    above ``RPM_ON``, flow above ``ZERO_FLOW``, head above ``MIN_HEAD``, and
    not at the meter ceiling (a value held ``STUCK_HOLD`` or longer above
    ``CEILING_FRACTION`` of the file's maximum flow). Lags
    are picked from ``RPM_TAUS`` / ``METER_TAUS`` by the smallest mean
    residual clipped at ``LAG_SCORE_CAP``, so that ramps decide and probe
    faults don't.

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Time [s], increasing.
    flow : :class:`~numpy.ndarray`
        Measured flow [l/s].
    rpm : :class:`~numpy.ndarray`
        Pump set point [rpm].
    head : :class:`~numpy.ndarray`
        Pump head ``HP − BP`` [bar].
    flow_params : dict, optional
        Water flow params of the magnet (``python_magnetcooling`` format).
        Its slope ``Fmax / (Vpmax + Vp0)`` is used when the set point moves
        less than ``MIN_RPM_RANGE`` in the file (otherwise the slope is
        fitted: it differs from run to run by up to ~10 %); the offset is
        always fitted.
    tau : float, optional
        Set-point lag [s] to use instead of picking one (e.g. a per-magnet
        value).

    Returns
    -------
    FlowWitnesses or None
        None when the file has fewer than ``MIN_SAMPLES`` running samples.
    """
    time, flow, rpm, head = (np.asarray(x, float) for x in (time, flow, rpm, head))
    finite = np.isfinite(flow) & np.isfinite(rpm) & np.isfinite(head)
    good = finite & (rpm > RPM_ON) & (flow > ZERO_FLOW) & (head > MIN_HEAD)
    good &= ~_at_ceiling(time, flow)
    if good.sum() < MIN_SAMPLES:
        return None

    idle = good & (rpm < rpm[good].min() + 20.0)
    idle_head = float(np.median(head[idle])) if idle.sum() >= 10 else IDLE_HEAD_FALLBACK

    # head witness: log Q = log c + n log head
    log_h = np.log(np.clip(head, MIN_HEAD, None))
    log_q = np.log(np.clip(flow, ZERO_FLOW, None))
    if np.ptp(log_h[good]) > 0.3:
        log_c, n, _, _ = _robust_line(log_h, log_q, good)
    else:
        log_c, n, _, _ = _robust_line(log_h, log_q, good, slope=0.5)
    q_head = np.exp(log_c) * np.clip(head, 0.05, None) ** n
    best = None
    for tm in METER_TAUS:
        r = (flow - lag_first_order(time, q_head, tm))[good]
        score = _lag_score(r)
        if best is None or score < best[0]:
            best = (score, tm, _mad(r))
    _, tau_m, sigma_head = best

    # set-point witness: slope fitted when the set point moves enough
    rpm_moves = np.ptp(rpm[good]) >= MIN_RPM_RANGE
    head_flow_range = np.subtract(*np.percentile(q_head[good], [95, 5]))
    rpm_usable = rpm_moves or head_flow_range < FROZEN_RPM_FLOW_RANGE
    if not rpm_usable:
        a = b = sigma_rpm = np.nan
        tau_rpm = np.nan if tau is None else tau
    else:
        slope = None
        if not rpm_moves:
            slope = _prior_slope(flow_params) if flow_params else 0.0
        best = None
        for t_lag in (
            ((tau,) if tau is not None else RPM_TAUS) if rpm_moves else (tau or 0.0,)
        ):
            x = lag_first_order(time, rpm, t_lag)
            a, b, sigma, r = _robust_line(x, flow, good, slope)
            score = _lag_score(r[good])
            if best is None or score < best[0]:
                best = (score, t_lag, a, b, sigma)
        _, tau_rpm, a, b, sigma_rpm = best

    return FlowWitnesses(
        tau=float(tau_rpm),
        a=float(a),
        b=float(b),
        sigma_rpm=float(sigma_rpm),
        rpm_usable=bool(rpm_usable),
        tau_m=float(tau_m),
        c=float(np.exp(log_c)),
        n=float(n),
        sigma_head=float(sigma_head),
        idle_head=idle_head,
    )


def _hold_groups(x: np.ndarray) -> np.ndarray:
    """Id of the run of identical consecutive values each sample belongs to."""
    return np.cumsum(np.r_[True, x[1:] != x[:-1]])


def _hold_seconds(time: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Duration [s] of the run of identical values each sample belongs to."""
    gid = _hold_groups(x)
    dt = np.median(np.diff(time)) if len(time) > 1 else 1.0
    return np.bincount(gid)[gid] * dt


def _at_ceiling(time: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Samples at a value held ``STUCK_HOLD`` within ``CEILING_FRACTION`` of the maximum (meter full scale)."""
    finite = np.isfinite(flow)
    if not finite.any():
        return np.zeros(len(flow), bool)
    held = finite & (_hold_seconds(time, flow) >= STUCK_HOLD)
    held &= flow >= CEILING_FRACTION * np.nanmax(flow)
    return finite & np.isin(flow, np.unique(flow[held]))


def classify_flow_samples(
    time: np.ndarray,
    flow: np.ndarray,
    rpm: np.ndarray,
    head: np.ndarray,
    witnesses: FlowWitnesses | None,
) -> np.ndarray:
    """Flag every flow sample of one group.

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Time [s], increasing.
    flow : :class:`~numpy.ndarray`
        Measured flow [l/s].
    rpm : :class:`~numpy.ndarray`
        Pump set point [rpm].
    head : :class:`~numpy.ndarray`
        Pump head ``HP − BP`` [bar].
    witnesses : FlowWitnesses or None
        Calibration from :func:`calibrate_flow_witnesses`; None leaves only
        the zero-flow tests.

    Returns
    -------
    :class:`~numpy.ndarray` of str
        One of :data:`FLAGS` per sample:

        - ``off``: set point at or below ``RPM_ON`` or missing value;
        - ``pump_off``: zero flow, head below the dropout limit (real stop);
        - ``dropout``: zero flow while the head is above
          ``IDLE_HEAD_FRACTION`` × idle head;
        - ``stuck``: off both witnesses, value held ≥ ``STUCK_HOLD`` while
          the witnesses are flat, or ≥ ``STALE_MAX_HOLD`` on a ramp; the
          whole run of that held value is flagged;
        - ``stale``: off either witness, value held between ``STALE_HOLD``
          and ``STALE_MAX_HOLD`` on a ramp (slow meter update);
        - ``probe``: off both witnesses otherwise (spikes, offsets);
        - ``hydraulic``: off the set point but on the head, value moving
          (real event);
        - ``ok``.

        A witness is "off" beyond ``max(K_SIGMA × sigma, THRESHOLD_FLOOR)``.
        With a frozen set point, the head is the only witness.
    """
    time, flow, rpm, head = (np.asarray(x, float) for x in (time, flow, rpm, head))
    flags = np.full(len(flow), "ok", dtype=object)
    on = np.isfinite(flow) & np.isfinite(rpm) & np.isfinite(head) & (rpm > RPM_ON)
    flags[~on] = "off"

    zero = on & (flow <= ZERO_FLOW)
    idle_head = witnesses.idle_head if witnesses else IDLE_HEAD_FALLBACK
    dropout = zero & (head > IDLE_HEAD_FRACTION * idle_head)
    flags[zero & ~dropout] = "pump_off"
    flags[dropout] = "dropout"
    if witnesses is None:
        return flags

    w = witnesses
    running = on & ~zero
    q_head0 = w.flow_from_head(time, head, lagged=False)
    dev_head = running & (
        np.abs(flow - lag_first_order(time, q_head0, w.tau_m))
        > max(K_SIGMA * w.sigma_head, THRESHOLD_FLOOR)
    )
    if w.rpm_usable:
        dev_rpm = running & (
            np.abs(flow - w.flow_from_rpm(time, rpm))
            > max(K_SIGMA * w.sigma_rpm, THRESHOLD_FLOOR)
        )
    else:
        dev_rpm = dev_head
    both = dev_rpm & dev_head

    smooth = (
        pd.Series(q_head0)
        .rolling(_samples(time, 11.0), center=True, min_periods=1)
        .median()
    )
    half = _samples(time, RAMP_HALF_WINDOW)
    slope = (
        smooth.shift(-half).bfill().ffill() - smooth.shift(half).bfill().ffill()
    ).abs()
    ramp = (slope / (2 * RAMP_HALF_WINDOW)).to_numpy() > RAMP_SLOPE
    hold = _hold_seconds(time, flow)

    stale = (dev_rpm | dev_head) & ramp & (hold >= STALE_HOLD) & (hold < STALE_MAX_HOLD)
    stuck = both & (hold >= STUCK_HOLD) & (~ramp | (hold >= STALE_MAX_HOLD))
    gid = _hold_groups(flow)
    stuck = running & np.isin(gid, gid[stuck])  # a held value is one reading: all of it
    flags[both & ~stale & ~stuck] = "probe"
    flags[dev_rpm & ~dev_head & ~stale] = "hydraulic"
    flags[stale] = "stale"
    flags[stuck] = "stuck"
    return flags


def reconstruct_flow(
    time: np.ndarray,
    flow: np.ndarray,
    bad: np.ndarray,
    model: np.ndarray,
    edge: float = EDGE_WINDOW,
) -> np.ndarray:
    """Fill bad samples with a model shifted by the offset at the gap edges.

    For each run of bad samples, the offset ``flow − model`` is the median
    over ``edge`` seconds of good samples before and after the run, and is
    interpolated linearly in time across it (one-sided at the file edges).

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Time [s], increasing.
    flow : :class:`~numpy.ndarray`
        Measured flow [l/s].
    bad : :class:`~numpy.ndarray` of bool
        Samples to replace.
    model : :class:`~numpy.ndarray`
        Witness flow [l/s] (set point or head).
    edge : float, optional
        Width of the offset window at each edge [s].

    Returns
    -------
    :class:`~numpy.ndarray`
        Flow [l/s], equal to ``flow`` outside ``bad``.
    """
    time, flow, model = (np.asarray(x, float) for x in (time, flow, model))
    bad = np.asarray(bad, bool)
    out = flow.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        before = ~bad & (time < time[start]) & (time >= time[start] - edge)
        after = ~bad & (time > time[stop - 1]) & (time <= time[stop - 1] + edge)
        left = np.median((flow - model)[before]) if before.any() else None
        right = np.median((flow - model)[after]) if after.any() else None
        if left is None and right is None:
            left = right = 0.0
        left = right if left is None else left
        right = left if right is None else right
        t0 = time[start - 1] if start > 0 else time[start]
        t1 = time[stop] if stop < len(time) else time[stop - 1]
        w = (time[start:stop] - t0) / (t1 - t0) if t1 > t0 else np.zeros(stop - start)
        out[start:stop] = model[start:stop] + (1 - w) * left + w * right
    return out


def check_flow_probe(
    time: np.ndarray,
    flow: np.ndarray,
    rpm: np.ndarray,
    hp: np.ndarray,
    bp: np.ndarray,
    *,
    flow_params: dict | None = None,
    tau: float | None = None,
) -> tuple[pd.DataFrame, FlowWitnesses | None]:
    """Detect flow-probe faults of one group and fill them.

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Time [s], increasing.
    flow : :class:`~numpy.ndarray`
        Measured flow ``Flow{g}`` [l/s].
    rpm : :class:`~numpy.ndarray`
        Pump set point ``Rpm{g}`` [rpm].
    hp : :class:`~numpy.ndarray`
        Pump outlet pressure ``HP{g}`` [bar].
    bp : :class:`~numpy.ndarray`
        Back pressure ``BP`` [bar].
    flow_params : dict, optional
        Water flow params of the magnet; set-point slope for files where the
        set point barely moves (see :func:`calibrate_flow_witnesses`).
    tau : float, optional
        Set-point lag [s] (per-magnet value) instead of picking one per file.

    Returns
    -------
    frame : :class:`~pandas.DataFrame`
        ``time`` [s], ``flow`` [l/s] as measured, ``flow_rpm`` and
        ``flow_head`` (witnesses, lagged as compared) [l/s], ``flag`` (one
        of :data:`FLAGS`) and ``flow_filled`` [l/s]: the measured flow with
        the :data:`REPAIRED` samples replaced by the head witness shifted to
        the gap edges (the head follows the pump state with only the meter
        lag, so it matches the flow better than the set point on the ramps
        where gaps often start). Without a calibration the flow is left as
        measured.
    witnesses : FlowWitnesses or None
        The calibration used, None when the group does not run long enough.
    """
    time, flow, rpm = (np.asarray(x, float) for x in (time, flow, rpm))
    head = np.asarray(hp, float) - np.asarray(bp, float)
    w = calibrate_flow_witnesses(
        time, flow, rpm, head, flow_params=flow_params, tau=tau
    )
    flags = classify_flow_samples(time, flow, rpm, head, w)
    if w is None:
        q_rpm = q_head = np.full(len(flow), np.nan)
        filled = flow.copy()
    else:
        q_rpm = w.flow_from_rpm(time, rpm)
        q_head = w.flow_from_head(time, head)
        filled = reconstruct_flow(time, flow, np.isin(flags, REPAIRED), q_head)
    frame = pd.DataFrame(
        {
            "time": time,
            "flow": flow,
            "flow_rpm": q_rpm,
            "flow_head": q_head,
            "flag": flags,
            "flow_filled": filled,
        }
    )
    return frame, w
