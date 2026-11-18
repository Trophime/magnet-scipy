"""Tests for magnet_scipy.flow_probe."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from magnet_scipy.flow_probe import (
    FLAGS,
    REPAIRED,
    calibrate_flow_witnesses,
    check_flow_probe,
    lag_first_order,
    reconstruct_flow,
)

M10_REFERENCE = (
    Path.home() / "github/python_magnetrun/srvdata/M10_2019.05.30---17:22:17.txt"
)

# synthetic housing: pump law and hydraulics close to M9
B_FLOW = 0.048  # [l/s per rpm]
A_FLOW = 1.0  # [l/s]
C_HEAD = 31.6  # Q = C_HEAD * head**0.5, head [bar]
TAU_PUMP = 10.0  # set point -> pump speed [s]
TAU_METER = 3.0  # flowmeter response [s]
BP = 4.0  # [bar]


def _lag(u, tau):
    return lag_first_order(np.arange(len(u), dtype=float), u, tau)


def _run(seed=0, rpm=None):
    """Idle, ramp up, plateau, ramp down, idle at 1 Hz (or the given set point); dict of arrays."""
    if rpm is None:
        rpm = np.r_[
            np.full(100, 1000.0),
            np.linspace(1000, 2500, 200),
            np.full(600, 2500.0),
            np.linspace(2500, 1000, 200),
            np.full(200, 1000.0),
        ]
    t = np.arange(len(rpm), dtype=float)
    q_true = A_FLOW + B_FLOW * _lag(rpm, TAU_PUMP)
    rng = np.random.default_rng(seed)
    flow = _lag(q_true, TAU_METER) + rng.normal(0, 0.3, len(t))
    head = (q_true / C_HEAD) ** 2 + rng.normal(0, 0.03, len(t))
    return {
        "time": t,
        "rpm": rpm,
        "flow": flow,
        "q_true": q_true,
        "hp": head + BP,
        "bp": np.full(len(t), BP),
    }


def _check(r, **kw):
    return check_flow_probe(r["time"], r["flow"], r["rpm"], r["hp"], r["bp"], **kw)


def _flags_in(frame, sl):
    return set(frame["flag"].to_numpy()[sl])


PLATEAU = slice(450, 540)  # inside the plateau, after the pump lag


def test_lag_first_order():
    t = np.arange(100, dtype=float)
    step = np.r_[np.zeros(10), np.ones(90)]
    assert np.array_equal(lag_first_order(t, step, 0.0), step)
    y = lag_first_order(t, step, 5.0)
    assert np.all(np.diff(y) >= 0)
    assert y[9] == 0.0 and y[-1] > 0.99
    # uneven sampling: same response at the common times
    keep = np.r_[np.arange(0, 50), np.arange(50, 100, 2)]
    y2 = lag_first_order(t[keep], step[keep], 5.0)
    assert np.allclose(y2[:50], y[:50])
    assert abs(y2[-1] - y[-1]) < 1e-3


def test_clean_run_has_no_flag():
    frame, w = _check(_run())
    assert set(frame["flag"]) == {"ok"}
    assert w is not None and w.rpm_usable
    assert abs(w.b - B_FLOW) < 0.002
    assert 0.4 < w.n < 0.6
    assert w.tau in (5.0, 10.0, 15.0)
    assert np.array_equal(frame["flow_filled"], frame["flow"])


def test_dropout_with_head_kept_is_found_and_filled():
    r = _run()
    r["flow"][PLATEAU] = 0.0
    frame, _ = _check(r)
    flags = frame["flag"].to_numpy()
    assert np.all(flags[PLATEAU] == "dropout")
    assert set(np.delete(flags, np.arange(450, 540))) == {"ok"}
    err = frame["flow_filled"].to_numpy()[PLATEAU] - r["q_true"][PLATEAU]
    assert np.sqrt(np.mean(err**2)) < 1.0


def test_pump_stop_is_real_not_repaired():
    r = _run()
    # pump stops, set point unchanged, static head of 1.1 bar left
    r["flow"][PLATEAU] = 0.0
    r["hp"][PLATEAU] = BP + 1.1
    frame, _ = _check(r)
    assert _flags_in(frame, PLATEAU) == {"pump_off"}
    assert np.array_equal(frame["flow_filled"].to_numpy()[PLATEAU], r["flow"][PLATEAU])


def test_stuck_at_full_scale():
    r = _run()
    r["flow"][PLATEAU] = 150.0
    frame, _ = _check(r)
    assert _flags_in(frame, PLATEAU) == {"stuck"}
    err = frame["flow_filled"].to_numpy()[PLATEAU] - r["q_true"][PLATEAU]
    assert np.sqrt(np.mean(err**2)) < 1.0


def test_meter_ceiling_does_not_spoil_calibration():
    r = _run()
    ceiling = r["flow"] > 110.0  # meter saturates on the plateau and part of the ramps
    r["flow"][ceiling] = 110.0
    frame, w = _check(r)
    assert abs(w.b - B_FLOW) < 0.002
    flags = frame["flag"].to_numpy()
    assert set(flags[PLATEAU]) == {"stuck"}
    assert set(flags[~ceiling]) == {"ok"}
    err = frame["flow_filled"].to_numpy()[PLATEAU] - r["q_true"][PLATEAU]
    assert np.sqrt(np.mean(err**2)) < 1.0


def test_spikes_are_probe_faults():
    r = _run()
    spikes = np.arange(460, 530, 7)
    r["flow"][spikes] += 30.0
    frame, _ = _check(r)
    flags = frame["flag"].to_numpy()
    assert np.all(flags[spikes] == "probe")
    assert set(np.delete(flags, spikes)) == {"ok"}


def test_stale_updates_on_ramp_are_flagged_not_repaired():
    r = _run()
    ramp = slice(920, 1080)  # ramp-down only; the ramp-up stays clean
    held = r["flow"][ramp].copy()
    for start in range(0, len(held), 20):  # meter updates every 20 s
        held[start : start + 20] = held[start]
    r["flow"][ramp] = held
    frame, _ = _check(r, tau=TAU_PUMP)
    flags = frame["flag"].to_numpy()
    assert "stale" in flags[ramp]
    assert set(flags[ramp]) <= {"ok", "stale"}
    assert np.array_equal(frame["flow_filled"].to_numpy(), r["flow"])


def test_real_flow_change_is_hydraulic():
    r = _run()
    # pump slows down for 40 s while the set point stays: flow and head agree
    dip = slice(460, 500)
    q = r["q_true"].copy()
    q[dip] -= 20.0
    r["flow"] += _lag(q, TAU_METER) - _lag(r["q_true"], TAU_METER)
    r["hp"] += (q / C_HEAD) ** 2 - (r["q_true"] / C_HEAD) ** 2
    frame, _ = _check(r)
    assert "hydraulic" in _flags_in(frame, dip)
    assert _flags_in(frame, dip) <= {"ok", "hydraulic"}
    assert set(frame["flag"]) <= {"ok", "hydraulic"}


def test_pump_off_group_is_not_tested():
    r = _run()
    r["rpm"][:] = 1000.0
    r["flow"][:] = 0.0
    r["hp"][:] = BP
    frame, w = _check(r)
    assert w is None
    assert set(frame["flag"]) == {"pump_off"}


def test_frozen_rpm_falls_back_on_head():
    r = _run()
    r["rpm"][:] = 1000.0  # set point channel frozen, hydraulics move
    r["flow"][PLATEAU] = 0.0
    frame, w = _check(r)
    assert not w.rpm_usable
    flags = frame["flag"].to_numpy()
    assert np.all(flags[PLATEAU] == "dropout")
    assert set(np.delete(flags, np.arange(450, 540))) <= {"ok"}
    err = frame["flow_filled"].to_numpy()[PLATEAU] - r["q_true"][PLATEAU]
    assert np.sqrt(np.mean(err**2)) < 1.5


PARAMS = {
    "Vp0": {"value": 1000.0, "unit": "rpm"},
    "Vpmax": {"value": 1800.0, "unit": "rpm"},
    "F0": {"value": A_FLOW, "unit": "l/s"},
    "Fmax": {"value": 0.046 * 2800.0, "unit": "l/s"},  # slope 0.046, not B_FLOW
    "Imax": {"value": 27000.0, "unit": "A"},
}


def test_slope_is_fitted_when_set_point_moves():
    r = _run()
    r["flow"] += 4.0  # run-to-run offset vs the prior
    frame, w = _check(r, flow_params=PARAMS)
    assert w.b == pytest.approx(B_FLOW, abs=0.001)
    assert set(frame["flag"]) == {"ok"}


def test_flow_params_slope_when_set_point_barely_moves():
    r = _run(
        rpm=np.r_[
            np.full(100, 1000.0), np.linspace(1000, 1060, 300), np.full(300, 1060.0)
        ]
    )
    frame, w = _check(r, flow_params=PARAMS)
    assert w.b == pytest.approx(0.046)
    assert set(frame["flag"]) == {"ok"}


def test_fixed_tau_is_used():
    _, w = _check(_run(), tau=10.0)
    assert w.tau == 10.0


def test_calibration_needs_running_pump():
    t = np.arange(50, dtype=float)
    assert (
        calibrate_flow_witnesses(
            t, np.full(50, 50.0), np.full(50, 1000.0), np.full(50, 2.5)
        )
        is None
    )


def test_reconstruct_follows_model_with_edge_offset():
    t = np.arange(100, dtype=float)
    model = np.linspace(50, 100, 100)
    truth = model + np.linspace(2.0, 4.0, 100)  # drifting offset
    bad = np.zeros(100, bool)
    bad[40:60] = True
    flow = np.where(bad, 0.0, truth)
    filled = reconstruct_flow(t, flow, bad, model)
    assert np.allclose(filled[~bad], truth[~bad])
    assert np.max(np.abs(filled[bad] - truth[bad])) < 0.3
    # gap at the file start: one-sided offset
    bad = np.zeros(100, bool)
    bad[:10] = True
    filled = reconstruct_flow(t, np.where(bad, 0.0, truth), bad, model)
    assert np.max(np.abs(filled[:10] - truth[:10])) < 0.5


def test_flags_vocabulary():
    assert set(REPAIRED) <= set(FLAGS)
    assert set(REPAIRED) == {"dropout", "stuck", "probe"}


@pytest.mark.skipif(
    not M10_REFERENCE.exists(), reason="M10 reference pupitre file not available"
)
def test_m10_reference_dropouts():
    df = pd.read_csv(M10_REFERENCE, sep=r"\s+", skiprows=1)
    stamp = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")
    t = (stamp - stamp.iloc[0]).dt.total_seconds().to_numpy()
    expected = {"1": [], "2": [(407, 496), (1142, 1159)]}
    for g, runs in expected.items():
        frame, _ = check_flow_probe(
            t, df[f"Flow{g}"], df[f"Rpm{g}"], df[f"HP{g}"], df["BP"]
        )
        flagged = frame["flag"].to_numpy() != "ok"
        edges = np.flatnonzero(np.diff(np.r_[0, flagged.astype(np.int8), 0]))
        assert list(zip(edges[::2], edges[1::2], strict=True)) == runs
        assert set(frame["flag"].to_numpy()[flagged]) <= {"dropout"}
