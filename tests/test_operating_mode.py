"""Tests for magnet_scipy.operating_mode."""

import numpy as np

from magnet_scipy.operating_mode import infer_operating_mode, shot_modes


def _shot(ratio, n=200, ib_max=20000.0):
    """One ramp up and down with IH = ratio * IB."""
    ib = np.r_[np.linspace(0, ib_max, n), np.linspace(ib_max, 0, n)]
    return ratio * ib, ib


def test_infer_matches_magnetdb_rule():
    assert infer_operating_mode(*_shot(1.0)) == "NORMAL"
    assert infer_operating_mode(*_shot(2.4)) == "ECO"
    assert infer_operating_mode(*_shot(0.5)) == "ECO"  # below 0.66 too
    ih, ib = _shot(1.0)
    assert infer_operating_mode(np.zeros_like(ib), ib) == "NORMAL"  # one GR powered
    assert infer_operating_mode(np.array([np.nan]), np.array([np.nan])) == "UNKNOWN"
    assert infer_operating_mode(np.array([1000.0]), np.array([1000.0])) == "UNKNOWN"


def test_shot_modes_split_at_rest():
    rest = np.zeros(5)
    ih1, ib1 = _shot(2.4)
    ih2, ib2 = _shot(1.0)
    ih = np.r_[rest, ih1, rest, ih2, rest]
    ib = np.r_[rest, ib1, rest, ib2, rest]
    modes = shot_modes(ih, ib)
    assert modes.shape == ih.shape
    assert set(modes[:5]) == {"rest"} and set(modes[-5:]) == {"rest"}
    assert set(modes[5 + 1 : 5 + 399]) == {"ECO"}
    second = slice(5 + 400 + 5 + 1, 5 + 400 + 5 + 399)
    assert set(modes[second]) == {"NORMAL"}


def test_shot_modes_insert_sign_ignored_for_activity():
    _, ib = _shot(2.4)
    modes = shot_modes(np.full_like(ib, -6.0), ib)  # insert idle at -6 A
    assert set(modes[1:-1]) == {"NORMAL"}  # only the Bitters is powered
