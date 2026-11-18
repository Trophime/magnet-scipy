"""Operating mode (ECO / NORMAL) of a two-supply housing from its currents.

The rule is ``infer_operating_mode`` of magnetdb-duckdb-demo
(``to_duckdb/crud.py``), which magnetdb applies to each Overview file to fill
``overview_records.mode``: fit IH against IB on the start of the ramp
(``50 A < IB < 0.25 max(IB)``); a slope in ``[0.66, 1.5]`` is NORMAL, any
other slope ECO; a record where IH or IB never exceeds 50 A is NORMAL.

:func:`shot_modes` applies it per shot (stretch between rests) instead of
per file, so a record holding several shots, or a pupitre window shorter
than its Overview file, gets the mode of each of its own shots.
"""

from __future__ import annotations

import numpy as np

ZERO_CURRENT_A = 50.0  # same threshold for IH and IB as magnetdb
NORMAL_SLOPE = (0.66, 1.5)  # dIH/dIB range classified NORMAL


def infer_operating_mode(ih: np.ndarray, ib: np.ndarray) -> str:
    """Classify the operating mode from the IH(IB) relation near IB = 0.

    Port of magnetdb-duckdb-demo ``crud.infer_operating_mode`` for two
    current arrays (both channels present).

    Parameters
    ----------
    ih : :class:`~numpy.ndarray`
        Insert current IH [A].
    ib : :class:`~numpy.ndarray`
        Bitters current IB [A].

    Returns
    -------
    str
        ``"NORMAL"``, ``"ECO"`` or ``"UNKNOWN"`` (no finite sample, or fewer
        than two samples with IB above the zero threshold).
    """
    ih, ib = np.asarray(ih, float), np.asarray(ib, float)
    mask = np.isfinite(ih) & np.isfinite(ib)
    ih, ib = ih[mask], ib[mask]
    if len(ih) == 0:
        return "UNKNOWN"
    if np.max(np.abs(ih)) < ZERO_CURRENT_A or np.max(np.abs(ib)) < ZERO_CURRENT_A:
        return "NORMAL"
    fit = (ZERO_CURRENT_A < ib) & (0.25 * np.max(ib) > ib)
    if fit.sum() < 10:
        fit = ZERO_CURRENT_A < ib
    if fit.sum() < 2:
        return "UNKNOWN"
    slope, _ = np.polyfit(ib[fit], ih[fit], 1)
    return "NORMAL" if NORMAL_SLOPE[0] <= slope <= NORMAL_SLOPE[1] else "ECO"


def shot_modes(ih: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Operating mode of every sample, decided shot by shot.

    A shot is a maximal stretch of samples where ``|IH|`` or ``|IB|`` exceeds
    the zero threshold (50 A); each shot is classified with
    :func:`infer_operating_mode`.

    Parameters
    ----------
    ih : :class:`~numpy.ndarray`
        Insert current IH [A], in time order.
    ib : :class:`~numpy.ndarray`
        Bitters current IB [A], same samples.

    Returns
    -------
    :class:`~numpy.ndarray` of str
        ``"ECO"``, ``"NORMAL"`` or ``"UNKNOWN"`` for samples of a shot,
        ``"rest"`` for samples where both currents are at or below 50 A (or
        not finite).
    """
    ih, ib = np.asarray(ih, float), np.asarray(ib, float)
    active = (np.abs(ih) > ZERO_CURRENT_A) | (np.abs(ib) > ZERO_CURRENT_A)
    modes = np.full(len(ih), "rest", dtype=object)
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        modes[start:stop] = infer_operating_mode(ih[start:stop], ib[start:stop])
    return modes
