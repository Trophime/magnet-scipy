"""Readers for the experimental series listed under ``experiment_data``.

An entry is either a CSV export::

    {"type": "current", "key": "current", "file": "Courant_GR1_vs_Time.csv"}

or one or several pigbrother TDMS files (Overview or Archive), the same list
as ``sources_archive`` / ``sources_overview`` of a magnetdb overview record::

    {"type": "current", "key": "current",
     "file": ["M9_Archive_230916-1755.tdms", "M9_Archive_230916-1902.tdms"],
     "group": "Courants_Alimentations", "channel": "Courant GR1",
     "time_shift": 0.0}

TDMS support needs the optional ``nptdms`` package (``pip install
magnet-scipy[tdms]``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .csv_utils import _read_csv


def is_tdms(file: str | list[str]) -> bool:
    """Tell whether an ``experiment_data`` ``file`` value refers to TDMS data.

    Parameters
    ----------
    file : str or list of str
        Value of the ``file`` field.

    Returns
    -------
    bool
        ``True`` for a list or a ``.tdms`` path.
    """
    return isinstance(file, list) or str(file).endswith(".tdms")


def source_paths(entry: dict, base_dir: str | Path | None = None) -> list[Path]:
    """Return the source paths of an entry, resolved against ``base_dir``.

    Parameters
    ----------
    entry : dict
        ``experiment_data`` entry.
    base_dir : str or :class:`~pathlib.Path`, optional
        Directory relative paths are resolved against (default: cwd).

    Returns
    -------
    list of :class:`~pathlib.Path`
        One path per source file, in the given order (CSV: a single one).
    """
    files = entry["file"] if isinstance(entry["file"], list) else [entry["file"]]
    base = Path(base_dir) if base_dir else Path()
    return [base / f for f in files]


def read_tdms_channel(
    files: list[str | Path], group: str, channel: str, time_shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Read one channel from one or several TDMS files on a common time axis.

    Files are ordered by ``wf_start_time`` and concatenated; time is counted
    from the start of the earliest file.

    Parameters
    ----------
    files : list of str or :class:`~pathlib.Path`
        TDMS files (e.g. ``sources_archive`` of an overview record).
    group : str
        TDMS group, e.g. ``"Courants_Alimentations"``.
    channel : str
        TDMS channel, e.g. ``"Courant GR1"``.
    time_shift : float
        Added to every time value [s]. Use ``0.5`` to reproduce the time
        axis of the pigbrother CSV exports (sample index + 0.5 s).

    Returns
    -------
    t : :class:`~numpy.ndarray`
        Time [s], increasing.
    y : :class:`~numpy.ndarray`
        Channel values.

    Raises
    ------
    ImportError
        If ``nptdms`` is not installed.
    FileNotFoundError
        If a file does not exist.
    KeyError
        If the group or channel is missing from a file.
    """
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError(
            "TDMS experiment data needs nptdms: pip install magnet-scipy[tdms]"
        ) from exc

    parts = []
    for f in files:
        if not Path(f).is_file():
            raise FileNotFoundError(f"TDMS file not found: {f}")
        with TdmsFile.open(f) as tdms:  # streaming: only this channel is read
            ch = tdms[group][channel]
            p = ch.properties
            parts.append(
                (
                    np.datetime64(p["wf_start_time"]),
                    p.get("wf_start_offset", 0.0)
                    + p["wf_increment"] * np.arange(len(ch)),
                    np.asarray(ch[:], float),
                )
            )
    parts.sort(key=lambda x: x[0])
    t0 = parts[0][0]
    t = np.concatenate([(s - t0) / np.timedelta64(1, "s") + dt for s, dt, _ in parts])
    return t + time_shift, np.concatenate([y for *_, y in parts])


def read_experiment(
    entry: dict, base_dir: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read the time series of an ``experiment_data`` entry.

    Parameters
    ----------
    entry : dict
        ``experiment_data`` entry (see module docstring). For a CSV the
        value column is ``entry["key"]`` if present, else the only column
        other than ``time``.
    base_dir : str or :class:`~pathlib.Path`, optional
        Directory relative paths are resolved against (default: cwd).

    Returns
    -------
    t : :class:`~numpy.ndarray`
        Time [s].
    y : :class:`~numpy.ndarray`
        Values.
    """
    if is_tdms(entry["file"]):
        return read_tdms_channel(
            source_paths(entry, base_dir),
            entry["group"],
            entry["channel"],
            entry.get("time_shift", 0.0),
        )
    df = _read_csv(str(source_paths(entry, base_dir)[0]))
    tcol = "time" if "time" in df.columns else df.columns[0]
    key = entry.get("key")
    vcol = key if key in df.columns else next(c for c in df.columns if c != tcol)
    df = df.sort_values(tcol)
    return df[tcol].to_numpy(float), df[vcol].to_numpy(float)
