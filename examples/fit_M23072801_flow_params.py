"""Fit the water flow params of the M9_M23072801 magnets from the campaign pupitre data.

Reads the pupitre files of all runs listed in
``examples/M9_M23072801_C*/provenance.json``, keeps the samples whose
operating mode is ``--mode`` (default ``ECO``), fits the pump speed, flow and
inlet pressure of supply group ``--group`` against its current (GR1 = insert
M23072801, GR2 = M9 Bitters), and writes
``examples/M9_M23072801/<magnet>-flow_params.json`` in the format read by
:func:`python_magnetcooling.waterflow_factory.from_flow_params`.

Fit, all on samples with |I| above ``--threshold``:

1. Imax: unless ``--imax`` is given, the breakpoint of
   :func:`python_magnetcooling.fitting.fit_pump_speed_piecewise` (2 segments)
   on the median pump speed per 100 A bin, sorted by current. Only the
   breakpoint is used: the library's piecewise ``vpmax`` is the speed at Imax
   (not the rise above Vp0 that the model expects) and its extracted
   coefficients are not usable.
2. Pump: ``Vp = Vpmax (min(I, Imax)/Imax)^2 + Vp0`` by least squares, i.e. the
   clamped model of ``WaterFlow`` (``fit_pump_speed_simple`` fits the
   unclamped parabola, which samples above Imax pull off).
3. Flow and inlet pressure:
   :func:`~python_magnetcooling.fitting.fit_flow_rate` and
   :func:`~python_magnetcooling.fitting.fit_pressure` with that pump fit;
   back pressure: mean and standard deviation of ``BP``.

The mode of each sample is inferred shot by shot from the pupitre supply
currents (:func:`magnet_scipy.operating_mode.shot_modes`, magnetdb's
``infer_operating_mode`` rule). A shot the rule cannot classify falls back to
the run's ``modes`` in ``provenance.json`` (``overview_records.mode``) when
there is exactly one.

Requires ``python_magnetcooling``. See ``prompts/flow_params_check.md``.

Example
-------
::

    python examples/fit_M23072801_flow_params.py --data-root ~/LNCMIG-Data/records
    python examples/fit_M23072801_flow_params.py --data-root ~/LNCMIG-Data/records --group GR2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLES = Path(__file__).resolve().parent
BASE_CASE = EXAMPLES / "M9_M23072801"
# Pupitre current and hydraulics of each supply group; BP is shared.
CURRENT = {"GR1": ("Idcct1", "Idcct2"), "GR2": ("Idcct3", "Idcct4")}
COLUMNS = {
    "GR1": {"pump_speed": "Rpm1", "flow_rate": "Flow1", "pressure": "HP1"},
    "GR2": {"pump_speed": "Rpm2", "flow_rate": "Flow2", "pressure": "HP2"},
}
OUTPUT = {
    "GR1": BASE_CASE / "M23072801-flow_params.json",
    "GR2": BASE_CASE / "M9Bitters_18MW-flow_params.json",
}
BIN_A = 100.0  # current bin for the Imax breakpoint search [A]


def campaign_runs() -> dict[str, list[str]]:
    """Return the campaign runs: pupitre path -> ``modes`` of ``provenance.json``.

    Returns
    -------
    dict
        Pupitre paths relative to the data root, without duplicates, each
        with the ``overview_records.mode`` list of its run (may be empty).
    """
    runs: dict[str, list[str]] = {}
    for prov in sorted(EXAMPLES.glob("M9_M23072801_C*/provenance.json")):
        for run in json.loads(prov.read_text())["runs"].values():
            runs.setdefault(run["pupitre"], run.get("modes") or [])
    return runs


def load(
    data_root: Path, runs: dict[str, list[str]], group: str = "GR1"
) -> pd.DataFrame:
    """Concatenate the hydraulics of one supply group, with the mode of each sample.

    Parameters
    ----------
    data_root : Path
        Root of the raw records.
    runs : dict
        Output of :func:`campaign_runs`.
    group : str
        ``"GR1"`` (insert) or ``"GR2"`` (Bitters).

    Returns
    -------
    :class:`~pandas.DataFrame`
        Columns ``run``, ``mode``, ``current`` (|I| of the group) [A],
        ``pump_speed`` [rpm], ``flow_rate`` [l/s], ``pressure`` and
        ``back_pressure`` [bar].
    """
    from magnet_scipy.operating_mode import shot_modes

    frames = []
    for run, prov_modes in runs.items():
        pup = pd.read_csv(
            data_root / run, sep="\t", skiprows=1
        )  # line 1: magnet metadata
        ih, ib = (pup[list(CURRENT[g])].sum(axis=1).to_numpy() for g in ("GR1", "GR2"))
        mode = shot_modes(ih, ib)
        if len(prov_modes) == 1:
            mode[mode == "UNKNOWN"] = prov_modes[0]
        frames.append(
            pd.DataFrame(
                {
                    "run": Path(run).stem,
                    "mode": mode,
                    "current": np.abs(ih if group == "GR1" else ib),
                    **{k: pup[c] for k, c in COLUMNS[group].items()},
                    "back_pressure": pup["BP"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True).dropna()


def detect_imax(current: np.ndarray, pump_speed: np.ndarray) -> float:
    """Current at which the pump speed saturates [A], from a 2-segment piecewise fit.

    Parameters
    ----------
    current : :class:`~numpy.ndarray`
        |I| [A].
    pump_speed : :class:`~numpy.ndarray`
        Pump speed [rpm].

    Returns
    -------
    float
        Breakpoint of :func:`python_magnetcooling.fitting.fit_pump_speed_piecewise`
        on the median speed per ``BIN_A`` bin (equal weight per current,
        sorted as the library's end-point check expects).

    Raises
    ------
    RuntimeError
        If the piecewise fit keeps a single segment (no saturation found).
    """
    from python_magnetcooling.fitting import fit_pump_speed_piecewise

    binned = (
        pd.Series(pump_speed).groupby((current // BIN_A) * BIN_A + BIN_A / 2).median()
    )
    pw = fit_pump_speed_piecewise(
        binned.index.to_numpy(float), binned.to_numpy(float), max_segments=2
    )
    if not pw.imax_detected:
        raise RuntimeError("no pump speed saturation found; give --imax")
    return float(pw.imax)


def fit_pump_clamped(current: np.ndarray, pump_speed: np.ndarray, imax: float):
    """Fit ``Vp = Vpmax (min(I, Imax)/Imax)^2 + Vp0`` at a given Imax.

    Parameters
    ----------
    current : :class:`~numpy.ndarray`
        |I| [A].
    pump_speed : :class:`~numpy.ndarray`
        Pump speed [rpm].
    imax : float
        Current at full pump speed [A].

    Returns
    -------
    :class:`python_magnetcooling.fitting.PumpSpeedFit`
        ``vpmax`` (rise above Vp0) and ``vp0`` [rpm], with a ``FitResult``.
    """
    from python_magnetcooling.fitting import FitResult, PumpSpeedFit
    from scipy.optimize import curve_fit

    def model(i, vpmax, vp0):
        return vpmax * (np.minimum(i, imax) / imax) ** 2 + vp0

    popt, pcov = curve_fit(model, current, pump_speed, p0=(1800.0, 1000.0))
    residuals = pump_speed - model(current, *popt)
    r2 = 1.0 - np.sum(residuals**2) / np.sum((pump_speed - pump_speed.mean()) ** 2)
    fit = FitResult(popt, np.sqrt(np.diag(pcov)), r2, residuals)
    return PumpSpeedFit(
        vpmax=float(popt[0]), vp0=float(popt[1]), imax=imax, fit_result=fit
    )


def main(argv: list[str] | None = None) -> None:
    """Fit and write the flow params file; print the fit statistics."""
    from python_magnetcooling.fitting import (
        compute_back_pressure_stats,
        fit_flow_rate,
        fit_pressure,
    )

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--data-root", type=Path, required=True, help="root of the raw records"
    )
    p.add_argument(
        "--group",
        default="GR1",
        choices=sorted(CURRENT),
        help="supply group: GR1 = insert, GR2 = Bitters",
    )
    p.add_argument("--mode", default="ECO", help="keep the samples in this mode")
    p.add_argument(
        "--imax",
        type=float,
        default=None,
        help="current at full pump speed [A] (default: detected)",
    )
    p.add_argument(
        "--threshold", type=float, default=300.0, help="ignore |I| below [A]"
    )
    p.add_argument("--output", type=Path, help="default: <magnet>-flow_params.json")
    a = p.parse_args(argv)

    output = a.output or OUTPUT[a.group]
    runs = campaign_runs()
    data = load(a.data_root.expanduser(), runs, a.group)
    print(
        f"{a.group} samples above the threshold per run and inferred mode "
        "(provenance: overview_records.mode):"
    )
    above = data[data["current"] > a.threshold]
    counts = above.groupby(["run", "mode"]).size().unstack(fill_value=0)
    prov = {Path(r).stem: "/".join(m) or "-" for r, m in runs.items()}
    print(counts.assign(provenance=counts.index.map(prov)).to_string())

    df = above[above["mode"] == a.mode]
    i, vp, flow, hp, bp = (
        df[c].to_numpy(float) for c in ("current", *COLUMNS[a.group], "back_pressure")
    )
    imax = a.imax if a.imax is not None else detect_imax(i, vp)
    pump = fit_pump_clamped(i, vp, imax)
    flow_fit = fit_flow_rate(i, flow, pump)
    hp_fit = fit_pressure(i, hp, pump)
    bp_mean, bp_std = compute_back_pressure_stats(bp)
    params = {
        "Vp0": {"value": pump.vp0, "unit": "rpm"},
        "Vpmax": {"value": pump.vpmax, "unit": "rpm"},
        "F0": {"value": float(flow_fit.parameters[0]), "unit": "l/s"},
        "Fmax": {"value": float(flow_fit.parameters[1]), "unit": "l/s"},
        "Pmax": {"value": float(hp_fit.parameters[1]), "unit": "bar"},
        "Pmin": {"value": float(hp_fit.parameters[0]), "unit": "bar"},
        "BP": {"value": bp_mean, "unit": "bar"},
        "Imax": {"value": imax, "unit": "A"},
    }
    output.write_text(json.dumps(params, indent=4) + "\n")

    source = "given" if a.imax is not None else "detected"
    print(
        f"{df['run'].nunique()} runs, {len(df)} {a.mode} samples above "
        f"{a.threshold:g} A, Imax {source} -> {output}"
    )
    for k, v in params.items():
        print(f"  {k:6s} = {v['value']:10.4f} {v['unit']}")
    print(f"  BP std = {bp_std:.3f} bar")
    for name, fit in (("pump", pump.fit_result), ("flow", flow_fit), ("HP", hp_fit)):
        rms = float(np.sqrt(np.mean(np.asarray(fit.residuals) ** 2)))
        print(f"  {name:5s} R2 = {fit.r_squared:.5f}  RMS = {rms:.3f}")


if __name__ == "__main__":
    main()
