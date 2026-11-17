"""Energy balance of one magnet run from pupitre and pigbrother records.

Power data come from pigbrother (Overview 1 Hz or Archive 120 Hz ``.tdms``),
cooling data from the pupitre (``.txt``). Both are put on a common 1 s UTC
grid; the time offset between the two records is estimated by matching the
pupitre ``Pmagnet`` with the pigbrother magnet power. On run 2024.07.15 it is
+1 s against the Archive but +24 s against the Overview, so the offset comes
from the Overview time stamps, not from the pupitre clock.

Terms, per steady plateau and integrated over the run:

- magnets: magnet voltage times the current of the supply group feeding it,
  ``ALL_internes`` (insert taps) and ``ALL_externes`` (Bitter taps) paired
  with ``Courant_GR1``/``Courant_GR2`` according to the housing wiring
  (M9: insert on GR1; M8, M10: insert on GR2), the pigbrother equivalent of
  the pupitre ``Pmagnet``;
- busbars: converter DC output ``Tension_A * Courant_A`` of the group minus
  the power of the magnet it feeds; lost to the hall air (natural convection
  and radiation), so left out of every water balance;
- converters: conduction losses of the Graetz bridges from ``Courants_ponts``,
  ``2 * (V_T0 * I + r_T * I**2)`` per bridge (two thyristors always conduct
  in series). ``V_T0`` and ``r_T`` are model parameters, not measurements;
- pumps: hydraulic power dissipated between the HP and BP taps,
  ``sum (HP_i - BP) * Flow_i`` (measured), and an electrical estimate
  ``c * (Rpm1**3 + Rpm2**3)`` (``c`` from a regression on the HT1 feeder);
- power supplies from the pupitre ``Ptot``:
  ``P_alim = Ptot - P_magnets - P_busbars`` (``- P_pumps`` with
  ``--ptot-includes-pumps``). ``Ptot`` is 0 at rest while the pumps run at
  1000 rpm, so by default it is taken as excluding the pumps.

Temperatures:

- ``Tout`` estimate: ``Tin + (P_magnets + P_hyd) / (rho*cp*(Flow1 + Flow2))``
  with ``Tin = (Tin1 + Tin2) / 2`` (both sensors see the same exchanger
  outlet water; their ~0.5 K difference is a sensor offset, so they are not
  flow-weighted), optionally delayed and low-pass filtered;
- ``TAlimout`` estimate: ``Tin + P_converters / (rho*cp*Flow_conv)``, only when
  the converter cooling flow is given; otherwise the flow that would close the
  balance is reported.

Heat exchanger:

- hot (primary) side, from the pupitre: inlet is ``Tout`` mixed with the
  power-supply return ``TAlimout`` (weight ``Flow_conv``, 0 when not given),
  outlet is ``Tin``; ``Q_hot = rho*cp*(Flow1 + Flow2 + Flow_conv)*(T_in - T_out)``;
- hot side, from SUPERVISION (``--bdd``): same formula with ``ted`` in place
  of ``Tout`` and ``tsd`` in place of ``Tin`` (same measurement points, other
  sensors); ``ted - Tout`` and ``tsd - Tin`` are reported as sensor checks;
- cold (secondary, river water) side: ``Q_cold = rho*cp*debitbrut*(tsb - teb)``,
  ``debitbrut`` [m3/h] from the pupitre, ``teb``/``tsb`` from SUPERVISION
  when given (finer resolution), from the pupitre otherwise.

SUPERVISION ``bdd.csv`` timestamps are naive UTC: magnetdb-duckdb-demo's
``build_from_scratch.sh`` exports it with ``mysql_connect.py --utc`` (checked
on M9 runs 2026.10.02 and 2026.10.28: ``teb``/``tsb`` agree with the pupitre
within 0.05 K once the pupitre's local time is converted to UTC). Exports
made before that option (e.g. a ``bdd.csv`` dated 2026-11-08) are in local
time and give a 1-2 h shift.

Example
-------
::

    python examples/energy_balance.py \\
        "~/LNCMIG-Data/records/srv-data-install/M9/2024.07.15 - 09:07:12.txt" \\
        ~/LNCMIG-Data/records/pbsurv/M9/Overview/M9_Overview_240715-0907.tdms \\
        --delay 22 --tau 19 --csv balance.csv --plot balance.png

    python examples/energy_balance.py \\
        "~/LNCMIG-Data/records/srv-data-install/M9/2026.10.02 - 09:10:07.txt" \\
        ~/LNCMIG-Data/records/pbsurv/M9/Overview/M9_Overview_261002-0909.tdms \\
        --bdd ~/github/magnetdb-duckdb-demo/Data/bdd.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile

SITE_TZ = "Europe/Paris"  # pupitre timestamps are local time; pigbrother is UTC
RHO_CP = 4.17e6  # [J/(m3 K)] water at ~20 C (flowmeters are on the cold side)
ACTIVE_CURRENT_A = 100.0  # a group below this pupitre current is off
CONVERTERS = {"GR1": ("A1", "A2"), "GR2": ("A3", "A4")}
PUPITRE_CURRENT = {"GR1": ("Idcct1", "Idcct2"), "GR2": ("Idcct3", "Idcct4")}
BRIDGES = range(1, 5)
MAGNET_TAPS = {"IH": "ALL_internes", "IB": "ALL_externes"}  # insert / Bitter taps
# Base wiring (supply group -> pupitre current) of python_magnetrun's housing
# configs; used only when python_magnetrun is not installed (no wiring versions).
HOUSING_BASE = {
    "M5": {"GR1": "IB"},
    "M7": {"GR1": "IH"},
    "M8": {"GR1": "IB", "GR2": "IH"},
    "M9": {"GR1": "IH", "GR2": "IB"},
    "M10": {"GR1": "IB", "GR2": "IH"},
}


def _group_channels(gr: str, tap: str | None = None) -> list[tuple[str, str]]:
    """Return the ``(group, channel)`` pigbrother pairs needed for one GR."""
    chans = [("Courants_Alimentations", f"Courant_{gr}")]
    if tap:
        chans.append(("Tensions_Aimant", tap))
    for a in CONVERTERS[gr]:
        chans += [
            ("Courants_Alimentations", f"Courant_{a}"),
            ("Tensions_Alimentations", f"Tension_{a}"),
        ]
        chans += [("Courants_ponts", f"{a}_p{p}") for p in BRIDGES]
    return chans


def read_pupitre(path: Path) -> pd.DataFrame:
    """Read a pupitre file on a naive UTC time index.

    Parameters
    ----------
    path : :class:`~pathlib.Path`
        Pupitre ``.txt`` file (tab separated, ``Date`` and ``Time`` columns,
        site local time).

    Returns
    -------
    :class:`~pandas.DataFrame`
        Numeric pupitre columns indexed by UTC time [1 s], duplicated or
        invalid timestamps dropped.
    """
    df = pd.read_csv(path, sep="\t", skiprows=1)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    local = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")
    utc = local.dt.tz_localize(SITE_TZ, ambiguous="NaT", nonexistent="shift_forward")
    df.index = pd.DatetimeIndex(
        utc.dt.tz_convert("UTC").dt.tz_localize(None), name="utc"
    )
    df = df[df.index.notna()]
    return df[~df.index.duplicated()].select_dtypes("number")


def read_pigbrother(paths: list[Path]) -> pd.DataFrame:
    """Read the power channels of pigbrother files, averaged on a 1 s grid.

    Group and channel names are normalised (spaces replaced by ``_``, as in
    the 2023 files). Channels missing from a file are NaN for that file.

    Parameters
    ----------
    paths : list of :class:`~pathlib.Path`
        Overview (1 Hz) or Archive (120 Hz) ``.tdms`` files of the run.

    Returns
    -------
    :class:`~pandas.DataFrame`
        One column per channel [A] or [V], indexed by UTC time floored to
        the second.

    Raises
    ------
    ValueError
        If none of the files holds any of the channels.
    """
    wanted = [gc for gr in CONVERTERS for gc in _group_channels(gr)]
    wanted += [("Tensions_Aimant", tap) for tap in MAGNET_TAPS.values()]
    frames = []
    for path in paths:
        cols = {}
        with TdmsFile.open(path) as tdms:  # streaming: one channel in memory at a time
            groups = {g.name.replace(" ", "_"): g for g in tdms.groups()}
            for grp, ch in wanted:
                if grp not in groups:
                    continue
                chans = {c.name.replace(" ", "_"): c for c in groups[grp].channels()}
                if ch not in chans or len(chans[ch]) == 0:
                    continue
                c = chans[ch]
                t0 = pd.Timestamp(c.properties["wf_start_time"])
                t = t0 + pd.to_timedelta(
                    np.arange(len(c)) * float(c.properties["wf_increment"]), unit="s"
                )
                cols[ch] = (
                    pd.Series(np.asarray(c[:], dtype=float), index=t)
                    .resample("1s")
                    .mean()
                )
        if cols:
            frames.append(pd.DataFrame(cols))
    if not frames:
        raise ValueError("no power channel found in the pigbrother files")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated()]


def read_supervision(
    path: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Read the SUPERVISION exchanger temperatures for one time window.

    The multi-GB ``bdd.csv`` export is filtered on read with DuckDB.

    Parameters
    ----------
    path : :class:`~pathlib.Path`
        SUPERVISION ``bdd.csv`` export (``timestamp`` in naive UTC, about
        5 s sampling).
    start, end : :class:`~pandas.Timestamp`
        Naive UTC window.

    Returns
    -------
    :class:`~pandas.DataFrame`
        Columns ``bdd_teb``, ``bdd_tsb`` (river water in/out) and ``bdd_ted``,
        ``bdd_tsd`` (primary water at the ``Tout`` and ``Tin`` measurement
        points) [°C], naive UTC index; empty
        if the export does not cover the window.
    """
    import duckdb

    with duckdb.connect() as con:
        df = con.execute(
            "SELECT timestamp, teb, tsb, ted, tsd FROM read_csv_auto(?) "
            "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            [str(path), start - pd.Timedelta("1min"), end + pd.Timedelta("1min")],
        ).df()
    df.index = pd.DatetimeIndex(df.pop("timestamp"))
    return df[~df.index.duplicated()].add_prefix("bdd_")


def housing_wiring(housing: str, when: pd.Timestamp) -> tuple[dict[str, str], str]:
    """Return the magnet voltage channel paired with each supply group.

    Uses ``python_magnetrun``'s housing configuration (with the wiring
    version in force at *when*) when installed, else :data:`HOUSING_BASE`.

    Parameters
    ----------
    housing : str
        Housing name, e.g. ``"M9"``.
    when : :class:`~pandas.Timestamp`
        Naive UTC start of the run.

    Returns
    -------
    dict
        ``{"GR1": "ALL_internes", ...}``; a group feeding no magnet is absent.
    str
        Where the wiring comes from, for the report.
    """
    try:
        from python_magnetrun.housing_config import get_housing_config
    except ImportError:
        currents = HOUSING_BASE[housing]
        source = "built-in base wiring (python_magnetrun not installed, versions not applied)"
    else:
        cfg = get_housing_config(housing).at(when.tz_localize("UTC").to_pydatetime())
        currents = {gr: cfg.get_pupitre_channel(f"Référence_{gr}") for gr in CONVERTERS}
        source = f"python_magnetrun housing config, version {cfg.version or 'base'}"
    wiring = {gr: MAGNET_TAPS[c] for gr, c in currents.items() if c in MAGNET_TAPS}
    return wiring, source


def estimate_lag(p_ref: pd.Series, p_pupitre: pd.Series, max_lag: int = 120) -> int:
    """Return the shift to add to pupitre times to match pigbrother.

    Parameters
    ----------
    p_ref : :class:`~pandas.Series`
        Pigbrother magnet power [MW] on a 1 s UTC grid.
    p_pupitre : :class:`~pandas.Series`
        Pupitre ``Pmagnet`` [MW] on a 1 s UTC grid.
    max_lag : int, optional
        Search range [s], both directions.

    Returns
    -------
    int
        Shift [s] minimising the rms difference; 0 when fewer than 60
        common samples are available.
    """
    best = (np.inf, 0)
    for lag in range(-max_lag, max_lag + 1):
        d = (
            p_ref - p_pupitre.set_axis(p_pupitre.index + pd.Timedelta(seconds=lag))
        ).dropna()
        if len(d) >= 60:
            best = min(best, (float(np.mean(d**2)), lag))
    return best[1]


def low_pass(x: pd.Series, delay: float, tau: float) -> pd.Series:
    """Delay then first-order filter a series on a regular 1 s grid.

    Parameters
    ----------
    x : :class:`~pandas.Series`
        Input on a 1 s grid; NaN samples hold the filter state.
    delay : float
        Pure transport delay [s], rounded to the second.
    tau : float
        Time constant [s]; 0 disables the filter.

    Returns
    -------
    :class:`~pandas.Series`
        Filtered series, same index.
    """
    x = x.shift(int(round(delay)))
    if tau <= 0:
        return x
    a = 1.0 - np.exp(-1.0 / tau)
    v = x.to_numpy()
    y = np.full_like(v, np.nan)
    state = np.nan
    for k, xk in enumerate(v):
        if not np.isnan(xk):
            state = xk if np.isnan(state) else state + a * (xk - state)
        y[k] = state
    return pd.Series(y, index=x.index)


def energy_balance(
    df: pd.DataFrame,
    wiring: dict[str, str],
    vt0: float,
    rt: float,
    pump_coef: float,
    conv_flow: float | None,
    delay: float,
    tau: float,
    talim_tau: float,
    ptot_pumps: bool = False,
) -> pd.DataFrame:
    """Compute power terms and temperature estimates on the merged 1 s grid.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Merged pigbrother channels and pupitre columns, regular 1 s grid.
    wiring : dict
        Magnet voltage channel fed by each supply group, from
        :func:`housing_wiring`; a group absent from it feeds no magnet.
    vt0 : float
        Thyristor threshold voltage [V].
    rt : float
        Thyristor slope resistance [Ohm].
    pump_coef : float
        Pump electrical power per pump at 1000 rpm [MW]; scaled as rpm**3.
    conv_flow : float or None
        Total cooling flow of the converter circuits [l/s]; None if unknown.
    delay : float
        Transport delay applied to the ``Tout`` estimate [s].
    tau : float
        Time constant of the ``Tout`` estimate [s].
    talim_tau : float
        Time constant of the ``TAlimout`` estimate [s].
    ptot_pumps : bool, optional
        Whether the pupitre ``Ptot`` includes the pumps; if so, ``P_pump_el``
        is subtracted from it to get ``P_alim``.

    Returns
    -------
    :class:`~pandas.DataFrame`
        Powers [MW] (``P_magnet``, ``P_busbar``, ``P_conv``, ``P_hyd``,
        ``P_pump_el``, per group where relevant), the power-supply losses
        from ``Ptot`` ``P_alim``, ``Pmagnet_pupitre`` and
        ``Ptot_pupitre`` [MW], temperatures [°C] (``Tin``, ``Tout``,
        ``Tout_est``, ``TAlimout``, ``TAlim_est``), calorimetric ratio
        ``k_cal`` [-], the converter flow closing the ``TAlimout`` balance
        ``F_conv_implied`` [l/s], and the heat exchanger terms [MW]
        ``Q_hx_hot`` (pupitre), ``Q_hx_hot_bdd`` (SUPERVISION ``ted``/``tsd``,
        NaN without ``bdd_*`` columns) and ``Q_hx_cold``, with the sensor
        checks ``dT_ted`` = ``ted - Tout`` and ``dT_tsd`` = ``tsd - Tin`` [K].
    """
    out = pd.DataFrame(index=df.index)
    for gr, convs in CONVERTERS.items():
        p_mag = (
            df[wiring[gr]] * df[f"Courant_{gr}"]
            if gr in wiring
            else pd.Series(0.0, index=df.index)
        )
        p_dc = sum(df[f"Tension_{a}"] * df[f"Courant_{a}"] for a in convs)
        bridges = [df[f"{a}_p{p}"] for a in convs for p in BRIDGES]
        out[f"P_magnet_{gr}"] = p_mag / 1e6
        out[f"P_busbar_{gr}"] = (p_dc - p_mag) / 1e6
        out[f"P_conv_{gr}"] = (
            sum(2 * (vt0 * b.abs() + rt * b**2) for b in bridges) / 1e6
        )
    for term in ("P_magnet", "P_busbar", "P_conv"):
        out[term] = out[f"{term}_GR1"] + out[f"{term}_GR2"]
    out["Pmagnet_pupitre"] = df["Pmagnet"]
    out["P_hyd"] = (
        (df.HP1 - df.BP) * df.Flow1 + (df.HP2 - df.BP) * df.Flow2
    ) * 1e-4  # bar*l/s -> MW
    out["P_pump_el"] = pump_coef * ((df.Rpm1 / 1e3) ** 3 + (df.Rpm2 / 1e3) ** 3)
    out["P_alim"] = (
        df.Ptot - out.P_magnet - out.P_busbar - (out.P_pump_el if ptot_pumps else 0.0)
    )

    flow = (df.Flow1 + df.Flow2) / 1e3  # [m3/s]
    tin = (df.Tin1 + df.Tin2) / 2
    heat = out.P_magnet + out.P_hyd
    out["Tin"], out["Tout"], out["TAlimout"] = tin, df.Tout, df.TAlimout
    out["Tout_est"] = low_pass(tin + heat * 1e6 / (RHO_CP * flow), delay, tau)
    out["k_cal"] = RHO_CP * flow * (df.Tout - tin) / (heat * 1e6)
    out["F_conv_implied"] = 1e3 * out.P_conv * 1e6 / (RHO_CP * (df.TAlimout - tin))
    out["TAlim_est"] = (
        low_pass(tin + out.P_conv * 1e6 / (RHO_CP * conv_flow / 1e3), 0, talim_tau)
        if conv_flow
        else np.nan
    )

    # heat exchanger: hot side = magnet branches (Tout) + power-supply return
    # (TAlimout), both cooled down to Tin
    f_conv = (conv_flow or 0.0) / 1e3  # [m3/s]
    out["Ptot_pupitre"] = df["Ptot"]
    out["Q_hx_hot"] = (
        RHO_CP * (flow * (df.Tout - tin) + f_conv * (df.TAlimout - tin)) / 1e6
    )
    bdd = "bdd_ted" in df
    if bdd:  # SUPERVISION ted/tsd: same points as Tout/Tin, other sensors
        out["Q_hx_hot_bdd"] = (
            RHO_CP
            * (flow * (df.bdd_ted - df.bdd_tsd) + f_conv * (df.TAlimout - df.bdd_tsd))
            / 1e6
        )
    else:
        out["Q_hx_hot_bdd"] = np.nan
    out["dT_ted"] = df.bdd_ted - df.Tout if bdd else np.nan
    out["dT_tsd"] = df.bdd_tsd - tin if bdd else np.nan
    teb = df.bdd_teb.fillna(df.teb) if bdd else df.teb
    tsb = df.bdd_tsb.fillna(df.tsb) if bdd else df.tsb
    out["Q_hx_cold"] = RHO_CP * df.debitbrut / 3600 * (tsb - teb) / 1e6
    return out


def steady_segments(
    p: pd.Series, p_min: float = 1.0, tol: float = 0.2, min_len: int = 300
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the steady plateaus of a power series.

    Parameters
    ----------
    p : :class:`~pandas.Series`
        Magnet power [MW] on a 1 s grid.
    p_min : float, optional
        Minimum power of a plateau [MW].
    tol : float, optional
        Maximum standard deviation over 5 min [MW].
    min_len : int, optional
        Minimum plateau length [s].

    Returns
    -------
    list of tuple
        ``(start, end)`` UTC timestamps of each plateau.
    """
    ok = (p.rolling(120, center=True, min_periods=60).median() > p_min) & (
        p.rolling(300, center=True, min_periods=150).std() < tol
    )
    seg = (ok != ok.shift()).cumsum()
    return [
        (s.index[0], s.index[-1])
        for _, s in ok[ok].groupby(seg[ok])
        if len(s) >= min_len
    ]


def summarize(bal: pd.DataFrame, settle: int = 120) -> pd.DataFrame:
    """Return the median of each term over every steady plateau.

    Parameters
    ----------
    bal : :class:`~pandas.DataFrame`
        Output of :func:`energy_balance`.
    settle : int, optional
        Seconds skipped at the start of each plateau [s].

    Returns
    -------
    :class:`~pandas.DataFrame`
        One row per plateau, indexed by its local start time.
    """
    rows = {}
    for t0, t1 in steady_segments(bal.P_magnet.fillna(bal.Pmagnet_pupitre)):
        s = bal.loc[t0 + pd.Timedelta(seconds=settle) : t1].median()
        start = t0.tz_localize("UTC").tz_convert(SITE_TZ).strftime("%H:%M:%S")
        rows[start] = {
            "dur[min]": (t1 - t0).total_seconds() / 60,
            "P_mag": s.P_magnet,
            "P_pup": s.Pmagnet_pupitre,
            "P_bus_GR1": s.P_busbar_GR1,
            "P_bus_GR2": s.P_busbar_GR2,
            "P_conv": s.P_conv,
            "P_hyd": s.P_hyd,
            "P_pump_el": s.P_pump_el,
            "P_alim": s.P_alim,
            "Tin": s.Tin,
            "Tout": s.Tout,
            "Tout_est": s.Tout_est,
            "dTout": s.Tout - s.Tout_est,
            "k_cal": s.k_cal,
            "TAlim": s.TAlimout,
            "TAlim_est": s.TAlim_est,
            "F_conv_impl[l/s]": s.F_conv_implied,
            "Ptot": s.Ptot_pupitre,
            "Q_hot": s.Q_hx_hot,
            "Q_hot_bdd": s.Q_hx_hot_bdd,
            "Q_cold": s.Q_hx_cold,
            "cold/hot": s.Q_hx_cold / s.Q_hx_hot,
            "ted-Tout": s.dT_ted,
            "tsd-Tin": s.dT_tsd,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def plot(bal: pd.DataFrame, path: Path, title: str) -> None:
    """Save powers, ``Tout``, ``TAlimout`` and heat exchanger balance to an image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = bal.index.tz_localize("UTC").tz_convert(SITE_TZ)
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(11, 12))
    for col in ("P_magnet", "P_busbar", "P_conv", "P_hyd", "P_pump_el", "P_alim"):
        ax[0].plot(t, bal[col], label=col)
    ax[0].set_ylabel("power [MW]")
    ax[0].set_yscale("symlog", linthresh=0.1)
    ax[1].plot(t, bal.Tout, label="Tout measured")
    ax[1].plot(t, bal.Tout_est, label="Tout estimated")
    ax[1].plot(t, bal.Tin, label="Tin = (Tin1 + Tin2)/2")
    ax[1].set_ylabel("[°C]")
    ax[2].plot(t, bal.TAlimout, label="TAlimout measured")
    if bal.TAlim_est.notna().any():
        ax[2].plot(t, bal.TAlim_est, label="TAlimout estimated")
    ax[2].plot(t, bal.Tin, label="Tin = (Tin1 + Tin2)/2")
    ax[2].set_ylabel("[°C]")
    for col, label in (
        ("Q_hx_hot", "HX hot side (pupitre Tout/Tin)"),
        ("Q_hx_hot_bdd", "HX hot side (SUPERVISION ted/tsd)"),
        ("Q_hx_cold", "HX cold side (debitbrut, tsb - teb)"),
        ("P_magnet", "P_magnet"),
        ("Ptot_pupitre", "Ptot (pupitre)"),
    ):
        if bal[col].notna().any():
            ax[3].plot(t, bal[col], label=label)
    ax[3].set_ylabel("power [MW]")
    for a in ax:
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pupitre", type=Path, help="pupitre .txt file of the run")
    ap.add_argument(
        "pigbrother",
        type=Path,
        nargs="+",
        help="pigbrother Overview/Archive .tdms files",
    )
    ap.add_argument(
        "--vt0",
        type=float,
        default=1.0,
        help="thyristor threshold voltage [V] (default: 1.0)",
    )
    ap.add_argument(
        "--rt",
        type=float,
        default=1e-4,
        help="thyristor slope resistance [Ohm] (default: 1e-4)",
    )
    ap.add_argument(
        "--pump-coef",
        type=float,
        default=0.0267,
        help="pump electrical power at 1000 rpm, per pump [MW] (default: 0.0267)",
    )
    ap.add_argument(
        "--conv-flow",
        type=float,
        default=None,
        help="total converter cooling flow [l/s]; enables the TAlimout estimate",
    )
    ap.add_argument(
        "--delay", type=float, default=0.0, help="Tout transport delay [s] (default: 0)"
    )
    ap.add_argument(
        "--tau", type=float, default=0.0, help="Tout time constant [s] (default: 0)"
    )
    ap.add_argument(
        "--talim-tau",
        type=float,
        default=0.0,
        help="TAlimout time constant [s] (default: 0)",
    )
    ap.add_argument(
        "--max-lag",
        type=int,
        default=120,
        help="pupitre/pigbrother time offset search range [s]",
    )
    ap.add_argument(
        "--bdd",
        type=Path,
        help="SUPERVISION bdd.csv export (teb, tsb, ted, tsd); needs duckdb",
    )
    ap.add_argument(
        "--housing",
        default="M9",
        choices=sorted(HOUSING_BASE),
        help="housing, for the supply group -> magnet wiring (default: M9)",
    )
    ap.add_argument(
        "--ptot-includes-pumps",
        action="store_true",
        help="subtract the pump estimate from Ptot to get P_alim",
    )
    ap.add_argument("--csv", type=Path, help="write the 1 s balance to this CSV")
    ap.add_argument("--plot", type=Path, help="save a figure to this file")
    args = ap.parse_args()

    pup = read_pupitre(args.pupitre)
    pb = read_pigbrother(args.pigbrother)
    wiring, wiring_source = housing_wiring(args.housing, pup.index[0])
    for gr in CONVERTERS:  # a group missing from the files is fine only if it is off
        cols = [ch for _, ch in _group_channels(gr, wiring.get(gr))]
        pb = pb.reindex(columns=pb.columns.union(cols))
    p_ref = sum((pb[tap] * pb[f"Courant_{gr}"]).fillna(0) for gr, tap in wiring.items())
    lag = estimate_lag(p_ref / 1e6, pup.Pmagnet, args.max_lag)
    pup.index = pup.index + pd.Timedelta(seconds=lag)

    df = pb.join(pup, how="inner").asfreq("1s")
    for gr in CONVERTERS:
        off = df[list(PUPITRE_CURRENT[gr])].abs().sum(axis=1) < ACTIVE_CURRENT_A
        cols = [ch for _, ch in _group_channels(gr, wiring.get(gr))]
        df.loc[off, cols] = df.loc[off, cols].fillna(0.0)
    if args.bdd:  # same clock as the pupitre: apply the same offset
        shift = pd.Timedelta(seconds=lag)
        sup = read_supervision(args.bdd, df.index[0] - shift, df.index[-1] - shift)
        sup.index = sup.index + pd.Timedelta(seconds=lag)
        if sup.empty:
            print(f"warning: {args.bdd} has no data for this run")
        else:  # ~5 s samples -> 1 s grid, no interpolation across gaps > 15 s
            grid = sup.reindex(sup.index.union(df.index)).interpolate(
                method="time", limit=15, limit_area="inside"
            )
            df = df.join(grid.reindex(df.index))

    bal = energy_balance(
        df,
        wiring,
        args.vt0,
        args.rt,
        args.pump_coef,
        args.conv_flow,
        args.delay,
        args.tau,
        args.talim_tau,
        args.ptot_includes_pumps,
    )

    print(f"run: {args.pupitre.name}")
    print(f"pupitre/pigbrother time offset: {lag:+d} s (added to pupitre time)")
    print(
        f"wiring ({wiring_source}): "
        + ", ".join(f"{gr} -> {tap}" for gr, tap in wiring.items())
    )
    print(
        f"model: V_T0={args.vt0} V, r_T={args.rt} Ohm, pump {args.pump_coef} MW/krpm^3, "
        f"rho*cp={RHO_CP:.3g} J/(m3 K), Tout delay={args.delay} s tau={args.tau} s, "
        f"converter flow={args.conv_flow} l/s, "
        f"Ptot {'includes' if args.ptot_includes_pumps else 'excludes'} pumps"
    )
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    summary = summarize(bal).round(3)
    tables = {
        "power split [MW]": [
            "dur[min]",
            "P_mag",
            "P_pup",
            "P_bus_GR1",
            "P_bus_GR2",
            "P_conv",
            "P_hyd",
            "P_pump_el",
            "Ptot",
            "P_alim",
        ],
        "temperatures [°C]": [
            "Tin",
            "Tout",
            "Tout_est",
            "dTout",
            "k_cal",
            "TAlim",
            "TAlim_est",
            "F_conv_impl[l/s]",
        ],
        "heat exchanger [MW], sensor checks [K]": [
            "P_mag",
            "Ptot",
            "Q_hot",
            "Q_hot_bdd",
            "Q_cold",
            "cold/hot",
            "ted-Tout",
            "tsd-Tin",
        ],
    }
    for title, cols in tables.items():
        print(f"\nsteady plateaus, {title} (medians):")
        print(summary.reindex(columns=cols).to_string())
    energy = (
        bal[
            [
                "P_magnet",
                "P_busbar_GR1",
                "P_busbar_GR2",
                "P_conv",
                "P_hyd",
                "P_pump_el",
                "P_alim",
                "Ptot_pupitre",
                "Q_hx_hot",
                "Q_hx_hot_bdd",
                "Q_hx_cold",
            ]
        ].sum(min_count=1)
        / 3600
    )
    print("\nwhole run energy [MWh]:")
    print(energy.round(3).to_string())

    if args.csv:
        bal.to_csv(args.csv, float_format="%.6g")
    if args.plot:
        plot(bal, args.plot, args.pupitre.name)


if __name__ == "__main__":
    main()
