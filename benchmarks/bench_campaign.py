"""Benchmark the ODE solvers on every run of a test campaign.

A campaign is the set of ``examples/<prefix>_C*`` directories written by
``examples/build_M23072801_campaign.py``. Each directory holds a
``provenance.json`` that lists its runs with their config, initial currents,
time window and max step. This script turns every run into a
``bench_solvers.py`` case (named ``<group>_<tag>``, e.g. ``C0_230916-175522``),
writes them to ``<output-dir>/cases.json``, runs
:func:`bench_solvers.main` on them, and adds a ``summary.md`` with one row
per run and one column per method.

Options this script does not know are forwarded to ``bench_solvers.py``
(``--rtol``, ``--repeat``, ``--max-duration``, ``--timeout``,
``--no-reference``, ...). Only the implicit methods run by default: the
long campaign runs (up to 13 h of data at ``max_step`` = 0.1 s) are out of
reach of the explicit ones.

Usage
-----
::

    python benchmarks/bench_campaign.py --list
    python benchmarks/bench_campaign.py --groups C0 C2 --max-duration 600
    python benchmarks/bench_campaign.py --runs 230916-175522 --methods Radau BDF LSODA RK45
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import bench_solvers

REPO_ROOT = bench_solvers.REPO_ROOT
DEFAULT_PREFIX = "M9_M23072801"
DEFAULT_METHODS = ["Radau", "BDF", "LSODA"]


def collect_cases(prefix: str = DEFAULT_PREFIX) -> dict[str, dict[str, Any]]:
    """Build ``bench_solvers`` cases from the campaign ``provenance.json`` files.

    Parameters
    ----------
    prefix : str, default ``"M9_M23072801"``
        Campaign directories are ``examples/<prefix>_C*``.

    Returns
    -------
    dict of str to dict
        Cases keyed ``<group>_<tag>``, in the format of ``cases.json``, with
        the extra keys ``group`` and ``tag``. ``t_start`` is 0 [s], ``t_end``
        and ``max_step`` [s] come from the run's ``time_end`` and
        ``time_step``, ``initial_values`` [A] from ``value_start``.
    """
    cases = {}
    for provenance in sorted(REPO_ROOT.glob(f"examples/{prefix}_C*/provenance.json")):
        info = json.loads(provenance.read_text())
        wd = provenance.parent.relative_to(REPO_ROOT).as_posix()
        for run in sorted(info["runs"].values(), key=lambda r: r["tag"]):
            initial_values = [float(v) for v in run["value_start"]]
            cases[f"{info['group']}_{run['tag']}"] = {
                "group": info["group"],
                "tag": run["tag"],
                "wd": wd,
                "config": run["config"],
                "kind": "single" if len(initial_values) == 1 else "coupled",
                "initial_values": initial_values,
                "t_start": 0.0,
                "t_end": float(run["time_end"]),
                "max_step": float(run["time_step"]),
            }
    return cases


def write_summary(output_dir: Path, cases: dict[str, dict[str, Any]]) -> Path:
    """Write ``summary.md``: wall time, current and voltage RMSE per run and method.

    Parameters
    ----------
    output_dir : Path
        Directory holding the ``performance.csv`` and ``accuracy.csv`` written
        by ``bench_solvers.py``.
    cases : dict of str to dict
        Cases that were run, as returned by :func:`collect_cases`.

    Returns
    -------
    Path
        Path of the written ``summary.md``.
    """
    perf = pd.read_csv(output_dir / "performance.csv")
    index = ["case", "rtol"]
    wall = perf.copy()
    # Show the failure status instead of a blank cell.
    wall["wall_s"] = wall["wall_s_median"].map("{:.1f}".format).where(
        wall["status"] == "ok", wall["status"]
    )
    tables = [
        ("Wall time of the solve [s]", wall.pivot_table(
            index=index, columns="method", values="wall_s", aggfunc="first")),
        ("Accepted steps", perf.pivot_table(
            index=index, columns="method", values="n_steps", aggfunc="first"
        ).astype("Int64")),
    ]

    acc_path = output_dir / "accuracy.csv"
    if acc_path.stat().st_size > 1:
        acc = pd.read_csv(acc_path)
        for quantity, unit in [("current", "A"), ("voltage", "V")]:
            rows = acc[acc["quantity"] == quantity]
            for column, versus in [
                ("rmse_exp", "measurements"),
                ("rmse_ref", "reference solution"),
            ]:
                tables.append((
                    f"{quantity.capitalize()} RMSE vs {versus} [{unit}]",
                    rows.pivot_table(index=[*index, "circuit"], columns="method",
                                     values=column, aggfunc="first"),
                ))

    lines = ["# Solver benchmark on the test campaign", "", "## Runs", ""]
    runs = pd.DataFrame(
        [{"case": name, **{k: c[k] for k in ("wd", "config", "kind", "t_end", "max_step")}}
         for name, c in cases.items()]
    )
    lines += [runs.to_markdown(index=False), ""]
    for title, table in tables:
        lines += [f"## {title}", "",
                  table.reset_index().to_markdown(index=False, floatfmt=".3g"), ""]
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse this script's options and keep the rest for ``bench_solvers.py``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Options of this script.
    list of str
        Remaining arguments, forwarded to ``bench_solvers.py``.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Other options are forwarded to bench_solvers.py (see its --help).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help="Campaign directories are examples/<prefix>_C*")
    parser.add_argument("--groups", nargs="+", help="Only these groups, e.g. C0 C2")
    parser.add_argument("--runs", nargs="+", help="Only these run tags, e.g. 230916-175522")
    parser.add_argument("--max-t-end", type=float,
                        help="Skip runs whose window is longer than this [s]")
    parser.add_argument("--list", action="store_true", help="List the runs and exit")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("results") / "campaign")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run ``bench_solvers.py`` on the selected campaign runs.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args, forwarded = parse_args(argv)
    cases = collect_cases(args.prefix)
    if args.groups:
        cases = {k: c for k, c in cases.items() if c["group"] in args.groups}
    if args.runs:
        cases = {k: c for k, c in cases.items() if c["tag"] in args.runs}
    if args.max_t_end is not None:
        cases = {k: c for k, c in cases.items() if c["t_end"] <= args.max_t_end}
    if not cases:
        print(f"No campaign run selected under examples/{args.prefix}_C*")
        return 1

    if args.list:
        for name, c in cases.items():
            print(f"{name:<20} {c['kind']:<8} t_end={c['t_end']:>8g} s  "
                  f"max_step={c['max_step']:g} s  {c['wd']}/{c['config']}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_file = args.output_dir / "cases.json"
    cases_file.write_text(json.dumps(cases, indent=2))

    # Later options win in argparse, so a forwarded --methods replaces the default.
    status = bench_solvers.main(
        ["--cases-file", str(cases_file), "--output-dir", str(args.output_dir),
         "--methods", *DEFAULT_METHODS, *forwarded]
    )
    if status == 0:
        print(f"Summary written to {write_summary(args.output_dir, cases)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
