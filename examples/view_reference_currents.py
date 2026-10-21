#!/usr/bin/env python3
"""Plot the current data referenced by a circuits JSON file.

Usage:
    python examples/view_reference_currents.py examples/M9_M23072801/circuits.json
    python examples/view_reference_currents.py circuits.json --save out.png --no-show
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def collect_current_files(config: dict, base: Path):
    """Yield (circuit_id, key, path) for each current entry of each circuit."""
    for circuit in config.get("circuits", [config]):
        cid = circuit.get("circuit_id", "circuit")
        for entry in circuit.get("experiment_data", []):
            if entry.get("type") == "current":
                yield cid, entry.get("key", "current"), base / entry["file"]


def find_sequences(t, y, tol: float, min_duration: float):
    """Return (t_start, t_end, level) of each plateau: a run where |y - y_start| <= tol."""
    t, y = list(t), list(y)
    seqs, i, n = [], 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and abs(y[j + 1] - y[i]) <= tol:
            j += 1
        if t[j] - t[i] >= min_duration:
            seqs.append((t[i], t[j], sum(y[i : j + 1]) / (j - i + 1)))
        i = j + 1
    return seqs


def find_reference_file(path: Path):
    """Locate the '<prefix>-Courants_Alimentations_Référence_GRn.csv' matching a measured file."""
    m = re.match(r"(.+?)[-_]Courants_Alimentations_Courant_(GR\d+)", path.name)
    if not m:
        return None
    ref = path.with_name(f"{m.group(1)}-Courants_Alimentations_Référence_{m.group(2)}.csv")
    return ref if ref.is_file() else None


def read_csv(path: Path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    tcol = next((c for c in df.columns if c.lower() == "time"), df.columns[0])
    ycol = next((c for c in df.columns if c != tcol), None)
    return df, tcol, ycol


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("json_file", type=Path)
    p.add_argument("--data-dir", type=Path, help="CSV directory (default: JSON file directory)")
    p.add_argument("--plateaus", action="store_true",
                   help="Detect plateaus (sequences) and mark each with a dashed line and label")
    p.add_argument("--tol", type=float, default=0.5, help="Plateau tolerance in A (default 0.5)")
    p.add_argument("--min-duration", type=float, default=20.0,
                   help="Minimum plateau duration in s to count as a sequence (default 20)")
    p.add_argument("--save", type=Path, help="Save figure to this file")
    p.add_argument("--no-show", action="store_true", help="Do not open a window")
    args = p.parse_args()

    config = json.loads(args.json_file.read_text())
    base = args.data_dir or args.json_file.parent

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0
    for cid, key, path in collect_current_files(config, base):
        if not path.is_file():
            print(f"warning: {path} not found, skipping", file=sys.stderr)
            continue
        df, tcol, ycol = read_csv(path)
        if ycol is None:
            print(f"warning: {path} has a single column, skipping", file=sys.stderr)
            continue
        (line,) = ax.plot(df[tcol], df[ycol], label=f"{cid} ({path.name})")
        ref = find_reference_file(path)
        if ref is None:
            print(f"warning: no reference file found for {path.name}, no vlines", file=sys.stderr)
        else:
            rdf, rt, ry = read_csv(ref)
            ax.plot(rdf[rt], rdf[ry], "o", ms=3, color=line.get_color())
            for t_ref in rdf[rt].unique():
                ax.axvline(t_ref, color=line.get_color(), linestyle=":", linewidth=0.5, alpha=0.6)
        if args.plateaus:
            for t0, t1, level in find_sequences(df[tcol], df[ycol], args.tol, args.min_duration):
                ax.hlines(level, t0, t1, colors=line.get_color(), linestyles="--", linewidth=1)
                ax.annotate(f"{level:.0f}", (t0, level), fontsize=7, color=line.get_color())
        plotted += 1

    if not plotted:
        print("error: no current data could be plotted", file=sys.stderr)
        return 1

    ax.set(xlabel="time [s]", ylabel="current [A]", title=config.get("name", "Reference currents"))
    ax.grid(True)
    ax.legend(fontsize="small")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
