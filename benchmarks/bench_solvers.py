"""Benchmark the scipy ODE solvers on the ``examples/`` cases.

For every (case, method, rtol) combination the script reports:

* cost: wall-clock and CPU time [s], RHS evaluations, Jacobian evaluations,
  LU decompositions, accepted steps;
* memory: peak RSS increase during the solve [MB] and, optionally, the
  Python-level peak traced by :mod:`tracemalloc` [MB];
* precision against the measurements (model + numerical error): RMSE,
  range-normalised RMSE, MAE, max absolute error and bias;
* precision against a tight-tolerance reference solution (numerical error
  only): RMSE and max absolute error.

Each run executes in a freshly spawned process so that timings and peak RSS
are not polluted by earlier runs, and so that a run can be killed on timeout.

Usage
-----
::

    python benchmarks/bench_solvers.py --max-duration 300
    python benchmarks/bench_solvers.py --cases M9_coupled_pid --methods BDF Radau LSODA \\
        --rtol 1e-4 1e-6 1e-8 --repeat 3
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import queue
import resource
import sys
import time
import traceback
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CASES = Path(__file__).with_name("cases.json")
DEFAULT_OUTPUT = Path(__file__).with_name("results")
ALL_METHODS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]


def error_metrics(simulated: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """Compute pointwise error metrics between two sampled signals.

    Parameters
    ----------
    simulated : numpy.ndarray
        Simulated values on the comparison time grid.
    measured : numpy.ndarray
        Reference values (experiment or reference solution) on the same grid.

    Returns
    -------
    dict of str to float
        ``rmse``, ``nrmse`` (RMSE divided by the full scale ``max|measured|``;
        NaN if it is zero), ``mae``, ``max_abs_error`` and ``bias`` (mean of
        ``simulated - measured``), in the unit of the signal ([A] or [V])
        except ``nrmse`` which is dimensionless.
    """
    err = np.asarray(simulated, dtype=float) - np.asarray(measured, dtype=float)
    scale = float(np.max(np.abs(measured))) if len(measured) else 0.0
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else np.nan
    return {
        "rmse": rmse,
        "nrmse": rmse / scale if scale > 0 else np.nan,
        "mae": float(np.mean(np.abs(err))) if len(err) else np.nan,
        "max_abs_error": float(np.max(np.abs(err))) if len(err) else np.nan,
        "bias": float(np.mean(err)) if len(err) else np.nan,
    }


def _maxrss_mb() -> float:
    """Peak resident set size of the current process [MB]."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


def _load_system(case: dict[str, Any]):
    """Build the single or coupled circuit system described by ``case``."""
    from magnet_scipy.cli_core import ConfigurationLoader
    from magnet_scipy.coupled_circuits import CoupledRLCircuitsPID

    if case["kind"] == "single":
        return ConfigurationLoader.load_single_circuit(case["config"])
    circuits, mutual_inductances = ConfigurationLoader.load_coupled_circuits(
        case["config"]
    )
    return CoupledRLCircuitsPID(circuits, mutual_inductances=mutual_inductances)


def _pid_voltage(system, k: int, te: np.ndarray, t: np.ndarray, y: np.ndarray):
    """Controller output u = Kp e + Ki ∫e + Kd ė [V] of circuit ``k`` at times ``te``."""
    coupled = hasattr(system, "circuits")
    circuit = system.circuits[k] if coupled else system
    n = system.n_circuits if coupled else 1
    ys = np.array([np.interp(te, t, row) for row in y])
    u = np.empty(len(te))
    for j, tj in enumerate(te):
        if coupled:
            di_ref = np.array(system.get_reference_current_derivatives(tj))
            di_dt = system.vector_field(tj, ys[:, j], di_ref_dt=di_ref)[k]
        else:
            di_ref = np.array([system.reference_current_derivative(tj)])
            di_dt = system.vector_field(tj, ys[:, j], di_ref_dt=di_ref[0])[0]
        i_ref = float(circuit.reference_current(tj))
        kp, ki, kd = circuit.get_pid_parameters(i_ref)
        error = i_ref - ys[k, j]
        u[j] = kp * error + ki * ys[n + k, j] + kd * (di_ref[k] - di_dt)
    return u


def _sample(system, strategy: str, t: np.ndarray, y: np.ndarray) -> list[dict]:
    """Sample the solution on each experimental time grid within the solved window."""
    circuits = system.circuits if hasattr(system, "circuits") else [system]
    samples = []
    for k, circuit in enumerate(circuits):
        circuit._load_experimental_data()
        for exp in circuit.experimental_data:
            quantity = exp["type"]
            # In voltage mode the voltage is the model input: nothing to compare.
            if quantity == "voltage" and strategy != "pid":
                continue
            if quantity not in ("current", "voltage"):
                continue
            info = circuit.experimental_functions[f'{exp["type"]}_{exp["key"]}']
            te = np.asarray(info["time_data"], dtype=float)
            ve = np.asarray(info["values_data"], dtype=float)
            mask = (te >= t[0]) & (te <= t[-1])
            te, ve = te[mask], ve[mask]
            if quantity == "current":
                simulated = np.interp(te, t, y[k])
            else:
                simulated = _pid_voltage(system, k, te, t, y)
            samples.append(
                {
                    "circuit": circuit.circuit_id,
                    "quantity": quantity,
                    "time": te.tolist(),
                    "simulated": simulated.tolist(),
                    "measured": ve.tolist(),
                }
            )
    return samples


def _worker(case, method, rtol, atol, max_step, use_tracemalloc, out_queue):
    """Load ``case``, solve it once with ``method`` and report cost and samples."""
    try:
        from magnet_scipy.simulation_types import SimulationParameters
        from magnet_scipy.single_circuit_adapter import create_simulation_runner

        os.chdir(REPO_ROOT / case["wd"])
        with contextlib.redirect_stdout(io.StringIO()):
            system = _load_system(case)
            runner = create_simulation_runner(system)
            strategy = runner.detect_strategy(system)
            params = SimulationParameters(
                t_start=case["t_start"],
                t_end=case["t_end"],
                dt=max_step,
                method=method,
                rtol=rtol,
                atol=atol,
                initial_values=case["initial_values"],
            )

            if use_tracemalloc:
                tracemalloc.start()
            rss_before = _maxrss_mb()
            wall0, cpu0 = time.perf_counter(), time.process_time()
            result = runner.run_simulation(system, params, strategy)
            wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
            rss_after = _maxrss_mb()
            traced_peak = np.nan
            if use_tracemalloc:
                traced_peak = tracemalloc.get_traced_memory()[1] / 1024**2
                tracemalloc.stop()

            samples = []
            if result.success:
                samples = _sample(system, strategy, result.time, result.solution)

        meta = result.metadata
        out_queue.put(
            {
                "status": "ok" if result.success else "failed",
                "message": result.error_message or "",
                "strategy": strategy,
                "wall_s": wall,
                "cpu_s": cpu,
                "nfev": meta.get("n_evaluations"),
                "njev": meta.get("n_jacobian_evaluations"),
                "nlu": meta.get("n_lu_decompositions"),
                "n_steps": max(len(result.time) - 1, 0),
                "rss_delta_mb": rss_after - rss_before,
                "tracemalloc_peak_mb": traced_peak,
                "samples": samples,
            }
        )
    except Exception:
        out_queue.put({"status": "error", "message": traceback.format_exc(limit=4)})


def run_isolated(
    case: dict[str, Any],
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
    timeout: float,
    use_tracemalloc: bool = False,
) -> dict[str, Any]:
    """Run one solve in a freshly spawned process.

    Parameters
    ----------
    case : dict
        Case description (see ``cases.json``), already truncated if requested.
    method : str
        :func:`scipy.integrate.solve_ivp` method name.
    rtol, atol : float
        Solver relative and absolute tolerances.
    max_step : float
        Maximum solver step [s]; ``numpy.inf`` for no limit.
    timeout : float
        Wall-clock limit for the whole child process [s].
    use_tracemalloc : bool, default False
        Trace Python allocations during the solve (slows it down).

    Returns
    -------
    dict
        Worker report; ``status`` is ``"ok"``, ``"failed"`` (solver gave up),
        ``"error"`` (exception) or ``"timeout"``.
    """
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(case, method, rtol, atol, max_step, use_tracemalloc, out_queue),
    )
    proc.start()
    deadline = time.monotonic() + timeout
    report: dict[str, Any] | None = None
    while report is None:
        try:
            report = out_queue.get(timeout=1.0)
        except queue.Empty:
            if not proc.is_alive():
                try:
                    report = out_queue.get(timeout=1.0)
                except queue.Empty:
                    report = {
                        "status": "error",
                        "message": f"worker exited with code {proc.exitcode}",
                    }
            elif time.monotonic() > deadline:
                proc.terminate()
                report = {"status": "timeout", "message": f"> {timeout:.0f} s"}
    proc.join()
    return report


def _reference_samples(
    case_name: str,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict] | None:
    """Return (and cache on disk) the reference-solution samples for a case."""
    config_bytes = (REPO_ROOT / case["wd"] / case["config"]).read_bytes()
    key = json.dumps(
        [case, args.reference_method, args.reference_rtol, args.reference_atol],
        sort_keys=True,
    ).encode() + config_bytes
    digest = hashlib.sha1(key).hexdigest()[:12]
    cache = args.output_dir / "references" / f"{case_name}-{digest}.json"
    if cache.exists():
        print(f"  reference: cached ({cache.name})")
        return json.loads(cache.read_text())

    print(
        f"  reference: {args.reference_method} rtol={args.reference_rtol:g} "
        f"atol={args.reference_atol:g} ...",
        flush=True,
    )
    report = run_isolated(
        case,
        args.reference_method,
        args.reference_rtol,
        args.reference_atol,
        case["max_step"],
        args.timeout,
    )
    if report["status"] != "ok":
        print(f"  reference {report['status']}: {report['message'].strip()}")
        return None
    print(f"  reference: done in {report['wall_s']:.1f} s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(report["samples"]))
    return report["samples"]


def _accuracy_rows(samples: list[dict], reference: list[dict] | None) -> list[dict]:
    """Build accuracy rows (vs experiment and vs reference) for one run."""
    ref_by_key = {(s["circuit"], s["quantity"]): s for s in reference or []}
    rows = []
    for s in samples:
        row = {"circuit": s["circuit"], "quantity": s["quantity"]}
        row["n_points"] = len(s["time"])
        exp = error_metrics(np.array(s["simulated"]), np.array(s["measured"]))
        row.update({f"{k}_exp": v for k, v in exp.items()})
        ref = ref_by_key.get((s["circuit"], s["quantity"]))
        if ref is not None and len(ref["simulated"]) == len(s["simulated"]):
            num = error_metrics(np.array(s["simulated"]), np.array(ref["simulated"]))
            row.update({"rmse_ref": num["rmse"], "max_abs_error_ref": num["max_abs_error"]})
        else:
            row.update({"rmse_ref": np.nan, "max_abs_error_ref": np.nan})
        rows.append(row)
    return rows


def _environment(args: argparse.Namespace) -> dict[str, Any]:
    """Collect machine and library versions for the results header."""
    import scipy

    cpu = platform.processor()
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "args": {k: str(v) for k, v in vars(args).items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--cases", nargs="+", help="Subset of case names to run")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--rtol", nargs="+", type=float, default=[1e-6])
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument(
        "--max-step",
        type=float,
        help="Override every case's max_step [s] ('inf' for no limit)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Truncate every case to t_start + MAX_DURATION [s]",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Timed runs per setting")
    parser.add_argument(
        "--tracemalloc",
        action="store_true",
        help="Add one extra run per setting to measure the tracemalloc peak",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="Per run [s]")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--reference-method", default="Radau", choices=ALL_METHODS)
    parser.add_argument("--reference-rtol", type=float, default=1e-10)
    parser.add_argument("--reference-atol", type=float, default=1e-10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write ``performance.csv`` and ``accuracy.csv``.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    # Keep BLAS single-threaded so CPU time and wall time stay comparable.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    all_cases = json.loads(args.cases_file.read_text())
    names = args.cases or list(all_cases)
    unknown = sorted(set(names) - set(all_cases))
    if unknown:
        print(f"Unknown cases: {unknown}. Available: {sorted(all_cases)}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    perf_rows, acc_rows = [], []

    for name in names:
        case = dict(all_cases[name])
        if args.max_duration is not None:
            case["t_end"] = min(case["t_end"], case["t_start"] + args.max_duration)
        max_step = case["max_step"] if args.max_step is None else args.max_step
        print(f"\n=== {name}: t=[{case['t_start']}, {case['t_end']}] s, max_step={max_step}")

        reference = None if args.no_reference else _reference_samples(name, case, args)

        for method in args.methods:
            for rtol in args.rtol:
                label = f"  {method:<6} rtol={rtol:<8g}"
                reports = []
                for _ in range(args.repeat):
                    report = run_isolated(case, method, rtol, args.atol, max_step, args.timeout)
                    reports.append(report)
                    if report["status"] != "ok":
                        break
                ok = [r for r in reports if r["status"] == "ok"]
                last = reports[-1]
                setting = {
                    "case": name,
                    "method": method,
                    "rtol": rtol,
                    "atol": args.atol,
                    "max_step": max_step,
                    "t_start": case["t_start"],
                    "t_end": case["t_end"],
                }
                row = {**setting, "status": last["status"], "message": last["message"].strip()}
                if ok:
                    walls = [r["wall_s"] for r in ok]
                    row.update(
                        {
                            "strategy": ok[0]["strategy"],
                            "repeats": len(ok),
                            "wall_s_median": float(np.median(walls)),
                            "wall_s_min": float(np.min(walls)),
                            "cpu_s_median": float(np.median([r["cpu_s"] for r in ok])),
                            "nfev": ok[0]["nfev"],
                            "njev": ok[0]["njev"],
                            "nlu": ok[0]["nlu"],
                            "n_steps": ok[0]["n_steps"],
                            "rss_delta_mb": max(r["rss_delta_mb"] for r in ok),
                            "tracemalloc_peak_mb": np.nan,
                        }
                    )
                    if args.tracemalloc:
                        mem = run_isolated(
                            case, method, rtol, args.atol, max_step, args.timeout, True
                        )
                        if mem["status"] == "ok":
                            row["tracemalloc_peak_mb"] = mem["tracemalloc_peak_mb"]
                    for acc in _accuracy_rows(ok[0]["samples"], reference):
                        acc_rows.append({**setting, **acc})
                    print(
                        f"{label} {row['wall_s_median']:9.2f} s  nfev={row['nfev']:<9} "
                        f"steps={row['n_steps']:<9} dRSS={row['rss_delta_mb']:8.1f} MB",
                        flush=True,
                    )
                else:
                    print(f"{label} {last['status']}: {row['message'].splitlines()[-1:]}")
                perf_rows.append(row)

    perf = pd.DataFrame(perf_rows)
    acc = pd.DataFrame(acc_rows)
    perf.to_csv(args.output_dir / "performance.csv", index=False)
    acc.to_csv(args.output_dir / "accuracy.csv", index=False)
    (args.output_dir / "environment.json").write_text(
        json.dumps(_environment(args), indent=2)
    )

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\n=== Performance ===")
        cols = ["case", "method", "rtol", "status", "wall_s_median", "cpu_s_median",
                "nfev", "njev", "nlu", "n_steps", "rss_delta_mb", "tracemalloc_peak_mb"]
        print(perf.reindex(columns=cols).to_string(index=False, float_format="%.3g"))
        if not acc.empty:
            print("\n=== Accuracy (exp: vs measurements, ref: vs reference solution) ===")
            cols = ["case", "method", "rtol", "circuit", "quantity",
                    "rmse_exp", "nrmse_exp", "max_abs_error_exp", "bias_exp",
                    "rmse_ref", "max_abs_error_ref"]
            print(acc.reindex(columns=cols).to_string(index=False, float_format="%.3g"))
    print(f"\nResults written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
