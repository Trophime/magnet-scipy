#!/usr/bin/env python3
"""
Example usage of the coupled RL circuits simulation

This script demonstrates how to use the coupled RL circuits system
with different configurations and analysis options.

Usage:
    python examples/example_usage.py [EXAMPLE] [--save-dir DIR] [--no-show]

EXAMPLE selects a single example (1-5); without it all examples are run.

Each circuit tracks a piecewise-linear reference current that ramps through
the low, medium and high PID regions. The simulation API loads references
from CSV, so the profiles are written to a temporary directory for the
duration of the run.
"""

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from magnet_scipy import PIDController, RLCircuitPID, create_adaptive_pid_controller
from magnet_scipy.coupled_circuits import CoupledRLCircuitsPID
from magnet_scipy.coupled_plotting import (
    analyze_coupling_effects,
    create_coupling_comparison_plot,
    plot_region_analysis,
)
from magnet_scipy.plotting_core import PlottingManager
from magnet_scipy.plotting_strategies import ProcessedResults
from magnet_scipy.simulation_strategies import SimulationRunner
from magnet_scipy.simulation_types import SimulationParameters, SimulationResult

# Reference profile shared by all circuits: knots as fractions of the
# (delayed) simulation span, and currents (A) chosen to cross the default
# PID thresholds at 60 A and 800 A.
REFERENCE_FRACTIONS = [0.0, 0.1, 0.3, 0.4, 0.6, 0.7, 1.0]
REFERENCE_CURRENTS = [0.0, 50.0, 50.0, 400.0, 400.0, 900.0, 900.0]


@dataclass
class Output:
    """Where and how to display the figures produced by the examples"""

    save_dir: Optional[str] = None
    show: bool = True

    def path(self, name: str) -> Optional[str]:
        return os.path.join(self.save_dir, name) if self.save_dir else None

    def finish(self, fig, name: str):
        """Save and/or show a figure created directly in this script"""
        save_path = self.path(name)
        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        if self.show:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_reference_csv(path: str, t_end: float, delay: float = 0.0) -> str:
    """Write the reference current profile, started after `delay` seconds"""
    times = delay + np.array(REFERENCE_FRACTIONS) * (t_end - delay)
    currents = list(REFERENCE_CURRENTS)
    if delay > 0:
        times = np.concatenate([[0.0], times])
        currents = [0.0] + currents
    pd.DataFrame({"time": times, "current": currents}).to_csv(path, index=False)
    return path


def make_circuit(
    circuit_id: str,
    workdir: str,
    t_end: float,
    R: float = 1.0,
    L: float = 0.1,
    temperature: float = 25.0,
    delay: float = 0.0,
    pid_controller: Optional[PIDController] = None,
) -> RLCircuitPID:
    """Create a PID-controlled circuit tracking the example reference profile"""
    reference_csv = write_reference_csv(
        os.path.join(workdir, f"{circuit_id}_reference.csv"), t_end, delay
    )
    return RLCircuitPID(
        R=R,
        L=L,
        temperature=temperature,
        circuit_id=circuit_id,
        reference_csv=reference_csv,
        pid_controller=pid_controller or create_adaptive_pid_controller(),
    )


def coupling_vector(inductances: Sequence[float], k: float) -> np.ndarray:
    """
    Mutual inductances M_ij = k * sqrt(L_i * L_j) for a uniform coupling
    coefficient k, as the upper-triangle vector CoupledRLCircuitsPID expects.
    The inductance matrix stays positive definite for 0 <= k < 1.
    """
    L = np.asarray(inductances)
    i, j = np.triu_indices(len(L), k=1)
    return k * np.sqrt(L[i] * L[j])


def create_example_coupled_circuits(
    n_circuits: int, coupling_strength: float, workdir: str, t_end: float
) -> CoupledRLCircuitsPID:
    """
    Create n slightly different circuits with staggered references, coupled
    with coefficient `coupling_strength`
    """
    circuits = [
        make_circuit(
            f"circuit_{i + 1}",
            workdir,
            t_end,
            R=1.0 + 0.2 * i,
            L=0.1 + 0.02 * i,
            delay=0.05 * i * t_end,
        )
        for i in range(n_circuits)
    ]
    mutual_inductances = coupling_vector([c.L for c in circuits], coupling_strength)
    return CoupledRLCircuitsPID(circuits, mutual_inductances)


def run_pid_simulation(system, t_end: float, dt: float) -> SimulationResult:
    """Run a PID-control simulation from t=0, with dt as the maximum step"""
    params = SimulationParameters(t_start=0.0, t_end=t_end, dt=dt)
    result = SimulationRunner().run_simulation(system, params, "pid")
    if not result.success:
        raise RuntimeError(f"Simulation failed: {result.error_message}")
    return result


def rms(t: np.ndarray, values: np.ndarray) -> float:
    """Time-weighted RMS (the solver output grid is not uniform)"""
    return float(np.sqrt(trapezoid(values**2, t) / (t[-1] - t[0])))


def resample(results: ProcessedResults, t: np.ndarray) -> Dict[str, Dict]:
    """Interpolate every per-sample array of each circuit onto the time grid t"""
    n_samples = len(results.time)
    return {
        circuit_id: {
            key: np.interp(t, results.time, values)
            for key, values in data.items()
            if isinstance(values, np.ndarray) and values.shape == (n_samples,)
        }
        for circuit_id, data in results.circuits.items()
    }


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def example_1_basic_coupled_system(workdir: str, out: Output):
    """Example 1: Basic 3-circuit coupled system with default settings"""

    print("=== Example 1: Basic Coupled System ===")

    # Create a basic 3-circuit system
    t_end, dt = 5.0, 0.001
    coupled_system = create_example_coupled_circuits(
        n_circuits=3, coupling_strength=0.08, workdir=workdir, t_end=t_end
    )
    coupled_system.print_configuration()

    # Run simulation
    print("Running simulation...")
    result = run_pid_simulation(coupled_system, t_end, dt)

    # Post-process, plot and analyze
    results, analytics = PlottingManager().create_plots(
        result,
        coupled_system,
        "pid_control",
        save_path=out.path("example1_coupled_pid.png"),
        show=out.show,
        show_analytics=True,
    )
    analyze_coupling_effects(coupled_system, results.time, results.circuits)

    return coupled_system, result, results


def example_2_custom_circuits(workdir: str, out: Output):
    """Example 2: Custom circuit configurations with specific PID tuning"""

    print("\n=== Example 2: Custom Circuit Configurations ===")

    t_end, dt = 6.0, 0.001
    circuits = []

    # Circuit 1: High-performance, low inductance
    circuit1 = make_circuit(
        "high_performance",
        workdir,
        t_end,
        R=0.8,
        L=0.05,
        temperature=20.0,
        pid_controller=create_adaptive_pid_controller(
            Kp_low=25.0, Ki_low=18.0, Kd_low=0.12,
            Kp_medium=30.0, Ki_medium=22.0, Kd_medium=0.08,
            Kp_high=35.0, Ki_high=25.0, Kd_high=0.05,
            low_threshold=50.0, high_threshold=300.0,
        ),
    )
    circuits.append(circuit1)

    # Circuit 2: Standard configuration
    circuit2 = make_circuit(
        "standard",
        workdir,
        t_end,
        R=1.2,
        L=0.1,
        temperature=25.0,
        delay=0.3,
        pid_controller=create_adaptive_pid_controller(),  # Default PID
    )
    circuits.append(circuit2)

    # Circuit 3: Heavy-duty, high inductance
    circuit3 = make_circuit(
        "heavy_duty",
        workdir,
        t_end,
        R=1.8,
        L=0.15,
        temperature=35.0,
        delay=0.6,
        pid_controller=create_adaptive_pid_controller(
            Kp_low=8.0, Ki_low=4.0, Kd_low=0.15,
            Kp_medium=12.0, Ki_medium=6.0, Kd_medium=0.10,
            Kp_high=15.0, Ki_high=8.0, Kd_high=0.08,
            low_threshold=80.0, high_threshold=600.0,
        ),
    )
    circuits.append(circuit3)

    # Asymmetric coupling, given as the upper triangle [M12, M13, M23] (H):
    # circuit 1 couples strongly with 2 and weakly with 3,
    # circuit 2 couples strongly with both
    mutual_inductances = np.array([0.06, 0.03, 0.08])

    # Create coupled system
    coupled_system = CoupledRLCircuitsPID(circuits, mutual_inductances)
    coupled_system.print_configuration()

    # Run simulation
    result = run_pid_simulation(coupled_system, t_end, dt)

    # Analysis
    results = PlottingManager().prepare_results(result, coupled_system)
    analyze_coupling_effects(coupled_system, results.time, results.circuits)
    plot_region_analysis(
        coupled_system,
        results.time,
        results.circuits,
        save_path=out.path("example2_region_analysis.png"),
        show=out.show,
    )

    return coupled_system, result, results


def example_3_coupling_strength_study(workdir: str, out: Output):
    """Example 3: Study the effect of the coupling coefficient"""

    print("\n=== Example 3: Coupling Strength Study ===")

    coupling_strengths = [0.0, 0.02, 0.05, 0.1, 0.2]
    t_end, dt = 4.0, 0.002
    manager = PlottingManager()
    results_list = []

    for coupling in coupling_strengths:
        print(f"\nSimulating with coupling coefficient: {coupling}")

        # Create system with specific coupling
        coupled_system = create_example_coupled_circuits(
            n_circuits=2, coupling_strength=coupling, workdir=workdir, t_end=t_end
        )

        # Quick simulation
        result = run_pid_simulation(coupled_system, t_end, dt)
        results = manager.prepare_results(result, coupled_system)
        results_list.append((coupling, results))

        # Calculate performance metrics
        t = results.time
        rms1 = rms(t, results.circuits["circuit_1"]["error"])
        rms2 = rms(t, results.circuits["circuit_2"]["error"])

        print(f"  Circuit 1 RMS error: {rms1:.4f} A")
        print(f"  Circuit 2 RMS error: {rms2:.4f} A")

    # Plot comparison for different coupling strengths
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(coupling_strengths)))

    for color, (coupling, results) in zip(colors, results_list):
        t = results.time
        label = f"k = {coupling}"
        for col, circuit_id in enumerate(["circuit_1", "circuit_2"]):
            data = results.circuits[circuit_id]
            axes[0, col].plot(t, data["current"], color=color, label=label, linewidth=1.5)
            axes[1, col].plot(
                t, np.abs(data["error"]), color=color, label=label, linewidth=1.5
            )

    for col, name in enumerate(["Circuit 1", "Circuit 2"]):
        axes[0, col].set_title(f"{name} Current")
        axes[0, col].set_ylabel("Current (A)")
        axes[1, col].set_title(f"{name} |Error|")
        axes[1, col].set_ylabel("|Error| (A)")
        axes[1, col].set_xlabel("Time (s)")
        for row in range(2):
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    out.finish(fig, "example3_coupling_study.png")

    return results_list


def example_4_uncoupled_vs_coupled(workdir: str, out: Output):
    """Example 4: Compare uncoupled vs coupled behavior"""

    print("\n=== Example 4: Uncoupled vs Coupled Comparison ===")

    # Same simulation parameters for both
    t_end, dt = 5.0, 0.001

    # Create two identical systems - one with coupling, one without
    coupled_system = create_example_coupled_circuits(
        n_circuits=2, coupling_strength=0.1, workdir=workdir, t_end=t_end
    )
    uncoupled_system = create_example_coupled_circuits(
        n_circuits=2, coupling_strength=0.0, workdir=workdir, t_end=t_end
    )

    print("Running coupled simulation...")
    result_coupled = run_pid_simulation(coupled_system, t_end, dt)

    print("Running uncoupled simulation...")
    result_uncoupled = run_pid_simulation(uncoupled_system, t_end, dt)

    # Post-process both results onto a common time grid (the adaptive
    # solver produces a different grid for each run)
    manager = PlottingManager()
    t = np.linspace(0.0, t_end, int(round(t_end / dt)) + 1)
    results_coupled = resample(manager.prepare_results(result_coupled, coupled_system), t)
    results_uncoupled = resample(
        manager.prepare_results(result_uncoupled, uncoupled_system), t
    )

    # Create comparison plot
    circuit_ids = coupled_system.circuit_ids
    create_coupling_comparison_plot(
        results_uncoupled,
        results_coupled,
        t,
        circuit_ids,
        save_path=out.path("example4_coupled_vs_uncoupled.png"),
        show=out.show,
    )

    # Performance comparison
    print("\nPerformance Comparison:")
    for circuit_id in circuit_ids:
        uncoupled_rms = rms(t, results_uncoupled[circuit_id]["error"])
        coupled_rms = rms(t, results_coupled[circuit_id]["error"])

        print(f"  {circuit_id}:")
        print(f"    Uncoupled RMS error: {uncoupled_rms:.4f} A")
        print(f"    Coupled RMS error: {coupled_rms:.4f} A")
        print(
            f"    Relative change: {((coupled_rms - uncoupled_rms) / uncoupled_rms * 100):+.1f}%"
        )

    return (
        (coupled_system, result_coupled, results_coupled),
        (uncoupled_system, result_uncoupled, results_uncoupled),
    )


def example_5_scalability_test(workdir: str, out: Output):
    """Example 5: Test scalability with more circuits"""

    print("\n=== Example 5: Scalability Test ===")

    n_circuits_list = [2, 3, 5, 8]
    t_end, dt = 2.0, 0.005  # Shorter simulation for speed
    manager = PlottingManager()

    for n_circuits in n_circuits_list:
        print(f"\nTesting {n_circuits} circuits...")

        # Create system
        coupled_system = create_example_coupled_circuits(
            n_circuits=n_circuits, coupling_strength=0.05, workdir=workdir, t_end=t_end
        )

        print(f"  State vector size: {2 * n_circuits}")
        print(f"  Mutual inductance matrix: {coupled_system.M.shape}")

        start_time = time.perf_counter()
        result = run_pid_simulation(coupled_system, t_end, dt)
        simulation_time = time.perf_counter() - start_time

        print(f"  Simulation time: {simulation_time:.2f} seconds")
        print(f"  Time steps: {len(result.time)}")

        # Calculate average RMS error across all circuits
        results = manager.prepare_results(result, coupled_system)
        avg_rms = np.mean(
            [
                rms(results.time, results.circuits[circuit_id]["error"])
                for circuit_id in coupled_system.circuit_ids
            ]
        )
        print(f"  Average RMS error: {avg_rms:.4f} A")


EXAMPLES = {
    1: example_1_basic_coupled_system,
    2: example_2_custom_circuits,
    3: example_3_coupling_strength_study,
    4: example_4_uncoupled_vs_coupled,
    5: example_5_scalability_test,
}


def run_all_examples(workdir: str, out: Output) -> int:
    """Run all examples in sequence"""

    print("Running all coupled RL circuit examples...\n")

    try:
        for example in EXAMPLES.values():
            example(workdir, out)

        print("\n✓ All examples completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Error in examples: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "example",
        nargs="?",
        type=int,
        choices=sorted(EXAMPLES),
        help="Example to run (default: all)",
    )
    parser.add_argument("--save-dir", help="Directory where plots are saved")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plots interactively"
    )
    args = parser.parse_args(argv)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    out = Output(save_dir=args.save_dir, show=not args.no_show)

    with tempfile.TemporaryDirectory(prefix="magnet_scipy_example_") as workdir:
        if args.example is None:
            return run_all_examples(workdir, out)
        EXAMPLES[args.example](workdir, out)
        return 0


if __name__ == "__main__":
    sys.exit(main())
