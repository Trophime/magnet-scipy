"""
magnet_scipy/coupled_plotting.py

Refactored coupled circuits plotting module using strategy pattern
Maintains backward compatibility while eliminating complex mode switching
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from .plotting_core import PlottingManager, PlottingAnalytics, ResultsFileManager
from .plotting_strategies import ProcessedResults


# Backward compatible functions for coupled circuits
def prepare_coupled_post(
    sol, coupled_system, mode: str = "regular"
) -> Tuple[np.ndarray, Dict]:
    """
    Prepare coupled circuit simulation results for plotting

    Args:
        sol: Solution object from scipy.integrate.solve_ivp
        coupled_system: CoupledRLCircuitsPID instance
        mode: "regular" for voltage simulation, "cde" for PID control

    Returns:
        Tuple of (time_array, results_dict)
    """
    manager = PlottingManager()
    strategy_type = "voltage_input" if mode == "regular" else "pid_control"
    results, _ = manager.create_plots(
        sol, coupled_system, strategy_type, show=False, show_analytics=False
    )

    return results.time, results.circuits


def plot_coupled_vresults(
    sol,
    coupled_system,
    t: np.ndarray,
    results: Dict,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot voltage simulation results for coupled circuits

    Backward compatible wrapper that uses the new plotting system
    """
    manager = PlottingManager()
    manager.create_plots(
        sol, coupled_system, "voltage_input", save_path, show, show_analytics=False
    )


def plot_coupled_results(
    sol,
    coupled_system,
    t: np.ndarray,
    results: Dict,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot PID control results for coupled circuits

    Backward compatible wrapper that uses the new plotting system
    """
    manager = PlottingManager()
    manager.create_plots(
        sol, coupled_system, "pid_control", save_path, show, show_analytics=False
    )


def analyze_coupling_effects(coupled_system, t: np.ndarray, results: Dict):
    """
    Analyze coupling effects in coupled circuit system

    Enhanced version with more detailed coupling analysis
    """
    circuit_ids = coupled_system.circuit_ids
    n_circuits = len(circuit_ids)

    print("\n=== Coupling Effects Analysis ===")

    # Current statistics and cross-correlation analysis
    print("\nCurrent Statistics and Coupling Analysis:")
    current_matrix = np.array([results[cid]["current"] for cid in circuit_ids])

    for i, circuit_id in enumerate(circuit_ids):
        current = results[circuit_id]["current"]
        print(f"  {circuit_id}:")
        print(f"    Max current: {float(np.max(np.abs(current))):.3f} A")
        print(f"    RMS current: {float(np.sqrt(np.mean(current**2))):.3f} A")
        print(f"    Current variation (std): {float(np.std(current)):.3f} A")

    # Cross-correlation analysis
    print("\nCross-Correlation Analysis:")
    correlation_matrix = np.corrcoef(current_matrix)

    for i, circuit_i in enumerate(circuit_ids):
        for j, circuit_j in enumerate(circuit_ids):
            if i < j:  # Only upper triangle
                corr = correlation_matrix[i, j]
                coupling_strength = coupled_system.M[i, j]
                print(f"  {circuit_i} ↔ {circuit_j}:")
                print(f"    Mutual inductance: {coupling_strength:.6f} H")
                print(f"    Current correlation: {corr:.3f}")

                # Coupling effectiveness
                if abs(coupling_strength) > 1e-6:
                    effectiveness = abs(corr) / abs(coupling_strength)
                    print(f"    Coupling effectiveness: {effectiveness:.2e}")

    # Error analysis (for PID systems)
    has_errors = all("error" in results[cid] for cid in circuit_ids)
    if has_errors:
        print("\nTracking Performance:")
        total_rms_error = 0.0
        for circuit_id in circuit_ids:
            error = results[circuit_id]["error"]
            rms_error = float(np.sqrt(np.mean(error**2)))
            max_error = float(np.max(np.abs(error)))
            total_rms_error += rms_error**2

            print(f"  {circuit_id}:")
            print(f"    RMS error: {rms_error:.4f} A")
            print(f"    Max error: {max_error:.4f} A")

            # Analyze error coupling
            error_std = float(np.std(error))
            error_trend = np.polyfit(t, error, 1)[0]  # Linear trend
            print(f"    Error variability: {error_std:.4f} A")
            print(f"    Error trend: {error_trend:.6f} A/s")

        system_rms_error = float(np.sqrt(total_rms_error))
        print(f"  System overall RMS error: {system_rms_error:.4f} A")

        # PID region usage analysis
        print("\nPID Region Usage:")
        for circuit_id in circuit_ids:
            if "regions" in results[circuit_id]:
                regions = results[circuit_id]["regions"]
                unique_regions, counts = np.unique(regions, return_counts=True)

                print(f"  {circuit_id}:")
                for region, count in zip(unique_regions, counts):
                    time_percent = (count / len(regions)) * 100
                    print(f"    {region} region: {time_percent:.1f}% of time")

    # Energy analysis
    print("\nEnergy Analysis:")
    total_energy_dissipated = 0.0
    max_instantaneous_power = 0.0

    for circuit_id in circuit_ids:
        power = results[circuit_id]["power"]
        energy_dissipated = float(np.sum(power) * (t[1] - t[0]))
        max_power = float(np.max(power))
        avg_power = float(np.mean(power))

        total_energy_dissipated += energy_dissipated
        max_instantaneous_power = max(max_instantaneous_power, max_power)

        print(f"  {circuit_id}:")
        print(f"    Energy dissipated: {energy_dissipated:.3f} J")
        print(f"    Average power: {avg_power:.3f} W")
        print(f"    Peak power: {max_power:.3f} W")
        print(f"    Power efficiency: {avg_power/max_power*100:.1f}%")

    print(f"  Total system energy dissipated: {total_energy_dissipated:.3f} J")
    print(f"  Max instantaneous power (any circuit): {max_instantaneous_power:.3f} W")

    # Coupling efficiency analysis
    print("\nCoupling System Efficiency:")

    # Calculate energy transfer efficiency
    if n_circuits > 1:
        # Energy balance analysis
        total_input_energy = 0.0
        total_dissipated_energy = total_energy_dissipated

        # Estimate input energy from voltage and current
        for circuit_id in circuit_ids:
            voltage = results[circuit_id]["voltage"]
            current = results[circuit_id]["current"]
            input_power = voltage * current
            input_energy = float(np.sum(np.maximum(input_power, 0)) * (t[1] - t[0]))
            total_input_energy += input_energy

        if total_input_energy > 0:
            system_efficiency = total_dissipated_energy / total_input_energy * 100
            print(f"  System energy efficiency: {system_efficiency:.1f}%")
            print(f"  Total input energy: {total_input_energy:.3f} J")

        # Magnetic coupling strength assessment
        M_matrix = coupled_system.M
        L_diagonal = np.diag(M_matrix)
        coupling_ratios = []

        for i in range(n_circuits):
            for j in range(i + 1, n_circuits):
                coupling_ratio = abs(M_matrix[i, j]) / np.sqrt(
                    L_diagonal[i] * L_diagonal[j]
                )
                coupling_ratios.append(coupling_ratio)
                print(
                    f"  Coupling ratio {circuit_ids[i]}-{circuit_ids[j]}: {coupling_ratio:.3f}"
                )

        if coupling_ratios:
            avg_coupling = np.mean(coupling_ratios)
            print(f"  Average coupling strength: {avg_coupling:.3f}")


def save_coupled_results(
    coupled_system,
    t: np.ndarray,
    results: Dict,
    filename: str = "coupled_simulation_results.npz",
):
    """
    Save coupled circuit results to file

    Enhanced version using new results management system
    """
    # Determine strategy type from results
    first_circuit_id = list(results.keys())[0]
    if (
        "reference" in results[first_circuit_id]
        and "error" in results[first_circuit_id]
    ):
        strategy_type = "pid_control"
    else:
        strategy_type = "voltage_input"

    # Create ProcessedResults structure
    processed_results = ProcessedResults(
        time=t,
        circuits=results,
        strategy_type=strategy_type,
        metadata={
            "n_circuits": coupled_system.n_circuits,
            "circuit_ids": coupled_system.circuit_ids,
            "mutual_inductances": coupled_system.M.tolist(),
        },
    )

    # Generate analytics
    analytics = PlottingAnalytics.analyze_circuit_performance(processed_results)

    # Add coupling-specific analytics
    analytics["coupling_analysis"] = {
        "mutual_inductance_matrix": coupled_system.M.tolist(),
        "circuit_count": coupled_system.n_circuits,
        "circuit_ids": coupled_system.circuit_ids,
    }

    # Calculate cross-correlations
    current_matrix = np.array(
        [results[cid]["current"] for cid in coupled_system.circuit_ids]
    )
    correlation_matrix = np.corrcoef(current_matrix)
    analytics["coupling_analysis"]["current_correlations"] = correlation_matrix.tolist()

    ResultsFileManager.save_results(processed_results, analytics, filename)


def load_coupled_results(filename: str):
    """Load previously saved coupled circuit results"""
    return ResultsFileManager.load_results(filename)


def create_coupling_comparison_plot(
    results_uncoupled: Dict,
    results_coupled: Dict,
    time: np.ndarray,
    circuit_ids: List[str],
    save_path: str = None,
    show: bool = True,
):
    """
    Create specialized comparison plot showing coupling effects

    Args:
        results_uncoupled: Results from uncoupled simulation
        results_coupled: Results from coupled simulation
        time: Time array
        circuit_ids: List of circuit IDs to compare
        save_path: Optional path to save plot
        show: Whether to show the plot
    """
    n_circuits = len(circuit_ids)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = plt.cm.Set1(np.linspace(0, 1, n_circuits))

    for i, circuit_id in enumerate(circuit_ids):
        color = colors[i]

        # Current comparison
        axes[0].plot(
            time,
            results_uncoupled[circuit_id]["current"],
            color=color,
            linestyle="--",
            alpha=0.7,
            label=f"{circuit_id} Uncoupled",
        )
        axes[0].plot(
            time,
            results_coupled[circuit_id]["current"],
            color=color,
            linestyle="-",
            linewidth=2,
            label=f"{circuit_id} Coupled",
        )

        # Voltage comparison
        axes[1].plot(
            time,
            results_uncoupled[circuit_id]["voltage"],
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        axes[1].plot(
            time,
            results_coupled[circuit_id]["voltage"],
            color=color,
            linestyle="-",
            linewidth=2,
        )

        # Power comparison
        axes[2].plot(
            time,
            results_uncoupled[circuit_id]["power"],
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        axes[2].plot(
            time,
            results_coupled[circuit_id]["power"],
            color=color,
            linestyle="-",
            linewidth=2,
        )

        # Error comparison (if available)
        if (
            "error" in results_uncoupled[circuit_id]
            and "error" in results_coupled[circuit_id]
        ):
            axes[3].plot(
                time,
                np.abs(results_uncoupled[circuit_id]["error"]),
                color=color,
                linestyle="--",
                alpha=0.7,
            )
            axes[3].plot(
                time,
                np.abs(results_coupled[circuit_id]["error"]),
                color=color,
                linestyle="-",
                linewidth=2,
            )

    # Configure axes
    axes[0].set_title("Current: Uncoupled vs Coupled")
    axes[0].set_ylabel("Current (A)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Voltage: Uncoupled vs Coupled")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Power: Uncoupled vs Coupled")
    axes[2].set_ylabel("Power (W)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    if "error" in results_uncoupled[circuit_ids[0]]:
        axes[3].set_title("|Error|: Uncoupled vs Coupled")
        axes[3].set_ylabel("|Error| (A)")
        axes[3].set_xlabel("Time (s)")
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].set_title("Coupling Effect Magnitude")
        axes[3].set_ylabel("Current Difference (A)")
        axes[3].set_xlabel("Time (s)")
        axes[3].grid(True, alpha=0.3)

        # Plot current difference to show coupling effect
        for i, circuit_id in enumerate(circuit_ids):
            color = colors[i]
            current_diff = np.abs(
                results_coupled[circuit_id]["current"]
                - results_uncoupled[circuit_id]["current"]
            )
            axes[3].plot(
                time,
                current_diff,
                color=color,
                linewidth=2,
                label=f"{circuit_id} Coupling Effect",
            )
        axes[3].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Coupling comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_region_analysis(coupled_system, t: np.ndarray, results: Dict):
    """
    Create specialized plot for PID region analysis across coupled circuits

    Args:
        coupled_system: CoupledRLCircuitsPID instance
        t: Time array
        results: Results dictionary
    """
    # Check if PID regions are available
    has_regions = all("regions" in results[cid] for cid in coupled_system.circuit_ids)
    if not has_regions:
        print("⚠️ No PID region data available for region analysis")
        return

    n_circuits = len(coupled_system.circuit_ids)
    fig, axes = plt.subplots(
        n_circuits + 1, 1, figsize=(16, 4 * (n_circuits + 1)), sharex=True
    )

    if n_circuits == 1:
        axes = [axes]

    colors = plt.cm.Set1(np.linspace(0, 1, n_circuits))
    region_colors = {"low": "lightgreen", "medium": "lightyellow", "high": "lightcoral"}

    # Plot each circuit's current with region background
    for i, circuit_id in enumerate(coupled_system.circuit_ids):
        ax = axes[i]
        data = results[circuit_id]

        # Add region background
        regions = data["regions"]
        prev_region = None
        sampled_regions = regions[::100] if len(regions) > 100 else regions
        sampled_time = t[::100] if len(t) > 100 else t

        for j, region in enumerate(sampled_regions):
            if region != prev_region:
                region_start = sampled_time[j]
                region_end = sampled_time[-1]

                for k in range(j + 1, len(sampled_regions)):
                    if sampled_regions[k] != region:
                        region_end = sampled_time[k]
                        break

                ax.axvspan(
                    region_start,
                    region_end,
                    alpha=0.3,
                    color=region_colors.get(region.lower(), "lightgray"),
                )
                prev_region = region

        # Plot current and reference
        ax.plot(
            t,
            data["current"],
            color=colors[i],
            linewidth=2,
            label=f"{circuit_id} Current",
        )
        ax.plot(
            t,
            data["reference"],
            color=colors[i],
            linewidth=2,
            linestyle="--",
            alpha=0.7,
            label=f"{circuit_id} Reference",
        )

        ax.set_ylabel("Current (A)")
        ax.set_title(f"{circuit_id} - Current Tracking with PID Regions")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Summary plot showing all PID gains
    ax = axes[-1]
    for i, circuit_id in enumerate(coupled_system.circuit_ids):
        data = results[circuit_id]
        ax.plot(
            t,
            data["Kp"],
            color=colors[i],
            linewidth=2,
            linestyle="-",
            label=f"{circuit_id} Kp",
        )
        ax.plot(
            t,
            data["Ki"],
            color=colors[i],
            linewidth=2,
            linestyle=":",
            label=f"{circuit_id} Ki",
        )
        ax.plot(
            t,
            data["Kd"] * 100,
            color=colors[i],
            linewidth=2,
            linestyle="-.",
            label=f"{circuit_id} Kd×100",
        )

    ax.set_ylabel("PID Gains")
    ax.set_xlabel("Time (s)")
    ax.set_title("PID Gains for All Circuits")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Advanced coupled circuit analysis functions
def analyze_magnetic_coupling_efficiency(coupled_system, results: Dict, t: np.ndarray):
    """
    Analyze the efficiency of magnetic coupling in the system

    Returns detailed metrics about coupling effectiveness
    """
    circuit_ids = coupled_system.circuit_ids
    n_circuits = len(circuit_ids)
    M_matrix = coupled_system.M

    # Calculate mutual energy transfer
    coupling_analysis = {
        "energy_transfer": {},
        "coupling_effectiveness": {},
        "phase_relationships": {},
    }

    for i in range(n_circuits):
        for j in range(i + 1, n_circuits):
            circuit_i_id = circuit_ids[i]
            circuit_j_id = circuit_ids[j]

            current_i = results[circuit_i_id]["current"]
            current_j = results[circuit_j_id]["current"]

            # Mutual inductance between circuits i and j
            M_ij = M_matrix[i, j]

            # Energy stored in mutual field
            mutual_energy = 0.5 * M_ij * current_i * current_j
            total_mutual_energy = np.sum(np.abs(mutual_energy)) * (t[1] - t[0])

            # Phase relationship (correlation and phase shift)
            correlation = np.corrcoef(current_i, current_j)[0, 1]

            # Cross-correlation to find phase shift
            cross_corr = np.correlate(
                current_i - np.mean(current_i),
                current_j - np.mean(current_j),
                mode="full",
            )
            phase_shift_samples = np.argmax(cross_corr) - len(current_i) + 1
            phase_shift_time = phase_shift_samples * (t[1] - t[0])

            coupling_analysis["energy_transfer"][f"{circuit_i_id}_{circuit_j_id}"] = {
                "mutual_inductance": float(M_ij),
                "total_mutual_energy": float(total_mutual_energy),
                "average_mutual_power": float(total_mutual_energy / (t[-1] - t[0])),
            }

            coupling_analysis["coupling_effectiveness"][
                f"{circuit_i_id}_{circuit_j_id}"
            ] = {
                "current_correlation": float(correlation),
                "coupling_strength_ratio": float(
                    abs(M_ij) / np.sqrt(M_matrix[i, i] * M_matrix[j, j])
                ),
                "energy_coupling_ratio": float(
                    total_mutual_energy
                    / (np.sum(results[circuit_i_id]["power"]) * (t[1] - t[0]))
                ),
            }

            coupling_analysis["phase_relationships"][
                f"{circuit_i_id}_{circuit_j_id}"
            ] = {
                "phase_shift_time": float(phase_shift_time),
                "phase_shift_samples": int(phase_shift_samples),
                "correlation_coefficient": float(correlation),
            }

    return coupling_analysis


# Export all backward compatible functions
__all__ = [
    "prepare_coupled_post",
    "plot_coupled_vresults",
    "plot_coupled_results",
    "analyze_coupling_effects",
    "save_coupled_results",
    "load_coupled_results",
    "create_coupling_comparison_plot",
    "plot_region_analysis",
    "analyze_magnetic_coupling_efficiency",
]
