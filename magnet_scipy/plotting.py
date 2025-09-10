import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from .rlcircuitpid import RLCircuitPID
from scipy import stats
from .utils import exp_metrics


def prepare_post(
    sol,
    circuit: RLCircuitPID,
    mode: str = "regular",
) -> Tuple[np.ndarray, Dict]:
    """
    Post-process results for coupled RL circuits

    Returns data structures for plotting and analysis
    """
    t = sol.t
    circuit_id = circuit.circuit_id

    results = {}
    i_ref = None
    error = None
    Kp_array = None
    Ki_array = None
    Kd_array = None
    current_regions = None
    integral_error = None

    rms_diff = None
    mae_diff = None

    print(f"use_variable_temperature: {circuit.use_variable_temperature}")
    temperature_over_time = None
    if circuit.use_variable_temperature:
        temperature_over_time = np.array(
            [circuit.get_temperature(float(time)) for time in t]
        )
    print(circuit_id, ": t=", t.shape)
    if circuit.use_variable_temperature:
        print("temperature=", temperature_over_time.shape)

    # load experimental data
    circuit._load_experimental_data()

    # Extract state variables for this circuit
    if mode == "regular":
        current = sol.y.squeeze()
        voltage = circuit.voltage_func(t)

        print(
            "experimental_data[current_current]:",
            circuit.has_experimental_data(data_type="current", key="current"),
        )
        if circuit.has_experimental_data(data_type="current", key="current"):
            exp_time = circuit.experimental_functions["current_current"]["time_data"]
            exp_values = circuit.experimental_functions["current_current"][
                "values_data"
            ]
            rms_diff, mae_diff = exp_metrics(t, exp_time, current, exp_values)
            print(
                f"Comparison metrics for {circuit_id}: RMS Difference = {rms_diff:.4f} A, MAE = {mae_diff:.4f} A"
            )
            circuit.experimental_functions["current_current"]["rms_diff"] = rms_diff
            circuit.experimental_functions["current_current"]["mae_diff"] = mae_diff

    else:
        current = sol.y[0]
        integral_error = sol.y[1]
        print("current=", current.shape)
        print("integral_error=", integral_error.shape)

        # Calculate reference current
        i_ref = np.array([circuit.reference_current(t_val) for t_val in t])
        print("i_ref=", i_ref.shape)

        # Calculate adaptive PID gains over time
        Kp_over_time = []
        Ki_over_time = []
        Kd_over_time = []
        current_regions = []

        for j, i_ref_val in enumerate(i_ref):
            i_ref_float = float(i_ref_val)
            Kp, Ki, Kd = circuit.get_pid_parameters(i_ref_float)
            Kp_over_time.append(float(Kp))
            Ki_over_time.append(float(Ki))
            Kd_over_time.append(float(Kd))

            region_name = circuit.get_current_region(i_ref_float)
            current_regions.append(region_name)

        # Calculate control signals and errors
        error = i_ref - current
        derivative_error = np.gradient(error, t[1] - t[0])

        Kp_array = np.array(Kp_over_time)
        Ki_array = np.array(Ki_over_time)
        Kd_array = np.array(Kd_over_time)

        voltage = (
            Kp_array * error + Ki_array * integral_error + Kd_array * derivative_error
        )

        print(
            "experimental_data[voltage_voltage]:",
            circuit.has_experimental_data(data_type="voltage"),
            circuit.has_experimental_data(data_type="voltage", key="voltage"),
        )
        if circuit.has_experimental_data(data_type="voltage", key="voltage"):
            exp_time = circuit.experimental_functions["voltage_voltage"]["time_data"]
            exp_values = circuit.experimental_functions["voltage_voltage"][
                "values_data"
            ]
            rms_diff, mae_diff = exp_metrics(t, exp_time, voltage, exp_values)
            print(
                f"Comparison metrics for {circuit_id}: RMS Difference = {rms_diff:.4f} V, MAE = {mae_diff:.4f} V"
            )
            circuit.experimental_functions["voltage_voltage"]["rms_diff"] = rms_diff
            circuit.experimental_functions["voltage_voltage"]["mae_diff"] = mae_diff

    # Calculate variable resistance over time
    resistance_over_time = None
    if circuit.use_variable_temperature:
        resistance_over_time = np.array(
            [
                circuit.get_resistance(float(curr), float(temp))
                for curr, temp in zip(current, temperature_over_time)
            ]
        )
    else:
        resistance_over_time = np.array(
            [circuit.get_resistance(float(curr)) for curr in current]
        )

    # Calculate power dissipation
    power = resistance_over_time * current**2

    # Store results for this circuit
    results = {
        "current": current,
        "temperature": temperature_over_time,
        "reference": i_ref,
        "error": error,
        "voltage": voltage,
        "power": power,
        "resistance": resistance_over_time,
        "Kp": Kp_array,
        "Ki": Ki_array,
        "Kd": Kd_array,
        "regions": current_regions,
        "integral_error": integral_error,
    }

    for key in circuit.experimental_functions:
        results[key] = {}
        if "rms_diff" in circuit.experimental_functions[key]:
            results[key]["rms_diff"] = circuit.experimental_functions[key]["rms_diff"]
        if "mae_diff" in circuit.experimental_functions[key]:
            results[key]["mae_diff"] = circuit.experimental_functions[key]["mae_diff"]

    print(f"results[{circuit_id}]: {results.keys()}")
    print(f"\n{circuit_id} stats:")
    print("current stats: ", stats.describe(current))
    print("voltage stats: ", stats.describe(voltage))
    if temperature_over_time is not None:
        print("T stats: ", stats.describe(temperature_over_time))
    print("R stats: ", stats.describe(resistance_over_time))
    print("Power stats: ", stats.describe(power))

    return t, results


def plot_vresults(
    circuit: RLCircuitPID,
    t: np.ndarray,
    data: Dict,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot comprehensive results for coupled RL circuits with regular ODE
    """
    circuit_id = circuit.circuit_id

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    axes = axes.flatten()

    # 1. Current tracking for all circuits
    ax = axes[0]
    ax.plot(
        t,
        data["current"],
        linewidth=2,
        label=f"{circuit_id}",
        linestyle="-",
    )
    if circuit.has_experimental_data(data_type="current", key="current"):
        exp_time = circuit.experimental_functions["current_current"]["time_data"]
        exp_current = circuit.experimental_functions["current_current"]["values_data"]
        ax.plot(
            exp_time,
            exp_current,
            linewidth=1,
            label=f"{circuit_id} - Experimental",
            linestyle=":",
            alpha=0.7,
        )

        # Add comparison metrics to the plot
        rms_diff = circuit.experimental_functions["current_current"]["rms_diff"]
        mae_diff = circuit.experimental_functions["current_current"]["mae_diff"]

        ax.text(
            0.02,
            0.02,
            f"{circuit_id} RMS Diff: {rms_diff:.2f} A\n{circuit_id} MAE Diff: {mae_diff:.2f} A",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    # ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Currents")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Voltage
    ax = axes[1]
    ax.plot(t, data["voltage"], linewidth=2, label=circuit_id)

    # ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltages")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Variable resistance
    ax = axes[2]
    # Adding Twin Axes to plot using temperature - if not constant
    use_variable_temperature = circuit.use_variable_temperature
    if use_variable_temperature:
        ax2 = ax.twinx()

    if use_variable_temperature:
        label = f"{circuit_id}"  # change label if temp is contant
    else:
        label = f"{circuit_id} Tin={circuit.temperature}°C"
    ax.plot(t, data["resistance"], linewidth=2, label=label)

    # TODO add temperature if available in right yaxis - if not constant
    if use_variable_temperature:
        label = f"{circuit_id} Tin"
        ax2.plot(
            t,
            data["temperature"],
            marker="o",
            markersize=4,
            markevery=5,
            linestyle="",
            label=label,
            alpha=0.5,
        )

    # if there is at least a circuit with a non constant temperature
    if use_variable_temperature:
        ax2.set_ylabel("Tin (°C)")
        ax2.tick_params(axis="y")
        ax2.grid(False)
        # ax2.legend()

    # ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resistance (Ω)")
    ax.set_title("Resistances")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # # 4. Power dissipation
    ax = axes[3]
    ax.plot(t, data["power"], linewidth=2, label=circuit_id)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title(" P = R(I,T) × I² ")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if show:
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Plots saved to {save_path}")
    plt.close(fig)


def plot_results(
    sol,
    circuit: RLCircuitPID,
    t: np.ndarray,
    data: Dict,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot comprehensive results for coupled RL circuits
    """
    circuit_id = circuit.circuit_id

    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
    axes = axes.flatten()

    # 1. Current tracking for all circuits
    ax = axes[0]
    ax.plot(
        t,
        data["current"],
        linewidth=2,
        label=f"{circuit_id} - Actual",
        linestyle="-",
    )
    ax.plot(
        t,
        data["reference"],
        linewidth=2,
        label=f"{circuit_id} - Reference",
        linestyle="--",
        alpha=0.7,
    )

    # Color background by current region
    prev_region = None
    region_colors = {
        "Low": "lightgreen",
        "Medium": "lightyellow",
        "High": "lightcoral",
        "low": "lightgreen",
        "medium": "lightyellow",
        "high": "lightcoral",
    }

    current_regions = data["regions"]
    for i, region in enumerate(
        current_regions[::100]
    ):  # Sample every 100 points for performance
        if region != prev_region:
            region_start = t[i * 100] if i * 100 < len(t) else t[-1]
            # Find next region change
            region_end = t[-1]
            for j in range(i + 1, len(current_regions[::100])):
                if current_regions[j * 100] != region and j * 100 < len(t):
                    region_end = t[j * 100]
                    break

            color_key = region.lower() if region.lower() in region_colors else region
            ax.axvspan(
                region_start,
                region_end,
                alpha=0.2,
                color=region_colors.get(color_key, "lightgray"),
                label=f"{region} Current" if prev_region != region else "",
            )
            prev_region = region

    # ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Current Tracking - All Circuits")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. voltages
    ax = axes[1]
    ax.plot(t, data["voltage"], linewidth=2, label=circuit_id)
    if circuit.has_experimental_data(data_type="voltage", key="voltage"):
        exp_time = circuit.experimental_functions["voltage_voltage"]["time_data"]
        exp_current = circuit.experimental_functions["voltage_voltage"]["values_data"]
        ax.plot(
            exp_time,
            exp_current,
            linewidth=1,
            label=f"{circuit_id} - Experimental",
            linestyle=":",
            alpha=0.7,
        )
        # Add comparison metrics to the plot
        rms_diff = circuit.experimental_functions["voltage_voltage"]["rms_diff"]
        mae_diff = circuit.experimental_functions["voltage_voltage"]["mae_diff"]

        ax.text(
            0.02,
            0.02,
            f"{circuit_id} RMS Diff: {rms_diff:.2f} V\n{circuit_id} MAE Diff: {mae_diff:.2f} V",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    # ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Control Voltages")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Variable resistance
    ax = axes[2]
    # Adding Twin Axes to plot using temperature - if not constant
    use_variable_temperature = circuit.use_variable_temperature
    if use_variable_temperature:
        ax2 = ax.twinx()

    if "temperature" in data and data["temperature"] is not None:
        label = f"{circuit_id}"  # change label if temp is contant
    else:
        label = f"{circuit_id} (Tin={circuit.temperature}°C)"

    ax.plot(t, data["resistance"], linewidth=2, label=label)
    # TODO add temperature if available in right yaxis - if not constant
    if use_variable_temperature:
        label = f"{circuit_id} Tin"
        ax2.plot(
            t,
            data["temperature"],
            marker="o",
            markersize=4,
            markevery=5,
            linestyle="",
            label=label,
            alpha=0.5,
        )

    # if there is at least a circuit with a non constant temperature
    if use_variable_temperature:
        ax2.set_ylabel("Tin (°C)")
        ax2.tick_params(axis="y")
        ax2.grid(False)
        # ax2.legend()

    # ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resistance (Ω)")
    ax.set_title("Circuit Resistances")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Power dissipation
    ax = axes[3]
    ax.plot(t, data["power"], linewidth=2, label=circuit_id)

    # ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power Dissipation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5. Tracking errors
    ax = axes[4]
    ax.plot(t, data["error"], linewidth=2, label=circuit_id)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (A)")
    ax.set_title("Tracking Errors")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if show:
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Plots saved to {save_path}")
    plt.close(fig)


def analyze(circuit: RLCircuitPID, t: np.ndarray, data: Dict):
    """
    Provide detailed numerical analysis of coupling effects
    """
    circuit_id = circuit.circuit_id

    print("\n=== Coupling Effects Analysis ===")

    # Current statistics
    print("\nCurrent Statistics:")
    current = data["current"]
    print(f"  {circuit_id}:")
    print(f"    Max current: {float(np.max(np.abs(current))):.3f} A")
    print(f"    RMS current: {float(np.sqrt(np.mean(current**2))):.3f} A")
    print(f"    Current variation (std): {float(np.std(current)):.3f} A")

    # Error analysis
    print("\nTracking Performance:")
    error = data["error"]
    rms_error = float(np.sqrt(np.mean(error**2)))
    max_error = float(np.max(np.abs(error)))
    print(f"  {circuit_id}:")
    print(f"    RMS error: {rms_error:.4f} A")
    print(f"    Max error: {max_error:.4f} A")

    # PID region usage
    print("\nPID Region Usage:")
    regions = data["regions"]
    unique_regions, counts = np.unique(regions, return_counts=True)

    print(f"  {circuit_id}:")
    for region, count in zip(unique_regions, counts):
        time_percent = (count / len(regions)) * 100
        print(f"    {region} region: {time_percent:.1f}% of time")

    # Energy analysis
    print("\nEnergy Analysis:")
    total_energy_dissipated = 0.0
    max_instantaneous_power = 0.0

    power = data["power"]
    energy_dissipated = float(np.sum(power) * (t[1] - t[0]))
    max_power = float(np.max(power))

    total_energy_dissipated += energy_dissipated
    max_instantaneous_power = max(max_instantaneous_power, max_power)

    print(f"  {circuit_id}:")
    print(f"    Energy dissipated: {energy_dissipated:.3f} J")
    print(f"    Max power: {max_power:.3f} W")

    print(f"  Total system energy dissipated: {total_energy_dissipated:.3f} J")
    print(f"  Max instantaneous power (any circuit): {max_instantaneous_power:.3f} W")


def save_results(
    circuit: RLCircuitPID,
    t: np.ndarray,
    data: Dict,
    filename: str = "coupled_simulation_results.npz",
):
    """
    Save simulation results to a file for later analysis
    """

    circuit_id = circuit.circuit_id

    # Prepare data for saving
    save_data = {
        "time": t,
        "circuit_id": circuit_id,
    }

    # Add results for each circuit
    save_data[f"{circuit_id}_current"] = data["current"]
    save_data[f"{circuit_id}_reference"] = data["reference"]
    save_data[f"{circuit_id}_error"] = data["error"]
    save_data[f"{circuit_id}_voltage"] = data["voltage"]
    save_data[f"{circuit_id}_power"] = data["power"]
    save_data[f"{circuit_id}_resistance"] = data["resistance"]
    save_data[f"{circuit_id}_Kp"] = data["Kp"]
    save_data[f"{circuit_id}_Ki"] = data["Ki"]
    save_data[f"{circuit_id}_Kd"] = data["Kd"]

    # Save to file
    np.savez_compressed(filename, **save_data)
    print(f"Results saved to {filename}")


def load_coupled_results(
    filename: str = "coupled_simulation_results.npz",
) -> Tuple[np.ndarray, Dict]:
    """
    Load previously saved simulation results
    """
    data = np.load(filename)

    t = data["time"]
    circuit_id = data["circuit_id"]

    results = {
        "current": data[f"{circuit_id}_current"],
        "reference": data[f"{circuit_id}_reference"],
        "error": data[f"{circuit_id}_error"],
        "voltage": data[f"{circuit_id}_voltage"],
        "power": data[f"{circuit_id}_power"],
        "resistance": data[f"{circuit_id}_resistance"],
        "Kp": data[f"{circuit_id}_Kp"],
        "Ki": data[f"{circuit_id}_Ki"],
        "Kd": data[f"{circuit_id}_Kd"],
    }

    print(f"Results loaded from {filename}")
    return t, results
