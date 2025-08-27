#!/usr/bin/env python3
"""
Coupled RL Circuits PID Control System

Command-line interface for running coupled RL circuit simulations with
independent adaptive PID controllers and magnetic coupling.
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from typing import List

from .coupled_circuits import CoupledRLCircuitsPID
from .rlcircuitpid import RLCircuitPID
from .coupled_plotting import (
    prepare_coupled_post,
    plot_coupled_vresults,
    plot_coupled_results,
    plot_region_analysis,
    analyze_coupling_effects,
    save_coupled_results,
)
from pid_controller import create_adaptive_pid_controller


def load_circuit_configuration(config_file: str) -> List[RLCircuitPID]:
    """Load circuit configuration from JSON file"""
    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)

        circuits = []
        for circuit_data in config_data["circuits"]:
            # Create PID controller if parameters provided
            pid_controller = None
            if "pid_params" in circuit_data:
                pid_params = circuit_data["pid_params"]
                pid_controller = create_adaptive_pid_controller(**pid_params)

            # Create RLCircuitPID instance
            circuit = RLCircuitPID(
                R=circuit_data.get("R", 1.0),
                L=circuit_data.get("L", 0.1),
                pid_controller=pid_controller,
                reference_csv=circuit_data.get("reference_csv", None),
                voltage_csv=circuit_data.get("voltage_csv", None),
                resistance_csv=circuit_data.get("resistance_csv"),
                temperature=circuit_data.get("temperature", 25.0),
                circuit_id=circuit_data.get("circuit_id", f"circuit_{len(circuits)+1}"),
            )
            circuits.append(circuit)

        # Load mutual inductance matrix if provided
        mutual_inductances = None
        if "mutual_inductances" in config_data:
            mutual_inductances = np.array(config_data["mutual_inductances"])

        return circuits, mutual_inductances

    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


def run_coupled_simulation(args):
    """Run the coupled circuits simulation"""

    print("\n=== Coupled RL Circuits PID Simulation ===")

    # Load or create circuit configuration
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        circuits, mutual_inductances = load_circuit_configuration(args.config_file)
    else:
        raise RuntimeError("Configuration file must be provided with --config-file")

    # Create coupled system
    if mutual_inductances is not None:
        coupled_system = CoupledRLCircuitsPID(
            circuits, mutual_inductances=mutual_inductances
        )
    else:
        raise RuntimeError(
            "Mutual inductance matrix must be provided in configuration file."
        )

    # Set experimental data if provided
    voltage_csvs = [circuit.voltage_csv for circuit in coupled_system.circuits]
    reference_csvs = [circuit.reference_csv for circuit in coupled_system.circuits]

    key = "voltage"
    if all(v is None for v in voltage_csvs):
        key = "current"

    experimental_data = {}
    if args.experimental_csv:
        exp_files = args.experimental_csv.split(",")
        if len(exp_files) != coupled_system.n_circuits:
            print(
                f"Warning: Number of experimental files ({len(exp_files)}) does not match number of circuits ({coupled_system.n_circuits})."
            )
        for i, exp_file in enumerate(exp_files):
            circuit_id = coupled_system.circuits[i].circuit_id
            experimental_data[circuit_id] = exp_file
            try:
                exp_data = pd.read_csv(exp_file)

                # Validate required columns
                if "time" not in exp_data.columns or key not in exp_data.columns:
                    raise ValueError(
                        f"Experimental CSV must contain 'time' and {key} columns"
                    )
                print(f"✓ Experimental data loaded: {len(exp_data['time'])} points")
            except Exception as e:
                print(f"❌ Error loading experimental data: {e}")

            print(
                f"Loaded experimental data for {coupled_system.circuits[i].circuit_id} from {exp_file}"
            )

    # Print configuration
    coupled_system.print_configuration()

    # Time parameters
    t0, t1 = args.time_start, args.time_end
    dt = args.time_step

    mode = "regular"
    if key == "current":
        mode = "cde"

    print(f"\nRunning simulation {mode}...")
    print(f"Time span: {t0} to {t1} seconds")
    print(f"Time step: {dt} seconds")

    if mode == "regular":
        print(f"\nUsing input voltage CSV: {voltage_csvs}")

        i0 = args.value_start
        print(f"Initial current: {i0} A at t={t0} s")
        print(
            f"Initial voltage: {[circuit.input_voltage(t0) for circuit in coupled_system.circuits]} V at t={t0} s"
        )
        y0 = np.array(i0)

        t_span = (t0, t1)
        sol = solve_ivp(
            lambda t, current: coupled_system.voltage_vector_field(t, current),
            t_span,
            y0,
            method="RK45",
            dense_output=True,
            rtol=1e-6,
            atol=1e-9,
            max_step=dt,
        )

        # Run simulation
        print("✓ Simulation completed")

        # Post-process results
        print("Processing results...")
        t, results = prepare_coupled_post(sol, coupled_system)

        print("Generating plots...")
        plot_coupled_vresults(
            sol,
            coupled_system,
            t,
            results,
            experimental_data,
            save_path=args.save_plots,
            show=args.show_plots,
        )

    else:

        # Store results
        all_t = []
        all_y = []

        print(f"\nUsing CDE for PID control: {reference_csvs}")
        i0 = args.value_start
        print(f"\nInitial current: {i0} A at t={t0} s")
        i0_ref = [circuit.reference_current(t0) for circuit in coupled_system.circuits]
        print(f"init ref: {i0_ref:.3f} A at t={t0:.3f} s")
        v0 = []
        if experimental_data is not None:
            for circuit in coupled_system.circuits:
                circuit_id = circuit.circuit_id
                if circuit_id in experimental_data:
                    exp_file = experimental_data[circuit_id]
                    data = pd.read_csv(exp_file)

                    v0.append(np.interp(t0, data["time"], data["voltage"]))
            print(f"init exp: {v0:.3f} V at t={t0:.3f} s")

        # merge all references
        # sort references
        # remove duplicates
        merged_ref = pd.read_csv(reference_csvs[0])
        # rename current colum,
        merged_ref = merged_ref.rename(columns={"current": "current1"})
        for n in range(1, len(reference_csvs) - 1):
            df = pd.read_csv(reference_csvs[n])
            df = df.rename(columns={"current": f"current{n+1}"})
            merged_ref = pd.merge(merged_ref, df, on="time", how="outer").sort_values(
                "time"
            )

            # Interpolate missing values
            merged_ref[f"value{n}"] = merged_ref[f"value{n}"].interpolate(
                method="linear"
            )
            merged_ref[f"value{n+1}"] = merged_ref[f"value{n+1}"].interpolate(
                method="linear"
            )

        closest_index = np.argmin(np.abs(merged_ref["time"].to_numpy() - t0))
        # shall check if circuit.time_data[closest_index] is greater than t0
        error = i0_ref - i0
        print(
            f"t0: {t0}, closest_index={closest_index}, time_data={merged_ref['time'].iloc[closest_index]}, error={error}"
        )
        print(
            f"\nPID controller: run pid for each sequence of reference_current ({merged_ref.shape[0]} sequences)..."
        )
        for n in range(closest_index + 1, merged_ref.shape[0] - 1):
            print(
                f'n={n}, t={merged_ref["time"].iloc[n]:.3f} s, ref={merged_ref.filter(regex="current*").iloc[n]} A',
            )
        print("not implemented yet")

    # Save results if requested
    if args.save_results:
        print("save result to {args.save_results}")
        save_coupled_results(coupled_system, t, results, args.save_results)

    return sol, coupled_system, t, results


def main():
    """Main function with argument parsing"""

    parser = argparse.ArgumentParser(
        description="Coupled RL Circuits PID Control Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration options
    parser.add_argument(
        "--version",
        action="version",
        version="Coupled RL Circuits PID Simulation 1.0",
    )
    parser.add_argument("--wd", type=str, help="Working directory")
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        help="Path to JSON configuration file with circuit definitions",
    )

    # TODO make experimental_csv a list
    parser.add_argument(
        "--experimental_csv",
        "-e",
        nargs="?",
        type=str,
        help="Path to CSV file with experimental current (columns: time, input) or voltage data (columns: time, voltage) for comparison ",
    )

    # Simulation parameters
    parser.add_argument(
        "--value_start",
        nargs="?",
        type=float,
        help="Current values at start time in Ampere",
    )

    parser.add_argument(
        "--time-start", type=float, default=0.0, help="Simulation start time in seconds"
    )

    parser.add_argument(
        "--time-end", type=float, default=5.0, help="Simulation end time in seconds"
    )

    parser.add_argument(
        "--time-step", type=float, default=0.001, help="Simulation time step in seconds"
    )

    # Output options
    parser.add_argument(
        "--show-plots",
        "-p",
        action="store_true",
        help="Show plots of simulation results",
    )

    parser.add_argument(
        "--show-analytics",
        "-a",
        action="store_true",
        help="Show detailed analytics of simulation results",
    )

    parser.add_argument(
        "--save-results",
        type=str,
        help="Save results to specified file (e.g., results.npz)",
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        help="Save plots to specified file (e.g., plots.png)",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.wd:
        import os

        pwd = os.chdir(args.wd)
        print(f"Working directory set to: {args.wd}")
    print("✓ Working directory:", os.getcwd())

    # Validate configuration file if provided
    if args.config_file:
        try:
            with open(args.config_file, "r") as f:
                config = json.load(f)
            print(f"✓ Configuration file loaded: {args.config_file}")
        except FileNotFoundError:
            print(f"✗ Configuration file not found: {args.config_file}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error reading configuration file: {e}")
            sys.exit(1)

    # Check if experiments file exists
    if args.experimental_csv:
        for exp_file in args.experimental_csv.split(","):
            try:
                with open(exp_file, "r") as f:
                    pass
                print(f"✓ Experimental data file found: {exp_file}")
            except FileNotFoundError:
                print(f"✗ Experimental data file not found: {exp_file}")
                sys.exit(1)
            except Exception as e:
                print(f"✗ Error reading experimental data file: {e}")
                sys.exit(1)

    # Run simulation
    try:
        sol, coupled_system, t, results = run_coupled_simulation(args)
        print("\n✓ Coupled simulation completed successfully!")
        print(f"  Circuits simulated: {coupled_system.n_circuits}")
        print(f"  Time points: {len(t)}")
        print(f"  Total simulation time: {float(t[-1] - t[0]):.3f} seconds")

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if args.wd:
        os.chdir(pwd)
        print(f"Returned to original directory: {pwd}")


if __name__ == "__main__":
    main()
