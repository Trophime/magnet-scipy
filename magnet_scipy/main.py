#!/usr/bin/env python3
"""
Basic RL Circuit PID Control System

Simple command-line interface for running RL circuit simulations with
adaptive PID control. Supports loading resistance and reference data from CSV files.
"""

import argparse
import sys
import numpy as np
from scipy.integrate import solve_ivp

from .pid_controller import (
    create_adaptive_pid_controller,
)
from .rlcircuitpid import RLCircuitPID
from .plotting import (
    prepare_post,
    plot_vresults,
    plot_results,
    save_results,
)
from .utils import fake_sol
import json


def load_circuit_configuration(config_file: str) -> RLCircuitPID:
    """Load circuit configuration from JSON file"""
    try:
        with open(config_file, "r") as f:
            circuit_data = json.load(f)

        # Create PID controller if parameters provided
        pid_controller = None
        if "pid_params" in circuit_data:
            pid_params = circuit_data["pid_params"]
            pid_controller = create_adaptive_pid_controller(**pid_params)

        # Create RLCircuitPID instance
        circuit = RLCircuitPID(
            R=circuit_data.get("resistance", 1.0),
            L=circuit_data.get("inductance", 0.1),
            pid_controller=pid_controller,
            reference_csv=circuit_data.get("reference_csv", None),
            voltage_csv=circuit_data.get("voltage_csv", None),
            resistance_csv=circuit_data.get("resistance_csv"),
            temperature=circuit_data.get("temperature", 25.0),
            temperature_csv=circuit_data.get("temperature_csv", None),
            circuit_id=circuit_data.get("circuit_id", "circuit"),
            experimental_data=circuit_data.get("experiment_data", []),
        )
        return circuit

    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


def run_simulation(args):
    """Run the main simulation with given arguments"""

    # Load or create circuit configuration
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        circuit = load_circuit_configuration(args.config_file)
    else:
        raise RuntimeError("Configuration file must be provided with --config-file")

    circuit.validate()
    circuit.print_configuration()
    print(
        f"Reference Time span: {circuit.time_data[0]} to {circuit.time_data[-1]} seconds"
    )

    # Time parameters
    time_range_overwritten = False
    t0, t1 = args.time_start, args.time_end
    if t0 < circuit.time_data[0]:
        t0 = circuit.time_data[0]
        time_range_overwritten = True
    if t1 > circuit.time_data[-1]:
        t1 = circuit.time_data[-1]
        time_range_overwritten = True
    print(f"Time span: {t0} to {t1} seconds (overwritten: {time_range_overwritten})")

    dt = args.time_step

    # Regular ODE
    mode = None
    if circuit.voltage_csv:
        mode = "regular"
        print(f"\nUsing input voltage CSV: {circuit.voltage_csv}")
        i0 = args.value_start
        print(f"Initial current: {i0} A at t={t0} s")
        print(f"Initial voltage: {circuit.input_voltage(t0)} V at t={t0} s")

        y0 = np.array([i0])
        print("y0:", y0, type(y0))

        t_span = (t0, t1)
        sol = solve_ivp(
            lambda t, current: circuit.voltage_vector_field(t, current),
            t_span,
            y0,
            method=args.method,
            dense_output=True,
            rtol=1e-6,
            atol=1e-9,
            max_step=dt,
        )

        print("✓ Simulation completed")
        # Post-process results
        print("Processing results...")
        t, results = prepare_post(sol, circuit, mode)
        print(f"results: {results.keys()}")

        print("Generating plots...")
        plot_vresults(
            circuit,
            t,
            results,
            save_path=args.save_plots,
            show=args.show_plots,
        )
    else:
        mode = "cde"

        # Store results
        all_t = []
        all_y = []

        print(f"\nUsing reference current CSV: {circuit.reference_csv}")
        # Initial conditions [current, integral_error, prev_error]
        i0 = args.value_start
        print(f"\nInitial current: {i0} A at t={t0} s")
        i0_ref = circuit.reference_current(t0)
        print(f"init ref: {i0_ref:.3f} A at t={t0:.3f} s")

        closest_index = np.argmin(np.abs(circuit.time_data - t0))
        # shall check if circuit.time_data[closest_index] is greater than t0

        error = i0_ref - i0  # ??how to initial integral error??
        print(
            f"t0: {t0}, closest_index={closest_index}, time_data={circuit.time_data[closest_index]}, error={error}"
        )
        print(
            f"\nPID controller: run pid for each sequence of reference_current ({len(circuit.time_data)-1} sequences)..."
        )
        for n in range(closest_index + 1, len(circuit.time_data) - 1):
            t_actual = circuit.time_data[n]
            t_previous = circuit.time_data[n - 1]
            t_next = circuit.time_data[n + 1]
            iref = circuit.reference_current(t_actual)
            print(
                f"n={n}, t={t_actual:.3f} s, ref={iref:.3f} A",
                end=", ",
            )
            di_refdt = (
                circuit.reference_current(t_next)
                - circuit.reference_current(t_previous)
            ) / (t_next - t_previous)
            print(f"di_refdt={di_refdt:.3f} A/s", end=", ")
            t_span = (float(t_previous), float(t_actual))
            print(f"t_span: {t_span}", end=", ", flush=True)
            y0 = np.array([i0, error])
            print(f"y0: {y0}", end=": ", flush=True)
            sol = solve_ivp(
                lambda t, current: circuit.vector_field(t, current, di_ref_dt=di_refdt),
                t_span,
                y0,
                method=args.method,
                dense_output=True,
                rtol=1e-6,
                atol=1e-9,
                max_step=dt,
            )
            print(f"tfinal={float(sol.t[-1])} s", end=",", flush=True)
            print(f"i1={float(sol.y[0, -1])} A", end=", ", flush=True)
            print(f"integral_error1={float(sol.y[1, -1])} A.s", end=", ", flush=True)
            print("✓ Simulation completed")

            # Store the time points and solution
            all_t.append(sol.t)
            all_y.append(sol.y)
            # Handling Overlaps: If your intervals overlap or you want to avoid duplicate points at boundaries, you can slice appropriately:
            # all_t.append(sol.t[:-1] if n < len(circuit.time_data)-1 else sol.t)  # Remove last point except for final interval
            # all_y.append(sol.y[:, :-1] if n < len(circuit.time_data)-1 else sol.y)

            # print("Postprocessing for plots...")
            if args.debug:
                # Post-process results
                print("Processing results...")
                t, results = prepare_post(sol, circuit, mode)

                print("Generating plots...")
                plot_results(
                    sol,
                    circuit,
                    t,
                    results,
                    save_path=None,
                    show=True,
                )

            # update for next iteration
            i0 = float(sol.y[0, -1])
            # i0_ref = circuit.reference_current(t_span[-1])
            error = float(sol.y[1, -1])  # i0_ref - i0
            # if n == 10:
            #    print("limit to 10 iterations for testing reached")
            #    break

        # Concatenate all results
        t_global = np.concatenate(all_t)
        y_global = np.concatenate(all_y, axis=1)

        print(f"Total time points: {len(t_global)}")
        u, c = np.unique(t_global, return_counts=True)
        duplicates = u[c > 1]
        if len(duplicates) > 0:
            print(f"Warning: Duplicate time points found: {duplicates}")

        # Post-process results
        print("Processing results...")
        t, results = prepare_post(fake_sol(t_global, y_global), circuit, mode)

        print("Generating plots...")
        plot_results(
            sol,
            circuit,
            t,
            results,
            save_path=args.save_plots,
            show=args.show_plots,
        )

    # Save results if requested
    output_filename = args.config_file.replace(".json", ".res")
    if args.save_results:
        print("save result to {args.save_results}")
        output_filename = args.save_results
    save_results(circuit, t, results, output_filename)

    return sol, circuit


def main():
    """Main function with argument parsing"""

    parser = argparse.ArgumentParser(
        description="RL Circuit PID Control Simulation",
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
        type=str,
        help="Path to JSON configuration file with circuit definitions",
    )

    # Simulation parameters
    parser.add_argument(
        "--value_start",
        type=float,
        default=0.0,
        help="Current value at start time in Ampere",
    )

    parser.add_argument(
        "--time_start", type=float, default=0.0, help="Simulation start time in seconds"
    )

    parser.add_argument(
        "--time_end", type=float, default=5.0, help="Simulation end time in seconds"
    )

    parser.add_argument(
        "--time_step", type=float, default=0.001, help="Simulation time step in seconds"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        default="RK45",
        help="Select method used to solve ODE",
    )

    # Output options
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Show plots of simulation results",
    )

    parser.add_argument(
        "--show_analytics",
        "-a",
        action="store_true",
        help="Show detailed analytics of simulation results",
    )
    parser.add_argument(
        "--save_results",
        type=str,
        help="Save results to specified file (e.g., results.npz)",
    )
    parser.add_argument(
        "--save_plots",
        type=str,
        help="Save plots to specified file (e.g., plots.png)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional plots",
    )

    # Parse arguments
    args = parser.parse_args()
    print(f"args: {args}")

    if args.wd is not None:
        import os

        pwd = os.getcwd()
        os.chdir(args.wd)
        print(f"Working directory set to: {args.wd} (pwd={pwd})")
    print(f"✓ Working directory: {os.getcwd()}")

    # Validate configuration file if provided
    if args.config_file:
        try:
            with open(args.config_file, "r") as f:
                json.load(f)
            print(f"✓ Configuration file loaded: {args.config_file}")
        except FileNotFoundError:
            print(f"✗ Configuration file not found: {args.config_file}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error reading configuration file: {e}")
            sys.exit(1)

    if args.save_plots and args.show_plots:
        raise RuntimeError(
            "⚠️ Warning: Both --save-plots and --show-plots specified. Plots will be saved and shown."
        )
    if not args.save_plots and not args.show_plots:
        print(
            "⚠️ Warning: Neither --save-plots nor --show-plots specified. Force show_plots."
        )
        args.show_plots = True

    # Run simulation
    sol, circuit = run_simulation(args)
    print("\n✓ Simulation completed successfully!")

    if args.wd is not None:
        os.chdir(pwd)
        print(f"Returned to original directory: {pwd}")


if __name__ == "__main__":
    main()
