#!/usr/bin/env python3
"""
Basic RL Circuit PID Control System

Simple command-line interface for running RL circuit simulations with
adaptive PID control. Supports loading resistance and reference data from CSV files.
"""

import argparse
import sys
import pandas as pd
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


def run_simulation(args):
    """Run the main simulation with given arguments"""

    print("\n=== RL Circuit PID Simulation ===")
    print(f"Inductance: {args.inductance} H")
    if args.resistance_csv:
        print(f"Resistance: {args.resistance_csv} (variable with I,T)")
    else:
        print(f"Resistance: {args.resistance} Ω")
    print(f"Temperature: {args.temperature} °C")
    print(f"Time span: {args.time_start} to {args.time_end} seconds")

    # Load experimental data if provided
    experimental_data = None
    if args.experimental_csv:
        experimental_data = pd.read_csv(
            args.experimental_csv, sep=None, engine="python"
        )

        key = "voltage"
        if args.voltage_csv:
            key = "current"

        # Validate required columns
        if (
            "time" not in experimental_data.columns
            or key not in experimental_data.columns
        ):
            raise ValueError(f"Experimental CSV must contain 'time' and {key} columns")
        print(f"✓ Experimental data loaded: {len(experimental_data['time'])} points")
        print(f"experimental_data: {experimental_data}")

    # Create PID controller
    pid_controller = None
    if not args.voltage_csv:
        if args.custom_pid:
            print("Using custom PID parameters")
            pid_controller = create_adaptive_pid_controller(
                Kp_low=args.kp_low,
                Ki_low=args.ki_low,
                Kd_low=args.kd_low,
                Kp_medium=args.kp_medium,
                Ki_medium=args.ki_medium,
                Kd_medium=args.kd_medium,
                Kp_high=args.kp_high,
                Ki_high=args.ki_high,
                Kd_high=args.kd_high,
                low_threshold=args.low_threshold,
                high_threshold=args.high_threshold,
            )
        else:
            raise RuntimeError(
                "Defining a PID is required -- add --custom_pid option with parameters"
            )

    # Create circuit
    circuit = RLCircuitPID(
        circuit_id=args.circuit_id,
        R=args.resistance,
        L=args.inductance,
        pid_controller=pid_controller,
        reference_csv=args.reference_csv,
        voltage_csv=args.voltage_csv,
        resistance_csv=args.resistance_csv,
        temperature=args.temperature,
    )

    # Print configuration
    circuit.print_configuration()

    # Time parameters
    t0, t1 = args.time_start, args.time_end
    dt = args.time_step

    # Regular ODE
    post_data = None
    mode = None
    if args.voltage_csv:
        mode = "regular"
        print(f"\nUsing input voltage CSV: {args.voltage_csv}")
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
            method="RK45",
            dense_output=True,
            rtol=1e-6,
            atol=1e-9,
            max_step=dt,
        )

        print("✓ Simulation completed")
        # Post-process results
        print("Processing results...")
        t, results = prepare_post(sol, circuit, mode, experimental_data)

        print("Generating plots...")
        plot_vresults(
            sol,
            circuit,
            t,
            results,
            experimental_data,
            save_path=args.save_plots,
            show=args.show_plots,
        )
    else:
        mode = "cde"

        # Store results
        all_t = []
        all_y = []

        print(f"\nUsing reference current CSV: {args.reference_csv}")
        # Initial conditions [current, integral_error, prev_error]
        i0 = args.value_start
        print(f"\nInitial current: {i0} A at t={t0} s")
        i0_ref = circuit.reference_current(t0)
        print(f"init ref: {i0_ref:.3f} A at t={t0:.3f} s")
        if experimental_data is not None:
            v0 = np.interp(t0, experimental_data["time"], experimental_data["voltage"])
            print(f"init exp: {v0:.3f} V at t={t0:.3f} s")

        closest_index = np.argmin(np.abs(circuit.time_data - t0))
        # shall check if circuit.time_data[closest_index] is greater than t0

        error = i0_ref - i0  # ??how to initial integral error??
        print(
            f"t0: {t0}, closest_index={closest_index}, time_data={circuit.time_data[closest_index]}, error={error}"
        )
        print(
            f"\nPID controller: run pid for each sequence of reference_current ({len(circuit.time_data)-1} sequences)..."
        )
        stop = False
        for n in range(closest_index + 1, len(circuit.time_data) - 1):
            t_actual = circuit.time_data[n]
            if circuit.time_data[n] > args.time_end:
                t_actual = args.time_end
                stop = True
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
                method="RK45",
                dense_output=True,
                rtol=1e-6,
                atol=1e-9,
                max_step=dt,
            )
            print(f"tfinal={float(sol.t[-1])} s", end=",", flush=True)
            print(f"i1={float(sol.y[0, -1])} A", end=", ", flush=True)
            print(f"integral_error1={float(sol.y[1, -1])} A", end=", ", flush=True)
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
                t, results = prepare_post(sol, circuit, mode, experimental_data)

                print("Generating plots...")
                plot_results(
                    sol,
                    circuit,
                    t,
                    results,
                    experimental_data,
                    save_path=None,
                    show=True,
                )
            if stop:
                break

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
        t, results = prepare_post(
            fake_sol(t_global, y_global), circuit, mode, experimental_data
        )

        print("Generating plots...")
        plot_results(
            sol,
            circuit,
            t,
            results,
            experimental_data,
            save_path=args.save_plots,
            show=args.show_plots,
        )

    # Save results if requested
    if args.save_results:
        print("save result to {args.save_results}")
        save_results(circuit, t, results, args.save_results)

    return sol, circuit, post_data


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

    # Input files
    parser.add_argument(
        "--circuit_id",
        type=str,
        help="Circuit id (ex: M9Bitters)",
        default="M9Bitters",
    )

    # Input files
    parser.add_argument(
        "--reference_csv",
        type=str,
        help="Path to CSV file with reference current data (columns: time, current)",
    )

    parser.add_argument(
        "--voltage_csv",
        type=str,
        help="Path to CSV file with input voltage data (columns: time, voltage)",
    )

    parser.add_argument(
        "--resistance_csv",
        type=str,
        help="Path to CSV file with resistance data (columns: current, temperature, resistance)",
    )

    parser.add_argument(
        "--experimental_csv",
        "-e",
        type=str,
        help="Path to CSV file with experimental voltage data for comparison (columns: time, voltage)",
    )

    # Circuit parameters
    parser.add_argument(
        "--inductance",
        type=float,
        default=0.1,
        help="Circuit inductance in Henry",
    )

    parser.add_argument(
        "--resistance",
        type=float,
        default=1.5,
        help="Constant resistance in Ohms (used if --resistance-csv not provided)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=25.0,
        help="Operating temperature in Celsius",
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

    # PID parameters (optional custom settings)
    parser.add_argument(
        "--custom_pid",
        action="store_true",
        help="Use custom PID parameters instead of defaults",
    )

    parser.add_argument(
        "--kp_low", type=float, default=20.0, help="Kp for low current region"
    )
    parser.add_argument(
        "--ki_low", type=float, default=15.0, help="Ki for low current region"
    )
    parser.add_argument(
        "--kd_low", type=float, default=0.1, help="Kd for low current region"
    )

    parser.add_argument(
        "--kp_medium", type=float, default=12.0, help="Kp for medium current region"
    )
    parser.add_argument(
        "--ki_medium", type=float, default=8.0, help="Ki for medium current region"
    )
    parser.add_argument(
        "--kd_medium", type=float, default=0.05, help="Kd for medium current region"
    )

    parser.add_argument(
        "--kp_high", type=float, default=8.0, help="Kp for high current region"
    )
    parser.add_argument(
        "--ki_high", type=float, default=5.0, help="Ki for high current region"
    )
    parser.add_argument(
        "--kd_high", type=float, default=0.02, help="Kd for high current region"
    )

    parser.add_argument(
        "--low_threshold",
        type=float,
        default=60.0,
        help="Low to medium current threshold",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=800.0,
        help="Medium to high current threshold",
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
    # print(f"args: {args}")

    if args.wd is not None:
        import os

        pwd = os.getcwd()
        os.chdir(args.wd)
        print(f"Working directory set to: {args.wd} (pwd={pwd})")
    print(f"✓ Working directory: {os.getcwd()}")

    if args.save_plots and args.show_plots:
        raise RuntimeError(
            "⚠️ Warning: Both --save-plots and --show-plots specified. Plots will be saved and shown."
        )
    if not args.save_plots and not args.show_plots:
        print(
            "⚠️ Warning: Neither --save-plots nor --show-plots specified. Force show_plots."
        )
        args.show_plots = True

    # Check if files exist when specified
    if args.reference_csv:
        try:
            pd.read_csv(args.reference_csv, sep=None, engine="python")
            print(f"✓ Reference CSV loaded: {args.reference_csv}")
        except FileNotFoundError:
            print(f"❌ Reference CSV file not found: {args.reference_csv}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading reference CSV: {e}")
            sys.exit(1)

    # Check if files exist when specified
    if args.voltage_csv:
        try:
            pd.read_csv(args.voltage_csv, sep=None, engine="python")
            print(f"✓ Input Voltage CSV loaded: {args.voltage_csv}")
        except FileNotFoundError:
            print(f"❌ Input voltage CSV file not found: {args.voltage_csv}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading reference CSV: {e}")
            sys.exit(1)

    if args.resistance_csv:
        try:
            pd.read_csv(args.resistance_csv, sep=None, engine="python")
            print(f"✓ Resistance CSV loaded: {args.resistance_csv}")
        except FileNotFoundError:
            print(f"❌ Resistance CSV file not found: {args.resistance_csv}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading resistance CSV: {e}")
            sys.exit(1)

    if args.experimental_csv:
        try:
            pd.read_csv(args.experimental_csv, sep=None, engine="python")
            print(f"✓ Experimental CSV loaded: {args.experimental_csv}")
        except FileNotFoundError:
            print(f"❌ Experimental CSV file not found: {args.experimental_csv}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading experimental CSV: {e}")
            sys.exit(1)

    # Run simulation
    sol, circuit, post_data = run_simulation(args)
    print("\n✓ Simulation completed successfully!")

    if args.wd is not None:
        os.chdir(pwd)
        print(f"Returned to original directory: {pwd}")


if __name__ == "__main__":
    main()
