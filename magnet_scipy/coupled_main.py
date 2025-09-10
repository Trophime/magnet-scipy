#!/usr/bin/env python3
"""
Updated coupled_main.py using Strategy Pattern
Simplified and cleaner simulation logic
"""

import os
import argparse
import sys
import json
import numpy as np
from typing import List

from .coupled_circuits import CoupledRLCircuitsPID
from .rlcircuitpid import RLCircuitPID
from .simulation_strategies import SimulationRunner, SimulationParameters
from .coupled_plotting import (
    prepare_coupled_post,
    plot_coupled_vresults,
    plot_coupled_results,
    save_coupled_results,
)
from .pid_controller import create_adaptive_pid_controller
from .utils import fake_sol


def run_coupled_simulation_with_strategy(args):
    """
    Simplified simulation runner using strategy pattern
    Replaces the complex conditional logic
    """
    
    print("\n=== Coupled RL Circuits PID Simulation ===")
    
    # Load system configuration (unchanged)
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        circuits, mutual_inductances = load_circuit_configuration(args.config_file)
    else:
        raise RuntimeError("Configuration file must be provided with --config-file")
    
    # Create coupled system (unchanged)
    if mutual_inductances is not None:
        coupled_system = CoupledRLCircuitsPID(
            circuits, mutual_inductances=mutual_inductances
        )
    else:
        raise RuntimeError(
            "Mutual inductance matrix must be provided in configuration file."
        )
    
    coupled_system.print_configuration()
    
    # Validate initial values
    if len(args.value_start) != coupled_system.n_circuits:
        raise ValueError(
            f"Number of initial values ({len(args.value_start)}) must match number of circuits ({coupled_system.n_circuits})"
        )
    
    # Create simulation parameters
    params = SimulationParameters(
        t_start=args.time_start,
        t_end=args.time_end,
        dt=args.time_step,
        method=args.method,
        initial_values=args.value_start
    )
    
    print(f"\nSimulation parameters:")
    print(f"  Time span: {params.t_start} to {params.t_end} seconds")
    print(f"  Time step: {params.dt} seconds")
    print(f"  Method: {params.method}")
    print(f"  Initial values: {params.initial_values}")
    
    # Run simulation using strategy pattern
    runner = SimulationRunner()
    
    try:
        result = runner.run_simulation(coupled_system, params)
        print(f"✓ Simulation completed using {result.strategy_type} strategy")
        
        # Convert result back to expected format for plotting
        sol = convert_result_to_sol_format(result)
        
        # Determine mode for post-processing (temporary compatibility)
        mode = "regular" if result.strategy_type == "voltage_input" else "cde"
        
        # Post-process results
        print("Processing results...")
        t, results = prepare_coupled_post(sol, coupled_system, mode)
        
        # Generate plots
        print("Generating plots...")
        if result.strategy_type == "voltage_input":
            plot_coupled_vresults(
                sol,
                coupled_system,
                t,
                results,
                save_path=args.save_plots,
                show=args.show_plots,
            )
        else:
            plot_coupled_results(
                sol,
                coupled_system,
                t,
                results,
                save_path=args.save_plots,
                show=args.show_plots,
            )
        
        # Save results
        output_filename = args.config_file.replace(".json", ".res")
        if args.save_results:
            output_filename = args.save_results
        save_coupled_results(coupled_system, t, results, output_filename)
        
        return sol, coupled_system, t, results
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        raise


def convert_result_to_sol_format(result):
    """
    Convert SimulationResult back to scipy solve_ivp format for compatibility
    This is temporary until plotting functions are updated
    """
    class FakeSol:
        def __init__(self, t, y):
            self.t = t
            self.y = y
            self.success = True
            self.message = "Strategy simulation completed"
    
    return FakeSol(result.time, result.solution)


def load_circuit_configuration(config_file: str) -> List[RLCircuitPID]:
    """Load circuit configuration from JSON file (unchanged)"""
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
                R=circuit_data.get("resistance", 1.0),
                L=circuit_data.get("inductance", 0.1),
                pid_controller=pid_controller,
                reference_csv=circuit_data.get("reference_csv", None),
                voltage_csv=circuit_data.get("voltage_csv", None),
                resistance_csv=circuit_data.get("resistance_csv"),
                temperature=circuit_data.get("temperature", 25.0),
                temperature_csv=circuit_data.get("temperature_csv", None),
                circuit_id=circuit_data.get("circuit_id", f"circuit_{len(circuits)+1}"),
                experimental_data=circuit_data.get("experiment_data", []),
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


def main():
    """Main function with argument parsing (simplified)"""
    
    parser = argparse.ArgumentParser(
        description="Coupled RL Circuits PID Control Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--wd", 
        type=str, 
        help="Working directory"
    )

    # Configuration options
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to JSON configuration file with circuit definitions"
    )

    # Simulation parameters
    parser.add_argument(
        "--value_start",
        nargs="+",
        type=float,
        default=[0.0],
        help="Current values at start time in Ampere",
    )

    parser.add_argument(
        "--time_start", type=float, default=0.0, help="Simulation start time in seconds"
    )

    parser.add_argument(
        "--time_end", type=float, default=5.0, help="Simulation end time in seconds"
    )

    parser.add_argument(
        "--time_step", type=float, default=0.01, help="Simulation time step in seconds"
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
        "--save_results",
        type=str,
        help="Save results to specified file (e.g., results.npz)",
    )
    
    parser.add_argument(
        "--save_plots",
        type=str,
        help="Save plots to specified file (e.g., plots.png)",
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not args.save_plots and not args.show_plots:
        print("⚠️ Warning: Neither --save-plots nor --show-plots specified. Forcing show_plots.")
        args.show_plots = True

    # Handle working directory
    original_wd = None
    if args.wd is not None:
        original_wd = os.getcwd()
        os.chdir(args.wd)
        print(f"✓ Working directory set to: {args.wd}")

    # Run simulation
    try:
        sol, coupled_system, t, results = run_coupled_simulation_with_strategy(args)
        print("\n✓ Coupled simulation completed successfully!")
        print(f"  Circuits simulated: {coupled_system.n_circuits}")
        print(f"  Time points: {len(t)}")
        print(f"  Total simulation time: {float(t[-1] - t[0]):.3f} seconds")

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Restore original working directory
        if original_wd is not None:
            os.chdir(original_wd)
            print(f"✓ Returned to original directory: {original_wd}")


if __name__ == "__main__":
    main()
