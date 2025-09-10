#!/usr/bin/env python3
"""
Updated RL Circuit PID Control System for Single Circuits

Command-line interface for running single RL circuit simulations with
adaptive PID control using the Strategy Pattern. Supports loading resistance 
and reference data from CSV files.
"""

import argparse
import sys
import numpy as np
import json
import os

from .pid_controller import create_adaptive_pid_controller
from .rlcircuitpid import RLCircuitPID
from .simulation_strategies import SimulationParameters
from .single_circuit_adapter import SingleCircuitSimulationRunner
from .plotting import (
    prepare_post,
    plot_vresults,
    plot_results,
    save_results,
)


def load_single_circuit_configuration(config_file: str) -> RLCircuitPID:
    """Load single circuit configuration from JSON file"""
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
            circuit_id=circuit_data.get("circuit_id", "single_circuit"),
            experimental_data=circuit_data.get("experiment_data", []),
        )
        
        return circuit

    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


def validate_time_range(circuit: RLCircuitPID, args) -> tuple:
    """
    Validate and adjust time range based on available data
    Returns (t_start, t_end, time_range_modified)
    """
    t_start, t_end = args.time_start, args.time_end
    time_range_modified = False
    
    # Check if circuit has time-dependent data
    if hasattr(circuit, 'time_data') and circuit.time_data is not None:
        data_t_min = float(circuit.time_data[0])
        data_t_max = float(circuit.time_data[-1])
        
        if t_start < data_t_min:
            print(f"⚠️ Warning: Requested start time {t_start} is before data start {data_t_min}")
            t_start = data_t_min
            time_range_modified = True
            
        if t_end > data_t_max:
            print(f"⚠️ Warning: Requested end time {t_end} is after data end {data_t_max}")
            t_end = data_t_max
            time_range_modified = True
    
    return t_start, t_end, time_range_modified


def run_single_circuit_simulation_with_strategy(args):
    """
    Run single circuit simulation using strategy pattern
    Replaces the complex conditional logic from the original main.py
    """
    print("\n=== Single RL Circuit PID Simulation ===")
    
    # Load circuit configuration
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        circuit = load_single_circuit_configuration(args.config_file)
    else:
        raise RuntimeError("Configuration file must be provided with --config-file")

    # Validate circuit configuration
    circuit.validate()
    circuit.print_configuration()
    
    # Validate and adjust time range
    t_start, t_end, time_modified = validate_time_range(circuit, args)
    if time_modified:
        print(f"✓ Adjusted time range: {t_start} to {t_end} seconds")
    else:
        print(f"✓ Time range: {t_start} to {t_end} seconds")

    # Create simulation parameters
    params = SimulationParameters(
        t_start=t_start,
        t_end=t_end,
        dt=args.time_step,
        method=args.method,
        initial_values=[args.value_start] if args.value_start else [0.0]
    )
    
    print(f"\nSimulation parameters:")
    print(f"  Time span: {params.t_start} to {params.t_end} seconds")
    print(f"  Time step: {params.dt} seconds")
    print(f"  Method: {params.method}")
    print(f"  Initial current: {params.initial_values[0]} A")
    
    # Run simulation using strategy pattern
    runner = SingleCircuitSimulationRunner()
    
    try:
        # Determine simulation strategy
        if args.strategy == "auto":
            strategy_name = runner.detect_strategy(circuit)
            print(f"✓ Auto-detected simulation strategy: {strategy_name}")
        else:
            strategy_name = args.strategy
            print(f"✓ Using specified simulation strategy: {strategy_name}")
        
        # List available strategies for user information
        available_strategies = runner.list_available_strategies()
        print(f"✓ Available strategies: {', '.join(available_strategies)}")
        
        result = runner.run_simulation(circuit, params, strategy_name)
        print(f"✓ Simulation completed using {result.strategy_type} strategy")
        print(f"  Time points: {len(result.time)}")
        print(f"  Function evaluations: {result.metadata['n_evaluations']}")
        
        # Convert result back to expected format for plotting compatibility
        sol = convert_result_to_sol_format(result)
        
        # Determine mode for post-processing (temporary compatibility)
        mode = "regular" if result.strategy_type == "voltage_input" else "cde"
        
        # Post-process results
        print("Processing results...")
        t, results = prepare_post(sol, circuit, mode)
        
        # Generate plots
        print("Generating plots...")
        if result.strategy_type == "voltage_input":
            plot_vresults(
                circuit,
                t,
                results,
                save_path=args.save_plots,
                show=args.show_plots,
            )
        else:
            plot_results(
                sol,
                circuit,
                t,
                results,
                save_path=args.save_plots,
                show=args.show_plots,
            )
        
        # Show analytics if requested
        if args.show_analytics:
            from .plotting import analyze
            analyze(circuit, t, results)
        
        # Save results
        output_filename = args.config_file.replace(".json", ".res") 
        if args.save_results:
            output_filename = args.save_results
        save_results(circuit, t, results, output_filename)
        
        return sol, circuit, t, results
        
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
            self.nfev = 0  # Will be updated from metadata
    
    sol = FakeSol(result.time, result.solution)
    if 'n_evaluations' in result.metadata:
        sol.nfev = result.metadata['n_evaluations']
    
    return sol


def create_sample_config(args):
    """Create a sample configuration file for testing"""
    config = {
        "circuit_id": "sample_circuit",
        "inductance": 0.1,
        "resistance": 1.5,
        "temperature": 25.0,
        "pid_params": {
            "Kp_low": 20.0,
            "Ki_low": 15.0,
            "Kd_low": 0.1,
            "Kp_medium": 12.0,
            "Ki_medium": 8.0,
            "Kd_medium": 0.05,
            "Kp_high": 8.0,
            "Ki_high": 5.0,
            "Kd_high": 0.02,
            "low_threshold": 60.0,
            "high_threshold": 800.0
        }
    }
    
    # Add CSV files if specified
    if hasattr(args, 'reference_csv') and args.reference_csv:
        config["reference_csv"] = args.reference_csv
    if hasattr(args, 'voltage_csv') and args.voltage_csv:
        config["voltage_csv"] = args.voltage_csv
    if hasattr(args, 'resistance_csv') and args.resistance_csv:
        config["resistance_csv"] = args.resistance_csv
    
    filename = "sample_single_circuit.json"
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created sample configuration: {filename}")
    return filename


def main():
    """Main function with argument parsing"""

    parser = argparse.ArgumentParser(
        description="Single RL Circuit PID Control Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration options
    parser.add_argument(
        "--version",
        action="version",
        version="Single RL Circuit PID Simulation 1.0",
    )
    
    parser.add_argument(
        "--wd", 
        type=str, 
        help="Working directory"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to JSON configuration file with circuit definition"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample configuration file and exit"
    )

    # Simulation parameters
    parser.add_argument(
        "--value_start",
        type=float,
        default=0.0,
        help="Current value at start time in Ampere",
    )

    parser.add_argument(
        "--time_start", 
        type=float, 
        default=0.0, 
        help="Simulation start time in seconds"
    )

    parser.add_argument(
        "--time_end", 
        type=float, 
        default=5.0, 
        help="Simulation end time in seconds"
    )

    parser.add_argument(
        "--time_step", 
        type=float, 
        default=0.001, 
        help="Simulation time step in seconds"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["voltage", "pid", "auto"],
        default="auto",
        help="Simulation strategy to use (auto-detect if not specified)"
    )

    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available simulation strategies and exit"
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
        help="Enable debug mode with additional output",
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Handle working directory
    original_wd = None
    if args.wd is not None:
        original_wd = os.getcwd()
        os.chdir(args.wd)
        print(f"✓ Working directory set to: {args.wd}")
    
    try:
        # Handle strategy listing
        if args.list_strategies:
            from .single_circuit_adapter import SingleCircuitSimulationRunner
            runner = SingleCircuitSimulationRunner()
            print("Available simulation strategies for single circuits:")
            for strategy_name in runner.list_available_strategies():
                info = runner.get_strategy_info(strategy_name)
                print(f"  {strategy_name}: {info['description']}")
                state_desc = info['state_description']
                for key, desc in state_desc.items():
                    print(f"    - {key}: {desc}")
            return
        
        # Handle sample creation
        if args.create_sample:
            create_sample_config(args)
            return
        
        # Handle legacy mode (create config from command line args)
        if not args.config_file and (args.reference_csv or args.voltage_csv):
            print("⚠️ Using legacy command line arguments - creating temporary config file")
            args.config_file = create_sample_config(args)
        
        # Validate configuration file
        if not args.config_file:
            print("✗ No configuration file specified. Use --config-file or --create-sample")
            parser.print_help()
            sys.exit(1)
            
        if not os.path.exists(args.config_file):
            print(f"✗ Configuration file not found: {args.config_file}")
            sys.exit(1)
        
        # Validate output options
        if args.save_plots and args.show_plots:
            print("⚠️ Both --save-plots and --show-plots specified. Plots will be saved and shown.")
        elif not args.save_plots and not args.show_plots:
            print("⚠️ Neither --save-plots nor --show-plots specified. Forcing show_plots.")
            args.show_plots = True

        print(f"✓ Configuration file: {args.config_file}")

        # Run simulation
        sol, circuit, t, results = run_single_circuit_simulation_with_strategy(args)
        
        print("\n✓ Single circuit simulation completed successfully!")
        print(f"  Circuit ID: {circuit.circuit_id}")
        print(f"  Time points: {len(sol.t)}")
        print(f"  Total simulation time: {float(sol.t[-1] - sol.t[0]):.3f} seconds")
        
        # Print strategy-specific information
        if hasattr(circuit, 'voltage_csv') and circuit.voltage_csv:
            print(f"  Strategy: Voltage-driven simulation")
        elif hasattr(circuit, 'reference_csv') and circuit.reference_csv:
            print(f"  Strategy: PID control simulation")

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        if args.debug:
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
