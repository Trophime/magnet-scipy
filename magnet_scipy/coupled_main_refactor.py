#!/usr/bin/env python3
"""
magnet_scipy/coupled_main_refactored.py

Refactored coupled circuits main function using modular components
Replaces the monolithic coupled_main.py with clean, focused components
"""

import sys
from typing import Optional, List

from .cli_core import (
    ArgumentParser, 
    ConfigurationLoader, 
    WorkingDirectoryManager,
    SampleConfigGenerator,
    ValidationHelper,
    create_time_parameters_from_args,
    create_output_options_from_args,
    handle_common_cli_tasks,
    print_simulation_header
)
from .cli_simulation import (
    SimulationOrchestrator,
    ResultProcessor,
    PlottingManager,
    AnalyticsManager,
    FileManager,
    SimulationSummary,
    CLIErrorHandler
)
from .coupled_circuits import CoupledRLCircuitsPID


def create_coupled_circuits_parser():
    """Create argument parser for coupled circuits simulation"""
    parser = ArgumentParser.create_base_parser(
        "Coupled RL Circuits PID Control Simulation"
    )
    
    # Add version info
    parser.add_argument(
        "--version",
        action="version",
        version="Coupled RL Circuits PID Simulation 2.0 (Refactored)",
    )
    
    # Coupled circuits specific arguments
    parser.add_argument(
        "--value_start",
        nargs="+",
        type=float,
        default=[0.0],
        help="Current values at start time in Ampere (one per circuit)",
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
    
    # Add common argument groups
    ArgumentParser.add_time_arguments(parser)
    ArgumentParser.add_output_arguments(parser)
    
    return parser


def handle_strategy_listing():
    """Handle --list-strategies command"""
    from .simulation_strategies import SimulationRunner
    
    print("Available simulation strategies for coupled circuits:")
    print("  voltage: Voltage-driven simulation using regular ODE")
    print("    - current: All circuit currents in Amperes")
    print("  pid: PID control simulation with reference tracking")
    print("    - current: All circuit currents in Amperes")
    print("    - integral_error: PID integral errors in Ampere-seconds")
    print("  auto: Automatically detect appropriate strategy")
    
    return 0  # Exit successfully


def validate_coupled_circuits_arguments(args) -> int:
    """Validate command line arguments for coupled circuits simulation"""
    # Validate configuration file is required
    if not args.config_file:
        if args.create_sample:
            return 0  # Will be handled by common tasks
        print("✗ Configuration file must be provided with --config-file")
        return 1
    
    # Validate file exists
    try:
        ValidationHelper.validate_file_exists(args.config_file, "Configuration file")
    except (ValueError, FileNotFoundError) as e:
        return CLIErrorHandler.handle_validation_error(e, args.debug)
    
    return 0  # Success


def load_and_validate_coupled_system(config_file: str, initial_values: List[float]):
    """
    Load coupled system configuration and validate consistency
    Separated from main workflow for clarity
    """
    # Load circuits and mutual inductances
    circuits, mutual_inductances = ConfigurationLoader.load_coupled_circuits(config_file)
    
    # Validate mutual inductances
    if mutual_inductances is None:
        raise RuntimeError(
            "Mutual inductance matrix must be provided in configuration file."
        )
    
    # Create coupled system
    coupled_system = CoupledRLCircuitsPID(
        circuits, mutual_inductances=mutual_inductances
    )
    
    # Validate initial values count matches circuit count
    ValidationHelper.validate_initial_values(initial_values, coupled_system.n_circuits)
    
    return coupled_system


def run_coupled_circuits_workflow(args) -> int:
    """
    Main workflow for coupled circuits simulation
    Replaces the complex logic from original coupled_main.py
    """
    try:
        # Create time and output parameters
        time_params = create_time_parameters_from_args(args)
        output_options = create_output_options_from_args(args)
        
        print_simulation_header("Coupled RL Circuits PID Simulation", args.config_file)
        
        # Load and validate system
        print(f"Loading configuration from: {args.config_file}")
        coupled_system = load_and_validate_coupled_system(
            args.config_file, args.value_start
        )
        
        # Create components
        orchestrator = SimulationOrchestrator()
        processor = ResultProcessor()
        plotter = PlottingManager()
        analytics = AnalyticsManager()
        file_manager = FileManager()
        
        # Run simulation
        simulation_result = orchestrator.run_coupled_simulation(
            coupled_system, time_params, args.value_start, args.strategy
        )
        
        if not simulation_result.success:
            SimulationSummary.print_error_summary(simulation_result)
            return 1
        
        # Process results
        t, processed_results = processor.process_coupled_circuits_result(
            simulation_result, coupled_system, output_options
        )
        
        # Generate plots
        plotter.plot_coupled_circuits_results(
            coupled_system, t, processed_results,
            simulation_result.strategy_type, output_options
        )
        
        # Show analytics
        analytics.analyze_coupled_circuits_results(
            coupled_system, t, processed_results, output_options
        )
        
        # Save results
        file_manager.save_coupled_circuits_results(
            coupled_system, t, processed_results, output_options, args.config_file
        )
        
        # Print summary
        SimulationSummary.print_coupled_circuits_summary(
            simulation_result, coupled_system, time_params
        )
        
        return 0  # Success
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def main():
    """
    Refactored main function for coupled circuits simulation
    Much cleaner and more maintainable than the original
    """
    
    # Parse arguments
    parser = create_coupled_circuits_parser()
    args = parser.parse_args()
    
    # Handle working directory context
    with WorkingDirectoryManager(args.wd):
        
        # Handle common CLI tasks that might exit early
        if not handle_common_cli_tasks(args):
            # Special case for coupled circuits sample creation
            if hasattr(args, 'create_sample') and args.create_sample:
                SampleConfigGenerator.create_coupled_circuits_config()
            return 0
        
        # Handle strategy listing
        if hasattr(args, 'list_strategies') and args.list_strategies:
            return handle_strategy_listing()
        
        # Validate arguments
        validation_result = validate_coupled_circuits_arguments(args)
        if validation_result != 0:
            return validation_result
        
        # Run main workflow
        return run_coupled_circuits_workflow(args)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
