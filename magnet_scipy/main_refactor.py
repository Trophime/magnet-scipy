#!/usr/bin/env python3
"""
magnet_scipy/main_refactored.py

Refactored single circuit main function using modular components
Replaces the monolithic main.py with clean, focused components
"""

import sys
from typing import Optional

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


def create_single_circuit_parser():
    """Create argument parser for single circuit simulation"""
    parser = ArgumentParser.create_base_parser(
        "Single RL Circuit PID Control Simulation"
    )
    
    # Add version info
    parser.add_argument(
        "--version",
        action="version",
        version="Single RL Circuit PID Simulation 2.0 (Refactored)",
    )
    
    # Single circuit specific arguments
    parser.add_argument(
        "--value_start",
        type=float,
        default=0.0,
        help="Current value at start time in Ampere",
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
    from .single_circuit_adapter import SingleCircuitSimulationRunner
    
    runner = SingleCircuitSimulationRunner()
    print("Available simulation strategies for single circuits:")
    
    for strategy_name in runner.list_available_strategies():
        info = runner.get_strategy_info(strategy_name)
        print(f"  {strategy_name}: {info['description']}")
        state_desc = info['state_description']
        for key, desc in state_desc.items():
            print(f"    - {key}: {desc}")
    
    return 0  # Exit successfully


def validate_single_circuit_arguments(args) -> int:
    """Validate command line arguments for single circuit simulation"""
    # Validate configuration file
    if not args.config_file:
        if args.create_sample:
            return 0  # Will be handled by common tasks
        print("✗ No configuration file specified. Use --config-file or --create-sample")
        return 1
    
    # Validate file exists
    try:
        ValidationHelper.validate_file_exists(args.config_file, "Configuration file")
    except (ValueError, FileNotFoundError) as e:
        return CLIErrorHandler.handle_validation_error(e, args.debug)
    
    return 0  # Success


def run_single_circuit_workflow(args) -> int:
    """
    Main workflow for single circuit simulation
    Replaces the complex logic from original main.py
    """
    try:
        # Create time and output parameters
        time_params = create_time_parameters_from_args(args)
        output_options = create_output_options_from_args(args)
        
        print_simulation_header("Single RL Circuit PID Simulation", args.config_file)
        
        # Load configuration
        print(f"Loading configuration from: {args.config_file}")
        circuit = ConfigurationLoader.load_single_circuit(args.config_file)
        
        # Create components
        orchestrator = SimulationOrchestrator()
        processor = ResultProcessor()
        plotter = PlottingManager()
        analytics = AnalyticsManager()
        file_manager = FileManager()
        
        # Run simulation
        simulation_result = orchestrator.run_single_circuit_simulation(
            circuit, time_params, args.strategy
        )
        
        if not simulation_result.success:
            SimulationSummary.print_error_summary(simulation_result)
            return 1
        
        # Process results
        t, processed_results = processor.process_single_circuit_result(
            simulation_result, circuit, output_options
        )
        
        # Generate plots
        plotter.plot_single_circuit_results(
            circuit, t, processed_results, 
            simulation_result.strategy_type, output_options
        )
        
        # Show analytics
        analytics.analyze_single_circuit_results(
            circuit, t, processed_results, output_options
        )
        
        # Save results
        file_manager.save_single_circuit_results(
            circuit, t, processed_results, output_options, args.config_file
        )
        
        # Print summary
        SimulationSummary.print_single_circuit_summary(
            simulation_result, circuit, time_params
        )
        
        return 0  # Success
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def main():
    """
    Refactored main function for single circuit simulation
    Much cleaner and more maintainable than the original
    """
    
    # Parse arguments
    parser = create_single_circuit_parser()
    args = parser.parse_args()
    
    # Handle working directory context
    with WorkingDirectoryManager(args.wd):
        
        # Handle common CLI tasks that might exit early
        if not handle_common_cli_tasks(args):
            return 0
        
        # Handle strategy listing
        if hasattr(args, 'list_strategies') and args.list_strategies:
            return handle_strategy_listing()
        
        # Validate arguments
        validation_result = validate_single_circuit_arguments(args)
        if validation_result != 0:
            return validation_result
        
        # Run main workflow
        return run_single_circuit_workflow(args)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
