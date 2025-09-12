#!/usr/bin/env python3
"""
magnet_scipy/main_refactor.py

Version 3.0: Clean single circuit main function with backward compatibility removed
Breaking changes: Removed legacy class imports, updated version number, simplified workflow
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
    create_plotting_config_from_args,
    handle_common_cli_tasks,
    print_simulation_header
)
# Version 3.0: Only import essential classes - removed backward compatibility classes
from .cli_simulation import (
    SimulationOrchestrator,
    SimulationSummary,
    CLIErrorHandler
)
from .cli_plotting_integration import EnhancedPlottingManager


def create_single_circuit_parser():
    """Create argument parser for single circuit simulation"""
    parser = ArgumentParser.create_base_parser(
        "Single RL Circuit PID Control Simulation"
    )
    
    # Version 3.0: Updated version number
    parser.add_argument(
        "--version",
        action="version",
        version="Single RL Circuit PID Simulation 3.0 (Breaking Changes)",
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


def load_and_validate_single_circuit(config_file: str, initial_value: float):
    """
    Load single circuit configuration and validate consistency
    Separated from main workflow for clarity
    """
    # Load circuit
    circuit = ConfigurationLoader.load_single_circuit(config_file)
    
    # Validate circuit configuration
    circuit.validate()
    
    return circuit


def run_single_circuit_workflow(args) -> int:
    """
    Version 3.0: Simplified workflow for single circuit simulation with integrated plotting
    
    This represents the key improvement: unified workflow that eliminates 
    separate plotting, analytics, and file management steps using only essential components
    """
    try:
        # Create time and output parameters
        time_params = create_time_parameters_from_args(args)
        output_options = create_output_options_from_args(args)
        plot_config = create_plotting_config_from_args(args)
        
        print_simulation_header("Single RL Circuit PID Simulation", args.config_file)
        
        # Load and validate system
        print(f"Loading configuration from: {args.config_file}")
        circuit = load_and_validate_single_circuit(args.config_file, args.value_start)
        
        # Handle benchmark mode
        if output_options.benchmark_plotting:
            return run_benchmark_workflow(circuit, time_params, output_options, plot_config, args)
        
        # Handle comparison mode
        if output_options.comparison_mode:
            return run_comparison_workflow(circuit, time_params, output_options, plot_config, args)
        
        # Standard simulation workflow
        print(f"✓ Circuit loaded: {circuit.circuit_id}")
        print(f"  L = {circuit.L:.3f} H, R = {circuit.get_resistance(0, circuit.temperature):.3f} Ω at I=0 A, T={circuit.temperature}°C")
        
        # Version 3.0: Use only essential classes - SimulationOrchestrator
        orchestrator = SimulationOrchestrator()
        simulation_result = orchestrator.run_single_circuit_simulation(
            circuit, time_params, args.strategy
        )
        
        if not simulation_result.success:
            SimulationSummary.print_error_summary(simulation_result)
            return 1
        
        # Version 3.0: Use only EnhancedPlottingManager - no legacy classes
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        
        # Process and plot results (unified operation)
        print("Processing results and creating plots...")
        processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
            simulation_result, circuit, args.config_file
        )
        
        # Print summary (analytics are automatically displayed if requested)
        SimulationSummary.print_single_circuit_summary(
            simulation_result, circuit, time_params
        )
        
        # Print analytics summary if requested
        if output_options.show_analytics and analytics:
            print_analytics_summary(analytics, circuit)
        
        print("✓ Single circuit simulation completed successfully")
        return 0  # Success
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def run_benchmark_workflow(circuit, time_params, output_options, plot_config, args) -> int:
    """Run single circuit simulation with plotting performance benchmarking"""
    print("🔍 Running single circuit plotting performance benchmark...")
    
    try:
        from .plotting import benchmark_plotting_performance
        
        orchestrator = SimulationOrchestrator()
        simulation_result = orchestrator.run_single_circuit_simulation(
            circuit, time_params, args.strategy
        )
        
        if not simulation_result.success:
            print(f"✗ Simulation failed: {simulation_result.error_message}")
            return 1
        
        # Benchmark the plotting performance
        timing_results = benchmark_plotting_performance(simulation_result, circuit)
        
        print("📊 Plotting Performance Benchmark Results:")
        for operation, timing in timing_results.items():
            print(f"  {operation}: {timing['mean']:.3f}s ± {timing['std']:.3f}s")
        
        return 0
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def run_comparison_workflow(circuit, time_params, output_options, plot_config, args) -> int:
    """Run comparison between different simulation strategies"""
    print("🔍 Running strategy comparison workflow...")
    
    try:
        from .single_circuit_adapter import SingleCircuitSimulationRunner
        
        runner = SingleCircuitSimulationRunner()
        strategies = runner.list_available_strategies()
        
        results = {}
        for strategy in strategies:
            print(f"Running {strategy} strategy...")
            
            orchestrator = SimulationOrchestrator()
            result = orchestrator.run_single_circuit_simulation(
                circuit, time_params, strategy
            )
            
            if result.success:
                results[strategy] = result
            else:
                print(f"  ✗ {strategy} failed: {result.error_message}")
        
        if len(results) < 2:
            print("✗ Need at least 2 successful strategies for comparison")
            return 1
        
        # Create comparison plots
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        
        # Convert results to format expected by comparison plotting
        solutions_and_systems = [(result, circuit) for result in results.values()]
        labels = list(results.keys())
        
        from .plotting import create_comparison_plots
        create_comparison_plots(
            solutions_and_systems, 
            labels=labels,
            save_path=output_options.save_plots,
            show=output_options.show_plots
        )
        
        print("✓ Strategy comparison completed successfully")
        return 0
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def print_analytics_summary(analytics: dict, circuit):
    """Print analytics summary for single circuit"""
    print("\n" + "="*50)
    print("ANALYTICS SUMMARY")
    print("="*50)
    
    circuit_id = circuit.circuit_id
    if circuit_id in analytics:
        circuit_analytics = analytics[circuit_id]
        
        if 'performance_metrics' in circuit_analytics:
            metrics = circuit_analytics['performance_metrics']
            print(f"Performance Metrics for {circuit_id}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        if 'experimental_comparison' in circuit_analytics:
            exp_comparison = circuit_analytics['experimental_comparison']
            print(f"Experimental Data Comparison:")
            for exp_name, comparison in exp_comparison.items():
                print(f"  {exp_name}:")
                for metric, value in comparison.items():
                    print(f"    {metric}: {value}")
    
    print("="*50)


def main():
    """
    Version 3.0: Clean main function for single circuit simulation
    Demonstrates the simplified workflow achieved through removing backward compatibility
    """
    
    # Parse arguments
    parser = create_single_circuit_parser()
    args = parser.parse_args()
    
    # Handle working directory context
    with WorkingDirectoryManager(args.wd):
        
        # Handle common CLI tasks that might exit early
        if not handle_common_cli_tasks(args):
            # Special case for single circuit sample creation
            if hasattr(args, 'create_sample') and args.create_sample:
                SampleConfigGenerator.create_single_circuit_config()
            return 0
        
        # Handle strategy listing
        if hasattr(args, 'list_strategies') and args.list_strategies:
            return handle_strategy_listing()
        
        # Validate arguments
        validation_result = validate_single_circuit_arguments(args)
        if validation_result != 0:
            return validation_result
        
        # Run simplified workflow
        return run_single_circuit_workflow(args)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


# Version 3.0 Breaking Changes Summary
#
# REMOVED IMPORTS:
# - ResultProcessor → Use EnhancedPlottingManager.create_plots_from_simulation_result()
# - PlottingManager → Use EnhancedPlottingManager.create_plots()  
# - AnalyticsManager → Analytics integrated into plotting workflow
# - FileManager → File saving integrated into plotting workflow
#
# UPDATED VERSION:
# - Old: "Single RL Circuit PID Simulation 2.0 (Fully Integrated)"
# - New: "Single RL Circuit PID Simulation 3.0 (Breaking Changes)"
#
# SIMPLIFIED WORKFLOW:
# The workflow now uses only essential components:
# 1. SimulationOrchestrator for running simulations
# 2. EnhancedPlottingManager for all plotting, analytics, and file saving
# 3. SimulationSummary for reporting
# 4. CLIErrorHandler for error handling
#
# All legacy backward compatibility components have been removed.
# The CLI still works exactly the same for end users, but the internal 
# API is now clean and modern with no backward compatibility cruft.