#!/usr/bin/env python3
"""
magnet_scipy/coupled_main_refactor.py

Version 3.0: Clean coupled circuits main function with backward compatibility removed
Breaking changes: Removed legacy class imports, updated version number, simplified workflow
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
from .coupled_circuits import CoupledRLCircuitsPID


def create_coupled_circuits_parser():
    """Create argument parser for coupled circuits simulation"""
    parser = ArgumentParser.create_base_parser(
        "Coupled RL Circuits PID Control Simulation"
    )
    
    # Version 3.0: Updated version number
    parser.add_argument(
        "--version",
        action="version",
        version="Coupled RL Circuits PID Simulation 3.0 (Breaking Changes)",
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
    Version 3.0: Simplified workflow for coupled circuits simulation with integrated plotting
    
    This represents the key improvement: unified workflow that eliminates 
    separate plotting, analytics, and file management steps using only essential components
    """
    try:
        # Create time and output parameters
        time_params = create_time_parameters_from_args(args)
        output_options = create_output_options_from_args(args)
        plot_config = create_plotting_config_from_args(args)
        
        print_simulation_header("Coupled RL Circuits PID Simulation", args.config_file)
        
        # Load and validate system
        print(f"Loading configuration from: {args.config_file}")
        coupled_system = load_and_validate_coupled_system(
            args.config_file, args.value_start
        )
        
        # Handle benchmark mode
        if output_options.benchmark_plotting:
            return run_benchmark_workflow(coupled_system, time_params, output_options, plot_config, args)
        
        # Handle comparison mode
        if output_options.comparison_mode:
            return run_comparison_workflow(coupled_system, time_params, output_options, plot_config, args)
        
        # Standard simulation workflow
        print(f"✓ Coupled system loaded: {coupled_system.n_circuits} circuits")
        for i, circuit in enumerate(coupled_system.circuits):
            print(f"  Circuit {i+1} ({circuit.circuit_id}): L = {circuit.L:.3f} H, R = {circuit.get_resistance(0, circuit.temperature):.3f} Ω at I=0 A, T={circuit.temperature}°C")
        
        # Version 3.0: Use only essential classes - SimulationOrchestrator
        orchestrator = SimulationOrchestrator()
        simulation_result = orchestrator.run_coupled_simulation(
            coupled_system, time_params, args.value_start, args.strategy
        )
        
        if not simulation_result.success:
            SimulationSummary.print_error_summary(simulation_result)
            return 1
        
        # Version 3.0: Use only EnhancedPlottingManager - no legacy classes
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        
        # Process and plot results (unified operation)
        print("Processing results and creating plots...")
        processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
            simulation_result, coupled_system, args.config_file
        )
        
        # Print summary (analytics are automatically displayed if requested)
        SimulationSummary.print_coupled_circuits_summary(
            simulation_result, coupled_system, time_params
        )
        
        # Print analytics summary if requested
        if output_options.show_analytics:
            print_analytics_summary(analytics, coupled_system)
        
        # Print coupling analysis if available
        if output_options.show_analytics and 'coupling_analysis' in analytics:
            print_coupling_analysis(analytics['coupling_analysis'])
        
        print("✓ Coupled circuits simulation completed successfully")
        return 0  # Success
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def run_benchmark_workflow(coupled_system, time_params, output_options, plot_config, args) -> int:
    """Run coupled simulation with plotting performance benchmarking"""
    print("🔍 Running coupled circuits plotting performance benchmark...")
    
    try:
        from .coupled_plotting import benchmark_coupled_plotting_performance
        
        orchestrator = SimulationOrchestrator()
        simulation_result = orchestrator.run_coupled_simulation(
            coupled_system, time_params, args.value_start, args.strategy
        )
        
        if not simulation_result.success:
            print(f"✗ Simulation failed: {simulation_result.error_message}")
            return 1
        
        # Benchmark the plotting performance
        timing_results = benchmark_coupled_plotting_performance(simulation_result, coupled_system)
        
        print("📊 Coupled Circuits Plotting Performance Benchmark Results:")
        for operation, timing in timing_results.items():
            print(f"  {operation}: {timing['mean']:.3f}s ± {timing['std']:.3f}s")
        
        return 0
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def run_comparison_workflow(coupled_system, time_params, output_options, plot_config, args) -> int:
    """Run comparison between different simulation strategies"""
    print("🔍 Running coupled circuits strategy comparison workflow...")
    
    try:
        from .simulation_strategies import SimulationRunner
        
        runner = SimulationRunner()
        strategies = runner.list_available_strategies()
        
        results = {}
        for strategy in strategies:
            print(f"Running {strategy} strategy...")
            
            orchestrator = SimulationOrchestrator()
            result = orchestrator.run_coupled_simulation(
                coupled_system, time_params, args.value_start, strategy
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
        solutions_and_systems = [(result, coupled_system) for result in results.values()]
        labels = list(results.keys())
        
        from .plotting import create_comparison_plots
        create_comparison_plots(
            solutions_and_systems, 
            labels=labels,
            save_path=output_options.save_plots,
            show=output_options.show_plots
        )
        
        print("✓ Coupled circuits strategy comparison completed successfully")
        return 0
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def print_analytics_summary(analytics: dict, coupled_system):
    """Print analytics summary for coupled circuits"""
    print("\n" + "="*60)
    print("COUPLED CIRCUITS ANALYTICS SUMMARY")
    print("="*60)
    
    for circuit in coupled_system.circuits:
        circuit_id = circuit.circuit_id
        if circuit_id in analytics:
            circuit_analytics = analytics[circuit_id]
            
            print(f"Circuit {circuit_id}:")
            
            if 'performance_metrics' in circuit_analytics:
                metrics = circuit_analytics['performance_metrics']
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            
            if 'experimental_comparison' in circuit_analytics:
                exp_comparison = circuit_analytics['experimental_comparison']
                print(f"  Experimental Data Comparison:")
                for exp_name, comparison in exp_comparison.items():
                    for metric, value in comparison.items():
                        print(f"    {exp_name} {metric}: {value}")
            
            print()  # Add spacing between circuits
    
    print("="*60)


def print_coupling_analysis(coupling_analysis: dict):
    """Print coupling analysis results"""
    print("\n" + "="*60)
    print("MAGNETIC COUPLING ANALYSIS")
    print("="*60)
    
    # Mutual inductance effects
    if 'mutual_inductance_effects' in coupling_analysis:
        effects = coupling_analysis['mutual_inductance_effects']
        print(f"  Mutual Inductance Effects:")
        for effect in effects:
            print(f"    - {effect}")
    
    # Cross-coupling metrics
    if 'cross_coupling' in coupling_analysis:
        cross_coupling = coupling_analysis['cross_coupling']
        print(f"  Cross-Coupling Metrics:")
        print(f"    - Maximum coupling strength: {cross_coupling.get('max_coupling', 'N/A'):.3f}")
        print(f"    - Average coupling: {cross_coupling.get('avg_coupling', 'N/A'):.3f}")
    
    # Stability analysis
    if 'stability' in coupling_analysis:
        stability = coupling_analysis['stability']
        print(f"  System Stability:")
        print(f"    - System stable: {stability.get('is_stable', 'Unknown')}")
        print(f"    - Dominant pole: {stability.get('dominant_pole', 'N/A')}")
    
    print("="*60)


def main():
    """
    Version 3.0: Clean main function for coupled circuits simulation
    Demonstrates the simplified workflow achieved through removing backward compatibility
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
        
        # Run simplified workflow
        return run_coupled_circuits_workflow(args)


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
# - Old: "Coupled RL Circuits PID Simulation 2.0 (Fully Integrated)"
# - New: "Coupled RL Circuits PID Simulation 3.0 (Breaking Changes)"
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
#
# CONSISTENT DATA STRUCTURES:
# All modules now use the enhanced SimulationResult from cli_simulation.py
# with unified fields: circuit_ids, success, error_message for proper
# error handling and tracking across single and coupled circuit simulations.