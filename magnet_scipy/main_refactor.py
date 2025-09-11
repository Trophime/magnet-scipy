#!/usr/bin/env python3
"""
magnet_scipy/main_refactor.py

Fully integrated single circuit main function using the new plotting system
Demonstrates the simplified workflow with unified plotting, analytics, and saving
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
    
    # Add version info
    parser.add_argument(
        "--version",
        action="version",
        version="Single RL Circuit PID Simulation 2.0 (Fully Integrated)",
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
    Simplified workflow for single circuit simulation with integrated plotting
    
    This represents the key improvement: unified workflow that eliminates 
    separate plotting, analytics, and file management steps
    """
    try:
        # Create time and output parameters
        time_params = create_time_parameters_from_args(args)
        output_options = create_output_options_from_args(args)
        plot_config = create_plotting_config_from_args(args)
        
        print_simulation_header("Single RL Circuit PID Simulation", args.config_file)
        
        # Load configuration
        print(f"Loading configuration from: {args.config_file}")
        circuit = ConfigurationLoader.load_single_circuit(args.config_file)
        
        # Handle benchmark mode
        if output_options.benchmark_plotting:
            return run_benchmark_workflow(circuit, time_params, output_options, plot_config, args)
        
        # Handle comparison mode
        if output_options.comparison_mode:
            return run_comparison_workflow(circuit, time_params, output_options, plot_config, args)
        
        # Standard simulation workflow
        print(f"✓ Circuit loaded: {circuit.circuit_id}")
        print(f"  L = {circuit.L:.3f} H, R = {circuit.get_resistance(0, circuit.temperature):.3f} Ω at I=0 A, T={circuit.temperature}°C")
        
        # Run simulation
        orchestrator = SimulationOrchestrator()
        simulation_result = orchestrator.run_single_circuit_simulation(
            circuit, time_params, args.strategy
        )
        
        if not simulation_result.success:
            SimulationSummary.print_error_summary(simulation_result)
            return 1
        
        # Create integrated plotting manager
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
        if output_options.show_analytics:
            print_analytics_summary(analytics, circuit)
        
        print("✓ Single circuit simulation completed successfully")
        return 0  # Success
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def run_benchmark_workflow(circuit, time_params, output_options, plot_config, args) -> int:
    """Run simulation with plotting performance benchmarking"""
    print("🔍 Running plotting performance benchmark...")
    
    try:
        from .plotting import benchmark_plotting_performance
        
        orchestrator = SimulationOrchestrator()
        simulation_result = orchestrator.run_single_circuit_simulation(
            circuit, time_params, args.strategy
        )
        
        if not simulation_result.success:
            SimulationSummary.print_error_summary(simulation_result)
            return 1
        
        # Convert to solution format for benchmarking
        class FakeSol:
            def __init__(self, result):
                self.t = result.time
                self.y = result.solution
                self.success = result.success
        
        sol = FakeSol(simulation_result)
        
        # Run benchmark
        benchmark_results = benchmark_plotting_performance(sol, circuit, n_runs=5)
        
        print("📊 Plotting Performance Benchmark Results:")
        print(f"  Average plotting time: {benchmark_results['avg_time']:.3f} seconds")
        print(f"  Standard deviation: {benchmark_results['std_time']:.3f} seconds")
        print(f"  Min time: {benchmark_results['min_time']:.3f} seconds")
        print(f"  Max time: {benchmark_results['max_time']:.3f} seconds")
        
        # Still create the plots normally
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
            simulation_result, circuit, args.config_file
        )
        
        return 0
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def run_comparison_workflow(circuit, time_params, output_options, plot_config, args) -> int:
    """Run simulation in comparison mode with different strategies"""
    print("🔄 Running comparison mode with multiple strategies...")
    
    try:
        strategies = ["voltage", "pid"] if args.strategy == "auto" else [args.strategy]
        results = []
        labels = []
        
        orchestrator = SimulationOrchestrator()
        
        for strategy in strategies:
            print(f"  Running simulation with {strategy} strategy...")
            simulation_result = orchestrator.run_single_circuit_simulation(
                circuit, time_params, strategy
            )
            
            if simulation_result.success:
                results.append(simulation_result)
                labels.append(f"{strategy.title()} Control")
            else:
                print(f"    ⚠️ {strategy} strategy failed: {simulation_result.error_message}")
        
        if not results:
            print("✗ All simulation strategies failed")
            return 1
        
        # Create comparison plots
        from .plotting import create_comparison_plots
        
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        
        # Process each result
        processed_results = []
        for result, label in zip(results, labels):
            processed, analytics = plotting_manager.create_plots_from_simulation_result(
                result, circuit, show=False  # Don't show individual plots
            )
            processed_results.append((processed, label))
        
        # Create comparison plot
        comparison_fig = create_comparison_plots(
            processed_results,
            labels,
            save_path=output_options.save_plots,
            show=output_options.show_plots,
            config=plot_config
        )
        
        print(f"✓ Comparison simulation completed with {len(results)} strategies")
        return 0
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def print_analytics_summary(analytics: dict, circuit):
    """Print a summary of the analytics results"""
    print("\n📈 Analytics Summary:")
    
    circuit_id = circuit.circuit_id
    if circuit_id in analytics:
        circuit_analytics = analytics[circuit_id]
        
        # Performance metrics
        if 'performance' in circuit_analytics:
            perf = circuit_analytics['performance']
            print(f"  Performance Metrics:")
            print(f"    - Steady-state error: {perf.get('steady_state_error', 'N/A')}")
            print(f"    - Rise time: {perf.get('rise_time', 'N/A')} s")
            print(f"    - Settling time: {perf.get('settling_time', 'N/A')} s")
            print(f"    - Overshoot: {perf.get('overshoot', 'N/A')}%")
        
        # Statistical metrics
        if 'statistics' in circuit_analytics:
            stats = circuit_analytics['statistics']
            print(f"  Statistical Analysis:")
            print(f"    - Mean current: {stats.get('mean_current', 'N/A'):.3f} A")
            print(f"    - RMS current: {stats.get('rms_current', 'N/A'):.3f} A")
            print(f"    - Peak current: {stats.get('peak_current', 'N/A'):.3f} A")
        
        # Experimental comparison
        if 'experimental_comparison' in circuit_analytics:
            exp_comp = circuit_analytics['experimental_comparison']
            print(f"  Experimental Comparison:")
            print(f"    - RMS difference: {exp_comp.get('rms_difference', 'N/A'):.3f}")
            print(f"    - MAE difference: {exp_comp.get('mae_difference', 'N/A'):.3f}")


def main():
    """
    Fully integrated main function for single circuit simulation
    Demonstrates the simplified workflow achieved through refactoring
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
        
        # Run simplified workflow
        return run_single_circuit_workflow(args)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)