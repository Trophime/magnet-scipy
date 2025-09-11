#!/usr/bin/env python3
"""
magnet_scipy/coupled_main_refactor.py

Fully integrated coupled circuits main function using the new plotting system
Demonstrates the simplified workflow with unified plotting, analytics, and saving
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
    
    # Add version info
    parser.add_argument(
        "--version",
        action="version",
        version="Coupled RL Circuits PID Simulation 2.0 (Fully Integrated)",
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
    Simplified workflow for coupled circuits simulation with integrated plotting
    
    This represents the key improvement: unified workflow that eliminates 
    separate plotting, analytics, and file management steps
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
        
        # Run simulation
        orchestrator = SimulationOrchestrator()
        simulation_result = orchestrator.run_coupled_simulation(
            coupled_system, time_params, args.value_start, args.strategy
        )
        
        if not simulation_result.success:
            SimulationSummary.print_error_summary(simulation_result)
            return 1
        
        # Create integrated plotting manager
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
        benchmark_results = benchmark_coupled_plotting_performance(
            sol, coupled_system, n_runs=3  # Fewer runs for coupled systems
        )
        
        print("📊 Coupled Circuits Plotting Performance Benchmark Results:")
        print(f"  Average plotting time: {benchmark_results['avg_time']:.3f} seconds")
        print(f"  Standard deviation: {benchmark_results['std_time']:.3f} seconds")
        print(f"  Min time: {benchmark_results['min_time']:.3f} seconds")
        print(f"  Max time: {benchmark_results['max_time']:.3f} seconds")
        print(f"  Circuits processed: {coupled_system.n_circuits}")
        print(f"  Time per circuit: {benchmark_results['avg_time']/coupled_system.n_circuits:.3f} seconds")
        
        # Still create the plots normally
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
            simulation_result, coupled_system, args.config_file
        )
        
        return 0
        
    except Exception as e:
        return CLIErrorHandler.handle_simulation_error(e, output_options.debug)


def run_comparison_workflow(coupled_system, time_params, output_options, plot_config, args) -> int:
    """Run coupled simulation in comparison mode with different strategies"""
    print("🔄 Running comparison mode with multiple strategies...")
    
    try:
        strategies = ["voltage", "pid"] if args.strategy == "auto" else [args.strategy]
        results = []
        labels = []
        
        orchestrator = SimulationOrchestrator()
        
        for strategy in strategies:
            print(f"  Running coupled simulation with {strategy} strategy...")
            simulation_result = orchestrator.run_coupled_simulation(
                coupled_system, time_params, args.value_start, strategy
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
        from .coupled_plotting import create_coupled_comparison_plots
        
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        
        # Process each result
        processed_results = []
        for result, label in zip(results, labels):
            processed, analytics = plotting_manager.create_plots_from_simulation_result(
                result, coupled_system, show=False  # Don't show individual plots
            )
            processed_results.append((processed, label))
        
        # Create comparison plot
        comparison_fig = create_coupled_comparison_plots(
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


def print_analytics_summary(analytics: dict, coupled_system):
    """Print a summary of the analytics results for coupled circuits"""
    print("\n📈 Analytics Summary:")
    
    # Per-circuit analytics
    for circuit in coupled_system.circuits:
        circuit_id = circuit.circuit_id
        if circuit_id in analytics:
            circuit_analytics = analytics[circuit_id]
            print(f"  Circuit {circuit_id}:")
            
            # Performance metrics
            if 'performance' in circuit_analytics:
                perf = circuit_analytics['performance']
                print(f"    Performance Metrics:")
                print(f"      - Steady-state error: {perf.get('steady_state_error', 'N/A')}")
                print(f"      - Rise time: {perf.get('rise_time', 'N/A')} s")
                print(f"      - Settling time: {perf.get('settling_time', 'N/A')} s")
                print(f"      - Overshoot: {perf.get('overshoot', 'N/A')}%")
            
            # Statistical metrics
            if 'statistics' in circuit_analytics:
                stats = circuit_analytics['statistics']
                print(f"    Statistical Analysis:")
                print(f"      - Mean current: {stats.get('mean_current', 'N/A'):.3f} A")
                print(f"      - RMS current: {stats.get('rms_current', 'N/A'):.3f} A")
                print(f"      - Peak current: {stats.get('peak_current', 'N/A'):.3f} A")
            
            # Experimental comparison
            if 'experimental_comparison' in circuit_analytics:
                exp_comp = circuit_analytics['experimental_comparison']
                print(f"    Experimental Comparison:")
                print(f"      - RMS difference: {exp_comp.get('rms_difference', 'N/A'):.3f}")
                print(f"      - MAE difference: {exp_comp.get('mae_difference', 'N/A'):.3f}")


def print_coupling_analysis(coupling_analysis: dict):
    """Print coupling analysis results"""
    print("\n🔗 Coupling Analysis:")
    
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


def main():
    """
    Fully integrated main function for coupled circuits simulation
    Demonstrates the simplified workflow achieved through refactoring
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
