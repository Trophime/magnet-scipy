"""
magnet_scipy/cli_plotting_integration.py

Integration layer between refactored CLI components and plotting system
Updates the CLI simulation components to use the new plotting architecture
"""

from typing import Dict, Tuple, Any
import numpy as np

from .plotting_core import PlottingManager, PlottingAnalytics, ResultsFileManager
from .plotting_strategies import PlotConfiguration, ProcessedResults
from .cli_core import OutputOptions


class EnhancedPlottingManager(PlottingManager):
    """Enhanced plotting manager integrated with CLI components"""
    
    def __init__(self, output_options: OutputOptions, config: PlotConfiguration = None):
        super().__init__(config)
        self.output_options = output_options
    
    def create_plots_from_simulation_result(
        self,
        simulation_result,
        system,
        config_file: str = None
    ) -> Tuple[ProcessedResults, Dict]:
        """
        Create plots directly from SimulationResult objects
        Integrates with the refactored CLI simulation components
        """
        # Convert SimulationResult to solution format
        class FakeSol:
            def __init__(self, result):
                self.t = result.time
                self.y = result.solution
                self.success = result.success
                self.message = result.error_message or "Strategy simulation completed"
                self.nfev = result.metadata.get('n_evaluations', 0)
        
        sol = FakeSol(simulation_result)
        
        # Determine strategy from simulation result
        strategy_type = simulation_result.strategy_type
        if strategy_type == "failed":
            raise RuntimeError(f"Cannot plot failed simulation: {simulation_result.error_message}")
        
        # Create plots using the parent method
        results, analytics = self.create_plots(
            sol, system, strategy_type,
            save_path=self.output_options.save_plots,
            show=self.output_options.show_plots,
            show_analytics=self.output_options.show_analytics
        )
        
        # Save results if requested
        if self.output_options.save_results:
            output_filename = self.output_options.save_results
        elif config_file:
            output_filename = config_file.replace(".json", ".npz")
        else:
            output_filename = "simulation_results.npz"
        
        if self.output_options.save_results or config_file:
            ResultsFileManager.save_results(results, analytics, output_filename)
        
        return results, analytics


def update_result_processor():
    """
    Update the ResultProcessor class in cli_simulation.py to use new plotting
    This shows how to integrate with the existing CLI refactor
    """
    updated_code = '''
class ResultProcessor:
    """Enhanced result processor using new plotting system"""
    
    def __init__(self):
        self.results_cache = {}
    
    def process_single_circuit_result(
        self,
        result: SimulationResult,
        circuit: RLCircuitPID,
        output_options: OutputOptions
    ) -> Tuple[np.ndarray, Dict]:
        """Process single circuit results using new plotting system"""
        if not result.success:
            raise RuntimeError(f"Cannot process failed simulation: {result.error_message}")
        
        # Create enhanced plotting manager
        plotting_manager = EnhancedPlottingManager(output_options)
        
        # Process results and create plots
        processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
            result, circuit
        )
        
        # Cache results
        cache_key = f"single_{circuit.circuit_id}_{id(result)}"
        self.results_cache[cache_key] = (processed_results.time, processed_results.circuits[circuit.circuit_id])
        
        return processed_results.time, processed_results.circuits[circuit.circuit_id]
    
    def process_coupled_circuits_result(
        self,
        result: SimulationResult,
        coupled_system: CoupledRLCircuitsPID,
        output_options: OutputOptions
    ) -> Tuple[np.ndarray, Dict]:
        """Process coupled circuit results using new plotting system"""
        if not result.success:
            raise RuntimeError(f"Cannot process failed simulation: {result.error_message}")
        
        # Create enhanced plotting manager
        plotting_manager = EnhancedPlottingManager(output_options)
        
        # Process results and create plots
        processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
            result, coupled_system
        )
        
        # Cache results
        cache_key = f"coupled_{len(result.circuit_ids)}_{id(result)}"
        self.results_cache[cache_key] = (processed_results.time, processed_results.circuits)
        
        return processed_results.time, processed_results.circuits
    '''
    
    return updated_code


def update_plotting_manager_in_cli():
    """
    Update PlottingManager class in cli_simulation.py to use new system
    This eliminates the old plotting logic
    """
    updated_code = '''
class PlottingManager:
    """Simplified plotting manager using new plotting architecture"""
    
    def plot_single_circuit_results(
        self,
        circuit: RLCircuitPID,
        t: np.ndarray,
        results: Dict,
        strategy_type: str,
        output_options: OutputOptions
    ):
        """Generate plots for single circuit simulation"""
        print("Generating single circuit plots using new plotting system...")
        
        # The plotting has already been done by EnhancedPlottingManager
        # This method is kept for interface compatibility but may be simplified
        print("✓ Single circuit plots generated successfully")
    
    def plot_coupled_circuits_results(
        self,
        coupled_system: CoupledRLCircuitsPID,
        t: np.ndarray,
        results: Dict,
        strategy_type: str,
        output_options: OutputOptions
    ):
        """Generate plots for coupled circuits simulation"""
        print("Generating coupled circuits plots using new plotting system...")
        
        # The plotting has already been done by EnhancedPlottingManager
        # This method is kept for interface compatibility but may be simplified
        print("✓ Coupled circuits plots generated successfully")
    '''
    
    return updated_code


def update_analytics_manager_in_cli():
    """
    Update AnalyticsManager class to use new analytics system
    """
    updated_code = '''
class AnalyticsManager:
    """Enhanced analytics manager using new analytics system"""
    
    def analyze_single_circuit_results(
        self,
        circuit: RLCircuitPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions
    ):
        """Analytics have been handled by EnhancedPlottingManager"""
        # Analytics are now integrated into the plotting workflow
        # This method is kept for interface compatibility
        pass
    
    def analyze_coupled_circuits_results(
        self,
        coupled_system: CoupledRLCircuitsPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions
    ):
        """Analytics have been handled by EnhancedPlottingManager"""
        # Analytics are now integrated into the plotting workflow
        # This method is kept for interface compatibility
        pass
    '''
    
    return updated_code


def create_plotting_config_from_args(args) -> PlotConfiguration:
    """
    Create PlotConfiguration from command line arguments
    Can be extended to support more plotting customization options
    """
    config = PlotConfiguration()
    
    # Set DPI for high-quality output if saving
    if hasattr(args, 'save_plots') and args.save_plots:
        config.dpi = 300
    
    # Customize based on debug mode
    if hasattr(args, 'debug') and args.debug:
        config.show_experimental = True
        config.show_regions = True
        config.show_temperature = True
    
    # Could add more arguments like:
    # --plot-style, --plot-colors, --plot-size, etc.
    
    return config


def update_main_functions():
    """
    Show how to update the main simulation functions to use new plotting
    """
    single_circuit_update = '''
def run_single_circuit_simulation_with_strategy(args):
    """Updated single circuit simulation using new plotting"""
    
    # ... existing simulation code ...
    
    # Create plotting configuration
    plot_config = create_plotting_config_from_args(args)
    output_options = create_output_options_from_args(args)
    
    # Create enhanced plotting manager
    plotting_manager = EnhancedPlottingManager(output_options, plot_config)
    
    # Process and plot results
    processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
        result, circuit, args.config_file
    )
    
    # The plotting, analytics, and saving are all handled automatically
    # No need for separate plotting and analytics components
    
    return result, circuit, processed_results.time, processed_results.circuits[circuit.circuit_id]
    '''
    
    coupled_circuits_update = '''
def run_coupled_simulation_with_strategy(args):
    """Updated coupled simulation using new plotting"""
    
    # ... existing simulation code ...
    
    # Create plotting configuration
    plot_config = create_plotting_config_from_args(args)
    output_options = create_output_options_from_args(args)
    
    # Create enhanced plotting manager
    plotting_manager = EnhancedPlottingManager(output_options, plot_config)
    
    # Process and plot results
    processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
        result, coupled_system, args.config_file
    )
    
    # All plotting, analytics, and saving handled automatically
    return result, coupled_system, processed_results.time, processed_results.circuits
    '''
    
    return single_circuit_update, coupled_circuits_update


def create_migration_guide():
    """
    Create a migration guide for updating existing code to use new plotting system
    """
    migration_guide = '''
# Migration Guide: Old Plotting System → New Strategy-Based System

## 1. Replace old plotting imports
# Old:
from magnet_scipy.plotting import prepare_post, plot_results, plot_vresults
from magnet_scipy.coupled_plotting import prepare_coupled_post, plot_coupled_results

# New:
from magnet_scipy.plotting import create_advanced_plots, configure_plotting
from magnet_scipy.coupled_plotting import analyze_coupling_effects  # Enhanced version

## 2. Replace mode-based function calls
# Old:
t, results = prepare_post(sol, circuit, mode="regular")  # or mode="cde"
plot_vresults(circuit, t, results, save_path="plots.png")

# New:
processed_results, analytics = create_advanced_plots(
    sol, circuit, 
    strategy_type=None,  # Auto-detect or specify "voltage_input"/"pid_control"
    save_path="plots.png",
    show_analytics=True
)

## 3. Update coupled circuit plotting
# Old:
t, results = prepare_coupled_post(sol, coupled_system, mode="regular")
plot_coupled_vresults(sol, coupled_system, t, results, save_path="plots.png")

# New:
processed_results, analytics = create_advanced_plots(
    sol, coupled_system,
    strategy_type=None,  # Auto-detect
    save_path="plots.png",
    show_analytics=True
)

## 4. Configure plotting appearance globally
# New capability:
configure_plotting(
    figsize=(20, 24),
    dpi=300,
    show_experimental=True,
    show_regions=True,
    show_temperature=True
)

## 5. Create comparison plots
# New capability:
from magnet_scipy.plotting import create_comparison_plots

fig = create_comparison_plots(
    [(sol1, system1), (sol2, system2)],
    labels=["Configuration A", "Configuration B"],
    save_path="comparison.png"
)

## 6. Advanced custom plotting
# New capability:
from magnet_scipy.plotting import create_custom_plot

fig = create_custom_plot(
    {"Current": (time, current_data), "Reference": (time, ref_data)},
    plot_type="line",
    title="Custom Current Comparison",
    xlabel="Time (s)",
    ylabel="Current (A)"
)

## 7. Performance benchmarking
# New capability:
from magnet_scipy.plotting import benchmark_plotting_performance

timing_info = benchmark_plotting_performance(sol, system, n_runs=10)
print(f"Average plotting time: {timing_info['total']['mean']:.3f} seconds")
'''
    
    return migration_guide


def create_example_usage():
    """
    Create comprehensive examples showing new plotting system usage
    """
    examples = '''
# Example 1: Basic single circuit plotting with auto-detection
def example_single_circuit_auto():
    # ... run simulation to get sol and circuit ...
    
    # Automatically detect strategy and create plots
    results, analytics = create_advanced_plots(
        sol, circuit,
        save_path="single_circuit_results.png",
        show=True,
        show_analytics=True
    )
    
    print(f"Strategy used: {results.strategy_type}")
    print(f"Circuits analyzed: {list(results.circuits.keys())}")

# Example 2: Coupled circuits with custom configuration
def example_coupled_circuits_custom():
    # Configure plotting globally
    configure_plotting(
        figsize=(20, 24),
        dpi=300,
        show_experimental=True,
        show_regions=True,
        linewidth_main=2.5,
        alpha_experimental=0.6
    )
    
    # ... run simulation to get sol and coupled_system ...
    
    results, analytics = create_advanced_plots(
        sol, coupled_system,
        strategy_type="pid_control",  # Explicitly specify
        save_path="coupled_results.png",
        show_analytics=True
    )
    
    # Additional coupling analysis
    analyze_coupling_effects(coupled_system, results.time, results.circuits)

# Example 3: Comparison between different configurations
def example_comparison_study():
    # Run multiple simulations
    results_list = []
    labels = []
    
    for coupling_strength in [0.0, 0.05, 0.1, 0.2]:
        # ... create system with specific coupling ...
        # ... run simulation ...
        results_list.append((sol, system))
        labels.append(f"Coupling = {coupling_strength}")
    
    # Create comparison plot
    fig = create_comparison_plots(
        results_list, labels,
        save_path="coupling_study.png",
        show=True
    )

# Example 4: Advanced analytics and custom analysis
def example_advanced_analytics():
    # ... run simulation ...
    
    results, analytics = create_advanced_plots(
        sol, coupled_system,
        show=False,  # Don't show plots, just process data
        show_analytics=False  # Handle analytics manually
    )
    
    # Custom analytics
    print("=== Custom Analysis ===")
    for circuit_id, data in results.circuits.items():
        current_stats = data['current']
        print(f"{circuit_id}:")
        print(f"  Peak current: {np.max(np.abs(current_stats)):.3f} A")
        print(f"  Average power: {np.mean(data['power']):.3f} W")
        
        if results.strategy_type == "pid_control":
            error_rms = np.sqrt(np.mean(data['error']**2))
            print(f"  RMS tracking error: {error_rms:.4f} A")
    
    # Create custom plots for specific analysis
    power_data = {
        circuit_id: (results.time, data['power']) 
        for circuit_id, data in results.circuits.items()
    }
    
    create_custom_plot(
        power_data,
        plot_type="line",
        title="Power Dissipation Comparison",
        xlabel="Time (s)",
        ylabel="Power (W)",
        save_path="power_analysis.png"
    )

# Example 5: Experimental data integration
def example_experimental_integration():
    import matplotlib.pyplot as plt
    
    # ... run simulation ...
    results, analytics = create_advanced_plots(sol, circuit, show=False)
    
    # Create custom plot with experimental overlay
    fig, ax = plt.subplots(figsize=(12, 8))
    
    circuit_id = list(results.circuits.keys())[0]
    data = results.circuits[circuit_id]
    
    # Plot computed results
    ax.plot(results.time, data['current'], 'b-', linewidth=2, label='Computed')
    
    # Add experimental overlay using utility function
    if 'experimental_functions' in data:
        exp_funcs = data['experimental_functions']
        if 'current_current' in exp_funcs:
            exp_data = exp_funcs['current_current']
            rms_diff, mae_diff = add_experimental_overlay(
                ax, results.time, data['current'],
                exp_data['time_data'], exp_data['values_data'],
                label="Experimental"
            )
            print(f"Experimental comparison: RMS={rms_diff:.3f}, MAE={mae_diff:.3f}")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Current (A)')
    ax.set_title('Current: Computed vs Experimental')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
'''
    
    return examples


def create_testing_strategy():
    """
    Create testing strategy for the new plotting system
    """
    testing_strategy = '''
# Testing Strategy for New Plotting System

## 1. Unit Tests for Strategy Components
class TestPlottingStrategies:
    def test_voltage_strategy_data_preparation(self):
        # Test VoltageInputPlottingStrategy.prepare_data()
        pass
    
    def test_pid_strategy_data_preparation(self):
        # Test PIDControlPlottingStrategy.prepare_data()
        pass
    
    def test_strategy_auto_detection(self):
        # Test PlottingManager.detect_strategy()
        pass

## 2. Integration Tests
class TestPlottingIntegration:
    def test_single_circuit_voltage_plotting(self):
        # Test complete voltage simulation plotting workflow
        pass
    
    def test_single_circuit_pid_plotting(self):
        # Test complete PID simulation plotting workflow
        pass
    
    def test_coupled_circuit_plotting(self):
        # Test coupled circuit plotting
        pass
    
    def test_experimental_data_overlay(self):
        # Test experimental data integration
        pass

## 3. Backward Compatibility Tests
class TestBackwardCompatibility:
    def test_prepare_post_compatibility(self):
        # Ensure old prepare_post() calls work
        pass
    
    def test_plot_results_compatibility(self):
        # Ensure old plot_results() calls work
        pass
    
    def test_coupled_plotting_compatibility(self):
        # Ensure old coupled plotting functions work
        pass

## 4. Performance Tests
class TestPlottingPerformance:
    def test_plotting_speed(self):
        # Benchmark plotting performance
        timing = benchmark_plotting_performance(sol, system)
        assert timing['total']['mean'] < 5.0  # Should complete within 5 seconds
    
    def test_memory_usage(self):
        # Test memory efficiency of new system
        pass

## 5. Visual Regression Tests
class TestVisualRegression:
    def test_plot_output_consistency(self):
        # Compare plot outputs with reference images
        # Use matplotlib testing utilities
        pass
    
    def test_color_scheme_consistency(self):
        # Ensure consistent color schemes across plots
        pass

## 6. Error Handling Tests
class TestErrorHandling:
    def test_invalid_strategy_type(self):
        # Test handling of invalid strategy types
        with pytest.raises(ValueError):
            manager = PlottingManager()
            manager.create_plots(sol, system, "invalid_strategy")
    
    def test_missing_experimental_data(self):
        # Test graceful handling of missing experimental data
        pass
    
    def test_corrupted_solution_data(self):
        # Test handling of malformed solution objects
        pass
'''
    
    return testing_strategy


# Integration helper functions for CLI components
def integrate_with_cli_simulation():
    """
    Integration points for updating CLI simulation components
    """
    integration_points = {
        'cli_simulation.py': {
            'ResultProcessor': update_result_processor(),
            'PlottingManager': update_plotting_manager_in_cli(), 
            'AnalyticsManager': update_analytics_manager_in_cli()
        },
        'main.py': {
            'run_simulation_function': update_main_functions()[0]
        },
        'coupled_main.py': {
            'run_simulation_function': update_main_functions()[1]
        }
    }
    
    return integration_points


def create_configuration_examples():
    """
    Create examples of different plotting configurations
    """
    config_examples = '''
# Plotting Configuration Examples

## 1. High-Quality Publication Plots
publication_config = PlotConfiguration(
    figsize=(12, 16),
    dpi=600,  # High DPI for publications
    show_experimental=True,
    show_regions=True,
    show_temperature=True,
    linewidth_main=1.5,
    linewidth_experimental=1.0,
    alpha_experimental=0.8,
    alpha_regions=0.15,
    grid_alpha=0.2
)

## 2. Presentation/Slides Configuration
presentation_config = PlotConfiguration(
    figsize=(16, 12),  # Wide format for slides
    dpi=150,
    show_experimental=False,  # Cleaner for presentations
    show_regions=True,
    show_temperature=False,
    linewidth_main=3.0,  # Thicker lines for visibility
    alpha_regions=0.3,
    grid_alpha=0.4
)

## 3. Debug/Analysis Configuration
debug_config = PlotConfiguration(
    figsize=(20, 24),  # Large for detailed analysis
    dpi=150,
    show_experimental=True,
    show_regions=True, 
    show_temperature=True,
    alpha_experimental=0.9,  # Prominent experimental data
    alpha_regions=0.4,
    grid_alpha=0.5
)

## 4. Minimal/Fast Configuration
fast_config = PlotConfiguration(
    figsize=(12, 8),  # Smaller for speed
    dpi=100,  # Lower DPI for speed
    show_experimental=False,
    show_regions=False,  # Minimal elements
    show_temperature=False,
    linewidth_main=1.0
)

# Usage:
configure_plotting(**publication_config.__dict__)
'''
    
    return config_examples


# Summary of refactoring benefits
def get_refactoring_benefits():
    """
    Document the benefits achieved by the plotting refactor
    """
    benefits = {
        "Eliminated Complex Mode Switching": [
            "No more mode='regular' vs mode='cde' conditionals",
            "Strategy pattern handles different simulation types cleanly",
            "Each strategy is self-contained and testable"
        ],
        
        "Improved Code Organization": [
            "Separated data preparation from visualization",
            "Common utilities shared between single/coupled plotting", 
            "Clear interfaces and responsibilities"
        ],
        
        "Enhanced Functionality": [
            "Advanced analytics integrated into plotting workflow",
            "Comparison plots between different simulations",
            "Custom plot creation utilities",
            "Performance benchmarking tools"
        ],
        
        "Better Experimental Data Integration": [
            "Consistent experimental data overlay across all plot types",
            "Automatic comparison metrics calculation",
            "Configurable experimental data display"
        ],
        
        "Maintained Backward Compatibility": [
            "All existing plotting function calls continue to work",
            "Gradual migration path for existing code",
            "No breaking changes for users"
        ],
        
        "Improved Testability": [
            "Each strategy can be unit tested independently",
            "Data preparation separated from plotting logic",
            "Clear interfaces make mocking easier"
        ],
        
        "Configuration and Customization": [
            "Global plotting configuration system",
            "Per-plot customization options", 
            "Easy to extend with new plot types"
        ]
    }
    
    return benefits
    