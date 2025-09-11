"""
magnet_scipy/plotting.py

Refactored plotting module using strategy pattern
Maintains backward compatibility while providing cleaner architecture
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from .plotting_core import PlottingManager, PlottingAnalytics, ResultsFileManager
from .plotting_strategies import PlotConfiguration
from .utils import exp_metrics

# Global plotting manager instance
_plotting_manager = None


def get_plotting_manager(config: PlotConfiguration = None) -> PlottingManager:
    """Get or create the global plotting manager instance"""
    global _plotting_manager
    if _plotting_manager is None or config is not None:
        _plotting_manager = PlottingManager(config)
    return _plotting_manager


def configure_plotting(
    figsize: Tuple[int, int] = (16, 20),
    dpi: int = 300,
    show_experimental: bool = True,
    show_regions: bool = True,
    show_temperature: bool = True,
    **kwargs
):
    """Configure global plotting settings"""
    config = PlotConfiguration(
        figsize=figsize,
        dpi=dpi,
        show_experimental=show_experimental,
        show_regions=show_regions,
        show_temperature=show_temperature,
        **kwargs
    )
    global _plotting_manager
    _plotting_manager = PlottingManager(config)


# Backward compatible functions for single circuit
def prepare_post(sol, circuit, mode: str = "regular") -> Tuple[np.ndarray, Dict]:
    """
    Prepare single circuit simulation results for plotting
    
    Args:
        sol: Solution object from scipy.integrate.solve_ivp
        circuit: RLCircuitPID instance
        mode: "regular" for voltage simulation, "cde" for PID control
        
    Returns:
        Tuple of (time_array, results_dict)
    """
    manager = get_plotting_manager()
    strategy_type = "voltage_input" if mode == "regular" else "pid_control"
    results, _ = manager.create_plots(
        sol, circuit, strategy_type, show=False, show_analytics=False
    )
    
    # Convert to old format for compatibility
    circuit_id = circuit.circuit_id
    data = results.circuits[circuit_id]
    
    return results.time, data


def plot_vresults(
    circuit,
    t: np.ndarray,
    data: Dict,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot voltage simulation results for single circuit
    
    Backward compatible wrapper that uses the new plotting system
    """
    # Create a fake solution object for the new system
    class FakeSol:
        def __init__(self, t, current):
            self.t = t
            if isinstance(current, dict):
                self.y = current['current']
            else:
                self.y = current
            self.success = True
    
    current = data.get('current', np.zeros_like(t))
    sol = FakeSol(t, current)
    
    manager = get_plotting_manager()
    manager.create_plots(
        sol, circuit, "voltage_input", save_path, show, show_analytics=False
    )


def plot_results(
    sol,
    circuit,
    t: np.ndarray,
    data: Dict,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot PID control results for single circuit
    
    Backward compatible wrapper that uses the new plotting system
    """
    manager = get_plotting_manager()
    manager.create_plots(
        sol, circuit, "pid_control", save_path, show, show_analytics=False
    )


def analyze(circuit, t: np.ndarray, data: Dict):
    """
    Perform detailed analysis of single circuit simulation
    
    Backward compatible wrapper
    """
    # Create results structure for new analytics system
    from .plotting_strategies import ProcessedResults
    
    circuits_data = {circuit.circuit_id: data}
    
    # Determine strategy type based on available data
    if 'reference' in data and 'error' in data:
        strategy_type = "pid_control"
    else:
        strategy_type = "voltage_input"
    
    results = ProcessedResults(
        time=t,
        circuits=circuits_data,
        strategy_type=strategy_type,
        metadata={}
    )
    
    analytics = PlottingAnalytics.analyze_circuit_performance(results)
    PlottingAnalytics.print_detailed_analytics(analytics)


def save_results(
    circuit,
    t: np.ndarray,
    results: Dict,
    filename: str = "single_circuit_results.npz"
):
    """
    Save single circuit results to file
    
    Backward compatible wrapper
    """
    from .plotting_strategies import ProcessedResults
    
    circuits_data = {circuit.circuit_id: results}
    
    # Determine strategy type
    if 'reference' in results and 'error' in results:
        strategy_type = "pid_control"
    else:
        strategy_type = "voltage_input"
    
    processed_results = ProcessedResults(
        time=t,
        circuits=circuits_data,
        strategy_type=strategy_type,
        metadata={"circuit_id": circuit.circuit_id}
    )
    
    # Generate analytics for saving
    analytics = PlottingAnalytics.analyze_circuit_performance(processed_results)
    
    ResultsFileManager.save_results(processed_results, analytics, filename)


def load_results(filename: str):
    """Load previously saved single circuit results"""
    return ResultsFileManager.load_results(filename)


# New advanced plotting functions
def create_advanced_plots(
    sol,
    system,
    strategy_type: str = None,
    save_path: str = None,
    show: bool = True,
    show_analytics: bool = False,
    config: PlotConfiguration = None
):
    """
    Create advanced plots with full control over configuration
    
    Args:
        sol: Solution object
        system: Circuit or coupled system
        strategy_type: "voltage_input", "pid_control", or None for auto-detect
        save_path: Path to save plots
        show: Whether to show plots
        show_analytics: Whether to print detailed analytics
        config: Custom plot configuration
        
    Returns:
        Tuple of (processed_results, analytics, figure)
    """
    manager = get_plotting_manager(config)
    results, analytics = manager.create_plots(
        sol, system, strategy_type, save_path, show, show_analytics
    )
    return results, analytics


def create_comparison_plots(
    results_list,
    labels: list,
    save_path: str = None,
    show: bool = True,
    config: PlotConfiguration = None
):
    """
    Create comparison plots between multiple simulation runs
    
    Args:
        results_list: List of (sol, system) tuples or ProcessedResults
        labels: List of labels for each result
        save_path: Path to save comparison plots
        show: Whether to show plots
        config: Custom plot configuration
        
    Returns:
        Figure object
    """
    manager = get_plotting_manager(config)
    
    # Process results if needed
    processed_results = []
    for i, item in enumerate(results_list):
        if hasattr(item, 'circuits'):  # Already ProcessedResults
            processed_results.append((item, labels[i]))
        else:
            # Assume (sol, system) tuple
            sol, system = item
            results, _ = manager.create_plots(
                sol, system, show=False, show_analytics=False
            )
            processed_results.append((results, labels[i]))
    
    return manager.create_comparison_plots(processed_results, save_path, show)


def create_custom_plot(
    data_dict: Dict,
    plot_type: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save_path: str = None,
    show: bool = True,
    **kwargs
):
    """
    Create custom plots with user-provided data
    
    Args:
        data_dict: Dictionary of {label: (x_data, y_data)} pairs
        plot_type: "line", "scatter", "bar", etc.
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save plot
        show: Whether to show plot
        **kwargs: Additional matplotlib arguments
    """
    manager = get_plotting_manager()
    colors = manager.config.colors
    
    fig, ax = plt.subplots(figsize=manager.config.figsize)
    
    for i, (label, (x_data, y_data)) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        
        if plot_type == "line":
            ax.plot(x_data, y_data, color=color, label=label, 
                   linewidth=manager.config.linewidth_main, **kwargs)
        elif plot_type == "scatter":
            ax.scatter(x_data, y_data, color=color, label=label, **kwargs)
        elif plot_type == "bar":
            ax.bar(x_data, y_data, color=color, label=label, alpha=0.7, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=manager.config.grid_alpha)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=manager.config.dpi)
        print(f"Custom plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# Experimental data overlay utilities
def add_experimental_overlay(
    ax,
    time_data: np.ndarray,
    computed_data: np.ndarray,
    exp_time: np.ndarray,
    exp_data: np.ndarray,
    label: str = "Experimental",
    color: str = None
):
    """
    Add experimental data overlay to an existing plot
    
    Args:
        ax: Matplotlib axis object
        time_data: Computed time array
        computed_data: Computed data array
        exp_time: Experimental time array
        exp_data: Experimental data array
        label: Label for experimental data
        color: Color for experimental data (auto if None)
    """
    if color is None:
        color = ax.get_lines()[-1].get_color()  # Use same color as last line
    
    # Plot experimental data
    ax.plot(exp_time, exp_data, color=color, linestyle=":", 
           alpha=0.7, linewidth=1.0, label=label)
    
    # Compute and display comparison metrics
    rms_diff, mae_diff = exp_metrics(time_data, exp_time, computed_data, exp_data)
    
    ax.text(
        0.02, 0.98,
        f"RMS Diff: {rms_diff:.3f}\nMAE Diff: {mae_diff:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8)
    )
    
    return rms_diff, mae_diff


# Performance benchmarking utilities
def benchmark_plotting_performance(sol, system, n_runs: int = 5):
    """
    Benchmark plotting performance
    
    Args:
        sol: Solution object
        system: Circuit system
        n_runs: Number of runs for averaging
        
    Returns:
        Dictionary with timing information
    """
    import time
    
    manager = get_plotting_manager()
    
    # Warm up
    manager.create_plots(sol, system, show=False, show_analytics=False)
    
    # Benchmark data preparation
    prep_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        strategy_type = manager.detect_strategy(sol, system)
        strategy = manager.strategies[strategy_type]
        results = strategy.prepare_data(sol, system)
        end = time.perf_counter()
        prep_times.append(end - start)
    
    # Benchmark plot creation
    plot_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        strategy.create_plots(results, system, show=False)
        end = time.perf_counter()
        plot_times.append(end - start)
    
    return {
        "data_preparation": {
            "mean": np.mean(prep_times),
            "std": np.std(prep_times),
            "min": np.min(prep_times),
            "max": np.max(prep_times)
        },
        "plot_creation": {
            "mean": np.mean(plot_times),
            "std": np.std(plot_times),
            "min": np.min(plot_times),
            "max": np.max(plot_times)
        },
        "total": {
            "mean": np.mean(prep_times) + np.mean(plot_times)
        }
    }
