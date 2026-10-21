"""
magnet_scipy/plotting.py

Version 3.0: Clean plotting module with backward compatibility functions removed
Breaking changes: All legacy wrapper functions removed, only new system functions remain
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .plotting_core import PlottingManager, as_solution
from .plotting_strategies import PlotConfiguration

# Global plotting manager instance
_plotting_manager = None


def get_plotting_manager(config: Optional[PlotConfiguration] = None) -> PlottingManager:
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
    **kwargs,
):
    """Configure global plotting settings"""
    config = PlotConfiguration(
        figsize=figsize,
        dpi=dpi,
        show_experimental=show_experimental,
        show_regions=show_regions,
        show_temperature=show_temperature,
        **kwargs,
    )
    global _plotting_manager
    _plotting_manager = PlottingManager(config)


def create_advanced_plots(
    sol,
    system,
    strategy_type: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    show_analytics: bool = True,
    config: Optional[PlotConfiguration] = None,
):
    """Create plots for a simulation result using the plotting strategies.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult or SimulationResult
        Solution to plot.
    system : RLCircuitPID or CoupledRLCircuitsPID
        Simulated circuit or coupled system.
    strategy_type : {"voltage_input", "pid_control"}, optional
        Plotting strategy; auto-detected if None.
    save_path : str, optional
        Path to save the plots.
    show : bool, default True
        Whether to display the plots.
    show_analytics : bool, default True
        Whether to print the analytics.
    config : PlotConfiguration, optional
        Custom plotting configuration.

    Returns
    -------
    results : ProcessedResults
        Processed simulation data.
    analytics : dict
        Circuit performance analytics.
    """
    manager = get_plotting_manager(config)
    return manager.create_plots(
        sol, system, strategy_type, save_path, show, show_analytics
    )


def create_comparison_plots(
    solutions_and_systems: list,
    labels: Optional[list] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    config: Optional[PlotConfiguration] = None,
):
    """Create comparison plots between multiple simulations.

    Parameters
    ----------
    solutions_and_systems : list of tuple
        ``(sol, system)`` pairs to compare.
    labels : list of str, optional
        Label for each simulation; defaults to ``"Simulation 1"``, ...
    save_path : str, optional
        Path to save the plots.
    show : bool, default True
        Whether to display the plots.
    config : PlotConfiguration, optional
        Custom plotting configuration.

    Returns
    -------
    matplotlib.figure.Figure
        The comparison figure.
    """
    manager = get_plotting_manager(config)
    if labels is None:
        labels = [f"Simulation {i + 1}" for i in range(len(solutions_and_systems))]
    results_list = [
        (manager.prepare_results(sol, system), label)
        for (sol, system), label in zip(solutions_and_systems, labels)
    ]
    return manager.create_comparison_plots(results_list, save_path, show)


def create_custom_plot(
    data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plot_type: str = "line",
    title: str = "Custom Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    save_path: Optional[str] = None,
    show: bool = True,
    config: Optional[PlotConfiguration] = None,
):
    """Create a custom plot from user-defined data.

    Parameters
    ----------
    data_dict : dict of str to tuple of numpy.ndarray
        Series label to ``(x_data, y_data)``.
    plot_type : {"line", "scatter", "bar"}, default "line"
        Type of plot.
    title : str, default "Custom Plot"
        Plot title.
    xlabel : str, default "X"
        X-axis label.
    ylabel : str, default "Y"
        Y-axis label.
    save_path : str, optional
        Path to save the plot.
    show : bool, default True
        Whether to display the plot; otherwise the figure is closed.
    config : PlotConfiguration, optional
        Custom plotting configuration (figure size and dpi).

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        If ``plot_type`` is not supported.
    """
    if config:
        plt.rcParams["figure.figsize"] = config.figsize
        plt.rcParams["figure.dpi"] = config.dpi

    fig, ax = plt.subplots()

    for label, (x_data, y_data) in data_dict.items():
        if plot_type == "line":
            ax.plot(x_data, y_data, label=label)
        elif plot_type == "scatter":
            ax.scatter(x_data, y_data, label=label)
        elif plot_type == "bar":
            ax.bar(x_data, y_data, label=label, alpha=0.7)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.dpi if config else 300)
        print(f"Custom plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def benchmark_plotting_performance(sol, system, n_runs: int = 5) -> Dict[str, Dict]:
    """Benchmark plotting performance.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult or SimulationResult
        Solution to plot.
    system : RLCircuitPID or CoupledRLCircuitsPID
        Simulated circuit or coupled system.
    n_runs : int, default 5
        Number of timed runs, after one warm-up run.

    Returns
    -------
    dict
        ``"data_preparation"``, ``"plot_creation"`` and ``"total"``, each
        mapping to ``{"mean", "std", "min", "max"}`` timings [s].
    """
    import time

    manager = get_plotting_manager()
    sol = as_solution(sol)
    strategy = manager.strategies[manager.detect_strategy(sol, system)]

    # Warm up
    results = strategy.prepare_data(sol, system)
    strategy.create_plots(results, system, show=False)

    prep_times = []
    plot_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        results = strategy.prepare_data(sol, system)
        prep_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        strategy.create_plots(results, system, show=False)
        plot_times.append(time.perf_counter() - start)

    def _stats(times) -> Dict[str, float]:
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
        }

    return {
        "data_preparation": _stats(prep_times),
        "plot_creation": _stats(plot_times),
        "total": _stats(np.add(prep_times, plot_times)),
    }


# Export clean API functions only
__all__ = [
    "get_plotting_manager",
    "configure_plotting",
    "create_advanced_plots",
    "create_comparison_plots",
    "create_custom_plot",
    "benchmark_plotting_performance",
]


# Version 3.0 Breaking Changes Notice
#
# REMOVED FUNCTIONS (no longer available):
# - prepare_post() → Use create_advanced_plots()
# - plot_vresults() → Use create_advanced_plots() with strategy_type="voltage_input"
# - plot_results() → Use create_advanced_plots() with strategy_type="pid_control"
# - analyze() → Analytics integrated into create_advanced_plots()
#
# MIGRATION GUIDE:
#
# Old voltage plotting:
#   t, data = prepare_post(sol, circuit, mode="regular")
#   plot_vresults(circuit, t, data, save_path="plot.png")
#
# New voltage plotting:
#   processed_results, analytics = create_advanced_plots(
#       sol, circuit,
#       strategy_type="voltage_input",
#       save_path="plot.png",
#       show_analytics=True
#   )
#
# Old PID plotting:
#   t, data = prepare_post(sol, circuit, mode="cde")
#   plot_results(sol, circuit, t, data, save_path="plot.png")
#
# New PID plotting:
#   processed_results, analytics = create_advanced_plots(
#       sol, circuit,
#       strategy_type="pid_control",
#       save_path="plot.png",
#       show_analytics=True
#   )
#
# Old analytics:
#   analyze(circuit, t, data)
#
# New analytics (integrated):
#   processed_results, analytics = create_advanced_plots(
#       sol, circuit,
#       show_analytics=True
#   )
#   # Analytics are automatically displayed and returned in analytics dict
