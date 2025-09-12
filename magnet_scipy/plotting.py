"""
magnet_scipy/plotting.py

Version 3.0: Clean plotting module with backward compatibility functions removed
Breaking changes: All legacy wrapper functions removed, only new system functions remain
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from .plotting_core import PlottingManager, PlottingAnalytics, ResultsFileManager
from .plotting_strategies import PlotConfiguration

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


def create_advanced_plots(
    sol,
    system,
    strategy_type: str = None,
    save_path: str = None,
    show: bool = True,
    show_analytics: bool = True,
    config: PlotConfiguration = None
):
    """
    Create advanced plots using the new plotting system
    
    Args:
        sol: Solution object from scipy.integrate.solve_ivp
        system: Circuit or coupled system instance
        strategy_type: "voltage_input", "pid_control", or None for auto-detect
        save_path: Path to save plots (optional)
        show: Whether to display plots
        show_analytics: Whether to display analytics
        config: Custom plotting configuration
        
    Returns:
        Tuple of (ProcessedResults, analytics_dict)
    """
    manager = get_plotting_manager(config)
    return manager.create_plots(
        sol, system, strategy_type, save_path, show, show_analytics
    )


def create_comparison_plots(
    solutions_and_systems: list,
    labels: list = None,
    save_path: str = None,
    show: bool = True,
    config: PlotConfiguration = None
):
    """
    Create comparison plots between multiple simulations
    
    Args:
        solutions_and_systems: List of (sol, system) tuples
        labels: Labels for each simulation (optional)
        save_path: Path to save plots (optional)
        show: Whether to display plots
        config: Custom plotting configuration
        
    Returns:
        matplotlib Figure object
    """
    manager = get_plotting_manager(config)
    return manager.create_comparison_plots(
        solutions_and_systems, labels, save_path, show
    )


def create_custom_plot(
    data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    plot_type: str = "line",
    title: str = "Custom Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    save_path: str = None,
    show: bool = True,
    config: PlotConfiguration = None
):
    """
    Create custom plots with user-defined data
    
    Args:
        data_dict: Dictionary of {label: (x_data, y_data)} pairs
        plot_type: Type of plot ("line", "scatter", "bar")
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save plot (optional)
        show: Whether to display plot
        config: Custom plotting configuration
        
    Returns:
        matplotlib Figure object
    """
    if config:
        plt.rcParams['figure.figsize'] = config.figsize
        plt.rcParams['figure.dpi'] = config.dpi
    
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


# Export clean API functions only
__all__ = [
    'get_plotting_manager',
    'configure_plotting',
    'create_advanced_plots',
    'create_comparison_plots',
    'create_custom_plot'
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
