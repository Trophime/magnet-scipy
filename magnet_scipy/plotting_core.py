"""
magnet_scipy/plotting_core.py

Core plotting components, utilities, and analytics
Shared functionality between different plotting strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from dataclasses import dataclass

from .plotting_strategies import ProcessedResults, PlotConfiguration
from .utils import exp_metrics


class PlottingStyleManager:
    """Manage consistent styling across all plots"""
    
    @staticmethod
    def setup_matplotlib_defaults():
        """Set up default matplotlib parameters"""
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 9
    
    @staticmethod
    def get_color_palette(n_colors: int) -> List[str]:
        """Get a consistent color palette"""
        if n_colors <= 10:
            return plt.cm.Set1(np.linspace(0, 1, max(n_colors, 3)))[:n_colors].tolist()
        else:
            return plt.cm.viridis(np.linspace(0, 1, n_colors)).tolist()
    
    @staticmethod
    def get_line_styles() -> Dict[str, str]:
        """Get consistent line styles for different data types"""
        return {
            'actual': '-',
            'reference': '--',
            'experimental': ':',
            'pid_gains': '-'
        }


class DataProcessor:
    """Process raw simulation data for plotting"""
    
    @staticmethod
    def compute_statistics(data: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics for data"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'rms': float(np.sqrt(np.mean(data**2))),
            'range': float(np.ptp(data))  # peak-to-peak
        }
    
    @staticmethod
    def compute_error_metrics(actual: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        """Compute error metrics between actual and reference signals"""
        error = actual - reference
        return {
            'mae': float(np.mean(np.abs(error))),  # Mean Absolute Error
            'mse': float(np.mean(error**2)),       # Mean Squared Error
            'rmse': float(np.sqrt(np.mean(error**2))),  # Root Mean Squared Error
            'max_error': float(np.max(np.abs(error))),
            'bias': float(np.mean(error))  # Average bias
        }
    
    @staticmethod
    def compute_experimental_comparison(
        time: np.ndarray,
        computed: np.ndarray,
        exp_time: np.ndarray,
        exp_data: np.ndarray
    ) -> Tuple[float, float]:
        """Compute RMS and MAE differences with experimental data"""
        return exp_metrics(time, exp_time, computed, exp_data)


class PlottingAnalytics:
    """Analytics and detailed analysis for simulation results"""
    
    @staticmethod
    def analyze_circuit_performance(results: ProcessedResults) -> Dict[str, Dict]:
        """Analyze performance metrics for all circuits"""
        print("analyze_circuit_performance")
        analytics = {}
        
        for circuit_id, data in results.circuits.items():
            circuit_analytics = {
                'current_stats': DataProcessor.compute_statistics(data['current']),
                'voltage_stats': DataProcessor.compute_statistics(data['voltage']),
                'power_stats': DataProcessor.compute_statistics(data['power']),
                'resistance_stats': DataProcessor.compute_statistics(data['resistance'])
            }
            
            # Add PID-specific analytics if available
            if results.strategy_type == "pid_control":
                circuit_analytics.update({
                    'error_stats': DataProcessor.compute_statistics(data['error']),
                    'tracking_metrics': DataProcessor.compute_error_metrics(
                        data['current'], data['reference']
                    ),
                    'region_usage': PlottingAnalytics._analyze_region_usage(data['regions'])
                })
            
            # Add experimental comparison if available
            if 'experimental_functions' in data and data['experimental_functions']:
                circuit_analytics['experimental_comparison'] = (
                    PlottingAnalytics._analyze_experimental_comparison(data)
                )
            
            analytics[circuit_id] = circuit_analytics
        
        return analytics
    
    @staticmethod
    def _analyze_region_usage(regions: List[str]) -> Dict[str, float]:
        """Analyze PID region usage percentages"""
        if not regions:
            return {}
        
        unique_regions, counts = np.unique(regions, return_counts=True)
        total = len(regions)
        
        return {
            region: float(count / total * 100)
            for region, count in zip(unique_regions, counts)
        }
    
    @staticmethod
    def _analyze_experimental_comparison(data: Dict) -> Dict[str, Dict]:
        """Analyze experimental data comparison metrics"""
        comparison = {}
        
        for exp_key, exp_info in data['experimental_functions'].items():
            if 'rms_diff' in exp_info and 'mae_diff' in exp_info:
                data_type = exp_key.split('_')[0]  # Extract data type (current, voltage, etc.)
                comparison[data_type] = {
                    'rms_difference': exp_info['rms_diff'],
                    'mae_difference': exp_info['mae_diff'],
                    'data_points': len(exp_info.get('time_data', [])),
                    'time_range': {
                        'start': float(np.min(exp_info['time_data'])) if 'time_data' in exp_info else 0,
                        'end': float(np.max(exp_info['time_data'])) if 'time_data' in exp_info else 0
                    }
                }
        
        return comparison
    
    @staticmethod
    def print_detailed_analytics(analytics: Dict[str, Dict]):
        """Print detailed analytics to console"""
        print("\n=== Detailed Simulation Analytics ===")
        
        for circuit_id, circuit_analytics in analytics.items():
            print(f"\n{circuit_id} Performance Metrics:")
            
            # Current statistics
            current_stats = circuit_analytics['current_stats']
            print(f"  Current:")
            print(f"    RMS: {current_stats['rms']:.3f} A")
            print(f"    Range: {current_stats['min']:.3f} to {current_stats['max']:.3f} A")
            print(f"    Std Dev: {current_stats['std']:.3f} A")
            
            # Voltage statistics
            voltage_stats = circuit_analytics['voltage_stats']
            print(f"  Voltage:")
            print(f"    RMS: {voltage_stats['rms']:.3f} V")
            print(f"    Range: {voltage_stats['min']:.3f} to {voltage_stats['max']:.3f} V")
            
            # Power statistics
            power_stats = circuit_analytics['power_stats']
            print(f"  Power:")
            print(f"    Average: {power_stats['mean']:.3f} W")
            print(f"    Peak: {power_stats['max']:.3f} W")
            print(f"    Total Energy (approx): {power_stats['mean'] * 1:.3f} J")  # Assuming 1s simulation
            
            # PID-specific analytics
            if 'tracking_metrics' in circuit_analytics:
                tracking = circuit_analytics['tracking_metrics']
                print(f"  Tracking Performance:")
                print(f"    RMS Error: {tracking['rmse']:.4f} A")
                print(f"    Max Error: {tracking['max_error']:.4f} A")
                print(f"    Mean Absolute Error: {tracking['mae']:.4f} A")
                
                if 'region_usage' in circuit_analytics:
                    print(f"  PID Region Usage:")
                    for region, percentage in circuit_analytics['region_usage'].items():
                        print(f"    {region}: {percentage:.1f}%")
            
            # Experimental comparison
            if 'experimental_comparison' in circuit_analytics:
                exp_comp = circuit_analytics['experimental_comparison']
                print(f"  Experimental Data Comparison:")
                for data_type, metrics in exp_comp.items():
                    print(f"    {data_type.title()}:")
                    print(f"      RMS Difference: {metrics['rms_difference']:.4f}")
                    print(f"      MAE Difference: {metrics['mae_difference']:.4f}")


class PlottingManager:
    """Main plotting manager that orchestrates different strategies"""
    
    def __init__(self, config: PlotConfiguration = None):
        self.config = config or PlotConfiguration()
        PlottingStyleManager.setup_matplotlib_defaults()
        
        # Initialize strategies
        from .plotting_strategies import VoltageInputPlottingStrategy, PIDControlPlottingStrategy
        self.strategies = {
            "voltage_input": VoltageInputPlottingStrategy(self.config),
            "pid_control": PIDControlPlottingStrategy(self.config)
        }
    
    def detect_strategy(self, sol, system) -> str:
        """Detect appropriate plotting strategy based on system and solution"""
        # Check if system has PID controllers and reference data
        if hasattr(system, 'circuits'):  # Coupled system
            has_pid = all(
                hasattr(c, 'pid_controller') and c.pid_controller is not None 
                for c in system.circuits
            )
            has_reference = all(
                hasattr(c, 'reference_csv') and c.reference_csv is not None
                for c in system.circuits
            )
        else:  # Single circuit
            has_pid = hasattr(system, 'pid_controller') and system.pid_controller is not None
            has_reference = hasattr(system, 'reference_csv') and system.reference_csv is not None
        
        # Check solution structure - PID has more state variables
        expected_pid_states = 2  # current + integral_error per circuit
        if hasattr(system, 'circuits'):
            expected_pid_states *= len(system.circuits)
        
        solution_suggests_pid = (
            hasattr(sol, 'y') and 
            len(sol.y.shape) > 1 and 
            sol.y.shape[0] >= expected_pid_states
        )
        
        if has_pid and has_reference and solution_suggests_pid:
            return "pid_control"
        else:
            return "voltage_input"
    
    def create_plots(
        self,
        sol,
        system,
        strategy_type: str = None,
        save_path: str = None,
        show: bool = True,
        show_analytics: bool = False
    ) -> Tuple[ProcessedResults, Dict]:
        """
        Main plotting interface - detects strategy and creates appropriate plots
        
        Returns:
            Tuple of (processed_results, analytics)
        """
        print(f"PlottingManager.create_plots: strategy_type={strategy_type}, save_path={save_path}, show={show}, show_analytics={show_analytics}")
        # Detect strategy if not specified
        if strategy_type is None:
            strategy_type = self.detect_strategy(sol, system)
        
        if strategy_type not in self.strategies:
            raise ValueError(f"Unknown plotting strategy: {strategy_type}")
        
        # Get appropriate strategy
        strategy = self.strategies[strategy_type]
        
        # Prepare data
        results = strategy.prepare_data(sol, system)

        # Create plots
        fig = strategy.create_plots(results, system, save_path, show)

        # Generate analytics
        analytics = PlottingAnalytics.analyze_circuit_performance(results)

        # Print analytics if requested
        if show_analytics:
            PlottingAnalytics.print_detailed_analytics(analytics)
        
        return results, analytics
    
    def create_comparison_plots(
        self,
        results_list: List[Tuple[ProcessedResults, str]],
        save_path: str = None,
        show: bool = True
    ):
        """Create comparison plots between different simulation runs"""
        if not results_list:
            raise ValueError("No results provided for comparison")
        
        n_results = len(results_list)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = PlottingStyleManager.get_color_palette(n_results)
        
        for i, (results, label) in enumerate(results_list):
            color = colors[i]
            
            # Get first circuit's data for comparison
            first_circuit_id = list(results.circuits.keys())[0]
            data = results.circuits[first_circuit_id]
            
            # Current comparison
            axes[0].plot(results.time, data['current'], 
                        color=color, label=f"{label} - Current", linewidth=2)
            
            # Voltage comparison
            axes[1].plot(results.time, data['voltage'],
                        color=color, label=f"{label} - Voltage", linewidth=2)
            
            # Power comparison
            axes[2].plot(results.time, data['power'],
                        color=color, label=f"{label} - Power", linewidth=2)
            
            # Error comparison (if available)
            if 'error' in data:
                axes[3].plot(results.time, np.abs(data['error']),
                           color=color, label=f"{label} - |Error|", linewidth=2)
        
        # Setup axes
        axes[0].set_title("Current Comparison")
        axes[0].set_ylabel("Current (A)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_title("Voltage Comparison")
        axes[1].set_ylabel("Voltage (V)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        axes[2].set_title("Power Comparison")
        axes[2].set_ylabel("Power (W)")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        axes[3].set_title("Error Comparison")
        axes[3].set_ylabel("|Error| (A)")
        axes[3].set_xlabel("Time (s)")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi)
            print(f"Comparison plots saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig


class ResultsFileManager:
    """Handle saving and loading of simulation results"""
    
    @staticmethod
    def save_results(
        results: ProcessedResults,
        analytics: Dict,
        filename: str = "simulation_results.npz"
    ):
        """Save processed results and analytics to file"""
        save_data = {
            "time": results.time,
            "strategy_type": results.strategy_type,
            "metadata": results.metadata,
            "analytics": analytics
        }
        
        # Add circuit data
        for circuit_id, data in results.circuits.items():
            for key, values in data.items():
                if isinstance(values, np.ndarray):
                    save_data[f"{circuit_id}_{key}"] = values
                elif key == "experimental_functions":
                    # Save experimental comparison metrics
                    for exp_key, exp_data in values.items():
                        if "rms_diff" in exp_data:
                            save_data[f"{circuit_id}_{exp_key}_rms"] = exp_data["rms_diff"]
                        if "mae_diff" in exp_data:
                            save_data[f"{circuit_id}_{exp_key}_mae"] = exp_data["mae_diff"]
        
        np.savez_compressed(filename, **save_data)
        print(f"Results saved to {filename}")
    
    @staticmethod
    def load_results(filename: str) -> Tuple[ProcessedResults, Dict]:
        """Load previously saved results"""
        data = np.load(filename, allow_pickle=True)
        
        time = data["time"]
        strategy_type = str(data["strategy_type"])
        metadata = data["metadata"].item() if "metadata" in data else {}
        analytics = data["analytics"].item() if "analytics" in data else {}
        
        # Reconstruct circuits data
        circuits = {}
        circuit_keys = set()
        
        for key in data.files:
            if "_" in key and key not in ["time", "strategy_type", "metadata", "analytics"]:
                circuit_id = key.split("_")[0]
                circuit_keys.add(circuit_id)
        
        for circuit_id in circuit_keys:
            circuit_data = {}
            for key in data.files:
                if key.startswith(f"{circuit_id}_"):
                    data_key = key[len(circuit_id) + 1:]
                    if not data_key.endswith(("_rms", "_mae")):
                        circuit_data[data_key] = data[key]
            circuits[circuit_id] = circuit_data
        
        results = ProcessedResults(
            time=time,
            circuits=circuits,
            strategy_type=strategy_type,
            metadata=metadata
        )
        
        print(f"Results loaded from {filename}")
        return results, analytics


# Backward compatibility functions
def prepare_post(sol, circuit, mode: str = "regular"):
    """
    Backward compatible wrapper for single circuit post-processing
    """
    manager = PlottingManager()
    strategy_type = "voltage_input" if mode == "regular" else "pid_control"
    results, analytics = manager.create_plots(
        sol, circuit, strategy_type, show=False, show_analytics=False
    )
    
    # Convert to old format for compatibility
    circuit_id = circuit.circuit_id
    data = results.circuits[circuit_id]
    
    return results.time, data


def prepare_coupled_post(sol, coupled_system, mode: str = "regular"):
    """
    Backward compatible wrapper for coupled circuit post-processing
    """
    manager = PlottingManager()
    strategy_type = "voltage_input" if mode == "regular" else "pid_control"
    results, analytics = manager.create_plots(
        sol, coupled_system, strategy_type, show=False, show_analytics=False
    )
    
    return results.time, results.circuits