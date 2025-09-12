"""
magnet_scipy/plotting_strategies.py

Strategy pattern implementation for different types of simulation result plotting
Eliminates mode-switching complexity in plotting functions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .utils import exp_metrics
from scipy import stats


@dataclass
class PlotConfiguration:
    """Configuration for plot appearance and behavior"""
    figsize: Tuple[int, int] = (16, 20)
    dpi: int = 300
    show_experimental: bool = True
    show_regions: bool = True
    show_temperature: bool = True
    alpha_experimental: float = 0.7
    alpha_regions: float = 0.2
    linewidth_main: float = 2.0
    linewidth_experimental: float = 1.0
    grid_alpha: float = 0.3
    colors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.colors is None:
            # Default color palette
            self.colors = plt.cm.Set1(np.linspace(0, 1, 10)).tolist()


@dataclass
class ProcessedResults:
    """Standardized container for processed simulation results"""
    time: np.ndarray
    circuits: Dict[str, Dict[str, Any]]  # circuit_id -> {current, voltage, etc.}
    strategy_type: str
    metadata: Dict[str, Any]


class PlottingStrategy(ABC):
    """Abstract base class for different plotting strategies"""
    
    def __init__(self, config: PlotConfiguration = None):
        self.config = config or PlotConfiguration()
    
    @abstractmethod
    def prepare_data(self, sol, system, **kwargs) -> ProcessedResults:
        """Prepare simulation data for plotting"""
        pass
    
    @abstractmethod
    def get_subplot_layout(self, n_circuits: int) -> Tuple[int, int]:
        """Get subplot layout for the given number of circuits"""
        pass
    
    @abstractmethod
    def create_plots(
        self, 
        results: ProcessedResults, 
        system,
        save_path: str = None,
        show: bool = True
    ) -> Figure:
        """Create the complete plot figure"""
        pass
    
    def setup_axis(self, ax: Axes, title: str, ylabel: str, xlabel: str = None):
        """Common axis setup"""
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(True, alpha=self.config.grid_alpha)
    
    def add_experimental_overlay(
        self, 
        ax: Axes, 
        circuit_id: str, 
        circuit_data: Dict,
        data_type: str,
        color: str
    ):
        """Add experimental data overlay to a plot"""
        if not self.config.show_experimental:
            print(f"self.config.show_experimental={self.config.show_experimental}")
            return
        
        exp_key = f"{data_type}_{data_type}"
        if exp_key not in circuit_data["experimental_functions"]:
            print(f"{exp_key} not in {circuit_data["experimental_functions"].keys()}")
            return
        
        exp_info = circuit_data["experimental_functions"][exp_key]
        if "time_data" not in exp_info or "values_data" not in exp_info:
            print(f"time_data or values_data not in {exp_info.keys()}")
            return
        
        # Plot experimental data
        ax.plot(
            exp_info["time_data"],
            exp_info["values_data"],
            color=color,
            linewidth=self.config.linewidth_experimental,
            linestyle=":",
            alpha=self.config.alpha_experimental,
            label=f"{circuit_id} - Experimental"
        )
        
        # Add comparison metrics if available
        if "rms_diff" in exp_info and "mae_diff" in exp_info:
            rms_diff = exp_info["rms_diff"]
            mae_diff = exp_info["mae_diff"]
            
            ax.text(
                0.02, 0.98,
                f"{circuit_id}\nRMS: {rms_diff:.2f}\nMAE: {mae_diff:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )


class VoltageInputPlottingStrategy(PlottingStrategy):
    """Strategy for plotting voltage-driven simulation results"""
    
    def prepare_data(self, sol, system, **kwargs) -> ProcessedResults:
        """Prepare voltage simulation data for plotting"""
        t = sol.t
        
        # Determine if single or coupled system
        if hasattr(system, 'circuits'):
            # Coupled system
            circuits_data = {}
            for i, circuit in enumerate(system.circuits):
                circuit_id = circuit.circuit_id
                current = sol.y[i, :]
                voltage = circuit.voltage_func(t)
                
                # Load experimental data
                circuit._load_experimental_data()
                
                # Process experimental data comparisons
                rms_diff, mae_diff = None, None
                if circuit.has_experimental_data(data_type="current", key="current"):
                    exp_time = circuit.experimental_functions["current_current"]["time_data"]
                    exp_current = circuit.experimental_functions["current_current"]["values_data"]
                    rms_diff, mae_diff = exp_metrics(t, exp_time, current, exp_current)
                    circuit.experimental_functions["current_current"]["rms_diff"] = rms_diff
                    circuit.experimental_functions["current_current"]["mae_diff"] = mae_diff
                
                # Calculate additional metrics
                temperature_over_time = self._get_temperature_over_time(circuit, t)
                resistance_over_time = self._get_resistance_over_time(circuit, current, temperature_over_time)
                power = resistance_over_time * current**2
                
                circuits_data[circuit_id] = {
                    "current": current,
                    "voltage": voltage,
                    "temperature": temperature_over_time,
                    "resistance": resistance_over_time,
                    "power": power,
                    "experimental_functions": getattr(circuit, 'experimental_functions', {}),
                    "circuit_info": circuit
                }
        else:
            # Single circuit
            circuit_id = system.circuit_id
            current = sol.y.squeeze()
            voltage = system.voltage_func(t)
            
            # Load experimental data
            system._load_experimental_data()
            
            # Process experimental data comparisons
            if system.has_experimental_data(data_type="current", key="current"):
                exp_time = system.experimental_functions["current_current"]["time_data"]
                exp_current = system.experimental_functions["current_current"]["values_data"]
                rms_diff, mae_diff = exp_metrics(t, exp_time, current, exp_current)
                system.experimental_functions["current_current"]["rms_diff"] = rms_diff
                system.experimental_functions["current_current"]["mae_diff"] = mae_diff
            
            # Calculate additional metrics
            temperature_over_time = self._get_temperature_over_time(system, t)
            resistance_over_time = self._get_resistance_over_time(system, current, temperature_over_time)
            power = resistance_over_time * current**2
            
            circuits_data = {
                circuit_id: {
                    "current": current,
                    "voltage": voltage,
                    "temperature": temperature_over_time,
                    "resistance": resistance_over_time,
                    "power": power,
                    "experimental_functions": getattr(system, 'experimental_functions', {}),
                    "circuit_info": system
                }
            }
        
        return ProcessedResults(
            time=t,
            circuits=circuits_data,
            strategy_type="voltage_input",
            metadata={"sol_info": {"success": getattr(sol, 'success', True)}}
        )
    
    def _get_temperature_over_time(self, circuit, t):
        """Get temperature over time for a circuit"""
        if circuit.use_variable_temperature:
            return np.array([circuit.get_temperature(float(time)) for time in t])
        else:
            return None
    
    def _get_resistance_over_time(self, circuit, current, temperature_over_time):
        """Get resistance over time for a circuit"""
        if temperature_over_time is not None:
            return np.array([
                circuit.get_resistance(float(curr), float(temp))
                for curr, temp in zip(current, temperature_over_time)
            ])
        else:
            return np.array([circuit.get_resistance(float(curr)) for curr in current])
    
    def get_subplot_layout(self, n_circuits: int) -> Tuple[int, int]:
        """4 subplots for voltage simulation: current, voltage, resistance, power"""
        return (4, 1)
    
    def create_plots(
        self,
        results: ProcessedResults,
        system,
        save_path: str = None,
        show: bool = True
    ) -> Figure:
        """Create voltage simulation plots"""
        n_circuits = len(results.circuits)
        rows, cols = self.get_subplot_layout(n_circuits)
        print(f"create_plots for voltage: {n_circuits} circuits (save_path={save_path})" )

        fig, axes = plt.subplots(rows, cols, figsize=self.config.figsize, sharex=True)
        if rows == 1:
            axes = [axes]
        
        circuit_ids = list(results.circuits.keys())
        colors = self.config.colors[:n_circuits]
        
        # 1. Current plot
        ax = axes[0]
        self.setup_axis(ax, "Currents", "Current (A)")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            ax.plot(
                results.time, data["current"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=circuit_id, linestyle="-"
            )
            
            # Add experimental overlay
            self.add_experimental_overlay(ax, circuit_id, data, "current", colors[i])
        
        ax.legend()
        
        # 2. Voltage plot
        ax = axes[1]
        self.setup_axis(ax, "Voltages", "Voltage (V)")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            ax.plot(
                results.time, data["voltage"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=circuit_id
            )
        
        ax.legend()
        
        # 3. Resistance plot with temperature
        ax = axes[2]
        ax2 = None
        if self.config.show_temperature:
            # Check if any circuit has variable temperature
            has_variable_temp = any(
                data["temperature"] is not None 
                for data in results.circuits.values()
            )
            if has_variable_temp:
                ax2 = ax.twinx()
        
        self.setup_axis(ax, "Resistances", "Resistance (Ω)")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            circuit = data["circuit_info"]
            
            if data["temperature"] is not None:
                label = circuit_id
            else:
                label = f"{circuit_id} (T={circuit.temperature}°C)"
            
            ax.plot(
                results.time, data["resistance"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=label
            )
            
            # Add temperature on right axis if available
            if ax2 and data["temperature"] is not None:
                ax2.plot(
                    results.time, data["temperature"],
                    color=colors[i], marker="o", markersize=4, markevery=5,
                    linestyle="", alpha=0.5, label=f"{circuit_id} T"
                )
        
        if ax2:
            ax2.set_ylabel("Temperature (°C)")
            ax2.tick_params(axis="y")
            ax2.grid(False)
        
        ax.legend()
        
        # 4. Power plot
        ax = axes[3]
        self.setup_axis(ax, "P = R(I,T) × I²", "Power (W)", "Time (s)")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            ax.plot(
                results.time, data["power"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=circuit_id
            )
        
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi)
            print(f"Plots saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig


class PIDControlPlottingStrategy(PlottingStrategy):
    """Strategy for plotting PID control simulation results"""
    
    def prepare_data(self, sol, system, **kwargs) -> ProcessedResults:
        """Prepare PID simulation data for plotting"""
        t = sol.t

        # Determine if single or coupled system
        if hasattr(system, 'circuits'):
            # Coupled system
            n_circuits = system.n_circuits
            currents = sol.y[:n_circuits, :]
            integral_errors = sol.y[n_circuits:, :]
            
            circuits_data = {}
            for i, circuit in enumerate(system.circuits):
                circuit_id = circuit.circuit_id
                current = currents[i]
                integral_error = integral_errors[i]
                
                # Process PID data
                pid_data = self._process_pid_data(circuit, t, current, integral_error)
                
                circuits_data[circuit_id] = pid_data
        else:
            # Single circuit
            current = sol.y[0]
            integral_error = sol.y[1]
            print(f'single circuit: current={current.shape}, integral_error={integral_error.shape}')
            
            pid_data = self._process_pid_data(system, t, current, integral_error)
            circuits_data = {system.circuit_id: pid_data}
        
        return ProcessedResults(
            time=t,
            circuits=circuits_data,
            strategy_type="pid_control",
            metadata={"sol_info": {"success": getattr(sol, 'success', True)}}
        )
    
    def _process_pid_data(self, circuit, t, current, integral_error):
        """Process PID-specific data for a circuit"""
        # Load experimental data
        circuit._load_experimental_data()
        
        # Calculate reference current
        i_ref = np.array([circuit.reference_current(t_val) for t_val in t])
        print("current:", current.shape)
        print("i_ref:", i_ref.shape)
        
        # Calculate adaptive PID gains over time
        Kp_over_time, Ki_over_time, Kd_over_time = [], [], []
        current_regions = []
        
        for i_ref_val in i_ref:
            i_ref_float = float(i_ref_val)
            Kp, Ki, Kd = circuit.get_pid_parameters(i_ref_float)
            Kp_over_time.append(float(Kp))
            Ki_over_time.append(float(Ki))
            Kd_over_time.append(float(Kd))
            
            region_name = circuit.get_current_region(i_ref_float)
            current_regions.append(region_name)
        
        # Calculate control signals and errors
        error = i_ref - current
        derivative_error = np.gradient(error, t[1] - t[0])
        
        Kp_array = np.array(Kp_over_time)
        Ki_array = np.array(Ki_over_time)
        Kd_array = np.array(Kd_over_time)
        
        voltage = (
            Kp_array * error + Ki_array * integral_error + Kd_array * derivative_error
        )
        
        # Process experimental data comparisons
        if circuit.has_experimental_data(data_type="voltage", key="voltage"):
            exp_time = circuit.experimental_functions["voltage_voltage"]["time_data"]
            exp_values = circuit.experimental_functions["voltage_voltage"]["values_data"]
            rms_diff, mae_diff = exp_metrics(t, exp_time, voltage, exp_values)
            circuit.experimental_functions["voltage_voltage"]["rms_diff"] = rms_diff
            circuit.experimental_functions["voltage_voltage"]["mae_diff"] = mae_diff
        
        # Calculate additional metrics
        temperature_over_time = self._get_temperature_over_time(circuit, t)
        resistance_over_time = self._get_resistance_over_time(circuit, current, temperature_over_time)
        power = resistance_over_time * current**2
        
        return {
            "current": current,
            "reference": i_ref,
            "error": error,
            "voltage": voltage,
            "power": power,
            "resistance": resistance_over_time,
            "temperature": temperature_over_time,
            "Kp": Kp_array,
            "Ki": Ki_array,
            "Kd": Kd_array,
            "regions": current_regions,
            "integral_error": integral_error,
            "experimental_functions": getattr(circuit, 'experimental_functions', {}),
            "circuit_info": circuit
        }
    
    def _get_temperature_over_time(self, circuit, t):
        """Get temperature over time for a circuit"""
        if circuit.use_variable_temperature:
            return np.array([circuit.get_temperature(float(time)) for time in t])
        else:
            return None
    
    def _get_resistance_over_time(self, circuit, current, temperature_over_time):
        """Get resistance over time for a circuit"""
        if temperature_over_time is not None:
            return np.array([
                circuit.get_resistance(float(curr), float(temp))
                for curr, temp in zip(current, temperature_over_time)
            ])
        else:
            return np.array([circuit.get_resistance(float(curr)) for curr in current])
    
    def get_subplot_layout(self, n_circuits: int) -> Tuple[int, int]:
        """6 subplots for PID: current, PID gains, voltage, resistance, power, error"""
        return (6, 1)
    
    def add_region_background(self, ax: Axes, time: np.ndarray, regions: List[str]):
        """Add colored background for PID regions"""
        if not self.config.show_regions:
            return
        
        region_colors = {
            "low": "lightgreen",
            "medium": "lightyellow", 
            "high": "lightcoral"
        }
        
        # Sample regions for performance (every 100 points)
        sampled_regions = regions[::100] if len(regions) > 100 else regions
        sampled_time = time[::100] if len(time) > 100 else time
        
        prev_region = None
        for i, region in enumerate(sampled_regions):
            if region != prev_region:
                region_start = sampled_time[i]
                
                # Find next region change
                region_end = sampled_time[-1]
                for j in range(i + 1, len(sampled_regions)):
                    if sampled_regions[j] != region:
                        region_end = sampled_time[j]
                        break
                
                color_key = region.lower()
                ax.axvspan(
                    region_start, region_end,
                    alpha=self.config.alpha_regions,
                    color=region_colors.get(color_key, "lightgray"),
                    label=f"{region} Current" if prev_region != region else ""
                )
                prev_region = region
    
    def create_plots(
        self,
        results: ProcessedResults,
        system,
        save_path: str = None,
        show: bool = True
    ) -> Figure:
        """Create PID control plots"""

        n_circuits = len(results.circuits)
        rows, cols = self.get_subplot_layout(n_circuits)
        print(f"PIDController.create_plots for pid: {n_circuits} circuits (save_path={save_path})", flush=True )

        fig, axes = plt.subplots(rows, cols, figsize=self.config.figsize, sharex=True)
        
        circuit_ids = list(results.circuits.keys())
        colors = self.config.colors[:n_circuits]
        
        # 1. Current tracking plot
        ax = axes[0]
        self.setup_axis(ax, "Current Tracking - All Circuits", "Current (A)")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            
            # Add region background (use first circuit's regions)
            if i == 0 and n_circuits == 1:
                self.add_region_background(ax, results.time, data["regions"])
            
            ax.plot(
                results.time, data["current"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=f"{circuit_id} - Actual", linestyle="-"
            )
            ax.plot(
                results.time, data["reference"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=f"{circuit_id} - Reference", linestyle="--", alpha=0.7
            )
        
        ax.legend()
        
        # 2. PID Gains
        ax = axes[1]
        self.setup_axis(ax, "Adaptive PID Parameters", "PID Gains")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            ax.plot(results.time, data["Kp"], "g-", label=f"{circuit_id} Kp", linewidth=2)
            ax.plot(results.time, data["Ki"], "b-", label=f"{circuit_id} Ki", linewidth=2)
            ax.plot(results.time, data["Kd"] * 100, "r-", 
                   label=f"{circuit_id} Kd × 100", linewidth=2)
        
        ax.legend()

        # 3. Voltages
        ax = axes[2]
        self.setup_axis(ax, "Control Voltages", "Voltage (V)")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            ax.plot(
                results.time, data["voltage"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=circuit_id
            )
            
            # Add experimental overlay
            self.add_experimental_overlay(ax, circuit_id, data, "voltage", colors[i])
       
        ax.legend()

        # 4. Resistance plot with temperature
        ax = axes[3]
        ax2 = None
        if self.config.show_temperature:
            has_variable_temp = any(
                data["temperature"] is not None 
                for data in results.circuits.values()
            )
            if has_variable_temp:
                ax2 = ax.twinx()
        
        self.setup_axis(ax, "Circuit Resistances", "Resistance (Ω)")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            circuit = data["circuit_info"]
            
            if data["temperature"] is not None:
                label = circuit_id
            else:
                label = f"{circuit_id} (T={circuit.temperature}°C)"
            
            ax.plot(
                results.time, data["resistance"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=label
            )
            
            if ax2 and data["temperature"] is not None:
                ax2.plot(
                    results.time, data["temperature"],
                    color=colors[i], marker="o", markersize=4, markevery=5,
                    linestyle="", alpha=0.5, label=f"{circuit_id} T"
                )
        
        if ax2:
            ax2.set_ylabel("Temperature (°C)")
            ax2.tick_params(axis="y")
            ax2.grid(False)
        
        ax.legend()

        # 5. Power dissipation
        ax = axes[4]
        self.setup_axis(ax, "Power Dissipation", "Power (W)")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            ax.plot(
                results.time, data["power"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=circuit_id
            )
        
        ax.legend()

        # 6. Tracking errors
        ax = axes[5]
        self.setup_axis(ax, "Tracking Errors", "Error (A)", "Time (s)")
        
        for i, circuit_id in enumerate(circuit_ids):
            data = results.circuits[circuit_id]
            ax.plot(
                results.time, data["error"],
                color=colors[i], linewidth=self.config.linewidth_main,
                label=circuit_id
            )
        
        ax.legend()

        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi)
            print(f"Plots saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
