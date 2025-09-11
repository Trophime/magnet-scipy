"""
magnet_scipy/cli_core.py

Enhanced CLI core components with plotting configuration support
Integrates the new plotting system with existing CLI architecture
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .rlcircuitpid import RLCircuitPID
from .pid_controller import create_adaptive_pid_controller
from .plotting_strategies import PlotConfiguration


@dataclass
class TimeParameters:
    """Encapsulate all time-related simulation parameters"""
    start: float
    end: float
    step: float
    method: str = "RK45"
    
    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError("Start time must be less than end time")
        if self.step <= 0:
            raise ValueError("Time step must be positive")
    
    @property
    def span(self) -> float:
        """Total simulation time span"""
        return self.end - self.start
    
    @property
    def n_steps(self) -> int:
        """Estimated number of time steps"""
        return int(self.span / self.step)


@dataclass
class OutputOptions:
    """Enhanced output options with plotting configuration"""
    show_plots: bool = False
    save_plots: Optional[str] = None
    show_analytics: bool = False
    save_results: Optional[str] = None
    debug: bool = False
    
    # New plotting configuration options
    plot_config: Optional[str] = None
    comparison_mode: bool = False
    benchmark_plotting: bool = False
    plot_profile: str = "default"
    
    def validate(self):
        """Validate output options and provide warnings"""
        if not self.show_plots and not self.save_plots:
            print("⚠️ Neither --save-plots nor --show-plots specified. Forcing show_plots.")
            self.show_plots = True
        
        if self.save_plots and self.show_plots:
            print("⚠️ Both --save-plots and --show-plots specified. Plots will be saved and shown.")


class ArgumentParser:
    """Enhanced argument parsing with plotting configuration support"""
    
    @staticmethod
    def create_base_parser(description: str) -> argparse.ArgumentParser:
        """Create base parser with common arguments"""
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        
        # Working directory
        parser.add_argument(
            "--wd", 
            type=str, 
            help="Working directory"
        )
        
        # Configuration
        parser.add_argument(
            "--config-file",
            type=str,
            help="Path to JSON configuration file"
        )
        
        parser.add_argument(
            "--create-sample",
            action="store_true",
            help="Create a sample configuration file and exit"
        )
        
        return parser
    
    @staticmethod
    def add_time_arguments(parser: argparse.ArgumentParser):
        """Add time-related arguments"""
        parser.add_argument(
            "--time_start", 
            type=float, 
            default=0.0, 
            help="Simulation start time in seconds"
        )
        
        parser.add_argument(
            "--time_end", 
            type=float, 
            default=5.0, 
            help="Simulation end time in seconds"
        )
        
        parser.add_argument(
            "--time_step", 
            type=float, 
            default=0.001, 
            help="Simulation time step in seconds"
        )
        
        parser.add_argument(
            "--method",
            type=str,
            choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
            default="RK45",
            help="ODE solver method",
        )
    
    @staticmethod
    def add_output_arguments(parser: argparse.ArgumentParser):
        """Add enhanced output-related arguments with plotting configuration"""
        # Basic output arguments
        parser.add_argument(
            "--show_plots",
            action="store_true",
            help="Show plots of simulation results",
        )
        
        parser.add_argument(
            "--show_analytics",
            action="store_true",
            help="Show detailed analytics of simulation results",
        )
        
        parser.add_argument(
            "--save_plots",
            type=str,
            metavar="PATH",
            help="Save plots to specified path (PNG format)",
        )
        
        parser.add_argument(
            "--save_results",
            type=str,
            metavar="PATH",
            help="Save simulation results to file (NPZ format)",
        )
        
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode with additional information",
        )
        
        # Enhanced plotting configuration arguments
        parser.add_argument(
            "--plot-config",
            type=str,
            metavar="PATH",
            help="Path to custom plotting configuration file (JSON format)"
        )
        
        parser.add_argument(
            "--plot-profile",
            type=str,
            choices=["default", "publication", "presentation", "debug", "minimal"],
            default="default",
            help="Predefined plotting profile for different use cases"
        )
        
        parser.add_argument(
            "--comparison-mode",
            action="store_true",
            help="Enable comparison mode for multiple simulation runs"
        )
        
        parser.add_argument(
            "--benchmark-plotting",
            action="store_true",
            help="Enable plotting performance benchmarking and profiling"
        )


# Rest of the existing CLI core components...
class ConfigurationLoader:
    """Load circuit configurations from JSON files"""
    
    @staticmethod
    def load_single_circuit(config_file: str) -> RLCircuitPID:
        """Load single circuit configuration"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
            # Create PID controller if parameters provided
            pid_controller = None
            if "pid_params" in config:
                pid_params = config["pid_params"]
                pid_controller = create_adaptive_pid_controller(**pid_params)

            # Create RLCircuitPID instance
            circuit = RLCircuitPID(
                R=config.get("resistance", 1.0),
                L=config.get("inductance", 0.1),
                pid_controller=pid_controller,
                reference_csv=config.get("reference_csv", None),
                voltage_csv=config.get("voltage_csv", None),
                resistance_csv=config.get("resistance_csv"),
                temperature=config.get("temperature", 25.0),
                temperature_csv=config.get("temperature_csv", None),
                circuit_id=config.get("circuit_id", "single_circuit"),
                experimental_data=config.get("experiment_data", []),        # Extract circuit configuration
            )

        return circuit
    
    @staticmethod
    def load_coupled_circuits(config_file: str):
        """Load coupled circuits configuration"""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        circuits = []
        for config in config_data["circuits"]:
            # Create PID controller if parameters provided
            pid_controller = None
            if "pid_params" in config:
                pid_params = config["pid_params"]
                pid_controller = create_adaptive_pid_controller(**pid_params)

            # Create RLCircuitPID instance
            circuit = RLCircuitPID(
                R=config.get("resistance", 1.0),
                L=config.get("inductance", 0.1),
                pid_controller=pid_controller,
                reference_csv=config.get("reference_csv", None),
                voltage_csv=config.get("voltage_csv", None),
                resistance_csv=config.get("resistance_csv"),
                temperature=config.get("temperature", 25.0),
                temperature_csv=config.get("temperature_csv", None),
                circuit_id=config.get("circuit_id", f"circuit_{len(circuits)+1}"),
                experimental_data=config.get("experiment_data", []),
            )
            circuits.append(circuit)
        
        # Load mutual inductances
        mutual_inductances = config_data.get('mutual_inductances', [])
        if mutual_inductances:
            mutual_inductances = np.array(mutual_inductances)
        
        return circuits, mutual_inductances


class WorkingDirectoryManager:
    """Context manager for working directory operations"""
    
    def __init__(self, wd: Optional[str] = None):
        self.wd = wd
        self.original_wd = None
    
    def __enter__(self):
        if self.wd:
            self.original_wd = os.getcwd()
            print(f"Changing working directory to: {self.wd}")
            os.chdir(self.wd)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_wd:
            os.chdir(self.original_wd)


class SampleConfigGenerator:
    """Generate sample configuration files"""
    
    @staticmethod
    def create_single_circuit_config(filename: str = "single_circuit_config.json"):
        """Create a sample single circuit configuration"""
        config = {
            "circuit": {
                "circuit_id": "sample_circuit",
                "L": 0.1,
                "R": 1.0,
                "temperature": 25.0,
                "reference_csv": "reference_current.csv",
                "pid": {
                    "Kp": 10.0,
                    "Ki": 0.5,
                    "Kd": 0.01
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Sample single circuit configuration created: {filename}")
    
    @staticmethod
    def create_coupled_circuits_config(filename: str = "coupled_circuits_config.json"):
        """Create a sample coupled circuits configuration"""
        config = {
            "circuits": [
                {
                    "circuit_id": "circuit_1",
                    "L": 0.1,
                    "R": 1.0,
                    "temperature": 25.0,
                    "reference_csv": "reference_current_1.csv",
                    "pid": {
                        "Kp": 10.0,
                        "Ki": 0.5,
                        "Kd": 0.01
                    }
                },
                {
                    "circuit_id": "circuit_2",
                    "L": 0.15,
                    "R": 1.2,
                    "temperature": 25.0,
                    "reference_csv": "reference_current_2.csv",
                    "pid": {
                        "Kp": 8.0,
                        "Ki": 0.3,
                        "Kd": 0.02
                    }
                }
            ],
            "mutual_inductances": [
                [0.1, 0.02],
                [0.02, 0.15]
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Sample coupled circuits configuration created: {filename}")


class ValidationHelper:
    """Utility functions for validating arguments and configurations"""
    
    @staticmethod
    def validate_file_exists(filepath: str, description: str = "File"):
        """Validate that a file exists"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{description} not found: {filepath}")
        
        if not os.path.isfile(filepath):
            raise ValueError(f"{description} is not a file: {filepath}")
    
    @staticmethod
    def validate_time_range(circuit, time_params: TimeParameters) -> TimeParameters:
        """Validate and potentially adjust time range for circuit simulation"""
        # Check if time range is reasonable
        if time_params.span > 100:
            print(f"⚠️ Long simulation time ({time_params.span:.1f}s). Consider reducing for faster execution.")
        
        if time_params.step > time_params.span / 10:
            print(f"⚠️ Large time step ({time_params.step:.3f}s) relative to simulation span. Consider reducing.")
        
        return time_params
    
    @staticmethod
    def validate_initial_values(initial_values: List[float], n_circuits: int):
        """Validate initial values match circuit count"""
        if len(initial_values) != n_circuits:
            if len(initial_values) == 1:
                # Extend single value to all circuits
                return initial_values * n_circuits
            else:
                raise ValueError(
                    f"Number of initial values ({len(initial_values)}) "
                    f"must match number of circuits ({n_circuits})"
                )
        return initial_values


def create_time_parameters_from_args(args) -> TimeParameters:
    """Convert command line arguments to TimeParameters"""
    return TimeParameters(
        start=args.time_start,
        end=args.time_end,
        step=args.time_step,
        method=args.method
    )


def create_output_options_from_args(args) -> OutputOptions:
    """Convert command line arguments to enhanced OutputOptions"""
    options = OutputOptions(
        show_plots=args.show_plots,
        save_plots=args.save_plots,
        show_analytics=getattr(args, 'show_analytics', False),
        save_results=args.save_results,
        debug=getattr(args, 'debug', False),
        
        # New plotting configuration options
        plot_config=getattr(args, 'plot_config', None),
        comparison_mode=getattr(args, 'comparison_mode', False),
        benchmark_plotting=getattr(args, 'benchmark_plotting', False),
        plot_profile=getattr(args, 'plot_profile', 'default')
    )
    options.validate()
    return options


def create_plotting_config_from_args(args) -> PlotConfiguration:
    """Create PlotConfiguration from command line arguments"""
    from .plotting_strategies import PlotConfiguration
    
    # Start with base configuration
    config = PlotConfiguration()
    
    # Apply profile-based configuration
    if hasattr(args, 'plot_profile'):
        config = _apply_plot_profile(config, args.plot_profile)
    
    # Override with debug settings
    if getattr(args, 'debug', False):
        config.show_experimental = True
        config.show_regions = True
        config.show_temperature = True
        config.dpi = 150  # Higher DPI for debug analysis
    
    # Override with save settings
    if getattr(args, 'save_plots', None):
        config.dpi = 300  # High DPI for saved plots
    
    # Load custom configuration if provided
    if getattr(args, 'plot_config', None):
        config = _load_custom_plot_config(args.plot_config, config)
    
    return config


def _apply_plot_profile(config: PlotConfiguration, profile: str) -> PlotConfiguration:
    """Apply predefined plotting profile"""
    from .plotting_strategies import PlotConfiguration
    
    if profile == "publication":
        return PlotConfiguration(
            figsize=(12, 16),
            dpi=600,
            show_experimental=True,
            show_regions=True,
            show_temperature=True,
            linewidth_main=1.5,
            linewidth_experimental=1.0,
            alpha_experimental=0.8,
            alpha_regions=0.15,
            grid_alpha=0.2
        )
    elif profile == "presentation":
        return PlotConfiguration(
            figsize=(16, 12),
            dpi=150,
            show_experimental=False,
            show_regions=True,
            show_temperature=False,
            linewidth_main=3.0,
            alpha_regions=0.3,
            grid_alpha=0.4
        )
    elif profile == "debug":
        return PlotConfiguration(
            figsize=(20, 24),
            dpi=150,
            show_experimental=True,
            show_regions=True,
            show_temperature=True,
            alpha_experimental=0.9,
            alpha_regions=0.4,
            grid_alpha=0.5
        )
    elif profile == "minimal":
        return PlotConfiguration(
            figsize=(12, 8),
            dpi=100,
            show_experimental=False,
            show_regions=False,
            show_temperature=False,
            linewidth_main=1.0
        )
    else:  # default
        return config


def _load_custom_plot_config(config_file: str, base_config: PlotConfiguration) -> PlotConfiguration:
    """Load custom plotting configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            custom_config = json.load(f)
        
        # Update base config with custom values
        for key, value in custom_config.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
            else:
                print(f"⚠️ Unknown plotting configuration option: {key}")
        
        print(f"✓ Custom plotting configuration loaded from: {config_file}")
        return base_config
    
    except Exception as e:
        print(f"⚠️ Failed to load custom plotting configuration: {e}")
        print("Using default configuration")
        return base_config


def handle_common_cli_tasks(args) -> bool:
    """
    Handle common CLI tasks that might exit early
    Returns True if the program should continue, False if it should exit
    """
    # Handle working directory early
    if hasattr(args, 'wd') and args.wd:
        # This will be handled by WorkingDirectoryManager context
        pass
    
    # Handle sample creation
    if hasattr(args, 'create_sample') and args.create_sample:
        if hasattr(args, 'config_file') and 'coupled' in str(args.config_file):
            SampleConfigGenerator.create_coupled_circuits_config()
        else:
            SampleConfigGenerator.create_single_circuit_config()
        return False  # Exit after creating sample
    
    return True  # Continue execution


def print_simulation_header(title: str, config_file: str):
    """Print a consistent header for simulations"""
    print(f"\n=== {title} ===")
    if config_file:
        print(f"Configuration: {config_file}")
    else:
        print("Using command-line parameters")
