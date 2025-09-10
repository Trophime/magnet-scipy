"""
magnet_scipy/cli_core.py

Core CLI components for separating concerns in main functions
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
    """Encapsulate all output-related options"""
    show_plots: bool = False
    save_plots: Optional[str] = None
    show_analytics: bool = False
    save_results: Optional[str] = None
    debug: bool = False
    
    def validate(self):
        """Validate output options and provide warnings"""
        if not self.show_plots and not self.save_plots:
            print("⚠️ Neither --save-plots nor --show-plots specified. Forcing show_plots.")
            self.show_plots = True
        
        if self.save_plots and self.show_plots:
            print("⚠️ Both --save-plots and --show-plots specified. Plots will be saved and shown.")


class ArgumentParser:
    """Centralized argument parsing for both single and coupled circuits"""
    
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
        """Add output-related arguments"""
        parser.add_argument(
            "--show_plots",
            action="store_true",
            help="Show plots of simulation results",
        )
        
        parser.add_argument(
            "--show_analytics",
            "-a",
            action="store_true",
            help="Show detailed analytics of simulation results",
        )
        
        parser.add_argument(
            "--save_results",
            type=str,
            help="Save results to specified file (e.g., results.npz)",
        )
        
        parser.add_argument(
            "--save_plots",
            type=str,
            help="Save plots to specified file (e.g., plots.png)",
        )
        
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode with additional output",
        )


class ConfigurationLoader:
    """Handle loading and validation of configuration files"""
    
    @staticmethod
    def load_single_circuit(config_file: str) -> RLCircuitPID:
        """Load single circuit configuration from JSON file"""
        try:
            with open(config_file, "r") as f:
                circuit_data = json.load(f)

            # Create PID controller if parameters provided
            pid_controller = None
            if "pid_params" in circuit_data:
                pid_params = circuit_data["pid_params"]
                pid_controller = create_adaptive_pid_controller(**pid_params)

            # Create RLCircuitPID instance
            circuit = RLCircuitPID(
                R=circuit_data.get("resistance", 1.0),
                L=circuit_data.get("inductance", 0.1),
                pid_controller=pid_controller,
                reference_csv=circuit_data.get("reference_csv", None),
                voltage_csv=circuit_data.get("voltage_csv", None),
                resistance_csv=circuit_data.get("resistance_csv"),
                temperature=circuit_data.get("temperature", 25.0),
                temperature_csv=circuit_data.get("temperature_csv", None),
                circuit_id=circuit_data.get("circuit_id", "single_circuit"),
                experimental_data=circuit_data.get("experiment_data", []),
            )
            
            return circuit

        except Exception as e:
            raise RuntimeError(f"Error loading configuration file {config_file}: {e}")
    
    @staticmethod
    def load_coupled_circuits(config_file: str) -> Tuple[List[RLCircuitPID], np.ndarray]:
        """Load coupled circuits configuration from JSON file"""
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)

            circuits = []
            for circuit_data in config_data["circuits"]:
                # Create PID controller if parameters provided
                pid_controller = None
                if "pid_params" in circuit_data:
                    pid_params = circuit_data["pid_params"]
                    pid_controller = create_adaptive_pid_controller(**pid_params)

                # Create RLCircuitPID instance
                circuit = RLCircuitPID(
                    R=circuit_data.get("resistance", 1.0),
                    L=circuit_data.get("inductance", 0.1),
                    pid_controller=pid_controller,
                    reference_csv=circuit_data.get("reference_csv", None),
                    voltage_csv=circuit_data.get("voltage_csv", None),
                    resistance_csv=circuit_data.get("resistance_csv"),
                    temperature=circuit_data.get("temperature", 25.0),
                    temperature_csv=circuit_data.get("temperature_csv", None),
                    circuit_id=circuit_data.get("circuit_id", f"circuit_{len(circuits)+1}"),
                    experimental_data=circuit_data.get("experiment_data", []),
                )
                circuits.append(circuit)

            # Load mutual inductance matrix
            mutual_inductances = None
            if "mutual_inductances" in config_data:
                mutual_inductances = np.array(config_data["mutual_inductances"])

            return circuits, mutual_inductances

        except Exception as e:
            raise RuntimeError(f"Error loading configuration file {config_file}: {e}")


class WorkingDirectoryManager:
    """Handle working directory changes safely"""
    
    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = working_dir
        self.original_dir = None
    
    def __enter__(self):
        if self.working_dir is not None:
            self.original_dir = os.getcwd()
            os.chdir(self.working_dir)
            print(f"✓ Working directory set to: {self.working_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_dir is not None:
            os.chdir(self.original_dir)
            print(f"✓ Returned to original directory: {self.original_dir}")


class SampleConfigGenerator:
    """Generate sample configuration files for testing"""
    
    @staticmethod
    def create_single_circuit_config(filename: str = "sample_single_circuit.json") -> str:
        """Create a sample single circuit configuration file"""
        config = {
            "circuit_id": "sample_circuit",
            "inductance": 0.1,
            "resistance": 1.5,
            "temperature": 25.0,
            "pid_params": {
                "Kp_low": 20.0,
                "Ki_low": 15.0,
                "Kd_low": 0.1,
                "Kp_medium": 12.0,
                "Ki_medium": 8.0,
                "Kd_medium": 0.05,
                "Kp_high": 8.0,
                "Ki_high": 5.0,
                "Kd_high": 0.02,
                "low_threshold": 60.0,
                "high_threshold": 800.0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Created sample single circuit configuration: {filename}")
        return filename
    
    @staticmethod
    def create_coupled_circuits_config(n_circuits: int = 3, filename: str = "sample_coupled_circuits.json") -> str:
        """Create a sample coupled circuits configuration file"""
        config = {
            "name": f"Sample_{n_circuits}_Circuits",
            "circuits": [],
            "mutual_inductances": []
        }
        
        # Create circuit configurations
        for i in range(n_circuits):
            circuit_config = {
                "circuit_id": f"circuit_{i+1}",
                "inductance": 0.08 + 0.02 * i,
                "resistance": 1.0 + 0.2 * i,
                "temperature": 25.0 + 5.0 * i,
                "pid_params": {
                    "Kp_low": 15.0 + 2.0 * i,
                    "Ki_low": 8.0 + 1.0 * i,
                    "Kd_low": 0.08 + 0.01 * i,
                    "Kp_medium": 18.0 + 2.0 * i,
                    "Ki_medium": 10.0 + 1.0 * i,
                    "Kd_medium": 0.06 + 0.01 * i,
                    "Kp_high": 22.0 + 2.0 * i,
                    "Ki_high": 12.0 + 1.0 * i,
                    "Kd_high": 0.04 + 0.01 * i,
                    "low_threshold": 60.0,
                    "high_threshold": 200.0 + 100.0 * i,
                },
            }
            config["circuits"].append(circuit_config)

        # Create mutual inductance matrix (off-diagonal terms only)
        coupling_strength = 0.02
        extra_diag_terms = int(n_circuits * (n_circuits - 1) / 2)
        if extra_diag_terms > 0:
            config["mutual_inductances"] = [coupling_strength] * extra_diag_terms
        else:
            config["mutual_inductances"] = []

        with open(filename, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✓ Created sample coupled circuits configuration: {filename}")
        return filename


class ValidationHelper:
    """Helper functions for validation and error checking"""
    
    @staticmethod
    def validate_file_exists(filepath: str, description: str = "File"):
        """Validate that a file exists"""
        if not filepath:
            raise ValueError(f"{description} path cannot be empty")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{description} not found: {filepath}")
    
    @staticmethod
    def validate_time_range(circuit: RLCircuitPID, time_params: TimeParameters) -> TimeParameters:
        """
        Validate and adjust time range based on available data
        Returns updated TimeParameters if adjustments were made
        """
        t_start, t_end = time_params.start, time_params.end
        time_range_modified = False
        
        # Check if circuit has time-dependent data
        if hasattr(circuit, 'time_data') and circuit.time_data is not None:
            data_t_min = float(circuit.time_data[0])
            data_t_max = float(circuit.time_data[-1])
            
            if t_start < data_t_min:
                print(f"⚠️ Warning: Requested start time {t_start} is before data start {data_t_min}")
                t_start = data_t_min
                time_range_modified = True
                
            if t_end > data_t_max:
                print(f"⚠️ Warning: Requested end time {t_end} is after data end {data_t_max}")
                t_end = data_t_max
                time_range_modified = True
        
        if time_range_modified:
            updated_params = TimeParameters(
                start=t_start, 
                end=t_end, 
                step=time_params.step,
                method=time_params.method
            )
            print(f"✓ Adjusted time range: {t_start} to {t_end} seconds")
            return updated_params
        else:
            print(f"✓ Time range: {t_start} to {t_end} seconds")
            return time_params
    
    @staticmethod
    def validate_initial_values(initial_values: List[float], expected_count: int) -> List[float]:
        """Validate initial values match expected circuit count"""
        if len(initial_values) != expected_count:
            raise ValueError(
                f"Number of initial values ({len(initial_values)}) must match "
                f"number of circuits ({expected_count})"
            )
        return initial_values


class ResultManager:
    """Handle saving and organizing simulation results"""
    
    @staticmethod
    def generate_output_filename(config_file: str, suffix: str = ".res") -> str:
        """Generate output filename based on config file"""
        if config_file:
            return config_file.replace(".json", suffix)
        else:
            return f"simulation_results{suffix}"
    
    @staticmethod
    def save_simulation_summary(
        output_dir: str,
        circuit_ids: List[str],
        time_params: TimeParameters,
        simulation_info: Dict[str, Any]
    ):
        """Save a summary of the simulation parameters and results"""
        summary = {
            "circuits": circuit_ids,
            "time_parameters": {
                "start": time_params.start,
                "end": time_params.end,
                "step": time_params.step,
                "method": time_params.method,
                "total_time": time_params.span,
                "estimated_steps": time_params.n_steps
            },
            "simulation_info": simulation_info
        }
        
        summary_file = os.path.join(output_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✓ Simulation summary saved to: {summary_file}")


def create_time_parameters_from_args(args) -> TimeParameters:
    """Convert command line arguments to TimeParameters"""
    return TimeParameters(
        start=args.time_start,
        end=args.time_end,
        step=args.time_step,
        method=args.method
    )


def create_output_options_from_args(args) -> OutputOptions:
    """Convert command line arguments to OutputOptions"""
    options = OutputOptions(
        show_plots=args.show_plots,
        save_plots=args.save_plots,
        show_analytics=getattr(args, 'show_analytics', False),
        save_results=args.save_results,
        debug=getattr(args, 'debug', False)
    )
    options.validate()
    return options


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
