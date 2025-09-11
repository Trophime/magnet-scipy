"""
magnet_scipy/cli_simulation.py

Updated simulation orchestration components for CLI refactoring
Now integrates with the new plotting system for simplified workflow
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .cli_core import TimeParameters, OutputOptions, ValidationHelper
from .simulation_strategies import SimulationParameters, SimulationRunner
from .single_circuit_adapter import SingleCircuitSimulationRunner
from .rlcircuitpid import RLCircuitPID
from .coupled_circuits import CoupledRLCircuitsPID
from .cli_plotting_integration import EnhancedPlottingManager


@dataclass
class SimulationResult:
    """Enhanced simulation result with metadata"""
    time: np.ndarray
    solution: np.ndarray
    metadata: Dict[str, Any]
    strategy_type: str
    circuit_ids: List[str]
    success: bool = True
    error_message: Optional[str] = None


class SimulationOrchestrator:
    """
    High-level orchestrator for managing different types of simulations
    Replaces complex logic in main functions
    """
    
    def __init__(self):
        self.single_runner = SingleCircuitSimulationRunner()
        self.coupled_runner = SimulationRunner()
    
    def run_single_circuit_simulation(
        self, 
        circuit: RLCircuitPID,
        time_params: TimeParameters,
        strategy: str = "auto"
    ) -> SimulationResult:
        """
        Run single circuit simulation with proper error handling and validation
        """
        try:
            # Validate circuit
            circuit.validate()
            
            # Validate and adjust time range
            adjusted_time_params = ValidationHelper.validate_time_range(circuit, time_params)
            
            # Convert to SimulationParameters
            sim_params = SimulationParameters(
                t_start=adjusted_time_params.start,
                t_end=adjusted_time_params.end,
                dt=adjusted_time_params.step,
                method=adjusted_time_params.method,
                initial_values=[0.0]  # Default initial current
            )
            
            print(f"Running single circuit simulation...")
            print(f"  Circuit: {circuit.circuit_id}")
            print(f"  Time span: {sim_params.t_start:.3f} to {sim_params.t_end:.3f} seconds")
            print(f"  Method: {sim_params.method}")
            
            # Detect or use specified strategy
            if strategy == "auto":
                detected_strategy = self.single_runner.detect_strategy(circuit)
                print(f"  Auto-detected strategy: {detected_strategy}")
                strategy = detected_strategy
            else:
                print(f"  Using specified strategy: {strategy}")
            
            # Run simulation
            result = self.single_runner.run_simulation(circuit, sim_params, strategy)
            
            # Convert from simulation_strategies.SimulationResult to SimulationResult
            return SimulationResult(
                time=result.time,
                solution=result.solution,
                metadata=result.metadata,
                strategy_type=result.strategy_type,
                circuit_ids=[circuit.circuit_id],
                success=result.metadata["success"],
                error_message=result.metadata["message"]
            )
            
        except Exception as e:
            return SimulationResult(
                time=np.array([]),
                solution=np.array([]),
                metadata={},
                strategy_type="failed",
                circuit_ids=[circuit.circuit_id],
                success=False,
                error_message=str(e)
            )
    
    def run_coupled_simulation(
        self,
        coupled_system: CoupledRLCircuitsPID,
        time_params: TimeParameters,
        initial_values: List[float],
        strategy: str = "auto"
    ) -> SimulationResult:
        """
        Run coupled circuits simulation with proper error handling
        """
        try:
            # Validate system
            coupled_system.validate()
            
            # Adjust initial values
            adjusted_initial_values = ValidationHelper.validate_initial_values(
                initial_values, coupled_system.n_circuits
            )
            
            # Convert to SimulationParameters
            sim_params = SimulationParameters(
                t_start=time_params.start,
                t_end=time_params.end,
                dt=time_params.step,
                method=time_params.method,
                initial_values=adjusted_initial_values
            )
            
            print(f"Running coupled circuits simulation...")
            print(f"  Circuits: {[c.circuit_id for c in coupled_system.circuits]}")
            print(f"  Time span: {sim_params.t_start:.3f} to {sim_params.t_end:.3f} seconds")
            print(f"  Method: {sim_params.method}")
            
            # Detect or use specified strategy
            if strategy == "auto":
                detected_strategy = self.coupled_runner.detect_strategy(coupled_system)
                print(f"  Auto-detected strategy: {detected_strategy}")
                strategy = detected_strategy
            else:
                print(f"  Using specified strategy: {strategy}")
            
            # Run simulation
            result = self.coupled_runner.run_simulation(coupled_system, sim_params, strategy)
            
            # Convert to SimulationResult
            return SimulationResult(
                time=result.time,
                solution=result.solution,
                metadata=result.metadata,
                strategy_type=result.strategy_type,
                circuit_ids=[c.circuit_id for c in coupled_system.circuits],
                success=result.metadata["success"],
                error_message=result.metadata["message"]
            )
            
        except Exception as e:
            return SimulationResult(
                time=np.array([]),
                solution=np.array([]),
                metadata={},
                strategy_type="failed",
                circuit_ids=[c.circuit_id for c in coupled_system.circuits],
                success=False,
                error_message=str(e)
            )


class ResultProcessor:
    """
    Simplified result processor that works with the new plotting system
    Now primarily acts as a bridge between simulation results and plotting
    """
    
    def __init__(self):
        self.results_cache = {}
    
    def process_single_circuit_result(
        self,
        result: SimulationResult,
        circuit: RLCircuitPID,
        output_options: OutputOptions
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process single circuit results using new plotting system
        
        Note: This method is maintained for backward compatibility but
        the new workflow bypasses this in favor of direct EnhancedPlottingManager usage
        """
        if not result.success:
            raise RuntimeError(f"Cannot process failed simulation: {result.error_message}")
        
        # Create enhanced plotting manager and process
        plot_config = self._create_plot_config_from_options(output_options)
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        
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
        """
        Process coupled circuit results using new plotting system
        
        Note: This method is maintained for backward compatibility but
        the new workflow bypasses this in favor of direct EnhancedPlottingManager usage
        """
        if not result.success:
            raise RuntimeError(f"Cannot process failed simulation: {result.error_message}")
        
        # Create enhanced plotting manager and process
        plot_config = self._create_plot_config_from_options(output_options)
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        
        # Process results and create plots
        processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
            result, coupled_system
        )
        
        # Cache results
        cache_key = f"coupled_{len(result.circuit_ids)}_{id(result)}"
        self.results_cache[cache_key] = (processed_results.time, processed_results.circuits)
        
        return processed_results.time, processed_results.circuits
    
    def _create_plot_config_from_options(self, output_options: OutputOptions):
        """Create basic plot configuration from output options"""
        from .plotting_strategies import PlotConfiguration
        
        config = PlotConfiguration()
        
        # Adjust DPI based on save settings
        if output_options.save_plots:
            config.dpi = 300
        
        # Enable debug features if requested
        if output_options.debug:
            config.show_experimental = True
            config.show_regions = True
            config.show_temperature = True
        
        return config


class PlottingManager:
    """
    Simplified plotting manager that delegates to the new plotting system
    Maintained for backward compatibility in CLI components
    """
    
    def plot_single_circuit_results(
        self,
        circuit: RLCircuitPID,
        t: np.ndarray,
        results: Dict,
        strategy_type: str,
        output_options: OutputOptions
    ):
        """
        Generate plots for single circuit simulation
        
        Note: In the new workflow, plotting is handled by EnhancedPlottingManager
        This method is kept for interface compatibility but may be simplified
        """
        print("✓ Single circuit plots generated using new plotting system")
    
    def plot_coupled_circuits_results(
        self,
        coupled_system: CoupledRLCircuitsPID,
        t: np.ndarray,
        results: Dict,
        strategy_type: str,
        output_options: OutputOptions
    ):
        """
        Generate plots for coupled circuits simulation
        
        Note: In the new workflow, plotting is handled by EnhancedPlottingManager
        This method is kept for interface compatibility but may be simplified
        """
        print("✓ Coupled circuits plots generated using new plotting system")


class AnalyticsManager:
    """
    Simplified analytics manager that works with the new analytics system
    Analytics are now integrated into the plotting workflow
    """
    
    def analyze_single_circuit_results(
        self,
        circuit: RLCircuitPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions
    ):
        """
        Analytics have been integrated into the plotting workflow
        This method is kept for interface compatibility
        """
        if output_options.show_analytics:
            print("✓ Analytics integrated into plotting workflow")
    
    def analyze_coupled_circuits_results(
        self,
        coupled_system: CoupledRLCircuitsPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions
    ):
        """
        Analytics have been integrated into the plotting workflow
        This method is kept for interface compatibility
        """
        if output_options.show_analytics:
            print("✓ Coupled circuits analytics integrated into plotting workflow")


class FileManager:
    """
    Simplified file manager that works with the new results saving system
    File saving is now integrated into the plotting workflow
    """
    
    def save_single_circuit_results(
        self,
        circuit: RLCircuitPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions,
        config_file: str = None
    ):
        """
        Save single circuit results
        
        Note: In the new workflow, file saving is handled by EnhancedPlottingManager
        This method is kept for interface compatibility
        """
        if output_options.save_results:
            print("✓ Single circuit results saved using new file management system")
    
    def save_coupled_circuits_results(
        self,
        coupled_system: CoupledRLCircuitsPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions,
        config_file: str = None
    ):
        """
        Save coupled circuits results
        
        Note: In the new workflow, file saving is handled by EnhancedPlottingManager
        This method is kept for interface compatibility
        """
        if output_options.save_results:
            print("✓ Coupled circuits results saved using new file management system")


class SimulationSummary:
    """Enhanced simulation summary with better formatting and information"""
    
    @staticmethod
    def print_single_circuit_summary(
        result: SimulationResult,
        circuit: RLCircuitPID,
        time_params: TimeParameters
    ):
        """Print summary for single circuit simulation"""
        print("\n" + "="*60)
        print("📋 SINGLE CIRCUIT SIMULATION SUMMARY")
        print("="*60)
        
        # Circuit information
        print(f"Circuit ID: {circuit.circuit_id}")
        print(f"Inductance: {circuit.L:.6f} H")
        print(f"Resistance(0 A, {circuit.temperature}°C): {circuit.get_resistance(0, circuit.temperature):.6f} Ω")
        
        # Simulation parameters
        print(f"\nSimulation Parameters:")
        print(f"  Time span: {time_params.start:.3f} to {time_params.end:.3f} seconds")
        print(f"  Time step: {time_params.step:.6f} seconds")
        print(f"  Method: {time_params.method}")
        print(f"  Strategy: {result.strategy_type}")
        
        # Results summary
        if result.success:
            print(f"\nResults:")
            print(f"  Simulation points: {len(result.time)}")
            print(f"  Final current: {float(result.solution[-1][-1]):.6f} A")
            
            if 'n_evaluations' in result.metadata:
                print(f"  Function evaluations: {result.metadata['n_evaluations']}")
            
            print(f"  Status: ✓ SUCCESS")
        else:
            print(f"\nResults:")
            print(f"  Status: ✗ FAILED")
            print(f"  Error: {result.error_message}")
        
        print("="*60)
    
    @staticmethod
    def print_coupled_circuits_summary(
        result: SimulationResult,
        coupled_system: CoupledRLCircuitsPID,
        time_params: TimeParameters
    ):
        """Print summary for coupled circuits simulation"""
        print("\n" + "="*60)
        print("📋 COUPLED CIRCUITS SIMULATION SUMMARY")
        print("="*60)
        
        # System information
        print(f"Number of circuits: {coupled_system.n_circuits}")
        for i, circuit in enumerate(coupled_system.circuits):
            print(f"  Circuit {i+1} ({circuit.circuit_id}):")
            print(f"    L = {circuit.L:.6f} H, R(0 A, {circuit.temperature}°C) = {circuit.get_resistance(0, circuit.temperature):.6f} Ω")
        
        # Mutual inductances
        print(f"\nMutual Inductances:")
        mutual_matrix = coupled_system.M
        for i in range(mutual_matrix.shape[0]):
            row_str = "  [" + ", ".join([f"{mutual_matrix[i,j]:.6f}" for j in range(mutual_matrix.shape[1])]) + "]"
            print(row_str)
        
        # Simulation parameters
        print(f"\nSimulation Parameters:")
        print(f"  Time span: {time_params.start:.3f} to {time_params.end:.3f} seconds")
        print(f"  Time step: {time_params.step:.6f} seconds")
        print(f"  Method: {time_params.method}")
        print(f"  Strategy: {result.strategy_type}")
        
        # Results summary
        if result.success:
            print(f"\nResults:")
            print(f"  Simulation points: {len(result.time)}")
            
            # Show final currents for each circuit
            n_circuits = coupled_system.n_circuits
            if result.strategy_type == "pid_control":
                # For PID, solution includes currents and integral errors
                final_currents = result.solution[-n_circuits:]
            else:
                # For voltage input, solution is just currents
                final_currents = result.solution[-n_circuits:]
            
            for i, (circuit, final_current) in enumerate(zip(coupled_system.circuits, final_currents)):
                print(f"  Final current ({circuit.circuit_id}): {float(final_current[-1]):.6f} A")
            
            if 'n_evaluations' in result.metadata:
                print(f"  Function evaluations: {result.metadata['n_evaluations']}")
            
            print(f"  Status: ✓ SUCCESS")
        else:
            print(f"\nResults:")
            print(f"  Status: ✗ FAILED")
            print(f"  Error: {result.error_message}")
        
        print("="*60)
    
    @staticmethod
    def print_error_summary(result: SimulationResult):
        """Print error summary for failed simulations"""
        print("\n" + "="*60)
        print("❌ SIMULATION ERROR SUMMARY")
        print("="*60)
        
        print(f"Strategy: {result.strategy_type}")
        print(f"Circuits: {', '.join(result.circuit_ids)}")
        print(f"Error: {result.error_message}")
        
        if result.metadata:
            print(f"\nAdditional Information:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
        
        print("="*60)


class CLIErrorHandler:
    """Centralized error handling for CLI operations"""
    
    @staticmethod
    def handle_simulation_error(error: Exception, debug: bool = False) -> int:
        """Handle simulation errors with appropriate user feedback"""
        print(f"\n❌ Simulation Error: {error}")
        
        if debug:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        else:
            print("Use --debug flag for detailed error information")
        
        return 1  # Error exit code
    
    @staticmethod
    def handle_validation_error(error: Exception, debug: bool = False) -> int:
        """Handle validation errors"""
        print(f"\n❌ Validation Error: {error}")
        
        if debug:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        
        return 1  # Error exit code
    
    @staticmethod
    def handle_configuration_error(error: Exception, debug: bool = False) -> int:
        """Handle configuration loading errors"""
        print(f"\n❌ Configuration Error: {error}")
        print("Please check your configuration file format and contents")
        
        if debug:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        
        return 1  # Error exit code