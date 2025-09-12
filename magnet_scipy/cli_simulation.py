"""
magnet_scipy/cli_simulation.py

Version 3.0: Clean simulation orchestration components with backward compatibility removed
Breaking changes: All legacy classes removed, only essential components remain
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .cli_core import TimeParameters, OutputOptions, ValidationHelper
from .simulation_types import SimulationResult, SimulationParameters
from .simulation_strategies import SimulationRunner
from .single_circuit_adapter import SingleCircuitSimulationRunner
from .rlcircuitpid import RLCircuitPID
from .coupled_circuits import CoupledRLCircuitsPID
from .cli_plotting_integration import EnhancedPlottingManager


@dataclass
class SimulationResult:
    """Enhanced simulation result with metadata - Version 3.0 unified structure"""
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
    Replaces complex logic in main functions - Version 3.0 core component
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
            
            # Convert to enhanced SimulationResult
            return SimulationResult(
                time=result.time,
                solution=result.solution,
                metadata=result.metadata,
                strategy_type=result.strategy_type,
                circuit_ids=[circuit.circuit_id],
                success=result.metadata["success"],
                error_message=result.metadata["message"] if not result.metadata["success"] else None
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
        Run coupled circuits simulation with proper error handling and validation
        """
        try:
            # Validate system
            coupled_system.validate()
            
            # Validate and adjust time range
            adjusted_time_params = ValidationHelper.validate_time_range(coupled_system, time_params)
            
            # Convert to SimulationParameters
            sim_params = SimulationParameters(
                t_start=adjusted_time_params.start,
                t_end=adjusted_time_params.end,
                dt=adjusted_time_params.step,
                method=adjusted_time_params.method,
                initial_values=initial_values
            )
            
            print(f"Running coupled circuits simulation...")
            print(f"  Circuits: {coupled_system.n_circuits}")
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
            
            # Convert to enhanced SimulationResult
            return SimulationResult(
                time=result.time,
                solution=result.solution,
                metadata=result.metadata,
                strategy_type=result.strategy_type,
                circuit_ids=[c.circuit_id for c in coupled_system.circuits],
                success=result.metadata.get("success", True),
                error_message=result.metadata.get("message", None) if not result.metadata.get("success", True) else None
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


class SimulationSummary:
    """
    Generate and display simulation summaries
    Version 3.0 core component - kept for reporting functionality
    """
    
    @staticmethod
    def print_single_circuit_summary(
        result: SimulationResult,
        circuit: RLCircuitPID,
        time_params: TimeParameters
    ):
        """Print summary for single circuit simulation"""
        print("\n" + "="*60)
        print("SINGLE CIRCUIT SIMULATION SUMMARY")
        print("="*60)
        
        print(f"Circuit ID: {circuit.circuit_id}")
        print(f"Strategy: {result.strategy_type}")
        print(f"Success: {'✓' if result.success else '✗'}")
        
        if result.success:
            print(f"Time points: {len(result.time)}")
            print(f"Final time: {result.time[-1]:.3f} s")
            print(f"Final current: {result.solution[-1, 0]:.3f} A")
        else:
            print(f"Error: {result.error_message}")
        
        print("="*60)
    
    @staticmethod
    def print_coupled_circuits_summary(
        result: SimulationResult,
        coupled_system: CoupledRLCircuitsPID,
        time_params: TimeParameters
    ):
        """Print summary for coupled circuits simulation"""
        print("\n" + "="*60)
        print("COUPLED CIRCUITS SIMULATION SUMMARY")
        print("="*60)
        
        print(f"Number of circuits: {coupled_system.n_circuits}")
        print(f"Strategy: {result.strategy_type}")
        print(f"Success: {'✓' if result.success else '✗'}")
        
        if result.success:
            print(f"Time points: {len(result.time)}")
            print(f"Final time: {result.time[-1]:.3f} s")
            
            # Print final currents for each circuit
            for i, circuit_id in enumerate(result.circuit_ids):
                final_current = result.solution[-1, i * 2]  # Assuming current is first state variable
                print(f"Final current [{circuit_id}]: {final_current:.3f} A")
        else:
            print(f"Error: {result.error_message}")
        
        print("="*60)
    
    @staticmethod
    def print_error_summary(result: SimulationResult):
        """Print error summary for failed simulations"""
        print("\n" + "="*60)
        print("SIMULATION ERROR SUMMARY")
        print("="*60)
        print(f"Strategy: {result.strategy_type}")
        print(f"Circuit(s): {', '.join(result.circuit_ids)}")
        print(f"Error: {result.error_message}")
        print("="*60)


class CLIErrorHandler:
    """
    Centralized error handling for CLI operations
    Version 3.0 core component - kept for error management
    """
    
    @staticmethod
    def handle_validation_error(error: Exception, debug: bool = False) -> int:
        """Handle validation errors with optional debug info"""
        print(f"✗ Validation Error: {error}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1
    
    @staticmethod
    def handle_simulation_error(error: Exception, debug: bool = False) -> int:
        """Handle simulation errors with optional debug info"""
        print(f"✗ Simulation Error: {error}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1
    
    @staticmethod
    def handle_plotting_error(error: Exception, debug: bool = False) -> int:
        """Handle plotting errors with optional debug info"""
        print(f"✗ Plotting Error: {error}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1


# Version 3.0 Breaking Changes Notice
# 
# REMOVED CLASSES (no longer available):
# - ResultProcessor → Use EnhancedPlottingManager.create_plots_from_simulation_result()
# - PlottingManager → Use EnhancedPlottingManager.create_plots()
# - AnalyticsManager → Analytics integrated into plotting workflow
# - FileManager → File saving integrated into plotting workflow
#
# MOVED CLASSES:
# - SimulationResult → Moved to simulation_types.py to resolve circular imports
# - SimulationParameters → Moved to simulation_types.py for consistency
#
# MIGRATION GUIDE:
# Old: processor = ResultProcessor()
#      t, data = processor.process_single_circuit_result(result, circuit, options)
# New: plotting_manager = EnhancedPlottingManager(options)
#      processed_results, analytics = plotting_manager.create_plots_from_simulation_result(result, circuit)
#      t, data = processed_results.time, processed_results.circuits[circuit.circuit_id]
#
# IMPORT CHANGES:
# Old: from .cli_simulation import SimulationResult
# New: from .simulation_types import SimulationResult
