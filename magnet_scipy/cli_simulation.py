"""
magnet_scipy/cli_simulation.py

Simulation orchestration components for CLI refactoring
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .cli_core import TimeParameters, OutputOptions, ValidationHelper
from .simulation_strategies import SimulationParameters, SimulationRunner
from .single_circuit_adapter import SingleCircuitSimulationRunner
from .rlcircuitpid import RLCircuitPID
from .coupled_circuits import CoupledRLCircuitsPID


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
            else:
                detected_strategy = strategy
                print(f"  Using specified strategy: {detected_strategy}")
            
            # Run simulation
            result = self.single_runner.run_simulation(circuit, sim_params, detected_strategy)
            
            return SimulationResult(
                time=result.time,
                solution=result.solution,
                metadata=result.metadata,
                strategy_type=result.strategy_type,
                circuit_ids=[circuit.circuit_id],
                success=True
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
            coupled_system.print_configuration()
            
            # Validate initial values
            validated_initial_values = ValidationHelper.validate_initial_values(
                initial_values, coupled_system.n_circuits
            )
            
            # Convert to SimulationParameters
            sim_params = SimulationParameters(
                t_start=time_params.start,
                t_end=time_params.end,
                dt=time_params.step,
                method=time_params.method,
                initial_values=validated_initial_values
            )
            
            print(f"Running coupled circuits simulation...")
            print(f"  Circuits: {coupled_system.n_circuits}")
            print(f"  Circuit IDs: {coupled_system.circuit_ids}")
            print(f"  Time span: {sim_params.t_start:.3f} to {sim_params.t_end:.3f} seconds")
            print(f"  Method: {sim_params.method}")
            print(f"  Initial values: {sim_params.initial_values}")
            
            # Detect or use specified strategy
            if strategy == "auto":
                detected_strategy = self.coupled_runner.detect_strategy(coupled_system)
                print(f"  Auto-detected strategy: {detected_strategy}")
            else:
                detected_strategy = strategy
                print(f"  Using specified strategy: {detected_strategy}")
            
            # Run simulation
            result = self.coupled_runner.run_simulation(coupled_system, sim_params, detected_strategy)
            
            return SimulationResult(
                time=result.time,
                solution=result.solution,
                metadata=result.metadata,
                strategy_type=result.strategy_type,
                circuit_ids=coupled_system.circuit_ids,
                success=True
            )
            
        except Exception as e:
            return SimulationResult(
                time=np.array([]),
                solution=np.array([]),
                metadata={},
                strategy_type="failed",
                circuit_ids=coupled_system.circuit_ids,
                success=False,
                error_message=str(e)
            )


class ResultProcessor:
    """
    Handle post-processing of simulation results
    Separates plotting and analysis logic from main simulation loop
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
        Process single circuit simulation result for plotting and analysis
        """
        if not result.success:
            raise RuntimeError(f"Cannot process failed simulation: {result.error_message}")
        
        # Convert result back to expected format for plotting compatibility
        sol = self._convert_result_to_sol_format(result)
        
        # Determine mode for post-processing
        mode = "regular" if result.strategy_type == "voltage_input" else "cde"
        
        print("Processing single circuit results...")
        
        # Import here to avoid circular imports
        from .plotting import prepare_post
        t, processed_results = prepare_post(sol, circuit, mode)
        
        # Cache results
        cache_key = f"single_{circuit.circuit_id}_{id(result)}"
        self.results_cache[cache_key] = (t, processed_results, sol, circuit)
        
        return t, processed_results
    
    def process_coupled_circuits_result(
        self,
        result: SimulationResult,
        coupled_system: CoupledRLCircuitsPID,
        output_options: OutputOptions
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process coupled circuits simulation result for plotting and analysis
        """
        if not result.success:
            raise RuntimeError(f"Cannot process failed simulation: {result.error_message}")
        
        # Convert result back to expected format for plotting compatibility
        sol = self._convert_result_to_sol_format(result)
        
        # Determine mode for post-processing
        mode = "regular" if result.strategy_type == "voltage_input" else "cde"
        
        print("Processing coupled circuits results...")
        
        # Import here to avoid circular imports
        from .coupled_plotting import prepare_coupled_post
        t, processed_results = prepare_coupled_post(sol, coupled_system, mode)
        
        # Cache results
        cache_key = f"coupled_{len(result.circuit_ids)}_{id(result)}"
        self.results_cache[cache_key] = (t, processed_results, sol, coupled_system)
        
        return t, processed_results
    
    def _convert_result_to_sol_format(self, result: SimulationResult):
        """Convert SimulationResult back to scipy solve_ivp format for compatibility"""
        class FakeSol:
            def __init__(self, t, y, metadata):
                self.t = t
                self.y = y
                self.success = True
                self.message = "Strategy simulation completed"
                self.nfev = metadata.get('n_evaluations', 0)
        
        return FakeSol(result.time, result.solution, result.metadata)


class PlottingManager:
    """
    Manage plotting operations for different simulation types
    Centralizes plotting logic and handles different output options
    """
    
    def plot_single_circuit_results(
        self,
        circuit: RLCircuitPID,
        t: np.ndarray,
        results: Dict,
        strategy_type: str,
        output_options: OutputOptions
    ):
        """Generate plots for single circuit simulation"""
        print("Generating single circuit plots...")
        
        try:
            if strategy_type == "voltage_input":
                from .plotting import plot_vresults
                plot_vresults(
                    circuit,
                    t,
                    results,
                    save_path=output_options.save_plots,
                    show=output_options.show_plots,
                )
            else:
                # Import sol from cache or create minimal sol object
                from .plotting import plot_results
                from .utils import fake_sol
                currents  = results.get('current', np.array([]))
                integral_errors = results.get('integral_error', np.array([]))
                print('currents:', type(currents))
                print('integral_errors:', type(integral_errors))
                sol = fake_sol(t, [currents, integral_errors])
                print(f'restore sol: sol={sol.y}')
                
                plot_results(
                    sol,
                    circuit,
                    t,
                    results,
                    save_path=output_options.save_plots,
                    show=output_options.show_plots,
                )
            
            print("✓ Single circuit plots generated successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to generate plots: {e}")
            if output_options.debug:
                import traceback
                traceback.print_exc()
    
    def plot_coupled_circuits_results(
        self,
        coupled_system: CoupledRLCircuitsPID,
        t: np.ndarray,
        results: Dict,
        strategy_type: str,
        output_options: OutputOptions
    ):
        """Generate plots for coupled circuits simulation"""
        print("Generating coupled circuits plots...")
        
        try:
            if strategy_type == "voltage_input":
                from .coupled_plotting import plot_coupled_vresults
                from .utils import fake_sol

                currents = [results[key].get('current', np.array([])) for key in results]
                sol = fake_sol(t, np.stack(currents))
                
                plot_coupled_vresults(
                    sol,
                    coupled_system,
                    t,
                    results,
                    save_path=output_options.save_plots,
                    show=output_options.show_plots,
                )
            else:
                from .coupled_plotting import plot_coupled_results
                from .utils import fake_sol
                
                currents  = [results[key].get('current', np.array([])) for key in results]
                integral_errors = [results[key].get('integral_error', np.array([])) for key in results]
                np_currents= np.vstack(currents + integral_errors)
                sol = fake_sol(t, np_currents)
                
                
                plot_coupled_results(
                    sol,
                    coupled_system,
                    t,
                    results,
                    save_path=output_options.save_plots,
                    show=output_options.show_plots,
                )
                
            print("✓ Coupled circuits plots generated successfully")
                
        except Exception as e:
            print(f"⚠️ Warning: Failed to generate plots: {e}")
            if output_options.debug:
                import traceback
                traceback.print_exc()


class AnalyticsManager:
    """
    Handle analytics and detailed analysis of simulation results
    """
    
    def analyze_single_circuit_results(
        self,
        circuit: RLCircuitPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions
    ):
        """Perform analytics for single circuit simulation"""
        if not output_options.show_analytics:
            return
        
        print("\n=== Single Circuit Analytics ===")
        
        try:
            from .plotting import analyze
            analyze(circuit, t, results)
            
        except Exception as e:
           print(f"⚠️ Warning: Failed to generate analytics: {e}")
           if output_options.debug:
               import traceback
               traceback.print_exc()
    
    def analyze_coupled_circuits_results(
        self,
        coupled_system: CoupledRLCircuitsPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions
    ):
        """Perform analytics for coupled circuits simulation"""
        if not output_options.show_analytics:
            return
        
        print("\n=== Coupled Circuits Analytics ===")
        
        try:
            from .coupled_plotting import analyze_coupling_effects
            analyze_coupling_effects(coupled_system, t, results)
            
        except Exception as e:
           print(f"⚠️ Warning: Failed to generate analytics: {e}")
           if output_options.debug:
               import traceback
               traceback.print_exc()


class FileManager:
    """
    Handle file operations for saving results and managing outputs
    """
    
    def save_single_circuit_results(
        self,
        circuit: RLCircuitPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions,
        config_file: str = None
    ):
        """Save single circuit simulation results"""
        if not output_options.save_results:
            return
        
        try:
            # Determine output filename
            if output_options.save_results:
                output_filename = output_options.save_results
            else:
                from .cli_core import ResultManager
                output_filename = ResultManager.generate_output_filename(config_file)
            
            from .plotting import save_results
            save_results(circuit, t, results, output_filename)
            print(f"✓ Single circuit results saved to: {output_filename}")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to save results: {e}")
    
    def save_coupled_circuits_results(
        self,
        coupled_system: CoupledRLCircuitsPID,
        t: np.ndarray,
        results: Dict,
        output_options: OutputOptions,
        config_file: str = None
    ):
        """Save coupled circuits simulation results"""
        if not output_options.save_results:
            return
        
        try:
            # Determine output filename
            if output_options.save_results:
                output_filename = output_options.save_results
            else:
                from .cli_core import ResultManager
                output_filename = ResultManager.generate_output_filename(config_file)
            
            from .coupled_plotting import save_coupled_results
            save_coupled_results(coupled_system, t, results, output_filename)
            print(f"✓ Coupled circuits results saved to: {output_filename}")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to save results: {e}")


class SimulationSummary:
    """
    Generate and display simulation summaries and statistics
    """
    
    @staticmethod
    def print_single_circuit_summary(
        result: SimulationResult,
        circuit: RLCircuitPID,
        time_params: TimeParameters
    ):
        """Print summary for single circuit simulation"""
        print(f"\n✓ Single circuit simulation completed successfully!")
        print(f"  Circuit ID: {circuit.circuit_id}")
        print(f"  Strategy: {result.strategy_type}")
        print(f"  Time points: {len(result.time)}")
        print(f"  Total simulation time: {time_params.span:.3f} seconds")
        print(f"  Function evaluations: {result.metadata.get('n_evaluations', 'Unknown')}")
        
        # Print experimental data comparison if available
        if hasattr(circuit, 'experimental_data') and circuit.experimental_data:
            print(f"  Experimental data sets: {len(circuit.experimental_data)}")
    
    @staticmethod
    def print_coupled_circuits_summary(
        result: SimulationResult,
        coupled_system: CoupledRLCircuitsPID,
        time_params: TimeParameters
    ):
        """Print summary for coupled circuits simulation"""
        print(f"\n✓ Coupled circuits simulation completed successfully!")
        print(f"  Circuits: {coupled_system.n_circuits}")
        print(f"  Circuit IDs: {coupled_system.circuit_ids}")
        print(f"  Strategy: {result.strategy_type}")
        print(f"  Time points: {len(result.time)}")
        print(f"  Total simulation time: {time_params.span:.3f} seconds")
        print(f"  Function evaluations: {result.metadata.get('n_evaluations', 'Unknown')}")
        print(f"  Mutual inductance matrix: {coupled_system.M.shape}")
    
    @staticmethod
    def print_error_summary(result: SimulationResult):
        """Print error summary for failed simulation"""
        print(f"\n✗ Simulation failed!")
        print(f"  Error: {result.error_message}")
        print(f"  Strategy attempted: {result.strategy_type}")
        if result.circuit_ids:
            print(f"  Affected circuits: {result.circuit_ids}")


class CLIErrorHandler:
    """
    Centralized error handling for CLI operations
    """
    
    @staticmethod
    def handle_configuration_error(error: Exception, debug: bool = False):
        """Handle configuration loading errors"""
        print(f"✗ Configuration error: {error}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1  # Exit code
    
    @staticmethod
    def handle_simulation_error(error: Exception, debug: bool = False):
        """Handle simulation execution errors"""
        print(f"✗ Simulation error: {error}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1  # Exit code
    
    @staticmethod
    def handle_validation_error(error: Exception, debug: bool = False):
        """Handle validation errors"""
        print(f"✗ Validation error: {error}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1  # Exit code
    
    @staticmethod
    def handle_file_error(error: Exception, debug: bool = False):
        """Handle file operation errors"""
        print(f"✗ File error: {error}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1  # Exit code
    