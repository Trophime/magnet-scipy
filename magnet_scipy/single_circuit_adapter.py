"""
Single Circuit Strategy Adapter
Version 3.0: Fixed import conflicts - uses unified SimulationResult from cli_simulation
Breaking change: Imports enhanced SimulationResult instead of conflicting local version
"""

import numpy as np
from typing import List
from .simulation_strategies import (
    VoltageInputStrategy, 
    PIDControlStrategy,
    SimulationParameters
)
# Version 3.0: Import enhanced SimulationResult from cli_simulation to avoid conflicts
from .simulation_types import SimulationResult, SimulationParameters

class SingleCircuitVoltageStrategy(VoltageInputStrategy):
    """
    Voltage strategy adapted for single circuit
    Version 3.0: Returns enhanced SimulationResult with all required fields
    """
    
    def validate_system(self, circuit) -> List[str]:
        """Validate single circuit has voltage data"""
        errors = []
        
        if not hasattr(circuit, 'voltage_csv') or circuit.voltage_csv is None:
            errors.append("Circuit missing voltage CSV")
        if not hasattr(circuit, 'voltage_func') or circuit.voltage_func is None:
            errors.append("Circuit voltage function not loaded")
                
        return errors
    
    def get_initial_conditions(self, circuit, params: SimulationParameters) -> np.ndarray:
        """Initial conditions: just current for single circuit"""
        if params.initial_values:
            return np.array([params.initial_values[0]])
        else:
            return np.array([0.0])
    
    def run_simulation(self, circuit, params: SimulationParameters) -> SimulationResult:
        """Run voltage-driven simulation for single circuit"""
        from scipy.integrate import solve_ivp
        
        y0 = self.get_initial_conditions(circuit, params)
        t_span = (params.t_start, params.t_end)
        
        def vector_field(t, y):
            return circuit.voltage_vector_field(t, y)
        
        sol = solve_ivp(
            vector_field,
            t_span,
            y0,
            method=params.method,
            dense_output=True,
            rtol=params.rtol,
            atol=params.atol,
            max_step=params.dt,
        )
        
        # Version 3.0: Return enhanced SimulationResult with all required fields
        return SimulationResult(
            time=sol.t,
            solution=sol.y,  # Transpose to match expected format
            metadata={
                "success": sol.success,
                "message": sol.message,
                "n_evaluations": sol.nfev,
                "state_size": len(y0),
                "circuit_id": circuit.circuit_id
            },
            strategy_type="voltage_input",
            circuit_ids=[circuit.circuit_id],  # Version 3.0: Required field
            success=sol.success,                # Version 3.0: Required field
            error_message=sol.message if not sol.success else None  # Version 3.0: Required field
        )


class SingleCircuitPIDStrategy(PIDControlStrategy):
    """
    PID strategy adapted for single circuit
    Version 3.0: Returns enhanced SimulationResult with all required fields
    """
    
    def validate_system(self, circuit) -> List[str]:
        """Validate single circuit has reference data and PID controller"""
        errors = []
        
        if not hasattr(circuit, 'reference_csv') or circuit.reference_csv is None:
            errors.append("Circuit missing reference CSV")
        if not hasattr(circuit, 'reference_func') or circuit.reference_func is None:
            errors.append("Circuit reference function not loaded")
        if not hasattr(circuit, 'pid_controller') or circuit.pid_controller is None:
            errors.append("Circuit missing PID controller")
                
        return errors
    
    def get_initial_conditions(self, circuit, params: SimulationParameters) -> np.ndarray:
        """Initial conditions: [current, integral_error] for single circuit"""
        if params.initial_values:
            current = params.initial_values[0]
        else:
            current = 0.0
        integral_error = 0.0
        return np.array([current, integral_error])
    
    def run_simulation(self, circuit, params: SimulationParameters) -> SimulationResult:
        """Run PID control simulation for single circuit"""
        from scipy.integrate import solve_ivp
        
        y0 = self.get_initial_conditions(circuit, params)
        t_span = (params.t_start, params.t_end)
        
        def vector_field(t, y):
            # Estimate reference derivative for single circuit
            di_ref_dt = self._estimate_single_circuit_reference_derivative(circuit, t)
            return circuit.vector_field(t, y, di_ref_dt=di_ref_dt)
        
        sol = solve_ivp(
            vector_field,
            t_span,
            y0,
            method=params.method,
            dense_output=True,
            rtol=params.rtol,
            atol=params.atol,
            max_step=params.dt,
        )
        
        # Version 3.0: Return enhanced SimulationResult with all required fields
        return SimulationResult(
            time=sol.t,
            solution=sol.y,  # Transpose to match expected format
            metadata={
                "success": sol.success,
                "message": sol.message,
                "n_evaluations": sol.nfev,
                "state_size": len(y0),
                "circuit_id": circuit.circuit_id
            },
            strategy_type="pid_control",
            circuit_ids=[circuit.circuit_id],  # Version 3.0: Required field
            success=sol.success,                # Version 3.0: Required field
            error_message=sol.message if not sol.success else None  # Version 3.0: Required field
        )
    
    def _estimate_single_circuit_reference_derivative(self, circuit, t: float, dt: float = 1e-4) -> float:
        """Estimate derivative of reference current for single circuit"""
        i_ref_current = circuit.reference_current(t)
        i_ref_future = circuit.reference_current(t + dt)
        return (i_ref_future - i_ref_current) / dt


class SingleCircuitSimulationRunner:
    """
    Simulation runner specifically for single circuits
    Version 3.0: Uses enhanced SimulationResult with all required fields
    """
    
    def __init__(self):
        self.strategies = {
            "voltage": SingleCircuitVoltageStrategy(),
            "pid": SingleCircuitPIDStrategy(),
        }
    
    def detect_strategy(self, circuit) -> str:
        """Detect appropriate strategy for single circuit"""
        has_voltage = hasattr(circuit, 'voltage_csv') and circuit.voltage_csv
        has_reference = hasattr(circuit, 'reference_csv') and circuit.reference_csv
        has_pid = hasattr(circuit, 'pid_controller') and circuit.pid_controller
        
        if has_voltage and not (has_reference and has_pid):
            return "voltage"
        elif has_reference and has_pid and not has_voltage:
            return "pid"
        else:
            raise ValueError("Circuit configuration doesn't match any strategy")
    
    def run_simulation(self, circuit, params: SimulationParameters, strategy_name: str = None) -> SimulationResult:
        """
        Run simulation using specified or auto-detected strategy for single circuit
        Version 3.0: Returns enhanced SimulationResult with proper error handling
        """
        print("singlecircuitsimulationrunner.run_simulation ***", flush=True)
        if strategy_name is None:
            strategy_name = self.detect_strategy(circuit)
        
        if strategy_name not in self.strategies:
            # Version 3.0: Return failed SimulationResult instead of raising exception
            return SimulationResult(
                time=np.array([]),
                solution=np.array([]),
                metadata={},
                strategy_type="failed",
                circuit_ids=[circuit.circuit_id],
                success=False,
                error_message=f"Unknown strategy: {strategy_name}"
            )
        
        strategy = self.strategies[strategy_name]
        
        # Validate circuit compatibility
        errors = strategy.validate_system(circuit)
        if errors:
            # Version 3.0: Return failed SimulationResult instead of raising exception
            return SimulationResult(
                time=np.array([]),
                solution=np.array([]),
                metadata={},
                strategy_type="failed",
                circuit_ids=[circuit.circuit_id],
                success=False,
                error_message=f"Circuit validation failed: {'; '.join(errors)}"
            )
        
        print(f"Running {strategy_name} simulation strategy for single circuit")
        print(f"State description: {strategy.get_state_description()}")
        
        try:
            print("strategy.run_simulation ***", type(strategy))
            return strategy.run_simulation(circuit, params)
        except Exception as e:
            # Version 3.0: Return failed SimulationResult for any simulation errors
            print("something went wrong in run_simulation ***", flush=True)
            return SimulationResult(
                time=np.array([]),
                solution=np.array([]),
                metadata={},
                strategy_type="failed",
                circuit_ids=[circuit.circuit_id],
                success=False,
                error_message=f"Simulation failed: {str(e)}"
            )
    
    def list_available_strategies(self) -> List[str]:
        """List available strategies for single circuits"""
        return list(self.strategies.keys())
    
    def get_strategy_info(self, strategy_name: str) -> dict:
        """Get information about a specific strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return {
            "name": strategy_name,
            "description": strategy.__doc__.strip() if strategy.__doc__ else "No description",
            "state_description": strategy.get_state_description()
        }


# Update the main simulation runner to use single circuit strategies when appropriate
def create_simulation_runner(system):
    """
    Factory function to create appropriate simulation runner
    Version 3.0: Uses enhanced SimulationResult consistently
    """
    if hasattr(system, 'circuits'):  # Coupled system
        from .simulation_strategies import SimulationRunner
        return SimulationRunner()
    else:  # Single circuit
        return SingleCircuitSimulationRunner()


# Integration with the updated main.py
def run_single_circuit_with_strategy(circuit, params: SimulationParameters, strategy_name: str = None):
    """
    Convenience function for running single circuit simulations
    Version 3.0: Returns enhanced SimulationResult
    """
    runner = SingleCircuitSimulationRunner()
    return runner.run_simulation(circuit, params, strategy_name)


# Version 3.0 Breaking Changes Notice
#
# FIXED IMPORT CONFLICTS:
# Old: from .simulation_strategies import SimulationResult  # Conflicting local version
# Old: from .cli_simulation import SimulationResult         # Caused circular import
# New: from .simulation_types import SimulationResult       # Clean, unified version
#
# Old: from .simulation_strategies import SimulationParameters # Moved to avoid conflicts
# New: from .simulation_types import SimulationParameters      # Unified location
#
# ENHANCED RETURN VALUES:
# All strategy methods now return enhanced SimulationResult with:
# - circuit_ids: List[str]              # Required for tracking circuits
# - success: bool = True                # Required for error handling  
# - error_message: Optional[str] = None # Required for debugging
#
# ERROR HANDLING IMPROVEMENT:
# - Validation errors return failed SimulationResult instead of raising exceptions
# - Simulation errors return failed SimulationResult instead of raising exceptions
# - Consistent error reporting across all strategies
#
# This resolves the circular import issue that was preventing Version 3.0 from running.
