"""
Strategy Pattern Implementation for Simulation Modes
Version 3.0: Fixed data structure conflicts - uses unified SimulationResult from cli_simulation
Breaking change: Local SimulationResult class removed, imports enhanced version
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.integrate import solve_ivp

# Version 3.0: Import unified SimulationResult from cli_simulation
from .simulation_types import SimulationResult, SimulationParameters

@dataclass
class SimulationParameters:
    """Unified simulation parameters for all strategies"""
    t_start: float
    t_end: float
    dt: float
    method: str = "RK45"
    rtol: float = 1e-6
    atol: float = 1e-9
    initial_values: List[float] = None


class SimulationStrategy(ABC):
    """Abstract base class for simulation strategies"""
    
    @abstractmethod
    def validate_system(self, system) -> List[str]:
        """Validate that the system is compatible with this strategy"""
        pass
    
    @abstractmethod
    def get_initial_conditions(self, system, params: SimulationParameters) -> np.ndarray:
        """Get initial conditions for the simulation"""
        pass
    
    @abstractmethod
    def run_simulation(self, system, params: SimulationParameters) -> SimulationResult:
        """Run the simulation using this strategy"""
        pass
    
    @abstractmethod
    def get_state_description(self) -> Dict[str, str]:
        """Describe what each element of the state vector represents"""
        pass


class VoltageInputStrategy(SimulationStrategy):
    """
    Strategy for voltage-driven simulations
    Uses regular ODE with voltage as input, current as state
    """
    
    def validate_system(self, system) -> List[str]:
        """Validate system has voltage data for all circuits"""
        errors = []
        
        if hasattr(system, 'circuits'):  # Coupled system
            for circuit in system.circuits:
                if not hasattr(circuit, 'voltage_csv') or circuit.voltage_csv is None:
                    errors.append(f"Circuit {circuit.circuit_id} missing voltage CSV")
        else:  # Single circuit
            if not hasattr(system, 'voltage_csv') or system.voltage_csv is None:
                errors.append("Circuit missing voltage CSV")
                
        return errors
    
    def get_initial_conditions(self, system, params: SimulationParameters) -> np.ndarray:
        """Get initial conditions for voltage-driven simulation"""
        if hasattr(system, 'circuits'):  # Coupled system
            n_circuits = len(system.circuits)
            if params.initial_values and len(params.initial_values) == n_circuits:
                return np.array(params.initial_values)
            else:
                return np.zeros(n_circuits)
        else:  # Single circuit
            if params.initial_values:
                return np.array([params.initial_values[0]])
            else:
                return np.array([0.0])
    
    def run_simulation(self, system, params: SimulationParameters) -> SimulationResult:
        """Run voltage-driven simulation"""
        print("run Voltage-driven simulation ***", flush=True)
        y0 = self.get_initial_conditions(system, params)
        t_span = (params.t_start, params.t_end)
        
        # Use the existing voltage_vector_field method
        def vector_field(t, y):
            if hasattr(system, 'voltage_vector_field'):
                return system.voltage_vector_field(t, y)
            else:
                # Single circuit case
                return system.voltage_vector_field(t, y)
        
        # Use system's voltage vector field
        if hasattr(system, 'circuits'):  # Coupled system
            circuit_ids = [c.circuit_id for c in system.circuits]
        else:  # Single circuit
            circuit_ids = [system.circuit_id]
        
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
        
        # Return enhanced SimulationResult with all required fields
        return SimulationResult(
            time=sol.t,
            solution=sol.y,  # Transpose to match expected format
            metadata={
                'n_evaluations': sol.nfev, 
                'state_size': len(sol.y),
                'success': sol.success,
                'message': sol.message
            },
            strategy_type="voltage_input",
            circuit_ids=circuit_ids,
            success=sol.success,
            error_message=sol.message if not sol.success else None
        )
    
    def get_state_description(self) -> Dict[str, str]:
        return {
            "current": "Circuit currents in Amperes"
        }


class PIDControlStrategy(SimulationStrategy):
    """
    Strategy for PID control simulations
    Uses extended ODE with current and integral error as states
    """
    
    def validate_system(self, system) -> List[str]:
        """Validate system has reference data and PID controllers"""
        errors = []
        
        if hasattr(system, 'circuits'):  # Coupled system
            for circuit in system.circuits:
                if not hasattr(circuit, 'reference_csv') or circuit.reference_csv is None:
                    errors.append(f"Circuit {circuit.circuit_id} missing reference CSV")
                if not hasattr(circuit, 'pid_controller') or circuit.pid_controller is None:
                    errors.append(f"Circuit {circuit.circuit_id} missing PID controller")
        else:  # Single circuit
            if not hasattr(system, 'reference_csv') or system.reference_csv is None:
                errors.append("Circuit missing reference CSV")
            if not hasattr(system, 'pid_controller') or system.pid_controller is None:
                errors.append("Circuit missing PID controller")
                
        return errors
    
    def get_initial_conditions(self, system, params: SimulationParameters) -> np.ndarray:
        """Get initial conditions for PID simulation: [currents, integral_errors]"""
        if hasattr(system, 'circuits'):  # Coupled system
            n_circuits = len(system.circuits)
            currents = np.zeros(n_circuits)
            integral_errors = np.zeros(n_circuits)
            
            if params.initial_values and len(params.initial_values) == n_circuits:
                currents = np.array(params.initial_values)
            
            return np.concatenate([currents, integral_errors])
            
        else:  # Single circuit
            current = params.initial_values[0] if params.initial_values else 0.0
            integral_error = 0.0
            return np.array([current, integral_error])
    
    def run_simulation(self, system, params: SimulationParameters) -> SimulationResult:
        """Run PID control simulation"""
        print(f"run PID control simulation from {params.t_start}, {params.t_end} ***", flush=True)
        y0 = self.get_initial_conditions(system, params)
        t_span = (params.t_start, params.t_end)
        
        def vector_field(t, y):
            if hasattr(system, 'vector_field'):
                # For coupled systems, estimate di_ref_dt
                di_ref_dt = self._estimate_reference_derivative(system, t)
                return system.vector_field(t, y, di_ref_dt=di_ref_dt)
            else:
                # Single circuit case
                return system.vector_field(t, y)
        
        # Use system's PID vector field
        if hasattr(system, 'circuits'):  # Coupled system
            circuit_ids = [c.circuit_id for c in system.circuits]
        else:  # Single circuit
            circuit_ids = [system.circuit_id]
        
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
        print("pid sol: ", sol.y.shape, flush=True)

        # Return enhanced SimulationResult with all required fields
        return SimulationResult(
            time=sol.t,
            solution=sol.y,  # Transpose to match expected format
            metadata={
                'n_evaluations': sol.nfev, 
                'state_size': len(sol.y),
                'success': sol.success,
                'message': sol.message
            },
            strategy_type="pid_control",
            circuit_ids=circuit_ids,
            success=sol.success,
            error_message=sol.message if not sol.success else None
        )
    
    def _estimate_reference_derivative(self, system, t: float, dt: float = 1e-4) -> np.ndarray:
        """Estimate derivative of reference current using finite differences"""
        if hasattr(system, 'get_reference_currents'):
            i_ref_current = np.array(system.get_reference_currents(t))
            i_ref_future = np.array(system.get_reference_currents(t + dt))
            return (i_ref_future - i_ref_current) / dt
        else:
            # Single circuit
            i_ref_current = system.reference_current(t)
            i_ref_future = system.reference_current(t + dt)
            return (i_ref_future - i_ref_current) / dt
    
    def get_state_description(self) -> Dict[str, str]:
        return {
            "current": "Circuit currents in Amperes",
            "integral_error": "PID integral errors in Ampere-seconds"
        }


class HybridStrategy(SimulationStrategy):
    """
    Strategy for mixed voltage/PID simulations
    Some circuits use voltage input, others use PID control
    """
    
    def __init__(self, voltage_circuit_ids: List[str], pid_circuit_ids: List[str]):
        self.voltage_circuit_ids = voltage_circuit_ids
        self.pid_circuit_ids = pid_circuit_ids
    
    def validate_system(self, system) -> List[str]:
        """Validate hybrid system configuration"""
        errors = []
        
        if not hasattr(system, 'circuits'):
            errors.append("Hybrid strategy only supported for coupled systems")
            return errors
        
        # Validate voltage circuits
        for circuit in system.circuits:
            if circuit.circuit_id in self.voltage_circuit_ids:
                if not hasattr(circuit, 'voltage_csv') or circuit.voltage_csv is None:
                    errors.append(f"Voltage circuit {circuit.circuit_id} missing voltage CSV")
        
        # Validate PID circuits
        for circuit in system.circuits:
            if circuit.circuit_id in self.pid_circuit_ids:
                if not hasattr(circuit, 'reference_csv') or circuit.reference_csv is None:
                    errors.append(f"PID circuit {circuit.circuit_id} missing reference CSV")
                if not hasattr(circuit, 'pid_controller') or circuit.pid_controller is None:
                    errors.append(f"PID circuit {circuit.circuit_id} missing PID controller")
        
        return errors
    
    def get_initial_conditions(self, system, params: SimulationParameters) -> np.ndarray:
        """Get initial conditions for hybrid simulation"""
        n_circuits = len(system.circuits)
        n_pid_circuits = len(self.pid_circuit_ids)
        
        # State vector: [currents] + [integral_errors for PID circuits]
        state_size = n_circuits + n_pid_circuits
        y0 = np.zeros(state_size)
        
        # Set initial currents
        if params.initial_values and len(params.initial_values) == n_circuits:
            y0[:n_circuits] = params.initial_values
        
        # Integral errors start at zero (already initialized)
        return y0
    
    def run_simulation(self, system, params: SimulationParameters) -> SimulationResult:
        """Run hybrid simulation"""
        # Hybrid strategy requires more complex implementation
        # For now, raise an error to indicate this needs development
        raise NotImplementedError("Hybrid strategy requires more complex implementation")
    
    def get_state_description(self) -> Dict[str, str]:
        return {
            "current": "All circuit currents in Amperes",
            "pid_integral_error": "PID integral errors for PID-controlled circuits"
        }


class SimulationRunner:
    """
    Main simulation runner that uses strategies
    Replaces the complex logic in coupled_main.py
    Version 3.0: Uses enhanced SimulationResult from cli_simulation
    """
    
    def __init__(self):
        self.strategies = {
            "voltage": VoltageInputStrategy(),
            "pid": PIDControlStrategy(),
            "hybrid": None  # Created on demand
        }
    
    def detect_strategy(self, system) -> str:
        """Automatically detect appropriate strategy based on system configuration"""
        if hasattr(system, 'circuits'):  # Coupled system
            has_voltage = all(hasattr(c, 'voltage_csv') and c.voltage_csv for c in system.circuits)
            has_reference = all(hasattr(c, 'reference_csv') and c.reference_csv for c in system.circuits)
            has_pid = all(hasattr(c, 'pid_controller') and c.pid_controller for c in system.circuits)
            
            if has_voltage and not (has_reference and has_pid):
                return "voltage"
            elif has_reference and has_pid and not has_voltage:
                return "pid"
            elif has_voltage and has_reference and has_pid:
                # Mixed case - need to analyze individual circuits
                voltage_circuits = [c.circuit_id for c in system.circuits if c.voltage_csv]
                pid_circuits = [c.circuit_id for c in system.circuits if c.reference_csv and c.pid_controller]
                self.strategies["hybrid"] = HybridStrategy(voltage_circuits, pid_circuits)
                return "hybrid"
            else:
                raise ValueError("System configuration doesn't match any strategy")
        else:  # Single circuit
            has_voltage = hasattr(system, 'voltage_csv') and system.voltage_csv
            has_reference = hasattr(system, 'reference_csv') and system.reference_csv
            has_pid = hasattr(system, 'pid_controller') and system.pid_controller
            
            if has_voltage and not (has_reference and has_pid):
                return "voltage"
            elif has_reference and has_pid and not has_voltage:
                return "pid"
            elif has_voltage and has_reference and has_pid:
                # Both available - default to PID
                return "pid"
            else:
                raise ValueError("Circuit configuration doesn't match any strategy")
    
    def run_simulation(self, system, params: SimulationParameters, strategy_name: str = None) -> SimulationResult:
        """
        Run simulation using specified or auto-detected strategy
        Version 3.0: Returns enhanced SimulationResult with all required fields
        """
        print("SimulationRunner:run_simulation ***", flush=True)
        if strategy_name is None:
            strategy_name = self.detect_strategy(system)
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        
        # Validate system compatibility
        errors = strategy.validate_system(system)
        if errors:
            print("errors detected in validate strategy ***", flush=True)
            # Return failed SimulationResult instead of raising exception
            circuit_ids = ([c.circuit_id for c in system.circuits] 
                          if hasattr(system, 'circuits') else [system.circuit_id])
            return SimulationResult(
                time=np.array([]),
                solution=np.array([]),
                metadata={},
                strategy_type="failed",
                circuit_ids=circuit_ids,
                success=False,
                error_message=f"System validation failed: {'; '.join(errors)}"
            )
        
        print(f"Running {strategy_name} simulation strategy")
        print(f"State description: {strategy.get_state_description()}")
        
        try:
            print("try to strategy.run_simulation ***", flush=True)
            return strategy.run_simulation(system, params)
        except Exception as e:
            # Return failed SimulationResult for any simulation errors
            # print("try to strategy.run_simulation ***", flush=True)
            circuit_ids = ([c.circuit_id for c in system.circuits] 
                          if hasattr(system, 'circuits') else [system.circuit_id])
            return SimulationResult(
                time=np.array([]),
                solution=np.array([]),
                metadata={},
                strategy_type="failed",
                circuit_ids=circuit_ids,
                success=False,
                error_message=f"Simulation failed: {str(e)}"
            )
    
    def list_available_strategies(self) -> List[str]:
        """List available strategies"""
        return [name for name, strategy in self.strategies.items() if strategy is not None]
    
    def get_strategy_info(self, strategy_name: str) -> dict:
        """Get information about a specific strategy"""
        if strategy_name not in self.strategies or self.strategies[strategy_name] is None:
            raise ValueError(f"Unknown or unavailable strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return {
            "name": strategy_name,
            "description": strategy.__doc__.strip() if strategy.__doc__ else "No description",
            "state_description": strategy.get_state_description()
        }


# Version 3.0 Breaking Changes Notice
#
# IMPORT CHANGES:
# Old: from .cli_simulation import SimulationResult  # Caused circular import
# New: from .simulation_types import SimulationResult  # Clean import
#
# Old: Local SimulationParameters class definition
# New: from .simulation_types import SimulationParameters  # Unified location
#
# The enhanced SimulationResult now includes:
# - circuit_ids: List[str]              # Required for tracking multiple circuits
# - success: bool = True                # Required for error handling  
# - error_message: Optional[str] = None # Required for debugging
#
# All strategy methods now return the enhanced SimulationResult with these additional fields
# This resolves the circular import issue that was preventing Version 3.0 from running.