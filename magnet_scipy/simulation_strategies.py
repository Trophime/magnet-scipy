"""
Strategy Pattern Implementation for Simulation Modes
Separates voltage-driven and PID control simulations into distinct strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.integrate import solve_ivp


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


@dataclass
class SimulationResult:
    """Unified result structure for all simulation strategies"""
    time: np.ndarray
    solution: np.ndarray
    metadata: Dict[str, Any]
    strategy_type: str


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
        """Initial conditions: just currents for each circuit"""
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
        y0 = self.get_initial_conditions(system, params)
        t_span = (params.t_start, params.t_end)
        
        # Use the existing voltage_vector_field method
        def vector_field(t, y):
            if hasattr(system, 'voltage_vector_field'):
                return system.voltage_vector_field(t, y)
            else:
                # Single circuit case
                return system.voltage_vector_field(t, y)
        
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
        
        return SimulationResult(
            time=sol.t,
            solution=sol.y,
            metadata={
                "success": sol.success,
                "message": sol.message,
                "n_evaluations": sol.nfev,
                "state_size": len(y0)
            },
            strategy_type="voltage_input"
        )
    
    def get_state_description(self) -> Dict[str, str]:
        return {"current": "Circuit current(s) in Amperes"}


class PIDControlStrategy(SimulationStrategy):
    """
    Strategy for PID control simulations
    Uses reference current tracking with adaptive PID
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
        """Initial conditions: [currents, integral_errors] for each circuit"""
        if hasattr(system, 'circuits'):  # Coupled system
            n_circuits = len(system.circuits)
            if params.initial_values and len(params.initial_values) == n_circuits:
                currents = np.array(params.initial_values)
            else:
                currents = np.zeros(n_circuits)
            integral_errors = np.zeros(n_circuits)
            return np.concatenate([currents, integral_errors])
        else:  # Single circuit
            if params.initial_values:
                current = params.initial_values[0]
            else:
                current = 0.0
            return np.array([current, 0.0])  # [current, integral_error]
    
    def run_simulation(self, system, params: SimulationParameters) -> SimulationResult:
        """Run PID control simulation with improved time-stepping"""
        return self._run_continuous_pid_simulation(system, params)
    
    def _run_continuous_pid_simulation(self, system, params: SimulationParameters) -> SimulationResult:
        """
        Improved continuous PID simulation using solve_ivp with time-varying parameters
        Avoids complex manual time-stepping
        """
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
        
        return SimulationResult(
            time=sol.t,
            solution=sol.y,
            metadata={
                "success": sol.success,
                "message": sol.message,
                "n_evaluations": sol.nfev,
                "state_size": len(y0)
            },
            strategy_type="pid_control"
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
            "current": "Circuit current(s) in Amperes",
            "integral_error": "PID integral error(s) in Ampere-seconds"
        }


class HybridStrategy(SimulationStrategy):
    """
    Strategy for mixed voltage/PID simulations
    Some circuits voltage-driven, others PID-controlled
    """
    
    def __init__(self, voltage_circuits: List[str], pid_circuits: List[str]):
        self.voltage_circuits = voltage_circuits
        self.pid_circuits = pid_circuits
    
    def validate_system(self, system) -> List[str]:
        """Validate mixed system configuration"""
        errors = []
        
        if not hasattr(system, 'circuits'):
            errors.append("Hybrid strategy requires coupled system")
            return errors
        
        circuit_ids = [c.circuit_id for c in system.circuits]
        
        # Check voltage circuits
        for circuit_id in self.voltage_circuits:
            if circuit_id not in circuit_ids:
                errors.append(f"Voltage circuit {circuit_id} not found in system")
            else:
                circuit = next(c for c in system.circuits if c.circuit_id == circuit_id)
                if not hasattr(circuit, 'voltage_csv') or circuit.voltage_csv is None:
                    errors.append(f"Voltage circuit {circuit_id} missing voltage CSV")
        
        # Check PID circuits
        for circuit_id in self.pid_circuits:
            if circuit_id not in circuit_ids:
                errors.append(f"PID circuit {circuit_id} not found in system")
            else:
                circuit = next(c for c in system.circuits if c.circuit_id == circuit_id)
                if not hasattr(circuit, 'reference_csv') or circuit.reference_csv is None:
                    errors.append(f"PID circuit {circuit_id} missing reference CSV")
                if not hasattr(circuit, 'pid_controller') or circuit.pid_controller is None:
                    errors.append(f"PID circuit {circuit_id} missing PID controller")
        
        return errors
    
    def get_initial_conditions(self, system, params: SimulationParameters) -> np.ndarray:
        """Initial conditions: [all_currents, pid_integral_errors]"""
        n_circuits = len(system.circuits)
        n_pid = len(self.pid_circuits)
        
        if params.initial_values and len(params.initial_values) == n_circuits:
            currents = np.array(params.initial_values)
        else:
            currents = np.zeros(n_circuits)
        
        pid_integral_errors = np.zeros(n_pid)
        return np.concatenate([currents, pid_integral_errors])
    
    def run_simulation(self, system, params: SimulationParameters) -> SimulationResult:
        """Run hybrid simulation - more complex implementation needed"""
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
            elif has_reference and has_pid:
                return "pid"
            else:
                raise ValueError("Single circuit configuration doesn't match any strategy")
    
    def run_simulation(self, system, params: SimulationParameters, strategy_name: str = None) -> SimulationResult:
        """
        Run simulation using specified or auto-detected strategy
        """
        if strategy_name is None:
            strategy_name = self.detect_strategy(system)
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        if strategy is None:
            raise ValueError(f"Strategy {strategy_name} not initialized")
        
        # Validate system compatibility
        errors = strategy.validate_system(system)
        if errors:
            raise ValueError(f"System validation failed: {'; '.join(errors)}")
        
        print(f"Running {strategy_name} simulation strategy")
        print(f"State description: {strategy.get_state_description()}")
        
        return strategy.run_simulation(system, params)


# Example usage and integration
def create_simulation_parameters_from_args(args) -> SimulationParameters:
    """Convert command line arguments to SimulationParameters"""
    return SimulationParameters(
        t_start=args.time_start,
        t_end=args.time_end,
        dt=args.time_step,
        method=args.method,
        initial_values=args.value_start if hasattr(args, 'value_start') else None
    )


def run_simulation_with_strategy(system, args):
    """
    Replacement for the complex simulation logic in coupled_main.py
    """
    # Create parameters
    params = create_simulation_parameters_from_args(args)
    
    # Create runner and run simulation
    runner = SimulationRunner()
    
    try:
        result = runner.run_simulation(system, params)
        print(f"✓ Simulation completed using {result.strategy_type} strategy")
        print(f"  Time points: {len(result.time)}")
        print(f"  State vector size: {result.metadata['state_size']}")
        
        return result
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        raise
