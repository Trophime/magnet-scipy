import numpy as np

# Import here to avoid circular imports
from magnet_scipy.pid_controller import create_adaptive_pid_controller
from magnet_scipy.rlcircuitpid import RLCircuitPID
from magnet_scipy.coupled_circuits import CoupledRLCircuitsPID

from typing import List, Tuple, Dict

def create_example_coupled_circuits(
    n_circuits: int = 3, coupling_strength: float = 0.05
):
    """
    Create an example system of coupled RL circuits for testing

    Args:
        n_circuits: Number of circuits to create
        coupling_strength: Mutual inductance between circuits

    Returns:
        Configured CoupledRLCircuitsPID instance
    """
    
    circuits = []

    for i in range(n_circuits):
        # Create different PID controllers for each circuit
        if i % 2 == 0:
            # More aggressive PID for even circuits
            pid_controller = create_adaptive_pid_controller(
                Kp_low=15.0,
                Ki_low=10.0,
                Kd_low=0.08,
                Kp_medium=20.0,
                Ki_medium=12.0,
                Kd_medium=0.06,
                Kp_high=25.0,
                Ki_high=15.0,
                Kd_high=0.04,
            )
        else:
            # More conservative PID for odd circuits
            pid_controller = create_adaptive_pid_controller(
                Kp_low=10.0,
                Ki_low=6.0,
                Kd_low=0.12,
                Kp_medium=12.0,
                Ki_medium=8.0,
                Kd_medium=0.08,
                Kp_high=15.0,
                Ki_high=10.0,
                Kd_high=0.06,
            )

        # Create RLCircuitPID instance
        circuit = RLCircuitPID(
            R=1.0 + 0.2 * i,  # Different resistances
            L=0.1 + 0.02 * i,  # Different inductances
            pid_controller=pid_controller,
            temperature=25.0 + 5.0 * i,  # Different temperatures
            circuit_id=f"circuit_{i+1}",
        )

        circuits.append(circuit)

    # Create symmetric coupling matrix
    M = np.full((n_circuits, n_circuits), coupling_strength)
    np.fill_diagonal(M, 0.0)

    return CoupledRLCircuitsPID(circuits, M)


def create_custom_coupled_system(
    circuit_params: List[Dict],
    mutual_inductances: np.ndarray = None,
    coupling_strength: float = 0.05,
):
    """
    Create a custom coupled system from parameter dictionaries

    Args:
        circuit_params: List of dictionaries with circuit parameters
        mutual_inductances: Optional coupling matrix
        coupling_strength: Default coupling if no matrix provided

    Returns:
        CoupledRLCircuitsPID instance

    Example:
        params = [
            {'R': 1.0, 'L': 0.1, 'circuit_id': 'motor1', 'temperature': 25},
            {'R': 1.2, 'L': 0.12, 'circuit_id': 'motor2', 'temperature': 30}
        ]
        system = create_custom_coupled_system(params)
    """
    from magnet_scipy.rlcircuitpid import RLCircuitPID

    circuits = []

    for i, params in enumerate(circuit_params):
        # Set defaults
        circuit_params_with_defaults = {
            "R": 1.0,
            "L": 0.1,
            "temperature": 25.0,
            "circuit_id": f"circuit_{i+1}",
            **params,  # Override with user parameters
        }

        circuit = RLCircuitPID(**circuit_params_with_defaults)
        circuits.append(circuit)

    return CoupledRLCircuitsPID(circuits, mutual_inductances, coupling_strength)
