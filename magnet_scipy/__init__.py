"""
Magnet Scipy - Magnetic coupling simulation for RL circuits with adaptive PID control

This package provides tools for simulating RL circuits with:
- Adaptive PID control based on current magnitude
- Variable resistance dependent on current and temperature
- Magnetic coupling between multiple circuits
- CSV data integration for experimental validation

Main modules:
- rlcircuitpid: Single RL circuit with adaptive PID control
- coupled_circuits: Multiple magnetically coupled RL circuits
- pid_controller: Flexible adaptive PID controller implementation
- csv_utils: Scipy-compatible CSV data loading utilities
- plotting: Visualization tools for simulation results

Example usage:
    >>> from magnet_scipy import RLCircuitPID, create_default_pid_controller
    >>> from magnet_scipy.coupled_circuits import CoupledRLCircuitsPID

    # Single circuit simulation
    >>> circuit = RLCircuitPID(R=1.5, L=0.1, temperature=25.0)
    >>> circuit.print_configuration()

    # Coupled circuits simulation
    >>> circuits = [RLCircuitPID(circuit_id=f"motor_{i}") for i in range(3)]
    >>> coupled = CoupledRLCircuitsPID(circuits, coupling_strength=0.05)
"""

__version__ = "0.1.0"
__author__ = "MagnetDB Team"
__email__ = "team@magnetScipy.com"

# Core classes and functions
from .rlcircuitpid import RLCircuitPID
from .pid_controller import (
    PIDController,
    PIDParams,
    RegionConfig,
    create_default_pid_controller,
    create_adaptive_pid_controller,
    create_custom_pid_controller,
)
from .csv_utils import (
    create_function_from_csv,
    create_2d_function_from_csv,
)


# Define what gets imported with "from magnet_Scipy import *"
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core classes
    "RLCircuitPID",
    "PIDController",
    "PIDParams",
    "RegionConfig",
    # Factory functions
    "create_default_pid_controller",
    "create_adaptive_pid_controller",
    "create_custom_pid_controller",
    "create_example_coupled_circuits",
    # CSV utilities
    "create_scipy_function_from_csv",
    "create_2d_scipy_function_from_csv",
    "create_multi_column_scipy_function_from_csv",
    "create_parametric_scipy_function_from_csv",
]


def get_version() -> str:
    """Get the package version string."""
    return __version__


def check_dependencies() -> dict:
    """
    Check if all required dependencies are available.

    Returns:
        Dictionary with dependency status information
    """
    deps = {
        "Scipy": False,
        "numpy": False,
        "pandas": False,
        "matplotlib": False,
        "seaborn": False,
    }

    for dep in deps:
        try:
            __import__(dep)
            deps[dep] = True
        except ImportError:
            deps[dep] = False

    return {
        "dependencies": deps,
        "all_required": all(deps[d] for d in ["Scipy", "numpy", "pandas"]),
    }


def print_info():
    """Print package information and dependency status."""
    print(f"Magnet Scipy v{__version__}")
    print("Magnetic coupling simulation for RL circuits with adaptive PID control\n")

    status = check_dependencies()

    print("Dependency Status:")
    for dep, available in status["dependencies"].items():
        status_symbol = "✓" if available else "✗"
        print(f"  {status_symbol} {dep}")

    print("\nFeature Availability:")
    print(
        "  ✓ Core simulation"
        if status["all_required"]
        else "  ✗ Core simulation (missing dependencies)"
    )
    print("  ✓ Plotting" if status["plotting_available"] else "  ✗ Plotting")
    print("  ✓ CLI tools" if status["cli_available"] else "  ✗ CLI tools")
