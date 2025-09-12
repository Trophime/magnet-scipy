"""
magnet_scipy/simulation_types.py

Version 3.0: Shared data structures and types to avoid circular imports
Breaking change: SimulationResult moved here from cli_simulation to resolve circular dependencies
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """
    Enhanced simulation result with metadata - Version 3.0 unified structure
    
    This class is now the single source of truth for simulation results across
    all modules, eliminating the circular import issue.
    """
    time: np.ndarray
    solution: np.ndarray
    metadata: Dict[str, Any]
    strategy_type: str
    circuit_ids: List[str]
    success: bool = True
    error_message: Optional[str] = None


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


# Version 3.0 Breaking Changes Notice
#
# MOVED FROM cli_simulation.py:
# - SimulationResult class moved here to break circular imports
#
# MOVED FROM simulation_strategies.py:
# - SimulationParameters class moved here for consistency
#
# IMPORT CHANGES:
# Old: from .cli_simulation import SimulationResult
# New: from .simulation_types import SimulationResult
#
# Old: from .simulation_strategies import SimulationParameters
# New: from .simulation_types import SimulationParameters
#
# This resolves the circular import:
# cli_simulation.py → simulation_strategies.py → cli_simulation.py (✗)
#
# New clean import hierarchy:
# cli_simulation.py → simulation_types.py (✓)
# simulation_strategies.py → simulation_types.py (✓)
# single_circuit_adapter.py → simulation_types.py (✓)
