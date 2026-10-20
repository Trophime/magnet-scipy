"""Tests for magnet_scipy.cli_simulation."""

from unittest.mock import Mock

import pytest

from magnet_scipy import cli_simulation
from magnet_scipy.cli_core import TimeParameters
from magnet_scipy.simulation_types import SimulationResult


@pytest.mark.unit
def test_simulation_result_is_canonical_class():
    assert cli_simulation.SimulationResult is SimulationResult


@pytest.mark.unit
def test_failed_single_circuit_simulation_returns_canonical_result():
    circuit = Mock(circuit_id="c1")
    circuit.validate.side_effect = ValueError("bad circuit")

    result = cli_simulation.SimulationOrchestrator().run_single_circuit_simulation(
        circuit, TimeParameters(start=0.0, end=1.0, step=0.1)
    )

    assert isinstance(result, SimulationResult)
    assert not result.success
    assert result.error_message == "bad circuit"
    assert result.circuit_ids == ["c1"]


@pytest.mark.unit
def test_failed_coupled_simulation_returns_canonical_result():
    coupled = Mock(circuit_ids=["c1", "c2"])
    coupled.validate.side_effect = ValueError("bad system")

    result = cli_simulation.SimulationOrchestrator().run_coupled_simulation(
        coupled, TimeParameters(start=0.0, end=1.0, step=0.1), [0.0, 0.0]
    )

    assert isinstance(result, SimulationResult)
    assert not result.success
    assert result.error_message == "bad system"
    assert result.circuit_ids == ["c1", "c2"]
