"""Tests for magnet_scipy.cli_simulation."""

from unittest.mock import Mock

import numpy as np
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


@pytest.mark.unit
def test_single_circuit_simulation_uses_initial_value():
    circuit = Mock(circuit_id="c1")
    orchestrator = cli_simulation.SimulationOrchestrator()
    orchestrator.single_runner = Mock()
    orchestrator.single_runner.run_simulation.return_value = SimulationResult(
        time=np.array([0.0]), solution=np.array([[200.0]]), metadata={},
        strategy_type="voltage_input", circuit_ids=["c1"],
    )

    orchestrator.run_single_circuit_simulation(
        circuit, TimeParameters(start=0.0, end=1.0, step=0.1), "voltage",
        initial_value=200.0,
    )

    sim_params = orchestrator.single_runner.run_simulation.call_args.args[1]
    assert sim_params.initial_values == [200.0]


@pytest.mark.unit
def test_single_circuit_summary_prints_last_time_point(capsys):
    # solution layout is (n_states, n_time), as returned by solve_ivp
    result = SimulationResult(
        time=np.array([0.0, 1.0, 2.0]),
        solution=np.array([[0.0, 100.0, 189.5]]),
        metadata={}, strategy_type="voltage_input", circuit_ids=["c1"],
    )

    cli_simulation.SimulationSummary.print_single_circuit_summary(
        result, Mock(circuit_id="c1"), TimeParameters(start=0.0, end=2.0, step=0.1)
    )

    assert "Final current: 189.500 A" in capsys.readouterr().out


@pytest.mark.unit
def test_coupled_summary_prints_last_time_point(capsys):
    # PID layout: [currents..., integral_errors...] x n_time
    result = SimulationResult(
        time=np.array([0.0, 1.0, 2.0]),
        solution=np.array([
            [200.0, 210.0, 220.0],
            [100.0, 110.0, 120.0],
            [0.0, 1.0, 2.0],
            [0.0, 3.0, 4.0],
        ]),
        metadata={}, strategy_type="pid_control", circuit_ids=["c1", "c2"],
    )

    cli_simulation.SimulationSummary.print_coupled_circuits_summary(
        result, Mock(n_circuits=2), TimeParameters(start=0.0, end=2.0, step=0.1)
    )

    out = capsys.readouterr().out
    assert "Final current [c1]: 220.000 A" in out
    assert "Final current [c2]: 120.000 A" in out
