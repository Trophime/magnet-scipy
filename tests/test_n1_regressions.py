"""Regression tests for the scipy bugs confirmed by the theory review (ROADMAP N1).

See ``prompts/scipy_theory_review.md``, "Code bugs (confirmed)".
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from magnet_scipy import cli_simulation
from magnet_scipy.cli_core import TimeParameters
from magnet_scipy.coupled_circuits import CoupledRLCircuitsPID
from magnet_scipy.pid_controller import create_default_pid_controller
from magnet_scipy.plotting_strategies import PIDControlPlottingStrategy
from magnet_scipy.rlcircuitpid import RLCircuitPID
from magnet_scipy.simulation_strategies import SimulationRunner
from magnet_scipy.simulation_types import SimulationParameters, SimulationResult


@pytest.fixture
def review_reference_csv(tmp_path):
    """Reference of the theory review: ramps through both gain switches (60 A, 800 A)."""
    path = tmp_path / "reference.csv"
    pd.DataFrame(
        {"time": [0.0, 1.0, 2.0, 3.0, 5.0], "current": [0.0, 100.0, 100.0, 1000.0, 1000.0]}
    ).to_csv(path, index=False)
    return str(path)


def _review_circuit(reference_csv, circuit_id="c1"):
    return RLCircuitPID(
        R=0.5,
        L=0.2,
        pid_controller=create_default_pid_controller(),
        reference_csv=reference_csv,
        circuit_id=circuit_id,
    )


# Bug 1: failed coupled run reported as success
@pytest.mark.unit
def test_failed_coupled_run_is_reported_as_failure():
    coupled = Mock(circuit_ids=["c1", "c2"], n_circuits=2)
    orchestrator = cli_simulation.SimulationOrchestrator()
    orchestrator.coupled_runner = Mock()
    orchestrator.coupled_runner.run_simulation.return_value = SimulationResult(
        time=np.array([]), solution=np.array([]), metadata={},
        strategy_type="failed", circuit_ids=["c1", "c2"],
        success=False, error_message="Simulation failed: boom",
    )

    result = orchestrator.run_coupled_simulation(
        coupled, TimeParameters(start=0.0, end=1.0, step=0.1), [0.0, 0.0], "pid"
    )

    assert not result.success
    assert result.error_message == "Simulation failed: boom"


# Hybrid picked when every circuit has both voltage and reference CSV
@pytest.mark.unit
def test_coupled_with_both_csvs_on_every_circuit_defaults_to_pid():
    circuits = [
        Mock(circuit_id=cid, voltage_csv="u.csv", reference_csv="i.csv", pid_controller=object())
        for cid in ("c1", "c2")
    ]

    assert SimulationRunner().detect_strategy(Mock(circuits=circuits)) == "pid"


# Signed vs abs(i) current in single vs coupled R
@pytest.mark.unit
def test_single_circuit_resistance_uses_current_magnitude(sample_resistance_csv):
    circuit = RLCircuitPID(
        L=0.1, pid_controller=create_default_pid_controller(),
        resistance_csv=sample_resistance_csv, temperature=30.0, circuit_id="c1",
    )

    assert circuit.get_resistance(-100.0, 30.0) == pytest.approx(
        circuit.get_resistance(100.0, 30.0)
    )


# PID voltage rebuilt with np.gradient
@pytest.mark.unit
def test_single_circuit_pid_voltage_matches_circuit_equation(review_reference_csv):
    circuit = _review_circuit(review_reference_csv)

    def rhs(t, y):
        return circuit.vector_field(t, y, di_ref_dt=circuit.reference_current_derivative(t))

    sol = solve_ivp(rhs, (0.0, 5.0), [0.0, 0.0], rtol=1e-6, atol=1e-9, max_step=0.01)
    data = PIDControlPlottingStrategy().prepare_data(sol, circuit).circuits["c1"]

    di_dt = np.array([rhs(t, y)[0] for t, y in zip(sol.t, sol.y.T)])
    expected = 0.5 * sol.y[0] + 0.2 * di_dt
    np.testing.assert_allclose(data["voltage"], expected, rtol=1e-9, atol=1e-9)


@pytest.mark.unit
def test_coupled_pid_voltage_matches_circuit_equation(review_reference_csv):
    circuits = [_review_circuit(review_reference_csv, cid) for cid in ("c1", "c2")]
    system = CoupledRLCircuitsPID(circuits, mutual_inductances=np.array([0.1]))

    result = SimulationRunner().run_simulation(
        system, SimulationParameters(t_start=0.0, t_end=5.0, dt=0.01), "pid"
    )
    assert result.success
    sol = SimpleNamespace(t=result.time, y=result.solution)
    processed = PIDControlPlottingStrategy().prepare_data(sol, system)

    di_dt = np.array([
        system.vector_field(
            t, y, di_ref_dt=np.array(system.get_reference_current_derivatives(t))
        )[:2]
        for t, y in zip(sol.t, sol.y.T)
    ]).T
    expected = 0.5 * sol.y[:2] + system.M @ di_dt
    for k, cid in enumerate(("c1", "c2")):
        np.testing.assert_allclose(
            processed.circuits[cid]["voltage"], expected[k], rtol=1e-9, atol=1e-9
        )
