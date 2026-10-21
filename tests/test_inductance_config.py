"""Tests for the inductance matrix configuration of coupled circuits."""

import json

import numpy as np
import pytest

from magnet_scipy.cli_core import ConfigurationLoader, SampleConfigGenerator
from magnet_scipy.coupled_circuits import CoupledRLCircuitsPID
from magnet_scipy.rlcircuitpid import RLCircuitPID


def _circuits(inductances):
    return [
        RLCircuitPID(L=L, circuit_id=f"c{n}") for n, L in enumerate(inductances)
    ]


@pytest.fixture
def no_reference_csv(monkeypatch):
    """Skip loading reference CSV files referenced by sample configs."""
    monkeypatch.setattr(RLCircuitPID, "load_reference_from_csv", lambda self, f: None)


@pytest.mark.unit
def test_packed_vector_fills_upper_triangle_in_row_major_order():
    coupled = CoupledRLCircuitsPID(
        _circuits([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])
    )

    expected = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])
    np.testing.assert_allclose(coupled.M, expected)


@pytest.mark.unit
def test_full_matrix_is_accepted():
    M = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])

    coupled = CoupledRLCircuitsPID(_circuits([1.0, 2.0, 3.0]), M)

    np.testing.assert_allclose(coupled.M, M)


@pytest.mark.unit
def test_full_matrix_with_mismatched_diagonal_is_rejected():
    M = np.array([[1.0, 0.1], [0.1, 5.0]])

    with pytest.raises(ValueError, match="diagonal"):
        CoupledRLCircuitsPID(_circuits([1.0, 2.0]), M)


@pytest.mark.unit
def test_loader_warns_on_unknown_circuit_keys(tmp_path):
    config_file = tmp_path / "coupled.json"
    config_file.write_text(
        json.dumps(
            {
                "circuits": [
                    {"circuit_id": "a", "L": 0.08, "inductance": 0.08},
                    {"circuit_id": "b", "inductance": 0.09},
                ],
                "mutual_inductances": [0.02],
            }
        )
    )

    with pytest.warns(UserWarning, match="'L'"):
        ConfigurationLoader.load_coupled_circuits(str(config_file))


@pytest.mark.unit
def test_sample_coupled_config_round_trip(tmp_path, recwarn, no_reference_csv):
    config_file = tmp_path / "coupled.json"
    SampleConfigGenerator.create_coupled_circuits_config(str(config_file))

    circuits, mutual_inductances = ConfigurationLoader.load_coupled_circuits(
        str(config_file)
    )
    coupled = CoupledRLCircuitsPID(circuits, mutual_inductances)

    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    np.testing.assert_allclose(np.diag(coupled.M), [0.1, 0.15])
    assert all(c.pid_controller is not None for c in circuits)


@pytest.mark.unit
def test_sample_single_config_round_trip(tmp_path, recwarn, no_reference_csv):
    config_file = tmp_path / "single.json"
    SampleConfigGenerator.create_single_circuit_config(str(config_file))

    circuit = ConfigurationLoader.load_single_circuit(str(config_file))

    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    assert circuit.circuit_id == "sample_circuit"
    assert circuit.L == 0.1
