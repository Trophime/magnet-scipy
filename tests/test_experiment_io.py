"""Tests for magnet_scipy.experiment_io (CSV and pigbrother-like TDMS sources)."""

import numpy as np
import pandas as pd
import pytest

from magnet_scipy.experiment_io import is_tdms, read_experiment


def _write_tdms(path, start, values, increment=1.0):
    """Write one 'Courants_Alimentations/Courant GR1' channel, pigbrother style."""
    nptdms = pytest.importorskip("nptdms")
    ch = nptdms.ChannelObject(
        "Courants_Alimentations",
        "Courant GR1",
        np.asarray(values, float),
        properties={
            "wf_start_time": np.datetime64(start, "us"),
            "wf_start_offset": 0.0,
            "wf_increment": increment,
        },
    )
    with nptdms.TdmsWriter(path) as w:
        w.write_segment([nptdms.GroupObject("Courants_Alimentations"), ch])


def _entry(files, **kw):
    return {
        "type": "current",
        "key": "current",
        "file": files,
        "group": "Courants_Alimentations",
        "channel": "Courant GR1",
        **kw,
    }


def test_is_tdms():
    assert is_tdms("a.tdms")
    assert is_tdms(["a.tdms", "b.tdms"])
    assert not is_tdms("a.csv")


def test_csv_key_column_any_order_any_separator(tmp_path):
    pd.DataFrame({"current": [3.0, 1.0], "time": [1.0, 0.0]}).to_csv(
        tmp_path / "e.csv", sep="\t", index=False
    )
    t, y = read_experiment({"key": "current", "file": "e.csv"}, tmp_path)
    np.testing.assert_array_equal(t, [0.0, 1.0])  # sorted by time
    np.testing.assert_array_equal(y, [1.0, 3.0])


def test_tdms_list_is_ordered_by_start_time(tmp_path):
    # second archive starts 10 s after the first; listed out of order on purpose
    _write_tdms(tmp_path / "b.tdms", "2023-09-16T15:55:42", [20.0, 21.0], 0.5)
    _write_tdms(tmp_path / "a.tdms", "2023-09-16T15:55:32", [10.0, 11.0, 12.0], 0.5)
    t, y = read_experiment(_entry(["b.tdms", "a.tdms"], time_shift=0.25), tmp_path)
    np.testing.assert_allclose(t, [0.25, 0.75, 1.25, 10.25, 10.75])
    np.testing.assert_array_equal(y, [10, 11, 12, 20, 21])


def test_tdms_single_path(tmp_path):
    _write_tdms(tmp_path / "a.tdms", "2023-09-16T15:55:32", [1.0, 2.0])
    t, y = read_experiment(_entry("a.tdms"), tmp_path)
    np.testing.assert_array_equal(t, [0.0, 1.0])


def test_tdms_errors(tmp_path):
    _write_tdms(tmp_path / "a.tdms", "2023-09-16T15:55:32", [1.0])
    with pytest.raises(KeyError):
        read_experiment(_entry("a.tdms", channel="Courant GR9"), tmp_path)
    with pytest.raises(FileNotFoundError):
        read_experiment(_entry(["a.tdms", "missing.tdms"]), tmp_path)


def test_solver_loads_tdms_experiment(tmp_path):
    from magnet_scipy import RLCircuitPID

    _write_tdms(tmp_path / "a.tdms", "2023-09-16T15:55:32", [0.0, 10.0, 20.0])
    circuit = RLCircuitPID(
        experimental_data=[_entry([str(tmp_path / "a.tdms")])], circuit_id="A"
    )
    circuit._load_experimental_data()
    exp = circuit.experimental_functions["current_current"]
    assert float(exp["function"](1.5)) == pytest.approx(15.0)
