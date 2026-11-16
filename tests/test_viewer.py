"""Tests for magnet_scipy.viewer using small synthetic results."""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from magnet_scipy import viewer


@pytest.fixture
def run_files(tmp_path):
    t = np.linspace(0, 100, 1001)
    np.savez(
        tmp_path / "run.npz",
        time=t,
        A_current=2 * t,
        A_reference=2 * t,
        A_voltage=np.ones_like(t),
        A_temperature=300 + 0 * t,
        A_voltage_voltage_rms=np.float64(0.1),  # scalar metric, must be ignored
    )
    # experiment: current column first, offset by 1 A; one point outside window
    te = np.array([-5.0, 10.0, 50.0, 90.0])
    pd.DataFrame({"current": 2 * te - 1, "time": te}).to_csv(
        tmp_path / "exp_i.csv", index=False
    )
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "circuits": [
                    {
                        "circuit_id": "A",
                        "experiment_data": [
                            {"type": "current", "key": "current", "file": "exp_i.csv"},
                            {
                                "type": "voltage",
                                "key": "voltage",
                                "file": "missing.csv",
                            },
                        ],
                    }
                ]
            }
        )
    )
    return tmp_path


def test_load_run(run_files):
    run = viewer.load_run(run_files / "run.npz")
    assert run.circuits == ["A"]
    assert set(run.series["A"]) == {"current", "reference", "voltage", "temperature"}
    assert set(run.experiments["A"]) == {"current"}  # missing file skipped


def test_residual_is_sim_minus_exp_inside_window(run_files):
    run = viewer.load_run(run_files / "run.npz")
    t, r = viewer.residual(run, "A", "current")
    assert t.min() >= 0
    np.testing.assert_allclose(r, 1.0, atol=1e-9)
    assert viewer.residual(run, "A", "voltage") is None


def test_no_config_means_no_experiments(run_files):
    (run_files / "run.json").unlink()
    run = viewer.load_run(run_files / "run.npz")
    assert run.experiments == {}
    viewer.plot_residuals(run)  # must not raise


def test_views_render(run_files, tmp_path):
    run = viewer.load_run(run_files / "run.npz")
    for fig in (
        viewer.plot_overview(run),
        viewer.plot_residuals(run),
        viewer.plot_compare([run, run], "A", "current"),
    ):
        fig.savefig(tmp_path / "out.png")


def test_circuit_toggle_hides_lines(run_files):
    run = viewer.load_run(run_files / "run.npz")
    fig = viewer.plot_overview(run)
    fig._toggle.set_active(0)
    lines = [ln for ax in fig.axes[:-1] for ln in ax.lines]
    assert lines and not any(ln.get_visible() for ln in lines)


def test_flow_fit(tmp_path):
    pytest.importorskip("python_magnetcooling")
    rng = np.random.default_rng(0)
    i = np.linspace(400, 28000, 200)
    vp = 2840 * (i / 28000) ** 2 + 1000
    x = vp / 3840
    pd.DataFrame(
        {
            "current": i,
            "pump_speed": vp + rng.normal(0, 5, i.size),
            "flow_rate": 10 + 60 * x + rng.normal(0, 0.3, i.size),
            "pressure": 4 + 20 * x**2 + rng.normal(0, 0.1, i.size),
            "back_pressure": 4 + rng.normal(0, 0.05, i.size),
        }
    ).to_csv(tmp_path / "rec.csv", index=False)
    viewer.plot_flow_fit(tmp_path / "rec.csv", imax=28000).savefig(tmp_path / "f.png")
