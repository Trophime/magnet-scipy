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


def test_unreadable_source_warns_and_is_skipped(run_files):
    cfg = json.loads((run_files / "run.json").read_text())
    cfg["circuits"][0]["experiment_data"].append(
        {
            "type": "voltage",
            "key": "voltage",
            "file": ["missing.tdms"],
            "group": "Tensions_Aimant",
            "channel": "ALL internes",
        }
    )
    (run_files / "run.json").write_text(json.dumps(cfg))
    with pytest.warns(UserWarning, match="missing.tdms"):
        run = viewer.load_run(run_files / "run.npz")
    assert set(run.experiments["A"]) == {"current"}


def test_decimate_keeps_envelope():
    t = np.arange(1_000_000, dtype=float)
    y = np.sin(t / 1000.0)
    y[123_457] = 5.0  # single-sample spike must survive
    td, yd = viewer._decimate(t, y, max_points=2000)
    assert len(td) <= 2000
    assert yd.max() == 5.0 and yd.min() == pytest.approx(-1.0, abs=1e-6)
    assert np.all(np.diff(td) >= 0)
    t5, y5 = viewer._decimate(t[:10], y[:10], max_points=2000)
    assert len(t5) == 10


@pytest.mark.parametrize(
    "name, expected",
    [
        # summer: CEST = UTC+2
        (
            "pbsurv/M9/Fichiers_Default/M9_Default_230919-193827_Courants50Hz.tdms",
            ("default", "2023-09-19 17:38:27", "Courants50Hz"),
        ),
        ("M9_Spikes_240719-182431.tdms", ("spike", "2024-07-19 16:24:31", "")),
        # winter: CET = UTC+1, HHMM form
        ("M8_Default_251105-0953_raw.tdms", ("default", "2025-11-05 08:53:00", "raw")),
        # autumn fall-back 02:30 occurs twice -> first (DST) occurrence
        ("M9_Spikes_231029-023000.tdms", ("spike", "2023-10-29 00:30:00", "")),
        # spring gap 02:30 does not exist -> shifted forward to 03:00 CEST
        ("M9_Spikes_240331-023000.tdms", ("spike", "2024-03-31 01:00:00", "")),
    ],
)
def test_parse_event_file(name, expected):
    kind, utc, subtype = viewer.parse_event_file(name)
    assert (kind, str(utc), subtype) == expected


@pytest.mark.parametrize(
    "name",
    ["M9_Overview_230916-1755.tdms", "M9_Default_bad-stamp.tdms", "M9_Default.csv"],
)
def test_parse_event_file_rejects(name):
    assert viewer.parse_event_file(name) is None


def _events_source():
    return {
        "window_utc": ["2023-09-19 17:38:00", "2023-09-19 17:40:00"],
        "default": ["x/M9_Default_230919-193857_Courants50Hz.tdms"],
        "spike": ["M9_Spikes_230919-193810.tdms", "M9_Overview_230919-1906.tdms"],
    }


def test_load_events_record_and_provenance_keys():
    ev = viewer.load_events(_events_source())
    assert [(e.kind, e.t, e.subtype) for e in ev] == [
        ("spike", 10.0, ""),
        ("default", 57.0, "Courants50Hz"),
    ]
    record = {
        "t0": "2023-09-19 17:38:00",
        "sources_default": _events_source()["default"],
        "sources_spike": [],
    }
    assert [e.t for e in viewer.load_events(record)] == [57.0]


def test_events_from_provenance_and_toggle(run_files):
    (run_files / "provenance.json").write_text(
        json.dumps({"runs": {"r": {**_events_source(), "config": "run.json"}}})
    )
    run = viewer.load_run(run_files / "run.npz")
    assert [e.kind for e in run.events] == ["spike", "default"]

    fig = viewer.plot_overview(run)
    assert [t.get_text() for t in fig._toggle.labels] == ["A", "spike", "default"]
    fig._toggle.set_active(2)  # hide defaults
    defaults = [
        ln for ax in fig.axes[:-1] for ln in ax.lines if ln.get_color() == "red"
    ]
    assert defaults and not any(ln.get_visible() for ln in defaults)
    viewer.plot_residuals(run)


def test_events_option_overrides_provenance(run_files, tmp_path):
    (tmp_path / "ev.json").write_text(
        json.dumps({"t0": "2023-09-19 17:38:00", "sources_spike": []})
    )
    run = viewer.load_run(run_files / "run.npz", events=tmp_path / "ev.json")
    assert run.events == []


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
