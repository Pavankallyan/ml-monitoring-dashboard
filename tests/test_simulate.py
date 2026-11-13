"""Unit tests for src/simulate.py — the 12-batch drift simulation.

batch_plan() is pure and tested directly. main() is run end to end
against a tmp artifact dir (train.main() first, so a real model and
baseline exist), never touching the committed artifacts/ directory.
"""

import json
import os
import sys

import pandas as pd
import pytest

# simulate.py does `from make_data import ...`, so src/ must be importable.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ),
)

import src.simulate as simulate_mod  # noqa: E402
import src.train as train_mod  # noqa: E402
from src.make_data import FEATURES  # noqa: E402
from src.simulate import N_BATCHES, batch_plan, main  # noqa: E402
from src.train import main as train_main  # noqa: E402


# ------------------------------------------------------------ batch_plan


def test_batch_plan_clean_batches():
    for batch_id in (1, 2, 3, 4):
        assert batch_plan(batch_id) == (0.0, 0.0)


def test_batch_plan_gradual_drift_batches():
    assert batch_plan(5) == (1.5, 0.0)
    assert batch_plan(6) == (3.0, 0.0)
    assert batch_plan(7) == (4.5, 0.0)
    assert batch_plan(8) == (6.0, 0.0)


def test_batch_plan_strong_drift_plus_concept_flip():
    assert batch_plan(9) == (7.5, 0.05)
    assert batch_plan(10) == (9.0, 0.05)
    assert batch_plan(12) == (12.0, 0.05)


def test_batch_plan_drift_ramp_is_monotonic():
    shifts = [batch_plan(b)[0] for b in range(1, N_BATCHES + 1)]
    assert shifts == sorted(shifts)
    assert shifts[4] < shifts[11]  # drift grows from batch 5 to batch 12
    flips = [batch_plan(b)[1] for b in range(1, N_BATCHES + 1)]
    assert flips[:8] == [0.0] * 8
    assert flips[8:] == [0.05] * 4


# ------------------------------------------------------------ main (end to end)


@pytest.fixture()
def trained_artifacts(tmp_path, monkeypatch):
    """Train a real model into a tmp artifact dir; simulate uses the same."""
    d = tmp_path / "artifacts"
    monkeypatch.setattr(train_mod, "ARTIFACT_DIR", str(d))
    monkeypatch.setattr(simulate_mod, "ARTIFACT_DIR", str(d))
    train_main()
    return d


def test_main_writes_twelve_batch_rows(trained_artifacts):
    main()
    metrics = pd.read_csv(trained_artifacts / "batch_metrics.csv")
    assert len(metrics) == N_BATCHES
    assert list(metrics["batch"]) == list(range(1, N_BATCHES + 1))
    assert set(metrics.columns) >= {
        "batch", "drift_shift", "concept_flip", "accuracy", "f1",
        "positive_rate", "max_psi", "max_psi_feature", "positive_rate_pred",
        *[f"psi_{f}" for f in FEATURES],
        *[f"ks_{f}" for f in FEATURES],
    }


def test_batch_rows_match_batch_plan(trained_artifacts):
    main()
    metrics = pd.read_csv(trained_artifacts / "batch_metrics.csv")
    for _, row in metrics.iterrows():
        shift, flip = batch_plan(int(row["batch"]))
        assert row["drift_shift"] == pytest.approx(shift)
        assert row["concept_flip"] == pytest.approx(flip)


def test_drift_detected_on_first_drifted_batch(trained_artifacts):
    main()
    metrics = pd.read_csv(trained_artifacts / "batch_metrics.csv")
    clean = metrics[metrics["batch"] <= 4]
    assert (clean["max_psi"] < 0.25).all()  # no drift alert on clean batches
    drifted = metrics[metrics["batch"] >= 5]
    assert (drifted["max_psi"] > 0.25).all()
    assert drifted["max_psi"].iloc[0] == pytest.approx(0.558, abs=0.01)
    # battery_v is the feature the drift moves; it dominates every drifted batch.
    assert (drifted["max_psi_feature"] == "battery_v").all()


def test_accuracy_decays_under_drift(trained_artifacts):
    main()
    metrics = pd.read_csv(trained_artifacts / "batch_metrics.csv")
    first = metrics["accuracy"].iloc[0]
    worst = metrics["accuracy"].min()
    assert first == pytest.approx(0.8275, abs=0.01)
    assert worst < first - 0.15  # bottoms out well below the clean batches


def test_alerts_fire_from_batch_five_onwards(trained_artifacts):
    main()
    with open(trained_artifacts / "alerts.json") as f:
        alerts = json.load(f)
    assert [a["batch"] for a in alerts] == list(range(5, N_BATCHES + 1))
    for entry in alerts:
        assert entry["alerts"]  # every alerted batch names at least one cause
    # Batch 5 is drift-only: accuracy had only slipped ~5 pts, so the
    # PSI alert fires a full batch before the accuracy alert would.
    assert alerts[0]["alerts"] == ["Major drift on 'battery_v': PSI=0.558"]
    # Later batches also carry the accuracy-drop alert.
    assert any("Accuracy dropped" in a for a in alerts[-1]["alerts"])


def test_main_requires_trained_model(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    monkeypatch.setattr(simulate_mod, "ARTIFACT_DIR", str(d))
    with pytest.raises(FileNotFoundError):
        main()
