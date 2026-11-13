"""Unit tests for src/train.py — model training + artifact persistence.

main() is run against a tmp artifact dir (monkeypatched ARTIFACT_DIR),
so the suite never touches the committed artifacts/ directory.
Training is fully seeded (random_state=42 everywhere), so the baseline
metrics are exact: accuracy 0.81, F1 0.4956, positive rate 0.13.
"""

import json
import os
import pickle
import sys

import pandas as pd
import pytest

# train.py does `from make_data import ...`, so src/ must be importable.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ),
)

import src.train as train_mod  # noqa: E402
from src.make_data import FEATURES, make_batch  # noqa: E402
from src.train import main  # noqa: E402


@pytest.fixture()
def artifact_dir(tmp_path, monkeypatch):
    d = tmp_path / "artifacts"
    monkeypatch.setattr(train_mod, "ARTIFACT_DIR", str(d))
    return d


def test_main_returns_baseline_dict(artifact_dir):
    baseline = main()
    assert set(baseline) == {"accuracy", "f1", "positive_rate", "n_train"}
    assert baseline["n_train"] == 2400  # 80% of the 3000-row training set
    assert 0.0 <= baseline["accuracy"] <= 1.0
    assert 0.0 <= baseline["f1"] <= 1.0
    assert 0.0 <= baseline["positive_rate"] <= 1.0


def test_baseline_matches_committed_values(artifact_dir):
    # Seeded end to end: these are the exact numbers in
    # artifacts/baseline_metrics.json and the README results table.
    baseline = main()
    assert baseline["accuracy"] == pytest.approx(0.81)
    assert baseline["f1"] == pytest.approx(0.4956, abs=1e-3)
    assert baseline["positive_rate"] == pytest.approx(0.13)


def test_artifacts_written(artifact_dir):
    baseline = main()
    assert (artifact_dir / "model.pkl").is_file()
    assert (artifact_dir / "baseline.parquet").is_file()
    assert (artifact_dir / "baseline_metrics.json").is_file()

    with open(artifact_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    assert type(model).__name__ == "RandomForestClassifier"
    assert list(model.feature_names_in_) == FEATURES

    snapshot = pd.read_parquet(artifact_dir / "baseline.parquet")
    assert list(snapshot.columns) == FEATURES
    assert len(snapshot) == 2400  # the drift baseline is the training split

    with open(artifact_dir / "baseline_metrics.json") as f:
        saved = json.load(f)
    assert saved == baseline


def test_training_is_deterministic(tmp_path, monkeypatch):
    first_dir = tmp_path / "run1"
    monkeypatch.setattr(train_mod, "ARTIFACT_DIR", str(first_dir))
    first = main()
    second_dir = tmp_path / "run2"
    monkeypatch.setattr(train_mod, "ARTIFACT_DIR", str(second_dir))
    second = main()
    assert first == second


def test_creates_artifact_dir(tmp_path, monkeypatch):
    d = tmp_path / "fresh" / "artifacts"
    assert not d.exists()
    monkeypatch.setattr(train_mod, "ARTIFACT_DIR", str(d))
    main()
    assert d.is_dir()
    assert (d / "model.pkl").is_file()


def test_model_predicts_binary_labels(artifact_dir):
    main()
    with open(artifact_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    batch = make_batch(50, drift_shift=0.0, seed=7)
    preds = model.predict(batch[FEATURES])
    assert set(preds.tolist()) <= {0, 1}
    assert len(preds) == 50
