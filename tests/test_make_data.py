"""Unit tests for src/make_data.py — the synthetic IoT telemetry generator.

Covers the batch schema, seeded determinism, the drift controls
(temp_c up / battery_v down with drift_shift), and the label contract.
All tests use small seeded batches, so the suite runs fast with no
artifacts or model files.
"""

from src.make_data import FEATURES, LABEL, make_batch, make_training_set


def test_make_batch_schema_and_size():
    batch = make_batch(200, seed=7)
    assert list(batch.columns) == FEATURES + [LABEL]
    assert len(batch) == 200


def test_make_batch_label_is_binary():
    batch = make_batch(500, seed=7)
    assert set(batch[LABEL].unique()) <= {0, 1}


def test_make_batch_seed_is_deterministic():
    a = make_batch(300, seed=42)
    b = make_batch(300, seed=42)
    assert a.equals(b)


def test_make_batch_different_seeds_differ():
    a = make_batch(300, seed=1)
    b = make_batch(300, seed=2)
    assert not a.equals(b)


def test_make_batch_drift_shifts_temp_up_and_battery_down():
    clean = make_batch(5000, drift_shift=0.0, seed=11)
    drifted = make_batch(5000, drift_shift=4.0, seed=11)
    # Same seed, so the only systematic change is the drift shift:
    # temp_c mean rises by ~4 C, battery_v mean falls by ~0.32 V.
    assert drifted["temp_c"].mean() > clean["temp_c"].mean() + 2.0
    assert drifted["battery_v"].mean() < clean["battery_v"].mean() - 0.2
    # Undrifted features stay close (same seed, same draws in sequence).
    assert abs(drifted["humidity"].mean() - clean["humidity"].mean()) < 1.0


def test_make_batch_no_drift_matches_training_distribution():
    batch = make_batch(5000, drift_shift=0.0, seed=3)
    assert 45.0 < batch["temp_c"].mean() < 59.0
    assert 3.1 < batch["battery_v"].mean() < 3.5


def test_make_batch_uptime_is_positive():
    batch = make_batch(500, seed=9)
    assert (batch["uptime_hrs"] > 0).all()


def test_make_batch_concept_flip_changes_some_labels():
    # With the same seed, flipping labels must change at least one label
    # but keep the label column binary.
    clean = make_batch(2000, concept_flip=0.0, seed=5)
    flipped = make_batch(2000, concept_flip=0.5, seed=5)
    assert set(flipped[LABEL].unique()) <= {0, 1}
    assert (flipped[LABEL] != clean[LABEL]).sum() > 0


def test_make_training_set_defaults():
    df = make_training_set()
    assert len(df) == 3000
    assert list(df.columns) == FEATURES + [LABEL]
    # Clean training data: no drift applied.
    assert 45.0 < df["temp_c"].mean() < 59.0
    assert set(df[LABEL].unique()) <= {0, 1}
