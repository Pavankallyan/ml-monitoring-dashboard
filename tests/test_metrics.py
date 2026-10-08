"""Unit tests for src/metrics.py — the from-scratch drift metrics.

Everything here uses small seeded synthetic samples so the suite runs
fast with no artifacts or model files.
"""

import numpy as np
import pandas as pd
import pytest

from src.make_data import FEATURES
from src.metrics import (
    _quantile_edges,
    batch_metrics,
    ks_statistic,
    psi,
    psi_band,
)


# ---------------------------------------------------------------- psi


def test_psi_identical_samples_is_zero(rng):
    x = rng.normal(0, 1, 2000)
    assert psi(x, x) == 0.0


def test_psi_same_distribution_near_zero(rng):
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0, 1, 2000)
    value = psi(a, b)
    assert value < 0.1  # "no shift" band for same-distribution samples


def test_psi_shifted_distribution_is_major_shift(rng):
    a = rng.normal(0, 1, 2000)
    b = rng.normal(3, 1, 2000)
    value = psi(a, b)
    assert value > 0.25
    assert psi_band(value) == "major shift"


def test_psi_is_nonnegative_and_finite(rng):
    a = rng.normal(0, 1, 500)
    b = rng.normal(0.5, 2, 500)
    value = psi(a, b)
    assert 0.0 <= value < float("inf")


def test_psi_band_boundaries():
    assert psi_band(0.05) == "no shift"
    assert psi_band(0.099) == "no shift"
    assert psi_band(0.1) == "small shift"
    assert psi_band(0.249) == "small shift"
    assert psi_band(0.25) == "major shift"
    assert psi_band(10.0) == "major shift"


def test_quantile_edges_constant_feature():
    edges = _quantile_edges(np.full(100, 3.3))
    assert len(edges) == 11
    # The epsilon guard keeps edges strictly increasing even with no variance.
    assert all(b > a for a, b in zip(edges[:-1], edges[1:]))


# ------------------------------------------------------ ks_statistic


def test_ks_identical_samples_is_zero(rng):
    x = rng.normal(0, 1, 2000)
    assert ks_statistic(x, x) == 0.0


def test_ks_shifted_distribution_near_one(rng):
    a = rng.normal(0, 1, 2000)
    b = rng.normal(4, 1, 2000)
    assert ks_statistic(a, b) > 0.9


def test_ks_bounded_and_small_for_mild_shift(rng):
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0.1, 1, 2000)
    value = ks_statistic(a, b)
    assert 0.0 <= value <= 1.0
    assert value < 0.3


# ------------------------------------------------------ batch_metrics


def _perfect_predictions(batch):
    y_true = batch["will_fail"]
    return y_true, y_true.to_numpy()


def test_batch_metrics_keys(baseline_batch, drifted_batch):
    y_true, y_pred = _perfect_predictions(drifted_batch)
    out = batch_metrics(baseline_batch, drifted_batch, FEATURES, y_true, y_pred)
    assert set(out) == {
        "accuracy",
        "f1",
        "positive_rate",
        "psi_by_feature",
        "ks_by_feature",
        "max_psi",
        "max_psi_feature",
    }
    assert set(out["psi_by_feature"]) == set(FEATURES)
    assert set(out["ks_by_feature"]) == set(FEATURES)


def test_batch_metrics_perfect_predictions(baseline_batch, drifted_batch):
    y_true, y_pred = _perfect_predictions(drifted_batch)
    out = batch_metrics(baseline_batch, drifted_batch, FEATURES, y_true, y_pred)
    assert out["accuracy"] == 1.0
    assert out["f1"] == 1.0
    assert out["positive_rate"] == pytest.approx(float(y_true.mean()))


def test_batch_metrics_flags_drifted_feature(baseline_batch, drifted_batch):
    y_true, y_pred = _perfect_predictions(drifted_batch)
    out = batch_metrics(baseline_batch, drifted_batch, FEATURES, y_true, y_pred)
    # drift_shift=4.0 moves battery_v by >2 sigma; it must dominate the PSI table.
    assert out["max_psi_feature"] == "battery_v"
    assert out["max_psi"] > 0.25
    assert psi_band(out["max_psi"]) == "major shift"


def test_batch_metrics_zero_predictions(baseline_batch, drifted_batch):
    y_true = drifted_batch["will_fail"]
    y_pred = np.zeros(len(drifted_batch), dtype=int)
    out = batch_metrics(baseline_batch, drifted_batch, FEATURES, y_true, y_pred)
    assert out["positive_rate"] == 0.0
    assert out["accuracy"] == pytest.approx(float((y_true == 0).mean()))
    # all-negative predictions: F1 is defined via zero_division=0, no exception
    assert out["f1"] == 0.0


def test_batch_metrics_accepts_series_inputs(rng):
    n = 300
    baseline = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(5, 1, n)})
    batch = pd.DataFrame({"x": rng.normal(2, 1, n), "y": rng.normal(5, 1, n)})
    y_true = pd.Series(rng.integers(0, 2, n))
    y_pred = y_true.to_numpy()
    out = batch_metrics(baseline, batch, ["x", "y"], y_true, y_pred)
    assert out["max_psi_feature"] == "x"
    assert out["accuracy"] == 1.0
