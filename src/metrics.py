"""From-scratch drift & performance metrics (no evidently dependency).

Pure functions here are intentionally unit-testable: they take numpy
arrays / pandas Series and return floats or small dicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# PSI bands (standard industry practice)
PSI_NO_SHIFT = 0.1
PSI_MAJOR_SHIFT = 0.25

# Alert policy
PSI_ALERT = 0.25          # alert when PSI > 0.25 on any feature
ACCURACY_DROP_ALERT = 0.05  # alert when batch accuracy drops >5 pts vs baseline

_N_BINS = 10
_EPS = 1e-6


def _quantile_edges(expected: np.ndarray, n_bins: int = _N_BINS) -> np.ndarray:
    """Bin edges from quantiles of the expected (baseline) distribution."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(expected, qs)
    # Guard against duplicate edges when the feature has little variance.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + _EPS
    return edges


def psi(expected: np.ndarray | pd.Series, actual: np.ndarray | pd.Series) -> float:
    """Population Stability Index between two samples.

    Bands: < 0.1 = no significant shift, 0.1-0.25 = small shift,
    > 0.25 = significant shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = _quantile_edges(expected)
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = e_counts / e_counts.sum()
    a_pct = a_counts / a_counts.sum()
    e_pct = np.clip(e_pct, _EPS, None)
    a_pct = np.clip(a_pct, _EPS, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_band(value: float) -> str:
    if value < PSI_NO_SHIFT:
        return "no shift"
    if value < PSI_MAJOR_SHIFT:
        return "small shift"
    return "major shift"


def ks_statistic(expected: np.ndarray | pd.Series, actual: np.ndarray | pd.Series) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max ECDF distance)."""
    e = np.sort(np.asarray(expected, dtype=float))
    a = np.sort(np.asarray(actual, dtype=float))
    grid = np.sort(np.concatenate([e, a]))
    ecdf_e = np.searchsorted(e, grid, side="right") / len(e)
    ecdf_a = np.searchsorted(a, grid, side="right") / len(a)
    return float(np.max(np.abs(ecdf_e - ecdf_a)))


def batch_metrics(
    baseline: pd.DataFrame,
    batch: pd.DataFrame,
    features: list[str],
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """All drift + performance metrics for one production batch."""
    from sklearn.metrics import accuracy_score, f1_score

    psi_by_feature = {f: psi(baseline[f], batch[f]) for f in features}
    ks_by_feature = {f: ks_statistic(baseline[f], batch[f]) for f in features}
    max_psi_feature = max(psi_by_feature, key=psi_by_feature.get)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_pred)),
        "psi_by_feature": psi_by_feature,
        "ks_by_feature": ks_by_feature,
        "max_psi": float(psi_by_feature[max_psi_feature]),
        "max_psi_feature": max_psi_feature,
    }
