"""Shared fixtures for ml-monitoring-dashboard unit tests."""

import os
import sys

import numpy as np
import pytest

# Let tests import the repo's src package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.make_data import FEATURES, LABEL, make_batch  # noqa: E402


@pytest.fixture()
def rng():
    return np.random.default_rng(1234)


@pytest.fixture()
def baseline_batch():
    """Small seeded clean batch, like the drift baseline snapshot."""
    return make_batch(500, drift_shift=0.0, seed=11)


@pytest.fixture()
def drifted_batch():
    """Same generator with a strong sensor drift (drift_shift=4.0)."""
    return make_batch(500, drift_shift=4.0, seed=22)
