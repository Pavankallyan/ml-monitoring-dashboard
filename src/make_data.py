"""Synthetic IoT device telemetry generator with controllable drift.

Features:
    temp_c       - device temperature (°C)
    humidity     - ambient relative humidity (%)
    battery_v    - battery voltage (V)
    signal_dbm   - radio signal strength (dBm)
    uptime_hrs   - hours since last reboot

Label: will_fail (1 if the device is expected to fail soon).

`make_batch(n, drift_shift, concept_flip)` produces a production-style batch:
  - drift_shift shifts the means of temp_c and battery_v (sensor drift)
  - concept_flip flips a fraction of labels (concept drift / label noise)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["temp_c", "humidity", "battery_v", "signal_dbm", "uptime_hrs"]
LABEL = "will_fail"

_rng = np.random.default_rng(42)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_batch(
    n: int,
    drift_shift: float = 0.0,
    concept_flip: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate n synthetic IoT telemetry rows.

    Args:
        n: number of rows
        drift_shift: added to the mean of temp_c (°C) and subtracted
            (scaled) from battery_v — simulates gradual sensor drift.
        concept_flip: fraction of labels randomly flipped — simulates
            concept drift / label noise.
        seed: optional RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed) if seed is not None else _rng

    temp_c = rng.normal(52.0 + drift_shift, 8.0, n)
    humidity = rng.normal(45.0, 12.0, n)
    battery_v = rng.normal(3.30 - 0.08 * drift_shift, 0.15, n)
    signal_dbm = rng.normal(-68.0, 9.0, n)
    uptime_hrs = rng.exponential(900.0, n)

    # Failure likelihood: heat + weak battery + long uptime drive risk.
    logit = (
        0.09 * (temp_c - 52.0)
        - 6.0 * (battery_v - 3.30)
        + 0.0012 * uptime_hrs
        - 2.6
    )
    p_fail = _logistic(logit)
    will_fail = (rng.random(n) < p_fail).astype(int)

    if concept_flip > 0:
        flip = rng.random(n) < concept_flip
        will_fail = np.where(flip, 1 - will_fail, will_fail)

    return pd.DataFrame(
        {
            "temp_c": temp_c,
            "humidity": humidity,
            "battery_v": battery_v,
            "signal_dbm": signal_dbm,
            "uptime_hrs": uptime_hrs,
            "will_fail": will_fail,
        }
    )


def make_training_set(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Clean, no-drift training data."""
    return make_batch(n, drift_shift=0.0, concept_flip=0.0, seed=seed)


if __name__ == "__main__":
    df = make_training_set()
    print(df.head())
    print(df.describe().round(2))
