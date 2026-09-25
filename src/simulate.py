"""Simulate 12 sequential 'production' batches with injected drift.

Batches 1-4 : clean (no drift)
Batches 5-8 : gradual mean-shift drift on temp_c / battery_v
Batches 9-12: stronger drift + mild concept drift (label flips)

Each batch is scored with the trained model; drift (PSI, KS) and
performance (accuracy, F1) are computed and persisted to artifacts.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from make_data import FEATURES, LABEL, make_batch
from metrics import ACCURACY_DROP_ALERT, PSI_ALERT, batch_metrics

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
N_BATCHES = 12
BATCH_SIZE = 400


def batch_plan(batch_id: int) -> tuple[float, float]:
    """Return (drift_shift, concept_flip) for a given 1-based batch id."""
    if 1 <= batch_id <= 4:
        return 0.0, 0.0
    if 5 <= batch_id <= 8:
        # gradual mean shift: 1.5°C … 6°C
        return 1.5 * (batch_id - 4), 0.0
    # stronger drift + mild concept drift
    return 6.0 + 1.5 * (batch_id - 8), 0.05


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    baseline = pd.read_parquet(os.path.join(ARTIFACT_DIR, "baseline.parquet"))
    with open(os.path.join(ARTIFACT_DIR, "baseline_metrics.json")) as f:
        baseline_metrics = json.load(f)

    rows: list[dict] = []
    alerts: list[dict] = []

    for batch_id in range(1, N_BATCHES + 1):
        drift_shift, concept_flip = batch_plan(batch_id)
        batch = make_batch(
            BATCH_SIZE,
            drift_shift=drift_shift,
            concept_flip=concept_flip,
            seed=1000 + batch_id,
        )
        y_pred = model.predict(batch[FEATURES])
        m = batch_metrics(baseline, batch, FEATURES, batch[LABEL], y_pred)

        rows.append(
            {
                "batch": batch_id,
                "drift_shift": drift_shift,
                "concept_flip": concept_flip,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "positive_rate": m["positive_rate"],
                "max_psi": m["max_psi"],
                "max_psi_feature": m["max_psi_feature"],
                "positive_rate_pred": float(np.mean(y_pred)),
                **{f"psi_{f}": m["psi_by_feature"][f] for f in FEATURES},
                **{f"ks_{f}": m["ks_by_feature"][f] for f in FEATURES},
            }
        )

        batch_alerts: list[str] = []
        for f in FEATURES:
            if m["psi_by_feature"][f] > PSI_ALERT:
                batch_alerts.append(
                    f"Major drift on '{f}': PSI={m['psi_by_feature'][f]:.3f}"
                )
        acc_drop = baseline_metrics["accuracy"] - m["accuracy"]
        if acc_drop > ACCURACY_DROP_ALERT:
            batch_alerts.append(
                f"Accuracy dropped {acc_drop:.1%} vs baseline "
                f"({baseline_metrics['accuracy']:.3f} -> {m['accuracy']:.3f})"
            )
        if batch_alerts:
            alerts.append({"batch": batch_id, "alerts": batch_alerts})

    pd.DataFrame(rows).to_csv(
        os.path.join(ARTIFACT_DIR, "batch_metrics.csv"), index=False
    )
    with open(os.path.join(ARTIFACT_DIR, "alerts.json"), "w") as f:
        json.dump(alerts, f, indent=2)

    print(f"Wrote {N_BATCHES} batch metrics; {len(alerts)} batches raised alerts.")


if __name__ == "__main__":
    main()
