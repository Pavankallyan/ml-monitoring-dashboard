"""Train the failure-prediction model and persist artifacts."""

from __future__ import annotations

import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from make_data import FEATURES, LABEL, make_training_set

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def main() -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = make_training_set()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[LABEL], test_size=0.2, random_state=42, stratify=df[LABEL]
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    baseline = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
        "n_train": len(X_train),
    }

    with open(os.path.join(ARTIFACT_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    X_train.to_parquet(os.path.join(ARTIFACT_DIR, "baseline.parquet"))
    with open(os.path.join(ARTIFACT_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline, f, indent=2)

    print("Baseline metrics:", baseline)
    return baseline


if __name__ == "__main__":
    main()
