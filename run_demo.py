"""One command end-to-end: train -> simulate -> console summary."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from train import main as train_main  # noqa: E402
from simulate import main as simulate_main  # noqa: E402

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def main() -> None:
    baseline = train_main()
    simulate_main()

    import pandas as pd

    metrics = pd.read_csv(os.path.join(ARTIFACT_DIR, "batch_metrics.csv"))
    with open(os.path.join(ARTIFACT_DIR, "alerts.json")) as f:
        alerts = json.load(f)
    alert_batches = {a["batch"] for a in alerts}

    print("\n" + "=" * 64)
    print("ML MONITORING DASHBOARD - DEMO SUMMARY")
    print("=" * 64)
    print(
        f"Baseline (held-out test): accuracy={baseline['accuracy']:.3f}, "
        f"f1={baseline['f1']:.3f}, positive_rate={baseline['positive_rate']:.3f}"
    )
    print("-" * 64)
    print(f"{'batch':>5} {'accuracy':>8} {'f1':>6} {'max_PSI':>7} {'max_PSI_feat':>12}  alerts")
    for _, r in metrics.iterrows():
        flag = "ALERT" if r["batch"] in alert_batches else "-"
        print(
            f"{int(r['batch']):>5} {r['accuracy']:>8.3f} {r['f1']:>6.3f} "
            f"{r['max_psi']:>7.3f} {r['max_psi_feature']:>12}  {flag}"
        )
    print("-" * 64)
    print(f"Batches with alerts: {sorted(alert_batches) if alert_batches else 'none'}")
    print("Run the dashboard with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
