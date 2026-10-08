"""Walkthrough: catch a silently drifting sensor with from-scratch drift metrics.

Generates a clean baseline batch plus three production-style batches —
clean, mildly drifted, severely drifted — and runs the same checks the
alert pipeline uses: PSI + two-sample KS per feature, with the standard
PSI bands (<0.1 no shift, 0.1-0.25 small shift, >0.25 major shift).

No model or labels needed: that is the point. These are the leading
indicators that fire before true accuracy can be measured in production.

Usage:  python examples/drift_walkthrough.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.make_data import FEATURES, make_batch
from src.metrics import PSI_ALERT, ks_statistic, psi, psi_band


def check_batch(baseline, batch, name):
    print(f"\n=== {name} ===")
    print(f"{'feature':<12} {'PSI':>8} {'band':>12} {'KS':>8}  alert?")
    any_alert = False
    for f in FEATURES:
        v_psi = psi(baseline[f], batch[f])
        v_ks = ks_statistic(baseline[f], batch[f])
        alert = v_psi > PSI_ALERT
        any_alert |= alert
        print(f"{f:<12} {v_psi:>8.3f} {psi_band(v_psi):>12} {v_ks:>8.3f}  {'🚨' if alert else '–'}")
    print(f"=> drift alert: {'YES' if any_alert else 'no'}")
    return any_alert


def main():
    baseline = make_batch(2000, drift_shift=0.0, seed=42)

    clean = make_batch(500, drift_shift=0.0, seed=101)
    mild = make_batch(500, drift_shift=0.8, seed=102)
    severe = make_batch(500, drift_shift=5.0, seed=103)

    assert not check_batch(baseline, clean, "batch: clean (no drift)")
    assert not check_batch(baseline, mild, "batch: mild sensor drift")
    # Battery voltage (the most sensitive channel) already shows a small shift.
    assert psi_band(psi(baseline["battery_v"], mild["battery_v"])) == "small shift"
    assert check_batch(baseline, severe, "batch: severe sensor drift")

    print("\nDone — the PSI alert fired on the severely drifted batch, "
          "with zero labels required.")


if __name__ == "__main__":
    main()
