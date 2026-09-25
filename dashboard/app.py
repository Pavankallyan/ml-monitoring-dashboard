"""Streamlit dashboard: model monitoring for the IoT failure predictor.

Reads artifacts/ produced by run_demo.py:
  - batch_metrics.csv, alerts.json, baseline_metrics.json
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(BASE, "artifacts")
FEATURES = ["temp_c", "humidity", "battery_v", "signal_dbm", "uptime_hrs"]

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")
st.title("📉 ML Monitoring Dashboard — IoT Failure Predictor")


@st.cache_data
def load_artifacts():
    metrics = pd.read_csv(os.path.join(ART, "batch_metrics.csv"))
    with open(os.path.join(ART, "baseline_metrics.json")) as f:
        baseline = json.load(f)
    with open(os.path.join(ART, "alerts.json")) as f:
        alerts = json.load(f)
    return metrics, baseline, alerts


try:
    metrics, baseline, alerts = load_artifacts()
except FileNotFoundError:
    st.error(
        "Artifacts not found. Run `python run_demo.py` from the project root first."
    )
    st.stop()

alert_map = {a["batch"]: a["alerts"] for a in alerts}

# ---- KPI row ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline accuracy", f"{baseline['accuracy']:.3f}")
c2.metric("Latest batch accuracy", f"{metrics['accuracy'].iloc[-1]:.3f}")
c3.metric("Batches with alerts", f"{len(alert_map)} / {len(metrics)}")
c4.metric("Worst max-PSI batch", int(metrics.loc[metrics['max_psi'].idxmax(), 'batch']))

# ---- Batch slider ----
batch = st.slider("Production batch", 1, len(metrics), 1)
row = metrics.iloc[batch - 1]

# ---- Accuracy / F1 decay ----
st.subheader("Performance decay over batches")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(metrics["batch"], metrics["accuracy"], marker="o", label="accuracy")
ax.plot(metrics["batch"], metrics["f1"], marker="s", label="F1")
ax.axhline(baseline["accuracy"], color="gray", linestyle="--", label="baseline accuracy")
ax.set_xlabel("Batch")
ax.set_ylabel("Score")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# ---- PSI heatmap/table ----
st.subheader("PSI per feature per batch (red = major drift, PSI > 0.25)")
psi_cols = [f"psi_{f}" for f in FEATURES]
psi_df = metrics.set_index("batch")[psi_cols]
psi_df.columns = FEATURES


def _color(v):
    if v > 0.25:
        return "background-color: #ffb3b3"
    if v > 0.1:
        return "background-color: #ffe0b3"
    return ""


st.dataframe(psi_df.style.applymap(_color).format("{:.3f}"))

# ---- Prediction distribution shift ----
st.subheader(f"Batch {batch}: prediction positive-rate shift")
fig2, ax2 = plt.subplots(figsize=(8, 3.5))
ax2.bar(["baseline", f"batch {batch}"], [baseline["positive_rate"], row["positive_rate_pred"]], color=["gray", "steelblue"])
ax2.set_ylabel("Predicted failure rate")
st.pyplot(fig2)

# ---- Alerts panel ----
st.subheader("🚨 Alerts")
if alert_map:
    for b in sorted(alert_map):
        with st.expander(f"Batch {b} — {len(alert_map[b])} alert(s)", expanded=(b == batch)):
            for a in alert_map[b]:
                st.write("•", a)
else:
    st.success("No alerts raised across all batches.")
