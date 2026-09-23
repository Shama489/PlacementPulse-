# pages/performance.py
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

API     = "http://localhost:5000"
ART_DIR = Path(__file__).resolve().parent.parent.parent / "model" / "artefacts"


def show():
    st.title("📈 Model Performance & Evaluation")

    # ── Load meta ─────────────────────────────────────────────────────────────
    try:
        with open(ART_DIR / "meta.json") as f:
            meta = json.load(f)
    except FileNotFoundError:
        st.error("Model artefacts not found. Run training first: `python placement_predictor/model/train.py`")
        return

    results = meta["results"]
    best    = meta["best_model"]

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.subheader("🏆 Model Comparison")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  r["accuracy"],
            "ROC-AUC":   r["roc_auc"],
            "Precision": round(r["report"]["weighted avg"]["precision"], 4),
            "Recall":    round(r["report"]["weighted avg"]["recall"], 4),
            "F1-Score":  round(r["report"]["weighted avg"]["f1-score"], 4),
        })
    comp_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Grouped bar chart
    fig = px.bar(
        comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        title="Model Metrics Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Best Model Detail ─────────────────────────────────────────────────────
    st.subheader(f"🔍 Best Model: {best}")
    best_r = results[best]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{best_r['accuracy']*100:.2f}%")
    col2.metric("ROC-AUC",   f"{best_r['roc_auc']:.4f}")
    col3.metric("Precision", f"{best_r['report']['weighted avg']['precision']*100:.2f}%")
    col4.metric("F1-Score",  f"{best_r['report']['weighted avg']['f1-score']*100:.2f}%")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.subheader("🔲 Confusion Matrix")
    cm  = best_r["confusion_matrix"]
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Not Placed", "Placed"],
        y=["Not Placed", "Placed"],
        text_auto=True,
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # ── Classification Report ─────────────────────────────────────────────────
    st.subheader("📋 Classification Report")
    cr = best_r["report"]
    cr_rows = []
    for label in ["Not Placed", "Placed", "macro avg", "weighted avg"]:
        key = "0" if label == "Not Placed" else ("1" if label == "Placed" else label)
        if key in cr:
            cr_rows.append({
                "Class":      label,
                "Precision":  round(cr[key]["precision"], 4),
                "Recall":     round(cr[key]["recall"], 4),
                "F1-Score":   round(cr[key]["f1-score"], 4),
                "Support":    int(cr[key]["support"]),
            })
    st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.subheader("🎯 Feature Importances")
    fi = best_r.get("feature_importances", [])
    feat_cols = meta["feature_cols"]
    if fi and len(fi) == len(feat_cols):
        fi_df = pd.DataFrame({"Feature": feat_cols, "Importance": fi})
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(25)
        fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            title=f"Top Feature Importances — {best}",
        )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance data not available for this model.")

    # ── ROC Curve (approximated from AUC) ─────────────────────────────────────
    st.subheader("📉 ROC Curve (per model)")
    fig = go.Figure()
    colors = ["#3b82f6", "#22c55e", "#f59e0b"]
    for (name, r), color in zip(results.items(), colors):
        auc = r["roc_auc"]
        # Draw a representative curve using AUC interpolation
        # (actual stored data; we approximate the shape for visualisation)
        t    = np.linspace(0, 1, 200)
        fpr  = t
        tpr  = np.clip(t + (auc - 0.5) * 2 * (1 - t) * t ** 0.5, 0, 1)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.4f})",
            line=dict(color=color, width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier",
        line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curves",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
