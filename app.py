"""
app.py — Flask REST API for Student Placement Prediction
Endpoints:
  GET  /health          → service health
  GET  /stats           → dataset & model statistics
  POST /predict         → single student prediction
  POST /predict-batch   → batch CSV upload prediction
"""

import json
import io
import traceback
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
ART_DIR   = BASE_DIR.parent / "model" / "artefacts"
DATA_PATH = BASE_DIR.parent / "data" / "dataset.csv"

app = Flask(__name__)
CORS(app)

# ── Load artefacts ────────────────────────────────────────────────────────────
model          = joblib.load(ART_DIR / "model.pkl")
scaler         = joblib.load(ART_DIR / "scaler.pkl")
label_encoders = joblib.load(ART_DIR / "label_encoders.pkl")
with open(ART_DIR / "meta.json") as f:
    meta = json.load(f)

FEATURE_COLS = meta["feature_cols"]
CAT_COLS     = meta["cat_cols"]


# ── Feature engineering helper ────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same feature engineering used during training."""
    df = df.copy()
    df["skill_avg"]        = (df["coding_skill_score"] + df["aptitude_score"] +
                               df["communication_skill_score"] + df["logical_reasoning_score"]) / 4
    df["activity_score"]   = (df["internships_count"] * 3 + df["projects_count"] * 2 +
                               df["certifications_count"] + df["hackathons_participated"] * 2 +
                               df["github_repos"])
    df["cgpa_skill"]       = df["cgpa"] * df["skill_avg"] / 100
    df["readiness_score"]  = (df["skill_avg"] * 0.4 + df["mock_interview_score"] * 0.3 +
                               df["cgpa"] * 10 * 0.3)
    df["social_capital"]   = np.log1p(df["linkedin_connections"])
    df["backlogs_penalty"] = (df["backlogs"] > 0).astype(int)
    df["high_cgpa"]        = (df["cgpa"] >= 8.0).astype(int)
    return df


def encode_and_scale(df: pd.DataFrame):
    """Encode categoricals, reorder features, scale."""
    for col in CAT_COLS:
        le = label_encoders[col]
        df[col] = df[col].astype(str).map(
            lambda v, _le=le: (
                _le.transform([v])[0]
                if v in _le.classes_
                else _le.transform([_le.classes_[0]])[0]
            )
        )
    X = df[FEATURE_COLS].values
    return scaler.transform(X)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": meta["best_model"]})


@app.route("/stats")
def stats():
    df = pd.read_csv(DATA_PATH)
    placed_pct = round((df["placement_status"] == "Placed").mean() * 100, 2)
    branch_rate = (
        df.groupby("branch")["placement_status"]
        .apply(lambda s: round((s == "Placed").mean() * 100, 2))
        .to_dict()
    )
    tier_rate = (
        df.groupby("college_tier")["placement_status"]
        .apply(lambda s: round((s == "Placed").mean() * 100, 2))
        .to_dict()
    )
    gender_rate = (
        df.groupby("gender")["placement_status"]
        .apply(lambda s: round((s == "Placed").mean() * 100, 2))
        .to_dict()
    )
    avg_salary = round(df.loc[df["salary_package_lpa"] > 0, "salary_package_lpa"].mean(), 2)
    return jsonify({
        "total_students": len(df),
        "placed_percentage": placed_pct,
        "avg_salary_lpa": avg_salary,
        "branch_placement_rate": branch_rate,
        "tier_placement_rate": tier_rate,
        "gender_placement_rate": gender_rate,
        "model_results": meta["results"],
        "best_model": meta["best_model"],
        "feature_cols": FEATURE_COLS,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        df = engineer_features(df)
        X_sc = encode_and_scale(df)
        proba = model.predict_proba(X_sc)[0]
        pred  = int(np.argmax(proba))
        return jsonify({
            "prediction":   "Placed" if pred == 1 else "Not Placed",
            "probability":  round(float(proba[1]), 4),
            "confidence":   round(float(max(proba)), 4),
            "label":        pred,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        df_orig = pd.read_csv(file)
        df = df_orig.copy()

        # Drop target columns if present
        df.drop(columns=["placement_status", "salary_package_lpa", "student_id"],
                inplace=True, errors="ignore")

        df = engineer_features(df)
        X_sc = encode_and_scale(df)
        probas = model.predict_proba(X_sc)
        preds  = np.argmax(probas, axis=1)

        df_orig["Predicted_Status"]      = ["Placed" if p == 1 else "Not Placed" for p in preds]
        df_orig["Placement_Probability"] = np.round(probas[:, 1], 4)

        out = io.StringIO()
        df_orig.to_csv(out, index=False)
        out.seek(0)
        return send_file(
            io.BytesIO(out.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="predictions.csv",
        )
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400


if __name__ == "__main__":
    print("Starting Flask API on http://localhost:5000")
    app.run(debug=True, port=5000)
