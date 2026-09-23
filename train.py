"""
train.py — Preprocess dataset, train placement prediction models, save artefacts.
Run once: python placement_predictor/model/train.py
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = BASE_DIR.parent / "data" / "dataset.csv"
ART_DIR    = BASE_DIR / "artefacts"
ART_DIR.mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ── Encode target first (keep for reference) ─────────────────────────────────
df["placed"] = (df["placement_status"] == "Placed").astype(int)

# ── Feature Engineering ───────────────────────────────────────────────────────
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

# ── Drop non-predictive columns ───────────────────────────────────────────────
df.drop(columns=["student_id", "salary_package_lpa", "placement_status"],
        inplace=True, errors="ignore")

# ── Encode categoricals ───────────────────────────────────────────────────────
label_encoders = {}
CAT_COLS = ["gender", "branch", "college_tier", "volunteer_experience"]
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ── Feature / Target split ────────────────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c != "placed"]
X = df[FEATURE_COLS].values
y = df["placed"].values

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Handle imbalance ──────────────────────────────────────────────────────────
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_sc, y_train)
print(f"  After SMOTE: {np.bincount(y_res)}")

# ── Train models ──────────────────────────────────────────────────────────────
models = {
    "RandomForest":       RandomForestClassifier(n_estimators=300, max_depth=20,
                                                  min_samples_leaf=2,
                                                  random_state=42, n_jobs=-1),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                      max_depth=6, subsample=0.8,
                                                      random_state=42),
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000, random_state=42),
}

results = {}
best_name, best_model, best_auc = None, None, 0.0

for name, mdl in models.items():
    print(f"\nTraining {name} …")
    mdl.fit(X_res, y_res)
    y_pred  = mdl.predict(X_test_sc)
    y_proba = mdl.predict_proba(X_test_sc)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred).tolist()

    results[name] = {
        "accuracy":     round(acc, 4),
        "roc_auc":      round(auc, 4),
        "report":       report,
        "confusion_matrix": cm,
        "feature_importances": (
            mdl.feature_importances_.tolist()
            if hasattr(mdl, "feature_importances_") else
            np.abs(mdl.coef_[0]).tolist()
        ),
    }
    print(f"  Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc, best_name, best_model = auc, name, mdl

print(f"\nBest model: {best_name}  (AUC={best_auc:.4f})")

# ── Save artefacts ────────────────────────────────────────────────────────────
joblib.dump(best_model,     ART_DIR / "model.pkl")
joblib.dump(scaler,         ART_DIR / "scaler.pkl")
joblib.dump(label_encoders, ART_DIR / "label_encoders.pkl")

meta = {
    "best_model":    best_name,
    "feature_cols":  FEATURE_COLS,
    "cat_cols":      CAT_COLS,
    "results":       results,
    "class_names":   ["Not Placed", "Placed"],
    "label_maps": {
        col: {str(cls): int(i) for i, cls in enumerate(le.classes_)}
        for col, le in label_encoders.items()
    },
}
with open(ART_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nArtefacts saved to:", ART_DIR)
print("Done - training complete!")
