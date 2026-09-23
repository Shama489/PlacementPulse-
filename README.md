# Student Placement Prediction — Project README

## 🎓 Student Placement Prediction System

A full-stack ML application that predicts student placement outcomes using:
- **Backend**: Flask REST API with trained ML models
- **Frontend**: Streamlit interactive dashboard (pure Python UI)
- **Dataset**: 100,000 student records with 25 features

---

## 📁 Project Structure

```
placement_predictor/
├── data/
│   └── dataset.csv                    ← 100k student dataset
├── model/
│   ├── train.py                       ← Model training script
│   └── artefacts/
│       ├── model.pkl                  ← Best trained model
│       ├── scaler.pkl                 ← Feature scaler
│       ├── label_encoders.pkl         ← Categorical encoders
│       └── meta.json                  ← Model metadata & metrics
├── backend/
│   └── app.py                         ← Flask REST API (port 5000)
├── frontend/
│   ├── app.py                         ← Streamlit entry point
│   └── pages/
│       ├── home.py                    ← Home page with key metrics
│       ├── eda.py                     ← EDA Dashboard
│       ├── predict.py                 ← Single prediction form
│       ├── batch.py                   ← Batch CSV prediction
│       └── performance.py             ← Model performance metrics
└── requirements.txt
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r placement_predictor/requirements.txt
```

### Step 2 — Train the model (run once)
```bash
python placement_predictor/model/train.py
```

### Step 3 — Start the Flask API backend
```bash
python placement_predictor/backend/app.py
```
API runs at: `http://localhost:5000`

### Step 4 — Launch the Streamlit frontend (new terminal)
```bash
streamlit run placement_predictor/frontend/app.py
```
App opens at: `http://localhost:8501`

---

## 🌐 API Endpoints

| Method | Endpoint          | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Health check                         |
| GET    | `/stats`          | Dataset & model statistics           |
| POST   | `/predict`        | Single student prediction (JSON)     |
| POST   | `/predict-batch`  | Batch CSV upload → predictions CSV   |

### Example — Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"cgpa": 8.2, "gender": "Female", "branch": "CSE", "college_tier": "Tier 1",
       "internships_count": 2, "projects_count": 4, "certifications_count": 3,
       "coding_skill_score": 85, "aptitude_score": 78, "communication_skill_score": 82,
       "logical_reasoning_score": 80, "hackathons_participated": 2, "github_repos": 5,
       "linkedin_connections": 400, "mock_interview_score": 80, "attendance_percentage": 88,
       "backlogs": 0, "extracurricular_score": 60, "leadership_score": 65,
       "volunteer_experience": "Yes", "sleep_hours": 7, "study_hours_per_day": 5, "age": 22}'
```

---

## 🤖 ML Models

| Model               | Role           |
|---------------------|----------------|
| Random Forest       | Ensemble (trees) |
| Gradient Boosting   | Boosted trees  |
| Logistic Regression | Linear baseline |

Best model selected automatically by ROC-AUC score.  
SMOTE oversampling used to handle class imbalance.

---

## 🎨 Frontend Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, key metrics, placement rates by branch/tier |
| 📊 EDA Dashboard | Interactive charts: distributions, heatmaps, skill box plots |
| 🔮 Single Prediction | Form-based input → gauge chart + probability score |
| 📁 Batch Prediction | Upload CSV → bulk predictions + download results |
| 📈 Model Performance | Confusion matrix, ROC curve, feature importances |
