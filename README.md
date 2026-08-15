# CareGrid — Hospital Readmission Prediction

A clinical decision-support system that predicts 30-day hospital readmission risk using an XGBoost classifier trained on 25,000 patient encounters.

---

## Project Structure

```
Hospital_Readmission_Pred/          ← project root (git repo)
│
├── dataset/                        ← dataset folder (excluded from git)
│   ├── .gitkeep
│   └── hospital_readmissions.csv
│
├── backend/                        ← Python ML backend
│   ├── models/                     ← serialised model checkpoints (excluded from git)
│   │   ├── .gitkeep
│   │   └── readmission_model.pkl
│   ├── scripts/                    ← training & analysis scripts
│   │   ├── main.py                 ← model training pipeline
│   │   ├── evalutation.py          ← ROC-AUC & classification metrics
│   │   ├── threshold_analysis.py   ← Precision/Recall/F1 across thresholds
│   │   ├── gains_analysis.py       ← cumulative gains chart
│   │   └── top_risk.py             ← top-N% risk capture analysis
│   └── requirements.txt            ← pinned Python dependencies
│
├── frontend/                       ← React frontend (to be implemented)
│   └── .gitkeep
│
├── .gitignore
└── README.md
```

---

## Model Summary

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Features | 16 (7 numeric, 1 ordinal, 8 categorical) |
| Target | `readmitted` (yes / no → 1 / 0) |
| Train/Test Split | 80% / 20%, stratified |
| ROC-AUC | 0.6571 |
| Operating Threshold | 0.35 |
| Recall @ 0.35 | 0.866 |
| Precision @ 0.35 | 0.519 |
| Top-30% Capture | 40.3% of actual readmissions |

---

## Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py

# Run evaluation
python evalutation.py

# Threshold analysis
python threshold_analysis.py

# Gains chart
python gains_analysis.py
```

> **Important:** `readmission_model.pkl` was serialised with `scikit-learn 1.9.0`.  
> Do **not** upgrade or downgrade `scikit-learn` or `xgboost` without retraining and re-serialising the model.

---

## Frontend Setup

> Coming soon — React application with FastAPI integration.

---

## Planned Architecture

```
React Frontend  ──POST /predict──►  FastAPI Backend  ──►  XGBoost Pipeline
                ◄── JSON response ──                  ◄──  SHAP Explainer
```

**API Response shape:**
```json
{
  "risk_score": 0.61,
  "risk_level": "High",
  "explanations": [
    { "feature": "n_inpatient", "direction": "increases risk", "importance": 0.18 }
  ],
  "disclaimer": "Risk scores are model predictions, not clinical diagnoses."
}
```

---

## Audit Status

See [`audit_report.md`](.gemini/brain/audit_report.md) for the full pre-deployment audit.  
**Current status: Research-Demo Ready (B)** — API and frontend integration pending.
