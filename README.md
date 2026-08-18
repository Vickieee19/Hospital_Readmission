# CareGrid — Hospital Readmission Prediction & Clinical Decision Support

A clinical decision-support system that predicts 30-day hospital readmission risk using an optimized machine learning ensemble (XGBoost + LightGBM) and provides grounded medical guideline severity analysis via RAG.

> 📖 **Full Technical Architecture & Workflow Document**: For an in-depth, step-by-step breakdown of the ML pipeline, feature engineering, threshold calibration, RAG engine, and UI architecture, see **[`PROJECT_WORKFLOW.md`](file:///c:/Users/Asus/OneDrive/Documents/Desktop/projects/Hospital_Readmission_Pred/PROJECT_WORKFLOW.md)**.

---

## Project Structure

```
Hospital_Readmission_Pred/          ← project root (git repo)
│
├── backend/                        ← Training entrypoint & orchestration
│   ├── __init__.py
│   └── train.py                    ← Master training pipeline (Phases 3–15)
│
├── src/                            ← Core reusable ML modules
│   ├── __init__.py
│   ├── features.py                 ← Feature transformer (FunctionTransformer)
│   ├── preprocessing.py            ← ColumnTransformer (impute, scale, OHE, ordinal)
│   ├── model.py                    ← Pipeline builder, 5-fold CV, RandomizedSearchCV
│   ├── calibration.py              ← Brier score assessment & Platt scaling wrapper
│   ├── evaluation.py               ← ROC-AUC, PR-AUC, F2 thresholding, gains & lift
│   └── explainability.py          ← SHAP TreeExplainer & feature attribution
│
├── models/                         ← Serialized models & metadata (excluded from git)
│   ├── .gitkeep
│   ├── readmission_model_baseline.pkl  ← Preserved original baseline model
│   ├── readmission_model_final.pkl     ← Optimized deployable model pipeline
│   ├── model_metadata.json             ← Pinned training metadata & metrics
│   └── charts/                         ← Automated evaluation charts
│       ├── calibration_curve.png
│       ├── cumulative_gains.png
│       ├── lift_chart.png
│       ├── roc_curve.png
│       └── shap_global_importance.png
│
├── dataset/                        ← Dataset folder (excluded from git)
│   ├── .gitkeep
│   └── hospital_readmissions.csv   ← 25,000 patient encounters
│
├── tests/                          ← Automated unit & integration tests
│   ├── __init__.py
│   └── test_model.py               ← 15 test cases (edge cases, ranges, SHAP, determinism)
│
├── frontend/                       ← React frontend (to be implemented)
│   └── .gitkeep
│
├── conftest.py                     ← Pytest configuration
├── requirements.txt                ← Pinned Python dependencies
├── .gitignore
└── README.md
```

---

## Model Performance Summary

| Metric | Baseline Model | Optimized Final Model |
|---|---|---|
| **Algorithm** | XGBoost Classifier | XGBoost Classifier (Tuned) |
| **Features** | 16 (7 numeric, 1 ordinal, 8 categorical) | 16 (7 numeric, 1 ordinal, 8 categorical) |
| **Target** | `readmitted` (yes / no → 1 / 0) | `readmitted` (yes / no → 1 / 0) |
| **Validation Strategy** | Single 80/20 train/test split | 5-Fold Stratified CV + Holdout Test (20%) |
| **ROC-AUC** | 0.6571 | **0.6581** |
| **PR-AUC** | Not reported | **0.6282** |
| **Operating Threshold** | 0.35 | **0.3496** (F2-selected on validation) |
| **Recall @ Threshold** | 0.866 | **0.8762** |
| **Precision @ Threshold** | 0.519 | **0.5156** |
| **F1 @ Threshold** | 0.649 | **0.6492** |
| **F2 @ Threshold** | Not reported | **0.7687** |
| **Top-30% Capture** | 40.3% | **40.4%** |
| **Top-30% Lift** | ~1.34x | **1.346x** |

---

## Quickstart & Execution

```bash
# 1. Activate virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Run the master training pipeline
python backend/train.py

# 3. Run all automated test suites
python -m pytest tests/test_model.py -v
```

> **Important:** `readmission_model_final.pkl` was serialized with `scikit-learn 1.9.0` and `xgboost 3.4.0`.  
> Do **not** upgrade or downgrade core dependencies without retraining and re-verifying the test suite.

---

## Planned Architecture

```
React Frontend  ──POST /predict──►  FastAPI Backend  ──►  XGBoost Pipeline (models/readmission_model_final.pkl)
                ◄── JSON response ──                  ◄──  SHAP Explainer (src/explainability.py)
```

**API Response shape:**
```json
{
  "risk_score": 0.4526,
  "risk_level": "Medium",
  "explanations": [
    { "feature": "n_inpatient", "direction": "decreases risk", "importance": -0.2167 },
    { "feature": "age", "direction": "increases risk", "importance": 0.0964 }
  ],
  "disclaimer": "Risk scores are model predictions for clinical prioritization, not diagnostic conclusions."
}
```

---

## Current Status

**Status: Model Ready for API/UI Integration (C)** — All ML validation, thresholding, calibration assessment, explainability, tests, and clean architecture completed.
