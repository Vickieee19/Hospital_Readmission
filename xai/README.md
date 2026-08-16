# XAI Module

This folder holds the explainability layer for prediction explanations without changing the original ML pipeline structure.

## Purpose

- generate patient-level explanations from model predictions
- highlight the features that increased or decreased predicted readmission risk
- return structured JSON-friendly payloads that can be consumed by the backend or frontend later

## Entry point

```python
from xai import explain_prediction

result = explain_prediction(model, patient_df, top_n=5)
```

## Output format

```python
{
  "risk_score": 0.7134,
  "prediction": "high_risk",
  "base_value": 0.2148,
  "feature_contributions": [
    {"feature": "n_inpatient", "shap_value": 0.1867, "direction": "increases risk"}
  ],
  "top_increasing_risk": [
    {"feature": "n_inpatient", "shap_value": 0.1867, "direction": "increases risk"}
  ],
  "top_decreasing_risk": [
    {"feature": "glucose_test_no", "shap_value": -0.0941, "direction": "decreases risk"}
  ],
  "disclaimer": "These feature contributions explain..."
}
```

## Notes

- This stays separate from the existing model/training code in the root project.
- It relies on the fitted pipeline and SHAP TreeExplainer logic.
- It is designed to be later integrated by the backend API and frontend UI.
