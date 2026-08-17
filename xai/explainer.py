from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


def _unwrap_model(model: Any) -> Any:
    """Return the underlying estimator from a wrapped sklearn model or ensemble."""
    actual = model

    while hasattr(actual, "estimator"):
        actual = actual.estimator

    if hasattr(actual, "estimators_") and len(actual.estimators_) > 0:
        first_estimator = actual.estimators_[0]
        if hasattr(first_estimator, "steps") or hasattr(first_estimator, "named_steps"):
            actual = first_estimator

    return actual


def _get_feature_names(model: Any) -> list[str]:
    """Return the list of transformed feature names for a pipeline."""
    model = _unwrap_model(model)
    if hasattr(model, "steps"):
        preprocessor = dict(model.steps).get("preprocessor")
        if preprocessor is not None:
            return list(preprocessor.get_feature_names_out())
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return []


def _compute_shap_values(model: Any, patient_df: pd.DataFrame) -> tuple[np.ndarray, float, list[str]]:
    """Compute SHAP values using the fitted classifier inside the pipeline."""
    raw_model = _unwrap_model(model)

    if hasattr(raw_model, "steps"):
        steps = dict(raw_model.steps)
        feature_eng = steps.get("feature_eng")
        preprocessor = steps["preprocessor"]
        classifier = steps["classifier"]

        X_input = feature_eng.transform(patient_df) if feature_eng is not None else patient_df
        X_transformed = preprocessor.transform(X_input)
        feature_names = list(preprocessor.get_feature_names_out())
        explainer = shap.TreeExplainer(classifier)
        explanation = explainer(X_transformed)
        values = explanation.values[:, :, 1] if explanation.values.ndim == 3 else explanation.values
        base_value = float(explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[-1])
        return values[0], base_value, feature_names

    if hasattr(raw_model, "predict_proba") and hasattr(raw_model, "feature_names_in_"):
        explainer = shap.Explainer(raw_model, patient_df)
        explanation = explainer(patient_df)
        values = explanation.values
        if values.ndim == 3:
            values = values[:, :, 1]
        return values[0], float(explanation.base_values[0]), list(raw_model.feature_names_in_)

    raise TypeError("Model must be a fitted sklearn pipeline or wrapped estimator with SHAP-compatible tree explainer.")


def _safe_prediction(model: Any, patient_df: pd.DataFrame) -> float:
    """Return class-1 probability from the model."""
    actual = _unwrap_model(model)
    if hasattr(actual, "predict_proba"):
        return float(actual.predict_proba(patient_df)[0, 1])
    if hasattr(actual, "predict"):
        return float(actual.predict(patient_df)[0])
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(patient_df)[0, 1])
    if hasattr(model, "predict"):
        return float(model.predict(patient_df)[0])
    raise TypeError("Model does not expose a prediction interface.")


def explain_prediction(model: Any, patient_row: pd.DataFrame, top_n: int = 5, threshold: float = 0.5) -> dict[str, Any]:
    """Return a structured explanation for a single patient prediction.

    Parameters
    ----------
    model : fitted sklearn pipeline or wrapped estimator
    patient_row : single-row pandas DataFrame
    top_n : number of top positive and negative contributors to return
    threshold : decision boundary for "high_risk" vs "low_risk"
    """
    if patient_row is None or len(patient_row) == 0:
        raise ValueError("patient_row must contain at least one sample row.")
    if len(patient_row) != 1:
        raise ValueError("explain_prediction expects exactly one patient row.")

    risk_score = _safe_prediction(model, patient_row)
    prediction = "high_risk" if risk_score >= threshold else "low_risk"

    shap_values, base_value, feature_names = _compute_shap_values(model, patient_row)

    ranking = sorted(
        enumerate(feature_names),
        key=lambda item: abs(float(shap_values[item[0]])),
        reverse=True,
    )

    top_increasing = []
    top_decreasing = []

    for idx, feature_name in ranking:
        value = float(shap_values[idx])
        item = {
            "feature": feature_name,
            "shap_value": round(value, 6),
            "direction": "increases risk" if value > 0 else "decreases risk",
        }

        if value > 0 and len(top_increasing) < max(top_n, 0):
            top_increasing.append(item)
        elif value < 0 and len(top_decreasing) < max(top_n, 0):
            top_decreasing.append(item)

    feature_contributions = [
        {
            "feature": feature_name,
            "shap_value": round(float(shap_values[idx]), 6),
            "direction": "increases risk" if float(shap_values[idx]) > 0 else "decreases risk",
        }
        for idx, feature_name in ranking
    ]

    return {
        "risk_score": round(float(risk_score), 6),
        "prediction": prediction,
        "base_value": round(float(base_value), 6),
        "feature_contributions": feature_contributions[:max(top_n, 0)],
        "top_increasing_risk": top_increasing,
        "top_decreasing_risk": top_decreasing,
        "disclaimer": (
            "These feature contributions explain the model's predicted risk for this patient "
            "relative to the model's baseline. They are attribution values, not proof of causation."
        ),
    }


get_prediction_explanation = explain_prediction
