"""
explainability.py
-----------------
SHAP-based explanations for the readmission-risk pipeline.

Architecture note
-----------------
The outer sklearn Pipeline CANNOT be passed directly to
shap.TreeExplainer — it only understands tree models, not pipelines.

Correct approach:
  1. Extract the fitted preprocessor from the pipeline
  2. Extract the fitted XGBClassifier from the pipeline
  3. Transform raw input through the preprocessor
  4. Apply TreeExplainer to the XGBClassifier
  5. Map transformed feature names back to meaningful source names
  6. Aggregate one-hot encoded features back to their source column
     (optional — reported both ways)

Causal disclaimer
-----------------
SHAP values describe the contribution of each feature to the model's
predicted score relative to the base rate. They do NOT prove causality.
Never write "X caused the readmission." Always write "X contributed to
a higher/lower model-predicted risk score."
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


# ── Extractor helpers ──────────────────────────────────────────────────────

def extract_components(pipeline) -> tuple:
    """
    Extract (feature_transformer_or_None, preprocessor, xgb_classifier)
    from a fitted pipeline or ensemble.

    Handles:
      - 3-step pipeline: feature_eng -> preprocessor -> classifier
      - 2-step pipeline: preprocessor -> classifier  (no feature engineering)
      - Calibration wrapper: unwraps via .estimator
      - VotingClassifier ensemble: extracts first pipeline (XGBoost sub-pipeline)
    """
    actual_pipeline = pipeline
    if hasattr(actual_pipeline, "estimator"):
        actual_pipeline = actual_pipeline.estimator

    if hasattr(actual_pipeline, "estimators_"):
        # For VotingClassifier or stacking ensemble, use the XGBoost component
        actual_pipeline = actual_pipeline.estimators_[0]

    steps = dict(actual_pipeline.steps)

    feature_eng  = steps.get("feature_eng", None)   # may be absent
    preprocessor = steps["preprocessor"]
    classifier   = steps["classifier"]

    return feature_eng, preprocessor, classifier


def get_feature_names(preprocessor) -> list[str]:
    """
    Extract human-readable feature names from the fitted ColumnTransformer.

    Returns a list aligned with the preprocessor's output columns.
    """
    return list(preprocessor.get_feature_names_out())


# ── SHAP explainer ─────────────────────────────────────────────────────────

def build_explainer(pipeline: Pipeline) -> tuple:
    """
    Build a shap.TreeExplainer for the XGBClassifier inside the pipeline.

    Returns
    -------
    (explainer, feature_names)
    """
    _, preprocessor, classifier = extract_components(pipeline)
    feature_names = get_feature_names(preprocessor)

    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation="interventional",
        model_output="raw",   # log-odds; convert to probability space in UI
    )

    return explainer, feature_names


def get_shap_values(
    pipeline: Pipeline,
    X_raw: pd.DataFrame,
) -> tuple[np.ndarray, float, list[str]]:
    """
    Compute SHAP values for X_raw.

    Parameters
    ----------
    pipeline : fitted Pipeline (or CalibratedClassifierCV wrapping one)
    X_raw    : raw patient DataFrame (1 or more rows)

    Returns
    -------
    (shap_values, base_value, feature_names)
        shap_values  : array shape (n_samples, n_features)
        base_value   : scalar base value in log-odds space
        feature_names: list of transformed feature names
    """
    feat_eng, preprocessor, classifier = extract_components(pipeline)

    # Step 1: Apply feature engineering (if the pipeline has it)
    if feat_eng is not None:
        X_engineered = feat_eng.transform(X_raw)
    else:
        X_engineered = X_raw

    # Step 2: Apply preprocessing
    X_transformed = preprocessor.transform(X_engineered)

    feature_names = get_feature_names(preprocessor)

    # Step 3: Compute SHAP
    explainer   = shap.TreeExplainer(classifier)
    explanation = explainer(X_transformed)

    # For binary classification, shap_values can be shape (n, f) or (n, f, 2)
    if explanation.values.ndim == 3:
        shap_vals = explanation.values[:, :, 1]   # positive class
    else:
        shap_vals = explanation.values

    base_val = float(explainer.expected_value)
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[-1])

    return shap_vals, base_val, feature_names


# ── Individual explanation ─────────────────────────────────────────────────

def explain_patient(
    pipeline: Pipeline,
    patient_row: pd.DataFrame,
    top_n: int = 8,
) -> dict:
    """
    Generate a structured explanation for a single patient prediction.

    Parameters
    ----------
    pipeline    : fitted Pipeline
    patient_row : single-row DataFrame with raw feature values
    top_n       : number of top contributors to return

    Returns
    -------
    dict with keys:
        raw_risk_score          float  (XGBoost predict_proba output)
        base_value              float  (SHAP log-odds base value)
        shap_values             list   (all feature SHAP values)
        feature_names           list
        top_increasing_risk     list of (feature_name, shap_val) dicts
        top_decreasing_risk     list of (feature_name, shap_val) dicts
        disclaimer              str
    """
    assert len(patient_row) == 1, "explain_patient expects exactly one row"

    # Raw score
    actual_pipeline = pipeline
    if hasattr(pipeline, "calibrated_classifiers_"):
        actual_pipeline = pipeline.estimator
    raw_score = float(actual_pipeline.predict_proba(patient_row)[0, 1])

    # SHAP
    shap_vals, base_val, feat_names = get_shap_values(pipeline, patient_row)
    sv = shap_vals[0]   # shape (n_features,)

    # Rank by absolute magnitude
    ranked_idx  = np.argsort(np.abs(sv))[::-1]
    pos_features = [
        {"feature": feat_names[i], "shap_value": round(float(sv[i]), 4)}
        for i in ranked_idx if sv[i] > 0
    ][:top_n]
    neg_features = [
        {"feature": feat_names[i], "shap_value": round(float(sv[i]), 4)}
        for i in ranked_idx if sv[i] < 0
    ][:top_n]

    return {
        "raw_risk_score":       round(raw_score, 4),
        "base_value":           round(base_val,  4),
        "shap_values":          [round(float(v), 4) for v in sv],
        "feature_names":        feat_names,
        "top_increasing_risk":  pos_features,
        "top_decreasing_risk":  neg_features,
        "disclaimer": (
            "SHAP values describe each feature's contribution to the model's "
            "predicted risk score relative to the average prediction. "
            "They do NOT establish causality."
        ),
    }


# ── Global importance ──────────────────────────────────────────────────────

def compute_global_importance(
    pipeline: Pipeline,
    X_raw: pd.DataFrame,
    top_n: int = 20,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Compute mean |SHAP| over a sample of rows for global feature importance.

    Parameters
    ----------
    pipeline   : fitted Pipeline
    X_raw      : sample of raw patient records
    top_n      : number of top features to display / return
    output_dir : if provided, save beeswarm/bar chart PNG here

    Returns
    -------
    pd.DataFrame  columns: feature, mean_abs_shap  (sorted descending)
    """
    shap_vals, _, feat_names = get_shap_values(pipeline, X_raw)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            importance_df["feature"][::-1],
            importance_df["mean_abs_shap"][::-1],
            color="steelblue",
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Global Feature Importance (top {top_n}) — SHAP")
        fig.tight_layout()
        fig.savefig(output_dir / "shap_global_importance.png", dpi=150)
        plt.close(fig)
        print(f"[SHAP] Global importance chart saved to {output_dir}")

    return importance_df
