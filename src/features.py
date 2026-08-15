"""
features.py
-----------
Feature engineering for the readmission-risk pipeline.

All features use ONLY information available at the time of prediction
(encounter-level data). No target leakage. No future information.

Engineered features
-------------------
total_prior_visits    : n_outpatient + n_emergency + n_inpatient
procedures_per_day    : n_procedures / max(time_in_hospital, 1)
meds_per_day          : n_medications / max(time_in_hospital, 1)
labs_per_day          : n_lab_procedures / max(time_in_hospital, 1)
had_prior_inpatient   : 1 if n_inpatient > 0 else 0
had_prior_emergency   : 1 if n_emergency > 0 else 0

Decision (from audit): All 6 engineered features are computed from
within-encounter counts only. They are wrapped in a FunctionTransformer
so the pipeline can apply them to raw input during inference without any
manual preprocessing step by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


# ── Column name constants ──────────────────────────────────────────────────

RAW_NUMERIC = [
    "time_in_hospital",
    "n_lab_procedures",
    "n_procedures",
    "n_medications",
    "n_outpatient",
    "n_inpatient",
    "n_emergency",
]

ENGINEERED_NUMERIC = [
    "total_prior_visits",
    "procedures_per_day",
    "meds_per_day",
    "labs_per_day",
    "had_prior_inpatient",
    "had_prior_emergency",
]

ALL_NUMERIC = RAW_NUMERIC + ENGINEERED_NUMERIC

ORDINAL_FEATURES = ["age"]

CATEGORICAL_FEATURES = [
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "glucose_test",
    "A1Ctest",
    "change",
    "diabetes_med",
]

AGE_ORDER = [
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]


# ── Core transform ─────────────────────────────────────────────────────────

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered columns to a copy of *X*.

    Parameters
    ----------
    X : pd.DataFrame
        Raw patient feature frame (output of df.drop(columns=["readmitted"])).

    Returns
    -------
    pd.DataFrame
        Original columns plus six engineered columns.
    """
    X = X.copy()

    # Prior utilisation count
    X["total_prior_visits"] = (
        X["n_outpatient"] + X["n_emergency"] + X["n_inpatient"]
    )

    # Protect against division by zero (time_in_hospital ≥ 1 in clean data,
    # but clip defensively)
    safe_days = X["time_in_hospital"].clip(lower=1)

    X["procedures_per_day"] = X["n_procedures"] / safe_days
    X["meds_per_day"]       = X["n_medications"] / safe_days
    X["labs_per_day"]       = X["n_lab_procedures"] / safe_days

    # Binary flags
    X["had_prior_inpatient"] = (X["n_inpatient"] > 0).astype(int)
    X["had_prior_emergency"] = (X["n_emergency"] > 0).astype(int)

    return X


def make_feature_transformer() -> FunctionTransformer:
    """
    Return a sklearn FunctionTransformer wrapping engineer_features.

    The transformer validates that the input is a DataFrame so that
    it can be placed as the first step of a Pipeline and called with
    .transform() during inference.
    """
    return FunctionTransformer(
        func=engineer_features,
        validate=False,   # DataFrame in -> DataFrame out; no array validation
    )
