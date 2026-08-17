"""
features.py
-----------
Feature engineering for the readmission-risk pipeline.

All features use ONLY information available at the time of prediction
(encounter-level data). No target leakage. No future information.

Engineered features
-------------------
Original 6:
  total_prior_visits    : n_outpatient + n_emergency + n_inpatient
  procedures_per_day    : n_procedures / max(time_in_hospital, 1)
  meds_per_day          : n_medications / max(time_in_hospital, 1)
  labs_per_day          : n_lab_procedures / max(time_in_hospital, 1)
  had_prior_inpatient   : 1 if n_inpatient > 0 else 0
  had_prior_emergency   : 1 if n_emergency > 0 else 0

New 8 (v2):
  lab_to_med_ratio      : n_lab_procedures / (n_medications + 1)
  utilisation_intensity : weighted prior visits adjusted by age bracket
  is_high_utiliser      : 1 if total_prior_visits >= 3 else 0
  meds_x_inpatient      : n_medications * n_inpatient (interaction)
  long_stay_flag        : 1 if time_in_hospital >= 7 else 0
  no_test_flag          : 1 if glucose_test=='no' AND A1Ctest=='no'
  diag_complexity       : count of non-'Other' diagnoses (0-3)
  specialty_x_inpatient : (medical_specialty=='Missing') * n_inpatient
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
    # Original 6
    "total_prior_visits",
    "procedures_per_day",
    "meds_per_day",
    "labs_per_day",
    "had_prior_inpatient",
    "had_prior_emergency",
    # New 8 (v2)
    "lab_to_med_ratio",
    "utilisation_intensity",
    "is_high_utiliser",
    "meds_x_inpatient",
    "long_stay_flag",
    "no_test_flag",
    "diag_complexity",
    "specialty_x_inpatient",
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

# Map age bracket to ordinal index for numeric operations
AGE_TO_IDX = {a: i for i, a in enumerate(AGE_ORDER)}


# ── Core transform ─────────────────────────────────────────────────────────

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add 14 engineered columns to a copy of *X* (6 original + 8 new).

    Parameters
    ----------
    X : pd.DataFrame
        Raw patient feature frame (output of df.drop(columns=["readmitted"])).

    Returns
    -------
    pd.DataFrame
        Original columns plus 14 engineered columns.
    """
    X = X.copy()

    # ── Original 6 features ────────────────────────────────────────────────
    X["total_prior_visits"] = (
        X["n_outpatient"] + X["n_emergency"] + X["n_inpatient"]
    )

    safe_days = X["time_in_hospital"].clip(lower=1)

    X["procedures_per_day"] = X["n_procedures"]     / safe_days
    X["meds_per_day"]       = X["n_medications"]    / safe_days
    X["labs_per_day"]       = X["n_lab_procedures"] / safe_days

    X["had_prior_inpatient"] = (X["n_inpatient"] > 0).astype(int)
    X["had_prior_emergency"] = (X["n_emergency"] > 0).astype(int)

    # ── New 8 features (v2) ────────────────────────────────────────────────

    # 1. Lab-to-medication ratio: high labs vs meds = complex / undertreated
    X["lab_to_med_ratio"] = X["n_lab_procedures"] / (X["n_medications"] + 1)

    # 2. Utilisation intensity: severity-adjusted prior use
    age_idx = X["age"].map(AGE_TO_IDX).fillna(2).astype(float)
    X["utilisation_intensity"] = (
        X["n_inpatient"] * 3 + X["n_emergency"] * 2 + X["n_outpatient"]
    ) / (age_idx + 1)

    # 3. High utiliser flag: >= 3 prior visits of any type
    X["is_high_utiliser"] = (X["total_prior_visits"] >= 3).astype(int)

    # 4. Meds × inpatient: complex medication burden + inpatient history
    X["meds_x_inpatient"] = X["n_medications"] * X["n_inpatient"]

    # 5. Long stay flag: extended stays (>= 7 days) predict readmission
    X["long_stay_flag"] = (X["time_in_hospital"] >= 7).astype(int)

    # 6. No-test flag: diabetic patients with no glucose AND no A1C test
    no_glucose = (X["glucose_test"] == "no").astype(int)
    no_a1c     = (X["A1Ctest"]      == "no").astype(int)
    X["no_test_flag"] = no_glucose * no_a1c

    # 7. Diagnosis complexity: count of non-'Other' diagnoses (multi-morbidity)
    X["diag_complexity"] = (
        (X["diag_1"] != "Other").astype(int)
        + (X["diag_2"] != "Other").astype(int)
        + (X["diag_3"] != "Other").astype(int)
    )

    # 8. Missing specialty × inpatient: ED admissions with prior inpatient history
    X["specialty_x_inpatient"] = (
        (X["medical_specialty"] == "Missing").astype(int) * X["n_inpatient"]
    )

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
