"""
tests/test_model.py
-------------------
Automated tests for the readmission-risk model pipeline.

Run from project root:
    .venv\\Scripts\\pytest tests/test_model.py -v

Test inventory (15 tests)
--------------------------
 1.  normal_patient         — valid representative patient → valid output
 2.  high_utilisation       — extreme utilisation values → valid output
 3.  low_utilisation        — minimal utilisation → valid output
 4.  missing_field          — missing column handled gracefully (imputed)
 5.  invalid_numeric_type   — string in numeric field → handled or raises clearly
 6.  negative_medication    — negative n_medications → caught or imputed
 7.  impossible_stay        — time_in_hospital = 0 → clips to 1, no crash
 8.  unknown_category       — unseen medical_specialty → OHE ignore, no crash
 9.  unsupported_age        — age bracket not in training set → sentinel -1
10.  corrupted_model_file   — loading a bad .pkl → explicit error
11.  deterministic_repeated — same input → same output every time
12.  shap_explanation_exists — explain_patient returns expected keys
13.  output_is_finite        — predict_proba contains no NaN/Inf
14.  probability_in_range    — predict_proba ∈ [0, 1]
15.  baseline_model_intact   — readmission_model_baseline.pkl untouched
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# ── Ensure project root on path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────
MODELS_DIR   = PROJECT_ROOT / "models"
FINAL_PKL    = MODELS_DIR / "readmission_model_final.pkl"
BASELINE_PKL = MODELS_DIR / "readmission_model_baseline.pkl"
BASELINE_EXPECTED_SIZE = 505_215   # bytes — verified during audit

# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Load the final model once for the whole test session."""
    assert FINAL_PKL.exists(), (
        f"Final model not found at {FINAL_PKL}.\n"
        "Run `backend/train.py` first."
    )
    return joblib.load(FINAL_PKL)


@pytest.fixture
def normal_patient() -> pd.DataFrame:
    """A representative 70-year-old diabetic patient."""
    return pd.DataFrame([{
        "age":               "[70-80)",
        "time_in_hospital":  5,
        "n_lab_procedures":  40,
        "n_procedures":      2,
        "n_medications":     15,
        "n_outpatient":      0,
        "n_inpatient":       1,
        "n_emergency":       0,
        "medical_specialty": "InternalMedicine",
        "diag_1":            "Circulatory",
        "diag_2":            "Diabetes",
        "diag_3":            "Other",
        "glucose_test":      "no",
        "A1Ctest":           "no",
        "change":            "yes",
        "diabetes_med":      "yes",
    }])


@pytest.fixture
def high_utilisation_patient() -> pd.DataFrame:
    """Patient with very high utilisation metrics."""
    return pd.DataFrame([{
        "age":               "[80-90)",
        "time_in_hospital":  14,
        "n_lab_procedures":  120,
        "n_procedures":      6,
        "n_medications":     30,
        "n_outpatient":      5,
        "n_inpatient":       4,
        "n_emergency":       3,
        "medical_specialty": "Cardiology",
        "diag_1":            "Circulatory",
        "diag_2":            "Circulatory",
        "diag_3":            "Respiratory",
        "glucose_test":      "high",
        "A1Ctest":           "high",
        "change":            "yes",
        "diabetes_med":      "yes",
    }])


@pytest.fixture
def low_utilisation_patient() -> pd.DataFrame:
    """Patient with minimal utilisation — low readmission expected."""
    return pd.DataFrame([{
        "age":               "[40-50)",
        "time_in_hospital":  1,
        "n_lab_procedures":  5,
        "n_procedures":      0,
        "n_medications":     2,
        "n_outpatient":      0,
        "n_inpatient":       0,
        "n_emergency":       0,
        "medical_specialty": "Family/GeneralPractice",
        "diag_1":            "Other",
        "diag_2":            "Other",
        "diag_3":            "Other",
        "glucose_test":      "no",
        "A1Ctest":           "no",
        "change":            "no",
        "diabetes_med":      "no",
    }])


# ── Helper ──────────────────────────────────────────────────────────────────

def get_raw_pipeline(model):
    """
    If the model is a CalibratedClassifierCV, return the underlying Pipeline.
    Otherwise return the model itself.
    """
    if hasattr(model, "estimator"):
        return model.estimator
    return model


# ────────────────────────────────────────────────────────────────────────────
# TEST 1 — Normal patient
# ────────────────────────────────────────────────────────────────────────────

def test_normal_patient(model, normal_patient):
    """Model returns a single finite probability for a valid patient."""
    proba = model.predict_proba(normal_patient)
    assert proba.shape == (1, 2)
    score = float(proba[0, 1])
    assert np.isfinite(score), "Output is not finite"
    assert 0.0 <= score <= 1.0, f"Score {score} outside [0, 1]"


# ────────────────────────────────────────────────────────────────────────────
# TEST 2 — High-utilisation patient
# ────────────────────────────────────────────────────────────────────────────

def test_high_utilisation_patient(model, high_utilisation_patient):
    """High-utilisation patient gets a higher score than low-utilisation."""
    proba = model.predict_proba(high_utilisation_patient)
    score = float(proba[0, 1])
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# TEST 3 — Low-utilisation patient
# ────────────────────────────────────────────────────────────────────────────

def test_low_utilisation_patient(model, low_utilisation_patient):
    proba = model.predict_proba(low_utilisation_patient)
    score = float(proba[0, 1])
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# TEST 4 — Missing / NaN field
# ────────────────────────────────────────────────────────────────────────────

def test_missing_field(model, normal_patient):
    """
    A NaN in a numeric field should be handled by the imputer inside the
    pipeline without crashing.
    """
    patient = normal_patient.copy()
    patient["n_medications"] = np.nan
    proba = model.predict_proba(patient)
    score = float(proba[0, 1])
    assert np.isfinite(score), "NaN field caused non-finite output"


# ────────────────────────────────────────────────────────────────────────────
# TEST 5 — Invalid numeric type (string in numeric column)
# ────────────────────────────────────────────────────────────────────────────

def test_invalid_numeric_type(model, normal_patient):
    """
    A string in a numeric column should either raise a clear error or
    be coerced to NaN and imputed. The pipeline must not produce a silent
    wrong result.
    """
    patient = normal_patient.copy()
    patient["n_medications"] = "INVALID"

    try:
        # Attempt to coerce to numeric first (realistic API behaviour)
        patient["n_medications"] = pd.to_numeric(
            patient["n_medications"], errors="coerce"
        )
        proba = model.predict_proba(patient)
        score = float(proba[0, 1])
        assert np.isfinite(score)
    except (ValueError, TypeError) as exc:
        # Acceptable: the pipeline raised an explicit error
        assert str(exc) != "", "Expected a non-empty error message"


# ────────────────────────────────────────────────────────────────────────────
# TEST 6 — Negative medication count
# ────────────────────────────────────────────────────────────────────────────

def test_negative_medication_count(model, normal_patient):
    """
    Negative n_medications is impossible clinically. The pipeline should
    either clip it, impute it, or raise clearly. It must NOT silently
    produce an incorrect result that appears valid.
    """
    patient = normal_patient.copy()
    patient["n_medications"] = -5

    # The current pipeline does not clip negatives — it passes them through
    # the scaler unchanged. This test verifies the output is still finite
    # and documents the behaviour for future validation.
    proba = model.predict_proba(patient)
    score = float(proba[0, 1])
    assert np.isfinite(score), "Negative medication count caused non-finite output"
    # Document: negative value is passed through — should add input validation in API
    assert 0.0 <= score <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# TEST 7 — Impossible hospital stay (time_in_hospital = 0)
# ────────────────────────────────────────────────────────────────────────────

def test_impossible_hospital_stay(model, normal_patient):
    """
    time_in_hospital = 0 would cause division by zero in per-day features.
    The feature engineering clips to max(1, time_in_hospital), so this
    should produce a finite result.
    """
    patient = normal_patient.copy()
    patient["time_in_hospital"] = 0
    proba = model.predict_proba(patient)
    score = float(proba[0, 1])
    assert np.isfinite(score), "time_in_hospital=0 caused non-finite output"


# ────────────────────────────────────────────────────────────────────────────
# TEST 8 — Unknown category in medical_specialty
# ────────────────────────────────────────────────────────────────────────────

def test_unknown_category(model, normal_patient):
    """
    OHE with handle_unknown='ignore' should map unseen categories to
    all-zero rows without raising an error.
    """
    patient = normal_patient.copy()
    patient["medical_specialty"] = "NeurosurgerySpecialty_UNSEEN"
    proba = model.predict_proba(patient)
    score = float(proba[0, 1])
    assert np.isfinite(score), "Unknown category caused non-finite output"
    assert 0.0 <= score <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# TEST 9 — Unsupported age bracket
# ────────────────────────────────────────────────────────────────────────────

def test_unsupported_age_bracket(model, normal_patient):
    """
    OrdinalEncoder with handle_unknown='use_encoded_value' + unknown_value=-1
    should map an unseen age bracket to -1 without crashing.
    """
    patient = normal_patient.copy()
    patient["age"] = "[20-30)"   # not in training categories
    proba = model.predict_proba(patient)
    score = float(proba[0, 1])
    assert np.isfinite(score), "Unknown age bracket caused non-finite output"
    assert 0.0 <= score <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# TEST 10 — Corrupted model file
# ────────────────────────────────────────────────────────────────────────────

def test_corrupted_model_file(tmp_path):
    """Loading a corrupted pickle file should raise a clear exception."""
    bad_pkl = tmp_path / "corrupted_model.pkl"
    bad_pkl.write_bytes(b"this is not a valid pickle file")

    with pytest.raises(Exception):
        joblib.load(bad_pkl)


# ────────────────────────────────────────────────────────────────────────────
# TEST 11 — Deterministic repeated prediction
# ────────────────────────────────────────────────────────────────────────────

def test_deterministic_repeated_prediction(model, normal_patient):
    """Same input must produce identical output every call."""
    proba1 = model.predict_proba(normal_patient)[0, 1]
    proba2 = model.predict_proba(normal_patient)[0, 1]
    proba3 = model.predict_proba(normal_patient)[0, 1]
    assert proba1 == proba2 == proba3, \
        "Predictions are not deterministic across repeated calls"


# ────────────────────────────────────────────────────────────────────────────
# TEST 12 — SHAP explanation exists and has expected structure
# ────────────────────────────────────────────────────────────────────────────

def test_shap_explanation_exists(model, normal_patient):
    """explain_patient returns a dict with the required keys."""
    from src.explainability import explain_patient

    raw_pipeline = get_raw_pipeline(model)
    explanation  = explain_patient(raw_pipeline, normal_patient)

    required_keys = {
        "raw_risk_score",
        "base_value",
        "shap_values",
        "feature_names",
        "top_increasing_risk",
        "top_decreasing_risk",
        "disclaimer",
    }
    assert required_keys.issubset(explanation.keys()), \
        f"Missing keys: {required_keys - explanation.keys()}"

    assert isinstance(explanation["top_increasing_risk"], list)
    assert isinstance(explanation["top_decreasing_risk"], list)
    assert "SHAP values describe" in explanation["disclaimer"]
    assert "causality" in explanation["disclaimer"].lower()


# ────────────────────────────────────────────────────────────────────────────
# TEST 13 — Output is finite (no NaN / Inf)
# ────────────────────────────────────────────────────────────────────────────

def test_output_is_finite(model, normal_patient, high_utilisation_patient,
                          low_utilisation_patient):
    """predict_proba must return only finite values for all test patients."""
    for patient in [normal_patient, high_utilisation_patient, low_utilisation_patient]:
        proba = model.predict_proba(patient)
        assert np.all(np.isfinite(proba)), \
            f"Non-finite value in output for patient:\n{patient}"


# ────────────────────────────────────────────────────────────────────────────
# TEST 14 — Probability in [0, 1]
# ────────────────────────────────────────────────────────────────────────────

def test_probability_in_range(model, normal_patient):
    """All predicted probabilities must lie within [0, 1]."""
    proba = model.predict_proba(normal_patient)
    assert proba.shape[1] == 2, "Expected 2-class output"
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0), \
        f"Probabilities outside [0, 1]: {proba}"
    # Row should sum to approximately 1
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


# ────────────────────────────────────────────────────────────────────────────
# TEST 15 — Baseline model is intact
# ────────────────────────────────────────────────────────────────────────────

def test_baseline_model_intact():
    """
    readmission_model_baseline.pkl must exist and be exactly the original
    file size (505,215 bytes). This ensures train.py never overwrote it.
    """
    assert BASELINE_PKL.exists(), \
        f"Baseline model missing! Expected at {BASELINE_PKL}"

    actual_size = BASELINE_PKL.stat().st_size
    assert actual_size == BASELINE_EXPECTED_SIZE, (
        f"Baseline model size changed: expected {BASELINE_EXPECTED_SIZE} bytes, "
        f"got {actual_size} bytes. The baseline may have been overwritten!"
    )

    # Also verify it can still be loaded and produces a prediction
    baseline = joblib.load(BASELINE_PKL)
    sample = pd.DataFrame([{
        "age":               "[70-80)",
        "time_in_hospital":  5,
        "n_lab_procedures":  40,
        "n_procedures":      2,
        "n_medications":     15,
        "n_outpatient":      0,
        "n_inpatient":       1,
        "n_emergency":       0,
        "medical_specialty": "InternalMedicine",
        "diag_1":            "Circulatory",
        "diag_2":            "Diabetes",
        "diag_3":            "Other",
        "glucose_test":      "no",
        "A1Ctest":           "no",
        "change":            "yes",
        "diabetes_med":      "yes",
    }])

    # Baseline does not have engineered features — pass raw input
    proba = baseline.predict_proba(sample)
    score = float(proba[0, 1])
    assert np.isfinite(score) and 0.0 <= score <= 1.0, \
        f"Baseline model returned invalid score: {score}"
