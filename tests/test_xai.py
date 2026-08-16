from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xai import explain_prediction

MODELS_DIR = PROJECT_ROOT / "models"
FINAL_PKL = MODELS_DIR / "readmission_model_final.pkl"


@pytest.fixture(scope="module")
def model():
    assert FINAL_PKL.exists(), (
        "Final model not found at "
        f"{FINAL_PKL}. Run `python backend/train.py` first."
    )
    return joblib.load(FINAL_PKL)


@pytest.fixture
def normal_patient() -> pd.DataFrame:
    return pd.DataFrame([{
        "age": "[70-80)",
        "time_in_hospital": 5,
        "n_lab_procedures": 40,
        "n_procedures": 2,
        "n_medications": 15,
        "n_outpatient": 0,
        "n_inpatient": 1,
        "n_emergency": 0,
        "medical_specialty": "InternalMedicine",
        "diag_1": "Circulatory",
        "diag_2": "Diabetes",
        "diag_3": "Other",
        "glucose_test": "no",
        "A1Ctest": "no",
        "change": "yes",
        "diabetes_med": "yes",
    }])


def test_explain_prediction_returns_structured_output(model, normal_patient):
    result = explain_prediction(model, normal_patient, top_n=5)

    assert isinstance(result, dict)
    assert {"risk_score", "prediction", "top_increasing_risk", "top_decreasing_risk", "feature_contributions", "disclaimer"}.issubset(result.keys())
    assert 0.0 <= float(result["risk_score"]) <= 1.0
    assert result["prediction"] in {"low_risk", "high_risk"}
    assert isinstance(result["top_increasing_risk"], list)
    assert isinstance(result["top_decreasing_risk"], list)
    assert isinstance(result["feature_contributions"], list)
    assert result["disclaimer"]


def test_explain_prediction_handles_empty_top_n(model, normal_patient):
    result = explain_prediction(model, normal_patient, top_n=0)
    assert isinstance(result["top_increasing_risk"], list)
    assert isinstance(result["top_decreasing_risk"], list)
    assert isinstance(result["feature_contributions"], list)
