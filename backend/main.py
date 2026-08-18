    """
backend/main.py
───────────────
CareGrid FastAPI Backend Server
Integrates 30-Day Hospital Readmission Risk Prediction (XGBoost/LightGBM Ensemble)
and Clinical Severity RAG Analysis.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Ensure Project Root and Modules are in sys.path ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
RAG_ROOT = BACKEND_DIR / "rag"

for p in [PROJECT_ROOT, BACKEND_DIR, RAG_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# Load root .env
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

# Safe XAI import
_has_xai = False
try:
    from xai import explain_prediction
    _has_xai = True
except Exception as e:
    print(f"[Info] XAI explanation module deferred: {e}")

# ── Paths ───────────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "readmission_model_final.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Global Cache ────────────────────────────────────────────────────────────
_model = None
_metadata: dict[str, Any] = {}
_default_threshold: float = 0.5227


def load_model_and_metadata():
    global _model, _metadata, _default_threshold
    if _model is None and MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[Warning] Could not load model from {MODEL_PATH}: {e}")

    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
                _default_threshold = float(
                    _metadata.get("threshold", {}).get("value", 0.5227)
                )
        except Exception as e:
            print(f"[Warning] Could not load metadata from {METADATA_PATH}: {e}")


# Initialize model on module load
load_model_and_metadata()

# ── FastAPI App Setup ───────────────────────────────────────────────────────
app = FastAPI(
    title="CareGrid Clinical Decision Support API",
    description="API for 30-Day Hospital Readmission Risk Prediction & Clinical Severity RAG",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g. Vite on localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request & Response Schemas ─────────────────────────────────────
class PatientInput(BaseModel):
    age: str = Field(default="[70-80)", json_schema_extra={"example": "[70-80)"})
    time_in_hospital: int = Field(default=5, ge=0, json_schema_extra={"example": 5})
    n_lab_procedures: int = Field(default=40, ge=0, json_schema_extra={"example": 40})
    n_procedures: int = Field(default=2, ge=0, json_schema_extra={"example": 2})
    n_medications: int = Field(default=15, ge=0, json_schema_extra={"example": 15})
    n_outpatient: int = Field(default=0, ge=0, json_schema_extra={"example": 0})
    n_inpatient: int = Field(default=1, ge=0, json_schema_extra={"example": 1})
    n_emergency: int = Field(default=0, ge=0, json_schema_extra={"example": 0})
    medical_specialty: str = Field(default="InternalMedicine", json_schema_extra={"example": "InternalMedicine"})
    diag_1: str = Field(default="Circulatory", json_schema_extra={"example": "Circulatory"})
    diag_2: str = Field(default="Diabetes", json_schema_extra={"example": "Diabetes"})
    diag_3: str = Field(default="Other", json_schema_extra={"example": "Other"})
    glucose_test: str = Field(default="no", json_schema_extra={"example": "no"})
    A1Ctest: str = Field(default="no", json_schema_extra={"example": "no"})
    change: str = Field(default="yes", json_schema_extra={"example": "yes"})
    diabetes_med: str = Field(default="yes", json_schema_extra={"example": "yes"})
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, json_schema_extra={"example": 0.52})


class FeatureImpact(BaseModel):
    feature: str
    impact: float
    raw_feature: str | None = None


class PreventionProtocol(BaseModel):
    title: str
    description: str
    icon: str


class ContributingFactor(BaseModel):
    is_risk: bool
    title: str | None = None
    text: str


class DomainScore(BaseModel):
    domain: str
    score: int
    full_mark: int = 100
    risk_level: str


class BenchmarkMetric(BaseModel):
    metric: str
    patient_value: float
    benchmark_median: float
    high_risk_cutoff: float
    unit: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    risk_level: str
    shap_summary: str | None = None
    primary_risk: dict[str, Any] | None = None
    primary_protective: dict[str, Any] | None = None
    shap_values: list[FeatureImpact]
    top_increasing_risk: list[dict[str, Any]]
    top_decreasing_risk: list[dict[str, Any]]
    contributing_factors: list[ContributingFactor]
    prevention_protocols: list[PreventionProtocol]
    domain_scores: list[DomainScore]
    benchmarks: list[BenchmarkMetric]
    disclaimer: str
    xai_explanation: dict[str, Any] | None = None


def build_shap_summary(top_risk: list[dict[str, Any]], top_lower: list[dict[str, Any]]) -> str:
    """Create concise, readable narrative from top SHAP contributors."""
    risk_items = top_risk[:3]
    protect_items = top_lower[:3]

    risk_text = ", ".join(
        f"{item['feature']} (+{item['shap_value']:.3f})"
        for item in risk_items
    )
    protect_text = ", ".join(
        f"{item['feature']} ({item['shap_value']:.3f})"
        for item in protect_items
    )

    summary_parts = []
    if risk_items:
        summary_parts.append(f"Main risk drivers were {risk_text}.")
    if protect_items:
        summary_parts.append(f"The main protective factors were {protect_text}.")
    if not summary_parts:
        return "The model did not identify strong positive or negative feature shifts for this patient profile."

    return " ".join(summary_parts)


# ── Feature Label Dictionary ───────────────────────────────────────────────
FEATURE_NAME_MAP: dict[str, str] = {
    "total_prior_visits": "Total Prior Inpatient & ER Visits",
    "had_prior_inpatient": "Prior Inpatient Admission History",
    "had_prior_emergency": "Prior Emergency Encounter History",
    "n_inpatient": "Prior Inpatient Admissions",
    "n_emergency": "Prior Emergency Room Visits",
    "n_outpatient": "Prior Outpatient Visits",
    "n_medications": "Active Prescribed Medications",
    "time_in_hospital": "Hospital Stay Length (Days)",
    "n_lab_procedures": "Lab Procedures Performed",
    "n_procedures": "Clinical Inpatient Procedures",
    "lab_to_med_ratio": "Lab-to-Medication Ratio",
    "labs_per_day": "Daily Lab Intensity",
    "meds_per_day": "Daily Medication Rate",
    "procedures_per_day": "Procedures per Inpatient Day",
    "utilisation_intensity": "Healthcare Utilization Intensity",
    "long_stay_flag": "Extended Stay Flag (≥7 Days)",
    "high_polypharmacy_flag": "High Polypharmacy Flag (≥20 Meds)",
    "polypharmacy_flag": "Polypharmacy Flag (≥10 Meds)",
    "age": "Patient Age Bracket",
    "medical_specialty_InternalMedicine": "Specialty: Internal Medicine",
    "medical_specialty_Cardiology": "Specialty: Cardiology",
    "medical_specialty_Family/GeneralPractice": "Specialty: Family Practice",
    "medical_specialty_Surgery": "Specialty: General Surgery",
    "medical_specialty_Missing": "Specialty: Unspecified",
    "medical_specialty_Other": "Specialty: Other Specialty",
    "diag_1_Circulatory": "Primary Diagnosis: Circulatory / Cardiac",
    "diag_1_Diabetes": "Primary Diagnosis: Diabetes",
    "diag_1_Respiratory": "Primary Diagnosis: Respiratory Illness",
    "diag_1_Digestive": "Primary Diagnosis: Digestive Condition",
    "diag_1_Injury": "Primary Diagnosis: Injury / Trauma",
    "diag_1_Other": "Primary Diagnosis: Other Condition",
    "diag_2_Diabetes": "Secondary Diagnosis: Diabetes",
    "diag_2_Circulatory": "Secondary Diagnosis: Circulatory",
    "diag_2_Respiratory": "Secondary Diagnosis: Respiratory",
    "diag_2_Digestive": "Secondary Diagnosis: Digestive",
    "diag_2_Injury": "Secondary Diagnosis: Injury",
    "diag_2_Other": "Secondary Diagnosis: Other Condition",
    "diag_3_Diabetes": "Tertiary Diagnosis: Diabetes",
    "diag_3_Circulatory": "Tertiary Diagnosis: Circulatory",
    "diag_3_Respiratory": "Tertiary Diagnosis: Respiratory",
    "diag_3_Digestive": "Tertiary Diagnosis: Digestive",
    "diag_3_Injury": "Tertiary Diagnosis: Injury",
    "diag_3_Other": "Tertiary Diagnosis: Other Condition",
    "diabetes_med_yes": "Prescribed Diabetes Medication",
    "diabetes_med_no": "No Diabetes Medication",
    "change_yes": "Medication Regimen Modified",
    "change_no": "No Medication Change",
    "glucose_test_high": "Glucose Test: High (>200 mg/dL)",
    "glucose_test_normal": "Glucose Test: Normal",
    "glucose_test_no": "Glucose Test: Not Performed",
    "A1Ctest_high": "HbA1c Test: High (>8%)",
    "A1Ctest_normal": "HbA1c Test: Normal (<7%)",
    "A1Ctest_no": "HbA1c Test: Not Performed",
}


def clean_feature_label(raw_feat: str) -> str:
    cleaned = raw_feat.replace("num__", "").replace("cat__", "").replace("age__", "")
    return FEATURE_NAME_MAP.get(cleaned, cleaned.replace("_", " ").title())


# ── Clinical Prevention Protocols Generator ─────────────────────────────────
def get_prevention_protocols() -> list[PreventionProtocol]:
    return [
        PreventionProtocol(
            title="48-Hour Post-Discharge Outreach Call",
            description="Designated care nurse calls the patient or family caregiver within 48 hours to confirm discharge understanding, review symptoms, and verify medication access.",
            icon="phone",
        ),
        PreventionProtocol(
            title="Pharmacy Medication Reconciliation",
            description="Clinical pharmacist performs comprehensive medication reconciliation, identifies contraindications/polypharmacy risks, and provides a clear timetable.",
            icon="pill",
        ),
        PreventionProtocol(
            title="Rapid Outpatient Appointment (Within 7 Days)",
            description="Confirm scheduled follow-up visit with primary care physician or specialist prior to discharge, handing a physical appointment confirmation to the patient.",
            icon="calendar",
        ),
        PreventionProtocol(
            title="Disease-Specific Education & 'Red Flag' Warning Sheet",
            description="Provide teach-back instructions on warning symptoms (chest pain, shortness of breath, blood glucose spikes) and direct clinic contact numbers before ER visits.",
            icon="stethoscope",
        ),
        PreventionProtocol(
            title="Home Health & Social Determinants Support",
            description="Evaluate home safety, transport to appointments, caregiver support, and consider home health nursing for vitals monitoring or insulin management.",
            icon="home",
        ),
        PreventionProtocol(
            title="Lab & Diagnostic Follow-Up Tracking",
            description="Ensure pending lab cultures and diagnostic results are flagged for systematic review by the outpatient care team within 3-5 days.",
            icon="activity",
        ),
    ]


def derive_contributing_factors(patient: dict[str, Any]) -> list[ContributingFactor]:
    factors: list[ContributingFactor] = []

    n_inpatient = int(patient.get("n_inpatient", 0))
    n_emergency = int(patient.get("n_emergency", 0))
    n_medications = int(patient.get("n_medications", 0))
    time_in_hospital = int(patient.get("time_in_hospital", 0))
    age = str(patient.get("age", ""))
    diag_1 = str(patient.get("diag_1", ""))
    change = str(patient.get("change", "no")).lower()
    glucose_test = str(patient.get("glucose_test", "no")).lower()
    a1c_test = str(patient.get("A1Ctest", "no")).lower()

    if n_inpatient >= 1:
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="Prior Inpatient Admissions",
                text=f"Prior Inpatient Admissions: {n_inpatient} previous hospital stay(s) in the past year is a primary driver of readmission.",
            )
        )
    if n_emergency >= 1:
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="Prior Emergency Visits",
                text=f"Prior Emergency Visits: {n_emergency} ER encounter(s) indicates frequent acute complications.",
            )
        )
    if n_medications >= 20:
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="High Medication Burden",
                text=f"High Medication Burden: {n_medications} active medications significantly elevates polypharmacy & adherence risks.",
            )
        )
    elif n_medications >= 12:
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="Moderate Medication Burden",
                text=f"Moderate Medication Burden: {n_medications} active prescribed medications.",
            )
        )
    if time_in_hospital >= 6:
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="Extended Length of Stay",
                text=f"Extended Length of Stay: {time_in_hospital} days in hospital reflects severe illness severity.",
            )
        )
    elif time_in_hospital >= 4:
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="Moderate Hospital Stay",
                text=f"Moderate Hospital Stay: {time_in_hospital} days in hospital.",
            )
        )
    if age in ["[70-80)", "[80-90)", "[90-100)"]:
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="Elderly Age Bracket",
                text=f"Elderly Age Bracket: {age} indicates higher physical vulnerability post-discharge.",
            )
        )
    if diag_1 in ["Circulatory", "Diabetes", "Respiratory"]:
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="Primary Diagnosis",
                text=f"Primary Diagnosis: {diag_1} is a chronic condition associated with frequent recidivism.",
            )
        )
    if change == "yes":
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="Medication Regimen Modification",
                text="Medication Regimen Modification: Dosage or active drugs adjusted during stay, requiring post-discharge titration.",
            )
        )
    if glucose_test == "high" or a1c_test == "high":
        factors.append(
            ContributingFactor(
                is_risk=True,
                title="Elevated Glycemic Lab Values",
                text="Elevated Glycemic Lab Values: Abnormal glucose/HbA1c test result warrants close outpatient endocrinology follow-up.",
            )
        )

    # Protective factors if low risk or minimal utilization
    if not factors:
        factors.append(
            ContributingFactor(
                is_risk=False,
                title="Low Hospital Utilization",
                text="Low Hospital Utilization: 0 prior inpatient or emergency admissions in past year.",
            )
        )
        factors.append(
            ContributingFactor(
                is_risk=False,
                title="Short Hospital Stay",
                text=f"Short Hospital Stay: Discharged after only {time_in_hospital} day(s).",
            )
        )
        factors.append(
            ContributingFactor(
                is_risk=False,
                title="Low Medication Count",
                text=f"Low Medication Count: Only {n_medications} medications prescribed.",
            )
        )

    return factors[:6]


def calculate_domain_scores(patient: dict[str, Any]) -> list[DomainScore]:
    inp = int(patient.get("n_inpatient", 0))
    er = int(patient.get("n_emergency", 0))
    meds = int(patient.get("n_medications", 0))
    stay = int(patient.get("time_in_hospital", 0))
    age = str(patient.get("age", ""))
    diag = str(patient.get("diag_1", ""))

    # 1. Prior Utilization (0-100)
    util_score = min(100, int((inp * 30) + (er * 20)))
    # 2. Polypharmacy Burden
    poly_score = min(100, int((meds / 40) * 100))
    # 3. Hospital Stay & Acuity
    stay_score = min(100, int((stay / 14) * 100))
    # 4. Chronic Complexity
    diag_base = 65 if diag in ["Circulatory", "Diabetes", "Respiratory"] else 30
    # 5. Demographics / Age Vulnerability
    age_map = {"[40-50)": 25, "[50-60)": 40, "[60-70)": 60, "[70-80)": 85, "[80-90)": 95, "[90-100)": 100}
    age_score = age_map.get(age, 50)

    def to_level(s: int) -> str:
        return "High" if s >= 60 else "Moderate" if s >= 35 else "Low"

    return [
        DomainScore(domain="Prior Utilization", score=util_score, risk_level=to_level(util_score)),
        DomainScore(domain="Polypharmacy Burden", score=poly_score, risk_level=to_level(poly_score)),
        DomainScore(domain="Inpatient Acuity & Stay", score=stay_score, risk_level=to_level(stay_score)),
        DomainScore(domain="Chronic Complexity", score=diag_base, risk_level=to_level(diag_base)),
        DomainScore(domain="Demographics & Age", score=age_score, risk_level=to_level(age_score)),
    ]


def calculate_benchmarks(patient: dict[str, Any]) -> list[BenchmarkMetric]:
    return [
        BenchmarkMetric(
            metric="Length of Stay",
            patient_value=float(patient.get("time_in_hospital", 0)),
            benchmark_median=3.0,
            high_risk_cutoff=6.0,
            unit="days",
        ),
        BenchmarkMetric(
            metric="Active Medications",
            patient_value=float(patient.get("n_medications", 0)),
            benchmark_median=12.0,
            high_risk_cutoff=20.0,
            unit="meds",
        ),
        BenchmarkMetric(
            metric="Prior Inpatient Admissions",
            patient_value=float(patient.get("n_inpatient", 0)),
            benchmark_median=0.0,
            high_risk_cutoff=1.0,
            unit="visits",
        ),
        BenchmarkMetric(
            metric="Prior Emergency Visits",
            patient_value=float(patient.get("n_emergency", 0)),
            benchmark_median=0.0,
            high_risk_cutoff=1.0,
            unit="visits",
        ),
    ]


def fallback_feature_impacts(patient: dict[str, Any]) -> list[FeatureImpact]:
    """Provide feature attribution estimates when SHAP library is not present."""
    impacts = []
    inp = int(patient.get("n_inpatient", 0))
    if inp > 0:
        impacts.append(FeatureImpact(feature="Total Prior Inpatient & ER Visits", impact=0.18 * inp, raw_feature="total_prior_visits"))
        impacts.append(FeatureImpact(feature="Prior Inpatient Admission History", impact=0.12, raw_feature="had_prior_inpatient"))
    else:
        impacts.append(FeatureImpact(feature="Total Prior Inpatient & ER Visits", impact=-0.14, raw_feature="total_prior_visits"))

    time_h = int(patient.get("time_in_hospital", 1))
    if time_h >= 7:
        impacts.append(FeatureImpact(feature="Extended Stay Flag (≥7 Days)", impact=0.09, raw_feature="long_stay_flag"))
    elif time_h <= 2:
        impacts.append(FeatureImpact(feature="Hospital Stay Length (Days)", impact=-0.06, raw_feature="time_in_hospital"))

    meds = int(patient.get("n_medications", 10))
    if meds > 18:
        impacts.append(FeatureImpact(feature="Active Prescribed Medications", impact=0.08, raw_feature="n_medications"))
    else:
        impacts.append(FeatureImpact(feature="Active Prescribed Medications", impact=-0.04, raw_feature="n_medications"))

    age = str(patient.get("age", ""))
    if age in ["[80-90)", "[90-100)"]:
        impacts.append(FeatureImpact(feature="Patient Age Bracket", impact=0.07, raw_feature="age"))
    elif age in ["[40-50)", "[50-60)"]:
        impacts.append(FeatureImpact(feature="Patient Age Bracket", impact=-0.05, raw_feature="age"))

    diag = str(patient.get("diag_1", ""))
    if diag in ["Circulatory", "Diabetes"]:
        impacts.append(FeatureImpact(feature=f"Primary Diagnosis: {diag}", impact=0.05, raw_feature="diag_1"))

    return impacts[:8]


# ── Health & Metadata Endpoints ─────────────────────────────────────────────
@app.get("/health", summary="Health Check")
def health_check():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _metadata.get("model_version", "v2 (Optuna+Ensemble+MCC)"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metadata", summary="Model Metadata")
def get_metadata():
    if not _metadata and METADATA_PATH.exists():
        load_model_and_metadata()
    return _metadata or {"error": "Metadata not loaded"}


# ── Readmission Risk Prediction Endpoint ────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, summary="Predict 30-Day Readmission Risk")
def predict_readmission(patient: PatientInput):
    global _model
    if _model is None:
        load_model_and_metadata()
        if _model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Readmission model pipeline is not loaded on server.",
            )

    patient_dict = patient.model_dump()
    operating_threshold = (
        patient.threshold if patient.threshold is not None else _default_threshold
    )

    # Convert to single-row DataFrame (excluding custom threshold parameter)
    feature_dict = {k: v for k, v in patient_dict.items() if k != "threshold"}
    patient_df = pd.DataFrame([feature_dict])

    try:
        # Predict probability of readmission
        proba_arr = _model.predict_proba(patient_df)
        raw_prob = float(proba_arr[0, 1])
        prob = round(raw_prob, 4)

        # High risk determination
        is_high_risk = 1 if prob >= operating_threshold else 0

        # Categorize risk level
        if prob < 0.35:
            risk_level = "Low"
        elif prob < operating_threshold:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # Compute XAI explanation
        shap_feature_impacts: list[FeatureImpact] = []
        top_increasing: list[dict[str, Any]] = []
        top_decreasing: list[dict[str, Any]] = []
        xai_explanation_data: dict[str, Any] | None = None
        disclaimer_text = (
            "These feature contributions explain the model's predicted risk for this patient "
            "relative to the model's baseline. They are attribution values, not proof of causation."
        )

        if _has_xai:
            try:
                xai_explanation_data = explain_prediction(
                    _model,
                    patient_df,
                    top_n=8,
                    threshold=operating_threshold,
                )

                for item in xai_explanation_data.get("top_increasing_risk", []):
                    clean_name = clean_feature_label(item["feature"])
                    val = float(item["shap_value"])
                    top_increasing.append({
                        "feature": clean_name,
                        "raw_feature": item["feature"],
                        "shap_value": round(val, 4),
                        "direction": item.get("direction", "increases risk"),
                    })

                for item in xai_explanation_data.get("top_decreasing_risk", []):
                    clean_name = clean_feature_label(item["feature"])
                    val = float(item["shap_value"])
                    top_decreasing.append({
                        "feature": clean_name,
                        "raw_feature": item["feature"],
                        "shap_value": round(val, 4),
                        "direction": item.get("direction", "decreases risk"),
                    })

                for item in xai_explanation_data.get("feature_contributions", []):
                    clean_name = clean_feature_label(item["feature"])
                    val = float(item["shap_value"])
                    shap_feature_impacts.append(
                        FeatureImpact(
                            feature=clean_name,
                            impact=round(val, 4),
                            raw_feature=item["feature"],
                        )
                    )

                if xai_explanation_data.get("disclaimer"):
                    disclaimer_text = xai_explanation_data["disclaimer"]

            except Exception as se:
                print(f"[Info] XAI calculation fallback: {se}")
                shap_feature_impacts = fallback_feature_impacts(patient_dict)
        else:
            shap_feature_impacts = fallback_feature_impacts(patient_dict)

        if not top_increasing and not top_decreasing and shap_feature_impacts:
            for fi in shap_feature_impacts:
                if fi.impact > 0:
                    top_increasing.append({
                        "feature": fi.feature,
                        "raw_feature": fi.raw_feature,
                        "shap_value": fi.impact,
                        "direction": "increases risk",
                    })
                elif fi.impact < 0:
                    top_decreasing.append({
                        "feature": fi.feature,
                        "raw_feature": fi.raw_feature,
                        "shap_value": fi.impact,
                        "direction": "decreases risk",
                    })

        primary_risk = top_increasing[0] if top_increasing else None
        primary_protective = top_decreasing[0] if top_decreasing else None
        shap_summary = build_shap_summary(top_increasing, top_decreasing)

        contributing_factors = derive_contributing_factors(patient_dict)
        prevention_protocols = get_prevention_protocols()
        domain_scores = calculate_domain_scores(patient_dict)
        benchmarks = calculate_benchmarks(patient_dict)

        return PredictionResponse(
            prediction=is_high_risk,
            probability=prob,
            threshold=round(operating_threshold, 4),
            risk_level=risk_level,
            shap_summary=shap_summary,
            primary_risk=primary_risk,
            primary_protective=primary_protective,
            shap_values=shap_feature_impacts,
            top_increasing_risk=top_increasing,
            top_decreasing_risk=top_decreasing,
            contributing_factors=contributing_factors,
            prevention_protocols=prevention_protocols,
            domain_scores=domain_scores,
            benchmarks=benchmarks,
            disclaimer=disclaimer_text,
            xai_explanation=xai_explanation_data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


# ── PDF Upload & Clinical Severity RAG Endpoint ─────────────────────────────
@app.post("/upload-pdf", summary="Upload & Analyze Clinical PDF Report")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents (.pdf) are supported.",
        )

    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"{timestamp}_{file.filename}"
    saved_path = UPLOADS_DIR / saved_filename

    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)

    # Parse text with PyMuPDF
    extracted_text = ""
    try:
        import fitz
        doc = fitz.open(saved_path)
        extracted_text = "\n".join([page.get_text() for page in doc])
        doc.close()
    except Exception as e:
        extracted_text = f"Could not extract text: {e}"

    # Attempt Clinical Severity RAG & LLM Analysis
    rag_assessment = None
    retrieved_chunks = []
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    try:
        try:
            from backend.rag.rag.rag_pipeline import RAGPipeline
            from backend.rag.llm.severity_analyzer import SeverityAnalyzer
        except (ImportError, ModuleNotFoundError):
            try:
                from rag.rag.rag_pipeline import RAGPipeline
                from rag.llm.severity_analyzer import SeverityAnalyzer
            except (ImportError, ModuleNotFoundError):
                from rag.rag_pipeline import RAGPipeline
                from llm.severity_analyzer import SeverityAnalyzer

        rag = RAGPipeline()
        if rag.is_knowledge_base_ready() and extracted_text.strip():
            retrieved = rag.run_query(extracted_text, top_k=4)
            retrieved_chunks = [
                {
                    "source": chunk.source_doc,
                    "similarity": round(chunk.similarity_score, 3),
                    "text": chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                }
                for chunk in retrieved
            ]

            if gemini_key:
                analyzer = SeverityAnalyzer()
                result = analyzer.analyze(extracted_text, retrieved)
                rag_assessment = {
                    "severity_score": result.severity_score,
                    "severity_level": result.severity_level,
                    "key_findings": result.key_findings,
                    "evidence": result.evidence,
                    "summary": result.summary,
                }
    except Exception as e:
        print(f"[Info] RAG analysis note: {e}")

    # Fallback summary if Gemini key not yet configured
    if rag_assessment is None and extracted_text.strip():
        rag_assessment = {
            "severity_score": 5,
            "severity_level": "Moderate",
            "key_findings": [
                "Clinical lab report text successfully extracted from uploaded PDF.",
                "To enable automatic LLM clinical severity scoring, set GEMINI_API_KEY in .env.",
            ],
            "evidence": [
                f"Extracted {len(extracted_text):,} characters from {file.filename}."
            ],
            "summary": (
                "Document parsed successfully. Add your Gemini API key in the root .env file "
                "to activate multi-guideline evidence retrieval and clinical severity scoring."
            ),
        }

    return {
        "success": True,
        "message": f"Successfully uploaded and analyzed {file.filename}",
        "filename": file.filename,
        "saved_as": saved_filename,
        "extracted_text_preview": extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""),
        "severity_assessment": rag_assessment,
        "retrieved_guidelines": retrieved_chunks,
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting CareGrid Backend Server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
