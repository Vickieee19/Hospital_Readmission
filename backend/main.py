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

# Safe SHAP import
_has_shap = False
try:
    from src.explainability import explain_patient
    _has_shap = True
except Exception as e:
    print(f"[Info] SHAP explanation module deferred: {e}")

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
    age: str = Field(default="[70-80)", example="[70-80)")
    time_in_hospital: int = Field(default=5, ge=0, example=5)
    n_lab_procedures: int = Field(default=40, ge=0, example=40)
    n_procedures: int = Field(default=2, ge=0, example=2)
    n_medications: int = Field(default=15, ge=0, example=15)
    n_outpatient: int = Field(default=0, ge=0, example=0)
    n_inpatient: int = Field(default=1, ge=0, example=1)
    n_emergency: int = Field(default=0, ge=0, example=0)
    medical_specialty: str = Field(default="InternalMedicine", example="InternalMedicine")
    diag_1: str = Field(default="Circulatory", example="Circulatory")
    diag_2: str = Field(default="Diabetes", example="Diabetes")
    diag_3: str = Field(default="Other", example="Other")
    glucose_test: str = Field(default="no", example="no")
    A1Ctest: str = Field(default="no", example="no")
    change: str = Field(default="yes", example="yes")
    diabetes_med: str = Field(default="yes", example="yes")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, example=0.52)


class FeatureImpact(BaseModel):
    feature: str
    impact: float


class PreventionProtocol(BaseModel):
    title: str
    description: str
    icon: str


class ContributingFactor(BaseModel):
    is_risk: bool
    text: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    risk_level: str
    shap_values: list[FeatureImpact]
    top_increasing_risk: list[dict[str, Any]]
    top_decreasing_risk: list[dict[str, Any]]
    contributing_factors: list[ContributingFactor]
    prevention_protocols: list[PreventionProtocol]
    disclaimer: str


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

    if n_inpatient >= 1:
        factors.append(
            ContributingFactor(
                is_risk=True,
                text=f"Prior Inpatient Admissions: {n_inpatient} hospital stay(s) in past year is a primary driver of readmission.",
            )
        )
    if n_emergency >= 1:
        factors.append(
            ContributingFactor(
                is_risk=True,
                text=f"Prior Emergency Visits: {n_emergency} ER encounter(s) indicates frequent acute complications.",
            )
        )
    if n_medications >= 20:
        factors.append(
            ContributingFactor(
                is_risk=True,
                text=f"High Medication Burden: {n_medications} active medications significantly elevates polypharmacy risks.",
            )
        )
    elif n_medications >= 12:
        factors.append(
            ContributingFactor(
                is_risk=True,
                text=f"Moderate Medication Count: {n_medications} active prescribed medications.",
            )
        )
    if time_in_hospital >= 6:
        factors.append(
            ContributingFactor(
                is_risk=True,
                text=f"Extended Hospital Stay: {time_in_hospital} days in hospital reflects severe illness condition.",
            )
        )
    if age in ["[70-80)", "[80-90)", "[90-100)"]:
        factors.append(
            ContributingFactor(
                is_risk=True,
                text=f"Elderly Age Bracket: {age} indicates higher physical vulnerability post-discharge.",
            )
        )
    if diag_1 in ["Circulatory", "Diabetes", "Respiratory"]:
        factors.append(
            ContributingFactor(
                is_risk=True,
                text=f"Primary Diagnosis: {diag_1} is a chronic condition associated with frequent recidivism.",
            )
        )

    if not factors:
        factors.append(
            ContributingFactor(
                is_risk=False,
                text="Low Hospital Utilization: 0 prior inpatient or emergency admissions in past year.",
            )
        )
        factors.append(
            ContributingFactor(
                is_risk=False,
                text=f"Short Hospital Stay: Discharged after only {time_in_hospital} day(s).",
            )
        )
        factors.append(
            ContributingFactor(
                is_risk=False,
                text=f"Low Medication Count: Only {n_medications} medications prescribed.",
            )
        )

    return factors[:4]


def fallback_feature_impacts(patient: dict[str, Any]) -> list[FeatureImpact]:
    """Provide feature attribution estimates when SHAP library is not present."""
    impacts = []
    inp = int(patient.get("n_inpatient", 0))
    if inp > 0:
        impacts.append(FeatureImpact(feature="total_prior_visits", impact=0.18 * inp))
        impacts.append(FeatureImpact(feature="had_prior_inpatient", impact=0.12))
    else:
        impacts.append(FeatureImpact(feature="total_prior_visits", impact=-0.14))

    time_h = int(patient.get("time_in_hospital", 1))
    if time_h >= 7:
        impacts.append(FeatureImpact(feature="long_stay_flag", impact=0.09))
    elif time_h <= 2:
        impacts.append(FeatureImpact(feature="time_in_hospital", impact=-0.06))

    meds = int(patient.get("n_medications", 10))
    if meds > 18:
        impacts.append(FeatureImpact(feature="n_medications", impact=0.08))
    else:
        impacts.append(FeatureImpact(feature="n_medications", impact=-0.04))

    age = str(patient.get("age", ""))
    if age in ["[80-90)", "[90-100)"]:
        impacts.append(FeatureImpact(feature="age", impact=0.07))
    elif age in ["[40-50)", "[50-60)"]:
        impacts.append(FeatureImpact(feature="age", impact=-0.05))

    diag = str(patient.get("diag_1", ""))
    if diag in ["Circulatory", "Diabetes"]:
        impacts.append(FeatureImpact(feature="diag_complexity", impact=0.05))

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

        # Compute SHAP explanation if available
        shap_feature_impacts: list[FeatureImpact] = []
        top_increasing: list[dict[str, Any]] = []
        top_decreasing: list[dict[str, Any]] = []

        if _has_shap:
            try:
                shap_explanation = explain_patient(_model, patient_df, top_n=8)
                feat_names = shap_explanation.get("feature_names", [])
                shap_values = shap_explanation.get("shap_values", [])

                clean_impacts = []
                for feat, val in zip(feat_names, shap_values):
                    clean_name = feat.replace("num__", "").replace("cat__", "").replace("age__", "")
                    clean_impacts.append((clean_name, float(val)))

                clean_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                for clean_name, val in clean_impacts[:10]:
                    shap_feature_impacts.append(
                        FeatureImpact(feature=clean_name, impact=round(val, 4))
                    )

                top_increasing = shap_explanation.get("top_increasing_risk", [])
                top_decreasing = shap_explanation.get("top_decreasing_risk", [])
            except Exception as se:
                print(f"[Info] SHAP calculation fallback: {se}")
                shap_feature_impacts = fallback_feature_impacts(patient_dict)
        else:
            shap_feature_impacts = fallback_feature_impacts(patient_dict)

        contributing_factors = derive_contributing_factors(patient_dict)
        prevention_protocols = get_prevention_protocols()

        return PredictionResponse(
            prediction=is_high_risk,
            probability=prob,
            threshold=round(operating_threshold, 4),
            risk_level=risk_level,
            shap_values=shap_feature_impacts,
            top_increasing_risk=top_increasing,
            top_decreasing_risk=top_decreasing,
            contributing_factors=contributing_factors,
            prevention_protocols=prevention_protocols,
            disclaimer=(
                "Feature impacts describe each parameter's relative contribution to the model's "
                "predicted risk score compared to baseline averages. They do not establish direct clinical causality."
            ),
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
