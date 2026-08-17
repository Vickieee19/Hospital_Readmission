"""
streamlit_app.py
────────────────
Hospital Readmission Prediction & Prevention Console
A clean, clinician-friendly interface to predict readmission risk and generate
actionable prevention protocols to keep patients safe.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Readmission Risk & Prevention",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for Clean, Modern Clinical UI ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
}

.header-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-subtitle {
    margin-top: 0.4rem;
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
}

/* Card styling */
.input-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}

.card-heading {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 0.5rem;
}

/* Verdict Banners */
.verdict-high-box {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.12);
}

.verdict-low-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.12);
}

.verdict-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.01em;
}

.verdict-sub {
    font-size: 1rem;
    margin-top: 0.35rem;
    font-weight: 500;
}

/* Prevention Action Box */
.prevention-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-left: 6px solid #2563eb;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
}

.prevention-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.prevention-desc {
    color: #334155;
    font-size: 0.92rem;
    line-height: 1.45;
    margin: 0;
}

/* Risk Factor Pills */
.risk-factor-item {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    color: #9f1239;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.safe-factor-item {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    color: #166534;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Button enhancements */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    border: none;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    transition: all 0.2s ease-in-out;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
}
</style>
""", unsafe_allow_html=True)

# ── Constants & Model Loading ─────────────────────────────────────────────────
MODEL_PATH    = "models/readmission_model_final.pkl"
METADATA_PATH = "models/model_metadata.json"

AGE_OPTIONS      = ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
SPECIALTIES      = ["InternalMedicine", "Family/GeneralPractice", "Cardiology", "Surgery", "Emergency/Trauma", "Other", "Missing"]
DIAG_OPTIONS     = ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury", "Other"]
BINARY_OPTIONS   = ["no", "yes"]
GLUCOSE_OPTIONS  = ["no", "normal", "high", "Missing"]
A1C_OPTIONS      = ["no", "normal", "high", "Missing"]

CLINICAL_THRESHOLD = 0.52  # Standard calibrated threshold (MCC-optimized)

@st.cache_resource(show_spinner="Loading readmission prediction model…")
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    st.error(f"Error loading model pipeline: {e}")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">
        <span>🏥</span> Hospital Readmission Risk & Prevention Console
    </div>
    <div class="header-subtitle">
        Enter patient encounter data to determine 30-day readmission risk and generate targeted clinical prevention protocols.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Patient Input Form ────────────────────────────────────────────────────────
with st.form("patient_intake_form"):
    st.markdown("### 📋 Patient Details & Clinical Record")
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-heading'>👤 Demographics & Stay</div>", unsafe_allow_html=True)
        age = st.selectbox("Age Bracket", AGE_OPTIONS, index=2)
        time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=30, value=3)
        n_medications = st.number_input("Number of Medications", min_value=0, max_value=80, value=12)
        n_procedures = st.number_input("Clinical Procedures", min_value=0, max_value=10, value=1)
        n_lab_procedures = st.number_input("Lab Procedures", min_value=0, max_value=130, value=35)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-heading'>🏨 Prior Visits (Past 12 Months)</div>", unsafe_allow_html=True)
        n_inpatient = st.number_input("Prior Inpatient Admissions", min_value=0, max_value=20, value=0, help="Number of hospital admissions in the past year (Major readmission risk factor).")
        n_emergency = st.number_input("Prior Emergency Visits", min_value=0, max_value=20, value=0, help="Number of ER visits in the past year.")
        n_outpatient = st.number_input("Prior Outpatient Visits", min_value=0, max_value=40, value=0)
        medical_specialty = st.selectbox("Admitting Specialty", SPECIALTIES, index=0)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-heading'>🩺 Diagnoses & Lab Tests</div>", unsafe_allow_html=True)
        diag_1 = st.selectbox("Primary Diagnosis", DIAG_OPTIONS, index=0)
        diag_2 = st.selectbox("Secondary Diagnosis", DIAG_OPTIONS, index=1)
        diag_3 = st.selectbox("Tertiary Diagnosis", DIAG_OPTIONS, index=5)
        glucose_test = st.selectbox("Glucose Serum Test", GLUCOSE_OPTIONS, index=0)
        A1Ctest = st.selectbox("HbA1c (A1C) Test", A1C_OPTIONS, index=0)
        change = st.selectbox("Medication Changed During Stay", BINARY_OPTIONS, index=0)
        diabetes_med = st.selectbox("Prescribed Diabetes Medication", BINARY_OPTIONS, index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.form_submit_button("🔍  Assess Readmission Risk & Generate Prevention Plan", use_container_width=True)

# ── Prediction & Prevention Output ────────────────────────────────────────────
if model_ok and predict_btn:
    patient_df = pd.DataFrame([{
        "age": age,
        "time_in_hospital": time_in_hospital,
        "n_lab_procedures": n_lab_procedures,
        "n_procedures": n_procedures,
        "n_medications": n_medications,
        "n_outpatient": n_outpatient,
        "n_inpatient": n_inpatient,
        "n_emergency": n_emergency,
        "medical_specialty": medical_specialty,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
        "glucose_test": glucose_test,
        "A1Ctest": A1Ctest,
        "change": change,
        "diabetes_med": diabetes_med,
    }])
    
    with st.spinner("Analyzing clinical risk factors..."):
        risk_proba = float(model.predict_proba(patient_df)[0, 1])
        is_high_risk = risk_proba >= CLINICAL_THRESHOLD
        
    st.markdown("<hr style='margin: 1.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
    st.markdown("## 📊 Assessment Results")
    
    col_res_left, col_res_right = st.columns([1.2, 1], gap="large")
    
    with col_res_left:
        # Verdict banner
        if is_high_risk:
            st.markdown(f"""
            <div class="verdict-high-box">
                <div class="verdict-title" style="color: #b91c1c;">🚨 HIGH RISK OF READMISSION</div>
                <div class="verdict-sub" style="color: #991b1b;">
                    Predicted Probability: <strong>{risk_proba * 100:.1f}%</strong> · Action Required
                </div>
                <p style="margin-top: 0.6rem; color: #7f1d1d; font-size: 0.95rem; margin-bottom: 0;">
                    This patient has a high likelihood of unplanned 30-day hospital readmission. Proactive discharge interventions must be implemented before discharge.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-low-box">
                <div class="verdict-title" style="color: #15803d;">✅ LOW RISK (SAFE FOR DISCHARGE)</div>
                <div class="verdict-sub" style="color: #166534;">
                    Predicted Probability: <strong>{risk_proba * 100:.1f}%</strong> · Low Readmission Likelihood
                </div>
                <p style="margin-top: 0.6rem; color: #14532d; font-size: 0.95rem; margin-bottom: 0;">
                    This patient shows low readmission risk. Standard discharge planning, patient education, and routine outpatient follow-up are appropriate.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Key Contributing Factors
        st.markdown("#### 🔍 Why this patient received this score:")
        factors = []
        
        if n_inpatient >= 1:
            factors.append((True, f"<b>Prior Inpatient Admissions:</b> {n_inpatient} previous hospital stay(s) in the past year is a primary driver of readmission."))
        if n_emergency >= 1:
            factors.append((True, f"<b>Prior Emergency Visits:</b> {n_emergency} ER encounter(s) indicates frequent acute complications."))
        if n_medications >= 20:
            factors.append((True, f"<b>High Medication Burden:</b> {n_medications} active medications significantly elevates polypharmacy & adherence risks."))
        elif n_medications >= 12:
            factors.append((True, f"<b>Moderate Medication Count:</b> {n_medications} active medications."))
        if time_in_hospital >= 6:
            factors.append((True, f"<b>Extended Length of Stay:</b> {time_in_hospital} days in hospital reflects severe illness severity."))
        if age in ["[70-80)", "[80-90)", "[90-100)"]:
            factors.append((True, f"<b>Elderly Age Bracket:</b> {age} increases vulnerability post-discharge."))
        if diag_1 in ["Circulatory", "Diabetes", "Respiratory"]:
            factors.append((True, f"<b>Primary Diagnosis:</b> {diag_1} is a chronic condition associated with frequent recidivism."))
            
        if not factors:
            factors.append((False, "<b>Low Hospital Utilization:</b> 0 prior inpatient or emergency admissions in the past year."))
            factors.append((False, f"<b>Short Hospital Stay:</b> Discharged after only {time_in_hospital} day(s)."))
            factors.append((False, f"<b>Low Medication Count:</b> Only {n_medications} medications prescribed."))
            
        for is_risk, text in factors[:4]:
            css_cls = "risk-factor-item" if is_risk else "safe-factor-item"
            icon = "⚠️" if is_risk else "✓"
            st.markdown(f"<div class='{css_cls}'><span>{icon}</span> <span>{text}</span></div>", unsafe_allow_html=True)
            
    with col_res_right:
        # Visual Risk Gauge
        gauge_color = "#ef4444" if is_high_risk else ("#f59e0b" if risk_proba >= 0.35 else "#22c55e")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(risk_proba * 100, 1),
            number={"suffix": "%", "font": {"size": 48, "color": gauge_color, "family": "Inter"}},
            title={"text": "30-Day Readmission Risk Score", "font": {"size": 16, "color": "#334155", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8", "ticksuffix": "%"},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "#dcfce7"},
                    {"range": [35, 52], "color": "#fef3c7"},
                    {"range": [52, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 3},
                    "thickness": 0.85,
                    "value": CLINICAL_THRESHOLD * 100
                },
            },
        ))
        fig_gauge.update_layout(
            height=290,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("<div style='text-align:center; color:#64748b; font-size:0.85rem;'>Green: Low (&lt;35%) · Yellow: Moderate (35-52%) · Red: High Risk (&ge;52%)</div>", unsafe_allow_html=True)

    # ── Tailored Readmission Prevention Plan ──────────────────────────────────
    st.markdown("<hr style='margin: 1.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
    st.markdown("### 🛡️ Tailored Readmission Prevention Protocol")
    st.markdown("Clinical actions recommended before and immediately after discharge to avoid 30-day readmission:")

    col_act1, col_act2 = st.columns(2, gap="medium")
    
    with col_act1:
        st.markdown("""
        <div class="prevention-card">
            <div class="prevention-title">📞 1. 48-Hour Post-Discharge Outreach Call</div>
            <div class="prevention-desc">
                Designated nurse must call the patient or family caregiver within 48 hours to confirm discharge understanding, review symptoms, and verify medication access.
            </div>
        </div>
        
        <div class="prevention-card">
            <div class="prevention-title">💊 2. Pharmacy Medication Reconciliation</div>
            <div class="prevention-desc">
                Clinical pharmacist must perform complete medication reconciliation, identify contraindications, and provide a simplified medication timetable.
            </div>
        </div>
        
        <div class="prevention-card">
            <div class="prevention-title">📅 3. Rapid Outpatient Appointment (Within 7 Days)</div>
            <div class="prevention-desc">
                Confirm a scheduled follow-up visit with primary care or specialist prior to patient discharge. Hand physical appointment slip to patient.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_act2:
        st.markdown("""
        <div class="prevention-card">
            <div class="prevention-title">🩺 4. Disease-Specific Education & 'Red Flag' Warning Sheet</div>
            <div class="prevention-desc">
                Provide teach-back instructions on warning symptoms (chest pain, shortness of breath, blood glucose spikes) and direct contact numbers before visiting the ER.
            </div>
        </div>
        
        <div class="prevention-card">
            <div class="prevention-title">🏡 5. Home Health & Social Determinants Support</div>
            <div class="prevention-desc">
                Evaluate home safety, transport to appointments, caregiver support, and consider home health nursing for wound care, vitals monitoring, or insulin management.
            </div>
        </div>
        
        <div class="prevention-card">
            <div class="prevention-title">🩸 6. Lab & Diagnostic Follow-Up Tracking</div>
            <div class="prevention-desc">
                Ensure pending lab cultures and diagnostic tests are flagged for follow-up by the outpatient care team within 3-5 days.
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👈 **Fill in the patient clinical details above and click 'Assess Readmission Risk'** to get an immediate prediction and prevention protocol.", icon="ℹ️")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top: 3rem; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:.85rem;'>"
    "🏥 Hospital Readmission Prevention System · AI Clinical Decision Support · "
    "Designed for Healthcare Teams and Care Coordinators"
    "</p>",
    unsafe_allow_html=True,
)
