import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Add repository root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import explain_patient

# ── Feature Schemas ────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "time_in_hospital",
    "n_lab_procedures",
    "n_procedures",
    "n_medications",
    "n_outpatient",
    "n_inpatient",
    "n_emergency",
]

ORDINAL_FEATURES = ["age"]

AGE_ORDER = [
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]

SPECIALTY_OPTIONS = [
    "InternalMedicine",
    "Cardiology",
    "Emergency/Trauma",
    "Family/GeneralPractice",
    "Surgery",
    "Missing",
    "Other",
]

DIAGNOSIS_OPTIONS = [
    "Circulatory",
    "Respiratory",
    "Digestive",
    "Diabetes",
    "Injury",
    "Musculoskeletal",
    "Other",
    "Missing",
]

TEST_OPTIONS = ["no", "normal", "high"]
BINARY_OPTIONS = ["no", "yes"]

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

ALL_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES
DEFAULT_THRESHOLD = 0.3496


@st.cache_resource
def load_model():
    """Load the final trained model pipeline."""
    model_path = PROJECT_ROOT / "models" / "readmission_model_final.pkl"
    if not model_path.exists():
        # Fallback check
        model_path = PROJECT_ROOT / "models" / "readmission_model_baseline.pkl"

    if not model_path.exists():
        return None, str(model_path)

    model = joblib.load(model_path)
    return model, str(model_path)


def predict_dataframe(model, df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD):
    X = df[ALL_FEATURES].copy()
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["readmission_proba"] = probs
    df["flagged_high_risk"] = df["readmission_proba"] >= threshold
    df["risk_level"] = df["readmission_proba"].apply(
        lambda p: "High" if p >= threshold else ("Medium" if p >= threshold * 0.7 else "Low")
    )
    return df


def main():
    st.set_page_config(
        page_title="CareGrid — Hospital Readmission Risk",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 CareGrid: Hospital Readmission Prediction")
    st.markdown(
        "Predicting 30-day hospital readmission risk using an optimized **XGBoost** model "
        "with automated **SHAP explainability**."
    )

    model, model_path = load_model()

    if model is None:
        st.error(
            f"❌ Model artifact not found at `{model_path}`.\n\n"
            "Please run `python backend/train.py` from the project root to generate the model."
        )
        return

    st.sidebar.header("⚙️ Model Configuration")
    threshold = st.sidebar.slider(
        "Operating Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Validated optimal operating threshold for clinical prioritization is 0.3496 (F2 score maximizing).",
    )
    st.sidebar.info(
        f"**Active Model:** `readmission_model_final.pkl`\n\n"
        f"**Operating Threshold:** `{threshold:.4f}`\n\n"
        "Patients with risk $\\ge$ threshold are flagged for care coordinator intervention."
    )

    tab1, tab2, tab3 = st.tabs(["👤 Single Patient Assessment", "📁 Batch CSV Prediction", "📈 Model Insights"])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1: Single Patient Prediction
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Patient Clinical Data")

        with st.form("single_patient_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Demographics & Hospital Stay")
                age = st.selectbox("Age Bracket", AGE_ORDER, index=3)
                time_in_hospital = st.number_input("Time in Hospital (Days)", min_value=1, max_value=14, value=5)
                medical_specialty = st.selectbox("Admitting Specialty", SPECIALTY_OPTIONS, index=0)

            with col2:
                st.markdown("#### Clinical Procedures & Utilization")
                n_medications = st.number_input("Number of Medications", min_value=1, max_value=100, value=15)
                n_lab_procedures = st.number_input("Lab Procedures Count", min_value=0, max_value=150, value=42)
                n_procedures = st.number_input("Other Procedures Count", min_value=0, max_value=10, value=1)
                n_inpatient = st.number_input("Prior Inpatient Visits (Past Year)", min_value=0, max_value=20, value=1)
                n_outpatient = st.number_input("Prior Outpatient Visits (Past Year)", min_value=0, max_value=30, value=0)
                n_emergency = st.number_input("Prior Emergency Visits (Past Year)", min_value=0, max_value=30, value=0)

            with col3:
                st.markdown("#### Diagnoses & Diabetes Management")
                diag_1 = st.selectbox("Primary Diagnosis (diag_1)", DIAGNOSIS_OPTIONS, index=0)
                diag_2 = st.selectbox("Secondary Diagnosis (diag_2)", DIAGNOSIS_OPTIONS, index=3)
                diag_3 = st.selectbox("Tertiary Diagnosis (diag_3)", DIAGNOSIS_OPTIONS, index=6)
                glucose_test = st.selectbox("Glucose Serum Test", TEST_OPTIONS, index=0)
                A1Ctest = st.selectbox("A1C Test Result", TEST_OPTIONS, index=0)
                change = st.selectbox("Change in Diabetes Medications", BINARY_OPTIONS, index=1)
                diabetes_med = st.selectbox("Prescribed Diabetes Medication", BINARY_OPTIONS, index=1)

            submit_btn = st.form_submit_button("🔍 Calculate Readmission Risk", use_container_width=True)

        if submit_btn:
            patient_dict = {
                "age": age,
                "time_in_hospital": int(time_in_hospital),
                "n_lab_procedures": int(n_lab_procedures),
                "n_procedures": int(n_procedures),
                "n_medications": int(n_medications),
                "n_outpatient": int(n_outpatient),
                "n_inpatient": int(n_inpatient),
                "n_emergency": int(n_emergency),
                "medical_specialty": medical_specialty,
                "diag_1": diag_1,
                "diag_2": diag_2,
                "diag_3": diag_3,
                "glucose_test": glucose_test,
                "A1Ctest": A1Ctest,
                "change": change,
                "diabetes_med": diabetes_med,
            }

            patient_df = pd.DataFrame([patient_dict])[ALL_FEATURES]

            # Prediction
            risk_score = float(model.predict_proba(patient_df)[0, 1])
            is_high_risk = risk_score >= threshold

            st.markdown("---")
            st.subheader("Assessment Result")

            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.metric("Predicted Readmission Risk", f"{risk_score * 100:.1f}%")
            with r_col2:
                risk_badge = "🔴 HIGH RISK" if is_high_risk else ("🟡 MEDIUM RISK" if risk_score >= threshold * 0.7 else "🟢 LOW RISK")
                st.metric("Risk Classification", risk_badge)
            with r_col3:
                st.metric("Flagged for Intervention", "YES" if is_high_risk else "NO")

            # SHAP Explainability
            st.markdown("#### 🔬 Key Factors Driving This Prediction (SHAP)")
            try:
                explanation = explain_patient(model, patient_df)
                exp_col1, exp_col2 = st.columns(2)

                with exp_col1:
                    st.markdown("**Top Factors Increasing Risk (↑):**")
                    if explanation["top_increasing_risk"]:
                        for item in explanation["top_increasing_risk"][:5]:
                            clean_feat = item["feature"].replace("num__", "").replace("cat__", "").replace("age__", "")
                            st.write(f"• **`{clean_feat}`**: `+{item['shap_value']:.4f}`")
                    else:
                        st.write("None")

                with exp_col2:
                    st.markdown("**Top Factors Decreasing Risk (↓):**")
                    if explanation["top_decreasing_risk"]:
                        for item in explanation["top_decreasing_risk"][:5]:
                            clean_feat = item["feature"].replace("num__", "").replace("cat__", "").replace("age__", "")
                            st.write(f"• **`{clean_feat}`**: `{item['shap_value']:.4f}`")
                    else:
                        st.write("None")

                st.caption(f"ℹ️ {explanation['disclaimer']}")
            except Exception as e:
                st.warning(f"Could not compute SHAP explanation: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2: Batch CSV Prediction
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Batch Patient Risk Scoring")
        uploaded_file = st.file_uploader("Upload CSV file containing patient encounters", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                missing = [c for c in ALL_FEATURES if c not in batch_df.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing required columns: `{missing}`")
                else:
                    results = predict_dataframe(model, batch_df, threshold=threshold)

                    high_count = int(results["flagged_high_risk"].sum())
                    pct_high = high_count / len(results) * 100

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Patients", f"{len(results):,}")
                    m2.metric("Flagged High Risk", f"{high_count:,}")
                    m3.metric("Flagged Rate", f"{pct_high:.1f}%")

                    st.dataframe(
                        results[[*ALL_FEATURES, "readmission_proba", "risk_level"]].sort_values(
                            "readmission_proba", ascending=False
                        ),
                        use_container_width=True,
                    )

                    csv_data = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Scored Predictions CSV",
                        csv_data,
                        file_name="readmission_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Failed to process CSV: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3: Model Insights & Visualizations
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Model Evaluation & Global Performance")
        chart_dir = PROJECT_ROOT / "models" / "charts"

        c1, c2 = st.columns(2)
        with c1:
            gains_img = chart_dir / "cumulative_gains.png"
            if gains_img.exists():
                st.image(str(gains_img), caption="Cumulative Gains: Top 30% patients capture 40.4% readmissions")
            roc_img = chart_dir / "roc_curve.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="ROC Curve (AUC = 0.6581)")

        with c2:
            shap_img = chart_dir / "shap_global_importance.png"
            if shap_img.exists():
                st.image(str(shap_img), caption="Global Feature Importance (SHAP)")
            cal_img = chart_dir / "calibration_curve.png"
            if cal_img.exists():
                st.image(str(cal_img), caption="Reliability Diagram / Calibration Curve")


if __name__ == "__main__":
    main()