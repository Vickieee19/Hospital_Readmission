import joblib
import pandas as pd

# 1. Load the trained model
model = joblib.load("models/readmission_model_final.pkl")

# 2. Define a sample patient encounter
sample_patient = pd.DataFrame([{
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
    "diabetes_med": "yes"
}])

# 3. Predict probability (Threshold = 0.3496)
risk_score = model.predict_proba(sample_patient)[0, 1]
flagged = risk_score >= 0.3496

print("=" * 45)
print("  PATIENT READMISSION RISK PREDICTION")
print("=" * 45)
print(f"Predicted Risk Score : {risk_score:.4f} ({risk_score * 100:.1f}%)")
print(f"Decision Threshold   : 0.3496")
print(f"High-Risk Flagged    : {'YES (Prioritize)' if flagged else 'NO'}")
print("=" * 45)
