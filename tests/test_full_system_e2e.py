from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.database import SessionLocal, init_db_schema
from backend.auth import seed_default_admin
from backend.models_db import User

client = TestClient(app)


def test_full_system_end_to_end():
    """
    Comprehensive Full-System End-to-End Simulation:
    1. Super Admin Auth & Setup
    2. Sub-Admin Provisioning with Designation & Selective Roles
    3. Sub-Admin Login & Nurse Provisioning
    4. Nurse Login & Clinical Prediction (Binary YES/NO, SHAP, Gauge metrics)
    5. PDF Extraction & Edge Case Handling
    6. Admin Password Control (Viewing, Editing) & Account Deletion
    """
    init_db_schema()
    with SessionLocal() as db:
        seed_default_admin(db)

    print("\n[E2E] 1. Super Admin Login...")
    admin_login = client.post("/auth/login", json={"username": "admin", "password": "Admin@123"})
    assert admin_login.status_code == 200
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    print("[E2E] 2. Super Admin provisions Sub-Admin (Dr. Gregory House, MD)...")
    sub_admin_user = "dr_house_e2e"
    sub_admin_pwd = "HousePassword@123"
    with SessionLocal() as db:
        old = db.query(User).filter(User.username == sub_admin_user).first()
        if old:
            db.delete(old)
            db.commit()

    sub_create = client.post(
        "/admin/users",
        headers=admin_headers,
        json={
            "username": sub_admin_user,
            "full_name": "Dr. Gregory House, MD",
            "designation": "Doctor / Diagnostic Lead",
            "password": sub_admin_pwd,
            "role": "sub_admin",
            "permissions": "create_users,audit_reports",
        },
    )
    assert sub_create.status_code == 201
    sub_data = sub_create.json()
    assert sub_data["designation"] == "Doctor / Diagnostic Lead"
    assert sub_data["initial_password"] == sub_admin_pwd

    print("[E2E] 3. Sub-Admin logs in & provisions a Clinical Staff Nurse...")
    sub_login = client.post("/auth/login", json={"username": sub_admin_user, "password": sub_admin_pwd})
    assert sub_login.status_code == 200
    sub_token = sub_login.json()["access_token"]
    sub_headers = {"Authorization": f"Bearer {sub_token}"}

    nurse_user = "nurse_e2e_claire"
    nurse_pwd = "ClairePass@123"
    with SessionLocal() as db:
        old_n = db.query(User).filter(User.username == nurse_user).first()
        if old_n:
            db.delete(old_n)
            db.commit()

    nurse_create = client.post(
        "/admin/users",
        headers=sub_headers,
        json={
            "username": nurse_user,
            "full_name": "Claire Temple, RN",
            "designation": "ICU Staff Nurse",
            "password": nurse_pwd,
            "role": "nurse",
        },
    )
    assert nurse_create.status_code == 201
    nurse_id = nurse_create.json()["id"]

    print("[E2E] 4. Nurse logs in and runs a Clinical Readmission Prediction...")
    nurse_login = client.post("/auth/login", json={"username": nurse_user, "password": nurse_pwd})
    assert nurse_login.status_code == 200
    nurse_token = nurse_login.json()["access_token"]
    nurse_headers = {"Authorization": f"Bearer {nurse_token}"}

    # High risk patient profile
    patient_payload = {
        "age": "[70-80)",
        "time_in_hospital": 7,
        "n_lab_procedures": 65,
        "n_procedures": 3,
        "n_medications": 22,
        "n_outpatient": 1,
        "n_inpatient": 3,
        "n_emergency": 2,
        "medical_specialty": "Cardiology",
        "diag_1": "Circulatory",
        "diag_2": "Diabetes",
        "diag_3": "Other",
        "glucose_test": "high",
        "A1Ctest": "high",
        "change": "yes",
        "diabetes_med": "yes",
    }
    pred_res = client.post("/predict", headers=nurse_headers, json=patient_payload)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    assert "prediction" in pred_data
    assert "probability" in pred_data
    assert "threshold" in pred_data
    assert "risk_level" in pred_data
    assert "shap_summary" in pred_data
    assert "top_increasing_risk" in pred_data
    assert "top_decreasing_risk" in pred_data
    assert "contributing_factors" in pred_data
    assert "domain_scores" in pred_data
    assert "benchmarks" in pred_data
    print(f"[E2E] -> Prediction Result: Readmitted={pred_data['prediction']} (Prob: {pred_data['probability']:.1%}, Level: {pred_data['risk_level']})")

    print("[E2E] 5. PDF Upload Extraction Testing...")
    # Test valid medical PDF
    high_risk_pdf = PROJECT_ROOT / "uploads" / "sample_patient_report_high_risk.pdf"
    if high_risk_pdf.exists():
        with open(high_risk_pdf, "rb") as f:
            pdf_res = client.post(
                "/upload-pdf",
                headers=nurse_headers,
                files={"file": ("sample_report.pdf", f, "application/pdf")},
            )
        assert pdf_res.status_code == 200
        pdf_data = pdf_res.json()
        assert pdf_data["is_medical_report"] is True
        assert "extracted_patient" in pdf_data
        assert pdf_data["extracted_fields_count"] > 0
        print(f"[E2E] -> Medical PDF parsed successfully ({pdf_data['extracted_fields_count']} fields extracted).")

    # Test non-medical PDF rejection
    non_med_pdf = PROJECT_ROOT / "uploads" / "sample_invoice_non_medical.pdf"
    if non_med_pdf.exists():
        with open(non_med_pdf, "rb") as f:
            non_med_res = client.post(
                "/upload-pdf",
                headers=nurse_headers,
                files={"file": ("invoice.pdf", f, "application/pdf")},
            )
        assert non_med_res.status_code == 200
        non_med_data = non_med_res.json()
        assert non_med_data["is_medical_report"] is False
        assert non_med_data["error_type"] == "NON_MEDICAL_DOCUMENT"
        print("[E2E] -> Non-medical PDF correctly rejected with warning.")

    print("[E2E] 6. Admin Password Editing & Account Deletion...")
    # Admin resets nurse password
    new_nurse_pwd = "ClaireNewPass@456"
    pwd_res = client.patch(
        f"/admin/users/{nurse_id}/password",
        headers=admin_headers,
        json={"new_password": new_nurse_pwd},
    )
    assert pwd_res.status_code == 200
    assert pwd_res.json()["initial_password"] == new_nurse_pwd

    # Nurse logs in with new password
    new_nurse_login = client.post("/auth/login", json={"username": nurse_user, "password": new_nurse_pwd})
    assert new_nurse_login.status_code == 200

    # Admin deletes nurse
    del_res = client.delete(f"/admin/users/{nurse_id}", headers=admin_headers)
    assert del_res.status_code == 200

    # Verify deleted nurse cannot log in
    del_nurse_login = client.post("/auth/login", json={"username": nurse_user, "password": new_nurse_pwd})
    assert del_nurse_login.status_code == 401

    print("\n[E2E SUCCESS] All End-to-End System Components Verified Successfully!")
