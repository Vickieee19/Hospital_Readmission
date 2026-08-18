from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.database import Base, SessionLocal, engine, init_db_schema
from backend.auth import seed_default_admin
from backend.models_db import User

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_db_schema()
    with SessionLocal() as db:
        seed_default_admin(db)
    yield


def test_default_admin_login():
    """Test login with default seeded administrator credentials."""
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "Admin@123"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "admin"


def test_invalid_credentials_rejected():
    """Test that incorrect credentials return 401 Unauthorized."""
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "WrongPassword123"},
    )
    assert response.status_code == 401


def test_predict_requires_authentication():
    """Test that /predict rejects unauthenticated requests with 401."""
    patient_payload = {
        "age": "[70-80)",
        "time_in_hospital": 5,
        "n_lab_procedures": 40,
        "n_procedures": 2,
        "n_medications": 15,
        "n_outpatient": 0,
        "n_inpatient": 1,
        "n_emergency": 0,
        "medical_specialty": "Cardiology",
        "diag_1": "Circulatory",
        "diag_2": "Diabetes",
        "diag_3": "Other",
        "glucose_test": "high",
        "A1Ctest": "high",
        "change": "yes",
        "diabetes_med": "yes",
    }
    response = client.post("/predict", json=patient_payload)
    assert response.status_code == 401


def test_admin_create_nurse_and_nurse_workflow():
    """
    Test end-to-end admin creation of a nurse account,
    nurse login, prediction execution, and RBAC restriction on admin routes.
    """
    # 1. Login as admin
    admin_login_res = client.post(
        "/auth/login",
        json={"username": "admin", "password": "Admin@123"},
    )
    admin_token = admin_login_res.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 2. Admin creates nurse account
    nurse_username = "nurse_workflow_test"
    nurse_pwd = "NursePassword123"
    with SessionLocal() as db:
        existing = db.query(User).filter(User.username == nurse_username).first()
        if existing:
            db.delete(existing)
            db.commit()

    create_res = client.post(
        "/admin/users",
        headers=admin_headers,
        json={
            "username": nurse_username,
            "full_name": "Sarah Connor, RN",
            "designation": "Staff Nurse",
            "password": nurse_pwd,
            "role": "nurse",
        },
    )
    assert create_res.status_code == 201, create_res.text

    # 3. Nurse logs in
    nurse_login_res = client.post(
        "/auth/login",
        json={"username": nurse_username, "password": nurse_pwd},
    )
    assert nurse_login_res.status_code == 200
    nurse_token = nurse_login_res.json()["access_token"]
    nurse_headers = {"Authorization": f"Bearer {nurse_token}"}

    # 4. Nurse checks profile
    me_res = client.get("/auth/me", headers=nurse_headers)
    assert me_res.status_code == 200
    assert me_res.json()["role"] == "nurse"

    # 5. Nurse accesses /predict (should SUCCEED)
    patient_payload = {
        "age": "[60-70)",
        "time_in_hospital": 4,
        "n_lab_procedures": 30,
        "n_procedures": 1,
        "n_medications": 10,
        "n_outpatient": 0,
        "n_inpatient": 0,
        "n_emergency": 0,
        "medical_specialty": "InternalMedicine",
        "diag_1": "Circulatory",
        "diag_2": "Other",
        "diag_3": "Other",
        "glucose_test": "normal",
        "A1Ctest": "normal",
        "change": "no",
        "diabetes_med": "no",
    }
    pred_res = client.post("/predict", headers=nurse_headers, json=patient_payload)
    assert pred_res.status_code == 200, pred_res.text
    pred_data = pred_res.json()
    assert "probability" in pred_data
    assert "shap_summary" in pred_data

    # 6. Nurse attempts to access /admin/users (should be 403 FORBIDDEN)
    forbidden_res = client.get("/admin/users", headers=nurse_headers)
    assert forbidden_res.status_code == 403


def test_sub_admin_creation_and_selective_role_workflow():
    """
    Test that Super Admin can create a Sub-Admin (e.g. Doctor) with selective roles,
    and Sub-Admin can add regular nurses/doctors, but cannot create or delete Super Admins.
    """
    # 1. Login as Super Admin
    admin_login_res = client.post(
        "/auth/login",
        json={"username": "admin", "password": "Admin@123"},
    )
    admin_token = admin_login_res.json()["access_token"]
    admin_id = admin_login_res.json()["user"]["id"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 2. Super Admin creates Sub-Admin with Doctor designation & selective roles
    sub_admin_user = "dr_chen_lead"
    sub_admin_pwd = "DoctorPass@123"
    with SessionLocal() as db:
        existing = db.query(User).filter(User.username == sub_admin_user).first()
        if existing:
            db.delete(existing)
            db.commit()

    sub_res = client.post(
        "/admin/users",
        headers=admin_headers,
        json={
            "username": sub_admin_user,
            "full_name": "Dr. Robert Chen, MD",
            "designation": "Doctor / Department Lead",
            "password": sub_admin_pwd,
            "role": "sub_admin",
            "permissions": "create_users,audit_reports",
        },
    )
    assert sub_res.status_code == 201, sub_res.text
    sub_data = sub_res.json()
    assert sub_data["role"] == "sub_admin"
    assert sub_data["designation"] == "Doctor / Department Lead"
    assert "create_users" in sub_data["permissions"]

    # 3. Sub-Admin logs in
    sub_login_res = client.post(
        "/auth/login",
        json={"username": sub_admin_user, "password": sub_admin_pwd},
    )
    assert sub_login_res.status_code == 200
    sub_token = sub_login_res.json()["access_token"]
    sub_headers = {"Authorization": f"Bearer {sub_token}"}

    # 4. Sub-Admin creates a new clinical user (Nurse) -> SUCCESS
    sub_nurse = "nurse_by_subadmin"
    with SessionLocal() as db:
        existing = db.query(User).filter(User.username == sub_nurse).first()
        if existing:
            db.delete(existing)
            db.commit()

    nurse_res = client.post(
        "/admin/users",
        headers=sub_headers,
        json={
            "username": sub_nurse,
            "full_name": "Anna Karenina, RN",
            "designation": "Staff Nurse",
            "password": "NursePass@123",
            "role": "nurse",
        },
    )
    assert nurse_res.status_code == 201

    # 5. Sub-Admin tries to create a Super Admin -> FORBIDDEN (403)
    illegal_admin_res = client.post(
        "/admin/users",
        headers=sub_headers,
        json={
            "username": "illegal_admin",
            "full_name": "Fake Admin",
            "password": "Password123",
            "role": "admin",
        },
    )
    assert illegal_admin_res.status_code == 403

    # 6. Sub-Admin tries to delete Super Admin -> FORBIDDEN (403)
    illegal_del_res = client.delete(f"/admin/users/{admin_id}", headers=sub_headers)
    assert illegal_del_res.status_code == 403


def test_admin_view_and_edit_password_and_delete_user():
    """
    Test that admin can view initial passwords, edit passwords, and delete users.
    """
    # 1. Login as admin
    admin_login_res = client.post(
        "/auth/login",
        json={"username": "admin", "password": "Admin@123"},
    )
    admin_token = admin_login_res.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 2. Create test user
    test_user = "nurse_test_edit"
    with SessionLocal() as db:
        existing = db.query(User).filter(User.username == test_user).first()
        if existing:
            db.delete(existing)
            db.commit()

    client.post(
        "/admin/users",
        headers=admin_headers,
        json={
            "username": test_user,
            "full_name": "Test Nurse",
            "designation": "Staff Nurse",
            "password": "OldPassword123",
            "role": "nurse",
        },
    )

    # 3. View users and verify initial_password is visible to admin
    users_res = client.get("/admin/users", headers=admin_headers)
    assert users_res.status_code == 200
    user_item = next(u for u in users_res.json() if u["username"] == test_user)
    assert user_item["initial_password"] == "OldPassword123"
    target_id = user_item["id"]

    # 4. Admin edits password
    new_pwd = "NewSecurePassword456"
    pwd_update_res = client.patch(
        f"/admin/users/{target_id}/password",
        headers=admin_headers,
        json={"new_password": new_pwd},
    )
    assert pwd_update_res.status_code == 200
    assert pwd_update_res.json()["initial_password"] == new_pwd

    # 5. Verify nurse can now log in with the new password
    new_login_res = client.post(
        "/auth/login",
        json={"username": test_user, "password": new_pwd},
    )
    assert new_login_res.status_code == 200

    # 6. Admin deletes user
    delete_res = client.delete(f"/admin/users/{target_id}", headers=admin_headers)
    assert delete_res.status_code == 200
    assert delete_res.json()["status"] == "success"

    # 7. Verify deleted user cannot log in
    deleted_login_res = client.post(
        "/auth/login",
        json={"username": test_user, "password": new_pwd},
    )
    assert deleted_login_res.status_code == 401


def test_admin_cannot_delete_self():
    """Test safety check: admin cannot delete their own active account."""
    admin_login_res = client.post(
        "/auth/login",
        json={"username": "admin", "password": "Admin@123"},
    )
    admin_token = admin_login_res.json()["access_token"]
    admin_id = admin_login_res.json()["user"]["id"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    delete_res = client.delete(f"/admin/users/{admin_id}", headers=admin_headers)
    assert delete_res.status_code == 400
