<<<<<<< HEAD
# CareGrid — Clinical Decision Support & 30-Day Readmission Risk System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_19-61DAFB.svg?style=flat&logo=react)](https://react.dev)
[![Vite](https://img.shields.io/badge/Bundler-Vite_8-646CFF.svg?style=flat&logo=vite)](https://vitejs.dev)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost_Ensemble-EB5424.svg?style=flat)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/XAI-SHAP_TreeExplainer-blue.svg?style=flat)](https://shap.readthedocs.io)
[![Pytest](https://img.shields.io/badge/Tests-30_Passing-brightgreen.svg?style=flat&logo=pytest)](https://docs.pytest.org)

**CareGrid** is a clinical decision-support system designed to forecast and mitigate unplanned 30-day hospital readmissions. Powered by a soft-voting ensemble machine learning pipeline (XGBoost + LightGBM) trained on 25,000 patient encounters, CareGrid pairs accurate predictive risk scoring with individualized **Explainable AI (SHAP)** feature attributions, automated **Medical PDF intake**, and multi-tier **Role-Based Access Control (RBAC)** for clinical nurses, doctors, sub-administrators, and system admins.

---

## Key Capabilities & Features

### 1. Dual-Channel Clinical Risk Prediction
- **Continuous Calibrated Probability**: High-precision risk percentage (0.0% – 100.0%) calibrated to population readmission rates.
- **Prominent Binary Verdict**: Unmistakable **`30-Day Readmission Predicted: YES`** *(Red Badge for $\ge 52.3\%$ Cutoff)* or **`NO`** *(Green Badge for $< 52.3\%$ Cutoff)*.
- **Acuity Stratification**: High Risk ($\ge 52.3\%$), Moderate Risk ($35.0\% - 52.3\%$), and Low Risk ($< 35.0\%$).
- **Calibrated Semicircular Risk Gauge**: Interactive radial gauge with clinical zone markings and indicator needle.
- **Summary KPI Metric Strip**: 4-card overview displaying Binary Outcome, Probability, Decision Cutoff Threshold, and Acuity Tier.

### 2. Generative AI Clinical Summary & Transition Brief
- **Automated Clinical Synthesis**: When the ML ensemble generates a prediction and SHAP values, a GenAI engine (Gemini / Groq with high-fidelity clinical synthesis fallback) automatically generates an executive clinical summary (e.g. *“This patient has been identified as HIGH RISK (72.0%) for readmission. Previous inpatient utilization and medication burden are among the factors contributing to the elevated risk.”*).
- **1-Click Copy & Download**: Clinicians can copy the synthesized narrative directly or download a complete formatted **Patient Clinical Readmission Brief (`.txt`)** for EHR transition-of-care documentation.

### 3. Explainable AI (XAI) with SHAP
- **Local Feature Attributions**: Computes exact positive (risk-increasing) and negative (risk-reducing) SHAP impacts for each patient encounter.
- **SHAP Waterfall & Impact Bar Charts**: Visual breakdown of patient-specific clinical risk drivers.
- **5-Domain Vulnerability Radar**: Evaluates Prior Utilization, Polypharmacy, Inpatient Stay Acuity, Chronic Disease Complexity, and Age Vulnerability.
- **Cohort Benchmarking**: Compares the patient's individual clinical parameters (LOS, medications, prior admissions) against hospital population medians and 90th percentile high-risk thresholds.
- **Actionable Discharge Protocols**: Generates tailored transition-of-care recommendations (e.g., 48h phone follow-up, home health aide, pharmacist medication reconciliation).

### 4. Smart Clinical PDF Intake & Edge Case Guardrails
- **Automated Text Extraction**: Rule-based & regex parser extracting all 16 clinical parameters from PDF discharge summaries in milliseconds.
- **Medical Report Verification**: Rejects non-medical documents (invoices, ID cards, general forms) with clear clinical warning alerts.
- **Partial Report Recovery**:
  - **Option A (Manual Completion)**: Fills extracted parameters and highlights missing inputs for nurse review.
  - **Option B (Direct Fast Prediction)**: Automatically imputes missing parameters using population medians and runs instant prediction.
- **Physiological Value Clamping**: Bounds out-of-range numerical values (e.g. days in hospital clamped to $[1, 14]$) to safeguard model stability.

### 4. Multi-Tier Authentication & Sub-Admin Role Management
- **Security**: SQLite database (`caregrid.db`) managed via SQLAlchemy ORM, **Bcrypt** salted password hashing, and 8-hour JWT Bearer tokens.
- **Role Hierarchy**:
  - **Super Administrator (`admin`)**: Full unrestricted system control, user provisioning, password editing, and account deletion.
  - **Sub-Administrator (`sub_admin`)**: Provisioned with custom editable designations (e.g., `Doctor`, `Diagnostic Lead`, `Nurse Supervisor`) and selective capability scopes (`can_manage_users`, `can_audit_reports`). Guardrails prevent sub-admins from modifying Super Admins.
  - **Medical Doctor (`doctor`)** & **Clinical Nurse (`nurse`)**: Standard clinical users authorized for patient intake, PDF uploads, and prediction serving.
- **Admin Password Control**: Plain assigned credentials visible to administrators in the directory with an **Eye toggle (`👁️`)**, **1-Click Copy (`📋`)**, interactive **Password Edit Modal**, and **Account Deletion**.

---

## System Architecture

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                     CAREGRID SYSTEM                         │
                                    └─────────────────────────────────────────────────────────────┘

       ┌───────────────────────────────┐                             ┌───────────────────────────────┐
       │   Staff Login Portal (/login) │                             │   Admin Portal (/admin-login) │
       └──────────────┬────────────────┘                             └──────────────┬────────────────┘
                      │                                                             │
                      ▼                                                             ▼
       ┌───────────────────────────────┐                             ┌───────────────────────────────┐
       │ Clinical Dashboard (/)        │                             │ Admin Management (/admin)     │
       │ - Manual Patient Intake Form  │                             │ - Sub-Admin Provisioning      │
       │ - PDF Report Auto-Extractor   │                             │ - Editable Designations       │
       │ - Binary YES/NO & Risk Gauge  │                             │ - Password Visibility (Eye)   │
       │ - SHAP Explanations & Radar   │                             │ - Edit Passwords & Deletion   │
       └──────────────┬────────────────┘                             └──────────────┬────────────────┘
                      │                                                             │
                      │  HTTP REST (JWT Auth Bearer Token)                          │
                      ▼                                                             ▼
       ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
       │                                  FASTAPI BACKEND SERVER                                     │
       │                                                                                             │
       │  /predict          /upload-pdf          /auth/login, me          /admin/users/*             │
       └──────────────┬───────────────────────────────┬─────────────────────────────┬────────────────┘
                      │                               │                             │
                      ▼                               ▼                             ▼
       ┌───────────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
       │     ML INFERENCE PIPELINE     │ │   PDF EXTRACTION ENGINE   │ │    SQLITE DB & AUTH LAYER │
       │ - readmission_model_final.pkl │ │ - Regex Clinical Parser   │ │ - caregrid.db (SQLAlchemy)│
       │ - Soft-Voting Ensemble        │ │ - Non-Medical Validator   │ │ - Bcrypt Password Hashing │
       │ - SHAP TreeExplainer          │ │ - Option A/B Partial Flow │ │ - JWT Token Validation    │
       └───────────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
```
=======
# CareGrid — Hospital Readmission Prediction & Clinical Decision Support

A clinical decision-support system that predicts 30-day hospital readmission risk using an optimized machine learning ensemble (XGBoost + LightGBM) and provides grounded medical guideline severity analysis via RAG.

> 📖 **Full Technical Architecture & Workflow Document**: For an in-depth, step-by-step breakdown of the ML pipeline, feature engineering, threshold calibration, RAG engine, and UI architecture, see **[`PROJECT_WORKFLOW.md`](file:///c:/Users/Asus/OneDrive/Documents/Desktop/projects/Hospital_Readmission_Pred/PROJECT_WORKFLOW.md)**.
>>>>>>> 642687495a4825e9c55ae9978d0c0a4d46b8571e

---

## Project Structure

```
Hospital_Readmission/
├── backend/
│   ├── auth.py                   ← Bcrypt hashing, JWT tokens, RBAC dependencies, seed admin
│   ├── database.py               ← SQLite SQLAlchemy engine, session maker, schema auto-migration
│   ├── main.py                   ← FastAPI REST API application, routes mounting, CORS
│   ├── models_db.py              ← SQLAlchemy User model (username, role, designation, permissions)
│   ├── pdf_extractor.py          ← 16-field regex extractor, medical validation, edge case handler
│   ├── routes_admin.py           ← Admin endpoints (user listing, provisioning, password edit, delete)
│   ├── routes_auth.py            ← Auth endpoints (/auth/login, /auth/logout, /auth/me)
│   └── train.py                  ← ML training pipeline orchestration
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DischargeProtocol.jsx  ← Transition-of-care checklist & discharge recommendations
│   │   │   ├── PatientForm.jsx        ← 16-parameter manual intake form with preset defaults
│   │   │   ├── PdfUploadPanel.jsx     ← Drag-and-drop PDF extraction with Option A/B recovery
│   │   │   ├── PredictionResult.jsx   ← Binary YES/NO badge, KPI strip, SHAP chart, Radar
│   │   │   ├── ProtectedRoute.jsx     ← Route authorization guard for nurses & admins
│   │   │   └── RiskGauge.jsx          ← Semicircular SVG risk gauge with YES/NO status tag
│   │   ├── context/
│   │   │   └── AuthContext.jsx        ← Global auth state, session sync, login/logout handlers
│   │   ├── pages/
│   │   │   ├── AdminDashboardPage.jsx ← Sub-admin creation, designations, password view/edit, delete
│   │   │   ├── AdminLoginPage.jsx     ← Direct-URL administrator login portal (/admin-login)
│   │   │   ├── ClinicalDashboardPage.jsx ← Primary prediction & XAI dashboard (/)
│   │   │   └── StaffLoginPage.jsx     ← Clinical staff login portal (/login)
│   │   ├── services/
│   │   │   └── api.js                 ← Axios API client with automatic JWT bearer injection
│   │   ├── App.jsx                    ← React Router route configuration
│   │   ├── index.css                  ← Tailwind CSS utility styling & print stylesheet
│   │   └── main.jsx                   ← React application entry point
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   ├── readmission_model_final.pkl    ← Serialized soft-voting ensemble pipeline (XGBoost + LightGBM)
│   ├── readmission_model_baseline.pkl ← Preserved baseline model
│   └── model_metadata.json            ← Optimal hyper-parameters, MCC cutoff threshold, gains table
│
├── dataset/
│   └── hospital_readmissions.csv      ← 25,000 historical patient encounters
│
├── uploads/                           ← Sample test documents for validation
│   ├── sample_patient_report_high_risk.pdf
│   ├── sample_patient_report_low_risk.pdf
│   └── sample_invoice_non_medical.pdf
│
├── tests/
│   ├── test_auth.py                   ← 7 tests: login, RBAC, sub-admin creation, password edit, delete
│   ├── test_full_system_e2e.py        ← Comprehensive end-to-end integration workflow simulation
│   ├── test_model.py                  ← 15 tests: ML edge cases, clamping, ranges, determinism
│   ├── test_pdf_extractor.py          ← 5 tests: full report, partial report, non-medical rejection
│   └── test_xai.py                    ← 2 tests: SHAP explanation generation & fallback handling
│
├── .env.example                       ← Clean environment configuration template
├── conftest.py                        ← Pytest root path configuration
└── requirements.txt                   ← Python dependencies
```

---

## Clinical Feature Set (16 Parameters)

The machine learning ensemble evaluates 16 clinical parameters stratified by their predictive weight:

| Priority Tier | Feature Key | Data Type / Values | Clinical Description |
| :--- | :--- | :--- | :--- |
| 🔴 **Tier 1: Core Driver** | `n_inpatient` | Integer ($0 - 15$) | Number of inpatient admissions in the preceding 12 months *(#1 Risk Factor)* |
| 🔴 **Tier 1: Core Driver** | `n_emergency` | Integer ($0 - 15$) | Number of emergency department visits in the preceding 12 months |
| 🔴 **Tier 1: Core Driver** | `time_in_hospital`| Integer ($1 - 14$) | Inpatient length of stay in days |
| 🔴 **Tier 1: Core Driver** | `n_medications` | Integer ($1 - 81$) | Total number of distinct active medications prescribed during stay |
| 🔴 **Tier 1: Core Driver** | `age` | Ordinal Brackets | Patient age range: `[0-10)`, `[10-20)`, ..., `[70-80)`, `[80-90)`, `[90-100)` |
| 🔴 **Tier 1: Core Driver** | `diag_1` | Categorical | Primary diagnosis: `Circulatory`, `Diabetes`, `Respiratory`, `Digestive`, etc. |
| 🔴 **Tier 1: Core Driver** | `n_outpatient` | Integer ($0 - 15$) | Outpatient clinical encounters in the preceding 12 months |
| 🔴 **Tier 1: Core Driver** | `diabetes_med` | `yes` / `no` | Active glycemic / diabetes management medication during stay |
| 🟡 **Tier 2: Refiner** | `n_lab_procedures`| Integer ($1 - 132$) | Total count of diagnostic lab tests performed during hospital stay |
| 🟡 **Tier 2: Refiner** | `n_procedures` | Integer ($0 - 6$) | Total count of surgical / clinical procedures performed |
| 🟡 **Tier 2: Refiner** | `medical_specialty`| Categorical | Admitting specialty: `Cardiology`, `InternalMedicine`, `Surgery`, etc. |
| 🟡 **Tier 2: Refiner** | `diag_2` | Categorical | Secondary diagnosis / primary comorbidity |
| 🟡 **Tier 2: Refiner** | `diag_3` | Categorical | Tertiary comorbidity |
| 🟡 **Tier 2: Refiner** | `change` | `yes` / `no` | Whether dosage or class of medication was titrated during admission |
| 🟡 **Tier 2: Refiner** | `glucose_test` | `normal` / `high` / `no` | Fasting serum glucose test result |
| 🟡 **Tier 2: Refiner** | `A1Ctest` | `normal` / `high` / `no` | Glycated hemoglobin (HbA1c) diagnostic test result |

---

## API Endpoints Reference

### Authentication & Session Management
- `POST /auth/login`: Authenticate with username and password; returns JWT access token and user metadata.
- `POST /auth/logout`: Invalidate user session.
- `GET /auth/me`: Retrieve current logged-in user profile and permissions *(Requires JWT)*.

### Clinical Predictions & Medical PDF Intake
- `POST /predict`: Submit patient data (16 parameters); returns readmission probability, binary verdict, SHAP attributions, radar scores, and discharge protocol *(Requires JWT)*.
- `POST /upload-pdf`: Upload medical PDF report (`multipart/form-data`); returns extracted features, field counts, and validation status *(Requires JWT)*.

### Administrative Management (Super Admins & Sub-Admins)
- `GET /admin/users`: List all staff accounts, designations, roles, permissions, and visible initial passwords.
- `POST /admin/users`: Provision a new account (`nurse`, `doctor`, `sub_admin`, or `admin`).
- `PATCH /admin/users/{id}/password`: Edit / reset a user's password.
- `PATCH /admin/users/{id}/status`: Activate or deactivate a user account.
- `DELETE /admin/users/{id}`: Permanently delete a user account *(Super admin self-deletion protected)*.

---

## Quickstart & Installation

### 1. Prerequisites
- **Python**: Version 3.11 or higher
- **Node.js**: Version 18 or higher (with `npm`)

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/Vickieee19/Hospital_Readmission.git
cd Hospital_Readmission

# Create and activate Python virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / CMD
# source .venv/bin/activate     # macOS / Linux

# Install backend dependencies
pip install -r requirements.txt

# Start the FastAPI server (Runs on http://localhost:8000)
python backend/main.py
```

### 3. Frontend Setup
```bash
# Open a second terminal and navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Launch Vite development server (Runs on http://localhost:5173)
npm run dev
```

---

## Default User Accounts

The database seeds a default Super Administrator account automatically upon initial startup:

| Role | Username | Password | Access URL | Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| **Super Admin** | `admin` | `Admin@123` | `http://localhost:5173/admin-login` | Full system control, Sub-Admin provisioning, password view/edit, user deletion |
| **Clinical Staff** | *(Create via Admin)* | *(Assigned)* | `http://localhost:5173/login` | Patient intake, PDF uploads, prediction scoring, XAI reports |

---

## Automated Test Suite

CareGrid includes 30 unit, edge case, and end-to-end integration tests covering:
- Baseline & optimized ML model inference determinism
- Continuous risk probability and binary classification
- SHAP TreeExplainer feature attributions
- PDF text extraction, partial recovery, and non-medical file rejection
- SQLite database schema migrations & Bcrypt authentication
- Sub-Admin provisioning and permission boundary enforcement

```bash
# Run all 30 tests with verbose output
pytest tests/ -v
```

```
============================== 30 passed in 12.44s ==============================
```

---

## License & Clinical Disclaimer

**Clinical Disclaimer**: This application is a decision-support and clinical triage tool designed to assist healthcare professionals in identifying patients vulnerable to readmission. Predictions and SHAP feature attributions represent statistical associations and are not a substitute for clinical judgment.
