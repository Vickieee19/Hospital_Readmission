from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pdf_extractor import extract_patient_data_from_text, ALL_16_FIELDS, is_valid_medical_report


def test_extract_patient_data_full_report():
    sample_text = """
    DISCHARGE SUMMARY
    Patient: John Doe, 74 years old
    Admitting Specialty: Cardiology
    Length of Stay: 5 days
    Primary Diagnosis: Acute Myocardial Infarction / Circulatory Disease
    Secondary Diagnosis: Type 2 Diabetes Mellitus
    Tertiary Diagnosis: Chronic Kidney Disease / Other
    
    Hospital Course & Interventions:
    Total lab procedures performed: 42
    Clinical procedures performed: 2
    Active medications prescribed: 16 (including Metformin and Lisinopril)
    Medication change during stay: Yes, insulin glargine initiated and titrated.
    
    Prior History:
    Prior inpatient admissions: 1 in past 12 months
    Prior emergency visits: 1
    Prior outpatient visits: 2
    
    Lab Results:
    Fasting Blood Glucose: 215 mg/dL (High)
    HbA1c: 8.6% (High)
    Diabetes medication: Yes
    """

    res = extract_patient_data_from_text(sample_text)

    assert res["is_medical_report"] is True
    assert res["is_partial"] is False
    assert res["error_type"] is None
    assert res["extracted_fields_count"] == 16
    assert res["missing_fields_count"] == 0

    patient = res["extracted_patient"]
    assert patient["age"] == "[70-80)"
    assert patient["time_in_hospital"] == 5
    assert patient["n_lab_procedures"] == 42
    assert patient["n_procedures"] == 2
    assert patient["n_medications"] == 16
    assert patient["n_inpatient"] == 1
    assert patient["n_emergency"] == 1
    assert patient["n_outpatient"] == 2
    assert patient["medical_specialty"] == "Cardiology"
    assert patient["diag_1"] == "Circulatory"
    assert patient["diag_2"] == "Diabetes"
    assert patient["diag_3"] == "Other"
    assert patient["glucose_test"] == "high"
    assert patient["A1Ctest"] == "high"
    assert patient["change"] == "yes"
    assert patient["diabetes_med"] == "yes"


def test_extract_patient_data_partial_report():
    """Test partial document containing only lab & glucose data."""
    partial_text = """
    OUTPATIENT LAB REPORT
    Patient: Jane Smith
    Fasting Glucose: 240 mg/dL (High)
    HbA1c Test: 9.1% (High)
    Total lab procedures: 12
    Active medications: 6
    """

    res = extract_patient_data_from_text(partial_text)

    assert res["is_medical_report"] is True
    assert res["is_partial"] is True
    assert res["error_type"] is None
    assert res["extracted_fields_count"] >= 4
    assert res["missing_fields_count"] > 0
    assert "time_in_hospital" in res["missing_fields_list"]
    assert "diag_1" in res["missing_fields_list"]
    assert "glucose_test" in res["extracted_fields_list"]
    assert "A1Ctest" in res["extracted_fields_list"]


def test_extract_patient_data_irrelevant_document():
    """Test non-clinical PDF document (invoice, resume, receipt) is strictly rejected."""
    irrelevant_text = """
    ACME CORPORATION INVOICE
    Invoice Number: #INV-98213
    Date: 2026-08-18
    Bill To: Global Logistics Inc.
    Item 1: Enterprise Cloud Hosting Subscription ($1,200.00)
    Item 2: Software Maintenance Service ($350.00)
    Total Due: $1,550.00
    Payment Terms: Net 30 Days. Thank you for your business.
    """

    res = extract_patient_data_from_text(irrelevant_text)

    assert res["is_medical_report"] is False
    assert res["error_type"] == "NON_MEDICAL_DOCUMENT"
    assert res["extracted_fields_count"] == 0
    assert res["missing_fields_count"] == 16
    assert "not recognized as a medical report" in res["status_message"]


def test_extract_patient_data_empty_or_scanned():
    """Test empty string or whitespace from un-OCRed scanned PDF."""
    res_empty = extract_patient_data_from_text("")
    assert res_empty["is_medical_report"] is False
    assert res_empty["error_type"] == "EMPTY_OR_SCANNED_PDF"

    res_spaces = extract_patient_data_from_text("   \n\t  ")
    assert res_spaces["is_medical_report"] is False
    assert res_spaces["error_type"] == "EMPTY_OR_SCANNED_PDF"


def test_extract_patient_data_out_of_range_clamping():
    """Test that out-of-range clinical values are safely clamped."""
    extreme_text = """
    HOSPITAL DISCHARGE SUMMARY
    Patient Age: 82 years old
    Length of Stay: 95 days
    Total lab procedures: 400
    Clinical procedures: 25
    Active medications: 150
    """

    res = extract_patient_data_from_text(extreme_text)
    patient = res["extracted_patient"]

    assert res["is_medical_report"] is True
    assert patient["age"] == "[80-90)"
    assert patient["time_in_hospital"] == 30  # clamped to max 30
    assert patient["n_lab_procedures"] == 150  # clamped to max 150
    assert patient["n_procedures"] == 10  # clamped to max 10
    assert patient["n_medications"] == 80  # clamped to max 80
