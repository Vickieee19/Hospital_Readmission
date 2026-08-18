"""
backend/pdf_extractor.py
────────────────────────
Resilient Rule-Based & Regex Clinical Data Extractor for PDF Reports.
Parses unstructured discharge summaries, lab reports, and clinical notes to extract
the 16 clinical parameters required for 30-day hospital readmission prediction.

Document Verification:
- Validates whether an uploaded PDF is genuinely a clinical/medical report.
- Strictly rejects non-medical documents (invoices, receipts, resumes, academic papers, general documents).
- Ignores invalid documents and alerts the user with a clinical warning.
- Handles partial clinical reports by auto-filling detected fields and listing missing fields for manual entry.
- Sanitizes and clamps out-of-range clinical values.
"""

from __future__ import annotations

import re
from typing import Any

VALID_AGES = ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]

ALL_16_FIELDS = [
    "age",
    "time_in_hospital",
    "n_lab_procedures",
    "n_procedures",
    "n_medications",
    "n_outpatient",
    "n_inpatient",
    "n_emergency",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "glucose_test",
    "A1Ctest",
    "change",
    "diabetes_med",
]

FIELD_LABELS_MAP: dict[str, str] = {
    "age": "Age Bracket",
    "time_in_hospital": "Length of Stay (Days)",
    "n_lab_procedures": "Lab Procedures Count",
    "n_procedures": "Clinical Procedures Count",
    "n_medications": "Active Prescribed Medications",
    "n_outpatient": "Prior Outpatient Visits",
    "n_inpatient": "Prior Inpatient Admissions",
    "n_emergency": "Prior Emergency Room Visits",
    "medical_specialty": "Admitting Medical Specialty",
    "diag_1": "Primary Diagnosis",
    "diag_2": "Secondary Diagnosis",
    "diag_3": "Tertiary Diagnosis",
    "glucose_test": "Fasting Glucose Lab Result",
    "A1Ctest": "HbA1c Lab Result",
    "change": "Inpatient Medication Modification",
    "diabetes_med": "Diabetes Medication Prescribed",
}

# Strong Clinical Document Context Markers
CLINICAL_CONTEXT_PATTERNS = [
    r"\bdischarge\s+summary\b",
    r"\bclinical\s+(?:note|summary|report|record)\b",
    r"\bmedical\s+(?:report|record|history|assessment|chart)\b",
    r"\blaboratory\s+(?:report|results?|findings?)\b",
    r"\blab\s+(?:report|results?|panel)\b",
    r"\bpatient\s+(?:name|age|id|mrn|demographics|history)\b",
    r"\bhospital\s+(?:course|stay|admission|encounter)\b",
    r"\badmission\s+(?:note|date|summary)\b",
    r"\bphysician\s+(?:notes?|signature|order)\b",
    r"\bprimary\s+diagnosis\b",
    r"\bsecondary\s+diagnosis\b",
    r"\bdiagnostic\s+(?:impression|findings?)\b",
    r"\bactive\s+medications?\b",
    r"\bprescribed\s+medications?\b",
    r"\bhistory\s+of\s+present\s+illness\b",
    r"\bcare\s+plan\b",
    r"\btreatment\s+plan\b",
]

# Non-Medical Disqualifying Indicators
NON_MEDICAL_PATTERNS = [
    r"\binvoice\b",
    r"\btax\s+invoice\b",
    r"\bbill\s+to\b",
    r"\btotal\s+due\b",
    r"\bpayment\s+terms\b",
    r"\bpurchase\s+order\b",
    r"\breceipt\b",
    r"\bcurriculum\s+vitae\b",
    r"\bresume\b",
    r"\bemployment\s+history\b",
    r"\bacademic\s+transcript\b",
    r"\bbank\s+statement\b",
    r"\baccount\s+balance\b",
    r"\bwire\s+transfer\b",
]


def is_valid_medical_report(text: str) -> tuple[bool, str]:
    """
    Validate whether the document text has verifiable clinical/medical report characteristics.

    Returns
    -------
    (is_valid: bool, reason: str)
    """
    if not text or len(text.strip()) < 20:
        return False, "Document contains no readable text or is empty."

    # Check for strong clinical headers/markers
    clinical_matches = 0
    for pattern in CLINICAL_CONTEXT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            clinical_matches += 1

    # Check for non-medical markers
    non_medical_matches = 0
    for pattern in NON_MEDICAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            non_medical_matches += 1

    # Broad medical terminology keywords
    medical_keywords = [
        r"\bpatient\b", r"\bhospital\b", r"\bclinic\b", r"\bdiagnosis\b",
        r"\bmedication\b", r"\bmedications\b", r"\bprescribed\b", r"\btreatment\b",
        r"\bphysician\b", r"\bdoctor\b", r"\bglucose\b", r"\bhba1c\b",
        r"\bcardiology\b", r"\binpatient\b", r"\boutpatient\b", r"\badmission\b",
        r"\bdischarge\b", r"\blaboratory\b", r"\bvital\s+signs\b", r"\bheart\s+rate\b",
        r"\bblood\s+pressure\b", r"\bhypertension\b", r"\bdiabetes\b"
    ]
    keyword_count = sum(1 for kw in medical_keywords if re.search(kw, text, re.IGNORECASE))

    # Rejection: clearly a non-medical document
    if non_medical_matches >= 2 and clinical_matches == 0:
        return False, "Document appears to be a financial, commercial, or non-medical document."

    # Acceptance criteria: Must have either strong clinical context OR sufficient medical terminology
    if clinical_matches >= 1 or keyword_count >= 3:
        return True, "Valid clinical document structure confirmed."

    return False, "Document lacks standard clinical encounter or medical report structure."


def _match_age(text: str) -> str | None:
    """Extract age and map to categorical bracket."""
    for bracket in VALID_AGES:
        if bracket in text:
            return bracket

    patterns = [
        r"(?:age|patient\s+age)[\s:]+(\d{1,3})",
        r"(\d{1,3})\s*(?:years?\s+old|yo|y\.o\.)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                age_num = int(match.group(1))
                if 18 <= age_num <= 120:
                    if age_num < 50:
                        return "[40-50)"
                    elif age_num < 60:
                        return "[50-60)"
                    elif age_num < 70:
                        return "[60-70)"
                    elif age_num < 80:
                        return "[70-80)"
                    elif age_num < 90:
                        return "[80-90)"
                    else:
                        return "[90-100)"
            except (ValueError, IndexError):
                continue
    return None


def _match_int_field(patterns: list[str], text: str, min_val: int = 0, max_val: int = 200) -> int | None:
    """Extract an integer field using regex patterns, checking all matches and clamping to bounds."""
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                val = int(match.group(1))
                return max(min_val, min(val, max_val))
            except (ValueError, IndexError):
                continue
    return None


def _match_specialty(text: str) -> str | None:
    """Identify admitting/treating medical specialty using word boundaries."""
    if re.search(r"\bcardio\w*", text, re.IGNORECASE):
        return "Cardiology"
    if re.search(r"\binternal\s+med\w*|\binternist\b", text, re.IGNORECASE):
        return "InternalMedicine"
    if re.search(r"\bfamily\b|\bgeneral\s+practice\b|\bpcp\b", text, re.IGNORECASE):
        return "Family/GeneralPractice"
    if re.search(r"\bsurg\w*|\bortho\w*", text, re.IGNORECASE):
        return "Surgery"
    if re.search(r"\bemergency\b|\btrauma\b", text, re.IGNORECASE):
        return "Other"
    return None


def _match_diagnosis(text: str) -> str | None:
    """Map disease mentions to standard diagnostic category using word boundaries."""
    circ_patterns = [
        r"\bcirculatory\b", r"\bcardiac\b", r"\bheart\s+failure\b", r"\bchf\b",
        r"\bcoronary\b", r"\bcad\b", r"\bmyocardial\b", r"\binfarction\b",
        r"\bhypertension\b", r"\bhtn\b", r"\bstroke\b", r"\bcva\b",
        r"\barrhythmia\b", r"\batrial\s+fibrill"
    ]
    for p in circ_patterns:
        if re.search(p, text, re.IGNORECASE):
            return "Circulatory"

    diab_patterns = [
        r"\bdiabetes\b", r"\bdiabetic\b", r"\bhyperglycemia\b", r"\bdka\b",
        r"\bt2d\b", r"\bt1d\b", r"\btype\s+[12]\s+diabetes\b"
    ]
    for p in diab_patterns:
        if re.search(p, text, re.IGNORECASE):
            return "Diabetes"

    resp_patterns = [
        r"\brespiratory\b", r"\bcopd\b", r"\bpneumonia\b", r"\basthma\b",
        r"\bbronchitis\b", r"\bdyspnea\b", r"\bpulmonary\b", r"\brespiratory\s+failure\b"
    ]
    for p in resp_patterns:
        if re.search(p, text, re.IGNORECASE):
            return "Respiratory"

    digest_patterns = [
        r"\bdigestive\b", r"\bgastrointestinal\b", r"\bpancreatitis\b",
        r"\bcirrhosis\b", r"\bhepatitis\b", r"\bgastritis\b", r"\bcolitis\b",
        r"\bdiverticul", r"\bgi\s+bleed"
    ]
    for p in digest_patterns:
        if re.search(p, text, re.IGNORECASE):
            return "Digestive"

    injury_patterns = [
        r"\binjury\b", r"\btrauma\b", r"\bfracture\b", r"\bfall\b",
        r"\bcontusion\b", r"\blaceration\b", r"\bwound\b"
    ]
    for p in injury_patterns:
        if re.search(p, text, re.IGNORECASE):
            return "Injury"

    other_patterns = [
        r"\bneoplasm\b", r"\bcancer\b", r"\btumor\b", r"\brenal\b",
        r"\bkidney\b", r"\buti\b", r"\bsepsis\b", r"\binfection\b"
    ]
    for p in other_patterns:
        if re.search(p, text, re.IGNORECASE):
            return "Other"

    return None


def _match_glucose_test(text: str) -> str | None:
    """Determine glucose lab result category."""
    match = re.search(r"(?:glucose|blood\s+sugar|fasting\s+glucose|glu)[\s:]+(\d{2,3})", text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        return "high" if val > 200 else "normal"

    match_qual = re.search(r"(?:glucose|blood\s+sugar)[\s:\w]*\((high|normal|elevated|abnormal|no|negative)\)", text, re.IGNORECASE)
    if match_qual:
        res = match_qual.group(1).lower()
        if res in ["high", "elevated", "abnormal"]:
            return "high"
        if res == "normal":
            return "normal"
        return "no"

    match_qual2 = re.search(r"(?:glucose|blood\s+sugar)[\s:]+(high|normal|elevated|abnormal|no|negative)", text, re.IGNORECASE)
    if match_qual2:
        res = match_qual2.group(1).lower()
        if res in ["high", "elevated", "abnormal"]:
            return "high"
        if res == "normal":
            return "normal"
        return "no"

    if re.search(r"\bglucose\b|\bblood\s+sugar\b", text, re.IGNORECASE):
        return "normal"
    return None


def _match_a1c_test(text: str) -> str | None:
    """Determine HbA1c test category."""
    match = re.search(r"(?:hba1c|a1c|glycated\s+hemoglobin)(?:\s+test)?[\s:]+(\d{1,2}(?:\.\d{1,2})?)\s*%", text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return "high" if val > 8.0 else "normal"

    match_qual = re.search(r"(?:hba1c|a1c)[\s:\w%]*\((high|normal|elevated|abnormal|no|negative)\)", text, re.IGNORECASE)
    if match_qual:
        res = match_qual.group(1).lower()
        if res in ["high", "elevated", "abnormal"]:
            return "high"
        if res == "normal":
            return "normal"
        return "no"

    match_qual2 = re.search(r"(?:hba1c|a1c)[\s:]+(high|normal|elevated|abnormal|no|not\s+performed)", text, re.IGNORECASE)
    if match_qual2:
        res = match_qual2.group(1).lower()
        if res in ["high", "elevated", "abnormal"]:
            return "high"
        if res == "normal":
            return "normal"
        return "no"

    if re.search(r"\bhba1c\b|\ba1c\s+test\b|\bglycated\s+hemoglobin\b", text, re.IGNORECASE):
        return "normal"
    return None


def _match_medication_change(text: str) -> str | None:
    """Determine if medication was modified during encounter."""
    lower = text.lower()
    if any(k in lower for k in ["medication change: yes", "medication change during stay: yes", "medication changed", "dose adjusted", "dosage adjusted", "regimen modified", "titrated", "medication modified", "change in medication: yes"]):
        return "yes"
    if any(k in lower for k in ["medication change: no", "medication change during stay: no", "no change in medication", "regimen maintained", "dose unchanged"]):
        return "no"
    return None


def _match_diabetes_med(text: str) -> str | None:
    """Determine if patient is prescribed active diabetes medications."""
    lower = text.lower()
    if any(k in lower for k in ["diabetes med: yes", "diabetes medication: yes", "insulin", "metformin", "glipizide", "glimepiride", "glyburide", "sitagliptin", "empagliflozin", "liraglutide", "dapagliflozin", "semaglutide"]):
        return "yes"
    if any(k in lower for k in ["diabetes med: no", "no diabetes medication"]):
        return "no"
    return None


def extract_patient_data_from_text(raw_text: str | None) -> dict[str, Any]:
    """
    Parse clinical report text and return a diagnostic result dictionary.

    Returns
    -------
    dict with keys:
        is_medical_report       : bool (True if valid medical document)
        extracted_patient       : dict of matched field values
        extracted_fields_list   : list of matched field keys
        extracted_fields_count  : int
        missing_fields_list     : list of missing field keys
        missing_fields_count    : int
        missing_fields_labels   : list of human-readable missing field names
        is_partial              : bool (True if some fields found, but not all 16)
        error_type              : str | None ("EMPTY_OR_SCANNED_PDF", "NON_MEDICAL_DOCUMENT", or None)
        status_message          : str human-friendly explanation
    """
    # 1. Check for empty or unreadable text
    if not raw_text or len(raw_text.strip()) < 15:
        return {
            "is_medical_report": False,
            "extracted_patient": {},
            "extracted_fields_list": [],
            "extracted_fields_count": 0,
            "missing_fields_list": ALL_16_FIELDS,
            "missing_fields_count": len(ALL_16_FIELDS),
            "missing_fields_labels": [FIELD_LABELS_MAP[k] for k in ALL_16_FIELDS],
            "is_partial": False,
            "error_type": "EMPTY_OR_SCANNED_PDF",
            "status_message": (
                "The uploaded document contains no readable text. It may be a scanned image "
                "without an OCR text layer or an empty file. This document was not taken into consideration."
            ),
        }

    # 2. Strict Medical Document Verification
    is_medical, reason = is_valid_medical_report(raw_text)
    if not is_medical:
        return {
            "is_medical_report": False,
            "extracted_patient": {},
            "extracted_fields_list": [],
            "extracted_fields_count": 0,
            "missing_fields_list": ALL_16_FIELDS,
            "missing_fields_count": len(ALL_16_FIELDS),
            "missing_fields_labels": [FIELD_LABELS_MAP[k] for k in ALL_16_FIELDS],
            "is_partial": False,
            "error_type": "NON_MEDICAL_DOCUMENT",
            "status_message": (
                "Warning: The uploaded file is not recognized as a medical report. "
                "Only clinical discharge summaries, lab reports, and encounter notes are accepted. "
                "This document was not taken into consideration, and no patient fields were modified."
            ),
        }

    extracted: dict[str, Any] = {}
    found_fields: list[str] = []

    # 1. Age
    age = _match_age(raw_text)
    if age:
        extracted["age"] = age
        found_fields.append("age")

    # 2. Length of Stay (Time in Hospital) — Clamp to 1..30 days
    stay = _match_int_field(
        [
            r"(?:length\s+of\s+stay|time\s+in\s+hospital|hospital\s+stay|stay\s+duration|days\s+admitted|duration\s+of\s+stay)[\s:]+(\d+)\s*(?:days?)?",
            r"stayed\s+(\d+)\s+days?",
            r"admitted\s+for\s+(\d+)\s+days?",
        ],
        raw_text,
        min_val=1,
        max_val=30,
    )
    if stay is not None:
        extracted["time_in_hospital"] = stay
        found_fields.append("time_in_hospital")

    # 3. Lab procedures — Clamp to 0..150
    labs = _match_int_field(
        [
            r"(?:total\s+)?(?:lab\s+procedures?|number\s+of\s+lab\s+tests?|total\s+labs?|labs?\s+performed|lab\s+count)(?:\s+performed|\s+ordered|\s+count)?[\s:]+(\d+)",
            r"(\d+)\s+lab\s+procedures?",
        ],
        raw_text,
        min_val=0,
        max_val=150,
    )
    if labs is not None:
        extracted["n_lab_procedures"] = labs
        found_fields.append("n_lab_procedures")

    # 4. Inpatient procedures (excluding lab procedures) — Clamp to 0..10
    procs = _match_int_field(
        [
            r"(?:clinical|surgical|inpatient|non-lab)\s+procedures?(?:\s+performed|\s+done)?[\s:]+(\d+)",
            r"(?:number\s+of\s+procedures?|procedure\s+count)[\s:]+(\d+)",
            r"(?<!lab\s)procedures?(?:\s+performed|\s+done)?[\s:]+(\d+)",
            r"(\d+)\s+(?:clinical\s+|surgical\s+)?procedures?\s+performed",
        ],
        raw_text,
        min_val=0,
        max_val=10,
    )
    if procs is not None:
        extracted["n_procedures"] = procs
        found_fields.append("n_procedures")

    # 5. Medications — Clamp to 1..80
    meds = _match_int_field(
        [
            r"(?:active\s+|prescribed\s+|total\s+)?(?:medications?|medication\s+count|prescribed\s+meds?|total\s+meds?)(?:\s+prescribed|\s+count)?[\s:]+(\d+)",
            r"(\d+)\s+(?:active\s+)?medications?",
        ],
        raw_text,
        min_val=1,
        max_val=80,
    )
    if meds is not None:
        extracted["n_medications"] = meds
        found_fields.append("n_medications")

    # 6. Prior Outpatient visits — Clamp to 0..40
    outpatient = _match_int_field(
        [
            r"(?:outpatient\s+visits?|prior\s+outpatient|previous\s+outpatient)(?:\s+visits?|\s+encounters?)?[\s:]+(\d+)",
            r"(\d+)\s+prior\s+outpatient",
        ],
        raw_text,
        min_val=0,
        max_val=40,
    )
    if outpatient is not None:
        extracted["n_outpatient"] = outpatient
        found_fields.append("n_outpatient")

    # 7. Prior Inpatient admissions — Clamp to 0..25
    inpatient = _match_int_field(
        [
            r"(?:inpatient\s+admissions?|prior\s+inpatient|previous\s+admissions?|past\s+inpatient\s+stays?)(?:\s+admissions?|\s+visits?)?[\s:]+(\d+)",
            r"(\d+)\s+prior\s+inpatient",
        ],
        raw_text,
        min_val=0,
        max_val=25,
    )
    if inpatient is not None:
        extracted["n_inpatient"] = inpatient
        found_fields.append("n_inpatient")

    # 8. Prior Emergency visits — Clamp to 0..30
    er = _match_int_field(
        [
            r"(?:prior\s+|previous\s+|past\s+)?(?:emergency\s+room\s+visits?|emergency\s+department\s+visits?|emergency\s+visits?|er\s+visits?|emergency\s+encounters?)(?:\s+visits?|\s+encounters?)?[\s:]+(\d+)",
            r"(\d+)\s+prior\s+(?:emergency|er)",
        ],
        raw_text,
        min_val=0,
        max_val=30,
    )
    if er is not None:
        extracted["n_emergency"] = er
        found_fields.append("n_emergency")

    # 9. Specialty
    spec = _match_specialty(raw_text)
    if spec:
        extracted["medical_specialty"] = spec
        found_fields.append("medical_specialty")

    # 10. Primary Diagnosis
    primary_diag_match = re.search(
        r"(?:primary\s+diagnosis|admitting\s+diagnosis|principal\s+diagnosis|diag_1)[\s:]+([^\n\r,;.]+)",
        raw_text,
        re.IGNORECASE,
    )
    if primary_diag_match:
        diag1 = _match_diagnosis(primary_diag_match.group(1))
        if diag1:
            extracted["diag_1"] = diag1
            found_fields.append("diag_1")
    if "diag_1" not in extracted:
        general_diag = _match_diagnosis(raw_text)
        if general_diag:
            extracted["diag_1"] = general_diag
            found_fields.append("diag_1")

    # 11. Secondary & Tertiary Diagnoses
    sec_diag_match = re.search(
        r"(?:secondary\s+diagnosis|diag_2|comorbidity|additional\s+diagnosis)[\s:]+([^\n\r,;.]+)",
        raw_text,
        re.IGNORECASE,
    )
    if sec_diag_match:
        diag2 = _match_diagnosis(sec_diag_match.group(1))
        if diag2:
            extracted["diag_2"] = diag2
            found_fields.append("diag_2")

    tert_diag_match = re.search(
        r"(?:tertiary\s+diagnosis|diag_3)[\s:]+([^\n\r,;.]+)",
        raw_text,
        re.IGNORECASE,
    )
    if tert_diag_match:
        diag3 = _match_diagnosis(tert_diag_match.group(1))
        if diag3:
            extracted["diag_3"] = diag3
            found_fields.append("diag_3")

    # 12. Glucose Test
    glu = _match_glucose_test(raw_text)
    if glu:
        extracted["glucose_test"] = glu
        found_fields.append("glucose_test")

    # 13. A1C Test
    a1c = _match_a1c_test(raw_text)
    if a1c:
        extracted["A1Ctest"] = a1c
        found_fields.append("A1Ctest")

    # 14. Medication Change
    med_change = _match_medication_change(raw_text)
    if med_change:
        extracted["change"] = med_change
        found_fields.append("change")

    # 15. Diabetes Medication
    diab_med = _match_diabetes_med(raw_text)
    if diab_med:
        extracted["diabetes_med"] = diab_med
        found_fields.append("diabetes_med")

    # If document has medical keywords but 0 fields matched
    if len(found_fields) == 0:
        return {
            "is_medical_report": False,
            "extracted_patient": {},
            "extracted_fields_list": [],
            "extracted_fields_count": 0,
            "missing_fields_list": ALL_16_FIELDS,
            "missing_fields_count": len(ALL_16_FIELDS),
            "missing_fields_labels": [FIELD_LABELS_MAP[k] for k in ALL_16_FIELDS],
            "is_partial": False,
            "error_type": "NON_MEDICAL_DOCUMENT",
            "status_message": (
                "Document Not Applicable: Although some general terms were found, no valid patient encounter parameters "
                "could be extracted. This file was not taken into consideration."
            ),
        }

    # Determine missing fields
    missing_fields = [f for f in ALL_16_FIELDS if f not in extracted]
    missing_labels = [FIELD_LABELS_MAP[f] for f in missing_fields]
    is_partial = len(missing_fields) > 0

    if is_partial:
        status_msg = (
            f"Verified Medical Report: Successfully extracted {len(found_fields)} of 16 parameters. "
            f"{len(missing_fields)} field(s) were not detected in the document."
        )
    else:
        status_msg = f"Verified Medical Report: Successfully extracted all {len(found_fields)} clinical parameters."

    return {
        "is_medical_report": True,
        "extracted_patient": extracted,
        "extracted_fields_list": found_fields,
        "extracted_fields_count": len(found_fields),
        "missing_fields_list": missing_fields,
        "missing_fields_count": len(missing_fields),
        "missing_fields_labels": missing_labels,
        "is_partial": is_partial,
        "error_type": None,
        "status_message": status_msg,
    }
