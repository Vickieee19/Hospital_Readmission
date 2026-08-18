"""
backend/clinical_summary.py
────────────────────────────
Generative AI Clinical Summary Engine for CareGrid.
Synthesizes ML model prediction outcomes, SHAP feature attributions,
and patient encounter data into clear, physician/nurse discharge narratives.

Supports:
- Google Gemini API (gemini-2.0-flash / gemini-1.5-flash)
- Groq / OpenAI API
- High-fidelity Deterministic Clinical AI Synthesizer fallback (for offline or keyless operation)
"""

from __future__ import annotations

import os
import json
from typing import Any


def format_plain_text_report(patient: dict[str, Any], prediction: dict[str, Any], summary_narrative: str) -> str:
    """Format exact 3-section human-understandable clinical report."""
    name = str(patient.get('patient_name', 'Patient') or 'Patient').strip()
    raw_age = str(patient.get('age', '[70-80)')).replace('[', '').replace(')', '').replace('-', '–')
    stay = f"{patient.get('time_in_hospital', 5)} days"
    inpatient = patient.get('n_inpatient', 0)
    meds = patient.get('n_medications', 15)
    
    diag1 = patient.get('diag_1', 'Circulatory')
    diag2 = patient.get('diag_2', '')
    if diag2 and diag2 not in ['None', 'no', 'Missing']:
        diagnoses = f"{diag1} / {diag2}"
    else:
        diagnoses = diag1
        
    prob = int(round(prediction.get('probability', 0.5) * 100))
    risk_level = str(prediction.get('risk_level') or 'High').upper()
    if 'RISK' not in risk_level:
        risk_level = f"{risk_level} RISK"
        
    is_yes = prediction.get('prediction', 0) == 1 or prediction.get('probability', 0) >= prediction.get('threshold', 0.5227)
    verdict = "YES" if is_yes else "NO"

    return f"""Patient: {name}
Age: {raw_age}
Hospital stay: {stay}
Previous inpatient visits: {inpatient}
Medications: {meds}
Diagnoses: {diagnoses}

Model Result
Readmission Risk: {prob}% — {risk_level} (Predicted: {verdict})

AI Clinical Summary
“{summary_narrative}”"""


def build_clinical_prompt(patient: dict[str, Any], prediction: dict[str, Any]) -> str:
    """Build structured prompt for human-understandable LLM clinical narrative generation."""
    risk_level = prediction.get('risk_level', 'Moderate')
    is_high = prediction.get('prediction', 0) == 1 or risk_level == 'High'
    name = str(patient.get('patient_name', 'The patient') or 'The patient').strip()

    top_risk = [item.get("feature", "") for item in prediction.get("top_increasing_risk", [])[:3]]
    top_protective = [item.get("feature", "") for item in prediction.get("top_decreasing_risk", [])[:3]]

    prompt = f"""You are a compassionate healthcare clinician writing a clear, 1 to 2 sentence discharge summary note.

Patient Profile:
- Patient Name: {name}
- Age: {patient.get('age', 'Senior')}
- Hospital stay: {patient.get('time_in_hospital', 0)} days
- Previous inpatient visits: {patient.get('n_inpatient', 0)}
- Medications: {patient.get('n_medications', 0)}
- Diagnoses: {patient.get('diag_1', 'Unspecified')} / {patient.get('diag_2', '')}
- Risk Level: {'high risk' if is_high else 'moderate risk' if risk_level == 'Moderate' else 'low risk'}
- Main Risk Drivers: {', '.join(top_risk) if top_risk else 'General clinical status'}

CRITICAL INSTRUCTIONS:
1. Write EXACTLY 1 to 2 concise, human-understandable sentences in plain English.
2. Refer to the patient by name (e.g. "{name} has been identified as high risk for readmission...").
3. DO NOT include raw numbers, percentages, or decimals.
4. Plainly mention the key factors (like previous inpatient utilization, medication burden, or chronic diagnoses) and what post-discharge care is needed.
5. Output ONLY the summary text (no quotes, no markdown, no headers).

Example style:
{name} has been identified as high risk for readmission. Previous inpatient utilization and medication burden are among the factors contributing to the elevated risk.
"""
    return prompt


def generate_fallback_summary(patient: dict[str, Any], prediction: dict[str, Any]) -> str:
    """
    Deterministic plain-language clinical synthesizer.
    Produces clean, human-understandable clinical summaries without raw numbers.
    """
    risk_level = prediction.get('risk_level', 'Moderate')
    is_high = prediction.get('prediction', 0) == 1 or risk_level == 'High'
    name = str(patient.get('patient_name', 'This patient') or 'This patient').strip()

    inpatient = patient.get('n_inpatient', 0)
    er_visits = patient.get('n_emergency', 0)
    meds = patient.get('n_medications', 15)
    los = patient.get('time_in_hospital', 5)
    diag1 = patient.get('diag_1', 'Circulatory')

    drivers = []
    if inpatient > 0:
        drivers.append("previous inpatient hospital admissions")
    if er_visits > 0:
        drivers.append("recent emergency department encounters")
    if meds >= 12:
        drivers.append("a high medication burden")
    if los >= 6:
        drivers.append("an extended hospital stay")
    if diag1 in ["Circulatory", "Diabetes", "Respiratory"]:
        drivers.append(f"ongoing management of {diag1.lower()} conditions")

    if not drivers:
        drivers.append("overall clinical presentation and patient history")

    if len(drivers) == 1:
        drivers_text = drivers[0]
    elif len(drivers) == 2:
        drivers_text = f"{drivers[0]} and {drivers[1]}"
    else:
        drivers_text = f"{drivers[0]}, {drivers[1]}, and {drivers[2]}"

    if is_high:
        return (
            f"{name} has been identified as high risk for readmission. "
            f"{drivers_text.capitalize()} are among the key factors contributing to the elevated risk. "
            f"Transition planning should prioritize a comprehensive medication review, caregiver outreach within 48 hours, "
            f"and a scheduled primary care follow-up within seven days."
        )
    elif risk_level == "Moderate":
        return (
            f"{name} demonstrates a moderate risk for hospital readmission. "
            f"Contributing factors include {drivers_text}. "
            f"Enhanced discharge coordination, clear patient teach-back instructions, and a follow-up consultation within one to two weeks are advised."
        )
    else:
        return (
            f"{name} demonstrates a low risk for hospital readmission with stable clinical indicators. "
            f"Standard discharge protocols, patient education on warning signs, and routine primary care follow-up are recommended."
        )


def generate_ai_clinical_summary(patient: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """
    Main entrypoint to generate the GenAI Clinical Summary.
    Attempts LLM generation via Gemini or Groq, falling back gracefully to the clinical synthesizer.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    prompt = build_clinical_prompt(patient, prediction)

    # 1. Try Google Gemini API (Direct REST + SDK fallback)
    if gemini_key and gemini_key.strip():
        # Try Direct REST call (fast, zero dependency errors)
        try:
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key.strip()}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 350}
            }
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as response:
                if response.status == 200:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    candidates = resp_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            summary_text = parts[0].get("text", "").strip(' "“’‘”\n')
                            if summary_text:
                                return {
                                    "summary": summary_text,
                                    "full_plain_text": format_plain_text_report(patient, prediction, summary_text),
                                    "engine": "Google Gemini (Generative AI)",
                                    "status": "success",
                                }
        except Exception as e:
            print(f"[Clinical AI] Gemini REST call notice: {e}")

    # 2. Try Groq API (using Groq SDK & available models)
    if groq_key and groq_key.strip():
        groq_models_to_try = [
            "openai/gpt-oss-120b",
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-20b",
            "groq/compound",
            "allam-2-7b",
        ]
        try:
            from groq import Groq
            groq_client = Groq(api_key=groq_key.strip())
            for model_id in groq_models_to_try:
                try:
                    completion = groq_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.3,
                    )
                    if completion.choices and completion.choices[0].message.content:
                        summary_text = completion.choices[0].message.content.strip(' "“’‘”\n')
                        if summary_text:
                            return {
                                "summary": summary_text,
                                "full_plain_text": format_plain_text_report(patient, prediction, summary_text),
                                "engine": f"Groq ({model_id})",
                                "status": "success",
                            }
                except Exception as model_err:
                    print(f"[Clinical AI] Groq model {model_id} notice: {model_err}")
                    continue
        except Exception as e:
            print(f"[Clinical AI] Groq SDK call notice: {e}")

    # 3. High-fidelity Clinical Synthesizer Fallback
    summary_text = generate_fallback_summary(patient, prediction)
    return {
        "summary": summary_text,
        "full_plain_text": format_plain_text_report(patient, prediction, summary_text),
        "engine": "CareGrid Clinical AI Synthesizer",
        "status": "success",
    }
