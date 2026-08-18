import React from 'react'
import {
  Activity,
  Sparkles,
  ClipboardList,
  CheckCircle2,
  AlertTriangle,
  User,
} from 'lucide-react'
import {
  AGE_OPTIONS,
  SPECIALTIES,
  DIAG_OPTIONS,
  BINARY_OPTIONS,
  GLUCOSE_OPTIONS,
  A1C_OPTIONS,
} from '../constants/patient'

function FormField({ number, label, helpText, isAutoFilled, isMissing, children }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-700 flex items-center gap-1.5 flex-wrap">
          <span
            className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-mono ${
              isAutoFilled
                ? 'bg-emerald-100 border border-emerald-300 text-emerald-800 font-bold'
                : isMissing
                ? 'bg-amber-100 border border-amber-300 text-amber-800 font-bold'
                : 'bg-slate-100 border border-slate-300 text-slate-700'
            }`}
          >
            {number}
          </span>
          <span>{label}</span>
          {isAutoFilled && (
            <span className="rounded bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 text-[9px] font-bold text-emerald-700 tracking-tight">
              ✓ Auto-filled
            </span>
          )}
          {isMissing && (
            <span className="rounded bg-amber-50 border border-amber-200 px-1.5 py-0.5 text-[9px] font-bold text-amber-800 tracking-tight">
              ⚠️ Needs Input
            </span>
          )}
        </label>
        {helpText && <span className="text-[10px] text-slate-500">{helpText}</span>}
      </div>
      <div className={isMissing ? 'rounded-xl ring-2 ring-amber-300/80 ring-offset-1' : ''}>
        {children}
      </div>
    </div>
  )
}

export default function PatientForm({
  formData,
  setFormData,
  onSubmit,
  isLoading,
  extractedFields = [],
  missingFields = [],
}) {
  const handleChange = (key, value) => {
    setFormData((prev) => ({ ...prev, [key]: value }))
  }

  const handleNumberChange = (key, rawValue) => {
    if (rawValue === '') {
      handleChange(key, '')
      return
    }
    const parsed = parseInt(rawValue, 10)
    handleChange(key, isNaN(parsed) ? '' : parsed)
  }

  const handleFocus = (e) => {
    e.target.select()
  }

  const isFilled = (k) => extractedFields.includes(k)
  const isReq = (k) => missingFields.includes(k)

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8 shadow-sm space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-200 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-50 border border-blue-100 text-blue-600 shadow-xs">
            <ClipboardList className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-900">Clinical Intake</h3>
            <p className="text-xs text-slate-500">
              16 standard clinical parameters for 30-day readmission risk stratification.
            </p>
          </div>
        </div>

        {extractedFields.length > 0 && (
          <div className="flex items-center gap-2 text-xs">
            <span className="inline-flex items-center gap-1 rounded-lg bg-emerald-50 border border-emerald-200 px-2.5 py-1 font-semibold text-emerald-800">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
              <span>{extractedFields.length} Auto-filled</span>
            </span>
            {missingFields.length > 0 && (
              <span className="inline-flex items-center gap-1 rounded-lg bg-amber-50 border border-amber-200 px-2.5 py-1 font-semibold text-amber-800">
                <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
                <span>{missingFields.length} Manual Input</span>
              </span>
            )}
          </div>
        )}
      </div>

      {/* ─────────────────────────────────────────────────────────────
          PATIENT IDENTIFICATION / FULL NAME
      ───────────────────────────────────────────────────────────── */}
      <div className="rounded-xl border border-blue-100 bg-blue-50/40 p-4">
        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          <div className="w-full sm:w-1/2 space-y-1.5">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
              <User className="h-3.5 w-3.5 text-blue-600" />
              <span>Patient Full Name</span>
              <span className="text-[10px] text-slate-500 font-normal">(Used in AI Narrative & Brief)</span>
            </label>
            <input
              type="text"
              placeholder="e.g. Eleanor Vance, John Doe"
              value={formData.patient_name || ''}
              onChange={(e) => handleChange('patient_name', e.target.value)}
              className="input-clinical font-semibold text-slate-900 bg-white"
            />
          </div>
          <div className="text-xs text-slate-500 italic pt-1 sm:pt-4">
            Patient name will be personalized into the AI Clinical Summary and exported transition brief.
          </div>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────
          16 ENCOUNTER INPUT FIELDS IN EXACT DATASET CSV COLUMN ORDER:
      ───────────────────────────────────────────────────────────── */}
      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {/* 1. age */}
        <FormField number="1" label="Age" helpText="Age bracket" isAutoFilled={isFilled('age')} isMissing={isReq('age')}>
          <select
            value={formData.age}
            onChange={(e) => handleChange('age', e.target.value)}
            className="input-clinical"
          >
            {AGE_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </FormField>

        {/* 2. time_in_hospital */}
        <FormField number="2" label="Time in Hospital" helpText="Length of stay (days)" isAutoFilled={isFilled('time_in_hospital')} isMissing={isReq('time_in_hospital')}>
          <input
            type="number"
            min="1"
            max="30"
            step="1"
            value={formData.time_in_hospital ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('time_in_hospital', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 3. n_lab_procedures */}
        <FormField number="3" label="N Lab Procedures" helpText="Lab tests performed" isAutoFilled={isFilled('n_lab_procedures')} isMissing={isReq('n_lab_procedures')}>
          <input
            type="number"
            min="0"
            max="150"
            step="1"
            value={formData.n_lab_procedures ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_lab_procedures', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 4. n_procedures */}
        <FormField number="4" label="N Procedures" helpText="Inpatient procedures" isAutoFilled={isFilled('n_procedures')} isMissing={isReq('n_procedures')}>
          <input
            type="number"
            min="0"
            max="10"
            step="1"
            value={formData.n_procedures ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_procedures', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 5. n_medications */}
        <FormField number="5" label="N Medications" helpText="Active prescribed meds" isAutoFilled={isFilled('n_medications')} isMissing={isReq('n_medications')}>
          <input
            type="number"
            min="1"
            max="80"
            step="1"
            value={formData.n_medications ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_medications', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 6. n_outpatient */}
        <FormField number="6" label="N Outpatient" helpText="Prior outpatient visits" isAutoFilled={isFilled('n_outpatient')} isMissing={isReq('n_outpatient')}>
          <input
            type="number"
            min="0"
            max="40"
            step="1"
            value={formData.n_outpatient ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_outpatient', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 7. n_inpatient */}
        <FormField number="7" label="N Inpatient" helpText="Prior admissions (yr)" isAutoFilled={isFilled('n_inpatient')} isMissing={isReq('n_inpatient')}>
          <input
            type="number"
            min="0"
            max="25"
            step="1"
            value={formData.n_inpatient ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_inpatient', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 8. n_emergency */}
        <FormField number="8" label="N Emergency" helpText="Prior ER encounters" isAutoFilled={isFilled('n_emergency')} isMissing={isReq('n_emergency')}>
          <input
            type="number"
            min="0"
            max="30"
            step="1"
            value={formData.n_emergency ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_emergency', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 9. medical_specialty */}
        <FormField number="9" label="Medical Specialty" helpText="Admitting specialty" isAutoFilled={isFilled('medical_specialty')} isMissing={isReq('medical_specialty')}>
          <select
            value={formData.medical_specialty}
            onChange={(e) => handleChange('medical_specialty', e.target.value)}
            className="input-clinical"
          >
            {SPECIALTIES.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </FormField>

        {/* 10. diag_1 */}
        <FormField number="10" label="Primary Diagnosis (diag_1)" helpText="Principal condition" isAutoFilled={isFilled('diag_1')} isMissing={isReq('diag_1')}>
          <select
            value={formData.diag_1}
            onChange={(e) => handleChange('diag_1', e.target.value)}
            className="input-clinical"
          >
            {DIAG_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </FormField>

        {/* 11. diag_2 */}
        <FormField number="11" label="Secondary Diagnosis (diag_2)" helpText="Secondary condition" isAutoFilled={isFilled('diag_2')} isMissing={isReq('diag_2')}>
          <select
            value={formData.diag_2}
            onChange={(e) => handleChange('diag_2', e.target.value)}
            className="input-clinical"
          >
            {DIAG_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </FormField>

        {/* 12. diag_3 */}
        <FormField number="12" label="Tertiary Diagnosis (diag_3)" helpText="Tertiary condition" isAutoFilled={isFilled('diag_3')} isMissing={isReq('diag_3')}>
          <select
            value={formData.diag_3}
            onChange={(e) => handleChange('diag_3', e.target.value)}
            className="input-clinical"
          >
            {DIAG_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </FormField>

        {/* 13. glucose_test */}
        <FormField number="13" label="Glucose Test" helpText="Fasting blood sugar test" isAutoFilled={isFilled('glucose_test')} isMissing={isReq('glucose_test')}>
          <select
            value={formData.glucose_test}
            onChange={(e) => handleChange('glucose_test', e.target.value)}
            className="input-clinical"
          >
            {GLUCOSE_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </FormField>

        {/* 14. A1Ctest */}
        <FormField number="14" label="A1C Test" helpText="HbA1c test result" isAutoFilled={isFilled('A1Ctest')} isMissing={isReq('A1Ctest')}>
          <select
            value={formData.A1Ctest}
            onChange={(e) => handleChange('A1Ctest', e.target.value)}
            className="input-clinical"
          >
            {A1C_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </FormField>

        {/* 15. change */}
        <FormField number="15" label="Medication Change" helpText="Dosage/regimen adjusted" isAutoFilled={isFilled('change')} isMissing={isReq('change')}>
          <select
            value={formData.change}
            onChange={(e) => handleChange('change', e.target.value)}
            className="input-clinical"
          >
            {BINARY_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </FormField>

        {/* 16. diabetes_med */}
        <FormField number="16" label="Diabetes Med" helpText="Prescribed diabetes meds" isAutoFilled={isFilled('diabetes_med')} isMissing={isReq('diabetes_med')}>
          <select
            value={formData.diabetes_med}
            onChange={(e) => handleChange('diabetes_med', e.target.value)}
            className="input-clinical"
          >
            {BINARY_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </FormField>
      </div>

      {/* ─────────────────────────────────────────────────────────────
          SUBMIT ACTION BUTTON
      ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pt-4 border-t border-slate-200">
        <p className="text-xs text-slate-500">
          All 16 parameters are processed through the calibrated ensemble pipeline.
        </p>

        <button
          type="button"
          onClick={onSubmit}
          disabled={isLoading}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-semibold text-white shadow-xs transition-all hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500 cursor-pointer"
        >
          {isLoading ? (
            <>
              <Activity className="h-4 w-4 animate-spin" />
              <span>Analyzing Risk & SHAP Reasoning...</span>
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              <span>Predict Readmission Risk</span>
            </>
          )}
        </button>
      </div>
    </div>
  )
}
