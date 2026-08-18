import React from 'react'
import {
  Activity,
  Sparkles,
  ClipboardList,
} from 'lucide-react'
import {
  AGE_OPTIONS,
  SPECIALTIES,
  DIAG_OPTIONS,
  BINARY_OPTIONS,
  GLUCOSE_OPTIONS,
  A1C_OPTIONS,
} from '../constants/patient'

function FormField({ number, label, helpText, children }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
          <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-slate-100 border border-slate-300 text-[10px] font-mono text-slate-700">
            {number}
          </span>
          <span>{label}</span>
        </label>
        {helpText && <span className="text-[10px] text-slate-500">{helpText}</span>}
      </div>
      {children}
    </div>
  )
}

export default function PatientForm({ formData, setFormData, onSubmit, isLoading }) {
  const handleChange = (key, value) => {
    setFormData((prev) => ({ ...prev, [key]: value }))
  }

  const handleNumberChange = (key, rawValue) => {
    if (rawValue === '') {
      handleChange(key, '')
      return
    }
    // Parse integer, automatically converting e.g. "060" -> 60
    const parsed = parseInt(rawValue, 10)
    handleChange(key, isNaN(parsed) ? '' : parsed)
  }

  const handleFocus = (e) => {
    e.target.select()
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8 shadow-sm space-y-6">
      <div className="flex items-center justify-between border-b border-slate-200 pb-4">
        <div className="flex items-center gap-2">
          <ClipboardList className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-bold text-slate-900">
            Clinical Intake
          </h3>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────
          16 ENCOUNTER INPUT FIELDS IN EXACT DATASET CSV COLUMN ORDER:
          1. age
          2. time_in_hospital
          3. n_lab_procedures
          4. n_procedures
          5. n_medications
          6. n_outpatient
          7. n_inpatient
          8. n_emergency
          9. medical_specialty
          10. diag_1
          11. diag_2
          12. diag_3
          13. glucose_test
          14. A1Ctest
          15. change
          16. diabetes_med
      ───────────────────────────────────────────────────────────── */}
      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {/* 1. age */}
        <FormField number="1" label="Age" helpText="Age bracket">
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
        <FormField number="2" label="Time in Hospital" helpText="Length of stay (days)">
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
        <FormField number="3" label="N Lab Procedures" helpText="Lab tests performed">
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
        <FormField number="4" label="N Procedures" helpText="Inpatient procedures">
          <input
            type="number"
            min="0"
            max="15"
            step="1"
            value={formData.n_procedures ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_procedures', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 5. n_medications */}
        <FormField number="5" label="N Medications" helpText="Prescribed medications">
          <input
            type="number"
            min="1"
            max="100"
            step="1"
            value={formData.n_medications ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_medications', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 6. n_outpatient */}
        <FormField number="6" label="N Outpatient" helpText="Past 1 year visits">
          <input
            type="number"
            min="0"
            max="30"
            step="1"
            value={formData.n_outpatient ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_outpatient', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 7. n_inpatient */}
        <FormField number="7" label="N Inpatient" helpText="Past 1 year admissions">
          <input
            type="number"
            min="0"
            max="30"
            step="1"
            value={formData.n_inpatient ?? ''}
            onFocus={handleFocus}
            onChange={(e) => handleNumberChange('n_inpatient', e.target.value)}
            className="input-clinical"
          />
        </FormField>

        {/* 8. n_emergency */}
        <FormField number="8" label="N Emergency" helpText="Past 1 year ER visits">
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
        <FormField number="9" label="Medical Specialty" helpText="Admitting physician">
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
        <FormField number="10" label="Diagnosis 1" helpText="Primary diagnosis">
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
        <FormField number="11" label="Diagnosis 2" helpText="Secondary diagnosis">
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
        <FormField number="12" label="Diagnosis 3" helpText="Tertiary diagnosis">
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
        <FormField number="13" label="Glucose Test" helpText="Serum glucose result">
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
        <FormField number="14" label="A1C Test" helpText="HbA1c test result">
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
        <FormField number="15" label="Change" helpText="Medication changed?">
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
        <FormField number="16" label="Diabetes Med" helpText="Prescribed diabetes med?">
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

      {/* Form Submission Action Button */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-end gap-3 pt-4 border-t border-slate-200">
        <button
          type="button"
          onClick={onSubmit}
          disabled={isLoading}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-7 py-3 text-sm font-semibold text-white shadow-sm transition-all hover:bg-blue-700 active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
        >
          {isLoading ? (
            <>
              <Sparkles className="h-4 w-4 animate-spin" />
              <span>Evaluating Encounter...</span>
            </>
          ) : (
            <>
              <Activity className="h-4 w-4" />
              <span>Assess Readmission Risk</span>
            </>
          )}
        </button>
      </div>
    </div>
  )
}
