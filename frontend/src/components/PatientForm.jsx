import { useMemo } from 'react'
import { Activity, HeartPulse } from 'lucide-react'
import {
  AGE_OPTIONS,
  SPECIALTIES,
  DIAG_OPTIONS,
  BINARY_OPTIONS,
  GLUCOSE_OPTIONS,
  A1C_OPTIONS,
} from '../constants/patient'

function Field({ label, children }) {
  return (
    <div className="space-y-2">
      <label>{label}</label>
      {children}
    </div>
  )
}

export default function PatientForm({ formData, setFormData, onSubmit, isLoading }) {
  const numericFields = useMemo(
    () => [
      { key: 'time_in_hospital', label: 'Time in hospital', min: 0 },
      { key: 'n_lab_procedures', label: 'Lab procedures', min: 0 },
      { key: 'n_procedures', label: 'Procedures', min: 0 },
      { key: 'n_medications', label: 'Medications', min: 0 },
      { key: 'n_outpatient', label: 'Outpatient visits', min: 0 },
      { key: 'n_inpatient', label: 'Inpatient visits', min: 0 },
      { key: 'n_emergency', label: 'Emergency visits', min: 0 },
    ],
    [],
  )

  const handleChange = (key, value) => {
    setFormData((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="card p-6 sm:p-8">
      <div className="mb-6 flex items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-clinical-700">Patient details</p>
          <h2 className="mt-2 text-2xl font-bold text-slate-900">Clinical intake form</h2>
        </div>
        <div className="rounded-2xl bg-clinical-50 p-3 text-clinical-700">
          <HeartPulse className="h-6 w-6" />
        </div>
      </div>

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        <Field label="Age bracket">
          <select value={formData.age} onChange={(e) => handleChange('age', e.target.value)}>
            {AGE_OPTIONS.map((option) => (
              <option value={option} key={option}>{option}</option>
            ))}
          </select>
        </Field>

        {numericFields.map(({ key, label, min }) => (
          <Field key={key} label={label}>
            <input
              type="number"
              min={min}
              step="1"
              value={formData[key]}
              onChange={(e) => handleChange(key, Number(e.target.value || 0))}
            />
          </Field>
        ))}

        <Field label="Medical specialty">
          <select value={formData.medical_specialty} onChange={(e) => handleChange('medical_specialty', e.target.value)}>
            {SPECIALTIES.map((opt) => (
              <option value={opt} key={opt}>{opt}</option>
            ))}
          </select>
        </Field>

        <Field label="Diagnosis 1">
          <select value={formData.diag_1} onChange={(e) => handleChange('diag_1', e.target.value)}>
            {DIAG_OPTIONS.map((opt) => (
              <option value={opt} key={opt}>{opt}</option>
            ))}
          </select>
        </Field>

        <Field label="Diagnosis 2">
          <select value={formData.diag_2} onChange={(e) => handleChange('diag_2', e.target.value)}>
            {DIAG_OPTIONS.map((opt) => (
              <option value={opt} key={opt}>{opt}</option>
            ))}
          </select>
        </Field>

        <Field label="Diagnosis 3">
          <select value={formData.diag_3} onChange={(e) => handleChange('diag_3', e.target.value)}>
            {DIAG_OPTIONS.map((opt) => (
              <option value={opt} key={opt}>{opt}</option>
            ))}
          </select>
        </Field>

        <Field label="Glucose test">
          <select value={formData.glucose_test} onChange={(e) => handleChange('glucose_test', e.target.value)}>
            {GLUCOSE_OPTIONS.map((opt) => (
              <option value={opt} key={opt}>{opt}</option>
            ))}
          </select>
        </Field>

        <Field label="A1C test">
          <select value={formData.A1Ctest} onChange={(e) => handleChange('A1Ctest', e.target.value)}>
            {A1C_OPTIONS.map((opt) => (
              <option value={opt} key={opt}>{opt}</option>
            ))}
          </select>
        </Field>

        <Field label="Change in medication">
          <select value={formData.change} onChange={(e) => handleChange('change', e.target.value)}>
            {BINARY_OPTIONS.map((opt) => (
              <option value={opt} key={opt}>{opt}</option>
            ))}
          </select>
        </Field>

        <Field label="Diabetes medication">
          <select value={formData.diabetes_med} onChange={(e) => handleChange('diabetes_med', e.target.value)}>
            {BINARY_OPTIONS.map((opt) => (
              <option value={opt} key={opt}>{opt}</option>
            ))}
          </select>
        </Field>
      </div>

      <div className="mt-8 border-t border-slate-200 pt-6">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <Activity className="h-4 w-4 text-clinical-600" />
            <span>Risk threshold</span>
          </div>
          <span className="rounded-full bg-clinical-50 px-2.5 py-1 text-sm font-semibold text-clinical-700">
            {formData.threshold.toFixed(2)}
          </span>
        </div>

        <input
          type="range"
          min="0.05"
          max="0.9"
          step="0.01"
          value={formData.threshold}
          onChange={(e) => handleChange('threshold', Number(e.target.value))}
          className="mb-4 w-full accent-clinical-600"
        />

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-sm text-slate-500">Set a custom cutoff for high-risk detection.</div>

          <button
            type="button"
            onClick={onSubmit}
            disabled={isLoading}
            className="inline-flex items-center justify-center rounded-xl bg-clinical-600 px-5 py-3 text-sm font-semibold text-white shadow-soft transition hover:bg-clinical-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {isLoading ? 'Predicting...' : 'Predict readmission risk'}
          </button>
        </div>
      </div>
    </div>
  )
}
