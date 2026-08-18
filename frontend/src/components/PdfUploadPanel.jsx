import React, { useRef, useState } from 'react'
import {
  FileText,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  Sparkles,
  ArrowRight,
  ClipboardCheck,
  Edit3,
  ShieldAlert,
  ShieldCheck,
  Zap,
} from 'lucide-react'

const FEATURE_LABELS = {
  age: 'Age Bracket',
  time_in_hospital: 'Length of Stay (Days)',
  n_lab_procedures: 'Lab Procedures',
  n_procedures: 'Clinical Procedures',
  n_medications: 'Prescribed Medications',
  n_outpatient: 'Prior Outpatient Visits',
  n_inpatient: 'Prior Inpatient Admissions',
  n_emergency: 'Prior Emergency Visits',
  medical_specialty: 'Admitting Specialty',
  diag_1: 'Primary Diagnosis',
  diag_2: 'Secondary Diagnosis',
  diag_3: 'Tertiary Diagnosis',
  glucose_test: 'Glucose Test',
  A1Ctest: 'HbA1c Test',
  change: 'Medication Changed',
  diabetes_med: 'Diabetes Medication',
}

export default function PdfUploadPanel({
  onUpload,
  isUploading,
  status,
  uploadResult,
  onSwitchToIntake,
  onPredictDirectly,
}) {
  const inputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] ?? null
    if (!file) {
      setSelectedFile(null)
      return
    }

    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setSelectedFile(null)
      event.target.value = ''
      return
    }

    setSelectedFile(file)
  }

  const handleUpload = () => {
    if (!selectedFile) return
    onUpload(selectedFile)
  }

  const isMedicalReport = uploadResult?.is_medical_report === true
  const extracted = isMedicalReport ? (uploadResult?.extracted_patient || {}) : {}
  const foundList = isMedicalReport ? (uploadResult?.extracted_fields_list || []) : []
  const missingList = isMedicalReport ? (uploadResult?.missing_fields_list || []) : []
  const isPartial = uploadResult?.is_partial || false
  const hasExtracted = isMedicalReport && Object.keys(extracted).length > 0
  const isNonMedical = uploadResult && uploadResult.is_medical_report === false

  return (
    <div className="space-y-6">
      {/* Upload Box */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-7 shadow-xs space-y-5">
        <div className="flex items-center gap-3 border-b border-slate-100 pb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-600 border border-blue-100 shadow-xs">
            <FileText className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-slate-900">
              Upload Clinical Report (Auto-Fill Assistant)
            </h2>
            <p className="text-xs text-slate-500">
              Upload a discharge summary or lab report PDF. Non-medical PDFs are automatically rejected and not taken into consideration.
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-5 transition-colors hover:border-blue-400 hover:bg-blue-50/20">
            <UploadCloud className="h-6 w-6 text-slate-500 shrink-0" />
            <input
              ref={inputRef}
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              className="w-full text-sm text-slate-600 file:mr-4 file:rounded-xl file:border-0 file:bg-blue-600 file:px-4 file:py-2 file:text-xs file:font-bold file:text-white hover:file:bg-blue-700 cursor-pointer"
            />
          </div>

          <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
            <span>
              {selectedFile
                ? `Selected: ${selectedFile.name}`
                : 'Select a clinical lab report or discharge summary PDF.'}
            </span>
            {selectedFile && (
              <span className="text-xs font-semibold text-slate-500">
                {(selectedFile.size / 1024).toFixed(1)} KB
              </span>
            )}
          </div>

          <button
            type="button"
            onClick={handleUpload}
            disabled={!selectedFile || isUploading}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-xs transition-all hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500 cursor-pointer"
          >
            {isUploading ? (
              <>
                <Sparkles className="h-4 w-4 animate-spin" />
                <span>Verifying Document & Extracting Values...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                <span>Verify & Extract Patient Data</span>
              </>
            )}
          </button>

          {status?.message && (
            <div
              className={`flex items-start gap-3 rounded-xl border p-4 text-sm font-medium ${
                status.type === 'success'
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
                  : status.type === 'warning'
                  ? 'border-amber-200 bg-amber-50 text-amber-900'
                  : 'border-rose-200 bg-rose-50 text-rose-900'
              }`}
            >
              {status.type === 'success' ? (
                <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-600 mt-0.5" />
              ) : status.type === 'warning' ? (
                <AlertTriangle className="h-5 w-5 shrink-0 text-amber-600 mt-0.5" />
              ) : (
                <ShieldAlert className="h-5 w-5 shrink-0 text-rose-600 mt-0.5" />
              )}
              <div className="space-y-1">
                <p>{status.message}</p>
                {status.type === 'error' && (
                  <p className="text-xs text-rose-700">
                    The document was not taken into consideration. You can manually complete the intake form or upload a valid medical report.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Non-Medical Document Rejection Card */}
      {isNonMedical && (
        <div className="rounded-2xl border border-rose-200 bg-rose-50/80 p-6 shadow-xs space-y-3">
          <div className="flex items-center gap-2.5 text-rose-900">
            <ShieldAlert className="h-5 w-5 text-rose-600 shrink-0" />
            <h3 className="font-bold text-base">Document Rejected: Non-Medical File Detected</h3>
          </div>
          <p className="text-xs leading-relaxed text-rose-800">
            CareGrid only processes verified medical documents (discharge summaries, encounter notes, pathology/lab reports).
            Because this document is not a medical report, <strong>it was completely ignored and no clinical intake fields were changed</strong>.
          </p>
          {uploadResult.extracted_text_preview && (
            <div className="rounded-xl border border-rose-200 bg-white p-3.5">
              <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">
                Rejected Document Text Snippet
              </span>
              <p className="mt-1 text-xs font-mono text-slate-700 line-clamp-3">
                {uploadResult.extracted_text_preview}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Extraction Results Breakdown (Only for Verified Medical Reports) */}
      {hasExtracted && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-7 shadow-xs space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-100 pb-4">
            <div className="flex items-center gap-2.5">
              <ShieldCheck className="h-5 w-5 text-emerald-600 shrink-0" />
              <div>
                <h3 className="text-lg font-bold text-slate-900">Verified Clinical Data Extracted</h3>
                <p className="text-xs text-slate-500">
                  {foundList.length} of 16 parameters detected from{' '}
                  <span className="font-semibold text-slate-700">{uploadResult.filename}</span>
                </p>
              </div>
            </div>

            {/* Quick Actions for Partial vs Complete Reports */}
            {!isPartial && onSwitchToIntake && (
              <button
                type="button"
                onClick={onSwitchToIntake}
                className="inline-flex items-center gap-1.5 rounded-xl bg-blue-600 px-4 py-2 text-xs font-bold text-white shadow-xs transition hover:bg-blue-700 cursor-pointer"
              >
                <span>Review in Intake & Predict</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          {/* Section 1: Successfully Extracted */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-800 flex items-center gap-1.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                <span>Successfully Auto-Filled ({foundList.length} fields)</span>
              </span>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {Object.entries(extracted).map(([key, val]) => (
                <div
                  key={key}
                  className="flex flex-col justify-between rounded-xl border border-emerald-100 bg-emerald-50/40 p-3.5 shadow-2xs"
                >
                  <span className="text-[11px] font-semibold text-emerald-700 uppercase tracking-wider">
                    {FEATURE_LABELS[key] || key}
                  </span>
                  <span className="mt-1 font-mono text-sm font-bold text-slate-900">
                    {String(val)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Section 2: Missing Fields Requiring Manual Input or Direct Predict Option */}
          {isPartial && missingList.length > 0 && (
            <div className="space-y-4 pt-4 border-t border-slate-100">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-amber-800 flex items-center gap-1.5">
                  <Edit3 className="h-4 w-4 text-amber-600" />
                  <span>Missing from PDF ({missingList.length} fields)</span>
                </span>
                <span className="text-[11px] text-slate-500">
                  Choose how you would like to proceed:
                </span>
              </div>

              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                {missingList.map((key) => (
                  <div
                    key={key}
                    className="flex items-center justify-between rounded-xl border border-amber-200 bg-amber-50/60 p-2.5 shadow-2xs"
                  >
                    <span className="text-xs font-medium text-amber-900">
                      {FEATURE_LABELS[key] || key}
                    </span>
                    <span className="rounded bg-amber-200/70 px-1.5 py-0.5 text-[10px] font-bold text-amber-800 uppercase tracking-tight">
                      Needs Input
                    </span>
                  </div>
                ))}
              </div>

              {/* Two Direct Options for Partial Medical PDF */}
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-2">
                <button
                  type="button"
                  onClick={onSwitchToIntake}
                  className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-xs font-bold text-white shadow-xs transition hover:bg-blue-700 cursor-pointer"
                >
                  <Edit3 className="h-4 w-4" />
                  <span>Option A: Complete {missingList.length} Missing Fields Manually (Recommended)</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </button>

                {onPredictDirectly && (
                  <button
                    type="button"
                    onClick={() => onPredictDirectly(extracted)}
                    className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl border border-blue-200 bg-blue-50/70 px-4 py-3 text-xs font-bold text-blue-800 shadow-xs transition hover:bg-blue-100 hover:border-blue-300 cursor-pointer"
                  >
                    <Zap className="h-4 w-4 text-blue-600" />
                    <span>Option B: Predict Directly with {foundList.length} Extracted Features</span>
                  </button>
                )}
              </div>
            </div>
          )}

          {uploadResult.extracted_text_preview && (
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
              <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                Document Text Preview
              </span>
              <p className="mt-1 text-xs font-mono text-slate-700 leading-relaxed whitespace-pre-wrap">
                {uploadResult.extracted_text_preview}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
