import React, { useState, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { predictReadmission, uploadPdf } from '../services/api'
import PatientForm from '../components/PatientForm'
import PredictionResult from '../components/PredictionResult'
import Loading from '../components/Loading'
import PdfUploadPanel from '../components/PdfUploadPanel'
import {
  Stethoscope,
  UserRound,
  FileText,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  X,
  ArrowRight,
  LogOut,
  Shield,
  User,
} from 'lucide-react'

const defaultValues = {
  age: '[70-80)',
  time_in_hospital: 5,
  n_lab_procedures: 40,
  n_procedures: 2,
  n_medications: 15,
  n_outpatient: 0,
  n_inpatient: 1,
  n_emergency: 0,
  medical_specialty: 'InternalMedicine',
  diag_1: 'Circulatory',
  diag_2: 'Diabetes',
  diag_3: 'Other',
  glucose_test: 'no',
  A1Ctest: 'no',
  change: 'yes',
  diabetes_med: 'yes',
  threshold: 0.5227,
}

export default function ClinicalDashboardPage() {
  const navigate = useNavigate()
  const { user, logout, isAdmin } = useAuth()

  const [activeTab, setActiveTab] = useState('patient') // 'patient' | 'pdf'
  const [formData, setFormData] = useState(defaultValues)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // PDF Upload State
  const [pdfUploading, setPdfUploading] = useState(false)
  const [pdfResult, setPdfResult] = useState(null)
  const [pdfStatus, setPdfStatus] = useState({ type: '', message: '' })

  const resultsRef = useRef(null)

  const handlePredict = async (dataToPredict, shouldScroll = true) => {
    setLoading(true)
    setError(null)
    try {
      const predictionData = await predictReadmission(dataToPredict)
      setResult(predictionData)

      if (shouldScroll && resultsRef.current) {
        setTimeout(() => {
          resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }, 100)
      }
    } catch (err) {
      setError(err.message || 'Unable to connect to the prediction server. Please ensure backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const handlePdfUpload = async (file) => {
    setPdfUploading(true)
    setPdfStatus({ type: '', message: '' })

    try {
      const response = await uploadPdf(file)
      setPdfResult(response)

      // Strict validation: Only auto-fill if the document is a verified medical report
      if (response.is_medical_report && response.success && response.extracted_patient && Object.keys(response.extracted_patient).length > 0) {
        setFormData((prev) => ({
          ...prev,
          ...response.extracted_patient,
        }))

        if (response.is_partial) {
          setPdfStatus({
            type: 'warning',
            message: `Verified Medical Report: Auto-filled ${response.extracted_fields_count} fields from ${response.filename}. ${response.missing_fields_count} fields were not found and need manual completion.`,
          })
        } else {
          setPdfStatus({
            type: 'success',
            message: `Verified Medical Report: Successfully extracted all ${response.extracted_fields_count} clinical parameters from ${response.filename}.`,
          })
        }
      } else {
        // NON-MEDICAL OR UNRECOGNIZED: Document is completely rejected and ignored
        setPdfStatus({
          type: 'error',
          message: response.message || 'Warning: The uploaded file is not recognized as a medical/clinical report and was not taken into consideration.',
        })
      }
    } catch (err) {
      setPdfStatus({
        type: 'error',
        message: err.message || 'Unable to parse PDF document.',
      })
      setPdfResult(null)
    } finally {
      setPdfUploading(false)
    }
  }

  const handlePredictDirectlyWithExtracted = async (extractedData) => {
    // Merge baseline defaults with extracted features
    const mergedData = {
      ...defaultValues,
      ...extractedData,
    }
    setFormData(mergedData)
    setActiveTab('patient')
    await handlePredict(mergedData, true)
  }

  const handleClearPdfState = () => {
    setPdfResult(null)
    setPdfStatus({ type: '', message: '' })
  }

  const handleLogout = async () => {
    await logout()
    navigate('/login', { replace: true })
  }

  const extractedList = pdfResult?.extracted_fields_list || []
  const missingList = pdfResult?.is_partial ? (pdfResult?.missing_fields_list || []) : []

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col selection:bg-blue-600 selection:text-white">
      {/* ─────────────────────────────────────────────────────────────
          1. CLEAN WHITE HEADER WITH STAFF INFO & LOGOUT
      ───────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur-sm shadow-2xs">
        <div className="mx-auto max-w-7xl px-4 py-3.5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-50 border border-blue-100 text-blue-600 shadow-xs">
                <Stethoscope className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">CareGrid</h1>
                <p className="text-xs text-slate-500">
                  30-Day Hospital Readmission Risk Prediction & Clinical XAI
                </p>
              </div>
            </div>

            {/* Authenticated Staff Identity & Controls */}
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="hidden sm:flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-700">
                <User className="h-3.5 w-3.5 text-slate-500" />
                <span className="font-bold text-slate-900">{user?.full_name || user?.username}</span>
                <span className="rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-bold text-blue-800 uppercase">
                  {user?.role === 'admin' ? 'Administrator' : 'Nurse / Staff'}
                </span>
              </div>

              {isAdmin && (
                <Link
                  to="/admin"
                  className="inline-flex items-center gap-1.5 rounded-xl border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-xs font-bold text-indigo-700 shadow-xs transition hover:bg-indigo-100 cursor-pointer"
                >
                  <Shield className="h-3.5 w-3.5" />
                  <span>Admin Console</span>
                </Link>
              )}

              <button
                type="button"
                onClick={handleLogout}
                className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-bold text-slate-700 shadow-xs transition hover:bg-rose-50 hover:text-rose-700 hover:border-rose-200 cursor-pointer"
              >
                <LogOut className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Sign Out</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ─────────────────────────────────────────────────────────────
          2. MAIN CLINICAL DASHBOARD
      ───────────────────────────────────────────────────────────── */}
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8 space-y-6">
        {/* Navigation Tabs */}
        <div className="flex items-center gap-3 border-b border-slate-200 pb-3">
          <button
            type="button"
            onClick={() => setActiveTab('patient')}
            className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-bold transition-all cursor-pointer ${
              activeTab === 'patient'
                ? 'bg-blue-600 text-white shadow-xs'
                : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900'
            }`}
          >
            <UserRound className="h-4 w-4" />
            <span>Clinical Intake</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('pdf')}
            className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-bold transition-all cursor-pointer ${
              activeTab === 'pdf'
                ? 'bg-blue-600 text-white shadow-xs'
                : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900'
            }`}
          >
            <FileText className="h-4 w-4" />
            <span>Auto-Fill from PDF</span>
          </button>
        </div>

        {/* Tab 1: Clinical Intake & Prediction Results */}
        {activeTab === 'patient' && (
          <div className="space-y-6">
            {/* Auto-fill notification if PDF was uploaded */}
            {pdfResult?.extracted_patient && Object.keys(pdfResult.extracted_patient).length > 0 && (
              <div
                className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 rounded-2xl border p-4 shadow-xs ${
                  pdfResult.is_partial
                    ? 'border-amber-200 bg-amber-50 text-amber-900'
                    : 'border-emerald-200 bg-emerald-50 text-emerald-900'
                }`}
              >
                <div className="flex items-start gap-3">
                  {pdfResult.is_partial ? (
                    <AlertTriangle className="h-5 w-5 shrink-0 text-amber-600 mt-0.5" />
                  ) : (
                    <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-600 mt-0.5" />
                  )}
                  <div className="space-y-1">
                    <div className="text-sm font-semibold">
                      {pdfResult.is_partial
                        ? `Partial Auto-Fill: ${pdfResult.extracted_fields_count} fields populated from ${pdfResult.filename}`
                        : `Complete Auto-Fill: All 16 fields populated from ${pdfResult.filename}`}
                    </div>
                    {pdfResult.is_partial && (
                      <p className="text-xs text-amber-800">
                        {pdfResult.missing_fields_count} field(s) were not found in the document (marked with ⚠️ Needs Input). You can manually adjust them or predict right away.
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 self-end sm:self-center">
                  <button
                    type="button"
                    onClick={() => handlePredict(formData, true)}
                    className="inline-flex items-center gap-1.5 rounded-xl bg-blue-600 px-3.5 py-1.5 text-xs font-bold text-white shadow-xs hover:bg-blue-700 cursor-pointer"
                  >
                    <span>Predict Risk</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={handleClearPdfState}
                    title="Dismiss PDF Indicators"
                    className="rounded-lg p-1.5 text-slate-500 hover:bg-black/5 cursor-pointer"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>
            )}

            <section>
              <PatientForm
                formData={formData}
                setFormData={setFormData}
                onSubmit={() => handlePredict(formData, true)}
                isLoading={loading}
                extractedFields={extractedList}
                missingFields={missingList}
              />
            </section>

            {/* Error Display */}
            {error && (
              <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-rose-800 flex items-start gap-3 shadow-xs">
                <AlertCircle className="h-5 w-5 shrink-0 text-rose-600 mt-0.5" />
                <div className="space-y-1">
                  <strong className="text-sm font-semibold">Prediction Server Notice</strong>
                  <p className="text-xs text-rose-700">{error}</p>
                </div>
              </div>
            )}

            {/* Assessment Results Section */}
            <section ref={resultsRef} className="pt-2 scroll-mt-20">
              {loading ? (
                <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center shadow-xs">
                  <Loading />
                  <p className="mt-4 text-sm font-medium text-slate-600">
                    Evaluating patient encounter & computing XAI explanation...
                  </p>
                </div>
              ) : result ? (
                <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8 shadow-xs">
                  <PredictionResult result={result} />
                </div>
              ) : null}
            </section>
          </div>
        )}

        {/* Tab 2: Clinical PDF Auto-Fill Panel */}
        {activeTab === 'pdf' && (
          <div className="space-y-6">
            <PdfUploadPanel
              onUpload={handlePdfUpload}
              isUploading={pdfUploading}
              status={pdfStatus}
              uploadResult={pdfResult}
              onSwitchToIntake={() => setActiveTab('patient')}
              onPredictDirectly={handlePredictDirectlyWithExtracted}
            />
          </div>
        )}
      </main>

      {/* ─────────────────────────────────────────────────────────────
          3. CLEAN FOOTER
      ───────────────────────────────────────────────────────────── */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-4 text-center text-xs text-slate-500">
        <div className="mx-auto max-w-7xl px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>CareGrid Clinical Decision Support • 30-Day Readmission Risk System</span>
          <span>Powered by Calibrated ML & SHAP Tree Explainability</span>
        </div>
      </footer>
    </div>
  )
}
