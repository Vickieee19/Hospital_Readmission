import React, { useState, useEffect, useRef } from 'react'
import {
  Activity,
  AlertCircle,
  Stethoscope,
  Sparkles,
  FileText,
  UserRound,
} from 'lucide-react'
import PatientForm from './components/PatientForm'
import { defaultValues } from './constants/patient'
import PredictionResult from './components/PredictionResult'
import Loading from './components/Loading'
import PdfUploadPanel from './components/PdfUploadPanel'
import { predictReadmission, uploadPdf } from './services/api'

export default function App() {
  const [formData, setFormData] = useState(defaultValues)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('patient') // 'patient' | 'pdf'
  const [pdfUploading, setPdfUploading] = useState(false)
  const [pdfStatus, setPdfStatus] = useState({ type: '', message: '' })
  const [pdfResult, setPdfResult] = useState(null)

  const resultsRef = useRef(null)

  // Run initial prediction on mount
  useEffect(() => {
    handlePredict(defaultValues, false)
  }, [])

  const handlePredict = async (dataToPredict = formData, shouldScroll = true) => {
    setLoading(true)
    setError('')

    const sanitizedData = {
      ...dataToPredict,
      time_in_hospital: Math.max(1, Number(dataToPredict.time_in_hospital || 1)),
      n_lab_procedures: Number(dataToPredict.n_lab_procedures || 0),
      n_procedures: Number(dataToPredict.n_procedures || 0),
      n_medications: Number(dataToPredict.n_medications || 0),
      n_outpatient: Number(dataToPredict.n_outpatient || 0),
      n_inpatient: Number(dataToPredict.n_inpatient || 0),
      n_emergency: Number(dataToPredict.n_emergency || 0),
    }

    try {
      const response = await predictReadmission(sanitizedData)
      setResult(response)

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
      setPdfStatus({
        type: 'success',
        message: response.message || 'PDF analyzed successfully.',
      })
    } catch (err) {
      setPdfStatus({
        type: 'error',
        message: err.message || 'Unable to analyze PDF document.',
      })
      setPdfResult(null)
    } finally {
      setPdfUploading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col selection:bg-blue-600 selection:text-white">
      {/* ─────────────────────────────────────────────────────────────
          1. CLEAN WHITE HEADER (No Badges, Pure Clinical Branding)
      ───────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-3.5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-50 border border-blue-100 text-blue-600 shadow-xs">
                <Stethoscope className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">CareGrid</h1>
                <p className="text-xs text-slate-500">
                  30-Day Hospital Readmission Risk Prediction
                </p>
              </div>
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
            className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-bold transition-all ${
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
            className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-bold transition-all ${
              activeTab === 'pdf'
                ? 'bg-blue-600 text-white shadow-xs'
                : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900'
            }`}
          >
            <FileText className="h-4 w-4" />
            <span>Upload Clinical PDF</span>
          </button>
        </div>

        {/* Tab 1: Clinical Intake & Prediction Results */}
        {activeTab === 'patient' && (
          <div className="space-y-6">
            <section>
              <PatientForm
                formData={formData}
                setFormData={setFormData}
                onSubmit={() => handlePredict(formData, true)}
                isLoading={loading}
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
                    Evaluating patient encounter...
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

        {/* Tab 2: Clinical PDF Document Analysis */}
        {activeTab === 'pdf' && (
          <div className="space-y-6">
            <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8 shadow-xs">
              <div className="mb-4">
                <h3 className="text-lg font-bold text-slate-900">Clinical Document Ingestion & Analysis</h3>
                <p className="text-xs text-slate-500">
                  Upload lab reports or discharge summaries for automated clinical severity scoring.
                </p>
              </div>

              <PdfUploadPanel
                onUpload={handlePdfUpload}
                isUploading={pdfUploading}
                status={pdfStatus}
                uploadResult={pdfResult}
              />
            </div>
          </div>
        )}
      </main>

      {/* ─────────────────────────────────────────────────────────────
          3. CLEAN FOOTER
      ───────────────────────────────────────────────────────────── */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-4 text-center text-xs text-slate-500">
        <div className="mx-auto max-w-7xl px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>CareGrid Clinical Decision Support • 30-Day Readmission Risk System</span>
          <span>Validated against Clinical Benchmark Dataset</span>
        </div>
      </footer>
    </div>
  )
}
