import { useState } from 'react'
import { Activity, AlertCircle, Stethoscope, Sparkles, UserRound, FileText, ArrowLeft, ArrowRight, ShieldAlert, CheckCircle2 } from 'lucide-react'
import PatientForm from './components/PatientForm'
import { defaultValues } from './constants/patient'
import PredictionResult from './components/PredictionResult'
import Loading from './components/Loading'
import PdfUploadPanel from './components/PdfUploadPanel'
import { predictReadmission, uploadPdf } from './services/api'

const INITIAL_ERROR = 'Unable to connect to the prediction server. Please make sure the backend is running.'

function App() {
  const [formData, setFormData] = useState(defaultValues)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [pdfStatus, setPdfStatus] = useState({ type: '', message: '' })
  const [pdfUploading, setPdfUploading] = useState(false)
  const [pdfResult, setPdfResult] = useState(null)
  const [currentStep, setCurrentStep] = useState('selection')
  const [selectedInputMethod, setSelectedInputMethod] = useState('manual')

  const handlePredict = async () => {
    setLoading(true)
    setError('')

    try {
      const response = await predictReadmission(formData)
      setResult(response)
      setCurrentStep('result')
    } catch (err) {
      setError(err.message || INITIAL_ERROR)
      setResult(null)
      setCurrentStep('manual')
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
        message: response.message || 'PDF analyzed successfully',
      })
    } catch (err) {
      setPdfStatus({
        type: 'error',
        message: err.message || 'Unable to upload PDF.',
      })
      setPdfResult(null)
    } finally {
      setPdfUploading(false)
    }
  }

  const renderSelectionPage = () => (
    <section className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold tracking-tight text-slate-900">Clinical Assessment Intake</h2>
        <p className="mt-2 text-base text-slate-600">Choose an evaluation method to determine readmission risk or clinical severity.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {[
          {
            key: 'manual',
            title: 'Manual Patient Input',
            description: 'Enter 16 clinical encounter parameters to predict 30-day readmission risk & discharge protocols.',
            activeClass: 'border-clinical-500 bg-clinical-50 shadow-soft ring-2 ring-clinical-500/20',
            icon: <UserRound className="h-7 w-7 text-clinical-600" />,
            badge: 'Readmission Risk ML',
          },
          {
            key: 'pdf',
            title: 'Upload Clinical PDF',
            description: 'Upload lab reports or discharge summaries for automated RAG severity scoring (0–10) & guideline citation.',
            activeClass: 'border-violet-500 bg-violet-50 shadow-soft ring-2 ring-violet-500/20',
            icon: <FileText className="h-7 w-7 text-violet-600" />,
            badge: 'Multi-Modal RAG',
          },
        ].map((option) => {
          const isActive = selectedInputMethod === option.key

          return (
            <button
              key={option.key}
              type="button"
              onClick={() => {
                setSelectedInputMethod(option.key)
                if (option.key === 'manual') {
                  setCurrentStep('manual')
                }
              }}
              className={`card flex min-h-[240px] flex-col justify-between border-2 p-6 text-left transition duration-200 hover:-translate-y-0.5 hover:shadow-lg ${
                isActive ? option.activeClass : 'border-slate-200 bg-white/80'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="rounded-xl bg-white p-3 shadow-sm">{option.icon}</div>
                <span className="rounded-full bg-slate-900 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.15em] text-white">
                  {option.badge}
                </span>
              </div>

              <div className="mt-6">
                <h3 className="text-2xl font-bold text-slate-900">{option.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-600">{option.description}</p>
              </div>

              <div className="mt-8">
                <span className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50">
                  <span>{isActive && option.key === 'pdf' ? 'Ready to Upload' : 'Select Mode'}</span>
                  <ArrowRight className="h-4 w-4 text-slate-500" />
                </span>
              </div>
            </button>
          )
        })}
      </div>

      {selectedInputMethod === 'pdf' ? (
        <div className="mx-auto w-full max-w-4xl">
          <PdfUploadPanel
            onUpload={handlePdfUpload}
            isUploading={pdfUploading}
            status={pdfStatus}
            uploadResult={pdfResult}
          />
        </div>
      ) : null}
    </section>
  )

  const renderManualPage = () => (
    <div className="card p-6 sm:p-8">
      <div className="mb-6 flex items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <button
          type="button"
          onClick={() => {
            setCurrentStep('selection')
            setSelectedInputMethod('manual')
          }}
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
        >
          <ArrowLeft className="h-4 w-4 text-slate-600" />
          <span>Back to Selection</span>
        </button>

        <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Step 2 of 2</div>
      </div>

      <div className="mb-6 text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-clinical-700">Clinical Data Intake</p>
        <h2 className="mt-2 text-3xl font-bold text-slate-900">Patient Encounter Record</h2>
        <p className="mt-1 text-sm text-slate-600">Provide encounter details to predict 30-day readmission risk and generate prevention plans.</p>
      </div>

      <PatientForm formData={formData} setFormData={setFormData} onSubmit={handlePredict} isLoading={loading} />

      <div className="mt-6 flex flex-col gap-3 border-t border-slate-200 pt-6 sm:flex-row sm:items-center sm:justify-between">
        <button
          type="button"
          onClick={() => {
            setCurrentStep('selection')
            setSelectedInputMethod('manual')
          }}
          className="inline-flex items-center justify-center rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
        >
          Back
        </button>

        <button
          type="button"
          onClick={handlePredict}
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-clinical-600 px-6 py-3 text-sm font-semibold text-white shadow-soft transition hover:bg-clinical-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {loading ? (
            <>
              <Sparkles className="h-4 w-4 animate-spin" />
              Calculating Prediction...
            </>
          ) : (
            <>
              <span>Assess Readmission Risk</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </div>
    </div>
  )

  const renderResultPage = () => (
    <div className="card p-6 sm:p-8">
      <div className="mb-6 flex items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <button
          type="button"
          onClick={() => {
            setCurrentStep('manual')
            setError('')
          }}
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
        >
          <ArrowLeft className="h-4 w-4 text-slate-600" />
          <span>Edit Patient Details</span>
        </button>

        <button
          type="button"
          onClick={() => {
            setCurrentStep('selection')
            setResult(null)
          }}
          className="text-xs font-semibold uppercase tracking-[0.15em] text-clinical-700 hover:underline"
        >
          New Patient Intake
        </button>
      </div>

      {loading ? (
        <Loading />
      ) : error ? (
        <div className="card flex items-start gap-3 border border-rose-200 bg-rose-50 p-4 text-rose-700">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
          <div>{error}</div>
        </div>
      ) : result ? (
        <PredictionResult result={result} />
      ) : null}
    </div>
  )

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-clinical-100 p-2.5 text-clinical-700">
                <Stethoscope className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">CareGrid</h1>
                <p className="text-xs text-slate-500 sm:text-sm">Clinical Decision Support & Readmission Prevention</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="hidden rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 sm:inline-block">
                Backend API Connected
              </span>
              <span className="rounded-full border border-clinical-200 bg-clinical-50 px-3 py-1 text-xs font-semibold text-clinical-700">
                v2 Ensemble
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <section className="mb-8 grid gap-4 sm:grid-cols-3">
          <div className="card flex items-center gap-4 p-4 shadow-sm">
            <div className="rounded-xl bg-clinical-100 p-2 text-clinical-700"><Activity className="h-5 w-5" /></div>
            <div>
              <p className="text-xs text-slate-500">Model Engine</p>
              <p className="text-base font-bold text-slate-900">XGB + LGBM Ensemble</p>
            </div>
          </div>
          <div className="card flex items-center gap-4 p-4 shadow-sm">
            <div className="rounded-xl bg-emerald-100 p-2 text-emerald-700"><Activity className="h-5 w-5" /></div>
            <div>
              <p className="text-xs text-slate-500">Operating Threshold</p>
              <p className="text-base font-bold text-slate-900">{(formData.threshold ?? 0.52).toFixed(2)} (MCC Optimal)</p>
            </div>
          </div>
          <div className="card flex items-center gap-4 p-4 shadow-sm">
            <div className="rounded-xl bg-amber-100 p-2 text-amber-700"><Activity className="h-5 w-5" /></div>
            <div>
              <p className="text-xs text-slate-500">Clinical Focus</p>
              <p className="text-base font-bold text-slate-900">30-Day Discharge Safety</p>
            </div>
          </div>
        </section>

        {currentStep === 'selection' && renderSelectionPage()}
        {currentStep === 'manual' && renderManualPage()}
        {currentStep === 'result' && renderResultPage()}
      </main>
    </div>
  )
}

export default App
