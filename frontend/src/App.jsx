import { useState } from 'react'
import { Activity, AlertCircle, Stethoscope } from 'lucide-react'
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
      setPdfStatus({
        type: 'success',
        message: response.message || 'PDF uploaded successfully',
      })
    } catch (err) {
      setPdfStatus({
        type: 'error',
        message: err.message || 'Unable to upload PDF.',
      })
    } finally {
      setPdfUploading(false)
    }
  }

  const renderSelectionPage = () => (
    <section className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold tracking-tight text-slate-900">Input Method</h2>
        <p className="mt-3 text-base text-slate-600">Choose how you want to provide patient information.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {[
          {
            key: 'manual',
            title: 'Manual Input',
            description: 'Enter patient details manually',
            activeClass: 'border-clinical-500 bg-clinical-50 shadow-soft',
            icon: '🧑‍⚕️',
          },
          {
            key: 'pdf',
            title: 'Upload PDF',
            description: 'Upload patient PDF document',
            activeClass: 'border-violet-500 bg-violet-50 shadow-soft',
            icon: '📄',
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
                <div className="rounded-xl bg-white p-3 text-3xl shadow-sm">{option.icon}</div>
                {isActive ? (
                  <span className="rounded-full bg-slate-900 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.15em] text-white">
                    Selected
                  </span>
                ) : null}
              </div>

              <div className="mt-6">
                <h3 className="text-2xl font-bold text-slate-900">{option.title}</h3>
                <p className="mt-3 text-sm leading-6 text-slate-600">{option.description}</p>
              </div>

              <div className="mt-8">
                <span className="inline-flex w-full items-center justify-center rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 shadow-sm">
                  {isActive ? 'Selected' : 'Select'}
                </span>
              </div>
            </button>
          )
        })}
      </div>

      {selectedInputMethod === 'pdf' ? (
        <div className="mx-auto w-full max-w-4xl">
          <PdfUploadPanel onUpload={handlePdfUpload} isUploading={pdfUploading} status={pdfStatus} />
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
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
        >
          <span aria-hidden="true">←</span>
          Back
        </button>

        <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Step 2</div>
      </div>

      <div className="mb-6 text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-clinical-700">Manual input</p>
        <h2 className="mt-3 text-3xl font-bold text-slate-900">Manual Patient Input</h2>
        <p className="mt-2 text-slate-600">Enter the patient details below.</p>
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
          Return
        </button>

        <button
          type="button"
          onClick={handlePredict}
          disabled={loading}
          className="inline-flex items-center justify-center rounded-xl bg-clinical-600 px-5 py-3 text-sm font-semibold text-white shadow-soft transition hover:bg-clinical-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {loading ? 'Predicting...' : 'Continue →'}
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
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
        >
          <span aria-hidden="true">←</span>
          Back
        </button>

        <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Result</div>
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
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-clinical-100 p-3 text-clinical-700">
                <Stethoscope className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-slate-900">CareGrid</h1>
                <p className="text-sm text-slate-600">Hospital Readmission Prediction</p>
              </div>
            </div>
            <div className="rounded-full border border-clinical-200 bg-clinical-50 px-3 py-1 text-sm font-semibold text-clinical-700">
              30-day risk model
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <section className="mb-8 grid gap-4 md:grid-cols-3">
          <div className="card flex items-center gap-4 p-5">
            <div className="rounded-xl bg-clinical-100 p-2 text-clinical-700"><Activity className="h-5 w-5" /></div>
            <div>
              <p className="text-sm text-slate-500">Model</p>
              <p className="text-lg font-semibold">XGBoost</p>
            </div>
          </div>
          <div className="card flex items-center gap-4 p-5">
            <div className="rounded-xl bg-emerald-100 p-2 text-emerald-700"><Activity className="h-5 w-5" /></div>
            <div>
              <p className="text-sm text-slate-500">Threshold</p>
              <p className="text-lg font-semibold">{(formData.threshold ?? 0.35).toFixed(2)}</p>
            </div>
          </div>
          <div className="card flex items-center gap-4 p-5">
            <div className="rounded-xl bg-amber-100 p-2 text-amber-700"><Activity className="h-5 w-5" /></div>
            <div>
              <p className="text-sm text-slate-500">Use case</p>
              <p className="text-lg font-semibold">Clinical prioritization</p>
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
