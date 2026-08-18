import React, { useRef, useState } from 'react'
import { FileText, UploadCloud, CheckCircle2, AlertCircle, Sparkles, BookOpen, ShieldCheck } from 'lucide-react'

export default function PdfUploadPanel({ onUpload, isUploading, status, uploadResult }) {
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

  const severity = uploadResult?.severity_assessment
  const guidelines = uploadResult?.retrieved_guidelines || []

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="flex items-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-5">
          <UploadCloud className="h-6 w-6 text-slate-500" />
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileChange}
            className="w-full text-sm text-slate-600 file:mr-4 file:rounded-xl file:border-0 file:bg-blue-600 file:px-4 file:py-2 file:text-xs file:font-bold file:text-white hover:file:bg-blue-700 cursor-pointer"
          />
        </div>

        <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
          <span>{selectedFile ? `Selected: ${selectedFile.name}` : 'Select a clinical lab report or discharge summary PDF.'}</span>
          {selectedFile && <span className="text-xs text-slate-500">{(selectedFile.size / 1024).toFixed(1)} KB</span>}
        </div>

        <button
          type="button"
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-xs transition-all hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
        >
          {isUploading ? (
            <>
              <Sparkles className="h-4 w-4 animate-spin" />
              <span>Analyzing Document...</span>
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              <span>Analyze Clinical Severity</span>
            </>
          )}
        </button>

        {status?.message && (
          <div
            className={`flex items-center gap-2 rounded-xl border px-4 py-3 text-sm ${
              status.type === 'success'
                ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                : 'border-rose-200 bg-rose-50 text-rose-800'
            }`}
          >
            {status.type === 'success' ? (
              <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
            ) : (
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-600" />
            )}
            <span>{status.message}</span>
          </div>
        )}
      </div>

      {/* RAG Severity Assessment Results */}
      {severity && (
        <div className="rounded-2xl border border-slate-200 bg-white space-y-6 p-6 sm:p-8 shadow-xs">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-200 pb-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-blue-600">Clinical Assessment</p>
              <h3 className="mt-1 text-xl font-bold text-slate-900">Severity Assessment & Medical Evidence</h3>
            </div>
            <span
              className={`inline-flex items-center rounded-full border px-3.5 py-1 text-sm font-bold ${
                severity.severity_level === 'Critical' || severity.severity_level === 'High'
                  ? 'border-rose-200 bg-rose-50 text-rose-700'
                  : severity.severity_level === 'Moderate'
                  ? 'border-amber-200 bg-amber-50 text-amber-700'
                  : 'border-emerald-200 bg-emerald-50 text-emerald-700'
              }`}
            >
              {severity.severity_level} Severity ({severity.severity_score}/10)
            </span>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-600">
                <ShieldCheck className="h-4 w-4 text-blue-600" />
                <span>Clinical Summary</span>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-slate-700">{severity.summary}</p>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-600">
                <BookOpen className="h-4 w-4 text-blue-600" />
                <span>Key Findings Detected</span>
              </div>
              <ul className="mt-2 space-y-1 text-sm text-slate-700">
                {(severity.key_findings || []).map((finding, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-blue-600">•</span>
                    <span>{finding}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Retrieved Guidelines */}
          {guidelines.length > 0 && (
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="mb-2.5 flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-blue-600" />
                <h4 className="text-sm font-semibold text-slate-900">Grounded Medical Guidelines Retrieved:</h4>
              </div>
              <div className="space-y-2">
                {guidelines.map((guide, idx) => (
                  <div key={idx} className="rounded-xl border border-slate-200 bg-white p-3 text-xs">
                    <div className="flex items-center justify-between font-semibold text-slate-800">
                      <span>{guide.source}</span>
                      <span className="rounded bg-blue-50 border border-blue-200 px-2 py-0.5 text-[11px] text-blue-700">
                        Similarity: {guide.similarity}
                      </span>
                    </div>
                    <p className="mt-1 text-slate-600">{guide.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
