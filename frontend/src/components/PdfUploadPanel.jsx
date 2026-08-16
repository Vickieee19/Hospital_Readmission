import { useRef, useState } from 'react'
import { FileText, UploadCloud } from 'lucide-react'

export default function PdfUploadPanel({ onUpload, isUploading, status }) {
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
    if (!selectedFile) {
      return
    }

    onUpload(selectedFile)
  }

  return (
    <div className="card p-6 sm:p-8">
      <div className="mb-6 flex items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-clinical-700">Document intake</p>
          <h2 className="mt-2 text-2xl font-bold text-slate-900">Upload PDF</h2>
        </div>
        <div className="rounded-2xl bg-violet-50 p-3 text-violet-700">
          <FileText className="h-6 w-6" />
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex items-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-4">
          <UploadCloud className="h-5 w-5 text-slate-500" />
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileChange}
            className="w-full text-sm text-slate-600 file:mr-4 file:rounded-xl file:border-0 file:bg-clinical-600 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-white"
          />
        </div>

        <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
          {selectedFile ? `Selected file: ${selectedFile.name}` : 'No PDF selected yet.'}
        </div>

        <button
          type="button"
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          className="inline-flex w-full items-center justify-center rounded-xl bg-violet-600 px-5 py-3 text-sm font-semibold text-white shadow-soft transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {isUploading ? 'Uploading PDF...' : 'Upload PDF'}
        </button>

        {status?.message ? (
          <div
            className={`rounded-xl border px-3 py-2 text-sm ${
              status.type === 'success'
                ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                : 'border-rose-200 bg-rose-50 text-rose-700'
            }`}
          >
            {status.message}
          </div>
        ) : null}
      </div>
    </div>
  )
}
