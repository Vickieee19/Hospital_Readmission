export default function Loading() {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white p-4 text-slate-600 shadow-soft">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-clinical-200 border-t-clinical-600" />
      <span className="font-medium">Predicting...</span>
    </div>
  )
}
