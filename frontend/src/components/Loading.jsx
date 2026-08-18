import React from 'react'
import { Sparkles } from 'lucide-react'

export default function Loading() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-6 text-slate-700">
      <div className="relative flex h-12 w-12 items-center justify-center">
        <div className="absolute h-12 w-12 animate-spin rounded-full border-2 border-slate-200 border-t-blue-600" />
        <Sparkles className="h-5 w-5 text-blue-600 animate-pulse" />
      </div>
      <span className="text-sm font-semibold tracking-wide text-slate-700">
        Evaluating Encounter & Calculating Predictions...
      </span>
    </div>
  )
}
