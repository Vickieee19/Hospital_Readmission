import { AlertTriangle, CircleCheckBig, ShieldAlert } from 'lucide-react'
import { BarChart, Bar, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

function RiskBadge({ level }) {
  const palette = {
    High: 'bg-rose-100 text-rose-700 ring-rose-200',
    Moderate: 'bg-amber-100 text-amber-700 ring-amber-200',
    Low: 'bg-emerald-100 text-emerald-700 ring-emerald-200',
  }

  return (
    <span className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ring-1 ${palette[level] || palette.Low}`}>
      {level}
    </span>
  )
}

export default function PredictionResult({ result }) {
  if (!result) return null

  const probabilityPct = ((result.probability || 0) * 100).toFixed(1)
  const thresholdPct = ((result.threshold || 0) * 100).toFixed(1)
  const chartData = (result.shap_values || []).map((item) => ({
    name: item.feature,
    impact: Number(item.impact || 0),
  }))

  return (
    <div className="card p-6 sm:p-8">
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-clinical-700">Prediction result</p>
          <h2 className="mt-2 text-2xl font-bold text-slate-900">Readmission risk</h2>
        </div>
        {result.risk_level ? <RiskBadge level={result.risk_level} /> : null}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <CircleCheckBig className="h-4 w-4 text-clinical-600" />
            Prediction
          </div>
          <div className="mt-4 text-3xl font-bold text-slate-900">{result.prediction === 1 ? 'High risk' : 'Low risk'}</div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <ShieldAlert className="h-4 w-4 text-clinical-600" />
            Probability
          </div>
          <div className="mt-4 text-3xl font-bold text-slate-900">{probabilityPct}%</div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <AlertTriangle className="h-4 w-4 text-clinical-600" />
            Risk threshold
          </div>
          <div className="mt-4 text-3xl font-bold text-slate-900">{thresholdPct}%</div>
        </div>
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <h3 className="mb-4 text-lg font-semibold text-slate-800">Feature impact</h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 12, right: 18, left: 8, bottom: 12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" stroke="#64748b" />
                <YAxis type="category" dataKey="name" width={110} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip formatter={(value) => `${Number(value).toFixed(3)}`} />
                <Bar dataKey="impact" fill="#2563eb" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <h3 className="mb-4 text-lg font-semibold text-slate-800">Top contributors</h3>
          <div className="space-y-3">
            {(result.shap_values || []).slice(0, 6).map((item) => (
              <div key={item.feature} className="rounded-xl border border-slate-200 bg-white p-3">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-sm font-medium text-slate-700">{item.feature}</span>
                  <span className={`text-sm font-semibold ${item.impact >= 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                    {item.impact >= 0 ? '+' : ''}{item.impact.toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-6 rounded-2xl border border-blue-100 bg-blue-50 p-4 text-sm text-slate-700">
        {result.disclaimer}
      </div>
    </div>
  )
}
