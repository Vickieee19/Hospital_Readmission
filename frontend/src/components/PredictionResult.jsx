import {
  AlertTriangle,
  CircleCheckBig,
  ShieldAlert,
  CheckCircle2,
  AlertCircle,
  PhoneCall,
  Pill,
  CalendarCheck,
  FileText,
  Home,
  Activity,
  Layers,
  ShieldCheck,
  Info,
} from 'lucide-react'
import { BarChart, Bar, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

function RiskBadge({ level }) {
  const palette = {
    High: 'bg-rose-100 text-rose-700 ring-rose-200',
    Moderate: 'bg-amber-100 text-amber-700 ring-amber-200',
    Low: 'bg-emerald-100 text-emerald-700 ring-emerald-200',
  }

  return (
    <span className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ring-1 ${palette[level] || palette.Low}`}>
      {level} Risk
    </span>
  )
}

const PROTOCOL_ICONS = {
  phone: <PhoneCall className="h-5 w-5 text-blue-600" />,
  pill: <Pill className="h-5 w-5 text-indigo-600" />,
  calendar: <CalendarCheck className="h-5 w-5 text-emerald-600" />,
  stethoscope: <FileText className="h-5 w-5 text-amber-600" />,
  home: <Home className="h-5 w-5 text-purple-600" />,
  activity: <Activity className="h-5 w-5 text-rose-600" />,
  // Fallbacks
  '📞': <PhoneCall className="h-5 w-5 text-blue-600" />,
  '💊': <Pill className="h-5 w-5 text-indigo-600" />,
  '📅': <CalendarCheck className="h-5 w-5 text-emerald-600" />,
  '🩺': <FileText className="h-5 w-5 text-amber-600" />,
  '🏡': <Home className="h-5 w-5 text-purple-600" />,
  '🩸': <Activity className="h-5 w-5 text-rose-600" />,
}

export default function PredictionResult({ result }) {
  if (!result) return null

  const probabilityPct = ((result.probability || 0) * 100).toFixed(1)
  const thresholdPct = ((result.threshold || 0) * 100).toFixed(1)
  const chartData = (result.shap_values || []).map((item) => ({
    name: item.feature,
    impact: Number(item.impact || 0),
  }))

  const isHighRisk = result.prediction === 1 || result.risk_level === 'High'

  return (
    <div className="space-y-6">
      {/* Verdict Header Banner */}
      <div
        className={`rounded-2xl border p-6 transition-all ${
          isHighRisk
            ? 'border-rose-300 bg-rose-50/80 text-rose-900'
            : 'border-emerald-300 bg-emerald-50/80 text-emerald-900'
        }`}
      >
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3.5">
            <div
              className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl ${
                isHighRisk ? 'bg-rose-100 text-rose-600' : 'bg-emerald-100 text-emerald-600'
              }`}
            >
              {isHighRisk ? <AlertTriangle className="h-6 w-6" /> : <CircleCheckBig className="h-6 w-6" />}
            </div>
            <div>
              <h2 className="text-xl font-bold">
                {isHighRisk ? 'High Risk of 30-Day Readmission' : 'Low Risk (Routine Discharge Safe)'}
              </h2>
              <p className="text-sm opacity-90">
                {isHighRisk
                  ? `Predicted Probability: ${probabilityPct}% — Proactive discharge interventions recommended.`
                  : `Predicted Probability: ${probabilityPct}% — Standard outpatient follow-up appropriate.`}
              </p>
            </div>
          </div>
          {result.risk_level && <RiskBadge level={result.risk_level} />}
        </div>
      </div>

      {/* Top Metric Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <CircleCheckBig className="h-4 w-4 text-clinical-600" />
            Decision Status
          </div>
          <div className="mt-3 text-2xl font-bold text-slate-900">
            {result.prediction === 1 ? 'High Risk Prioritization' : 'Standard Routine Follow-up'}
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <ShieldAlert className="h-4 w-4 text-clinical-600" />
            Readmission Probability
          </div>
          <div className="mt-3 text-3xl font-extrabold text-slate-900">{probabilityPct}%</div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <AlertTriangle className="h-4 w-4 text-clinical-600" />
            Operating Cutoff Threshold
          </div>
          <div className="mt-3 text-3xl font-extrabold text-slate-900">{thresholdPct}%</div>
        </div>
      </div>

      {/* Contributing Factors Section */}
      {result.contributing_factors && result.contributing_factors.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-3.5 flex items-center gap-2">
            <Layers className="h-4 w-4 text-clinical-600" />
            <h3 className="text-base font-bold text-slate-900">Key Contributing Factors</h3>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {result.contributing_factors.map((factor, idx) => (
              <div
                key={idx}
                className={`flex items-start gap-3 rounded-xl border p-3.5 text-sm ${
                  factor.is_risk
                    ? 'border-rose-200 bg-rose-50/70 text-rose-800'
                    : 'border-emerald-200 bg-emerald-50/70 text-emerald-800'
                }`}
              >
                {factor.is_risk ? (
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
                ) : (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
                )}
                <span>{factor.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* SHAP Visualizations */}
      <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="mb-4 text-base font-bold text-slate-900">Feature Attribution (SHAP Importance)</h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={120} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip formatter={(value) => `${Number(value).toFixed(3)}`} />
                <Bar dataKey="impact" fill="#2563eb" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="mb-4 text-base font-bold text-slate-900">Top Feature Contributors</h3>
          <div className="space-y-2.5">
            {(result.shap_values || []).slice(0, 6).map((item) => (
              <div key={item.feature} className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50/80 p-3 text-sm">
                <span className="font-medium text-slate-700">{item.feature}</span>
                <span className={`font-semibold ${item.impact >= 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {item.impact >= 0 ? '+' : ''}
                  {item.impact.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tailored Readmission Prevention Protocols */}
      {result.prevention_protocols && result.prevention_protocols.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-clinical-600" />
            <div>
              <h3 className="text-lg font-bold text-slate-900">Tailored Readmission Prevention Protocols</h3>
              <p className="text-sm text-slate-500">
                Evidence-based clinical actions recommended prior to and immediately following discharge:
              </p>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {result.prevention_protocols.map((protocol, idx) => (
              <div
                key={idx}
                className="flex items-start gap-3.5 rounded-xl border border-slate-200 bg-slate-50/50 p-4 transition-all hover:border-slate-300 hover:bg-white hover:shadow-sm"
              >
                <div className="rounded-xl bg-white p-2.5 shadow-sm">
                  {PROTOCOL_ICONS[protocol.icon] || <Activity className="h-5 w-5 text-blue-600" />}
                </div>
                <div>
                  <h4 className="font-semibold text-slate-900">{protocol.title}</h4>
                  <p className="mt-1 text-xs leading-relaxed text-slate-600">{protocol.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clinical Disclaimer */}
      <div className="flex items-start gap-2.5 rounded-xl border border-blue-100 bg-blue-50/80 p-4 text-xs leading-relaxed text-slate-600">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
        <div>
          <strong>Clinical Decision Support Note:</strong> {result.disclaimer}
        </div>
      </div>
    </div>
  )
}
