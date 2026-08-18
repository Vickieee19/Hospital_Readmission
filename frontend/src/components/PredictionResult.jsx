import React, { useState } from 'react'
import {
  PhoneCall,
  Pill,
  CalendarCheck,
  FileText,
  Home,
  Activity,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  BarChart2,
  Info,
  Printer,
  ChevronDown,
  ChevronUp,
  BarChart3,
  Search,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts'
import RiskGauge from './RiskGauge'

const PROTOCOL_ICONS = {
  phone: <PhoneCall className="h-5 w-5 text-blue-600" />,
  pill: <Pill className="h-5 w-5 text-indigo-600" />,
  calendar: <CalendarCheck className="h-5 w-5 text-emerald-600" />,
  stethoscope: <FileText className="h-5 w-5 text-amber-600" />,
  home: <Home className="h-5 w-5 text-purple-600" />,
  activity: <Activity className="h-5 w-5 text-rose-600" />,
}

export default function PredictionResult({ result }) {
  if (!result) return null

  const [showAllFactors, setShowAllFactors] = useState(true)

  const probabilityPct = ((result.probability || 0) * 100).toFixed(1)
  const isHighRisk = result.prediction === 1 || result.risk_level === 'High'
  const isModerateRisk = result.risk_level === 'Moderate'

  // SHAP Chart data
  const shapData = (result.shap_values || []).map((item) => {
    const val = Number(item.impact || 0)
    return {
      name: item.feature,
      impact: val,
      absImpact: Math.abs(val),
      isPositive: val >= 0,
    }
  })

  // Domain radar data
  const radarData = (result.domain_scores || [
    { domain: 'Prior Utilization', score: 85 },
    { domain: 'Polypharmacy', score: 90 },
    { domain: 'Stay Acuity', score: 65 },
    { domain: 'Chronic Complexity', score: 70 },
    { domain: 'Age Vulnerability', score: 80 },
  ]).map((d) => ({
    subject: d.domain,
    patient: d.score,
    baseline: 30,
    fullMark: 100,
  }))

  const printReport = () => {
    window.print()
  }

  return (
    <div className="space-y-8 text-slate-900 print:text-black">
      {/* ─────────────────────────────────────────────────────────────
          1. HEADER: Assessment Results
      ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-200 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-600 border border-blue-100 shadow-xs">
            <BarChart2 className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">Assessment Results</h2>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={printReport}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 hover:text-slate-900"
          >
            <Printer className="h-4 w-4" />
            <span>Export / Print Report</span>
          </button>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────
          2. TWO-COLUMN TOP SECTION: Verdict Banner + Risk Gauge
      ───────────────────────────────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr] items-stretch">
        {/* Left Column: Verdict Banner */}
        <div
          className={`flex flex-col justify-center rounded-2xl p-6 sm:p-7 border transition-all shadow-sm ${
            isHighRisk
              ? 'border-rose-200 bg-[#fff5f5] text-rose-950'
              : isModerateRisk
              ? 'border-amber-200 bg-[#fffbeb] text-amber-950'
              : 'border-emerald-200 bg-[#f0fdf4] text-emerald-950'
          }`}
        >
          <div>
            {/* Risk Title with Clinical SVG Indicator Icon */}
            <div className="flex items-center gap-3">
              <div
                className={`flex h-9 w-9 items-center justify-center rounded-xl ${
                  isHighRisk
                    ? 'bg-rose-100 text-rose-700'
                    : isModerateRisk
                    ? 'bg-amber-100 text-amber-700'
                    : 'bg-emerald-100 text-emerald-700'
                }`}
              >
                {isHighRisk ? (
                  <ShieldAlert className="h-5 w-5" />
                ) : isModerateRisk ? (
                  <AlertTriangle className="h-5 w-5" />
                ) : (
                  <CheckCircle2 className="h-5 w-5" />
                )}
              </div>
              <h3
                className={`text-xl sm:text-2xl font-bold uppercase tracking-wider ${
                  isHighRisk ? 'text-rose-700' : isModerateRisk ? 'text-amber-700' : 'text-emerald-700'
                }`}
              >
                {isHighRisk
                  ? 'HIGH RISK OF READMISSION'
                  : isModerateRisk
                  ? 'MODERATE RISK OF READMISSION'
                  : 'LOW RISK OF READMISSION'}
              </h3>
            </div>

            {/* Probability & Action Required Subtitle */}
            <div className="mt-3 text-sm font-semibold tracking-wide text-slate-800">
              Predicted Probability: <span className="font-mono text-base font-bold text-slate-900">{probabilityPct}%</span>
              {' • '}
              <span
                className={`font-semibold ${
                  isHighRisk ? 'text-rose-700' : isModerateRisk ? 'text-amber-700' : 'text-emerald-700'
                }`}
              >
                {isHighRisk
                  ? 'Action Required'
                  : isModerateRisk
                  ? 'Enhanced Follow-Up Advised'
                  : 'Standard Discharge Safe'}
              </span>
            </div>

            {/* Clinical Description */}
            <p className="mt-3 text-sm leading-relaxed text-slate-700">
              {isHighRisk
                ? 'This patient has a high likelihood of unplanned 30-day hospital readmission. Proactive discharge interventions must be implemented before discharge.'
                : isModerateRisk
                ? 'This patient exhibits moderate vulnerability for hospital recidivism. Targeted post-discharge follow-up and medication review recommended.'
                : 'This patient has a low probability of 30-day readmission. Standard outpatient discharge protocols and routine check-ins are appropriate.'}
            </p>
          </div>
        </div>

        {/* Right Column: Calibrated Semi-Circle Risk Gauge */}
        <div className="flex flex-col items-center justify-center rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <RiskGauge
            probability={result.probability || 0}
            threshold={result.threshold || 0.5227}
            riskLevel={result.risk_level || 'Moderate'}
          />
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────
          3. WHY THIS PATIENT RECEIVED THIS SCORE (Factor Cards)
      ───────────────────────────────────────────────────────────── */}
      <div className="space-y-3.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-lg font-bold text-slate-900 sm:text-xl">
            <Search className="h-5 w-5 text-slate-700" />
            <span>Why this patient received this score:</span>
          </div>

          {result.contributing_factors && result.contributing_factors.length > 4 && (
            <button
              type="button"
              onClick={() => setShowAllFactors(!showAllFactors)}
              className="inline-flex items-center gap-1 text-xs font-semibold text-blue-600 hover:text-blue-700"
            >
              {showAllFactors ? (
                <>
                  <span>Show Top Factors</span>
                  <ChevronUp className="h-3.5 w-3.5" />
                </>
              ) : (
                <>
                  <span>Show All ({result.contributing_factors.length})</span>
                  <ChevronDown className="h-3.5 w-3.5" />
                </>
              )}
            </button>
          )}
        </div>

        <div className="space-y-2.5">
          {(result.contributing_factors || [])
            .slice(0, showAllFactors ? undefined : 4)
            .map((factor, idx) => {
              const [title, ...descParts] = factor.text.split(':')
              const description = descParts.join(':')

              return (
                <div
                  key={idx}
                  className={`flex items-start gap-3.5 rounded-xl border p-4 text-sm transition-all duration-200 shadow-xs ${
                    factor.is_risk
                      ? 'border-rose-200 bg-[#fff5f5] text-slate-800'
                      : 'border-emerald-200 bg-[#f0fdf4] text-slate-800'
                  }`}
                >
                  <div className="mt-0.5 shrink-0">
                    {factor.is_risk ? (
                      <AlertTriangle className="h-4 w-4 text-rose-600" />
                    ) : (
                      <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    )}
                  </div>
                  <div className="leading-relaxed">
                    {description ? (
                      <>
                        <strong className="font-bold text-slate-900">{title}:</strong>
                        <span className="text-slate-700">{description}</span>
                      </>
                    ) : (
                      <span className="text-slate-700">{factor.text}</span>
                    )}
                  </div>
                </div>
              )
            })}
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────
          4. CHARTS & GRAPHS: Clinical Explainability
      ───────────────────────────────────────────────────────────── */}
      <div className="space-y-6 pt-4 border-t border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-blue-600" />
              <h3 className="text-xl font-bold text-slate-900">Model Explainability & Feature Contribution Breakdown</h3>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Detailed machine learning attribution analysis showing exact factor weights and clinical risk benchmarks.
            </p>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Chart 1: SHAP Feature Attribution Bar Chart */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-sm flex flex-col justify-between">
            <div className="mb-4">
              <div className="flex items-center justify-between">
                <h4 className="text-base font-bold text-slate-900">Feature Attribution (SHAP Importance)</h4>
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Log-Odds Shift
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                <span className="text-rose-600 font-semibold">Red (+)</span> increases predicted risk;{' '}
                <span className="text-blue-600 font-semibold">Blue (-)</span> lowers readmission risk.
              </p>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={shapData}
                  layout="vertical"
                  margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                  <XAxis
                    type="number"
                    stroke="#94a3b8"
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    domain={['auto', 'auto']}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={150}
                    stroke="#94a3b8"
                    tick={{ fontSize: 10, fill: '#334155' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#ffffff',
                      borderColor: '#e2e8f0',
                      borderRadius: '12px',
                      color: '#0f172a',
                      fontSize: '12px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    }}
                    formatter={(val) => [
                      `${Number(val) > 0 ? '+' : ''}${Number(val).toFixed(4)}`,
                      'SHAP Value',
                    ]}
                  />
                  <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
                    {shapData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.isPositive ? '#ef4444' : '#2563eb'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-3 flex items-center justify-between text-[11px] text-slate-500 border-t border-slate-100 pt-3">
              <div className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-full bg-rose-500" />
                <span>Elevates Risk (+ Attribution)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-full bg-blue-600" />
                <span>Protective (- Attribution)</span>
              </div>
            </div>
          </div>

          {/* Chart 2: Clinical Risk Multi-Domain Radar Chart */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-sm flex flex-col justify-between">
            <div className="mb-4">
              <div className="flex items-center justify-between">
                <h4 className="text-base font-bold text-slate-900">Clinical Risk Domain Balance</h4>
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  0–100 Scale
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Multi-dimensional evaluation of patient encounter across 5 clinical pillars.
              </p>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} outerRadius="75%">
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis
                    dataKey="subject"
                    tick={{ fill: '#334155', fontSize: 11, fontWeight: 500 }}
                  />
                  <PolarRadiusAxis
                    angle={30}
                    domain={[0, 100]}
                    stroke="#94a3b8"
                    tick={{ fill: '#64748b', fontSize: 9 }}
                  />
                  <Radar
                    name="Patient Encounter"
                    dataKey="patient"
                    stroke="#ef4444"
                    fill="#ef4444"
                    fillOpacity={0.35}
                  />
                  <Radar
                    name="Low-Risk Baseline"
                    dataKey="baseline"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.15}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                    formatter={(val) => <span className="text-slate-700">{val}</span>}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#ffffff',
                      borderColor: '#e2e8f0',
                      borderRadius: '12px',
                      color: '#0f172a',
                      fontSize: '12px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-3 text-[11px] text-slate-500 border-t border-slate-100 pt-3">
              Higher area overlap indicates compounded multi-system vulnerability.
            </div>
          </div>
        </div>

        {/* ─────────────────────────────────────────────────────────────
            5. TAILORED READMISSION PREVENTION PROTOCOLS
        ───────────────────────────────────────────────────────────── */}
        {result.prevention_protocols && result.prevention_protocols.length > 0 && (
          <div className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-sm">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-50 border border-blue-200 text-blue-600">
                <ShieldCheck className="h-5 w-5" />
              </div>
              <div>
                <h4 className="text-lg font-bold text-slate-900">Tailored Readmission Prevention Protocols</h4>
                <p className="text-xs text-slate-500">
                  Evidence-based clinical actions recommended prior to and immediately following discharge:
                </p>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {result.prevention_protocols.map((protocol, idx) => (
                <div
                  key={idx}
                  className="flex flex-col justify-between rounded-xl border border-slate-200 bg-slate-50/70 p-4 transition-all duration-200 hover:border-slate-300 hover:bg-white hover:shadow-sm"
                >
                  <div>
                    <div className="mb-2.5 flex items-center gap-2.5">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white border border-slate-200 shadow-xs">
                        {PROTOCOL_ICONS[protocol.icon] || <Activity className="h-4 w-4 text-blue-600" />}
                      </div>
                      <h5 className="font-semibold text-sm text-slate-900 leading-snug">{protocol.title}</h5>
                    </div>
                    <p className="text-xs leading-relaxed text-slate-600">{protocol.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─────────────────────────────────────────────────────────────
            7. CLINICAL DECISION SUPPORT DISCLAIMER
        ───────────────────────────────────────────────────────────── */}
        <div className="flex items-start gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs leading-relaxed text-slate-600">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
          <div>
            <strong className="font-semibold text-slate-800">Clinical Decision Support Note:</strong>{' '}
            {result.disclaimer ||
              'Feature impacts describe each parameter’s relative contribution to the model’s predicted risk score compared to baseline averages. They do not establish direct clinical causality.'}
          </div>
        </div>
      </div>
    </div>
  )
}
