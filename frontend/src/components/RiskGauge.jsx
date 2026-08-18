import React from 'react'

/**
 * RiskGauge Component (Light / White theme)
 * Semicircular risk gauge with three clinical risk zones (Green <35%, Yellow 35-52%, Red >=52%),
 * radial tick markers (0%, 20%, 40%, 60%, 80%, 100%), calibrated indicator needle,
 * large center percentage, and clinical cutoff threshold guide.
 */
export default function RiskGauge({
  probability = 0,
  threshold = 0.5227,
  riskLevel = 'Moderate',
  predictedYesNo = null,
}) {
  const percentage = Math.min(100, Math.max(0, Number((probability * 100).toFixed(1))))
  const thresholdPct = Number((threshold * 100).toFixed(1))
  const isYes = predictedYesNo === 'YES' || percentage >= thresholdPct

  // Semicircle geometry
  const width = 360
  const height = 210
  const cx = 180
  const cy = 175
  const radius = 135
  const strokeWidth = 24

  // Helper to convert polar to cartesian coordinates for SVG arcs (180° = Left, 0° = Right)
  const polarToCartesian = (centerX, centerY, r, angleInDegrees) => {
    const angleInRadians = ((180 - angleInDegrees) * Math.PI) / 180.0
    return {
      x: centerX + r * Math.cos(angleInRadians),
      y: centerY - r * Math.sin(angleInRadians),
    }
  }

  const describeArc = (x, y, r, startAngle, endAngle) => {
    const start = polarToCartesian(x, y, r, startAngle)
    const end = polarToCartesian(x, y, r, endAngle)
    const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1'
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`
  }

  const greenEndDeg = (35 / 100) * 180
  const yellowEndDeg = Math.min(180, (thresholdPct / 100) * 180)

  const greenArc = describeArc(cx, cy, radius, 0, greenEndDeg)
  const yellowArc = describeArc(cx, cy, radius, greenEndDeg, yellowEndDeg)
  const redArc = describeArc(cx, cy, radius, yellowEndDeg, 180)
  const backgroundTrack = describeArc(cx, cy, radius, 0, 180)

  const ticks = [
    { pct: 0, label: '0%' },
    { pct: 20, label: '20%' },
    { pct: 40, label: '40%' },
    { pct: 60, label: '60%' },
    { pct: 80, label: '80%' },
    { pct: 100, label: '100%' },
  ]

  const needleDeg = (percentage / 100) * 180
  const needleOuter = polarToCartesian(cx, cy, radius + 18, needleDeg)
  const needleInner = polarToCartesian(cx, cy, radius - strokeWidth - 6, needleDeg)

  const getRiskColor = () => {
    if (percentage >= thresholdPct) return '#dc2626' // Red
    if (percentage >= 35) return '#d97706' // Amber
    return '#16a34a' // Green
  }

  return (
    <div className="flex flex-col items-center justify-center p-2 text-center select-none">
      <div className="text-xs font-semibold tracking-wider text-slate-500 uppercase mb-1">
        30-Day Readmission Risk Score
      </div>

      <div className="relative w-full max-w-[360px]">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
          {/* Background Track */}
          <path
            d={backgroundTrack}
            fill="none"
            stroke="#e2e8f0"
            strokeWidth={strokeWidth + 2}
            strokeLinecap="round"
          />

          {/* Green Zone (<35%) */}
          <path
            d={greenArc}
            fill="none"
            stroke="#22c55e"
            strokeWidth={strokeWidth}
            className="transition-all duration-500"
          />

          {/* Yellow Zone (35% to Threshold) */}
          <path
            d={yellowArc}
            fill="none"
            stroke="#fde047"
            strokeWidth={strokeWidth}
            className="transition-all duration-500"
          />

          {/* Red Zone (Threshold to 100%) */}
          <path
            d={redArc}
            fill="none"
            stroke="#f87171"
            strokeWidth={strokeWidth}
            className="transition-all duration-500"
          />

          {/* Tick Marks and Labels */}
          {ticks.map(({ pct, label }) => {
            const tickDeg = (pct / 100) * 180
            const tickInner = polarToCartesian(cx, cy, radius - strokeWidth / 2 - 14, tickDeg)
            const tickOuter = polarToCartesian(cx, cy, radius + strokeWidth / 2 + 6, tickDeg)
            const labelPos = polarToCartesian(cx, cy, radius + strokeWidth / 2 + 18, tickDeg)

            return (
              <g key={pct}>
                <line
                  x1={tickInner.x}
                  y1={tickInner.y}
                  x2={tickOuter.x}
                  y2={tickOuter.y}
                  stroke="#94a3b8"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
                <text
                  x={labelPos.x}
                  y={labelPos.y + (pct === 0 || pct === 100 ? 4 : 2)}
                  fill="#475569"
                  fontSize="11"
                  fontWeight="600"
                  fontFamily="system-ui, -apple-system, sans-serif"
                  textAnchor={pct === 0 ? 'end' : pct === 100 ? 'start' : 'middle'}
                  dominantBaseline="central"
                >
                  {label}
                </text>
              </g>
            )
          })}

          {/* Indicator Needle */}
          <line
            x1={needleInner.x}
            y1={needleInner.y}
            x2={needleOuter.x}
            y2={needleOuter.y}
            stroke="#0f172a"
            strokeWidth="3.5"
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />

          {/* Center Pivot Circle */}
          <circle cx={cx} cy={cy} r="6" fill="#0f172a" stroke="#ffffff" strokeWidth="2" />
        </svg>

        {/* Center Percentage Display & Binary YES/NO Tag */}
        <div className="absolute inset-x-0 bottom-2 flex flex-col items-center justify-center space-y-0.5">
          <span
            className="text-4xl sm:text-5xl font-extrabold tracking-tight font-mono transition-colors duration-300"
            style={{ color: getRiskColor() }}
          >
            {percentage}%
          </span>
          <span
            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-extrabold uppercase tracking-wider ${
              isYes ? 'bg-rose-100 text-rose-800 border border-rose-200' : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
            }`}
          >
            <span>Readmission:</span>
            <span className="font-black underline">{isYes ? 'YES' : 'NO'}</span>
          </span>
        </div>
      </div>

      {/* Threshold and Zone Legend matching reference */}
      <div className="mt-3 text-[11px] font-medium text-slate-500">
        Green: Low (&lt;35%) • Yellow: Moderate (35-{thresholdPct}%) • Red: High Risk (&ge;{thresholdPct}%)
      </div>
    </div>
  )
}
