import type { StepResult } from '@/types'

const DIAG_COLORS: Record<string, string> = {
  'AD':     '#1565C0',
  'VAD':    '#C62828',
  'FTD':    '#E65100',
  'PD':     '#2E7D32',
  'SCD':    '#F9A825',
  'LATE':   '#6A1B9A',
  'MIXED':  '#00695C',
  'AD-PPA': '#AD1457',
}

const PROB_FALLBACK: Record<string, number> = {
  ALTA: 0.40, MEDIA: 0.15, BASSA: 0.06, ESCLUSA: 0.0,
}

interface DiagProb {
  diagnosis: string
  label: string
  confidence: number
}

interface NodeLayout {
  diagnosis: string
  label: string
  confidence: number
  y: number
  h: number
  color: string
}

const SVG_W = 720
const SVG_H = 460
const NODE_W = 90
const COL_CENTERS = [110, 360, 610]
const TOP_PAD = 50
const BOT_PAD = 20
const USABLE_H = SVG_H - TOP_PAD - BOT_PAD
const NODE_GAP = 5
const MIN_NODE_H = 3
const LABEL_THRESHOLD = 18

function extractDiagProbs(step: StepResult | null): DiagProb[] {
  if (!step?.result || step.result.error) return []
  const all: DiagProb[] = []

  const p = step.result.primary_diagnosis
  if (p?.diagnosis) {
    all.push({
      diagnosis: p.diagnosis,
      label: p.label ?? p.diagnosis,
      confidence: p.confidence_score ?? 0.5,
    })
  }

  for (const d of step.result.differential_diagnoses ?? []) {
    if (!d?.diagnosis) continue
    const conf = d.confidence_score ?? PROB_FALLBACK[d.probability] ?? 0
    if (conf > 0) {
      all.push({ diagnosis: d.diagnosis, label: d.label ?? d.diagnosis, confidence: conf })
    }
  }

  const total = all.reduce((s, d) => s + d.confidence, 0)
  if (total <= 0) return []
  return all
    .map(d => ({ ...d, confidence: d.confidence / total }))
    .sort((a, b) => b.confidence - a.confidence)
}

function computeLayout(probs: DiagProb[]): NodeLayout[] {
  if (probs.length === 0) return []
  const active = probs.filter(p => p.confidence > 0.005)
  const totalGap = NODE_GAP * (active.length - 1)
  const availH = Math.max(0, USABLE_H - totalGap)

  let curY = TOP_PAD
  return active.map(p => {
    const h = Math.max(MIN_NODE_H, availH * p.confidence)
    const node: NodeLayout = {
      ...p,
      y: curY,
      h,
      color: DIAG_COLORS[p.diagnosis] ?? '#546E7A',
    }
    curY += h + NODE_GAP
    return node
  })
}

function flowPath(
  x1: number, cy1: number, h1: number,
  x2: number, cy2: number, h2: number,
): string {
  const midX = (x1 + x2) / 2
  const t1 = cy1 - h1 / 2
  const b1 = cy1 + h1 / 2
  const t2 = cy2 - h2 / 2
  const b2 = cy2 + h2 / 2
  return [
    `M ${x1} ${t1}`,
    `C ${midX} ${t1}, ${midX} ${t2}, ${x2} ${t2}`,
    `L ${x2} ${b2}`,
    `C ${midX} ${b2}, ${midX} ${b1}, ${x1} ${b1}`,
    'Z',
  ].join(' ')
}

interface Props {
  step1: StepResult | null
  step2: StepResult | null
  step3: StepResult | null
}

export default function AlluvialDiagram({ step1, step2, step3 }: Props) {
  const cols = [step1, step2, step3].map(s => {
    const probs = extractDiagProbs(s)
    return computeLayout(probs)
  })

  const stepLabels = ['Step 1\nDati Clinici', 'Step 2\nPlasma', 'Step 3\nLiquor']

  const nodeMap = (layout: NodeLayout[]) => {
    const m: Record<string, NodeLayout> = {}
    for (const n of layout) m[n.diagnosis] = n
    return m
  }

  const maps = cols.map(nodeMap)

  return (
    <div className="bg-white rounded-xl border border-border shadow-sm p-4">
      <h3 className="text-sm font-semibold text-slate-700 mb-3">
        Evoluzione Diagnostica — Alluvial Diagram
      </h3>
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        width="100%"
        style={{ maxHeight: 500, display: 'block' }}
      >
        {/* Column labels */}
        {COL_CENTERS.map((cx, ci) => (
          <g key={ci}>
            <text
              x={cx}
              y={22}
              textAnchor="middle"
              fontSize={12}
              fontWeight="600"
              fill="#475569"
            >
              {stepLabels[ci].split('\n')[0]}
            </text>
            <text
              x={cx}
              y={36}
              textAnchor="middle"
              fontSize={10}
              fill="#94a3b8"
            >
              {stepLabels[ci].split('\n')[1]}
            </text>
          </g>
        ))}

        {/* Flows between columns */}
        {[0, 1].map(pairIdx => {
          const leftMap = maps[pairIdx]
          const rightMap = maps[pairIdx + 1]
          const leftCx = COL_CENTERS[pairIdx] + NODE_W / 2
          const rightCx = COL_CENTERS[pairIdx + 1] - NODE_W / 2

          return Object.keys(leftMap).map(diag => {
            const L = leftMap[diag]
            const R = rightMap[diag]
            if (!L || !R) return null
            const cy1 = L.y + L.h / 2
            const cy2 = R.y + R.h / 2
            const path = flowPath(leftCx, cy1, L.h, rightCx, cy2, R.h)
            const color = DIAG_COLORS[diag] ?? '#546E7A'
            return (
              <path
                key={`flow-${pairIdx}-${diag}`}
                d={path}
                fill={color}
                fillOpacity={0.18}
                stroke={color}
                strokeOpacity={0.35}
                strokeWidth={0.5}
              />
            )
          })
        })}

        {/* Nodes */}
        {cols.map((layout, ci) => {
          const cx = COL_CENTERS[ci]
          return layout.map(node => (
            <g key={`node-${ci}-${node.diagnosis}`}>
              <rect
                x={cx - NODE_W / 2}
                y={node.y}
                width={NODE_W}
                height={node.h}
                fill={node.color}
                rx={3}
              />
              {node.h >= LABEL_THRESHOLD && (
                <text
                  x={cx}
                  y={node.y + node.h / 2 + 4}
                  textAnchor="middle"
                  fontSize={node.h >= 28 ? 11 : 9}
                  fontWeight="700"
                  fill="white"
                >
                  {node.diagnosis}
                </text>
              )}
              {node.h >= 36 && (
                <text
                  x={cx}
                  y={node.y + node.h / 2 + 16}
                  textAnchor="middle"
                  fontSize={8.5}
                  fill="rgba(255,255,255,0.85)"
                >
                  {(node.confidence * 100).toFixed(0)}%
                </text>
              )}
            </g>
          ))
        })}
      </svg>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap gap-2">
        {Object.entries(DIAG_COLORS).map(([diag, color]) => (
          <div key={diag} className="flex items-center gap-1 text-xs text-slate-600">
            <span
              className="inline-block w-3 h-3 rounded-sm flex-shrink-0"
              style={{ backgroundColor: color }}
            />
            {diag}
          </div>
        ))}
      </div>
    </div>
  )
}
