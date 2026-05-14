import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis } from 'recharts'
import { FileText, TrendingUp, CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react'
import { api } from '@/api/client'
import type { ResultFile, EvaluationData, RecordEvaluation } from '@/types'

const STEP_COLORS = ['#1565C0', '#6A1B9A', '#2E7D32']

function MetricBadge({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value * 100)
  const color = pct >= 70 ? 'text-green-700 bg-green-50' : pct >= 50 ? 'text-amber-700 bg-amber-50' : 'text-red-700 bg-red-50'
  return (
    <div className={`rounded-lg p-3 text-center ${color}`}>
      <p className="text-2xl font-bold">{pct}%</p>
      <p className="text-xs font-medium mt-0.5">{label}</p>
    </div>
  )
}

function RecordRow({ rec }: { rec: RecordEvaluation }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <tr
        className="hover:bg-slate-50 cursor-pointer border-b border-slate-100"
        onClick={() => setOpen(o => !o)}
      >
        <td className="px-3 py-2 text-xs font-bold text-slate-700">{rec.patient_code}</td>
        <td className="px-3 py-2">
          <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-navy-100 text-navy-700">{rec.ground_truth}</span>
        </td>
        {[1, 2, 3].map(s => {
          const pred = rec[`step${s}_prediction` as keyof RecordEvaluation] as string | null
          const conc = rec[`step${s}_concordant` as keyof RecordEvaluation] as boolean | null
          const feas = rec[`step${s}_feasible` as keyof RecordEvaluation] as boolean
          return (
            <td key={s} className="px-3 py-2 text-center">
              {!feas ? (
                <span className="text-xs text-slate-300">—</span>
              ) : conc === true ? (
                <div className="flex items-center justify-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5 text-green-500" />
                  <span className="text-xs font-semibold text-slate-600">{pred}</span>
                </div>
              ) : (
                <div className="flex items-center justify-center gap-1">
                  <XCircle className="w-3.5 h-3.5 text-red-400" />
                  <span className="text-xs font-semibold text-red-600">{pred ?? '?'}</span>
                </div>
              )}
            </td>
          )
        })}
        <td className="px-3 py-2 text-slate-400">
          {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </td>
      </tr>
      {open && (
        <tr className="bg-slate-50">
          <td colSpan={6} className="px-4 py-3">
            <div className="grid grid-cols-3 gap-3 text-xs">
              {[1, 2, 3].map(s => {
                const conf = rec[`step${s}_confidence` as keyof RecordEvaluation] as number | null
                return (
                  <div key={s} className="bg-white rounded-lg p-2 border border-slate-200">
                    <p className="font-bold text-slate-500 mb-1">Step {s}</p>
                    <p>Predizione: <strong>{(rec[`step${s}_prediction` as keyof RecordEvaluation] as string) ?? '—'}</strong></p>
                    {conf != null && <p>Confidence: {(conf * 100).toFixed(0)}%</p>}
                    <p>Feasible: {rec[`step${s}_feasible` as keyof RecordEvaluation] ? 'Sì' : 'No'}</p>
                  </div>
                )
              })}
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function Results() {
  const [files, setFiles] = useState<ResultFile[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [evalData, setEvalData] = useState<EvaluationData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.getResults().then(setFiles)
  }, [])

  async function loadEval(filename: string) {
    setLoading(true)
    setSelected(filename)
    try {
      const data = await api.getEvaluation(filename)
      setEvalData(data)
    } finally {
      setLoading(false)
    }
  }

  const stepMetrics = evalData
    ? [1, 2, 3].map(s => {
        const m = evalData.metrics_per_step[`step${s}`]
        return {
          step: `Step ${s}`,
          accuracy: m ? Math.round(m.accuracy * 100) : 0,
          kappa: m ? Math.round(m.cohen_kappa * 100) : 0,
          color: STEP_COLORS[s - 1],
        }
      })
    : []

  const concordanceData = evalData?.summary
    ? [1, 2, 3].map(s => {
        const sm = evalData.summary[`step${s}` as 'step1' | 'step2' | 'step3']
        return {
          step: `Step ${s}`,
          concordant: sm.concordant_patients,
          feasible: sm.feasible_patients,
        }
      })
    : []

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-navy-700 to-navy-500 rounded-xl p-6 text-white shadow-lg">
        <h1 className="text-xl font-bold mb-1">Risultati & Metriche</h1>
        <p className="text-white/70 text-sm">Analisi di concordanza per step diagnostici</p>
      </div>

      {/* File list */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
        <p className="section-title">File di Risultati</p>
        {files.length === 0 ? (
          <div className="flex flex-col items-center py-8 text-slate-400">
            <FileText className="w-10 h-10 mb-2 opacity-30" />
            <p className="text-sm">Nessun risultato disponibile. Esegui prima un batch.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {files.map(f => (
              <button
                key={f.filename}
                onClick={() => loadEval(f.filename)}
                className={`w-full text-left flex items-center justify-between px-4 py-3 rounded-lg border transition-all ${selected === f.filename ? 'border-navy-500 bg-navy-50' : 'border-border hover:border-navy-300 hover:bg-slate-50'}`}
              >
                <div>
                  <p className="text-sm font-semibold text-slate-700">{f.model}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{f.timestamp} · {f.total_patients} pazienti</p>
                </div>
                <div className="flex gap-2 text-xs">
                  {[f.step1_accuracy, f.step2_accuracy, f.step3_accuracy].map((acc, i) => (
                    acc != null && (
                      <span key={i} className={`px-2 py-1 rounded font-bold ${acc >= 0.7 ? 'bg-green-100 text-green-700' : acc >= 0.5 ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'}`}>
                        S{i + 1}: {Math.round(acc * 100)}%
                      </span>
                    )
                  ))}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Evaluation details */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-navy-700" />
        </div>
      )}

      {evalData && !loading && (
        <>
          {/* Summary metrics */}
          <div className="grid grid-cols-3 gap-4">
            {[1, 2, 3].map(s => {
              const m = evalData.metrics_per_step[`step${s}`]
              const sm = evalData.summary[`step${s}` as 'step1']
              return (
                <div key={s} className="bg-white rounded-xl p-5 shadow-sm border-t-4" style={{ borderTopColor: STEP_COLORS[s - 1] }}>
                  <p className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: STEP_COLORS[s - 1] }}>Step {s}</p>
                  <div className="grid grid-cols-2 gap-2">
                    <MetricBadge value={m?.accuracy ?? 0} label="Accuracy" />
                    <MetricBadge value={m?.cohen_kappa ?? 0} label="Cohen κ" />
                  </div>
                  <p className="text-xs text-slate-500 mt-3">
                    {sm.concordant_patients}/{sm.feasible_patients} concordanti · {sm.feasible_patients} processati
                  </p>
                </div>
              )
            })}
          </div>

          {/* Charts */}
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
              <p className="section-title">Accuracy per Step</p>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={stepMetrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F0" />
                  <XAxis dataKey="step" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip formatter={(v: number) => [`${v}%`]} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.15)' }} />
                  <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                    {stepMetrics.map((s, i) => <Cell key={i} fill={s.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
              <p className="section-title">Pazienti Concordanti per Step</p>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={concordanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F0" />
                  <XAxis dataKey="step" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.15)' }} />
                  <Bar dataKey="feasible" fill="#E3F2FD" name="Processati" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="concordant" fill="#1565C0" name="Concordanti" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Per-patient table */}
          <div className="bg-white rounded-xl shadow-sm border border-border overflow-hidden">
            <div className="px-5 py-4 border-b border-border">
              <p className="section-title mb-0">Concordanza per Paziente</p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th className="px-3 py-2.5 text-left font-semibold text-slate-500 uppercase tracking-wider">Codice</th>
                    <th className="px-3 py-2.5 text-left font-semibold text-slate-500 uppercase tracking-wider">GT</th>
                    <th className="px-3 py-2.5 text-center font-semibold text-blue-600 uppercase tracking-wider">Step 1</th>
                    <th className="px-3 py-2.5 text-center font-semibold text-purple-600 uppercase tracking-wider">Step 2</th>
                    <th className="px-3 py-2.5 text-center font-semibold text-green-600 uppercase tracking-wider">Step 3</th>
                    <th className="px-3 py-2.5 w-8"></th>
                  </tr>
                </thead>
                <tbody>
                  {evalData.records.map(rec => (
                    <RecordRow key={rec.patient_code} rec={rec} />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
