import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ChevronRight, Search, User, Brain, FlaskConical, CheckCircle, XCircle, AlertTriangle, Loader2 } from 'lucide-react'
import { api } from '@/api/client'
import type { PatientSummary, PatientDetail, StepResult } from '@/types'

const PROB_BADGE: Record<string, string> = {
  ALTA: 'badge-alta', MEDIA: 'badge-media', BASSA: 'badge-bassa', ESCLUSA: 'badge-esclusa',
}

function ProbBadge({ prob }: { prob: string }) {
  return <span className={PROB_BADGE[prob] ?? 'badge-esclusa'}>{prob}</span>
}

function StepCard({
  step, label, color, result, loading, onRun, disabled,
}: {
  step: number; label: string; color: string
  result: StepResult | null; loading: boolean
  onRun: () => void; disabled: boolean
}) {
  const primary = result?.result?.primary_diagnosis
  const feasible = result?.feasible
  return (
    <div className={`bg-white rounded-xl border-t-4 shadow-sm`} style={{ borderTopColor: color }}>
      <div className="p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <span className="text-xs font-bold uppercase tracking-widest" style={{ color }}>{`Step ${step}`}</span>
            <h3 className="text-sm font-semibold text-slate-700 mt-0.5">{label}</h3>
          </div>
          <button
            onClick={onRun}
            disabled={disabled || loading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white transition-opacity disabled:opacity-40"
            style={{ backgroundColor: color }}
          >
            {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Brain className="w-3 h-3" />}
            {loading ? 'Analisi...' : 'Esegui'}
          </button>
        </div>

        {result === null && !loading && (
          <p className="text-xs text-slate-400 italic">In attesa di esecuzione...</p>
        )}

        {result !== null && !result.feasible && (
          <div className="flex items-start gap-2 text-amber-700 bg-amber-50 rounded-lg p-3 text-xs">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{result.skip_reason}</span>
          </div>
        )}

        {result?.result?.error && (
          <div className="flex items-start gap-2 text-red-700 bg-red-50 rounded-lg p-3 text-xs">
            <XCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{result.result.error}</span>
          </div>
        )}

        {result?.feasible && primary && (
          <div className="space-y-3">
            <div className="bg-slate-50 rounded-lg p-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-slate-500">Diagnosi primaria</span>
                <ProbBadge prob={primary.probability} />
              </div>
              <p className="text-base font-bold text-navy-700">{primary.diagnosis}</p>
              <p className="text-xs text-slate-500 mt-0.5">{primary.label}</p>
              {primary.confidence_score != null && (
                <div className="mt-2">
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>Confidence</span>
                    <span>{(primary.confidence_score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-200 rounded-full">
                    <div className="h-1.5 rounded-full transition-all" style={{ width: `${primary.confidence_score * 100}%`, backgroundColor: color }} />
                  </div>
                </div>
              )}
            </div>

            {result.result?.differential_diagnoses && result.result.differential_diagnoses.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-400 mb-1.5">Diagnosi differenziale</p>
                <div className="space-y-1">
                  {result.result.differential_diagnoses.map((d, i) => (
                    <div key={i} className="flex items-center justify-between py-1 px-2 rounded bg-slate-50 text-xs">
                      <span className="font-medium text-slate-600">{d.diagnosis}</span>
                      <ProbBadge prob={d.probability} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {primary.reasoning && (
              <details className="text-xs">
                <summary className="cursor-pointer text-navy-600 font-medium hover:text-navy-800">Ragionamento clinico</summary>
                <p className="mt-2 text-slate-600 leading-relaxed bg-slate-50 rounded p-2">{primary.reasoning}</p>
              </details>
            )}

            <p className="text-xs text-slate-400">Durata: {result.duration_s}s · Modello: {result.model_used}</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default function PatientAnalysis() {
  const { code } = useParams<{ code: string }>()
  const navigate = useNavigate()
  const [patients, setPatients] = useState<PatientSummary[]>([])
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<PatientDetail | null>(null)
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState('')
  const [step1, setStep1] = useState<StepResult | null>(null)
  const [step2, setStep2] = useState<StepResult | null>(null)
  const [step3, setStep3] = useState<StepResult | null>(null)
  const [loading, setLoading] = useState<Record<number, boolean>>({ 1: false, 2: false, 3: false })

  useEffect(() => {
    api.getPatients().then(setPatients)
    api.getModels().then(ms => {
      const names = ms.map(m => m.name)
      setModels(names)
      if (names.length > 0) setModel(names[0])
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (code) loadPatient(code)
  }, [code])

  async function loadPatient(c: string) {
    const p = await api.getPatient(c)
    setSelected(p)
    setStep1(null); setStep2(null); setStep3(null)
    navigate(`/patients/${c}`, { replace: true })
  }

  async function runStep(step: number) {
    if (!selected || !model) return
    setLoading(l => ({ ...l, [step]: true }))
    try {
      const res = await api.runStep({
        patient_code: selected.codice,
        step,
        model,
        step1_result: step >= 2 ? step1?.result ?? null : null,
        step2_result: step >= 3 ? step2?.result ?? null : null,
      })
      if (step === 1) setStep1(res)
      if (step === 2) setStep2(res)
      if (step === 3) setStep3(res)
    } finally {
      setLoading(l => ({ ...l, [step]: false }))
    }
  }

  const filtered = patients.filter(p =>
    p.codice.toLowerCase().includes(search.toLowerCase()) ||
    p.diagnosis_gt.toLowerCase().includes(search.toLowerCase())
  )

  const finalDiag = step3?.result?.primary_diagnosis?.diagnosis
    ?? step2?.result?.primary_diagnosis?.diagnosis
    ?? step1?.result?.primary_diagnosis?.diagnosis
  const concordant = finalDiag && selected ? finalDiag.toUpperCase() === selected.diagnosis_gt.toUpperCase() : null

  return (
    <div className="flex gap-6 h-full">
      {/* Patient list */}
      <div className="w-64 flex-shrink-0">
        <div className="bg-white rounded-xl shadow-sm border border-border overflow-hidden">
          <div className="p-3 border-b border-border">
            <div className="relative">
              <Search className="absolute left-2.5 top-2 w-4 h-4 text-slate-400" />
              <input
                className="w-full pl-8 pr-3 py-1.5 text-xs rounded-lg border border-border bg-slate-50 focus:outline-none focus:ring-2 focus:ring-navy-500"
                placeholder="Cerca paziente..."
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
          </div>
          <div className="overflow-y-auto max-h-[calc(100vh-200px)]">
            {filtered.map(p => (
              <button
                key={p.codice}
                onClick={() => loadPatient(p.codice)}
                className={`w-full text-left px-3 py-2.5 border-b border-border hover:bg-slate-50 transition-colors ${selected?.codice === p.codice ? 'bg-blue-50 border-l-2 border-l-navy-500' : ''}`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-700">{p.codice}</span>
                  <span className="text-xs font-semibold px-1.5 py-0.5 rounded bg-navy-50 text-navy-700">{p.diagnosis_gt}</span>
                </div>
                <div className="flex items-center gap-2 mt-0.5 text-xs text-slate-400">
                  <span>{p.age}a {p.gender}</span>
                  {p.has_plasma && <span className="text-purple-400">P</span>}
                  {p.has_csf && <span className="text-green-500">L</span>}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto space-y-4">
        {!selected ? (
          <div className="flex flex-col items-center justify-center h-64 text-slate-400">
            <User className="w-12 h-12 mb-3 opacity-20" />
            <p>Seleziona un paziente dalla lista</p>
          </div>
        ) : (
          <>
            {/* Patient header */}
            <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-lg font-bold text-navy-700">{selected.codice}</span>
                    <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-navy-100 text-navy-700">{selected.diagnosis_gt}</span>
                    {concordant !== null && (
                      <span className={concordant ? 'badge-concordant' : 'badge-discordant'}>
                        {concordant ? '✓ Concordante' : '✗ Discordante'}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-slate-500">{selected.age} anni · {selected.gender} · MMSE: {selected.mmse ?? 'N/D'}/30 · Edu: {selected.anni_edu ?? '?'} anni</p>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={model}
                    onChange={e => setModel(e.target.value)}
                    className="text-xs border border-border rounded-lg px-2 py-1.5 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-navy-500"
                  >
                    {models.length === 0 && <option value="">Nessun modello</option>}
                    {models.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                <div>
                  <p className="text-slate-400 font-semibold uppercase tracking-wider mb-1">Fattori di rischio</p>
                  <div className="flex flex-wrap gap-1">
                    {selected.ipertensione && <span className="px-1.5 py-0.5 bg-red-50 text-red-700 rounded">Ipertensione</span>}
                    {selected.diabete && <span className="px-1.5 py-0.5 bg-red-50 text-red-700 rounded">Diabete</span>}
                    {selected.cardiovascolare && <span className="px-1.5 py-0.5 bg-red-50 text-red-700 rounded">Cardiovascolare</span>}
                    {selected.dislipidemia && <span className="px-1.5 py-0.5 bg-amber-50 text-amber-700 rounded">Dislipidemia</span>}
                    {selected.fam && <span className="px-1.5 py-0.5 bg-purple-50 text-purple-700 rounded">Familiarità</span>}
                    {selected.fumo && <span className="px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded">Fumo</span>}
                    {!selected.ipertensione && !selected.diabete && !selected.cardiovascolare && !selected.dislipidemia && !selected.fam && !selected.fumo && <span className="text-slate-400 italic">Nessuno segnalato</span>}
                  </div>
                </div>
                <div>
                  <p className="text-slate-400 font-semibold uppercase tracking-wider mb-1">Terapia</p>
                  <p className="text-slate-600">{selected.terapia_standardized ?? selected.terapia_raw ?? 'N/D'}</p>
                  {selected.terapia_warnings?.map((w, i) => (
                    <p key={i} className="text-amber-600 mt-0.5">⚠ {w}</p>
                  ))}
                </div>
              </div>
            </div>

            {/* 3-step grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <StepCard step={1} label="Dati Clinici" color="#1565C0"
                result={step1} loading={loading[1]}
                onRun={() => runStep(1)}
                disabled={!model}
              />
              <StepCard step={2} label="Biomarcatori Plasma" color="#6A1B9A"
                result={step2} loading={loading[2]}
                onRun={() => runStep(2)}
                disabled={!model || !step1?.feasible}
              />
              <StepCard step={3} label="Liquor (CSF)" color="#2E7D32"
                result={step3} loading={loading[3]}
                onRun={() => runStep(3)}
                disabled={!model || !step2?.feasible}
              />
            </div>

            {/* Raw data */}
            <details className="bg-white rounded-xl shadow-sm border border-border">
              <summary className="p-4 cursor-pointer text-sm font-semibold text-slate-600 hover:text-navy-700">
                Dati clinici grezzi (Anamnesi + EON)
              </summary>
              <div className="px-4 pb-4 space-y-3 text-xs text-slate-600">
                <div><p className="font-bold text-slate-400 mb-1">ANAMNESI</p><p className="leading-relaxed">{selected.anamnesi ?? 'N/D'}</p></div>
                <div><p className="font-bold text-slate-400 mb-1">ESAME OBIETTIVO NEUROLOGICO</p><p className="leading-relaxed">{selected.eon ?? 'N/D'}</p></div>
              </div>
            </details>
          </>
        )}
      </div>
    </div>
  )
}
