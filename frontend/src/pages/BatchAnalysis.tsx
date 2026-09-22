import { useState, useEffect, useRef } from 'react'
import { Play, StopCircle, Loader2, CheckCircle, XCircle, BarChart3 } from 'lucide-react'
import { api } from '@/api/client'
import type { ModelInfo, BatchProgress, RagMode } from '@/types'

export default function BatchAnalysis() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [model, setModel] = useState('')
  const [ragMode, setRagMode] = useState<RagMode>('budson')
  const [progress, setProgress] = useState<BatchProgress>({ status: 'idle' })
  const [starting, setStarting] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    api.getModels().then(ms => {
      setModels(ms)
      if (ms.length > 0) setModel(ms[0].name)
    }).catch(() => {})
    api.getBatchProgress().then(setProgress).catch(() => {})
  }, [])

  useEffect(() => {
    if (progress.status === 'running') {
      pollRef.current = setInterval(async () => {
        const p = await api.getBatchProgress()
        setProgress(p)
        if (p.status !== 'running') {
          clearInterval(pollRef.current!)
        }
      }, 2000)
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [progress.status])

  async function startBatch() {
    if (!model) return
    setStarting(true)
    try {
      await api.runBatch({ model, rag_mode: ragMode })
      const p = await api.getBatchProgress()
      setProgress(p)
    } finally {
      setStarting(false)
    }
  }

  const pct = progress.total && progress.current
    ? Math.round((progress.current / progress.total) * 100)
    : 0

  return (
    <div className="max-w-3xl space-y-6">
      <div className="bg-gradient-to-r from-navy-700 to-navy-500 rounded-xl p-6 text-white shadow-lg">
        <h1 className="text-xl font-bold mb-1">Analisi Batch</h1>
        <p className="text-white/70 text-sm">Elabora tutti i pazienti del database in sequenza con i 3 step diagnostici</p>
      </div>

      {/* Config */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
        <p className="section-title">Configurazione</p>
        <div className="flex items-center gap-4 mt-3">
          <div className="flex-1">
            <label className="block text-xs font-semibold text-slate-500 mb-1.5">Modello Ollama</label>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              disabled={progress.status === 'running'}
              className="w-full border border-border rounded-lg px-3 py-2 text-sm bg-slate-50 focus:outline-none focus:ring-2 focus:ring-navy-500 disabled:opacity-50"
            >
              {models.length === 0 && <option value="">Nessun modello disponibile</option>}
              {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
          </div>
          <div className="flex-1">
            <label className="block text-xs font-semibold text-slate-500 mb-1.5">Corpus RAG</label>
            <select
              value={ragMode}
              onChange={e => setRagMode(e.target.value as RagMode)}
              disabled={progress.status === 'running'}
              className="w-full border border-border rounded-lg px-3 py-2 text-sm bg-slate-50 focus:outline-none focus:ring-2 focus:ring-navy-500 disabled:opacity-50"
            >
              <option value="budson">Budson & Solomon</option>
              <option value="casebook">Casebook of Dementia</option>
              <option value="both">Entrambi</option>
            </select>
          </div>
          <div className="flex-shrink-0 pt-5">
            <button
              onClick={startBatch}
              disabled={!model || progress.status === 'running' || starting}
              className="flex items-center gap-2 px-5 py-2.5 bg-navy-700 hover:bg-navy-900 text-white rounded-lg font-semibold text-sm transition-colors disabled:opacity-40"
            >
              {starting || progress.status === 'running'
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : <Play className="w-4 h-4" />}
              {progress.status === 'running' ? 'In esecuzione...' : 'Avvia Batch'}
            </button>
          </div>
        </div>
      </div>

      {/* Progress */}
      {progress.status !== 'idle' && (
        <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
          <div className="flex items-center justify-between mb-3">
            <p className="section-title mb-0">Progresso</p>
            <span className={`text-xs font-bold px-2 py-1 rounded-full ${
              progress.status === 'running' ? 'bg-blue-100 text-blue-700' :
              progress.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'
            }`}>
              {progress.status === 'running' ? 'In corso' :
               progress.status === 'completed' ? 'Completato' : 'In attesa'}
            </span>
          </div>

          {progress.total && (
            <>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>Paziente {progress.current ?? 0} / {progress.total}</span>
                <span>{pct}%</span>
              </div>
              <div className="h-2 bg-slate-100 rounded-full mb-3">
                <div
                  className="h-2 rounded-full bg-navy-600 transition-all duration-500"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </>
          )}

          {progress.current_patient && progress.status === 'running' && (
            <p className="text-xs text-slate-500">
              <span className="font-medium">Paziente corrente:</span> {progress.current_patient}
            </p>
          )}

          {progress.status === 'completed' && progress.result_file && (
            <div className="mt-3 flex items-start gap-2 bg-green-50 text-green-700 rounded-lg p-3 text-xs">
              <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-semibold">Analisi completata!</p>
                <p className="mt-0.5 text-green-600">File salvato: {progress.result_file.split('/').pop()}</p>
                <p className="mt-1">Vai su <span className="font-semibold">Risultati</span> per visualizzare le metriche.</p>
              </div>
            </div>
          )}

          {(progress.errors?.length ?? 0) > 0 && (
            <div className="mt-3">
              <p className="text-xs font-semibold text-red-600 mb-1">
                Errori ({progress.errors!.length}):
              </p>
              <div className="max-h-32 overflow-y-auto space-y-1">
                {progress.errors!.map((e, i) => (
                  <div key={i} className="flex items-start gap-1 text-xs text-red-600">
                    <XCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                    <span>{e}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-800">
        <div className="flex items-start gap-3">
          <BarChart3 className="w-5 h-5 mt-0.5 flex-shrink-0 text-blue-600" />
          <div>
            <p className="font-semibold mb-1">Come funziona il batch</p>
            <ul className="space-y-0.5 text-xs text-blue-700 list-disc list-inside">
              <li>Ogni paziente viene elaborato sequenzialmente con Step 1 → 2 → 3</li>
              <li>Step 2 e 3 vengono saltati automaticamente se mancano i dati necessari</li>
              <li>La diagnosi ground truth (Diagnosi_CODIFICATA) <strong>non è visibile al modello</strong></li>
              <li>I risultati vengono salvati nella cartella <code>results/</code> in formato JSON</li>
              <li>Le metriche di concordanza sono calcolate automaticamente al termine</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
