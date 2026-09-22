import { useState, useEffect } from 'react'
import { Brain, Download, RefreshCw, CheckCircle, AlertCircle, Loader2, HardDrive } from 'lucide-react'
import { api } from '@/api/client'
import type { ModelInfo } from '@/types'

const SUGGESTED_MODELS = [
  { name: 'qwen3.5:9b-bf16', desc: 'Qwen 3.5 9B BF16 — studio, server 24 GB',           size: '19 GB' },
  { name: 'ministral-3:8b-instruct-2512-fp16', desc: 'Ministral 3 8B FP16 — studio, server 24 GB', size: '18 GB' },
  { name: 'llama3.1:8b-instruct-fp16', desc: 'Llama 3.1 8B FP16 — studio, server 24 GB', size: '16 GB' },
  { name: 'bge-m3',          desc: 'BGE-M3 — embedding multilingue RAG obbligatorio',  size: '1.2 GB' },
]

export default function ModelManagement() {
  const [installed, setInstalled] = useState<ModelInfo[]>([])
  const [ollamaOk, setOllamaOk] = useState(false)
  const [loading, setLoading] = useState(true)
  const [pullTarget, setPullTarget] = useState('')
  const [pulling, setPulling] = useState<string | null>(null)
  const [pullStatus, setPullStatus] = useState('')
  const [error, setError] = useState('')

  async function refresh() {
    setLoading(true)
    try {
      const ok = await api.getOllamaStatus()
      setOllamaOk(ok.running)
      if (ok.running) {
        const ms = await api.getModels()
        setInstalled(ms)
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { refresh() }, [])

  async function pullModel(name: string) {
    setError('')
    setPulling(name)
    setPullStatus('Avvio download...')
    try {
      await api.pullModel(name, status => {
        if (status !== 'done') setPullStatus(status)
      })
      setPullStatus('✓ Download completato!')
      await refresh()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Errore download')
    } finally {
      setPulling(null)
    }
  }

  const installedNames = new Set(installed.map(m => m.name))

  return (
    <div className="max-w-3xl space-y-6">
      <div className="bg-gradient-to-r from-navy-700 to-navy-500 rounded-xl p-6 text-white shadow-lg">
        <h1 className="text-xl font-bold mb-1">Gestione Modelli</h1>
        <p className="text-white/70 text-sm">Scarica e gestisci i modelli Ollama per l'analisi diagnostica</p>
      </div>

      {/* Ollama status */}
      <div className={`rounded-xl p-4 flex items-center gap-3 ${ollamaOk ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
        {ollamaOk
          ? <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
          : <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />}
        <div className="flex-1">
          <p className={`text-sm font-semibold ${ollamaOk ? 'text-green-800' : 'text-red-800'}`}>
            {ollamaOk ? 'Ollama attivo su localhost:11434' : 'Ollama non raggiungibile'}
          </p>
          {!ollamaOk && (
            <p className="text-xs text-red-600 mt-0.5">
              Avvia Ollama con: <code className="bg-red-100 px-1 rounded">ollama serve</code>
            </p>
          )}
        </div>
        <button
          onClick={refresh}
          className="p-2 rounded-lg hover:bg-white/50 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''} ${ollamaOk ? 'text-green-600' : 'text-red-600'}`} />
        </button>
      </div>

      {/* Installed models */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
        <p className="section-title">Modelli Installati ({installed.length})</p>
        {installed.length === 0 ? (
          <div className="flex flex-col items-center py-6 text-slate-400">
            <Brain className="w-10 h-10 mb-2 opacity-20" />
            <p className="text-sm">Nessun modello installato</p>
          </div>
        ) : (
          <div className="space-y-2 mt-2">
            {installed.map(m => (
              <div key={m.name} className="flex items-center justify-between py-2.5 px-3 rounded-lg bg-slate-50 border border-slate-200">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-navy-500" />
                  <span className="text-sm font-semibold text-slate-700">{m.name}</span>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-400">
                  {m.size && (
                    <span className="flex items-center gap-1">
                      <HardDrive className="w-3 h-3" />
                      {(m.size / 1e9).toFixed(1)} GB
                    </span>
                  )}
                  <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded-full font-semibold">Installato</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Pull model */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
        <p className="section-title">Scarica Modello</p>
        <div className="flex gap-2 mt-3">
          <input
            className="flex-1 border border-border rounded-lg px-3 py-2 text-sm bg-slate-50 focus:outline-none focus:ring-2 focus:ring-navy-500"
            placeholder="es. llama3.1:8b"
            value={pullTarget}
            onChange={e => setPullTarget(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && pullTarget && pullModel(pullTarget)}
          />
          <button
            onClick={() => pullTarget && pullModel(pullTarget)}
            disabled={!pullTarget || !!pulling || !ollamaOk}
            className="flex items-center gap-2 px-4 py-2 bg-navy-700 hover:bg-navy-900 text-white rounded-lg font-semibold text-sm transition-colors disabled:opacity-40"
          >
            {pulling === pullTarget ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            Pull
          </button>
        </div>
        {pulling && (
          <p className="mt-2 text-xs text-slate-500 flex items-center gap-2">
            <Loader2 className="w-3 h-3 animate-spin" /> {pullStatus}
          </p>
        )}
        {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
      </div>

      {/* Suggested models */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
        <p className="section-title">Modelli Consigliati</p>
        <div className="space-y-2 mt-2">
          {SUGGESTED_MODELS.map(m => (
            <div key={m.name} className="flex items-center justify-between py-2.5 px-3 rounded-lg border border-slate-200 hover:border-navy-300 transition-colors">
              <div>
                <p className="text-sm font-semibold text-slate-700">{m.name}</p>
                <p className="text-xs text-slate-400">{m.desc} · {m.size}</p>
              </div>
              {installedNames.has(m.name) ? (
                <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full font-semibold">✓ Installato</span>
              ) : (
                <button
                  onClick={() => pullModel(m.name)}
                  disabled={!!pulling || !ollamaOk}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-navy-700 hover:bg-navy-900 text-white rounded-lg text-xs font-semibold disabled:opacity-40 transition-colors"
                >
                  {pulling === m.name ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
                  Installa
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
