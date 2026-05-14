import { useState, useEffect } from 'react'
import { Save, Database, BookOpen, RefreshCw, CheckCircle, Loader2, AlertTriangle } from 'lucide-react'
import { api } from '@/api/client'
import type { RAGStatus } from '@/types'

export default function Settings() {
  const [rag, setRag] = useState<RAGStatus | null>(null)
  const [ragBuilding, setRagBuilding] = useState(false)
  const [ragMsg, setRagMsg] = useState('')
  const [ragError, setRagError] = useState('')

  useEffect(() => { api.getRAGStatus().then(setRag) }, [])

  async function buildRAG(force = false) {
    setRagBuilding(true)
    setRagMsg('')
    setRagError('')
    try {
      const res = await api.buildRAG(force)
      setRagMsg(res.message)
      setTimeout(async () => {
        const status = await api.getRAGStatus()
        setRag(status)
      }, 5000)
    } catch (e: unknown) {
      setRagError(e instanceof Error ? e.message : 'Errore')
    } finally {
      setRagBuilding(false)
    }
  }

  async function loadRAG() {
    setRagMsg('')
    setRagError('')
    try {
      const res = await api.loadRAG()
      setRagMsg(res.message)
      const status = await api.getRAGStatus()
      setRag(status)
    } catch (e: unknown) {
      setRagError(e instanceof Error ? e.message : 'Errore')
    }
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div className="bg-gradient-to-r from-navy-700 to-navy-500 rounded-xl p-6 text-white shadow-lg">
        <h1 className="text-xl font-bold mb-1">Impostazioni</h1>
        <p className="text-white/70 text-sm">Configurazione del sistema e gestione RAG</p>
      </div>

      {/* RAG Management */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="w-5 h-5 text-navy-600" />
          <h2 className="font-semibold text-slate-700">Gestione Knowledge Base (RAG)</h2>
        </div>

        <div className={`rounded-xl p-4 flex items-center gap-3 mb-4 ${rag?.ready ? 'bg-green-50 border border-green-200' : 'bg-amber-50 border border-amber-200'}`}>
          {rag?.ready
            ? <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
            : <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />}
          <div>
            <p className={`text-sm font-semibold ${rag?.ready ? 'text-green-800' : 'text-amber-800'}`}>
              {rag?.ready ? 'Vector store inizializzato' : 'Vector store non inizializzato'}
            </p>
            {rag?.ready && (
              <p className="text-xs text-green-600 mt-0.5">
                {rag.parent_chunks} parent chunks · {rag.child_chunks} child chunks · {rag.documents.length} documenti
              </p>
            )}
            {!rag?.ready && (
              <p className="text-xs text-amber-600 mt-0.5">
                Clicca "Inizializza RAG" per caricare i documenti nel vector store
              </p>
            )}
          </div>
        </div>

        {rag?.ready && rag.documents.length > 0 && (
          <div className="mb-4">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Documenti indicizzati</p>
            <div className="grid grid-cols-1 gap-1.5 max-h-48 overflow-y-auto">
              {rag.documents.map((d, i) => (
                <div key={i} className="flex items-center justify-between text-xs py-1.5 px-3 bg-slate-50 rounded-lg">
                  <span className="text-slate-600 truncate">{d.name}</span>
                  <span className="text-slate-400 flex-shrink-0 ml-2">{d.folder} · {d.size_kb} KB</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {ragMsg && (
          <div className="mb-3 flex items-start gap-2 text-green-700 bg-green-50 rounded-lg p-3 text-xs">
            <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{ragMsg}</span>
          </div>
        )}
        {ragError && (
          <div className="mb-3 flex items-start gap-2 text-red-700 bg-red-50 rounded-lg p-3 text-xs">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{ragError}</span>
          </div>
        )}

        <div className="flex gap-2">
          {!rag?.ready && (
            <button
              onClick={() => loadRAG()}
              className="flex items-center gap-2 px-4 py-2 border border-navy-300 text-navy-700 hover:bg-navy-50 rounded-lg text-sm font-semibold transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Carica esistente
            </button>
          )}
          <button
            onClick={() => buildRAG(false)}
            disabled={ragBuilding}
            className="flex items-center gap-2 px-4 py-2 bg-navy-700 hover:bg-navy-900 text-white rounded-lg text-sm font-semibold transition-colors disabled:opacity-40"
          >
            {ragBuilding ? <Loader2 className="w-4 h-4 animate-spin" /> : <BookOpen className="w-4 h-4" />}
            {rag?.ready ? 'Aggiorna RAG' : 'Inizializza RAG'}
          </button>
          {rag?.ready && (
            <button
              onClick={() => buildRAG(true)}
              disabled={ragBuilding}
              className="flex items-center gap-2 px-4 py-2 border border-red-300 text-red-600 hover:bg-red-50 rounded-lg text-sm font-semibold transition-colors disabled:opacity-40"
            >
              <RefreshCw className="w-4 h-4" />
              Ricostruisci
            </button>
          )}
        </div>

        <p className="text-xs text-slate-400 mt-3">
          ⚠ L'inizializzazione richiede il modello <code className="bg-slate-100 px-1 rounded">nomic-embed-text</code> installato su Ollama.
          Il processo viene eseguito in background e può richiedere qualche minuto.
        </p>
      </div>

      {/* Info */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-5 h-5 text-navy-600" />
          <h2 className="font-semibold text-slate-700">Configurazione (da .env)</h2>
        </div>
        <div className="space-y-2 text-xs font-mono">
          {[
            ['OLLAMA_BASE_URL', 'http://localhost:11434'],
            ['OLLAMA_MODEL', 'configurato nel .env'],
            ['OLLAMA_EMBED_MODEL', 'nomic-embed-text'],
            ['CHROMA_DB_PATH', './chroma_db'],
            ['RAG_TOP_K', '6'],
            ['RAG_CHUNK_SIZE', '800 token'],
          ].map(([k, v]) => (
            <div key={k} className="flex gap-3 py-1.5 px-3 bg-slate-50 rounded-lg">
              <span className="text-navy-600 font-semibold w-40 flex-shrink-0">{k}</span>
              <span className="text-slate-500">{v}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-400 mt-3">
          Modifica il file <code className="bg-slate-100 px-1 rounded">.env</code> nella root del progetto e riavvia il server.
        </p>
      </div>
    </div>
  )
}
