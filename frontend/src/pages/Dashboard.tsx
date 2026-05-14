import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Users, Brain, FlaskConical, Activity, AlertCircle } from 'lucide-react'
import { api } from '@/api/client'
import type { DBStats, RAGStatus, ModelInfo } from '@/types'

const DIAG_COLORS: Record<string, string> = {
  AD: '#1565C0', VAD: '#6A1B9A', MIXED: '#2E7D32',
  SCD: '#E65100', LATE: '#C62828', 'AD-PPA': '#00838F',
  FTD: '#558B2F', PD: '#4527A0',
}

function StatCard({ icon: Icon, label, value, sub, color }: {
  icon: React.ElementType; label: string; value: string | number; sub?: string; color?: string
}) {
  return (
    <div className="stat-card border-l-4" style={{ borderLeftColor: color ?? '#1565C0' }}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-1">{label}</p>
          <p className="text-3xl font-bold text-navy-700">{value}</p>
          {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
        </div>
        <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ backgroundColor: (color ?? '#1565C0') + '18' }}>
          <Icon className="w-5 h-5" style={{ color: color ?? '#1565C0' }} />
        </div>
      </div>
    </div>
  )
}

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-block w-2.5 h-2.5 rounded-full mr-2 ${ok ? 'bg-green-500' : 'bg-red-400'}`} />
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState<DBStats | null>(null)
  const [rag, setRag] = useState<RAGStatus | null>(null)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [ollamaOk, setOllamaOk] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([
      api.getDBStats().then(setStats),
      api.getRAGStatus().then(setRag),
      api.getOllamaStatus().then(s => setOllamaOk(s.running)),
      api.getModels().then(setModels).catch(() => {}),
    ]).finally(() => setLoading(false))
  }, [])

  const diagData = stats
    ? Object.entries(stats.diagnosis_distribution)
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) => ({ name: k, count: v, fill: DIAG_COLORS[k] ?? '#90A4AE' }))
    : []

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-navy-700" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-navy-700 to-navy-500 rounded-xl p-6 text-white shadow-lg">
        <h1 className="text-2xl font-bold mb-1">LLM Dementia — Diagnosis Support</h1>
        <p className="text-white/70 text-sm">Sistema di supporto alla diagnosi differenziale delle demenze tramite modelli linguistici locali</p>
      </div>

      {/* System status */}
      <div className="bg-white rounded-xl p-4 shadow-sm border border-border">
        <p className="section-title">Stato Sistema</p>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
          <div className="flex items-center">
            <StatusDot ok={ollamaOk} />
            <span className="text-slate-600">Ollama</span>
            <span className={`ml-2 text-xs font-semibold ${ollamaOk ? 'text-green-700' : 'text-red-600'}`}>
              {ollamaOk ? 'Attivo' : 'Non raggiungibile'}
            </span>
          </div>
          <div className="flex items-center">
            <StatusDot ok={rag?.ready ?? false} />
            <span className="text-slate-600">RAG Vector Store</span>
            <span className={`ml-2 text-xs font-semibold ${rag?.ready ? 'text-green-700' : 'text-amber-600'}`}>
              {rag?.ready ? `${rag.child_chunks} chunks` : 'Non inizializzato'}
            </span>
          </div>
          <div className="flex items-center">
            <StatusDot ok={models.length > 0} />
            <span className="text-slate-600">Modelli disponibili</span>
            <span className="ml-2 text-xs font-semibold text-slate-700">{models.length}</span>
          </div>
        </div>
        {(!rag?.ready || !ollamaOk) && (
          <div className="mt-3 flex items-start gap-2 text-amber-700 bg-amber-50 rounded-lg p-3 text-xs">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>
              {!ollamaOk && 'Avvia Ollama prima di procedere. '}
              {!rag?.ready && 'Vai su Impostazioni → Gestione RAG per inizializzare il vector store.'}
            </span>
          </div>
        )}
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={Users} label="Pazienti nel DB" value={stats?.total_patients ?? '—'} sub="Record totali" />
        <StatCard icon={Brain} label="Diagnosi AD" value={stats?.diagnosis_distribution['AD'] ?? '—'} sub="Alzheimer's Disease" color="#1565C0" />
        <StatCard icon={FlaskConical} label="Con Biomarcatori Plasma" color="#6A1B9A"
          value={stats ? Object.values(stats.missing_values).length > 0 ? '—' : '—' : '—'}
          sub="Disponibili per Step 2" />
        <StatCard icon={FlaskConical} label="Con Liquor (CSF)" value="—" sub="Disponibili per Step 3" color="#2E7D32" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Diagnosis distribution */}
        <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
          <p className="section-title">Distribuzione Diagnosi</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={diagData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.15)' }}
                formatter={(v: number) => [v, 'Pazienti']}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {diagData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Models installed */}
        <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
          <p className="section-title">Modelli Ollama Installati</p>
          {models.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-40 text-slate-400">
              <Brain className="w-10 h-10 mb-2 opacity-30" />
              <p className="text-sm">Nessun modello trovato</p>
              <p className="text-xs mt-1">Vai su Gestione Modelli per scaricare un modello</p>
            </div>
          ) : (
            <div className="space-y-2 mt-2">
              {models.map(m => (
                <div key={m.name} className="flex items-center justify-between py-2 px-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
                  <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4 text-navy-500" />
                    <span className="text-sm font-medium text-slate-700">{m.name}</span>
                  </div>
                  <span className="text-xs text-slate-400">
                    {m.size ? `${(m.size / 1e9).toFixed(1)} GB` : '—'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* RAG documents */}
      {rag?.ready && rag.documents.length > 0 && (
        <div className="bg-white rounded-xl p-5 shadow-sm border border-border">
          <p className="section-title">Documenti nella Knowledge Base ({rag.documents.length})</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
            {rag.documents.map((d, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-slate-600 py-1.5 px-3 bg-blue-50 rounded-lg">
                <span className="font-medium text-blue-700 truncate">{d.name}</span>
                <span className="text-slate-400 flex-shrink-0">{d.size_kb} KB</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
