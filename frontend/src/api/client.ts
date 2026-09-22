import type {
  PatientSummary, PatientDetail, DBStats,
  StepResult, PipelineResult, BatchProgress,
  ModelInfo, RagMode, RAGStatus, ResultFile, EvaluationData,
} from '@/types'

const BASE = '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  // ── Patients ───────────────────────────────────────────────────────────
  getPatients: () => request<PatientSummary[]>('/patients/'),
  getPatient: (code: string) => request<PatientDetail>(`/patients/${code}`),
  getDBStats: () => request<DBStats>('/patients/stats'),

  // ── Analysis ───────────────────────────────────────────────────────────
  runStep: (payload: {
    patient_code: string
    step: number
    model: string
    seed?: number | null
    rag_mode?: RagMode
    step1_result?: object | null
    step2_result?: object | null
  }) => request<StepResult>('/analysis/step', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),

  runPipeline: (payload: { patient_code: string; model: string; seed?: number | null; rag_mode?: RagMode }) =>
    request<PipelineResult>('/analysis/pipeline', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  runBatch: (payload: { model: string; rag_mode?: RagMode; seed?: number | null; patient_codes?: string[] }) =>
    request<{ message: string; total: number }>('/analysis/batch', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getBatchProgress: () => request<BatchProgress>('/analysis/batch/progress'),

  saveIndividual: (payload: {
    patient_code: string
    model: string
    rag_mode?: RagMode
    step1?: object | null
    step2?: object | null
    step3?: object | null
  }) => request<{ message: string; filename: string }>('/analysis/save-individual', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),

  // ── Models ─────────────────────────────────────────────────────────────
  getOllamaStatus: () => request<{ running: boolean }>('/models/status'),
  getModels: () => request<ModelInfo[]>('/models/'),

  pullModelStream: (modelName: string): EventSource => {
    return new EventSource(`/api/models/pull-sse?model=${encodeURIComponent(modelName)}`)
  },

  pullModel: async (modelName: string, onStatus: (s: string) => void): Promise<void> => {
    const res = await fetch(`${BASE}/models/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const reader = res.body?.getReader()
    if (!reader) return
    const decoder = new TextDecoder()
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const text = decoder.decode(value)
      const lines = text.split('\n').filter(l => l.startsWith('data: '))
      for (const line of lines) {
        const status = line.replace('data: ', '').trim()
        onStatus(status)
        if (status === 'done') return
      }
    }
  },

  // ── RAG ────────────────────────────────────────────────────────────────
  getRAGStatus: () => request<RAGStatus>('/rag/status'),
  buildRAG: (force = false) =>
    request<{ message: string; ready: boolean }>(`/rag/build?force=${force}`, { method: 'POST' }),
  loadRAG: () => request<{ message: string }>('/rag/load', { method: 'POST' }),

  // ── Results ────────────────────────────────────────────────────────────
  getResults: () => request<ResultFile[]>('/results/'),
  getEvaluation: (filename: string) => request<EvaluationData>(`/results/${filename}/evaluation`),
  getFullResult: (filename: string) => request<{ evaluation: EvaluationData; pipeline_results: PipelineResult[] }>(`/results/${filename}`),
}
