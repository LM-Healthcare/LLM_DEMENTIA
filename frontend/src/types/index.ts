export interface PatientSummary {
  codice: string
  age: number | null
  gender: string | null
  diagnosis_gt: string
  has_plasma: boolean
  has_csf: boolean
  mmse: number | null
}

export interface PatientDetail extends PatientSummary {
  anamnesi: string | null
  eon: string | null
  terapia_raw: string | null
  terapia_standardized: string | null
  terapia_warnings: string[]
  anni_edu: number | null
  fam: boolean | null
  fumo: boolean | null
  ipertensione: boolean | null
  cardiovascolare: boolean | null
  diabete: boolean | null
  dislipidemia: boolean | null
  plasma: Record<string, number | null>
  csf: Record<string, number | null>
  safety_labs: Record<string, number | null>
}

export interface RagSource {
  index: number
  source: string
  source_path: string
  section: string
  page: number | null
  page_end: number | null
  page_label: string
  page_label_end: string
  page_ref: string
  doc_id: string
  chunk_id: string
  retrieval_score: number | null
  /** Estratti esatti che hanno prodotto il match nel retrieval. */
  matched_quotes: string[]
  /** Passaggio integrale citato, non troncato. */
  text: string
}

export interface RagEvidence {
  source_index: number
  quote: string
  relevance?: string
}

export interface StepResult {
  step: number
  patient_code: string
  feasible: boolean
  skip_reason: string | null
  result: StepOutput | null
  raw_response: string | null
  duration_s: number
  model_used: string
  rag_sources: RagSource[]
  prompt_chars?: number | null
}

export interface DiagnosisEntry {
  diagnosis: string
  label: string
  probability: 'ALTA' | 'MEDIA' | 'BASSA' | 'ESCLUSA'
  confidence_score?: number
  reasoning: string
}

export interface StepOutput {
  step: number
  patient_code: string
  primary_diagnosis: DiagnosisEntry
  differential_diagnoses: DiagnosisEntry[]
  clinical_summary?: string
  final_clinical_summary?: string
  key_clinical_features?: string[]
  missing_information?: string[]
  rag_sources_used?: number[]
  rag_evidence?: RagEvidence[]
  step1_limitations?: string
  plasma_csf_concordance?: string
  biomarker_reliability?: {
    renal_function_ok: boolean
    hepatic_function_ok: boolean
    biomarkers_reliable: boolean
    reliability_notes: string
  }
  plasma_biomarker_interpretation?: Record<string, BiomarkerEntry>
  csf_biomarker_interpretation?: Record<string, BiomarkerEntry>
  at_profile?: string
  update_from_step1?: string
  update_from_step2?: string
  error?: string
}

export interface BiomarkerEntry {
  value: number | null
  status: 'NORMALE' | 'PATOLOGICO' | 'MANCANTE'
  interpretation: string
}

export interface PipelineResult {
  patient_code: string
  ground_truth: string
  step1: StepResult
  step2: StepResult
  step3: StepResult
  total_duration_s: number
}

export interface ModelInfo {
  name: string
  size: number | null
  modified_at: string | null
  details: Record<string, unknown> | null
}

export interface RAGStatus {
  ready: boolean
  parent_chunks: number
  child_chunks: number
  embedding_model: string
  documents: Array<{ name: string; folder: string; size_kb: number }>
}

export interface DBStats {
  total_patients: number
  diagnosis_distribution: Record<string, number>
  missing_values: Record<string, number>
}

export interface ResultFile {
  filename: string
  model: string
  timestamp: string
  total_patients: number
  step1_accuracy: number | null
  step2_accuracy: number | null
  step3_accuracy: number | null
}

export interface EvaluationData {
  records: RecordEvaluation[]
  metrics_per_step: Record<string, StepMetrics>
  total_patients: number
  summary: EvaluationSummary
}

export interface RecordEvaluation {
  patient_code: string
  ground_truth: string
  step1_prediction: string | null
  step2_prediction: string | null
  step3_prediction: string | null
  step1_concordant: boolean | null
  step2_concordant: boolean | null
  step3_concordant: boolean | null
  step1_confidence: number | null
  step2_confidence: number | null
  step3_confidence: number | null
  step1_feasible: boolean
  step2_feasible: boolean
  step3_feasible: boolean
}

export interface StepMetrics {
  n_evaluated: number
  accuracy: number
  cohen_kappa: number
  per_class: Record<string, ClassMetrics>
  confusion_matrix: Record<string, Record<string, number>>
  classes: string[]
}

export interface ClassMetrics {
  tp: number; fp: number; fn: number
  precision: number; recall: number; f1: number; support: number
}

export interface EvaluationSummary {
  total_patients: number
  step1: StepSummary
  step2: StepSummary
  step3: StepSummary
}

export interface StepSummary {
  feasible_patients: number
  concordant_patients: number
  accuracy: number
  cohen_kappa: number
}

export interface BatchProgress {
  status: 'idle' | 'running' | 'completed'
  current?: number
  total?: number
  current_patient?: string
  errors?: string[]
  result_file?: string
}
