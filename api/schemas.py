from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class PatientSummary(BaseModel):
    codice: str
    age: Optional[int]
    gender: Optional[str]
    diagnosis_gt: str
    has_plasma: bool
    has_csf: bool
    mmse: Optional[float]


class PatientDetail(PatientSummary):
    anamnesi: Optional[str]
    eon: Optional[str]
    terapia_raw: Optional[str]
    terapia_standardized: Optional[str]
    terapia_warnings: list[str]
    anni_edu: Optional[int]
    fam: Optional[bool]
    fumo: Optional[bool]
    ipertensione: Optional[bool]
    cardiovascolare: Optional[bool]
    diabete: Optional[bool]
    dislipidemia: Optional[bool]
    plasma: dict[str, Optional[float]]
    csf: dict[str, Optional[float]]
    safety_labs: dict[str, Optional[float]]


class RunStepRequest(BaseModel):
    patient_code: str
    step: int
    model: str
    rag_mode: Literal["budson", "casebook", "both"] = "budson"
    seed: Optional[int] = None
    step1_result: Optional[dict] = None
    step2_result: Optional[dict] = None


class RunPipelineRequest(BaseModel):
    patient_code: str
    model: str
    rag_mode: Literal["budson", "casebook", "both"] = "budson"
    seed: Optional[int] = None


class BatchRunRequest(BaseModel):
    model: str
    rag_mode: Literal["budson", "casebook", "both"] = "budson"
    seed: Optional[int] = None
    patient_codes: Optional[list[str]] = None


class SaveIndividualRequest(BaseModel):
    patient_code: str
    model: str
    rag_mode: Literal["budson", "casebook", "both"] = "budson"
    step1: Optional[dict] = None
    step2: Optional[dict] = None
    step3: Optional[dict] = None


class RagSource(BaseModel):
    index: int
    source: str
    source_path: str = ""
    section: str = ""
    page: Optional[int] = None
    page_end: Optional[int] = None
    page_label: str = ""
    page_label_end: str = ""
    page_ref: str = ""
    doc_id: str = ""
    chunk_id: str = ""
    retrieval_score: Optional[float] = None
    # Estratti esatti che hanno prodotto il match nel retrieval.
    matched_quotes: list[str] = Field(default_factory=list)
    # Passaggio integrale citato: serve a verificare la fonte, non è troncato.
    text: str = ""


class StepResult(BaseModel):
    step: int
    patient_code: str
    eligible: bool = True
    execution_status: str = "completed"
    feasible: bool
    skip_reason: Optional[str]
    result: Optional[dict]
    model_result: Optional[dict] = None
    raw_response: Optional[str]
    parse_metadata: Optional[dict] = None
    generation_metadata: Optional[dict] = None
    generation_attempts: list[dict] = Field(default_factory=list)
    model_input: Optional[dict] = None
    duration_s: float
    model_used: str
    seed: Optional[int] = None
    computed_biomarkers: Optional[dict] = None
    rag_sources: list[RagSource] = Field(default_factory=list)
    prompt_chars: Optional[int] = None


class PipelineResult(BaseModel):
    patient_code: str
    ground_truth: str
    step1: StepResult
    step2: StepResult
    step3: StepResult
    model: Optional[str] = None
    rag_mode: Literal["budson", "casebook", "both"] = "budson"
    base_seed: Optional[int] = None
    step_seeds: dict[int, Optional[int]] = Field(default_factory=dict)
    rag_bundle_sha256: Optional[str] = None
    input_availability: dict = Field(default_factory=dict)
    total_duration_s: float


class ModelInfo(BaseModel):
    name: str
    size: Optional[int]
    modified_at: Optional[str]
    details: Optional[dict]


class PullModelRequest(BaseModel):
    model_name: str


class RAGStatus(BaseModel):
    ready: bool
    parent_chunks: int
    child_chunks: int
    embedding_model: str = ""
    corpora: dict[str, list[str]] = Field(default_factory=dict)
    documents: list[dict]


class EvaluationSummary(BaseModel):
    total_patients: int
    step1: dict
    step2: dict
    step3: dict


class ResultFile(BaseModel):
    filename: str
    model: str
    timestamp: str
    total_patients: int
    step1_accuracy: Optional[float]
    step2_accuracy: Optional[float]
    step3_accuracy: Optional[float]


class DBStats(BaseModel):
    total_patients: int
    diagnosis_distribution: dict[str, int]
    missing_values: dict[str, int]
