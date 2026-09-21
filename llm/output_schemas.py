"""JSON Schema vincolanti per gli output dei tre step diagnostici."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DiagnosisCode = Literal["AD", "AD-PPA", "MIXED", "VAD", "SCD", "LATE", "FTD", "PD"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DiagnosisAssessment(StrictModel):
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class DiagnosisAssessments(StrictModel):
    AD: DiagnosisAssessment
    AD_PPA: DiagnosisAssessment = Field(alias="AD-PPA")
    MIXED: DiagnosisAssessment
    VAD: DiagnosisAssessment
    SCD: DiagnosisAssessment
    LATE: DiagnosisAssessment
    FTD: DiagnosisAssessment
    PD: DiagnosisAssessment


class RagEvidence(StrictModel):
    source_index: int = Field(ge=1)
    quote: str
    relevance: str


class Step1Output(StrictModel):
    step: Literal[1]
    patient_code: str
    key_clinical_features: list[str]
    diagnostic_reasoning: str
    diagnosis_assessments: DiagnosisAssessments
    primary_diagnosis: DiagnosisCode
    clinical_summary: str
    missing_information: list[str]
    rag_sources_used: list[int]
    rag_evidence: list[RagEvidence]
    step1_limitations: str


class PlasmaReasoning(StrictModel):
    Plasma_Ab4240: str
    plasma_ptau217: str
    plasma_pt181: str
    plasma_NfL: str
    epato_renal_reliability: str


class Step2Output(StrictModel):
    step: Literal[2]
    patient_code: str
    plasma_biomarker_reasoning: PlasmaReasoning
    diagnostic_reasoning: str
    diagnosis_assessments: DiagnosisAssessments
    primary_diagnosis: DiagnosisCode
    update_from_step1: str
    clinical_summary: str


class CsfReasoning(StrictModel):
    CSF_Ab42: str
    CSF_Ab40: str
    CSF_Ab4240: str
    CSF_ttau: str
    CSF_ptau: str
    CSF_NfL: str


class Step3Output(StrictModel):
    step: Literal[3]
    patient_code: str
    csf_biomarker_reasoning: CsfReasoning
    atn_interpretation: str
    plasma_csf_interpretation: str
    diagnostic_reasoning: str
    diagnosis_assessments: DiagnosisAssessments
    primary_diagnosis: DiagnosisCode
    final_clinical_summary: str
    update_from_step2: str


_STEP_MODELS: dict[int, type[BaseModel]] = {
    1: Step1Output,
    2: Step2Output,
    3: Step3Output,
}


def schema_for_step(step: int) -> dict:
    try:
        return _STEP_MODELS[step].model_json_schema(by_alias=True)
    except KeyError as exc:
        raise ValueError(f"Step non valido: {step}") from exc
