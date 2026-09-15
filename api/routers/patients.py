from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import PatientSummary, PatientDetail, DBStats
from data.loader import (
    CSF_BIOMARKER_COLS,
    PLASMA_BIOMARKER_COLS,
    SAFETY_LAB_COLS,
    get_database as _get_db,
    get_patient_record,
    summary_stats,
)
from data.preprocessor import standardize_therapy

router = APIRouter(prefix="/patients", tags=["patients"])


def _has_any(row, columns: list[str]) -> bool:
    return any(pd.notna(row.get(c)) for c in columns)


@router.get("/", response_model=list[PatientSummary])
def list_patients():
    df = _get_db()
    result = []
    for _, row in df.iterrows():
        result.append(PatientSummary(
            codice=str(row.get("Codice", "")),
            age=int(row["Age"]) if pd.notna(row.get("Age")) else None,
            gender=str(row.get("Gender")) if pd.notna(row.get("Gender")) else None,
            diagnosis_gt=str(row.get("Diagnosi_CODIFICATA", "")),
            has_plasma=_has_any(row, PLASMA_BIOMARKER_COLS),
            has_csf=_has_any(row, CSF_BIOMARKER_COLS),
            mmse=float(row["MMSE"]) if pd.notna(row.get("MMSE")) else None,
        ))
    return result


@router.get("/stats", response_model=DBStats)
def db_stats():
    df = _get_db()
    s = summary_stats(df)
    return DBStats(
        total_patients=s["total_patients"],
        diagnosis_distribution=s["diagnosis_distribution"],
        missing_values=s["missing_values"],
    )


@router.get("/{code}", response_model=PatientDetail)
def get_patient(code: str):
    df = _get_db()
    record = get_patient_record(df, code)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Paziente '{code}' non trovato")

    terapia_raw = str(record.get("TERAPIA", "")) if pd.notna(record.get("TERAPIA")) else None
    parsed = standardize_therapy(terapia_raw or "")

    def _float(v):
        return float(v) if pd.notna(v) else None

    def _bool(v):
        if v is True:
            return True
        if v is False:
            return False
        return None

    return PatientDetail(
        codice=str(record.get("Codice", "")),
        age=int(record["Age"]) if pd.notna(record.get("Age")) else None,
        gender=str(record.get("Gender")) if pd.notna(record.get("Gender")) else None,
        diagnosis_gt=str(record.get("Diagnosi_CODIFICATA", "")),
        has_plasma=_has_any(record, PLASMA_BIOMARKER_COLS),
        has_csf=_has_any(record, CSF_BIOMARKER_COLS),
        mmse=_float(record.get("MMSE")),
        anamnesi=str(record.get("ANAMNESI")) if pd.notna(record.get("ANAMNESI")) else None,
        eon=str(record.get("EON")) if pd.notna(record.get("EON")) else None,
        terapia_raw=terapia_raw,
        terapia_standardized=parsed.get("summary"),
        terapia_warnings=parsed.get("warnings", []),
        anni_edu=int(record["Anni_edu"]) if pd.notna(record.get("Anni_edu")) else None,
        fam=_bool(record.get("Fam")),
        fumo=_bool(record.get("Fumo (attivo o pregresso)")),
        ipertensione=_bool(record.get("Ipertensione")),
        cardiovascolare=_bool(record.get("Malattia_cardiovascolare")),
        diabete=_bool(record.get("Diabete")),
        dislipidemia=_bool(record.get("Dislipidemia")),
        plasma={c: _float(record.get(c)) for c in PLASMA_BIOMARKER_COLS},
        csf={c: _float(record.get(c)) for c in CSF_BIOMARKER_COLS},
        safety_labs={c: _float(record.get(c)) for c in SAFETY_LAB_COLS},
    )
