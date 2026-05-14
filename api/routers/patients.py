from __future__ import annotations
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import PatientSummary, PatientDetail, DBStats
from config.settings import DATABASE_PATH
from data.loader import load_database, get_patient_record, summary_stats
from data.preprocessor import standardize_therapy

router = APIRouter(prefix="/patients", tags=["patients"])


@lru_cache(maxsize=1)
def _get_db() -> pd.DataFrame:
    return load_database(DATABASE_PATH)


def _reload_db() -> pd.DataFrame:
    _get_db.cache_clear()
    return _get_db()


@router.get("/", response_model=list[PatientSummary])
def list_patients():
    df = _get_db()
    result = []
    for _, row in df.iterrows():
        plasma_cols = ["Plasma_Ab4240", "plasma_ptau217", "plasma_pt181", "plasma_NfL"]
        csf_cols = ["CSF_Ab42", "CSF_Ab40", "CSF_Ab4240", "CSF_ttau", "CSF_ptau"]
        has_plasma = any(pd.notna(row.get(c)) for c in plasma_cols)
        has_csf = any(pd.notna(row.get(c)) for c in csf_cols)
        result.append(PatientSummary(
            codice=str(row.get("Codice", "")),
            age=int(row["Age"]) if pd.notna(row.get("Age")) else None,
            gender=str(row.get("Gender")) if pd.notna(row.get("Gender")) else None,
            diagnosis_gt=str(row.get("Diagnosi_CODIFICATA", "")),
            has_plasma=has_plasma,
            has_csf=has_csf,
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
        has_plasma=any(_float(record.get(c)) is not None for c in
                       ["Plasma_Ab4240", "plasma_ptau217", "plasma_pt181", "plasma_NfL"]),
        has_csf=any(_float(record.get(c)) is not None for c in
                    ["CSF_Ab42", "CSF_Ab40", "CSF_Ab4240", "CSF_ttau", "CSF_ptau"]),
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
        plasma={
            "Plasma_Ab4240": _float(record.get("Plasma_Ab4240")),
            "plasma_ptau217": _float(record.get("plasma_ptau217")),
            "plasma_pt181": _float(record.get("plasma_pt181")),
            "plasma_NfL": _float(record.get("plasma_NfL")),
        },
        csf={
            "CSF_Ab42": _float(record.get("CSF_Ab42")),
            "CSF_Ab40": _float(record.get("CSF_Ab40")),
            "CSF_Ab4240": _float(record.get("CSF_Ab4240")),
            "CSF_ttau": _float(record.get("CSF_ttau")),
            "CSF_ptau": _float(record.get("CSF_ptau")),
            "CSF_NfL": _float(record.get("CSF_NfL")),
        },
        safety_labs={
            "Creatinina": _float(record.get("Creatinina")),
            "AST": _float(record.get("AST")),
            "ALT": _float(record.get("ALT")),
            "eGFR_2021": _float(record.get("eGFR_2021")),
        },
    )
