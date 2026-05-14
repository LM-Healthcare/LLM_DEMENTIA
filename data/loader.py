"""
Database loader: carica il dataset Excel, normalizza le diagnosi,
gestisce i valori mancanti e produce record strutturati per la pipeline.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from config.settings import DATABASE_PATH, AD_PPA_VARIANTS, DIAGNOSIS_LABELS


STEP1_CLINICAL_COLS = [
    "ANAMNESI", "TERAPIA", "EON", "MMSE",
    "Fam", "Fumo (attivo o pregresso)",
    "Ipertensione", "Malattia_cardiovascolare",
    "Diabete", "Dislipidemia",
    "Age", "Gender", "Anni_edu",
]

STEP2_PLASMA_COLS = [
    "Plasma_Ab4240", "plasma_ptau217", "plasma_pt181", "plasma_NfL",
    "Creatinina", "AST", "ALT", "eGFR_2021",
]

STEP3_CSF_COLS = [
    "CSF_Ab42", "CSF_Ab40", "CSF_Ab4240",
    "CSF_ttau", "CSF_ptau", "CSF_NfL",
]

HIDDEN_COLS = ["Diagnosi_CODIFICATA", "Diagnosi_TESTUALE", "PAZIENTE"]


def load_database(path: str = DATABASE_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Foglio 1")
    df = _normalize_diagnoses(df)
    df = _clean_columns(df)
    return df


def _normalize_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    def normalize(val: str) -> str:
        if not isinstance(val, str):
            return str(val)
        v = val.strip()
        if v in AD_PPA_VARIANTS:
            return "AD-PPA"
        return v.upper() if v.upper() in DIAGNOSIS_LABELS else v

    df["Diagnosi_CODIFICATA"] = df["Diagnosi_CODIFICATA"].apply(normalize)
    return df


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = [
        "Fam", "Fumo (attivo o pregresso)",
        "Ipertensione", "Malattia_cardiovascolare",
        "Diabete", "Dislipidemia",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: True if str(x).strip().upper() in ("SI", "SÌ", "1", "TRUE", "YES")
                else (False if str(x).strip().upper() in ("NO", "0", "FALSE") else None)
            )
    return df


def get_patient_record(df: pd.DataFrame, codice: str) -> Optional[dict]:
    row = df[df["Codice"] == codice]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def build_step_payload(
    record: dict,
    step: int,
    terapia_standardized: Optional[str] = None,
) -> dict:
    """
    Costruisce il payload di un paziente per lo step specificato (1, 2, 3).
    La diagnosi ground-truth è sempre esclusa dal payload.
    """
    payload: dict = {
        "codice": record.get("Codice"),
        "step": step,
    }

    for col in STEP1_CLINICAL_COLS:
        val = record.get(col)
        key = col.lower().replace(" ", "_").replace("(", "").replace(")", "")
        payload[key] = val if pd.notna(val) else None

    if terapia_standardized is not None:
        payload["terapia_standardized"] = terapia_standardized

    if step >= 2:
        for col in STEP2_PLASMA_COLS:
            val = record.get(col)
            payload[col] = float(val) if pd.notna(val) else None

    if step >= 3:
        for col in STEP3_CSF_COLS:
            val = record.get(col)
            payload[col] = float(val) if pd.notna(val) else None

    return payload


def get_ground_truth(record: dict) -> str:
    return str(record.get("Diagnosi_CODIFICATA", "")).strip()


def get_all_codes(df: pd.DataFrame) -> list[str]:
    return df["Codice"].dropna().tolist()


def summary_stats(df: pd.DataFrame) -> dict:
    diag_counts = df["Diagnosi_CODIFICATA"].value_counts().to_dict()
    missing = {
        col: int(df[col].isna().sum())
        for col in STEP1_CLINICAL_COLS + STEP2_PLASMA_COLS + STEP3_CSF_COLS
        if col in df.columns
    }
    return {
        "total_patients": len(df),
        "diagnosis_distribution": diag_counts,
        "missing_values": missing,
    }
