"""
Verifica che ogni step riceva effettivamente tutti gli input previsti.

Costruisce i prompt reali dei tre step per un paziente del database e controlla
la presenza di ogni blocco informativo obbligatorio: dati clinici, terapia
standardizzata, fattori di rischio, contesto RAG, biomarcatori plasmatici e
liquorali, indici epato-renali e output integrale degli step precedenti.

Uso:
    python scripts/verify_prompts.py             # paziente con dati più completi
    python scripts/verify_prompts.py --code T45
    python scripts/verify_prompts.py --dump 1    # stampa il prompt dello step 1
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from data.loader import (
    CSF_BIOMARKER_COLS,
    PLASMA_BIOMARKER_COLS,
    SAFETY_LAB_COLS,
    build_step_payload,
    get_database,
    get_patient_record,
)
from data.preprocessor import format_therapy_for_prompt, standardize_therapy
from llm.prompt_builder import build_step1_prompt, build_step2_prompt, build_step3_prompt
from pipeline.step_runner import _clinical_context
from rag.retriever import build_queries, extract_rag_sources, format_context_for_prompt, retrieve_context
from rag.vector_store import load_existing_store

# Esito plausibile di uno step 1, usato per verificare la propagazione.
FAKE_STEP1 = {
    "primary_diagnosis": {
        "diagnosis": "AD", "label": "Malattia di Alzheimer", "probability": "MEDIA",
        "confidence_score": 0.45,
        "reasoning": "RAGIONAMENTO_STEP1_SENTINELLA: deficit mnesico episodico progressivo.",
    },
    "differential_diagnoses": [
        {"diagnosis": "VAD", "label": "Demenza Vascolare", "probability": "MEDIA",
         "confidence_score": 0.3, "reasoning": "DIFFERENZIALE_STEP1_SENTINELLA: fattori di rischio vascolari."},
    ],
    "key_clinical_features": ["FEATURE_STEP1_SENTINELLA"],
    "missing_information": ["MANCANTE_STEP1_SENTINELLA"],
    "clinical_summary": "SINTESI_STEP1_SENTINELLA",
    "step1_limitations": "LIMITI_STEP1_SENTINELLA",
}

FAKE_STEP2 = {
    "primary_diagnosis": {
        "diagnosis": "AD", "label": "Malattia di Alzheimer", "probability": "ALTA",
        "confidence_score": 0.7,
        "reasoning": "RAGIONAMENTO_STEP2_SENTINELLA: p-tau217 elevata.",
    },
    "differential_diagnoses": [
        {"diagnosis": "VAD", "label": "Demenza Vascolare", "probability": "BASSA",
         "confidence_score": 0.15, "reasoning": "DIFFERENZIALE_STEP2_SENTINELLA"},
    ],
    "biomarker_reliability": {
        "renal_function_ok": True, "hepatic_function_ok": True, "biomarkers_reliable": True,
        "reliability_notes": "AFFIDABILITA_STEP2_SENTINELLA",
    },
    "plasma_biomarker_interpretation": {
        "plasma_ptau217": {"value": 0.4, "status": "PATOLOGICO", "interpretation": "x"},
    },
    "update_from_step1": "AGGIORNAMENTO_STEP2_SENTINELLA",
    "clinical_summary": "SINTESI_STEP2_SENTINELLA",
}


def pick_patient(df: pd.DataFrame, code: str | None) -> dict:
    if code:
        record = get_patient_record(df, code)
        if record is None:
            sys.exit(f"Paziente '{code}' non trovato")
        return record

    cols = (PLASMA_BIOMARKER_COLS + CSF_BIOMARKER_COLS + SAFETY_LAB_COLS
            + ["ANAMNESI", "EON", "TERAPIA", "MMSE"])
    present = [c for c in cols if c in df.columns]
    best = df.assign(_filled=df[present].notna().sum(axis=1)).sort_values("_filled", ascending=False)
    return best.iloc[0].drop(labels=["_filled"]).to_dict()


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if condition else 'MANCA'}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


def verify(record: dict, dump: int | None) -> bool:
    codice = str(record.get("Codice"))
    terapia = standardize_therapy(
        str(record["TERAPIA"]) if pd.notna(record.get("TERAPIA")) else ""
    )
    terapia_fmt = format_therapy_for_prompt(terapia)

    parent_store, child_store = load_existing_store()
    if parent_store is None:
        sys.exit("Indice RAG assente: esegui prima 'python scripts/build_rag.py --force'")

    queries = build_queries(1, clinical_context=_clinical_context(record))
    docs = retrieve_context(child_store, parent_store, queries)
    rag_context = format_context_for_prompt(docs)
    sources = extract_rag_sources(docs)

    anamnesi = str(record.get("ANAMNESI", ""))[:60].strip()
    eon = str(record.get("EON", ""))[:60].strip() if pd.notna(record.get("EON")) else ""
    first_drug = next((d["molecule"] for d in terapia["drugs"] if d["status"] == "found"), "")

    prompts = {
        1: build_step1_prompt(build_step_payload(record, 1, terapia["summary"]), rag_context, terapia_fmt),
        2: build_step2_prompt(build_step_payload(record, 2, terapia["summary"]), FAKE_STEP1, terapia_fmt),
        3: build_step3_prompt(build_step_payload(record, 3, terapia["summary"]), FAKE_STEP2,
                              terapia_fmt, step1_result=FAKE_STEP1),
    }

    print(f"\nPaziente: {codice}")
    print(f"Fonti RAG recuperate: {len(sources)} "
          f"(lunghezza media {sum(len(s['text']) for s in sources) // max(1, len(sources))} chars)")
    print(f"Query RAG generate: {len(queries)}")

    ok = True
    for step, (system, user) in prompts.items():
        print(f"\n--- STEP {step} ({len(system):,} chars system + {len(user):,} chars user) ---")

        ok &= check("diagnosi ammesse + valori di riferimento", "VALORI DI RIFERIMENTO" in system)
        ok &= check("schema JSON dello step", f'"step": {step}' in system)
        ok &= check("anamnesi del paziente", bool(anamnesi) and anamnesi in user)
        if eon:
            ok &= check("esame obiettivo neurologico", eon in user)
        ok &= check("MMSE", "MMSE:" in user)
        ok &= check("terapia standardizzata", "Terapia domiciliare" in user or "Nessuna terapia" in user)
        if first_drug:
            ok &= check("molecole della terapia", first_drug in user, first_drug)
        ok &= check("fattori di rischio", "FATTORI DI RISCHIO" in user)

        has_rag = "CONTESTO DALLA KNOWLEDGE BASE" in user
        ok &= check("contesto RAG", has_rag if step == 1 else not has_rag,
                    "atteso solo allo step 1")
        if step == 1:
            ok &= check("fonti numerate e citabili", "[Fonte 1]" in user)
            ok &= check("istruzioni di citazione verbatim", "rag_evidence" in system)

        if step >= 2:
            ok &= check("biomarcatori plasmatici", "BIOMARCATORI EMATICI" in user)
            ok &= check("indici epato-renali", "FUNZIONALITÀ EPATO-RENALE" in user)
            ok &= check("output integrale step 1", "RAGIONAMENTO_STEP1_SENTINELLA" in user)
            ok &= check("differenziali step 1 con motivazione", "DIFFERENZIALE_STEP1_SENTINELLA" in user)
            ok &= check("sintesi step 1", "SINTESI_STEP1_SENTINELLA" in user)
            ok &= check("istruzione anti-anchoring", "ISTRUZIONE CRITICA" in user)

        if step == 3:
            ok &= check("biomarcatori liquorali", "BIOMARCATORI LIQUOR" in user)
            ok &= check("output integrale step 2", "RAGIONAMENTO_STEP2_SENTINELLA" in user)
            ok &= check("affidabilità biomarcatori da step 2", "AFFIDABILITA_STEP2_SENTINELLA" in user)
            ok &= check("richiesta profilo ATN", "ATN" in user)
            ok &= check("richiesta concordanza plasma-liquor", "concordanza plasma-liquor" in user)

        if dump == step:
            print("\n" + "=" * 70 + f"\nSYSTEM PROMPT STEP {step}\n" + "=" * 70)
            print(system)
            print("\n" + "=" * 70 + f"\nUSER PROMPT STEP {step}\n" + "=" * 70)
            print(user)

    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verifica completezza degli input dei prompt")
    ap.add_argument("--code", help="codice paziente (default: quello con più dati)")
    ap.add_argument("--dump", type=int, choices=[1, 2, 3], help="stampa il prompt dello step")
    args = ap.parse_args()

    ok = verify(pick_patient(get_database(), args.code), args.dump)
    print("\n" + ("TUTTI I CONTROLLI SUPERATI" if ok else "ALCUNI CONTROLLI FALLITI"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
