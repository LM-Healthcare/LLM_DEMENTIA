"""
Esecutore dei 3 step diagnostici per un singolo paziente.

Flusso:
  1. Carica i dati del paziente (step payload)
  2. Verifica disponibilità biomarcatori
  3. Esegue il retrieval RAG (step 1) o usa contesto minimo (step 2-3)
  4. Costruisce il prompt appropriato
  5. Chiama Ollama e parsa la risposta JSON
  6. Restituisce il risultato strutturato
"""

from __future__ import annotations

import json
import time
from typing import Optional

from config.settings import OLLAMA_MODEL
from data.loader import build_step_payload, STEP2_PLASMA_COLS, STEP3_CSF_COLS
from data.preprocessor import format_therapy_for_prompt
from llm.ollama_client import generate, parse_json_response
from llm.prompt_builder import build_step1_prompt, build_step2_prompt, build_step3_prompt
from rag.retriever import retrieve_context, build_queries, format_context_for_prompt, extract_rag_sources


def _check_step_feasibility(record: dict, step: int) -> tuple[bool, str]:
    """Verifica se il paziente ha abbastanza dati per lo step."""
    if step == 1:
        has_anamnesi = bool(record.get("ANAMNESI", "").strip())
        if not has_anamnesi:
            return False, "Anamnesi mancante: Step 1 non eseguibile"
        return True, ""

    if step == 2:
        plasma_cols = ["Plasma_Ab4240", "plasma_ptau217", "plasma_pt181", "plasma_NfL"]
        available = [c for c in plasma_cols if record.get(c) is not None]
        if len(available) == 0:
            return False, "Tutti i biomarcatori plasmatici mancanti: Step 2 non eseguibile"
        return True, f"Biomarcatori plasma disponibili: {len(available)}/{len(plasma_cols)}"

    if step == 3:
        csf_cols = ["CSF_Ab42", "CSF_Ab4240", "CSF_ttau", "CSF_ptau"]
        available = [c for c in csf_cols if record.get(c) is not None]
        if len(available) < 2:
            return False, "Biomarcatori liquorali insufficienti: Step 3 non eseguibile"
        return True, f"Biomarcatori CSF disponibili: {len(available)}/{len(csf_cols)}"

    return False, f"Step {step} non valido"


def run_step(
    step: int,
    record: dict,
    terapia_parsed: dict,
    model: str = OLLAMA_MODEL,
    parent_store=None,
    child_store=None,
    step1_result: Optional[dict] = None,
    step2_result: Optional[dict] = None,
) -> dict:
    """
    Esegue lo step specificato per un paziente.

    Args:
        step: 1, 2 o 3
        record: riga del DataFrame come dict
        terapia_parsed: output di standardize_therapy()
        model: nome modello Ollama
        parent_store: ChromaDB parent collection (obbligatorio per step 1)
        child_store: ChromaDB child collection (obbligatorio per step 1)
        step1_result: risultato step 1 (obbligatorio per step 2)
        step2_result: risultato step 2 (obbligatorio per step 3)

    Returns:
        dict con chiavi: step, patient_code, result, raw_response, duration_s,
                         feasible, skip_reason, model_used
    """
    t_start = time.time()
    codice = record.get("Codice", "N/D")

    feasible, skip_reason = _check_step_feasibility(record, step)
    if not feasible:
        return {
            "step": step,
            "patient_code": codice,
            "feasible": False,
            "skip_reason": skip_reason,
            "result": None,
            "raw_response": None,
            "duration_s": 0.0,
            "model_used": model,
        }

    payload = build_step_payload(record, step, terapia_parsed.get("summary"))
    terapia_fmt = format_therapy_for_prompt(terapia_parsed)

    if step == 1:
        anamnesi_snippet = str(record.get("ANAMNESI", ""))[:300]
        queries = build_queries(1, clinical_context=anamnesi_snippet)
        if child_store and parent_store:
            docs = retrieve_context(child_store, parent_store, queries)
            rag_context = format_context_for_prompt(docs)
            rag_sources = extract_rag_sources(docs)
        else:
            rag_context = "Knowledge base non disponibile."
            rag_sources = []
        system_prompt, user_prompt = build_step1_prompt(payload, rag_context, terapia_fmt)

    elif step == 2:
        if step1_result is None:
            return {
                "step": 2, "patient_code": codice, "feasible": False,
                "skip_reason": "Risultato Step 1 mancante",
                "result": None, "raw_response": None,
                "duration_s": 0.0, "model_used": model,
            }
        system_prompt, user_prompt = build_step2_prompt(payload, step1_result, terapia_fmt)

    elif step == 3:
        if step2_result is None:
            return {
                "step": 3, "patient_code": codice, "feasible": False,
                "skip_reason": "Risultato Step 2 mancante",
                "result": None, "raw_response": None,
                "duration_s": 0.0, "model_used": model,
            }
        system_prompt, user_prompt = build_step3_prompt(payload, step2_result, terapia_fmt)

    else:
        return {
            "step": step, "patient_code": codice, "feasible": False,
            "skip_reason": f"Step {step} non valido",
            "result": None, "raw_response": None,
            "duration_s": 0.0, "model_used": model,
        }

    raw_response = generate(
        prompt=user_prompt,
        system=system_prompt,
        model=model,
        temperature=0.1,
        max_tokens=4096,
    )

    result = parse_json_response(raw_response)

    duration = round(time.time() - t_start, 2)

    return {
        "step": step,
        "patient_code": codice,
        "feasible": True,
        "skip_reason": None,
        "result": result,
        "raw_response": raw_response,
        "duration_s": duration,
        "model_used": model,
        "rag_sources": rag_sources if step == 1 else [],
    }


def run_full_pipeline(
    record: dict,
    terapia_parsed: dict,
    model: str = OLLAMA_MODEL,
    parent_store=None,
    child_store=None,
) -> dict:
    """
    Esegue l'intera pipeline (step 1→2→3) per un paziente.
    """
    results = {}

    s1 = run_step(1, record, terapia_parsed, model, parent_store, child_store)
    results["step1"] = s1
    step1_result = s1.get("result") if s1.get("feasible") else None

    s2 = run_step(2, record, terapia_parsed, model,
                  step1_result=step1_result)
    results["step2"] = s2
    step2_result = s2.get("result") if s2.get("feasible") else step1_result

    s3 = run_step(3, record, terapia_parsed, model,
                  step2_result=step2_result)
    results["step3"] = s3

    results["total_duration_s"] = round(
        sum(r.get("duration_s", 0) for r in [s1, s2, s3]), 2
    )

    return results
