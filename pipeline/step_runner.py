"""
Esecutore dei 3 step diagnostici per un singolo paziente.

Il flusso è cumulativo: ogni step riceve il quadro clinico completo, l'output
integrale degli step precedenti e i dati di laboratorio specifici del proprio
livello. Il contesto RAG è iniettato negli step elencati in RAG_STEPS.

Flusso di uno step:
  1. Verifica la fattibilità sui dati disponibili
  2. Costruisce il payload del paziente (ground truth sempre esclusa)
  3. Esegue il retrieval RAG se previsto per lo step
  4. Costruisce il prompt
  5. Chiama Ollama e parsa la risposta JSON
"""

from __future__ import annotations

import time
from typing import Optional

from config.settings import (
    LLM_MAX_TOKENS,
    LLM_RETRIES_ON_INVALID,
    LLM_TEMPERATURE,
    OLLAMA_MODEL,
    RAG_STEPS,
)
from data.loader import CSF_CORE_BIOMARKER_COLS, PLASMA_BIOMARKER_COLS, build_step_payload
from data.preprocessor import format_therapy_for_prompt
from llm.ollama_client import generate, parse_json_response
from llm.prompt_builder import build_step1_prompt, build_step2_prompt, build_step3_prompt
from pipeline.validator import validate_step_result
from rag.retriever import retrieve_context, build_queries, format_context_for_prompt, extract_rag_sources


def _text(value) -> str:
    """Coercizione sicura a stringa: le celle vuote di pandas sono float('nan')."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def _n_available(record: dict, columns: list[str]) -> int:
    """Quante colonne hanno un valore utilizzabile (NaN di pandas incluso)."""
    return sum(
        1 for c in columns
        if (v := record.get(c)) is not None and not (isinstance(v, float) and v != v)
    )


def _check_step_feasibility(record: dict, step: int) -> tuple[bool, str]:
    """
    Verifica se il paziente ha abbastanza dati per lo step.
    Step 2 e 3 girano sempre: se i biomarcatori mancano, il modello mantiene le
    probabilità dello step precedente dichiarandolo nel ragionamento.
    """
    if step == 1:
        if not _text(record.get("ANAMNESI")):
            return False, "Anamnesi mancante: Step 1 non eseguibile"
        return True, ""

    if step == 2:
        n = _n_available(record, PLASMA_BIOMARKER_COLS)
        return True, f"Plasma: {n}/{len(PLASMA_BIOMARKER_COLS)} biomarcatori disponibili"

    if step == 3:
        n = _n_available(record, CSF_CORE_BIOMARKER_COLS)
        return True, f"CSF: {n}/{len(CSF_CORE_BIOMARKER_COLS)} biomarcatori disponibili"

    return False, f"Step {step} non valido"


def _clinical_context(record: dict) -> str:
    """
    Quadro clinico in testo libero usato come query semantica sulla knowledge base.
    Anamnesi ed EON insieme: l'esame obiettivo contiene i segni che discriminano
    le diagnosi (parkinsonismo, segni focali, disinibizione).
    """
    parts = [_text(record.get("ANAMNESI")), _text(record.get("EON"))]
    return "\n".join(p for p in parts if p)


def usable_result(step_result: Optional[dict]) -> Optional[dict]:
    """
    Output di uno step utilizzabile come input del successivo.

    Un JSON non parsabile produce {"error": ...}: passarlo allo step seguente
    inietterebbe una distribuzione diagnostica vuota spacciata per valida.
    """
    if not step_result or not step_result.get("feasible"):
        return None
    result = step_result.get("result")
    if not isinstance(result, dict) or "error" in result:
        return None
    return result


def _skipped(step: int, codice: str, reason: str, model: str) -> dict:
    return {
        "step": step,
        "patient_code": codice,
        "feasible": False,
        "skip_reason": reason,
        "result": None,
        "raw_response": None,
        "duration_s": 0.0,
        "model_used": model,
        "rag_sources": [],
    }


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
        parent_store/child_store: collezioni ChromaDB (necessarie per il RAG)
        step1_result: risultato step 1 (richiesto dagli step 2 e 3)
        step2_result: risultato step 2 (richiesto dallo step 3)

    Returns:
        dict con chiavi: step, patient_code, result, raw_response, duration_s,
                         feasible, skip_reason, model_used, rag_sources
    """
    t_start = time.time()
    codice = _text(record.get("Codice")) or "N/D"

    feasible, skip_reason = _check_step_feasibility(record, step)
    if not feasible:
        return _skipped(step, codice, skip_reason, model)

    if step == 2 and step1_result is None:
        return _skipped(2, codice, "Risultato Step 1 mancante", model)
    if step == 3 and step2_result is None:
        return _skipped(3, codice, "Risultato Step 2 mancante", model)
    if step not in (1, 2, 3):
        return _skipped(step, codice, f"Step {step} non valido", model)

    payload = build_step_payload(record, step, terapia_parsed.get("summary"))
    terapia_fmt = format_therapy_for_prompt(terapia_parsed)

    rag_sources: list[dict] = []
    if step in RAG_STEPS:
        rag_context, rag_sources = _retrieve(step, record, parent_store, child_store)
    else:
        rag_context = ""

    if step == 1:
        system_prompt, user_prompt = build_step1_prompt(payload, rag_context, terapia_fmt)
    elif step == 2:
        system_prompt, user_prompt = build_step2_prompt(payload, step1_result, terapia_fmt)
    else:
        system_prompt, user_prompt = build_step3_prompt(
            payload, step2_result, terapia_fmt, step1_result=step1_result
        )

    raw_response, result = _generate_validated(system_prompt, user_prompt, model, step, codice)
    for warning in (result or {}).get("consistency", {}).get("warnings", []):
        print(f"[Step {step}/{codice}] {warning}")

    return {
        "step": step,
        "patient_code": codice,
        "feasible": True,
        "skip_reason": None,
        "result": result,
        "raw_response": raw_response,
        "duration_s": round(time.time() - t_start, 2),
        "model_used": model,
        "rag_sources": rag_sources,
        "prompt_chars": len(system_prompt) + len(user_prompt),
    }


def _is_structurally_valid(result: dict | None) -> bool:
    """Una risposta è utilizzabile solo se contiene la diagnosi primaria."""
    if not isinstance(result, dict) or "error" in result:
        return False
    return bool((result.get("primary_diagnosis") or {}).get("diagnosis"))


def _generate_validated(
    system_prompt: str, user_prompt: str, model: str, step: int, codice: str
) -> tuple[str, dict]:
    """
    Chiama il modello e valida l'output, ritentando una volta se la risposta è
    inutilizzabile.

    I modelli piccoli a volte chiudono il JSON prima dei campi obbligatori: un
    singolo nuovo tentativo recupera il paziente invece di perderlo, e costa meno
    di rieseguire l'intera pipeline.
    """
    attempts = 1 + max(0, LLM_RETRIES_ON_INVALID)
    raw_response, result = "", {}

    for attempt in range(1, attempts + 1):
        raw_response = generate(
            prompt=user_prompt,
            system=system_prompt,
            model=model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        result = validate_step_result(parse_json_response(raw_response))
        if _is_structurally_valid(result):
            return raw_response, result
        if attempt < attempts:
            print(f"[Step {step}/{codice}] risposta priva di diagnosi primaria, "
                  f"nuovo tentativo ({attempt + 1}/{attempts})")

    return raw_response, result


def _retrieve(step: int, record: dict, parent_store, child_store) -> tuple[str, list[dict]]:
    """Recupera il contesto dalla knowledge base per lo step indicato."""
    if not (child_store and parent_store):
        return "Knowledge base non disponibile.", []

    queries = build_queries(step, clinical_context=_clinical_context(record))
    docs = retrieve_context(child_store, parent_store, queries)
    return format_context_for_prompt(docs), extract_rag_sources(docs)


def run_full_pipeline(
    record: dict,
    terapia_parsed: dict,
    model: str = OLLAMA_MODEL,
    parent_store=None,
    child_store=None,
) -> dict:
    """
    Esegue l'intera pipeline (step 1→2→3) per un paziente, propagando a ogni
    step l'output integrale di tutti quelli precedenti.
    """
    s1 = run_step(1, record, terapia_parsed, model, parent_store, child_store)
    step1_result = usable_result(s1)

    s2 = run_step(2, record, terapia_parsed, model, parent_store, child_store,
                  step1_result=step1_result)
    step2_result = usable_result(s2)

    # Se lo step 2 non ha prodotto un output valido, lo step 3 riparte da quello
    # dello step 1: perdere l'intera pipeline per un JSON malformato sarebbe peggio.
    s3 = run_step(3, record, terapia_parsed, model, parent_store, child_store,
                  step1_result=step1_result,
                  step2_result=step2_result or step1_result)

    return {
        "step1": s1,
        "step2": s2,
        "step3": s3,
        "total_duration_s": round(sum(r.get("duration_s", 0) for r in (s1, s2, s3)), 2),
    }
