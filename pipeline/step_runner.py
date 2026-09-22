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

import hashlib
import time
from typing import Optional

from config.settings import (
    LLM_MAX_TOKENS,
    LLM_RETRIES_ON_INVALID,
    LLM_TEMPERATURE,
    OLLAMA_MODEL,
    RAG_CORPORA,
    RAG_DEFAULT_MODE,
    RAG_STEPS,
)
from data.biomarkers import compute_biomarker_assessment
from data.loader import (
    CSF_CORE_BIOMARKER_COLS,
    PLASMA_BIOMARKER_COLS,
    PLASMA_HIERARCHY_COLS,
    build_step_payload,
)
from data.preprocessor import format_therapy_for_prompt
from llm.ollama_client import generate_detailed, parse_json_response_detailed
from llm.output_schemas import schema_for_step
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
        n = _n_available(record, PLASMA_HIERARCHY_COLS)
        if n == 0:
            return False, "Nessun biomarcatore plasmatico gerarchico disponibile"
        return True, f"Plasma gerarchico: {n}/{len(PLASMA_HIERARCHY_COLS)} disponibili"

    if step == 3:
        n = _n_available(record, CSF_CORE_BIOMARKER_COLS)
        if n == 0:
            return False, "Nessun biomarcatore liquorale core disponibile"
        return True, f"CSF core: {n}/{len(CSF_CORE_BIOMARKER_COLS)} disponibili"

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
    if not step_result or not step_result.get("feasible"):
        return None
    result = step_result.get("model_result", step_result.get("result"))
    if not isinstance(result, dict) or "error" in result:
        return None
    primary = result.get("primary_diagnosis")
    if isinstance(primary, str):
        return result if primary.strip() else None
    if not isinstance(primary, dict) or not primary.get("diagnosis"):
        return None
    return result


def _skipped(
    step: int,
    codice: str,
    reason: str,
    model: str,
    seed: int | None = None,
    eligible: bool = False,
    status: str = "ineligible",
) -> dict:
    return {
        "step": step,
        "patient_code": codice,
        "eligible": eligible,
        "execution_status": status,
        "feasible": False,
        "skip_reason": reason,
        "result": None,
        "model_result": None,
        "raw_response": None,
        "parse_metadata": None,
        "generation_metadata": None,
        "generation_attempts": [],
        "model_input": None,
        "duration_s": 0.0,
        "model_used": model,
        "seed": seed,
        "computed_biomarkers": None,
        "rag_sources": [],
        "prompt_chars": None,
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
    seed: int | None = None,
    rag_bundle: Optional[dict] = None,
    rag_mode: str = RAG_DEFAULT_MODE,
    retries_on_invalid: int = LLM_RETRIES_ON_INVALID,
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
        return _skipped(step, codice, skip_reason, model, seed)

    if step == 2 and step1_result is None:
        return _skipped(
            2, codice, "Risultato Step 1 mancante", model, seed,
            eligible=True, status="blocked",
        )
    if step == 3 and step1_result is None:
        return _skipped(
            3, codice, "Risultato Step 1 mancante", model, seed,
            eligible=True, status="blocked",
        )
    if step not in (1, 2, 3):
        return _skipped(step, codice, f"Step {step} non valido", model, seed)

    payload = build_step_payload(record, step, terapia_parsed.get("summary"))
    terapia_fmt = format_therapy_for_prompt(terapia_parsed)
    biomarker_assessment = compute_biomarker_assessment(record, step) if step >= 2 else None

    active_rag = rag_bundle
    if step in RAG_STEPS and active_rag is None:
        active_rag = prepare_rag_bundle(
            step, record, parent_store, child_store, rag_mode=rag_mode
        )
    rag_context = (active_rag or {}).get("context", "")
    rag_sources = (active_rag or {}).get("sources", [])

    if step == 1:
        system_prompt, user_prompt = build_step1_prompt(payload, rag_context, terapia_fmt)
    elif step == 2:
        system_prompt, user_prompt = build_step2_prompt(
            payload, step1_result, terapia_fmt, biomarker_assessment
        )
    else:
        system_prompt, user_prompt = build_step3_prompt(
            payload, step2_result, terapia_fmt, biomarker_assessment,
            step1_result=step1_result,
        )

    generated = _generate_validated(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        step=step,
        codice=codice,
        seed=seed,
        retries_on_invalid=retries_on_invalid,
        computed_biomarkers=biomarker_assessment,
        rag_sources=rag_sources,
    )
    result = generated["result"]
    for warning in (result or {}).get("consistency", {}).get("warnings", []):
        print(f"[Step {step}/{codice}] {warning}")

    prompt_hash = hashlib.sha256(
        f"{system_prompt}\0{user_prompt}".encode("utf-8")
    ).hexdigest()
    return {
        "step": step,
        "patient_code": codice,
        "eligible": True,
        "execution_status": "completed",
        "feasible": True,
        "skip_reason": None,
        "result": result,
        "model_result": generated["model_result"],
        "raw_response": generated["raw_response"],
        "parse_metadata": generated["parse_metadata"],
        "generation_metadata": generated["generation_metadata"],
        "generation_attempts": generated["attempts"],
        "model_input": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "prompt_sha256": prompt_hash,
            "payload": payload,
            "computed_biomarkers": biomarker_assessment,
            "rag_mode": rag_mode,
            "rag_queries": (active_rag or {}).get("queries", []),
            "rag_bundle_sha256": (active_rag or {}).get("sha256"),
        },
        "duration_s": round(time.time() - t_start, 2),
        "model_used": model,
        "seed": seed,
        "computed_biomarkers": biomarker_assessment,
        "rag_sources": rag_sources,
        "prompt_chars": len(system_prompt) + len(user_prompt),
    }


def _is_structurally_valid(result: dict | None) -> bool:
    """Una risposta è utilizzabile solo se contiene la diagnosi primaria."""
    if not isinstance(result, dict) or "error" in result:
        return False
    return bool((result.get("primary_diagnosis") or {}).get("diagnosis"))


def _generate_validated(
    system_prompt: str,
    user_prompt: str,
    model: str,
    step: int,
    codice: str,
    seed: int | None,
    retries_on_invalid: int,
    computed_biomarkers: dict | None,
    rag_sources: list[dict],
) -> dict:
    max_attempts = 1 + max(0, retries_on_invalid)
    attempts: list[dict] = []
    final: dict = {}

    for attempt_index in range(max_attempts):
        attempt_seed = None if seed is None else seed + attempt_index * 1_000_003
        generation = generate_detailed(
            prompt=user_prompt,
            system=system_prompt,
            model=model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            seed=attempt_seed,
            output_schema=schema_for_step(step),
        )
        parsed = parse_json_response_detailed(generation["response"])
        model_result = parsed["data"]
        result = validate_step_result(
            model_result,
            computed_biomarkers=computed_biomarkers,
            rag_sources=rag_sources,
        )
        generation_metadata = {
            key: value for key, value in generation.items() if key != "response"
        }
        attempt_record = {
            "attempt": attempt_index + 1,
            "seed": attempt_seed,
            "raw_response": generation["response"],
            "model_result": model_result,
            "parse_metadata": parsed["metadata"],
            "generation_metadata": generation_metadata,
            "structurally_valid": _is_structurally_valid(result),
        }
        attempts.append(attempt_record)
        final = {
            "raw_response": generation["response"],
            "model_result": model_result,
            "result": result,
            "parse_metadata": parsed["metadata"],
            "generation_metadata": generation_metadata,
            "attempts": attempts,
        }
        if attempt_record["structurally_valid"]:
            break
        if attempt_index + 1 < max_attempts:
            print(f"[Step {step}/{codice}] risposta priva di diagnosi primaria, "
                  f"nuovo tentativo ({attempt_index + 2}/{max_attempts})")

    return final


def prepare_rag_bundle(
    step: int,
    record: dict,
    parent_store,
    child_store,
    rag_mode: str = RAG_DEFAULT_MODE,
) -> dict:
    if rag_mode not in RAG_CORPORA:
        raise ValueError(f"Modalità RAG non valida: {rag_mode}")
    allowed_sources = set(RAG_CORPORA[rag_mode])
    if not (child_store and parent_store):
        context = "Knowledge base non disponibile."
        return {
            "step": step,
            "rag_mode": rag_mode,
            "allowed_sources": sorted(allowed_sources),
            "queries": [],
            "context": context,
            "sources": [],
            "sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        }

    queries = build_queries(step, clinical_context=_clinical_context(record))
    docs = retrieve_context(
        child_store, parent_store, queries, allowed_sources=allowed_sources
    )
    context = format_context_for_prompt(docs)
    sources = extract_rag_sources(docs)
    digest_material = rag_mode + "\0" + "\n".join(queries) + "\0" + context
    return {
        "step": step,
        "rag_mode": rag_mode,
        "allowed_sources": sorted(allowed_sources),
        "queries": queries,
        "context": context,
        "sources": sources,
        "sha256": hashlib.sha256(digest_material.encode("utf-8")).hexdigest(),
    }


def run_full_pipeline(
    record: dict,
    terapia_parsed: dict,
    model: str = OLLAMA_MODEL,
    parent_store=None,
    child_store=None,
    seed: int | None = None,
    rag_bundle: Optional[dict] = None,
    rag_mode: str = RAG_DEFAULT_MODE,
    retries_on_invalid: int = LLM_RETRIES_ON_INVALID,
) -> dict:
    """
    Esegue l'intera pipeline (step 1→2→3) per un paziente, propagando a ogni
    step l'output integrale di tutti quelli precedenti.
    """
    step_seeds = {
        step: None if seed is None else seed + step - 1 for step in (1, 2, 3)
    }
    s1 = run_step(
        1, record, terapia_parsed, model, parent_store, child_store,
        seed=step_seeds[1], rag_bundle=rag_bundle, rag_mode=rag_mode,
        retries_on_invalid=retries_on_invalid,
    )
    step1_result = usable_result(s1)

    s2 = run_step(
        2, record, terapia_parsed, model, parent_store, child_store,
        step1_result=step1_result, seed=step_seeds[2], rag_mode=rag_mode,
        retries_on_invalid=retries_on_invalid,
    )
    step2_result = usable_result(s2)

    s3 = run_step(
        3, record, terapia_parsed, model, parent_store, child_store,
        step1_result=step1_result, step2_result=step2_result,
        seed=step_seeds[3], rag_mode=rag_mode,
        retries_on_invalid=retries_on_invalid,
    )

    return {
        "step1": s1,
        "step2": s2,
        "step3": s3,
        "model": model,
        "rag_mode": rag_mode,
        "base_seed": seed,
        "step_seeds": step_seeds,
        "rag_bundle_sha256": (
            (rag_bundle or {}).get("sha256")
            or (s1.get("model_input") or {}).get("rag_bundle_sha256")
        ),
        "input_availability": {
            "plasma_available": _n_available(record, PLASMA_BIOMARKER_COLS),
            "plasma_total": len(PLASMA_BIOMARKER_COLS),
            "plasma_hierarchy_available": _n_available(record, PLASMA_HIERARCHY_COLS),
            "plasma_hierarchy_total": len(PLASMA_HIERARCHY_COLS),
            "csf_core_available": _n_available(record, CSF_CORE_BIOMARKER_COLS),
            "csf_core_total": len(CSF_CORE_BIOMARKER_COLS),
        },
        "total_duration_s": round(sum(r.get("duration_s", 0) for r in (s1, s2, s3)), 2),
    }
