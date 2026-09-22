"""Esporta il JSONL autorevole in una tabella Excel per l'analisi."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _step_fields(pipeline: dict, step: int) -> dict:
    step_result = pipeline.get(f"step{step}") or {}
    result = step_result.get("result") or {}
    primary = result.get("primary_diagnosis") or {}
    consistency = result.get("consistency") or {}
    parse = step_result.get("parse_metadata") or {}
    generation = (step_result.get("generation_metadata") or {}).get("ollama") or {}
    return {
        f"STEP {step}": primary.get("diagnosis"),
        f"Confidence step {step}": primary.get("confidence_score"),
        f"Confidence raw step {step}": primary.get("reported_confidence_score"),
        f"Probabilità step {step}": primary.get("probability"),
        f"Motivazione diagnosi STEP {step}": primary.get("reasoning"),
        f"Analisi differenziale STEP {step}": result.get("diagnostic_reasoning"),
        f"Corretto STEP {step}": step_result.get("concordant"),
        f"Eleggibile STEP {step}": step_result.get("eligible"),
        f"Stato STEP {step}": step_result.get("execution_status"),
        f"Warning STEP {step}": " | ".join(consistency.get("warnings") or []),
        f"Argmax STEP {step}": consistency.get("argmax_diagnosis"),
        f"Somma score raw STEP {step}": consistency.get("score_sum"),
        f"Parse STEP {step}": parse.get("status"),
        f"Prompt token STEP {step}": generation.get("prompt_eval_count"),
        f"Output token STEP {step}": generation.get("eval_count"),
        f"Durata STEP {step} (s)": step_result.get("duration_s"),
    }


def flatten_record(record: dict) -> dict:
    pipeline = record.get("pipeline") or {}
    row = {
        "PAZIENTE": record.get("patient_code"),
        "RUN": record.get("run"),
        "SEED": record.get("seed"),
        "MODELLO": record.get("model"),
        "RAG_MODE": record.get("rag_mode"),
        "DIAGNOSI REALE": record.get("ground_truth"),
        "STATO RUN": record.get("status"),
        "DIAGNOSI FINALE CORRETTA": record.get("final_correct"),
        "PRIMO STEP CORRETTO": record.get("first_correct_step"),
        "CORRETTA PRIMA DEL LIQUOR": record.get("correct_before_csf"),
        "CORRETTA E PERSISTENTE DA STEP 1": record.get("sustained_from_step1"),
        "CORRETTA E PERSISTENTE DA STEP 2": record.get("sustained_from_step2"),
        "CSF CORREGGE ERRORE": record.get("csf_corrected_error"),
        "CSF INTRODUCE ERRORE": record.get("csf_introduced_error"),
        "PLASMA GERARCHICO DISPONIBILE": (pipeline.get("input_availability") or {}).get(
            "plasma_hierarchy_available"
        ),
        "CSF CORE DISPONIBILE": (pipeline.get("input_availability") or {}).get(
            "csf_core_available"
        ),
        "DURATA TOTALE (s)": pipeline.get("total_duration_s"),
        "ERRORE": record.get("error"),
    }
    for step in (1, 2, 3):
        row.update(_step_fields(pipeline, step))
    return row


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL corrotto alla riga {line_number}: {exc}") from exc
    return records


def export_excel(jsonl_path: Path, output_path: Path, manifest: dict | None = None) -> Path:
    records = read_jsonl(jsonl_path)
    latest = {
        (str(record.get("patient_code")), int(record.get("run"))): record
        for record in records
    }
    rows = [flatten_record(record) for _, record in sorted(latest.items())]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="runs", index=False)
        if manifest:
            manifest_rows = [
                {"chiave": key, "valore": json.dumps(value, ensure_ascii=False, default=str)}
                for key, value in manifest.items()
            ]
            pd.DataFrame(manifest_rows).to_excel(writer, sheet_name="manifest", index=False)
    return output_path
