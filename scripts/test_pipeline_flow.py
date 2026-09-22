"""Test del flusso con step non eleggibili per dati mancanti."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DIAGNOSIS_LABELS
from data.loader import get_database, get_patient_record
from data.preprocessor import standardize_therapy
from pipeline.step_runner import run_full_pipeline
from pipeline.validator import validate_step_result


def fake_generation(**kwargs) -> dict:
    step = kwargs["step"]
    scores = {code: (1.0 if code == "AD" else 0.0) for code in DIAGNOSIS_LABELS}
    model_result = {
        "step": step,
        "patient_code": kwargs["codice"],
        "diagnostic_reasoning": "test",
        "diagnosis_assessments": {
            code: {"reasoning": "test", "confidence_score": score}
            for code, score in scores.items()
        },
        "primary_diagnosis": "AD",
    }
    return {
        "raw_response": "{}",
        "model_result": model_result,
        "result": validate_step_result(
            model_result,
            computed_biomarkers=kwargs.get("computed_biomarkers"),
            rag_sources=kwargs.get("rag_sources"),
        ),
        "parse_metadata": {"status": "exact", "repaired": False},
        "generation_metadata": {"ollama": {}},
        "attempts": [],
    }


def therapy(record: dict) -> dict:
    raw = str(record["TERAPIA"]) if pd.notna(record.get("TERAPIA")) else ""
    return standardize_therapy(raw)


def main() -> None:
    df = get_database()
    bundle = {"context": "test", "sources": [], "queries": [], "sha256": "test"}
    failures = []

    with patch("pipeline.step_runner._generate_validated", side_effect=fake_generation):
        no_plasma = get_patient_record(df, "T4")
        result = run_full_pipeline(
            no_plasma, therapy(no_plasma), "test", seed=1001, rag_bundle=bundle
        )
        checks = {
            "T4 Step 1 eseguito": result["step1"]["feasible"] is True,
            "T4 Step 2 non eleggibile": result["step2"]["eligible"] is False,
            "T4 Step 3 eseguito direttamente da Step 1": result["step3"]["feasible"] is True,
            "Step 2 non viene falsamente sostituito":
                "OUTPUT STEP 2: non disponibile" in result["step3"]["model_input"]["user_prompt"],
        }

        no_csf = get_patient_record(df, "T94")
        result_t94 = run_full_pipeline(
            no_csf, therapy(no_csf), "test", seed=1001, rag_bundle=bundle
        )
        checks.update({
            "T94 Step 2 eleggibile": result_t94["step2"]["eligible"] is True,
            "T94 Step 3 non eleggibile": result_t94["step3"]["eligible"] is False,
            "T94 Step 3 status ineligible":
                result_t94["step3"]["execution_status"] == "ineligible",
        })

    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FALLITO'}] {name}")
        if not ok:
            failures.append(name)
    if failures:
        print("Controlli falliti: " + ", ".join(failures))
        sys.exit(1)
    print("Flusso con dati mancanti corretto.")


if __name__ == "__main__":
    main()
