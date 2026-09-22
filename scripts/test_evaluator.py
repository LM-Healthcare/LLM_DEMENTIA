"""Test delle metriche intention-to-evaluate e temporali."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.evaluator import evaluate_batch


def step(
    code: str | None, confidence: float = 0.8, eligible: bool = True
) -> dict:
    if not eligible:
        return {
            "eligible": False,
            "execution_status": "ineligible",
            "feasible": False,
            "patient_code": "T",
            "result": None,
            "duration_s": 0.0,
        }
    if code is None:
        return {
            "eligible": True,
            "execution_status": "completed",
            "feasible": True,
            "patient_code": "T",
            "result": {"error": "output non valido"},
            "duration_s": 1.0,
        }
    return {
        "eligible": True,
        "execution_status": "completed",
        "feasible": True,
        "patient_code": "T",
        "result": {
            "primary_diagnosis": {
                "diagnosis": code,
                "confidence_score": confidence,
                "reported_confidence_score": confidence,
            },
            "consistency": {"warnings": [], "primary_is_argmax": True},
        },
        "parse_metadata": {"status": "exact", "repaired": False},
        "duration_s": 1.0,
    }


def main() -> None:
    pipelines = [
        {
            "step1": step("AD"), "step2": step("AD"), "step3": step("AD"),
            "input_availability": {"plasma_available": 4, "plasma_total": 4},
        },
        {
            "step1": step(None), "step2": step("VAD"), "step3": step("AD"),
            "input_availability": {"plasma_available": 2, "plasma_total": 4},
        },
        {
            "step1": step("AD"), "step2": step(None, eligible=False),
            "step3": step(None, eligible=False),
            "input_availability": {"plasma_available": 0, "plasma_total": 4},
        },
    ]
    evaluation = evaluate_batch(pipelines, ["AD", "VAD", "AD"])
    step1 = evaluation["metrics_per_step"]["step1"]
    step2 = evaluation["metrics_per_step"]["step2"]
    records = evaluation["records"]

    checks = {
        "n_total include output invalido": step1["n_total"] == 3,
        "n_invalid": step1["n_invalid"] == 1,
        "accuracy intention-to-evaluate": step1["accuracy"] == 0.6667,
        "accuracy sui soli output validi": step1["valid_output_accuracy"] == 1.0,
        "Step 2 esclude non eleggibili": step2["n_total"] == 2 and step2["n_ineligible"] == 1,
        "primo paziente corretto da step 1": records[0]["first_correct_step"] == 1,
        "secondo paziente corretto da step 2": records[1]["first_correct_step"] == 2,
        "CSF introduce errore": records[1]["csf_introduced_error"] is True,
        "completezza preservata": records[1]["plasma_available"] == 2,
    }
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FALLITO'}] {name}")
    if failed:
        print(f"\nControlli falliti: {', '.join(failed)}")
        sys.exit(1)
    print("\nTutti i controlli dell'evaluator superati.")


if __name__ == "__main__":
    main()
