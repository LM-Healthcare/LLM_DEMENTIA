"""
Valutazione della concordanza tra diagnosi LLM e ground truth.

Metriche calcolate per step:
  - Accuracy complessiva
  - Per-class precision, recall, F1
  - Confusion matrix
  - Cohen's Kappa
  - Concordanza per record (True/False)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from config.settings import RESULTS_DIR


def extract_llm_diagnosis(step_result: dict) -> Optional[str]:
    """Estrae il codice diagnosi dal risultato di uno step."""
    if step_result is None or not step_result.get("feasible"):
        return None
    result = step_result.get("result")
    if not result or "error" in result:
        return None
    primary = result.get("primary_diagnosis", {})
    return primary.get("diagnosis", None)


def extract_confidence(step_result: dict) -> Optional[float]:
    if step_result is None or not step_result.get("feasible"):
        return None
    result = step_result.get("result", {})
    if not result:
        return None
    return result.get("primary_diagnosis", {}).get("confidence_score", None)


def concordance_record(llm_diagnosis: Optional[str], ground_truth: str) -> Optional[bool]:
    """True se la diagnosi LLM coincide col ground truth, None se non processabile."""
    if llm_diagnosis is None:
        return None
    return str(llm_diagnosis).strip().upper() == str(ground_truth).strip().upper()


def evaluate_batch(
    pipeline_results: list[dict],
    ground_truths: list[str],
) -> dict:
    """
    Valuta i risultati di un batch di pazienti.

    Args:
        pipeline_results: lista di dict con chiavi step1, step2, step3
        ground_truths: lista di diagnosi ground truth (stessa lunghezza)

    Returns:
        dict con metriche aggregate e record-level concordance
    """
    if len(pipeline_results) != len(ground_truths):
        raise ValueError(
            f"Risultati ({len(pipeline_results)}) e ground truth "
            f"({len(ground_truths)}) non corrispondono"
        )

    records = []
    step_preds: dict[int, list] = {1: [], 2: [], 3: []}
    step_truths: dict[int, list] = {1: [], 2: [], 3: []}

    for pr, gt in zip(pipeline_results, ground_truths):
        patient_code = pr.get("step1", {}).get("patient_code", "?")
        record = {
            "patient_code": patient_code,
            "ground_truth": gt,
        }

        for step in [1, 2, 3]:
            key = f"step{step}"
            sr = pr.get(key, {})
            llm_diag = extract_llm_diagnosis(sr)
            conf = extract_confidence(sr)
            concord = concordance_record(llm_diag, gt)
            feasible = sr.get("feasible", False)

            record[f"step{step}_prediction"] = llm_diag
            record[f"step{step}_confidence"] = conf
            record[f"step{step}_concordant"] = concord
            record[f"step{step}_feasible"] = feasible
            record[f"step{step}_duration_s"] = sr.get("duration_s", 0)

            if feasible and llm_diag is not None:
                step_preds[step].append(llm_diag.upper())
                step_truths[step].append(gt.upper())

        records.append(record)

    metrics_per_step = {}
    for step in [1, 2, 3]:
        preds = step_preds[step]
        truths = step_truths[step]
        if preds:
            metrics_per_step[f"step{step}"] = _compute_metrics(preds, truths)
        else:
            metrics_per_step[f"step{step}"] = {"error": "Nessuna predizione disponibile"}

    return {
        "records": records,
        "metrics_per_step": metrics_per_step,
        "total_patients": len(records),
        "summary": _build_summary(records, metrics_per_step),
    }


def _compute_metrics(predictions: list[str], truths: list[str]) -> dict:
    all_classes = sorted(set(predictions + truths))
    n = len(predictions)
    correct = sum(p == t for p, t in zip(predictions, truths))
    accuracy = correct / n if n > 0 else 0.0

    per_class: dict[str, dict] = {}
    for cls in all_classes:
        tp = sum(1 for p, t in zip(predictions, truths) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(predictions, truths) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(predictions, truths) if p != cls and t == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for t in truths if t == cls),
        }

    kappa = _cohen_kappa(predictions, truths, all_classes)

    confusion = defaultdict(lambda: defaultdict(int))
    for p, t in zip(predictions, truths):
        confusion[t][p] += 1

    return {
        "n_evaluated": n,
        "accuracy": round(accuracy, 4),
        "cohen_kappa": round(kappa, 4),
        "per_class": per_class,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "classes": all_classes,
    }


def _cohen_kappa(predictions: list[str], truths: list[str], classes: list[str]) -> float:
    n = len(predictions)
    if n == 0:
        return 0.0
    po = sum(p == t for p, t in zip(predictions, truths)) / n
    pred_freq = {c: predictions.count(c) / n for c in classes}
    truth_freq = {c: truths.count(c) / n for c in classes}
    pe = sum(pred_freq.get(c, 0) * truth_freq.get(c, 0) for c in classes)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0


def _build_summary(records: list[dict], metrics: dict) -> dict:
    total = len(records)
    summary = {"total_patients": total}

    for step in [1, 2, 3]:
        key = f"step{step}"
        feasible = sum(1 for r in records if r.get(f"{key}_feasible"))
        concordant = sum(1 for r in records if r.get(f"{key}_concordant") is True)
        m = metrics.get(key, {})
        summary[key] = {
            "feasible_patients": feasible,
            "concordant_patients": concordant,
            "accuracy": m.get("accuracy", 0),
            "cohen_kappa": m.get("cohen_kappa", 0),
        }

    return summary


def save_results(
    evaluation: dict,
    pipeline_results: list[dict],
    model_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Salva i risultati completi su disco in JSON."""
    out_dir = output_dir or RESULTS_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model_name.replace(":", "_").replace("/", "_")
    filename = out_dir / f"results_{model_safe}_{ts}.json"

    output = {
        "model": model_name,
        "timestamp": ts,
        "evaluation": evaluation,
        "pipeline_results": pipeline_results,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"[Evaluator] Results saved to {filename}")
    return filename


def load_results(filepath: str | Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_result_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("results_*.json"), reverse=True)
