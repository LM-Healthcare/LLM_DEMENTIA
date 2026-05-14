from __future__ import annotations

from fastapi import APIRouter, HTTPException
from api.schemas import ResultFile
from pipeline.evaluator import list_result_files, load_results

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/", response_model=list[ResultFile])
def list_results():
    files = list_result_files()
    out = []
    for f in files:
        try:
            data = load_results(f)
            ev = data.get("evaluation", {})
            summary = ev.get("summary", {})
            out.append(ResultFile(
                filename=f.name,
                model=data.get("model", "?"),
                timestamp=data.get("timestamp", "?"),
                total_patients=summary.get("total_patients", 0),
                step1_accuracy=summary.get("step1", {}).get("accuracy"),
                step2_accuracy=summary.get("step2", {}).get("accuracy"),
                step3_accuracy=summary.get("step3", {}).get("accuracy"),
            ))
        except Exception:
            continue
    return out


@router.get("/{filename}")
def get_result(filename: str):
    files = list_result_files()
    match = next((f for f in files if f.name == filename), None)
    if match is None:
        raise HTTPException(404, f"File '{filename}' non trovato")
    return load_results(match)


@router.get("/{filename}/evaluation")
def get_evaluation(filename: str):
    files = list_result_files()
    match = next((f for f in files if f.name == filename), None)
    if match is None:
        raise HTTPException(404, f"File '{filename}' non trovato")
    data = load_results(match)
    return data.get("evaluation", {})
