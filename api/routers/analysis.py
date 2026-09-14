from __future__ import annotations

import asyncio
from typing import Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import RunStepRequest, RunPipelineRequest, BatchRunRequest, StepResult, SaveIndividualRequest
from api.state import get_app_state
from data.loader import get_database, get_patient_record, get_all_codes, get_ground_truth
from data.preprocessor import standardize_therapy
from pipeline.step_runner import run_step, run_full_pipeline
from pipeline.evaluator import evaluate_batch, save_results

router = APIRouter(prefix="/analysis", tags=["analysis"])

# Ogni quanti pazienti salvare i risultati parziali di un batch.
BATCH_CHECKPOINT_EVERY = 5


def _parse_therapy(record: dict) -> dict:
    raw = record.get("TERAPIA")
    return standardize_therapy(str(raw) if pd.notna(raw) else "")


def _get_record(code: str) -> tuple[dict, dict]:
    record = get_patient_record(get_database(), code)
    if record is None:
        raise HTTPException(404, f"Paziente '{code}' non trovato")
    return record, _parse_therapy(record)


@router.post("/step", response_model=StepResult)
def run_single_step(req: RunStepRequest):
    record, terapia_parsed = _get_record(req.patient_code)
    state = get_app_state()
    result = run_step(
        step=req.step,
        record=record,
        terapia_parsed=terapia_parsed,
        model=req.model,
        parent_store=state.parent_store,
        child_store=state.child_store,
        step1_result=req.step1_result,
        step2_result=req.step2_result,
    )
    return StepResult(**result)


@router.post("/pipeline")
def run_pipeline(req: RunPipelineRequest):
    record, terapia_parsed = _get_record(req.patient_code)
    state = get_app_state()
    gt = get_ground_truth(record)
    results = run_full_pipeline(
        record=record,
        terapia_parsed=terapia_parsed,
        model=req.model,
        parent_store=state.parent_store,
        child_store=state.child_store,
    )
    results["ground_truth"] = gt
    results["patient_code"] = req.patient_code
    return results


@router.post("/batch")
async def run_batch(req: BatchRunRequest, background_tasks: BackgroundTasks):
    state = get_app_state()
    if state.batch_running:
        raise HTTPException(409, "Un batch è già in esecuzione")

    codes = req.patient_codes or get_all_codes(get_database())

    background_tasks.add_task(_batch_task, codes, req.model, state)
    return {"message": f"Batch avviato per {len(codes)} pazienti", "total": len(codes)}


async def _batch_task(codes: list[str], model: str, state):
    state.batch_running = True
    state.batch_progress = {"current": 0, "total": len(codes), "status": "running", "errors": []}

    df = get_database()
    pipeline_results: list[dict] = []
    ground_truths: list[str] = []
    loop = asyncio.get_event_loop()

    def _persist(status: str) -> None:
        """Salva valutazione e risultati parziali: un batch lungo non va perso."""
        if not pipeline_results:
            return
        evaluation = evaluate_batch(pipeline_results, ground_truths)
        filepath = save_results(evaluation, pipeline_results, model)
        state.batch_progress["result_file"] = str(filepath)
        state.batch_progress["saved_at_status"] = status
        state.last_evaluation = evaluation

    try:
        for i, code in enumerate(codes):
            state.batch_progress["current"] = i + 1
            state.batch_progress["current_patient"] = code
            try:
                record = get_patient_record(df, code)
                if record is None:
                    state.batch_progress["errors"].append(f"{code}: paziente non trovato")
                    continue
                terapia_parsed = _parse_therapy(record)

                result = await loop.run_in_executor(
                    None,
                    lambda r=record, t=terapia_parsed: run_full_pipeline(
                        record=r,
                        terapia_parsed=t,
                        model=model,
                        parent_store=state.parent_store,
                        child_store=state.child_store,
                    )
                )
                pipeline_results.append(result)
                ground_truths.append(get_ground_truth(record))
            except Exception as e:
                state.batch_progress["errors"].append(f"{code}: {e}")

            # Checkpoint periodico: se il processo muore il lavoro resta su disco.
            if (i + 1) % BATCH_CHECKPOINT_EVERY == 0:
                _persist("checkpoint")

        _persist("completed")
        state.batch_progress["status"] = "completed"
    except BaseException as e:
        _persist("interrupted")
        state.batch_progress["status"] = "interrupted"
        state.batch_progress["errors"].append(f"batch interrotto: {type(e).__name__}: {e}")
        raise
    finally:
        state.batch_running = False


@router.get("/batch/progress")
def batch_progress():
    state = get_app_state()
    return state.batch_progress or {"status": "idle"}


@router.post("/save-individual")
def save_individual(req: SaveIndividualRequest):
    import json
    from datetime import datetime
    from pathlib import Path
    from config.settings import RESULTS_DIR

    record, _ = _get_record(req.patient_code)
    gt = get_ground_truth(record)

    out_dir = Path(RESULTS_DIR) / "individual"
    out_dir.mkdir(exist_ok=True, parents=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = req.model.replace(":", "_").replace("/", "_")
    filename = out_dir / f"{req.patient_code}_{model_safe}_{ts}.json"

    data = {
        "patient_code": req.patient_code,
        "ground_truth": gt,
        "model": req.model,
        "timestamp": ts,
        "step1": req.step1,
        "step2": req.step2,
        "step3": req.step3,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return {"message": f"Risultati salvati", "filename": filename.name}
