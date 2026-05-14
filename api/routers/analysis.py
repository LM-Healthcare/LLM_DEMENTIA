from __future__ import annotations

import asyncio
from typing import Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import RunStepRequest, RunPipelineRequest, BatchRunRequest, StepResult
from api.state import get_app_state
from config.settings import DATABASE_PATH
from data.loader import load_database, get_patient_record, get_all_codes, get_ground_truth
from data.preprocessor import standardize_therapy
from pipeline.step_runner import run_step, run_full_pipeline
from pipeline.evaluator import evaluate_batch, save_results

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _get_record(code: str) -> tuple[dict, dict]:
    df = load_database(DATABASE_PATH)
    record = get_patient_record(df, code)
    if record is None:
        raise HTTPException(404, f"Paziente '{code}' non trovato")
    terapia_raw = str(record.get("TERAPIA", "")) if pd.notna(record.get("TERAPIA")) else ""
    terapia_parsed = standardize_therapy(terapia_raw)
    return record, terapia_parsed


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

    df = load_database(DATABASE_PATH)
    codes = req.patient_codes or get_all_codes(df)

    background_tasks.add_task(_batch_task, codes, req.model, state)
    return {"message": f"Batch avviato per {len(codes)} pazienti", "total": len(codes)}


async def _batch_task(codes: list[str], model: str, state):
    state.batch_running = True
    state.batch_progress = {"current": 0, "total": len(codes), "status": "running", "errors": []}

    df = load_database(DATABASE_PATH)
    pipeline_results = []
    ground_truths = []

    for i, code in enumerate(codes):
        state.batch_progress["current"] = i + 1
        state.batch_progress["current_patient"] = code
        try:
            record = get_patient_record(df, code)
            if record is None:
                state.batch_progress["errors"].append(f"{code}: not found")
                continue
            terapia_raw = str(record.get("TERAPIA", "")) if pd.notna(record.get("TERAPIA")) else ""
            terapia_parsed = standardize_therapy(terapia_raw)
            gt = get_ground_truth(record)

            result = await asyncio.get_event_loop().run_in_executor(
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
            ground_truths.append(gt)
        except Exception as e:
            state.batch_progress["errors"].append(f"{code}: {str(e)}")

    if pipeline_results:
        evaluation = evaluate_batch(pipeline_results, ground_truths)
        filepath = save_results(evaluation, pipeline_results, model)
        state.batch_progress["result_file"] = str(filepath)
        state.last_evaluation = evaluation

    state.batch_progress["status"] = "completed"
    state.batch_running = False


@router.get("/batch/progress")
def batch_progress():
    state = get_app_state()
    return state.batch_progress or {"status": "idle"}
