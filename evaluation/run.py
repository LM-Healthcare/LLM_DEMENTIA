"""CLI headless, resumable e auditabile per l'evaluation multi-run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from config.settings import (
    ACCEPTED_EXTREME_VALUES,
    BASE_DIR,
    DATABASE_PATH,
    DOCS_FOLDER,
    LLM_MAX_TOKENS,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    RAG_CORPORA,
    REFERENCE_VALUES,
    REFERENCE_VALUES_PATH,
)
from data.loader import get_all_codes, get_database, get_ground_truth, get_patient_record
from data.preprocessor import standardize_therapy
from evaluation.export import export_excel, read_jsonl
from llm.ollama_client import get_model_info, is_ollama_running
from pipeline.evaluator import concordance_record, extract_llm_diagnosis
from pipeline.step_runner import prepare_rag_bundle, run_full_pipeline
from rag.vector_store import iter_child_corpus, load_existing_store

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def git_info() -> dict:
    def command(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=BASE_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": command("rev-parse", "HEAD"),
            "branch": command("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(command("status", "--porcelain")),
        }
    except Exception as exc:
        commit = os.getenv("APP_GIT_COMMIT", "unknown")
        return {"commit": commit, "branch": "container", "dirty": False,
                "git_error": str(exc)}


def runtime_info() -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        version = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
        version.raise_for_status()
        info["ollama"] = version.json()
    except Exception as exc:
        info["ollama_error"] = str(exc)
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["gpu"] = gpu
    except Exception:
        info["gpu"] = "non disponibile"
    return info


def data_warnings(df: pd.DataFrame) -> list[str]:
    warnings = []
    for column, ref in REFERENCE_VALUES.items():
        maximum = ref.get("max")
        if maximum is None or ref.get("direction") not in {"upper", "range"}:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        for index in df.index[values > maximum * 10]:
            code = str(df.at[index, "Codice"])
            if (code, column) in ACCEPTED_EXTREME_VALUES:
                continue
            warnings.append(
                f"{code}: {column}={df.at[index, column]} supera 10x il cut-off {maximum}"
            )
    return warnings


def select_patients(df: pd.DataFrame, requested: list[str] | None, limit: int | None) -> list[str]:
    all_codes = [str(code) for code in get_all_codes(df)]
    if requested:
        unknown = sorted(set(requested) - set(all_codes))
        if unknown:
            raise ValueError(f"Pazienti non trovati: {', '.join(unknown)}")
        codes = requested
    else:
        codes = all_codes
    return codes[:limit] if limit else codes


def check_index_sources(child_store, rag_mode: str) -> None:
    indexed = {doc.metadata.get("source") for doc in iter_child_corpus(child_store)}
    required = set(RAG_CORPORA[rag_mode])
    missing = required - indexed
    if missing:
        raise RuntimeError(
            "Documenti non presenti nell'indice RAG: " + ", ".join(sorted(missing))
            + ". Ricostruire con: python scripts/build_rag.py --force"
        )


def build_manifest(args, patients: list[str], model_info: dict, warnings: list[str]) -> dict:
    corpus_files = [Path(DOCS_FOLDER) / name for name in RAG_CORPORA[args.rag_mode]]
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "model": args.model,
        "model_info": model_info,
        "rag_mode": args.rag_mode,
        "corpus": [
            {"name": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in corpus_files
        ],
        "embedding_model": OLLAMA_EMBED_MODEL,
        "dataset": {"path": Path(DATABASE_PATH).name, "sha256": sha256_file(DATABASE_PATH)},
        "reference_values": {
            "path": Path(REFERENCE_VALUES_PATH).name,
            "sha256": sha256_file(REFERENCE_VALUES_PATH),
        },
        "git": git_info(),
        "runtime": runtime_info(),
        "generation": {
            "temperature": LLM_TEMPERATURE,
            "num_ctx": LLM_NUM_CTX,
            "num_predict": LLM_MAX_TOKENS,
            "seed_start": args.seed_start,
            "retries_on_invalid": 0,
        },
        "requested_runs": args.runs,
        "patients": patients,
        "patient_count": len(patients),
        "data_warnings": warnings,
        "accepted_extreme_values": [
            {"patient_code": code, "column": column}
            for code, column in sorted(ACCEPTED_EXTREME_VALUES)
        ],
    }


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict) -> None:
    line = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_or_build_rag_cache(
    cache_dir: Path,
    patient_code: str,
    record: dict,
    rag_mode: str,
    dataset_hash: str,
    parent_store,
    child_store,
    refresh: bool,
) -> dict:
    cache_path = cache_dir / f"{patient_code}.json"
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("rag_mode") == rag_mode and cached.get("dataset_sha256") == dataset_hash:
            return cached["bundle"]

    bundle = prepare_rag_bundle(
        1, record, parent_store, child_store, rag_mode=rag_mode
    )
    write_json(cache_path, {
        "patient_code": patient_code,
        "rag_mode": rag_mode,
        "dataset_sha256": dataset_hash,
        "created_at": utc_now(),
        "bundle": bundle,
    })
    return bundle


def completed_keys(jsonl_path: Path, retry_errors: bool) -> set[tuple[str, int]]:
    keys = set()
    for record in read_jsonl(jsonl_path):
        if record.get("status") == "completed" or not retry_errors:
            keys.add((str(record.get("patient_code")), int(record.get("run"))))
    return keys


def annotate_pipeline(pipeline: dict, ground_truth: str) -> dict:
    correct_by_step: list[bool] = []
    for step in (1, 2, 3):
        step_result = pipeline[f"step{step}"]
        diagnosis = extract_llm_diagnosis(step_result)
        concordant = concordance_record(diagnosis, ground_truth)
        step_result["predicted_diagnosis"] = diagnosis
        step_result["concordant"] = concordant
        correct_by_step.append(concordant is True)

    completed_steps = [
        step for step in (1, 2, 3) if pipeline[f"step{step}"].get("feasible")
    ]
    final_step = max(completed_steps) if completed_steps else None
    final_correct = correct_by_step[final_step - 1] if final_step else False
    first_correct = next(
        (step for step, correct in enumerate(correct_by_step, 1) if correct), None
    )
    pipeline["ground_truth"] = ground_truth
    pipeline["final_step"] = final_step
    pipeline["final_correct"] = final_correct
    pipeline["first_correct_step"] = first_correct
    pipeline["correct_before_csf"] = correct_by_step[0] or correct_by_step[1]
    pipeline["sustained_from_step1"] = all(correct_by_step)
    pipeline["sustained_from_step2"] = correct_by_step[1] and correct_by_step[2]
    pipeline["csf_corrected_error"] = not correct_by_step[1] and correct_by_step[2]
    pipeline["csf_introduced_error"] = correct_by_step[1] and not correct_by_step[2]
    return pipeline


def result_record(
    patient_code: str,
    run_number: int,
    seed: int,
    model: str,
    rag_mode: str,
    ground_truth: str,
    pipeline: dict | None,
    started_at: str,
    error: str | None = None,
) -> dict:
    completed = pipeline is not None and error is None
    return {
        "schema_version": 1,
        "status": "completed" if completed else "error",
        "patient_code": patient_code,
        "run": run_number,
        "seed": seed,
        "model": model,
        "rag_mode": rag_mode,
        "ground_truth": ground_truth,
        "started_at": started_at,
        "finished_at": utc_now(),
        "final_correct": pipeline.get("final_correct") if pipeline else False,
        "first_correct_step": pipeline.get("first_correct_step") if pipeline else None,
        "correct_before_csf": pipeline.get("correct_before_csf") if pipeline else False,
        "sustained_from_step1": pipeline.get("sustained_from_step1") if pipeline else False,
        "sustained_from_step2": pipeline.get("sustained_from_step2") if pipeline else False,
        "csf_corrected_error": pipeline.get("csf_corrected_error") if pipeline else False,
        "csf_introduced_error": pipeline.get("csf_introduced_error") if pipeline else False,
        "error": error,
        "pipeline": pipeline,
    }


def acquire_lock(path: Path, force: bool) -> int:
    if force and path.exists():
        path.unlink()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Evaluation già in esecuzione o lock residuo: {path}. "
            "Usare --force-unlock solo dopo aver verificato che non ci siano processi attivi."
        ) from exc
    os.write(descriptor, f"pid={os.getpid()} started={utc_now()}".encode("utf-8"))
    return descriptor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation headless LLM Dementia")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--rag-mode", choices=sorted(RAG_CORPORA), default="budson")
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument("--patients", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=BASE_DIR / "results" / "evaluation")
    parser.add_argument("--experiment-name")
    parser.add_argument("--refresh-rag-cache", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--allow-data-warnings", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--excel-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs deve essere >= 1")
    return args


def main() -> None:
    args = parse_args()
    if not is_ollama_running():
        sys.exit(f"Ollama non raggiungibile su {OLLAMA_BASE_URL}")
    model_info = get_model_info(args.model)
    if model_info is None:
        sys.exit(f"Modello non installato: {args.model}")

    df = get_database()
    warnings = data_warnings(df)
    if warnings and not args.allow_data_warnings:
        print("Evaluation bloccata da valori da verificare:")
        for warning in warnings:
            print(f"  ! {warning}")
        sys.exit("Correggere il database o usare consapevolmente --allow-data-warnings")

    patients = select_patients(df, args.patients, args.limit)
    experiment_name = args.experiment_name or f"{safe_name(args.model)}__{args.rag_mode}"
    output_dir = args.output_root / experiment_name
    cache_dir = output_dir / "rag_cache"
    jsonl_path = output_dir / "runs.jsonl"
    manifest_path = output_dir / "manifest.json"
    excel_path = output_dir / "results.xlsx"
    lock_path = output_dir / ".evaluation.lock"

    manifest = build_manifest(args, patients, model_info, warnings)
    if manifest.get("git", {}).get("dirty") and not args.allow_dirty:
        sys.exit(
            "Repository con modifiche non committate: commit prima dell'evaluation "
            "o usa --allow-dirty solo per smoke test"
        )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = ("model", "rag_mode", "patients", "corpus", "generation")
        mismatches = [key for key in immutable if existing.get(key) != manifest.get(key)]
        nested_checks = (
            ("dataset.sha256", existing.get("dataset", {}).get("sha256"), manifest["dataset"]["sha256"]),
            ("reference_values.sha256", existing.get("reference_values", {}).get("sha256"),
             manifest["reference_values"]["sha256"]),
            ("model_info.digest", existing.get("model_info", {}).get("digest"),
             manifest.get("model_info", {}).get("digest")),
            ("git.commit", existing.get("git", {}).get("commit"), manifest.get("git", {}).get("commit")),
        )
        mismatches.extend(name for name, old, new in nested_checks if old != new)
        if mismatches:
            sys.exit(f"Manifest incompatibile per resume: {', '.join(mismatches)}")
        existing["requested_runs"] = max(existing.get("requested_runs", 0), args.runs)
        existing["last_resumed_at"] = utc_now()
        manifest = existing
    total = len(patients) * args.runs
    done = completed_keys(jsonl_path, args.retry_errors)
    pending = [(code, run) for code in patients for run in range(1, args.runs + 1)
               if (code, run) not in done]
    print(f"Esperimento : {experiment_name}")
    print(f"Modello     : {args.model}")
    print(f"RAG         : {args.rag_mode} — {', '.join(RAG_CORPORA[args.rag_mode])}")
    print(f"Pazienti    : {len(patients)}")
    print(f"Run         : {args.runs} ({total} pipeline totali)")
    print(f"Completate  : {len(done)}")
    print(f"Da eseguire : {len(pending)}")
    print(f"Output      : {output_dir}")
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    write_json(manifest_path, manifest)
    if not pending:
        if not args.no_excel:
            export_excel(jsonl_path, excel_path, manifest)
        print("Nessuna run da eseguire: esperimento già completo per il target richiesto.")
        return

    lock_descriptor = acquire_lock(lock_path, args.force_unlock)
    try:
        parent_store, child_store = load_existing_store()
        if parent_store is None:
            raise RuntimeError("Indice RAG non disponibile")
        check_index_sources(child_store, args.rag_mode)

        dataset_hash = manifest["dataset"]["sha256"]
        bundles: dict[str, dict] = {}
        print("\nPrecalcolo/lettura cache RAG...")
        for index, code in enumerate(patients, 1):
            record = get_patient_record(df, code)
            bundles[code] = load_or_build_rag_cache(
                cache_dir, code, record, args.rag_mode, dataset_hash,
                parent_store, child_store, args.refresh_rag_cache,
            )
            print(f"  RAG {index}/{len(patients)} {code}", end="\r")
        print()

        completed_count = len(done)
        for code, run_number in pending:
            seed = args.seed_start + run_number - 1
            record = get_patient_record(df, code)
            ground_truth = get_ground_truth(record)
            therapy_raw = str(record["TERAPIA"]) if pd.notna(record.get("TERAPIA")) else ""
            therapy = standardize_therapy(therapy_raw)
            started_at = utc_now()
            start = time.time()
            try:
                pipeline = run_full_pipeline(
                    record=record,
                    terapia_parsed=therapy,
                    model=args.model,
                    parent_store=parent_store,
                    child_store=child_store,
                    seed=seed,
                    rag_bundle=bundles[code],
                    rag_mode=args.rag_mode,
                    retries_on_invalid=0,
                )
                pipeline = annotate_pipeline(pipeline, ground_truth)
                output = result_record(
                    code, run_number, seed, args.model, args.rag_mode,
                    ground_truth, pipeline, started_at,
                )
            except BaseException as exc:
                output = result_record(
                    code, run_number, seed, args.model, args.rag_mode,
                    ground_truth, None, started_at,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            append_jsonl(jsonl_path, output)
            completed_count += 1
            elapsed = time.time() - start
            print(
                f"[{completed_count}/{total}] {code} run={run_number} seed={seed} "
                f"status={output['status']} {elapsed:.1f}s"
            )
            if not args.no_excel and completed_count % max(1, args.excel_every) == 0:
                export_excel(jsonl_path, excel_path, manifest)
    except KeyboardInterrupt:
        print("\nInterruzione richiesta: il JSONL è già salvato e il comando è resumable.")
    finally:
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)

    if not args.no_excel:
        export_excel(jsonl_path, excel_path, manifest)
    print("Evaluation terminata.")


if __name__ == "__main__":
    main()
