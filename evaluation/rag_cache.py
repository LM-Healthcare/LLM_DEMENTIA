"""Cache RAG condivisa tra modelli, versionata e ripristinabile."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config.settings import (
    DATABASE_PATH,
    DOCS_FOLDER,
    OLLAMA_EMBED_MODEL,
    RAG_BM25_WEIGHT,
    RAG_CANDIDATES_PER_QUERY,
    RAG_CHILD_CHUNK_OVERLAP,
    RAG_CHILD_CHUNK_SIZE,
    RAG_CLINICAL_QUERY_WEIGHT,
    RAG_CORPORA,
    RAG_MAX_PER_PAGE,
    RAG_MAX_PER_SOURCE,
    RAG_PARENT_CHUNK_OVERLAP,
    RAG_PARENT_CHUNK_SIZE,
    RAG_RRF_K,
    RAG_TOP_K,
)
from data.loader import get_database, get_patient_record
from pipeline.step_runner import prepare_rag_bundle
from rag.vector_store import iter_child_corpus, load_existing_store

RAG_CACHE_VERSION = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def ensure_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".write_test_{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"Directory cache non scrivibile: {directory}") from exc


def cache_root(output_root: Path, rag_mode: str) -> Path:
    if rag_mode not in RAG_CORPORA:
        raise ValueError(f"Modalità RAG non valida: {rag_mode}")
    return output_root / "rag_cache" / rag_mode


def cache_configuration(rag_mode: str) -> dict:
    corpus = []
    for name in RAG_CORPORA[rag_mode]:
        path = Path(DOCS_FOLDER) / name
        corpus.append({"name": name, "sha256": sha256_file(path), "size": path.stat().st_size})
    return {
        "cache_version": RAG_CACHE_VERSION,
        "rag_mode": rag_mode,
        "dataset_sha256": sha256_file(DATABASE_PATH),
        "corpus": corpus,
        "embedding_model": OLLAMA_EMBED_MODEL,
        "retrieval": {
            "top_k": RAG_TOP_K,
            "candidates_per_query": RAG_CANDIDATES_PER_QUERY,
            "bm25_weight": RAG_BM25_WEIGHT,
            "rrf_k": RAG_RRF_K,
            "clinical_query_weight": RAG_CLINICAL_QUERY_WEIGHT,
            "max_per_page": RAG_MAX_PER_PAGE,
            "max_per_source": RAG_MAX_PER_SOURCE,
            "parent_chunk_size": RAG_PARENT_CHUNK_SIZE,
            "parent_chunk_overlap": RAG_PARENT_CHUNK_OVERLAP,
            "child_chunk_size": RAG_CHILD_CHUNK_SIZE,
            "child_chunk_overlap": RAG_CHILD_CHUNK_OVERLAP,
        },
    }


def configuration_sha256(configuration: dict) -> str:
    payload = json.dumps(configuration, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_manifest(output_root: Path, rag_mode: str) -> dict:
    configuration = cache_configuration(rag_mode)
    return {
        "created_at": utc_now(),
        "configuration": configuration,
        "configuration_sha256": configuration_sha256(configuration),
    }


def check_index_sources(child_store, rag_mode: str) -> None:
    indexed = {doc.metadata.get("source") for doc in iter_child_corpus(child_store)}
    missing = set(RAG_CORPORA[rag_mode]) - indexed
    if missing:
        raise RuntimeError(
            "Documenti assenti nell'indice: " + ", ".join(sorted(missing))
            + ". Eseguire python scripts/build_rag.py --force"
        )


def load_bundle(path: Path, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if cached.get("configuration_sha256") != fingerprint:
        return None
    return cached.get("bundle")


def build_bundle(
    path: Path,
    patient_code: str,
    record: dict,
    rag_mode: str,
    configuration: dict,
    fingerprint: str,
    parent_store,
    child_store,
) -> dict:
    bundle = prepare_rag_bundle(1, record, parent_store, child_store, rag_mode=rag_mode)
    write_json(path, {
        "cache_version": RAG_CACHE_VERSION,
        "configuration_sha256": fingerprint,
        "patient_code": patient_code,
        "rag_mode": rag_mode,
        "dataset_sha256": configuration["dataset_sha256"],
        "created_at": utc_now(),
        "bundle": bundle,
    })
    return bundle


def acquire_lock(path: Path, force: bool = False) -> int:
    if force and path.exists():
        path.unlink()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Cache RAG già in costruzione o lock residuo: {path}"
        ) from exc
    os.write(descriptor, f"pid={os.getpid()} started={utc_now()}".encode("utf-8"))
    return descriptor


def write_progress(path: Path, **values) -> None:
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            current = {}
    current.update(values)
    current["updated_at"] = utc_now()
    write_json(path, current)


def precompute_cache(
    output_root: Path,
    rag_mode: str,
    patient_codes: list[str],
    refresh: bool = False,
    force_unlock: bool = False,
) -> dict[str, dict]:
    root = cache_root(output_root, rag_mode)
    ensure_writable(root)
    manifest = cache_manifest(output_root, rag_mode)
    configuration = manifest["configuration"]
    fingerprint = manifest["configuration_sha256"]
    manifest_path = root / "manifest.json"
    progress_path = root / "progress.json"
    lock_path = root / ".cache.lock"
    write_json(manifest_path, manifest)

    descriptor = acquire_lock(lock_path, force_unlock)
    bundles: dict[str, dict] = {}
    try:
        parent_store, child_store = load_existing_store()
        if parent_store is None:
            raise RuntimeError("Indice RAG non disponibile")
        check_index_sources(child_store, rag_mode)
        dataframe = get_database()
        write_progress(
            progress_path,
            status="running",
            rag_mode=rag_mode,
            cached=0,
            total=len(patient_codes),
        )
        for index, code in enumerate(patient_codes, 1):
            path = root / f"{code}.json"
            bundle = None if refresh else load_bundle(path, fingerprint)
            if bundle is None:
                record = get_patient_record(dataframe, code)
                if record is None:
                    raise ValueError(f"Paziente non trovato: {code}")
                bundle = build_bundle(
                    path, code, record, rag_mode, configuration, fingerprint,
                    parent_store, child_store,
                )
            bundles[code] = bundle
            write_progress(
                progress_path,
                status="running",
                current_patient=code,
                cached=index,
                total=len(patient_codes),
            )
            print(f"  RAG {index}/{len(patient_codes)} {code}", end="\r")
        print()
        write_progress(
            progress_path,
            status="completed",
            current_patient=None,
            cached=len(patient_codes),
            total=len(patient_codes),
        )
        return bundles
    except KeyboardInterrupt:
        write_progress(progress_path, status="interrupted")
        raise
    except BaseException as exc:
        write_progress(progress_path, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def load_shared_bundles(
    output_root: Path,
    rag_mode: str,
    patient_codes: list[str],
) -> dict[str, dict]:
    root = cache_root(output_root, rag_mode)
    manifest = cache_manifest(output_root, rag_mode)
    fingerprint = manifest["configuration_sha256"]
    bundles = {}
    missing = []
    for code in patient_codes:
        bundle = load_bundle(root / f"{code}.json", fingerprint)
        if bundle is None:
            missing.append(code)
        else:
            bundles[code] = bundle
    if missing:
        raise RuntimeError(
            f"Cache RAG {rag_mode} mancante/obsoleta per {len(missing)} pazienti. "
            f"Eseguire: python -m evaluation.cache --rag-mode {rag_mode}"
        )
    return bundles
