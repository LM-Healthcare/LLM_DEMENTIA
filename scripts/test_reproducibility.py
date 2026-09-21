"""Verifica riproducibilità byte-per-byte dato lo stesso seed."""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from data.loader import get_database, get_patient_record
from data.preprocessor import standardize_therapy
from pipeline.step_runner import prepare_rag_bundle, run_step
from rag.vector_store import load_existing_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Test seed Ollama riproducibile")
    parser.add_argument("--model", required=True)
    parser.add_argument("--code", default="T2")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--compare-next-seed", action="store_true")
    args = parser.parse_args()

    record = get_patient_record(get_database(), args.code)
    if record is None:
        sys.exit(f"Paziente {args.code} non trovato")
    raw_therapy = str(record["TERAPIA"]) if pd.notna(record.get("TERAPIA")) else ""
    therapy = standardize_therapy(raw_therapy)
    parent_store, child_store = load_existing_store()
    if parent_store is None:
        sys.exit("Indice RAG non disponibile")

    bundle = prepare_rag_bundle(1, record, parent_store, child_store)
    results = [
        run_step(
            1, record, therapy, args.model, parent_store, child_store,
            seed=args.seed, rag_bundle=bundle, retries_on_invalid=0,
        )
        for _ in range(2)
    ]

    hashes = [hashlib.sha256((r.get("raw_response") or "").encode("utf-8")).hexdigest()
              for r in results]
    prompts = [(r.get("model_input") or {}).get("prompt_sha256") for r in results]
    print(f"Modello : {args.model}")
    print(f"Paziente: {args.code}")
    print(f"Seed    : {args.seed}")
    print(f"Prompt  : {prompts[0]}")
    print(f"Output 1: {hashes[0]}")
    print(f"Output 2: {hashes[1]}")
    print(f"Prompt identici: {prompts[0] == prompts[1]}")
    print(f"Output identici: {hashes[0] == hashes[1]}")

    if prompts[0] != prompts[1] or hashes[0] != hashes[1]:
        sys.exit(1)

    if args.compare_next_seed:
        different = run_step(
            1, record, therapy, args.model, parent_store, child_store,
            seed=args.seed + 1, rag_bundle=bundle, retries_on_invalid=0,
        )
        different_hash = hashlib.sha256(
            (different.get("raw_response") or "").encode("utf-8")
        ).hexdigest()
        print(f"Output seed {args.seed + 1}: {different_hash}")
        print(f"Seed diversi, output diversi: {different_hash != hashes[0]}")


if __name__ == "__main__":
    main()
