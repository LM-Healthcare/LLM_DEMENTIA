"""
Test end-to-end della pipeline su un paziente reale.

Esegue i tre step con un modello Ollama e riporta diagnosi, tempi, fonti citate
e la valutazione di concordanza. Serve a validare l'intera catena (dati →
prompt → LLM → parsing → valutazione) senza passare dall'interfaccia.

Uso:
    python scripts/smoke_pipeline.py --model qwen3.5:4b
    python scripts/smoke_pipeline.py --model llama3.2 --code T85
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from data.loader import get_database, get_ground_truth, get_patient_record
from data.preprocessor import standardize_therapy
from pipeline.evaluator import evaluate_batch
from pipeline.step_runner import run_full_pipeline
from rag.vector_store import load_existing_store


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test della pipeline")
    ap.add_argument("--model", required=True, help="nome del modello Ollama")
    ap.add_argument("--code", help="codice paziente (default: il primo con CSF)")
    ap.add_argument("--seed", type=int, default=1001, help="seed riproducibile")
    ap.add_argument("--rag-mode", choices=["budson", "casebook", "both"], default="budson")
    args = ap.parse_args()

    df = get_database()
    if args.code:
        record = get_patient_record(df, args.code)
        if record is None:
            sys.exit(f"Paziente '{args.code}' non trovato")
    else:
        with_csf = df[df["CSF_Ab42"].notna()]
        record = (with_csf if not with_csf.empty else df).iloc[0].to_dict()

    parent_store, child_store = load_existing_store()
    if parent_store is None:
        sys.exit("Indice RAG assente: esegui 'python scripts/build_rag.py --force'")

    terapia = standardize_therapy(
        str(record["TERAPIA"]) if pd.notna(record.get("TERAPIA")) else ""
    )
    gt = get_ground_truth(record)

    print(f"Paziente {record.get('Codice')} | ground truth: {gt} | "
          f"modello: {args.model} | RAG: {args.rag_mode}\n")

    results = run_full_pipeline(
        record, terapia, args.model, parent_store, child_store,
        seed=args.seed, rag_mode=args.rag_mode,
    )

    for step in (1, 2, 3):
        sr = results[f"step{step}"]
        out = sr.get("result") or {}
        primary = out.get("primary_diagnosis") or {}
        error = out.get("error")

        print(f"--- STEP {step} | {sr['duration_s']}s | prompt {sr.get('prompt_chars', 0):,} chars ---")
        if not sr["feasible"]:
            print(f"  saltato: {sr['skip_reason']}\n")
            continue
        if error:
            print(f"  ERRORE: {error}\n")
            continue

        print(f"  diagnosi    : {primary.get('diagnosis')} "
              f"({primary.get('probability')}, score={primary.get('confidence_score')})")
        print(f"  differenziali: {len(out.get('differential_diagnoses') or [])}")
        consistency = out.get("consistency") or {}
        print(f"  warning      : {len(consistency.get('warnings') or [])}")
        print(f"  parse        : {(sr.get('parse_metadata') or {}).get('status')}")
        ollama_meta = (sr.get("generation_metadata") or {}).get("ollama") or {}
        print(f"  token        : prompt={ollama_meta.get('prompt_eval_count')} "
              f"output={ollama_meta.get('eval_count')} reason={ollama_meta.get('done_reason')}")
        if sr["rag_sources"]:
            used = out.get("rag_sources_used") or []
            print(f"  fonti RAG   : {len(sr['rag_sources'])} recuperate, {len(used)} citate {used}")
            for ev in (out.get("rag_evidence") or [])[:2]:
                quote = str(ev.get("quote", ""))[:140]
                print(f"    [{ev.get('source_index')}] \"{quote}...\"")
        print(f"  reasoning   : {str(primary.get('reasoning', ''))[:200]}...\n")

    evaluation = evaluate_batch([results], [gt])
    summary = evaluation["summary"]
    print("=== CONCORDANZA CON IL GROUND TRUTH ===")
    for step in (1, 2, 3):
        rec = evaluation["records"][0]
        print(f"  step {step}: predetto={rec[f'step{step}_prediction']} "
              f"concordante={rec[f'step{step}_concordant']}")
    print(f"\nDurata totale: {results['total_duration_s']}s")
    print(f"Pazienti valutati: {summary['total_patients']}")


if __name__ == "__main__":
    main()
