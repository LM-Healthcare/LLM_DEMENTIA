"""Verifica filtri e diversificazione delle tre modalità RAG."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RAG_CORPORA
from data.loader import get_database, get_patient_record
from pipeline.step_runner import prepare_rag_bundle
from rag.vector_store import load_existing_store


def main() -> None:
    record = get_patient_record(get_database(), "T2")
    parent_store, child_store = load_existing_store()
    if parent_store is None:
        sys.exit("Indice RAG assente")

    bundles = {
        mode: prepare_rag_bundle(1, record, parent_store, child_store, rag_mode=mode)
        for mode in ("budson", "casebook", "both")
    }
    failures = []
    for mode, bundle in bundles.items():
        sources = [source["source"] for source in bundle["sources"]]
        allowed = set(RAG_CORPORA[mode])
        ok = bool(sources) and set(sources) <= allowed
        print(f"  [{'ok' if ok else 'FALLITO'}] {mode}: {Counter(sources)}")
        if not ok:
            failures.append(mode)

    both_sources = {source["source"] for source in bundles["both"]["sources"]}
    both_ok = both_sources == set(RAG_CORPORA["both"])
    print(f"  [{'ok' if both_ok else 'FALLITO'}] both contiene entrambi i manuali")
    if not both_ok:
        failures.append("both_diversification")

    hashes = {bundle["sha256"] for bundle in bundles.values()}
    hash_ok = len(hashes) == 3
    print(f"  [{'ok' if hash_ok else 'FALLITO'}] hash distinti per modalità")
    if not hash_ok:
        failures.append("hash")

    t1 = get_patient_record(get_database(), "T1")
    t1_bundle = prepare_rag_bundle(1, t1, parent_store, child_store, rag_mode="budson")
    personalized = t1_bundle["context"] != bundles["budson"]["context"]
    print(f"  [{'ok' if personalized else 'FALLITO'}] contesto personalizzato T1/T2")
    if not personalized:
        failures.append("patient_specific_context")

    if failures:
        print("Controlli falliti: " + ", ".join(failures))
        sys.exit(1)
    print("Tutte le modalità RAG sono corrette.")


if __name__ == "__main__":
    main()
