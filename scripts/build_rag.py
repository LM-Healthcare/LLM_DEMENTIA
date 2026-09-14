"""
Costruzione e ispezione dell'indice RAG.

Uso:
    python scripts/build_rag.py --dry-run     # solo chunking, nessun embedding
    python scripts/build_rag.py --force       # reindicizza da zero
    python scripts/build_rag.py --probe       # testa il retrieval sull'indice
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config.settings import DOCS_FOLDER, OLLAMA_EMBED_MODEL, RAG_MIN_CHUNK_CHARS
from rag.document_processor import load_all_documents, create_parent_chunks, create_child_chunks


def chunk_report(parents, children) -> None:
    print("\n=== DOCUMENTI ===")
    for src, n in Counter(p.metadata["source"] for p in parents).most_common():
        print(f"  {n:6d} parent chunks  {src}")

    lens = sorted(len(p.page_content) for p in parents)
    print("\n=== LUNGHEZZA PARENT CHUNKS ===")
    print(f"  min={lens[0]}  mediana={int(statistics.median(lens))}  "
          f"media={statistics.mean(lens):.0f}  max={lens[-1]}")
    for thr in (RAG_MIN_CHUNK_CHARS, 200, 100, 50):
        n = sum(1 for l in lens if l < thr)
        print(f"  < {thr:4d} chars: {n:5d} ({n / len(lens) * 100:.2f}%)")

    with_section = sum(1 for p in parents if p.metadata.get("section"))
    print(f"\n  con sezione rilevata: {with_section}/{len(parents)} "
          f"({with_section / len(parents) * 100:.1f}%)")
    print(f"  child chunks: {len(children)}")

    print("\n=== CHUNK PIU' CORTI (verifica che non siano soli titoli) ===")
    for p in sorted(parents, key=lambda d: len(d.page_content))[:3]:
        print("-" * 70)
        print(f"  p.{p.metadata['page']} capitolo='{p.metadata.get('chapter', '')}' "
              f"sezione='{p.metadata.get('section', '')}' len={len(p.page_content)}")
        print("  " + p.page_content[:400].replace("\n", "\n  "))


def probe(top_k: int) -> None:
    from rag.vector_store import load_existing_store
    from rag.retriever import build_queries, retrieve_context, extract_rag_sources

    parent_store, child_store = load_existing_store()
    if parent_store is None:
        print("Nessun indice trovato. Esegui prima --force.")
        return

    clinical = (
        "Donna di 74 anni, 8 anni di scolarita'. Da circa due anni deficit mnesico "
        "progressivo per fatti recenti, con ripetitivita' nel discorso e disorientamento "
        "spaziale in ambienti noti. Riferita apatia e riduzione delle iniziative. "
        "Nessun disturbo della marcia, non cadute, non allucinazioni. MMSE 21/30. "
        "Ipertensione arteriosa in trattamento, dislipidemia."
    )
    queries = build_queries(1, clinical_context=clinical)
    print(f"{len(queries)} query generate. Retrieval in corso...\n")

    docs = retrieve_context(child_store, parent_store, queries, top_k=top_k)
    for s in extract_rag_sources(docs):
        print("=" * 78)
        print(f"[Fonte {s['index']}] score={s['retrieval_score']}  "
              f"{s['source']}, {s['page_ref']}")
        print(f"  sezione : {s['section'] or '(nessuna)'}")
        print(f"  lunghezza: {len(s['text'])} chars")
        if s["matched_quotes"]:
            print(f"  match   : {s['matched_quotes'][0][:150]}...")
        print("  ---")
        print("  " + s["text"][:500].replace("\n", "\n  "))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/ispezione indice RAG")
    ap.add_argument("--dry-run", action="store_true", help="solo chunking, nessun embedding")
    ap.add_argument("--force", action="store_true", help="cancella e reindicizza da zero")
    ap.add_argument("--probe", action="store_true", help="testa il retrieval")
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    if args.probe:
        probe(args.top_k)
        return

    print(f"Knowledge base : {DOCS_FOLDER}")
    print(f"Embedding model: {OLLAMA_EMBED_MODEL}\n")

    pages = load_all_documents()
    if not pages:
        print("Nessun documento caricato: controlla DOCS_FOLDER.")
        sys.exit(1)

    parents = create_parent_chunks(pages)
    children = create_child_chunks(parents)
    chunk_report(parents, children)

    if args.dry_run:
        print("\n[dry-run] nessuna indicizzazione eseguita.")
        return

    from rag.vector_store import build_vector_store
    build_vector_store(parents, children, force_rebuild=args.force)
    print("\nIndice pronto.")


if __name__ == "__main__":
    main()
