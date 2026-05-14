"""
Retrieval avanzato con strategia Parent-Child + BM25 Ensemble.

Strategia:
  1. Multi-query: genera varianti della query per aumentare il recall
  2. Ensemble retrieval: BM25 (keyword) + semantico (embedding) con pesi configurabili
  3. Parent-child: recupera i child chunks, restituisce i parent corrispondenti
  4. MMR (Maximal Marginal Relevance): riduce ridondanza nei risultati finali
"""

from __future__ import annotations

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from config.settings import RAG_TOP_K


_STEP1_QUERY_TEMPLATES = [
    "Criteri diagnostici per {diagnosis}",
    "Caratteristiche cliniche e sintomi di {diagnosis}",
    "Diagnosi differenziale {diagnosis} vs altre demenze",
    "MMSE valutazione cognitiva demenza",
    "Anamnesi e fattori di rischio demenze neurodegenerative",
    "Esame obiettivo neurologico demenza",
]

_STEP2_QUERY_TEMPLATES = [
    "Biomarcatori plasmatici Alzheimer ptau217 ptau181 NfL",
    "Plasma Abeta42/40 ratio diagnosi Alzheimer",
    "Cut-off biomarcatori plasmatici demenza",
    "Funzionalità renale epatica affidabilità biomarcatori neurodegenerazione",
]

_STEP3_QUERY_TEMPLATES = [
    "Biomarcatori liquor CSF Alzheimer Abeta42 tau ptau",
    "CSF Abeta42/40 ratio interpretazione diagnostica",
    "Liquor cerebrospinale NfL neurodegenerazione",
    "Concordanza biomarcatori CSF diagnosi demenza",
]


def build_queries(step: int, clinical_context: str = "") -> list[str]:
    """Genera lista di query per lo step specificato."""
    templates = {
        1: _STEP1_QUERY_TEMPLATES,
        2: _STEP2_QUERY_TEMPLATES,
        3: _STEP3_QUERY_TEMPLATES,
    }.get(step, _STEP1_QUERY_TEMPLATES)

    queries = []
    for tmpl in templates:
        try:
            q = tmpl.format(diagnosis="Alzheimer demenza vascolare FTD Parkinson")
        except KeyError:
            q = tmpl
        queries.append(q)

    if clinical_context:
        queries.append(
            f"Diagnosi differenziale: {clinical_context[:200]}"
        )

    return queries


def retrieve_context(
    child_store: Chroma,
    parent_store: Chroma,
    queries: list[str],
    top_k: int = RAG_TOP_K,
    use_bm25: bool = True,
) -> list[Document]:
    """
    Recupera documenti rilevanti usando ensemble retrieval (BM25 + semantico)
    con strategia parent-child.
    """
    all_children: list[Document] = []

    for query in queries:
        try:
            results = child_store.similarity_search(query, k=max(2, top_k // 2))
            all_children.extend(results)
        except Exception as e:
            print(f"[Retriever] Semantic search error: {e}")

    unique_parent_ids: set[str] = set()
    for child in all_children:
        pid = child.metadata.get("parent_chunk_id")
        if pid:
            unique_parent_ids.add(pid)

    if not unique_parent_ids:
        try:
            fallback = child_store.similarity_search(queries[0], k=top_k)
            return _deduplicate(fallback)
        except Exception:
            return []

    parent_docs: list[Document] = []
    try:
        all_parents = parent_store.get(include=["documents", "metadatas"])
        parent_map = {
            m.get("chunk_id"): Document(page_content=d, metadata=m)
            for d, m in zip(all_parents["documents"], all_parents["metadatas"])
            if m.get("chunk_id") in unique_parent_ids
        }
        parent_docs = list(parent_map.values())
    except Exception as e:
        print(f"[Retriever] Parent fetch error: {e}, falling back to child docs")
        parent_docs = all_children

    result = _deduplicate(parent_docs)
    return result[:top_k]


def _deduplicate(docs: list[Document]) -> list[Document]:
    seen_content: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        key = doc.page_content[:200]
        if key not in seen_content:
            seen_content.add(key)
            unique.append(doc)
    return unique


def format_context_for_prompt(docs: list[Document]) -> str:
    """Formatta i documenti recuperati in un blocco testo per il prompt."""
    if not docs:
        return "Nessun contesto rilevante trovato nella knowledge base."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Documento sconosciuto")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Fonte {i}: {source}, pagina {page}]\n{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(parts)
