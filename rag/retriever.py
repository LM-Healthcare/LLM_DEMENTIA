"""
Retrieval ibrido parent-child con fusione dei ranking.

Strategia:
  1. Multi-query bilingue: ogni concetto clinico è interrogato in italiano e in
     inglese, perché la knowledge base è in inglese e la cartella clinica in
     italiano (senza questo passaggio il retrieval cross-lingua perde le fonti)
  2. Retrieval ibrido per ogni query: semantico (embedding) + lessicale (BM25),
     indispensabile per i termini esatti come "p-tau217" o "A+T+N+"
  3. Reciprocal Rank Fusion (RRF) per fondere i ranking delle due modalità
  4. Aggregazione child → parent: il punteggio di un parent è la somma dei
     contributi RRF dei suoi child, quindi le fonti sono ORDINATE per rilevanza
  5. Diversificazione: tetto massimo di fonti dalla stessa pagina
  6. Le citazioni restituite contengono il passaggio integrale del parent più
     gli estratti esatti (child) che hanno prodotto il match
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from langchain.schema import Document
from langchain_chroma import Chroma

from config.settings import (
    DIAGNOSIS_LABELS,
    RAG_BM25_WEIGHT,
    RAG_CANDIDATES_PER_QUERY,
    RAG_MAX_PER_PAGE,
    RAG_MAX_PER_SOURCE,
    RAG_RRF_K,
    RAG_TOP_K,
)
from rag.vector_store import fetch_parents_by_ids, iter_child_corpus


# Termini di ricerca per diagnosi, in italiano e in inglese.
_DIAGNOSIS_TERMS: dict[str, tuple[str, str]] = {
    "AD": ("malattia di Alzheimer", "Alzheimer disease"),
    "AD-PPA": ("afasia progressiva primaria variante logopenica",
               "logopenic variant primary progressive aphasia"),
    "MIXED": ("demenza mista Alzheimer e vascolare", "mixed Alzheimer and vascular dementia"),
    "VAD": ("demenza vascolare", "vascular dementia vascular cognitive impairment"),
    "SCD": ("disturbo soggettivo di memoria", "subjective cognitive decline"),
    "LATE": ("encefalopatia TDP-43 correlata all'età",
             "limbic-predominant age-related TDP-43 encephalopathy"),
    "FTD": ("demenza frontotemporale", "frontotemporal dementia behavioural variant"),
    "PD": ("demenza associata a Parkinson spettro corpi di Lewy",
           "Parkinson disease dementia Lewy body disease spectrum PDD DLB"),
}

_STEP1_GENERAL = [
    ("diagnosi differenziale delle demenze", "differential diagnosis of dementia syndromes"),
    ("valutazione cognitiva MMSE punteggio",
     "Mini-Mental State Examination score cognitive screening interpretation"),
    ("esame obiettivo neurologico nel decadimento cognitivo",
     "neurological examination findings in cognitive impairment"),
    ("fattori di rischio vascolari e decadimento cognitivo",
     "vascular risk factors hypertension diabetes cognitive decline"),
    ("esordio e progressione dei sintomi cognitivi",
     "onset and progression of cognitive symptoms course"),
    ("farmaci che compromettono la cognizione",
     "medications causing cognitive impairment anticholinergic benzodiazepine"),
]

_STEP2_GENERAL = [
    ("biomarcatori plasmatici p-tau217 p-tau181 NfL",
     "plasma ptau217 ptau181 neurofilament light biomarkers Alzheimer"),
    ("rapporto plasmatico Abeta42/40", "plasma amyloid beta 42/40 ratio cut-off"),
    ("affidabilità dei biomarcatori e funzione renale ed epatica",
     "renal hepatic function confounders blood biomarker reliability eGFR"),
]

_STEP3_GENERAL = [
    ("biomarcatori liquorali Abeta42 tau fosforilata",
     "cerebrospinal fluid amyloid beta 42 total tau phosphorylated tau"),
    ("classificazione ATN amiloide tau neurodegenerazione",
     "ATN classification amyloid tau neurodegeneration biomarker profile"),
    ("interpretazione del rapporto Abeta42/40 liquorale",
     "CSF Abeta42/40 ratio interpretation diagnostic accuracy"),
]

_GENERAL_BY_STEP = {1: _STEP1_GENERAL, 2: _STEP2_GENERAL, 3: _STEP3_GENERAL}

_MAX_CLINICAL_QUERY_CHARS = 1200

# Se si aggiunge una diagnosi in DIAGNOSIS_LABELS senza i termini di ricerca
# corrispondenti, lo Step 1 non recupera nulla su quella diagnosi e la esclude
# di fatto dalla differenziale. Meglio accorgersene all'avvio.
_MISSING_QUERY_TERMS = set(DIAGNOSIS_LABELS) - set(_DIAGNOSIS_TERMS)
if _MISSING_QUERY_TERMS:
    print(f"[Retriever] ATTENZIONE: nessun termine di ricerca per {sorted(_MISSING_QUERY_TERMS)}")


def build_queries(step: int, clinical_context: str = "") -> list[str]:
    """
    Genera le query per lo step indicato: una coppia italiano/inglese per ogni
    concetto, più il quadro clinico del paziente quando disponibile.
    """
    queries: list[str] = []

    for it, en in _GENERAL_BY_STEP.get(step, _STEP1_GENERAL):
        queries.extend([it, en])

    if step == 1:
        for code, (it, en) in _DIAGNOSIS_TERMS.items():
            queries.append(f"criteri diagnostici {it}")
            queries.append(f"diagnostic criteria clinical features of {en}")

    if clinical_context and clinical_context.strip():
        queries.append(clinical_context.strip()[:_MAX_CLINICAL_QUERY_CHARS])

    seen: set[str] = set()
    return [q for q in queries if not (q.lower() in seen or seen.add(q.lower()))]


# ─── BM25 lessicale ───────────────────────────────────────────────────────────

_bm25_cache: dict | None = None


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("-", "").replace("/", " ")
    return [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1]


def reset_lexical_index() -> None:
    """Invalida l'indice BM25 in cache (da chiamare dopo una reindicizzazione)."""
    global _bm25_cache
    _bm25_cache = None


def _get_lexical_index(child_store: Chroma) -> dict | None:
    """Costruisce una sola volta l'indice BM25 sul corpus dei child chunk."""
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[Retriever] rank_bm25 non installato: retrieval solo semantico")
        return None

    corpus = iter_child_corpus(child_store)
    if not corpus:
        return None

    print(f"[Retriever] Costruzione indice BM25 su {len(corpus)} child chunks...")
    _bm25_cache = {
        "bm25": BM25Okapi([_tokenize(d.page_content) for d in corpus]),
        "docs": corpus,
    }
    return _bm25_cache


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_context(
    child_store: Chroma,
    parent_store: Chroma,
    queries: list[str],
    top_k: int = RAG_TOP_K,
    use_bm25: bool = True,
    allowed_sources: set[str] | None = None,
) -> list[Document]:
    """
    Esegue il retrieval ibrido multi-query e restituisce i parent chunk
    ordinati per rilevanza decrescente.

    Ogni Document restituito ha nei metadati:
      - retrieval_score: punteggio RRF aggregato
      - matched_quotes:  estratti esatti che hanno prodotto il match
    """
    if not queries:
        return []

    sem_weight = 1.0 - RAG_BM25_WEIGHT
    lexical = _get_lexical_index(child_store) if use_bm25 else None

    parent_scores: dict[str, float] = defaultdict(float)
    parent_quotes: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for query in queries:
        for rank, child in enumerate(_semantic_hits(child_store, query, allowed_sources)):
            _accumulate(parent_scores, parent_quotes, child, rank, sem_weight)
        if lexical:
            for rank, child in enumerate(_lexical_hits(lexical, query, allowed_sources)):
                _accumulate(parent_scores, parent_quotes, child, rank, RAG_BM25_WEIGHT)

    if not parent_scores:
        return []

    ranked_ids = sorted(parent_scores, key=parent_scores.get, reverse=True)
    parents = fetch_parents_by_ids(parent_store, ranked_ids[: top_k * 4])

    selected: list[Document] = []
    per_page: dict[tuple[str, int], int] = defaultdict(int)
    per_source: dict[str, int] = defaultdict(int)
    diversify_sources = allowed_sources is not None and len(allowed_sources) > 1

    for cid in ranked_ids:
        if len(selected) >= top_k:
            break
        doc = parents.get(cid)
        if doc is None:
            continue
        source = doc.metadata.get("source", "")
        key = (source, doc.metadata.get("page", -1))
        if per_page[key] >= RAG_MAX_PER_PAGE:
            continue
        if diversify_sources and per_source[source] >= RAG_MAX_PER_SOURCE:
            continue
        per_page[key] += 1
        per_source[source] += 1

        quotes = [q for _, q in sorted(parent_quotes[cid], key=lambda x: -x[0])]
        doc.metadata = {
            **doc.metadata,
            "retrieval_score": round(parent_scores[cid], 5),
            "matched_quotes": _unique_quotes(quotes)[:3],
        }
        selected.append(doc)

    return selected


def _accumulate(scores, quotes, child: Document, rank: int, weight: float) -> None:
    pid = child.metadata.get("parent_chunk_id")
    if not pid:
        return
    contribution = weight / (RAG_RRF_K + rank + 1)
    scores[pid] += contribution
    quote = child.metadata.get("raw_text") or child.page_content
    quotes[pid].append((contribution, quote.strip()))


def _semantic_hits(
    child_store: Chroma, query: str, allowed_sources: set[str] | None
) -> list[Document]:
    try:
        source_filter = (
            {"source": {"$in": sorted(allowed_sources)}} if allowed_sources else None
        )
        if source_filter:
            return child_store.similarity_search(
                query, k=RAG_CANDIDATES_PER_QUERY, filter=source_filter
            )
        return child_store.similarity_search(query, k=RAG_CANDIDATES_PER_QUERY)
    except BaseException as e:
        print(f"[Retriever] Errore ricerca semantica: {e}")
        return []


def _lexical_hits(
    lexical: dict, query: str, allowed_sources: set[str] | None
) -> list[Document]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = lexical["bm25"].get_scores(tokens)
    docs = lexical["docs"]
    eligible = (
        range(len(scores)) if not allowed_sources
        else (i for i, doc in enumerate(docs) if doc.metadata.get("source") in allowed_sources)
    )
    order = sorted(eligible, key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in order[:RAG_CANDIDATES_PER_QUERY] if scores[i] > 0]


def _unique_quotes(quotes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in quotes:
        key = q[:80].lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


# ─── Formattazione per il prompt e per l'audit ────────────────────────────────

def _display_page(label, index) -> str:
    """Pagina stampata sul documento se disponibile, altrimenti indice PDF 1-based."""
    if label not in (None, ""):
        return str(label)
    idx = _as_int(index)
    return str(idx + 1) if idx is not None else "?"


def _page_ref(meta: dict) -> str:
    start = _display_page(meta.get("page_label"), meta.get("page"))
    end = _display_page(meta.get("page_label_end"), meta.get("page_end"))
    return f"pp. {start}-{end}" if end != start else f"p. {start}"


def format_context_for_prompt(docs: list[Document]) -> str:
    """Formatta le fonti recuperate in un blocco testo per il prompt."""
    if not docs:
        return "Nessun contesto rilevante trovato nella knowledge base."

    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"[Fonte {i}] {meta.get('source', 'documento sconosciuto')}, {_page_ref(meta)}"
        if meta.get("section"):
            header += f" — sezione: {meta['section']}"
        parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


def extract_rag_sources(docs: list[Document]) -> list[dict]:
    """
    Metadati strutturati delle fonti per il layer di auditability.
    Il testo è restituito integrale: la citazione deve essere verificabile.
    """
    sources = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        sources.append({
            "index": i,
            "source": meta.get("source", "Documento sconosciuto"),
            "source_path": meta.get("source_path", ""),
            "section": meta.get("section", ""),
            "page": _pdf_page_1based(meta.get("page")),
            "page_end": _pdf_page_1based(meta.get("page_end")),
            "page_label": str(meta.get("page_label") or ""),
            "page_label_end": str(meta.get("page_label_end") or ""),
            "page_ref": _page_ref(meta),
            "doc_id": meta.get("doc_id", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "retrieval_score": meta.get("retrieval_score"),
            "matched_quotes": meta.get("matched_quotes", []),
            "text": doc.page_content.strip(),
        })
    return sources


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pdf_page_1based(value) -> int | None:
    idx = _as_int(value)
    return idx + 1 if idx is not None else None
