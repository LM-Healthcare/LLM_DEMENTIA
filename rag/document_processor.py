"""
Caricamento, normalizzazione e chunking dei documenti della knowledge base.

Pipeline di ingestion:
  1. Scansione ricorsiva dei PDF, deduplicati per hash del contenuto
  2. Normalizzazione del testo estratto (legature, sillabazione, spaziatura)
  3. Rimozione del boilerplate (copyright, numeri di pagina, URL, disclosure)
  4. Concatenazione delle pagine per documento con mappa degli offset
     (evita che un titolo a fine pagina diventi un frammento isolato)
  5. Rilevamento delle intestazioni di sezione, propagate come metadato
  6. Chunking parent-child con fusione dei frammenti troppo corti

Strategia parent-child:
  - child (RAG_CHILD_CHUNK_SIZE): usati per il retrieval, alta precisione
  - parent (RAG_PARENT_CHUNK_SIZE): contesto restituito al modello e citato
    per esteso, garantito sopra RAG_MIN_CHUNK_CHARS caratteri
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from config.settings import (
    DOCS_FOLDER,
    RAG_PARENT_CHUNK_SIZE,
    RAG_PARENT_CHUNK_OVERLAP,
    RAG_CHILD_CHUNK_SIZE,
    RAG_CHILD_CHUNK_OVERLAP,
    RAG_MIN_CHUNK_CHARS,
)

_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

# Soglia sotto la quale un chunk viene scartato anche dopo la fusione.
_HARD_MIN_CHARS = 120

# Legature e caratteri tipografici che PyPDF estrae come codepoint esotici.
_CHAR_FIXES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00ad": "",
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\ufeff": "",
}

_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*\d{1,4}\s*$",                      # numero di pagina isolato
        r"copyright\s*(©|\(c\))",
        r"all rights reserved",
        r"tutti i diritti riservati",
        r"unauthorized reproduction",
        r"downloaded from",
        r"^\s*https?://",
        r"^\s*www\.",
        r"^\s*doi:",
        r"^\s*isbn",
        r"relationship disclosure",
        r"^\s*(e-?mail|correspondence)\s*:",
        r"^[\s\W\d]*$",                          # solo simboli/spazi/cifre
    )
]

_HEADING_PREFIX = re.compile(
    r"^\s*(chapter|capitolo|part|parte|section|sezione|appendix|appendice)\b",
    re.IGNORECASE,
)

# Didascalie di figure e tabelle: vanno trattate come righe a sé (non fuse nei
# paragrafi) ma non sono titoli di sezione utilizzabili in una citazione.
_CAPTION = re.compile(
    r"^\s*(fig|figure|figura|tab|table|tabella|box)\.?\s*[0-9ivxlc]", re.IGNORECASE
)
_NUMBERED_HEADING = re.compile(r"^\s*\d+(\.\d+)*\s+\S")

# Intestazione di capitolo priva di titolo: utile come metadato, inutile come sezione.
_BARE_CHAPTER = re.compile(
    r"^\s*(chapter|capitolo|part|parte)\s+([0-9]+|[ivxlc]+)\s*[.:]?\s*$", re.IGNORECASE
)

# Riferimenti bibliografici: righe brevi e capitalizzate che l'euristica dei
# titoli scambierebbe per intestazioni (es. "Traber, 2008").
_CITATION_LIKE = re.compile(r"(et al\.?|\b(19|20)\d{2}\b)", re.IGNORECASE)

# Numero di righe in testa/coda a una pagina esaminate per trovare header e footer.
_RUNNING_LINE_WINDOW = 3


def _doc_id(path: Path, content_hash: str) -> str:
    return content_hash[:10]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    """Normalizza il testo estratto dal PDF preservando la struttura dei paragrafi."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    for bad, good in _CHAR_FIXES.items():
        text = text.replace(bad, good)

    # Ricompone le parole spezzate dalla sillabazione a fine riga ("neurode-\ngeneration").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Ripulisce i richiami bibliografici svuotati dall'estrazione: "( Seshadri et al., )".
    text = re.sub(r",\s*\)", ")", text)
    text = re.sub(r"\(\s*\)", "", text)

    lines = [ln for ln in text.split("\n") if not _is_boilerplate(ln)]

    # Riunisce le righe dello stesso paragrafo: PyPDF spezza a ogni riga stampata,
    # e senza questo passaggio il chunking taglia a metà frase.
    out: list[str] = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if out and out[-1] and not _looks_like_heading(out[-1]) and not _looks_like_heading(line):
            if not re.search(r"[.!?:;]$", out[-1]):
                out[-1] = f"{out[-1]} {line}"
                continue
        out.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _is_boilerplate(line: str) -> bool:
    return any(p.search(line) for p in _BOILERPLATE_PATTERNS)


def _looks_like_heading(line: str) -> bool:
    """Euristica per riconoscere un'intestazione di sezione."""
    s = line.strip()
    if not (3 <= len(s) <= 90):
        return False
    if s.endswith((".", ",", ";")):
        return False
    if not re.search(r"[A-Za-zÀ-ÿ]", s):
        return False
    if _CITATION_LIKE.search(s):
        return False
    # Una riga che inizia in minuscolo o con un simbolo è la continuazione di un
    # paragrafo (es. ") One"), non un titolo.
    if not s[0].isalnum() or s[0].islower():
        return False
    if _CAPTION.match(s):
        return True
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ][\w'-]*", s)]
    if not words:
        return False
    if _HEADING_PREFIX.match(s):
        return True
    if _NUMBERED_HEADING.match(s):
        return len(words) <= 8
    if s.isupper() and len(words) >= 1:
        return True
    capitalized = sum(1 for w in words if w[0].isupper())
    return capitalized / len(words) >= 0.6 and len(words) <= 12


def load_all_documents(folder: str | Path = DOCS_FOLDER) -> list[Document]:
    """
    Carica ricorsivamente i PDF della knowledge base, una Document per pagina,
    con testo normalizzato. I file duplicati (stesso contenuto) sono ignorati.
    """
    root = Path(folder)
    if not root.exists():
        print(f"[RAG] Cartella knowledge base non trovata: {root}")
        return []

    pdf_files = sorted(root.rglob("*.pdf"))
    print(f"[RAG] Knowledge base: {root} — {len(pdf_files)} PDF trovati")

    all_docs: list[Document] = []
    seen_hashes: dict[str, Path] = {}

    for pdf_path in pdf_files:
        content_hash = _file_hash(pdf_path)
        if content_hash in seen_hashes:
            print(f"  ~ {pdf_path.name}: duplicato di {seen_hashes[content_hash].name}, ignorato")
            continue
        seen_hashes[content_hash] = pdf_path

        try:
            pages = PyPDFLoader(str(pdf_path)).load()
        except Exception as e:
            print(f"  x {pdf_path.name}: errore di caricamento — {e}")
            continue

        did = _doc_id(pdf_path, content_hash)
        doc_pages: list[Document] = []
        for page in pages:
            content = normalize_text(page.page_content)
            if len(content) < 40:
                continue
            page.page_content = content
            page.metadata = {
                "source": pdf_path.name,
                "source_path": str(pdf_path.relative_to(root)),
                "doc_id": did,
                "page": int(page.metadata.get("page", 0)),
                "page_label": str(page.metadata.get("page_label", "")),
                "chapter": "",
            }
            doc_pages.append(page)

        _strip_running_lines(doc_pages)
        doc_pages = [p for p in doc_pages if len(p.page_content) >= 40]
        all_docs.extend(doc_pages)
        print(f"  + {pdf_path.name}: {len(doc_pages)}/{len(pages)} pagine con testo utile")

    print(f"[RAG] Totale: {len(all_docs)} pagine da {len(seen_hashes)} documenti")
    return all_docs


def _strip_running_lines(pages: list[Document]) -> None:
    """
    Rimuove le intestazioni e i piè di pagina ricorrenti di un documento.

    Sono righe che compaiono in testa o in coda a molte pagine (es. "Chapter 15",
    il titolo del libro): se restano nel testo spezzano i paragrafi e vengono
    scambiate per titoli di sezione. Quando indicano il capitolo, il valore viene
    conservato nel metadato 'chapter' invece di essere buttato.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.page_content.split("\n") if ln.strip()]
        window = lines[:_RUNNING_LINE_WINDOW] + lines[-_RUNNING_LINE_WINDOW:]
        counts.update({ln for ln in window if len(ln) <= 100})

    threshold = max(5, int(len(pages) * 0.02))
    running = {ln for ln, n in counts.items() if n >= threshold}
    if not running:
        return

    chapter_lines = {ln for ln in running if _HEADING_PREFIX.match(ln)}
    print(f"  · {len(running)} righe ricorrenti rimosse (header/footer)")

    for page in pages:
        kept: list[str] = []
        chapter = ""
        for line in page.page_content.split("\n"):
            stripped = line.strip()
            # I riferimenti di capitolo vengono rimossi in qualunque posizione:
            # l'estrazione PDF li deposita anche in mezzo ai paragrafi.
            if stripped in running or _BARE_CHAPTER.match(stripped):
                if stripped in chapter_lines or _BARE_CHAPTER.match(stripped):
                    chapter = stripped
                continue
            kept.append(line)
        page.page_content = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        page.metadata["chapter"] = chapter


def _concat_pages(pages: list[Document]) -> tuple[str, list[tuple[int, int, int, str]]]:
    """
    Concatena le pagine di un documento in un testo continuo.
    Restituisce (full_text, spans) con spans = [(start, end, page, page_label)].
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int, str]] = []
    cursor = 0
    for page in sorted(pages, key=lambda p: p.metadata.get("page", 0)):
        body = page.page_content
        parts.append(body)
        spans.append((cursor, cursor + len(body), page.metadata.get("page", 0),
                      page.metadata.get("page_label", "")))
        cursor += len(body) + 2
    return "\n\n".join(parts), spans


def _heading_index(full_text: str) -> list[tuple[int, str]]:
    """
    Offset e testo di ogni riga utilizzabile come titolo di sezione.
    Sono escluse le intestazioni di solo numero di capitolo (non dicono nulla,
    la posizione è già nei metadati) e le didascalie di figure e tabelle.
    """
    index: list[tuple[int, str]] = []
    offset = 0
    for line in full_text.split("\n"):
        stripped = line.strip()
        if (_looks_like_heading(line)
                and not _BARE_CHAPTER.match(stripped)
                and not _CAPTION.match(stripped)):
            index.append((offset, stripped))
        offset += len(line) + 1
    return index


def _locate(full_text: str, piece: str, cursor: int) -> int:
    start = full_text.find(piece, max(0, cursor - RAG_PARENT_CHUNK_OVERLAP - 100))
    return start if start >= 0 else cursor


def _merge_short(pieces: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    Fonde i chunk sotto RAG_MIN_CHUNK_CHARS con il vicino.
    È il passaggio che impedisce a un titolo di sezione di diventare una fonte a sé.
    """
    max_merged = int(RAG_PARENT_CHUNK_SIZE * 1.6)
    merged: list[tuple[str, int]] = []
    pending: tuple[str, int] | None = None

    for text, start in pieces:
        if pending:
            text, start = f"{pending[0]}\n{text}", pending[1]
            pending = None
        if len(text) >= RAG_MIN_CHUNK_CHARS:
            merged.append((text, start))
        elif merged and len(merged[-1][0]) + len(text) <= max_merged:
            merged[-1] = (f"{merged[-1][0]}\n{text}", merged[-1][1])
        else:
            pending = (text, start)

    if pending:
        if merged and len(merged[-1][0]) + len(pending[0]) <= max_merged:
            merged[-1] = (f"{merged[-1][0]}\n{pending[0]}", merged[-1][1])
        else:
            merged.append(pending)

    return [
        (t, s) for t, s in merged
        if len(t) >= _HARD_MIN_CHARS and re.search(r"[A-Za-zÀ-ÿ]{3}", t)
    ]


def create_parent_chunks(documents: list[Document]) -> list[Document]:
    """
    Crea i chunk 'parent' (contesto citabile) lavorando sul testo continuo di
    ogni documento, con sezione e intervallo di pagine nei metadati.
    """
    by_doc: dict[str, list[Document]] = {}
    for doc in documents:
        by_doc.setdefault(doc.metadata["doc_id"], []).append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_PARENT_CHUNK_SIZE,
        chunk_overlap=RAG_PARENT_CHUNK_OVERLAP,
        separators=_SPLIT_SEPARATORS,
    )

    parents: list[Document] = []
    for doc_id, pages in by_doc.items():
        base = pages[0].metadata
        full_text, spans = _concat_pages(pages)
        headings = _heading_index(full_text)
        chapter_by_page = {
            p.metadata["page"]: p.metadata.get("chapter", "") for p in pages
        }

        located: list[tuple[str, int]] = []
        cursor = 0
        for piece in splitter.split_text(full_text):
            start = _locate(full_text, piece, cursor)
            located.append((piece, start))
            cursor = start + max(1, len(piece) - RAG_PARENT_CHUNK_OVERLAP)

        for i, (text, start) in enumerate(_merge_short(located)):
            end = start + len(text)
            covered = [sp for sp in spans if sp[0] < end and sp[1] > start] or [
                (start, end, base["page"], base.get("page_label", ""))
            ]
            pages_covered = [(p, lbl) for _, _, p, lbl in covered]
            section = next(
                (h for off, h in reversed(headings) if off <= start), ""
            )
            chapter = next(
                (chapter_by_page.get(p, "") for p, _ in pages_covered if chapter_by_page.get(p)),
                "",
            )
            parents.append(Document(
                page_content=text,
                metadata={
                    "source": base["source"],
                    "source_path": base["source_path"],
                    "doc_id": doc_id,
                    "chunk_type": "parent",
                    "chunk_id": f"{doc_id}_p{i:05d}",
                    "section": section,
                    "chapter": chapter,
                    "page": pages_covered[0][0],
                    "page_end": pages_covered[-1][0],
                    "page_label": pages_covered[0][1],
                    "page_label_end": pages_covered[-1][1],
                    "n_chars": len(text),
                },
            ))

    print(f"[RAG] {len(parents)} parent chunks creati")
    return parents


def create_child_chunks(parent_chunks: list[Document]) -> list[Document]:
    """
    Crea i chunk 'child' (unità di retrieval) da ogni parent.
    Il testo del child è prefissato dalla sezione di appartenenza: dà all'embedding
    il contesto gerarchico che il frammento da solo non avrebbe.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHILD_CHUNK_SIZE,
        chunk_overlap=RAG_CHILD_CHUNK_OVERLAP,
        separators=_SPLIT_SEPARATORS,
    )

    children: list[Document] = []
    for parent in parent_chunks:
        meta = parent.metadata
        header = " — ".join(
            part for part in (meta["source"], meta.get("chapter"), meta.get("section")) if part
        )
        for j, piece in enumerate(splitter.split_text(parent.page_content)):
            if len(piece) < 60 or not re.search(r"[A-Za-zÀ-ÿ]{3}", piece):
                continue
            children.append(Document(
                page_content=f"[{header}]\n{piece}" if header else piece,
                metadata={
                    **{k: v for k, v in meta.items() if k != "chunk_type"},
                    "chunk_type": "child",
                    "parent_chunk_id": meta["chunk_id"],
                    "chunk_id": f"{meta['chunk_id']}_c{j:03d}",
                    "raw_text": piece,
                },
            ))

    print(f"[RAG] {len(children)} child chunks creati")
    return children

