"""
Caricamento e chunking avanzato dei documenti PDF.

Strategia di chunking: Parent-Child
  - Small chunks (128 token, overlap 20): usati per il retrieval semantico
  - Parent chunks (800 token, overlap 150): restituiti come contesto al LLM
  - Metadati: source, page, doc_type, language
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from config.settings import DOCS_FOLDER_1, DOCS_FOLDER_2, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP


_DOC_TYPE_MAP = {
    "Continuum Demenze": "continuum_review",
    "Documenti_": "reference_document",
}


def _doc_type_from_path(path: Path) -> str:
    for part in path.parts:
        for folder_key, doc_type in _DOC_TYPE_MAP.items():
            if folder_key in part:
                return doc_type
    return "unknown"


def _doc_id(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:8]


def load_all_documents() -> list[Document]:
    """Carica tutti i PDF dalle due cartelle documenti."""
    folders = [Path(DOCS_FOLDER_1), Path(DOCS_FOLDER_2)]
    all_docs: list[Document] = []

    for folder in folders:
        if not folder.exists():
            print(f"[WARNING] Cartella non trovata: {folder}")
            continue

        pdf_files = list(folder.glob("*.pdf"))
        print(f"[RAG] Cartella: {folder.name} — {len(pdf_files)} PDF trovati")

        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                doc_type = _doc_type_from_path(pdf_path)
                did = _doc_id(pdf_path)

                for page in pages:
                    page.metadata.update({
                        "source": pdf_path.name,
                        "source_path": str(pdf_path),
                        "doc_type": doc_type,
                        "doc_id": did,
                        "page": page.metadata.get("page", 0),
                    })
                all_docs.extend(pages)
                print(f"  ✓ {pdf_path.name}: {len(pages)} pagine caricate")
            except Exception as e:
                print(f"  ✗ Errore caricamento {pdf_path.name}: {e}")

    print(f"[RAG] Totale: {len(all_docs)} pagine caricate")
    return all_docs


def create_parent_chunks(documents: list[Document]) -> list[Document]:
    """Crea i chunk 'parent' (grandi) per il contesto LLM."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_type"] = "parent"
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('doc_id', 'unk')}_{i:04d}"
    return chunks


def create_child_chunks(parent_chunks: list[Document]) -> list[Document]:
    """
    Crea i chunk 'child' (piccoli) a partire dai parent.
    Ogni child mantiene il riferimento al parent_chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_docs: list[Document] = []
    for parent in parent_chunks:
        children = splitter.split_documents([parent])
        for j, child in enumerate(children):
            child.metadata["chunk_type"] = "child"
            child.metadata["parent_chunk_id"] = parent.metadata.get("chunk_id")
            child.metadata["child_id"] = f"{parent.metadata.get('chunk_id', 'unk')}_c{j:02d}"
        child_docs.extend(children)
    return child_docs


def get_document_stats(documents: list[Document]) -> dict:
    from collections import Counter
    sources = Counter(d.metadata.get("source", "?") for d in documents)
    doc_types = Counter(d.metadata.get("doc_type", "?") for d in documents)
    return {
        "total_pages": len(documents),
        "by_source": dict(sources),
        "by_type": dict(doc_types),
    }
