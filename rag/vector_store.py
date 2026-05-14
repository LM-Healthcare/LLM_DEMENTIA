"""
Gestione del Vector Store ChromaDB con embedding via Ollama.

Strategia:
  - Collezione 'parent_chunks': chunk grandi (800 token) per il contesto LLM
  - Collezione 'child_chunks': chunk piccoli (256 token) per il retrieval semantico
  - Il retrieval avviene sui child chunks; il contesto restituito sono i parent chunks
  - Persistenza su disco in CHROMA_DB_PATH
"""

from __future__ import annotations

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config.settings import CHROMA_DB_PATH, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

PARENT_COLLECTION = "parent_chunks"
CHILD_COLLECTION = "child_chunks"
CHROMA_BATCH_SIZE = 5000


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def build_vector_store(
    parent_chunks: list[Document],
    child_chunks: list[Document],
    force_rebuild: bool = False,
) -> tuple[Chroma, Chroma]:
    """
    Costruisce (o ricarica) le due collezioni ChromaDB.
    Restituisce (parent_store, child_store).
    """
    embeddings = _get_embeddings()

    child_store = Chroma(
        collection_name=CHILD_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    parent_store = Chroma(
        collection_name=PARENT_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    if force_rebuild or _is_empty(child_store):
        print(f"[VectorStore] Building child collection ({len(child_chunks)} chunks)...")
        _add_batched(child_store, child_chunks)
        print(f"[VectorStore] Building parent collection ({len(parent_chunks)} chunks)...")
        _add_batched(parent_store, parent_chunks)
        print("[VectorStore] Build complete.")
    else:
        n = child_store._collection.count()
        print(f"[VectorStore] Loaded existing store ({n} child chunks).")

    return parent_store, child_store


def load_existing_store() -> tuple[Chroma | None, Chroma | None]:
    """Carica uno store esistente senza ricostruire."""
    try:
        embeddings = _get_embeddings()
        child_store = Chroma(
            collection_name=CHILD_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
        parent_store = Chroma(
            collection_name=PARENT_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
        if _is_empty(child_store):
            return None, None
        return parent_store, child_store
    except Exception as e:
        print(f"[VectorStore] Could not load existing store: {e}")
        return None, None


def _add_batched(store: Chroma, docs: list[Document]) -> None:
    """Inserisce documenti in batch per rispettare il limite di ChromaDB (max ~5461)."""
    for i in range(0, len(docs), CHROMA_BATCH_SIZE):
        batch = docs[i : i + CHROMA_BATCH_SIZE]
        store.add_documents(batch)
        print(f"[VectorStore]   ... {min(i + CHROMA_BATCH_SIZE, len(docs))}/{len(docs)} chunks inseriti")


def _is_empty(store: Chroma) -> bool:
    try:
        return store._collection.count() == 0
    except Exception:
        return True


def is_store_ready() -> bool:
    """Controlla se il vector store è già stato costruito."""
    import os
    from pathlib import Path
    db_path = Path(CHROMA_DB_PATH)
    if not db_path.exists():
        return False
    chroma_files = list(db_path.rglob("*.sqlite3")) + list(db_path.rglob("*.bin"))
    return len(chroma_files) > 0


def get_store_stats() -> dict:
    parent_store, child_store = load_existing_store()
    if parent_store is None:
        return {"status": "not_built", "parent_chunks": 0, "child_chunks": 0}
    try:
        return {
            "status": "ready",
            "parent_chunks": parent_store._collection.count(),
            "child_chunks": child_store._collection.count(),
        }
    except Exception:
        return {"status": "error", "parent_chunks": 0, "child_chunks": 0}
