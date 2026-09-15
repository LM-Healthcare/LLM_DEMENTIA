"""
Gestione del Vector Store ChromaDB con embedding via Ollama.

Strategia:
  - Collezione 'parent_chunks': contesto citabile restituito al modello
  - Collezione 'child_chunks': unità di retrieval semantico
  - Il retrieval avviene sui child; il contesto restituito sono i parent
  - Distanza cosine (appropriata per embedding normalizzati)
  - Persistenza su disco in CHROMA_DB_PATH
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config.settings import CHROMA_DB_PATH, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

PARENT_COLLECTION = "parent_chunks"
CHILD_COLLECTION = "child_chunks"
CHROMA_BATCH_SIZE = 2000

_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def _open_collection(name: str) -> Chroma:
    return Chroma(
        collection_name=name,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_DB_PATH,
        collection_metadata=_COLLECTION_METADATA,
    )


def build_vector_store(
    parent_chunks: list[Document],
    child_chunks: list[Document],
    force_rebuild: bool = False,
) -> tuple[Chroma, Chroma]:
    """
    Costruisce (o ricarica) le due collezioni ChromaDB.
    Con force_rebuild=True l'indice su disco viene rimosso e ricostruito: necessario
    quando cambia il modello di embedding (la dimensione dei vettori non coincide).
    """
    if force_rebuild:
        _reset_store()

    child_store = _open_collection(CHILD_COLLECTION)
    parent_store = _open_collection(PARENT_COLLECTION)

    if force_rebuild or _count(child_store) == 0:
        print(f"[VectorStore] Indicizzazione parent ({len(parent_chunks)} chunks)...")
        _add_batched(parent_store, parent_chunks)
        print(f"[VectorStore] Indicizzazione child ({len(child_chunks)} chunks)...")
        _add_batched(child_store, child_chunks)
        print(f"[VectorStore] Build completato (embedding: {OLLAMA_EMBED_MODEL}).")
    else:
        print(f"[VectorStore] Store esistente caricato ({_count(child_store)} child chunks).")

    return parent_store, child_store


def _reset_store() -> None:
    """
    Azzera l'indice esistente.

    Le collezioni vengono eliminate tramite l'API di Chroma: la sola rimozione
    della cartella può fallire in silenzio (file lock su Windows, sincronizzazione
    OneDrive) lasciando in vita collezioni con la vecchia dimensione di embedding,
    che poi rifiutano i nuovi vettori.
    """
    path = Path(CHROMA_DB_PATH)
    if not path.exists():
        return

    print(f"[VectorStore] Azzeramento indice esistente: {path}")
    try:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        for name in (PARENT_COLLECTION, CHILD_COLLECTION):
            try:
                client.delete_collection(name)
                print(f"[VectorStore]   collezione '{name}' eliminata")
            except Exception:
                pass
        del client
    except BaseException as e:
        print(f"[VectorStore]   API Chroma non utilizzabile ({type(e).__name__}), procedo su file")

    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        print("[VectorStore]   cartella non rimovibile (file in uso): collezioni comunque azzerate")


def load_existing_store() -> tuple[Chroma | None, Chroma | None]:
    """
    Carica uno store esistente senza ricostruirlo.
    Cattura BaseException perché i binding Rust di ChromaDB sollevano PanicException
    (che non deriva da Exception) quando l'indice è stato scritto da un'altra versione.
    """
    try:
        child_store = _open_collection(CHILD_COLLECTION)
        parent_store = _open_collection(PARENT_COLLECTION)
        if _count(child_store) == 0:
            return None, None
        return parent_store, child_store
    except BaseException as e:
        print(f"[VectorStore] Impossibile caricare lo store esistente: {type(e).__name__}: {e}")
        return None, None


def _add_batched(store: Chroma, docs: list[Document]) -> None:
    """
    Inserisce i documenti a blocchi, usando chunk_id come id stabile:
    una reindicizzazione sovrascrive invece di duplicare.
    """
    for i in range(0, len(docs), CHROMA_BATCH_SIZE):
        batch = docs[i : i + CHROMA_BATCH_SIZE]
        store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        print(f"[VectorStore]   {min(i + CHROMA_BATCH_SIZE, len(docs))}/{len(docs)} indicizzati")


def _count(store: Chroma) -> int:
    try:
        return store._collection.count()
    except BaseException:
        return 0


def fetch_parents_by_ids(parent_store: Chroma, chunk_ids: list[str]) -> dict[str, Document]:
    """
    Recupera i parent chunk richiesti filtrando lato database.
    Evita la scansione completa della collezione a ogni retrieval.
    """
    if not chunk_ids:
        return {}
    try:
        res = parent_store.get(ids=chunk_ids, include=["documents", "metadatas"])
    except BaseException as e:
        print(f"[VectorStore] Errore fetch parent: {e}")
        return {}

    out: dict[str, Document] = {}
    for doc, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
        cid = (meta or {}).get("chunk_id")
        if cid:
            out[cid] = Document(page_content=doc, metadata=meta)
    return out


def iter_child_corpus(child_store: Chroma) -> list[Document]:
    """Restituisce tutti i child chunk: serve a costruire l'indice BM25."""
    try:
        res = child_store.get(include=["documents", "metadatas"])
    except BaseException as e:
        print(f"[VectorStore] Errore lettura corpus child: {e}")
        return []
    return [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(res.get("documents") or [], res.get("metadatas") or [])
    ]


def get_store_stats() -> dict:
    parent_store, child_store = load_existing_store()
    if parent_store is None:
        return {"status": "not_built", "parent_chunks": 0, "child_chunks": 0,
                "embedding_model": OLLAMA_EMBED_MODEL}
    return {
        "status": "ready",
        "parent_chunks": _count(parent_store),
        "child_chunks": _count(child_store),
        "embedding_model": OLLAMA_EMBED_MODEL,
    }
