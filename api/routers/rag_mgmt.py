from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import RAGStatus
from api.state import get_app_state
from config.settings import DOCS_FOLDER, RAG_CORPORA
from rag.document_processor import load_all_documents, create_parent_chunks, create_child_chunks
from rag.retriever import reset_lexical_index
from rag.vector_store import build_vector_store, load_existing_store, get_store_stats

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/status", response_model=RAGStatus)
def rag_status():
    state = get_app_state()
    stats = get_store_stats()
    root = Path(DOCS_FOLDER)
    docs_info = [
        {
            "name": pdf.name,
            "folder": str(pdf.parent.relative_to(root)) or root.name,
            "size_kb": round(pdf.stat().st_size / 1024),
        }
        for pdf in sorted(root.rglob("*.pdf"))
    ] if root.exists() else []
    return RAGStatus(
        ready=state.rag_ready,
        parent_chunks=stats.get("parent_chunks", 0),
        child_chunks=stats.get("child_chunks", 0),
        embedding_model=stats.get("embedding_model", ""),
        corpora={name: list(sources) for name, sources in RAG_CORPORA.items()},
        documents=docs_info,
    )


@router.post("/build")
async def build_rag(background_tasks: BackgroundTasks, force: bool = False):
    state = get_app_state()
    if state.rag_building:
        raise HTTPException(409, "Un build del RAG è già in corso")
    if state.rag_ready and not force:
        return {"message": "RAG già inizializzato", "ready": True}
    background_tasks.add_task(_build_rag_task, state, force)
    return {"message": "Build RAG avviato in background", "ready": False}


@router.post("/load")
async def load_rag():
    state = get_app_state()
    parent_store, child_store = load_existing_store()
    if parent_store is None:
        raise HTTPException(404, "Nessun vector store trovato. Esegui prima /rag/build")
    state.parent_store = parent_store
    state.child_store = child_store
    state.rag_ready = True
    reset_lexical_index()
    return {"message": "RAG caricato con successo", **get_store_stats()}


def _build_rag_sync(force: bool):
    docs = load_all_documents()
    parent_chunks = create_parent_chunks(docs)
    child_chunks = create_child_chunks(parent_chunks)
    return build_vector_store(parent_chunks, child_chunks, force_rebuild=force)


async def _build_rag_task(state, force: bool):
    state.rag_building = True
    try:
        parent_store, child_store = await asyncio.get_event_loop().run_in_executor(
            None, _build_rag_sync, force
        )
        reset_lexical_index()
        state.parent_store = parent_store
        state.child_store = child_store
        state.rag_ready = True
        print("[RAG] Build completato con successo")
    except BaseException as e:
        print(f"[RAG] Errore durante il build: {type(e).__name__}: {e}")
        state.rag_ready = False
    finally:
        state.rag_building = False
