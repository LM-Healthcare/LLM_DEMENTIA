from __future__ import annotations

import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import RAGStatus
from api.state import get_app_state
from rag.document_processor import load_all_documents, create_parent_chunks, create_child_chunks, get_document_stats
from rag.vector_store import build_vector_store, load_existing_store, get_store_stats

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/status", response_model=RAGStatus)
def rag_status():
    state = get_app_state()
    stats = get_store_stats()
    docs_info = []
    if state.rag_ready and state.child_store:
        try:
            from config.settings import DOCS_FOLDER_1, DOCS_FOLDER_2
            from pathlib import Path
            for folder in [DOCS_FOLDER_1, DOCS_FOLDER_2]:
                for pdf in Path(folder).glob("*.pdf"):
                    docs_info.append({"name": pdf.name, "folder": Path(folder).name, "size_kb": round(pdf.stat().st_size / 1024)})
        except Exception:
            pass
    return RAGStatus(
        ready=state.rag_ready,
        parent_chunks=stats.get("parent_chunks", 0),
        child_chunks=stats.get("child_chunks", 0),
        documents=docs_info,
    )


@router.post("/build")
async def build_rag(background_tasks: BackgroundTasks, force: bool = False):
    state = get_app_state()
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
    stats = get_store_stats()
    return {"message": "RAG caricato con successo", **stats}


async def _build_rag_task(state, force: bool):
    try:
        docs = await asyncio.get_event_loop().run_in_executor(None, load_all_documents)
        parent_chunks = await asyncio.get_event_loop().run_in_executor(None, create_parent_chunks, docs)
        child_chunks = await asyncio.get_event_loop().run_in_executor(None, create_child_chunks, parent_chunks)
        parent_store, child_store = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: build_vector_store(parent_chunks, child_chunks, force_rebuild=force)
        )
        state.parent_store = parent_store
        state.child_store = child_store
        state.rag_ready = True
        print("[RAG] Build completato con successo")
    except Exception as e:
        print(f"[RAG] Errore durante il build: {e}")
        state.rag_ready = False
