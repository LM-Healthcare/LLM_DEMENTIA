from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers import patients, analysis, models_mgmt, rag_mgmt, results
from api.state import get_app_state
from rag.retriever import reset_lexical_index
from rag.vector_store import get_store_stats, load_existing_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_app_state()
    print("[Startup] Caricamento indice RAG...")
    parent_store, child_store = load_existing_store()
    if parent_store is not None:
        state.parent_store = parent_store
        state.child_store = child_store
        state.rag_ready = True
        reset_lexical_index()
        stats = get_store_stats()
        print(f"[Startup] RAG pronto — {stats['parent_chunks']} passaggi, "
              f"{stats['child_chunks']} unità di retrieval "
              f"(embedding: {stats['embedding_model']})")
    else:
        print("[Startup] Nessun indice RAG. Costruiscilo con "
              "'python scripts/build_rag.py --force' o POST /api/rag/build")
    yield
    print("[Shutdown] Cleanup completato")


app = FastAPI(
    title="LLM Dementia Diagnosis API",
    description="Sistema di supporto alla diagnosi differenziale delle demenze tramite LLM/SLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(models_mgmt.router, prefix="/api")
app.include_router(rag_mgmt.router, prefix="/api")
app.include_router(results.router, prefix="/api")

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Una rotta /api/* inesistente deve restituire un 404 JSON: servire la SPA
        # farebbe fallire il client con un errore di parsing incomprensibile.
        if full_path.startswith("api/"):
            raise HTTPException(404, f"Endpoint /{full_path} non trovato")
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
