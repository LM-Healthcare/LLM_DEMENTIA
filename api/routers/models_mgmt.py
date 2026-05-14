from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import ModelInfo, PullModelRequest
from llm.ollama_client import is_ollama_running, list_models, pull_model

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/status")
def ollama_status():
    running = is_ollama_running()
    return {"running": running}


@router.get("/", response_model=list[ModelInfo])
def get_models():
    if not is_ollama_running():
        raise HTTPException(503, "Ollama non raggiungibile su localhost:11434")
    raw = list_models()
    result = []
    for m in raw:
        details = m.get("details", {})
        result.append(ModelInfo(
            name=m.get("name", ""),
            size=m.get("size"),
            modified_at=str(m.get("modified_at", "")),
            details=details,
        ))
    return result


@router.post("/pull")
def pull_model_endpoint(req: PullModelRequest):
    if not is_ollama_running():
        raise HTTPException(503, "Ollama non raggiungibile")

    def _stream():
        for status in pull_model(req.model_name):
            yield f"data: {status}\n\n"
        yield "data: done\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
