"""
Stato condiviso dell'applicazione FastAPI.
Contiene riferimenti ai vector stores RAG e allo stato del batch.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class AppState:
    parent_store: Optional[Any] = None
    child_store: Optional[Any] = None
    rag_ready: bool = False
    batch_running: bool = False
    batch_progress: dict = field(default_factory=dict)
    last_evaluation: Optional[dict] = None


_state = AppState()


def get_app_state() -> AppState:
    return _state
