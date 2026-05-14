"""
Client per Ollama: gestione modelli, generazione testo e pulling.
"""

from __future__ import annotations

import json
import requests
from typing import Optional, Generator

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list[dict]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return r.json().get("models", [])
    except Exception as e:
        print(f"[Ollama] Error listing models: {e}")
        return []


def get_model_names() -> list[str]:
    return [m["name"] for m in list_models()]


def pull_model(model_name: str) -> Generator[str, None, None]:
    """
    Scarica un modello da Ollama. Yield di stringhe di stato.
    """
    url = f"{OLLAMA_BASE_URL}/api/pull"
    payload = {"name": model_name, "stream": True}
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        status = data.get("status", "")
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        if total > 0:
                            pct = int(completed / total * 100)
                            yield f"{status} ({pct}%)"
                        else:
                            yield status
                        if data.get("status") == "success":
                            return
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"Errore: {e}"


def generate(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    stream: bool = False,
) -> str:
    """
    Genera testo tramite Ollama /api/generate.
    temperature bassa (0.1) per risposta deterministica e clinicamente rigorosa.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.Timeout:
        return '{"error": "Timeout: il modello ha impiegato troppo tempo"}'
    except Exception as e:
        return f'{{"error": "Errore Ollama: {str(e)}"}}'


def generate_stream(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    """Genera testo in streaming (per UI real-time)."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            return
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"\n[Errore streaming: {e}]"


def parse_json_response(raw: str) -> dict:
    """
    Estrae il JSON dalla risposta del modello.
    Gestisce casi in cui il modello aggiunge testo prima/dopo il JSON.
    """
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        start = raw.find("[")
        end = raw.rfind("]") + 1

    if start >= 0 and end > start:
        json_str = raw[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Impossibile parsare JSON", "raw_response": raw}
