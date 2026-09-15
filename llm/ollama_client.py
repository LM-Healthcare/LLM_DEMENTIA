"""
Client per Ollama: gestione modelli, generazione testo e pulling.
"""

from __future__ import annotations

import json
import re
import requests
from typing import Optional, Generator

from config.settings import (
    LLM_MAX_TOKENS,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


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
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """
    Genera testo tramite Ollama /api/generate.
    temperature bassa per risposta deterministica e clinicamente rigorosa.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": LLM_NUM_CTX,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=LLM_TIMEOUT_S
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.Timeout:
        return _error_json(f"Timeout: nessuna risposta entro {LLM_TIMEOUT_S}s")
    except Exception as e:
        return _error_json(f"Errore Ollama: {e}")


def _error_json(message: str) -> str:
    """Serializza l'errore con json.dumps: il messaggio può contenere virgolette."""
    return json.dumps({"error": message}, ensure_ascii=False)


# Quanti delimitatori arretrare al massimo cercando un punto di taglio valido.
_REPAIR_BACKTRACK_LIMIT = 400


def _try_repair_json(text: str) -> Optional[dict]:
    """
    Ripara un JSON troncato chiudendo stringa e parentesi rimaste aperte.

    Serve quando un modello esaurisce il budget di token a metà risposta: senza
    questo recupero l'intero step viene perso. I delimitatori vanno chiusi
    nell'ordine inverso di apertura, quindi si traccia la pila e non due
    semplici contatori (che sbagliano l'ordine sulle strutture annidate).
    """
    stack, in_string = _scan_delimiters(text)
    if not in_string and not stack:
        return None

    # Primo tentativo: chiudere quello che è rimasto aperto così com'è.
    candidate = _close_json(text)
    if candidate is not None:
        return candidate

    # Il taglio può cadere su un frammento non recuperabile (una chiave senza
    # valore, un numero a metà): si arretra al delimitatore precedente e si
    # riprova, scartando l'ultimo elemento incompleto.
    cut = len(text)
    for _ in range(_REPAIR_BACKTRACK_LIMIT):
        cut = max(text.rfind(",", 0, cut), text.rfind("}", 0, cut), text.rfind("]", 0, cut))
        if cut <= 0:
            break
        candidate = _close_json(text[:cut])
        if candidate is not None:
            return candidate
    return None


def _scan_delimiters(text: str) -> tuple[list[str], bool]:
    """Pila dei delimitatori ancora aperti e se il testo termina dentro una stringa."""
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
        elif in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()

    return stack, in_string


def _close_json(text: str) -> Optional[dict]:
    """
    Chiude stringa e delimitatori aperti e prova a parsare.
    I delimitatori vanno chiusi nell'ordine inverso di apertura: due semplici
    contatori sbaglierebbero l'ordine sulle strutture annidate.
    """
    stack, in_string = _scan_delimiters(text)
    repaired = text + ('"' if in_string else "")

    # Una coppia chiave/valore lasciata a metà ("reasoning": ) non è recuperabile.
    repaired = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", repaired.rstrip())
    repaired = repaired.rstrip().rstrip(",").rstrip()

    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"

    try:
        result = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def _strip_think_tags(text: str) -> str:
    """Rimuove i blocchi <think>...</think> prodotti da Qwen3 e modelli simili."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_markdown_fence(text: str) -> str:
    """Rimuove le code fence markdown (```json ... ``` o ``` ... ```)."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def parse_json_response(raw: str) -> dict:
    """
    Estrae il JSON dalla risposta del modello.
    Gestisce: testo prima/dopo, blocchi <think> (Qwen3), markdown code fences.
    """
    if not raw or not raw.strip():
        print("[LLM] Risposta vuota dal modello")
        return {"error": "Risposta vuota dal modello", "raw_response": raw}

    cleaned = _strip_think_tags(raw)
    cleaned = _strip_markdown_fence(cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1

    if start >= 0 and end > start:
        json_str = cleaned[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            repaired = _try_repair_json(json_str)
            if repaired:
                print(f"[LLM] JSON riparato da troncamento ({len(json_str)} chars)")
                return repaired

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _try_repair_json(cleaned)
        if repaired:
            print(f"[LLM] JSON riparato da troncamento (full cleaned, {len(cleaned)} chars)")
            return repaired
        print(f"[LLM] Impossibile parsare JSON. Lunghezza risposta: {len(raw)} chars")
        print(f"[LLM] Inizio (600 chars):\n{raw[:600]}")
        print(f"[LLM] Fine (300 chars):\n{raw[-300:]}" if len(raw) > 600 else "")
        return {"error": "Impossibile parsare JSON", "raw_response": raw}
