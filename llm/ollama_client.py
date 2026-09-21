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


def get_model_info(model_name: str) -> dict | None:
    requested = model_name.lower()
    for model in list_models():
        name = str(model.get("name", "")).lower()
        if name == requested or name.split(":")[0] == requested.split(":")[0]:
            return model
    return None


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


def generate_detailed(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    seed: int | None = None,
    output_schema: dict | None = None,
) -> dict:
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": LLM_NUM_CTX,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
    if seed is not None:
        options["seed"] = int(seed)

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": False,
        "format": output_schema or "json",
        "options": options,
    }
    request_metadata = {
        "model": model,
        "seed": seed,
        "think": False,
        "structured_output": output_schema is not None,
        "options": options,
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=LLM_TIMEOUT_S
        )
        response.raise_for_status()
        data = response.json()
        return {
            "response": data.get("response", ""),
            "error": None,
            "request": request_metadata,
            "ollama": {
                key: value for key, value in data.items()
                if key not in {"response", "context"}
            },
        }
    except requests.Timeout:
        message = f"Timeout: nessuna risposta entro {LLM_TIMEOUT_S}s"
    except Exception as exc:
        message = f"Errore Ollama: {exc}"
    return {
        "response": _error_json(message),
        "error": message,
        "request": request_metadata,
        "ollama": {},
    }


def generate(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    seed: int | None = None,
    output_schema: dict | None = None,
) -> str:
    return generate_detailed(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        output_schema=output_schema,
    )["response"]


def _error_json(message: str) -> str:
    """Serializza l'errore con json.dumps: il messaggio può contenere virgolette."""
    return json.dumps({"error": message}, ensure_ascii=False)


# Quanti delimitatori arretrare al massimo cercando un punto di taglio valido.
_REPAIR_BACKTRACK_LIMIT = 400


def _try_repair_json_detailed(text: str) -> tuple[Optional[dict], dict]:
    stack, in_string = _scan_delimiters(text)
    if not in_string and not stack:
        return None, {"discarded_chars": 0, "backtrack_steps": 0}

    candidate = _close_json(text)
    if candidate is not None:
        return candidate, {"discarded_chars": 0, "backtrack_steps": 0}

    cut = len(text)
    for step in range(1, _REPAIR_BACKTRACK_LIMIT + 1):
        cut = max(text.rfind(",", 0, cut), text.rfind("}", 0, cut), text.rfind("]", 0, cut))
        if cut <= 0:
            break
        candidate = _close_json(text[:cut])
        if candidate is not None:
            return candidate, {
                "discarded_chars": len(text) - cut,
                "backtrack_steps": step,
            }
    return None, {"discarded_chars": len(text), "backtrack_steps": _REPAIR_BACKTRACK_LIMIT}


def _try_repair_json(text: str) -> Optional[dict]:
    return _try_repair_json_detailed(text)[0]


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


def parse_json_response_detailed(raw: str) -> dict:
    metadata = {
        "status": "error",
        "repaired": False,
        "source": None,
        "raw_chars": len(raw or ""),
        "cleaned_chars": 0,
        "discarded_chars": 0,
        "backtrack_steps": 0,
    }
    if not raw or not raw.strip():
        return {
            "data": {"error": "Risposta vuota dal modello", "raw_response": raw},
            "metadata": metadata,
        }

    cleaned = _strip_markdown_fence(_strip_think_tags(raw))
    metadata["cleaned_chars"] = len(cleaned)
    candidates: list[tuple[str, str]] = []

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        candidates.append(("extracted", cleaned[start:end]))
    candidates.append(("full", cleaned))

    seen: set[str] = set()
    for source, candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return {
                    "data": data,
                    "metadata": {**metadata, "status": "exact", "source": source},
                }
        except json.JSONDecodeError:
            pass

        repaired, repair_meta = _try_repair_json_detailed(candidate)
        if repaired is not None:
            return {
                "data": repaired,
                "metadata": {
                    **metadata,
                    **repair_meta,
                    "status": "repaired",
                    "repaired": True,
                    "source": source,
                },
            }

    return {
        "data": {"error": "Impossibile parsare JSON", "raw_response": raw},
        "metadata": metadata,
    }


def parse_json_response(raw: str) -> dict:
    return parse_json_response_detailed(raw)["data"]
