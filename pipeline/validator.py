"""
Normalizzazione e controllo di coerenza dell'output diagnostico del modello.

Serve perché un LLM può produrre un JSON formalmente valido ma internamente
contraddittorio: diagnosi primaria diversa dalla conclusione del ragionamento,
punteggi che non sommano a 1, una diagnosi "ESCLUSA" con probabilità residua,
la primaria ripetuta tra le differenziali con un'etichetta opposta.

Principio: le normalizzazioni sono solo sintattiche (codici, etichette, tipi).
Le incoerenze di merito vengono SEGNALATE, non corrette in silenzio: sostituire
la diagnosi scelta dal modello falsificherebbe il dato che lo studio misura.
"""

from __future__ import annotations

import re

from config.settings import AD_PPA_VARIANTS, DIAGNOSIS_LABELS

# Tolleranza sulla somma dei punteggi prima di segnalarla.
SCORE_SUM_TOLERANCE = 0.05

# Punteggio massimo ammesso per una diagnosi dichiarata ESCLUSA.
EXCLUDED_MAX_SCORE = 0.001

# Bande di probabilità dichiarate nel system prompt.
_PROBABILITY_BANDS = {
    "ALTA": (0.5, 1.0),
    "MEDIA": (0.2, 0.5),
    "BASSA": (0.0, 0.2),
    "ESCLUSA": (0.0, 0.0),
}

_CANONICAL = {code.upper(): code for code in DIAGNOSIS_LABELS}
_CANONICAL.update({v.upper().replace(" ", ""): "AD-PPA" for v in AD_PPA_VARIANTS})


def canonical_code(value) -> str:
    """
    Riporta un codice diagnostico alla forma canonica di DIAGNOSIS_LABELS.
    I modelli scrivono "Mixed", "mixed", "AD PPA", "ad-ppa": sono la stessa cosa.
    """
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper in _CANONICAL:
        return _CANONICAL[upper]
    compact = upper.replace(" ", "").replace("_", "")
    if compact in _CANONICAL:
        return _CANONICAL[compact]
    if compact.replace("-", "") == "ADPPA":
        return "AD-PPA"
    return raw


def _score(entry: dict) -> float:
    value = entry.get("confidence_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def expand_combined(entry: dict) -> list[dict]:
    """
    Separa una voce che accorpa più codici ("FTD|PD", "FTD/PD") in voci distinte.

    I modelli raggruppano le diagnosi che escludono in blocco: senza questa
    espansione risulterebbero "non valutate" e il punteggio resterebbe attribuito
    a un codice inesistente. Il punteggio viene ripartito per non alterare la somma.
    """
    raw = entry.get("diagnosis")
    if not isinstance(raw, str) or not any(sep in raw for sep in ("|", "/", ",")):
        return [entry]

    parts = [p.strip() for p in re.split(r"[|/,]", raw) if p.strip()]
    codes = [canonical_code(p) for p in parts]
    if len(codes) < 2 or not all(c in DIAGNOSIS_LABELS for c in codes):
        return [entry]

    share = _score(entry) / len(codes)
    return [{**entry, "diagnosis": code, "confidence_score": share} for code in codes]


def _normalize_entry(entry: dict) -> dict:
    """Canonicalizza codice, etichetta e punteggio di una singola voce diagnostica."""
    if not isinstance(entry, dict):
        return {}
    code = canonical_code(entry.get("diagnosis"))
    out = dict(entry)
    out["diagnosis"] = code
    out["confidence_score"] = _score(entry)
    if code in DIAGNOSIS_LABELS and not str(entry.get("label") or "").strip():
        out["label"] = DIAGNOSIS_LABELS[code]
    probability = str(entry.get("probability") or "").strip().upper()
    out["probability"] = probability if probability in _PROBABILITY_BANDS else probability
    return out


def validate_step_result(result: dict | None) -> dict | None:
    """
    Restituisce il risultato con codici normalizzati e un blocco "consistency".

    consistency contiene:
      warnings          elenco di incoerenze rilevate, in italiano
      score_sum         somma dei confidence_score dichiarati
      argmax_diagnosis  diagnosi con il punteggio più alto
      primary_is_argmax se la primaria coincide con l'argmax
      normalized_scores punteggi riscalati a somma 1, per i grafici
    """
    if not isinstance(result, dict) or "error" in result:
        return result

    primary = _normalize_entry(result.get("primary_diagnosis") or {})
    differentials = [
        _normalize_entry(expanded)
        for d in (result.get("differential_diagnoses") or []) if isinstance(d, dict)
        for expanded in expand_combined(d)
    ]

    warnings: list[str] = []
    primary_code = primary.get("diagnosis", "")

    if not primary_code:
        warnings.append("Diagnosi primaria assente o non riconosciuta")
    elif primary_code not in DIAGNOSIS_LABELS:
        warnings.append(f"Codice diagnostico non previsto: '{primary_code}'")

    # La primaria ripetuta tra le differenziali produce doppi conteggi e, quando
    # l'etichetta differisce, una contraddizione esplicita.
    repeated = [d for d in differentials if d.get("diagnosis") == primary_code]
    for d in repeated:
        warnings.append(
            f"{primary_code} è la diagnosi primaria ma compare anche tra le "
            f"differenziali come {d.get('probability') or '?'}"
        )
    differentials = [d for d in differentials if d.get("diagnosis") != primary_code]

    seen: set[str] = set()
    deduped: list[dict] = []
    for d in differentials:
        code = d.get("diagnosis", "")
        if code in seen:
            warnings.append(f"Diagnosi differenziale duplicata: {code}")
            continue
        seen.add(code)
        deduped.append(d)
    differentials = deduped

    entries = [primary] + differentials
    score_sum = sum(_score(e) for e in entries)
    if abs(score_sum - 1.0) > SCORE_SUM_TOLERANCE:
        warnings.append(
            f"I confidence_score sommano a {score_sum:.2f} invece di 1.00"
        )

    argmax = max(entries, key=_score, default={})
    argmax_code = argmax.get("diagnosis", "")
    primary_is_argmax = bool(primary_code) and argmax_code == primary_code
    if not primary_is_argmax and argmax_code:
        warnings.append(
            f"INCOERENZA: la primaria è {primary_code} (score {_score(primary):.2f}) "
            f"ma {argmax_code} ha un punteggio più alto ({_score(argmax):.2f})"
        )

    for entry in entries:
        code = entry.get("diagnosis", "?")
        probability = entry.get("probability", "")
        score = _score(entry)
        if probability == "ESCLUSA" and score > EXCLUDED_MAX_SCORE:
            warnings.append(f"{code} è dichiarata ESCLUSA ma ha score {score:.2f}")
        elif probability in _PROBABILITY_BANDS and probability != "ESCLUSA":
            low, high = _PROBABILITY_BANDS[probability]
            if not (low <= score <= high):
                warnings.append(
                    f"{code}: probabilità {probability} incompatibile con score {score:.2f}"
                )
        elif probability and probability not in _PROBABILITY_BANDS:
            warnings.append(f"{code}: livello di probabilità non previsto '{probability}'")

    missing = set(DIAGNOSIS_LABELS) - {e.get("diagnosis") for e in entries}
    if missing:
        warnings.append(f"Diagnosi non valutate: {', '.join(sorted(missing))}")

    total = score_sum or 1.0
    normalized_scores = {
        e.get("diagnosis", "?"): round(_score(e) / total, 4) for e in entries
    }

    return {
        **result,
        "primary_diagnosis": primary,
        "differential_diagnoses": differentials,
        "consistency": {
            "warnings": warnings,
            "score_sum": round(score_sum, 4),
            "argmax_diagnosis": argmax_code,
            "primary_is_argmax": primary_is_argmax,
            "normalized_scores": normalized_scores,
        },
    }
