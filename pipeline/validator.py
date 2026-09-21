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

import math
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


def _probability_from_score(score: float) -> str:
    if score <= EXCLUDED_MAX_SCORE:
        return "ESCLUSA"
    if score < 0.2:
        return "BASSA"
    if score < 0.5:
        return "MEDIA"
    return "ALTA"


def _normalize_entry(entry: dict) -> dict:
    if not isinstance(entry, dict):
        return {}
    code = canonical_code(entry.get("diagnosis"))
    out = dict(entry)
    out["diagnosis"] = code
    out["confidence_score"] = _score(entry)
    if code in DIAGNOSIS_LABELS and not str(entry.get("label") or "").strip():
        out["label"] = DIAGNOSIS_LABELS[code]
    reported_probability = str(entry.get("probability") or "").strip().upper()
    if reported_probability:
        out["reported_probability"] = reported_probability
    return out


def _entries_from_fixed_assessments(result: dict) -> tuple[dict, list[dict]] | None:
    assessments = result.get("diagnosis_assessments")
    primary_code = canonical_code(result.get("primary_diagnosis"))
    if not isinstance(assessments, dict) or not isinstance(result.get("primary_diagnosis"), str):
        return None

    entries = []
    for code in DIAGNOSIS_LABELS:
        assessment = assessments.get(code) or {}
        entries.append(_normalize_entry({
            "diagnosis": code,
            "label": DIAGNOSIS_LABELS[code],
            "reasoning": assessment.get("reasoning", ""),
            "confidence_score": assessment.get("confidence_score"),
        }))
    primary = next((entry for entry in entries if entry["diagnosis"] == primary_code), {})
    differentials = [entry for entry in entries if entry["diagnosis"] != primary_code]
    return primary, differentials


def _same_number(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-4, abs_tol=5e-5)
    except (TypeError, ValueError):
        return False


def _biomarker_warnings(result: dict, computed: dict | None) -> list[str]:
    if not computed:
        return []
    step = result.get("step")
    group = "plasma" if step == 2 else ("csf" if step == 3 else None)
    output_key = (
        "plasma_biomarker_interpretation" if step == 2
        else "csf_biomarker_interpretation" if step == 3 else None
    )
    warnings: list[str] = []
    if group and output_key and output_key in result:
        reported = result.get(output_key) or {}
        for name, expected in computed.get(group, {}).items():
            actual = reported.get(name) or {}
            if actual.get("status") != expected.get("status"):
                warnings.append(
                    f"{name}: status del modello {actual.get('status')} diverso dal "
                    f"calcolo deterministico {expected.get('status')}"
                )
            if not _same_number(actual.get("value"), expected.get("value")):
                warnings.append(
                    f"{name}: valore riportato dal modello {actual.get('value')} diverso "
                    f"dall'input {expected.get('value')}"
                )

    reliability = result.get("biomarker_reliability")
    safety = computed.get("safety") or {}
    if isinstance(reliability, dict):
        expected_renal = safety.get("renal_status") == "OK"
        expected_hepatic = safety.get("hepatic_status") == "OK"
        if safety.get("renal_status") in {"OK", "ALTERED"} and \
                reliability.get("renal_function_ok") != expected_renal:
            warnings.append("Valutazione renale del modello discordante dal calcolo deterministico")
        if safety.get("hepatic_status") in {"OK", "ALTERED"} and \
                reliability.get("hepatic_function_ok") != expected_hepatic:
            warnings.append("Valutazione epatica del modello discordante dal calcolo deterministico")

    if step == 3 and "at_profile" in result:
        expected_atn = (computed.get("atn") or {}).get("profile")
        reported_atn = str(result.get("at_profile") or "").replace(" ", "").upper()
        if expected_atn and reported_atn != expected_atn.upper():
            warnings.append(
                f"Profilo ATN del modello '{result.get('at_profile')}' diverso dal "
                f"calcolo deterministico '{expected_atn}'"
            )
    return warnings


def _normalize_citation_text(value) -> str:
    text = str(value or "").replace("**", "").replace("__", "").replace("`", "")
    return " ".join(text.split()).strip(' "“”').lower()


def _citation_warnings(result: dict, rag_sources: list[dict] | None) -> list[str]:
    if not rag_sources:
        return []
    source_map = {source.get("index"): source for source in rag_sources}
    warnings: list[str] = []
    used = result.get("rag_sources_used") or []
    for index in used:
        if index not in source_map:
            warnings.append(f"Indice fonte inesistente in rag_sources_used: {index}")
    for evidence in result.get("rag_evidence") or []:
        index = evidence.get("source_index")
        source = source_map.get(index)
        if source is None:
            warnings.append(f"Evidenza riferita a fonte inesistente: {index}")
            continue
        quote = _normalize_citation_text(evidence.get("quote"))
        text = _normalize_citation_text(source.get("text"))
        if not quote or quote not in text:
            warnings.append(f"Citazione non letterale o non trovata nella Fonte {index}")
    return warnings


def validate_step_result(
    result: dict | None,
    computed_biomarkers: dict | None = None,
    rag_sources: list[dict] | None = None,
) -> dict | None:
    """Normalizza la rappresentazione e segnala incoerenze senza correggere il verdetto."""
    if not isinstance(result, dict) or "error" in result:
        return result

    fixed_entries = _entries_from_fixed_assessments(result)
    if fixed_entries is not None:
        primary, differentials = fixed_entries
    else:
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

    normalization_denominator = score_sum if score_sum > 0 else 1.0
    for entry in entries:
        reported_score = _score(entry)
        entry["reported_confidence_score"] = reported_score
        entry["confidence_score"] = reported_score / normalization_denominator
        entry["probability"] = _probability_from_score(entry["confidence_score"])

    for entry in entries:
        code = entry.get("diagnosis", "?")
        probability = entry.get("probability", "")
        score = _score(entry)
        reported_probability = entry.get("reported_probability")
        if reported_probability and reported_probability != probability:
            warnings.append(
                f"{code}: categoria dichiarata {reported_probability} diversa dalla "
                f"categoria derivata {probability}"
            )
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

    biomarker_warnings = _biomarker_warnings(result, computed_biomarkers)
    citation_warnings = _citation_warnings(result, rag_sources)
    warnings.extend(biomarker_warnings)
    warnings.extend(citation_warnings)

    normalized_scores = {
        e.get("diagnosis", "?"): round(_score(e), 4) for e in entries
    }

    output = {
        **result,
        "primary_diagnosis": primary,
        "differential_diagnoses": differentials,
        "consistency": {
            "warnings": warnings,
            "score_sum": round(score_sum, 4),
            "argmax_diagnosis": argmax_code,
            "primary_is_argmax": primary_is_argmax,
            "normalized_scores": normalized_scores,
            "biomarker_warnings": biomarker_warnings,
            "citation_warnings": citation_warnings,
        },
    }
    if computed_biomarkers:
        output["computed_biomarkers"] = computed_biomarkers
        output["biomarker_reliability"] = computed_biomarkers.get("safety")
        if computed_biomarkers.get("atn"):
            output["at_profile"] = computed_biomarkers["atn"]["profile"]
        if computed_biomarkers.get("plasma_csf_concordance"):
            output["plasma_csf_concordance"] = computed_biomarkers["plasma_csf_concordance"]
    return output
