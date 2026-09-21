"""Interpretazione deterministica dei biomarcatori secondo i cut-off dello studio."""

from __future__ import annotations

import math

from config.settings import REFERENCE_VALUES, reference_range
from data.loader import CSF_BIOMARKER_COLS, PLASMA_BIOMARKER_COLS, SAFETY_LAB_COLS

_LOW_IS_PATHOLOGICAL = {"CSF_Ab42", "CSF_Ab4240", "Plasma_Ab4240"}
_HIGH_IS_PATHOLOGICAL = {
    "CSF_ttau", "CSF_ptau", "CSF_NfL",
    "plasma_ptau217", "plasma_pt181", "plasma_NfL",
}


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def assess_marker(name: str, value) -> dict:
    numeric = _number(value)
    ref = REFERENCE_VALUES.get(name, {})
    if numeric is None:
        return {
            "value": None,
            "status": "MANCANTE",
            "pathology_positive": None,
            "range_position": "MISSING",
            "reference": reference_range(name),
        }

    minimum, maximum = ref.get("min"), ref.get("max")
    if minimum is not None and numeric < minimum:
        position = "LOW"
    elif maximum is not None and numeric > maximum:
        position = "HIGH"
    else:
        position = "IN_RANGE"

    if name in _LOW_IS_PATHOLOGICAL:
        positive = minimum is not None and numeric < minimum
    elif name in _HIGH_IS_PATHOLOGICAL:
        positive = maximum is not None and numeric > maximum
    else:
        positive = None

    return {
        "value": numeric,
        "status": "PATOLOGICO" if positive else "NORMALE",
        "pathology_positive": positive,
        "range_position": position,
        "reference": reference_range(name),
    }


def _organ_status(assessments: dict[str, dict], names: tuple[str, ...]) -> str:
    available = [assessments[name] for name in names if assessments[name]["value"] is not None]
    if not available:
        return "UNKNOWN"
    if any(item["confounder"] for item in available):
        return "ALTERED"
    return "OK" if len(available) == len(names) else "PARTIAL"


def assess_safety_labs(record: dict) -> dict:
    assessments: dict[str, dict] = {}
    for name in SAFETY_LAB_COLS:
        base = assess_marker(name, record.get(name))
        value = base["value"]
        if name == "Creatinina":
            confounder = value is not None and value > REFERENCE_VALUES[name]["max"]
        elif name == "eGFR_2021":
            confounder = value is not None and value < REFERENCE_VALUES[name]["min"]
        else:
            confounder = value is not None and value > REFERENCE_VALUES[name]["max"]
        assessments[name] = {**base, "confounder": confounder}

    renal = _organ_status(assessments, ("Creatinina", "eGFR_2021"))
    hepatic = _organ_status(assessments, ("AST", "ALT"))
    altered = renal == "ALTERED" or hepatic == "ALTERED"
    incomplete = renal in {"UNKNOWN", "PARTIAL"} or hepatic in {"UNKNOWN", "PARTIAL"}
    return {
        "markers": assessments,
        "renal_status": renal,
        "hepatic_status": hepatic,
        "interpretation_caution": altered,
        "incomplete": incomplete,
        "reliability": "CAUTION" if altered else ("INCOMPLETE" if incomplete else "STANDARD"),
        "note": (
            "Sono presenti possibili confondenti epato-renali: interpretare i biomarcatori "
            "plasmatici con cautela, senza invalidarli automaticamente."
            if altered else
            "Nessun confondente epato-renale rilevato nei dati disponibili."
        ),
    }


def _axis(symbol: str, assessment: dict | None) -> dict:
    if not assessment or assessment.get("pathology_positive") is None:
        return {"symbol": f"{symbol}?", "positive": None}
    positive = bool(assessment["pathology_positive"])
    return {"symbol": f"{symbol}{'+' if positive else '-'}", "positive": positive}


def compute_atn(csf: dict[str, dict]) -> dict:
    amyloid = csf.get("CSF_Ab4240")
    amyloid_source = "CSF_Ab4240"
    if not amyloid or amyloid.get("value") is None:
        amyloid = csf.get("CSF_Ab42")
        amyloid_source = "CSF_Ab42"

    a_axis = _axis("A", amyloid)
    t_axis = _axis("T", csf.get("CSF_ptau"))

    n_candidates = [csf.get("CSF_ttau"), csf.get("CSF_NfL")]
    available_n = [item for item in n_candidates if item and item.get("pathology_positive") is not None]
    if not available_n:
        n_axis = {"symbol": "N?", "positive": None}
    else:
        n_positive = any(bool(item["pathology_positive"]) for item in available_n)
        n_axis = {"symbol": f"N{'+' if n_positive else '-'}", "positive": n_positive}

    return {
        "profile": f"{a_axis['symbol']}{t_axis['symbol']}{n_axis['symbol']}",
        "A": {**a_axis, "source": amyloid_source},
        "T": {**t_axis, "source": "CSF_ptau"},
        "N": {**n_axis, "sources": ["CSF_ttau", "CSF_NfL"]},
        "rule": (
            "A: CSF Aβ42/40 prioritario, Aβ42 solo se il rapporto manca; "
            "T: CSF p-tau; N: CSF t-tau o NfL. Definizione operativa dello studio."
        ),
    }


def _concordance(plasma: dict[str, dict], csf: dict[str, dict]) -> dict:
    pairs = {
        "amyloid": (plasma.get("Plasma_Ab4240"), csf.get("CSF_Ab4240")),
        "tau181": (plasma.get("plasma_pt181"), csf.get("CSF_ptau")),
    }
    output = {}
    for axis, (blood, liquor) in pairs.items():
        blood_status = None if not blood else blood.get("pathology_positive")
        liquor_status = None if not liquor else liquor.get("pathology_positive")
        if blood_status is None or liquor_status is None:
            status = "INDETERMINATE"
        else:
            status = "CONCORDANT" if blood_status == liquor_status else "DISCORDANT"
        output[axis] = {
            "status": status,
            "plasma_positive": blood_status,
            "csf_positive": liquor_status,
        }
    return output


def compute_biomarker_assessment(record: dict, step: int) -> dict:
    plasma = {name: assess_marker(name, record.get(name)) for name in PLASMA_BIOMARKER_COLS}
    safety = assess_safety_labs(record)
    output = {"plasma": plasma, "safety": safety}
    if step >= 3:
        csf = {name: assess_marker(name, record.get(name)) for name in CSF_BIOMARKER_COLS}
        output.update({
            "csf": csf,
            "atn": compute_atn(csf),
            "plasma_csf_concordance": _concordance(plasma, csf),
        })
    return output


def format_biomarker_assessment(assessment: dict, step: int) -> str:
    lines = ["=== INTERPRETAZIONE DETERMINISTICA CALCOLATA DAL SISTEMA ==="]
    if step >= 2:
        lines.append("Plasma:")
        for name, item in assessment["plasma"].items():
            lines.append(
                f"  - {name}: {item['status']} "
                f"(valore={item['value']}, riferimento={item['reference'] or 'N/D'})"
            )
        safety = assessment["safety"]
        lines.extend([
            f"Funzione renale: {safety['renal_status']}",
            f"Funzione epatica: {safety['hepatic_status']}",
            f"Affidabilità plasma: {safety['reliability']} — {safety['note']}",
        ])
    if step >= 3:
        lines.append("Liquor:")
        for name, item in assessment["csf"].items():
            lines.append(
                f"  - {name}: {item['status']} "
                f"(valore={item['value']}, riferimento={item['reference'] or 'N/D'})"
            )
        lines.append(f"Profilo ATN operativo: {assessment['atn']['profile']}")
        lines.append(f"Regola ATN: {assessment['atn']['rule']}")
    lines.append(
        "Questi status sono calcolati deterministicamente dal sistema: non ricalcolarli "
        "e non modificarli. Usali nell'analisi diagnostica."
    )
    return "\n".join(lines)
