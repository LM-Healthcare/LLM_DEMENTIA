"""
Test di regressione del validatore dell'output diagnostico.

Il primo caso riproduce il difetto osservato sul paziente T2: diagnosi primaria
SCD dichiarata ALTA, mentre tra le differenziali VAD aveva probabilità superiore
e la stessa SCD compariva come ESCLUSA. Senza controllo, l'interfaccia mostrava
la contraddizione senza segnalarla.

Uso:
    python scripts/test_validator.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pipeline.validator import canonical_code, expand_combined, validate_step_result

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok' if condition else 'FALLITO'}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def test_canonical_codes() -> None:
    print("\ncanonicalizzazione dei codici")
    for raw, expected in [
        ("Mixed", "MIXED"), ("mixed", "MIXED"), ("  vad ", "VAD"),
        ("AD PPA", "AD-PPA"), ("ad_ppa", "AD-PPA"), ("AD -PPA", "AD-PPA"),
        ("AD", "AD"), ("late", "LATE"),
    ]:
        check(f"'{raw}' -> {expected}", canonical_code(raw) == expected, canonical_code(raw))
    check("valore non stringa -> ''", canonical_code(None) == "")


def test_combined_codes() -> None:
    print("\nespansione dei codici accorpati")
    out = expand_combined({"diagnosis": "FTD|PD", "confidence_score": 0.2,
                           "probability": "ESCLUSA"})
    check("'FTD|PD' produce due voci", len(out) == 2, str([o["diagnosis"] for o in out]))
    check("il punteggio viene ripartito",
          abs(sum(o["confidence_score"] for o in out) - 0.2) < 1e-9)
    single = expand_combined({"diagnosis": "AD-PPA", "confidence_score": 0.3})
    check("'AD-PPA' non viene spezzato sul trattino", len(single) == 1)


def test_t2_inconsistency() -> None:
    print("\ncaso T2: primaria diversa dalla diagnosi col punteggio più alto")
    result = validate_step_result({
        "primary_diagnosis": {"diagnosis": "SCD", "probability": "ALTA",
                              "confidence_score": 0.45, "reasoning": "conclude VAD"},
        "differential_diagnoses": [
            {"diagnosis": "VAD", "probability": "ALTA", "confidence_score": 0.60},
            {"diagnosis": "SCD", "probability": "ESCLUSA", "confidence_score": 0.10},
        ],
    })
    c = result["consistency"]
    check("l'incoerenza viene rilevata", c["primary_is_argmax"] is False)
    check("argmax individuato correttamente", c["argmax_diagnosis"] == "VAD", c["argmax_diagnosis"])
    check("la primaria NON viene sovrascritta",
          result["primary_diagnosis"]["diagnosis"] == "SCD")
    check("la primaria duplicata è rimossa dalle differenziali",
          all(d["diagnosis"] != "SCD" for d in result["differential_diagnoses"]))
    check("la duplicazione è segnalata",
          any("compare anche tra le differenziali" in w for w in c["warnings"]))
    check("la somma errata è segnalata", any("sommano a" in w for w in c["warnings"]))


def test_coherent_result() -> None:
    print("\nrisultato coerente: nessun falso allarme")
    entries = [{"diagnosis": code, "probability": "ESCLUSA", "confidence_score": 0.0}
               for code in ("AD", "AD-PPA", "MIXED", "SCD", "LATE", "FTD", "PD")]
    result = validate_step_result({
        "primary_diagnosis": {"diagnosis": "VAD", "probability": "ALTA",
                              "confidence_score": 1.0, "reasoning": "..."},
        "differential_diagnoses": entries,
    })
    c = result["consistency"]
    check("coerenza confermata", c["primary_is_argmax"] is True)
    check("nessun avviso", c["warnings"] == [], "; ".join(c["warnings"]))
    check("etichetta compilata dal codice",
          result["primary_diagnosis"]["label"] == "Demenza Vascolare")


def test_excluded_with_score() -> None:
    print("\nESCLUSA con punteggio residuo")
    result = validate_step_result({
        "primary_diagnosis": {"diagnosis": "AD", "probability": "ALTA", "confidence_score": 0.8},
        "differential_diagnoses": [
            {"diagnosis": "VAD", "probability": "ESCLUSA", "confidence_score": 0.2},
        ],
    })
    check("contraddizione ESCLUSA/score segnalata",
          any("ESCLUSA ma ha score" in w for w in result["consistency"]["warnings"]))


def test_error_passthrough() -> None:
    print("\nrisultato con errore di parsing")
    payload = {"error": "Impossibile parsare JSON"}
    check("il validatore non altera un errore", validate_step_result(payload) is payload)
    check("None resta None", validate_step_result(None) is None)


def main() -> None:
    test_canonical_codes()
    test_combined_codes()
    test_t2_inconsistency()
    test_coherent_result()
    test_excluded_with_score()
    test_error_passthrough()

    print()
    if failures:
        print(f"{len(failures)} CONTROLLI FALLITI: {', '.join(failures)}")
        sys.exit(1)
    print("Tutti i controlli del validatore superati.")


if __name__ == "__main__":
    main()
