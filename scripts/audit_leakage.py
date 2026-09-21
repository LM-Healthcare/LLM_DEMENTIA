"""
Audit di integrità scientifica: il modello deve essere ignaro della diagnosi.

Esegue tre controlli indipendenti su tutti i pazienti del database.

  A. Strutturale — le colonne riservate (Diagnosi_CODIFICATA, Diagnosi_TESTUALE,
     PAZIENTE) non compaiono nel payload passato ai costruttori di prompt.
  B. Differenziale — i prompt e le query RAG vengono costruiti due volte: con il
     record reale e con le colonne riservate sostituite da un valore sentinella.
     Se i due risultati sono identici byte a byte, è dimostrato che quelle
     colonne non possono influenzare ciò che il modello riceve. È una prova, non
     un controllo euristico su sottostringhe.
  C. Testuale — i campi in testo libero (ANAMNESI, EON, TERAPIA) non contengono
     termini che rivelano la diagnosi. È il rischio più insidioso: nessun filtro
     software lo intercetta, perché il dato è già nel campo clinico.

Uso:
    python scripts/audit_leakage.py            # riepilogo
    python scripts/audit_leakage.py --verbose  # elenca ogni occorrenza
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from data.biomarkers import compute_biomarker_assessment
from data.loader import HIDDEN_COLS, build_step_payload, get_database, get_ground_truth
from data.preprocessor import format_therapy_for_prompt, standardize_therapy
from llm.prompt_builder import build_step1_prompt, build_step2_prompt, build_step3_prompt
from pipeline.step_runner import _clinical_context
from rag.retriever import build_queries

# Valore con cui si sostituiscono le colonne riservate nel test differenziale.
_SENTINEL = "ZZZ_COLONNA_RISERVATA_ZZZ"

# Termini che rivelano una diagnosi se presenti nei campi clinici in testo libero.
# La chiave è il codice diagnostico rivelato.
#
# Gli acronimi vanno cercati SENZA ignorecase: "AD" maiuscolo è la diagnosi,
# "ad" minuscolo è la preposizione italiana ("ad eccezione", "ad esordio"), e
# confonderli produce decine di falsi positivi.
_TERMS_CASE_INSENSITIVE: dict[str, tuple[str, ...]] = {
    "AD": (r"alzheimer",),
    "VAD": (r"demenza vascolare", r"eziologia vascolare",
            r"decadimento cognitivo[^.]{0,40}vascolare", r"deterioramento[^.]{0,40}vascolare"),
    "MIXED": (r"eziologia mista", r"demenza mista", r"forma mista"),
    "FTD": (r"frontotemporale", r"fronto-temporale", r"degenerazione lobare"),
    "PD": (r"malattia di parkinson", r"corpi di lewy"),
    "LATE": (r"TDP-?43",),
    "SCD": (r"disturbo soggettivo",),
    "AD-PPA": (r"logopenica", r"afasia progressiva"),
}

_TERMS_CASE_SENSITIVE: dict[str, tuple[str, ...]] = {
    "AD": (r"\bAD\b", r"\bNIA-?AA\b"),
    "VAD": (r"\bVaD\b", r"\bVCI\b"),
    "FTD": (r"\bFTD\b", r"\bbvFTD\b"),
    "PD": (r"\bDLB\b", r"\bLBD\b", r"\bPDD\b"),
    "LATE": (r"\bLATE\b",),
    "SCD": (r"\bSCD\b",),
    "AD-PPA": (r"\bPPA\b", r"\blvPPA\b"),
}

_FREE_TEXT_FIELDS = ("ANAMNESI", "EON", "TERAPIA")

# Risultati sintetici degli step precedenti: servono solo a costruire i prompt
# 2 e 3, e non contengono alcun riferimento alla diagnosi reale.
_PREV = {
    "primary_diagnosis": {"diagnosis": "AD", "label": "Malattia di Alzheimer",
                          "probability": "MEDIA", "confidence_score": 0.4,
                          "reasoning": "ipotesi in corso di verifica"},
    "differential_diagnoses": [],
    "clinical_summary": "sintesi provvisoria",
}


def _text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def build_all_prompts(record: dict) -> dict[int, str]:
    """Testo completo (system + user) dei tre prompt per un paziente."""
    terapia = standardize_therapy(_text(record.get("TERAPIA")))
    fmt = format_therapy_for_prompt(terapia)
    summary = terapia.get("summary")

    builders = {
        1: lambda: build_step1_prompt(
            build_step_payload(record, 1, summary), "CONTESTO_RAG_PLACEHOLDER", fmt),
        2: lambda: build_step2_prompt(
            build_step_payload(record, 2, summary), _PREV, fmt,
            compute_biomarker_assessment(record, 2)),
        3: lambda: build_step3_prompt(
            build_step_payload(record, 3, summary), _PREV, fmt,
            compute_biomarker_assessment(record, 3), step1_result=_PREV),
    }
    return {step: "\n".join(build()) for step, build in builders.items()}


def audit(df: pd.DataFrame, verbose: bool) -> bool:
    structural: list[str] = []
    literal: list[str] = []
    textual: list[tuple[str, str, str, str, str]] = []
    revealing_terms: Counter[str] = Counter()

    for _, row in df.iterrows():
        record = row.to_dict()
        code = _text(record.get("Codice"))
        gt = get_ground_truth(record)

        # ── A. strutturale ────────────────────────────────────────────────
        for step in (1, 2, 3):
            payload = build_step_payload(record, step, None)
            leaked = [k for k in payload if k in HIDDEN_COLS or k.lower() in
                      {c.lower() for c in HIDDEN_COLS}]
            if leaked:
                structural.append(f"{code} step {step}: chiavi riservate {leaked}")

        prompts = build_all_prompts(record)

        # ── B. differenziale ──────────────────────────────────────────────
        masked = {**record, **{col: _SENTINEL for col in HIDDEN_COLS}}
        masked_prompts = build_all_prompts(masked)
        for step, text in prompts.items():
            if text != masked_prompts[step]:
                literal.append(f"{code} step {step}: il prompt dipende da una colonna riservata")
            if _SENTINEL in text:
                literal.append(f"{code} step {step}: valore sentinella presente nel prompt")

        if build_queries(1, clinical_context=_clinical_context(record)) != \
                build_queries(1, clinical_context=_clinical_context(masked)):
            literal.append(f"{code}: le query RAG dipendono da una colonna riservata")

        # ── C. testuale sui campi liberi ──────────────────────────────────
        for field in _FREE_TEXT_FIELDS:
            content = _text(record.get(field))
            if not content:
                continue
            for diag_code, patterns, flags in (
                *((c, p, re.IGNORECASE) for c, p in _TERMS_CASE_INSENSITIVE.items()),
                *((c, p, 0) for c, p in _TERMS_CASE_SENSITIVE.items()),
            ):
                for pattern in patterns:
                    match = re.search(pattern, content, flags)
                    if not match:
                        continue
                    textual.append((code, field, diag_code, match.group(0), gt))
                    revealing_terms[f"'{match.group(0)}' rivela {diag_code}"] += 1
                    break

    print("=" * 74)
    print("A. CONTROLLO STRUTTURALE — colonne riservate nel payload")
    print("=" * 74)
    print(f"  colonne riservate: {HIDDEN_COLS}")
    print(f"  violazioni: {len(structural)}")
    for v in structural[:20]:
        print(f"    ! {v}")

    print()
    print("=" * 74)
    print("B. TEST DIFFERENZIALE — i prompt cambiano se si maschera la diagnosi?")
    print("=" * 74)
    print("  confronto prompt (3 step) e query RAG con colonne riservate mascherate")
    print(f"  prompt confrontati: {len(df) * 3}")
    print(f"  violazioni: {len(literal)}")
    for v in literal[:20]:
        print(f"    ! {v}")

    print()
    print("=" * 74)
    print("C. CONTROLLO TESTUALE — termini diagnostici nei campi in testo libero")
    print("=" * 74)
    critical = [t for t in textual if t[2] == t[4]]
    print(f"  pazienti totali            : {len(df)}")
    print(f"  occorrenze di termini      : {len(textual)}")
    print(f"  pazienti con occorrenze    : {len({t[0] for t in textual})}")
    print("  occorrenze CHE COINCIDONO")
    print(f"  con il ground truth        : {len(critical)} "
          f"(su {len({t[0] for t in critical})} pazienti)")

    if revealing_terms:
        print("\n  termini trovati (frequenza):")
        for term, n in revealing_terms.most_common(20):
            print(f"    {n:4d}  {term}")

    if verbose and textual:
        print("\n  dettaglio:")
        for code, field, diag, term, gt in textual:
            flag = "  <<< COINCIDE" if diag == gt else ""
            print(f"    {code:6s} {field:9s} '{term}' rivela {diag:7s} | gt={gt}{flag}")

    ok = not structural and not literal
    print()
    print("=" * 74)
    print("ESITO: " + ("nessuna fuga strutturale o letterale" if ok
                       else "FUGA DI INFORMAZIONE RILEVATA"))
    if critical:
        print("ATTENZIONE: i campi clinici in testo libero contengono termini che")
        print("coincidono con la diagnosi di riferimento. Non è un difetto del")
        print("codice ma del dato: va valutato con i clinici se anonimizzare i")
        print("campi ANAMNESI/EON prima di usare i risultati come evidenza.")
    print("=" * 74)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit di fuga della diagnosi")
    ap.add_argument("--verbose", action="store_true", help="elenca ogni occorrenza")
    args = ap.parse_args()
    sys.exit(0 if audit(get_database(), args.verbose) else 1)


if __name__ == "__main__":
    main()
