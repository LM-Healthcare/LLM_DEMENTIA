"""
Standardizzazione della colonna TERAPIA:
  1. Parsing del testo libero (nomi commerciali, molecole, dosaggi)
  2. Lookup nel dizionario DRUG_MAPPING
  3. Fallback: marcatura come 'Non identificato'
  4. Output: stringa leggibile + lista strutturata JSON

Documentazione:
  - I farmaci sono ricercati case-insensitive con normalizzazione dei caratteri
  - I dosaggi vengono estratti ma non modificano la classificazione
  - I farmaci non trovati nel dizionario sono segnalati esplicitamente
  - La standardizzazione è completamente offline (no API esterne)
"""

from __future__ import annotations

import re
import unicodedata
import pandas as pd
from typing import Optional

from data.therapy_drugs import DRUG_MAPPING


_DOSAGE_PATTERN = re.compile(
    r"""
    \d+(?:[.,]\d+)?      # numero (es. 10, 2.5, 100)
    \s*
    (?:mg|mcg|µg|g|ml|ui|u\.i\.|iu|%|mg/ml|mg/die|cp|cps|gtt|puff)
    (?:/(?:die|gg|os|im|ev|sc|td))?   # modalità somministrazione opzionale
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SEPARATOR_PATTERN = re.compile(r"[,;/\n\+]+")


def _normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().strip()


def _extract_dosage(token: str) -> Optional[str]:
    match = _DOSAGE_PATTERN.search(token)
    return match.group(0).strip() if match else None


def _clean_token(token: str) -> str:
    cleaned = _DOSAGE_PATTERN.sub("", token)
    cleaned = re.sub(r"[\(\)\[\]]", "", cleaned)
    return cleaned.strip()


# Chiavi del dizionario ordinate dalla più lunga alla più corta: garantisce che
# "acido acetilsalicilico" vinca su "asa" e che il match non dipenda
# dall'ordine di inserimento nel dizionario.
_DRUG_KEYS_BY_LENGTH = sorted(DRUG_MAPPING, key=len, reverse=True)


def _match_drug(norm: str) -> Optional[dict]:
    """
    Cerca il farmaco nel dizionario con corrispondenza a confine di parola.

    Il match per sottostringa nuda produceva falsi positivi (una chiave breve
    come "asa" agganciava qualsiasi token che la contenesse) e falsi negativi
    dipendenti dall'ordine del dizionario.
    """
    if not norm:
        return None

    if norm in DRUG_MAPPING:
        return DRUG_MAPPING[norm]

    tokens = set(re.findall(r"[a-z0-9]+", norm))
    for key in _DRUG_KEYS_BY_LENGTH:
        if " " in key:
            if re.search(rf"\b{re.escape(key)}\b", norm):
                return DRUG_MAPPING[key]
        elif key in tokens:
            return DRUG_MAPPING[key]
    return None


def standardize_therapy(raw_text: str) -> dict:
    """
    Prende il testo grezzo della colonna TERAPIA e restituisce:
      {
        "original": str,
        "drugs": [
          {
            "original_token": str,
            "molecule": str,
            "class": str,
            "atc": str,
            "dosage": str | None,
            "status": "found" | "not_found"
          }
        ],
        "summary": str,
        "classes_present": list[str],
        "not_identified": list[str],
        "warnings": list[str]
      }
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "original": raw_text or "",
            "drugs": [],
            "summary": "Nessuna terapia riportata",
            "classes_present": [],
            "not_identified": [],
            "warnings": [],
        }

    tokens = _SEPARATOR_PATTERN.split(raw_text)
    drugs = []
    classes_present = []
    not_identified = []
    warnings = []

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        dosage = _extract_dosage(token)
        clean = _clean_token(token)
        norm = _normalize_text(clean)

        if not norm:
            continue

        matched = _match_drug(norm)

        if matched:
            entry = {
                "original_token": token,
                "molecule": matched["molecule"],
                "class": matched["class"],
                "atc": matched["atc"],
                "dosage": dosage,
                "status": "found",
            }
            if matched["class"] not in classes_present:
                classes_present.append(matched["class"])
        else:
            entry = {
                "original_token": token,
                "molecule": "Non identificato",
                "class": "Non identificato",
                "atc": None,
                "dosage": dosage,
                "status": "not_found",
            }
            not_identified.append(token)

        drugs.append(entry)

    if not_identified:
        warnings.append(
            f"Farmaci non identificati nel dizionario: {', '.join(not_identified)}. "
            f"Verificare manualmente o aggiornare il dizionario."
        )

    benzo_classes = [d["class"] for d in drugs if "benzodiazepina" in d.get("class", "").lower()]
    if benzo_classes:
        warnings.append(
            "ATTENZIONE: Benzodiazepina/e rilevata/e nella terapia. "
            "Può interferire con la valutazione cognitiva (MMSE)."
        )

    anticholin = [d for d in drugs if "ache" in d.get("class", "").lower()]
    if anticholin and len(anticholin) > 1:
        warnings.append(
            f"Rilevati {len(anticholin)} inibitori AChE. Verificare se intenzionalmente combinati."
        )

    summary_parts = []
    for d in drugs:
        part = d["molecule"]
        if d["dosage"]:
            part += f" {d['dosage']}"
        if d["status"] == "not_found":
            part = f"[?]{d['original_token']}"
        summary_parts.append(part)

    return {
        "original": raw_text,
        "drugs": drugs,
        "summary": " | ".join(summary_parts) if summary_parts else "Non disponibile",
        "classes_present": classes_present,
        "not_identified": not_identified,
        "warnings": warnings,
    }


def standardize_therapy_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge la colonna 'TERAPIA_STANDARDIZED' al DataFrame.
    """
    results = df["TERAPIA"].apply(lambda x: standardize_therapy(x) if pd.notna(x) else standardize_therapy(""))
    df = df.copy()
    df["TERAPIA_PARSED"] = results
    df["TERAPIA_STANDARDIZED"] = results.apply(lambda r: r["summary"])
    return df


def format_therapy_for_prompt(parsed: dict) -> str:
    """
    Formatta la terapia standardizzata per l'inserimento nel prompt LLM.
    """
    if not parsed or not parsed.get("drugs"):
        return "Nessuna terapia disponibile"

    lines = []
    for d in parsed["drugs"]:
        if d["status"] == "found":
            line = f"  - {d['molecule']} ({d['class']})"
            if d["dosage"]:
                line += f" — dosaggio: {d['dosage']}"
        else:
            line = f"  - [Non identificato] {d['original_token']}"
        lines.append(line)

    text = "Terapia domiciliare standardizzata:\n" + "\n".join(lines)

    if parsed.get("warnings"):
        text += "\n  ⚠ " + "\n  ⚠ ".join(parsed["warnings"])

    return text
