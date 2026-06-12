"""
Estrae tutti i farmaci/token non identificati dal dizionario
presenti nella colonna TERAPIA del database.

Uso:
    python scripts/extract_unidentified_drugs.py

Output:
    FARMACI_NON_IDENTIFICATI.txt  (nella root del progetto)
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data.loader import load_database
from data.preprocessor import standardize_therapy


def main():
    print("Caricamento database...")
    df = load_database()

    all_not_identified: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        codice = str(row.get("Codice", "N/D"))
        terapia_raw = row.get("TERAPIA")
        if not isinstance(terapia_raw, str) or not terapia_raw.strip():
            continue

        parsed = standardize_therapy(terapia_raw)
        for token in parsed.get("not_identified", []):
            token_clean = token.strip()
            if token_clean:
                all_not_identified.append((token_clean, codice))

    if not all_not_identified:
        print("Nessun farmaco non identificato trovato.")
        return

    counter = Counter(token for token, _ in all_not_identified)

    unique_tokens = sorted(counter.keys(), key=lambda t: (-counter[t], t.lower()))

    lines = [
        "=" * 70,
        "FARMACI / TOKEN NON IDENTIFICATI NEL DIZIONARIO TERAPEUTICO",
        "=" * 70,
        f"Totale token unici non identificati: {len(unique_tokens)}",
        f"Totale occorrenze (su tutti i pazienti): {len(all_not_identified)}",
        "",
        f"{'TOKEN (originale dal DB)':<40}  {'N. occorrenze':<15}  PAZIENTI",
        "-" * 70,
    ]

    for token in unique_tokens:
        patients_with = sorted({cod for t, cod in all_not_identified if t == token})
        lines.append(
            f"{token:<40}  {counter[token]:<15}  {', '.join(patients_with)}"
        )

    lines += [
        "",
        "=" * 70,
        "LISTA ALFABETICA (solo token, per invio ai colleghi medici)",
        "=" * 70,
    ]
    for token in sorted(unique_tokens, key=str.lower):
        lines.append(f"  - {token}")

    out_path = Path(__file__).parent.parent / "FARMACI_NON_IDENTIFICATI.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFile scritto: {out_path}")
    print(f"Token unici non identificati: {len(unique_tokens)}")
    print(f"Totale occorrenze: {len(all_not_identified)}")


if __name__ == "__main__":
    main()
