"""
Estrae tutti i farmaci/brand name non identificati nel dizionario terapeutico,
deduplicati e filtrati, con colonna vuota da compilare (molecola/classe).

Uso:
    python scripts/extract_unidentified_drugs.py

Output:
    FARMACI_NON_IDENTIFICATI.txt  (nella root del progetto)

Istruzioni per i colleghi medici:
    Per ogni riga, completare la colonna MOLECOLA / CLASSE con:
      - la molecola attiva (se è un brand name)
      - il brand name italiano più comune (se è già una molecola)
    Il file aggiornato può essere usato per aggiornare il dizionario DRUG_MAPPING.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import get_database as load_database
from data.preprocessor import standardize_therapy


_NOISE_PATTERN = re.compile(r'^[\s\-\./\\|(){}\[\]_=+*#@!?,;:\'\"<>%&]+$')


def _is_noise(token: str) -> bool:
    """Restituisce True se il token è solo rumore (separatore, numero puro, troppo corto)."""
    clean = token.strip()
    if not clean:
        return True
    if _NOISE_PATTERN.match(clean):
        return True
    if len(clean) <= 2:
        return True
    if clean.replace('.', '').replace(',', '').isdigit():
        return True
    return False


def main():
    print("Caricamento database...")
    df = load_database()

    raw_occurrences: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        codice = str(row.get("Codice", "N/D"))
        terapia_raw = row.get("TERAPIA")
        if not isinstance(terapia_raw, str) or not terapia_raw.strip():
            continue

        parsed = standardize_therapy(terapia_raw)
        for token in parsed.get("not_identified", []):
            token_clean = token.strip()
            if token_clean and not _is_noise(token_clean):
                raw_occurrences.append((token_clean, codice))

    if not raw_occurrences:
        print("Nessun farmaco non identificato trovato.")
        return

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for token, codice in raw_occurrences:
        grouped[token.lower()].append((token, codice))

    entries = []
    for lower_key, occurrences in grouped.items():
        token_counter = Counter(t for t, _ in occurrences)
        best_casing = token_counter.most_common(1)[0][0]
        n = len(occurrences)
        patients = sorted({cod for _, cod in occurrences})
        entries.append((best_casing, n, patients))

    entries.sort(key=lambda x: (-x[1], x[0].lower()))

    col1 = max(len(e[0]) for e in entries) + 2
    col1 = max(col1, 35)
    header_sep = "=" * (col1 + 70)

    lines = [
        header_sep,
        "FARMACI / BRAND NAME NON IDENTIFICATI NEL DIZIONARIO TERAPEUTICO",
        header_sep,
        "",
        "ISTRUZIONI PER I COLLEGHI MEDICI:",
        "  Per ogni riga completare la colonna MOLECOLA / CLASSE con:",
        "    - la molecola attiva (se il token e' un nome commerciale)",
        "    - la classe farmacologica (se gia' si conosce la molecola)",
        "  Il file aggiornato servira' ad arricchire il dizionario interno.",
        "",
        f"  Farmaci unici non identificati : {len(entries)}",
        f"  Occorrenze totali nel database  : {len(raw_occurrences)}",
        "",
        header_sep,
        f"  {'FARMACO / BRAND NAME (dal DB)':<{col1}}  {'N.':<4}  {'PAZIENTI':<30}  MOLECOLA / CLASSE",
        header_sep,
    ]

    for name, n, patients in entries:
        patients_str = ', '.join(patients)
        if len(patients_str) > 28:
            patients_str = patients_str[:25] + '...'
        lines.append(
            f"  {name:<{col1}}  {n:<4}  {patients_str:<30}  ___________________________"
        )

    lines += [
        "",
        header_sep,
        "LISTA ALFABETICA SEMPLICE (copia-incolla per invio ai colleghi)",
        header_sep,
    ]
    for name, _, _ in sorted(entries, key=lambda x: x[0].lower()):
        lines.append(f"  {name}")

    out_path = Path(__file__).parent.parent / "FARMACI_NON_IDENTIFICATI.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFile scritto: {out_path}")
    print(f"Farmaci unici (deduplicati): {len(entries)}")
    print(f"Occorrenze totali: {len(raw_occurrences)}")


if __name__ == "__main__":
    main()
