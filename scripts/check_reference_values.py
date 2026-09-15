"""
Confronta i cut-off usati nei prompt con "Valori di riferimento_lab.xlsx".

I valori di riferimento vivono in REFERENCE_VALUES (config/settings.py), che è la
fonte usata per costruire i prompt. Il file Excel è il documento autorevole dei
clinici: questo script verifica che i due coincidano, così una revisione dei
cut-off sul foglio non passa inosservata.

Uso:
    python scripts/check_reference_values.py
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from config.settings import REFERENCE_VALUES, REFERENCE_VALUES_PATH, reference_range


def parse_excel_range(raw: str) -> tuple[str, float | None, float | None]:
    """
    Interpreta le notazioni del foglio: '725-1777', '0.068 - 0.115', '<300',
    '>0,0807', '2,5-59'. La virgola decimale italiana va normalizzata.
    """
    text = str(raw).strip().replace(",", ".")
    if match := re.fullmatch(r"<\s*([\d.]+)", text):
        return "upper", None, float(match.group(1))
    if match := re.fullmatch(r">\s*([\d.]+)", text):
        return "lower", float(match.group(1)), None
    if match := re.fullmatch(r"([\d.]+)\s*-\s*([\d.]+)", text):
        return "range", float(match.group(1)), float(match.group(2))
    raise ValueError(f"notazione non riconosciuta: {raw!r}")


def main() -> None:
    path = Path(REFERENCE_VALUES_PATH)
    if not path.exists():
        print(f"File non trovato: {path}")
        sys.exit(1)

    df = pd.read_excel(path)
    marker_col, value_col = df.columns[0], df.columns[1]

    print(f"Foglio    : {path.name}")
    print(f"Marcatori : {len(df)} nel foglio, {len(REFERENCE_VALUES)} nel codice\n")

    mismatches: list[str] = []
    checked: set[str] = set()

    for _, row in df.iterrows():
        marker = str(row[marker_col]).strip()
        if marker in ("nan", ""):
            continue
        ref = REFERENCE_VALUES.get(marker)
        if ref is None:
            mismatches.append(f"{marker}: presente nel foglio ma assente nel codice")
            continue
        checked.add(marker)

        try:
            direction, low, high = parse_excel_range(row[value_col])
        except ValueError as e:
            mismatches.append(f"{marker}: {e}")
            continue

        same = (direction == ref["direction"]
                and _close(low, ref["min"]) and _close(high, ref["max"]))
        status = "ok  " if same else "DIFF"
        print(f"  [{status}] {marker:15s} foglio: {str(row[value_col]).strip():<15} "
              f"codice: {reference_range(marker)}")
        if not same:
            mismatches.append(
                f"{marker}: foglio {row[value_col]!r} != codice {reference_range(marker)!r}"
            )

    only_code = set(REFERENCE_VALUES) - checked
    if only_code:
        print(f"\n  Presenti solo nel codice (nessun riscontro sul foglio): "
              f"{', '.join(sorted(only_code))}")

    print()
    if mismatches:
        print("DIFFERENZE RILEVATE:")
        for m in mismatches:
            print(f"  ! {m}")
        sys.exit(1)
    print("Cut-off del codice allineati al foglio dei clinici.")


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < 1e-9


if __name__ == "__main__":
    main()
