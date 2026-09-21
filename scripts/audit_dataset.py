"""Audit riproducibile di schema, completezza e valori sospetti del dataset."""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from config.settings import DIAGNOSIS_LABELS, REFERENCE_VALUES
from data.loader import (
    CSF_BIOMARKER_COLS,
    PLASMA_BIOMARKER_COLS,
    SAFETY_LAB_COLS,
    STEP1_CLINICAL_COLS,
    get_database,
)


def main() -> None:
    df = get_database()
    required = {"Codice", "Diagnosi_CODIFICATA", *STEP1_CLINICAL_COLS,
                *PLASMA_BIOMARKER_COLS, *SAFETY_LAB_COLS, *CSF_BIOMARKER_COLS}
    missing_columns = sorted(required - set(df.columns))
    duplicate_codes = df[df["Codice"].duplicated(keep=False)]["Codice"].astype(str).tolist()
    unknown_diagnoses = sorted(set(df["Diagnosi_CODIFICATA"].dropna()) - set(DIAGNOSIS_LABELS))

    print(f"Pazienti: {len(df)}")
    print(f"Colonne obbligatorie mancanti: {missing_columns or 'nessuna'}")
    print(f"Codici paziente duplicati: {duplicate_codes or 'nessuno'}")
    print(f"Diagnosi non previste: {unknown_diagnoses or 'nessuna'}")
    print("\nDistribuzione diagnosi:")
    print(df["Diagnosi_CODIFICATA"].value_counts().to_string())

    print("\nCompletezza biomarcatori per paziente:")
    for label, columns in (
        ("plasma", PLASMA_BIOMARKER_COLS),
        ("epato-renali", SAFETY_LAB_COLS),
        ("CSF", CSF_BIOMARKER_COLS),
    ):
        counts = df[columns].notna().sum(axis=1).value_counts().sort_index().to_dict()
        print(f"  {label:13s}: {counts}")

    print("\nValori estremi (oltre 10 volte il cut-off patologico superiore):")
    suspicious: list[tuple[str, str, float, float]] = []
    for column, ref in REFERENCE_VALUES.items():
        maximum = ref.get("max")
        if maximum is None or ref.get("direction") not in {"upper", "range"}:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        rows = df[values > maximum * 10]
        for _, row in rows.iterrows():
            suspicious.append((str(row["Codice"]), column, float(row[column]), float(maximum)))
    if suspicious:
        for code, column, value, cutoff in suspicious:
            print(f"  ! {code}: {column}={value:g}, cut-off={cutoff:g}")
    else:
        print("  nessuno")

    critical = bool(missing_columns or duplicate_codes or unknown_diagnoses)
    print("\nESITO:", "ERRORI STRUTTURALI" if critical else "struttura valida; verificare gli avvisi clinici")
    sys.exit(1 if critical else 0)


if __name__ == "__main__":
    main()
