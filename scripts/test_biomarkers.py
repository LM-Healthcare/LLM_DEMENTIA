"""Test di regressione dell'interpretazione deterministica dei biomarcatori."""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data.biomarkers import assess_marker, compute_biomarker_assessment
from data.loader import get_database, get_patient_record

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok' if condition else 'FALLITO'}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main() -> None:
    print("\nCut-off individuali")
    check("plasma Aβ42/40 sotto soglia patologico",
          assess_marker("Plasma_Ab4240", 0.0806)["status"] == "PATOLOGICO")
    check("plasma Aβ42/40 alla soglia normale",
          assess_marker("Plasma_Ab4240", 0.0807)["status"] == "NORMALE")
    check("p-tau217 sopra soglia patologico",
          assess_marker("plasma_ptau217", 0.211)["status"] == "PATOLOGICO")
    check("p-tau217 alla soglia normale",
          assess_marker("plasma_ptau217", 0.21)["status"] == "NORMALE")
    check("CSF ratio sotto soglia A+",
          assess_marker("CSF_Ab4240", 0.0679)["pathology_positive"] is True)
    check("CSF p-tau sopra soglia T+",
          assess_marker("CSF_ptau", 59.1)["pathology_positive"] is True)
    check("valore NaN mancante", assess_marker("CSF_ptau", float("nan"))["status"] == "MANCANTE")

    print("\nCaso reale T2")
    record = get_patient_record(get_database(), "T2")
    assessment = compute_biomarker_assessment(record, 3)
    check("plasma Aβ42/40 patologico",
          assessment["plasma"]["Plasma_Ab4240"]["status"] == "PATOLOGICO")
    check("funzione renale OK", assessment["safety"]["renal_status"] == "OK")
    check("funzione epatica OK", assessment["safety"]["hepatic_status"] == "OK")
    check("profilo T2 A-T-N-", assessment["atn"]["profile"] == "A-T-N-",
          assessment["atn"]["profile"])
    check("discordanza amiloide plasma/liquor",
          assessment["plasma_csf_concordance"]["amyloid"]["status"] == "DISCORDANT")

    print("\nRegole ATN")
    ratio_priority = {
        "CSF_Ab42": 500,
        "CSF_Ab40": 10000,
        "CSF_Ab4240": 0.08,
        "CSF_ttau": 500,
        "CSF_ptau": 70,
        "CSF_NfL": None,
    }
    profile = compute_biomarker_assessment(ratio_priority, 3)["atn"]["profile"]
    check("rapporto normale prioritario su Aβ42 basso", profile.startswith("A-"), profile)

    ratio_missing = dict(ratio_priority, CSF_Ab4240=None)
    profile = compute_biomarker_assessment(ratio_missing, 3)["atn"]["profile"]
    check("fallback ad Aβ42 se rapporto mancante", profile.startswith("A+"), profile)

    print()
    if failures:
        print(f"{len(failures)} CONTROLLI FALLITI: {', '.join(failures)}")
        sys.exit(1)
    print("Tutti i controlli dei biomarcatori superati.")


if __name__ == "__main__":
    main()
