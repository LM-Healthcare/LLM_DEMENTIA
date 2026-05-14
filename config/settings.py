import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
DOCS_FOLDER_1: str = os.getenv("DOCS_FOLDER_1", str(BASE_DIR / "Continuum Demenze"))
DOCS_FOLDER_2: str = os.getenv("DOCS_FOLDER_2", str(BASE_DIR / "Documenti_"))

DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(BASE_DIR / "Database_FINALE_codificato.xlsx"))
REFERENCE_VALUES_PATH: str = os.getenv("REFERENCE_VALUES_PATH", str(BASE_DIR / "Valori di riferimento_lab.xlsx"))

RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "6"))
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))

RESULTS_DIR: Path = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DIAGNOSIS_LABELS = {
    "AD": "Malattia di Alzheimer",
    "AD-PPA": "Variante Logopenica della Malattia di Alzheimer",
    "MIXED": "Demenza Mista (Alzheimer + Vascolare)",
    "VAD": "Demenza Vascolare",
    "SCD": "Disturbo Soggettivo di Memoria",
    "LATE": "TDP-43 Encephalopathy (LATE)",
    "FTD": "Demenza Frontotemporale",
    "PD": "Malattia di Parkinson",
}

AD_PPA_VARIANTS = ["AD -PPA", "AD-PPA", "AD_PPA", "AD PPA"]

REFERENCE_VALUES = {
    "CSF_Ab42":     {"min": 725,    "max": 1777,   "unit": "pg/mL",        "direction": "range"},
    "CSF_Ab4240":   {"min": 0.068,  "max": 0.115,  "unit": "ratio",        "direction": "range"},
    "CSF_ttau":     {"min": 146,    "max": 410,    "unit": "pg/mL",        "direction": "range"},
    "CSF_ptau":     {"min": 2.5,    "max": 59,     "unit": "pg/mL",        "direction": "range"},
    "CSF_NfL":      {"min": None,   "max": 300,    "unit": "pg/mL",        "direction": "upper"},
    "Plasma_Ab4240":{"min": 0.0807, "max": None,   "unit": "ratio",        "direction": "lower"},
    "plasma_ptau217":{"min": None,  "max": 0.21,   "unit": "pg/mL",        "direction": "upper"},
    "plasma_pt181": {"min": None,   "max": 1.8,    "unit": "pg/mL",        "direction": "upper"},
    "plasma_NfL":   {"min": None,   "max": 8.5,    "unit": "pg/mL",        "direction": "upper"},
    "Creatinina":   {"min": None,   "max": 1.1,    "unit": "mg/dL",        "direction": "upper"},
    "AST":          {"min": 5,      "max": 34,     "unit": "U/L",          "direction": "range"},
    "ALT":          {"min": 0,      "max": 55,     "unit": "U/L",          "direction": "range"},
    "eGFR_2021":    {"min": 60,     "max": None,   "unit": "mL/min/1.73m²","direction": "lower"},
}

REFERENCE_VALUES_TEXT = """
=== VALORI DI RIFERIMENTO BIOMARCATORI ===

LIQUOR CEREBROSPINALE (CSF):
  - CSF Aβ42:        725–1777 pg/mL        (basso → patologico per AD)
  - CSF Aβ42/40:     0.068–0.115           (basso → patologico per AD)
  - CSF t-tau:       146–410 pg/mL         (alto → danno neuronale)
  - CSF p-tau:       2.5–59 pg/mL          (alto → patologia tau AD)
  - CSF NfL:         < 300 pg/mL           (alto → neurodegenerazione)

PLASMA:
  - Plasma Aβ42/40:  > 0.0807             (basso → patologico per AD)
  - Plasma p-tau217: < 0.21 pg/mL         (alto → patologico per AD)
  - Plasma p-tau181: < 1.8 pg/mL          (alto → patologico per AD)
  - Plasma NfL:      < 8.5 pg/mL          (alto → neurodegenerazione)

MARCATORI EPATICO-RENALI (affidabilità biomarcatori):
  - Creatinina:      < 1.1 mg/dL          (alto → possibile compromissione renale)
  - AST:             5–34 U/L             (alto → possibile compromissione epatica)
  - ALT:             0–55 U/L             (alto → possibile compromissione epatica)
  - eGFR:            > 60 mL/min/1.73m²  (basso → compromissione renale)

NOTA: Alterazioni di Creatinina, AST, ALT o eGFR possono ridurre l'affidabilità
dei biomarcatori plasmatici di neurodegenerazione.
"""
