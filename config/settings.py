import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))

# Knowledge base RAG: cartella unica, scansionata ricorsivamente.
DOCS_FOLDER: str = os.getenv("DOCS_FOLDER", str(BASE_DIR / "Documenti_"))

DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(BASE_DIR / "Database_FINALE_codificato.xlsx"))
REFERENCE_VALUES_PATH: str = os.getenv("REFERENCE_VALUES_PATH", str(BASE_DIR / "Valori di riferimento_lab.xlsx"))

# ─── Parametri RAG ────────────────────────────────────────────────────────────
# Strategia parent-child: i child (piccoli, precisi) servono al retrieval,
# i parent (grandi) sono il contesto restituito al modello e citato per esteso.
RAG_PARENT_CHUNK_SIZE: int = int(os.getenv("RAG_PARENT_CHUNK_SIZE", "2000"))
RAG_PARENT_CHUNK_OVERLAP: int = int(os.getenv("RAG_PARENT_CHUNK_OVERLAP", "250"))
RAG_CHILD_CHUNK_SIZE: int = int(os.getenv("RAG_CHILD_CHUNK_SIZE", "450"))
RAG_CHILD_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHILD_CHUNK_OVERLAP", "60"))

# Lunghezza minima di un chunk: sotto questa soglia viene fuso con il vicino
# invece di diventare una "fonte" a sé (evita le citazioni che sono solo titoli).
RAG_MIN_CHUNK_CHARS: int = int(os.getenv("RAG_MIN_CHUNK_CHARS", "350"))

RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "8"))
RAG_CANDIDATES_PER_QUERY: int = int(os.getenv("RAG_CANDIDATES_PER_QUERY", "10"))
# Peso del retrieval lessicale BM25 nella fusione con quello semantico (0-1).
RAG_BM25_WEIGHT: float = float(os.getenv("RAG_BM25_WEIGHT", "0.4"))
# Costante della Reciprocal Rank Fusion.
RAG_RRF_K: int = int(os.getenv("RAG_RRF_K", "60"))
# Massimo numero di fonti provenienti dalla stessa pagina (diversificazione).
RAG_MAX_PER_PAGE: int = int(os.getenv("RAG_MAX_PER_PAGE", "2"))

# Step in cui iniettare il contesto della knowledge base. Per impostazione
# predefinita solo lo Step 1: gli step successivi si basano sui biomarcatori,
# i cui valori di riferimento sono già nel system prompt.
RAG_STEPS: frozenset[int] = frozenset(
    int(s) for s in os.getenv("RAG_STEPS", "1").split(",") if s.strip()
)

# ─── Parametri di generazione LLM ─────────────────────────────────────────────
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "16384"))
LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "32768"))
LLM_TIMEOUT_S: int = int(os.getenv("LLM_TIMEOUT_S", "600"))

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
