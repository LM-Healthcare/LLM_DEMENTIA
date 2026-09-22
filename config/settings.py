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
RAG_CORPORA: dict[str, tuple[str, ...]] = {
    "budson": ("Budson e Solomon - A Practical Guide for Clinicians.pdf",),
    "casebook": ("Casebook_of_Dementia.pdf",),
    "both": (
        "Budson e Solomon - A Practical Guide for Clinicians.pdf",
        "Casebook_of_Dementia.pdf",
    ),
}
RAG_DEFAULT_MODE: str = os.getenv("RAG_DEFAULT_MODE", "budson").strip().lower()
if RAG_DEFAULT_MODE not in RAG_CORPORA:
    raise ValueError(f"RAG_DEFAULT_MODE non valido: {RAG_DEFAULT_MODE}")

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
RAG_CLINICAL_QUERY_WEIGHT: float = float(os.getenv("RAG_CLINICAL_QUERY_WEIGHT", "20"))
# Diversificazione delle fonti finali.
RAG_MAX_PER_PAGE: int = int(os.getenv("RAG_MAX_PER_PAGE", "2"))
RAG_MAX_PER_SOURCE: int = int(os.getenv("RAG_MAX_PER_SOURCE", "5"))

# Step in cui iniettare il contesto della knowledge base. Per impostazione
# predefinita solo lo Step 1: gli step successivi si basano sui biomarcatori,
# i cui valori di riferimento sono già nel system prompt.
RAG_STEPS: frozenset[int] = frozenset(
    int(s) for s in os.getenv("RAG_STEPS", "1").split(",") if s.strip()
)

# ─── Parametri di generazione LLM ─────────────────────────────────────────────
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "12288"))
LLM_TIMEOUT_S: int = int(os.getenv("LLM_TIMEOUT_S", "600"))
# Nuovi tentativi se il modello restituisce una risposta priva di diagnosi primaria.
LLM_RETRIES_ON_INVALID: int = int(os.getenv("LLM_RETRIES_ON_INVALID", "0"))

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
    "PD": "Demenza associata a Parkinson / spettro Lewy body (PDD/DLB)",
}

AD_PPA_VARIANTS = ["AD -PPA", "AD-PPA", "AD_PPA", "AD PPA"]

# Outlier verificati e confermati dal team clinico; restano nei dati e nel manifest.
ACCEPTED_EXTREME_VALUES: frozenset[tuple[str, str]] = frozenset({("T81", "CSF_NfL")})

# ─── Valori di riferimento dei biomarcatori ───────────────────────────────────
# Unica fonte di verità: da qui si generano sia il blocco del system prompt sia
# le annotazioni "(normale: ...)" accanto a ogni valore nei prompt. Prima i
# cut-off erano ripetuti in tre punti indipendenti e potevano divergere in
# silenzio. Corrispondono a "Valori di riferimento_lab.xlsx", con cui possono
# essere confrontati da scripts/check_reference_values.py.
REFERENCE_VALUES: dict[str, dict] = {
    "CSF_Ab42":      {"min": 725,    "max": 1777,  "unit": "pg/mL", "direction": "range",
                      "label": "CSF Aβ42", "group": "CSF", "note": "basso → patologico per AD"},
    "CSF_Ab40":      {"min": None,   "max": None,  "unit": "pg/mL", "direction": "none",
                      "label": "CSF Aβ40", "group": "CSF",
                      "note": "nessun intervallo: si interpreta come denominatore del rapporto"},
    "CSF_Ab4240":    {"min": 0.068,  "max": 0.115, "unit": "",      "direction": "range",
                      "label": "CSF Aβ42/40", "group": "CSF", "note": "basso → patologico per AD"},
    "CSF_ttau":      {"min": 146,    "max": 410,   "unit": "pg/mL", "direction": "range",
                      "label": "CSF t-tau", "group": "CSF", "note": "alto → danno neuronale"},
    "CSF_ptau":      {"min": 2.5,    "max": 59,    "unit": "pg/mL", "direction": "range",
                      "label": "CSF p-tau", "group": "CSF", "note": "alto → patologia tau AD"},
    "CSF_NfL":       {"min": None,   "max": 300,   "unit": "pg/mL", "direction": "upper",
                      "label": "CSF NfL", "group": "CSF", "note": "alto → neurodegenerazione"},
    "Plasma_Ab4240": {"min": 0.0807, "max": None,  "unit": "",      "direction": "lower",
                      "label": "Plasma Aβ42/40", "group": "PLASMA", "note": "basso → patologico per AD"},
    "plasma_ptau217":{"min": None,   "max": 0.21,  "unit": "pg/mL", "direction": "upper",
                      "label": "Plasma p-tau217", "group": "PLASMA", "note": "alto → patologico per AD"},
    "plasma_pt181":  {"min": None,   "max": 1.8,   "unit": "pg/mL", "direction": "upper",
                      "label": "Plasma p-tau181", "group": "PLASMA", "note": "alto → patologico per AD"},
    "plasma_NfL":    {"min": None,   "max": 8.5,   "unit": "pg/mL", "direction": "upper",
                      "label": "Plasma NfL", "group": "PLASMA", "note": "alto → neurodegenerazione"},
    "Creatinina":    {"min": None,   "max": 1.1,   "unit": "mg/dL", "direction": "upper",
                      "label": "Creatinina", "group": "SAFETY", "note": "alto → possibile compromissione renale"},
    "AST":           {"min": 5,      "max": 34,    "unit": "U/L",   "direction": "range",
                      "label": "AST", "group": "SAFETY", "note": "alto → possibile compromissione epatica"},
    "ALT":           {"min": 0,      "max": 55,    "unit": "U/L",   "direction": "range",
                      "label": "ALT", "group": "SAFETY", "note": "alto → possibile compromissione epatica"},
    "eGFR_2021":     {"min": 60,     "max": None,  "unit": "mL/min/1.73m²", "direction": "lower",
                      "label": "eGFR", "group": "SAFETY", "note": "basso → compromissione renale"},
}

def _fmt_number(value: float) -> str:
    return f"{value:g}"


def reference_range(col: str) -> str:
    """
    Intervallo di normalità di un marcatore, es. '725-1777 pg/mL' o '<0.21 pg/mL'.
    Stringa vuota per i marcatori privi di intervallo (direction 'none').
    """
    ref = REFERENCE_VALUES.get(col)
    if ref is None or ref["direction"] == "none":
        return ""
    direction, unit = ref["direction"], ref["unit"]
    if direction == "range":
        body = f"{_fmt_number(ref['min'])}-{_fmt_number(ref['max'])}"
    elif direction == "upper":
        body = f"<{_fmt_number(ref['max'])}"
    else:
        body = f">{_fmt_number(ref['min'])}"
    return f"{body} {unit}".strip()


def reference_hint(col: str) -> str:
    """Annotazione da affiancare al valore nel prompt, es. '(normale: <0.21 pg/mL)'."""
    rng = reference_range(col)
    return f"(normale: {rng})" if rng else ""


def _build_reference_text() -> str:
    groups = (
        ("LIQUOR CEREBROSPINALE (CSF)", "CSF"),
        ("PLASMA", "PLASMA"),
        ("MARCATORI EPATICO-RENALI (affidabilità biomarcatori)", "SAFETY"),
    )
    lines = ["", "=== VALORI DI RIFERIMENTO BIOMARCATORI ===", ""]
    for title, group in groups:
        lines.append(f"{title}:")
        for col, ref in REFERENCE_VALUES.items():
            if ref["group"] == group and reference_range(col):
                lines.append(f"  - {ref['label']+':':<18} {reference_range(col):<22} ({ref['note']})")
        lines.append("")
    lines.append("NOTA: Alterazioni di Creatinina, AST, ALT o eGFR possono ridurre "
                 "l'affidabilità\ndei biomarcatori plasmatici di neurodegenerazione.")
    return "\n".join(lines)


REFERENCE_VALUES_TEXT = _build_reference_text()
