"""
Costruzione dei prompt per i 3 step diagnostici.

Modello di informazione cumulativo: come un clinico che richiede esami via via
più invasivi, ogni step vede TUTTO quello che era disponibile ai precedenti più
i nuovi dati e il ragionamento già prodotto.

  Step 1  quadro clinico + contesto RAG dalla knowledge base
  Step 2  quadro clinico + output integrale dello step 1 + biomarcatori plasmatici
          e indici epato-renali
  Step 3  quadro clinico + output integrale di step 1 e 2 + plasma + liquor

Struttura di ogni prompt:
  SYSTEM: ruolo clinico, diagnosi ammesse, valori di riferimento, schema JSON
  USER:   contesto informativo dello step + istruzioni di aggiornamento
"""

from __future__ import annotations

from config.settings import (
    DIAGNOSIS_LABELS,
    REFERENCE_VALUES,
    REFERENCE_VALUES_TEXT,
    reference_hint,
)
from data.biomarkers import format_biomarker_assessment
from data.loader import CSF_BIOMARKER_COLS, PLASMA_BIOMARKER_COLS, SAFETY_LAB_COLS

_DIAGNOSIS_LIST = "\n".join(f"  - {k}: {v}" for k, v in DIAGNOSIS_LABELS.items())

_SYSTEM_BASE = f"""Sei un neurologo esperto specializzato in disturbi cognitivi e demenze.
Il tuo compito è analizzare i dati clinici del paziente e fornire una valutazione diagnostica differenziale strutturata.

DIAGNOSI POSSIBILI:
{_DIAGNOSIS_LIST}

{REFERENCE_VALUES_TEXT}

ISTRUZIONI GENERALI:
- Ragiona in modo sistematico e clinicamente rigoroso
- Esplicita il ragionamento clinico per ogni diagnosi considerata
- Livelli di probabilità: ALTA / MEDIA / BASSA / ESCLUSA
- Se un dato è mancante, segnalalo nel ragionamento senza inventare valori
- Rispondi ESCLUSIVAMENTE in formato JSON come specificato
- Non includere testo fuori dal JSON nella risposta
- In "diagnosis_assessments" valuta separatamente TUTTE le otto diagnosi usando le chiavi esatte dello schema.
- Attribuisci a ogni diagnosi un confidence_score tra 0 e 1 e una motivazione specifica.

ORDINE DI RAGIONAMENTO (vincolante):
- Compila i campi nell'ordine dello schema: prima "diagnostic_reasoning", poi gli
  otto "diagnosis_assessments", infine "primary_diagnosis".
- Scegli "primary_diagnosis" solo dopo l'analisi e usa esclusivamente uno dei
  codici brevi ammessi. Deve essere la chiave col confidence_score più alto.

COERENZA NUMERICA (vincolante):
- Gli otto confidence_score devono sommare esattamente a 1.0.
- Non usare lo stesso punteggio per tutte le diagnosi: distribuisci la probabilità
  in base all'evidenza disponibile.
- Il sistema deriverà deterministicamente ALTA/MEDIA/BASSA/ESCLUSA dai punteggi;
  non devi produrre queste etichette.

COMPLETEZZA (vincolante):
- Il JSON deve contenere tutti i campi richiesti e tutte le otto chiavi di
  "diagnosis_assessments". Tieni "diagnostic_reasoning" entro il limite indicato.
"""

_JSON_SCHEMA_STEP1 = """
{
  "step": 1,
  "patient_code": "<codice>",
  "key_clinical_features": ["<feature 1>", "<feature 2>"],
  "diagnostic_reasoning": "<analisi differenziale in massimo 8 frasi: confronta le ipotesi tra loro PRIMA di scegliere. Chiudi indicando quale diagnosi risulta la più probabile>",
  "diagnosis_assessments": {
    "AD": {"reasoning": "<valutazione AD>", "confidence_score": <0.0-1.0>},
    "AD-PPA": {"reasoning": "<valutazione AD-PPA>", "confidence_score": <0.0-1.0>},
    "MIXED": {"reasoning": "<valutazione MIXED>", "confidence_score": <0.0-1.0>},
    "VAD": {"reasoning": "<valutazione VAD>", "confidence_score": <0.0-1.0>},
    "SCD": {"reasoning": "<valutazione SCD>", "confidence_score": <0.0-1.0>},
    "LATE": {"reasoning": "<valutazione LATE>", "confidence_score": <0.0-1.0>},
    "FTD": {"reasoning": "<valutazione FTD>", "confidence_score": <0.0-1.0>},
    "PD": {"reasoning": "<valutazione PD>", "confidence_score": <0.0-1.0>}
  },
  "primary_diagnosis": "<codice con confidence_score più alto>",
  "clinical_summary": "<riassunto clinico in 2-3 frasi>",
  "missing_information": ["<info mancante 1>"],
  "rag_sources_used": [1, 3],
  "rag_evidence": [
    {"source_index": 1, "quote": "<citazione testuale dalla fonte>", "relevance": "<come sostiene il ragionamento>"}
  ],
  "step1_limitations": "<limitazioni della valutazione basata solo su dati clinici>"
}"""

_RAG_CITATION_INSTRUCTION = """
ISTRUZIONI PER LE FONTI BIBLIOGRAFICHE:
- Le fonti numerate nel contesto sono estratti integrali dalla knowledge base clinica.
- In "rag_sources_used" inserisci i NUMERI delle fonti che hai effettivamente usato (es. [1, 3]).
- In "rag_evidence" riporta, per ogni fonte usata, una CITAZIONE TESTUALE letterale
  presa dal testo della fonte (copiala, non parafrasarla) e spiega come sostiene la valutazione.
- Nel campo "reasoning" richiama le fonti con il riferimento numerico, es: "i criteri richiedono [Fonte 1]...".
- Usa solo le fonti che contengono informazione clinica utilizzabile: se una fonte non
  è pertinente al caso, NON citarla e non includerla in "rag_sources_used".
- Non attribuire a una fonte affermazioni che non vi compaiono.
"""

_JSON_SCHEMA_STEP2 = """
{
  "step": 2,
  "patient_code": "<codice>",
  "plasma_biomarker_reasoning": {
    "Plasma_Ab4240": "<significato diagnostico dello status calcolato>",
    "plasma_ptau217": "<significato diagnostico dello status calcolato>",
    "plasma_pt181": "<significato diagnostico dello status calcolato>",
    "plasma_NfL": "<significato diagnostico dello status calcolato>",
    "epato_renal_reliability": "<come i confondenti calcolati influenzano l'interpretazione>"
  },
  "diagnostic_reasoning": "<analisi differenziale aggiornata in massimo 8 frasi: come i biomarcatori plasmatici modificano ciascuna ipotesi, PRIMA di scegliere la primaria. Chiudi indicando la diagnosi più probabile>",
  "diagnosis_assessments": {
    "AD": {"reasoning": "<valutazione AD>", "confidence_score": <0.0-1.0>},
    "AD-PPA": {"reasoning": "<valutazione AD-PPA>", "confidence_score": <0.0-1.0>},
    "MIXED": {"reasoning": "<valutazione MIXED>", "confidence_score": <0.0-1.0>},
    "VAD": {"reasoning": "<valutazione VAD>", "confidence_score": <0.0-1.0>},
    "SCD": {"reasoning": "<valutazione SCD>", "confidence_score": <0.0-1.0>},
    "LATE": {"reasoning": "<valutazione LATE>", "confidence_score": <0.0-1.0>},
    "FTD": {"reasoning": "<valutazione FTD>", "confidence_score": <0.0-1.0>},
    "PD": {"reasoning": "<valutazione PD>", "confidence_score": <0.0-1.0>}
  },
  "primary_diagnosis": "<codice con confidence_score più alto>",
  "update_from_step1": "<come i biomarcatori hanno modificato la valutazione precedente>",
  "clinical_summary": "<riassunto aggiornato>"
}"""

_JSON_SCHEMA_STEP3 = """
{
  "step": 3,
  "patient_code": "<codice>",
  "csf_biomarker_reasoning": {
    "CSF_Ab42": "<significato diagnostico dello status calcolato>",
    "CSF_Ab40": "<ruolo come denominatore del rapporto>",
    "CSF_Ab4240": "<significato diagnostico dello status calcolato>",
    "CSF_ttau": "<significato diagnostico dello status calcolato>",
    "CSF_ptau": "<significato diagnostico dello status calcolato>",
    "CSF_NfL": "<significato diagnostico dello status calcolato>"
  },
  "atn_interpretation": "<interpretazione clinica del profilo ATN già calcolato dal sistema>",
  "plasma_csf_interpretation": "<interpretazione della concordanza già calcolata dal sistema>",
  "diagnostic_reasoning": "<analisi differenziale finale in massimo 8 frasi: come il profilo ATN modifica ciascuna ipotesi, PRIMA di scegliere la primaria. Chiudi indicando la diagnosi più probabile>",
  "diagnosis_assessments": {
    "AD": {"reasoning": "<valutazione AD>", "confidence_score": <0.0-1.0>},
    "AD-PPA": {"reasoning": "<valutazione AD-PPA>", "confidence_score": <0.0-1.0>},
    "MIXED": {"reasoning": "<valutazione MIXED>", "confidence_score": <0.0-1.0>},
    "VAD": {"reasoning": "<valutazione VAD>", "confidence_score": <0.0-1.0>},
    "SCD": {"reasoning": "<valutazione SCD>", "confidence_score": <0.0-1.0>},
    "LATE": {"reasoning": "<valutazione LATE>", "confidence_score": <0.0-1.0>},
    "FTD": {"reasoning": "<valutazione FTD>", "confidence_score": <0.0-1.0>},
    "PD": {"reasoning": "<valutazione PD>", "confidence_score": <0.0-1.0>}
  },
  "primary_diagnosis": "<codice con confidence_score più alto>",
  "final_clinical_summary": "<sintesi diagnostica completa>",
  "update_from_step2": "<come il liquor ha modificato la valutazione>"
}"""

_MISSING = "MANCANTE"


# ─── Blocchi informativi riutilizzabili ───────────────────────────────────────

def _fmt(value, decimals: int = 2) -> str:
    if value is None:
        return _MISSING
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _risk_factors(patient: dict) -> str:
    labels = [
        ("fam", "Familiarità per demenza"),
        ("fumo_attivo_o_pregresso", "Fumo (attivo o pregresso)"),
        ("ipertensione", "Ipertensione arteriosa"),
        ("malattia_cardiovascolare", "Malattia cardiovascolare"),
        ("diabete", "Diabete mellito"),
        ("dislipidemia", "Dislipidemia"),
    ]
    lines = []
    for key, label in labels:
        value = patient.get(key)
        if value is True:
            lines.append(f"  - {label}: SÌ")
        elif value is False:
            lines.append(f"  - {label}: NO")
        else:
            lines.append(f"  - {label}: non riportato")
    return "\n".join(lines)


def _clinical_baseline(patient: dict, terapia_formatted: str) -> str:
    """
    Quadro clinico di base: presente in TUTTI gli step.
    Un clinico non dimentica l'anamnesi quando arrivano i referti di laboratorio.
    """
    mmse = patient.get("mmse")
    edu = patient.get("anni_edu")
    return f"""=== DATI CLINICI PAZIENTE ===
Codice paziente: {patient.get('codice', 'N/D')}
Età: {_fmt(patient.get('age'), 0)} anni
Sesso: {patient.get('gender') or 'N/D'}
Anni di istruzione: {f'{edu} anni' if edu is not None else 'Non disponibile'}

ANAMNESI:
{patient.get('anamnesi') or 'Non disponibile'}

ESAME OBIETTIVO NEUROLOGICO (EON):
{patient.get('eon') or 'Non disponibile'}

VALUTAZIONE COGNITIVA:
MMSE: {f'{mmse}/30' if mmse is not None else 'Non disponibile'}

{terapia_formatted}

FATTORI DI RISCHIO:
{_risk_factors(patient)}"""


def _marker_lines(patient: dict, columns: list[str], decimals: dict[str, int]) -> str:
    """
    Una riga per marcatore: etichetta, valore, unità e intervallo di normalità.
    Etichette e cut-off vengono da REFERENCE_VALUES: nessun valore hardcoded qui,
    così i riferimenti nel prompt non possono divergere da quelli del system prompt.
    """
    lines = []
    for col in columns:
        ref = REFERENCE_VALUES.get(col, {})
        raw = patient.get(col)
        value = _fmt(raw, decimals.get(col, 2))
        # Unità e intervallo si stampano solo accanto a un valore reale:
        # "MANCANTE pg/mL" sarebbe fuorviante.
        unit = ref.get("unit", "") if raw is not None else ""
        hint = reference_hint(col) if raw is not None else ""
        lines.append(" ".join(p for p in (f"{ref.get('label', col)}:", value, unit, hint) if p))
    return "\n".join(lines)


# I rapporti Aβ42/40 e la p-tau plasmatica richiedono più decimali dei valori assoluti.
_DECIMALS = {"Plasma_Ab4240": 4, "plasma_ptau217": 4, "plasma_pt181": 4, "CSF_Ab4240": 4}


def _plasma_block(patient: dict) -> str:
    return ("=== BIOMARCATORI EMATICI ===\n"
            + _marker_lines(patient, PLASMA_BIOMARKER_COLS, _DECIMALS))


def _safety_block(patient: dict) -> str:
    return ("=== FUNZIONALITÀ EPATO-RENALE (affidabilità biomarcatori) ===\n"
            + _marker_lines(patient, SAFETY_LAB_COLS, _DECIMALS))


def _csf_block(patient: dict) -> str:
    return ("=== BIOMARCATORI LIQUOR CEREBROSPINALE (CSF) ===\n"
            + _marker_lines(patient, CSF_BIOMARKER_COLS, _DECIMALS))


def _diagnosis_line(entry: dict, marker: str = "") -> str:
    score = entry.get("confidence_score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
    probability = entry.get("probability")
    if not probability and isinstance(score, (int, float)):
        probability = (
            "ESCLUSA" if score <= 0.001 else
            "BASSA" if score < 0.2 else
            "MEDIA" if score < 0.5 else "ALTA"
        )
    head = (f"  - {entry.get('diagnosis', '?')} ({entry.get('label', '?')}): "
            f"{probability or '?'} | score={score_str}{marker}")
    reasoning = (entry.get("reasoning") or "").strip()
    return f"{head}\n      motivazione: {reasoning}" if reasoning else head


def _step_output_block(step: int, result: dict) -> str:
    """
    Rende l'output integrale di uno step precedente: diagnosi, distribuzione di
    probabilità, ragionamento per ciascuna ipotesi e valutazioni derivate.
    """
    if not result:
        return f"=== OUTPUT STEP {step}: non disponibile ==="

    primary = result.get("primary_diagnosis", {}) or {}
    differentials = result.get("differential_diagnoses", []) or []
    assessments = result.get("diagnosis_assessments")
    if isinstance(primary, str) and isinstance(assessments, dict):
        primary_code = primary
        entries = {
            code: {
                "diagnosis": code,
                "label": DIAGNOSIS_LABELS[code],
                "reasoning": (assessments.get(code) or {}).get("reasoning", ""),
                "confidence_score": (assessments.get(code) or {}).get("confidence_score"),
            }
            for code in DIAGNOSIS_LABELS
        }
        primary = entries.get(primary_code, {"diagnosis": primary_code})
        differentials = [entry for code, entry in entries.items() if code != primary_code]

    lines = [f"=== OUTPUT INTEGRALE STEP {step} ==="]
    if result.get("diagnostic_reasoning"):
        lines.append(f"Analisi differenziale globale: {result['diagnostic_reasoning']}")
    lines.extend([
        "Distribuzione diagnostica prodotta:",
        _diagnosis_line(primary, "  [PRIMARIA]"),
    ])
    lines.extend(_diagnosis_line(d) for d in differentials)

    if result.get("rag_sources_used"):
        lines.append(f"Fonti RAG usate: {result['rag_sources_used']}")
    for evidence in result.get("rag_evidence") or []:
        lines.append(
            f"  Fonte {evidence.get('source_index')}: \"{evidence.get('quote', '')}\" — "
            f"{evidence.get('relevance', '')}"
        )

    summary = result.get("clinical_summary") or result.get("final_clinical_summary")
    if summary:
        lines.append(f"\nSintesi clinica step {step}: {summary}")

    for key, label in (
        ("key_clinical_features", "Elementi clinici chiave individuati"),
        ("missing_information", "Informazioni segnalate come mancanti"),
    ):
        values = result.get(key)
        if values:
            lines.append(f"{label}: {'; '.join(str(v) for v in values)}")

    for key, label in (
        ("step1_limitations", f"Limitazioni dichiarate allo step {step}"),
        ("update_from_step1", "Aggiornamento rispetto allo step 1"),
        ("at_profile", "Profilo ATN"),
    ):
        value = result.get(key)
        if value:
            lines.append(f"{label}: {value}")

    reliability = result.get("biomarker_reliability")
    if isinstance(reliability, dict) and reliability:
        lines.append(
            "Affidabilità biomarcatori valutata allo step 2: "
            f"renale_ok={reliability.get('renal_function_ok')}, "
            f"epatica_ok={reliability.get('hepatic_function_ok')}, "
            f"affidabili={reliability.get('biomarkers_reliable')}. "
            f"{reliability.get('reliability_notes') or ''}".strip()
        )

    interpretation = result.get("plasma_biomarker_interpretation")
    if isinstance(interpretation, dict) and interpretation:
        parts = [
            f"{name}={(v or {}).get('status', '?')}"
            for name, v in interpretation.items() if isinstance(v, dict)
        ]
        if parts:
            lines.append(f"Interpretazione plasma allo step 2: {', '.join(parts)}")

    for key, label in (
        ("plasma_biomarker_reasoning", "Ragionamento sui biomarcatori plasmatici"),
        ("csf_biomarker_reasoning", "Ragionamento sui biomarcatori liquorali"),
        ("atn_interpretation", "Interpretazione ATN"),
        ("plasma_csf_interpretation", "Interpretazione concordanza plasma-liquor"),
    ):
        value = result.get(key)
        if value:
            lines.append(f"{label}: {value}")

    return "\n".join(lines)


_ANTI_ANCHORING_STEP2 = """⚠ ISTRUZIONE CRITICA: i biomarcatori seguenti sono misure OGGETTIVE e più affidabili dell'impressione clinica.
Se contraddicono la diagnosi dello step 1, DEVI aggiornare le probabilità in modo significativo.
Non cercare di confermare la valutazione precedente: il ragionamento dello step 1 ti è fornito come
documentazione del percorso diagnostico, non come conclusione da difendere. Rivaluta TUTTE le diagnosi."""

_ANTI_ANCHORING_STEP3 = """⚠ ISTRUZIONE CRITICA: i biomarcatori liquorali forniscono evidenza biologica prioritaria sulla presenza o assenza di patologia Alzheimer, ma non identificano da soli tutte le cause di demenza.
Se il profilo ATN contraddice la diagnosi corrente, aggiorna le probabilità in modo sostanziale.
Gli output degli step 1 e 2 ti sono forniti come storia del percorso diagnostico, non come conclusione
da confermare: rivaluta TUTTE le diagnosi. Un profilo AD-positivo può coesistere con patologia
vascolare o altre copatologie; un profilo AD-negativo riduce la probabilità di AD ma non dimostra
automaticamente quale diagnosi alternativa sia corretta."""


# ─── Costruttori dei prompt ───────────────────────────────────────────────────

def build_step1_prompt(
    patient: dict,
    rag_context: str,
    terapia_formatted: str,
) -> tuple[str, str]:
    """(system, user) per lo Step 1: quadro clinico + knowledge base."""
    system = (_SYSTEM_BASE + _RAG_CITATION_INSTRUCTION
              + f"\n\nSCHEMA JSON ATTESO (Step 1):\n{_JSON_SCHEMA_STEP1}")

    user = f"""=== CONTESTO DALLA KNOWLEDGE BASE (linee guida e letteratura) ===
{rag_context}

{_clinical_baseline(patient, terapia_formatted)}

---
Analizza questi dati clinici e fornisci la valutazione diagnostica differenziale nel formato JSON specificato.
In questo step NON sono disponibili biomarcatori: basa la valutazione su anamnesi, esame obiettivo,
profilo cognitivo, terapia in atto e fattori di rischio.
Il codice paziente nel JSON deve essere: {patient.get('codice', 'N/D')}"""

    return system, user


def build_step2_prompt(
    patient: dict,
    step1_result: dict,
    terapia_formatted: str,
    biomarker_assessment: dict,
) -> tuple[str, str]:
    """(system, user) per lo Step 2: quadro clinico + step 1 integrale + plasma."""
    system = _SYSTEM_BASE + f"\n\nSCHEMA JSON ATTESO (Step 2):\n{_JSON_SCHEMA_STEP2}"

    user = f"""{_step_output_block(1, step1_result)}

{_ANTI_ANCHORING_STEP2}

{_clinical_baseline(patient, terapia_formatted)}

{_plasma_block(patient)}

{_safety_block(patient)}

{format_biomarker_assessment(biomarker_assessment, 2)}

---
Aggiorna la valutazione diagnostica integrando i biomarcatori plasmatici con il quadro clinico.
Verifica prima l'affidabilità dei biomarcatori sulla base della funzionalità epato-renale.
Se tutti i biomarcatori plasmatici risultano {_MISSING}, conferma la diagnosi dello Step 1 mantenendo
le stesse probabilità e specifica nel reasoning che l'assenza di dati non permette aggiornamenti.
Il codice paziente nel JSON deve essere: {patient.get('codice', 'N/D')}"""

    return system, user


def build_step3_prompt(
    patient: dict,
    step2_result: dict | None,
    terapia_formatted: str,
    biomarker_assessment: dict,
    step1_result: dict | None = None,
) -> tuple[str, str]:
    """(system, user) per lo Step 3: quadro clinico + step 1 e 2 integrali + plasma + CSF."""
    system = _SYSTEM_BASE + f"\n\nSCHEMA JSON ATTESO (Step 3):\n{_JSON_SCHEMA_STEP3}"

    history = _step_output_block(2, step2_result)
    if step1_result:
        history = f"{_step_output_block(1, step1_result)}\n\n{history}"

    user = f"""{history}

{_ANTI_ANCHORING_STEP3}

{_clinical_baseline(patient, terapia_formatted)}

{_plasma_block(patient)}

{_safety_block(patient)}

{_csf_block(patient)}

{format_biomarker_assessment(biomarker_assessment, 3)}

---
Integra l'evidenza biologica liquorale con il quadro clinico e i dati plasmatici.
Riporta il profilo ATN calcolato dal sistema e valuta la concordanza plasma-liquor senza ricalcolare i cut-off.
Fornisci la diagnosi finale con il massimo grado di certezza possibile.
Se tutti i biomarcatori CSF risultano {_MISSING}, conferma la diagnosi dello Step 2 mantenendo le stesse
probabilità e specifica nel reasoning che l'assenza di dati non permette aggiornamenti.
Il codice paziente nel JSON deve essere: {patient.get('codice', 'N/D')}"""

    return system, user
