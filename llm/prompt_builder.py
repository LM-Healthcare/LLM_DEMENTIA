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

from config.settings import REFERENCE_VALUES_TEXT, DIAGNOSIS_LABELS

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
- IMPORTANTE: nel campo "diagnosis" usa SOLO il codice breve esatto dalla lista sopra (es. "VAD", "AD", "FTD", "PD"), NON il nome esteso della malattia
- OBBLIGATORIO: in "differential_diagnoses" includi TUTTE le diagnosi possibili non scelte come primaria, ciascuna con confidence_score esplicito
- I confidence_score di primary_diagnosis + tutti i differential_diagnoses devono sommare a circa 1.0
"""

_JSON_SCHEMA_STEP1 = """
{
  "step": 1,
  "patient_code": "<codice>",
  "primary_diagnosis": {
    "diagnosis": "<codice esatto dalla lista: AD|VAD|SCD|PD|FTD|Mixed|LATE|AD-PPA>",
    "label": "<etichetta>",
    "probability": "ALTA|MEDIA|BASSA",
    "confidence_score": <0.0-1.0>,
    "reasoning": "<spiegazione clinica dettagliata>"
  },
  "differential_diagnoses": [
    {
      "diagnosis": "<codice>",
      "label": "<etichetta>",
      "probability": "ALTA|MEDIA|BASSA|ESCLUSA",
      "confidence_score": <0.0-1.0>,
      "reasoning": "<spiegazione>"
    }
  ],
  "key_clinical_features": ["<feature 1>", "<feature 2>"],
  "missing_information": ["<info mancante 1>"],
  "rag_sources_used": [1, 3],
  "rag_evidence": [
    {"source_index": 1, "quote": "<citazione testuale dalla fonte>", "relevance": "<come sostiene il ragionamento>"}
  ],
  "clinical_summary": "<riassunto clinico in 2-3 frasi>",
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
  "biomarker_reliability": {
    "renal_function_ok": true|false,
    "hepatic_function_ok": true|false,
    "biomarkers_reliable": true|false,
    "reliability_notes": "<note>"
  },
  "plasma_biomarker_interpretation": {
    "Plasma_Ab4240": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "plasma_ptau217": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "plasma_pt181": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "plasma_NfL": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"}
  },
  "primary_diagnosis": {
    "diagnosis": "<codice>",
    "label": "<etichetta>",
    "probability": "ALTA|MEDIA|BASSA",
    "confidence_score": <0.0-1.0>,
    "reasoning": "<ragionamento aggiornato con i biomarcatori>"
  },
  "differential_diagnoses": [
    {
      "diagnosis": "<codice>",
      "label": "<etichetta>",
      "probability": "ALTA|MEDIA|BASSA|ESCLUSA",
      "confidence_score": <0.0-1.0>,
      "reasoning": "<ragionamento>"
    }
  ],
  "update_from_step1": "<come i biomarcatori hanno modificato la valutazione precedente>",
  "clinical_summary": "<riassunto aggiornato>"
}"""

_JSON_SCHEMA_STEP3 = """
{
  "step": 3,
  "patient_code": "<codice>",
  "biomarker_reliability": {
    "renal_function_ok": true|false,
    "hepatic_function_ok": true|false,
    "biomarkers_reliable": true|false,
    "reliability_notes": "<note>"
  },
  "csf_biomarker_interpretation": {
    "CSF_Ab42": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "CSF_Ab4240": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "CSF_ttau": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "CSF_ptau": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "CSF_NfL": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"}
  },
  "at_profile": "<A+T+N+ | A+T-N+ | A-T-N+ | ... (classificazione ATN)>",
  "plasma_csf_concordance": "<concordanza o discordanza tra plasma e liquor, con interpretazione>",
  "primary_diagnosis": {
    "diagnosis": "<codice>",
    "label": "<etichetta>",
    "probability": "ALTA|MEDIA|BASSA",
    "confidence_score": <0.0-1.0>,
    "reasoning": "<ragionamento finale con tutti i dati>"
  },
  "differential_diagnoses": [
    {
      "diagnosis": "<codice>",
      "label": "<etichetta>",
      "probability": "ALTA|MEDIA|BASSA|ESCLUSA",
      "confidence_score": <0.0-1.0>,
      "reasoning": "<ragionamento>"
    }
  ],
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


def _plasma_block(patient: dict) -> str:
    return f"""=== BIOMARCATORI EMATICI ===
Plasma Aβ42/40: {_fmt(patient.get('Plasma_Ab4240'), 4)} (normale: >0.0807)
Plasma p-tau217: {_fmt(patient.get('plasma_ptau217'), 4)} pg/mL (normale: <0.21)
Plasma p-tau181: {_fmt(patient.get('plasma_pt181'), 4)} pg/mL (normale: <1.8)
Plasma NfL: {_fmt(patient.get('plasma_NfL'), 2)} pg/mL (normale: <8.5)"""


def _safety_block(patient: dict) -> str:
    return f"""=== FUNZIONALITÀ EPATO-RENALE (affidabilità biomarcatori) ===
Creatinina: {_fmt(patient.get('Creatinina'))} mg/dL (normale: <1.1)
AST: {_fmt(patient.get('AST'))} U/L (normale: 5-34)
ALT: {_fmt(patient.get('ALT'))} U/L (normale: 0-55)
eGFR: {_fmt(patient.get('eGFR_2021'))} mL/min/1.73m² (normale: >60)"""


def _csf_block(patient: dict) -> str:
    return f"""=== BIOMARCATORI LIQUOR CEREBROSPINALE (CSF) ===
CSF Aβ42: {_fmt(patient.get('CSF_Ab42'))} pg/mL (normale: 725-1777)
CSF Aβ40: {_fmt(patient.get('CSF_Ab40'))} pg/mL
CSF Aβ42/40: {_fmt(patient.get('CSF_Ab4240'), 4)} (normale: 0.068-0.115)
CSF t-tau: {_fmt(patient.get('CSF_ttau'))} pg/mL (normale: 146-410)
CSF p-tau: {_fmt(patient.get('CSF_ptau'))} pg/mL (normale: 2.5-59)
CSF NfL: {_fmt(patient.get('CSF_NfL'))} pg/mL (normale: <300)"""


def _diagnosis_line(entry: dict, marker: str = "") -> str:
    score = entry.get("confidence_score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
    head = (f"  - {entry.get('diagnosis', '?')} ({entry.get('label', '?')}): "
            f"{entry.get('probability', '?')} | score={score_str}{marker}")
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

    lines = [f"=== OUTPUT INTEGRALE STEP {step} ===",
             "Distribuzione diagnostica prodotta:",
             _diagnosis_line(primary, "  [PRIMARIA]")]
    lines.extend(_diagnosis_line(d) for d in differentials)

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

    return "\n".join(lines)


_ANTI_ANCHORING_STEP2 = """⚠ ISTRUZIONE CRITICA: i biomarcatori seguenti sono misure OGGETTIVE e più affidabili dell'impressione clinica.
Se contraddicono la diagnosi dello step 1, DEVI aggiornare le probabilità in modo significativo.
Non cercare di confermare la valutazione precedente: il ragionamento dello step 1 ti è fornito come
documentazione del percorso diagnostico, non come conclusione da difendere. Rivaluta TUTTE le diagnosi."""

_ANTI_ANCHORING_STEP3 = """⚠ ISTRUZIONE CRITICA: i biomarcatori liquorali (CSF) sono il GOLD STANDARD diagnostico per le demenze
e hanno PESO PRIORITARIO su qualsiasi valutazione precedente.
Se il profilo ATN contraddice la diagnosi corrente, DEVI correggerla con alta fiducia.
Gli output degli step 1 e 2 ti sono forniti per intero come storia del percorso diagnostico, non come
conclusione da confermare: rivaluta TUTTE le diagnosi in modo indipendente.
Un profilo A+T+N+ esclude quasi certamente le diagnosi non-AD."""


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
) -> tuple[str, str]:
    """(system, user) per lo Step 2: quadro clinico + step 1 integrale + plasma."""
    system = _SYSTEM_BASE + f"\n\nSCHEMA JSON ATTESO (Step 2):\n{_JSON_SCHEMA_STEP2}"

    user = f"""{_step_output_block(1, step1_result)}

{_ANTI_ANCHORING_STEP2}

{_clinical_baseline(patient, terapia_formatted)}

{_plasma_block(patient)}

{_safety_block(patient)}

---
Aggiorna la valutazione diagnostica integrando i biomarcatori plasmatici con il quadro clinico.
Verifica prima l'affidabilità dei biomarcatori sulla base della funzionalità epato-renale.
Se tutti i biomarcatori plasmatici risultano {_MISSING}, conferma la diagnosi dello Step 1 mantenendo
le stesse probabilità e specifica nel reasoning che l'assenza di dati non permette aggiornamenti.
Il codice paziente nel JSON deve essere: {patient.get('codice', 'N/D')}"""

    return system, user


def build_step3_prompt(
    patient: dict,
    step2_result: dict,
    terapia_formatted: str,
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

---
Integra i biomarcatori liquorali (gold standard diagnostico) con il quadro clinico e i dati plasmatici.
Classifica il profilo ATN (Amyloid/Tau/Neurodegeneration) e valuta la concordanza plasma-liquor.
Fornisci la diagnosi finale con il massimo grado di certezza possibile.
Se tutti i biomarcatori CSF risultano {_MISSING}, conferma la diagnosi dello Step 2 mantenendo le stesse
probabilità e specifica nel reasoning che l'assenza di dati non permette aggiornamenti.
Il codice paziente nel JSON deve essere: {patient.get('codice', 'N/D')}"""

    return system, user
