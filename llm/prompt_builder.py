"""
Costruzione dei prompt per i 3 step diagnostici.

Struttura:
  - SYSTEM: ruolo clinico + valori di riferimento iniettati staticamente
  - USER: dati paziente + contesto RAG (step 1) + istruzioni output JSON
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
  "clinical_summary": "<riassunto clinico in 2-3 frasi>",
  "step1_limitations": "<limitazioni della valutazione basata solo su dati clinici>"
}"""

_RAG_CITATION_INSTRUCTION = """
ISTRUZIONI PER LE FONTI BIBLIOGRAFICHE:
- Nel campo "rag_sources_used" inserisci i NUMERI delle fonti (es. [1, 3]) che hai effettivamente usato per supportare il ragionamento.
- Nel campo "reasoning" cita le fonti con il riferimento numerico tra parentesi quadre, es: "Secondo i criteri NIA-AA [Fonte 1], la presenza di..."
- Cita solo le fonti che hanno concretamente influenzato la diagnosi.
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
  "csf_biomarker_interpretation": {
    "CSF_Ab42": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "CSF_Ab4240": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "CSF_ttau": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "CSF_ptau": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"},
    "CSF_NfL": {"value": <valore|null>, "status": "NORMALE|PATOLOGICO|MANCANTE", "interpretation": "<note>"}
  },
  "at_profile": "<A+T+N+ | A+T-N+ | A-T-N+ | ... (classificazione ATN)>",
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


def build_step1_prompt(
    patient: dict,
    rag_context: str,
    terapia_formatted: str,
) -> tuple[str, str]:
    """
    Restituisce (system_prompt, user_prompt) per lo Step 1.
    """
    system = _SYSTEM_BASE + _RAG_CITATION_INSTRUCTION + f"\n\nSCHEMA JSON ATTESO (Step 1):\n{_JSON_SCHEMA_STEP1}"

    risk_factors = []
    if patient.get("fam"):
        risk_factors.append("Familiarità per demenza: SÌ")
    if patient.get("fumo_attivo_o_pregresso"):
        risk_factors.append("Fumo (attivo o pregresso): SÌ")
    if patient.get("ipertensione"):
        risk_factors.append("Ipertensione arteriosa: SÌ")
    if patient.get("malattia_cardiovascolare"):
        risk_factors.append("Malattia cardiovascolare: SÌ")
    if patient.get("diabete"):
        risk_factors.append("Diabete mellito: SÌ")
    if patient.get("dislipidemia"):
        risk_factors.append("Dislipidemia: SÌ")

    mmse = patient.get("mmse")
    mmse_str = f"{mmse}/30" if mmse is not None else "Non disponibile"

    edu = patient.get("anni_edu")
    edu_str = f"{edu} anni" if edu is not None else "Non disponibile"

    user = f"""=== CONTESTO DALLA KNOWLEDGE BASE (linee guida e letteratura) ===
{rag_context}

=== DATI CLINICI PAZIENTE ===
Codice paziente: {patient.get('codice', 'N/D')}
Età: {patient.get('age', 'N/D')} anni
Sesso: {patient.get('gender', 'N/D')}
Anni di istruzione: {edu_str}

ANAMNESI:
{patient.get('anamnesi', 'Non disponibile')}

ESAME OBIETTIVO NEUROLOGICO (EON):
{patient.get('eon', 'Non disponibile')}

VALUTAZIONE COGNITIVA:
MMSE: {mmse_str}

{terapia_formatted}

FATTORI DI RISCHIO:
{chr(10).join(risk_factors) if risk_factors else 'Nessun fattore di rischio riportato'}

---
Analizza questi dati clinici e fornisci la valutazione diagnostica differenziale nel formato JSON specificato.
Il codice paziente nel JSON deve essere: {patient.get('codice', 'N/D')}"""

    return system, user


def build_step2_prompt(
    patient: dict,
    step1_result: dict,
    terapia_formatted: str,
) -> tuple[str, str]:
    """
    Restituisce (system_prompt, user_prompt) per lo Step 2.
    Incorpora il risultato dello step 1 per mantenere il ragionamento progressivo.
    """
    import json as _json

    system = _SYSTEM_BASE + f"\n\nSCHEMA JSON ATTESO (Step 2):\n{_JSON_SCHEMA_STEP2}"

    prev_diag = step1_result.get("primary_diagnosis", {})
    prev_differentials = step1_result.get("differential_diagnoses", [])
    prev_summary = step1_result.get("clinical_summary", "Non disponibile")

    def fmt_val(v):
        return f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "MANCANTE")

    def fmt_diag_score(d: dict) -> str:
        score = d.get('confidence_score')
        score_str = f"{score:.2f}" if isinstance(score, float) else "?"
        return f"  - {d.get('diagnosis','?')} ({d.get('label','?')}): {d.get('probability','?')} | score={score_str}"

    all_diags_lines = [fmt_diag_score(prev_diag) + " [CORRENTE]"] + [fmt_diag_score(d) for d in prev_differentials]
    all_diags_str = "\n".join(all_diags_lines)

    user = f"""=== DISTRIBUZIONE DIAGNOSTICA DA STEP 1 (dati clinici) ===
Tutte le diagnosi con probabilità assegnate:
{all_diags_str}

Sintesi clinica: {prev_summary}

⚠ ISTRUZIONE CRITICA: I biomarcatori seguenti sono misure OGGETTIVE e più affidabili dell'impressione clinica.
Se i biomarcatori contraddicono la diagnosi corrente, DEVI AGGIORNARE le probabilità in modo significativo.
Non cercare di confermare la diagnosi precedente: rivaluta in modo INDIPENDENTE tutte le diagnosi sulla base dei nuovi dati.

=== DATI PAZIENTE ===
Codice: {patient.get('codice', 'N/D')} | Età: {patient.get('age')} | Sesso: {patient.get('gender')}
MMSE: {patient.get('mmse', 'N/D')}/30

{terapia_formatted}

=== BIOMARCATORI EMATICI ===
Plasma Aβ42/40: {fmt_val(patient.get('Plasma_Ab4240'))} (normale: >0.0807)
Plasma p-tau217: {fmt_val(patient.get('plasma_ptau217'))} pg/mL (normale: <0.21)
Plasma p-tau181: {fmt_val(patient.get('plasma_pt181'))} pg/mL (normale: <1.8)
Plasma NfL: {fmt_val(patient.get('plasma_NfL'))} pg/mL (normale: <8.5)

=== FUNZIONALITÀ EPATO-RENALE (affidabilità biomarcatori) ===
Creatinina: {fmt_val(patient.get('Creatinina'))} mg/dL (normale: <1.1)
AST: {fmt_val(patient.get('AST'))} U/L (normale: 5-34)
ALT: {fmt_val(patient.get('ALT'))} U/L (normale: 0-55)
eGFR: {fmt_val(patient.get('eGFR_2021'))} mL/min/1.73m² (normale: >60)

---
Aggiorna la valutazione diagnostica incorporando i biomarcatori plasmatici.
Verifica prima l'affidabilità dei biomarcatori sulla base della funzionalità epato-renale.
Se tutti i biomarcatori plasmatici riportano MANCANTE, conferma la diagnosi dello Step 1 mantenendo le stesse probabilità e specifica nel reasoning che l'assenza di dati non permette aggiornamenti.
Il codice paziente nel JSON deve essere: {patient.get('codice', 'N/D')}"""

    return system, user


def build_step3_prompt(
    patient: dict,
    step2_result: dict,
    terapia_formatted: str,
) -> tuple[str, str]:
    """
    Restituisce (system_prompt, user_prompt) per lo Step 3.
    """
    system = _SYSTEM_BASE + f"\n\nSCHEMA JSON ATTESO (Step 3):\n{_JSON_SCHEMA_STEP3}"

    prev_diag = step2_result.get("primary_diagnosis", {})
    prev_differentials = step2_result.get("differential_diagnoses", [])
    prev_summary = step2_result.get("clinical_summary", step2_result.get("final_clinical_summary", "Non disponibile"))

    def fmt_val(v):
        return f"{v:.2f}" if isinstance(v, float) else (str(v) if v is not None else "MANCANTE")

    def fmt_diag_score(d: dict) -> str:
        score = d.get('confidence_score')
        score_str = f"{score:.2f}" if isinstance(score, float) else "?"
        return f"  - {d.get('diagnosis','?')} ({d.get('label','?')}): {d.get('probability','?')} | score={score_str}"

    all_diags_lines = [fmt_diag_score(prev_diag) + " [CORRENTE]"] + [fmt_diag_score(d) for d in prev_differentials]
    all_diags_str = "\n".join(all_diags_lines)

    user = f"""=== DISTRIBUZIONE DIAGNOSTICA DA STEP 2 (clinica + plasma) ===
Tutte le diagnosi con probabilità assegnate:
{all_diags_str}

Sintesi: {prev_summary}

⚠ ISTRUZIONE CRITICA: I biomarcatori liquorali (CSF) sono il GOLD STANDARD diagnostico per le demenze.
Devono avere PESO PRIORITARIO rispetto a qualsiasi valutazione precedente.
Se il profilo ATN contraddict la diagnosi corrente, DEVI CORREGGERLA con alta fiducia.
Rivaluta TUTTE le diagnosi in modo indipendente: un profilo A+T+N+ esclude quasi certamente le diagnosi non-AD.

=== DATI PAZIENTE ===
Codice: {patient.get('codice', 'N/D')} | Età: {patient.get('age')} | Sesso: {patient.get('gender')}
MMSE: {patient.get('mmse', 'N/D')}/30

=== BIOMARCATORI LIQUOR CEREBROSPINALE (CSF) ===
CSF Aβ42: {fmt_val(patient.get('CSF_Ab42'))} pg/mL (normale: 725-1777)
CSF Aβ40: {fmt_val(patient.get('CSF_Ab40'))} pg/mL
CSF Aβ42/40: {fmt_val(patient.get('CSF_Ab4240'))} (normale: 0.068-0.115)
CSF t-tau: {fmt_val(patient.get('CSF_ttau'))} pg/mL (normale: 146-410)
CSF p-tau: {fmt_val(patient.get('CSF_ptau'))} pg/mL (normale: 2.5-59)
CSF NfL: {fmt_val(patient.get('CSF_NfL'))} pg/mL (normale: <300)

---
Integra i biomarcatori liquorali (gold standard diagnostico) nella valutazione.
Classifica il profilo ATN (Amyloid/Tau/Neurodegeneration).
Fornisci la diagnosi finale con il massimo grado di certezza possibile.
Se tutti i biomarcatori CSF riportano MANCANTE, conferma la diagnosi dello Step 2 mantenendo le stesse probabilità e specifica nel reasoning che l'assenza di dati non permette aggiornamenti.
Il codice paziente nel JSON deve essere: {patient.get('codice', 'N/D')}"""

    return system, user
