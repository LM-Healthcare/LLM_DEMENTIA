# LLM Dementia — Diagnosis Support System

Sistema di supporto alla diagnosi differenziale delle demenze basato su LLM/SLM locali (Ollama), RAG e pipeline 3-step.

## Stack

| Componente | Tecnologia |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript + TailwindCSS |
| LLM/Embeddings | Ollama (locale, privacy-first) |
| Vector Store | ChromaDB |
| RAG | LangChain (BM25 + Semantic ensemble) |
| Data | Pandas + OpenPyXL |
| Environment | Conda (`LLM_DEMENTIA`) |

---

## Setup rapido

### 1. Crea l'ambiente conda

```bash
conda env create -f environment.yml
conda activate LLM_DEMENTIA
```

### 2. Copia il file .env

```bash
copy .env.example .env
# Modifica .env se necessario
```

### 3. Installa il frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Avvia Ollama e scarica i modelli necessari

```bash
# In un terminale separato
ollama serve

# Modello per embeddings (obbligatorio per RAG)
ollama pull nomic-embed-text

# Modello diagnostico (scegli uno)
ollama pull llama3.1:8b
```

### 5. Avvia il backend

```bash
python run.py
# API disponibile su http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 6. Avvia il frontend (sviluppo)

```bash
cd frontend
npm run dev
# UI disponibile su http://localhost:5173
```

---

## Struttura del progetto

```
├── api/                    # FastAPI backend
│   ├── main.py             # App entry point + lifespan
│   ├── schemas.py          # Pydantic models
│   ├── state.py            # Stato condiviso (RAG stores)
│   └── routers/            # Route handlers
│       ├── patients.py     # CRUD pazienti
│       ├── analysis.py     # Step 1-3 + batch
│       ├── models_mgmt.py  # Gestione modelli Ollama
│       ├── rag_mgmt.py     # Build/load RAG
│       └── results.py      # Lettura risultati
├── config/
│   └── settings.py         # Configurazione centralizzata
├── data/
│   ├── loader.py           # Caricamento e normalizzazione DB
│   ├── preprocessor.py     # Standardizzazione terapia
│   └── therapy_drugs.py    # Dizionario farmaci italiani
├── frontend/               # React + TypeScript + TailwindCSS
│   └── src/
│       ├── pages/          # Dashboard, Analisi, Batch, Risultati...
│       ├── api/client.ts   # Client API tipizzato
│       └── types/index.ts  # TypeScript interfaces
├── llm/
│   ├── ollama_client.py    # Client HTTP Ollama
│   └── prompt_builder.py   # Template prompt per 3 step
├── pipeline/
│   ├── step_runner.py      # Esecutore pipeline 3-step
│   └── evaluator.py        # Metriche concordanza
├── rag/
│   ├── document_processor.py  # PDF loading + chunking
│   ├── vector_store.py        # ChromaDB management
│   └── retriever.py           # Ensemble BM25 + semantic
├── results/                # JSON risultati batch (gitignored)
├── chroma_db/              # Vector store (gitignored)
├── environment.yml         # Conda environment
├── requirements.txt        # Pip dependencies
└── run.py                  # Startup script
```

---

## Pipeline diagnostica

```
Step 1: Dati Clinici + RAG
  Input: Anamnesi, EON, MMSE, Terapia, Fattori di rischio
  RAG: Ricerca semantica su linee guida cliniche (PDF)
  Output: Diagnosi primaria + differenziale con probabilità

Step 2: Biomarcatori Plasma  [se disponibili]
  Input: Step 1 result + Ab42/40, ptau217, ptau181, NfL
  Check: Funzionalità renale/epatica
  Output: Diagnosi aggiornata con profilo AT

Step 3: Liquor CSF  [se disponibili]
  Input: Step 2 result + CSF Ab42, ttau, ptau
  Output: Diagnosi finale con concordanza biomarker
```

---

## Primo utilizzo

1. Apri `http://localhost:5173`
2. Dashboard → verifica stato Ollama e RAG
3. Impostazioni → **Inizializza RAG** (prima volta, richiede `nomic-embed-text`)
4. Gestione Modelli → scarica il modello diagnostico preferito
5. Analisi Paziente → seleziona paziente → esegui gli step
6. Analisi Batch → elabora tutti i pazienti e genera metriche
7. Risultati → visualizza accuracy, Cohen's κ, concordanza per paziente
