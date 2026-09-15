# LLM Dementia — Diagnosis Support System

Sistema di supporto alla diagnosi differenziale delle demenze basato su LLM/SLM locali (Ollama), RAG e pipeline 3-step.

## Stack

| Componente | Tecnologia |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript + TailwindCSS |
| LLM/Embeddings | Ollama (locale, privacy-first) — `bge-m3` per gli embedding |
| Vector Store | ChromaDB (distanza cosine) |
| RAG | Parent-child + ibrido BM25/semantico con Reciprocal Rank Fusion |
| Data | Pandas + OpenPyXL |
| Environment | Conda (`LLM_DEMENTIA`) |

---

## Setup

Una volta sola:

```bash
conda env create -f environment.yml
conda activate LLM_DEMENTIA

ollama pull bge-m3        # embedding della knowledge base (obbligatorio)
ollama pull llama3.1:8b   # modello diagnostico (uno a scelta)
```

## Avvio

Ollama deve essere in esecuzione (`ollama serve`). Poi, **un solo comando**:

```bash
conda activate LLM_DEMENTIA
python run.py
```

`run.py` verifica le dipendenze, Ollama, il modello di embedding e l'indice RAG,
compila il frontend se manca e serve API e interfaccia su
<http://127.0.0.1:8000>, aprendo il browser.

| Opzione | Effetto |
|---|---|
| `--dev` | avvia anche il dev server Vite con hot reload (porta 5173) |
| `--reload` | ricarica il backend a ogni modifica dei file Python |
| `--rebuild-rag` | reindicizza la knowledge base prima di partire |
| `--skip-checks` | salta i controlli preliminari |
| `--no-browser` | non apre il browser |

> Esegui sempre dall'ambiente `LLM_DEMENTIA`. Dall'ambiente `base` la versione di
> ChromaDB è diversa e l'indice su disco non è leggibile.

---

## Knowledge base RAG

La KB è la cartella `Documenti_/`, scansionata ricorsivamente. Attualmente
contiene un solo testo di riferimento: *Budson & Solomon — A Practical Guide for
Clinicians*. I PDF non sono versionati.

```bash
python scripts/build_rag.py --dry-run   # solo chunking, nessun embedding
python scripts/build_rag.py --force     # reindicizza da zero
python scripts/build_rag.py --probe     # ispeziona il retrieval
```

Il chunking garantisce che ogni passaggio citabile superi i 350 caratteri: i
titoli di sezione vengono fusi nel testo che introducono invece di diventare
"fonti" prive di contenuto. Le citazioni restituite all'interfaccia contengono il
passaggio integrale, la sezione, l'intervallo di pagine e gli estratti esatti che
hanno prodotto il match.

---

## Verifica

```bash
python scripts/verify_prompts.py                      # ogni step riceve tutti gli input previsti
python scripts/audit_leakage.py                       # la diagnosi non raggiunge il modello
python scripts/smoke_pipeline.py --model qwen3.5:4b   # pipeline end-to-end
cd frontend && npm run build                          # typecheck + build
```

`verify_prompts.py` deve stampare "TUTTI I CONTROLLI SUPERATI" e
`audit_leakage.py` "nessuna fuga strutturale o letterale". Entrambi restituiscono
un exit code diverso da zero in caso di problema, quindi sono utilizzabili in CI.

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
│   ├── document_processor.py  # ingestion PDF: normalizzazione + chunking
│   ├── vector_store.py        # gestione ChromaDB
│   └── retriever.py           # retrieval ibrido BM25 + semantico (RRF)
├── scripts/
│   ├── build_rag.py           # build/ispezione indice RAG
│   ├── verify_prompts.py      # verifica completezza input dei prompt
│   ├── audit_leakage.py       # audit: la diagnosi non raggiunge il modello
│   ├── smoke_pipeline.py      # test end-to-end della pipeline
│   └── extract_unidentified_drugs.py  # lista farmaci da mappare
├── Documenti_/             # knowledge base PDF (gitignored)
├── results/                # risultati batch e individuali (gitignored)
├── chroma_db/              # vector store (gitignored)
├── environment.yml         # ambiente conda
├── requirements.txt        # dipendenze pip
└── run.py                  # entry point unico
```

---

## Pipeline diagnostica

L'informazione è **cumulativa**: come un clinico che richiede esami via via più
invasivi, ogni step vede tutto ciò che era disponibile ai precedenti più il
ragionamento già prodotto e i nuovi dati del proprio livello. Il quadro clinico
non viene mai dimenticato.

### Step 1 — Valutazione clinica + knowledge base

| Colonne del database | Ruolo |
|---|---|
| `ANAMNESI`, `EON` | quadro clinico in testo libero |
| `MMSE` | profilo cognitivo |
| `TERAPIA` | standardizzata in molecola + classe ATC |
| `Fam`, `Fumo (attivo o pregresso)`, `Ipertensione`, `Malattia_cardiovascolare`, `Diabete`, `Dislipidemia` | fattori di rischio |
| `Age`, `Gender`, `Anni_edu` | demografia (l'istruzione corregge l'MMSE) |

È l'**unico step con RAG**: retrieval ibrido multi-query bilingue sul testo di
riferimento. Output: diagnosi primaria e tutte le differenziali con probabilità
(ALTA/MEDIA/BASSA/ESCLUSA) e citazioni verbatim delle fonti usate.

### Step 2 — Biomarcatori plasmatici

| Colonne aggiunte | Ruolo |
|---|---|
| `Plasma_Ab4240`, `plasma_ptau217`, `plasma_pt181`, `plasma_NfL` | biomarcatori di neurodegenerazione |
| `Creatinina`, `AST`, `ALT`, `eGFR_2021` | attendibilità dei biomarcatori: un'alterazione epatica o renale li rende non interpretabili |

Riceve inoltre il quadro clinico completo dello Step 1 e il suo **output
integrale** (diagnosi, distribuzione di probabilità, ragionamento per ogni
ipotesi, elementi chiave, limitazioni dichiarate). Valuta l'attendibilità prima
di interpretare, poi aggiorna le probabilità.

### Step 3 — Biomarcatori liquorali

| Colonne aggiunte | Ruolo |
|---|---|
| `CSF_Ab42`, `CSF_Ab40`, `CSF_Ab4240`, `CSF_ttau`, `CSF_ptau`, `CSF_NfL` | gold standard diagnostico |

Riceve il quadro clinico, l'**output integrale degli Step 1 e 2**, il plasma e
gli indici epato-renali. Produce diagnosi finale, classificazione ATN e
concordanza plasma-liquor. La concordanza con `Diagnosi_CODIFICATA` è calcolata
**a valle**, da `pipeline/evaluator.py`: il modello non la vede mai.

Gli step 2 e 3 girano sempre: se i biomarcatori mancano, il modello conferma la
valutazione precedente dichiarando che l'assenza di dati non permette
aggiornamenti. Ogni step riceve istruzioni esplicite contro l'ancoraggio alla
diagnosi precedente, perché il ragionamento pregresso gli è fornito come storia
del percorso diagnostico, non come conclusione da difendere.

---

## Integrità scientifica

Il modello deve essere **ignaro della diagnosi di riferimento**: qualsiasi fuga
invaliderebbe i risultati. Tre garanzie, verificabili con
`python scripts/audit_leakage.py`:

1. **Per costruzione.** `build_step_payload` costruisce il payload da una
   whitelist di colonne. `Diagnosi_CODIFICATA`, `Diagnosi_TESTUALE` e `PAZIENTE`
   (nome e cognome reali) non ne fanno parte e non possono entrare per errore.
2. **Per dimostrazione.** L'audit costruisce i tre prompt e le query RAG due
   volte, con il record reale e con le colonne riservate sostituite da un valore
   sentinella, e verifica che i risultati siano identici byte a byte. Sui 106
   pazienti: **318 prompt confrontati, 0 differenze**.
3. **Sul dato, non sul codice.** L'audit cerca termini diagnostici nei campi in
   testo libero. Un solo paziente su 106 ha il termine "Alzheimer" in `ANAMNESI`
   con ground truth `AD`; altri due menzionano un'eziologia vascolare pur avendo
   ground truth `AD` (informazione fuorviante, non favorevole). Nessun filtro
   software può intercettarlo: va deciso con i clinici se anonimizzare.

### Caveat sul dataset

La distribuzione delle classi è fortemente sbilanciata:

| AD | VAD | SCD | PD | FTD | MIXED | LATE | AD-PPA |
|---|---|---|---|---|---|---|---|
| 72 | 6 | 6 | 6 | 6 | 4 | 3 | 3 |

Su 106 pazienti, **AD è il 68%**: un modello che rispondesse sempre "AD"
otterrebbe il 68% di accuracy. L'accuracy da sola non è informativa. Per questo
`evaluator.py` riporta anche **Cohen's κ**, precision/recall/F1 per classe e la
matrice di confusione, che sono le metriche da usare nelle conclusioni.

---

## Primo utilizzo

1. `python run.py` apre l'interfaccia
2. Dashboard → verifica stato Ollama e RAG
3. Gestione Modelli → scarica il modello diagnostico preferito
4. Analisi Paziente → seleziona paziente → esegui i singoli step o l'intera
   pipeline, poi **Salva** per archiviare il caso in `results/individual/`
5. Analisi Batch → elabora tutti i pazienti e genera metriche
   (checkpoint automatico ogni 5 pazienti)
6. Risultati → accuracy, Cohen's κ, concordanza per paziente
