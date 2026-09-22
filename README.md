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

ollama pull bge-m3           # embedding RAG obbligatorio
# Tag Q4 per sviluppo locale/smoke test su GPU 8 GB:
ollama pull qwen3.5:latest
ollama pull ministral-3:8b
ollama pull llama3.1:8b

# Per l'evaluation RTX 3090 usare invece i tag BF16/FP16 in SERVER_README.md.
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
>
> Se `ollama serve` risponde `bind ... only one usage`, Ollama è già in ascolto:
> non avviare una seconda istanza. Verificare con `ollama list` o con
> `curl http://127.0.0.1:11434/api/tags`; `ollama status` non esiste.

---

## Docker / server NVIDIA

Per una RTX 3090 usare `docker-compose.yml`, che avvia Ollama in un container con
GPU NVIDIA e monta database, manuali, indice e risultati come volumi. L'Ollama
container è esposto sull'host a `127.0.0.1:11435` per non confliggere con un
Ollama locale sulla porta 11434. Avvio rapido in <SERVER_README.md>; protocollo
completo in <evaluation/README.md>.

---

## Knowledge base RAG

La KB è la cartella `Documenti_/`, scansionata ricorsivamente. Contiene
*Budson & Solomon — A Practical Guide for Clinicians* e *Casebook of Dementia*.
L'applicazione e l'evaluation consentono tre modalità: `budson`, `casebook` e
`both`. In modalità combinata, al massimo 5 delle 8 fonti possono provenire dallo
stesso manuale, così entrambi contribuiscono al contesto. I PDF della knowledge
base non sono versionati. `Class_DX.pdf` è un documento di protocollo fornito dal
team clinico e non viene indicizzato nel RAG.

```bash
python scripts/build_rag.py --dry-run   # solo chunking, nessun embedding
python scripts/build_rag.py --force     # reindicizza da zero
python scripts/build_rag.py --probe     # ispeziona il retrieval
```

Il chunking garantisce che ogni passaggio citabile superi i 350 caratteri: i
titoli di sezione vengono fusi nel testo che introducono invece di diventare
"fonti" prive di contenuto. La query clinica paziente-specifica ha peso maggiore
nella RRF rispetto alle query generiche: il test verifica che T1 e T2 non ricevano
lo stesso contesto. Le citazioni restituite all'interfaccia contengono il
passaggio integrale, la sezione, l'intervallo di pagine e gli estratti esatti che
hanno prodotto il match.

---

## Verifica

```bash
python scripts/test_validator.py                      # coerenza dell'output diagnostico
python scripts/test_biomarkers.py                     # cut-off e profilo ATN deterministico
python scripts/test_json_parser.py                    # parsing esatto/riparato tracciato
python scripts/test_evaluator.py                      # metriche e output invalidi
python scripts/test_pipeline_flow.py                  # step non eleggibili e fallback
python scripts/test_rag_modes.py                      # filtri Budson/Casebook/both
python scripts/test_shared_cache.py                   # cache unica per tutti i modelli
python scripts/verify_prompts.py                      # input effettivi dei tre step
python scripts/audit_leakage.py                       # la diagnosi non raggiunge il modello
python scripts/audit_dataset.py                       # schema, missingness e outlier
python scripts/check_reference_values.py              # cut-off vs foglio dei clinici
python scripts/smoke_pipeline.py --model qwen3.5:latest --code T2 --seed 1001
cd frontend && npm run build
```

I cut-off dei biomarcatori sono dichiarati una sola volta, in `REFERENCE_VALUES`
(`config/settings.py`), e da lì vengono generati sia il blocco di riferimento del
system prompt sia le annotazioni `(normale: ...)` accanto a ogni valore.
`check_reference_values.py` li confronta con *Valori di riferimento_lab.xlsx*, il
documento autorevole dei clinici: una revisione dei cut-off sul foglio non passa
inosservata. Unica eccezione, `eGFR > 60`, che non compare sul foglio.

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
│   ├── loader.py           # caricamento e whitelist colonne per step
│   ├── biomarkers.py       # status, affidabilità, ATN e concordanza deterministici
│   ├── preprocessor.py     # standardizzazione terapia
│   └── therapy_drugs.py    # dizionario farmaci italiani
├── evaluation/             # CLI headless, JSONL/Excel, resume e README dedicato
├── frontend/               # React + TypeScript + TailwindCSS
│   └── src/
│       ├── pages/          # Dashboard, Analisi, Batch, Risultati...
│       ├── api/client.ts   # Client API tipizzato
│       └── types/index.ts  # TypeScript interfaces
├── llm/
│   ├── ollama_client.py    # client HTTP, seed, tempi/token e parser tracciato
│   ├── output_schemas.py   # JSON Schema vincolanti per i tre step
│   └── prompt_builder.py   # template prompt per 3 step
├── pipeline/
│   ├── step_runner.py      # esecutore pipeline 3-step
│   ├── validator.py        # normalizzazione e coerenza dell'output diagnostico
│   └── evaluator.py        # metriche di concordanza
├── rag/
│   ├── document_processor.py  # ingestion PDF: normalizzazione + chunking
│   ├── vector_store.py        # gestione ChromaDB
│   └── retriever.py           # retrieval ibrido BM25 + semantico (RRF)
├── scripts/
│   ├── build_rag.py, smoke_pipeline.py, test_reproducibility.py
│   ├── test_validator.py, test_biomarkers.py, test_json_parser.py
│   ├── test_evaluator.py, verify_prompts.py
│   ├── audit_leakage.py, audit_dataset.py, check_reference_values.py
│   └── extract_unidentified_drugs.py
├── Documenti_/             # knowledge base PDF (gitignored)
├── Class_DX.pdf            # schema clinico delle classi, non indicizzato
├── results/                # risultati batch e individuali (gitignored)
├── chroma_db/              # vector store (gitignored)
├── Dockerfile, docker-compose.yml  # deploy portabile NVIDIA/Ollama
├── environment.yml         # ambiente conda
├── requirements.txt        # dipendenze pip
└── run.py                  # entry point unico
```

---

## Pipeline diagnostica

L'informazione è **cumulativa**: come un clinico che richiede esami via via più
invasivi, ogni step vede tutto ciò che era disponibile ai precedenti più il
ragionamento già prodotto e i nuovi dati del proprio livello. Il quadro clinico
non viene mai dimenticato. Il bottone frontend “Esegui Pipeline”, il batch API e
l'evaluation headless chiamano tutti lo stesso `run_full_pipeline`: non esistono
orchestrazioni parallele con logiche differenti.

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
| `Creatinina`, `AST`, `ALT`, `eGFR_2021` | possibili confondenti: un'alterazione genera un avviso di cautela, senza invalidare automaticamente tutto il plasma |

Riceve inoltre il quadro clinico completo dello Step 1 e il suo **output
integrale**. I marker diagnostici seguono la gerarchia concordata col neurologo:
p-tau217, poi p-tau181, poi plasma Aβ42/40; viene usato il primo disponibile.
Plasma NfL è informazione aggiuntiva, non dirimente. Se nessuno dei tre marker
gerarchici è presente, lo Step 2 è non eleggibile e non viene chiamato il modello.

### Step 3 — Biomarcatori liquorali

| Colonne aggiunte | Ruolo |
|---|---|
| `CSF_Ab42`, `CSF_Ab40`, `CSF_Ab4240`, `CSF_ttau`, `CSF_ptau`, `CSF_NfL` | evidenza biologica prioritaria sulla presenza o assenza di patologia Alzheimer |

Riceve il quadro clinico, l'**output integrale degli Step 1 e 2**, il plasma e
gli indici epato-renali. Il codice calcola deterministicamente status dei
marcatori, profilo ATN e concordanza plasma-liquor; il modello interpreta questi
risultati senza ricalcolare cut-off. La normalità dei biomarcatori AD non dimostra
automaticamente quale demenza alternativa sia corretta e la positività non
esclude copatologie. CSF NfL è aggiuntivo e non determina l'eleggibilità. T94,
privo di marker CSF core, resta nel dataset ma è non eleggibile allo Step 3. La
concordanza con `Diagnosi_CODIFICATA` è calcolata **a valle**, da
`pipeline/evaluator.py`: il modello non la vede mai.

Gli step 2 e 3 girano sempre: se i biomarcatori mancano, il modello conferma la
valutazione precedente dichiarando che l'assenza di dati non permette
aggiornamenti. Ogni step riceve istruzioni esplicite contro l'ancoraggio alla
diagnosi precedente, perché il ragionamento pregresso gli è fornito come storia
del percorso diagnostico, non come conclusione da difendere.

### Ordine di ragionamento e coerenza dell'output

Ollama riceve un vero JSON Schema. Il modello deve prima produrre
`diagnostic_reasoning`, poi un oggetto `diagnosis_assessments` con **otto chiavi
fisse**, una per classe, e solo alla fine `primary_diagnosis`. Questo impedisce
omissioni, duplicati e codici combinati tipici degli array liberi nei modelli
7–9B.

I confidence score grezzi sono conservati in `model_result` e come
`reported_confidence_score`. Poiché i modelli non rispettano sempre la somma 1,
il validatore produce anche una distribuzione normalizzata per interfaccia e
analisi, senza modificare l'argmax. Le categorie ALTA/MEDIA/BASSA/ESCLUSA sono
derivate deterministicamente dagli score normalizzati, non riscritte dal
modello.

**Il validatore segnala, non corregge la diagnosi.** Se la primaria dichiarata
non coincide con l'argmax, entrambe vengono conservate e compare un alert. Sono
tracciati anche somma grezza, citazioni non letterali, JSON riparati e output
invalidi. L'accuracy principale è intention-to-evaluate: un output invalido
resta nel denominatore; `valid_output_accuracy` è riportata separatamente.

Per lo studio `LLM_RETRIES_ON_INVALID=0`: ritentare fino a ottenere una risposta
valida selezionerebbe artificialmente gli output migliori. Ogni risposta salva
prompt esatti, payload, query e fonti RAG, risposta grezza, JSON originario,
risultato normalizzato, seed, opzioni, token e tempi Ollama.

---

## Evaluation headless

La pipeline multi-run è in `evaluation/` e ha documentazione dedicata:
<evaluation/README.md>. Le tre cache condivise si creano con
`python -m evaluation.cache --rag-mode all`, senza modello, run o seed. Supporta
resume, JSONL append-only, Excel, manifest riproducibile e container NVIDIA.

## Protocollo dell'evaluation headless

I tre modelli dello studio sul server 24 GB sono `qwen3.5:9b-bf16`,
`ministral-3:8b-instruct-2512-fp16` e `llama3.1:8b-instruct-fp16`, eseguiti un
modello per processo. I tag quantizzati `:latest`/`:8b` sono solo per sviluppo e
smoke test, non per i risultati definitivi. Per ogni modello: 106 pazienti,
20–30 run, tre step concatenati, con una lista prefissata di seed identica tra i
modelli. Il test su Qwen ha verificato che stesso prompt + stesso seed produce
output byte-per-byte identico e che seed diversi producono output diversi. Una configurazione da 30 run richiede 3.180 pipeline e 9.540 chiamate
LLM per modello.

La CLI importa direttamente `run_full_pipeline`: non contiene una copia
dei prompt o della logica applicativa. I bundle RAG saranno precomputati una
volta per paziente e verificati via SHA-256, evitando centinaia di migliaia di
embedding identici senza cambiare il prompt ricevuto dal modello.

JSONL sarà il formato autorevole, append-only e ripristinabile; Excel sarà una
vista tabellare derivata. Oltre alle colonne richieste, verranno salvati seed,
digest del modello, commit Git, hash dataset/KB, input e prompt esatti, output
grezzo e parsato, warning, fonti/citazioni, token, tempi, completezza dei
biomarcatori e status di parsing.

Le metriche saranno separate: correttezza a ogni step, primo step corretto,
correttezza persistente, errore corretto dal CSF, errore introdotto dal CSF e
correttezza pre-CSF. Le run sono osservazioni ripetute annidate nei pazienti:
intervalli di confidenza e confronti tra modelli dovranno essere calcolati a
livello paziente, non trattando le 3.180 esecuzioni come pazienti indipendenti.

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
matrice di confusione.

La completezza grezza è eterogenea: solo 29 pazienti hanno tutti e quattro i
marker plasmatici e 6 non ne hanno nessuno. Con la gerarchia clinica, però, 100
pazienti sono eleggibili allo Step 2: 36 usano p-tau217 e 64 p-tau181; nessun
paziente deve ricorrere ad Aβ42/40 come unico marker. Per il CSF, 105/106 sono
eleggibili allo Step 3. L'analisi riporta comunque disponibilità e marker scelto.
`audit_dataset.py` registra anche gli outlier: `T101 plasma_ptau217` è stato
corretto da 87 a 0.87; `T81 CSF_NfL=3501` è stato confermato dal team e resta
esplicitamente tracciato come valore estremo accettato.

`TERAPIA` è intenzionalmente inclusa ma può incorporare la precedente impressione
clinica: Donepezil/Memantina possono indurre il modello a favorire AD. Questo
possibile incorporation bias deve essere dichiarato e può essere studiato con
un'analisi di sensibilità senza terapia.

La classe mantiene il codice `PD` per corrispondere al database, ma la sua
etichetta clinica segue `Class_DX.pdf`: “Demenza associata a Parkinson / spettro
Lewy body (PDD/DLB)”. Le diagnosi possibili restano otto.

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
