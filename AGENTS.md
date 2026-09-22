# Note di progetto

## Ambiente

**Usare sempre l'ambiente conda `LLM_DEMENTIA`.** L'ambiente `base` ha una
versione diversa di ChromaDB e non riesce a leggere l'indice su disco: i binding
Rust vanno in panic con `PanicException: range start index N out of range`
(e `PanicException` deriva da `BaseException`, quindi non viene catturata da
`except Exception`).

```bash
conda activate LLM_DEMENTIA
python run.py
```

Interprete diretto, se serve: `C:/Users/filow/miniconda3/envs/LLM_DEMENTIA/python.exe`

## Comandi

| Scopo | Comando |
|---|---|
| Avvio applicazione | `python run.py` |
| Avvio con hot reload | `python run.py --dev` |
| Build indice RAG | `python scripts/build_rag.py --force` |
| Ispezione chunking (senza embedding) | `python scripts/build_rag.py --dry-run` |
| Test retrieval | `python scripts/build_rag.py --probe` |
| Test del validatore | `python scripts/test_validator.py` |
| Test biomarcatori/ATN | `python scripts/test_biomarkers.py` |
| Test flusso missing data | `python scripts/test_pipeline_flow.py` |
| Test parser/evaluator | `python scripts/test_json_parser.py && python scripts/test_evaluator.py` |
| Verifica input dei prompt | `python scripts/verify_prompts.py` |
| Audit leakage/dataset | `python scripts/audit_leakage.py && python scripts/audit_dataset.py` |
| Cut-off vs foglio clinici | `python scripts/check_reference_values.py` |
| Test pipeline end-to-end | `python scripts/smoke_pipeline.py --model qwen3.5:latest --code T2 --seed 1001` |
| Linter Python | `python -m pyflakes api config data llm pipeline rag scripts run.py` |
| Typecheck + build frontend | `cd frontend && npm run build` |
| Test modalità RAG | `python scripts/test_rag_modes.py` |
| Smoke evaluation | `python -m evaluation.run --model llama3.1:8b --runs 1 --rag-mode casebook --patients T2 --experiment-name smoke --allow-dirty` |
| Lista farmaci da mappare | `python scripts/extract_unidentified_drugs.py` |

Ollama deve essere in esecuzione (`ollama serve`). Modelli dello studio:
`qwen3.5:9b-bf16`, `ministral-3:8b-instruct-2512-fp16`,
`llama3.1:8b-instruct-fp16`; embedding `bge-m3`. I tag Q4 sono ammessi solo per
smoke test con `--allow-quantized`, mai per l'esperimento definitivo. Parametri condivisi:
temperature 0.1, num_ctx 12288, num_predict 4096, retry 0.

## Verifica prima di considerare un lavoro concluso

1. Eseguire `test_validator.py`, `test_biomarkers.py`, `test_json_parser.py` e
   `test_evaluator.py`.
2. `python scripts/verify_prompts.py` — deve stampare "TUTTI I CONTROLLI SUPERATI".
3. `python scripts/audit_leakage.py` — deve stampare "nessuna fuga strutturale o
   letterale"; `audit_dataset.py` non deve trovare errori strutturali.
4. `python scripts/check_reference_values.py` — cut-off allineati al foglio.
5. `python -m pyflakes api config data llm pipeline rag scripts run.py` — nessun
   output atteso.
6. `cd frontend && npm run build` — il typecheck TypeScript è parte del build.
7. Se sono stati toccati `rag/` o i parametri di chunking:
   `python scripts/build_rag.py --dry-run` e controllare che nessun parent chunk
   sia sotto `RAG_MIN_CHUNK_CHARS`.

## Vincoli da non violare

- **Reindicizzare dopo aver cambiato embedding o chunking.** Le collezioni
  ChromaDB sono legate alla dimensione dei vettori (bge-m3 = 1024, nomic = 768) e
  rifiutano inserimenti di dimensione diversa.
- **`shutil.rmtree` sull'indice può fallire in silenzio** su Windows/OneDrive.
  Il reset va fatto eliminando le collezioni via API Chroma (`_reset_store`),
  altrimenti restano vive collezioni con la vecchia dimensione.
- **La knowledge base è `Documenti_/`**, con Budson & Solomon e Casebook of
  Dementia. Le modalità `budson`, `casebook`, `both` filtrano lo stesso indice;
  `both` limita ogni manuale a 5 fonti su 8. Ogni modifica ai PDF richiede rebuild.
- **Il modello non deve mai vedere `Diagnosi_CODIFICATA`, `Diagnosi_TESTUALE` né
  `PAZIENTE`** (nomi reali). Il payload è costruito da whitelist in
  `build_step_payload`: non aggiungere passaggi che leggano il `record` grezzo nei
  prompt. Le colonne dei biomarcatori si dichiarano in `data/loader.py`, unica
  fonte di verità importata da prompt, API e script.
- **Le celle vuote dell'Excel sono `float('nan')`, non `""`.** `record.get(k, "")`
  non protegge se la chiave esiste: usare la coercizione `_text()` di
  `pipeline/step_runner.py`.
- **L'output LLM usa otto chiavi diagnostiche fisse.** L'ordine è
  `diagnostic_reasoning` → `diagnosis_assessments` (AD, AD-PPA, MIXED, VAD, SCD,
  LATE, FTD, PD) → `primary_diagnosis`. Non reintrodurre array liberi: i modelli
  7–9B duplicano classi e ne omettono altre anche sotto JSON Schema.
- **Conservare sempre grezzo e normalizzato.** `model_result` è il JSON originario
  e viene passato allo step successivo; `result` è la vista normalizzata per UI e
  metriche. I raw score restano in `reported_confidence_score`, quelli normalizzati
  in `confidence_score`. Il validatore segnala ma non cambia la diagnosi primaria.
- **Nessun retry nell'evaluation.** `LLM_RETRIES_ON_INVALID=0`: selezionare un
  tentativo valido introdurrebbe bias. Parse status, riparazioni e tentativi sono
  parte dell'audit trail.
- **Il modello non ricalcola biomarcatori o ATN.** `data/biomarkers.py` calcola
  status, confondenti epato-renali, ATN e concordanza. Il modello riceve questi
  risultati e produce solo l'interpretazione clinica. La regola operativa ATN usa
  Aβ42/40 prioritario (Aβ42 come fallback), p-tau per T, t-tau o NfL per N.
- **I cut-off dei biomarcatori si dichiarano solo in `REFERENCE_VALUES`**
  (`config/settings.py`). Da lì si generano il blocco del system prompt e le
  annotazioni `(normale: ...)` nei prompt: non reintrodurre valori hardcoded in
  `prompt_builder.py`. `scripts/check_reference_values.py` verifica
  l'allineamento con *Valori di riferimento_lab.xlsx*.
- **Il matching dei farmaci è a confine di parola** con preferenza per la chiave
  più lunga (`_match_drug` in `data/preprocessor.py`). Il match per sottostringa
  nuda produceva errori clinici reali (`citalopram` → Escitalopram,
  `delorazepam` → Lorazepam, `statina` → Rosuvastatina). Per aggiungere un
  farmaco si estende `data/therapy_drugs.py`, non si allenta il matching.

## Architettura del RAG

Retrieval ibrido con Reciprocal Rank Fusion:

1. Query multi-concetto **bilingui** (le cartelle cliniche sono in italiano, la
   knowledge base in inglese): senza le varianti inglesi il retrieval cross-lingua
   perde le fonti sui criteri diagnostici.
2. Per ogni query: ricerca semantica (bge-m3) + lessicale (BM25 su `rank_bm25`).
   BM25 è necessario per i termini esatti tipo `p-tau217` o `A+T+N+`.
3. I punteggi RRF dei child chunk si sommano sul parent: **le fonti restituite
   sono ordinate per rilevanza**. Prima l'ordine dipendeva dall'iterazione di un
   `set` ed era di fatto casuale.
4. Chunking con fusione dei frammenti sotto `RAG_MIN_CHUNK_CHARS`: un titolo di
   sezione viene inglobato nel testo che introduce invece di diventare una
   "fonte" priva di contenuto.
5. La query clinica paziente-specifica usa peso RRF 20: non ridurlo senza
   verificare `test_rag_modes.py`, che deve mostrare contesti diversi tra T1/T2.
   Ogni modifica al retrieval richiede incremento di `RAG_CACHE_VERSION`.
6. Le citazioni esposte all'interfaccia contengono il passaggio **integrale**,
   sezione, capitolo, intervallo di pagine, punteggio e gli estratti esatti che
   hanno prodotto il match.
