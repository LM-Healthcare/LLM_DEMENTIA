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
| Verifica input dei prompt | `python scripts/verify_prompts.py` |
| Test pipeline end-to-end | `python scripts/smoke_pipeline.py --model qwen3.5:4b` |
| Typecheck + build frontend | `cd frontend && npm run build` |
| Lista farmaci da mappare | `python scripts/extract_unidentified_drugs.py` |

Ollama deve essere in esecuzione (`ollama serve`). Modelli richiesti: `bge-m3`
(embedding, obbligatorio) più almeno un modello diagnostico.

## Verifica prima di considerare un lavoro concluso

1. `python scripts/verify_prompts.py` — deve stampare "TUTTI I CONTROLLI SUPERATI".
   Controlla che ogni step riceva davvero anamnesi, EON, MMSE, terapia, fattori
   di rischio, biomarcatori del proprio livello e output integrale degli step
   precedenti.
2. `cd frontend && npm run build` — il typecheck TypeScript è parte del build.
3. Se sono stati toccati `rag/` o i parametri di chunking:
   `python scripts/build_rag.py --dry-run` e controllare che nessun parent chunk
   sia sotto `RAG_MIN_CHUNK_CHARS`.

## Vincoli da non violare

- **Reindicizzare dopo aver cambiato embedding o chunking.** Le collezioni
  ChromaDB sono legate alla dimensione dei vettori (bge-m3 = 1024, nomic = 768) e
  rifiutano inserimenti di dimensione diversa.
- **`shutil.rmtree` sull'indice può fallire in silenzio** su Windows/OneDrive.
  Il reset va fatto eliminando le collezioni via API Chroma (`_reset_store`),
  altrimenti restano vive collezioni con la vecchia dimensione.
- **La knowledge base è `Documenti_/`**, un solo testo: Budson & Solomon,
  *A Practical Guide for Clinicians*. `Documenti_old/` e `Continuum Demenze/`
  sono archivi non indicizzati.
- **Le celle vuote dell'Excel sono `float('nan')`, non `""`.** `record.get(k, "")`
  non protegge se la chiave esiste: usare la coercizione `_text()` di
  `pipeline/step_runner.py`.
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
5. Le citazioni esposte all'interfaccia contengono il passaggio **integrale**,
   sezione, capitolo, intervallo di pagine, punteggio e gli estratti esatti che
   hanno prodotto il match.
