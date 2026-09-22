# Evaluation headless

Questa cartella contiene la pipeline sperimentale senza interfaccia. Non duplica
prompt, RAG o logica diagnostica: importa direttamente `run_full_pipeline`, lo
stesso motore usato dall'applicazione FastAPI.

## Protocollo

Modelli non quantizzati previsti, uno per processo:

- `qwen3.5:9b-bf16` (9.7B, circa 19 GB)
- `ministral-3:8b-instruct-2512-fp16` (8.9B, circa 18 GB)
- `llama3.1:8b-instruct-fp16` (8.0B, circa 16 GB)

I tag brevi `qwen3.5:latest`, `ministral-3:8b` e `llama3.1:8b` sono Q4 e
non devono essere usati per i risultati definitivi.

Per ogni modello e modalità RAG: 106 pazienti × 20/30 run × 3 step. I seed della
run sono `seed_start + run - 1` e sono identici tra modelli e corpus. I seed dei
tre step sono rispettivamente `seed`, `seed+1`, `seed+2`. La CLI rifiuta per
default modelli quantizzati e modelli non quasi interamente residenti in VRAM.

## Modalità RAG

| Valore | Corpus |
|---|---|
| `budson` | *Budson & Solomon — A Practical Guide for Clinicians* |
| `casebook` | *Casebook of Dementia* |
| `both` | entrambi i manuali |

L'indice Chroma contiene entrambi i documenti; il filtro corpus viene applicato
sia alla ricerca semantica sia a BM25. In modalità `both`, massimo 5 fonti su 8
possono provenire dallo stesso manuale, garantendo contributi da entrambi. I
bundle RAG sono precalcolati una volta
per paziente/modalità, salvati in `rag_cache/` e verificati tramite SHA-256. Il
prompt è identico a quello ottenuto col retrieval live.

Dopo aver aggiunto o modificato un PDF:

```bash
python scripts/build_rag.py --force
```

## Preparazione locale

```bash
conda activate LLM_DEMENTIA
ollama serve

ollama pull bge-m3
ollama pull qwen3.5:9b-bf16
ollama pull ministral-3:8b-instruct-2512-fp16
ollama pull llama3.1:8b-instruct-fp16

python scripts/build_rag.py --force
```

Se `ollama serve` restituisce `bind ... only one usage`, Ollama è già attivo: non
avviarne una seconda istanza. Verificare con `ollama list` oppure:

```bash
curl http://127.0.0.1:11434/api/tags
```

`ollama status` non è un comando valido.

## Preflight

Il database viene bloccato se contiene valori estremi non approvati. Al momento
`T81/CSF_NfL=3501` è registrato come outlier confermato;
`T101/plasma_ptau217` è stato corretto a `0.87`. `audit_dataset.py` deve essere
rieseguito sul file definitivo prima dell'esperimento.

```bash
python scripts/audit_dataset.py
python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 30 \
  --rag-mode budson \
  --dry-run
```

Prima delle run la CLI esegue un warm-up e interroga `/api/ps`: stampa GB e
percentuale del modello in VRAM, bloccando CPU-only e offload parziale. I flag
`--allow-quantized`, `--allow-partial-gpu` e `--allow-data-warnings` sono solo per
smoke test e non vanno usati nell'esperimento definitivo.

## Esecuzione

Il precalcolo può essere eseguito separatamente:

```bash
python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 30 \
  --rag-mode budson \
  --seed-start 1001 \
  --cache-only
```

Crea subito `manifest.json`, `progress.json`, un `runs.jsonl` vuoto e i file
paziente in `rag_cache/`. Non crea un Excel vuoto: `results.xlsx` nasce dopo la
prima pipeline LLM completata. Cache e progressi sopravvivono a Ctrl+C. Le cache
vecchie prive di `cache_version` vengono rigenerate automaticamente; la versione
corrente usa query clinica pesata e produce contesti paziente-specifici.

Esempio Qwen, 30 run, Budson:

```bash
python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 30 \
  --rag-mode budson \
  --seed-start 1001
```

Casebook:

```bash
python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 30 \
  --rag-mode casebook \
  --seed-start 1001
```

Entrambi:

```bash
python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 30 \
  --rag-mode both \
  --seed-start 1001
```

Ripetere gli stessi tre comandi sostituendo `--model` con `ministral-3:8b-instruct-2512-fp16` e
`llama3.1:8b-instruct-fp16`.

### Prova breve

```bash
python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 2 \
  --rag-mode budson \
  --patients T1 T2 \
  --experiment-name smoke_qwen_budson \
  --allow-data-warnings
```

Oppure usare `--limit 2`. Su una macchina di sviluppo con tag Q4 usare
esplicitamente `--allow-quantized --allow-partial-gpu --allow-dirty`; queste
opzioni non sono ammesse nell'esperimento definitivo.

## Resume

Il JSONL viene scritto e sincronizzato su disco dopo ogni pipeline. Per
riprendere è sufficiente rilanciare lo stesso comando: le coppie paziente/run già
presenti vengono saltate.

```bash
python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 30 \
  --rag-mode budson \
  --seed-start 1001
```

- `--retry-errors`: ripete solo le run registrate con `status=error`.
- `--refresh-rag-cache`: ricostruisce la cache RAG.
- `--no-excel`: salva solo JSONL; rigenerare dopo con
  `python -m evaluation.export_results <cartella-esperimento>`.
- `--excel-every 20`: frequenza dei checkpoint Excel.
- `--force-unlock`: rimuove un lock residuo, solo dopo aver escluso processi attivi.

Ctrl+C durante il precalcolo conserva ogni cache già scritta; al rilancio queste
vengono lette e si riparte dal primo paziente mancante. Ctrl+C durante le run non
perde le pipeline concluse. `progress.json` indica fase, paziente e contatori.
Una pipeline interrotta a metà verrà rieseguita con lo stesso seed.

## Output

Percorso predefinito:

```text
results/evaluation/<modello>__<rag_mode>/
├── manifest.json
├── progress.json
├── runs.jsonl
├── results.xlsx  # solo dopo la prima run LLM completata
└── rag_cache/
```

`runs.jsonl` è la fonte autorevole append-only. Contiene input e prompt esatti,
query/fonti RAG, risposta grezza, JSON originario, risultato normalizzato, seed,
warning, parse status, token e tempi. Excel è una vista derivata e contiene:

- PAZIENTE, RUN, SEED, MODELLO, RAG_MODE;
- diagnosi, confidence normalizzata/raw e motivazione per Step 1/2/3;
- warning, argmax, score sum e parse status;
- DIAGNOSI REALE e correttezza per step/finale;
- primo step corretto, correttezza prima del CSF e persistente;
- errore corretto/introdotto dal CSF;
- eleggibilità e disponibilità dei biomarcatori.

`manifest.json` registra digest modello, commit Git, hash dataset/manuali,
hardware, versione Ollama e parametri di generazione.

## Eleggibilità biomarcatori

Step 2 usa la gerarchia concordata:

1. p-tau217;
2. p-tau181;
3. plasma Aβ42/40.

Viene usato il primo disponibile. Plasma NfL è aggiuntivo, non dirimente. Nel
dataset attuale 36 pazienti usano p-tau217, 64 p-tau181 e 6 non hanno alcun
marker gerarchico; nessuno usa Aβ42/40 come unico fallback. Senza marker
principale, Step 2 è `ineligible`, non un errore del modello.

CSF NfL è anch'esso aggiuntivo. Step 3 è eleggibile se è presente almeno un
marcatore CSF core. T94 viene mantenuto nel dataset ma risulta non eleggibile a
Step 3; la sua diagnosi finale è quindi l'ultimo step completato.

Le accuracy intention-to-evaluate includono nel denominatore gli output invalidi
dei pazienti eleggibili, ma escludono i pazienti non eleggibili per assenza dei
dati richiesti. Entrambi i conteggi sono riportati.

## Aggiungere un modello

1. Installarlo: `ollama pull <nome>`.
2. Verificare che supporti contesto 12.288 e structured output Ollama.
3. Eseguire uno smoke test:

```bash
python scripts/smoke_pipeline.py --model <nome> --code T2 --seed 1001
```

4. Eseguire `test_reproducibility.py`.
5. Avviare un esperimento con un nome distinto.

Non cambiare temperatura, prompt, seed schedule o parametri tra modelli nello
stesso confronto.

## Container NVIDIA

Dalla root del repository, con Docker e NVIDIA Container Toolkit:

```bash
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
export GIT_COMMIT=$(git rev-parse HEAD)
docker compose build

docker compose up -d ollama
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull qwen3.5:9b-bf16
docker compose exec ollama ollama pull ministral-3:8b-instruct-2512-fp16
docker compose exec ollama ollama pull llama3.1:8b-instruct-fp16

docker compose run --rm app python scripts/build_rag.py --force

docker compose run --rm evaluation \
  python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 30 \
  --rag-mode budson \
  --seed-start 1001
```

I dati clinici, PDF, indice e risultati sono montati come volumi e non inclusi
nell'immagine. Su server non eseguire contemporaneamente l'Ollama host e il
container sulla stessa porta.
