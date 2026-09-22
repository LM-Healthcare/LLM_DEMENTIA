# Avvio rapido sul server NVIDIA

Guida minima per Ubuntu/Linux con RTX 3090, Docker, Docker Compose e NVIDIA
Container Toolkit già installati.

## 1. Clone e file dati

```bash
git clone https://github.com/LM-Healthcare/LLM_DEMENTIA.git
cd LLM_DEMENTIA
```

Database e manuali non sono nel repository per privacy, copyright e dimensione.
Copiarli esattamente in queste posizioni:

```text
Database_FINALE_codificato.xlsx
Valori di riferimento_lab.xlsx
Documenti_/Budson e Solomon - A Practical Guide for Clinicians.pdf
Documenti_/Casebook_of_Dementia.pdf
```

## 2. Permessi e variabili

```bash
mkdir -p chroma_db results Documenti_
sudo chown -R "$USER":"$USER" chroma_db results Documenti_
chmod -R u+rwX chroma_db results Documenti_

export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
export GIT_COMMIT=$(git rev-parse HEAD)
```

## 3. Build e Ollama GPU

```bash
docker compose build
docker compose up -d ollama
docker compose exec ollama nvidia-smi
```

Scaricare i modelli non quantizzati dello studio:

```bash
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull qwen3.5:9b-bf16
docker compose exec ollama ollama pull ministral-3:8b-instruct-2512-fp16
docker compose exec ollama ollama pull llama3.1:8b-instruct-fp16
```

## 4. Costruzione RAG e controlli

```bash
docker compose run --rm app python scripts/build_rag.py --force
docker compose run --rm app python scripts/audit_dataset.py
docker compose run --rm app python scripts/test_rag_modes.py
```

## 5. Evaluation

Prima si può precalcolare il RAG, con resume automatico. Le cache create da
versioni precedenti senza `cache_version` vengono rigenerate automaticamente:

```bash
docker compose run --rm evaluation \
  python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 30 \
  --rag-mode budson \
  --seed-start 1001 \
  --cache-only
```

Poi avviare le run rilanciando lo stesso comando senza `--cache-only`:

```bash
docker compose run --rm evaluation \
  python -m evaluation.run \
  --model qwen3.5:9b-bf16 \
  --runs 30 \
  --rag-mode budson \
  --seed-start 1001
```

Ripetere per `casebook` e `both`, quindi per Ministral e Llama. Ogni comando è
resumable: dopo Ctrl+C basta rilanciarlo identico.

## 6. Monitoraggio

```bash
docker compose exec ollama ollama ps
cat results/evaluation/qwen3.5_9b-bf16__budson/progress.json
tail -f results/evaluation/qwen3.5_9b-bf16__budson/runs.jsonl
```

`ollama ps` deve mostrare il modello sulla GPU. La CLI esegue inoltre un warm-up
e blocca l'esperimento se il modello è quantizzato, non usa la GPU o non è quasi
interamente residente in VRAM.

Output:

```text
results/evaluation/<modello>__<rag_mode>/
├── manifest.json
├── progress.json
├── runs.jsonl
├── results.xlsx
└── rag_cache/
```

L'Excel nasce solo dopo la prima pipeline LLM completata; durante il solo
precalcolo RAG `runs.jsonl` esiste ma rimane vuoto.

## Applicazione web opzionale

```bash
docker compose up -d app
```

Aprire `http://IP_SERVER:8000`.
