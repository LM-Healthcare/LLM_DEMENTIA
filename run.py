"""
Entry point unico dell'applicazione.

Un solo comando avvia tutto: verifica Ollama, i modelli e l'indice RAG,
costruisce il frontend se serve e serve API + interfaccia sulla stessa porta.

Uso:
    python run.py                 # avvia tutto su http://127.0.0.1:8000
    python run.py --dev           # aggiunge il dev server Vite (hot reload)
    python run.py --rebuild-rag   # reindicizza la knowledge base prima di partire
    python run.py --skip-checks   # salta i controlli preliminari
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"


def _fail(message: str, hint: str = "") -> None:
    print(f"\n[X] {message}")
    if hint:
        print(f"    {hint}")
    sys.exit(1)


def check_python_env() -> None:
    """Verifica che le dipendenze siano importabili nell'interprete corrente."""
    missing = []
    for module in ("fastapi", "uvicorn", "pandas", "chromadb", "langchain_chroma", "rank_bm25"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        _fail(
            f"Dipendenze non trovate in {sys.prefix}: {', '.join(missing)}",
            "Attiva l'ambiente del progetto (conda activate LLM_DEMENTIA) e riprova.",
        )


def check_ollama() -> None:
    from config.settings import OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL
    from llm.ollama_client import is_ollama_running, get_model_names

    if not is_ollama_running():
        _fail(
            f"Ollama non raggiungibile su {OLLAMA_BASE_URL}",
            "Avvialo in un altro terminale con: ollama serve",
        )

    models = get_model_names()
    print(f"[ok] Ollama attivo — {len(models)} modelli disponibili")

    if not any(m.split(":")[0] == OLLAMA_EMBED_MODEL.split(":")[0] for m in models):
        _fail(
            f"Modello di embedding '{OLLAMA_EMBED_MODEL}' non installato",
            f"Scaricalo con: ollama pull {OLLAMA_EMBED_MODEL}",
        )
    print(f"[ok] Embedding model '{OLLAMA_EMBED_MODEL}' presente")


def check_rag(rebuild: bool) -> None:
    from config.settings import DOCS_FOLDER
    from rag.vector_store import get_store_stats

    if rebuild:
        print("[..] Reindicizzazione knowledge base richiesta")
        subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "build_rag.py"), "--force"],
                       check=True)

    stats = get_store_stats()
    if stats["status"] != "ready":
        pdfs = list(Path(DOCS_FOLDER).rglob("*.pdf")) if Path(DOCS_FOLDER).exists() else []
        if not pdfs:
            _fail(
                f"Nessun PDF nella knowledge base: {DOCS_FOLDER}",
                "Inserisci i documenti nella cartella e riavvia.",
            )
        print("[!!] Indice RAG assente. Costruzione in corso (una volta sola)...")
        subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "build_rag.py"), "--force"],
                       check=True)
        stats = get_store_stats()

    print(f"[ok] RAG pronto — {stats['parent_chunks']} passaggi citabili, "
          f"{stats['child_chunks']} unità di retrieval")


def check_frontend(dev: bool) -> None:
    if dev:
        return
    if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        print("[ok] Frontend compilato presente")
        return

    npm = shutil.which("npm")
    if not npm:
        _fail("Frontend non compilato e npm non trovato",
              "Installa Node.js, oppure avvia con --dev usando un server Vite esterno.")

    if not (FRONTEND_DIR / "node_modules").exists():
        print("[..] Installazione dipendenze frontend (npm install)...")
        subprocess.run([npm, "install"], cwd=FRONTEND_DIR, check=True, shell=False)

    print("[..] Compilazione frontend (npm run build)...")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, check=True, shell=False)
    print("[ok] Frontend compilato")


def start_vite() -> subprocess.Popen | None:
    npm = shutil.which("npm")
    if not npm:
        print("[!!] npm non trovato: dev server Vite non avviato")
        return None
    if not (FRONTEND_DIR / "node_modules").exists():
        subprocess.run([npm, "install"], cwd=FRONTEND_DIR, check=True)
    print("[..] Avvio dev server Vite su http://localhost:5173")
    return subprocess.Popen([npm, "run", "dev"], cwd=FRONTEND_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Dementia — avvio applicazione")
    parser.add_argument("--host", default="127.0.0.1", help="indirizzo host")
    parser.add_argument("--port", type=int, default=8000, help="porta")
    parser.add_argument("--dev", action="store_true",
                        help="avvia anche il dev server Vite con hot reload")
    parser.add_argument("--reload", action="store_true",
                        help="ricarica il backend a ogni modifica dei file Python")
    parser.add_argument("--rebuild-rag", action="store_true",
                        help="reindicizza la knowledge base prima di avviare")
    parser.add_argument("--skip-checks", action="store_true", help="salta i controlli preliminari")
    parser.add_argument("--no-browser", action="store_true", help="non aprire il browser")
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    sys.path.insert(0, str(BASE_DIR))

    print("=" * 66)
    print("  LLM Dementia — Sistema di supporto alla diagnosi differenziale")
    print("=" * 66)

    check_python_env()
    if not args.skip_checks:
        check_ollama()
        check_rag(args.rebuild_rag)
        check_frontend(args.dev)

    vite = start_vite() if args.dev else None
    url = f"http://{args.host}:{args.port}"
    print("-" * 66)
    print(f"  Applicazione : {'http://localhost:5173' if args.dev else url}")
    print(f"  API          : {url}/api")
    print(f"  Documentazione: {url}/docs")
    print("-" * 66)

    if not args.no_browser and not args.dev:
        webbrowser.open(url)

    cmd = [sys.executable, "-m", "uvicorn", "api.main:app",
           "--host", args.host, "--port", str(args.port)]
    if args.reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[..] Arresto richiesto")
    finally:
        if vite:
            vite.terminate()


if __name__ == "__main__":
    main()
