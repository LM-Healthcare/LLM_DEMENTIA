"""
Entry point: avvia il server FastAPI con uvicorn.

Uso:
    python run.py                  # sviluppo (reload automatico)
    python run.py --prod           # produzione (no reload)
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="LLM Dementia API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Indirizzo host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Porta (default: 8000)")
    parser.add_argument("--prod", action="store_true", help="Modalità produzione (no reload)")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if not args.prod:
        cmd.append("--reload")

    print(f"[LLM Dementia] Avvio server su http://{args.host}:{args.port}")
    print(f"[LLM Dementia] Docs API: http://{args.host}:{args.port}/docs")
    print(f"[LLM Dementia] Frontend dev: http://localhost:5173 (se npm run dev attivo)")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
