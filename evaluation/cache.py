"""CLI dedicata al precalcolo RAG condiviso tra tutti i modelli."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BASE_DIR, DATABASE_PATH, DOCS_FOLDER, OLLAMA_EMBED_MODEL, RAG_CORPORA
from data.loader import get_all_codes, get_database
from evaluation.rag_cache import cache_root, precompute_cache
from llm.ollama_client import get_model_names, is_ollama_running


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precalcolo cache RAG condivisa")
    parser.add_argument(
        "--rag-mode", choices=[*sorted(RAG_CORPORA), "all"], default="all"
    )
    parser.add_argument("--patients", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-root", type=Path, default=BASE_DIR / "results" / "evaluation"
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    return parser.parse_args()


def required_files(modes: list[str]) -> None:
    paths = [Path(DATABASE_PATH)]
    paths.extend(
        Path(DOCS_FOLDER) / name
        for mode in modes
        for name in RAG_CORPORA[mode]
    )
    missing = sorted({str(path) for path in paths if not path.is_file()})
    if missing:
        raise FileNotFoundError(
            "File mancanti:\n  - " + "\n  - ".join(missing)
            + "\nVedere SERVER_README.md."
        )


def select_patients(requested: list[str] | None, limit: int | None) -> list[str]:
    all_codes = [str(code) for code in get_all_codes(get_database())]
    if requested:
        unknown = sorted(set(requested) - set(all_codes))
        if unknown:
            raise ValueError("Pazienti non trovati: " + ", ".join(unknown))
        codes = requested
    else:
        codes = all_codes
    return codes[:limit] if limit else codes


def main() -> None:
    args = parse_args()
    modes = sorted(RAG_CORPORA) if args.rag_mode == "all" else [args.rag_mode]
    required_files(modes)
    if not is_ollama_running():
        sys.exit("Ollama non raggiungibile")
    models = get_model_names()
    if not any(name.split(":")[0] == OLLAMA_EMBED_MODEL.split(":")[0] for name in models):
        sys.exit(f"Embedding model non installato: {OLLAMA_EMBED_MODEL}")
    patients = select_patients(args.patients, args.limit)

    print(f"Pazienti: {len(patients)}")
    print(f"Modalità: {', '.join(modes)}")
    for mode in modes:
        print(f"\nCache condivisa {mode}: {cache_root(args.output_root, mode)}")
        try:
            precompute_cache(
                args.output_root,
                mode,
                patients,
                refresh=args.refresh,
                force_unlock=args.force_unlock,
            )
        except KeyboardInterrupt:
            print("\nInterruzione: cache già completate salvate. Rilanciare lo stesso comando.")
            return
    print("\nPrecalcolo cache RAG completato.")


if __name__ == "__main__":
    main()
