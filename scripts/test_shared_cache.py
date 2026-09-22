"""Test della cache RAG condivisa e indipendente dal modello."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.rag_cache import (
    RAG_CACHE_VERSION,
    cache_manifest,
    cache_root,
    load_shared_bundles,
)


def main() -> None:
    failures = []
    with TemporaryDirectory() as temporary:
        output_root = Path(temporary)
        root = cache_root(output_root, "budson")
        root.mkdir(parents=True)
        manifest = cache_manifest(output_root, "budson")
        fingerprint = manifest["configuration_sha256"]
        bundle = {"sha256": "bundle-test", "context": "patient-specific"}
        (root / "T1.json").write_text(
            json.dumps({
                "cache_version": RAG_CACHE_VERSION,
                "configuration_sha256": fingerprint,
                "patient_code": "T1",
                "rag_mode": "budson",
                "bundle": bundle,
            }),
            encoding="utf-8",
        )

        qwen = load_shared_bundles(output_root, "budson", ["T1"])
        llama = load_shared_bundles(output_root, "budson", ["T1"])
        checks = {
            "path globale senza modello": root == output_root / "rag_cache" / "budson",
            "Qwen e Llama leggono lo stesso bundle": qwen["T1"] == llama["T1"] == bundle,
            "cache version corrente": RAG_CACHE_VERSION >= 3,
        }
        for name, ok in checks.items():
            print(f"  [{'ok' if ok else 'FALLITO'}] {name}")
            if not ok:
                failures.append(name)

        stale = json.loads((root / "T1.json").read_text(encoding="utf-8"))
        stale["configuration_sha256"] = "obsolete"
        (root / "T1.json").write_text(json.dumps(stale), encoding="utf-8")
        try:
            load_shared_bundles(output_root, "budson", ["T1"])
            rejected = False
        except RuntimeError:
            rejected = True
        print(f"  [{'ok' if rejected else 'FALLITO'}] fingerprint obsoleto rifiutato")
        if not rejected:
            failures.append("fingerprint obsoleto")

    if failures:
        print("Controlli falliti: " + ", ".join(failures))
        sys.exit(1)
    print("Cache condivisa corretta.")


if __name__ == "__main__":
    main()
