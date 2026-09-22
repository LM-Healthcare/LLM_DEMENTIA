"""Rigenera results.xlsx da runs.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.export import export_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Esporta evaluation JSONL in Excel")
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    manifest_path = args.experiment_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = export_excel(
        args.experiment_dir / "runs.jsonl",
        args.experiment_dir / "results.xlsx",
        manifest,
    )
    print(output)


if __name__ == "__main__":
    main()
