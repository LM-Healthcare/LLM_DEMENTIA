"""Test di regressione del parser JSON e dei metadati di riparazione."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.ollama_client import parse_json_response_detailed


def main() -> None:
    cases = {
        "exact": ('{"step": 1, "primary_diagnosis": "AD"}', "exact"),
        "markdown": ('```json\n{"step": 1}\n```', "exact"),
        "think": ('<think>non salvare</think>{"step": 1}', "exact"),
        "truncated_string": ('{"step": 1, "reasoning": "testo', "repaired"),
        "truncated_array": ('{"step": 1, "items": [1, 2,', "repaired"),
        "invalid": ('non è JSON', "error"),
        "empty": ('', "error"),
    }
    failed = []
    for name, (raw, expected) in cases.items():
        parsed = parse_json_response_detailed(raw)
        actual = parsed["metadata"]["status"]
        ok = actual == expected
        print(f"  [{'ok' if ok else 'FALLITO'}] {name}: {actual}")
        if not ok:
            failed.append(name)
    repaired = parse_json_response_detailed('{"a": 1, "b": [2, 3')
    meta = repaired["metadata"]
    if not meta["repaired"]:
        failed.append("repair_metadata")
    if failed:
        print(f"\nControlli falliti: {', '.join(failed)}")
        sys.exit(1)
    print("\nTutti i controlli del parser superati.")


if __name__ == "__main__":
    main()
