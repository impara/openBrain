#!/usr/bin/env python3
"""
Import a JSON chat export into OpenBrain.

Expects a JSON file: array of {"content": "...", "external_id": "optional"}.
Run from repo root so brain_core and .env are available:

  python connectors/ingest_chat_export.py path/to/export.json --source chatgpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run from repo root so parent is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

from brain_core import ingest


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a JSON chat export into OpenBrain.")
    ap.add_argument("file", type=Path, help="Path to JSON file (array of {content, external_id?})")
    ap.add_argument("--source", default="import", help="Origin label (e.g. chatgpt, claude)")
    ap.add_argument("--dry-run", action="store_true", help="Print only, do not ingest")
    args = ap.parse_args()

    path = args.file
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: JSON root must be an array of items.", file=sys.stderr)
        sys.exit(1)

    source = (args.source or "import").strip()
    dry_run = args.dry_run

    ok = 0
    skip = 0
    err = 0

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"[{i}] skip: not an object", file=sys.stderr)
            skip += 1
            continue
        content = (item.get("content") or item.get("text") or "").strip()
        if not content:
            skip += 1
            continue
        external_id = item.get("external_id") or item.get("id")
        if isinstance(external_id, (int, float)):
            external_id = str(external_id)

        if dry_run:
            ext = f" external_id={external_id!r}" if external_id else ""
            print(f"[{i}] would ingest len={len(content)}{ext}")
            ok += 1
            continue

        result = ingest(
            content=content,
            source=source,
            external_id=external_id,
        )
        if "Warning" in result or "Error" in result or "Failed" in result:
            print(f"[{i}] {result}", file=sys.stderr)
            err += 1
        elif "Already ingested" in result or "duplicate" in result.lower():
            skip += 1
        else:
            ok += 1

    if dry_run:
        print(f"Dry run: {ok} items would be ingested.")
    else:
        print(f"Done: {ok} ingested, {skip} skipped, {err} errors.")


if __name__ == "__main__":
    main()
