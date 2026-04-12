#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Generate paintings.yaml from arthag archive.csv + optional pipeline outputs.

Reads the arthag archive CSV as source of truth for painting metadata.
Optionally merges titles.json, tags.json, sort_ids.json from the ML
pipelines (when present) to override/augment CSV data.

Usage:
    uv run scripts/generate/paintings.py
    uv run scripts/generate/paintings.py --csv /path/to/archive.csv
"""

import csv
import json
import argparse
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
CSV_DEFAULT = Path.home() / "Downloads" / "artag" / "data" / "archive.csv"
OUTPUT_YAML = Path("src/content/paintings.yaml")


def load_json(path: Path) -> dict | None:
    """Load a JSON file if it exists, else return None."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paintings.yaml from arthag archive CSV"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_DEFAULT,
        help=f"Path to archive.csv (default: {CSV_DEFAULT})",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        raise SystemExit(f"Error: CSV not found at {csv_path}")

    # Load optional pipeline outputs
    titles = load_json(SCRIPT_DIR / "titles.json")
    tags = load_json(SCRIPT_DIR / "tags.json")
    sort_ids = load_json(SCRIPT_DIR / "sort_ids.json")

    # Read CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Read {len(rows)} paintings from {csv_path}")

    if titles:
        print(f"  Merging titles.json ({len(titles)} entries)")
    if tags:
        print(f"  Merging tags.json ({len(tags)} entries)")
    if sort_ids:
        print(f"  Merging sort_ids.json ({len(sort_ids)} entries)")

    paintings = []
    for row in rows:
        uid = row["id"]

        # Base fields from CSV (some rows may have empty year/title)
        year_str = row["year"].strip()
        title = row["title"].strip() or "Senza titolo"

        entry = {
            "id": uid,
            "sort_id": int(row["custom_id"]),
            "title": title,
            "year": int(year_str) if year_str else None,
            "width_cm": int(row["width"]),
            "height_cm": int(row["height"]),
            "technique": row["technique"].strip(),
            "tags": [],
        }

        # Override with pipeline outputs when available
        if sort_ids and uid in sort_ids:
            entry["sort_id"] = sort_ids[uid]
        if titles and uid in titles:
            entry["title"] = titles[uid]
        if tags and uid in tags:
            entry["tags"] = tags[uid]

        paintings.append(entry)

    # Sort by sort_id for deterministic output
    paintings.sort(key=lambda p: p["sort_id"])

    # Write YAML
    OUTPUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_YAML.write_text(
        yaml.dump(paintings, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(paintings)} paintings to {OUTPUT_YAML}")


if __name__ == "__main__":
    main()
