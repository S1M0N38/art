# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Merge titles.json, tags.json, and dimensions.json into paintings.yaml.

Usage:
    uv run scripts/generate/paintings.py
"""

import json
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUTPUT = SCRIPT_DIR / "paintings.yaml"


def main():
    titles = json.loads((SCRIPT_DIR / "titles.json").read_text())
    tags = json.loads((SCRIPT_DIR / "tags.json").read_text())
    dimensions = json.loads((SCRIPT_DIR / "dimensions.json").read_text())
    sort_ids = json.loads((SCRIPT_DIR / "sort_ids.json").read_text())

    # Validate all files have the same UUIDs
    assert titles.keys() == tags.keys() == dimensions.keys() == sort_ids.keys(), "UUID mismatch across files"

    paintings = []
    for uid in sorted(titles):
        paintings.append({
            "id": uid,
            "sort_id": sort_ids[uid],
            "title": titles[uid],
            "width_cm": dimensions[uid]["width_cm"],
            "height_cm": dimensions[uid]["height_cm"],
            "tags": tags[uid],
        })

    OUTPUT.write_text(
        yaml.dump(paintings, allow_unicode=True, default_flow_style=False, sort_keys=False)
    )
    print(f"Wrote {len(paintings)} paintings to {OUTPUT}")


if __name__ == "__main__":
    main()
