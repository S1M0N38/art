#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow"]
# ///
"""
Generate plausible cm dimensions for paintings based on pixel aspect ratios.

Reads each image from images/optimized/originals/, computes the aspect ratio,
and maps it to the closest standard Italian canvas size (20–80 cm range,
rounded to nearest 5 cm). Results are deterministic — a hash of the painting
UUID selects among same-ratio size variants so small/large sizes are evenly
distributed across the collection.

Usage:
    uv run scripts/generate/dimensions.py                                         # all images
    uv run scripts/generate/dimensions.py images/optimized/originals/00cb4951.jpg # single file
    uv run scripts/generate/dimensions.py --skip-existing                         # resume
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


ORIGINALS_DIR = Path("images/optimized/originals")
OUTPUT_FILE = Path("scripts/generate/dimensions.json")

# Standard Italian canvas sizes (width_cm, height_cm).
# Each aspect-ratio family has 2–3 size options so paintings of the same
# proportions get a mix of small and large canvases.
STANDARD_SIZES: list[tuple[int, int]] = [
    # ── Landscape (w > h) ──────────────────────────────────
    (30, 20),   # 3:2 small
    (60, 40),   # 3:2 large
    (35, 25),   # 7:5 small
    (70, 50),   # 7:5 large
    (40, 30),   # 4:3 small
    (80, 60),   # 4:3 large
    (50, 30),   # 5:3
    (50, 35),   # ~10:7
    (50, 40),   # 5:4
    (60, 50),   # 6:5
    (70, 40),   # 7:4 wide
    (80, 45),   # ~16:9 very wide
    # ── Portrait (h > w) ───────────────────────────────────
    (20, 30),   # 2:3 small
    (40, 60),   # 2:3 large
    (25, 35),   # 5:7 small
    (50, 70),   # 5:7 large
    (30, 40),   # 3:4 small
    (60, 80),   # 3:4 large
    (30, 50),   # 3:5
    (35, 50),   # 7:10
    (40, 50),   # 4:5
    (50, 60),   # 5:6
    (40, 55),   # ~3:4 medium
    (45, 80),   # ~9:16 very tall
]


def find_best_size(w_px: int, h_px: int, painting_id: str) -> tuple[int, int]:
    """Map pixel dimensions to the closest standard canvas size.

    When multiple standard sizes share the same aspect ratio (e.g. 30×20 and
    60×40 are both 3:2), one is chosen deterministically via an MD5 hash of the
    painting UUID so the collection gets a realistic mix of sizes.
    """
    ar = w_px / h_px

    # Rank all standard sizes by how close their AR is to the image AR.
    ranked = sorted(STANDARD_SIZES, key=lambda s: abs(s[0] / s[1] - ar))

    # Collect all sizes that share the best AR (within a tiny tolerance).
    best_ar = ranked[0][0] / ranked[0][1]
    candidates = [s for s in ranked if abs(s[0] / s[1] - best_ar) < 0.01]

    # Deterministic pick among candidates.
    idx = int(hashlib.md5(painting_id.encode()).hexdigest(), 16) % len(candidates)
    return candidates[idx]


def print_summary(results: dict[str, dict[str, int]]) -> None:
    """Print a distribution table of assigned sizes."""
    size_counts: Counter[str] = Counter()
    for dims in results.values():
        key = f"{dims['width_cm']}×{dims['height_cm']}"
        size_counts[key] += 1

    print("\nDistribution:")
    for size, count in size_counts.most_common():
        w, h = size.split("×")
        orientation = "L" if int(w) > int(h) else "P"
        print(f"  {size:>7}  ({orientation}):  {count:>3} paintings")

    print(f"\n  Total: {len(results)} paintings")

    if results:
        areas = [d["width_cm"] * d["height_cm"] for d in results.values()]
        print(f"  Area range: {min(areas)}–{max(areas)} cm²")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cm dimensions for paintings from pixel aspect ratios",
    )
    parser.add_argument("files", nargs="*", help="Specific image files (default: all)")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip IDs already in output",
    )
    args = parser.parse_args()

    if args.files:
        images = [Path(f) for f in args.files]
    else:
        images = sorted(ORIGINALS_DIR.glob("*.jpg"))

    if not images:
        sys.exit("No images found.")

    # Load existing results for idempotent merge.
    results: dict[str, dict[str, int]] = {}
    if OUTPUT_FILE.exists():
        results = json.loads(OUTPUT_FILE.read_text())

    if args.skip_existing:
        images = [img for img in images if img.stem not in results]

    if not images:
        print("Nothing to do — all images already processed.")
        print_summary(results)
        return

    print(f"Processing {len(images)} images…")

    for img_path in images:
        painting_id = img_path.stem
        with Image.open(img_path) as img:
            w_px, h_px = img.size

        w_cm, h_cm = find_best_size(w_px, h_px, painting_id)
        results[painting_id] = {"width_cm": w_cm, "height_cm": h_cm}

    # Write output (sorted keys for stable diffs).
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    print(f"\nWrote {len(results)} entries to {OUTPUT_FILE}")
    print_summary(results)


if __name__ == "__main__":
    main()
