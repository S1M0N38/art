#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "Pillow",
# ]
# ///

"""
Generate optimized image variants for the gallery.

Reads images from images/processed/ (named {uuid}.jpg) and generates:
  - originals: full-res copy
  - thumbs: ≤800px WebP
  - placeholders: ≤20px WebP

All output goes to images/optimized/{originals,thumbs,placeholders}/.
public/images/ is a symlink to images/optimized/ for the website.

Usage:
    uv run scripts/utils/optimize.py                     # all images in images/processed/
    uv run scripts/utils/optimize.py --skip-existing     # skip if all 3 variants exist
"""

import shutil
import sys
import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PROCESSED_DIR = Path("images/processed")
OUTPUT_DIR = Path("images/optimized")

VARIANTS = {
    "originals": {"max_width": None, "quality": None, "ext": ".jpg"},
    "thumbs": {"max_width": 800, "quality": 80, "ext": ".webp"},
    "placeholders": {"max_width": 20, "quality": 10, "ext": ".webp"},
}


def variant_path(painting_id: str, variant: str) -> Path:
    """Return the output path for a given variant."""
    cfg = VARIANTS[variant]
    return OUTPUT_DIR / variant / f"{painting_id}{cfg['ext']}"


def all_variants_exist(painting_id: str) -> bool:
    """Check whether all 3 output files already exist."""
    return all(variant_path(painting_id, v).exists() for v in VARIANTS)


def discover_processed() -> list[tuple[str, Path]]:
    """Find all images in images/processed/.

    Returns list of (painting_id, path) tuples sorted by painting_id.
    """
    if not PROCESSED_DIR.exists():
        raise SystemExit(f"Error: {PROCESSED_DIR}/ not found")

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for p in PROCESSED_DIR.glob(ext):
            images.append((p.stem, p))

    images.sort(key=lambda x: x[0])
    return images


def optimize_file(
    src: Path, painting_id: str, skip_existing: bool
) -> bool | None:
    """Generate all variants for a single source image.

    Returns True if optimized, False if skipped, None if invalid.
    """
    if skip_existing and all_variants_exist(painting_id):
        return False

    try:
        img = Image.open(src)
        img.load()
    except (UnidentifiedImageError, OSError):
        print(f"  {src.name}: skipped (not a valid image)")
        return None

    orig_size = f"{img.size[0]}x{img.size[1]}"

    # originals — straight copy, no re-encoding
    dst_orig = variant_path(painting_id, "originals")
    shutil.copy2(src, dst_orig)

    # thumbs and placeholders — resize + WebP
    for variant in ("thumbs", "placeholders"):
        cfg = VARIANTS[variant]
        max_width = cfg["max_width"]
        dst = variant_path(painting_id, variant)

        w, h = img.size
        if w > max_width:
            ratio = max_width / w
            new_size = (max_width, round(h * ratio))
            resized = img.resize(new_size, Image.LANCZOS)
        else:
            resized = img.copy()

        resized.save(dst, format="webp", quality=cfg["quality"])

    print(
        f"  {painting_id}: {orig_size}"
        f" → originals/{painting_id}.jpg,"
        f" thumbs/{painting_id}.webp,"
        f" placeholders/{painting_id}.webp"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate optimized image variants from images/processed/"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip paintings where all 3 variants already exist",
    )
    args = parser.parse_args()

    # Create output directories
    for variant in VARIANTS:
        (OUTPUT_DIR / variant).mkdir(parents=True, exist_ok=True)

    # Discover images
    processed = discover_processed()
    print(f"Found {len(processed)} images in {PROCESSED_DIR}/")

    optimized = 0
    skipped = 0

    for painting_id, src in processed:
        result = optimize_file(src, painting_id, skip_existing=args.skip_existing)
        if result is True:
            optimized += 1
        elif result is False:
            skipped += 1

    print(f"Done. {optimized} optimized, {skipped} skipped.")


if __name__ == "__main__":
    main()
