#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "Pillow",
# ]
# ///

"""
Generate optimized image variants for CDN upload.

Produces three variants per source image:
  - originals/  (full-size JPG copy)
  - thumbs/     (≤800px wide WebP, quality 80)
  - placeholders/ (≤20px wide WebP, quality 10)

Usage:
    uv run scripts/utils/optimize.py                          # optimize all from images/processed/
    uv run scripts/utils/optimize.py images/processed/abc.jpg # single file
    uv run scripts/utils/optimize.py --skip-existing          # skip if all 3 variants exist
    uv run scripts/utils/optimize.py --force                  # overwrite existing (default behavior)
"""

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PROCESSED_DIR = Path("images/processed")
OPTIMIZED_DIR = Path("images/optimized")

VARIANTS = {
    "originals": {"max_width": None, "quality": None, "ext": ".jpg"},
    "thumbs": {"max_width": 800, "quality": 80, "ext": ".webp"},
    "placeholders": {"max_width": 20, "quality": 10, "ext": ".webp"},
}


def variant_path(uuid: str, variant: str) -> Path:
    """Return the output path for a given variant."""
    cfg = VARIANTS[variant]
    return OPTIMIZED_DIR / variant / f"{uuid}{cfg['ext']}"


def all_variants_exist(uuid: str) -> bool:
    """Check whether all 3 output files already exist."""
    return all(variant_path(uuid, v).exists() for v in VARIANTS)


def optimize_file(src: Path, skip_existing: bool) -> bool | None:
    """Generate all variants for a single source image.

    Returns True if optimized, False if skipped, None if invalid.
    """
    uuid = src.stem

    if skip_existing and all_variants_exist(uuid):
        return False

    try:
        img = Image.open(src)
        img.load()
    except (UnidentifiedImageError, OSError):
        print(f"  {src.name}: skipped (not a valid image)")
        return None

    orig_size = f"{img.size[0]}x{img.size[1]}"

    # originals — straight copy, no re-encoding
    dst_orig = variant_path(uuid, "originals")
    shutil.copy2(src, dst_orig)

    # thumbs and placeholders — resize + WebP
    for variant in ("thumbs", "placeholders"):
        cfg = VARIANTS[variant]
        max_width = cfg["max_width"]
        dst = variant_path(uuid, variant)

        w, h = img.size
        if w > max_width:
            ratio = max_width / w
            new_size = (max_width, round(h * ratio))
            resized = img.resize(new_size, Image.LANCZOS)
        else:
            resized = img.copy()

        resized.save(dst, format="webp", quality=cfg["quality"])

    print(
        f"  {src.name}: {orig_size}"
        f" → originals/{uuid}.jpg,"
        f" thumbs/{uuid}.webp,"
        f" placeholders/{uuid}.webp"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate optimized image variants for CDN upload"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to process (default: all in images/processed/)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images where all 3 variants already exist",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs (default behavior, for explicitness)",
    )
    args = parser.parse_args()

    # Create output directories
    for variant in VARIANTS:
        (OPTIMIZED_DIR / variant).mkdir(parents=True, exist_ok=True)

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(PROCESSED_DIR.glob("*.jpg"))

    if not files:
        print(f"No images found in {PROCESSED_DIR}/")
        sys.exit(1)

    print(f"Optimizing {len(files)} image(s)...")

    optimized = 0
    skipped = 0
    for f in files:
        result = optimize_file(f, skip_existing=args.skip_existing)
        if result is True:
            optimized += 1
        elif result is False:
            skipped += 1

    print(f"Done. {optimized} optimized, {skipped} skipped.")


if __name__ == "__main__":
    main()
