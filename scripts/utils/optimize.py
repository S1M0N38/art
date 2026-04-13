#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "Pillow",
#   "pyyaml",
# ]
# ///

"""
Generate optimized image variants for the gallery (front + back).

Front: reads originals from images/front/original/, generates thumbs + placeholders
       into images/front/thumbs/ and images/front/placeholders/.

Back:  reads the painting_id → back_photo mapping from paintings.yaml, finds source
       images in images/back/original/ (already renamed to painting_id), and generates
       thumbs + placeholders into images/back/thumbs/ and images/back/placeholders/.

Usage:
    uv run scripts/utils/optimize.py                     # front + back
    uv run scripts/utils/optimize.py --front             # front only
    uv run scripts/utils/optimize.py --back              # back only
    uv run scripts/utils/optimize.py --skip-existing     # skip if variants exist
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

IMAGES_DIR = Path("images")

VARIANTS = {
    "thumbs": {"max_width": 800, "quality": 80, "ext": ".webp"},
    "placeholders": {"max_width": 20, "quality": 10, "ext": ".webp"},
}


def variant_path(side: str, painting_id: str, variant: str) -> Path:
    """Return the output path for a given side/variant."""
    cfg = VARIANTS[variant]
    return IMAGES_DIR / side / variant / f"{painting_id}{cfg['ext']}"


def all_variants_exist(side: str, painting_id: str) -> bool:
    """Check whether all output variants already exist for a painting."""
    return all(variant_path(side, painting_id, v).exists() for v in VARIANTS)


def optimize_file(src: Path, painting_id: str, side: str, skip_existing: bool) -> bool | None:
    """Generate thumbs + placeholders for a single source image.

    Returns True if optimized, False if skipped, None if invalid.
    """
    if skip_existing and all_variants_exist(side, painting_id):
        return False

    try:
        img = Image.open(src)
        img.load()
    except (UnidentifiedImageError, OSError):
        print(f"  {src.name}: skipped (not a valid image)")
        return None

    orig_size = f"{img.size[0]}x{img.size[1]}"

    for variant, cfg in VARIANTS.items():
        max_width = cfg["max_width"]
        dst = variant_path(side, painting_id, variant)

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
        f" → {side}/thumbs/{painting_id}.webp,"
        f" {side}/placeholders/{painting_id}.webp"
    )
    return True


def optimize_front(skip_existing: bool) -> tuple[int, int]:
    """Optimize all front images. Returns (optimized, skipped)."""
    original_dir = IMAGES_DIR / "front" / "original"
    if not original_dir.exists():
        raise SystemExit(f"Error: {original_dir}/ not found")

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        images.extend(original_dir.glob(ext))
    images.sort(key=lambda p: p.stem)

    print(f"\n[front] Found {len(images)} originals in {original_dir}/")

    optimized, skipped = 0, 0
    for src in images:
        result = optimize_file(src, src.stem, "front", skip_existing)
        if result is True:
            optimized += 1
        elif result is False:
            skipped += 1

    print(f"[front] {optimized} optimized, {skipped} skipped.")
    return optimized, skipped


def optimize_back(skip_existing: bool) -> tuple[int, int, int]:
    """Optimize all back images. Returns (optimized, skipped, missing)."""
    original_dir = IMAGES_DIR / "back" / "original"
    if not original_dir.exists():
        raise SystemExit(f"Error: {original_dir}/ not found")

    # Load paintings.yaml to know which paintings have back photos
    paintings_yaml = Path("src/content/paintings.yaml")
    if not paintings_yaml.exists():
        raise SystemExit(f"Error: {paintings_yaml} not found. Run paintings.py first.")

    paintings = yaml.safe_load(paintings_yaml.read_text(encoding="utf-8"))

    # Collect painting IDs that have a back photo and a source file exists
    painting_ids = [p["id"] for p in paintings if p.get("back_photo")]

    print(f"\n[back] Found {len(painting_ids)} paintings with back photos in YAML")
    print(f"[back] Source directory: {original_dir}/")

    optimized, skipped, missing = 0, 0, 0
    for painting_id in painting_ids:
        # Source is already named by painting_id in back/original/
        src = original_dir / f"{painting_id}.jpg"
        if not src.exists():
            # Try other extensions
            for ext in (".jpeg", ".png", ".webp"):
                alt = original_dir / f"{painting_id}{ext}"
                if alt.exists():
                    src = alt
                    break
            else:
                missing += 1
                continue

        result = optimize_file(src, painting_id, "back", skip_existing)
        if result is True:
            optimized += 1
        elif result is False:
            skipped += 1

    print(f"[back] {optimized} optimized, {skipped} skipped, {missing} missing source files.")
    return optimized, skipped, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate optimized image variants (front + back)"
    )
    parser.add_argument(
        "--front",
        action="store_true",
        help="Optimize front images only",
    )
    parser.add_argument(
        "--back",
        action="store_true",
        help="Optimize back images only",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip paintings where all variants already exist",
    )
    args = parser.parse_args()

    # Default: both front and back
    do_front = not args.back or args.front
    do_back = not args.front or args.back

    # Create output directories
    for side in ("front", "back"):
        for variant in VARIANTS:
            (IMAGES_DIR / side / variant).mkdir(parents=True, exist_ok=True)

    total_optimized = 0
    total_skipped = 0

    if do_front:
        opt, skip = optimize_front(skip_existing=args.skip_existing)
        total_optimized += opt
        total_skipped += skip

    if do_back:
        opt, skip, _ = optimize_back(skip_existing=args.skip_existing)
        total_optimized += opt
        total_skipped += skip

    print(f"\nDone. {total_optimized} optimized, {total_skipped} skipped.")


if __name__ == "__main__":
    main()
