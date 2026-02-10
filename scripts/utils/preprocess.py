#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "Pillow",
# ]
# ///

"""
Preprocess painting images: auto-crop white/light borders.

Usage:
    uv run scripts/utils/preprocess.py                          # process all images in images/raw/
    uv run scripts/utils/preprocess.py images/raw/01.jpg        # process a single file
    uv run scripts/utils/preprocess.py --threshold 230          # custom brightness threshold (default: 240)
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageFilter, UnidentifiedImageError

RAW_DIR = Path("images/raw")
PROCESSED_DIR = Path("images/processed")


def auto_crop(img: Image.Image, threshold: int = 240, margin: int = 2) -> Image.Image:
    """Remove white/light borders from a painting photo.

    Works by converting to grayscale, finding rows/columns where the average
    brightness drops below `threshold` (i.e. actual painting content), and
    cropping to that bounding box.
    """
    gray = img.convert("L")
    # Slight blur to reduce noise at the edges
    gray = gray.filter(ImageFilter.GaussianBlur(radius=2))
    pixels = gray.load()
    w, h = gray.size

    def row_avg(y: int) -> float:
        return sum(pixels[x, y] for x in range(w)) / w  # type: ignore

    def col_avg(x: int) -> float:
        return sum(pixels[x, y] for y in range(h)) / h  # type: ignore

    # Find first/last rows and columns that are not white
    top = 0
    for y in range(h):
        if row_avg(y) < threshold:
            top = y
            break

    bottom = h - 1
    for y in range(h - 1, -1, -1):
        if row_avg(y) < threshold:
            bottom = y
            break

    left = 0
    for x in range(w):
        if col_avg(x) < threshold:
            left = x
            break

    right = w - 1
    for x in range(w - 1, -1, -1):
        if col_avg(x) < threshold:
            right = x
            break

    # Apply a small inward margin to ensure we clip any residual edge
    top = min(top + margin, bottom)
    bottom = max(bottom - margin, top)
    left = min(left + margin, right)
    right = max(right - margin, left)

    return img.crop((left, top, right + 1, bottom + 1))


def process_file(src: Path, dst: Path, threshold: int) -> None:
    try:
        img = Image.open(src)
        img.load()
    except (UnidentifiedImageError, OSError):
        print(f"  {src.name}: skipped (not a valid image)")
        return
    cropped = auto_crop(img, threshold=threshold)
    cropped.save(dst, quality=95)
    orig = f"{img.size[0]}x{img.size[1]}"
    final = f"{cropped.size[0]}x{cropped.size[1]}"
    print(f"  {src.name}: {orig} -> {final}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-crop white borders from painting images"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to process (default: all in images/raw/)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=240,
        help="Brightness threshold for border detection (0-255, default: 240)",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = (
            sorted(RAW_DIR.glob("*.jpg"))
            + sorted(RAW_DIR.glob("*.jpeg"))
            + sorted(RAW_DIR.glob("*.png"))
        )

    if not files:
        print(f"No images found in {RAW_DIR}/")
        sys.exit(1)

    print(f"Processing {len(files)} image(s)...")
    for f in files:
        dst = PROCESSED_DIR / f.name
        process_file(f, dst, threshold=args.threshold)

    print("Done.")


if __name__ == "__main__":
    main()
