#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx",
#   "python-dotenv",
#   "tqdm",
# ]
# ///
"""
Generate Italian titles for paintings using a vision LLM.

Sends each thumbnail to an OpenAI-compatible vision endpoint and asks for a
short, evocative title in Italian. Results are saved to scripts/titles.json.

Reads API settings from environment variables or a .env file in the project root.
Required env vars: VLM_BASE_API, VLM_MODEL.

Usage:
    uv run scripts/utils/generate_titles.py                                        # all thumbs
    uv run scripts/utils/generate_titles.py images/optimized/thumbs/00cb4951.webp  # single file
    uv run scripts/utils/generate_titles.py --skip-existing                        # resume interrupted run
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

THUMBS_DIR = Path("images/optimized/thumbs")
OUTPUT_FILE = Path("scripts/titles.json")
ENV_FILE = Path(".env")

SYSTEM_PROMPT = (
    "Sei un esperto d'arte italiana. "
    "Osserva attentamente questo dipinto a olio e genera un titolo breve, "
    "evocativo e in lingua italiana. "
    "Rispondi SOLO con il titolo, senza virgolette, senza spiegazioni."
)


def get_config() -> dict[str, str]:
    """Load VLM API settings from environment / .env file."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)

    base_url = os.environ.get("VLM_BASE_API", "")
    model = os.environ.get("VLM_MODEL", "")

    if not base_url or not model:
        print("Error: VLM_BASE_API and VLM_MODEL must be set.")
        print("Create a .env file from .env.example or export them in your shell.")
        sys.exit(1)

    return {"base_url": base_url.rstrip("/"), "model": model}


def encode_image(path: Path) -> str:
    """Base64-encode an image file."""
    data = path.read_bytes()
    return base64.b64encode(data).decode()


def generate_title(client: httpx.Client, config: dict, image_b64: str) -> str:
    """Call the vision LLM to generate a title for a painting."""
    resp = client.post(
        f"{config['base_url']}/chat/completions",
        json={
            "model": config["model"],
            "max_tokens": 64,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate titles for paintings via vision LLM",
    )
    parser.add_argument("files", nargs="*", help="Specific thumb files (default: all)")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip IDs already in output"
    )
    args = parser.parse_args()

    config = get_config()

    if args.files:
        thumbs = [Path(f) for f in args.files]
    else:
        thumbs = sorted(THUMBS_DIR.glob("*.webp"))

    if not thumbs:
        print("No thumbnail images found.")
        sys.exit(1)

    # Load existing results
    results: dict[str, str] = {}
    if OUTPUT_FILE.exists():
        results = json.loads(OUTPUT_FILE.read_text())

    if args.skip_existing:
        thumbs = [t for t in thumbs if t.stem not in results]

    print(f"Generating titles for {len(thumbs)} painting(s)...")

    with httpx.Client() as client:
        for thumb in tqdm(thumbs, desc="Titles"):
            painting_id = thumb.stem
            try:
                image_b64 = encode_image(thumb)
                title = generate_title(client, config, image_b64)
                results[painting_id] = title
                tqdm.write(f"  {painting_id}: {title}")
            except Exception as e:
                tqdm.write(f"  {painting_id}: ERROR - {e}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
    )
    print(f"Done. {len(results)} titles saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
