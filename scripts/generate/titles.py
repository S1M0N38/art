#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx",
#   "pillow",
#   "python-dotenv",
#   "tqdm",
# ]
# ///
"""
Generate Italian titles for paintings using a vision LLM.

Sends each thumbnail to an OpenAI-compatible vision endpoint and asks for a
short, evocative title in Italian. Results are saved to scripts/generate/titles.json.

Reads API settings from environment variables or a .env file in the project root.
Required env vars: VLM_BASE_API, VLM_MODEL.

Usage:
    uv run scripts/generate/titles.py                                        # all thumbs
    uv run scripts/generate/titles.py images/optimized/thumbs/00cb4951.webp  # single file
    uv run scripts/generate/titles.py --skip-existing                        # resume interrupted run
"""

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

THUMBS_DIR = Path("images/optimized/thumbs")
OUTPUT_FILE = Path("scripts/generate/titles.json")
ENV_FILE = Path(".env")

SYSTEM_PROMPT = (
    "Sei un critico d'arte e curatore museale italiano. "
    "Osserva attentamente il dipinto a olio e assegnagli un titolo breve "
    "ed evocativo in lingua italiana, come apparrebbe sulla targhetta di "
    "una galleria. Il titolo deve catturare l'essenza del soggetto o "
    "dell'atmosfera dell'opera."
)

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "painting_title",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "titolo": {
                    "type": "string",
                    "description": "Titolo breve ed evocativo del dipinto in italiano",
                },
            },
            "required": ["titolo"],
            "additionalProperties": False,
        },
    },
}


def get_config() -> dict[str, str]:
    """Load VLM API settings from environment / .env file."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)

    base_url = os.environ.get("VLM_BASE_API", "")
    model = os.environ.get("VLM_MODEL", "")

    if not base_url or not model:
        sys.exit("Error: VLM_BASE_API and VLM_MODEL must be set. "
                 "Create a .env file from .env.example or export them.")

    return {"base_url": base_url.rstrip("/"), "model": model}


def encode_image(path: Path) -> str:
    """Load an image, convert to JPEG, and return as base64 string."""
    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


MAX_CONCURRENT = 4


async def generate_title(
    client: httpx.AsyncClient,
    config: dict,
    image_b64: str,
) -> str:
    """Call the vision LLM to generate a title for a painting.

    Uses structured outputs (json_schema response_format) so the model
    returns ``{"titolo": "..."}`` deterministically.
    """
    resp = await client.post(
        f"{config['base_url']}/chat/completions",
        json={
            "model": config["model"],
            "response_format": RESPONSE_SCHEMA,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)["titolo"].strip("*_`#")


async def process_thumb(
    thumb: Path,
    client: httpx.AsyncClient,
    config: dict,
    semaphore: asyncio.Semaphore,
    results: dict[str, str],
    progress: tqdm,
) -> None:
    """Encode one thumbnail, call the LLM, and store the result."""
    painting_id = thumb.stem
    async with semaphore:
        try:
            image_b64 = encode_image(thumb)
            title = await generate_title(client, config, image_b64)
            results[painting_id] = title
            progress.write(f"  {painting_id}: {title}")
        except Exception as e:
            progress.write(f"  {painting_id}: ERROR - {e}")
        finally:
            progress.update()


async def main() -> None:
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
        sys.exit(1)

    # Load existing results
    results: dict[str, str] = {}
    if OUTPUT_FILE.exists():
        results = json.loads(OUTPUT_FILE.read_text())

    if args.skip_existing:
        thumbs = [t for t in thumbs if t.stem not in results]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient() as client:
        with tqdm(total=len(thumbs), desc="Titles") as progress:
            tasks = [
                process_thumb(t, client, config, semaphore, results, progress)
                for t in thumbs
            ]
            await asyncio.gather(*tasks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
    )


if __name__ == "__main__":
    asyncio.run(main())
