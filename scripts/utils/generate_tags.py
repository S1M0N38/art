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
Generate tags for paintings using a vision LLM with structured outputs.

Sends each thumbnail to an OpenAI-compatible vision endpoint and asks for 2–4
tags from a predefined set (paesaggio, città, interni, astratto, ritratto,
natura morta, notturno, fiori). Uses JSON schema structured outputs to guarantee
well-formed responses. Results are saved to scripts/tags.json.

Reads API settings from environment variables or a .env file in the project root.
Required env vars: VLM_BASE_API, VLM_MODEL.

Usage:
    uv run scripts/utils/generate_tags.py                                        # all thumbs
    uv run scripts/utils/generate_tags.py images/optimized/thumbs/00cb4951.webp  # single file
    uv run scripts/utils/generate_tags.py --skip-existing                        # resume interrupted run
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
OUTPUT_FILE = Path("scripts/tags.json")
ENV_FILE = Path(".env")

VALID_TAGS = [
    "paesaggio",
    "città",
    "interni",
    "astratto",
    "ritratto",
    "natura morta",
    "notturno",
    "fiori",
]

TAGS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "painting_tags",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": VALID_TAGS,
                    },
                }
            },
            "required": ["tags"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "Sei un esperto d'arte italiana. "
    "Osserva attentamente questo dipinto a olio e assegna da 2 a 4 etichette "
    "scelte ESCLUSIVAMENTE da questo elenco:\n"
    "paesaggio, città, interni, astratto, ritratto, natura morta, notturno, fiori\n\n"
    "Rispondi con un oggetto JSON contenente una chiave \"tags\" con la lista delle etichette scelte."
)


def get_config() -> dict[str, str]:
    """Load VLM API settings from environment / .env file."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)

    base_url = os.environ.get("VLM_BASE_API", "")
    model = os.environ.get("VLM_MODEL", "")

    if not base_url or not model:
        sys.exit(
            "Error: VLM_BASE_API and VLM_MODEL must be set.\n"
            "Create a .env file from .env.example or export them in your shell."
        )

    return {"base_url": base_url.rstrip("/"), "model": model}


def encode_image(path: Path) -> str:
    """Load an image, convert to JPEG, and return as base64 string."""
    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


MAX_CONCURRENT = 4


async def generate_tags(
    client: httpx.AsyncClient, config: dict, image_b64: str,
) -> list[str]:
    """Call the vision LLM to generate tags for a painting."""
    resp = await client.post(
        f"{config['base_url']}/chat/completions",
        json={
            "model": config["model"],
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
            "response_format": TAGS_SCHEMA,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(content)
    # Validate tags even though the schema constrains them
    return [t for t in data["tags"] if t in VALID_TAGS]


async def process_thumb(
    thumb: Path,
    client: httpx.AsyncClient,
    config: dict,
    semaphore: asyncio.Semaphore,
    results: dict[str, list[str]],
    pbar: tqdm,
) -> None:
    """Encode one thumbnail, call the LLM (rate-limited), and store results."""
    painting_id = thumb.stem
    async with semaphore:
        try:
            image_b64 = encode_image(thumb)
            tags = await generate_tags(client, config, image_b64)
            results[painting_id] = tags
            pbar.set_postfix_str(f"{painting_id}: {', '.join(tags)}")
        except Exception as e:
            pbar.set_postfix_str(f"{painting_id}: ERROR - {e}")
        finally:
            pbar.update(1)


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tags for paintings via vision LLM",
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
        sys.exit("No thumbnail images found.")

    # Load existing results
    results: dict[str, list[str]] = {}
    if OUTPUT_FILE.exists():
        results = json.loads(OUTPUT_FILE.read_text())

    if args.skip_existing:
        thumbs = [t for t in thumbs if t.stem not in results]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient() as client:
        with tqdm(total=len(thumbs), desc="Tags") as pbar:
            tasks = [
                process_thumb(thumb, client, config, semaphore, results, pbar)
                for thumb in thumbs
            ]
            await asyncio.gather(*tasks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
    )


if __name__ == "__main__":
    asyncio.run(async_main())
