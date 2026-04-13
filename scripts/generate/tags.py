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
tags from a predefined set. Includes the painting's real title (from archive.csv)
to help the model contextualize the scene. Uses JSON schema structured outputs
to guarantee well-formed responses. Results are saved to scripts/generate/tags.json.

Reads API settings from environment variables or a .env file in the project root.
Required env vars: VLM_BASE_API, VLM_MODEL.

Usage:
    uv run scripts/generate/tags.py                  # all paintings
    uv run scripts/generate/tags.py --csv /path.csv  # custom CSV path
    uv run scripts/generate/tags.py --ids abc def    # specific painting IDs
    uv run scripts/generate/tags.py --skip-existing  # resume interrupted run
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

THUMBS_DIR = Path("images/front/thumbs")
OUTPUT_FILE = Path("scripts/generate/tags.json")
ENV_FILE = Path(".env")
DEFAULT_CSV = Path(os.environ.get(
    "ARTAG_CSV", str(Path(__file__).resolve().parents[2] / "data" / "archive.csv"),
))

VALID_TAGS = [
    "paesaggio",
    "città",
    "interni",
    "notturno",
    "fiori",
    "natura morta",
    "figura",
    "marina",
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
    "paesaggio, città, interni, notturno, fiori, natura morta, figura, marina\n\n"
    "Definizioni:\n"
    "- paesaggio: natura, campagna, montagna, collina, prati, campi, fiume\n"
    "- città: scene urbane, strade, piazze, portici, edifici, periferia\n"
    "- interni: ambienti chiusi, stanze, gallerie, bar, interni domestici\n"
    "- notturno: scene di notte, illuminazione artificiale, chiaro di luna\n"
    "- fiori: fiori come soggetto principale o molto evidente\n"
    "- natura morta: composizioni di oggetti, vasi, bottiglie, frutta, tavoli apparecchiati\n"
    "- figura: persona/e come soggetto principale (non sfondo)\n"
    "- marina: mare, costa, spiaggia, scogliera, lungomare\n\n"
    "Le etichette possono sovrapporsi: un dipinto può essere sia 'città' che 'notturno', "
    "o sia 'paesaggio' che 'fiori'.\n\n"
    "Rispondi con un oggetto JSON contenente una chiave \"tags\" con la lista delle etichette scelte."
)

MAX_CONCURRENT = 4


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


def load_titles(csv_path: Path) -> dict[str, str]:
    """Load painting id → title mapping from archive.csv."""
    if not csv_path.exists():
        print(f"Warning: CSV not found at {csv_path}, titles won't be included", file=sys.stderr)
        return {}

    mapping: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            painting_id = row.get("id", "").strip()
            title = row.get("title", "").strip()
            if painting_id:
                mapping[painting_id] = title
    return mapping


def encode_image(path: Path) -> str:
    """Load an image, convert to JPEG, and return as base64 string."""
    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


async def generate_tags(
    client: httpx.AsyncClient,
    config: dict,
    image_b64: str,
    title: str | None = None,
) -> list[str]:
    """Call the vision LLM to generate tags for a painting."""
    # Build user message: image + optional title hint
    user_content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        },
    ]
    if title:
        user_content.insert(0, {
            "type": "text",
            "text": f"Titolo del dipinto: \"{title}\"",
        })

    resp = await client.post(
        f"{config['base_url']}/chat/completions",
        json={
            "model": config["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "response_format": TAGS_SCHEMA,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(content)
    return [t for t in data["tags"] if t in VALID_TAGS]


async def process_painting(
    painting_id: str,
    thumb: Path,
    title: str | None,
    client: httpx.AsyncClient,
    config: dict,
    semaphore: asyncio.Semaphore,
    results: dict[str, list[str]],
    progress: tqdm,
) -> None:
    """Encode one thumbnail, call the LLM (rate-limited), and store results."""
    async with semaphore:
        try:
            image_b64 = encode_image(thumb)
            tags = await generate_tags(client, config, image_b64, title)
            results[painting_id] = tags
            label = f"\"{title}\"" if title else "(senza titolo)"
            progress.write(f"  {painting_id[:8]}… {label}: {', '.join(tags)}")
        except Exception as e:
            progress.write(f"  {painting_id[:8]}… ERROR - {e}")
        finally:
            progress.update()


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tags for paintings via vision LLM",
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help="Path to archive.csv (default: $ARTAG_CSV or ~/Downloads/artag/data/archive.csv)",
    )
    parser.add_argument(
        "--ids", nargs="*", default=[],
        help="Specific painting IDs to process",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip IDs already in output file",
    )
    parser.add_argument(
        "--concurrency", type=int, default=MAX_CONCURRENT,
        help=f"Max concurrent requests (default: {MAX_CONCURRENT})",
    )
    args = parser.parse_args()

    config = get_config()
    titles = load_titles(args.csv)
    print(f"Loaded {len(titles)} titles from {args.csv}")

    # Discover available thumbnails
    available: dict[str, Path] = {p.stem: p for p in THUMBS_DIR.glob("*.webp")}
    print(f"Found {len(available)} thumbnails in {THUMBS_DIR}")

    # Determine which IDs to process
    if args.ids:
        target_ids = args.ids
    elif titles:
        target_ids = list(titles.keys())
    else:
        target_ids = list(available.keys())

    # Filter to IDs that have thumbnails
    painting_ids = [pid for pid in target_ids if pid in available]
    skipped = len(target_ids) - len(painting_ids)
    if skipped:
        print(f"Skipped {skipped} IDs with no thumbnail")

    # Load existing results
    results: dict[str, list[str]] = {}
    if OUTPUT_FILE.exists():
        results = json.loads(OUTPUT_FILE.read_text())

    if args.skip_existing:
        before = len(painting_ids)
        painting_ids = [pid for pid in painting_ids if pid not in results]
        print(f"Skipping {before - len(painting_ids)} already-tagged paintings")

    if not painting_ids:
        sys.exit("Nothing to process.")

    print(f"Tagging {len(painting_ids)} paintings with {config['model']}")

    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:
        with tqdm(total=len(painting_ids), desc="Tags") as progress:
            tasks = [
                process_painting(
                    pid, available[pid], titles.get(pid),
                    client, config, semaphore, results, progress,
                )
                for pid in painting_ids
            ]
            await asyncio.gather(*tasks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
    )
    print(f"\nSaved {len(results)} results to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(async_main())
