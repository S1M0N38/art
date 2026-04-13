# Scripts

Utility scripts for image processing and metadata generation.
All scripts use `uv run` with inline dependencies — no virtualenv needed.

## Image Directory Structure

```
images/
├── front/
│   ├── original/        # Full-res JPGs (copied from arthag)
│   ├── thumbs/          # 800px WebP (generated)
│   └── placeholders/    # 20px WebP (generated)
└── back/
    ├── original/        # Full-res JPGs (copied from arthag)
    ├── thumbs/          # 800px WebP (generated)
    └── placeholders/    # 20px WebP (generated)
```

`public/images` is a symlink to `../images/` so Astro serves them.

## Image Processing Pipeline

```
arthag edit/front/  ──copy──►  images/front/original/
                                    ↓
                                optimize.py
                                    ↓
                              images/front/thumbs/       (800px WebP)
                              images/front/placeholders/  (20px WebP)

arthag edit/back/   ──copy──►  images/back/original/
                                    ↓
                                optimize.py
                                    ↓
                              images/back/thumbs/        (800px WebP)
                              images/back/placeholders/   (20px WebP)
```

### 1. Optimize — generate gallery variants

Creates thumbs and placeholders from the original images. Handles both front and back
in a single script.

```bash
uv run scripts/utils/optimize.py                     # front + back
uv run scripts/utils/optimize.py --front             # front only
uv run scripts/utils/optimize.py --back              # back only
uv run scripts/utils/optimize.py --skip-existing     # skip if variants exist
```

**Input:** `images/{front,back}/original/`
**Output:** `images/{front,back}/thumbs/` + `images/{front,back}/placeholders/`

| Variant          | Format | Max width | Quality | Use case          |
| ---------------- | ------ | --------- | ------- | ----------------- |
| `original/`      | JPG    | original  | —       | Lightbox, download |
| `thumbs/`        | WebP   | 800px     | 80      | Masonry grid       |
| `placeholders/`  | WebP   | 20px      | 10      | Blur-up effect     |

### 2. Upload — push to BunnyCDN

Uploads variants to BunnyCDN storage, preserving the directory structure.

Requires `.env` with `BUNNY_STORAGE_ZONE`, `BUNNY_STORAGE_PASSWORD`, `BUNNY_STORAGE_ENDPOINT`.

```bash
uv run scripts/utils/upload_cdn.py                          # upload all variants
uv run scripts/utils/upload_cdn.py --variant front/thumbs   # upload only front thumbnails
uv run scripts/utils/upload_cdn.py --skip-existing          # skip already uploaded
uv run scripts/utils/upload_cdn.py --dry-run                # preview without uploading
uv run scripts/utils/upload_cdn.py --list                   # list remote files
```

---

## Metadata Generation Pipeline

Generates `paintings.yaml` from the optimized images using a Vision Language Model (VLM)
and an aspect-ratio algorithm. Each generator outputs a JSON file; the final step merges
them into YAML.

```
images/front/original/  →  dimensions.py  →  dimensions.json ─┐
images/front/thumbs/    →  tags.py        →  tags.json ───────┤
images/front/thumbs/    →  titles.py      →  titles.json ─────┤
images/front/thumbs/    →  sort_ids.py    →  sort_ids.json ───┤
                                                                ↓
                                                          paintings.py
                                                                ↓
                                                    scripts/generate/paintings.yaml
                                                                ↓
                                                      (manual copy to src/content/)
```

VLM scripts require `.env` with `VLM_BASE_API` and `VLM_MODEL` (OpenAI-compatible endpoint).

### 1. Dimensions — map aspect ratios to standard canvas sizes

Reads pixel dimensions from originals, matches the aspect ratio to the closest standard
Italian canvas size (24 presets, 20–80 cm range). When multiple sizes share the same ratio,
picks one deterministically using a hash of the UUID — so repeated runs produce the same
result while creating a realistic size distribution.

```bash
uv run scripts/generate/dimensions.py                                        # all originals
uv run scripts/generate/dimensions.py images/front/original/abc.jpg          # single image
uv run scripts/generate/dimensions.py --skip-existing                        # resume
```

**Output:** `scripts/generate/dimensions.json` — `{uuid: {width_cm, height_cm}}`

### 2. Tags — classify paintings via VLM

Sends each thumbnail to a VLM with an Italian art expert prompt. Returns 2–4 semantic tags
per painting using structured JSON output.

Valid tags: `paesaggio`, `città`, `interni`, `astratto`, `ritratto`, `natura morta`, `notturno`, `fiori`.

```bash
uv run scripts/generate/tags.py                                              # all thumbnails
uv run scripts/generate/tags.py images/front/thumbs/abc.webp                 # single image
uv run scripts/generate/tags.py --skip-existing                              # resume
```

**Output:** `scripts/generate/tags.json` — `{uuid: [tag1, tag2, ...]}`

### 3. Titles — generate Italian titles via VLM

Sends each thumbnail to a VLM with an Italian art curator prompt. Returns a short, evocative
gallery-style title in Italian.

```bash
uv run scripts/generate/titles.py                                            # all thumbnails
uv run scripts/generate/titles.py images/front/thumbs/abc.webp               # single image
uv run scripts/generate/titles.py --skip-existing                            # resume
```

**Output:** `scripts/generate/titles.json` — `{uuid: "Titolo del Dipinto"}`

### 4. Sort IDs — compute visual ordering via CLIP + t-SNE

Generates a visually coherent display order by embedding each thumbnail with CLIP, reducing
to 1D with t-SNE, and assigning sequential `sort_id` values so similar paintings appear near
each other in the gallery.

```bash
uv run scripts/generate/sort_ids.py                                          # all thumbnails
uv run scripts/generate/sort_ids.py --skip-existing                          # resume
```

**Output:** `scripts/generate/sort_ids.json` — `{uuid: sort_id}`

### 5. Paintings — merge into YAML

Merges the four JSON files into a single `paintings.yaml`. Validates that all files contain
the same set of UUIDs before writing.

```bash
uv run scripts/generate/paintings.py
```

**Output:** `scripts/generate/paintings.yaml`

The generated file should be reviewed and then manually copied to `src/content/paintings.yaml`.

---

## Common Flags

All processing scripts support these patterns:

| Flag               | Description                                  |
| ------------------ | -------------------------------------------- |
| (no args)          | Process all files in the default input dir   |
| `path/to/file`     | Process a single file                        |
| `--skip-existing`  | Skip files that already have output          |
