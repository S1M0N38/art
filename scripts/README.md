# Scripts

Utility scripts for image processing and metadata generation.
All scripts use `uv run` with inline dependencies — no virtualenv needed.

## Image Processing Pipeline

Raw photos go through three steps: crop, optimize, upload.

```
images/raw/  →  preprocess  →  images/processed/
                                    ↓
                                optimize
                                    ↓
                              images/optimized/
                              ├── originals/       (full-res JPG)
                              ├── thumbs/          (800px WebP)
                              └── placeholders/    (20px WebP)
                                    ↓
                                upload_cdn  →  BunnyCDN
```

### 1. Preprocess — crop white borders

Detects and removes white borders from raw photos using grayscale threshold analysis.

```bash
uv run scripts/utils/preprocess.py                          # all images in images/raw/
uv run scripts/utils/preprocess.py images/raw/photo.jpg     # single image
uv run scripts/utils/preprocess.py --threshold 230          # custom brightness threshold (default: 240)
```

**Input:** `images/raw/` (JPG, JPEG, PNG)
**Output:** `images/processed/` (same filenames, quality 95)

### 2. Optimize — generate CDN variants

Creates three size variants per image, using the filename (UUID) as the output name.

```bash
uv run scripts/utils/optimize.py                            # all from images/processed/
uv run scripts/utils/optimize.py images/processed/abc.jpg   # single image
uv run scripts/utils/optimize.py --skip-existing            # skip already optimized
```

**Input:** `images/processed/`
**Output:** `images/optimized/`

| Variant        | Format | Max width | Quality | Use case          |
| -------------- | ------ | --------- | ------- | ----------------- |
| `originals/`   | JPG    | original  | —       | Lightbox, download |
| `thumbs/`      | WebP   | 800px     | 80      | Masonry grid       |
| `placeholders/` | WebP  | 20px      | 10      | Blur-up effect     |

### 3. Upload — push to BunnyCDN

Uploads optimized variants to BunnyCDN storage, preserving the directory structure.

Requires `.env` with `BUNNY_STORAGE_ZONE`, `BUNNY_STORAGE_PASSWORD`, `BUNNY_STORAGE_ENDPOINT`.

```bash
uv run scripts/utils/upload_cdn.py                          # upload all variants
uv run scripts/utils/upload_cdn.py --variant thumbs         # upload only thumbnails
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
images/optimized/originals/  →  dimensions.py  →  dimensions.json ─┐
images/optimized/thumbs/     →  tags.py        →  tags.json ───────┤
images/optimized/thumbs/     →  titles.py      →  titles.json ─────┤
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
uv run scripts/generate/dimensions.py                                           # all originals
uv run scripts/generate/dimensions.py images/optimized/originals/abc.jpg        # single image
uv run scripts/generate/dimensions.py --skip-existing                           # resume
```

**Output:** `scripts/generate/dimensions.json` — `{uuid: {width_cm, height_cm}}`

### 2. Tags — classify paintings via VLM

Sends each thumbnail to a VLM with an Italian art expert prompt. Returns 2–4 semantic tags
per painting using structured JSON output.

Valid tags: `paesaggio`, `città`, `interni`, `astratto`, `ritratto`, `natura morta`, `notturno`, `fiori`.

```bash
uv run scripts/generate/tags.py                                                 # all thumbnails
uv run scripts/generate/tags.py images/optimized/thumbs/abc.webp                # single image
uv run scripts/generate/tags.py --skip-existing                                 # resume
```

**Output:** `scripts/generate/tags.json` — `{uuid: [tag1, tag2, ...]}`

### 3. Titles — generate Italian titles via VLM

Sends each thumbnail to a VLM with an Italian art curator prompt. Returns a short, evocative
gallery-style title in Italian.

```bash
uv run scripts/generate/titles.py                                               # all thumbnails
uv run scripts/generate/titles.py images/optimized/thumbs/abc.webp              # single image
uv run scripts/generate/titles.py --skip-existing                               # resume
```

**Output:** `scripts/generate/titles.json` — `{uuid: "Titolo del Dipinto"}`

### 4. Paintings — merge into YAML

Merges the three JSON files into a single `paintings.yaml`. Validates that all files contain
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
