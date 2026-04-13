# Francesco Luchino — Portfolio

Artist portfolio site for Italian painter Francesco Luchino (~500 oil paintings).
Artwork-first, immersive experience with proportional sizing that reflects real physical dimensions.

## Tech Stack

- **Astro** — static site generator (zero JS by default, islands architecture)
- **CSS columns + vanilla JS** — masonry layout with proportional sizing
- **PhotoSwipe v5** — lightbox (pinch-to-zoom, swipe navigation)
- **Vanilla CSS** — custom properties, dark theme, no frameworks
- **Vanilla JS** — client-side filtering, Intersection Observer lazy loading
- **Local images** — served from `public/images/`, generated from arthag edit/front via optimize.py
- **GitHub Pages** — hosting via GitHub Actions

## Project Structure

```
art/
├── src/
│   ├── layouts/Base.astro           # Base layout (head, meta, dark theme)
│   ├── pages/
│   │   ├── index.astro              # Main masonry gallery
│   │   └── artista.astro            # About the artist page
│   ├── components/
│   │   ├── Gallery.astro            # Masonry grid
│   │   ├── PaintingCard.astro       # Single painting card
│   │   ├── Lightbox.astro           # PhotoSwipe wrapper
│   │   ├── FilterPanel.astro        # Filter panel with chips
│   │   └── Header.astro             # Minimal nav
│   ├── scripts/
│   │   ├── masonry.js               # Proportional sizing + layout reflow
│   │   ├── lightbox.js              # PhotoSwipe init + metadata panel
│   │   └── filters.js               # Client-side filtering logic
│   └── styles/global.css            # Dark theme, variables, typography
├── src/content/paintings.yaml       # All painting metadata
├── images/                           # All image variants (gitignored)
│   ├── front/
│   │   ├── original/                # Full-res JPGs (copied from arthag)
│   │   ├── thumbs/                  # 800px WebP thumbnails (generated)
│   │   └── placeholders/            # 20px WebP placeholders (generated)
│   └── back/
│       ├── original/                # Full-res JPGs (copied from arthag)
│       ├── thumbs/                  # 800px WebP thumbnails (generated)
│       └── placeholders/            # 20px WebP placeholders (generated)
├── scripts/utils/optimize.py        # Generate thumbs + placeholders (uv run)
├── scripts/utils/upload_cdn.py      # Upload images to BunnyCDN (uv run)
└── spec/PRD.md                      # Full product requirements
```

## Data Schema (paintings.yaml)

```yaml
- id: "7b0411a2-351e-47bf-9e10-320be6dde816"
  sort_id: 2000             # Display order in gallery (from custom_id; recomputed by sort_ids.py)
  title: "Tetti Rossi"
  year: 2007
  width_cm: 50              # Real painting width in cm
  height_cm: 35             # Real painting height in cm
  technique: "olio su tavola"
  tags: []                  # Empty until tags.py pipeline fills them
  back_photo: "2c031cc9-....jpg"  # Source back photo filename from archive.csv (or null)
```

**Source of truth**: `artag/data/archive.csv` — the arthag project manages all painting metadata.

Image variants (local, in `public/images/` → symlink to `images/`, gitignored):
- Front original: `public/images/front/original/{uuid}.jpg` (lightbox + download)
- Front thumbnail: `public/images/front/thumbs/{uuid}.webp` (masonry grid, 800px)
- Front placeholder: `public/images/front/placeholders/{uuid}.webp` (blur-up, 20px)
- Back original: `public/images/back/original/{uuid}.jpg` (lightbox toggle + download)
- Back thumbnail: `public/images/back/thumbs/{uuid}.webp` (lightbox toggle, 800px)
- Back placeholder: `public/images/back/placeholders/{uuid}.webp` (blur-up, 20px)

`public/images` is a symlink to `../images/`. Images are served locally via Astro's `public/` directory. CDN migration planned for later.

## Key Design Decisions

- **Proportional sizing**: card size ∝ `sqrt(normalized_area)` of real painting dimensions. Largest painting = max card size; others scale relative to it.
- **Dark theme**: black/charcoal background so artwork commands attention.
- **Zero gaps masonry**: tightly packed, no whitespace between cards.
- **Progressive blur-up**: 20px placeholder with CSS blur → full image on load.
- **Filter logic**: OR — selecting multiple categories shows works matching any of them. Categories: Paesaggio, Città, Interni, Astratto, Ritratto, Natura morta.
- **Italian language**: all UI labels in Italian. No i18n for now.
- **No database**: all data in YAML, site is fully static.

## Commands

```bash
# Development
npm run dev              # Astro dev server
npm run build            # Production build
npm run preview          # Preview production build

# Data pipeline
uv run scripts/generate/paintings.py                  # Regenerate paintings.yaml from archive.csv
uv run scripts/generate/paintings.py --csv /path/to/archive.csv  # Custom CSV path

# Image optimization — front + back (from images/{front,back}/original/)
uv run scripts/utils/optimize.py                      # Generate all variants (front + back)
uv run scripts/utils/optimize.py --front              # Front only
uv run scripts/utils/optimize.py --back               # Back only
uv run scripts/utils/optimize.py --skip-existing      # Skip already optimized

# ML pipelines (optional, generate JSON overrides)
uv run scripts/generate/sort_ids.py                   # Compute visual similarity sort order
uv run scripts/generate/tags.py                       # Generate category tags via VLM
```

## Performance Targets

- Lighthouse Performance > 90
- LCP < 2.5s, CLS < 0.1
- Lazy loading with Intersection Observer
- No layout libraries — CSS columns only
- Use the **lighthouse-auditor** subagent to audit performance against these targets

## Commit Conventions

Conventional commits. Title must be **< 72 characters**.

Format: `type(scope): description`

### Types

`feat`, `fix`, `refactor`, `style`, `perf`, `docs`, `chore`, `ci`, `test`

### Scopes

| Scope        | Area                                                  |
| ------------ | ----------------------------------------------------- |
| `gallery`    | Masonry grid, Gallery.astro, PaintingCard.astro       |
| `lightbox`   | PhotoSwipe, Lightbox.astro, lightbox.js               |
| `filters`    | FilterPanel.astro, filters.js, chip logic             |
| `artist`     | "L'artista" about page (artista.astro)                |
| `layout`     | Base.astro, HTML structure, meta tags                 |
| `nav`        | Header.astro, navigation                              |
| `styles`     | global.css, theme, typography, variables              |
| `data`       | paintings.yaml, data schema changes                   |
| `images`     | Image preprocessing, CDN config, upload scripts       |
| `scripts`    | Build/utility scripts (scripts/ directory)            |
| `ci`         | GitHub Actions, deploy.yml                            |
| `config`     | astro.config.mjs, package.json, tooling               |
| `spec`       | PRD.md, project documentation                         |

Scope is optional for cross-cutting changes. Examples:

```
feat(gallery): add proportional sizing based on real cm
fix(lightbox): restore scroll position on close
style(styles): adjust dark theme contrast values
chore(data): add 20 new paintings to YAML
perf(gallery): defer offscreen images with IO
ci: add GitHub Pages deploy workflow
docs(spec): clarify filter logic in PRD
```

## Context7 Libraries

Reference docs available via Context7 `resolve-library-id` / `query-docs`:

- **Astro** (`/withastro/docs`): Official Astro framework documentation — components, layouts, pages, content collections, islands architecture.
- **PhotoSwipe** (`/dimsemenov/photoswipe`): PhotoSwipe v5 lightbox — initialization, options, events, custom UI, responsive images.
- **bunny.net Developer Hub** (`/websites/bunny_net`): Product docs, guides, and troubleshooting for bunny.net CDN, storage, and image optimization.
- **bunny.net API** (`/websites/bunny_net_reference`): API reference for managing CDN pull zones, storage zones, purge cache, and edge rules programmatically.
- **GitHub Actions** (`/websites/github_en_actions`): Workflow syntax, action authoring, CI/CD pipelines, deployment to GitHub Pages.
- **MDN Web Docs** (`/mdn/content`): Vanilla JS, CSS, Web APIs (Intersection Observer, Fetch, DOM), HTML elements and attributes.

## Conventions

- Keep JS minimal — Astro islands only where interactivity is needed
- No images in the git repo — only code and YAML metadata
- Scripts use `uv run` (inline script dependencies, no virtualenv needed)
- Adding a painting = one YAML entry + upload image to BunnyCDN
- **Playwright MCP**: always use a port other than 1234 (reserved for manual testing)
