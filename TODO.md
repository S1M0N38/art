# Migration: arthag CSV → art gallery (local images) ✅ COMPLETE

## Summary
- Replaced 441 placeholder paintings with 518 real ones from arthag `archive.csv`
- Images served locally from `public/images/` (no CDN)
- Pipelines adapted: `paintings.py` reads CSV as source of truth

## What was done

### Phase 1: Data pipeline ✅
- Rewrote `scripts/generate/paintings.py` to read `archive.csv`
- Added `year`, `technique` fields to YAML schema
- Handles empty year (→ null) and empty title (→ "Senza titolo")
- Merges `sort_ids.json`, `tags.json`, `titles.json` when present
- Cleared old pipeline JSON files (old placeholder UUIDs)
- Generated `paintings.yaml` with 518 entries

### Phase 2: Image pipeline ✅
- Rewrote `scripts/utils/optimize.py` to copy from arthag `edit/front/` via CSV mapping
- Generates originals (JPG), thumbs (800px WebP), placeholders (20px WebP)
- Added `public/images/` to `.gitignore`
- Generated all 518 image variants: 2.7GB originals, 26MB thumbs, 2MB placeholders

### Phase 3: Component updates ✅
- `PaintingCard.astro` — local `/images/` paths with Astro base URL prefix
- `Gallery.astro` — new schema fields (year, technique)
- `lightbox.js` — caption shows year, technique, tags
- `global.css` — styles for caption meta elements
- `index.astro` — updated TypeScript types

### Phase 4: Verify ✅
- `npm run build` — clean, no errors
- Playwright: 518 paintings rendering with real titles, zero console errors
- Lightbox opens correctly with real images
- `AGENTS.md` updated to reflect new workflow

## Future work
- Run `sort_ids.py` to compute visual similarity ordering (replaces custom_id ordering)
- Run `tags.py` to generate category tags
- CDN migration: upload images to BunnyCDN, switch `PaintingCard.astro` back to CDN URLs
- Add artist photo (`francesco-luchino.webp`) to `public/images/`
