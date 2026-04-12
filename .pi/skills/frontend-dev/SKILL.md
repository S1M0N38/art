---
name: frontend-dev
description: "Frontend development skill for the Francesco Luchino art portfolio. Use when building Astro components, CSS styles, vanilla JS, masonry layout, lightbox, or filters. Knows project conventions: Astro islands, CSS columns, PhotoSwipe v5, dark theme, progressive blur-up, Italian UI labels."
---

# Frontend Development — Art Portfolio

## Tech Stack
- **Astro** — static site, zero JS by default, islands architecture
- **CSS columns + vanilla JS** — masonry layout with proportional sizing
- **PhotoSwipe v5** — lightbox (pinch-to-zoom, swipe navigation)
- **Vanilla CSS** — custom properties, dark theme, no frameworks
- **Vanilla JS** — client-side filtering, Intersection Observer lazy loading

## Project Structure
See `AGENTS.md` for full structure. Key files:
- `src/components/` — Astro components (Gallery, PaintingCard, Lightbox, FilterPanel, Header)
- `src/scripts/` — Client-side JS (masonry.js, lightbox.js, filters.js)
- `src/styles/global.css` — Dark theme, variables, typography
- `src/pages/` — index.astro (gallery), artista.astro (about), critica.astro (reviews)

## Conventions
- Keep JS minimal — Astro islands only where interactivity is needed
- No images in the git repo — only code and YAML metadata
- Italian language for all UI labels
- Dark theme: black/charcoal background, artwork commands attention
- Proportional sizing: card size ∝ sqrt(normalized_area) of real painting dimensions
- Progressive blur-up: 20px placeholder → full image on load
- Filter logic: OR across categories

## When Making Components
1. Check existing patterns in `src/components/` and `src/styles/global.css`
2. Use Astro component syntax (`.astro` files)
3. Add CSS to `global.css` using existing custom properties
4. For interactivity, add vanilla JS to `src/scripts/` and import via `<script>` tag
5. Use CDN URLs for images: `https://francescoluchino-art.b-cdn.net/`

## Testing
- Run `npm run dev` to start dev server (port 1234)
- Use Playwright MCP for browser testing (different port than 1234)
- Check responsive behavior at mobile/tablet/desktop widths
- Verify dark theme contrast and image loading

## Performance Targets
- Lighthouse Performance > 90
- LCP < 2.5s, CLS < 0.1
- Lazy loading with Intersection Observer
