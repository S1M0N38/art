---
name: lighthouse-auditor
description: Run Lighthouse performance audits against the art portfolio site. Use when checking LCP, CLS, Lighthouse scores, or verifying performance targets (Performance > 90, LCP < 2.5s, CLS < 0.1).
---

# Lighthouse Auditor

## Targets
- Performance > 90
- LCP < 2.5s
- CLS < 0.1

## Usage

### Quick audit (CLI)
```bash
npx lighthouse http://localhost:4321 --output=json --chrome-flags="--headless" | npx lighthouse-ci
```

### With Playwright MCP
1. Start dev server on port 4321: `npm run dev -- --port 4321`
2. Use Playwright MCP to navigate and take screenshots
3. Check page load timing via browser performance APIs

## Checklist
- [ ] Images use lazy loading (Intersection Observer)
- [ ] Placeholders use blur-up pattern (20px WebP → full)
- [ ] No layout shift on image load (CLS)
- [ ] CSS is minimal, no unused frameworks
- [ ] JS loaded only where needed (Astro islands)
- [ ] WebP thumbnails from CDN (800px)
- [ ] No render-blocking resources
