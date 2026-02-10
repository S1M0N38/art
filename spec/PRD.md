# PRD: Francesco Luchino — Portfolio

## Document Status

| Field   | Value                      |
| ------- | -------------------------- |
| Author  | Simo                       |
| Version | 1.0                        |
| Date    | 2025-02-10                 |
| Status  | Draft                      |

---

## 1. Problem

Francesco Luchino is an Italian painter with a body of over 500 works (oil on canvas, styles ranging from figurative to abstract). There is currently no dedicated digital platform to explore his work. The goal is a portfolio site that puts the artwork front and center, with an immersive and fluid experience, free of distractions.

---

## 2. Goals

1. **Immersive showcase** — The artwork commands 100% of attention. The interface disappears.
2. **Dimensional fidelity** — Display sizes proportionally reflect the real physical dimensions of the paintings (a 200x150 cm painting appears larger than a 30x20 cm one).
3. **Performance** — Perceived load time < 2s on 3G. Lighthouse Performance > 90.
4. **Scalability** — Handle 500+ works without degrading the experience or bloating the repository.
5. **Simple management** — Adding a work = one YAML entry + uploading the image to the CDN.

---

## 3. Non-Goals

| Excluded                        | Rationale                                                    |
| ------------------------------- | ------------------------------------------------------------ |
| E-commerce / sales              | This is purely a showcase portfolio, not a shop.             |
| CMS / upload UI                 | The maintainer is comfortable with YAML + git.               |
| Database                        | Static site. Data lives in YAML files in the repository.     |
| Multilingual (IT/EN)            | Initial release is Italian only. English may come later.     |
| Full-text search                | Tag/period filtering is sufficient for navigation.           |
| Authentication / private areas  | All content is entirely public.                              |

---

## 4. User Stories

### Visitor

- **As a** visitor, **I want to** vertically scroll through a grid of artworks **so that** I can explore the entire collection naturally.
- **As a** visitor, **I want to** perceive the real proportions of the artworks in the grid **so that** I can appreciate the relative scale between paintings.
- **As a** visitor, **I want to** click on a work to see it in high definition **so that** I can observe its details.
- **As a** visitor, **I want to** see the title, dimensions, year, and tags in the detail view **so that** I have context about the work.
- **As a** visitor, **I want to** download the full-resolution image **so that** I can save or print it.
- **As a** visitor, **I want to** filter works by category and period **so that** I can find the paintings I'm interested in.
- **As a** visitor, **I want to** learn about the artist **so that** I understand the context of his work.

### Maintainer (Simo)

- **As a** maintainer, **I want to** add a work by editing a YAML file and uploading an image to the CDN **so that** I can keep the site updated without complex tools.
- **As a** maintainer, **I want** deployment to be automatic on push to GitHub **so that** I don't have to manage manual builds.

---

## 5. Requirements

### P0 — Must Have

#### 5.1 Main Gallery (Masonry Grid)

- Column-based masonry layout, vertical scroll, dark background (black / charcoal).
- Painting cards are sized **proportionally to the real physical dimensions** of the painting:
  - Real dimensions (e.g., `120x80 cm`) are read from metadata.
  - A normalized scale factor is calculated: the largest painting defines the maximum card size; others scale accordingly.
  - The image aspect ratio always matches the real painting's aspect ratio.
- **Lazy loading** with Intersection Observer — images load only as they enter the viewport.
- **Progressive blur-up**: ultra-low-resolution placeholder (20px, blurred via CSS) → web-quality image on load. The placeholder is served by the CDN (same image, `?width=20&quality=10`).
- **Zero gaps** — Tightly packed masonry, no whitespace between cards.
- On mobile: 1–2 column layout. On desktop: 3–5 adaptive columns.

#### 5.2 Detail View (Lightbox)

- Click/tap on a work opens a full-screen lightbox view.
- Displays the image in high quality (served by the CDN with screen-optimized dimensions).
- **Visible metadata**: title, physical dimensions, year, tags.
- **Download button**: downloads the original full-resolution file from the CDN.
- **Close**: returns to the exact scroll position in the gallery.
- **Pinch-to-zoom** on mobile.
- Navigation between works (arrows / swipe) without returning to the grid.

#### 5.3 Filters

- Fixed button in the bottom-left corner (floating action button).
- On click, opens a panel with selectable chips organized in two groups:
  - **Categories**: "Tutti" (All), "Paesaggio" (Landscape), "Città" (City), "Interni" (Interior), "Astratto" (Abstract), "Ritratto" (Portrait), "Natura morta" (Still life).
  - **Periods**: "Anni '70" (1970s), "Anni '80" (1980s), "Anni '90" (1990s), "Anni 2000" (2000s), "Anni 2010" (2010s).
- **Filter logic**: OR within the same group, AND across different groups.
  - Example: selecting "Paesaggio" + "Astratto" + "Anni '90" shows all landscapes and abstracts that are from the 1990s.
- Multi-selection within each group.
- "Tutti" (All) is selected by default and deselects all other filters.
- Filtering happens client-side with smooth transitions (animated layout reflow).

> **Note**: Filter labels are in Italian (the site's language). English translations above are for developer reference only.

#### 5.4 "L'artista" (About the Artist) Page

- Accessible via a discreet link in the upper corner (e.g., an "i" icon or "L'artista" text).
- Content: bio, artist statement, exhibition history, photo of the artist.
- Contact: email, social links.
- Design consistent with the site's dark theme.
- Content is placeholder for now; real text will be written later.

#### 5.5 Image Infrastructure

- Original high-resolution images are stored on an **external CDN** (BunnyCDN — account already active).
- The CDN provides on-the-fly resizing via URL parameters (e.g., `?width=800&quality=80`).
- The Git repository contains **only** code and YAML metadata — no images.
- Three variants of each image, generated on-the-fly by the CDN:
  - **Placeholder**: ~20px width, minimum quality (for blur-up).
  - **Thumbnail**: ~800px width, quality 80, WebP (for the masonry grid).
  - **Full**: original size (for lightbox and download).

#### 5.6 Static Deployment

- Static build, automatic deployment to **GitHub Pages** via **GitHub Actions**.
- Push to `main` → build → deploy.

### P1 — Nice to Have

- **Smooth transitions** between grid and lightbox (the image "expands" from its position in the grid).
- **Smart preloading**: preload adjacent images in the lightbox for instant navigation.
- **Hash URLs** to share a specific work (e.g., `#painting-42`).
- **Work counter** visible in the gallery ("342 opere" or similar, discreet).
- **Favicon and Open Graph meta tags** for social sharing.

### P2 — Future Considerations

- Multilingual support (Italian/English).
- Custom domain `francescoluchino.com`.
- "Exhibitions" section with photos and dates.
- Print-on-demand or gallery links for sales.

---

## 6. Architecture and Tech Stack

### 6.1 Alternatives Evaluated

| Aspect              | Recommended              | Alternatives Evaluated                  | Rationale                                                                                            |
| ------------------- | ------------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Site generator**  | **Astro**                | Hugo, 11ty, Next.js (static export)    | Astro has an "islands" architecture: zero JS by default, adds interactivity only where needed. Great for content-driven static sites. Native support for data collections. Hugo is fastest but Go templating is less ergonomic; 11ty is solid but Astro's ecosystem is more modern. |
| **Masonry layout**  | **CSS columns + JS**     | Masonry.js, Isotope.js, CSS Grid       | CSS `columns` provides native masonry with zero dependencies. A small JS script handles proportional sizing and reflow on filtering. Isotope.js is robust but heavy (~30KB) for a site aiming for lightness. |
| **Lightbox**        | **PhotoSwipe v5**        | GLightbox, Fancybox, custom            | PhotoSwipe is the standard for photography/art sites: zero dependencies, native pinch-to-zoom, flexible API for custom metadata panels, smooth thumbnail-to-full transitions. ~15KB gzipped. |
| **Image CDN**       | **BunnyCDN**             | Cloudflare R2 + Images, imgix, Sirv    | BunnyCDN: storage ~$0.01/GB/month, bandwidth ~$0.01/GB, Bunny Optimizer for on-the-fly resize and WebP/AVIF conversion. Simple setup. Estimated cost for 500 images (~1GB storage + moderate traffic): < $2/month. |
| **Styling**         | **Vanilla CSS (custom)** | Tailwind CSS, Sass                     | For a minimalist site with few components, custom CSS with variables is sufficient and adds no build steps. If complexity grows, Tailwind remains an option. |
| **Deployment**      | **GitHub Pages**         | Netlify, Vercel, Cloudflare Pages      | Free, already integrated with GitHub. GitHub Actions for automatic Astro build on push. |

### 6.2 Recommended Final Stack

```
Astro (static site generator)
├── Data: YAML collections (paintings.yaml)
├── Layout: CSS columns + vanilla JS for proportional sizing
├── Lightbox: PhotoSwipe v5
├── Filters: Vanilla JS (client-side filtering + CSS transitions)
├── Blur-up: Intersection Observer + CSS filter:blur
├── Styling: Custom CSS with variables (dark theme)
│
├── Images: BunnyCDN (storage + on-the-fly optimization)
├── Hosting: GitHub Pages
└── CI/CD: GitHub Actions
```

### 6.3 Project Structure

```
art/
├── src/
│   ├── layouts/
│   │   └── Base.astro              # Base layout (head, meta, dark theme)
│   ├── pages/
│   │   ├── index.astro             # Main masonry gallery
│   │   └── artista.astro           # "About the Artist" page
│   ├── components/
│   │   ├── Gallery.astro           # Masonry grid
│   │   ├── PaintingCard.astro      # Single painting card
│   │   ├── Lightbox.astro          # PhotoSwipe wrapper
│   │   ├── FilterPanel.astro       # Filter panel with chips
│   │   └── Header.astro            # Minimal nav (name + artist link)
│   ├── scripts/
│   │   ├── masonry.js              # Proportional sizing + layout reflow
│   │   ├── lightbox.js             # PhotoSwipe init + metadata panel
│   │   └── filters.js              # Client-side filtering logic
│   └── styles/
│       └── global.css              # Dark theme, variables, typography
├── src/content/
│   └── paintings.yaml              # Metadata for all works
├── public/
│   └── favicon.svg
├── images/                          # Local images (gitignored)
│   ├── raw/                         # Raw originals before processing
│   └── processed/                   # Cropped images ready for CDN upload
├── scripts/
│   └── utils/
│       └── preprocess.py            # Auto-crop white borders (uv run)
├── spec/                            # Project specifications
│   ├── PRD.md                       # Product requirements document
│   └── example-images/              # Example images for documentation
├── astro.config.mjs
├── package.json
└── .github/
    └── workflows/
        └── deploy.yml              # Astro build + GitHub Pages deploy
```

### 6.4 Data Schema (`paintings.yaml`)

```yaml
- id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  file: "a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg"  # UUID filename on BunnyCDN
  title: "Tramonto su Fossano"
  width_cm: 120                      # Real painting width in cm
  height_cm: 80                      # Real painting height in cm
  year: 1995
  tags:
    - paesaggio
```

**File naming convention**: Each painting uses a UUID v4 as both its `id` and CDN filename (e.g., `a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg`). The title lives only in YAML metadata.

The `file` field corresponds to the path on the CDN base URL (e.g., `https://luchino.b-cdn.net/a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg`). Size variants are generated on-the-fly:

- Placeholder: `?width=20&quality=10`
- Thumbnail: `?width=800&quality=80`
- Full: direct URL without parameters

---

## 7. Proportional Sizing — Algorithm

The site's distinctive feature. Here's the logic:

1. **Input**: each work has `width_cm` and `height_cm`.
2. **Relative area calculation**: `area = width_cm * height_cm`. Normalize against the maximum area across all works.
3. **Grid scaling**: the card width in the masonry is proportional to `sqrt(normalized_area)` — the square root prevents large works from dominating excessively.
4. **Aspect ratio**: the image always maintains the `width_cm / height_cm` ratio.
5. **Bounds**: a minimum width (e.g., 150px) and maximum width (e.g., masonry column width) are defined to ensure usability.

Example:
- Painting A: 200x150 cm → area 30,000 → scale 1.0 (maximum)
- Painting B: 50x40 cm → area 2,000 → scale ~0.26 → visibly smaller card

---

## 8. Performance

| Technique                            | Impact                                                |
| ------------------------------------ | ----------------------------------------------------- |
| Zero JS by default (Astro)           | Only necessary JS is loaded (islands)                 |
| Lazy loading (Intersection Observer) | Images loaded only in viewport + buffer               |
| Blur-up placeholder                  | Perceived instant load, placeholder < 1KB             |
| WebP/AVIF via CDN                    | -40/60% image weight compared to JPEG                 |
| CSS columns (no layout library)      | Zero JS cost for base masonry layout                  |
| Static HTML                          | Minimal TTFB, 100% cacheable                          |

**Targets**: Lighthouse Performance > 90, LCP < 2.5s, CLS < 0.1.

---

## 9. Success Metrics

| Metric                           | Target              | Type    |
| -------------------------------- | ------------------- | ------- |
| Lighthouse Performance           | > 90                | Leading |
| LCP (Largest Contentful Paint)   | < 2.5s              | Leading |
| Average time on page             | > 2 minutes         | Lagging |
| Works viewed per session         | > 20 (scroll depth) | Lagging |
| Bounce rate                      | < 40%               | Lagging |

---

## 10. Resolved Questions

| #  | Question                   | Answer                                                                                  |
| -- | -------------------------- | --------------------------------------------------------------------------------------- |
| 1  | **Filter logic**           | OR within the same group, AND across different groups.                                  |
| 2  | **Filter categories**      | Paesaggio, Città, Interni, Astratto, Ritratto, Natura morta.                            |
| 3  | **Periods**                | 1970s, 1980s, 1990s, 2000s, 2010s.                                                     |
| 4  | **CDN file names**         | UUID v4 as filename: `a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg`.                          |
| 5  | **"L'artista" content**    | Will be written later. Placeholder page for now.                                        |
| 6  | **BunnyCDN account**       | Active. Storage zone `francescoluchino`, Frankfurt (DE) endpoint.                      |

---

## 11. Timeline and Phases

### Phase 1 — MVP (First Release)
- Masonry gallery with proportional sizing
- Lightbox with metadata and download
- Blur-up lazy loading
- Dark theme
- Deploy to GitHub Pages
- BunnyCDN setup + initial batch upload
- Data population: at least 50 works to validate layout and performance

### Phase 2 — Completion
- Filters by category and period
- "L'artista" (About) page
- Lightbox navigation (prev/next)
- Shareable URLs for individual works
- Open Graph meta tags
- Full population: 500+ works

### Phase 3 — Evolution
- Custom domain
- Multilingual IT/EN
- Advanced performance optimizations
- Potential new sections (exhibitions, press)
