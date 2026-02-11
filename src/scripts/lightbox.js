/**
 * lightbox.js — PhotoSwipe v5 integration.
 *
 * Opens a full-screen lightbox when any .painting-card is clicked.
 * Shows the original high-res image with a custom caption bar
 * (title, dimensions, tags, download button).
 */

import PhotoSwipeLightbox from 'photoswipe/lightbox';
import 'photoswipe/style.css';

// Build data source from all painting cards in DOM order
function buildDataSource() {
  const cards = document.querySelectorAll('.painting-card');
  return Array.from(cards).map((card) => {
    const widthCm = Number(card.dataset.width);
    const heightCm = Number(card.dataset.height);
    const aspect = widthCm / heightCm;

    // Estimate pixel dimensions from aspect ratio.
    // Originals vary but we use 2000px on the longest side
    // so PhotoSwipe has good initial dimensions for the opening animation.
    const maxDim = 2000;
    let w, h;
    if (aspect >= 1) {
      w = maxDim;
      h = Math.round(maxDim / aspect);
    } else {
      h = maxDim;
      w = Math.round(maxDim * aspect);
    }

    return {
      src: card.dataset.full,          // original JPG (loaded in background)
      msrc: card.dataset.thumb,        // 800px thumb (shown immediately as placeholder)
      width: w,
      height: h,
      // Custom data for caption
      title: card.dataset.title,
      widthCm: widthCm,
      heightCm: heightCm,
      tags: card.dataset.tags ? card.dataset.tags.split(',') : [],
      element: card, // for thumbnail animation
    };
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const dataSource = buildDataSource();
  const cards = document.querySelectorAll('.painting-card');

  // Map card elements to their index in the data source
  const cardIndexMap = new Map();
  cards.forEach((card, i) => {
    cardIndexMap.set(card, i);
  });

  const lightbox = new PhotoSwipeLightbox({
    dataSource: dataSource,
    pswpModule: () => import('photoswipe'),

    // Thumbnail animation: link to the card's img element
    showHideAnimationType: 'zoom',
    
    // Padding around the image
    paddingFn: () => ({
      top: 40,
      bottom: 80,
      left: 0,
      right: 0,
    }),

    // Zoom levels
    initialZoomLevel: 'fit',
    secondaryZoomLevel: 2,
    maxZoomLevel: 4,

    // Close on vertical drag
    closeOnVerticalDrag: true,

    // Background opacity
    bgOpacity: 0.95,
  });

  // --- Thumbnail element for zoom animation ---
  lightbox.addFilter('thumbEl', (_thumbEl, data) => {
    if (data.element) {
      const img = data.element.querySelector('img');
      if (img) return img;
    }
    return _thumbEl;
  });

  // --- Placeholder: always use the 800px thumb for instant display ---
  lightbox.addFilter('placeholderSrc', (_src, slide) => {
    return slide.data.msrc || _src;
  });

  // --- Crossfade: hide original until fully loaded, then fade in ---
  // Without this the browser progressively renders the original JPG
  // top-to-bottom on top of the placeholder, causing visible flicker.
  lightbox.on('contentLoadImage', ({ content }) => {
    const img = content.element;
    if (img && img.tagName === 'IMG') {
      // If the image is already cached the browser fires `complete`
      // synchronously — skip hiding so there's no flash.
      if (!img.complete) {
        img.style.opacity = '0';
      }
    }
  });

  lightbox.on('loadComplete', ({ content }) => {
    const img = content.element;
    if (img && img.tagName === 'IMG') {
      // Use rAF so the browser composites the placeholder first,
      // then transitions the original in on the next frame.
      requestAnimationFrame(() => {
        img.style.opacity = '1';
      });
    }
  });

  // --- Custom caption UI element ---
  lightbox.on('uiRegister', () => {
    lightbox.pswp.ui.registerElement({
      name: 'custom-caption',
      order: 9,
      isButton: false,
      appendTo: 'root',
      html: '',
      onInit: (el) => {
        el.classList.add('pswp__custom-caption');

        lightbox.pswp.on('change', () => {
          const slide = lightbox.pswp.currSlide;
          const data = slide.data;

          const tagsHtml = data.tags
            .map((t) => `<span class="pswp__caption-tag">${t}</span>`)
            .join('');

          el.innerHTML = `
            <div class="pswp__caption-title">${data.title}</div>
            <div class="pswp__caption-meta">
              <span class="pswp__caption-dims">${data.widthCm} × ${data.heightCm} cm</span>
              ${tagsHtml}
            </div>
          `;
        });
      },
    });

    // --- Download button ---
    lightbox.pswp.ui.registerElement({
      name: 'download-button',
      order: 8,
      isButton: true,
      tagName: 'a',
      html: {
        isCustomSVG: true,
        inner:
          '<path d="M20.5 14.3 17.1 18V10h-2.2v7.9l-3.4-3.6L10 16l6 6.1 6-6.1ZM23 23H9v2h14Z" id="pswp__icn-download"/>',
        outlineID: 'pswp__icn-download',
      },
      onInit: (el, pswp) => {
        el.setAttribute('download', '');
        el.setAttribute('target', '_blank');
        el.setAttribute('rel', 'noopener');
        el.setAttribute('title', 'Scarica originale');

        pswp.on('change', () => {
          el.href = pswp.currSlide.data.src;
        });
      },
    });
  });

  lightbox.init();

  // --- Open lightbox on card click ---
  cards.forEach((card) => {
    card.addEventListener('click', (e) => {
      e.preventDefault();
      const index = cardIndexMap.get(card);
      if (index !== undefined) {
        lightbox.loadAndOpen(index);
      }
    });
  });
});
