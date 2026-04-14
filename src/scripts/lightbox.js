/**
 * lightbox.js — PhotoSwipe v5 integration.
 *
 * Opens a full-screen lightbox when any .painting-card is clicked.
 * Shows the original high-res image with a custom caption bar
 * (title, dimensions, tags, download button).
 *
 * Supports toggling between front and back of painting.
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

    const frontSrc = card.dataset.full;
    const frontMsrc = card.dataset.thumb;
    const hasBack = card.dataset.hasBack === 'true';

    return {
      // These are the "live" src/msrc that PhotoSwipe reads
      src: frontSrc,
      msrc: frontMsrc,
      width: w,
      height: h,
      // Custom data for caption
      title: card.dataset.title,
      year: card.dataset.year || null,
      widthCm: widthCm,
      heightCm: heightCm,
      technique: card.dataset.technique || '',
      status: card.dataset.status || '0',
      tags: card.dataset.tags ? card.dataset.tags.split(',') : [],
      element: card,
      // Front/back state
      hasBack,
      frontSrc,
      frontMsrc,
      backSrc: hasBack ? card.dataset.backFull : '',
      backMsrc: hasBack ? card.dataset.backThumb : '',
      currentSide: 'front',
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

    // Apply custom class for light theme overrides
    mainClass: 'pswp--light-bg',

    // Thumbnail animation: link to the card's img element
    showHideAnimationType: 'zoom',
    
    // Padding around the image (responsive)
    paddingFn: () => {
      const isNarrow = window.innerWidth <= 1024;
      return isNarrow
        ? { top: 8, bottom: 110, left: 8, right: 8 }
        : { top: 40, bottom: 100, left: 40, right: 40 };
    },

    // Zoom levels
    initialZoomLevel: (zoomLevelObject) => zoomLevelObject.fit * 0.85,
    secondaryZoomLevel: 2,
    maxZoomLevel: 4,

    // Close on vertical drag
    closeOnVerticalDrag: true,

    // Background opacity (white bg)
    bgOpacity: 1,
  });

  // --- Thumbnail element for zoom animation ---
  lightbox.addFilter('thumbEl', (_thumbEl, data) => {
    if (data.element) {
      const img = data.element.querySelector('img');
      if (img) return img;
    }
    return _thumbEl;
  });

  // --- Placeholder: always use the msrc for instant display ---
  lightbox.addFilter('placeholderSrc', (_src, slide) => {
    return slide.data.msrc || _src;
  });

  // --- Crossfade: hide image until fully loaded, then fade in ---
  lightbox.on('contentLoadImage', ({ content }) => {
    const img = content.element;
    if (img && img.tagName === 'IMG') {
      if (!img.complete) {
        img.style.transition = 'opacity 0.25s ease';
        img.style.opacity = '0';
      }
    }
  });

  lightbox.on('loadComplete', ({ content }) => {
    const img = content.element;
    if (img && img.tagName === 'IMG') {
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

          const tagsHtml = data.tags.length > 0
            ? data.tags
                .map((t) => `<span class="pswp__caption-tag">${t}</span>`)
                .join('')
            : '';

          const yearStr = data.year ? `<span class="pswp__caption-year">${data.year}</span>` : '';
          const techStr = data.technique ? `<span class="pswp__caption-technique">${data.technique}</span>` : '';

          // Status badge
          const statusMap = {
            '0': { label: 'Sconosciuto', cls: 'status-unknown' },
            '1': { label: 'Critico', cls: 'status-critical' },
            '2': { label: 'Scarso', cls: 'status-poor' },
            '3': { label: 'Buono', cls: 'status-good' },
            '4': { label: 'Ottimo', cls: 'status-excellent' },
          };
          const statusInfo = statusMap[data.status] || statusMap['0'];
          const statusHtml = `<span class="pswp__caption-status ${statusInfo.cls}">${statusInfo.label}</span>`;

          el.innerHTML = `
            <div class="pswp__caption-row">
              <div class="pswp__caption-title">${data.title}</div>
              <div class="pswp__caption-meta">${yearStr}${techStr}</div>
              <div class="pswp__caption-tags">${statusHtml}${tagsHtml}</div>
              <div class="pswp__caption-dims">${data.widthCm} × ${data.heightCm} cm</div>
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
          const data = pswp.currSlide.data;
          el.href = data.currentSide === 'back' ? data.backSrc : data.frontSrc;
        });
      },
    });

    // --- Front/Back toggle button ---
    lightbox.pswp.ui.registerElement({
      name: 'flip-button',
      order: 7,
      isButton: true,
      tagName: 'button',
      html: {
        isCustomSVG: true,
        inner:
          '<path d="m4.4 12.8 3.6-3.6 3.6 3.6" id="pswp__icn-flip" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>' +
          '<path d="M17.6 23.6H10.4a2.4 2.4 0 0 1-2.4-2.4V9.2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>' +
          '<path d="m28.4 20-3.6 3.6-3.6-3.6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>' +
          '<path d="M15.2 9.2h7.2a2.4 2.4 0 0 1 2.4 2.4v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>',
        outlineID: 'pswp__icn-flip',
      },
      onInit: (el, pswp) => {
        el.classList.add('pswp__button--flip');
        el.setAttribute('title', 'Vedi retro');

        // Flag to prevent change handler from resetting during a flip
        let isFlipping = false;

        function updateButton(data) {
          if (!data.hasBack) {
            el.style.display = 'none';
          } else {
            el.style.display = '';
            if (data.currentSide === 'back') {
              el.classList.add('pswp__button--flip-active');
              el.setAttribute('title', 'Vedi fronte');
            } else {
              el.classList.remove('pswp__button--flip-active');
              el.setAttribute('title', 'Vedi retro');
            }
          }
        }

        el.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          const slide = pswp.currSlide;
          if (!slide || !slide.data.hasBack) return;

          const data = slide.data;

          // Toggle side
          data.currentSide = data.currentSide === 'front' ? 'back' : 'front';

          const newSrc = data.currentSide === 'back' ? data.backSrc : data.frontSrc;
          const newMsrc = data.currentSide === 'back' ? data.backMsrc : data.frontMsrc;

          // Replace the entire dataSource item with updated src/msrc.
          // PhotoSwipe's refreshSlideContent re-initializes data from dataSource,
          // so we must replace the full object (keeping all custom properties).
          if (pswp.options && pswp.options.dataSource) {
            pswp.options.dataSource[slide.index] = {
              ...data,
              src: newSrc,
              msrc: newMsrc,
            };
          }

          // Use PhotoSwipe's built-in method to reload the slide content.
          // This properly destroys old content, creates new content,
          // rebinds zoom/pan gestures, and triggers the loading lifecycle.
          // Note: refreshSlideContent dispatches 'change' internally,
          // so we set isFlipping to prevent the change handler from resetting.
          isFlipping = true;
          pswp.refreshSlideContent(slide.index);
          isFlipping = false;

          // Update download button href
          const downloadBtn = el.parentElement?.querySelector('a[download]');
          if (downloadBtn) {
            downloadBtn.href = newSrc;
          }

          // Read fresh data (refreshSlideContent creates a new data object)
          const freshData = pswp.currSlide?.data;
          if (freshData) updateButton(freshData);
        });

        // Reset to front and update button on slide change
        // (only for actual navigation, not during a flip)
        pswp.on('change', () => {
          const data = pswp.currSlide.data;
          if (!isFlipping && data.currentSide === 'back') {
            if (pswp.options && pswp.options.dataSource) {
              pswp.options.dataSource[pswp.currSlide.index] = {
                ...data,
                currentSide: 'front',
                src: data.frontSrc,
                msrc: data.frontMsrc,
              };
            }
            pswp.refreshSlideContent(pswp.currSlide.index);
          }
          const freshData = pswp.currSlide?.data || data;
          updateButton(freshData);
        });

        // Initial state for first slide
        if (pswp.currSlide) {
          updateButton(pswp.currSlide.data);
        }
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
