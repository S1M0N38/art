/**
 * masonry.js — Masonry layout + Intersection Observer lazy loading.
 *
 * Uses Desandro's masonry-layout for positioning and imagesloaded
 * to re-layout as images load. Lazy loading with blur-up is handled
 * via Intersection Observer.
 *
 * Mobile (≤640px): single full-width column.
 * Desktop (>640px): multi-column with 340px column width.
 */

import Masonry from 'masonry-layout';
import imagesLoaded from 'imagesloaded';

(function () {
  'use strict';

  const DESKTOP_COLUMN_WIDTH = 340;
  const GUTTER = 16; // matches --masonry-gap
  const ROOTMARGIN = '200px 0px';
  const MOBILE_BREAKPOINT = 640;

  const gallery = document.querySelector('.gallery');
  if (!gallery) return;

  function isMobile() {
    return window.innerWidth <= MOBILE_BREAKPOINT;
  }

  function getColumnWidth() {
    if (isMobile()) {
      // Account for gallery left+right padding so card fits within bounds
      const style = getComputedStyle(gallery);
      const paddingH = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
      return gallery.offsetWidth - paddingH;
    }
    return DESKTOP_COLUMN_WIDTH;
  }

  function getGutter() {
    return isMobile() ? 8 : GUTTER;
  }

  // --- Initialize Masonry ---
  let msnry = new Masonry(gallery, {
    itemSelector: '.painting-card',
    columnWidth: getColumnWidth(),
    gutter: getGutter(),
    fitWidth: true,
    transitionDuration: '0.3s',
    initLayout: true,
  });

  // --- imagesLoaded: re-layout as images load ---
  imagesLoaded(gallery).on('progress', function () {
    msnry.layout();
  });

  // --- Intersection Observer lazy loading (blur-up) ---
  function loadImage(img) {
    const src = img.dataset.src;
    if (!src) return;

    const loader = new Image();
    loader.onload = function () {
      img.src = src;
      img.classList.remove('painting-card__img--placeholder');
      img.classList.add('painting-card__img--loaded');
    };
    loader.src = src;
    delete img.dataset.src;
  }

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      function (entries) {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            loadImage(entry.target);
            observer.unobserve(entry.target);
          }
        }
      },
      { rootMargin: ROOTMARGIN }
    );
    document.querySelectorAll('.painting-card__img[data-src]').forEach((img) => {
      observer.observe(img);
    });
  } else {
    document.querySelectorAll('.painting-card__img[data-src]').forEach(loadImage);
  }

  // --- Responsive: rebuild masonry on breakpoint change ---
  let lastWasMobile = isMobile();

  function handleResize() {
    const nowMobile = isMobile();
    if (nowMobile !== lastWasMobile) {
      lastWasMobile = nowMobile;
      msnry.destroy();
      msnry = new Masonry(gallery, {
        itemSelector: '.painting-card',
        columnWidth: getColumnWidth(),
        gutter: getGutter(),
        fitWidth: true,
        transitionDuration: '0.3s',
        initLayout: true,
      });
    }
  }

  // Debounced resize listener
  let resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(handleResize, 200);
  });

  // Expose instance for filter integration later
  window.__msnry = msnry;
})();
